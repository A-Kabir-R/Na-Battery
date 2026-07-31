#!/usr/bin/env bash
# run.sh — full classical-ML pipeline on the pod, then upload to R2, then self-terminate.
#
#   cd /workspace/sib/code
#   tmux new -s run
#   bash runpod/run.sh 2>&1 | tee run.log
#
# Runs, in order (override with RUN_STAGES in r2.env):
#   1. scripts/00_build_canonical.py               — raw IRD -> canonical tables + QC
#   2. scripts/01_build_features.py                — parquets for P1/P2/P3
#   3. scripts/05_analyze_degradation.py           — RPT-SOH and projected threshold tables
#   4. scripts/02_run_all_experiments.py           — locked 75/25 + five-fold CV (classical)
#   5. scripts/03_aggregate_results.py             — pooled/macro/bootstrap classical tables
#   6. scripts/04_plot_results.py                  — classical publication figure suite
#   7. scripts/06_run_pinn_experiments.py          — DNN-Q + NaPINN-Q fold training
#   8. scripts/07_aggregate_pinn_results.py        — PINN metric aggregation
#   9. scripts/08_plot_pinn_results.py             — PINN publication plots
#  10. scripts/10_run_pinn_ablations.py            — physics ablation A0..A5
#  11. scripts/09_combine_classical_pinn_results.py — combined report
#
# Exit codes: 0 = pipeline + upload + shutdown attempted. Non-zero = pipeline
# stage failed; we STILL try to upload whatever exists but leave the pod up.
set -o pipefail

# Preserve one-off caller overrides even though r2.env is sourced below.
CALLER_RUN_STAGES="${RUN_STAGES:-}"
CALLER_STOP_MODE="${STOP_MODE:-}"

echo "[run] booting $(date -u +%Y-%m-%dT%H:%M:%SZ)  pid=$$  bash=${BASH_VERSION}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$HERE/.." && pwd)"
ENV_FILE="$HERE/r2.env"
POD_ENV_FILE="$HERE/pod.env"
echo "[run] CODE_DIR=$CODE_DIR"
echo "[run] ENV_FILE=$ENV_FILE"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing. Push it from laptop with:  ./runpod/push.sh --with-env" >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
[[ -n "$CALLER_RUN_STAGES" ]] && RUN_STAGES="$CALLER_RUN_STAGES"
[[ -n "$CALLER_STOP_MODE" ]] && STOP_MODE="$CALLER_STOP_MODE"
if [[ -f "$POD_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$POD_ENV_FILE"; set +a
  echo "[run] pod.env loaded  SIB_DATASET_ROOT=${SIB_DATASET_ROOT:-<unset>}"
else
  echo "ERROR: $POD_ENV_FILE missing. Run  bash runpod/setup.sh  first." >&2
  exit 1
fi
echo "[run] r2.env loaded  R2_BUCKET=${R2_BUCKET:-<unset>}  STOP_MODE=${STOP_MODE:-<unset>}"

: "${POD_WORKSPACE:=/workspace}"
: "${POD_CODE_DIR:=${CODE_DIR}}"
: "${STOP_MODE:=terminate}"
: "${R2_PREFIX:=sodium_ion_battery}"
: "${RUN_STAGES:=all}"

# Expand the alias 'all' to the full classical + PINN pipeline.
# NOTE: pinn_ablation is excluded — ablation suite is deferred (re-enable A0–A7 in future).
if [[ "$RUN_STAGES" == "all" ]]; then
  RUN_STAGES="canonical,build,degradation,experiments,aggregate,plot,pinn,pinn_aggregate,pinn_plot,combined_report"
fi

# Alias for the unified-preprocessing revision. 'tests' runs first and gates
# everything else: the whole point of the revision is that the wiring is
# verified, so a failing test must stop the run before hours of GPU time.
# 'pinn' trains the primary seed only; 'pinn_seeds' adds the robustness spread.
# NOTE: pinn_ablation excluded — deferred for future work.
if [[ "$RUN_STAGES" == "revision" ]]; then
  RUN_STAGES="tests,build_unified,degradation,experiments,aggregate,plot,pinn,pinn_seeds,pinn_aggregate,pinn_plot,combined_report"
fi

# Development-only variant: everything except the holdout. This is what to run
# until every setting is frozen (config experiment.evaluate_holdout stays false).
if [[ "$RUN_STAGES" == "revision_dev" ]]; then
  RUN_STAGES="tests,build_unified,experiments,aggregate,pinn,pinn_aggregate,combined_report"
fi

# Fast wiring check: tests + feature build + one short fold per model.
if [[ "$RUN_STAGES" == "revision_smoke" ]]; then
  RUN_STAGES="tests,build_unified,pinn_smoke"
fi

log(){ printf '\n[run %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# --- venv --------------------------------------------------------------------
if [[ ! -f "${POD_CODE_DIR}/.venv/bin/activate" ]]; then
  echo "ERROR: venv missing at ${POD_CODE_DIR}/.venv. Run setup.sh first." >&2
  exit 1
fi
: "${PS1:=}"; export PS1     # some activate scripts read $PS1 unconditionally
# shellcheck disable=SC1091
source "${POD_CODE_DIR}/.venv/bin/activate"
echo "[run] python: $(command -v python)  $(python --version 2>&1)"

# --- export SIB_* so config.yaml env-var interpolation picks them up ---------
: "${SIB_CANONICAL:=${SIB_ARTIFACTS}/canonical}"
export SIB_ROOT SIB_DATA_STATICS SIB_DATASET_ROOT SIB_ARTIFACTS SIB_CANONICAL SIB_LOGS
mkdir -p "${SIB_ARTIFACTS}" "${SIB_CANONICAL}" "${SIB_LOGS}"

# --- memory / thread hygiene ------------------------------------------------
# glibc can create up to 8*CPUs malloc arenas per process; cap it.
export MALLOC_ARENA_MAX=2
# Force BLAS / OpenMP libraries single-threaded so joblib workers don't
# multiply out to (workers * BLAS_threads) threads competing on the box.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Optional worker-count override: overrides config.yaml at run time.
# scripts/02 uses cfg["experiment"]["parallel_workers"] but honours the env var
# by way of the yaml env interpolation you'd add — for now we log it and let
# the config default win unless the user has bumped it there manually.
if [[ -n "${SIB_PARALLEL_WORKERS:-}" ]]; then
  log "SIB_PARALLEL_WORKERS=${SIB_PARALLEL_WORKERS} (edit config.yaml if you want the scripts to honour this)"
fi

cd "$POD_CODE_DIR"

# Fail before a long run when an rsync deployed only part of the new pipeline.
REQUIRED_SOURCE=(
  scripts/00_build_canonical.py
  scripts/01_build_features.py
  scripts/05_analyze_degradation.py
  scripts/02_run_all_experiments.py
  scripts/03_aggregate_results.py
  scripts/04_plot_results.py
  scripts/06_run_pinn_experiments.py
  scripts/07_aggregate_pinn_results.py
  scripts/08_plot_pinn_results.py
  scripts/09_combine_classical_pinn_results.py
  # scripts/10_run_pinn_ablations.py  # DEFERRED
  src/io/canonical.py
  src/preprocessing/protocol_segmentation.py
  src/preprocessing/p3_waveform.py
  src/splits/group_kfold.py
  src/evaluation/report.py
  src/pinn/__init__.py
  src/pinn/dataset.py
  src/pinn/models.py
  src/pinn/physics.py
  src/pinn/trainer.py
  src/pinn/batching.py
  src/preprocessing/unified.py
  src/preprocessing/unified_feature_registry.py
  src/preprocessing/curve_features.py
  src/preprocessing/dcir.py
  src/preprocessing/fold_transform.py
  tests/test_unified_registry.py
  tests/test_unified_parity.py
  tests/test_pinn_config_wiring.py
  tests/test_pinn_hard_initial_condition.py
)
for required in "${REQUIRED_SOURCE[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: incomplete code deployment; missing ${POD_CODE_DIR}/${required}" >&2
    exit 1
  fi
done
if ! python -c 'from src.io.canonical import build_canonical_tables; from src.preprocessing.protocol_segmentation import segment_cyc_protocol'; then
  echo "ERROR: canonical pipeline import preflight failed" >&2
  exit 1
fi
if ! python -c 'from src.pinn.models import NaPINNQ; from src.pinn.physics import autograd_du_ds; import torch'; then
  echo "ERROR: PINN import preflight failed (torch or src.pinn.* missing)" >&2
  exit 1
fi

# Unified-preprocessing preflight. Also asserts the invariants that make the
# comparison fair, so a stale deployment cannot quietly run the old wiring.
if ! python - <<'PY'
import sys
from src.pinn.batching import CellBalancedSampler
from src.preprocessing.fold_transform import UnifiedFoldPreprocessor
from src.preprocessing.unified import build_unified
from src.preprocessing.unified_feature_registry import (
    MAX_MODEL_FEATURES, N_MODEL_FEATURES, model_feature_columns,
    registry_hash, initial_state_column, stress_coordinate_column,
)
from src.pinn.losses import GradientBalancer, LossWeights
from src.io.loaders import load_config

problems = []
if N_MODEL_FEATURES > MAX_MODEL_FEATURES:
    problems.append(f"registry declares {N_MODEL_FEATURES} > {MAX_MODEL_FEATURES}")

aux = model_feature_columns(exclude_roles=("stress_coordinate", "initial_state"))
if stress_coordinate_column() in aux or initial_state_column() in aux:
    problems.append("stress coordinate / initial state duplicated in PINN features")

cfg = load_config()
if (cfg.get("experiment") or {}).get("preprocessing_levels") != ["unified"]:
    problems.append("experiment.preprocessing_levels is not ['unified']")
if cfg["pinn"]["preprocessing"] != ["unified"]:
    problems.append("pinn.preprocessing is not ['unified']")
if cfg["pinn"]["model"]["rate_uses_u_hat"] is not False:
    problems.append("pinn.model.rate_uses_u_hat must be false")
if cfg["pinn"]["model"]["predict_delta_u"] is not True:
    problems.append("pinn.model.predict_delta_u must be true")
if cfg["pinn"]["training"]["checkpoint_after_physics_active"] is not True:
    problems.append("checkpoint_after_physics_active must be true")
if problems:
    for problem in problems:
        print(f"PREFLIGHT: {problem}", file=sys.stderr)
    sys.exit(1)

print(f"[preflight] unified registry ok: {N_MODEL_FEATURES} features "
      f"({len(aux)} PINN auxiliary), hash={registry_hash()[:12]}")
PY
then
  echo "ERROR: unified-preprocessing preflight failed" >&2
  exit 1
fi

# The locked holdout must stay untouched until every setting is frozen.
if python -c 'import sys; from src.io.loaders import load_config; sys.exit(0 if (load_config().get("experiment") or {}).get("evaluate_holdout") else 1)'; then
  log "WARNING: experiment.evaluate_holdout=TRUE — this run WILL read the locked holdout."
  log "WARNING: only do this once, after preprocessing/architecture/losses are frozen."
else
  log "locked holdout: gated off (development-only run)"
fi

# --- run pipeline ------------------------------------------------------------
PIPELINE_RC=0
IFS=',' read -r -a STAGES <<< "$RUN_STAGES"

# run_stage <name> <script> [extra args...]
# Extra args are forwarded to the script, so stages can pass flags such as
# --unified-only or --seed without needing a bespoke case branch.
run_stage(){
  local name="$1" script="$2"; shift 2
  if [[ $PIPELINE_RC -ne 0 ]]; then
    log "skipping stage '$name' — earlier stage failed"
    return 0
  fi
  log "===== stage: $name -> python $script $* ====="
  local t0 rc
  t0=$SECONDS
  python "$script" "$@"
  rc=$?
  log "stage '$name' finished rc=$rc in $((SECONDS - t0))s"
  if [[ $rc -ne 0 ]]; then
    PIPELINE_RC=$rc
  fi
}

# run_cmd <name> <command...> — same accounting for non-`python script` stages.
run_cmd(){
  local name="$1"; shift
  if [[ $PIPELINE_RC -ne 0 ]]; then
    log "skipping stage '$name' — earlier stage failed"
    return 0
  fi
  log "===== stage: $name -> $* ====="
  local t0 rc
  t0=$SECONDS
  "$@"
  rc=$?
  log "stage '$name' finished rc=$rc in $((SECONDS - t0))s"
  if [[ $rc -ne 0 ]]; then
    PIPELINE_RC=$rc
  fi
}

# require_cuda — fail-fast check before any PINN stage.
# Prints diagnostics and returns non-zero when CUDA is unavailable.
require_cuda(){
  python - <<'PY'
import sys, torch
print(f"[CUDA CHECK] torch={torch.__version__}  build={torch.version.cuda}  available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print(
        "[FATAL] CUDA is unavailable in the active Python environment. "
        "PINN training must not fall back to CPU.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"[CUDA CHECK] device={torch.cuda.get_device_name(0)}")
PY
}

log "===== pipeline start (stages: $RUN_STAGES) ====="
for stage in "${STAGES[@]}"; do
  case "$stage" in
    canonical)
      if [[ -f "${SIB_CANONICAL}/file_manifest.parquet" && \
            -f "${SIB_CANONICAL}/cycles.parquet" && \
            -f "${SIB_CANONICAL}/rpt_measurements.parquet" ]]; then
        log "canonical tables already present at ${SIB_CANONICAL}; skipping canonical stage"
      else
        run_stage canonical scripts/00_build_canonical.py
      fi
      ;;
    build)
      if [[ ! -f "${SIB_CANONICAL}/file_manifest.parquet" || \
            ! -f "${SIB_CANONICAL}/cycles.parquet" || \
            ! -f "${SIB_CANONICAL}/rpt_measurements.parquet" ]]; then
        log "canonical prerequisites missing; running canonical stage before build"
        run_stage canonical scripts/00_build_canonical.py
      fi
      if [[ -f "${SIB_ARTIFACTS}/features/unified.parquet" && \
            -f "${SIB_ARTIFACTS}/features/cu_capacity.parquet" ]]; then
        log "feature tables already present at ${SIB_ARTIFACTS}/features; skipping build stage"
      else
        run_stage build scripts/01_build_features.py
      fi
      ;;
    build_unified)
      if [[ ! -f "${SIB_CANONICAL}/file_manifest.parquet" || \
            ! -f "${SIB_CANONICAL}/cycles.parquet" || \
            ! -f "${SIB_CANONICAL}/steps.parquet" || \
            ! -f "${SIB_CANONICAL}/rpt_measurements.parquet" ]]; then
        log "canonical prerequisites missing; running canonical stage before build_unified"
        run_stage canonical scripts/00_build_canonical.py
      fi
      if [[ -f "${SIB_ARTIFACTS}/features/unified.parquet" && \
            -f "${SIB_ARTIFACTS}/features/cu_capacity.parquet" ]]; then
        log "unified features already present at ${SIB_ARTIFACTS}/features; skipping build_unified stage"
      else
        # Only cu_capacity + unified.parquet. Much cheaper than the full build:
        # two cycles per CYC file instead of every cycle.
        run_stage build_unified scripts/01_build_features.py --unified-only
      fi
      ;;
    tests)
      # Gate the run on the wiring tests. -x so the first failure stops the
      # suite instead of burning time on a broken deployment.
      run_cmd tests python -m pytest tests/ -q -x
      ;;
    experiments) run_stage experiments scripts/02_run_all_experiments.py --tuned ;;
    degradation) run_stage degradation scripts/05_analyze_degradation.py ;;
    aggregate)   run_stage aggregate   scripts/03_aggregate_results.py ;;
    plot)        run_stage plot        scripts/04_plot_results.py ;;
    pinn)
      # Primary result: seed 42 only. This is the seed every tuning decision
      # was made on, and the only one that may appear as a headline number.
      export SIB_DEVICE=cuda
      require_cuda || { PIPELINE_RC=1; continue; }
      run_stage pinn scripts/06_run_pinn_experiments.py \
        --seed "${SIB_PRIMARY_SEED:-42}"
      ;;
    pinn_seeds)
      # Robustness spread ONLY. Reported as "spread across random
      # initializations", never seed-averaged into the headline metric and never
      # used to select anything. A failure here must not invalidate the primary
      # seed, so the loop records but does not propagate a non-zero rc.
      if [[ $PIPELINE_RC -ne 0 ]]; then
        log "skipping stage 'pinn_seeds' — earlier stage failed"
      else
        export SIB_DEVICE=cuda
        if ! require_cuda; then
          log "WARNING: pinn_seeds skipped — CUDA unavailable"
        else
          for extra_seed in ${SIB_SPREAD_SEEDS:-43 44 45 46}; do
            log "===== stage: pinn_seeds (seed=$extra_seed) ====="
            t0=$SECONDS
            python scripts/06_run_pinn_experiments.py --seed "$extra_seed"
            rc=$?
            log "pinn_seeds seed=$extra_seed finished rc=$rc in $((SECONDS - t0))s"
            if [[ $rc -ne 0 ]]; then
              log "WARNING: spread seed $extra_seed failed; primary seed result is unaffected"
            fi
          done
        fi
      fi
      ;;
    pinn_smoke)
      # Short single-fold wiring check: dropout active, hard IC enforced,
      # PDE gradients non-zero, checkpoint eligibility and refit curriculum sane.
      export SIB_DEVICE=cuda
      require_cuda || { PIPELINE_RC=1; continue; }
      run_stage pinn_smoke scripts/06_run_pinn_experiments.py \
        --smoke-test --fold 0 --seed "${SIB_PRIMARY_SEED:-42}"
      ;;
    pinn_aggregate)   run_stage pinn_aggregate   scripts/07_aggregate_pinn_results.py ;;
    pinn_plot)        run_stage pinn_plot        scripts/08_plot_pinn_results.py ;;
    # pinn_ablation)  # DEFERRED: re-enable when ablation suite (A0–A7) is ready
    #   # Primary seed and the single active preprocessing level, explicitly.
    #   # Left to its defaults the script would fan out over every seed.
    #   run_stage pinn_ablation scripts/10_run_pinn_ablations.py \
    #     --seed "${SIB_PRIMARY_SEED:-42}" --preprocessing unified
    # ;;
    combined_report)
      if [[ $PIPELINE_RC -ne 0 ]]; then
        log "skipping combined_report — earlier stage failed"
        break
      fi
      # Resolve the classical results directory from config.yaml (via the
      # loaded config) rather than duplicating the YAML default in Bash — the
      # classical pipeline writes to experiment.results_subdir which is
      # 'extratrees_preprocessing_selection', not the legacy fallback.
      CLASSICAL_RESULTS_DIR="$(python - <<'PY'
from pathlib import Path
from src.io.loaders import load_config
cfg = load_config()
subdir = (cfg.get("experiment") or {}).get("results_subdir") or ""
print(Path(cfg["paths"]["artifacts"]) / "results" / subdir)
PY
)"
      COMBINED_ARGS=""
      if [[ -f "${CLASSICAL_RESULTS_DIR}/raw_results.csv" ]]; then
        # If both pipelines are part of the requested run, require both.
        if [[ ",${RUN_STAGES}," == *",pinn,"* || "${RUN_STAGES}" == "all" ]]; then
          COMBINED_ARGS="--require-classical --require-pinn"
        else
          COMBINED_ARGS="--require-classical"
        fi
      fi
      log "===== stage: combined_report -> python scripts/09_combine_classical_pinn_results.py ${COMBINED_ARGS} ====="
      t0=$SECONDS
      python scripts/09_combine_classical_pinn_results.py ${COMBINED_ARGS}
      rc=$?
      log "stage 'combined_report' finished rc=$rc in $((SECONDS - t0))s"
      if [[ $rc -ne 0 ]]; then
        PIPELINE_RC=$rc
      fi
      ;;
    smoke)       run_stage smoke       scripts/00_smoke_test.py ;;
    "" )         ;;
    *) log "unknown stage '$stage' in RUN_STAGES"; PIPELINE_RC=2 ;;
  esac
done
log "===== pipeline end (rc=$PIPELINE_RC) ====="

# --- bundle results ----------------------------------------------------------
# Drop any old bundles first so tar can't ENOSPC on a full volume.
log "cleaning old bundles from ${POD_WORKSPACE}"
rm -f "${POD_WORKSPACE}"/results_*.tar.zst \
      "${POD_WORKSPACE}"/results_*.tar.gz \
      "${POD_WORKSPACE}"/results_*.meta.json

TS="$(date -u +%Y%m%dT%H%M%SZ)"
if command -v zstd >/dev/null 2>&1; then
  BUNDLE="${POD_WORKSPACE}/results_${TS}.tar.zst"
  TAR_COMPRESS=(--zstd)
else
  log "zstd not found; falling back to gzip"
  BUNDLE="${POD_WORKSPACE}/results_${TS}.tar.gz"
  TAR_COMPRESS=(-z)
fi

# Bundle what's useful:
#   artifacts/results/            — CSVs + plots (the outputs)
#   artifacts/canonical/          — raw-data lineage and QC
#   artifacts/features/          — complete auditable feature matrices
#   logs/                         — pipeline logs
log "bundling results -> $BUNDLE"
ARTIFACT_PARENT="$(dirname "$SIB_ARTIFACTS")"
ARTIFACT_NAME="$(basename "$SIB_ARTIFACTS")"
LOG_PARENT="$(dirname "$SIB_LOGS")"
LOG_NAME="$(basename "$SIB_LOGS")"
BUNDLE_VALID=0
if tar "${TAR_COMPRESS[@]}" -cf "$BUNDLE" \
    -C "$ARTIFACT_PARENT" \
    "$ARTIFACT_NAME" \
    -C "$LOG_PARENT" "$LOG_NAME"; then
  if tar "${TAR_COMPRESS[@]}" -tf "$BUNDLE" >/dev/null; then
    BUNDLE_VALID=1
  else
    log "ERROR: bundle verification failed"
  fi
else
  log "ERROR: archive creation failed; refusing upload and shutdown"
fi
if [[ $BUNDLE_VALID -ne 1 && $PIPELINE_RC -eq 0 ]]; then
  PIPELINE_RC=3
fi

BUNDLE_SIZE="$(du -h "$BUNDLE" 2>/dev/null | cut -f1)"
BUNDLE_SHA256="$(sha256sum "$BUNDLE" 2>/dev/null | cut -d' ' -f1)"
log "bundle size: ${BUNDLE_SIZE:-unknown}  sha256: ${BUNDLE_SHA256:-unavailable}"

# --- upload to R2 (S3-compatible) -------------------------------------------
UPLOAD_RC=1
if [[ $BUNDLE_VALID -eq 1 && -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" && -n "${R2_BUCKET:-}" ]]; then
  export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
  export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
  export AWS_DEFAULT_REGION=auto
  R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
  R2_KEY="${R2_PREFIX}/$(basename "$BUNDLE")"

  log "uploading to s3://${R2_BUCKET}/${R2_KEY}"
  if aws --endpoint-url "$R2_ENDPOINT" s3 cp "$BUNDLE" "s3://${R2_BUCKET}/${R2_KEY}"; then
    UPLOAD_RC=0
    META="${POD_WORKSPACE}/results_${TS}.meta.json"
    cat > "$META" <<JSON
{
  "timestamp": "${TS}",
  "pipeline_rc": ${PIPELINE_RC},
  "pod_id": "${RUNPOD_POD_ID:-unknown}",
  "bundle_key": "${R2_KEY}",
  "bundle_size": "${BUNDLE_SIZE}",
  "bundle_sha256": "${BUNDLE_SHA256}",
  "stages": "${RUN_STAGES}"
}
JSON
    aws --endpoint-url "$R2_ENDPOINT" s3 cp "$META" \
      "s3://${R2_BUCKET}/${R2_PREFIX}/results_${TS}.meta.json" || true
    rm -f "$BUNDLE" "$META"
  else
    log "ERROR: R2 upload failed"
  fi
elif [[ $BUNDLE_VALID -ne 1 ]]; then
  log "bundle invalid; skipping upload"
else
  log "R2 credentials not fully set; skipping upload"
fi

# --- self-terminate ----------------------------------------------------------
maybe_shutdown(){
  local action="$1" pod_id="${RUNPOD_POD_ID:-}"
  if [[ -z "$pod_id" ]]; then
    log "RUNPOD_POD_ID not set — can't identify pod, skipping $action"; return 0
  fi
  if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    log "RUNPOD_API_KEY not set — skipping $action"; return 0
  fi
  local mutation
  case "$action" in
    terminate) mutation="mutation { podTerminate(input: { podId: \"$pod_id\" }) }" ;;
    stop)      mutation="mutation { podStop(input: { podId: \"$pod_id\" }) { id desiredStatus } }" ;;
    *) log "unknown action $action"; return 1 ;;
  esac
  log "requesting RunPod $action for pod $pod_id"
  curl -sS -X POST \
    "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$(jq -cn --arg q "$mutation" '{query:$q}')" \
    || log "curl to RunPod API failed"
  echo
}

if [[ $PIPELINE_RC -ne 0 ]]; then
  log "pipeline failed (rc=$PIPELINE_RC); leaving pod alive for inspection"
elif [[ $UPLOAD_RC -ne 0 ]]; then
  log "upload failed; leaving pod alive so bundle isn't lost"
else
  case "$STOP_MODE" in
    terminate) maybe_shutdown terminate ;;
    stop)      maybe_shutdown stop ;;
    none|"")   log "STOP_MODE=$STOP_MODE — not shutting down" ;;
    *)         log "STOP_MODE=$STOP_MODE unknown — not shutting down" ;;
  esac
fi

exit $PIPELINE_RC
