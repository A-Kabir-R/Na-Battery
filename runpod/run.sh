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
: "${RUN_STAGES:=canonical,build,degradation,experiments,aggregate,plot}"

# Expand the alias 'all' to the full classical + PINN pipeline.
if [[ "$RUN_STAGES" == "all" ]]; then
  RUN_STAGES="canonical,build,degradation,experiments,aggregate,plot,pinn,pinn_aggregate,pinn_plot,pinn_ablation,combined_report"
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
  scripts/10_run_pinn_ablations.py
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

# --- run pipeline ------------------------------------------------------------
PIPELINE_RC=0
IFS=',' read -r -a STAGES <<< "$RUN_STAGES"

run_stage(){
  local name="$1" script="$2"
  if [[ $PIPELINE_RC -ne 0 ]]; then
    log "skipping stage '$name' — earlier stage failed"
    return 0
  fi
  log "===== stage: $name -> python $script ====="
  local t0 rc
  t0=$SECONDS
  python "$script"
  rc=$?
  log "stage '$name' finished rc=$rc in $((SECONDS - t0))s"
  if [[ $rc -ne 0 ]]; then
    PIPELINE_RC=$rc
  fi
}

log "===== pipeline start (stages: $RUN_STAGES) ====="
for stage in "${STAGES[@]}"; do
  case "$stage" in
    canonical)   run_stage canonical   scripts/00_build_canonical.py ;;
    build)
      if [[ ! -f "${SIB_CANONICAL}/file_manifest.parquet" || \
            ! -f "${SIB_CANONICAL}/cycles.parquet" || \
            ! -f "${SIB_CANONICAL}/rpt_measurements.parquet" ]]; then
        log "canonical prerequisites missing; running canonical stage before build"
        run_stage canonical scripts/00_build_canonical.py
      fi
      run_stage build scripts/01_build_features.py
      ;;
    experiments) run_stage experiments scripts/02_run_all_experiments.py ;;
    degradation) run_stage degradation scripts/05_analyze_degradation.py ;;
    aggregate)   run_stage aggregate   scripts/03_aggregate_results.py ;;
    plot)        run_stage plot        scripts/04_plot_results.py ;;
    pinn)             run_stage pinn             scripts/06_run_pinn_experiments.py ;;
    pinn_aggregate)   run_stage pinn_aggregate   scripts/07_aggregate_pinn_results.py ;;
    pinn_plot)        run_stage pinn_plot        scripts/08_plot_pinn_results.py ;;
    pinn_ablation)    run_stage pinn_ablation    scripts/10_run_pinn_ablations.py ;;
    combined_report)  run_stage combined_report  scripts/09_combine_classical_pinn_results.py ;;
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
