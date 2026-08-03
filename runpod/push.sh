#!/usr/bin/env bash
# push.sh — rsync code + data-statics tables to the RunPod pod.
# Reads SSH info + POD_* dirs from runpod/r2.env.
#
# Usage:  ./runpod/push.sh                 # push code + data statics tables
#         ./runpod/push.sh --with-env      # also push r2.env (needed on pod)
#         ./runpod/push.sh --dry-run       # rsync --dry-run
#         ./runpod/push.sh --code-only     # skip data statics (fast redeploy)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$HERE/.." && pwd)"               # .../Standard cycling/code
PROJECT_ROOT="$(cd "$CODE_ROOT/.." && pwd)"       # .../Standard cycling
ENV_FILE="$HERE/r2.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing. Copy r2.env.example -> r2.env and fill it in." >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

: "${RUNPOD_SSH_HOST:?RUNPOD_SSH_HOST unset}"
: "${RUNPOD_SSH_PORT:=22}"
: "${RUNPOD_SSH_USER:=root}"
: "${RUNPOD_SSH_KEY:=$HOME/.ssh/id_ed25519}"
: "${POD_ROOT:=/workspace/sib}"
: "${POD_CODE_DIR:=${POD_ROOT}/code}"
: "${POD_DATA_STATICS_DIR:=${POD_CODE_DIR}/data_statics}"

RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY/#\~/$HOME}"
SSH_OPTS=(-i "$RUNPOD_SSH_KEY" -p "$RUNPOD_SSH_PORT" -o StrictHostKeyChecking=accept-new)

WITH_ENV=0
CODE_ONLY=0
DRY_RUN=()
for a in "$@"; do
  case "$a" in
    --with-env)  WITH_ENV=1 ;;
    --code-only) CODE_ONLY=1 ;;
    --dry-run)   DRY_RUN=(--dry-run) ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

echo "[push] target: ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}:${RUNPOD_SSH_PORT}"
echo "[push] pod code dir:         ${POD_CODE_DIR}"
echo "[push] pod data-statics dir: ${POD_DATA_STATICS_DIR}"

# Ensure rsync exists on the pod and destination dirs exist.
ssh "${SSH_OPTS[@]}" "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}" bash -s <<REMOTE
set -e
mkdir -p '${POD_CODE_DIR}' '${POD_DATA_STATICS_DIR}/tables' '${POD_DATA_STATICS_DIR}/logs'
if ! command -v rsync >/dev/null 2>&1; then
  echo "[push:remote] installing rsync"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null
  apt-get install -y --no-install-recommends rsync >/dev/null
fi
REMOTE

CODE_EXCLUDES=(
  --exclude=".git/"
  --exclude="__pycache__/"
  --exclude="*.pyc"
  --exclude=".mypy_cache/"
  --exclude=".pytest_cache/"
  --exclude=".ruff_cache/"
  --exclude=".venv/"
  --exclude="venv/"
  --exclude="node_modules/"
  --exclude="artifacts/"
  --exclude="logs/"
  --exclude="runpod/downloads/"
  --exclude="/preprocessing/"
  --exclude="results_*.tar.zst"
  --exclude="results_*.tar.gz"
  --exclude="results_*.meta.json"
  --exclude="*.zip"
  --exclude="*.tar"
  --exclude="*.tar.gz"
  --exclude="*.tar.zst"
  --exclude="*.tgz"
  --exclude="runpod/r2.env"
  --exclude="runpod/pod.env"
)
if [[ $WITH_ENV -eq 1 ]]; then
  _kept=()
  for e in "${CODE_EXCLUDES[@]}"; do
    [[ "$e" == "--exclude=runpod/r2.env" ]] && continue
    _kept+=("$e")
  done
  CODE_EXCLUDES=("${_kept[@]}")
fi

echo "[push] rsyncing code -> ${POD_CODE_DIR}/"
rsync -avz --delete "${DRY_RUN[@]}" \
  -e "ssh ${SSH_OPTS[*]}" \
  "${CODE_EXCLUDES[@]}" \
  "${CODE_ROOT}/" \
  "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}:${POD_CODE_DIR}/"

# Push pre-built canonical + feature tables so the pod can skip parsing.
# preprocessing/ maps directly onto the pod's artifacts/ directory layout.
if [[ -d "${CODE_ROOT}/preprocessing" ]]; then
  echo "[push] rsyncing preprocessing -> ${POD_CODE_DIR}/artifacts/"
  ssh "${SSH_OPTS[@]}" "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}" \
    mkdir -p "${POD_CODE_DIR}/artifacts/canonical" \
             "${POD_CODE_DIR}/artifacts/features" \
             "${POD_CODE_DIR}/artifacts/splits"
  rsync -avz "${DRY_RUN[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    "${CODE_ROOT}/preprocessing/" \
    "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}:${POD_CODE_DIR}/artifacts/"
else
  echo "[push] preprocessing/ not found locally; skipping pre-built tables push"
fi

# Fix pod.env so SIB_ARTIFACTS and SIB_CANONICAL always point to the correct
# artifacts dir regardless of what setup.sh derived from r2.env.
POD_ENV_REMOTE="${POD_CODE_DIR}/runpod/pod.env"
echo "[push] patching pod.env SIB_ARTIFACTS and SIB_CANONICAL paths"
ssh "${SSH_OPTS[@]}" "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}" bash -s <<REMOTE
if [[ -f '${POD_ENV_REMOTE}' ]]; then
  sed -i "s|SIB_ARTIFACTS='[^']*'|SIB_ARTIFACTS='${POD_CODE_DIR}/artifacts'|" '${POD_ENV_REMOTE}'
  sed -i "s|SIB_CANONICAL='[^']*'|SIB_CANONICAL='${POD_CODE_DIR}/artifacts/canonical'|" '${POD_ENV_REMOTE}'
  echo "[push:remote] pod.env patched:"
  grep -E "SIB_ARTIFACTS|SIB_CANONICAL" '${POD_ENV_REMOTE}'
else
  echo "[push:remote] pod.env not found yet (run setup.sh first)"
fi
REMOTE

if [[ ${#DRY_RUN[@]} -eq 0 ]]; then
  echo "[push] validating required pipeline files on pod"
  ssh "${SSH_OPTS[@]}" "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}" bash -s <<REMOTE
set -e
for required in \
  scripts/00_build_canonical.py \
  scripts/01_build_features.py \
  src/io/canonical.py \
  src/preprocessing/protocol_segmentation.py \
  src/preprocessing/unified.py \
  src/preprocessing/unified_feature_registry.py \
  src/preprocessing/curve_features.py \
  src/preprocessing/dcir.py \
  src/preprocessing/fold_transform.py \
  src/pinn/batching.py \
  tests/test_unified_parity.py \
  tests/test_pinn_config_wiring.py; do
  test -f '${POD_CODE_DIR}/'"\$required" || {
    echo "[push:remote] ERROR: incomplete deployment; missing ${POD_CODE_DIR}/\$required" >&2
    exit 1
  }
done
REMOTE
fi

if [[ $CODE_ONLY -eq 0 ]]; then
  # Only push what the ML pipeline actually reads: tables + parse_errors log.
  # Skip the ~hundreds-of-MB plots/ tree.
  DS_LOCAL="${CODE_ROOT}/data_statics"
  if [[ ! -d "${DS_LOCAL}/tables" ]]; then
    echo "[push] WARNING: ${DS_LOCAL}/tables not found — the pod won't have the CSVs the pipeline needs."
  else
    echo "[push] rsyncing data statics/tables -> ${POD_DATA_STATICS_DIR}/tables/"
    rsync -avz "${DRY_RUN[@]}" \
      -e "ssh ${SSH_OPTS[*]}" \
      "${DS_LOCAL}/tables/" \
      "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}:${POD_DATA_STATICS_DIR}/tables/"
  fi
  if [[ -f "${DS_LOCAL}/logs/parse_errors.csv" ]]; then
    echo "[push] rsyncing data statics/logs/parse_errors.csv"
    rsync -avz "${DRY_RUN[@]}" \
      -e "ssh ${SSH_OPTS[*]}" \
      "${DS_LOCAL}/logs/parse_errors.csv" \
      "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}:${POD_DATA_STATICS_DIR}/logs/parse_errors.csv"
  fi
fi

if [[ $WITH_ENV -eq 1 ]]; then
  echo "[push] r2.env included (needed for run.sh self-terminate + R2 upload)"
else
  echo "[push] r2.env NOT pushed. Re-run with --with-env before invoking run.sh on the pod."
fi
echo "[push] done."
