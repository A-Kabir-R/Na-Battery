#!/usr/bin/env bash
# setup.sh — one-shot pod bootstrap. Run this on the RunPod pod after push.sh.
#
#   cd /workspace/sib/code
#   bash runpod/setup.sh
#
# Installs system packages, uv + venv, python deps from requirements.txt,
# downloads + extracts the RWTH dataset zip, then auto-detects the DODxxx_*
# subtree and writes SIB_DATASET_ROOT into runpod/pod.env for run.sh.
# Idempotent — safe to re-run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$HERE/.." && pwd)"
ENV_FILE="$HERE/r2.env"
POD_ENV_FILE="$HERE/pod.env"
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }

: "${POD_WORKSPACE:=/workspace}"
: "${POD_ROOT:=/workspace/sib}"
: "${POD_CODE_DIR:=${CODE_DIR}}"
: "${POD_DATA_STATICS_DIR:=${POD_ROOT}/data statics}"
: "${POD_DATASET_DIR:=${POD_ROOT}/dataset}"
: "${POD_RESULTS_DIR:=${POD_CODE_DIR}/artifacts}"

DATASET_URL="${DATASET_URL:-https://publications.rwth-aachen.de/record/987579/files/Data_Failure_Mode_and_Degradation_Analysis_of_a_Commercial_Sodium-Ion_Battery_With_Severe_Gassing_Issue.zip}"
DATASET_ZIP="${POD_WORKSPACE}/dataset.zip"
DATASET_MARKER="${POD_DATASET_DIR}/.download_ok"

log(){ printf '\n[setup] %s\n' "$*"; }

# --- 1. system packages ------------------------------------------------------
declare -A APT_PKGS=(
  [tmux]=tmux [nvim]=neovim [htop]=htop [curl]=curl [wget]=wget
  [unzip]=unzip [zip]=zip [git]=git [rsync]=rsync [jq]=jq
  [zstd]=zstd [gcc]=build-essential
)
MISSING_APT=()
for cmd in "${!APT_PKGS[@]}"; do
  command -v "$cmd" >/dev/null 2>&1 || MISSING_APT+=("${APT_PKGS[$cmd]}")
done
dpkg -s ca-certificates >/dev/null 2>&1 || MISSING_APT+=(ca-certificates)
python3 -c 'import venv' >/dev/null 2>&1 || MISSING_APT+=(python3-venv)
dpkg -s python3-pip >/dev/null 2>&1 || MISSING_APT+=(python3-pip)
# libgomp1 is required by lightgbm; libstdc++/openmp for xgboost + catboost.
dpkg -s libgomp1 >/dev/null 2>&1 || MISSING_APT+=(libgomp1)

if [[ ${#MISSING_APT[@]} -gt 0 ]]; then
  log "installing apt packages: ${MISSING_APT[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends "${MISSING_APT[@]}"
  apt-get clean; rm -rf /var/lib/apt/lists/*
else
  log "apt packages already present, skipping"
fi

# --- 2. uv (fast pip) --------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  log "uv already installed ($(uv --version))"
else
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  grep -q 'HOME/.local/bin' ~/.bashrc 2>/dev/null || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
export PATH="$HOME/.local/bin:$PATH"

# --- 3. python venv + deps ---------------------------------------------------
VENV="${POD_CODE_DIR}/.venv"
USE_UV=0
command -v uv >/dev/null 2>&1 && USE_UV=1

if [[ ! -d "$VENV" ]]; then
  log "creating venv at $VENV (uv=${USE_UV})"
  if [[ $USE_UV -eq 1 ]]; then
    uv venv "$VENV" --python 3.11 --seed || uv venv "$VENV" --seed || python3 -m venv "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  log "bootstrapping pip via ensurepip"
  python -m ensurepip --upgrade || true
fi

# Reinstall when requirements change; import presence alone misses version bumps.
REQ_FILE="${POD_CODE_DIR}/requirements.txt"
REQ_HASH="$(sha256sum "$REQ_FILE" | cut -d' ' -f1)"
REQ_MARKER="${VENV}/.requirements.sha256"
DEPS_OK=0
if [[ -f "$REQ_MARKER" && "$(cat "$REQ_MARKER")" == "$REQ_HASH" ]]; then
python - <<'PY' >/tmp/_deps_ok 2>/dev/null && DEPS_OK=1 || DEPS_OK=0
import importlib.util
required = ["numpy","pandas","scipy","pyarrow","sklearn",
            "xgboost","lightgbm","catboost",
            "matplotlib","seaborn","joblib","yaml","tqdm"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
PY
fi

if [[ "$DEPS_OK" == "1" ]]; then
  log "python deps already satisfied, skipping requirements install"
else
  log "installing project requirements (this can take a few minutes)"
  if [[ $USE_UV -eq 1 ]]; then
    uv pip install --upgrade pip wheel setuptools
    uv pip install -r "${POD_CODE_DIR}/requirements.txt"
  else
    python -m pip install --upgrade pip wheel setuptools
    python -m pip install -r "${POD_CODE_DIR}/requirements.txt"
  fi
  printf '%s\n' "$REQ_HASH" > "$REQ_MARKER"
fi

# awscli lives outside requirements.txt (heavy, only needed by run.sh).
command -v aws >/dev/null 2>&1 || {
  log "installing awscli"
  if [[ $USE_UV -eq 1 ]]; then uv pip install awscli; else python -m pip install awscli; fi
}

# --- 4. dataset download + extract ------------------------------------------
mkdir -p "$POD_DATASET_DIR" "$POD_RESULTS_DIR" "$POD_DATA_STATICS_DIR/tables"

is_zip(){
  local f="$1"
  [[ -f "$f" ]] || return 1
  local sig
  sig=$(head -c 2 "$f" 2>/dev/null | od -An -c | tr -d ' \n')
  [[ "$sig" == "PK" ]]
}

download_dataset(){
  local url="$1" out="$2"
  rm -f "$out"
  # RWTH publications gate downloads behind a JS "fast-challenge" (APP_INIT=1).
  local ua="Mozilla/5.0 (X11; Linux x86_64) sodium-ion-setup/1.0"
  local ck="APP_INIT=1"
  log "attempt 1: curl -L --cookie APP_INIT=1"
  if curl -fL --retry 3 --retry-delay 5 -A "$ua" --cookie "$ck" \
       -C - -o "$out" "$url"; then
    is_zip "$out" && return 0
  fi
  log "attempt 2: wget --header Cookie: APP_INIT=1"
  rm -f "$out"
  if wget --header="Cookie: $ck" --content-disposition --trust-server-names \
       -U "$ua" --tries=3 --timeout=60 -O "$out" "$url"; then
    is_zip "$out" && return 0
  fi
  return 1
}

if [[ -f "$DATASET_MARKER" ]]; then
  log "dataset already present at $POD_DATASET_DIR (marker found), skipping download"
else
  log "downloading dataset ($DATASET_URL)"
  if ! download_dataset "$DATASET_URL" "$DATASET_ZIP"; then
    log "ERROR: dataset download did not produce a zip file."
    log "server response saved at $DATASET_ZIP — first 40 lines:"
    head -c 4000 "$DATASET_ZIP" 2>/dev/null || true
    echo
    log "workarounds:"
    log "  1) open $DATASET_URL in a browser, copy the direct file link,"
    log "     re-run:  DATASET_URL='<direct-url>' bash runpod/setup.sh"
    log "  2) upload the zip yourself:  scp -P <port> file.zip root@<host>:$DATASET_ZIP"
    log "     then re-run this script; it will detect and extract it."
    exit 1
  fi
  zip_size=$(stat -c%s "$DATASET_ZIP")
  log "downloaded ${zip_size} bytes; extracting to $POD_DATASET_DIR"
  unzip -q -o "$DATASET_ZIP" -d "$POD_DATASET_DIR"
  touch "$DATASET_MARKER"
  rm -f "$DATASET_ZIP"
fi

# --- 5. locate the DOD subtree (zip nests it under one or two dirs) ---------
# We want DATASET_ROOT to be the directory that DIRECTLY contains DOD*/ folders,
# because inventory.csv paths look like DOD100_..../CU_at_.../*.ird.
log "probing for DODxxx_* subtree under $POD_DATASET_DIR ..."
DOD_DIR="$(find "$POD_DATASET_DIR" -maxdepth 4 -type d -name 'DOD*_*C*C_*degC' -printf '%h\n' 2>/dev/null | sort -u | head -1 || true)"
if [[ -z "${DOD_DIR:-}" ]]; then
  log "ERROR: could not find any DOD*_*C*C_*degC folder under $POD_DATASET_DIR"
  log "ls of dataset dir:"
  find "$POD_DATASET_DIR" -maxdepth 3 -type d | head -40
  exit 1
fi
log "detected DOD subtree at: $DOD_DIR"
IRD_COUNT="$(find "$DOD_DIR" -maxdepth 3 -type f -name '*.ird' | wc -l)"
if [[ "$IRD_COUNT" -eq 0 ]]; then
  log "ERROR: detected dataset root contains no IRD files: $DOD_DIR"
  exit 1
fi
log "detected ${IRD_COUNT} Standard-cycling IRD files"

# --- 6. check that data statics tables were shipped -------------------------
if [[ ! -f "$POD_DATA_STATICS_DIR/tables/inventory.csv" ]]; then
  log "WARNING: $POD_DATA_STATICS_DIR/tables/inventory.csv missing."
  log "         push it from the laptop with:  ./runpod/push.sh"
  log "         (the pipeline will fail without cycle_metrics.csv + inventory.csv)"
fi

# --- 7. write pod.env with the resolved paths -------------------------------
cat > "$POD_ENV_FILE" <<POD
# Auto-generated by runpod/setup.sh — sourced by run.sh.
SIB_ROOT='${POD_ROOT}'
SIB_DATA_STATICS='${POD_DATA_STATICS_DIR}'
SIB_DATASET_ROOT='${DOD_DIR}'
SIB_ARTIFACTS='${POD_RESULTS_DIR}'
SIB_CANONICAL='${POD_RESULTS_DIR}/canonical'
SIB_LOGS='${POD_CODE_DIR}/logs'
POD
log "wrote $POD_ENV_FILE:"
cat "$POD_ENV_FILE"

# --- 8. quick python sanity --------------------------------------------------
log "python + core deps sanity"
cd "$POD_CODE_DIR"
python - <<'PY'
import sys, importlib
print("python:", sys.version.split()[0])
for m in ("numpy","pandas","scipy","sklearn","xgboost","lightgbm","catboost","yaml"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m}:", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m}: MISSING ({e})")
from src.io.canonical import build_canonical_tables
from src.preprocessing.protocol_segmentation import segment_cyc_protocol
print("  canonical pipeline imports: ok")
PY

log "done. next:  bash runpod/run.sh"
