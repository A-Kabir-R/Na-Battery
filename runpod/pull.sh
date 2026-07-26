#!/usr/bin/env bash
# pull.sh — download result bundles from R2 to the laptop.
#
#   ./runpod/pull.sh                       # download the newest bundle
#   ./runpod/pull.sh --all                 # download every bundle under the prefix
#   ./runpod/pull.sh results_20260722...   # download a specific key (bare name ok)
#   ./runpod/pull.sh --list                # list bundles, don't download
#
# Bundles land in ./runpod/downloads/ by default. Override with OUT_DIR=... .
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$HERE/r2.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing." >&2; exit 1
fi
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

: "${R2_ACCOUNT_ID:?}"
: "${R2_ACCESS_KEY_ID:?}"
: "${R2_SECRET_ACCESS_KEY:?}"
: "${R2_BUCKET:?}"
: "${R2_PREFIX:=sodium_ion_battery}"

OUT_DIR="${OUT_DIR:-$HERE/downloads}"
mkdir -p "$OUT_DIR"

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION=auto
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: awscli not installed on laptop. Install with: pipx install awscli   (or pip install awscli)" >&2
  exit 1
fi

list_bundles(){
  aws --endpoint-url "$R2_ENDPOINT" s3 ls "s3://${R2_BUCKET}/${R2_PREFIX}/" \
    | awk '$4 ~ /\.tar\.zst$/ {print $4, $3}' || true
}

download_key(){
  local key="$1"
  # accept bare filenames
  case "$key" in
    */*) : ;;
    *)   key="${R2_PREFIX}/${key}" ;;
  esac
  local dest="${OUT_DIR}/$(basename "$key")"
  echo "[pull] s3://${R2_BUCKET}/${key}  ->  ${dest}"
  aws --endpoint-url "$R2_ENDPOINT" s3 cp "s3://${R2_BUCKET}/${key}" "$dest"
  # also try to pull the sidecar meta if it exists
  local meta_key="${key%.tar.zst}.meta.json"
  aws --endpoint-url "$R2_ENDPOINT" s3 cp \
    "s3://${R2_BUCKET}/${meta_key}" "${OUT_DIR}/$(basename "$meta_key")" 2>/dev/null || true
  echo "[pull] done: $dest"
}

case "${1:-}" in
  --list)
    echo "[pull] bundles under s3://${R2_BUCKET}/${R2_PREFIX}/:"
    list_bundles
    ;;
  --all)
    while read -r name _size; do
      [[ -z "$name" ]] && continue
      download_key "${R2_PREFIX}/${name}"
    done < <(list_bundles)
    ;;
  "")
    latest="$(list_bundles | sort | tail -n1 | awk '{print $1}')"
    if [[ -z "$latest" ]]; then
      echo "no bundles found under s3://${R2_BUCKET}/${R2_PREFIX}/" >&2
      exit 1
    fi
    download_key "${R2_PREFIX}/${latest}"
    ;;
  *)
    download_key "$1"
    ;;
esac
