#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DATA_ROOT=/path/to/dataset_root \
#   MODEL_PATH=/path/to/models/latest_xxxx.pt \
#   HOST=0.0.0.0 \
#   PORT=8000 \
#   DEVICE=cpu \
#   TOPK=10 \
#   ./scripts/run_twotower_tfidf_infer.sh

DATA_ROOT=${DATA_ROOT:-""}
MODEL_PATH=${MODEL_PATH:-""}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-"cpu"}
TOPK=${TOPK:-10}
MAX_FEATURES=${MAX_FEATURES:-8192}

if [[ -z "${DATA_ROOT}" || -z "${MODEL_PATH}" ]]; then
  echo "[error] Please set DATA_ROOT and MODEL_PATH before running." >&2
  exit 1
fi

python infer_twotower_tfidf.py \
  --data_root "${DATA_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --topk "${TOPK}" \
  --max_features "${MAX_FEATURES}" \
  --serve 1
