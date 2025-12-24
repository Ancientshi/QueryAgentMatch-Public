#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

RUN_EPOCHS="$EPOCHS"
RUN_BATCH_SIZE="$BATCH_SIZE"
if [[ -z "${USER_SET_EPOCHS}" ]]; then
  RUN_EPOCHS=5
fi
if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
  RUN_BATCH_SIZE=4096
fi
EPOCHS="$RUN_EPOCHS"
BATCH_SIZE="$RUN_BATCH_SIZE"

TEXT_HIDDEN="${TEXT_HIDDEN:-256}"
ID_DIM="${ID_DIM:-32}"
MAX_FEATURES="${MAX_FEATURES:-5000}"
NEG_PER_POS="${NEG_PER_POS:-1}"
EVAL_CAND_SIZE="${EVAL_CAND_SIZE:-100}"
USE_QUERY_ID_EMB="${USE_QUERY_ID_EMB:-0}"

log_cfg "model=dnn text_hidden=$TEXT_HIDDEN id_dim=$ID_DIM max_features=$MAX_FEATURES"

python "$SCRIPT_DIR/../../run_dnn.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$RUN_EPOCHS" \
  --batch_size "$RUN_BATCH_SIZE" \
  --text_hidden "$TEXT_HIDDEN" \
  --id_dim "$ID_DIM" \
  --max_features "$MAX_FEATURES" \
  --neg_per_pos "$NEG_PER_POS" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --use_query_id_emb "$USE_QUERY_ID_EMB" \
  --exp_name "bpr_dnn${EXP_SUFFIX:+_$EXP_SUFFIX}"
