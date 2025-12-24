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

FACTORS="${FACTORS:-128}"
NEG_PER_POS="${NEG_PER_POS:-1}"
ALPHA_ID="${ALPHA_ID:-1.0}"
ALPHA_FEAT="${ALPHA_FEAT:-1.0}"
MAX_FEATURES="${MAX_FEATURES:-5000}"
KNN_N="${KNN_N:-3}"
EVAL_CAND_SIZE="${EVAL_CAND_SIZE:-100}"
USE_TOOL_ID_EMB="${USE_TOOL_ID_EMB:-1}"

log_cfg "model=lightfm factors=$FACTORS max_features=$MAX_FEATURES use_tool_id_emb=$USE_TOOL_ID_EMB"

python "$SCRIPT_DIR/../../run_lightfm.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$RUN_EPOCHS" \
  --batch_size "$RUN_BATCH_SIZE" \
  --factors "$FACTORS" \
  --neg_per_pos "$NEG_PER_POS" \
  --alpha_id "$ALPHA_ID" \
  --alpha_feat "$ALPHA_FEAT" \
  --max_features "$MAX_FEATURES" \
  --knn_N "$KNN_N" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --use_tool_id_emb "$USE_TOOL_ID_EMB" \
  --exp_name "lightfm${EXP_SUFFIX:+_$EXP_SUFFIX}"
