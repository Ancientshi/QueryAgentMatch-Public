#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

RUN_EPOCHS="$EPOCHS"
RUN_BATCH_SIZE="$BATCH_SIZE"
if [[ -z "${USER_SET_EPOCHS}" ]]; then
  RUN_EPOCHS=10
fi
if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
  RUN_BATCH_SIZE=512
fi
EPOCHS="$RUN_EPOCHS"
BATCH_SIZE="$RUN_BATCH_SIZE"

EMBED_URL="${EMBED_URL:-http://127.0.0.1:8502/get_embedding}"
EMBED_BATCH="${EMBED_BATCH:-64}"
HID="${HID:-256}"
TEMPERATURE="${TEMPERATURE:-0.07}"
EVAL_CHUNK="${EVAL_CHUNK:-8192}"
USE_TOOL_EMB="${USE_TOOL_EMB:-0}"
USE_AGENT_ID_EMB="${USE_AGENT_ID_EMB:-0}"
USE_QUERY_ID_EMB="${USE_QUERY_ID_EMB:-0}"
TOPK="${TOPK:-10}"

log_cfg "model=two_tower_bge embed_url=$EMBED_URL embed_batch=$EMBED_BATCH hid=$HID use_tool_emb=$USE_TOOL_EMB"

python "$SCRIPT_DIR/../../run_twotower_bge.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$RUN_EPOCHS" \
  --batch_size "$RUN_BATCH_SIZE" \
  --embed_url "$EMBED_URL" \
  --embed_batch "$EMBED_BATCH" \
  --hid "$HID" \
  --temperature "$TEMPERATURE" \
  --topk "$TOPK" \
  --eval_chunk "$EVAL_CHUNK" \
  --use_tool_emb "$USE_TOOL_EMB" \
  --use_agent_id_emb "$USE_AGENT_ID_EMB" \
  --use_query_id_emb "$USE_QUERY_ID_EMB" \
  --exp_name "two_tower_bge${EXP_SUFFIX:+_$EXP_SUFFIX}"
