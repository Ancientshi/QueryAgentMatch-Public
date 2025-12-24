#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

RUN_EPOCHS="$EPOCHS"
RUN_BATCH_SIZE="$BATCH_SIZE"
if [[ -z "${USER_SET_EPOCHS}" ]]; then
  RUN_EPOCHS=3
fi
if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
  RUN_BATCH_SIZE=256
fi
EPOCHS="$RUN_EPOCHS"
BATCH_SIZE="$RUN_BATCH_SIZE"

TUNE_MODE="${TUNE_MODE:-lora}"
UNFREEZE_LAST_N="${UNFREEZE_LAST_N:-2}"
UNFREEZE_EMB="${UNFREEZE_EMB:-1}"
GRAD_CKPT="${GRAD_CKPT:-0}"
POOLING="${POOLING:-cls}"
MAX_LEN="${MAX_LEN:-128}"
TEXT_HIDDEN="${TEXT_HIDDEN:-256}"
ID_DIM="${ID_DIM:-32}"
NEG_PER_POS="${NEG_PER_POS:-1}"
TOPK="${TOPK:-10}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-distilbert-base-uncased}"
ENCODER_LR="${ENCODER_LR:-5e-5}"
ENCODER_WEIGHT_DECAY="${ENCODER_WEIGHT_DECAY:-0.01}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGETS="${LORA_TARGETS:-q_lin,k_lin,v_lin,out_lin}"
USE_QUERY_ID_EMB="${USE_QUERY_ID_EMB:-0}"

log_cfg "model=bpr_bert tune_mode=$TUNE_MODE pretrained_model=$PRETRAINED_MODEL pooling=$POOLING"

python "$SCRIPT_DIR/../../run_bpr_bert.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$RUN_EPOCHS" \
  --batch_size "$RUN_BATCH_SIZE" \
  --pretrained_model "$PRETRAINED_MODEL" \
  --max_len "$MAX_LEN" \
  --text_hidden "$TEXT_HIDDEN" \
  --id_dim "$ID_DIM" \
  --neg_per_pos "$NEG_PER_POS" \
  --topk "$TOPK" \
  --pooling "$POOLING" \
  --tune_mode "$TUNE_MODE" \
  --unfreeze_last_n "$UNFREEZE_LAST_N" \
  --unfreeze_emb "$UNFREEZE_EMB" \
  --grad_ckpt "$GRAD_CKPT" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_targets "$LORA_TARGETS" \
  --encoder_lr "$ENCODER_LR" \
  --encoder_weight_decay "$ENCODER_WEIGHT_DECAY" \
  --use_query_id_emb "$USE_QUERY_ID_EMB" \
  --exp_name "bpr_bert${EXP_SUFFIX:+_$EXP_SUFFIX}"
