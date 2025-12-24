#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

log_cfg "model=bpr_bert_lora"

python "$SCRIPT_DIR/../../run_bpr_bert.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "${EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --tune_mode lora \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --lora_targets q_lin,k_lin,v_lin,out_lin \
  --encoder_lr 5e-5 --encoder_weight_decay 0.01 \
  --pretrained_model distilbert-base-uncased \
  --max_len 128 --text_hidden 256 --id_dim 32 \
  --neg_per_pos 1 --topk 10 --pooling cls \
  --exp_name "bpr_bert_lora${EXP_SUFFIX:+_$EXP_SUFFIX}"
