#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=10
BATCH_SIZE=2048

python "$SCRIPT_DIR/../../run_kgat.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --embed_dim 128 --num_layers 2 --att_dropout 0.2 \
  --neg_per_pos "$NEG_PER_POS" \
  --knn_N "$KNN_N" --eval_cand_size "$EVAL_CAND_SIZE" --score_mode dot \
  --exp_name "kgat${EXP_SUFFIX}"
