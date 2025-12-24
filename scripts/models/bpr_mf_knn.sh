#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

log_cfg "model=bpr_mf_knn"

python "$SCRIPT_DIR/../../run_bpr_mf_knn.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-4096}" \
  --factors 128 --neg_per_pos 1 \
  --knn_N 3 --eval_cand_size 100 --score_mode dot \
  --exp_name "bpr_mf_knn${EXP_SUFFIX:+_$EXP_SUFFIX}"
