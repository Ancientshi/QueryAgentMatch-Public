#!/usr/bin/env bash
set -euo pipefail

# Lite-DPO finetuning for OneRec++ (requires a prior GEN/SFT checkpoint).
# Assumes the CE model is saved under $DATA_ROOT/.cache/OneRec_plus/models/.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

python "$SCRIPT_DIR/../../OneRec_plus.py" \
    --data_root "$DATA_ROOT" \
    --device "${DEVICE:-cuda:0}" \
    --mode dpo --topk 10 \
    --dpo_steps 1200 --dpo_batch 64 \
    --lr 1e-5 \
    --use_sessions 1 --session_len 4 \
    --train_mask 0 --candidate_size 1000 \
    --enc_chunk 512 \
    --amp 1