#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

KS="${KS:-10}"
VALID_RATIO="${VALID_RATIO:-0.2}"
MAX_EVAL="${MAX_EVAL:-0}"
SAMPLE_PER_PART="${SAMPLE_PER_PART:-200}"
EVAL_CAND_SIZE="${EVAL_CAND_SIZE:-1000}"
POS_TOPK="${POS_TOPK:-0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SEED="${SEED:-1234}"
MAX_LEN="${MAX_LEN:-384}"
ENCODE_BATCH="${ENCODE_BATCH:-64}"
NORMALIZE="${NORMALIZE:-1}"
USE_AMP="${USE_AMP:-1}"

# Set this to the Sentence-Transformers export directory from KALM fine-tuning
KALM_MODEL_DIR="${KALM_MODEL_DIR:-/home/yunxshi/Data/workspace/QueryAgentMatch/benchmark/KALM/st-biencoder}"

python "$SCRIPT_DIR/../infer_KALM.py" \
  --data_root "$DATA_ROOT" \
  --exp_name "infer_kalm_embedding${EXP_SUFFIX}" \
  --model_dir "$KALM_MODEL_DIR" \
  --device "$DEVICE" \
  --encode_batch "$ENCODE_BATCH" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --pos_topk "$POS_TOPK" \
  --ks "$KS" \
  --max_len "$MAX_LEN" \
  --use_amp "$USE_AMP" \
  --normalize "$NORMALIZE" \
  --seed "$SEED" \
  --split_seed "$SPLIT_SEED" \
  --valid_ratio "$VALID_RATIO" \
  --sample_per_part "$SAMPLE_PER_PART" \
  --max_eval "$MAX_EVAL"
