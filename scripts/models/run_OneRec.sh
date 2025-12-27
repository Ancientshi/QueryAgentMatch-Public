export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

python "$SCRIPT_DIR/../../OneRec_plus.py" \
    --data_root "$DATA_ROOT" \
    --device "cuda:0" \
    --batch_size 4096 \
    --bge_feature_dir /home/yunxshi/Data/workspace/QueryAgentMatch/QueryAgentMatch-Public/dataset/.cache/shared/features/twotower_bge_fd6ccb32_01915b90 \
    --init_tok_from_bge 1 \
    --freeze_tok_emb 1 \
    --epochs 5 \
    --mode gen --topk 10 \
    --use_sessions 0 --session_len 0 \
    --train_mask 0 --candidate_size 200 \
    --eval_candidate_size 200 \
    --skip_eval 0 \
    --amp 1 \


# python "$SCRIPT_DIR/../../OneRec_plus.py" \
#   --data_root "$DATA_ROOT" \
#   --device "$DEVICE" \
#   --lr 1e-5 \
#   --mode dpo --topk 10 --device cuda:0 \
#   --dpo_steps 1200 --dpo_batch 64 \
#   --max_features 5000 \
#   --use_sessions 1 --session_len 4 \
#   --amp 1 --enc_chunk 512 \
#   --train_mask 0 --candidate_size 200
