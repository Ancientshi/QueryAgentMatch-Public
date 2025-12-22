#!/usr/bin/env bash
set -euo pipefail

python run_twotower_tfidf.py \
  --data_root /home/yunxshi/Data/workspace/QueryAgentMatch/benchmark \
  --epochs 5 \
  --batch_size 512 \
  --max_features 5000 \
  --hid 256 \
  --temperature 0.07 \
  --topk 10 \
  --device cuda:1 \
  --eval_chunk 8192 \
  --use_tool_emb 1 \
  --use_agent_id_emb 1 \
  --exp_name two_tower_tfidf_toolid_agentid \