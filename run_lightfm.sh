python run_lightfm.py \
  --data_root /home/yunxshi/Data/workspace/QueryAgentMatch/benchmark \
  --device cuda:0 \
  --epochs 5 --batch_size 4096 --factors 128 --neg_per_pos 1 \
  --alpha_id 1.0 --alpha_feat 1.0 --max_features 5000 \
  --knn_N 3 --eval_cand_size 100 \
  --use_tool_id_emb 1 \