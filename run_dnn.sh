python run_dnn.py \
  --data_root /home/yunxshi/Data/workspace/QueryAgentMatch/benchmark \
  --device cuda:0 \
  --epochs 5 --batch_size 4096 --id_dim 32 --neg_per_pos 1 \
  --max_features 5000 \
  --eval_cand_size 100 \