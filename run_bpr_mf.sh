python bpr_mf.py \
  --data_root /home/yunxshi/Data/workspace/QueryAgentMatch/benchmark \
  --epochs 10 --batch_size 4096 --factors 128 \
  --neg_per_pos 1 --topk 5 --device cuda:0 \
  --eval_cand_size 100 --knn_N 3 
