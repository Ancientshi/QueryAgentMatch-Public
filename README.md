# agent_rec (decoupled research scaffold)

This is a decoupled scaffold for agent recommendation research.

- Positives: ranking top-5
- Evaluation: fixed Top-10
- Report: PartI / PartII / PartIII / Overall

## Run (BPR-MF + KNN q-vector)

```bash
python run_bpr_mf_knn.py \
  --data_root /path/to/dataset_root \
  --device cuda:0 \
  --epochs 5 --batch_size 4096 --factors 128 --neg_per_pos 1 \
  --knn_N 3 --eval_cand_size 100 --score_mode dot
```

## Run (LightFM)

```bash
python run_lightfm_handwritten.py \
  --data_root /path/to/dataset_root \
  --device cuda:0 \
  --epochs 5 --batch_size 4096 --factors 128 --neg_per_pos 1 \
  --alpha_id 1.0 --alpha_feat 1.0 --max_features 5000 \
  --knn_N 3 --eval_cand_size 100 \
  --use_tool_id_emb 1 \
```

Notes:
- This scaffold assumes you already have `utils.py` in the same folder as `run_bpr_mf_knn.py`,
  providing `print_metrics_table(...)` as your original code uses.