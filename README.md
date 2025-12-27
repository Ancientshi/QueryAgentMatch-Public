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

## Run (Generative structured recommender)

Generate structured token outputs (LLM + tools) for a query:

```bash
python run_generative.py \
  --data_root /path/to/dataset_root \
  --query "如何写个爬虫？" \
  --top_k 3 \
  --with_metadata 1
```

Export supervised pairs for seq2seq finetuning (JSONL):

```bash
python run_generative.py \
  --data_root /path/to/dataset_root \
  --export_pairs /tmp/generative_pairs.jsonl \
  --max_examples 5000
```

Shell helper (env-style) to avoid long CLI strings:

```bash
DATA_ROOT=/path/to/dataset_root \
QUERY="如何写个爬虫？" \
TOP_K=3 \
WITH_METADATA=1 \
./scripts/run_generative.sh
```
