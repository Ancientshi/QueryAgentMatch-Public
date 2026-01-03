## Regime shift analysis

This folder contains a lightweight script to quantify the dense → one-off shift between benchmark parts.

### Usage

```bash
python analysis/regime_shift_analysis.py \
  --data_root /path/to/dataset_root \
  --output_dir analysis/artifacts
```

Requirements:

- The dataset root should contain `PartI`, `PartII`, and `PartIII` subfolders with `agents/merge.json`, `questions/merge.json`, and `rankings/merge.json`.
- `matplotlib` must be available in the environment (preinstalled in the scaffold containers).

Outputs (written to `--output_dir`):

- `regime_shift_metrics.csv`: one table of per-part topology metrics (unique agent ratio, head coverage for top 1%/5%, entropy and effective directory size).
- `regime_shift_frequency.png`: one figure with side-by-side frequency curves (Part I vs Part II/III) plus a log-scale agent-degree histogram.

The defaults cover Parts I/II/III, but you can limit analysis to a subset with `--parts PartI,PartII`.
