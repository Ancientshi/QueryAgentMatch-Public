#!/usr/bin/env bash
set -euo pipefail

# Cross-model defaults. Override via environment variables when launching a script.
DATA_ROOT="${DATA_ROOT:-/path/to/dataset_root}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXP_SUFFIX="${EXP_SUFFIX:-}"

# Log the current configuration for reproducibility.
log_cfg() {
  echo "[cfg] DATA_ROOT=$DATA_ROOT DEVICE=$DEVICE EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE EXP_SUFFIX=$EXP_SUFFIX $*"
}
