#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="$PROJECT_DIR/results/mvtec_flow/diagnostics"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,COLL
export NCCL_DEBUG_FILE="$LOG_DIR/nccl_%h_%p.log"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29501 \
  train.py \
  --config configs/mvtec_flow.yml \
  --epochs 5 \
  --log_interval 20 \
  --save_diagnostics \
  2>&1 | tee "$LOG_DIR/torchrun_smoke.log"
