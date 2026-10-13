#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if [[ "$NPROC_PER_NODE" -lt 2 ]]; then
  echo "ERROR: train_ddp.sh requires at least two local CUDA processes." >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

exec torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train.py \
  --config configs/train_expert_ddp.yaml \
  "$@"
