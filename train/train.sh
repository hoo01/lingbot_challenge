#!/usr/bin/env bash
set -euo pipefail

DELIVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$DELIVERY_ROOT/train/configs/train_single_gpu.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/data/miniconda3}"

if [[ "$CONFIG" != /* ]]; then
  if [[ -f "$CONFIG" ]]; then
    CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
  elif [[ -f "$DELIVERY_ROOT/$CONFIG" ]]; then
    CONFIG="$DELIVERY_ROOT/$CONFIG"
  fi
fi

cd "$PROJECT_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate lingbot-vla

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

if [[ ! -f scripts/train.py ]]; then
  echo "ERROR: PROJECT_ROOT is not a lingbot-vla-robotwin checkout: $PROJECT_ROOT" >&2
  exit 2
fi

if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "ERROR: the current training implementation is single-GPU only." >&2
  echo "Do not launch it with torchrun until DDP/FSDP support is implemented and tested." >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/train.py --config "$CONFIG" "${@:2}"
