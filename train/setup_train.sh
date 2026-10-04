#!/usr/bin/env bash
set -euo pipefail

DELIVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${1:-${PROJECT_ROOT:-$(pwd)}}"
CONDA_ROOT="${CONDA_ROOT:-/data/miniconda3}"

if [[ ! -f "$PROJECT_ROOT/scripts/bootstrap.sh" ]]; then
  echo "ERROR: $PROJECT_ROOT is not a lingbot-vla-robotwin checkout." >&2
  echo "Usage: bash train/setup_train.sh /path/to/lingbot-vla-robotwin" >&2
  exit 2
fi

export CONDA_ROOT
export INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

cd "$PROJECT_ROOT"
bash scripts/bootstrap.sh

if ! grep -q "swanlab_run_id" src/lrvla/training.py; then
  PATCH="$DELIVERY_ROOT/server_patch/swanlab_integration.patch"
  if [[ ! -f "$PATCH" ]]; then
    echo "ERROR: SwanLab integration is absent and patch was not found: $PATCH" >&2
    exit 2
  fi
  git apply --check "$PATCH"
  git apply "$PATCH"
  echo "Applied verified SwanLab integration patch."
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate lingbot-vla

python - <<'PY'
import torch
import transformers, datasets, peft, accelerate, safetensors, cv2

assert torch.cuda.is_available(), "CUDA is not available"
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("BF16:", torch.cuda.is_bf16_supported())
PY

python scripts/train.py --smoke
echo "Environment and synthetic smoke test passed."
