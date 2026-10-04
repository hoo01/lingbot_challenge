#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash train/backup_checkpoint.sh RUN_DIR HF_REPO_ID PATH_IN_REPO" >&2
  exit 2
fi

DELIVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/data/miniconda3}"
cd "$PROJECT_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate lingbot-vla

python "$DELIVERY_ROOT/train/backup_checkpoint.py" "$1" "$2" "$3"
