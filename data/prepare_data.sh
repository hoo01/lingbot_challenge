#!/usr/bin/env bash
set -euo pipefail

DELIVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${1:-${PROJECT_ROOT:-$(pwd)}}"
CONDA_ROOT="${CONDA_ROOT:-/data/miniconda3}"
WORKERS="${WORKERS:-8}"

if [[ ! -f "$PROJECT_ROOT/scripts/download_assets.py" || ! -f "$PROJECT_ROOT/scripts/prepare_data.py" ]]; then
  echo "ERROR: $PROJECT_ROOT is not a lingbot-vla-robotwin checkout." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate lingbot-vla

# With no --tasks filter these commands target the complete official clean set.
python scripts/download_assets.py --asset clean --workers "$WORKERS"
python scripts/prepare_data.py \
  --output data/lerobot \
  --manifest data/clean_manifest.json \
  --norm-output data/norm_stats_clean.json \
  --val-episodes 5

python "$DELIVERY_ROOT/data/verify_clean_data.py" \
  --manifest data/clean_manifest.json \
  --norm-stats data/norm_stats_clean.json
