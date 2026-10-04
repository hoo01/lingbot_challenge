#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash train/resume.sh CONFIG TRAINING_PT [extra train.py arguments...]" >&2
  exit 2
fi

CONFIG="$1"
CHECKPOINT="$2"
shift 2

DELIVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

bash "$DELIVERY_ROOT/train/train.sh" "$CONFIG" --resume "$CHECKPOINT" "$@"
