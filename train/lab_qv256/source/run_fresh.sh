#!/usr/bin/env bash
set -euo pipefail
source /mnt/raid1/lingbotvla_smoke_20261009/code/lab_teacher_smoke/env.sh
export PYTHONPATH="/mnt/raid1/lingbotvla_qv256_20261012/code:$ROOT/code/src:$ROOT/code/lingbot-vla-v2"
cd "$ROOT/code/lingbot-vla-v2"
exec torchrun --standalone --nnodes=1 --nproc_per_node=4 -- /mnt/raid1/lingbotvla_qv256_20261012/code/train_fresh.py "$@"
