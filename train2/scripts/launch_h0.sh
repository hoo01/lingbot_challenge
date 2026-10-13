#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: TRAIN_ROOT=/data/lingbot-vla-robotwin H0_NORM_PROVENANCE=/path/provenance.json bash $0 --check|--launch 2|8" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage
MODE="$1"
WORLD_SIZE="$2"
[[ "$MODE" == --check || "$MODE" == --launch ]] || usage
[[ "$WORLD_SIZE" == 2 || "$WORLD_SIZE" == 8 ]] || usage

: "${TRAIN_ROOT:?Set TRAIN_ROOT to the actual training repository root}"
: "${H0_NORM_PROVENANCE:?Set H0_NORM_PROVENANCE to the Clean norm provenance JSON}"

SELF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$SELF_ROOT/configs/train_h0_full_expert_${WORLD_SIZE}gpu.yaml"
PREFLIGHT="$SELF_ROOT/scripts/preflight_h0.py"

[[ -d "$TRAIN_ROOT/.git" ]] || { echo "Not a git training repository: $TRAIN_ROOT" >&2; exit 1; }
[[ -x "$TRAIN_ROOT/scripts/train_ddp.sh" || -f "$TRAIN_ROOT/scripts/train_ddp.sh" ]] || {
    echo "Missing existing training entry: $TRAIN_ROOT/scripts/train_ddp.sh" >&2
    exit 1
}

cd "$TRAIN_ROOT"
[[ -z "$(git status --porcelain)" ]] || {
    echo "Training repository is dirty. Commit and record the exact source before a formal run." >&2
    exit 1
}

python "$PREFLIGHT" \
    --training-root "$TRAIN_ROOT" \
    --config "$CONFIG" \
    --norm-provenance "$H0_NORM_PROVENANCE" \
    --world-size "$WORLD_SIZE"

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$GPU_COUNT" -ge "$WORLD_SIZE" ]] || {
    echo "Found $GPU_COUNT GPUs, need $WORLD_SIZE." >&2
    exit 1
}

SOURCE_COMMIT="$(git rev-parse HEAD)"
CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
printf 'Preflight complete: source=%s config_sha=%s world_size=%s\n' "$SOURCE_COMMIT" "$CONFIG_SHA" "$WORLD_SIZE"

[[ "$MODE" == --launch ]] || exit 0

RUN_ID="h0_full_expert_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
RUN_DIR="$TRAIN_ROOT/runs/$RUN_ID"
mkdir -- "$RUN_DIR"

{
    printf 'experiment_id=%s\n' "$RUN_ID"
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'source_dirty=false\nworld_size=%s\n' "$WORLD_SIZE"
    printf 'config=%s\nconfig_sha256=%s\n' "$CONFIG" "$CONFIG_SHA"
    printf 'norm_provenance=%s\n' "$H0_NORM_PROVENANCE"
    sha256sum data/clean_manifest.json data/clean_norm_stats.json "$H0_NORM_PROVENANCE" "$SELF_ROOT/configs/robotwin_absolute.yaml"
    printf 'command=bash scripts/train_ddp.sh --config %s --output-dir %s --max-steps 15000\n' "$CONFIG" "$RUN_DIR"
} > "$RUN_DIR/launch_record.txt"

GPU_LIST="$(seq -s, 0 $((WORLD_SIZE - 1)))"
screen -dmS "$RUN_ID" bash -lc "
    set -uo pipefail
    cd '$TRAIN_ROOT'
    source scripts/activate.sh lingbot-vla
    export CUDA_VISIBLE_DEVICES='$GPU_LIST'
    bash scripts/train_ddp.sh --config '$CONFIG' --output-dir '$RUN_DIR' --max-steps 15000 > '$RUN_DIR/train.log' 2>&1
    code=\$?
    printf '%s\n' \"\$code\" > '$RUN_DIR/exit_code'
    exit \"\$code\"
"

sleep 3
if screen -ls | grep -Fq ".$RUN_ID"; then
    printf 'H0 launched: screen=%s\nlog=%s/train.log\n' "$RUN_ID" "$RUN_DIR"
else
    echo "H0 did not remain alive. Inspect $RUN_DIR/train.log" >&2
    exit 1
fi
