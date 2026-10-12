#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/train_e1_relative_ddp.yaml"
MIN_FREE_GIB=80

record_exit() {
    local code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$RUN_DIR/exit_code"
    printf '[%s] launcher exit_code=%s\n' "$(date -Is)" "$code"
}

if [[ "${1:-}" == --worker ]]; then
    RUN_DIR="${2:?missing run directory}"
    exec > "$RUN_DIR/train.log" 2>&1
    trap record_exit EXIT
    cd "$ROOT"
    printf '[%s] E1 worker pid=%s, cwd=%s\n' "$(date -Is)" "$$" "$PWD"
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
    source scripts/activate.sh lingbot-vla
    command -v torchrun >/dev/null
    printf 'python=%s\n' "$(command -v python)"
    printf 'torchrun=%s\n' "$(command -v torchrun)"
    python --version
    conda list --explicit > "$RUN_DIR/conda-explicit.txt"
    printf '[%s] conda activated; starting E1 DDP training\n' "$(date -Is)"
    : > "$RUN_DIR/launcher.ready"
    bash scripts/train_ddp.sh \
        --config configs/train_e1_relative_ddp.yaml \
        --output-dir "$RUN_DIR" \
        --max-steps 15000
    exit 0
fi

if [[ "${1:-}" != --check && "${1:-}" != --launch ]]; then
    printf 'Usage: E1_EVAL_PROTOCOL_FILE=/absolute/path/eval-protocol.json bash %s --check|--launch\n' "$0" >&2
    exit 2
fi

cd "$ROOT"
[[ -f "$CONFIG" ]] || { echo "Missing E1 config: $CONFIG" >&2; exit 1; }
[[ -f data/clean_manifest.json && -f data/clean_norm_stats_e1_relative_h50.json ]] || {
    echo 'Missing E1 clean manifest or relative-action norm.' >&2
    exit 1
}
[[ -z "$(git status --porcelain)" ]] || {
    echo 'Training repository is dirty; commit or record changes before a formal run.' >&2
    exit 1
}
[[ -n "${E1_EVAL_PROTOCOL_FILE:-}" && -s "$E1_EVAL_PROTOCOL_FILE" ]] || {
    echo 'Set E1_EVAL_PROTOCOL_FILE to the frozen, non-empty E0/E1 eval-protocol.json before formal launch.' >&2
    exit 1
}
[[ "$(basename "$E1_EVAL_PROTOCOL_FILE")" == eval-protocol.json ]] || {
    echo 'The frozen protocol file must be named eval-protocol.json.' >&2
    exit 1
}
/home/linux/anaconda3/envs/lingbot-vla/bin/python -m json.tool "$E1_EVAL_PROTOCOL_FILE" >/dev/null
command -v screen >/dev/null || { echo 'screen is unavailable.' >&2; exit 1; }
FREE_GIB="$(df -BG --output=avail /data | awk 'NR==2 {gsub(/G/, "", $1); print $1}')"
[[ "$FREE_GIB" =~ ^[0-9]+$ ]] || { echo 'Could not read /data free space.' >&2; exit 1; }
if (( FREE_GIB < MIN_FREE_GIB )); then
    printf '/data has %s GiB free; the measured final-only export plan requires at least %s GiB.\n' \
        "$FREE_GIB" "$MIN_FREE_GIB" >&2
    exit 1
fi
printf 'Preflight: source=%s, /data free=%s GiB, eval protocol SHA256=%s\n' \
    "$(git rev-parse HEAD)" "$FREE_GIB" "$(sha256sum "$E1_EVAL_PROTOCOL_FILE" | awk '{print $1}')"
printf 'Storage policy: keep each 500-step training.pt; export a complete FP32 HF model at 15000 only.\n'
printf 'Reference: F1 final-only run measured 58G; minimum free-space gate is 80 GiB.\n'
if [[ "$1" == --check ]]; then
    exit 0
fi

RUN_ID="e1_shuweilin_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/$RUN_ID"
mkdir -- "$RUN_DIR"  # Fail on a duplicate formal experiment ID.
{
    printf 'experiment_id=%s\nstarted_at=%s\nsource_commit=%s\nsource_dirty=false\n' \
        "$RUN_ID" "$(date -Is)" "$(git rev-parse HEAD)"
    printf 'free_gib_at_launch=%s\nstorage_policy=all_500_step_training_pt_final_fp32_export_only\n' "$FREE_GIB"
    printf 'eval_protocol_path=%s\n' "$E1_EVAL_PROTOCOL_FILE"
    sha256sum "$CONFIG" data/clean_manifest.json data/clean_norm_stats_e1_relative_h50.json \
        configs/robot_configs/robotwin_e1_relative.yaml "$E1_EVAL_PROTOCOL_FILE"
    printf 'command=bash scripts/train_ddp.sh --config configs/train_e1_relative_ddp.yaml --output-dir %s --max-steps 15000\n' "$RUN_DIR"
} > "$RUN_DIR/launch_record.txt"

if ! screen -dmS "$RUN_ID" \
    env CONDA_ROOT=/home/linux/anaconda3 CUDA_VISIBLE_DEVICES=0,1 \
    bash "$ROOT/scripts/launch_e1_screen.sh" --worker "$RUN_DIR"; then
    echo 'screen failed to start; see the run directory.' >&2
    exit 1
fi

for ((attempt=0; attempt<30; attempt++)); do
    [[ -f "$RUN_DIR/launcher.ready" || -f "$RUN_DIR/exit_code" ]] && break
    sleep 1
done
if [[ -f "$RUN_DIR/exit_code" ]]; then
    echo 'E1 worker exited during startup:' >&2
    tail -n 50 "$RUN_DIR/train.log" >&2
    exit 1
fi
if [[ -f "$RUN_DIR/launcher.ready" ]] && screen -ls | grep -Fq ".$RUN_ID"; then
    printf 'E1 screen session is running: %s\nlog=%s/train.log\n' "$RUN_ID" "$RUN_DIR"
else
    echo 'E1 screen session did not pass startup verification; inspect train.log and exit_code.' >&2
    exit 1
fi
