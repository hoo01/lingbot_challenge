#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  bash train/h0/scripts/launch_h0.sh --check
  bash train/h0/scripts/launch_h0.sh --dry-run
  bash train/h0/scripts/launch_h0.sh --state-test [OUTPUT_DIR]
  bash train/h0/scripts/launch_h0.sh --smoke [STEPS]
  bash train/h0/scripts/launch_h0.sh --resume-test [OUTPUT_DIR]
  bash train/h0/scripts/launch_h0.sh --launch
  bash train/h0/scripts/launch_h0.sh --resume CHECKPOINT_DIR
  bash train/h0/scripts/launch_h0.sh --continue-c1 H0_A_STEP_15000_DCP
  bash train/h0/scripts/launch_h0.sh --resume-c1 H0_C1_CHECKPOINT_DIR
  bash train/h0/scripts/launch_h0.sh --continue-c2 H0_C1_STEP_25000_DCP
  bash train/h0/scripts/launch_h0.sh --resume-c2 H0_C2_CHECKPOINT_DIR

Optional environment:
  TRAIN_ROOT             Runtime root (default: <challenge>/training/lingbot-vla-robotwin)
  H0_NORM_PROVENANCE     Norm provenance (default: <runtime>/data/clean_norm_stats.provenance.json)
  H0_OUTPUT_DIR          Explicit fresh run directory for --smoke or --launch
  H0_RUN_ID              Fresh run name when H0_OUTPUT_DIR is unset
  H0_SMOKE_RUN           Completed 100-step smoke dir; lets --resume-test write the final timing budget
EOF
    exit 2
}

SELF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHALLENGE_ROOT="$(cd "$SELF_ROOT/../.." && pwd)"
TRAIN_ROOT="${TRAIN_ROOT:-${CHALLENGE_ROOT}/training/lingbot-vla-robotwin}"
H0_NORM_PROVENANCE="${H0_NORM_PROVENANCE:-${TRAIN_ROOT}/data/clean_norm_stats.provenance.json}"
CONFIG_A="$SELF_ROOT/configs/train_h0_full_expert_2gpu.yaml"
CONFIG_C1="$SELF_ROOT/configs/train_h0_c1_2gpu.yaml"
CONFIG_C2="$SELF_ROOT/configs/train_h0_c2_2gpu.yaml"
CONFIG="$CONFIG_A"
PREFLIGHT="$SELF_ROOT/scripts/preflight_h0.py"
BUDGET_SCRIPT="$SELF_ROOT/scripts/plan_h0_timing.py"
WORLD_SIZE=2

[[ $# -ge 1 && $# -le 2 ]] || usage
MODE="$1"
ARGUMENT="${2:-}"
case "$MODE" in
    --check|--dry-run|--launch) [[ -z "$ARGUMENT" ]] || usage ;;
    --smoke) [[ -z "$ARGUMENT" || "$ARGUMENT" =~ ^[1-9][0-9]*$ ]] || usage ;;
    --state-test|--resume-test) ;;
    --resume|--continue-c1|--resume-c1|--continue-c2|--resume-c2) [[ -n "$ARGUMENT" ]] || usage ;;
    *) usage ;;
esac

PHASE="h0-a"
EXPECTED_PARENT_STEP=""
EXPECTED_PARENT_PHASE=""
IN_PLACE_RESUME=false
case "$MODE" in
    --continue-c1) PHASE="h0-c1"; CONFIG="$CONFIG_C1"; EXPECTED_PARENT_STEP=15000; EXPECTED_PARENT_PHASE="h0-a" ;;
    --resume-c1) PHASE="h0-c1"; CONFIG="$CONFIG_C1"; EXPECTED_PARENT_PHASE="h0-c1"; IN_PLACE_RESUME=true ;;
    --continue-c2) PHASE="h0-c2"; CONFIG="$CONFIG_C2"; EXPECTED_PARENT_STEP=25000; EXPECTED_PARENT_PHASE="h0-c1" ;;
    --resume-c2) PHASE="h0-c2"; CONFIG="$CONFIG_C2"; EXPECTED_PARENT_PHASE="h0-c2"; IN_PLACE_RESUME=true ;;
    --resume) EXPECTED_PARENT_PHASE="h0-a"; IN_PLACE_RESUME=true ;;
esac

[[ -d "$TRAIN_ROOT" ]] || { echo "Training runtime missing: $TRAIN_ROOT" >&2; exit 1; }
[[ -f "$TRAIN_ROOT/scripts/activate.sh" ]] || { echo "Activation script missing under $TRAIN_ROOT" >&2; exit 1; }

cd "$TRAIN_ROOT"
source scripts/activate.sh lingbot-vla

GIT_ROOT="$(git -C "$TRAIN_ROOT" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Training runtime is not inside a Git checkout: $TRAIN_ROOT" >&2
    exit 1
}
[[ -z "$(git -C "$GIT_ROOT" status --porcelain --untracked-files=no)" ]] || {
    echo "Tracked files are dirty. Commit the exact source before any H0/G2 run." >&2
    exit 1
}

if [[ "$MODE" != --check && "$MODE" != --dry-run ]]; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    [[ "$GPU_COUNT" -eq "$WORLD_SIZE" ]] || {
        echo "H0/G2 currently requires exactly two visible GPUs; found $GPU_COUNT." >&2
        exit 1
    }
fi

SOURCE_COMMIT="$(git -C "$GIT_ROOT" rev-parse HEAD)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
PYTHON="${CONDA_PREFIX}/bin/python"
PREFLIGHT_RECEIPT=""
PLANNED_RUN_DIR=""
CHECKPOINT=""

cleanup() {
    [[ -z "$PREFLIGHT_RECEIPT" ]] || rm -f -- "$PREFLIGHT_RECEIPT"
}
trap cleanup EXIT

if [[ "$MODE" == --state-test ]]; then
    STATE_DIR="${ARGUMENT:-/data/tmp/h0_fsdp_state_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
    [[ ! -e "$STATE_DIR" ]] || { echo "State-test output already exists: $STATE_DIR" >&2; exit 1; }
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/test_g2_fsdp_state.py --phase save --output "$STATE_DIR"
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/test_g2_fsdp_state.py --phase resume --output "$STATE_DIR"
    printf 'Two-rank toy DCP save/resume test passed: %s\n' "$STATE_DIR"
    exit 0
fi

case "$MODE" in
    --check)
        PLANNED_RUN_DIR="${H0_OUTPUT_DIR:-${TRAIN_ROOT}/runs}"
        ;;
    --dry-run|--launch)
        RUN_ID="${H0_RUN_ID:-h0_full_expert_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
        PLANNED_RUN_DIR="${H0_OUTPUT_DIR:-${TRAIN_ROOT}/runs/${RUN_ID}}"
        ;;
    --smoke)
        RUN_ID="${H0_RUN_ID:-h0_fsdp_smoke_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
        PLANNED_RUN_DIR="${H0_OUTPUT_DIR:-${TRAIN_ROOT}/runs/${RUN_ID}}"
        ;;
    --resume-test)
        RUN_ID="${H0_RUN_ID:-h0_fsdp_resume_test_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
        PLANNED_RUN_DIR="${ARGUMENT:-${H0_OUTPUT_DIR:-${TRAIN_ROOT}/runs/${RUN_ID}}}"
        ;;
    --resume|--resume-c1|--resume-c2)
        [[ -d "$ARGUMENT" ]] || { echo "DCP checkpoint directory missing: $ARGUMENT" >&2; exit 1; }
        CHECKPOINT="$(cd "$ARGUMENT" && pwd)"
        [[ -f "$CHECKPOINT/COMPLETED.json" ]] || { echo "Incomplete DCP checkpoint: $CHECKPOINT" >&2; exit 1; }
        PLANNED_RUN_DIR="$(cd "$CHECKPOINT/../.." && pwd)"
        ;;
    --continue-c1|--continue-c2)
        [[ -d "$ARGUMENT" ]] || { echo "Parent DCP checkpoint directory missing: $ARGUMENT" >&2; exit 1; }
        CHECKPOINT="$(cd "$ARGUMENT" && pwd)"
        [[ -f "$CHECKPOINT/COMPLETED.json" ]] || { echo "Incomplete parent DCP checkpoint: $CHECKPOINT" >&2; exit 1; }
        RUN_ID="${H0_RUN_ID:-${PHASE}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
        PLANNED_RUN_DIR="${H0_OUTPUT_DIR:-${TRAIN_ROOT}/runs/${RUN_ID}}"
        ;;
esac

PREFLIGHT_ARGS=(
    --training-root "$TRAIN_ROOT"
    --config "$CONFIG"
    --norm-provenance "$H0_NORM_PROVENANCE"
    --world-size 2
    --output-dir "$PLANNED_RUN_DIR"
    --phase "$PHASE"
)
if [[ -n "$CHECKPOINT" ]]; then
    PREFLIGHT_ARGS+=(--parent-checkpoint "$CHECKPOINT")
fi
if [[ -n "$EXPECTED_PARENT_STEP" ]]; then
    PREFLIGHT_ARGS+=(--expected-parent-step "$EXPECTED_PARENT_STEP")
fi
if [[ -n "$EXPECTED_PARENT_PHASE" ]]; then
    PREFLIGHT_ARGS+=(--expected-parent-phase "$EXPECTED_PARENT_PHASE")
fi
if [[ "$MODE" != --check && "$MODE" != --dry-run ]]; then
    PREFLIGHT_RECEIPT="$(mktemp "${TMPDIR:-/tmp}/h0-preflight.XXXXXX")"
    PREFLIGHT_ARGS+=(--receipt "$PREFLIGHT_RECEIPT")
fi
"$PYTHON" "$PREFLIGHT" "${PREFLIGHT_ARGS[@]}"
if [[ -n "$PREFLIGHT_RECEIPT" ]]; then
    [[ -s "$PREFLIGHT_RECEIPT" ]] || {
        echo "Preflight did not produce a non-empty receipt: $PREFLIGHT_RECEIPT" >&2
        exit 1
    }
fi

CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
printf 'Preflight complete: source=%s config_sha=%s world_size=2\n' "$SOURCE_COMMIT" "$CONFIG_SHA"

[[ "$MODE" != --check ]] || exit 0

if [[ "$MODE" == --dry-run ]]; then
    RUN_DIR="$PLANNED_RUN_DIR"
    [[ ! -e "$RUN_DIR" ]] || { echo "Formal output already exists: $RUN_DIR" >&2; exit 1; }
    COMMAND=(
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2
        scripts/train.py --config "$CONFIG" --output-dir "$RUN_DIR"
    )
    printf 'DRY RUN PASSED: no files created and no training started.\n'
    printf 'source_commit=%s\nrun_dir=%s\ncommand=' "$SOURCE_COMMIT" "$RUN_DIR"
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

if [[ "$MODE" == --smoke ]]; then
    STEPS="${ARGUMENT:-10}"
    RUN_DIR="$PLANNED_RUN_DIR"
    [[ ! -e "$RUN_DIR" ]] || { echo "Smoke output already exists: $RUN_DIR" >&2; exit 1; }
    mkdir -p "$RUN_DIR"
    mv -- "$PREFLIGHT_RECEIPT" "$RUN_DIR/preflight_receipt.json"
    PREFLIGHT_RECEIPT=""
    set +e
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/train.py \
        --config configs/train_g2_fsdp_smoke.yaml \
        --output-dir "$RUN_DIR" \
        --max-steps "$STEPS" 2>&1 | tee "$RUN_DIR/train.log"
    SMOKE_STATUS="${PIPESTATUS[0]}"
    set -e
    if [[ "$SMOKE_STATUS" -eq 0 && "$STEPS" -ge 100 ]]; then
        "$PYTHON" "$BUDGET_SCRIPT" \
            --metrics "$RUN_DIR/metrics.jsonl" \
            --validation-scale 4 \
            --output "$RUN_DIR/h0_timing_budget_initial.json"
    fi
    exit "$SMOKE_STATUS"
fi

if [[ "$MODE" == --resume-test ]]; then
    RUN_DIR="$PLANNED_RUN_DIR"
    [[ ! -e "$RUN_DIR" ]] || { echo "Resume-test output already exists: $RUN_DIR" >&2; exit 1; }
    mkdir -p "$RUN_DIR"
    mv -- "$PREFLIGHT_RECEIPT" "$RUN_DIR/preflight_receipt.json"
    PREFLIGHT_RECEIPT=""
    set +e
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/train.py \
        --config configs/train_g2_fsdp_state_test.yaml \
        --output-dir "$RUN_DIR" \
        --stop-after-step 1 \
        --no-export 2>&1 | tee "$RUN_DIR/phase1.log"
    PHASE1_STATUS="${PIPESTATUS[0]}"
    set -e
    [[ "$PHASE1_STATUS" -eq 0 ]] || exit "$PHASE1_STATUS"
    CHECKPOINT="$RUN_DIR/checkpoints/global_step_1"
    [[ -f "$CHECKPOINT/COMPLETED.json" ]] || {
        echo "Phase 1 did not create a complete checkpoint: $CHECKPOINT" >&2
        exit 1
    }
    set +e
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/train.py \
        --config configs/train_g2_fsdp_state_test.yaml \
        --output-dir "$RUN_DIR" \
        --resume "$CHECKPOINT" \
        --no-export 2>&1 | tee "$RUN_DIR/phase2.log"
    PHASE2_STATUS="${PIPESTATUS[0]}"
    set -e
    [[ "$PHASE2_STATUS" -eq 0 ]] || exit "$PHASE2_STATUS"
    [[ -f "$RUN_DIR/checkpoints/global_step_2/COMPLETED.json" ]] || {
        echo "Phase 2 did not create a complete checkpoint at step 2." >&2
        exit 1
    }
    if [[ -n "${H0_SMOKE_RUN:-}" ]]; then
        [[ -f "$H0_SMOKE_RUN/metrics.jsonl" ]] || {
            echo "H0_SMOKE_RUN lacks metrics.jsonl: $H0_SMOKE_RUN" >&2
            exit 1
        }
        "$PYTHON" "$BUDGET_SCRIPT" \
            --metrics "$H0_SMOKE_RUN/metrics.jsonl" \
            --checkpoint-metrics "$RUN_DIR/metrics.jsonl" \
            --validation-scale 4 \
            --output "$H0_SMOKE_RUN/h0_timing_budget.json"
    fi
    printf 'Real-model upstream FSDP2 save/resume test passed: %s\n' "$RUN_DIR"
    exit 0
fi

if [[ "$IN_PLACE_RESUME" == true ]]; then
    RUN_DIR="$PLANNED_RUN_DIR"
    LOG_PATH="$RUN_DIR/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
else
    RUN_DIR="$PLANNED_RUN_DIR"
    [[ ! -e "$RUN_DIR" ]] || { echo "Formal output already exists: $RUN_DIR" >&2; exit 1; }
    mkdir -p "$RUN_DIR"
    LOG_PATH="$RUN_DIR/train.log"
fi

if [[ "$IN_PLACE_RESUME" == true ]]; then
    RECEIPT_PATH="$RUN_DIR/preflight_resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N).json"
    [[ ! -e "$RECEIPT_PATH" ]] || { echo "Resume receipt already exists: $RECEIPT_PATH" >&2; exit 1; }
else
    RECEIPT_PATH="$RUN_DIR/preflight_receipt.json"
fi
mv -- "$PREFLIGHT_RECEIPT" "$RECEIPT_PATH"
PREFLIGHT_RECEIPT=""

{
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'source_commit=%s\nsource_dirty=false\n' "$SOURCE_COMMIT"
    printf 'phase=%s\nconfig=%s\nconfig_sha256=%s\nworld_size=2\n' "$PHASE" "$CONFIG" "$CONFIG_SHA"
    printf 'norm_provenance=%s\n' "$H0_NORM_PROVENANCE"
    printf 'preflight_receipt=%s\n' "$RECEIPT_PATH"
    [[ -z "$CHECKPOINT" ]] || printf 'resume=%s\n' "$CHECKPOINT"
} >> "$RUN_DIR/launch_record.txt"

COMMAND=(
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2
    scripts/train.py --config "$CONFIG" --output-dir "$RUN_DIR"
)
[[ -z "$CHECKPOINT" ]] || COMMAND+=(--resume "$CHECKPOINT")

set +e
"${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code"
exit "$STATUS"
