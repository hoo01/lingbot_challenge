#!/bin/bash
# Source this file: source scripts/activate.sh [lingbot-vla|robotwin-sim]
LRVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LRVLA_ROOT
export XDG_CACHE_HOME="${LRVLA_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_LEROBOT_HOME="${LRVLA_ROOT}/data/lerobot"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
LRVLA_ENV_NAME="${1:-lingbot-vla}"
if [[ "${LRVLA_ENV_NAME}" == robotwin-sim ]]; then
    export PYTHONPATH="${LRVLA_ROOT}/src:${LRVLA_ROOT}/vendor/RoboTwin:${LRVLA_ROOT}/vendor/RoboTwin/description/utils"
else
    export PYTHONPATH="${LRVLA_ROOT}/src:${LRVLA_ROOT}/vendor/lingbot-vla-v2${PYTHONPATH:+:${PYTHONPATH}}"
fi
export QWEN3VL_PATH="${QWEN3VL_PATH:-${LRVLA_ROOT}/models/Qwen3-VL-4B-Instruct}"
export QWEN3_PATH="${QWEN3VL_PATH}"
export CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
export CONDA_ENVS_PATH="${CONDA_ROOT}/envs"
LRVLA_ENV_PREFIX="${CONDA_ROOT}/envs/${LRVLA_ENV_NAME}"
[[ -x "${LRVLA_ENV_PREFIX}/bin/python" ]] || {
    printf 'Conda environment Python is missing: %s\n' "${LRVLA_ENV_PREFIX}/bin/python" >&2
    return 1
}
# Compiler activation hooks assume optional variables may be unset.
LRVLA_RESTORE_NOUNSET=0
[[ $- == *u* ]] && LRVLA_RESTORE_NOUNSET=1
set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${LRVLA_ENV_PREFIX}"
# A shell can inherit an identically named environment from another Conda
# installation.  Conda may then leave that installation first on PATH even
# though CONDA_PREFIX names this environment.  Put the requested interpreter
# first explicitly and fail closed if command resolution still disagrees.
export PATH="${LRVLA_ENV_PREFIX}/bin:${PATH}"
hash -r
LRVLA_RESOLVED_PYTHON="$(command -v python)"
[[ "${LRVLA_RESOLVED_PYTHON}" == "${LRVLA_ENV_PREFIX}/bin/python" ]] || {
    printf 'Activated Python mismatch: expected %s, resolved %s\n' \
        "${LRVLA_ENV_PREFIX}/bin/python" "${LRVLA_RESOLVED_PYTHON}" >&2
    return 1
}
[[ "${CONDA_PREFIX}" == "${LRVLA_ENV_PREFIX}" ]] || {
    printf 'Activated Conda prefix mismatch: expected %s, got %s\n' \
        "${LRVLA_ENV_PREFIX}" "${CONDA_PREFIX}" >&2
    return 1
}
if [[ "${LRVLA_ENV_NAME}" == robotwin-sim ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
[[ "${LRVLA_RESTORE_NOUNSET}" == 1 ]] && set -u
unset LRVLA_RESTORE_NOUNSET
"${CONDA_PREFIX}/bin/python" "${LRVLA_ROOT}/scripts/check_wsl_tools.py" \
    "python=${CONDA_PREFIX}/bin/python" git=/usr/bin/git curl=/usr/bin/curl >/dev/null || return 1
unset LRVLA_ENV_NAME LRVLA_ENV_PREFIX LRVLA_RESOLVED_PYTHON
