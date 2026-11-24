#!/usr/bin/env bash
set -euo pipefail

# Slurm executes a spool copy of the script. Pass the source directory explicitly.
exp_dir=${TIGER2_EXP1_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
scratch=${TIGER2_EXP1_SCRATCH:-/net/spaces/scratch/$USER}
model_python=${TIGER2_EXP1_MODEL_PYTHON:-$scratch/venvs/ideation-rm-py312/bin/python}
export PATH="$exp_dir/.tools:$HOME/.local/bin:$PATH"

case "$1" in
    score)
        export TIGER2_EXP1_RM_SNAPSHOT=${TIGER2_EXP1_RM_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--chailabideation--qwen3.8-27b-rm-terra-with-literature-2000-combined-no-coherence/snapshots/d515befe7b58ba617793c551b30ad959f723b580}
        test -f "$TIGER2_EXP1_RM_SNAPSHOT/reward_model_config.json"
        export HF_HUB_CACHE=${HF_HUB_CACHE:-$scratch/huggingface/hub}
        export PYTHONPATH="$TIGER2_EXP1_RM_SNAPSHOT:$TIGER2_EXP1_RM_SNAPSHOT/src:${TIGER2_EXP1_FLA_DEPS:-$scratch/rm_qwen38_terra_fast_20260910/python_deps}${PYTHONPATH:+:$PYTHONPATH}"
        exec "$model_python" "$exp_dir/pipeline.py" "$@"
        ;;
    generate)
        exec "$model_python" "$exp_dir/pipeline.py" "$@"
        ;;
    execute)
        exec python3 "$exp_dir/pipeline.py" "$@" --rank "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
        ;;
    *)
        echo "Expected generate, score, or execute" >&2
        exit 2
        ;;
esac
