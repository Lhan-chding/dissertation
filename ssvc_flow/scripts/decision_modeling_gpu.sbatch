#!/bin/bash
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue

set -euo pipefail
[[ "$#" -ge 5 ]] || { echo 'Expected PYTHON PROJECT_ROOT RUN_ROOT CACHE_ROOT COMMAND [ARGS...]' >&2; exit 2; }
python_bin=$1 project_root=$2 run_root=$3 cache_root=$4
shift 4
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?A GPU allocation is required}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^([0-9]+|GPU-[0-9a-fA-F-]+)$ ]] || exit 2
export XDG_CACHE_HOME="$cache_root" HF_HOME="$cache_root/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$cache_root/cuda" TRITON_CACHE_DIR="$cache_root/triton"
export TORCH_HOME="$cache_root/torch" TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export MPLCONFIGDIR="$cache_root/matplotlib"
export TMPDIR="$run_root/tmp/$SLURM_JOB_ID" TMP="$run_root/tmp/$SLURM_JOB_ID"
export TEMP="$TMPDIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$project_root"
printf 'DECISION_MODELING_START job=%s utc=%s\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)"
exec /usr/bin/time -v "$python_bin" -m src.decision_modeling.cli "$@"
