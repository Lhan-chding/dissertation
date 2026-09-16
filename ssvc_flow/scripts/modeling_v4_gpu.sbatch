#!/bin/bash
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --time=3-00:00:00

# One independent whole-GPU worker. Its visible device ordinal is always zero.
set -euo pipefail
[[ "$#" -ge 4 ]] || { echo 'Expected PYTHON PROJECT_ROOT CAMPAIGN TASKS' >&2; exit 2; }
python_bin=$1 project_root=$2 campaign_root=$3 tasks=$4
shift 4
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate one visible whole GPU}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
[[ "$CUDA_VISIBLE_DEVICES" =~ ^([0-9]+|GPU-[0-9a-fA-F-]+)$ ]] || {
  echo 'Expected one whole GPU, without manually selecting a physical device' >&2; exit 2;
}
export SSVC_WORK=/projects/varunssd/louis-ssvc
export XDG_CACHE_HOME="$SSVC_WORK/cache" HF_HOME="$SSVC_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$campaign_root/tmp/$SLURM_JOB_ID" TMP="$campaign_root/tmp/$SLURM_JOB_ID"
export TEMP="$TMPDIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export SSVC_V4_SCRATCH_DIR="$SLURM_TMPDIR/ssvc_v4_directions"
  mkdir -p "$SSVC_V4_SCRATCH_DIR"
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$project_root"
exec /usr/bin/time -v "$python_bin" -m src.modeling_v4.cli run-worker \
  --tasks "$tasks" --device cuda:0 --execute-gpu "$@"
