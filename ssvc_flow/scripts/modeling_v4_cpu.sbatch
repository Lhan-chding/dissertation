#!/bin/bash
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --time=3-00:00:00

# Arguments are explicit; this template never submits another job.
set -euo pipefail
[[ "$#" -ge 5 ]] || { echo 'Expected PYTHON PROJECT_ROOT CAMPAIGN CONFIG REUSE' >&2; exit 2; }
python_bin=$1 project_root=$2 campaign_root=$3 config=$4 reuse=$5
shift 5
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export SSVC_WORK=/projects/varunssd/louis-ssvc
export XDG_CACHE_HOME="$SSVC_WORK/cache" HF_HOME="$SSVC_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$campaign_root/tmp/$SLURM_JOB_ID" TMP="$campaign_root/tmp/$SLURM_JOB_ID"
export TEMP="$TMPDIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$project_root"
exec /usr/bin/time -v "$python_bin" -m src.modeling_v4.cli cpu-diagnose \
  --config "$config" --reuse "$reuse" --out "$campaign_root/cpu" "$@"
