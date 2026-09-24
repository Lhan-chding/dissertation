#!/usr/bin/env bash
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
[[ "$#" -eq 6 ]] || exit 2
python_bin=$1 project_root=$2 run_root=$3 cache_root=$4 task_file=$5 runtime_file=$6
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^([0-9]+|GPU-[0-9a-fA-F-]+)$ ]] || exit 2
export XDG_CACHE_HOME="$cache_root" HF_HOME="$cache_root/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$cache_root/cuda" TRITON_CACHE_DIR="$cache_root/triton"
export TORCH_HOME="$cache_root/torch" TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export MPLCONFIGDIR="$cache_root/matplotlib"
export TMPDIR="$run_root/tmp/$SLURM_JOB_ID" TMP="$run_root/tmp/$SLURM_JOB_ID" TEMP="$run_root/tmp/$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$project_root"
exec /usr/bin/time -v "$python_bin" -m src.prospective_selection.cli worker \
  --config "$run_root/protocol.json" --root "$run_root" --task "$task_file" \
  --runtime "$runtime_file" --allow-gpu
