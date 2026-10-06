#!/usr/bin/env bash
# One independent model on one Slurm-assigned PRO6000; resources supplied by sbatch.
set -euo pipefail
[[ "$#" -eq 5 ]] || exit 2
python_bin=$1 project_root=$2 run_root=$3 machine_file=$4 lane=$5
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?One allocated GPU required}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^([0-9]+|GPU-[0-9a-fA-F-]+)$ ]] || exit 2
cache_root=/projects/varunssd/louis-ssvc/cache
export XDG_CACHE_HOME="$cache_root" HF_HOME="$cache_root/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$cache_root/cuda" TRITON_CACHE_DIR="$cache_root/triton"
export TORCH_HOME="$cache_root/torch" TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TMPDIR="$run_root/tmp/$SLURM_JOB_ID" TMP="$run_root/tmp/$SLURM_JOB_ID" TEMP="$run_root/tmp/$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$project_root"
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv
if [[ "$lane" == bridge ]]; then
  "$python_bin" -m src.verified_discovery_transfer.cli bridge \
    --plan "$run_root/protocol.json" --machine "$machine_file" --out "$run_root"
else
  "$python_bin" -m src.verified_discovery_transfer.cli worker \
    --queue "$run_root/queue.sqlite" --worker-id "$lane" --device cuda:0
fi
