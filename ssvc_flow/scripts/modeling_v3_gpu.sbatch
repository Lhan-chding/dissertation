#!/bin/bash
#SBATCH --job-name=ssvc-v3-q4
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:1
#SBATCH --ntasks=1
#SBATCH --no-requeue

# This is an explicit launch template. Neither importing V3 nor creating a plan
# submits this job. The first new GPU submission needs the user's authorization.
set -euo pipefail
: "${SSVC_V3_CHECKOUT:?Set the verified immutable V3 checkout}"
: "${SSVC_V3_PYTHON:?Set the verified existing Python}"
: "${SSVC_V3_BINDINGS:?Set the hash-bound V3 source and stage bindings}"
: "${SSVC_V3_OUTPUT:?Set the fresh Q4 output root}"
export SSVC_WORK=/projects/varunssd/louis-ssvc
export XDG_CACHE_HOME="$SSVC_WORK/cache" HF_HOME="$SSVC_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$SSVC_WORK/modeling_v3_20260915/tmp/$SLURM_JOB_ID"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$SSVC_V3_CHECKOUT/ssvc_flow"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
worker=${SLURM_ARRAY_TASK_ID:-0}
/usr/bin/time -v "$SSVC_V3_PYTHON" -m src.modeling_v3.cli vlm-smoke \
  --config configs/modeling_v3/protocol.json --bindings "$SSVC_V3_BINDINGS" \
  --device cuda:0 --out "$SSVC_V3_OUTPUT/worker_$worker" \
  --worker-index "$worker" \
  --allow-gpu --acknowledge-new-experiment
