#!/bin/bash
#SBATCH --job-name=ssvc-v3-q4-pair
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:2
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --no-requeue

# Explicit launch template only. Use an already confirmed immutable Q4 stage.
set -euo pipefail
: "${SSVC_V3_CHECKOUT:?Set the verified immutable V3 checkout}"
: "${SSVC_V3_PYTHON:?Set the verified existing Python}"
: "${SSVC_V3_BINDINGS:?Set the authorized Q4 source and stage bindings}"
: "${SSVC_V3_OUTPUT:?Set the Q4 pair output root}"
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate two visible whole GPUs}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != --resume ]; }; then
  echo 'Only an explicit --resume argument is supported' >&2
  exit 2
fi
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
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$SSVC_V3_CHECKOUT/ssvc_flow"
# Pass --resume explicitly after this script to retain and retry existing outputs.
exec /usr/bin/time -v "$SSVC_V3_PYTHON" scripts/run_modeling_v3_q4_pair.py \
  --config configs/modeling_v3/protocol.json --bindings "$SSVC_V3_BINDINGS" \
  --out "$SSVC_V3_OUTPUT" "$@"
