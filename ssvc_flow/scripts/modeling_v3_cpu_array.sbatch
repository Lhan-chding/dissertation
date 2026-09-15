#!/bin/bash
#SBATCH --job-name=ssvc-v3-cpu-array
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --no-requeue

# Supply the frozen zero-based --array range and concurrency when submitting.
# Freeze disjoint seed scopes and distinct outputs in requests before submission.
set -euo pipefail
: "${SSVC_V3_CHECKOUT:?Set an immutable V3 checkout}"
: "${SSVC_V3_PYTHON:?Set the verified production Python}"
: "${SSVC_V3_REQUEST_LIST:?Set a frozen ordered CPU request list}"
: "${SSVC_V3_REQUEST_LIST_SHA256:?Set the exact list SHA256}"
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${SLURM_ARRAY_TASK_ID:?A Slurm array task is required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm array task index' >&2; exit 2; }
case "$(uname -s)" in
  Linux) ;;
  *) echo 'CPU arrays require Linux Slurm execution' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export SSVC_WORK=/projects/varunssd/louis-ssvc
export XDG_CACHE_HOME="$SSVC_WORK/cache" HF_HOME="$SSVC_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$SSVC_WORK/modeling_v3_20260915/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
cd "$SSVC_V3_CHECKOUT/ssvc_flow"
exec /usr/bin/time -v "$SSVC_V3_PYTHON" scripts/run_modeling_v3_cpu_array.py \
  --request-list "$SSVC_V3_REQUEST_LIST" --sha256 "$SSVC_V3_REQUEST_LIST_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID"
