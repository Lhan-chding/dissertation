#!/usr/bin/env bash
# Resource requests must be supplied using currently approved cluster resources.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
umask 077
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo 'A Slurm allocation is required.' >&2; exit 2; }
: "${SSVC_PYTHON:?Set verified Python executable}"
: "${SSVC_PROJECT:?Set project ssvc_flow directory}"
: "${SSVC_PLAN:?Set validated plan path}"
: "${SSVC_OUT:?Set fresh run directory or exact resume target}"
cd "$SSVC_PROJECT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
resume_args=()
if [[ ${SSVC_RESUME:-0} == 1 ]]; then resume_args=(--resume); fi
exec "$SSVC_PYTHON" -m src.followup_cli smoke \
  --validated-plan "$SSVC_PLAN" --out "$SSVC_OUT" --allow-gpu-execution "${resume_args[@]}"
