#!/bin/bash
#SBATCH --job-name=ssvc-contrast-n4
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --time=04:00:00

set -euo pipefail
: "${SSVC_N4_ROOT:?Set the isolated server experiment root}"
: "${SSVC_N4_MODE:?Set preflight or execute}"
: "${SSVC_N4_PYTHON:?Set the verified existing Python executable}"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${SSVC_N4_ROOT}/tmp/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
cd "${SSVC_N4_ROOT}/checkout/ssvc_flow"
/usr/bin/time -v "${SSVC_N4_PYTHON}" -m src.modeling_contrast.server_cli \
  "${SSVC_N4_MODE}" --settings "${SSVC_N4_ROOT}/server_settings.json"
