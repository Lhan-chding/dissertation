#!/bin/bash
#SBATCH --job-name=ssvc-modeling-cpu
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --time=00:45:00

# Explicitly submitted CPU regression job; never invoked by the modeling CLI.
set -euo pipefail
: "${MODELING_CHECKOUT:?Set an isolated checkout under the project storage}"
: "${MODELING_PYTHON:?Set the existing verified Python executable}"
: "${MODELING_RESULT:?Set a fresh output directory under the project storage}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd "${MODELING_CHECKOUT}/ssvc_flow"
# This legacy test reads every real split, including sealed confirm. The modeling
# qualification scope forbids opening that split; retain this explicit exclusion.
/usr/bin/time -v "${MODELING_PYTHON}" scripts/verify_followup_cpu.py --out "${MODELING_RESULT}" -- \
  tests --deselect=tests/test_audit_r0_remaining.py::test_cross_split_passes_generated_dataset
