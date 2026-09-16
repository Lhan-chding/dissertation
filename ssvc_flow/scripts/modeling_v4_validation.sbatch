#!/bin/bash
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --time=3-00:00:00

# Run the repository regression once on the final immutable campaign checkout.
# This is CPU validation, never a GPU experiment or a scientific success gate.
set -euo pipefail
[[ "$#" -eq 3 ]] || { echo 'Expected PYTHON PROJECT_ROOT OUTPUT_ROOT' >&2; exit 2; }
python_bin=$1 project_root=$2 output_root=$3
: "${SLURM_JOB_ID:?A server CPU Slurm allocation is required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$output_root/tmp/$SLURM_JOB_ID"
export TMP="$TMPDIR" TEMP="$TMPDIR"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS
mkdir -p "$output_root" "$TMPDIR"
cd "$project_root"
# Legacy tests import sibling fixture helpers by their bare module name.
/usr/bin/time -v "$python_bin" -m pytest tests --import-mode=importlib \
  -o 'pythonpath=tests tests/modeling_v3' --cov=src --cov-report=term-missing \
  --junitxml="$output_root/pytest.xml" -q
"$python_bin" -m ruff check src/modeling_v4 tests/modeling_v4 scripts/submit_modeling_v4.py
"$python_bin" -m compileall -q src/modeling_v4 scripts/submit_modeling_v4.py
bash -n scripts/modeling_v4_cpu.sbatch scripts/modeling_v4_gpu.sbatch scripts/modeling_v4_analysis.sbatch
