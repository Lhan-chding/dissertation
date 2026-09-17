#!/bin/bash
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

# Read completed originals; the analysis checkout is separate from collection.
set -euo pipefail
[[ "$#" -ge 6 ]] || { echo 'Expected PYTHON ANALYSIS_ROOT TASKS COLLECTION_ROOT OUTPUT TASK_ID...'; exit 2; }
python_bin=$1 analysis_root=$2 tasks=$3 collection_root=$4 output=$5
shift 5
: "${SLURM_JOB_ID:?Server CPU allocation required}"
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || {
  echo 'CPU allocation required' >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
task_args=()
for task_id in "$@"; do task_args+=(--task-id "$task_id"); done
cd "$analysis_root"
exec "$python_bin" -m src.modeling_v4.measurement_report \
  --tasks "$tasks" --collection-code "$collection_root" --out "$output" "${task_args[@]}"
