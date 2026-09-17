#!/bin/bash
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

# Analyze saved B-stage samples only. The standalone module is loaded alongside
# the frozen experiment code without changing the running GPU worker's source.
set -euo pipefail
[[ "$#" -eq 5 ]] || { echo 'Expected PYTHON PROJECT_ROOT MODULE CAMPAIGN OUTPUT'; exit 2; }
python_bin=$1 project_root=$2 analysis_module=$3 campaign=$4 output=$5
: "${SLURM_JOB_ID:?Server CPU allocation required}"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
cd "$project_root"
exec "$python_bin" -c '
import importlib.util, sys
path = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location("src.modeling_v4.bridge_analysis", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
' "$analysis_module" --campaign "$campaign" --out "$output" --bootstrap 1000
