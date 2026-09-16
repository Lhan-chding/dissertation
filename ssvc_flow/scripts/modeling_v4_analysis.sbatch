#!/bin/bash
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --time=3-00:00:00

# Large response fits and signature learning run only on allocated server CPUs.
set -euo pipefail
[[ "$#" -ge 4 ]] || { echo 'Expected PYTHON PROJECT_ROOT OUTPUT_ROOT ANALYSIS_COMMAND [ARGS]' >&2; exit 2; }
python_bin=$1 project_root=$2 output_root=$3 analysis_command=$4
shift 4
: "${SLURM_JOB_ID:?A server CPU Slurm allocation is required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
case "$analysis_command" in
  fit-response-map|export-calibration-labels|fit-signature-development|freeze-signature-models|predict-signature-models|phase-evidence|point-response-evidence|summarize) ;;
  *) echo 'This template only runs registered CPU analysis commands' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$output_root/tmp/$SLURM_JOB_ID"
export TMP="$TMPDIR" TEMP="$TMPDIR"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export SSVC_V4_ANALYSIS_SCRATCH="$SLURM_TMPDIR/ssvc_v4_analysis"
  mkdir -p "$SSVC_V4_ANALYSIS_SCRATCH"
fi
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS
mkdir -p "$TMPDIR"
cd "$project_root"
exec /usr/bin/time -v "$python_bin" -m src.modeling_v4.cli "$analysis_command" "$@"
