#!/bin/bash
#SBATCH --job-name=LH-SRF4-CTRL
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
set -euo pipefail
SRF4_ROOT=/projects/_ssd/varunssd/louis-ssvc/sr_f4_c_local_pilot_20261024
SRF4_SOURCE=/projects/_ssd/varunssd/louis-ssvc/sr_f4_consolidated_20261022
SRF4_PYTHON=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
: "${SRF4_CODE:?Set the immutable deployed code directory}"
: "${SRF4_REVISION:?Set the committed local-credit pilot revision}"
cd "$SRF4_CODE"
export PYTHONPATH="$SRF4_CODE/src:$SRF4_CODE"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
# Preparation is real CPU data/rendering work in this allocation. A partial
# root fails closed and is diagnosed; it is never silently re-prepared.
if [ ! -f "$SRF4_ROOT/READY.json" ]; then
  "$SRF4_PYTHON" -u -m sr_f4.c_local --root "$SRF4_ROOT" \
    --source-root "$SRF4_SOURCE" --revision "$SRF4_REVISION"

fi
exec "$SRF4_PYTHON" -u -m sr_f4.pool controller --root "$SRF4_ROOT" \
  --code "$SRF4_CODE" --python "$SRF4_PYTHON" --interval 150 \
  --maximum-new-jobs 2 --gpus-per-job 4 --walltime 06:00:00
