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
#SBATCH --output=/projects/_ssd/varunssd/louis-ssvc/sr_f4_consolidated_20261022/pool/logs/controller-%j.log
set -euo pipefail
SRF4_ROOT=/projects/_ssd/varunssd/louis-ssvc/sr_f4_consolidated_20261022
SRF4_PYTHON=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
cd "$SRF4_ROOT/code"
export PYTHONPATH="$SRF4_ROOT/code/src:$SRF4_ROOT/code"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
exec "$SRF4_PYTHON" -u -m sr_f4.pool controller --root "$SRF4_ROOT" \
  --code "$SRF4_ROOT/code" --python "$SRF4_PYTHON" --interval 150 --walltime 06:00:00
