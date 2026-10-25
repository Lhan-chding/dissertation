#!/bin/bash
#SBATCH --job-name=SRF5-CPU-LH
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
set -euo pipefail
: "${SRF5_CODE:?immutable source directory}"
: "${SRF5_REVISION:?committed revision}"
export PYTHONPATH="$SRF5_CODE/src:$SRF5_CODE"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
cd "$SRF5_CODE"
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -u -m sr_f5.prepare \
  --root /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025 \
  --revision "$SRF5_REVISION"
