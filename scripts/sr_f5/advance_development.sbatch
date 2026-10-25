#!/bin/bash
#SBATCH --job-name=LH-SRF5-G2-DECISION
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SRF5_CODE:?immutable development source}"
: "${SRF5_REVISION:?committed revision}"
: "${SRF5_G2_RECEIPT:?actual census submission receipt filename}"
export PYTHONPATH="$SRF5_CODE/src:$SRF5_CODE"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
cd "$SRF5_CODE"
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -u \
  scripts/sr_f5/advance_development.py \
  --root /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025 \
  --code "$SRF5_CODE" --revision "$SRF5_REVISION" --receipt "$SRF5_G2_RECEIPT"
