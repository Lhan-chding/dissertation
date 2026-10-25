#!/bin/bash
#SBATCH --job-name=SRF5-G2-LH
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
: "${SRF5_CODE:?immutable source directory}"
export PYTHONPATH="$SRF5_CODE/src:$SRF5_CODE"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$SRF5_CODE"
# Four equal shards keep all workers useful across all three registered parents.
# Each process exits before the next parent is loaded, releasing its GPU memory.
for SRF5_CURRENT_PARENT in ${SRF5_PARENTS:-b0 b1 b2}; do
  /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -u -m sr_f5.calibration \
    --root /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025 \
    --parent "$SRF5_CURRENT_PARENT" --shard "${SRF5_SHARD:-0}" --shards "${SRF5_SHARDS:-4}"
done
