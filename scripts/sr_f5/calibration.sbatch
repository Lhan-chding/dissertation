#!/bin/bash
#SBATCH --job-name=LH-SRF5-G2
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
set -euo pipefail
: "${SRF5_CODE:?immutable source directory}"
: "${SRF5_LAUNCHER_CODE:?versioned scheduling source directory}"
: "${SRF5_PAIR:?registered shard pair 0 or 1}"
export PYTHONPATH="$SRF5_CODE/src:$SRF5_CODE"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$SRF5_CODE"
# Launcher changes allocation only; each child imports the frozen science code.
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -u \
  "$SRF5_LAUNCHER_CODE/src/sr_f5/dual_gpu.py" --pair "$SRF5_PAIR" \
  --code "$SRF5_CODE" --root "${SRF5_ROOT:-/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025}"
