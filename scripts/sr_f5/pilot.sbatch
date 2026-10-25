#!/bin/bash
#SBATCH --job-name=LH-SRF5-G3
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=11:30:00
set -euo pipefail
: "${SRF5_CODE:?immutable scientific source directory}"
: "${SRF5_REVISION:?committed frozen revision}"
: "${SRF5_ROOT:?registered experiment root}"
: "${SRF5_PAIR:?registered worker pair 0 or 1}"
export PYTHONPATH="$SRF5_CODE/src:$SRF5_CODE"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$SRF5_CODE"
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -u \
  -m sr_f5.pilot_pool --pair "$SRF5_PAIR" --code "$SRF5_CODE" \
  --root "$SRF5_ROOT" --revision "$SRF5_REVISION"
