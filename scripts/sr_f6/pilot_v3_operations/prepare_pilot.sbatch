#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-PilotV3-Prepare
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
code="$project/sr_f6_code_7631725"
ops="$project/sr_f6_operations_20261027/PILOT_V3_EXECUTION_20261029"
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ && "${SLURM_JOB_NAME:?}" == LH-SRF6-PilotV3-Prepare ]] || exit 2
for directory in "$project" "$code" "$ops"; do
  [[ -d "$directory" && ! -L "$directory" ]] || exit 2
  [[ "$(cd -P -- "$directory" && pwd -P)" == "$directory" ]] || exit 2
done
cache="$ops/cpu_cache_$SLURM_JOB_ID"
[[ ! -e "$cache" && ! -L "$cache" ]] || exit 2
mkdir -- "$cache" "$cache/tmp"
export PYTHONPATH="$code/src:$code" PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
export XDG_CACHE_HOME="$cache/xdg" HF_HOME="$cache/huggingface"
export HF_HUB_CACHE="$cache/huggingface/hub" HUGGINGFACE_HUB_CACHE="$cache/huggingface/hub"
export TRANSFORMERS_CACHE="$cache/huggingface/hub" HF_MODULES_CACHE="$cache/huggingface/modules"
export HF_DATASETS_CACHE="$cache/huggingface/datasets" HF_ASSETS_CACHE="$cache/huggingface/assets"
export TORCH_HOME="$cache/torch" TORCH_EXTENSIONS_DIR="$cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton" CUDA_CACHE_PATH="$cache/cuda"
unset BASH_ENV ENV
cd "$code"
exec "$project/envs/ssvc-py312/bin/python3.12" -B -u "$ops/prepare_pilot.py"
