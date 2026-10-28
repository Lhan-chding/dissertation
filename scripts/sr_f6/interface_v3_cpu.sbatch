#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-V3-CPU
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
: "${SRF6_CODE:?immutable source deployment required}"
: "${SRF6_ROOT:?registered screen root required}"
: "${SRF6_ACTION:?preflight or assess action required}"
: "${SRF6_STAGE:?registered stage required}"
: "${SLURM_JOB_ID:?scheduler allocation required}"
: "${SLURM_JOB_NAME:?LH-SRF6- name required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_JOB_NAME" == LH-SRF6-* ]]
[[ "$SRF6_ROOT" == "$project/sr_f6_interface_v3_20261028" ]]
[[ "$SRF6_CODE" =~ ^/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_[0-9a-f]+$ ]]
for directory in "$project" "$SRF6_CODE" "$SRF6_ROOT"; do
  [[ -d "$directory" && ! -L "$directory" ]]
  [[ "$(cd -P -- "$directory" && pwd -P)" == "$directory" ]]
done
[[ "$SRF6_STAGE" =~ ^[12]$ ]]
[[ "$SRF6_ACTION" == preflight || "$SRF6_ACTION" == assess ]]
stage_root="$SRF6_ROOT/STAGE$SRF6_STAGE"
[[ -d "$stage_root" && ! -L "$stage_root" ]]
[[ "$(cd -P -- "$stage_root" && pwd -P)" == "$stage_root" ]]
cache="$stage_root/cpu_cache_$SLURM_JOB_ID"
[[ ! -e "$cache" && ! -L "$cache" ]]
mkdir -- "$cache"
mkdir -- "$cache/tmp"
export PYTHONPATH="$SRF6_CODE/src" PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
export XDG_CACHE_HOME="$cache/xdg" HF_HOME="$cache/huggingface"
export HF_HUB_CACHE="$cache/huggingface/hub" HUGGINGFACE_HUB_CACHE="$cache/huggingface/hub"
export TRANSFORMERS_CACHE="$cache/huggingface/hub" HF_MODULES_CACHE="$cache/huggingface/modules"
export HF_DATASETS_CACHE="$cache/huggingface/datasets" HF_ASSETS_CACHE="$cache/huggingface/assets"
export TORCH_HOME="$cache/torch" TORCH_EXTENSIONS_DIR="$cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
export CUDA_CACHE_PATH="$cache/cuda"
unset BASH_ENV ENV
cd "$SRF6_CODE"
if [[ "$SRF6_ACTION" == preflight ]]; then
  exec "$project/envs/ssvc-py312/bin/python3.12" -u -m sr_f6.interface_v3_registration \
    preflight --root "$SRF6_ROOT"
fi
exec "$project/envs/ssvc-py312/bin/python3.12" -u -m sr_f6.interface_v3_screen \
  assess --root "$SRF6_ROOT" --stage "$SRF6_STAGE"
