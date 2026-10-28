#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-Confirm-CPU
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
: "${SRF6_CODE:?immutable confirmation deployment required}"
: "${SRF6_ROOT:?registered confirmation root required}"
: "${SRF6_ACTION:?prepare, preflight, or assess action required}"
: "${SLURM_JOB_ID:?scheduler allocation required}"
: "${SLURM_JOB_NAME:?LH-SRF6- name required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_JOB_NAME" == LH-SRF6-* ]] || exit 2
[[ "$SRF6_ROOT" == "$project/sr_f6_confirm_v3_20261028" ]] || exit 2
[[ "$SRF6_CODE" =~ ^/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_[0-9a-f]+$ ]] || exit 2
[[ "$SRF6_ACTION" == prepare || "$SRF6_ACTION" == preflight || "$SRF6_ACTION" == assess ]] || exit 2
operations="$project/sr_f6_operations_20261027/CONFIRM_V3_EXECUTION_20261028"
for directory in "$project" "$SRF6_CODE" "$operations"; do
  [[ -d "$directory" && ! -L "$directory" ]] || exit 2
  [[ "$(cd -P -- "$directory" && pwd -P)" == "$directory" ]] || exit 2
done
if [[ "$SRF6_ACTION" == prepare ]]; then
  : "${SRF6_REVISION:?full immutable confirmation revision required}"
  [[ "$SRF6_REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
  code_suffix="${SRF6_CODE##*/sr_f6_code_}"
  [[ "$SRF6_REVISION" == "$code_suffix"* ]] || exit 2
  [[ ! -e "$SRF6_ROOT" && ! -L "$SRF6_ROOT" ]] || exit 2
else
  [[ -d "$SRF6_ROOT" && ! -L "$SRF6_ROOT" ]] || exit 2
  [[ "$(cd -P -- "$SRF6_ROOT" && pwd -P)" == "$SRF6_ROOT" ]] || exit 2
  if [[ "$SRF6_ACTION" == preflight ]]; then
    [[ ! -e "$SRF6_ROOT/PREFLIGHT/PROCESSOR.json" ]] || exit 2
  fi
fi
cache="$operations/cpu_cache_$SLURM_JOB_ID"
[[ ! -e "$cache" && ! -L "$cache" ]] || exit 2
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
if [[ "$SRF6_ACTION" == prepare ]]; then
  "$project/envs/ssvc-py312/bin/python3.12" -u -m sr_f6.confirm_registration \
    register --root "$SRF6_ROOT" --revision "$SRF6_REVISION"
fi
if [[ "$SRF6_ACTION" == assess ]]; then
  exec "$project/envs/ssvc-py312/bin/python3.12" -u -m sr_f6.confirm_analysis \
    --root "$SRF6_ROOT"
fi
exec "$project/envs/ssvc-py312/bin/python3.12" -u -m sr_f6.confirm_registration \
  preflight --root "$SRF6_ROOT"
