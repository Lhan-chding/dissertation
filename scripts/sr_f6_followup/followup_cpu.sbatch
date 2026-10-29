#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-Followup-CPU
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
: "${SRF6_CODE:?immutable followup consumer deployment required}"
: "${SRF6_ROOT:?registered followup root required}"
: "${SRF6_ACTION:?explicit CPU action required}"
: "${SLURM_JOB_ID:?scheduler allocation required}"
: "${SLURM_JOB_NAME:?LH-SRF6- job name required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_JOB_NAME" == LH-SRF6-* ]] || exit 2
[[ "$SRF6_ROOT" == "$project/sr_f6_confirm_followup_v2_20261029" ]] || exit 2
[[ "$SRF6_CODE" =~ ^/projects/_ssd/varunssd/louis-ssvc/sr_f6_followup_code_[0-9a-f]+$ ]] || exit 2
producer="$project/sr_f6_code_6e5d363"
operations="$project/sr_f6_operations_20261027/CONFIRM_FOLLOWUP_V2_EXECUTION_20261029"
for directory in "$project" "$SRF6_CODE" "$producer" "$operations"; do
  [[ -d "$directory" && ! -L "$directory" ]] || exit 2
  [[ "$(cd -P -- "$directory" && pwd -P)" == "$directory" ]] || exit 2
done
case "$SRF6_ACTION" in
  register|repair-prepare)
    : "${SRF6_REVISION:?full consumer revision required}"
    [[ "$SRF6_REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
    suffix="${SRF6_CODE##*/sr_f6_followup_code_}"
    [[ "$SRF6_REVISION" == "$suffix"* ]] || exit 2
    [[ ! -e "$SRF6_ROOT" && ! -L "$SRF6_ROOT" ]] || exit 2
    ;;
  bind-support|review-package|review-seal|t03-freeze|admission-seal)
    [[ -d "$SRF6_ROOT" && ! -L "$SRF6_ROOT" ]] || exit 2
    [[ "$(cd -P -- "$SRF6_ROOT" && pwd -P)" == "$SRF6_ROOT" ]] || exit 2
    ;;
  *) exit 2 ;;
esac
cache="$operations/cpu_cache_$SLURM_JOB_ID"
[[ ! -e "$cache" && ! -L "$cache" ]] || exit 2
mkdir -- "$cache" "$cache/tmp"
export PYTHONPATH="$producer/src:$SRF6_CODE/src" PYTHONDONTWRITEBYTECODE=1
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
python="$project/envs/ssvc-py312/bin/python3.12"
case "$SRF6_ACTION" in
  repair-prepare)
    "$python" -m pytest -q -p no:cacheprovider --basetemp "$cache/pytest" tests/sr_f6/test_followup_review_contract.py
    exec "$python" -u -m sr_f6_followup.repair_prepare --root "$SRF6_ROOT" --revision "$SRF6_REVISION"
    ;;
  register) exec "$python" -u -m sr_f6_followup.source register --root "$SRF6_ROOT" --revision "$SRF6_REVISION" ;;
  bind-support) exec "$python" -u -m sr_f6_followup.source bind-support --root "$SRF6_ROOT" ;;
  review-package) exec "$python" -u -m sr_f6_followup.review_package write --root "$SRF6_ROOT" ;;
  review-seal) exec "$python" -u -m sr_f6_followup.review_audit seal --root "$SRF6_ROOT" ;;
  t03-freeze) exec "$python" -u -m sr_f6_followup.t03 freeze --root "$SRF6_ROOT" ;;
  admission-seal) exec "$python" -u -m sr_f6_followup.admission seal --root "$SRF6_ROOT" ;;
esac
