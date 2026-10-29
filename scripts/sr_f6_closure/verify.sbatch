#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-Closure-Repair
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
: "${SRF6_REVISION:?registered immutable new verifier revision required}"
: "${SLURM_JOB_ID:?actual CPU allocation required}"
[[ "$SRF6_REVISION" =~ ^[0-9a-f]{40}$ && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
[[ "$SLURM_JOB_NAME" == LH-SRF6-Closure-Repair ]] || exit 2
code="$project/sr_f6_closure_code_${SRF6_REVISION:0:7}"
ops="$project/sr_f6_operations_20261027/CLOSURE_REPAIR_V1_20261029"
root="$project/sr_f6_confirmation_closure_v1_20261029"
for directory in "$project" "$code" "$ops"; do
  [[ -d "$directory" && ! -L "$directory" ]] || exit 2
  [[ "$(cd -P -- "$directory" && pwd -P)" == "$directory" ]] || exit 2
done
[[ ! -e "$root" && ! -L "$root" ]] || exit 2
cache="$ops/cache_$SLURM_JOB_ID"
[[ ! -e "$cache" && ! -L "$cache" ]] || exit 2
mkdir -- "$cache" "$cache/tmp"
export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
export XDG_CACHE_HOME="$cache/xdg" TORCH_HOME="$cache/torch"
export HF_HOME="$cache/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH BASH_ENV ENV
cd "$code"
exec "$project/envs/ssvc-py312/bin/python3.12" -I -B -u \
  "$code/scripts/sr_f6_closure/bootstrap.py" --revision "$SRF6_REVISION"
