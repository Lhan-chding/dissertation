#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-V3
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=66G
#SBATCH --time=06:00:00
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
fail() { echo "Screen launcher: $*" >&2; exit 2; }
canonical_directory() {
  [[ -d "$1" && ! -L "$1" ]] || fail "not a plain directory: $1"
  [[ "$(cd -P -- "$1" && pwd -P)" == "$1" ]] || fail "noncanonical directory: $1"
}
make_directory() {
  [[ -e "$1" || -L "$1" ]] || mkdir -- "$1"
  canonical_directory "$1"
}
: "${SRF6_CODE:?immutable deployed source required}"
: "${SRF6_ROOT:?registered interface-v3 root required}"
: "${SRF6_PARENT:?registered parent required}"
: "${SRF6_STAGE:?registered stage required}"
: "${SLURM_JOB_ID:?scheduler job identity required}"
: "${SLURM_JOB_NAME:?LH-SRF6- job name required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || fail 'unsafe scheduler job id'
[[ "$SLURM_JOB_NAME" == LH-SRF6-* ]] || fail 'wrong job prefix'
[[ "$SRF6_STAGE" =~ ^[12]$ ]] || fail 'unknown stage'
[[ "$SRF6_PARENT" =~ ^b[012]$ ]] || fail 'unknown parent'
[[ "$SRF6_ROOT" == "$project/sr_f6_interface_v3_20261028" ]] || fail 'wrong run root'
[[ "$SRF6_CODE" =~ ^/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_[0-9a-f]+$ ]] || fail 'source must be an immutable project deployment'
canonical_directory "$project"
canonical_directory "$SRF6_CODE"
canonical_directory "$SRF6_ROOT"
stage_root="$SRF6_ROOT/STAGE$SRF6_STAGE"
canonical_directory "$stage_root"
cd "$SRF6_CODE"
export PYTHONPATH="$SRF6_CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset BASH_ENV ENV
make_directory "$stage_root/execution_cache"
make_directory "$stage_root/execution_cache/$SLURM_JOB_ID"
python=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
launch_worker() {
  local shard="$1"
  local cache="$stage_root/execution_cache/$SLURM_JOB_ID/$shard"
  make_directory "$cache"
  make_directory "$cache/tmp"
  export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
  export XDG_CACHE_HOME="$cache/xdg" HF_HOME="$cache/huggingface"
  export HF_HUB_CACHE="$cache/huggingface/hub" HUGGINGFACE_HUB_CACHE="$cache/huggingface/hub"
  export TRANSFORMERS_CACHE="$cache/huggingface/hub" HF_MODULES_CACHE="$cache/huggingface/modules"
  export HF_DATASETS_CACHE="$cache/huggingface/datasets" HF_ASSETS_CACHE="$cache/huggingface/assets"
  export TORCH_HOME="$cache/torch" TORCH_EXTENSIONS_DIR="$cache/torch-extensions"
  export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
  export CUDA_CACHE_PATH="$cache/cuda"
  exec srun --exclusive --ntasks=1 --gres=gpu:pro6000:1 --cpus-per-task=4 --mem=33G \
    "$python" -u -m sr_f6.interface_v3_screen run --root "$SRF6_ROOT" --stage "$SRF6_STAGE" --parent "$SRF6_PARENT" \
    --shard "$shard" --task-id "LH-SRF6-V3-S$SRF6_STAGE-$SLURM_JOB_ID-$SRF6_PARENT-$shard"
}
launch_worker 0 &
pid0=$!
launch_worker 1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
exit "$status"
