#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-T03
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=66G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
fail() { echo "T03 launcher: $*" >&2; exit 2; }
canonical_directory() {
  [[ -d "$1" && ! -L "$1" ]] || fail "not a plain directory: $1"
  [[ "$(cd -P -- "$1" && pwd -P)" == "$1" ]] || fail "noncanonical directory: $1"
}
make_directory() {
  [[ -e "$1" || -L "$1" ]] || mkdir -- "$1" || exit 2
  canonical_directory "$1"
}
: "${SRF6_CODE:?immutable consumer source required}"
: "${SRF6_PRODUCER:?immutable producer source required}"
: "${SRF6_ROOT:?registered followup root required}"
: "${SLURM_JOB_ID:?scheduler job identity required}"
: "${SLURM_JOB_NAME:?registered job name required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || fail 'unsafe scheduler job id'
[[ "$SLURM_JOB_NAME" == LH-SRF6-T03 ]] || fail 'wrong job prefix'
[[ "$SRF6_ROOT" == "$project/sr_f6_confirm_followup_v2_20261029" ]] || fail 'wrong run root'
[[ "$SRF6_PRODUCER" == "$project/sr_f6_code_6e5d363" ]] || fail 'wrong scientific producer'
[[ "$SRF6_CODE" =~ ^/projects/_ssd/varunssd/louis-ssvc/sr_f6_followup_code_[0-9a-f]+$ ]] || fail 'wrong consumer deployment'
canonical_directory "$project"
canonical_directory "$SRF6_CODE"
canonical_directory "$SRF6_PRODUCER"
canonical_directory "$SRF6_ROOT"
cd "$SRF6_ROOT" || exit 2
export PYTHONPATH="$SRF6_PRODUCER/src:$SRF6_CODE/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset BASH_ENV ENV
make_directory "$SRF6_ROOT/execution_cache"
make_directory "$SRF6_ROOT/execution_cache/$SLURM_JOB_ID"
python="$project/envs/ssvc-py312/bin/python3.12"
run_parent() {
  local parent="$1"
  local cache="$SRF6_ROOT/execution_cache/$SLURM_JOB_ID/$parent"
  make_directory "$cache"
  make_directory "$cache/tmp"
  export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
  export XDG_CACHE_HOME="$cache/xdg" HF_HOME="$cache/huggingface"
  export HF_HUB_CACHE="$cache/huggingface/hub" HUGGINGFACE_HUB_CACHE="$cache/huggingface/hub"
  export TRANSFORMERS_CACHE="$cache/huggingface/hub" HF_MODULES_CACHE="$cache/huggingface/modules"
  export HF_DATASETS_CACHE="$cache/huggingface/datasets" HF_ASSETS_CACHE="$cache/huggingface/assets"
  export TORCH_HOME="$cache/torch" TORCH_EXTENSIONS_DIR="$cache/torch-extensions"
  export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton" CUDA_CACHE_PATH="$cache/cuda"
  srun --exclusive --ntasks=1 --gres=gpu:pro6000:1 --cpus-per-task=4 --mem=33G \
    "$python" -u -m sr_f6_followup.t03 run --root "$SRF6_ROOT" --parent "$parent" \
    --task-id "LH-SRF6-T03-$SLURM_JOB_ID-$parent"
}
lane0() {
  run_parent b0 || return 1
  run_parent b2 || return 1
}
lane0 &
pid0=$!
run_parent b1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
exit "$status"
