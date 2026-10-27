#!/usr/bin/env bash
# Operations-only wrapper. Bind separately; Python remains frozen d49e9ae.
# Requires an existing two-GPU allocation. Never submits or retries.
set -euo pipefail
umask 077
project='/projects/_ssd/varunssd/louis-ssvc'
operations="$project/sr_f6_operations_20261027"
fail() { echo "T03 launcher: $*" >&2; exit 2; }
canonical_directory() {
  [[ -d "$1" && ! -L "$1" ]] || fail "not a plain directory: $1"
  [[ "$(cd -P -- "$1" && pwd -P)" == "$1" ]] || fail "noncanonical directory: $1"
}
make_directory() {
  [[ -e "$1" || -L "$1" ]] || mkdir -- "$1"
  canonical_directory "$1"
}
: "${SLURM_JOB_ID:?existing two-GPU Slurm allocation required}"
: "${SLURM_JOB_NAME:?LH-SRF6- job name required}"
[[ "$SLURM_JOB_NAME" == LH-SRF6-* ]] || fail 'expected LH-SRF6- job prefix'
: "${SRF6_CODE:?explicit frozen d49e9ae deployment required}"
: "${SRF6_ROOT:?explicit frozen experiment root required}"
: "${SRF6_PHASE_IDENTITY_SHA256:?explicit PILOT identity hash required}"
: "${SRF6_T03_SPEC_SHA256:?explicit frozen T03 spec hash required}"
: "${SRF6_T03_ATTEMPT:?new unique T03 attempt required}"
[[ "$SRF6_CODE" == "$project/sr_f6_code_d49e9ae" ]] || fail 'only frozen d49e9ae Python is allowed'
[[ "$SRF6_ROOT" == "$project/sr_f6_semantic_factorial_20261027" ]] || fail 'wrong canonical experiment root'
[[ "$SRF6_PHASE_IDENTITY_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail 'invalid PILOT identity hash'
[[ "$SRF6_T03_SPEC_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail 'invalid T03 spec hash'
[[ "$SRF6_T03_ATTEMPT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || fail 'unsafe attempt ID'
canonical_directory "$project"
canonical_directory "$operations"
canonical_directory "$SRF6_CODE"
canonical_directory "$SRF6_ROOT"
launcher="${BASH_SOURCE[0]}"
[[ "$launcher" == "$operations/"* && -f "$launcher" && ! -L "$launcher" ]] || fail 'launcher must be separately bound inside operations'
canonical_directory "$(dirname -- "$launcher")"
export SRF6_T03_PYTHON="$project/envs/ssvc-py312/bin/python3.12"
canonical_directory "$(dirname -- "$SRF6_T03_PYTHON")"
[[ -x "$SRF6_T03_PYTHON" ]] || fail 'frozen Python entry is not executable'
make_directory "$operations/T03_LAUNCHES"
export SRF6_T03_WORKDIR="$operations/T03_LAUNCHES/$SRF6_T03_ATTEMPT"
[[ ! -e "$SRF6_T03_WORKDIR" && ! -L "$SRF6_T03_WORKDIR" ]] || fail 'attempt exists; preserve it and do not retry'
mkdir -- "$SRF6_T03_WORKDIR"
canonical_directory "$SRF6_T03_WORKDIR"
for parent in b0 b1 b2; do
  make_directory "$SRF6_T03_WORKDIR/$parent"
  make_directory "$SRF6_T03_WORKDIR/$parent/cache"
  make_directory "$SRF6_T03_WORKDIR/$parent/tmp"
done
export SRF6_CODE SRF6_ROOT SRF6_PHASE_IDENTITY_SHA256 SRF6_T03_SPEC_SHA256 SRF6_T03_ATTEMPT
export PYTHONPATH="$SRF6_CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset BASH_ENV ENV
cd "$SRF6_CODE"
# Each parent uses a new Python process and separate writable cache/tmp paths.
# The inherited model snapshot remains at its original absolute frozen path.
worker='
set -euo pipefail
active=""
stop_owned_python() {
  trap - TERM INT
  [[ -z "$active" ]] || kill -TERM "$active" 2>/dev/null || true
  [[ -z "$active" ]] || wait "$active" || true
  exit "$1"
}
trap "stop_owned_python 143" TERM
trap "stop_owned_python 130" INT
for parent in "$@"; do
  case "$parent" in b0|b1|b2) ;; *) exit 2 ;; esac
  cache="$SRF6_T03_WORKDIR/$parent/cache"
  export TMPDIR="$SRF6_T03_WORKDIR/$parent/tmp" TMP="$SRF6_T03_WORKDIR/$parent/tmp" TEMP="$SRF6_T03_WORKDIR/$parent/tmp"
  export XDG_CACHE_HOME="$cache/xdg" HF_HOME="$cache/huggingface"
  export HF_HUB_CACHE="$cache/huggingface/hub" HUGGINGFACE_HUB_CACHE="$cache/huggingface/hub"
  export TRANSFORMERS_CACHE="$cache/huggingface/hub" HF_MODULES_CACHE="$cache/huggingface/modules"
  export HF_DATASETS_CACHE="$cache/huggingface/datasets" HF_ASSETS_CACHE="$cache/huggingface/assets"
  export TORCH_HOME="$cache/torch" TORCH_EXTENSIONS_DIR="$cache/torch-extensions"
  export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton" CUDA_CACHE_PATH="$cache/cuda"
  "$SRF6_T03_PYTHON" -u -m sr_f6.numerical_replay run --root "$SRF6_ROOT" --phase-identity-sha256 "$SRF6_PHASE_IDENTITY_SHA256" --spec-sha256 "$SRF6_T03_SPEC_SHA256" --parent "$parent" --task-id "LH-SRF6-T03-$SRF6_T03_ATTEMPT-$parent" &
  active=$!
  status=0
  wait "$active" || status=$?
  active=""
  [[ "$status" == 0 ]] || exit "$status"
done
'
pid0=''
pid1=''
stop_owned_srun() {
  trap - TERM INT
  [[ -z "$pid0" ]] || kill -TERM "$pid0" 2>/dev/null || true
  [[ -z "$pid1" ]] || kill -TERM "$pid1" 2>/dev/null || true
  [[ -z "$pid0" ]] || wait "$pid0" || true
  [[ -z "$pid1" ]] || wait "$pid1" || true
  exit "$1"
}
trap 'stop_owned_srun 143' TERM
trap 'stop_owned_srun 130' INT
srun --exclusive --ntasks=1 --gres=gpu:pro6000:1 --cpus-per-task=4 --mem=33G /bin/bash -c "$worker" LH-SRF6-T03-worker0 b0 b2 &
pid0=$!
srun --exclusive --ntasks=1 --gres=gpu:pro6000:1 --cpus-per-task=4 --mem=33G /bin/bash -c "$worker" LH-SRF6-T03-worker1 b1 &
pid1=$!
# A failed lane never retries. The independent peer finishes its fixed queue.
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
exit "$status"
