#!/usr/bin/env bash
# Slurm resource/account options are supplied by the verified submission command.
set -euo pipefail
[[ "$#" -ge 6 ]] || exit 2
python_bin=$1 project_root=$2 run_root=$3 runtime_file=$4 cache_root=$5 lane=$6
shift 6
[[ "$#" -ge 1 ]] || exit 2
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?A single allocated GPU is required}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^([0-9]+|GPU-[0-9a-fA-F-]+)$ ]] || exit 2
export XDG_CACHE_HOME="$cache_root" HF_HOME="$cache_root/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$cache_root/cuda" TRITON_CACHE_DIR="$cache_root/triton"
export TORCH_HOME="$cache_root/torch" TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TMPDIR="$run_root/tmp/$SLURM_JOB_ID" TMP="$run_root/tmp/$SLURM_JOB_ID" TEMP="$run_root/tmp/$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$project_root"
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv
first=1
exit_status=0
for checkpoint in "$@"; do
  if [[ "$first" -eq 1 ]]; then
    if "$python_bin" -m src.protocol_state_probes.cli worker --run "$run_root" --checkpoint "$checkpoint" --runtime "$runtime_file" --allow-gpu --resume --with-smoke --lane-id "$lane"; then :; else exit_status=1; fi
    first=0
  else
    if "$python_bin" -m src.protocol_state_probes.cli worker --run "$run_root" --checkpoint "$checkpoint" --runtime "$runtime_file" --allow-gpu --resume --lane-id "$lane"; then :; else exit_status=1; fi
  fi
done
exit "$exit_status"
