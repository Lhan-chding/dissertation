#!/usr/bin/env bash
set -euo pipefail
if (( $# != 2 )); then
    echo 'Usage: run_training_pair.sh {R|G|D} {11031|11032}' >&2
    exit 2
fi
arm=$1
train_seed=$2
case "$arm" in R|G|D) ;; *) echo 'Unregistered training arm' >&2; exit 2 ;; esac
case "$train_seed" in 11031|11032) ;; *) echo 'Unregistered training seed' >&2; exit 2 ;; esac
root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
cd "$root"
[[ ${SLURM_MEM_PER_NODE:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_CPUS_ON_NODE:-} =~ ^[0-9]+$ ]] || exit 2
worker_mem_mb=$((SLURM_MEM_PER_NODE / 2))
worker_cpus=$((SLURM_CPUS_ON_NODE / 2))
((worker_mem_mb > 0 && worker_cpus > 0)) || exit 2
# Each worker declares its CPU/GPU resources; remove the conflicting inherited TRES option.
step=(env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$worker_cpus" --gpus-per-task=1 --mem="${worker_mem_mb}M")
"${step[@]}" "$python_bin" -B -m adapter.driver --binding execution/RUN_BINDING.json --parent b0 --mode TRAINING --arm "$arm" --train-seed "$train_seed" &
p0=$!
"${step[@]}" "$python_bin" -B -m adapter.driver --binding execution/RUN_BINDING.json --parent b1 --mode TRAINING --arm "$arm" --train-seed "$train_seed" &
p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
exit "$status"
