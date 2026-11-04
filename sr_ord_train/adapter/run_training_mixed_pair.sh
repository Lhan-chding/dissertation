#!/usr/bin/env bash
set -euo pipefail
if (( $# != 6 )); then
    echo 'Usage: run_training_mixed_pair.sh PARENT ARM SEED PARENT ARM SEED' >&2
    exit 2
fi
parent0=$1 arm0=$2 seed0=$3
parent1=$4 arm1=$5 seed1=$6
for parent in "$parent0" "$parent1"; do
    case "$parent" in b0|b1) ;; *) echo 'Unregistered parent' >&2; exit 2 ;; esac
done
for arm in "$arm0" "$arm1"; do
    case "$arm" in R|G|D) ;; *) echo 'Unregistered training arm' >&2; exit 2 ;; esac
done
for train_seed in "$seed0" "$seed1"; do
    case "$train_seed" in 11031|11032) ;; *) echo 'Unregistered training seed' >&2; exit 2 ;; esac
done
if [[ "$parent0/$arm0/$seed0" == "$parent1/$arm1/$seed1" ]]; then
    echo 'Duplicate training path' >&2
    exit 2
fi
root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$root"
[[ ${SLURM_MEM_PER_NODE:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_CPUS_ON_NODE:-} =~ ^[0-9]+$ ]] || exit 2
worker_mem_mb=$((SLURM_MEM_PER_NODE / 2))
worker_cpus=$((SLURM_CPUS_ON_NODE / 2))
((worker_mem_mb > 0 && worker_cpus > 0)) || exit 2
printf 'Training paths: %s/%s/%s %s/%s/%s; PYTORCH_ALLOC_CONF=%s; PYTORCH_CUDA_ALLOC_CONF=unset\n' "$parent0" "$arm0" "$seed0" "$parent1" "$arm1" "$seed1" "$PYTORCH_ALLOC_CONF"
# The bound driver verifies checkpoints and original attempts; no launcher retry loop.
step=(env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$worker_cpus" --gpus-per-task=1 --mem="${worker_mem_mb}M")
"${step[@]}" "$python_bin" -B -m adapter.driver --binding execution/RUN_BINDING.json --parent "$parent0" --mode TRAINING --arm "$arm0" --train-seed "$seed0" &
p0=$!
"${step[@]}" "$python_bin" -B -m adapter.driver --binding execution/RUN_BINDING.json --parent "$parent1" --mode TRAINING --arm "$arm1" --train-seed "$seed1" &
p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
exit "$status"
