#!/usr/bin/env bash
set -euo pipefail
if (( $# != 0 )); then
    echo 'Usage: run_b1_rg_recovery.sh (fixed b1 R/G seed 11031)' >&2
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
printf 'Recovery b1 R/G seed 11031; PYTORCH_ALLOC_CONF=%s; PYTORCH_CUDA_ALLOC_CONF=unset\n' "$PYTORCH_ALLOC_CONF"
# Reuse the bound driver and committed original trajectories without a retry loop.
step=(env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$worker_cpus" --gpus-per-task=1 --mem="${worker_mem_mb}M")
"${step[@]}" "$python_bin" -B -m adapter.driver --binding execution/RUN_BINDING.json --parent b1 --mode TRAINING --arm R --train-seed 11031 &
p0=$!
"${step[@]}" "$python_bin" -B -m adapter.driver --binding execution/RUN_BINDING.json --parent b1 --mode TRAINING --arm G --train-seed 11031 &
p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
exit "$status"
