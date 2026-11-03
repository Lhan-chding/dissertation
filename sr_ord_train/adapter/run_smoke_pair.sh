#!/usr/bin/env bash
set -euo pipefail
root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
cd "$root"
[[ ${SLURM_MEM_PER_NODE:-} =~ ^[0-9]+$ ]]
[[ ${SLURM_CPUS_ON_NODE:-} =~ ^[0-9]+$ ]]
worker_mem_mb=$((SLURM_MEM_PER_NODE / 2))
worker_cpus=$((SLURM_CPUS_ON_NODE / 2))
((worker_mem_mb > 0 && worker_cpus > 0))
step=(srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$worker_cpus" --gpus-per-task=1 --mem="${worker_mem_mb}M")
"${step[@]}" "$python_bin" -B -m adapter.driver --binding execution/RUN_BINDING.json --parent b0 --mode DEV_SMOKE &
p0=$!
"${step[@]}" "$python_bin" -B -m adapter.driver --binding execution/RUN_BINDING.json --parent b1 --mode DEV_SMOKE &
p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
exit "$status"
