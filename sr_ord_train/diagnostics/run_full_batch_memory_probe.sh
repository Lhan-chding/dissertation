#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_MEM_PER_NODE:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_CPUS_ON_NODE:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_NTASKS:-} == 2 ]] || exit 2
worker_mem_mib=$((SLURM_MEM_PER_NODE / 2))
worker_cpus=$((SLURM_CPUS_ON_NODE / 2))
(( worker_mem_mib >= 28672 && worker_cpus > 0 )) || exit 2
cd "$root"
run_lane() {
    local result=0
    local path
    for path in "$@"; do
        env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 \
            --cpus-per-task="$worker_cpus" --gpus-per-task=1 --mem="${worker_mem_mib}M" \
            "$python_bin" -B -m diagnostics.full_batch_memory_probe \
            --binding execution/RUN_BINDING.json --path "$path" \
            --worker-memory-mib "$worker_mem_mib" \
            --output "run/fullbatch_memory_diag_${SLURM_JOB_ID}_${path}" || result=1
    done
    return "$result"
}
# Four fixed failed paths, each invoked exactly once; no production retries.
run_lane b0-11031-R b0-11031-G &
p0=$!
run_lane b1-11031-R b1-11031-G &
p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
exit "$status"
