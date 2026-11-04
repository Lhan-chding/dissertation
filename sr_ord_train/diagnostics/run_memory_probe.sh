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
[[ ${SLURM_NTASKS:-} == 1 ]] || exit 2
(( SLURM_MEM_PER_NODE >= 28672 )) || exit 2
cd "$root"
status=0
for path in b0-11031-R b1-11031-R b1-11031-G; do
    # Each registered diagnostic is invoked once; no sampling or training retry.
    env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 \
        --cpus-per-task="$SLURM_CPUS_ON_NODE" --gpus-per-task=1 --mem="${SLURM_MEM_PER_NODE}M" \
        "$python_bin" -B -m diagnostics.memory_equivalence \
        --binding execution/RUN_BINDING.json --path "$path" \
        --output "run/memory_diag_${SLURM_JOB_ID}_${path}" || status=1
done
exit "$status"
