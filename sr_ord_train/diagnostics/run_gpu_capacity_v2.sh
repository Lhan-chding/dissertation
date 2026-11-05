#!/usr/bin/env bash
set -euo pipefail
(( $# == 1 )) || exit 2
path_name=$1
case "$path_name" in b0-11031-G) ;; *) exit 2 ;; esac
root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_MEM_PER_NODE:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_CPUS_ON_NODE:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_NTASKS:-} == 1 && ${SLURM_NNODES:-} == 1 ]] || exit 2
(( SLURM_MEM_PER_NODE > 0 && SLURM_CPUS_ON_NODE > 0 )) || exit 2
cd "$root"
printf 'GPU-only original-batch diagnostic; no generation or Adam; granted CPU=%s memoryMiB=%s; CPU offload forbidden\n' "$SLURM_CPUS_ON_NODE" "$SLURM_MEM_PER_NODE"
env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_ON_NODE" --gpus-per-task=pro6000:2 --mem="${SLURM_MEM_PER_NODE}M" \
    "$python_bin" -B -m diagnostics.gpu_capacity_full_batch_v2 --binding execution/RUN_BINDING.json --path "$path_name" \
    --worker-memory-mib "$SLURM_MEM_PER_NODE" --output "run/gpu_capacity_v2_diag_${SLURM_JOB_ID}_${path_name}"
