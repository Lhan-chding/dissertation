#!/usr/bin/env bash
set -euo pipefail
(( $# == 3 )) || exit 2
path_name=$1 execution_commit=$2 acceptance_path=$3
[[ "$path_name" =~ ^(b[01])-(11031|11032)-([RGD])$ ]] || exit 2
parent=${BASH_REMATCH[1]} train_seed=${BASH_REMATCH[2]} arm=${BASH_REMATCH[3]}
[[ "$execution_commit" =~ ^[a-f0-9]{40}$ ]] || exit 2
root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
[[ "$acceptance_path" == "$root"/run/* && -f "$acceptance_path" ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_MEM_PER_NODE:-} =~ ^[0-9]+$ && ${SLURM_CPUS_ON_NODE:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_NTASKS:-} == 1 && ${SLURM_NNODES:-} == 1 ]] || exit 2
(( SLURM_MEM_PER_NODE >= 28672 && SLURM_CPUS_ON_NODE > 0 )) || exit 2
cd "$root"
printf 'GPU-only capacity48 v2 production %s; execution version %s; original scientific budget 24 steps\n' "$path_name" "$execution_commit"
env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_ON_NODE" --gpus-per-task=pro6000:2 --mem="${SLURM_MEM_PER_NODE}M" \
    "$python_bin" -B -m gpu_production_v2.driver --binding "$root/execution/RUN_BINDING.json" \
    --parent "$parent" --mode TRAINING --arm "$arm" --train-seed "$train_seed" \
    --execution-code-commit "$execution_commit" --acceptance "$acceptance_path"
