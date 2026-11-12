#!/usr/bin/env bash
set -euo pipefail
base=/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_NTASKS:-} == 1 && ${SLURM_NNODES:-} == 1 ]] || exit 2
actual_freeze=$(sha256sum "$base/DESIGN_FREEZE.json")
[[ ${actual_freeze%% *} == 6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0 ]] || exit 2
cd "$base/code"
mkdir -p "$base/runs"
for start_seed in 11031 11032; do
  env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --gpus-per-task=pro6000:2 --mem=66G \
    "$python_bin" -B -m runtime.dry_driver \
    --config "$base/code/supplied_package/config/experiment.json" \
    --evidence-root "$base" --freeze "$base/DESIGN_FREEZE.json" \
    --inputs "$base/inputs.jsonl" --images "$base/images.jsonl" \
    --source-lock "$base/runtime_source_lock.json" \
    --manifests "$base/train_manifest.jsonl" "$base/dry_manifest.jsonl" \
      "$base/dev_manifest.jsonl" "$base/prefix_calib_manifest.jsonl" "$base/main_manifest.jsonl" \
    --start-seed "$start_seed" --scope DRY --output "$base/runs/DRY_$start_seed"
done
