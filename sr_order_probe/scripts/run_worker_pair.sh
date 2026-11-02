#!/usr/bin/env bash
# One parent, two independent batch-one GPU workers. No cross-request state.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: run_worker_pair.sh CODE_DIR RUN_ROOT SMOKE|MAIN b0|b1|b2' >&2
  exit 2
fi
code_dir=$1
run_root=$2
split=$3
parent=$4
[[ "$split" == SMOKE || "$split" == MAIN ]]
[[ "$parent" =~ ^b[012]$ ]]
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$code_dir"
# Each Slurm step receives exactly one allocated GPU. Children cannot see both.
pids=()
for shard in 0 1; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
    "$python_bin" -u -m order_probe_server run \
    --run-root "$run_root" --split "$split" --parent "$parent" \
    --shard-index "$shard" --shard-count 2 &
  pids+=("$!")
done
status=0
for worker_pid in "${pids[@]}"; do
  if wait "$worker_pid"; then :; else status=1; fi
done
exit "$status"
