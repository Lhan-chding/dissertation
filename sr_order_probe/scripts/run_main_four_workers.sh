#!/usr/bin/env bash
# Fixed MAIN requests: four batch-one workers per parent, parents sequential.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: run_main_four_workers.sh CODE_DIR RUN_ROOT' >&2; exit 2; }
code_dir=$1
run_root=$2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
[[ ${SLURM_MEM_PER_NODE:-} =~ ^[0-9]+$ ]]
worker_mem_mb=$((SLURM_MEM_PER_NODE / 4))
((worker_mem_mb > 0))
cd "$code_dir"
step=(srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --mem="${worker_mem_mb}M")
for parent in b0 b1 b2; do
  pids=()
  for shard in 0 1 2 3; do
    "${step[@]}" "$python_bin" -u -m order_probe_server run \
      --run-root "$run_root" --split MAIN --parent "$parent" \
      --shard-index "$shard" --shard-count 4 &
    pids+=("$!")
  done
  status=0
  for worker_pid in "${pids[@]}"; do
    if wait "$worker_pid"; then :; else status=1; fi
  done
  ((status == 0)) || exit "$status"
done
# Finite postprocessing only, after all workers have closed and integrity passes.
"${step[@]}" "$python_bin" -m order_probe_server collect --run-root "$run_root" --split MAIN
"${step[@]}" "$python_bin" scripts/score_answers.py --inputs data --answers "$run_root/RAW/MAIN.jsonl" --split MAIN --out "$run_root/SCORES_MAIN.jsonl"
"${step[@]}" "$python_bin" scripts/analyze.py --scores "$run_root/SCORES_MAIN.jsonl" --out "$run_root/ANALYSIS_MAIN"
"${step[@]}" "$python_bin" scripts/supplementary_report.py --inputs data --raw "$run_root/RAW/MAIN.jsonl" --scores "$run_root/SCORES_MAIN.jsonl" --out "$run_root/SUPPLEMENT_MAIN"
# The independent 72-record review and scientific interpretation remain pending.
