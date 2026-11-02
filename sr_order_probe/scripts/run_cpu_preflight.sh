#!/usr/bin/env bash
set -euo pipefail
root=/projects/_ssd/varunssd/louis-ssvc/sr_order_probe_20261102
code=${1:-"$root/deploy/v2"}
old=/projects/_ssd/varunssd/louis-ssvc/sr_f6_semantic_factorial_20261027
native=/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_b7e7b06
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$code"
"$python_bin" -m unittest discover -s tests -v
"$python_bin" -m order_probe_server prepare --config config/plan.json --inputs data --run-root "$root"
"$python_bin" scripts/server_preflight.py --inputs "$code/data" \
  --native-code-root "$native" \
  --model-path /projects/_ssd/varunssd/louis-ssvc/cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --history "$old/DATA/TRAIN/TASKS_PRIVATE.jsonl" \
  --history "$old/DATA/DEV_CAL/TASKS_PRIVATE.jsonl" \
  --history "$old/DATA/DEV_PILOT/TASKS_PRIVATE.jsonl" \
  --history "$old/DATA/DEV_EVAL/TASKS_PRIVATE.jsonl" \
  --history /projects/_ssd/varunssd/louis-ssvc/sr_f6_interface_v3_20261028/INPUT_PLAN/STAGE1/TASKS_PRIVATE.jsonl \
  --history /projects/_ssd/varunssd/louis-ssvc/sr_f6_interface_v3_20261028/INPUT_PLAN/STAGE2/TASKS_PRIVATE.jsonl \
  --out "$root/PREFLIGHT_CPU_v2"
