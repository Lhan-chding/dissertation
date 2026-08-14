#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${COMPBIAS_PROJECT_ROOT:-/cloud/cloud-ssd1/dissertation}"
PATHS_CONFIG="${COMPBIAS_PATHS_CONFIG:-${PROJECT_ROOT}/configs/paths.yaml}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${PROJECT_ROOT}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash scripts/server/preflight.sh
python experiments/gpu_pilot/01_smoke_qwen.py --paths "${PATHS_CONFIG}"
