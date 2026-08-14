#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${COMPBIAS_PROJECT_ROOT:-/cloud/cloud-ssd1/dissertation}"
cd "${PROJECT_ROOT}"

python3 -c 'import sys, torch; assert sys.version_info[:2] == (3, 12); assert str(torch.__version__).startswith("2.8.0"); assert torch.version.cuda == "12.8"'

if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate

python -m pip install --no-deps -e .
python -m pip install --constraint constraints-gpu.txt -r requirements-gpu.in
python -m pip check
python -c 'import torch; assert str(torch.__version__).startswith("2.8.0"); assert torch.version.cuda == "12.8"'

echo "Candidate GPU environment installed without replacing torch."
echo "Generate requirements-gpu.lock.txt only after preflight and Qwen smoke pass."
