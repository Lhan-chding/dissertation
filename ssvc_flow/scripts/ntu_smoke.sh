#!/usr/bin/env bash
set -euo pipefail
ssvc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ssvc_root"
ssvc_python="${SSVC_PYTHON:-$ssvc_root/.venv/bin/python}"
"$ssvc_python" -c 'import torch; assert torch.cuda.is_available(), "Run inside your allocated NTU CUDA job; CPU/MPS is not P1"; print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory)'
if [[ ! -f data/generated/manifest.json ]]; then
  "$ssvc_python" -m src.generate_worlds --out data/generated --seed 17
fi
# Caller may pass --resume; existing output is never automatically deleted.
"$ssvc_python" -m src.rollout --phase smoke --model qwen35_9b --out runs/P1 "$@"
"$ssvc_python" -m src.report --run-root runs --out reports
