#!/usr/bin/env bash
set -euo pipefail
ssvc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ssvc_root"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the verified CPU package set"'
if [[ -e .venv ]]; then
  echo 'ssvc_flow/.venv already exists; activate and inspect that environment instead of replacing it.' >&2
  exit 2
fi
python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements-gpu-bootstrap.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest tests -q
echo 'Bootstrap and CPU tests complete. Allocate an NTU CUDA GPU before running scripts/ntu_smoke.sh.'
