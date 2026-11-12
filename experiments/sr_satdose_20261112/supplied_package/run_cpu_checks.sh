#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash run_cpu_checks.sh /absolute/path/to/old_bundle /new/owned/output_directory' >&2
  exit 2
fi
BUNDLE="$1"
OUT="$2"
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -e "$OUT" ]]; then
  echo 'Refusing to overwrite an existing verification directory.' >&2
  exit 2
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export SR_HISTORICAL_BUNDLE="$BUNDLE"
python -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required"'
cd "$ROOT"
python -m unittest discover -s tests -v > "$OUT/unit_tests.log" 2>&1
python tools/build_schedule.py --check config/training_slots.jsonl > "$OUT/schedule_check.json"
python tools/audit_historical.py --bundle "$BUNDLE" --out "$OUT/HISTORICAL_AUDIT.json" > "$OUT/audit_stdout.log"
echo "CPU checks completed: $OUT"
echo 'This does NOT authorize model calls or training; actual GPU admission is still required.'
