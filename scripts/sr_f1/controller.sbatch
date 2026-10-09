#!/bin/bash
# CPU orchestration only. Account and CPU QoS must be supplied to sbatch.
# All 4320-minute durations are renewable scheduler leases, never research stops.
# Usage: sbatch --account=<approved> --qos=<cpu-qos> controller.sbatch \
#        <python> <code-root> <config-path> <run-root>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4320
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "Expected python, code-root, config-path, run-root" >&2
    exit 2
fi
runtime_python="$1"
source_root="$2"
config_path="$3"
study_root="$4"
export PYTHONPATH="$source_root/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$runtime_python" "$source_root/scripts/sr_f1/submit_matrix.py" \
    --plan "$config_path" --run-root "$study_root" \
    --code-root "$source_root" --python "$runtime_python" \
    --lease-minutes 4320 --engine-lease-minutes 4320 \
    --watch --controller-requeue-on-lease --poll-seconds 30
