#!/bin/bash
# One allocated GPU and one independent model per worker. Resource/QoS supplied
# explicitly by sbatch after inspecting the current cluster allocation.
set -euo pipefail
ser_python="$1"
ser_code="$2"
ser_run="$3"
ser_mode="$4"
ser_worker="${5:-bridge}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$ser_run/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$ser_code"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.free --format=csv
df -h "$ser_run"
if [[ "$ser_mode" == bridge ]]; then
    exec "$ser_python" -m src.exposure_substitution.cli bridge --run "$ser_run" --parent S96 --allow-gpu
elif [[ "$ser_mode" == worker ]]; then
    exec "$ser_python" -m src.exposure_substitution.cli worker --run "$ser_run" --worker "$ser_worker" --allow-gpu
else
    echo 'Unknown SER-J2 launch mode' >&2
    exit 2
fi
