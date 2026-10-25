#!/bin/bash
#SBATCH --job-name=LH-SRF5-G3-CACHE
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu-pro6000-12
set -euo pipefail
: "${SRF5_ROOT:?registered root}"
: "${SRF5_CODE:?frozen source}"
: "${SRF5_REVISION:?frozen revision}"
: "${SRF5_DIAG_SCRIPT:?committed diagnostic script}"
export PYTHONPATH="$SRF5_CODE/src:$SRF5_CODE"
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SRF5_CODE"
python=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
"$python" - <<'PY'
import os
from pathlib import Path
from sr_f4.common import read
from sr_f4.pool import Slurm, gpu_count, ACCOUNT, QOS
root = Path(os.environ['SRF5_ROOT']).resolve(strict=True)
assert root.is_relative_to(Path('/projects/_ssd/varunssd/louis-ssvc'))
receipt = read(root/'CONTINUATION/PROBABILITY_CACHE_226283/SUBMISSION.json')
assert receipt['job_id'] == os.environ['SLURM_JOB_ID']
assert receipt['science_revision'] == os.environ['SRF5_REVISION']
assert receipt['diagnostic_script'] == os.environ['SRF5_DIAG_SCRIPT']
job = Slurm().show(os.environ['SLURM_JOB_ID'])
assert job['JobName'] == 'LH-SRF5-G3-CACHE' and job['QOS'] == QOS and job['Account'] == ACCOUNT
assert gpu_count(job['ReqTRES']) == 2
PY
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader --loop=30 > "$SRF5_ROOT/slurm/GPU-$SLURM_JOB_ID.csv" &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true; wait "$telemetry" 2>/dev/null || true' EXIT
"$python" -u - <<'PY'
import os, subprocess, sys
from pathlib import Path
from sr_f5.dual_gpu import worker_environments
environments = worker_environments(0, os.environ['CUDA_VISIBLE_DEVICES'], dict(os.environ))
root = Path(os.environ['SRF5_ROOT'])
children = []
# Each diagnostic is independent: retain the other precision result if one fails.
for mode, env in zip(('bf16', 'fp32'), environments, strict=True):
    command = [sys.executable, '-u', os.environ['SRF5_DIAG_SCRIPT'],
               '--root', str(root), '--code', os.environ['SRF5_CODE'],
               '--revision', os.environ['SRF5_REVISION'], '--mode', mode,
               '--output', str(root/'CONTINUATION/PROBABILITY_CACHE_226283'/f'{mode}.json')]
    with (root/'slurm'/f'G3-CACHE-{os.environ["SLURM_JOB_ID"]}-{mode}.log').open('x') as stream:
        children.append(subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT))
statuses = [child.wait() for child in children]
print({'diagnostic_exit_codes': statuses}, flush=True)
raise SystemExit(int(any(statuses)))
PY
