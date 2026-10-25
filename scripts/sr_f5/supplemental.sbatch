#!/bin/bash
#SBATCH --job-name=LH-SRF5-G3-EXTRA
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:30:00
#SBATCH --exclude=gpu-pro6000-12
set -euo pipefail
: "${SRF5_ROOT:?registered run root}"
: "${SRF5_CODE:?frozen science deployment}"
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
from sr_f5.pilot import require_frozen
root=Path(os.environ['SRF5_ROOT']).resolve()
assert root.is_relative_to(Path('/projects/_ssd/varunssd/louis-ssvc'))
spec=read(root/'CONTINUATION/SUPPLEMENTAL_G3.json')
assert spec['science_revision']==require_frozen(root)['code_revision']
assert spec['job_id']==os.environ['SLURM_JOB_ID']
actual=Slurm().show(os.environ['SLURM_JOB_ID'])
assert actual['JobName']=='LH-SRF5-G3-EXTRA' and actual['QOS']==QOS and actual['Account']==ACCOUNT
assert gpu_count(actual['ReqTRES'])==1
assert spec['paths']==[{'parent':'b2','arm':'DENSE'},{'parent':'b2','arm':'LOCAL'}]
assert spec['pair0_restart_barrier']=='supplemental_terminal_and_both_paths_COMPLETE'
PY
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader --loop=30 > "$SRF5_ROOT/slurm/GPU-$SLURM_JOB_ID.csv" &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true; wait "$telemetry" 2>/dev/null || true' EXIT
for arm in DENSE LOCAL; do
  "$python" -u -m sr_f5.pilot --root "$SRF5_ROOT" --parent b2 --arm "$arm"
done
"$python" - <<'PY'
import os
from pathlib import Path
from sr_f4.common import read, write, now
root=Path(os.environ['SRF5_ROOT'])
for arm in ('DENSE','LOCAL'):
    result=read(root/f'G3/RUNS/b2/{arm}/primary/COMPLETE.json')
    assert result['status']=='COMPLETE' and result['committed_step']==12
write(root/'CONTINUATION/SUPPLEMENTAL_G3_COMPLETE.json',dict(status='COMPLETE',job_id=os.environ['SLURM_JOB_ID'],at=now(),scientific_gate='NOT_YET_AGGREGATED'))
PY
