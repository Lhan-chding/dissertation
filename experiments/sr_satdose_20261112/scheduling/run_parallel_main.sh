#!/usr/bin/env bash
set -euo pipefail
[[ $# == 6 && $2 =~ ^[0-9a-f]{64}$ && $6 =~ ^[0-6]$ ]] || exit 2
[[ $3 == 11031 || $3 == 11032 ]] || exit 2
[[ $4 == START || $4 == T0 || $4 == T1 ]] || exit 2
[[ $5 == 0 || $5 == 8 || $5 == 16 || $5 == 24 ]] || exit 2
[[ ${SLURM_RESTART_COUNT:-0} == 0 && ${SLURM_NTASKS:-} == 1 && ${SLURM_NNODES:-} == 1 ]] || exit 2
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || exit 2
root=/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1
code="$root/external_parallel_main_v1"
batch="$root/execution_releases/parallel_main_v1"
[[ "$1" == "$batch/RELEASE.json" ]] || exit 2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
unset PYTORCH_CUDA_ALLOC_CONF PYTHONPATH PYTHONHOME
launcher=${BASH_SOURCE[0]}
[[ "$launcher" == /* ]] || launcher="$PWD/$launcher"

# A worker can start while its peer remains scheduler-pending. Both jobs must
# already have been admitted and independently observed as released before srun.
"$python_bin" -I -B - "$root" "$code" "$launcher" "$@" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path
root, code, launcher, release = map(Path, sys.argv[1:5])
release_sha, seed, arm, step, wave = sys.argv[5:]
job = int(os.environ['SLURM_JOB_ID'])
pairs = (('START',0),('T0',8),('T0',16),('T0',24),('T1',8),('T1',16),('T1',24))
assert (arm,int(step)) == pairs[int(wave)]
def data(path):
    assert path.resolve(strict=True) == path and path.is_file()
    return path.read_bytes()
def sha(path):
    return hashlib.sha256(data(path)).hexdigest()
assert sha(release) == release_sha
value=json.loads(data(release))
assert value['parallel_root'] == str(code) and value['permission'] == 'FIXED_MAIN_ONLY_NO_TRAINING'
inventory={}
for f in code.rglob('*'):
    assert f.resolve(strict=True)==f and not f.is_symlink()
    if f.is_file(): inventory[str(f.relative_to(code))]=sha(f)
    else: assert f.is_dir()
assert inventory==value['parallel_sources'] and sha(launcher)==inventory['scheduling/run_parallel_main.sh']
d=release.parent/f'wave-{int(wave):02d}'
pair=json.loads(data(d/'PAIR.json'))
assert pair['jobs'][seed]==job and len(set(pair['jobs'].values()))==2
assert pair['wave']==int(wave) and (pair['arm'],pair['step'])==(arm,int(step))
assert pair['release']=={'path':str(release.relative_to(root)),'sha256':release_sha}
for _ in range(150):
    assert not any('ERROR' in p.name or 'HOLD' in p.name for p in d.iterdir())
    if (d/'RELEASE_RESULT.json').exists():
        result=json.loads(data(d/'RELEASE_RESULT.json'))
        assert result['status']=='MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED' and result['jobs']==pair['jobs']
        assert result['pair']=={'path':str((d/'PAIR.json').relative_to(root)),'sha256':sha(d/'PAIR.json')}
        break
    time.sleep(2)
else:
    raise RuntimeError('Pair release remains unconfirmed')
PY
cd "$code"
if env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --gpus-per-task=pro6000:2 --mem=66G \
  "$python_bin" -I -B "$code/execution/parallel_main_control.py" \
  --release-receipt "$1" --release-sha256 "$2" --start-seed "$3" --arm "$4" --step "$5" --execute; then
  exit 0
else
  rc=$?
  "$python_bin" -I -B - "$batch" "$3" "$6" "$SLURM_JOB_ID" "$rc" <<'PY'
import json,os,sys
from pathlib import Path
batch,seed,wave,job,rc=sys.argv[1:]
d=Path(batch)/f'wave-{int(wave):02d}'
assert d.resolve(strict=True)==d and json.loads((d/'PAIR.json').read_bytes())['jobs'][seed]==int(job)
with (d/f'ERROR_{seed}.json').open('x') as f:
    json.dump(dict(status='MAIN_PARALLEL_SRUN_FAILED',job_id=int(job),returncode=int(rc)),f)
    f.flush(); os.fsync(f.fileno())
PY
  exit "$rc"
fi
