#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=pro6000:2
#SBATCH --mem=66G
#SBATCH --no-requeue
set -euo pipefail
# Three independently recorded inputs; never self-compute an authorization SHA.
[[ $# == 3 && "$1" == /* ]] || exit 2
receipt=$1
receipt_sha=$2
authorization_sha=$3
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
unset PYTORCH_CUDA_ALLOC_CONF PYTHONPATH PYTHONHOME
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
[[ ${SLURM_RESTART_COUNT:-0} == 0 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
[[ ${SLURM_NTASKS:-} == 1 && ${SLURM_NNODES:-} == 1 ]] || exit 2
# Stdlib bootstrap before importing any release code; spool path may differ,
# but actual executing script bytes must equal the independently audited file.
release_root=$("$python_bin" -I -B - "$receipt" "$receipt_sha" "$authorization_sha" "${BASH_SOURCE[0]}" <<'PY'
import hashlib,json,os,pathlib,re,stat,sys
BASE=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
FREEZE='6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
CONFIG='58841639b0f36a69ac77824bc8c683ef7fbeb5c50a41e9975b89b54880f4ecfe'
WAIVER='36012b46494f4fcb3409f4816156406caad7ae1ad65c4688b972b75c83d73783'
def require(ok,message):
    if not ok:raise SystemExit('HOLD_PREFIX_CONTINUATION: '+message)
def sha(path):
    require(path.resolve(strict=True)==path and stat.S_ISREG(path.lstat().st_mode),'Canonical ordinary file required')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def pairs(items):
    require(len(dict(items))==len(items),'Duplicate JSON key');return dict(items)
path=pathlib.Path(sys.argv[1])
require(path.is_relative_to(BASE) and re.fullmatch('[a-f0-9]{64}',sys.argv[2]) and sha(path)==sys.argv[2],'Independent release SHA differs')
record=json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=lambda _:require(False,'Nonfinite JSON'))
require(sys.argv[3]==WAIVER and sha(BASE/'protocol_clarifications/PREFIX_GATE_WAIVER_20261114.json')==WAIVER,'Authorization SHA differs')
require(sha(BASE/'DESIGN_FREEZE.json')==FREEZE,'Original freeze changed')
expected=dict(schema='sr-satdose-prefix-continuation-release-v1',status='AUDITED_FOR_PREFIX_11032_CONTINUATION_ONLY',
    permission='NO_TRAINING',experiment_id='SR-SATDOSE-1-20261112',evidence_root=str(BASE),
    design_freeze_sha256=FREEZE,config_sha256=CONFIG,frozen_code_root=str(BASE/'code'),
    waiver=dict(path='protocol_clarifications/PREFIX_GATE_WAIVER_20261114.json',sha256=WAIVER),
    original_release=dict(path='execution_releases/PREFIX_RELEASE_734e921.json',sha256='fd8f7a780af3e841f891cac95683a5c71e0151551dc89142afd696202a0b713d'))
require(all(record.get(k)==v for k,v in expected.items()),'Wrong continuation release/waiver identity')
root=pathlib.Path(record['release_root'])
require(root.is_absolute() and root.resolve(strict=True)==root and root.is_dir() and not path.is_relative_to(root),'Receipt must be outside canonical release')
actual={}
def error(exc):raise exc
for directory,dirs,files in os.walk(root,followlinks=False,onerror=error):
    for name in dirs:
        p=pathlib.Path(directory)/name
        require(p.resolve(strict=True)==p and not p.is_symlink(),'Directory symlink forbidden')
    for name in files:
        p=pathlib.Path(directory)/name;actual[str(p.relative_to(root))]=sha(p)
require(actual==record['sources'],'Full release inventory differs')
launcher=pathlib.Path(sys.argv[4]).resolve(strict=True)
require(sha(launcher)==actual.get('scheduling/run_prefix_continuation.sh'),'Executing spool launcher SHA differs')
require({'execution/prefix_continuation.py','execution/prefix_waiver.py','execution/prefix_control.py'}<=set(actual),'Missing continuation code')
print(root)
PY
)
cd "$release_root"
exec "$python_bin" -B -m execution.prefix_continuation --release-receipt "$receipt" \
  --release-sha256 "$receipt_sha" --authorization-sha256 "$authorization_sha" --execute
