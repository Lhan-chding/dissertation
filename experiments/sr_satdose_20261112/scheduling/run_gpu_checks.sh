#!/usr/bin/env bash
#SBATCH --job-name=sr-satdose-gpu-checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=pro6000:2
#SBATCH --mem=66G
#SBATCH --no-requeue
set -euo pipefail
# The one argument is independently recorded, never computed here to self-authorize.
[[ $# == 1 && $1 =~ ^[0-9a-f]{64}$ ]] || exit 2
binding_sha=$1
base=/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
launcher_source=${BASH_SOURCE[0]}
[[ "$launcher_source" == /* ]] || launcher_source="$PWD/$launcher_source"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
unset PYTORCH_CUDA_ALLOC_CONF PYTHONPATH PYTHONHOME
[[ ${SLURM_RESTART_COUNT:-0} == 0 ]] || exit 2
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_NTASKS:-} == 1 && ${SLURM_NNODES:-} == 1 ]] || exit 2

# Stdlib only: no release import or GPU construction until every check succeeds.
# Repeated before EACH finite stage; disk observations are not a reservation.
guard() {
  "$python_bin" -I -B - "$base" "$binding_sha" "$launcher_source" "$1" "$2" <<'PY'
import datetime, errno, hashlib, json, os, pathlib, re, stat, sys, time, uuid
ROOT = pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
ANCESTOR = pathlib.Path('/projects/_ssd/varunssd')
FREEZE_SHA = '6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
EXPERIMENT = 'SR-SATDOSE-1-20261112'

def require(ok,message):
    if not ok:raise SystemExit('HOLD_GPU_LAUNCHER: '+message)

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()

def pairs(items):
    value=dict(items);require(len(value)==len(items),'Duplicate JSON key');return value

def read(path):
    return json.loads(path.read_text(),object_pairs_hook=pairs,
        parse_constant=lambda _:require(False,'Nonfinite JSON'))

def ordinary(path):
    require(path.resolve(strict=True)==path and stat.S_ISREG(path.lstat().st_mode),'Canonical ordinary file required')
    return path

def bound(ref):
    relative=pathlib.Path(ref['path'])
    require(not relative.is_absolute() and '..' not in relative.parts,'Evidence escapes fixed root')
    path=ordinary(ROOT/relative)
    require(re.fullmatch('[0-9a-f]{64}',ref['sha256']) and sha(path)==ref['sha256'],'Bound artifact SHA mismatch')
    return path

require(pathlib.Path(sys.argv[1])==ROOT and ROOT.resolve(strict=True)==ROOT and ROOT.is_relative_to(ANCESTOR),
        'Fixed evidence root required')
expected=sys.argv[2];start=int(sys.argv[4]);stage=sys.argv[5]
require(re.fullmatch('[0-9a-f]{64}',expected) and start in (11031,11032) and
        stage in ('evaluation_probe','backend_equivalence','allzero_check','diagnostic_worker:0','diagnostic_worker:1'),
        'Invalid fixed stage/binding SHA')
binding_path=ordinary(ROOT/'TECHNICAL_EXECUTION_BINDING.json')
require(sha(binding_path)==expected,'Technical binding SHA mismatch')
binding=read(binding_path)
require(binding.get('schema')=='sr-satdose-technical-execution-v1' and binding.get('mode')=='production' and
        binding.get('experiment_id')==EXPERIMENT and binding.get('experiment_root')==str(ROOT),
        'Foreign technical binding')
require(binding.get('max_shadow_optimizer_steps')==4 and binding.get('steps_per_slot')==1,'Expanded shadow budget')
require(binding['design_freeze']==dict(path='DESIGN_FREEZE.json',sha256=FREEZE_SHA),'Original freeze pin differs')
freeze=read(bound(binding['design_freeze']))
require(freeze.get('schema')=='sr-satdose-design-freeze-v1' and freeze.get('experiment_id')==EXPERIMENT and
        binding['config']==freeze['artifacts']['config'] and binding['source_lock']==freeze['artifacts']['source_lock'],
        'Frozen design/config/source identity differs')
bound(binding['config']);bound(binding['source_lock'])
receipt_path=bound(binding['gpu_release']);receipt=read(receipt_path)
require(receipt.get('schema')=='sr-satdose-gpu-technical-release-v1' and
        receipt.get('status')=='AUDITED_FOR_GPU_CHECKS_ONLY' and receipt.get('permission')=='NO_TRAINING' and
        receipt.get('experiment_id')==EXPERIMENT and receipt.get('evidence_root')==str(ROOT) and
        receipt.get('design_freeze_sha256')==FREEZE_SHA and receipt.get('config_sha256')==binding['config']['sha256'],
        'Independent audited GPU release receipt required')
release=pathlib.Path(receipt['release_root'])
require(release.is_absolute() and release.resolve(strict=True)==release and release.is_dir() and
        not receipt_path.is_relative_to(release),'Release/receipt must be canonical and separate')
actual={}
def walk_error(exc):raise SystemExit('HOLD_GPU_LAUNCHER: unreadable release inventory: '+str(exc))
for current,directories,files in os.walk(release,followlinks=False,onerror=walk_error):
    current=pathlib.Path(current)
    for name in directories:require(stat.S_ISDIR((current/name).lstat().st_mode),'Release symlink/special directory')
    for name in files:
        path=ordinary(current/name);actual[str(path.relative_to(release))]=sha(path)
require(actual==receipt['sources'],'Full release ordinary-file inventory/SHA mismatch')
needed={'execution/evaluation_probe.py','execution/evaluation_loader.py','execution/backend_equivalence.py',
        'execution/allzero_check.py','execution/diagnostic_worker.py','execution/shadow_replay.py','execution/production.py',
        'execution/training_state.py','execution/training_transaction.py','execution/diagnostic_inputs.py',
        'scheduling/run_gpu_checks.sh','analysis/calibration.py','supplied_package/config/source_bindings.json',
        'supplied_package/tools/satdose_core.py'}
require(needed<=set(actual),'Incomplete GPU technical release')
require({key.removeprefix('runtime/'):value for key,value in actual.items() if
        key.startswith('runtime/') and key.endswith('.py')}==freeze['runtime_sources'],
        'Release frozen runtime inventory differs')
launcher=pathlib.Path(sys.argv[3]).resolve(strict=True)
require(stat.S_ISREG(launcher.lstat().st_mode) and sha(launcher)==actual['scheduling/run_gpu_checks.sh'] and
        binding['code']['gpu_launcher']==dict(path=str(release/'scheduling/run_gpu_checks.sh'),
                                              sha256=actual['scheduling/run_gpu_checks.sh']),
        'Executing Slurm-spooled launcher bytes/code pin differ')
for name in ('evaluation_probe','evaluation_loader','backend_equivalence','allzero_check','diagnostic_worker',
             'shadow_replay','production','training_state','training_transaction','diagnostic_inputs'):
    require(binding['code'][name]==dict(path=str(release/'execution'/(name+'.py')),
            sha256=actual['execution/'+name+'.py']),'Code pin does not belong to audited release: '+name)
waiver_keys={'prefix_waiver','prefix_continuation'}
if waiver_keys.intersection(binding) or 'prefix_waiver' in binding.get('code',{}):
    require(waiver_keys<=set(binding) and 'prefix_waiver' in binding.get('code',{}),
            'Incomplete source-bound PREFIX waiver binding')
    relative='execution/prefix_waiver.py';source=ordinary(release/relative)
    require(relative in actual and binding['code']['prefix_waiver']==dict(path=str(source),sha256=actual[relative]),
            'PREFIX waiver code pin does not belong to audited release')
    bound(binding['prefix_waiver']);bound(binding['prefix_continuation'])
    # Only after the entire release and source pin have been authenticated.
    import importlib.util
    sys.path.insert(0,str(release))
    spec=importlib.util.spec_from_file_location('execution._bound_prefix_waiver',source)
    module=importlib.util.module_from_spec(spec);source_bytes=source.read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest()==actual[relative],'PREFIX waiver executable bytes differ')
    exec(compile(source_bytes,str(source),'exec'),module.__dict__)
    require(pathlib.Path(module.__file__)==source and sha(source)==actual[relative],
            'PREFIX waiver source changed during import')
    module.verify_completed(ROOT,binding)
    require(sha(source)==actual[relative],'PREFIX waiver source changed during verification')
else:
    prefix=read(bound(binding['prefix_result']))
    require(binding['prefix_result']['path']=='prefix_control/RESULT.json' and
            not (ROOT/'prefix_control/HOLD.json').exists() and
            prefix.get('status')=='PREFIX_METRICS_COMPLETE_GPU_CHECK_STILL_REQUIRED' and
            prefix.get('permission')=='NO_TRAINING' and prefix.get('responses')==96 and
            set(prefix.get('prefix_analyses',{}))=={'11031','11032'} and
            all(value.get('status')=='PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED' and value.get('candidate_gate_passed') is True
                for value in prefix['prefix_analyses'].values()),'Both PREFIX starts must PASS; module verifies full evidence chain')

# Explicit, independently hashed CPU storage budget; no invented free-space threshold.
budget_path=bound(binding['storage_budget']);budget=read(budget_path)
def now():return datetime.datetime.now(datetime.timezone.utc)
def utc(text):
    value=datetime.datetime.fromisoformat(text.replace('Z','+00:00'))
    require(value.tzinfo is not None and value.utcoffset().total_seconds()==0,'Budget timestamp must be UTC')
    return value
require(budget.get('schema')=='sr-satdose-storage-budget-v1' and budget.get('scope')=='GPU_DIAGNOSTICS_ONLY' and
        budget.get('experiment_id')==EXPERIMENT and budget.get('evidence_root')==str(ROOT) and
        budget.get('design_freeze_sha256')==FREEZE_SHA and type(budget.get('required_bytes')) is int and
        budget['required_bytes']>0 and type(budget.get('max_observation_age_seconds')) is int and
        budget['max_observation_age_seconds']>0,'Missing trusted explicit GPU storage budget')
require(utc(budget['created_at_utc'])<=now()<utc(budget['expires_at_utc']),'Storage budget stale/not yet valid')
if stage=='evaluation_probe':
    target=ROOT/f'evaluation_loader_checks/start-{start}'
    require(binding['evaluation_loader_slots'][str(start)]==str(target.relative_to(ROOT)),'Evaluation slot relocated')
elif stage=='backend_equivalence':
    target=ROOT/f'backend_equivalence/start-{start}'
    require(binding['backend_equivalence_slots'][str(start)]==str(target.relative_to(ROOT)),'Backend slot relocated')
elif stage=='allzero_check':
    target=ROOT/f'allzero_check/start-{start}'
    require(binding['allzero_slots'][str(start)]==str(target.relative_to(ROOT)),'Allzero slot relocated')
else:
    trial=int(stage[-1]);target=ROOT/f'shadow_inputs/start-{start}/trial-{trial}'
    require(binding['shadow_slots'][f'{start}/{trial}']['input_receipt']==str((target/'INPUT_RECEIPT.json').relative_to(ROOT)),
            'Shadow slot relocated')
    require(not (ROOT/f'shadow_claims/start-{start}/trial-{trial}').exists(),'Existing shadow attempt; never retry')
require(not target.exists() and not target.is_symlink(),'Existing stage output; never retry')
parent=target.parent
while not parent.exists():parent=parent.parent
require(parent.resolve(strict=True)==parent and parent.is_relative_to(ROOT),'Noncanonical output parent')

def xattr(path,key,absent_ok=False):
    require(callable(getattr(os,'getxattr',None)),'Ceph xattrs unavailable')
    try:return os.getxattr(path,key,follow_symlinks=False)
    except OSError as exc:
        if absent_ok and exc.errno==errno.ENODATA:return None
        raise SystemExit(f'HOLD_GPU_LAUNCHER: cannot verify {key} at {path}: errno {exc.errno}')

def integer(raw):
    require(isinstance(raw,bytes) and re.fullmatch(rb'[0-9]+',raw),'Malformed Ceph integer');return int(raw)

def observe():
    entries=[];limits=[];path=parent
    while True:
        require(path.resolve(strict=True)==path,'Quota path became noncanonical')
        quota=xattr(path,'ceph.quota.max_bytes',True)
        if quota is None:
            require(path!=ANCESTOR,'Fixed Ceph ancestor has no verified quota')
            entries.append(dict(path=str(path),status='NO_LOCAL_LIMIT_ENODATA'))
        else:
            maximum=integer(quota);require(maximum>0,'Unverified zero quota semantics')
            used=integer(xattr(path,'ceph.dir.rbytes'))
            require(xattr(path,'ceph.quota.max_bytes')==quota,'Quota changed during observation')
            available=max(0,maximum-used);limits.append(available)
            entries.append(dict(path=str(path),max_bytes=maximum,rbytes=used,available_bytes=available))
        if path==ANCESTOR:break
        path=path.parent
    filesystem=os.statvfs(parent)
    require(type(filesystem.f_bavail) is int and filesystem.f_bavail>=0 and
            type(filesystem.f_frsize) is int and filesystem.f_frsize>0,'Invalid statvfs')
    filesystem_available=filesystem.f_bavail*filesystem.f_frsize
    return dict(quota_observations=entries,filesystem_available_bytes=filesystem_available,
                effective_available_bytes=min([filesystem_available,*limits]),observed_at_utc=now().isoformat())

def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)

started=time.monotonic();first=observe()
require(first['effective_available_bytes']>=budget['required_bytes'],'Insufficient live quota/filesystem capacity')
probe=parent/('.gpu-checks-storage-probe-'+uuid.uuid4().hex);created=False;owned=None
try:
    fd=os.open(probe,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);created=True;owned=os.fstat(fd)
    with os.fdopen(fd,'wb') as stream:
        stream.write(b'SATDOSE GPU diagnostics writable/fsync probe\n');stream.flush();os.fsync(stream.fileno())
    sync_directory(parent)
finally:
    if created:
        current=probe.lstat()
        require((current.st_dev,current.st_ino)==(owned.st_dev,owned.st_ino),'Probe owner changed; do not delete')
        probe.unlink();sync_directory(parent)
second=observe();elapsed=time.monotonic()-started
require(second['effective_available_bytes']>=budget['required_bytes'] and elapsed<=budget['max_observation_age_seconds'] and
        now()<utc(budget['expires_at_utc']),'Storage guard stale/insufficient after fsync probe')
require(sha(binding_path)==expected and sha(budget_path)==binding['storage_budget']['sha256'],'Binding/budget changed during guard')
print(json.dumps(dict(schema='sr-satdose-gpu-launcher-storage-observation-v1',permission='NO_TRAINING',
    binding_sha256=expected,start_seed=start,stage=stage,budget=binding['storage_budget'],required_bytes=budget['required_bytes'],
    observations=[first,second],elapsed_seconds=elapsed,writable_probe_fsynced_and_removed=True),sort_keys=True),file=sys.stderr)
print(release)
PY
}

# Ten independent Python/srun processes. A nonzero guard or stage stops the job.
for start_seed in 11031 11032; do
  for stage in evaluation_probe backend_equivalence allzero_check diagnostic_worker:0 diagnostic_worker:1; do
    release_root=$(guard "$start_seed" "$stage")
    cd "$release_root"
    module=${stage%%:*}
    args=(--start-seed "$start_seed" --binding-sha256 "$binding_sha")
    if [[ "$module" == diagnostic_worker ]]; then args+=(--trial "${stage##*:}"); fi
    env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 \
      --cpus-per-task=8 --gpus-per-task=pro6000:2 --mem=66G \
      "$python_bin" -B -m "execution.$module" "${args[@]}"
  done
done
