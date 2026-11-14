"""One authorized 11032 PREFIX remainder; no resampling or training permission.

Original 11031 measurements/HOLD, old release and old control bytes remain
immutable. Only the measurement gate is waived; technical errors remain HOLD.
No model is created here: the sole subprocess uses the original frozen driver
and original BASE/code cwd with its original 11032/48 request population.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

PACKAGE = Path(__file__).resolve().parents[1]
BASE = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
FREEZE_SHA = '6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
CONFIG_SHA = '58841639b0f36a69ac77824bc8c683ef7fbeb5c50a41e9975b89b54880f4ecfe'
OLD_RELEASE = 'execution_releases/PREFIX_RELEASE_734e921.json'
OLD_RELEASE_SHA = 'fd8f7a780af3e841f891cac95683a5c71e0151551dc89142afd696202a0b713d'
WAIVER = 'protocol_clarifications/PREFIX_GATE_WAIVER_20261114.json'
WAIVER_SHA = '36012b46494f4fcb3409f4816156406caad7ae1ad65c4688b972b75c83d73783'
CONTROL = 'prefix_waiver_control'
EXPERIMENT = 'SR-SATDOSE-1-20261112'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),'Canonical regular file required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    def pairs(items):
        require(len(dict(items)) == len(items),'Duplicate JSON key')
        return dict(items)
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda _:require(False,'Nonfinite JSON'))


def inventory(root, *, old=False):
    require(root.resolve(strict=True) == root and root.is_dir(),'Canonical release/evidence directory required')
    result = {}
    def error(exc):raise exc
    for directory,dirs,files in os.walk(root,followlinks=False,onerror=error):
        if old:dirs[:] = [name for name in dirs if name != '__pycache__']
        for name in dirs:
            path = Path(directory)/name
            require(path.resolve(strict=True) == path and not path.is_symlink(),'Directory symlink forbidden')
        for name in files:
            path = Path(directory)/name
            result[str(path.relative_to(root))] = sha(path)
    return result


def ref(path):
    return dict(path=str(path.relative_to(BASE)),sha256=sha(path))


def sync_dir(path):
    fd = os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_once(path,value):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    sync_dir(path.parent)


def verify_release(receipt_path,expected_sha,authorization_sha):
    path = Path(receipt_path)
    require(re.fullmatch('[0-9a-f]{64}',expected_sha or '') and sha(path) == expected_sha,'Continuation release receipt SHA differs')
    require(authorization_sha == WAIVER_SHA and sha(BASE/WAIVER) == authorization_sha,'Independent authorization SHA differs')
    record = read(path)
    expected = dict(schema='sr-satdose-prefix-continuation-release-v1',status='AUDITED_FOR_PREFIX_11032_CONTINUATION_ONLY',
        permission='NO_TRAINING',experiment_id=EXPERIMENT,evidence_root=str(BASE),design_freeze_sha256=FREEZE_SHA,
        config_sha256=CONFIG_SHA,frozen_code_root=str(BASE/'code'),release_root=str(PACKAGE),
        original_release=dict(path=OLD_RELEASE,sha256=OLD_RELEASE_SHA),waiver=dict(path=WAIVER,sha256=authorization_sha))
    require(all(record.get(k) == v for k,v in expected.items()) and not path.is_relative_to(PACKAGE),
            'Continuation release identity/authority differs')
    sources = inventory(PACKAGE)
    required = {'execution/prefix_continuation.py','execution/prefix_waiver.py','execution/prefix_control.py',
                'scheduling/run_prefix_continuation.sh','analysis/calibration.py','supplied_package/tools/satdose_core.py'}
    require(required <= set(sources) and sources == record['sources'],'Complete continuation release inventory differs')
    return ref(path),record


def _modules():
    pc = importlib.import_module('execution.prefix_control')
    waiver = importlib.import_module('execution.prefix_waiver')
    for module,relative in ((pc,'execution/prefix_control.py'),(waiver,'execution/prefix_waiver.py'),
                            (pc.calibration,'analysis/calibration.py'),(pc.core,'supplied_package/tools/satdose_core.py')):
        require(Path(module.__file__).resolve() == PACKAGE/relative,'Shadowed continuation dependency')
    require(pc.BASE == BASE and pc.FREEZE_SHA == FREEZE_SHA and pc.CONFIG_SHA == CONFIG_SHA,'Controller frozen identity differs')
    return pc,waiver


def _old_release():
    path = BASE/OLD_RELEASE
    require(sha(path) == OLD_RELEASE_SHA,'Original 734 release receipt changed')
    record = read(path);root = BASE/'calibration_code_734e921'
    require(record['schema'] == 'sr-satdose-prefix-technical-release-v1' and
            record['status'] == 'AUDITED_FOR_PREFIX_CALIB_ONLY' and record['release_root'] == str(root) and
            record['design_freeze_sha256'] == FREEZE_SHA and record['config_sha256'] == CONFIG_SHA,
            'Wrong original release')
    require(inventory(root,old=True) == record['sources'],'Original 734 release inventory changed')
    # Reused control/scoring functions must be byte-identical to the old release.
    for relative in ('execution/prefix_control.py','analysis/calibration.py','supplied_package/tools/satdose_core.py'):
        require(sha(PACKAGE/relative) == record['sources'][relative],'Original PREFIX semantics changed')
    for name,digest in read(BASE/'DESIGN_FREEZE.json')['runtime_sources'].items():
        require(record['sources']['runtime/'+name] == digest == sha(BASE/'code/runtime'/name),
                'Original frozen subprocess runtime differs')
    return record


def no_gpu_attempts():
    require(not os.path.lexists(BASE/'TECHNICAL_EXECUTION_BINDING.json'),'GPU binding exists; this continuation is not a GPU retry')
    for name in ('shadow_inputs','shadow_claims','backend_equivalence','allzero_check','evaluation_loader_checks'):
        path = BASE/name
        require(not path.is_symlink() and (not path.exists() or (path.is_dir() and not any(path.iterdir()))),
                'Existing/unknown GPU slot blocks continuation')


def no_11032_attempts(pc):
    allowed = {'calibration_allocations/PREFIX_CALIB-11031.json','runs/PREFIX_CALIB_11031'}
    require(set(pc.prefix_attempts(BASE)) == allowed,'Existing 11032/unknown PREFIX attempt; no retry or resampling')
    allocation = read(BASE/'calibration_allocations/PREFIX_CALIB-11031.json')
    require(allocation.get('start_seed') == 11031 and allocation.get('scope') == 'PREFIX_CALIB' and
            allocation.get('output') == str(BASE/'runs/PREFIX_CALIB_11031') and
            allocation.get('design_freeze_sha256') == FREEZE_SHA,'Original allocation differs')


def protected_inventory():
    protected = {}
    for relative in ('prefix_control','runs/PREFIX_CALIB_11031'):
        protected.update({relative+'/'+name:digest for name,digest in inventory(BASE/relative).items()})
    for relative in (OLD_RELEASE,WAIVER,'calibration_allocations/PREFIX_CALIB-11031.json'):
        protected[relative] = sha(BASE/relative)
    return protected


def preflight(receipt_path,release_sha,authorization_sha):
    release,record = verify_release(receipt_path,release_sha,authorization_sha)
    _old_release();pc,waiver = _modules()
    authorization = waiver.verify_waiver(BASE,dict(path=WAIVER,sha256=authorization_sha),
        expected_freeze_sha256=FREEZE_SHA,expected_config_sha256=CONFIG_SHA)
    no_gpu_attempts();no_11032_attempts(pc)
    require(not os.path.lexists(BASE/CONTROL),'Existing continuation claim is UNKNOWN; no automatic retry')
    deployed = pc.verify_deployment()
    clarification,protocol = pc.protocol_identity(deployed['config'],deployed['freeze'])
    receipts = {s:BASE/f'runs/DRY_{s}/CALIBRATION_COMPLETE.json' for s in (11031,11032)}
    for s,path in receipts.items():
        header = read(path)
        require(header.get('status') == 'CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION' and
                header.get('scope') == 'DRY' and header.get('start_seed') == s and header.get('responses') == 640,
                'Both complete DRY populations required before raw reads')
    dry = {str(s):pc.calibration.analyze_dry(deployed['config'],deployed['freeze'],BASE,path,s,deployed['native_src'])
           for s,path in receipts.items()}
    require(all(r['status'] == 'DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED' and not r['failures'] and
                all(c['passed'] for c in r['checks'].values()) for r in dry.values()),'Original DRY gates must both pass')
    old = pc.calibration.analyze_prefix(deployed['config'],deployed['freeze'],BASE,
        BASE/'runs/PREFIX_CALIB_11031/CALIBRATION_COMPLETE.json',11031,clarification=clarification)
    original = read(BASE/authorization['original_result']['path'])
    require(original['status'] == 'HOLD_PREFIX_MEASUREMENT' and original['stopped_start'] == 11031 and
            set(original['prefix_analyses']) == {'11031'} and old == original['prefix_analyses']['11031'] ==
            read(BASE/'prefix_control/PREFIX_11031_ANALYSIS.json') and old['candidate_gate_passed'] is False,
            'Original complete 11031 failed measurement changed')
    return dict(schema='sr-satdose-prefix-waiver-preflight-v1',status='READY_FOR_PREFIX_11032_ONLY',permission='NO_TRAINING',
        waiver=dict(path=WAIVER,sha256=authorization_sha),release=release,original_result=authorization['original_result'],
        original_release=dict(path=OLD_RELEASE,sha256=OLD_RELEASE_SHA),design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA,
        start_seed=11032,responses=48,model_calls=0,source_inventory_sha256=pc.core.object_sha(record['sources']),
        dry_analyses=dry,original_analysis=old,clarification=str(clarification),protocol_clarification=protocol,
        protected_files=protected_inventory())


@contextmanager
def _lock():
    fd = os.open(BASE/'DIAGNOSTIC_WORKER.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:os.close(fd)


def command(pc):
    result = pc.command(11032)
    require(result[result.index('--start-seed')+1] == '11032' and result[result.index('--scope')+1] == 'PREFIX_CALIB' and
            result[result.index('--output')+1] == str(BASE/'runs/PREFIX_CALIB_11032'),'Unexpected continuation command')
    return result


def execute(receipt_path,release_sha,authorization_sha):
    verify_release(receipt_path,release_sha,authorization_sha)
    with _lock():
        return _execute_locked(receipt_path,release_sha,authorization_sha)


def _execute_locked(receipt_path,release_sha,authorization_sha):
    require(re.fullmatch('[0-9]+',os.environ.get('SLURM_JOB_ID','')) and os.environ.get('SLURM_NTASKS') == '1' and
            os.environ.get('SLURM_NNODES') == '1' and os.environ.get('SLURM_RESTART_COUNT','0') == '0',
            'Fresh single-task one-node Slurm allocation required')
    report = preflight(receipt_path,release_sha,authorization_sha)
    pc,waiver = _modules();control = BASE/CONTROL
    control.mkdir(exist_ok=False);sync_dir(BASE)
    write_once(control/'PREFLIGHT.json',report)
    write_once(control/'CLAIM.json',dict(schema='sr-satdose-prefix-waiver-claim-v1',status='RESERVED_UNKNOWN_NO_RETRY',
        permission='NO_TRAINING',start_seed=11032,responses=48,waiver=report['waiver'],release=report['release'],
        preflight=ref(control/'PREFLIGHT.json')))
    try:
        verify_release(receipt_path,release_sha,authorization_sha);_old_release();pc.verify_deployment()
        no_gpu_attempts();no_11032_attempts(pc)
        require(protected_inventory() == report['protected_files'],'Original evidence changed before launch')
        env = os.environ.copy()
        for name in ('PYTHONPATH','PYTHONHOME','PYTORCH_CUDA_ALLOC_CONF'):env.pop(name,None)
        env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',
            PYTORCH_ALLOC_CONF='expandable_segments:True')
        code = subprocess.run(command(pc),cwd=BASE/'code',env=env,check=False).returncode
        require(code == 0,'Original PREFIX 11032 subprocess failed; no automatic retry')
        fresh = pc.calibration.analyze_prefix(BASE/'code/supplied_package/config/experiment.json',BASE/'DESIGN_FREEZE.json',
            BASE,BASE/'runs/PREFIX_CALIB_11032/CALIBRATION_COMPLETE.json',11032,clarification=Path(report['clarification']))
        require(fresh['permission'] == 'NO_TRAINING' and fresh['status'] in
                ('HOLD_PREFIX_MEASUREMENT','PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED'), 'Technical/pending PREFIX outcome not waived')
        require(protected_inventory() == report['protected_files'],'Original evidence changed during continuation')
        verify_release(receipt_path,release_sha,authorization_sha);_old_release();no_gpu_attempts()
        write_once(control/'PREFIX_11032_ANALYSIS.json',fresh)
        result = dict(schema='sr-satdose-prefix-waiver-continuation-v1',
            status='PREFIX_CAPTURE_COMPLETE_WITH_AUTHORIZED_MEASUREMENT_WAIVER',gate_decision='PASS_USER_OVERRIDE',permission='NO_TRAINING',
            waiver=report['waiver'],original_result=report['original_result'],original_release=report['original_release'],
            design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA,responses=96,new_responses=48,resampled_responses=0,
            prefix_analyses={'11031':report['original_analysis'],'11032':fresh},release=report['release'],
            source_inventory_sha256=report['source_inventory_sha256'],preflight=ref(control/'PREFLIGHT.json'),claim=ref(control/'CLAIM.json'))
        write_once(control/'RESULT.json',result)
        waiver.verify_completed(BASE,dict(design_freeze=dict(path='DESIGN_FREEZE.json',sha256=FREEZE_SHA),
            config=read(BASE/'DESIGN_FREEZE.json')['artifacts']['config'],
            source_lock=read(BASE/'DESIGN_FREEZE.json')['artifacts']['source_lock'],
            prefix_result=report['original_result'],prefix_waiver=report['waiver'],prefix_continuation=ref(control/'RESULT.json')))
        return result
    except BaseException as exc:
        hold = dict(status='HOLD_PREFIX_WAIVER_TECHNICAL',permission='NO_TRAINING',error_type=type(exc).__name__,error=str(exc),
                    recovery='READ_ONLY_AUDIT_NO_AUTOMATIC_RETRY')
        write_once(control/'HOLD.json',hold)
        if not (control/'RESULT.json').exists():write_once(control/'RESULT.json',hold)
        return hold


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-receipt',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    p.add_argument('--authorization-sha256',required=True);p.add_argument('--execute',action='store_true')
    args = p.parse_args(argv)
    try:result = (execute if args.execute else preflight)(args.release_receipt,args.release_sha256,args.authorization_sha256)
    except Exception as exc:result = dict(status='HOLD_PREFIX_WAIVER_CONTROL',permission='NO_TRAINING',error=str(exc))
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 2 if result['status'].startswith('HOLD') else 0


if __name__ == '__main__':raise SystemExit(main())
