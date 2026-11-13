"""Live Ceph quota/writability callback; no CLI, model, remote call or cleanup.

The driver must independently authorize and SHA-bind a storage-budget artifact.
This module never estimates the total required space. Measurements are a bounded
snapshot, not a reservation against concurrent users of the shared Ceph quota.
Historical transaction verification must not apply this module's *live* expiry
check to an old committed receipt; the transaction already hashes its evidence.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
QUOTA_ANCESTOR = Path('/projects/_ssd/varunssd')
EXPERIMENT = 'SR-SATDOSE-1-20261112'
FREEZE_SHA = '6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
QUOTA_KEY = 'ceph.quota.max_bytes'
USAGE_KEY = 'ceph.dir.rbytes'


def require(ok, message):
    if not ok:
        raise PermissionError('HOLD_STORAGE: '+message)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    def pairs(items):
        result=dict(items);require(len(result)==len(items),'duplicate JSON key');return result
    return json.loads(path.read_text(),object_pairs_hook=pairs,
        parse_constant=lambda value:require(False,'nonfinite JSON '+value))


def _now():
    return datetime.now(timezone.utc)


def _utc(value):
    require(isinstance(value,str),'missing UTC timestamp')
    try:
        parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError as exc:
        raise PermissionError('HOLD_STORAGE: malformed UTC timestamp') from exc
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds()==0,'timestamp must be explicit UTC')
    return parsed


def _ref(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=_sha(path))


def _bound(reference):
    relative=Path(reference['path']);require(not relative.is_absolute() and '..' not in relative.parts,'artifact escapes root')
    path=ROOT/relative
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'noncanonical bound artifact')
    require(re.fullmatch('[0-9a-f]{64}',reference.get('sha256','')) is not None and
            _sha(path)==reference['sha256'],'bound artifact SHA differs')
    return path


def _stage(stage,step):
    stage=Path(stage)
    require(type(step) is int and 1<=step<=24,'unregistered training step')
    require(ROOT.resolve(strict=True)==ROOT and QUOTA_ANCESTOR.resolve(strict=True)==QUOTA_ANCESTOR and
            ROOT.is_relative_to(QUOTA_ANCESTOR) and stage.is_absolute() and stage.resolve(strict=True)==stage and
            stage.is_dir() and not stage.is_symlink(),'stage/root must be canonical directories')
    require(stage.is_relative_to(ROOT/'training_transactions'),'stage outside transaction tree')
    parts=stage.relative_to(ROOT/'training_transactions').parts
    require(len(parts)==3 and parts[0] in ('start-11031','start-11032') and parts[1] in ('T0','T1') and
            parts[2]==f'.staging-step-{step:03d}','stage is not the fixed current transaction')
    require(not (stage/'UPDATE_INTENT.json').exists() and not (stage/'COMMIT.json').exists(),
            'storage callback must run before UPDATE_INTENT')
    intent_path=stage/'STEP_INTENT.json'
    require(intent_path.resolve(strict=True)==intent_path and intent_path.is_file(),'missing canonical STEP_INTENT')
    intent=_read(intent_path);identity=intent.get('identity',{})
    require(intent.get('step')==step and intent.get('design_freeze_sha256')==FREEZE_SHA and
            identity.get('experiment_id')==EXPERIMENT and identity.get('train_seed')==int(parts[0].removeprefix('start-')) and
            identity.get('arm')==parts[1],'stage intent identity differs')
    require(_sha(ROOT/'DESIGN_FREEZE.json')==FREEZE_SHA,'immutable freeze changed')
    return stage,dict(start_seed=identity['train_seed'],arm=identity['arm'],step=step,
        transaction_identity_sha256=hashlib.sha256(_canonical(identity)).hexdigest())


def _budget(reference,now):
    path=_bound(reference)
    require(not path.is_relative_to(ROOT/'training_transactions'),'budget must be outside renamed staging')
    value=_read(path)
    require(value.get('schema')=='sr-satdose-storage-budget-v1' and value.get('experiment_id')==EXPERIMENT and
            value.get('evidence_root')==str(ROOT) and value.get('design_freeze_sha256')==FREEZE_SHA,
            'budget identity differs')
    require(type(value.get('required_bytes')) is int and value['required_bytes']>0 and
            type(value.get('max_observation_age_seconds')) is int and value['max_observation_age_seconds']>0,
            'budget must explicitly supply positive required bytes and observation lifetime')
    created,expires=_utc(value.get('created_at_utc')),_utc(value.get('expires_at_utc'))
    require(created<=now<expires,'budget is stale or not yet valid')
    return value


def _integer(raw,name):
    require(isinstance(raw,bytes) and re.fullmatch(rb'[0-9]+',raw) is not None,
            'malformed Ceph integer: '+name)
    return int(raw)


def _getxattr(path,key,*,absent_ok=False):
    require(callable(getattr(os,'getxattr',None)),'platform cannot read Ceph xattrs')
    try:
        return os.getxattr(path,key,follow_symlinks=False)
    except OSError as exc:
        if absent_ok and exc.errno==errno.ENODATA:
            return None
        raise PermissionError(f'HOLD_STORAGE: cannot verify {key} at {path}: errno {exc.errno}') from exc


def _observe(stage):
    observations=[];available=[];path=stage
    while True:
        raw=_getxattr(path,QUOTA_KEY,absent_ok=True)
        observed_at=_now().isoformat()
        if raw is None:
            require(path!=QUOTA_ANCESTOR,'fixed Ceph quota ancestor has no verified positive quota')
            observations.append(dict(path=str(path),status='NO_LOCAL_LIMIT_ENODATA',observed_at_utc=observed_at))
        else:
            limit=_integer(raw,QUOTA_KEY)
            # No unverified interpretation of Ceph's zero-valued virtual xattr.
            require(limit>0,'zero quota semantics are unverified; explicit audit required')
            used=_integer(_getxattr(path,USAGE_KEY),USAGE_KEY)
            require(_getxattr(path,QUOTA_KEY)==raw,'quota limit changed while reading usage')
            free=max(0,limit-used);available.append(free)
            observations.append(dict(path=str(path),status='POSITIVE_QUOTA',max_bytes=limit,rbytes=used,
                available_bytes=free,observed_at_utc=observed_at))
        if path==QUOTA_ANCESTOR:
            break
        path=path.parent
    fs=os.statvfs(stage)
    require(type(fs.f_bavail) is int and fs.f_bavail>=0 and type(fs.f_frsize) is int and fs.f_frsize>0,
            'malformed statvfs writable availability')
    filesystem_bytes=fs.f_bavail*fs.f_frsize
    return dict(quota_observations=observations,filesystem_available_bytes=filesystem_bytes,
                effective_available_bytes=min([filesystem_bytes,*available]),observed_at_utc=_now().isoformat())


def _sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _probe(stage):
    """Clean only the exclusively created probe, including failure paths."""
    path=stage/('.storage_probe-'+uuid.uuid4().hex)
    descriptor=None;created=False;own_stat=None
    payload=b'SATDOSE storage writability and fsync probe\n'
    try:
        descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        created=True;own_stat=os.fstat(descriptor)
        with os.fdopen(descriptor,'wb') as stream:
            descriptor=None;stream.write(payload);stream.flush();os.fsync(stream.fileno())
        _sync_directory(stage)
        return dict(bytes_written=len(payload),file_fsync=True,directory_fsync=True,probe_removed=True)
    finally:
        if descriptor is not None:os.close(descriptor)
        if created:
            found=path.lstat()
            require((found.st_dev,found.st_ino)==(own_stat.st_dev,own_stat.st_ino),'probe ownership changed; no cleanup attempted')
            path.unlink();_sync_directory(stage)


def _write_measurement(path,value):
    # Only create this helper's canonical audit hierarchy, with no symlinks.
    relative=path.parent.relative_to(ROOT);parent=ROOT
    for part in relative.parts:
        parent=parent/part
        if not parent.exists():parent.mkdir();_sync_directory(parent.parent)
        require(parent.resolve(strict=True)==parent and parent.is_dir() and not parent.is_symlink(),'measurement directory symlink')
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'wb') as stream:
        stream.write(_canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    _sync_directory(path.parent)


def validate_readiness(value,*,budget_reference,stage,step):
    """Verify a fresh callback receipt immediately before update, not on replay."""
    stage,identity=_stage(stage,step);now=_now();budget=_budget(budget_reference,now)
    evidence=value.get('evidence',[])
    require(len(evidence)==2 and evidence[0]==budget_reference,'readiness must bind exact independently supplied budget')
    measurement_path=_bound(evidence[1]);measurement=_read(measurement_path)
    expected_path=ROOT/f"storage_readiness/start-{identity['start_seed']}/{identity['arm']}/step-{step:03d}.json"
    require(measurement_path==expected_path,'relocated storage measurement')
    expected=dict(measurement,schema='sr-satdose-storage-readiness-v1',evidence=evidence)
    require(measurement.get('schema')=='sr-satdose-storage-measurement-v1' and value==expected and
            value.get('experiment_id')==EXPERIMENT and value.get('permission')=='NO_TRAINING_ADMISSION' and
            value.get('evidence_root')==str(ROOT) and value.get('quota_ancestor')==str(QUOTA_ANCESTOR) and
            value.get('budget_reference')==budget_reference and value.get('identity')==identity and
            value.get('stage')==str(stage) and value.get('design_freeze_sha256')==FREEZE_SHA and
            value.get('required_bytes')==budget['required_bytes'],'readiness identity/budget/content mismatch')
    started,finished=_utc(value['started_at_utc']),_utc(value['checked_at_utc'])
    require(started<=finished<=now and (now-started).total_seconds()<=budget['max_observation_age_seconds'] and
            type(value.get('elapsed_seconds')) in (int,float) and
            0<=value['elapsed_seconds']<=budget['max_observation_age_seconds'],'stale storage observations')
    snapshots=value['observations'];require(isinstance(snapshots,list) and len(snapshots)==2,'missing before/after observations')
    expected_paths=[];path=stage
    while True:
        expected_paths.append(str(path))
        if path==QUOTA_ANCESTOR:break
        path=path.parent
    available=[]
    for snapshot in snapshots:
        quotas=snapshot['quota_observations']
        require([q['path'] for q in quotas]==expected_paths,'incomplete quota ancestor observations')
        constraints=[snapshot['filesystem_available_bytes']]
        require(type(constraints[0]) is int and constraints[0]>=0,'invalid filesystem availability')
        for quota in quotas:
            require(started<=_utc(quota['observed_at_utc'])<=finished,'quota observation time outside measurement')
            if quota['status']=='NO_LOCAL_LIMIT_ENODATA':
                require(quota['path']!=str(QUOTA_ANCESTOR),'missing fixed ancestor limit')
            else:
                require(quota['status']=='POSITIVE_QUOTA' and type(quota.get('max_bytes')) is int and quota['max_bytes']>0 and
                        type(quota.get('rbytes')) is int and quota['rbytes']>=0 and
                        quota.get('available_bytes')==max(0,quota['max_bytes']-quota['rbytes']),'invalid quota constraint')
                constraints.append(quota['available_bytes'])
        require(snapshot['effective_available_bytes']==min(constraints),'incorrect effective quota minimum')
        available.append(min(constraints))
    require(value.get('effective_available_bytes')==min(available) and min(available)>=budget['required_bytes'] and
            value.get('ready') is True and value.get('quota_verified') is True and value.get('writable_verified') is True and
            value.get('probe')==dict(bytes_written=len(b'SATDOSE storage writability and fsync probe\n'),
                file_fsync=True,directory_fsync=True,probe_removed=True),'insufficient/unverified writable capacity')
    return copy.deepcopy(value)


def make_storage_ready(budget_reference):
    """Return training_transaction.commit_step's required storage_ready callback.

    Budget schema: sr-satdose-storage-budget-v1; experiment_id, evidence_root,
    design_freeze_sha256, positive integer required_bytes and
    max_observation_age_seconds, explicit UTC created_at_utc/expires_at_utc.
    Extra budget accounting/rationale fields are permitted and SHA-bound.
    """
    budget_reference=copy.deepcopy(budget_reference)
    def storage_ready(stage,step):
        started=_now();tick=time.monotonic();stage,identity=_stage(stage,step)
        budget=_budget(budget_reference,started)
        path=ROOT/f"storage_readiness/start-{identity['start_seed']}/{identity['arm']}/step-{step:03d}.json"
        require(not path.exists() and not path.is_symlink(),'storage measurement already exists; no automatic retry')
        before=_observe(stage)
        require(before['effective_available_bytes']>=budget['required_bytes'],'insufficient live quota/filesystem capacity')
        probe=_probe(stage)
        after=_observe(stage);finished=_now();elapsed=time.monotonic()-tick
        _budget(budget_reference,finished)
        require(0<=elapsed<=budget['max_observation_age_seconds'] and started<=finished and
                (finished-started).total_seconds()<=budget['max_observation_age_seconds'],'stale storage observations')
        effective=min(before['effective_available_bytes'],after['effective_available_bytes'])
        require(effective>=budget['required_bytes'],'capacity decreased below required budget during probe')
        measurement=dict(schema='sr-satdose-storage-measurement-v1',experiment_id=EXPERIMENT,
            permission='NO_TRAINING_ADMISSION',ready=True,quota_verified=True,writable_verified=True,
            evidence_root=str(ROOT),quota_ancestor=str(QUOTA_ANCESTOR),stage=str(stage),identity=identity,
            design_freeze_sha256=FREEZE_SHA,budget_reference=budget_reference,required_bytes=budget['required_bytes'],
            effective_available_bytes=effective,started_at_utc=started.isoformat(),checked_at_utc=finished.isoformat(),
            elapsed_seconds=elapsed,observations=[before,after],probe=probe,
            limitation='point_in_time_shared_quota_measurement_not_a_space_reservation')
        _write_measurement(path,measurement)
        result=dict(measurement,schema='sr-satdose-storage-readiness-v1',evidence=[budget_reference,_ref(path)])
        return validate_readiness(result,budget_reference=budget_reference,stage=stage,step=step)
    return storage_ready
