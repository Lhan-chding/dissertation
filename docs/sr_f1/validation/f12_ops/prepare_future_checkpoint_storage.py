"""One-shot SR-F1.2 storage preparation; never move existing scientific outputs.

Run on a CPU allocation with CUDA hidden. Immutable receipts identify the eight
future checkpoint directories. A partial execution requires explicit inspection,
not re-running this script. Production source and the first seed remain untouched.
"""

import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010')
TARGET = Path('/projects/_hdd/varunhdd/louis-ssvc/sr_f12_20261010_checkpoints')
OPS = ROOT / 'deployment/checkpoint_storage_20261010'
CODE = ROOT / 'code_repair_r0001b'
EXPECTED = {
    'SCIENCE_FREEZE.json': 'c926f431d9976108e7bbbdbbd63d037bf2246f61d37db28a1f53ef6f1ac4dd48',
    'AMENDMENT.json': '1c79ec11f6dbab9b3c5b2dc4c5bf7c40c34c0007ecb6531c1b9899a1b8343827',
    'config/SR_F1_2_FROZEN.json': '16c20d23cd424307b26700f7b5a6086dbf44de7106358d102ba8c7bd3fa07cda',
    'source_r0001b.json': '7ee26b646ecbcf2e78e3a5dcf071620cdf2d2322aa1ddc5fb17835f53a04c047',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    with (OPS / name).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def quota(base):
    cap = int(os.getxattr(base, 'ceph.quota.max_bytes'))
    used = int(os.getxattr(base, 'ceph.dir.rbytes'))
    fs = os.statvfs(base)
    if cap <= 0:
        raise RuntimeError('A finite directory quota must be observed')
    return dict(path=str(base),limit=cap,used=used,remaining=cap-used,
                statvfs_free=fs.f_bavail*fs.f_frsize)


def effective_quota(path, base):
    limits = [quota(base)]
    for directory in (path, *path.parents):
        if directory == base:
            break
        try:
            cap = int(os.getxattr(directory, 'ceph.quota.max_bytes'))
        except OSError as error:
            if error.errno != errno.ENODATA:
                raise
            cap = 0
        if cap > 0:
            limits.append(quota(directory))
    fs = os.statvfs(path)
    return dict(limits=limits,remaining=min(x['remaining'] for x in limits),
                statvfs_free=fs.f_bavail*fs.f_frsize)


def assert_unstarted(models):
    state = read(ROOT / 'orchestration/STATE.json')
    if state['stage'] != 'MAIN_TRAINING' or state['failures'] or state['unknown_submissions']:
        raise RuntimeError('Controller is not in a known healthy training state')
    for arm in ('A','J','GATE','DEC'):
        current = 'SRF1_2_'+arm+'_s71001'
        if state['tasks'].get(current,{}).get('status') != 'RUNNING':
            raise RuntimeError('First wave must still be running')
        # Bound the operation well before a wave transition; never race submission.
        steps = list((ROOT/'runs'/current/'steps').glob('*.json'))
        if not steps or max(int(p.stem) for p in steps) >= 64:
            raise RuntimeError('Too close to the future wave; inspect manually')
    queue = subprocess.check_output(
        ['squeue', '-u', 'varun024', '-h', '-o', '%i|%j|%T|%q|%b'], text=True)
    for model in models:
        if model in state['tasks'] or model in queue:
            raise RuntimeError('Future path already registered or queued: '+model)
        if os.path.lexists(ROOT / 'orchestration/tasks' / model):
            raise RuntimeError('Future task has submission evidence: '+model)
        source = ROOT / 'runs' / model
        if os.path.lexists(source):
            raise RuntimeError('Future output path already exists: '+model)
    return dict(state=state,queue=queue)


def probe(ssd_parent, hdd_parent):
    """Run existing production commit/load regression through a real directory link."""
    from test_training import test_resume_restores_actual_adam_scheduler_cursor_parameters_and_rng
    from sr_f12.endpoint import bounded_file
    with tempfile.TemporaryDirectory(prefix='storage-probe-', dir=ssd_parent) as a:
        with tempfile.TemporaryDirectory(prefix='storage-probe-', dir=hdd_parent) as b:
            link = Path(a) / 'checkpoints'
            link.symlink_to(Path(b), target_is_directory=True)
            # Exercises fsynced state commit/hardlink, atomic latest replacement,
            # full Adam/parameters/scheduler/cursor/RNG restore and tamper rejection.
            test_resume_restores_actual_adam_scheduler_cursor_parameters_and_rng(link)
            marker = bounded_file(link, 'LATEST.json')
            assert marker.parent == Path(b).resolve()
            outside = Path(a) / 'outside.json'
            outside.write_text('{}')
            (Path(b) / 'escape.json').symlink_to(outside)
            try:
                bounded_file(link, 'escape.json')
            except PermissionError:
                pass
            else:
                raise AssertionError('Endpoint containment was bypassed')
    return dict(status='PASS',synthetic_only=True,production_commit_restore=True,
                adam_scheduler_cursor_rng=True,tamper_rejected=True,
                endpoint_escape_rejected=True,temporary_probe_cleaned=True)


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Requires CPU Slurm allocation with CUDA hidden')
    if ROOT.resolve(strict=True) != ROOT or TARGET.parent.resolve(strict=True) != TARGET.parent:
        raise RuntimeError('Unexpected canonical storage root')
    for directory in (ROOT/'runs', OPS):
        if directory.resolve(strict=True) != directory or directory.is_symlink():
            raise RuntimeError('Unexpected linked operational directory')
    for name, expected in EXPECTED.items():
        assert sha(ROOT/name) == expected, name
    manifest = read(ROOT/'source_r0001b.json')
    assert all(sha(CODE/name)==checksum for name,checksum in manifest['files'].items())
    sys.path.insert(0, str(CODE/'src'))
    sys.path.insert(0, str(CODE/'tests/sr_f12'))
    from sr_f12.protocol import scientific_matrix
    models = [v['model_id'] for v in scientific_matrix() if v['seed'] in (71002,71003)]
    assert len(models)==8 and len(set(models))==8
    before = assert_unstarted(models)
    ssd = effective_quota(ROOT, ROOT.parents[1])
    hdd = effective_quota(TARGET.parent, TARGET.parents[1])
    assert min(hdd['remaining'],hdd['statvfs_free'])>100_000_000_000
    assert min(ssd['remaining'],ssd['statvfs_free'])>15_000_000_000
    assert not os.path.lexists(TARGET)
    assert not (OPS/'INTENT.json').exists()
    save('INTENT.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        job_id=os.environ['SLURM_JOB_ID'],script_sha256=sha(Path(__file__)),
        source_files_verified=len(manifest['files']),frozen_file_sha256=EXPECTED,
        ssd_quota=ssd,hdd_quota=hdd,before=before,
        mappings={str(ROOT/'runs'/m/'checkpoints'):str(TARGET/m) for m in models}))
    TARGET.mkdir(mode=0o700)
    result = probe(OPS, TARGET)
    save('PROBE.json',result)
    # Recheck after the probe, immediately before changing unused output paths.
    assert_unstarted(models)
    applied = {}
    for model in models:
        if os.path.lexists(ROOT/'orchestration/tasks'/model):
            raise RuntimeError('Future submission appeared; stop before directory creation')
        target = TARGET/model
        target.mkdir(mode=0o700)
        source = ROOT/'runs'/model
        source.mkdir()
        link = source/'checkpoints'
        link.symlink_to(target, target_is_directory=True)
        assert link.is_symlink() and link.resolve(strict=True)==target
        assert not list(target.iterdir())
        applied[str(link)] = str(target)
    for name, expected in EXPECTED.items():
        assert sha(ROOT/name)==expected
    save('APPLIED.json',dict(status='PREPARED_EMPTY_FUTURE_CHECKPOINTS',
        at=datetime.datetime.now(datetime.timezone.utc).isoformat(),mappings=applied,
        current_seed_untouched=True,scientific_files_moved=0,source_changed=False,
        frozen_file_sha256=EXPECTED,probe=result,
        monitor_requirement='Explicitly traverse mapped targets; generic rglob may omit symlinks.'))
    print('FUTURE_CHECKPOINT_STORAGE_PREPARED',len(applied),flush=True)


if __name__ == '__main__':
    main()
