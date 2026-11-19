#!/usr/bin/env python3
"""Run the unchanged pinned DEV auditor once, then recover its exact bundle.

Default is a local plan with zero SSH calls. --execute uses one original audit
invocation followed, only on confirmed complete publication, by the separately
pinned read-only download operation. An occupied attempt is never retried. A
download failure retains the successful audit manifest for explicit download-only
recovery; it does not authorize rerunning the audit, training, or a successor.
"""
import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
from types import ModuleType, SimpleNamespace

AUDITOR_SHA = '7c29df29ab6115b322505c59e44bb666d1ef79bf0fe6048c111b25826e0227e7'
RECOVERY_SHA = 'c168dcfc27f351a49d31a38c4680e8772271b6c08dea184a10995175722d54de'
OPERATIONS = Path(__file__).resolve().parent
AUDITOR_PATH = OPERATIONS / 'satdose_audit_trained_dev.py'
RECOVERY_PATH = OPERATIONS / 'satdose_recover_dev_audit_download.py'
PERMISSION = 'ORIGINAL_AUDIT_AND_EXACT_DOWNLOAD_ONLY_NO_TRAINING_MAIN_OR_SUCCESSOR_AUTHORITY'


def require(condition, message):
    if not condition:
        raise ValueError('HOLD_DEV_AUDIT_DRIVER: ' + message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def loads(data):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), 'Duplicate JSON key')
        return value
    def constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and not path.is_symlink(), 'Noncanonical regular file: ' + str(path))
    return path


def local_ref(path):
    data = ordinary(path).read_bytes()
    return dict(path=str(path), sha256=digest(data), bytes=len(data))


def write_once(path, value):
    require(path.parent.resolve(strict=True) == path.parent, 'Noncanonical output parent')
    data = value if isinstance(value, bytes) else canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_pinned(path, expected_sha, name):
    """Compile exactly the hash-verified bytes; never re-read code during import."""
    path = ordinary(path)
    source = path.read_bytes()
    require(digest(source) == expected_sha, 'Pinned operation source SHA differs: ' + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module, source


def prepare(args):
    auditor, auditor_source = load_pinned(AUDITOR_PATH, AUDITOR_SHA, 'satdose_original_dev_auditor_pinned')
    recovery, recovery_source = load_pinned(RECOVERY_PATH, RECOVERY_SHA, 'satdose_original_dev_download_pinned')
    require(recovery.AUDITOR_SHA == AUDITOR_SHA and recovery.ROOT == auditor.ROOT,
            'Pinned operations do not bind the same original audit/root')
    # This is the original local-only provenance validator. Never call auditor.local().
    payload = auditor.local_payload(args)
    payload['operation_source_sha256'] = AUDITOR_SHA
    recovery.endpoint(args.start_seed, args.arm, args.step)
    require(type(args.job_id) is int and args.job_id > 0 and payload['job_id'] == args.job_id,
            'Explicit job ID differs')
    return auditor, recovery, auditor_source, recovery_source, payload


def manifest_from_stdout(raw):
    prefix = b'SATDOSE_DEV_AUDIT_JSON='
    records = [line[len(prefix):] for line in raw.splitlines() if line.startswith(prefix)]
    require(len(records) == 1, 'Missing or ambiguous original audit response; preserve raw logs')
    # Preserve the actual JSON bytes rather than synthesizing a positive manifest.
    data = records[0] + b'\n'
    value = loads(data)
    require(isinstance(value, dict), 'Original audit response is not a manifest object')
    return data, value


def local(args):
    auditor, recovery, auditor_source, recovery_source, payload = prepare(args)
    ep = payload['endpoint']
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_PLAN_ONLY_NO_SSH', job_id=args.job_id, endpoint=ep,
            original_auditor_sha256=AUDITOR_SHA, recovery_sha256=RECOVERY_SHA,
            original_remote_publication=str(auditor.audit_directory(args.job_id, ep)),
            source_files=84, execution_source_keys=19, permission=PERMISSION,
            planned_audit_invocations=1, planned_bundle_downloads=1, next_stage_authorized=False), indent=2))
        return 0
    destination = args.local_dir
    require(destination is not None and destination.is_absolute() and str(destination) == str(destination.absolute())
            and '..' not in destination.parts and not os.path.lexists(destination)
            and destination.parent.resolve(strict=True) == destination.parent,
            'Explicit new canonical local attempt directory required; never overwrite or retry')
    destination.mkdir()
    audit_dir = destination / 'audit'
    audit_dir.mkdir()
    download_dir = destination / 'download'
    driver_source = ordinary(Path(__file__)).read_bytes()
    write_once(destination / 'DRIVER_SOURCE.py', driver_source)
    write_once(destination / 'ORIGINAL_AUDITOR_SOURCE.py', auditor_source)
    write_once(destination / 'RECOVERY_SOURCE.py', recovery_source)
    sources = dict(driver=local_ref(destination / 'DRIVER_SOURCE.py'),
                   original_auditor=local_ref(destination / 'ORIGINAL_AUDITOR_SOURCE.py'),
                   recovery=local_ref(destination / 'RECOVERY_SOURCE.py'))
    write_once(destination / 'DRIVER_INTENT.json', dict(
        status='ORIGINAL_AUDIT_AND_DOWNLOAD_ATTEMPT_UNKNOWN_NO_AUTOMATIC_RETRY', permission=PERMISSION,
        sources=sources, original_payload=payload, audit_directory=str(audit_dir), download_directory=str(download_dir),
        created_at_utc=utc(), original_auditor_modified=False, frozen_train_sources_modified=False,
        next_stage_authorized=False))
    encoded = base64.b64encode(canonical(payload)).decode('ascii')
    bootstrap = ('import hashlib,sys; s=sys.stdin.buffer.read(); '
                 f'assert hashlib.sha256(s).hexdigest()=={AUDITOR_SHA!r}; '
                 "exec(compile(s,'<pinned-independent-dev-auditor>','exec'))")
    command = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'OMP_NUM_THREADS=1', auditor.PYTHON,
        '-I', '-B', '-c', bootstrap, '--remote-worker', encoded])
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=4', 'eee-cluster', command]
    stdout, stderr = audit_dir / 'REMOTE_STDOUT.raw', audit_dir / 'REMOTE_STDERR.raw'
    manifest_path = audit_dir / 'REMOTE_MANIFEST.json'
    response = None
    audit_complete = False
    phase = 'ORIGINAL_AUDIT'
    try:
        with stdout.open('xb') as out, stderr.open('xb') as err:
            try:
                response = subprocess.run(argv, input=auditor_source, stdout=out, stderr=err, check=False)
            finally:
                out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
        manifest_raw, manifest = manifest_from_stdout(stdout.read_bytes())
        write_once(manifest_path, manifest_raw)
        write_once(destination / 'AUDIT_RESPONSE_RECEIPT.json', dict(
            status='ORIGINAL_AUDIT_RESPONSE_RETAINED_NOT_DOWNLOAD_CONFIRMATION', permission=PERMISSION,
            original_auditor_sha256=AUDITOR_SHA, job_id=args.job_id, endpoint=ep,
            manifest=local_ref(manifest_path), ssh_returncode=response.returncode,
            raw_logs=dict(stdout=local_ref(stdout), stderr=local_ref(stderr)), received_at_utc=utc(),
            reported_audit_success=manifest.get('success'), next_stage_authorized=False))
        require(response.returncode == 0, 'Original audit SSH did not exit successfully; no automatic audit or download retry')
        recovery.validate_manifest(manifest, args.job_id, ep)
        audit_complete = True
        phase = 'EXACT_BUNDLE_DOWNLOAD'
        # Compile the preserved pinned bytes and bind __file__ to their saved copy,
        # so recovery.local sends those exact bytes even if workspace files change.
        recovery, saved = load_pinned(destination / 'RECOVERY_SOURCE.py', RECOVERY_SHA,
                                       'satdose_saved_dev_download_pinned')
        require(saved == recovery_source, 'Preserved recovery implementation changed')
        recovery_args = SimpleNamespace(manifest=manifest_path, manifest_sha256=digest(manifest_raw),
            job_id=args.job_id, start_seed=args.start_seed, arm=args.arm, step=args.step,
            local_dir=download_dir, execute=True)
        download_stdout, download_stderr = destination / 'DOWNLOAD_DRIVER_STDOUT.log', destination / 'DOWNLOAD_DRIVER_STDERR.log'
        with download_stdout.open('x') as out, download_stderr.open('x') as err:
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    recovered = recovery.local(recovery_args)
            finally:
                out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
        require(recovered == 0, 'Pinned bundle download did not complete')
        receipt_path = ordinary(download_dir / 'RECOVERY_RECEIPT.json')
        receipt = loads(receipt_path.read_bytes())
        require(receipt.get('status') == 'EXACT_ORIGINAL_AUDIT_BUNDLE_RECOVERED'
                and receipt.get('permission') == recovery.PERMISSION and receipt.get('job_id') == args.job_id
                and receipt.get('endpoint') == ep and receipt.get('original_manifest') == local_ref(manifest_path)
                and receipt.get('operation_source', {}).get('sha256') == RECOVERY_SHA
                and receipt.get('next_stage_authorized') is False and receipt.get('remote_audit_rerun') is False,
                'Pinned download receipt does not bind this original audit')
        validation = receipt['validation']
        require(validation.get('files_verified') == 10 and validation.get('responses') == 108
                and validation.get('exact_post_risk_object_reproduced') is True,
                'Bundle was not fully verified')
        write_once(destination / 'DRIVER_RECEIPT.json', dict(
            schema='sr-satdose-original-dev-audit-driver-v1', status='ORIGINAL_AUDIT_AND_EXACT_BUNDLE_COMPLETE',
            permission=PERMISSION, job_id=args.job_id, endpoint=ep, sources=sources,
            original_audit_manifest=local_ref(manifest_path), audit_ssh_returncode=response.returncode,
            audit_raw_logs=dict(stdout=local_ref(stdout), stderr=local_ref(stderr)),
            download_driver_logs=dict(stdout=local_ref(download_stdout), stderr=local_ref(download_stderr)),
            download_receipt=local_ref(receipt_path), validation=validation,
            original_audit_invocations=1, bundle_download_invocations=1,
            source_files=84, execution_source_keys=19, original_auditor_modified=False,
            frozen_train_sources_modified=False, risk_or_scoring_modified=False, next_stage_authorized=False,
            completed_at_utc=utc()))
        print(json.dumps(dict(status='ORIGINAL_AUDIT_AND_EXACT_BUNDLE_COMPLETE', local_dir=str(destination),
            job_id=args.job_id, endpoint=ep, global_scientific_hold=validation['global_scientific_hold'],
            pending_required_dev=validation['pending_required_dev'], permission=PERMISSION), indent=2))
        return 0
    except BaseException as exc:
        write_once(destination / 'DRIVER_ERROR.json', dict(
            status='ORIGINAL_AUDIT_COMPLETE_DOWNLOAD_FAILED_OR_UNKNOWN_NO_REAUDIT' if audit_complete else
                   'ORIGINAL_AUDIT_FAILED_OR_UNKNOWN_NO_AUTOMATIC_RETRY', permission=PERMISSION,
            phase=phase, original_audit_complete=audit_complete, job_id=args.job_id, endpoint=ep, sources=sources,
            audit_ssh_returncode=None if response is None else response.returncode,
            audit_manifest=local_ref(manifest_path) if manifest_path.exists() else None,
            audit_raw_logs={key:local_ref(path) for key,path in (('stdout',stdout),('stderr',stderr)) if path.is_file()},
            retained_download_directory=str(download_dir) if download_dir.exists() else None,
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            recovery='Preserve this attempt. If the original audit completed, use the pinned download-only operation with its retained manifest and a new directory; never rerun the audit.',
            original_auditor_modified=False, frozen_train_sources_modified=False, next_stage_authorized=False,
            completed_at_utc=utc()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission-receipt', type=Path, required=True)
    parser.add_argument('--submission-sha256', required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--start-seed', type=int, choices=(11031,11032), required=True)
    parser.add_argument('--arm', choices=('T0','T1'), required=True)
    parser.add_argument('--step', type=int, choices=(2,8,16), required=True)
    parser.add_argument('--local-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    return local(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
