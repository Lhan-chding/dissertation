#!/usr/bin/env python3
"""Recover one already-published DEV audit bundle; default is local plan only.

This is byte transfer and package consistency validation, not a new server audit.
The explicit original manifest SHA is the trust anchor. --execute uses one
read-only SSH call, retains its complete stdout/stderr even on corruption, and
never retries, overwrites old evidence, runs models, or grants a successor.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import traceback

ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
PROJECT = ROOT.parent.parent
PYTHON = '/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
AUDITOR_SHA = '7c29df29ab6115b322505c59e44bb666d1ef79bf0fe6048c111b25826e0227e7'
ORIGINAL = dict(path='execution_releases/FIRST_TWO_RELEASE_71018e5.json',
                sha256='22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f')
PERMISSION = 'DOWNLOAD_RECOVERY_ONLY_NO_AUDIT_TRAINING_MAIN_OR_SUCCESSOR_AUTHORITY'
AUDIT_PERMISSION = 'ANALYSIS_ONLY_NO_TRAINING_MAIN_OR_SUCCESSOR_AUTHORITY'
FILES = frozenset(('CLAIM.json', 'SCHEDULER.json', 'OWN_JOB_STDOUT.log', 'OWN_JOB_STDERR.log',
                  'ORIGINAL_RESULT.json', 'ORIGINAL_POST_RISK.json', 'ANALYSIS.json',
                  'RISK_REPLAY.json', 'AUDIT_REPORT.json', 'PUBLICATION.json'))
MAX_FILE_BYTES = 64 << 20
MAX_TOTAL_BYTES = 256 << 20


def require(condition, message):
    if not condition:
        raise ValueError('HOLD_DEV_AUDIT_DOWNLOAD: ' + message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def loads(data):
    def pairs(items):
        value = dict(items)
        require(len(items) == len(value), 'Duplicate JSON key')
        return value
    def constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def hash_value(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Malformed SHA256')
    return value


def clean_path(value, *, absolute):
    require(isinstance(value, str) and value and '\x00' not in value, 'Malformed path')
    path = Path(value)
    require(path.is_absolute() == absolute and '..' not in path.parts and '.' not in path.parts
            and str(path) == value, 'Noncanonical path spelling')
    return path


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and not path.is_symlink() and path.is_file(), 'Noncanonical regular input: ' + str(path))
    return path


def endpoint(start, arm, step):
    require(type(start) is int and start in (11031, 11032) and arm in ('T0', 'T1')
            and type(step) is int and step in (2, 8, 16), 'Unsupported fixed DEV endpoint')
    return dict(start_seed=start, arm=arm, step=step, responses=108)


def audit_directory(job, ep):
    return ROOT / 'execution_releases' / f"trained_dev_audit_job_{job}_s{ep['start_seed']}_{ep['arm']}_step{ep['step']:03d}"


def validate_manifest(value, job, ep):
    require(type(job) is int and job > 0, 'Invalid job ID')
    require(value.get('status') == 'ANALYSIS_PUBLICATION_COMPLETE' and value.get('success') is True
            and value.get('permission') == AUDIT_PERMISSION and value.get('job_id') == job
            and value.get('endpoint') == ep, 'Manifest does not bind completed audit/job/endpoint')
    require(isinstance(value.get('files'), dict) and set(value['files']) == FILES,
            'Only the original complete ten-file audit publication is accepted')
    directory = audit_directory(job, ep)
    for name, reference in value['files'].items():
        require(isinstance(reference, dict) and set(reference) == {'path', 'sha256', 'bytes'},
                'Malformed manifest file reference')
        require(clean_path(reference['path'], absolute=True) == directory / name, 'Foreign audit file path')
        hash_value(reference['sha256'])
        require(type(reference['bytes']) is int and 0 <= reference['bytes'] <= MAX_FILE_BYTES,
                'Invalid or oversized file length')
    require(sum(ref['bytes'] for ref in value['files'].values()) <= MAX_TOTAL_BYTES, 'Oversized bundle')
    return value


def local_input(args):
    ep = endpoint(args.start_seed, args.arm, args.step)
    path = ordinary(args.manifest)
    raw = path.read_bytes()
    require(digest(raw) == hash_value(args.manifest_sha256), 'Original manifest SHA mismatch')
    value = validate_manifest(loads(raw), args.job_id, ep)
    return path, raw, value, ep


def reference_path(reference):
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'}, 'Malformed embedded reference')
    hash_value(reference['sha256'])
    value = reference['path']
    require(isinstance(value, str), 'Malformed embedded path')
    path = clean_path(value, absolute=Path(value).is_absolute())
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(PROJECT), 'Embedded evidence escapes project namespace')
    return str(path)


def file_ref(manifest, name):
    ref = manifest['files'][name]
    return dict(path=str(Path(ref['path']).relative_to(ROOT)), sha256=ref['sha256'])


def validate_package(blobs, manifest):
    """Validate exact transferred bytes and all package references, without rescoring."""
    require(set(blobs) == FILES, 'Incomplete decoded bundle')
    for name, data in blobs.items():
        ref = manifest['files'][name]
        require(type(data) is bytes and len(data) == ref['bytes'] and digest(data) == ref['sha256'],
                'Transferred bytes differ: ' + name)
    objects = {name: loads(data) for name, data in blobs.items() if name.endswith('.json')}
    publication = objects['PUBLICATION.json']
    report = objects['AUDIT_REPORT.json']
    claim = objects['CLAIM.json']
    analysis = objects['ANALYSIS.json']
    result = objects['ORIGINAL_RESULT.json']
    risk = objects['RISK_REPLAY.json']
    job, ep = manifest['job_id'], manifest['endpoint']
    for value in (publication, report, claim):
        require(value.get('job_id') == job and value.get('endpoint') == ep
                and value.get('permission') == AUDIT_PERMISSION, 'Package identity/permission mismatch')
    require(publication.get('status') == 'ANALYSIS_ONLY_ARTIFACTS_PUBLISHED'
            and publication.get('report') == file_ref(manifest, 'AUDIT_REPORT.json')
            and publication.get('files') == {name: file_ref(manifest, name) for name in FILES - {'PUBLICATION.json'}},
            'Publication does not bind every original artifact')
    hold = risk.get('global_hold')
    require(type(hold) is bool and objects['ORIGINAL_POST_RISK.json'] == risk,
            'Complete original POST_RISK and replay objects differ')
    require(risk.get('schema') == 'sr-satdose-scientific-risk-assessment-v1'
            and risk.get('permission') == 'RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY'
            and risk.get('training_permission') is False and risk.get('main_permission') is False,
            'Risk record grants unsupported authority')
    expected_status = 'TRAINED_DEV_INDEPENDENT_AUDIT_COMPLETE' + ('_SCIENTIFIC_HOLD' if hold else '')
    require(report.get('schema') == 'sr-satdose-independent-trained-dev-audit-v1'
            and report.get('status') == expected_status and report.get('next_stage_authorized') is False
            and report.get('exact_post_risk_object_reproduced') is True
            and report.get('scientific_outcome_preserved') is True
            and report.get('original_train_release') == ORIGINAL
            and report.get('source_files') == 84 and report.get('execution_source_keys') == 19
            and report.get('responses') == 108, 'Independent audit report contract differs')
    require(report.get('operation_source_sha256') == AUDITOR_SHA
            and claim.get('operation_source_sha256') == AUDITOR_SHA
            and claim.get('status') == 'ANALYSIS_ATTEMPT_UNKNOWN_UNTIL_PUBLICATION'
            and claim.get('submission') == report.get('submission'), 'Original audited source/submission differs')
    for key in ('optimizer_steps', 'scheduler_steps', 'parameter_updates', 'audit_model_calls', 'audit_optimizer_calls'):
        require(type(report.get(key)) is int and report[key] == 0, 'Audit or DEV changed model state')
    for value in (publication, report, result):
        require(value.get('global_scientific_hold') is hold, 'Scientific HOLD changed during recovery')
    require(report.get('hold_reasons') == risk.get('hold_reasons')
            and report.get('pending_required_dev') == risk.get('pending_required_dev'), 'Risk decisions differ')
    for key, name in (('analysis', 'ANALYSIS.json'), ('original_post_risk', 'ORIGINAL_POST_RISK.json'),
                      ('risk_replay', 'RISK_REPLAY.json')):
        require(report.get(key) == file_ref(manifest, name), 'Audit report artifact reference differs: ' + key)
    slot = f"trained_dev/start-{ep['start_seed']}/{ep['arm']}/step-{ep['step']:03d}"
    require(report.get('result') == dict(path=slot + '/RESULT.json', sha256=manifest['files']['ORIGINAL_RESULT.json']['sha256'])
            and result.get('post_risk') == dict(path=slot + '/POST_RISK.json', sha256=manifest['files']['ORIGINAL_POST_RISK.json']['sha256']),
            'Original result/POST_RISK copy binding differs')
    require(result.get('schema') == 'sr-satdose-trained-dev-result-v1'
            and result.get('status') == 'TRAINED_DEV_ENDPOINT_COMPLETE'
            and result.get('permission') == 'NO_TRAINING_OR_MAIN'
            and all(result.get(key) == value for key, value in ep.items())
            and result.get('release') == report.get('release'), 'Original DEV result endpoint differs')
    summary = analysis.get('summary', {})
    rows = analysis.get('rows')
    require(isinstance(rows, list) and len(rows) == 108
            and len({row['request_id'] for row in rows}) == 108
            and all(all(row.get(key) == ep[key] for key in ('start_seed', 'arm', 'step')) for row in rows)
            and summary.get('schema') == 'sr-satdose-evaluation-analysis-v1' and summary.get('scope') == 'DEV'
            and summary.get('permission') == 'ANALYSIS_ONLY_NO_TRAINING_OR_MAIN_RELEASE'
            and summary.get('metrics', {}).get('assigned') == 108
            and all(summary.get('endpoint', {}).get(key) == ep[key] for key in ('start_seed', 'arm', 'step')),
            'Full analysis population/endpoint differs')
    key = f"{ep['start_seed']}/{ep['arm']}/{ep['step']}"
    require(risk.get('dev_analysis_evidence', {}).get(key) == summary.get('evidence')
            and risk.get('dev_endpoints', {}).get(key) == dict(receipt=result['receipt'], binding=result['endpoint_binding']),
            'Analysis evidence does not bind original risk endpoint')
    require(summary['evidence']['receipt_sha256'] == result['receipt']['sha256']
            and summary['evidence']['endpoint_binding_sha256'] == result['endpoint_binding']['sha256'],
            'Original receipt/binding differs from complete analysis')
    observed = report.get('observed_files')
    require(isinstance(observed, dict) and observed, 'Missing original observed evidence inventory')
    for path, sha256 in observed.items():
        reference_path(dict(path=path, sha256=sha256))
        require(Path(path).is_absolute(), 'Observed evidence path is not absolute')
    require(isinstance(risk.get('observed_files'), dict) and risk['observed_files']
            and all(observed.get(path) == sha256 for path, sha256 in risk['observed_files'].items()),
            'Original risk observations differ from audited observations')
    package_hashes = {ref['path']: ref['sha256'] for ref in manifest['files'].values()}
    external_refs = {}
    def walk(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}:
                path = reference_path(value)
                if path in package_hashes:
                    require(value['sha256'] == package_hashes[path], 'Package reference hash mismatch')
                else:
                    require(path not in external_refs or external_refs[path] == value['sha256'], 'Conflicting external evidence references')
                    external_refs[path] = value['sha256']
                    if path in observed:
                        require(observed[path] == value['sha256'], 'Embedded reference contradicts original observations')
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for value in objects.values():
        walk(value)
    return dict(global_scientific_hold=hold, hold_reasons=risk['hold_reasons'],
                pending_required_dev=risk['pending_required_dev'], responses=108,
                files_verified=10, exact_post_risk_object_reproduced=True,
                external_evidence_rehashed=False, external_reference_count=len(external_refs),
                limitation='Original package recovered exactly; external evidence hashes are retained and cross-checked internally, not re-read or independently re-audited.')


def read_remote_bundle(payload):
    """Read only the explicit original ten files. No mkdir/write/import/model/scheduler."""
    raw = base64.b64decode(payload['manifest_base64'], validate=True)
    require(digest(raw) == hash_value(payload['manifest_sha256']), 'Transported original manifest differs')
    ep = endpoint(payload['endpoint']['start_seed'], payload['endpoint']['arm'], payload['endpoint']['step'])
    require(payload['endpoint'] == ep, 'Transported endpoint differs')
    manifest = validate_manifest(loads(raw), payload['job_id'], ep)
    directory = audit_directory(payload['job_id'], ep)
    require(directory.resolve(strict=True) == directory and directory.is_dir(), 'Noncanonical original publication directory')
    require({p.name for p in directory.iterdir()} == FILES, 'Original publication has missing or extra entries')
    blobs = {}
    for name in sorted(FILES):
        reference = manifest['files'][name]
        path = ordinary(directory / name)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == reference['bytes'], 'Original file type/size differs')
            data = stream.read(reference['bytes'] + 1)
            after = os.fstat(stream.fileno())
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    'Original file changed while reading')
        current = path.stat()
        require(not path.is_symlink() and (current.st_dev, current.st_ino) == (before.st_dev, before.st_ino)
                and len(data) == reference['bytes'] and digest(data) == reference['sha256'], 'Original file bytes changed: ' + name)
        blobs[name] = dict(reference, data_base64=base64.b64encode(data).decode('ascii'))
    require(directory.resolve(strict=True) == directory and {p.name for p in directory.iterdir()} == FILES,
            'Original publication directory changed')
    return dict(schema='sr-satdose-dev-audit-download-bundle-v1', job_id=payload['job_id'], endpoint=ep,
                manifest_sha256=payload['manifest_sha256'], files=blobs, permission=PERMISSION)


def decode_bundle(raw, manifest, manifest_sha256):
    bundle = loads(raw)
    require(bundle.get('schema') == 'sr-satdose-dev-audit-download-bundle-v1'
            and bundle.get('permission') == PERMISSION and bundle.get('job_id') == manifest['job_id']
            and bundle.get('endpoint') == manifest['endpoint'] and bundle.get('manifest_sha256') == manifest_sha256,
            'Downloaded bundle identity differs')
    require(isinstance(bundle.get('files'), dict) and set(bundle['files']) == FILES, 'Incomplete downloaded bundle')
    blobs = {}
    for name, value in bundle['files'].items():
        require(isinstance(value, dict) and set(value) == {'path', 'sha256', 'bytes', 'data_base64'}
                and {key: value[key] for key in ('path', 'sha256', 'bytes')} == manifest['files'][name],
                'Downloaded bundle reference differs: ' + name)
        data = base64.b64decode(value['data_base64'], validate=True)
        require(len(data) == value['bytes'] and digest(data) == value['sha256'], 'Downloaded bytes differ: ' + name)
        blobs[name] = data
    return blobs


def write_once(path, data):
    require(path.parent.resolve(strict=True) == path.parent, 'Noncanonical local output parent')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data if isinstance(data, bytes) else canonical(data))
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def local_ref(path):
    raw = ordinary(path).read_bytes()
    return dict(path=str(path), sha256=digest(raw), bytes=len(raw))


def utc():
    return datetime.now(timezone.utc).isoformat()


def local(args):
    manifest_path, manifest_raw, manifest, ep = local_input(args)
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_PLAN_ONLY_NO_SSH', job_id=args.job_id, endpoint=ep,
            manifest_sha256=args.manifest_sha256, file_count=10, total_bytes=sum(r['bytes'] for r in manifest['files'].values()),
            permission=PERMISSION, remote_output_mutations=0, next_stage_authorized=False), indent=2))
        return 0
    destination = args.local_dir
    require(destination is not None and destination.is_absolute() and not os.path.lexists(destination)
            and destination.parent.resolve(strict=True) == destination.parent, 'Explicit new canonical local directory required')
    require(not destination.is_relative_to(manifest_path.parent), 'Recovery must be separate from the old audit attempt')
    destination.mkdir()
    source = Path(__file__).read_bytes()
    write_once(destination / 'OPERATION_SOURCE.py', source)
    write_once(destination / 'ORIGINAL_REMOTE_MANIFEST.json', manifest_raw)
    payload = dict(manifest_base64=base64.b64encode(manifest_raw).decode('ascii'),
                   manifest_sha256=args.manifest_sha256, job_id=args.job_id, endpoint=ep)
    encoded = base64.b64encode(canonical(payload)).decode('ascii')
    bootstrap = ('import hashlib,sys; s=sys.stdin.buffer.read(); '
                 f'assert hashlib.sha256(s).hexdigest()=={digest(source)!r}; '
                 "exec(compile(s,'<pinned-dev-audit-download-recovery>','exec'))")
    command = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
                          PYTHON, '-I', '-B', '-c', bootstrap, '--remote-read-only', encoded])
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=4', 'eee-cluster', command]
    write_once(destination / 'RECOVERY_INTENT.json', dict(status='READ_ONLY_DOWNLOAD_ATTEMPT_NO_AUTOMATIC_RETRY',
        original_manifest=local_ref(manifest_path), operation_source=local_ref(destination / 'OPERATION_SOURCE.py'),
        job_id=args.job_id, endpoint=ep, permission=PERMISSION, ssh_argv=argv, started_at_utc=utc(),
        old_evidence_modified=False, next_stage_authorized=False))
    stdout, stderr = destination / 'SSH_STDOUT.raw', destination / 'SSH_STDERR.raw'
    response = None
    try:
        with stdout.open('xb') as out, stderr.open('xb') as err:
            try:
                response = subprocess.run(argv, input=source, stdout=out, stderr=err, check=False)
            finally:
                out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
        require(response.returncode == 0, 'Read-only SSH failed; retain full raw output without retry')
        require(manifest_path.read_bytes() == manifest_raw, 'Original local manifest changed during transfer')
        blobs = decode_bundle(stdout.read_bytes(), manifest, args.manifest_sha256)
        # Raw transport is already durable; preserve exact decoded bytes before semantic checks too.
        for name in sorted(FILES):
            write_once(destination / name, blobs[name])
        validation = validate_package(blobs, manifest)
        write_once(destination / 'RECOVERY_RECEIPT.json', dict(
            schema='sr-satdose-dev-audit-download-recovery-v1', status='EXACT_ORIGINAL_AUDIT_BUNDLE_RECOVERED',
            permission=PERMISSION, original_manifest=local_ref(manifest_path), job_id=args.job_id, endpoint=ep,
            operation_source=local_ref(destination / 'OPERATION_SOURCE.py'), files={n: local_ref(destination / n) for n in sorted(FILES)},
            raw_logs=dict(stdout=local_ref(stdout), stderr=local_ref(stderr)), ssh_returncode=response.returncode,
            validation=validation, old_evidence_modified=False, remote_files_written=0, remote_audit_rerun=False,
            model_calls=0, optimizer_calls=0, next_stage_authorized=False, completed_at_utc=utc()))
        print(json.dumps(dict(status='EXACT_ORIGINAL_AUDIT_BUNDLE_RECOVERED', local_dir=str(destination),
            job_id=args.job_id, files=10, global_scientific_hold=validation['global_scientific_hold'],
            pending_required_dev=validation['pending_required_dev'], permission=PERMISSION), indent=2))
        return 0
    except BaseException as exc:
        write_once(destination / 'RECOVERY_ERROR.json', dict(status='DOWNLOAD_RECOVERY_FAILED_OR_UNKNOWN_NO_RETRY',
            permission=PERMISSION, job_id=args.job_id, endpoint=ep, error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), ssh_returncode=None if response is None else response.returncode,
            raw_logs={key: local_ref(path) for key, path in (('stdout', stdout), ('stderr', stderr)) if path.is_file()},
            old_evidence_modified=False, remote_audit_rerun=False, next_stage_authorized=False, completed_at_utc=utc()))
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == '--remote-read-only':
        payload = loads(base64.b64decode(argv[1], validate=True))
        sys.stdout.buffer.write(canonical(read_remote_bundle(payload)))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--start-seed', type=int, choices=(11031, 11032), required=True)
    parser.add_argument('--arm', choices=('T0', 'T1'), required=True)
    parser.add_argument('--step', type=int, choices=(2, 8, 16), required=True)
    parser.add_argument('--local-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    return local(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
