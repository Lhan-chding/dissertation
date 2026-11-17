#!/usr/bin/env python3
"""One bounded CPU recovery of the COMPLETE known namespace-auditor failure.

Only job 1742958 / 11031 T0 / chunk 3-8 and the exact retained failed publication
are eligible. Default is local plan. Explicit --execute uses the unchanged
source-pinned original audit flow with a strict namespace-package verifier in a
new immutable directory. Unknown attempts, scientific HOLD, other jobs and any
changed original evidence are ineligible. Download-only recovery never reaudits.
"""
import base64
import copy
import hashlib
import importlib.machinery
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from types import ModuleType

ORIGINAL_SHA = '9379665201d95df4b7e0671779e19fe2d686a435e6f15e68e36203832db5e352'
FAILED_MANIFEST_SHA = '4ad9830e8dcac87551462373d6f4c8d1c3e4646c6fc5850d3de3e133f6e68b1a'
FAILED_ERROR_SHA = 'd8567e00f9be83d0ad58f5f83aafc3ec0c751c238b6377ac0b9c0253c55d00cf'
FAILED_PAYLOAD_SHA = 'cb7dbd6c23f19c1f2043a81aa5c1f17cd3287d567c9945e836552c91cdcb58dd'
FAILED_LOCAL_RECEIPT_SHA = '8a5fbcb7b19ca69173a08a3c25dff780a312abb6466842f942223d425ba8cb7f'
KNOWN_JOB = 1742958
KNOWN_SELECTION = dict(start_seed=11031, arm='T0', first_step=3, last_step=8)
ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
OLD_DIRECTORY = ROOT.parent / 'continuation_audits_71018e5/start-11031/T0/chunk-003-008'
NEW_DIRECTORY = ROOT.parent / 'continuation_audits_namespace_recovery_v1_1742958/start-11031/T0/chunk-003-008'
NAMESPACE_NAMES = frozenset(('execution', 'analysis', 'runtime'))
FAILED_FILES = frozenset(('ERROR.json', 'INCIDENTS.json', 'INTENT.json', 'OPERATION_SOURCE.py',
    'ORIGINAL_CLAIM.json', 'ORIGINAL_RELEASE.json', 'ORIGINAL_RESULT.json', 'OWN_JOB_STDERR.log',
    'OWN_JOB_STDOUT.log', 'RISK_REPLAY.json', 'SCHEDULER_BEFORE.json', 'SUBMISSION_EVIDENCE.json'))
RECOVERY_FILES = frozenset(('KNOWN_FAILURE_BINDING.json', 'ORIGINAL_AUDITOR_SOURCE.py'))


def require(ok, message):
    if not ok: raise ValueError('HOLD_KNOWN_NAMESPACE_RECOVERY: ' + message)


def load_original(remote=False):
    path = OLD_DIRECTORY / 'OPERATION_SOURCE.py' if remote else Path(__file__).with_name('satdose_audit_continuation.py')
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
            'Noncanonical original auditor source')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA, 'Original immutable auditor source changed')
    module = ModuleType('satdose_known_namespace_recovery_original_' + ORIGINAL_SHA)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module, raw


def verify_modules_pinned(sources, *, code, modules, ordinary, sha):
    """Accept only three REAL canonical namespace packages, never missing modules.

    Reusable independently of this job-specific recovery. Ordinary modules still
    require their exact original SHA. No None-valued submodule is exempted.
    """
    code = Path(code)
    require(code.resolve(strict=True) == code and code.is_dir(), 'Noncanonical frozen code root')
    for name, module in tuple(modules.items()):
        if name not in NAMESPACE_NAMES and not name.startswith(('execution.', 'analysis.', 'runtime.')): continue
        file = getattr(module, '__file__', None)
        if file is not None:
            path = ordinary(Path(file))
            require(path.is_relative_to(code) and sources.get(str(path.relative_to(code))) == sha(path),
                    'Foreign/unpinned regular module: ' + name)
            continue
        require(name in NAMESPACE_NAMES, 'Missing ordinary module file: ' + name)
        package = code / name
        require(package.is_dir() and package.resolve(strict=True) == package and not package.is_symlink()
                and not os.path.lexists(package / '__init__.py') and name + '/__init__.py' not in sources
                and any(key.startswith(name + '/') for key in sources), 'Namespace not established by frozen inventory')
        spec = getattr(module, '__spec__', None); paths = getattr(module, '__path__', None)
        require(spec is not None and spec.name == name and spec.origin is None and not spec.has_location
                and getattr(module, '__name__', None) == name and getattr(module, '__package__', None) == name,
                'Not a genuine namespace module: ' + name)
        loader = getattr(module, '__loader__', None)
        require(type(loader) is importlib.machinery.NamespaceLoader and spec.loader is loader
                and paths is spec.submodule_search_locations, 'Invalid namespace loader/spec: ' + name)
        require(list(paths) == [str(package)] and list(spec.submodule_search_locations) == [str(package)],
                'Duplicate/foreign namespace search path: ' + name)


def verify_failure_manifest(a, manifest, original_payload):
    require(manifest.get('job_id') == KNOWN_JOB and manifest.get('selection') == KNOWN_SELECTION
            and manifest.get('success') is False and manifest.get('status') == 'AUDIT_BLOCKED_OR_UNKNOWN_EVIDENCE_RETAINED'
            and manifest.get('operation_source_sha256') == ORIGINAL_SHA
            and manifest.get('payload_sha256') == FAILED_PAYLOAD_SHA
            and a.digest(a.canonical(original_payload)) == FAILED_PAYLOAD_SHA
            and original_payload['operation_source_sha256'] == ORIGINAL_SHA, 'Not this exact complete known failed attempt')
    require(set(manifest['files']) == FAILED_FILES
            and manifest['files']['ERROR.json']['sha256'] == FAILED_ERROR_SHA
            and manifest['files']['OPERATION_SOURCE.py']['sha256'] == ORIGINAL_SHA, 'Partial/unknown/different failure inventory')
    # Original validator intentionally uses its original directory and file set.
    original_validator = a._recovery_original_validate_manifest if hasattr(a, '_recovery_original_validate_manifest') else a.validate_manifest
    original_audit_for = a.audit_for
    try:
        a.audit_for = lambda s: OLD_DIRECTORY
        original_validator(manifest, original_payload)
    finally: a.audit_for = original_audit_for


def validate_failed_members(a, manifest, payload, contents):
    verify_failure_manifest(a, manifest, payload)
    require(set(contents) == FAILED_FILES, 'Failed artifact population incomplete')
    for name, raw in contents.items():
        ref = manifest['files'][name]
        require(ref['path'] == str(OLD_DIRECTORY / name) and len(raw) == ref['bytes'] and a.digest(raw) == ref['sha256'],
                'Known failed member changed: ' + name)
    error = a.loads(contents['ERROR.json'])
    require(error['error_type'] == 'TypeError'
            and error['error'] == "argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'"
            and 'line 540, in audit_remote' in error['traceback'] and 'line 380, in modules_pinned' in error['traceback']
            and error['next_stage_authorized'] is False, 'Not the pinned Path(None) technical failure')
    require(a.loads(contents['INCIDENTS.json']) == {}, 'Original failure includes a scientific/technical incident')
    intent = a.loads(contents['INTENT.json']); result = a.loads(contents['ORIGINAL_RESULT.json'])
    release = a.loads(contents['ORIGINAL_RELEASE.json']); risk = a.loads(contents['RISK_REPLAY.json'])
    require(intent['payload'] == payload and result['release'] == payload['submitted']['release']
            and release['selection'] == KNOWN_SELECTION and result['status'] == 'CONTINUATION_CHUNK_COMPLETE'
            and result['train_responses'] == 768 and result['formal_optimizer_steps'] == result['scheduler_steps'] == 6
            and all(result[k] == v for k, v in KNOWN_SELECTION.items()), 'Original job/release/result differs')
    require(risk['global_hold'] is False and risk['hold_reasons'] == [] and risk['status'] == 'PENDING_REQUIRED_DEV'
            and risk['pending_required_dev'] == ['11031/T0/8']
            and a.digest(contents['RISK_REPLAY.json']) == result['post_risk']['sha256'],
            'Original known failure was not a clean completed risk replay')
    a.scheduler_complete(a.loads(contents['SCHEDULER_BEFORE.json']), KNOWN_JOB, KNOWN_SELECTION)
    a.worker_stdout(contents['OWN_JOB_STDOUT.log'], result)
    return dict(kind='KNOWN_COMPLETE_PATH_NONE_NAMESPACE_TECHNICAL_FAILURE_ONLY', job_id=KNOWN_JOB,
        selection=KNOWN_SELECTION, original_manifest_sha256=FAILED_MANIFEST_SHA, original_source_sha256=ORIGINAL_SHA,
        original_error_sha256=FAILED_ERROR_SHA, original_payload_sha256=FAILED_PAYLOAD_SHA,
        original_directory=str(OLD_DIRECTORY), recovery_directory=str(NEW_DIRECTORY),
        original_result=manifest['files']['ORIGINAL_RESULT.json'], original_release=manifest['files']['ORIGINAL_RELEASE.json'],
        original_failure_preserved=True, original_scientific_hold=False,
        complete_new_cpu_audit_required=True, original_risk_is_not_recovery_admission=True, next_stage_authorized=False)


def verify_local_failure(a, directory, expected, current_payload):
    require(expected == FAILED_MANIFEST_SHA, 'Only explicit known failed manifest SHA accepted')
    require(directory.is_absolute() and directory.resolve(strict=True) == directory and directory.is_dir(),
            'Canonical existing failed audit download directory required')
    manifest_path = a.pinned(dict(path=str(directory / 'REMOTE_MANIFEST.json'), sha256=expected))
    manifest = a.loads(manifest_path.read_bytes())
    local = a.loads(a.pinned(dict(path=str(directory / 'LOCAL_RECEIPT.json'), sha256=FAILED_LOCAL_RECEIPT_SHA)).read_bytes())
    require(local['status'] == 'BLOCKED_AUDIT_EVIDENCE_BUNDLE_VERIFIED' and local['bundle_verified'] is True
            and local['audit_success'] is False and local['remote_audit_success'] is False
            and local['manifest']['sha256'] == expected, 'Prior attempt not completely downloaded/verified')
    artifact_directory = directory / 'artifacts'
    require(artifact_directory.resolve(strict=True) == artifact_directory and artifact_directory.is_dir()
            and {p.name for p in artifact_directory.iterdir()} == FAILED_FILES, 'Incomplete failed extracted bundle')
    contents = {name: a.ordinary(artifact_directory / name).read_bytes() for name in FAILED_FILES}
    prior = a.loads(contents['INTENT.json'])['payload']
    require(prior == dict(current_payload, operation_source_sha256=ORIGINAL_SHA), 'Actual submit/job/release changed since failed audit')
    binding = validate_failed_members(a, manifest, prior, contents)
    bundle_path = a.ordinary(directory / 'BUNDLE.tar.gz')
    require(a.sha(bundle_path) == manifest['bundle']['sha256'] and bundle_path.stat().st_size == manifest['bundle']['bytes'],
            'Prior downloaded bundle changed')
    with tarfile.open(bundle_path, 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(FAILED_FILES) and {m.name for m in members} == FAILED_FILES, 'Prior tar missing/duplicate/foreign entries')
        for member in members:
            require(member.isfile() and not member.issym() and not member.islnk()
                    and member.size == len(contents[member.name])
                    and archive.extractfile(member).read() == contents[member.name], 'Prior tar differs from exact downloaded evidence')
    return dict(manifest=manifest, original_payload=prior, binding=binding,
                local_manifest=a.reference(manifest_path), local_receipt=a.reference(directory / 'LOCAL_RECEIPT.json'))


def verify_remote_failure(a, proof):
    manifest_path = a.pinned(dict(path=str(OLD_DIRECTORY / 'MANIFEST.json'), sha256=FAILED_MANIFEST_SHA))
    manifest = a.loads(manifest_path.read_bytes())
    require(manifest == proof['manifest'] and OLD_DIRECTORY.resolve(strict=True) == OLD_DIRECTORY
            and {p.name for p in OLD_DIRECTORY.iterdir()} == FAILED_FILES | {'MANIFEST.json', 'BUNDLE.tar.gz'},
            'Known failed remote attempt changed or is partial/unknown')
    contents = {name: a.pinned(ref).read_bytes() for name, ref in manifest['files'].items()}
    a.pinned(manifest['bundle'])
    binding = validate_failed_members(a, manifest, proof['original_payload'], contents)
    require(binding == proof['binding'], 'Recovery failure binding changed')
    # Keep actual original TRAIN/result/claim/release bytes unchanged, not just copies.
    s = KNOWN_SELECTION; result = a.loads(contents['ORIGINAL_RESULT.json'])
    for actual, copy_name in [(a.slot_for(s) / 'RESULT.json', 'ORIGINAL_RESULT.json'),
                             (a.slot_for(s) / 'CLAIM.json', 'ORIGINAL_CLAIM.json'),
                             (a.ROOT / result['release']['path'], 'ORIGINAL_RELEASE.json')]:
        require(a.ordinary(actual).read_bytes() == contents[copy_name], 'Actual TRAIN evidence changed after known failure')
    require(not a.incidents(s), 'Scientific or technical HOLD prevents recovery')
    return binding


def configure(a, original_source, *, local=False):
    """Only explicit operation-layer extensions; original TRAIN package is untouched."""
    original_inventory = a.inventory; original_payload = a.local_payload; original_parser = a.parser
    a._recovery_original_validate_manifest = a.validate_manifest
    original_validate_manifest = a.validate_manifest; original_validate_bundle = a.validate_bundle
    a.FILES = a.FILES | set(RECOVERY_FILES); a.SUCCESS_FILES = a.SUCCESS_FILES | set(RECOVERY_FILES)
    def recovery_directory(s):
        require(s == KNOWN_SELECTION, 'Recovery is limited to the known failed chunk')
        return NEW_DIRECTORY
    a.audit_for = recovery_directory
    a.modules_pinned = lambda sources: verify_modules_pinned(sources, code=a.CODE,
        modules=sys.modules, ordinary=a.ordinary, sha=a.sha)
    def inventory(payload):
        require(payload['job_id'] == KNOWN_JOB and payload['selection'] == KNOWN_SELECTION, 'Foreign recovery invocation')
        proof = payload['known_failure']; binding = verify_remote_failure(a, proof)
        current = {k: v for k, v in payload.items() if k not in ('known_failure', 'operation_source_sha256')}
        require(dict(current, operation_source_sha256=ORIGINAL_SHA) == proof['original_payload'],
                'Recovery does not use the exact original job/release/source identity')
        for name, raw in [('KNOWN_FAILURE_BINDING.json', a.canonical(binding)), ('ORIGINAL_AUDITOR_SOURCE.py', original_source)]:
            path = NEW_DIRECTORY / name
            if os.path.lexists(path): require(a.ordinary(path).read_bytes() == raw, 'Recovery proof changed')
            else: a.write_once(path, raw)
        return original_inventory(payload)
    a.inventory = inventory
    def validate_manifest(value, payload):
        original_validate_manifest(value, payload)
        proof = payload['known_failure']
        require(proof['binding']['original_manifest_sha256'] == FAILED_MANIFEST_SHA,
                'Recovery publication lacks the known failure binding')
        for name, raw in [('KNOWN_FAILURE_BINDING.json', a.canonical(proof['binding'])),
                          ('ORIGINAL_AUDITOR_SOURCE.py', original_source)]:
            ref = value['files'].get(name)
            require(not value['success'] or ref is not None, 'Successful recovery omitted source/failure proof')
            if ref is not None:
                require(ref['sha256'] == a.digest(raw) and ref['bytes'] == len(raw), 'Recovery proof bytes differ')
        return value
    def validate_bundle(path, manifest, payload, destination):
        value = original_validate_bundle(path, manifest, payload, destination)
        if manifest['success']:
            require((destination / 'ORIGINAL_AUDITOR_SOURCE.py').read_bytes() == original_source
                    and a.loads((destination / 'KNOWN_FAILURE_BINDING.json').read_bytes()) == payload['known_failure']['binding'],
                    'Downloaded recovery proof differs')
        return value
    a.validate_manifest = validate_manifest; a.validate_bundle = validate_bundle
    if local:
        # Original main stores/transmits this recovery source. Its local submitter
        # loader still resolves the unchanged sibling submit operator by SHA.
        a.__file__ = str(Path(__file__))
        def parser():
            p = original_parser()
            p.description = __doc__
            p.add_argument('--failed-audit-dir', type=Path, required=True)
            p.add_argument('--failed-manifest-sha256', required=True)
            return p
        def payload(args):
            value = original_payload(args)
            require(value['job_id'] == KNOWN_JOB and value['selection'] == KNOWN_SELECTION, 'Only this known failed job eligible')
            proof = verify_local_failure(a, args.failed_audit_dir, args.failed_manifest_sha256, value)
            value['known_failure'] = proof
            return value
        a.parser = parser; a.local_payload = payload
    return a


def main(argv=None):
    a, original_source = load_original()
    return configure(a, original_source, local=True).main(argv)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] in ('--remote-worker', '--remote-download'):
        a, original_source = load_original(remote=True)
        value = a.loads(base64.b64decode(sys.argv[2], validate=True))
        configure(a, original_source)
        if sys.argv[1] == '--remote-download': a.download_remote(value['manifest'], value['payload'])
        else:
            # No inherited old failed report substitutes for any original CPU gate.
            result = a.audit_remote(value, source)
            print(a.MARKER.decode() + a.canonical(result).decode(), end='', flush=True)
            raise SystemExit(0 if result['success'] else 2)
    else: raise SystemExit(main())
