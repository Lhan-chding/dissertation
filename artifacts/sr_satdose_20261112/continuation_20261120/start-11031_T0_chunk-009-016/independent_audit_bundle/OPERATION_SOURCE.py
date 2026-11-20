#!/usr/bin/env python3
"""Normal continuation CPU audit with pinned namespace-package verification.

Default is local plan. Explicit --execute audits one never-audited chunk in a
new versioned namespace, then downloads its complete bundle. Any existing audit
attempt for this chunk is ineligible, irrespective of success or failure.
Job 1742958 is excluded: only its dedicated known-failure recovery may audit it.
Original auditor and verifier bytes travel on stdin; no hidden remote code file
is needed. Explicit manifest recovery is download-only and never reaudits.
"""
import base64
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType

ORIGINAL_SHA = '9379665201d95df4b7e0671779e19fe2d686a435e6f15e68e36203832db5e352'
HELPER_SHA = 'a6a3d3265da51dad54f16f7027afd2b9cf8be2ce2af2bf23ceef863fd63fa4bd'
DEPENDENCIES = {
    'original_auditor': ('satdose_audit_continuation.py', ORIGINAL_SHA, 'ORIGINAL_AUDITOR_SOURCE.py'),
    'namespace_verifier': ('satdose_recover_continuation_namespace.py', HELPER_SHA, 'NAMESPACE_VERIFIER_SOURCE.py'),
}
EXCLUDED_JOB = 1742958
REMOTE_FAMILY = 'continuation_audits_namespace_v2'


def require(ok, message):
    if not ok: raise ValueError('HOLD_CONTINUATION_AUDIT_V2: ' + message)


def dependency_bytes(value):
    require(isinstance(value, dict) and set(value) == set(DEPENDENCIES), 'Missing/extra pinned operation dependencies')
    result = {}
    for name, (_, expected, _) in DEPENDENCIES.items():
        item = value[name]
        require(set(item) == {'sha256', 'bytes', 'source_b64'} and item['sha256'] == expected
                and type(item['bytes']) is int and 0 < item['bytes'] < 1 << 20, 'Invalid dependency pin')
        raw = base64.b64decode(item['source_b64'], validate=True)
        require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == expected, 'Dependency source bytes changed')
        result[name] = raw
    return result


def local_dependencies():
    value = {}
    for name, (filename, expected, _) in DEPENDENCIES.items():
        path = Path(__file__).with_name(filename)
        require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
                'Noncanonical original dependency source')
        raw = path.read_bytes()
        value[name] = dict(sha256=expected, bytes=len(raw), source_b64=base64.b64encode(raw).decode())
    dependency_bytes(value)
    return value


def load_modules(dependencies):
    raw = dependency_bytes(dependencies); modules = {}
    for name, source in raw.items():
        module = ModuleType('satdose_normal_v2_' + name + '_' + DEPENDENCIES[name][1])
        module.__file__ = '<embedded-pinned-' + name + '>'
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        modules[name] = module
    a = modules['original_auditor']; helper = modules['namespace_verifier']
    require(helper.ORIGINAL_SHA == ORIGINAL_SHA and helper.NAMESPACE_NAMES == {'execution', 'analysis', 'runtime'},
            'Pinned verifier/original dependency contract differs')
    return a, helper, raw


def selection_gate(a, payload):
    s = payload['selection']
    require(type(payload.get('job_id')) is int and payload['job_id'] > 0 and payload['job_id'] != EXCLUDED_JOB,
            'Job 1742958 requires its dedicated known-failure recovery; no normal retry')
    require(s == a.selection(s['start_seed'], s['arm'], s['first_step']) and 'known_failure' not in payload,
            'Foreign recovery or invalid normal chunk selection')
    dependency_bytes(payload['operator_dependencies'])
    return s


def normal_directory(a, s):
    require(s == a.selection(s['start_seed'], s['arm'], s['first_step']), 'Invalid fixed chunk')
    return a.ROOT.parent / REMOTE_FAMILY / f"start-{s['start_seed']}" / s['arm'] / f"chunk-{s['first_step']:03d}-{s['last_step']:03d}"


def no_previous_attempt(a, payload, *, current_created=False):
    """Read only project audit families; another selected slot can never be retried."""
    s = selection_gate(a, payload); root = a.ROOT.parent
    require(root.resolve(strict=True) == root and root.is_dir(), 'Noncanonical project parent')
    suffix = Path(f"start-{s['start_seed']}") / s['arm'] / f"chunk-{s['first_step']:03d}-{s['last_step']:03d}"
    current = normal_directory(a, s)
    for family in root.iterdir():
        if not family.name.startswith('continuation_audits'): continue
        require(family.resolve(strict=True) == family and family.is_dir() and not family.is_symlink(),
                'Noncanonical audit family')
        path = family
        for part in suffix.parts:
            path = path / part
            if not os.path.lexists(path): break
            require(path.resolve(strict=True) == path and path.is_dir() and not path.is_symlink(),
                    'Noncanonical selected audit descendant')
        candidate = family / suffix
        if not os.path.lexists(candidate): continue
        require(current_created and candidate == current, 'Existing selected-chunk audit attempt; no normal retry')
        # The unchanged original audit creates this exact directory exclusively
        # before inventory. Its own write-once INTENT/source must already exist.
        intent = a.loads(a.ordinary(candidate / 'INTENT.json').read_bytes())
        require(intent['payload'] == payload and a.sha(candidate / 'OPERATION_SOURCE.py') == payload['operation_source_sha256'],
                'Current audit directory is not this original write-once invocation')


def configure(a, helper, raw, *, local=False):
    original_inventory = a.inventory; original_payload = a.local_payload
    original_validate_manifest = a.validate_manifest; original_validate_bundle = a.validate_bundle
    extra = {item[2] for item in DEPENDENCIES.values()}
    a.FILES = a.FILES | extra; a.SUCCESS_FILES = a.SUCCESS_FILES | extra
    a.audit_for = lambda s: normal_directory(a, s)
    a.modules_pinned = lambda sources: helper.verify_modules_pinned(sources, code=a.CODE,
        modules=sys.modules, ordinary=a.ordinary, sha=a.sha)
    def inventory(payload):
        no_previous_attempt(a, payload, current_created=True)
        supplied = dependency_bytes(payload['operator_dependencies'])
        require(supplied == raw, 'Execution dependency source differs from payload')
        directory = normal_directory(a, payload['selection'])
        for name, source in raw.items():
            path = directory / DEPENDENCIES[name][2]
            if os.path.lexists(path): require(a.ordinary(path).read_bytes() == source, 'Archived dependency changed')
            else: a.write_once(path, source)
        return original_inventory(payload)
    a.inventory = inventory
    def validate_manifest(value, payload):
        selection_gate(a, payload); original_validate_manifest(value, payload)
        for name, source in raw.items():
            ref = value['files'].get(DEPENDENCIES[name][2])
            require(not value['success'] or ref is not None, 'Successful audit omitted pinned source dependency')
            if ref is not None:
                require(ref['sha256'] == DEPENDENCIES[name][1] and ref['bytes'] == len(source),
                        'Publication dependency source bytes differ')
        return value
    def validate_bundle(path, manifest, payload, destination):
        value = original_validate_bundle(path, manifest, payload, destination)
        if manifest['success']:
            require(all((destination / DEPENDENCIES[name][2]).read_bytes() == source for name, source in raw.items()),
                    'Downloaded dependency source changed')
        return value
    a.validate_manifest = validate_manifest; a.validate_bundle = validate_bundle
    if local:
        a.__file__ = str(Path(__file__))
        def payload(args):
            value = original_payload(args)
            value['operator_dependencies'] = local_dependencies()
            selection_gate(a, value)
            return value
        a.local_payload = payload
    return a


def main(argv=None):
    dependencies = local_dependencies(); a, helper, raw = load_modules(dependencies)
    return configure(a, helper, raw, local=True).main(argv)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] in ('--remote-worker', '--remote-download'):
        # Only stdlib is loaded before both immutable source hashes are verified.
        import json
        value = json.loads(base64.b64decode(sys.argv[2], validate=True))
        payload = value if sys.argv[1] == '--remote-worker' else value['payload']
        a, helper, raw = load_modules(payload['operator_dependencies']); configure(a, helper, raw)
        selection_gate(a, payload)
        if sys.argv[1] == '--remote-download': a.download_remote(value['manifest'], payload)
        else:
            no_previous_attempt(a, payload)
            result = a.audit_remote(payload, source)
            print(a.MARKER.decode() + a.canonical(result).decode(), end='', flush=True)
            raise SystemExit(0 if result['success'] else 2)
    else: raise SystemExit(main())
