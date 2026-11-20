#!/usr/bin/env python3
"""Read-only recovery of ONE already-published first9 preparation after SSH255.

Default local plan; --execute downloads once to a NEW local directory. Never
runs prepare, preflight, risk, storage probes, model, optimizer or scheduler.
The recovery command's REAL stdout is publication JSON plus recovery_evidence.
Original SSH255/empty stdout remain bound and are never replaced or relabelled.
"""
import argparse
import base64
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import ModuleType

PROJECT = Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
CODE = ROOT / 'train_code_71018e5'
PYTHON = '/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
PREPARER_SHA = '737954eb256b1c29f5a381e864d3a1bc74a5fa1a8e95ab4a3373c2e66ad151c2'
SUBMITTER_SHA = 'eceec5aab4de56b53d8e1483d2e49093e28f4048b2ef818302bb3db499fb0f94'
FAILED_SHA = '7b6ef477b140c75ccb4e44717886258affdc6a75ae32787c39998d61dfd7535a'
FAILED_INTENT_SHA = 'f526d00654e700ab8b6a6082be56a3726880730a41ff561a731d656aab71527c'
INSPECTION_SHA = 'da612e986764bb89ec38b36c0d651204a38027c3c31266c865b46132a8eee327'
SELECTION = dict(start_seed=11031, arm='T0', first_step=9, last_step=16)
PREFIX = 'CONTINUATION_71018e5_s11031_T0_009_016'
LOCAL_PREFIX = 'CONTINUATION_PREPARATION_71018e5_s11031_T0_009_016'
PERMISSION = 'READ_ONLY_PREPARATION_PUBLICATION_RECOVERY_NO_MODEL_OR_GPU_SUBMISSION'
ORIGIN = 'READ_ONLY_PUBLICATION_RECOVERY'
PATHS = tuple(f'{s}/{a}' for s in (11031, 11032) for a in ('T0', 'T1'))
FIVE = {
    'INTENT': ('ec484da62c9edd1b4606399978c650d59c7506f1c546fe80efe75509b5f26c71', 11453),
    'BUDGET': ('e08304775261d2418aa2af926f5b5bdf9ea41fb4df9c80289782201ebd623745', 1229),
    'RELEASE': ('eda1f586dc7b69d1f04b132726856e1c8c282ab52b5deb520ef7134b2e36b9fd', 21947),
    'RISK': ('2414de5bf0736f9ac6690772853fcb9e0ea7c8c72758a09abae710ce67a10984', 4785853),
    'PREFLIGHT': ('d0f2f8f7f5cd6cf680d4ba85d35cfa18179cdda23f3d6715c22591ae5fb6e165', 7059),
}


def require(ok, message):
    if not ok: raise ValueError('HOLD_PREPARATION_RECOVERY: ' + message)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
def utc(): return datetime.now(timezone.utc).isoformat()


def loads(raw):
    def pairs(items):
        value = dict(items); require(len(value) == len(items), 'Duplicate JSON key/path'); return value
    def constant(value): raise ValueError('Nonfinite JSON ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
            'Noncanonical ordinary file: ' + str(path))
    return path


def sha(path): return digest(ordinary(path).read_bytes())
def reference(path): return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def pinned(ref):
    require(isinstance(ref, dict) and {'path', 'sha256'} <= set(ref)
            and re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')), 'Malformed pinned absolute reference')
    path = ordinary(Path(ref['path']))
    require(sha(path) == ref['sha256'] and ('bytes' not in ref or path.stat().st_size == ref['bytes']), 'Pinned evidence changed')
    return path


def rootref(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and isinstance(ref['path'], str)
            and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Malformed root reference')
    path = Path(ref['path'])
    require(not path.is_absolute() and '..' not in path.parts and str(path) == ref['path'], 'Escaping/noncanonical reference')
    return ROOT / path


def bound(ref):
    path = rootref(ref); require(sha(path) == ref['sha256'], 'Changed bound reference: ' + str(path)); return path


def five_path(key): return ROOT / 'execution_releases' / (PREFIX + '_' + key + '.json')
def five_ref(key): return dict(path=str(five_path(key).relative_to(ROOT)), sha256=FIVE[key][0])


def write_once(path, value):
    require(path.parent.resolve(strict=True) == path.parent, 'Noncanonical local output parent')
    raw = value if isinstance(value, bytes) else canonical(value)
    with path.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def load_submitter():
    path = Path(__file__).with_name('satdose_submit_continuation.py'); raw = ordinary(path).read_bytes()
    require(digest(raw) == SUBMITTER_SHA, 'Frozen submitter changed')
    module = ModuleType('satdose_recovered_preparation_legacy_submitter')
    module.__file__ = str(path); exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def local_payload(args):
    require(args.failed_sha256 == FAILED_SHA, 'Only this exact failed SSH255 invocation is eligible')
    path = pinned(dict(path=str(args.failed_receipt), sha256=FAILED_SHA)); failed = loads(path.read_bytes())
    require(path.name == LOCAL_PREFIX + '_LOCAL_RECEIPT.json' and failed.get('status') == 'SSH_CPU_PREPARATION_FAILED'
            and failed.get('returncode') == 255 and failed.get('stdout') == '', 'Not the exact known SSH255/empty-stdout failure')
    require(set(failed['raw_logs']) == {'stdout', 'stderr'}, 'Missing original raw failure logs')
    raw = {name: pinned(ref).read_bytes() for name, ref in failed['raw_logs'].items()}
    require(raw['stdout'] == b'' and raw['stderr'].decode(errors='replace') == failed['stderr'], 'Original raw failure logs differ')
    source = pinned(failed['local_operation_source'])
    require(sha(source) == PREPARER_SHA and failed['local_operation_source']['bytes'] == source.stat().st_size,
            'Original producer source differs')
    intent_path = pinned(dict(path=str(path.with_name(LOCAL_PREFIX + '_LOCAL_INTENT.json')), sha256=FAILED_INTENT_SHA))
    intent = loads(intent_path.read_bytes()); summary = failed['summary']
    require(summary['selection'] == SELECTION and summary['mode'] == 'WRITE_ONCE_CPU_PREPARATION'
            and summary['gpu_submission'] is False and all(intent[k] == v for k, v in summary.items())
            and intent['status'] == 'LOCAL_INTENT_UNKNOWN_UNTIL_SSH_RECEIPT'
            and intent['local_operation_source'] == failed['local_operation_source']
            and intent['script_sha256'] == PREPARER_SHA and intent['started_at_utc'] == failed['started_at_utc'],
            'Original failed intent/selection/source differs')
    submitter = load_submitter(); pins = []
    for name, expected in submitter.PINS.items():
        pin = pinned(dict(path=str(PROJECT / 'artifacts/sr_satdose_20261112/intake' / name), sha256=expected))
        pins.append(loads(pin.read_bytes()))
    deployment, execution = pins
    require(summary['original_train_release'] == submitter.ORIGINAL and deployment['release_root'] == str(CODE)
            and len(deployment['sources']) == 84 and len(execution) == 19, 'Original 84/19 identity differs')
    inspection_path = pinned(dict(path=str(PROJECT / 'artifacts/sr_satdose_20261112/intake/FIRST9_PREPARATION_READONLY_INSPECTION_20261120T0045Z.json'), sha256=INSPECTION_SHA))
    inspection_record = loads(inspection_path.read_bytes()); inspection = loads(inspection_record['stdout'])
    require(inspection_record['returncode'] == 0 and inspection['exclusive_existing_locks_acquired'] is True
            and not inspection['error_exists'] and not inspection['controller_slot_exists'] and not inspection['scheduling_slot_exists'],
            'Prior read-only inspection incomplete')
    for key, (expected, size) in FIVE.items():
        require(inspection['files'][key] == dict(path=str(five_path(key)), sha256=expected, bytes=size), 'Publication pins differ from actual inspection')
    return dict(origin=ORIGIN, selection=SELECTION, summary=summary, failed_receipt=reference(path),
                failed_intent=reference(intent_path), original_failure=failed, original_intent=intent,
                original_source=reference(source), sources=deployment['sources'], execution_sources=execution,
                original_train_release=submitter.ORIGINAL, inspection=reference(inspection_path),
                inspected_preflight=inspection['preflight'], inspected_budget=inspection['budget'])


def timestamp(text):
    value = datetime.fromisoformat(text); require(value.tzinfo is not None, 'Naive publication timestamp'); return value


def validate_documents(documents, p):
    require(set(documents) == set(FIVE), 'Exactly five published documents required')
    intent, budget, release, risk, report = (documents[k] for k in ('INTENT', 'BUDGET', 'RELEASE', 'RISK', 'PREFLIGHT'))
    require(report == p['inspected_preflight'] and budget == p['inspected_budget'], 'Publication differs from independent pinned inspection')
    require(report['status'] == 'CONTINUATION_CPU_PREFLIGHT_COMPLETE_NO_MODEL_CALLS' and report['permission'] == 'NO_GPU_SUBMISSION'
            and report['preparation_script_sha256'] == PREPARER_SHA and report['selection'] == SELECTION
            and all(type(report[k]) is int and report[k] == 0 for k in ('model_calls', 'optimizer_calls', 'gpu_jobs_submitted')),
            'Incomplete or nonzero-call original PREFLIGHT')
    for field, key in [('intent', 'INTENT'), ('storage_budget', 'BUDGET'), ('release', 'RELEASE'), ('risk', 'RISK')]:
        require(report[field] == five_ref(key), 'PREFLIGHT reference mismatch: ' + field)
    require(intent['status'] == 'CPU_PREPARATION_INTENT_UNKNOWN_UNTIL_PREFLIGHT' and intent['selection'] == SELECTION
            and intent['preparation_script_sha256'] == PREPARER_SHA and intent['model_calls'] == 0
            and intent['gpu_submission_permission'] is False and intent['original_train_release'] == p['original_train_release']
            and intent['canary_spec'] == p['summary']['canary_spec'], 'Remote intent differs from exact failed local intent')
    require(release['schema'] == 'sr-satdose-continuation-release-v1' and release['status'] == 'AUDITED_FOR_ONE_FIXED_TRAIN_CHUNK'
            and release['permission'] == 'ONE_FIXED_CHUNK_NO_MAIN' and release['selection'] == SELECTION
            and release['evidence_root'] == str(ROOT) and release['release_root'] == str(CODE)
            and release['sources'] == p['sources'] and release['original_train_release'] == p['original_train_release']
            and release['canary_spec'] == intent['canary_spec'] and release['storage_budget'] == five_ref('BUDGET'), 'Foreign release/identity')
    trusted = release['trusted_commits']; dev = release['dev_endpoints']; prior = release['prior_chunks']
    require(set(trusted) == set(PATHS) and all(len(refs) == 8 for refs in trusted.values())
            and report['cursor'] == {key: 8 for key in PATHS}, 'Exactly 32 current COMMITs required')
    expected_dev = {f'{s}/START/0' for s in (11031, 11032)} | {f'{key}/{n}' for key in PATHS for n in (2, 8)}
    require(set(dev) == expected_dev and report['completed_dev_endpoints'] == 10
            and set(prior) == {key + '/chunk-003-008' for key in PATHS} and report['completed_chunks'] == 4,
            'Exactly ten DEV endpoints and four prior chunks required')
    for field in ('trusted_commits', 'dev_endpoints', 'prior_chunks'): require(intent[field] == release[field], 'Intent/release population mismatch')
    require(risk['schema'] == 'sr-satdose-scientific-risk-assessment-v1'
            and risk['permission'] == 'RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY'
            and risk['status'] == 'NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED' and risk['global_hold'] is False
            and risk['hold_reasons'] == [] and risk['pending_required_dev'] == []
            and risk['trusted_commits'] == trusted and risk['dev_endpoints'] == dev
            and risk['clarification'] == release['risk_clarification']
            and risk['verifier'] == p['execution_sources']['risk_verifier']
            and set(risk['paths']) == set(PATHS) and all(risk['paths'][key]['completed_steps'] == 8 for key in PATHS),
            'Original risk population/source differs or HOLD/pending exists')
    require(budget['schema'] == 'sr-satdose-storage-budget-v1' and budget['scope'] == 'CONTINUATION_FIXED_CHUNK_ONLY'
            and budget['evidence_root'] == str(ROOT) and budget['required_bytes'] == report['required_bytes']
            and budget['design_freeze_sha256'] == release['design_freeze_sha256'], 'Budget scope differs')
    sizing = budget['sizing']; allowance = max(4 * sizing['max_triplet_bytes'], 1 << 20)
    calculated = 2 * (8 * (sizing['max_checkpoint_bytes'] + sizing['max_capture_bytes'] + 128 * allowance)
                      + sizing['max_checkpoint_bytes']) + (1 << 30)
    require(sizing['observed_checkpoints'] == 32 and sizing['observed_responses'] == 4096
            and sizing['new_steps'] == 8 and sizing['new_responses'] == 1024
            and sizing['per_response_allowance_bytes'] == allowance and sizing['calculated_bytes'] == calculated
            and budget['required_bytes'] == max(16 << 30, calculated) and budget['max_observation_age_seconds'] == 300,
            'Original storage sizing differs')
    start = timestamp(p['original_failure']['started_at_utc']); disconnected = timestamp(p['original_failure']['finished_at_utc'])
    created = timestamp(intent['created_at_utc']); budget_time = timestamp(budget['created_at_utc']); checked = timestamp(report['checked_at_utc'])
    require(start <= created < disconnected < budget_time < checked
            and timestamp(budget['expires_at_utc']) == budget_time + timedelta(days=7), 'Publication timeline changed or refreshed')
    for field in ('writability_probe', 'final_writability_probe'):
        require(report[field] == dict(bytes_written=44, file_fsync=True, directory_fsync=True, probe_removed=True), 'Missing original fsync probe')
    observations = [report[k] for k in ('observed_quota_before', 'observed_quota_after', 'final_quota_before', 'final_quota_after')]
    times = [timestamp(item['observed_at_utc']) for item in observations]
    require(budget_time <= times[0] <= times[1] <= times[2] <= times[3] <= checked, 'Quota publication ordering differs')
    for item in observations:
        require(type(item['effective_available_bytes']) is int and item['effective_available_bytes'] >= budget['required_bytes'], 'Original quota insufficient')
        limits = [v['available_bytes'] for v in item['quota_observations'] if v['status'] == 'POSITIVE_QUOTA']
        require(limits and item['effective_available_bytes'] == min(item['filesystem_available_bytes'], *limits)
                and all(v['status'] in ('POSITIVE_QUOTA', 'NO_LOCAL_LIMIT_ENODATA') for v in item['quota_observations']),
                'Unknown or inconsistent original quota observations')
    return dict(report=report, preflight=five_ref('PREFLIGHT'), release_path=str(five_path('RELEASE')), release_sha256=FIVE['RELEASE'][0])


def decode_files(files):
    require(isinstance(files, dict) and set(files) == set(FIVE), 'Missing/extra/UNKNOWN five-file envelope')
    documents = {}; raw_files = {}; paths = set()
    for key, (expected, size) in FIVE.items():
        item = files[key]
        require(set(item) == {'path', 'sha256', 'bytes', 'data_b64'} and item['path'] == str(five_path(key))
                and item['path'] not in paths and item['sha256'] == expected and item['bytes'] == size, 'Foreign/duplicate path or hash')
        paths.add(item['path']); raw = base64.b64decode(item['data_b64'], validate=True)
        require(len(raw) == size and digest(raw) == expected, 'Changed or truncated publication bytes')
        raw_files[key] = raw; documents[key] = loads(raw)
    return documents, raw_files


def source_inventory(p):
    require(CODE.resolve(strict=True) == CODE and CODE.is_dir(), 'Noncanonical frozen source root')
    actual = {}
    for path in CODE.rglob('*'):
        require(path.resolve(strict=True) == path and not path.is_symlink() and (path.is_file() or path.is_dir()), 'Noncanonical source entry')
        if path.is_file(): actual[str(path.relative_to(CODE))] = sha(path)
    require(actual == p['sources'] and len(actual) == 84 and len(p['execution_sources']) == 19, 'Frozen 84/19 changed')
    for ref in p['execution_sources'].values():
        path = Path(ref['path']); require(path.is_relative_to(CODE)
            and actual.get(str(path.relative_to(CODE))) == ref['sha256'], 'Original execution identity changed')


@contextmanager
def existing_lock(name, mode):
    path = ordinary(ROOT / name); fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    def stable():
        opened = os.fstat(fd); current = path.stat()
        require(path.resolve(strict=True) == path and not path.is_symlink()
                and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino), 'Original lock inode replaced')
    try:
        fcntl.flock(fd, mode | fcntl.LOCK_NB); stable(); yield stable; stable()
    finally: os.close(fd)


def guard():
    require(ROOT.resolve(strict=True) == ROOT and ROOT.is_dir(), 'Noncanonical project root')
    absent = [five_path('ERROR'), ROOT / 'HOLD.json', ROOT / 'ERROR.json',
              ROOT / 'continuation_control/start-11031/T0/chunk-009-016',
              ROOT.parent / 'scheduling/continuation_71018e5/start-11031/T0/chunk-009-016']
    require(not any(os.path.lexists(path) for path in absent), 'ERROR/HOLD or occupied actual/scheduling target')
    for family in ('training_transactions', 'training_control', 'first_two_control', 'first_two_launcher',
                   'canary_replay', 'start_dev', 'trained_dev', 'continuation_control'):
        parent = ROOT / family
        if not os.path.lexists(parent): continue
        require(parent.resolve(strict=True) == parent and parent.is_dir(), 'Noncanonical control family')
        for path in parent.rglob('*'):
            require(path.resolve(strict=True) == path and not path.is_symlink()
                    and (path.is_file() or path.is_dir())
                    and 'HOLD' not in path.name and 'ERROR' not in path.name and not path.name.startswith('.staging'),
                    'Global failed/unknown control evidence')
        relocated = {ROOT / f'canary_replay/start-{seed}/{arm}/restored/audit/original_tokens'
                     for seed in (11031, 11032) for arm in ('T0', 'T1')}
        for begin in parent.rglob('*.BEGIN.json'):
            stem = begin.name.removesuffix('.BEGIN.json')
            require((begin.parent / (stem + '.CAPTURE.json')).is_file()
                    and (begin.parent in relocated or (begin.parent / (stem + '.COMPLETE.json')).is_file()),
                    'Unknown/unclosed original generation attempt')


def exact_population(parent, leaves):
    require(parent.resolve(strict=True) == parent and parent.is_dir(), 'Missing canonical controller population')
    expected = {Path(name) for name in leaves}; prefixes = set()
    for item in expected:
        prefixes.update(item.parents); prefixes.add(item)
    prefixes.discard(Path('.'))
    for relative in prefixes:
        path = parent / relative
        require(path.resolve(strict=True) == path and path.is_dir() and not path.is_symlink(), 'Missing/noncanonical required controller slot')
    for relative in {Path('.')} | (prefixes - expected):
        wanted = {str(p.relative_to(relative)).split('/')[0] for p in expected if relative == Path('.') or relative in p.parents}
        require({p.name for p in (parent / relative).iterdir()} == wanted, 'Unknown extra controller/scheduling population')


def closed_populations():
    starts = {f'start-{s}' for s in (11031, 11032)}
    trained = {f'start-{s}/{a}/step-{n:03d}' for s in (11031, 11032) for a in ('T0', 'T1') for n in (2, 8)}
    chunks = {f'start-{s}/{a}/chunk-003-008' for s in (11031, 11032) for a in ('T0', 'T1')}
    exact_population(ROOT / 'start_dev', starts); exact_population(ROOT / 'trained_dev', trained)
    exact_population(ROOT / 'continuation_control', chunks)
    transactions = ROOT / 'training_transactions'
    require({p.name for p in transactions.iterdir()} == starts | {'TRANSACTION.lock'}, 'Unknown transaction root entry')
    for start in starts:
        require({p.name for p in (transactions / start).iterdir()} == {'T0', 'T1'}, 'Unknown transaction arm')
    for family, leaves, controller, status in [('trained_dev_71018e5', trained, 'trained_dev', 'TRAINED_DEV'),
                                               ('continuation_71018e5', chunks, 'continuation_control', 'CONTINUATION')]:
        parent = ROOT.parent / 'scheduling' / family; exact_population(parent, leaves)
        for relative in leaves:
            slot = parent / relative
            require(not os.path.lexists(slot / 'ERROR.json') and not os.path.lexists(slot / 'HOLD.json'), 'Failed/unknown scheduling attempt')
            released = loads(ordinary(slot / 'RELEASE_RESULT.json').read_bytes())
            accepted = loads(ordinary(slot / 'SUBMISSION.json').read_bytes())
            result = loads(ordinary(ROOT / controller / relative / 'RESULT.json').read_bytes())
            require(released['status'] == status + '_SUBMITTED_RELEASE_CONFIRMED'
                    and accepted['status'] == status + '_SUBMITTED_HELD_IDENTITY_VERIFIED'
                    and released['job_id'] == accepted['job_id']
                    and released['release'] == accepted['release'] == result['release'], 'Queued/unreleased/foreign prior scheduler attempt')


def current_evidence(documents, p):
    release = documents['RELEASE']; observed = {}
    def read(ref):
        path = bound(ref); observed[str(path)] = ref['sha256']; return loads(path.read_bytes())
    original = read(p['original_train_release'])
    require(original['sources'] == p['sources'] and original['execution_sources'] == p['execution_sources'], 'Original TRAIN release identity differs')
    for field in ('admission_spec', 'admission', 'risk_clarification'):
        require(release[field] == original[field], 'Original admission/risk binding changed'); read(release[field])
    freeze = read(dict(path='DESIGN_FREEZE.json', sha256=release['design_freeze_sha256']))
    require(freeze['config_sha256'] == release['config_sha256'] == original['config_sha256'], 'Original freeze/config changed')
    read(freeze['artifacts']['config'])
    canary = read(release['canary_spec'])
    require(canary['execution_sources'] == p['execution_sources'] and canary['root'] == str(ROOT), 'Foreign canary source identity')
    read(canary['published']); read(canary['report'])
    for key, refs in release['trusted_commits'].items():
        seed, arm = key.split('/'); directory = ROOT / f'training_transactions/start-{seed}/{arm}'
        require(directory.resolve(strict=True) == directory and {x.name for x in directory.iterdir()} == {f'step-{n:03d}' for n in range(1,9)}, 'Current commit cursor changed/unknown')
        for step, ref in enumerate(refs, 1):
            require(rootref(ref) == directory / f'step-{step:03d}/COMMIT.json', 'Foreign COMMIT path')
            marker = read(ref)
            require(marker['step'] == step and marker['identity']['train_seed'] == int(seed) and marker['identity']['arm'] == arm
                    and marker['sampler_state']['generated_responses'] == 128 * step, 'COMMIT identity/count differs')
    for key, refs in release['dev_endpoints'].items():
        seed, arm, step = key.split('/'); step = int(step)
        parent = ROOT / (f'start_dev/start-{seed}' if arm == 'START' else f'trained_dev/start-{seed}/{arm}/step-{step:03d}')
        result_path = ordinary(parent / 'RESULT.json'); result = loads(result_path.read_bytes()); observed[str(result_path)] = sha(result_path)
        require(result['receipt'] == refs['receipt'] and result['endpoint_binding'] == refs['binding']
                and result['responses'] == 108 and result['status'] == ('START_DEV_ENDPOINT_COMPLETE' if arm == 'START' else 'TRAINED_DEV_ENDPOINT_COMPLETE'), 'DEV not completely published')
        binding = read(refs['binding']); read(refs['receipt']); endpoint = binding['endpoint']
        require(binding['scope'] == 'DEV' and endpoint['start_seed'] == int(seed) and endpoint['arm'] == arm and endpoint['step'] == step, 'DEV binding differs')
        if arm != 'START': require(binding['checkpoint']['commit'] == release['trusted_commits'][f'{seed}/{arm}'][step-1], 'DEV checkpoint differs')
    for key, ref in release['prior_chunks'].items():
        seed, arm, chunk = key.split('/'); require(rootref(ref) == ROOT / f'continuation_control/start-{seed}/{arm}/{chunk}/RESULT.json', 'Foreign previous chunk result')
        result = read(ref)
        require(result['status'] == 'CONTINUATION_CHUNK_COMPLETE' and result['first_step'] == 3 and result['last_step'] == 8
                and result['start_seed'] == int(seed) and result['arm'] == arm and result['train_responses'] == 768
                and result['commits'] == release['trusted_commits'][f'{seed}/{arm}'][2:8], 'Prior chunk completion differs')
    return observed


def recover_remote(p):
    require(p['origin'] == ORIGIN and p['selection'] == SELECTION and p['failed_receipt']['sha256'] == FAILED_SHA
            and p['failed_intent']['sha256'] == FAILED_INTENT_SHA and p['original_source']['sha256'] == PREPARER_SHA
            and p['inspection']['sha256'] == INSPECTION_SHA, 'Foreign recovery request')
    with ExitStack() as stack:
        checks = [stack.enter_context(existing_lock(name, mode)) for name, mode in
                  [('DIAGNOSTIC_WORKER.lock', fcntl.LOCK_EX), ('FIRST_TWO_LAUNCHER.lock', fcntl.LOCK_SH),
                   ('training_transactions/TRANSACTION.lock', fcntl.LOCK_SH)]]
        guard(); closed_populations(); source_inventory(p)
        files = {key: dict(reference(five_path(key)), data_b64=base64.b64encode(ordinary(five_path(key)).read_bytes()).decode()) for key in FIVE}
        documents, raw = decode_files(files); publication = validate_documents(documents, p)
        observed = current_evidence(documents, p)
        guard(); closed_populations(); source_inventory(p)
        for key, original in raw.items(): require(ordinary(five_path(key)).read_bytes() == original, 'Original publication changed during recovery')
        for path, expected in observed.items(): require(sha(Path(path)) == expected, 'Referenced evidence changed during recovery')
        for check in checks: check()
        recovered = utc()
        require(timestamp(documents['PREFLIGHT']['checked_at_utc']) <= timestamp(recovered)
                < timestamp(documents['BUDGET']['expires_at_utc']), 'Original budget expired; recovery cannot refresh it')
        publication['recovery_evidence'] = dict(origin=ORIGIN, permission=PERMISSION, files=files,
            failed_receipt=p['failed_receipt'], failed_intent=p['failed_intent'], independent_inspection=p['inspection'],
            original_producer_source=p['original_source'], recovery_source_sha256=p['recovery_source_sha256'],
            recovered_at_utc=recovered, original_preflight_checked_at_utc=documents['PREFLIGHT']['checked_at_utc'],
            original_budget_expires_at_utc=documents['BUDGET']['expires_at_utc'],
            source_files=84, execution_source_keys=19, current_commits=32, completed_dev_endpoints=10, completed_chunks=4,
            exclusive_worker_lock_verified=True, lock_inodes_stable=True, original_error_absent=True,
            target_controller_absent=True, target_scheduling_absent=True, observed_files=observed,
            original_ssh_returncode=255, original_stdout_recovered=False, original_stdout_was_empty=True,
            prepare_invocations=0, preflight_invocations=0, risk_replays=0, model_calls=0, optimizer_calls=0,
            storage_probes=0, remote_files_written=0, gpu_jobs_submitted=0,
            limitation='Existing publication only. Original SSH255 stdout was empty and is not reconstructed. Budget time is unchanged; submitter must rerun its own real preflight before submission.')
    return publication


def validate_publication(publication, p):
    require(set(publication) == {'report', 'preflight', 'release_path', 'release_sha256', 'recovery_evidence'}, 'Incomplete/extra recovery publication fields')
    evidence = publication['recovery_evidence']; documents, raw = decode_files(evidence['files'])
    expected = validate_documents(documents, p)
    require(all(publication[k] == v for k, v in expected.items()) and evidence['origin'] == ORIGIN and evidence['permission'] == PERMISSION
            and evidence['failed_receipt'] == p['failed_receipt'] and evidence['failed_intent'] == p['failed_intent']
            and evidence['independent_inspection'] == p['inspection'] and evidence['original_producer_source'] == p['original_source']
            and evidence['recovery_source_sha256'] == p['recovery_source_sha256'], 'Recovery evidence provenance differs')
    require(evidence['original_ssh_returncode'] == 255 and evidence['original_stdout_recovered'] is False
            and evidence['original_stdout_was_empty'] is True
            and all(evidence[k] is True for k in ('exclusive_worker_lock_verified', 'lock_inodes_stable', 'original_error_absent',
                'target_controller_absent', 'target_scheduling_absent'))
            and all(type(evidence[k]) is int and evidence[k] == 0 for k in ('prepare_invocations', 'preflight_invocations', 'risk_replays',
                'model_calls', 'optimizer_calls', 'storage_probes', 'remote_files_written', 'gpu_jobs_submitted')), 'Recovery claims changed/unsafe operation')
    require(timestamp(documents['PREFLIGHT']['checked_at_utc']) <= timestamp(evidence['recovered_at_utc'])
            < timestamp(documents['BUDGET']['expires_at_utc'])
            and evidence['original_preflight_checked_at_utc'] == documents['PREFLIGHT']['checked_at_utc']
            and evidence['original_budget_expires_at_utc'] == documents['BUDGET']['expires_at_utc'], 'Publication times refreshed or inverted')
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-receipt', type=Path, required=True); parser.add_argument('--failed-sha256', required=True)
    parser.add_argument('--local-dir', type=Path); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv); p = local_payload(args); source = ordinary(Path(__file__)).read_bytes()
    p['recovery_source_sha256'] = digest(source)
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_PLAN_ONLY_NO_SSH', origin=ORIGIN, selection=SELECTION,
            failed_receipt=p['failed_receipt'], publication_hashes={k:v[0] for k,v in FIVE.items()},
            recovery_downloads=1, prepare_invocations=0, preflight_invocations=0, permission=PERMISSION), indent=2)); return 0
    directory = args.local_dir
    require(directory is not None and directory.is_absolute() and directory.parent.resolve(strict=True) == directory.parent
            and not os.path.lexists(directory), 'New canonical local directory required; never overwrite an unknown recovery')
    directory.mkdir(); write_once(directory / 'RECOVERY_SOURCE.py', source)
    write_once(directory / 'LOCAL_RECOVERY_INTENT.json', dict(payload=p, source=reference(directory / 'RECOVERY_SOURCE.py'),
               status='READ_ONLY_DOWNLOAD_ATTEMPT_UNKNOWN_UNTIL_VALIDATED', created_at_utc=utc()))
    stdout = directory / 'RECOVERY_STDOUT.log'; stderr = directory / 'RECOVERY_STDERR.log'
    bootstrap = ('import json,base64,hashlib,sys; envelope=json.loads(sys.stdin.buffer.read()); '
        "source=base64.b64decode(envelope['source_b64'],validate=True); "
        f'assert hashlib.sha256(source).hexdigest()=={digest(source)!r}; '
        "sys.argv.append(base64.b64encode(json.dumps(envelope['payload']).encode()).decode()); "
        "exec(compile(source,'<pinned-preparation-publication-recovery>','exec'))")
    command = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', PYTHON, '-I', '-B', '-c', bootstrap, '--remote-recover'])
    receipt = dict(status='READ_ONLY_RECOVERY_FAILED_OR_UNKNOWN_NO_RETRY', origin=ORIGIN, permission=PERMISSION,
        original_failed_receipt=p['failed_receipt'], original_failed_intent=p['failed_intent'], original_ssh_returncode=255,
        original_stdout_recovered=False, recovery_operation_source=reference(directory / 'RECOVERY_SOURCE.py'))
    try:
        with stdout.open('xb') as out, stderr.open('xb') as err:
            try:
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
                    '-o', 'ServerAliveCountMax=4', 'eee-cluster', command],
                    input=canonical(dict(source_b64=base64.b64encode(source).decode(), payload=p)), stdout=out, stderr=err, check=False)
            finally: out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
        receipt.update(returncode=result.returncode, returncode_role='THIS_READ_ONLY_RECOVERY_SSH_NOT_ORIGINAL_PREPARATION_SSH',
            raw_logs=dict(stdout=reference(stdout), stderr=reference(stderr)), stdout=stdout.read_bytes().decode(errors='replace'),
            stderr=stderr.read_bytes().decode(errors='replace'))
        require(result.returncode == 0, 'Recovery SSH failed/unknown; no preparation or submission authorized')
        publication = loads(stdout.read_bytes()); raw = validate_publication(publication, p)
        for key, value in raw.items(): write_once(directory / (key + '.json'), value)
        receipt.update(status='CPU_PREPARATION_PUBLICATION_RECEIPT_CAPTURED', publication=publication, summary=p['summary'],
            summary_role='UNCHANGED_ORIGINAL_PREPARATION_INTENT_NOT_CURRENT_OPERATION_MODE',
            local_operation_source=p['original_source'], local_operation_source_role='ORIGINAL_PUBLISHED_PREPARATION_PRODUCER_NOT_RECOVERY_EXECUTABLE',
            stdout_role='ACTUAL_RECOVERY_COMMAND_STDOUT_PUBLICATION_PLUS_EVIDENCE_NOT_ORIGINAL_PREPARER_STDOUT',
            original_failure=p['original_failure'], publication_validation_complete=True)
        # The legacy parser is a compatibility check only, not fresh server admission.
        write_once(directory / 'LEGACY_FORMAT_CHECK_CANDIDATE.json', receipt)
        submitter = load_submitter()
        submitter.payload_from_receipt(directory / 'LEGACY_FORMAT_CHECK_CANDIDATE.json', sha(directory / 'LEGACY_FORMAT_CHECK_CANDIDATE.json'))
        write_once(directory / 'RECOVERED_PREPARATION_RECEIPT.json', receipt)
        write_once(directory / 'RECOVERY_VALIDATION.json', dict(status='COMPLETE_READ_ONLY_RECOVERY_LEGACY_FORMAT_VERIFIED',
                   receipt=reference(directory / 'RECOVERED_PREPARATION_RECEIPT.json'), permission=PERMISSION,
                   original_ssh_returncode=255, recovery_ssh_returncode=0, original_stdout_recovered=False,
                   fresh_preflight_still_required=True, gpu_submission_authorized=False))
        print(json.dumps(dict(receipt=reference(directory / 'RECOVERED_PREPARATION_RECEIPT.json'),
            validation=reference(directory / 'RECOVERY_VALIDATION.json'), origin=ORIGIN), indent=2)); return 0
    except BaseException as exc:
        receipt.update(status='READ_ONLY_RECOVERY_FAILED_OR_UNKNOWN_NO_RETRY', legacy_format_validated=False,
            error_type=type(exc).__name__, error=str(exc),
            recovery='Preserve all bytes. Do not rerun preparation, fabricate original stdout, or submit from partial recovery.')
        receipt['retained_logs'] = {k:reference(path) for k,path in [('stdout',stdout),('stderr',stderr)] if path.is_file()}
        write_once(directory / 'RECOVERY_ERROR.json', receipt)
        print(json.dumps(dict(error=reference(directory / 'RECOVERY_ERROR.json'), status=receipt['status']), indent=2)); return 2


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--remote-recover':
        payload = loads(base64.b64decode(sys.argv[2], validate=True))
        require(digest(source) == payload['recovery_source_sha256'], 'Remote recovery source identity differs')
        print(json.dumps(recover_remote(payload), sort_keys=True, allow_nan=False), flush=True)
    else: raise SystemExit(main())
