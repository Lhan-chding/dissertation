#!/usr/bin/env python3
"""Audit one completed fixed TRAIN chunk, without granting a successor.

Default: local plan, zero SSH. --execute: one CPU audit followed by one complete
bundle download. --recover-manifest/--recover-sha256: download-only recovery into
a NEW local directory; never rerun an unknown or completed remote audit.
No model construction, optimizer, update, scheduler mutation or cleanup exists.
"""
import argparse
import base64
from contextlib import contextmanager, ExitStack
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import traceback
from types import ModuleType

WORKSPACE = Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
CODE = ROOT / 'train_code_71018e5'
PYTHON = '/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
SUBMITTER_SHA = 'eceec5aab4de56b53d8e1483d2e49093e28f4048b2ef818302bb3db499fb0f94'
QOS = 'soujanya-poria-startfund-2026-03'
PERMISSION = 'INDEPENDENT_CHUNK_AUDIT_ONLY_NO_TRAINING_MAIN_OR_SUCCESSOR'
CHUNKS = {3: 8, 9: 16, 17: 24}
MARKER = b'SATDOSE_CONTINUATION_AUDIT_JSON='
LIMITATION = ('Slurm parent/all expected steps exited 0:0 and existing cooperative leases were idle. '
              'No direct compute-node /proc process-death observation. Historical intermediate risk reports '
              'are hash/structure/prefix verified, not independently replayed against truncated history. '
              'Bundle contains this complete audit publication and observed hashes; large original TRAIN '
              'raw/state files remain on the server and were verified there.')
FILES = {'INTENT.json', 'OPERATION_SOURCE.py', 'INCIDENTS.json', 'SCHEDULER_BEFORE.json', 'SCHEDULER_AFTER.json',
         'OWN_JOB_STDOUT.log', 'OWN_JOB_STDERR.log', 'SUBMISSION_EVIDENCE.json', 'ORIGINAL_RESULT.json',
         'ORIGINAL_CLAIM.json', 'ORIGINAL_RELEASE.json', 'RISK_REPLAY.json', 'AUDIT_REPORT.json',
         'ERROR.json', 'PUBLICATION.json', 'MANIFEST.json', 'BUNDLE.tar.gz'}
SUCCESS_FILES = FILES - {'ERROR.json', 'MANIFEST.json', 'BUNDLE.tar.gz'}


def require(ok, message):
    if not ok:
        raise ValueError('HOLD_CONTINUATION_AUDIT: ' + message)


def utc(): return datetime.now(timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def loads(data):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), 'Duplicate JSON key')
        return value
    def constant(value): raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and not path.is_symlink(), 'Noncanonical ordinary file: ' + str(path))
    return path


def sha(path):
    h = hashlib.sha256()
    with ordinary(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def reference(path):
    path = ordinary(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def pinned(ref):
    require(isinstance(ref, dict) and {'path', 'sha256'} <= set(ref)
            and re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')), 'Malformed absolute reference')
    path = ordinary(Path(ref['path']))
    require(sha(path) == ref['sha256'] and ('bytes' not in ref or path.stat().st_size == ref['bytes']),
            'Pinned bytes changed: ' + str(path))
    return path


def write_once(path, value):
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, 'Noncanonical output parent')
    raw = value if isinstance(value, bytes) else canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    syncdir(path.parent)


def syncdir(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def selection(start, arm, first):
    require(type(start) is int and start in (11031, 11032) and arm in ('T0', 'T1')
            and type(first) is int and first in CHUNKS, 'Invalid fixed chunk selection')
    return dict(start_seed=start, arm=arm, first_step=first, last_step=CHUNKS[first])


def slot_for(s): return ROOT / 'continuation_control' / f"start-{s['start_seed']}" / s['arm'] / f"chunk-{s['first_step']:03d}-{s['last_step']:03d}"
def schedule_for(s): return ROOT.parent / 'scheduling/continuation_71018e5' / f"start-{s['start_seed']}" / s['arm'] / slot_for(s).name
def audit_for(s): return ROOT.parent / 'continuation_audits_71018e5' / f"start-{s['start_seed']}" / s['arm'] / slot_for(s).name
def job_name(s): return f"LH-SATDOSE-TRAIN-{s['start_seed']}-{s['arm']}-{s['first_step']:03d}-{s['last_step']:03d}"


def marker(raw, prefix):
    records = [line[len(prefix):] for line in raw.splitlines() if line.startswith(prefix)]
    require(len(records) == 1, 'Missing or ambiguous response marker')
    return loads(records[0])


def load_submitter():
    path = Path(__file__).with_name('satdose_submit_continuation.py')
    raw = ordinary(path).read_bytes()
    require(digest(raw) == SUBMITTER_SHA, 'Original submitter source changed')
    module = ModuleType('satdose_continuation_submitter_readonly_pinned')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def local_payload(args):
    s = selection(args.start_seed, args.arm, args.first_step)
    require(type(args.job_id) is int and args.job_id > 0, 'Positive explicit job ID required')
    path = pinned(dict(path=str(args.submission_receipt), sha256=args.submission_sha256))
    receipt = loads(path.read_bytes())
    require(receipt.get('status') == 'SUBMISSION_RESPONSE_RETURNED' and receipt.get('returncode') == 0,
            'Actual successful submission receipt required')
    require(receipt['source']['sha256'] == SUBMITTER_SHA, 'Unexpected submitter source')
    pinned(receipt['source'])
    require(set(receipt['raw_logs']) == {'stdout', 'stderr'}, 'Incomplete original raw logs')
    raw = {k: pinned(v).read_bytes() for k, v in receipt['raw_logs'].items()}
    require(marker(raw['stdout'], b'SATDOSE_SUBMIT_JSON=') == receipt['response']
            and raw['stdout'].decode(errors='replace') == receipt['stdout']
            and raw['stderr'].decode(errors='replace') == receipt['stderr'], 'Submit raw bytes/receipt disagree')
    intent_path = ordinary(path.with_name(path.name + '.intent.json'))
    intent = loads(intent_path.read_bytes())
    require(intent['source'] == receipt['source'] and intent['status'] == 'SSH_SUBMISSION_ATTEMPT_UNKNOWN_NO_RETRY',
            'Submit local intent differs')
    submitter = load_submitter()
    prior = intent['payload']
    prep = prior['preparation_local']
    original_payload = submitter.payload_from_receipt(Path(prep['path']), prep['sha256'])
    require(prior == dict(original_payload, source_sha256=SUBMITTER_SHA), 'Actual preparation/source lineage differs')
    require(all(original_payload[k] == v for k, v in s.items()), 'Explicit selection differs from preparation')
    response = receipt['response']; submitted = response['result']
    require(submitted.get('status') == 'CONTINUATION_SUBMITTED_RELEASE_CONFIRMED'
            and type(submitted.get('job_id')) is int and submitted['job_id'] == args.job_id
            and submitted['selection'] == s and submitted['job_name'] == job_name(s)
            and submitted['release'] == original_payload['preparation']['release']
            and submitted.get('other_jobs_modified') is False, 'Foreign/unknown submission or chunk')
    require(response['receipt']['path'] == str(schedule_for(s) / 'RELEASE_RESULT.json')
            and re.fullmatch('[0-9a-f]{64}', response['receipt']['sha256']), 'Relocated scheduler receipt')
    return dict(selection=s, job_id=args.job_id, submitted=submitted, submission_ref=response['receipt'],
                preparation_payload=original_payload, local_submission=reference(path),
                local_submit_intent=reference(intent_path), submitter_sha256=SUBMITTER_SHA)


def inventory(payload):
    expected = payload['preparation_payload']['sources']
    require(CODE.resolve(strict=True) == CODE and CODE.is_dir(), 'Noncanonical immutable code root')
    actual = {}
    for path in CODE.rglob('*'):
        require(path.resolve(strict=True) == path and not path.is_symlink(), 'Noncanonical original source')
        if path.is_file(): actual[str(path.relative_to(CODE))] = sha(path)
        else: require(path.is_dir(), 'Nonordinary code entry')
    require(actual == expected and len(actual) == 84, 'Original 84 source inventory changed')
    sources = payload['preparation_payload']['execution_sources']
    require(len(sources) == 19, 'Original execution identity is not 19 keys')
    for ref in sources.values():
        path = pinned(ref)
        require(path.is_relative_to(CODE) and actual[str(path.relative_to(CODE))] == ref['sha256'],
                'Original execution source identity differs')
    return actual


@contextmanager
def idle_lock(path):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    def stable():
        current = path.stat(); opened = os.fstat(fd)
        require(path.resolve(strict=True) == path and not path.is_symlink()
                and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino), 'Replaced original lease')
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB); stable()
        yield stable
        stable()
    finally: os.close(fd)


def run_readonly(argv):
    require(argv[0] in ('sacct', 'squeue'), 'Only read-only scheduler queries permitted')
    result = subprocess.run(argv, capture_output=True, check=False,
                            env={k: v for k, v in os.environ.items() if k != 'SLURM_CLUSTERS'})
    return dict(argv=argv, returncode=result.returncode, stdout_b64=base64.b64encode(result.stdout).decode(),
                stderr_b64=base64.b64encode(result.stderr).decode(), observed_at_utc=utc())


def scheduler(job):
    return dict(accounting=run_readonly(['sacct', '-j', str(job), '-n', '-P',
                '--format=JobIDRaw,State%40,ExitCode,JobName%120,Account%40,QOS%80,Partition,Start,End,NodeList']),
                queue=run_readonly(['squeue', '-j', str(job), '-h', '-o', '%i|%j|%T|%R']))


def scheduler_complete(value, job, s, node=None):
    require(value['accounting']['returncode'] == 0, 'Scheduler accounting observation failed')
    queue = value['queue']; queue_stdout = base64.b64decode(queue['stdout_b64'], validate=True)
    queue_stderr = base64.b64decode(queue.get('stderr_b64', ''), validate=True)
    require(not queue_stdout.strip(), 'Own job remains active')
    require(queue['returncode'] == 0 or (queue['returncode'] == 1 and queue_stdout == b''
            and queue_stderr == b'slurm_load_jobs error: Invalid job id specified\n'),
            'Queue query failed for a reason other than the exact purged-job response')
    raw = base64.b64decode(value['accounting']['stdout_b64'], validate=True).decode()
    rows = [line.strip().split('|') for line in raw.splitlines() if line.strip()]
    require(all(len(row) == 10 for row in rows), 'Malformed accounting records')
    ids = [row[0] for row in rows]
    require(len(set(ids)) == len(ids) and set(ids) == {str(job), f'{job}.batch', f'{job}.extern', f'{job}.0'},
            'Missing/duplicate/foreign scheduler parent or actual steps')
    parent = next(row for row in rows if row[0] == str(job))
    require(parent[3:7] == [job_name(s), 'rose', QOS, 'cluster02'], 'Own parent scheduler identity differs')
    for row in rows:
        require(row[1:3] == ['COMPLETED', '0:0'] and all(row[i] not in ('', 'Unknown', 'None') for i in (7, 8, 9)),
                'Failed/unknown/incomplete parent or child step')
        require(row[9] == parent[9], 'Scheduler steps moved nodes')
    require(node is None or parent[9] == node, 'CLAIM node differs from terminal accounting')
    return sorted(rows, key=lambda row: row[0])


def validate_process(value):
    require(isinstance(value, dict) and set(value) == {'host', 'boot_id', 'pid', 'start_ticks'}
            and isinstance(value['host'], str) and value['host']
            and re.fullmatch('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', value['boot_id'])
            and type(value['pid']) is int and value['pid'] > 0
            and type(value['start_ticks']) is int and value['start_ticks'] > 0, 'Invalid process incarnation')


def check_claim(claim, result, payload):
    s = payload['selection']; job = payload['job_id']; release_ref = payload['submitted']['release']
    require(claim.get('schema') == 'sr-satdose-continuation-claim-v1'
            and claim.get('status') == 'ATTEMPTED_UNKNOWN_NO_RETRY'
            and claim['selection'] == s and claim['release'] == release_ref
            and claim['initial_commit'] == result['initial_commit'] and claim['process'] == result['process'],
            'CLAIM/result/invocation differs')
    validate_process(result['process'])
    allocation = claim['allocation']; fields = allocation['observation']
    require(allocation['job_id'] == str(job) and allocation['step_id'] == '0'
            and allocation['task_count'] == 1 and allocation['qos'] == QOS, 'Wrong actual worker allocation')
    expected = dict(JobId=str(job), JobName=job_name(s), JobState='RUNNING', Account='rose', QOS=QOS,
                    Partition='cluster02', NumNodes='1', NumCPUs='8', NumTasks='1', Requeue='0',
                    Command=str(CODE / 'scheduling/run_continuation.sh'), WorkDir=str(CODE),
                    StdOut=str(schedule_for(s) / f'slurm-{job}.out'), StdErr=str(schedule_for(s) / f'slurm-{job}.err'))
    require(all(fields.get(k) == v for k, v in expected.items())
            and fields.get('TimeLimit') in ('1-00:00:00', '24:00:00')
            and fields.get('UserId', '').startswith('varun024('), 'Actual worker identity/resources differ')
    tres = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    require(all(tres.get(k) == v for k, v in {'cpu': '8', 'mem': '66G', 'node': '1',
            'gres/gpu': '2', 'gres/gpu:pro6000': '2'}.items()), 'Actual worker TRES differs')
    require(result['process']['host'].split('.')[0] == fields['NodeList'], 'Process host differs from allocation')
    return fields['NodeList']


def validate_result(result, release, s):
    count = s['last_step'] - s['first_step'] + 1
    require(result.get('schema') == 'sr-satdose-continuation-result-v1'
            and result.get('status') == 'CONTINUATION_CHUNK_COMPLETE'
            and result.get('permission') == 'NO_AUTOMATIC_SUCCESSOR_OR_MAIN'
            and all(result.get(k) == v for k, v in s.items())
            and result.get('process_exit_observed') is False and result.get('main_authorized') is False,
            'Incomplete/foreign chunk or invented exit/MAIN authority')
    for name, expected in [('train_responses', 128 * count), ('formal_optimizer_steps', count), ('scheduler_steps', count)]:
        require(type(result.get(name)) is int and result[name] == expected, 'Incorrect actual fixed chunk budget')
    require(len(result['commits']) == len(result['risk_assessments']) == count
            and result['post_risk'] == result['risk_assessments'][-1]
            and result['original_train_release'] == release['original_train_release'], 'Incomplete commit/risk/source references')
    key = f"{s['start_seed']}/{s['arm']}"
    require(result['initial_commit'] == release['trusted_commits'][key][-1], 'Wrong initial checkpoint')
    trusted = copy.deepcopy(release['trusted_commits']); trusted[key].extend(result['commits'])
    require(len(trusted[key]) == s['last_step'], 'Wrong completed cursor')
    return trusted


def historical_risks(c, result, release, original, s):
    expected = copy.deepcopy(release['trusted_commits']); key = f"{s['start_seed']}/{s['arm']}"
    evidence = []
    for index, step in enumerate(range(s['first_step'], s['last_step'] + 1)):
        ref = result['risk_assessments'][index]
        require(c.bound(ref) == slot_for(s) / f'RISK_AFTER_STEP_{step:03d}.json', 'Relocated historical risk')
        value = c.read(ref); expected[key].append(result['commits'][index])
        require(value.get('schema') == 'sr-satdose-scientific-risk-assessment-v1'
                and value.get('permission') == 'RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY'
                and value.get('status') in ('NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED', 'PENDING_REQUIRED_DEV')
                and value.get('global_hold') is False and value.get('hold_reasons') == []
                and value['trusted_commits'] == expected and value['dev_endpoints'] == release['dev_endpoints']
                and value['verifier'] == original['execution_sources']['risk_verifier']
                and value['clarification'] == release['risk_clarification'], 'Historical risk foreign/incomplete/HOLD')
        evidence.append(dict(step=step, reference=ref, verification='BOUND_SHA_STRUCTURE_EXACT_PREFIX_NOT_FULL_REPLAY'))
    return evidence


def capture_population(c, result, s):
    """Exactly the committed draws; extra completed or orphan captures are UNKNOWN."""
    expected = {suffix: set() for suffix in ('BEGIN', 'CAPTURE', 'COMPLETE')}
    for ref in result['commits']:
        marker_value = c.read(ref); parent = c.bound(ref).parent
        evidence = c.tx.read_json(c.tx.bound(parent, marker_value['transaction']['evidence']))
        rollout = c.read(evidence['rollout_receipt'])
        require(len(rollout['complete_files']) == 128, 'Committed step must contain exactly 128 captures')
        for complete_ref in rollout['complete_files']:
            complete = c.bound(complete_ref); value = c.read(complete_ref)
            capture = pinned(dict(path=value['native_capture_path'], sha256=value['native_capture_sha256']))
            require(capture.is_relative_to(slot_for(s) / 'audit') and complete.is_relative_to(slot_for(s) / 'audit'),
                    'Chunk capture escaped its original audit directory')
            begin = capture.with_name(capture.name.replace('.CAPTURE.json', '.BEGIN.json'))
            for suffix, path in [('BEGIN', begin), ('CAPTURE', capture), ('COMPLETE', complete)]:
                ordinary(path); require(path not in expected[suffix], 'Duplicated committed draw evidence')
                expected[suffix].add(path)
    for suffix, paths in expected.items():
        actual = set((slot_for(s) / 'audit').rglob('*.' + suffix + '.json'))
        require(actual == paths and len(paths) == result['train_responses'], 'Extra/missing/orphan chunk draw evidence')
    return {k: len(v) for k, v in expected.items()}


def submission_evidence(payload):
    s = payload['selection']; directory = schedule_for(s); job = payload['job_id']
    require(directory.resolve(strict=True) == directory and directory.is_dir(), 'Noncanonical own scheduling directory')
    allowed = {'INTENT.json', 'RESPONSE.json', 'SUBMISSION.json', 'RELEASE_INTENT.json',
               'RELEASE_RESPONSE.json', 'RELEASE_RESULT.json', f'slurm-{job}.out', f'slurm-{job}.err'}
    require({p.name for p in directory.iterdir()} == allowed, 'Missing/extra/ERROR own submission evidence')
    data = {name: loads(ordinary(directory / name).read_bytes()) for name in allowed if name.endswith('.json')}
    require(pinned(payload['submission_ref']) == directory / 'RELEASE_RESULT.json'
            and data['RELEASE_RESULT.json'] == payload['submitted'], 'Actual server submission differs')
    result = data['RELEASE_RESULT.json']; accepted = data['SUBMISSION.json']; intent = data['INTENT.json']
    require(pinned(result['submission']) == directory / 'SUBMISSION.json'
            and pinned(result['release_response']) == directory / 'RELEASE_RESPONSE.json'
            and pinned(accepted['intent']) == directory / 'INTENT.json'
            and pinned(accepted['response']) == directory / 'RESPONSE.json', 'Broken scheduler reference chain')
    require(accepted['status'] == 'CONTINUATION_SUBMITTED_HELD_IDENTITY_VERIFIED'
            and accepted['job_id'] == job and accepted['job_name'] == job_name(s)
            and accepted['selection'] == s and accepted['release'] == result['release']
            and intent['selection'] == s and intent['release'] == result['release']
            and intent['source_sha256'] == SUBMITTER_SHA
            and intent['preparation'] == payload['preparation_payload']['preparation_ref']
            and accepted['preparation'] == intent['preparation'], 'Submit selection/source/preparation differs')
    expected_tail = [str(CODE / 'scheduling/run_continuation.sh'), str(ROOT / result['release']['path']),
                     result['release']['sha256'], str(s['start_seed']), s['arm'], str(s['first_step'])]
    require(intent['args'][-6:] == expected_tail and '--hold' in intent['args'] and '--no-requeue' in intent['args']
            and '--output=' + str(directory / 'slurm-%j.out') in intent['args']
            and '--error=' + str(directory / 'slurm-%j.err') in intent['args'], 'Submitted command/log target differs')
    response = data['RESPONSE.json']; release_intent = data['RELEASE_INTENT.json']; released = data['RELEASE_RESPONSE.json']
    require(response['returncode'] == 0 and response['stdout'].strip() == str(job)
            and response['argv'] == intent['args'] and release_intent['job_id'] == job
            and pinned(release_intent['submission']) == directory / 'SUBMISSION.json'
            and release_intent['args'] == released['argv'] == ['scontrol', 'release', str(job)]
            and released['returncode'] == 0, 'Scheduler acceptance/release differs')
    return dict(records=data, files={p.name: reference(p) for p in sorted(directory.iterdir())})


def modules_pinned(sources):
    for name, module in tuple(sys.modules.items()):
        if name in ('execution', 'analysis', 'runtime') or name.startswith(('execution.', 'analysis.', 'runtime.')):
            path = ordinary(Path(getattr(module, '__file__', '')))
            require(path.is_relative_to(CODE) and sources.get(str(path.relative_to(CODE))) == sha(path),
                    'Shadowed frozen namespace: ' + name)


def worker_stdout(raw, result):
    matches = []
    for line in raw.splitlines():
        if not line.startswith(b'{'): continue
        try: value = loads(line)
        except ValueError: continue
        if isinstance(value, dict) and value.get('status') in ('CONTINUATION_CHUNK_COMPLETE', 'HOLD_CONTINUATION_CONTROL'):
            matches.append(value)
    require(matches == [result], 'Own job stdout does not bind exactly one completed chunk')


def submission_population(release, s):
    """Reject a queued/unknown successor even before it creates a worker slot."""
    expected = {
        'trained_dev_71018e5': {f'start-{k.split("/")[0]}/{k.split("/")[1]}/step-{int(k.split("/")[2]):03d}'
                              for k in release['dev_endpoints'] if '/START/' not in k},
        'continuation_71018e5': {f'start-{k}' for k in release['prior_chunks']} |
                              {str(slot_for(s).relative_to(ROOT / 'continuation_control'))}}
    for family, names in expected.items():
        parent = ROOT.parent / 'scheduling' / family
        require(parent.resolve(strict=True) == parent and parent.is_dir(), 'Missing/canonical scheduling population')
        found = set()
        for sd in parent.iterdir():
            require(sd.name in ('start-11031', 'start-11032') and sd.is_dir() and sd.resolve(strict=True) == sd, 'Unknown scheduling seed')
            for ad in sd.iterdir():
                require(ad.name in ('T0', 'T1') and ad.is_dir() and ad.resolve(strict=True) == ad, 'Unknown scheduling arm')
                for directory in ad.iterdir():
                    key = str(directory.relative_to(parent)); require(key in names and directory.resolve(strict=True) == directory
                        and directory.is_dir(), 'Active/unknown/foreign successor submission')
                    ordinary(directory / 'SUBMISSION.json'); final = loads(ordinary(directory / 'RELEASE_RESULT.json').read_bytes())
                    status = 'CONTINUATION_SUBMITTED_RELEASE_CONFIRMED' if family.startswith('continuation') else 'TRAINED_DEV_SUBMITTED_RELEASE_CONFIRMED'
                    require(final.get('status') == status and not os.path.lexists(directory / 'ERROR.json')
                            and not os.path.lexists(directory / 'HOLD.json'), 'Incomplete or failed submitted population')
                    found.add(key)
        require(found == names, 'Missing submitted population')


def incidents(s):
    values = {}
    paths = [ROOT / name for name in ('HOLD.json', 'ERROR.json', 'training_control/HOLD.json', 'training_control/ERROR.json')]
    slot = slot_for(s)
    if os.path.lexists(slot):
        require(slot.resolve(strict=True) == slot and slot.is_dir(), 'Noncanonical chunk slot')
        paths += [p for p in slot.rglob('*') if 'HOLD' in p.name or 'ERROR' in p.name]
    for path in paths:
        if os.path.lexists(path): values[str(path)] = dict(reference=reference(path), value=loads(path.read_bytes()))
    return values


def mkdir_new(path):
    require(path.is_absolute() and '..' not in path.parts, 'Invalid output directory')
    missing = []; parent = path.parent
    while not parent.exists(): missing.append(parent); parent = parent.parent
    require(parent.resolve(strict=True) == parent and parent.is_dir(), 'Noncanonical output ancestor')
    for item in reversed(missing): item.mkdir(); syncdir(item.parent)
    require(path.parent.resolve(strict=True) == path.parent and not os.path.lexists(path), 'Occupied attempt; never retry')
    path.mkdir(); syncdir(path.parent)


def publish_bundle(directory, payload, success):
    entries = list(directory.iterdir())
    require(all(p.name in FILES - {'MANIFEST.json', 'BUNDLE.tar.gz'} for p in entries), 'Unexpected audit artifact')
    files = {p.name: reference(p) for p in sorted(entries)}
    require(not success or set(files) == SUCCESS_FILES, 'Incomplete successful audit publication')
    bundle = directory / 'BUNDLE.tar.gz'
    with bundle.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name in sorted(files): archive.add(ordinary(directory / name), arcname=name, recursive=False)
        stream.flush(); os.fsync(stream.fileno())
    value = dict(schema='sr-satdose-continuation-audit-manifest-v1', success=success,
                 status='CONTINUATION_AUDIT_PUBLICATION_COMPLETE' if success else 'AUDIT_BLOCKED_OR_UNKNOWN_EVIDENCE_RETAINED',
                 job_id=payload['job_id'], selection=payload['selection'], permission=PERMISSION,
                 operation_source_sha256=payload['operation_source_sha256'], payload_sha256=digest(canonical(payload)),
                 files=files, bundle=reference(bundle), next_stage_authorized=False)
    write_once(directory / 'MANIFEST.json', value)
    return value


def audit_remote(payload, source):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and ROOT.resolve(strict=True) == ROOT,
            'Canonical CPU-only remote execution required')
    require(digest(source) == payload['operation_source_sha256'], 'Auditor source differs')
    s = payload['selection']
    require(s == selection(s['start_seed'], s['arm'], s['first_step']), 'Malformed selected chunk')
    directory = audit_for(s); mkdir_new(directory)
    write_once(directory / 'OPERATION_SOURCE.py', source)
    write_once(directory / 'INTENT.json', dict(status='AUDIT_ATTEMPT_UNKNOWN_NO_RETRY', payload=payload,
               source=reference(directory / 'OPERATION_SOURCE.py'), created_at_utc=utc(), permission=PERMISSION))
    success = False
    try:
        sources = inventory(payload)
        with ExitStack() as stack:
            stable = [stack.enter_context(idle_lock(ROOT / name)) for name in
                      ('FIRST_TWO_LAUNCHER.lock', 'DIAGNOSTIC_WORKER.lock', 'training_transactions/TRANSACTION.lock')]
            write_once(directory / 'INCIDENTS.json', incidents(s))
            before = scheduler(payload['job_id']); write_once(directory / 'SCHEDULER_BEFORE.json', before)
            scheduler_complete(before, payload['job_id'], s)
            submit = submission_evidence(payload); write_once(directory / 'SUBMISSION_EVIDENCE.json', submit)
            for suffix, label in [('out', 'STDOUT'), ('err', 'STDERR')]:
                write_once(directory / f'OWN_JOB_{label}.log', ordinary(schedule_for(s) / f"slurm-{payload['job_id']}.{suffix}").read_bytes())
            sys.path.insert(0, str(CODE))
            from execution import continuation_control as c
            require(c.ROOT == ROOT and c.PACKAGE == CODE, 'Foreign original TRAIN namespace')
            c._hold_free()
            ref = payload['submitted']['release']; release, original = c.verify_release(c.bound(ref), ref['sha256'])
            pp = payload['preparation_payload']
            require(release['sources'] == sources and original['execution_sources'] == pp['execution_sources']
                    and release['original_train_release'] == pp['original'] and release['canary_spec'] == pp['canary_spec']
                    and release['selection'] == s and c.read(pp['preparation_ref']) == pp['preparation'],
                    'Original admission/preparation/canary/source identity differs')
            c._barrier(release['trusted_commits'], release['dev_endpoints'], s['start_seed'], s['arm'], s['first_step'])
            slot = slot_for(s); result_path = ordinary(slot / 'RESULT.json'); result = loads(result_path.read_bytes())
            claim_path = ordinary(slot / 'CLAIM.json'); claim = loads(claim_path.read_bytes())
            fixed_evidence = [reference(result_path), reference(claim_path), reference(c.bound(ref)),
                              reference(c.bound(pp['preparation_ref']))]
            require(result['release'] == ref, 'Chunk executed another release')
            worker_stdout((directory / 'OWN_JOB_STDOUT.log').read_bytes(), result)
            trusted = validate_result(result, release, s)
            node = check_claim(claim, result, payload); scheduler_complete(before, payload['job_id'], s, node)
            for name, value in [('ORIGINAL_RESULT.json', result_path.read_bytes()), ('ORIGINAL_CLAIM.json', claim_path.read_bytes()),
                                ('ORIGINAL_RELEASE.json', c.bound(ref).read_bytes())]: write_once(directory / name, value)
            for family, field in [('first_two_control', 'process'), ('continuation_control', 'process'),
                                  ('trained_dev', 'process_identity'), ('start_dev', 'process_identity')]:
                for path in (ROOT / family).rglob('RESULT.json'):
                    if path == result_path: continue
                    value = loads(ordinary(path).read_bytes())
                    if field in value: require(value[field] != result['process'], 'Reused model worker incarnation')
            c._cursor(trusted)
            chunks = copy.deepcopy(release['prior_chunks']); key = f"{s['start_seed']}/{s['arm']}"
            chunks[f'{key}/{slot.name}'] = c.ref(result_path)
            c._completed_attempts(trusted, release['dev_endpoints'], chunks)
            submission_population(release, s)
            require(result['identity'] == c.read(result['initial_commit'])['identity'], 'Checkpoint execution identity changed')
            history = historical_risks(c, result, release, original, s)
            captures = capture_population(c, result, s)
            fixed_evidence += [reference(c.bound(item)) for item in result['risk_assessments']]
            freeze = loads(ordinary(ROOT / 'DESIGN_FREEZE.json').read_bytes())
            inputs = c._risk_inputs(release, original, freeze); inputs['trusted_commits'] = trusted
            spec = c.read(release['admission_spec'])
            c.first_two._risk(release['risk_clarification'])
            authority = c.admission_authority.ContinuingAdmission(spec, admission=release['admission'], continuation_spec=dict(
                root=str(ROOT), helper=original['execution_sources']['continuation_verifier'],
                execution_sources=original['execution_sources'], canary_spec=c.read(release['canary_spec']), risk_inputs=inputs))
            replay = authority.risk_snapshot(); write_once(directory / 'RISK_REPLAY.json', replay)
            require(replay == c.read(result['post_risk']) and replay['global_hold'] is False
                    and result['pending_required_dev'] == replay['pending_required_dev'], 'Last risk differs or scientific HOLD')
            pending = replay['pending_required_dev']
            require((s['last_step'] in (8, 16) and f"{key}/{s['last_step']}" in pending
                     and all(int(k.rsplit('/', 1)[1]) == s['last_step'] for k in pending))
                    or (s['last_step'] == 24 and not pending), 'Completed chunk DEV barrier differs')
            # This covers original model snapshot symlinks and observed full admission/canary evidence too.
            authority._helper().original._observed.stable(); authority._helper().canary.stable()
            c._hold_free(); c._cursor(trusted); c._completed_attempts(trusted, release['dev_endpoints'], chunks)
            submission_population(release, s)
            require(capture_population(c, result, s) == captures, 'Draw inventory changed')
            modules_pinned(sources); inventory(payload)
            for path, expected in replay['observed_files'].items(): require(sha(Path(path)) == expected, 'Risk evidence mutated')
            for ref_value in submit['files'].values(): pinned(ref_value)
            for ref_value in fixed_evidence: pinned(ref_value)
            c.verify_release(c.bound(ref), ref['sha256'])
            observed = dict(replay['observed_files'])
            for item in fixed_evidence + list(submit['files'].values()): observed[item['path']] = item['sha256']
            for name, expected in sources.items(): observed[str(CODE / name)] = expected
            model_links = {}; model_directories = {}; observed_directories = {}
            for obs in (authority._helper().original._observed, authority._helper().canary._observed):
                for path, expected in obs.files.items():
                    require(str(path) not in observed or observed[str(path)] == expected, 'Conflicting observed evidence')
                    observed[str(path)] = expected
                for path, entries in obs.directories.items(): observed_directories[str(path)] = sorted(entries)
                for path, entries in obs.model_directories.items(): model_directories[str(path)] = sorted(entries)
                for path, (link, target) in obs.model_links.items():
                    model_links[str(path)] = dict(link=str(link), target=str(target))
            for check in stable: check()
            after = scheduler(payload['job_id']); write_once(directory / 'SCHEDULER_AFTER.json', after)
            require(scheduler_complete(after, payload['job_id'], s, node) == scheduler_complete(before, payload['job_id'], s, node),
                    'Scheduler terminal evidence changed during audit')
            report = dict(schema='sr-satdose-independent-continuation-audit-v1', status='CONTINUATION_INDEPENDENT_AUDIT_COMPLETE',
                permission=PERMISSION, job_id=payload['job_id'], selection=s, result=c.ref(result_path), claim=c.ref(claim_path),
                release=ref, original_train_release=pp['original'], source_files=84, execution_source_keys=19,
                added_commits=len(result['commits']), train_responses=result['train_responses'],
                actual_committed_prefixes={k: len(v) for k, v in trusted.items()},
                full_current_post_risk_exactly_reproduced=True, historical_risks=history,
                global_scientific_hold=False, pending_required_dev=pending, observed_files=observed,
                observed_directories=observed_directories, observed_model_links=model_links,
                observed_model_directories=model_directories, capture_counts=captures,
                scheduler_exit_observed=True, direct_compute_node_process_death_observed=False,
                original_process_exit_flag_preserved=result['process_exit_observed'], lease_boundary_verified=True,
                audit_model_calls=0, audit_optimizer_calls=0, audit_parameter_updates=0, audit_scheduler_mutations=0,
                next_stage_authorized=False, limitation=LIMITATION, operation_source_sha256=payload['operation_source_sha256'])
            write_once(directory / 'AUDIT_REPORT.json', report)
            write_once(directory / 'PUBLICATION.json', dict(status='CONTINUATION_AUDIT_PUBLISHED', permission=PERMISSION,
                report=reference(directory / 'AUDIT_REPORT.json'), job_id=payload['job_id'], selection=s,
                files={p.name: reference(p) for p in sorted(directory.iterdir())}, next_stage_authorized=False))
            success = True
    except BaseException as exc:
        write_once(directory / 'ERROR.json', dict(status='CONTINUATION_AUDIT_BLOCKED_OR_UNKNOWN_NO_RETRY',
            permission=PERMISSION, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            next_stage_authorized=False, scientific_hold_preserved=True))
    return publish_bundle(directory, payload, success)


def validate_manifest(value, payload):
    s = payload['selection']; directory = audit_for(s)
    require(value.get('schema') == 'sr-satdose-continuation-audit-manifest-v1'
            and value.get('permission') == PERMISSION and value.get('job_id') == payload['job_id']
            and value.get('selection') == s and value.get('payload_sha256') == digest(canonical(payload))
            and value.get('operation_source_sha256') == payload['operation_source_sha256']
            and value.get('next_stage_authorized') is False and type(value.get('success')) is bool, 'Foreign audit manifest')
    files = value['files']
    require(isinstance(files, dict) and set(files) <= FILES - {'MANIFEST.json', 'BUNDLE.tar.gz'}
            and {'INTENT.json', 'OPERATION_SOURCE.py'} <= set(files), 'Unknown/partial artifact inventory')
    if value['success']:
        require(set(files) == SUCCESS_FILES and value['status'] == 'CONTINUATION_AUDIT_PUBLICATION_COMPLETE',
                'Incomplete successful manifest')
    else:
        require('ERROR.json' in files and value['status'] == 'AUDIT_BLOCKED_OR_UNKNOWN_EVIDENCE_RETAINED', 'Failure lacks retained evidence')
    for name, ref in dict(files, **{'BUNDLE.tar.gz': value['bundle']}).items():
        require(set(ref) == {'path', 'sha256', 'bytes'} and ref['path'] == str(directory / name)
                and re.fullmatch('[0-9a-f]{64}', ref['sha256']) and type(ref['bytes']) is int
                and 0 <= ref['bytes'] <= (2 << 30), 'Unsafe artifact reference')
    require(files['OPERATION_SOURCE.py']['sha256'] == payload['operation_source_sha256'], 'Manifest source mismatch')
    return value


def download_remote(value, payload):
    validate_manifest(value, payload)
    directory = audit_for(payload['selection'])
    require(loads(ordinary(directory / 'MANIFEST.json').read_bytes()) == value, 'Remote manifest changed')
    require({p.name for p in directory.iterdir()} == set(value['files']) | {'MANIFEST.json', 'BUNDLE.tar.gz'},
            'Remote publication incomplete/extra')
    for ref in value['files'].values(): pinned(ref)
    bundle = pinned(value['bundle'])
    with bundle.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): sys.stdout.buffer.write(block)
    sys.stdout.buffer.flush()


def validate_bundle(path, manifest, payload, destination):
    validate_manifest(manifest, payload)
    require(sha(path) == manifest['bundle']['sha256'] and path.stat().st_size == manifest['bundle']['bytes'],
            'Partial or changed downloaded bundle; retain raw bytes')
    expected = manifest['files']; contents = {}
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(expected) and {m.name for m in members} == set(expected), 'Bundle missing/extra/duplicate entries')
        for member in members:
            require(member.isfile() and not member.issym() and not member.islnk() and member.name in expected
                    and member.size == expected[member.name]['bytes'], 'Unsafe bundle member')
            raw = archive.extractfile(member).read()
            require(len(raw) == member.size and digest(raw) == expected[member.name]['sha256'], 'Bundle member hash differs')
            contents[member.name] = raw
    require(loads(contents['INTENT.json'])['payload'] == payload, 'Downloaded intent differs')
    if manifest['success']:
        report = loads(contents['AUDIT_REPORT.json']); publication = loads(contents['PUBLICATION.json'])
        require(report['status'] == 'CONTINUATION_INDEPENDENT_AUDIT_COMPLETE'
                and report['job_id'] == payload['job_id'] and report['selection'] == payload['selection']
                and report['permission'] == PERMISSION and report['next_stage_authorized'] is False
                and report['full_current_post_risk_exactly_reproduced'] is True
                and report['global_scientific_hold'] is False and report['direct_compute_node_process_death_observed'] is False
                and publication['status'] == 'CONTINUATION_AUDIT_PUBLISHED'
                and publication['report'] == expected['AUDIT_REPORT.json']
                and publication['files'] == {k: v for k, v in expected.items() if k != 'PUBLICATION.json'},
                'Downloaded semantic publication incomplete')
    require(not os.path.lexists(destination), 'Extraction directory occupied')
    destination.mkdir()
    for name, raw in contents.items(): write_once(destination / name, raw)
    return dict(bundle_verified=True, audit_success=manifest['success'], next_stage_authorized=False)


def ssh_call(source, mode, data, out_path, err_path):
    # Payload and embedded helper code travel on stdin, never a potentially
    # >128KiB Linux argv element. Preserve the exact source bytes before dispatch.
    envelope = canonical(dict(source_b64=base64.b64encode(source).decode(), data=data))
    bootstrap = ('import hashlib,sys,json,base64; envelope=json.loads(sys.stdin.buffer.read()); '
                 "source=base64.b64decode(envelope['source_b64'],validate=True); "
                 f'assert hashlib.sha256(source).hexdigest()=={digest(source)!r}; '
                 "sys.argv.append(base64.b64encode(json.dumps(envelope['data'],allow_nan=False).encode()).decode()); "
                 "exec(compile(source,'<pinned-continuation-auditor>','exec'))")
    command = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
                          'TRANSFORMERS_OFFLINE=1', 'OMP_NUM_THREADS=1', PYTHON, '-I', '-B', '-c', bootstrap, mode])
    with out_path.open('xb') as out, err_path.open('xb') as err:
        try:
            value = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o',
                'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4', 'eee-cluster', command],
                input=envelope, stdout=out, stderr=err, check=False)
        finally: out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
    return value.returncode


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submission-receipt', type=Path, required=True)
    p.add_argument('--submission-sha256', required=True)
    p.add_argument('--job-id', type=int, required=True)
    p.add_argument('--start-seed', type=int, choices=(11031, 11032), required=True)
    p.add_argument('--arm', choices=('T0', 'T1'), required=True)
    p.add_argument('--first-step', type=int, choices=tuple(CHUNKS), required=True)
    p.add_argument('--local-dir', type=Path)
    p.add_argument('--recover-manifest', type=Path)
    p.add_argument('--recover-sha256')
    p.add_argument('--execute', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv); payload = local_payload(args)
    source = ordinary(Path(__file__)).read_bytes(); payload['operation_source_sha256'] = digest(source)
    require(bool(args.recover_manifest) == bool(args.recover_sha256), 'Recovery requires exact manifest and SHA')
    manifest = None
    if args.recover_manifest:
        manifest = loads(pinned(dict(path=str(args.recover_manifest), sha256=args.recover_sha256)).read_bytes())
        validate_manifest(manifest, payload)
    if not args.execute:
        print(canonical(dict(status='LOCAL_PLAN_ONLY_NO_SSH', selection=payload['selection'], job_id=args.job_id,
            remote_directory=str(audit_for(payload['selection'])), planned_audit_invocations=0 if manifest else 1,
            planned_bundle_downloads=1, permission=PERMISSION, next_stage_authorized=False)).decode(), end='')
        return 0
    require(args.local_dir is not None and args.local_dir.is_absolute()
            and args.local_dir.parent.resolve(strict=True) == args.local_dir.parent, 'New canonical local directory required')
    mkdir_new(args.local_dir); directory = args.local_dir
    write_once(directory / 'OPERATION_SOURCE.py', source)
    write_once(directory / 'LOCAL_INTENT.json', dict(payload=payload, recovery_manifest=manifest, created_at_utc=utc(),
               status='DOWNLOAD_ONLY_NO_REAUDIT' if manifest else 'AUDIT_ATTEMPT_UNKNOWN_NO_RETRY'))
    receipt = dict(status='AUDIT_OR_DOWNLOAD_UNKNOWN_NO_RETRY', permission=PERMISSION, next_stage_authorized=False)
    try:
        if manifest is None:
            code = ssh_call(source, '--remote-worker', payload, directory / 'AUDIT_STDOUT.log', directory / 'AUDIT_STDERR.log')
            manifest = marker((directory / 'AUDIT_STDOUT.log').read_bytes(), MARKER)
            validate_manifest(manifest, payload)
            # Preserve the actual successful audit manifest BEFORE any download attempt.
            write_once(directory / 'REMOTE_MANIFEST.json', manifest)
            require(code == (0 if manifest['success'] else 2), 'SSH exit and audit manifest disagree')
        else: write_once(directory / 'REMOTE_MANIFEST.json', manifest)
        receipt['remote_audit_success'] = manifest['success']
        receipt['manifest'] = reference(directory / 'REMOTE_MANIFEST.json')
        code = ssh_call(source, '--remote-download', dict(payload=payload, manifest=manifest),
                        directory / 'BUNDLE.tar.gz', directory / 'DOWNLOAD_STDERR.log')
        require(code == 0, 'Bundle download failed; explicit download-only recovery, never repeat audit')
        receipt.update(validate_bundle(directory / 'BUNDLE.tar.gz', manifest, payload, directory / 'artifacts'))
        receipt['status'] = 'AUDIT_BUNDLE_VERIFIED' if manifest['success'] else 'BLOCKED_AUDIT_EVIDENCE_BUNDLE_VERIFIED'
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc),
                       recovery='Preserve raw bytes. With saved manifest use explicit --recover-manifest and a NEW directory; never reaudit.')
    receipt['local_files'] = {p.name: reference(p) for p in sorted(directory.iterdir()) if p.is_file()}
    write_once(directory / 'LOCAL_RECEIPT.json', receipt)
    print(canonical(dict(receipt=str(directory / 'LOCAL_RECEIPT.json'), **receipt)).decode(), end='')
    return 0 if receipt['status'] == 'AUDIT_BUNDLE_VERIFIED' else 2


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] in ('--remote-worker', '--remote-download'):
        value = loads(base64.b64decode(sys.argv[2], validate=True))
        if sys.argv[1] == '--remote-download': download_remote(value['manifest'], value['payload'])
        else:
            result = audit_remote(value, source)
            print(MARKER.decode() + canonical(result).decode(), end='', flush=True)
            raise SystemExit(0 if result['success'] else 2)
    else: raise SystemExit(main())
