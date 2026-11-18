#!/usr/bin/env python3
"""Read-only snapshot of one authenticated continuation job. Default: local plan.

--execute reads only this job, its registered chunk and global project incidents.
It writes timestamped local source/intent/raw/receipt files, never remote files.
Counts are a live non-atomic observation, not independent completion acceptance.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

# Exact independently reviewed auditor helper; no unpinned dynamic dependency.
AUDITOR_SHA = '9379665201d95df4b7e0671779e19fe2d686a435e6f15e68e36203832db5e352'
PERMISSION = 'READ_ONLY_OWN_CHUNK_MONITOR_NO_TRAINING_OR_ACCEPTANCE'


def load_auditor():
    path = Path(__file__).with_name('satdose_audit_continuation.py')
    if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
        raise ValueError('Noncanonical continuation auditor')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUDITOR_SHA:
        raise ValueError('Pinned continuation auditor changed')
    module = ModuleType('satdose_monitor_continuation_helpers')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module, raw


def observe(a, payload):
    s = payload['selection']; job = payload['job_id']; slot = a.slot_for(s); schedule = a.schedule_for(s)
    a.require(s == a.selection(s['start_seed'], s['arm'], s['first_step']) and type(job) is int and job > 0,
              'Invalid monitor selection/job')
    a.require(a.ROOT.resolve(strict=True) == a.ROOT, 'Noncanonical project root')
    actual = a.loads(a.pinned(payload['submission_ref']).read_bytes())
    a.require(payload['submission_ref']['path'] == str(schedule / 'RELEASE_RESULT.json')
              and actual == payload['submitted'] and actual['job_id'] == job and actual['selection'] == s,
              'Own remote submission identity differs')
    accepted = a.loads(a.pinned(actual['submission']).read_bytes())
    a.require(accepted['job_id'] == job and accepted['selection'] == s and accepted['release'] == actual['release']
              and accepted['job_name'] == a.job_name(s), 'Accepted job identity differs')
    observation = a.scheduler(job)
    # A pending job may not yet have accounting rows. Never read a foreign returned row.
    accounting = observation['accounting']
    if accounting['returncode'] == 0:
        rows = [line.split('|') for line in base64.b64decode(accounting['stdout_b64'], validate=True).decode().splitlines() if line.strip()]
        a.require(all(len(row) == 10 and (row[0] == str(job) or row[0] in
                  (f'{job}.batch', f'{job}.extern', f'{job}.0')) for row in rows), 'Foreign/malformed monitor accounting')
        parents = [r for r in rows if r[0] == str(job)]
        a.require(len(parents) <= 1 and (not parents or parents[0][3:7] == [a.job_name(s), 'rose', a.QOS, 'cluster02']),
                  'Monitor scheduler identity differs')
    queue = observation['queue']
    if queue['returncode'] == 0:
        rows = [line.split('|') for line in base64.b64decode(queue['stdout_b64'], validate=True).decode().splitlines() if line.strip()]
        a.require(len(rows) <= 1 and all(len(r) == 4 and r[:2] == [str(job), a.job_name(s)] for r in rows),
                  'Foreign/malformed monitor queue')
    records = {}; steps = {str(n): dict(BEGIN=0, CAPTURE=0, COMPLETE=0, commit=None, risk=None)
                           for n in range(s['first_step'], s['last_step'] + 1)}
    observed = {}; unresolved = []
    def read(path):
        ref = a.reference(path); observed[str(path)] = ref
        return a.loads(path.read_bytes())
    if os.path.lexists(slot):
        a.require(slot.resolve(strict=True) == slot and slot.is_dir(), 'Noncanonical own chunk slot')
        for entry in slot.rglob('*'):
            a.require(entry.resolve(strict=True) == entry and not entry.is_symlink()
                      and (entry.is_file() or entry.is_dir()), 'Noncanonical own chunk entry')
        for name in ('CLAIM.json', 'RESULT.json', 'HOLD.json', 'ERROR.json'):
            path = slot / name
            if os.path.lexists(path): records[name] = read(path)
        if 'CLAIM.json' in records:
            claim = records['CLAIM.json']
            a.require(claim['selection'] == s and claim['release'] == actual['release']
                      and claim['allocation']['job_id'] == str(job), 'Live claim belongs to a different job')
        audit = slot / 'audit'
        begins = {}
        for path in sorted(audit.rglob('*.BEGIN.json')):
            value = read(path); ep = value['endpoint']; step = ep['step'] + 1
            a.require(ep['start_seed'] == s['start_seed'] and ep['arm'] == s['arm']
                      and s['first_step'] <= step <= s['last_step'], 'Foreign live generation endpoint')
            stem = path.name.removesuffix('.BEGIN.json')
            a.require(stem == a.digest(value['request_id'].encode()) and stem not in begins, 'Duplicate/foreign BEGIN')
            begins[stem] = step; steps[str(step)]['BEGIN'] += 1
            for suffix in ('CAPTURE', 'COMPLETE'):
                capture = path.with_name(stem + '.' + suffix + '.json')
                if os.path.lexists(capture): a.ordinary(capture); steps[str(step)][suffix] += 1
                else: unresolved.append(str(capture.relative_to(slot)))
        for suffix in ('CAPTURE', 'COMPLETE'):
            for path in audit.rglob('*.' + suffix + '.json'):
                a.require(path.name.removesuffix('.' + suffix + '.json') in begins, 'Capture without BEGIN')
    for step in steps:
        path = a.ROOT / 'training_transactions' / f"start-{s['start_seed']}" / s['arm'] / f'step-{int(step):03d}' / 'COMMIT.json'
        if os.path.lexists(path):
            commit = read(path)
            a.require(commit['step'] == int(step) and commit['identity']['train_seed'] == s['start_seed']
                      and commit['identity']['arm'] == s['arm'], 'Foreign live COMMIT')
            steps[step]['commit'] = observed[str(path)]
        path = slot / f'RISK_AFTER_STEP_{int(step):03d}.json'
        if os.path.lexists(path): steps[step]['risk'] = read(path)
    tails = {}
    for suffix in ('out', 'err'):
        path = schedule / f'slurm-{job}.{suffix}'
        if os.path.lexists(path):
            a.ordinary(path)
            with path.open('rb') as stream:
                stream.seek(0, 2); size = stream.tell(); stream.seek(max(0, size - 8192)); raw = stream.read(8192)
            tails[path.name] = dict(bytes_observed=size, tail_b64=base64.b64encode(raw).decode())
    return dict(status='OWN_CONTINUATION_READ_ONLY_SNAPSHOT', permission=PERMISSION, job_id=job, selection=s,
                observed_at_utc=a.utc(), scheduler=observation, slot_exists=slot.exists(), records=records,
                per_step=steps, unresolved_capture_paths=unresolved, incidents=a.incidents(s), log_tails=tails,
                observed_metadata_files=observed, next_stage_authorized=False,
                limitation='Non-atomic live observation; incomplete captures may be in flight. No completion/risk admission.')


def main(argv=None):
    a, auditor_source = load_auditor()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission-receipt', type=Path, required=True)
    parser.add_argument('--submission-sha256', required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--start-seed', type=int, choices=(11031, 11032), required=True)
    parser.add_argument('--arm', choices=('T0', 'T1'), required=True)
    parser.add_argument('--first-step', type=int, choices=tuple(a.CHUNKS), required=True)
    parser.add_argument('--output-dir', type=Path, default=a.WORKSPACE / 'artifacts/sr_satdose_20261112/intake')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv); payload = a.local_payload(args)
    if not args.execute:
        print(a.canonical(dict(status='LOCAL_PLAN_ONLY_NO_SSH', job_id=args.job_id, selection=payload['selection'],
                              permission=PERMISSION, next_stage_authorized=False)).decode(), end=''); return 0
    a.require(args.output_dir.is_absolute() and args.output_dir.resolve(strict=True) == args.output_dir
              and args.output_dir.is_dir(), 'Canonical existing local monitor output directory required')
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = args.output_dir / f'MONITOR_CONTINUATION_{args.job_id}_{tag}'
    a.mkdir_new(directory); source = a.ordinary(Path(__file__)).read_bytes()
    a.write_once(directory / 'MONITOR_SOURCE.py', source); a.write_once(directory / 'AUDITOR_SOURCE.py', auditor_source)
    data = dict(payload=payload, auditor_source_b64=base64.b64encode(auditor_source).decode())
    a.write_once(directory / 'INTENT.json', dict(status='READ_ONLY_MONITOR_ATTEMPT', payload=payload,
                 auditor_sha256=AUDITOR_SHA, monitor_sha256=a.digest(source)))
    receipt = dict(status='MONITOR_UNKNOWN', permission=PERMISSION, next_stage_authorized=False)
    try:
        code = a.ssh_call(source, '--remote-monitor', data, directory / 'STDOUT.log', directory / 'STDERR.log')
        a.require(code == 0, 'Read-only monitor SSH failed; retain raw logs')
        value = a.loads((directory / 'STDOUT.log').read_bytes())
        a.require(value['status'] == 'OWN_CONTINUATION_READ_ONLY_SNAPSHOT' and value['job_id'] == args.job_id
                  and value['selection'] == payload['selection'] and value['permission'] == PERMISSION,
                  'Foreign monitor response')
        receipt.update(status='MONITOR_SNAPSHOT_CAPTURED', snapshot=value)
    except BaseException as exc: receipt.update(error_type=type(exc).__name__, error=str(exc))
    receipt['files'] = {p.name: a.reference(p) for p in directory.iterdir() if p.is_file()}
    a.write_once(directory / 'RECEIPT.json', receipt)
    print(a.canonical(dict(receipt=str(directory / 'RECEIPT.json'), **receipt)).decode(), end='')
    return 0 if receipt['status'] == 'MONITOR_SNAPSHOT_CAPTURED' else 2


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--remote-monitor':
        data = json.loads(base64.b64decode(sys.argv[2], validate=True))
        raw = base64.b64decode(data['auditor_source_b64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != AUDITOR_SHA: raise ValueError('Remote auditor helper source differs')
        helper = ModuleType('satdose_monitor_continuation_pinned_remote')
        exec(compile(raw, '<pinned-audit-monitor-helpers>', 'exec'), helper.__dict__)
        print(helper.canonical(observe(helper, data['payload'])).decode(), end='', flush=True)
    else: raise SystemExit(main())
