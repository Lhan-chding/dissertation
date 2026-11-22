#!/usr/bin/env python3
"""Observe the fixed 2026-11-22 SSH255 attempt; never authorize resubmission.

Default is a local evidence check. --execute requires a new --local-dir and
collects remote read-only observations. No project module is imported remotely.
The old UNKNOWN is immutable, including when this observation succeeds.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys

WORKSPACE = Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
PYTHON = '/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
ATTEMPT = WORKSPACE / 'artifacts/sr_satdose_20261112/continuation_20261122/start-11032_T1_chunk-009-016/SUBMISSION.json'
RECEIPT_SHA = 'e1bb871f00c9ac57dc1cae494e6d3de966065221eeeea12056b42311d79000e8'
INTENT_SHA = '9b25b4a51e079fd15a01cfcee5ef46c35b9913088df63daa4f9adebf4f931e25'
SUBMITTER_SHA = 'eceec5aab4de56b53d8e1483d2e49093e28f4048b2ef818302bb3db499fb0f94'
PREPARATION_SHA = '2c7e83df52b7edaa842908e95c8ebc71d6c4161dba41219655b6aae0e39a1bbb'
STDERR_SHA = '27b8e25ec24ea1acc2777e8c1dac47538e844d41e856496ed237e4a1d346c44f'
SELECTION = dict(start_seed=11032, arm='T1', first_step=9, last_step=16)
JOB_NAME = 'LH-SATDOSE-TRAIN-11032-T1-009-016'
EXPECTED_HOSTNAME = 'login-3.cluster02.eee.ntu.edu.sg'
PERMISSION = 'OBSERVATION_ONLY_NO_RESUBMISSION_TRAINING_OR_MAIN_AUTHORITY'
INSPECTOR_SOURCE_SHA = None  # Set by authenticated in-memory bootstrap, never read from the payload.
REMOTE_PROGRESS = {}
CONTROL_FAMILIES = ('training_transactions', 'training_control', 'first_two_control',
                    'first_two_launcher', 'canary_replay', 'start_dev', 'trained_dev', 'continuation_control')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode(data):
    def pairs(items):
        result = dict(items)
        require(len(result) == len(items), 'Duplicate JSON key')
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'Nonfinite JSON'))


def canonical(path, directory=False):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
            'Noncanonical path: ' + str(path))
    mode = path.stat().st_mode
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode), 'Nonordinary path: ' + str(path))
    return path


def checked(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str), 'Malformed reference')
    path = canonical(Path(ref['path']))
    data = path.read_bytes()
    require(digest(data) == ref.get('sha256'), 'Changed evidence: ' + str(path))
    return data


def original_remote(source):
    tree = ast.parse(source)
    matches = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in node.targets)
               and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    require(len(matches) == 1, 'Original REMOTE source unavailable')
    return matches[0]


def local_payload():
    receipt = decode(checked(dict(path=str(ATTEMPT), sha256=RECEIPT_SHA)))
    intent_path = ATTEMPT.with_name(ATTEMPT.name + '.intent.json')
    intent = decode(checked(dict(path=str(intent_path), sha256=INTENT_SHA)))
    require(receipt['status'] == 'SUBMISSION_UNKNOWN_NO_RETRY' and receipt['returncode'] == 255
            and receipt['response'] is None, 'Original interrupted receipt differs')
    require(intent['status'] == 'SSH_SUBMISSION_ATTEMPT_UNKNOWN_NO_RETRY'
            and intent['source'] == receipt['source'], 'Original attempt intent differs')
    require(receipt['source']['sha256'] == SUBMITTER_SHA, 'Original submitter pin differs')
    source = checked(receipt['source'])
    require(set(receipt['raw_logs']) == {'stdout', 'stderr'}, 'Incomplete raw logs')
    for name, expected in (('stdout', digest(b'')), ('stderr', STDERR_SHA)):
        ref = receipt['raw_logs'][name]
        require(ref['sha256'] == expected and ref['path'] == intent[name + '_path'], 'Raw log pin differs')
        require(text_read(checked(ref)) == receipt[name], 'Raw log and receipt disagree')
    payload = intent['payload']
    require({key: payload[key] for key in SELECTION} == SELECTION, 'Fixed selection differs')
    require(payload['source_sha256'] == SUBMITTER_SHA
            and payload['preparation_local']['sha256'] == PREPARATION_SHA, 'Original lineage differs')
    # Byte-pinned original function checks all local preparation logs and the
    # original immutable84/19-key inventories; it performs no SSH or preflight.
    namespace = {'__name__': 'authenticated_original_submitter', '__file__': receipt['source']['path']}
    exec(compile(source, receipt['source']['path'], 'exec'), namespace)
    rebuilt = namespace['payload_from_receipt'](Path(payload['preparation_local']['path']), PREPARATION_SHA)
    rebuilt['source_sha256'] = SUBMITTER_SHA
    require(rebuilt == payload, 'Original intent payload does not match authenticated preparation')
    return dict(original_payload=payload, failed_receipt_sha256=RECEIPT_SHA,
                original_intent_sha256=INTENT_SHA, submitter_sha256=SUBMITTER_SHA,
                original_remote_sha256=digest(original_remote(source).encode()), selection=SELECTION,
                expected_hostname=EXPECTED_HOSTNAME, expected_username='varun024')


def absent(path):
    """Check every existing ancestor even if the final component is missing."""
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Invalid absence path')
    for ancestor in reversed(path.parents):
        if os.path.lexists(ancestor):
            canonical(ancestor, directory=True)
    require(not os.path.lexists(path), 'Occupied slot or HOLD: ' + str(path))
    return dict(path=str(path), state='ABSENT_LEXISTS_FALSE')


def text_read(data):
    """Match the original Path.read_text universal-newline projection, retaining raw hashes."""
    return data.decode(errors='replace').replace('\r\n', '\n').replace('\r', '\n')


def inventory(root, expected):
    code = canonical(root / 'train_code_71018e5', directory=True)
    values = {}
    for parent, directories, files in os.walk(code, followlinks=False):
        for name in directories:
            canonical(Path(parent) / name, directory=True)
        for name in files:
            path = canonical(Path(parent) / name)
            values[str(path.relative_to(code))] = digest(path.read_bytes())
    require(len(values) == 84 and values == expected, 'Original source84 changed')
    return dict(file_count=84, inventory_sha256=digest(json.dumps(values, sort_keys=True).encode()))


@contextmanager
def leases(root):
    import fcntl
    opened = []
    try:
        for name, mode in (('FIRST_TWO_LAUNCHER.lock', fcntl.LOCK_SH),
                           ('DIAGNOSTIC_WORKER.lock', fcntl.LOCK_EX)):
            path = canonical(root / name)
            # Match the original open mode for flock on network filesystems.
            # No create/truncate/write call is made on either existing lock.
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            opened.append((fd, path, 'SHARED' if mode == fcntl.LOCK_SH else 'EXCLUSIVE'))
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
        yield opened
    finally:
        while opened:
            os.close(opened.pop()[0])


def stable_locks(opened):
    result = []
    for fd, path, mode in opened:
        canonical(path)
        current, held = path.stat(), os.fstat(fd)
        require((current.st_dev, current.st_ino) == (held.st_dev, held.st_ino), 'Lease inode replaced')
        result.append(dict(path=str(path), mode=mode, device=held.st_dev, inode=held.st_ino))
    return result


def slots_and_holds(root):
    paths = [root.parent / 'scheduling/continuation_71018e5/start-11032/T1/chunk-009-016',
             root / 'continuation_control/start-11032/T1/chunk-009-016', root / 'HOLD.json', root / 'ERROR.json']
    for family in CONTROL_FAMILIES:
        directory = root / family
        if not os.path.lexists(directory):
            absent(directory)
            continue
        canonical(directory, directory=True)
        for parent, dirs, files in os.walk(directory, followlinks=False):
            for name in dirs + files:
                path = Path(parent) / name
                require(not path.is_symlink(), 'Noncanonical control tree')
                require('HOLD' not in name and 'ERROR' not in name and not name.startswith('.staging'),
                        'Retained control HOLD/error/unknown')
    for family in ('trained_dev_71018e5', 'continuation_71018e5'):
        directory = root.parent / 'scheduling' / family
        if os.path.lexists(directory):
            canonical(directory, directory=True)
            for parent, dirs, files in os.walk(directory, followlinks=False):
                for name in dirs + files:
                    require(not (Path(parent) / name).is_symlink() and 'HOLD' not in name and 'ERROR' not in name,
                            'Retained submission HOLD/error or symlink')
    return [absent(path) for path in paths]


def scheduler(run=subprocess.run, observations=None):
    commands = [
        ['squeue', '--user=varun024', '--noheader', '--format=%i|%200j|%T'],
        ['sacct', '--user=varun024', '--starttime=2026-11-12T00:00:00', '--name=' + JOB_NAME,
         '--noheader', '--parsable2', '--format=JobIDRaw,JobName%200,State%40']]
    observations = [] if observations is None else observations
    for command in commands:
        result = run(command, capture_output=True, text=True, check=False, timeout=60,
                     env={k: v for k, v in os.environ.items() if k != 'SLURM_CLUSTERS'})
        observation = dict(argv=command, returncode=result.returncode, stdout=result.stdout,
                           stderr=result.stderr, observed_at_utc=now())
        observations.append(observation)
        require(result.returncode == 0 and not result.stderr.strip(), 'Scheduler query failed or warned')
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            row = [v.strip() for v in line.split('|')]
            require(len(row) == 3 and re.fullmatch(r'[1-9][0-9]*(?:[._+][A-Za-z0-9_+\[\]-]+)*', row[0])
                    and row[1] and re.fullmatch(r'[A-Z][A-Z_ +:()0-9-]*', row[2]), 'Malformed scheduler record')
            require(row[1] != JOB_NAME and not row[1].startswith(JOB_NAME + '.'), 'Matching job exists')
            require(not (row[1].endswith('+') and JOB_NAME.startswith(row[1][:-1])), 'Possibly truncated target name')
        if command[0] == 'sacct':
            require(not result.stdout.strip(), 'Exact-name accounting returned nonempty unexpected records')
    return observations


def processes(payload, proc=Path('/proc'), uid=None, self_pid=None):
    """Full own-UID census; cmdlines stay in memory, only digests are returned."""
    uid = os.getuid() if uid is None else uid
    self_pid = os.getpid() if self_pid is None else self_pid
    require(proc.is_dir(), 'No procfs: process absence UNKNOWN')
    records = []
    for entry in sorted(proc.iterdir(), key=lambda p: p.name):
        if not entry.name.isdecimal():
            continue
        try:
            # proc directory ownership can change for nondumpable processes;
            # only status credentials determine whether this is an own-UID PID.
            # Read no other user's cmdline, even when directory ownership differs.
            status = (entry / 'status').read_text()
            uids = next(line.split()[1:] for line in status.splitlines() if line.startswith('Uid:'))
            require(len(uids) == 4 and all(v.isdecimal() for v in uids), 'Malformed process UID')
            if uid not in {int(v) for v in uids}:
                continue
            require(all(int(v) == uid for v in uids), 'Ambiguous own process UID')
            stat_before = (entry / 'stat').read_text()
            tail = stat_before[stat_before.rindex(')') + 2:].split()
            require(len(tail) >= 20 and tail[1].isdecimal() and tail[19].isdecimal(), 'Malformed process identity')
            cmdline = (entry / 'cmdline').read_bytes()
            comm = (entry / 'comm').read_text().strip()
            after = (entry / 'stat').read_text()
            require(after[after.rindex(')') + 2:].split()[19] == tail[19], 'Process PID reused')
        except (OSError, StopIteration, IndexError) as exc:
            raise ValueError('Unreadable or vanished process; process absence UNKNOWN') from exc
        pid = int(entry.name)
        parts = cmdline.split(b'\0')
        exact = any(digest(part) == payload['original_remote_sha256'] for part in parts)
        markers = [str(ROOT).encode(), JOB_NAME.encode(), b'satdose_submit_continuation',
                   b'CONTINUATION_71018e5_s11032_T1_009_016', b'chunk-009-016']
        candidate = exact or any(marker in cmdline for marker in markers)
        require(cmdline or tail[0] == 'Z' or pid == self_pid, 'Empty live own cmdline: UNKNOWN')
        record = dict(pid=pid, ppid=int(tail[1]), start_ticks=int(tail[19]), state=tail[0],
                      command_basename=comm if re.fullmatch(r'[A-Za-z0-9_.() -]{1,64}', comm) else 'REDACTED',
                      cmdline_sha256=digest(cmdline), original_remote_exact_match=exact,
                      candidate=candidate, inspector_self=pid == self_pid)
        records.append(record)
        require(pid == self_pid or not candidate, 'Candidate original submitter/project process still present')
    require(any(row['pid'] == self_pid for row in records), 'Inspector missing from own-UID census')
    return dict(uid=uid, self_pid=self_pid, clock_ticks_per_second=os.sysconf('SC_CLK_TCK'),
                processes=records, scope='THIS_HOST_OWN_UID_ONLY_NOT_GLOBAL_PROCESS_ABSENCE', observed_at_utc=now())


def remote_inspect(payload, emit=None):
    import pwd
    global REMOTE_PROGRESS
    REMOTE_PROGRESS = {'stage': 'BOOTSTRAP'}
    require(INSPECTOR_SOURCE_SHA is not None and payload.get('inspector_sha256') == INSPECTOR_SOURCE_SHA,
            'Inspector source bootstrap differs')
    require(payload['selection'] == SELECTION and payload['failed_receipt_sha256'] == RECEIPT_SHA
            and payload['submitter_sha256'] == SUBMITTER_SHA and payload['original_intent_sha256'] == INTENT_SHA,
            'Payload lineage differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
    canonical(ROOT, directory=True)
    host = socket.gethostname()
    REMOTE_PROGRESS.update(observed_hostname=host, expected_hostname=EXPECTED_HOSTNAME,
                           original_gateway_banner_alias='login-1')
    require(host == payload['expected_hostname'] == EXPECTED_HOSTNAME,
            'Different exact login FQDN; process scope UNKNOWN')
    require(pwd.getpwuid(os.getuid()).pw_name == payload['expected_username'] == 'varun024', 'Wrong user')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', boot), 'Invalid boot identity')
    observation = dict(status='INSPECTION_INCOMPLETE_UNKNOWN', permission=PERMISSION,
                       original_status='SUBMISSION_UNKNOWN_NO_RETRY', original_unknown_preserved=True,
                       hostname=host, original_gateway_banner_alias='login-1', kernel_boot_id=boot,
                       uid=os.getuid(), username='varun024', started_at_utc=now(),
                       selection=SELECTION, job_name=JOB_NAME, failed_receipt_sha256=RECEIPT_SHA,
                       original_intent_sha256=INTENT_SHA, submitter_sha256=SUBMITTER_SHA,
                       inspector_sha256=INSPECTOR_SOURCE_SHA)
    REMOTE_PROGRESS = observation
    observation['stage'] = 'ACQUIRE_LOCKS'
    with leases(ROOT) as opened:
        observation['locks'] = stable_locks(opened)
        observation['stage'] = 'SOURCE_AND_PREPARATION_PINS'
        observation['sources'] = inventory(ROOT, payload['original_payload']['sources'])
        original = payload['original_payload']
        refs = [original[key] for key in ('preparation_ref', 'original', 'canary_spec')]
        refs.extend(original['preparation'][key] for key in ('intent', 'release', 'storage_budget', 'risk'))
        observed_refs = []
        for ref in refs:
            path = Path(ref['path'])
            require(not path.is_absolute() and '..' not in path.parts, 'Invalid remote reference')
            observed = dict(path=str(ROOT / path), sha256=ref['sha256'])
            checked(observed)
            observed_refs.append(observed)
        observation['observed_refs'] = observed_refs
        observation['stage'] = 'SLOT_AND_HOLD_ABSENCE_BEFORE'
        observation['absences_before'] = slots_and_holds(ROOT)
        observation['stage'] = 'OWN_UID_PROCESS_CENSUS_BEFORE'
        observation['processes_before'] = processes(payload)
        observation['stage'] = 'SCHEDULER_READ_ONLY_QUERIES'
        observation['scheduler'] = []
        scheduler(observations=observation['scheduler'])
        observation['stage'] = 'OWN_UID_PROCESS_CENSUS_AFTER'
        observation['processes_after'] = processes(payload)
        observation['stage'] = 'REVERIFY_SLOTS_SOURCES_REFS_AND_LOCKS'
        observation['absences_after'] = slots_and_holds(ROOT)
        observation['sources_after'] = inventory(ROOT, payload['original_payload']['sources'])
        for ref in observed_refs:
            checked(ref)
        observation['observed_refs_reverified_at_utc'] = now()
        observation['locks_after'] = stable_locks(opened)
        observation.update(status='PRE_SUBMISSION_ABSENCE_VERIFIED', stage='COMPLETE_UNDER_LOCKS', observed_at_utc=now(), emitted_under_locks=True,
                           limits=['Host-specific snapshot, not global or future process absence.',
                                   'Original process observed on login-1 at 17:16 SGT. This census covers login-3 only. Shared locks, slot ordering and the known original execution phase are evaluated separately.',
                                   'Empty successful scheduler results are observations, not proof of accounting retention.',
                                   'Absent slot inference assumes no deletion/move of prior evidence; operator does not establish history.',
                                   'No automatic retry or resubmission permission. Original SSH255 remains UNKNOWN.'],
                           ordering='Pinned original source acquires same locks, creates exact scheduling slot and INTENT before its sole sbatch.')
        serialized = json.dumps(observation, sort_keys=True, allow_nan=False)
        if emit is None:
            print(serialized, flush=True)
        else:
            emit(serialized)
    return observation


def write_once(path, value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def validate_report(report, source_sha):
    require(isinstance(report, dict), 'Missing remote observation')
    for key, expected in dict(status='PRE_SUBMISSION_ABSENCE_VERIFIED', permission=PERMISSION,
                              original_status='SUBMISSION_UNKNOWN_NO_RETRY', original_unknown_preserved=True,
                              failed_receipt_sha256=RECEIPT_SHA, original_intent_sha256=INTENT_SHA,
                              submitter_sha256=SUBMITTER_SHA, inspector_sha256=source_sha,
                              selection=SELECTION, job_name=JOB_NAME, hostname=EXPECTED_HOSTNAME,
                              username='varun024', emitted_under_locks=True).items():
        require(report.get(key) == expected, 'Remote observation identity differs: ' + key)
    require(type(report.get('uid')) is int and report['uid'] >= 0 and
            re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', report.get('kernel_boot_id', '')),
            'Missing host/UID/boot identity')
    require(report.get('locks') == report.get('locks_after') and len(report.get('locks', [])) == 2,
            'Incomplete stable lease observations')
    for row, name, mode in zip(report['locks'], ('FIRST_TWO_LAUNCHER.lock', 'DIAGNOSTIC_WORKER.lock'),
                               ('SHARED', 'EXCLUSIVE')):
        require(row.get('path') == str(ROOT / name) and row.get('mode') == mode and
                type(row.get('inode')) is int and row['inode'] > 0 and type(row.get('device')) is int,
                'Wrong observed lock identity')
    expected_paths = [str(ROOT.parent / 'scheduling/continuation_71018e5/start-11032/T1/chunk-009-016'),
                      str(ROOT / 'continuation_control/start-11032/T1/chunk-009-016'),
                      str(ROOT / 'HOLD.json'), str(ROOT / 'ERROR.json')]
    for key in ('absences_before', 'absences_after'):
        require(report.get(key) == [dict(path=path, state='ABSENT_LEXISTS_FALSE') for path in expected_paths],
                'Incomplete absence observations')
    require(report.get('sources') == report.get('sources_after') and report.get('sources', {}).get('file_count') == 84,
            'Incomplete source84 observations')
    require(len(report.get('observed_refs', [])) == 7 and report.get('observed_refs_reverified_at_utc'),
            'Incomplete reverified source references')
    for key in ('processes_before', 'processes_after'):
        census = report.get(key, {})
        rows = census.get('processes', [])
        require(census.get('uid') == report['uid'] and census.get('scope') == 'THIS_HOST_OWN_UID_ONLY_NOT_GLOBAL_PROCESS_ABSENCE'
                and rows and len({r['pid'] for r in rows}) == len(rows)
                and sum(r.get('inspector_self') is True for r in rows) == 1
                and all(not r.get('candidate') or r.get('inspector_self') is True for r in rows),
                'Incomplete or candidate process census')
    require(len(report.get('scheduler', [])) == 2 and all(r.get('returncode') == 0 for r in report['scheduler']),
            'Incomplete scheduler observations')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--local-dir', type=Path)
    parser.add_argument('--remote', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.remote:
        try:
            remote_inspect(decode(sys.stdin.buffer.read()))
            return 0
        except Exception as exc:
            # Do not serialize arbitrary exception values (e.g. process args).
            print(json.dumps(dict(status='INSPECTION_FAILED_UNKNOWN', permission=PERMISSION,
                                  error_type=type(exc).__name__,
                                  reason=str(exc) if isinstance(exc, ValueError) else 'Read-only inspection failed',
                                  observations=REMOTE_PROGRESS,
                                  original_unknown_preserved=True)))
            return 2
    payload = local_payload()
    source = canonical(Path(__file__)).read_bytes()
    payload['inspector_sha256'] = digest(source)
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_PLAN_ONLY_NO_SSH', permission=PERMISSION, selection=SELECTION,
                              source_sha256=digest(source), failed_receipt_sha256=RECEIPT_SHA,
                              original_unknown_preserved=True, hostname=EXPECTED_HOSTNAME,
                              original_gateway_banner_alias='login-1', job_name=JOB_NAME)))
        return 0
    require(args.local_dir is not None and args.local_dir.is_absolute(), 'New absolute local directory required')
    canonical(args.local_dir.parent, directory=True)
    absent(args.local_dir)
    args.local_dir.mkdir(mode=0o700)
    write_once(args.local_dir / 'INSPECTOR_SOURCE.py', source)
    write_once(args.local_dir / 'INTENT.json', payload)
    bootstrap = ('_inspector_bytes=bytes.fromhex(' + repr(source.hex()) + ');exec(_inspector_bytes);'
                 'INSPECTOR_SOURCE_SHA=digest(_inspector_bytes);raise SystemExit(main(["--remote"]))')
    # __name__ differs from __main__ so the source's normal entry point cannot run twice.
    bootstrap = '__name__="authenticated_inspector";' + bootstrap
    command = ' '.join(shlex.quote(v) for v in ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
                                                PYTHON, '-I', '-B', '-c', bootstrap])
    try:
        with (args.local_dir / 'SSH_STDOUT.raw').open('xb') as out, (args.local_dir / 'SSH_STDERR.raw').open('xb') as err:
            try:
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                                         '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4', 'eee-cluster', command],
                                        input=json.dumps(payload).encode(), stdout=out, stderr=err, check=False, timeout=300)
            finally:
                out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
        stdout = (args.local_dir / 'SSH_STDOUT.raw').read_bytes()
        report = decode(stdout) if stdout.strip() else None
        success = False
        if result.returncode == 0:
            success = validate_report(report, digest(source))
        receipt = dict(status='OBSERVATION_CAPTURED' if success else 'INSPECTION_FAILED_UNKNOWN',
                       permission=PERMISSION, original_unknown_preserved=True, returncode=result.returncode,
                       report=report, source_sha256=digest(source),
                       raw_logs={name: dict(path=str(args.local_dir / name),
                                           sha256=digest((args.local_dir / name).read_bytes()))
                                 for name in ('SSH_STDOUT.raw', 'SSH_STDERR.raw')})
        write_once(args.local_dir / 'RECEIPT.json', receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0 if success else 2
    except BaseException as exc:
        write_once(args.local_dir / 'ERROR.json', dict(status='INSPECTION_FAILED_UNKNOWN', permission=PERMISSION,
                      error_type=type(exc).__name__, original_unknown_preserved=True,
                      raw_logs={name: dict(path=str(args.local_dir / name),
                                          sha256=digest((args.local_dir / name).read_bytes()))
                                for name in ('SSH_STDOUT.raw', 'SSH_STDERR.raw')
                                if (args.local_dir / name).is_file()}))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
