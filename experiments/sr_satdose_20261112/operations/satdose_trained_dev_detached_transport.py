"""Transport for the November 25 recovery of 11031/T1 DEV16 only.

Use: python THIS.py prepare|submit [--collect] -- <original native CLI arguments>.
The native CLI creates its own intent, logs and receipt. --collect requires the
same native inputs and a fresh local output receipt; it never starts a worker.
Only initial launch uses a mutating SSH request. All reconciliation is read-only.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


PROJECT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112')
BASE = PROJECT / 'transport_trained_dev_11031_T1_s016_20261125_v1'
DATA = PROJECT / 'prepared_data_v1'
PYTHON = '/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
TAG = '71018e5_20261125_recovered'
RESULT = f'execution_releases/TRAINED_DEV_{TAG}_start-11031_T1_step-016_PREPARATION_RESULT.json'
TRAIN = {'path': 'execution_releases/FIRST_TWO_RELEASE_71018e5.json',
         'sha256': '22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f'}
CANARY = {'path': 'execution_releases/canary_job_1740834/CANARY_SPEC.json',
          'sha256': '44a8db68c655bea987629ea210da1a32e268806f6e0c7b1506d6d77828d4b260'}
SOURCE = {'prepare': '188b510417dd648971ee1216d6468a0fb5b0d4297fb5f95158cd1ac65b468b88',
          'submit': '40c465fea1719dc13d31b390c7583180b00d5cd9d05e90addee7f04bcc307bf5'}
REMOTE = {'prepare': '953e67fdab44589bd3c0e5e03ab3731ee97ed8c273278e888f91d0b3b11955d9',
          'submit': '9375cbdf79b9c97a8810336b74bbb41a67b0cd3f78ebd3e4cf4eb67ac4694110'}
# Original 11031/T1/16 preparation request with only the approved tag changed.
PREP_REQUEST_SHA = 'f86fb6ca81e0627f4fb965e3a22c2646070469e8579f490e3f6077275d485966'
SOURCE84_SHA = 'c54293036f63093ab1203180476a85a73aec2ff05e1769cff3cbb6036e2383df'
EXEC19_SHA = 'ab25b75ab7abb255a7f8d795a20f4f930be1a6cb3a61bb7d1f16ac46003869cf'
SSH_FLAGS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
             '-o', 'ServerAliveCountMax=4', 'eee-cluster']
NATIVE_PREFIX = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
                 'TRANSFORMERS_OFFLINE=1', PYTHON, '-I', '-B', '-c']
LOCAL_ROOT = Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ARTIFACTS = LOCAL_ROOT / 'artifacts/sr_satdose_20261112'
SELF = Path(__file__).absolute()
RAW_RUN = subprocess.run


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unpack(text):
    return base64.b64decode(text, validate=True)


def pack(raw):
    return base64.b64encode(raw).decode()


def read(path):
    check(path.resolve(strict=True) == path, 'Noncanonical path: ' + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        check(stat.S_ISREG(os.fstat(fd).st_mode), 'Expected a regular file')
        return f.read()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path, raw):
    check(path.parent.resolve(strict=True) == path.parent, 'Noncanonical parent')
    # Never expose a partial JSON record, including when two clients reconnect
    # while a claim or completed child wait is being published.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.publishing-') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(f.name, path, follow_symlinks=False)
    sync_dir(path.parent)


def private_dir(path):
    check(path.resolve(strict=True) == path and path.is_dir(), 'Noncanonical transport directory')
    st = path.stat()
    check(st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode) == 0o700, 'Transport directory permissions differ')


def validate(op, argv, stdin, code_sha):
    check(op in SOURCE and argv[:-1] == NATIVE_PREFIX, 'Unexpected native operation or environment')
    check(digest(argv[-1].encode()) == REMOTE[op], 'Unexpected native remote program')
    p = json.loads(stdin)
    check((p['start_seed'], p['arm'], p['step']) == (11031, 'T1', 16), 'Unexpected frozen endpoint')
    if op == 'prepare':
        check(digest(encoded(p)) == PREP_REQUEST_SHA and p['root'] == str(DATA) and p['code'] == str(DATA/'train_code_71018e5')
              and p['tag'] == TAG and p['budget_hours'] == 168 and p['preparation_script_sha256'] == SOURCE[op]
              and p['original_train_release'] == TRAIN and p['canary_spec'] == CANARY, 'Unexpected preparation lineage')
    else:
        check(digest(encoded(p['sources'])) == SOURCE84_SHA and digest(encoded(p['execution_sources'])) == EXEC19_SHA
              and p['source_sha256'] == SOURCE[op] and p['preparer_sha'] == SOURCE['prepare'] and p['original'] == TRAIN
              and p['preparation']['release']['path'] == RESULT.replace('_PREPARATION_RESULT.json', '_RELEASE.json')
              and p['preparation_ref']['path'] == RESULT and all(p['preparation'][k] == p[k] for k in
              ('start_seed','arm','step')), 'Unexpected submission lineage')
    return dict(op=op, native_source_sha256=SOURCE[op], remote_source_sha256=REMOTE[op],
                transport_source_sha256=code_sha, argv_sha256=digest(encoded(argv)), stdin_sha256=digest(stdin))


def identify(pid):
    row = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, hostname=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=int(row[19]))


def new_log(path):
    return os.fdopen(os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600), 'wb')


def run_child(d, argv, stdin):
    with new_log(d/'stdout.bin') as out, new_log(d/'stderr.bin') as err:
        with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err, close_fds=True) as child:
            save(d/'NATIVE.json', encoded(identify(child.pid)))
            child.communicate(input=stdin)
            rc = child.returncode
        for f in (out, err):
            f.flush()
            os.fsync(f.fileno())
    return rc


def worker(op, key):
    d = BASE / op
    private_dir(BASE)
    private_dir(d)
    request = read(d/'CLAIM.json')
    check(digest(request) == key, 'Worker claim changed')
    claim = json.loads(request)
    argv = json.loads(read(d/'argv.json'))
    stdin = read(d/'stdin.bin')
    check(validate(op, argv, stdin, digest(read(SELF))) == claim, 'Worker inputs changed')
    save(d/'WORKER.json', encoded(identify(os.getpid())))  # Single executor, including after a crash.
    rc = run_child(d, argv, stdin)
    record = dict(key=key, returncode=rc, native=json.loads(read(d/'NATIVE.json')),
                  stdout_sha256=digest(read(d/'stdout.bin')), stderr_sha256=digest(read(d/'stderr.bin')))
    save(d/'WAIT.json', encoded(record))


def launch(d, op, claim, argv, stdin, source):
    check(d == BASE/op and op in SOURCE and PROJECT.resolve(strict=True) == PROJECT, 'Noncanonical transport root')
    BASE.mkdir(mode=0o700, exist_ok=True)
    private_dir(BASE)
    sync_dir(BASE.parent)
    d.mkdir(mode=0o700, exist_ok=False)  # Never enter another invocation, even for identical inputs.
    sync_dir(BASE)
    private_dir(d)
    save(d/'CLAIM.json', encoded(claim))
    key = digest(encoded(claim))
    save(d/'argv.json', encoded(argv))
    save(d/'stdin.bin', stdin)
    script = d/'TRANSPORT.py'
    save(script, source)
    with new_log(d/'worker.stderr') as err:
        subprocess.Popen([PYTHON, '-I', '-B', str(script), '--worker', op, key], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, close_fds=True)


def observe(d, claim):
    if os.path.lexists(BASE):
        private_dir(BASE)
    if not os.path.lexists(d):
        return dict(state='ABSENT')
    private_dir(d)
    if not os.path.lexists(d/'CLAIM.json'):
        return dict(state='INITIALIZING')
    request = read(d/'CLAIM.json')
    check(request == encoded(claim), 'Different attempt already owns this operation')
    key = digest(request)
    if os.path.lexists(d/'WAIT.json'):
        wait = json.loads(read(d/'WAIT.json'))
        check(wait['key'] == key and type(wait['returncode']) is int, 'Unexpected child wait record')
        out, err = read(d/'stdout.bin'), read(d/'stderr.bin')
        check(digest(out) == wait['stdout_sha256'] and digest(err) == wait['stderr_sha256'], 'Child result bytes changed')
        return dict(state='FINISHED', wait=wait, stdout=pack(out), stderr=pack(err))
    if not os.path.lexists(d/'WORKER.json'):
        return dict(state='INITIALIZING')
    identity = json.loads(read(d/'WORKER.json'))
    if identity['hostname'] == socket.gethostname():
        try:
            same = identify(identity['pid']) == identity
        except (FileNotFoundError, ProcessLookupError):
            same = False
        if not same:
            # WAIT is linked before normal exit. Close a race with its publication.
            if os.path.lexists(d/'WAIT.json'):
                return observe(d, claim)
            return dict(state='WORKER_GONE_WITHOUT_WAIT', worker=identity)
    return dict(state='PENDING', worker=identity)


def serve(action, op, claim, payload, source):
    check(op in SOURCE and claim['op'] == op and claim['transport_source_sha256'] == digest(source), 'Invalid request')
    d = BASE/op
    if action == 'launch':
        argv, stdin = payload['argv'], unpack(payload['stdin'])
        check(validate(op, argv, stdin, digest(source)) == claim, 'Launch binding mismatch')
        launch(d, op, claim, argv, stdin, source)
    else:
        check(action == 'observe', 'Unknown transport action')
    return observe(d, claim)


BOOTSTRAP = """import json,sys,base64,hashlib
x=json.load(sys.stdin);src=base64.b64decode(x['source'],validate=True)
if hashlib.sha256(src).hexdigest()!=sys.argv[1]: raise ValueError('transport SHA mismatch')
ns={'__name__':'detached_transport','__file__':'/transport-bootstrap'}
exec(compile(src,'<detached-transport>','exec'),ns)
ans=ns['serve'](x['action'],x['op'],x['claim'],x['payload'],src)
print(json.dumps(ans,sort_keys=True,allow_nan=False))
"""


def exchange(action, op, claim, payload, source):
    message = encoded(dict(action=action, op=op, claim=claim, payload=payload, source=pack(source)))
    remote = shlex.join([PYTHON, '-I', '-B', '-c', BOOTSTRAP, digest(source)])
    return RAW_RUN(['/usr/bin/ssh', *SSH_FLAGS, remote], input=message, capture_output=True, check=False)


def relay(op, argv, stdin, stdout, stderr, receipt, collect, source):
    claim = validate(op, argv, stdin, digest(source))
    key = digest(encoded(claim))
    prefix = receipt.with_name(receipt.name+'.transport')
    save(Path(str(prefix)+'.source.py'), source)
    save(Path(str(prefix)+'.json'), encoded(dict(claim=claim, key=key, remote_directory=str(BASE/op),
                                                 collect_only=collect, native_receipt=str(receipt))))
    action = 'observe' if collect else 'launch'
    payload = {} if collect else dict(argv=argv, stdin=pack(stdin))
    missing_since = None
    last_network_error = None
    while True:
        reply = exchange(action, op, claim, payload, source)
        first_launch = action == 'launch'
        action, payload = 'observe', {}  # No path in this loop can transmit another launch.
        if first_launch:
            save(Path(str(prefix)+'.launch.json'), encoded(dict(ssh_returncode=reply.returncode, stderr=pack(reply.stderr))))
        if reply.returncode:
            if reply.returncode != 255 and not first_launch:
                raise RuntimeError('Read-only reconciliation failed: '+reply.stderr.decode(errors='replace')[-800:])
            if reply.returncode == 255 and reply.stderr != last_network_error:
                print('DEV16 transport waiting for SSH; subsequent requests are read-only.', file=sys.stderr, flush=True)
                last_network_error = reply.stderr
            time.sleep(60)
            continue
        answer = json.loads(reply.stdout)
        state = answer['state']
        if state == 'FINISHED':
            wait = answer['wait']
            out, err = unpack(answer['stdout']), unpack(answer['stderr'])
            check(wait['key'] == key and type(wait['returncode']) is int and
                  digest(out) == wait['stdout_sha256'] and digest(err) == wait['stderr_sha256'], 'Invalid returned wait/bytes')
            save(Path(str(prefix)+'.wait.json'), encoded(dict(wait=wait, provenance='REMOTE_NATIVE_CHILD_WAIT',
                                                              final_collection_ssh_returncode=reply.returncode)))
            stdout.write(out)
            stderr.write(err)
            return subprocess.CompletedProcess(argv, wait['returncode'])
        check(state in ('PENDING','INITIALIZING','ABSENT'), 'Native remote result unknown: '+str(answer))
        if state == 'PENDING':
            missing_since = None
        else:
            if missing_since is None:
                missing_since = time.monotonic()
            check(time.monotonic()-missing_since < 120, 'No worker acknowledgement; inspect attempt without relaunch')
        time.sleep(60)


def native(op, args, collect):
    args = list(args)
    check(args.count('--receipt') == 1 and ('--execute-preparation' if op == 'prepare' else '--execute') in args,
          'Explicit native execution flag and fresh absolute receipt are required')
    receipt = Path(args[args.index('--receipt')+1])
    check(receipt.is_absolute() and receipt.is_relative_to(ARTIFACTS) and receipt.parent.resolve(strict=True) == receipt.parent
          and not os.path.lexists(receipt), 'Expected a fresh project receipt')
    path = LOCAL_ROOT/'experiments/sr_satdose_20261112/operations'/f'satdose_{op}_trained_dev.py'
    raw = read(path)
    check(digest(raw) == SOURCE[op], 'Original local CLI source differs')
    ns = {'__name__': 'verified_native_cli', '__file__': str(path)}
    exec(compile(raw, str(path), 'exec'), ns)
    calls = []
    def bridge(args, *, input, text, stdout, stderr, check):
        if (calls or args[:-1] != ['ssh',*SSH_FLAGS] or text is not True or check is not False
                or not isinstance(input, str)):
            raise RuntimeError('Unexpected native SSH call')
        calls.append(True)
        argv = shlex.split(args[-1])
        if shlex.join(argv) != args[-1] or argv[-1] != ns['REMOTE']:
            raise RuntimeError('Noncanonical native SSH command')
        for stream, suffix in ((stdout,'.stdout.log'),(stderr,'.stderr.log')):
            dest = receipt.with_name(receipt.name+suffix)
            observed = os.fstat(stream.fileno())
            if dest.resolve(strict=True) != dest or not os.path.samestat(observed, dest.stat()):
                raise RuntimeError('Native output is not bound to expected receipt')
        return relay(op, argv, input.encode(), stdout, stderr, receipt, collect, read(SELF))
    ns['subprocess'] = SimpleNamespace(run=bridge)
    old = sys.argv
    try:
        sys.argv = [str(path), *args]
        result = ns['main']()
        check(calls == [True], 'Native CLI did not issue its expected request')
        return result
    finally:
        sys.argv = old


def main(argv):
    if argv[:1] == ['--worker']:
        check(len(argv) == 3 and argv[1] in SOURCE and SELF == BASE/argv[1]/'TRANSPORT.py', 'Invalid worker entry')
        worker(argv[1], argv[2])
        return 0
    check('--' in argv, 'Separate transport options and native CLI arguments with --')
    split = argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=SOURCE)
    parser.add_argument('--collect', action='store_true')
    options = parser.parse_args(argv[:split])
    return native(options.operation, argv[split+1:], options.collect)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
