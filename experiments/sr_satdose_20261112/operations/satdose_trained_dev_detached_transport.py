"""Slurm CPU transport for the November 25 recovery of 11031/T1 DEV16 only.

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
import re
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
BASE = PROJECT / 'transport_trained_dev_11031_T1_s016_20261125_cpu_v2'
QOS = 'soujanya-poria-startfund-2026-03'
USER = 'varun024'
ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'}
ACCOUNT_FIELDS = 'JobIDRaw,State%40,ExitCode,JobName%96,User,Account,QOS%80,Partition,ReqTRES%256,AllocTRES%256,TimelimitRaw,WorkDir%300'
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


def latest(path, raw):
    check(path.parent.resolve(strict=True)==path.parent and not path.is_symlink(), 'Noncanonical local status path')
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.cpu-status-')
    try:
        with os.fdopen(fd,'wb') as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp,path)
        sync_dir(path.parent)
    finally:
        if os.path.lexists(temp):
            os.unlink(temp)


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


def clean_environment():
    # Do not export the CPU allocation's identity/resource selectors to the
    # original submitter's nested GPU sbatch or leak array/requeue controls.
    return {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_', 'SLURMD_', 'SBATCH_', 'SRUN_'))}


def run_child(d, argv, stdin):
    with new_log(d/'stdout.bin') as out, new_log(d/'stderr.bin') as err:
        with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err, close_fds=True, env=clean_environment()) as child:
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
    cpu_id = os.environ.get('SLURM_JOB_ID', '')
    check(cpu_id.isdecimal() and int(cpu_id)>0 and os.environ.get('SLURM_RESTART_COUNT','0')=='0',
          'Worker outside first Slurm CPU allocation')
    verify_intent(d, op, key)
    fields = live_identity(d, op, key, cpu_id, running=True)
    candidates=scheduler_ids(d,op,key)
    check(not candidates-{cpu_id}, 'Another CPU job has this wrapper identity')
    if os.path.lexists(d/'SBATCH_RESPONSE.json'):
        response=json.loads(read(d/'SBATCH_RESPONSE.json'))
        if response['returncode']==0:
            check(response['key']==key and unpack(response['stdout']).strip()==cpu_id.encode(),
                  'CPU executor differs from accepted wrapper')
    record = dict(job_id=cpu_id, key=key, identity=identify(os.getpid()), slurm=fields)
    save(d/'WORKER.json', encoded(record))  # Even an unrequested requeue cannot re-run this native operation.
    rc = run_child(d, argv, stdin)
    record = dict(key=key, cpu_job_id=cpu_id, returncode=rc, native=json.loads(read(d/'NATIVE.json')),
                  stdout_sha256=digest(read(d/'stdout.bin')), stderr_sha256=digest(read(d/'stderr.bin')))
    save(d/'WAIT.json', encoded(record))
    return shell_exit(rc)


def shell_exit(rc):
    return 128-rc if rc < 0 else rc


def scheduler(argv):
    return RAW_RUN(argv, capture_output=True, check=False, env=clean_environment())


class SchedulerUnavailable(Exception):
    pass


def query(argv):
    result = scheduler(argv)
    if result.returncode:
        raise SchedulerUnavailable('Slurm query failed: '+result.stderr.decode(errors='replace')[-600:])
    return result.stdout.decode()


def job_name(op, key):
    return f'LH-SATDOSE-CPU-{op}-11031-T1-s16-{key[:16]}'


def cpu_script(d, op, key):
    return ('#!/bin/sh\nset -eu\nexport CUDA_VISIBLE_DEVICES=\nexec ' +
            shlex.join([PYTHON,'-I','-B',str(d/'TRANSPORT.py'),'--worker',op,key])+'\n').encode()


def cpu_command(d, op, key):
    return ['sbatch','--parsable','--account=rose','--partition=cluster02','--qos='+QOS,
            '--job-name='+job_name(op,key),'--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=6G',
            '--time=04:00:00','--no-requeue','--export=ALL','--chdir='+str(d),
            '--output='+str(d/'cpu-%j.out'),'--error='+str(d/'cpu-%j.err'),str(d/'CPU_WRAPPER.sh')]


def verify_intent(d, op, key):
    expected = dict(key=key, argv=cpu_command(d,op,key), script_sha256=digest(cpu_script(d,op,key)))
    check(json.loads(read(d/'SBATCH_INTENT.json'))==expected and
          digest(read(d/'CPU_WRAPPER.sh'))==expected['script_sha256'], 'CPU submission identity differs')


def verify_tres(text):
    entries = text.split(',')
    tres = dict(item.split('=',1) for item in entries)
    check(len(tres)==len(entries) and tres.get('cpu')=='2' and tres.get('node')=='1'
          and tres.get('mem') in ('6G','6144M','6144','6291456K')
          and all(int(v)==0 for k,v in tres.items() if k=='gres/gpu' or k.startswith('gres/gpu:')),
          'Expected 2 CPUs, 6 GiB and zero GPUs for wrapper')


def live_identity(d, op, key, jid, *, running=False):
    text = query(['scontrol','show','job','--oneliner',jid])
    fields = dict(part.split('=',1) for part in text.split() if '=' in part)
    expected = dict(JobId=jid,JobName=job_name(op,key),Account='rose',QOS=QOS,Partition='cluster02',
                    TimeLimit='04:00:00',Requeue='0',NumCPUs='2',NumTasks='1',WorkDir=str(d),
                    Command=str(d/'CPU_WRAPPER.sh'),StdOut=str(d/f'cpu-{jid}.out'),StdErr=str(d/f'cpu-{jid}.err'))
    check(all(fields.get(k)==v for k,v in expected.items()) and fields.get('UserId','').startswith(USER+'(')
          and fields.get('NumNodes') in ('1','1-1'), 'Slurm CPU wrapper identity differs')
    verify_tres(fields['ReqTRES'])
    if running:
        check(fields.get('JobState')=='RUNNING','Worker allocation is not RUNNING')
        verify_tres(fields['AllocTRES'])
    # Store only this CPU allocation's verified fields.
    return {k:fields.get(k) for k in (*expected,'UserId','JobState','ReqTRES','AllocTRES')}


def scheduler_ids(d, op, key):
    name = job_name(op,key)
    # squeue and sacct may briefly disagree. Every visible candidate must bind
    # to this endpoint-specific name; ambiguous results never start another job.
    commands = (['squeue','--user='+USER,'--name='+name,'--noheader','--format=%i|%200j'],
                ['sacct','-X','--user='+USER,'--starttime=2026-11-25T00:00:00','--name='+name,
                 '-nP','-o','JobIDRaw,JobName%160'])
    ids=set()
    for argv in commands:
        for row in query(argv).splitlines():
            jid, observed_name = (v.strip() for v in row.split('|'))
            check(jid.isdecimal() and observed_name==name, 'Foreign Slurm identity on wrapper lookup')
            ids.add(jid)
    return ids


def accounting(d, op, key, jid):
    raw = query(['sacct','-nP','-j',jid,'-o',ACCOUNT_FIELDS])
    result = {}
    for line in raw.splitlines():
        r = line.strip().split('|')
        check(len(r)==12 and r[0] in (jid,jid+'.batch',jid+'.extern') and r[0] not in result,
              'Unexpected Slurm CPU accounting row')
        if r[0]==jid:
            check(r[3:8]==[job_name(op,key),USER,'rose',QOS,'cluster02'] and r[10:12]==['240',str(d)],
                  'CPU wrapper accounting identity changed')
            verify_tres(r[8])
            if r[9] and r[1]!='PENDING':
                verify_tres(r[9])
        result[r[0]] = r[1:3]
    return result


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
    command = cpu_command(d,op,key)
    script = cpu_script(d,op,key)
    save(d/'CPU_WRAPPER.sh',script)
    save(d/'SBATCH_INTENT.json',encoded(dict(key=key,argv=command,script_sha256=digest(script))))
    # The only wrapper submission. A disconnect before the response is persisted
    # is reconciled by exact job name/claim/worker/accounting, never resubmitted.
    response = scheduler(command)
    save(d/'SBATCH_RESPONSE.json',encoded(dict(key=key,returncode=response.returncode,
                                              stdout=pack(response.stdout),stderr=pack(response.stderr))))


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
    op,key=claim['op'],digest(request)
    if not os.path.lexists(d/'SBATCH_INTENT.json'):
        return dict(state='INITIALIZING')
    verify_intent(d,op,key)
    ids=scheduler_ids(d,op,key)
    response=None
    if os.path.lexists(d/'SBATCH_RESPONSE.json'):
        response=json.loads(read(d/'SBATCH_RESPONSE.json'))
        check(response['key']==key,'Foreign sbatch response')
        raw_id=unpack(response['stdout']).strip()
        if response['returncode']==0 and re.fullmatch(rb'[1-9][0-9]*',raw_id):
            ids.add(raw_id.decode())
    worker_record=None
    if os.path.lexists(d/'WORKER.json'):
        worker_record=json.loads(read(d/'WORKER.json'))
        check(worker_record['key']==key and re.fullmatch('[1-9][0-9]*',worker_record['job_id']),
              'Foreign CPU executor')
        ids.add(worker_record['job_id'])
    check(len(ids)<=1,'Conflicting CPU wrapper JobIDs')
    if not ids:
        return dict(state='INITIALIZING',sbatch_response=response)
    jid=ids.pop()
    rows=accounting(d,op,key,jid)
    cpu=dict(job_id=jid,job_name=job_name(op,key),accounting=rows,sbatch_response=response)
    parent=rows.get(jid)
    if parent is None:
        # A known job may predate accounting visibility. Require a matching
        # live identity before reporting it as pending and never retry sbatch.
        live=live_identity(d,op,key,jid)
        return dict(state='PENDING',cpu=cpu,slurm_state=live['JobState'])
    if parent[0] in ACTIVE:
        return dict(state='PENDING',cpu=cpu,slurm_state=parent[0])
    if parent[0] not in ('FAILED','COMPLETED') or not worker_record or not os.path.lexists(d/'WAIT.json'):
        return dict(state='CPU_WRAPPER_TERMINATED_WITHOUT_CONFIRMED_NATIVE_WAIT',cpu=cpu)
    wait=json.loads(read(d/'WAIT.json'))
    check(wait['key']==key and wait['cpu_job_id']==jid and type(wait['returncode']) is int,
          'Unexpected child wait record')
    rc=wait['returncode']
    expected=['COMPLETED','0:0'] if rc==0 else ['FAILED',str(shell_exit(rc))+':0']
    check(parent==expected, 'CPU wrapper scheduler exit and real child wait disagree')
    # Step accounting can arrive after the parent. Wait for the batch step.
    if jid+'.batch' not in rows or rows[jid+'.batch'][0] in ACTIVE or rows.get(jid+'.extern',[''])[0] in ACTIVE:
        return dict(state='PENDING',cpu=cpu,slurm_state='WAIT_FOR_BATCH_ACCOUNTING')
    check(rows[jid+'.batch']==expected and (jid+'.extern' not in rows or
          rows[jid+'.extern']==['COMPLETED','0:0']), 'Unsuccessful wrapper auxiliary step')
    out,err=read(d/'stdout.bin'),read(d/'stderr.bin')
    check(digest(out)==wait['stdout_sha256'] and digest(err)==wait['stderr_sha256'], 'Child result bytes changed')
    if op=='submit' and rc==0:
        marker=b'SATDOSE_SUBMIT_JSON='
        results=[json.loads(line[len(marker):]) for line in out.splitlines() if line.startswith(marker)]
        check(len(results)==1 and type(results[0]['result']['job_id']) is int and
              results[0]['result']['job_id']>0 and str(results[0]['result']['job_id'])!=jid,
              'Actual submitted GPU job ID missing or equal to CPU wrapper job')
        cpu['native_submitted_gpu_job_id']=results[0]['result']['job_id']
    return dict(state='FINISHED',wait=wait,cpu=cpu,stdout=pack(out),stderr=pack(err))


def serve(action, op, claim, payload, source):
    check(op in SOURCE and claim['op'] == op and claim['transport_source_sha256'] == digest(source), 'Invalid request')
    d = BASE/op
    if action == 'launch':
        argv, stdin = payload['argv'], unpack(payload['stdin'])
        check(validate(op, argv, stdin, digest(source)) == claim, 'Launch binding mismatch')
        launch(d, op, claim, argv, stdin, source)
    else:
        check(action == 'observe', 'Unknown transport action')
    try:
        return observe(d, claim)
    except SchedulerUnavailable as exc:
        return dict(state='SCHEDULER_UNAVAILABLE',error=str(exc))


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
    previous_status = None
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
        status = {k:v for k,v in answer.items() if k not in ('stdout','stderr','wait')}
        if status != previous_status:
            latest(Path(str(prefix)+'.scheduler.json'), encoded(status))
            previous_status=status
        if state == 'SCHEDULER_UNAVAILABLE':
            time.sleep(60)
            continue
        if state == 'FINISHED':
            wait = answer['wait']
            out, err = unpack(answer['stdout']), unpack(answer['stderr'])
            check(wait['key'] == key and type(wait['returncode']) is int and
                  digest(out) == wait['stdout_sha256'] and digest(err) == wait['stderr_sha256'], 'Invalid returned wait/bytes')
            save(Path(str(prefix)+'.wait.json'), encoded(dict(wait=wait, provenance='REMOTE_NATIVE_CHILD_WAIT',
                                                              final_collection_ssh_returncode=reply.returncode, cpu=answer['cpu'])))
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
        return worker(argv[1], argv[2])
    check('--' in argv, 'Separate transport options and native CLI arguments with --')
    split = argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=SOURCE)
    parser.add_argument('--collect', action='store_true')
    options = parser.parse_args(argv[:split])
    return native(options.operation, argv[split+1:], options.collect)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
