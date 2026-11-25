"""Offline transport regression tests. No SSH, cluster, GPU or scheduler calls."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'satdose_trained_dev_detached_transport.py'
spec = importlib.util.spec_from_file_location('transport_test_target', SOURCE)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
BASELINE = t.ARTIFACTS / 'trained_dev_20261124'
PREP_ARGS = ['--start-seed','11031','--arm','T1','--step','16','--publication-tag',t.TAG,
             '--original-train-release',t.TRAIN['path'],'--original-train-sha256',t.TRAIN['sha256'],
             '--canary-spec',t.CANARY['path'],'--canary-spec-sha256',t.CANARY['sha256'],'--execute-preparation']


def remote(op):
    native = SOURCE.parent/f'satdose_{op}_trained_dev.py'
    assert t.digest(native.read_bytes()) == t.SOURCE[op]
    tree = ast.parse(native.read_bytes())
    return next(ast.literal_eval(x.value) for x in tree.body if isinstance(x, ast.Assign)
                and any(isinstance(k, ast.Name) and k.id == 'REMOTE' for k in x.targets))


def prepare_input():
    p = json.loads((BASELINE/'start-11031_T1_step-016/PREPARATION.json.intent.json').read_bytes())['request']
    p['tag'] = t.TAG
    return p


def response(rc, value=None, error=b''):
    return subprocess.CompletedProcess([], rc, t.encoded(value) if value is not None else b'', error)


def finished(claim, stdout=b'test\r\n', stderr=b'\x00\xff', rc=17):
    wait = dict(key=t.digest(t.encoded(claim)), returncode=rc, stdout_sha256=t.digest(stdout), stderr_sha256=t.digest(stderr))
    return dict(state='FINISHED', wait=wait, stdout=t.pack(stdout), stderr=t.pack(stderr))


class DetachedTransport(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name).resolve()
        self.argv = t.NATIVE_PREFIX + [remote('prepare')]
        self.raw = t.encoded(prepare_input())
        self.source = SOURCE.read_bytes()
        self.claim = t.validate('prepare', self.argv, self.raw, t.digest(self.source))

    def test_frozen_command_and_input_rules(self):
        for mutation in ('wrong_tag','foreign_endpoint','different_sources','changed_program','env_option'):
            obj = prepare_input()
            args = self.argv.copy()
            if mutation == 'wrong_tag':
                obj['tag'] += '_another'
            elif mutation == 'foreign_endpoint':
                obj['start_seed'] = 11032
            elif mutation == 'different_sources':
                obj['sources'].clear()
            elif mutation == 'changed_program':
                args[-1] += '\n# change'
            else:
                args[1] = 'CUDA_VISIBLE_DEVICES=0'
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                t.validate('prepare', args, t.encoded(obj), t.digest(self.source))

    def test_only_one_mutating_exchange_after_lost_ack_and_subsequent_disconnect(self):
        record = []
        def exchange(action, op, claim, payload, source):
            record.append(action)
            if len(record) in (1, 3):
                return response(255, error=b'network unavailable')
            if len(record) == 2:
                return response(0, dict(state='PENDING'))
            return response(0, finished(claim))
        out, err = io.BytesIO(), io.BytesIO()
        with patch.object(t,'exchange',exchange), patch.object(t.time,'sleep'), contextlib.redirect_stderr(io.StringIO()):
            receipt = self.dir/'NATIVE'
            res = t.relay('prepare',self.argv,self.raw,out,err,receipt,False,self.source)
        self.assertEqual(record,['launch','observe','observe','observe'])
        self.assertEqual((res.returncode,out.getvalue(),err.getvalue()),(17,b'test\r\n',b'\x00\xff'))
        self.assertEqual(json.loads((self.dir/'NATIVE.transport.launch.json').read_bytes())['ssh_returncode'],255)
        self.assertEqual(json.loads((self.dir/'NATIVE.transport.wait.json').read_bytes())['provenance'],'REMOTE_NATIVE_CHILD_WAIT')

    def test_collect_never_launches_and_rejects_changed_result(self):
        record = []
        def exchange(action, op, claim, payload, source):
            record.append((action,payload))
            answer = finished(claim)
            answer['stdout'] = t.pack(b'altered')
            return response(0,answer)
        out, err = io.BytesIO(), io.BytesIO()
        with patch.object(t,'exchange',exchange), self.assertRaisesRegex(RuntimeError,'wait/bytes'):
            t.relay('prepare',self.argv,self.raw,out,err,self.dir/'NATIVE',True,self.source)
        self.assertEqual(record,[('observe',{})])
        self.assertEqual((out.getvalue(),err.getvalue()),(b'',b''))

    def test_absent_after_lost_launch_requires_resolution(self):
        record = []
        def exchange(action, *args):
            record.append(action)
            return response(255) if action == 'launch' else response(0,dict(state='ABSENT'))
        with patch.object(t,'exchange',exchange), patch.object(t.time,'sleep'), \
             patch.object(t.time,'monotonic',side_effect=[0,0,130]), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError,'No worker acknowledgement'):
            t.relay('prepare',self.argv,self.raw,io.BytesIO(),io.BytesIO(),self.dir/'N',False,self.source)
        self.assertEqual(record,['launch','observe','observe'])

    def test_live_native_prepare_preserves_original_source_and_observed_output(self):
        dest = self.dir/'PREPARATION.json'
        def exchange(action, op, claim, payload, source):
            self.assertEqual(action,'launch')
            self.assertEqual(claim['native_source_sha256'],t.SOURCE['prepare'])
            self.assertEqual(json.loads(t.unpack(payload['stdin'])),prepare_input())
            return response(0,finished(claim, stdout=b'', stderr=b'fixture native error\r\n',rc=13))
        with patch.object(t,'ARTIFACTS',self.dir), patch.object(t,'exchange',exchange), contextlib.redirect_stdout(io.StringIO()):
            rc = t.native('prepare',PREP_ARGS+['--receipt',str(dest)],False)
        receipt=json.loads(dest.read_bytes())
        self.assertNotEqual(rc,0)
        self.assertEqual(receipt['returncode'],13)
        self.assertEqual(Path(receipt['source']['path']).read_bytes(), (SOURCE.parent/'satdose_prepare_trained_dev.py').read_bytes())
        self.assertEqual(Path(receipt['stderr_file']['path']).read_bytes(),b'fixture native error\r\n')

    def test_native_submit_binds_successful_preparer_then_replays_real_child_exit(self):
        # Only files within temp are fixtures. The original submitter checks its
        # own preparation lineage, source84 and execution19 from the real intake.
        previous = json.loads((BASELINE/'start-11031_T0_step-016/PREPARATION.json').read_bytes())['response']
        old = '_71018e5_20261116_start-11031_T0_step-016'
        new = '_'+t.TAG+'_start-11031_T1_step-016'
        synthetic = json.loads(json.dumps(previous).replace(old,new))
        synthetic['result']['arm']='T1'
        output = b'SATDOSE_PREPARE_JSON='+t.encoded(synthetic)+b'\n'
        prepared=self.dir/'P.json'; submitted=self.dir/'S.json'
        def exchange(action,op,claim,payload,source):
            self.assertEqual(action, 'observe' if op=='submit' else 'launch')
            if op=='prepare':
                return response(0,finished(claim,stdout=output,stderr=b'',rc=0))
            self.assertEqual(claim['remote_source_sha256'],t.digest(remote('submit').encode()))
            return response(0,finished(claim,stdout=b'',stderr=b'fixture native submit failure',rc=23))
        with patch.object(t,'ARTIFACTS',self.dir), patch.object(t,'exchange',exchange), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(t.native('prepare',PREP_ARGS+['--receipt',str(prepared)],False),0)
            native_args=['--preparation-receipt',str(prepared),'--preparation-sha256',t.digest(prepared.read_bytes()),
                         '--receipt',str(submitted),'--execute']
            self.assertNotEqual(t.native('submit',native_args,True),0)
        receipt=json.loads(submitted.read_bytes())
        self.assertEqual(receipt['returncode'],23)
        self.assertEqual(receipt['raw_logs']['stderr']['sha256'],t.digest(b'fixture native submit failure'))
        self.assertEqual(receipt['source']['sha256'],t.SOURCE['submit'])

    def test_claim_rejects_foreign_input_and_symlink(self):
        with patch.object(t,'BASE',self.dir):
            d=self.dir/'prepare'; d.mkdir(mode=0o700)
            t.save(d/'CLAIM.json',t.encoded(dict(self.claim,stdin_sha256='0'*64)))
            with self.assertRaisesRegex(RuntimeError,'Different attempt'):
                t.observe(d,self.claim)
            d2=self.dir/'submit';d2.symlink_to(d)
            with self.assertRaisesRegex(RuntimeError,'Noncanonical'):
                t.observe(d2,self.claim)

    def test_worker_disappearance_does_not_return_wait(self):
        with patch.object(t,'BASE',self.dir), patch.object(t,'identify',side_effect=ProcessLookupError):
            d=self.dir/'prepare';d.mkdir(mode=0o700)
            t.save(d/'CLAIM.json',t.encoded(self.claim))
            t.save(d/'WORKER.json',t.encoded(dict(pid=12345,hostname=socket.gethostname())))
            self.assertEqual(t.observe(d,self.claim)['state'],'WORKER_GONE_WITHOUT_WAIT')

    def test_detached_fixture_survives_client_exit_and_cannot_run_twice(self):
        # Real subprocess/setsid, private files, wait, collect; replace only the
        # production paths, the frozen Python payload and Linux /proc identity.
        target = self.dir/'project';target.mkdir()
        base = target/'transport'
        stdin=self.raw
        command = ("import os,sys,time; data=sys.stdin.buffer.read(); time.sleep(0.25); "
                   "sys.stdout.buffer.write(data); sys.stderr.buffer.write(b'unclean\\x00\\xff'); "
                   "sys.exit(14 if os.getsid(0)!=int(os.environ['TEST_LAUNCHER_SID']) else 15)")
        prefix=[*t.NATIVE_PREFIX]; prefix[prefix.index(t.PYTHON)] = sys.executable
        overrides=(f"\nPROJECT=Path({str(target)!r}); BASE=Path({str(base)!r}); PYTHON={sys.executable!r}\n"
                   f"NATIVE_PREFIX={prefix!r}; REMOTE['prepare']={t.digest(command.encode())!r}\n"
                   "def identify(pid):\n    os.kill(pid,0)\n    return dict(pid=pid,hostname=socket.gethostname(),boot_id='fixture',start_ticks=1)\n")
        source=self.source.replace(b"if __name__ == '__main__':",overrides.encode()+b"\nif __name__ == '__main__':")
        argv=prefix+[command]
        claim=dict(self.claim,transport_source_sha256=t.digest(source),remote_source_sha256=t.digest(command.encode()),
                   argv_sha256=t.digest(t.encoded(argv)))
        def channel(action):
            payload=dict(argv=argv,stdin=t.pack(stdin)) if action=='launch' else {}
            message=dict(action=action,op='prepare',source=t.pack(source),claim=claim,payload=payload)
            return subprocess.run([sys.executable,'-I','-B','-c',t.BOOTSTRAP,t.digest(source)],
                                  input=t.encoded(message),capture_output=True,
                                  env=dict(os.environ,TEST_LAUNCHER_SID=str(os.getsid(0))),timeout=5)
        first=channel('launch')
        self.assertEqual(first.returncode,0,first.stderr)
        self.assertIn(json.loads(first.stdout)['state'],('INITIALIZING','PENDING'))
        duplicate=channel('launch')
        self.assertNotEqual(duplicate.returncode,0)
        answer=None
        for i in range(25):
            observed=channel('observe')
            self.assertEqual(observed.returncode,0,observed.stderr)
            answer=json.loads(observed.stdout)
            if answer['state']=='FINISHED':
                break
            time.sleep(0.04)
        self.assertEqual(answer['state'],'FINISHED',answer)
        self.assertEqual(answer['wait']['returncode'],14)
        self.assertEqual(t.unpack(answer['stdout']),stdin)
        self.assertEqual(t.unpack(answer['stderr']),b'unclean\x00\xff')
        self.assertEqual((base/'prepare'/'stdin.bin').stat().st_mode & 0o777,0o600)
        self.assertEqual(json.loads(channel('observe').stdout),answer)


if __name__ == '__main__':
    unittest.main()
