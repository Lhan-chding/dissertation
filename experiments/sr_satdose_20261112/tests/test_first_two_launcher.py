"""Real CPU subprocess/wait tests; synthetic canary artifacts are NOT GPU proof.

The production observer is Linux-only. On macOS the fixture observer obtains
actual PID/start-time/boot metadata through ps/sysctl, with epoch-second start
identity rather than Linux ticks. Popen, wait, exit order and no-retry are real.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from scheduling import first_two_launcher as f

OBSERVER=r'''
import datetime,os,pathlib,socket,subprocess,sys

def observed(pid=None):
    pid=os.getpid() if pid is None else pid
    if sys.platform=='linux':
        raw=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
        ticks=int(raw[19]);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    else:
        raw=subprocess.check_output(['ps','-o','lstart=','-p',str(pid)],text=True).strip()
        ticks=int(datetime.datetime.strptime(raw,'%a %b %d %H:%M:%S %Y').timestamp())
        boot=subprocess.check_output(['sysctl','-n','kern.boottime'],text=True).strip()
    return dict(host=socket.gethostname(),boot_id=boot,pid=pid,start_ticks=ticks)
'''

WORKER=r'''
import hashlib,json,os,pathlib,sys,time
from cpu_observer import observed
root=pathlib.Path(sys.argv[1]);start=int(sys.argv[2]);arm=sys.argv[3];phase=sys.argv[4];release_sha=sys.argv[5]
mode=sys.argv[6];baseline_sha=sys.argv[7] if len(sys.argv)>7 else None

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=str(path.relative_to(root)),sha256=sha(path))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream)
    return ref(path)
def obj(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
process=observed()
with (root/'actual_children.jsonl').open('a') as stream:
    stream.write(json.dumps(dict(phase=phase,start=start,arm=arm,process=process))+'\n');stream.flush();os.fsync(stream.fileno())
if mode=='sleep':time.sleep(60)
if mode=='missing':raise SystemExit(0)
release=json.loads((root/'release.json').read_text())
identity=dict(train_seed=start,arm=arm,execution_sources_sha256=obj(release['execution_sources']))
canary=root/f'canary_replay/start-{start}/{arm}';name='baseline' if phase=='baseline' else 'restored';slot=canary/name
control=root/f'first_two_control/start-{start}/{arm}/{phase}';control.mkdir(parents=True,exist_ok=False)
if phase=='baseline':
    spec=write(canary/'TECHNICAL_EXECUTION.json',dict(identity=identity,execution_sources=release['execution_sources']))
else:
    prior=root/f'first_two_control/start-{start}/{arm}/baseline/RESULT.json'
    assert sha(prior)==baseline_sha
    spec=ref(canary/'TECHNICAL_EXECUTION.json')
if mode=='spoof_pid':process['pid']+=10000000
claim=write(slot/'CLAIM.json',dict(schema='sr-satdose-canary-replay-claim-v1',status='UNKNOWN_UNTIL_COMPLETE_NO_RETRY',
    phase=name,identity=identity,process=process,spec=spec))
refs={key:write(slot/filename,{'CPU_FIXTURE_ONLY':True}) for key,filename in
    [('tensors','TEACHER.pt'),('state_identity','STATE_IDENTITY.json'),('accounting','ACCOUNTING.json'),
     ('relocation','RELOCATION.json'),('fork_load','FORK_LOAD.json'),('raw','RAW.jsonl')]}
complete=dict(schema='sr-satdose-canary-replay-result-v1',phase=name,
    status='STEP2_BASELINE_COMPLETE_NOT_RESUME_ADMISSION' if phase=='baseline' else 'STEP2_RELOAD_BIT_EXACT_NOT_AGGREGATE_ADMISSION',
    permission='NO_STEP3_OR_AGGREGATE_ADMISSION',request_count=128,measured_counts=dict(training_policy=128,training_reference=128),
    identity=identity,process=process,spec=spec,claim=claim,**refs)
if phase=='reload':complete['baseline']=ref(canary/'baseline/COMPLETE.json')
comp=write(slot/'COMPLETE.json',complete)
result=dict(schema='sr-satdose-first-two-result-v1',
    status='FIRST_TWO_BASELINE_COMPLETE_NOT_CANARY_ADMISSION' if phase=='baseline' else 'FIRST_TWO_RELOAD_COMPLETE_NOT_CANARY_ADMISSION',
    permission='NO_STEP3_OR_CANARY_ADMISSION',start_seed=start,arm=arm,identity=identity,
    release=dict(path='release.json',sha256=release_sha),process=process,process_exit_observed=False,
    train_responses=256 if phase=='baseline' else 0,formal_optimizer_steps=2 if phase=='baseline' else 0,
    scheduler_steps=2 if phase=='baseline' else 0,canary_spec=spec,**{name:comp})
if phase=='baseline':result['frozen_witnesses']={'before':refs['fork_load'],'after':refs['fork_load']}
if mode=='overbudget':result['formal_optimizer_steps']=3
write(control/'RESULT.json',result)
if mode=='mutate_release':(root/'release.json').write_text('{}')
if mode=='nonzero':raise SystemExit(7)
'''


class FirstTwoLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        (self.root/'cpu_observer.py').write_text(OBSERVER);self.script=self.root/'cpu_worker.py';self.script.write_text(WORKER)
        spec=importlib.util.spec_from_file_location('_actual_cpu_process_observer',self.root/'cpu_observer.py')
        self.observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.observer)
        self.source=dict(path=str(Path(f.__file__).resolve()),sha256=f._sha(Path(f.__file__).resolve()))
        self.release={'execution_sources':{'canary_controller':self.source},'CPU_FIXTURE_ONLY':True}
        self.release_path=self.root/'release.json';self.release_path.write_text(json.dumps(self.release));self.release_sha=f._sha(self.release_path)
        self.modes={};self.commands=[]
        def release(path,digest):
            f.require(path==self.release_path and f._sha(path)==digest==self.release_sha,'CPU trusted release changed')
            return self.release,self.source
        def command(path,digest,start,arm,phase,baseline_sha=None):
            self.commands.append((start,arm,phase,baseline_sha))
            result=[sys.executable,'-B',str(self.script),str(self.root),str(start),arm,phase,digest,
                    self.modes.get((start,arm,phase),'success')]
            if baseline_sha:result.append(baseline_sha)
            return result
        self.patches=[patch.object(f,'ROOT',self.root),patch.object(f,'PACKAGE',self.root),
            patch.object(f,'_process',side_effect=self.observer.observed),patch.object(f,'_release',side_effect=release),
            patch.object(f,'_command',side_effect=command),patch.object(f.worker,'_allocation',return_value={'CPU_FIXTURE':True})]
        self.mocks=[p.start() for p in self.patches]
        for p in self.patches:self.addCleanup(p.stop)

    def execute(self):return f.execute(self.release_path,self.release_sha)
    def children(self):
        path=self.root/'actual_children.jsonl'
        return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]

    def test_real_eight_children_wait_order_and_no_canary_pass(self):
        result=self.execute()
        self.assertEqual(result['permission'],'NO_STEP3_OR_CANARY_ADMISSION');self.assertEqual(result['formal_optimizer_steps'],8)
        self.assertEqual(len(self.children()),8);self.assertEqual(len({v['process']['pid'] for v in self.children()}),8)
        self.assertEqual([(s,a,p) for s,a,p,_ in self.commands],[(s,a,p) for s,a in f.PATHS for p in ('baseline','reload')])
        for key,value in result['paths'].items():
            proof=f._read(value['process_exit_proof']);wait=f._read(proof['wait_event']);launch=f._read(proof['restored_launch_event'])
            rewait=f._read(proof['restored_wait_event'])
            self.assertEqual(wait['returncode'],0);self.assertEqual(rewait['returncode'],0)
            self.assertEqual(wait['waited_pid'],proof['baseline_process']['pid'])
            self.assertEqual(rewait['waited_pid'],proof['restored_process']['pid'])
            self.assertLess(wait['monotonic_ns'],launch['monotonic_ns']);self.assertLess(launch['monotonic_ns'],rewait['monotonic_ns'])
            for proc in (proof['baseline_process'],proof['restored_process']):
                with self.assertRaises(ProcessLookupError):os.kill(proc['pid'],0)
        self.assertTrue((self.root/'first_two_launcher/RESULT.json').exists())

    def test_valid_result_with_nonzero_exit_never_launches_reload(self):
        self.modes[11031,'T0','baseline']='nonzero'
        with self.assertRaisesRegex(PermissionError,'exited nonzero'):self.execute()
        self.assertEqual(len(self.children()),1);self.assertEqual(len(self.commands),1)
        self.assertTrue((self.root/'first_two_control/start-11031/T0/baseline/RESULT.json').exists())
        self.assertTrue((self.root/'first_two_launcher/HOLD.json').exists())
        with self.assertRaisesRegex(PermissionError,'Existing launcher attempt'):self.execute()
        self.assertEqual(len(self.children()),1)

    def test_zero_exit_missing_result_is_unknown_not_success(self):
        self.modes[11031,'T0','baseline']='missing'
        with self.assertRaises((FileNotFoundError,PermissionError)):self.execute()
        self.assertEqual(len(self.children()),1)
        self.assertFalse((self.root/'first_two_launcher/start-11031/T0/BASELINE_WAIT_EVENT.json').exists())
        self.assertTrue((self.root/'first_two_launcher/start-11031/T0/BASELINE_WAIT_OBSERVATION.json').exists())

    def test_spoofed_model_pid_rejected_after_actual_wait(self):
        self.modes[11031,'T0','baseline']='spoof_pid'
        with self.assertRaisesRegex(PermissionError,'actual waited child'):self.execute()
        self.assertEqual(len(self.children()),1)

    def test_reload_failure_stops_remaining_paths_and_no_exit_proof(self):
        self.modes[11031,'T0','reload']='nonzero'
        with self.assertRaisesRegex(PermissionError,'exited nonzero'):self.execute()
        self.assertEqual(len(self.children()),2)
        self.assertFalse((self.root/'first_two_launcher/start-11031/T0/EXIT_PROOF.json').exists())
        self.assertFalse((self.root/'first_two_launcher/RESULT.json').exists())

    def test_any_step_over_two_is_rejected(self):
        self.modes[11031,'T0','baseline']='overbudget'
        with self.assertRaisesRegex(PermissionError,'fixed first-two budget'):self.execute()
        self.assertEqual(len(self.children()),1)

    def test_release_change_blocks_next_worker(self):
        self.modes[11031,'T0','baseline']='mutate_release'
        with self.assertRaisesRegex(PermissionError,'trusted release changed'):self.execute()
        self.assertEqual(len(self.children()),1)

    def test_existing_unknown_launcher_claim_is_not_restarted(self):
        path=self.root/'first_two_launcher';path.mkdir();(path/'CLAIM.json').write_text('{}')
        with self.assertRaisesRegex(PermissionError,'Existing launcher attempt'):self.execute()
        self.assertEqual(self.children(),[])

    def test_real_lock_excludes_second_launcher(self):
        with f._lease():
            with self.assertRaises(BlockingIOError):self.execute()
        self.assertEqual(self.children(),[])

    def test_failed_popen_intent_is_retained_and_never_retried(self):
        actual=subprocess.Popen
        def fail_worker(command,*args,**kwargs):
            if str(self.script) in command:raise OSError('synthetic exec failure')
            return actual(command,*args,**kwargs)  # Preserve real ps/sysctl identity observations.
        with patch.object(f.subprocess,'Popen',side_effect=fail_worker):
            with self.assertRaisesRegex(OSError,'exec failure'):self.execute()
        self.assertTrue((self.root/'first_two_launcher/start-11031/T0/BASELINE_LAUNCH_INTENT.json').exists())
        with self.assertRaisesRegex(PermissionError,'Existing launcher attempt'):self.execute()

    def test_controller_interrupt_stops_only_own_child_and_retains_unknown(self):
        self.modes[11031,'T0','baseline']='sleep'
        other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        try:
            with patch.object(f,'_wait',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):self.execute()
            self.assertIsNone(other.poll())
            path=self.root/'first_two_launcher/start-11031/T0/BASELINE_ABORT_WAIT.json'
            observed=json.loads(path.read_text());self.assertNotEqual(observed['returncode'],0)
            with self.assertRaises(ProcessLookupError):os.kill(observed['waited_pid'],0)
            self.assertTrue((self.root/'first_two_launcher/HOLD.json').exists())
            self.assertFalse((self.root/'first_two_launcher/RESULT.json').exists())
        finally:other.terminate();other.wait()

    def test_production_command_exact_phase_and_baseline_sha(self):
        self.patches[4].stop()
        command=f._command(self.release_path,self.release_sha,11031,'T0','baseline')
        self.assertEqual(command[1:4],['-B','-m','execution.first_two_control'])
        self.assertNotIn('--baseline-result-sha256',command)
        command=f._command(self.release_path,self.release_sha,11031,'T0','reload','a'*64)
        self.assertEqual(command[-2:],['--baseline-result-sha256','a'*64])
        for args in [(11031,'T0','train',None),(11031,'T0','reload',None),(999,'T0','baseline',None)]:
            with self.assertRaises(PermissionError):f._command(self.release_path,self.release_sha,*args)


if __name__=='__main__':unittest.main()
