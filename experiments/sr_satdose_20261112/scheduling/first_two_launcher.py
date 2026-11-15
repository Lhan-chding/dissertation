"""Observe actual first-two worker exits before fresh read-only reloads.

Run INSIDE one existing two-GPU Slurm task. Each model worker is a direct Python
child, never an srun wrapper. This controller records actual Popen PID/kernel
incarnation and wait return; it grants no canary PASS and never requests step3.
Every launch has an exclusive durable intent. Unknown/nonzero attempts stop the
four-path sequence and are never retried, even if a RESULT file exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from execution import first_two_control as worker
from execution import canary_audit, canary_replay, training_transaction as tx
from runtime.backend import object_sha

ROOT=worker.ROOT
PACKAGE=Path(__file__).resolve().parents[1]
PYTHON=Path('/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python')
PATHS=tuple((start,arm) for start in (11031,11032) for arm in ('T0','T1'))
require=tx.require


def _sha(path):
    path=Path(path)
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Canonical launcher evidence required')
    return tx.file_sha(path)


def _ref(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=_sha(path))
def _read(reference):return tx.read_json(tx.bound(ROOT,reference))
def _write(path,value):tx.write_once(path,value);return _ref(path)


def _process(pid=None):
    """Real Linux process incarnation. No PID-only or synthetic fallback."""
    pid=os.getpid() if pid is None else pid
    require(type(pid) is int and pid>0,'Invalid observed PID')
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    value=dict(host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               pid=pid,start_ticks=int(fields[19]))
    return canary_audit._process(value)


def _release(path,digest):
    require(ROOT==worker.ROOT and PACKAGE==worker.PACKAGE,'Foreign controller roots')
    release=worker.verify_release(path,digest)
    source=dict(path=str(Path(__file__).resolve()),sha256=_sha(Path(__file__).resolve()))
    require(release['execution_sources']['canary_controller']==source and
            release['sources'].get('scheduling/first_two_launcher.py')==source['sha256'],'Unpinned actual canary controller')
    require(Path(sys.executable).resolve(strict=True)==PYTHON.resolve(strict=True),'Unexpected worker interpreter')
    return release,source


@contextmanager
def _lease():
    require(ROOT.resolve(strict=True)==ROOT,'Noncanonical evidence root')
    fd=os.open(ROOT/'FIRST_TWO_LAUNCHER.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actual=(ROOT/'FIRST_TWO_LAUNCHER.lock').stat();opened=os.fstat(fd)
        require((actual.st_dev,actual.st_ino)==(opened.st_dev,opened.st_ino),'Launcher lock replaced')
        yield
    finally:os.close(fd)


def _command(release_path,release_sha,start,arm,phase,baseline_sha=None):
    require((start,arm) in PATHS and phase in ('baseline','reload'),'Unregistered first-two worker command')
    require((phase=='baseline' and baseline_sha is None) or
            (phase=='reload' and re.fullmatch('[a-f0-9]{64}',baseline_sha or '') is not None),'Wrong baseline/reload dependency')
    command=[str(PYTHON),'-B','-m','execution.first_two_control','--release-receipt',str(release_path),
        '--release-sha256',release_sha,'--start-seed',str(start),'--arm',arm,'--phase',phase,'--execute']
    if baseline_sha is not None:command+=['--baseline-result-sha256',baseline_sha]
    return command


def _environment():
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME','PYTORCH_CUDA_ALLOC_CONF'):env.pop(name,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    return env


def _wait(child):return child.wait()


def _child(directory,phase,command,controller,source):
    """One actual child and wait; raw observations survive any failed evidence audit."""
    prefix=phase.upper();tick=time.monotonic_ns()
    launch=_write(directory/(prefix+'_LAUNCH_INTENT.json'),dict(schema='sr-satdose-controller-launch-intent-v1',
        phase=phase,command=command,controller_process=controller,controller_source=source,monotonic_ns=tick,
        status='LAUNCH_REQUESTED_UNKNOWN_UNTIL_WAIT'))
    child=None
    try:
        # Keep full stdout/stderr on disk; never wait on a pipe that a worker could fill.
        with (directory/(prefix+'.out')).open('x') as out,(directory/(prefix+'.err')).open('x') as err:
            child=subprocess.Popen(command,cwd=PACKAGE,env=_environment(),stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            observed=_process(child.pid)
            _write(directory/(prefix+'_SPAWNED.json'),dict(schema='sr-satdose-controller-spawn-observation-v1',
                launch=launch,controller_process=controller,controller_source=source,worker_process=observed,
                direct_child_pid=child.pid,monotonic_ns=time.monotonic_ns()))
            code=_wait(child);wait_tick=time.monotonic_ns()
            wait=_write(directory/(prefix+'_WAIT_OBSERVATION.json'),dict(schema='sr-satdose-controller-wait-observation-v1',
                launch=launch,controller_process=controller,controller_source=source,worker_process=observed,
                kind='DIRECT_CHILD_WAIT',waited_pid=child.pid,returncode=code,monotonic_ns=wait_tick))
            require(type(code) is int and code==0,'Worker exited nonzero; no retry or next phase')
            require(_process()==controller,'Controller process identity changed')
            return dict(process=observed,launch=launch,launch_tick=tick,wait=wait,wait_tick=wait_tick,returncode=code,pid=child.pid)
    except BaseException:
        if child is not None and child.poll() is None:
            # Only this controller's own direct child, never a foreign Slurm job.
            child.terminate()
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();code=child.wait()
            _write(directory/(prefix+'_ABORT_WAIT.json'),dict(status='OWN_CHILD_STOPPED_AFTER_CONTROLLER_FAILURE',
                waited_pid=child.pid,returncode=code,monotonic_ns=time.monotonic_ns(),launch=launch))
        raise


def _result(start,arm,phase,observation,release_ref,release):
    path=ROOT/f'first_two_control/start-{start}/{arm}/{phase}/RESULT.json'
    reference=_ref(path);value=tx.read_json(path)
    require(value.get('schema')=='sr-satdose-first-two-result-v1' and value.get('status')==
            (worker.BASELINE_STATUS if phase=='baseline' else worker.RELOAD_STATUS) and
            value.get('permission')=='NO_STEP3_OR_CANARY_ADMISSION' and value.get('start_seed')==start and
            value.get('arm')==arm and value.get('release')==release_ref and
            value.get('process')==observation['process'] and value.get('process_exit_observed') is False,
            'Worker result does not belong to actual waited child')
    require(value['identity']['train_seed']==start and value['identity']['arm']==arm and
            value['identity']['execution_sources_sha256']==object_sha(release['execution_sources']),'Worker path/source identity differs')
    counts=(256,2,2) if phase=='baseline' else (0,0,0)
    require(all(type(value.get(k)) is int and value[k]==n for k,n in
        zip(('train_responses','formal_optimizer_steps','scheduler_steps'),counts)),'Worker exceeded fixed first-two budget')
    base=ROOT/f'canary_replay/start-{start}/{arm}'
    require(tx.bound(ROOT,value['canary_spec'])==base/'TECHNICAL_EXECUTION.json','Canary spec relocated')
    spec=_read(value['canary_spec'])
    require(spec['identity']==value['identity'] and spec['execution_sources']==release['execution_sources'],
            'Canary source/path identity differs')
    name='baseline' if phase=='baseline' else 'restored'
    complete=canary_replay._complete(ROOT,base/name,name)
    require(tx.bound(ROOT,value[name])==base/name/'COMPLETE.json' and complete['identity']==value['identity'] and
            complete['process']==observation['process'] and complete['spec']==value['canary_spec'],
            'Canary capture not produced by actual waited child')
    require(not any(p.name in ('HOLD.json','ERROR.json') or p.name.endswith('.ERROR.json') or p.is_symlink()
                    for p in path.parent.rglob('*')),'Worker retained failed/unknown evidence')
    return dict(result=value,reference=reference,canary=complete,canary_reference=value[name])


def _event(directory,name,kind,ctx,observation,source,controller,*,launch=False):
    value=dict(schema='sr-satdose-controller-process-event-v1',event=kind,controller_source=source,
        controller_process=controller,identity=ctx['result']['identity'],spec=ctx['result']['canary_spec'],
        worker_process=ctx['canary']['process'],worker_claim=ctx['canary']['claim'],
        monotonic_ns=observation['launch_tick'] if launch else observation['wait_tick'],
        observation=observation['launch'] if launch else observation['wait'])
    if not launch:value.update(kind='DIRECT_CHILD_WAIT',waited_pid=observation['pid'],returncode=observation['returncode'])
    return _write(directory/name,value)


def _path(directory,release_path,release_sha,release,source,controller,start,arm):
    release_ref=_ref(release_path)
    _release(release_path,release_sha)
    baseline_observed=_child(directory,'baseline',_command(release_path,release_sha,start,arm,'baseline'),controller,source)
    baseline=_result(start,arm,'baseline',baseline_observed,release_ref,release)
    wait=_event(directory,'BASELINE_WAIT_EVENT.json','BASELINE_WAIT_RETURNED',baseline,baseline_observed,source,controller)
    # Baseline exit zero AND authenticated complete evidence must precede launch.
    _release(release_path,release_sha)
    restored_observed=_child(directory,'reload',_command(release_path,release_sha,start,arm,'reload',
        baseline['reference']['sha256']),controller,source)
    require(baseline_observed['wait_tick']<restored_observed['launch_tick'] and
            baseline_observed['process']!=restored_observed['process'],'Reload preceded baseline exit/reused process')
    restored=_result(start,arm,'reload',restored_observed,release_ref,release)
    require(restored['result']['identity']==baseline['result']['identity'] and
            restored['result']['canary_spec']==baseline['result']['canary_spec'] and
            restored['canary']['baseline']==baseline['canary_reference'],'Reload binds another baseline')
    launch=_event(directory,'RESTORED_LAUNCH_EVENT.json','RESTORED_LAUNCH_REQUESTED',restored,restored_observed,source,controller,launch=True)
    reload_wait=_event(directory,'RESTORED_WAIT_EVENT.json','RESTORED_WAIT_RETURNED',restored,restored_observed,source,controller)
    proof=dict(schema=canary_audit.EXIT_SCHEMA,identity=baseline['result']['identity'],spec=baseline['result']['canary_spec'],
        controller_source=source,controller_process=controller,baseline_complete=baseline['canary_reference'],
        restored_claim=restored['canary']['claim'],baseline_process=baseline_observed['process'],restored_process=restored_observed['process'],
        wait_event=wait,restored_launch_event=launch,restored_wait_event=reload_wait)
    proof_ref=_write(directory/'EXIT_PROOF.json',proof)
    # Exercise the same independent source-bound proof validator before publishing.
    evidence=canary_audit._Evidence(ROOT)
    context=dict(store=SimpleIdentity(baseline['result']['identity']),spec_ref=baseline['result']['canary_spec'],
        phases={k:dict(result=v['canary'],reference=v['canary_reference']) for k,v in [('baseline',baseline),('restored',restored)]})
    canary_audit._exit_proof(evidence,proof_ref,context,release['execution_sources']);evidence.stable()
    result=dict(status='FIRST_TWO_PATH_EXIT_OBSERVED_NOT_CANARY_ADMISSION',permission='NO_STEP3_OR_CANARY_ADMISSION',
        start_seed=start,arm=arm,baseline_result=baseline['reference'],reload_result=restored['reference'],
        canary_spec=baseline['result']['canary_spec'],frozen_witnesses=baseline['result']['frozen_witnesses'],
        process_exit_proof=proof_ref,train_responses=256,formal_optimizer_steps=2)
    _write(directory/'RESULT.json',result);return result


class SimpleIdentity:
    """Only the immutable identity field consumed by the exit-proof auditor."""
    def __init__(self,value):self.identity=copy.deepcopy(value)


def execute(release_path,release_sha):
    release_path=Path(release_path)
    release,source=_release(release_path,release_sha);worker._allocation()
    controller=_process();directory=ROOT/'first_two_launcher'
    with _lease():
        require(not os.path.lexists(directory),'Existing launcher attempt: no automatic retry')
        directory.mkdir();tx.fsync_directory(ROOT)
        _write(directory/'CLAIM.json',dict(schema='sr-satdose-first-two-launcher-claim-v1',status='UNKNOWN_UNTIL_COMPLETE_NO_RETRY',
            release=_ref(release_path),controller_source=source,controller_process=controller,paths=[f'{s}/{a}' for s,a in PATHS],
            permission='FOUR_PATHS_MAX_STEP_2_NO_STEP3'))
        try:
            results={}
            for start,arm in PATHS:
                slot=directory/f'start-{start}'/arm;slot.mkdir(parents=True);tx.fsync_directory(slot.parent)
                results[f'{start}/{arm}']=_path(slot,release_path,release_sha,release,source,controller,start,arm)
            _release(release_path,release_sha)
            result=dict(schema='sr-satdose-first-two-launcher-result-v1',status='FOUR_FIRST_TWO_PATHS_EXIT_OBSERVED_NOT_CANARY_ADMISSION',
                permission='NO_STEP3_OR_CANARY_ADMISSION',release=_ref(release_path),paths=results,
                train_responses=1024,formal_optimizer_steps=8,model_worker_processes=8,
                limitation='Actual direct-child exits observed; independent full canary tensor/frozen-weight audit still required')
            _write(directory/'RESULT.json',result);return result
        except BaseException as exc:
            _write(directory/'HOLD.json',dict(status='HOLD_FIRST_TWO_LAUNCHER_NO_RETRY',permission='NO_TRAINING',
                error_type=type(exc).__name__,error=str(exc),recovery='AUDIT_ALL_EXISTING_ATTEMPTS_NO_AUTOMATIC_RETRY'))
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-receipt',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    a=p.parse_args(argv)
    try:result=execute(a.release_receipt,a.release_sha256)
    except Exception as exc:
        result=dict(status='HOLD_FIRST_TWO_LAUNCHER',permission='NO_TRAINING',error_type=type(exc).__name__,error=str(exc))
    print(tx.canonical(result).decode());return 2 if result['status'].startswith('HOLD') else 0


if __name__=='__main__':raise SystemExit(main())
