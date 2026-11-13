"""CPU-only real native Adam/state codec checks; no model forward or GPU claims."""
import copy
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import training_state as s

NATIVE=Path(os.environ.get('SATDOSE_NATIVE_SRC',str(ROOT.parent/'dissertation for NTU/sr-f6/src')))


class ScalarLoRA(torch.nn.Module):
    """496 scalar parameters satisfy the REAL frozen loader; forward is forbidden."""
    def __init__(self):
        super().__init__()
        self.lora_fixture=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([.2+i/10000])) for i in range(496)])

    def forward(self,*args,**kwargs):
        raise AssertionError('No model forward is allowed in codec tests')


@unittest.skipUnless((NATIVE/'sr_f3_ce/training.py').exists(),'Pinned native sources required')
class TrainingStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespaces={p.relative_to(NATIVE).parts[0].removesuffix('.py') for p in NATIVE.rglob('*.py')}
        before={n:m for n,m in sys.modules.items() if n.split('.')[0] in namespaces}
        def restore():
            for n in list(sys.modules):
                if n.split('.')[0] in namespaces:del sys.modules[n]
            sys.modules.update(before)
        cls.addClassCleanup(restore)
        for n in before:del sys.modules[n]
        pins={rel:s._file_sha(NATIVE/rel) for rel in ('mm_core/training.py','sr_f3_ce/training.py')}
        cls.ops=s.fork_loader.load_native_ops(NATIVE,pins)

    def setUp(self):
        cuda=patch.object(torch.cuda,'is_available',return_value=False);cuda.start();self.addCleanup(cuda.stop)
        self.rng=self.ops.capture_rng();self.addCleanup(self.ops.restore_rng,self.rng)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.runtime,self.optimizer,self.scheduler=self.make_runtime()
        self.reference={n:p.detach().clone()-.1 for n,p in self.runtime.model.named_parameters()}
        self.base=dict(experiment_id=s.EXPERIMENT,parent='b0',train_seed=11031,arm='T0',
                       run_id=s.EXPERIMENT+'/b0-11031-T0',config_sha256='c'*64,source_sha256='d'*64)
        self.source=dict(original_state_path='/synthetic/original/R24/state.pt',state_sha256='a'*64,
            policy_sha256=self.ops.state_hash(self.ops.trainable_state(self.runtime.model)),
            reference_sha256=self.ops.state_hash(self.reference))
        self.identity=s.bind_identity(self.base,self.runtime.model,self.optimizer,source_binding=self.source)

    def make_runtime(self):
        runtime=SimpleNamespace(model=ScalarLoRA(),identity=dict(experiment_id=s.EXPERIMENT,train_seed=11031,arm='T0',
            run_id=s.EXPERIMENT+'/b0-11031-T0',reference_parent='b0',step=0))
        opt,scheduler=self.ops.fresh_optimizer(runtime.model)
        return runtime,opt,scheduler

    def steps(self,count=2):
        for step in range(1,count+1):
            for i,param in enumerate(self.runtime.model.parameters()):
                param.grad=torch.full_like(param,(i+1)/1000 * (1 if step==1 else -.3))
            self.optimizer.step();self.scheduler.step();self.optimizer.zero_grad(set_to_none=True)
            random.random();np.random.random();torch.rand(2)

    def write(self,step=2,**kwargs):
        directory=self.root/f'step-{step:03d}';directory.mkdir()
        result=s.write_checkpoint_files(directory,self.runtime,self.optimizer,self.scheduler,self.reference,self.ops,
            identity=self.identity,step=step,sampler_state=dict(committed_step=step,next_step=step+1,
            generated_responses=step*128,original_manifest_sha256='e'*64),**kwargs)
        return directory,result

    def fingerprint(self,runtime=None,optimizer=None,scheduler=None):
        runtime=runtime or self.runtime;optimizer=optimizer or self.optimizer;scheduler=scheduler or self.scheduler
        return self.ops.state_hash(dict(policy=self.ops.trainable_state(runtime.model),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng=self.ops.capture_rng(),identity=runtime.identity))

    def test_real_two_step_full_restore_via_frozen_loader(self):
        self.steps();directory,result=self.write()
        expected=dict(policy=self.ops.trainable_state(self.runtime.model),optimizer=copy.deepcopy(self.optimizer.state_dict()),
                      scheduler=copy.deepcopy(self.scheduler.state_dict()),rng=self.ops.capture_rng())
        other,opt,scheduler=self.make_runtime()
        random.random();np.random.random();torch.rand(5)
        restored=s.restore_checkpoint(other,opt,scheduler,self.reference,self.ops,checkpoint_dir=directory,
            expected_commit_sha256=result['commit_sha256'],expected_identity=self.identity)
        self.assertEqual(restored,result['commit'])
        self.assertEqual(self.ops.state_hash(self.ops.trainable_state(other.model)),self.ops.state_hash(expected['policy']))
        self.assertEqual(self.ops.state_hash(opt.state_dict()),self.ops.state_hash(expected['optimizer']))
        self.assertEqual(self.ops.state_hash(scheduler.state_dict()),self.ops.state_hash(expected['scheduler']))
        self.assertEqual(self.ops.state_hash(self.ops.capture_rng()),self.ops.state_hash(expected['rng']))
        self.assertEqual(other.identity['step'],2)
        self.assertEqual(restored['sampler_state']['generated_responses'],256)
        self.assertEqual(restored['sampler_state']['next_step'],3)
        self.assertEqual(len(opt.state),496)
        self.assertTrue(all(float(v['step'])==2 for v in opt.state.values()))
        self.assertTrue(all(set(v)=={'step','exp_avg','exp_avg_sq'} for v in opt.state.values()))

    def test_weights_only_numpy_uint32_native_rng_roundtrip(self):
        native=self.ops.capture_rng();packed=s.pack_rng(native)
        self.assertIsInstance(packed['numpy'][1],list)
        stream=io.BytesIO();torch.save(packed,stream);stream.seek(0)
        loaded=torch.load(stream,weights_only=True,map_location='cpu')
        restored=s.unpack_rng(loaded)
        self.assertEqual(restored['numpy'][1].dtype,np.dtype('uint32'))
        self.assertIsInstance(restored['python'],tuple)
        self.assertEqual(self.ops.state_hash(restored),self.ops.state_hash(native))
        self.assertEqual(self.ops.state_hash(self.ops.capture_rng()),self.ops.state_hash(native))
        self.steps();directory,_=self.write()
        state=torch.load(directory/'state.pt',weights_only=True,map_location='cpu')
        self.assertEqual(state['schema'],s.SCHEMA)
        self.assertIsInstance(state['rng']['numpy'][1],list)
        self.assertEqual(len(state['policy']),496)

    def test_packed_uint32_overflow_and_negative_rejected(self):
        for bad in (-1,2**32,True,1.5):
            packed=s.pack_rng(self.ops.capture_rng());packed['numpy'][1][0]=bad
            with self.assertRaisesRegex(PermissionError,'uint32 words'):s.unpack_rng(packed)

    def test_equal_shape_parameter_order_swap_rejected_without_mutation(self):
        self.steps();directory,result=self.write()
        other,opt,scheduler=self.make_runtime()
        opt.param_groups[0]['params'][0],opt.param_groups[0]['params'][1]=opt.param_groups[0]['params'][1],opt.param_groups[0]['params'][0]
        before=self.fingerprint(other,opt,scheduler)
        with self.assertRaisesRegex(PermissionError,'Actual optimizer parameter order'):
            s.restore_checkpoint(other,opt,scheduler,self.reference,self.ops,checkpoint_dir=directory,
                expected_commit_sha256=result['commit_sha256'],expected_identity=self.identity)
        self.assertEqual(before,self.fingerprint(other,opt,scheduler))

    def test_cross_arm_or_source_identity_rejected_without_mutation(self):
        self.steps();directory,result=self.write()
        for field in ('arm','source'):
            expected=copy.deepcopy(self.identity)
            if field=='arm':expected['arm']='T1'
            else:expected['source_binding']['state_sha256']='b'*64
            before=self.fingerprint()
            with self.assertRaises(PermissionError):
                s.restore_checkpoint(self.runtime,self.optimizer,self.scheduler,self.reference,self.ops,
                    checkpoint_dir=directory,expected_commit_sha256=result['commit_sha256'],expected_identity=expected)
            self.assertEqual(before,self.fingerprint())

    def test_wrong_trusted_sha_rejected_without_mutation(self):
        self.steps();directory,_=self.write();before=self.fingerprint()
        with self.assertRaisesRegex(PermissionError,'COMMIT SHA'):
            s.restore_checkpoint(self.runtime,self.optimizer,self.scheduler,self.reference,self.ops,
                checkpoint_dir=directory,expected_commit_sha256='0'*64,expected_identity=self.identity)
        self.assertEqual(before,self.fingerprint())

    def test_state_file_tamper_rejected_without_mutation(self):
        self.steps();directory,result=self.write();before=self.fingerprint()
        with (directory/'state.pt').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(PermissionError,'file bytes'):
            s.restore_checkpoint(self.runtime,self.optimizer,self.scheduler,self.reference,self.ops,
                checkpoint_dir=directory,expected_commit_sha256=result['commit_sha256'],expected_identity=self.identity)
        self.assertEqual(before,self.fingerprint())

    def test_write_once_and_transaction_fields_cannot_overwrite_core(self):
        self.steps();directory,result=self.write(commit_fields={'transaction':{'previous_commit_sha256':'e'*64}})
        self.assertEqual(result['commit']['transaction']['previous_commit_sha256'],'e'*64)
        with self.assertRaisesRegex(PermissionError,'already exist'):
            s.write_checkpoint_files(directory,self.runtime,self.optimizer,self.scheduler,self.reference,self.ops,
                identity=self.identity,step=2,sampler_state={})
        for key in ('identity','schema','step','state_sha256','rng_sha256','sampler_state','commit_sha256'):
            stage=self.root/('bad-'+key);stage.mkdir()
            with self.assertRaisesRegex(PermissionError,'overwrite core'):
                s.write_checkpoint_files(stage,self.runtime,self.optimizer,self.scheduler,self.reference,self.ops,
                    identity=self.identity,step=2,sampler_state={'committed_step':2,'next_step':3,'generated_responses':256},
                    commit_fields={key:'override'})
            self.assertFalse((stage/'state.pt').exists())

    def test_staging_validation_does_not_publish_or_require_step_basename(self):
        self.steps();directory,result=self.write()
        staging=self.root/'.staging-step-002';directory.rename(staging)
        validated=s.validate_checkpoint(staging,result['commit_sha256'],self.identity,self.ops)
        self.assertEqual(validated,result['commit'])
        self.assertFalse(directory.exists())
        before=self.fingerprint()
        with self.assertRaisesRegex(PermissionError,'Explicit step mismatch'):
            s.restore_checkpoint(self.runtime,self.optimizer,self.scheduler,self.reference,self.ops,
                checkpoint_dir=staging,expected_commit_sha256=result['commit_sha256'],expected_identity=self.identity)
        self.assertEqual(before,self.fingerprint())

    def test_fresh_step_zero_r24_differs_from_reference(self):
        directory,result=self.write(step=0)
        self.assertNotEqual(result['commit']['policy_sha256'],result['commit']['reference_sha256'])
        self.assertFalse(torch.load(directory/'state.pt',weights_only=True)['optimizer']['state'])
        self.assertEqual(result['commit']['sampler_state']['generated_responses'],0)

    def test_wrong_scheduler_sampler_or_adam_counter_fails_before_write(self):
        self.steps()
        stage=self.root/'bad-cursor';stage.mkdir()
        for error in ('scheduler','sampler','adam'):
            old_epoch=self.scheduler.last_epoch
            first=next(iter(self.optimizer.state.values()));old_step=first['step'].clone()
            if error=='scheduler':self.scheduler.last_epoch=1
            if error=='adam':first['step'].fill_(1)
            sampler=dict(committed_step=2,next_step=3,generated_responses=255 if error=='sampler' else 256)
            with self.assertRaises(PermissionError):
                s.write_checkpoint_files(stage,self.runtime,self.optimizer,self.scheduler,self.reference,self.ops,
                    identity=self.identity,step=2,sampler_state=sampler)
            self.assertFalse((stage/'state.pt').exists())
            self.scheduler.last_epoch=old_epoch;first['step'].copy_(old_step)


if __name__=='__main__':unittest.main()
