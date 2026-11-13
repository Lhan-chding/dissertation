"""Explicit synthetic CPU trials; no production/GPU receipts or model calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from execution import shadow_replay as s
from runtime.fork_loader import load_native_ops
NATIVE=Path(os.environ.get('SATDOSE_NATIVE_SRC',str(HERE.parents[1].parent/'dissertation for NTU/sr-f6/src')))
MODE='SYNTHETIC_CPU_TEST_ONLY'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f)


def ref(root,path):return dict(path=str(path.relative_to(root)),sha256=s.file_sha(path))


@unittest.skipUnless((NATIVE/'mm_core/training.py').exists(),'Full native sources required')
class ShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespaces={p.relative_to(NATIVE).parts[0].removesuffix('.py') for p in NATIVE.rglob('*.py')}
        old={n:m for n,m in sys.modules.items() if n.split('.')[0] in namespaces}
        def restore():
            for n in list(sys.modules):
                if n.split('.')[0] in namespaces:del sys.modules[n]
            sys.modules.update(old)
        cls.addClassCleanup(restore)
        for n in old:del sys.modules[n]
        cls.pins={n:s.file_sha(NATIVE/n) for n in ('mm_core/training.py','sr_f3_ce/training.py')}
        cls.ops=load_native_ops(NATIVE,cls.pins)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.policy={'lora_A.weight':torch.tensor([.2,-.3]),'lora_B.weight':torch.tensor([.4])}
        self.grad={'lora_A.weight':torch.tensor([.1,-.2]),'lora_B.weight':torch.tensor([.3])}
        policy_sha=self.ops.state_hash(self.policy);reference_sha='a'*64
        self.config=dict(experiment_id=s.EXPERIMENT,start_bindings={})
        for start in (11031,11032):
            state=self.root/f'original-{start}.pt';torch.save({'policy':self.policy,'reference_sha256':reference_sha},state)
            self.config['start_bindings'][str(start)]=dict(policy_sha256=policy_sha,reference_sha256=reference_sha,
                original_state_path=str(state),state_sha256=s.file_sha(state))
        config=self.root/'config.json';dump(config,self.config)
        lock=self.root/'source_lock.json';dump(lock,dict(native_src=str(NATIVE),native_files=self.pins))
        freeze=self.root/'DESIGN_FREEZE.json';dump(freeze,dict(schema='sr-satdose-design-freeze-v1',experiment_id=s.EXPERIMENT,
            artifacts=dict(config=ref(self.root,config),source_lock=ref(self.root,lock))))
        self.binding=dict(schema='sr-satdose-technical-execution-v1',mode=MODE,experiment_id=s.EXPERIMENT,
            experiment_root=str(self.root),shadow_claims_root='shadow_claims',shadow_slots=s.expected_slots(),
            max_shadow_optimizer_steps=4,steps_per_slot=1,design_freeze=ref(self.root,freeze),
            config=ref(self.root,config),source_lock=ref(self.root,lock),code={})
        for name,file in (('shadow_replay',s.__file__),('production',s.production.__file__)):
            self.binding['code'][name]=dict(path=str(Path(file).resolve()),sha256=s.file_sha(file))
        self.binding_path=self.root/s.BINDING_NAME;dump(self.binding_path,self.binding)
        self.binding_sha=s.file_sha(self.binding_path)

    def endpoint(self,start):
        expected=self.config['start_bindings'][str(start)]
        return dict(experiment_id=s.EXPERIMENT,start_seed=start,arm='START',step=0,
                    policy_sha256=expected['policy_sha256'],reference_sha256=expected['reference_sha256'])

    def inputs(self,start,trial,*,instance=None,grad=None):
        endpoint=self.endpoint(start);slot=self.binding['shadow_slots'][f'{start}/{trial}']
        instance=instance or f'LOADER_START_{start}_TRIAL_{trial}'
        expected=self.config['start_bindings'][str(start)]
        base=dict(mode=MODE,technical_binding_sha256=self.binding_sha,
            design_freeze_sha256=self.binding['design_freeze']['sha256'],endpoint=endpoint,loader_instance_id=instance)
        restore=self.root/slot['restore'];dump(restore,dict(base,schema='sr-satdose-restored-r24-v1',
            source_state_path=expected['original_state_path'],source_state_sha256=expected['state_sha256'],restored_from_original_state=True))
        gradients=grad or self.grad
        production=dict(status='NUMERICAL_CAPTURE_NOT_UPDATE_AUTHORIZATION',policy_sha256=endpoint['policy_sha256'],
            reference_sha256=endpoint['reference_sha256'],sequence_denominator=128,row_forward_callback_calls=128,
            backward_calls=128,optimizer_created=False,optimizer_steps=0,scheduler_steps=0,parameter_updates=0,
            preclip_gradients=gradients,postclip_gradients=copy.deepcopy(gradients),
            source_sha256={'native':s.production.PINS,'legacy':s.production.LEGACY_PINS})
        capture=self.root/slot['capture'];torch.save(dict(base,schema='sr-satdose-production-capture-v1',production=production,
            request_ids_sha256='b'*64,token_manifest_sha256='c'*64),capture)
        input_path=self.root/slot['input_receipt'];dump(input_path,dict(schema='sr-satdose-shadow-input-v1',mode=MODE,
            technical_binding_sha256=self.binding_sha,design_freeze_sha256=self.binding['design_freeze']['sha256'],
            start_seed=start,trial=trial,endpoint=endpoint,restore=ref(self.root,restore),capture=ref(self.root,capture)))
        return s.file_sha(input_path)

    def run_trial(self,start,trial,input_sha,**kw):
        return s.one_synthetic_trial_for_test(self.root,self.policy,self.grad,self.endpoint(start),
            start_seed=start,trial=trial,expected_binding_sha256=self.binding_sha,
            expected_input_receipt_sha256=input_sha,**kw)

    def test_four_slots_and_restore_replay_match_without_live_mutation(self):
        original=copy.deepcopy(self.policy)
        for start in (11031,11032):
            first=self.run_trial(start,0,self.inputs(start,0))
            second=self.run_trial(start,1,self.inputs(start,1))
            self.assertEqual(first['optimizer_steps'],1);self.assertTrue(second['restore_replay_compared'])
            self.assertEqual(second['mode'],MODE)
        self.assertEqual(len(list((self.root/'shadow_claims').glob('*/trial-*/CLAIM.json'))),4)
        self.assertTrue(all(torch.equal(original[n],self.policy[n]) for n in original))
        with self.assertRaises(PermissionError):self.run_trial(11031,2,'d'*64)

    def test_exact_adam_numeric_against_separate_fresh_optimizer(self):
        receipt=self.run_trial(11031,0,self.inputs(11031,0))
        result=torch.load(self.root/'shadow_claims/start-11031/trial-0/RESULT.pt',weights_only=True)
        params={n:torch.nn.Parameter(v.clone()) for n,v in self.policy.items()}
        opt=torch.optim.AdamW(list(params.values()),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
        for n,param in params.items():param.grad=self.grad[n].clone()
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True);opt.step()
        self.assertTrue(all(torch.equal(result['delta'][n],params[n].detach()-self.policy[n]) for n in params))
        self.assertEqual(receipt['optimizer_body_calls'],1)

    def test_existing_claim_refuses_automatic_retry(self):
        sha=self.inputs(11031,0);self.run_trial(11031,0,sha)
        with self.assertRaises(FileExistsError):self.run_trial(11031,0,sha)

    def test_failure_after_adam_is_unknown_and_no_retry(self):
        sha=self.inputs(11031,0)
        def fail(stage):
            if stage=='after_adam':raise RuntimeError('injected crash after Adam')
        with self.assertRaisesRegex(RuntimeError,'injected'):self.run_trial(11031,0,sha,failpoint=fail)
        root=self.root/'shadow_claims/start-11031/trial-0'
        self.assertTrue((root/'CLAIM.json').exists());self.assertFalse((root/'COMPLETE.json').exists())
        self.assertEqual(s.read_json(root/'HOLD.json')['optimizer_returned_calls'],1)
        with self.assertRaises(FileExistsError):self.run_trial(11031,0,sha)

    def test_failure_before_adam_still_consumes_unknown_claim(self):
        sha=self.inputs(11031,0)
        def fail(stage):
            if stage=='after_claim':raise RuntimeError('injected pre-Adam crash')
        with self.assertRaises(RuntimeError):self.run_trial(11031,0,sha,failpoint=fail)
        with self.assertRaises(FileExistsError):self.run_trial(11031,0,sha)

    def test_trial1_requires_completed_trial0_and_distinct_reload(self):
        sha=self.inputs(11031,1,instance='LOADER_START_11031_TRIAL_0')
        with self.assertRaises(FileNotFoundError):self.run_trial(11031,1,sha)
        self.run_trial(11031,0,self.inputs(11031,0))
        with self.assertRaisesRegex(PermissionError,'independent R24 reload'):self.run_trial(11031,1,sha)

    def test_changed_recaptured_gradient_rejected_before_claim(self):
        self.run_trial(11031,0,self.inputs(11031,0))
        changed={n:v*2 for n,v in self.grad.items()};sha=self.inputs(11031,1,grad=changed)
        old=self.grad;self.grad=changed
        try:
            with self.assertRaisesRegex(PermissionError,'not bit-exact'):self.run_trial(11031,1,sha)
        finally:self.grad=old
        self.assertFalse((self.root/'shadow_claims/start-11031/trial-1').exists())

    def test_binding_cannot_relocate_claims(self):
        self.binding['shadow_claims_root']='different_output'
        self.binding_path.write_text(json.dumps(self.binding));self.binding_sha=s.file_sha(self.binding_path)
        with self.assertRaisesRegex(PermissionError,'slots/budget'):self.run_trial(11031,0,'d'*64)

    def test_inputs_cannot_be_substituted(self):
        sha=self.inputs(11031,0)
        restore=self.root/self.binding['shadow_slots']['11031/0']['restore'];restore.write_text('{}')
        with self.assertRaisesRegex(PermissionError,'Changed restore'):self.run_trial(11031,0,sha)
        self.assertFalse((self.root/'shadow_claims').exists())

    def test_production_entry_cannot_use_test_binding_or_cpu(self):
        sha=self.inputs(11031,0)
        with patch.object(s,'PRODUCTION_ROOT',self.root):
            with self.assertRaisesRegex(PermissionError,'Wrong technical execution identity'):
                s.one_trial(self.policy,self.grad,self.endpoint(11031),start_seed=11031,trial=0,
                    expected_binding_sha256=self.binding_sha,expected_input_receipt_sha256=sha)
        with patch.object(s,'PRODUCTION_ROOT',self.root):
            with self.assertRaisesRegex(PermissionError,'cannot touch production'):
                self.run_trial(11031,0,sha)
        self.binding['mode']='production'
        self.binding_path.write_text(json.dumps(self.binding));self.binding_sha=s.file_sha(self.binding_path)
        with patch.object(s,'PRODUCTION_ROOT',self.root), patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(PermissionError,'needs CUDA 0'):
                s.one_trial(self.policy,self.grad,self.endpoint(11031),start_seed=11031,trial=0,
                    expected_binding_sha256=self.binding_sha,expected_input_receipt_sha256=sha)
        self.assertFalse((self.root/'shadow_claims').exists())


if __name__=='__main__':unittest.main()
