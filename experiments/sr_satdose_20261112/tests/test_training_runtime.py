"""CPU fake-native boundaries plus real codec/Adam; never GPU or model inference."""
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import torch

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import training_runtime as t

NATIVE=Path(os.environ.get('SATDOSE_NATIVE_SRC',str(ROOT.parent/'dissertation for NTU/sr-f6/src')))


class FakeRuntime(t.TrainingRuntimeMixin):
    def batch_reference_forward(self,records,reference):
        self.reserve('extra_forward_sequences',1,purpose='training_reference')
        return [torch.tensor(records[0]['sampler_logprobs'])-.1]

    def batch_sequence_forward(self,records,*,purpose,grad):
        self.reserve('extra_forward_sequences',1,purpose=purpose)
        values=sum(p.sum() for p in self.model.parameters())/496
        result=values.reshape(1).repeat(len(records[0]['tokens']))
        return [result if grad else result.detach()]


@unittest.skipUnless((NATIVE/'sr_f3_ce/training.py').exists(),'Pinned native sources required')
class TrainingRuntimeTests(unittest.TestCase):
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
        cls.ops=t.fork_loader.load_native_ops(NATIVE,{rel:t.file_sha(NATIVE/rel) for rel in
            ('mm_core/training.py','sr_f3_ce/training.py')})

    def setUp(self):
        cuda=patch.object(torch.cuda,'is_available',return_value=False);cuda.start();self.addCleanup(cuda.stop)
        rng=self.ops.capture_rng();self.addCleanup(self.ops.restore_rng,rng)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.output=self.root/'audit';self.output.mkdir()
        self.model=torch.nn.Module();self.model.lora_fixture=torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor([-.4])) for _ in range(496)])
        self.opt,self.scheduler=self.ops.fresh_optimizer(self.model)
        self.reference={n:p.detach().clone()-.1 for n,p in self.model.named_parameters()}
        self.source=dict(original_state_path='/synthetic/R24/state.pt',state_sha256='a'*64,
            policy_sha256=self.ops.state_hash(self.ops.trainable_state(self.model)),
            reference_sha256=self.ops.state_hash(self.reference))
        self.runtime=FakeRuntime();r=self.runtime
        r.model=self.model;r.torch=torch;r._satdose_active=None;r._satdose_closed=False;r._train_phase=None
        r._satdose_accounting=[];r._satdose_output=self.output;r._vision_hook=SimpleNamespace(remove=Mock())
        r.identity=dict(experiment_id=t.state.EXPERIMENT,train_seed=11031,arm='T0',run_id='synthetic-T0',
                        reference_parent='b0',step=0)
        self.requests={}
        self.registry=SimpleNamespace(root=self.root,freeze_sha256='f'*64,
            config=dict(start_bindings={'11031':self.source}),request=lambda rid:copy.deepcopy(self.requests[rid]))
        r._satdose_registry=self.registry
        fork=SimpleNamespace(reference=self.reference,optimizer=self.opt,scheduler=self.scheduler)
        self.storage=SimpleNamespace(GPUSavedTensors=lambda *a,**kw:SimpleNamespace(stats={'CPU_FAKE':True},context=nullcontext),
                                     validate_snapshot_stats=Mock())
        identity=t.state.bind_identity(dict(experiment_id=t.state.EXPERIMENT,parent='b0',train_seed=11031,arm='T0',
            run_id='synthetic-T0'),self.model,self.opt,source_binding=self.source)
        self.handle=t.TrainingHandle(r,fork,self.ops,self.storage,self.registry,11031,'T0',self.output,identity)
        self.store=object.__new__(t.transactions.TransactionStore)
        self.store.root=self.root;self.store.identity=identity;self.store.freeze_sha='f'*64
        self.store.path=self.root/'training_transactions/start-11031/T0';self.store.path.mkdir(parents=True)
        self.store.genesis_reference={'path':'genesis.json','sha256':'c'*64}
        self.commits={};self.store.inspect_committed=self.inspect

    def inspect(self,step,sha):
        marker,digest=self.commits[step]
        if sha!=digest:raise PermissionError('wrong inspected COMMIT')
        return copy.deepcopy(marker)

    def row(self,step=1,endpoint=None):
        endpoint=endpoint or self.handle.endpoint();rid=f'r{step}'
        public=dict(prompt='unchanged original',image_file='image.png',image_sha256='e'*64)
        request=dict(public,request_id=rid,scope='TRAIN',assistant_prefill='{',start_seed=11031,arm='T0',step=step,seed=44)
        self.requests[rid]=request
        values=torch.tensor([-.4,-.5]).tolist()
        row=dict(request_id=rid,start_seed=11031,arm='T0',scope='TRAIN',step=step,seed=44,tokens=[1,9],
            raw_tokens=[1,9],completion_token_count=2,sampler_logprobs=values,prompt_tensor_hash='d'*64,
            model_input=public,policy_sha256=endpoint['policy_sha256'],reference_sha256=endpoint['reference_sha256'],
            assistant_prefill='{',generation_status='COMPLETE',technical_validation_errors=[],finish_reason='balanced',
            sampler_logprob_source='actual_generate_raw_logits_selected_token')
        name=hashlib.sha256(rid.encode()).hexdigest()+'.CAPTURE.json';path=self.output/name
        path.write_text(json.dumps(row))
        path.with_name(name.replace('.CAPTURE.json','.BEGIN.json')).write_text(json.dumps(dict(request_id=rid,
            endpoint=endpoint,design_freeze_sha256='f'*64)))
        row.update(native_capture_path=str(path),native_capture_sha256=t.file_sha(path))
        return row

    def update_publish(self,step):
        for p in self.model.parameters():p.grad=torch.ones_like(p)*.01
        self.opt.step();self.scheduler.step();self.opt.zero_grad(set_to_none=True)
        path=self.store.path/f'step-{step:03d}';path.mkdir()
        previous=self.commits[step-1][1] if step>1 else None
        saved=t.state.write_checkpoint_files(path,self.runtime,self.opt,self.scheduler,self.reference,self.ops,
            identity=self.handle.path_identity,step=step,sampler_state=dict(committed_step=step,next_step=step+1,generated_responses=128*step),
            commit_fields={'transaction':{'previous_commit_sha256':previous}})
        self.commits[step]=(saved['commit'],saved['commit_sha256'])
        return saved

    def test_current_path_binds_and_dirty_policy_blocks_generation(self):
        row=self.row();self.runtime.bind_request(self.requests[row['request_id']]);self.runtime.unbind_request()
        before=self.handle.endpoint()
        with torch.no_grad():self.model.lora_fixture[0].add_(1.)
        with self.assertRaisesRegex(PermissionError,'Dirty/uncommitted'):
            self.runtime.bind_request(self.requests[row['request_id']])
        self.assertEqual(self.handle._step,0);self.assertEqual(before['step'],0)
        self.assertIsNone(self.runtime._satdose_active)

    def test_cross_arm_and_future_step_rejected(self):
        row=self.row();request=self.requests[row['request_id']]
        for key,value in [('arm','T1'),('step',2),('start_seed',11032)]:
            changed=dict(request,**{key:value});self.requests[row['request_id']]=changed
            with self.assertRaisesRegex(PermissionError,'committed path/cursor'):self.runtime.bind_request(changed)
        self.assertIsNone(self.runtime._satdose_active)

    def test_original_forward_keeps_all_tokens_sampler_and_rows_unchanged(self):
        row=self.row();before=copy.deepcopy(row)
        current,sampler,reference=self.handle.forward_original(row)
        self.assertEqual(current.numel(),2);self.assertEqual(sampler.tolist(),row['sampler_logprobs'])
        current.sum().backward();self.handle.verify_storage_after_backward()
        self.model.zero_grad(set_to_none=True)
        self.assertEqual(row,before);self.assertIsNone(self.runtime._satdose_active)
        self.storage.validate_snapshot_stats.assert_called_once()
        self.assertTrue(all(x['metadata']['execution_phase']=='TRAIN' for x in self.runtime._satdose_accounting))

    def test_tampered_capture_and_prepared_and_sampled_policy_rejected(self):
        row=self.row()
        for key,value in [('tokens',[5,6]),('prepared',{}),('policy_sha256','0'*64),('seed',45)]:
            with self.assertRaises(PermissionError):self.handle.forward_original(dict(row,**{key:value}))
        self.assertFalse(self.runtime._satdose_accounting)

    def test_advance_only_published_state_matching_live_hashes_and_next_step(self):
        self.row();saved=self.update_publish(1)
        with self.assertRaises(PermissionError):self.handle.endpoint()
        with self.assertRaisesRegex(PermissionError,'wrong inspected'):
            self.handle.advance_committed(self.store,1,'0'*64)
        self.assertEqual(self.handle._step,0)
        self.handle.advance_committed(self.store,1,saved['commit_sha256'])
        endpoint=self.handle.endpoint();self.assertEqual(endpoint['step'],1)
        self.handle.advance_committed(self.store,1,saved['commit_sha256'])
        row=self.row(2);self.runtime.bind_request(self.requests[row['request_id']]);self.runtime.unbind_request()
        with self.assertRaises(PermissionError):self.handle.forward_original(self.row(3,endpoint=endpoint))

    def test_uncommitted_optimizer_only_mutation_also_blocks_generation(self):
        self.row();self.opt.param_groups[0]['lr']=.01
        with self.assertRaisesRegex(PermissionError,'Uncommitted optimizer'):self.handle.endpoint()

    def test_live_mismatch_prevents_cursor_advance(self):
        saved=self.update_publish(1)
        with torch.no_grad():self.model.lora_fixture[0].add_(.01)
        with self.assertRaisesRegex(PermissionError,'Live state differs'):self.handle.advance_committed(self.store,1,saved['commit_sha256'])
        self.assertEqual(self.handle._step,0);self.assertEqual(self.runtime.identity['step'],0)

    def test_discard_dirty_state_releases_hook_without_unchanged_claim(self):
        with torch.no_grad():self.model.lora_fixture[0].add_(1.)
        self.model.lora_fixture[0].grad=torch.ones(1)
        result=self.handle.discard();self.handle.discard()
        self.assertTrue(result['discarded']);self.assertFalse(result['state_unchanged_asserted'])
        self.runtime._vision_hook.remove.assert_called_once()
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        with self.assertRaisesRegex(PermissionError,'Closed'):self.handle.endpoint()

    def canary_fixture(self):
        first=self.update_publish(1);self.handle.advance_committed(self.store,1,first['commit_sha256'])
        row=self.row(2)
        second=self.update_publish(2);self.handle.advance_committed(self.store,2,second['commit_sha256'])
        raw=self.root/'raw.jsonl';raw.write_text(json.dumps(row)+'\n')
        receipt=self.root/'rollout.json';receipt.write_text(json.dumps({'raw':{'path':'raw.jsonl','sha256':t.file_sha(raw)}}))
        evidence=self.store.path/'step-002/EVIDENCE.json'
        evidence.write_text(json.dumps({'rollout_receipt':{'path':'rollout.json','sha256':t.file_sha(receipt)}}))
        # Store full transaction inspection is an explicit fake boundary here;
        # production TransactionStore independently verifies 128 actual originals.
        self.commits[2][0]['transaction']['evidence']={'path':'EVIDENCE.json','sha256':t.file_sha(evidence)}
        return row,first['commit_sha256'],second['commit_sha256']

    def test_canary_old_tokens_are_not_misrepresented_as_current_training_teacher(self):
        row,first,second=self.canary_fixture();original=copy.deepcopy(row)
        with self.assertRaises(PermissionError):self.handle.forward_original(row)
        before=self.ops.state_hash(self.ops.capture_rng())
        result=self.handle.canary_teacher(row,self.store,sampled_commit_sha256=first,teacher_commit_sha256=second)
        self.assertEqual(result['sampled_endpoint']['step'],1);self.assertEqual(result['teacher_endpoint']['step'],2)
        self.assertEqual(result['tokens'],[1,9]);self.assertEqual(result['backward_calls'],0)
        self.assertEqual(result['status'],'CANARY_TEACHER_CAPTURE_NOT_RESUME_ADMISSION')
        self.assertEqual(row,original);self.assertIsNone(self.runtime._satdose_active)
        self.assertEqual(before,self.ops.state_hash(self.ops.capture_rng()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(x['metadata']['execution_phase']=='CANARY_RESTORE_TEACHER' for x in result['accounting']))

    def test_canary_missing_committed_membership_or_changed_original_rejected(self):
        row,first,second=self.canary_fixture()
        changed=dict(row,prompt_tensor_hash='wrong')
        with self.assertRaises(PermissionError):
            self.handle.canary_teacher(changed,self.store,sampled_commit_sha256=first,teacher_commit_sha256=second)
        with self.assertRaises(PermissionError):
            self.handle.canary_teacher(row,self.store,sampled_commit_sha256='0'*64,teacher_commit_sha256=second)
        self.assertFalse(self.runtime._satdose_accounting)

    def test_full_sequence_math_preserves_prefix_completion_masks(self):
        self.row();self.runtime.bind_request(self.requests['r1'])
        self.runtime.processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0));self.runtime.eos_ids=[9]
        self.runtime._satdose_repeat_prompt=lambda inputs,n:{k:v.clone() for k,v in inputs.items()}
        self.runtime._native_forward=Mock(return_value='same-native-forward')
        prepared={'routing':{'protocol':'structured_v4','protocol_revision':'v2','assistant_prefill':'{'},
            'inputs':{'input_ids':torch.tensor([[7,8]]),'attention_mask':torch.tensor([[1,1]]),
                      'mm_token_type_ids':torch.tensor([[0,0]])}}
        result=self.runtime._batch_forward(prepared,[[1,9]],purpose='training_policy',grad=True)
        self.assertEqual(result,'same-native-forward')
        args,kwargs=self.runtime._native_forward.call_args
        self.assertEqual(args[0]['input_ids'].tolist(),[[7,8,1,9]])
        self.assertEqual(args[0]['attention_mask'].tolist(),[[1,1,1,1]])
        self.assertEqual(kwargs,dict(completion_start=2,lengths=[2],purpose='training_policy',grad=True))
        self.runtime.unbind_request()

    def test_constructor_rejects_non_registry_before_loading_any_model(self):
        with patch.object(t,'verify_sources') as verify:
            with self.assertRaises(PermissionError):
                t.build_training_runtime(self.registry,self.root/'source.json',start_seed=11031,arm='T0',
                    output_dir=self.output,execution_sources={})
            verify.assert_not_called()


if __name__=='__main__':unittest.main()
