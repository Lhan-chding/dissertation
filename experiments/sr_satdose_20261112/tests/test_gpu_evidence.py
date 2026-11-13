"""Synthetic CPU artifact failures only; never production or CUDA evidence."""
import ast
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from execution import gpu_evidence as a
from execution import backend_equivalence, allzero_check
from runtime import diagnostics
sys.path.pop(0)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the unchanged pure validator AST from its frozen-SHA source,
        # avoiding native import namespace pollution or optimizer construction.
        root = PACKAGE.parents[1]
        source = Path(os.environ.get('SATDOSE_NATIVE_SRC', str(root.parent/'dissertation for NTU/sr-f6/src')))/'sr_f3_ce/training.py'
        if not source.exists():raise unittest.SkipTest('Pinned native validator source required')
        lock = json.loads((root/'artifacts/sr_satdose_20261112/intake/RUNTIME_SOURCE_LOCK.json').read_text())
        if a._sha(source) != lock['native_files']['sr_f3_ce/training.py']:
            raise AssertionError('Actual native validator source SHA differs')
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if
                 (isinstance(node,ast.FunctionDef) and node.name == 'validate_optimizer_state') or
                 (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'LEARNING_RATE' for t in node.targets))]
        if len(nodes) != 2:raise AssertionError('Native validator/constant AST not found')
        namespace = {}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        cls.ops = SimpleNamespace(validate_optimizer_state=namespace['validate_optimizer_state'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.e = a.Evidence(self.root)
        self.policy = {f'lora_{i}':torch.zeros(1) for i in range(496)}

    def write(self, name, value):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))
        return dict(path=str(name), sha256=a._sha(path))

    def production(self):
        rows = [dict(request_id=f'SYNTHETIC_CPU/{i}', tokens=[10,20],sampler_logprobs=[-.5,-.25]) for i in range(128)]
        grads = {n:torch.full_like(v,.001) for n,v in self.policy.items()}
        norm = math.sqrt(sum(float(v.double().square().sum()) for v in grads.values()))
        traces = [dict(row_id=r['request_id'],tokens=r['tokens'][:],sampler_logprobs=r['sampler_logprobs'][:],
                       train_logprobs=[-.4,-.3],reference_logprobs=[-.6,-.2]) for r in rows]
        c = dict(traces=traces,row_forward_callback_calls=128,backward_calls=128,sequence_denominator=128,
                 optimizer_created=False,optimizer_steps=0,scheduler_steps=0,parameter_updates=0,
                 preclip_gradients=copy.deepcopy(grads),postclip_gradients=grads,
                 numerics=dict(status='FINITE',gradient_norm_after_clip=norm,kl_unweighted=0.),
                 totals=dict(loss=0.,native=0.,aux=0.,kl=0.,kl_unweighted=0.))
        kernels = SimpleNamespace(numerical_audit=lambda _:dict(status='FINITE'),check_numerics=lambda _:None)
        return rows,c,kernels

    def test_cpu_production_values_no_cuda_no_optimizer(self):
        rows,c,k = self.production()
        with (patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA forbidden')),
              patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('Adam forbidden'))):
            a._production(c,rows,self.policy,k)

    def test_production_actual_counter_tamper(self):
        rows,c,k = self.production();c['backward_calls']=127
        with self.assertRaisesRegex(ValueError,'counter'):a._production(c,rows,self.policy,k)

    def test_production_tensor_tamper_without_scalar_update(self):
        rows,c,k = self.production();c['postclip_gradients']['lora_0'][0]=2.
        with self.assertRaisesRegex(ValueError,'norm'):a._production(c,rows,self.policy,k)

    def test_actual_preclip_norm_cannot_hide_behind_metadata(self):
        rows,c,k=self.production();c['preclip_gradients']['lora_0'][0]=10000.
        with self.assertRaisesRegex(ValueError,'preclip.*1000'):a._production(c,rows,self.policy,k)
        rows,c,k=self.production()
        for value in c['preclip_gradients'].values():value.zero_()
        c['preclip_gradients']['lora_0'][0]=1000.
        a._production(c,rows,self.policy,k)
        c['preclip_gradients']['lora_0'][0]=torch.nextafter(torch.tensor(1000.),torch.tensor(float('inf')))
        with self.assertRaisesRegex(ValueError,'preclip.*1000'):a._production(c,rows,self.policy,k)

    def test_original_token_or_sampler_tamper(self):
        rows,c,k = self.production();c['traces'][0]['tokens'][0]=99
        with self.assertRaisesRegex(ValueError,'tokens'):a._production(c,rows,self.policy,k)
        rows,c,k = self.production();c['traces'][0]['sampler_logprobs'][0]=-.9
        with self.assertRaisesRegex(ValueError,'sampler'):a._production(c,rows,self.policy,k)

    def test_missing_gradient_or_nonfinite_teacher(self):
        rows,c,k = self.production();del c['preclip_gradients']['lora_495']
        with self.assertRaisesRegex(ValueError,'496'):a._production(c,rows,self.policy,k)
        rows,c,k = self.production();c['traces'][0]['train_logprobs'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):a._production(c,rows,self.policy,k)

    def test_trace_long_row_requires_actual_full_vector(self):
        rows,c,k = self.production();rows[0].update(tokens=[7]*1536,sampler_logprobs=[-.5]*1536)
        c['traces'][0].update(tokens=[7]*1536,sampler_logprobs=[-.5]*1536,
                             train_logprobs=[-.4]*1536,reference_logprobs=[-.6]*1536)
        a._production(c,rows,self.policy,k)
        c['traces'][0]['reference_logprobs'].pop()
        with self.assertRaisesRegex(ValueError,'misaligned'):a._production(c,rows,self.policy,k)

    def test_accounting_requires_actual_events_not_claimed_totals(self):
        rows,_,_ = self.production()
        account=dict(events=[dict(kind='extra_forward_sequences',count=128,metadata=dict(purpose=p))
                             for p in ('training_policy','training_reference')],
                     measured_forward_sequences=dict(training_policy=128,training_reference=128),
                     storage=[dict(row_id=r['request_id'],forward_returned=True,
                                   stats=dict(row_id=r['request_id'])) for r in rows])
        a._accounting(account,rows,lambda _:None)
        account['events'][0]['count']=127
        with self.assertRaisesRegex(ValueError,'counters'):a._accounting(account,rows,lambda _:None)

    def test_shadow_moment_step_and_replay_actual_tensors(self):
        _,c,_=self.production()
        result=self.shadow_result(c)
        a._shadow_result(result,self.policy,c,self.ops)
        replay=copy.deepcopy(result);replay['optimizer']['state'][3]['exp_avg'][0]=.1
        self.assertFalse(a._same(result,replay))
        result['optimizer']['state'][3]['step']=torch.tensor(2.)
        with self.assertRaisesRegex(ValueError,'step'):a._shadow_result(result,self.policy,c,self.ops)

    def shadow_result(self,c):
        return dict(delta=copy.deepcopy(c['preclip_gradients']),clipped=copy.deepcopy(c['postclip_gradients']),
            optimizer=dict(param_groups=[dict(params=list(range(496)),lr=1e-5,initial_lr=1e-5,eps=1e-8,weight_decay=0.,
                betas=(.9,.999),foreach=False,capturable=False,fused=None,amsgrad=False,maximize=False,differentiable=False)],
                state={i:dict(step=torch.tensor(1.),exp_avg=torch.zeros(1),exp_avg_sq=torch.zeros(1)) for i in range(496)}),
            scheduler=dict(last_epoch=0,base_lrs=[1e-5],_last_lr=[1e-5]))

    def test_matching_wrong_shadow_recipes_rejected_without_adam(self):
        _,c,_=self.production()
        for key,value in [('lr',2e-5),('initial_lr',2e-5),('betas',(.8,.99)),('eps',1e-6),('weight_decay',.1),
                          ('foreach',True),('capturable',True),('fused',True),('amsgrad',True),('maximize',True),('differentiable',True)]:
            with self.subTest(key=key):
                result=self.shadow_result(c);result['optimizer']['param_groups'][0][key]=value
                replay=copy.deepcopy(result);self.assertTrue(a._same(result,replay))
                with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No optimizer construction')):
                    with self.assertRaises((ValueError,PermissionError)):a._shadow_result(result,self.policy,c,self.ops)
        for key in ('base_lrs','_last_lr'):
            result=self.shadow_result(c);result['scheduler'][key]=[2e-5]
            with self.assertRaisesRegex(PermissionError,'scheduler'):a._shadow_result(result,self.policy,c,self.ops)
        result=self.shadow_result(c);del result['optimizer']['param_groups'][0]['lr']
        with self.assertRaises(PermissionError):a._shadow_result(result,self.policy,c,self.ops)

    def test_sha_tamper_and_end_of_audit_mutation(self):
        ref=self.write('file.json',dict(scope='SYNTHETIC_CPU_ONLY'));self.e.ref(ref)
        (self.root/'file.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA'):a.Evidence(self.root).ref(ref)
        with self.assertRaisesRegex(ValueError,'changed'):self.e.stable()

    def test_evidence_escape_and_duplicate_json(self):
        with self.assertRaisesRegex(ValueError,'Escaped'):self.e.path('../x')
        (self.root/'x.json').write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.e.json('x.json')

    def chain(self):
        binding=dict(shadow_slots={})
        for field,prefix in [('evaluation_loader_slots','evaluation_loader_checks'),
                             ('backend_equivalence_slots','backend_equivalence'),('allzero_slots','allzero_check')]:
            binding[field]={str(s):f'{prefix}/start-{s}' for s in a.STARTS}
            for s,rel in binding[field].items():
                ref=self.write(rel+'/DATA.json',dict(scope='SYNTHETIC_CPU_CHAIN_ONLY'))
                self.write(rel+'/COMPLETE.json',dict(technical_binding_sha256='b'*64,start_seed=int(s),evidence=dict(data=ref)))
        for s in a.STARTS:
            for t in (0,1):
                rel=f'shadow_inputs/start-{s}/trial-{t}';claim=f'shadow_claims/start-{s}/trial-{t}'
                slot=dict(input_receipt=rel+'/INPUT_RECEIPT.json',restore=rel+'/RESTORE.json',capture=rel+'/PRODUCTION_CAPTURE.pt')
                binding['shadow_slots'][f'{s}/{t}']=slot
                restored=self.write(slot['restore'],{});inp=self.write(slot['input_receipt'],{})
                result=self.write(claim+'/RESULT.pt',{});acc=self.write(rel+'/ACCOUNTING.json',{})
                sc=dict(status='SHADOW_TRIAL_COMPLETE_NOT_GPU_ADMISSION',mode='production',technical_binding_sha256='b'*64,
                        start_seed=s,trial=t,optimizer_body_calls=1,optimizer_steps=1,formal_optimizer_steps=0,scheduler_steps=0,
                        input_receipt_sha256=inp['sha256'],result_sha256=result['sha256'])
                complete=self.write(claim+'/COMPLETE.json',sc)
                self.write(rel+'/WORKER_COMPLETE.json',dict(technical_binding_sha256='b'*64,start_seed=s,trial=t,
                    shadow_complete_sha256=complete['sha256'],restore_sha256=restored['sha256'],
                    actual_accounting_sha256=acc['sha256'],component_files={}))
        return binding

    def test_all_ten_completion_chain_required(self):
        binding=self.chain();self.assertEqual(len(a._complete_chain(self.e,binding,'b'*64)),10)
        (self.root/'allzero_check/start-11032/COMPLETE.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):a._complete_chain(a.Evidence(self.root),binding,'b'*64)

    def test_unknown_extra_shadow_slot_rejected(self):
        binding=self.chain();(self.root/'shadow_claims/start-11031/trial-2').mkdir()
        with self.assertRaisesRegex(ValueError,'Unknown shadow'):a._complete_chain(self.e,binding,'b'*64)

    def test_chain_wrong_binding_existing_hold_and_tampered_result(self):
        b=self.chain()
        with self.assertRaisesRegex(ValueError,'Foreign'):a._complete_chain(self.e,b,'c'*64)
        hold=self.root/'shadow_inputs/start-11031/trial-0/HOLD.json';hold.write_text('{}')
        with self.assertRaisesRegex(ValueError,'HOLD'):a._complete_chain(self.e,b,'b'*64)
        hold.unlink();(self.root/'shadow_claims/start-11031/trial-0/RESULT.pt').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'SHA'):a._complete_chain(a.Evidence(self.root),b,'b'*64)

    def test_existing_lock_never_created_and_busy_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing'):
            with a._existing_shared_lock(self.root):pass
        lock=self.root/'DIAGNOSTIC_WORKER.lock';lock.touch()
        with lock.open('r+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'producer'):
                with a._existing_shared_lock(self.root):pass
        with a._existing_shared_lock(self.root):pass

    def test_source_inventory_checks_noncode_extras_and_auditor_sha(self):
        release=self.root/'release';release.mkdir();(release/'code.py').write_text('# synthetic CPU only\n')
        sources={'code.py':a._sha(release/'code.py')}
        receipt=dict(schema='sr-satdose-gpu-technical-release-v1',status='AUDITED_FOR_GPU_CHECKS_ONLY',permission='NO_TRAINING',
                     experiment_id='SYNTHETIC_CPU_ONLY',evidence_root=str(self.root),design_freeze_sha256='f'*64,
                     config_sha256='c'*64,release_root=str(release),sources=sources)
        binding=dict(gpu_release=self.write('RELEASE.json',receipt),experiment_id='SYNTHETIC_CPU_ONLY',
                     design_freeze=dict(sha256='f'*64),config=dict(sha256='c'*64),
                     code=dict(synthetic=dict(path=str(release/'code.py'),sha256=sources['code.py'])))
        actual=a._sha(Path(a.__file__).resolve());a._release(self.e,binding,actual)
        with self.assertRaisesRegex(ValueError,'Auditor'):a._release(self.e,binding,'0'*64)
        (release/'pins.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'inventory'):a._release(self.e,binding,actual)

    def test_real_direction_summary_uses_global_repeat_floor(self):
        # Different halves: the global floor=.3 must fail effect1=.1 even though its own repeat error is zero.
        g=lambda value:{'lora_scalar':torch.tensor([value],dtype=torch.float32)}
        caps={k:dict(gradients=g(1.)) for k in ('J','source','total','J_half_1','J_half_2','source_half_1','source_half_2')}
        caps['KL']=dict(gradients=g(0.))
        for name,value in [('witness_half_1',.1),('witness_half_1_repeat',.1),('witness_half_2',1.),('witness_half_2_repeat',.7)]:
            caps[name]=dict(gradients=g(value))
        report=diagnostics.direction_summary(caps,g(1.),g(1.))
        self.assertFalse(report['direction_checks_satisfied']);self.assertGreater(report['effect_repeat_error_floor'],.29)

    def test_existing_backend_compare_reads_actual_gradient_values(self):
        c=dict(original_tokens=[1],original_sampler=[-.5],advantage=0.,sequence_denominator=128,
               teacher=torch.tensor([-.5]),reference=torch.tensor([-.5]),sampler=torch.tensor([-.5]),
               loss=torch.tensor(0.),scaled_loss=torch.tensor(0.),terms=dict(loss=torch.tensor(0.)),
               preclip_gradients=copy.deepcopy(self.policy))
        other=copy.deepcopy(c);other['preclip_gradients']['lora_495'][0]=1.
        self.assertFalse(backend_equivalence.compare(c,other)['bit_exact'])

    def test_matching_wrong_backend_advantages_and_sources_rejected(self):
        sources=dict(native='SYNTHETIC_CPU_ONLY',legacy='SYNTHETIC_CPU_ONLY');selection=dict(advantage=.5)
        c=dict(advantage=.5,status='BACKEND_PATH_CAPTURE_NOT_GPU_ADMISSION',sources=sources,visible_gpu_count=2,
               storage_path='single_device_activation_retention',storage=None)
        a._backend_identity(c,selection,sources,two_device=False)
        for key,value in [('advantage',.1),('sources',{}),('visible_gpu_count',1),('status','OTHER'),
                          ('storage_path','two_device_saved_tensors'),('storage',{})]:
            with self.subTest(key=key):
                bad=copy.deepcopy(c);bad[key]=value
                self.assertTrue(a._same(bad,copy.deepcopy(bad)))
                with self.assertRaisesRegex(ValueError,'identity'):a._backend_identity(bad,selection,sources,two_device=False)

    def test_existing_allzero_compare_checks_actual_pg_and_recomposition(self):
        def capture(value):
            return dict(sequence_denominator=128,original_scores_unchanged=True,explicit_diagnostic_advantage=0.,
                native_terms=[dict(native=0.,loss=.02,kl=.02,kl_unweighted=1.) for _ in range(128)],
                policy_before_sha256='p',policy_after_sha256='p',original_reference_sha256='r',reference_after_sha256='r',
                rng_unchanged=True,traces=[],original_rows_sha256='synthetic',
                gradients={n:torch.full_like(v,value) for n,v in self.policy.items()})
        captures=dict(PG_ZERO=capture(0.),KL=capture(1.),TOTAL=capture(.02))
        report=allzero_check.compare(captures);self.assertTrue(report['pg_exact_zero'])
        captures['PG_ZERO']['gradients']['lora_495'][0]=1e-8
        with self.assertRaisesRegex(PermissionError,'exactly zero'):allzero_check.compare(captures)

    def test_formal_update_rejected_even_with_rewritten_marker_sha(self):
        b=self.chain();rel='shadow_claims/start-11031/trial-0/COMPLETE.json'
        sc=a._read(self.root/rel);sc['formal_optimizer_steps']=1;ref=self.write(rel,sc)
        worker='shadow_inputs/start-11031/trial-0/WORKER_COMPLETE.json'
        wc=a._read(self.root/worker);wc['shadow_complete_sha256']=ref['sha256'];self.write(worker,wc)
        with self.assertRaisesRegex(ValueError,'counters'):a._complete_chain(self.e,b,'b'*64)


if __name__ == '__main__':
    unittest.main()
