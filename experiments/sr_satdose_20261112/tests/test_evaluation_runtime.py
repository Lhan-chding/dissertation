"""CPU synthetic sampler + real durable ledger; never production authority/GPU proof."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from execution import evaluation_runtime as e
from runtime.backend import GenerationBridge, GenerationLedger


def tensor_hash(value):
    def convert(x):
        if isinstance(x, torch.Tensor):
            return dict(dtype=str(x.dtype), shape=list(x.shape), bytes=x.detach().cpu().numpy().tobytes().hex())
        if isinstance(x, dict):return {k:convert(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)):return [convert(v) for v in x]
        return x
    return e.object_sha(convert(value))


class CPUAuthority(e.AuthenticatedEvaluationAuthority):
    """Only a test source trusted by explicit test fixture references."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.mutate = lambda value:value
        self.calls = []
        super().__init__(fixture.authority_ref)

    def verify(self, *, root, selection, endpoint, design_freeze_sha256,
               execution_sources_sha256, request_ids_sha256, request_id):
        self.calls.append(request_id)
        f = self.fixture
        value = dict(schema='sr-satdose-authenticated-evaluation-authority-v1',
            selection=selection, endpoint=endpoint, design_freeze_sha256=design_freeze_sha256,
            config_sha256=f.config_sha, execution_sources_sha256=execution_sources_sha256,
            request_ids_sha256=request_ids_sha256, request_id=request_id, **copy.deepcopy(f.refs))
        if request_id is not None:
            value['begin'] = f.ref(f.output/(hashlib.sha256(request_id.encode()).hexdigest()+'.BEGIN.json'))
        if selection['scope'] == 'MAIN':
            if not f.allow_main:raise PermissionError('CPU fixture terminal authority denied MAIN')
            value['terminal'] = f.ref(f.terminal)
        return self.mutate(value)


class CPURuntime(e.EvaluationRuntimeMixin):
    pass


class EvaluationRuntimeTests(unittest.TestCase):
    def setUp(self):
        rng = torch.get_rng_state().clone(); self.addCleanup(torch.set_rng_state,rng)
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve();self.output = self.root/'evaluation';self.output.mkdir()
        self.config_sha = 'c'*64
        self.freeze = self.root/'DESIGN_FREEZE.json';self.freeze.write_text(json.dumps(dict(config_sha256=self.config_sha)))
        self.freeze_sha = e.file_sha(self.freeze)
        self.authority_ref = dict(path=str(Path(__file__).resolve()),sha256=e.file_sha(__file__))
        self.sources = {key:dict(path=str(Path(path).resolve()),sha256=e.file_sha(path)) for key,path in
            [('evaluation_runtime',e.__file__),('evaluation_loader',e.loader.__file__),
             ('training_state',e.state.__file__),('training_transaction',e.tx.__file__)]}
        self.sources['authority'] = self.authority_ref
        self.sources_sha = e.object_sha(self.sources)
        self.refs = {}
        self.refs['registration'] = self.write('registration.json',dict(schema='sr-satdose-registration-v1',
            mode='production',experiment_id=e.state.EXPERIMENT,config_sha256=self.config_sha,
            artifacts=dict(design_freeze=dict(path='DESIGN_FREEZE.json',sha256=self.freeze_sha))))
        self.refs['gpu_report'] = self.write('gpu.json',dict(schema='sr-satdose-gpu-verification-v1',mode='production',
            experiment_id=e.state.EXPERIMENT,config_sha256=self.config_sha,registration_sha256=self.refs['registration']['sha256']))
        self.refs['admission'] = self.write('admission.json',dict(schema='sr-satdose-admission-v1',
            experiment_id=e.state.EXPERIMENT,status='ADMITTED_FIRST_TWO_STEPS_ONLY',permission='FOUR_PATHS_MAX_STEP_2',
            failures=[],config_sha256=self.config_sha,registration_sha256=self.refs['registration']['sha256'],
            gpu_report_sha256=self.refs['gpu_report']['sha256']))
        self.terminal = self.root/'terminal.json';self.terminal.write_text('{"CPU_TEST_ONLY":true}')
        self.allow_main = False
        self.image = self.root/'image.png';self.image.write_bytes(b'CPU test fixture, no decoded image')
        self.requests = {f'dev-{i}':dict(request_id=f'dev-{i}',scope='DEV',start_seed=11031,arm='START',step=0,
            seed=1000+i,prompt='frozen public prompt',image_file='image.png',image_sha256=e.file_sha(self.image),
            assistant_prefill='{') for i in range(108)}
        self.registered_requests = copy.deepcopy(self.requests)
        self.registry = SimpleNamespace(root=self.root,freeze_path=self.freeze,freeze_sha256=self.freeze_sha,
            requests=self.requests,config=dict(experiment_id=e.state.EXPERIMENT,parent='b0'),
            request=lambda rid:copy.deepcopy(self.registered_requests[rid]))
        self.model = torch.nn.Module();self.model.lora_fixture = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor([.1])) for _ in range(496)])
        self.reference = {n:p.detach().clone()-.2 for n,p in self.model.named_parameters()}
        self.ops = SimpleNamespace(state_hash=tensor_hash, capture_rng=lambda:torch.get_rng_state().clone(),
            restore_rng=torch.set_rng_state, trainable_state=lambda m:{n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad})
        self.endpoint = dict(experiment_id=e.state.EXPERIMENT,start_seed=11031,arm='START',step=0,
            policy_sha256=tensor_hash(self.ops.trainable_state(self.model)),reference_sha256=tensor_hash(self.reference))
        self.selection = {k:v for k,v in self.endpoint.items() if k not in ('policy_sha256','reference_sha256')} | dict(scope='DEV')
        self.runtime = CPURuntime();r=self.runtime
        r.model=self.model;r.torch=torch;r._satdose_active=None;r._satdose_closed=False;r._evaluation_handle=None
        r._satdose_accounting=[];r._satdose_output=self.output;r._satdose_registry=self.registry
        r._vision_hook=SimpleNamespace(remove=Mock())
        r.identity=dict(arm='START',step=0,execution_mode='AUTHENTICATED_EVALUATION')
        self.underlying=SimpleNamespace(runtime=r,registry=self.registry,output=self.output,native_ops=self.ops,
            fork=SimpleNamespace(reference=self.reference,optimizer=None,scheduler=None))
        self.authority=CPUAuthority(self)
        self.native_capture=dict(tokens=[1,2],raw_tokens=[1,2],completion_token_count=2,
            sampler_logprobs=[-.4,-.3],sampler_logprob_source='actual_generate_raw_logits_selected_token',
            assistant_prefill='{',generated_text='"selected":[]}',full_text='{"selected":[]}',
            generation_status='COMPLETE',technical_validation_errors=[],finish_reason='balanced')
        self.observed=[];self.inject=lambda:None
        self.handle=self.make_handle()
        self.bridge=GenerationBridge(self.registry,self.output,self.handle,self.handle.endpoint)

    def ref(self,path):return dict(path=str(path.relative_to(self.root)),sha256=e.file_sha(path))
    def write(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value));return self.ref(path)

    def make_handle(self):
        def runner(request,root,capture):
            self.runtime.bind_request(request)
            try:
                self.observed.append(copy.deepcopy(request))
                self.assertFalse(torch.is_grad_enabled())
                torch.rand(3)
                self.runtime.reserve('completion_attempts',1)
                self.inject()
                capture(copy.deepcopy(self.native_capture))
                self.runtime.reserve('generated_tokens',2)
            finally:self.runtime.unbind_request()
        h=e.EvaluationHandle(self.underlying,self.selection,self.endpoint,list(self.requests.values()),
            self.authority,self.authority_ref,self.sources,self.sources_sha,{},self.root/'source_lock.json',runner)
        self.runtime._evaluation_handle=h
        return h

    def authorization(self,request=None):
        return e._authorization(self.registry,self.selection,self.endpoint,e.object_sha(list(self.requests)),
            self.authority,self.authority_ref,self.sources_sha,self.output,request)

    def test_bridge_preserves_raw_capture_request_rng_and_policy_without_optimizer(self):
        before=copy.deepcopy(self.requests);rng=tensor_hash(self.ops.capture_rng())
        with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('optimizer forbidden')):
            result=self.bridge.generate_one('dev-0')
        self.assertEqual(self.requests,before);self.assertEqual(self.observed,[before['dev-0']])
        self.assertEqual(tensor_hash(self.ops.capture_rng()),rng);self.assertEqual(self.handle.endpoint(),self.endpoint)
        native=e.read_json(Path(result['native_capture_path']))
        self.assertEqual(native,self.native_capture)
        self.assertEqual(native['tokens'],result['tokens']);self.assertEqual(native['sampler_logprobs'],result['sampler_logprobs'])
        self.assertEqual(result['native_capture_sha256'],e.file_sha(result['native_capture_path']))
        audit=self.handle.audit();self.assertEqual(audit['attempted_requests'],1)
        self.assertEqual(audit['optimizer_steps'],0);self.assertEqual(audit['parameter_updates'],0)
        self.assertEqual(self.authority.calls,['dev-0'])
        self.assertTrue(self.handle.close()['closed']);self.runtime._vision_hook.remove.assert_called_once()

    def test_completed_request_cannot_repeat_even_with_new_handle(self):
        self.bridge.generate_one('dev-0')
        h=self.make_handle();bridge=GenerationBridge(self.registry,self.output,h,h.endpoint)
        with self.assertRaisesRegex(ValueError,'Existing attempt'):bridge.generate_one('dev-0')
        self.assertEqual(len(self.observed),1)

    def test_unknown_begin_blocks_fresh_handle_and_never_resamples(self):
        GenerationLedger(self.output).begin(self.requests['dev-0'],self.endpoint,self.freeze_sha)
        with self.assertRaisesRegex(ValueError,'Existing attempt'):self.bridge.generate_one('dev-0')
        self.assertEqual(self.observed,[])

    def test_sampler_failure_retains_begin_error_restores_rng_and_poisons(self):
        before=tensor_hash(self.ops.capture_rng())
        self.inject=lambda:(_ for _ in ()).throw(RuntimeError('synthetic sampler interruption'))
        with self.assertRaisesRegex(RuntimeError,'sampler interruption'):self.bridge.generate_one('dev-0')
        self.assertEqual(tensor_hash(self.ops.capture_rng()),before)
        self.assertEqual(len(list(self.output.glob('*.BEGIN.json'))),1)
        self.assertEqual(len(list(self.output.glob('*.ERROR.json'))),1)
        self.assertEqual(len(list(self.output.glob('*.CAPTURE.json'))),0)
        with self.assertRaisesRegex(PermissionError,'Closed/failed'):self.handle.generate('dev-1',lambda row:None)
        with self.assertRaises(PermissionError):self.handle.close()
        self.runtime._vision_hook.remove.assert_called_once()

    def test_rng_restore_failure_poisons_even_after_native_capture(self):
        self.ops.restore_rng=Mock(side_effect=RuntimeError('synthetic RNG restore failure'))
        with self.assertRaisesRegex(RuntimeError,'RNG restore failure'):self.bridge.generate_one('dev-0')
        self.assertEqual(len(list(self.output.glob('*.CAPTURE.json'))),1)
        self.assertEqual(len(list(self.output.glob('*.COMPLETE.json'))),0)
        with self.assertRaisesRegex(PermissionError,'Closed/failed'):self.handle.endpoint()

    def test_policy_mutation_is_rejected_and_attempt_not_retried(self):
        def mutate():
            with torch.no_grad():next(self.model.parameters()).add_(1)
        self.inject=mutate
        with self.assertRaisesRegex(PermissionError,'policy/reference changed'):self.bridge.generate_one('dev-0')
        self.assertTrue(self.handle._poisoned)
        self.assertEqual(len(list(self.output.glob('*.ERROR.json'))),1)

    def test_request_mutation_fails_before_sampler(self):
        self.handle._requests['dev-0']['seed']+=1
        with self.assertRaisesRegex(PermissionError,'Unregistered/repeated'):self.bridge.generate_one('dev-0')
        self.assertEqual(self.observed,[])

    def test_source_inventory_must_match_independent_expected_digest(self):
        bad=copy.deepcopy(self.sources);bad['evaluation_runtime']['sha256']='0'*64
        with self.assertRaisesRegex(PermissionError,'Untrusted evaluation execution'):
            e._execution_sources(self.authority,self.authority_ref,bad,self.sources_sha)
        with self.assertRaisesRegex(PermissionError,'Unbound evaluation source'):
            e._execution_sources(self.authority,self.authority_ref,bad,e.object_sha(bad))
        with self.assertRaisesRegex(PermissionError,'AuthenticatedEvaluationAuthority'):
            e._execution_sources(SimpleNamespace(verify=lambda **kw:{}),self.authority_ref,self.sources,self.sources_sha)

    def test_replaced_authority_method_and_wrong_trusted_source_rejected(self):
        with self.assertRaisesRegex(PermissionError,'source/method'):
            e._execution_sources(self.authority,dict(self.authority_ref,sha256='0'*64),self.sources,self.sources_sha)
        self.authority.verify=lambda **kw:{}
        with self.assertRaisesRegex(PermissionError,'source/method'):self.authorization()

    def test_scope_substitution_in_authenticated_reply_rejected(self):
        self.authority.mutate=lambda v:dict(v,request_ids_sha256='0'*64)
        with self.assertRaisesRegex(PermissionError,'scope differs'):self.authorization()

    def test_admission_file_tampering_rejected_by_hash(self):
        (self.root/'admission.json').write_text('{}')
        with self.assertRaises(PermissionError):self.authorization()

    def test_main_requires_terminal_authority_even_if_bridge_receives_true_flag(self):
        self.selection['scope']='MAIN';self.handle._selection['scope']='MAIN'
        self.requests['dev-0']['scope']='MAIN';self.registered_requests['dev-0']['scope']='MAIN'
        self.handle._requests['dev-0']['scope']='MAIN'
        with self.assertRaisesRegex(PermissionError,'denied MAIN'):
            self.bridge.generate_one('dev-0',terminal={'all_paths_terminal':True})
        self.assertEqual(self.observed,[])

    def test_main_missing_or_changed_terminal_reference_rejected(self):
        self.selection['scope']='MAIN';self.allow_main=True
        self.authority.mutate=lambda v:{k:x for k,x in v.items() if k!='terminal'}
        with self.assertRaises(KeyError):self.authorization()
        self.authority.mutate=lambda v:dict(v,terminal=dict(path='terminal.json',sha256='0'*64))
        with self.assertRaises(PermissionError):self.authorization()

    def test_begin_identity_and_prior_capture_rejected(self):
        request=self.requests['dev-0']
        prefix=GenerationLedger(self.output).begin(request,self.endpoint,self.freeze_sha)
        path=prefix.with_suffix('.BEGIN.json');raw=e.read_json(path);raw['seed']+=1;path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(PermissionError,'BEGIN identity'):self.authorization(request)
        raw['seed']-=1;path.write_text(json.dumps(raw))
        prefix.with_suffix('.CAPTURE.json').write_text('{}')
        with self.assertRaisesRegex(PermissionError,'Existing/foreign'):self.authorization(request)

    def test_constructor_cannot_claim_begin(self):
        self.authority.mutate=lambda v:dict(v,begin={})
        with self.assertRaisesRegex(PermissionError,'Constructor'):self.authorization()

    def test_full_fixed_endpoint_selection_no_extra_rows(self):
        selection,rows=e._selection(self.registry,11031,'START',0,'DEV')
        self.assertEqual(selection,self.selection);self.assertEqual(len(rows),108)
        rows[0]['seed']+=1;self.assertNotEqual(rows[0],self.requests['dev-0'])
        del self.requests['dev-107']
        with self.assertRaisesRegex(PermissionError,'Incomplete/expanded'):e._selection(self.registry,11031,'START',0,'DEV')
        with self.assertRaisesRegex(PermissionError,'Only independently'):e._selection(self.registry,11031,'START',0,'TRAIN')

    def test_live_optimizer_or_gradients_rejected(self):
        self.underlying.fork.optimizer=object()
        with self.assertRaisesRegex(PermissionError,'Dirty evaluation'):self.handle.endpoint()
        self.underlying.fork.optimizer=None;next(self.model.parameters()).grad=torch.ones(1)
        with self.assertRaisesRegex(PermissionError,'Dirty evaluation'):self.handle.endpoint()

    def test_teacher_and_gradient_accounting_are_unavailable(self):
        for name in ('_batch_forward','_native_forward','batch_sequence_forward','batch_reference_forward'):
            with self.subTest(name=name),self.assertRaises(PermissionError):getattr(self.runtime,name)()
        with self.assertRaisesRegex(PermissionError,'forbids'):self.runtime.reserve('completion_attempts',1)
        self.runtime._satdose_active=self.requests['dev-0']
        for kind in ('extra_forward_sequences','diagnostic_gradient_calls'):
            with self.assertRaisesRegex(PermissionError,'forbids'):self.runtime.reserve(kind,1)

    def test_public_input_preparation_preserves_frozen_image_and_rejects_gold(self):
        request=self.requests['dev-0'];self.runtime._satdose_active=copy.deepcopy(request)
        prepared={'routing':dict(protocol='structured',protocol_revision='v2',assistant_prefill='{')}
        self.runtime._satdose_legacy=SimpleNamespace(prepare_public_input=Mock(return_value=prepared))
        public={k:request[k] for k in e.PUBLIC}
        self.assertIs(self.runtime.prepare_input(public,self.root,protocol='structured'),prepared)
        self.runtime._validate_prepared_revision(prepared)
        self.runtime._validate_record_revision(dict(protocol='structured',protocol_revision='v2',assistant_prefill='{'))
        with self.assertRaisesRegex(ValueError,'Public input differs'):
            self.runtime.prepare_input(dict(public,gold_answer='private'),self.root,protocol='structured')
        self.image.write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError,'Frozen image changed'):
            self.runtime.prepare_input(public,self.root,protocol='structured')
        self.runtime._satdose_legacy.prepare_public_input.assert_called_once()

    def test_native_token_alias_failure_keeps_capture_and_no_resampling(self):
        self.native_capture['raw_tokens']=[7,8]
        with self.assertRaisesRegex(ValueError,'token aliases differ'):self.bridge.generate_one('dev-0')
        self.assertEqual(len(list(self.output.glob('*.CAPTURE.json'))),1)
        self.assertEqual(len(list(self.output.glob('*.ERROR.json'))),1)
        with self.assertRaisesRegex(ValueError,'Existing attempt'):self.bridge.generate_one('dev-0')
        self.assertEqual(len(self.observed),1)

    def test_interrupt_preserves_unknown_begin_and_poisons_runtime(self):
        rng=tensor_hash(self.ops.capture_rng())
        self.inject=lambda:(_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):self.bridge.generate_one('dev-0')
        self.assertEqual(tensor_hash(self.ops.capture_rng()),rng)
        self.assertEqual(len(list(self.output.glob('*.BEGIN.json'))),1)
        self.assertEqual(len(list(self.output.glob('*.CAPTURE.json'))),0)
        self.assertTrue(self.handle._poisoned)
        h=self.make_handle();bridge=GenerationBridge(self.registry,self.output,h,h.endpoint)
        with self.assertRaisesRegex(ValueError,'Existing attempt'):bridge.generate_one('dev-0')

    def test_direct_native_bind_without_handle_generation_rejected(self):
        with self.assertRaisesRegex(PermissionError,'Unregistered/repeated'):
            self.runtime.bind_request(self.requests['dev-0'])
        self.assertIsNone(self.runtime._satdose_active)

    def test_reference_mutation_rejected_without_starting_another_draw(self):
        self.reference[next(iter(self.reference))].add_(1)
        with self.assertRaisesRegex(PermissionError,'policy/reference changed'):self.bridge.generate_one('dev-0')
        self.assertEqual(self.observed,[])

    def test_original_file_change_detected_on_close_and_hook_removed(self):
        path=self.root/'original.pt';path.write_bytes(b'original')
        self.handle._files={str(path):e.file_sha(path)};path.write_bytes(b'changed')
        with self.assertRaisesRegex(PermissionError,'source changed'):self.handle.close()
        self.runtime._vision_hook.remove.assert_called_once();self.assertTrue(self.runtime._satdose_closed)

    def test_builder_failure_restores_sys_path_and_caller_rng_without_model(self):
        statefile=self.root/'original.pt';statefile.write_bytes(b'CPU checkpoint fixture')
        parent=self.root/'parent';parent.mkdir();(parent/'adapter_config.json').write_text('{}')
        source=dict(original_state_path=str(statefile),state_sha256=e.file_sha(statefile),
            policy_sha256=self.endpoint['policy_sha256'],reference_sha256=self.endpoint['reference_sha256'])
        self.registry.config['start_bindings']={'11031':source}
        lock=dict(native_files={},legacy_files={},parent_files={'adapter_config.json':e.file_sha(parent/'adapter_config.json')})
        before=list(sys.path);rng=tensor_hash(self.ops.capture_rng())
        def fail_import(*args):
            torch.rand(2)
            raise RuntimeError('synthetic import interruption')
        with patch.object(e,'_registry',return_value=self.registry), \
             patch.object(e.loader,'verify_sources',return_value=(lock,Path('/fixture/native'),Path('/fixture/legacy'),Path('/fixture/model'),parent)), \
             patch.object(e.fork_loader,'load_native_ops',return_value=self.ops), \
             patch.object(e.frozen,'_import',side_effect=fail_import):
            with self.assertRaisesRegex(RuntimeError,'import interruption'):
                e.build_evaluation_runtime(self.registry,self.root/'lock.json',start_seed=11031,arm='START',step=0,scope='DEV',
                    output_dir=self.output,authority=self.authority,authority_source_reference=self.authority_ref,
                    execution_sources=self.sources,expected_execution_sources_sha256=self.sources_sha)
        self.assertEqual(sys.path,before);self.assertEqual(tensor_hash(self.ops.capture_rng()),rng)
        self.assertEqual(list(self.output.iterdir()),[])


if __name__=='__main__':unittest.main()
