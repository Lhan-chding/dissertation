"""Standalone MAIN boundary tests; real bridge with a synthetic CPU sampler."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import evaluation_runtime,training_transaction,storage_ready,first_two_control
from runtime import backend,fork_loader
from analysis import evaluation

def standalone(name,file):
    spec=importlib.util.spec_from_file_location(name,PACKAGE/'execution'/file)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

c=standalone('satdose_main_control_tests','main_control.py')
ta=standalone('satdose_terminal_authority_for_main_tests','terminal_authority.py')


class FakeHandle:
    def __init__(self,fx,authority,output):
        self.fx=fx;self.authority=authority;self.output=output;self.attempted=0;self.closed=False
        self.ep=copy.deepcopy(authority.ctx['endpoint']);self.auth(None)
    def auth(self,rid):
        a=self.authority;x=a.ctx
        return a.verify(root=c.ROOT,selection=x['selection'],endpoint=self.ep,design_freeze_sha256=c.FREEZE_SHA,
            execution_sources_sha256=x['sources_sha256'],request_ids_sha256=backend.object_sha([r['request_id'] for r in x['requests']]),
            request_id=rid)
    def endpoint(self):return copy.deepcopy(self.ep)
    def __call__(self,request,root,capture):
        self.auth(request['request_id']);self.attempted+=1;self.fx.attempts.append(request['request_id'])
        if self.fx.fail_at==self.attempted:raise RuntimeError('synthetic interrupted MAIN draw')
        if self.fx.hook:self.fx.hook(self)
        capture(dict(tokens=[12,13],raw_tokens=[12,13],completion_token_count=2,sampler_logprobs=[-.2,-.3],
            sampler_logprob_source='actual_generate_raw_logits_selected_token',seed=request['seed'],assistant_prefill='{',
            full_text='{"selected":["A"]}',generated_text='"selected":["A"]}',
            model_input={k:request[k] for k in ('prompt','image_file','image_sha256')},prompt_tensor_hash='a'*64,
            finish_reason='balanced',generation_status='COMPLETE',technical_validation_errors=[]))
        with (self.output/'accounting.jsonl').open('a') as f:
            for kind,count in [('completion_attempts',1),('generated_tokens',2)]:
                f.write(json.dumps(dict(kind=kind,count=count,metadata=dict(group_seed=request['seed'])))+'\n')
    def close(self):
        self.closed=True
        return dict(self.ep,attempted_requests=self.attempted,optimizer_steps=0,scheduler_steps=0,parameter_updates=0)


class MainControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.process=dict(host='CPU_FIXTURE',boot_id='BOOT',pid=1001,start_ticks=1)
        self.handles=[];self.attempts=[];self.fail_at=None;self.hook=None
        (self.root/'DIAGNOSTIC_WORKER.lock').write_bytes(b'')
        self.cfg=dict(experiment_id=c.EXPERIMENT,parent='b0',start_bindings={str(s):dict(policy_sha256=str(s)[-1]*64,
            reference_sha256='a'*64) for s in (11031,11032)})
        self.write('config.json',self.cfg);self.write('source_lock.json',{'native_src':'/CPU_ONLY','native_files':{}})
        (self.root/'image.png').write_bytes(b'CPU image fixture');self.requests={};self.manifest=[]
        for (start,arm,step),count in c.allocations().items():
            for i in range(count):
                rid=f'CPU/MAIN/{start}/{arm}/{step}/{i}'
                m=dict(request_id=rid,split='MAIN',start_seed=start,arm=arm,step=step,seed=1000+i,panel='list8',view='conflict',
                    root_id=f'root-{i}',structure='TIES',draw=0,case_id='case',input_id='input')
                self.manifest.append(m);self.requests[rid]=dict(m,scope='MAIN',prompt='public unchanged',image_file='image.png',
                    image_sha256=c.sha(self.root/'image.png'),assistant_prefill='{')
        (self.root/'main.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in self.manifest))
        self.freeze=dict(config_sha256=c.sha(self.root/'config.json'),artifacts={k:self.ref(p) for k,p in
            [('config','config.json'),('source_lock','source_lock.json'),('main_manifest','main.jsonl'),('inputs','config.json'),
             ('images','config.json'),('train_manifest','main.jsonl'),('dry_manifest','main.jsonl'),
             ('dev_manifest','main.jsonl'),('prefix_calib_manifest','main.jsonl')]})
        self.write('DESIGN_FREEZE.json',self.freeze);self.freeze_sha=c.sha(self.root/'DESIGN_FREEZE.json')
        for key,value in [('ROOT',self.root),('FREEZE_SHA',self.freeze_sha),('CONFIG_SHA',self.freeze['config_sha256'])]:self.patch(c,key,value)
        self.registry=SimpleNamespace(root=self.root,freeze_path=self.root/'DESIGN_FREEZE.json',freeze_sha256=self.freeze_sha,
            config=self.cfg,requests=self.requests,request=lambda rid:copy.deepcopy(self.requests[rid]))
        for name in ('admission','registration','gpu','terminal','terminal_spec','original','budget'):self.write(name+'.json',{'CPU_ONLY':True})
        self.write('terminal.json',{'CPU_ONLY':True,'all_paths_terminal':True})
        self.write('admission_spec.json',dict(documents=dict(registration=self.ref('registration.json'),gpu_report=self.ref('gpu.json'))))
        self.modules={'execution.evaluation_runtime':evaluation_runtime,'execution.training_transaction':training_transaction,
            'execution.storage_ready':storage_ready,'execution.first_two_control':first_two_control,'runtime.backend':backend,
            'runtime.fork_loader':fork_loader,'analysis.evaluation':evaluation}
        sources={f'execution/{name}.py':c.sha(PACKAGE/'execution'/(name+'.py')) for name in
            ('evaluation_runtime','evaluation_loader','training_state','training_transaction')}
        self.terminal=SimpleNamespace(root=self.root,package=PACKAGE,original={'sources':sources,'execution_sources':{'CPU_ONLY':'fixed'}},
            receipt=self.ref('terminal.json'),modules=self.modules,stable=Mock(),audit_spec=dict(admission=self.ref('admission.json'),
                admission_spec=self.ref('admission_spec.json'),trusted_commits={}))
        self.stores={}
        for start in (11031,11032):
            for arm in ('T0','T1'):
                refs=[];path=self.root/'training_transactions'/f'start-{start}'/arm
                identity=dict(train_seed=start,arm=arm,execution_sources_sha256=backend.object_sha(self.terminal.original['execution_sources']))
                for step in range(1,25):
                    slot=path/f'step-{step:03d}';slot.mkdir(parents=True);(slot/'state.pt').write_bytes(b'CPU_ONLY')
                    marker=dict(step=step,identity=identity,policy_sha256='b'*64,reference_sha256='a'*64,
                        state_sha256=c.sha(slot/'state.pt'),transaction={'genesis_reference':self.ref('admission.json')})
                    (slot/'COMMIT.json').write_text(json.dumps(marker));refs.append(dict(path=str((slot/'COMMIT.json').relative_to(self.root)),sha256=c.sha(slot/'COMMIT.json')))
                self.terminal.audit_spec['trusted_commits'][f'{start}/{arm}']=refs
                def inspect(step,digest,path=path):
                    p=path/f'step-{step:03d}/COMMIT.json';c.require(c.sha(p)==digest,'COMMIT changed');return c.read(p)
                self.stores[f'{start}/{arm}']=SimpleNamespace(path=path,inspect_committed=inspect)
        self.release=dict(schema='sr-satdose-main-release-v1',endpoint=dict(start_seed=11031,arm='START',step=0),external_root=str(PACKAGE),
            external_sources={'execution/main_control.py':c.sha(PACKAGE/'execution/main_control.py')},
            original_train_release=self.ref('original.json'),terminal_spec=self.ref('terminal_spec.json'),
            storage_budget=self.ref('budget.json'),completed_endpoints={})
        self.release_path=self.root/'release.json';self.update_release()
        c._configure(ta,self.terminal)
        self.real_bootstrap=c.bootstrap;self.boot=self.patch(c,'bootstrap',return_value=(self.release,self.terminal))
        self.patch(c,'_process',side_effect=lambda:copy.deepcopy(self.process))
        self.patch(backend,'FrozenRegistry',return_value=self.registry)
        self.validate_manifest=self.patch(evaluation,'validate_manifest',return_value={})
        self.patch(first_two_control,'_allocation',return_value={})
        self.patch(fork_loader,'load_native_ops',return_value=object())
        self.patch(training_transaction,'TransactionStore',side_effect=lambda root,freeze,sha,identity,**kw:self.stores[f"{identity['train_seed']}/{identity['arm']}"])
        self.budget=self.patch(storage_ready,'_budget',return_value={'scope':'MAIN_ONLY'})
        self.storage=self.patch(c,'storage_check',return_value={'CPU_ONLY':True})
        self.factory_mock=self.patch(evaluation_runtime,'build_evaluation_runtime',side_effect=self.factory)
    def patch(self,obj,name,*args,**kw):
        p=patch.object(obj,name,*args,**kw);value=p.start();self.addCleanup(p.stop);return value
    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def ref(self,name):return dict(path=name,sha256=c.sha(self.root/name))
    def update_release(self):
        self.release_version=getattr(self,'release_version',0)+1;name=f'release-{self.release_version}.json'
        self.write(name,self.release);self.release_path=self.root/name;self.release_sha=c.sha(self.release_path)
    def factory(self,*args,**kw):
        self.assertEqual(kw['scope'],'MAIN');self.assertTrue(not any(Path(kw['output_dir']).iterdir()))
        if kw['arm']=='START':
            self.assertIsNone(kw['commit_store']);self.assertIsNone(kw['expected_identity']);self.assertIsNone(kw['expected_commit_sha256'])
        else:
            key=f"{kw['start_seed']}/{kw['arm']}";self.assertIs(kw['commit_store'],self.stores[key])
            self.assertEqual(kw['expected_commit_sha256'],self.terminal.audit_spec['trusted_commits'][key][kw['step']-1]['sha256'])
        h=FakeHandle(self,kw['authority'],Path(kw['output_dir']));self.handles.append(h);return h
    def execute(self,start=11031,arm='START',step=0):return c.execute(self.release_path,self.release_sha,start,arm,step)
    def primary_fixture(self):
        self.release['endpoint']=dict(start_seed=11031,arm='T0',step=8)
        for start in (11031,11032):
            slot=c._slot(start,'START',0);slot.mkdir(parents=True)
            self.write(str((slot/'RESULT.json').relative_to(self.root)),{'CPU_SYNTHETIC_PREVIOUS':True})
            self.release['completed_endpoints'][f'{start}/START/0']=self.ref(str((slot/'RESULT.json').relative_to(self.root)))
        self.update_release();self.patch(c,'_prior',return_value=(ta.Observed(self.root),[]))

    def test_frozen14_endpoint_population_has_13760_draws(self):
        values=c.allocations();self.assertEqual(len(values),14);self.assertEqual(sum(values.values()),13760)
        self.assertEqual(list(values.values()).count(1952),6);self.assertEqual(list(values.values()).count(256),8)
        self.assertEqual(values,evaluation.endpoint_allocations('MAIN'))

    def test_start_endpoint_actual1952_bridge_draws_raw_and_zero_update_receipt(self):
        result=self.execute();receipt=c.read(c.bound(result['receipt']));binding=c.read(c.bound(result['endpoint_binding']))
        self.assertEqual(result['responses'],1952);self.assertEqual(len(self.attempts),1952);self.assertIsNone(binding['checkpoint'])
        self.assertTrue(self.handles[0].closed);self.assertEqual(receipt['extra_forward_sequences'],0)
        selected=[r for r in self.manifest if (r['start_seed'],r['arm'],r['step'])==(11031,'START',0)]
        raw=training_transaction.rows(c.bound(receipt['raw']));self.assertEqual([r['request_id'] for r in raw],[r['request_id'] for r in selected])
        for row,manifest,reference in zip(raw,selected,receipt['complete_files']):
            complete=c.read(c.bound(reference));self.assertTrue(all(row[k]==v for k,v in complete.items()))
            self.assertEqual(evaluation._capture(self.root,c.bound(result['receipt']).parent,reference,manifest,binding['endpoint'],
                self.freeze_sha,dict(prompt='public unchanged',image_file='image.png',assistant_prefill='{'),
                dict(png_sha256=c.sha(self.root/'image.png'))),complete)
        with self.assertRaisesRegex(PermissionError,'never retry'):self.execute()
        # Validate the next fresh process against the actual previous 1,952
        # durable captures. Native scorer behavior is tested in evaluation.py;
        # this fixture isolates the controller's immutable evidence traversal.
        self.release['completed_endpoints']['11031/START/0']=self.ref(str((c._slot(11031,'START',0)/'RESULT.json').relative_to(self.root)))
        self.release['endpoint']=dict(start_seed=11032,arm='START',step=0);self.process['pid']+=1;self.update_release()
        with patch.object(evaluation,'analyze_completed_endpoint',return_value={'summary':{'scope':'MAIN','metrics':{'assigned':1952}}}) as analyze:
            ctx=c.preflight(self.release_path,self.release_sha,11032,'START',0)
        analyze.assert_called_once();self.assertGreaterEqual(len(ctx['prior'].files),1952*3)
        begin=next((c._slot(11031,'START',0)/'endpoint').glob('*.BEGIN.json'));begin.write_text('{}')
        with self.assertRaisesRegex(PermissionError,'Terminal evidence changed'):ctx['prior'].stable()

    def test_trained_primary_endpoint_has_exact256_and_original_checkpoint_arguments(self):
        self.primary_fixture();result=self.execute(11031,'T0',8);binding=c.read(c.bound(result['endpoint_binding']))
        self.assertEqual(result['responses'],256);self.assertEqual(len(self.attempts),256)
        self.assertEqual(binding['checkpoint']['commit'],self.terminal.audit_spec['trusted_commits']['11031/T0'][7])
        self.assertTrue(self.handles[0].closed)

    def test_early_or_unknown_terminal_blocks_before_claim_and_model(self):
        self.boot.side_effect=PermissionError('not all paths terminal')
        with self.assertRaisesRegex(PermissionError,'not all paths terminal'):self.execute()
        self.assertFalse((self.root/'main_control').exists());self.factory_mock.assert_not_called()

    def test_early_main_or_unregistered_checkpoint_rejected(self):
        for args in ((11031,'T0',2),(11031,'START',8),(11031,'T1',0),(11031,'T0',23),(999,'START',0)):
            with self.assertRaisesRegex(PermissionError,'fourteen MAIN'):self.execute(*args)
        self.factory_mock.assert_not_called()

    def test_cannot_select_later_endpoint_before_fixed_prefix(self):
        self.release['endpoint']=dict(start_seed=11032,arm='START',step=0);self.update_release()
        with self.assertRaisesRegex(PermissionError,'preceding endpoint prefix'):self.execute(11032)
        self.factory_mock.assert_not_called()

    def test_wrong_release_endpoint_and_wrong_storage_scope_rejected(self):
        self.release['endpoint']['start_seed']=11032
        with self.assertRaisesRegex(PermissionError,'different endpoint'):self.execute()
        self.release['endpoint']['start_seed']=11031;self.budget.return_value={'scope':'TRAINED_DEV_ONLY'}
        with self.assertRaisesRegex(PermissionError,'MAIN storage scope'):self.execute()
        self.factory_mock.assert_not_called()

    def test_unknown_draw_preserved_and_all_later_main_blocked(self):
        self.fail_at=3
        with self.assertRaisesRegex(RuntimeError,'interrupted MAIN'):self.execute()
        slot=c._slot(11031,'START',0)
        self.assertEqual(len(list((slot/'endpoint').glob('*.BEGIN.json'))),3)
        self.assertEqual(len(list((slot/'endpoint').glob('*.COMPLETE.json'))),2)
        self.assertTrue((slot/'HOLD.json').exists());self.assertFalse((slot/'RESULT.json').exists())
        with self.assertRaises(PermissionError):self.execute()
        self.assertEqual(len(self.attempts),3)

    def test_global_hold_or_lock_prevents_model(self):
        with c.Lease():
            with self.assertRaises(BlockingIOError):self.execute()
        self.write('training_control/HOLD.json',{})
        with self.assertRaisesRegex(PermissionError,'Global HOLD'):self.execute()
        self.factory_mock.assert_not_called()

    def test_terminal_state_changes_at_boundary_prevent_model(self):
        self.terminal.stable.side_effect=[None,PermissionError('committed evidence changed')]
        with self.assertRaisesRegex(PermissionError,'committed evidence changed'):self.execute()
        self.factory_mock.assert_not_called();self.assertTrue((c._slot(11031,'START',0)/'HOLD.json').exists())

    def test_storage_failure_retains_claim_without_model(self):
        self.storage.side_effect=PermissionError('insufficient quota')
        with self.assertRaisesRegex(PermissionError,'insufficient quota'):self.execute()
        self.assertTrue((c._slot(11031,'START',0)/'CLAIM.json').exists());self.factory_mock.assert_not_called()

    def test_scope_and_draw_order_cannot_change(self):
        def factory(*args,**kw):
            h=self.factory(*args,**kw);a=h.authority;x=a.ctx
            with self.assertRaisesRegex(PermissionError,'scope differs'):
                a.verify(root=c.ROOT,selection=dict(x['selection'],scope='DEV'),endpoint=x['endpoint'],
                    design_freeze_sha256=c.FREEZE_SHA,execution_sources_sha256=x['sources_sha256'],
                    request_ids_sha256=backend.object_sha([r['request_id'] for r in x['requests']]),request_id=None)
            with self.assertRaisesRegex(PermissionError,'out-of-order'):h.auth(x['requests'][1]['request_id'])
            self.fail_at=1;return h
        self.factory_mock.side_effect=factory
        with self.assertRaises(RuntimeError):self.execute()

    def test_terminal_receipt_mutation_blocks_next_draw(self):
        def hook(h):
            if h.attempted==1:self.write('terminal.json',{'changed':True})
        self.hook=hook
        with self.assertRaisesRegex(PermissionError,'SHA mismatch'):self.execute()
        self.assertEqual(len(self.attempts),1)

    def test_new_nested_control_hold_stops_next_draw(self):
        def hook(h):
            if h.attempted==1:self.write('continuation_control/start-11031/T0/chunk-017-024/HOLD.json',{})
        self.hook=hook
        with self.assertRaisesRegex(PermissionError,'Global failed/unknown control'):self.execute()
        self.assertEqual(len(self.attempts),1)

    def test_new_nested_main_error_stops_next_draw(self):
        def hook(h):
            if h.attempted==1:(h.output/'kernel.ERROR.json').write_text('{}')
        self.hook=hook
        with self.assertRaisesRegex(PermissionError,'Global failed/unknown control'):self.execute()
        self.assertEqual(len(self.attempts),1)

    def test_extra_unreferenced_capture_prevents_complete_receipt(self):
        self.primary_fixture()
        def hook(h):
            if h.attempted==256:(h.output/'extra.BEGIN.json').write_text('{}')
        self.hook=hook
        with self.assertRaisesRegex(PermissionError,'Extra/missing/unresolved MAIN'):self.execute(11031,'T0',8)
        self.assertFalse((c._slot(11031,'T0',8)/'endpoint/RECEIPT.json').exists())

    def test_no_model_preflight(self):
        result=c.preflight(self.release_path,self.release_sha,11031,'START',0)
        self.assertEqual(result['count'],1952);self.factory_mock.assert_not_called();self.assertFalse((self.root/'main_control').exists())

    def test_real_bootstrap_rejects_hash_before_terminal_source_load(self):
        with patch.object(c.importlib.util,'spec_from_file_location') as load:
            with self.assertRaisesRegex(PermissionError,'Untrusted MAIN release'):self.real_bootstrap(self.release_path,'0'*64)
            load.assert_not_called()


class StandaloneBootstrapTests(unittest.TestCase):
    """Actual standard-library release loading, with a synthetic authority."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.external=self.root/'main_code';self.external.mkdir();self.original=self.root/'train_code';self.original.mkdir()
        source=self.external/'main_control.py';source.write_bytes((PACKAGE/'execution/main_control.py').read_bytes())
        name='main_bootstrap_fixture_'+str(id(self));spec=importlib.util.spec_from_file_location(name,source)
        self.m=importlib.util.module_from_spec(spec);sys.modules[name]=self.m;spec.loader.exec_module(self.m)
        self.addCleanup(lambda:sys.modules.pop(name,None));self.m.ROOT=self.root
        (self.original/'fixture.py').write_text('# original immutable fixture\n')
        self.original_value=dict(schema='sr-satdose-first-two-release-v1',status='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY',
            experiment_id=self.m.EXPERIMENT,evidence_root=str(self.root),release_root=str(self.original),sources=self.m.inventory(self.original))
        self.write('original.json',self.original_value)
        authority=self.external/'terminal_authority.py'
        authority.write_text('import json\nfrom pathlib import Path\nfrom types import SimpleNamespace\n'
            'def verify(spec):\n    root=Path(spec["root"])\n    (root/"AUTHORITY_EXECUTED").write_text("fixture")\n'
            '    original=json.loads((root/spec["original_train_release"]["path"]).read_text())\n'
            '    return SimpleNamespace(root=root,original=original,package=Path(original["release_root"]),modules={})\n')
        authority_ref=dict(path=str(authority),sha256=self.m.sha(authority))
        inventory=self.m.inventory(self.external)
        terminal=dict(root=str(self.root),authority=authority_ref,original_train_release=self.ref('original.json'),
                      external_root=str(self.external),external_sources=inventory)
        self.write('terminal_spec.json',terminal)
        release=dict(schema='sr-satdose-main-release-v1',status='AUDITED_FOR_FIXED_MAIN_ONLY',permission='FIXED_MAIN_ONLY_NO_TRAINING',
            experiment_id=self.m.EXPERIMENT,evidence_root=str(self.root),design_freeze_sha256=self.m.FREEZE_SHA,config_sha256=self.m.CONFIG_SHA,
            external_root=str(self.external),external_sources=inventory,original_train_release=self.ref('original.json'),
            terminal_authority=authority_ref,terminal_spec=self.ref('terminal_spec.json'))
        self.write('release.json',release);self.path=self.root/'release.json';self.digest=self.m.sha(self.path)
        self.addCleanup(lambda:sys.modules.pop('satdose_terminal_authority_'+authority_ref['sha256'][:16],None))
        self.m.first_two=SimpleNamespace(ROOT=self.root);self.m.storage=SimpleNamespace(ROOT=self.root,FREEZE_SHA=self.m.FREEZE_SHA)
    def write(self,name,value):(self.root/name).write_text(json.dumps(value))
    def ref(self,name):return dict(path=name,sha256=self.m.sha(self.root/name))

    def test_actual_source_inventories_verified_before_authority_execution(self):
        with patch.object(self.m,'_configure') as configure:
            release,terminal=self.m.bootstrap(self.path,self.digest)
        configure.assert_called_once();self.assertTrue((self.root/'AUTHORITY_EXECUTED').is_file())
        self.assertEqual(terminal.original,self.original_value);self.assertTrue(sys.dont_write_bytecode)
        with self.assertRaisesRegex(PermissionError,'fresh MAIN process'):self.m.bootstrap(self.path,self.digest)

    def test_changed_original_code_rejected_without_loading_external_authority(self):
        (self.original/'fixture.py').write_text('# changed\n')
        with self.assertRaisesRegex(PermissionError,'Original immutable TRAIN inventory'):
            self.m.bootstrap(self.path,self.digest)
        self.assertFalse((self.root/'AUTHORITY_EXECUTED').exists())

    def test_extra_external_file_rejected_without_authority_execution(self):
        (self.external/'unexpected.py').write_text('# extra\n')
        with self.assertRaisesRegex(PermissionError,'External MAIN inventory'):
            self.m.bootstrap(self.path,self.digest)
        self.assertFalse((self.root/'AUTHORITY_EXECUTED').exists())


if __name__=='__main__':unittest.main()
