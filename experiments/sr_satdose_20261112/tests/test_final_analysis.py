"""Complete synthetic MAIN captures scored by the original native CPU functions.

No actual GPU/model, checkpoint tensor validity, terminal permission or production
cost is established. Test checkpoint bytes and cost records are explicitly fake.
The integration fixture isolates the unchanged original scoring source inventory;
production validates its entire immutable TRAIN release, tested separately here.
The supplied primary bootstrap itself is unmodified and uses its fixed defaults.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
SOURCE=PACKAGE/'analysis/final_analysis.py'
spec=importlib.util.spec_from_file_location('_final_endpoint_fixture',Path(__file__).with_name('test_evaluation_analysis.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
e=f.e


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n')
def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
def ref(root,path):return dict(path=str(path.relative_to(root)),sha256=e.core.file_sha(path))
def costs():return {k:dict(accounting=[],measurements=[],resource_evidence=[]) for k in
                    ('DRY','PREFIX_CALIB','SHADOW','DEV','TRAIN','MAIN','INCIDENTS')}


TEMP=tempfile.TemporaryDirectory()
EXTERNAL=Path(TEMP.name).resolve()/'final_analysis.py';EXTERNAL.write_bytes(SOURCE.read_bytes())
c=types.ModuleType('_satdose_test_external_final');c.__file__=str(EXTERNAL)
exec(compile(EXTERNAL.read_bytes(),str(EXTERNAL),'exec'),c.__dict__)


@unittest.skipUnless((f.NATIVE/'sr_f6/measurement.py').is_file() and (f.LEGACY/'score_eval.py').is_file(),'Pinned native CPU scorer required')
class CompleteMainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f.EvaluationAnalysisTests.setUpClass();cls.addClassCleanup(f.EvaluationAnalysisTests.doClassCleanups)
        case=f.EvaluationAnalysisTests('test_real_native_dev_complete_allocation_and_risk_inputs');case.setUp();cls.addClassCleanup(case.doCleanups)
        cls.root=case.root;cls.fixture=case;cls.refs={};cls.scored={};cls.cfg=case.cfg
        frozen=e.core.read_json(cls.root/'DESIGN_FREEZE.json');freeze_sha=e.core.file_sha(cls.root/'DESIGN_FREEZE.json')
        for endpoint,count in c.ALLOCATION.items():
            start,arm,step=endpoint.split('/');start=int(start);step=int(step)
            selected=[r for r in case.manifests['MAIN'] if (r['start_seed'],r['arm'],r['step'])==(start,arm,step)]
            directory=cls.root/f'main/start-{start}/{arm}/step-{step:03d}/endpoint';directory.mkdir(parents=True)
            source=cls.cfg['start_bindings'][str(start)]
            ep=dict(experiment_id=e.EXPERIMENT,start_seed=start,arm=arm,step=step,
                    policy_sha256=source['policy_sha256'] if arm=='START' else 'd'*64,reference_sha256=source['reference_sha256'])
            binding=dict(schema='sr-satdose-evaluation-endpoint-binding-v1',scope='MAIN',endpoint=ep,
                requests=count,request_ids_sha256=e.core.object_sha([r['request_id'] for r in selected]),
                design_freeze_sha256=freeze_sha,config_sha256=frozen['config_sha256'],
                source_lock_sha256=frozen['artifacts']['source_lock']['sha256'],manifest_sha256=frozen['artifacts']['main_manifest']['sha256'],
                execution_sources_sha256='e'*64,checkpoint=None)
            if arm!='START':
                state=directory.parent/'CPU_SYNTHETIC_STATE.pt';state.write_bytes(b'CPU NOT A REAL CHECKPOINT')
                lock=e.core.read_json(cls.root/'source_lock.json')
                identity=dict(experiment_id=e.EXPERIMENT,parent='b0',train_seed=start,arm=arm,
                    run_id=f'{e.EXPERIMENT}/b0-{start}-{arm}',source_binding=source,config_sha256=frozen['config_sha256'],
                    design_freeze_sha256=freeze_sha,manifest_sha256=frozen['artifacts']['train_manifest']['sha256'],
                    plan_sha256=frozen['artifacts']['training_groups']['sha256'],execution_sources_sha256='9'*64,
                    source_sha256=hashlib.sha256(json.dumps(lock['native_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
                commit=directory.parent/'CPU_SYNTHETIC_COMMIT.json';dump(commit,dict(step=step,identity=identity,
                    policy_sha256=ep['policy_sha256'],reference_sha256=ep['reference_sha256'],state_sha256=e.core.file_sha(state)))
                binding['checkpoint']=dict(commit=ref(cls.root,commit),state=ref(cls.root,state))
            binding_path=directory.parent/'BINDING.json';dump(binding_path,binding)
            raw=[];complete_refs=[];events=[]
            for m in selected:
                task=case.cases[m['case_id']]['task'];inp=case.inputs[m['input_id']];names=task['world']['categories']
                obj=dict(evidence=[dict(series=s,category=n,value=task['world']['series'][s][i])
                                   for s in ('Alpha','Beta') for i,n in enumerate(names)],
                    selected=(['B'] if task['gold']['selected']==['A'] else ['A']) if arm=='T0' else task['gold']['selected'],
                    bindings=[dict(category=n,operand=v) for n,v in zip(task['gold']['selected'],task['gold']['operands'])],
                    aggregate='sum',answer=task['gold']['answer'])
                pref=inp['assistant_prefill'];text=json.dumps(obj);finish='balanced'
                if m['panel']=='prefix10':del obj['evidence'];text=pref+json.dumps(obj)[1:]
                if m['panel']=='prefix10' and m['view']=='p_stable' and m['root_id'].endswith('_0') and m['draw']==0:
                    text=text[:-1];finish='eos'  # Strict UNKNOWN, limited sensitivity repairs one brace.
                if m['panel']=='list8' and m['view']=='conflict' and m['root_id'].endswith('_0') and m['draw']==1 and arm=='T1':
                    finish='length'  # Observed UNKNOWN is retained in assigned denominator.
                cap=dict(seed=m['seed'],full_text=text,generated_text=text[len(pref):],assistant_prefill=pref,
                    tokens=[12,13],raw_tokens=[12,13],completion_token_count=2,sampler_logprobs=[-.2,-.3],
                    sampler_logprob_source='actual_generate_raw_logits_selected_token',prompt_tensor_hash='a'*64,
                    model_input=dict(prompt=inp['prompt'],image_file=inp['image_file'],image_sha256=e.core.file_sha(cls.root/'fixture.png')),
                    finish_reason=finish,generation_status='COMPLETE',technical_validation_errors=[])
                stem=hashlib.sha256(m['request_id'].encode()).hexdigest();p=directory/(stem+'.CAPTURE.json');dump(p,cap)
                dump(directory/(stem+'.BEGIN.json'),dict(request_id=m['request_id'],seed=m['seed'],endpoint=ep,
                    design_freeze_sha256=freeze_sha,status='ATTEMPTED_UNKNOWN'))
                complete=dict(cap,request_id=m['request_id'],native_capture_path=str(p),native_capture_sha256=e.core.file_sha(p),
                    policy_sha256=ep['policy_sha256'],reference_sha256=ep['reference_sha256'],design_freeze_sha256=freeze_sha)
                p=directory/(stem+'.COMPLETE.json');dump(p,complete);complete_refs.append(ref(cls.root,p))
                raw.append(dict(complete,**{k:v for k,v in m.items() if k!='request_id'},scope='MAIN',suffix_text=cap['generated_text']))
                events.extend([dict(kind='completion_attempts',count=1,metadata={'group_seed':m['seed']}),
                               dict(kind='generated_tokens',count=2,metadata={'group_seed':m['seed']})])
            lines(directory/'RAW.jsonl',raw);lines(directory/'accounting.jsonl',events)
            receipt=dict(schema='sr-satdose-evaluation-completion-v1',status='EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION',
                scope='MAIN',endpoint_binding_sha256=e.core.file_sha(binding_path),endpoint_before=ep,endpoint_after=ep,responses=count,
                optimizer_created=False,scheduler_created=False,optimizer_steps=0,scheduler_steps=0,parameter_updates=0,extra_forward_sequences=0,
                raw=ref(cls.root,directory/'RAW.jsonl'),accounting=ref(cls.root,directory/'accounting.jsonl'),complete_files=complete_refs)
            p=directory/'RECEIPT.json';dump(p,receipt);cls.refs[endpoint]=dict(receipt=ref(cls.root,p),binding=ref(cls.root,binding_path))
        # Other agents can add external files to this mutable checkout concurrently.
        # Isolate that unrelated inventory boundary; keep real source bytes, native
        # scorer imports and every endpoint artifact check in the integration test.
        names=('analysis/evaluation.py','analysis/calibration.py','supplied_package/tools/satdose_core.py',
               'supplied_package/tools/analyze_primary.py','supplied_package/config/source_bindings.json')
        inventory_patch=patch.object(c,'_inventory',side_effect=lambda package:{name:c.sha(package/name) for name in names})
        inventory_patch.start();cls.addClassCleanup(inventory_patch.stop)
        original=dict(schema='sr-satdose-first-two-release-v1',status='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY',
            experiment_id=e.EXPERIMENT,evidence_root=str(cls.root),release_root=str(PACKAGE),sources=c._inventory(PACKAGE),
            design_freeze_sha256=freeze_sha,config_sha256=frozen['config_sha256'])
        dump(cls.root/'CPU_ORIGINAL_RELEASE.json',original)
        cls.source_spec=dict(analyzer=dict(path=str(EXTERNAL),sha256=c.sha(EXTERNAL)),
            original_train_release=ref(cls.root,cls.root/'CPU_ORIGINAL_RELEASE.json'),
            primary_helper=dict(path=str(PACKAGE/'supplied_package/tools/analyze_primary.py'),
                                sha256=c.sha(PACKAGE/'supplied_package/tools/analyze_primary.py')))
        cls.options=dict(config_ref=ref(cls.root,cls.root/'config.json'),freeze_ref=ref(cls.root,cls.root/'DESIGN_FREEZE.json'),
                         endpoints=cls.refs,source_spec=cls.source_spec,cost_refs=costs())
        cls.result=c.analyze_complete(cls.root,**cls.options)
        for key in c.ALLOCATION:
            start,arm,step=key.split('/')
            cls.scored[key]=[r for r in cls.result['rows'] if (r['start_seed'],r['arm'],r['step'])==(int(start),arm,int(step))]
    def test_complete_real_native_scoring_and_exact_frozen_primary_bootstrap(self):
        result=self.result
        self.assertEqual((result['assigned'],len(result['rows']),len(result['primary_root_rows'])),(13760,13760,256))
        self.assertEqual(len(result['endpoint_summaries']),14)
        ev,primary,_,_=c._sources(c.Evidence(self.root),self.source_spec)
        self.assertIs(ev,e);self.assertEqual(result['primary'],primary.analyze(result['primary_root_rows']))
        self.assertEqual((result['primary']['B'],result['primary']['bootstrap_seed']),(10000,2026111219))
        self.assertEqual(result['primary']['estimate'],255/256)
        self.assertTrue(result['primary']['both_starts_positive']);self.assertFalse(result['primary']['both_starts_absolute_gain_positive'])
        self.assertEqual(result['primary']['per_start_absolute_gain'],{'11031':-1/256,'11032':-1/256})
        self.assertFalse(result['followup_authorized']);self.assertFalse(result['main_terminal_authority_verified'])
    def test_unknown_full_denominator_prefix_sensitivity_and_native_NA_preserved(self):
        levels=self.result['prefix_bounds']['11031/START/0']
        self.assertEqual(levels['strict']['levels']['p_stable']['S_unknown'],1)
        self.assertEqual(levels['eos_single_brace_sensitivity']['levels']['p_stable']['S_unknown'],0)
        tables=self.result['panel_structure_tables']['11031/START/0']
        for row in tables:
            if row['panel']=='prefix10' or (row['panel'],row['view'])==('authority8','list'):
                self.assertEqual(row['J_native']['NA'],row['assigned']);self.assertEqual(row['J_native']['applicable'],0)
        self.assertEqual(sum(1-r['S_known'] for r in self.scored['11031/T1/24'] if r['panel']=='list8' and r['view']=='conflict'),1)
    def test_joint_and_marginals_preserve_unknown_and_all_assigned(self):
        for key,table in self.result['mechanism_joint_tables'].items():
            self.assertEqual(sum(r['count'] for r in table['joint']),c.ALLOCATION[key])
            self.assertEqual(sum(r['count'] for r in table['enumeration_marginal']),c.ALLOCATION[key])
            self.assertEqual(sum(r['count'] for r in table['selection_marginal']),c.ALLOCATION[key])
        states={r['enumeration']['status'] for r in self.result['mechanism_joint_tables']['11031/T1/24']['joint']}
        self.assertIn('INCOMPLETE_GENERATION',states);self.assertIn('PARTIAL_OR_DUPLICATE_ALPHA_SEQUENCE',states)
    def test_missing_endpoint_rejected_before_any_scoring_or_bootstrap(self):
        refs=copy.deepcopy(self.refs);refs.pop('11032/T1/24')
        with patch.object(e,'analyze_completed_endpoint',side_effect=AssertionError('must not score partial')):
            with self.assertRaisesRegex(PermissionError,'fourteen'):c.analyze_complete(self.root,**dict(self.options,endpoints=refs))
    def test_duplicate_endpoint_identity_rejected_before_scoring(self):
        refs=copy.deepcopy(self.refs);refs['11032/T1/24']=refs['11031/T1/24']
        with patch.object(e,'analyze_completed_endpoint',side_effect=AssertionError('must not score duplicate')):
            with self.assertRaisesRegex(PermissionError,'identity'):c.analyze_complete(self.root,**dict(self.options,endpoints=refs))
    def test_reordered_duplicate_missing_or_foreign_paired_cells_rejected(self):
        for failure in ('reordered','duplicate','missing','case','seed'):
            by=copy.deepcopy(self.scored);rows=by['11032/T1/24']
            selected=[i for i,r in enumerate(rows) if r['panel']=='list8' and r['view']=='conflict'];a,b=selected[:2]
            if failure=='reordered':rows[a],rows[b]=rows[b],rows[a]
            elif failure=='duplicate':rows[b]=copy.deepcopy(rows[a])
            elif failure=='missing':rows.pop(a)
            else:rows[a]['case_id' if failure=='case' else 'seed']='foreign' if failure=='case' else 9
            with self.subTest(failure=failure),self.assertRaises(PermissionError):c._primary_rows(by,24)
    def test_scored_order_and_binary_unknown_contract_rejected(self):
        rows=copy.deepcopy(self.scored['11031/START/0']);selected=[r for r in self.fixture.manifests['MAIN'] if r['start_seed']==11031 and r['arm']=='START']
        summary=self.result['endpoint_summaries']['11031/START/0'];rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaisesRegex(PermissionError,'reordered'):c._scored_rows(dict(rows=rows,summary=summary),selected,summary['endpoint'])
        rows=copy.deepcopy(self.scored['11031/START/0']);rows[0]['S_known']=0
        with self.assertRaisesRegex(PermissionError,'Unknown'):c._scored_rows(dict(rows=rows,summary=summary),selected,summary['endpoint'])
    def test_changed_native_capture_rejected_by_original_scorer(self):
        first=self.scored['11031/START/0'][0];p=Path(first['native_capture_path']);saved=p.read_bytes()
        try:
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA'):c.analyze_complete(self.root,**self.options)
        finally:p.write_bytes(saved)
    def test_source_identity_shadow_and_untrusted_primary_rejected(self):
        bad=copy.deepcopy(self.source_spec);bad['primary_helper']['sha256']='0'*64
        with self.assertRaisesRegex(PermissionError,'SHA'):c._sources(c.Evidence(self.root),bad)
        with patch.dict(sys.modules,{'analysis.shadow':types.SimpleNamespace(__file__='/tmp/foreign.py')}):
            with self.assertRaisesRegex(PermissionError,'Shadowed'):c._sources(c.Evidence(self.root),self.source_spec)
        self.assertNotIn('analysis.final_analysis',sys.modules)
    def test_cost_planned_and_actual_UNKNOWN_are_distinct(self):
        cost=self.result['costs'];self.assertEqual(cost['planned_total'],28936)
        self.assertEqual(cost['observed_MAIN_completion_attempts'],13760);self.assertEqual(cost['observed_MAIN_generated_tokens'],27520)
        self.assertIsNot(cost['planned_registered_generations'],c.PLANNED_GENERATIONS)
        self.assertIsNone(cost['actual_stage_costs']['TRAIN']['observed']['completion_attempts'])
        self.assertEqual(cost['actual_stage_costs']['MAIN']['metric_status']['peak_allocated_bytes'],'UNKNOWN')


class CostAndObservationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name).resolve()
        p=self.root/'main.jsonl';lines(p,[dict(kind='completion_attempts',count=13760),dict(kind='generated_tokens',count=27520)])
        self.main={'CPU':dict(accounting=ref(self.root,p))};self.spec=costs()
    def run_cost(self):return c._costs(c.Evidence(self.root),e.core,self.spec,self.main)
    def test_actual_counts_shadows_incidents_and_unknown_peak(self):
        p=self.root/'observed.json';dump(p,dict(CPU_FAKE=True,dry=1280,prefix=96,shadow=4,dev=1512,train=12288,failed=3))
        for stage,key,metric in (('DRY','dry','completion_attempts'),('PREFIX_CALIB','prefix','completion_attempts'),
                                ('SHADOW','shadow','optimizer_steps'),('DEV','dev','completion_attempts'),
                                ('TRAIN','train','completion_attempts'),('INCIDENTS','failed','completion_attempts')):
            self.spec[stage]['measurements'].append(dict(reference=ref(self.root,p),field_path=[key],metric=metric))
        result=self.run_cost();self.assertEqual(result['SHADOW']['observed']['optimizer_steps'],4)
        self.assertEqual(result['INCIDENTS']['observed']['completion_attempts'],3)
        self.assertIsNone(result['TRAIN']['observed']['peak_allocated_bytes'])
    def test_extra_unknown_main_attempt_cost_retained_without_extra_scored_samples(self):
        p=self.root/'failed-main.jsonl';lines(p,[dict(kind='completion_attempts',count=1)])
        self.spec['MAIN']['accounting']=[ref(self.root,p)]
        result=self.run_cost();self.assertEqual(result['MAIN']['observed']['completion_attempts'],13761)
        self.assertEqual(result['MAIN']['observed']['generated_tokens'],27520)
        self.assertEqual(sum(c.ALLOCATION.values()),13760)
    def test_real_full_inventory_detects_added_changed_or_symlink_source(self):
        directory=self.root/'package';directory.mkdir();p=directory/'original.py';p.write_text('x=1')
        baseline=c._inventory(directory);self.assertEqual(set(baseline),{'original.py'})
        p.write_text('x=2');self.assertNotEqual(c._inventory(directory),baseline)
        (directory/'external.py').write_text('x=3');self.assertNotEqual(set(c._inventory(directory)),set(baseline))
        (directory/'link.py').symlink_to(p)
        with self.assertRaises(PermissionError):c._inventory(directory)
    def test_duplicate_accounting_or_field_cannot_double_count(self):
        self.spec['TRAIN']['accounting']=[self.main['CPU']['accounting']]
        with self.assertRaisesRegex(PermissionError,'Duplicate'):self.run_cost()
        self.spec=costs();p=self.root/'observed.json';dump(p,{'steps':2})
        item=dict(reference=ref(self.root,p),field_path=['steps'],metric='optimizer_steps')
        self.spec['TRAIN']['measurements']=[item,copy.deepcopy(item)]
        with self.assertRaisesRegex(PermissionError,'Duplicate'):self.run_cost()
    def test_parent_job_child_step_overlap_and_fake_measured_compute_rejected(self):
        p=self.root/'resources.json';dump(p,{'job_seconds':60,'step_seconds':50})
        for key,level in (('job_seconds','job'),('step_seconds','step')):
            self.spec['TRAIN']['measurements'].append(dict(reference=ref(self.root,p),field_path=[key],metric='wall_seconds',
                resource_scope=dict(job_id='CPU_JOB',level=level,observation_id=key)))
        with self.assertRaisesRegex(PermissionError,'Overlapping'):self.run_cost()
        self.spec['TRAIN']['measurements']=self.spec['TRAIN']['measurements'][:1]
        self.spec['TRAIN']['measurements'][0]['metric']='measured_gpu_compute_seconds'
        with self.assertRaisesRegex(PermissionError,'cannot be inferred'):self.run_cost()
    def test_nonfinite_negative_or_missing_cost_is_not_zero(self):
        for value in (-1,'UNKNOWN',None,True):
            p=self.root/'invalid.json';dump(p,{'count':value});self.spec=costs()
            self.spec['DRY']['measurements']=[dict(reference=ref(self.root,p),field_path=['count'],metric='completion_attempts')]
            with self.subTest(value=value),self.assertRaises(PermissionError):self.run_cost()
    def test_native_diagnostic_accounting_kind_is_preserved(self):
        p=self.root/'shadow-accounting.jsonl';lines(p,[dict(kind='diagnostic_gradient_calls',count=2),
                                                     dict(kind='extra_forward_sequences',count=256)])
        self.spec['SHADOW']['accounting']=[ref(self.root,p)]
        result=self.run_cost();self.assertEqual(result['SHADOW']['observed']['diagnostic_gradient_calls'],2)
        self.assertEqual(result['SHADOW']['observed']['extra_forward_sequences'],256)
    def test_literal_sequence_preserves_partial_unknown_and_geometry_relation(self):
        case=dict(task={'world':{'categories':['A','B']}},audit={'list_order':['B','A']})
        raw=dict(finish_reason='eos',full_text=json.dumps({'evidence':[{'series':'Alpha','category':'B'},{'series':'Alpha','category':'A'}]}))
        result=c._enumeration(raw,case,e.core)
        self.assertTrue(result['equals_explicit_list_order']);self.assertFalse(result['equals_chart_category_order'])
        raw['finish_reason']='length';self.assertIsNone(c._enumeration(raw,case,e.core)['equals_chart_category_order'])
    def test_no_parameter_for_bootstrap_tuning_or_partial_mode(self):
        import inspect
        self.assertEqual(set(inspect.signature(c.analyze_complete).parameters),
                         {'root','config_ref','freeze_ref','endpoints','source_spec','cost_refs'})


if __name__=='__main__':unittest.main()
