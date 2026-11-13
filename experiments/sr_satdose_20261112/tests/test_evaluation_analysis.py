"""Synthetic CPU captures only; original pinned scorers, no GPU/model/real outputs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from analysis import evaluation as e

REPO=PACKAGE.parents[1]
LEGACY=Path(os.environ.get('SATDOSE_LEGACY_ROOT',str(REPO/'.worktrees/sr-ord-train-20261103/sr_ord_train')))
NATIVE=Path(os.environ.get('SATDOSE_NATIVE_SRC',str(REPO.parent/'dissertation for NTU/sr-f6/src')))


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def lines(path,values):path.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in values))
def ref(root,path):return dict(path=str(path.relative_to(root)),sha256=e.core.file_sha(path))


@unittest.skipUnless((NATIVE/'sr_f6/measurement.py').is_file() and (LEGACY/'score_eval.py').is_file(),'Pinned CPU scorer source required')
class EvaluationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.base=Path(cls.temp.name).resolve()
        # Isolate native import roots across test suites; production rejects shadowing.
        namespaces={'sr_f6','sr_f5','srf5_ref','srf6_ref','sr_f4','sr_f3_ce','order_probe','analyze'}
        previous={k:v for k,v in sys.modules.items() if k.split('.')[0] in namespaces}
        for k in previous:del sys.modules[k]
        def cleanup():
            for k in list(sys.modules):
                if k.split('.')[0] in namespaces:del sys.modules[k]
            sys.modules.update(previous)
        cls.addClassCleanup(cleanup)
        native_pins=e.core.read_json(e.calibration.PACKAGE/'config/source_bindings.json')['native_scorer_pins']
        legacy_names=('score_eval.py','analyze.py','vendor/probe_ref/order_probe/__init__.py',
                      'vendor/probe_ref/order_probe/common.py','vendor/probe_ref/order_probe/inputs.py',
                      'vendor/probe_ref/order_probe/measurement.py')
        lock=dict(native_src=str(NATIVE),legacy_root=str(LEGACY),native_files=native_pins,
                  legacy_files={n:e.core.file_sha(LEGACY/n) for n in legacy_names})
        dump(cls.base/'source_lock.json',lock)
        scorer,seed_for,_=e.load_scorers(cls.base/'source_lock.json',e.core.file_sha(cls.base/'source_lock.json'))
        cls.scorer=scorer;cls.seed_for=staticmethod(seed_for)
        cls.cfg=e.core.read_json(e.calibration.PACKAGE/'config/experiment.json')
        dump(cls.base/'config.json',cls.cfg)
        (cls.base/'fixture.png').write_bytes(b'SYNTHETIC CPU IMAGE BYTES; NOT A RENDER OR GPU INPUT')
        images=[dict(image_file='fixture.png',png_sha256=e.core.file_sha(cls.base/'fixture.png'))]
        cases={};inputs={};cls.manifests={}
        for scope in ('DEV','MAIN'):
            population=[]
            for (start,arm,step),count in e.endpoint_allocations(scope).items():
                for panel,(nroot,views,draws) in e._layout(scope,arm,step).items():
                    for i in range(nroot):
                        root=f'CPU_FIXTURE_{scope}_{panel}_{i}'
                        if panel=='retention8':
                            families=(['BINDING']*3+['IMPACT']*3+['SIGNED']*2+['INTERVAL']*2+['K_DIRECTION']*2 if scope=='DEV'
                                      else [f for f in ('BINDING','IMPACT','TIES','SIGNED','INTERVAL','K_DIRECTION') for _ in range(4)])
                            structure=families[i]
                        else:structure=e.core.STRUCTURES[i%4]
                        for view in views:
                            names=list('ABCDEFGHIJ' if panel=='prefix10' else 'ABCDEFGH')
                            alpha=[80,80]+list(range(20,20+len(names)-2));beta=[11,19]+list(range(30,30+len(names)-2))
                            selected=['B'] if panel=='authority8' and view=='list' else ['A']
                            world=dict(categories=names,series=dict(Alpha=alpha,Beta=beta),domain=[10,90])
                            query=dict(family='TOPK',selection_series='Alpha',direction='largest',k=1,value='Beta',aggregate='sum')
                            task=dict(root_id=root,structure=structure,world=world,query=query,r=1,
                                strictly_better_indices=[],boundary_indices=[0,1],authority='list' if selected==['B'] else 'chart',
                                gold=dict(selected=selected,operands=[19 if selected==['B'] else 11],answer=19 if selected==['B'] else 11))
                            audit=dict(native_J_applicable=panel!='prefix10' and not(panel=='authority8' and view=='list'),
                                list_order=([names[1],names[0]]+names[2:] if view in ('conflict','list') else names),
                                list_conflict=view in ('conflict','list'))
                            if panel=='prefix10':audit['prefix_induced_selected']=['B'] if view=='p_revtie' else ['A']
                            inp=dict(image_file='fixture.png',prompt=f'CPU SYNTHETIC {root}/{view}',
                                     assistant_prefill='{"evidence":[],' if panel=='prefix10' else '{')
                            iid=e.core.object_sha(inp);inp=dict(input_id=iid,**inp);inputs[iid]=inp
                            case=dict(input_id=iid,task=task,audit=audit);cid=e.core.object_sha(case)
                            cases[cid]=dict(case_id=cid,**case)
                            for draw in range(draws):
                                rid=f'SAT1/{scope}/{start}/{arm}/s{step:02d}/{panel}/{root}/{view}/d{draw:02d}'
                                population.append(dict(request_id=rid,split=scope,start_seed=start,arm=arm,step=step,
                                    panel=panel,view=view,root_id=root,structure=structure,draw=draw,
                                    seed=seed_for(2026110301,'eval','b0',panel,root,draw),case_id=cid,input_id=iid))
            cls.manifests[scope]=population;lines(cls.base/(scope.lower()+'_manifest.jsonl'),population)
        cls.cases=cases;cls.inputs=inputs
        lines(cls.base/'private_cases.jsonl',list(cases.values()));lines(cls.base/'inputs.jsonl',list(inputs.values()))
        lines(cls.base/'images.jsonl',images)
        frozen=dict(schema='sr-satdose-design-freeze-v1',experiment_id=e.EXPERIMENT,
            config_sha256=e.core.file_sha(cls.base/'config.json'),artifacts={})
        for name in ('source_lock','private_cases','inputs','images','dev_manifest','main_manifest'):
            path=cls.base/(name+('.json' if name=='source_lock' else '.jsonl'))
            frozen['artifacts'][name]=ref(cls.base,path)
        # Bound metadata only; test START endpoint never constructs a trained state.
        for name in ('train_manifest','training_groups'):
            (cls.base/(name+'.jsonl')).write_text('');frozen['artifacts'][name]=ref(cls.base,cls.base/(name+'.jsonl'))
        dump(cls.base/'DESIGN_FREEZE.json',frozen)

    def setUp(self):
        self.tempdir=tempfile.TemporaryDirectory();self.addCleanup(self.tempdir.cleanup)
        self.root=Path(self.tempdir.name).resolve()
        for path in self.base.iterdir():shutil.copyfile(path,self.root/path.name)
        self.receipt_path=self.root/'endpoint/RECEIPT.json';self.binding_path=self.root/'endpoint_binding.json'

    def capture(self,scope='DEV',transform=None,*,arm='START',step=0):
        output=self.receipt_path.parent;output.mkdir()
        self.selected=[r for r in self.manifests[scope] if r['start_seed']==11031 and r['arm']==arm and r['step']==step]
        source=self.cfg['start_bindings']['11031']
        endpoint=dict(experiment_id=e.EXPERIMENT,start_seed=11031,arm=arm,step=step,
                      policy_sha256=source['policy_sha256'] if arm=='START' else 'd'*64,reference_sha256=source['reference_sha256'])
        freeze=e.core.read_json(self.root/'DESIGN_FREEZE.json');freeze_sha=e.core.file_sha(self.root/'DESIGN_FREEZE.json')
        binding=dict(schema='sr-satdose-evaluation-endpoint-binding-v1',scope=scope,endpoint=endpoint,
            requests=len(self.selected),request_ids_sha256=e.core.object_sha([r['request_id'] for r in self.selected]),
            design_freeze_sha256=freeze_sha,config_sha256=freeze['config_sha256'],
            source_lock_sha256=freeze['artifacts']['source_lock']['sha256'],
            manifest_sha256=freeze['artifacts'][scope.lower()+'_manifest']['sha256'],execution_sources_sha256='e'*64,checkpoint=None)
        if arm!='START':
            state=self.root/'SYNTHETIC_STATE.pt';state.write_bytes(b'CPU FIXTURE BYTES; NOT A LOADABLE OR GPU CHECKPOINT')
            lock=e.core.read_json(self.root/'source_lock.json')
            identity=dict(experiment_id=e.EXPERIMENT,parent='b0',train_seed=11031,arm=arm,
                run_id=f'{e.EXPERIMENT}/b0-11031-{arm}',source_binding=source,config_sha256=freeze['config_sha256'],
                design_freeze_sha256=freeze_sha,manifest_sha256=freeze['artifacts']['train_manifest']['sha256'],
                plan_sha256=freeze['artifacts']['training_groups']['sha256'],execution_sources_sha256='9'*64,
                source_sha256=hashlib.sha256(json.dumps(lock['native_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
            commit=self.root/'SYNTHETIC_COMMIT.json'
            dump(commit,dict(step=step,identity=identity,policy_sha256=endpoint['policy_sha256'],
                reference_sha256=endpoint['reference_sha256'],state_sha256=e.core.file_sha(state)))
            binding['checkpoint']=dict(commit=ref(self.root,commit),state=ref(self.root,state))
        dump(self.binding_path,binding)
        raw=[];refs=[];events=[]
        for m in self.selected:
            case=self.cases[m['case_id']];t=case['task'];inp=self.inputs[m['input_id']]
            obj=dict(evidence=[dict(series=s,category=n,value=t['world']['series'][s][i]) for s in ('Alpha','Beta')
                               for i,n in enumerate(t['world']['categories'])],
                     selected=t['gold']['selected'],bindings=[dict(category=n,operand=v) for n,v in zip(t['gold']['selected'],t['gold']['operands'])],
                     aggregate='sum',answer=t['gold']['answer'])
            pref=inp['assistant_prefill']
            if m['panel']=='prefix10':del obj['evidence'];text=pref+json.dumps(obj)[1:]
            else:text=json.dumps(obj)
            finish='balanced'
            if transform:text,finish=transform(m,text,finish)
            cap=dict(seed=m['seed'],full_text=text,generated_text=text[len(pref):],assistant_prefill=pref,
                tokens=[12,13],raw_tokens=[12,13],completion_token_count=2,sampler_logprobs=[-.2,-.3],
                sampler_logprob_source='actual_generate_raw_logits_selected_token',prompt_tensor_hash='a'*64,
                model_input=dict(prompt=inp['prompt'],image_file=inp['image_file'],image_sha256=e.core.file_sha(self.root/'fixture.png')),
                finish_reason=finish,generation_status='COMPLETE',technical_validation_errors=[])
            stem=hashlib.sha256(m['request_id'].encode()).hexdigest();path=output/(stem+'.CAPTURE.json');dump(path,cap)
            dump(output/(stem+'.BEGIN.json'),dict(request_id=m['request_id'],seed=m['seed'],endpoint=endpoint,
                design_freeze_sha256=freeze_sha,status='ATTEMPTED_UNKNOWN'))
            complete=dict(cap,request_id=m['request_id'],native_capture_path=str(path),native_capture_sha256=e.core.file_sha(path),
                policy_sha256=endpoint['policy_sha256'],reference_sha256=endpoint['reference_sha256'],design_freeze_sha256=freeze_sha)
            path=output/(stem+'.COMPLETE.json');dump(path,complete);refs.append(ref(self.root,path))
            row=dict(complete,**{k:v for k,v in m.items() if k!='request_id'},scope=scope,suffix_text=cap['generated_text']);raw.append(row)
            events.extend([dict(kind='completion_attempts',count=1,metadata=dict(group_seed=m['seed'],protocol='structured')),
                           dict(kind='generated_tokens',count=2,metadata=dict(group_seed=m['seed'],row_index=0))])
        lines(output/'RAW.jsonl',raw);lines(output/'accounting.jsonl',events)
        receipt=dict(schema='sr-satdose-evaluation-completion-v1',status='EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION',
            scope=scope,endpoint_binding_sha256=e.core.file_sha(self.binding_path),endpoint_before=endpoint,endpoint_after=endpoint,
            responses=len(raw),optimizer_created=False,scheduler_created=False,optimizer_steps=0,scheduler_steps=0,
            parameter_updates=0,extra_forward_sequences=0,raw=ref(self.root,output/'RAW.jsonl'),
            accounting=ref(self.root,output/'accounting.jsonl'),complete_files=refs)
        dump(self.receipt_path,receipt)
        return raw

    def analyze(self):
        return e.analyze_completed_endpoint(self.root/'config.json',self.root/'DESIGN_FREEZE.json',self.root,self.receipt_path,
            expected_receipt_sha256=e.core.file_sha(self.receipt_path),endpoint_binding_path=self.binding_path,
            expected_endpoint_binding_sha256=e.core.file_sha(self.binding_path))

    def receipt_change(self,fn):
        receipt=e.core.read_json(self.receipt_path);fn(receipt);dump(self.receipt_path,receipt)

    def test_real_native_dev_complete_allocation_and_risk_inputs(self):
        self.capture();result=self.analyze();summary=result['summary']
        self.assertEqual(len(result['rows']),108);self.assertEqual(summary['metrics']['rates']['S_hit'],1)
        self.assertEqual(summary['metrics']['native_J']['applicable'],108)
        self.assertEqual(summary['dev_risk_inputs']['conflict']['assigned'],32)
        self.assertEqual(summary['dev_risk_inputs']['aligned_no_list']['assigned'],64)
        self.assertFalse(summary['training_admission']);self.assertFalse(summary['main_terminal_authority_verified'])
        self.assertEqual(summary['permission'],e.PERMISSION)
        self.assertEqual([r['request_id'] for r in result['rows']],[r['request_id'] for r in self.selected])

    def test_main_full_real_native_na_and_eos_sensitivity_never_replace_strict(self):
        def transform(m,text,finish):
            if m['panel']=='prefix10':return text[:-1],('eos' if m['draw']==0 else 'length')
            return text,finish
        self.capture('MAIN',transform);result=self.analyze()
        self.assertEqual(len(result['rows']),1952)
        prefix=[r for r in result['rows'] if r['panel']=='prefix10']
        self.assertTrue(all(r['J_native'] is None and r['native_score'] is None for r in prefix))
        eos=next(r for r in prefix if r['finish_reason']=='eos');length=next(r for r in prefix if r['finish_reason']=='length')
        self.assertEqual(eos['S_hit'],0);self.assertEqual(eos['eos_single_brace_sensitivity']['metrics']['S_hit'],1)
        self.assertEqual(length['eos_single_brace_sensitivity']['metrics']['S_hit'],0)
        self.assertTrue(all(r['J_native'] is None for r in result['rows'] if r['panel']=='authority8' and r['view']=='list'))
        self.assertEqual(result['summary']['metrics']['native_J']['NA'],576+192)

    def test_trained_main_intermediate_has_only256_and_checks_checkpoint_bytes(self):
        self.capture('MAIN',arm='T1',step=8);result=self.analyze()
        self.assertEqual(len(result['rows']),256)
        self.assertEqual({(r['panel'],r['view']) for r in result['rows']},{('list8','conflict')})
        self.assertEqual(result['summary']['endpoint']['policy_sha256'],'d'*64)
        (self.root/'SYNTHETIC_STATE.pt').write_bytes(b'mutated checkpoint bytes')
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.analyze()

    def test_trained_checkpoint_cannot_substitute_reference_or_path_identity(self):
        self.capture('DEV',arm='T0',step=2)
        path=self.root/'SYNTHETIC_COMMIT.json';marker=e.core.read_json(path);marker['identity']['arm']='T1';dump(path,marker)
        binding=e.core.read_json(self.binding_path);binding['checkpoint']['commit']=ref(self.root,path);dump(self.binding_path,binding)
        self.receipt_change(lambda r:r.update(endpoint_binding_sha256=e.core.file_sha(self.binding_path)))
        with self.assertRaisesRegex(ValueError,'scientific identity'):self.analyze()

    def test_field_local_semantics_duplicate_labels_unrelated_schema_length(self):
        def transform(m,text,finish):
            if m['panel']!='list8':return text,finish
            obj=json.loads(text)
            if m['view']=='aligned':obj['selected']=['A','A']
            elif m['view']=='conflict':obj['extra']='native protocol invalid, selected still readable'
            else:finish='length'
            return json.dumps(obj),finish
        self.capture(transform=transform);rows=self.analyze()['rows']
        aligned=next(r for r in rows if r['view']=='aligned');conflict=next(r for r in rows if r['view']=='conflict');length=next(r for r in rows if r['view']=='no_list')
        self.assertEqual((aligned['S_known'],aligned['S_hit'],aligned['source_class']),(1,0,'inadmissible'))
        self.assertEqual((conflict['S_known'],conflict['S_hit'],conflict['J_native']),(1,1,0))
        self.assertEqual((length['A'],length['S_known'],length['B_known']),(1,0,0))

    def test_duplicate_json_keys_and_trailing_text_unknown_not_repaired(self):
        def transform(m,text,finish):
            return (text[:-1]+',"selected":["A"]}' if m['view']=='aligned' else text+' trailing'),finish
        self.capture(transform=transform);result=self.analyze()
        self.assertEqual(result['summary']['metrics']['counts']['S_known'],0)
        self.assertEqual(result['summary']['metrics']['assigned'],108)

    def test_bad_receipt_fails_before_raw_open(self):
        self.capture();self.receipt_change(lambda r:r.update(responses=107))
        raw=self.receipt_path.parent/'RAW.jsonl';raw.unlink()
        with self.assertRaisesRegex(ValueError,'endpoint/count'):self.analyze()

    def test_unknown_extra_attempt_and_hold_are_not_zeros(self):
        self.capture();unknown=self.receipt_path.parent/('0'*64+'.BEGIN.json');dump(unknown,{})
        with self.assertRaisesRegex(ValueError,'Extra/unresolved'):self.analyze()
        unknown.unlink();dump(self.receipt_path.parent/'HOLD.json',{})
        with self.assertRaisesRegex(ValueError,'HOLD/ERROR'):self.analyze()

    def test_capture_hash_and_begin_endpoint_must_match(self):
        self.capture();receipt=e.core.read_json(self.receipt_path);complete=e.core.read_json(self.root/receipt['complete_files'][0]['path'])
        capture=Path(complete['native_capture_path']);capture.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.analyze()

    def test_begin_seed_tamper_rejected_even_when_raw_unchanged(self):
        self.capture();path=next(self.receipt_path.parent.glob('*.BEGIN.json'));data=e.core.read_json(path);data['seed']+=1;dump(path,data)
        with self.assertRaisesRegex(ValueError,'BEGIN identity'):self.analyze()

    def test_raw_missing_duplicate_reorder_and_changed_native_fields(self):
        original=self.capture();path=self.receipt_path.parent/'RAW.jsonl'
        for kind in ('missing','duplicate','reorder','tokens'):
            values=copy.deepcopy(original)
            if kind=='missing':values.pop()
            elif kind=='duplicate':values[-1]=values[0]
            elif kind=='reorder':values.reverse()
            else:values[0]['tokens']=[99]
            lines(path,values);self.receipt_change(lambda r:r.update(raw=ref(self.root,path)))
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'coverage/order|COMPLETE fields'):self.analyze()

    def test_accounting_extra_forward_or_wrong_token_count_rejected(self):
        self.capture();path=self.receipt_path.parent/'accounting.jsonl';events=list(e.core.read_jsonl(path))
        events[1]['count']=3;lines(path,events);self.receipt_change(lambda r:r.update(accounting=ref(self.root,path)))
        with self.assertRaisesRegex(ValueError,'accounting differs'):self.analyze()

    def test_wrong_scope_population_and_endpoint_order_rejected(self):
        for scope in ('DEV','MAIN'):
            original=self.manifests[scope];self.assertEqual(len(e.validate_manifest(original,scope)),14)
            with self.assertRaisesRegex(ValueError,'Incomplete/expanded'):e.validate_manifest(original[:-1],scope)
        self.capture();binding=e.core.read_json(self.binding_path)
        binding['request_ids_sha256']=e.core.object_sha(sorted(r['request_id'] for r in self.selected));dump(self.binding_path,binding)
        self.receipt_change(lambda r:r.update(endpoint_binding_sha256=e.core.file_sha(self.binding_path)))
        with self.assertRaisesRegex(ValueError,'allocation/order'):self.analyze()

    def test_real_legacy_scorer_rejects_wrong_na_flags_before_native_call(self):
        for panel,view in (('prefix10','p_stable'),('authority8','list'),('retention8','v0')):
            manifest=next(r for r in self.manifests['MAIN'] if r['panel']==panel and r['view']==view)
            case=copy.deepcopy(self.cases[manifest['case_id']]);case['audit']['native_J_applicable']=panel!='retention8'
            inp=self.inputs[manifest['input_id']]
            raw=dict(manifest,full_text=inp['assistant_prefill']+'"selected":[]}',assistant_prefill=inp['assistant_prefill'],
                     finish_reason='balanced',generation_status='COMPLETE',technical_validation_errors=[])
            with self.subTest(panel=panel),patch.object(self.scorer,'native_score',side_effect=AssertionError('inapplicable J invoked')):
                with self.assertRaisesRegex(ValueError,'native J'):
                    self.scorer.score_one(manifest,raw,case,inp)

    def test_pinned_scorer_hash_and_shadowed_import_rejected(self):
        lock=e.core.read_json(self.root/'source_lock.json');lock['legacy_files']['score_eval.py']='0'*64;dump(self.root/'source_lock.json',lock)
        with self.assertRaisesRegex(ValueError,'SHA differs'):e.load_scorers(self.root/'source_lock.json',e.core.file_sha(self.root/'source_lock.json'))
        lock['legacy_files']['score_eval.py']=e.core.file_sha(LEGACY/'score_eval.py');dump(self.root/'source_lock.json',lock)
        from types import SimpleNamespace
        with patch.dict(sys.modules,{'analyze':SimpleNamespace(__file__='/wrong/analyze.py')}):
            with self.assertRaisesRegex(ValueError,'Shadowed legacy'):e.load_scorers(self.root/'source_lock.json',e.core.file_sha(self.root/'source_lock.json'))

    def test_changed_intermediate_root_cannot_preserve_count_and_evade_pairing(self):
        rows=copy.deepcopy(self.manifests['MAIN'])
        old=next(r['root_id'] for r in rows if r['arm']=='T0' and r['step']==8)
        for row in rows:
            if row['arm']=='T0' and row['step']==8 and row['root_id']==old:
                row['root_id']=old+'_SUBSTITUTED'
                row['request_id']=row['request_id'].replace(old,row['root_id'])
        with self.assertRaisesRegex(ValueError,'Paired endpoint'):e.validate_manifest(rows,'MAIN')

    def test_source_and_completion_sha_are_explicit_not_self_authenticating(self):
        self.capture()
        with self.assertRaisesRegex(ValueError,'SHA differs'):
            e.analyze_completed_endpoint(self.root/'config.json',self.root/'DESIGN_FREEZE.json',self.root,self.receipt_path,
                expected_receipt_sha256='0'*64,endpoint_binding_path=self.binding_path,
                expected_endpoint_binding_sha256=e.core.file_sha(self.binding_path))


if __name__=='__main__':unittest.main()
