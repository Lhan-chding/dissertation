"""Synthetic CPU fixtures only: no live raw reads, model calls, or GPU receipts."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'analysis'))
import calibration as c
sys.path.pop(0)
NATIVE=Path(os.environ.get('SATDOSE_NATIVE_SRC',str(HERE.parents[1].parent/'dissertation for NTU/sr-f6/src')))


def dump(path,obj):path.write_text(json.dumps(obj,allow_nan=False)+'\n')
def lines(path,rows):path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))


def fixture_complete(row):
    """Mirror the original bridge fields; scorer/control additions stay in raw."""
    fields=('request_id','seed','assistant_prefill','full_text','generated_text','tokens','raw_tokens',
            'completion_token_count','sampler_logprobs','sampler_logprob_source','model_input','finish_reason',
            'generation_status','technical_validation_errors','policy_sha256','reference_sha256','design_freeze_sha256')
    return {key:row[key] for key in fields}


def write_fixture_complete(output,rows):
    for row in rows:
        name=hashlib.sha256(row['request_id'].encode()).hexdigest()+'.COMPLETE.json'
        dump(output/name,fixture_complete(row))
    return [dict(path=path.name,sha256=c.core.file_sha(path)) for path in sorted(output.glob('*.COMPLETE.json'))]


@unittest.skipUnless((NATIVE/'sr_f6/measurement.py').is_file(),'Pinned native scorer source required')
class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The real scorer and legacy data adapter intentionally bind different
        # immutable source copies. Isolate only these native package namespaces
        # for this fixture; production load_scorer keeps its shadowing rejection.
        pins=c.core.read_json(c.PACKAGE/'config/source_bindings.json')['native_scorer_pins']
        native_packages={relative.split('/')[0] for relative in pins}
        before={name:module for name,module in sys.modules.items()
                if name.split('.')[0] in native_packages}
        def restore_native_modules():
            for name in list(sys.modules):
                if name.split('.')[0] in native_packages:
                    del sys.modules[name]
            sys.modules.update(before)
        cls.addClassCleanup(restore_native_modules)
        for name in before:
            del sys.modules[name]
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name).resolve();cls.base=root
        cls.addClassCleanup(cls.temp.cleanup)
        cls.scorer=staticmethod(c.load_scorer(NATIVE)[0])
        cfg=c.core.read_json(c.PACKAGE/'config/experiment.json');cfg['dry']['bootstrap_B']=200
        dump(root/'config.json',cfg)
        cases=[];inputs=[];groups=[];requests=[];raw={s:[] for s in c.core.STARTS}
        cells=[(st,dr,a) for st in c.core.STRUCTURES for dr in ('largest','smallest') for a in ('sum','mean')]
        for kind,reps in [('CONFLICT',4),('ALIGNED',1)]:
            for st,dr,agg in cells:
                m,r=map(int,st.split('_choose_'))
                for rep in range(reps):
                    rt=f'FIXTURE_ONLY_{kind}_{st}_{dr}_{agg}_{rep}';cid=rt+'case';iid=rt+'input'
                    names=list('ABCDEFGH');idx=list(range(2,2+m));good=[0,1]+idx[:r];bad=[0,1]+idx[::-1][:r]
                    alpha=([78,72]+[48]*m+[24]*(6-m)) if dr=='largest' else ([12,18]+[42]*m+[72]*(6-m))
                    beta=[12,18,24,30,36,42,48,54]
                    world=dict(categories=names,domain=[10,90],series=dict(Alpha=alpha,Beta=beta))
                    query=dict(family='TOPK',selection_series='Alpha',value='Beta',k=2+r,direction=dr,aggregate=agg)
                    answer=lambda selected:str(Fraction(sum(beta[i] for i in selected),len(selected) if agg=='mean' else 1))
                    task=dict(root_id=rt,structure=st,world=world,query=query,r=r,strictly_better_indices=[0,1],boundary_indices=idx,
                              gold=dict(selected=[names[i] for i in good],answer=answer(good)))
                    order=names.copy()
                    if kind=='CONFLICT':
                        for dest,source in zip(idx,idx[::-1]):order[dest]=names[source]
                    cases.append(dict(case_id=cid,input_id=iid,task=task,audit=dict(list_order=order,native_J_applicable=True)))
                    inputs.append(dict(input_id=iid,assistant_prefill='{'))
                    for start in c.core.STARTS:
                        gid=f'{start}/{rt}';ids=[gid+f'/{i}' for i in range(8)];seeds=list(range(8))
                        groups.append(dict(group_id=gid,start_seed=start,kind=kind,root_id=rt,structure=st,case_id=cid,
                                           request_ids=ids,draw_seeds=seeds))
                        for i in range(8):
                            man=dict(request_id=ids[i],split='DRY',root_id=rt,case_id=cid,input_id=iid,seed=i,draw=i,
                                     group_id=gid,start_seed=start,arm='START',step=0)
                            requests.append(man)
                            chosen=good if i<4 else bad
                            obj=dict(evidence=[dict(series=s,category=n,value=world['series'][s][j]) for s in ('Alpha','Beta') for j,n in enumerate(names)],
                                     selected=[names[j] for j in chosen],bindings=[dict(category=names[j],operand=beta[j]) for j in chosen],
                                     aggregate=agg,answer=answer(chosen))
                            text=json.dumps(obj)
                            ns=cls.scorer(text,world,query,finished=True,finish_reason='balanced')
                            row=dict(man,scope='DRY',assistant_prefill='{',full_text=text,suffix_text=text[1:],finish_reason='balanced',
                                     native_score=ns,generation_status='COMPLETE',technical_validation_errors=[],dry_optimizer_steps=0,dry_parameter_updates=0,
                                     policy_sha256=cfg['start_bindings'][str(start)]['policy_sha256'],
                                     reference_sha256=cfg['start_bindings'][str(start)]['reference_sha256'])
                            raw[start].append(row)
        preqs=[];prefix_raw={s:[] for s in c.core.STARTS}
        for i in range(16):
            rt=f'FIXTURE_PREFIX_{i}';st=c.core.STRUCTURES[i//4];names=list('ABCDEFGHIJ');gold=['A','B','C']
            task=dict(root_id=rt,structure=st,world=dict(categories=names),gold=dict(selected=gold,answer='3',operands=['1','1','1']))
            for view in c.VIEWS:
                cid=rt+view;iid=cid+'input';prefill='{"evidence": [],'
                cases.append(dict(case_id=cid,input_id=iid,task=task,audit=dict(native_J_applicable=False)))
                inputs.append(dict(input_id=iid,assistant_prefill=prefill))
                for start in c.core.STARTS:
                    rid=f'{start}/{cid}';selected=gold if view!='p_revtie' or i<8 else ['A','B','D']
                    text=prefill+'"selected": '+json.dumps(selected)+'}'
                    man=dict(request_id=rid,split='PREFIX_CALIB',root_id=rt,case_id=cid,input_id=iid,seed=i,draw=0,
                             panel='prefix10',view=view,start_seed=start,arm='START',step=0)
                    preqs.append(man)
                    prefix_raw[start].append(dict(man,scope='PREFIX_CALIB',assistant_prefill=prefill,full_text=text,
                        suffix_text=text[len(prefill):],finish_reason='balanced',native_score=None,native_J=None,native_J_applicable=False,
                        generation_status='COMPLETE',technical_validation_errors=[],dry_optimizer_steps=0,dry_parameter_updates=0,
                        policy_sha256=cfg['start_bindings'][str(start)]['policy_sha256'],reference_sha256=cfg['start_bindings'][str(start)]['reference_sha256']))
        for inp in inputs:
            inp.update(prompt='Synthetic CPU fixture only',image_file=inp['input_id']+'.png')
        image_rows=[dict(image_file=inp['image_file'],png_sha256='0'*64) for inp in inputs]
        input_lookup={inp['input_id']:inp for inp in inputs}
        for population in (raw,prefix_raw):
            for rows in population.values():
                for row in rows:
                    inp=input_lookup[row['input_id']]
                    row['model_input']=dict(prompt=inp['prompt'],image_file=inp['image_file'],image_sha256='0'*64)
                    row.update(generated_text=row['suffix_text'],tokens=[1,2],raw_tokens=[1,2],completion_token_count=2,
                               sampler_logprobs=[-0.1,-0.2],sampler_logprob_source='actual_generate_raw_logits_selected_token')
        for name,rows in [('private_cases',cases),('inputs',inputs),('images',image_rows),('calibration_groups',groups),('dry_manifest',requests),('prefix_calib_manifest',preqs)]:
            lines(root/(name+'.jsonl'),rows)
        freeze=dict(schema='sr-satdose-design-freeze-v1',experiment_id=cfg['experiment_id'],config_sha256=c.core.file_sha(root/'config.json'),
                    artifacts={name:dict(path=name+'.jsonl',sha256=c.core.file_sha(root/(name+'.jsonl'))) for name in
                               ('private_cases','inputs','images','calibration_groups','dry_manifest','prefix_calib_manifest')})
        dump(root/'source_lock.json',dict(fixture_only=True))
        lock_sha=c.core.file_sha(root/'source_lock.json')
        freeze['artifacts']['source_lock']=dict(path='source_lock.json',sha256=lock_sha)
        dump(root/'DESIGN_FREEZE.json',freeze);freeze_sha=c.core.file_sha(root/'DESIGN_FREEZE.json')
        for scope,data in [('DRY',raw),('PREFIX_CALIB',prefix_raw)]:
            for start,rows in data.items():
                output=root/f'{scope}-{start}';output.mkdir()
                for row in rows:row['design_freeze_sha256']=freeze_sha
                lines(output/'raw.jsonl',rows)
                complete_files=write_fixture_complete(output,rows)
                endpoint={k:cfg['start_bindings'][str(start)][k] for k in ('policy_sha256','reference_sha256')}
                receipt=dict(status='CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',scope=scope,start_seed=start,responses=len(rows),
                             design_freeze_sha256=freeze_sha,optimizer_steps=0,parameter_updates=0,optimizer_created=False,scheduler_created=False,
                             source_lock_sha256=lock_sha,complete_files=complete_files,
                             policy_before=endpoint,policy_after=endpoint,raw=dict(path='raw.jsonl',sha256=c.core.file_sha(output/'raw.jsonl')))
                dump(output/'CALIBRATION_COMPLETE.json',receipt)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()/'fixture';shutil.copytree(self.base,self.root)
    def tearDown(self):self.tmp.cleanup()
    def receipt(self,scope='DRY',start=11031):return self.root/f'{scope}-{start}/CALIBRATION_COMPLETE.json'
    def dry(self,start=11031):return c.analyze_dry(self.root/'config.json',self.root/'DESIGN_FREEZE.json',self.root,self.receipt(start=start),start,NATIVE)
    def prefix(self,clarification=None):
        return c.analyze_prefix(self.root/'config.json',self.root/'DESIGN_FREEZE.json',self.root,
                                self.receipt('PREFIX_CALIB'),11031,clarification=clarification)
    def clarification(self):
        """Synthetic provenance fixture, never evidence of real pre-call ordering."""
        identity=dict(experiment_id=c.core.read_json(self.root/'config.json')['experiment_id'],
                      config_sha256=c.core.file_sha(self.root/'config.json'),
                      design_freeze_sha256=c.core.file_sha(self.root/'DESIGN_FREEZE.json'))
        proof=self.root/'PRE_PREFIX_CONFIRMATION.json'
        dump(proof,dict(schema='sr-satdose-pre-prefix-confirmation-v1',**identity,
                        status='NO_PREFIX_ATTEMPTS_OBSERVED',prefix_attempts=0,prefix_allocations=[],
                        checked_at='2026-11-13T00:00:00+08:00'))
        path=self.root/'PREFIX_PROTOCOL_CLARIFICATION.json'
        dump(path,dict(schema='sr-satdose-prefix-protocol-clarification-v1',**identity,
                       decision_rule=c.PREFIX_RULE,decision_origin='agent_interpretation_after_user_requested_judgment',
                       user_explicit_option_approval=False,
                       pre_prefix_confirmation=dict(path=proof.name,sha256=c.core.file_sha(proof))))
        return path
    def mutate(self,fn,scope='DRY',*,update_complete=True):
        path=self.root/f'{scope}-11031/raw.jsonl';rows=list(c.core.read_jsonl(path));fn(rows)
        receipt=c.core.read_json(self.receipt(scope))
        if update_complete:
            for row in rows:row['generated_text']=row['suffix_text']
            receipt['complete_files']=write_fixture_complete(path.parent,rows)
        lines(path,rows)
        receipt['raw']['sha256']=c.core.file_sha(path);dump(self.receipt(scope),receipt)
    def test_complete_dry_native_replay_and_fixed_denominators(self):
        report=self.dry()
        self.assertEqual(report['status'],'DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED')
        self.assertEqual(report['permission'],'NO_TRAINING')
        self.assertEqual(report['metrics']['dose_mean'],4)
        self.assertEqual(report['metrics']['length_denominator_all_dry'],640)
        self.assertEqual(len(report['rescored_rows']),640)
        self.assertEqual(report['J1_c0_details'],[])
    def test_unfinished_start_refused_before_raw_open(self):
        path=self.receipt();receipt=c.core.read_json(path);receipt['responses']=639;dump(path,receipt)
        (path.parent/'raw.jsonl').unlink()
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.dry()
    def test_source_lock_mismatch_refused_before_raw_open(self):
        path=self.receipt();receipt=c.core.read_json(path);receipt['source_lock_sha256']='0'*64;dump(path,receipt)
        (path.parent/'raw.jsonl').unlink()
        with self.assertRaisesRegex(ValueError,'source lock'):self.dry()
    def test_changed_frozen_source_lock_refused(self):
        dump(self.root/'source_lock.json',dict(changed=True))
        with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):self.dry()
    def test_missing_or_duplicate_complete_inventory_refused(self):
        path=self.receipt();original=c.core.read_json(path)
        for mode in ('missing','duplicate'):
            with self.subTest(mode=mode):
                receipt=copy.deepcopy(original)
                if mode=='missing':receipt['complete_files'].pop()
                else:receipt['complete_files'][-1]=receipt['complete_files'][0]
                dump(path,receipt)
                with self.assertRaisesRegex(ValueError,'COMPLETE'):self.dry()
    def test_missing_complete_file_refused(self):
        path=self.receipt();receipt=c.core.read_json(path)
        (path.parent/receipt['complete_files'][0]['path']).unlink()
        with self.assertRaises(FileNotFoundError):self.dry()
    def test_tampered_complete_sha_refused(self):
        path=self.receipt();receipt=c.core.read_json(path);target=path.parent/receipt['complete_files'][0]['path']
        item=c.core.read_json(target);item['tokens']=[999];dump(target,item)
        with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):self.dry()
    def test_duplicate_complete_request_with_updated_sha_refused(self):
        path=self.receipt();receipt=c.core.read_json(path)
        left,right=receipt['complete_files'][:2]
        item=c.core.read_json(path.parent/right['path'])
        item['request_id']=c.core.read_json(path.parent/left['path'])['request_id']
        dump(path.parent/right['path'],item);right['sha256']=c.core.file_sha(path.parent/right['path']);dump(path,receipt)
        with self.assertRaisesRegex(ValueError,'COMPLETE request'):self.dry()
    def test_complete_path_cannot_escape_receipt_directory(self):
        path=self.receipt();receipt=c.core.read_json(path)
        receipt['complete_files'][0]['path']='../outside.COMPLETE.json';dump(path,receipt)
        with self.assertRaisesRegex(ValueError,'directly inside'):self.dry()
    def test_raw_original_token_replacement_refused_even_with_new_raw_sha(self):
        self.mutate(lambda rows:rows[0].update(tokens=[99,100],raw_tokens=[99,100]),update_complete=False)
        with self.assertRaisesRegex(ValueError,'original COMPLETE fields'):self.dry()
    def test_hold_or_error_markers_refused(self):
        for name in ('HOLD.json','ERROR.json','attempt.ERROR.json'):
            with self.subTest(name=name):
                path=self.receipt().parent/name;dump(path,dict(status='fixture'))
                with self.assertRaisesRegex(ValueError,'HOLD/ERROR'):self.dry()
                path.unlink()
    def test_missing_duplicate_and_technical_rows_fail(self):
        for mode in ('missing','duplicate','technical'):
            with self.subTest(mode=mode):
                original=(self.root/'DRY-11031/raw.jsonl').read_text()
                def alter(rows):
                    if mode=='missing':rows.pop()
                    elif mode=='duplicate':rows.append(rows[0])
                    else:rows[0]['technical_validation_errors']=['OOM']
                self.mutate(alter)
                with self.assertRaises(ValueError):self.dry()
                (self.root/'DRY-11031/raw.jsonl').write_text(original)
    def test_reward_and_checkpoint_tamper_fail(self):
        self.mutate(lambda rows:rows[0]['native_score'].update(J=0))
        with self.assertRaisesRegex(ValueError,'score replay'):self.dry()
    def test_model_input_tamper_fails_despite_unchanged_case_id(self):
        self.mutate(lambda rows:rows[0]['model_input'].update(prompt='Wrong question'))
        with self.assertRaisesRegex(ValueError,'frozen public input'):self.dry()
    def test_native_j1_wrong_selected_is_reported_without_redefining_reward(self):
        def alter(rows):
            row=rows[0];obj=json.loads(row['full_text']);obj['selected']=['A','B','D']
            text=json.dumps(obj);row['full_text']=text;row['suffix_text']=text[1:]
            cases={x['case_id']:x for x in c.core.read_jsonl(self.root/'private_cases.jsonl')}
            task=cases[row['case_id']]['task']
            row['native_score']=self.scorer(text,task['world'],task['query'],finished=True,finish_reason='balanced')
            self.assertEqual(row['native_score']['J'],1)
        self.mutate(alter);report=self.dry()
        self.assertEqual(len(report['J1_c0_details']),1)
        self.assertEqual(report['J1_c0_details'][0]['selected'],['A','B','D'])
        self.assertIn('full_text',report['J1_c0_details'][0])
    def test_lengths_use_all_640_including_aligned(self):
        def alter(rows):
            cases={x['case_id']:x for x in c.core.read_jsonl(self.root/'private_cases.jsonl')}
            for row in [r for r in rows if 'ALIGNED' in r['root_id']][:13]:
                row['finish_reason']='length';task=cases[row['case_id']]['task']
                row['native_score']=self.scorer(row['full_text'],task['world'],task['query'],finished=False,finish_reason='length')
        self.mutate(alter);report=self.dry()
        self.assertEqual(report['metrics']['truncation_rate_all_dry'],13/640)
        self.assertEqual(report['failures'],['TRUNCATION_RATE'])
    def test_prefix_readouts_all_denominators(self):
        result=self.prefix();strict=result['readouts']['strict']
        self.assertTrue(strict['passed']);self.assertEqual(strict['stable_minus_revtie_count'],8)
        self.assertEqual(strict['views']['p_revtie']['denominator'],16)
        self.assertEqual(result['permission'],'NO_TRAINING')
    def test_only_eos_single_brace_improves_sensitivity(self):
        def alter(rows):
            for row in [r for r in rows if r['view']=='p_stable'][:3]:
                row['finish_reason']='eos';row['full_text']=row['full_text'][:-1];row['suffix_text']=row['suffix_text'][:-1]
        self.mutate(alter,'PREFIX_CALIB');result=self.prefix()
        self.assertFalse(result['readouts']['strict']['passed']);self.assertTrue(result['readouts']['sensitivity']['passed'])
        self.assertEqual(result['readouts']['strict']['views']['p_stable']['known'],13)
        self.assertEqual(result['readouts']['sensitivity']['views']['p_stable']['known'],16)
    def test_prefix_exact_registered_thresholds_and_gap_failure(self):
        def set_text(row,selected):
            row['full_text']=row['assistant_prefill']+'"selected": '+json.dumps(selected)+'}'
            row['suffix_text']=row['full_text'][len(row['assistant_prefill']):]
        def boundary(rows):
            stable=[r for r in rows if r['view']=='p_stable']
            for row in stable[12:14]:set_text(row,['A','B','D'])
            for row in stable[14:]:set_text(row,None)
        self.mutate(boundary,'PREFIX_CALIB');report=self.prefix()['readouts']['strict']
        self.assertTrue(report['passed']);self.assertEqual(report['stable_minus_revtie_count'],4)
        self.assertEqual(report['views']['p_stable']['known'],14)
        self.assertEqual(report['views']['p_stable']['correct'],12)
        self.assertEqual(report['views']['p_stable']['bounds'],[12/16,14/16])
        def gap(rows):set_text([r for r in rows if r['view']=='p_revtie'][8],['A','B','C'])
        self.mutate(gap,'PREFIX_CALIB');report=self.prefix()
        self.assertEqual(report['status'],'PENDING_PROTOCOL_CLARIFICATION')
        self.assertFalse(report['candidate_gate_passed'])
        self.assertEqual(report['readouts']['sensitivity']['stable_minus_revtie_count'],3)
    def test_strict_pass_sensitive_gap_failure_stays_pending(self):
        def alter(rows):
            stable=[r for r in rows if r['view']=='p_stable']
            revtie=[r for r in rows if r['view']=='p_revtie']
            for row in stable[12:]:
                row['full_text']=row['assistant_prefill']+'"selected": ["A","B","D"]}'
                row['suffix_text']=row['full_text'][len(row['assistant_prefill']):]
            for row in revtie[8:10]:
                row['full_text']=row['assistant_prefill']+'"selected": ["A","B","C"]'
                row['suffix_text']=row['full_text'][len(row['assistant_prefill']):]
                row['finish_reason']='eos'
        self.mutate(alter,'PREFIX_CALIB');report=self.prefix()
        self.assertTrue(report['readouts']['strict']['passed'])
        self.assertFalse(report['readouts']['sensitivity']['passed'])
        self.assertEqual(report['readouts']['strict']['stable_minus_revtie_count'],4)
        self.assertEqual(report['readouts']['sensitivity']['stable_minus_revtie_count'],2)
        self.assertTrue(report['candidate_gate_passed'])
        self.assertEqual(report['status'],'PENDING_PROTOCOL_CLARIFICATION')
        self.assertEqual(report['permission'],'NO_TRAINING')
        clarified=self.prefix(self.clarification())
        self.assertEqual(clarified['status'],'PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED')
        self.assertEqual(clarified['qualifying_readouts'],['strict'])
        self.assertEqual(clarified['permission'],'NO_TRAINING')
        self.assertFalse(clarified['protocol_clarification']['user_explicit_option_approval'])
    def test_missing_or_wrong_identity_clarification_does_not_pass(self):
        self.assertEqual(self.prefix()['status'],'PENDING_PROTOCOL_CLARIFICATION')
        with self.assertRaises(FileNotFoundError):self.prefix(self.root/'missing.json')
        path=self.clarification();original=c.core.read_json(path)
        for field in ('config_sha256','design_freeze_sha256'):
            with self.subTest(field=field):
                changed=copy.deepcopy(original);changed[field]='0'*64;dump(path,changed)
                with self.assertRaisesRegex(ValueError,'identity mismatch'):self.prefix(path)
        dump(path,original)
        proof=self.root/original['pre_prefix_confirmation']['path']
        proof.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):self.prefix(path)
    def test_clarification_cannot_claim_explicit_user_choice_or_ignore_attempts(self):
        path=self.clarification();record=c.core.read_json(path)
        record['user_explicit_option_approval']=True;dump(path,record)
        with self.assertRaisesRegex(ValueError,'explicit user option approval'):self.prefix(path)
        record['user_explicit_option_approval']=False
        proof=self.root/record['pre_prefix_confirmation']['path'];confirmation=c.core.read_json(proof)
        confirmation['prefix_attempts']=1;dump(proof,confirmation)
        record['pre_prefix_confirmation']['sha256']=c.core.file_sha(proof);dump(path,record)
        with self.assertRaisesRegex(ValueError,'zero-attempt confirmation'):self.prefix(path)
    def test_both_complete_readouts_fail_under_valid_clarification(self):
        def alter(rows):
            for row in rows:
                row['full_text']=row['assistant_prefill']+'"selected": []}'
                row['suffix_text']=row['full_text'][len(row['assistant_prefill']):]
        self.mutate(alter,'PREFIX_CALIB');report=self.prefix(self.clarification())
        self.assertEqual(report['status'],'HOLD_PREFIX_MEASUREMENT')
        self.assertFalse(report['candidate_gate_passed'])
        self.assertEqual(report['qualifying_readouts'],[])
        self.assertEqual(report['permission'],'NO_TRAINING')
    def test_registration_binds_two_starts_without_gpu_report(self):
        result=c.build_registration(self.root/'config.json',self.root/'DESIGN_FREEZE.json',self.root,
            {s:self.receipt(start=s) for s in c.core.STARTS},NATIVE,self.root/'registration')
        reg=c.core.read_json(self.root/result['registration']['path'])
        self.assertEqual(result['permission'],'NO_TRAINING');self.assertNotIn('gpu_verification_file',reg)
        raw=c.core.check_artifact(self.root,reg['artifacts']['dry_raw'])
        self.assertEqual(len(list(c.core.read_jsonl(raw))),1280)
        with self.assertRaises(ValueError):
            c.build_registration(self.root/'config.json',self.root/'DESIGN_FREEZE.json',self.root,
                {s:self.receipt(start=s) for s in c.core.STARTS},NATIVE,self.root/'registration')


class PrefixParserTests(unittest.TestCase):
    def test_reject_duplicate_keys_nonfinite_and_trailing_text(self):
        for text in ('{"selected":["A"],"selected":["A"]}', '{"selected":["A"],"x":NaN}',
                     '{"selected":["A"],"x":Infinity}', '{"selected":["A"]} trailing'):
            with self.subTest(text=text):
                row=c.prefix_selection(text,'eos',['A'])
                self.assertEqual(row['strict']['S_known'],0);self.assertEqual(row['sensitivity']['S_known'],0)
                self.assertEqual(row['brace_eligible'],0)
    def test_no_content_repair_or_length_repair(self):
        for text,finish in [('{"selected":["A"]','length'),('{"selected":["A"','eos'),
                            ('{"selected":["A"]','balanced')]:
            row=c.prefix_selection(text,finish,['A']);self.assertEqual(row['brace_eligible'],0)
            self.assertEqual(row['sensitivity']['S_hit'],0)
    def test_unknown_not_dropped_duplicate_selected_known_but_wrong(self):
        row=c.prefix_selection('{"selected":["A","A"]}','eos',['A'])
        self.assertEqual(row['strict']['S_known'],1);self.assertEqual(row['strict']['S_hit'],0)
        row=c.prefix_selection('{"selected":["B"]','eos',['A'])
        self.assertEqual(row['brace_eligible'],1);self.assertEqual(row['sensitivity']['S_hit'],0)


if __name__=='__main__':unittest.main()
