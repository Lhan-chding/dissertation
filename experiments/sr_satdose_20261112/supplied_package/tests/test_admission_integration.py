"""Synthetic requests scored by the REAL pinned CPU scorer.

These are deliberately small, artificial fixtures. They are never model outputs,
never GPU evidence, and never persisted as an experiment admission receipt.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from satdose_core import *
from check_admission import run, verify_gpu

BUNDLE=os.environ.get('SR_HISTORICAL_BUNDLE')
PACKAGE=Path(__file__).resolve().parents[1]

@unittest.skipUnless(BUNDLE and (Path(BUNDLE)/'vendor/native_src').exists(),
                     'Set SR_HISTORICAL_BUNDLE to run pinned-native integration tests')
class AdmissionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name)
        cls.cfg=read_json(PACKAGE/'config/experiment.json')
        # Lower bootstrap replication count only in isolated unit-test configuration.
        cls.cfg['dry']['bootstrap_B']=200
        write_json(cls.base/'config.json',cls.cfg)
        sys.path.insert(0,str(Path(BUNDLE)/'vendor/native_src'))
        from sr_f6.measurement import score_response
        cls.score=staticmethod(score_response)
        cells=[(st,d,a) for st in STRUCTURES for d in ('largest','smallest') for a in ('sum','mean')]
        manifests=[];cases={};raw=[]
        for kind,reps in [('CONFLICT',4),('ALIGNED',1)]:
            for st,d,a in cells:
                m,r=map(int,st.split('_choose_'))
                for repeat in range(reps):
                    names=list('ABCDEFGH');idx=list(range(2,2+m))
                    alpha=([78,72]+[48]*m+[24]*(6-m)) if d=='largest' else ([12,18]+[42]*m+[72]*(6-m))
                    beta=[12,18,24,30,36,42,48,54]
                    good=[0,1]+idx[:r];bad=[0,1]+idx[::-1][:r]
                    world=dict(categories=names,domain=[10,90],series={'Alpha':alpha,'Beta':beta})
                    query=dict(family='TOPK',selection_series='Alpha',value='Beta',k=2+r,direction=d,aggregate=a)
                    root=f'UNIT_ONLY_{kind}_{st}_{d}_{a}_{repeat}'
                    def result(indices):
                        z=Fraction(sum(beta[i] for i in indices))
                        return str(z if a=='sum' else z/len(indices))
                    task=dict(root_id=root,structure=st,world=world,query=query,
                              strictly_better_indices=[0,1],boundary_indices=idx,r=r,
                              gold=dict(selected=[names[i] for i in good],answer=result(good)))
                    order=names.copy()
                    if kind=='CONFLICT':
                        for target,source in zip(idx,idx[::-1]):order[target]=names[source]
                    case=dict(task=task,audit={'list_order':order,'native_J_applicable':True})
                    cid=object_sha(case);case['case_id']=cid;cases[cid]=case
                    for start in STARTS:
                        gid=f'{root}/{start}'
                        ids=[gid+f'/d{k}' for k in range(8)];seeds=[seed_for(gid,k) for k in range(8)]
                        manifests.append(dict(group_id=gid,start_seed=start,root_id=root,structure=st,
                                              case_id=cid,kind=kind,request_ids=ids,draw_seeds=seeds))
                        for k in range(8):
                            chosen=good if k<4 else bad
                            obj=dict(evidence=[dict(series=series,category=c,value=world['series'][series][i])
                                               for series in ('Alpha','Beta') for i,c in enumerate(names)],
                                     selected=[names[i] for i in chosen],
                                     bindings=[dict(category=names[i],operand=beta[i]) for i in chosen],
                                     aggregate=a,answer=result(chosen))
                            text=json.dumps(obj);ns=score_response(text,world,query,finished=True,finish_reason='balanced')
                            assert ns['J']==int(k<4)
                            raw.append(dict(request_id=ids[k],group_id=gid,case_id=cid,seed=seeds[k],
                                            policy_sha256=cls.cfg['start_bindings'][str(start)]['policy_sha256'],
                                            reference_sha256=cls.cfg['start_bindings'][str(start)]['reference_sha256'],
                                            generation_status='COMPLETE',technical_validation_errors=[],
                                            assistant_prefill='{',suffix_text=text[1:],full_text=text,
                                            finish_reason='balanced',native_score=ns))
        for fname,rows in [('calibration.jsonl',manifests),('cases.jsonl',list(cases.values())),('dry.jsonl',raw)]:
            (cls.base/fname).write_text(''.join(json.dumps(x)+'\n' for x in rows))
        def ref(n):return {'path':n,'sha256':file_sha(cls.base/n)}
        arts={'calibration_groups':ref('calibration.jsonl'),'private_cases':ref('cases.jsonl'),'dry_raw':ref('dry.jsonl')}
        write_json(cls.base/'freeze.json',dict(schema='sr-satdose-design-freeze-v1',config_sha256=file_sha(cls.base/'config.json'),artifacts=arts))
        arts['design_freeze']=ref('freeze.json')
        write_json(cls.base/'registration.json',dict(schema='sr-satdose-registration-v1',mode='production',
                                                     experiment_id=cls.cfg['experiment_id'],config_sha256=file_sha(cls.base/'config.json'),artifacts=arts))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for p in self.base.iterdir():shutil.copy2(p,self.root/p.name)
    def tearDown(self):self.tmp.cleanup()
    def audit(self):return run(self.root/'config.json',self.root/'registration.json',self.root,Path(BUNDLE)/'vendor/native_src')
    def mutate_raw(self,fn):
        p=self.root/'dry.jsonl';rows=list(read_jsonl(p));fn(rows)
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        reg=read_json(self.root/'registration.json');reg['artifacts']['dry_raw']['sha256']=file_sha(p)
        write_json(self.root/'registration.json',reg)
    def test_real_native_cpu_replay_does_not_authorize_training(self):
        x=self.audit();self.assertEqual(x['permission'],'NO_TRAINING_YET')
        self.assertEqual(x['status'],'DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED')
        self.assertEqual(x['starts']['11031']['metrics']['dose_mean'],4.)
    def test_native_reward_tamper(self):
        self.mutate_raw(lambda rows: rows[0]['native_score'].update(J=0))
        with self.assertRaisesRegex(ValueError,'reward replay'):self.audit()
    def test_duplicate_response(self):
        self.mutate_raw(lambda rows:rows.append(rows[0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):self.audit()
    def test_missing_response(self):
        self.mutate_raw(lambda rows:rows.pop())
        with self.assertRaisesRegex(ValueError,'missing'):self.audit()
    def test_wrong_checkpoint(self):
        self.mutate_raw(lambda rows:rows[0].update(policy_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'checkpoint'):self.audit()
    def test_wrong_reference(self):
        self.mutate_raw(lambda rows:rows[0].update(reference_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'reference'):self.audit()
    def test_technical_failure_not_zero_reward(self):
        self.mutate_raw(lambda rows:rows[0].update(technical_validation_errors=['OOM']))
        with self.assertRaisesRegex(ValueError,'failed/unknown'):self.audit()
    def test_posthoc_cases_rejected(self):
        p=self.root/'cases.jsonl';p.write_text(p.read_text()+'\n')
        reg=read_json(self.root/'registration.json');reg['artifacts']['private_cases']['sha256']=file_sha(p)
        write_json(self.root/'registration.json',reg)
        with self.assertRaisesRegex(ValueError,'post-generation'):self.audit()
    def test_missing_gpu_evidence_rejected(self):
        with self.assertRaises(KeyError):verify_gpu({},self.cfg,self.root/'config.json',self.root/'registration.json',self.root)
    def test_simulated_gpu_report_rejected(self):
        with self.assertRaises(ValueError):verify_gpu({'schema':'sr-satdose-gpu-verification-v1','mode':'simulation'},self.cfg,self.root/'config.json',self.root/'registration.json',self.root)
    def test_real_hash_mismatch_rejected(self):
        p=self.root/'dry.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.audit()

if __name__=='__main__':unittest.main()
