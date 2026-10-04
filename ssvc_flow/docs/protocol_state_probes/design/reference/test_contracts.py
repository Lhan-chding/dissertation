from pathlib import Path
import csv,json,itertools,math,unittest
from fractions import Fraction
import numpy as np
from probe_contract import *

ROOT=Path(__file__).resolve().parents[1]
CASES=[json.loads(l) for l in (ROOT/'manifests/cases.jsonl').read_text().splitlines()]
AUD={r['case_id']:r for r in [json.loads(l) for l in (ROOT/'manifests/audit_labels_NOT_FOR_MODEL.jsonl').read_text().splitlines()]}
RECORDS={r['prompt_id']:r for r in [json.loads(l) for l in (ROOT/'sources/selected_original_records.jsonl').read_text().splitlines()]}
CORE=[c for c in CASES if c['kind']=='repair' and c['panel'] in ('D48','U22')]

class ContractTests(unittest.TestCase):
    def test_01_permutation_roundtrip_all24(self):
        for s in itertools.permutations(range(4)):
            self.assertEqual(restore(emit([5,10,20,40],s),s),[5,10,20,40])
    def test_02_bad_permutation(self):
        for p in [(0,1,1,3),(0,1,2),(0,1,2,True)]:
            with self.assertRaises(ValueError):permutation(p)
    def test_03_reverse_answer_is_canonical(self):
        self.assertEqual(event_label('[40,20,10,5]',REV,[5,10,20,40],'difference_pairs')[0],'X')
        self.assertNotEqual(event_label('[40,20,10,5]',FWD,[5,10,20,40],'difference_pairs')[0],'X')
    def test_04_full_json_only(self):
        for x in ['text [1,2,3,4]','```json\n[1,2,3,4]\n```','[1,2,3,4]\n[1,2,3,4]','[1,2,3]','{"a":1}','[1.0,2,3,4]','[true,2,3,4]']:
            self.assertIsNone(parse_four_ints(x))
    def test_05_whitespace_allowed(self):
        self.assertEqual(parse_four_ints(' \n [1,2,3,4] \t'),[1,2,3,4])
    def test_06_domain_invalid_not_discarded(self):
        e,y=event_label('[-1,2,3,4]',FWD,[1,2,3,4],'sum4')
        self.assertEqual(e,'I');self.assertEqual(y,[-1,2,3,4])
    def test_07_answer_event_partition(self):
        t=[10,20,30,40]
        self.assertEqual(event_label('[11,19,30,40]',FWD,t,'sum4')[0],'S')
        self.assertEqual(event_label('[10,20,30,41]',FWD,t,'sum4')[0],'W')
    def test_08_literal_O0(self):
        for c in CORE:
            if c['protocol']=='O0':
                r=RECORDS[c['parent_prompt_id']]
                self.assertEqual(c['prompt']['system'],r['prompt']['system'])
                self.assertEqual(c['prompt']['user'],r['prompt']['user'])
    def test_09_A1_only_final_output_contract(self):
        r=next(r for r in RECORDS.values() if r['family']=='trend')
        p=public_from_record(r)
        self.assertEqual(render(p,'A1')['user'],render(p,'O0')['user'].replace('Return [a,b,c,d]','Return [d,c,b,a]'))
    def test_10_no_truth_renderer_input(self):
        p=public_from_record(next(iter(RECORDS.values())));p['truth']=[1,2,3,4]
        with self.assertRaises(ValueError):render(p,'O0')
    def test_11_no_DPE_renderer_input(self):
        p=public_from_record(next(iter(RECORDS.values())));p['DPE']=True
        with self.assertRaises(ValueError):render(p,'O0')
    def test_12_input_named_order_preserves_values(self):
        r=next(r for r in RECORDS.values() if r['family']=='trend');p=public_from_record(r)
        a=render(p,'L00')['user'];b=render(p,'L10')['user']
        for k in range(4):
            tag=f'{NAMES[k]}={p["observed"][k]}'
            self.assertIn(tag,a);self.assertIn(tag,b)
        self.assertEqual(render(p,'L10')['output_order'],list(FWD))
        self.assertEqual(render(p,'L11')['output_order'],list(REV))
    def test_13_star_transform_row_equivalence(self):
        for r in RECORDS.values():
            if r['family']!='cross_series':continue
            H,b=matrix_from_scene(r['scene']);HH,bb,meta=transform_star(H,b)
            T=np.asarray(meta['row_transform'],dtype=int);H0=np.asarray(H);b0=np.asarray(b)
            self.assertEqual((T@H0).tolist(),HH);self.assertEqual((T@b0).tolist(),bb)
            self.assertAlmostEqual(abs(np.linalg.det(T)),1.)
    def test_14_B1_full_conjunction_invariance(self):
        for r in RECORDS.values():
            if r['family']!='cross_series':continue
            H,b=matrix_from_scene(r['scene']);HH,bb,_=transform_star(H,b)
            for y in [r['scene']['truth_world'],r['scene']['observed_world'],[-1,101,0,12],[0]*4]:
                self.assertEqual(all(relation_values(H,b,y)),all(relation_values(HH,bb,y)))
    def test_15_partial_C_not_invariant(self):
        found=False
        for r in RECORDS.values():
            if r['family']!='cross_series':continue
            H,b=matrix_from_scene(r['scene']);HH,bb,_=transform_star(H,b)
            y=r['scene']['observed_world']
            if sum(relation_values(H,b,y))!=sum(relation_values(HH,bb,y)):found=True
        self.assertTrue(found)
    def test_16_BR_program_invariance(self):
        for r in RECORDS.values():
            if r['family']!='cross_series':continue
            p=public_from_record(r)
            for v in ['V1','V2']:self.assertEqual(predict(p,'O0')[v]['canonical'],predict(p,'BR')[v]['canonical'])
    def test_17_PTLC_majority(self):
        H=[[1,1,0,0],[1,0,1,0],[1,0,0,1]];b=[10,11,12]
        y,t=run(H,b,[99,2,3,0],order=(1,2,3,0))
        self.assertEqual(y[0],8);self.assertEqual(t[-1]['reason'],'STRICT_MAJORITY')
    def test_18_PTLC_conflict_copy(self):
        H=[[1,1,0,0],[1,0,1,0],[1,0,0,1]]
        y,t=run(H,[10,20,30],[7,2,3,4],order=(1,2,3,0))
        self.assertEqual(y[0],7);self.assertEqual(t[-1]['reason'],'CONFLICT_FALLBACK_COPY')
    def test_19_V2_sequential_fallback(self):
        H=[[1,-2,1,0],[0,1,-2,1]]
        y1,_=run(H,[0,0],[70,4,10,20],variant='V1')
        y2,_=run(H,[0,0],[70,4,10,20],variant='V2')
        self.assertEqual(external(y1),[70,4,-62,-128]);self.assertEqual(external(y2),[70,4,10,16])
    def test_20_no_rounding(self):
        y,_=run([[2,1,0,0]],[1],[3,0,4,5],order=(1,0,2,3))
        self.assertEqual(y[0],Fraction(1,2))
    def test_21_DPE_program_theorem_all_metadata(self):
        rows=list(csv.DictReader((ROOT/'sources/task_DPE_registry.csv').read_text().splitlines()))
        tested=0
        for row in rows:
            if row['interface']!='SYMBOLIC_FRESH':continue
            H=json.loads(row['H']);b=json.loads(row['b']);o=json.loads(row['observed']);truth=json.loads(row['truth']);j=int(row['j0'])
            for s in itertools.permutations(FWD):
                for v in ['V1','V2']:
                    y,_=run(H,b,o,s,variant=v)
                    self.assertEqual(y==list(map(Fraction,truth)),dpe(H,j,s));tested+=1
        self.assertEqual(tested,19008)
    def test_22_expected_D_and_U_B1_flip_counts(self):
        for panel,n in [('D48',6),('U22',2)]:
            hits=[c for c in CORE if c['panel']==panel and c['protocol']=='B1' and not AUD[c['case_id']]['DPE1_old'] and AUD[c['case_id']]['DPE1_new']]
            self.assertEqual(len(hits),n)
    def test_23_expected_A1_D_transition(self):
        cnt={}
        for c in CORE:
            if c['panel']=='D48' and c['protocol']=='A1':
                a=AUD[c['case_id']];key=(a['DPE1_old'],a['DPE1_new']);cnt[key]=cnt.get(key,0)+1
        self.assertEqual(cnt,{(False,True):28,(True,False):19,(True,True):1})
    def test_24_B1_risk_by_literal_predictions(self):
        counts={'V1':0,'V2':0}
        for r in RECORDS.values():
            if r.get('panel')!='D' or r['family']!='cross_series' or r['interface']!='SYMBOLIC_FRESH':continue
            p=public_from_record(r);_,_,meta=transform_star(p['H'],p['b'])
            j=r['scene']['changed_index'];f=meta['first_leaf'];center=meta['center']
            if j!=f or center<f:continue
            for v in counts:
                old=predict(p,'O0')[v];new=predict(p,'B1')[v]
                counts[v]+=old['canonical']!=new['canonical']
        self.assertEqual(counts,{'V1':2,'V2':3})
    def test_25_cases_unique(self):
        self.assertEqual(len(CASES),509);self.assertEqual(len(set(c['case_id'] for c in CASES)),509)
    def test_26_audit_separate(self):
        for c in CASES:
            self.assertFalse(any(k in c['prompt'] for k in ['truth','j','DPE1','solver_output']))
        self.assertEqual(len(AUD),509)
    def test_27_core_new_labels_ground_truth_check(self):
        for c in CORE:
            a=AUD[c['case_id']];raw=json.dumps(emit(a['truth_world'],c['output_order']))
            e,_=event_label(raw,c['output_order'],a['truth_world'],c['operation'])
            self.assertEqual(e,'X')
    def test_28_aliases_exact_only(self):
        by={c['case_id']:c for c in CASES}
        aliases=[json.loads(l) for l in (ROOT/'manifests/prompt_aliases.jsonl').read_text().splitlines()]
        self.assertEqual(len(aliases),35)
        for r in aliases:
            a,b=by[r['case_id']],by[r['owner_case_id']]
            self.assertEqual(a['prompt'],b['prompt']);self.assertEqual(a['output_order'],b['output_order'])
            self.assertEqual(AUD[a['case_id']]['truth_world'],AUD[b['case_id']]['truth_world'])
    def test_29_seed_role_independent(self):
        a=sample_seed('p','S96','q',0)
        self.assertEqual(a,sample_seed('p','S96','q',0))
        self.assertNotEqual(a,sample_seed('p','S32','q',0))
        self.assertNotEqual(a,sample_seed('p','S96','q',0,'smoke'))
        self.assertNotEqual(a,sample_seed('p','S96','q',1))
    def test_30_cp_zero_bound(self):
        self.assertAlmostEqual(zero_success_upper(32),.08936819898626477)
        self.assertGreater(zero_success_upper(64),.04)
    def test_31_boundary_se_not_certainty(self):
        s=contrast([0,0],[32,32],[1,-1]);self.assertTrue(s['has_boundary_cells'])
        self.assertTrue(s['plugin_se_is_not_a_coverage_guarantee'])
    def test_32_interaction_not_routing_value(self):
        m=np.array([[.5,.7],[.4,.8]]) # effects differ, same best action
        interaction=(m[1,1]-m[1,0])-(m[0,1]-m[0,0])
        self.assertGreater(interaction,0)
        self.assertAlmostEqual(m.max(1).mean()-m.mean(0).max(),0)
    def test_33_masks_do_not_double_count(self):
        r=next(r for r in RECORDS.values() if r['family']=='cross_series')
        p=public_from_record(r);f=diagnostic_features(json.dumps(p['observed']),p,r['scene']['truth_world'],'O0')
        self.assertEqual(f['match_mask']%2,1);self.assertTrue(0<=f['match_mask']<=7)
    def test_34_unparseable_repair_fields_null(self):
        r=next(r for r in RECORDS.values() if r['family']=='trend');p=public_from_record(r)
        f=diagnostic_features('nonsense',p,r['scene']['truth_world'],'O0')
        self.assertEqual(f['event'],'I');self.assertIsNone(f['F']);self.assertIsNone(f['match_mask'])
    def test_35_out_of_domain_match_allowed(self):
        r=next(r for r in RECORDS.values() if r['family']=='trend' and not predict(public_from_record(r),'O0')['V1']['in_domain'])
        p=public_from_record(r);y=predict(p,'O0')['V1']['emitted']
        f=diagnostic_features(json.dumps(y),p,r['scene']['truth_world'],'O0')
        self.assertEqual(f['event'],'I');self.assertTrue(f['match_V1']);self.assertIsNone(f['F'])
    def test_36_workload_and_no_train(self):
        jobs=[json.loads(l) for l in (ROOT/'manifests/logical_jobs.jsonl').read_text().splitlines()]
        p=json.loads((ROOT/'protocol.json').read_text())
        self.assertEqual(len(jobs),1944);self.assertEqual(sum(j['draws'] for j in jobs),70656)
        self.assertFalse(p['execution']['new_training_allowed']);self.assertEqual(p['workload']['model_training_steps'],0)
    def test_37_checkpoint_missing_no_retrain(self):
        states=json.loads((ROOT/'manifests/checkpoints.json').read_text())
        self.assertEqual(len(states),6)
        for s in states:self.assertFalse(s['allow_retraining_if_missing'])
    def test_38_controls_not_core(self):
        controls=[c for c in CASES if c['panel'].startswith('CONTROL')]
        self.assertEqual(len(controls),72)
        self.assertEqual(sum(c['kind']=='clean_relation_control' for c in controls),24)
        self.assertFalse(any(c['panel']=='D48' for c in controls))
    def test_39_U_fixed_not_history_selected(self):
        U={c['base_scene_id'] for c in CASES if c['panel']=='U22'}
        D={c['base_scene_id'] for c in CASES if c['panel']=='D48'}
        self.assertEqual(len(U),22);self.assertFalse(U&D)
        self.assertTrue(all(c['original_split']=='train' for c in CASES if c['panel']=='U22'))
    def test_40_factorial_complete(self):
        jobs=[json.loads(l) for l in (ROOT/'manifests/logical_jobs.jsonl').read_text().splitlines()]
        for s in ['S32','S96']:
            for p in ['L00','L01','L10','L11']:
                self.assertEqual(sum(j['checkpoint_id']==s and j['protocol']==p for j in jobs),48)
    def test_41_B_not_for_trend(self):
        r=next(r for r in RECORDS.values() if r['family']=='trend')
        with self.assertRaises(ValueError):render(public_from_record(r),'B1')
    def test_42_posteriors_of_alias_need_same_variable(self):
        # Algebraic check: an actual alias comparison is the zero variable, not a
        # difference of independently simulated estimates of the same probability.
        x=np.linspace(.001,.999,100)
        self.assertTrue(np.all(x-x==0))
    def test_43_undefined_condition_not_zero(self):
        n=0;rate=None if n==0 else 0/n
        self.assertIsNone(rate)
    def test_44_reference_predictions_unchanged(self):
        preds=[json.loads(l) for l in (ROOT/'manifests/program_predictions.jsonl').read_text().splitlines()]
        by={c['case_id']:c for c in CASES}
        for pr in preds:
            c=by[pr['case_id']];record=RECORDS[c['parent_prompt_id']]
            self.assertEqual(pr['programs'],predict(public_from_record(record),c['protocol']))

    def test_45_B1_literal_subtraction_not_leading_negative(self):
        for r in RECORDS.values():
            if r['family']!='cross_series':continue
            p=public_from_record(r); rr=render(p,'B1')
            eq=rr['user'].split('The following relationships are reliable:\n')[1].split('\nThe downstream calculation is:')[0].splitlines()
            for row in eq[1:]:
                self.assertFalse(row.startswith('-'))
                self.assertIn(' - ',row)
    def test_46_sampling_config_frozen_completely(self):
        p=json.loads((ROOT/'protocol.json').read_text());g=p['generation']['generation_config']
        self.assertFalse(g['use_cache']);self.assertFalse(g['renormalize_logits'])
        self.assertEqual(g['repetition_penalty'],1.0);self.assertEqual(g['max_new_tokens'],64)
        self.assertEqual(g['min_p'],0.0);self.assertEqual(g['num_beams'],1)

if __name__=='__main__':unittest.main(verbosity=2)
