"""Fixed scientific risk arithmetic and evidence-boundary CPU tests."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from analysis import risk
sys.path.pop(0)


def metric(n,hits,known=None):
    return dict(assigned=n,counts=dict(S_hit=hits,S_known=n if known is None else known))


def dev(conflict=25,aligned=64):
    return dict(conflict=metric(32,conflict),aligned_no_list=metric(64,aligned))


def fixture(n=8):
    progress={key:n for key in risk.PATHS};steps={};endpoints={}
    for start in (11031,11032):endpoints[f'{start}/START/0']=dev()
    for key in risk.PATHS:
        t1=key.endswith('/T1')
        steps[key]=[dict(step=i,conflict_g8_count=12 if t1 else 0,conflict_response_count=96 if t1 else 0,
            exact_authorised=72 if t1 else 0,known=96 if t1 else 0,conflict_group_doses=[.75]*12 if t1 else [])
            for i in range(1,n+1)]
        for step in risk.DEV_STEPS:
            if step<=n:endpoints[f'{key}/{step}']=dev()
    return progress,steps,endpoints


class RiskArithmeticTests(unittest.TestCase):
    def test_exact_d_threshold_is_not_stop(self):
        result=risk._decision(*fixture())
        self.assertEqual(result['status'],'NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED')
        self.assertFalse(result['training_permission'])

    def test_one_t1_path_triggers_global_hold_both_windows(self):
        p,s,d=fixture()
        for step in s['11032/T1']:step['conflict_group_doses']=[.749]*12
        out=risk._decision(p,s,d)
        self.assertEqual(out['hold_reasons'],['HOLD_SIGNAL_NOT_DELIVERED'])
        self.assertEqual(out['signals'][0]['path'],'11032/T1')
        self.assertEqual(out['signals'][0]['window_ends'],[4,8])
        self.assertEqual(out['paths']['11032/T1']['windows'][0]['conflict_response_count'],384)

    def test_only_one_low_window_does_not_trigger(self):
        p,s,d=fixture()
        for step in s['11031/T1'][:4]:step['conflict_group_doses']=[0.0]*12
        self.assertFalse(risk._decision(p,s,d)['global_hold'])

    def test_high_authorised_rate_prevents_false_learning_stop(self):
        p,s,d=fixture()
        for step in s['11031/T1']:
            step['conflict_group_doses']=[0.0]*12;step['exact_authorised']=77
        self.assertFalse(risk._decision(p,s,d)['global_hold'])
        for i in (3,7):s['11031/T1'][i]['exact_authorised']=76
        self.assertTrue(risk._decision(p,s,d)['global_hold'])

    def test_dev_26_of32_prevents_signal_stop_25_triggers(self):
        p,s,d=fixture()
        for step in s['11031/T1']:step['conflict_group_doses']=[0.0]*12
        d['11031/T1/8']=dev(conflict=26)
        self.assertFalse(risk._decision(p,s,d)['global_hold'])
        d['11031/T1/8']=dev(conflict=25)
        self.assertTrue(risk._decision(p,s,d)['global_hold'])

    def test_missing_current_dev8_is_pending_no_stale_dev2_substitution(self):
        p,s,d=fixture()
        for step in s['11031/T1']:step['conflict_group_doses']=[0.0]*12
        del d['11031/T1/8']
        out=risk._decision(p,s,d)
        self.assertEqual(out['status'],'PENDING_REQUIRED_DEV')
        self.assertEqual(out['pending_required_dev'],['11031/T1/8'])
        self.assertFalse(out['global_hold'])

    def test_missing_dev2_before_step3_blocks(self):
        p,s,d=fixture(3);del d['11032/T0/2']
        out=risk._decision(p,s,d)
        self.assertTrue(out['next_training_step_requires_complete_dev'])
        self.assertEqual(out['pending_required_dev'],['11032/T0/2'])

    def test_nonoverlapping_windows_do_not_slide(self):
        p,s,d=fixture(12)
        for index,step in enumerate(s['11031/T1']):
            step['conflict_group_doses']=[0.0 if index<4 or index>=8 else 2.0]*12
        out=risk._decision(p,s,d)
        self.assertFalse(out['global_hold'])
        self.assertEqual([x['last_step'] for x in out['paths']['11031/T1']['windows']],[4,8,12])

    def test_prior_trigger_not_erased_by_later_good_window(self):
        p,s,d=fixture(12)
        for step in s['11031/T1'][:8]:step['conflict_group_doses']=[0.0]*12
        self.assertTrue(risk._decision(p,s,d)['global_hold'])

    def test_step12_uses_latest_dev8(self):
        p,s,d=fixture(12)
        for step in s['11031/T1'][4:]:step['conflict_group_doses']=[0.0]*12
        out=risk._decision(p,s,d)
        self.assertEqual(out['signals'][0]['dev_endpoint'],'11031/T1/8')
        self.assertEqual(out['signals'][0]['window_ends'],[8,12])

    def test_severe_regression_two_scheduled_checkpoints_and_own_start(self):
        p,s,d=fixture()
        d['11031/START/0']=dev(aligned=40)
        d['11031/T0/2']=dev(aligned=27);d['11031/T0/8']=dev(aligned=27)
        out=risk._decision(p,s,d)
        self.assertEqual(out['hold_reasons'],['HOLD_SEVERE_REGRESSION'])
        self.assertEqual(out['regressions'][0]['path'],'11031/T0')
        d['11031/T0/8']=dev(aligned=28)
        self.assertFalse(risk._decision(p,s,d)['global_hold'])

    def test_one_severe_checkpoint_does_not_trigger(self):
        p,s,d=fixture(2);d['11031/T0/2']=dev(aligned=0)
        self.assertFalse(risk._decision(p,s,d)['global_hold'])

    def test_missing_intermediate_checkpoint_cannot_pair_2_and16(self):
        p,s,d=fixture(16)
        d['11031/T0/2']=dev(aligned=0);d['11031/T0/16']=dev(aligned=0);del d['11031/T0/8']
        out=risk._decision(p,s,d)
        self.assertFalse(out['global_hold']);self.assertEqual(out['status'],'PENDING_REQUIRED_DEV')

    def test_unknowns_keep_fixed_denominators(self):
        p,s,d=fixture()
        for step in s['11031/T1']:
            step['conflict_group_doses']=[0.0]*12;step['exact_authorised']=0;step['known']=0
        d['11031/T1/8']=dict(conflict=metric(32,0,0),aligned_no_list=metric(64,64))
        out=risk._decision(p,s,d);window=out['paths']['11031/T1']['windows'][0]
        self.assertEqual(window['unknown'],384);self.assertEqual(window['conflict_response_count'],384)
        self.assertEqual(out['signals'][0]['dev_conflict']['unknown'],32)

    def test_reject_denominator_changes_missing_steps_duplicate_and_nonfinite(self):
        for mutate in [lambda p,s,d:s['11031/T1'][0].update(conflict_response_count=128),
                       lambda p,s,d:s['11031/T1'].pop(),
                       lambda p,s,d:s['11031/T1'][0].update(step=2),
                       lambda p,s,d:s['11031/T1'][0].update(conflict_group_doses=[float('nan')]*12),
                       lambda p,s,d:s['11031/T1'][0].update(exact_authorised=True),
                       lambda p,s,d:d['11031/T1/8']['conflict'].update(assigned=31),
                       lambda p,s,d:d.update({'11031/T1/16':dev()}),
                       lambda p,s,d:d.pop('11031/START/0'),
                       lambda p,s,d:s['11031/T0'][0].update(conflict_group_doses=[0.0]*12)]:
            with self.subTest(mutate=mutate):
                p,s,d=fixture();mutate(p,s,d)
                with self.assertRaises(PermissionError):risk._decision(p,s,d)

    def test_inputs_unchanged_and_zero_steps_supported(self):
        args=fixture(0);before=copy.deepcopy(args);out=risk._decision(*args)
        self.assertEqual(args,before);self.assertFalse(out['global_hold']);self.assertEqual(out['pending_required_dev'],[])


class RiskEvidenceTests(unittest.TestCase):
    def test_untrusted_self_pin_rejected_before_native_load(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(risk.fork_loader,'load_native_ops') as load:
            with self.assertRaises(PermissionError):
                risk.assess(Path(temporary).resolve(),config_ref={},freeze_ref={},clarification_ref={},
                    trusted_commits={},dev_endpoints={},expected_source_sha256='0'*64)
            load.assert_not_called()

    def test_evidence_mutation_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();p=root/'proof.json';p.write_text('{}');e=risk._Evidence(root)
            e.ref(dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            p.write_text('{"mutated":true}')
            with self.assertRaises(PermissionError):e.stable()
            q=root/'link.json';q.symlink_to(p)
            with self.assertRaises(PermissionError):risk._Evidence(root).path(q)

    def test_clarification_matches_existing_frozen_user_answer(self):
        p=PACKAGE.parents[1]/'artifacts/sr_satdose_20261112/intake/TRAIN_RISK_CLARIFICATION_20261114.json'
        value=risk.tx.read_json(p);risk._clarification(value)
        value['per_window_conflict_response_count']=512
        with self.assertRaises(PermissionError):risk._clarification(value)

    def test_original_train_scores_recomputed_and_t0_has_no_conflict(self):
        def make(arm):
            groups=[];raw=[];cases={}
            for i in range(16):
                conflict=arm=='T1' and i<12;cid=f'case{i}';rid=f'root{i}';gid=f'group{i}'
                groups.append(dict(case_id=cid,root_id=rid,group_id=gid,start_seed=11031,arm=arm,step=1,
                    planned_conflict=conflict,request_ids=[f'r{i}-{j}' for j in range(8)]))
                cases[cid]=dict(task=dict(world={'marker':i},query={}),audit={'conflict':conflict})
                for j in range(8):
                    score=dict(J=j%2,S=j%2,known_mask={'S':1},witness={'selected':['a'] if j%2 else ['b']})
                    raw.append(dict(request_id=f'r{i}-{j}',case_id=cid,group_id=gid,start_seed=11031,arm=arm,step=1,
                        scope='TRAIN',generation_status='COMPLETE',technical_validation_errors=[],finish_reason='balanced',
                        full_text=str(j),native_score=score,J=j%2,exact_authorised=j%2))
            return raw,groups,cases
        def geometry(task,audit):return dict(chart=frozenset({'a'}),alt=frozenset({'b'} if audit['conflict'] else {'a'}),authorised=frozenset({'a'}))
        def scorer(text,*args,**kwargs):
            j=int(text);return dict(J=j%2,S=j%2,known_mask={'S':1},witness={'selected':['a'] if j%2 else ['b']})
        with patch.object(risk.core,'geometry',side_effect=geometry):
            for arm,n in [('T1',12),('T0',0)]:
                args=make(arm);value=risk._score_step(*args,scorer,11031,arm,1)
                self.assertEqual(value['conflict_g8_count'],n);self.assertEqual(value['conflict_response_count'],8*n)
                if n:self.assertGreater(value['conflict_group_doses'][0],0)
                args[0][0]['native_score']['J']=1
                with self.assertRaises(PermissionError):risk._score_step(*args,scorer,11031,arm,1)


if __name__=='__main__':unittest.main()
