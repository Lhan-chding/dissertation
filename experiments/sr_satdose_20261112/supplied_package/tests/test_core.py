import itertools
import math
import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from satdose_core import *
from build_schedule import build, validate
from analyze_primary import analyze

class JSONTests(unittest.TestCase):
    def test_duplicate(self):
        with self.assertRaises(ValueError):loads('{"a":1,"a":2}')
    def test_nonfinite(self):
        for s in ('NaN','Infinity','-Infinity'):
            with self.subTest(s=s),self.assertRaises(ValueError):loads('{"x":'+s+'}')
    def test_text_bound(self):self.assertEqual(parse_selected(' '*100001,'balanced'),(None,None))
    def test_markdown_unknown(self):self.assertIsNone(parse_selected('```{"selected":["A"]}```','balanced')[1])
    def test_partial_length_unknown(self):self.assertIsNone(parse_selected('{"selected":["A"]}','length')[1])
    def test_eos_readable(self):self.assertEqual(parse_selected('{"selected":["A"]}','eos')[1],['A'])
    def test_not_array_unknown(self):self.assertIsNone(parse_selected('{"selected":"A"}','balanced')[1])
    def test_top_list_unknown(self):self.assertIsNone(parse_selected('[]','balanced')[1])
    def test_duplicate_label_not_correct(self):self.assertEqual(exact_hit(['A','A'],['A']),0)
    def test_order_invariant(self):self.assertEqual(exact_hit(['B','A'],['A','B']),1)
    def test_unknown_not_correct(self):self.assertEqual(exact_hit(None,['A']),0)
    def test_no_auto_brace_repair(self):self.assertIsNone(parse_selected('{"selected":["A"]','eos')[1])

class SourceTests(unittest.TestCase):
    def setUp(self):
        self.g={'B':frozenset(['H']),'T':frozenset(['A','B','C']),'r':1,
                'chart':frozenset(['H','A']),'alt':frozenset(['H','B'])}
    def test_chart(self):self.assertEqual(source_class(['A','H'],self.g),'chart')
    def test_list(self):self.assertEqual(source_class(['B','H'],self.g),'list')
    def test_non_list_is_not_chart(self):self.assertEqual(source_class(['C','H'],self.g),'other_admissible')
    def test_missing_strict_better(self):self.assertEqual(source_class(['A'],self.g),'inadmissible')
    def test_extra(self):self.assertEqual(source_class(['A','H','X'],self.g),'inadmissible')
    def test_duplicate(self):self.assertEqual(source_class(['A','H','H'],self.g),'inadmissible')
    def test_unknown(self):self.assertEqual(source_class(None,self.g),'unknown')
    def test_same_is_not_evidence_for_source(self):
        self.g['alt']=self.g['chart'];self.assertEqual(source_class(['A','H'],self.g),'same')

class AdvantageTests(unittest.TestCase):
    def test_all_256_binary_vectors(self):
        for v in itertools.product((0,1),repeat=8):
            a=native_advantages(list(v));self.assertAlmostEqual(sum(a),0,places=10)
            self.assertLessEqual(max(map(abs,a)),math.sqrt(7)+1e-10)
            self.assertAlmostEqual(sum(x*x for x in a),8 if 0<sum(v)<8 else 0,places=10)
    def test_aligned_covariance_closed_form(self):
        for k in range(9):
            v=[1]*k+[0]*(8-k);m=group_metrics(v,v)
            self.assertAlmostEqual(m['dose'],math.sqrt(k*(8-k)),places=10)
    def test_wrong_reward_direction(self):
        m=group_metrics([1]*4+[0]*4,[0]*4+[1]*4)
        self.assertEqual(m['dose'],-4)
    def test_allzero_source_mixed(self):
        m=group_metrics([0]*8,[1]*4+[0]*4)
        self.assertEqual(m['dose'],0);self.assertEqual(m['J_silent_source_mixed'],1)
    def test_allone_source_mixed(self):
        m=group_metrics([1]*8,[1]*4+[0]*4)
        self.assertEqual(m['dose'],0);self.assertEqual(m['J_silent_source_mixed'],1)
    def test_eta_perfect(self):self.assertAlmostEqual(group_metrics([1]*4+[0]*4,[1]*4+[0]*4)['eta2'],1.)
    def test_reject_none_float_bool(self):
        for v in ([1]*7+[None],[1]*7+[1.0],[1]*7+[True],[1]*7,[1]*7+[2]):
            with self.subTest(v=v),self.assertRaises(ValueError):native_advantages(v)
    def test_unknown_retained(self):
        m=group_metrics([1]*4+[0]*4,[1]*4+[0]*4,[1]*7+[0]);self.assertEqual(m['known'],7)
    def test_unknown_cannot_be_correct(self):
        with self.assertRaises(ValueError):group_metrics([0]*8,[1]*8,[0]*8)
    def test_dry_keeps_zero_groups(self):
        gs=[dict(J=[1]*4+[0]*4,c=[1]*4+[0]*4,known=[1]*8) for _ in range(32)]
        gs += [dict(J=[0]*8,c=[0]*8,known=[1]*8) for _ in range(32)]
        m=dry_metrics(gs,B=200);self.assertEqual(m['dose_mean'],2);self.assertEqual(m['signal_groups'],32)
    def test_dry_missing_group(self):
        with self.assertRaises(ValueError):dry_metrics([])

class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=build()
    def test_counts(self):
        self.assertEqual(validate(self.rows)['responses'],12288)
    def test_deterministic(self):self.assertEqual(self.rows,build())
    def test_missing_group(self):
        with self.assertRaises(ValueError):validate(self.rows[:-1])
    def test_false_t0_conflict(self):
        rows=[dict(x) for x in self.rows];rows[0]['planned_conflict']=True
        with self.assertRaises(ValueError):validate(rows)
    def test_wrong_draw_seed(self):
        rows=[dict(x) for x in self.rows];rows[0]['draw_seeds']=[1]*8
        with self.assertRaises(ValueError):validate(rows)
    def test_duplicate_request(self):
        rows=[dict(x) for x in self.rows];rows[1]['request_ids']=rows[0]['request_ids']
        with self.assertRaises(ValueError):validate(rows)
    def test_off_pair_world(self):
        rows=[dict(x) for x in self.rows];rows[1]['world_seed']+=1
        with self.assertRaises(ValueError):validate(rows)
    def test_root_reuse(self):
        rows=[dict(x) for x in self.rows];rows[2]['root_id']=rows[0]['root_id']
        with self.assertRaises(ValueError):validate(rows)
    def test_bad_stratum(self):
        rows=[dict(x) for x in self.rows];rows[0]['structure']='new'
        with self.assertRaises(ValueError):validate(rows)

class StatisticsTests(unittest.TestCase):
    def test_interval_no_overlap(self):
        self.assertEqual(effect_classification(.06,.09),'POSITIVE_BUT_BELOW_TARGET_CONDITIONAL')
    def test_practical(self):self.assertEqual(effect_classification(.1,.2),'PRACTICAL_EFFECT_SUPPORTED_CONDITIONAL')
    def test_no_effect(self):self.assertEqual(effect_classification(-.03,.08),'PRACTICAL_EFFECT_EXCLUDED_CONDITIONAL')
    def test_uncertain(self):self.assertEqual(effect_classification(-.03,.2),'UNRESOLVED_CONDITIONAL')
    def test_positive_uncertain_size(self):self.assertEqual(effect_classification(.02,.2),'POSITIVE_SIZE_UNRESOLVED_CONDITIONAL')
    def test_invalid_interval(self):
        with self.assertRaises(ValueError):effect_classification(.2,.1)
    def test_bootstrap_deterministic(self):self.assertEqual(bootstrap_mean([0.,1.,2.],B=200),bootstrap_mean([0.,1.,2.],B=200))
    def test_stratified_root_analysis(self):
        rows=[]
        for st in STRUCTURES:
            for i in range(32):
                for s in STARTS:
                    rows.append(dict(root_id=f'{st}:{i}',structure=st,start_seed=s,draws=2,
                                     start_correct=0,T0_correct=0,T1_correct=1))
        out=analyze(rows,B=200)
        self.assertEqual(out['estimate'],.5);self.assertEqual(out['conditional_root_bootstrap_CI95'],[.5,.5])
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):analyze(rows,B=200)

class ArtifactTests(unittest.TestCase):
    def test_digest_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=r/'x';p.write_text('test');ref={'path':'x','sha256':file_sha(p)}
            self.assertEqual(check_artifact(r,ref),p);p.write_text('tamper')
            with self.assertRaises(ValueError):check_artifact(r,ref)
    def test_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):check_artifact(Path(d),{'path':'../x','sha256':'0'*64})
    def test_absolute(self):
        with self.assertRaises(ValueError):check_artifact(Path('/tmp'),{'path':'/etc/passwd','sha256':'0'*64})

if __name__=='__main__':unittest.main()
