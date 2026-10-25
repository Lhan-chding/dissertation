import copy,json,math,unittest
from fractions import Fraction
import numpy as np
from srf5_ref.generator import make_bank,make_root,audit_splits,seed_for
from srf5_ref.oracle import evaluate_a,evaluate_b,canonical_evidence,encode,heldout_tags
from srf5_ref.measurement import parse_response,score_response,map_token_mask
from srf5_ref.credit import credits,standardize_binary
from srf5_ref.schedule import build_schedule
from srf5_ref.statistics import paired_hierarchical_bootstrap,training_repeat_t,relation_metrics

class ReferenceTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.task=make_root('fixture','BINDING',0,9)[1]
 def response(self,selected=None):
  t=self.task;g=evaluate_a(t['world'],t['query'])
  return json.dumps({'evidence':canonical_evidence(t['world'],t['query']),'work':'ordinary work',
      'selected':g['selected'] if selected is None else selected,'answer':encode(g['answer'])})
 def rows(self):
  return [dict(qid='q',J=1,V=1,finish=1,P='PASS',S='PASS' if i<7 else 'FAIL',mask_ok=True,value_correct=1) for i in range(8)]
 def test_two_oracles_many_roots(self):
  for ood in (False,True):
   ts=make_bank('ood' if ood else 'train',12,101+ood,ood)
   for t in ts:
    self.assertEqual(evaluate_a(t['world'],t['query']),evaluate_b(t['world'],t['query']))
    self.assertEqual(bool(heldout_tags(t['world'],t['query'])),ood)
 def test_split_isolation(self):
  a=make_bank('a',3,1001);b=make_bank('b',3,1002)
  self.assertEqual(audit_splits({'a':a,'b':b})['cross_split_world_collisions'],0)
 def test_leakage_detected(self):
  t=copy.deepcopy(self.task);u=copy.deepcopy(t);u['root_id']='different';u['qid']='different'
  with self.assertRaises(AssertionError):audit_splits({'a':[t],'b':[u]})
 def test_no_unsupported_query(self):
  t=copy.deepcopy(self.task);t['query']['fake_filter']=True
  with self.assertRaises(ValueError):evaluate_a(t['world'],t['query'])
 def test_correct_response(self):
  r=score_response(self.response(),self.task['world'],self.task['query'])
  self.assertEqual((r['J'],r['S'],r['P']),(1,'PASS','PASS'))
 def test_wrong_selected_does_not_change_J(self):
  r=score_response(self.response([]),self.task['world'],self.task['query'])
  self.assertEqual((r['J'],r['V'],r['S']),(1,1,'FAIL'))
 def test_duplicate_selected_semantic_not_syntax(self):
  g=evaluate_a(self.task['world'],self.task['query'])['selected'];r=score_response(self.response(g+[g[0]]),self.task['world'],self.task['query'])
  self.assertEqual((r['V'],r['S']),(1,'FAIL'))
 def test_unknown_label_is_fail_not_parse_error(self):
  r=score_response(self.response(['not_a_category']),self.task['world'],self.task['query'])
  self.assertEqual((r['V'],r['S']),(1,'FAIL'))
 def test_order_invariant(self):
  g=evaluate_a(self.task['world'],self.task['query'])['selected']
  self.assertEqual(score_response(self.response(g[::-1]),self.task['world'],self.task['query'])['S'],'PASS')
 def test_missing_selected_unknown(self):
  o=json.loads(self.response());del o['selected'];e=parse_response(json.dumps(o));self.assertEqual(e['status'],'UNKNOWN')
 def test_prose_does_not_override_final_field(self):
  o=json.loads(self.response([]));o['work']='selected=["P01","P02"] candidate pool, correction comes later'
  self.assertEqual(score_response(json.dumps(o),self.task['world'],self.task['query'])['S'],'FAIL')
 def test_equal_sign_does_not_truncate(self):
  o=json.loads(self.response());o['work']='P01=20,P02=36; [candidate] then correct final field'
  self.assertEqual(score_response(json.dumps(o),self.task['world'],self.task['query'])['S'],'PASS')
 def test_duplicate_json_key_rejected(self):
  e=parse_response(self.response()[:-1]+',"selected":[]}');self.assertEqual(e['status'],'UNKNOWN')
 def test_extra_prose_rejected(self):self.assertEqual(parse_response(self.response()+' extra')['status'],'UNKNOWN')
 def test_nested_array_span(self):
  text=self.response();e=parse_response(text);a,b=e['selected_span'];self.assertEqual(json.loads(text[a:b]),e['selected'])
 def test_character_token_mapping(self):
  text=self.response();e=parse_response(text);ids=[ord(x) for x in text]
  m,reason=map_token_mask(ids,text,e['selected_span'],lambda ids:''.join(map(chr,ids)))
  a,b=e['selected_span'];self.assertEqual(reason,'MAPPED');self.assertEqual(sum(m),b-a)
 def test_prefill_excluded(self):
  text=self.response();e=parse_response(text)
  m,reason=map_token_mask([ord(x) for x in text[1:]],text,e['selected_span'],lambda ids:''.join(map(chr,ids)),prefill='{')
  self.assertEqual(reason,'MAPPED');self.assertEqual(sum(m),e['selected_span'][1]-e['selected_span'][0])
 def test_cross_boundary_token_excluded(self):
  m,r=map_token_mask([0,1],'xx["P01"]yy',[2,9],lambda ids:''.join(['xx["P01"]','yy'][i] for i in ids))
  self.assertEqual(sum(m),0)
 def test_decode_mismatch_abstains(self):
  self.assertEqual(map_token_mask([1],'abc',[0,2],lambda _: 'xyz')[1],'FULL_DECODE_MISMATCH')
 def test_all_correct_J_has_semantic_signal(self):
  c=credits(self.rows(),'LOCAL',1);self.assertTrue(c['native_zero']);self.assertTrue(c['nonzero_group']);self.assertTrue(c['swappable_group'])
  self.assertAlmostEqual(sum(c['aux']),0);self.assertAlmostEqual(c['aux'][-1],-.21875)
 def test_all_zero_J_signal(self):
  rows=self.rows()
  for r in rows:r['J']=0;r['value_correct']=0
  c=credits(rows,'LOCAL',1);self.assertTrue(c['all_J_zero']);self.assertTrue(c['nonzero_group'])
 def test_s_correlated_J_nonzero_not_exchangeable(self):
  rows=self.rows()
  for r in rows:r['J']=int(r['S']=='PASS')
  c=credits(rows,'SHUF',1);self.assertTrue(c['nonzero_group']);self.assertFalse(c['swappable_group']);self.assertFalse(c['changed_group'])
 def test_P_unknown_does_not_gate_S(self):
  rows=self.rows()
  for r in rows:r['P']='UNKNOWN'
  self.assertEqual(len(credits(rows,'LOCAL',1)['eligible']),8)
 def test_unknown_S_zero_aux_not_fail_label(self):
  rows=self.rows();rows[-1]['S']='UNKNOWN';c=credits(rows,'LOCAL',1)
  self.assertEqual(c['aux'][-1],0);self.assertFalse(c['nonzero_group']);self.assertEqual(rows[-1]['S'],'UNKNOWN')
 def test_no_75pct_gate(self):
  rows=self.rows()
  for i in range(6):rows[i]['S']='UNKNOWN'
  c=credits(rows,'LOCAL',1);self.assertTrue(c['nonzero_group']);self.assertEqual(len(c['eligible']),2)
 def test_HARD_is_conjunction(self):
  c=credits(self.rows(),'HARD',1);self.assertAlmostEqual(c['native'][-1],-math.sqrt(7));self.assertFalse(any(c['aux']))
 def test_center_never_across_questions(self):
  rows=self.rows();rows[-1]['qid']='other'
  with self.assertRaises(ValueError):credits(rows,'LOCAL',1)
 def test_lambda_not_multiplied_twice(self):
  c=credits(self.rows(),'LOCAL',1);self.assertLessEqual(max(map(abs,c['aux'])),.25)
 def test_shuffle_preserves_strata_and_multiset(self):
  rows=self.rows();rows[0]['P']='FAIL';c=credits(rows,'SHUF',42)
  self.assertEqual(sorted(c['aux']),sorted(c['counterfactual_true']))
  for i,j in enumerate(c['permutation']):self.assertEqual((rows[i]['J'],rows[i]['P']),(rows[j]['J'],rows[j]['P']))
 def test_schedule_exact_and_arm_free(self):
  ts=make_bank('schedule',48,3344);s=build_schedule(ts,100,'b0',0)
  self.assertEqual(len(s),72);self.assertEqual(sum(len(x['groups'])*8 for x in s),9216)
  ids=[g['qid'] for x in s for g in x['groups']];self.assertEqual(len(set(ids)),1152)
  self.assertEqual(s,build_schedule(ts,100,'b0',0));self.assertNotEqual(s,build_schedule(ts,100,'b0',1))
 def test_statistical_fixed_tensor(self):
  x=np.full((3,3,6,16),.02);r=paired_hierarchical_bootstrap(x,n=100)
  self.assertAlmostEqual(r['effect'],.02);self.assertAlmostEqual(r['lower'],.02)
  self.assertAlmostEqual(training_repeat_t(x)['lower'],.02)
 def test_relation_shared_wrong_offset(self):
  r=relation_metrics([[11,12,13,14]],[1,2,3,4],[0,1,2,3])
  self.assertEqual(r['relation_consistency'],1);self.assertEqual(r['all_four_correct'],0)
 def test_bad_num_does_not_count_correct(self):
  o=json.loads(self.response());o['answer']='1/0';r=score_response(json.dumps(o),self.task['world'],self.task['query']);self.assertEqual(r['value_correct'],0)
 def test_unfinished_never_J(self):
  self.assertEqual(score_response(self.response(),self.task['world'],self.task['query'],finished=False)['J'],0)

class ObjectiveTest(unittest.TestCase):
 def test_local_difference_gradient(self):
  import torch
  from srf5_ref.objective import sequence_loss
  x=torch.tensor([-1.,-2.,-3.,-4.],requires_grad=True);s=x.detach().clone();ref=s.clone()
  r=sequence_loss(x,s,ref,0,.25,[False,True,True,False],'selected',kl_beta=0)
  r['loss'].backward();self.assertTrue(torch.allclose(x.grad,torch.tensor([0.,-.125,-.125,0.])))
 def test_kl_once_and_aux_zero_equivalence(self):
  import torch
  from srf5_ref.objective import sequence_loss
  x=torch.tensor([-1.,-2.,-3.],requires_grad=True);s=x.detach().clone();ref=s+.1
  a=sequence_loss(x,s,ref,1);b=sequence_loss(x,s,ref,1,0,[True,True,False],'selected')
  self.assertEqual(a['loss'].item(),b['loss'].item())
 def test_forward_ratio_one_but_grad_nonzero(self):
  import torch
  from srf5_ref.objective import sequence_loss
  x=torch.tensor([-1.,-2.],requires_grad=True);r=sequence_loss(x,x.detach().clone(),x.detach().clone(),1,kl_beta=0)
  self.assertTrue(bool((r['ratio']==1).all()));r['loss'].backward();self.assertGreater(float(x.grad.abs().sum()),0)
 def test_full_and_local_same_logp_coefficient_mass(self):
  import torch
  from srf5_ref.objective import sequence_loss
  grads=[]
  for scope in ('full','selected'):
   x=torch.tensor([-1.,-2.,-3.,-4.],requires_grad=True);r=sequence_loss(x,x.detach().clone(),x.detach().clone(),0,.25,[False,True,True,False],scope,kl_beta=0);r['loss'].backward();grads.append(float(x.grad.sum()))
  self.assertAlmostEqual(grads[0],grads[1])
 def test_nonfinite_fails_closed(self):
  import torch
  from srf5_ref.objective import sequence_loss
  x=torch.tensor([float('nan')],requires_grad=True)
  with self.assertRaises(FloatingPointError):sequence_loss(x,torch.tensor([-1.]),torch.tensor([-1.]),1)
 def test_mask_type_checked(self):
  import torch
  from srf5_ref.objective import sequence_loss
  x=torch.tensor([-1.,-2.],requires_grad=True)
  with self.assertRaises(ValueError):sequence_loss(x,x.detach(),x.detach(),1,.25,[0,1],'selected')


class InterfaceTest(unittest.TestCase):
 def test_interface_one_factor_and_gold(self):
  from srf5_ref.generator import make_interface_bank
  ts=make_interface_bank(2,191999)
  for j in range(0,len(ts),4):
   a,b,c,d=ts[j:j+4]
   self.assertEqual(a['world'],b['world']);self.assertEqual(a['query'],b['query']);self.assertNotEqual(a['render_style'],b['render_style'])
   self.assertEqual(a['world'],c['world']);self.assertEqual(c['protocol'],'natural_work_v2')
   self.assertEqual(a['query'],d['query']);self.assertEqual(evaluate_a(d['world'],d['query']),evaluate_b(d['world'],d['query']))
 def test_pilot_schedule(self):
  ts=make_bank('pilot',8,8881);ss=build_schedule(ts,99,'b2',0,expected_roots_per_family=8)
  self.assertEqual(len(ss),12);self.assertEqual(sum(len(x['groups']) for x in ss),192)
 def test_product_labels_no_intrinsic_calendar_order(self):
  t=make_root('namecheck','INTERVAL',0,2299)[0]
  self.assertTrue(set(t['world']['categories'])<=set(('Mug','Lamp','Desk','Vase','Chair','Clock','Tray','Book')))

class RobustnessTest(unittest.TestCase):
 def test_case_and_whitespace_equivalence(self):
  from srf5_ref.measurement import verify_selection
  self.assertEqual(verify_selection({'status':'KNOWN','selected':[' mug ','LAMP']},['Mug','Lamp']),'PASS')
  self.assertEqual(verify_selection({'status':'KNOWN','selected':[' mug ','MUG']},['Mug']),'FAIL')
 def test_large_decimal_fails_before_fraction_allocation(self):
  from srf5_ref.measurement import rational
  from decimal import Decimal
  with self.assertRaises(ValueError):rational(Decimal('1e100000000'))
 def test_empty_gate_cannot_pass(self):
  from srf5_ref.gates import check_calibration,check_pilot
  self.assertEqual(check_calibration({})['status'],'FAIL');self.assertEqual(check_pilot({})['status'],'FAIL')
 def test_unused_auxiliary_rejected(self):
  import torch
  from srf5_ref.objective import sequence_loss
  x=torch.tensor([-1.],requires_grad=True)
  with self.assertRaises(ValueError):sequence_loss(x,x.detach(),x.detach(),1,.1,scope='none')

class FinalDeliveryTest(unittest.TestCase):
 def test_malformed_and_nonfinite_gate_metrics_fail(self):
  from srf5_ref.gates import check_calibration,check_pilot
  self.assertEqual(check_calibration({'unique_reviewed_responses':None,'blocks':None})['status'],'FAIL')
  self.assertEqual(check_pilot({'blocks':{'b0':{'windows':[{'LOCAL_nonzero_groups':None}], 'S_known_fraction':float('inf')}}})['status'],'FAIL')
 def test_collision_fixture_J_success_S_failure(self):
  from pathlib import Path
  fx=json.loads((Path(__file__).parent/'collision_J1_SFAIL.json').read_text())
  score=score_response(fx['raw_response'],fx['world'],fx['query'])
  self.assertEqual(score['J'],1);self.assertEqual(score['S'],'FAIL')
  self.assertEqual([x['gold_answer'] for x in fx['relations']],[90,96,90,84])



class StatisticalValidationTest(unittest.TestCase):
 def test_missing_repeat_or_nonfinite_rejected(self):
  from srf5_ref.statistics import summarize_effect,training_repeat_t
  for fn in (summarize_effect,training_repeat_t,paired_hierarchical_bootstrap):
   with self.assertRaises(ValueError):fn(np.full((3,3,6,2),float('nan')))
   with self.assertRaises(ValueError):fn(np.zeros((3,2,6,2)))
 def test_empty_relation_panel_rejected(self):
  with self.assertRaises(ValueError):relation_metrics([], [1,2,3,4],[0,1,2,3])
  with self.assertRaises(ValueError):relation_metrics([[1,2]], [1,2,3,4],[0,1,2,3])

if __name__=='__main__':unittest.main()
