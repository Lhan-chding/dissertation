"""Synthetic seals test arithmetic and fail-closed evidence; no model calls."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from adapter.seal_path import path_parts, put_json, verify_groups, verify_token_vectors, export_path, publish_receipt
from adapter.training import native_j_advantages, trajectory_hash


def vectors():
    records, traces = [], []
    for i in range(128):
        row = dict(row_id=f'r{i}', request_id=f'r{i}', qid=f'q{i//8}', tokens=[11, 12],
            sampler_logprobs=[-1., -2.], prompt_tensor_hash='a'*64, native_prompt_inputs={'input_ids': [[9]]},
            model_input={'prompt': 'fixture', 'image_file':'fixture.png','image_sha256':'b'*64},
            protocol='structured_v4',policy_hash='c'*64,reference_hash='d'*64,source_sha256='e'*64,
            seed=i,input_id='f'*64,assistant_prefill='{')
        row['token_path_sha256'] = trajectory_hash(row)
        records.append(row)
        traces.append(dict(row_id=row['row_id'],qid=row['qid'],tokens=row['tokens'],
            sampler_logprobs=row['sampler_logprobs'],train_logprobs=[-1.,-2.],
            reference_logprobs=[-1.,-2.],token_path_sha256=row['token_path_sha256']))
    update = dict(probability_vectors=traces,numerics=dict(sampler_train_mean_absolute_difference=0.,
        sampler_train_max_absolute_difference=0.,tis_truncated_fraction=0.,kl_unweighted=0.,
        gradient_norm_before_clip=0.,gradient_norm_after_clip=0.))
    return records, update


def groups(j):
    plan, actual, scores = [], [], {}
    for i in range(16):
        group=dict(group_id=f'g{i}',qid=f'q{i}',root_id=f'root{i}',seeds=list(range(i*8,i*8+8)),
                   request_ids=[f'r{d}' for d in range(i*8,i*8+8)])
        plan.append(group)
        actual.append(dict(group,J=[j]*8,native=native_j_advantages([j]*8),aux=[0.]*8))
        for rid in group['request_ids']: scores[rid]={'J':j}
    return {'groups':plan},{'groups':actual},scores


class SealPathTests(unittest.TestCase):
    def test_exact_registered_path_names(self):
        self.assertEqual(path_parts('b1-11032-D'),('b1',11032,'D'))
        for bad in ('b2-11032-D','b0-11033-R','../b0-11031-R','b0-11031-M'):
            with self.assertRaises(PermissionError): path_parts(bad)
    def test_constant_j_is_valid_observation(self):
        for j in (0,1): verify_groups(*groups(j))
    def test_wrong_reward_or_aux_blocks(self):
        plan, update, scores = groups(0)
        update['groups'][0]['J'][0]=1
        with self.assertRaises(PermissionError): verify_groups(plan, update, scores)
        plan, update, scores=groups(1)
        update['groups'][0]['aux'][0]=.1
        with self.assertRaises(PermissionError): verify_groups(plan,update,scores)
    def test_original_vectors_pass_and_token_changes_fail(self):
        rows, update = vectors()
        self.assertEqual(verify_token_vectors(rows, update)['tokens'],256)
        update['probability_vectors'][0]['tokens']=[12,11]
        with self.assertRaises(PermissionError): verify_token_vectors(rows,update)
    def test_false_numeric_summary_or_nonfinite_blocks(self):
        rows, update=vectors()
        update['probability_vectors'][0]['train_logprobs'][0]=0.
        with self.assertRaises(PermissionError): verify_token_vectors(rows,update)
        rows, update=vectors()
        update['probability_vectors'][0]['reference_logprobs'][0]=float('nan')
        with self.assertRaises(PermissionError): verify_token_vectors(rows,update)
    def test_kl_is_equal_sequence_mean(self):
        rows,update=vectors()
        # Different sequence lengths: KL gate must not become token weighted.
        rows[0]['tokens']=[11]
        rows[0]['sampler_logprobs']=[-1.]
        rows[0]['token_path_sha256']=trajectory_hash(rows[0])
        update['probability_vectors'][0].update(tokens=[11],sampler_logprobs=[-1.],train_logprobs=[-1.],
            reference_logprobs=[0.],token_path_sha256=rows[0]['token_path_sha256'])
        update['numerics']['kl_unweighted']=(math.expm1(1)-1)/128
        self.assertAlmostEqual(verify_token_vectors(rows,update)['kl_unweighted'],(math.expm1(1)-1)/128)
    def test_immutable_output_and_absent_path_no_formal_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            path=root/'evidence.json'
            put_json(path,{'a':1}); put_json(path,{'a':1})
            with self.assertRaises(PermissionError): put_json(path,{'a':2})
            with self.assertRaisesRegex(PermissionError,'Training path absent'):
                export_path(root,'b0-11031-R')
            self.assertFalse((root/'run').exists())
            seal=root/'PATH_RECEIPT.json'
            publish_receipt(seal,{'done':True})
            publish_receipt(seal,{'done':True})
            with self.assertRaises(PermissionError): publish_receipt(seal,{'done':False})
            self.assertFalse(list(root.glob('.PATH_RECEIPT-*.tmp')))

if __name__=='__main__': unittest.main()
