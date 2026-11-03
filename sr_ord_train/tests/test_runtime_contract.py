import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import runtime_contract as c


ROOT = Path(__file__).resolve().parents[1]


def hx(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def put(base, name, text):
    (base/name).write_text(text, encoding='utf-8')
    return {'path':name,'sha256':c.digest(base/name)}


def common(scope):
    return dict(schema='order-runtime-v1',scope=scope,manifest_sha256=hx('manifest'),snapshot=c.SNAPSHOT,
                dtype='float32',reward='native_srf6_J_only',protocol='structured_v4',aux_enabled=False,prefix_training=False,has_gpu_hour_stopping=False)


def numeric():
    return dict(sampler_train_mean_absolute_difference=.00012,sampler_train_max_absolute_difference=.23,
                tis_truncated_fraction=0.,kl_unweighted=.04,gradient_norm_before_clip=.34,gradient_norm_after_clip=.34)


class TestRuntimeContract(unittest.TestCase):
    def test_real_native_differential_fixture(self):
        a=c.native_fixtures(ROOT/'vendor'/'native_src')
        self.assertEqual(a['optimizer_updates'],0)
        self.assertEqual(a['scorer'],'srf6-native-measurement-v1')
        self.assertFalse(a['new_server_binding'])

    def test_numerical_gate_catches_reference_tail_independent_of_sampler_average(self):
        c.check_numerics(numeric())
        for k, value in [('kl_unweighted',5332670.04),('gradient_norm_before_clip',622797312.),
                         ('sampler_train_max_absolute_difference',28.501),('gradient_norm_after_clip',1.1),
                         ('sampler_train_mean_absolute_difference',float('nan'))]:
            with self.subTest(k=k),self.assertRaises(ValueError): c.check_numerics(dict(numeric(),**{k:value}))

    def test_smoke_checks_frozen_slots_no_updates_and_no_extra_draw(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);r=common('DEV_SMOKE'); expected=[];groups=[]
            for g in range(24):
                e=dict(group_id=f'g{g}',parent=c.PARENTS[g%2],qid=f'q{g}',root_id=f'r{g}',
                       seeds=list(range(g*8,g*8+8)),request_ids=[f'{g}-{i}' for i in range(8)])
                expected.append(e); groups.append(dict(e,policy_before_sha256=hx(g%2),policy_after_sha256=hx(g%2)))
            r.update(optimizer_steps=0,parameter_updates=0,generated_responses=192,groups=groups,
                     evidence={k:put(b,k,k) for k in ('raw','scorer_receipt','token_replay','native_binding','processor_checks')})
            self.assertEqual(c.check_smoke(r,hx('manifest'),b,expected)['responses'],192)
            with self.assertRaises(ValueError):c.check_smoke(dict(r,optimizer_steps=1),hx('manifest'),b,expected)
            bad=copy.deepcopy(r);bad['groups'][7]['request_ids'][-1]=groups[6]['request_ids'][0]
            with self.assertRaises(ValueError):c.check_smoke(bad,hx('manifest'),b,expected)

    def test_path_counts_on_policy_chain_and_commit_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);r=common('TRAIN_PATH');plan=[];updates=[]
            for s in range(1,25):
                groups=[]
                for i in range(16):
                    groups.append(dict(group_id=f'g{s}:{i}',qid=f'q{s}:{i}',root_id=f'r{s}:{i}',seeds=[i*8+j for j in range(8)],request_ids=[f'{s}/{i}/{d}' for d in range(8)]))
                plan.append(dict(step=s,groups=groups));updates.append(dict(step=s,policy_version_sampled=s-1,policy_before_sha256=hx(s-1),policy_after_sha256=hx(s),
                   reference_sha256=hx(0),optimizer_steps=1,responses=128,aux_nonzero_rows=0,prefix_rows=0,precommit_gate='PASS_BEFORE_ADAM',
                   groups=[dict(g,J=[0,1]*4) for g in groups],numerics=numeric(),
                   evidence={k:put(b,f'{s}_{k}',f'{s}{k}') for k in ('raw','token_trace','native_update','commit')}))
            r.update(parent='b1',arm='D',seed=11032,adapter_config_sha256=c.PARENT_FILES['b1'][0],adapter_file_sha256=c.PARENT_FILES['b1'][1],
                     parent_tensor_sha256=hx(0),reference_tensor_sha256=hx(0),total_optimizer_steps=24,total_generated_responses=3072,
                     evaluation_policy_sha256={'12':hx(12),'24':hx(24)},updates=updates,
                     evidence={k:put(b,k,k) for k in ('native_binding','recipe','zero_checkpoint','checkpoint_12','checkpoint_24')})
            r['evidence']['plan']=put(b,'plan.json',json.dumps(plan));self.assertEqual(c.check_path(r,hx('manifest'),b,r['evidence']['plan']['sha256'])['steps'],24)
            with self.assertRaises(ValueError):c.check_path(r,hx('manifest'),b,hx('some other plan'))
            bad=copy.deepcopy(r);bad['updates'][11]['policy_version_sampled']=0
            with self.assertRaises(ValueError):c.check_path(bad,hx('manifest'),b,r['evidence']['plan']['sha256'])
            bad=copy.deepcopy(r);bad['updates'][1]['aux_nonzero_rows']=2
            with self.assertRaises(ValueError):c.check_path(bad,hx('manifest'),b,r['evidence']['plan']['sha256'])
            (b/'24_commit').write_text('different bytes')
            with self.assertRaises(ValueError):c.check_path(r,hx('manifest'),b,r['evidence']['plan']['sha256'])

if __name__=='__main__':unittest.main()
