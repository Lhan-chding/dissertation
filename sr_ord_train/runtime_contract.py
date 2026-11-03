"""Read-only CPU contract for the ORDER training adapter. No GPU/job/network API.
CLI checks normalized evidence exported by the cluster adapter, not model quality.
See docs/03_RUNTIME.md for the receipt schema, historical bindings and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys

PARENTS = ('b0', 'b1')
ARMS = ('R', 'G', 'D')
SEEDS = (11031, 11032)
SNAPSHOT = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
PARENT_FILES = {
    'b0': ('b4cd04ac2482b3615ccdbba959aa129201ea321b9e59e66245bdd80f41535568',
           '501e9ff25dde77ab5c2042dba82236f74e8a39c5df89b5fb3ed1d2901314e5a0'),
    'b1': ('3df88f113e1e11158d717c836069287002be4b9585f821dc9888e05b8afe2314',
           '5185e60049aa1fa912ee8c49ddb9ff5e6dcd6515fe147bbb94c91e60e134c3ba'),
}
PINS = {
    'sr_f6/measurement.py': '90c26410af3f5526bce5b697138c72e0bf2d0cb58f3557d67fe42cf35db29a04',
    'sr_f5/measurement.py': '1b8e8cb7749f213b3b1e993f7e881ba4016ab6a76e520de750cd6b76ad0a6adb',
    'srf5_ref/measurement.py': '487c572e446c5ac48991477469cf6fef004ecdfc5c2b73eff67ada819f2e9621',
    'srf5_ref/oracle.py': '16ee245806248092d8970ac84e3f75c3f3fc2fe8495c06257d150fa6a17daa36',
    'srf6_ref/core.py': '9b0086b5cd4eea8ddd88669e0c2223c43b7b3a0aeb5c53cf27912caa887cf8bd',
    'sr_f4/task_contract.py': 'a98e73540165cc7c0daeb0f96375341184fd345d0010a052213dec17bbf1be72',
    'sr_f3_ce/contract.py': 'ef8862f340da0d932d878cddc76f398f25b53471218746d3120970ad4dbcce75',
    'sr_f6/runtime.py': 'c71d490c75221a05a9b05516219bf57b592bacd45ea6bba057c462032f1bbcd4',
    'sr_f6/credit.py': 'bf44763453859daa0048128965e70351f29117cbdde4e12b50f0f34380c805f4',
    'sr_f6/objective.py': '7be5b90a6d91261aa9f23803f217de8c42c30de6050157c23ccb7d3b2c880cea',
    'sr_f6/training.py': '4da259da843477d32bab2a5c0aac0a75a443f2a24ecbf45d15e54fc600630ec9',
    'sr_f6/transactions.py': 'eb44c77ac32e96ab1f6c4586753d222d8e9eeed272eac28075c98239dcd713f8',
}


def require(condition, why):
    if not condition:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    def pairs(x):
        require(len(dict(x)) == len(x), 'duplicate JSON keys')
        return dict(x)
    def invalid(x):
        raise ValueError('nonfinite JSON: ' + x)
    return json.loads(Path(path).read_text('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)


def sha(x):
    return type(x) is str and re.fullmatch('[a-f0-9]{64}', x) is not None


def finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def count(x, n):
    return type(x) is int and x == n


def native_fixtures(src, historical_kernel=False):
    src = Path(src).resolve(strict=True)
    checks = PINS if historical_kernel else dict(list(PINS.items())[:7])
    for name, hx in checks.items():
        require(digest(src / name) == hx, 'native source differs: ' + name)
    sys.path.insert(0, str(src))
    try:
        mod = importlib.import_module('sr_f6.measurement')
        for name in list(PINS)[:7]:
            m = importlib.import_module(name[:-3].replace('/', '.'))
            require(Path(m.__file__).resolve() == (src / name).resolve(), 'shadowed dependency: '+ name)
        world = {'categories':['Zed','Ada','Cora','Dan','Eva','Fox','Gus','Hal'],
                 'series':{'Alpha':[80,80,20,21,22,23,24,25], 'Beta':[11,19,30,31,32,33,34,35]}, 'domain':[10,90]}
        query = {'family':'TOPK','selection_series':'Alpha','direction':'largest','k':1,'value':'Beta','aggregate':'sum'}
        facts = [dict(series='Alpha',category=c,value=v) for c,v in zip(world['categories'], world['series']['Alpha'])]
        facts.append(dict(series='Beta',category='Zed',value=11))
        base = dict(evidence=facts,work='Read the ranking facts.',selected=['Zed'],
                    bindings=[dict(category='Zed',operand=11)],aggregate='sum',answer=11)
        def chk(obj, j, **opts):
            got = mod.score_response(json.dumps(obj), world, query, **opts)
            require(got['J'] == j and got['J_M3'] == j, 'native score fixture failed')
            return got
        chk(base,1); chk(dict(base,evidence=list(reversed(facts))),1)
        a=chk(dict(base,selected=['Ada'],bindings=[dict(category='Ada',operand=19)]),1)
        require((a['A'],a['V'],a['S'],a['B']) == (1,1,0,0), 'selector leaked into J')
        chk(dict(base,answer=19),0); chk(dict(base,evidence=facts[1:]),0)
        chk(dict(base,extra=True),0); chk(base,0,finished=False,finish_reason='length')
        bigger={'categories':world['categories']+['Ivy','Jo'], 'series':{k:v+[33,34] for k,v in world['series'].items()}, 'domain':[10,90]}
        for w,q in [(world,dict(query,tie_reference='alphabetical')), (bigger,query)]:
            try: mod.score_response(json.dumps(base),w,q)
            except ValueError: pass
            else: raise ValueError('F6 accepted unsupported reference/world size')
        return {'status':'HISTORICAL_NATIVE_CPU_FIXTURES_PASS','scorer':mod.VERSION,'source_files_checked':len(checks),'historical_kernel_sources_verified':historical_kernel,
                'generated_responses':0,'optimizer_updates':0,'new_server_binding':False}
    finally:
        sys.path.remove(str(src))


def check_numerics(r):
    """New precommit wrapper on native diagnostics; run BEFORE Adam. No auto clipping of KL."""
    for k in ('sampler_train_mean_absolute_difference','sampler_train_max_absolute_difference',
              'tis_truncated_fraction','kl_unweighted','gradient_norm_before_clip','gradient_norm_after_clip'):
        require(finite(r[k]) and r[k] >= 0,'negative or nonfinite '+k)
    require(r['sampler_train_mean_absolute_difference'] <= .01 and r['sampler_train_max_absolute_difference'] <= 5
            and r['tis_truncated_fraction'] <= .01,'sampler/teacher numerical gate')
    # Additional NEW failure sentinels (not historical findings/efficacy criteria).
    require(r['kl_unweighted'] <= 1. and r['gradient_norm_before_clip'] <= 1000., 'extreme KL/gradient: uncommitted technical hold')
    require(r['gradient_norm_after_clip'] <= 1.00001,'global grad clipping failed')
    require(r['sampler_train_mean_absolute_difference'] <= r['sampler_train_max_absolute_difference'], 'mean > max')


def checked_refs(base, refs, names):
    for key in names:
        ref=refs[key]; rel=Path(ref['path']); require(not rel.is_absolute() and '..' not in rel.parts, 'nonlocal artifact ref '+key)
        p=(base/rel).resolve(strict=True)
        require(p.is_relative_to(base.resolve()) and p.is_file() and sha(ref['sha256']) and digest(p)==ref['sha256'], 'artifact digest '+key)


def common(r, manifest_hash, scope):
    require(r['schema']=='order-runtime-v1' and r['scope']==scope and sha(manifest_hash) and r['manifest_sha256']==manifest_hash,'manifest or receipt scope mismatch')
    require(r['snapshot']==SNAPSHOT and r['dtype']=='float32' and r['reward']=='native_srf6_J_only' and r['protocol']=='structured_v4','unexpected objective/runtime')
    require(r['aux_enabled'] is False and r['prefix_training'] is False and r['has_gpu_hour_stopping'] is False,'unregistered scope')


def check_smoke(r, manifest_hash, base, expected):
    common(r,manifest_hash,'DEV_SMOKE')
    require(count(r['optimizer_steps'],0) and count(r['parameter_updates'],0) and count(r['generated_responses'],192),'smoke changed model or count')
    require(len(expected)==24 and len(r['groups'])==24,'exactly 24 predefined DEV groups')
    require(len({g['group_id'] for g in expected})==24,'duplicate assigned group')
    seen=set()
    for a,g in zip(expected,r['groups'],strict=True):
        require(all(g[k]==a[k] for k in ('group_id','parent','qid','root_id','seeds','request_ids')),'unexpected smoke slot')
        require(g['parent'] in PARENTS and len(g['seeds'])==8 and len(set(g['seeds']))==8,'invalid G8')
        require(g['policy_before_sha256']==g['policy_after_sha256'] and sha(g['policy_before_sha256']), 'smoke changed policy')
        ids=g['request_ids'];require(len(ids)==8 and all(type(i) is str and i for i in ids) and not seen.intersection(ids) and len(set(ids))==8,'duplicate/missing smoke draw'); seen.update(ids)
    checked_refs(base,r['evidence'],('raw','scorer_receipt','token_replay','native_binding','processor_checks'))
    return {'status':'DEV_SMOKE_RECEIPT_STRUCTURE_AND_ARTIFACT_DIGESTS_VERIFIED','groups':24,'responses':len(seen),'optimizer_steps':0}


def check_path(r, manifest_hash, base, expected_plan_sha256):
    common(r,manifest_hash,'TRAIN_PATH')
    require(r['parent'] in PARENTS and r['arm'] in ARMS and type(r['seed']) is int and r['seed'] in SEEDS,'unregistered path')
    require((r['adapter_config_sha256'],r['adapter_file_sha256'])==PARENT_FILES[r['parent']], 'parent artifact mismatch')
    ref=r['parent_tensor_sha256'];require(sha(ref) and ref==r['reference_tensor_sha256'], 'wrong frozen parent reference')
    require(count(r['total_optimizer_steps'],24) and count(r['total_generated_responses'],3072) and len(r['updates'])==24,'incorrect path budget')
    checked_refs(base,r['evidence'],('native_binding','plan','recipe','zero_checkpoint','checkpoint_12','checkpoint_24'))
    require(sha(expected_plan_sha256) and r['evidence']['plan']['sha256']==expected_plan_sha256, 'path plan not pinned by independent registration')
    plan=read(base/r['evidence']['plan']['path']); require(isinstance(plan,list) and len(plan)==24,'invalid expected path plan')
    prev=ref;seen=set()
    for step,(s,pl) in enumerate(zip(r['updates'],plan,strict=True),1):
        require(count(s['step'],step) and count(pl['step'],step) and count(s['optimizer_steps'],1) and count(s['responses'],128),'update coverage')
        require(s['policy_before_sha256']==prev and sha(s['policy_after_sha256']) and s['reference_sha256']==ref,'wrong policy/reference chain')
        require(count(s['aux_nonzero_rows'],0) and count(s['prefix_rows'],0),'unregistered intervention')
        require(len(s['groups'])==len(pl['groups'])==16,'missing G8')
        for g,a in zip(s['groups'],pl['groups'],strict=True):
            require(all(g[k]==a[k] for k in ('group_id','qid','root_id','seeds','request_ids')),'actual draw plan differs')
            ids=g['request_ids']; require(len(ids)==8 and all(type(i) is str and i for i in ids) and len(set(ids))==8 and not seen.intersection(ids),'duplicate/missing rollout');seen.update(ids)
            require(len(g['seeds'])==8 and len(set(g['seeds']))==8,'duplicate draw seed inside G8')
            require(len(g['J'])==8 and all(type(j) is int and j in (0,1) for j in g['J']), 'invalid native scores')
        require(s['precommit_gate']=='PASS_BEFORE_ADAM' and count(s['policy_version_sampled'], step-1), 'off-policy or missing precommit')
        check_numerics(s['numerics']); checked_refs(base,s['evidence'],('raw','token_trace','native_update','commit'))
        prev=s['policy_after_sha256']
        if step in (12,24):
            require(r['evaluation_policy_sha256'][str(step)] == prev,'wrong evaluation checkpoint')
    return {'status':'PATH_RECEIPT_STRUCTURE_AND_ARTIFACT_DIGESTS_VERIFIED','steps':24,'responses':len(seen),'path':[r['parent'],r['seed'],r['arm']]}


def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('check-native');p.add_argument('--src',required=True);p.add_argument('--historical-kernel',action='store_true',help='also pin full 2026-11-03 archived native kernel sources')
    for cmd in ('check-smoke','check-path'):
        p=sub.add_parser(cmd);p.add_argument('receipt',type=Path);p.add_argument('--manifest-sha256',required=True)
        p.add_argument('--plan-sha256',required=True,help='trusted pre-registration SHA of this path plan or these smoke slots; never copy from the receipt under check')
        if cmd=='check-smoke': p.add_argument('--slots',required=True,type=Path,help='fixed DEV plan JSON array of 24 groups')
    a=ap.parse_args()
    try:
        if a.cmd=='check-native':result=native_fixtures(a.src,a.historical_kernel)
        else:
            r=read(a.receipt);base=a.receipt.resolve().parent
            if a.cmd=='check-smoke':
                require(sha(a.plan_sha256) and digest(a.slots)==a.plan_sha256,'smoke slots not pinned by independent registration')
                result=check_smoke(r,a.manifest_sha256,base,read(a.slots))
            else: result=check_path(r,a.manifest_sha256,base,a.plan_sha256)
    except (KeyError,ValueError,TypeError,AttributeError,ImportError,OSError) as exc:
        ap.exit(2,f'FAIL_CLOSED: {exc}\n')
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))

if __name__=='__main__':main()
