from __future__ import annotations
import math
from typing import Iterable

def wilson_lower(correct: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0 or not 0 <= correct <= n: raise ValueError('Invalid count')
    p=correct/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def require_disjoint_worlds(manifests: dict[str,list[dict]]) -> None:
    seen={}
    for split,rows in manifests.items():
        for row in rows:
            # Multiple prompts of one world must stay in a single split.
            for key in ('world_id','base_content_sha256'):
                value=row[key]; index=(key,value)
                if index in seen and seen[index]!=split: raise ValueError(f'Leakage: {key} {value}')
                seen[index]=split

def c_gate(report: dict) -> dict:
    failures=[]
    for k in ('native_advantage_parity','binary_J_verified','label_lineage_verified',
              'heldout_isolation_verified','human_audit_complete','sampler_tests_passed'):
        if report.get(k) is not True: failures.append(k)
    for field in ('perception','selection'):
        a=report.get('audit',{}).get(field,{})
        n=a.get('reward_positive_n',0); good=a.get('reward_positive_correct',0)
        if n<80 or not 0<=good<=n or (n and wilson_lower(good,n)<.95): failures.append(f'audit_{field}')
    for family in ('INTERVAL','TOPK'):
        x=report.get('signal',{}).get(family,{})
        if x.get('select_vs_evid_changed_groups',0)<24: failures.append(f'signal_{family}')
        if x.get('independent_worlds',0)<8: failures.append(f'worlds_{family}')
    for block in ('b0','b1','b2'):
        if report.get('signal_by_block',{}).get(block,0)<8: failures.append(f'signal_{block}')
    if report.get('hard_fixture_false_positive',1)!=0: failures.append('hard_fixture_false_positive')
    return {'decision':'RUN_C' if not failures else 'BLOCK_C', 'failed_requirements':failures,
            'interpretation':'Operational evidence sufficiency, not a test that semantics is useful/useless.'}
