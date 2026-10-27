"""Read-only census summarization. No fabricated rollouts and no model calls.
Mandatory status fields are validated. One group must belong to one qid/policy.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from .credit import credit
from .core import seed_for

REQUIRED=('parent','factor','group_id','qid','root_id','family','draw','policy_hash','A','J','V','finished','S','B')

def summarize(rows:list[dict],master:int=2026102701)->dict:
    if not rows:raise ValueError('no actual census records')
    groups=defaultdict(list);cells=defaultdict(list);seen=set()
    for r in rows:
        if any(k not in r for k in REQUIRED):raise ValueError('missing required census fields')
        if r['factor'] not in ('P0','P1'):raise ValueError('factor')
        for k in ('A','J','V','finished'):
            if r[k] not in (0,1):raise ValueError('binary '+k)
        for k in ('S','B'):
            if r[k] not in (None,0,1):raise ValueError('binary/UNKNOWN '+k)
        key=(r['parent'],r['factor'],r['group_id']);dk=key+(r['draw'],)
        if dk in seen:raise ValueError('duplicate group/draw identity')
        seen.add(dk);groups[key].append(r);cells[(r['parent'],r['factor'])].append(r)
    state={};factor_out={};details=[]
    for (p,f),rr in cells.items():
        n=len(rr);state[f'{p}/{f}']={'n':n,'V_rate':sum(x['V'] for x in rr)/n,
         'S_known_rate':sum(x['S'] is not None for x in rr)/n,'B_known_rate':sum(x['B'] is not None for x in rr)/n,
         'A_rate':sum(x['A'] for x in rr)/n,'J_rate':sum(x['J'] for x in rr)/n,
         'A_J_S_B_cross':dict(Counter(f"A{x['A']}/J{x['J']}/S{x['S']}/B{x['B']}" for x in rr))}
    for (p,f,gid),rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda x:x['draw'])
        if len(rr)!=8 or set(x['draw'] for x in rr)!=set(range(8)):raise ValueError('G8 incomplete')
        if any(len({x[k] for x in rr})!=1 for k in ('qid','root_id','family','policy_hash')):raise ValueError('mixed group identity')
        s=seed_for(master,'census_shuffle',p,f,gid);cond=credit(rr,'COND');shuf=credit(rr,'SHUF',seed=s)
        exch=any(len({cond['candidate'][i] for i in ids})>1 for ids in cond['strata'])
        details.append({'parent':p,'factor':f,'group_id':gid,'root_id':rr[0]['root_id'],'family':rr[0]['family'],
            'J_class':'all_zero' if not any(x['J'] for x in rr) else 'all_one' if all(x['J'] for x in rr) else 'mixed',
            'nonzero':cond['nonzero'],'exchangeable':exch,'changed':shuf['permutation_changed'],
            'credit_L1':sum(abs(x) for x in cond['candidate'])})
    for f in ('P0','P1'):
        allg=[x for x in details if x['factor']==f];nz=[x for x in allg if x['nonzero']]
        factor_out[f]={'groups':len(allg),'nonzero_groups':len(nz),'roots':len({x['root_id'] for x in nz}),
          'families':len({x['family'] for x in nz}),'parents':len({x['parent'] for x in nz}),
          'exchangeable_groups':sum(x['exchangeable'] for x in allg),'actually_changed_groups':sum(x['changed'] for x in allg),
          'nonzero_by_J_class':dict(Counter(x['J_class'] for x in nz))}
    ch=[x for x in details if x['changed']]
    return {'rows':len(rows),'cells':state,'factors':factor_out,'total_changed_groups':len(ch),
       'total_changed_roots':len({x['root_id'] for x in ch}),'groups':details,
       'scope':'read-only summary; not a signed gate, no training effect, cached factor duplicates are not independent samples'}
