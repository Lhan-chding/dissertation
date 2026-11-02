"""Paired root effects. Fixed three-parent panel; no training-seed inference.

The p-value is a centered cluster-bootstrap approximation, not a Fisher-exact
randomization test. Resampling keeps naming views and parents within each root.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

PRIMARY = ('D_ENUM','I_ALT','D_REV','D_SPEC')
SECONDARY = ('D_NOLIST','D_NAMES','D_REF','D_PRINT','D_PLACEBO')


def _contrast(v, name):
    def avg(c): return (v['ALIGNED',c]+v['CONFLICT',c])/2
    if name=='D_ENUM': return avg('C0')-avg('C_ENUM')
    if name=='I_ALT':
        return (v['ALIGNED','C_ALT']-v['CONFLICT','C_ALT'])-(v['ALIGNED','C_PRINT']-v['CONFLICT','C_PRINT'])
    if name=='D_REV': return avg('P_STABLE')-avg('P_REVTIE')
    if name=='D_SPEC': return avg('P_PLACEBO')-avg('P_REVTIE')
    if name=='D_NOLIST': return avg('C0')-avg('C_NOLIST')
    if name=='D_NAMES': return avg('C_ENUM')-avg('C_NOLIST')
    if name=='D_REF': return avg('C_REF')-avg('C0')
    if name=='D_PRINT': return avg('C_PRINT')-avg('C0')
    if name=='D_PLACEBO': return avg('P_STABLE')-avg('P_PLACEBO')
    raise ValueError(name)


def root_effects(rows: list[dict], metric: str, name: str, parents: list[str]) -> tuple[list[str],list[str],np.ndarray]:
    buckets=defaultdict(list);structures={};roots=set()
    for row in rows:
        if row['split']!='MAIN': raise ValueError('Do not pool smoke with main')
        if row['parent'] not in parents: continue
        key=(row['root_id'],row['parent'],row['rename'],row['condition'])
        buckets[key].append(float(row[metric]))
        structures[row['root_id']]=row['structure']; roots.add(row['root_id'])
    for key, vals in buckets.items():
        expected = 2 if key[3].startswith('C') else 1
        if len(vals) != expected: raise ValueError('Wrong registered draw denominator')
    ordered=sorted(roots); effects=[]
    for root in ordered:
        ps=[]
        for parent in parents:
            v={(rename,cond):float(np.mean(vals)) for (r,p,rename,cond),vals in buckets.items() if r==root and p==parent}
            if len(v)!=18: raise ValueError('Missing condition/naming cells')
            ps.append(_contrast(v,name))
        effects.append(float(np.mean(ps)))
    return ordered,[structures[r] for r in ordered],np.array(effects,dtype=float)


def paired_bootstrap(values: np.ndarray, strata: list[str], *, B: int=10000, seed: int=2026110202) -> dict:
    values=np.asarray(values,dtype=float)
    if len(values)!=len(strata) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid root effects')
    rng=np.random.default_rng(seed); draws=np.zeros(B,dtype=float)
    for label in sorted(set(strata)):
        ids=np.array([i for i,x in enumerate(strata) if x==label])
        sampled=rng.choice(ids,size=(B,len(ids)),replace=True)
        draws += values[sampled].sum(axis=1)/len(values)
    estimate=float(values.mean())
    constant=np.allclose(values,values[0],rtol=0,atol=1e-15)
    if constant and abs(estimate)>1e-15:
        p=None;limitation='CONSTANT_NONZERO_SAMPLE_BOOTSTRAP_CANNOT_ESTIMATE_UNSEEN_ROOT_VARIATION'
    else:
        p=1.0 if np.allclose(values,0,rtol=0,atol=1e-15) else float((1+np.sum(np.abs(draws-estimate)>=abs(estimate)))/(B+1))
        limitation=None
    return {'estimate':estimate,'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'ci99_375':np.quantile(draws,[.003125,.996875]).tolist(),
            'p_approx_centered_bootstrap':p,'n_roots':len(values),'bootstrap_B':B,
            'limitation':limitation,'root_effect_sd':float(values.std(ddof=1)) if len(values)>1 else None}


def holm(pvalues: list[float | None]) -> list[float | None]:
    # Keep the registered family size even when one contrast is non-estimable.
    m=len(pvalues);order=sorted((p,i) for i,p in enumerate(pvalues) if p is not None)
    out=[None]*m;previous=0.0
    for rank,(p,i) in enumerate(order):
        previous=max(previous,min(1.0,(m-rank)*p));out[i]=previous
    return out


def descriptive(rows: list[dict]) -> list[dict]:
    groups=defaultdict(list)
    for row in rows:
        groups[(row['parent'],row['condition'],row['rename'])].append(row)
    out=[]
    for key,group in sorted(groups.items()):
        N=len(group);known=sum(r['S_known'] for r in group);hit=sum(r['S_hit'] for r in group)
        out.append(dict(zip(('parent','condition','rename'),key))|{
            'N':N,'A_correct':sum(r['A'] for r in group),'A_rate':sum(r['A'] for r in group)/N,
            'S_known':known,'S_pass':hit,'S_confirmed_rate':hit/N,'S_unknown':N-known,
            'S_identification_interval':[hit/N,(hit+N-known)/N],
            'S_pass_given_known':hit/known if known else None,
            'V_rate':sum(r['V_probe'] for r in group)/N,
            'rank_order_known':sum(r['rank_order_known'] for r in group),
            'rank_value_true_known':sum(r['rank_values_true'] is not None for r in group),
            'rank_value_true':sum(r['rank_values_true']==1 for r in group),
            'reversed_choice_rate':sum(r['reversed_boundary_choice_hit'] for r in group)/N,
            'enumeration_choice_rate':sum(r['enumeration_choice_hit'] for r in group)/N,
            'alphabetical_choice_rate':sum(r['alphabetical_choice_hit'] for r in group)/N})
    return out
