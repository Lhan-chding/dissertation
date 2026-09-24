"""Independent mathematical/plan reference, not the Qwen production runner."""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Mapping, Sequence
import numpy as np

EVENTS = ('X','S','W','I')
RECIPES = {
    'R0':(2,0,0,0),'R1':(0,2,0,0),'R2':(2,0,1,0),'R3':(2,0,0,1),
    'R4':(2,0,.5,.5),'R5':(0,2,1,0),'R6':(0,2,0,1),'R7':(0,2,.5,.5)}


def event_probs(x: float, a: float, v: float) -> np.ndarray:
    if not all(math.isfinite(t) for t in (x,a,v)) or not 0 <= x <= a <= v <= 1:
        raise ValueError('Expected 0 <= X <= A <= V <= 1')
    return np.array([x,a-x,v-a,1-v],dtype=float)


def rewards(event: str, c: float) -> np.ndarray:
    if event not in EVENTS or not math.isfinite(c) or not 0 <= c <= 1:
        raise ValueError('Unknown event or relation score')
    if (event=='I' and c!=0) or (event=='X' and c!=1):
        raise ValueError('Semantic consistency violated')
    return np.array([event=='X',event in ('X','S'),event!='I',c],float)


def _world(x):
    return isinstance(x,(list,tuple)) and len(x)==4 and all(type(v) is int for v in x)


def repair(observed: Sequence[int], truth: Sequence[int], prediction) -> dict:
    if not _world(observed) or not _world(truth):
        raise ValueError('Four integer coordinates required')
    changed=[k for k in range(4) if observed[k]!=truth[k]]
    if len(changed)!=1:
        raise ValueError('Exactly one corrupted coordinate required')
    if not _world(prediction):
        return {'valid':False,'F':None,'B':None,'M':None,'coord_accuracy':0.,'penalty':1.}
    j=changed[0]
    f=int(prediction[j]==truth[j])
    b=sum(prediction[k]!=truth[k] for k in range(4) if k!=j)
    m=sum(prediction[k]!=observed[k] for k in range(4))
    return {'valid':True,'F':f,'B':b,'M':m,'coord_accuracy':(f+3-b)/4.,'penalty':b/3.}


def training_reward(recipe: str,event: str,c: float,damage_penalty: float=0.) -> float:
    r=rewards(event,c)
    if recipe=='DIRECT_REPAIR_R4':
        if not math.isfinite(damage_penalty) or not 0 <= damage_penalty <= 1:
            raise ValueError('Bounded damage penalty required')
        if event=='I' and damage_penalty!=1:
            raise ValueError('Invalid output receives damage penalty one')
        return float(np.dot(RECIPES['R4'],r)-.5*damage_penalty)
    if recipe not in RECIPES:
        raise ValueError('Unknown static recipe')
    return float(np.dot(RECIPES[recipe],r))


def advantages(values, epsilon=1e-4):
    r=np.asarray(values,dtype=float)
    if r.ndim!=1 or len(r)<2 or not np.isfinite(r).all() or epsilon<=0:
        raise ValueError('Nonempty finite group and positive epsilon required')
    d=r-r.mean()
    return d/np.sqrt(np.mean(d*d)+epsilon*epsilon)


def choose(predictions: Mapping[str,float],default: str,tau=.005):
    if not predictions or default not in predictions or not math.isfinite(tau) or tau<0:
        raise ValueError('Valid default, predictions and tolerance required')
    if not all(math.isfinite(x) for x in predictions.values()):
        raise ValueError('Nonfinite utility')
    top=min(predictions,key=lambda k:(-predictions[k],k))
    selected=default if predictions[top]-predictions[default]<=tau else top
    return {'recipe':selected,'predicted_best':top,'margin_to_default':predictions[top]-predictions[default],
            'status':'POINT_DECISION_NOT_SAFETY_CERTIFICATE'}


def sqrt_histogram(values):
    h=np.asarray(values,float)
    if h.ndim!=1 or not len(h) or not np.isfinite(h).all() or (h<0).any() or h.sum()<=0:
        raise ValueError('Nonnegative nonempty histogram required')
    return np.sqrt(h/h.sum())


def krr_fit(k,y,alpha):
    k=np.asarray(k,float); y=np.asarray(y,float)
    if k.ndim!=2 or k.shape[0]!=k.shape[1] or not len(k) or len(y)!=len(k):
        raise ValueError('Aligned square kernel and targets required')
    if not np.isfinite(k).all() or not np.isfinite(y).all() or not np.allclose(k,k.T):
        raise ValueError('Finite symmetric kernel required')
    if not math.isfinite(alpha) or alpha<=0:
        raise ValueError('Positive regularization required')
    if np.linalg.eigvalsh(k).min() < -1e-9:
        raise ValueError('PSD kernel required')
    mean=y.mean(axis=0)
    beta=np.linalg.solve(k+len(k)*alpha*np.eye(len(k)),y-mean)
    return {'mean':mean,'beta':beta}


def check_lineage_splits(registry):
    seen={}
    for item in registry:
        lid=item['lineage_id']; role=item['role']
        if lid in seen and seen[lid]!=role:
            raise ValueError('Lineage leakage')
        seen[lid]=role
    return seen


def predecision_packet(data):
    allowed={'origin_id','lineage_id','source_recipe','step','current','history','known_training_metadata','feature_level'}
    if set(data)-allowed:
        raise ValueError('Forbidden future/path/outcome fields')
    if not {'origin_id','current','history','step'}<=set(data):
        raise ValueError('Incomplete predecision packet')
    def check_nested(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                text=str(key).lower()
                if text.startswith(('future_', 'outcome_', 'post_update_')) or text in {'outcomestore','future','outcomes'}:
                    raise ValueError('Nested future/outcome field')
                check_nested(value)
        elif isinstance(obj,(tuple,list)):
            for value in obj: check_nested(value)
    check_nested(data)
    return dict(data)


def unique_branches(origin, decisions, baselines, repeats=(1,2)):
    actions=set(decisions)|set(baselines)
    return sorted((origin,a,r) for a in actions for r in repeats)


def collapse_lineage(values):
    groups=defaultdict(list)
    for row in values:
        groups[row['lineage_id']].append(float(row['difference']))
    return {k:float(np.mean(v)) for k,v in groups.items()}


def plan_n(sd, effect=.01, choices=(12,16,20,24,32,48)):
    if not math.isfinite(sd) or sd<0 or not math.isfinite(effect) or effect<=0:
        raise ValueError('Finite SD and positive effect required')
    req=math.ceil(((1.96+.8416212336)*sd/effect)**2)
    selected=next((n for n in choices if n>=req),choices[-1])
    return {'required_approx':req,'selected':selected,'target_power_not_reached_by_plan':req>selected,
            'approx_detectable_effect':(1.96+.8416212336)*sd/math.sqrt(selected)}


def mc_se_upper(prompts,draws,lineages,repeats):
    args=(prompts,draws,lineages,repeats)
    if any(type(x) is not int or x<=0 for x in args):
        raise ValueError('Positive integer sizes required')
    return math.sqrt(.5/math.prod(args))


def plan_draws(lineages,repeats=2,prompts=288,effect=.01,choices=(16,32,64)):
    for m in choices:
        se=mc_se_upper(prompts,m,lineages,repeats)
        if se<=effect/4:
            return {'draws':m,'mc_se_upper':se,'meets_target':True}
    return {'draws':choices[-1],'mc_se_upper':mc_se_upper(prompts,choices[-1],lineages,repeats),
            'meets_target':False}


def information_value(utilities,probabilities,observed_groups):
    u=np.asarray(utilities,float); p=np.asarray(probabilities,float)
    if u.ndim!=2 or p.shape!=(len(u),) or len(observed_groups)!=len(u):
        raise ValueError('State/action utilities and probabilities must align')
    if (p<0).any() or not np.isclose(p.sum(),1) or not np.isfinite(u).all():
        raise ValueError('Probability distribution required')
    total=0.
    for g in set(observed_groups):
        ids=np.array([i for i,x in enumerate(observed_groups) if x==g],int)
        mass=p[ids].sum()
        if mass>0:
            total+=float((p[ids,None]*u[ids]).sum(axis=0).max())
    return total


def workload(test_n=12,draws=16):
    if type(test_n) is not int or test_n<12 or test_n%4 or test_n>48:
        raise ValueError('Balanced test cohort 12..48 required')
    source=16*96+test_n*64
    development=32*11*32
    final=test_n*2*9*32
    return {'source_updates':source,'development_updates':development,'test_updates_upper':final,
            'all_updates_upper':source+development+final,
            'all_training_outputs_upper':32*(source+development+final),
            'test_H32_outputs_upper':test_n*2*9*288*draws}


def stratified_summary(rows,expected_cells):
    """Paired differences, one row per independent lineage; fixed equal cell weights."""
    groups=defaultdict(list)
    seen=set()
    for row in rows:
        if row['lineage_id'] in seen: raise ValueError('Average repetitions before inference')
        seen.add(row['lineage_id'])
        d=float(row['difference'])
        if not math.isfinite(d): raise ValueError('Nonfinite difference')
        groups[row['cell']].append(d)
    if set(groups)!=set(expected_cells) or any(len(v)<2 for v in groups.values()):
        raise ValueError('Each registered cell requires two or more independent lineages')
    w=1/len(expected_cells)
    estimates=[]; variance_terms=[]; denom=[]
    for c in expected_cells:
        d=np.asarray(groups[c]); v=w*w*float(d.var(ddof=1))/len(d)
        estimates.append(w*float(d.mean())); variance_terms.append(v); denom.append(v*v/(len(d)-1))
    variance=math.fsum(variance_terms)
    df=variance*variance/math.fsum(denom) if math.fsum(denom)>0 else None
    return {'mean_difference':math.fsum(estimates),'se':math.sqrt(variance),'df':df,
            'status':'NO_OBSERVED_VARIABILITY' if variance==0 else 'APPROXIMATE_STRATIFIED_INFERENCE',
            'independent_lineages':len(seen)}
