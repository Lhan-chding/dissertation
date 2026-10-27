"""Paired estimates: root averaging first; training repeats are not answer draws.
The fixed-parent t interval is the registered small-sample approximate primary interval,
not a claim that three parental checkpoints are a random model population.
"""
from __future__ import annotations
import math
import numpy as np

CONTRASTS={
 'data_at_J':{'U10':1,'U00':-1},
 'feedback_P0':{'U01':1,'U00':-1},
 'feedback_P1':{'U11':1,'U10':-1},
 'interaction':{'U11':1,'U10':-1,'U01':-1,'U00':1},
 'total_package':{'U11':1,'U00':-1},
 'conditional_vs_global':{'U11':1,'C_GLOBAL':-1},
 'semantic_correspondence_P1':{'U11':1,'C_SHUF':-1}}

def factorial(values:dict[str,float])->dict:
    return {k:sum(values[a]*w for a,w in c.items()) for k,c in CONTRASTS.items() if all(a in values for a in c)}

def holm(pvals:dict[str,float])->dict:
    if any(not 0<=v<=1 for v in pvals.values()):raise ValueError('invalid p')
    ordered=sorted(pvals,key=pvals.get);out={};running=0.
    for i,k in enumerate(ordered):
        running=max(running,min(1.,(len(ordered)-i)*pvals[k]));out[k]=running
    return out

def fixed_parent_interval(deltas:np.ndarray,alpha:float=.05)->dict:
    """Shape parents x independent repeats, evaluated on a fixed common test panel."""
    from scipy.stats import t
    d=np.asarray(deltas,dtype=float)
    if d.ndim!=2 or d.shape[1]<2 or not np.isfinite(d).all():raise ValueError('parent x repeat finite matrix required')
    b,r=d.shape;theta=float(d.mean());vars_by_parent=d.var(axis=1,ddof=1)
    terms=vars_by_parent/(b*b*r);variance=float(terms.sum());se=math.sqrt(variance)
    nominal_df=b*(r-1)
    if se==0:return {'estimate':theta,'se':0.,'df':nominal_df,'nominal_df':nominal_df,'ci':None,'p':None,'status':'ZERO_EMPIRICAL_TRAIN_VARIANCE_NOT_PROOF_OF_CERTAINTY'}
    df=float(variance*variance/sum((float(x)**2)/(r-1) for x in terms))
    critical=float(t.ppf(1-alpha/2,df))
    return {'estimate':theta,'se':se,'df':df,'nominal_df':nominal_df,'ci':[theta-critical*se,theta+critical*se],
            'p':float(2*t.sf(abs(theta/se),df)),
            'status':'APPROXIMATE_FIXED_PARENT_SATTERTHWAITE_INTERVAL'}

def crossed_bootstrap(scores:np.ndarray,contrast:np.ndarray,reps:int=10000,seed:int=2026102731)->dict:
    """scores [parent, repeat, arm, family, root], already averaged over variants/draws.
    Resample runs WITHIN fixed parent; resample roots WITHIN family, sharing root
    indices across all parents, repeats and arms. This avoids iid-answer bootstrap.
    Few repeats make bootstrap fragile; always report fixed-parent/run diagnostics too.
    """
    x=np.asarray(scores,dtype=float);c=np.asarray(contrast,dtype=float)
    if x.ndim!=5 or x.shape[2]!=len(c) or not np.isfinite(x).all():raise ValueError('score cube')
    b,r,a,f,n=x.shape;rng=np.random.default_rng(seed);vals=np.empty(reps)
    for k in range(reps):
        root_idx=[rng.integers(0,n,size=n) for _ in range(f)]
        parent_vals=[]
        for p in range(b):
            ri=rng.integers(0,r,size=r);runmeans=[]
            for rr in ri:
                fm=[x[p,rr,:,ff,:][:,root_idx[ff]].mean(axis=1) for ff in range(f)]
                runmeans.append(float(np.dot(np.mean(fm,axis=0),c)))
            parent_vals.append(np.mean(runmeans))
        vals[k]=np.mean(parent_vals)
    return {'estimate':float(np.dot(x.mean(axis=(0,1,3,4)),c)),
            'ci95':np.quantile(vals,[.025,.975]).tolist(),
            'ci98_75':np.quantile(vals,[.00625,.99375]).tolist(),
            'replicates':reps,'seed':seed,'scope':'fixed parents; resampled observed repeats and root population',
            'warning':'few training repeats limit calibration; no automatic equivalence from p>0.05'}
