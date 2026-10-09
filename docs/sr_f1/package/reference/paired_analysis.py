"""Root-stratified effect estimates, conditional on the realised training paths.
Input x/y have [training_seed, root] shape; each root score already averages its
fixed questions/draws. Does NOT turn roots into independent training replicas.
"""
from __future__ import annotations
import numpy as np


def paired_root_ci(x,y,families,replicates=5000,seed=812901):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    fam=np.asarray(families)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=len(fam):raise ValueError('Shape mismatch')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Missing technical rows are not zero scores')
    groups=[np.flatnonzero(fam==f) for f in sorted(set(fam.tolist()))]
    if any(len(g)==0 for g in groups):raise ValueError('Empty family')
    delta=x-y
    seed_points=np.mean(np.stack([delta[:,g].mean(axis=1) for g in groups]),axis=0)
    root_average=delta.mean(axis=0)
    rng=np.random.default_rng(seed)
    draws=np.zeros(replicates)
    for inds in groups:
        idx=rng.choice(inds,size=(replicates,len(inds)),replace=True)
        draws+=root_average[idx].mean(axis=1)/len(groups)
    return {'effect_probability':float(seed_points.mean()),'effect_pp':float(100*seed_points.mean()),
            'seed_effects_pp':(100*seed_points).tolist(),
            'CI95_pp':(100*np.quantile(draws,[.025,.975])).tolist(),
            'CI99_pp':(100*np.quantile(draws,[.005,.995])).tolist(),
            'root_count':len(fam),'training_seed_count':x.shape[0],
            'scope':'root measurement uncertainty conditional on realised paths; not a training-population CI'}
