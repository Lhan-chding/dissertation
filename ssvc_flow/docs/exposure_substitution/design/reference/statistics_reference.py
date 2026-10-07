"""Statistics for frozen, matched numeric roots. Not a fitted predictor."""
from __future__ import annotations
import numpy as np


def primary_effect(probabilities, *, bootstrap_replicates=5000, seed=107079):
    """Array: [parent=2, schedule=3, arm=(A,B,C), root=128, hub=(3,4)].

    Bootstrap conditions on the trained parents/schedules and resamples numeric
    roots, preserving their entire joint vector. Separate training-path reporting
    must not be replaced by this conditional interval.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (2,3,3,128,2) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Complete 2x3x3x128x2 probability grid required')
    d4 = p[:,:,1,:,1] - p[:,:,2,:,1]
    d3 = p[:,:,1,:,0] - p[:,:,2,:,0]
    z = (d4-d3)/2
    root_mean = z.mean(axis=(0,1))
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,128,size=(bootstrap_replicates,128))
    boots=root_mean[samples].mean(axis=1)
    return {'Gamma':float(z.mean()),'Delta4':float(d4.mean()),'Delta3':float(d3.mean()),
            'ci95_root_conditional':np.quantile(boots,[.025,.975]).tolist(),
            'per_parent_schedule_Gamma':z.mean(axis=2).tolist(),
            'per_parent_Gamma':z.mean(axis=(1,2)).tolist(),
            'roots':128,'bootstrap_replicates':bootstrap_replicates,'seed':seed,
            'condition':'actual two parent checkpoints and actual three schedules per parent'}


def cross_macro(cell_probabilities):
    if set(cell_probabilities) != {(c,j) for c in range(4) for j in range(4)}:
        raise ValueError('All 16 cross cells required')
    vals=[float(cell_probabilities[k]) for k in sorted(cell_probabilities)]
    if any(not 0<=v<=1 for v in vals):raise ValueError('Invalid probability')
    return sum(vals)/16
