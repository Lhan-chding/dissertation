"""Fixed-parent, paired-training-repeat and paired-root uncertainty utilities."""
from __future__ import annotations
import numpy as np

def summarize_effect(d):
    # [parent,repeat,family,root]; each root has already averaged queries and draws.
    x=np.asarray(d,float)
    if x.ndim!=4 or x.shape[:2]!=(3,3) or min(x.shape[2:])<1 or not np.isfinite(x).all():raise ValueError('expected complete finite [3,3,F,R]')
    return float(x.mean())

def paired_hierarchical_bootstrap(d,seed=2026102510,n=20000,alpha=0.05/3):
    """Parents fixed; resample paired repeats within each parent, roots within family.
    Same root draws are used for all parents/repeats. Input is paired arm differences.
    Small numbers of repeats remain a limitation; report the train-only t sensitivity.
    """
    x=np.asarray(d,float)
    if x.ndim!=4 or x.shape[:2]!=(3,3) or min(x.shape[2:])<1 or not np.isfinite(x).all():raise ValueError('complete finite 3x3 tensor required')
    if not 0<alpha<1 or not isinstance(n,int) or isinstance(n,bool) or n<1:raise ValueError('invalid interval or bootstrap count')
    b,r,f,k=x.shape;rng=np.random.default_rng(seed);values=np.empty(n)
    for t in range(n):
        roots=rng.integers(0,k,size=(f,k));v=x[:,:,np.arange(f)[:,None],roots]
        reps=rng.integers(0,r,size=(b,r));v=v[np.arange(b)[:,None],reps]
        values[t]=v.mean()
    return dict(effect=float(x.mean()),lower=float(np.quantile(values,alpha/2)),upper=float(np.quantile(values,1-alpha/2)),
                interval_level=1-alpha,replicates=n,conditional_on='three fixed parents and frozen data-generating recipe')

def training_repeat_t(d,alpha=0.05/3):
    from scipy.stats import t
    raw=np.asarray(d,float)
    if raw.ndim!=4 or raw.shape[:2]!=(3,3) or min(raw.shape[2:])<1 or not np.isfinite(raw).all():raise ValueError('complete finite 3 parents x 3 repeats tensor required')
    if not 0<alpha<1:raise ValueError('invalid alpha')
    x=raw.mean(axis=(2,3))
    variances=x.var(axis=1,ddof=1)/3/9
    var=float(variances.sum())
    if var==0:return dict(effect=float(x.mean()),se=0.0,df=None,lower=float(x.mean()),upper=float(x.mean()))
    df=var*var/float(np.sum(variances**2/2));se=var**.5;crit=float(t.ppf(1-alpha/2,df))
    return dict(effect=float(x.mean()),se=se,df=df,lower=float(x.mean()-crit*se),upper=float(x.mean()+crit*se))

def relation_metrics(draws,gold,delta):
    """draws: [D,4] parsed numeric answers; None means invalid. Pair draw IDs predefined.
    Correct answer differences alone can be satisfied by a shared offset: also score
    absolute correctness and all-four success. No best-of matching across draws.
    """
    n=len(draws)
    if n<1 or len(gold)!=4 or len(delta)!=4 or any(len(a)!=4 for a in draws):raise ValueError('nonempty draws of exactly four variants required')
    relation=absolute=joint=0
    for answers in draws:
        good=[a is not None and a==g for a,g in zip(answers,gold)]
        absolute+=sum(good)/4
        rel=all(answers[v] is not None and answers[0] is not None and answers[v]-answers[0]==delta[v] for v in range(4))
        relation+=int(rel);joint+=int(all(good))
    return dict(answer_accuracy=absolute/n,relation_consistency=relation/n,all_four_correct=joint/n)
