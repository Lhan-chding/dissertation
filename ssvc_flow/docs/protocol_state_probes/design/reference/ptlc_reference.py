"""Literal PTLC rules from Claude round five. Deterministic arithmetic only.
The rules are not fitted. V1 emits a proposed rational/out-of-domain value;
V2 replaces that proposal with the input observation. No rounding or clipping.
Analyses here only interpret integer outputs as valid under the parent parser.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction


def validate(H,b,observed,order=None):
    n=len(observed)
    order=tuple(range(n)) if order is None else tuple(order)
    if sorted(order)!=list(range(n)): raise ValueError('order must be permutation')
    if len(H)!=len(b) or any(len(row)!=n for row in H):raise ValueError('shape')
    return [[Fraction(v) for v in row] for row in H],[Fraction(v) for v in b],[Fraction(v) for v in observed],order


def run(H,b,observed,order=None,*,variant='V1',low=0,high=99):
    H,b,x,order=validate(H,b,observed,order)
    if variant not in ('V1','V2'):raise ValueError('variant')
    y=[None]*len(x);done=set();trace=[]
    for k in order:
        votes=[];ids=[]
        for ri,(row,rhs) in enumerate(zip(H,b)):
            if row[k] and all(j in done for j,a in enumerate(row) if a and j!=k):
                v=(rhs-sum((row[j]*y[j] for j in done),Fraction(0)))/row[k]
                votes.append(v);ids.append(ri)
        if not votes:
            chosen=x[k];reason='NO_CLOSED_ROW_COPY'
        else:
            counts=Counter(votes);mode,freq=counts.most_common(1)[0]
            if freq>len(votes)/2:
                chosen=mode;reason='AGREE' if len(counts)==1 else 'STRICT_MAJORITY'
            else:
                chosen=x[k];reason='CONFLICT_FALLBACK_COPY'
        proposal=chosen
        if variant=='V2' and not (low<=chosen<=high):
            chosen=x[k];reason+='__DOMAIN_FALLBACK'
        y[k]=chosen;done.add(k)
        trace.append({'coordinate':k,'closed_rows':ids,'votes':[str(v) for v in votes],
                      'proposal':str(proposal),'value':str(chosen),'reason':reason})
    return y,trace


def valid(y):return all(v.denominator==1 and 0<=v<=99 for v in y)

def one_row(H,j,order=None):
    order=tuple(range(len(H[0]))) if order is None else tuple(order)
    before=set(order[:order.index(j)])
    return any(row[j] and all(i in before for i,c in enumerate(row) if c and i!=j) for row in H)


def transform_star(H,b,order=None):
    n=len(H[0]);order=tuple(range(n)) if order is None else tuple(order)
    supports=[{j for j,c in enumerate(row) if c} for row in H]
    centers=set.intersection(*supports)
    if len(H)!=n-1 or any(len(s)!=2 for s in supports) or len(centers)!=1:raise ValueError('not star')
    c=next(iter(centers))
    if any(row[j]!=1 for row in H for j in range(n) if row[j]):raise ValueError('sum-star only')
    byleaf={next(iter(s-{c})):(row,rhs,ri) for s,row,rhs,ri in zip(supports,H,b,range(len(H)))}
    f=next(k for k in order if k!=c)
    ref,rhsf,rif=byleaf[f]
    HH=[list(ref)];bb=[rhsf];T=[[int(i==rif) for i in range(len(H))]]
    for k in order:
        if k in (c,f):continue
        row,rhs,ri=byleaf[k]
        HH.append([u-v for u,v in zip(row,ref)]);bb.append(rhs-rhsf)
        T.append([int(i==ri)-int(i==rif) for i in range(len(H))])
    return HH,bb,{'center':c,'first_leaf':f,'row_transform':T}


def external(y):
    # For export, exact integers remain integers. Non-integers remain fraction strings.
    return [int(v) if v.denominator==1 else str(v) for v in y]
