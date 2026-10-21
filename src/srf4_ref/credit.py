"""Reference for the NEW, bounded failure-stratum semantic advantage residual.
This is an explicitly proposed experiment, not a previously validated method.
Production A_J must be supplied by the audited historical trainer.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Sequence

LABELS = ('PASS', 'FAIL', 'UNKNOWN')
@dataclass(frozen=True)
class Observation:
    row_id: str
    j: float
    protocol_valid: bool
    finished: bool
    sufficient: bool | None
    defined: bool
    perception: str
    selection: str
    def __post_init__(self):
        if not self.row_id or self.j not in (0.,1.): raise ValueError('C requires binary audited J')
        if self.perception not in LABELS or self.selection not in LABELS:
            raise ValueError('Bad semantic label')

def population_advantages(rewards: Sequence[float], eps: float = 1e-8, mode: str = 'floor') -> list[float]:
    """For tests/replay only: use native code in production and prove parity."""
    if not rewards or eps < 0 or mode not in ('floor','add'): raise ValueError('Invalid inputs')
    if any(not math.isfinite(x) for x in rewards): raise ValueError('Nonfinite reward')
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((x-mean)**2 for x in rewards)/len(rewards))
    if std == 0: return [0.]*len(rewards)
    denom = max(std,eps) if mode == 'floor' else std+eps
    return [(x-mean)/denom for x in rewards]

def semantic_residual(rows: Sequence[Observation], arm: str, beta: float = .25,
                      shuffle_seed: int = 0) -> tuple[list[float], dict]:
    if arm not in ('J','EVID','SELECT','SELECT_SHUF'): raise ValueError('Unknown arm')
    if not 0 <= beta <= .25: raise ValueError('Frozen C beta must be <=0.25')
    if len({r.row_id for r in rows}) != len(rows): raise ValueError('Duplicate row IDs')
    result = [0.]*len(rows)
    candidates = [i for i,r in enumerate(rows) if r.j == 0 and r.protocol_valid and r.finished
                  and r.sufficient is True and r.defined]
    reason = 'ACTIVE'
    if len(candidates) < 2: reason = 'TOO_FEW_ELIGIBLE_FAILURES'
    elif any(rows[i].perception == 'UNKNOWN' or rows[i].selection == 'UNKNOWN' for i in candidates):
        reason = 'ABSTAIN_UNKNOWN_IN_FAILURE_STRATUM'
    if reason != 'ACTIVE':
        return result, {'reason':reason, 'eligible_indices':candidates, 'residual_linf':0., 'scores':[]}
    p = {i:float(rows[i].perception == 'PASS') for i in candidates}
    s = {i:float(rows[i].selection == 'PASS') for i in candidates}
    if arm == 'SELECT_SHUF':
        rng = random.Random(shuffle_seed)
        # Permute ONLY the extra S label, within equal J, P and eligibility state.
        # All candidates have equal J=0, protocol/finish/sufficiency/defined=True.
        for pv in (0.,1.):
            ids = [i for i in candidates if p[i] == pv]
            values = [s[i] for i in ids]; rng.shuffle(values)
            for i,value in zip(ids,values): s[i] = value
    score = {i:.5*p[i] + (.5*p[i]*s[i] if arm in ('SELECT','SELECT_SHUF') else 0.) for i in candidates}
    mean_score = sum(score.values())/len(score)
    if arm != 'J':
        for i in candidates: result[i] = beta*(score[i]-mean_score)
    assert abs(sum(result)) < 1e-10
    assert all(abs(x) <= beta+1e-12 for x in result)
    return result, {'reason':reason if arm != 'J' else 'BASELINE_NO_RESIDUAL',
                    'eligible_indices':candidates, 'residual_linf':max(map(abs,result), default=0.),
                    'scores':[score[i] for i in candidates],
                    'active':any(abs(x)>1e-12 for x in result)}

def combined_advantages(base: Sequence[float], rows: Sequence[Observation], arm: str,
                        beta: float = .25, shuffle_seed: int = 0) -> tuple[list[float],dict]:
    if len(base) != len(rows) or any(not math.isfinite(x) for x in base):
        raise ValueError('Invalid native base advantages')
    residual,meta = semantic_residual(rows, arm, beta, shuffle_seed)
    # NO restandardization, whitening, or recentering after this addition.
    return [a+d for a,d in zip(base,residual)],meta
