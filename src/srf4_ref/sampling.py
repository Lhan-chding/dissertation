"""Frozen-within-step LZ sampler; standard-library-only reference implementation."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import random
from typing import Mapping, Sequence

TIERS = ('learning', 'mastered', 'starved')

def seed_for(master: int, *parts: object) -> int:
    """Stable across Python processes; never use Python's salted hash()."""
    import json
    payload = json.dumps([master, *parts], ensure_ascii=False, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**31 - 1)

@dataclass(frozen=True)
class State:
    p: float
    tier: str
    def __post_init__(self):
        if not math.isfinite(self.p) or not 0 <= self.p <= 1 or self.tier not in TIERS:
            raise ValueError('Invalid probability/tier')

@dataclass(frozen=True)
class Assignment:
    group_index: int
    variant_id: str
    qid: str
    g: int
    slots: int

def initialize(p: float) -> State:
    return State(p, 'mastered' if p >= .95 else 'starved' if p < .125 else 'learning')

def update_state(previous: State, observed_rewards: Sequence[float], alpha: float = .25) -> State:
    if not 0 < alpha <= 1: raise ValueError('alpha must be in (0,1]')
    if len(observed_rewards) == 0: return previous
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in observed_rewards):
        raise ValueError('Rewards must be finite and in [0,1]')
    p = (1-alpha)*previous.p + alpha*sum(observed_rewards)/len(observed_rewards)
    # One update per variant per optimizer step; aggregate all its answers first.
    if previous.tier == 'mastered' and p >= .90: tier = 'mastered'
    elif previous.tier == 'starved' and p < .25: tier = 'starved'
    elif p >= .95: tier = 'mastered'
    elif p < .125: tier = 'starved'
    else: tier = 'learning'
    return State(p, tier)

def plan_step(states: Mapping[str, State], roots: Mapping[str, Sequence[str]], *,
              seed: int, mode: str = 'lz', slots: int = 16) -> list[Assignment]:
    if mode not in ('uniform', 'lz') or slots < 1: raise ValueError('Invalid mode/budget')
    variants = sorted(states)
    if not variants or set(variants) != set(roots): raise ValueError('State/root keys mismatch')
    if any(not roots[v] or len(set(roots[v])) != len(roots[v]) for v in variants):
        raise ValueError('Empty or duplicated root inventory')
    rng = random.Random(seed)
    # Freeze tiers and weights here. No EMA update occurs during this call.
    costs = {v: (4 if mode == 'lz' and states[v].tier == 'starved' else 1) for v in variants}
    weights = {v: (.25 if mode == 'lz' and states[v].tier == 'mastered' else 1.) for v in variants}
    remaining = slots; result = []
    while remaining:
        feasible = [v for v in variants if costs[v] <= remaining]
        if not feasible: raise ValueError('No feasible tail; invalid slot configuration')
        # Equivalent in distribution to rejecting infeasible draws, not identical RNG consumption.
        v = rng.choices(feasible, weights=[weights[x] for x in feasible], k=1)[0]
        qid = rng.choice(sorted(roots[v]))
        result.append(Assignment(len(result), v, qid, costs[v]*8, costs[v]))
        remaining -= costs[v]
    return result

def one_starved_dp(slots: int = 16, small_count: int = 31,
                   small_weight: Fraction = Fraction(1,4), starved_weight: Fraction = Fraction(1)) -> dict:
    if slots < 0 or small_count < 1 or small_weight <= 0 or starved_weight <= 0:
        raise ValueError('Invalid DP configuration')
    q = starved_weight/(small_count*small_weight + starved_weight)
    e = [Fraction(0) for _ in range(max(4, slots+1))]
    p0 = [Fraction(1) for _ in range(max(4, slots+1))]
    for s in range(4, slots+1):
        e[s] = q*(1+e[s-4]) + (1-q)*e[s-1]
        p0[s] = (1-q)*p0[s-1]
    return {'expected_starved_groups':float(e[slots]), 'expected_starved_answers':32*float(e[slots]),
            'probability_no_starved_group':float(p0[slots]), 'initial_starved_draw_probability':float(q)}
