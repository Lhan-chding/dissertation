"""Executable design contracts; not the repository's Qwen implementation.

No model framework, network request, or GPU operation is imported. These functions
check mathematical identities, data boundaries, and the planned workload only.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from math import comb, sqrt
from typing import Sequence
import hashlib
import json


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    observed: tuple[int, int, int, int]
    H: tuple[tuple[int, int, int, int], ...]
    b: tuple[int, ...]
    lower: int = 0
    upper: int = 99

    def __post_init__(self):
        if len(self.observed) != 4 or len(self.H) != len(self.b):
            raise ValueError('Four coordinates and one RHS per relation required')
        if any(len(row) != 4 or not any(row) for row in self.H):
            raise ValueError('Invalid relation row')
        if self.lower > self.upper:
            raise ValueError('Invalid domain')
        if any(type(x) is not int or not self.lower <= x <= self.upper for x in self.observed):
            raise ValueError('Observed values must be legal integers')


def parse_vector(text: str) -> tuple[int, int, int, int]:
    try:
        out = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Not one strict JSON array') from exc
    if not isinstance(out, list) or len(out) != 4 or any(type(x) is not int for x in out):
        raise ValueError('Exactly four integers; booleans/floats are rejected')
    return tuple(out)


def validate_order(order: Sequence[int]) -> tuple[int, ...]:
    if len(order) != 4 or any(type(x) is not int for x in order) or sorted(order) != list(range(4)):
        raise ValueError('Not a permutation of canonical coordinates')
    return tuple(order)


def canonicalize(emitted: Sequence[int], order: Sequence[int]) -> tuple[int, ...]:
    order = validate_order(order)
    if len(emitted) != 4:
        raise ValueError('Four emitted coordinates required')
    canonical = [None] * 4
    for k, coordinate in enumerate(order):
        canonical[coordinate] = emitted[k]
    return tuple(canonical)


def relations_hold(task: PublicTask, y: Sequence[int]) -> bool:
    return len(y) == 4 and all(sum(a*v for a, v in zip(row, y)) == rhs
                               for row, rhs in zip(task.H, task.b))


def public_verifier(task: PublicTask, y: Sequence[int]) -> bool:
    return (len(y) == 4 and all(type(v) is int and task.lower <= v <= task.upper for v in y)
            and sum(a != b for a, b in zip(task.observed, y)) == 1 and relations_hold(task, y))


def solve_single_edit(task: PublicTask) -> list[tuple[int, ...]]:
    solutions = set()
    for j in range(4):
        for value in range(task.lower, task.upper + 1):
            if value == task.observed[j]:
                continue
            candidate = list(task.observed)
            candidate[j] = value
            if public_verifier(task, candidate):
                solutions.add(tuple(candidate))
    return sorted(solutions)


def canonical_json(y: Sequence[int]) -> str:
    if len(y) != 4 or any(type(v) is not int for v in y):
        raise ValueError('Canonical target must be four integers')
    return json.dumps(list(y), separators=(',', ':'))


def transform_b1(task: PublicTask) -> PublicTask:
    if len(task.H) != 3:
        raise ValueError('B1 requires a three-edge star')
    supports = []
    for row in task.H:
        idx = {j for j, a in enumerate(row) if a != 0}
        if len(idx) != 2 or any(row[j] != 1 for j in idx):
            raise ValueError('B1 reference expects original unit-sum star rows')
        supports.append(idx)
    centers = set.intersection(*supports)
    if len(centers) != 1 or len({tuple(sorted(s)) for s in supports}) != 3:
        raise ValueError('No unique star center')
    center = centers.pop()
    leaves = sorted(set(range(4)) - {center})
    first = leaves[0]  # Original canonical output order; no corruption label.
    leaf_to_row = {(s - {center}).pop(): j for j, s in enumerate(supports)}
    first_row = leaf_to_row[first]
    H_new = [task.H[first_row]]
    b_new = [task.b[first_row]]
    for leaf in leaves[1:]:
        row = [0]*4
        row[leaf], row[first] = 1, -1
        H_new.append(tuple(row))
        b_new.append(task.b[leaf_to_row[leaf]] - task.b[first_row])
    return PublicTask(task.task_id, task.observed, tuple(H_new), tuple(b_new), task.lower, task.upper)


def relation_score(task: PublicTask, y: Sequence[int]) -> float:
    return sum(sum(a*v for a, v in zip(row, y)) == rhs for row, rhs in zip(task.H, task.b))/len(task.H)


def pass_at_k(n: int, correct: int, k: int) -> float:
    if any(type(v) is not int for v in (n, correct, k)) or not (0 <= correct <= n and 1 <= k <= n):
        raise ValueError('Require 0 <= correct <= n and 1 <= k <= n')
    return 1.0 if n-correct < k else 1.0 - comb(n-correct, k)/comb(n, k)


def mixed_two_calls(n: int, c: int, m: int, d: int) -> float:
    if not (n > 0 and m > 0 and 0 <= c <= n and 0 <= d <= m):
        raise ValueError('Invalid counts')
    return 1-(1-c/n)*(1-d/m)


def replacement_gain(p: float, q: float) -> float:
    if not (0 <= p <= 1 and 0 <= q <= 1):
        raise ValueError('Probabilities must be in [0,1]')
    return (1-p)*(q-p)


def verified_budget_success(streams: dict[str, Sequence[bool]], allocation: dict[str, int], budget: int) -> bool:
    if sum(allocation.values()) != budget or any(type(v) is not int or v < 0 for v in allocation.values()):
        raise ValueError('The allocation must sum to the declared budget')
    for protocol, count in allocation.items():
        if protocol not in streams or len(streams[protocol]) < count:
            raise ValueError('Missing required atomic samples')
    return any(any(streams[protocol][:count]) for protocol, count in allocation.items())


def conservative_truth_orbit(y: Sequence[int]) -> tuple[int, ...]:
    if len(y) != 4 or any(type(v) is not int for v in y):
        raise ValueError('Four integer truth coordinates required')
    return tuple(sorted(y))


def trend_worlds(width: int) -> set[tuple[int, int, int, int]]:
    if width < 4:
        return set()
    return {(a, a+d, a+2*d, a+3*d) for a in range(width)
            for d in range(-(width-1)//3, (width-1)//3+1)
            if d != 0 and all(0 <= a+j*d < width for j in range(4))}


def trainable_role(role: str) -> bool:
    return role in {'T_train', 'R_replay'}


def training_view(records: list[dict]) -> list[tuple[str, str, str]]:
    out = {}
    for record in records:
        if not trainable_role(record['role']):
            raise ValueError('Validation/test/development probes cannot enter SFT')
        if not record.get('verified', False):
            raise ValueError('An unverified target cannot enter a SELF view')
        item = (record['task_id'], record['O0_prompt'], canonical_json(record['canonical']))
        if item[0] in out and out[item[0]] != item:
            raise ValueError('Conflicting target for the same task')
        out[item[0]] = item
    return sorted(out.values())


def alias_key(parent: str, repeat: int, view: list[tuple[str, str, str]], replay: list, settings: dict) -> str:
    # For small immutable metadata/views, not full-model hashes.
    payload = [parent, repeat, sorted(view), replay, settings]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def request_seed(parent: str, split: str, task: str, protocol: str, role: str, repeat: int, draw: int) -> int:
    payload = json.dumps([parent, split, task, protocol, role, repeat, draw, 'v1']).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big') % (2**63-1)


def completion_batch_row(prompt: Sequence[int], target: Sequence[int], eos: int, pad: int, total: int) -> dict:
    if not prompt or not target or eos in target:
        raise ValueError('Nonempty prompt/target; target must not already contain EOS')
    completion = list(target) + [eos]
    sequence = list(prompt) + completion
    if len(sequence) > total:
        raise ValueError('No silent truncation')
    padding = total-len(sequence)
    return {'input_ids': sequence+[pad]*padding,
            'labels': [-100]*len(prompt)+completion+[-100]*padding,
            'attention_mask': [1]*len(sequence)+[0]*padding,
            'prediction_positions': list(range(len(prompt)-1, len(sequence)-1)),
            'completion_length': len(completion)}


def weighted_sft_loss(focus_nll: Sequence[float], replay_nll: Sequence[float], replay_only=False) -> float:
    if len(replay_nll) != 4 or (not replay_only and len(focus_nll) != 12) or (replay_only and focus_nll):
        raise ValueError('Exactly 12 focus and 4 replay slots, or replay-only 4')
    return (0 if replay_only else .75*sum(focus_nll)/12) + .25*sum(replay_nll)/4


def lr_at_update(step: int, base=1e-5, warmup=8) -> float:
    if step < 1:
        raise ValueError('One-indexed optimizer update')
    return base*min(step/warmup, 1)


def wilson_interval(correct: int, n: int, z=1.959963984540054) -> tuple[float, float]:
    if n <= 0 or not 0 <= correct <= n:
        raise ValueError('Invalid binomial data')
    phat=correct/n
    denominator=1+z*z/n
    center=(phat+z*z/(2*n))/denominator
    half=z*sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denominator
    return max(0., center-half), min(1., center+half)


def program_signature(y: tuple, copy: tuple, truth: tuple, v1: tuple, v2: tuple) -> dict:
    return {'X': y == truth, 'copy': y == copy, 'V1': y == v1, 'V2': y == v2,
            'distinct_program_error': y in (v1,v2) and y not in (copy,truth)}


def planned_jobs(c: dict) -> list[dict]:
    jobs=[]
    for parent in c['model']['parents']:
        arms=c['sft']['main_arms']+(c['sft']['S96_additional_arms'] if parent == 'S96' else [])
        for repeat, seed in enumerate(c['sft']['seeds']):
            for arm in arms:
                jobs.append({'kind':'SFT','parent':parent,'repeat':repeat,'seed':seed,'arm':arm,
                             'steps':c['sft']['steps'],'status':'PENDING_IMPLEMENTATION_AND_SERVER_CHECK'})
    if c['R0_reference']['enabled']:
        for parent in c['R0_reference']['parents']:
            for repeat, seed in enumerate(c['R0_reference']['seeds']):
                jobs.append({'kind':'RL_CONTEXT','parent':parent,'repeat':repeat,'seed':seed,
                             'arm':'R0_RESET32','steps':c['R0_reference']['steps'],
                             'status':'PENDING_IMPLEMENTATION_AND_SERVER_CHECK'})
    return jobs


def workload(c: dict) -> dict:
    nparents=len(c['model']['parents']); counts=c['data']['new_task_counts']
    average_protocols=sum(len(c['protocols'][family]) for family in c['data']['families'])/len(c['data']['families'])
    teacher={}
    for split,repeats in [('T_train',c['discovery']['T_repeats']),('V_selection',c['discovery']['V_repeats']),('E_test',c['discovery']['E_repeats'])]:
        teacher[split]=int(nparents*counts[split]*average_protocols*c['discovery']['per_protocol_draws']*repeats)
    jobs=planned_jobs(c); sft=[j for j in jobs if j['kind']=='SFT']; rl=[j for j in jobs if j['kind']=='RL_CONTEXT']
    nS,nR=len(sft),len(rl); e=c['evaluation']; batch=c['sft']
    final=counts['E_test']*e['trained_E_draws']+counts['G_guard_prompts']*e['trained_G_draws']
    components={'teacher_T':teacher['T_train'],'teacher_V':teacher['V_selection'],'teacher_E':teacher['E_test'],
                'SFT_final_E_G':nS*final,
                'SFT_intermediate_V':nS*len(e['V_evaluation_steps'])*counts['V_selection']*e['trained_V_draws'],
                'SFT_train_sentinels':nS*e['train_sentinel_tasks']*len(e['train_sentinel_greedy_steps']),
                'parent_G':nparents*counts['G_guard_prompts']*e['trained_G_draws'],
                'RL_rollouts':nR*c['R0_reference']['steps']*c['R0_reference']['B']*c['R0_reference']['K'],
                'RL_evaluation':nR*(final+counts['V_selection']*e['trained_V_draws'])}
    processed=sum(j['steps']*(4 if j['arm']=='REPLAY_ONLY' else 16) for j in sft)
    return {'status':'planned_upper_counts_before_aliases_or_empty_return_parent',
            'generated_sequences':components,'total_generated_sequences':sum(components.values()),
            'technical_smoke_max':e['technical_smoke_max_draws'],'SFT_jobs':nS,'RL_jobs':nR,
            'SFT_updates':sum(j['steps'] for j in sft),'RL_updates':sum(j['steps'] for j in rl),
            'SFT_processed_target_sequences':processed,'new_model_results':False}
