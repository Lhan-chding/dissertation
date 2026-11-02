"""Outcome-blind paired worlds and two independent TOPK oracles.

Only task-structural rejection is allowed. The numerical world is the statistical
unit; its two renamings are paired views, never independent roots.
"""
from __future__ import annotations
import copy, itertools, random
from fractions import Fraction
from .common import digest, enc, finite, seed_for


def oracle_sort(world: dict, query: dict) -> dict:
    values = world['series'][query['selection_series']]
    sign = -1 if query['direction'] == 'largest' else 1
    chosen = sorted(sorted(range(len(values)), key=lambda i: (sign * values[i], i))[:query['k']])
    operands = [Fraction(world['series'][query['value']][i]) for i in chosen]
    answer = sum(operands, Fraction())
    if query['aggregate'] == 'mean':
        answer /= len(chosen)
    return {'indices': chosen, 'selected': [world['categories'][i] for i in chosen],
            'operands': [enc(v) for v in operands], 'answer': enc(answer)}


def oracle_enumerate(world: dict, query: dict) -> dict:
    """Enumerate all k-subsets. Does not call or reuse the sort selector."""
    vals = world['series'][query['selection_series']]
    n = len(vals)
    def outranks(i: int, j: int) -> bool:
        if vals[i] == vals[j]:
            return i < j
        return vals[i] > vals[j] if query['direction'] == 'largest' else vals[i] < vals[j]
    valid = []
    for subset in itertools.combinations(range(n), query['k']):
        others = set(range(n)) - set(subset)
        if all(outranks(i, j) for i in subset for j in others):
            valid.append(subset)
    if len(valid) != 1:
        raise ValueError('Independent selector is not unique')
    chosen = list(valid[0])
    nums = [Fraction(world['series'][query['value']][i]) for i in chosen]
    total = sum(nums, Fraction())
    if query['aggregate'] == 'mean':
        total /= query['k']
    return {'indices': chosen, 'selected': [world['categories'][i] for i in chosen],
            'operands': [enc(v) for v in nums], 'answer': enc(total)}


def world_key(world: dict, query: dict, *, strip_names: bool = False) -> str:
    w = copy.deepcopy(world)
    if strip_names:
        w['categories'] = [f'position_{i}' for i in range(len(w['categories']))]
    return digest({'world': w, 'query': query})


def _assign_values(rng: random.Random, m: int, direction: str, placebo_side: str, grid: list[int]):
    # Ten categories permit an equally sized, wholly unselected placebo tie block.
    # This avoids a 4-item boundary reversal versus a 2-item placebo perturbation.
    n = 10
    tie = rng.choice([42, 48])
    high = [v for v in grid if v > tie]
    low = [v for v in grid if v < tie]
    better, worse = (high, low) if direction == 'largest' else (low, high)
    top = rng.sample(better, 2)
    placebo_value = rng.choice(worse)
    remaining = n - 2 - 2*m
    bottom = [placebo_value]*m + rng.sample([v for v in worse if v != placebo_value], remaining)
    values = top + [tie]*m + bottom
    positional = list(range(n)); rng.shuffle(positional)
    vals = [values[i] for i in positional]
    boundary = sorted(positional.index(i) for i in range(2, 2+m))
    placebo = sorted(positional.index(i) for i in range(2+m, 2+2*m))
    strict = sorted(positional.index(i) for i in (0, 1))
    return vals, boundary, placebo, strict


def build_root(cfg: dict, split: str, index: int, m: int, r: int,
               direction: str, aggregate: str, placebo_side: str) -> dict:
    master = cfg['data']['master_seed']
    root_id = f'{split}_ORDER_{index:04d}'
    for attempt in range(cfg['data']['max_structural_attempts']):
        rng = random.Random(seed_for(master, root_id, attempt))
        alpha, boundary, placebo, strict = _assign_values(rng, m, direction, placebo_side, cfg['data']['grid'])
        beta = rng.sample(cfg['data']['grid'], cfg['data']['categories'])
        boundary_choices = list(itertools.combinations(boundary, r))
        sums = [sum(beta[i] for i in ids) for ids in boundary_choices]
        # Prevent answer collisions among all rank-admissible boundary choices.
        if len(set(sums)) != len(sums):
            continue
        names = rng.sample(cfg['data']['names'], cfg['data']['categories'])
        rng.shuffle(names)
        tie_names = sorted(names[i] for i in boundary)
        aligned = names.copy(); conflict = names.copy()
        for i, name in zip(boundary, tie_names): aligned[i] = name
        for i, name in zip(boundary, reversed(tie_names)): conflict[i] = name
        query = {'family': 'TOPK', 'selection_series': 'Alpha', 'value': 'Beta',
                 'k': 2 + r, 'direction': direction, 'aggregate': aggregate}
        # One positional permutation, applied to both naming views.
        for _ in range(1000):
            permutation = list(range(cfg['data']['categories'])); rng.shuffle(permutation)
            if [i for i in permutation if i in boundary][:r] != boundary[:r] and set([i for i in permutation if i in boundary][:r]) != set(boundary[:r]):
                break
        else:
            raise RuntimeError('Enumeration construction exhausted')
        naming_order = ['ALIGNED', 'CONFLICT']; rng.shuffle(naming_order)
        views = []
        for view_id, relation in enumerate(naming_order):
            labels = aligned if relation == 'ALIGNED' else conflict
            world = {'categories': labels, 'series': {'Alpha': alpha, 'Beta': beta}, 'domain': [10, 90]}
            a = oracle_sort(world, query); b = oracle_enumerate(world, query)
            if a != b: raise AssertionError('Dual oracle mismatch')
            if not finite(Fraction(a['answer'])): raise AssertionError('Non-terminating gold')
            alt_ids = sorted(strict + sorted(boundary, key=lambda i: labels[i])[:r])
            if (set(alt_ids) == set(a['indices'])) != (relation == 'ALIGNED'):
                raise AssertionError('Renaming alignment failed')
            views.append({'qid': root_id + f'_view{view_id}', 'root_id': root_id, 'split': split,
                          'rename': relation, 'structure': f'{m}_choose_{r}',
                          'world': world, 'query': query, 'gold': a,
                          'boundary_indices': boundary, 'required_from_boundary': r,
                          'strictly_better_indices': strict, 'placebo_indices': placebo,
                          'placebo_side': placebo_side, 'enum_permutation': permutation,
                          'alphabetical_selected_indices': alt_ids,
                          'structural_attempt': attempt,
                          'world_query_key': world_key(world, query),
                          'unlabeled_world_query_key': world_key(world, query, strip_names=True)})
        return {'root_id': root_id, 'structure': f'{m}_choose_{r}', 'direction': direction,
                'aggregate': aggregate, 'placebo_side': placebo_side, 'attempt': attempt,
                'boundary_alternative_count': len(boundary_choices), 'views': views}
    raise RuntimeError(f'No valid outcome-blind construction for {root_id}')


def build_worlds(cfg: dict, split: str) -> list[dict]:
    if split not in ('MAIN', 'SMOKE'): raise ValueError('Unknown split')
    # Within each boundary structure: direction x aggregate, 3 roots per cell.
    cells = [(d, a, 0) for d, a in itertools.product(range(2), repeat=2) for _ in range(3)]
    roots = []
    for si, (m, r) in enumerate(cfg['data']['structures']):
        design = cells.copy()
        random.Random(seed_for(cfg['data']['master_seed'], split, 'cells', si)).shuffle(design)
        if split == 'SMOKE': design = [design[0]]
        for db, ab, pb in design:
            roots.append(build_root(cfg, split, len(roots), m, r,
                         'largest' if db == 0 else 'smallest',
                         'sum' if ab == 0 else 'mean',
                         'unselected_matched_size'))
    keys = [r['views'][0]['unlabeled_world_query_key'] for r in roots]
    if len(set(keys)) != len(keys): raise RuntimeError('Duplicate base world; do not silently drop it')
    return roots


def validate_root(root: dict) -> None:
    a, b = root['views']
    if a['world']['series'] != b['world']['series'] or a['query'] != b['query']:
        raise ValueError('Paired views changed values/query')
    if set(a['world']['categories']) != set(b['world']['categories']):
        raise ValueError('Paired name sets differ')
    if a['gold']['indices'] != b['gold']['indices'] or a['gold']['answer'] != b['gold']['answer']:
        raise ValueError('Paired position gold changed')
    for v in root['views']:
        if oracle_sort(v['world'], v['query']) != oracle_enumerate(v['world'], v['query']):
            raise ValueError('Oracles disagree')
        p = v['placebo_indices']; s = set(v['gold']['indices'])
        if len(p) != len(v['boundary_indices']) or not (set(p) <= s or set(p).isdisjoint(s)):
            raise ValueError('Placebo is not selection-neutral')
        if set(p) & set(v['boundary_indices']): raise ValueError('Placebo overlaps boundary')
        if len({v['world']['series']['Alpha'][i] for i in p}) != 1:
            raise ValueError('Placebo entries differ in value')
