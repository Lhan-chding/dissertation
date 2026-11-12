"""CPU-only, fail-closed helpers for SR-SATDOSE-1. No model or scheduler API.

These functions are diagnostics, not a replacement for the pinned native scorer/loss.
Python >=3.10; standard library only. See docs/02_GATES_AND_STATISTICS_zh.md.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

STRUCTURES = ('2_choose_1', '3_choose_1', '3_choose_2', '4_choose_2')
STARTS = (11031, 11032)
ARMS = ('T0', 'T1')
KEY = ('panel', 'root_id', 'structure', 'parent', 'train_seed', 'arm', 'step', 'view', 'draw')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict:
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key: ' + key)
        out[key] = value
    return out


def reject_constant(value: str):
    raise ValueError('non-finite JSON constant: ' + value)


def loads(text: str, *, decimal: bool = False) -> Any:
    return json.loads(text, object_pairs_hook=strict_object,
                      parse_constant=reject_constant,
                      parse_float=Decimal if decimal else float)


def read_json(path: str | Path) -> Any:
    return loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                obj = loads(line)
                require(isinstance(obj, dict), f'{path}:{line_no}: expected JSON object')
                yield obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False,
                               indent=2) + '\n', encoding='utf-8')


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                      allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def seed_for(*parts: Any) -> int:
    # New experiment's schedule seed scheme, NOT a replacement for the historical eval seed.
    return int(object_sha(list(parts))[:16], 16) % (2**63 - 1)


def parse_selected(text: str, finish: str) -> tuple[dict | None, list[str] | None]:
    if not isinstance(text, str) or len(text) > 100000:
        return None, None
    try:
        obj = loads(text, decimal=True)
    except (ValueError, TypeError, ArithmeticError, RecursionError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    selected = obj.get('selected')
    if finish not in ('eos', 'balanced') or not isinstance(selected, list) \
            or not all(type(x) is str for x in selected):
        return obj, None
    return obj, selected


def exact_hit(selected: list[str] | None, gold: Iterable[str]) -> int:
    if selected is None:
        return 0
    require(isinstance(selected, list) and all(type(x) is str for x in selected),
            'selected must be a list of strings or None')
    return int(len(selected) == len(set(selected)) and set(selected) == set(gold))


def geometry(task: dict, audit: dict, panel: str = '') -> dict:
    """Boundary-set geometry; never infer a source from final answer or from 'not list'."""
    names = task['world']['categories']
    require(len(names) == len(set(names)), 'duplicate chart labels')
    bidx = sorted(task['strictly_better_indices'])
    tidx = sorted(task['boundary_indices'])
    r = task.get('r', task.get('required_from_boundary'))
    require(type(r) is int and 0 < r < len(tidx), 'invalid boundary cardinality')
    B, T = {names[i] for i in bidx}, [names[i] for i in tidx]
    require(not B.intersection(T), 'better and boundary overlap')
    chart = frozenset(B | set(T[:r]))
    alt = audit.get('expected_report_order')
    if alt is None:
        alt = audit.get('list_order')
    alt_set = None
    if alt is not None:
        require(len(alt) == len(names) and set(alt) == set(names), 'invalid alternative list')
        alt_t = [x for x in alt if x in T]
        alt_set = frozenset(B | set(alt_t[:r]))
    if panel == 'prefix10':
        induced = audit.get('prefix_induced_selected')
        alt_set = frozenset(induced) if induced is not None else None
    authority = task.get('authority', 'chart')
    require(authority in ('chart', 'list'), 'unknown authority')
    auth = alt_set if authority == 'list' else chart
    require(auth is not None and auth == frozenset(task['gold']['selected']),
            'independent geometry disagrees with gold')
    return dict(B=frozenset(B), T=frozenset(T), r=r, chart=chart, alt=alt_set,
                authorised=auth, chart_order=T)


def source_class(selected: list[str] | None, geo: dict) -> str:
    if selected is None:
        return 'unknown'
    require(isinstance(selected, list) and all(type(x) is str for x in selected),
            'malformed selected')
    s = frozenset(selected)
    admissible = (len(s) == len(selected) and geo['B'] <= s
                  and s <= geo['B'] | geo['T'] and len(s & geo['T']) == geo['r'])
    if not admissible:
        return 'inadmissible'
    if geo['chart'] == geo['alt'] and s == geo['chart']:
        return 'same'
    if s == geo['chart']:
        return 'chart'
    if geo['alt'] is not None and s == geo['alt']:
        return 'list'
    return 'other_admissible'


def binary(values: list[int], name: str, n: int | None = None) -> None:
    require(isinstance(values, list) and bool(values), f'{name}: empty/non-list')
    require(n is None or len(values) == n, f'{name}: expected {n} rows')
    require(all(type(x) is int and x in (0, 1) for x in values), f'{name}: not strict binary ints')


def native_advantages(reward: list[int]) -> list[float]:
    """Matches the pinned G8 population-SD formula, no epsilon, no group filtering."""
    binary(reward, 'reward', 8)
    mu = sum(reward) / 8
    sd = math.sqrt(sum((r - mu)**2 for r in reward) / 8)
    return [(r - mu)/sd for r in reward] if sd else [0.0]*8


def group_metrics(J: list[int], c: list[int], known: list[int] | None = None) -> dict:
    binary(J, 'J', 8); binary(c, 'authorised', 8)
    known = [1]*8 if known is None else known
    binary(known, 'known', 8)
    require(all(not x or k for x, k in zip(c, known)), 'unknown counted as authorised')
    a = native_advantages(J)
    p = sum(c)/8
    dose = sum(x*(y-p) for x, y in zip(a, c))
    ss = sum(x*x for x in a)
    between = 0.0
    for label in (0, 1):
        xs = [x for x, y in zip(a, c) if y == label]
        if xs:
            between += len(xs) * mean(xs)**2
    return dict(J_successes=sum(J), authorised=sum(c), known=sum(known),
                signal=int(0 < sum(J) < 8 and 0 < sum(c) < 8),
                J_mixed=int(0 < sum(J) < 8), source_mixed=int(0 < sum(c) < 8),
                all_J0=int(sum(J) == 0), all_J1=int(sum(J) == 8),
                J_silent_source_mixed=int(sum(J) in (0, 8) and 0 < sum(c) < 8),
                J1_source0=sum(j and not x for j, x in zip(J, c)),
                J1_source1=sum(j and x for j, x in zip(J, c)),
                dose=dose, eta2=(between/ss if ss else None), advantage=a)


def quantile(values: list[float], probability: float) -> float:
    require(bool(values) and 0 <= probability <= 1, 'invalid quantile input')
    s = sorted(values); pos = (len(s)-1)*probability
    lo = int(pos); hi = min(lo+1, len(s)-1)
    return s[lo] + (s[hi]-s[lo])*(pos-lo)


def bootstrap_mean(values: list[float], *, B: int = 10000, seed: int = 2026111207) -> dict:
    require(len(values) >= 2 and B >= 100, 'bootstrap requires >=2 roots and >=100 replicates')
    require(all(type(x) in (int, float) and math.isfinite(x) for x in values), 'nonfinite data')
    rng = random.Random(seed); n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(B)]
    bm = mean(boot)
    return dict(mean=mean(values), lower95=quantile(boot,.025), upper95=quantile(boot,.975),
                se=(sum((x-bm)**2 for x in boot)/(B-1))**.5,
                B=B, seed=seed)


def dry_metrics(groups: list[dict], *, B: int = 10000) -> dict:
    """One starting checkpoint, exactly 64 conflict G8 groups, no dropped unknowns."""
    require(len(groups) == 64, 'dry-run requires all 64 conflict groups')
    rows = [group_metrics(g['J'], g['c'], g['known']) for g in groups]
    boot = bootstrap_mean([g['dose'] for g in rows], B=B)
    jpos = sum(g['J_successes'] for g in rows)
    cpos = sum(g['authorised'] for g in rows)
    abssum = sum(abs(g['dose']) for g in rows)
    return dict(groups=64, responses=512,
                signal_groups=sum(g['signal'] for g in rows),
                unknown_rate=1-sum(g['known'] for g in rows)/512,
                authorised_rate=cpos/512,
                J1_source0_rate=(sum(g['J1_source0'] for g in rows)/jpos if jpos else None),
                q=(sum(g['J1_source1'] for g in rows)/cpos if cpos else None),
                dose_mean=boot['mean'], dose_lower95=boot['lower95'], dose_upper95=boot['upper95'],
                direction_fraction=(sum(g['dose'] for g in rows)/abssum if abssum else 0),
                zero_dose_groups=sum(abs(g['dose']) < 1e-12 for g in rows),
                planned_dose_point=288*boot['mean'], planned_dose_lower=288*boot['lower95'],
                bootstrap=boot)


def dry_gate(m: dict, cfg: dict) -> list[str]:
    failures = []
    checks = [
        (m['signal_groups'] >= cfg['min_signal_groups'], 'INSUFFICIENT_SIGNAL_GROUPS'),
        (m['dose_mean'] >= cfg['min_dose_mean'], 'LOW_DIRECTED_DOSE'),
        (m['dose_lower95'] >= cfg['min_dose_lower95'], 'DOSE_UNCERTAIN'),
        (m['unknown_rate'] <= cfg['max_unknown_rate'], 'LOW_SELECTION_COVERAGE'),
        (m['direction_fraction'] >= cfg['min_direction_fraction'], 'MISDIRECTED_REWARD'),
        (m['J1_source0_rate'] is not None and m['J1_source0_rate'] <= cfg['max_J1_source0_rate'],
         'J_SOURCE_MISALIGNMENT_OR_NO_REWARD'),
    ]
    for ok, why in checks:
        if not ok:
            failures.append(why)
    return failures


def effect_classification(lower: float, upper: float, target: float = .10) -> str:
    require(all(math.isfinite(x) for x in (lower, upper, target)) and lower <= upper,
            'invalid interval')
    if lower >= target:
        return 'PRACTICAL_EFFECT_SUPPORTED_CONDITIONAL'
    if upper < target:
        return ('POSITIVE_BUT_BELOW_TARGET_CONDITIONAL' if lower > 0
                else 'PRACTICAL_EFFECT_EXCLUDED_CONDITIONAL')
    return ('POSITIVE_SIZE_UNRESOLVED_CONDITIONAL' if lower > 0
            else 'UNRESOLVED_CONDITIONAL')


def check_artifact(root: Path, ref: dict) -> Path:
    require(set(ref) >= {'path', 'sha256'}, 'incomplete artifact reference')
    rel = Path(ref['path'])
    require(not rel.is_absolute() and '..' not in rel.parts, 'artifact path traversal')
    p = (root/rel).resolve(strict=True)
    require(p.is_relative_to(root.resolve()) and p.is_file(), 'artifact escapes root')
    require(file_sha(p) == ref['sha256'], 'artifact hash mismatch: '+str(rel))
    return p
