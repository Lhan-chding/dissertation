"""Exact task arithmetic and two independently structured oracles.
The second selector enumerates admissible sets; it does not call the first selector.
No production reward implementation is replaced by this reference.
"""
from __future__ import annotations
import hashlib, itertools, json, math, re
from decimal import Decimal
from fractions import Fraction
from typing import Any

FAMILIES = ('BINDING', 'IMPACT', 'TIES', 'SIGNED', 'INTERVAL', 'K_DIRECTION')
LABELS = ('Mug', 'Lamp', 'Desk', 'Vase', 'Chair', 'Clock', 'Tray', 'Book')

def seed_for(master: int, *parts: Any) -> int:
    payload = json.dumps([master, *parts], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63) - 1)

def exact(x: Any) -> Fraction:
    if x is None or isinstance(x, bool): raise ValueError('not an exact number')
    if isinstance(x, Fraction): out = x
    elif isinstance(x, (int, Decimal)):
        if isinstance(x, Decimal) and (not x.is_finite() or abs(x.adjusted()) > 24): raise ValueError('decimal bounds')
        out = Fraction(x)
    elif isinstance(x, float):
        if not math.isfinite(x): raise ValueError('nonfinite')
        out = Fraction(str(x))
    elif isinstance(x, str):
        s = x.strip()
        if len(s) > 64 or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,2})?|[+-]?\d+\s*/\s*[+-]?\d+', s):
            raise ValueError('number grammar')
        out = Fraction(s.replace(' ', ''))
    else: raise ValueError('number type')
    if abs(out.numerator) > 10**12 or out.denominator > 10**9: raise ValueError('number bounds')
    return out

def enc(x: Fraction | int | None) -> str | None:
    if x is None: return None
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f'{x.numerator}/{x.denominator}'

def finite_decimal(x: Fraction) -> bool:
    d = x.denominator
    for p in (2, 5):
        while d % p == 0: d //= p
    return d == 1

def value_at(w: dict, q: dict, idx: int) -> Fraction:
    if q['value'] == 'diff': return exact(w['series']['Alpha'][idx]) - exact(w['series']['Beta'][idx])
    return exact(w['series'][q['value']][idx])

def reduce_values(values: list[Fraction], aggregate: str) -> Fraction | None:
    if not values: return None
    if aggregate == 'sum': return sum(values, Fraction())
    if aggregate == 'mean': return sum(values, Fraction()) / len(values)
    raise ValueError('unsupported aggregate')

def select_a(w: dict, q: dict) -> list[int]:
    n = len(w['categories'])
    if q['family'] == 'TOPK':
        sign = -1 if q['direction'] == 'largest' else 1
        return sorted(sorted(range(n), key=lambda i: (sign * w['series'][q['selection_series']][i], i))[:q['k']])
    lo, hi = q['start'], q['end']
    indices = [i for i in range(n) if (i > lo or (i == lo and q['include_start'])) and (i < hi or (i == hi and q['include_end']))]
    if q['positive_only']: indices = [i for i in indices if w['series']['Alpha'][i] > w['series']['Beta'][i]]
    return indices

def oracle_a(w: dict, q: dict) -> dict:
    ids = select_a(w, q)
    vals = [value_at(w, q, i) for i in ids]
    return {'indices': ids, 'selected': [w['categories'][i] for i in ids], 'operands': vals, 'answer': reduce_values(vals, q['aggregate'])}

def oracle_b(w: dict, q: dict) -> dict:
    n = len(w['categories'])
    if q['family'] == 'TOPK':
        v = w['series'][q['selection_series']]
        def before(i, j):
            if v[i] == v[j]: return i < j
            return v[i] > v[j] if q['direction'] == 'largest' else v[i] < v[j]
        good = []
        for subset in itertools.combinations(range(n), q['k']):
            remaining = set(range(n)) - set(subset)
            if all(before(i, j) for i in subset for j in remaining): good.append(list(subset))
        if len(good) != 1: raise ValueError('non-unique independent set oracle')
        ids = good[0]
    else:
        interval = list(range(q['start'], q['end'] + 1))
        if not q['include_start']: interval = interval[1:]
        if not q['include_end'] and interval: interval = interval[:-1]
        ids = []
        for i in interval:
            if not q['positive_only'] or exact(w['series']['Alpha'][i]) - exact(w['series']['Beta'][i]) > 0: ids.append(i)
    vals = []
    for i in ids:
        if q['value'] == 'Alpha': vals.append(Fraction(w['series']['Alpha'][i]))
        elif q['value'] == 'Beta': vals.append(Fraction(w['series']['Beta'][i]))
        else: vals.append(Fraction(w['series']['Alpha'][i]) - Fraction(w['series']['Beta'][i]))
    answer = None if not vals else sum(vals, Fraction()) / (len(vals) if q['aggregate'] == 'mean' else 1)
    return {'indices': ids, 'selected': [w['categories'][i] for i in ids], 'operands': vals, 'answer': answer}

def heldout_tags(w: dict, q: dict) -> list[str]:
    out = []
    if q['family'] == 'TOPK':
        if q['selection_series'] == 'Beta' and q['value'] == 'Alpha' and q['aggregate'] == 'mean': out.append('H_BIND')
        if q['selection_series'] == 'Alpha' and q['value'] == 'diff' and q['direction'] == 'smallest' and q['aggregate'] == 'mean': out.append('H_DIFF_BOTTOM')
        sign = -1 if q['direction'] == 'largest' else 1
        vv = sorted(w['series'][q['selection_series']], reverse=sign < 0)
        tie = q['k'] < len(vv) and vv[q['k']-1] == vv[q['k']]
        if tie and q['direction'] == 'smallest': out.append('H_TIE_BOTTOM')
        if q['value'] == 'diff' and q['k'] == 4 and q['aggregate'] == 'mean': out.append('H_DIFF_K4')
        if q['k'] == 4 and q['direction'] == 'smallest' and q['aggregate'] == 'mean': out.append('H_K4_BOTTOM_MEAN')
    elif q['positive_only'] and q['aggregate'] == 'mean': out.append('H_INTERVAL_FILTER_MEAN')
    return out

def fingerprint(w: dict) -> str:
    # Rename/order-invariant fact multiset: conservative cross-split duplicate check.
    return hashlib.sha256(json.dumps(sorted(zip(w['series']['Alpha'], w['series']['Beta']))).encode()).hexdigest()
