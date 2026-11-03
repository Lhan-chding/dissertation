"""SR-F3-CE: exact, truth-anchored, sound (not complete) certificate contracts.
No model calls. Public domains contain no hidden generator selection constraints.
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, DecimalException
from fractions import Fraction
from itertools import pairwise, product
from typing import Any

FAMILIES = ("CROSS", "THRESHOLD", "TOPK", "INTERVAL")
NUM = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,3})?\Z")
FRAC = re.compile(r"([+-]?[0-9]+)\s*/\s*([+-]?[0-9]+)\Z")
COMMA = re.compile(r"[+-]?(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?\Z")


def rational(x: Any) -> Fraction:
    if isinstance(x, bool) or x is None:
        raise ValueError("not a scalar number")
    if isinstance(x, Fraction):
        f = x
    elif isinstance(x, int):
        f = Fraction(x)
    elif isinstance(x, Decimal):
        if not x.is_finite() or abs(x.adjusted()) > 30:
            raise ValueError("decimal bounds")
        f = Fraction(x)
    elif isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError("nonfinite")
        f = Fraction(str(x))
    elif isinstance(x, str):
        if len(x) > 64:
            raise ValueError("numeric length")
        t = x.strip().replace("\u2212", "-")
        if FRAC.fullmatch(t):
            m = FRAC.fullmatch(t)
            den = int(m.group(2))
            if den == 0:
                raise ValueError("zero denominator")
            f = Fraction(int(m.group(1)), den)
        elif NUM.fullmatch(t):
            dec = Decimal(t)
            if dec and abs(dec.adjusted()) > 30:
                raise ValueError("decimal bounds")
            f = Fraction(dec)
        else:
            raise ValueError("not exact scalar")
    else:
        raise ValueError("not exact scalar")
    if abs(f.numerator) > 10**12 or f.denominator > 10**9:
        raise ValueError("fraction bounds")
    return f


def encode(x):
    if x is None:
        return None
    x = rational(x)
    return x.numerator if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def canonical(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate JSON key")
        d[k] = v
    return d


def loads(text):
    if not isinstance(text, str) or len(text) > 64000:
        raise ValueError("text bounds")
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=Decimal,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
    except (DecimalException, OverflowError) as exc:
        raise ValueError("invalid JSON decimal") from exc


def validate_world(w):
    cats = w["categories"]
    lo, hi = w["domain"]
    if not (
        2 <= len(cats) <= 8
        and len(set(cats)) == len(cats)
        and type(lo) is int
        and type(hi) is int
        and lo <= hi
    ):
        raise ValueError("world schema")
    if not {"Alpha", "Beta"} <= set(w["series"]):
        raise ValueError("missing series")
    for vs in w["series"].values():
        if len(vs) != len(cats) or any(type(v) is not int or not lo <= v <= hi for v in vs):
            raise ValueError("world domain")


def selected(w, q):
    a = w["series"]["Alpha"]
    b = w["series"]["Beta"]
    n = len(a)
    f = q["family"]
    if f == "CROSS":
        return [
            i
            for i in range(n)
            if (a[i] > b[i] if q.get("comparator", "gt") == "gt" else a[i] >= b[i])
        ]
    if f == "THRESHOLD":
        t = q["threshold"]
        return [
            i
            for i in range(n)
            if (a[i] >= t if q.get("comparator", "ge") == "ge" else a[i] > t)
        ]
    if f == "TOPK":
        sign = -1 if q.get("direction", "largest") == "largest" else 1
        return sorted(range(n), key=lambda i: (sign * a[i], i))[: q.get("k", 2)]
    if f == "INTERVAL":
        start = q["start"] + (0 if q.get("include_start", True) else 1)
        end = q["end"]
        inds = list(range(start, end + 1))
        if q.get("positive_only", False):
            inds = [i for i in inds if a[i] > b[i]]
        return inds
    raise ValueError("family")


def operand(w, q, i):
    mode = q.get(
        "value",
        "Beta"
        if q["family"] == "TOPK"
        else "diff"
        if q["family"] == "INTERVAL"
        else "Alpha",
    )
    a = w["series"]["Alpha"][i]
    b = w["series"]["Beta"][i]
    if mode == "Alpha":
        return Fraction(a)
    if mode == "Beta":
        return Fraction(b)
    if mode == "diff":
        return Fraction(a - b)
    raise ValueError("value")


def aggregate(vals, op):
    if op == "count":
        return Fraction(len(vals))
    if op == "sum":
        return sum(vals, Fraction())
    if not vals:
        return None
    if op == "mean":
        return sum(vals, Fraction()) / len(vals)
    if op == "range":
        return max(vals) - min(vals)
    if op == "max":
        return max(vals)
    if op == "min":
        return min(vals)
    raise ValueError("aggregate")


def execute(w, q):
    return aggregate([operand(w, q, i) for i in selected(w, q)], q["aggregate"])


def full_keys(w, q):
    """Historical-like coverage; not used as the new sufficient reward."""
    n = len(w["categories"])
    f = q["family"]
    if f == "THRESHOLD" and q.get("value", "Alpha") == "Alpha":
        return {("Alpha", i) for i in range(n)}
    if f == "INTERVAL":
        start = q["start"] + (0 if q.get("include_start", True) else 1)
        return {(s, i) for s in ("Alpha", "Beta") for i in range(start, q["end"] + 1)}
    return {(s, i) for s in ("Alpha", "Beta") for i in range(n)}


def _value_keys(q, inds):
    if q["aggregate"] == "count":
        return set()
    mode = q.get(
        "value",
        "Beta"
        if q["family"] == "TOPK"
        else "diff"
        if q["family"] == "INTERVAL"
        else "Alpha",
    )
    cols = ["Alpha", "Beta"] if mode == "diff" else [mode]
    return {(s, i) for i in inds for s in cols}


def sufficient_template(w, q, facts):
    """Sound acceptance using reported facts + public domain only.
    Does not use hidden values to decide selected sets. May reject valid certificates.
    """
    lo, hi = w["domain"]
    n = len(w["categories"])
    f = q["family"]
    chosen = []
    if f == "CROSS":
        for i in range(n):
            av = facts.get(("Alpha", i))
            bv = facts.get(("Beta", i))
            if av is not None and bv is not None:
                yes = av > bv if q.get("comparator", "gt") == "gt" else av >= bv
                if yes:
                    chosen.append(i)
            elif q.get("comparator", "gt") == "gt" and (av == lo or bv == hi):
                pass
            else:
                return False
    elif f == "THRESHOLD":
        t = q["threshold"]
        ge = q.get("comparator", "ge") == "ge"
        for i in range(n):
            av = facts.get(("Alpha", i))
            if av is not None:
                yes = av >= t if ge else av > t
            elif lo >= t if ge else lo > t:
                yes = True
            elif hi < t if ge else hi <= t:
                yes = False
            else:
                return False
            if yes:
                chosen.append(i)
    elif f == "TOPK":
        if any(("Alpha", i) not in facts for i in range(n)):
            return False
        sign = -1 if q.get("direction", "largest") == "largest" else 1
        chosen = sorted(range(n), key=lambda i: (sign * facts["Alpha", i], i))[
            : q.get("k", 2)
        ]
    elif f == "INTERVAL":
        start = q["start"] + (0 if q.get("include_start", True) else 1)
        chosen = list(range(start, q["end"] + 1))
        if q.get("positive_only", False):
            if any((s, i) not in facts for i in chosen for s in ("Alpha", "Beta")):
                return False
            chosen = [i for i in chosen if facts["Alpha", i] > facts["Beta", i]]
        elif q["aggregate"] in ("max", "min") and q.get("value", "diff") == "diff":
            extreme = hi - lo if q["aggregate"] == "max" else lo - hi
            if any(
                ("Alpha", i) in facts
                and ("Beta", i) in facts
                and facts["Alpha", i] - facts["Beta", i] == extreme
                for i in chosen
            ):
                return True
    else:
        raise ValueError("family")
    return _value_keys(q, chosen) <= set(facts)


def oracle_facts(w, keys):
    return {k: Fraction(w["series"][k[0]][k[1]]) for k in keys}


def canonical_certificate(w, q):
    keys = set(full_keys(w, q))
    # Deterministic deletion produces an irreducible certificate in this TEMPLATE
    # family, not a globally minimum certificate. All labels remain exact.
    for k in sorted(keys):
        cand = keys - {k}
        if sufficient_template(w, q, oracle_facts(w, cand)):
            keys = cand
    return [
        {"series": s, "category": w["categories"][i], "value": w["series"][s][i]}
        for s, i in sorted(keys)
    ]


def parse_facts(items, w):
    if not isinstance(items, list) or len(items) > 32:
        raise ValueError("evidence length")
    d = {}
    for t in items:
        if not isinstance(t, dict) or set(t) != {"series", "category", "value"}:
            raise ValueError("fact fields")
        s = t["series"]
        c = t["category"]
        if s not in w["series"] or c not in w["categories"]:
            raise ValueError("unknown entity")
        k = s, w["categories"].index(c)
        if k in d:
            raise ValueError("duplicate fact")
        val = rational(t["value"])
        lo, hi = w["domain"]
        if val.denominator != 1 or not lo <= val <= hi:
            raise ValueError("fact domain")
        d[k] = val
    return d


def first_object_end(text):
    """End of the first object at the start of output; never search through junk."""
    if not isinstance(text, str):
        return None
    start = len(text) - len(text.lstrip())
    if start == len(text) or text[start] != "{":
        return None
    depth = 0
    quoted = False
    escaped = False
    for position, char in enumerate(text[start:], start):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return position + 1
    return None


def registered_completion(text, protocol):
    """Structural completion only; incorrect scalars do not erase this dimension."""
    if not isinstance(text, str):
        return False
    if protocol == "natural_work":
        start = text.find("<final>")
        return start >= 0 and text.find("</final>", start + len("<final>")) >= 0
    return first_object_end(text) is not None


def score(text, w, q, *, finish_reason="balanced", protocol="structured"):
    if protocol not in ("structured", "answer_only", "natural_work"):
        raise ValueError("unregistered synthetic protocol")
    ans_true = execute(w, q)
    out = dict(
        value_correct=0,
        value_parseable=0,
        presentation=0,
        finish=int(
            finish_reason
            in (
                ("eos", "final_tag")
                if protocol == "natural_work"
                else ("balanced", "eos")
            )
            and registered_completion(text, protocol)
        ),
        certificate=0,
        certificate_truth=0,
        certificate_sufficient=0,
        J=0,
        legacy_full=0,
        reported_facts=0,
        work_present=0,
        reason="invalid",
    )
    if protocol == "natural_work":
        try:
            if (
                not isinstance(text, str)
                or text.count("<final>") != 1
                or text.count("</final>") != 1
            ):
                raise ValueError("one final tag required")
            end = text.index("</final>") + len("</final>")
            answer = extract_final_tag(text[:end])
            out.update(
                value_parseable=1,
                value_correct=int(answer == ans_true),
                presentation=int(not text[end:].strip()),
                reason="value_only_protocol",
            )
        except (ValueError, TypeError, RecursionError):
            pass
        return out
    try:
        end = first_object_end(text)
        if end is None:
            raise ValueError("no initial complete JSON object")
        obj = loads(text[:end])
        if not isinstance(obj, dict):
            raise ValueError("root type")
    except (ValueError, TypeError, RecursionError, json.JSONDecodeError):
        return out
    try:
        y = None if obj["answer"] is None else rational(obj["answer"])
        out["value_parseable"] = 1
        out["value_correct"] = int(y == ans_true)
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        pass
    if protocol == "answer_only":
        schema = set(obj) == {"answer"}
    else:
        schema = (
            set(obj) <= {"evidence", "work", "answer"}
            and "answer" in obj
            and "evidence" in obj
            and ("work" not in obj or isinstance(obj["work"], str))
        )
    if "work" in obj and (
        len(obj["work"]) > 16000 if isinstance(obj["work"], str) else True
    ):
        schema = False
    # Order is not enforced by the VALUE/CERTIFICATE validator. The training prompt
    # asks work before answer; order compliance is a separate logged diagnostic.
    out["presentation"] = int(
        schema and out["value_parseable"] and not text[end:].strip()
    )
    out["nonwhitespace_suffix"] = bool(text[end:].strip())
    out["work_present"] = int(isinstance(obj.get("work"), str) and bool(obj["work"]))
    out["field_order"] = list(obj)
    try:
        facts = parse_facts(obj["evidence"], w)
        out["reported_facts"] = len(facts)
        truth = all(v == w["series"][s][i] for (s, i), v in facts.items())
        suff = sufficient_template(w, q, facts)
        out.update(
            certificate_truth=int(truth),
            certificate_sufficient=int(suff),
            certificate=int(truth and suff),
            legacy_full=int(truth and full_keys(w, q) <= set(facts)),
        )
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        if protocol == "structured":
            out["presentation"] = 0
    out["J"] = int(
        out["value_correct"]
        and out["certificate"]
        and out["presentation"]
        and out["finish"]
    )
    if out["J"]:
        out["reason"] = "ok"
    elif out["certificate_truth"] and not out["certificate_sufficient"]:
        out["reason"] = "not_proved_by_registered_template"
    else:
        out["reason"] = "contract_not_met"
    return out


def trace(w, q):
    inds = selected(w, q)
    cats = w["categories"]
    a = w["series"]["Alpha"]
    b = w["series"]["Beta"]
    f = q["family"]
    if f == "CROSS":
        op = ">" if q.get("comparator", "gt") == "gt" else ">="
        checks = [
            f"{cats[i]}:{a[i]}{op}{b[i]}={int(i in inds)}" for i in range(len(cats))
        ]
    elif f == "THRESHOLD":
        op = ">=" if q.get("comparator", "ge") == "ge" else ">"
        checks = [
            f"{cats[i]}:{a[i]}{op}{q['threshold']}={int(i in inds)}"
            for i in range(len(cats))
        ]
    elif f == "TOPK":
        checks = ["ranked_selection=" + ",".join(cats[i] for i in inds)]
    else:
        checks = ["interval_selection=" + ",".join(cats[i] for i in inds)]
    vals = [operand(w, q, i) for i in inds]
    if q.get("value", "diff" if f == "INTERVAL" else "") == "diff":
        checks += [f"{cats[i]}:{a[i]}-({b[i]})={a[i] - b[i]}" for i in inds]
    checks.append("operands=[" + ",".join(str(encode(v)) for v in vals) + "]")
    op = q["aggregate"]
    if op == "mean" and vals:
        checks.append(f"sum={encode(sum(vals, Fraction()))};count={len(vals)}")
    if op == "range" and vals:
        checks.append(f"minimum={encode(min(vals))};maximum={encode(max(vals))}")
    # No explicit final aggregate answer is written into the teacher work prefix.
    # It supplies a verified decomposition, not a second copy of the target answer.
    checks.append("reduce=" + op + "(operands)")
    return "; ".join(checks)


def support_target(w, q, with_trace):
    obj = {"evidence": canonical_certificate(w, q)}
    if with_trace:
        obj["work"] = trace(w, q)
    obj["answer"] = encode(execute(w, q))
    text = canonical(obj)
    # Supervised answer suffix includes the answer key and structural terminator.
    a_start = text.rfind(',"answer":') + 1
    spans = {"answer": [a_start, len(text)]}
    if with_trace:
        work_serial = canonical(obj["work"])
        start = text.index('"work":') + len('"work":')
        spans["work"] = [start, start + len(work_serial)]
    return {"text": text, "spans": spans, "object": obj}


def mask_offsets(offsets, span):
    """Entirely contained tokens only; crossing semantic fields is a hard error."""
    lo, hi = span
    m = []
    for a, b in offsets:
        intersects = b > a and max(a, lo) < min(b, hi)
        if intersects and not (lo <= a and b <= hi):
            raise ValueError(
                "cross-boundary token; use runtime registered boundary mapping"
            )
        m.append(int(intersects))
    if not any(m):
        raise ValueError("empty loss mask")
    return m


def possible_answers(w, q, facts, max_worlds=200000):
    """Tiny-domain exhaustive specification ORACLE, not rollout production code."""
    keys = [
        (s, i)
        for s in ("Alpha", "Beta")
        for i in range(len(w["categories"]))
        if (s, i) not in facts
    ]
    lo, hi = w["domain"]
    count = (hi - lo + 1) ** len(keys)
    if count > max_worlds:
        raise ValueError("enumeration budget exceeded")
    vals = set()
    for assignment in product(range(lo, hi + 1), repeat=len(keys)):
        ww = {
            "categories": w["categories"],
            "domain": w["domain"],
            "series": {
                "Alpha": [lo] * len(w["categories"]),
                "Beta": [lo] * len(w["categories"]),
            },
        }
        for (s, i), v in list(facts.items()) + list(zip(keys, assignment, strict=True)):
            ww["series"][s][i] = int(v)
        vals.add(execute(ww, q))
    return vals


def normalize_typed(text, kind):
    """No gold argument. Unsupported/ambiguous outputs raise rather than guess."""
    if not isinstance(text, str):
        raise ValueError("text")
    t = text.strip().replace("\u2212", "-")
    percent = False
    if t.endswith("%"):
        t = t[:-1].strip()
        percent = True
    elif t.lower().endswith(" percent"):
        t = t[:-8].strip()
        percent = True
    if "," in t:
        if not COMMA.fullmatch(t):
            raise ValueError("invalid grouping")
        t = t.replace(",", "")
    f = rational(t)
    if kind == "PERCENT_POINT":
        return f
    if kind == "RATIO_FRACTION":
        return f / 100 if percent else f
    if percent:
        raise ValueError("unit not permitted for this type")
    if kind in ("YEAR", "COUNT_INT"):
        if f.denominator != 1:
            raise ValueError("integer required")
        return f
    if kind == "GENERAL_NUMERIC":
        return f
    raise ValueError("unknown or textual type: requires registered nonnumeric scorer")


def extract_final_tag(text):
    """Only one designated final span. Never search for a gold-matching number."""
    if (
        not isinstance(text, str)
        or text.count("<final>") != 1
        or text.count("</final>") != 1
    ):
        raise ValueError("one final tag required")
    a = text.index("<final>") + len("<final>")
    b = text.index("</final>")
    if b < a or text[b + len("</final>") :].strip():
        raise ValueError("bad final ordering or suffix")
    val = text[a:b].strip()
    # The common frozen semantics promise quoted fraction strings. Accept that
    # single scalar form as well as a bare scalar, without any gold-dependent search.
    if val.startswith('"'):
        val = loads(val)
        if not isinstance(val, str):
            raise ValueError("quoted final must be a scalar string")
        return rational(val)
    return None if val == "null" else rational(val)


def prefix_completion(text, character_end_positions, budgets, protocol="structured"):
    """Descriptive trajectory-prefix diagnostic; input positions come from real tokens."""
    if protocol not in ("structured", "answer_only", "natural_work"):
        raise ValueError("unregistered synthetic protocol")
    if any(
        type(p) is not int or p < 0 or p > len(text) for p in character_end_positions
    ):
        raise ValueError("invalid character end positions")
    if any(type(b) is not int or b <= 0 for b in budgets):
        raise ValueError("budgets must be positive integers")
    result = {}
    if any(b < a for a, b in pairwise(character_end_positions)):
        raise ValueError("nonmonotone offsets")
    for budget in budgets:
        end = (
            character_end_positions[min(budget, len(character_end_positions)) - 1]
            if character_end_positions and budget > 0
            else 0
        )
        prefix = text[:end]
        try:
            if protocol == "natural_work":
                extract_final_tag(prefix)
            else:
                obj = loads(prefix)
                if not isinstance(obj, dict) or "answer" not in obj:
                    raise ValueError("no answer")
                if obj["answer"] is not None:
                    rational(obj["answer"])
            result[str(budget)] = True
        except (ValueError, TypeError, json.JSONDecodeError):
            result[str(budget)] = False
    return result
