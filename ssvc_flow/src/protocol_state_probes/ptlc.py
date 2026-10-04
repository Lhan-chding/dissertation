"""Literal, exact PTLC programs. These are registered diagnostics, not solvers.

All arithmetic is rational; V2 uses its actual fallback value in later steps.
"""

from collections import Counter
from fractions import Fraction


def permutation(order, n=4):
    order = tuple(order)
    if len(order) != n or any(type(k) is not int for k in order) or sorted(order) != list(range(n)):
        raise ValueError("Expected a coordinate permutation")
    return order


def _fraction(value):
    if isinstance(value, (bool, float)):
        raise ValueError("Exact coefficients required; bool and float are forbidden")
    return Fraction(value)


def validate(H, b, observed, order=None):
    n = len(observed)
    if not n or len(H) != len(b) or any(len(row) != n for row in H):
        raise ValueError("Equation shape mismatch")
    order = permutation(range(n) if order is None else order, n)
    return (
        [[_fraction(v) for v in row] for row in H],
        [_fraction(v) for v in b],
        [_fraction(v) for v in observed],
        order,
    )


def run(H, b, observed, order=None, *, variant="V1", low=0, high=99):
    H, b, observed, order = validate(H, b, observed, order)
    if variant not in ("V1", "V2"):
        raise ValueError("Unknown PTLC variant")
    y, done, trace = [None] * len(observed), set(), []
    for k in order:
        votes, ids = [], []
        for ri, (row, rhs) in enumerate(zip(H, b, strict=True)):
            if row[k] and all(j in done for j, a in enumerate(row) if a and j != k):
                votes.append((rhs - sum((row[j] * y[j] for j in done), Fraction(0))) / row[k])
                ids.append(ri)
        if not votes:
            chosen, reason = observed[k], "NO_CLOSED_ROW_COPY"
        else:
            counts = Counter(votes)
            mode, freq = counts.most_common(1)[0]
            if freq * 2 > len(votes):
                chosen = mode
                reason = "AGREE" if len(counts) == 1 else "STRICT_MAJORITY"
            else:
                chosen, reason = observed[k], "CONFLICT_FALLBACK_COPY"
        proposal = chosen
        if variant == "V2" and not low <= chosen <= high:
            chosen = observed[k]
            reason += "__DOMAIN_FALLBACK"
        y[k] = chosen
        done.add(k)
        trace.append(
            {
                "coordinate": k,
                "closed_rows": ids,
                "votes": [str(v) for v in votes],
                "proposal": str(proposal),
                "value": str(chosen),
                "reason": reason,
            }
        )
    return y, trace


def external(y):
    return [int(v) if Fraction(v).denominator == 1 else str(Fraction(v)) for v in y]


def one_row(H, coordinate, order=None):
    order = permutation(range(4) if order is None else order)
    if type(coordinate) is not int or coordinate not in order:
        raise ValueError("Invalid coordinate")
    before = set(order[: order.index(coordinate)])
    return any(
        row[coordinate] and all(j in before for j, a in enumerate(row) if a and j != coordinate)
        for row in H
    )


def anchors(H, order):
    return [k for k in permutation(order) if not one_row(H, k, order)]


def dpe(H, coordinate, order):
    return bool(one_row(H, coordinate, order))


def _rank(matrix):
    rows = [[Fraction(x) for x in row] for row in matrix]
    rank = 0
    for col in range(len(rows[0]) if rows else 0):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for i in range(len(rows)):
            if i != rank:
                scale = rows[i][col]
                rows[i] = [x - scale * y for x, y in zip(rows[i], rows[rank], strict=True)]
        rank += 1
    return rank


def transform_star(H, b, order=None):
    """Return the exact invertible B1 row transformation and its fixed first leaf."""
    order = permutation(range(4) if order is None else order)
    if len(H) != 3 or len(b) != 3 or any(len(row) != 4 for row in H):
        raise ValueError("B1 requires a four-coordinate, three-row sum star")
    supports = [{j for j, value in enumerate(row) if value} for row in H]
    centers = set.intersection(*supports)
    if any(len(s) != 2 for s in supports) or len(centers) != 1:
        raise ValueError("B1 requires a unique star center")
    center = next(iter(centers))
    if any(_fraction(value) != 1 for row in H for value in row if value):
        raise ValueError("B1 requires unit sum relations")
    by_leaf = {
        next(iter(s - {center})): (row, rhs, ri)
        for ri, (s, row, rhs) in enumerate(zip(supports, H, b, strict=True))
    }
    if len(by_leaf) != 3:
        raise ValueError("B1 requires distinct leaves")
    first = next(k for k in order if k != center)
    ref, ref_rhs, ref_id = by_leaf[first]
    HH, bb = [list(ref)], [ref_rhs]
    T = [[int(i == ref_id) for i in range(3)]]
    for k in order:
        if k in (center, first):
            continue
        row, rhs, ri = by_leaf[k]
        HH.append([u - v for u, v in zip(row, ref, strict=True)])
        bb.append(rhs - ref_rhs)
        T.append([int(i == ri) - int(i == ref_id) for i in range(3)])
    exact_H = [
        [sum(Fraction(t) * Fraction(H[i][j]) for i, t in enumerate(tr)) for j in range(4)]
        for tr in T
    ]
    exact_b = [sum(Fraction(t) * Fraction(b[i]) for i, t in enumerate(tr)) for tr in T]
    if exact_H != HH or exact_b != bb or _rank(T) != 3:
        raise ValueError("B1 failed its exact invertible row-transform contract")
    return HH, bb, {"center": center, "first_leaf": first, "row_transform": T}
