"""Nested whole-bank selection from training metadata and the full input Gram.

There is no response-label or query argument. The QR selector changes which
calibration banks are measured; it does not truncate the fitted model input.
"""

from __future__ import annotations

import numpy as np

from .config import digest


def nested_bank_order(
    bank_ids,
    calibration_gram,
    *,
    columns_per_bank=3,
    strata=None,
    method="BLOCK_PIVOT_QR",
    seed=0,
):
    """Return all bank indices once, allowing shared prefixes at each m.

    Rows of the Gram must be contiguous complete banks, with contrasts in the
    same order for every bank. Block QR greedily maximizes the sum of squared
    remaining column lengths. All directions in a selected block join the QR
    basis up to machine precision. No semantic outcomes enter this ordering.

    Stratified random uses a seed-hashed order within each metadata stratum,
    then cycles through strata. A missing strata argument explicitly means one
    stratum; callers must record the actual supplied metadata partition.
    """
    ids = list(bank_ids)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
        raise ValueError("Unique nonempty actual bank IDs required")
    if type(columns_per_bank) is not int or columns_per_bank < 1 or type(seed) is not int:
        raise ValueError("Positive whole-bank block size and integer seed required")
    gram = np.asarray(calibration_gram, dtype=np.float64)
    size = len(ids) * columns_per_bank
    if gram.shape != (size, size) or not np.isfinite(gram).all():
        raise ValueError("Finite full calibration Gram must cover every bank and contrast")
    scale = float(np.max(np.abs(gram)))
    tolerance = np.finfo(np.float64).eps * max(size, 1) * 64
    if scale and np.max(np.abs(gram - gram.T)) > tolerance * scale:
        raise ValueError("Calibration Gram must be symmetric")
    if method == "STRATIFIED_RANDOM":
        groups = ["ALL"] * len(ids) if strata is None else list(strata)
        if len(groups) != len(ids):
            raise ValueError("One metadata stratum per whole bank is required")
        pools = {}
        for index, group in enumerate(groups):
            pools.setdefault(digest(group), []).append(index)
        for indices in pools.values():
            indices.sort(key=lambda i: digest(["V4_BANK_RANDOM", seed, ids[i]]))
        ordered = sorted(pools, key=lambda g: digest(["V4_STRATUM_ORDER", seed, g]))
        return np.asarray(
            [
                pools[group][position]
                for position in range(max(map(len, pools.values())))
                for group in ordered
                if position < len(pools[group])
            ],
            dtype=np.int64,
        )
    if method != "BLOCK_PIVOT_QR":
        raise ValueError("Only the two registered V4 calibration selectors are supported")
    if not scale:
        return np.asarray(sorted(range(len(ids)), key=lambda i: ids[i]), dtype=np.int64)
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / (2 * scale))
    if float(eigenvalues.min()) < -tolerance * max(float(eigenvalues.max()), 1.0):
        raise ValueError("Calibration Gram is not positive semidefinite")
    # Gram coordinates retain every positive eigenvalue, irrespective of the
    # later model's rank diagnostics. Only roundoff-negative eigenvalues vanish.
    residual = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[None, :]
    original_norm = float(np.sqrt(max(float(eigenvalues.max()), 0)))
    direction_tolerance = np.finfo(np.float64).eps * size * original_norm * 8
    remaining, order = set(range(len(ids))), []
    for _ in ids:
        scores = np.einsum("ij,ij->i", residual, residual).reshape(-1, columns_per_bank).sum(1)
        pivot = min(remaining, key=lambda i: (-float(scores[i]), ids[i]))
        order.append(pivot)
        remaining.remove(pivot)
        block = residual[pivot * columns_per_bank : (pivot + 1) * columns_per_bank]
        _, values, directions = np.linalg.svd(block, full_matrices=False)
        active = directions[values > direction_tolerance]
        # Reorthogonalization limits drift in nearly dependent three-contrast
        # blocks, where one contrast is the difference of the other two.
        if len(active):
            for _pass in range(2):
                residual -= (residual @ active.T) @ active
    return np.asarray(order, dtype=np.int64)
