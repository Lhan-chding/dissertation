"""Geometry-only acquisition of complete calibration-bank blocks.

There is deliberately no response, truth, callback, or query argument. Every
bank's realized Adam contrasts are one acquisition unit, including zero rows.
"""

from __future__ import annotations

import numpy as np

from .coverage import build_subspace

SELECTION_RULES = ("FIRST", "STRATIFIED_RANDOM", "LARGEST_NORM", "BLOCK_PIVOT_QR", "BLOCK_LOGDET")


def select_banks(
    updates, n_banks, method, *, bank_ids=None, strata=None, seed=0, logdet_ridge=1e-8
):
    """Select whole banks using fit-pool parameter differences only.

    BLOCK_PIVOT_QR greedily maximizes block Frobenius residual energy against
    the uncentered selected span. BLOCK_LOGDET greedily maximizes determinant
    gain in the exact numerical geometry of the whole calibration pool, with
    an absolute, caller-frozen ridge. Ties use original pool order.
    """
    e = np.asarray(updates, dtype=np.float64)
    if e.ndim != 3 or e.shape[1] == 0 or e.shape[2] == 0 or not np.isfinite(e).all():
        raise ValueError("updates must be finite [bank, contrast, parameter] blocks")
    if (
        isinstance(n_banks, (bool, np.bool_))
        or not isinstance(n_banks, (int, np.integer))
        or not 0 <= n_banks <= len(e)
    ):
        raise ValueError("n_banks must be an integer within the candidate pool size")
    if method not in SELECTION_RULES:
        raise ValueError(f"unknown bank selection rule {method}")
    if not np.isfinite(logdet_ridge) or logdet_ridge <= 0:
        raise ValueError("logdet_ridge must be finite and positive")
    ids = list(range(len(e))) if bank_ids is None else list(bank_ids)
    if len(ids) != len(e) or len(set(ids)) != len(ids):
        raise ValueError("bank_ids must uniquely identify all candidate blocks")
    if strata is not None and len(strata) != len(e):
        raise ValueError("strata must match candidate bank count")
    rng = np.random.default_rng(seed)
    steps = []
    if method == "FIRST":
        selected = list(range(n_banks))
    elif method == "STRATIFIED_RANDOM":
        # A stratum is metadata declared before semantic measurements. Cycling
        # randomly ordered strata keeps early prefixes balanced as well.
        keys = ["all"] * len(e) if strata is None else list(strata)
        groups = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)
        queues = [list(rng.permutation(indices)) for indices in groups.values()]
        selected = []
        while len(selected) < n_banks:
            for j in rng.permutation(len(queues)):
                if queues[j] and len(selected) < n_banks:
                    selected.append(int(queues[j].pop()))
    elif method == "LARGEST_NORM":
        energy = np.square(e).sum(axis=(1, 2))
        selected = np.argsort(-energy, kind="stable")[:n_banks].tolist()
        steps = [{"bank_index": i, "score": float(energy[i])} for i in selected]
    elif method == "BLOCK_PIVOT_QR":
        selected = []
        q = np.empty((e.shape[2], 0))
        for _ in range(n_banks):
            residual = e - (e @ q) @ q.T
            scores = np.square(residual).sum(axis=(1, 2))
            scores[selected] = -np.inf
            i = int(np.argmax(scores))
            steps.append({"bank_index": i, "score": float(scores[i])})
            selected.append(i)
            q, _ = build_subspace(e[selected].reshape(-1, e.shape[2]))
    else:
        q, _ = build_subspace(e.reshape(-1, e.shape[2]))
        coordinates = e @ q
        normal = logdet_ridge * np.eye(q.shape[1])
        selected = []
        for _ in range(n_banks):
            baseline = np.linalg.slogdet(normal)[1]
            scores = np.full(len(e), -np.inf)
            for i, block in enumerate(coordinates):
                if i not in selected:
                    scores[i] = np.linalg.slogdet(normal + block.T @ block)[1] - baseline
            i = int(np.argmax(scores))
            selected.append(i)
            normal += coordinates[i].T @ coordinates[i]
            steps.append({"bank_index": i, "score": float(scores[i])})
    selected_updates = e[selected].copy()
    _, geometry = build_subspace(selected_updates.reshape(-1, e.shape[2]))
    return {
        "method": method,
        "selected_indices": selected,
        "selected_bank_ids": [ids[i] for i in selected],
        "selected_updates": selected_updates,
        "geometry": geometry,
        "audit": {
            "selection_seed": int(seed),
            "pool_bank_count": len(e),
            "selected_bank_count": int(n_banks),
            "contrasts_per_bank": int(e.shape[1]),
            "pool_geometry_contrasts": int(e.shape[0] * e.shape[1]),
            "pool_geometry_generation_cost_must_be_charged": True,
            "bank_acquisition_unit": "ALL_CONTRASTS",
            "heldout_responses_read": False,
            "fit_responses_read": False,
            "centered": False,
            "stratification": "SUPPLIED_METADATA" if strata is not None else "SINGLE_STRATUM",
            "logdet_ridge_absolute": float(logdet_ridge),
            "tie_break": "ORIGINAL_POOL_ORDER",
            "steps": steps,
        },
    }
