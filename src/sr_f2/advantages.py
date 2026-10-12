"""SR-F2 final advantages, passed unchanged into the shared SR-F1.2 loss."""

from __future__ import annotations

import numpy as np

from sr_f1.contract import FAMILIES, reference_module

_reference = reference_module("semantic_contract")
zscore = _reference.zscore
ARMS = ("EARLY", "J6", "DEC_PRIME", "ADAPT")
EARLY_GATE_LAST_STEP = 12
GATE_TOLERANCE = 1e-9
DEC_PRIME_ABSOLUTE_BOUND = 2 * np.sqrt(7)


def gate_one(j, e, p, kappa=0.25, tol=GATE_TOLERANCE):
    """Correct the constant-failed-h rounding defect without changing branches."""
    j, e, p = (np.asarray(value, dtype=np.float64) for value in (j, e, p))
    if j.ndim != 1 or not len(j) or j.shape != e.shape or j.shape != p.shape:
        raise ValueError("GATE components must be aligned nonempty vectors")
    if not 0 <= kappa < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Invalid GATE refinement constants")
    if not (np.isin(j, [0, 1]).all() and np.isin(e, [0, 1]).all()):
        raise ValueError("Invalid binary GATE scores")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or np.any(j > e) or np.any(j > p):
        raise ValueError("Invalid GATE reading or joint scores")
    if np.any(j) and not np.all(j):
        advantage = zscore(j)
        failed = j == 0
        h = 0.5 * e * (1 + p)
        failed_h = h[failed]
        if np.ptp(failed_h) > tol:
            centered = failed_h - failed_h.mean()
            scale = np.max(np.abs(centered))
            margin = np.min(-advantage[failed])
            adjustment = kappa * margin * centered / scale
            assert abs(adjustment.sum()) < tol, "Failure refinement is not zero-sum"
            assert np.all(np.abs(adjustment) <= kappa * margin + tol), "Refinement exceeds bound"
            advantage[failed] += adjustment
            assert np.all(advantage[failed] < 0), "A failed response received positive credit"
        return advantage, "target_J_refined"
    if np.all(j):
        return np.zeros_like(j), "all_joint_correct"
    if np.ptp(e) > 0:
        return zscore(e), "evidence"
    if np.all(e == 1) and np.ptp(p) > 0:
        return zscore(p), "readings"
    return np.zeros_like(j), "no_semantic_contrast"


def _components(scores):
    if len(scores) != 16 or any(len(group) != 8 for group in scores):
        raise ValueError("SR-F2 final advantages require the entire 16 x 8 update")
    a, j, e, p = (
        np.asarray([[row[key] for row in group] for group in scores], dtype=np.float64)
        for key in ("A", "J", "E", "p_read")
    )
    if not all(np.isin(value, [0, 1]).all() for value in (a, j, e)):
        raise ValueError("Invalid binary component scores")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid reading scores")
    if np.any(j > a) or np.any(j > e) or np.any(j > p):
        raise ValueError("Joint success contract violated")
    return a, j, e, p


def reward_advantages(arm, scores, *, step, families=None, active_families=()):
    """Return exactly 16 x 8 final coefficients and a JSON-serializable audit.

    ``step`` is the one-based update index. No caller may normalize or clip the
    returned coefficients. DEC_PRIME's registered weighted sum has bound
    2*sqrt(7), not the plan checklist's single-component bound of 2.65.
    """
    arm = {"DEC\u2032": "DEC_PRIME", "DEC'": "DEC_PRIME"}.get(arm, arm)
    if arm not in (*ARMS, "A", "J", "GATE", "DEC"):
        raise ValueError("Unknown SR-F2 reward arm")
    if type(step) is not int or not 1 <= step <= 96:
        raise ValueError("One-based update step must be in [1, 96]")
    a, j, e, p = _components(scores)
    active = frozenset(active_families)
    if not active <= set(FAMILIES):
        raise ValueError("Unknown activated family")
    if families is not None and (len(families) != 16 or any(f not in FAMILIES for f in families)):
        raise ValueError("Family routing requires sixteen known families")
    if arm == "ADAPT" and families is None:
        raise ValueError("ADAPT requires one family for each complete group")
    effective = "GATE" if arm == "EARLY" and step <= EARLY_GATE_LAST_STEP else arm
    if arm == "EARLY" and step > EARLY_GATE_LAST_STEP:
        effective = "J"
    info = dict(
        arm=arm,
        effective_arm=effective,
        step=step,
        group_size=8,
        prompt_count=16,
        branches=[],
        std_convention="population",
        normalization_epsilon=_reference.EPS,
        scalar_reward=None,
        active_families=sorted(active),
    )
    if effective in ("A", "J", "J6", "ADAPT"):
        reward = a if effective == "A" else j
        if effective == "J6":
            reward = np.asarray([[row["J6"] for row in group] for group in scores], dtype=float)
            if not np.isin(reward, [0, 1]).all() or np.any(reward > a):
                raise ValueError("Invalid sufficient-certificate joint event")
            for group_index, group in enumerate(scores):
                for row_index, row in enumerate(group):
                    if "P6" in row and reward[group_index, row_index] > row["P6"]:
                        raise ValueError("J6 exceeds P6")
                    if "E6" in row and reward[group_index, row_index] > row["E6"]:
                        raise ValueError("J6 exceeds E6")
                if (
                    families is not None
                    and families[group_index] != "TOPK"
                    and not np.array_equal(reward[group_index], j[group_index])
                ):
                    raise ValueError("J6 changes a non-TOPK event")
        if effective == "ADAPT":
            uses_semantics = np.asarray([family in active for family in families], dtype=bool)
            reward = np.where(uses_semantics[:, None], j, a)
            info["group_reward_arms"] = ["J" if value else "A" for value in uses_semantics]
            info["semantic_groups"] = int(uses_semantics.sum())
            info["semantic_group_fraction"] = float(uses_semantics.mean())
        advantage = np.stack([zscore(row) for row in reward])
        info["scalar_reward"] = reward.tolist()
    elif effective in ("DEC_PRIME", "DEC"):
        components = np.stack([j, e, e * p], axis=-1)
        normalized = np.empty_like(components)
        for group_index in range(16):
            for component in range(3):
                normalized[group_index, :, component] = zscore(
                    components[group_index, :, component]
                )
        weights = np.asarray([1.0, 0.5, 0.5])
        pre = (normalized * weights).sum(axis=-1)
        advantage = pre if effective == "DEC_PRIME" else zscore(pre.reshape(-1)).reshape(pre.shape)
        info.update(
            component_weights=weights.tolist(),
            component_std=components.std(axis=1).tolist(),
            normalised_components=normalized.tolist(),
            pre_batch_advantage=pre.tolist(),
            pre_batch_mean=float(pre.mean()),
            pre_batch_std=float(pre.std()),
            batch_normalized=effective == "DEC",
        )
        if effective == "DEC_PRIME":
            assert np.max(np.abs(advantage)) <= DEC_PRIME_ABSOLUTE_BOUND + GATE_TOLERANCE
            info["absolute_advantage_bound"] = float(DEC_PRIME_ABSOLUTE_BOUND)
    else:
        rows = [gate_one(j[index], e[index], p[index]) for index in range(16)]
        advantage = np.stack([row[0] for row in rows])
        info["branches"] = [row[1] for row in rows]
        baseline = np.stack([zscore(row) for row in j])
        info["refinement_changed_groups"] = [
            index
            for index, (_, branch) in enumerate(rows)
            if branch == "target_J_refined"
            and not np.array_equal(advantage[index], baseline[index])
        ]
        info["kappa"] = 0.25
        info["constant_h_tolerance"] = GATE_TOLERANCE
    info["final_advantage"] = advantage.tolist()
    info["advantage_sum_squares"] = float(np.square(advantage).sum())
    info["group_advantage_sum_squares"] = np.square(advantage).sum(axis=1).tolist()
    info["contrast_groups"] = int(np.count_nonzero(np.ptp(advantage, axis=1) > 0))
    info["zero_signal_step"] = bool(not np.any(advantage))
    return advantage, info
