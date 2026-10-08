"""Frozen SER-J23 paired-root inference, conditional on the six fixed paths.

Raw scored records enter ``analyze_confirmation`` only after the caller's release
gate. This module performs no I/O, model invocation, draw/parent/path resampling,
or data-dependent filtering. All intervals use one shared stratified PCG64 plan.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .semantics import EVENT_FIELDS, aggregate_events

PHASE_ID = "SER_J23_20261008"
PARENTS = ("S96", "REP96")
ARMS = ("A2_LOCAL_C1_J2", "B2_FORWARD_C4_J2", "A3_LOCAL_C1_J3", "B3_FORWARD_C4_J3")
BOOTSTRAP_SEED = 2026100809
BOOTSTRAP_REPLICATES = 10000
FOCUS_CELLS = ((3, 3), (3, 1), (3, 2), (2, 2))
METRICS = (
    *EVENT_FIELDS,
    "F_legal",
    *(f"q{k}" for k in range(4)),
    *(f"r{k}" for k in range(4)),
    *(f"out_of_domain{k}" for k in range(4)),
)
TASK_FIELDS = ("task_id", "root_id", "family", "center", "corrupted_index", "root_cohort")
MODEL_FIELDS = (
    "checkpoint_id",
    "parent",
    "block",
    "logical_arm_id",
    "step",
    "panel",
    "phase_id",
    "source_experiment_id",
    "source_checkpoint_id",
    "checkpoint_sha256",
)


class IncompleteDataError(ValueError):
    """The registered matrix is incomplete; observed subsets cannot replace it."""


def safe_ratio(numerator, denominator):
    if not np.isfinite([numerator, denominator]).all() or not 0 <= numerator <= denominator:
        raise ValueError("Invalid conditional numerator/denominator")
    return float(numerator / denominator) if denominator else None


def _arm(row):
    arm = row.get("logical_arm_id", row.get("arm"))
    if "arm" in row and "logical_arm_id" in row and row["arm"] != arm:
        raise ValueError("Logical arm aliases disagree")
    return arm


def summarize_tasks(rows, *, expected_draws=8):
    """Average each checkpoint/task's exact draw allocation, including every I answer.

    Rows require a globally unique ``sample_id`` and the native score_output
    fields, plus checkpoint_id/parent/block/logical_arm_id/step/panel/draw_index.
    ``arm`` may substitute for logical_arm_id only when it is the same full name.
    No input row is changed. Use expected_draws=4 for frozen development panels.
    """
    if type(expected_draws) is not int or expected_draws < 1:
        raise ValueError("A positive exact draw allocation is required")
    groups, seen_samples, seen_draws = defaultdict(list), set(), set()
    for source in rows:
        row = {**source, "logical_arm_id": _arm(source)}
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Nonempty globally unique sample_id required")
        if sample_id in seen_samples:
            raise ValueError("Duplicate sample_id")
        seen_samples.add(sample_id)
        draw = row.get("draw_index")
        if type(draw) is not int or not 0 <= draw < expected_draws:
            raise IncompleteDataError("Draw index outside the frozen allocation")
        key = (row["checkpoint_id"], row["panel"], row["task_id"])
        draw_key = (*key, draw)
        if draw_key in seen_draws:
            raise ValueError("Duplicate checkpoint/task/draw identity")
        seen_draws.add(draw_key)
        groups[key].append(row)
    output = []
    for group in groups.values():
        first = group[0]
        indices = sorted(row["draw_index"] for row in group)
        if indices != list(range(expected_draws)):
            raise IncompleteDataError("Every task requires its complete frozen draw indices")
        fields = (*MODEL_FIELDS, *TASK_FIELDS)
        if any(any(row.get(field) != first.get(field) for field in fields) for row in group):
            raise ValueError("Conflicting checkpoint/task metadata within one answer stream")
        item = {field: first.get(field) for field in fields}
        item.update(aggregate_events(group), draw_indices=indices)
        for k in range(4):
            item[f"q{k}"] = item["q"][k]
            item[f"r{k}"] = item["r"][k]
            item[f"out_of_domain{k}"] = item["out_of_domain_by_position"][k]
        output.append(item)
    return output


def _model_keys():
    return tuple((p, b, a) for p in PARENTS for b in range(3) for a in ARMS) + tuple(
        (p, None, "PARENT") for p in PARENTS
    )


def _validate_task_probabilities(row):
    values = np.asarray([row[metric] for metric in METRICS], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Task probabilities must be complete, finite, and in [0,1]")
    if not np.isclose(sum(row[name] for name in "XSWI"), 1, atol=1e-12):
        raise ValueError("Task event probabilities must partition all answers")
    if not np.isclose(row["X"], row["F_val"] - row["R"], atol=1e-12):
        raise ValueError("Task X/F_val/R decomposition failed")
    j = row["corrupted_index"]
    for k in range(4):
        if not np.isclose(row[f"q{k}"], row[f"r{k}"] + (row["X"] if k == j else 0), atol=1e-12):
            raise ValueError("Task q/r/X decomposition failed")
        if row[f"r{k}"] > 1 - row["X"] + 1e-12:
            raise ValueError("Non-X edit probability exceeds its denominator")


def _complete_confirmation(task_rows, expected_tasks=None):
    expected = set(_model_keys())
    groups = defaultdict(dict)
    checkpoint_ids = {}
    endpoint_bindings = {}
    for row in task_rows:
        if row["panel"] != "E_CONFIRM2" or row["phase_id"] != PHASE_ID:
            raise ValueError("Only this phase's E_CONFIRM2 answers may enter confirmation")
        arm, block = row["logical_arm_id"], row["block"]
        if (block is not None and type(block) is not int) or type(row["step"]) is not int:
            raise ValueError("Step and block require exact integer identities")
        key = (row["parent"], block, arm)
        if key not in expected or row["step"] != (0 if arm == "PARENT" else 256):
            raise ValueError("Unregistered confirmation model or nonterminal student")
        _validate_task_probabilities(row)
        if row["task_id"] in groups[key]:
            raise ValueError("Duplicate logical endpoint/task, possibly multiple checkpoints")
        if key in checkpoint_ids and checkpoint_ids[key] != row["checkpoint_id"]:
            raise ValueError("One logical endpoint cannot mix checkpoint identities")
        checkpoint_ids[key] = row["checkpoint_id"]
        binding = tuple(row.get(field) for field in MODEL_FIELDS)
        if key in endpoint_bindings and endpoint_bindings[key] != binding:
            raise ValueError(
                "One logical endpoint cannot mix source identities or checkpoint hashes"
            )
        endpoint_bindings[key] = binding
        groups[key][row["task_id"]] = row
    if set(groups) != expected:
        raise IncompleteDataError("Confirmation requires exactly 24 students and two parents")
    if len(set(checkpoint_ids.values())) != 26:
        raise ValueError("Checkpoint identity must not be duplicated across formal endpoints")
    reference = groups[_model_keys()[0]]
    if len(reference) != 992 or any(set(group) != set(reference) for group in groups.values()):
        raise IncompleteDataError("All 26 endpoints must contain the same 992-task panel")
    for group in groups.values():
        for task_id, row in group.items():
            if row["draw_indices"] != list(range(8)) or row["n_draws"] != 8:
                raise IncompleteDataError("Every confirmation task requires exactly eight draws")
            if any(row[field] != reference[task_id][field] for field in TASK_FIELDS):
                raise ValueError("Models disagree on frozen task/root/cell metadata")
    if expected_tasks is not None:
        expected_tasks = list(expected_tasks)
        registered = {row["task_id"]: row for row in expected_tasks}
        if len(registered) != len(expected_tasks) or set(registered) != set(reference):
            raise ValueError("Analysis tasks differ from the frozen task registry")
        if any(
            any(row[field] != registered[task_id][field] for field in TASK_FIELDS)
            for task_id, row in reference.items()
        ):
            raise ValueError("Analysis metadata differs from the frozen task registry")
    cohorts, root_cells, cell_tasks = defaultdict(set), defaultdict(set), defaultdict(dict)
    for task_id, row in reference.items():
        root = row["root_id"]
        cell = (row["family"], row["center"], row["corrupted_index"])
        if not isinstance(root, str) or not root:
            raise ValueError("Nonempty root identity required")
        if type(row["corrupted_index"]) is not int or row["corrupted_index"] not in range(4):
            raise ValueError("Malformed corrupted position")
        if row["family"] == "cross_series":
            if type(row["center"]) is not int or row["center"] not in range(4):
                raise ValueError("Malformed cross center")
        elif row["family"] not in ("trend", "duplicate_encoding") or row["center"] is not None:
            raise ValueError("Malformed non-cross task")
        if root in cell_tasks[cell]:
            raise ValueError("Duplicate numeric root/cell task")
        cohorts[row["root_cohort"]].add(root)
        root_cells[root].add(cell)
        cell_tasks[cell][root] = task_id
    if {name: len(roots) for name, roots in cohorts.items()} != {
        "CORE32": 32,
        "EXTRA96": 96,
        "trend": 64,
        "duplicate_encoding": 32,
    }:
        raise IncompleteDataError("Frozen root cohorts must be CORE32/EXTRA96/trend64/duplicate32")
    flat_roots = [root for roots in cohorts.values() for root in roots]
    if len(flat_roots) != len(set(flat_roots)):
        raise ValueError("A numeric root cannot belong to multiple cohorts")
    all_cross = {("cross_series", c, j) for c in range(4) for j in range(4)}
    focus = {("cross_series", c, j) for c, j in FOCUS_CELLS}
    for cohort, roots in cohorts.items():
        for root in roots:
            cells = root_cells[root]
            if cohort in ("CORE32", "EXTRA96"):
                if cells != (all_cross if cohort == "CORE32" else focus):
                    raise IncompleteDataError("Incomplete CORE/EXTRA within-root cell pairing")
            elif len(cells) != 1 or next(iter(cells))[0] != cohort:
                raise ValueError("Each trend/duplicate root belongs to exactly one position")
    for family, count in (("trend", 16), ("duplicate_encoding", 8)):
        if any(len(cell_tasks[family, None, j]) != count for j in range(4)):
            raise IncompleteDataError("Every trend/duplicate position needs its frozen root count")
    strata = {name: tuple(sorted(cohorts[name])) for name in ("CORE32", "EXTRA96")}
    for family in ("trend", "duplicate_encoding"):
        for j in range(4):
            strata[f"{family}:j{j}"] = tuple(sorted(cell_tasks[family, None, j]))
    return dict(groups), dict(cell_tasks), strata


@dataclass(frozen=True)
class PairedRootBootstrap:
    """One immutable shared root-count plan; all outputs use its identical indices.

    Roots are ordered lexically within the frozen strata. For each replicate,
    the RNG draws CORE32, EXTRA96, trend j0..j3, then duplicate j0..j3. Counts
    suffice because K-draw task means are computed before any resampling.
    """

    roots: tuple[str, ...]
    counts: np.ndarray
    strata: tuple[tuple[str, tuple[str, ...]], ...]
    seed: int
    sha256: str

    @classmethod
    def build(cls, strata, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
        if type(replicates) is not int or replicates < 1 or type(seed) is not int:
            raise ValueError("Exact integer replicate count and seed required")
        ordered = tuple((name, tuple(roots)) for name, roots in strata.items())
        if not ordered or any(not roots or len(set(roots)) != len(roots) for _, roots in ordered):
            raise ValueError("Nonempty unique root strata required")
        roots = tuple(root for _, group in ordered for root in group)
        if len(roots) != len(set(roots)):
            raise ValueError("Bootstrap strata must partition numeric roots exactly once")
        rng = np.random.Generator(np.random.PCG64(seed))
        counts = np.zeros((replicates, len(roots)), dtype=np.uint16)
        for r in range(replicates):
            offset = 0
            for _, group in ordered:
                n = len(group)
                if n > np.iinfo(np.uint16).max:
                    raise ValueError("Root stratum exceeds count storage capacity")
                selected = rng.integers(0, n, size=n)
                counts[r, offset : offset + n] = np.bincount(selected, minlength=n)
                offset += n
        digest = hashlib.sha256(repr(ordered).encode() + counts.astype("<u2").tobytes()).hexdigest()
        counts.flags.writeable = False
        return cls(roots, counts, ordered, seed, digest)

    @property
    def replicates(self):
        return self.counts.shape[0]

    def weights(self, roots):
        """Use whole registered strata only, preserving CORE32/EXTRA96 = 1/4,3/4."""
        roots = tuple(roots)
        selected = set(roots)
        if not selected or len(roots) != len(selected) or not selected <= set(self.roots):
            raise ValueError("Unknown, empty or duplicate bootstrap root selection")
        if any(selected & set(group) and not set(group) <= selected for _, group in self.strata):
            raise ValueError("A cell must contain complete frozen bootstrap strata")
        index = {root: i for i, root in enumerate(self.roots)}
        counts = self.counts[:, [index[root] for root in roots]].astype(float)
        if not np.all(counts.sum(axis=1) == len(roots)):
            raise AssertionError("Stratified root denominators changed")
        return counts / len(roots)

    def mean(self, values, roots):
        """Bootstrap means for finite arrays whose final axis is the given roots."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 0 or values.shape[-1] != len(roots) or not np.isfinite(values).all():
            raise ValueError("Complete finite root arrays required; no dropna")
        return np.einsum("rn,...n->r...", self.weights(roots), values, optimize=True)


def _ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap statistics cannot be dropped")
    return np.quantile(values, [alpha / 2, 1 - alpha / 2], method="linear").tolist()


def _effect(blocks, bootstrap_blocks):
    """Six fixed parent/order values; never bootstrap their indices."""
    blocks, bootstrap_blocks = np.asarray(blocks), np.asarray(bootstrap_blocks)
    if blocks.shape != (2, 3) or bootstrap_blocks.shape[1:] != (2, 3):
        raise ValueError("Effects require two fixed parents by three actual orders")
    boot = bootstrap_blocks.mean(axis=(1, 2))
    if not np.isfinite(blocks).all() or not np.isfinite(bootstrap_blocks).all():
        raise ValueError("Missing/nonfinite fixed-path effect")
    interval = _ci(boot)
    return {
        "estimate": float(blocks.mean()),
        "ci95_root_conditional": interval,
        "per_parent_order": blocks.tolist(),
        "per_parent": blocks.mean(axis=1).tolist(),
        "per_parent_ci95_root_conditional": [
            _ci(bootstrap_blocks[:, i, :].mean(axis=1)) for i in range(2)
        ],
        "per_parent_order_range": [[float(row.min()), float(row.max())] for row in blocks],
        "all_six_range": [float(blocks.min()), float(blocks.max())],
        "direction95": "negative"
        if interval[1] < 0
        else ("positive" if interval[0] > 0 else "uncertain"),
        "interval_scope": "numeric_roots_conditional_on_fixed_parents_and_actual_training_paths",
    }


def _model_metadata(key):
    return dict(zip(("parent", "block", "logical_arm_id"), key, strict=True))


def _probabilities(vector):
    result = dict(zip(METRICS, map(float, vector), strict=True))
    denominator = max(0.0, 1.0 - result["X"])
    result["q"] = [result[f"q{k}"] for k in range(4)]
    result["r"] = [result[f"r{k}"] for k in range(4)]
    result["c"] = [float(result[f"r{k}"] / denominator) if denominator else None for k in range(4)]
    result["denominator_not_X_probability"] = denominator
    result["c_undefined_reason"] = "no_non_X_answers" if denominator == 0 else None
    result["rate_denominator"] = "all_answers_with_prespecified_task_root_cell_path_weights"
    return result


def cross_macro(cell_probabilities):
    if set(cell_probabilities) != {(c, j) for c in range(4) for j in range(4)}:
        raise ValueError("All sixteen cross cells, including j1, are required")
    values = np.asarray(list(cell_probabilities.values()), dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Complete cell probabilities required")
    return float(values.mean())


def trend_macro(cell_probabilities):
    if set(cell_probabilities) != set(range(4)):
        raise ValueError("All four corrupted positions are required")
    values = np.asarray(list(cell_probabilities.values()), dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Complete position probabilities required")
    return float(values.mean())


CONTRASTS = ((ARMS[1], ARMS[0]), (ARMS[3], ARMS[2]), *((arm, "PARENT") for arm in ARMS))


def _contrast_arrays(point, boot, model_index, left, right, metric):
    m = METRICS.index(metric)
    blocks = np.empty((2, 3))
    samples = np.empty((len(boot), 2, 3))
    for i, parent in enumerate(PARENTS):
        for b in range(3):
            lhs = model_index[parent, b, left]
            rhs = model_index[parent, None if right == "PARENT" else b, right]
            blocks[i, b] = point[lhs, m] - point[rhs, m]
            samples[:, i, b] = boot[:, lhs, m] - boot[:, rhs, m]
    return blocks, samples


def _analyze_complete(groups, cell_tasks, strata, *, replicates, seed):
    keys = _model_keys()
    model_index = {key: i for i, key in enumerate(keys)}
    plan = PairedRootBootstrap.build(strata, replicates=replicates, seed=seed)
    macro_names = ("cross_macro", "trend_macro", "duplicate")
    macro_point = {name: np.zeros((26, len(METRICS))) for name in macro_names}
    macro_boot = {name: np.zeros((replicates, 26, len(METRICS))) for name in macro_names}
    cells, contrasts, focus = [], [], {}
    # Compute one cell at a time; roots and all K draws travel together for all models.
    for cell in sorted(cell_tasks, key=lambda c: (c[0], -1 if c[1] is None else c[1], c[2])):
        family, center, j = cell
        by_root = cell_tasks[cell]
        roots = tuple(root for root in plan.roots if root in by_root)
        data = np.asarray(
            [
                [[groups[key][by_root[root]][metric] for root in roots] for metric in METRICS]
                for key in keys
            ]
        )
        if data.shape != (26, len(METRICS), len(roots)) or not np.isfinite(data).all():
            raise ValueError("Incomplete per-root probability cube")
        point, boot = data.mean(axis=2), plan.mean(data, roots)
        name = {
            "cross_series": "cross_macro",
            "trend": "trend_macro",
            "duplicate_encoding": "duplicate",
        }[family]
        weight = 1 / (16 if family == "cross_series" else 4)
        macro_point[name] += weight * point
        macro_boot[name] += weight * boot
        cell_meta = {"family": family, "center": center, "corrupted_index": j}
        for i, key in enumerate(keys):
            cells.append(
                {
                    **cell_meta,
                    **_model_metadata(key),
                    "n_roots": len(roots),
                    "n_answers": len(roots) * 8,
                    **_probabilities(point[i]),
                }
            )
        for arm in (*ARMS, "PARENT"):
            indices = [i for i, key in enumerate(keys) if key[2] == arm]
            cells.append(
                {
                    **cell_meta,
                    "parent": "EQUAL_FIXED_PARENTS",
                    "block": None,
                    "logical_arm_id": arm,
                    "n_roots": len(roots),
                    "n_answers": len(roots) * 8 * len(indices),
                    **_probabilities(point[indices].mean(axis=0)),
                }
            )
        for left, right in CONTRASTS:
            for metric in ("X", "F_val", "R", "q0", "q1", "q2", "q3", "r0", "r1", "r2", "r3"):
                blocks, samples = _contrast_arrays(point, boot, model_index, left, right, metric)
                detail = _effect(blocks, samples)
                contrasts.append(
                    {**cell_meta, "left": left, "right": right, "metric": metric, **detail}
                )
                if family == "cross_series" and (center, j) in FOCUS_CELLS:
                    focus[center, j, left, right, metric] = (detail, blocks, samples)
    macro_point["cross_trend_50_50"] = 0.5 * (
        macro_point["cross_macro"] + macro_point["trend_macro"]
    )
    macro_boot["cross_trend_50_50"] = 0.5 * (macro_boot["cross_macro"] + macro_boot["trend_macro"])
    macro_metrics, macro_contrasts = [], []
    for name, point in macro_point.items():
        boot = macro_boot[name]
        for i, key in enumerate(keys):
            ci = np.quantile(boot[:, i, :], [0.025, 0.975], axis=0, method="linear")
            macro_metrics.append(
                {
                    **_model_metadata(key),
                    "macro": name,
                    **_probabilities(point[i]),
                    "ci95_root_conditional": {
                        metric: ci[:, m].tolist() for m, metric in enumerate(METRICS)
                    },
                }
            )
        for left, right in CONTRASTS:
            for metric in METRICS:
                blocks, samples = _contrast_arrays(point, boot, model_index, left, right, metric)
                macro_contrasts.append(
                    {
                        "macro": name,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        **_effect(blocks, samples),
                    }
                )

    def get(center, j, left=ARMS[3], right=ARMS[2], metric="X"):
        return focus[center, j, left, right, metric]

    h3 = dict(get(3, 3)[0])
    h2 = dict(get(3, 3, ARMS[1], ARMS[0])[0])
    # Q_3,3 - Q_3,2 - Q_2,3 + Q_2,2; coordinate labels here are zero-based.
    psi_terms = [
        get(3, 3, ARMS[t + 1], ARMS[t], f"q{k}") for t, k in ((2, 2), (2, 1), (0, 2), (0, 1))
    ]
    psi_blocks = sum(sign * item[1] for sign, item in zip((1, -1, -1, 1), psi_terms, strict=True))
    psi_samples = sum(sign * item[2] for sign, item in zip((1, -1, -1, 1), psi_terms, strict=True))
    secondary = {}
    for endpoint, detail, samples in (
        ("Psi_E", _effect(psi_blocks, psi_samples), psi_samples),
        ("G_3_2", get(3, 1)[0], get(3, 1)[2]),
        ("G_3_3", get(3, 2)[0], get(3, 2)[2]),
    ):
        ci = _ci(samples.mean(axis=(1, 2)), 0.05 / 3)
        secondary[endpoint] = {
            **detail,
            "ci98_333333_bonferroni_three": ci,
            "expected_direction": "positive",
            "direction_supported": ci[0] > 0,
        }
    noninferiority = {}
    for t, left, right in ((2, ARMS[1], ARMS[0]), (3, ARMS[3], ARMS[2])):
        detail, _, samples = get(2, 2, left, right)
        lower = float(np.quantile(samples.mean(axis=(1, 2)), 0.025, method="linear"))
        noninferiority[f"N{t}_c3j3"] = {
            **detail,
            "one_sided_97_5_lower": lower,
            "margin_probability": 0.03,
            "noninferiority_lower_threshold": -0.03,
            "noninferiority_established": lower > -0.03,
            "interpretation": "predefined_noninferiority_established"
            if lower > -0.03
            else "noninferiority_not_established",
            "margin_status": "new_operational_choice_not_equivalence_or_universal_safety",
        }
    gdiff = _effect(get(3, 2)[1] - get(3, 1)[1], get(3, 2)[2] - get(3, 1)[2])
    decomposition = {
        label: {metric: get(3, 3, left, right, metric)[0] for metric in ("X", "F_val", "R")}
        for label, left, right in (("H2", ARMS[1], ARMS[0]), ("H3", ARMS[3], ARMS[2]))
    }
    for parts in decomposition.values():
        if not np.isclose(
            parts["X"]["estimate"], parts["F_val"]["estimate"] - parts["R"]["estimate"], atol=1e-12
        ):
            raise AssertionError("Paired X/F_val/R effect decomposition failed")
    h3.update(
        name="H3",
        expected_direction="negative",
        scientific_effect_size_gate=None,
        harm_supported=h3["ci95_root_conditional"][1] < 0,
    )
    return {
        "primary": h3,
        "H2": h2,
        "parent_contrasts": {f"{arm}-PARENT": get(3, 3, arm, "PARENT")[0] for arm in ARMS},
        "secondary_family": {
            "family_alpha": 0.05,
            "multiplicity": "Bonferroni_m3",
            "endpoints": secondary,
        },
        "noninferiority_family": {
            "family_alpha": 0.05,
            "multiplicity": "Bonferroni_m2",
            "endpoints": noninferiority,
        },
        "descriptive": {
            "G_2_2": get(3, 1, ARMS[1], ARMS[0])[0],
            "G_2_3": get(3, 2, ARMS[1], ARMS[0])[0],
            "G_3_3_minus_G_3_2": gdiff,
        },
        "value_extra_decomposition": decomposition,
        "all_cell_metrics": cells,
        "all_cell_contrasts": contrasts,
        "macro": {
            "metrics": macro_metrics,
            "contrasts": macro_contrasts,
            "weights": "cross16=1/16; trend4=1/4; cross/trend=1/2; duplicate4=1/4_separate",
        },
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "rng": "numpy.PCG64",
            "quantile_method": "linear",
            "shared_root_plan_sha256": plan.sha256,
            "strata": {name: list(roots) for name, roots in plan.strata},
            "resample_draws": False,
            "resample_training_blocks": False,
            "all_models_metrics_and_endpoints_share_indices": True,
        },
        "scope": {
            "parent_order": list(PARENTS),
            "block_order": [0, 1, 2],
            "arm_order": list(ARMS),
            "parent_generation_streams": 2,
            "joint_overall_alpha_across_families_claimed": False,
            "degenerate_intervals_do_not_prove_population_probability_zero": True,
            "estimand": "conditional_on_two_fixed_parents_three_actual_orders_fixed_training_roots",
        },
    }


def analyze_confirmation(
    rows,
    *,
    expected_tasks=None,
    expected_draws=8,
    bootstrap_replicates=BOOTSTRAP_REPLICATES,
    seed=BOOTSTRAP_SEED,
):
    """Analyze the complete released 206,336-answer matrix using frozen constants.

    ``expected_tasks`` optionally binds all TASK_FIELDS to the auditor's immutable
    task registry (public and audit metadata joined by task_id). The release
    caller must verify the registry/file hashes and all training terminal states.
    This pure function cannot itself authorize access to sealed confirmation.
    No alternate resampling count/seed is accepted by this formal entry point.
    """
    if expected_draws != 8 or type(expected_draws) is not int:
        raise ValueError("Frozen confirmation requires eight draws")
    if (
        type(bootstrap_replicates) is not int
        or bootstrap_replicates != BOOTSTRAP_REPLICATES
        or type(seed) is not int
        or seed != BOOTSTRAP_SEED
    ):
        raise ValueError("Frozen confirmation requires PCG64 seed 2026100809 and 10000 replicates")
    tasks = summarize_tasks(rows, expected_draws=8)
    groups, cells, strata = _complete_confirmation(tasks, expected_tasks)
    result = _analyze_complete(groups, cells, strata, replicates=bootstrap_replicates, seed=seed)
    result.update(
        status="COMPLETE_CONDITIONAL_INFERENCE",
        task_metrics=tasks,
        frozen_task_registry_bound=expected_tasks is not None,
    )
    return result
