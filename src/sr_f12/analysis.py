"""SR-F1.2 pre-science semantic state and frozen secondary hypotheses."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .protocol import BASELINE, PRIMARY_CONTRASTS, SEEDS, immutable_json, object_hash

FAMILIES = ("CROSS", "THRESHOLD", "TOPK", "INTERVAL")
METRICS = ("A", "E", "p_read", "P", "J")


def _score(row):
    score = row.get("independent_score", row.get("score"))
    if not isinstance(score, dict):
        raise ValueError("Each record needs an independently scored event dictionary")
    for key in METRICS:
        value = score[key]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid semantic event")
        if key != "p_read" and value not in (0, 1):
            raise ValueError("Binary semantic event required")
    if score["J"] != score["A"] * score["P"] or score["P"] != int(
        score["E"] == 1 and score["p_read"] == 1
    ):
        raise ValueError("Official P/J semantic identity violated")
    return score


def _family_mean(rows, metric):
    """Draws -> questions -> roots; family callers never conflate root sizes."""
    questions = defaultdict(list)
    for row in rows:
        questions[(row["root_id"], row["qid"])].append(_score(row)[metric])
    roots = defaultdict(list)
    for (root, _), values in questions.items():
        roots[root].append(float(np.mean(values)))
    return float(np.mean([np.mean(v) for v in roots.values()]))


def semantic_state_table(rows, *, require_complete=True):
    """Consume only the newly generated common-start MONITOR and five views.

    D=P(A=0|P=1) uses raw draws (equal K). Gold-values uplift is paired against
    the same MONITOR v0 questions, rather than the full variant panel.
    """
    rows = list(rows)
    groups = defaultdict(list)
    seen = set()
    views = {"keys_hint", "gold_values", "text_values", "neutral_hint", "blank_image"}
    for row in rows:
        if row.get("model_id") != BASELINE or row.get("step") != 0:
            raise PermissionError("Only the new SR-F1.2 zero-LoRA baseline is eligible")
        if row.get("protocol") != "evidence_answer" or row["pool"] not in (
            "MONITOR",
            "MONITOR_DIAGNOSTIC_ONLY",
        ):
            raise PermissionError("Semantic predictions cannot consume TEST or training results")
        if (
            row["family"] not in FAMILIES
            or type(row["draw"]) is not int
            or row["draw"] not in range(8)
        ):
            raise ValueError("Unregistered semantic family/draw")
        view = row.get("view")
        if (row["pool"] == "MONITOR" and view is not None) or (
            row["pool"] != "MONITOR" and view not in views
        ):
            raise ValueError("Unregistered semantic panel/view")
        slot = (row["pool"], view, row["qid"], row["draw"])
        if slot in seen:
            raise ValueError("Duplicate baseline semantic slot")
        seen.add(slot)
        _score(row)
        groups[(row["family"], row["pool"], view, row["qid"])].append(row)
    if any(sorted(r["draw"] for r in g) != list(range(8)) for g in groups.values()):
        raise ValueError("Incomplete eight-draw semantic group")
    monitor = [r for r in rows if r["pool"] == "MONITOR"]
    diagnostic = [r for r in rows if r["pool"] == "MONITOR_DIAGNOSTIC_ONLY"]
    if require_complete and (len(monitor) != 4096 or len(diagnostic) != 2560):
        raise ValueError("Require exactly 4096 MONITOR plus 2560 diagnostic records")
    result = []
    for family in sorted(FAMILIES):
        original = [r for r in monitor if r["family"] == family]
        if not original:
            raise ValueError("Every registered family needs baseline support")
        group_list = [g for (f, p, _, _), g in groups.items() if f == family and p == "MONITOR"]
        signal_a, signal_j, signal_g = [], [], []
        for group in group_list:
            scores = [_score(r) for r in group]
            a, j, e, p = (np.array([s[key] for s in scores]) for key in ("A", "J", "E", "p_read"))
            signal_a.append(bool(np.ptp(a)))
            signal_j.append(bool(np.ptp(j)))
            signal_g.append(
                bool(np.all(j == 0) and (np.ptp(e) > 0 or (np.all(e == 1) and np.ptp(p) > 0)))
            )
        support = [r for r in original if _score(r)["P"] == 1]
        failure_count = sum(_score(r)["A"] == 0 for r in support)
        gold = [r for r in diagnostic if r["family"] == family and r["view"] == "gold_values"]
        v0 = [r for r in original if r["variant"] == "v0"]
        qids = {r["qid"] for r in v0}
        if not gold or {r["qid"] for r in gold} != qids:
            raise ValueError("gold_values requires exactly matching original v0 questions")
        for view in views:
            sample = [r for r in diagnostic if r["family"] == family and r["view"] == view]
            if {r["qid"] for r in sample} != qids:
                raise ValueError("All five diagnostic views require matched v0 coverage")
        if require_complete and (len(original) != 1024 or len(v0) != 128):
            raise ValueError("Incomplete family-balanced baseline panel")
        means = {key: _family_mean(original, key) for key in METRICS}
        result.append(
            dict(
                family=family,
                means=means,
                question_count=len(group_list),
                root_count=len({r["root_id"] for r in original}),
                s_A=float(np.mean(signal_a)),
                s_J=float(np.mean(signal_j)),
                s_G=float(np.mean(signal_g)),
                D=dict(
                    numerator=failure_count,
                    denominator=len(support),
                    probability=failure_count / len(support) if support else None,
                    status="DEFINED" if support else "UNDEFINED_NO_P_SUPPORT",
                ),
                gold_values_A=_family_mean(gold, "A"),
                original_v0_A=_family_mean(v0, "A"),
                gold_values_uplift=_family_mean(gold, "A") - _family_mean(v0, "A"),
            )
        )
    return dict(
        status="COMPLETE_BASELINE_SEMANTIC_STATE" if require_complete else "TEST_FIXTURE_STATE",
        baseline=BASELINE,
        monitor_samples=len(monitor),
        diagnostic_samples=len(diagnostic),
        families=result,
        input_semantic_sha256=object_hash(
            sorted(
                [
                    dict(
                        pool=r["pool"],
                        view=r.get("view"),
                        qid=r["qid"],
                        root_id=r["root_id"],
                        family=r["family"],
                        draw=r["draw"],
                        score=_score(r),
                    )
                    for r in rows
                ],
                key=lambda r: (r["pool"], r["view"] or "", r["qid"], r["draw"]),
            )
        ),
        weighting=(
            "draws within question, questions within root, "
            "roots within family; signals per question"
        ),
        D_denominator="MONITOR draws with P=1; unknown if zero",
        gold_uplift_comparator="same v0 question panel in original MONITOR",
    )


def _tertiles(families, key):
    ordered = sorted(families, key=lambda row: (key(row), row["family"]))
    # Four fixed families: array_split yields 2 low, 1 middle, 1 high.
    return {
        row["family"]: label
        for label, group in zip(
            ("low", "middle", "high"),
            np.array_split(np.array(ordered, dtype=object), 3),
            strict=True,
        )
        for row in group
    }


def preregister_predictions(state, train_family_counts):
    if state.get("status") != "COMPLETE_BASELINE_SEMANTIC_STATE":
        raise PermissionError("Predictions require complete new baseline state")
    families = state["families"]
    if len(families) != 4 or {r["family"] for r in families} != set(FAMILIES):
        raise ValueError("Require all four families")
    if set(train_family_counts) != set(FAMILIES) or any(
        type(n) is not int or n <= 0 for n in train_family_counts.values()
    ):
        raise ValueError("Positive pre-registered TRAIN question family counts required")
    p1 = _tertiles(families, lambda row: row["means"]["A"] - row["means"]["J"])
    p2 = _tertiles(families, lambda row: row["s_G"])
    p3 = {
        r["family"]: (
            "UNKNOWN_NO_P_SUPPORT"
            if r["D"]["probability"] is None
            else "computation_bottleneck"
            if r["D"]["probability"] >= 0.5 and r["gold_values_uplift"] < 0.1
            else "reading_bottleneck"
        )
        for r in families
    }
    weighted_sg = sum(r["s_G"] * train_family_counts[r["family"]] for r in families) / sum(
        train_family_counts.values()
    )
    return dict(
        status="PREREGISTERED_BEFORE_SCIENCE",
        baseline_state_sha256=object_hash(state),
        P1=dict(
            strata=p1, ordering="A_minus_J", contrast="J-A", prediction="high_gain_greater_than_low"
        ),
        P2=dict(
            strata=p2, ordering="s_G", contrast="GATE-J", prediction="high_gain_greater_than_low"
        ),
        P3=dict(
            strata=p3,
            contrast="GATE-J",
            prediction="computation_gain_less_than_reading",
            missing_support="UNKNOWN; excluded from class contrast, retained in output",
        ),
        mechanism=dict(
            train_family_counts=train_family_counts,
            expected_pure_semantic_branch_rate=weighted_sg,
            observed_steps=list(range(1, 9)),
            observed_branches=["evidence", "readings"],
            absolute_tolerance=0.10,
            comparison="descriptive secondary check, never a training stop gate",
        ),
        tertile_rule=(
            "ascending(value, family_name); "
            "four families split low=2, middle=1, high=1; ties by name"
        ),
        endpoint=(
            "TEST_ID J at96 minus common-start TEST_ID J; three seeds averaged; root bootstrap"
        ),
        role="secondary preregistered hypotheses; no main endpoint selection or reward adaptation",
    )


def freeze_predictions(root, state, predictions):
    from pathlib import Path

    root = Path(root)
    if predictions.get("baseline_state_sha256") != object_hash(state):
        raise PermissionError("Prediction/state identity mismatch")
    for folder in ("training", "runs"):
        if (root / folder).exists() and any((root / folder).rglob("*")):
            raise PermissionError("Cannot preregister after scientific training begins")
    immutable_json(root / "SEMANTIC_STATE.json", state)
    immutable_json(root / "SEMANTIC_PREDICTIONS.json", predictions)
    return dict(state_sha256=object_hash(state), predictions_sha256=object_hash(predictions))


def paired_root_ci(x, y, families):
    """Keep the original paired family-stratified root bootstrap, four comparisons."""
    x, y, fam = np.asarray(x, float), np.asarray(y, float), np.asarray(families)
    if x.shape != y.shape or x.ndim != 2 or x.shape != (3, len(fam)) or set(fam) != set(FAMILIES):
        raise ValueError("Require three seeds by matched roots in all four families")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Technical missingness cannot be a zero score")
    delta = x - y
    rng = np.random.default_rng(812901)
    draws = np.zeros(5000)
    seed_effects = np.zeros(3)
    for family in sorted(FAMILIES):
        indices = np.flatnonzero(fam == family)
        seed_effects += delta[:, indices].mean(axis=1) / 4
        samples = rng.choice(indices, size=(5000, len(indices)), replace=True)
        draws += delta.mean(axis=0)[samples].mean(axis=1) / 4
    return dict(
        effect_pp=float(seed_effects.mean() * 100),
        seed_effects_pp=(seed_effects * 100).tolist(),
        CI95_pp=(np.quantile(draws, [0.025, 0.975]) * 100).tolist(),
        CI98_75_pp=(np.quantile(draws, [0.00625, 0.99375]) * 100).tolist(),
        bootstrap_replicates=5000,
        bootstrap_seed=812901,
        multiplicity="Bonferroni family of 4; each two-sided CI=98.75%",
        primary_contrasts=[list(pair) for pair in PRIMARY_CONTRASTS],
        scope="root measurement uncertainty conditional on three realised training seeds",
    )


def _test_root_values(rows, *, released):
    """Caller supplies the authenticated release decision, never infer from files."""
    if released is not True:
        raise PermissionError("TEST analysis remains sealed")
    questions = defaultdict(list)
    families = {}
    for row in rows:
        if (
            row["pool"] != "TEST_ID"
            or row["protocol"] != "evidence_answer"
            or row.get("view") is not None
        ):
            continue
        model = row["model_id"]
        if row["step"] != (0 if model == BASELINE else 96):
            raise ValueError("Only common zero and final checkpoint support predictions")
        if row["family"] not in FAMILIES:
            raise ValueError("Unknown family")
        root = row["root_id"]
        if root in families and families[root] != row["family"]:
            raise ValueError("Conflicting root/family")
        families[root] = row["family"]
        questions[(model, root, row["qid"])].append((row["draw"], _score(row)["J"]))
    roots = defaultdict(list)
    for (model, root, _), samples in questions.items():
        if sorted(d for d, _ in samples) != list(range(4)):
            raise ValueError("Each TEST_ID question needs exactly four draws")
        roots[(model, root)].append(float(np.mean([v for _, v in samples])))
    if len(families) != 128 or any(len(v) != 16 for v in roots.values()):
        raise ValueError("Full TEST_ID roots and sixteen questions per root required")
    if any(sum(f == family for f in families.values()) != 32 for family in FAMILIES):
        raise ValueError("TEST_ID requires thirty-two roots per family")
    return {key: float(np.mean(values)) for key, values in roots.items()}, families


def primary_contrasts(rows, *, released=False):
    values, families = _test_root_values(rows, released=released)
    roots = sorted(families)
    result = []
    for left, right in PRIMARY_CONTRASTS:
        try:
            x = [[values[(f"SRF1_2_{left}_s{s}", r)] for r in roots] for s in SEEDS]
            y = [[values[(f"SRF1_2_{right}_s{s}", r)] for r in roots] for s in SEEDS]
        except KeyError as exc:
            raise ValueError(
                "Incomplete paired primary matrix; do not replace missing values"
            ) from exc
        result.append(
            dict(contrast=f"{left}-{right}", **paired_root_ci(x, y, [families[r] for r in roots]))
        )
    return result


def evaluate_predictions(rows, predictions, *, released=False):
    """Compare preregistered high-low (P1/P2) and compute-read (P3) gains.

    Baseline subtraction is explicit even though the same baseline cancels in
    between-arm effects. Root draws stay paired across all models and seeds.
    """
    if predictions.get("status") != "PREREGISTERED_BEFORE_SCIENCE":
        raise PermissionError("Require the pre-science prediction receipt")
    values, families = _test_root_values(rows, released=released)
    roots = sorted(families)
    output = []
    for hypothesis in ("P1", "P2", "P3"):
        spec = predictions[hypothesis]
        left, right = spec["contrast"].split("-")
        positive, negative = (
            ("computation_bottleneck", "reading_bottleneck")
            if hypothesis == "P3"
            else ("high", "low")
        )
        pos = sorted(f for f, label in spec["strata"].items() if label == positive)
        neg = sorted(f for f, label in spec["strata"].items() if label == negative)
        if not pos or not neg:
            output.append(
                dict(
                    hypothesis=hypothesis,
                    status="UNDEFINED_EMPTY_PREDICTED_STRATUM",
                    positive_families=pos,
                    negative_families=neg,
                )
            )
            continue
        family_effects, family_draws = {}, {}
        rng = np.random.default_rng(812901)
        for family in sorted(FAMILIES):
            ids = [r for r in roots if families[r] == family]
            try:
                baseline = np.asarray([values[(BASELINE, r)] for r in ids])
                x = (
                    np.asarray([[values[(f"SRF1_2_{left}_s{s}", r)] for r in ids] for s in SEEDS])
                    - baseline
                )
                y = (
                    np.asarray([[values[(f"SRF1_2_{right}_s{s}", r)] for r in ids] for s in SEEDS])
                    - baseline
                )
            except KeyError as exc:
                raise ValueError(
                    "Predictions require all matched endpoints and common-start TEST"
                ) from exc
            delta = (x - y).mean(axis=0)
            sampled = rng.choice(len(ids), size=(5000, len(ids)), replace=True)
            family_effects[family] = float(delta.mean())
            family_draws[family] = delta[sampled].mean(axis=1)
        effect = np.mean([family_effects[f] for f in pos]) - np.mean(
            [family_effects[f] for f in neg]
        )
        draws = np.mean([family_draws[f] for f in pos], axis=0) - np.mean(
            [family_draws[f] for f in neg], axis=0
        )
        output.append(
            dict(
                hypothesis=hypothesis,
                status="ESTIMATED",
                contrast=spec["contrast"],
                positive_families=pos,
                negative_families=neg,
                effect_pp=float(100 * effect),
                CI95_pp=(np.quantile(draws, [0.025, 0.975]) * 100).tolist(),
                predicted_direction="negative" if hypothesis == "P3" else "positive",
                point_estimate_matches_direction=bool(
                    effect < 0 if hypothesis == "P3" else effect > 0
                ),
                family_effects_pp={f: 100 * v for f, v in family_effects.items()},
                scope=(
                    "secondary directional prediction; "
                    "root uncertainty conditional on realised seeds"
                ),
            )
        )
    return output


def mechanism_check(branch_records, predictions):
    """All three scientific GATE seeds, first eight steps and all sixteen groups."""
    seen = set()
    pure = 0
    allowed = {
        "evidence",
        "readings",
        "all_joint_correct",
        "target_J_refined",
        "no_semantic_contrast",
    }
    for row in branch_records:
        key = (row["seed"], row["step"], row["prompt_index"])
        if (
            row.get("arm") != "GATE"
            or row["seed"] not in SEEDS
            or row["step"] not in range(1, 9)
            or row["prompt_index"] not in range(16)
        ):
            raise ValueError("Mechanism check requires GATE first-eight-step group records")
        if key in seen or row["branch"] not in allowed:
            raise ValueError("Duplicate or invalid branch record")
        seen.add(key)
        pure += row["branch"] in ("evidence", "readings")
    if len(seen) != 3 * 8 * 16:
        raise ValueError("Mechanism check cannot silently omit groups")
    expected = predictions["mechanism"]["expected_pure_semantic_branch_rate"]
    observed = pure / len(seen)
    return dict(
        expected=expected,
        observed=observed,
        pure_group_count=pure,
        total_group_count=len(seen),
        difference_pp=100 * (observed - expected),
        within_registered_ten_pp=abs(observed - expected) <= 0.10,
        status="SECONDARY_MECHANISM_CHECK",
        training_gate=False,
    )
