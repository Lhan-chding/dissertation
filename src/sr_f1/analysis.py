"""Read-only, root-paired SR-F1 statistics, credit audits and diagnostic summaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .evaluation import ARMS, BASELINE, SEEDS, TEST_POOLS, VIEWS

FAMILIES = ("CROSS", "THRESHOLD", "TOPK", "INTERVAL")
PRIMARY_CONTRASTS = (("J", "A"), ("PART", "J"), ("GATE", "J"), ("GATE", "PART"), ("GATE", "DEC"))
CI_SCOPE = (
    "root measurement uncertainty conditional on the realised training paths; "
    "not a training-seed or model-population confidence interval"
)


def paired_root_ci(x, y, families, replicates=5000, seed=812901):
    """A bootstrap root carries every seed/model/protocol/draw; never resample draws."""
    x, y, fam = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(families)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != len(fam) or x.shape[0] != 3:
        raise ValueError("Require three matched seeds by the identical roots")
    if set(fam.tolist()) != set(FAMILIES):
        raise ValueError("All four preregistered families are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Technical missing values must not become zero scores")
    if replicates != 5000 or seed != 812901:
        raise ValueError("Bootstrap identity is frozen at 5000 replicates, seed 812901")
    groups = [np.flatnonzero(fam == f) for f in sorted(FAMILIES)]
    delta = x - y
    points = np.stack([delta[:, group].mean(axis=1) for group in groups]).mean(axis=0)
    root_average = delta.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.zeros(replicates)
    for indices in groups:
        sampled = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        draws += root_average[sampled].mean(axis=1) / len(groups)
    return dict(
        effect_probability=float(points.mean()),
        effect_pp=float(100 * points.mean()),
        seed_effects_pp=(100 * points).tolist(),
        seed_effect_range_pp=[float(100 * points.min()), float(100 * points.max())],
        CI95_pp=(100 * np.quantile(draws, [0.025, 0.975])).tolist(),
        CI99_pp=(100 * np.quantile(draws, [0.005, 0.995])).tolist(),
        family_count=4,
        root_count=len(fam),
        training_seed_count=3,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        multiplicity="99% per contrast, Bonferroni family of 5 primary contrasts",
        practical_reference_pp=2,
        practical_reference_is_pass_gate=False,
        scope=CI_SCOPE,
    )


def conditional_rate(numerator, denominator):
    return dict(
        numerator=int(numerator),
        denominator=int(denominator),
        probability=float(numerator / denominator) if denominator else None,
        status="DEFINED" if denominator else "UNDEFINED_NO_SUPPORT",
    )


def aggregate_results(rows):
    """K within question -> questions within root -> roots within family -> families.

    Returns an auditable long table at model/panel/protocol/view/family/variant.
    Full-panel effects below additionally require complete frozen slot coverage.
    """
    groups = defaultdict(list)
    for row in rows:
        key = tuple(
            row[k] for k in ("model_id", "step", "pool", "protocol", "view", "family", "variant")
        )
        groups[key].append(row)
    output = []
    fields = ("model_id", "step", "pool", "protocol", "view", "family", "variant")
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        record = dict(zip(fields, key, strict=True))
        record["arm"] = None if record["model_id"] == BASELINE else record["model_id"].split("_")[1]
        record["paired_seed"] = (
            None if record["model_id"] == BASELINE else int(record["model_id"].split("_s")[1])
        )
        record["generated_count"] = len(group)
        record["root_count"] = len({r["root_id"] for r in group})
        record["truncated_count"] = sum(r["truncated"] for r in group)
        metrics = (
            ("A", "L_json", "L_answer")
            if record["protocol"] == "answer_only"
            else ("A", "E", "P", "J", "p_read", "L_json", "L_answer", "L_evidence")
        )
        for metric in metrics:
            questions = defaultdict(list)
            for row in group:
                questions[(row["root_id"], row["qid"])].append(row["independent_score"][metric])
            roots = defaultdict(list)
            for (root_id, _), values in questions.items():
                roots[root_id].append(float(np.mean(values)))
            record[metric] = float(np.mean([np.mean(v) for v in roots.values()]))
        if record["protocol"] == "evidence_answer":
            support = [r for r in group if r["independent_score"]["P"]]
            record["task_execution_failure_given_correct_evidence"] = conditional_rate(
                sum(not r["independent_score"]["A"] for r in support), len(support)
            )
        record["mean_completion_tokens"] = float(
            np.mean(
                [
                    r["completion_token_count"]
                    if "completion_token_count" in r
                    else len(r["tokens"])
                    for r in group
                ]
            )
        )
        record["mean_self_sampling_nll"] = float(
            np.mean(
                [
                    r["self_sampling_nll"]
                    if "self_sampling_nll" in r
                    else -np.mean(r["old_logprobs"])
                    for r in group
                ]
            )
        )
        output.append(record)
    return output


def absolute_panel_results(rows):
    """Absolute metrics with the registered K/question/root/family hierarchy."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["step"], row["pool"], row["protocol"], row["view"])].append(
            row
        )
    output = []
    for key, panel in sorted(grouped.items(), key=lambda item: str(item[0])):
        model, step, pool, protocol, view = key
        record = dict(
            model_id=model,
            step=step,
            pool=pool,
            protocol=protocol,
            view=view,
            generated_count=len(panel),
            root_count=len({r["root_id"] for r in panel}),
        )
        metrics = (
            ("A", "L_json", "L_answer")
            if protocol == "answer_only"
            else ("A", "E", "P", "J", "p_read", "L_json", "L_answer", "L_evidence")
        )
        for metric in metrics:
            questions = defaultdict(list)
            for row in panel:
                questions[(row["family"], row["root_id"], row["qid"])].append(
                    row["independent_score"][metric]
                )
            roots = defaultdict(list)
            for (family, root, _), values in questions.items():
                roots[(family, root)].append(float(np.mean(values)))
            families = defaultdict(list)
            for (family, _), values in roots.items():
                families[family].append(float(np.mean(values)))
            record[metric] = float(np.mean([np.mean(values) for values in families.values()]))
        record["weighting"] = (
            "draws within qid, qids within root, roots within family, families equally"
        )
        output.append(record)
    return output


def _root_scores(rows, pool, protocol, metric):
    questions = defaultdict(list)
    family_map = {}
    for row in rows:
        if row["pool"] != pool or row["protocol"] != protocol or row["view"] is not None:
            continue
        if row["model_id"] == BASELINE:
            continue
        if row["step"] != 96:
            raise ValueError("Tests cannot select intermediate checkpoints")
        key = (row["model_id"], row["root_id"], row["qid"])
        questions[key].append((row["draw"], row["independent_score"][metric]))
        family_map[row["root_id"]] = row["family"]
    roots = defaultdict(list)
    for (model, root, qid), values in questions.items():
        if sorted(draw for draw, _ in values) != list(range(4)):
            raise ValueError(f"Missing/duplicate repeat for {model}/{qid}")
        roots[(model, root)].append(np.mean([value for _, value in values]))
    expected_questions = 16 if pool == "TEST_ID" else 2
    output = {}
    for key, values in roots.items():
        if len(values) != expected_questions:
            raise ValueError("Incomplete root: cannot delete missing technical slots")
        output[key] = float(np.mean(values))
    return output, family_map


def panel_contrasts(
    rows, pool="TEST_ID", protocol="evidence_answer", metric="J", unavailable_models=()
):
    if pool not in TEST_POOLS or protocol not in ("evidence_answer", "answer_only"):
        raise ValueError("Unknown fixed test panel/protocol")
    if protocol == "answer_only" and metric != "A":
        raise ValueError("Answer-only evaluation has no evidence metric")
    retained = [r for r in rows if r["model_id"] not in unavailable_models]
    values, families = _root_scores(retained, pool, protocol, metric)
    roots = sorted(families)
    expected_roots = 128 if pool == "TEST_ID" else 64
    results = []
    for left, right in PRIMARY_CONTRASTS:
        required = [f"SRF1_{arm}_s{seed}" for arm in (left, right) for seed in SEEDS]
        missing = [m for m in required if m in unavailable_models]
        record = dict(contrast=f"{left}-{right}", pool=pool, protocol=protocol, metric=metric)
        if missing:
            results.append(
                dict(
                    record,
                    status="UNDEFINED_TECHNICAL_FAILURE",
                    unavailable_models=missing,
                    effect_pp=None,
                )
            )
            continue
        if len(roots) != expected_roots or any(
            (m, r) not in values for m in required for r in roots
        ):
            raise ValueError("Incomplete paired test panel")
        x = [[values[(f"SRF1_{left}_s{s}", r)] for r in roots] for s in SEEDS]
        y = [[values[(f"SRF1_{right}_s{s}", r)] for r in roots] for s in SEEDS]
        results.append(
            dict(record, status="ESTIMATED", **paired_root_ci(x, y, [families[r] for r in roots]))
        )
    return results


def monitor_credit_audit(rows):
    """All five feedbacks are re-evaluated on the SAME 16-prompt blocks."""
    from .contract import reward_advantages

    blocks = defaultdict(list)
    for row in rows:
        if row["pool"] == "MONITOR":
            blocks[(row["model_id"], row["monitor_block"])].append(row)
    result = []
    for (model, block), batch in sorted(blocks.items()):
        prompts = sorted({r["qid"] for r in batch})
        if len(prompts) != 16 or len(batch) != 128:
            raise ValueError("DEC audit requires the complete fixed 16x8 monitor block")
        lookup = {(r["qid"], r["draw"]): r for r in batch}
        if len(lookup) != 128:
            raise ValueError("Duplicate MONITOR draw")
        scores = [
            [lookup[(qid, draw)]["independent_score"] for draw in range(8)] for qid in prompts
        ]
        components = np.asarray(
            [
                [[s["A"], s["J"], s["E"], s["p_read"], s["E"] * s["p_read"]] for s in g]
                for g in scores
            ]
        )
        block_record = dict(
            model_id=model,
            monitor_block=block,
            qids=prompts,
            component_order=["A", "J", "E", "p_read", "E*p_read"],
            component_means=components.mean(axis=1).tolist(),
            component_population_std=components.std(axis=1).tolist(),
            component_covariance=[np.cov(c, rowvar=False, ddof=0).tolist() for c in components],
            arms={},
        )
        for arm in ARMS:
            advantages, info = reward_advantages(arm, scores)
            advances = np.asarray(advantages, dtype=float)
            if advances.shape != (16, 8):
                raise ValueError("Final advantage shape changed")
            events = {}
            for event, column in (("A", 0), ("J", 1), ("E", 2)):
                event_values = components[:, :, column]
                events[event] = dict(
                    W_plus=float((np.maximum(advances, 0) * event_values).sum()),
                    W_minus=float((np.maximum(-advances, 0) * event_values).sum()),
                    count=int(event_values.sum()),
                )
            ties = {}
            for coarse, col in (("A", 0), ("J", 1)):
                ties[coarse] = []
                for g in range(16):
                    coarse_values = components[g, :, col]
                    ties[coarse].append(
                        float(
                            sum(
                                np.sum(coarse_values == value)
                                / 8
                                * advances[g, coarse_values == value].var()
                                for value in np.unique(coarse_values)
                            )
                        )
                    )
            info.update(
                positive_negative_credit=events,
                within_prompt_coarse_tie_variance=ties,
                final_advantage_nonconstant_groups=[bool(np.ptp(g) > 0) for g in advances],
            )
            scalar = info.get("scalar_reward")
            info["scalar_reward_nonconstant_groups"] = (
                None if scalar is None else [bool(np.ptp(g) > 0) for g in scalar]
            )
            # Preserve the per-trajectory values alongside every arm's final credit.
            info["trajectory_components"] = components.tolist()
            block_record["arms"][arm] = info
        result.append(block_record)
    return result


def paired_variant_results(rows, tasks):
    """Independent contexts paired by draw index; agreement is distinct from success."""
    lookup = {}
    for row in rows:
        if row["pool"] == "TEST_ID":
            key = tuple(
                row[k] for k in ("model_id", "protocol", "root_id", "chart", "variant", "draw")
            )
            if key in lookup:
                raise ValueError("Duplicate variant-pair slot")
            lookup[key] = row
    groups = defaultdict(list)
    from .contract import fraction_of, load_json_strict

    for key, original in lookup.items():
        model, protocol, root, chart, variant, draw = key
        pairs = (
            ("v1", "v2", "v3", "v4", "v5")
            if variant == "v0"
            else (("v7",) if variant == "v6" else ())
        )
        for partner in pairs:
            other = lookup.get((model, protocol, root, chart, partner, draw))
            if other is None:
                raise ValueError("Missing registered same-root variant partner")
            equal = None
            try:
                a = (
                    original["parsed_answer"]
                    if "parsed_answer" in original
                    else load_json_strict(original["raw_text"])["answer"]
                )
                b = (
                    other["parsed_answer"]
                    if "parsed_answer" in other
                    else load_json_strict(other["raw_text"])["answer"]
                )
                equal = int(
                    (None if a is None else fraction_of(a, text=True))
                    == (None if b is None else fraction_of(b, text=True))
                )
            except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError):
                pass
            strata = "all"
            if partner == "v4":
                strata = (
                    "same_gold_answer"
                    if tasks[original["qid"]]["answer"] == tasks[other["qid"]]["answer"]
                    else "different_gold_answer"
                )
            metrics = ("A",) if protocol == "answer_only" else ("A", "J")
            for metric in metrics:
                groups[
                    (
                        model,
                        protocol,
                        variant + "_" + partner,
                        strata,
                        metric,
                        original["family"],
                        root,
                    )
                ].append(
                    (
                        original["independent_score"][metric] * other["independent_score"][metric],
                        equal,
                    )
                )
    output = []
    for key, vals in groups.items():
        model, protocol, pair, stratum, metric, family, root = key
        agreements = [x[1] for x in vals if x[1] is not None]
        output.append(
            dict(
                model_id=model,
                protocol=protocol,
                pair=pair,
                stratum=stratum,
                metric=metric,
                family=family,
                root_id=root,
                both_correct_probability=float(np.mean([x[0] for x in vals])),
                answer_agreement=conditional_rate(sum(agreements), len(agreements)),
                pair_count=len(vals),
                scope="independent samples paired by draw; not a trajectory causal effect",
            )
        )
    return output


def diagnostic_results(rows):
    selected = [r for r in rows if r["pool"] == "MONITOR_DIAGNOSTIC_ONLY"]
    summary = aggregate_results(selected)
    for row in summary:
        row["scope"] = (
            "behaviour under granted information; not pure perception/computation identification"
        )
        if row["view"] not in VIEWS:
            raise ValueError("Unregistered diagnostic view")
    return summary


def analyze(root, run_matrix_final):
    # The release module owns the seal gate; public analysis cannot bypass it.
    from .release import release

    return release(root, run_matrix_final)


def response_semantics(raw_text, task, scores):
    """Truth-anchored error locations; no inference about internal algorithms."""
    from .contract import fraction_of, load_json_strict

    result = dict(
        common_scorable=bool(scores["L_json"] and scores["L_evidence"] and scores["L_answer"]),
        semantic_event={k: scores[k] for k in ("A", "E", "P", "J")},
        reading_error_locations=[],
        answer_absolute_error=None,
        task_execution_failure_with_correct_evidence=bool(scores["P"] and not scores["A"]),
        C=scores["C"],
    )
    try:
        obj = load_json_strict(raw_text)
        if scores["L_answer"] and obj.get("answer") is not None and task["answer"] is not None:
            result["answer_absolute_error"] = float(
                abs(fraction_of(obj["answer"], text=True) - fraction_of(task["answer"], text=True))
            )
        if scores["L_evidence"]:
            evidence = {(e["series"], e["category"]): e["value"] for e in obj["evidence"]}
            for truth in task["required_evidence"]:
                key = (truth["series"], truth["category"])
                kind = (
                    "missing"
                    if key not in evidence
                    else (
                        "wrong_value"
                        if fraction_of(evidence[key]) != fraction_of(truth["value"])
                        else None
                    )
                )
                if kind:
                    result["reading_error_locations"].append(
                        dict(series=key[0], category=key[1], kind=kind)
                    )
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError):
        pass
    return result


def diagnostic_view_contrasts(rows):
    panels = defaultdict(dict)
    for row in rows:
        if row["pool"] == "MONITOR" and row["variant"] == "v0":
            view = "original"
        elif row["pool"] == "MONITOR_DIAGNOSTIC_ONLY":
            view = row["view"]
        else:
            continue
        panels[(row["model_id"], row["qid"], row["draw"])][view] = row
    pairs = (
        ("keys_hint", "original"),
        ("gold_values", "original"),
        ("text_values", "original"),
        ("gold_values", "neutral_hint"),
        ("original", "blank_image"),
    )
    groups = defaultdict(list)
    for (model, _qid, _draw), views in panels.items():
        for left, right in pairs:
            if left not in views or right not in views:
                raise ValueError("Diagnostic difference requires both independent-context views")
            a, b = views[left], views[right]
            for metric in ("A", "E", "P", "J"):
                groups[(model, left, right, metric, a["family"], a["root_id"])].append(
                    a["independent_score"][metric] - b["independent_score"][metric]
                )
    family_values = defaultdict(list)
    for (model, left, right, metric, family, _root), values in groups.items():
        family_values[(model, left, right, metric, family)].append(float(np.mean(values)))
    pooled = defaultdict(list)
    for (model, left, right, metric, _family), values in family_values.items():
        pooled[(model, left, right, metric)].append(float(np.mean(values)))
    return [
        dict(
            model_id=model,
            left_view=left,
            right_view=right,
            metric=metric,
            delta_probability=float(np.mean(values)),
            scope="behaviour under explicit information grants; not pure mechanism identification",
        )
        for (model, left, right, metric), values in sorted(pooled.items())
    ]
