"""Registered SR-F2 numerical contrasts, separate from mechanism conclusions.

Rows carry model_id, family, root_id, qid, draw, and a v2 ``score`` dictionary.
Optional pool/protocol/view/step metadata is filtered to final TEST_ID evidence
responses. Callers must validate and release TEST artifacts before using them.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

FAMILIES = ("CROSS", "THRESHOLD", "TOPK", "INTERVAL")
SEEDS = (71001, 71002, 71003)
NEW_ARMS = ("EARLY", "J6", "DEC_PRIME", "ADAPT")
METRICS = ("A", "E", "E6", "p_read", "P", "P6", "J", "J6")
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 812901
CONFIDENCE_LEVEL = 0.9875


def _models(arm):
    if arm not in (*NEW_ARMS, "A", "J", "GATE", "DEC"):
        raise ValueError("Unknown registered contrast arm")
    prefix = "SRF2" if arm in NEW_ARMS else "SRF1_2"
    return {seed: f"{prefix}_{arm}_s{seed}" for seed in SEEDS}


def _selected(row):
    return (
        row.get("pool", "TEST_ID") == "TEST_ID"
        and row.get("protocol", "evidence_answer") == "evidence_answer"
        and row.get("view") is None
        and row.get("step", 96) == 96
    )


def _collect(rows, models, metric, families):
    if metric not in METRICS:
        raise ValueError("Unsupported semantic contrast metric")
    model_seed = {value: key for key, value in models.items()}
    cells = {seed: {} for seed in SEEDS}
    for row in rows:
        if row.get("model_id") not in model_seed or not _selected(row):
            continue
        if row.get("family") not in FAMILIES:
            raise ValueError("Unknown endpoint family")
        if row["family"] not in families:
            continue
        root, qid, draw = row.get("root_id"), row.get("qid"), row.get("draw")
        if not isinstance(root, str) or not root or not isinstance(qid, str) or not qid:
            raise ValueError("Nonempty root and question identities are required")
        if type(draw) is not int or draw not in range(4):
            raise ValueError("Final TEST_ID evidence requires four registered draws")
        score = row.get("score")
        if not isinstance(score, dict) or metric not in score:
            raise ValueError("Required v2 event score is missing; missingness is never zero")
        value = score[metric]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid finite event score")
        if metric != "p_read" and value not in (0, 1):
            raise ValueError("A binary event score is required")
        seed = model_seed[row["model_id"]]
        key = (row["family"], root, qid, draw)
        if key in cells[seed]:
            raise ValueError("Duplicate endpoint slot")
        cells[seed][key] = float(value)
    return cells


def _paired(rows_new, rows_old, arm, metric, family):
    if family is not None and family not in FAMILIES:
        raise ValueError("Unknown contrast family")
    families = (family,) if family is not None else FAMILIES
    new_models, old_models = _models(arm), _models("J")
    new = _collect(rows_new, new_models, metric, families)
    old = _collect(rows_old, old_models, metric, families)
    support = set(new[SEEDS[0]])
    if not support or any(set(table[seed]) != support for table in (new, old) for seed in SEEDS):
        raise ValueError("Require identical root/question/draw support for all three paired seeds")
    questions = defaultdict(set)
    root_family, question_root = {}, {}
    for name, root, qid, draw in support:
        if root in root_family and root_family[root] != name:
            raise ValueError("Root identity belongs to conflicting families")
        if qid in question_root and question_root[qid] != (name, root):
            raise ValueError("Question identity belongs to conflicting roots")
        root_family[root], question_root[qid] = name, (name, root)
        questions[(name, root, qid)].add(draw)
    if any(draws != set(range(4)) for draws in questions.values()):
        raise ValueError("Every paired question requires exactly four draws")
    root_questions = defaultdict(list)
    for name, root, qid in sorted(questions):
        root_questions[(name, root)].append(qid)
    arrays = {}
    for name in sorted(families):
        roots = sorted(root for f, root in root_questions if f == name)
        if not roots:
            raise ValueError("Every requested family requires paired roots")
        values = []
        for table in (new, old):
            values.append(
                np.asarray(
                    [
                        [
                            np.mean(
                                [
                                    np.mean(
                                        [table[seed][(name, root, qid, draw)] for draw in range(4)]
                                    )
                                    for qid in root_questions[(name, root)]
                                ]
                            )
                            for root in roots
                        ]
                        for seed in SEEDS
                    ]
                )
            )
        arrays[name] = dict(new=values[0], old=values[1], roots=roots)
    per_seed = []
    for seed in SEEDS:
        per_seed.append(
            dict(
                seed=seed,
                new_model_id=new_models[seed],
                old_model_id=old_models[seed],
                denominators=dict(
                    samples=len(support),
                    questions=len(questions),
                    roots=len(root_questions),
                    by_family={
                        name: dict(
                            samples=sum(key[0] == name for key in support),
                            questions=sum(key[0] == name for key in questions),
                            roots=len(arrays[name]["roots"]),
                            questions_per_root={
                                root: len(root_questions[(name, root)])
                                for root in arrays[name]["roots"]
                            },
                        )
                        for name in sorted(families)
                    },
                ),
            )
        )
    return arrays, per_seed


def _t_sensitivity(seed_effects):
    values = np.asarray(seed_effects, dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    sem = sd / math.sqrt(3)
    if sem == 0:
        statistic, p_value = None, 1.0 if mean == 0 else 0.0
        status = "ZERO_VARIANCE_ZERO_EFFECT" if mean == 0 else "ZERO_VARIANCE_NONZERO_EFFECT"
    else:
        statistic = mean / sem
        p_value = float(2 * student_t.sf(abs(statistic), df=2))
        status = "DEFINED"
    return dict(
        role="training-seed sensitivity; does not replace registered decision rule",
        degrees_of_freedom=2,
        paired_sd_pp=sd,
        standard_error_pp=sem,
        t_statistic=statistic,
        two_sided_p_value=p_value,
        status=status,
        CI95_pp=[
            mean - float(student_t.ppf(0.975, 2)) * sem,
            mean + float(student_t.ppf(0.975, 2)) * sem,
        ],
        CI98_75_pp=[
            mean - float(student_t.ppf(0.99375, 2)) * sem,
            mean + float(student_t.ppf(0.99375, 2)) * sem,
        ],
    )


def _summarize(arrays, per_seed, *, selected_families=None):
    selected = selected_families or {seed: set(arrays) for seed in SEEDS}
    if any(not selected[seed] or not selected[seed] <= set(arrays) for seed in SEEDS):
        raise ValueError("All three seeds require a nonempty supported family selection")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstraps = np.zeros(BOOTSTRAP_REPLICATES)
    new_means, old_means = np.zeros(3), np.zeros(3)
    per_seed = [dict(row) for row in per_seed]
    family_effects = {}
    for family, values in sorted(arrays.items()):
        new, old = values["new"], values["old"]
        weights = np.asarray(
            [1 / len(selected[seed]) if family in selected[seed] else 0 for seed in SEEDS]
        )
        delta = new - old
        new_means += weights * new.mean(axis=1)
        old_means += weights * old.mean(axis=1)
        samples = rng.integers(
            0, len(values["roots"]), size=(BOOTSTRAP_REPLICATES, len(values["roots"]))
        )
        bootstraps += (weights[:, None] * delta[:, samples].mean(axis=2)).mean(axis=0)
        family_effects[family] = (delta.mean(axis=1) * 100).tolist()
    seed_effects = (new_means - old_means) * 100
    interval = (np.quantile(bootstraps, [0.00625, 0.99375]) * 100).tolist()
    positive, negative = bool(np.all(seed_effects > 0)), bool(np.all(seed_effects < 0))
    excludes_zero = interval[0] > 0 or interval[1] < 0
    confirmed_positive, confirmed_negative = (
        interval[0] > 0 and positive,
        interval[1] < 0 and negative,
    )
    for index, seed in enumerate(SEEDS):
        chosen = sorted(selected[seed])
        denominators = per_seed[index]["denominators"]
        per_seed[index].update(
            new_rate=float(new_means[index]),
            old_rate=float(old_means[index]),
            effect_pp=float(seed_effects[index]),
            selected_families=chosen,
            family_effects_pp={family: family_effects[family][index] for family in sorted(arrays)},
            selected_denominators={
                key: sum(denominators["by_family"][family][key] for family in chosen)
                for key in ("samples", "questions", "roots")
            },
        )
    return dict(
        effect_pp=float(seed_effects.mean()),
        seed_effects_pp=seed_effects.tolist(),
        new_rate=float(new_means.mean()),
        old_rate=float(old_means.mean()),
        per_seed=per_seed,
        CI98_75_pp=interval,
        CI95_pp=(np.quantile(bootstraps, [0.025, 0.975]) * 100).tolist(),
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
        root_interval_excludes_zero=excludes_zero,
        all_three_same_sign=positive or negative,
        contrast_confirmed=confirmed_positive or confirmed_negative,
        superiority_confirmed=confirmed_positive,
        status="CONFIRMED_POSITIVE"
        if confirmed_positive
        else "CONFIRMED_NEGATIVE"
        if confirmed_negative
        else "ROOT_LEVEL_ONLY"
        if excludes_zero
        else "NOT_CONFIRMED",
        paired_t_sensitivity=_t_sensitivity(seed_effects),
        weighting=(
            "equal draws per question; equal questions per root; equal roots per family; "
            "equal selected families per seed; equal three seeds"
        ),
        bootstrap_scope=(
            "paired family-stratified roots, shared across three realised seeds; "
            "conditional on those training seeds"
        ),
    )


def contrast(rows_new, rows_old, arm, metric="J", family=None):
    """Three paired seeds; no resampling, dropping, or zero-filling missing slots."""
    arrays, per_seed = _paired(rows_new, rows_old, arm, metric, family)
    return dict(contrast=f"{arm}-J", metric=metric, family=family, **_summarize(arrays, per_seed))


def _require_full_coverage(result):
    for seed in result["per_seed"]:
        denominators = seed["denominators"]
        expected_families = [result["family"]] if result["family"] else FAMILIES
        for family in expected_families:
            counts = denominators["by_family"][family]
            if counts["samples"] != 2048 or counts["questions"] != 512 or counts["roots"] != 32:
                raise ValueError(
                    "Main report requires complete TEST_ID: 32 roots and 2048 draws per family"
                )
            if any(count != 16 for count in counts["questions_per_root"].values()):
                raise ValueError("Main report requires sixteen questions per TEST_ID root")


def _event_rates(rows, old_rows, arm, metric, family):
    result = contrast(rows, old_rows, arm, metric, family)
    return dict(
        metric=metric,
        rate=result["new_rate"],
        per_seed=[
            dict(
                seed=row["seed"],
                model_id=row["new_model_id"],
                rate=row["new_rate"],
                denominators=row["denominators"],
            )
            for row in result["per_seed"]
        ],
    )


def _untriggered(rows_new, rows_old, trigger_states):
    models = _models("ADAPT")
    selected = {}
    for seed, model in models.items():
        state = trigger_states.get(model)
        if not isinstance(state, dict) or not isinstance(state.get("active_families"), list):
            raise ValueError("Require a final active_families list for every ADAPT seed")
        active = state["active_families"]
        if len(set(active)) != len(active) or not set(active) <= set(FAMILIES):
            raise ValueError("Invalid final triggered families")
        selected[seed] = set(FAMILIES) - set(active)
    selection = {str(seed): sorted(families) for seed, families in selected.items()}
    if any(not families for families in selected.values()):
        return dict(
            status="UNDEFINED_NO_UNTRIGGERED_FAMILY",
            selected_families=selection,
            superiority_confirmed=False,
            plus_2pp_prediction_met=None,
            plus_2pp_is_decision_gate=False,
        )
    arrays, per_seed = _paired(rows_new, rows_old, "ADAPT", "A", None)
    result = _summarize(arrays, per_seed, selected_families=selected)
    result.update(
        contrast="ADAPT-J",
        metric="A",
        selected_families=selection,
        plus_2pp_prediction_met=result["effect_pp"] >= 2,
        plus_2pp_is_decision_gate=False,
    )
    return result


def _usage(semantic_group_usage):
    per_seed = []
    for seed, model in _models("ADAPT").items():
        record = semantic_group_usage.get(model)
        if not isinstance(record, dict):
            raise ValueError("Semantic group usage is required for each ADAPT seed")
        used, total = record.get("semantic_groups"), record.get("total_groups")
        if (
            type(total) is not int
            or total != 1536
            or type(used) is not int
            or not 0 <= used <= total
        ):
            raise ValueError("Require complete 96 x 16 group usage; no inferred denominator")
        per_seed.append(
            dict(
                seed=seed,
                model_id=model,
                semantic_groups=used,
                total_groups=total,
                fraction=used / total,
            )
        )
    fraction = sum(row["fraction"] for row in per_seed) / 3
    return dict(
        per_seed=per_seed,
        fraction=fraction,
        semantic_groups=sum(row["semantic_groups"] for row in per_seed),
        total_groups=4608,
        limit=0.6,
        criterion_met=fraction <= 0.6,
    )


def _noninferiority(result):
    lower = result["CI98_75_pp"][0] > -2
    seeds = all(value > -2 for value in result["seed_effects_pp"])
    return dict(
        margin_pp=-2,
        strict_lower_bound_above_margin=lower,
        all_three_seeds_strictly_above_margin=seeds,
        zero_crossing_is_allowed=True,
        confirmed=lower and seeds,
    )


def report_main_contrasts(rows_new, rows_old, *, trigger_states, semantic_group_usage):
    """Numeric C1-C4 evidence from complete final TEST_ID panels only.

    ``trigger_states`` maps each ADAPT model ID to its final active_families list.
    ``semantic_group_usage`` maps each to integer semantic_groups/total_groups;
    each total must be 1536. Threshold predictions and confirmation are separate.
    This is a main-contrast report, not an assertion that all endpoint delivery
    or diagnostic work is complete, and it never generates causal conclusions.
    """
    rows_new, rows_old = list(rows_new), list(rows_old)
    results = {
        "C1": contrast(rows_new, rows_old, "EARLY"),
        "C2": contrast(rows_new, rows_old, "J6", "A", "TOPK"),
        "C3": contrast(rows_new, rows_old, "DEC_PRIME"),
        "C4": contrast(rows_new, rows_old, "ADAPT"),
    }
    for result in results.values():
        _require_full_coverage(result)
    gate = contrast(rows_old, rows_old, "GATE")
    _require_full_coverage(gate)
    c1 = results["C1"]
    c1["predictions"] = dict(
        effect_at_least_2pp=c1["effect_pp"] >= 2,
        effect_at_least_60pct_gate=c1["effect_pp"] >= 0.6 * gate["effect_pp"],
        gate_reference_effect_pp=gate["effect_pp"],
        ratio_to_gate=None if gate["effect_pp"] == 0 else c1["effect_pp"] / gate["effect_pp"],
    )
    c1["prediction_met"] = (
        c1["predictions"]["effect_at_least_2pp"] and c1["predictions"]["effect_at_least_60pct_gate"]
    )
    c2 = results["C2"]
    c2["certificate_rates"] = {
        metric: _event_rates(rows_new, rows_old, "J6", metric, "TOPK")
        for metric in ("E6", "P6", "E")
    }
    c2["predictions"] = dict(
        effect_at_least_4pp=c2["effect_pp"] >= 4,
        E6_at_least_97pct=c2["certificate_rates"]["E6"]["rate"] >= 0.97,
        E8_below_60pct=c2["certificate_rates"]["E"]["rate"] < 0.6,
    )
    c2["prediction_met"] = all(c2["predictions"].values())
    c3 = results["C3"]
    c3["predictions"] = dict(
        effect_between_0_and_4pp=0 <= c3["effect_pp"] <= 4,
        paired_sd_at_most_1_5pp=c3["paired_t_sensitivity"]["paired_sd_pp"] <= 1.5,
    )
    c3["prediction_met"] = all(c3["predictions"].values())
    c4 = results["C4"]
    c4["noninferiority"] = _noninferiority(c4)
    c4["decision_rule"] = "CI98_75_lower_gt_minus_2pp_and_every_seed_gt_minus_2pp"
    c4["primary_decision_confirmed"] = c4["noninferiority"]["confirmed"]
    c4["primary_status"] = (
        "NONINFERIORITY_CONFIRMED"
        if c4["primary_decision_confirmed"]
        else "NONINFERIORITY_NOT_CONFIRMED"
    )
    c4["contrast_confirmation_role"] = "superiority diagnostic only; not the C4 decision rule"
    c4["untriggered_A"] = _untriggered(rows_new, rows_old, trigger_states)
    c4["semantic_group_usage"] = _usage(semantic_group_usage)
    c4["criteria_met"] = (
        c4["noninferiority"]["confirmed"]
        and c4["untriggered_A"]["superiority_confirmed"]
        and c4["semantic_group_usage"]["criterion_met"]
    )
    c4["criteria_status"] = "MET" if c4["criteria_met"] else "NOT_MET"
    return dict(
        status="MAIN_CONTRASTS_COMPLETE",
        contrasts=results,
        old_GATE_minus_J=gate,
        score_version="v2",
        primary_interval=CONFIDENCE_LEVEL,
        multiplicity="four main contrasts; two-sided 98.75% intervals",
        conclusions="NUMERICAL_STATUSES_ONLY",
        diagnostic_or_endpoint_delivery_completion_claimed=False,
    )
