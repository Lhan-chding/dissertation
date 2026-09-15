"""Cold deployment costs versus the same-permission direct measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .observation_study import service_arguments
from .study import jwrite, rng_seed


def add_costs(rows):
    result = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = result.get(key, 0) + value
    return result


def priced(cost, ratio, label_ratio=0.05, *, historical_label_bound="lower"):
    """Scenario price; historical labels have bounds, not measured free acquisition.

    The default remains an optimistic lower bound. Historical reused observations
    are charged once per global inference fingerprint within this archive/noise
    unit, independently of how often the same counts are read by fit banks.
    """
    if historical_label_bound not in ("lower", "upper"):
        raise ValueError("Historical label bound must be lower or upper")
    acquisition = cost.get("historical_reused_actions_charged", cost.get("reused_actions", 0))
    labels = cost.get("verified_labels", 0) + cost.get(
        f"historical_label_queries_{historical_label_bound}_bound", 0
    )
    return (
        cost.get("generated_actions", 0)
        + acquisition
        + ratio * cost.get("cross_policy_action_scores", 0)
        + label_ratio * labels
    )


def fit_acquisition_cost(packet, bank_budget, *, noise=0, actions_per_prompt=None):
    """Price selected paid fit information without duplicating old alias samples.

    A historical aggregate count row has no retained action IDs or label-query
    log. Its measured replay verified_labels counter therefore stays unchanged.
    The deployment scenario separately charges a lower/upper bound on missing
    label queries under the service's deterministic per-prompt/action label cache.
    Observed event categories give a lower bound on distinct labelled actions;
    n draws and (when supplied) the verified action-space size give an upper bound.
    """
    if type(bank_budget) is not int or bank_budget < 1 or type(noise) is not int or noise < 0:
        raise ValueError("Positive bank budget and nonnegative noise replica required")
    if actions_per_prompt is not None and (
        type(actions_per_prompt) is not int or actions_per_prompt < 1
    ):
        raise ValueError("Positive known action-space size required")
    metadata = packet["metadata"]
    n = metadata["n"]
    rows = [
        row
        for row in metadata["packets"]
        if row["noise_replica"] == noise and row["bank"] < bank_budget
    ]
    if sorted(row["bank"] for row in rows) != list(range(bank_budget)):
        raise ValueError("Exactly one packet per requested fit bank required")
    result = add_costs([row["cost"] for row in rows])
    historical = [
        row
        for row in rows
        if metadata["method"].replace("-", "_") == "O_IND" and row.get("old_counts_reused") is True
    ]
    fingerprints = set()
    historical_reads = 0
    counts_by_policy = {}
    prompt_ids = None
    levels = packet.get("legacy_total_levels")
    if levels is not None:
        levels = np.asarray(levels, dtype=np.float64)
    for row in historical:
        ids = tuple(row["prompt_ids"])
        if prompt_ids is None:
            prompt_ids = ids
        if ids != prompt_ids:
            raise ValueError("Historical fit prompt identities differ")
        names = [item["name"] for item in row["sample_layout"] if "counts" in item["fields"]]
        if len(names) != len(set(names)) or set(names) != set(row["sample_names"]):
            raise ValueError("Historical samples must identify unique paid count policies")
        expected_reads = len(names) * len(ids) * n
        if row["cost"].get("reused_actions", 0) != expected_reads:
            raise ValueError("Historical replay count does not match paid sample identities")
        historical_reads += expected_reads
        fingerprints.update(names)
        if levels is None:
            continue
        pairs = row["policy_fingerprints"]
        if len(pairs) != 2 or pairs[0][0] != pairs[1][0]:
            raise ValueError("Parent candidate order required for paid count binding")
        op_ids = (pairs[0][0], pairs[0][1], pairs[1][1])
        count_rows = levels[noise, row["bank"]] * n
        if (
            count_rows.shape != (3, len(ids), 4)
            or not np.isfinite(count_rows).all()
            or (count_rows < 0).any()
            or not np.array_equal(count_rows, np.rint(count_rows))
            or not (count_rows.sum(-1) == n).all()
        ):
            raise ValueError("Integer historical count recovery failed")
        for policy, count in zip(op_ids, count_rows, strict=True):
            if policy not in names:
                continue
            if policy in counts_by_policy and not np.array_equal(counts_by_policy[policy], count):
                raise ValueError("Shared historical alias has different paid counts")
            counts_by_policy[policy] = count.astype(np.int64)
    prompts = len(prompt_ids or ())
    unique_actions = len(fingerprints) * prompts * n
    other_reads = result.get("reused_actions", 0) - historical_reads
    if other_reads < 0:
        raise ValueError("Historical replay accounting mismatch")
    label_min = 0
    if fingerprints and levels is not None:
        if set(counts_by_policy) != fingerprints:
            raise ValueError("Missing paid counts for historical policy")
        label_min = int(np.any(np.stack(list(counts_by_policy.values())) > 0, axis=0).sum())
    label_max = prompts * min(actions_per_prompt or len(fingerprints) * n, len(fingerprints) * n)
    measured_labels = result.get("verified_labels", 0)
    # In a mixed old/new fit, unknown historical actions may overlap already
    # counted new labels. Bounds avoid assuming either complete or zero overlap.
    additional_min = max(0, label_min - measured_labels)
    additional_max = label_max
    if actions_per_prompt is not None:
        additional_max = min(additional_max, max(0, prompts * actions_per_prompt - measured_labels))
    status = (
        (
            "BOUNDED_NOT_MEASURED"
            if levels is not None
            else "UNOBSERVED_COUNTS_AND_ACTION_IDENTITIES"
        )
        if fingerprints
        else "NO_HISTORICAL_LABEL_ACQUISITION"
    )
    result.update(
        historical_reused_actions_charged=unique_actions + other_reads,
        historical_duplicate_reused_actions=historical_reads - unique_actions,
        historical_unique_policy_count=len(fingerprints),
        historical_label_queries_lower_bound=additional_min,
        historical_label_queries_upper_bound=additional_max,
        historical_distinct_label_queries_lower_bound=label_min,
        historical_distinct_label_queries_upper_bound=label_max,
        historical_label_cost_status=status,
        historical_action_identity_available=False if fingerprints else None,
        historical_label_zero_lower_bound_is_not_free=bool(fingerprints and additional_min == 0),
        historical_acquisition_is_scenario_charge=True,
        label_cache_scope="deterministic_prompt_action; old action identities not retained",
        historical_actions_per_prompt_bound=actions_per_prompt,
    )
    return result


def cold_direct_costs(parent_root, entry, ai, method, n, packet, out):
    from .observations import ToyWorldService, full_enumeration_logp, known_event_logp, measure

    theta, features, categories, prompt_ids, _ = service_arguments(parent_root, entry, ai)
    service = ToyWorldService.from_parameters(theta, features, categories, prompt_ids=prompt_ids)
    for bank in range(10, 14):
        policies = [f"b{bank}o{o}" for o in range(3)]
        seed = rng_seed(
            2026091502, entry["seed"], entry["arm"], int(ai), int(bank), n, 0, "evaluation"
        )
        replay = measure(
            service,
            [(policies[0], policies[1]), (policies[0], policies[2])],
            origin_id="origin",
            method=method,
            n=n,
            seed=seed,
            purpose="evaluation",
        )
        np.testing.assert_array_equal(replay.helmert_estimate, packet["helmert"][0, bank])
    direct = service.ledger.snapshot()
    fit = {
        str(b): fit_acquisition_cost(packet, b, noise=0, actions_per_prompt=categories.shape[1])
        for b in (1, 2, 4, 8)
    }
    policies = [f"b{bank}o{o}" for bank in range(10, 14) for o in range(3)]
    known_service = ToyWorldService.from_parameters(
        theta, features, categories, prompt_ids=prompt_ids
    )
    known = known_event_logp(known_service, policies)
    full_service = ToyWorldService.from_parameters(
        theta, features, categories, prompt_ids=prompt_ids
    )
    enumeration = full_enumeration_logp(full_service, policies)
    # Explicit diagnostic check of the real action identity, not assumed zeros.
    from .parent import load_parent_oracle

    true = load_parent_oracle(parent_root, entry["id"])["branch_p"][ai, 10:14].reshape(12, 72, 4)
    np.testing.assert_allclose(known["pX"], true[..., 0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(known["v"], 1 - true[..., 3], rtol=0, atol=1e-14)
    record = {
        "cost_schema_version": "cold-acquisition-v2",
        "trajectory_id": entry["id"],
        "seed": entry["seed"],
        "arm": entry["arm"],
        "anchor": [8, 24, 40][ai],
        "observation": method,
        "n": n,
        "fit_by_bank_budget": fit,
        "direct_four_evaluation_banks": direct,
        "known_event_four_evaluation_banks": known["cost"],
        "full_enumeration_four_evaluation_banks": enumeration["cost"],
        "direct_replay_matches_paid_packet": True,
        "known_event_exact_error_max": float(
            max(
                np.abs(known["pX"] - true[..., 0]).max(),
                np.abs(known["v"] - (1 - true[..., 3])).max(),
            )
        ),
        "evaluation_banks": [10, 11, 12, 13],
        "cold_cache_each_baseline": True,
        "noise_replica_for_deployment_cost": 0,
        "cost_replay_additional_actions": direct["generated_actions"],
        "historical_fit_acquisition_cost_is_charged": True,
        "historical_alias_acquisition_dedup_scope": (
            "global fingerprint within archive/anchor/noise"
        ),
        "historical_labels_are_not_recorded_as_free": True,
        "historical_label_acquisition": (
            "bounds from paid event counts; original action-level verification logs unavailable"
        ),
        "new_optimizer_updates": 0,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    jwrite(out, record)
    return record


def frontier(configurations, records):
    """Break even is limited to the four actually tested new-bank queries."""
    from .cost import break_even

    result = []
    configs = {
        row["configuration_id"]: row
        for row in configurations
        if row.get("n", 0) > 0
        and row.get("method") in ("C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6")
    }
    for cid, c in configs.items():
        matched = [r for r in records if r["observation"] == c["observation"] and r["n"] == c["n"]]
        if not matched:
            continue
        access = "SAMPLE_AND_LOGP" if "LR" in c["observation"] else "SAMPLE_ONLY"
        calibration = add_costs([r["fit_by_bank_budget"][str(c["fit_banks"])] for r in matched])
        direct = add_costs([r["direct_four_evaluation_banks"] for r in matched])
        known = add_costs([r["known_event_four_evaluation_banks"] for r in matched])
        for ratio in (0.05, 0.25, 1.0):
            fit_cost = priced(calibration, ratio)
            fit_upper = priced(calibration, ratio, historical_label_bound="upper")
            for baseline, counters in [("D_DIRECT", direct)] + (
                [("D_KNOWN_EVENT_LOGP", known)] if access == "SAMPLE_AND_LOGP" else []
            ):
                total_direct = priced(counters, ratio)
                crossing = break_even(fit_cost, 0.0, total_direct / 4)
                queries = crossing["queries"]
                result.append(
                    {
                        "configuration_id": cid,
                        "access_regime": access,
                        "score_to_generation_ratio": ratio,
                        "label_to_generation_ratio": 0.05,
                        "comparison_baseline": baseline,
                        "includes_fit_score_label": True,
                        "calibration_cost": fit_cost,
                        "direct_cost_four_queries": total_direct,
                        "calibration_cost_lower_bound": fit_cost,
                        "calibration_cost_upper_bound": fit_upper,
                        "historical_label_cost_is_unmeasured": calibration.get(
                            "historical_unique_policy_count", 0
                        )
                        > 0,
                        "historical_label_queries_lower_bound": calibration.get(
                            "historical_label_queries_lower_bound", 0
                        ),
                        "historical_label_queries_upper_bound": calibration.get(
                            "historical_label_queries_upper_bound", 0
                        ),
                        "historical_acquisition_duplicate_actions_removed": calibration.get(
                            "historical_duplicate_reused_actions", 0
                        ),
                        "surrogate_query_incremental_model_evaluations": 0,
                        "fit_compute_seconds_separately_measured": True,
                        "break_even_queries": queries,
                        "tested_query_domain": 4,
                        "within_domain": queries is not None and queries <= 4,
                        "cost_feasible": fit_cost <= total_direct,
                        "anchor_unit_count": len(matched),
                        "cost_feasible_at_label_upper_bound": fit_upper <= total_direct,
                        "cost_feasible_is_optimistic_lower_bound": True,
                        "known_event_cost_advantage": fit_cost < priced(known, ratio),
                        "toy_forward_cost_not_vlm_cost": True,
                        "zero_incremental_surrogate_cost_is_optimistic_lower_bound": True,
                        "status": "COST_CROSSING_WITHIN_TESTED_DOMAIN"
                        if queries is not None and queries <= 4
                        else "NO_COST_FEASIBLE_SURROGATE",
                    }
                )
    return result
