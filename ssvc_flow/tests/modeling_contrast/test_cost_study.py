import numpy as np
import pytest

from src.modeling_contrast.cost_study import fit_acquisition_cost, frontier, priced


def historical_packet():
    counts = np.array(
        [
            [
                [[1, 2, 13, 0], [0, 1, 14, 1]],
                [[2, 2, 11, 1], [1, 1, 14, 0]],
                [[1, 2, 13, 0], [0, 1, 14, 1]],
            ],
            [
                [[1, 2, 13, 0], [0, 1, 14, 1]],
                [[3, 1, 12, 0], [0, 2, 11, 3]],
                [[1, 2, 13, 0], [0, 1, 14, 1]],
            ],
        ],
        dtype=np.uint16,
    )
    rows = []
    for bank, candidate in enumerate(("u", "v")):
        rows.append(
            dict(
                bank=bank,
                noise_replica=0,
                old_counts_reused=True,
                prompt_ids=["p0", "p1"],
                policy_fingerprints=[["b", candidate], ["b", "b"]],
                sample_names=["b", candidate],
                sample_layout=[{"name": x, "fields": ["counts"]} for x in ("b", candidate)],
                cost={
                    "generated_actions": 0,
                    "reused_actions": 64,
                    "verified_labels": 0,
                    "cross_policy_action_scores": 0,
                },
            )
        )
    return {
        "legacy_total_levels": counts[None].astype(float) / 16,
        "metadata": {"method": "O_IND", "n": 16, "packets": rows},
    }


def test_historical_fit_acquisition_deduplicates_global_policy_fingerprints():
    cost = fit_acquisition_cost(historical_packet(), 2, actions_per_prompt=16)
    assert cost["reused_actions"] == 128  # Original physical replay counter remains auditable.
    assert cost["historical_reused_actions_charged"] == 96
    assert cost["historical_duplicate_reused_actions"] == 32
    assert cost["historical_unique_policy_count"] == 3
    assert cost["verified_labels"] == 0
    assert cost["historical_label_queries_lower_bound"] == 8
    assert cost["historical_label_queries_upper_bound"] == 32
    assert cost["historical_label_cost_status"] == "BOUNDED_NOT_MEASURED"
    assert priced(cost, 0.25) == pytest.approx(96.4)
    assert priced(cost, 0.25, historical_label_bound="upper") == pytest.approx(97.6)


def test_new_independent_draws_not_deduplicated_by_policy_identity():
    packet = historical_packet()
    for row in packet["metadata"]["packets"]:
        row["old_counts_reused"] = False
        row["cost"].update(generated_actions=64, reused_actions=0, verified_labels=4)
    cost = fit_acquisition_cost(packet, 2, actions_per_prompt=16)
    assert cost["generated_actions"] == 128
    assert cost["historical_reused_actions_charged"] == 0
    assert cost["historical_unique_policy_count"] == 0
    assert cost["historical_label_queries_lower_bound"] == 0
    assert priced(cost, 0.25) == pytest.approx(128.4)


def test_historical_alias_count_conflict_fails_before_cost_claim():
    packet = historical_packet()
    packet["legacy_total_levels"][0, 1, 0, 0] += [1 / 16, -1 / 16, 0, 0]
    with pytest.raises(ValueError, match="alias"):
        fit_acquisition_cost(packet, 2, actions_per_prompt=16)


def test_missing_old_action_labels_are_explicitly_bounded_when_counts_unavailable():
    packet = historical_packet()
    del packet["legacy_total_levels"]
    cost = fit_acquisition_cost(packet, 2, actions_per_prompt=16)
    assert cost["historical_label_queries_lower_bound"] == 0
    assert cost["historical_label_queries_upper_bound"] == 32
    assert cost["historical_label_cost_status"] == "UNOBSERVED_COUNTS_AND_ACTION_IDENTITIES"
    assert cost["historical_label_zero_lower_bound_is_not_free"] is True


def test_frontier_keeps_access_regimes_and_discloses_label_cost_interval():
    cost = fit_acquisition_cost(historical_packet(), 2, actions_per_prompt=16)
    record = {
        "observation": "O_IND",
        "n": 16,
        "fit_by_bank_budget": {"2": cost},
        "direct_four_evaluation_banks": {"generated_actions": 120, "verified_labels": 10},
        "known_event_four_evaluation_banks": {
            "cross_policy_action_scores": 6,
            "verified_labels": 3,
        },
    }
    config = {
        "configuration_id": "finite",
        "method": "C2",
        "observation": "O_IND",
        "n": 16,
        "fit_banks": 2,
    }
    rows = frontier([config], [record])
    assert len(rows) == 3
    assert {row["comparison_baseline"] for row in rows} == {"D_DIRECT"}
    assert all(row["calibration_cost_upper_bound"] > row["calibration_cost"] for row in rows)
    assert all(row["historical_label_cost_is_unmeasured"] for row in rows)
    assert all(row["zero_incremental_surrogate_cost_is_optimistic_lower_bound"] for row in rows)


def test_price_rejects_unknown_label_bound():
    with pytest.raises(ValueError):
        priced({}, 0.25, historical_label_bound="exact")


def test_logp_frontier_retains_cold_known_event_comparison():
    cost = {"generated_actions": 12, "cross_policy_action_scores": 3, "verified_labels": 4}
    record = {
        "observation": "O_LR_ORIGIN",
        "n": 16,
        "fit_by_bank_budget": {"2": cost},
        "direct_four_evaluation_banks": {"generated_actions": 120, "verified_labels": 10},
        "known_event_four_evaluation_banks": {
            "cross_policy_action_scores": 6,
            "verified_labels": 3,
        },
    }
    config = {
        "configuration_id": "lr",
        "method": "C2",
        "observation": "O_LR_ORIGIN",
        "n": 16,
        "fit_banks": 2,
    }
    rows = frontier([config], [record])
    assert len(rows) == 6
    assert {row["comparison_baseline"] for row in rows} == {"D_DIRECT", "D_KNOWN_EVENT_LOGP"}
    assert all(
        not row["cost_feasible"]
        for row in rows
        if row["comparison_baseline"] == "D_KNOWN_EVENT_LOGP"
    )
