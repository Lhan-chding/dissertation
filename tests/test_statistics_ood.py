"""Deterministic statistical summaries and preregistered OOD comparisons."""

import pytest

from compbias.eval.ood import compute_ood_metrics
from compbias.eval.statistics import bootstrap_mean_ci, holm_adjust, paired_bootstrap_delta


def _primitive_compensability_profile(
    mechanism: str, *, compensatory: bool
) -> list[dict[str, object]]:
    return [
        {
            "error_id": mechanism,
            "rollout_rewards": [0, int(compensatory)],
        }
    ]


def test_bootstrap_interval_is_seeded_and_exact_for_a_constant_sample() -> None:
    first = bootstrap_mean_ci([5.0] * 20, confidence=0.95, n_resamples=500, seed=9)
    second = bootstrap_mean_ci([5.0] * 20, confidence=0.95, n_resamples=500, seed=9)

    assert first == second
    assert (first.mean, first.low, first.high) == (5.0, 5.0, 5.0)


def test_paired_bootstrap_preserves_pairing_and_delta_direction() -> None:
    interval = paired_bootstrap_delta(
        before=[3.0, 4.0, 8.0],
        after=[2.0, 3.0, 7.0],
        confidence=0.95,
        n_resamples=500,
        seed=11,
    )

    assert interval.mean == pytest.approx(-1.0)
    assert interval.low == pytest.approx(-1.0)
    assert interval.high == pytest.approx(-1.0)

    with pytest.raises(ValueError, match=r"paired|length"):
        paired_bootstrap_delta([1.0], [1.0, 2.0], seed=11)


def test_holm_adjustment_is_key_preserving_and_monotone() -> None:
    adjusted = holm_adjust({"h1": 0.01, "h2": 0.04, "h3": 0.03})

    assert adjusted == pytest.approx({"h1": 0.03, "h2": 0.06, "h3": 0.06})
    assert all(0 <= value <= 1 for value in adjusted.values())


def _iid_records() -> tuple[dict[str, object], ...]:
    return (
        {
            "sample_id": "a",
            "scene": {"value": 1},
            "error_mechanism": "offset_plus_2",
            "answer_correct": True,
            "prompt_error_profile": _primitive_compensability_profile(
                "offset_plus_2", compensatory=False
            ),
            "counterfactual_consistent": True,
        },
        {
            "sample_id": "b",
            "scene": {"value": 2},
            "error_mechanism": "offset_plus_2",
            "answer_correct": True,
            "prompt_error_profile": _primitive_compensability_profile(
                "offset_plus_2", compensatory=True
            ),
            "counterfactual_consistent": True,
        },
        {
            "sample_id": "c",
            "scene": {"value": 3},
            "error_mechanism": "offset_plus_2",
            "answer_correct": False,
            "prompt_error_profile": _primitive_compensability_profile(
                "offset_plus_2", compensatory=True
            ),
            "counterfactual_consistent": False,
        },
    )


def _ood_records() -> tuple[dict[str, object], ...]:
    return (
        {
            "sample_id": "a",
            "scene": {"value": 1},
            "error_mechanism": "offset_minus_2",
            "answer_correct": True,
            "prompt_error_profile": _primitive_compensability_profile(
                "offset_minus_2", compensatory=False
            ),
            "counterfactual_consistent": True,
        },
        {
            "sample_id": "b",
            "scene": {"value": 2},
            "error_mechanism": "offset_minus_2",
            "answer_correct": False,
            "prompt_error_profile": _primitive_compensability_profile(
                "offset_minus_2", compensatory=True
            ),
            "counterfactual_consistent": False,
        },
        {
            "sample_id": "c",
            "scene": {"value": 3},
            "error_mechanism": "offset_minus_2",
            "answer_correct": False,
            "prompt_error_profile": _primitive_compensability_profile(
                "offset_minus_2", compensatory=True
            ),
            "counterfactual_consistent": False,
        },
    )


def test_ood_metrics_include_accuracy_counterfactual_and_compensation_gaps() -> None:
    metrics = compute_ood_metrics(
        _iid_records(),
        _ood_records(),
        preregistered_shift=("error_mechanism",),
    )

    assert metrics.iid_accuracy == pytest.approx(2 / 3)
    assert metrics.ood_accuracy == pytest.approx(1 / 3)
    assert metrics.error_mechanism_generalization_gap == pytest.approx(1 / 3)
    assert metrics.counterfactual_consistency_gap == pytest.approx(1 / 3)
    assert metrics.compensation_generalization_gap == pytest.approx(0.5)
    assert metrics.shifted_factors == ("error_mechanism",)


def test_ood_comparison_requires_paired_ids_and_preregistered_factors() -> None:
    with pytest.raises(ValueError, match=r"sample_id|paired"):
        compute_ood_metrics(
            _iid_records(),
            _ood_records()[:-1],
            preregistered_shift=("error_mechanism",),
        )

    with pytest.raises(ValueError, match=r"preregistered|shift"):
        compute_ood_metrics(_iid_records(), _ood_records(), preregistered_shift=())


def test_ood_comparison_rejects_unregistered_confounding_changes() -> None:
    confounded = [dict(row) for row in _ood_records()]
    confounded[0]["scene"] = {"value": 999}

    with pytest.raises(ValueError, match=r"other than|preregistered|scene"):
        compute_ood_metrics(_iid_records(), confounded, preregistered_shift=("error_mechanism",))


def test_ood_comparison_rejects_semantic_split_change_unless_preregistered() -> None:
    iid = [
        {
            **{key: value for key, value in row.items() if key != "error_mechanism"},
            "split_keys": {
                "semantic_split": "iid_test",
                "error_mechanism": row["error_mechanism"],
            },
        }
        for row in _iid_records()
    ]
    ood = [
        {
            **{key: value for key, value in row.items() if key != "error_mechanism"},
            "split_keys": {
                "semantic_split": "ood_test",
                "error_mechanism": row["error_mechanism"],
            },
        }
        for row in _ood_records()
    ]

    with pytest.raises(ValueError, match=r"semantic_split|other than|preregistered"):
        compute_ood_metrics(iid, ood, preregistered_shift=("error_mechanism",))


def test_ood_comparison_rejects_ambiguous_duplicate_factor_locations() -> None:
    iid = [dict(row) for row in _iid_records()]
    ood = [dict(row) for row in _ood_records()]
    iid[0]["split_keys"] = {"error_mechanism": "offset_minus_999"}
    ood[0]["split_keys"] = {"error_mechanism": "offset_minus_2"}

    with pytest.raises(ValueError, match=r"ambiguous|conflict|location"):
        compute_ood_metrics(iid, ood, preregistered_shift=("error_mechanism",))
