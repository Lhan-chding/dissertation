from __future__ import annotations

from compensability.study_c3.statistics import holm_adjust, paired_factorial_effects


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index in range(4):
        for condition in ("collision", "separating"):
            scene_id = f"p{pair_index}-{condition}"
            for verifier in ("answer", "state"):
                for validity in ("binary", "lex"):
                    for decoder in ("free", "constrained"):
                        value = (
                            0.1
                            + 0.05 * (verifier == "state")
                            + 0.08 * (validity == "lex")
                            + 0.2 * (decoder == "constrained")
                            + 0.03
                            * (verifier == "state")
                            * (validity == "lex")
                        )
                        rows.append(
                            {
                                "pair_id": f"p{pair_index}",
                                "scene_id": scene_id,
                                "condition": condition,
                                "family": "trend" if pair_index % 2 else "cross_series",
                                "verifier": verifier,
                                "validity_channel": validity,
                                "eval_decoder": decoder,
                                "outcome": value,
                            }
                        )
    return rows


def test_registered_factorial_effects_are_paired_and_bootstrapped_by_pair() -> None:
    result = paired_factorial_effects(
        _rows(), outcome="outcome", resamples=200, seed=7
    )
    assert result["pair_count"] == 4
    assert result["effects"]["verifier"]["estimate"] == 0.065
    assert result["effects"]["validity_channel"]["estimate"] == 0.095
    assert result["effects"]["verifier_x_validity"]["estimate"] == 0.03
    assert result["effects"]["decoder"]["estimate"] == 0.2
    assert result["effects"]["verifier_x_decoder"]["estimate"] == 0.0
    assert all(len(effect["bootstrap_95_ci"]) == 2 for effect in result["effects"].values())


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}

