import json

import numpy as np
import pytest
from test_prospective_features import packet, row

from src.prospective_selection.features import LEVELS
from src.prospective_selection.selectors import (
    ACTIONS,
    ALPHA_GRID,
    FrozenSelector,
    choose,
    krr_fit,
    tune_selector,
)


def utilities(n):
    result = np.full((n, 8), 0.2)
    result[:, 0] = 0.1
    result[:, 4] = 0.7
    return result


def test_n_alpha_and_identically_zero_reference_target():
    k = np.array([[1.0, 0.2], [0.2, 1.0]])
    y = np.array([[0.0, 1.0], [0.0, 3.0]])
    mean, beta = krr_fit(k, y, 0.1)
    assert np.allclose((k + 2 * 0.1 * np.eye(2)) @ beta, y - mean)
    assert mean[0] == 0 and np.all(beta[:, 0] == 0)
    with pytest.raises(ValueError):
        krr_fit([[1, 2], [2, 1]], y, 0.1)


def test_practical_tie_and_deterministic_recipe_order_are_not_certification():
    predictions = dict.fromkeys(ACTIONS, 0.0)
    predictions["R1"] = 0.0049
    assert choose(predictions, "R0")["recipe"] == "R0"
    predictions["R1"] = predictions["R2"] = 0.006
    chosen = choose(predictions, "R0")
    assert chosen["recipe"] == "R1" and "NOT_SAFETY" in chosen["status"]


def test_tuning_selected_utility_six_alpha_and_larger_regularization_tie():
    fit = [packet(lineage=f"fit{i}", rows=[row(b=i)]) for i in range(3)]
    tune = [packet(lineage=f"tune{i}", rows=[row(b=i)]) for i in range(2)]
    model = tune_selector(fit, utilities(3), tune, utilities(2))
    data = model.to_dict()
    assert data["alpha"] == 10 and data["best_static"] == "R4"
    assert [s["alpha"] for s in data["tuning_scores"]] == list(ALPHA_GRID)
    assert all(s["selected_action_utility"] == 0.7 for s in data["tuning_scores"])
    assert len(data["training_packets"]) == 5
    with pytest.raises(ValueError, match="Lineage leakage"):
        tune_selector(fit, utilities(3), [packet(lineage="fit0", step=96)], utilities(1))


def test_serialized_frozen_inference_does_not_accept_or_read_future_outcomes(tmp_path):
    fit = [packet(lineage=f"fit{i}", rows=[row(b=i)]) for i in range(3)]
    model = FrozenSelector.fit(fit, utilities(3), alpha=0.1, default_action="R4")
    frozen = FrozenSelector.from_dict(json.loads(json.dumps(model.to_dict())))
    test = packet(lineage="new", rows=[row(b=2)])
    before = frozen.choose_action(test)
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text('{"R0": 1.0, "R4": 0.0}')
    assert before == frozen.choose_action(test)
    with pytest.raises(TypeError):
        frozen.choose_action(outcomes)
    with pytest.raises(TypeError):
        frozen.choose_action(json.loads(outcomes.read_text()))
    changed = frozen.to_dict()
    changed["alpha"] = 1.0
    with pytest.raises(ValueError, match="identity"):
        FrozenSelector.from_dict(changed)
    assert model.selector_id == frozen.selector_id
    assert (
        frozen.choose_action(packet(rows=[], lineage="empty"))["status"] == "FALLBACK_BEST_STATIC"
    )


def test_repair_differences_without_action_rank_changes_have_no_selection_value():
    fit = [packet(lineage=f"fit{i}", rows=[row(b=i)]) for i in range(3)]
    m3 = FrozenSelector.fit(fit, utilities(3), alpha=0.1, default_action="R4")
    fit2 = [packet(LEVELS[2], lineage=f"fit{i}", rows=[row(b=i)]) for i in range(3)]
    m2 = FrozenSelector.fit(fit2, utilities(3), alpha=0.1, default_action="R4")
    for b in range(4):
        assert m3.choose_action(packet(rows=[row(b=b)], lineage="unseen"))["recipe"] == "R4"
        assert (
            m2.choose_action(packet(LEVELS[2], rows=[row(b=b)], lineage="unseen"))["recipe"] == "R4"
        )


def test_leave_lineage_out_keeps_both_anchors_out_of_fit():
    from src.prospective_selection.selectors import leave_lineage_out

    packets = [
        packet(lineage=f"l{i}", step=step, rows=[row(b=i)]) for i in range(3) for step in (32, 96)
    ]
    records = leave_lineage_out(packets, utilities(6), alpha=0.1, default_action="R4")
    assert len(records) == 6
    for record in records:
        assert record["lineage_id"] not in record["fit_lineages"]
        assert len(record["fit_lineages"]) == 2
        assert record["selected_utility"] == 0.7 and record["alpha"] == 0.1
