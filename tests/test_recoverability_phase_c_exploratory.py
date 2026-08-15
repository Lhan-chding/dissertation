from __future__ import annotations

import json
from pathlib import Path

import pytest

from compbias.recoverability.operators import Operation
from compbias.recoverability.phase_c_exploratory import (
    ExploratoryArm,
    ExploratoryScene,
    build_arm_messages,
    build_cue_bundle,
    evaluate_arm_output,
    summarize_exploratory_phase_c,
)
from compbias.recoverability.phase_c_exploratory_config import (
    load_exploratory_phase_c_config,
)
from compbias.recoverability.phase_c_exploratory_data import (
    build_exploratory_phase_c_records,
)
from compbias.recoverability.phase_n_result import load_phase_n_frozen_result


ROOT = Path(__file__).resolve().parents[1]


def test_phase_n_server_result_is_frozen_as_inconclusive() -> None:
    result = load_phase_n_frozen_result(
        ROOT / "configs/recoverability/phase_n_frozen_result.yaml"
    )

    assert result.status == "FINAL_INCONCLUSIVE_PHASE_N_DO_NOT_RERUN"
    assert result.phase_n_exit == 3
    assert result.h1_supported is False
    assert result.inconclusive is True
    assert result.operator_sensitive_errors == 836
    assert result.strict_natural_repair_candidates == 33
    assert result.primary_rate == 33 / 836
    assert result.one_sided_cp_upper == 0.05242826275410656
    assert result.reason_code == "phase_n_h1_upper_not_below_threshold"
    assert result.allow_sample_extension is False
    assert result.training_invoked is False


def test_exploratory_config_cannot_authorize_confirmatory_claims_or_training(
    tmp_path: Path,
) -> None:
    path = ROOT / "configs/recoverability/phase_c_exploratory_v1.yaml"
    config = load_exploratory_phase_c_config(path)

    assert config.intake_scenes == 3000
    assert dict(config.selected_family_quotas) == {
        "cross_series": 60,
        "duplicate_encoding": 60,
        "trend": 60,
    }
    assert config.forks_per_arm == 1
    assert config.hypothesis_tested is False
    assert config.confirmatory_execution_authorized is False
    assert config.rl_authorized is False
    assert config.training_authorized is False

    payload = path.read_text(encoding="utf-8").replace(
        "training_authorized: false", "training_authorized: true"
    )
    altered = tmp_path / "altered.yaml"
    altered.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="exploratory contract"):
        load_exploratory_phase_c_config(altered)


def test_exploratory_dataset_is_deterministic_balanced_and_disjoint() -> None:
    config = load_exploratory_phase_c_config(
        ROOT / "configs/recoverability/phase_c_exploratory_v1.yaml"
    )
    reserved = {(2, 3, 4, 5), (5, 4, 3, 2)}

    first = build_exploratory_phase_c_records(config, reserved_numeric_tables=reserved)
    second = build_exploratory_phase_c_records(config, reserved_numeric_tables=reserved)

    assert first == second
    assert len(first) == 3000
    assert not ({item.values for item in first} & reserved)
    assert len({item.values for item in first}) == 3000
    cells = {
        (family, chart, operation): sum(
            item.family == family
            and item.chart_type == chart
            and item.operation == operation
            for item in first
        )
        for family in ("cross_series", "duplicate_encoding", "trend")
        for chart in ("grouped_bar", "line")
        for operation in ("difference", "max_minus_min", "sum")
    }
    assert max(cells.values()) - min(cells.values()) <= 1


@pytest.mark.parametrize("family", ["cross_series", "duplicate_encoding", "trend"])
def test_valid_and_counterfactual_cues_are_compatible_without_gold_in_prompt(
    family: str,
) -> None:
    truth = (8, 4, 6, 2) if family != "trend" else (4, 6, 8, 10)
    observed = (7, truth[1], truth[2], truth[3])
    bundle = build_cue_bundle(
        truth=truth,
        observed=observed,
        family=family,
        operation=Operation.DIFFERENCE,
    )
    scene = ExploratoryScene(
        scene_id=f"scene_{family}",
        family=family,
        chart_type="line",
        operation=Operation.DIFFERENCE,
        truth=truth,
        observed=observed,
        cues=bundle,
    )

    assert bundle.valid_answer == 4 if family == "trend" else truth[0] - truth[1]
    assert bundle.counterfactual_answer != bundle.valid_answer
    for arm in ExploratoryArm:
        messages = build_arm_messages(scene, arm)
        encoded = json.dumps(messages, sort_keys=True)
        assert "gold" not in encoded
        assert "cue_condition" not in encoded
        assert arm.value not in encoded


def _program(values: tuple[int, int, int, int], operation: Operation) -> str:
    a, b, c, d = values
    if operation is Operation.DIFFERENCE:
        steps = [{"op": "subtract", "inputs": ["a", "b"], "output": "result"}]
    elif operation is Operation.SUM:
        steps = [{"op": "add", "inputs": ["a", "b"], "output": "result"}]
    else:
        steps = [
            {"op": "max", "inputs": ["a", "b", "c", "d"], "output": "high"},
            {"op": "min", "inputs": ["a", "b", "c", "d"], "output": "low"},
            {"op": "subtract", "inputs": ["high", "low"], "output": "result"},
        ]
    return json.dumps(
        {
            "variables": {"a": a, "b": b, "c": c, "d": d},
            "steps": steps,
            "return": "result",
        },
        separators=(",", ":"),
    )


def test_arm_evaluation_requires_trusted_cue_dataflow() -> None:
    truth = (8, 4, 6, 2)
    observed = (7, 4, 6, 2)
    scene = ExploratoryScene(
        scene_id="scene_cross",
        family="cross_series",
        chart_type="grouped_bar",
        operation=Operation.DIFFERENCE,
        truth=truth,
        observed=observed,
        cues=build_cue_bundle(
            truth=truth,
            observed=observed,
            family="cross_series",
            operation=Operation.DIFFERENCE,
        ),
    )

    repaired = evaluate_arm_output(scene, ExploratoryArm.VALID, _program(truth, scene.operation))
    bypass = evaluate_arm_output(
        scene,
        ExploratoryArm.VALID,
        _program(observed, scene.operation),
    )
    ablated = evaluate_arm_output(
        scene,
        ExploratoryArm.ABLATED,
        _program(observed, scene.operation),
    )

    assert repaired.faithful_success is True
    assert repaired.executed_result == scene.cues.valid_answer
    assert repaired.consumed_cue is True
    assert bypass.faithful_success is False
    assert bypass.consumed_cue is False
    assert ablated.faithful_success is False


def test_exploratory_summary_never_authorizes_rl_or_training() -> None:
    summary = summarize_exploratory_phase_c(
        scene_count=180,
        arm_records=1080,
        parse_successes=1080,
        execution_successes=1080,
        faithful_by_arm={arm: 180 for arm in ExploratoryArm},
    )

    assert summary.exploratory_only is True
    assert summary.hypothesis_tested is False
    assert summary.confirmatory_execution_authorized is False
    assert summary.rl_authorized is False
    assert summary.training_authorized is False
    assert summary.training_invoked is False
