from __future__ import annotations

import pandas as pd
import pytest

from compbias.estimation.compensability import (
    estimate_forked_compensability,
    estimate_selection_compensability,
    estimate_synthetic_compensability,
    merge_compensability_estimates,
)
from compbias.trajectories.records import (
    ForkedContinuationRecord,
    NaturalMediatorRecord,
    SyntheticMediatorRecord,
)


def _natural(
    record_id: str,
    *,
    error_type: str,
    reward: int,
    rollout_id: str,
) -> NaturalMediatorRecord:
    return NaturalMediatorRecord(
        mediator_record_id=record_id,
        sample_id="sample-1",
        checkpoint_id="base",
        interface_id="scene-json-v1",
        rollout_id=rollout_id,
        image_path="artifacts/datasets/images/sample-1.png",
        question="What is the calibrated value?",
        gold_scene={"value": 5},
        natural_mediator_raw='{"value": 7}',
        natural_mediator_parsed={"value": 7},
        parser_confidence=0.99,
        parse_failed=False,
        error_type=error_type,
        task_severity=1.0 if error_type != "truth" else 0.0,
        original_answer="7",
        original_reward=reward,
        rng_seed=3,
        activation_ref=None,
    )


def test_three_compensabilities_remain_separate_and_report_both_gaps() -> None:
    natural = (
        _natural("m1", error_type="offset:+2", reward=1, rollout_id="r1"),
        _natural("m2", error_type="offset:+2", reward=0, rollout_id="r2"),
        _natural("m3", error_type="truth", reward=1, rollout_id="r3"),
    )
    natural_forks = tuple(
        ForkedContinuationRecord(
            mediator_record_id=mediator_id,
            fork_id=f"fork-{index}",
            image_cut_mode="remove_image_context",
            continuation_seed=index,
            answer="5",
            reward=reward,
            replay_fidelity_metadata={"image_replaced_gap": 0.0},
            source_kind="natural",
        )
        for mediator_id, rewards in (("m1", (1, 1)), ("m2", (1, 1)), ("m3", (1, 1)))
        for index, reward in enumerate(rewards)
    )
    synthetic = (
        SyntheticMediatorRecord(
            mediator_record_id="s1",
            sample_id="sample-1",
            checkpoint_id="base",
            interface_id="scene-json-v1",
            target_error_type="offset:+2",
            construction_method="scene_json_edit",
            synthetic_mediator={"value": 7},
            nearest_natural_state_ids=("m1", "m2"),
            transport_signature=(1.0, 0.0),
        ),
    )
    synthetic_forks = tuple(
        ForkedContinuationRecord(
            mediator_record_id="s1",
            fork_id=f"syn-{index}",
            image_cut_mode="remove_image_context",
            continuation_seed=10 + index,
            answer="7",
            reward=0,
            replay_fidelity_metadata={"off_support": False},
            source_kind="synthetic",
        )
        for index in range(2)
    )

    c_sel = estimate_selection_compensability(natural)
    c_fork = estimate_forked_compensability(natural, natural_forks)
    c_syn = estimate_synthetic_compensability(synthetic, synthetic_forks)
    merged = merge_compensability_estimates(c_sel, c_fork, c_syn)
    offset = merged.loc[merged["error_type"] == "offset:+2"].iloc[0]

    assert offset["c_sel"] == 0.5
    assert offset["c_fork"] == 1.0
    assert offset["c_syn"] == 0.0
    assert offset["mediator_gap"] == -0.5
    assert offset["transport_gap"] == -1.0
    assert set(merged.columns).issuperset(
        {"c_sel", "c_fork", "c_syn", "mediator_gap", "transport_gap"}
    )


def test_selection_estimator_rejects_synthetic_records() -> None:
    synthetic_mapping = {
        "source_kind": "synthetic",
        "sample_id": "sample-1",
        "checkpoint_id": "base",
        "interface_id": "scene-json-v1",
        "error_type": "offset:+2",
        "original_reward": 1,
    }
    with pytest.raises(ValueError, match="natural"):
        estimate_selection_compensability((synthetic_mapping,))


def test_records_detach_mutable_payloads_and_export_json_shapes() -> None:
    scene = {"value": [1, 2]}
    record = _natural("m1", error_type="truth", reward=1, rollout_id="r1")
    synthetic = SyntheticMediatorRecord(
        mediator_record_id="s1",
        sample_id="sample-1",
        checkpoint_id="base",
        interface_id="scene-json-v1",
        target_error_type="truth",
        construction_method="identity",
        synthetic_mediator=scene,
        nearest_natural_state_ids=("m1",),
        transport_signature=(0.0,),
    )
    scene["value"].append(99)

    assert synthetic.to_mapping()["synthetic_mediator"] == {"value": [1, 2]}
    assert isinstance(record.to_mapping(), dict)
    assert pd.notna(record.parser_confidence)
