"""Interventional rollouts hide images and retain a complete causal audit trail."""

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from compbias.envs.cva_world.schema import CVASample
from compbias.interventions.state_injection import InterventionRecord, run_state_injection
from compbias.models.structured_parser import ParseStatus, parse_trajectory


def _sample() -> CVASample:
    return CVASample.from_mapping(
        {
            "sample_id": "digit_001",
            "image_path": "images/digit_001.png",
            "task_family": "digit_offset",
            "scene": {"value": 7},
            "question": {"template": "add_constant", "operand": 3},
            "canonical_answer": 10,
            "canonical_reasoning": {"operation": "add", "operand": 3},
            "error_catalog": [
                {"error_id": "truth", "family": "truth", "severity": 0, "parameters": {}},
                {
                    "error_id": "numeric_offset:+2",
                    "family": "numeric_offset",
                    "severity": 2,
                    "parameters": {"field": "value", "delta": 2},
                },
            ],
            "split_keys": {
                "semantic_split": "calibration",
                "visual_style": "font_a",
                "error_mechanism": "standard",
            },
        }
    )


class SpyReasoner:
    """Its signature intentionally offers no image argument."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], dict[str, object], int]] = []

    def generate(
        self,
        perceived_state: dict[str, object],
        question: dict[str, object],
        *,
        seed: int,
    ) -> str:
        self.calls.append((dict(perceived_state), dict(question), seed))
        value = int(perceived_state["value"])
        operand = 10 - value
        return (
            f'<perception>{{"value": {value}}}</perception>'
            f'<reasoning>{{"operation": "add", "operand": {operand}}}</reasoning>'
            "<answer>10</answer>"
        )


def _verify(parsed: object, canonical_answer: object) -> float:
    return float(getattr(parsed, "answer", None) == canonical_answer)


def test_batch_intervention_hides_image_and_logs_every_error_seed_pair() -> None:
    reasoner = SpyReasoner()

    records = run_state_injection(
        (_sample(),),
        reasoner=reasoner,
        parser=parse_trajectory,
        verifier=_verify,
        model_id="reasoner_a",
        checkpoint="step_100",
        rollout_seeds=(3, 5),
        image=None,
    )

    assert isinstance(records, tuple)
    assert len(records) == 4
    assert all(isinstance(record, InterventionRecord) for record in records)
    assert [(call[0]["value"], call[2]) for call in reasoner.calls] == [
        (7, 3),
        (7, 5),
        (9, 3),
        (9, 5),
    ]
    assert {(row.sample_id, row.error_id) for row in records} == {
        ("digit_001", "truth"),
        ("digit_001", "numeric_offset:+2"),
    }
    assert {row.model_id for row in records} == {"reasoner_a"}
    assert {row.checkpoint for row in records} == {"step_100"}
    assert {row.rollout_seed for row in records} == {3, 5}
    assert all(row.parsed["status"] == ParseStatus.OK.value for row in records)
    assert all(row.reward == 1.0 for row in records)
    serialized = records[0].to_mapping()
    assert json.loads(json.dumps(serialized)) == serialized
    assert serialized["parsed"]["status"] == "ok"
    with pytest.raises(FrozenInstanceError):
        records[0].reward = 0.0  # type: ignore[misc]


def test_non_none_image_is_rejected_before_reasoner_execution() -> None:
    reasoner = SpyReasoner()

    with pytest.raises((AssertionError, ValueError), match=r"image.*None"):
        run_state_injection(
            (_sample(),),
            reasoner=reasoner,
            parser=parse_trajectory,
            verifier=_verify,
            model_id="reasoner_a",
            checkpoint="step_100",
            rollout_seeds=(3,),
            image=object(),
        )

    assert reasoner.calls == []


class SometimesMalformedReasoner(SpyReasoner):
    def generate(
        self,
        perceived_state: dict[str, object],
        question: dict[str, object],
        *,
        seed: int,
    ) -> str:
        if seed == 5:
            self.calls.append((dict(perceived_state), dict(question), seed))
            return "unparseable"
        return super().generate(perceived_state, question, seed=seed)


def test_parse_failures_remain_in_interventional_records_with_zero_reward() -> None:
    records = run_state_injection(
        (_sample(),),
        reasoner=SometimesMalformedReasoner(),
        parser=parse_trajectory,
        verifier=_verify,
        model_id="reasoner_a",
        checkpoint="step_100",
        rollout_seeds=(3, 5),
        image=None,
    )

    assert len(records) == 4
    failures = [row for row in records if row.parsed["status"] != ParseStatus.OK.value]
    assert len(failures) == 2
    assert all(row.raw_output == "unparseable" for row in failures)
    assert all(row.reward == 0.0 for row in failures)


def test_state_injection_rejects_non_string_generation_before_parsing() -> None:
    class InvalidReasoner(SpyReasoner):
        def generate(self, perceived_state, question, *, seed):
            return {"not": "text"}

    with pytest.raises(TypeError, match="raw output must be a string"):
        run_state_injection(
            (_sample(),),
            reasoner=InvalidReasoner(),
            parser=parse_trajectory,
            verifier=_verify,
            model_id="reasoner_a",
            checkpoint="step_100",
            rollout_seeds=(3,),
        )


@pytest.mark.parametrize("reward", [-0.1, 0.5, 1.1, float("nan")])
def test_state_injection_requires_binary_outcome_rewards(reward: float) -> None:
    with pytest.raises(ValueError, match="binary"):
        run_state_injection(
            (_sample(),),
            reasoner=SpyReasoner(),
            parser=parse_trajectory,
            verifier=lambda _parsed, _answer: reward,
            model_id="reasoner_a",
            checkpoint="step_100",
            rollout_seeds=(3,),
        )


def test_intervention_record_public_constructor_validates_scalar_contract() -> None:
    values = {
        "sample_id": "sample",
        "error_id": "truth",
        "error_family": "truth",
        "severity": 0.0,
        "error_parameters": {},
        "model_id": "model",
        "checkpoint": "step",
        "rollout_seed": 1,
        "perceived_state": {"value": 7},
        "question": {"operand": 3},
        "canonical_answer": 10,
        "raw_output": "valid",
        "parsed": SimpleNamespace(status=ParseStatus.OK),
        "reward": 1.0,
    }

    with pytest.raises(ValueError, match="severity"):
        InterventionRecord(**{**values, "severity": -1.0})
    with pytest.raises(ValueError, match="binary"):
        InterventionRecord(**{**values, "reward": 0.5})
    with pytest.raises(TypeError, match="raw_output"):
        InterventionRecord(**{**values, "raw_output": 3})


def test_intervention_record_detaches_mutable_parsed_payload() -> None:
    parsed = {"status": "ok", "answer": {"value": 10}}
    record = InterventionRecord(
        sample_id="sample",
        error_id="truth",
        error_family="truth",
        severity=0.0,
        error_parameters={},
        model_id="model",
        checkpoint="step",
        rollout_seed=1,
        perceived_state={"value": 7},
        question={"operand": 3},
        canonical_answer=10,
        raw_output="valid",
        parsed=parsed,
        reward=1.0,
    )
    parsed["answer"]["value"] = 99

    assert record.to_mapping()["parsed"]["answer"]["value"] == 10


def test_intervention_record_snapshots_custom_parser_mapping_once() -> None:
    class MutableParsed:
        def __init__(self) -> None:
            self.payload = {"status": "ok", "answer": {"value": 10}}

        def to_mapping(self) -> dict[str, object]:
            return self.payload

    parsed = MutableParsed()
    record = InterventionRecord(
        sample_id="sample",
        error_id="truth",
        error_family="truth",
        severity=0.0,
        error_parameters={},
        model_id="model",
        checkpoint="step",
        rollout_seed=1,
        perceived_state={"value": 7},
        question={"operand": 3},
        canonical_answer=10,
        raw_output="valid",
        parsed=parsed,
        reward=1.0,
    )
    parsed.payload["answer"]["value"] = 99

    assert record.to_mapping()["parsed"]["answer"]["value"] == 10


def test_intervention_record_serializes_unordered_parser_sets_deterministically() -> None:
    record = InterventionRecord(
        sample_id="sample",
        error_id="truth",
        error_family="truth",
        severity=0.0,
        error_parameters={},
        model_id="model",
        checkpoint="step",
        rollout_seed=1,
        perceived_state={"value": 7},
        question={"operand": 3},
        canonical_answer=10,
        raw_output="valid",
        parsed={"status": "ok", "labels": {"zeta", "alpha"}},
        reward=1.0,
    )

    first = record.to_mapping()
    second = record.to_mapping()
    assert first["parsed"]["labels"] == ["alpha", "zeta"]
    assert json.loads(json.dumps(first)) == first == second


def test_parser_internal_type_error_is_not_retried_or_hidden() -> None:
    calls = 0

    def broken_parser(raw_output: str, *, sample_id: str) -> object:
        nonlocal calls
        calls += 1
        raise TypeError(f"internal failure for sample_id={sample_id}")

    with pytest.raises(TypeError, match="internal failure"):
        run_state_injection(
            (_sample(),),
            reasoner=SpyReasoner(),
            parser=broken_parser,
            verifier=_verify,
            model_id="reasoner_a",
            checkpoint="step_100",
            rollout_seeds=(3,),
        )

    assert calls == 1


def test_one_argument_parser_is_called_without_sample_id_keyword() -> None:
    calls = 0

    def one_argument_parser(raw_output: str) -> object:
        nonlocal calls
        calls += 1
        return parse_trajectory(raw_output, sample_id="digit_001")

    records = run_state_injection(
        (_sample(),),
        reasoner=SpyReasoner(),
        parser=one_argument_parser,
        verifier=_verify,
        model_id="reasoner_a",
        checkpoint="step_100",
        rollout_seeds=(3,),
    )

    assert len(records) == 2
    assert calls == 2
