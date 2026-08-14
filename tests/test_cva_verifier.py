"""Strict, family-aware answer-verification contracts for CVA-World."""

from dataclasses import FrozenInstanceError, replace

import pytest

from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
from compbias.envs.cva_world.schema import CVASample, SemanticSplit, TaskFamily
from compbias.envs.cva_world.verifier import (
    AnswerVerification,
    MalformedOutputError,
    VerificationBatch,
    verify_answer,
    verify_many,
)
from compbias.models.structured_parser import ParseResult, ParseStatus, parse_trajectory


def _samples() -> tuple[CVASample, ...]:
    return generate_dataset(
        GeneratorConfig(
            seed=23,
            samples_per_family_per_split=1,
            splits=(SemanticSplit.TRAIN,),
        )
    )


def _parsed(sample: CVASample, answer: object) -> ParseResult:
    return ParseResult(
        status=ParseStatus.OK,
        sample_id=sample.sample_id,
        raw_text="fixture",
        perceived_scene={},
        reasoning_action={},
        answer=answer,
    )


def test_each_task_family_accepts_its_exact_canonical_answer() -> None:
    results = tuple(verify_answer(sample, sample.canonical_answer) for sample in _samples())

    assert {result.task_family for result in results} == set(TaskFamily)
    assert all(result.is_correct for result in results)
    assert all(result.outcome_reward == 1.0 for result in results)
    assert all(result.accuracy == 1.0 for result in results)


def test_family_aware_verification_returns_binary_reward_without_coercion() -> None:
    samples = {sample.task_family: sample for sample in _samples()}

    numeric = verify_answer(samples[TaskFamily.DIGIT_OFFSET], -999)
    relation = verify_answer(samples[TaskFamily.RELATION_RULE], "wrong_class")

    assert numeric.is_correct is False
    assert numeric.outcome_reward == 0.0
    assert numeric.accuracy == 0.0
    assert relation.is_correct is False
    assert relation.outcome_reward == 0.0


def test_successful_structured_parse_is_verified_and_sample_id_must_match() -> None:
    sample = _samples()[0]

    result = verify_answer(sample, _parsed(sample, sample.canonical_answer))

    assert result.is_correct is True
    with pytest.raises(MalformedOutputError, match="sample_id"):
        verify_answer(
            sample,
            replace(_parsed(sample, sample.canonical_answer), sample_id="different_sample"),
        )


@pytest.mark.parametrize(
    "raw_text",
    [
        "not structured",
        "<perception>{}</perception><reasoning>{}</reasoning><answer>oops</answer>",
        "<perception>{}</perception><reasoning>{}</reasoning>",
    ],
)
def test_malformed_structured_outputs_fail_explicitly_instead_of_scoring_zero(
    raw_text: str,
) -> None:
    sample = _samples()[0]
    parsed = parse_trajectory(raw_text, sample_id=sample.sample_id)
    assert parsed.status is not ParseStatus.OK

    with pytest.raises(MalformedOutputError, match=parsed.status.value):
        verify_answer(sample, parsed)


@pytest.mark.parametrize(
    ("family", "malformed"),
    [
        (TaskFamily.DIGIT_OFFSET, True),
        (TaskFamily.COUNT_TRANSFORM, "10"),
        (TaskFamily.GAUGE_CALIBRATION, float("nan")),
        pytest.param(
            TaskFamily.GAUGE_CALIBRATION,
            10**10_000,
            id="gauge-overflowing-real",
        ),
        (TaskFamily.BAR_CHART_AGGREGATE, [10]),
        (TaskFamily.RELATION_RULE, 10),
        (TaskFamily.RELATION_RULE, ""),
    ],
)
def test_task_family_contract_rejects_malformed_answer_types(
    family: TaskFamily, malformed: object
) -> None:
    sample = next(item for item in _samples() if item.task_family is family)

    with pytest.raises(MalformedOutputError, match=family.value):
        verify_answer(sample, malformed)


def test_stale_stored_canonical_answer_is_rejected_by_the_solver_before_scoring() -> None:
    sample = _samples()[0]
    stale = replace(sample, canonical_answer=-123)

    with pytest.raises(ValueError, match="canonical_answer"):
        verify_answer(stale, -123)


def test_public_verification_record_rejects_inconsistent_correctness() -> None:
    sample = _samples()[0]

    with pytest.raises(ValueError, match="is_correct"):
        AnswerVerification(
            sample_id=sample.sample_id,
            task_family=sample.task_family,
            predicted_answer=-1,
            canonical_answer=sample.canonical_answer,
            is_correct=True,
            outcome_reward=1.0,
        )


def test_batch_verification_is_ordered_immutable_and_reports_accuracy() -> None:
    samples = _samples()

    def answer_for(sample: CVASample, index: int) -> object:
        if index < 3:
            return sample.canonical_answer
        if sample.task_family is TaskFamily.RELATION_RULE:
            return "incorrect"
        return float(sample.canonical_answer) + 1.0  # type: ignore[arg-type]

    outputs = tuple(
        _parsed(sample, answer_for(sample, index)) for index, sample in enumerate(samples)
    )

    batch = verify_many(samples, outputs)

    assert isinstance(batch, VerificationBatch)
    assert tuple(result.sample_id for result in batch.results) == tuple(
        sample.sample_id for sample in samples
    )
    assert batch.total == len(samples)
    assert batch.correct == 3
    assert batch.accuracy == pytest.approx(3 / len(samples))
    assert batch.mean_outcome_reward == pytest.approx(batch.accuracy)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        batch.correct = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        batch.results[0].outcome_reward = 0.0  # type: ignore[misc]


def test_batch_rejects_empty_misaligned_or_duplicate_inputs_before_scoring() -> None:
    samples = _samples()

    with pytest.raises(ValueError, match="empty"):
        verify_many((), ())
    with pytest.raises(ValueError, match="same length"):
        verify_many(samples, (samples[0].canonical_answer,))
    with pytest.raises(ValueError, match="duplicate sample_id"):
        verify_many(
            (samples[0], samples[0]),
            (samples[0].canonical_answer, samples[0].canonical_answer),
        )


def test_verifier_requires_schema_records_in_single_and_batch_modes() -> None:
    sample = _samples()[0]

    with pytest.raises(TypeError, match="CVASample"):
        verify_answer(sample.to_mapping(), sample.canonical_answer)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CVASample"):
        verify_many((sample.to_mapping(),), (sample.canonical_answer,))  # type: ignore[arg-type]
