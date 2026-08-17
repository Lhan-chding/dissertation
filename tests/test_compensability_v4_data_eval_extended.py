"""Extended objective contracts for v4 data, diagnostics, and evaluation helpers.

The assertions in this module are structural or formula-based.  They deliberately do not
encode empirical success-rate gates: measured effects are reported exactly, including zero
and negative effects.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from compensability_v4.data.natural_error_pool import build_natural_error_pool
from compensability_v4.data.ood_generator import (
    generate_constraint_ood_scene,
    generate_style_ood_scene,
)
from compensability_v4.data.splits import DatasetSplit
from compensability_v4.data.v4_generator import generate_observed_world, generate_v4_scenes
from compensability_v4.diagnostics.interface_ladder import (
    CueCondition,
    Interface,
    InterfaceOutcome,
    interface_claim_name,
    revision_decomposition,
    validate_interface_ladder,
)
from compensability_v4.diagnostics.observation_anchor import (
    CandidateLogProbabilities,
    classify_assimilation_profile,
    observation_anchor_metrics,
    validate_prompt_pair,
)
from compensability_v4.eval.answer_source import AnswerSource, classify_answer_source
from compensability_v4.eval.ood import paired_ood_generalization
from compensability_v4.eval.statistics import (
    aggregate_scene_metrics,
    aggregate_stratified_scene_metrics,
    holm_adjust,
    paired_scene_difference,
    scene_clustered_bootstrap_ci,
)
from compensability_v4.eval.support_metrics import (
    summarize_group_reward_variance,
    summarize_policy_support,
)
from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.schemas.scene import RecoveryScene
from compensability_v4.theory.candidate_space import unique_constraint_projection
from compensability_v4.theory.policy_support import (
    expected_informative_groups,
    informative_group_probability,
    mean_informative_group_rate,
)
from compensability_v4.theory.recoverability_hierarchy import (
    RecoverabilityGaps,
    RecoverabilityLevel,
    interface_gap,
    localization_gap,
    search_gap,
)


def _scene(
    scene_id: str = "scene-001",
    *,
    split: DatasetSplit = DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
) -> RecoveryScene:
    return RecoveryScene(
        scene_id=scene_id,
        split=split,
        semantic_scene_id=f"semantic-{scene_id}",
        numeric_table_id=f"numbers-{scene_id}",
        constraint_graph_id=f"graph-{scene_id}",
        truth=(9, 4, 5, 6),
        facts=(
            {"type": "known_value", "index": 1, "value": 4},
            {
                "type": "pair_sum",
                "left_index": 0,
                "right_index": 1,
                "total": 13,
            },
        ),
        resized_height=280,
        resized_width=280,
        image_path=f"images/{scene_id}.png",
    )


def _observation(
    observation_id: str,
    scene_id: str = "scene-001",
    *,
    observed_values: tuple[int, int, int, int] = (8, 4, 5, 6),
    error_index: int = 0,
) -> NaturalObservation:
    return NaturalObservation(
        observation_id=observation_id,
        scene_id=scene_id,
        observed_values=observed_values,
        error_index=error_index,
        stage1_model_hash="a" * 64,
        image_grid_thw=(1, 20, 20),
        visual_token_count=100,
    )


def _candidate(
    condition: str,
    *,
    true_margin: float,
    observed_logp: float = -5.0,
) -> CandidateLogProbabilities:
    return CandidateLogProbabilities(
        condition=condition,
        logp_true=observed_logp + true_margin,
        logp_observed=observed_logp,
        true_rank=1,
        observed_rank=2,
    )


def _ladder_rows(scene_id: str) -> list[InterfaceOutcome]:
    truth = (9, 4, 5, 6)
    observed = (8, 4, 5, 6)
    counterfactual = (7, 4, 5, 6)
    return [
        InterfaceOutcome(
            scene_id=scene_id,
            family="pair_sum",
            interface=interface,
            condition=condition,
            true_world=truth,
            observed_world=observed,
            output_world=observed,
            counterfactual_world=(
                counterfactual if condition is CueCondition.COUNTERFACTUAL_CUE else None
            ),
        )
        for interface in (
            Interface.I0_HARD_TEXT,
            Interface.I3_SAME_CONVERSATION,
            Interface.I4_EXACT_CACHE,
        )
        for condition in CueCondition
    ]


def _replace_ladder_output(
    rows: list[InterfaceOutcome],
    interface: Interface,
    condition: CueCondition,
    output: tuple[int, int, int, int] | None,
) -> None:
    index = next(
        index
        for index, row in enumerate(rows)
        if row.interface is interface and row.condition is condition
    )
    row = rows[index]
    rows[index] = InterfaceOutcome(
        scene_id=row.scene_id,
        family=row.family,
        interface=row.interface,
        condition=row.condition,
        true_world=row.true_world,
        observed_world=row.observed_world,
        output_world=output,
        counterfactual_world=row.counterfactual_world,
    )


def test_natural_error_pool_is_sorted_complete_and_immutable() -> None:
    scenes = [_scene("scene-b"), _scene("scene-a")]
    observations = [
        _observation("obs-z", "scene-b"),
        _observation("obs-a", "scene-a"),
    ]

    result = build_natural_error_pool(iter(scenes), iter(observations))

    assert [example.observation_id for example in result] == ["obs-a", "obs-z"]
    assert result[0].truth == (9, 4, 5, 6)
    assert result[0].observed_values == (8, 4, 5, 6)
    assert result[0].stage1_model_hash == "a" * 64
    with pytest.raises(FrozenInstanceError):
        result[0].error_index = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("scenes", "observations", "error", "message"),
    [
        ([object()], [], TypeError, "RecoveryScene"),
        ([_scene(), _scene()], [], ValueError, "duplicate scene_id"),
        (
            [_scene(split=DatasetSplit.CONFIRM_IID)],
            [],
            ValueError,
            "confirm scenes",
        ),
        ([_scene()], [object()], TypeError, "NaturalObservation"),
        (
            [_scene()],
            [_observation("obs"), _observation("obs")],
            ValueError,
            "duplicate observation_id",
        ),
        (
            [_scene()],
            [_observation("obs", "missing")],
            ValueError,
            "unknown observation scene_id",
        ),
        (
            [_scene()],
            [_observation("obs", observed_values=(9, 4, 5, 6))],
            ValueError,
            "exactly its registered one-position error",
        ),
        (
            [_scene()],
            [_observation("obs", observed_values=(8, 3, 5, 6))],
            ValueError,
            "exactly its registered one-position error",
        ),
        (
            [_scene()],
            [_observation("obs", observed_values=(8, 4, 5, 6), error_index=1)],
            ValueError,
            "exactly its registered one-position error",
        ),
    ],
)
def test_natural_error_pool_rejects_nonpaired_or_contaminated_inputs(
    scenes: list[object],
    observations: list[object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        build_natural_error_pool(scenes, observations)  # type: ignore[arg-type]


def test_style_ood_changes_only_registered_rendering_provenance() -> None:
    source = _scene()
    before = source.to_mapping()

    generated = generate_style_ood_scene(
        source, scene_id="style-001", image_path="images/style-001.png"
    )

    assert source.to_mapping() == before
    expected = dict(before)
    expected.update(
        {
            "scene_id": "style-001",
            "split": DatasetSplit.CONFIRM_STYLE_OOD.value,
            "image_path": "images/style-001.png",
        }
    )
    assert generated.to_mapping() == expected
    assert generated.facts is not source.facts


def test_constraint_ood_replaces_only_constraint_provenance_and_facts() -> None:
    source = _scene()
    facts = [
        {"type": "known_value", "index": 0, "value": 9},
        {"type": "known_value", "index": 3, "value": 6},
    ]
    facts_before = [dict(fact) for fact in facts]

    generated = generate_constraint_ood_scene(
        source,
        scene_id="constraint-001",
        constraint_graph_id="graph-ood-001",
        facts=facts,
        image_path="images/constraint-001.png",
    )
    facts[0]["value"] = 123

    assert source.truth == generated.truth
    assert generated.split is DatasetSplit.CONFIRM_CONSTRAINT_OOD
    assert generated.constraint_graph_id == "graph-ood-001"
    assert generated.to_mapping()["facts"] == facts_before
    assert source.to_mapping()["facts"] != generated.to_mapping()["facts"]


@pytest.mark.parametrize("generator", [generate_style_ood_scene, generate_constraint_ood_scene])
def test_ood_generators_require_recovery_scenes(generator: object) -> None:
    if generator is generate_style_ood_scene:
        with pytest.raises(TypeError, match="RecoveryScene"):
            generate_style_ood_scene(object(), scene_id="x", image_path="x.png")  # type: ignore[arg-type]
    else:
        with pytest.raises(TypeError, match="RecoveryScene"):
            generate_constraint_ood_scene(  # type: ignore[arg-type]
                object(),
                scene_id="x",
                constraint_graph_id="g",
                facts=[],
                image_path="x.png",
            )


def test_constraint_ood_requires_facts_that_support_the_frozen_truth() -> None:
    with pytest.raises(ValueError, match="support the source truth"):
        generate_constraint_ood_scene(
            _scene(),
            scene_id="constraint-bad",
            constraint_graph_id="graph-bad",
            facts=[{"type": "known_value", "index": 0, "value": 8}],
            image_path="images/constraint-bad.png",
        )


def test_v4_generator_is_deterministic_unique_and_exactly_recoverable() -> None:
    first = generate_v4_scenes(
        count=8,
        seed=17,
        split=DatasetSplit.SYMBOLIC_SUPPORT_TRAIN,
        value_domain=(5, 3, 5, 7),
    )
    second = generate_v4_scenes(
        count=8,
        seed=17,
        split=DatasetSplit.SYMBOLIC_SUPPORT_TRAIN.value,  # type: ignore[arg-type]
        value_domain=(7, 5, 3),
    )

    assert first == second
    assert len({scene.scene_id for scene in first}) == len(first)
    assert all((scene.resized_height, scene.resized_width) == (280, 280) for scene in first)
    assert all(scene.split is DatasetSplit.SYMBOLIC_SUPPORT_TRAIN for scene in first)
    for scene in first:
        observed = (next(value for value in (3, 5, 7) if value != scene.truth[0]), *scene.truth[1:])
        assert unique_constraint_projection(observed, scene.facts, (3, 5, 7)) == scene.truth


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_v4_generator_rejects_nonpositive_or_nonintegral_counts(count: object) -> None:
    with pytest.raises(ValueError, match="count must be a positive integer"):
        generate_v4_scenes(count=count, seed=1, split=DatasetSplit.SUPPORT_DEV)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_v4_generator_rejects_nonintegral_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        generate_v4_scenes(count=1, seed=seed, split=DatasetSplit.SUPPORT_DEV)  # type: ignore[arg-type]


@pytest.mark.parametrize("split", ["unknown", None, 7])
def test_v4_generator_rejects_unregistered_splits(split: object) -> None:
    with pytest.raises(ValueError, match="registered by v4"):
        generate_v4_scenes(count=1, seed=1, split=split)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("domain", "error"),
    [
        ((), ValueError),
        ((1,), ValueError),
        ((1, True), ValueError),
        ((1, 2.5), ValueError),
        ((1, "2"), TypeError),
    ],
)
def test_v4_generator_requires_two_distinct_integer_domain_values(
    domain: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        generate_v4_scenes(
            count=1,
            seed=1,
            split=DatasetSplit.SUPPORT_DEV,
            value_domain=domain,  # type: ignore[arg-type]
        )


def test_v4_generator_defensively_checks_its_recoverability_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "compensability_v4.data.v4_generator.unique_constraint_projection",
        lambda *_args, **_kwargs: (0, 0, 0, 0),
    )

    with pytest.raises(AssertionError, match="non-recoverable"):
        generate_v4_scenes(count=1, seed=1, split=DatasetSplit.SUPPORT_DEV)


def test_generate_observed_world_changes_exactly_the_requested_coordinate() -> None:
    scene = _scene()
    original = scene.truth

    for index, replacement in enumerate((1, 2, 3, 4)):
        observed = generate_observed_world(scene, error_index=index, replacement_value=replacement)
        assert observed[index] == replacement
        assert sum(left != right for left, right in zip(observed, original, strict=True)) == 1
    assert scene.truth == original


@pytest.mark.parametrize("error_index", [-1, 4, True, 1.5])
def test_generate_observed_world_rejects_invalid_error_indices(error_index: object) -> None:
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        generate_observed_world(  # type: ignore[arg-type]
            _scene(), error_index=error_index, replacement_value=1
        )


@pytest.mark.parametrize("replacement", [True, 1.5, "8"])
def test_generate_observed_world_requires_integer_replacement(replacement: object) -> None:
    with pytest.raises(TypeError, match="replacement_value must be an integer"):
        generate_observed_world(  # type: ignore[arg-type]
            _scene(), error_index=0, replacement_value=replacement
        )


def test_generate_observed_world_requires_a_scene_and_an_actual_error() -> None:
    with pytest.raises(TypeError, match="RecoveryScene"):
        generate_observed_world(object(), error_index=0, replacement_value=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actual error"):
        generate_observed_world(_scene(), error_index=0, replacement_value=9)


def test_recoverability_hierarchy_and_gaps_are_named_differences_not_gates() -> None:
    assert [level.value for level in RecoverabilityLevel] == ["R0", "R1", "R2", "R3", "R4"]
    assert search_gap(0.25, 0.75) == pytest.approx(-0.5)
    assert localization_gap(0.4, 0.4) == 0.0
    assert interface_gap(0.9, 0.2) == pytest.approx(0.7)

    gaps = RecoverabilityGaps.from_accuracies(
        t3=0.3,
        t4_given_index=0.8,
        t5=0.6,
        t6=0.1,
        cache=0.7,
        hard_text=0.4,
    )
    assert gaps.search == pytest.approx(0.5)
    assert gaps.localization == pytest.approx(0.5)
    assert gaps.interface == pytest.approx(0.3)
    with pytest.raises(FrozenInstanceError):
        gaps.search = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("invalid", [True, "0.5", math.nan, math.inf, -0.01, 1.01])
def test_recoverability_gaps_reject_non_rates(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        search_gap(invalid, 0.5)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        localization_gap(0.5, invalid)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        interface_gap(invalid, 0.5)  # type: ignore[arg-type]


def test_candidate_log_probabilities_and_anchor_metrics_are_exact_and_immutable() -> None:
    no_cue = _candidate("no_cue", true_margin=-2.0)
    valid = _candidate("valid_cue", true_margin=-0.5)

    metrics = observation_anchor_metrics(no_cue, valid)

    assert no_cue.margin == pytest.approx(-2.0)
    assert metrics.no_cue_margin == pytest.approx(-2.0)
    assert metrics.valid_cue_margin == pytest.approx(-0.5)
    assert metrics.delta_f == pytest.approx(1.5)
    assert metrics.m_f == pytest.approx(-0.5)
    with pytest.raises(FrozenInstanceError):
        metrics.delta_f = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"condition": "unknown"},
        {"logp_true": math.nan},
        {"logp_observed": math.inf},
        {"true_rank": 0},
        {"observed_rank": -1},
    ],
)
def test_candidate_log_probabilities_validate_registered_finite_inputs(
    kwargs: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "condition": "no_cue",
        "logp_true": -1.0,
        "logp_observed": -2.0,
        "true_rank": 1,
        "observed_rank": 2,
    }
    payload.update(kwargs)
    with pytest.raises(ValueError):
        CandidateLogProbabilities(**payload)  # type: ignore[arg-type]


def test_observation_anchor_requires_no_cue_valid_cue_order() -> None:
    with pytest.raises(ValueError, match="no-cue/valid-cue"):
        observation_anchor_metrics(
            _candidate("valid_cue", true_margin=1.0),
            _candidate("no_cue", true_margin=1.0),
        )


def test_prompt_pair_accepts_only_condition_and_fact_differences() -> None:
    no_cue = {
        "scene_id": "scene-001",
        "condition": "no_cue",
        "facts": [],
        "question": "recover",
    }
    valid = {
        "scene_id": "scene-001",
        "condition": "valid_cue",
        "facts": [{"type": "known_value", "index": 0, "value": 9}],
        "question": "recover",
    }

    assert validate_prompt_pair(no_cue, valid) is None


@pytest.mark.parametrize(
    ("no_cue", "valid", "message"),
    [
        (
            {"condition": "no_cue", "facts": []},
            {"condition": "valid_cue", "facts": [], "extra": 1},
            "fields differ",
        ),
        (
            {"condition": "no_cue", "facts": []},
            {"condition": "no_cue", "facts": []},
            "must differ only",
        ),
        (
            {"condition": "no_cue", "facts": [], "question": "a"},
            {"condition": "valid_cue", "facts": [], "question": "b"},
            "must differ only",
        ),
        (
            {"condition": "sham_cue", "facts": []},
            {"condition": "valid_cue", "facts": [1]},
            "first paired payload",
        ),
        (
            {"condition": "no_cue", "facts": []},
            {"condition": "sham_cue", "facts": [1]},
            "second paired payload",
        ),
    ],
)
def test_prompt_pair_rejects_unpaired_payloads(
    no_cue: dict[str, object], valid: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_prompt_pair(no_cue, valid)


@pytest.mark.parametrize(
    ("no_cue", "valid", "expected"),
    [
        ((-1.0, -1.0), (-1.0, -1.0), "no_assimilation"),
        ((-1.0, -1.0), (0.0, -1.0), "transient_assimilation"),
        ((-1.0, -1.0), (-0.5, -0.25), "persistent_but_insufficient_assimilation"),
        ((-1.0, -1.0), (-0.5, 0.0), "successful_revision"),
    ],
)
def test_assimilation_profiles_use_layerwise_signs_without_empirical_thresholds(
    no_cue: tuple[float, ...], valid: tuple[float, ...], expected: str
) -> None:
    assert classify_assimilation_profile(no_cue, valid) == expected


def test_assimilation_profile_uses_only_explicit_numerical_tolerance() -> None:
    assert (
        classify_assimilation_profile((-1.0,), (-0.999999999,), numerical_tolerance=1e-8)
        == "no_assimilation"
    )
    assert (
        classify_assimilation_profile((-1.0,), (0.0,), numerical_tolerance=0.0)
        == "successful_revision"
    )


@pytest.mark.parametrize(
    ("no_cue", "valid", "tolerance"),
    [
        ((), (), 1e-8),
        ((0.0,), (0.0, 1.0), 1e-8),
        ((math.nan,), (0.0,), 1e-8),
        ((0.0,), (math.inf,), 1e-8),
        ((0.0,), (0.0,), -1.0),
        ((0.0,), (0.0,), math.inf),
    ],
)
def test_assimilation_profile_rejects_unpaired_nonfinite_inputs(
    no_cue: tuple[float, ...], valid: tuple[float, ...], tolerance: float
) -> None:
    with pytest.raises(ValueError):
        classify_assimilation_profile(no_cue, valid, numerical_tolerance=tolerance)


def test_interface_outcome_properties_distinguish_recovery_copy_and_counterfactual() -> None:
    truth = (9, 4, 5, 6)
    observed = (8, 4, 5, 6)
    recovered = InterfaceOutcome(
        "scene",
        "family",
        Interface.I4_EXACT_CACHE,
        CueCondition.VALID_CUE,
        truth,
        observed,
        truth,
    )
    counterfactual = InterfaceOutcome(
        "scene",
        "family",
        Interface.I4_EXACT_CACHE,
        CueCondition.COUNTERFACTUAL_CUE,
        truth,
        observed,
        (7, 4, 5, 6),
        (7, 4, 5, 6),
    )

    assert recovered.exact_world_recovery is True
    assert recovered.observation_copy is False
    assert recovered.counterfactual_compliance is None
    assert counterfactual.exact_world_recovery is False
    assert counterfactual.observation_copy is False
    assert counterfactual.counterfactual_compliance is True
    with pytest.raises(FrozenInstanceError):
        recovered.output_world = observed  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"scene_id": ""},
        {"family": ""},
        {"true_world": (1, 2, 3)},
        {"observed_world": (1, 2, 3, True)},
        {"output_world": (1, 2, 3)},
        {"counterfactual_world": (1, 2, 3, 4.0)},
        {"condition": CueCondition.COUNTERFACTUAL_CUE},
    ],
)
def test_interface_outcome_validates_world_shape_and_counterfactual_target(
    overrides: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "scene_id": "scene",
        "family": "family",
        "interface": Interface.I0_HARD_TEXT,
        "condition": CueCondition.NO_CUE,
        "true_world": (9, 4, 5, 6),
        "observed_world": (8, 4, 5, 6),
        "output_world": None,
        "counterfactual_world": None,
    }
    payload.update(overrides)
    with pytest.raises(ValueError):
        InterfaceOutcome(**payload)  # type: ignore[arg-type]


def test_interface_ladder_requires_unique_complete_primary_cells() -> None:
    rows = _ladder_rows("scene")
    assert validate_interface_ladder(iter(rows)) == tuple(rows)

    with pytest.raises(ValueError, match="duplicate"):
        validate_interface_ladder([*rows, rows[0]])
    with pytest.raises(ValueError, match=r"incomplete.*scene"):
        validate_interface_ladder(rows[:-1])


def test_revision_decomposition_uses_scene_cells_and_reports_signed_effects() -> None:
    rows_a = _ladder_rows("scene-a")
    rows_b = _ladder_rows("scene-b")
    truth = (9, 4, 5, 6)
    counterfactual = (7, 4, 5, 6)
    _replace_ladder_output(rows_a, Interface.I4_EXACT_CACHE, CueCondition.NO_CUE, truth)
    _replace_ladder_output(rows_a, Interface.I4_EXACT_CACHE, CueCondition.VALID_CUE, truth)
    _replace_ladder_output(
        rows_a,
        Interface.I4_EXACT_CACHE,
        CueCondition.COUNTERFACTUAL_CUE,
        counterfactual,
    )
    _replace_ladder_output(rows_b, Interface.I0_HARD_TEXT, CueCondition.NO_CUE, truth)

    summary = revision_decomposition([*rows_b, *rows_a])

    assert summary.i0_no_cue_accuracy == pytest.approx(0.5)
    assert summary.i4_no_cue_accuracy == pytest.approx(0.5)
    assert summary.i4_valid_cue_accuracy == pytest.approx(0.5)
    assert summary.spontaneous_visual_revision == pytest.approx(0.0)
    assert summary.fact_conditioned_revision == pytest.approx(0.0)
    assert summary.counterfactual_compliance == pytest.approx(0.5)


def test_revision_decomposition_rejects_an_empty_empirical_sample() -> None:
    with pytest.raises(ValueError, match="at least one scene"):
        validate_interface_ladder([])
    with pytest.raises(ValueError, match="at least one scene"):
        revision_decomposition([])


@pytest.mark.parametrize(
    ("interface", "claim"),
    [
        (Interface.I0_HARD_TEXT, "symbolic_downstream_recovery"),
        (Interface.I3_SAME_CONVERSATION, "natural_visual_revision"),
        (Interface.I4_EXACT_CACHE, "natural_visual_revision"),
        (Interface.I1_SOFT_REPORT, "intervention_diagnostic"),
        (Interface.I2_CANDIDATE_WORLD, "intervention_diagnostic"),
    ],
)
def test_interface_claim_names_preserve_mechanism_boundaries(
    interface: Interface, claim: str
) -> None:
    assert interface_claim_name(interface) == claim


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"answer_correct": True, "world_recovered": True}, AnswerSource.GENUINE_RECOVERY),
        (
            {"answer_correct": True, "world_recovered": False, "operator_invariant": True},
            AnswerSource.OPERATOR_INVARIANCE,
        ),
        (
            {"answer_correct": True, "world_recovered": False, "error_cancelled": True},
            AnswerSource.ERROR_CANCELLATION,
        ),
        (
            {
                "answer_correct": True,
                "world_recovered": False,
                "visual_reread_evidence": True,
            },
            AnswerSource.VISUAL_REREAD,
        ),
        (
            {
                "answer_correct": True,
                "world_recovered": False,
                "guess_or_prior_evidence": True,
            },
            AnswerSource.GUESS_OR_ANSWER_PRIOR,
        ),
        ({"answer_correct": False, "world_recovered": True}, AnswerSource.UNRESOLVED),
        ({"answer_correct": True, "world_recovered": False}, AnswerSource.UNRESOLVED),
        (
            {
                "answer_correct": True,
                "world_recovered": False,
                "operator_invariant": True,
                "error_cancelled": True,
            },
            AnswerSource.UNRESOLVED,
        ),
    ],
)
def test_answer_source_classification_is_conservative(
    kwargs: dict[str, bool], expected: AnswerSource
) -> None:
    assert classify_answer_source(**kwargs) is expected


def test_genuine_recovery_takes_precedence_over_answer_only_evidence() -> None:
    assert (
        classify_answer_source(
            answer_correct=True,
            world_recovered=True,
            operator_invariant=True,
            error_cancelled=True,
        )
        is AnswerSource.GENUINE_RECOVERY
    )


@pytest.mark.parametrize("field", ["answer_correct", "world_recovered", "operator_invariant"])
def test_answer_source_flags_must_be_boolean(field: str) -> None:
    kwargs: dict[str, object] = {"answer_correct": True, "world_recovered": False}
    kwargs[field] = 1
    with pytest.raises(TypeError, match="must be boolean"):
        classify_answer_source(**kwargs)  # type: ignore[arg-type]


def test_paired_ood_generalization_reports_exact_signed_gap_without_threshold() -> None:
    iid = {"scene-b": False, "scene-a": True, "scene-c": True}
    ood = {"scene-a": False, "scene-b": True, "scene-c": True}

    result = paired_ood_generalization(iid, ood)

    assert result.number_of_pairs == 3
    assert result.iid_accuracy == pytest.approx(2 / 3)
    assert result.ood_accuracy == pytest.approx(2 / 3)
    assert result.generalization_gap == pytest.approx(0.0)
    with pytest.raises(FrozenInstanceError):
        result.generalization_gap = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("iid", "ood", "error"),
    [
        ([], {}, TypeError),
        ({}, [], TypeError),
        ({}, {}, ValueError),
        ({"a": True}, {"b": True}, ValueError),
        ({"a": 1}, {"a": True}, TypeError),
        ({"a": True}, {"a": "yes"}, TypeError),
    ],
)
def test_paired_ood_generalization_requires_boolean_scene_pairs(
    iid: object, ood: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        paired_ood_generalization(iid, ood)  # type: ignore[arg-type]


def test_policy_support_summary_is_scene_balanced_and_formula_based() -> None:
    rows = [
        {"scene_id": "scene-b", "success": False},
        {"scene_id": "scene-a", "success": True},
        {"scene_id": "scene-a", "success": True},
        {"scene_id": "scene-a", "success": False},
    ]

    summary = summarize_policy_support(iter(rows), group_size=2)

    assert summary.group_size == 2
    assert summary.number_of_scenes == 2
    assert [item.scene_id for item in summary.by_scene] == ["scene-a", "scene-b"]
    assert summary.by_scene[0].rollout_count == 3
    assert summary.by_scene[0].success_count == 2
    assert summary.by_scene[0].success_probability == pytest.approx(2 / 3)
    assert summary.by_scene[0].pass_at_k == pytest.approx(8 / 9)
    assert summary.by_scene[0].informative_group_probability == pytest.approx(4 / 9)
    assert summary.mean_success_probability == pytest.approx(1 / 3)
    assert summary.mean_pass_at_k == pytest.approx(4 / 9)
    assert summary.informative_group_rate == pytest.approx(2 / 9)
    with pytest.raises(FrozenInstanceError):
        summary.number_of_scenes = 3  # type: ignore[misc]


@pytest.mark.parametrize("group_size", [0, -1, True, 2.0])
def test_policy_support_summary_requires_positive_integral_group_size(group_size: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        summarize_policy_support(
            [{"scene_id": "scene", "success": True}],
            group_size=group_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([], ValueError),
        ([object()], TypeError),
        ([{"scene_id": "", "success": True}], ValueError),
        ([{"scene_id": 1, "success": True}], ValueError),
        ([{"scene_id": "scene", "success": 1}], TypeError),
    ],
)
def test_policy_support_summary_validates_rows(rows: list[object], error: type[Exception]) -> None:
    with pytest.raises(error):
        summarize_policy_support(rows, group_size=2)  # type: ignore[arg-type]


def test_reward_variance_summary_reports_group_and_scene_macro_statistics() -> None:
    rows = [
        {"scene_id": "scene-b", "group_id": "one", "reward": 1},
        {"scene_id": "scene-a", "group_id": "mixed", "reward": 0.0},
        {"scene_id": "scene-a", "group_id": "mixed", "reward": 1.0},
        {"scene_id": "scene-a", "group_id": "zero", "reward": 0.0},
        {"scene_id": "scene-a", "group_id": "zero", "reward": 0.0},
    ]

    summary = summarize_group_reward_variance(iter(rows))

    assert summary.number_of_scenes == 2
    assert summary.number_of_groups == 3
    assert [(group.scene_id, group.group_id) for group in summary.groups] == [
        ("scene-a", "mixed"),
        ("scene-a", "zero"),
        ("scene-b", "one"),
    ]
    assert summary.groups[0].rollout_count == 2
    assert summary.groups[0].mean_reward == pytest.approx(0.5)
    assert summary.groups[0].reward_variance == pytest.approx(0.25)
    assert summary.mean_scene_reward_variance == pytest.approx(0.0625)
    assert summary.all_zero_group_rate == pytest.approx(1 / 3)
    assert summary.all_one_group_rate == pytest.approx(1 / 3)
    assert summary.non_degenerate_group_rate == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([], ValueError),
        ([object()], TypeError),
        ([{"scene_id": "", "group_id": "g", "reward": 0}], ValueError),
        ([{"scene_id": 1, "group_id": "g", "reward": 0}], ValueError),
        ([{"scene_id": "s", "group_id": "", "reward": 0}], ValueError),
        ([{"scene_id": "s", "group_id": 1, "reward": 0}], ValueError),
        ([{"scene_id": "s", "group_id": "g", "reward": True}], TypeError),
        ([{"scene_id": "s", "group_id": "g", "reward": "1"}], TypeError),
        ([{"scene_id": "s", "group_id": "g", "reward": -0.1}], ValueError),
        ([{"scene_id": "s", "group_id": "g", "reward": 1.1}], ValueError),
        ([{"scene_id": "s", "group_id": "g", "reward": math.nan}], ValueError),
    ],
)
def test_reward_variance_summary_validates_rows(rows: list[object], error: type[Exception]) -> None:
    with pytest.raises(error):
        summarize_group_reward_variance(rows)  # type: ignore[arg-type]


def test_scene_metric_aggregation_is_scene_balanced_sorted_and_immutable() -> None:
    rows = [
        {"scene_id": "scene-b", "rollout_id": 0, "score": False},
        {"scene_id": "scene-a", "rollout_id": 0, "score": 1.0},
        {"scene_id": "scene-a", "rollout_id": 1, "score": True},
    ]

    aggregate = aggregate_scene_metrics(rows, metric="score")

    assert aggregate.number_of_scenes == 2
    assert aggregate.number_of_rollouts == 3
    assert aggregate.scene_values == (("scene-a", 1.0), ("scene-b", 0.0))
    assert aggregate.point_estimate == pytest.approx(0.5)
    with pytest.raises(FrozenInstanceError):
        aggregate.point_estimate = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("rows", "metric", "error"),
    [
        ([{"scene_id": "s", "score": 1}], "", ValueError),
        ([object()], "score", TypeError),
        ([{"scene_id": "", "score": 1}], "score", ValueError),
        ([{"scene_id": 1, "score": 1}], "score", ValueError),
        ([{"scene_id": "s"}], "score", ValueError),
        (
            [
                {"scene_id": "s", "rollout_id": 1, "score": 1},
                {"scene_id": "s", "rollout_id": 1, "score": 0},
            ],
            "score",
            ValueError,
        ),
        ([{"scene_id": "s", "rollout_id": [], "score": 1}], "score", TypeError),
        ([{"scene_id": "s", "score": "1"}], "score", TypeError),
        ([{"scene_id": "s", "score": math.nan}], "score", ValueError),
        ([], "score", ValueError),
    ],
)
def test_scene_metric_aggregation_validates_metric_rows(
    rows: list[object], metric: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        aggregate_scene_metrics(rows, metric=metric)  # type: ignore[arg-type]


def test_stratified_scene_metrics_are_sorted_and_preserve_scene_unit() -> None:
    rows = [
        {"scene_id": "b", "family": "z", "rollout_id": 0, "score": 0},
        {"scene_id": "a", "family": "a", "rollout_id": 0, "score": 1},
        {"scene_id": "a", "family": "a", "rollout_id": 1, "score": 0},
    ]

    result = aggregate_stratified_scene_metrics(rows, metric="score")

    assert result.stratum_field == "family"
    assert [name for name, _aggregate in result.strata] == ["a", "z"]
    assert result.strata[0][1].point_estimate == pytest.approx(0.5)
    assert result.strata[0][1].number_of_scenes == 1


@pytest.mark.parametrize(
    ("rows", "field", "error"),
    [
        ([{"scene_id": "s", "family": "f", "score": 1}], "", ValueError),
        ([object()], "family", TypeError),
        ([{"scene_id": "", "family": "f", "score": 1}], "family", ValueError),
        ([{"scene_id": "s", "family": "", "score": 1}], "family", ValueError),
        ([{"scene_id": "s", "family": 1, "score": 1}], "family", ValueError),
        (
            [
                {"scene_id": "s", "family": "a", "score": 1},
                {"scene_id": "s", "family": "b", "score": 0},
            ],
            "family",
            ValueError,
        ),
        ([], "family", ValueError),
    ],
)
def test_stratified_scene_metrics_validate_strata(
    rows: list[object], field: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        aggregate_stratified_scene_metrics(  # type: ignore[arg-type]
            rows, metric="score", stratum_field=field
        )


def test_scene_clustered_bootstrap_is_seeded_and_resamples_scenes() -> None:
    rows = [
        {"scene_id": "a", "rollout_id": 0, "score": 0.0},
        {"scene_id": "a", "rollout_id": 1, "score": 0.0},
        {"scene_id": "b", "rollout_id": 0, "score": 1.0},
    ]

    first = scene_clustered_bootstrap_ci(
        rows, metric="score", confidence=0.8, n_resamples=101, seed=7
    )
    second = scene_clustered_bootstrap_ci(
        rows, metric="score", confidence=0.8, n_resamples=101, seed=7
    )

    assert first == second
    assert first.estimate == pytest.approx(0.5)
    assert first.number_of_scenes == 2
    assert first.low in {0.0, 0.5}
    assert first.high in {0.5, 1.0}
    assert first.low <= first.estimate <= first.high


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"confidence": True}, TypeError),
        ({"confidence": 0.0}, ValueError),
        ({"confidence": 1.0}, ValueError),
        ({"n_resamples": True}, ValueError),
        ({"n_resamples": 0}, ValueError),
        ({"seed": True}, TypeError),
        ({"seed": 1.5}, TypeError),
    ],
)
def test_scene_clustered_bootstrap_validates_control_parameters(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        scene_clustered_bootstrap_ci(  # type: ignore[arg-type]
            [{"scene_id": "s", "score": 1}], metric="score", **kwargs
        )


def test_paired_scene_difference_is_sorted_exact_and_nonmutating() -> None:
    before = {"b": 0.5, "a": False}
    after = {"a": True, "b": 0.25}

    assert paired_scene_difference(before, after) == (("a", 1.0), ("b", -0.25))
    assert before == {"b": 0.5, "a": False}
    assert after == {"a": True, "b": 0.25}


@pytest.mark.parametrize(
    ("before", "after", "error"),
    [
        ([], {}, TypeError),
        ({}, [], TypeError),
        ({}, {}, ValueError),
        ({"a": 1.0}, {"b": 1.0}, ValueError),
        ({"a": "bad"}, {"a": 1.0}, TypeError),
        ({"a": 1.0}, {"a": math.inf}, ValueError),
    ],
)
def test_paired_scene_difference_validates_pairing_and_values(
    before: object, after: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        paired_scene_difference(before, after)  # type: ignore[arg-type]


def test_holm_adjust_is_order_preserving_tie_stable_and_monotone() -> None:
    p_values = {"late": 0.04, "first_tie": 0.01, "second_tie": 0.01, "large": 0.9}

    adjusted = holm_adjust(p_values)

    assert list(adjusted) == list(p_values)
    assert adjusted == {
        "late": pytest.approx(0.08),
        "first_tie": pytest.approx(0.04),
        "second_tie": pytest.approx(0.04),
        "large": pytest.approx(0.9),
    }
    assert holm_adjust({}) == {}


@pytest.mark.parametrize(
    ("p_values", "error"),
    [
        ([], TypeError),
        ({"": 0.5}, ValueError),
        ({1: 0.5}, ValueError),
        ({"h": -0.1}, ValueError),
        ({"h": 1.1}, ValueError),
        ({"h": math.nan}, ValueError),
        ({"h": "0.5"}, TypeError),
    ],
)
def test_holm_adjust_validates_named_finite_probabilities(
    p_values: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        holm_adjust(p_values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("p", "k", "expected"),
    [
        (0.0, 8, 0.0),
        (1.0, 8, 0.0),
        (0.5, 1, 0.0),
        (0.5, 2, 0.5),
        (0.25, 4, 1.0 - 0.25**4 - 0.75**4),
    ],
)
def test_informative_group_probability_matches_closed_form(
    p: float, k: int, expected: float
) -> None:
    assert informative_group_probability(p, k) == pytest.approx(expected)


def test_mean_and_expected_informative_groups_accept_one_shot_iterables() -> None:
    probabilities = (value for value in (0.0, 0.5, 1.0))
    mean = mean_informative_group_rate(probabilities, 2)
    expected = expected_informative_groups((value for value in (0.0, 0.5, 1.0)), 2, 10)

    assert mean == pytest.approx(1 / 6)
    assert expected == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("p", "k", "error"),
    [
        (True, 2, TypeError),
        ("0.5", 2, TypeError),
        (math.nan, 2, ValueError),
        (-0.1, 2, ValueError),
        (1.1, 2, ValueError),
        (0.5, True, TypeError),
        (0.5, 2.0, TypeError),
        (0.5, 0, ValueError),
    ],
)
def test_informative_group_probability_validates_probability_and_group_size(
    p: object, k: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        informative_group_probability(p, k)  # type: ignore[arg-type]


def test_policy_support_aggregates_reject_empty_or_invalid_sequences() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean_informative_group_rate([], 2)
    with pytest.raises(ValueError, match="must not be empty"):
        expected_informative_groups([], 2, 1)
    with pytest.raises(ValueError):
        mean_informative_group_rate([0.5, 1.1], 2)
    with pytest.raises(ValueError):
        expected_informative_groups([0.5], 0, 1)


@pytest.mark.parametrize(
    ("groups_per_scene", "error"),
    [(True, TypeError), (1.5, TypeError), (-1, ValueError)],
)
def test_expected_informative_groups_validates_group_count(
    groups_per_scene: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        expected_informative_groups(  # type: ignore[arg-type]
            [0.5], 2, groups_per_scene
        )


def test_expected_informative_groups_allows_zero_requested_groups() -> None:
    assert expected_informative_groups([0.25, 0.75], 4, 0) == 0.0
