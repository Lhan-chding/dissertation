"""Dataset generation contracts for coverage, self-checking, and split hygiene."""

from collections import Counter
from dataclasses import replace

import pytest

from compbias.envs.cva_world.canonical_solver import solve, solve_sample
from compbias.envs.cva_world.corruptions import apply_error, reverse_error
from compbias.envs.cva_world.generator import (
    GeneratorConfig,
    SplitLeakageError,
    audit_splits,
    generate_dataset,
    generate_error_mechanism_counterfactuals,
)
from compbias.envs.cva_world.renderer import (
    SUPPORTED_VISUAL_STYLES,
    is_visual_style_applicable,
)
from compbias.envs.cva_world.schema import SemanticSplit, TaskFamily


def _config(seed: int = 17) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        samples_per_family_per_split=2,
        splits=tuple(SemanticSplit),
        task_families=tuple(TaskFamily),
        visual_styles=SUPPORTED_VISUAL_STYLES,
        train_error_mechanism="offset_plus_2",
        ood_error_mechanism="offset_minus_2",
        preregistered_ood_factors=("visual_style", "error_mechanism"),
        realizations_per_semantic=2,
        fully_cross_iid_visual_styles=True,
    )


def test_generator_covers_five_families_and_five_splits_deterministically() -> None:
    first = generate_dataset(_config())
    second = generate_dataset(_config())

    assert first == second
    expected_per_semantic = {
        family: sum(
            is_visual_style_applicable(style, family) for style in SUPPORTED_VISUAL_STYLES[:-1]
        )
        for family in TaskFamily
    }
    expected = sum(4 * 2 * expected_per_semantic[family] + 2 * 2 for family in TaskFamily)
    assert len(first) == expected
    assert {sample.task_family for sample in first} == set(TaskFamily)
    assert {sample.split_keys.semantic_split for sample in first} == set(SemanticSplit)
    assert len({sample.sample_id for sample in first}) == len(first)


def test_frozen_v2_fully_crossed_sample_and_style_counts_are_exact() -> None:
    config = replace(_config(), samples_per_family_per_split=10)

    assert config.expected_sample_count() == 1_820
    samples = generate_dataset(config)

    assert len(samples) == 1_820
    assert Counter(sample.split_keys.visual_style for sample in samples) == {
        "baseline": 200,
        "font_weight_bold": 120,
        "size_compact": 200,
        "rotation_tilted": 200,
        "contrast_low": 200,
        "background_grid": 200,
        "occlusion_local": 200,
        "blur_mild": 200,
        "distractor_marks": 200,
        "layout_shifted": 100,
    }
    assert Counter(sample.task_family for sample in samples) == {
        TaskFamily.DIGIT_OFFSET: 380,
        TaskFamily.COUNT_TRANSFORM: 340,
        TaskFamily.GAUGE_CALIBRATION: 380,
        TaskFamily.BAR_CHART_AGGREGATE: 380,
        TaskFamily.RELATION_RULE: 340,
    }
    assert Counter(sample.split_keys.semantic_split for sample in samples) == {
        SemanticSplit.TRAIN: 430,
        SemanticSplit.CALIBRATION: 430,
        SemanticSplit.VAL: 430,
        SemanticSplit.IID_TEST: 430,
        SemanticSplit.OOD_TEST: 100,
    }


def test_generator_assigns_only_applicable_canonical_styles_and_held_out_ood_style() -> None:
    samples = generate_dataset(_config())

    assert all(
        is_visual_style_applicable(sample.split_keys.visual_style, sample.task_family)
        for sample in samples
    )
    assert {
        sample.split_keys.visual_style
        for sample in samples
        if sample.split_keys.semantic_split is SemanticSplit.OOD_TEST
    } == {SUPPORTED_VISUAL_STYLES[-1]}
    assert all(
        sample.split_keys.visual_style != SUPPORTED_VISUAL_STYLES[-1]
        for sample in samples
        if sample.split_keys.semantic_split is not SemanticSplit.OOD_TEST
    )


def test_generator_rejects_legacy_or_inapplicable_visual_style_contracts() -> None:
    with pytest.raises(ValueError, match="canonical"):
        replace(_config(), visual_styles=("font_a", "font_b", "rotated"))
    with pytest.raises(ValueError, match="OOD visual style"):
        replace(
            _config(),
            visual_styles=(
                "baseline",
                "layout_shifted",
                "font_weight_bold",
            ),
        )


def test_generator_config_enforces_local_resource_budgets() -> None:
    with pytest.raises(ValueError, match="1 to 1000"):
        replace(_config(), samples_per_family_per_split=1_001)
    with pytest.raises(ValueError, match="2 to 16"):
        replace(_config(), realizations_per_semantic=17)


def test_every_generated_label_is_solver_checked_and_every_error_round_trips() -> None:
    samples = generate_dataset(_config())

    for sample in samples:
        checked = solve_sample(sample)
        assert checked.is_consistent
        assert checked.answer == sample.canonical_answer
        assert sample.error_catalog[0].error_id == "truth"
        for error in sample.error_catalog:
            perceived = apply_error(sample.scene, error)
            assert reverse_error(perceived, error) == sample.scene
            solve(perceived, sample.question, sample.task_family)


def test_relation_questions_use_a_full_fixed_rule_table_and_every_error_is_solvable() -> None:
    samples = generate_dataset(_config())
    relation_samples = tuple(
        sample for sample in samples if sample.task_family is TaskFamily.RELATION_RULE
    )
    tables = {repr(sample.question["rule"]) for sample in relation_samples}

    assert len(tables) == 1
    assert set(relation_samples[0].question["rule"]) == {
        "left_of",
        "right_of",
        "above",
        "below",
        "parallel",
        "intersect",
    }
    for sample in relation_samples:
        for error in sample.error_catalog:
            perceived = apply_error(sample.scene, error)
            solve(perceived, sample.question, sample.task_family)


def test_bar_catalog_includes_reversible_local_to_global_inconsistency() -> None:
    bars = tuple(
        sample
        for sample in generate_dataset(_config())
        if sample.task_family is TaskFamily.BAR_CHART_AGGREGATE
    )
    assert {sample.scene["maximum"] for sample in bars} == {100.0}
    for sample in bars:
        error = next(
            item for item in sample.error_catalog if item.family == "local_to_global_inconsistency"
        )
        perceived = apply_error(sample.scene, error)
        assert perceived["bars"] != sample.scene["bars"]
        assert solve(perceived, sample.question, sample.task_family) == sample.canonical_answer
        assert reverse_error(perceived, error) == sample.scene


def test_bar_chart_semantics_cover_sum_difference_and_ratio_in_every_split() -> None:
    samples = generate_dataset(replace(_config(), samples_per_family_per_split=10))
    bars = tuple(
        sample for sample in samples if sample.task_family is TaskFamily.BAR_CHART_AGGREGATE
    )
    expected_text = {
        "sum": "Sum the first two bar heights.",
        "difference": "Subtract the second bar height from the first.",
        "ratio": "Divide the first bar height by the second.",
    }

    for split in SemanticSplit:
        semantic_questions = {
            sample.sample_id.rsplit("_r", maxsplit=1)[0]: sample.question
            for sample in bars
            if sample.split_keys.semantic_split is split
        }
        assert Counter(question["operation"] for question in semantic_questions.values()) == {
            "sum": 4,
            "difference": 3,
            "ratio": 3,
        }

    for sample in bars:
        scene = sample.to_mapping()["scene"]
        question = sample.to_mapping()["question"]
        values = scene["bars"]
        operation = question["operation"]
        assert question["indices"] == [0, 1]
        assert question["text"] == expected_text[operation]
        if operation == "sum":
            expected = values[0] + values[1]
        elif operation == "difference":
            expected = values[0] - values[1]
        else:
            assert values[1] != 0
            assert values[0] % values[1] == 0
            expected = values[0] / values[1]
        assert sample.canonical_answer == expected
        assert solve(sample.scene, sample.question, sample.task_family) == expected


def test_split_audit_enforces_semantic_style_and_error_mechanism_boundaries() -> None:
    samples = generate_dataset(_config())
    audit = audit_splits(
        samples,
        preregistered_ood_factors=("visual_style", "error_mechanism"),
    )

    assert audit.scene_template_leaks == ()
    assert audit.answer_leaks == ()
    assert audit.visual_style_leaks == ()
    assert audit.error_mechanism_leaks == ()
    assert audit.ood_pair_mismatches == ()
    assert audit.ood_pair_count == len(TaskFamily) * 2 * 2
    assert audit.preregistered_ood_factors == ("visual_style", "error_mechanism")
    assert audit.ood_changed_factors == ("visual_style", "error_mechanism")
    assert audit.is_clean is True

    by_id = {sample.sample_id: sample for sample in samples}
    for ood in (
        sample for sample in samples if sample.split_keys.semantic_split is SemanticSplit.OOD_TEST
    ):
        assert ood.source_id is not None
        source = by_id[ood.source_id]
        assert source.split_keys.semantic_split is SemanticSplit.IID_TEST
        assert ood.task_family == source.task_family
        assert ood.scene == source.scene
        assert ood.question == source.question
        assert ood.canonical_answer == source.canonical_answer
        assert ood.canonical_reasoning == source.canonical_reasoning
        assert ood.split_keys.visual_style != source.split_keys.visual_style
        assert ood.split_keys.error_mechanism != source.split_keys.error_mechanism
        assert ood.error_catalog != source.error_catalog


def test_iid_style_assignment_is_counterbalanced_across_semantic_indices() -> None:
    samples = generate_dataset(replace(_config(), samples_per_family_per_split=12))
    iid = tuple(
        sample for sample in samples if sample.split_keys.semantic_split is SemanticSplit.IID_TEST
    )
    for family in TaskFamily:
        family_samples = tuple(sample for sample in iid if sample.task_family is family)
        applicable_styles = tuple(
            style
            for style in SUPPORTED_VISUAL_STYLES[:-1]
            if is_visual_style_applicable(style, family)
        )
        style_counts = {
            style: sum(sample.split_keys.visual_style == style for sample in family_samples)
            for style in applicable_styles
        }
        assert set(style_counts.values()) == {12}
        assert set(style_counts) == set(applicable_styles)
        assert all(count > 0 for count in style_counts.values())


def test_every_non_ood_semantic_state_has_multiple_visual_realizations() -> None:
    samples = generate_dataset(_config())
    groups: dict[tuple[object, ...], set[str]] = {}
    for sample in samples:
        if sample.split_keys.semantic_split is SemanticSplit.OOD_TEST:
            continue
        key = (
            sample.task_family,
            sample.split_keys.semantic_split,
            repr(sample.scene),
            repr(sample.question),
            repr(sample.canonical_answer),
        )
        groups.setdefault(key, set()).add(sample.split_keys.visual_style)

    assert groups
    assert all(
        styles
        == {
            style
            for style in SUPPORTED_VISUAL_STYLES[:-1]
            if is_visual_style_applicable(style, key[0])
        }
        for key, styles in groups.items()
    )


def test_audit_detects_scene_template_reuse_across_semantic_splits() -> None:
    samples = generate_dataset(_config())
    train = next(
        sample for sample in samples if sample.split_keys.semantic_split is SemanticSplit.TRAIN
    )
    leaked_keys = replace(train.split_keys, semantic_split=SemanticSplit.IID_TEST)
    leaked = replace(train, sample_id="leaked_iid_sample", split_keys=leaked_keys)

    with pytest.raises(SplitLeakageError, match=r"scene.*template"):
        audit_splits(
            (*samples, leaked),
            preregistered_ood_factors=("visual_style", "error_mechanism"),
        )


def test_split_audit_rejects_ood_pair_semantic_drift() -> None:
    samples = generate_dataset(_config())
    ood = next(
        sample for sample in samples if sample.split_keys.semantic_split is SemanticSplit.OOD_TEST
    )
    wrong_source = next(
        sample
        for sample in samples
        if sample.task_family is ood.task_family
        and sample.split_keys.semantic_split is SemanticSplit.IID_TEST
        and sample.sample_id != ood.source_id
    )
    drifted = replace(ood, source_id=wrong_source.sample_id)
    replaced = tuple(drifted if sample is ood else sample for sample in samples)

    with pytest.raises(SplitLeakageError, match="OOD pair"):
        audit_splits(
            replaced,
            preregistered_ood_factors=("visual_style", "error_mechanism"),
        )


def test_split_audit_uses_supplied_preregistered_factors_instead_of_hardcoding() -> None:
    samples = generate_dataset(_config())

    with pytest.raises(SplitLeakageError, match=r"unregistered.*error_mechanism"):
        audit_splits(samples, preregistered_ood_factors=("visual_style",))


@pytest.mark.parametrize(
    "mechanism",
    ("offset_plsu_2", "plus_2", "offset_plus_zero", "offset_plus_0"),
)
def test_generator_rejects_unknown_or_zero_error_mechanisms(mechanism: str) -> None:
    with pytest.raises(ValueError, match="error mechanism"):
        replace(_config(), train_error_mechanism=mechanism)


def test_seed_changes_realizations_but_not_registered_dataset_shape() -> None:
    first = generate_dataset(_config(seed=17))
    second = generate_dataset(_config(seed=18))

    assert first != second
    assert [(x.task_family, x.split_keys.semantic_split) for x in first] == [
        (x.task_family, x.split_keys.semantic_split) for x in second
    ]


def test_paired_error_shift_changes_only_mechanism_and_executable_catalog() -> None:
    source = tuple(
        sample
        for sample in generate_dataset(_config())
        if sample.split_keys.semantic_split is SemanticSplit.IID_TEST
    )
    shifted = generate_error_mechanism_counterfactuals(
        source,
        counterfactual_error_mechanism="offset_minus_2",
    )

    assert tuple(sample.sample_id for sample in shifted) == tuple(
        sample.sample_id for sample in source
    )
    for original, counterfactual in zip(source, shifted, strict=True):
        assert original.scene == counterfactual.scene
        assert original.question == counterfactual.question
        assert original.canonical_answer == counterfactual.canonical_answer
        assert original.image_path == counterfactual.image_path
        assert original.split_keys.visual_style == counterfactual.split_keys.visual_style
        assert original.split_keys.error_mechanism == "offset_plus_2"
        assert counterfactual.split_keys.error_mechanism == "offset_minus_2"
        assert original.error_catalog != counterfactual.error_catalog
