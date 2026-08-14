"""Pure deterministic renderers must expose scene evidence without answer leakage."""

import hashlib
from dataclasses import replace

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageStat

from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
from compbias.envs.cva_world.renderer import (
    SUPPORTED_VISUAL_STYLES,
    VISUAL_STYLE_APPLICABILITY,
    RenderConfig,
    build_contact_sheet,
    is_visual_style_applicable,
    render_sample,
    render_scene,
    sample_render_coordinates,
    validate_visual_style,
)
from compbias.envs.cva_world.schema import CVASample, SemanticSplit, TaskFamily

EXPECTED_VISUAL_STYLES = (
    "baseline",
    "font_weight_bold",
    "size_compact",
    "rotation_tilted",
    "contrast_low",
    "background_grid",
    "occlusion_local",
    "blur_mild",
    "distractor_marks",
    "layout_shifted",
)


def test_visual_style_catalog_is_closed_truthful_and_validated() -> None:
    assert SUPPORTED_VISUAL_STYLES == EXPECTED_VISUAL_STYLES
    assert tuple(validate_visual_style(style) for style in SUPPORTED_VISUAL_STYLES) == (
        SUPPORTED_VISUAL_STYLES
    )

    with pytest.raises(ValueError, match="unsupported visual style"):
        RenderConfig(style="font_typo")
    with pytest.raises(TypeError, match="visual style must be a string"):
        validate_visual_style(1)  # type: ignore[arg-type]


def test_visual_style_applicability_is_closed_and_font_specific() -> None:
    text_bearing_families = (
        TaskFamily.DIGIT_OFFSET,
        TaskFamily.GAUGE_CALIBRATION,
        TaskFamily.BAR_CHART_AGGREGATE,
    )

    assert tuple(VISUAL_STYLE_APPLICABILITY) == SUPPORTED_VISUAL_STYLES
    assert VISUAL_STYLE_APPLICABILITY["font_weight_bold"] == text_bearing_families
    assert is_visual_style_applicable("font_weight_bold", TaskFamily.DIGIT_OFFSET)
    assert not is_visual_style_applicable("font_weight_bold", TaskFamily.COUNT_TRANSFORM)
    assert all(
        VISUAL_STYLE_APPLICABILITY[style] == tuple(TaskFamily)
        for style in SUPPORTED_VISUAL_STYLES
        if style != "font_weight_bold"
    )
    with pytest.raises(TypeError):
        VISUAL_STYLE_APPLICABILITY["font_weight_bold"] = tuple(  # type: ignore[index]
            TaskFamily
        )
    with pytest.raises(ValueError, match="unsupported task_family"):
        is_visual_style_applicable("baseline", "typo")


@pytest.mark.parametrize(
    ("legacy_style", "canonical_style"),
    [
        ("font_a", "baseline"),
        ("font_b", "font_weight_bold"),
        ("rotated", "rotation_tilted"),
    ],
)
def test_legacy_style_aliases_render_the_truthful_canonical_factor(
    legacy_style: str,
    canonical_style: str,
) -> None:
    scene = {"value": 7}
    legacy = render_scene(
        scene,
        TaskFamily.DIGIT_OFFSET,
        RenderConfig(style=legacy_style, seed=19, realization_index=2),
    )
    canonical = render_scene(
        scene,
        TaskFamily.DIGIT_OFFSET,
        RenderConfig(style=canonical_style, seed=19, realization_index=2),
    )

    assert legacy.tobytes() == canonical.tobytes()


@pytest.mark.parametrize(
    ("family", "scene"),
    [
        (TaskFamily.DIGIT_OFFSET, {"value": 7}),
        (TaskFamily.COUNT_TRANSFORM, {"count": 5, "shape": "circle"}),
        (TaskFamily.GAUGE_CALIBRATION, {"reading": 2.5, "minimum": 0, "maximum": 10}),
        (TaskFamily.BAR_CHART_AGGREGATE, {"bars": [2, 5, 3]}),
        (TaskFamily.RELATION_RULE, {"relation": "left_of"}),
    ],
)
def test_all_task_families_render_nonblank_rgb_images(
    family: TaskFamily, scene: dict[str, object]
) -> None:
    image = render_scene(
        scene,
        family,
        RenderConfig(width=160, height=120, style="font_a", seed=11),
    )

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (160, 120)
    assert ImageStat.Stat(image.convert("L")).var[0] > 0


def test_rendering_is_byte_deterministic_for_same_scene_style_and_seed() -> None:
    config = RenderConfig(width=128, height=96, style="font_a", seed=22)
    first = render_scene({"value": 7}, TaskFamily.DIGIT_OFFSET, config)
    second = render_scene({"value": 7}, TaskFamily.DIGIT_OFFSET, config)

    assert hashlib.sha256(first.tobytes()).digest() == hashlib.sha256(second.tobytes()).digest()


def test_render_coordinates_are_semantic_stable_and_realization_explicit() -> None:
    first_seed, first_realization = sample_render_coordinates(
        "digit_offset_train_000007_r00", base_seed=23
    )
    second_seed, second_realization = sample_render_coordinates(
        "digit_offset_train_000007_r08", base_seed=23
    )

    assert first_seed == second_seed
    assert (first_realization, second_realization) == (0, 8)
    with pytest.raises(ValueError, match="realization suffix"):
        sample_render_coordinates("digit_offset_train_000007", base_seed=23)


def test_contact_sheet_builder_is_deterministic_and_labels_all_images() -> None:
    rendered = (
        ("sample_r00", Image.new("RGB", (64, 64), "red")),
        ("sample_r01", Image.new("RGB", (64, 64), "blue")),
    )

    first = build_contact_sheet(rendered)
    second = build_contact_sheet(rendered)

    assert first.size == (5 * 272, 292)
    assert first.tobytes() == second.tobytes()


def _sample(operand: int) -> CVASample:
    return CVASample.from_mapping(
        {
            "sample_id": f"digit_operand_{operand}",
            "image_path": f"images/digit_operand_{operand}.png",
            "task_family": "digit_offset",
            "scene": {"value": 7},
            "question": {"template": "add_constant", "operand": operand},
            "canonical_answer": 7 + operand,
            "canonical_reasoning": {"operation": "add", "operand": operand},
            "error_catalog": [
                {"error_id": "truth", "family": "truth", "severity": 0, "parameters": {}}
            ],
            "split_keys": {
                "semantic_split": "train",
                "visual_style": "font_a",
                "error_mechanism": "standard",
            },
        }
    )


def test_render_sample_depends_only_on_scene_and_registered_visual_style() -> None:
    config = RenderConfig(width=128, height=96, style="font_a", seed=22)

    image_add_three = render_sample(_sample(3), config)
    image_add_nine = render_sample(_sample(9), config)

    # Question operand and canonical answer must never be painted into the image.
    assert ImageChops.difference(image_add_three, image_add_nine).getbbox() is None


def test_rendering_returns_fresh_images_and_does_not_mutate_config() -> None:
    config = RenderConfig(width=128, height=96, style="font_a", seed=22)
    first = render_scene({"value": 7}, TaskFamily.DIGIT_OFFSET, config)
    second = render_scene({"value": 7}, TaskFamily.DIGIT_OFFSET, config)

    assert first is not second
    with pytest.raises(ValueError):
        replace(config, width=0)
    with pytest.raises(ValueError, match="4096"):
        replace(config, height=4097)


def test_bar_chart_uses_absolute_scale_and_does_not_collapse_distinct_scenes() -> None:
    """Different absolute bar values must never become the same visual problem."""

    config = RenderConfig(width=256, height=192, style="font_a", seed=17)
    first = render_scene(
        {"bars": [41, 45, 46], "minimum": 0, "maximum": 100},
        TaskFamily.BAR_CHART_AGGREGATE,
        config,
    )
    second = render_scene(
        {"bars": [42, 46, 47], "minimum": 0, "maximum": 100},
        TaskFamily.BAR_CHART_AGGREGATE,
        config,
    )

    assert first.tobytes() != second.tobytes()


def test_renderer_supports_every_registered_bar_question_without_answer_leakage() -> None:
    samples = generate_dataset(
        GeneratorConfig(
            seed=17,
            samples_per_family_per_split=10,
            splits=(SemanticSplit.TRAIN,),
            task_families=(TaskFamily.BAR_CHART_AGGREGATE,),
            visual_styles=SUPPORTED_VISUAL_STYLES,
            train_error_mechanism="offset_plus_2",
            ood_error_mechanism="offset_minus_2",
            preregistered_ood_factors=("visual_style", "error_mechanism"),
            realizations_per_semantic=2,
            fully_cross_iid_visual_styles=True,
        )
    )
    by_operation = {
        str(sample.question["operation"]): sample
        for sample in samples
        if sample.split_keys.visual_style == "baseline"
    }

    assert set(by_operation) == {"sum", "difference", "ratio"}
    for sample in by_operation.values():
        image = render_sample(sample, RenderConfig(width=256, height=192, seed=9))
        assert image.size == (256, 192)


def test_rotated_realizations_are_deterministically_distinct() -> None:
    scene = {"value": 7}
    first = render_scene(
        scene,
        TaskFamily.DIGIT_OFFSET,
        RenderConfig(width=128, height=128, style="rotated", seed=17, realization_index=0),
    )
    second = render_scene(
        scene,
        TaskFamily.DIGIT_OFFSET,
        RenderConfig(width=128, height=128, style="rotated", seed=17, realization_index=1),
    )

    assert first.tobytes() != second.tobytes()


@pytest.mark.parametrize(
    ("family", "scene", "expected_labels"),
    [
        (
            TaskFamily.GAUGE_CALIBRATION,
            {"reading": 41.25, "minimum": 0, "maximum": 100},
            {"0", "100", "41.25"},
        ),
        (
            TaskFamily.BAR_CHART_AGGREGATE,
            {"bars": [41, 45, 46], "minimum": 0, "maximum": 100},
            {"0", "100", "41", "45", "46"},
        ),
    ],
)
def test_quantitative_renderers_label_the_scene_scale_and_observations(
    monkeypatch: pytest.MonkeyPatch,
    family: TaskFamily,
    scene: dict[str, object],
    expected_labels: set[str],
) -> None:
    """The pixels must carry absolute scene evidence, not just relative geometry."""

    labels: list[str] = []
    original_text = ImageDraw.ImageDraw.text

    def recording_text(
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        labels.append(str(text))
        original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", recording_text)
    render_scene(scene, family, RenderConfig(width=256, height=192, seed=9))

    assert expected_labels <= set(labels)


@pytest.mark.parametrize(
    ("family", "scene"),
    [
        (TaskFamily.DIGIT_OFFSET, {"value": 7}),
        (TaskFamily.COUNT_TRANSFORM, {"count": 5, "shape": "circle"}),
        (TaskFamily.GAUGE_CALIBRATION, {"reading": 2.5, "minimum": 0, "maximum": 10}),
        (TaskFamily.BAR_CHART_AGGREGATE, {"bars": [2, 5, 3]}),
        (TaskFamily.RELATION_RULE, {"relation": "left_of"}),
    ],
)
def test_every_named_visual_factor_has_a_distinct_pixel_realization(
    family: TaskFamily,
    scene: dict[str, object],
) -> None:
    hashes = {
        style: hashlib.sha256(
            render_scene(
                scene,
                family,
                RenderConfig(width=192, height=144, style=style, seed=31),
            ).tobytes()
        ).hexdigest()
        for style in SUPPORTED_VISUAL_STYLES
    }

    applicable_hashes = {
        style: pixel_hash
        for style, pixel_hash in hashes.items()
        if is_visual_style_applicable(style, family)
    }
    assert len(set(applicable_hashes.values())) == len(applicable_hashes), hashes
    if family not in VISUAL_STYLE_APPLICABILITY["font_weight_bold"]:
        assert hashes["font_weight_bold"] == hashes["baseline"]


@pytest.mark.parametrize("style", SUPPORTED_VISUAL_STYLES)
def test_visual_factors_preserve_scene_evidence_without_answer_leakage(style: str) -> None:
    config = RenderConfig(width=192, height=144, style=style, seed=41, realization_index=1)
    seven = render_scene({"value": 7}, TaskFamily.DIGIT_OFFSET, config)
    eight = render_scene({"value": 8}, TaskFamily.DIGIT_OFFSET, config)
    add_three = render_sample(_sample(3), config)
    add_nine = render_sample(_sample(9), config)

    assert seven.tobytes() != eight.tobytes()
    assert ImageStat.Stat(seven.convert("L")).var[0] > 2
    assert ImageChops.difference(add_three, add_nine).getbbox() is None


@pytest.mark.parametrize(
    ("family", "scene"),
    [
        (TaskFamily.DIGIT_OFFSET, {"value": 7}),
        (TaskFamily.GAUGE_CALIBRATION, {"reading": 2.5, "minimum": 0, "maximum": 10}),
        (TaskFamily.BAR_CHART_AGGREGATE, {"bars": [2, 5, 3]}),
    ],
)
def test_font_weight_bold_changes_weight_without_system_font_search(
    monkeypatch: pytest.MonkeyPatch,
    family: TaskFamily,
    scene: dict[str, object],
) -> None:
    stroke_widths: list[int] = []
    font_names: list[tuple[str, str]] = []
    original_text = ImageDraw.ImageDraw.text

    def recording_text(
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        stroke_widths.append(int(kwargs.get("stroke_width", 0)))
        font = kwargs.get("font")
        if hasattr(font, "getname"):
            font_names.append(font.getname())
        original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", recording_text)
    render_scene(
        scene,
        family,
        RenderConfig(style="font_weight_bold"),
    )

    assert stroke_widths
    assert all(width > 0 for width in stroke_widths)
    assert font_names
    assert all(name == ("Aileron", "Regular") for name in font_names)


def test_size_compact_reduces_the_content_footprint() -> None:
    scene = {"value": 7}
    baseline = render_scene(
        scene,
        TaskFamily.DIGIT_OFFSET,
        RenderConfig(width=192, height=144, style="baseline"),
    )
    compact = render_scene(
        scene,
        TaskFamily.DIGIT_OFFSET,
        RenderConfig(width=192, height=144, style="size_compact"),
    )

    baseline_background = Image.new("RGB", baseline.size, baseline.getpixel((0, 0)))
    compact_background = Image.new("RGB", compact.size, compact.getpixel((0, 0)))
    baseline_box = ImageChops.difference(baseline, baseline_background).getbbox()
    compact_box = ImageChops.difference(compact, compact_background).getbbox()

    assert baseline_box is not None
    assert compact_box is not None
    assert (compact_box[2] - compact_box[0]) < (baseline_box[2] - baseline_box[0])
    assert (compact_box[3] - compact_box[1]) < (baseline_box[3] - baseline_box[1])


def test_low_contrast_reduces_luminance_range_but_remains_readable() -> None:
    scene = {"value": 7}
    baseline = render_scene(
        scene,
        TaskFamily.DIGIT_OFFSET,
        RenderConfig(width=192, height=144, style="baseline"),
    ).convert("L")
    low_contrast = render_scene(
        scene,
        TaskFamily.DIGIT_OFFSET,
        RenderConfig(width=192, height=144, style="contrast_low"),
    ).convert("L")
    baseline_range = baseline.getextrema()[1] - baseline.getextrema()[0]
    low_contrast_range = low_contrast.getextrema()[1] - low_contrast.getextrema()[0]

    assert 25 < low_contrast_range < baseline_range


@pytest.mark.parametrize(
    "style",
    (
        "rotation_tilted",
        "background_grid",
        "occlusion_local",
        "blur_mild",
        "distractor_marks",
        "layout_shifted",
    ),
)
def test_seeded_factors_use_realization_index_deterministically(style: str) -> None:
    scene = {"value": 7}
    first_config = RenderConfig(style=style, seed=47, realization_index=0)
    second_config = replace(first_config, realization_index=1)

    first = render_scene(scene, TaskFamily.DIGIT_OFFSET, first_config)
    repeat = render_scene(scene, TaskFamily.DIGIT_OFFSET, first_config)
    second = render_scene(scene, TaskFamily.DIGIT_OFFSET, second_config)

    assert first.tobytes() == repeat.tobytes()
    assert first.tobytes() != second.tobytes()
