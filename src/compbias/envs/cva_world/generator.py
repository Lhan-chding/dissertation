"""Deterministic generation and split-audit utilities for CVA-World."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .canonical_solver import canonical_reasoning, solve, solve_sample
from .corruptions import apply_error, reverse_error, validate_error_spec
from .schema import (
    BAR_CHART_INDICES,
    BAR_CHART_OPERATIONS,
    BAR_CHART_QUESTION_TEXTS,
    CVASample,
    ErrorSpec,
    SemanticSplit,
    SplitKeys,
    TaskFamily,
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Closed configuration for a deterministic dataset realization."""

    seed: int = 0
    samples_per_family_per_split: int = 1
    splits: tuple[SemanticSplit, ...] = tuple(SemanticSplit)
    task_families: tuple[TaskFamily, ...] = tuple(TaskFamily)
    visual_styles: tuple[str, ...] = (
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
    train_error_mechanism: str = "offset_plus_2"
    ood_error_mechanism: str = "offset_minus_2"
    realizations_per_semantic: int = 2
    fully_cross_iid_visual_styles: bool = False
    preregistered_ood_factors: tuple[str, ...] = (
        "visual_style",
        "error_mechanism",
    )

    def expected_sample_count(self) -> int:
        """Return the exact number of records without constructing sample payloads."""

        from .renderer import is_visual_style_applicable

        total = 0
        for family in self.task_families:
            applicable_iid_count = sum(
                is_visual_style_applicable(style, family) for style in self.visual_styles[:-1]
            )
            for split in self.splits:
                realizations = (
                    self.realizations_per_semantic
                    if split is SemanticSplit.OOD_TEST or not self.fully_cross_iid_visual_styles
                    else applicable_iid_count
                )
                total += self.samples_per_family_per_split * realizations
        return total

    def __post_init__(self) -> None:
        from .renderer import is_visual_style_applicable, validate_visual_style

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.samples_per_family_per_split, bool)
            or not isinstance(self.samples_per_family_per_split, int)
            or not 1 <= self.samples_per_family_per_split <= 1_000
        ):
            raise ValueError("samples_per_family_per_split must be an integer from 1 to 1000")
        try:
            splits = tuple(SemanticSplit(value) for value in self.splits)
        except (TypeError, ValueError) as error:
            raise ValueError("splits contains an unsupported semantic split") from error
        try:
            families = tuple(TaskFamily(value) for value in self.task_families)
        except (TypeError, ValueError) as error:
            raise ValueError("task_families contains an unsupported family") from error
        if not splits or len(set(splits)) != len(splits):
            raise ValueError("splits must be non-empty and unique")
        if not families or len(set(families)) != len(families):
            raise ValueError("task_families must be non-empty and unique")
        styles = tuple(self.visual_styles)
        if (
            len(styles) < 2
            or len(set(styles)) != len(styles)
            or any(not isinstance(style, str) or not style for style in styles)
        ):
            raise ValueError("visual_styles must contain at least two unique names")
        canonical_styles = tuple(validate_visual_style(style) for style in styles)
        if canonical_styles != styles:
            raise ValueError("visual_styles must use canonical renderer style names")
        if any(not is_visual_style_applicable(styles[-1], family) for family in families):
            raise ValueError("OOD visual style must apply to every configured task family")
        for field, value in (
            ("train_error_mechanism", self.train_error_mechanism),
            ("ood_error_mechanism", self.ood_error_mechanism),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        if self.train_error_mechanism == self.ood_error_mechanism:
            raise ValueError("OOD error mechanism must differ from the training mechanism")
        if (
            isinstance(self.realizations_per_semantic, bool)
            or not isinstance(self.realizations_per_semantic, int)
            or not 2 <= self.realizations_per_semantic <= 16
        ):
            raise ValueError("realizations_per_semantic must be an integer from 2 to 16")
        if not isinstance(self.fully_cross_iid_visual_styles, bool):
            raise TypeError("fully_cross_iid_visual_styles must be boolean")
        factors = tuple(self.preregistered_ood_factors)
        supported_factors = {"visual_style", "error_mechanism"}
        if (
            not factors
            or len(set(factors)) != len(factors)
            or any(factor not in supported_factors for factor in factors)
        ):
            raise ValueError("preregistered_ood_factors must be unique supported factor names")
        if SemanticSplit.OOD_TEST in splits and SemanticSplit.IID_TEST not in splits:
            raise ValueError("ood_test requires iid_test for explicit semantic pairing")
        _mechanism_delta(self.train_error_mechanism)
        _mechanism_delta(self.ood_error_mechanism)
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "task_families", families)
        object.__setattr__(self, "visual_styles", styles)
        object.__setattr__(self, "preregistered_ood_factors", factors)


class SplitLeakageError(ValueError):
    """Raised when a registered semantic or OOD split boundary is crossed."""


@dataclass(frozen=True)
class SplitAudit:
    """Evidence that semantic and factor-shift split contracts are clean."""

    scene_template_leaks: tuple[str, ...]
    answer_leaks: tuple[str, ...]
    visual_style_leaks: tuple[str, ...]
    error_mechanism_leaks: tuple[str, ...]
    ood_pair_mismatches: tuple[str, ...]
    ood_pair_count: int
    preregistered_ood_factors: tuple[str, ...]
    ood_changed_factors: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (
            self.scene_template_leaks
            or self.answer_leaks
            or self.visual_style_leaks
            or self.error_mechanism_leaks
            or self.ood_pair_mismatches
        )


def _mechanism_delta(mechanism: str) -> int:
    if not isinstance(mechanism, str):
        raise TypeError("error mechanism must be a string")
    match = re.fullmatch(r"offset_(plus|minus)_(\d+)", mechanism.lower())
    if match is None:
        raise ValueError(f"unsupported error mechanism: {mechanism!r}")
    magnitude = int(match.group(2))
    if magnitude == 0:
        raise ValueError("error mechanism offset must be nonzero")
    return magnitude if match.group(1) == "plus" else -magnitude


def _task_payload(
    family: TaskFamily,
    split_index: int,
    sample_index: int,
    samples_per_split: int,
    seed_variant: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Generate non-overlapping semantic realizations for any registered size."""

    semantic_base = split_index * samples_per_split + sample_index + 3
    if family is TaskFamily.DIGIT_OFFSET:
        operand = seed_variant % 4 + 1
        return (
            {"value": semantic_base},
            {
                "template": "add_constant",
                "operand": operand,
                "text": f"Add {operand} to the number shown in the image.",
            },
        )
    if family is TaskFamily.COUNT_TRANSFORM:
        scale = 2
        offset = seed_variant % 5 - 2
        return (
            {"count": semantic_base + 3, "shape": "circle"},
            {
                "template": "affine_transform",
                "scale": scale,
                "offset": offset,
                "text": "Apply the stated scale and offset to the object count.",
            },
        )
    if family is TaskFamily.GAUGE_CALIBRATION:
        reading = float(semantic_base) + (seed_variant % 4) / 4
        scale = 1.5
        offset = seed_variant % 5 - 2
        return (
            {
                "reading": reading,
                "minimum": 0.0,
                "maximum": float(max(100, len(SemanticSplit) * samples_per_split + 10)),
            },
            {
                "template": "calibrate",
                "scale": scale,
                "offset": float(offset),
                "text": "Calibrate the gauge reading with the stated affine rule.",
            },
        )
    if family is TaskFamily.BAR_CHART_AGGREGATE:
        operation = BAR_CHART_OPERATIONS[sample_index % len(BAR_CHART_OPERATIONS)]
        bar_variant = (seed_variant + sample_index) % 3
        target = semantic_base + 10
        second = 5 + bar_variant
        if operation == "sum":
            selected = (target - second, second)
        elif operation == "difference":
            selected = (target + second, second)
        else:
            selected = (target * second, second)
        selected_maximum = max(selected)
        bars = [*selected, selected_maximum + 1, selected_maximum + 2]
        return (
            {
                "bars": bars,
                "maximum": float(max(100, max(bars) + 10)),
            },
            {
                "template": "aggregate",
                "operation": operation,
                "indices": list(BAR_CHART_INDICES),
                "text": BAR_CHART_QUESTION_TEXTS[operation],
            },
        )

    relations = ("left_of", "right_of", "above", "below", "parallel", "intersect")
    relation = relations[(split_index + sample_index + seed_variant) % len(relations)]
    rule = {name: f"class_{position}_{seed_variant}" for position, name in enumerate(relations)}
    return (
        {
            "relation": relation,
            "entity_pair": f"pair_{split_index}_{sample_index}_{seed_variant}",
        },
        {
            "template": "relation_lookup",
            "rule": rule,
            "text": "Use the supplied relation rule to name the class.",
        },
    )


def _error_catalog(
    family: TaskFamily, scene: Mapping[str, object], mechanism: str
) -> tuple[ErrorSpec, ...]:
    truth = ErrorSpec.from_mapping(
        {"error_id": "truth", "family": "truth", "severity": 0, "parameters": {}}
    )
    delta = _mechanism_delta(mechanism)
    magnitude = abs(delta)
    if family is TaskFamily.DIGIT_OFFSET:
        corrupt = ErrorSpec.from_mapping(
            {
                "error_id": f"numeric_offset:{delta:+d}",
                "family": "numeric_offset",
                "severity": magnitude,
                "parameters": {"field": "value", "delta": delta},
            }
        )
    elif family is TaskFamily.COUNT_TRANSFORM:
        error_family = "duplication" if delta > 0 else "omission"
        corrupt = ErrorSpec.from_mapping(
            {
                "error_id": f"{error_family}:{magnitude}",
                "family": error_family,
                "severity": magnitude,
                "parameters": {"field": "count", "amount": magnitude},
            }
        )
    elif family is TaskFamily.GAUGE_CALIBRATION:
        corrupt = ErrorSpec.from_mapping(
            {
                "error_id": f"gauge_offset:{delta:+d}",
                "family": "numeric_offset",
                "severity": magnitude,
                "parameters": {"field": "reading", "delta": delta},
            }
        )
    elif family is TaskFamily.BAR_CHART_AGGREGATE:
        local = ErrorSpec.from_mapping(
            {
                "error_id": f"bar:0:{delta:+d}",
                "family": "local_offset",
                "severity": magnitude,
                "parameters": {"field": "bars", "index": 0, "delta": delta},
            }
        )
        inconsistency = ErrorSpec.from_mapping(
            {
                "error_id": "local_to_global:swap:2:3",
                "family": "local_to_global_inconsistency",
                "severity": 1,
                "parameters": {"field": "bars", "indices": [2, 3]},
            }
        )
    else:
        if delta > 0:
            pairs = {
                "left_of": "right_of",
                "right_of": "left_of",
                "above": "below",
                "below": "above",
                "parallel": "intersect",
                "intersect": "parallel",
            }
            mechanism_name = "opposite"
        else:
            pairs = {
                "left_of": "above",
                "above": "left_of",
                "right_of": "below",
                "below": "right_of",
                "parallel": "intersect",
                "intersect": "parallel",
            }
            mechanism_name = "cross_axis"
        corrupt = ErrorSpec.from_mapping(
            {
                "error_id": f"relation_flip:{mechanism_name}",
                "family": "relation_flip",
                "severity": 1,
                "parameters": {"field": "relation", "pairs": pairs},
            }
        )
    errors = (local, inconsistency) if family is TaskFamily.BAR_CHART_AGGREGATE else (corrupt,)
    for error in errors:
        validate_error_spec(error)
        perceived = apply_error(scene, error)
        if reverse_error(perceived, error) != scene:
            raise AssertionError("registered corruption failed its round-trip self-check")
    return (truth, *errors)


def generate_dataset(config: GeneratorConfig) -> tuple[CVASample, ...]:
    """Generate a deterministic, solver-checked dataset without global RNG effects."""

    from .renderer import is_visual_style_applicable

    if not isinstance(config, GeneratorConfig):
        raise TypeError("config must be a GeneratorConfig")
    samples: list[CVASample] = []
    split_positions = {split: index for index, split in enumerate(SemanticSplit)}
    for family_index, family in enumerate(config.task_families):
        seed_variant = (config.seed + 3 * family_index) % 60
        for split in config.splits:
            semantic_source_split = (
                SemanticSplit.IID_TEST if split is SemanticSplit.OOD_TEST else split
            )
            split_index = split_positions[semantic_source_split]
            for sample_index in range(config.samples_per_family_per_split):
                scene, question = _task_payload(
                    family,
                    split_index,
                    sample_index,
                    config.samples_per_family_per_split,
                    seed_variant,
                )
                answer = solve(scene, question, family)
                reasoning = canonical_reasoning(scene, question, family)
                mechanism = (
                    config.ood_error_mechanism
                    if split is SemanticSplit.OOD_TEST
                    else config.train_error_mechanism
                )
                applicable_iid_styles = tuple(
                    style
                    for style in config.visual_styles[:-1]
                    if is_visual_style_applicable(style, family)
                )
                realization_styles = (
                    (config.visual_styles[-1],) * config.realizations_per_semantic
                    if split is SemanticSplit.OOD_TEST
                    else (
                        applicable_iid_styles
                        if config.fully_cross_iid_visual_styles
                        else tuple(
                            applicable_iid_styles[
                                (
                                    realization_index
                                    + sample_index
                                    + family_index
                                    + split_index
                                    + config.seed
                                )
                                % len(applicable_iid_styles)
                            ]
                            for realization_index in range(config.realizations_per_semantic)
                        )
                    )
                )
                for realization_index, style in enumerate(realization_styles):
                    sample_id = (
                        f"{family.value}_{split.value}_{sample_index:06d}_r{realization_index:02d}"
                    )
                    if split is SemanticSplit.OOD_TEST:
                        applicable_sources = len(applicable_iid_styles)
                        source_id = (
                            f"{family.value}_{SemanticSplit.IID_TEST.value}_"
                            f"{sample_index:06d}_r{realization_index % applicable_sources:02d}"
                        )
                    else:
                        source_id = None
                        if len(applicable_iid_styles) < config.realizations_per_semantic:
                            raise ValueError(
                                f"{family.value} has too few applicable IID visual styles"
                            )
                    sample = CVASample(
                        sample_id=sample_id,
                        image_path=f"images/{sample_id}.png",
                        task_family=family,
                        scene=scene,
                        question=question,
                        canonical_answer=answer,
                        canonical_reasoning=reasoning,
                        error_catalog=_error_catalog(family, scene, mechanism),
                        split_keys=SplitKeys(
                            semantic_split=split,
                            visual_style=style,
                            error_mechanism=mechanism,
                        ),
                        source_id=source_id,
                    )
                    solve_sample(sample)
                    samples.append(sample)
    result = tuple(samples)
    audit_splits(
        result,
        preregistered_ood_factors=config.preregistered_ood_factors,
    )
    return result


def generate_error_mechanism_counterfactuals(
    samples: Iterable[CVASample], *, counterfactual_error_mechanism: str
) -> tuple[CVASample, ...]:
    """Create paired records that change only the executable error mechanism.

    These records are intentionally separate from the five disjoint semantic
    splits.  They share sample IDs and semantic content with their source set
    so paired OOD metrics can enforce a true single-factor intervention.
    """

    if not isinstance(counterfactual_error_mechanism, str) or not (
        counterfactual_error_mechanism.strip()
    ):
        raise ValueError("counterfactual_error_mechanism must be a non-empty string")
    records = tuple(samples)
    if not records:
        raise ValueError("samples must not be empty")
    counterfactuals: list[CVASample] = []
    for sample in records:
        if not isinstance(sample, CVASample):
            raise TypeError("all samples must be CVASample records")
        if sample.split_keys.error_mechanism == counterfactual_error_mechanism:
            raise ValueError(
                f"sample {sample.sample_id!r} already uses the counterfactual mechanism"
            )
        split_keys = replace(
            sample.split_keys,
            error_mechanism=counterfactual_error_mechanism,
        )
        counterfactual = replace(
            sample,
            error_catalog=_error_catalog(
                sample.task_family,
                sample.scene,
                counterfactual_error_mechanism,
            ),
            split_keys=split_keys,
        )
        solve_sample(counterfactual)
        counterfactuals.append(counterfactual)
    return tuple(counterfactuals)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cross_split_reuse(samples: Sequence[CVASample], *, kind: str) -> tuple[str, ...]:
    owners: dict[str, set[SemanticSplit]] = defaultdict(set)
    labels: dict[str, str] = {}
    for sample in samples:
        if kind == "scene template":
            value = {
                "task_family": sample.task_family.value,
                "scene": sample.to_mapping()["scene"],
                "question_template": sample.question.get("template"),
            }
        else:
            value = {
                "task_family": sample.task_family.value,
                "answer": sample.to_mapping()["canonical_answer"],
            }
            if sample.task_family is TaskFamily.RELATION_RULE:
                value["semantic_instance"] = sample.scene.get("entity_pair")
        fingerprint = _canonical_hash(value)
        owners[fingerprint].add(sample.split_keys.semantic_split)
        labels[fingerprint] = sample.sample_id
    allowed_pair = {SemanticSplit.IID_TEST, SemanticSplit.OOD_TEST}
    return tuple(
        f"{kind} reused by {labels[key]} across {','.join(sorted(x.value for x in splits))}"
        for key, splits in sorted(owners.items())
        if len(splits) > 1 and splits != allowed_pair
    )


def _ood_overlap(samples: Sequence[CVASample], attribute: str) -> tuple[str, ...]:
    iid_values: set[str] = set()
    ood_values: set[str] = set()
    for sample in samples:
        value = getattr(sample.split_keys, attribute)
        if sample.split_keys.semantic_split is SemanticSplit.OOD_TEST:
            ood_values.add(value)
        else:
            iid_values.add(value)
    return tuple(sorted(iid_values & ood_values))


def _audit_ood_pairs(
    records: Sequence[CVASample],
    preregistered_ood_factors: tuple[str, ...],
) -> tuple[tuple[str, ...], int, tuple[str, ...]]:
    by_id = {sample.sample_id: sample for sample in records}
    iid_ids = {
        sample.sample_id
        for sample in records
        if sample.split_keys.semantic_split is SemanticSplit.IID_TEST
    }
    observed_sources: set[str] = set()
    mismatches: list[str] = []
    observed_changes: set[str] = set()
    invariants = (
        "task_family",
        "scene",
        "question",
        "canonical_answer",
        "canonical_reasoning",
    )
    for sample in records:
        if sample.split_keys.semantic_split is not SemanticSplit.OOD_TEST:
            continue
        source = by_id.get(sample.source_id or "")
        if source is None or source.sample_id not in iid_ids:
            mismatches.append(f"{sample.sample_id}: missing IID source_id")
            continue
        if source.sample_id in observed_sources:
            mismatches.append(f"{sample.sample_id}: duplicate source_id {source.sample_id}")
        observed_sources.add(source.sample_id)
        changed = {
            "visual_style": (sample.split_keys.visual_style != source.split_keys.visual_style),
            "error_mechanism": (
                sample.split_keys.error_mechanism != source.split_keys.error_mechanism
                or sample.error_catalog != source.error_catalog
            ),
        }
        observed_changes.update(factor for factor, value in changed.items() if value)
        invariant_drift = tuple(
            field for field in invariants if getattr(sample, field) != getattr(source, field)
        )
        if invariant_drift:
            mismatches.append(
                f"{sample.sample_id}: OOD pair changed invariant fields {','.join(invariant_drift)}"
            )
        unregistered = tuple(
            factor
            for factor, value in changed.items()
            if value and factor not in preregistered_ood_factors
        )
        if unregistered:
            mismatches.append(
                f"{sample.sample_id}: unregistered OOD factors changed {','.join(unregistered)}"
            )
        unchanged_registered = tuple(
            factor for factor in preregistered_ood_factors if not changed[factor]
        )
        if unchanged_registered:
            mismatches.append(
                f"{sample.sample_id}: preregistered OOD factors unchanged "
                f"{','.join(unchanged_registered)}"
            )
    # OOD records are paired *to* IID semantic sources, but a fully crossed IID
    # design intentionally has more visual realizations than the two held-out
    # OOD realizations.  Requiring every IID realization to be referenced would
    # turn that valid asymmetric design into a false split-leakage failure.  The
    # loop above instead enforces the directional contract that every OOD row
    # names one unique, existing IID source and preserves all semantic fields.
    ordered_changes = tuple(
        factor for factor in ("visual_style", "error_mechanism") if factor in observed_changes
    )
    return tuple(mismatches), len(observed_sources), ordered_changes


def audit_splits(
    samples: Iterable[CVASample],
    *,
    preregistered_ood_factors: Iterable[str],
) -> SplitAudit:
    """Audit semantic reuse and held-out visual/error-factor boundaries."""

    records = tuple(samples)
    if not records:
        raise ValueError("samples must not be empty")
    if any(not isinstance(sample, CVASample) for sample in records):
        raise TypeError("all samples must be CVASample records")
    identifiers = tuple(sample.sample_id for sample in records)
    if len(identifiers) != len(set(identifiers)):
        raise SplitLeakageError("duplicate sample_id values")

    factors = tuple(preregistered_ood_factors)
    supported_factors = {"visual_style", "error_mechanism"}
    if (
        not factors
        or len(factors) != len(set(factors))
        or any(factor not in supported_factors for factor in factors)
    ):
        raise ValueError("preregistered_ood_factors contains unsupported values")
    semantic_records = tuple(
        sample
        for sample in records
        if sample.split_keys.semantic_split is not SemanticSplit.OOD_TEST
    )
    scene_leaks = _cross_split_reuse(semantic_records, kind="scene template")
    answer_leaks = _cross_split_reuse(semantic_records, kind="answer")
    style_leaks = _ood_overlap(records, "visual_style")
    mechanism_leaks = _ood_overlap(records, "error_mechanism")
    pair_mismatches, pair_count, changed_factors = _audit_ood_pairs(records, factors)
    audit = SplitAudit(
        scene_template_leaks=scene_leaks,
        answer_leaks=answer_leaks,
        visual_style_leaks=style_leaks,
        error_mechanism_leaks=mechanism_leaks,
        ood_pair_mismatches=pair_mismatches,
        ood_pair_count=pair_count,
        preregistered_ood_factors=factors,
        ood_changed_factors=changed_factors,
    )
    if not audit.is_clean:
        pieces: list[str] = []
        if scene_leaks:
            pieces.append(f"scene template leakage: {scene_leaks[0]}")
        if answer_leaks:
            pieces.append(f"answer leakage: {answer_leaks[0]}")
        if style_leaks:
            pieces.append(f"visual style leakage: {style_leaks}")
        if mechanism_leaks:
            pieces.append(f"error mechanism leakage: {mechanism_leaks}")
        if pair_mismatches:
            pieces.append(f"OOD pair mismatch: {pair_mismatches[0]}")
        raise SplitLeakageError("; ".join(pieces))
    return audit
