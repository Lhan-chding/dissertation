"""Real Phase-2 layerwise constraint-assimilation orchestration."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType

from compensability_v4.diagnostics.observation_anchor import (
    classify_assimilation_profile,
)
from compensability_v4.eval.statistics import scene_clustered_bootstrap_ci

from .layerwise_assimilation import layerwise_candidate_logits, layerwise_margins
from .phase2_candidate import CandidateScoringCall, CandidateScoringRecord, CueCondition

World = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class LayerwiseCall:
    call_id: str
    scene_id: str
    family: str
    condition: CueCondition
    messages: tuple[Mapping[str, str], ...]
    candidate_labels: tuple[str, ...]
    candidate_worlds: tuple[World, ...]
    true_label: str
    observed_label: str
    counterfactual_label: str
    source_prompt_sha256: str
    source_candidate_logits: tuple[tuple[str, float], ...]
    expected_language_layers: int


@dataclass(frozen=True, slots=True)
class LayerwiseRecord:
    call_id: str
    scene_id: str
    family: str
    condition: CueCondition
    prompt_sha256: str
    candidate_labels: tuple[str, ...]
    candidate_worlds: tuple[World, ...]
    true_label: str
    observed_label: str
    counterfactual_label: str
    language_layers: int
    candidate_logits_by_layer: tuple[tuple[tuple[str, float], ...], ...]
    margins_true_observed: tuple[float, ...]
    margins_counterfactual_observed: tuple[float, ...]
    delta_f_by_layer: tuple[float, ...]
    assimilation_profile: str | None
    final_forward_parity_verified: bool = True
    generation_invoked: bool = False

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cue_condition"] = self.condition.value
        del payload["condition"]
        payload["candidate_logits_by_layer"] = [
            dict(layer) for layer in self.candidate_logits_by_layer
        ]
        return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _closed_messages(
    messages: Iterable[Mapping[str, str]],
) -> tuple[Mapping[str, str], ...]:
    return tuple(MappingProxyType(dict(message)) for message in messages)


def build_layerwise_plan(
    scoring_calls: Iterable[CandidateScoringCall],
    scoring_records: Iterable[CandidateScoringRecord],
    *,
    expected_scenes: int,
    expected_language_layers: int,
) -> tuple[LayerwiseCall, ...]:
    """Bind each rebuilt prompt to its immutable S3 scoring record."""

    calls = tuple(scoring_calls)
    records = tuple(scoring_records)
    if (
        isinstance(expected_scenes, bool)
        or not isinstance(expected_scenes, int)
        or expected_scenes <= 0
    ):
        raise ValueError("expected_scenes must be a positive integer")
    if (
        isinstance(expected_language_layers, bool)
        or not isinstance(expected_language_layers, int)
        or expected_language_layers <= 0
    ):
        raise ValueError("expected_language_layers must be a positive integer")
    expected_calls = expected_scenes * len(CueCondition)
    if len(calls) != expected_calls or len(records) != expected_calls:
        raise RuntimeError("S4 requires exact scene-by-condition S3 inputs")
    if len({call.call_id for call in calls}) != len(calls):
        raise RuntimeError("rebuilt S4 call identifiers must be unique")
    by_id = {record.call_id: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {call.call_id for call in calls}:
        raise RuntimeError("S3 scoring records do not align with rebuilt S4 calls")
    conditions_by_scene: dict[str, set[CueCondition]] = {}
    plan: list[LayerwiseCall] = []
    for call in calls:
        source = by_id[call.call_id]
        if (
            source.scene_id != call.scene_id
            or source.family != call.family
            or source.condition is not call.condition
            or source.candidate_labels != call.candidate_labels
            or source.candidate_worlds != call.candidate_worlds
            or source.true_label != call.true_label
            or source.observed_label != call.observed_label
            or source.counterfactual_label != call.counterfactual_label
        ):
            raise RuntimeError("S3 scoring semantics drifted from the rebuilt S4 plan")
        if not _is_sha256(source.prompt_sha256):
            raise RuntimeError("S3 prompt SHA-256 is malformed")
        if len(call.candidate_labels) != 4 or len(set(call.candidate_worlds)) != 4:
            raise RuntimeError("S4 requires four labels and four unique candidate worlds")
        source_logits = tuple((label, float(value)) for label, value in source.candidate_logits)
        if {label for label, _value in source_logits} != set(call.candidate_labels) or any(
            not math.isfinite(value) for _label, value in source_logits
        ):
            raise RuntimeError("S3 candidate logits are malformed")
        conditions_by_scene.setdefault(call.scene_id, set()).add(call.condition)
        plan.append(
            LayerwiseCall(
                call_id=call.call_id,
                scene_id=call.scene_id,
                family=call.family,
                condition=call.condition,
                messages=_closed_messages(call.messages),
                candidate_labels=call.candidate_labels,
                candidate_worlds=call.candidate_worlds,
                true_label=call.true_label,
                observed_label=call.observed_label,
                counterfactual_label=call.counterfactual_label,
                source_prompt_sha256=source.prompt_sha256,
                source_candidate_logits=source_logits,
                expected_language_layers=expected_language_layers,
            )
        )
    if len(conditions_by_scene) != expected_scenes or any(
        conditions != set(CueCondition) for conditions in conditions_by_scene.values()
    ):
        raise RuntimeError("S4 scenes do not contain all four cue conditions")
    return tuple(plan)


def _render_prompt(processor: object, call: LayerwiseCall) -> str:
    template = getattr(processor, "apply_chat_template", None)
    if callable(template):
        rendered = template(
            [dict(message) for message in call.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError("processor chat template returned an invalid S4 prompt")
        return rendered
    return json.dumps([dict(message) for message in call.messages], separators=(",", ":"))


def _prepare_batch(processor: object, prompt: str, model: object) -> object:
    if not callable(processor):
        raise TypeError("S4 processor must support text tokenization")
    batch = processor(text=[prompt], padding=True, return_tensors="pt")
    device = getattr(model, "device", None)
    move = getattr(batch, "to", None)
    if device is not None and callable(move):
        return move(device)
    if isinstance(batch, Mapping) and device is not None:
        return {
            key: value.to(device) if callable(getattr(value, "to", None)) else value
            for key, value in batch.items()
        }
    return batch


def _float_layers(
    layers: Iterable[Mapping[str, float]], labels: tuple[str, ...]
) -> tuple[tuple[tuple[str, float], ...], ...]:
    result: list[tuple[tuple[str, float], ...]] = []
    for layer in layers:
        if set(layer) != set(labels):
            raise RuntimeError("S4 layer candidate labels drifted")
        values = tuple((label, float(layer[label])) for label in labels)
        if any(not math.isfinite(value) for _label, value in values):
            raise RuntimeError("S4 layer candidate logit is not finite")
        result.append(values)
    return tuple(result)


def execute_layerwise_plan(
    model: object,
    processor: object,
    calls: Iterable[LayerwiseCall],
    *,
    label_token_ids: Mapping[str, int],
    final_logit_absolute_tolerance: float = 1e-5,
    final_logit_relative_tolerance: float = 1e-5,
    numerical_equality_tolerance: float = 1e-8,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[LayerwiseRecord, ...]:
    """Execute one hidden-state forward per condition and never generate."""

    frozen = tuple(calls)
    if not frozen or len({call.call_id for call in frozen}) != len(frozen):
        raise ValueError("S4 calls must be non-empty with unique identifiers")
    if set(label_token_ids) != set(frozen[0].candidate_labels):
        raise RuntimeError("S4 label-token mapping differs from the S3 candidate labels")
    tolerances = (
        final_logit_absolute_tolerance,
        final_logit_relative_tolerance,
        numerical_equality_tolerance,
    )
    if any(value < 0 or not math.isfinite(value) for value in tolerances):
        raise ValueError("S4 numerical tolerances must be finite and non-negative")
    rendered = tuple(_render_prompt(processor, call) for call in frozen)
    observed_hashes = tuple(hashlib.sha256(prompt.encode()).hexdigest() for prompt in rendered)
    if any(
        observed != call.source_prompt_sha256
        for call, observed in zip(frozen, observed_hashes, strict=True)
    ):
        raise RuntimeError("S4 rendered prompt hash differs from the frozen S3 prompt hash")

    raw: list[LayerwiseRecord] = []
    for completed, (call, prompt, prompt_sha256) in enumerate(
        zip(frozen, rendered, observed_hashes, strict=True), start=1
    ):
        batch = _prepare_batch(processor, prompt, model)
        layers = _float_layers(
            layerwise_candidate_logits(
                model,
                batch,
                label_token_ids,
                absolute_tolerance=final_logit_absolute_tolerance,
                relative_tolerance=final_logit_relative_tolerance,
            ),
            call.candidate_labels,
        )
        if len(layers) != call.expected_language_layers:
            raise RuntimeError("S4 runtime language-layer count drifted")
        mappings = tuple(dict(layer) for layer in layers)
        true_margins = layerwise_margins(
            mappings,
            true_label=call.true_label,
            observed_label=call.observed_label,
        )
        counterfactual_margins = layerwise_margins(
            mappings,
            true_label=call.counterfactual_label,
            observed_label=call.observed_label,
        )
        raw.append(
            LayerwiseRecord(
                call_id=call.call_id,
                scene_id=call.scene_id,
                family=call.family,
                condition=call.condition,
                prompt_sha256=prompt_sha256,
                candidate_labels=call.candidate_labels,
                candidate_worlds=call.candidate_worlds,
                true_label=call.true_label,
                observed_label=call.observed_label,
                counterfactual_label=call.counterfactual_label,
                language_layers=len(layers),
                candidate_logits_by_layer=layers,
                margins_true_observed=true_margins,
                margins_counterfactual_observed=counterfactual_margins,
                delta_f_by_layer=(),
                assimilation_profile=None,
            )
        )
        if progress is not None:
            progress(completed, len(frozen))

    by_scene: dict[str, dict[CueCondition, LayerwiseRecord]] = {}
    for record in raw:
        group = by_scene.setdefault(record.scene_id, {})
        if record.condition in group:
            raise RuntimeError("S4 execution produced duplicate scene conditions")
        group[record.condition] = record
    if any(set(group) != set(CueCondition) for group in by_scene.values()):
        raise RuntimeError("S4 execution produced incomplete scene conditions")
    completed_records: list[LayerwiseRecord] = []
    for record in raw:
        group = by_scene[record.scene_id]
        no_cue = group[CueCondition.NO_CUE]
        if record.condition is CueCondition.COUNTERFACTUAL_CUE:
            margins = record.margins_counterfactual_observed
            baseline = no_cue.margins_counterfactual_observed
        else:
            margins = record.margins_true_observed
            baseline = no_cue.margins_true_observed
        deltas = tuple(value - base for value, base in zip(margins, baseline, strict=True))
        profile = None
        if record.condition is CueCondition.VALID_CUE:
            profile = classify_assimilation_profile(
                no_cue.margins_true_observed,
                record.margins_true_observed,
                numerical_tolerance=numerical_equality_tolerance,
            )
        completed_records.append(
            replace(record, delta_f_by_layer=deltas, assimilation_profile=profile)
        )
    return tuple(completed_records)


def _contrast_curves(group: Mapping[CueCondition, LayerwiseRecord], kind: str) -> tuple[float, ...]:
    no_cue = group[CueCondition.NO_CUE]
    valid = group[CueCondition.VALID_CUE]
    sham = group[CueCondition.SHAM_CUE]
    counterfactual = group[CueCondition.COUNTERFACTUAL_CUE]
    if kind == "valid_no":
        before, after = no_cue.margins_true_observed, valid.margins_true_observed
    elif kind == "sham_no":
        before, after = no_cue.margins_true_observed, sham.margins_true_observed
    elif kind == "valid_sham":
        before, after = sham.margins_true_observed, valid.margins_true_observed
    elif kind == "counterfactual_no":
        before = no_cue.margins_counterfactual_observed
        after = counterfactual.margins_counterfactual_observed
    else:
        before = sham.margins_counterfactual_observed
        after = counterfactual.margins_counterfactual_observed
    return tuple(value - base for value, base in zip(after, before, strict=True))


_CONTRASTS = {
    "valid_minus_no_cue_margin": "valid_no",
    "sham_minus_no_cue_margin": "sham_no",
    "valid_minus_sham_margin": "valid_sham",
    "counterfactual_minus_no_cue_target_margin": "counterfactual_no",
    "counterfactual_minus_sham_target_margin": "counterfactual_sham",
}


def _effect_summary(
    groups: Mapping[str, Mapping[CueCondition, LayerwiseRecord]],
    *,
    kind: str,
    bootstrap_resamples: int,
) -> dict[str, object]:
    curves = {scene_id: _contrast_curves(group, kind) for scene_id, group in groups.items()}
    layers = len(next(iter(curves.values())))
    mean = tuple(
        sum(curve[layer] for curve in curves.values()) / len(curves) for layer in range(layers)
    )
    final_rows = tuple(
        {"scene_id": scene_id, "difference": curve[-1]}
        for scene_id, curve in sorted(curves.items())
    )
    interval = scene_clustered_bootstrap_ci(
        final_rows,
        metric="difference",
        n_resamples=bootstrap_resamples,
        seed=2026081701,
    )
    return {
        "mean_by_layer": mean,
        "final_layer": {
            "estimate": interval.estimate,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "confidence": interval.confidence,
            "number_of_scenes": interval.number_of_scenes,
        },
    }


def summarize_layerwise_records(
    records: Iterable[LayerwiseRecord], *, bootstrap_resamples: int = 10_000
) -> dict[str, object]:
    frozen = tuple(records)
    if not frozen or len({record.call_id for record in frozen}) != len(frozen):
        raise ValueError("S4 records must be non-empty with unique identifiers")
    layer_counts = {record.language_layers for record in frozen}
    if len(layer_counts) != 1 or any(not record.final_forward_parity_verified for record in frozen):
        raise RuntimeError("S4 layer counts or final-forward parity are incomplete")
    groups: dict[str, dict[CueCondition, LayerwiseRecord]] = {}
    families: dict[str, str] = {}
    for record in frozen:
        previous = families.setdefault(record.scene_id, record.family)
        group = groups.setdefault(record.scene_id, {})
        if previous != record.family or record.condition in group:
            raise RuntimeError("S4 records cross scene or family boundaries")
        group[record.condition] = record
    if any(set(group) != set(CueCondition) for group in groups.values()):
        raise RuntimeError("S4 summary requires all four conditions per scene")
    effects = {
        name: _effect_summary(
            groups,
            kind=kind,
            bootstrap_resamples=bootstrap_resamples,
        )
        for name, kind in _CONTRASTS.items()
    }
    by_family: dict[str, object] = {}
    for family in sorted(set(families.values())):
        family_groups = {
            scene_id: groups[scene_id]
            for scene_id in sorted(groups)
            if families[scene_id] == family
        }
        by_family[family] = {
            "number_of_scenes": len(family_groups),
            "paired_effects": {
                name: _effect_summary(
                    family_groups,
                    kind=kind,
                    bootstrap_resamples=bootstrap_resamples,
                )
                for name, kind in _CONTRASTS.items()
            },
        }
    profiles = Counter(
        group[CueCondition.VALID_CUE].assimilation_profile for group in groups.values()
    )
    if None in profiles:
        raise RuntimeError("S4 valid-cue assimilation profiles are incomplete")
    return {
        "schema_version": 1,
        "status": "PHASE_2_LAYERWISE_ASSIMILATION_EXECUTED",
        "number_of_scenes": len(groups),
        "number_of_forward_calls": len(frozen),
        "language_layers": layer_counts.pop(),
        "family_counts": dict(sorted(Counter(families.values()).items())),
        "condition_counts": dict(
            sorted(Counter(record.condition.value for record in frozen).items())
        ),
        "profile_counts": dict(sorted(profiles.items())),
        "paired_effects": effects,
        "by_family": by_family,
        "final_forward_parity_verified": True,
        "generation_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
        "subjective_success_threshold_applied": False,
    }


def load_candidate_label_evidence(
    path: Path,
    *,
    expected_model_snapshot_sha256: str,
    expected_config_sha256: str,
    expected_package_lock_sha256: str,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Load the hash-bound S3 label mapping and reject provenance drift."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "PHASE_2_SINGLE_TOKEN_LABELS_VERIFIED",
        "model_snapshot_sha256": expected_model_snapshot_sha256,
        "config_sha256": expected_config_sha256,
        "package_lock_sha256": expected_package_lock_sha256,
        "generation_invoked": False,
        "subjective_success_threshold_applied": False,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("S3 candidate-label evidence provenance drifted")
    rows = payload.get("labels")
    if not isinstance(rows, list) or len(rows) != 4:
        raise RuntimeError("S3 candidate-label evidence must contain exactly four labels")
    labels: list[str] = []
    token_ids: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"label", "token_id"}:
            raise RuntimeError("S3 candidate-label row is malformed")
        label, token_id = row["label"], row["token_id"]
        if (
            not isinstance(label, str)
            or not label
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
        ):
            raise RuntimeError("S3 candidate label or token id is invalid")
        labels.append(label)
        token_ids[label] = token_id
    if len(set(labels)) != 4 or len(set(token_ids.values())) != 4:
        raise RuntimeError("S3 candidate labels and token ids must be unique")
    return tuple(labels), token_ids


def _world(value: object, name: str) -> World:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RuntimeError(f"S3 {name} must contain exactly four integers")
    return tuple(value)  # type: ignore[return-value]


def load_candidate_scoring_records(path: Path) -> tuple[CandidateScoringRecord, ...]:
    """Parse the immutable S3 per-condition records into closed dataclasses."""

    records: list[CandidateScoringRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                labels = tuple(row["candidate_labels"])
                logits = row["candidate_logits"]
                log_probabilities = row["candidate_log_probabilities"]
                if (
                    not isinstance(row, dict)
                    or len(labels) != 4
                    or not isinstance(logits, dict)
                    or not isinstance(log_probabilities, dict)
                    or set(logits) != set(labels)
                    or set(log_probabilities) != set(labels)
                ):
                    raise RuntimeError("candidate mapping fields are malformed")
                record = CandidateScoringRecord(
                    call_id=str(row["call_id"]),
                    scene_id=str(row["scene_id"]),
                    family=str(row["family"]),
                    condition=CueCondition(row["cue_condition"]),
                    prompt_sha256=str(row["prompt_sha256"]),
                    candidate_labels=labels,
                    candidate_worlds=tuple(
                        _world(world, "candidate world") for world in row["candidate_worlds"]
                    ),
                    true_world=_world(row["true_world"], "true world"),
                    observed_world=_world(row["observed_world"], "observed world"),
                    counterfactual_world=_world(
                        row["counterfactual_world"], "counterfactual world"
                    ),
                    true_label=str(row["true_label"]),
                    observed_label=str(row["observed_label"]),
                    counterfactual_label=str(row["counterfactual_label"]),
                    candidate_logits=tuple((label, float(logits[label])) for label in labels),
                    candidate_log_probabilities=tuple(
                        (label, float(log_probabilities[label])) for label in labels
                    ),
                    logp_true=float(row["logp_true"]),
                    logp_observed=float(row["logp_observed"]),
                    logp_counterfactual=float(row["logp_counterfactual"]),
                    margin_true_observed=float(row["margin_true_observed"]),
                    margin_counterfactual_observed=float(row["margin_counterfactual_observed"]),
                    true_rank=int(row["true_rank"]),
                    observed_rank=int(row["observed_rank"]),
                    counterfactual_rank=int(row["counterfactual_rank"]),
                    generation_invoked=bool(row["generation_invoked"]),
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise RuntimeError(f"S3 candidate record {line_number} is malformed") from error
            numeric = (
                *(value for _label, value in record.candidate_logits),
                *(value for _label, value in record.candidate_log_probabilities),
                record.logp_true,
                record.logp_observed,
                record.logp_counterfactual,
                record.margin_true_observed,
                record.margin_counterfactual_observed,
            )
            if record.generation_invoked or any(not math.isfinite(value) for value in numeric):
                raise RuntimeError(f"S3 candidate record {line_number} violates scoring invariants")
            records.append(record)
    if not records:
        raise RuntimeError("S3 candidate-scoring artifact contains no records")
    return tuple(records)


def validate_candidate_scoring_summary(
    path: Path,
    *,
    expected_scenes: int,
    expected_forward_calls: int,
    expected_family_counts: Mapping[str, int],
    expected_model_snapshot_sha256: str,
    expected_config_sha256: str,
    expected_package_lock_sha256: str,
) -> dict[str, object]:
    """Validate S3 structure and provenance without gating any measured effect."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "PHASE_2_CANDIDATE_SCORING_EXECUTED",
        "number_of_scenes": expected_scenes,
        "number_of_forward_calls": expected_forward_calls,
        "family_counts": dict(expected_family_counts),
        "condition_counts": {condition.value: expected_scenes for condition in CueCondition},
        "model_snapshot_sha256": expected_model_snapshot_sha256,
        "config_sha256": expected_config_sha256,
        "package_lock_sha256": expected_package_lock_sha256,
        "generation_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
        "subjective_success_threshold_applied": False,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("S3 candidate-scoring summary structure or provenance drifted")
    return payload


def write_layerwise_outputs(
    *,
    records_path: Path,
    summary_path: Path,
    records: Iterable[LayerwiseRecord],
    summary: Mapping[str, object],
) -> None:
    frozen = tuple(records)
    if not frozen:
        raise ValueError("S4 layerwise records must not be empty")
    if records_path == summary_path or any(
        path.exists() or path.is_symlink() for path in (records_path, summary_path)
    ):
        raise FileExistsError("refusing to overwrite an S4 layerwise artifact")
    if records_path.parent.is_symlink() or summary_path.parent.is_symlink():
        raise RuntimeError("S4 output parent must not be a symlink")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("x", encoding="utf-8") as stream:
        for record in frozen:
            json.dump(record.to_mapping(), stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(dict(summary), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


__all__ = [
    "LayerwiseCall",
    "LayerwiseRecord",
    "build_layerwise_plan",
    "execute_layerwise_plan",
    "load_candidate_label_evidence",
    "load_candidate_scoring_records",
    "summarize_layerwise_records",
    "validate_candidate_scoring_summary",
    "write_layerwise_outputs",
]
