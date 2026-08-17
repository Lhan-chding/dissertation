"""Real Phase-2 teacher-forced candidate-scoring design and execution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from compensability_v4.diagnostics.capability_chain import LegacyCapabilityScene
from compensability_v4.eval.statistics import scene_clustered_bootstrap_ci
from compensability_v4.theory.candidate_space import (
    constraint_supported_candidates,
    enumerate_one_edit_candidates,
)

from .candidate_scoring import (
    _encoded_label,
    candidate_log_probabilities,
    score_candidate_labels,
)

World = tuple[int, int, int, int]


class CueCondition(str, Enum):
    NO_CUE = "no_cue"
    VALID_CUE = "valid_cue"
    SHAM_CUE = "sham_cue"
    COUNTERFACTUAL_CUE = "counterfactual_cue"


@dataclass(frozen=True, slots=True)
class CandidateScoringCall:
    call_id: str
    scene_id: str
    family: str
    condition: CueCondition
    observed_world: World
    true_world: World
    counterfactual_world: World
    value_domain: tuple[int, ...]
    facts: tuple[Mapping[str, object], ...]
    candidate_labels: tuple[str, ...]
    candidate_worlds: tuple[World, ...]
    true_label: str
    observed_label: str
    counterfactual_label: str
    messages: tuple[Mapping[str, str], ...]


@dataclass(frozen=True, slots=True)
class CandidateScoringRecord:
    call_id: str
    scene_id: str
    family: str
    condition: CueCondition
    prompt_sha256: str
    candidate_labels: tuple[str, ...]
    candidate_worlds: tuple[World, ...]
    true_world: World
    observed_world: World
    counterfactual_world: World
    true_label: str
    observed_label: str
    counterfactual_label: str
    candidate_logits: tuple[tuple[str, float], ...]
    candidate_log_probabilities: tuple[tuple[str, float], ...]
    logp_true: float
    logp_observed: float
    logp_counterfactual: float
    margin_true_observed: float
    margin_counterfactual_observed: float
    true_rank: int
    observed_rank: int
    counterfactual_rank: int
    generation_invoked: bool = False

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cue_condition"] = self.condition.value
        del payload["condition"]
        payload["candidate_logits"] = dict(self.candidate_logits)
        payload["candidate_log_probabilities"] = dict(self.candidate_log_probabilities)
        return payload


def _closed_fact(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _rank(seed: int, *parts: object) -> bytes:
    text = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(text.encode()).digest()


def _messages(
    prompt: str,
    *,
    observed: World,
    candidates: tuple[tuple[str, World], ...],
    facts: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, str], ...]:
    payload = {
        "observed_world": list(observed),
        "candidates": [{"label": label, "world": list(world)} for label, world in candidates],
        "facts": [dict(fact) for fact in facts],
    }
    return (
        MappingProxyType({"role": "system", "content": prompt}),
        MappingProxyType(
            {
                "role": "user",
                "content": json.dumps(payload, separators=(",", ":"), sort_keys=True),
            }
        ),
    )


def _sham_facts(scene: LegacyCapabilityScene) -> tuple[Mapping[str, object], ...]:
    unchanged = next(
        index
        for index, (truth, observed) in enumerate(zip(scene.truth, scene.observed, strict=True))
        if truth == observed
    )
    facts = tuple(
        _closed_fact(
            {
                "type": "known_value",
                "index": unchanged,
                "value": scene.observed[unchanged],
                "fact_id": f"sham-{index:02d}",
            }
        )
        for index in range(len(scene.facts))
    )
    supported = constraint_supported_candidates(scene.observed, facts, scene.value_domain)
    if len(supported) <= 1:
        raise RuntimeError("sham facts must remain objectively non-recoverable")
    return facts


def _counterfactual_facts(world: World) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _closed_fact(
            {
                "type": "known_value",
                "index": index,
                "value": value,
                "fact_id": f"counterfactual-{index}",
            }
        )
        for index, value in enumerate(world)
    )


def _candidate_worlds(scene: LegacyCapabilityScene, *, seed: int) -> tuple[World, World]:
    alternatives = tuple(
        world
        for world in enumerate_one_edit_candidates(scene.observed, scene.value_domain)
        if world not in {scene.truth, scene.observed}
    )
    ranked = tuple(
        sorted(alternatives, key=lambda world: (_rank(seed, scene.scene_id, world), world))
    )
    if len(ranked) < 2:
        raise RuntimeError("Phase 2 requires two distinct one-edit alternative worlds")
    return ranked[0], ranked[1]


def _label_world_pairs(
    scene: LegacyCapabilityScene,
    labels: tuple[str, ...],
    *,
    true_slot: int,
    counterfactual: World,
    distractor: World,
    seed: int,
) -> tuple[tuple[str, World], ...]:
    remaining = tuple(
        sorted(
            (scene.observed, counterfactual, distractor),
            key=lambda world: (_rank(seed, "assign", scene.scene_id, world), world),
        )
    )
    worlds: list[World] = []
    pending = list(remaining)
    for index in range(4):
        worlds.append(scene.truth if index == true_slot else pending.pop(0))
    if len(set(worlds)) != 4:
        raise RuntimeError("Phase 2 candidate worlds must be unique")
    return tuple(zip(labels, worlds, strict=True))


def build_candidate_scoring_plan(
    scenes: Iterable[LegacyCapabilityScene],
    *,
    prompt: str,
    candidate_labels: Sequence[str],
    seed: int,
) -> tuple[CandidateScoringCall, ...]:
    """Build four paired scoring conditions per scene without model work."""

    frozen = tuple(sorted(scenes, key=lambda item: item.scene_id))
    labels = tuple(candidate_labels)
    if not frozen or len({scene.scene_id for scene in frozen}) != len(frozen):
        raise ValueError("Phase 2 scenes must be non-empty with unique identifiers")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Phase 2 candidate prompt must be non-empty")
    if len(labels) != 4 or len(set(labels)) != 4 or any(not label for label in labels):
        raise ValueError("Phase 2 requires exactly four unique candidate labels")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Phase 2 seed must be an integer")
    calls: list[CandidateScoringCall] = []
    for ordinal, scene in enumerate(frozen):
        if not isinstance(scene, LegacyCapabilityScene):
            raise TypeError("Phase 2 scenes must be LegacyCapabilityScene values")
        counterfactual, distractor = _candidate_worlds(scene, seed=seed)
        pairs = _label_world_pairs(
            scene,
            labels,
            true_slot=ordinal % 4,
            counterfactual=counterfactual,
            distractor=distractor,
            seed=seed,
        )
        label_by_world = {world: label for label, world in pairs}
        condition_facts = {
            CueCondition.NO_CUE: (),
            CueCondition.VALID_CUE: tuple(_closed_fact(fact) for fact in scene.facts),
            CueCondition.SHAM_CUE: _sham_facts(scene),
            CueCondition.COUNTERFACTUAL_CUE: _counterfactual_facts(counterfactual),
        }
        if constraint_supported_candidates(
            scene.observed,
            condition_facts[CueCondition.COUNTERFACTUAL_CUE],
            scene.value_domain,
        ) != (counterfactual,):
            raise RuntimeError("counterfactual facts must uniquely support their target world")
        for condition in CueCondition:
            facts = condition_facts[condition]
            calls.append(
                CandidateScoringCall(
                    call_id=f"{scene.scene_id}.{condition.value}",
                    scene_id=scene.scene_id,
                    family=scene.family,
                    condition=condition,
                    observed_world=scene.observed,
                    true_world=scene.truth,
                    counterfactual_world=counterfactual,
                    value_domain=scene.value_domain,
                    facts=facts,
                    candidate_labels=labels,
                    candidate_worlds=tuple(world for _label, world in pairs),
                    true_label=label_by_world[scene.truth],
                    observed_label=label_by_world[scene.observed],
                    counterfactual_label=label_by_world[counterfactual],
                    messages=_messages(
                        prompt,
                        observed=scene.observed,
                        candidates=pairs,
                        facts=facts,
                    ),
                )
            )
    return tuple(calls)


def _render_prompt(processor: object, call: CandidateScoringCall) -> str:
    template = getattr(processor, "apply_chat_template", None)
    if callable(template):
        rendered = template(list(call.messages), tokenize=False, add_generation_prompt=True)
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError("processor chat template returned invalid candidate prompt")
        return rendered
    shortcut = getattr(call, "messages", None)
    if shortcut is None:
        raise TypeError("processor must expose apply_chat_template()")
    return json.dumps([dict(message) for message in call.messages], separators=(",", ":"))


def _label_scores(
    model: object,
    processor: object,
    call: CandidateScoringCall,
    rendered_prompt: str,
) -> dict[str, float]:
    shortcut = getattr(model, "score_candidate_logits", None)
    if callable(shortcut):
        raw = shortcut(call.messages, call.candidate_labels)
        if not isinstance(raw, Mapping) or set(raw) != set(call.candidate_labels):
            raise RuntimeError("test scoring runtime returned an invalid label mapping")
        scores = {label: float(raw[label]) for label in call.candidate_labels}
    else:
        by_world = score_candidate_labels(
            model,
            processor,
            rendered_prompt,
            call.candidate_labels,
            call.candidate_worlds,
        )
        scores = {
            label: float(by_world[world])
            for label, world in zip(call.candidate_labels, call.candidate_worlds, strict=True)
        }
    if any(not math.isfinite(value) for value in scores.values()):
        raise RuntimeError("candidate logits must be finite")
    return scores


def _competition_rank(scores: Mapping[str, float], label: str) -> int:
    if label not in scores:
        raise ValueError("ranked candidate label is missing")
    target = scores[label]
    return 1 + sum(value > target for value in scores.values())


def execute_candidate_scoring_plan(
    model: object,
    processor: object,
    calls: Iterable[CandidateScoringCall],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[CandidateScoringRecord, ...]:
    """Execute one standard forward per call and never invoke generation."""

    frozen = tuple(calls)
    if not frozen or len({call.call_id for call in frozen}) != len(frozen):
        raise ValueError("candidate scoring calls must be non-empty and unique")
    records: list[CandidateScoringRecord] = []
    for completed, call in enumerate(frozen, start=1):
        rendered = _render_prompt(processor, call)
        scores = _label_scores(model, processor, call, rendered)
        log_probabilities = {
            str(label): float(value) for label, value in candidate_log_probabilities(scores).items()
        }
        records.append(
            CandidateScoringRecord(
                call_id=call.call_id,
                scene_id=call.scene_id,
                family=call.family,
                condition=call.condition,
                prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                candidate_labels=call.candidate_labels,
                candidate_worlds=call.candidate_worlds,
                true_world=call.true_world,
                observed_world=call.observed_world,
                counterfactual_world=call.counterfactual_world,
                true_label=call.true_label,
                observed_label=call.observed_label,
                counterfactual_label=call.counterfactual_label,
                candidate_logits=tuple((label, scores[label]) for label in call.candidate_labels),
                candidate_log_probabilities=tuple(
                    (label, log_probabilities[label]) for label in call.candidate_labels
                ),
                logp_true=log_probabilities[call.true_label],
                logp_observed=log_probabilities[call.observed_label],
                logp_counterfactual=log_probabilities[call.counterfactual_label],
                margin_true_observed=(
                    log_probabilities[call.true_label] - log_probabilities[call.observed_label]
                ),
                margin_counterfactual_observed=(
                    log_probabilities[call.counterfactual_label]
                    - log_probabilities[call.observed_label]
                ),
                true_rank=_competition_rank(scores, call.true_label),
                observed_rank=_competition_rank(scores, call.observed_label),
                counterfactual_rank=_competition_rank(scores, call.counterfactual_label),
            )
        )
        if progress is not None:
            progress(completed, len(frozen))
    return tuple(records)


def _interval(
    rows: Sequence[Mapping[str, object]], *, bootstrap_resamples: int
) -> dict[str, object]:
    result = scene_clustered_bootstrap_ci(
        rows,
        metric="difference",
        n_resamples=bootstrap_resamples,
        seed=2026081701,
    )
    return {
        "estimate": result.estimate,
        "ci_low": result.low,
        "ci_high": result.high,
        "confidence": result.confidence,
        "number_of_scenes": result.number_of_scenes,
    }


def summarize_candidate_scoring(
    records: Iterable[CandidateScoringRecord], *, bootstrap_resamples: int = 10_000
) -> dict[str, object]:
    frozen = tuple(records)
    if not frozen or len({record.call_id for record in frozen}) != len(frozen):
        raise ValueError("candidate scoring records must be non-empty and unique")
    by_scene: dict[str, dict[CueCondition, CandidateScoringRecord]] = {}
    families: dict[str, str] = {}
    for record in frozen:
        previous = families.setdefault(record.scene_id, record.family)
        group = by_scene.setdefault(record.scene_id, {})
        if previous != record.family or record.condition in group:
            raise ValueError("candidate scoring records cross scene or family boundaries")
        group[record.condition] = record
    if any(set(group) != set(CueCondition) for group in by_scene.values()):
        raise ValueError("each Phase 2 scene must contain all four cue conditions")

    def differences(scene_ids: Iterable[str], kind: str) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for scene_id in scene_ids:
            group = by_scene[scene_id]
            no_cue = group[CueCondition.NO_CUE]
            if kind == "valid":
                after = group[CueCondition.VALID_CUE].margin_true_observed
                before = no_cue.margin_true_observed
            elif kind == "sham":
                after = group[CueCondition.SHAM_CUE].margin_true_observed
                before = no_cue.margin_true_observed
            else:
                after = group[CueCondition.COUNTERFACTUAL_CUE].margin_counterfactual_observed
                before = no_cue.margin_counterfactual_observed
            rows.append({"scene_id": scene_id, "difference": after - before})
        return tuple(rows)

    scene_ids = tuple(sorted(by_scene))
    paired = {
        "valid_minus_no_cue_margin": _interval(
            differences(scene_ids, "valid"), bootstrap_resamples=bootstrap_resamples
        ),
        "sham_minus_no_cue_margin": _interval(
            differences(scene_ids, "sham"), bootstrap_resamples=bootstrap_resamples
        ),
        "counterfactual_minus_no_cue_target_margin": _interval(
            differences(scene_ids, "counterfactual"),
            bootstrap_resamples=bootstrap_resamples,
        ),
    }
    by_family: dict[str, object] = {}
    for family in sorted(set(families.values())):
        family_ids = tuple(scene_id for scene_id in scene_ids if families[scene_id] == family)
        by_family[family] = {
            "number_of_scenes": len(family_ids),
            "valid_minus_no_cue_margin": _interval(
                differences(family_ids, "valid"),
                bootstrap_resamples=bootstrap_resamples,
            ),
            "sham_minus_no_cue_margin": _interval(
                differences(family_ids, "sham"),
                bootstrap_resamples=bootstrap_resamples,
            ),
            "counterfactual_minus_no_cue_target_margin": _interval(
                differences(family_ids, "counterfactual"),
                bootstrap_resamples=bootstrap_resamples,
            ),
        }
    condition_metrics: dict[str, object] = {}
    for condition in CueCondition:
        group = tuple(record for record in frozen if record.condition is condition)
        condition_metrics[condition.value] = {
            "number_of_scenes": len(group),
            "mean_logp_true": sum(record.logp_true for record in group) / len(group),
            "mean_logp_observed": sum(record.logp_observed for record in group) / len(group),
            "mean_true_observed_margin": (
                sum(record.margin_true_observed for record in group) / len(group)
            ),
            "mean_true_rank": sum(record.true_rank for record in group) / len(group),
            "mean_observed_rank": sum(record.observed_rank for record in group) / len(group),
        }
    return {
        "schema_version": 1,
        "status": "PHASE_2_CANDIDATE_SCORING_EXECUTED",
        "number_of_scenes": len(scene_ids),
        "number_of_forward_calls": len(frozen),
        "family_counts": dict(sorted(Counter(families.values()).items())),
        "condition_counts": dict(
            sorted(Counter(record.condition.value for record in frozen).items())
        ),
        "condition_metrics": condition_metrics,
        "paired_effects": paired,
        "by_family": by_family,
        "generation_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
        "subjective_success_threshold_applied": False,
    }


def build_candidate_label_evidence(
    tokenizer: object,
    labels: Sequence[str],
    *,
    model_snapshot_sha256: str,
) -> dict[str, object]:
    frozen = tuple(labels)
    token_ids = tuple(_encoded_label(tokenizer, label) for label in frozen)
    if len(frozen) != 4 or any(len(ids) != 1 for ids in token_ids):
        raise RuntimeError("candidate label evidence requires four single-token labels")
    if len({ids[0] for ids in token_ids}) != 4:
        raise RuntimeError("candidate label token ids must be unique")
    if (
        not isinstance(model_snapshot_sha256, str)
        or len(model_snapshot_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_snapshot_sha256)
    ):
        raise ValueError("model snapshot SHA-256 is invalid")
    name = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)
    return {
        "schema_version": 1,
        "status": "PHASE_2_SINGLE_TOKEN_LABELS_VERIFIED",
        "labels": [
            {"label": label, "token_id": token_ids[index][0]} for index, label in enumerate(frozen)
        ],
        "tokenizer_name_or_path": str(name),
        "selection_rule": "first_four_distinct_single_token_labels",
        "model_snapshot_sha256": model_snapshot_sha256,
        "generation_invoked": False,
        "subjective_success_threshold_applied": False,
    }


def validate_phase1_candidate_source(
    scenes: Iterable[LegacyCapabilityScene],
    *,
    per_scene_path: Path,
    summary_path: Path,
    gaps_path: Path,
    expected_source_scenes: int,
    expected_family_counts: Mapping[str, int],
) -> dict[str, int]:
    """Validate Phase-1 structure without imposing an empirical outcome threshold."""

    frozen = tuple(scenes)
    scene_family = {scene.scene_id: scene.family for scene in frozen}
    if len(scene_family) != len(frozen) or Counter(scene_family.values()) != Counter(
        expected_family_counts
    ):
        raise RuntimeError("Phase 1 scene identities or family counts drifted")
    with per_scene_path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    required_tasks = {"T1", "T2", "T3", "T4", "T5", "T6"}
    if len(rows) != len(frozen) * len(required_tasks):
        raise RuntimeError("Phase 1 per-scene record count drifted")
    if len({row.get("call_id") for row in rows}) != len(rows):
        raise RuntimeError("Phase 1 call identifiers are not unique")
    tasks_by_scene: dict[str, set[str]] = {}
    for row in rows:
        scene_id = row.get("scene_id")
        family = row.get("family")
        task = row.get("task_type")
        if scene_id not in scene_family or family != scene_family[scene_id]:
            raise RuntimeError("Phase 1 result scene or family identity drifted")
        tasks_by_scene.setdefault(scene_id, set()).add(str(task))
    if set(tasks_by_scene) != set(scene_family) or any(
        tasks != required_tasks for tasks in tasks_by_scene.values()
    ):
        raise RuntimeError("Phase 1 results do not contain exact T1-T6 scene groups")

    with summary_path.open(newline="", encoding="utf-8") as stream:
        summaries = tuple(csv.DictReader(stream))
    expected_summary_keys = {
        (family, task) for family in expected_family_counts for task in required_tasks
    }
    if {(row.get("family"), row.get("task_type")) for row in summaries} != expected_summary_keys:
        raise RuntimeError("Phase 1 family summary structure drifted")
    for row in summaries:
        family = str(row["family"])
        try:
            number_of_scenes = int(row["number_of_scenes"])
            parse_rate = float(row["parse_rate"])
            accuracy = float(row["accuracy"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Phase 1 family summary value is malformed") from error
        if number_of_scenes != expected_family_counts[family]:
            raise RuntimeError("Phase 1 family summary scene count drifted")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in (parse_rate, accuracy)
        ):
            raise RuntimeError("Phase 1 family summary metric is invalid")

    gaps = json.loads(gaps_path.read_text(encoding="utf-8"))
    expected_provenance = {
        "status": "PHASE_1_EXECUTED",
        "source_eligible_scenes": expected_source_scenes,
        "world_recoverable_scenes": len(frozen),
        "model_calls": len(rows),
        "training_invoked": False,
        "rl_invoked": False,
        "subjective_success_threshold_applied": False,
    }
    if not isinstance(gaps, dict) or any(
        gaps.get(key) != value for key, value in expected_provenance.items()
    ):
        raise RuntimeError("Phase 1 paired-gap provenance drifted")
    return {"number_of_scenes": len(frozen), "number_of_records": len(rows)}


def write_candidate_scoring_outputs(
    *,
    labels_path: Path,
    records_path: Path,
    summary_path: Path,
    label_evidence: Mapping[str, object],
    records: Iterable[CandidateScoringRecord],
    summary: Mapping[str, object],
) -> None:
    frozen = tuple(records)
    paths = (labels_path, records_path, summary_path)
    if not frozen:
        raise ValueError("candidate scoring records must not be empty")
    if len(set(paths)) != 3 or any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("refusing to overwrite a Phase 2 candidate-scoring artifact")
    if any(path.parent.is_symlink() for path in paths):
        raise RuntimeError("Phase 2 output parent must not be a symlink")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("x", encoding="utf-8") as stream:
        json.dump(dict(label_evidence), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with records_path.open("x", encoding="utf-8") as stream:
        for record in frozen:
            json.dump(record.to_mapping(), stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(dict(summary), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


__all__ = [
    "CandidateScoringCall",
    "CandidateScoringRecord",
    "CueCondition",
    "build_candidate_label_evidence",
    "build_candidate_scoring_plan",
    "execute_candidate_scoring_plan",
    "summarize_candidate_scoring",
    "validate_phase1_candidate_source",
    "write_candidate_scoring_outputs",
]
