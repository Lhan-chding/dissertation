"""Phase 0 helpers for reproducing v4 raw-row audits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compbias.recoverability.operators import apply_operation
from compensability_v4.theory.candidate_space import enumerate_one_edit_candidates

RawRow = Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"cannot coerce boolean value from {value!r}")


def _as_world(value: object, *, label: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly four integers")
    if any(type(item) is not int for item in value):
        raise TypeError(f"{label} must contain exact integers")
    return value[0], value[1], value[2], value[3]


def _as_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error


def capability_chain_summary(
    rows: Iterable[RawRow],
) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, dict[str, list[tuple[bool, bool]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        family = str(row["family"])
        task = str(row["task_type"])
        grouped[family][task].append((_as_bool(row["parse_success"]), _as_bool(row["is_correct"])))
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for family, tasks in grouped.items():
        summary[family] = {}
        for task, values in sorted(tasks.items()):
            parse_total = sum(1 for parsed, _ in values if parsed)
            correct_total = sum(1 for _, correct in values if correct)
            count = len(values)
            summary[family][task] = {
                "scene_count": count,
                "parse_rate": parse_total / count,
                "accuracy": correct_total / count,
            }
    return summary


def candidate_margin_summary(rows: Iterable[RawRow]) -> dict[str, dict[str, float | int]]:
    per_scene: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["family"]), str(row["scene_id"]))
        per_scene[key][str(row["cue_condition"])] = _as_float(
            row["margin_true_observed"], label="margin_true_observed"
        )
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family, _scene_id), margins in per_scene.items():
        if "valid_cue" in margins and "sham_cue" in margins:
            by_family[family].append(margins["valid_cue"] - margins["sham_cue"])
    return {
        family: {
            "scene_count": len(values),
            "valid_minus_sham_target_margin_mean": sum(values) / len(values),
        }
        for family, values in sorted(by_family.items())
        if values
    }


def interface_revision_summary(rows: Iterable[RawRow]) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if str(row.get("interface")) != "I3_same_conversation_visual_revision":
            continue
        family = str(row["family"])
        cue = str(row["cue_condition"])
        if cue not in {"valid_cue", "no_cue"}:
            continue
        exact = _as_world(row["output_world"], label="output_world") == _as_world(
            row["true_world"], label="true_world"
        )
        counts[family][f"{cue}_total"] += 1
        if exact:
            counts[family][f"{cue}_exact"] += 1
    summary: dict[str, dict[str, float]] = {}
    for family, counter in sorted(counts.items()):
        valid_total = counter["valid_cue_total"]
        no_cue_total = counter["no_cue_total"]
        valid_rate = counter["valid_cue_exact"] / valid_total if valid_total else 0.0
        no_cue_rate = counter["no_cue_exact"] / no_cue_total if no_cue_total else 0.0
        summary[family] = {
            "valid_exact_revision_rate": valid_rate,
            "no_cue_exact_revision_rate": no_cue_rate,
            "valid_minus_no_cue": valid_rate - no_cue_rate,
        }
    return summary


def answer_fiber_statistics(
    rows: Iterable[RawRow], *, value_domain: Iterable[int] = range(2, 19)
) -> dict[str, float | int]:
    sizes: list[int] = []
    for row in rows:
        observed = _as_world(row["observed"], label="observed")
        answer = int(row["answer"])
        operation = str(row["operation"])
        candidates = enumerate_one_edit_candidates(observed, value_domain)
        size = sum(1 for candidate in candidates if apply_operation(candidate, operation) == answer)
        sizes.append(size)
    if not sizes:
        raise ValueError("rows must not be empty")
    ordered = sorted(sizes)
    return {
        "scene_count": len(sizes),
        "mean_size": sum(sizes) / len(sizes),
        "median_size": float(statistics.median(ordered)),
        "max_size": max(sizes),
        "singleton_count": sum(1 for size in sizes if size == 1),
    }


def support_budget_summary(rows: Iterable[RawRow]) -> dict[str, Any]:
    counts = Counter(str(row["variant"]) for row in rows)
    first_control = counts.get("C0_format_only", 0)
    recovery = counts.get("T_constraint_recovery", 0)
    ratio = recovery / first_control if first_control else None
    return {
        "counts_by_variant": dict(sorted(counts.items())),
        "budget_ratio_T_to_C0": ratio,
    }


def confirm_error_cardinality_summary(
    rows: Iterable[RawRow], *, in_domain: Iterable[int] = range(2, 19)
) -> dict[str, Any]:
    allowed = frozenset(in_domain)
    histogram: Counter[str] = Counter()
    out_of_domain = 0
    for row in rows:
        error_indices = row.get("error_indices", row.get("stage1_error_indices"))
        observed_values = row.get(
            "observed",
            row.get("observed_values", row.get("stage1_parsed_world")),
        )
        if not isinstance(error_indices, Sequence) or isinstance(error_indices, (str, bytes)):
            raise TypeError("error_indices must be a sequence")
        histogram[str(len(error_indices))] += 1
        observed = _as_world(observed_values, label="observed")
        if any(value not in allowed for value in observed):
            out_of_domain += 1
    return {
        "error_count_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "out_of_domain_scene_count": out_of_domain,
    }


def phase8_transition_counts(rows: Iterable[RawRow]) -> dict[str, dict[str, dict[str, int]]]:
    metrics = ("free_generation_answer_exact", "post_revision_world_exact")
    keyed: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    for row in rows:
        checkpoint = str(row["checkpoint"])
        scene_id = str(row["scene_id"])
        keyed[scene_id][checkpoint] = {
            metric: _as_bool(row.get(metric, False)) for metric in metrics
        }
    summary: dict[str, dict[str, dict[str, int]]] = {metric: {} for metric in metrics}
    comparisons = ("Recovery_LoRA_AnswerOnly_RL", "Recovery_LoRA_RecoveryOutcome_RL")
    for metric in metrics:
        for checkpoint in comparisons:
            gained = 0
            lost = 0
            for checkpoints in keyed.values():
                if "T" not in checkpoints or checkpoint not in checkpoints:
                    continue
                before = checkpoints["T"][metric]
                after = checkpoints[checkpoint][metric]
                if after and not before:
                    gained += 1
                if before and not after:
                    lost += 1
            summary[metric][checkpoint] = {"gained": gained, "lost": lost}
    return summary


@dataclass(frozen=True)
class AuditSection:
    member_path: str
    label: str
    aggregator: Any


PHASE0_SECTIONS: tuple[AuditSection, ...] = (
    AuditSection(
        "artifacts/v4/capability_chain/per_scene.csv",
        "capability_chain",
        capability_chain_summary,
    ),
    AuditSection(
        "artifacts/v4/candidate_scoring/per_scene.jsonl",
        "candidate_validity_effects",
        candidate_margin_summary,
    ),
    AuditSection(
        "artifacts/v4/interface_ladder/per_scene.jsonl",
        "natural_state_revision",
        interface_revision_summary,
    ),
    AuditSection(
        "artifacts/v4/rl/data/answer_only.jsonl",
        "answer_fiber_statistics",
        answer_fiber_statistics,
    ),
    AuditSection(
        "artifacts/v4/training/support.jsonl",
        "support_budget",
        support_budget_summary,
    ),
    AuditSection(
        "artifacts/v4/phase8/confirm_data/selection_trace.jsonl",
        "confirm_error_cardinality",
        confirm_error_cardinality_summary,
    ),
    AuditSection(
        "artifacts/v4/phase8/evaluation/per_scene.jsonl",
        "phase8_transitions",
        phase8_transition_counts,
    ),
)


def _read_csv_rows(archive: tarfile.TarFile, member_path: str) -> list[dict[str, str]]:
    handle = archive.extractfile(member_path)
    if handle is None:
        raise FileNotFoundError(member_path)
    with handle:
        text = io.TextIOWrapper(handle, encoding="utf-8")
        return list(csv.DictReader(text))


def _read_jsonl_rows(archive: tarfile.TarFile, member_path: str) -> list[dict[str, Any]]:
    handle = archive.extractfile(member_path)
    if handle is None:
        raise FileNotFoundError(member_path)
    rows: list[dict[str, Any]] = []
    with handle:
        text = io.TextIOWrapper(handle, encoding="utf-8")
        for line in text:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def reproduce_phase0_sections(raw_archive_path: str | Path) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    with tarfile.open(raw_archive_path, "r:gz") as archive:
        for section in PHASE0_SECTIONS:
            if section.member_path.endswith(".csv"):
                rows = _read_csv_rows(archive, section.member_path)
            else:
                rows = _read_jsonl_rows(archive, section.member_path)
            sections[section.label] = section.aggregator(rows)
    return sections


__all__ = [
    "PHASE0_SECTIONS",
    "answer_fiber_statistics",
    "candidate_margin_summary",
    "capability_chain_summary",
    "confirm_error_cardinality_summary",
    "interface_revision_summary",
    "phase8_transition_counts",
    "reproduce_phase0_sections",
    "sha256_file",
    "support_budget_summary",
]
