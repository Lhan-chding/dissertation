"""Secure, deterministic reproduction of the frozen Qwen v4 row audits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

from .error_order import phase8_error_summary
from .fiber_multiplicity import answer_fiber_statistics, validate_world

RawRow = Mapping[str, Any]
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MAX_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_CONTENT_BYTES = 4 * 1024 * 1024 * 1024

FACT_MANIFEST = "server_artifact_manifest.json"
CAPABILITY_ROWS = "artifacts/v4/capability_chain/per_scene.csv"
CANDIDATE_ROWS = "artifacts/v4/candidate_scoring/per_scene.jsonl"
INTERFACE_ROWS = "artifacts/v4/interface_ladder/per_scene.jsonl"
RL_ROWS = "artifacts/v4/rl/data/answer_only.jsonl"
SUPPORT_ROWS = "artifacts/v4/training/support.jsonl"
CONFIRM_SCENES = "artifacts/v4/phase8/confirm_data/confirm_scenes.jsonl"
CONFIRM_OBSERVATIONS = "artifacts/v4/phase8/confirm_data/confirm_observations.jsonl"
PHASE8_ROWS = "artifacts/v4/phase8/evaluation/per_scene.jsonl"


def sha256_file(path: str | Path) -> str:
    """Hash a local regular file without loading it into memory."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"input must be a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, label: str) -> str:
    normalized = value.lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return normalized


def _safe_member_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe tar member path: {name!r}")
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe tar member path: {name!r}")
    if candidate.parts and candidate.parts[0].endswith(":"):
        raise ValueError(f"unsafe tar member path: {name!r}")
    normalized = candidate.as_posix()
    if normalized in {".", ""} or normalized != name.rstrip("/"):
        raise ValueError(f"non-canonical tar member path: {name!r}")
    return normalized


class SafeTarArchive(AbstractContextManager["SafeTarArchive"]):
    """Read-only tar access that fails closed before exposing member contents."""

    def __init__(self, path: str | Path, *, expected_sha256: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = _validate_sha256(expected_sha256, label="expected_sha256")
        self.archive_sha256 = sha256_file(self.path)
        if self.archive_sha256 != self.expected_sha256:
            raise ValueError(
                f"archive SHA-256 mismatch: expected {self.expected_sha256}, "
                f"got {self.archive_sha256}"
            )
        self._archive = tarfile.open(self.path, mode="r:gz")  # noqa: SIM115
        try:
            self._members = self._validate_members(self._archive.getmembers())
        except Exception:
            self._archive.close()
            raise
        self._member_hashes: dict[str, str] = {}

    @staticmethod
    def _validate_members(members: Sequence[tarfile.TarInfo]) -> dict[str, tarfile.TarInfo]:
        validated: dict[str, tarfile.TarInfo] = {}
        total_bytes = 0
        for member in members:
            name = _safe_member_name(member.name)
            if name in validated:
                raise ValueError(f"duplicate tar member: {name}")
            if member.issym() or member.islnk():
                raise ValueError(f"tar links are forbidden: {name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported tar member type: {name}")
            if member.isfile():
                if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                    raise ValueError(f"tar member has unsafe size: {name}")
                total_bytes += member.size
                if total_bytes > MAX_ARCHIVE_CONTENT_BYTES:
                    raise ValueError("tar archive exceeds the uncompressed size limit")
            validated[name] = member
        return validated

    @property
    def file_names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, member in self._members.items() if member.isfile()))

    def _open_member(self, name: str) -> BinaryIO:
        normalized = _safe_member_name(name)
        member = self._members.get(normalized)
        if member is None or not member.isfile():
            raise FileNotFoundError(normalized)
        handle = self._archive.extractfile(member)
        if handle is None:
            raise OSError(f"could not read tar member: {normalized}")
        return cast(BinaryIO, handle)

    def member_sha256(self, name: str) -> str:
        normalized = _safe_member_name(name)
        if normalized not in self._member_hashes:
            digest = hashlib.sha256()
            with self._open_member(normalized) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._member_hashes[normalized] = digest.hexdigest()
        return self._member_hashes[normalized]

    def member_size(self, name: str) -> int:
        normalized = _safe_member_name(name)
        member = self._members.get(normalized)
        if member is None or not member.isfile():
            raise FileNotFoundError(normalized)
        return member.size

    def read_json(self, name: str) -> Mapping[str, Any]:
        with self._open_member(name) as handle:
            payload = json.load(io.TextIOWrapper(handle, encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(f"JSON member must contain an object: {name}")
        return payload

    def read_jsonl(self, name: str) -> tuple[Mapping[str, Any], ...]:
        rows: list[Mapping[str, Any]] = []
        with self._open_member(name) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8")
            for line_number, line in enumerate(text, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise TypeError(f"JSONL row must be an object: {name}:{line_number}")
                rows.append(row)
        return tuple(rows)

    def read_csv(self, name: str) -> tuple[Mapping[str, str], ...]:
        with self._open_member(name) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            rows = tuple(dict(row) for row in csv.DictReader(text))
        return rows

    def close(self) -> None:
        self._archive.close()

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _manifest_entries(payload: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    files = payload.get("files")
    if not isinstance(files, list):
        raise TypeError("server artifact manifest files must be a list")
    result: dict[str, tuple[str, int]] = {}
    for entry in files:
        if not isinstance(entry, Mapping):
            raise TypeError("server artifact manifest entries must be objects")
        path = _safe_member_name(str(entry.get("path", "")))
        if path in result:
            raise ValueError(f"duplicate server manifest path: {path}")
        digest = _validate_sha256(str(entry.get("sha256", "")), label=f"sha256 for {path}")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise TypeError(f"size_bytes for {path} must be a nonnegative integer")
        result[path] = digest, size
    return result


def _verify_archive_members(
    archive: SafeTarArchive, manifest: Mapping[str, tuple[str, int]], *, skip: frozenset[str]
) -> tuple[int, tuple[str, ...]]:
    verified = 0
    uncovered: list[str] = []
    for name in archive.file_names:
        if name in skip:
            continue
        expected = manifest.get(name)
        if expected is None:
            uncovered.append(name)
            continue
        expected_hash, expected_size = expected
        if archive.member_size(name) != expected_size:
            raise ValueError(f"manifest size mismatch for {name}")
        if archive.member_sha256(name) != expected_hash:
            raise ValueError(f"manifest SHA-256 mismatch for {name}")
        verified += 1
    return verified, tuple(sorted(uncovered))


def verify_v4_bundles(
    fact_bundle_path: str | Path,
    raw_archive_path: str | Path,
    *,
    expected_fact_sha256: str,
    expected_raw_sha256: str,
) -> dict[str, Any]:
    """Verify fixed archive hashes, safe members, and the server member manifest."""

    with (
        SafeTarArchive(fact_bundle_path, expected_sha256=expected_fact_sha256) as facts,
        SafeTarArchive(raw_archive_path, expected_sha256=expected_raw_sha256) as raw,
    ):
        manifest = _manifest_entries(facts.read_json(FACT_MANIFEST))
        fact_count, fact_uncovered = _verify_archive_members(
            facts, manifest, skip=frozenset({FACT_MANIFEST})
        )
        raw_count, raw_uncovered = _verify_archive_members(raw, manifest, skip=frozenset())
        supplied = set(facts.file_names) | set(raw.file_names)
        return {
            "schema_version": 1,
            "artifact_type": "qwen_v4_bundle_verification",
            "fact_bundle": {
                "sha256": facts.archive_sha256,
                "file_count": len(facts.file_names),
                "manifest_verified_file_count": fact_count,
                "archive_hash_only_files": list(fact_uncovered),
            },
            "raw_archive": {
                "sha256": raw.archive_sha256,
                "file_count": len(raw.file_names),
                "manifest_verified_file_count": raw_count,
                "archive_hash_only_files": list(raw_uncovered),
            },
            "server_manifest": {
                "sha256": facts.member_sha256(FACT_MANIFEST),
                "entry_count": len(manifest),
                "entries_not_supplied": sorted(set(manifest) - supplied),
            },
        }


def _as_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"{label} must be boolean")


def capability_chain_summary(rows: Iterable[RawRow]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[tuple[bool, bool]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["family"])][str(row["task_type"])].append(
            (
                _as_bool(row["parse_success"], label="parse_success"),
                _as_bool(row["is_correct"], label="is_correct"),
            )
        )
    return {
        family: {
            task: {
                "scene_count": len(values),
                "parse_rate": sum(parsed for parsed, _ in values) / len(values),
                "accuracy": sum(correct for _, correct in values) / len(values),
            }
            for task, values in sorted(tasks.items())
        }
        for family, tasks in sorted(grouped.items())
    }


def _paired_values(
    rows: Iterable[RawRow], *, left: str, right: str, value_field: str
) -> dict[str, list[float]]:
    cells: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        family = str(row["family"])
        scene_id = str(row["scene_id"])
        condition = str(row["cue_condition"])
        if condition not in {left, right}:
            continue
        key = (family, scene_id)
        if condition in cells[key]:
            raise ValueError(f"duplicate {condition} row for {family}/{scene_id}")
        value = row[value_field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{value_field} must be numeric")
        cells[key][condition] = float(value)
    paired: dict[str, list[float]] = defaultdict(list)
    for (family, _scene_id), values in cells.items():
        if values.keys() == {left, right}:
            paired[family].append(values[left] - values[right])
    return paired


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int = 20260821, samples: int = 10_000
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def candidate_margin_summary(rows: Iterable[RawRow]) -> dict[str, Any]:
    paired = _paired_values(
        rows,
        left="valid_cue",
        right="sham_cue",
        value_field="margin_true_observed",
    )
    summary: dict[str, Any] = {}
    for family, values in sorted(paired.items()):
        low, high = _bootstrap_mean_interval(values)
        summary[family] = {
            "paired_scene_count": len(values),
            "valid_minus_sham_target_margin_mean": sum(values) / len(values),
            "bootstrap_95_ci": [low, high],
        }
    return {
        "by_family": summary,
        "bootstrap": {"method": "paired_percentile", "samples": 10_000, "seed": 20260821},
    }


def interface_revision_summary(rows: Iterable[RawRow]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("interface") != "I3_same_conversation_visual_revision":
            continue
        condition = str(row.get("cue_condition"))
        if condition not in {"valid_cue", "no_cue"}:
            continue
        family = str(row["family"])
        counts[family][f"{condition}_total"] += 1
        output = row.get("output_world")
        truth = validate_world(row["true_world"], label="true_world")
        if isinstance(output, list) and validate_world(output, label="output_world") == truth:
            counts[family][f"{condition}_exact"] += 1
    result: dict[str, Any] = {}
    for family, counter in sorted(counts.items()):
        valid_total = counter["valid_cue_total"]
        no_cue_total = counter["no_cue_total"]
        if not valid_total or not no_cue_total:
            raise ValueError(f"I3 cue pair is incomplete for family {family}")
        valid_rate = counter["valid_cue_exact"] / valid_total
        no_cue_rate = counter["no_cue_exact"] / no_cue_total
        result[family] = {
            "valid_scene_count": valid_total,
            "no_cue_scene_count": no_cue_total,
            "valid_exact_revision_rate": valid_rate,
            "no_cue_exact_revision_rate": no_cue_rate,
            "valid_minus_no_cue": valid_rate - no_cue_rate,
        }
    return {"by_family": result}


def i1_top4_summary(rows: Iterable[RawRow]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("interface") != "I1_soft_report_diagnostic":
            continue
        family = str(row["family"])
        truth = validate_world(row["true_world"], label="true_world")
        observed = validate_world(row["observed_world"], label="observed_world")
        error_indices = [
            index
            for index, pair in enumerate(zip(truth, observed, strict=True))
            if pair[0] != pair[1]
        ]
        if len(error_indices) != 1:
            counts[family]["excluded_non_single_error"] += 1
            continue
        payload = row.get("diagnostic_payload")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("positions"), list):
            raise TypeError("I1 diagnostic payload must contain positions")
        index = error_indices[0]
        positions = {
            position.get("index"): position
            for position in payload["positions"]
            if isinstance(position, Mapping)
        }
        position = positions.get(index)
        if not isinstance(position, Mapping) or not isinstance(position.get("candidates"), list):
            raise ValueError(f"I1 row lacks candidates for erroneous coordinate {index}")
        candidate_values = {
            candidate.get("value")
            for candidate in position["candidates"]
            if isinstance(candidate, Mapping)
        }
        counts[family]["eligible"] += 1
        counts[family]["covered"] += truth[index] in candidate_values
    return {
        "by_family": {
            family: {
                "eligible_scene_count": counter["eligible"],
                "true_value_top4_count": counter["covered"],
                "true_value_top4_rate": counter["covered"] / counter["eligible"],
                "excluded_non_single_error_count": counter["excluded_non_single_error"],
            }
            for family, counter in sorted(counts.items())
            if counter["eligible"]
        },
        "score_basis": "first_token_logit",
        "top_k": 4,
    }


def support_budget_summary(rows: Iterable[RawRow]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_scenes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        variant = str(row["variant"])
        stage = str(row["curriculum_stage"])
        counts[variant] += 1
        stage_counts[variant][stage] += 1
        source_scenes[variant].add(str(row["source_scene_id"]))
    c0_count = counts.get("C0_format_only", 0)
    t_count = counts.get("T_constraint_recovery", 0)
    return {
        "row_counts_by_variant": dict(sorted(counts.items())),
        "row_counts_by_variant_and_stage": {
            variant: dict(sorted(values.items()))
            for variant, values in sorted(stage_counts.items())
        },
        "curriculum_stage_counts_by_variant": {
            variant: len(values) for variant, values in sorted(stage_counts.items())
        },
        "unique_source_scene_counts_by_variant": {
            variant: len(values) for variant, values in sorted(source_scenes.items())
        },
        "row_ratio_T_to_C0": t_count / c0_count if c0_count else None,
    }


def phase8_transition_counts(rows: Iterable[RawRow]) -> dict[str, Any]:
    metrics = ("free_generation_answer_exact", "post_revision_world_exact")
    keyed: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        scene_id = str(row["scene_id"])
        checkpoint = str(row["checkpoint"])
        if checkpoint in keyed[scene_id]:
            raise ValueError(f"duplicate Phase-8 row for {scene_id}/{checkpoint}")
        keyed[scene_id][checkpoint] = row
    comparisons = ("Recovery_LoRA_AnswerOnly_RL", "Recovery_LoRA_RecoveryOutcome_RL")
    result: dict[str, Any] = {}
    for metric in metrics:
        result[metric] = {}
        for checkpoint in comparisons:
            paired = [values for values in keyed.values() if "T" in values and checkpoint in values]
            gained = sum(
                not _as_bool(values["T"][metric], label=metric)
                and _as_bool(values[checkpoint][metric], label=metric)
                for values in paired
            )
            lost = sum(
                _as_bool(values["T"][metric], label=metric)
                and not _as_bool(values[checkpoint][metric], label=metric)
                for values in paired
            )
            result[metric][checkpoint] = {
                "paired_scene_count": len(paired),
                "gained": gained,
                "lost": lost,
            }
    return result


@dataclass(frozen=True)
class DerivedSection:
    evidence_class: str
    source_members: tuple[str, ...]
    statistics: Mapping[str, Any]


def _source_records(raw: SafeTarArchive, members: Sequence[str]) -> list[dict[str, str]]:
    return [{"path": member, "sha256": raw.member_sha256(member)} for member in members]


def _section_payload(raw: SafeTarArchive, section: DerivedSection) -> dict[str, Any]:
    return {
        "evidence_class": section.evidence_class,
        "source_members": _source_records(raw, section.source_members),
        "statistics": section.statistics,
    }


def _derivability_matrix(sections: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "analysis": name,
            "status": "derived_from_supplied_rows",
            "evidence_class": section["evidence_class"],
            "source_members": section["source_members"],
            "missing_reason": None,
        }
        for name, section in sorted(sections.items())
    ]


def derive_phase0_analysis(facts: SafeTarArchive, raw: SafeTarArchive) -> dict[str, Any]:
    """Recompute the scoped Phase-0 statistics from row-level artifacts only."""

    interface_rows = raw.read_jsonl(INTERFACE_ROWS)
    sections = {
        "capability_chain": _section_payload(
            raw,
            DerivedSection(
                "diagnostic",
                (CAPABILITY_ROWS,),
                capability_chain_summary(raw.read_csv(CAPABILITY_ROWS)),
            ),
        ),
        "candidate_valid_minus_sham_by_family": _section_payload(
            raw,
            DerivedSection(
                "exploratory",
                (CANDIDATE_ROWS,),
                candidate_margin_summary(raw.read_jsonl(CANDIDATE_ROWS)),
            ),
        ),
        "i3_valid_minus_no_cue_by_family": _section_payload(
            raw,
            DerivedSection(
                "exploratory", (INTERFACE_ROWS,), interface_revision_summary(interface_rows)
            ),
        ),
        "i1_true_value_top4_by_family": _section_payload(
            raw,
            DerivedSection("exploratory", (INTERFACE_ROWS,), i1_top4_summary(interface_rows)),
        ),
        "phase8_error_order_and_domain": _section_payload(
            raw,
            DerivedSection(
                "exploratory",
                (CONFIRM_SCENES, CONFIRM_OBSERVATIONS),
                phase8_error_summary(
                    raw.read_jsonl(CONFIRM_SCENES), raw.read_jsonl(CONFIRM_OBSERVATIONS)
                ),
            ),
        ),
        "answer_fiber_multiplicity": _section_payload(
            raw,
            DerivedSection(
                "exploratory", (RL_ROWS,), answer_fiber_statistics(raw.read_jsonl(RL_ROWS))
            ),
        ),
        "support_budget": _section_payload(
            raw,
            DerivedSection(
                "exploratory", (SUPPORT_ROWS,), support_budget_summary(raw.read_jsonl(SUPPORT_ROWS))
            ),
        ),
        "phase8_checkpoint_transitions": _section_payload(
            raw,
            DerivedSection(
                "exploratory", (PHASE8_ROWS,), phase8_transition_counts(raw.read_jsonl(PHASE8_ROWS))
            ),
        ),
    }
    return {
        "schema_version": 1,
        "artifact_type": "qwen_v5_phase0_derived_analysis",
        "inputs": {
            "fact_bundle_sha256": facts.archive_sha256,
            "raw_archive_sha256": raw.archive_sha256,
            "fact_manifest_sha256": facts.member_sha256(FACT_MANIFEST),
        },
        "sections": sections,
        "derivability_matrix": _derivability_matrix(sections),
    }


def reproduce_phase0_analysis(
    fact_bundle_path: str | Path,
    raw_archive_path: str | Path,
    *,
    expected_fact_sha256: str,
    expected_raw_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify both fixed inputs and return verification plus derived payloads."""

    verification = verify_v4_bundles(
        fact_bundle_path,
        raw_archive_path,
        expected_fact_sha256=expected_fact_sha256,
        expected_raw_sha256=expected_raw_sha256,
    )
    with (
        SafeTarArchive(fact_bundle_path, expected_sha256=expected_fact_sha256) as facts,
        SafeTarArchive(raw_archive_path, expected_sha256=expected_raw_sha256) as raw,
    ):
        return verification, derive_phase0_analysis(facts, raw)


__all__ = [
    "SafeTarArchive",
    "answer_fiber_statistics",
    "candidate_margin_summary",
    "capability_chain_summary",
    "derive_phase0_analysis",
    "i1_top4_summary",
    "interface_revision_summary",
    "phase8_transition_counts",
    "reproduce_phase0_analysis",
    "sha256_file",
    "support_budget_summary",
    "verify_v4_bundles",
]
