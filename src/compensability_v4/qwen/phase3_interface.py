"""Immutable records and objective summaries for the S6 interface ladder.

This module contains no model runtime.  It validates already measured I0--I4
cells, keeps intervention-only interfaces separate from primary claims, and
computes scene-level paired estimands without applying a success threshold.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from compensability_v4.diagnostics.interface_ladder import (
    CueCondition,
    Interface,
    interface_claim_name,
)
from compensability_v4.eval.statistics import scene_clustered_bootstrap_ci

World: TypeAlias = tuple[int, int, int, int]
ImmutablePayload: TypeAlias = Mapping[str, object]

_PRIMARY_INTERFACES = frozenset(
    {
        Interface.I0_HARD_TEXT,
        Interface.I3_SAME_CONVERSATION,
        Interface.I4_EXACT_CACHE,
    }
)
_INTERVENTION_INTERFACES = frozenset({Interface.I1_SOFT_REPORT, Interface.I2_CANDIDATE_WORLD})
_SOURCE_CONTRACT = {
    Interface.I0_HARD_TEXT: ("S6_runtime", "fresh_text_runtime"),
    Interface.I1_SOFT_REPORT: ("S6_runtime", "stage1_soft_report_runtime"),
    Interface.I2_CANDIDATE_WORLD: ("S3_candidate", "teacher_forced_candidate"),
    Interface.I3_SAME_CONVERSATION: ("S5_cache", "full_history"),
    Interface.I4_EXACT_CACHE: ("S5_cache", "cached_continuation"),
}


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (Interface, CueCondition)):
        return value.value
    return value


def _world(value: object, name: str, *, nullable: bool = False) -> World | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must contain exactly four integers")
    converted = tuple(value)
    if any(type(item) is not int for item in converted):
        raise ValueError(f"{name} must contain exactly four integers")
    return converted  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class InterfaceLadderRecord:
    """One immutable, hash-bound S6 measurement cell."""

    call_id: str
    scene_id: str
    family: str
    interface: Interface
    condition: CueCondition
    true_world: World
    observed_world: World
    counterfactual_world: World
    output_world: World | None
    parse_success: bool | None
    diagnostic_payload: ImmutablePayload | None
    source_stage: str
    source_branch: str
    source_call_id: str
    source_artifact_sha256: str
    structural_validity_verified: bool
    primary_eligible: bool
    diagnostic_only: bool
    diagnostic_reason: str | None

    def __post_init__(self) -> None:
        for name in (
            "call_id",
            "scene_id",
            "family",
            "source_stage",
            "source_branch",
            "source_call_id",
            "source_artifact_sha256",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        try:
            interface = Interface(self.interface)
            condition = CueCondition(self.condition)
        except ValueError as error:
            raise ValueError("interface and condition must use frozen S6 names") from error
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "true_world", _world(self.true_world, "true_world"))
        object.__setattr__(self, "observed_world", _world(self.observed_world, "observed_world"))
        object.__setattr__(
            self,
            "counterfactual_world",
            _world(self.counterfactual_world, "counterfactual_world"),
        )
        object.__setattr__(
            self,
            "output_world",
            _world(self.output_world, "output_world", nullable=True),
        )
        if self.parse_success is not None and type(self.parse_success) is not bool:
            raise TypeError("parse_success must be boolean or null")
        for name in (
            "structural_validity_verified",
            "primary_eligible",
            "diagnostic_only",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if self.diagnostic_reason is not None and (
            not isinstance(self.diagnostic_reason, str) or not self.diagnostic_reason
        ):
            raise ValueError("diagnostic_reason must be a non-empty string or null")
        if self.diagnostic_payload is not None:
            if not isinstance(self.diagnostic_payload, Mapping):
                raise TypeError("diagnostic_payload must be a mapping or null")
            object.__setattr__(self, "diagnostic_payload", _freeze(self.diagnostic_payload))

    @property
    def exact_world_recovery(self) -> bool:
        return self.parse_success is True and self.output_world == self.true_world

    @property
    def observation_copy(self) -> bool:
        return self.parse_success is True and self.output_world == self.observed_world

    @property
    def counterfactual_compliance(self) -> bool | None:
        if self.condition is not CueCondition.COUNTERFACTUAL_CUE:
            return None
        return self.parse_success is True and self.output_world == self.counterfactual_world

    def to_mapping(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "scene_id": self.scene_id,
            "family": self.family,
            "interface": self.interface.value,
            "cue_condition": self.condition.value,
            "true_world": list(self.true_world),
            "observed_world": list(self.observed_world),
            "counterfactual_world": list(self.counterfactual_world),
            "output_world": None if self.output_world is None else list(self.output_world),
            "parse_success": self.parse_success,
            "diagnostic_payload": _plain(self.diagnostic_payload),
            "source_stage": self.source_stage,
            "source_branch": self.source_branch,
            "source_call_id": self.source_call_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "structural_validity_verified": self.structural_validity_verified,
            "primary_eligible": self.primary_eligible,
            "diagnostic_only": self.diagnostic_only,
            "diagnostic_reason": self.diagnostic_reason,
        }


def _required_cells() -> frozenset[tuple[Interface, CueCondition]]:
    cells = {
        (interface, condition)
        for interface in Interface
        if interface is not Interface.I1_SOFT_REPORT
        for condition in CueCondition
    }
    cells.add((Interface.I1_SOFT_REPORT, CueCondition.NO_CUE))
    return frozenset(cells)


def _validate_soft_report(payload: ImmutablePayload | None) -> None:
    if payload is None:
        raise RuntimeError("I1 diagnostic top-k payload is required")
    top_k = payload.get("top_k")
    positions = payload.get("positions")
    format_valid = payload.get("output_format_valid")
    domain_valid = payload.get("numeric_domain_valid")
    raw_output = payload.get("raw_output")
    if (
        type(top_k) is not int
        or top_k <= 0
        or not isinstance(positions, tuple)
        or not isinstance(format_valid, bool)
        or not isinstance(domain_valid, bool)
        or not isinstance(raw_output, str)
    ):
        raise RuntimeError("I1 diagnostic top-k payload is malformed")
    if not format_valid:
        if domain_valid or positions:
            raise RuntimeError("I1 invalid-format diagnostic payload is malformed")
        return
    if len(positions) != 4:
        raise RuntimeError("I1 diagnostic payload must cover four positions")
    for expected_index, position in enumerate(positions):
        if not isinstance(position, Mapping) or position.get("index") != expected_index:
            raise RuntimeError("I1 diagnostic payload position indices drifted")
        candidates = position.get("candidates")
        if not isinstance(candidates, tuple) or len(candidates) != top_k:
            raise RuntimeError("I1 diagnostic top-k candidates are malformed")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise RuntimeError("I1 diagnostic candidate must be a mapping")
            value = candidate.get("value")
            relative_logit = candidate.get("relative_logit")
            if (
                type(value) is not int
                or isinstance(relative_logit, bool)
                or not isinstance(relative_logit, (int, float))
            ):
                raise RuntimeError("I1 diagnostic candidate values or logits are malformed")
            if not math.isfinite(float(relative_logit)):
                raise RuntimeError("I1 diagnostic relative logits must be finite")


def validate_interface_ladder_records(
    records: Iterable[InterfaceLadderRecord],
    *,
    expected_scenes: int,
    expected_conditions: int,
    expected_interfaces: int,
    expected_source_sha256: Mapping[str, str],
) -> tuple[InterfaceLadderRecord, ...]:
    """Fail closed on cell, provenance, or eligibility drift."""

    if type(expected_scenes) is not int or expected_scenes <= 0:
        raise ValueError("expected_scenes must be a positive integer")
    if expected_conditions != len(CueCondition) or expected_interfaces != len(Interface):
        raise RuntimeError("S6 expected condition or interface names drifted")
    source_hashes = dict(expected_source_sha256)
    if set(source_hashes) != {stage for stage, _branch in _SOURCE_CONTRACT.values()}:
        raise RuntimeError("S6 expected source hash stages drifted")
    for stage, digest in source_hashes.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"S6 source SHA-256 is invalid for {stage}")

    frozen = tuple(records)
    if not frozen or any(not isinstance(row, InterfaceLadderRecord) for row in frozen):
        raise TypeError("S6 records must be non-empty InterfaceLadderRecord values")
    call_ids = [row.call_id for row in frozen]
    cell_keys = [(row.scene_id, row.interface, row.condition) for row in frozen]
    if len(call_ids) != len(set(call_ids)) or len(cell_keys) != len(set(cell_keys)):
        raise RuntimeError("S6 records contain duplicate call or cell identifiers")

    required = _required_cells()
    by_scene: dict[str, list[InterfaceLadderRecord]] = {}
    for row in frozen:
        by_scene.setdefault(row.scene_id, []).append(row)
        expected_stage, expected_branch = _SOURCE_CONTRACT[row.interface]
        if row.source_stage != expected_stage:
            raise RuntimeError(
                f"S6 {row.interface.value} source stage must use its registered runtime"
            )
        if row.source_branch != expected_branch:
            raise RuntimeError(f"S6 {row.interface.value} source branch must be {expected_branch}")
        if row.source_artifact_sha256 != source_hashes[row.source_stage]:
            raise RuntimeError(f"S6 source SHA/hash drifted for {row.call_id}")
        if not row.structural_validity_verified:
            raise RuntimeError(f"S6 structural validity failed for {row.call_id}")
        if row.interface is Interface.I1_SOFT_REPORT:
            if row.condition is not CueCondition.NO_CUE:
                raise RuntimeError("I1 soft-report diagnostic is defined only for no_cue")
            _validate_soft_report(row.diagnostic_payload)
            if row.output_world is not None or row.parse_success is not None:
                raise RuntimeError("I1 diagnostic must not claim a parsed output world")
        elif row.parse_success is None:
            raise RuntimeError(f"S6 parse status is required for {row.interface.value}")

        if row.interface in _INTERVENTION_INTERFACES:
            if row.primary_eligible or not row.diagnostic_only:
                raise RuntimeError("I1/I2 intervention diagnostic cells cannot be primary")
            if row.diagnostic_reason is None:
                raise RuntimeError("intervention diagnostic cells require a diagnostic reason")
        elif row.interface is Interface.I4_EXACT_CACHE:
            if row.primary_eligible == row.diagnostic_only:
                raise RuntimeError("I4 eligible/diagnostic flags must be objective complements")
            if row.diagnostic_only and row.diagnostic_reason != "token_divergence":
                raise RuntimeError("I4 diagnostic cells require token_divergence evidence")
            if row.primary_eligible and row.diagnostic_reason is not None:
                raise RuntimeError("I4 eligible cells cannot carry a diagnostic reason")
        elif not row.primary_eligible or row.diagnostic_only or row.diagnostic_reason is not None:
            raise RuntimeError(f"S6 {row.interface.value} primary eligibility drifted")

    if len(by_scene) != expected_scenes:
        raise RuntimeError(
            f"S6 scene count drifted: expected {expected_scenes}, observed {len(by_scene)}"
        )
    incomplete: list[str] = []
    for scene_id, scene_rows in by_scene.items():
        cells = {(row.interface, row.condition) for row in scene_rows}
        if cells != required:
            incomplete.append(scene_id)
            continue
        first = scene_rows[0]
        if any(
            (
                row.family,
                row.true_world,
                row.observed_world,
                row.counterfactual_world,
            )
            != (
                first.family,
                first.true_world,
                first.observed_world,
                first.counterfactual_world,
            )
            for row in scene_rows[1:]
        ):
            raise RuntimeError(f"S6 scene metadata drifted for {scene_id}")
    if incomplete:
        raise RuntimeError(f"S6 interface ladder has missing or extra cells: {incomplete[:5]}")
    return tuple(sorted(frozen, key=lambda row: row.call_id))


def _interval(
    values: Mapping[str, float], *, bootstrap_resamples: int, seed: int
) -> dict[str, float | int]:
    rows = ({"scene_id": scene_id, "value": value} for scene_id, value in sorted(values.items()))
    result = scene_clustered_bootstrap_ci(
        rows,
        metric="value",
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    return {
        "estimate": result.estimate,
        "ci_low": result.low,
        "ci_high": result.high,
        "confidence": result.confidence,
        "number_of_scenes": result.number_of_scenes,
    }


def _cell_summary(
    rows: Iterable[InterfaceLadderRecord],
    *,
    eligible_scene_ids: frozenset[str],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, object]:
    frozen = tuple(rows)
    primary = tuple(
        row
        for row in frozen
        if row.interface in _PRIMARY_INTERFACES and row.scene_id in eligible_scene_ids
    )
    result: dict[str, object] = {
        "number_of_cells": len(frozen),
        "number_of_source_scenes": len({row.scene_id for row in frozen}),
        "primary_analysis_cell_count": len(primary),
        "diagnostic_only_cell_count": sum(row.diagnostic_only for row in frozen),
        "parse_success_count": sum(row.parse_success is True for row in frozen),
    }
    if primary:
        exact = _scene_means(primary, lambda row: row.exact_world_recovery)
        copied = _scene_means(primary, lambda row: row.observation_copy)
        result["exact_world_recovery"] = _interval(
            exact, bootstrap_resamples=bootstrap_resamples, seed=seed
        )
        result["observation_copy"] = _interval(
            copied, bootstrap_resamples=bootstrap_resamples, seed=seed + 1
        )
        counterfactual = _scene_means(
            (row for row in primary if row.condition is CueCondition.COUNTERFACTUAL_CUE),
            lambda row: bool(row.counterfactual_compliance),
        )
        if counterfactual:
            result["counterfactual_compliance"] = _interval(
                counterfactual,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + 2,
            )
    else:
        result["exact_world_recovery"] = None
        result["observation_copy"] = None
    return result


def _scene_means(
    rows: Iterable[InterfaceLadderRecord],
    metric: Callable[[InterfaceLadderRecord], bool],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.scene_id, []).append(float(metric(row)))
    return {scene_id: sum(values) / len(values) for scene_id, values in grouped.items()}


def summarize_interface_ladder(
    records: Iterable[InterfaceLadderRecord],
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Summarize objective strata and plan-defined effects at scene level."""

    unvalidated = tuple(records)
    observed_source_sha256: dict[str, str] = {}
    for row in unvalidated:
        previous = observed_source_sha256.setdefault(row.source_stage, row.source_artifact_sha256)
        if previous != row.source_artifact_sha256:
            raise RuntimeError(f"S6 source SHA/hash drifted within {row.source_stage}")
    rows = validate_interface_ladder_records(
        unvalidated,
        expected_scenes=len({row.scene_id for row in unvalidated}),
        expected_conditions=len(CueCondition),
        expected_interfaces=len(Interface),
        expected_source_sha256=observed_source_sha256,
    )
    if not rows:
        raise ValueError("S6 summary requires records")
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    scene_ids = frozenset(row.scene_id for row in rows)
    divergent_scene_ids = frozenset(
        row.scene_id
        for row in rows
        if row.interface is Interface.I4_EXACT_CACHE and row.diagnostic_only
    )
    eligible_scene_ids = scene_ids - divergent_scene_ids
    if not eligible_scene_ids:
        raise RuntimeError("S6 has no complete-case primary scene pairs")

    def grouped(field: str) -> dict[str, object]:
        if field == "interface":
            values = tuple(Interface)
            key = lambda row: row.interface  # noqa: E731
        elif field == "condition":
            values = tuple(CueCondition)
            key = lambda row: row.condition  # noqa: E731
        else:
            values = tuple(sorted({row.family for row in rows}))
            key = lambda row: row.family  # noqa: E731
        return {
            value.value if isinstance(value, (Interface, CueCondition)) else value: _cell_summary(
                (row for row in rows if key(row) == value),
                eligible_scene_ids=eligible_scene_ids,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )
            for value in values
        }

    by_interface: dict[str, object] = {}
    for interface in Interface:
        interface_rows = tuple(row for row in rows if row.interface is interface)
        item = _cell_summary(
            interface_rows,
            eligible_scene_ids=eligible_scene_ids,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        item["claim_family"] = interface_claim_name(interface)
        item["by_cue_condition"] = {
            condition.value: _cell_summary(
                (row for row in interface_rows if row.condition is condition),
                eligible_scene_ids=eligible_scene_ids,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )
            for condition in CueCondition
            if interface is not Interface.I1_SOFT_REPORT or condition is CueCondition.NO_CUE
        }
        by_interface[interface.value] = item

    indexed = {(row.scene_id, row.interface, row.condition): row for row in rows}
    spontaneous = {
        scene_id: float(
            indexed[(scene_id, Interface.I4_EXACT_CACHE, CueCondition.NO_CUE)].exact_world_recovery
        )
        - float(
            indexed[(scene_id, Interface.I0_HARD_TEXT, CueCondition.NO_CUE)].exact_world_recovery
        )
        for scene_id in eligible_scene_ids
    }
    fact_conditioned = {
        scene_id: float(
            indexed[
                (scene_id, Interface.I4_EXACT_CACHE, CueCondition.VALID_CUE)
            ].exact_world_recovery
        )
        - float(
            indexed[(scene_id, Interface.I4_EXACT_CACHE, CueCondition.NO_CUE)].exact_world_recovery
        )
        for scene_id in eligible_scene_ids
    }
    i4_rows = tuple(row for row in rows if row.interface is Interface.I4_EXACT_CACHE)
    i4_diagnostics = sorted(row.call_id for row in i4_rows if row.diagnostic_only)
    source_hashes = {
        row.source_stage: row.source_artifact_sha256
        for row in sorted(rows, key=lambda row: row.source_stage)
    }
    source_stage_counts = Counter(row.source_stage for row in rows)
    return {
        "schema_version": 1,
        "status": "PHASE_3_INTERFACE_LADDER_EXECUTED_WITH_DIAGNOSTICS",
        "number_of_source_scenes": len(scene_ids),
        "number_of_cells": len(rows),
        "primary_paired_scene_count": len(eligible_scene_ids),
        "excluded_primary_scene_ids": sorted(divergent_scene_ids),
        "primary_analysis_cell_count": len(eligible_scene_ids) * 12,
        "intervention_diagnostic_cell_count": sum(
            row.interface in _INTERVENTION_INTERFACES for row in rows
        ),
        "i4_exact_eligible_call_count": sum(row.primary_eligible for row in i4_rows),
        "i4_token_diagnostic_call_count": len(i4_diagnostics),
        "diagnostic_call_ids": i4_diagnostics,
        "source_artifact_sha256": source_hashes,
        "source_stage_cell_counts": dict(sorted(source_stage_counts.items())),
        "by_interface": by_interface,
        "by_cue_condition": grouped("condition"),
        "by_family": grouped("family"),
        "effects": {
            "spontaneous_visual_revision": _interval(
                spontaneous,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + 101,
            ),
            "fact_conditioned_revision": _interval(
                fact_conditioned,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + 102,
            ),
        },
        "scene_is_statistical_unit": True,
        "generation_invoked": any(row.source_stage == "S6_runtime" for row in rows),
        "training_invoked": False,
        "rl_invoked": False,
        "subjective_success_threshold_applied": False,
    }


def write_interface_ladder_outputs(
    per_scene_path: Path,
    summary_path: Path,
    *,
    records: Iterable[InterfaceLadderRecord],
    summary: Mapping[str, object],
) -> None:
    """Write deterministic S6 outputs, refusing any overwrite."""

    per_scene = Path(per_scene_path)
    summary_file = Path(summary_path)
    if per_scene == summary_file:
        raise ValueError("S6 output paths must be distinct")
    existing = [str(path) for path in (per_scene, summary_file) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite S6 outputs: {existing}")
    frozen = tuple(records)
    if not frozen or not isinstance(summary, Mapping):
        raise ValueError("S6 records and summary are required")
    per_scene.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    per_scene_text = "".join(
        json.dumps(row.to_mapping(), separators=(",", ":"), sort_keys=True) + "\n" for row in frozen
    )
    summary_text = json.dumps(_plain(summary), indent=2, sort_keys=True) + "\n"
    per_scene_created = False
    summary_created = False
    complete = False
    try:
        with per_scene.open("x", encoding="utf-8") as stream:
            per_scene_created = True
            stream.write(per_scene_text)
        with summary_file.open("x", encoding="utf-8") as stream:
            summary_created = True
            stream.write(summary_text)
        complete = True
    except FileExistsError as error:  # pragma: no cover - filesystem race
        raise FileExistsError("refusing to overwrite S6 outputs") from error
    finally:
        if not complete:  # pragma: no cover - defensive partial-write cleanup
            if per_scene_created:
                per_scene.unlink(missing_ok=True)
            if summary_created:
                summary_file.unlink(missing_ok=True)


__all__ = [
    "InterfaceLadderRecord",
    "summarize_interface_ladder",
    "validate_interface_ladder_records",
    "write_interface_ladder_outputs",
]
