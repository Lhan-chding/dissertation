"""Deterministic, split-isolated source preparation for Phase 4 support data.

The natural examples in Phase 4 must be observations made by the frozen base
model.  The v4 plan permits the completed legacy-diagnostic S6 I1 records as
an SFT source, while permanently excluding future confirmatory scenes.  This
module contains only pure selection and filtering logic.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from compbias.recoverability.compatibility import (
    ArithmeticProgressionConstraint,
    KnownValueConstraint,
    PairSumConstraint,
)
from compbias.recoverability.phase_c_screen import build_family_constraints
from compensability_v4.data.splits import DatasetSplit
from compensability_v4.data.v4_generator import generate_v4_scenes
from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.schemas.scene import RecoveryScene

_FAMILIES = frozenset({"cross_series", "duplicate_encoding", "trend"})
_S6_I1 = "I1_soft_report_diagnostic"
_S6_NO_CUE = "no_cue"
_S6_STAGE = "S6_runtime"
_S6_BRANCH = "stage1_soft_report_runtime"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WORLD = re.compile(r"\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\Z")
_S6_STATUS = "PHASE_3_INTERFACE_LADDER_EXECUTED_WITH_DIAGNOSTICS"
_S6_CELLS = frozenset(
    {
        (interface, cue)
        for interface in (
            "I0_hard_text_symbolic_recovery",
            "I2_candidate_world_diagnostic",
            "I3_same_conversation_visual_revision",
            "I4_exact_cached_natural_continuation",
        )
        for cue in ("no_cue", "valid_cue", "sham_cue", "counterfactual_cue")
    }
    | {(_S6_I1, _S6_NO_CUE)}
)
_PREPARED_HASH_KEYS = frozenset(
    {"s6_per_scene", "s6_summary", "dataset_manifest", "dataset_records"}
)


@dataclass(frozen=True, slots=True)
class NaturalObservationCapture:
    """One frozen-base Stage-1 observation with the visual-state evidence."""

    scene_id: str
    raw_output: str
    image_grid_thw: tuple[int, int, int]
    visual_token_count: int


@dataclass(frozen=True, slots=True)
class PreparedSupportSources:
    """Fully validated in-memory Phase 4 inputs derived from frozen S6 evidence."""

    symbolic_scenes: tuple[RecoveryScene, ...]
    natural_scenes: tuple[RecoveryScene, ...]
    natural_observations: tuple[NaturalObservation, ...]
    selection_traces: tuple[Mapping[str, object], ...]
    selection_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.symbolic_scenes or not self.natural_scenes or not self.natural_observations:
            raise ValueError("Phase 4 prepared support sources must be non-empty")
        if len(self.natural_scenes) != len(self.natural_observations):
            raise ValueError("Phase 4 prepared natural source pairing drifted")
        counts = dict(self.selection_counts)
        if not counts or any(
            not isinstance(key, str) or type(value) is not int for key, value in counts.items()
        ):
            raise ValueError("Phase 4 prepared selection counts are malformed")
        object.__setattr__(self, "selection_counts", MappingProxyType(dict(sorted(counts.items()))))


@dataclass(frozen=True, slots=True)
class PreparedSourcePaths:
    """Canonical files emitted by the autonomous Phase 4 preparation step."""

    symbolic_scenes: Path
    natural_scenes: Path
    natural_observations: Path
    selection_trace: Path
    summary: Path


def _world(value: object, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"Phase 4 {label} must contain exactly four integers")
    return tuple(value)  # type: ignore[return-value]


def _fact_mapping(constraint: object) -> Mapping[str, object]:
    if isinstance(constraint, PairSumConstraint):
        return MappingProxyType(
            {
                "type": "pair_sum",
                "left_index": constraint.left_index,
                "right_index": constraint.right_index,
                "total": constraint.total,
                "fact_id": constraint.constraint_id,
            }
        )
    if isinstance(constraint, KnownValueConstraint):
        return MappingProxyType(
            {
                "type": "known_value",
                "index": constraint.index,
                "value": constraint.value,
                "fact_id": constraint.constraint_id,
            }
        )
    if isinstance(constraint, ArithmeticProgressionConstraint):
        return MappingProxyType(
            {
                "type": "arithmetic_progression",
                "indices": constraint.indices,
                "fact_id": constraint.constraint_id,
            }
        )
    raise TypeError("Phase 4 natural scene uses an unregistered constraint")


def _safe_image_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Phase 4 natural image path must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".png":
        raise ValueError("Phase 4 natural image path must be a safe relative PNG")
    return value


def _record_to_scene(record: Mapping[str, object]) -> RecoveryScene:
    scene_id = record.get("scene_id")
    family = record.get("family")
    if not isinstance(scene_id, str) or not scene_id or scene_id.strip() != scene_id:
        raise ValueError("Phase 4 natural source scene_id is invalid")
    if family not in _FAMILIES:
        raise ValueError("Phase 4 natural source family is invalid")
    truth = _world(record.get("values"), "natural source values")
    facts = tuple(_fact_mapping(item) for item in build_family_constraints(family, truth))
    return RecoveryScene(
        scene_id=scene_id,
        split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
        semantic_scene_id=f"phase4-semantic-{scene_id}",
        numeric_table_id=f"phase4-numbers-{scene_id}",
        constraint_graph_id=f"phase4-graph-{scene_id}",
        truth=truth,
        facts=facts,
        resized_height=280,
        resized_width=280,
        image_path=_safe_image_path(record.get("image")),
    )


def _stable_rank(seed: int, scene_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{scene_id}".encode()).digest()


def build_independent_natural_scenes(
    records: Iterable[Mapping[str, object]],
    *,
    confirm_scene_ids: frozenset[str],
    candidate_cap: int,
    selection_seed: int,
) -> tuple[RecoveryScene, ...]:
    """Select a deterministic independent pool for a future non-legacy run."""

    if isinstance(candidate_cap, bool) or not isinstance(candidate_cap, int) or candidate_cap <= 0:
        raise ValueError("Phase 4 natural candidate_cap must be a positive integer")
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise TypeError("Phase 4 natural selection_seed must be an integer")
    if any(not isinstance(scene_id, str) or not scene_id for scene_id in confirm_scene_ids):
        raise ValueError("Phase 4 confirm scene identifiers are malformed")
    scene_rows: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Phase 4 natural source records must be mappings")
        scene_id = record.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("Phase 4 natural source record has no scene_id")
        if scene_id in scene_rows:
            raise ValueError("Phase 4 natural source scene identifiers are not unique")
        scene_rows[scene_id] = record
    eligible = tuple(
        record for scene_id, record in scene_rows.items() if scene_id not in confirm_scene_ids
    )
    if len(eligible) < candidate_cap:
        raise RuntimeError("Phase 4 independent natural source pool is smaller than candidate_cap")
    selected = sorted(
        eligible,
        key=lambda record: _stable_rank(selection_seed, str(record["scene_id"])),
    )[:candidate_cap]
    scenes = tuple(_record_to_scene(record) for record in selected)
    if any(scene.scene_id in confirm_scene_ids for scene in scenes):
        raise AssertionError("Phase 4 natural selection leaked a protected scene")
    return scenes


def build_legacy_s6_natural_candidates(
    records: Iterable[Mapping[str, object]],
    *,
    image_paths: Mapping[str, str],
    image_grid_thw: tuple[int, int, int],
) -> tuple[tuple[RecoveryScene, ...], tuple[NaturalObservationCapture, ...]]:
    """Reconstruct Phase 4 candidates from frozen S6 I1 legacy observations.

    S6 deliberately stores I1 as a diagnostic payload rather than a primary
    parsed-world cell.  This function preserves the raw frozen output and
    recreates only the immutable scene contract needed for Phase 4's later
    one-error filter.
    """

    if (
        not isinstance(image_grid_thw, tuple)
        or len(image_grid_thw) != 3
        or any(type(item) is not int or item <= 0 for item in image_grid_thw)
        or image_grid_thw[1] % 2
        or image_grid_thw[2] % 2
    ):
        raise ValueError("Phase 4 legacy S6 image_grid_thw is invalid")
    visual_token_count = image_grid_thw[0] * image_grid_thw[1] * image_grid_thw[2] // 4
    candidates: list[tuple[RecoveryScene, NaturalObservationCapture]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Phase 4 legacy S6 records must be mappings")
        if record.get("interface") != _S6_I1:
            continue
        if (
            record.get("cue_condition") != _S6_NO_CUE
            or record.get("source_stage") != _S6_STAGE
            or record.get("source_branch") != _S6_BRANCH
        ):
            raise RuntimeError("Phase 4 legacy S6 I1 provenance drifted")
        scene_id = record.get("scene_id")
        family = record.get("family")
        payload = record.get("diagnostic_payload")
        if (
            not isinstance(scene_id, str)
            or scene_id in seen
            or family not in _FAMILIES
            or not isinstance(payload, Mapping)
            or not isinstance(payload.get("raw_output"), str)
        ):
            raise RuntimeError("Phase 4 legacy S6 I1 record is malformed")
        image_path = image_paths.get(scene_id)
        if image_path is None:
            raise RuntimeError("Phase 4 legacy S6 scene is missing its frozen image path")
        scene = _record_to_scene(
            {
                "scene_id": scene_id,
                "family": family,
                "values": record.get("true_world"),
                "image": image_path,
            }
        )
        capture = _validated_capture(
            NaturalObservationCapture(
                scene_id=scene_id,
                raw_output=str(payload["raw_output"]),
                image_grid_thw=image_grid_thw,
                visual_token_count=visual_token_count,
            )
        )
        seen.add(scene_id)
        candidates.append((scene, capture))
    if not candidates:
        raise RuntimeError("Phase 4 legacy S6 source contains no I1 observations")
    candidates.sort(key=lambda item: item[0].scene_id)
    return tuple(item[0] for item in candidates), tuple(item[1] for item in candidates)


def _parse_world(raw_output: object) -> tuple[int, int, int, int] | None:
    match = _WORLD.fullmatch(raw_output) if isinstance(raw_output, str) else None
    if match is None:
        return None
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _validated_capture(capture: NaturalObservationCapture) -> NaturalObservationCapture:
    if not isinstance(capture, NaturalObservationCapture):
        raise TypeError("Phase 4 captures must be NaturalObservationCapture records")
    if not isinstance(capture.scene_id, str) or not capture.scene_id:
        raise ValueError("Phase 4 capture scene_id is invalid")
    if not isinstance(capture.raw_output, str):
        raise TypeError("Phase 4 capture raw_output must be a string")
    grid = capture.image_grid_thw
    if (
        not isinstance(grid, tuple)
        or len(grid) != 3
        or any(type(item) is not int or item <= 0 for item in grid)
        or grid[1] % 2
        or grid[2] % 2
    ):
        raise ValueError("Phase 4 capture image_grid_thw is invalid")
    if type(capture.visual_token_count) is not int or capture.visual_token_count <= 0:
        raise ValueError("Phase 4 capture visual_token_count is invalid")
    if capture.visual_token_count != grid[0] * grid[1] * grid[2] // 4:
        raise ValueError("Phase 4 capture visual_token_count does not match image_grid_thw")
    return capture


def retain_natural_single_error_scenes(
    scenes: Iterable[RecoveryScene],
    captures: Iterable[NaturalObservationCapture],
    *,
    model_snapshot_sha256: str,
    target_count: int | None,
    value_domain: Iterable[int] = range(2, 19),
) -> tuple[
    tuple[RecoveryScene, ...],
    tuple[NaturalObservation, ...],
    tuple[Mapping[str, object], ...],
]:
    """Keep exactly-one-error frozen Stage-1 observations and audit every candidate."""

    if _SHA256.fullmatch(model_snapshot_sha256) is None:
        raise ValueError("Phase 4 model snapshot SHA-256 is invalid")
    if target_count is not None and (
        isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 0
    ):
        raise ValueError("Phase 4 natural target_count must be a nonnegative integer or null")
    domain = tuple(sorted(set(value_domain)))
    if not domain or any(type(value) is not int for value in domain):
        raise ValueError("Phase 4 frozen value domain must contain integers")
    natural_scenes = tuple(scenes)
    if not natural_scenes:
        raise ValueError("Phase 4 natural candidate scenes are empty")
    scene_index: dict[str, RecoveryScene] = {}
    for scene in natural_scenes:
        if not isinstance(scene, RecoveryScene):
            raise TypeError("Phase 4 natural scenes must be RecoveryScene records")
        if scene.split is not DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN:
            raise ValueError("Phase 4 natural candidates must use natural_error_support_train")
        if scene.scene_id in scene_index:
            raise ValueError("Phase 4 natural candidate scene identifiers are duplicated")
        if any(value not in domain for value in scene.truth):
            raise ValueError("Phase 4 natural candidate truth lies outside the frozen value domain")
        scene_index[scene.scene_id] = scene
    capture_index: dict[str, NaturalObservationCapture] = {}
    for capture in captures:
        validated = _validated_capture(capture)
        if validated.scene_id in capture_index:
            raise ValueError("Phase 4 natural capture scene identifiers are duplicated")
        capture_index[validated.scene_id] = validated
    if set(capture_index) != set(scene_index):
        raise ValueError("Phase 4 natural candidate and capture identifiers differ")

    retained_scenes: list[RecoveryScene] = []
    observations: list[NaturalObservation] = []
    traces: list[Mapping[str, object]] = []
    for scene in natural_scenes:
        capture = capture_index[scene.scene_id]
        observed = _parse_world(capture.raw_output)
        trace: dict[str, object] = {
            "scene_id": scene.scene_id,
            "raw_output": capture.raw_output,
            "image_grid_thw": list(capture.image_grid_thw),
            "visual_token_count": capture.visual_token_count,
            "parsed_values": list(observed) if observed is not None else None,
        }
        if observed is None:
            trace["selection_status"] = "rejected_unparseable"
        elif any(value not in domain for value in observed):
            trace["selection_status"] = "rejected_outside_frozen_domain"
        else:
            mismatches = tuple(
                index
                for index, (truth, value) in enumerate(zip(scene.truth, observed, strict=True))
                if truth != value
            )
            if not mismatches:
                trace["selection_status"] = "rejected_zero_error"
            elif len(mismatches) != 1:
                trace["selection_status"] = "rejected_multiple_errors"
                trace["mismatch_indices"] = list(mismatches)
            else:
                error_index = mismatches[0]
                trace["selection_status"] = "accepted_single_error"
                trace["error_index"] = error_index
                if target_count is None or len(retained_scenes) < target_count:
                    retained_scenes.append(scene)
                    observations.append(
                        NaturalObservation(
                            observation_id=f"phase4-stage1-{scene.scene_id}",
                            scene_id=scene.scene_id,
                            observed_values=observed,
                            error_index=error_index,
                            stage1_model_hash=model_snapshot_sha256,
                            image_grid_thw=capture.image_grid_thw,
                            visual_token_count=capture.visual_token_count,
                        )
                    )
                    trace["retained"] = True
                else:
                    trace["retained"] = False
        traces.append(MappingProxyType(trace))
    if target_count is not None and len(retained_scenes) != target_count:
        raise RuntimeError(
            "Phase 4 natural single-error target not met: "
            f"required={target_count}, observed={len(retained_scenes)}"
        )
    if target_count is None and not retained_scenes:
        raise RuntimeError("Phase 4 legacy S6 source contains no natural single-error observations")
    return tuple(retained_scenes), tuple(observations), tuple(traces)


def _validate_interface_summary(
    summary: Mapping[str, object], *, expected_scenes: int, model_snapshot_sha256: str
) -> None:
    if (
        summary.get("status") != _S6_STATUS
        or summary.get("number_of_source_scenes") != expected_scenes
        or summary.get("number_of_cells") != expected_scenes * len(_S6_CELLS)
        or summary.get("model_snapshot_sha256") != model_snapshot_sha256
        or summary.get("training_invoked") is not False
        or summary.get("rl_invoked") is not False
        or summary.get("subjective_success_threshold_applied") is not False
    ):
        raise RuntimeError("Phase 4 S6 summary contract drifted")


def _validated_s6_i1_records(
    records: Iterable[Mapping[str, object]], *, expected_scenes: int
) -> tuple[Mapping[str, object], ...]:
    rows = tuple(records)
    if len(rows) != expected_scenes * len(_S6_CELLS):
        raise RuntimeError("Phase 4 S6 17-cell closure drifted")
    call_ids: set[str] = set()
    by_scene: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Phase 4 S6 records must be mappings")
        call_id, scene_id = row.get("call_id"), row.get("scene_id")
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id in call_ids
            or not isinstance(scene_id, str)
            or not scene_id
        ):
            raise RuntimeError("Phase 4 S6 call/scene identifiers are malformed")
        call_ids.add(call_id)
        by_scene.setdefault(scene_id, []).append(row)
    if len(by_scene) != expected_scenes:
        raise RuntimeError("Phase 4 S6 scene closure drifted")
    i1_rows: list[Mapping[str, object]] = []
    for scene_id, scene_rows in by_scene.items():
        cells = {(row.get("interface"), row.get("cue_condition")) for row in scene_rows}
        if len(scene_rows) != len(_S6_CELLS) or cells != _S6_CELLS:
            raise RuntimeError("Phase 4 S6 17-cell closure drifted")
        families = {row.get("family") for row in scene_rows}
        worlds = {tuple(row.get("true_world", ())) for row in scene_rows}
        if len(families) != 1 or len(worlds) != 1:
            raise RuntimeError(f"Phase 4 S6 scene semantics drifted: {scene_id}")
        i1_rows.append(next(row for row in scene_rows if row.get("interface") == _S6_I1))
    return tuple(sorted(i1_rows, key=lambda row: str(row["scene_id"])))


def _dataset_image_paths(
    records: Iterable[Mapping[str, object]], *, i1_records: Sequence[Mapping[str, object]]
) -> dict[str, str]:
    required = {str(row["scene_id"]): row for row in i1_records}
    selected: dict[str, str] = {}
    seen: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("Phase 4 visual dataset records must be mappings")
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id or scene_id in seen:
            raise RuntimeError("Phase 4 visual dataset scene identifiers are malformed")
        seen.add(scene_id)
        source = required.get(scene_id)
        if source is None:
            continue
        if row.get("family") != source.get("family") or _world(
            row.get("values"), "visual dataset values"
        ) != _world(source.get("true_world"), "S6 true world"):
            raise RuntimeError("Phase 4 visual dataset differs from S6 scene semantics")
        selected[scene_id] = _safe_image_path(row.get("image"))
    if set(selected) != set(required):
        raise RuntimeError("Phase 4 visual dataset is missing an S6 source scene")
    return selected


def prepare_legacy_s6_support_sources(
    *,
    interface_records: Iterable[Mapping[str, object]],
    interface_summary: Mapping[str, object],
    dataset_records: Iterable[Mapping[str, object]],
    model_snapshot_sha256: str,
    expected_scenes: int,
    symbolic_scene_count: int,
    symbolic_seed: int,
    value_domain: Iterable[int],
    image_grid_thw: tuple[int, int, int],
) -> PreparedSupportSources:
    """Derive all Phase 4 sources from completed S6 and deterministic CPU generation."""

    if type(expected_scenes) is not int or expected_scenes <= 0:
        raise ValueError("Phase 4 expected_scenes must be a positive integer")
    if _SHA256.fullmatch(model_snapshot_sha256) is None:
        raise ValueError("Phase 4 model snapshot SHA-256 is invalid")
    if not isinstance(interface_summary, Mapping):
        raise TypeError("Phase 4 S6 summary must be a mapping")
    _validate_interface_summary(
        interface_summary,
        expected_scenes=expected_scenes,
        model_snapshot_sha256=model_snapshot_sha256,
    )
    i1_records = _validated_s6_i1_records(interface_records, expected_scenes=expected_scenes)
    image_paths = _dataset_image_paths(dataset_records, i1_records=i1_records)
    candidates, captures = build_legacy_s6_natural_candidates(
        i1_records,
        image_paths=image_paths,
        image_grid_thw=image_grid_thw,
    )
    natural, observations, traces = retain_natural_single_error_scenes(
        candidates,
        captures,
        model_snapshot_sha256=model_snapshot_sha256,
        target_count=None,
        value_domain=value_domain,
    )
    symbolic = generate_v4_scenes(
        count=symbolic_scene_count,
        seed=symbolic_seed,
        split=DatasetSplit.SYMBOLIC_SUPPORT_TRAIN,
        value_domain=value_domain,
    )
    if {scene.scene_id for scene in symbolic} & {scene.scene_id for scene in natural}:
        raise RuntimeError("Phase 4 symbolic and natural source identifiers overlap")
    counts = Counter(str(trace["selection_status"]) for trace in traces)
    return PreparedSupportSources(
        symbolic_scenes=symbolic,
        natural_scenes=natural,
        natural_observations=observations,
        selection_traces=traces,
        selection_counts=counts,
    )


def _prepared_paths(output_root: Path) -> PreparedSourcePaths:
    root = Path(output_root)
    return PreparedSourcePaths(
        symbolic_scenes=root / "symbolic_scenes.jsonl",
        natural_scenes=root / "natural_scenes.jsonl",
        natural_observations=root / "natural_observations.jsonl",
        selection_trace=root / "selection_trace.jsonl",
        summary=root / "source_summary.json",
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_text(rows: Iterable[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(dict(row), separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def write_prepared_support_sources(
    *,
    output_root: Path,
    prepared: PreparedSupportSources,
    source_hashes: Mapping[str, str],
) -> PreparedSourcePaths:
    """Publish deterministic Phase 4 source files and their complete provenance."""

    if not isinstance(prepared, PreparedSupportSources):
        raise TypeError("prepared must be a PreparedSupportSources record")
    hashes = dict(source_hashes)
    if set(hashes) != _PREPARED_HASH_KEYS or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in hashes.values()
    ):
        raise ValueError("Phase 4 prepared input hashes are malformed")
    paths = _prepared_paths(output_root)
    files = tuple(getattr(paths, name) for name in paths.__dataclass_fields__)
    if Path(output_root).is_symlink() or any(path.exists() or path.is_symlink() for path in files):
        raise FileExistsError("refusing to overwrite Phase 4 prepared sources")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    contents = {
        paths.symbolic_scenes: _jsonl_text(
            scene.to_mapping() for scene in prepared.symbolic_scenes
        ),
        paths.natural_scenes: _jsonl_text(scene.to_mapping() for scene in prepared.natural_scenes),
        paths.natural_observations: _jsonl_text(
            observation.to_mapping() for observation in prepared.natural_observations
        ),
        paths.selection_trace: _jsonl_text(prepared.selection_traces),
    }
    created: list[Path] = []
    complete = False
    try:
        for path, text in contents.items():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
            created.append(path)
        output_hashes = {
            "symbolic_scenes": _sha256_path(paths.symbolic_scenes),
            "natural_scenes": _sha256_path(paths.natural_scenes),
            "natural_observations": _sha256_path(paths.natural_observations),
            "selection_trace": _sha256_path(paths.selection_trace),
        }
        summary = {
            "schema_version": 1,
            "artifact_type": "phase_4_prepared_support_sources",
            "status": "PHASE_4_SUPPORT_SOURCES_PREPARED_FROM_FROZEN_S6",
            "contains_confirmatory_data": False,
            "counts": {
                "symbolic_scenes": len(prepared.symbolic_scenes),
                "natural_single_error_scenes": len(prepared.natural_scenes),
                "natural_observations": len(prepared.natural_observations),
                "selection_candidates": len(prepared.selection_traces),
                "selection_status": dict(prepared.selection_counts),
            },
            "source_hashes": dict(sorted(hashes.items())),
            "output_hashes": output_hashes,
        }
        with paths.summary.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        created.append(paths.summary)
        complete = True
    finally:
        if not complete:  # pragma: no cover - defensive filesystem cleanup
            for path in created:
                path.unlink(missing_ok=True)
    return paths


def validate_prepared_source_summary(
    summary_path: Path, *, paths: PreparedSourcePaths | None = None
) -> dict[str, str]:
    """Verify prepared source closure and return hashes accepted by the support builder."""

    summary_file = Path(summary_path)
    if summary_file.is_symlink() or not summary_file.is_file():
        raise RuntimeError("Phase 4 prepared source summary is missing")
    prepared_paths = paths or _prepared_paths(summary_file.parent)
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    output_hashes = payload.get("output_hashes") if isinstance(payload, dict) else None
    counts = payload.get("counts") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "phase_4_prepared_support_sources"
        or payload.get("status") != "PHASE_4_SUPPORT_SOURCES_PREPARED_FROM_FROZEN_S6"
        or payload.get("contains_confirmatory_data") is not False
        or not isinstance(output_hashes, dict)
        or set(output_hashes)
        != {"symbolic_scenes", "natural_scenes", "natural_observations", "selection_trace"}
        or not isinstance(counts, dict)
        or type(counts.get("symbolic_scenes")) is not int
        or type(counts.get("natural_single_error_scenes")) is not int
        or counts["symbolic_scenes"] <= 0
        or counts["natural_single_error_scenes"] <= 0
    ):
        raise RuntimeError("Phase 4 prepared source summary is malformed")
    actual = {
        "symbolic_scenes": prepared_paths.symbolic_scenes,
        "natural_scenes": prepared_paths.natural_scenes,
        "natural_observations": prepared_paths.natural_observations,
        "selection_trace": prepared_paths.selection_trace,
    }
    for name, path in actual.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(output_hashes.get(name), str)
            or _sha256_path(path) != output_hashes[name]
        ):
            raise RuntimeError(f"Phase 4 prepared source hash mismatch: {name}")
    return {
        name: str(output_hashes[name])
        for name in (
            "symbolic_scenes",
            "natural_scenes",
            "natural_observations",
        )
    }


__all__ = [
    "NaturalObservationCapture",
    "PreparedSourcePaths",
    "PreparedSupportSources",
    "build_independent_natural_scenes",
    "build_legacy_s6_natural_candidates",
    "prepare_legacy_s6_support_sources",
    "retain_natural_single_error_scenes",
    "validate_prepared_source_summary",
    "write_prepared_support_sources",
]
