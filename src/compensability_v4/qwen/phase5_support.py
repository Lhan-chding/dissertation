"""Held-out Phase 5 policy-support contracts and deterministic reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from compbias.recoverability.compatibility import (
    ArithmeticProgressionConstraint,
    KnownValueConstraint,
    PairSumConstraint,
)
from compbias.recoverability.phase_c_screen import build_family_constraints
from compensability_v4.data.splits import DatasetSplit
from compensability_v4.schemas._common import freeze_json, thaw_json
from compensability_v4.schemas.scene import RecoveryScene
from compensability_v4.theory.constraint_system import validate_world

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WORLD = re.compile(r"\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\Z")
_FAMILIES = frozenset({"cross_series", "duplicate_encoding", "trend"})


class PolicyCheckpoint(str, Enum):
    BASE = "Base"
    FORMAT_ONLY = "C0"
    FORWARD_ARITHMETIC = "C1"
    RECOVERY = "T"


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _world(value: object, label: str) -> tuple[int, int, int, int]:
    try:
        return validate_world(value, label)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain exactly four integers") from error


def parse_world(text: object) -> tuple[int, int, int, int] | None:
    match = _WORLD.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        return None
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def _fact_mapping(constraint: object) -> Mapping[str, object]:
    if isinstance(constraint, PairSumConstraint):
        result: dict[str, object] = {
            "type": "pair_sum",
            "left_index": constraint.left_index,
            "right_index": constraint.right_index,
            "total": constraint.total,
            "fact_id": constraint.constraint_id,
        }
    elif isinstance(constraint, KnownValueConstraint):
        result = {
            "type": "known_value",
            "index": constraint.index,
            "value": constraint.value,
            "fact_id": constraint.constraint_id,
        }
    elif isinstance(constraint, ArithmeticProgressionConstraint):
        result = {
            "type": "arithmetic_progression",
            "indices": constraint.indices,
            "fact_id": constraint.constraint_id,
        }
    else:
        raise TypeError("Phase 5 support-dev scene uses an unregistered constraint")
    return MappingProxyType(result)


def _stable_rank(seed: int, scene_id: str) -> bytes:
    return hashlib.sha256(f"phase5-support-dev:{seed}:{scene_id}".encode()).digest()


def build_support_dev_candidates(
    records: Iterable[Mapping[str, object]],
    *,
    excluded_scene_ids: frozenset[str],
    count: int,
    seed: int,
) -> tuple[RecoveryScene, ...]:
    """Choose a fixed support-dev candidate cohort before observing model outcomes."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Phase 5 support-dev count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Phase 5 support-dev seed must be an integer")
    if any(not isinstance(item, str) or not item for item in excluded_scene_ids):
        raise ValueError("Phase 5 excluded scene identifiers are malformed")
    indexed: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Phase 5 dataset records must be mappings")
        scene_id = record.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("Phase 5 dataset scene identifier is malformed")
        if scene_id in indexed:
            raise ValueError("Phase 5 dataset scene identifiers are duplicated")
        indexed[scene_id] = record
    eligible = tuple(
        record for scene_id, record in indexed.items() if scene_id not in excluded_scene_ids
    )
    if len(eligible) < count:
        raise RuntimeError("Phase 5 eligible support-dev source pool is smaller than requested")
    rank_key = lambda record: (  # noqa: E731
        _stable_rank(seed, str(record["scene_id"])),
        str(record["scene_id"]),
    )
    if count % len(_FAMILIES) == 0:
        per_family = count // len(_FAMILIES)
        selected_rows: list[Mapping[str, object]] = []
        for family in sorted(_FAMILIES):
            family_rows = sorted(
                (record for record in eligible if record.get("family") == family), key=rank_key
            )
            if len(family_rows) < per_family:
                raise RuntimeError(
                    f"Phase 5 support-dev {family} pool is smaller than the balanced quota"
                )
            selected_rows.extend(family_rows[:per_family])
        selected = selected_rows
    else:
        selected = sorted(eligible, key=rank_key)[:count]
    scenes: list[RecoveryScene] = []
    for record in selected:
        scene_id, family, image = record.get("scene_id"), record.get("family"), record.get("image")
        if not isinstance(scene_id, str) or family not in _FAMILIES or not isinstance(image, str):
            raise ValueError("Phase 5 support-dev dataset record is malformed")
        path = PurePosixPath(image)
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".png":
            raise ValueError("Phase 5 support-dev image path is unsafe")
        truth = _world(record.get("values"), "Phase 5 support-dev truth")
        facts = tuple(_fact_mapping(item) for item in build_family_constraints(str(family), truth))
        scenes.append(
            RecoveryScene(
                scene_id=scene_id,
                split=DatasetSplit.SUPPORT_DEV,
                semantic_scene_id=f"phase5-semantic-{scene_id}",
                numeric_table_id=f"phase5-numbers-{scene_id}",
                constraint_graph_id=f"phase5-graph-{scene_id}",
                truth=truth,
                facts=facts,
                resized_height=280,
                resized_width=280,
                image_path=image,
            )
        )
    return tuple(sorted(scenes, key=lambda scene: scene.scene_id))


@dataclass(frozen=True, slots=True)
class HeldOutNaturalError:
    scene_id: str
    family: str
    split: DatasetSplit
    truth: tuple[int, int, int, int]
    observed: tuple[int, int, int, int]
    error_indices: tuple[int, ...]
    facts: tuple[Mapping[str, object], ...]
    image_path: str
    stage1_model_sha256: str
    stage1_raw_output: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("Phase 5 natural-error scene_id is invalid")
        if self.family not in _FAMILIES:
            raise ValueError("Phase 5 natural-error family is invalid")
        if self.split is not DatasetSplit.SUPPORT_DEV:
            raise ValueError("Phase 5 natural errors must use support_dev")
        object.__setattr__(self, "truth", _world(self.truth, "Phase 5 truth"))
        object.__setattr__(self, "observed", _world(self.observed, "Phase 5 observation"))
        expected = tuple(
            index
            for index, pair in enumerate(zip(self.truth, self.observed, strict=True))
            if pair[0] != pair[1]
        )
        if not expected or self.error_indices != expected:
            raise ValueError("Phase 5 natural-error indices differ from the observed error")
        object.__setattr__(
            self,
            "facts",
            tuple(
                freeze_json(dict(fact), f"facts[{index}]") for index, fact in enumerate(self.facts)
            ),
        )
        _require_sha256(self.stage1_model_sha256, "Phase 5 Stage-1 model hash")
        if not isinstance(self.stage1_raw_output, str):
            raise TypeError("Phase 5 Stage-1 raw output must be text")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scene_id": self.scene_id,
            "family": self.family,
            "split": self.split.value,
            "truth": list(self.truth),
            "observed": list(self.observed),
            "error_indices": list(self.error_indices),
            "facts": [thaw_json(fact) for fact in self.facts],
            "image_path": self.image_path,
            "stage1_model_sha256": self.stage1_model_sha256,
            "stage1_raw_output": self.stage1_raw_output,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> HeldOutNaturalError:
        expected = {
            "schema_version",
            "scene_id",
            "family",
            "split",
            "truth",
            "observed",
            "error_indices",
            "facts",
            "image_path",
            "stage1_model_sha256",
            "stage1_raw_output",
        }
        if set(value) != expected or value.get("schema_version") != 1:
            raise ValueError("Phase 5 held-out natural-error schema drifted")
        facts = value["facts"]
        if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes)):
            raise TypeError("Phase 5 facts must be a sequence")
        return cls(
            scene_id=value["scene_id"],  # type: ignore[arg-type]
            family=value["family"],  # type: ignore[arg-type]
            split=DatasetSplit(value["split"]),  # type: ignore[arg-type]
            truth=_world(value["truth"], "Phase 5 truth"),
            observed=_world(value["observed"], "Phase 5 observation"),
            error_indices=tuple(value["error_indices"]),  # type: ignore[arg-type]
            facts=tuple(facts),  # type: ignore[arg-type]
            image_path=value["image_path"],  # type: ignore[arg-type]
            stage1_model_sha256=value["stage1_model_sha256"],  # type: ignore[arg-type]
            stage1_raw_output=value["stage1_raw_output"],  # type: ignore[arg-type]
        )


def retain_held_out_natural_errors(
    scenes: Iterable[RecoveryScene],
    *,
    output_by_scene: Mapping[str, str],
    stage1_model_sha256: str,
    value_domain: Iterable[int] = range(2, 19),
) -> tuple[tuple[HeldOutNaturalError, ...], tuple[Mapping[str, object], ...]]:
    """Retain all single-position errors; never select by recovery success."""

    model_hash = _require_sha256(stage1_model_sha256, "Phase 5 Stage-1 model hash")
    domain = frozenset(value_domain)
    if not domain or any(type(value) is not int for value in domain):
        raise ValueError("Phase 5 value domain must contain integers")
    candidates = tuple(sorted(scenes, key=lambda scene: scene.scene_id))
    if not candidates or len({scene.scene_id for scene in candidates}) != len(candidates):
        raise ValueError("Phase 5 support-dev candidates must be non-empty and unique")
    if set(output_by_scene) != {scene.scene_id for scene in candidates}:
        raise ValueError("Phase 5 Stage-1 output/candidate closure differs")
    errors: list[HeldOutNaturalError] = []
    traces: list[Mapping[str, object]] = []
    for scene in candidates:
        if scene.split is not DatasetSplit.SUPPORT_DEV:
            raise ValueError("Phase 5 candidates must use support_dev")
        raw = output_by_scene[scene.scene_id]
        parsed = parse_world(raw)
        trace: dict[str, object] = {
            "scene_id": scene.scene_id,
            "family": _family_from_scene(scene),
            "stage1_raw_output": raw,
            "parsed_world": list(parsed) if parsed is not None else None,
        }
        if parsed is None:
            trace["selection_status"] = "excluded_unparseable"
        elif any(value not in domain for value in parsed):
            trace["selection_status"] = "excluded_outside_domain"
        else:
            indices = tuple(
                index
                for index, pair in enumerate(zip(scene.truth, parsed, strict=True))
                if pair[0] != pair[1]
            )
            if not indices:
                trace["selection_status"] = "excluded_correct"
            elif len(indices) != 1:
                trace["selection_status"] = "excluded_multiple_errors"
                trace["error_indices"] = list(indices)
            else:
                family = _family_from_scene(scene)
                trace["family"] = family
                trace["selection_status"] = "included_natural_error"
                trace["error_indices"] = list(indices)
                errors.append(
                    HeldOutNaturalError(
                        scene_id=scene.scene_id,
                        family=family,
                        split=scene.split,
                        truth=scene.truth,
                        observed=parsed,
                        error_indices=indices,
                        facts=scene.facts,
                        image_path=scene.image_path,
                        stage1_model_sha256=model_hash,
                        stage1_raw_output=raw,
                    )
                )
        traces.append(MappingProxyType(trace))
    return tuple(errors), tuple(traces)


def _family_from_scene(scene: RecoveryScene) -> str:
    facts = tuple(dict(fact) for fact in scene.facts)
    types = {str(fact.get("type")) for fact in facts}
    if "arithmetic_progression" in types:
        return "trend"
    pair_sums = sum(fact.get("type") == "pair_sum" for fact in facts)
    return "cross_series" if pair_sums >= 2 else "duplicate_encoding"


@dataclass(frozen=True, slots=True)
class CheckpointSceneMeasurement:
    scene_id: str
    family: str
    split: DatasetSplit
    checkpoint: PolicyCheckpoint
    checkpoint_sha256: str
    truth: tuple[int, int, int, int]
    observed: tuple[int, int, int, int]
    greedy_raw_output: str
    greedy_token_ids: tuple[int, ...]
    greedy_output: tuple[int, int, int, int] | None
    greedy_parse_success: bool
    greedy_success: bool
    greedy_observation_copy: bool
    candidate_logp_true: float
    candidate_logp_observed: float
    candidate_margin_true_observed: float
    sample_raw_outputs: tuple[str, ...]
    sample_token_ids: tuple[tuple[int, ...], ...]
    sample_seeds: tuple[int, ...]
    sample_outputs: tuple[tuple[int, int, int, int] | None, ...]
    sample_parse_success: tuple[bool, ...]
    sample_success: tuple[bool, ...]
    sample_observation_copy: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, PolicyCheckpoint):
            object.__setattr__(self, "checkpoint", PolicyCheckpoint(self.checkpoint))
        if self.split is not DatasetSplit.SUPPORT_DEV:
            raise ValueError("Phase 5 measurements must use support_dev")
        if self.family not in _FAMILIES or not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("Phase 5 measurement identity is invalid")
        _require_sha256(self.checkpoint_sha256, "Phase 5 checkpoint hash")
        object.__setattr__(self, "truth", _world(self.truth, "Phase 5 truth"))
        object.__setattr__(self, "observed", _world(self.observed, "Phase 5 observation"))
        if self.greedy_output is not None:
            object.__setattr__(
                self, "greedy_output", _world(self.greedy_output, "Phase 5 greedy output")
            )
        if not isinstance(self.greedy_raw_output, str) or any(
            type(item) is not int for item in self.greedy_token_ids
        ):
            raise TypeError("Phase 5 greedy raw/token evidence is malformed")
        if any(
            type(item) is not bool
            for item in (
                self.greedy_parse_success,
                self.greedy_success,
                self.greedy_observation_copy,
                *self.sample_parse_success,
                *self.sample_success,
                *self.sample_observation_copy,
            )
        ):
            raise TypeError("Phase 5 success/copy flags must be boolean")
        lengths = {
            len(self.sample_raw_outputs),
            len(self.sample_token_ids),
            len(self.sample_seeds),
            len(self.sample_outputs),
            len(self.sample_parse_success),
            len(self.sample_success),
            len(self.sample_observation_copy),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("Phase 5 sample evidence is empty or misaligned")
        if any(not isinstance(item, str) for item in self.sample_raw_outputs) or any(
            type(seed) is not int or seed < 0 for seed in self.sample_seeds
        ):
            raise TypeError("Phase 5 sampled raw/seed evidence is malformed")
        if len(set(self.sample_seeds)) != len(self.sample_seeds) or any(
            any(type(token_id) is not int for token_id in token_ids)
            for token_ids in self.sample_token_ids
        ):
            raise ValueError("Phase 5 sampled token/seed evidence is invalid")
        for output in self.sample_outputs:
            if output is not None:
                _world(output, "Phase 5 sampled output")
        numeric = (
            self.candidate_logp_true,
            self.candidate_logp_observed,
            self.candidate_margin_true_observed,
        )
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in numeric
        ):
            raise ValueError("Phase 5 candidate scores must be finite")
        if not math.isclose(
            self.candidate_logp_true - self.candidate_logp_observed,
            self.candidate_margin_true_observed,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("Phase 5 candidate margin is inconsistent")

    @property
    def rollout_count(self) -> int:
        return len(self.sample_success)

    @property
    def p_i(self) -> float:
        return sum(self.sample_success) / self.rollout_count

    @property
    def observation_copy_rate(self) -> float:
        return sum(self.sample_observation_copy) / self.rollout_count

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["split"] = self.split.value
        payload["checkpoint"] = self.checkpoint.value
        payload["truth"] = list(self.truth)
        payload["observed"] = list(self.observed)
        payload["greedy_output"] = list(self.greedy_output) if self.greedy_output else None
        payload["greedy_token_ids"] = list(self.greedy_token_ids)
        payload["sample_token_ids"] = [list(item) for item in self.sample_token_ids]
        payload["sample_seeds"] = list(self.sample_seeds)
        payload["sample_outputs"] = [
            list(item) if item is not None else None for item in self.sample_outputs
        ]
        payload["rollout_count"] = self.rollout_count
        payload["success_count"] = sum(self.sample_success)
        payload["p_i"] = self.p_i
        payload["observation_copy_rate"] = self.observation_copy_rate
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CheckpointSceneMeasurement:
        payload = dict(value)
        for derived in ("rollout_count", "success_count", "p_i", "observation_copy_rate"):
            payload.pop(derived, None)
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(payload) != expected:
            raise ValueError("Phase 5 checkpoint measurement schema drifted")
        payload["split"] = DatasetSplit(payload["split"])
        payload["checkpoint"] = PolicyCheckpoint(payload["checkpoint"])
        tuple_fields = (
            "truth",
            "observed",
            "greedy_token_ids",
            "sample_raw_outputs",
            "sample_seeds",
            "sample_parse_success",
            "sample_success",
            "sample_observation_copy",
        )
        for field in tuple_fields:
            payload[field] = tuple(payload[field])
        greedy = payload["greedy_output"]
        payload["greedy_output"] = tuple(greedy) if greedy is not None else None
        payload["sample_token_ids"] = tuple(tuple(item) for item in payload["sample_token_ids"])
        payload["sample_outputs"] = tuple(
            tuple(item) if item is not None else None for item in payload["sample_outputs"]
        )
        return cls(**payload)  # type: ignore[arg-type]


def _positive_int_tuple(values: Sequence[int], label: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(type(item) is not int or item <= 0 for item in result):
        raise ValueError(f"{label} must contain positive integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{label} must be unique and increasing")
    return result


def summarize_phase5_policy_support(
    *,
    errors: Sequence[HeldOutNaturalError],
    measurements: Sequence[CheckpointSceneMeasurement],
    pass_at_k: Sequence[int],
    informative_group_size: int,
    sampling_temperature: float,
    sampling_seed: int,
) -> dict[str, object]:
    """Summarize empirical policy support without applying a success threshold."""

    ks = _positive_int_tuple(pass_at_k, "Phase 5 pass@K")
    if type(informative_group_size) is not int or informative_group_size <= 0:
        raise ValueError("Phase 5 informative group size must be positive")
    if not isinstance(sampling_temperature, float) or sampling_temperature <= 0:
        raise ValueError("Phase 5 sampling temperature must be positive")
    if type(sampling_seed) is not int:
        raise TypeError("Phase 5 sampling seed must be an integer")
    error_index = {error.scene_id: error for error in errors}
    if not error_index or len(error_index) != len(errors):
        raise ValueError("Phase 5 held-out natural errors must be non-empty and unique")
    indexed: dict[tuple[str, PolicyCheckpoint], CheckpointSceneMeasurement] = {}
    for row in measurements:
        if not isinstance(row, CheckpointSceneMeasurement):
            raise TypeError("Phase 5 measurements must use CheckpointSceneMeasurement")
        key = (row.scene_id, row.checkpoint)
        if key in indexed:
            raise ValueError("Phase 5 checkpoint/scene measurement is duplicated")
        error = error_index.get(row.scene_id)
        if error is None or (row.family, row.truth, row.observed) != (
            error.family,
            error.truth,
            error.observed,
        ):
            raise ValueError("Phase 5 measurement differs from held-out natural-error evidence")
        indexed[key] = row
    expected = {
        (scene_id, checkpoint) for scene_id in error_index for checkpoint in PolicyCheckpoint
    }
    if set(indexed) != expected:
        raise ValueError("Phase 5 four-checkpoint scene closure differs")
    rollout_counts = {row.rollout_count for row in measurements}
    if len(rollout_counts) != 1:
        raise ValueError("Phase 5 rollout count must be fixed across all scenes/checkpoints")
    rollouts = rollout_counts.pop()
    if max((*ks, informative_group_size)) > rollouts:
        raise ValueError("Phase 5 K cannot exceed the fixed rollout count")
    by_checkpoint: dict[str, object] = {}
    pass_rows: list[dict[str, object]] = []
    for checkpoint in PolicyCheckpoint:
        rows = tuple(indexed[(scene_id, checkpoint)] for scene_id in sorted(error_index))
        probabilities = tuple(row.p_i for row in rows)
        checkpoint_pass = {
            str(k): sum(1.0 - (1.0 - probability) ** k for probability in probabilities)
            / len(probabilities)
            for k in ks
        }
        g_values = tuple(
            1.0
            - probability**informative_group_size
            - (1.0 - probability) ** informative_group_size
            for probability in probabilities
        )
        by_family: dict[str, object] = {}
        for family in sorted(_FAMILIES):
            family_rows = tuple(row for row in rows if row.family == family)
            if not family_rows:
                continue
            by_family[family] = {
                "scene_count": len(family_rows),
                "mean_p_i": sum(row.p_i for row in family_rows) / len(family_rows),
                "observation_copy_rate": sum(row.observation_copy_rate for row in family_rows)
                / len(family_rows),
            }
        by_checkpoint[checkpoint.value] = {
            "scene_count": len(rows),
            "greedy_success_rate": sum(row.greedy_success for row in rows) / len(rows),
            "greedy_observation_copy_rate": sum(row.greedy_observation_copy for row in rows)
            / len(rows),
            "mean_p_i": sum(probabilities) / len(probabilities),
            "mean_G_K": sum(g_values) / len(g_values),
            "observation_copy_rate": sum(row.observation_copy_rate for row in rows) / len(rows),
            "candidate_margin_true_observed_mean": sum(
                row.candidate_margin_true_observed for row in rows
            )
            / len(rows),
            "pass_at_k": checkpoint_pass,
            "by_family": by_family,
        }
        pass_rows.extend(
            {
                "checkpoint": checkpoint.value,
                "k": k,
                "scene_count": len(rows),
                "mean_pass_at_k": checkpoint_pass[str(k)],
            }
            for k in ks
        )
    return {
        "schema_version": 1,
        "status": "PHASE_5_POLICY_SUPPORT_EXECUTED",
        "number_of_held_out_natural_errors": len(errors),
        "held_out_family_counts": dict(sorted(Counter(error.family for error in errors).items())),
        "number_of_checkpoint_scene_rows": len(measurements),
        "sampling_rollouts_per_scene": rollouts,
        "sampling_temperature": sampling_temperature,
        "sampling_seed": sampling_seed,
        "pass_at_k": list(ks),
        "informative_group_size": informative_group_size,
        "by_checkpoint": by_checkpoint,
        "pass_at_k_rows": pass_rows,
        "scene_is_statistical_unit": True,
        "subjective_success_threshold_applied": False,
        "confirmatory_data_used": False,
        "training_invoked": False,
        "rl_invoked": False,
    }


def _validated_source_hashes(values: Mapping[str, str]) -> dict[str, str]:
    required = {"support_dev", *(checkpoint.value for checkpoint in PolicyCheckpoint)}
    if set(values) != required:
        raise ValueError("Phase 5 source hashes differ from the required inputs")
    return {
        key: _require_sha256(value, f"Phase 5 {key} hash") for key, value in sorted(values.items())
    }


def write_phase5_outputs(
    *,
    parquet_path: Path,
    informative_path: Path,
    pass_at_k_path: Path,
    measurements: Sequence[CheckpointSceneMeasurement],
    summary: Mapping[str, object],
    source_sha256: Mapping[str, str],
) -> None:
    """Atomically publish the three preregistered Phase 5 artifacts."""

    paths = (parquet_path, informative_path, pass_at_k_path)
    output_roots = {path.parent for path in paths}
    if len(output_roots) != 1:
        raise ValueError("Phase 5 artifacts must share one publication directory")
    output_root = output_roots.pop()
    if (
        output_root.exists()
        or output_root.is_symlink()
        or any(path.exists() or path.is_symlink() for path in paths)
    ):
        raise FileExistsError("refusing to overwrite a Phase 5 artifact")
    hashes = _validated_source_hashes(source_sha256)
    if summary.get("status") != "PHASE_5_POLICY_SUPPORT_EXECUTED":
        raise ValueError("Phase 5 summary is malformed")
    rows = [row.to_mapping() for row in measurements]
    if not rows:
        raise ValueError("Phase 5 measurements are empty")
    pass_rows = summary.get("pass_at_k_rows")
    if not isinstance(pass_rows, list) or not pass_rows:
        raise ValueError("Phase 5 pass@K summary rows are malformed")
    import pyarrow as pa
    import pyarrow.parquet as pq

    payload = {**dict(summary), "source_sha256": hashes}
    payload.pop("pass_at_k_rows", None)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".phase5-support-", dir=str(output_root.parent)))
    temporary_paths = tuple(temporary / path.name for path in paths)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary_paths[0], compression="zstd")
        with temporary_paths[1].open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        with temporary_paths[2].open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("checkpoint", "k", "scene_count", "mean_pass_at_k")
            )
            writer.writeheader()
            writer.writerows(pass_rows)
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def write_support_dev_outputs(
    *,
    output_root: Path,
    candidates: Sequence[RecoveryScene],
    errors: Sequence[HeldOutNaturalError],
    traces: Sequence[Mapping[str, object]],
    source_sha256: Mapping[str, str],
    config_sha256: str,
    package_lock_sha256: str,
) -> dict[str, Path]:
    """Atomically publish the frozen Phase 5 intake and eligible error pool."""

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("refusing to overwrite Phase 5 support-dev artifacts")
    if not candidates or len(candidates) != len(traces):
        raise ValueError("Phase 5 support-dev candidate/trace closure differs")
    if len({scene.scene_id for scene in candidates}) != len(candidates):
        raise ValueError("Phase 5 support-dev candidate identifiers are duplicated")
    if {error.scene_id for error in errors} - {scene.scene_id for scene in candidates}:
        raise ValueError("Phase 5 held-out errors are not a subset of support-dev candidates")
    hashes = {
        key: _require_sha256(value, f"Phase 5 support-dev {key} hash")
        for key, value in sorted(source_sha256.items())
    }
    if not hashes:
        raise ValueError("Phase 5 support-dev source hashes are empty")
    config_hash = _require_sha256(config_sha256, "Phase 5 config hash")
    lock_hash = _require_sha256(package_lock_sha256, "Phase 5 package-lock hash")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".phase5-support-dev-", dir=str(output_root.parent)))
    paths = {
        "candidates": temporary / "candidates.jsonl",
        "errors": temporary / "held_out_natural_errors.jsonl",
        "trace": temporary / "selection_trace.jsonl",
        "summary": temporary / "summary.json",
    }
    try:
        with paths["candidates"].open("x", encoding="utf-8") as stream:
            for scene in sorted(candidates, key=lambda item: item.scene_id):
                stream.write(json.dumps(scene.to_mapping(), sort_keys=True, allow_nan=False) + "\n")
        with paths["errors"].open("x", encoding="utf-8") as stream:
            for error in sorted(errors, key=lambda item: item.scene_id):
                stream.write(json.dumps(error.to_mapping(), sort_keys=True, allow_nan=False) + "\n")
        with paths["trace"].open("x", encoding="utf-8") as stream:
            for trace in sorted(traces, key=lambda item: str(item["scene_id"])):
                stream.write(json.dumps(dict(trace), sort_keys=True, allow_nan=False) + "\n")
        status_counts = Counter(str(trace["selection_status"]) for trace in traces)
        payload = {
            "schema_version": 1,
            "status": "PHASE_5_SUPPORT_DEV_FROZEN",
            "candidate_count": len(candidates),
            "held_out_natural_error_count": len(errors),
            "selection_status_counts": dict(sorted(status_counts.items())),
            "family_counts": dict(sorted(Counter(error.family for error in errors).items())),
            "source_sha256": hashes,
            "config_sha256": config_hash,
            "package_lock_sha256": lock_hash,
            "candidate_sha256": _sha256_file(paths["candidates"]),
            "held_out_natural_errors_sha256": _sha256_file(paths["errors"]),
            "selection_trace_sha256": _sha256_file(paths["trace"]),
            "selection_uses_model_outcome_threshold": False,
            "confirmatory_data_used": False,
            "training_invoked": False,
            "rl_invoked": False,
        }
        with paths["summary"].open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {name: output_root / path.name for name, path in paths.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CheckpointSceneMeasurement",
    "HeldOutNaturalError",
    "PolicyCheckpoint",
    "build_support_dev_candidates",
    "parse_world",
    "retain_held_out_natural_errors",
    "summarize_phase5_policy_support",
    "write_phase5_outputs",
    "write_support_dev_outputs",
]
