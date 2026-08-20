"""Fail-closed support data, preflight, and language-only LoRA helpers for Phase 4.

Optional GPU packages are deliberately imported only by the executor.  The data
and provenance checks remain executable on a CPU-only development machine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from compensability_v4.data.natural_error_pool import NaturalErrorExample, build_natural_error_pool
from compensability_v4.data.splits import DatasetSplit, validate_split_isolation
from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.schemas.scene import RecoveryScene


class SupportVariant(str, Enum):
    """The preregistered Phase 4 control and recovery adapters."""

    FORMAT_ONLY = "C0_format_only"
    FORWARD_ARITHMETIC = "C1_forward_arithmetic"
    RECOVERY = "T_constraint_recovery"


_LORA_LEAF_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
_TRAINING_SPLITS = frozenset(
    {DatasetSplit.SYMBOLIC_SUPPORT_TRAIN, DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN}
)


@dataclass(frozen=True, slots=True)
class Phase4TrainingConfig:
    """Frozen Phase 4 training parameters selected without confirm-set access."""

    precision: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    gradient_checkpointing: bool
    vision_frozen: bool
    merger_frozen: bool
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    max_sequence_length: int
    seed: int
    selection_split: str

    @classmethod
    def default(cls) -> Phase4TrainingConfig:
        return cls(
            precision="bf16",
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.0,
            gradient_checkpointing=True,
            vision_frozen=True,
            merger_frozen=True,
            learning_rate=2.0e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            max_sequence_length=512,
            seed=2026081804,
            selection_split=DatasetSplit.SUPPORT_DEV.value,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Phase4TrainingConfig:
        expected = set(asdict(cls.default()))
        if set(value) != expected:
            raise ValueError("Phase 4 training config fields differ from the frozen contract")
        config = cls(**dict(value))  # type: ignore[arg-type]
        if (
            config.precision != "bf16"
            or config.lora_rank != 16
            or config.lora_alpha != 32
            or config.lora_dropout != 0.0
            or config.gradient_checkpointing is not True
            or config.vision_frozen is not True
            or config.merger_frozen is not True
            or config.selection_split != DatasetSplit.SUPPORT_DEV.value
        ):
            raise ValueError("Phase 4 LoRA/freeze contract drifted")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in (
                config.per_device_train_batch_size,
                config.gradient_accumulation_steps,
                config.num_train_epochs,
                config.max_sequence_length,
                config.seed,
            )
        ):
            raise ValueError("Phase 4 integer training parameters must be positive")
        if not isinstance(config.learning_rate, float) or config.learning_rate <= 0:
            raise ValueError("Phase 4 learning_rate must be positive")
        return config


@dataclass(frozen=True, slots=True)
class SupportExample:
    variant: SupportVariant
    example_id: str
    source_kind: str
    source_scene_id: str
    source_observation_id: str | None
    split: DatasetSplit
    curriculum_stage: str
    prompt: str
    completion: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "variant": self.variant.value,
            "example_id": self.example_id,
            "source_kind": self.source_kind,
            "source_scene_id": self.source_scene_id,
            "source_observation_id": self.source_observation_id,
            "split": self.split.value,
            "curriculum_stage": self.curriculum_stage,
            "prompt": self.prompt,
            "completion": self.completion,
        }


def _world_text(values: Sequence[int]) -> str:
    invalid = len(values) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    )
    if invalid:
        raise ValueError("Phase 4 worlds must contain exactly four integers")
    return ",".join(str(value) for value in values)


def _facts_text(scene: RecoveryScene) -> str:
    return json.dumps(list(scene.to_mapping()["facts"]), sort_keys=True, separators=(",", ":"))


def _example(
    *,
    variant: SupportVariant,
    scene: RecoveryScene,
    source_kind: str,
    source_observation_id: str | None,
    observed: tuple[int, int, int, int],
) -> tuple[SupportExample, ...]:
    truth = _world_text(scene.truth)
    observed_text = _world_text(observed)
    prefix = f"{variant.value}:{source_kind}:{scene.scene_id}"
    if variant is SupportVariant.FORMAT_ONLY:
        return (
            SupportExample(
                variant=variant,
                example_id=prefix,
                source_kind=source_kind,
                source_scene_id=scene.scene_id,
                source_observation_id=source_observation_id,
                split=scene.split,
                curriculum_stage="format_only",
                prompt=(
                    f"Values to format: {truth}\n"
                    "Return exactly four comma-separated integers with no label or explanation."
                ),
                completion=truth,
            ),
        )
    if variant is SupportVariant.FORWARD_ARITHMETIC:
        return (
            SupportExample(
                variant=variant,
                example_id=prefix,
                source_kind=source_kind,
                source_scene_id=scene.scene_id,
                source_observation_id=source_observation_id,
                split=scene.split,
                curriculum_stage="forward_fact_verification",
                prompt=(
                    f"Correct values: {truth}\nFacts: {_facts_text(scene)}\n"
                    "Verify the supplied facts from the supplied correct values, then return "
                    "exactly "
                    "the four supplied comma-separated integers."
                ),
                completion=truth,
            ),
        )
    error_index = next(
        index
        for index, (expected, actual) in enumerate(zip(scene.truth, observed, strict=True))
        if expected != actual
    )
    replacement = scene.truth[error_index]
    common = (
        f"Observed values: {observed_text}\nFacts: {_facts_text(scene)}\n"
        "Use only the observations and facts in this prompt. "
    )
    staged = (
        (
            "fact_verification",
            common + "State whether the facts are jointly satisfiable.",
            "FACTS_VALID",
        ),
        (
            "conflict_detection",
            common + "State whether the observations conflict with the facts.",
            "CONFLICT",
        ),
        (
            "error_index",
            common + "Return the zero-based index of the inconsistent value.",
            str(error_index),
        ),
        (
            "replacement_value",
            common + "Return the required replacement value.",
            str(replacement),
        ),
        (
            "global_fact_verification",
            common + "After correction, state whether every fact is satisfied.",
            "FACTS_VALID",
        ),
        (
            "final_free_recovery",
            common + "Recover the full world. Return exactly four comma-separated integers only.",
            truth,
        ),
    )
    return tuple(
        SupportExample(
            variant=variant,
            example_id=f"{prefix}:{stage}",
            source_kind=source_kind,
            source_scene_id=scene.scene_id,
            source_observation_id=source_observation_id,
            split=scene.split,
            curriculum_stage=stage,
            prompt=prompt,
            completion=completion,
        )
        for stage, prompt, completion in staged
    )


def _validate_support_scene(scene: RecoveryScene, expected_split: DatasetSplit) -> None:
    if not isinstance(scene, RecoveryScene):
        raise TypeError("Phase 4 support scenes must be RecoveryScene records")
    if scene.split is not expected_split:
        raise ValueError(f"Phase 4 scene must use {expected_split.value}")


def build_support_sets(
    *,
    symbolic_scenes: Iterable[RecoveryScene],
    natural_scenes: Iterable[RecoveryScene],
    natural_errors: Iterable[NaturalErrorExample],
) -> dict[SupportVariant, tuple[SupportExample, ...]]:
    """Build C0, C1, and T examples without touching confirmatory data."""

    symbolic = tuple(sorted(symbolic_scenes, key=lambda scene: scene.scene_id))
    natural = tuple(sorted(natural_scenes, key=lambda scene: scene.scene_id))
    for scene in symbolic:
        _validate_support_scene(scene, DatasetSplit.SYMBOLIC_SUPPORT_TRAIN)
    for scene in natural:
        _validate_support_scene(scene, DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN)
    all_scenes = symbolic + natural
    if not all_scenes:
        raise ValueError("Phase 4 support data must include at least one training scene")
    if len({scene.scene_id for scene in all_scenes}) != len(all_scenes):
        raise ValueError("Phase 4 support scene_id values must be globally unique")
    validate_split_isolation(all_scenes)
    natural_index = {scene.scene_id: scene for scene in natural}
    examples = tuple(sorted(natural_errors, key=lambda item: item.observation_id))
    if not examples:
        raise ValueError("Phase 4 natural_error_support_train examples are required")
    if len({item.observation_id for item in examples}) != len(examples):
        raise ValueError("Phase 4 natural observation_id values must be unique")
    natural_by_scene: dict[str, NaturalErrorExample] = {}
    for item in examples:
        if not isinstance(item, NaturalErrorExample):
            raise TypeError("Phase 4 natural errors must be NaturalErrorExample records")
        scene = natural_index.get(item.scene_id)
        if scene is None:
            raise ValueError("natural error is not paired with a natural_error_support_train scene")
        if item.truth != scene.truth:
            raise ValueError("natural error truth differs from its registered scene")
        changed = tuple(
            index
            for index, (truth, observed) in enumerate(
                zip(item.truth, item.observed_values, strict=True)
            )
            if truth != observed
        )
        if changed != (item.error_index,):
            raise ValueError("natural error must contain exactly its registered one-position error")
        if item.scene_id in natural_by_scene:
            raise ValueError("Phase 4 accepts one natural observation per support scene")
        natural_by_scene[item.scene_id] = item
    if set(natural_by_scene) != set(natural_index):
        raise ValueError("every natural_error_support_train scene requires one natural observation")
    output: dict[SupportVariant, list[SupportExample]] = {variant: [] for variant in SupportVariant}
    for scene in symbolic:
        observed = list(scene.truth)
        observed[0] = scene.truth[0] + 1
        for variant in SupportVariant:
            output[variant].extend(
                _example(
                    variant=variant,
                    scene=scene,
                    source_kind="symbolic",
                    source_observation_id=None,
                    observed=tuple(observed),  # type: ignore[arg-type]
                )
            )
    for scene in natural:
        item = natural_by_scene[scene.scene_id]
        for variant in SupportVariant:
            output[variant].extend(
                _example(
                    variant=variant,
                    scene=scene,
                    source_kind="natural_error",
                    source_observation_id=item.observation_id,
                    observed=item.observed_values,
                )
            )
    return {variant: tuple(rows) for variant, rows in output.items()}


def _require_hashes(source_hashes: Mapping[str, str]) -> dict[str, str]:
    required = {"symbolic_scenes", "natural_scenes", "natural_observations"}
    if set(source_hashes) != required or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in source_hashes.values()
    ):
        raise ValueError("Phase 4 source hashes must include three SHA-256 values")
    return dict(sorted(source_hashes.items()))


def write_support_artifact(
    *,
    output_path: Path,
    summary_path: Path,
    support_sets: Mapping[SupportVariant, Sequence[SupportExample]],
    source_hashes: Mapping[str, str],
) -> None:
    """Publish a non-overwriting support corpus and its exact provenance summary."""

    if any(path.exists() or path.is_symlink() for path in (output_path, summary_path)):
        raise FileExistsError("refusing to overwrite Phase 4 support artifacts")
    hashes = _require_hashes(source_hashes)
    rows = [row for variant in SupportVariant for row in support_sets.get(variant, ())]
    if not rows:
        raise ValueError("Phase 4 support artifact cannot be empty")
    if len({row.example_id for row in rows}) != len(rows):
        raise ValueError("Phase 4 support example_id values must be unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row.to_mapping(), sort_keys=True, allow_nan=False) + "\n")
    digest = sha256_path(output_path)
    payload = {
        "schema_version": 1,
        "artifact_type": "phase_4_support_data",
        "example_count": len(rows),
        "counts_by_variant": {
            variant.value: len(support_sets.get(variant, ())) for variant in SupportVariant
        },
        "source_hashes": hashes,
        "support_jsonl_sha256": digest,
        "contains_confirmatory_data": False,
        "final_recovery_format": "a,b,c,d",
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase4_config(path: Path) -> Phase4TrainingConfig:
    """Read the dedicated Phase 4 config without weakening the frozen 0-3 config."""

    import yaml

    if path.is_symlink() or not path.is_file():
        raise ValueError("Phase 4 config must be a regular file")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 4 config must be a mapping")
    expected = {
        "schema_version",
        "status",
        "model",
        "authorization",
        "training",
        "integrity_gates",
        "artifacts",
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise ValueError("Phase 4 config schema differs from the frozen contract")
    authorization = payload["authorization"]
    gates = payload["integrity_gates"]
    if not isinstance(authorization, dict) or not isinstance(gates, dict):
        raise ValueError("Phase 4 authorization/integrity gates are malformed")
    if authorization != {
        "training_authorized": True,
        "rl_authorized": False,
        "downloads_authorized": False,
    }:
        raise ValueError("Phase 4 authorization contract drifted")
    required_gates = {
        "require_hash_bound_inputs": True,
        "require_confirm_split_exclusion": True,
        "require_language_only_lora": True,
        "require_frozen_hashes": True,
        "forbid_artifact_overwrite": True,
        "require_explicit_gpu_acknowledgement": True,
    }
    if gates != required_gates:
        raise ValueError("Phase 4 integrity gates differ from the frozen contract")
    model = payload["model"]
    if not isinstance(model, dict):
        raise ValueError("Phase 4 model pin is malformed")
    from compensability_v4.qwen.model_loader import MODEL_PATH, MODEL_SNAPSHOT_SHA256

    if model != {"local_path": MODEL_PATH, "snapshot_sha256": MODEL_SNAPSHOT_SHA256}:
        raise ValueError("Phase 4 model pin differs from the archived Qwen snapshot")
    training = payload["training"]
    if not isinstance(training, dict):
        raise ValueError("Phase 4 training config is malformed")
    return Phase4TrainingConfig.from_mapping(training)


def verify_phase4_package_lock(
    *, lock_path: Path, repository_root: Path, expected_paths: Iterable[str]
) -> str:
    """Verify the complete Phase 4 executable surface against immutable file hashes."""

    import yaml

    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Phase 4 package lock must be a regular file")
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    rows = payload.get("files") if isinstance(payload, dict) else None
    if payload.get("schema_version") != 1 or not isinstance(rows, list) or not rows:
        raise ValueError("Phase 4 package lock is malformed")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("Phase 4 package lock row is malformed")
        relative, digest = row["path"], row["sha256"]
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or relative in observed
        ):
            raise ValueError("Phase 4 package lock row has invalid fields")
        candidate = repository_root / relative
        if candidate.is_symlink() or not candidate.is_file() or sha256_path(candidate) != digest:
            raise RuntimeError(f"Phase 4 package lock mismatch: {relative}")
        observed.add(relative)
    required = set(expected_paths)
    if observed != required:
        raise RuntimeError(
            "Phase 4 package lock closure mismatch; "
            f"missing={sorted(required - observed)}, extra={sorted(observed - required)}"
        )
    return sha256_path(lock_path)


def load_support_sources(
    *,
    symbolic_scenes_path: Path,
    natural_scenes_path: Path,
    natural_observations_path: Path,
) -> tuple[tuple[RecoveryScene, ...], tuple[RecoveryScene, ...], tuple[NaturalErrorExample, ...]]:
    """Load strict JSONL sources and derive the natural one-error support pool."""

    def read_jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Phase 4 {label} must be a regular JSONL file")
        rows = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Phase 4 {label} must contain nonempty JSON objects")
        return rows  # type: ignore[return-value]

    symbolic_rows = read_jsonl(symbolic_scenes_path, "symbolic scenes")
    natural_rows = read_jsonl(natural_scenes_path, "natural scenes")
    observation_rows = read_jsonl(natural_observations_path, "natural observations")
    symbolic = tuple(RecoveryScene.from_mapping(row) for row in symbolic_rows)
    natural = tuple(RecoveryScene.from_mapping(row) for row in natural_rows)
    observations = tuple(NaturalObservation.from_mapping(row) for row in observation_rows)
    return symbolic, natural, build_natural_error_pool(natural, observations)


def load_support_rows(path: Path, *, variant: SupportVariant) -> tuple[SupportExample, ...]:
    """Read a published support corpus and reject drift before training is constructed."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("Phase 4 support corpus must be a regular JSONL file")
    rows: list[SupportExample] = []
    expected = {
        "schema_version",
        "variant",
        "example_id",
        "source_kind",
        "source_scene_id",
        "source_observation_id",
        "split",
        "curriculum_stage",
        "prompt",
        "completion",
    }
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("schema_version") != 1:
            raise ValueError(f"Phase 4 support row {line_number} has a malformed schema")
        try:
            row_variant = SupportVariant(raw["variant"])
            split = DatasetSplit(raw["split"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Phase 4 support row {line_number} has unregistered fields"
            ) from error
        strings = (
            "example_id",
            "source_kind",
            "source_scene_id",
            "curriculum_stage",
            "prompt",
            "completion",
        )
        if any(not isinstance(raw[name], str) or not raw[name] for name in strings):
            raise ValueError(f"Phase 4 support row {line_number} has an invalid text field")
        observation_id = raw["source_observation_id"]
        if observation_id is not None and (
            not isinstance(observation_id, str) or not observation_id
        ):
            raise ValueError(
                f"Phase 4 support row {line_number} has an invalid observation identifier"
            )
        if split not in _TRAINING_SPLITS:
            raise ValueError("Phase 4 support corpus contains a non-training split")
        if row_variant is variant:
            rows.append(
                SupportExample(
                    variant=row_variant,
                    example_id=raw["example_id"],
                    source_kind=raw["source_kind"],
                    source_scene_id=raw["source_scene_id"],
                    source_observation_id=observation_id,
                    split=split,
                    curriculum_stage=raw["curriculum_stage"],
                    prompt=raw["prompt"],
                    completion=raw["completion"],
                )
            )
    if not rows:
        raise ValueError(f"Phase 4 support corpus contains no {variant.value} rows")
    if len({row.example_id for row in rows}) != len(rows):
        raise ValueError("Phase 4 support corpus contains duplicate example identifiers")
    if variant is SupportVariant.RECOVERY and not any(
        row.curriculum_stage == "final_free_recovery"
        and re.fullmatch(r"-?\d+,-?\d+,-?\d+,-?\d+", row.completion)
        for row in rows
    ):
        raise ValueError("Phase 4 recovery corpus lacks free a,b,c,d final-stage supervision")
    return tuple(rows)


def discover_language_lora_targets(model: object) -> tuple[str, ...]:
    """Discover exact Qwen language attention/MLP targets from named modules."""

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("model must expose named_modules() for Phase 4 target discovery")
    targets = sorted(
        name
        for name, _module in named_modules()
        if isinstance(name, str)
        and ".language_model.layers." in name
        and name.rsplit(".", maxsplit=1)[-1] in _LORA_LEAF_NAMES
    )
    if not targets:
        raise RuntimeError("no language attention/MLP LoRA targets were discovered")
    if any("visual" in name or "merger" in name for name in targets):
        raise RuntimeError("Phase 4 discovered a non-language LoRA target")
    return tuple(targets)


def _parameter_bytes(parameter: object) -> bytes:
    if isinstance(parameter, (bytes, bytearray, memoryview)):
        return bytes(parameter)
    detach = getattr(parameter, "detach", None)
    value = detach() if callable(detach) else parameter
    for method in ("cpu", "contiguous"):
        candidate = getattr(value, method, None)
        if callable(candidate):
            value = candidate()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        try:
            value = numpy()
        except TypeError as error:
            view = getattr(value, "view", None)
            if not callable(view):
                raise TypeError("model parameter dtype cannot be serialized losslessly") from error
            torch = _import_torch()
            try:
                reshape = getattr(value, "reshape", None)
                byte_source = reshape(-1) if callable(reshape) else value
                value = byte_source.view(torch.uint8).numpy()
            except (AttributeError, RuntimeError, TypeError) as fallback_error:
                raise TypeError(
                    "model parameter dtype cannot be serialized losslessly"
                ) from fallback_error
    tobytes = getattr(value, "tobytes", None)
    if not callable(tobytes):
        raise TypeError("model parameter does not expose serializable tensor bytes")
    return tobytes()


def freeze_base_parameters(model: object) -> dict[str, object]:
    """Freeze every base tensor and record component-level exact tensor hashes."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model must expose named_parameters() for Phase 4 freezing")
    groups: dict[str, hashlib._Hash] = {}
    counts: dict[str, int] = {}
    for name, parameter in named_parameters():
        requires_grad = getattr(parameter, "requires_grad", None)
        if requires_grad is None:
            raise TypeError(f"model parameter {name!r} lacks requires_grad")
        parameter.requires_grad = False
        group = (
            "merger"
            if ".merger" in name
            else "vision"
            if ".visual." in name
            else "language_base"
            if ".language_model." in name
            else "other_base"
        )
        digest = groups.setdefault(group, hashlib.sha256())
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_parameter_bytes(parameter))
        digest.update(b"\n")
        counts[group] = counts.get(group, 0) + 1
    required_groups = ("vision", "merger", "language_base")
    if not groups or any(not counts.get(group) for group in required_groups):
        raise RuntimeError("Phase 4 cannot verify all required frozen model components")
    return {
        "schema_version": 1,
        "artifact_type": "phase_4_frozen_parameter_hashes",
        "parameter_counts": dict(sorted(counts.items())),
        "sha256_by_component": {
            name: digest.hexdigest() for name, digest in sorted(groups.items())
        },
    }


def trainable_parameter_manifest(model: object, targets: Sequence[str]) -> dict[str, object]:
    """Reject any trainability outside adapters attached to discovered language targets."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model must expose named_parameters() for Phase 4 manifest")
    trainable = sorted(name for name, parameter in named_parameters() if parameter.requires_grad)
    if not trainable:
        raise RuntimeError("Phase 4 LoRA attachment produced no trainable parameters")
    if any("lora_" not in name or ".language_model." not in name for name in trainable):
        raise RuntimeError("only language LoRA parameters may be trainable in Phase 4")
    if any(".visual." in name or ".merger" in name for name in trainable):
        raise RuntimeError("vision or merger parameter became trainable in Phase 4")
    return {
        "schema_version": 1,
        "artifact_type": "phase_4_trainable_parameter_manifest",
        "target_modules": list(targets),
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": len(trainable),
        "vision_frozen": True,
        "merger_frozen": True,
        "base_language_frozen": True,
    }


def _missing_gpu_dependencies() -> tuple[str, ...]:
    return tuple(
        package
        for package in ("accelerate", "datasets", "peft", "transformers")
        if importlib.util.find_spec(package) is None
    )


def _import_torch() -> Any:
    import torch

    return torch


def validate_phase4_preflight(*, config: Phase4TrainingConfig, output_root: Path) -> None:
    """Check executable prerequisites before model mutation or optimizer construction."""

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Phase 4 output root already exists; refusing to overwrite")
    missing = _missing_gpu_dependencies()
    if missing:
        raise RuntimeError("Phase 4 missing required GPU packages: " + ", ".join(missing))
    torch = _import_torch()
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise RuntimeError("Phase 4 requires CUDA before any optimizer step")
    bf16 = getattr(cuda, "is_bf16_supported", None)
    if not callable(bf16) or not bf16():
        raise RuntimeError("Phase 4 requires CUDA bf16 support before any optimizer step")
    if config.precision != "bf16":
        raise RuntimeError("Phase 4 precision must remain bf16")


def attach_language_lora(
    model: object, *, config: Phase4TrainingConfig, targets: Sequence[str]
) -> object:
    """Attach PEFT only after all base parameters have been frozen."""

    from peft import LoraConfig, get_peft_model

    if not targets:
        raise ValueError("Phase 4 LoRA targets cannot be empty")
    enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
    if config.gradient_checkpointing and callable(enable_checkpointing):
        enable_checkpointing()
    enable_input_grads = getattr(model, "enable_input_require_grads", None)
    if config.gradient_checkpointing and callable(enable_input_grads):
        enable_input_grads()
    adapter = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(targets),
        ),
    )
    return adapter


def write_parameter_manifests(
    *,
    artifact_root: Path,
    trainable_manifest: Mapping[str, object],
    frozen_hashes: Mapping[str, object],
) -> tuple[Path, Path]:
    """Write the required Phase 4 trainability evidence before optimizer construction."""

    trainable_path = artifact_root / "trainable_parameter_manifest.json"
    frozen_path = artifact_root / "frozen_hashes.json"
    if any(path.exists() or path.is_symlink() for path in (trainable_path, frozen_path)):
        raise FileExistsError("refusing to overwrite Phase 4 parameter manifests")
    artifact_root.mkdir(parents=True, exist_ok=True)
    for path, payload in ((trainable_path, trainable_manifest), (frozen_path, frozen_hashes)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    return trainable_path, frozen_path


def _chat_training_features(
    *,
    tokenizer: object,
    rows: Sequence[SupportExample],
    max_sequence_length: int,
) -> list[dict[str, list[int]]]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    encode = getattr(tokenizer, "encode", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        not callable(apply_chat_template)
        or not callable(encode)
        or not isinstance(eos_token_id, int)
    ):
        raise RuntimeError("Phase 4 tokenizer lacks the required chat-template interface")
    features: list[dict[str, list[int]]] = []
    for row in rows:
        prefix = apply_chat_template(
            [{"role": "user", "content": row.prompt}], tokenize=False, add_generation_prompt=True
        )
        if not isinstance(prefix, str):
            raise RuntimeError("Phase 4 chat template did not return text")
        prefix_ids = list(encode(prefix, add_special_tokens=False))
        completion_ids = [*encode(row.completion, add_special_tokens=False), eos_token_id]
        input_ids = prefix_ids + completion_ids
        if len(input_ids) > max_sequence_length:
            raise RuntimeError(
                f"Phase 4 example {row.example_id} exceeds max_sequence_length before training"
            )
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prefix_ids) + completion_ids,
            }
        )
    return features


def _causal_collator(tokenizer: object) -> Any:
    torch = _import_torch()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(pad_token_id, int):
        if not isinstance(eos_token_id, int):
            raise RuntimeError("Phase 4 tokenizer has no pad or EOS token")
        pad_token_id = eos_token_id

    def collate(features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        maximum = max(len(row["input_ids"]) for row in features)
        result: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            padding = maximum - len(row["input_ids"])
            result["input_ids"].append(list(row["input_ids"]) + [pad_token_id] * padding)
            result["attention_mask"].append(list(row["attention_mask"]) + [0] * padding)
            result["labels"].append(list(row["labels"]) + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in result.items()}

    return collate


def execute_lora_sft(
    *,
    model: object,
    processor: object,
    rows: Sequence[SupportExample],
    variant: SupportVariant,
    config: Phase4TrainingConfig,
    output_root: Path,
) -> Path:
    """Run one adapter SFT only after preflight and trainability checks have completed."""

    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    tokenizer = getattr(processor, "tokenizer", processor)
    features = _chat_training_features(
        tokenizer=tokenizer, rows=rows, max_sequence_length=config.max_sequence_length
    )
    variant_root = output_root / variant.value
    if variant_root.exists() or variant_root.is_symlink():
        raise FileExistsError("refusing to overwrite a Phase 4 adapter output")
    arguments = TrainingArguments(
        output_dir=str(variant_root),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        bf16=True,
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(features),
        data_collator=_causal_collator(tokenizer),
    )
    trainer.train()
    adapter = variant_root / "final_adapter"
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise RuntimeError("Phase 4 PEFT model cannot save its adapter")
    save_pretrained(str(adapter))
    metrics_path = variant_root / "metrics.json"
    with metrics_path.open("x", encoding="utf-8") as stream:
        json.dump(trainer.state.log_history, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return variant_root


__all__ = [
    "Phase4TrainingConfig",
    "SupportExample",
    "SupportVariant",
    "attach_language_lora",
    "build_support_sets",
    "discover_language_lora_targets",
    "execute_lora_sft",
    "freeze_base_parameters",
    "load_phase4_config",
    "load_support_rows",
    "load_support_sources",
    "sha256_path",
    "trainable_parameter_manifest",
    "validate_phase4_preflight",
    "verify_phase4_package_lock",
    "write_parameter_manifests",
    "write_support_artifact",
]
