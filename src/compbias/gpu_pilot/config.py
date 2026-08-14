"""Strict public configuration contracts for the GPU pilot."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_PATH_TOP = frozenset({"schema_version", "project_root", "model", "storage"})
_MODEL_FIELDS = frozenset({"qwen25vl3b"})
_MODEL_ENTRY_FIELDS = frozenset({"path"})
_STORAGE_FIELDS = frozenset({"data", "outputs", "checkpoints", "trajectories", "cache"})
_DATA_TOP = frozenset(
    {
        "schema_version",
        "dataset_id",
        "seed",
        "image_size",
        "chart_types",
        "operations",
        "split_counts",
        "counterfactual_pairs",
        "natural_audit",
    }
)
_SPLITS = (
    "calibration",
    "smoke_train",
    "pilot_train",
    "dev",
    "iid_test",
    "mechanism_ood",
)
_ENV_OVERRIDES = {
    "project_root": "COMPBIAS_PROJECT_ROOT",
    "model_path": "COMPBIAS_MODEL_PATH",
    "data": "COMPBIAS_DATA_DIR",
    "outputs": "COMPBIAS_OUTPUTS_DIR",
    "checkpoints": "COMPBIAS_CHECKPOINTS_DIR",
    "trajectories": "COMPBIAS_TRAJECTORIES_DIR",
    "cache": "COMPBIAS_CACHE_DIR",
}

ACTIVE_PILOT_DATASET_ID = "CVA-Chart-Pilot-v0.3"
ACTIVE_PILOT_DATA_CONFIG = "configs/data/cva_chart_pilot_v0_3.yaml"
ACTIVE_PILOT_OUTPUT_SLUG = "cva_chart_pilot_v0_3"
ACTIVE_CALIBRATION_RECORDS_NAME = "calibration_records_v0_3.jsonl"
ACTIVE_CALIBRATION_SUMMARY_NAME = "calibration_records_v0_3.summary.json"
_DATASET_DESIGNS = {
    "CVA-Chart-Pilot-v0.1": ("cva_chart_pilot_v0_1", "direct_labels_v0_1"),
    "CVA-Chart-Pilot-v0.2": ("cva_chart_pilot_v0_2", "axis_scale_v0_2"),
    ACTIVE_PILOT_DATASET_ID: (ACTIVE_PILOT_OUTPUT_SLUG, "axis_scale_v0_3"),
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty absolute path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return Path(os.path.abspath(os.fspath(path)))


def _positive_int(value: object, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class PilotPaths:
    """Resolved storage and offline-model paths for one server checkout."""

    project_root: Path
    model_path: Path
    data: Path
    outputs: Path
    checkpoints: Path
    trajectories: Path
    cache: Path

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            path = getattr(self, field_name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field_name} must be an absolute pathlib.Path")
            object.__setattr__(self, field_name, Path(os.path.abspath(os.fspath(path))))
        writable_names = ("data", "outputs", "checkpoints", "trajectories", "cache")
        writable = {name: getattr(self, name) for name in writable_names}
        for left_name, left in writable.items():
            if left == self.project_root or self.project_root.is_relative_to(left):
                raise ValueError(f"{left_name} must not equal or contain project_root")
            if (
                left == self.model_path
                or left.is_relative_to(self.model_path)
                or self.model_path.is_relative_to(left)
            ):
                raise ValueError(f"{left_name} must be disjoint from model_path")
            for right_name, right in writable.items():
                if left_name >= right_name:
                    continue
                if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                    raise ValueError(f"{left_name} and {right_name} must be disjoint storage roots")

    def to_mapping(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "model_path": str(self.model_path),
            "data": str(self.data),
            "outputs": str(self.outputs),
            "checkpoints": str(self.checkpoints),
            "trajectories": str(self.trajectories),
            "cache": str(self.cache),
        }


@dataclass(frozen=True, slots=True)
class PilotDataConfig:
    dataset_id: str
    seed: int
    image_size: tuple[int, int]
    chart_types: tuple[str, ...]
    operations: tuple[str, ...]
    split_counts: Mapping[str, int]
    counterfactual_pairs: int
    natural_audit: int

    def __post_init__(self) -> None:
        if self.dataset_id not in _DATASET_DESIGNS:
            supported = ", ".join(sorted(_DATASET_DESIGNS))
            raise ValueError(f"dataset_id must equal one of: {supported}")
        if self.chart_types != ("grouped_bar", "line"):
            raise ValueError("chart_types must equal grouped_bar and line")
        if self.operations != ("difference", "sum", "max_minus_min"):
            raise ValueError("operations must equal difference, sum, and max_minus_min")
        if tuple(self.split_counts) != _SPLITS:
            raise ValueError("split_counts must contain the registered ordered split set")

    @property
    def output_slug(self) -> str:
        return _DATASET_DESIGNS[self.dataset_id][0]

    @property
    def render_mode(self) -> str:
        return _DATASET_DESIGNS[self.dataset_id][1]


def load_pilot_paths(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> PilotPaths:
    raw = load_yaml_mapping(path, label="GPU pilot paths configuration")
    reject_unknown_fields(raw, _PATH_TOP, label="paths configuration")
    if raw.get("schema_version") != 1:
        raise ValueError("paths schema_version must equal 1")
    model = _mapping(raw.get("model"), "model")
    reject_unknown_fields(model, _MODEL_FIELDS, label="model")
    qwen = _mapping(model.get("qwen25vl3b"), "model.qwen25vl3b")
    reject_unknown_fields(qwen, _MODEL_ENTRY_FIELDS, label="model.qwen25vl3b")
    storage = _mapping(raw.get("storage"), "storage")
    reject_unknown_fields(storage, _STORAGE_FIELDS, label="storage")
    environment = os.environ if environ is None else environ

    values: dict[str, object] = {
        "project_root": raw.get("project_root"),
        "model_path": qwen.get("path"),
        **{name: storage.get(name) for name in _STORAGE_FIELDS},
    }
    for field_name, variable in _ENV_OVERRIDES.items():
        override = environment.get(variable)
        if override:
            values[field_name] = override
    return PilotPaths(**{name: _absolute_path(value, name) for name, value in values.items()})


def load_pilot_data_config(path: Path) -> PilotDataConfig:
    raw = load_yaml_mapping(path, label="GPU pilot data configuration")
    reject_unknown_fields(raw, _DATA_TOP, label="pilot data configuration")
    if raw.get("schema_version") != 1:
        raise ValueError("pilot data schema_version must equal 1")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be a non-negative 32-bit integer")
    image_size_raw = raw.get("image_size")
    if not isinstance(image_size_raw, list) or len(image_size_raw) != 2:
        raise ValueError("image_size must contain width and height")
    image_size = tuple(_positive_int(value, "image_size", maximum=2048) for value in image_size_raw)
    chart_types = raw.get("chart_types")
    operations = raw.get("operations")
    if not isinstance(chart_types, list) or not all(isinstance(item, str) for item in chart_types):
        raise ValueError("chart_types must be a string list")
    if not isinstance(operations, list) or not all(isinstance(item, str) for item in operations):
        raise ValueError("operations must be a string list")
    split_raw = _mapping(raw.get("split_counts"), "split_counts")
    reject_unknown_fields(split_raw, frozenset(_SPLITS), label="split_counts")
    split_counts = {
        split: _positive_int(split_raw.get(split), f"split_counts.{split}", maximum=10_000)
        for split in _SPLITS
    }
    return PilotDataConfig(
        dataset_id=str(raw.get("dataset_id")),
        seed=seed,
        image_size=(image_size[0], image_size[1]),
        chart_types=tuple(chart_types),
        operations=tuple(operations),
        split_counts=split_counts,
        counterfactual_pairs=_positive_int(
            raw.get("counterfactual_pairs"), "counterfactual_pairs", maximum=2_000
        ),
        natural_audit=_positive_int(raw.get("natural_audit"), "natural_audit", maximum=2_000),
    )
