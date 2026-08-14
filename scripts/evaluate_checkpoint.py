#!/usr/bin/env python3
"""Run the strictly supported subset of a preregistered checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_MAX_CONFIG_BYTES = 1024 * 1024
_REGISTERED_VLM_SEEDS = (11, 17, 23)
_REGISTERED_HELD_CONSTANTS = (
    "task_rule",
    "semantic_state",
    "answer_distribution",
)
_REGISTERED_METRICS = {
    "outcome": ("exact_answer_accuracy", "numeric_mae", "numeric_mse"),
    "perception": ("state_exact_match", "mean_severity", "error_family_frequency"),
    "reasoning": (
        "oracle_state_accuracy",
        "perceived_state_canonicality",
        "compensator_mode_frequency",
    ),
    "compensability": (
        "per_prompt_compensability",
        "severity_compensability_covariance",
        "relative_compensability_gain",
        "pairwise_odds_residual",
    ),
    "coupling": (
        "perception_loss",
        "reasoning_loss",
        "coupling",
        "outcome_loss",
        "normalized_cancellation",
    ),
    "ood": (
        "error_mechanism_generalization_gap",
        "compensation_generalization_gap",
    ),
}


class AnalysisBlocked(RuntimeError):
    """A required recorded GPU artifact is absent."""


_PREDICTION_FIELDS = frozenset(
    {
        "sample_id",
        "paired_sample_id",
        "image_path",
        "image_sha256",
        "scene",
        "checkpoint_sha256",
        "model_revision",
        "dataset_manifest_sha256",
        "dataset_content_sha256",
        "decoder_revision",
        "seed",
        "error_mechanism",
        "error_family",
        "severity",
        "perceived_scene",
        "canonical_answer",
        "predicted_answer",
        "numeric_target",
        "numeric_prediction",
        "answer_correct",
        "counterfactual_consistent",
        "prompt_error_profile",
        "scaling_probe",
        "selection_probe",
    }
)


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_yaml_mapping(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int, float, bool, type(None), tuple)):
            raise ValueError("YAML mapping keys must be hashable scalars")
        if key in result:
            raise ValueError(f"duplicate YAML mapping key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("checkpoint must not be a symbolic link")
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise AnalysisBlocked(f"checkpoint artifact is missing: {path}")
    nodes = tuple(sorted(path.rglob("*")))
    if any(candidate.is_symlink() for candidate in nodes):
        raise ValueError("checkpoint directory must not contain symbolic links")
    invalid = tuple(
        candidate for candidate in nodes if not candidate.is_file() and not candidate.is_dir()
    )
    if invalid:
        raise ValueError("checkpoint directory contains an unsupported filesystem node")
    files = tuple(candidate for candidate in nodes if candidate.is_file())
    if not files:
        raise ValueError("checkpoint directory must contain at least one file")
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _expect_keys(
    value: object, name: str, *, required: Sequence[str], optional: Sequence[str] = ()
) -> dict[str, Any]:
    result = _mapping(value, name)
    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(result))
    unknown = sorted(set(result) - allowed)
    if missing:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {', '.join(unknown)}")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path(value: object, name: str) -> Path:
    return Path(_text(value, name)).expanduser()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256_value(value: object, name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        qualifier = "null or " if allow_none else ""
        raise ValueError(f"{name} must be {qualifier}a lowercase SHA-256")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisBlocked(f"configuration file is missing: {path}")
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("configuration exceeds 1 MiB")
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictSafeLoader)
        _validate_yaml_tree(loaded)
    except (OSError, RecursionError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read configuration: {error}") from error
    config = _expect_keys(
        loaded,
        "configuration",
        required=(
            "schema_version",
            "experiment",
            "execution_status",
            "selection",
            "inputs",
            "metrics",
            "statistics",
            "shifts",
            "provenance",
            "outputs",
        ),
    )
    if config["schema_version"] != 1:
        raise ValueError("configuration schema_version must equal 1")
    return config


def _validate_yaml_tree(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if depth > 64 or nodes > 100_000:
            raise ValueError("configuration YAML exceeds depth/node safety limits")
        if isinstance(node, (Mapping, list)):
            identity = id(node)
            if identity in seen:
                raise ValueError("configuration YAML aliases/cycles are forbidden")
            seen.add(identity)
            children = node.values() if isinstance(node, Mapping) else node
            stack.extend((child, depth + 1) for child in children)


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_json_value(value: object, path: str = "$", depth: int = 0) -> None:
    if depth > 20:
        raise ValueError(f"JSON nesting exceeds 20 levels at {path}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {path}")
    if isinstance(value, str) and len(value) > 100_000:
        raise ValueError(f"JSON string exceeds 100000 characters at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_json_value(child, f"{path}.{key}", depth + 1)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]", depth + 1)


def _reject_sensitive_or_local_strings(value: object, path: str = "$", depth: int = 0) -> None:
    if depth > 20:
        raise ValueError(f"JSON nesting exceeds 20 levels at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(
                token in lowered
                for token in ("api_key", "apikey", "password", "passwd", "secret", "token")
            ):
                raise ValueError(f"sensitive field name is forbidden at {path}.{key}")
            _reject_sensitive_or_local_strings(child, f"{path}.{key}", depth + 1)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_or_local_strings(child, f"{path}[{index}]", depth + 1)
    elif isinstance(value, str):
        lowered = value.lower()
        if value.startswith(("/Users/", "/home/", "C:\\Users\\")):
            raise ValueError(f"machine-specific absolute path is forbidden at {path}")
        if any(marker in lowered for marker in ("-----begin private key-----", "sk-")):
            raise ValueError(f"secret-like string is forbidden at {path}")


def _read_hashed_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisBlocked(f"required {name} is missing: {path}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"{name} exceeds 16 MiB")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected_sha256}, got {observed}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, RecursionError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot parse {name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    _validate_json_value(value)
    _reject_sensitive_or_local_strings(value)
    return value


def _read_jsonl(
    path: Path, *, label: str, max_bytes: int, max_line_bytes: int
) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise AnalysisBlocked(f"required GPU {label} prediction artifact is missing: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{label} prediction JSONL must not be empty")
    if size > max_bytes:
        raise ValueError(f"{label} prediction JSONL exceeds max_jsonl_bytes")
    records: list[dict[str, object]] = []
    try:
        with path.open("rb") as stream:
            for line_number, raw_line_with_ending in enumerate(stream, start=1):
                if len(raw_line_with_ending) > max_line_bytes + 2:
                    raise ValueError(
                        f"{label} prediction JSONL line {line_number} exceeds max_jsonl_line_bytes"
                    )
                raw_line = raw_line_with_ending.rstrip(b"\r\n")
                if not raw_line.strip():
                    raise ValueError(
                        f"{label} prediction JSONL contains a blank line at line {line_number}"
                    )
                if len(raw_line) > max_line_bytes:
                    raise ValueError(
                        f"{label} prediction JSONL line {line_number} exceeds max_jsonl_line_bytes"
                    )
                try:
                    value = json.loads(
                        raw_line.decode("utf-8"),
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                except (RecursionError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"invalid {label} prediction JSONL at line {line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{label} prediction JSONL line {line_number} must be a JSON object"
                    )
                _validate_json_value(value)
                _reject_sensitive_or_local_strings(value)
                records.append(value)
    except OSError as error:
        raise ValueError(f"cannot read {label} prediction JSONL: {error}") from error
    if not records:
        raise ValueError(f"{label} prediction JSONL must not be empty")
    return tuple(records)


def _string_list(value: object, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}list")
    result = tuple(_text(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validate_protocol(config: Mapping[str, Any]) -> dict[str, object]:
    experiment = _text(config["experiment"], "experiment")
    _text(config["execution_status"], "execution_status")
    selection = _expect_keys(
        config["selection"],
        "selection",
        required=("calibration_split_only", "test_set_checkpoint_selection"),
    )
    if selection["calibration_split_only"] is not True:
        raise ValueError("selection.calibration_split_only must be true")
    if selection["test_set_checkpoint_selection"] != "forbidden":
        raise ValueError("selection.test_set_checkpoint_selection must be forbidden")

    inputs = _expect_keys(
        config["inputs"],
        "inputs",
        required=(
            "iid_predictions",
            "ood_predictions",
            "dataset_manifest",
            "prediction_manifest",
            "prediction_manifest_sha256",
            "execution_audit",
            "execution_audit_sha256",
            "phase_d_audit",
            "phase_d_audit_sha256",
            "max_jsonl_bytes",
            "max_jsonl_line_bytes",
        ),
    )
    metrics = _expect_keys(
        config["metrics"],
        "metrics",
        required=(
            "outcome",
            "perception",
            "reasoning",
            "compensability",
            "coupling",
            "ood",
        ),
    )
    requested: list[str] = []
    for family in (
        "outcome",
        "perception",
        "reasoning",
        "compensability",
        "coupling",
        "ood",
    ):
        family_metrics = _string_list(metrics[family], f"metrics.{family}", nonempty=False)
        if family_metrics != _REGISTERED_METRICS[family]:
            raise ValueError(f"metrics.{family} must equal the registered frozen metric list")
        requested.extend(family_metrics)
    if not requested:
        raise ValueError("metrics must request at least one metric")
    if len(set(requested)) != len(requested):
        raise ValueError("requested metric names must be unique across metric families")

    statistics = _expect_keys(
        config["statistics"],
        "statistics",
        required=(
            "vlm_seeds",
            "bootstrap_resamples",
            "confidence",
            "paired_bootstrap",
            "multiple_comparisons",
        ),
    )
    seeds = statistics["vlm_seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("statistics.vlm_seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("statistics.vlm_seeds must contain integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("statistics.vlm_seeds must be unique")
    if tuple(seeds) != _REGISTERED_VLM_SEEDS:
        raise ValueError("statistics.vlm_seeds must equal the registered seeds [11, 17, 23]")
    bootstrap_resamples = _positive_integer(
        statistics["bootstrap_resamples"], "bootstrap_resamples"
    )
    if bootstrap_resamples != 10_000:
        raise ValueError("statistics.bootstrap_resamples must equal the registered value 10000")
    confidence = statistics["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("statistics.confidence must be numeric")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("statistics.confidence must lie strictly between zero and one")
    if statistics["paired_bootstrap"] is not True:
        raise ValueError("statistics.paired_bootstrap must be true")
    if statistics["multiple_comparisons"] != "holm":
        raise ValueError("statistics.multiple_comparisons must be holm")

    shifts = _expect_keys(config["shifts"], "shifts", required=("primary", "held_constant"))
    primary_shift = _text(shifts["primary"], "shifts.primary")
    if primary_shift != "error_mechanism":
        raise ValueError("this evaluator currently supports only the error_mechanism shift")
    held_constant = _string_list(shifts["held_constant"], "shifts.held_constant")
    if held_constant != _REGISTERED_HELD_CONSTANTS:
        raise ValueError(
            "shifts.held_constant must equal the registered frozen controls "
            "[task_rule, semantic_state, answer_distribution]"
        )

    provenance = _expect_keys(
        config["provenance"],
        "provenance",
        required=("model_revision", "verl_revision", "decoder_revision"),
    )
    outputs = _expect_keys(
        config["outputs"],
        "outputs",
        required=("metrics", "report", "log_root"),
    )
    return {
        "experiment": experiment,
        "execution_status": config["execution_status"],
        "iid": _path(inputs["iid_predictions"], "inputs.iid_predictions"),
        "ood": _path(inputs["ood_predictions"], "inputs.ood_predictions"),
        "manifest": _path(inputs["dataset_manifest"], "inputs.dataset_manifest"),
        "prediction_manifest": _path(inputs["prediction_manifest"], "inputs.prediction_manifest"),
        "prediction_manifest_sha256": _sha256_value(
            inputs["prediction_manifest_sha256"],
            "inputs.prediction_manifest_sha256",
            allow_none=True,
        ),
        "execution_audit": _path(inputs["execution_audit"], "inputs.execution_audit"),
        "execution_audit_sha256": _sha256_value(
            inputs["execution_audit_sha256"], "inputs.execution_audit_sha256", allow_none=True
        ),
        "phase_d_audit": _path(inputs["phase_d_audit"], "inputs.phase_d_audit"),
        "phase_d_audit_sha256": _sha256_value(
            inputs["phase_d_audit_sha256"], "inputs.phase_d_audit_sha256", allow_none=True
        ),
        "max_bytes": _positive_integer(inputs["max_jsonl_bytes"], "max_jsonl_bytes"),
        "max_line_bytes": _positive_integer(inputs["max_jsonl_line_bytes"], "max_jsonl_line_bytes"),
        "requested": tuple(requested),
        "primary_shift": primary_shift,
        "held_constant": held_constant,
        "seed": int(seeds[0]),
        "seeds": tuple(seeds),
        "bootstrap_resamples": bootstrap_resamples,
        "confidence": float(confidence),
        "model_revision": _text(provenance["model_revision"], "model_revision"),
        "verl_revision": _text(provenance["verl_revision"], "verl_revision"),
        "decoder_revision": _text(provenance["decoder_revision"], "decoder_revision"),
        "metrics_output": _path(outputs["metrics"], "outputs.metrics"),
        "report_output": _path(outputs["report"], "outputs.report"),
        "log_root": _path(outputs["log_root"], "outputs.log_root"),
    }


def _ensure_new_paths(*paths: Path) -> None:
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"output already exists; refusing to overwrite: {path}")


def _require_disjoint_paths(*, inputs: Mapping[str, Path], outputs: Mapping[str, Path]) -> None:
    """Reject output trees that can mutate or replace any hashed input tree."""

    normalized_inputs = {
        name: path.expanduser().resolve(strict=False) for name, path in inputs.items()
    }
    normalized_outputs = {
        name: path.expanduser().resolve(strict=False) for name, path in outputs.items()
    }
    for output_name, output in normalized_outputs.items():
        for input_name, input_path in normalized_inputs.items():
            if output == input_path or output in input_path.parents or input_path in output.parents:
                raise ValueError(
                    f"output {output_name} must be disjoint from hashed input {input_name}"
                )
    output_items = tuple(normalized_outputs.items())
    for index, (left_name, left) in enumerate(output_items):
        for right_name, right in output_items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"outputs {left_name} and {right_name} must be disjoint")


def _require_safe_path(path: Path, *, root: Path, name: str) -> Path:
    lexical_root = Path(os.path.abspath(os.fspath(root.expanduser())))
    lexical_path = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical_root.is_symlink():
        raise ValueError("artifact root must not be a symbolic link")
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside the approved artifact root") from error
    cursor = lexical_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{name} must not traverse a symbolic-link parent")

    resolved_root = lexical_root.resolve()
    resolved = lexical_path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside the approved artifact root") from error
    return path


def _validate_prediction_records(
    records: tuple[dict[str, object], ...],
    *,
    label: str,
    checkpoint_hash: str,
    model_revision: str,
    dataset_manifest_hash: str,
    dataset_content_hash: str,
    decoder_revision: str,
    allowed_seeds: frozenset[int],
    allowed_sample_ids: frozenset[str],
    expected_scenes: Mapping[str, object],
    expected_records: Mapping[str, object],
    expected_pair_ids: Mapping[str, str],
    expected_image_hashes: Mapping[str, str],
) -> None:
    observed_pairs: set[tuple[str, int]] = set()
    for index, record in enumerate(records, start=1):
        missing = sorted(_PREDICTION_FIELDS - set(record))
        unknown = sorted(set(record) - _PREDICTION_FIELDS)
        if missing:
            raise ValueError(f"{label} prediction {index} is missing fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"{label} prediction {index} has unknown fields: {', '.join(unknown)}")
        expected_values = {
            "checkpoint_sha256": checkpoint_hash,
            "model_revision": model_revision,
            "dataset_manifest_sha256": dataset_manifest_hash,
            "dataset_content_sha256": dataset_content_hash,
            "decoder_revision": decoder_revision,
        }
        for name, expected in expected_values.items():
            if record[name] != expected:
                raise ValueError(
                    f"{label} prediction {index} {name} does not match the recorded source"
                )
        if record["seed"] not in allowed_seeds:
            raise ValueError(f"{label} prediction {index} seed is not preregistered")
        if record["sample_id"] not in allowed_sample_ids:
            raise ValueError(f"{label} prediction {index} sample_id is outside the dataset")
        sample_id = str(record["sample_id"])
        if record["paired_sample_id"] != expected_pair_ids[sample_id]:
            raise ValueError(f"{label} prediction {index} paired_sample_id is invalid")
        if record["scene"] != expected_scenes[sample_id]:
            raise ValueError(
                f"{label} prediction {index} scene differs from the frozen dataset record"
            )
        sample = expected_records[sample_id]
        sample_mapping = sample.to_mapping()  # type: ignore[union-attr]
        if record["image_path"] != sample_mapping["image_path"]:
            raise ValueError(f"{label} prediction {index} image_path differs from dataset")
        if record["image_sha256"] != expected_image_hashes[sample_id]:
            raise ValueError(f"{label} prediction {index} image SHA-256 differs from manifest")
        catalog = {error.error_id: error for error in sample.error_catalog}  # type: ignore[union-attr]
        mechanism = record["error_mechanism"]
        if not isinstance(mechanism, str) or mechanism not in catalog:
            raise ValueError(f"{label} prediction {index} error_mechanism is not registered")
        error = catalog[mechanism]
        if record["error_family"] != error.family or record["severity"] != error.severity:
            raise ValueError(f"{label} prediction {index} error metadata differs from catalog")
        from compbias.envs.cva_world.corruptions import apply_error

        expected_perceived = dict(apply_error(sample.scene, error))  # type: ignore[union-attr]
        from compbias.io.manifests import canonical_json

        if canonical_json(record["perceived_scene"]) != canonical_json(expected_perceived):
            raise ValueError(f"{label} prediction {index} perceived_scene is not reproducible")
        if record["canonical_answer"] != sample_mapping["canonical_answer"]:
            raise ValueError(f"{label} prediction {index} canonical_answer differs from dataset")
        canonical = sample_mapping["canonical_answer"]
        numeric = isinstance(canonical, (int, float)) and not isinstance(canonical, bool)
        if record["numeric_target"] != (float(canonical) if numeric else None):
            raise ValueError(f"{label} prediction {index} numeric_target is inconsistent")
        prediction = record["predicted_answer"]
        if record["answer_correct"] is not (prediction == canonical):
            raise ValueError(f"{label} prediction {index} answer_correct is inconsistent")
        if record["counterfactual_consistent"] is not record["answer_correct"]:
            raise ValueError(
                f"{label} prediction {index} counterfactual_consistent is inconsistent"
            )
        numeric_prediction = record["numeric_prediction"]
        if numeric:
            if (
                isinstance(numeric_prediction, bool)
                or not isinstance(numeric_prediction, (int, float))
                or not math.isfinite(float(numeric_prediction))
                or prediction != numeric_prediction
            ):
                raise ValueError(f"{label} prediction {index} numeric_prediction is invalid")
        elif numeric_prediction is not None:
            raise ValueError(f"{label} prediction {index} numeric_prediction must be null")
        profile = record["prompt_error_profile"]
        if not isinstance(profile, list) or len(profile) != len(catalog):
            raise ValueError(f"{label} prediction {index} prompt_error_profile is incomplete")
        expected_profile_ids = [
            error.error_id
            for error in sample.error_catalog  # type: ignore[union-attr]
        ]
        observed_profile_ids = [
            entry.get("error_id") for entry in profile if isinstance(entry, Mapping)
        ]
        if observed_profile_ids != expected_profile_ids:
            raise ValueError(
                f"{label} prediction {index} prompt_error_profile order differs from catalog"
            )
        probability_sum = 0.0
        for profile_index, entry in enumerate(profile):
            profile_name = f"{label} prediction {index} prompt_error_profile[{profile_index}]"
            closed_entry = _expect_keys(
                entry,
                profile_name,
                required=(
                    "error_id",
                    "severity",
                    "base_probability",
                    "rollout_rewards",
                ),
            )
            profile_error = catalog[str(closed_entry["error_id"])]
            if closed_entry["severity"] != profile_error.severity:
                raise ValueError(f"{profile_name} severity differs from catalog")
            probability = closed_entry["base_probability"]
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or not 0.0 <= float(probability) <= 1.0
            ):
                raise ValueError(f"{profile_name} base_probability must lie in [0, 1]")
            rewards = closed_entry["rollout_rewards"]
            if (
                not isinstance(rewards, list)
                or len(rewards) != 32
                or any(value not in {0, 1} or isinstance(value, bool) for value in rewards)
            ):
                raise ValueError(f"{profile_name} must contain 32 binary intervention rewards")
            probability_sum += float(probability)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} prediction {index} profile probabilities must sum to one")
        scaling = _expect_keys(
            record["scaling_probe"],
            f"{label} prediction {index} scaling_probe",
            required=("multiplier", "multiplier_derivative"),
        )
        for field in ("multiplier", "multiplier_derivative"):
            value = scaling[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{label} prediction {index} scaling_probe {field} is invalid")
        if float(scaling["multiplier"]) <= 0.0:
            raise ValueError(f"{label} prediction {index} scaling multiplier must be positive")
        selection = _expect_keys(
            record["selection_probe"],
            f"{label} prediction {index} selection_probe",
            required=("reference_probabilities", "selected_probabilities", "rewards", "beta"),
        )
        reference = selection["reference_probabilities"]
        selected = selection["selected_probabilities"]
        rewards = selection["rewards"]
        if (
            not isinstance(reference, list)
            or not isinstance(selected, list)
            or not isinstance(rewards, list)
            or len(reference) != len(selected)
            or len(reference) != len(rewards)
            or len(reference) < 2
        ):
            raise ValueError(f"{label} prediction {index} selection probe vectors mismatch")
        for values, field in ((reference, "reference"), (selected, "selected")):
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in values
            ) or not math.isclose(sum(float(value) for value in values), 1.0, abs_tol=1e-12):
                raise ValueError(f"{label} prediction {index} {field} probabilities are invalid")
        beta = selection["beta"]
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or float(beta) <= 0.0:
            raise ValueError(f"{label} prediction {index} selection beta must be positive")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in rewards
        ):
            raise ValueError(f"{label} prediction {index} selection rewards are invalid")
        pair = (sample_id, int(record["seed"]))
        if pair in observed_pairs:
            raise ValueError(f"{label} predictions contain duplicate sample/seed pair {pair!r}")
        observed_pairs.add(pair)
    expected_pairs = {
        (sample_id, seed) for sample_id in allowed_sample_ids for seed in allowed_seeds
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            f"{label} predictions do not have exact sample-by-seed Cartesian coverage; "
            f"missing={missing!r}, extra={extra!r}"
        )


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_text_bundle(items: Sequence[tuple[Path, str]]) -> None:
    written: list[Path] = []
    try:
        for path, content in items:
            _write_new_text(path, content)
            written.append(path)
    except BaseException:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise


def _command(argv: Sequence[str] | None) -> tuple[str, ...]:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    return (sys.executable, str(Path(__file__).resolve()), *arguments)


def _render_report(payload: Mapping[str, object]) -> str:
    computed = _mapping(payload["computed_metrics"], "computed_metrics")
    uncomputed = _mapping(payload["uncomputed_metrics"], "uncomputed_metrics")
    lines = [
        "# Checkpoint evaluation",
        "",
        f"Status: `{payload['status']}`",
        "",
    ]
    if payload["full_evaluation_complete"] is not True:
        lines.extend(
            [
                "This is not a completed full evaluation. Only metrics derivable from the two ",
                "strictly paired prediction files are recorded; every other preregistered metric ",
                "remains gated.",
                "",
            ]
        )
    lines.extend(["## Computed metrics", "", "| Metric | Value |", "|---|---:|"])
    for name, value in sorted(computed.items()):
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
        lines.append(f"| `{name}` | `{rendered}` |")
    lines.extend(["", "## Uncomputed preregistered metrics", ""])
    if uncomputed:
        lines.extend(["| Metric | Gate |", "|---|---|"])
        for name, reason in sorted(uncomputed.items()):
            lines.append(f"| `{name}` | {reason} |")
    else:
        lines.append("None.")
    lines.extend(["", "## Source hashes", ""])
    for name, digest in sorted(_mapping(payload["source_hashes"], "source_hashes").items()):
        lines.append(f"- `{name}`: `{digest}`")
    return "\n".join(lines) + "\n"


def _run_evaluation(
    config_path: Path,
    checkpoint_path: Path,
    artifact_root: Path,
    argv: Sequence[str] | None,
) -> bool:
    from compbias.eval.checkpoint_metrics import full_checkpoint_metrics
    from compbias.eval.dataset_contract import validate_frozen_cva_v2_dataset
    from compbias.eval.ood import compute_ood_metrics
    from compbias.eval.paired_inference import paired_ood_inference
    from compbias.io.logging import (
        RunLogger,
        capture_environment,
        publishable_config_snapshot,
        publishable_path,
    )

    config = _load_config(config_path)
    # Publication safety is a configuration boundary, so validate it before
    # any expensive frozen-dataset reconstruction or evidence replay.
    _reject_sensitive_or_local_strings(config, path="$.configuration")
    protocol = _validate_protocol(config)
    for key in (
        "iid",
        "ood",
        "manifest",
        "prediction_manifest",
        "execution_audit",
        "phase_d_audit",
        "metrics_output",
        "report_output",
        "log_root",
    ):
        path = protocol[key]
        assert isinstance(path, Path)
        _require_safe_path(
            path,
            root=artifact_root,
            name=key,
        )
    _require_safe_path(
        checkpoint_path,
        root=artifact_root,
        name="checkpoint",
    )
    checkpoint_hash = _checkpoint_sha256(checkpoint_path)

    iid_path = protocol["iid"]
    ood_path = protocol["ood"]
    assert isinstance(iid_path, Path)
    assert isinstance(ood_path, Path)
    iid_records = _read_jsonl(
        iid_path,
        label="IID",
        max_bytes=int(protocol["max_bytes"]),
        max_line_bytes=int(protocol["max_line_bytes"]),
    )
    ood_records = _read_jsonl(
        ood_path,
        label="OOD",
        max_bytes=int(protocol["max_bytes"]),
        max_line_bytes=int(protocol["max_line_bytes"]),
    )
    manifest_path = protocol["manifest"]
    assert isinstance(manifest_path, Path)
    if not manifest_path.is_file():
        raise AnalysisBlocked(f"required dataset manifest is missing: {manifest_path}")
    dataset = validate_frozen_cva_v2_dataset(manifest_path, artifact_root=artifact_root)
    manifest_hash = dataset.manifest_file_sha256
    dataset_content_hash = dataset.content_sha256
    ood_sample_ids = dataset.sample_ids_for_partition("ood_test")
    ood_records_by_id = {sample_id: dataset.records[sample_id] for sample_id in ood_sample_ids}
    evaluation_sample_ids = tuple(
        sorted(str(ood_records_by_id[sample_id].source_id) for sample_id in ood_sample_ids)
    )
    if len(evaluation_sample_ids) != 100 or len(set(evaluation_sample_ids)) != 100:
        raise ValueError("frozen OOD rows must define exactly 100 unique source pairs")
    expected_records = {
        sample_id: dataset.records[sample_id] for sample_id in evaluation_sample_ids
    }
    expected_ood_records = {sample.sample_id: sample for sample in ood_records_by_id.values()}
    iid_pair_ids = {
        sample_id: next(
            ood_id
            for ood_id, sample in expected_ood_records.items()
            if sample.source_id == sample_id
        )
        for sample_id in evaluation_sample_ids
    }
    ood_pair_ids = {
        sample_id: str(sample.source_id) for sample_id, sample in expected_ood_records.items()
    }
    for ood_id, ood_sample in expected_ood_records.items():
        source = expected_records[str(ood_sample.source_id)]
        source_mapping = source.to_mapping()
        ood_mapping = ood_sample.to_mapping()
        for field in ("task_family", "scene", "question", "canonical_answer"):
            if source_mapping[field] != ood_mapping[field]:
                raise ValueError(f"frozen OOD pair {ood_id} changes held-constant {field}")
        if (
            source.split_keys.visual_style == ood_sample.split_keys.visual_style
            or source.split_keys.error_mechanism == ood_sample.split_keys.error_mechanism
        ):
            raise ValueError(f"frozen OOD pair {ood_id} does not shift registered factors")
    image_hashes = dataset.image_sha256
    prediction_manifest_hash = protocol["prediction_manifest_sha256"]
    if not isinstance(prediction_manifest_hash, str):
        raise AnalysisBlocked(
            "prediction_manifest_sha256 is absent; predictions cannot be bound to the checkpoint"
        )
    prediction_manifest_path = protocol["prediction_manifest"]
    assert isinstance(prediction_manifest_path, Path)
    prediction_manifest = _read_hashed_json(
        prediction_manifest_path,
        prediction_manifest_hash,
        "prediction manifest",
    )
    phase_d_hash = protocol["phase_d_audit_sha256"]
    execution_audit_hash = protocol["execution_audit_sha256"]
    if not isinstance(phase_d_hash, str) or not isinstance(execution_audit_hash, str):
        raise AnalysisBlocked("reviewed Phase-D/post-GPU execution audit hashes are absent")
    phase_d_path = protocol["phase_d_audit"]
    execution_audit_path = protocol["execution_audit"]
    assert isinstance(phase_d_path, Path)
    assert isinstance(execution_audit_path, Path)
    phase_d_audit = _read_hashed_json(phase_d_path, phase_d_hash, "Phase-D audit")
    execution_audit = _read_hashed_json(
        execution_audit_path, execution_audit_hash, "post-GPU execution audit"
    )
    from compbias.eval.post_gpu_evidence import validate_ready_phase_d_audit
    from compbias.io.manifests import manifest_sha256

    validate_ready_phase_d_audit(
        phase_d_audit,
        dataset_manifest_sha256=manifest_hash,
        dataset_manifest_self_sha256=dataset.manifest_self_sha256,
        dataset_content_sha256=dataset_content_hash,
        dataset_image_set_sha256=manifest_sha256(dataset.image_sha256),
        sample_ids=tuple(dataset.records),
    )
    if protocol["execution_status"] != "recorded_gpu_artifacts":
        raise AnalysisBlocked(
            "execution_status is not recorded_gpu_artifacts; no full evaluation may be claimed"
        )
    if (
        prediction_manifest.get("schema_version") != 1
        or prediction_manifest.get("artifact_type") != "paired_checkpoint_predictions"
    ):
        raise ValueError("prediction manifest schema or artifact_type is invalid")
    bindings = {
        "checkpoint_sha256": checkpoint_hash,
        "model_revision": protocol["model_revision"],
        "dataset_manifest_sha256": manifest_hash,
        "dataset_content_sha256": dataset_content_hash,
        "decoder_revision": protocol["decoder_revision"],
        "vlm_seeds": list(protocol["seeds"]),
        "primary_shift": protocol["primary_shift"],
    }
    for name, expected in bindings.items():
        if prediction_manifest.get(name) != expected:
            raise ValueError(
                f"prediction manifest {name} does not match the checkpoint/config source"
            )
    producer_config_hash = _sha256_value(
        prediction_manifest.get("producer_config_sha256"),
        "prediction manifest producer_config_sha256",
    )
    if (
        prediction_manifest.get("dataset_partition") != "paired_iid_ood_test"
        or prediction_manifest.get("prediction_scope") != "exact_100_source_pairs"
    ):
        raise ValueError("prediction manifest must bind the registered exact 100 OOD source pairs")
    if prediction_manifest.get("iid_sample_ids") != list(evaluation_sample_ids):
        raise ValueError("prediction manifest must cover the exact 100 IID/OOD source pairs")
    if prediction_manifest.get("ood_sample_ids") != list(ood_sample_ids):
        raise ValueError("prediction manifest must cover exact frozen OOD rows")
    for label, path, records, registered_records, pair_ids in (
        ("iid", iid_path, iid_records, expected_records, iid_pair_ids),
        ("ood", ood_path, ood_records, expected_ood_records, ood_pair_ids),
    ):
        source = prediction_manifest.get(label)
        if not isinstance(source, Mapping):
            raise ValueError(f"prediction manifest {label} source record is missing")
        if source.get("sha256") != _sha256_file(path):
            raise ValueError(f"prediction manifest {label} SHA-256 does not match")
        if source.get("record_count") != len(records):
            raise ValueError(f"prediction manifest {label} record_count does not match")
        _validate_prediction_records(
            records,
            label=label.upper(),
            checkpoint_hash=checkpoint_hash,
            model_revision=str(protocol["model_revision"]),
            dataset_manifest_hash=manifest_hash,
            dataset_content_hash=str(dataset_content_hash),
            decoder_revision=str(protocol["decoder_revision"]),
            allowed_seeds=frozenset(protocol["seeds"]),
            allowed_sample_ids=frozenset(registered_records),
            expected_scenes={
                sample_id: sample.to_mapping()["scene"]
                for sample_id, sample in registered_records.items()
            },
            expected_records=registered_records,
            expected_pair_ids=pair_ids,
            expected_image_hashes={
                sample_id: str(image_hashes[sample_id]) for sample_id in registered_records
            },
        )

    from compbias.eval.post_gpu_evidence import (
        PostGPUAuthenticationPending,
        validate_post_gpu_execution_audit,
    )

    try:
        validate_post_gpu_execution_audit(
            execution_audit,
            artifact_type="checkpoint_evaluation_execution_audit",
            stage="full_checkpoint_evaluation",
            checkpoint_sha256=checkpoint_hash,
            dataset_manifest_sha256=manifest_hash,
            dataset_manifest_self_sha256=dataset.manifest_self_sha256,
            dataset_content_sha256=dataset_content_hash,
            phase_d_audit_sha256=phase_d_hash,
            prediction_or_rollout_manifest_sha256=prediction_manifest_hash,
            producer_config_sha256=str(producer_config_hash),
            producer_records_path=iid_path,
            producer_records_sha256=_sha256_file(iid_path),
            producer_record_count=len(iid_records),
            seeds=protocol["seeds"],
            model_revision=str(protocol["model_revision"]),
            verl_revision=str(protocol["verl_revision"]),
            sha256_file=_sha256_file,
        )
    except PostGPUAuthenticationPending as error:
        raise AnalysisBlocked(str(error)) from error
    raise AnalysisBlocked("unreachable until authenticated post-GPU gate extension exists")

    metrics_output = protocol["metrics_output"]
    report_output = protocol["report_output"]
    log_root = protocol["log_root"]
    assert isinstance(metrics_output, Path)
    assert isinstance(report_output, Path)
    assert isinstance(log_root, Path)
    _ensure_new_paths(metrics_output, report_output)

    per_seed_metrics = {}
    for seed in protocol["seeds"]:
        iid_for_seed = tuple(record for record in iid_records if record["seed"] == seed)
        ood_for_seed = tuple(record for record in ood_records if record["seed"] == seed)
        per_seed_metrics[str(seed)] = compute_ood_metrics(
            iid_for_seed,
            tuple({**record, "sample_id": record["paired_sample_id"]} for record in ood_for_seed),
            preregistered_shift=(str(protocol["primary_shift"]),),
        )
    aggregate_iid = tuple(
        {**record, "sample_id": f"{record['sample_id']}::seed={record['seed']}"}
        for record in iid_records
    )
    aggregate_ood = tuple(
        {**record, "sample_id": f"{record['paired_sample_id']}::seed={record['seed']}"}
        for record in ood_records
    )
    ood_metrics = compute_ood_metrics(
        aggregate_iid,
        aggregate_ood,
        preregistered_shift=(str(protocol["primary_shift"]),),
    )
    raw_ood = {
        "iid_accuracy": ood_metrics.iid_accuracy,
        "ood_accuracy": ood_metrics.ood_accuracy,
        "error_mechanism_generalization_gap": (ood_metrics.error_mechanism_generalization_gap),
        "iid_compensatory_count": ood_metrics.iid_compensatory_count,
        "ood_compensatory_count": ood_metrics.ood_compensatory_count,
        "shifted_factors": list(ood_metrics.shifted_factors),
        "n_pairs": ood_metrics.n_pairs,
        "unsupported_diagnostics": {
            "compensation_generalization_gap": None,
            "counterfactual_consistency_gap": None,
            "reason": "requires additional independently bound primitive artifacts",
        },
    }
    compensation_defined = ood_metrics.compensation_generalization_gap is not None and all(
        metrics.compensation_generalization_gap is not None for metrics in per_seed_metrics.values()
    )
    supported = full_checkpoint_metrics(
        iid_records,
        ood_records,
        error_mechanism_generalization_gap=ood_metrics.error_mechanism_generalization_gap,
    )
    unsupported_provenance = {
        "state_exact_match",
        "oracle_state_accuracy",
        "perceived_state_canonicality",
        "compensator_mode_frequency",
        "perception_loss",
        "reasoning_loss",
        "coupling",
        "outcome_loss",
        "normalized_cancellation",
        "compensation_generalization_gap",
    }
    requested = protocol["requested"]
    assert isinstance(requested, tuple)
    computed = {name: supported[name] for name in requested if name in supported}
    uncomputed = {
        name: (
            "zero compensatory denominator in at least one registered seed/partition; "
            "the conditional compensation gap is undefined"
            if name == "compensation_generalization_gap" and not compensation_defined
            else (
                "requires hash-bound raw model text, parser status, perceived-state and "
                "reasoning-action primitives; self-reported derived fields are forbidden"
                if name in unsupported_provenance
                else "the registered metric could not be computed from its required primitives"
            )
        )
        for name in sorted(requested)
        if name not in supported
    }
    inference_metric_names = tuple(
        name
        for name in computed
        if name
        in {
            "error_mechanism_generalization_gap",
            "compensation_generalization_gap",
        }
    )
    statistical_inference = paired_ood_inference(
        iid_records,
        ood_records,
        seeds=protocol["seeds"],
        metric_names=inference_metric_names,
        confidence=float(protocol["confidence"]),
        n_resamples=int(protocol["bootstrap_resamples"]),
    )
    complete = not uncomputed
    status = "COMPLETE" if complete else "PARTIAL_GATE"
    config_hash = _sha256_file(config_path)
    source_hashes = {
        "config": config_hash,
        "dataset_manifest": manifest_hash,
        "dataset_manifest_self": dataset.manifest_self_sha256,
        "dataset_content": dataset_content_hash,
        "dataset_file": dataset.dataset_file_sha256,
        "prediction_manifest": prediction_manifest_hash,
        "execution_audit": execution_audit_hash,
        "phase_d_audit": phase_d_hash,
        "iid_predictions": _sha256_file(iid_path),
        "ood_predictions": _sha256_file(ood_path),
    }
    payload = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "status": status,
        "full_evaluation_complete": complete,
        "checkpoint": {
            "path": publishable_path(checkpoint_path, worktree=Path.cwd()),
            "sha256": checkpoint_hash,
        },
        "shift": {
            "primary": protocol["primary_shift"],
            "held_constant": list(protocol["held_constant"]),
        },
        "evaluation_scope": {
            "dataset_name": "cva_v2",
            "dataset_partition": "paired_iid_ood_test",
            "prediction_scope": "exact_100_source_pairs",
            "pair_count": len(ood_sample_ids),
            "iid_sample_ids": list(evaluation_sample_ids),
            "ood_sample_ids": list(ood_sample_ids),
            "seeds": list(protocol["seeds"]),
        },
        "computed_metrics": computed,
        "uncomputed_metrics": uncomputed,
        "statistical_inference": statistical_inference,
        "paired_ood_diagnostics": raw_ood,
        "per_seed_ood_diagnostics": {
            seed: {
                "iid_accuracy": metrics.iid_accuracy,
                "ood_accuracy": metrics.ood_accuracy,
                "error_mechanism_generalization_gap": (metrics.error_mechanism_generalization_gap),
                "iid_compensatory_count": metrics.iid_compensatory_count,
                "ood_compensatory_count": metrics.ood_compensatory_count,
                "n_pairs": metrics.n_pairs,
            }
            for seed, metrics in per_seed_metrics.items()
        },
        "source_hashes": source_hashes,
    }
    report = _render_report(payload)
    run_id = (
        f"checkpoint-{checkpoint_hash[:12]}-iid-{source_hashes['iid_predictions'][:12]}-"
        f"config-{config_hash[:12]}"
    )
    run_dir = log_root / str(protocol["experiment"]) / run_id
    _require_disjoint_paths(
        inputs={
            "config": config_path,
            "checkpoint": checkpoint_path,
            "iid_predictions": iid_path,
            "ood_predictions": ood_path,
            "dataset_manifest": manifest_path,
            "prediction_manifest": prediction_manifest_path,
            "execution_audit": execution_audit_path,
            "phase_d_audit": phase_d_path,
        },
        outputs={
            "metrics": metrics_output,
            "report": report_output,
            "run_log": run_dir,
        },
    )
    if os.path.lexists(run_dir):
        raise FileExistsError(f"run log already exists; refusing to overwrite: {run_dir}")
    environment = capture_environment(
        worktree=Path.cwd(),
        dataset_manifest_hash=manifest_hash,
        seed=int(protocol["seed"]),
        model_revision=str(protocol["model_revision"]),
        verl_revision=str(protocol["verl_revision"]),
        command=_command(argv),
    )
    publishable_config = publishable_config_snapshot(
        config,
        worktree=Path.cwd(),
        path_fields=(
            ("inputs", "iid_predictions"),
            ("inputs", "ood_predictions"),
            ("inputs", "dataset_manifest"),
            ("inputs", "prediction_manifest"),
            ("inputs", "execution_audit"),
            ("inputs", "phase_d_audit"),
            ("outputs", "metrics"),
            ("outputs", "report"),
            ("outputs", "log_root"),
        ),
    )
    _reject_sensitive_or_local_strings(publishable_config, path="$.logged_config")
    logged_config = {**publishable_config, "analysis_provenance": source_hashes}
    with RunLogger(
        root=log_root,
        experiment=str(protocol["experiment"]),
        run_id=run_id,
        config=logged_config,
        environment=environment,
    ) as logger:
        logger.log_metrics(payload)
        for partition, records in (
            ("iid_test", iid_records),
            ("error_mechanism_ood", ood_records),
        ):
            for record in records:
                logger.log_rollout({**record, "partition": partition})
        logger.save_predictions(
            {
                "iid_answer_correct": [record["answer_correct"] for record in iid_records],
                "ood_answer_correct": [record["answer_correct"] for record in ood_records],
            }
        )
        logger.write_report(report)
        logger.finalize(checkpoint_hash=checkpoint_hash)
    _publish_text_bundle(
        (
            (metrics_output, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
            (report_output, report),
        )
    )
    return complete


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="frozen evaluation YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="recorded checkpoint")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="approved root containing inputs, checkpoint, and outputs (default: artifacts)",
    )
    args = parser.parse_args(argv)
    try:
        complete = _run_evaluation(
            args.config,
            args.checkpoint,
            args.artifact_root,
            argv,
        )
    except AnalysisBlocked as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    except (FileExistsError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if not complete:
        print(
            "PARTIAL_GATE: paired OOD metrics were recorded, but the full metric contract "
            "remains incomplete",
            file=sys.stderr,
        )
        return 3
    print("COMPLETE: every requested metric was computed from registered inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
