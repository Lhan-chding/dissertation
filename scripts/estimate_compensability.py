#!/usr/bin/env python3
"""Estimate fixed-reasoner compensability from a preregistered rollout config."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_MAX_CONFIG_BYTES = 1024 * 1024
_REGISTERED_ROLLOUT_SEEDS = tuple(range(1000, 1032))


class AnalysisBlocked(RuntimeError):
    """A required recorded artifact is absent, so analysis must not proceed."""


_ROLLOUT_FIELDS = frozenset(
    {
        "sample_id",
        "error_id",
        "severity",
        "base_probability",
        "checkpoint",
        "checkpoint_sha256",
        "model_revision",
        "dataset_manifest_sha256",
        "dataset_content_sha256",
        "state_adapter_sha256",
        "rollout_seed",
        "reward",
        "view",
        "image",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("checkpoint must not be a symbolic link")
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise AnalysisBlocked(f"recorded checkpoint artifact is missing: {path}")
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
        digest.update(bytes.fromhex(_sha256(candidate)))
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _expect_keys(
    value: object,
    name: str,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> dict[str, Any]:
    result = _mapping(value, name)
    expected = set(required) | set(optional)
    missing = sorted(set(required) - set(result))
    unknown = sorted(set(result) - expected)
    if missing:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {', '.join(unknown)}")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path(value: object, name: str) -> Path:
    return Path(_nonempty_string(value, name)).expanduser()


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
            "model",
            "checkpoint_role",
            "input",
            "sampling",
            "statistics",
            "provenance",
            "outputs",
        ),
    )
    if config["schema_version"] != 1:
        raise ValueError("configuration schema_version must equal 1")
    _nonempty_string(config["experiment"], "experiment")
    _nonempty_string(config["execution_status"], "execution_status")
    if config["checkpoint_role"] != "pre_rl_fixed_reasoner":
        raise ValueError("checkpoint_role must be pre_rl_fixed_reasoner")
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
    observed = _sha256(path)
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


def _read_strict_jsonl(
    path: Path, *, max_bytes: int, max_line_bytes: int
) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise AnalysisBlocked(
            f"required GPU rollout artifact is missing: {path}; no estimate was produced"
        )
    size = path.stat().st_size
    if size == 0:
        raise ValueError("rollout JSONL must not be empty")
    if size > max_bytes:
        raise ValueError(f"rollout JSONL exceeds max_jsonl_bytes ({size} > {max_bytes})")
    records: list[dict[str, object]] = []
    try:
        with path.open("rb") as stream:
            for line_number, raw_line_with_ending in enumerate(stream, start=1):
                if len(raw_line_with_ending) > max_line_bytes + 2:
                    raise ValueError(
                        f"rollout JSONL line {line_number} exceeds max_jsonl_line_bytes"
                    )
                raw_line = raw_line_with_ending.rstrip(b"\r\n")
                if not raw_line.strip():
                    raise ValueError(f"rollout JSONL contains a blank line at line {line_number}")
                if len(raw_line) > max_line_bytes:
                    raise ValueError(
                        f"rollout JSONL line {line_number} exceeds max_jsonl_line_bytes"
                    )
                try:
                    record = json.loads(
                        raw_line.decode("utf-8"),
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                except (RecursionError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"invalid rollout JSONL at line {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"rollout JSONL line {line_number} must be a JSON object")
                _validate_json_value(record)
                _reject_sensitive_or_local_strings(record)
                records.append(record)
    except OSError as error:
        raise ValueError(f"cannot read rollout JSONL: {error}") from error
    if not records:
        raise ValueError("rollout JSONL must not be empty")
    return tuple(records)


def _validate_protocol(config: Mapping[str, Any]) -> dict[str, object]:
    model = _expect_keys(config["model"], "model", required=("name", "revision"))
    input_config = _expect_keys(
        config["input"],
        "input",
        required=(
            "split",
            "view",
            "image_access",
            "require_image_is_none",
            "error_catalog",
            "dataset_manifest",
            "max_jsonl_bytes",
            "max_jsonl_line_bytes",
        ),
    )
    sampling = _expect_keys(config["sampling"], "sampling", required=("rollout_seeds",))
    statistics = _expect_keys(
        config["statistics"],
        "statistics",
        required=(
            "confidence",
            "interval",
            "covariance_scope",
            "pooled_covariance_is_primary",
        ),
    )
    provenance = _expect_keys(
        config["provenance"],
        "provenance",
        required=(
            "checkpoint_path",
            "checkpoint_sha256",
            "execution_audit",
            "execution_audit_sha256",
            "phase_d_audit",
            "phase_d_audit_sha256",
            "rollout_manifest",
            "rollout_manifest_sha256",
            "verl_revision",
        ),
    )
    outputs = _expect_keys(
        config["outputs"],
        "outputs",
        required=("rollouts", "long_table", "prompt_covariances", "log_root"),
    )

    if input_config["split"] != "calibration":
        raise ValueError("input.split must be calibration")
    if input_config["view"] != "interventional":
        raise ValueError("input.view must be interventional")
    if input_config["image_access"] != "forbidden":
        raise ValueError("input.image_access must be forbidden")
    if input_config["require_image_is_none"] is not True:
        raise ValueError("input.require_image_is_none must be true")
    if input_config["error_catalog"] != "exhaustive_per_sample":
        raise ValueError("input.error_catalog must be exhaustive_per_sample")
    if statistics["interval"] != "wilson":
        raise ValueError("statistics.interval must be wilson")
    if statistics["covariance_scope"] != "per_prompt":
        raise ValueError("statistics.covariance_scope must be per_prompt")
    if statistics["pooled_covariance_is_primary"] is not False:
        raise ValueError("statistics.pooled_covariance_is_primary must be false")
    confidence = statistics["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("statistics.confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("statistics.confidence must lie strictly between zero and one")

    seeds_value = sampling["rollout_seeds"]
    if not isinstance(seeds_value, list) or not seeds_value:
        raise ValueError("sampling.rollout_seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_value):
        raise ValueError("sampling.rollout_seeds must contain only integers")
    seeds = tuple(seeds_value)
    if len(set(seeds)) != len(seeds):
        raise ValueError("sampling.rollout_seeds must be unique")
    if seeds != _REGISTERED_ROLLOUT_SEEDS:
        raise ValueError("sampling.rollout_seeds must equal the registered seeds 1000 through 1031")

    checkpoint_sha256 = _sha256_value(
        provenance["checkpoint_sha256"],
        "provenance.checkpoint_sha256",
        allow_none=True,
    )
    execution_audit_sha256 = _sha256_value(
        provenance["execution_audit_sha256"],
        "provenance.execution_audit_sha256",
        allow_none=True,
    )
    rollout_manifest_sha256 = _sha256_value(
        provenance["rollout_manifest_sha256"],
        "provenance.rollout_manifest_sha256",
        allow_none=True,
    )
    phase_d_audit_sha256 = _sha256_value(
        provenance["phase_d_audit_sha256"],
        "provenance.phase_d_audit_sha256",
        allow_none=True,
    )

    return {
        "experiment": _nonempty_string(config["experiment"], "experiment"),
        "execution_status": config["execution_status"],
        "model_name": _nonempty_string(model["name"], "model.name"),
        "model_revision": _nonempty_string(model["revision"], "model.revision"),
        "manifest": _path(input_config["dataset_manifest"], "input.dataset_manifest"),
        "max_bytes": _positive_integer(input_config["max_jsonl_bytes"], "max_jsonl_bytes"),
        "max_line_bytes": _positive_integer(
            input_config["max_jsonl_line_bytes"], "max_jsonl_line_bytes"
        ),
        "seeds": seeds,
        "confidence": confidence,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": _path(provenance["checkpoint_path"], "checkpoint_path"),
        "execution_audit": _path(provenance["execution_audit"], "execution_audit"),
        "execution_audit_sha256": execution_audit_sha256,
        "phase_d_audit": _path(provenance["phase_d_audit"], "phase_d_audit"),
        "phase_d_audit_sha256": phase_d_audit_sha256,
        "rollout_manifest": _path(provenance["rollout_manifest"], "rollout_manifest"),
        "rollout_manifest_sha256": rollout_manifest_sha256,
        "verl_revision": _nonempty_string(provenance["verl_revision"], "provenance.verl_revision"),
        "rollouts": _path(outputs["rollouts"], "outputs.rollouts"),
        "long_table": _path(outputs["long_table"], "outputs.long_table"),
        "covariances": _path(outputs["prompt_covariances"], "outputs.prompt_covariances"),
        "log_root": _path(outputs["log_root"], "outputs.log_root"),
    }


def _validate_records(
    records: tuple[dict[str, object], ...],
    *,
    seeds: tuple[int, ...],
    checkpoint_hash: str,
    model_revision: str,
    dataset_manifest_hash: str,
    dataset_content_hash: str,
    state_adapter_hash: str,
    dataset_sample_ids: frozenset[str],
    expected_error_severities: Mapping[str, Mapping[str, float]],
    checkpoint_label: str,
) -> None:
    expected_seeds = frozenset(seeds)
    observed: dict[tuple[object, object, object], set[object]] = {}
    observed_error_ids: dict[str, set[str]] = {}
    seen_rows: set[tuple[object, object, object, object]] = set()
    for index, record in enumerate(records, start=1):
        unknown = sorted(set(record) - _ROLLOUT_FIELDS)
        missing = sorted(_ROLLOUT_FIELDS - set(record))
        if missing:
            raise ValueError(f"rollout {index} is missing required fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"rollout {index} has unknown fields: {', '.join(unknown)}")
        if record.get("image") is not None:
            raise ValueError(
                f"rollout {index} violates image-hidden intervention: image is not null"
            )
        if record.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"rollout {index} checkpoint_sha256 does not match the config")
        if record.get("model_revision") != model_revision:
            raise ValueError(f"rollout {index} model_revision does not match the config")
        if record.get("dataset_manifest_sha256") != dataset_manifest_hash:
            raise ValueError(f"rollout {index} dataset manifest hash does not match")
        if record.get("dataset_content_sha256") != dataset_content_hash:
            raise ValueError(f"rollout {index} dataset content hash does not match")
        if record.get("state_adapter_sha256") != state_adapter_hash:
            raise ValueError(f"rollout {index} state adapter hash does not match")
        reward = record.get("reward")
        if isinstance(reward, bool) or reward not in {0, 1}:
            raise ValueError(f"rollout {index} reward must be binary")
        probability = record.get("base_probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError(f"rollout {index} base_probability must be finite in [0, 1]")
        for field in ("sample_id", "error_id", "checkpoint"):
            value = record.get(field)
            if isinstance(value, str) and (
                value.lstrip().startswith(("=", "+", "-", "@"))
                or value.startswith(("\t", "\r", "\n"))
            ):
                raise ValueError(
                    f"{field} contains a spreadsheet formula prefix; refusing unsafe CSV export"
                )
        if record.get("checkpoint") != checkpoint_label:
            raise ValueError("rollout checkpoint label differs from the bound manifest label")
        sample_id = record.get("sample_id")
        error_id = record.get("error_id")
        if not isinstance(sample_id, str) or sample_id not in dataset_sample_ids:
            raise ValueError(f"rollout {index} sample_id is outside the frozen dataset manifest")
        if not isinstance(error_id, str):
            raise ValueError(f"rollout {index} error_id must be a string")
        expected_severity = expected_error_severities.get(sample_id, {}).get(error_id)
        if expected_severity is None or record.get("severity") != expected_severity:
            raise ValueError(
                f"rollout {index} error ID/severity differs from the frozen dataset catalog"
            )
        observed_error_ids.setdefault(sample_id, set()).add(error_id)
        row_key = (
            record.get("sample_id"),
            record.get("error_id"),
            record.get("checkpoint"),
            record.get("rollout_seed"),
        )
        if row_key in seen_rows:
            raise ValueError(f"duplicate rollout sample/error/checkpoint/seed row: {row_key!r}")
        seen_rows.add(row_key)
        key = (record.get("sample_id"), record.get("error_id"), record.get("checkpoint"))
        observed.setdefault(key, set()).add(record.get("rollout_seed"))
    for key, observed_seeds in observed.items():
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"rollout seeds for sample/error/checkpoint group {key!r} do not match "
                "sampling.rollout_seeds"
            )
    if set(observed_error_ids) != set(expected_error_severities):
        raise ValueError("rollout sample set does not match the rollout manifest error catalog")
    for sample_id, expected in expected_error_severities.items():
        if observed_error_ids[sample_id] != set(expected):
            raise ValueError(
                f"rollout error catalog for sample {sample_id!r} is incomplete or unexpected"
            )


def _reject_formula_cells(table: object) -> None:
    for column in ("sample_id", "error_id", "checkpoint"):
        values = getattr(table, column)
        for value in values:
            text = str(value)
            if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(
                ("\t", "\r", "\n")
            ):
                raise ValueError(
                    f"{column} contains a spreadsheet formula prefix; refusing unsafe CSV export"
                )


def _ensure_new_paths(*paths: Path) -> None:
    normalized = tuple(path.expanduser().resolve() for path in paths)
    if len(set(normalized)) != len(normalized):
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


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
            temporary_path = Path(stream.name)
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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


def _run_analysis(config_path: Path, artifact_root: Path, argv: Sequence[str] | None) -> None:
    from compbias.eval.compensability import (
        build_compensability_long_table,
        per_prompt_covariances,
    )
    from compbias.eval.dataset_contract import validate_frozen_cva_v2_dataset
    from compbias.io.logging import (
        RunLogger,
        capture_environment,
        publishable_config_snapshot,
        publishable_path,
    )

    config = _load_config(config_path)
    # Reject publish-unsafe free text before reconstructing the frozen dataset.
    # This keeps an invalid configuration fail-fast even under coverage or on
    # machines where deterministic CVA replay is comparatively slow.
    _reject_sensitive_or_local_strings(config, path="$.configuration")
    protocol = _validate_protocol(config)
    for key in (
        "manifest",
        "checkpoint_path",
        "execution_audit",
        "phase_d_audit",
        "rollout_manifest",
        "rollouts",
        "long_table",
        "covariances",
        "log_root",
    ):
        path = protocol[key]
        assert isinstance(path, Path)
        _require_safe_path(
            path,
            root=artifact_root,
            name=key,
        )
    checkpoint_hash = protocol["checkpoint_sha256"]
    if not isinstance(checkpoint_hash, str):
        raise AnalysisBlocked(
            "GPU rollout provenance.checkpoint_sha256 is not recorded; no estimate was produced"
        )
    if protocol["execution_status"] != "recorded_gpu_artifacts":
        raise AnalysisBlocked(
            "execution_status is not recorded_gpu_artifacts; "
            "no completed GPU analysis may be claimed"
        )
    checkpoint_path = protocol["checkpoint_path"]
    assert isinstance(checkpoint_path, Path)
    observed_checkpoint_hash = _artifact_sha256(checkpoint_path)
    if observed_checkpoint_hash != checkpoint_hash:
        raise ValueError("checkpoint SHA-256 mismatch between the recorded path and configuration")
    rollout_path = protocol["rollouts"]
    assert isinstance(rollout_path, Path)
    records = _read_strict_jsonl(
        rollout_path,
        max_bytes=int(protocol["max_bytes"]),
        max_line_bytes=int(protocol["max_line_bytes"]),
    )
    seeds = protocol["seeds"]
    assert isinstance(seeds, tuple)

    manifest_path = protocol["manifest"]
    assert isinstance(manifest_path, Path)
    if not manifest_path.is_file():
        raise AnalysisBlocked(f"required dataset manifest is missing: {manifest_path}")
    dataset = validate_frozen_cva_v2_dataset(manifest_path, artifact_root=artifact_root)
    manifest_hash = dataset.manifest_file_sha256
    dataset_content_hash = dataset.content_sha256
    calibration_sample_ids = dataset.sample_ids_for_partition("calibration")
    expected_error_severities = {
        sample_id: {
            error.error_id: error.severity for error in dataset.records[sample_id].error_catalog
        }
        for sample_id in calibration_sample_ids
    }

    config_hash = _sha256(config_path)
    rollout_hash = _sha256(rollout_path)
    rollout_manifest_path = protocol["rollout_manifest"]
    execution_audit_path = protocol["execution_audit"]
    phase_d_audit_path = protocol["phase_d_audit"]
    assert isinstance(rollout_manifest_path, Path)
    assert isinstance(execution_audit_path, Path)
    assert isinstance(phase_d_audit_path, Path)
    rollout_manifest_hash = protocol["rollout_manifest_sha256"]
    execution_audit_hash = protocol["execution_audit_sha256"]
    phase_d_audit_hash = protocol["phase_d_audit_sha256"]
    if (
        not isinstance(rollout_manifest_hash, str)
        or not isinstance(execution_audit_hash, str)
        or not isinstance(phase_d_audit_hash, str)
    ):
        raise AnalysisBlocked(
            "rollout/execution/Phase-D audit hashes are absent; no estimate may be claimed"
        )
    rollout_manifest = _read_hashed_json(
        rollout_manifest_path, rollout_manifest_hash, "rollout manifest"
    )
    execution_audit = _read_hashed_json(
        execution_audit_path, execution_audit_hash, "execution audit"
    )
    phase_d_audit = _read_hashed_json(phase_d_audit_path, phase_d_audit_hash, "Phase-D audit")
    from compbias.eval.post_gpu_evidence import (
        PostGPUAuthenticationPending,
        validate_post_gpu_execution_audit,
        validate_ready_phase_d_audit,
    )
    from compbias.io.manifests import manifest_sha256

    validate_ready_phase_d_audit(
        phase_d_audit,
        dataset_manifest_sha256=manifest_hash,
        dataset_manifest_self_sha256=dataset.manifest_self_sha256,
        dataset_content_sha256=dataset_content_hash,
        dataset_image_set_sha256=manifest_sha256(dataset.image_sha256),
        sample_ids=tuple(dataset.records),
    )
    if (
        rollout_manifest.get("schema_version") != 1
        or rollout_manifest.get("artifact_type") != "fixed_reasoner_compensability_rollouts"
    ):
        raise ValueError("rollout manifest schema or artifact_type is invalid")
    if rollout_manifest.get("rollouts_sha256") != rollout_hash:
        raise ValueError("rollout manifest is bound to a different rollout JSONL")
    if rollout_manifest.get("record_count") != len(records):
        raise ValueError("rollout manifest record_count does not match the JSONL")
    if rollout_manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("rollout manifest is bound to a different checkpoint")
    if rollout_manifest.get("model_revision") != protocol["model_revision"]:
        raise ValueError("rollout manifest is bound to a different model revision")
    if rollout_manifest.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("rollout manifest is bound to a different dataset manifest")
    if rollout_manifest.get("dataset_content_sha256") != dataset_content_hash:
        raise ValueError("rollout manifest is bound to different dataset content")
    if rollout_manifest.get("rollout_seeds") != list(seeds):
        raise ValueError("rollout manifest seeds differ from the config")
    if (
        rollout_manifest.get("dataset_partition") != "calibration"
        or rollout_manifest.get("prediction_scope") != "exact_partition"
    ):
        raise ValueError(
            "rollout manifest must bind the exact calibration partition of frozen cva_v2"
        )
    sample_ids = rollout_manifest.get("sample_ids")
    if sample_ids != list(calibration_sample_ids):
        raise ValueError("rollout manifest must cover the exact calibration partition")
    catalog = rollout_manifest.get("error_ids_by_sample")
    expected_catalog = {
        sample_id: sorted(expected_error_severities[sample_id])
        for sample_id in calibration_sample_ids
    }
    if catalog != expected_catalog:
        raise ValueError(
            "rollout manifest error catalog must exactly match each frozen calibration sample"
        )
    checkpoint_label = rollout_manifest.get("checkpoint_label")
    if (
        not isinstance(checkpoint_label, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", checkpoint_label) is None
    ):
        raise ValueError("rollout manifest checkpoint label must be a safe component")
    state_adapter_hash = _sha256_value(
        rollout_manifest.get("state_adapter_sha256"),
        "rollout manifest state_adapter_sha256",
    )
    producer_config_hash = _sha256_value(
        rollout_manifest.get("producer_config_sha256"),
        "rollout manifest producer_config_sha256",
    )
    _validate_records(
        records,
        seeds=seeds,
        checkpoint_hash=checkpoint_hash,
        model_revision=str(protocol["model_revision"]),
        dataset_manifest_hash=manifest_hash,
        dataset_content_hash=str(dataset_content_hash),
        state_adapter_hash=str(state_adapter_hash),
        dataset_sample_ids=frozenset(calibration_sample_ids),
        expected_error_severities=expected_error_severities,
        checkpoint_label=checkpoint_label,
    )
    try:
        validate_post_gpu_execution_audit(
            execution_audit,
            artifact_type="fixed_reasoner_compensability_execution_audit",
            stage="fixed_reasoner_compensability",
            checkpoint_sha256=checkpoint_hash,
            dataset_manifest_sha256=manifest_hash,
            dataset_manifest_self_sha256=dataset.manifest_self_sha256,
            dataset_content_sha256=dataset_content_hash,
            phase_d_audit_sha256=phase_d_audit_hash,
            prediction_or_rollout_manifest_sha256=rollout_manifest_hash,
            producer_config_sha256=str(producer_config_hash),
            producer_records_path=rollout_path,
            producer_records_sha256=rollout_hash,
            producer_record_count=len(records),
            seeds=seeds,
            model_revision=str(protocol["model_revision"]),
            verl_revision=str(protocol["verl_revision"]),
            sha256_file=_sha256,
        )
    except PostGPUAuthenticationPending as error:
        raise AnalysisBlocked(str(error)) from error
    state_audit = execution_audit.get("state_injection_audit")
    if not isinstance(state_audit, Mapping):
        raise ValueError("execution audit state_injection_audit is missing")
    if (
        state_audit.get("passed") is not True
        or state_audit.get("image_hidden") is not True
        or state_audit.get("isolation_mode") != "separate_text_only_worker"
        or state_audit.get("adapter_sha256") != state_adapter_hash
        or state_audit.get("reviewed_adapter_sha256") != state_adapter_hash
    ):
        raise AnalysisBlocked("isolated image-hidden state-injection audit has not passed")

    table = build_compensability_long_table(records, confidence=float(protocol["confidence"]))
    raise AnalysisBlocked("unreachable until authenticated post-GPU gate extension exists")
    covariances = per_prompt_covariances(table)  # pragma: no cover - future gate extension
    _reject_formula_cells(table)

    long_table_path = protocol["long_table"]
    covariance_path = protocol["covariances"]
    log_root = protocol["log_root"]
    assert isinstance(long_table_path, Path)
    assert isinstance(covariance_path, Path)
    assert isinstance(log_root, Path)
    _ensure_new_paths(long_table_path, covariance_path)
    run_id = f"source-{rollout_hash[:12]}-config-{config_hash[:12]}"
    run_dir = log_root / str(protocol["experiment"]) / run_id
    _require_disjoint_paths(
        inputs={
            "config": config_path,
            "checkpoint": checkpoint_path,
            "dataset_manifest": manifest_path,
            "execution_audit": execution_audit_path,
            "phase_d_audit": phase_d_audit_path,
            "rollout_manifest": rollout_manifest_path,
            "rollouts": rollout_path,
        },
        outputs={
            "long_table": long_table_path,
            "prompt_covariances": covariance_path,
            "run_log": run_dir,
        },
    )
    if os.path.lexists(run_dir):
        raise FileExistsError(f"run log already exists; refusing to overwrite: {run_dir}")

    command = _command(argv)
    environment = capture_environment(
        worktree=Path.cwd(),
        dataset_manifest_hash=manifest_hash,
        seed=int(seeds[0]),
        model_revision=str(protocol["model_revision"]),
        verl_revision=str(protocol["verl_revision"]),
        command=command,
    )
    publishable_config = publishable_config_snapshot(
        config,
        worktree=Path.cwd(),
        path_fields=(
            ("input", "dataset_manifest"),
            ("provenance", "checkpoint_path"),
            ("provenance", "execution_audit"),
            ("provenance", "phase_d_audit"),
            ("provenance", "rollout_manifest"),
            ("outputs", "rollouts"),
            ("outputs", "long_table"),
            ("outputs", "prompt_covariances"),
            ("outputs", "log_root"),
        ),
    )
    _reject_sensitive_or_local_strings(publishable_config, path="$.logged_config")
    logged_config = {
        **publishable_config,
        "analysis_provenance": {
            "config_sha256": config_hash,
            "rollouts_sha256": rollout_hash,
            "dataset_manifest_sha256": manifest_hash,
        },
    }
    summary = {
        "status": "COMPLETE",
        "config_sha256": config_hash,
        "rollouts_sha256": rollout_hash,
        "dataset_manifest_sha256": manifest_hash,
        "dataset_manifest_self_sha256": dataset.manifest_self_sha256,
        "dataset_content_sha256": dataset_content_hash,
        "dataset_file_sha256": dataset.dataset_file_sha256,
        "dataset_partition": "calibration",
        "dataset_sample_count": len(calibration_sample_ids),
        "execution_audit_sha256": execution_audit_hash,
        "phase_d_audit_sha256": phase_d_audit_hash,
        "rollout_manifest_sha256": rollout_manifest_hash,
        "rollout_rows": len(table),
        "prompt_checkpoint_rows": len(covariances),
        "checkpoint_sha256": checkpoint_hash,
    }
    report = (
        "# Fixed-reasoner compensability estimate\n\n"
        "Status: `COMPLETE`\n\n"
        f"- Config SHA-256: `{config_hash}`\n"
        f"- Rollout SHA-256: `{rollout_hash}`\n"
        f"- Dataset manifest SHA-256: `{manifest_hash}`\n"
        f"- Checkpoint SHA-256: `{checkpoint_hash}`\n"
        f"- Rollout rows: {len(table)}\n"
        f"- Prompt/checkpoint covariance rows: {len(covariances)}\n"
    )

    with RunLogger(
        root=log_root,
        experiment=str(protocol["experiment"]),
        run_id=run_id,
        config=logged_config,
        environment=environment,
    ) as logger:
        logger.log_metrics(summary)
        for record in records:
            logger.log_rollout(record)
        logger.save_predictions(
            {
                "severity": table["severity"].to_numpy(dtype=float, copy=True),
                "reward": table["reward"].to_numpy(dtype=float, copy=True),
                "c_hat": table["c_hat"].to_numpy(dtype=float, copy=True),
            }
        )
        logger.write_report(report)
        logger.finalize(checkpoint_hash=checkpoint_hash)
    _publish_text_bundle(
        (
            (long_table_path, table.to_csv(index=False, lineterminator="\n")),
            (covariance_path, covariances.to_csv(index=False, lineterminator="\n")),
        )
    )

    print(
        "COMPLETE: wrote "
        f"{publishable_path(long_table_path, worktree=Path.cwd())} and "
        f"{publishable_path(covariance_path, worktree=Path.cwd())}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="preregistered YAML config")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="approved root containing all inputs and outputs (default: artifacts)",
    )
    args = parser.parse_args(argv)
    try:
        _run_analysis(args.config, args.artifact_root, argv)
    except AnalysisBlocked as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    except (FileExistsError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
