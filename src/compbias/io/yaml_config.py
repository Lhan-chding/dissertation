"""Strict YAML configuration loading for reproducible experiment entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MAX_CONFIG_BYTES = 1_048_576
_MAX_CONFIG_DEPTH = 64
_MAX_CONFIG_NODES = 100_000


def _validate_yaml_complexity(source: str, *, label: str) -> None:
    """Reject aliases and bound parser work before constructing Python objects."""

    import yaml

    depth = 0
    node_count = 0
    collection_starts = (yaml.MappingStartEvent, yaml.SequenceStartEvent)
    collection_ends = (yaml.MappingEndEvent, yaml.SequenceEndEvent)
    try:
        events = yaml.parse(source)
        for event in events:
            if isinstance(event, yaml.AliasEvent):
                raise ValueError(f"{label} must not contain YAML aliases")
            if isinstance(event, collection_starts):
                depth += 1
                node_count += 1
                if depth > _MAX_CONFIG_DEPTH:
                    raise ValueError(f"{label} YAML nesting depth exceeds {_MAX_CONFIG_DEPTH}")
            elif isinstance(event, collection_ends):
                depth -= 1
            elif isinstance(event, yaml.ScalarEvent):
                node_count += 1
            if node_count > _MAX_CONFIG_NODES:
                raise ValueError(f"{label} YAML node count exceeds {_MAX_CONFIG_NODES}")
    except ValueError:
        raise
    except (RecursionError, yaml.YAMLError) as error:
        raise ValueError(f"invalid YAML in {label}: {error}") from error


def reject_unknown_fields(
    value: object,
    allowed: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    """Reject misspelled fields in one string-keyed configuration mapping."""

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")


def load_yaml_mapping(
    path: Path,
    *,
    label: str = "configuration",
    max_bytes: int = _MAX_CONFIG_BYTES,
) -> dict[str, Any]:
    """Load one safe YAML mapping while rejecting ambiguous duplicate keys."""

    import yaml

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
    try:
        source = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 text") from error

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: UniqueKeySafeLoader,
        node: yaml.MappingNode,
        deep: bool = False,
    ) -> dict[object, object]:
        loader.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicated = key in result
            except TypeError as error:
                raise ValueError(f"YAML keys in {label} must be hashable scalars") from error
            if duplicated:
                raise ValueError(f"duplicate YAML key in {label}: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    _validate_yaml_complexity(source, label=label)
    try:
        loaded = yaml.load(source, Loader=UniqueKeySafeLoader)
    except ValueError:
        raise
    except (RecursionError, yaml.YAMLError) as error:
        raise ValueError(f"invalid YAML in {label} {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    if any(not isinstance(key, str) for key in loaded):
        raise ValueError(f"top-level keys in {label} must be strings")
    return loaded
