#!/usr/bin/env python3
"""Generate, render, and manifest a deterministic CVA-World dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

_MAX_GENERATED_IMAGES = 10_000
_MAX_RENDER_PIXELS = 256_000_000


def _read_regular_bytes(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    """Read one bounded regular file without following links or blocking on FIFOs."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds {maximum_bytes} byte limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            return stream.read(maximum_bytes + 1)
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular file: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_regular_text(path: Path, *, label: str, maximum_bytes: int) -> str:
    raw = _read_regular_bytes(path, label=label, maximum_bytes=maximum_bytes)
    if len(raw) > maximum_bytes:
        raise ValueError(f"{label} exceeds {maximum_bytes} byte limit")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 text") from error


def _strict_json_mapping(path: Path) -> dict[str, object]:
    text = _read_regular_text(
        path,
        label="prior manifest",
        maximum_bytes=16 * 1024 * 1024,
    )

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number is not permitted: {token}")
        return value

    def reject_constant(token: str) -> object:
        raise ValueError(f"non-finite JSON constant is not permitted: {token}")

    try:
        loaded = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (OSError, RecursionError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid prior manifest: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError("prior manifest must contain a JSON object")
    stack = [(loaded, 0)]
    while stack:
        value, depth = stack.pop()
        if depth > 64:
            raise ValueError("prior manifest nesting exceeds 64 levels")
        if isinstance(value, Mapping):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
    return loaded


def _yaml_mapping(path: Path) -> Mapping[str, object]:
    import yaml

    text = _read_regular_text(
        path,
        label="configuration",
        maximum_bytes=1024 * 1024,
    )

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(
        loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
    ) -> dict[object, object]:
        loader.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ValueError("YAML mapping keys must be hashable")
            if key in result:
                raise ValueError(f"duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping,
    )
    try:
        loaded = (
            yaml.load(
                text,
                Loader=UniqueKeySafeLoader,
            )
            or {}
        )
    except (RecursionError, yaml.YAMLError) as error:
        raise ValueError(f"invalid YAML configuration: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    stack: list[tuple[object, int]] = [(loaded, 0)]
    seen: set[int] = set()
    node_count = 0
    while stack:
        value, depth = stack.pop()
        node_count += 1
        if depth > 64 or node_count > 100_000:
            raise ValueError("configuration YAML exceeds depth/node safety limits")
        if isinstance(value, (Mapping, list)):
            identity = id(value)
            if identity in seen:
                raise ValueError("configuration YAML aliases/cycles are forbidden")
            seen.add(identity)
            children = value.values() if isinstance(value, Mapping) else value
            stack.extend((child, depth + 1) for child in children)
    return loaded


def _section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _closed_keys(mapping: Mapping[str, object], *, allowed: set[str], name: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(sorted(unknown))}")


def _safe_component(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"{label} must be a safe path component")
    return value


def _strict_integer(
    mapping: Mapping[str, object],
    field: str,
    default: int,
    *,
    label: str,
) -> int:
    value = mapping.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _strict_string(
    mapping: Mapping[str, object],
    field: str,
    default: str,
    *,
    label: str,
) -> str:
    value = mapping.get(field, default)
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _strict_string_sequence(
    mapping: Mapping[str, object],
    field: str,
    default: tuple[str, ...] | None,
    *,
    label: str,
) -> tuple[str, ...]:
    value = mapping.get(field, default)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a sequence containing only strings")
    return tuple(value)


def _strict_path(
    mapping: Mapping[str, object],
    field: str,
    default: str | Path,
    *,
    label: str,
) -> Path:
    value = mapping.get(field, default)
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be a path string")
    return Path(value)


def _has_symlink_between(path: Path, root: Path) -> bool:
    current = path.absolute()
    boundary = root.absolute()
    while current != boundary:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return True
        current = parent
    return boundary.is_symlink()


def _within_root(path: Path, *, root: Path, label: str) -> Path:
    lexical_root = root.expanduser().absolute()
    lexical_candidate = (
        path.expanduser().absolute()
        if path.is_absolute()
        else (Path.cwd() / path.expanduser()).absolute()
    )
    approved = root.expanduser().resolve(strict=False)
    candidate = path.expanduser().resolve(strict=False)
    try:
        candidate.relative_to(approved)
    except ValueError as error:
        raise ValueError(f"{label} path must remain inside its approved root") from error
    try:
        lexical_candidate.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"{label} path must remain inside its approved root") from error
    if _has_symlink_between(lexical_candidate, lexical_root):
        raise ValueError(f"{label} path must not traverse symlinks")
    return candidate


def _relative_publishable(path: Path, *, repository_root: Path, publication_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.relative_to(publication_root).as_posix()


def _targets_are_disjoint(files: Sequence[Path], directories: Sequence[Path]) -> bool:
    if len(set(files)) != len(files) or len(set(directories)) != len(directories):
        return False
    all_targets = (*files, *directories)
    for index, first in enumerate(all_targets):
        for second in all_targets[index + 1 :]:
            if first == second:
                return False
            if first in second.parents or second in first.parents:
                return False
    return True


def _canonical_manifest_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _published_path_matches(
    value: object,
    *,
    expected: Path,
    repository_root: Path,
    publication_root: Path,
) -> bool:
    if not isinstance(value, str) or not value:
        return False
    raw = Path(value)
    candidates = (
        raw.resolve(strict=False),
        (repository_root / raw).resolve(strict=False),
        (publication_root / raw).resolve(strict=False),
    )
    return expected.resolve(strict=False) in candidates


def _validate_prior_generation(
    *,
    manifest_path: Path,
    dataset_path: Path,
    images_dir: Path,
    contact_sheet_dir: Path,
    expected_sample_ids: tuple[str, ...],
    expected_sheet_names: set[str],
    repository_root: Path,
    publication_root: Path,
) -> None:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("--overwrite requires a real prior generated manifest")
    try:
        loaded = _strict_json_mapping(manifest_path)
    except (OSError, ValueError) as error:
        raise ValueError(
            "--overwrite requires a readable prior generated manifest: " + str(error)
        ) from error
    unsigned = {key: value for key, value in loaded.items() if key != "manifest_sha256"}
    if loaded.get("manifest_sha256") != _canonical_manifest_digest(unsigned):
        raise ValueError("--overwrite requires a valid prior manifest self-hash")
    if not _published_path_matches(
        loaded.get("jsonl_path"),
        expected=dataset_path,
        repository_root=repository_root,
        publication_root=publication_root,
    ) or not _published_path_matches(
        loaded.get("images_dir"),
        expected=images_dir,
        repository_root=repository_root,
        publication_root=publication_root,
    ):
        raise ValueError("--overwrite prior manifest paths do not bind these targets")
    if loaded.get("sample_ids") != list(expected_sample_ids):
        raise ValueError("--overwrite prior manifest sample set does not match")
    if dataset_path.is_symlink() or not dataset_path.is_file():
        raise ValueError("--overwrite prior dataset is missing or a symlink")
    if loaded.get("dataset_file_sha256") != hashlib.sha256(dataset_path.read_bytes()).hexdigest():
        raise ValueError("--overwrite prior dataset hash does not match")
    image_hashes = loaded.get("image_sha256")
    if not isinstance(image_hashes, dict) or set(image_hashes) != set(expected_sample_ids):
        raise ValueError("--overwrite prior image hash set does not match")
    for sample_id in expected_sample_ids:
        image = images_dir / f"{sample_id}.png"
        if image.is_symlink() or not image.is_file():
            raise ValueError("--overwrite prior image set is incomplete or unsafe")
        if image_hashes.get(sample_id) != hashlib.sha256(image.read_bytes()).hexdigest():
            raise ValueError("--overwrite prior image hash does not match")
    sheet_values = loaded.get("contact_sheets")
    sheet_hashes = loaded.get("contact_sheet_sha256")
    if (
        not isinstance(sheet_values, list)
        or {Path(value).name for value in sheet_values if isinstance(value, str)}
        != expected_sheet_names
    ):
        raise ValueError("--overwrite prior contact-sheet set does not match")
    if not isinstance(sheet_hashes, dict) or set(sheet_hashes) != expected_sheet_names:
        raise ValueError("--overwrite prior contact-sheet hash set does not match")
    for name in expected_sheet_names:
        sheet = contact_sheet_dir / name
        if sheet.is_symlink() or not sheet.is_file():
            raise ValueError("--overwrite prior contact sheets are incomplete or unsafe")
        if sheet_hashes.get(name) != hashlib.sha256(sheet.read_bytes()).hexdigest():
            raise ValueError("--overwrite prior contact-sheet hash does not match")


def _ensure_file_target(path: Path, *, overwrite: bool, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} target must not be a symlink")
    if path.exists() and path.is_dir():
        raise ValueError(f"{label} target must name a file")
    if path.exists() and not overwrite:
        raise ValueError(f"{label} target already exists; pass --overwrite")


def _ensure_exact_generated_tree(
    directory: Path,
    *,
    expected_names: set[str],
    overwrite: bool,
    label: str,
) -> None:
    if directory.is_symlink():
        raise ValueError(f"{label} directory must not be a symlink")
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"{label} destination must be a directory")
    if not directory.exists():
        return
    entries = tuple(directory.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError(f"{label} targets must not contain symlinks")
    existing_names = {entry.name for entry in entries}
    if existing_names and not overwrite:
        raise ValueError(f"{label} targets already exist; pass --overwrite")
    if overwrite and existing_names and existing_names != expected_names:
        raise ValueError(f"{label} overwrite requires an empty or complete prior generated set")
    if existing_names and any(not entry.is_file() for entry in entries):
        raise ValueError(f"{label} directory contains non-file entries")


def _remove_promoted(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _promote_transaction(promotions: Sequence[tuple[Path, Path]]) -> None:
    completed: list[tuple[Path, Path | None]] = []
    try:
        for staged, destination in promotions:
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup: Path | None = None
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.cva-backup-{secrets.token_hex(8)}"
                )
                os.replace(destination, backup)
            try:
                os.replace(staged, destination)
            except BaseException:
                if backup is not None:
                    os.replace(backup, destination)
                raise
            completed.append((destination, backup))
    except BaseException:
        for destination, backup in reversed(completed):
            _remove_promoted(destination)
            if backup is not None:
                os.replace(backup, destination)
        raise
    for _destination, backup in completed:
        if backup is not None:
            _remove_promoted(backup)


def _write_contact_sheets(
    rendered: Sequence[tuple[str, object]],
    *,
    directory: Path,
    per_sheet: int,
    start_index: int = 1,
) -> tuple[Path, ...]:
    from compbias.envs.cva_world.renderer import build_contact_sheet

    if per_sheet < 1:
        raise ValueError("samples_per_contact_sheet must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sheet_index, start in enumerate(range(0, len(rendered), per_sheet), start=start_index):
        batch = rendered[start : start + per_sheet]
        sheet = build_contact_sheet(batch)
        destination = directory / f"cva_contact_sheet_{sheet_index:02d}.png"
        try:
            sheet.save(destination, format="PNG", optimize=True)
        finally:
            sheet.close()
        paths.append(destination)
    return tuple(paths)


def main(argv=None) -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="strict YAML generation contract",
    )
    parser.add_argument("--output", type=Path, help="destination JSONL file")
    parser.add_argument("--manifest", type=Path, help="destination manifest JSON")
    parser.add_argument("--images-dir", type=Path, help="rendered image directory")
    parser.add_argument("--contact-sheet-dir", type=Path, help="visual-audit sheets")
    parser.add_argument("--output-root", type=Path, default=repository_root / "artifacts")
    parser.add_argument("--figure-root", type=Path, default=repository_root / "artifacts/figures")
    parser.add_argument("--log-root", type=Path, default=repository_root / "artifacts/logs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples-per-family-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        raw: Mapping[str, object] = _yaml_mapping(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        _closed_keys(raw, allowed={"dataset", "rendering", "logging"}, name="configuration")
        dataset = _section(raw, "dataset")
        rendering = _section(raw, "rendering")
        logging = _section(raw, "logging")
        _closed_keys(
            dataset,
            allowed={
                "name",
                "schema_version",
                "output",
                "manifest",
                "seed",
                "samples_per_family_per_split",
                "realizations_per_semantic",
                "fully_cross_iid_visual_styles",
                "visual_styles",
                "train_error_mechanism",
                "ood_error_mechanism",
                "preregistered_ood_factors",
            },
            name="dataset",
        )
        _closed_keys(
            rendering,
            allowed={
                "width",
                "height",
                "images_dir",
                "contact_sheet_dir",
                "samples_per_contact_sheet",
            },
            name="rendering",
        )
        _closed_keys(logging, allowed={"root", "experiment"}, name="logging")
        output = _within_root(
            args.output
            or _strict_path(
                dataset,
                "output",
                "artifacts/datasets/cva_v2/dataset.jsonl",
                label="dataset.output",
            ),
            root=args.output_root,
            label="output",
        )
        manifest_path = _within_root(
            args.manifest
            or _strict_path(
                dataset,
                "manifest",
                "artifacts/manifests/cva_v2.json",
                label="dataset.manifest",
            ),
            root=args.output_root,
            label="output manifest",
        )
        images_dir = _within_root(
            args.images_dir
            or _strict_path(
                rendering,
                "images_dir",
                output.parent / "images",
                label="rendering.images_dir",
            ),
            root=args.output_root,
            label="output images",
        )
        contact_sheet_dir = _within_root(
            args.contact_sheet_dir
            or _strict_path(
                rendering,
                "contact_sheet_dir",
                "artifacts/figures",
                label="rendering.contact_sheet_dir",
            ),
            root=args.figure_root,
            label="figures",
        )
        configured_log_root = _strict_path(
            logging,
            "root",
            "artifacts/logs",
            label="logging.root",
        )
        log_root = _within_root(
            configured_log_root,
            root=args.log_root,
            label="logs",
        )
        seed = (
            args.seed
            if args.seed is not None
            else _strict_integer(dataset, "seed", 0, label="seed")
        )
        sample_count = (
            args.samples_per_family_per_split
            if args.samples_per_family_per_split is not None
            else _strict_integer(
                dataset,
                "samples_per_family_per_split",
                10,
                label="samples_per_family_per_split",
            )
        )
        realizations_per_semantic = _strict_integer(
            dataset,
            "realizations_per_semantic",
            2,
            label="realizations_per_semantic",
        )
        fully_cross_iid_visual_styles = dataset.get("fully_cross_iid_visual_styles", False)
        if not isinstance(fully_cross_iid_visual_styles, bool):
            raise ValueError("dataset.fully_cross_iid_visual_styles must be boolean")
        styles = _strict_string_sequence(
            dataset,
            "visual_styles",
            ("font_a", "font_b", "rotated"),
            label="dataset.visual_styles",
        )
        factors = _strict_string_sequence(
            dataset,
            "preregistered_ood_factors",
            None,
            label="dataset.preregistered_ood_factors",
        )
        train_error_mechanism = _strict_string(
            dataset,
            "train_error_mechanism",
            "",
            label="dataset.train_error_mechanism",
        )
        ood_error_mechanism = _strict_string(
            dataset,
            "ood_error_mechanism",
            "",
            label="dataset.ood_error_mechanism",
        )
        schema_version = _strict_string(
            dataset,
            "schema_version",
            "2.0",
            label="dataset.schema_version",
        )
        width = _strict_integer(rendering, "width", 256, label="rendering.width")
        height = _strict_integer(rendering, "height", 192, label="rendering.height")
        per_sheet = _strict_integer(
            rendering,
            "samples_per_contact_sheet",
            25,
            label="rendering.samples_per_contact_sheet",
        )
        if not 1 <= sample_count <= 1_000:
            raise ValueError("samples_per_family_per_split must be between 1 and 1000")
        if not 2 <= realizations_per_semantic <= 16:
            raise ValueError("realizations_per_semantic must be between 2 and 16")
        if not 32 <= width <= 4096 or not 32 <= height <= 4096:
            raise ValueError("render dimensions must each be between 32 and 4096 pixels")
        if not 1 <= per_sheet <= 100:
            raise ValueError("samples_per_contact_sheet must be between 1 and 100")
        dataset_name = _safe_component(dataset.get("name", "cva_v2"), label="dataset.name")
        log_experiment = _safe_component(
            _strict_string(
                logging,
                "experiment",
                "cva_v2_generation",
                label="logging.experiment",
            ),
            label="logging.experiment",
        )
        publication_root = Path(
            os.path.commonpath(
                (
                    args.output_root.resolve(strict=False),
                    args.figure_root.resolve(strict=False),
                    args.log_root.resolve(strict=False),
                )
            )
        )
        if publication_root == Path(publication_root.anchor):
            raise ValueError("approved roots must share a scoped non-filesystem-root parent")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    from compbias.envs.cva_world.canonical_solver import solve_sample
    from compbias.envs.cva_world.corruptions import apply_error, reverse_error
    from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
    from compbias.envs.cva_world.renderer import (
        RenderConfig,
        render_sample,
        sample_render_coordinates,
    )
    from compbias.envs.cva_world.schema import SemanticSplit, TaskFamily
    from compbias.io.jsonl import write_jsonl
    from compbias.io.logging import RunLogger, capture_environment
    from compbias.io.manifests import (
        build_dataset_manifest,
        canonical_json,
        manifest_sha256,
    )

    try:
        command_arguments = sys.argv[1:] if argv is None else argv
        base_environment = capture_environment(
            worktree=Path(__file__).resolve().parents[1],
            dataset_manifest_hash=None,
            seed=seed,
            model_revision=None,
            verl_revision=None,
            command=(sys.executable, str(Path(__file__).resolve()), *command_arguments),
        )
        config = GeneratorConfig(
            seed=seed,
            samples_per_family_per_split=sample_count,
            splits=tuple(SemanticSplit),
            task_families=tuple(TaskFamily),
            visual_styles=styles,
            train_error_mechanism=train_error_mechanism,
            ood_error_mechanism=ood_error_mechanism,
            preregistered_ood_factors=factors,
            realizations_per_semantic=realizations_per_semantic,
            fully_cross_iid_visual_styles=fully_cross_iid_visual_styles,
        )
        expected_image_count = config.expected_sample_count()
        if expected_image_count > _MAX_GENERATED_IMAGES:
            raise ValueError(
                f"generated image count {expected_image_count} exceeds "
                f"{_MAX_GENERATED_IMAGES} limit"
            )
        expected_render_pixels = expected_image_count * width * height
        if expected_render_pixels > _MAX_RENDER_PIXELS:
            raise ValueError(
                "render pixel budget exceeds "
                f"{_MAX_RENDER_PIXELS}; reduce sample count or image dimensions"
            )
        samples = generate_dataset(config)
        if len(samples) != expected_image_count:
            raise RuntimeError("generated sample count does not match the validated plan")
        if not _targets_are_disjoint(
            (output, manifest_path),
            (images_dir, contact_sheet_dir, log_root),
        ):
            raise ValueError(
                "generated file and directory targets must be distinct and non-overlapping"
            )
        image_names = {f"{sample.sample_id}.png" for sample in samples}
        sheet_count = (len(samples) + per_sheet - 1) // per_sheet
        sheet_names = {f"cva_contact_sheet_{index:02d}.png" for index in range(1, sheet_count + 1)}
        generated_targets_exist = (
            output.exists()
            or manifest_path.exists()
            or images_dir.exists()
            or any((contact_sheet_dir / name).exists() for name in sheet_names)
        )
        if args.overwrite and generated_targets_exist:
            _validate_prior_generation(
                manifest_path=manifest_path,
                dataset_path=output,
                images_dir=images_dir,
                contact_sheet_dir=contact_sheet_dir,
                expected_sample_ids=tuple(sorted(sample.sample_id for sample in samples)),
                expected_sheet_names=sheet_names,
                repository_root=repository_root,
                publication_root=publication_root,
            )
        _ensure_file_target(output, overwrite=args.overwrite, label="dataset")
        _ensure_file_target(manifest_path, overwrite=args.overwrite, label="manifest")
        _ensure_exact_generated_tree(
            images_dir,
            expected_names=image_names,
            overwrite=args.overwrite,
            label="image",
        )
        _ensure_exact_generated_tree(
            contact_sheet_dir,
            expected_names=sheet_names,
            overwrite=args.overwrite,
            label="contact sheet",
        )

        for root in {
            args.output_root.resolve(strict=False),
            args.figure_root.resolve(strict=False),
            args.log_root.resolve(strict=False),
        }:
            root.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(
                prefix=".cva-output-stage-", dir=args.output_root
            ) as raw_output_stage,
            tempfile.TemporaryDirectory(
                prefix=".cva-figure-stage-", dir=args.figure_root
            ) as raw_figure_stage,
            tempfile.TemporaryDirectory(
                prefix=".cva-log-stage-", dir=args.log_root
            ) as raw_log_stage,
        ):
            output_stage = Path(raw_output_stage)
            figure_stage = Path(raw_figure_stage)
            log_stage = Path(raw_log_stage)
            staged_dataset = output_stage / "dataset.jsonl"
            staged_manifest = output_stage / "manifest.json"
            staged_images = output_stage / "images"
            staged_images.mkdir()
            rendered_batch: list[tuple[str, object]] = []
            staged_sheets: list[Path] = []
            next_sheet_index = 1
            image_sha256: dict[str, str] = {}
            solver_checks = 0
            roundtrip_checks = 0
            for sample in samples:
                solve_sample(sample)
                solver_checks += 1
                for error in sample.error_catalog:
                    if reverse_error(apply_error(sample.scene, error), error) != sample.scene:
                        raise RuntimeError(
                            f"round-trip failed for {sample.sample_id}/{error.error_id}"
                        )
                    roundtrip_checks += 1
                render_seed, realization_index = sample_render_coordinates(
                    sample.sample_id,
                    base_seed=seed,
                )
                image = render_sample(
                    sample,
                    RenderConfig(
                        width=width,
                        height=height,
                        seed=render_seed,
                        realization_index=realization_index,
                    ),
                )
                staged_image = staged_images / f"{sample.sample_id}.png"
                image.save(staged_image, format="PNG", optimize=True)
                image_sha256[sample.sample_id] = hashlib.sha256(
                    staged_image.read_bytes()
                ).hexdigest()
                rendered_batch.append((sample.sample_id, image))
                if len(rendered_batch) == per_sheet:
                    written = _write_contact_sheets(
                        rendered_batch,
                        directory=figure_stage,
                        per_sheet=per_sheet,
                        start_index=next_sheet_index,
                    )
                    staged_sheets.extend(written)
                    next_sheet_index += len(written)
                    for _sample_id, rendered_image in rendered_batch:
                        rendered_image.close()
                    rendered_batch.clear()
            if rendered_batch:
                written = _write_contact_sheets(
                    rendered_batch,
                    directory=figure_stage,
                    per_sheet=per_sheet,
                    start_index=next_sheet_index,
                )
                staged_sheets.extend(written)
                for _sample_id, rendered_image in rendered_batch:
                    rendered_image.close()
                rendered_batch.clear()
            write_jsonl(staged_dataset, (sample.to_mapping() for sample in samples))
            manifest = build_dataset_manifest(
                samples,
                config=config,
                dataset_name=dataset_name,
                schema_version=schema_version,
            )
            generator_config = json.loads(canonical_json(config))
            render_config = {
                "height": height,
                "samples_per_contact_sheet": per_sheet,
                "width": width,
            }
            public_output = _relative_publishable(
                output,
                repository_root=repository_root,
                publication_root=publication_root,
            )
            public_images = _relative_publishable(
                images_dir,
                repository_root=repository_root,
                publication_root=publication_root,
            )
            public_sheets = [
                _relative_publishable(
                    contact_sheet_dir / path.name,
                    repository_root=repository_root,
                    publication_root=publication_root,
                )
                for path in staged_sheets
            ]
            unsigned_manifest = {
                **manifest.to_mapping(),
                "generator_config": generator_config,
                "render_config": render_config,
                "dataset_file_sha256": hashlib.sha256(staged_dataset.read_bytes()).hexdigest(),
                "image_sha256": image_sha256,
                "jsonl_path": public_output,
                "images_dir": public_images,
                "rendered_image_count": len(samples),
                "solver_checks": solver_checks,
                "solver_pass_rate": solver_checks / len(samples),
                "roundtrip_checks": roundtrip_checks,
                "roundtrip_pass_rate": 1.0,
                "contact_sheets": public_sheets,
                "contact_sheet_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in staged_sheets
                },
                "preregistered_ood_factors": list(factors),
            }
            manifest_payload = {
                **unsigned_manifest,
                "manifest_sha256": manifest_sha256(unsigned_manifest),
            }
            staged_manifest.write_text(
                json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_file_hash = hashlib.sha256(staged_manifest.read_bytes()).hexdigest()

            config_file_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
            environment = {
                **base_environment,
                "dataset_manifest_hash": manifest_file_hash,
            }
            timestamp = re.sub(r"[^0-9TZ]", "", str(environment["start_timestamp"]))
            run_id = f"generation-{timestamp}-{config_file_hash[:12]}"
            with RunLogger(
                root=log_stage,
                experiment=log_experiment,
                run_id=run_id,
                config={
                    "generator_config": generator_config,
                    "render_config": render_config,
                    "logging": {
                        "experiment": log_experiment,
                        "root": _relative_publishable(
                            log_root,
                            repository_root=repository_root,
                            publication_root=publication_root,
                        ),
                    },
                    "output": public_output,
                    "manifest": _relative_publishable(
                        manifest_path,
                        repository_root=repository_root,
                        publication_root=publication_root,
                    ),
                    "images_dir": public_images,
                    "contact_sheets": public_sheets,
                    "manifest_file_sha256": manifest_file_hash,
                    "manifest_self_sha256": manifest_payload["manifest_sha256"],
                    "content_sha256": manifest.content_sha256,
                },
                environment={
                    **environment,
                    "config_file_sha256": config_file_hash,
                    "generator_config_sha256": manifest.config_sha256,
                    "sample_count": len(samples),
                },
            ) as logger:
                logger.log_metrics(
                    {
                        "sample_count": len(samples),
                        "solver_pass_rate": solver_checks / len(samples),
                        "roundtrip_pass_rate": 1.0,
                        "rendered_image_count": len(samples),
                    }
                )
                logger.log_rollout(
                    {
                        "manifest": _relative_publishable(
                            manifest_path,
                            repository_root=repository_root,
                            publication_root=publication_root,
                        ),
                        "content_sha256": manifest.content_sha256,
                    }
                )
                logger.save_predictions(
                    {
                        "sample_index": list(range(len(samples))),
                        "solver_pass": [1] * len(samples),
                    }
                )
                logger.write_report(
                    f"# CVA-World {dataset_name} generation\n\n"
                    f"- Samples: `{len(samples)}`\n"
                    f"- Solver pass rate: `{solver_checks / len(samples):.6f}`\n"
                    "- Error round-trip pass rate: `1.000000`\n"
                    f"- Content SHA-256: `{manifest.content_sha256}`\n"
                )
                logger.finalize(checkpoint_hash=None)

            promotions: list[tuple[Path, Path]] = [
                (staged_images, images_dir),
                (staged_dataset, output),
            ]
            promotions.extend((path, contact_sheet_dir / path.name) for path in staged_sheets)
            promotions.extend(
                (
                    (log_stage / log_experiment / run_id, log_root / log_experiment / run_id),
                    (staged_manifest, manifest_path),
                )
            )
            _promote_transaction(promotions)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "samples": len(samples),
                "images": len(samples),
                "contact_sheets": len(staged_sheets),
                "content_sha256": manifest.content_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
