#!/usr/bin/env python3
"""Strictly audit a CVA-World dataset against its self-hashed generation manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path


def _open_regular_file(path: Path, *, label: str, maximum_bytes: int):
    """Open a bounded regular file without following links or blocking on FIFOs."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular file: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds {maximum_bytes} byte limit")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_text(path: Path, *, label: str, maximum_bytes: int) -> str:
    with _open_regular_file(path, label=label, maximum_bytes=maximum_bytes) as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise ValueError(f"{label} exceeds {maximum_bytes} byte limit")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 text") from error


def _reject_constant(token: str) -> object:
    raise ValueError(f"non-finite JSON constant is not permitted: {token}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number is not permitted: {token}")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(text: str, *, source: Path) -> object:
    if len(text.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError(f"{source}: JSON input exceeds 16 MiB limit")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{source}: {error}") from error
    stack = [(value, 0)]
    node_count = 0
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if depth > 64:
            raise ValueError(f"{source}: JSON nesting exceeds 64 levels")
        if node_count > 1_000_000:
            raise ValueError(f"{source}: JSON structure exceeds one million nodes")
        if isinstance(node, Mapping):
            stack.extend((item, depth + 1) for item in node.values())
        elif isinstance(node, list):
            stack.extend((item, depth + 1) for item in node)
    return value


def _strict_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with _open_regular_file(
        path,
        label="dataset",
        maximum_bytes=256 * 1024 * 1024,
    ) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if len(raw_line) > 1024 * 1024:
                raise ValueError(f"{path}:{line_number}: JSONL row exceeds 1 MiB limit")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{path}:{line_number}: row must be UTF-8 text") from error
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = _strict_json(line, source=Path(f"{path}:{line_number}"))
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            records.append(value)
    if not records:
        raise ValueError("dataset must contain at least one record")
    return tuple(records)


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


def _manifest_path(value: object, *, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest {field} must be a non-empty path")
    lexical_root = root.expanduser().absolute()
    raw_candidate = Path(value).expanduser()
    if raw_candidate.is_absolute():
        raise ValueError(f"manifest {field} must be a relative path inside artifact root")
    if str(raw_candidate).startswith("<artifact-root>/"):
        raw_candidate = root / raw_candidate.name
    lexical_candidate = (
        raw_candidate.absolute()
        if raw_candidate.is_absolute()
        else (lexical_root / raw_candidate).absolute()
    )
    approved = root.expanduser().resolve(strict=False)
    candidate = (
        raw_candidate.resolve(strict=False)
        if raw_candidate.is_absolute()
        else (approved / raw_candidate).resolve(strict=False)
    )
    try:
        candidate.relative_to(approved)
    except ValueError as error:
        raise ValueError(f"manifest {field} must remain inside artifact root") from error
    try:
        lexical_candidate.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"manifest {field} must remain inside artifact root") from error
    if _has_symlink_between(lexical_candidate, lexical_root):
        raise ValueError(f"manifest {field} must not traverse a symlink")
    return candidate


def _destination_path(path: Path, *, root: Path, label: str) -> Path:
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
        lexical_candidate.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside its approved root") from error
    if _has_symlink_between(lexical_candidate, lexical_root):
        raise ValueError(f"{label} must not traverse a symlink")
    return candidate


def _publishable_path(path: Path, *, repository_root: Path, artifact_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        try:
            return f"<artifact-root>/{path.relative_to(artifact_root).as_posix()}"
        except ValueError:
            return f"<external>/{path.name}"


def _safe_component(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"{label} must be a safe path component")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.audit-stage-",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            staged = Path(stream.name)
        os.replace(staged, path)
    finally:
        if staged is not None and staged.exists():
            staged.unlink()


def _remove_promoted(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _promote_transaction(promotions: tuple[tuple[Path, Path], ...]) -> None:
    completed: list[tuple[Path, Path | None]] = []
    try:
        for staged, destination in promotions:
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup: Path | None = None
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.audit-backup-{secrets.token_hex(8)}"
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


def _validate_prior_report(path: Path, *, manifest_file_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("--overwrite requires a real prior CVA audit report")
    loaded = _strict_json(
        _read_regular_text(
            path,
            label="prior audit report",
            maximum_bytes=16 * 1024 * 1024,
        ),
        source=path,
    )
    dataset = loaded.get("dataset") if isinstance(loaded, Mapping) else None
    if (
        not isinstance(loaded, Mapping)
        or loaded.get("audit_report_schema_version") != 2
        or not isinstance(dataset, Mapping)
        or dataset.get("manifest_file_sha256") != manifest_file_sha256
    ):
        raise ValueError("--overwrite requires a provenance-bound prior CVA audit report")


def _privacy_issues(value: object, *, path: str = "record") -> tuple[str, ...]:
    issues: list[str] = []
    sensitive_key = re.compile(
        r"(?:secret|password|passwd|api[_-]?key|access[_-]?token|private[_-]?key)",
        re.IGNORECASE,
    )
    absolute_path = re.compile(r"(?:^/Users/|^/home/|^[A-Za-z]:[\\/]Users[\\/])")
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if sensitive_key.search(str(key)):
                issues.append(f"sensitive field at {child}")
            issues.extend(_privacy_issues(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_privacy_issues(item, path=f"{path}[{index}]"))
    elif isinstance(value, str) and absolute_path.search(value):
        issues.append(f"machine-specific absolute path at {path}")
    return tuple(issues)


def _validate_image_tree(
    images_dir: Path,
    *,
    sample_ids: tuple[str, ...],
    expected_hashes: object,
) -> tuple[int, tuple[str, ...], tuple[str, ...], bool, str]:
    from compbias.io.manifests import manifest_sha256

    if not isinstance(expected_hashes, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in expected_hashes.items()
    ):
        raise ValueError("manifest image_sha256 must map sample IDs to digest strings")
    if images_dir.is_symlink() or not images_dir.is_dir():
        raise ValueError("manifest images_dir must be a real directory, not a symlink")
    entries = tuple(images_dir.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("image directory contains a symlink")
    if any(not entry.is_file() for entry in entries):
        raise ValueError("image directory contains a non-file entry")
    expected_names = {f"{sample_id}.png" for sample_id in sample_ids}
    actual_names = {entry.name for entry in entries}
    missing = tuple(sorted(expected_names - actual_names))
    extras = tuple(sorted(actual_names - expected_names))
    observed_hashes = {
        sample_id: _sha256_file(images_dir / f"{sample_id}.png")
        for sample_id in sample_ids
        if (images_dir / f"{sample_id}.png").is_file()
    }
    hashes_match = set(expected_hashes) == set(sample_ids) and observed_hashes == dict(
        expected_hashes
    )
    return (
        len(observed_hashes),
        missing,
        extras,
        hashes_match,
        manifest_sha256(observed_hashes),
    )


def _human_review_binding(
    path: Path | None,
    *,
    manifest_sha256: str,
    image_set_sha256: str,
    sample_ids: tuple[str, ...],
    contact_sheets: tuple[str, ...],
) -> tuple[bool, bool, bool, dict[str, object] | None]:
    if path is None:
        return False, False, False, None
    loaded = _strict_json(
        _read_regular_text(
            path,
            label="visual audit",
            maximum_bytes=16 * 1024 * 1024,
        ),
        source=path,
    )
    if not isinstance(loaded, Mapping):
        raise ValueError("visual audit must contain a JSON object")
    fields = {
        "schema_version",
        "reviewer",
        "reviewer_type",
        "review_date",
        "review_result",
        "human_reviewer_signoff",
        "images_reviewed",
        "reviewed_sample_ids",
        "contact_sheets_reviewed",
        "reviewed_contact_sheets",
        "manifest_sha256",
        "image_set_sha256",
    }
    missing = fields - set(loaded)
    unknown = set(loaded) - fields
    if missing or unknown:
        raise ValueError(
            "visual audit fields must match the closed schema; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if loaded.get("schema_version") != 1:
        raise ValueError("visual audit schema_version must be 1")
    if not isinstance(loaded.get("human_reviewer_signoff"), bool):
        raise ValueError("visual audit human_reviewer_signoff must be boolean")
    reviewer = loaded.get("reviewer")
    reviewer_type = loaded.get("reviewer_type")
    review_date = loaded.get("review_date")
    review_result = loaded.get("review_result")
    reviewed_sample_ids = loaded.get("reviewed_sample_ids")
    reviewed_contact_sheets = loaded.get("reviewed_contact_sheets")
    images_reviewed = loaded.get("images_reviewed")
    sheets_reviewed = loaded.get("contact_sheets_reviewed")
    if (
        not isinstance(reviewer, str)
        or re.fullmatch(r"reviewer-[a-z0-9][a-z0-9-]{0,30}", reviewer) is None
    ):
        raise ValueError("visual audit reviewer must be a bounded public pseudonym")
    if reviewer_type not in {"human", "codex_agent"}:
        raise ValueError("visual audit reviewer_type must be human or codex_agent")
    if not isinstance(review_date, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", review_date) is None:
        raise ValueError("visual audit review_date must be YYYY-MM-DD")
    if review_result != "pass":
        raise ValueError("visual audit review_result must be pass")
    if (
        not isinstance(reviewed_sample_ids, list)
        or any(not isinstance(value, str) for value in reviewed_sample_ids)
        or len(reviewed_sample_ids) != len(set(reviewed_sample_ids))
        or set(reviewed_sample_ids) - set(sample_ids)
    ):
        raise ValueError("visual audit reviewed_sample_ids are invalid")
    if not isinstance(images_reviewed, int) or images_reviewed != len(reviewed_sample_ids):
        raise ValueError("visual audit images_reviewed must match reviewed_sample_ids")
    if images_reviewed > len(sample_ids):
        raise ValueError("visual audit images_reviewed exceeds rendered image count")
    if (
        not isinstance(reviewed_contact_sheets, list)
        or any(not isinstance(value, str) for value in reviewed_contact_sheets)
        or set(reviewed_contact_sheets) != set(contact_sheets)
    ):
        raise ValueError("visual audit must cover every contact sheet")
    if not isinstance(sheets_reviewed, int) or sheets_reviewed != len(contact_sheets):
        raise ValueError("visual audit contact_sheets_reviewed is inconsistent")
    declared_signoff = loaded.get("human_reviewer_signoff") is True
    if declared_signoff and reviewer_type != "human":
        raise ValueError("only reviewer_type human may declare human signoff")
    signed_off = declared_signoff and reviewer_type == "human" and images_reviewed >= 200
    binding_matches = (
        loaded.get("manifest_sha256") == manifest_sha256
        and loaded.get("image_set_sha256") == image_set_sha256
    )
    summary = {
        "signoff": signed_off,
        "reviewer": reviewer,
        "reviewer_type": reviewer_type,
        "review_date": review_date,
        "review_result": review_result,
        "reviewed_image_count": images_reviewed,
        "reviewed_sample_ids": list(reviewed_sample_ids),
        "contact_sheets_reviewed": sheets_reviewed,
        "binding_matches": binding_matches,
        "manifest_self_sha256": manifest_sha256,
        "integrity_scope": "self-reported review record; no external signature verified",
    }
    return True, signed_off, binding_matches, summary


def _contact_sheet_checks(
    manifest: Mapping[str, object], *, artifact_root: Path
) -> tuple[bool, tuple[str, ...]]:
    values = manifest.get("contact_sheets")
    expected_hashes = manifest.get("contact_sheet_sha256")
    if not isinstance(values, list) or not isinstance(expected_hashes, Mapping):
        raise ValueError("manifest contact-sheet evidence is malformed")
    observed: dict[str, str] = {}
    for value in values:
        path = _manifest_path(value, root=artifact_root, field="contact_sheets")
        if path.is_symlink() or not path.is_file():
            raise ValueError("manifest contact sheet is missing or a symlink")
        if path.name in observed:
            raise ValueError("manifest contact sheet basenames must be unique")
        observed[path.name] = _sha256_file(path)
    mismatches = tuple(
        sorted(
            name
            for name in set(observed) | set(expected_hashes)
            if observed.get(name) != expected_hashes.get(name)
        )
    )
    return not mismatches, mismatches


def _style_counterbalance_violations(
    samples: tuple[object, ...],
    *,
    iid_styles: tuple[str, ...],
    expected_realizations: int,
    fully_cross_iid_visual_styles: bool,
    image_sha256: Mapping[str, object],
) -> tuple[str, ...]:
    from compbias.envs.cva_world.renderer import is_visual_style_applicable
    from compbias.envs.cva_world.schema import SemanticSplit, TaskFamily
    from compbias.io.manifests import manifest_sha256

    violations: list[str] = []
    semantic_groups: dict[str, list[object]] = {}
    for sample in samples:
        semantic_key = manifest_sha256(
            {
                "task_family": sample.task_family.value,
                "semantic_split": sample.split_keys.semantic_split.value,
                "scene": sample.scene,
                "question": sample.question,
                "answer": sample.canonical_answer,
            }
        )
        semantic_groups.setdefault(semantic_key, []).append(sample)
    for key, group in semantic_groups.items():
        styles = {sample.split_keys.visual_style for sample in group}
        image_hashes = {image_sha256.get(sample.sample_id) for sample in group}
        is_ood = group[0].split_keys.semantic_split is SemanticSplit.OOD_TEST
        applicable_iid_styles = tuple(
            style for style in iid_styles if is_visual_style_applicable(style, group[0].task_family)
        )
        expected_count = (
            expected_realizations
            if is_ood or not fully_cross_iid_visual_styles
            else len(applicable_iid_styles)
        )
        if len(group) != expected_count or len(image_hashes) != expected_count:
            violations.append(
                f"semantic state {key}: expected {expected_count} realizations "
                "with distinct rendered images"
            )
        if not is_ood:
            if fully_cross_iid_visual_styles and styles != set(applicable_iid_styles):
                violations.append(
                    f"semantic state {key}: IID styles do not fully cross the applicable catalog"
                )
            elif not fully_cross_iid_visual_styles and len(styles) < 2:
                violations.append(f"semantic state {key}: expected at least two IID styles")
    for family in TaskFamily:
        for split in SemanticSplit:
            if split is SemanticSplit.OOD_TEST:
                continue
            group = tuple(
                sample
                for sample in samples
                if sample.task_family is family and sample.split_keys.semantic_split is split
            )
            if len(group) < len(iid_styles):
                continue
            applicable_iid_styles = tuple(
                style for style in iid_styles if is_visual_style_applicable(style, family)
            )
            counts = tuple(
                sum(sample.split_keys.visual_style == style for sample in group)
                for style in applicable_iid_styles
            )
            permitted_difference = 0 if fully_cross_iid_visual_styles else 1
            if max(counts) - min(counts) > permitted_difference:
                violations.append(f"{family.value}/{split.value}: unbalanced style counts")
    return tuple(violations)


def _visual_factor_realization_audit(
    samples: tuple[object, ...],
    *,
    configured_styles: tuple[str, ...],
    expected_style_counts: Mapping[str, int],
    style_counterbalance_violations: tuple[str, ...],
    fully_cross_iid_visual_styles: bool,
) -> dict[str, object]:
    from compbias.envs.cva_world.renderer import (
        SUPPORTED_VISUAL_STYLES,
        VISUAL_STYLE_APPLICABILITY,
        RenderConfig,
        is_visual_style_applicable,
        render_scene,
    )
    from compbias.envs.cva_world.schema import TaskFamily

    catalog = tuple(SUPPORTED_VISUAL_STYLES)
    counts = {
        style: sum(sample.split_keys.visual_style == style for sample in samples)
        for style in catalog
    }
    observed = tuple(style for style in catalog if counts[style] > 0)
    unknown_observed = {
        sample.split_keys.visual_style
        for sample in samples
        if sample.split_keys.visual_style not in catalog
    }
    applicability = {
        style: [family.value for family in VISUAL_STYLE_APPLICABILITY[style]] for style in catalog
    }
    applicability_violations = sorted(
        f"{sample.sample_id}: {sample.split_keys.visual_style} is not applicable to "
        f"{sample.task_family.value}"
        for sample in samples
        if not is_visual_style_applicable(sample.split_keys.visual_style, sample.task_family)
    )
    applicable_sample_counts = {
        style: {
            family.value: sum(
                sample.split_keys.visual_style == style and sample.task_family is family
                for sample in samples
            )
            for family in VISUAL_STYLE_APPLICABILITY[style]
        }
        for style in catalog
    }
    probe_scenes = {
        TaskFamily.DIGIT_OFFSET: {"value": 7},
        TaskFamily.COUNT_TRANSFORM: {"count": 5, "shape": "circle"},
        TaskFamily.GAUGE_CALIBRATION: {
            "reading": 2.5,
            "minimum": 0.0,
            "maximum": 10.0,
        },
        TaskFamily.BAR_CHART_AGGREGATE: {"bars": [2, 5, 3], "maximum": 10.0},
        TaskFamily.RELATION_RULE: {
            "relation": "left_of",
            "entity_pair": "audit_probe",
        },
    }
    renderer_applicable_effects = True
    renderer_nonapplicable_baseline = True
    for family, scene in probe_scenes.items():
        hashes = {
            style: hashlib.sha256(
                render_scene(
                    scene,
                    family,
                    RenderConfig(width=192, height=144, style=style, seed=31),
                ).tobytes()
            ).hexdigest()
            for style in catalog
        }
        applicable_hashes = {
            hashes[style] for style in catalog if is_visual_style_applicable(style, family)
        }
        if len(applicable_hashes) != len(VISUAL_STYLE_APPLICABILITY) - sum(
            family not in VISUAL_STYLE_APPLICABILITY[style] for style in catalog
        ):
            renderer_applicable_effects = False
        if any(
            hashes[style] != hashes["baseline"]
            for style in catalog
            if family not in VISUAL_STYLE_APPLICABILITY[style]
        ):
            renderer_nonapplicable_baseline = False
    frozen_full_coverage_required = (
        fully_cross_iid_visual_styles
        and configured_styles == catalog
        and sum(counts.values()) == len(samples)
    )
    applicable_coverage = renderer_applicable_effects and (
        not frozen_full_coverage_required
        or all(
            all(count > 0 for count in family_counts.values())
            for family_counts in applicable_sample_counts.values()
        )
    )
    nonapplicable_baseline_contract = renderer_nonapplicable_baseline and all(
        sample.split_keys.visual_style != style
        for style in catalog
        for family in TaskFamily
        if family not in VISUAL_STYLE_APPLICABILITY[style]
        for sample in samples
        if sample.task_family is family
    )
    complete = (
        configured_styles == catalog
        and observed == catalog
        and not unknown_observed
        and sum(counts.values()) == len(samples)
        and counts == dict(expected_style_counts)
        and not style_counterbalance_violations
        and not applicability_violations
        and applicable_coverage
        and nonapplicable_baseline_contract
    )
    return {
        "complete": complete,
        "catalog": list(catalog),
        "observed_styles": list(observed),
        "sample_counts": counts,
        "applicability": applicability,
        "applicable_sample_counts": applicable_sample_counts,
        "applicability_violations": applicability_violations,
        "applicable_coverage": applicable_coverage,
        "nonapplicable_baseline_contract": nonapplicable_baseline_contract,
    }


def _answer_balance(
    samples: tuple[object, ...],
    *,
    expected_samples: tuple[object, ...],
    samples_per_family_per_split: int,
) -> dict[str, object]:
    from compbias.envs.cva_world.schema import SemanticSplit, TaskFamily
    from compbias.io.manifests import canonical_json

    def collect(
        records: tuple[object, ...],
    ) -> tuple[
        dict[tuple[TaskFamily, SemanticSplit], list[object]],
        dict[tuple[TaskFamily, SemanticSplit], dict[str, object]],
    ]:
        grouped: dict[tuple[TaskFamily, SemanticSplit], list[object]] = {}
        semantic_answers: dict[tuple[TaskFamily, SemanticSplit], dict[str, object]] = {}
        for sample in records:
            key = (sample.task_family, sample.split_keys.semantic_split)
            grouped.setdefault(key, []).append(sample.canonical_answer)
            semantic_key = canonical_json(
                {
                    "task_family": sample.task_family.value,
                    "scene": sample.scene,
                    "question": sample.question,
                    "answer": sample.canonical_answer,
                }
            )
            semantic_answers.setdefault(key, {})[semantic_key] = sample.canonical_answer
        return grouped, semantic_answers

    grouped, semantic_answers = collect(samples)
    expected_grouped, expected_semantic_answers = collect(expected_samples)

    def answer_counts(answers: list[object]) -> dict[str, int]:
        result: dict[str, int] = {}
        for answer in answers:
            encoded = canonical_json(answer)
            result[encoded] = result.get(encoded, 0) + 1
        return result

    def semantic_counts(values: Mapping[str, object]) -> dict[str, int]:
        return answer_counts(list(values.values()))

    groups: dict[str, object] = {}
    counters: dict[tuple[TaskFamily, SemanticSplit], dict[str, int]] = {}
    decoded: dict[str, object] = {}
    for family in TaskFamily:
        for split in SemanticSplit:
            answers = grouped.get((family, split), [])
            counts = answer_counts(answers)
            for answer in answers:
                decoded.setdefault(canonical_json(answer), answer)
            counters[(family, split)] = counts
            groups[f"{family.value}/{split.value}"] = {
                "sample_count": len(answers),
                "support": [decoded[key] for key in sorted(counts)],
                "frequencies": [
                    {"answer": decoded[key], "count": counts[key]} for key in sorted(counts)
                ],
            }

    expected_counters = {key: answer_counts(answers) for key, answers in expected_grouped.items()}
    semantic_counters = {key: semantic_counts(values) for key, values in semantic_answers.items()}
    expected_semantic_counters = {
        key: semantic_counts(values) for key, values in expected_semantic_answers.items()
    }
    violations: list[str] = []
    for family in TaskFamily:
        for split in SemanticSplit:
            key = (family, split)
            if counters[key] != expected_counters.get(key, {}):
                violations.append(f"{family.value}/{split.value}: raw answer frequencies drifted")
    iid_ood_exact_match = True
    for family in TaskFamily:
        if semantic_counters.get((family, SemanticSplit.IID_TEST), {}) != semantic_counters.get(
            (family, SemanticSplit.OOD_TEST), {}
        ):
            iid_ood_exact_match = False
            violations.append(f"{family.value}: IID/OOD semantic answer distributions differ")
    numeric_families = tuple(
        family for family in TaskFamily if family is not TaskFamily.RELATION_RULE
    )
    numeric_exact_balance = all(
        semantic_counters.get((family, split)) == expected_semantic_counters.get((family, split))
        and len(semantic_counters.get((family, split), {})) == samples_per_family_per_split
        and set(semantic_counters.get((family, split), {}).values()) == {1}
        and counters[(family, split)] == expected_counters.get((family, split), {})
        for family in numeric_families
        for split in SemanticSplit
    )
    relation_multiclass_coverage = all(
        semantic_counters.get((TaskFamily.RELATION_RULE, split))
        == expected_semantic_counters.get((TaskFamily.RELATION_RULE, split))
        and len(semantic_counters.get((TaskFamily.RELATION_RULE, split), {}))
        == min(6, samples_per_family_per_split)
        and (
            not semantic_counters.get((TaskFamily.RELATION_RULE, split), {})
            or max(semantic_counters[(TaskFamily.RELATION_RULE, split)].values())
            - min(semantic_counters[(TaskFamily.RELATION_RULE, split)].values())
            <= 1
        )
        and counters[(TaskFamily.RELATION_RULE, split)]
        == expected_counters.get((TaskFamily.RELATION_RULE, split), {})
        for split in SemanticSplit
    )
    if not numeric_exact_balance:
        violations.append(
            "numeric groups must exactly match the registered raw frequencies and "
            "one-answer-per-semantic-state support"
        )
    if not relation_multiclass_coverage:
        violations.append(
            "relation groups must match registered frequencies with balanced multiclass support"
        )
    return {
        "complete": (
            iid_ood_exact_match and numeric_exact_balance and relation_multiclass_coverage
        ),
        "groups": groups,
        "iid_ood_exact_match": iid_ood_exact_match,
        "numeric_exact_balance": numeric_exact_balance,
        "relation_multiclass_coverage": relation_multiclass_coverage,
        "violations": violations,
    }


def _ood_image_shift(
    samples: tuple[object, ...], image_sha256: Mapping[str, object]
) -> dict[str, object]:
    from compbias.envs.cva_world.schema import SemanticSplit

    by_id = {sample.sample_id: sample for sample in samples}
    ood_samples = tuple(
        sample for sample in samples if sample.split_keys.semantic_split is SemanticSplit.OOD_TEST
    )
    violations: list[str] = []
    checked = 0
    for sample in ood_samples:
        source_id = sample.source_id
        source = by_id.get(source_id) if isinstance(source_id, str) else None
        if source is None:
            violations.append(f"{sample.sample_id}: missing IID source sample")
            continue
        ood_hash = image_sha256.get(sample.sample_id)
        source_hash = image_sha256.get(source_id)
        if not isinstance(ood_hash, str) or not isinstance(source_hash, str):
            violations.append(f"{sample.sample_id}: missing paired image hash")
            continue
        checked += 1
        if ood_hash == source_hash:
            violations.append(f"{sample.sample_id}: OOD image equals IID source image")
        if sample.split_keys.visual_style == source.split_keys.visual_style:
            violations.append(f"{sample.sample_id}: OOD visual style equals IID source style")
    return {
        "complete": checked == len(ood_samples) and not violations,
        "checked_pair_count": checked,
        "violations": violations,
    }


def _style_semantic_joint_independence(
    samples: tuple[object, ...], *, fully_cross_iid_visual_styles: bool
) -> dict[str, object]:
    from compbias.envs.cva_world.renderer import (
        SUPPORTED_VISUAL_STYLES,
        is_visual_style_applicable,
    )
    from compbias.envs.cva_world.schema import SemanticSplit, TaskFamily
    from compbias.io.manifests import canonical_json

    groups: dict[str, object] = {}
    violations: list[str] = []
    for family in TaskFamily:
        applicable_styles = tuple(
            style
            for style in SUPPORTED_VISUAL_STYLES[:-1]
            if is_visual_style_applicable(style, family)
        )
        for split in SemanticSplit:
            if split is SemanticSplit.OOD_TEST:
                continue
            state_styles: dict[str, dict[str, int]] = {}
            style_counts = {style: 0 for style in applicable_styles}
            sample_count = 0
            for sample in samples:
                if (
                    sample.task_family is not family
                    or sample.split_keys.semantic_split is not split
                ):
                    continue
                sample_count += 1
                style = sample.split_keys.visual_style
                if style not in style_counts:
                    violations.append(
                        f"{sample.sample_id}: style is outside applicable IID catalog"
                    )
                    continue
                style_counts[style] += 1
                semantic_key = canonical_json(
                    {
                        "scene": sample.scene,
                        "question": sample.question,
                        "answer": sample.canonical_answer,
                    }
                )
                state_counts = state_styles.setdefault(
                    semantic_key, {item: 0 for item in applicable_styles}
                )
                state_counts[style] += 1
            group_name = f"{family.value}/{split.value}"
            fully_crossed = sum(
                counts == {style: 1 for style in applicable_styles}
                for counts in state_styles.values()
            )
            groups[group_name] = {
                "semantic_state_count": len(state_styles),
                "expected_styles": list(applicable_styles),
                "fully_crossed_state_count": fully_crossed,
                "sample_count": sample_count,
                "style_counts": style_counts,
            }
            if not state_styles:
                violations.append(f"{group_name}: no semantic states")
            elif fully_crossed != len(state_styles):
                violations.append(
                    f"{group_name}: every semantic state must contain every "
                    "applicable IID style once"
                )
            if any(count != len(state_styles) for count in style_counts.values()):
                violations.append(f"{group_name}: style/semantic contingency is not fully crossed")
    if not fully_cross_iid_visual_styles:
        violations.append("generator config does not enable fully crossed IID visual styles")
    return {
        "complete": not violations,
        "criterion": "fully_crossed_style_by_semantic_state",
        "groups": groups,
        "violations": violations,
    }


def _deterministic_replay(
    samples: tuple[object, ...],
    *,
    expected_samples: tuple[object, ...],
    images_dir: Path,
    expected_hashes: Mapping[str, object],
    contact_sheet_paths: Mapping[str, Path],
    expected_contact_sheet_hashes: Mapping[str, object],
    samples_per_contact_sheet: int,
    seed: int,
    width: int,
    height: int,
) -> dict[str, object]:
    from compbias.envs.cva_world.renderer import (
        RenderConfig,
        build_contact_sheet,
        render_sample,
        sample_render_coordinates,
    )
    from compbias.io.manifests import canonical_json

    observed_by_id = {sample.sample_id: sample for sample in samples}
    expected_by_id = {sample.sample_id: sample for sample in expected_samples}
    generator_mismatches = sorted(
        sample_id
        for sample_id in set(observed_by_id) | set(expected_by_id)
        if sample_id not in observed_by_id
        or sample_id not in expected_by_id
        or canonical_json(observed_by_id[sample_id].to_mapping())
        != canonical_json(expected_by_id[sample_id].to_mapping())
    )
    renderer_mismatches: list[str] = []
    generated_sheet_hashes: dict[str, str] = {}
    rendered_batch: list[tuple[str, object]] = []
    sheet_index = 0
    for sample in expected_samples:
        render_seed, realization_index = sample_render_coordinates(
            sample.sample_id,
            base_seed=seed,
        )
        rendered = render_sample(
            sample,
            RenderConfig(
                width=width,
                height=height,
                seed=render_seed,
                realization_index=realization_index,
            ),
        )
        encoded = io.BytesIO()
        rendered.save(encoded, format="PNG", optimize=True)
        expected_hash = hashlib.sha256(encoded.getvalue()).hexdigest()
        actual_path = images_dir / f"{sample.sample_id}.png"
        actual_hash = _sha256_file(actual_path) if actual_path.is_file() else None
        if actual_hash != expected_hash or expected_hashes.get(sample.sample_id) != expected_hash:
            renderer_mismatches.append(sample.sample_id)
        rendered_batch.append((sample.sample_id, rendered))
        if len(rendered_batch) == samples_per_contact_sheet:
            sheet_index += 1
            sheet = build_contact_sheet(rendered_batch)
            sheet_bytes = io.BytesIO()
            sheet.save(sheet_bytes, format="PNG", optimize=True)
            generated_sheet_hashes[f"cva_contact_sheet_{sheet_index:02d}.png"] = hashlib.sha256(
                sheet_bytes.getvalue()
            ).hexdigest()
            rendered_batch = []
    if rendered_batch:
        sheet_index += 1
        sheet = build_contact_sheet(rendered_batch)
        sheet_bytes = io.BytesIO()
        sheet.save(sheet_bytes, format="PNG", optimize=True)
        generated_sheet_hashes[f"cva_contact_sheet_{sheet_index:02d}.png"] = hashlib.sha256(
            sheet_bytes.getvalue()
        ).hexdigest()
    contact_sheet_mismatches = sorted(
        name
        for name in (
            set(generated_sheet_hashes)
            | set(contact_sheet_paths)
            | set(expected_contact_sheet_hashes)
        )
        if generated_sheet_hashes.get(name) != expected_contact_sheet_hashes.get(name)
        or generated_sheet_hashes.get(name)
        != (
            _sha256_file(contact_sheet_paths[name])
            if name in contact_sheet_paths and contact_sheet_paths[name].is_file()
            else None
        )
    )
    generator_matches = not generator_mismatches
    renderer_matches = not renderer_mismatches
    contact_sheets_match = not contact_sheet_mismatches
    return {
        "complete": generator_matches and renderer_matches and contact_sheets_match,
        "generator_matches": generator_matches,
        "renderer_matches": renderer_matches,
        "contact_sheets_match": contact_sheets_match,
        "generator_mismatches": generator_mismatches,
        "renderer_mismatches": renderer_mismatches,
        "contact_sheet_mismatches": contact_sheet_mismatches,
    }


def _image_question_answer_collisions(
    samples: tuple[object, ...], image_sha256: object
) -> tuple[str, ...]:
    from compbias.io.manifests import manifest_sha256

    if not isinstance(image_sha256, Mapping):
        return ("manifest image_sha256 is not a mapping",)
    owners: dict[tuple[str, str], set[str]] = {}
    labels: dict[tuple[str, str], list[str]] = {}
    for sample in samples:
        sample_id = sample.sample_id
        question = sample.question
        answer = sample.canonical_answer
        image_hash = image_sha256.get(sample_id)
        if not isinstance(image_hash, str):
            continue
        key = (image_hash, manifest_sha256(question))
        owners.setdefault(key, set()).add(manifest_sha256(answer))
        labels.setdefault(key, []).append(sample_id)
    return tuple(
        f"{','.join(labels[key])}: identical image/question has multiple answers"
        for key, answers in sorted(owners.items())
        if len(answers) > 1
    )


def main(argv=None) -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", help="optional JSONL override")
    parser.add_argument("--manifest", type=Path, required=True, help="generation manifest JSON")
    parser.add_argument("--artifact-root", type=Path, default=repository_root)
    parser.add_argument("--visual-audit", type=Path, help="optional human-review record")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON audit report committed atomically with the run log",
    )
    parser.add_argument("--report-root", type=Path, default=repository_root / "artifacts")
    parser.add_argument("--log-root", type=Path, default=Path("artifacts/logs"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        artifact_root = args.artifact_root.expanduser().resolve(strict=False)
        if args.artifact_root.expanduser().absolute().is_symlink():
            raise ValueError("artifact root must not be a symlink")
        args.manifest = _destination_path(
            args.manifest,
            root=artifact_root,
            label="manifest input",
        )
        if args.visual_audit is not None:
            args.visual_audit = _destination_path(
                args.visual_audit,
                root=artifact_root,
                label="visual audit input",
            )
        args.output = _destination_path(
            args.output,
            root=args.report_root,
            label="audit report output",
        )
        if args.output.exists() and not args.overwrite:
            raise ValueError("audit report output exists; pass --overwrite")
        if args.output.exists():
            _validate_prior_report(
                args.output,
                manifest_file_sha256=_sha256_file(args.manifest),
            )
        args.log_root = _destination_path(
            args.log_root,
            root=args.report_root,
            label="audit log root",
        )
        loaded = _strict_json(
            _read_regular_text(
                args.manifest,
                label="manifest input",
                maximum_bytes=16 * 1024 * 1024,
            ),
            source=args.manifest,
        )
        if not isinstance(loaded, dict):
            raise ValueError("manifest must contain a JSON object")
        manifest = loaded
        manifest_fields = {
            "dataset_name",
            "schema_version",
            "sample_count",
            "sample_ids",
            "content_sha256",
            "config_sha256",
            "generator_config",
            "render_config",
            "dataset_file_sha256",
            "image_sha256",
            "jsonl_path",
            "images_dir",
            "rendered_image_count",
            "solver_checks",
            "solver_pass_rate",
            "roundtrip_checks",
            "roundtrip_pass_rate",
            "contact_sheets",
            "contact_sheet_sha256",
            "preregistered_ood_factors",
            "manifest_sha256",
        }
        missing_manifest_fields = manifest_fields - set(manifest)
        unknown_manifest_fields = set(manifest) - manifest_fields
        if missing_manifest_fields or unknown_manifest_fields:
            raise ValueError(
                "manifest fields must match the closed schema; "
                f"missing={sorted(missing_manifest_fields)}, "
                f"unknown={sorted(unknown_manifest_fields)}"
            )
        dataset_name = _safe_component(manifest.get("dataset_name"), label="manifest dataset_name")
        dataset = (
            _manifest_path(str(args.dataset), root=artifact_root, field="jsonl_path")
            if args.dataset is not None
            else _manifest_path(manifest.get("jsonl_path"), root=artifact_root, field="jsonl_path")
        )
        images_dir = _manifest_path(
            manifest.get("images_dir"), root=artifact_root, field="images_dir"
        )
        contact_sheet_paths = tuple(
            _manifest_path(value, root=artifact_root, field="contact_sheets")
            for value in manifest["contact_sheets"]
        )
        protected_inputs = (
            args.manifest,
            dataset,
            images_dir,
            *contact_sheet_paths,
            *((args.visual_audit,) if args.visual_audit is not None else ()),
        )
        if any(
            args.output == path or path in args.output.parents or args.output in path.parents
            for path in protected_inputs
        ):
            raise ValueError("audit report output must not overlap an audit input")
        if any(
            args.log_root == path or path in args.log_root.parents or args.log_root in path.parents
            for path in protected_inputs
        ):
            raise ValueError("audit log root must not overlap an audit input")
        if (
            args.output == args.log_root
            or args.output in args.log_root.parents
            or args.log_root in args.output.parents
        ):
            raise ValueError("audit report output and audit log root must be disjoint")
        experiment_dir = args.log_root / f"{dataset_name}_audit"
        if experiment_dir.is_symlink():
            raise ValueError("audit log experiment directory must not be a symlink")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    from compbias.envs.cva_world.canonical_solver import solve, solve_sample
    from compbias.envs.cva_world.corruptions import apply_error, reverse_error
    from compbias.envs.cva_world.generator import (
        GeneratorConfig,
        SplitLeakageError,
        audit_splits,
        generate_dataset,
    )
    from compbias.envs.cva_world.schema import CVASample
    from compbias.io.logging import RunLogger, capture_environment
    from compbias.io.manifests import (
        build_dataset_manifest,
        canonical_json,
        manifest_sha256,
    )

    try:
        raw_records = _strict_jsonl(dataset)
        samples = tuple(CVASample.from_mapping(record) for record in raw_records)
        noncanonical_rows = tuple(
            index
            for index, (record, sample) in enumerate(
                zip(raw_records, samples, strict=True), start=1
            )
            if canonical_json(record) != canonical_json(sample.to_mapping())
        )
        image_path_mismatches = tuple(
            sample.sample_id
            for sample in samples
            if sample.image_path != f"images/{sample.sample_id}.png"
        )
        privacy_issues = tuple(issue for record in raw_records for issue in _privacy_issues(record))

        generator_config = manifest.get("generator_config")
        if not isinstance(generator_config, Mapping):
            raise ValueError("manifest generator_config must be a mapping")
        validated_config = GeneratorConfig(**dict(generator_config))
        render_config = manifest.get("render_config")
        if not isinstance(render_config, Mapping) or set(render_config) != {
            "height",
            "samples_per_contact_sheet",
            "width",
        }:
            raise ValueError("manifest render_config must match the closed schema")
        render_width = render_config["width"]
        render_height = render_config["height"]
        per_sheet = render_config["samples_per_contact_sheet"]
        if (
            isinstance(render_width, bool)
            or not isinstance(render_width, int)
            or not 32 <= render_width <= 4096
            or isinstance(render_height, bool)
            or not isinstance(render_height, int)
            or not 32 <= render_height <= 4096
            or isinstance(per_sheet, bool)
            or not isinstance(per_sheet, int)
            or not 1 <= per_sheet <= 100
        ):
            raise ValueError("manifest render_config values are outside supported bounds")
        factors_value = manifest.get("preregistered_ood_factors")
        if not isinstance(factors_value, list) or any(
            not isinstance(value, str) for value in factors_value
        ):
            raise ValueError("manifest preregistered_ood_factors must be a string list")
        factors = tuple(factors_value)
        factors_match_config = factors == validated_config.preregistered_ood_factors
        try:
            split_audit = audit_splits(
                samples,
                preregistered_ood_factors=factors,
            )
            split_clean = split_audit.is_clean
            split_audit_payload: object = asdict(split_audit)
            split_audit_error: str | None = None
        except SplitLeakageError as error:
            split_clean = False
            split_audit_payload = None
            split_audit_error = str(error)

        solver_passes = sum(solve_sample(sample).is_consistent for sample in samples)
        roundtrip_total = 0
        roundtrip_passes = 0
        error_solver_passes = 0
        for sample in samples:
            for error in sample.error_catalog:
                roundtrip_total += 1
                perceived = apply_error(sample.scene, error)
                if reverse_error(perceived, error) == sample.scene:
                    roundtrip_passes += 1
                solve(perceived, sample.question, sample.task_family)
                error_solver_passes += 1

        rebuilt = build_dataset_manifest(
            samples,
            config=generator_config,
            dataset_name=dataset_name,
            schema_version=str(manifest.get("schema_version", "")),
        )
        observed_sample_ids = tuple(sorted(sample.sample_id for sample in samples))
        manifest_sample_ids_match = manifest.get("sample_ids") == list(observed_sample_ids)
        manifest_content_matches = manifest.get("content_sha256") == rebuilt.content_sha256
        manifest_config_matches = manifest.get("config_sha256") == manifest_sha256(generator_config)
        dataset_file_matches = manifest.get("dataset_file_sha256") == _sha256_file(dataset)
        manifest_sample_count_matches = manifest.get("sample_count") == len(samples)
        unsigned_manifest = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        observed_manifest_self_hash = manifest_sha256(unsigned_manifest)
        manifest_self_matches = manifest.get("manifest_sha256") == observed_manifest_self_hash
        (
            image_count,
            missing_images,
            extra_images,
            image_hashes_match,
            image_set_sha256,
        ) = _validate_image_tree(
            images_dir,
            sample_ids=observed_sample_ids,
            expected_hashes=manifest.get("image_sha256"),
        )
        image_set_matches = not missing_images and not extra_images
        rendered_count_matches = manifest.get("rendered_image_count") == image_count
        contact_sheet_hashes_match, contact_sheet_hash_mismatches = _contact_sheet_checks(
            manifest,
            artifact_root=artifact_root,
        )
        contact_sheet_paths = {
            Path(value).name: _manifest_path(
                value,
                root=artifact_root,
                field="contact_sheets",
            )
            for value in manifest["contact_sheets"]
        }
        style_counterbalance_violations = _style_counterbalance_violations(
            samples,
            iid_styles=tuple(validated_config.visual_styles[:-1]),
            expected_realizations=validated_config.realizations_per_semantic,
            fully_cross_iid_visual_styles=(validated_config.fully_cross_iid_visual_styles),
            image_sha256=manifest["image_sha256"],
        )
        expected_samples = generate_dataset(validated_config)
        expected_style_counts = {
            style: sum(sample.split_keys.visual_style == style for sample in expected_samples)
            for style in validated_config.visual_styles
        }
        visual_factor_realization_audit = _visual_factor_realization_audit(
            samples,
            configured_styles=validated_config.visual_styles,
            expected_style_counts=expected_style_counts,
            style_counterbalance_violations=style_counterbalance_violations,
            fully_cross_iid_visual_styles=(validated_config.fully_cross_iid_visual_styles),
        )
        answer_balance = _answer_balance(
            samples,
            expected_samples=expected_samples,
            samples_per_family_per_split=(validated_config.samples_per_family_per_split),
        )
        ood_image_shift = _ood_image_shift(samples, manifest["image_sha256"])
        style_semantic_joint_independence = _style_semantic_joint_independence(
            samples,
            fully_cross_iid_visual_styles=(validated_config.fully_cross_iid_visual_styles),
        )
        deterministic_replay = _deterministic_replay(
            samples,
            expected_samples=expected_samples,
            images_dir=images_dir,
            expected_hashes=manifest["image_sha256"],
            contact_sheet_paths=contact_sheet_paths,
            expected_contact_sheet_hashes=manifest["contact_sheet_sha256"],
            samples_per_contact_sheet=per_sheet,
            seed=validated_config.seed,
            width=render_width,
            height=render_height,
        )
        image_question_answer_collisions = _image_question_answer_collisions(
            samples,
            manifest.get("image_sha256"),
        )
        (
            visual_review_present,
            human_signoff,
            human_binding_matches,
            human_review,
        ) = _human_review_binding(
            args.visual_audit,
            manifest_sha256=observed_manifest_self_hash,
            image_set_sha256=image_set_sha256,
            sample_ids=observed_sample_ids,
            contact_sheets=tuple(manifest["contact_sheets"]),
        )
        report = {
            "audit_report_schema_version": 2,
            "sample_count": len(samples),
            "split_audit": split_audit_payload,
            "split_audit_error": split_audit_error,
            "split_clean": split_clean,
            "solver_passes": solver_passes,
            "solver_pass_rate": solver_passes / len(samples),
            "roundtrip_passes": roundtrip_passes,
            "roundtrip_total": roundtrip_total,
            "roundtrip_pass_rate": roundtrip_passes / roundtrip_total,
            "error_solver_passes": error_solver_passes,
            "error_solver_pass_rate": error_solver_passes / roundtrip_total,
            "rendered_image_count": image_count,
            "missing_images": list(missing_images),
            "extra_images": list(extra_images),
            "image_set_matches": image_set_matches,
            "rendered_image_count_matches": rendered_count_matches,
            "contact_sheet_sha256_matches": contact_sheet_hashes_match,
            "contact_sheet_hash_mismatches": list(contact_sheet_hash_mismatches),
            "manifest_sample_count_matches": manifest_sample_count_matches,
            "manifest_sample_ids_match": manifest_sample_ids_match,
            "manifest_content_sha256_matches": manifest_content_matches,
            "manifest_config_sha256_matches": manifest_config_matches,
            "manifest_dataset_file_sha256_matches": dataset_file_matches,
            "manifest_image_sha256_matches": image_hashes_match,
            "manifest_self_sha256_matches": manifest_self_matches,
            "preregistered_ood_factors_match_config": factors_match_config,
            "noncanonical_rows": list(noncanonical_rows),
            "image_path_mismatches": list(image_path_mismatches),
            "privacy_issues": list(privacy_issues),
            "image_question_answer_collisions": list(image_question_answer_collisions),
            "style_counterbalance_violations": list(style_counterbalance_violations),
            "visual_factor_realization_audit": visual_factor_realization_audit,
            "answer_balance": answer_balance,
            "ood_image_shift": ood_image_shift,
            "style_semantic_joint_independence": style_semantic_joint_independence,
            "deterministic_replay": deterministic_replay,
            "evidence_manifest_sha256": observed_manifest_self_hash,
            "evidence_image_set_sha256": image_set_sha256,
            "visual_review_present": visual_review_present,
            "human_reviewer_signoff": human_signoff,
            "human_review_binding_matches": human_binding_matches,
            "human_review": human_review,
            "dataset": {
                "manifest_path": _publishable_path(
                    args.manifest.resolve(strict=False),
                    repository_root=repository_root,
                    artifact_root=artifact_root,
                ),
                "manifest_file_sha256": _sha256_file(args.manifest),
                "manifest_self_sha256": observed_manifest_self_hash,
                "content_sha256": rebuilt.content_sha256,
                "image_set_sha256": image_set_sha256,
            },
        }
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    automatic_checks = (
        report["split_clean"],
        report["solver_pass_rate"] == 1.0,
        report["roundtrip_pass_rate"] == 1.0,
        report["error_solver_pass_rate"] == 1.0,
        report["image_set_matches"],
        report["rendered_image_count_matches"],
        report["contact_sheet_sha256_matches"],
        report["manifest_sample_count_matches"],
        report["manifest_sample_ids_match"],
        report["manifest_content_sha256_matches"],
        report["manifest_config_sha256_matches"],
        report["manifest_dataset_file_sha256_matches"],
        report["manifest_image_sha256_matches"],
        report["manifest_self_sha256_matches"],
        report["preregistered_ood_factors_match_config"],
        not report["noncanonical_rows"],
        not report["image_path_mismatches"],
        not report["privacy_issues"],
        not report["image_question_answer_collisions"],
        not report["style_counterbalance_violations"],
        report["visual_factor_realization_audit"]["complete"],
        report["answer_balance"]["complete"],
        report["ood_image_shift"]["complete"],
        report["style_semantic_joint_independence"]["complete"],
        report["deterministic_replay"]["complete"],
    )
    automatic_clean = all(automatic_checks)
    command_success = automatic_clean and (
        not report["visual_review_present"] or report["human_review_binding_matches"]
    )
    phase_d_ready = (
        automatic_clean
        and report["visual_review_present"]
        and report["human_reviewer_signoff"]
        and report["human_review_binding_matches"]
    )
    report["automatic_audit_clean"] = automatic_clean
    report["phase_d_ready"] = phase_d_ready

    payload = json.dumps(report, indent=2, sort_keys=True, default=list) + "\n"

    manifest_file_hash = _sha256_file(args.manifest)
    command_arguments = sys.argv[1:] if argv is None else argv
    environment = capture_environment(
        worktree=Path(__file__).resolve().parents[1],
        dataset_manifest_hash=manifest_file_hash,
        seed=int(validated_config.seed),
        model_revision=None,
        verl_revision=None,
        command=(sys.executable, str(Path(__file__).resolve()), *command_arguments),
    )
    timestamp = re.sub(r"[^0-9TZ]", "", str(environment["start_timestamp"]))
    experiment = f"{dataset_name}_audit"
    run_id = f"audit-{timestamp}-{manifest_file_hash[:12]}"
    logger_config = {
        "dataset": _publishable_path(
            dataset,
            repository_root=repository_root,
            artifact_root=artifact_root,
        ),
        "manifest": _publishable_path(
            args.manifest.resolve(strict=False),
            repository_root=repository_root,
            artifact_root=artifact_root,
        ),
        "artifact_root": _publishable_path(
            artifact_root,
            repository_root=repository_root,
            artifact_root=artifact_root,
        ),
        "report_root": _publishable_path(
            args.report_root.resolve(strict=False),
            repository_root=repository_root,
            artifact_root=artifact_root,
        ),
        "visual_audit": (
            None
            if args.visual_audit is None
            else _publishable_path(
                args.visual_audit.resolve(strict=False),
                repository_root=repository_root,
                artifact_root=artifact_root,
            )
        ),
        "output": _publishable_path(
            args.output.resolve(strict=False),
            repository_root=repository_root,
            artifact_root=artifact_root,
        ),
        "log_root": _publishable_path(
            args.log_root.resolve(strict=False),
            repository_root=repository_root,
            artifact_root=artifact_root,
        ),
        "overwrite": args.overwrite,
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cva-audit-publication-stage-",
        dir=args.report_root,
    ) as raw_stage:
        stage = Path(raw_stage)
        stage_logs = stage / "logs"
        with RunLogger(
            root=stage_logs,
            experiment=experiment,
            run_id=run_id,
            config=logger_config,
            environment=environment,
        ) as logger:
            logger.log_metrics(report)
            logger.log_rollout(
                {
                    "automatic_audit_clean": automatic_clean,
                    "phase_d_ready": phase_d_ready,
                    "sample_count": len(samples),
                }
            )
            logger.save_predictions(
                {
                    "solver_pass": [1] * solver_passes,
                    "roundtrip_pass": [1] * roundtrip_passes,
                }
            )
            logger.write_report(
                f"# CVA-World {dataset_name} audit\n\n"
                f"- Automatic audit clean: `{report['automatic_audit_clean']}`\n"
                f"- Human reviewer signoff: `{report['human_reviewer_signoff']}`\n"
                f"- Samples: `{len(samples)}`\n"
                f"- Solver pass rate: `{report['solver_pass_rate']:.6f}`\n"
                f"- Round-trip pass rate: `{report['roundtrip_pass_rate']:.6f}`\n"
                f"- Manifest binding: `{observed_manifest_self_hash}`\n"
                f"- Image-set binding: `{image_set_sha256}`\n"
            )
            logger.finalize(checkpoint_hash=None)
        staged_run = stage_logs / experiment / run_id
        final_run = args.log_root / experiment / run_id
        staged_report = stage / "audit.json"
        _atomic_write_text(staged_report, payload)
        _promote_transaction(
            (
                (staged_run, final_run),
                (staged_report, args.output),
            )
        )
    return 0 if command_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
