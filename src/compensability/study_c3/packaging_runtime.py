"""Deterministic, symlink-free Study C3 evidence archive construction."""

from __future__ import annotations

import gzip
import io
import tarfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from .io import read_json, sha256_file, write_json_new
from .paths import (
    ACTION_AUDIT_MANIFEST,
    ACTION_AUDIT_SUMMARY,
    ANALYSIS_INTERVALS,
    ANALYSIS_MANIFEST,
    ANALYSIS_SUMMARY,
    ANALYSIS_TABLES,
    C2_EVALUATION_MANIFEST,
    C2_EVALUATION_RAW,
    C2_EVALUATION_SUMMARY,
    C2_FIBER_ROWS,
    C2_TRAINING_MANIFEST,
    CONFIG,
    EXECUTION_CONTRACT,
    EXISTING_EVAL_MANIFEST,
    EXISTING_EVAL_SUMMARY,
    FACT_REPORT,
    FACTORIAL_EVAL_MANIFEST,
    FACTORIAL_EVAL_SUMMARY,
    GRADIENT_MANIFEST,
    GRADIENT_SUMMARY,
    PACKAGE_LOCK,
    REPORT_ROOT,
    REPORT_SHA_MANIFEST,
    ROOT,
    TRAINING_MANIFEST,
)
from .report_runtime import fact_report_markdown


def _safe_relative(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Study C3 archive member is unsafe: {path}")
    return path


def create_deterministic_evidence_archive(
    *, root: Path, sources: Iterable[Path], output: Path
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Study C3 evidence root is missing or unsafe: {root}")
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"Study C3 archive overwrite is forbidden: {output}")
    members = tuple(sorted({_safe_relative(Path(path)) for path in sources}, key=str))
    if not members:
        raise ValueError("Study C3 evidence archive cannot be empty")
    cached: list[tuple[Path, bytes]] = []
    for relative in members:
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Study C3 evidence source is missing or unsafe: {relative}")
        cached.append((relative, source.read_bytes()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as raw_stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for relative, data in cached:
            info = tarfile.TarInfo(relative.as_posix())
            info.size = len(data)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))


_PREREQUISITES = (
    (ACTION_AUDIT_MANIFEST, "STUDY_C3_ACTION_CHANNEL_AUDIT_COMPLETE"),
    (
        EXISTING_EVAL_MANIFEST,
        "STUDY_C3_EXISTING_CHECKPOINT_DECODER_INTERVENTION_COMPLETE",
    ),
    (TRAINING_MANIFEST, "STUDY_C3_FACTORIAL_TRAINING_COMPLETE"),
    (
        FACTORIAL_EVAL_MANIFEST,
        "STUDY_C3_FACTORIAL_CHECKPOINT_EVALUATION_COMPLETE",
    ),
    (GRADIENT_MANIFEST, "STUDY_C3_SHARED_GRADIENT_VALIDITY_AUDIT_COMPLETE"),
    (ANALYSIS_MANIFEST, "STUDY_C3_RESOLUTION_VALIDITY_ANALYSIS_COMPLETE"),
)


def _evidence_sources(repo_root: Path) -> tuple[Path, ...]:
    root = repo_root / ROOT
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Study C3 evidence root is missing or unsafe")
    sources: list[Path] = [
        CONFIG,
        PACKAGE_LOCK,
        C2_FIBER_ROWS,
        C2_TRAINING_MANIFEST,
        C2_EVALUATION_RAW,
        C2_EVALUATION_SUMMARY,
        C2_EVALUATION_MANIFEST,
    ]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Study C3 evidence contains a symlink: {path}")
        if not path.is_file():
            continue
        relative_to_artifacts = path.relative_to(root)
        if any(
            part == "final_adapter" or part.startswith("checkpoint-")
            for part in relative_to_artifacts.parts
        ):
            continue
        if path.suffixes[-2:] == [".tar", ".gz"]:
            continue
        sources.append(path.relative_to(repo_root))
    return tuple(sorted(set(sources), key=str))


def preflight_package(repo_root: Path = Path(".")) -> dict[str, object]:
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise ValueError("Study C3 repository root is missing or unsafe")
    manifests: dict[str, str] = {}
    for path, status in _PREREQUISITES:
        payload = read_json(repo_root / path)
        if payload.get("status") != status:
            raise ValueError(f"Study C3 package prerequisite is incomplete: {path}")
        manifests[str(path)] = sha256_file(repo_root / path)
    contract = read_json(repo_root / EXECUTION_CONTRACT)
    if contract.get("status") != "STUDY_C3_FACTORIAL_EXECUTION_CONTRACT_FROZEN":
        raise ValueError("Study C3 package execution contract is incomplete")
    sources = _evidence_sources(repo_root)
    return {
        "schema_version": 3,
        "status": "STUDY_C3_EVIDENCE_PACKAGE_PREFLIGHT_OK",
        "source_file_count": len(sources),
        "source_manifests": manifests,
        "git_commit_sha": contract["git_commit_sha"],
        "model_snapshot_sha256": contract["model_snapshot_sha256"],
        "single_seed_mechanism_pilot": True,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def run_package(*, repo_root: Path = Path("."), date_label: str | None = None) -> dict[str, object]:
    preflight = preflight_package(repo_root)
    if date_label is None:
        date_label = date.today().strftime("%Y%m%d")
    if len(date_label) != 8 or not date_label.isdigit():
        raise ValueError("Study C3 package date must be YYYYMMDD")
    facts = {
        "git_commit_sha": preflight["git_commit_sha"],
        "model_snapshot_sha256": preflight["model_snapshot_sha256"],
        "single_seed_mechanism_pilot": True,
        "action_audit": read_json(repo_root / ACTION_AUDIT_SUMMARY),
        "existing_checkpoint_decoder_intervention": read_json(repo_root / EXISTING_EVAL_SUMMARY),
        "factorial_training": read_json(repo_root / TRAINING_MANIFEST),
        "factorial_evaluation": read_json(repo_root / FACTORIAL_EVAL_SUMMARY),
        "shared_gradient_validity_audit": read_json(repo_root / GRADIENT_SUMMARY),
        "analysis_summary": read_json(repo_root / ANALYSIS_SUMMARY),
        "analysis_tables": read_json(repo_root / ANALYSIS_TABLES),
        "confidence_intervals": read_json(repo_root / ANALYSIS_INTERVALS),
    }
    report = repo_root / FACT_REPORT
    if report.is_symlink() or report.exists():
        raise FileExistsError(f"Study C3 fact report overwrite is forbidden: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        stream.write(fact_report_markdown(facts))
    sources_without_inventory = _evidence_sources(repo_root)
    inventory = {
        "schema_version": 3,
        "status": "STUDY_C3_EVIDENCE_SHA256_MANIFEST_COMPLETE",
        "files": {str(path): sha256_file(repo_root / path) for path in sources_without_inventory},
    }
    write_json_new(repo_root / REPORT_SHA_MANIFEST, inventory)
    sources = _evidence_sources(repo_root)
    archive = repo_root / REPORT_ROOT / f"study_c3_resolution_validity_evidence_{date_label}.tar.gz"
    create_deterministic_evidence_archive(root=repo_root, sources=sources, output=archive)
    return {
        **preflight,
        "status": "STUDY_C3_EVIDENCE_PACKAGE_COMPLETE",
        "fact_report_path": str(FACT_REPORT),
        "fact_report_sha256": sha256_file(report),
        "sha256_manifest_path": str(REPORT_SHA_MANIFEST),
        "sha256_manifest_sha256": sha256_file(repo_root / REPORT_SHA_MANIFEST),
        "archive_path": str(archive.relative_to(repo_root)),
        "archive_sha256": sha256_file(archive),
        "archive_member_count": len(sources),
    }


__all__ = [
    "create_deterministic_evidence_archive",
    "preflight_package",
    "run_package",
]
