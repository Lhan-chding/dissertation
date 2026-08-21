"""Fail-closed construction of the local v5 advisor packet."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

_REQUIRED_RESULTS = ("support_results", "confirmation_results", "reward_results")
_NATIVE_STATUS = {
    "support_results": "STUDY_B_SINGLE_SEED_COMPLETE",
    "confirmation_results": "V5_STUDY_A_EXECUTED",
    "reward_results": "STUDY_C_DIAGNOSTICS_COMPLETE",
}
_STUDY_B_ARMS = frozenset({"B0", "B1", "B2", "B3"})
_RAW_EVIDENCE_SUFFIXES = frozenset(
    {
        ".json",
        ".jsonl",
        ".md",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".log",
        ".txt",
        ".csv",
        ".tsv",
        ".lock",
    }
)
_FORBIDDEN_EVIDENCE_COMPONENTS = ("adapter", "checkpoint")
_MAX_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_SOURCE_BYTES = 256 * 1024 * 1024


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stop_signal(value: object, label: str) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("triggered"), bool)
        or not isinstance(value.get("rule"), str)
        or not value["rule"]
    ):
        raise ValueError(f"{label} is not a registered stop-signal mapping")
    return deepcopy(dict(value))


def _study_c_signals(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Study C registered_stop_signals must be a mapping")
    names = ("reward_by_fiber_interaction", "answer_up_world_down_large_fibers")
    signals = {name: _stop_signal(value.get(name), f"Study C {name}") for name in names}
    aggregate = value.get("any_registered_signal_triggered")
    if not isinstance(aggregate, bool) or aggregate != any(
        signal["triggered"] is True for signal in signals.values()
    ):
        raise ValueError("Study C registered stop-signal aggregate drifted")
    if value.get("subjective_threshold_used") is not False:
        raise ValueError("Study C stop signals must not use a subjective threshold")
    return deepcopy(dict(value))


def _normalize_native_result(name: str, payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a native result mapping")
    expected_status = _NATIVE_STATUS[name]
    if payload.get("schema_version") != 1 or payload.get("status") != expected_status:
        raise ValueError(f"{name} must use native status {expected_status}")
    if name == "confirmation_results":
        sources = payload.get("source_sha256")
        if (
            not isinstance(sources, Mapping)
            or not sources
            or any(not _sha256(value) for value in sources.values())
            or not isinstance(payload.get("by_checkpoint"), Mapping)
            or not isinstance(payload.get("by_graph_axis"), Mapping)
            or any(
                not _positive_integer(payload.get(field))
                for field in (
                    "semantic_scene_count",
                    "scenario_count",
                    "scenario_checkpoint_count",
                )
            )
            or any(
                payload.get(field) is not False
                for field in (
                    "training_invoked",
                    "rl_invoked",
                    "prompt_search_invoked",
                    "confirmatory_data_used",
                )
            )
        ):
            raise ValueError("confirmation_results is not a complete native Study A summary")
    elif name == "support_results":
        arms = payload.get("arm_results")
        if (
            payload.get("seed") != 2026082201
            or not _sha256(payload.get("model_snapshot_sha256"))
            or not isinstance(arms, Mapping)
            or set(arms) != _STUDY_B_ARMS
            or not isinstance(payload.get("primary_contrasts"), Mapping)
        ):
            raise ValueError("support_results is not a complete native Study B result")
        _stop_signal(payload.get("stop_signal"), "Study B stop_signal")
    else:
        by_arm, per_scene = payload.get("by_arm"), payload.get("per_scene")
        if (
            payload.get("seed") != 2026082301
            or not _positive_integer(payload.get("group_size"))
            or not isinstance(by_arm, Mapping)
            or not by_arm
            or not isinstance(per_scene, Sequence)
            or isinstance(per_scene, (str, bytes))
        ):
            raise ValueError("reward_results is not a complete native Study C summary")
        _study_c_signals(payload.get("registered_stop_signals"))
    return deepcopy(dict(payload))


def build_advisor_packet(results: Mapping[str, object]) -> dict[str, object]:
    """Build a compact advisor packet or return a status-only blocker."""

    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping")
    unknown = set(results) - set(_REQUIRED_RESULTS)
    if unknown:
        raise ValueError(f"unregistered result payloads: {sorted(map(str, unknown))}")
    normalized = {key: _normalize_native_result(key, payload) for key, payload in results.items()}
    missing = sorted(key for key in _REQUIRED_RESULTS if key not in normalized)
    study_b_signal = (
        _stop_signal(normalized["support_results"]["stop_signal"], "Study B stop_signal")
        if "support_results" in normalized
        else None
    )
    if (
        missing == ["reward_results"]
        and study_b_signal is not None
        and study_b_signal["triggered"] is True
    ):
        return {
            "schema_version": 1,
            "status": "PARTIAL_DECISIVE_PILOT",
            "study_c_status": "NOT_RUN_DUE_TO_REGISTERED_STOP",
            "results": deepcopy(normalized),
            "registered_stop_signals": {"study_b": study_b_signal},
        }
    if missing:
        return {
            "schema_version": 1,
            "status": "BLOCKED_MISSING_RESULTS",
            "missing_results": missing,
        }
    return {
        "schema_version": 1,
        "status": "ADVISOR_PACKET_READY",
        "results": deepcopy(normalized),
        "registered_stop_signals": {
            "study_b": study_b_signal,
            "study_c": _study_c_signals(normalized["reward_results"]["registered_stop_signals"]),
        },
    }


def write_advisor_artifacts(
    packet: Mapping[str, object], *, artifact_roots: Mapping[str, Path], output_root: Path
) -> dict[str, str]:
    """Write factual Markdown, deterministic raw archive, and SHA manifest."""

    if packet.get("status") not in {"ADVISOR_PACKET_READY", "PARTIAL_DECISIVE_PILOT"}:
        raise ValueError("advisor artifacts require complete or registered-stop facts")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("advisor artifact output already exists")
    entries: list[tuple[str, bytes]] = []
    total_bytes = 0
    for label, root in sorted(artifact_roots.items()):
        if (
            not label
            or label in {".", ".."}
            or "/" in label
            or "\\" in label
            or root.is_symlink()
            or not root.is_dir()
        ):
            raise ValueError("artifact roots must be safe named directories")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("artifact roots must not contain symlinks")
            if path.is_file():
                relative = path.relative_to(root)
                lowered_parts = tuple(part.casefold() for part in relative.parts)
                if path.suffix.casefold() not in _RAW_EVIDENCE_SUFFIXES or any(
                    marker in part
                    for part in lowered_parts
                    for marker in _FORBIDDEN_EVIDENCE_COMPONENTS
                ):
                    continue
                size = path.stat().st_size
                if size > _MAX_EVIDENCE_FILE_BYTES:
                    raise ValueError(f"raw evidence file exceeds size limit: {relative}")
                data = path.read_bytes()
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(f"raw evidence is not UTF-8 text: {relative}") from error
                total_bytes += len(data)
                if total_bytes > _MAX_ARCHIVE_SOURCE_BYTES:
                    raise ValueError("raw evidence archive exceeds the registered size limit")
                entries.append((f"{label}/{relative.as_posix()}", data))
    if not entries or len({name for name, _ in entries}) != len(entries):
        raise ValueError("raw artifact archive is empty or contains duplicate paths")
    output_root.mkdir(parents=True)
    markdown = (
        "# Qwen V5 Pilot Result Facts\n\n```json\n"
        + json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False)
        + "\n```\n"
    )
    md_path = output_root / "QWEN_V5_PILOT_RESULT_FACTS.md"
    md_path.write_text(markdown, encoding="utf-8")
    archive_path = output_root / "qwen_v5_pilot_raw_rows.tar.gz"
    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    hashes = {
        md_path.name: hashlib.sha256(md_path.read_bytes()).hexdigest(),
        archive_path.name: hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    manifest_path = output_root / "sha256_manifest.json"
    manifest_path.write_text(json.dumps(hashes, sort_keys=True, separators=(",", ":")) + "\n")
    return hashes
