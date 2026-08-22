"""CPU-only, fail-closed Stage 27 fact-report construction for Study C2."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import tarfile
from collections.abc import Mapping
from pathlib import Path

REPORT_MARKDOWN = "STUDY_C2_IDENTIFIABLE_REWARD_GRPO_FACTS.md"
REPORT_ARCHIVE = "study_c2_identifiable_reward_grpo_evidence.tar.gz"
REPORT_MANIFEST = "sha256_manifest.json"
ARMS = ("C2_answer_reward", "C2_exact_state_reward")
MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 256 * 1024 * 1024
SENSITIVE_PATTERN = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|Authorization:\s*Bearer\s+\S+|"
    rb'\"(?:api[_-]?key|access[_-]?token|password)\"\s*:\s*\"[^\"]+\")',
    re.IGNORECASE,
)

CONFIG = Path("configs/v5/study_c2_identifiable_reward.yaml")
PACKAGE_LOCK = Path("configs/v5/server_package_lock.yaml")
C2_ROOT = Path("artifacts/v5/study_c2")
STAGE24_CONTRACT = C2_ROOT / "stage24_execution_contract.json"
STAGE25_CONTRACT = C2_ROOT / "stage25_execution_contract.json"
SUPPORT_ROOT = C2_ROOT / "frozen_policy_support"
GRADIENT_ROOT = C2_ROOT / "shared_gradient_audit"
TRAINING_ROOT = C2_ROOT / "training"
EVALUATION_ROOT = C2_ROOT / "evaluation"


def _arm_sources(arm: str) -> tuple[Path, ...]:
    root = TRAINING_ROOT / arm
    return (
        root / "arm_config.json",
        root / "raw_reward_trace.jsonl",
        root / "group_diagnostics.jsonl",
        root / "summary.json",
        root / "trainer_log_history.json",
        root / "manifest.json",
    )


SOURCE_FILES = (
    CONFIG,
    PACKAGE_LOCK,
    STAGE24_CONTRACT,
    STAGE25_CONTRACT,
    SUPPORT_ROOT / "raw_rows.jsonl",
    SUPPORT_ROOT / "summary.json",
    SUPPORT_ROOT / "manifest.json",
    GRADIENT_ROOT / "per_group.jsonl",
    GRADIENT_ROOT / "summary.json",
    GRADIENT_ROOT / "manifest.json",
    TRAINING_ROOT / "manifest.json",
    *_arm_sources(ARMS[0]),
    *_arm_sources(ARMS[1]),
    EVALUATION_ROOT / "raw_rows.jsonl",
    EVALUATION_ROOT / "summary.json",
    EVALUATION_ROOT / "manifest.json",
)


def _require_symlink_free_directory(path: Path, label: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"unsafe symlink component in {label}: {current}")
    if not absolute.is_dir():
        raise ValueError(f"unsafe or missing {label}: {path}")
    return absolute


def _safe_file(root: Path, relative: Path) -> Path:
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"unsafe evidence symlink: {relative}")
    if not candidate.is_file():
        raise ValueError(f"missing Stage 27 evidence file: {relative}")
    return candidate


def _expect(payload: Mapping[str, object], expected: Mapping[str, object], label: str) -> None:
    drifted = sorted(key for key, value in expected.items() if payload.get(key) != value)
    if drifted:
        raise ValueError(f"{label} drifted: {drifted}")


def _expect_hash(observed: str, expected: object, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or observed != expected:
        raise ValueError(f"{label} SHA-256 drifted")


def _validate_support(
    payloads: Mapping[Path, dict[str, object]], hashes: Mapping[Path, str]
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = payloads[SUPPORT_ROOT / "manifest.json"]
    summary = payloads[SUPPORT_ROOT / "summary.json"]
    _expect(
        manifest,
        {
            "schema_version": 2,
            "status": "STUDY_C2_FROZEN_SUPPORT_COMPLETE",
            "prompt_count": 96,
            "rollout_count": 6144,
            "rollouts_per_prompt": 64,
            "gpu_invoked": True,
            "training_invoked": False,
            "rl_invoked": False,
        },
        "Stage 23 manifest",
    )
    _expect(
        summary,
        {
            "schema_version": 2,
            "status": "REWARD_CONTRAST_IDENTIFIED",
            "rollout_count": 6144,
            "gpu_invoked": True,
        },
        "Stage 23 summary",
    )
    counts = summary.get("counts")
    selection = summary.get("k_selection")
    if (
        not isinstance(counts, Mapping)
        or sum(int(counts.get(kind, -7000)) for kind in "FSUX") != 6144
        or not isinstance(selection, Mapping)
        or selection.get("selected_k") != 8
    ):
        raise ValueError("Stage 23 summary counts or selected K drifted")
    _expect_hash(
        hashes[SUPPORT_ROOT / "raw_rows.jsonl"],
        manifest.get("raw_rows_sha256"),
        "Stage 23 raw rows",
    )
    _expect_hash(
        hashes[SUPPORT_ROOT / "summary.json"],
        manifest.get("summary_sha256"),
        "Stage 23 summary",
    )
    return manifest, summary


def _validate_gradient(
    payloads: Mapping[Path, dict[str, object]], hashes: Mapping[Path, str]
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = payloads[GRADIENT_ROOT / "manifest.json"]
    summary = payloads[GRADIENT_ROOT / "summary.json"]
    _expect(
        manifest,
        {
            "schema_version": 2,
            "status": "STUDY_C2_SHARED_GRADIENT_AUDIT_COMPLETE",
            "scientific_status": "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED",
            "continue_to_main_rl": True,
            "group_count": 768,
            "group_size": 8,
            "rollout_count": 6144,
            "gpu_invoked": True,
            "optimizer_step_invoked": False,
            "training_invoked": False,
            "rl_invoked": False,
        },
        "Stage 24 manifest",
    )
    _expect(
        summary,
        {
            "schema_version": 2,
            "status": "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED",
            "continue_to_main_rl": True,
            "group_count": 768,
            "group_size": 8,
            "rollout_count": 6144,
            "reward_hamming_distance": 635,
            "gpu_invoked": True,
            "optimizer_step_invoked": False,
            "training_invoked": False,
            "rl_invoked": False,
        },
        "Stage 24 summary",
    )
    for path, field, label in (
        (GRADIENT_ROOT / "per_group.jsonl", "per_group_sha256", "Stage 24 per-group rows"),
        (GRADIENT_ROOT / "summary.json", "summary_sha256", "Stage 24 summary"),
        (SUPPORT_ROOT / "manifest.json", "support_manifest_sha256", "Stage 23 manifest"),
        (SUPPORT_ROOT / "raw_rows.jsonl", "support_raw_rows_sha256", "Stage 23 raw rows"),
        (SUPPORT_ROOT / "summary.json", "support_summary_sha256", "Stage 23 summary"),
        (STAGE24_CONTRACT, "execution_contract_sha256", "Stage 24 execution contract"),
    ):
        _expect_hash(hashes[path], manifest.get(field), label)
    return manifest, summary


def _validate_training(
    payloads: Mapping[Path, dict[str, object]], hashes: Mapping[Path, str]
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    pair = payloads[TRAINING_ROOT / "manifest.json"]
    _expect(
        pair,
        {
            "schema_version": 2,
            "status": "STUDY_C2_TWO_ARM_TRAINING_COMPLETE",
            "training_prompt_count_per_arm": 192,
            "optimizer_steps_per_arm": 192,
            "reward_only_pair_verified": True,
            "gpu_invoked": True,
            "training_invoked": True,
            "rl_invoked": True,
        },
        "Stage 25 pair manifest",
    )
    pair_arms = pair.get("arms")
    if not isinstance(pair_arms, Mapping) or set(pair_arms) != set(ARMS):
        raise ValueError("Stage 25 pair manifest arm set drifted")
    facts: dict[str, dict[str, object]] = {}
    for arm, reward_id in zip(ARMS, ("answer_reward_v1", "exact_state_reward_v1"), strict=True):
        root = TRAINING_ROOT / arm
        manifest = payloads[root / "manifest.json"]
        summary = payloads[root / "summary.json"]
        _expect(
            manifest,
            {
                "schema_version": 2,
                "status": "STUDY_C2_ARM_TRAINING_COMPLETE",
                "arm": arm,
                "reward_function_id": reward_id,
                "training_prompt_count": 192,
                "expected_optimizer_steps": 192,
                "group_size": 8,
                "matched_pair_count": 96,
                "reward_only_pair_verified": True,
                "gpu_invoked": True,
                "optimizer_step_invoked": True,
                "training_invoked": True,
                "rl_invoked": True,
            },
            f"Stage 25 {arm} manifest",
        )
        _expect(
            summary,
            {
                "schema_version": 2,
                "status": "STUDY_C2_ARM_TRAINING_SUMMARIZED",
                "arm": arm,
                "reward_function_id": reward_id,
                "training_prompt_count": 192,
                "optimizer_steps": 192,
                "rollout_count": 1536,
                "group_size": 8,
            },
            f"Stage 25 {arm} summary",
        )
        pair_arm = pair_arms[arm]
        if not isinstance(pair_arm, Mapping):
            raise ValueError(f"Stage 25 pair entry drifted for {arm}")
        for path, field, label in (
            (root / "manifest.json", "manifest_sha256", "manifest"),
            (root / "raw_reward_trace.jsonl", "raw_reward_trace_sha256", "reward trace"),
        ):
            _expect_hash(hashes[path], pair_arm.get(field), f"Stage 25 {arm} {label}")
        for path, field, label in (
            (root / "arm_config.json", "arm_config_sha256", "arm config"),
            (root / "raw_reward_trace.jsonl", "raw_reward_trace_sha256", "reward trace"),
            (root / "group_diagnostics.jsonl", "group_diagnostics_sha256", "diagnostics"),
            (root / "summary.json", "summary_sha256", "summary"),
            (root / "trainer_log_history.json", "trainer_log_sha256", "trainer log"),
            (GRADIENT_ROOT / "manifest.json", "stage24_manifest_sha256", "Stage 24 manifest"),
            (GRADIENT_ROOT / "per_group.jsonl", "stage24_per_group_sha256", "Stage 24 rows"),
            (GRADIENT_ROOT / "summary.json", "stage24_summary_sha256", "Stage 24 summary"),
            (STAGE25_CONTRACT, "execution_contract_sha256", "execution contract"),
        ):
            _expect_hash(hashes[path], manifest.get(field), f"Stage 25 {arm} {label}")
        if manifest.get("final_adapter_sha256") != pair_arm.get("final_adapter_sha256"):
            raise ValueError(f"Stage 25 {arm} final adapter provenance drifted")
        facts[arm] = {"manifest": manifest, "summary": summary}
    return pair, facts


def _validate_evaluation(
    payloads: Mapping[Path, dict[str, object]], hashes: Mapping[Path, str]
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = payloads[EVALUATION_ROOT / "manifest.json"]
    summary = payloads[EVALUATION_ROOT / "summary.json"]
    expected = {
        "schema_version": 2,
        "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
        "evaluation_pair_count": 88,
        "evaluation_scene_count": 176,
        "sampled_rollouts": 16,
        "raw_row_count": 5632,
        "reward_only_pair_verified": True,
        "gpu_invoked": True,
        "optimizer_step_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
    }
    _expect(manifest, expected, "Stage 26 manifest")
    _expect(summary, expected, "Stage 26 summary")
    by_arm = summary.get("by_arm")
    bootstrap = summary.get("pair_bootstrap")
    if (
        not isinstance(by_arm, Mapping)
        or set(by_arm) != set(ARMS)
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("pair_count") != 88
        or bootstrap.get("bootstrap_resamples") != 10000
        or bootstrap.get("bootstrap_seed") != 2026082403
    ):
        raise ValueError("Stage 26 registered metrics drifted")
    _expect_hash(
        hashes[EVALUATION_ROOT / "raw_rows.jsonl"],
        manifest.get("raw_rows_sha256"),
        "Stage 26 raw rows",
    )
    _expect_hash(
        hashes[EVALUATION_ROOT / "summary.json"],
        manifest.get("summary_sha256"),
        "Stage 26 summary",
    )
    pair_sha = hashes[TRAINING_ROOT / "manifest.json"]
    _expect_hash(pair_sha, manifest.get("training_pair_manifest_sha256"), "Stage 25 pair manifest")
    _expect_hash(pair_sha, summary.get("training_pair_manifest_sha256"), "Stage 25 pair manifest")
    arm_manifests = manifest.get("arm_manifests")
    pair = payloads[TRAINING_ROOT / "manifest.json"]
    pair_arms = pair.get("arms")
    if not isinstance(arm_manifests, Mapping) or set(arm_manifests) != set(ARMS):
        raise ValueError("Stage 26 arm manifest provenance drifted")
    if not isinstance(pair_arms, Mapping) or set(pair_arms) != set(ARMS):
        raise ValueError("Stage 25 pair manifest arm set drifted")
    for arm in ARMS:
        entry = arm_manifests[arm]
        pair_entry = pair_arms[arm]
        arm_manifest = payloads[TRAINING_ROOT / arm / "manifest.json"]
        if not isinstance(entry, Mapping) or not isinstance(pair_entry, Mapping):
            raise ValueError(f"Stage 26 arm manifest provenance drifted for {arm}")
        _expect_hash(
            hashes[TRAINING_ROOT / arm / "manifest.json"],
            entry.get("manifest_sha256"),
            f"Stage 26 {arm} manifest",
        )
        adapter_hashes = {
            entry.get("final_adapter_sha256"),
            pair_entry.get("final_adapter_sha256"),
            arm_manifest.get("final_adapter_sha256"),
        }
        if len(adapter_hashes) != 1 or not all(
            isinstance(value, str) and len(value) == 64 for value in adapter_hashes
        ):
            raise ValueError(f"Stage 26 {arm} final adapter provenance drifted")
    return manifest, summary


def _common_provenance(
    manifests: tuple[Mapping[str, object], ...], hashes: Mapping[Path, str]
) -> None:
    for field in ("config_sha256", "fiber_rows_sha256", "b3_adapter_sha256"):
        values = {manifest.get(field) for manifest in manifests}
        if len(values) != 1 or not all(
            isinstance(value, str) and len(value) == 64 for value in values
        ):
            raise ValueError(f"Study C2 {field} provenance drifted across stages")
    for manifest in manifests:
        _expect_hash(hashes[CONFIG], manifest.get("config_sha256"), "Study C2 config")
    for manifest in manifests[1:]:
        _expect_hash(hashes[PACKAGE_LOCK], manifest.get("package_lock_sha256"), "package lock")


def _load_evidence(
    evidence_root: Path,
) -> tuple[dict[str, object], dict[str, str], dict[Path, bytes]]:
    evidence_root = _require_symlink_free_directory(evidence_root, "evidence root")
    paths = {relative: _safe_file(evidence_root, relative) for relative in SOURCE_FILES}
    source_bytes: dict[Path, bytes] = {}
    total_bytes = 0
    for relative, path in paths.items():
        size = path.stat().st_size
        if size > MAX_SOURCE_FILE_BYTES:
            raise ValueError(f"Stage 27 evidence file exceeds size limit: {relative}")
        data = path.read_bytes()
        if len(data) > MAX_SOURCE_FILE_BYTES:
            raise ValueError(f"Stage 27 evidence file exceeds size limit: {relative}")
        data.decode("utf-8")
        if SENSITIVE_PATTERN.search(data):
            raise ValueError(f"Stage 27 evidence contains a sensitive-value pattern: {relative}")
        total_bytes += len(data)
        if total_bytes > MAX_SOURCE_TOTAL_BYTES:
            raise ValueError("Stage 27 evidence exceeds total size limit")
        source_bytes[relative] = data
    hashes = {
        relative: hashlib.sha256(data).hexdigest() for relative, data in source_bytes.items()
    }
    json_paths = tuple(
        path
        for path in SOURCE_FILES
        if path.suffix == ".json" and path.name != "trainer_log_history.json"
    )
    payloads: dict[Path, dict[str, object]] = {}
    for relative in json_paths:
        payload = json.loads(source_bytes[relative])
        if not isinstance(payload, dict):
            raise ValueError(f"Stage 27 JSON root must be a mapping: {relative}")
        payloads[relative] = payload
    support_manifest, support_summary = _validate_support(payloads, hashes)
    gradient_manifest, gradient_summary = _validate_gradient(payloads, hashes)
    training_pair, training_arms = _validate_training(payloads, hashes)
    evaluation_manifest, evaluation_summary = _validate_evaluation(payloads, hashes)
    arm_manifests = tuple(training_arms[arm]["manifest"] for arm in ARMS)
    _common_provenance(
        (support_manifest, gradient_manifest, *arm_manifests, evaluation_manifest), hashes
    )
    support_summary_facts = {
        key: value for key, value in support_summary.items() if key != "per_scene"
    }
    per_scene = support_summary.get("per_scene")
    support_summary_facts["per_scene_count"] = len(per_scene) if isinstance(per_scene, list) else 0
    source_sha256 = {path.as_posix(): hashes[path] for path in SOURCE_FILES}
    facts: dict[str, object] = {
        "schema_version": 2,
        "status": "STUDY_C2_IDENTIFIABLE_REWARD_GRPO_FACTS_COMPLETE",
        "report_scope": "registered_stage23_through_stage26_facts_only",
        "gpu_invoked": False,
        "optimizer_step_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
        "stages": {
            "stage23": {"manifest": support_manifest, "summary": support_summary_facts},
            "stage24": {"manifest": gradient_manifest, "summary": gradient_summary},
            "stage25": {"pair_manifest": training_pair, "arms": training_arms},
            "stage26": {"manifest": evaluation_manifest, "summary": evaluation_summary},
        },
        "source_sha256": source_sha256,
    }
    return facts, source_sha256, source_bytes


def preflight_report(*, evidence_root: Path) -> dict[str, object]:
    """Validate all Stage 23--26 report inputs without invoking GPU code."""

    facts, sources, _ = _load_evidence(evidence_root)
    stages = facts["stages"]
    if not isinstance(stages, Mapping):  # pragma: no cover - built above
        raise RuntimeError("Stage 27 facts lost their stage mapping")
    return {
        "schema_version": 2,
        "status": "STUDY_C2_IDENTIFIABLE_REWARD_GRPO_REPORT_PREFLIGHT_OK",
        "source_file_count": len(sources),
        "source_sha256": sources,
        "upstream_statuses": {
            stage: stages[stage]["manifest" if stage != "stage25" else "pair_manifest"]["status"]
            for stage in ("stage23", "stage24", "stage25", "stage26")
        },
        "gpu_invoked": False,
        "optimizer_step_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
    }


def _write_archive(path: Path, sources: Mapping[Path, bytes]) -> None:
    with (
        path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for relative in SOURCE_FILES:
            data = sources[relative]
            info = tarfile.TarInfo(relative.as_posix())
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))


def run_report(*, evidence_root: Path, output_root: Path) -> dict[str, object]:
    """Write a deterministic, non-overwriting Stage 27 fact packet."""

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Stage 27 output exists; overwrite forbidden: {output_root}")
    facts, sources, paths = _load_evidence(evidence_root)
    output_parent = _require_symlink_free_directory(output_root.parent, "Stage 27 output parent")
    output_root = output_parent / output_root.name
    output_root.mkdir()
    markdown_path = output_root / REPORT_MARKDOWN
    archive_path = output_root / REPORT_ARCHIVE
    markdown = (
        "# Study C2 Identifiable Reward GRPO Facts\n\n```json\n"
        + json.dumps(facts, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n```\n"
    )
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(markdown)
    _write_archive(archive_path, paths)
    outputs = {
        REPORT_MARKDOWN: hashlib.sha256(markdown_path.read_bytes()).hexdigest(),
        REPORT_ARCHIVE: hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    result = {
        "schema_version": 2,
        "status": "STUDY_C2_IDENTIFIABLE_REWARD_GRPO_REPORT_COMPLETE",
        "source_file_count": len(sources),
        "source_sha256": sources,
        "outputs": outputs,
        "gpu_invoked": False,
        "optimizer_step_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
    }
    with (output_root / REPORT_MANIFEST).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


__all__ = ["preflight_report", "run_report"]
