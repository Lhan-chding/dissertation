"""Read-only provenance inventory; source code never certifies historical execution."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

CONTRACT_FIELDS = (
    "branch",
    "real_visual_input",
    "model_generated_content",
    "final_answer_source",
    "joint_event_source",
    "ground_truth_source",
    "recoverability",
    "evidence_level",
    "source_files",
    "path_a_eligible",
)

REQUIRED_ARCHIVE_ROWS = {
    "v4_raw": {
        "artifacts/v4/phase8/confirm_data/confirm_observations.jsonl": {"image_grid_thw"},
        "artifacts/v4/phase8/confirm_data/confirm_scenes.jsonl": {"truth", "image_path"},
        "artifacts/v4/phase8/evaluation/per_scene.jsonl": {
            "image_sha256",
            "deterministic_chain_answer_exact",
            "free_generation_answer_exact",
        },
    },
    "c2": {
        "artifacts/v5/study_c2/frozen_policy_support/raw_rows.jsonl": {
            "completion",
            "token_ids",
            "truth",
            "operation",
        },
    },
    "c3": {
        "artifacts/v5/study_c3/factorial_evaluation/raw_rows.jsonl": {
            "completion",
            "completion_token_ids",
            "prompt_sha256",
            "eval_decoder",
        },
    },
}

# These are reviewed code contracts, not declarations that the code ran.
BRANCHES = (
    {
        "branch": "compbias_structured_visual",
        "real_visual_input": "unknown; source routes images through processor to generate",
        "model_generated_content": "unknown; source requests perception/reasoning/answer tags",
        "final_answer_source": "unknown",
        "joint_event_source": "unknown; same-response implementation exists but raw run absent",
        "ground_truth_source": "unknown; source renderer uses numeric values",
        "recoverability": "source/prompt/parser present; run/weights/image/raw_tokens unknown",
        "source_files": [
            "src/compbias/gpu_pilot/qwen_smoke.py",
            "src/compbias/gpu_pilot/structured_generation.py",
            "src/compbias/models/structured_parser.py",
        ],
    },
    {
        "branch": "compbias_recoverability_bridge",
        "real_visual_input": "unknown; first stage visual and second stage text in source",
        "model_generated_content": "unknown; separate stage1 and stage2 completions in source",
        "final_answer_source": "unknown",
        "joint_event_source": "not_same_completion_by_source; no new P/C/A reconstruction",
        "ground_truth_source": "unknown; semantic scene contract in source",
        "recoverability": "source present; historical execution payload/weights unknown",
        "source_files": ["src/compbias/recoverability/bridge.py"],
    },
    {
        "branch": "compensability_v4_visual_observation",
        "real_visual_input": "partial; recorded image references/grid; image bytes unavailable",
        "model_generated_content": "recorded observed_values; completion provenance partial",
        "final_answer_source": "unknown",
        "joint_event_source": "observation_only; no independently generated same-call answer",
        "ground_truth_source": "historical confirm_scenes truth and generator source",
        "recoverability": "source/config/scene/observation records; weights/images missing",
        "archive_required": "v4_raw",
        "source_files": [
            "src/compensability_v4/qwen/manual_generation.py",
            "src/compensability_v4/qwen/model_loader.py",
            "src/compensability_v4/data/v4_generator.py",
        ],
    },
    {
        "branch": "compensability_v4_free_generation_chain",
        "real_visual_input": "partial; image hash and source image route; image bytes unavailable",
        "model_generated_content": "per_scene metrics recorded; chained model calls in source",
        "final_answer_source": "second_prompt",
        "joint_event_source": "multiple_calls; cannot compose MM-CORE same-completion P/C/A",
        "ground_truth_source": "historical confirm_scenes truth; deterministic scene generator",
        "recoverability": "code/config/metrics present; original answer token evidence incomplete",
        "archive_required": "v4_raw",
        "source_files": [
            "src/compensability_v4/qwen/phase8_execution.py",
            "src/compensability_v4/qwen/phase8_confirm_runtime.py",
            "docs/QWEN_V4_EXPERIMENT_DATA_FACTS.md",
        ],
    },
    {
        "branch": "compensability_v4_deterministic_chain_endpoint",
        "real_visual_input": "partial; inherited visual observation route; image bytes unavailable",
        "model_generated_content": "world/operation in separate calls; answer computed in executor",
        "final_answer_source": "executor_h",
        "joint_event_source": "executor_degenerate; not independent answer field",
        "ground_truth_source": "historical confirm_scenes truth; deterministic scene generator",
        "recoverability": "source/config/derived endpoint metrics present; weights/images missing",
        "archive_required": "v4_raw",
        "source_files": [
            "src/compensability_v4/qwen/phase8_execution.py",
            "src/compensability_v4/qwen/phase8_confirm_runtime.py",
        ],
    },
    {
        "branch": "compensability_v5_study_a_visual_capture",
        "real_visual_input": "unknown; source opens image and calls visual observation runtime",
        "model_generated_content": "unknown; source stores four-value observation raw text/tokens",
        "final_answer_source": "unknown",
        "joint_event_source": "observation_only_by_source",
        "ground_truth_source": "unknown; source scene contract",
        "recoverability": "source present; actual run/images/weights/raw capture unknown",
        "source_files": [
            "src/compensability_v5/qwen/study_a_capture.py",
            "src/compensability_v5/qwen/study_a_runtime.py",
        ],
    },
    {
        "branch": "compensability_v5_study_b_c_common_world",
        "real_visual_input": "unknown; common-world generation backend is text-only in source",
        "model_generated_content": "unknown; source requests four-integer world action",
        "final_answer_source": "unknown",
        "joint_event_source": "executor_degenerate_by_source; no same-call answer field",
        "ground_truth_source": "unknown; source world and operation records",
        "recoverability": "source/config present; raw historical execution not inventoried",
        "source_files": [
            "src/compensability_v5/qwen/study_b_backend.py",
            "src/compensability_v5/data/common_action_schema.py",
            "src/compensability_v5/qwen/study_c_runtime.py",
        ],
    },
    {
        "branch": "study_c2_first_line_world",
        "real_visual_input": "no; backend text only; archived rows contain worlds",
        "model_generated_content": "archived completion text; support rows include token_ids",
        "final_answer_source": "executor_h",
        "joint_event_source": "same world action with executor reward; no model answer field",
        "ground_truth_source": "archived support truth/operation and local reward fiber records",
        "recoverability": "code/config/raw text/tokens/manifest present; adapter bytes absent",
        "archive_required": "c2",
        "source_files": [
            "src/compensability_v5/study_c2/action_protocol.py",
            "src/compensability_v5/study_c2/training_backend.py",
            "src/compensability_v5/study_c2/rewards.py",
            "artifacts/v5/study_c2/data/reward_fibers.jsonl",
            "artifacts/v5/study_c2/stage25_execution_contract.json",
        ],
    },
    {
        "branch": "study_c3_free_and_fsa_world",
        "real_visual_input": "no; backend text only; archived rows contain worlds",
        "model_generated_content": "archived completion and token IDs; free/FSA separate",
        "final_answer_source": "executor_h",
        "joint_event_source": "executor reward on parsed world; no model answer field",
        "ground_truth_source": "registered world/operation data; reward-channel verifier",
        "recoverability": "code/config/raw text/tokens/manifest present; adapter bytes absent",
        "archive_required": "c3",
        "source_files": [
            "src/compensability/study_c3/qwen_backend.py",
            "src/compensability/study_c3/action_taxonomy.py",
            "src/compensability/study_c3/reward_channels.py",
            "configs/v5/study_c3_resolution_validity.yaml",
        ],
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archive(path: Path) -> dict[str, Any]:
    """Hash/read every member without extraction; reject unsafe paths and links."""
    result: dict[str, Any] = {"path": str(path.resolve()), "status": "missing"}
    if not path.is_file():
        return result
    result.update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
    members: list[dict[str, Any]] = []
    names: set[str] = set()
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                normalized = PurePosixPath(member.name)
                if (
                    normalized.is_absolute()
                    or ".." in normalized.parts
                    or str(normalized) in names
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe, duplicate, or non-regular archive member")
                names.add(str(normalized))
                if member.isdir():
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("regular archive member could not be read")
                digest = hashlib.sha256()
                size = 0
                first_line = b""
                line_complete = False
                for chunk in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    if not line_complete:
                        first_line += chunk.split(b"\n", 1)[0]
                        line_complete = b"\n" in chunk or len(first_line) > 65536
                        first_line = first_line[:65536]
                if size != member.size:
                    raise ValueError("archive member byte count mismatch")
                item: dict[str, Any] = {
                    "path": member.name,
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
                if member.name.endswith(".jsonl"):
                    try:
                        row = json.loads(first_line)
                        item["first_row_keys"] = sorted(row) if isinstance(row, dict) else None
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        item["first_row_schema"] = "unknown"
                members.append(item)
        # tarfile can stop at the tar footer before validating the gzip trailer.
        with gzip.open(path, "rb") as stream:
            for _ in iter(lambda: stream.read(1024 * 1024), b""):
                pass
        result.update(status="verified_read_to_eof", members=members, member_count=len(members))
    except (OSError, EOFError, tarfile.TarError, ValueError) as error:
        result.update(status="invalid_archive", error_type=type(error).__name__, error=str(error))
    return result


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _archive_supports_contract(name: str, archive: dict[str, Any]) -> bool:
    if archive.get("status") != "verified_read_to_eof":
        return False
    members = {item["path"]: set(item.get("first_row_keys") or []) for item in archive["members"]}
    requirements = REQUIRED_ARCHIVE_ROWS.get(name)
    return bool(requirements) and all(
        keys <= members.get(path, set()) for path, keys in requirements.items()
    )


def inspect_model_receipt(path: Path) -> dict[str, Any]:
    """Import exact server hash evidence; do not label it a local weight rehash."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    revision = receipt.get("revision", "")
    model_path = receipt.get("model_path", "")
    if (
        receipt.get("status") != "DOWNLOADED_AND_HASHED"
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not isinstance(model_path, str)
        or not model_path.endswith("/snapshots/" + revision)
        or "models--Qwen--Qwen2.5-VL-3B-Instruct/" not in model_path
    ):
        raise ValueError("receipt does not identify the required verified model family")
    entries = receipt.get("files", [])
    names: set[str] = set()
    for item in entries:
        name = item.get("name", "")
        digest = item.get("sha256", "")
        size = item.get("bytes")
        if (
            not isinstance(name, str)
            or not name
            or PurePosixPath(name).name != name
            or name in names
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError("malformed or duplicate model receipt file")
        names.add(name)
    required = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    if not required <= names:
        raise ValueError("model receipt is incomplete")
    weights = [item for item in entries if item["name"].endswith(".safetensors")]
    processor = [
        item
        for item in entries
        if item["name"]
        in {
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.json",
            "vocab.json",
            "merges.txt",
        }
    ]

    def group_hash(items: list[dict[str, Any]]) -> str:
        canonical = json.dumps(
            sorted(items, key=lambda row: row["name"]), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    return {
        "status": "server_receipt_verified",
        "path": model_path,
        "revision": revision,
        "weights_sha256": group_hash(weights),
        "processor_sha256": group_hash(processor),
        "hash_algorithm": "SHA256 of sorted canonical JSON per-file name/bytes/sha256 entries",
        "file_hashes": entries,
        "active_adapter": None,
        "receipt_path": str(path.resolve()),
        "receipt_sha256": sha256_file(path),
        "verification_level": "server_download_receipt; no local weight copy",
        "historical_snapshot_equivalence": "unknown; do not equate to old ModelScope snapshot",
        "new_base_asset": True,
    }


def inventory_assets(
    repo_root: Path,
    output_dir: Path,
    *,
    archives: dict[str, Path] | None = None,
    server_evidence: Path | None = None,
    model_receipt: Path | None = None,
) -> dict[str, Any]:
    """Inventory explicit sources only; do not search unrelated user directories."""
    repo_root = repo_root.resolve()
    supplied = dict(archives or {})
    supplied.setdefault(
        "c2",
        repo_root / "artifacts/v5/study_c2/report/"
        "study_c2_identifiable_reward_grpo_evidence.tar.gz",
    )
    archive_evidence = {name: inspect_archive(path) for name, path in supplied.items()}
    sources: dict[str, dict[str, Any]] = {}
    contracts: list[dict[str, Any]] = []
    for specification in BRANCHES:
        branch = dict(specification)
        paths = list(branch.pop("source_files"))
        required = branch.pop("archive_required", None)
        for relative in paths:
            path = repo_root / relative
            sources[relative] = (
                {
                    "status": "present",
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                if path.is_file()
                else {"status": "missing", "sha256": None}
            )
        available = required and _archive_supports_contract(
            required, archive_evidence.get(required, {})
        )
        if required and not available:
            for field in CONTRACT_FIELDS[1:7]:
                branch[field] = "unknown; required original archive not verified"
        branch["evidence_level"] = "source_and_archived_records" if available else "source_only"
        branch["source_files"] = ";".join(paths)
        branch["path_a_eligible"] = False
        contracts.append(branch)
    for relative in ("src/mm_core/inventory.py", "scripts/mm_core/inventory_assets.py"):
        path = repo_root / relative
        sources[relative] = (
            {"status": "present", "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            if path.is_file()
            else {"status": "missing", "sha256": None}
        )
    manifest: dict[str, Any] = {
        "schema": "mm-core-asset-inventory-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "Qwen/Qwen2.5-VL-3B-Instruct",
        "selected_path": "B",
        "selection_reason": "no verified historical same-completion visual readings+answer",
        "historical_results_rescored": False,
        "model_calls_invoked": False,
        "source_code_is_execution_evidence": False,
        "repo_root": str(repo_root),
        "source_files": sources,
        "archives": archive_evidence,
        "contracts": contracts,
        "server_evidence": None,
        "model_snapshot": {
            "path": None,
            "revision": "unknown",
            "weights_sha256": "unknown",
            "processor_sha256": "unknown",
            "status": "not_verified",
        },
    }
    if server_evidence is not None:
        payload = json.loads(server_evidence.read_text(encoding="utf-8"))
        manifest["server_evidence"] = {
            "path": str(server_evidence.resolve()),
            "sha256": sha256_file(server_evidence),
            "observations": payload,
        }
    if model_receipt is not None:
        manifest["model_snapshot"] = inspect_model_receipt(model_receipt)
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=CONTRACT_FIELDS)
    writer.writeheader()
    writer.writerows(contracts)
    _atomic_write(output_dir / "ASSET_CONTRACTS.csv", csv_stream.getvalue())
    manifest["asset_contracts_sha256"] = sha256_file(output_dir / "ASSET_CONTRACTS.csv")
    _atomic_write(
        output_dir / "ASSET_MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest
