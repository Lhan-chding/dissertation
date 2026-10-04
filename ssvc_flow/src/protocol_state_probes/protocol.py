"""Frozen protocol preparation and stable, portable experiment identities."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PHASES = (
    "primary_core_and_controls",
    "input_output_factorial",
    "holdout_replication",
    "replication_existing_checkpoint",
)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_bytes(path, json_bytes(value))


def write_once_json(path: str | Path, value: Any) -> None:
    """An immutable observation may be re-published only byte-identically."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != json_bytes(value):
            raise ValueError(f"IMMUTABLE_RECORD_CONFLICT: {path}")
        return
    atomic_json(path, value)


def source_hashes() -> dict[str, str]:
    """Hash the small inference/semantics code once per process, never per draw."""
    module = Path(__file__).resolve().parent
    paths = list(module.glob("*.py"))
    root = module.parent
    for relative in (
        "followup_backend.py",
        "followup_updates.py",
        "optimizer_fork.py",
        "next_stage_runtime.py",
        "model_adapters/base.py",
        "model_adapters/qwen35.py",
        "modeling_v3/vlm_observation.py",
        "modeling_v3/io.py",
        "core.py",
    ):
        if (root / relative).is_file():
            paths.append(root / relative)
    return {str(path.relative_to(root)): file_hash(path) for path in sorted(paths)}


def validate_protocol(protocol: dict) -> None:
    execution = protocol["execution"]
    if (
        execution.get("frozen_inference_only") is not True
        or execution.get("new_training_allowed") is not False
    ):
        raise ValueError("Frozen inference and forbidden new training must be explicit")
    if execution.get("new_selector_fit_allowed") is not False:
        raise ValueError("New selector fitting is outside this experiment")
    if execution.get("max_concurrent_gpus") != 5 or execution.get("gpus_per_job") != 1:
        raise ValueError("Protocol requires at most five single-GPU workers")
    if execution.get("resume_chunk_draws") != 8 or execution.get("max_technical_retries") != 2:
        raise ValueError("Protocol requires eight-draw commits and at most two technical retries")
    if protocol["generation"].get("max_new_tokens") != 64:
        raise ValueError("The registered generation limit is 64 tokens")
    if protocol["generation"].get("new_likelihood_scoring") is not False:
        raise ValueError("New likelihood rescoring is forbidden")


def prepare(
    protocol_path: str | Path,
    prepared_path: str | Path,
    out: str | Path,
    runtime_path: str | Path | None = None,
) -> dict:
    """Validate precompiled public prompts against prepared data without a model."""
    from .cases import load_bundle

    protocol_path, prepared_path, out = Path(protocol_path), Path(prepared_path), Path(out)
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    bundle = load_bundle(protocol_path.parent, prepared_path=prepared_path)
    if bundle["protocol"] != protocol:
        raise ValueError("Protocol differs from the verified design bundle")
    if (
        bundle["verification"].get("recompiled_from_prepared") is not True
        or bundle["verification"].get("all_fields_match") is not True
    ):
        raise ValueError(
            "Real prepared metadata must match every compiled field before preparing a run"
        )
    cases = bundle["cases"]
    aliases = {case["case_id"]: case["case_id"] for case in cases}
    aliases.update({row["case_id"]: row["owner_case_id"] for row in bundle["aliases"]})
    by_id = {case["case_id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate case identities")
    for alias, owner in aliases.items():
        if (
            by_id[alias]["prompt"] != by_id[owner]["prompt"]
            or by_id[alias]["output_order"] != by_id[owner]["output_order"]
        ):
            raise ValueError(f"Alias is not identical public evidence: {alias}")
    for case in cases:
        if set(case["prompt"]) != {"system", "user"} or not all(
            isinstance(v, str) for v in case["prompt"].values()
        ):
            raise ValueError(f"Public prompt has unexpected fields: {case['case_id']}")
    jobs = bundle["jobs"]
    job_keys = [(job["checkpoint_id"], aliases[job["case_id"]]) for job in jobs]
    if len(set(job_keys)) != len(job_keys):
        raise ValueError("Manifest contains duplicated checkpoint/alias owner cells")
    if any(job["phase"] not in PHASES or job["draws"] % 8 for job in jobs):
        raise ValueError("Unexpected phase or non-eight-draw planned cell")
    files = {
        "protocol.json": json_bytes(protocol),
        "cases.jsonl": b"".join(json_bytes(row) for row in cases),
        "audit_labels.jsonl": b"".join(json_bytes(row) for row in bundle["audits"]),
        "program_predictions.jsonl": b"".join(json_bytes(row) for row in bundle["predictions"]),
        "alias_map.json": json_bytes(aliases),
        "checkpoints.json": json_bytes(
            read_json(protocol_path.parent / "manifests/checkpoints.json")
        ),
        "COMPILATION_VERIFICATION.json": json_bytes(bundle["verification"]),
    }
    module_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=module_root, text=True, capture_output=True, check=False
    )
    source = {
        "git_head": result.stdout.strip() if result.returncode == 0 else None,
        "code_hashes": source_hashes(),
    }
    runtime = None
    if runtime_path is not None:
        runtime_path = Path(runtime_path).resolve()
        runtime = {"path": str(runtime_path), "sha256": file_hash(runtime_path)}
    manifest = {
        "schema": "protocol-state-probes-execution-v1",
        "protocol_id": protocol["protocol_id"],
        "jobs": jobs,
        "file_hashes": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()
        },
        "source": source,
        "prepared": {"path": str(prepared_path.resolve()), "sha256": file_hash(prepared_path)},
        "runtime": runtime,
        "verification": bundle["verification"],
        "status": "PREPARED_NO_MODEL_INSTANTIATION",
        "model_instantiated": False,
        "new_training_allowed": False,
    }
    manifest["identity"] = digest(manifest)
    existing = out / "execution_manifest.json"
    if existing.exists() and read_json(existing) != manifest:
        raise ValueError(
            "Existing prepared run has a different identity; choose a new output directory"
        )
    for name, payload in files.items():
        target = out / name
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"Prepared artifact conflict: {target}")
        atomic_bytes(target, payload)
    atomic_json(existing, manifest)
    return {
        "status": manifest["status"],
        "run": str(out.resolve()),
        "identity": manifest["identity"],
        "jobs": len(jobs),
        "planned_outputs": sum(job["draws"] for job in jobs),
    }


def load_run(
    run_path: str | Path, runtime_path: str | Path | None = None, verify_code: bool = True
) -> dict:
    root = Path(run_path)
    manifest = read_json(root / "execution_manifest.json")
    identity_fields = {key: value for key, value in manifest.items() if key != "identity"}
    if digest(identity_fields) != manifest["identity"]:
        raise ValueError("Execution manifest identity is corrupt")
    verification = manifest["verification"]
    if (
        verification.get("recompiled_from_prepared") is not True
        or verification.get("all_fields_match") is not True
    ):
        raise ValueError("GPU runs require completed real prepared-data recompilation")
    for name, expected in manifest["file_hashes"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Frozen run artifact changed: {name}")
    if verify_code and source_hashes() != manifest["source"]["code_hashes"]:
        raise ValueError(
            "Frozen implementation changed since prepare; create a new reviewed run identity"
        )
    configured = manifest.get("runtime")
    runtime_path = (
        Path(runtime_path) if runtime_path else Path(configured["path"]) if configured else None
    )
    if runtime_path is not None and configured and file_hash(runtime_path) != configured["sha256"]:
        raise ValueError("Runtime configuration differs from preparation")
    protocol = read_json(root / "protocol.json")
    validate_protocol(protocol)
    return {
        "root": root,
        "manifest": manifest,
        "protocol": protocol,
        "runtime_path": runtime_path,
        "cases": {row["case_id"]: row for row in read_jsonl(root / "cases.jsonl")},
        "audits": {row["case_id"]: row for row in read_jsonl(root / "audit_labels.jsonl")},
        "predictions": {
            row["case_id"]: row for row in read_jsonl(root / "program_predictions.jsonl")
        },
        "aliases": read_json(root / "alias_map.json"),
    }


def sample_identity(
    experiment: str, checkpoint: str, owner: str, role: str, draw: int
) -> tuple[str, int]:
    if role not in {"frozen_probe", "smoke"} or draw < 0:
        raise ValueError("Unexpected generation role or draw index")
    key = digest([experiment, checkpoint, owner, role, draw])
    return key, int(key[:16], 16) % (2**63 - 1)


def safe_component(value: str) -> str:
    """Readable identifiers with a digest suffix to avoid sanitization collisions."""
    clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return clean[:120] + "-" + hashlib.sha256(value.encode()).hexdigest()[:12]
