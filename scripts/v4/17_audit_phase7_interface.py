"""Audit frozen Phase 7 traces without model loading or additional inference."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compensability_v4.qwen.phase7_runtime import (  # noqa: E402
    PHASE7_LOCKED_PATHS,
    load_phase7_config,
    summarize_phase7_interface_evidence,
    verify_phase7_package_lock,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/recoverability/v4_phase_7.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_7.yaml"
SUMMARY = ROOT / "artifacts/v4/phase7/evaluation/summary.json"
WORK_ROOT = ROOT / "artifacts/v4/phase7/work/phase7-r1"
OUTPUT = ROOT / "artifacts/v4/phase7/interface_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Phase 7 summary is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Phase 7 summary must contain one JSON object")
    return payload


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Phase 7 trace evidence is missing or unsafe")
    rows = tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("Phase 7 trace evidence is empty or malformed")
    return rows


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite Phase 7 interface audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase7-summary", type=Path, default=SUMMARY)
    parser.add_argument("--phase7-summary-sha256")
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 7 interface audit requires explicit --execute.")
        return 2
    if not arguments.phase7_summary_sha256:
        print("BLOCKED: Phase 7 interface audit requires --phase7-summary-sha256.")
        return 2
    try:
        summary_hash = _sha256(arguments.phase7_summary)
        if summary_hash != arguments.phase7_summary_sha256:
            raise RuntimeError("Phase 7 summary SHA-256 mismatch")
        summary = _json(arguments.phase7_summary)
        if summary.get("status") != "PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED":
            raise RuntimeError("Phase 7 summary status is not auditable")
        config = load_phase7_config(arguments.config)
        lock_hash = verify_phase7_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE7_LOCKED_PATHS,
        )
        evidence: list[dict[str, object]] = []
        trace_hashes: dict[str, str] = {}
        for checkpoint in config.checkpoints:
            trace = arguments.work_root / checkpoint / "trace_evidence.jsonl"
            rows = _jsonl(trace)
            if any(row.get("chain_row", {}).get("checkpoint") != checkpoint for row in rows):
                raise RuntimeError("Phase 7 trace checkpoint identity drifted")
            evidence.extend(rows)
            trace_hashes[checkpoint] = _sha256(trace)
        audit = {
            **summarize_phase7_interface_evidence(
                evidence,
                bootstrap_resamples=config.bootstrap_resamples,
                bootstrap_seed=config.bootstrap_seed,
                tost_margin=config.tost_margin,
            ),
            "phase7_summary_sha256": summary_hash,
            "trace_evidence_sha256": trace_hashes,
            "config_sha256": _sha256(arguments.config),
            "package_lock_sha256": lock_hash,
            "training_invoked": False,
            "rl_invoked": False,
        }
        _atomic_json(arguments.output, audit)
    except Exception as error:
        print(f"BLOCKED: Phase 7 interface audit {error}")
        return 2
    print(f"READY: Phase 7 interface audit written to {arguments.output}")
    print(f"SHA256 {_sha256(arguments.output)}  {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
