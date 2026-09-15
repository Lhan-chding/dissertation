#!/usr/bin/env python3
"""Execute server CPU acceptance against either git or a hashed code snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("targets", nargs="*", default=["tests/modeling_v3"])
    args = parser.parse_args()
    if sys.platform != "linux" or not str(os.environ.get("SLURM_JOB_ID", "")).isdigit():
        raise SystemExit("CPU acceptance requires a server Slurm CPU allocation")
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def sources():
        return {
            str(p.relative_to(root)): digest(p)
            for name in ("src", "tests", "scripts")
            for p in sorted((root / name).rglob("*.py"))
        }

    versions = {}
    for name in ("torch", "numpy", "scipy", "pytest", "transformers", "peft", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    overrides = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        # Legacy tests import shared fixtures by their file basename. Keep
        # these explicit helper roots while pytest gives collected tests
        # distinct module names (several stages have test_evaluation.py).
        "PYTHONPATH": os.pathsep.join(
            str(root / relative)
            for relative in (
                ".",
                "tests",
                "tests/modeling_v3",
                "docs/modeling_v3/design/reference",
            )
        ),
    }
    if os.environ.get("SSVC_TEST_PARENT_ROOT"):
        parent = Path(os.environ["SSVC_TEST_PARENT_ROOT"]).resolve()
        if not (parent / "manifest.json").is_file():
            raise SystemExit("Explicit legacy regression source is unavailable")
        overrides["SSVC_TEST_PARENT_ROOT"] = str(parent)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--import-mode=importlib",
        *args.targets,
        "-q",
        "-ra",
        "--junitxml",
        str(out / "junit.xml"),
        "--basetemp",
        str(out / "fixtures"),
    ]
    snapshot = root.parent / "CODE_SNAPSHOT_MANIFEST.json"
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    receipt = {
        "schema": "ssvc-v3-cpu-verification-1",
        "execution_kind": "SERVER_CPU",
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "versions": versions,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command": command,
        "requested_targets": args.targets,
        "acceptance_scope": "FULL_REPOSITORY_EXCEPT_SEALED_DATA_TEST"
        if args.targets
        == [
            "tests",
            "docs/modeling_v3/design/reference/test_math_contracts.py",
            "--deselect=tests/test_audit_r0_remaining.py::test_cross_split_passes_generated_dataset",
        ]
        else "SELECTED_ENGINEERING_TESTS",
        "environment_overrides": overrides,
        "source_and_test_hashes_before": sources(),
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "snapshot_manifest_sha256": digest(snapshot) if snapshot.exists() else None,
        "legacy_regression_manifest": {
            "path": str(parent / "manifest.json"),
            "sha256": digest(parent / "manifest.json"),
        }
        if "SSVC_TEST_PARENT_ROOT" in overrides
        else None,
        "started_unix": time.time(),
        "scientific_status": "NOT_EVALUATED",
    }
    (out / "invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (out / "pytest.log").open("wb") as log:
        run = subprocess.run(
            command,
            cwd=root,
            env={**os.environ, **overrides},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    counts = dict(passed=0, failed=0, errors=0, skipped=0)
    if (out / "junit.xml").exists():
        for case in ET.parse(out / "junit.xml").iter("testcase"):
            key = next(
                (
                    name
                    for tag, name in (
                        ("failure", "failed"),
                        ("error", "errors"),
                        ("skipped", "skipped"),
                    )
                    if case.find(tag) is not None
                ),
                "passed",
            )
            counts[key] += 1
    receipt.update(
        exit_code=run.returncode,
        finished_unix=time.time(),
        counts=counts,
        sources_unchanged_during_run=receipt["source_and_test_hashes_before"] == sources(),
        log_sha256=digest(out / "pytest.log"),
        junit_sha256=digest(out / "junit.xml") if (out / "junit.xml").exists() else None,
    )
    receipt["wall_seconds"] = receipt["finished_unix"] - receipt["started_unix"]
    (out / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("exit_code", "counts", "wall_seconds", "sources_unchanged_during_run")
            }
        )
    )
    return run.returncode if receipt["sources_unchanged_during_run"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
