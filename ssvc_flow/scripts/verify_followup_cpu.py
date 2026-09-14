#!/usr/bin/env python3
"""Run a reproducible CPU test command and retain its exact evidence locally."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Fresh ignored evidence directory")
    parser.add_argument("targets", nargs="*", default=["tests"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    def versions():
        result = {}
        for name in (
            "torch",
            "numpy",
            "pytest",
            "scipy",
            "PyYAML",
            "Pillow",
            "transformers",
            "peft",
        ):
            try:
                result[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result[name] = None
        return result

    def sources():
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for base in ("src", "tests")
            for p in sorted((root / base).rglob("*.py"))
        }

    command = [
        sys.executable,
        "-m",
        "pytest",
        *args.targets,
        "-q",
        "-ra",
        "--junitxml",
        str(out / "junit.xml"),
        "--basetemp",
        str(out / "fixtures"),
    ]
    env = os.environ.copy()
    overrides = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TOKENIZERS_PARALLELISM": "false",
    }
    env.update(overrides)
    record = {
        "cwd": str(root),
        "python_path": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "library_versions": versions(),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "dirty_state": git("status", "--short"),
        "command": command,
        "copyable_command": shlex.join(command),
        "environment_overrides": overrides,
        "source_and_test_hashes_before": sources(),
        "started_unix": time.time(),
    }
    (out / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
    with (out / "pytest.log").open("wb") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    counts = dict(passed=0, failed=0, skipped=0, errors=0)
    if (out / "junit.xml").exists():
        for case in ET.parse(out / "junit.xml").iter("testcase"):
            category = (
                "failed"
                if case.find("failure") is not None
                else "errors"
                if case.find("error") is not None
                else "skipped"
                if case.find("skipped") is not None
                else "passed"
            )
            counts[category] += 1
    record.update(
        exit_code=result.returncode,
        finished_unix=time.time(),
        counts=counts,
        log_sha256=hashlib.sha256((out / "pytest.log").read_bytes()).hexdigest(),
        sources_unchanged_during_run=record["source_and_test_hashes_before"] == sources(),
    )
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("exit_code", "counts", "log_sha256", "sources_unchanged_during_run")
            }
        )
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
