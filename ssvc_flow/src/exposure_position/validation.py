"""Run the two import-isolated CPU suites and bind their actual output to source."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .control import code_bindings, validate_design, write_once_bytes
from .schema import EXPERIMENT_ID, file_digest, read_json, verify_source_bound, write_json


def validate(run, source_package):
    run, package = Path(run).resolve(), Path(source_package).resolve()
    if (run / "FROZEN_PLAN.json").exists() or (run / "CONTINUITY_BRIDGE.json").exists():
        raise ValueError("Validation must precede the numerical bridge and freeze")
    validate_design(package / "SER_J23_DESIGN.json")
    source = verify_source_bound(run)
    machine = read_json(run / "machine.json")
    flow = Path(__file__).resolve().parents[2]
    repository = flow.parent
    suite_groups = [
        (flow, sorted((flow / "tests").glob("test_exposure_*.py"))),
        (repository, sorted((repository / "tests").glob("test_exposure_position_*.py"))),
    ]
    tests = {
        str(p.relative_to(repository)): file_digest(p) for _, paths in suite_groups for p in paths
    }
    if not all(paths for _, paths in suite_groups):
        raise ValueError("Both SER-J23 CPU test groups must be present")
    code = code_bindings()
    attempt = run / "validation_attempts" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "SER_J23_SOURCE_PACKAGE": str(package),
        "SER_J23_LEGACY_RUN": machine["legacy_run"],
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    commands, counts = [], {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for number, (cwd, paths) in enumerate(suite_groups):
        junit = attempt / f"suite_{number}.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            *map(str, paths),
            "-q",
            f"--junitxml={junit}",
            f"--basetemp={attempt / f'suite_{number}_tmp'}",
        ]
        completed = subprocess.run(
            command, cwd=cwd, env=environment, capture_output=True, text=True, check=False
        )
        log = attempt / f"suite_{number}.log"
        write_once_bytes(log, (completed.stdout + completed.stderr).encode())
        commands.append({"argv": command, "cwd": str(cwd), "returncode": completed.returncode})
        if junit.exists():
            tree = ET.parse(junit).getroot()
            suites = [tree] if tree.tag == "testsuite" else list(tree.findall("testsuite"))
            for suite in suites:
                for field in counts:
                    counts[field] += int(suite.get(field, "0"))
        else:
            counts["errors"] += 1
    for number, arguments in enumerate((["check"], ["format", "--check"]), start=len(suite_groups)):
        targets = [str(flow / "src/exposure_position"), *map(str, tests)]
        command = [sys.executable, "-m", "ruff", *arguments, *targets]
        completed = subprocess.run(
            command, cwd=repository, env=environment, capture_output=True, text=True, check=False
        )
        write_once_bytes(
            attempt / f"suite_{number}.log", (completed.stdout + completed.stderr).encode()
        )
        commands.append(
            {"argv": command, "cwd": str(repository), "returncode": completed.returncode}
        )
    current_tests = {relative: file_digest(repository / relative) for relative in tests}
    passed = (
        counts["tests"] > 0
        and counts["errors"] == counts["failures"] == 0
        and all(c["returncode"] == 0 for c in commands)
        and code_bindings() == code
        and current_tests == tests
        and verify_source_bound(run) == source
    )
    result = {
        "schema": "ser-j23-integrated-cpu-validation-v1",
        "phase_id": EXPERIMENT_ID,
        "status": "PASS" if passed else "FAIL",
        **counts,
        "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
        "code_sha256": code,
        "tests_sha256": tests,
        "source_bound_hash": source["source_hash"],
        "commands": commands,
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "logs": [
            {"path": str(p), "sha256": file_digest(p)}
            for p in sorted(attempt.iterdir())
            if p.is_file()
        ],
        "retained_test_directories": [str(p) for p in sorted(attempt.iterdir()) if p.is_dir()],
        "real_model_calls": 0,
        "new_confirmation_generated": False,
    }
    write_json(attempt / "RECEIPT.json", result)
    write_json(run / "VALIDATION_RECEIPT.json", result)
    return result
