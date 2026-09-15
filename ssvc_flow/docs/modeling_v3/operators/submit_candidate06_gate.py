"""Submit exactly one reviewed candidate06 CPU freeze/preparation request."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path("/projects/varunssd/louis-ssvc/modeling_v3_20260915/candidate06")
CHECKOUT = BASE / "checkout"
PYTHON = "/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
QOS = "soujanya-poria-startfund-2026-03"
SNAPSHOT = "cd703f28725212aa17e86ac8d0be82890f253c955aba7293fc9d84728d34b52d"
SOURCE = "765d0450ed37a0b3fefd55cc0226c7787f85469254121e9da2f00164a4bcdfa2"
SELECTION_SHA = "d3b7ae5966d1e0435acfa9d83c66ed7c4febcb8672c1b132659352bf593c2ac9"
INPUT_BINDINGS_SHA = "5c7aea38273ed627c183179eeb2e29241e9937a4bc3b6d751b88682fd4a938b4"
TARGETS = [
    "tests",
    "docs/modeling_v3/design/reference/test_math_contracts.py",
    "--deselect=tests/test_audit_r0_remaining.py::test_cross_split_passes_generated_dataset",
]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_file(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or digest(path) != expected:
        raise ValueError("Original file is missing or changed: " + str(path))


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_cpu_acceptance(root, inventory):
    result = json.loads((root / "result.json").read_text())
    counts = result.get("counts", {})
    if (
        result.get("execution_kind") != "SERVER_CPU"
        or not str(result.get("platform", "")).startswith("Linux-")
        or not str(result.get("slurm_job_id", "")).isdigit()
        or result.get("exit_code") != 0
        or result.get("sources_unchanged_during_run") is not True
        or result.get("acceptance_scope") != "FULL_REPOSITORY_EXCEPT_SEALED_DATA_TEST"
        or result.get("requested_targets") != TARGETS
        or result.get("snapshot_manifest_sha256") != SNAPSHOT
        or any(counts.get(key) != 0 for key in ("failed", "errors", "skipped"))
        or type(counts.get("passed")) is not int
        or counts["passed"] <= 0
    ):
        raise ValueError("Current complete server CPU acceptance did not pass")
    tested = result.get("source_and_test_hashes_before", {})
    required = {
        name.removeprefix("ssvc_flow/"): expected
        for name, expected in inventory.items()
        if name.startswith(("ssvc_flow/src/", "ssvc_flow/tests/", "ssvc_flow/scripts/"))
        and name.endswith(".py")
    }
    if not required or any(tested.get(name) != expected for name, expected in required.items()):
        raise ValueError("CPU acceptance source, test or launcher identity differs")
    for name, key in (("pytest.log", "log_sha256"), ("junit.xml", "junit_sha256")):
        verify_file(root / name, result.get(key))
    cases = list(ET.parse(root / "junit.xml").iter("testcase"))
    if len(cases) != counts["passed"] or any(
        child.tag in {"failure", "error", "skipped"} for case in cases for child in case
    ):
        raise ValueError("CPU acceptance JUnit does not match successful counts")
    for module in ("test_observation_geometry", "test_coverage_models"):
        if not any(module in case.get("classname", "") for case in cases):
            raise ValueError("CPU acceptance omits required Q1/Q2 technical tests")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["Q3_FREEZE", "Q4_PREPARE"])
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument(
        "--cpu-tests", required=True, help="Actual completed full CPU acceptance directory"
    )
    args = parser.parse_args()
    manifest = CHECKOUT / "CODE_SNAPSHOT_MANIFEST.json"
    verify_file(manifest, SNAPSHOT)
    inventory = json.loads(manifest.read_text())
    for relative, expected in inventory.items():
        path = CHECKOUT / relative
        if not path.resolve().is_relative_to(CHECKOUT.resolve()):
            raise ValueError("Snapshot path escapes checkout")
        verify_file(path, expected)

    review_path = BASE / "CPU_DEVELOPMENT_REVIEW_01.json"
    verify_file(review_path, args.review_sha256)
    review = json.loads(review_path.read_text())
    if (
        review.get("status") != "REVIEWED_FOR_FREEZE_AND_Q4_PREPARATION"
        or review.get("source_sha256") != SOURCE
    ):
        raise ValueError("Current development review is required")
    cpu_tests = Path(args.cpu_tests)
    if cpu_tests.parent.resolve() != BASE.resolve() or not cpu_tests.name.startswith(
        "CPU_acceptance_"
    ):
        raise ValueError("CPU acceptance must be a candidate06 attempt")
    required = {
        str(cpu_tests / "result.json"),
        str(BASE / "Q1_observation_full/COMPLETE.json"),
        str(BASE / "Q2_coverage_full/COMPLETE.json"),
        str(BASE / "Q1_analysis_full/COMPLETE.json"),
        str(BASE / "Q2_analysis_full/COMPLETE.json"),
    }
    if not required <= set(review.get("originals", {})):
        raise ValueError("Review must bind actual CPU acceptance and both full analyses")
    for path, expected in review["originals"].items():
        verify_file(path, expected)
    verify_cpu_acceptance(cpu_tests, inventory)
    inputs = BASE / "Q3_Q4_OPERATOR_INPUTS_01"
    expected_inputs = {
        "Q3_SELECTION_DRAFT.json": SELECTION_SHA,
        "PRE_GPU_INPUT_BINDINGS.json": INPUT_BINDINGS_SHA,
    }
    for name, expected in expected_inputs.items():
        verify_file(inputs / name, expected)
    roots = [str(BASE / "Q1_observation_full"), str(BASE / "Q2_coverage_full")]
    if args.stage == "Q3_FREEZE":
        target = BASE / "Q3_SELECTION"
        argv = [
            "freeze",
            "--inputs",
            *roots,
            "--selection",
            str(inputs / "Q3_SELECTION_DRAFT.json"),
        ]
    else:
        target = BASE / "Q4_prepared_01"
        argv = [
            "prepare-gpu",
            "--bindings",
            str(inputs / "PRE_GPU_INPUT_BINDINGS.json"),
            "--cpu-tests",
            str(cpu_tests),
            "--development-roots",
            *roots,
        ]
    if target.exists():
        raise FileExistsError("Preserve existing output: " + str(target))
    intent_path = BASE / (args.stage + "_SUBMIT_INTENT.json")
    request_path = BASE / (args.stage + "_REQUEST.json")
    receipt_path = BASE / (args.stage + "_SUBMIT_RECEIPT.json")
    if any(path.exists() for path in (intent_path, request_path, receipt_path)):
        raise FileExistsError("A prior attempt exists; inspect it before any successor")
    queue = subprocess.run(
        ["squeue", "--array", "--noheader", "--user", "varun024", "--format=%q"],
        capture_output=True,
        text=True,
        check=True,
    )
    if sum(line.strip() == QOS for line in queue.stdout.splitlines()) >= 5:
        raise ValueError("Ordinary QOS has no submission capacity")
    write_new(
        request_path,
        {"argv": [*argv, "--config", "configs/modeling_v3/protocol.json", "--out", str(target)]},
    )
    exports = [
        "ALL",
        "SSVC_V3_CHECKOUT=" + str(CHECKOUT),
        "SSVC_V3_PYTHON=" + PYTHON,
        "SSVC_V3_OUTPUT=" + str(target),
        "SSVC_V3_STAGE=request",
        "SSVC_V3_REQUEST=" + str(request_path),
        "SSVC_V3_REQUEST_SHA256=" + digest(request_path),
    ]
    command = [
        "sbatch",
        "--parsable",
        "--account=rose",
        "--qos=" + QOS,
        "--time=1-00:00:00",
        "--job-name=ssvc-v3-c06-" + args.stage.lower(),
        "--output=" + str(BASE / (args.stage + "-%j.out")),
        "--error=" + str(BASE / (args.stage + "-%j.err")),
        "--export=" + ",".join(exports),
        str(CHECKOUT / "ssvc_flow/scripts/modeling_v3_cpu.sbatch"),
    ]
    write_new(
        intent_path,
        {
            "command": command,
            "gpu_requested": False,
            "snapshot_sha256": SNAPSHOT,
            "review": {"path": str(review_path), "sha256": args.review_sha256},
            "operator_sha256": digest(__file__),
        },
    )
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(("SBATCH_", "PYTHON"))
    }
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    job_match = re.fullmatch(r"([0-9]+)(?:;[^\s;]+)?", result.stdout.strip())
    receipt = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "job_id": job_match.group(1) if result.returncode == 0 and job_match else None,
    }
    write_new(receipt_path, receipt)
    print(json.dumps(receipt))
    result.check_returncode()
    if receipt["job_id"] is None:
        raise RuntimeError("Submission response is uncertain; preserve intent and do not retry")


if __name__ == "__main__":
    main()
