"""Explicit CPU-only submissions for the immutable candidate06 snapshot."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path("/projects/varunssd/louis-ssvc/modeling_v3_20260915")
OUT = BASE / "candidate06"
CHECKOUT = OUT / "checkout"
PYTHON = "/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
PARENT = "/projects/varunssd/louis-ssvc/modeling_contrast_v2_20260915"
QOS = "soujanya-poria-startfund-2026-03"


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        choices=[
            "CPU_TEST",
            "Q1_PREP",
            "Q2_COLLECTION",
            "Q1_OBSERVATION",
            "Q2_COVERAGE",
            "Q1_ANALYSIS",
            "Q2_ANALYSIS",
        ],
    )
    args = parser.parse_args()
    if len(set(args.tags)) != len(args.tags):
        raise ValueError("duplicate submission tag")
    receipt = json.loads((OUT / "DEPLOYMENT_RECEIPT.json").read_text())
    manifest_path = CHECKOUT / "CODE_SNAPSHOT_MANIFEST.json"
    if digest(manifest_path) != receipt["snapshot_manifest_sha256"]:
        raise ValueError("deployed snapshot manifest differs")
    inventory = json.loads(manifest_path.read_text())
    if any(digest(CHECKOUT / name) != value for name, value in inventory.items()):
        raise ValueError("immutable checkout bytes changed")
    specs = {
        "CPU_TEST": ("test", "CPU_acceptance_01", None, None),
        "Q1_PREP": (
            "request",
            "Q1_historical_collection",
            [
                "prepare-historical",
                "--parent",
                PARENT + "/checkout/ssvc_flow/runs/modeling_contrast_v2/N4/raw",
            ],
            None,
        ),
        "Q2_COLLECTION": (
            "request",
            "Q2_development_collection",
            [
                "collect-toy",
                "--role",
                "development",
                "--seeds",
                "101",
                "201",
                "--arms",
                "X_BASE",
                "X_VALID",
            ],
            None,
        ),
        "Q1_OBSERVATION": (
            "request",
            "Q1_observation_full",
            ["observe-toy", "--collection", str(OUT / "Q1_historical_collection")],
            "Q1_historical_collection",
        ),
        "Q2_COVERAGE": (
            "request",
            "Q2_coverage_full",
            ["coverage-study", "--collection", str(OUT / "Q2_development_collection")],
            "Q2_development_collection",
        ),
        "Q1_ANALYSIS": (
            "request",
            "Q1_analysis_full",
            ["analyze-observation", "--inputs", str(OUT / "Q1_observation_full")],
            "Q1_observation_full",
        ),
        "Q2_ANALYSIS": (
            "request",
            "Q2_analysis_full",
            ["analyze-coverage", "--inputs", str(OUT / "Q2_coverage_full")],
            "Q2_coverage_full",
        ),
    }
    # Slurm MaxSubmitPU is shared by running and pending jobs, including arrays.
    queue = subprocess.run(
        ["squeue", "--noheader", "--user", "varun024", "--format=%q"],
        capture_output=True,
        text=True,
        check=True,
    )
    active = sum(line.strip() == QOS for line in queue.stdout.splitlines())
    if active + len(args.tags) > 5:
        raise ValueError("ordinary QOS submission limit would be exceeded")
    for tag in args.tags:
        stage, relative, argv, prerequisite = specs[tag]
        if (OUT / (tag + "_SUBMIT_INTENT.json")).exists():
            raise ValueError("existing immutable submission intent: " + tag)
        if prerequisite and not (OUT / prerequisite / "COMPLETE.json").is_file():
            raise ValueError("required completed stage missing: " + prerequisite)
        target = OUT / relative
        if target.exists():
            raise ValueError("new output already exists: " + str(target))
        exports = [
            "ALL",
            "SSVC_V3_CHECKOUT=" + str(CHECKOUT),
            "SSVC_V3_PYTHON=" + PYTHON,
            "SSVC_V3_OUTPUT=" + str(target),
            "SSVC_V3_STAGE=" + stage,
            "SSVC_TEST_PARENT_ROOT=" + PARENT + "/parent_M2",
        ]
        if argv is not None:
            request = {
                "argv": [
                    *argv,
                    "--config",
                    "configs/modeling_v3/protocol.json",
                    "--out",
                    str(target),
                ]
            }
            request_path = OUT / (tag + "_REQUEST.json")
            write_new(request_path, request)
            exports += [
                "SSVC_V3_REQUEST=" + str(request_path),
                "SSVC_V3_REQUEST_SHA256=" + digest(request_path),
            ]
        command = [
            "sbatch",
            "--parsable",
            "--account=rose",
            "--qos=" + QOS,
            "--time=1-00:00:00",
            "--job-name=ssvc-v3-c06-" + tag.lower(),
            "--output=" + str(OUT / (tag + "-%j.out")),
            "--error=" + str(OUT / (tag + "-%j.err")),
            "--export=" + ",".join(exports),
            str(CHECKOUT / "ssvc_flow/scripts/modeling_v3_cpu.sbatch"),
        ]
        write_new(
            OUT / (tag + "_SUBMIT_INTENT.json"),
            {
                "command": command,
                "gpu_requested": False,
                "snapshot_manifest_sha256": digest(manifest_path),
            },
        )
        result = subprocess.run(command, capture_output=True, text=True)
        body = {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "job_id": result.stdout.strip().split(";")[0] if result.returncode == 0 else None,
        }
        write_new(OUT / (tag + "_SUBMIT_RECEIPT.json"), body)
        print(json.dumps(body), flush=True)
        result.check_returncode()


if __name__ == "__main__":
    main()
