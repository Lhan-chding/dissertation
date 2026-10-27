"""Seal the census preflight only after actual program and visual evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import file_hash, now, read, safe_evidence, write_once
from .gates import PREFLIGHT_REQUIRED, _context, require_census_ready


def native_regression(root):
    root = Path(root)
    destination = root / "PREFLIGHT/NATIVE_REGRESSION.json"
    if destination.exists():
        raise FileExistsError("Preserve existing native regression result")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--import-mode=importlib",
        "-q",
        "tests/sr_f6/test_measurement.py",
        "tests/sr_f6/test_objective.py",
        "tests/sr_f6/test_credit.py",
        "tests/sr_f6/test_runtime.py",
    ]
    started = now()
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    log = root / "PREFLIGHT/NATIVE_REGRESSION.log"
    with log.open("x") as stream:
        stream.write(result.stdout + result.stderr)
    report = {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "command": command,
        "started_at": started,
        "completed_at": now(),
        "returncode": result.returncode,
        "log_sha256": file_hash(log),
        "config_sha256": file_hash(root / "CONFIG.json"),
        "source_manifest_sha256": file_hash(root / "SOURCE_MANIFEST.json"),
        "model_calls": 0,
        "optimizer_updates": 0,
        "scope": "server native code and CPU tensors; actual GPU trajectory check remains separate",
    }
    write_once(destination, report)
    if result.returncode:
        raise RuntimeError("Native regression failed; see retained log")
    return report


def seal_census(root):
    root = Path(root)
    context = _context(root, root / "CONFIG.json")
    source = read(root / "SOURCE_BINDING.json")
    processor = read(root / "PREFLIGHT/PROCESSOR.json")
    visual = read(root / "PREFLIGHT/VISUAL_REVIEW.json")
    regression = read(root / "PREFLIGHT/NATIVE_REGRESSION.json")
    audit = read(root / "DATA/CPU_AUDIT.json")
    history = read(root / "PREFLIGHT/HISTORICAL_WORLD_AUDIT.json")
    if source.get("execution_origin") != "SERVER_ACTUAL" or not source.get("model_files"):
        raise PermissionError("No actual server source binding")
    if set(source.get("parents", {})) != {"b0", "b1", "b2"}:
        raise PermissionError("Three frozen parent identities required")
    if (
        processor.get("execution_origin") != "SERVER_ACTUAL"
        or processor.get("config_file_sha256") != context["config_sha256"]
        or processor.get("data_manifest_sha256") != context["data_manifest"]["sha256"]
    ):
        raise PermissionError("Processor evidence identity differs")
    images = {r["qid"]: r for r in processor.get("images", [])}
    if len(images) != 48:
        raise PermissionError("Exactly 48 real processor images required")
    reviews = visual.get("items", [])
    if (
        visual.get("status") != "PASS"
        or visual.get("reviewer_type") not in {"HUMAN", "MODEL_VISUAL"}
        or visual.get("processor_receipt_sha256") != file_hash(root / "PREFLIGHT/PROCESSOR.json")
        or len(reviews) != 48
        or {r["qid"] for r in reviews} != set(images)
    ):
        raise PermissionError("Actual independent visual review is incomplete")
    for item in reviews:
        image = images[item["qid"]]
        if (
            item.get("status") != "PASS"
            or not item.get("observation")
            or item.get("processed_file_sha256") != image["processed_file_sha256"]
            or item.get("source_sha256") != image["source_sha256"]
            or file_hash(safe_evidence(root, image["processed_file"]))
            != image["processed_file_sha256"]
        ):
            raise PermissionError("Visual item missing, changed, or unresolved")
    if (
        regression.get("status") != "PASS"
        or regression.get("returncode") != 0
        or regression.get("config_sha256") != context["config_sha256"]
        or regression.get("source_manifest_sha256") != context["source_manifest"]["sha256"]
        or regression.get("log_sha256") != file_hash(root / "PREFLIGHT/NATIVE_REGRESSION.log")
    ):
        raise PermissionError("Actual server native regression is not bound")
    if audit.get("status") != "PASS" or history.get("status") != "PASS":
        raise PermissionError("Data mathematical/historical audit failed")
    if audit.get("training_manipulation", {}).get("status") != "PASS":
        raise PermissionError("INSUFFICIENT_REGISTERED_EXPOSURE")
    fixtures = processor.get("tokenizer_fixtures", [])
    if len(fixtures) < 3 or any(x.get("exact_roundtrip") is not True for x in fixtures):
        raise PermissionError("Actual tokenizer fixtures incomplete")
    from .census import census_inputs, validate_census_spec

    public, tasks = census_inputs(root)
    validate_census_spec(root, read(root / "CONFIG.json"), public, tasks)
    reports = [
        "SOURCE_BINDING.json",
        "PREFLIGHT/PROCESSOR.json",
        "PREFLIGHT/VISUAL_REVIEW.json",
        "PREFLIGHT/NATIVE_REGRESSION.json",
        "DATA/CPU_AUDIT.json",
        "PREFLIGHT/HISTORICAL_WORLD_AUDIT.json",
        "CENSUS_SPEC_FROZEN.json",
    ]
    report = {
        "status": "PASS",
        "context": context,
        "checks": dict.fromkeys(PREFLIGHT_REQUIRED, True),
        "verified_at": now(),
        "inputs": [{"path": p, "sha256": file_hash(root / p)} for p in reports],
        "limitation": (
            "permits DEV_CAL collection only; "
            "first registered G8 requires actual probability replay"
        ),
    }
    path = "PREFLIGHT/CENSUS_PREREQUISITES.json"
    write_once(root / path, report)
    receipt = {
        "status": "READY_FOR_CENSUS",
        "phase": "DEV_CAL",
        "context": context,
        "execution_origin": "SERVER_ACTUAL",
        "optimizer_updates": 0,
        "evidence": [
            {"path": path, "sha256": file_hash(root / path), "checks": sorted(PREFLIGHT_REQUIRED)}
        ],
    }
    write_once(root / "PREFLIGHT/CENSUS_READY.json", receipt)
    require_census_ready(root)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["native-regression", "seal-census"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = (
        native_regression(args.root)
        if args.command == "native-regression"
        else seal_census(args.root)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
