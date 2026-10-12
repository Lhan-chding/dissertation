#!/usr/bin/env python3
"""Allocated CPU checks, native prompt validation and pre-GPU registration."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f1.contract import file_hash
from sr_f1.data import load_inputs, load_tasks
from sr_f2.diagnostics import interval_slots, prepare_input
from sr_f2.protocol import register, validate_config, verify_input_files
from sr_f12.evaluation import verify_source_manifest
from sr_f12.protocol import BASELINE, immutable_json

CODE = Path(__file__).resolve().parents[2]
SUITES = ("tests/sr_f2", "tests/sr_f12", "tests/sr_f1/test_contract_boundary.py")


def reuse_cpu_acceptance(root, source_sha256):
    """Repair an interrupted registration without deleting a passed CPU attempt."""
    path = root / "CPU_ACCEPTANCE.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "PASS" or receipt.get("source_manifest_sha256") != source_sha256:
        raise PermissionError("Existing CPU acceptance belongs to different source or failed")
    checks = receipt["checks"]
    if len(checks) != len(SUITES):
        raise PermissionError("Incomplete accepted CPU checks")
    for check, suite in zip(checks, SUITES, strict=True):
        log = Path(check["log"]).resolve(strict=True)
        expected = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", suite, "-q"]
        if (
            not log.is_relative_to(root / "deployment/cpu_attempts")
            or check["exit_code"] != 0
            or check["command"] != expected
            or file_hash(log) != check["log_sha256"]
        ):
            raise PermissionError("Accepted CPU check evidence differs")
    if (
        file_hash(root / "NATIVE_DIAGNOSTIC_PROMPTS.json")
        != receipt["native_diagnostic_prompts_sha256"]
    ):
        raise PermissionError("Accepted native prompt manifest differs")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise PermissionError("Preflight requires an allocated CPU process")
    source_sha256 = verify_source_manifest(args.source_manifest, CODE)
    for name in (
        "CLAUDE_SR_F2_PLAN_zh.md",
        "DECISIONS.json",
        "CHARTQA_122_USER_CLARIFICATION.json",
    ):
        if not (CODE / "docs/sr_f2" / name).is_file():
            raise FileNotFoundError("Missing private deployment sidecar: " + name)
    if reuse_cpu_acceptance(root, source_sha256):
        verify_input_files(root)
        finish_registration(root, args.source_manifest)
        return
    attempt = root / "deployment/cpu_attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    checks = []
    for index, suite in enumerate(SUITES):
        command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", suite, "-q"]
        log = attempt / f"cpu-checks-{index}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as stream:
            result = subprocess.run(
                command,
                cwd=CODE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": str(CODE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
                check=False,
            )
        checks.append(
            dict(
                command=command,
                exit_code=result.returncode,
                log=str(log),
                log_sha256=file_hash(log),
            )
        )
        if result.returncode:
            immutable_json(
                attempt / "CPU_ACCEPTANCE_FAILURE.json", dict(status="FAIL", checks=checks)
            )
            raise RuntimeError("CPU suite failed; original log preserved")
    identity = verify_input_files(root)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(identity["model_path"], local_files_only=True)
    manifest = json.loads((root / "DIAGNOSTIC_MANIFEST.json").read_text())
    tasks, inputs = load_tasks(root), load_inputs(root)
    views = [slot for slot in interval_slots(tasks, BASELINE) if slot["draw"] == 0]
    actual = [
        dict(
            qid=slot["qid"],
            view=slot["view"],
            request=prepare_input(slot, inputs, manifest, tokenizer),
        )
        for slot in views
    ]
    immutable_json(root / "NATIVE_DIAGNOSTIC_PROMPTS.json", actual)
    receipt = dict(
        status="PASS",
        model_calls=0,
        checks=checks,
        source_manifest_sha256=source_sha256,
        native_diagnostic_prompts=len(actual),
        native_diagnostic_prompts_sha256=file_hash(root / "NATIVE_DIAGNOSTIC_PROMPTS.json"),
    )
    immutable_json(root / "CPU_ACCEPTANCE.json", receipt)
    finish_registration(root, args.source_manifest)


def finish_registration(root, source_manifest):
    decision = json.loads((CODE / "docs/sr_f2/DECISIONS.json").read_text())
    registered = register(
        root,
        validate_config(root / "config/SR_F2.json"),
        source_manifest=source_manifest,
        code_root=CODE,
        prior_audit=root / "prior_audit/inventory/SUMMARY.json",
        decisions=decision,
    )
    print(json.dumps(dict(status=registered["status"], source_commit=registered["source_commit"])))


if __name__ == "__main__":
    main()
