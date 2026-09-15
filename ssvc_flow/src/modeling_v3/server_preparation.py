"""CPU-only creation of exact Q4 bindings from executed engineering evidence."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .io import atomic_json, canonical_hash, sha256_file, source_identity


def record_q4_authorization(config, bindings, *, confirmation_text, reviewed_handoff, out):
    """Record an already received user confirmation; never request it or submit.

    The operator calls this only after presenting the bound command/timing
    handoff and receiving explicit authorization. The prepared stage remains
    unchanged; the execution uses a new stage and SOURCE_BINDINGS.json.
    """
    from .vlm_campaign import _bound_json

    stage = _bound_json(bindings["v3_stage_lock"])
    if (
        stage.get("phase") != "Q4"
        or stage.get("operator_authorization") != "PENDING_FIRST_GPU_CONFIRMATION"
        or stage.get("config_hash") != canonical_hash(config)
        or stage.get("source_hash") != source_identity()["sha256"]
    ):
        raise ValueError("authorization requires the unchanged prepared Q4 stage")
    if not isinstance(confirmation_text, str) or not confirmation_text.strip():
        raise ValueError("explicit user confirmation text is required")
    if sha256_file(reviewed_handoff["path"]) != reviewed_handoff["sha256"]:
        raise ValueError("reviewed command and timing handoff changed")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    authorized = {
        **stage,
        "operator_authorization": "CONFIRMED_FIRST_GPU_SUBMISSION",
        "first_gpu_confirmation": {
            "scope": "Q4",
            "confirmation_text": confirmation_text,
            "reviewed_handoff": reviewed_handoff,
            "prepared_stage": bindings["v3_stage_lock"],
        },
    }
    path = out / "Q4_AUTHORIZED_STAGE.json"
    atomic_json(path, authorized)
    resolved = {
        **bindings,
        "v3_stage_lock": {"path": str(path), "sha256": sha256_file(path)},
    }
    atomic_json(out / "SOURCE_BINDINGS.json", resolved)
    return resolved


def _verify_development_runs(config, roots):
    """Bind actual Q1/Q2 development originals without imposing a winning model."""
    from .cpu_campaign import _sources, _verify_complete

    if not roots or len(roots) != 2:
        raise ValueError("completed actual Q1 and Q2 development roots are required")
    current, result = _sources(), {}
    for root in map(Path, roots):
        manifest = _verify_complete(root)
        summary, binding = manifest["summary"], manifest["binding"]
        phase = summary.get("stage")
        if (
            phase not in {"Q1", "Q2"}
            or phase in result
            or summary.get("pilot") is not False
            or summary.get("scientific_status") != "DEVELOPMENT_ONLY"
            or binding.get("config_sha256") != canonical_hash(config)
            or binding.get("source_hashes") != current
        ):
            raise ValueError("actual development stage, full scope, or CPU source identity differs")
        if phase == "Q1":
            units = summary.get("units", [])
            if (
                not units
                or summary.get("unit_count") != len(units)
                or summary.get("repeat_count_per_unit") != config["cpu"]["repeat_measurements_dev"]
                or {row["n"] for row in units} != set(config["observation"]["total_draws_grid"])
                or {row["proposal"] for row in units} != set(config["observation"]["proposals"])
            ):
                raise ValueError("full Q1 observation grid and repetitions required")
        elif not summary.get("results") or summary.get("fit_count") != len(summary["results"]):
            raise ValueError("actual full Q2 fitted response results required")
        result[phase] = {
            "path": str((root / "COMPLETE.json").resolve()),
            "sha256": sha256_file(root / "COMPLETE.json"),
            "scientific_status": summary["scientific_status"],
            "comparison_success_required": False,
        }
    if set(result) != {"Q1", "Q2"}:
        raise ValueError("both actual Q1 and Q2 development stages required")
    return result


def prepare_gpu(config, bindings, cpu_results, out, *, development_roots=None):
    """Verify actual CPU test receipts and inherited originals; never load a model."""
    from .vlm_campaign import audit_v3_compatibility, build_q4_bridge_plan

    root, out = Path(cpu_results).resolve(), Path(out).resolve()
    result = json.loads((root / "result.json").read_text())
    if (
        result.get("execution_kind") != "SERVER_CPU"
        or not str(result.get("platform", "")).startswith("Linux-")
        or not str(result.get("slurm_job_id") or "").isdigit()
    ):
        raise ValueError("Q4 preparation requires executed server Slurm CPU acceptance")
    if result.get("exit_code") != 0 or not result.get("sources_unchanged_during_run"):
        raise ValueError("executed CPU acceptance did not pass unchanged sources")
    if result["counts"]["failed"] or result["counts"]["errors"]:
        raise ValueError("CPU acceptance still has failures or errors")
    required_targets = [
        "tests",
        "docs/modeling_v3/design/reference/test_math_contracts.py",
        "--deselect=tests/test_audit_r0_remaining.py::test_cross_split_passes_generated_dataset",
    ]
    if (
        result.get("acceptance_scope") != "FULL_REPOSITORY_EXCEPT_SEALED_DATA_TEST"
        or result.get("requested_targets") != required_targets
    ):
        raise ValueError(
            "full server repository acceptance required; selected test cases are insufficient"
        )
    if sha256_file(root / "pytest.log") != result["log_sha256"]:
        raise ValueError("CPU verification log was changed")
    if sha256_file(root / "junit.xml") != result.get("junit_sha256"):
        raise ValueError("CPU verification test-case evidence was changed")
    source = source_identity()
    for path, digest in source["files"].items():
        if result["source_and_test_hashes_before"].get(path) != digest:
            raise ValueError("CPU acceptance used a different V3 source snapshot")
    project = Path(__file__).resolve().parents[2]
    for directory in ("tests", "scripts"):
        for path in (project / directory).rglob("*.py"):
            relative = str(path.relative_to(project))
            if result["source_and_test_hashes_before"].get(relative) != sha256_file(path):
                raise ValueError("CPU acceptance test or script changed: " + relative)
    development = _verify_development_runs(config, development_roots)
    tests = list(ET.parse(root / "junit.xml").iter("testcase"))
    out.mkdir(parents=True, exist_ok=False)
    receipt_bindings = {}
    for phase, module in (("Q1", "test_observation_geometry"), ("Q2", "test_coverage_models")):
        cases = [case for case in tests if module in case.get("classname", "")]
        if not cases or any(
            len(case) and any(child.tag in {"failure", "error", "skipped"} for child in case)
            for case in cases
        ):
            raise ValueError("required Q1/Q2 technical tests are absent, skipped, or failing")
        receipt = {
            "status": "PASS",
            "scope": "EXECUTED_ENGINEERING_TESTS_ONLY",
            "phase": phase,
            "config_hash": canonical_hash(config),
            "source_hash": source["sha256"],
            "checks": [
                {"name": case.get("classname", "") + "::" + case.get("name", ""), "passed": True}
                for case in cases
            ],
            "cpu_results": {
                "path": str(root / "result.json"),
                "sha256": sha256_file(root / "result.json"),
            },
            "junit_sha256": sha256_file(root / "junit.xml"),
            "scientific_effectiveness": "NOT_EVALUATED",
            "development_original": development[phase],
        }
        path = out / (phase + "_TECHNICAL_RECEIPT.json")
        atomic_json(path, receipt)
        receipt_bindings[phase] = {"path": str(path), "sha256": sha256_file(path)}
    compatibility = audit_v3_compatibility(config, bindings)
    atomic_json(out / "V3_COMPATIBILITY.json", compatibility)
    plan = build_q4_bridge_plan(config, bindings, out=out)
    stage = {
        "schema": "ssvc-v3-execution-stage-1",
        "phase": "Q4",
        "config_hash": canonical_hash(config),
        "source_hash": source["sha256"],
        "technical_receipts": receipt_bindings,
        "development_originals": development,
        "operations": ["vlm-smoke"],
        "seeds": [],
        "workers": [0, 1],
        "max_concurrent_project_gpus": 2,
        "q4_plan_hash": plan["plan_hash"],
        "operator_authorization": "PENDING_FIRST_GPU_CONFIRMATION",
        "online_ssvc": False,
    }
    atomic_json(out / "Q4_STAGE_LOCK.json", stage)
    resolved = {
        **bindings,
        "v3_stage_lock": {
            "path": str(out / "Q4_STAGE_LOCK.json"),
            "sha256": sha256_file(out / "Q4_STAGE_LOCK.json"),
        },
        "q4_bridge_plan": {
            "path": str(out / "Q4_BRIDGE_PLAN.json"),
            "sha256": sha256_file(out / "Q4_BRIDGE_PLAN.json"),
        },
    }
    atomic_json(out / "SOURCE_BINDINGS.json", resolved)
    return {
        "status": "PREPARED_GPU_NOT_SUBMITTED",
        "out": str(out),
        "workers": 2,
        "source_hash": source["sha256"],
        "Q4_plan_hash": plan["plan_hash"],
    }
