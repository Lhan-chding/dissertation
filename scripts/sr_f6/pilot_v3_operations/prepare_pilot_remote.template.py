"""One CPU preparation over frozen production APIs; never submits a GPU job.

Activation requires an actual completed confirmation receipt. No placeholder
permission, source patch, registration retry, or scientific threshold lives here.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from sr_f6 import (
    historical_confirmation,
    phases,
    pilot_jobs,
    pilot_v3_admission,
    pilot_v3_inputs,
    pilot_v3_registration,
)
from sr_f6.common import now, write_once

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
CODE = PROJECT / "sr_f6_code_7631725"
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_V3_EXECUTION_20261029"
REVISION = "763172582c4e16a0d9583deecbbe72b0817a4a37"
stage = "preconditions"


def require(value, message):
    if not value:
        raise PermissionError(message)


def read_bound(ref):
    path = phases._path(ref["path"])
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "Changed input: " + str(path))
    return json.loads(raw)


def record(name, value):
    write_once(phases._path(OPS / name, exists=False), value)


def main():
    global stage
    require(os.environ.get("SLURM_JOB_NAME") == "LH-SRF6-PilotV3-Prepare", "Wrong CPU job")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only preparation required")
    require(os.environ.get("SLURM_JOB_ID", "").isdigit(), "Actual scheduler job required")
    for path in (PROJECT, ROOT, CODE, OPS):
        phases._path(path)
    intent = json.loads(phases._path(OPS / "CPU_PREPARE_ATTEMPT.json").read_text())
    inputs_path = phases._path(OPS / "PREPARE_INPUTS.json")
    require(
        hashlib.sha256(inputs_path.read_bytes()).hexdigest() == intent["inputs_sha256"],
        "Preparation inputs differ from the submission intent",
    )
    require(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == intent["program_sha256"],
        "Operations program differs from submission intent",
    )
    inputs = json.loads(inputs_path.read_text())
    require(
        inputs["actual_source_revision"] == REVISION
        and inputs["source_deployment"] == str(CODE)
        and inputs["root"] == str(ROOT),
        "Wrong frozen new producer",
    )
    for module in (
        phases,
        historical_confirmation,
        pilot_jobs,
        pilot_v3_admission,
        pilot_v3_inputs,
        pilot_v3_registration,
    ):
        require(
            Path(module.__file__).resolve()
            == CODE / "src" / (module.__name__.replace(".", "/") + ".py"),
            "Wrong loaded producer",
        )
    for relative in ("PHASES/PILOT_V3", "PHASE_RUNS/PILOT_V3"):
        path = phases._path(ROOT / relative, exists=False)
        require(not path.exists(), "Preserve previous or partial V3 preparation")
    confirmation = read_bound(inputs["confirmation_receipt_ref"])
    require(
        confirmation["status"] == "CONFIRMATION_GATES_PASS"
        and confirmation["gate_status"] == "CONFIRMATION_FACTS_SEALED"
        and all(
            confirmation[k] is False
            for k in ("generation_allowed", "training_allowed", "evaluation_allowed")
        ),
        "Actual complete confirmation must precede preparation",
    )
    record(
        "CPU_PREPARE_STARTED.json",
        dict(
            at=now(),
            job_id=os.environ["SLURM_JOB_ID"],
            producer_revision=REVISION,
            new_model_calls=0,
        ),
    )
    stage = "actual_deployment_native_cpu_regression"
    cache = phases._path(Path(os.environ["TMPDIR"])).parent
    basetemp = phases._path(cache / "native-tests", exists=False)
    require(not basetemp.exists(), "Preserve prior native regression output")
    tests = [
        "tests/sr_f6/test_phases.py",
        "tests/sr_f6/test_pilot_v3_inputs.py",
        "tests/sr_f6/test_pilot_v3_registration.py",
        "tests/sr_f6/test_pilot_v3_kernel_binding.py",
        "tests/sr_f6/test_transactions.py::test_crash_boundaries_recover_exact_baseline_and_single_logical_commit",
        "tests/sr_f6/test_transactions.py::test_second_process_cannot_acquire_the_same_run_lock",
    ]
    argv = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        "/dev/null",
        "--rootdir=" + str(CODE),
        "--confcutdir=" + str(CODE),
        "--basetemp=" + str(basetemp),
        *tests,
    ]
    with phases._path(OPS / "NATIVE_CPU_REGRESSION.log", exists=False).open("xb") as stream:
        completed = subprocess.run(
            argv, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT, timeout=1800, check=False
        )
    record(
        "NATIVE_CPU_REGRESSION.json",
        dict(
            at=now(),
            argv=argv,
            returncode=completed.returncode,
            scope=(
                "Actual server CPU synthetic regressions and shared filesystem publish/lock checks"
            ),
            scientific_pass=False,
            new_model_calls=0,
        ),
    )
    require(completed.returncode == 0, "Native CPU regression failed; preserve all outputs")
    stage = "complete_source_delta_and_original_provenance"
    before = read_bound(
        dict(
            path=str(ROOT / "SOURCE_MANIFEST.json"),
            sha256=inputs["origin"]["source_manifest"]["sha256"],
        )
    )
    manifest = phases.build_manifest(CODE, revision=REVISION)
    require(
        manifest == inputs["expected_source_manifest"], "Actual deployed source inventory differs"
    )
    require(
        phases.source_delta(before, manifest, reasons=inputs["source_change_reasons"])
        == inputs["expected_delta"],
        "Source delta differs from the prospective declaration",
    )
    context = phases.register_phase(
        ROOT,
        phase="PILOT_V3",
        deployment=CODE,
        revision=REVISION,
        origin_deployment=inputs["origin"]["deployment"],
        upstream=inputs["upstream"],
        expected_delta=inputs["expected_delta"],
    )
    require(context["status"] == "BOUND_NOT_AUTHORIZED", "Source registration is not permission")
    record("SOURCE_REGISTERED.json", dict(at=now(), phase_context=context))
    stage = "isolated_actual_confirmation_binding"
    binding = historical_confirmation.freeze_confirmation_binding(
        ROOT,
        phase_context=context,
        followup_registration_ref=inputs["followup_registration_ref"],
        confirmation_receipt_ref=inputs["confirmation_receipt_ref"],
    )
    record(
        "CONFIRMATION_BOUND.json",
        dict(
            at=now(),
            binding_ref=binding["binding_ref"],
            facts_sha256=binding["verification"]["facts_sha256"],
        ),
    )
    stage = "fixed_public_development_overlay"
    overlay = pilot_v3_inputs.freeze_overlay(ROOT, phase_context=context)
    record("INPUT_OVERLAY_FROZEN.json", dict(at=now(), manifest_ref=overlay["manifest_ref"]))
    stage = "complete_nine_path_registration"
    registered = pilot_v3_registration.freeze_pilot_v3_registration(
        ROOT,
        phase_context=context,
        parent_references=inputs["parent_references"],
        overlay_ref=overlay["manifest_ref"],
        confirmation_binding_ref=binding["binding_ref"],
    )
    require(
        registered["status"] == "FROZEN_NOT_AUTHORIZED"
        and registered["generation_authorized"] is False
        and registered["training_authorized"] is False,
        "Plan registration is not permission",
    )
    pins = dict(
        phase_identity_sha256=context["identity_sha256"],
        pilot_spec_sha256=registered["spec_sha256"],
    )
    record(
        "PILOT_REGISTRATION_FROZEN.json",
        dict(at=now(), **pins, budgets=registered["spec"]["budgets"]),
    )
    stage = "actual_pilot_admission_and_numerical_correspondence"
    admitted = pilot_v3_admission.seal_pilot_v3_admission(ROOT, **pins)
    if admitted["status"] != "VERIFIED_PILOT_ADMISSION":
        record(
            "PILOT_PREPARE_RESULT.json",
            dict(
                at=now(),
                status="ADMISSION_NOT_GRANTED",
                stage=stage,
                result=admitted,
                new_model_calls=0,
                optimizer_updates=0,
            ),
        )
        raise PermissionError("Original pilot admission refused; do not retry or expand")
    require(
        admitted["formal_authorized"] is False and admitted["test_authorized"] is False,
        "Later stages must remain closed",
    )
    stage = "fixed_two_worker_operations_plan"
    plan_ref = pilot_jobs.freeze_plan(
        ROOT, phase="PILOT_V3", admission_ref=admitted["admission_ref"], **pins
    )
    record(
        "PILOT_PREPARE_RESULT.json",
        dict(
            at=now(),
            status="READY_FOR_REGISTERED_PILOT_ONLY",
            producer_revision=REVISION,
            **pins,
            admission_ref=admitted["admission_ref"],
            jobs_plan_ref=plan_ref,
            overlay_ref=overlay["manifest_ref"],
            confirmation_binding_ref=binding["binding_ref"],
            new_model_calls=0,
            optimizer_updates=0,
            gpu_jobs_submitted=0,
            formal_authorized=False,
            test_authorized=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="READY_FOR_REGISTERED_PILOT_ONLY",
                **pins,
                plan_ref=plan_ref,
                new_model_calls=0,
                optimizer_updates=0,
                gpu_jobs_submitted=0,
            )
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record(
            "CPU_PREPARE_FAILURE.json",
            dict(
                at=now(),
                status="STOP_PRESERVE_NO_AUTO_RETRY",
                stage=stage,
                error=repr(exc),
                new_model_calls=0,
                optimizer_updates=0,
            ),
        )
        raise
