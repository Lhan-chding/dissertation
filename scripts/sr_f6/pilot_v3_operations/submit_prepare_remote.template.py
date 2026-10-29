"""Prospective unique CPU submission; activated payload is injected locally.

No source import from the old followup package; full scientific verification is
performed only inside the frozen new producer's isolated historical bridge.
"""

import fcntl
import getpass
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from sr_f6 import phases
from sr_f6.common import now, write_once
from sr_f6.confirm_dispatch import _job_states
from sr_f6.interface_screen_dispatch import USER, capacity

# ACTIVATED_PAYLOAD_INSERTION_POINT
# An unactivated template raises KeyError here. Only the validated local
# assembler supplies actual inputs and source texts before execution.
inputs = globals()["inputs"]
program = globals()["program"]
wrapper = globals()["wrapper"]

project = Path("/projects/_ssd/varunssd/louis-ssvc")
ops = phases._path(project / "sr_f6_operations_20261027/PILOT_V3_EXECUTION_20261029")
root = phases._root(project / "sr_f6_semantic_factorial_20261027")
revision = "763172582c4e16a0d9583deecbbe72b0817a4a37"
code = phases._deployment(project / "sr_f6_code_7631725", revision)
followup = phases._path(project / "sr_f6_confirm_followup_v2_20261029")
followup_ops = phases._path(
    project / "sr_f6_operations_20261027/CONFIRM_FOLLOWUP_V2_EXECUTION_20261029"
)
assert getpass.getuser() == USER
assert Path(phases.__file__).resolve() == code / "src/sr_f6/phases.py"


def actual_json(path):
    return json.loads(phases._path(path).read_text())


for reference in (inputs["followup_registration_ref"], inputs["confirmation_receipt_ref"]):
    assert (
        hashlib.sha256(phases._path(reference["path"]).read_bytes()).hexdigest()
        == reference["sha256"]
    )
confirmation = actual_json(inputs["confirmation_receipt_ref"]["path"])
assert confirmation["status"] == "CONFIRMATION_GATES_PASS"
assert confirmation["gate_status"] == "CONFIRMATION_FACTS_SEALED"
assert all(
    confirmation[k] is False
    for k in ("training_allowed", "generation_allowed", "evaluation_allowed")
)
admission_submission = actual_json(followup_ops / "CPU_ADMISSION_SUBMISSION.json")
t03_submission = actual_json(followup / "T03/SUBMISSIONS/t03.result.json")
assert admission_submission["status"] == t03_submission["status"] == "SUBMITTED"
assert t03_submission["job_id"] == "233723"
assert admission_submission["job_id"] == inputs["confirmation_admission_job_id"]
jobs = {admission_submission["job_id"], t03_submission["job_id"]}
assert _job_states(jobs) == {job: ("COMPLETED", "0:0") for job in jobs}
# Accounting can become terminal shortly before queue cleanup; never treat a
# still-listed CPU reservation as zero or resubmit while the old job is exiting.
live = subprocess.check_output(["squeue", "-h", "-u", USER, "-o", "%i"], text=True, timeout=15)
assert not (jobs & set(live.split())), "Wait for completed prerequisite jobs to leave queue"
assert not list((followup / "T03/FAILURES").iterdir())
assert {p.name for p in (followup / "T03/RUNS").iterdir()} == {
    "LH-SRF6-T03-233723-" + parent for parent in ("b0", "b1", "b2")
}
for parent in ("b0", "b1", "b2"):
    result = actual_json(followup / ("T03/RUNS/LH-SRF6-T03-233723-" + parent + "/T03/RESULT.json"))
    assert result["status"] == "PASS"
    assert result["generated_responses"] == result["optimizer_updates"] == 0
for relative in ("PHASES/PILOT_V3", "PHASE_RUNS/PILOT_V3"):
    path = phases._path(root / relative, exists=False)
    assert not path.exists(), "Preserve previous or partial preparation"
assert inputs["actual_source_revision"] == revision and inputs["root"] == str(root)
assert inputs["source_deployment"] == str(code)
compile(program, "prepare_pilot.py", "exec")
syntax = subprocess.run(["bash", "-n"], input=wrapper, capture_output=True, text=True)
assert syntax.returncode == 0, syntax.stderr
input_bytes = (json.dumps(inputs, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
files = {
    "prepare_pilot.py": program.encode(),
    "prepare_pilot.sbatch": wrapper.encode(),
    "PREPARE_INPUTS.json": input_bytes,
}
with phases._path(ops / "CPU_PREPARE.lock", exists=False).open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for name in (
        *files,
        "CPU_PREPARE_ATTEMPT.json",
        "CPU_PREPARE_SUBMISSION.json",
        "CPU_PREPARE_STARTED.json",
        "CPU_PREPARE_FAILURE.json",
        "PILOT_PREPARE_RESULT.json",
    ):
        assert not phases._path(ops / name, exists=False).exists(), "No retry or overwrite"
    snapshot = capacity()
    if (
        snapshot["submitted_by_user"] >= snapshot["submit_limit"]
        or snapshot["reserved_gpus"] > snapshot["gpu_limit"]
    ):
        print(json.dumps(dict(status="NO_LEGAL_CPU_SUBMIT_SLOT", capacity=snapshot)))
        raise SystemExit(0)
    argv = [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--chdir=" + str(code),
        "--output=" + str(ops / "prepare-%j.log"),
        str(ops / "prepare_pilot.sbatch"),
    ]
    write_once(
        ops / "CPU_PREPARE_ATTEMPT.json",
        dict(
            at=now(),
            argv=argv,
            capacity=snapshot,
            requested_gpus=0,
            producer_revision=revision,
            inputs_sha256=hashlib.sha256(input_bytes).hexdigest(),
            program_sha256=hashlib.sha256(program.encode()).hexdigest(),
            wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),
            scope=(
                "Once-only new-source V3 CPU registration/confirmation/overlay/admission/plan. "
                "No model calls or GPU submissions."
            ),
        ),
    )
    for name, payload in files.items():
        fd = os.open(
            phases._path(ops / name, exists=False),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)
    job = result.stdout.strip().split(";", 1)[0]
    valid = result.returncode == 0 and re.fullmatch("[0-9]+", job)
    receipt = dict(
        at=now(),
        status="SUBMITTED" if valid else "SUBMISSION_FAILED_OR_UNKNOWN",
        job_id=job if valid else None,
        requested_gpus=0,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        argv=argv,
    )
    write_once(ops / "CPU_PREPARE_SUBMISSION.json", receipt)
    print(json.dumps(receipt))
    if not valid:
        raise SystemExit(1)
