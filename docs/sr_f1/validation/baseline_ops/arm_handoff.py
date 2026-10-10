"""Arm the tested CPU-only handoff; this script never operates a GPU job."""

import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010")
INC = ROOT / "technical_incidents/baseline_parallel_20261010"
CANDIDATE = ROOT / "code_baseline_parallel_candidate_20261010"
OLD_CPU = "196501"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=45)
    return dict(
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def fields(observation):
    assert observation["returncode"] == 0, observation
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", observation["stdout"]))


def main():
    assert ROOT.resolve() == ROOT and not (ROOT / "STOP").exists()
    assert not (INC / "HANDOFF_SUBMISSION_INTENT.json").exists()
    inventory = read(INC / "inventory.json")
    for name, expected in inventory["files"].items():
        assert sha(CANDIDATE / name) == expected, name
    deployment = read(CANDIDATE / "SOURCE_DEPLOYMENT.json")
    assert deployment["source_commit"] == inventory["source_commit"]
    test_submission = read(INC / "CANDIDATE_CPU_SUBMISSION.json")
    assert test_submission["returncode"] == 0
    test_job = test_submission["stdout"].strip().split(";")[0]
    assert re.fullmatch("[0-9]+", test_job)
    accounting = run(
        ["sacct", "--allocations", "-j", test_job, "--format=JobIDRaw,State,ExitCode", "-n", "-P"]
    )
    assert accounting["returncode"] == 0
    rows = [line.split("|")[:3] for line in accounting["stdout"].splitlines() if line.strip()]
    assert rows == [[test_job, "COMPLETED", "0:0"]], accounting
    source_check = read(INC / "CANDIDATE_SOURCE_CHECK.json")
    assert source_check["source"] == deployment and source_check["package"]["status"] == "PASS"
    test_log = (INC / "CANDIDATE_SERVER_TESTS.log").read_text()
    matches = re.findall(r"(\d+) passed, (?:(\d+) skipped, )?(\d+) subtests passed", test_log)
    assert matches and int(matches[-1][0]) >= 747, test_log[-2000:]
    validation = dict(
        status="PASS",
        source_commit=deployment["source_commit"],
        source_tree_sha256=deployment["source_tree_sha256"],
        cpu_job_id=test_job,
        tests_passed=int(matches[-1][0]),
        skipped=int(matches[-1][1] or 0),
        subtests_passed=int(matches[-1][2]),
        accounting=accounting,
        validation_artifact_hashes={
            name: sha(INC / name)
            for name in [
                "STAGING_VERIFIED.json",
                "CANDIDATE_SERVER_TESTS.log",
                "CANDIDATE_SOURCE_CHECK.json",
                "CANDIDATE_CPU_SUBMISSION.json",
            ]
        },
    )
    save(INC / "CANDIDATE_VALIDATION.json", validation)
    detail = run(["scontrol", "show", "job", OLD_CPU, "--oneliner"])
    old = fields(detail)
    assert old["JobId"] == OLD_CPU and old["JobState"] == "RUNNING"
    assert old["JobName"] == "srf11-controller-multigpu" and old["Account"] == "rose"
    assert old["UserId"].split("(")[0] == "varun024"
    assert all("gres/gpu" not in old[key] for key in ["ReqTRES", "AllocTRES"])
    identity = {
        key: old.get(key, "")
        for key in ["JobName", "UserId", "Account", "QOS", "Command", "Comment", "StdOut"]
    }
    identity["UserId"] = identity["UserId"].split("(")[0]
    if identity["Comment"] == "(null)":
        identity["Comment"] = ""
    authorization = dict(
        status="AUTHORIZED_BASELINE_PARALLEL_HANDOFF",
        run_root=str(ROOT),
        original_execution_freeze_sha256=sha(ROOT / "EXECUTION_FREEZE.json"),
        previous_engine_multigpu_repair_sha256=sha(ROOT / "ENGINE_MULTIGPU_REPAIR.json"),
        old_controller_job_id=OLD_CPU,
        old_controller_identity=identity,
        candidate_source_deployment_sha256=sha(CANDIDATE / "SOURCE_DEPLOYMENT.json"),
        candidate_validation_sha256=sha(INC / "CANDIDATE_VALIDATION.json"),
        authorized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorized_user_message="后面的基线测量也要用多张卡并行提速，提前准备好，ENGINE通过后自动接着跑。仍然用老师的qos，最多五张卡，不要动无关作业。",  # noqa: RUF001
    )
    save(INC / "HANDOFF_AUTHORIZATION.json", authorization)
    save(INC / "CPU_IDENTITY_BEFORE_ARMING.json", detail)
    # Stable entry point survives candidate->code promotion and Slurm requeue.
    watcher = INC / "baseline_handoff.py"
    with watcher.open("xb") as stream:
        stream.write((CANDIDATE / "scripts/sr_f1/baseline_handoff.py").read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    assert sha(watcher) == deployment["source_file_hashes"]["scripts/sr_f1/baseline_handoff.py"]
    python = read(ROOT / "orchestration/REGISTRATION.json")["python"]
    script = INC / "handoff.sbatch"
    with script.open("x") as stream:
        stream.write(f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=4320
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
exec {python} {watcher} \\
 --plan {ROOT}/config/SR_F1_1.json --run-root {ROOT} \\
 --candidate-root {CANDIDATE} --python {python} --old-controller-id {OLD_CPU}
""")
    command = [
        "sbatch",
        "--hold",
        "--parsable",
        "--account=rose",
        "--qos=override-limits-but-killable",
        "--partition=cluster02",
        "--job-name=srf11-baseline-handoff",
        "--comment=srf11-baseline-parallel-20261010",
        "--output=" + str(INC / "handoff_%j.log"),
        str(script),
    ]
    save(
        INC / "HANDOFF_SUBMISSION_INTENT.json",
        dict(
            command=command,
            authorization_sha256=sha(INC / "HANDOFF_AUTHORIZATION.json"),
            watcher_sha256=sha(watcher),
            script_sha256=sha(script),
        ),
    )
    result = run(command)
    save(INC / "HANDOFF_SUBMISSION.json", result)
    assert result["returncode"] == 0, result
    job = result["stdout"].strip().split(";")[0]
    assert re.fullmatch("[0-9]+", job)
    observed = run(["scontrol", "show", "job", job, "--oneliner"])
    actual = fields(observed)
    assert actual["JobId"] == job and actual["JobName"] == "srf11-baseline-handoff"
    assert actual["UserId"].split("(")[0] == "varun024" and actual["Account"] == "rose"
    assert actual["QOS"] == "override-limits-but-killable"
    assert actual["Comment"] == "srf11-baseline-parallel-20261010"
    assert actual["Command"] == str(script) and actual["JobState"] == "PENDING"
    assert "gres/gpu" not in actual["ReqTRES"]
    save(INC / "HANDOFF_RELEASE_PREFLIGHT.json", observed)
    released = run(["scontrol", "release", job])
    save(INC / "HANDOFF_RELEASE.json", released)
    assert released["returncode"] == 0, released
    print(
        json.dumps(
            dict(
                status="TESTED_CPU_HANDOFF_ARMED",
                job_id=job,
                source_commit=deployment["source_commit"],
                baseline_started=False,
                engine_job_unchanged=True,
            )
        )
    )


if __name__ == "__main__":
    main()
