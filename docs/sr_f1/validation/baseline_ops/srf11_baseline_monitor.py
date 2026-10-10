"""Read-only ENGINE handoff and parallel-baseline status, without TEST outputs."""

import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010")
INC = ROOT / "technical_incidents/baseline_parallel_20261010"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    return dict(
        command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
    )


def main():
    state = read(ROOT / "orchestration/STATE.json")
    result = dict(
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_root=str(ROOT),
        phase=state["phase"],
        test_sealed=state["test_sealed"],
        technical_blockers=state.get("technical_blockers", []),
        tasks={
            name: dict(
                status=task["status"],
                attempts=len(task["attempts"]),
                latest=task["attempts"][-1] if task["attempts"] else None,
            )
            for name, task in state["tasks"].items()
        },
        receipts={
            name: read(INC / name)
            for name in [
                "STAGING_VERIFIED.json",
                "CANDIDATE_VALIDATION.json",
                "HANDOFF_SUBMISSION.json",
                "CPU_TERMINAL.json",
                "activation/SOURCE_ACTIVATED.json",
            ]
        },
    )
    result["freeze_sha256"] = sha(ROOT / "EXECUTION_FREEZE.json")
    result["registration_sha256"] = sha(ROOT / "orchestration/REGISTRATION.json")
    result["parallel_repair_sha256"] = sha(ROOT / "BASELINE_PARALLEL_REPAIR.json")
    result["baseline_allocation"] = read(ROOT / "BASELINE_PARALLEL_ALLOCATION.json")
    source = read(ROOT / "code/SOURCE_DEPLOYMENT.json")
    result["production_source_commit"] = source["source_commit"]
    result["production_source_files_verified"] = all(
        sha(ROOT / "code" / name) == expected
        for name, expected in source["source_file_hashes"].items()
    )
    candidate = ROOT / "code_baseline_parallel_candidate_20261010/SOURCE_DEPLOYMENT.json"
    result["candidate_source_commit"] = (
        read(candidate)["source_commit"] if candidate.exists() else None
    )
    result["engine_tracks"] = {}
    for track in ("continuous", "split"):
        folder = ROOT / "engineering/engine/natural" / track
        result["engine_tracks"][track] = dict(
            latest=read(folder / "checkpoints/LATEST.json"),
            raw_records=len(list((folder / "rollouts").glob("*.json"))),
        )
    result["engine_acceptance_exists"] = (
        ROOT / "ENGINE_PROBABILITY_GRADIENT_RESUME.json"
    ).is_file()
    result["baseline_shards"] = []
    for path in sorted((ROOT / "raw/evaluation").glob("SRF1_COMMON_START_step0_baseline*.jsonl")):
        with path.open() as stream:
            count = sum(1 for line in stream if line.strip())
        result["baseline_shards"].append(
            dict(path=str(path.relative_to(ROOT)), records=count, bytes=path.stat().st_size)
        )
    result["baseline_coverage"] = [
        read(path)
        for path in sorted((ROOT / "evaluation/baseline_parallel/attempts").glob("*/COVERAGE.json"))
    ]
    jobs = {
        attempt["job_id"]
        for task in state["tasks"].values()
        for attempt in task["attempts"]
        if attempt.get("job_id") and "accounting" not in attempt
    }
    handoff = read(INC / "HANDOFF_SUBMISSION.json")
    if handoff and handoff["returncode"] == 0:
        jobs.add(handoff["stdout"].strip().split(";")[0])
    assert all(re.fullmatch("[0-9]+", job) for job in jobs)
    result["queue"] = run(
        ["squeue", "--noheader", "--user=varun024", "--format=%i|%j|%T|%q|%R|%M|%b"]
    )
    result["accounting"] = (
        run(
            [
                "sacct",
                "-X",
                "-j",
                ",".join(sorted(jobs)),
                "--format=JobIDRaw,JobName,State,ExitCode,AllocTRES",
                "-n",
                "-P",
            ]
        )
        if jobs
        else None
    )
    result["job_details"] = {
        job: run(["scontrol", "show", "job", job, "--oneliner"]) for job in sorted(jobs)
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
