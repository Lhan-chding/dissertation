"""One immutable B allocation with two independent lanes; no evaluation entrypoint."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_ROUND_B_20261031"
CODE = PROJECT / "sr_f6_code_b7e7b06"
PYTHON = PROJECT / "envs/ssvc-py312/bin/python3.12"
V8 = PROJECT / "sr_f6_memory_resume_code_e4d2b0c"
V8_REVISION = "e4d2b0ca4155b489a21f0b3cd2696b6860df7626"
V8_BINDING = "0b3df1de667a6b172e40e03646eeec329f4c9e1d34862d5ee4bc4e335458618a"
SCRIPT = "scripts/sr_f6_round_b_ops/operations.py"
LAUNCHER = "scripts/sr_f6_round_b_ops/launch.sbatch"
JOB_NAME = "LH-SRF6-PilotV3-B"
USER = "varun024"
QOS = "soujanya-poria-startfund-2026-03"
JOBS = ROOT / "PHASE_RUNS/PILOT_V3/JOBS"
QUEUES = [["PRIMARY-b2-U10", "PRIMARY-b2-U11", "PRIMARY-b2-C_SHUF"], ["RECOVERY-b0-U11"]]


def require(value, message):
    if not value:
        raise PermissionError(message)


def frozen_resume():
    """Load helpers at their real immutable origin; never impersonate a prior source."""
    target = V8 / "RESUME_SOURCE.json"
    require(
        target.resolve(strict=True) == target and target.stat().st_size <= 2**20,
        "Canonical frozen manifest required",
    )
    require(
        hashlib.sha256(target.read_bytes()).hexdigest() == V8_BINDING, "Frozen v8 manifest changed"
    )
    manifest = json.loads(target.read_bytes())
    name = "scripts/sr_f6_memory_resume_ops/operations.py"
    path = V8 / name
    require(
        path.resolve(strict=True) == path
        and path.stat().st_size <= 2**20
        and hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name]["sha256"],
        "Frozen v8 operator bytes changed",
    )
    spec = importlib.util.spec_from_file_location("srf6_b_frozen_resume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.identity(V8_REVISION, str(V8), V8_BINDING)
    return module


def contract_module(actual):
    previous = frozen_resume()
    sys.path[:0] = [
        str(CODE / "src"),
        str(previous.FULL_DEPLOYMENT / "src"),
        str(V8 / "src"),
        str(Path(actual["deployment"]) / "src"),
    ]
    module = importlib.import_module("sr_f6_round_b.contract")
    require(
        Path(module.__file__).resolve()
        == Path(actual["deployment"]) / "src/sr_f6_round_b/contract.py",
        "Wrong B contract origin",
    )
    return module


def identity(revision, deployment, binding_sha256):
    previous = frozen_resume()
    require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "Actual new committed revision required",
    )
    target = PROJECT / ("sr_f6_round_b_code_" + revision[:7])
    require(
        Path(deployment) == target and Path(__file__).resolve() == target / SCRIPT,
        "Wrong B operator origin",
    )
    require(previous.sha(target / "ROUND_B_SOURCE.json") == binding_sha256, "B manifest changed")
    actual = dict(revision=revision, deployment=str(target), binding_sha256=binding_sha256)
    manifest = previous.read(target / "ROUND_B_SOURCE.json")
    source = dict(
        revision=revision,
        deployment=str(target),
        files=[dict(path=name, sha256=pin["sha256"]) for name, pin in manifest["files"].items()],
    )
    contract_module(actual).sources(source)
    return actual


def metadata(actual, path, sha256):
    previous = frozen_resume()
    require(Path(path) == OPS / "CONTRACT.json", "Final unique B contract required")
    value = contract_module(actual).validate_metadata(path, sha256)
    source = value["execution_source"]
    require(
        source["revision"] == actual["revision"] and source["deployment"] == actual["deployment"],
        "B contract source differs",
    )
    require(
        previous.sha(Path(actual["deployment"]) / "ROUND_B_SOURCE.json")
        == actual["binding_sha256"],
        "B binding changed",
    )
    return value


def terminal(old, job="235953"):
    rows = [
        line.split("|")
        for line in old.query(
            ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"]
        ).splitlines()
    ]
    main = [row for row in rows if len(row) >= 3 and row[0] == job]
    require(
        len(main) == 1 and main[0][1:3] == ["COMPLETED", "0:0"], "A must be actual COMPLETED 0:0"
    )
    queue = old.query(["squeue", "-h", "-u", USER, "-o", "%i"])
    require(job not in queue.split(), "Completed A job still queued")
    return dict(job_id=job, state="COMPLETED", exit_code="0:0", queue_gone=True)


def untouched_b(previous):
    """Reject any existing B intent or output, including incomplete or unknown state."""
    target = ROOT / "PHASE_RUNS/PILOT_V3/RECOVERY/b0/r0/U11"
    previous.canonical(target, exists=False)
    require(
        not target.exists() and not any(target.parent.glob(".recovery-staging-*")),
        "Prior recovery or staging forbids retry",
    )
    for worker, queue in enumerate(QUEUES):
        require(not (JOBS / f"WORKERS/B/{worker}/ATTEMPTS").exists(), "Prior B worker attempt")
        for task in queue:
            folder = JOBS / "TASKS" / task
            if folder.exists():
                previous.canonical(folder)
                require(
                    not any(p.name != ".task.lock" for p in folder.iterdir()),
                    "Prior B task intent/output",
                )
            if worker == 0:
                arm = task.removeprefix("PRIMARY-b2-")
                directory = ROOT / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/b2/r0" / arm
                require(directory.is_dir(), "Original b2 registration directory required")
                if directory.exists():
                    previous.canonical(directory)
                    require(
                        {"RUN_IDENTITY.json", "ROLLOUT_REGISTRATION.json"}
                        <= {p.name for p in directory.iterdir()},
                        "Both b2 registration files required",
                    )
                    for path in directory.iterdir():
                        previous.canonical(path)
                        require(
                            path.is_file()
                            and path.name
                            in {"RUN_IDENTITY.json", "ROLLOUT_REGISTRATION.json", ".run.lock"}
                            and (path.name != ".run.lock" or path.stat().st_size == 0),
                            "Prior b2 model output",
                        )
    # The original operator has no B submit entry, but preserve any hand-created
    # B intent/partial records instead of assuming their absence from the queue.
    original_ops = PROJECT / "sr_f6_operations_20261027/PILOT_V3_GPU_EXECUTION_20261029"
    if original_ops.exists():
        require(not any(original_ops.glob("*B_*.json")), "Original B intent requires review")


@contextmanager
def transition_locks(previous, legacy):
    with ExitStack() as stack:
        stack.enter_context(previous.transition_locks(legacy))
        stack.enter_context(legacy.lock(previous.OPS / "resume.lock", existing=True))
        stack.enter_context(legacy.lock(previous.OPS / "resume_LAUNCH.lock", existing=True))
        lock_paths = [JOBS / f"WORKERS/B/{worker}/.worker.lock" for worker in (0, 1)]
        lock_paths += [JOBS / "TASKS" / task / ".task.lock" for queue in QUEUES for task in queue]
        for path in lock_paths:
            previous.canonical(path.parent, exists=False).mkdir(parents=True, exist_ok=True)
            stack.enter_context(legacy.lock(path))
        yield


def argv_for(actual, path, sha256):
    exports = dict(
        PATH="/usr/bin:/bin",
        SRF6_B_REVISION=actual["revision"],
        SRF6_B_DEPLOYMENT=actual["deployment"],
        SRF6_B_BINDING=actual["binding_sha256"],
        SRF6_B_CONTRACT=str(path),
        SRF6_B_CONTRACT_SHA=sha256,
    )
    require(
        all("," not in value and "\n" not in value for value in exports.values()), "Unsafe export"
    )
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--job-name=" + JOB_NAME,
        "--account=rose",
        "--partition=cluster02",
        "--qos=" + QOS,
        "--gres=gpu:pro6000:2",
        "--cpus-per-task=8",
        "--mem=66G",
        "--time=23:00:00",
        "--nodes=1",
        "--chdir=" + str(CODE),
        "--output=" + str(OPS / "round_b-%j.log"),
        "--export=" + ",".join(k + "=" + v for k, v in exports.items()),
        str(Path(actual["deployment"]) / LAUNCHER),
    ]


def submit(actual, path, sha256):
    require(getpass.getuser() == USER, "Wrong scheduler account")
    previous = frozen_resume()
    legacy, _ = previous.frozen_v4()
    old = legacy.original()
    old.clean_submission_environment()
    metadata(actual, path, sha256)
    with transition_locks(previous, legacy), legacy.lock(OPS / "round_b.lock"):
        prior = terminal(old)
        untouched_b(previous)
        require(
            not any(OPS.glob("round_b_*.json"))
            and not any(OPS.glob(".pending-*"))
            and not (OPS / "EXECUTIONS").exists()
            and not (OPS / "LANES").exists(),
            "Any prior/partial B intent forbids retry",
        )
        scheduler, _ = old.scheduler(old.metadata())
        capacity = scheduler.capacity(query=old.query)
        require(
            capacity["gpu_limit"] == 8 and capacity["submit_limit"] == 5, "Wrong resource limits"
        )
        if capacity["reserved_gpus"] + 2 > 8 or capacity["submitted_by_user"] + 1 > 5:
            return dict(status="NO_LEGAL_CAPACITY", capacity=capacity)
        argv = argv_for(actual, path, sha256)
        old.write_once(
            OPS / "round_b_ATTEMPT.json",
            dict(
                status="SUBMISSION_INTENT_NOT_COMPLETION",
                at=old.now(),
                source=actual,
                contract=dict(path=str(path), sha256=sha256),
                argv=argv,
                capacity=capacity,
                predecessor=prior,
            ),
        )
        try:
            process = subprocess.run(argv, capture_output=True, text=True, timeout=25, check=False)
            job = process.stdout.strip().split(";", 1)[0]
            ok = process.returncode == 0 and re.fullmatch(r"[0-9]+", job) is not None
            result = dict(
                status="SUBMITTED" if ok else "FAILED_OR_UNKNOWN",
                job_id=job if ok else None,
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            result = dict(status="SUBMISSION_UNKNOWN", job_id=None, error=str(exc))
        result.update(
            source=actual, at=old.now(), attempt_sha256=previous.sha(OPS / "round_b_ATTEMPT.json")
        )
        old.write_once(OPS / "round_b_SUBMISSION.json", result)
        return result


def allocation(old, job):
    fields = dict(
        re.findall(
            r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)",
            old.query(["scontrol", "show", "job", "-o", job]),
        )
    )
    require(
        fields.get("JobId") == job
        and fields.get("JobName") == JOB_NAME
        and fields.get("JobState") == "RUNNING"
        and fields.get("QOS") == QOS
        and fields.get("Partition") == "cluster02"
        and fields.get("Account") == "rose"
        and fields.get("UserId", "").startswith(USER + "(")
        and fields.get("Requeue") == "0"
        and fields.get("TimeLimit") == "23:00:00",
        "Wrong actual B allocation",
    )
    for key in ("ReqTRES", "AllocTRES"):
        tres = dict(item.split("=", 1) for item in fields.get(key, "").split(",") if "=" in item)
        require(
            tres.get("gres/gpu:pro6000") == "2"
            and tres.get("gres/gpu", "2") == "2"
            and tres.get("cpu") == "8"
            and tres.get("mem") in ("66G", "67584M"),
            "Wrong actual B resources",
        )
    return fields


def arguments(actual, path, sha256):
    return [
        "--revision",
        actual["revision"],
        "--deployment",
        actual["deployment"],
        "--binding-sha256",
        actual["binding_sha256"],
        "--contract",
        str(path),
        "--contract-sha256",
        sha256,
    ]


def worker_argv(actual, path, sha256, worker):
    require(type(worker) is int and worker in (0, 1), "Only B lanes 0/1")
    return [
        "srun",
        "--exclusive",
        "--exact",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:pro6000:1",
        "--cpus-per-task=4",
        "--mem=33G",
        "--kill-on-bad-exit=0",
        "--export=ALL",
        str(PYTHON),
        "-I",
        "-B",
        "-u",
        str(Path(actual["deployment"]) / SCRIPT),
        "lane",
        *arguments(actual, path, sha256),
        "--worker",
        str(worker),
    ]


def child_environment(cache):
    env = dict(
        PATH="/usr/bin:/bin",
        TMPDIR=str(cache),
        XDG_CACHE_HOME=str(cache),
        HF_HOME=str(cache / "hf"),
        TORCH_HOME=str(cache / "torch"),
        PYTHONPYCACHEPREFIX=str(cache / "pycache"),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SLURM_EXPORT_ENV="ALL",
    )
    env.update(
        {
            k: v
            for k, v in os.environ.items()
            if k.startswith("SLURM_") or k == "CUDA_VISIBLE_DEVICES"
        }
    )
    env["SLURM_EXPORT_ENV"] = "ALL"
    return env


def launch(actual, path, sha256):
    previous = frozen_resume()
    legacy, _ = previous.frozen_v4()
    old = legacy.original()
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        re.fullmatch(r"[0-9]+", job)
        and os.environ.get("SLURM_JOB_NAME") == JOB_NAME
        and os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
        "Matching non-requeued allocation required",
    )
    with legacy.lock(OPS / "round_b_LAUNCH.lock"):
        require(
            not any(
                (OPS / ("round_b_" + name + ".json")).exists()
                for name in ("STARTED", "COMPLETE", "FAILURE")
            )
            and not (OPS / "LANES").exists()
            and not (OPS / "EXECUTIONS").exists(),
            "Prior B launch forbids retry",
        )
        try:
            deadline = time.monotonic() + 30
            while not (OPS / "round_b_SUBMISSION.json").exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            receipt = previous.read(OPS / "round_b_SUBMISSION.json")
            intent = previous.read(OPS / "round_b_ATTEMPT.json")
            require(
                receipt.get("status") == "SUBMITTED"
                and receipt.get("job_id") == job
                and receipt.get("source") == actual
                and receipt.get("attempt_sha256") == previous.sha(OPS / "round_b_ATTEMPT.json")
                and intent.get("source") == actual
                and intent.get("contract") == dict(path=str(path), sha256=sha256)
                and intent.get("argv") == argv_for(actual, path, sha256),
                "Wrong or partial B submission",
            )
            hardware = allocation(old, job)
            contract = metadata(actual, path, sha256)
            with transition_locks(previous, legacy):
                terminal(old)
                untouched_b(previous)
            cache = previous.canonical(OPS / ("job-" + job), exists=False)
            cache.mkdir()
            env = child_environment(cache)
            old.write_once(
                OPS / "round_b_STARTED.json",
                dict(
                    at=old.now(),
                    job_id=job,
                    source=actual,
                    contract_ref=dict(path=str(path), sha256=sha256),
                    allocation=hardware,
                ),
            )
            codes = run_lanes(actual, path, sha256, env)
            require(codes == [0, 0], "B lane failed; independent peer retained, no retry")
            completed = [
                task_complete(previous, contract, path, sha256, task, job)
                for queue in QUEUES
                for task in queue
            ]
            result = dict(
                status="ROUND_B_COMPLETE_NOT_GATE_SIGNED",
                at=old.now(),
                job_id=job,
                source=actual,
                contract_ref=dict(path=str(path), sha256=sha256),
                queues=QUEUES,
                completed=completed,
                scientific_gate_status="NOT_SIGNED",
                evaluation_authorized=False,
            )
            old.write_once(OPS / "round_b_COMPLETE.json", result)
            return result
        except BaseException as exc:
            old.write_once(
                OPS / "round_b_FAILURE.json",
                dict(
                    status="FAILED_PRESERVED",
                    at=old.now(),
                    job_id=job,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    retry_authorized=False,
                    peer_cancellation_requested=False,
                    scientific_failure=False,
                ),
            )
            raise


def run_lanes(actual, path, sha256, env):
    """Always wait for every started independent lane, even if peer creation fails."""
    children, error = [], None
    try:
        for worker in (0, 1):
            children.append(subprocess.Popen(worker_argv(actual, path, sha256, worker), env=env))
    except BaseException as exc:
        error = exc
    codes = [child.wait() for child in children]
    if error is not None:
        raise error
    return codes


def task_complete(previous, contract, path, sha256, task, job):
    worker = 0 if task in QUEUES[0] else 1
    extension = OPS / "EXECUTIONS" / task
    require(not (extension / "FAILURE.json").exists(), "Failed B execution")
    original_path = JOBS / "TASKS" / task / "COMPLETE.json"
    original, receipt = previous.read(original_path), previous.read(extension / "COMPLETE.json")
    cr = dict(path=str(path), sha256=sha256)
    require(
        original.get("status") == "TASK_COMPLETE_NOT_GATE_SIGNED"
        and original.get("task", {}).get("task_id") == task
        and original.get("gate_status") == "NOT_SIGNED"
        and all(receipt.get(k) == v for k, v in original.items())
        and receipt.get("round_b_contract_ref") == cr
        and receipt.get("execution_source") == contract["execution_source"]
        and receipt.get("full_trainer_numerical_pass") is False
        and receipt.get("full_validation_ref")
        == dict(
            path=str(previous.FULL_OPS / "VALIDATION/RESULT.json"), sha256=previous.FULL_RESULT_SHA
        ),
        "Original B closure and actual execution differ",
    )
    facts_ref = dict(
        path=str(extension / "STORAGE_FACTS.json"),
        sha256=previous.sha(extension / "STORAGE_FACTS.json"),
    )
    require(receipt.get("storage_facts_ref") == facts_ref, "B storage binding differs")
    facts = previous.small_ref(extension, facts_ref)
    require(
        facts.get("status") == "ACTUAL_STORAGE_CONTEXT_COMPLETE_NOT_SCIENTIFIC_PASS"
        and facts.get("contract_ref") == cr
        and facts.get("execution_id") == f"LH-SRF6-B-{worker}-B-{job}-{task}"
        and facts.get("offload") == receipt.get("offload")
        and facts.get("validation_ref") == receipt.get("full_validation_ref"),
        "Actual B storage facts missing",
    )
    return dict(
        task_id=task,
        task_complete_ref=dict(path=str(original_path), sha256=previous.sha(original_path)),
        execution_complete_ref=dict(
            path=str(extension / "COMPLETE.json"), sha256=previous.sha(extension / "COMPLETE.json")
        ),
        storage_facts_ref=facts_ref,
    )


def lane(actual, path, sha256, worker):
    require(type(worker) is int and worker in (0, 1), "Fixed B lane required")
    require(
        os.environ.get("OMP_NUM_THREADS") == "1" and os.environ.get("SLURM_CPUS_PER_TASK") == "4",
        "Original thread environment and lane CPU allocation required",
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(
        visible and "," not in visible and visible != "-1", "One assigned GPU per lane required"
    )
    previous = frozen_resume()
    legacy, _ = previous.frozen_v4()
    old = legacy.original()
    job = os.environ.get("SLURM_JOB_ID", "")
    allocation(old, job)
    contract = metadata(actual, path, sha256)
    started = previous.read(OPS / "round_b_STARTED.json")
    require(
        started.get("job_id") == job
        and started.get("source") == actual
        and started.get("contract_ref") == dict(path=str(path), sha256=sha256),
        "Matching B launch required",
    )
    folder = OPS / "LANES" / str(worker)
    with legacy.lock(OPS / f"lane-{worker}.lock"):
        require(not folder.exists(), "Prior B lane forbids retry")
        previous.canonical(folder, exists=False).mkdir(parents=True)
        old.write_once(
            folder / "STARTED.json",
            dict(at=old.now(), job_id=job, worker=worker, queue=QUEUES[worker], source=actual),
        )
        try:
            bootstrap = (
                "import runpy,sys;sys.path[:0]="
                + repr(
                    [
                        str(CODE / "src"),
                        str(previous.FULL_DEPLOYMENT / "src"),
                        str(V8 / "src"),
                        str(Path(actual["deployment"]) / "src"),
                    ]
                )
                + ";sys.argv=['sr_f6_round_b.execution']+sys.argv[1:];"
                "runpy.run_module('sr_f6_round_b.execution',run_name='__main__')"
            )
            completed = []
            for task in QUEUES[worker]:
                argv = [
                    str(PYTHON),
                    "-I",
                    "-B",
                    "-u",
                    "-c",
                    bootstrap,
                    "run",
                    "--contract",
                    str(path),
                    "--contract-sha256",
                    sha256,
                    "--task-id",
                    task,
                    "--attempt-id",
                    "B-" + job,
                ]
                with (folder / (task + ".log")).open("xb") as stream:
                    process = subprocess.run(
                        argv, stdout=stream, stderr=subprocess.STDOUT, check=False
                    )
                require(process.returncode == 0, "B task failed; stop this lane, no retry")
                completed.append(task_complete(previous, contract, path, sha256, task, job))
            result = dict(
                status="QUEUE_COMPLETE_NOT_GATE_SIGNED",
                worker=worker,
                job_id=job,
                completed=completed,
            )
            old.write_once(folder / "COMPLETE.json", result)
            return result
        except BaseException as exc:
            old.write_once(
                folder / "FAILURE.json",
                dict(
                    status="FAILED_PRESERVED",
                    worker=worker,
                    job_id=job,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    peer_cancellation_requested=False,
                    retry_authorized=False,
                ),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("submit", "launch", "lane"))
    for field in ("revision", "deployment", "binding-sha256", "contract", "contract-sha256"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1))
    args = parser.parse_args()
    actual = identity(args.revision, args.deployment, args.binding_sha256)
    if args.operation == "lane":
        result = lane(actual, args.contract, args.contract_sha256, args.worker)
    else:
        require(args.worker is None, "Worker only applies to lane")
        result = (submit if args.operation == "submit" else launch)(
            actual, args.contract, args.contract_sha256
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
