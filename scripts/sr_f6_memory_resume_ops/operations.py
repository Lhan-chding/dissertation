"""Unique original b1 queue recovery after actual full validation; no automatic retry."""

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
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_RESUME_V8_20261030"
PRODUCER = "b7e7b067f237b360b1dfb2c3075d3f14be7c281f"
CODE = PROJECT / "sr_f6_code_b7e7b06"
PYTHON = PROJECT / "envs/ssvc-py312/bin/python3.12"
USER = "varun024"
QOS = "soujanya-poria-startfund-2026-03"
JOB_NAME = "LH-SRF6-Memory-Resume"
SCRIPT = "scripts/sr_f6_memory_resume_ops/operations.py"
LAUNCHER = "scripts/sr_f6_memory_resume_ops/launch.sbatch"
MEMBERS = (
    SCRIPT,
    LAUNCHER,
    "src/sr_f6_memory_resume/__init__.py",
    "src/sr_f6_memory_resume/contract.py",
    "src/sr_f6_memory_resume/execution.py",
)
QUEUE = ("PRIMARY-b1-U10", "PRIMARY-b1-C_SHUF", "PRIMARY-b1-U11")
FULL_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V7_20261030"
FULL_REVISION = "4be18ca1c9bd2db0b905b0d3c3f22873f97cc081"
FULL_DEPLOYMENT = PROJECT / "sr_f6_memory_full_code_4be18ca"
FULL_BINDING = "797bc640bea90b03f21a4952196df1d18ef196459aaae481d0b740482787fe02"
FULL_CONTRACT_SHA = "e5e14a0ce0036b9b640c930d01fc9f34ce68a7f48f0303eebb44d323b0e28cd9"
FULL_RESULT_SHA = "2f6c641d2dd84b1691459eccca766e02ffbd4bd1c6af6047986e843edd2e34e7"
FULL_COMPLETE_SHA = "29a4204f3e7a3d02de0b1486cddaa4d95768d90f7f9b95f1eb51287afba76f4e"
FULL_COLD_SHA = "376448405f87ef45c09ab3a91e15e871da756ffe07b91c24c18758c7a1bd7108"


def require(value, message):
    if not value:
        raise PermissionError(message)


def canonical(path, *, exists=True):
    path = Path(path)
    require(
        path.is_absolute()
        and path.is_relative_to(PROJECT)
        and path != PROJECT
        and ".." not in path.parts
        and path.resolve(strict=exists) == path
        and not path.is_symlink(),
        "Canonical project descendant required",
    )
    return path


def read(path):
    path = canonical(path)
    require(path.is_file() and path.stat().st_size <= 4 * 2**20, "Bounded regular JSON required")
    return json.loads(path.read_bytes())


def sha(path):
    path = canonical(path)
    require(
        path.is_file() and path.stat().st_size <= 4 * 2**20, "Bounded source/receipt hash required"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def small_ref(base, ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact reference required")
    path = Path(ref["path"])
    path = path if path.is_absolute() else base / path
    require(sha(path) == ref["sha256"], "Immutable reference changed")
    return read(path)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, canonical(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def identity(revision, deployment, binding_sha256):
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Full resume revision required")
    target = PROJECT / ("sr_f6_memory_resume_code_" + revision[:7])
    require(
        Path(deployment) == target and Path(__file__).resolve() == target / SCRIPT,
        "Wrong actual resume operator origin",
    )
    require(sha(target / "RESUME_SOURCE.json") == binding_sha256, "Resume source binding changed")
    manifest = read(target / "RESUME_SOURCE.json")
    require(
        manifest.get("schema") == "srf6-memory-resume-source-v1"
        and manifest.get("revision") == revision
        and manifest.get("deployment") == str(target)
        and manifest.get("producer_revision") == PRODUCER
        and set(manifest.get("files", {})) == set(MEMBERS),
        "Exact resume source closure required",
    )
    for name, pin in manifest["files"].items():
        path = canonical(target / name)
        require(
            set(pin) == {"bytes", "sha256"}
            and path.stat().st_size == pin["bytes"]
            and sha(path) == pin["sha256"],
            "Resume source bytes changed",
        )
    return dict(revision=revision, deployment=str(target), binding_sha256=binding_sha256)


def frozen_full():
    require(
        sha(FULL_DEPLOYMENT / "FULL_SOURCE.json") == FULL_BINDING, "Frozen full binding changed"
    )
    manifest = read(FULL_DEPLOYMENT / "FULL_SOURCE.json")
    name = "scripts/sr_f6_memory_full_ops/operations.py"
    source = FULL_DEPLOYMENT / name
    require(sha(source) == manifest["files"][name]["sha256"], "Frozen full operator changed")
    module = load_file("srf6_resume_frozen_full", source)
    return module, module.identity(FULL_REVISION, str(FULL_DEPLOYMENT), FULL_BINDING)


def frozen_v4():
    module, _ = frozen_full()
    return module.frozen_v4()


def contract_module(actual):
    sys.path[:0] = [
        str(CODE / "src"),
        str(FULL_DEPLOYMENT / "src"),
        str(Path(actual["deployment"]) / "src"),
    ]
    module = importlib.import_module("sr_f6_memory_resume.contract")
    require(
        Path(module.__file__).resolve()
        == Path(actual["deployment"]) / "src/sr_f6_memory_resume/contract.py",
        "Wrong actual resume verifier source",
    )
    return module


def metadata(actual, contract_path, contract_sha256):
    require(Path(contract_path) == OPS / "CONTRACT.json", "Unique resume contract required")
    contract = contract_module(actual).validate_metadata(contract_path, contract_sha256)
    require(
        contract["execution_source"]["revision"] == actual["revision"]
        and contract["execution_source"]["deployment"] == actual["deployment"]
        and contract["queue"] == list(QUEUE),
        "Resume contract source or queue differs",
    )
    manifest = read(Path(actual["deployment"]) / "RESUME_SOURCE.json")
    expected = {name: pin["sha256"] for name, pin in manifest["files"].items()}
    refs = contract["execution_source"]["files"]
    observed = {ref["path"]: ref["sha256"] for ref in refs}
    require(
        len(refs) == len(observed) and observed == expected, "Exact resume source closure required"
    )
    return contract


@contextmanager
def transition_locks(legacy):
    module, _ = frozen_full()
    # The inherited chain owns v2/v3/v4 resume locks once, plus v5/v6 smoke locks.
    with (
        module.transition_locks(legacy),
        legacy.lock(FULL_OPS / "validate.lock", existing=True),
        legacy.lock(FULL_OPS / "validate_LAUNCH.lock", existing=True),
    ):
        yield


def no_prior_resume():
    # Empty lock files are synchronization, never an execution intent.
    for version, day in (
        (1, "20261029"),
        (2, "20261029"),
        (3, "20261030"),
        (4, "20261030"),
        (5, "20261030"),
        (6, "20261030"),
        (7, "20261030"),
    ):
        prior = PROJECT / f"sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V{version}_{day}"
        if prior.exists():
            canonical(prior)
            require(
                not list(prior.glob("resume_*.json")) and not (prior / "EXECUTIONS").exists(),
                "Historical resume intent or execution forbids duplicate recovery",
            )
    for task in QUEUE:
        require(
            not (ROOT / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task / "COMPLETE.json").exists(),
            "Original completed task cannot be repeated",
        )


def predecessor(legacy, old, contract):
    module, actual = frozen_full()
    refs = {
        "full_contract_ref": dict(path=str(FULL_OPS / "CONTRACT.json"), sha256=FULL_CONTRACT_SHA),
        "full_result_ref": dict(
            path=str(FULL_OPS / "VALIDATION/RESULT.json"), sha256=FULL_RESULT_SHA
        ),
        "full_complete_ref": dict(
            path=str(FULL_OPS / "validate_COMPLETE.json"), sha256=FULL_COMPLETE_SHA
        ),
        "full_cold_ref": dict(
            path=str(FULL_OPS / "VALIDATION/COLD_VERIFICATION.json"), sha256=FULL_COLD_SHA
        ),
    }
    for name, ref in refs.items():
        require(contract.get(name) == ref, "Exact completed full predecessor required")
        small_ref(FULL_OPS, ref)
    # Preserve the entire historical chain and use metadata only on the login node.
    module.predecessor(legacy, old, read(FULL_OPS / "CONTRACT.json"))
    terminal = module.validation_terminal(
        old, actual, str(FULL_OPS / "CONTRACT.json"), FULL_CONTRACT_SHA, refs["full_result_ref"]
    )
    require(
        terminal.get("job_id") == "235864"
        and terminal.get("state") == "COMPLETED"
        and terminal.get("exit_code") == "0:0"
        and terminal.get("queue_gone") is True
        and terminal.get("result_ref") == refs["full_result_ref"]
        and terminal.get("cold_verification_ref") == refs["full_cold_ref"],
        "Actual full terminal proof differs",
    )
    no_prior_resume()
    quiescent = legacy.quiescent(old.query)
    return dict(full=terminal, original_b1=quiescent)


def argv_for(actual, contract_path, contract_sha256):
    exports = dict(
        PATH="/usr/bin:/bin",
        SRF6_RESUME_REVISION=actual["revision"],
        SRF6_RESUME_DEPLOYMENT=actual["deployment"],
        SRF6_RESUME_BINDING=actual["binding_sha256"],
        SRF6_RESUME_CONTRACT=str(contract_path),
        SRF6_RESUME_CONTRACT_SHA=contract_sha256,
    )
    require(all("," not in v and "\n" not in v for v in exports.values()), "Unsafe export value")
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--job-name=" + JOB_NAME,
        "--account=rose",
        "--partition=cluster02",
        "--qos=" + QOS,
        "--gres=gpu:pro6000:1",
        "--cpus-per-task=4",
        "--mem=33G",
        "--time=23:00:00",
        "--nodes=1",
        "--chdir=" + str(CODE),
        "--output=" + str(OPS / "resume-%j.log"),
        "--export=" + ",".join(k + "=" + v for k, v in exports.items()),
        str(Path(actual["deployment"]) / LAUNCHER),
    ]


def submit(actual, contract_path, contract_sha256):
    require(getpass.getuser() == USER, "Wrong scheduler account")
    legacy, _ = frozen_v4()
    old = legacy.original()
    old.clean_submission_environment()
    contract = metadata(actual, contract_path, contract_sha256)
    with transition_locks(legacy), legacy.lock(OPS / "resume.lock"):
        prior = predecessor(legacy, old, contract)
        require(
            not any(
                (OPS / ("resume_" + s + ".json")).exists()
                for s in (
                    "ATTEMPT",
                    "SUBMISSION",
                    "STARTED",
                    "COMPLETE",
                    "FAILURE",
                    "WATCHDOG_ATTEMPT",
                    "WATCHDOG_RESULT",
                )
            )
            and not (OPS / "EXECUTIONS").exists(),
            "Prior or partial resume forbids retry",
        )
        capacity_module, _ = old.scheduler(old.metadata())
        capacity = capacity_module.capacity(query=old.query)
        require(
            capacity["gpu_limit"] == 8 and capacity["submit_limit"] == 5, "Resource limits differ"
        )
        if capacity["reserved_gpus"] + 1 > 8 or capacity["submitted_by_user"] + 1 > 5:
            return dict(status="NO_LEGAL_CAPACITY", capacity=capacity)
        argv = argv_for(actual, contract_path, contract_sha256)
        old.write_once(
            OPS / "resume_ATTEMPT.json",
            dict(
                status="SUBMISSION_INTENT_NOT_COMPLETION",
                at=old.now(),
                source=actual,
                contract=dict(path=str(contract_path), sha256=contract_sha256),
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
        result.update(source=actual, at=old.now(), attempt_sha256=sha(OPS / "resume_ATTEMPT.json"))
        old.write_once(OPS / "resume_SUBMISSION.json", result)
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
        and fields.get("QOS") == QOS
        and fields.get("Partition") == "cluster02"
        and fields.get("Account") == "rose"
        and fields.get("UserId", "").startswith(USER + "("),
        "Wrong actual resume allocation",
    )
    for key in ("ReqTRES", "AllocTRES"):
        tres = dict(item.split("=", 1) for item in fields.get(key, "").split(",") if "=" in item)
        require(
            tres.get("gres/gpu:pro6000") == "1"
            and tres.get("gres/gpu", "1") == "1"
            and tres.get("cpu") == "4"
            and tres.get("mem") in ("33G", "33792M"),
            "Wrong actual resume resources",
        )
    return fields


def launch(actual, contract_path, contract_sha256):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        re.fullmatch(r"[0-9]+", job) is not None
        and os.environ.get("SLURM_JOB_NAME") == JOB_NAME
        and os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
        "Existing matching non-requeued job required",
    )
    legacy, _ = frozen_v4()
    old = legacy.original()
    with legacy.lock(OPS / "resume_LAUNCH.lock"):
        require(
            not any(
                (OPS / ("resume_" + s + ".json")).exists()
                for s in ("STARTED", "COMPLETE", "FAILURE", "WATCHDOG_ATTEMPT", "WATCHDOG_RESULT")
            )
            and not (OPS / "EXECUTIONS").exists(),
            "Prior launch cannot repeat",
        )
        try:
            deadline = time.monotonic() + 30
            while not (OPS / "resume_SUBMISSION.json").exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            receipt = read(OPS / "resume_SUBMISSION.json")
            intent = read(OPS / "resume_ATTEMPT.json")
            require(
                receipt.get("status") == "SUBMITTED"
                and receipt.get("job_id") == job
                and receipt.get("source") == actual
                and receipt.get("attempt_sha256") == sha(OPS / "resume_ATTEMPT.json")
                and intent.get("source") == actual
                and intent.get("contract") == dict(path=str(contract_path), sha256=contract_sha256)
                and intent.get("argv") == argv_for(actual, contract_path, contract_sha256),
                "Wrong/partial resume submission",
            )
            hardware = allocation(old, job)
            contract = metadata(actual, contract_path, contract_sha256)
            with transition_locks(legacy):
                predecessor(legacy, old, contract)
            cache = canonical(OPS / ("job-" + job), exists=False)
            cache.mkdir()
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
            require(
                bool(env.get("CUDA_VISIBLE_DEVICES")) and "," not in env["CUDA_VISIBLE_DEVICES"],
                "One assigned GPU required",
            )
            old.write_once(
                OPS / "resume_STARTED.json",
                dict(at=old.now(), job_id=job, source=actual, allocation=hardware),
            )
            bootstrap = (
                "import runpy,sys;sys.path[:0]="
                + repr(
                    [
                        str(CODE / "src"),
                        str(FULL_DEPLOYMENT / "src"),
                        str(Path(actual["deployment"]) / "src"),
                    ]
                )
                + ";sys.argv=['sr_f6_memory_resume.execution']+sys.argv[1:];"
                "runpy.run_module('sr_f6_memory_resume.execution',run_name='__main__')"
            )
            completed = []
            for task_id in QUEUE:
                argv = [
                    str(PYTHON),
                    "-I",
                    "-B",
                    "-u",
                    "-c",
                    bootstrap,
                    "run",
                    "--contract",
                    str(contract_path),
                    "--contract-sha256",
                    contract_sha256,
                    "--task-id",
                    task_id,
                    "--attempt-id",
                    "MEM-" + job,
                ]
                process = subprocess.run(argv, env=env, check=False)
                require(
                    process.returncode == 0, "Resume child failed; preserved, no automatic retry"
                )
                completed.append(
                    task_complete(contract, contract_path, contract_sha256, task_id, job)
                )
            complete = dict(
                status="QUEUE_COMPLETE_NOT_GATE_SIGNED",
                at=old.now(),
                job_id=job,
                source=actual,
                contract_ref=dict(path=str(contract_path), sha256=contract_sha256),
                full_validation_ref=contract["full_result_ref"],
                queue=list(QUEUE),
                completed=completed,
                original_worker_failure_preserved=True,
                scientific_gate_status="NOT_SIGNED",
            )
            old.write_once(OPS / "resume_COMPLETE.json", complete)
            return complete
        except BaseException as exc:
            old.write_once(
                OPS / "resume_FAILURE.json",
                dict(
                    status="FAILED_PRESERVED",
                    at=old.now(),
                    job_id=job,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    retry_authorized=False,
                    scientific_failure=False,
                ),
            )
            raise


def task_complete(contract, contract_path, contract_sha256, task_id, job):
    """Check small publication links; GPU child performed original complete task verification."""
    original_path = ROOT / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task_id / "COMPLETE.json"
    extension = OPS / "EXECUTIONS" / task_id
    require(not (extension / "FAILURE.json").exists(), "Failed execution cannot complete")
    original = read(original_path)
    receipt = read(extension / "COMPLETE.json")
    cr = dict(path=str(contract_path), sha256=contract_sha256)
    require(
        original.get("status") == "TASK_COMPLETE_NOT_GATE_SIGNED"
        and original.get("task", {}).get("task_id") == task_id
        and original.get("gate_status") == "NOT_SIGNED"
        and all(receipt.get(k) == v for k, v in original.items())
        and receipt.get("resume_contract_ref") == cr
        and receipt.get("execution_source") == contract["execution_source"]
        and receipt.get("full_validation_ref") == contract["full_result_ref"]
        and receipt.get("full_trainer_numerical_pass") is False,
        "Original and resumed task completion must agree",
    )
    facts_ref = dict(
        path=str(extension / "STORAGE_FACTS.json"), sha256=sha(extension / "STORAGE_FACTS.json")
    )
    require(receipt.get("storage_facts_ref") == facts_ref, "Storage completion binding differs")
    facts = small_ref(extension, facts_ref)
    require(
        facts.get("status") == "ACTUAL_STORAGE_CONTEXT_COMPLETE_NOT_SCIENTIFIC_PASS"
        and facts.get("contract_ref") == cr
        and facts.get("validation_ref") == contract["full_result_ref"]
        and facts.get("execution_id") == f"LH-SRF6-A-1-MEM-{job}-{task_id}"
        and facts.get("offload") == receipt.get("offload"),
        "Actual task storage facts missing",
    )
    return dict(
        task_id=task_id,
        task_complete_ref=dict(path=str(original_path), sha256=sha(original_path)),
        execution_complete_ref=dict(
            path=str(extension / "COMPLETE.json"), sha256=sha(extension / "COMPLETE.json")
        ),
        storage_facts_ref=facts_ref,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("submit", "launch"))
    for field in ("revision", "deployment", "binding-sha256", "contract", "contract-sha256"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    actual = identity(args.revision, args.deployment, args.binding_sha256)
    result = (submit if args.operation == "submit" else launch)(
        actual, args.contract, args.contract_sha256
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
