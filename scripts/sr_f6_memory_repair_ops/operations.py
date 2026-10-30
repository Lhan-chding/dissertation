"""Unique source-bound operations for the original b1 memory repair.

One independent GPU lane exists. These operations never restart Round A or touch
b0, and never treat an inner success flag as validation/admission evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V3_20261030"
PREDECESSOR_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V2_20261029"
PREDECESSOR_REVISION = "f4b882dfbfa28e73da22244c082d324bdc7f2d2c"
PREDECESSOR_CONTRACT_SHA = "7986ec98efac680254368e6431c5652d307e4816c7a58986fd733240c002bec6"
FAILED_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V1_20261029"
FAILED_REVISION = "e24d557c381e53bbf85773c4c6551530a1a084e6"
CODE = PROJECT / "sr_f6_code_b7e7b06"
PRODUCER = "b7e7b067f237b360b1dfb2c3075d3f14be7c281f"
OLD_OPS = PROJECT / "sr_f6_pilot_ops_code_74a95f5"
OLD_REVISION = "74a95f5b94e5c4e1d92ba7a5ba8b250b850ac4e8"
OLD_BINDING = "0ed394b9ebabc4f752b0f7a44df67acbc5b4fdcfa601146be0ef5434dddb7c64"
PYTHON = PROJECT / "envs/ssvc-py312/bin/python3.12"
USER = "varun024"
QOS = "soujanya-poria-startfund-2026-03"
SCRIPT = "scripts/sr_f6_memory_repair_ops/operations.py"
LAUNCHER = "scripts/sr_f6_memory_repair_ops/launch.sbatch"
MEMBERS = (
    SCRIPT,
    LAUNCHER,
    "scripts/sr_f6_memory_repair_ops/benchmark_storage.py",
    "src/sr_f6_memory_repair/__init__.py",
    "src/sr_f6_memory_repair/offload.py",
    "src/sr_f6_memory_repair/arena.py",
    "src/sr_f6_memory_repair/execution.py",
    "src/sr_f6_memory_repair/validation.py",
)
TASKS = ("PRIMARY-b1-U10", "PRIMARY-b1-C_SHUF", "PRIMARY-b1-U11")
NAMES = dict(validate="LH-SRF6-Memory-Validate", resume="LH-SRF6-Memory-Resume")


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def canonical(path, *, exists=True):
    path = Path(path)
    require(
        path.is_absolute() and path.is_relative_to(PROJECT) and ".." not in path.parts,
        "Canonical project descendant required",
    )
    require(path.resolve(strict=exists) == path and not path.is_symlink(), "Symlink forbidden")
    return path


def read(path):
    path = canonical(path)
    require(path.is_file(), "Regular file required")
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(canonical(path).read_bytes()).hexdigest()


def original():
    """Reuse already frozen source and capacity checks, never edit that module."""
    path = canonical(OLD_OPS / "scripts/sr_f6_pilot_ops/round_a.py")
    binding = read(OLD_OPS / "OPERATION_BINDING.json")
    require(sha(OLD_OPS / "OPERATION_BINDING.json") == OLD_BINDING, "Old source binding changed")
    require(
        sha(path) == binding["files"]["scripts/sr_f6_pilot_ops/round_a.py"]["sha256"],
        "Old operator bytes changed",
    )
    spec = importlib.util.spec_from_file_location("srf6_frozen_round_a_ops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.operation_identity(OLD_REVISION, str(OLD_OPS), OLD_BINDING)
    return module


def identity(revision, deployment, binding_sha256):
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Full repair revision required")
    target = PROJECT / ("sr_f6_memory_repair_code_" + revision[:7])
    require(
        Path(deployment) == target and Path(__file__).resolve() == target / SCRIPT,
        "Wrong actual repair operations origin",
    )
    manifest = canonical(target / "MEMORY_REPAIR_SOURCE.json")
    require(sha(manifest) == binding_sha256, "Repair source binding changed")
    value = read(manifest)
    require(
        value.get("schema") == "srf6-memory-repair-source-v1"
        and value.get("revision") == revision
        and value.get("deployment") == str(target)
        and value.get("producer_revision") == PRODUCER
        and set(value.get("files", {})) == set(MEMBERS),
        "Exact complete composed source required",
    )
    for name, pin in value["files"].items():
        path = canonical(target / name)
        require(
            set(pin) == {"bytes", "sha256"}
            and path.stat().st_size == pin["bytes"]
            and sha(path) == pin["sha256"],
            "Repair source bytes changed",
        )
    return dict(revision=revision, deployment=str(target), binding_sha256=binding_sha256)


def small_ref(base, ref):
    """Bound only small JSON receipts; never load tensors or recursively hash trees."""
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact small ref required")
    path = Path(ref["path"])
    path = canonical(path if path.is_absolute() else base / path)
    require(path.stat().st_size <= 4 * 1024 * 1024, "Receipt exceeds lightweight bound")
    require(sha(path) == ref["sha256"], "Small immutable reference changed")
    return read(path)


def submission_metadata(actual, contract_path, contract_sha256, facts):
    """Submission precheck only: full cold verification remains inside allocation."""
    value = small_ref(ROOT, dict(path=str(contract_path), sha256=contract_sha256))
    require(
        value.get("schema") == "srf6-memory-repair-contract-v3"
        and value.get("storage_backend") == "arena_v3"
        and value.get("max_host_bytes") == 8 * 2**30
        and value.get("max_disk_bytes") == 96 * 2**30
        and value.get("experiment_root") == str(ROOT)
        and value.get("original_job_id") == "233996"
        and value.get("original_step_id") == "233996.1"
        and value.get("original_execution_id") == "LH-SRF6-A-1-A-233996-PRIMARY-b1-U10",
        "Original repair target changed",
    )
    producer, repair = value["producer"], value["repair_source"]
    require(
        producer["revision"] == PRODUCER
        and producer["deployment"] == str(CODE)
        and repair["revision"] == actual["revision"]
        and repair["deployment"] == actual["deployment"],
        "Contract source identity differs",
    )
    binding = read(Path(actual["deployment"]) / "MEMORY_REPAIR_SOURCE.json")
    expected = {
        str(Path(actual["deployment"]) / name): pin["sha256"]
        for name, pin in binding["files"].items()
    }
    refs = repair["files"]
    observed = {}
    for ref in refs:
        require(set(ref) == {"path", "sha256"}, "Exact source ref required")
        path = Path(ref["path"])
        path = canonical(path if path.is_absolute() else Path(actual["deployment"]) / path)
        require(str(path) not in observed, "Duplicate repair source")
        observed[str(path)] = ref["sha256"]
    require(observed == expected, "Contract must bind exact operation and repair source")
    require(small_ref(ROOT, value["plan_ref"]) == facts["plan"], "Original plan differs")
    require(
        small_ref(ROOT, producer["source_manifest_ref"]) == facts["manifest"], "Producer differs"
    )
    for key in (
        "original_ready_ref",
        "step0_commit_ref",
        "original_failure_ref",
        "original_task_failure_ref",
        "original_worker_failure_ref",
    ):
        small_ref(ROOT, value[key])
    prior = value["old_allocation_failure_ref"]
    require(
        Path(prior["path"]) == FAILED_OPS / "validate_FAILURE.json",
        "Original allocation failure location differs",
    )
    failure = small_ref(FAILED_OPS, prior)
    require(
        failure.get("job_id") == "234358"
        and failure.get("status") == "FAILED_PRESERVED"
        and failure.get("error_type") == "PermissionError"
        and failure.get("error") == "Wrong actual GPU/CPU/host-memory allocation"
        and failure.get("retry_authorized") is False,
        "Preserved pre-model allocation failure required",
    )
    return dict(status="SUBMISSION_METADATA_ONLY_NOT_ADMISSION", full_verification="IN_GPU_CHILD")


def prior_allocation_terminal(old):
    """New independent root is allowed only for the evidenced zero-model failure."""
    receipt = read(FAILED_OPS / "validate_SUBMISSION.json")
    intent = read(FAILED_OPS / "validate_ATTEMPT.json")
    source = receipt.get("source", {})
    require(
        receipt.get("status") == "SUBMITTED"
        and receipt.get("job_id") == "234358"
        and receipt.get("attempt_sha256") == sha(FAILED_OPS / "validate_ATTEMPT.json")
        and source.get("revision") == FAILED_REVISION
        and source.get("deployment") == str(PROJECT / "sr_f6_memory_repair_code_e24d557")
        and intent.get("source") == source
        and intent.get("action") == "validate"
        and intent.get("validation") is None
        and "--mem=128G" in intent.get("argv", []),
        "Original unique failed allocation identity differs",
    )
    require(
        not (FAILED_OPS / "validate_STARTED.json").exists()
        and not (FAILED_OPS / "validate_COMPLETE.json").exists()
        and not (FAILED_OPS / "VALIDATION").exists()
        and not any(FAILED_OPS.glob("resume*")),
        "Original failed allocation has model or recovery evidence",
    )
    rows = [
        line.split("|")
        for line in old.query(
            ["sacct", "-n", "-P", "-j", "234358", "--format=JobIDRaw,State,ExitCode"]
        ).splitlines()
    ]
    main = [row for row in rows if row and row[0] == "234358"]
    require(
        len(main) == 1 and main[0][1:3] == ["FAILED", "1:0"],
        "Original validation allocation is not terminal failed",
    )
    require(
        "234358" not in old.query(["squeue", "-h", "-u", USER, "-o", "%i"]).split(),
        "Original validation allocation still queued",
    )
    return dict(job_id="234358", status="FAILED_BEFORE_MODEL_PRESERVED")


@contextmanager
def predecessor_locks():
    """Serialize against immutable v2 submitters during the transition check."""
    with ExitStack() as stack:
        # resume.lock may not exist until first observation; it is not evidence.
        stack.enter_context(lock(PREDECESSOR_OPS / "resume.lock"))
        stack.enter_context(lock(PREDECESSOR_OPS / "validate.lock", existing=True))
        yield


def predecessor_records(actual, contract_path, contract_sha256):
    contract = small_ref(ROOT, dict(path=str(contract_path), sha256=contract_sha256))
    module_path = canonical(Path(actual["deployment"]) / "src/sr_f6_memory_repair/execution.py")
    spec = importlib.util.spec_from_file_location("srf6_v3_predecessor_verifier", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predecessor_evidence(contract)


def predecessor_terminal(old, actual, contract_path, contract_sha256):
    """A failed controlled interruption permanently fences immutable v2 resume."""
    disposition = predecessor_records(actual, contract_path, contract_sha256)
    receipt = read(PREDECESSOR_OPS / "validate_SUBMISSION.json")
    intent = read(PREDECESSOR_OPS / "validate_ATTEMPT.json")
    source = receipt.get("source", {})
    require(
        receipt.get("status") == "SUBMITTED"
        and receipt.get("job_id") == "234384"
        and receipt.get("attempt_sha256") == sha(PREDECESSOR_OPS / "validate_ATTEMPT.json")
        and source.get("revision") == PREDECESSOR_REVISION
        and source.get("deployment") == str(PROJECT / "sr_f6_memory_repair_code_f4b882d")
        and source.get("binding_sha256")
        == "bfb521f15b6caae49225548643a54f5b72a90e117f26288470fcd35db638be28"
        and intent.get("source") == source
        and intent.get("action") == "validate"
        and intent.get("contract")
        == {"path": str(PREDECESSOR_OPS / "CONTRACT.json"), "sha256": PREDECESSOR_CONTRACT_SHA}
        and intent.get("validation") is None,
        "Unique v2 predecessor submission differs",
    )
    rows = [
        line.split("|")
        for line in old.query(
            ["sacct", "-n", "-P", "-j", "234384", "--format=JobIDRaw,State,ExitCode"]
        ).splitlines()
    ]
    main = [row for row in rows if row and row[0] == "234384"]
    require(
        len(main) == 1 and main[0][1:3] == ["FAILED", "1:0"],
        "Predecessor must be terminal controlled FAILED 1:0; late PASS blocks v3",
    )
    require(
        "234384" not in old.query(["squeue", "-h", "-u", USER, "-o", "%i"]).split(),
        "Predecessor still queued",
    )
    for task in TASKS:
        require(
            not (ROOT / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task / "COMPLETE.json").exists(),
            "Original b1 task already complete",
        )
    return disposition


def validation_terminal(old, actual, contract_path, contract_sha256, validation_ref):
    """A receipt alone cannot authorize a race with unfinished cold verification."""
    require(validation_ref is not None, "Actual immutable validation required")
    require(Path(validation_ref["path"]) == OPS / "VALIDATION/RESULT.json", "Wrong validation path")
    result = small_ref(OPS, validation_ref)
    require(
        result.get("schema") == "srf6-memory-storage-validation-v3"
        and result.get("contract_sha256") == contract_sha256
        and not (OPS / "VALIDATION/FAILURE.json").exists()
        and not (OPS / "validate_FAILURE.json").exists(),
        "Failed or unrelated validation",
    )
    expected_verifier = Path(actual["deployment"]) / "src/sr_f6_memory_repair/validation.py"
    require(
        result["verifier_source_ref"]
        == dict(path=str(expected_verifier), sha256=sha(expected_verifier)),
        "Validation verifier identity differs",
    )
    # Tensors are inspected by the cold verifier inside the recovery allocation.
    for name in ("evidence_ref", "tensors_ref"):
        ref = result[name]
        require(
            set(ref) == {"path", "sha256"} and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "Malformed validation artifact ref",
        )
        path = canonical(ref["path"])
        require(path.parent == OPS / "VALIDATION" and path.is_file(), "Validation artifact absent")
    receipt = read(OPS / "validate_SUBMISSION.json")
    intent = read(OPS / "validate_ATTEMPT.json")
    complete = read(OPS / "validate_COMPLETE.json")
    job = receipt.get("job_id", "")
    require(
        receipt.get("status") == "SUBMITTED"
        and re.fullmatch(r"[0-9]+", job)
        and receipt.get("source") == actual
        and receipt.get("attempt_sha256") == sha(OPS / "validate_ATTEMPT.json")
        and intent.get("source") == actual
        and intent.get("contract") == dict(path=str(contract_path), sha256=contract_sha256)
        and complete.get("source") == actual
        and complete.get("job_id") == job
        and complete.get("completed") == [validation_ref],
        "Validation operation closure incomplete",
    )
    rows = [
        line.split("|")
        for line in old.query(
            ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"]
        ).splitlines()
    ]
    main = [row for row in rows if row and row[0] == job]
    require(
        len(main) == 1 and main[0][1:3] == ["COMPLETED", "0:0"],
        "Validation job not successful terminal",
    )
    require(
        job not in old.query(["squeue", "-h", "-u", USER, "-o", "%i"]).split(),
        "Validation job still queued",
    )
    return dict(status="TERMINAL_RECEIPT_ONLY_NOT_ADMISSION", job_id=job)


@contextmanager
def lock(path, *, existing=False):
    path = canonical(path, exists=existing)
    mode = "rb" if existing else "a+b"
    with path.open(mode) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def quiescent(query):
    """Require the failed b1 Slurm step terminal and all existing b1 locks free."""
    rows = [
        line.split("|")
        for line in query(
            ["sacct", "-n", "-P", "-j", "233996", "--format=JobIDRaw,State,ExitCode"]
        ).splitlines()
        if line.strip()
    ]
    step = [r for r in rows if r[0] == "233996.1"]
    require(
        len(step) == 1 and step[0][1] == "FAILED" and step[0][2] == "1:0",
        "Original b1 step is not the preserved terminal failure",
    )
    live_steps = query(["squeue", "-h", "--steps", "-u", USER, "-o", "%i"]).split()
    require("233996.1" not in live_steps, "Original b1 step still live")
    jobs = ROOT / "PHASE_RUNS/PILOT_V3/JOBS"
    failure = read(jobs / "WORKERS/A/1/ATTEMPTS/A-233996/FAILURE.json")
    require(
        failure.get("status") == "FAILED_PRESERVED"
        and failure.get("completed") == []
        and failure.get("sampling_retry_authorized") is False,
        "Original worker failure changed",
    )
    locks = [
        jobs / "WORKERS/A/1/.worker.lock",
        jobs / "TASKS/PRIMARY-b1-U10/.task.lock",
        ROOT / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/b1/r0/U10/.run.lock",
    ]
    with ExitStack() as stack:
        for path in locks:
            stack.enter_context(lock(path, existing=True))
    return dict(
        original_job="233996",
        original_step="233996.1",
        step_state="FAILED",
        original_b0="UNTOUCHED",
        target_locks="OBSERVED_FREE",
    )


def argv_for(actual, action, contract_path, contract_sha256, validation_ref=None):
    require(action in NAMES, "Only validate or resume actions are registered")
    exports = dict(
        PATH="/usr/bin:/bin",
        SRF6_MEMORY_ACTION=action,
        SRF6_MEMORY_REVISION=actual["revision"],
        SRF6_MEMORY_DEPLOYMENT=actual["deployment"],
        SRF6_MEMORY_BINDING=actual["binding_sha256"],
        SRF6_MEMORY_CONTRACT=str(contract_path),
        SRF6_MEMORY_CONTRACT_SHA=contract_sha256,
    )
    if validation_ref is not None:
        exports.update(
            SRF6_MEMORY_VALIDATION=validation_ref["path"],
            SRF6_MEMORY_VALIDATION_SHA=validation_ref["sha256"],
        )
    require(
        all("," not in value and "\n" not in value for value in exports.values()),
        "Unsafe Slurm export value",
    )
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--job-name=" + NAMES[action],
        "--account=rose",
        "--partition=cluster02",
        "--qos=" + QOS,
        "--gres=gpu:pro6000:1",
        "--cpus-per-task=4",
        "--mem=33G",
        "--time=23:00:00",
        "--nodes=1",
        "--chdir=" + str(CODE),
        "--output=" + str(OPS / (action + "-%j.log")),
        "--export=" + ",".join(f"{k}={v}" for k, v in exports.items()),
        str(Path(actual["deployment"]) / LAUNCHER),
    ]


def submit(actual, action, contract_path, contract_sha256, validation_ref=None):
    require(getpass.getuser() == USER, "Wrong scheduler account")
    old = original()
    old.clean_submission_environment()
    facts = old.metadata()
    capacity_module, _ = old.scheduler(facts)
    metadata = submission_metadata(actual, contract_path, contract_sha256, facts)
    prior_allocation_terminal(old)
    if action == "resume":
        validation_terminal(old, actual, contract_path, contract_sha256, validation_ref)
    else:
        require(action == "validate" and validation_ref is None, "Wrong validation request")
    canonical(OPS, exists=False).mkdir(parents=True, exist_ok=True)
    with predecessor_locks(), lock(OPS / (action + ".lock")):
        predecessor_terminal(old, actual, contract_path, contract_sha256)
        require(
            not any(
                (OPS / (action + suffix)).exists()
                for suffix in (
                    "_ATTEMPT.json",
                    "_SUBMISSION.json",
                    "_STARTED.json",
                    "_FAILURE.json",
                    "_COMPLETE.json",
                )
            ),
            "Prior or partial action forbids automatic retry",
        )
        require(
            not (OPS / ("VALIDATION" if action == "validate" else "EXECUTIONS")).exists(),
            "Prior partial action output forbids automatic retry",
        )
        quiet = quiescent(old.query)
        snapshot = capacity_module.capacity(query=old.query)
        require(snapshot["gpu_limit"] == 8 and snapshot["submit_limit"] == 5, "Changed limits")
        if snapshot["reserved_gpus"] + 1 > 8 or snapshot["submitted_by_user"] + 1 > 5:
            return dict(status="NO_LEGAL_CAPACITY", capacity=snapshot)
        argv = argv_for(actual, action, contract_path, contract_sha256, validation_ref)
        intent = dict(
            status="SUBMISSION_INTENT_NOT_COMPLETION",
            at=old.now(),
            action=action,
            source=actual,
            contract=dict(path=str(contract_path), sha256=contract_sha256),
            validation=validation_ref,
            capacity=snapshot,
            quiescence=quiet,
            argv=argv,
            requested_gpus=1,
            verification=metadata,
            reason="One independent b1 repair lane; no extra task selected",
        )
        old.write_once(OPS / (action + "_ATTEMPT.json"), intent)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=25, check=False)
            job_id = result.stdout.strip().split(";", 1)[0]
            valid = result.returncode == 0 and re.fullmatch(r"[0-9]+", job_id) is not None
            receipt = dict(
                status="SUBMITTED" if valid else "FAILED_OR_UNKNOWN",
                job_id=job_id if valid else None,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            receipt = dict(status="SUBMISSION_UNKNOWN", job_id=None, error=str(exc))
        receipt.update(
            at=old.now(),
            action=action,
            source=actual,
            attempt_sha256=sha(OPS / (action + "_ATTEMPT.json")),
        )
        old.write_once(OPS / (action + "_SUBMISSION.json"), receipt)
        return receipt


def allocation(old, action, job_id):
    fields = dict(
        re.findall(
            r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)",
            old.query(["scontrol", "show", "job", "-o", job_id]),
        )
    )
    require(
        fields.get("JobId") == job_id
        and fields.get("JobName") == NAMES[action]
        and fields.get("QOS") == QOS
        and fields.get("NumCPUs") == "4"
        and fields.get("Partition") == "cluster02"
        and fields.get("Account") == "rose"
        and fields.get("UserId", "").startswith(USER + "("),
        "Wrong actual allocation",
    )
    for key in ("ReqTRES", "AllocTRES"):
        tres = dict(item.split("=", 1) for item in fields.get(key, "").split(",") if "=" in item)
        require(
            tres.get("gres/gpu:pro6000") == "1"
            and tres.get("gres/gpu", "1") == "1"
            and tres.get("cpu") == "4"
            and tres.get("mem") in ("33G", "33792M"),
            "Wrong actual GPU/CPU/host-memory allocation",
        )
    return fields


def launch(actual, action, contract_path, contract_sha256, validation_ref=None):
    require(
        action in NAMES and os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
        "Unregistered/requeued launch",
    )
    job_id = os.environ.get("SLURM_JOB_ID", "")
    require(
        re.fullmatch(r"[0-9]+", job_id) is not None
        and os.environ.get("SLURM_JOB_NAME") == NAMES[action],
        "Existing matching job required",
    )
    old = original()
    with lock(OPS / (action + "_LAUNCH.lock")):
        require(
            not any(
                (OPS / (action + suffix)).exists()
                for suffix in ("_STARTED.json", "_FAILURE.json", "_COMPLETE.json")
            ),
            "Prior launch cannot be repeated",
        )
        try:
            receipt_path = OPS / (action + "_SUBMISSION.json")
            deadline = time.monotonic() + 30
            while not receipt_path.exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            receipt = read(receipt_path)
            intent = read(OPS / (action + "_ATTEMPT.json"))
            require(
                receipt.get("status") == "SUBMITTED"
                and receipt.get("job_id") == job_id
                and receipt.get("source") == actual
                and receipt.get("attempt_sha256") == sha(OPS / (action + "_ATTEMPT.json")),
                "Wrong/partial original submission receipt",
            )
            require(
                intent.get("source") == actual
                and intent.get("action") == action
                and intent.get("contract") == dict(path=str(contract_path), sha256=contract_sha256)
                and intent.get("validation") == validation_ref
                and intent.get("argv")
                == argv_for(actual, action, contract_path, contract_sha256, validation_ref),
                "Launch differs from unique submitted request",
            )
            hardware = allocation(old, action, job_id)
            quiescent(old.query)
            facts = old.metadata()
            submission_metadata(actual, contract_path, contract_sha256, facts)
            prior_allocation_terminal(old)
            with predecessor_locks():
                predecessor_terminal(old, actual, contract_path, contract_sha256)
            if action == "resume":
                validation_terminal(old, actual, contract_path, contract_sha256, validation_ref)
            cache = canonical(OPS / ("job-" + job_id), exists=False)
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
                "Exactly one assigned GPU required",
            )
            old.write_once(
                OPS / (action + "_STARTED.json"),
                dict(at=old.now(), job_id=job_id, source=actual, allocation=hardware),
            )
            common = ["--contract", str(contract_path), "--contract-sha256", contract_sha256]
            if action == "validate":
                commands = [
                    (
                        "sr_f6_memory_repair.validation",
                        ["run", *common, "--output", str(OPS / "VALIDATION")],
                    )
                ]
            else:
                require(validation_ref is not None, "Verified actual validation required")
                commands = [
                    (
                        "sr_f6_memory_repair.execution",
                        [
                            "run",
                            *common,
                            "--validation-path",
                            validation_ref["path"],
                            "--validation-sha256",
                            validation_ref["sha256"],
                            "--task-id",
                            task,
                            "--attempt-id",
                            "MEM-" + job_id,
                        ],
                    )
                    for task in TASKS
                ]
            completed = []
            for module, args in commands:
                bootstrap = (
                    "import runpy,sys;sys.path[:0]="
                    + repr([str(CODE / "src"), str(Path(actual["deployment"]) / "src")])
                    + ";sys.argv=["
                    + repr(module)
                    + "]+sys.argv[1:];runpy.run_module("
                    + repr(module)
                    + ',run_name="__main__")'
                )
                result = subprocess.run(
                    [str(PYTHON), "-I", "-B", "-u", "-c", bootstrap, *args],
                    env=env,
                    cwd=CODE,
                    check=False,
                )
                require(
                    result.returncode == 0, "Repair child failed; original/new attempt preserved"
                )
                if action == "validate":
                    paths = [OPS / "VALIDATION/RESULT.json"]
                else:
                    task = args[args.index("--task-id") + 1]
                    paths = [
                        ROOT / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task / "COMPLETE.json",
                        Path(contract_path).parent / "EXECUTIONS" / task / "COMPLETE.json",
                    ]
                completed.extend(dict(path=str(path), sha256=sha(path)) for path in paths)
            old.write_once(
                OPS / (action + "_COMPLETE.json"),
                dict(
                    at=old.now(),
                    job_id=job_id,
                    status="OPERATION_COMPLETE_NOT_SCIENTIFIC_PASS",
                    source=actual,
                    contract=dict(path=str(contract_path), sha256=contract_sha256),
                    validation=validation_ref,
                    completed=completed,
                    queue=list(TASKS) if action == "resume" else [],
                    original_worker_failure_preserved=True,
                ),
            )
        except BaseException as exc:
            old.write_once(
                OPS / (action + "_FAILURE.json"),
                dict(
                    at=old.now(),
                    job_id=job_id,
                    status="FAILED_PRESERVED",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    retry_authorized=False,
                ),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("submit", "launch"))
    parser.add_argument("--action", required=True, choices=tuple(NAMES))
    for name in ("revision", "deployment", "binding-sha256", "contract", "contract-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--validation-path")
    parser.add_argument("--validation-sha256")
    args = parser.parse_args()
    require(
        bool(args.validation_path) == bool(args.validation_sha256),
        "Complete validation ref required",
    )
    actual = identity(args.revision, args.deployment, args.binding_sha256)
    validation_ref = (
        dict(path=args.validation_path, sha256=args.validation_sha256)
        if args.validation_path
        else None
    )
    result = (submit if args.operation == "submit" else launch)(
        actual, args.action, canonical(args.contract), args.contract_sha256, validation_ref
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
