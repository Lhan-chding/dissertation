"""Unique staged memory qualification; no training, sampling, or full-batch admission."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V6_20261030"
V4_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V4_20261030"
V4_REVISION = "c84c275fd3e2c51fa7c0045aa62409e20edeebd3"
V4_DEPLOYMENT = PROJECT / "sr_f6_memory_repair_code_c84c275"
V4_BINDING = "2151d1d7c8273ee751ec6d4249eeacb753d5d1b24486018f9c013a24e8845af4"
V4_CONTRACT_SHA = "dfd8d792aed9fb2dc0d9a33ed4962afefb410609b974c92a34659efccc444a34"
PRODUCER = "b7e7b067f237b360b1dfb2c3075d3f14be7c281f"
CODE = PROJECT / "sr_f6_code_b7e7b06"
PYTHON = PROJECT / "envs/ssvc-py312/bin/python3.12"
USER = "varun024"
QOS = "soujanya-poria-startfund-2026-03"
JOB_NAME = "LH-SRF6-Memory-Reuse"
SCRIPT = "scripts/sr_f6_memory_smoke_reuse_ops/operations.py"
LAUNCHER = "scripts/sr_f6_memory_smoke_reuse_ops/launch.sbatch"
MEMBERS = (
    SCRIPT,
    LAUNCHER,
    "src/sr_f6_memory_repair/__init__.py",
    "src/sr_f6_memory_repair/execution.py",
    "src/sr_f6_memory_repair/validation.py",
    "src/sr_f6_memory_repair/offload.py",
    "src/sr_f6_memory_repair/arena.py",
    "src/sr_f6_memory_repair/residency.py",
    "src/sr_f6_memory_repair/smoke.py",
    "src/sr_f6_memory_repair/smoke_reuse.py",
)

V5_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V5_20261030"
V5_REVISION = "3bbcc88c5ac15edd8de15f304c3874c96cbcf299"
V5_DEPLOYMENT = PROJECT / "sr_f6_memory_smoke_code_3bbcc88"
V5_BINDING = "118072e2b4522c28a36edf6e328728ba6922016f43e01853e45cc0b6d3f0c9b3"
V5_CONTRACT_SHA = "d8031ca63eab7cb17ed6b9272b03dab72b3fd42b9ca057cee853680fcba5deb7"
V5_RESULT_SHA = "be1f794c8ce608ac9c6b1943e14ddae9086c486322308edcbadba81900fc9d37"


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


def frozen_v4():
    require(
        sha(V4_DEPLOYMENT / "MEMORY_REPAIR_SOURCE.json") == V4_BINDING, "Frozen v4 binding changed"
    )
    manifest = read(V4_DEPLOYMENT / "MEMORY_REPAIR_SOURCE.json")
    source = V4_DEPLOYMENT / "scripts/sr_f6_memory_repair_ops/operations.py"
    require(
        sha(source) == manifest["files"]["scripts/sr_f6_memory_repair_ops/operations.py"]["sha256"],
        "Frozen v4 operator changed",
    )
    module = load_file("srf6_smoke_frozen_v4", source)
    actual = module.identity(V4_REVISION, str(V4_DEPLOYMENT), V4_BINDING)
    return module, actual


def identity(revision, deployment, binding_sha256):
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Full smoke revision required")
    target = PROJECT / ("sr_f6_memory_smoke_reuse_code_" + revision[:7])
    require(
        Path(deployment) == target and Path(__file__).resolve() == target / SCRIPT,
        "Wrong actual smoke operator origin",
    )
    require(sha(target / "SMOKE_SOURCE.json") == binding_sha256, "Smoke source binding changed")
    manifest = read(target / "SMOKE_SOURCE.json")
    require(
        manifest.get("schema") == "srf6-memory-smoke-reuse-source-v1"
        and manifest.get("revision") == revision
        and manifest.get("deployment") == str(target)
        and manifest.get("producer_revision") == PRODUCER
        and set(manifest.get("files", {})) == set(MEMBERS),
        "Exact smoke source closure required",
    )
    for name, pin in manifest["files"].items():
        path = canonical(target / name)
        require(
            set(pin) == {"bytes", "sha256"}
            and path.stat().st_size == pin["bytes"]
            and sha(path) == pin["sha256"],
            "Smoke source bytes changed",
        )
    return dict(revision=revision, deployment=str(target), binding_sha256=binding_sha256)


def smoke_module(actual):
    sys.path.insert(0, str(Path(actual["deployment"]) / "src"))
    module = importlib.import_module("sr_f6_memory_repair.smoke_reuse")
    require(
        Path(module.__file__).resolve()
        == Path(actual["deployment"]) / "src/sr_f6_memory_repair/smoke_reuse.py",
        "Wrong smoke implementation origin",
    )
    return module


def frozen_v5():
    require(sha(V5_DEPLOYMENT / "SMOKE_SOURCE.json") == V5_BINDING, "Frozen v5 binding changed")
    manifest = read(V5_DEPLOYMENT / "SMOKE_SOURCE.json")
    source = V5_DEPLOYMENT / "scripts/sr_f6_memory_smoke_ops/operations.py"
    require(
        sha(source) == manifest["files"]["scripts/sr_f6_memory_smoke_ops/operations.py"]["sha256"],
        "Frozen v5 operator changed",
    )
    module = load_file("srf6_reuse_frozen_v5", source)
    actual = module.identity(V5_REVISION, str(V5_DEPLOYMENT), V5_BINDING)
    return module, actual


def metadata(actual, contract_path, contract_sha256):
    require(Path(contract_path) == OPS / "CONTRACT.json", "Unique reuse contract required")
    contract = small_ref(OPS, dict(path=str(contract_path), sha256=contract_sha256))
    require(
        contract.get("schema") == "srf6-memory-smoke-reuse-contract-v1"
        and contract.get("experiment_root") == str(ROOT)
        and contract.get("max_host_bytes") == 8 * 2**30
        and contract.get("max_disk_bytes") == 96 * 2**30
        and contract.get("max_device_bytes") == 44 * 2**30
        and contract.get("storage_backend") == "arena_v3",
        "Frozen reuse bounds differ",
    )
    module = smoke_module(actual)
    require(contract.get("smoke_plan") == module.SMOKE_PLAN, "Frozen reuse plan differs")
    require(
        contract.get("prior_smoke_contract_ref")
        == dict(path=str(V5_OPS / "CONTRACT.json"), sha256=V5_CONTRACT_SHA),
        "Exact v5 contract required",
    )
    require(
        contract.get("prior_smoke_result_ref")
        == dict(path=str(V5_OPS / "SMOKE/RESULT.json"), sha256=V5_RESULT_SHA),
        "Exact v5 result required",
    )
    prior = small_ref(V5_OPS, contract["prior_smoke_contract_ref"])
    for key in (
        "experiment_root",
        "producer",
        "original_ready_ref",
        "original_audit_ref",
        "step0_commit_ref",
        "original_failure_ref",
        "config_ref",
        "source_binding_ref",
        "pilot_spec_ref",
    ):
        require(
            contract.get(key) == prior[key], "Original retained scientific evidence changed: " + key
        )
    manifest = read(Path(actual["deployment"]) / "SMOKE_SOURCE.json")
    require(
        contract["repair_source"]["revision"] == actual["revision"]
        and contract["repair_source"]["deployment"] == actual["deployment"],
        "Reuse source differs",
    )
    expected = {
        str(Path(actual["deployment"]) / k): v["sha256"] for k, v in manifest["files"].items()
    }
    observed = {}
    for ref in contract["repair_source"]["files"]:
        path = Path(ref["path"])
        path = path if path.is_absolute() else Path(actual["deployment"]) / path
        require(
            str(path) not in observed and sha(path) == ref["sha256"], "Duplicate or changed source"
        )
        observed[str(path)] = ref["sha256"]
    require(observed == expected, "Exact reuse source closure required")
    module.validate_contract(contract_path, contract_sha256, source=True)
    return contract


@contextmanager
def transition_locks(legacy):
    with (
        legacy.predecessor_locks(),
        legacy.lock(V4_OPS / "resume.lock"),
        legacy.lock(V4_OPS / "validate.lock", existing=True),
        legacy.lock(V5_OPS / "smoke.lock", existing=True),
        legacy.lock(V5_OPS / "smoke_LAUNCH.lock", existing=True),
    ):
        yield


def predecessor(legacy, old, contract):
    """The successful, closed v5 run is the sole reference; earlier failures stay factual."""
    module, actual = frozen_v5()
    cr = dict(path=str(V5_OPS / "CONTRACT.json"), sha256=V5_CONTRACT_SHA)
    rr = dict(path=str(V5_OPS / "SMOKE/RESULT.json"), sha256=V5_RESULT_SHA)
    require(
        contract.get("prior_smoke_contract_ref") == cr
        and contract.get("prior_smoke_result_ref") == rr,
        "Wrong frozen v5 references",
    )
    prior = small_ref(V5_OPS, cr)
    # Pure original predecessor checks do not import the old smoke module into
    # the new numerical package. The new verifier validates its bound reports.
    module.predecessor(legacy, old, prior)
    for name in (
        "smoke_FAILURE.json",
        "SMOKE/FAILURE.json",
        "smoke_WATCHDOG_ATTEMPT.json",
        "smoke_WATCHDOG_RESULT.json",
    ):
        require(not (V5_OPS / name).exists(), "V5 failure/watchdog cannot admit reference reuse")
    intent = read(V5_OPS / "smoke_ATTEMPT.json")
    submission = read(V5_OPS / "smoke_SUBMISSION.json")
    complete = read(V5_OPS / "smoke_COMPLETE.json")
    require(
        submission.get("status") == "SUBMITTED"
        and submission.get("job_id") == "235713"
        and submission.get("source") == actual
        and submission.get("attempt_sha256") == sha(V5_OPS / "smoke_ATTEMPT.json")
        and intent.get("source") == actual
        and intent.get("contract") == cr,
        "Unique v5 submission differs",
    )
    require(
        complete.get("status") == "COMPLETED_STAGED_SMOKE_NOT_FULL_BATCH_ADMISSION"
        and complete.get("job_id") == "235713"
        and complete.get("source") == actual
        and complete.get("contract_ref") == cr
        and complete.get("result_ref") == rr,
        "V5 completion binding differs",
    )
    small_ref(V5_OPS, rr)
    rows = [
        line.split("|")
        for line in old.query(
            ["sacct", "-n", "-P", "-j", "235713", "--format=JobIDRaw,State,ExitCode"]
        ).splitlines()
    ]
    rows = [row for row in rows if row and row[0] == "235713"]
    require(
        len(rows) == 1 and rows[0][1:3] == ["COMPLETED", "0:0"], "V5 must actually be COMPLETED 0:0"
    )
    require(
        "235713" not in old.query(["squeue", "-h", "-u", USER, "-o", "%i"]).split(),
        "V5 still queued",
    )
    return complete


def argv_for(actual, contract_path, contract_sha256):
    exports = dict(
        PATH="/usr/bin:/bin",
        SRF6_SMOKE_REVISION=actual["revision"],
        SRF6_SMOKE_DEPLOYMENT=actual["deployment"],
        SRF6_SMOKE_BINDING=actual["binding_sha256"],
        SRF6_SMOKE_CONTRACT=str(contract_path),
        SRF6_SMOKE_CONTRACT_SHA=contract_sha256,
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
        "--time=00:35:00",
        "--nodes=1",
        "--chdir=" + str(CODE),
        "--output=" + str(OPS / "smoke-%j.log"),
        "--export=" + ",".join(k + "=" + v for k, v in exports.items()),
        str(Path(actual["deployment"]) / LAUNCHER),
    ]


def submit(actual, contract_path, contract_sha256):
    require(getpass.getuser() == USER, "Wrong scheduler account")
    legacy, _ = frozen_v4()
    old = legacy.original()
    old.clean_submission_environment()
    contract = metadata(actual, contract_path, contract_sha256)
    with transition_locks(legacy), legacy.lock(OPS / "smoke.lock"):
        predecessor(legacy, old, contract)
        require(
            not any(
                (OPS / ("smoke_" + s + ".json")).exists()
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
            and not (OPS / "SMOKE").exists(),
            "Prior or partial smoke forbids retry",
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
            OPS / "smoke_ATTEMPT.json",
            dict(
                status="SUBMISSION_INTENT_NOT_COMPLETION",
                at=old.now(),
                source=actual,
                contract=dict(path=str(contract_path), sha256=contract_sha256),
                argv=argv,
                capacity=capacity,
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
        result.update(source=actual, at=old.now(), attempt_sha256=sha(OPS / "smoke_ATTEMPT.json"))
        old.write_once(OPS / "smoke_SUBMISSION.json", result)
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
        "Wrong actual smoke allocation",
    )
    for key in ("ReqTRES", "AllocTRES"):
        tres = dict(item.split("=", 1) for item in fields.get(key, "").split(",") if "=" in item)
        require(
            tres.get("gres/gpu:pro6000") == "1"
            and tres.get("gres/gpu", "1") == "1"
            and tres.get("cpu") == "4"
            and tres.get("mem") in ("33G", "33792M"),
            "Wrong actual smoke resources",
        )
    return fields


def run_child(argv, env, old):
    """Allow the internal alarm to restore first; interrupt, then hard stop by 30 minutes."""
    child = subprocess.Popen(argv, env=env)
    try:
        return dict(returncode=child.wait(timeout=1650), timed_out=False, killed=False)
    except subprocess.TimeoutExpired:
        old.write_once(
            OPS / "smoke_WATCHDOG_ATTEMPT.json",
            dict(
                status="DEADLINE_INTERRUPT_INTENT",
                at=old.now(),
                child_pid=child.pid,
                watchdog_interrupt_seconds=1650,
                restoration_grace_seconds=150,
                retry_authorized=False,
            ),
        )
        # Popen retains this direct child and has not reaped it: the PID cannot be
        # recycled while its wait state is outstanding. Never signal another job.
        child.send_signal(signal.SIGINT)
        killed = False
        try:
            code = child.wait(timeout=150)
        except subprocess.TimeoutExpired:
            child.kill()
            killed = True
            code = child.wait(timeout=10)
        result = dict(
            status="DEADLINE_FAILED_PRESERVED",
            at=old.now(),
            child_pid=child.pid,
            returncode=code,
            timed_out=True,
            killed=killed,
            restoration_status="UNKNOWN_AFTER_HARD_KILL"
            if killed
            else "CHECK_INNER_PARTIAL_EVIDENCE",
            retry_authorized=False,
        )
        old.write_once(OPS / "smoke_WATCHDOG_RESULT.json", result)
        return result


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
    with legacy.lock(OPS / "smoke_LAUNCH.lock"):
        require(
            not any(
                (OPS / ("smoke_" + s + ".json")).exists()
                for s in ("STARTED", "COMPLETE", "FAILURE", "WATCHDOG_ATTEMPT", "WATCHDOG_RESULT")
            ),
            "Prior launch cannot repeat",
        )
        try:
            deadline = time.monotonic() + 30
            while not (OPS / "smoke_SUBMISSION.json").exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            receipt = read(OPS / "smoke_SUBMISSION.json")
            intent = read(OPS / "smoke_ATTEMPT.json")
            require(
                receipt.get("status") == "SUBMITTED"
                and receipt.get("job_id") == job
                and receipt.get("source") == actual
                and receipt.get("attempt_sha256") == sha(OPS / "smoke_ATTEMPT.json")
                and intent.get("source") == actual
                and intent.get("contract") == dict(path=str(contract_path), sha256=contract_sha256)
                and intent.get("argv") == argv_for(actual, contract_path, contract_sha256),
                "Wrong/partial smoke submission",
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
                OPS / "smoke_STARTED.json",
                dict(at=old.now(), job_id=job, source=actual, allocation=hardware),
            )
            bootstrap = (
                "import runpy,sys;sys.path[:0]="
                + repr([str(CODE / "src"), str(Path(actual["deployment"]) / "src")])
                + ";sys.argv=['sr_f6_memory_repair.smoke_reuse']+sys.argv[1:];"
                "runpy.run_module('sr_f6_memory_repair.smoke_reuse',run_name='__main__')"
            )
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
                "--output",
                str(OPS / "SMOKE"),
            ]
            process = run_child(argv, env, old)
            require(
                process["returncode"] == 0 and not process["timed_out"],
                "Smoke child failed or exceeded deadline; preserved, no automatic retry",
            )
            result_path = OPS / "SMOKE/RESULT.json"
            ref = dict(path=str(result_path), sha256=sha(result_path))
            proof = smoke_module(actual).verify_smoke_metadata(contract_path, contract_sha256, ref)
            require(
                proof["status"] == "SMOKE_NUMERICAL_PERFORMANCE_MEASURED"
                and proof["admission"] == "NOT_FULL_BATCH_ADMISSION",
                "Smoke cannot admit full batch",
            )
            complete = dict(
                status="COMPLETED_STAGED_SMOKE_NOT_FULL_BATCH_ADMISSION",
                at=old.now(),
                job_id=job,
                source=actual,
                result_ref=ref,
                contract_ref=dict(path=str(contract_path), sha256=contract_sha256),
            )
            old.write_once(OPS / "smoke_COMPLETE.json", complete)
            return complete
        except BaseException as exc:
            old.write_once(
                OPS / "smoke_FAILURE.json",
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
