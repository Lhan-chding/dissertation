"""Unique full-batch validation; no training, sampling, or automatic retry."""

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
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V7_20261030"
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
JOB_NAME = "LH-SRF6-Memory-Full"
SCRIPT = "scripts/sr_f6_memory_full_ops/operations.py"
LAUNCHER = "scripts/sr_f6_memory_full_ops/launch.sbatch"
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
    "src/sr_f6_memory_repair/full_contract.py",
    "src/sr_f6_memory_repair/validation_full.py",
)

V6_OPS = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V6_20261030"
V6_REVISION = "1af637e0b4aaded4e444a4904091ec309927a9ff"
V6_DEPLOYMENT = PROJECT / "sr_f6_memory_smoke_reuse_code_1af637e"
V6_BINDING = "95f4b5c31c4e42c35afb3734815c5b0328a765a8b934b81468982da8c87ab50a"
V6_CONTRACT_SHA = "ec1d146141f67794b0bb4c78a9a8164e5c67038044f738e3dad19f758f6b150c"
V6_RESULT_SHA = "0a5b59becd74a3286be5c16d2dcc19937c340a28b3e2c05bcdde801739b83aca"


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
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Full full revision required")
    target = PROJECT / ("sr_f6_memory_full_code_" + revision[:7])
    require(
        Path(deployment) == target and Path(__file__).resolve() == target / SCRIPT,
        "Wrong actual full operator origin",
    )
    require(sha(target / "FULL_SOURCE.json") == binding_sha256, "Full source binding changed")
    manifest = read(target / "FULL_SOURCE.json")
    require(
        manifest.get("schema") == "srf6-memory-full-source-v1"
        and manifest.get("revision") == revision
        and manifest.get("deployment") == str(target)
        and manifest.get("producer_revision") == PRODUCER
        and set(manifest.get("files", {})) == set(MEMBERS),
        "Exact full source closure required",
    )
    for name, pin in manifest["files"].items():
        path = canonical(target / name)
        require(
            set(pin) == {"bytes", "sha256"}
            and path.stat().st_size == pin["bytes"]
            and sha(path) == pin["sha256"],
            "Full source bytes changed",
        )
    return dict(revision=revision, deployment=str(target), binding_sha256=binding_sha256)


def frozen_v6():
    require(sha(V6_DEPLOYMENT / "SMOKE_SOURCE.json") == V6_BINDING, "Frozen v6 binding changed")
    manifest = read(V6_DEPLOYMENT / "SMOKE_SOURCE.json")
    name = "scripts/sr_f6_memory_smoke_reuse_ops/operations.py"
    source = V6_DEPLOYMENT / name
    require(sha(source) == manifest["files"][name]["sha256"], "Frozen v6 operator changed")
    module = load_file("srf6_full_frozen_v6", source)
    return module, module.identity(V6_REVISION, str(V6_DEPLOYMENT), V6_BINDING)


def frozen_v4():
    module, _ = frozen_v6()
    return module.frozen_v4()


def contract_module(actual):
    sys.path.insert(0, str(Path(actual["deployment"]) / "src"))
    module = importlib.import_module("sr_f6_memory_repair.full_contract")
    require(
        Path(module.__file__).resolve()
        == Path(actual["deployment"]) / "src/sr_f6_memory_repair/full_contract.py",
        "Wrong actual full verifier source",
    )
    return module


def metadata(actual, contract_path, contract_sha256):
    require(Path(contract_path) == OPS / "CONTRACT.json", "Unique full contract required")
    contract = contract_module(actual).validate_metadata(contract_path, contract_sha256)
    require(
        contract["repair_source"]["revision"] == actual["revision"]
        and contract["repair_source"]["deployment"] == actual["deployment"],
        "Full contract source differs",
    )
    manifest = read(Path(actual["deployment"]) / "FULL_SOURCE.json")
    expected = {name: pin["sha256"] for name, pin in manifest["files"].items()}
    refs = contract["repair_source"]["files"]
    observed = {ref["path"]: ref["sha256"] for ref in refs}
    require(
        len(refs) == len(observed) and observed == expected, "Exact full source closure required"
    )
    return contract


@contextmanager
def transition_locks(legacy):
    module, _ = frozen_v6()
    with (
        module.transition_locks(legacy),
        legacy.lock(V6_OPS / "smoke.lock", existing=True),
        legacy.lock(V6_OPS / "smoke_LAUNCH.lock", existing=True),
    ):
        yield


def predecessor(legacy, old, contract):
    module, actual = frozen_v6()
    cr = dict(path=str(V6_OPS / "CONTRACT.json"), sha256=V6_CONTRACT_SHA)
    rr = dict(path=str(V6_OPS / "SMOKE/RESULT.json"), sha256=V6_RESULT_SHA)
    require(
        contract.get("prior_smoke_contract_ref") == cr
        and contract.get("prior_smoke_result_ref") == rr,
        "Exact v6 predecessor required",
    )
    prior = small_ref(V6_OPS, cr)
    module.predecessor(legacy, old, prior)
    for name in (
        "smoke_FAILURE.json",
        "SMOKE/FAILURE.json",
        "smoke_WATCHDOG_ATTEMPT.json",
        "smoke_WATCHDOG_RESULT.json",
    ):
        require(not (V6_OPS / name).exists(), "V6 failure/watchdog blocks full validation")
    intent = read(V6_OPS / "smoke_ATTEMPT.json")
    sub = read(V6_OPS / "smoke_SUBMISSION.json")
    complete = read(V6_OPS / "smoke_COMPLETE.json")
    require(
        sub.get("status") == "SUBMITTED"
        and sub.get("job_id") == "235771"
        and sub.get("source") == actual
        and intent.get("source") == actual
        and sub.get("attempt_sha256") == sha(V6_OPS / "smoke_ATTEMPT.json")
        and intent.get("contract") == cr,
        "Unique v6 submission differs",
    )
    require(
        complete.get("status") == "COMPLETED_STAGED_SMOKE_NOT_FULL_BATCH_ADMISSION"
        and complete.get("job_id") == "235771"
        and complete.get("source") == actual
        and complete.get("contract_ref") == cr
        and complete.get("result_ref") == rr,
        "V6 completion differs",
    )
    require(
        contract.get("prior_smoke_complete_ref")
        == dict(
            path=str(V6_OPS / "smoke_COMPLETE.json"), sha256=sha(V6_OPS / "smoke_COMPLETE.json")
        ),
        "V6 complete reference differs",
    )
    result = small_ref(V6_OPS, rr)
    cold = contract["prior_smoke_cold_ref"]
    require(
        cold == result["cold_verification_ref"]
        and Path(cold["path"]) == V6_OPS / "SMOKE/COLD_VERIFICATION.json",
        "V6 cold proof differs",
    )
    proof = small_ref(V6_OPS, cold)
    require(
        proof.get("status") == "VERIFIED_ACTUAL_TWO_ROW_REUSED_REFERENCE_SMOKE",
        "V6 cold numerical proof missing",
    )
    require(result.get("admission") == "NOT_FULL_BATCH_ADMISSION", "Preserve v6 admission boundary")
    rows = [
        line.split("|")
        for line in old.query(
            ["sacct", "-n", "-P", "-j", "235771", "--format=JobIDRaw,State,ExitCode"]
        ).splitlines()
    ]
    rows = [row for row in rows if row and row[0] == "235771"]
    require(
        len(rows) == 1 and rows[0][1:3] == ["COMPLETED", "0:0"], "V6 must be actual COMPLETED 0:0"
    )
    require(
        "235771" not in old.query(["squeue", "-h", "-u", USER, "-o", "%i"]).split(),
        "V6 still queued",
    )
    return complete


def argv_for(actual, contract_path, contract_sha256):
    exports = dict(
        PATH="/usr/bin:/bin",
        SRF6_FULL_REVISION=actual["revision"],
        SRF6_FULL_DEPLOYMENT=actual["deployment"],
        SRF6_FULL_BINDING=actual["binding_sha256"],
        SRF6_FULL_CONTRACT=str(contract_path),
        SRF6_FULL_CONTRACT_SHA=contract_sha256,
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
        "--output=" + str(OPS / "validate-%j.log"),
        "--export=" + ",".join(k + "=" + v for k, v in exports.items()),
        str(Path(actual["deployment"]) / LAUNCHER),
    ]


def submit(actual, contract_path, contract_sha256):
    require(getpass.getuser() == USER, "Wrong scheduler account")
    legacy, _ = frozen_v4()
    old = legacy.original()
    old.clean_submission_environment()
    contract = metadata(actual, contract_path, contract_sha256)
    with transition_locks(legacy), legacy.lock(OPS / "validate.lock"):
        predecessor(legacy, old, contract)
        require(
            not any(
                (OPS / ("validate_" + s + ".json")).exists()
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
            and not (OPS / "VALIDATION").exists(),
            "Prior or partial full validation forbids retry",
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
            OPS / "validate_ATTEMPT.json",
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
        result.update(
            source=actual, at=old.now(), attempt_sha256=sha(OPS / "validate_ATTEMPT.json")
        )
        old.write_once(OPS / "validate_SUBMISSION.json", result)
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
        "Wrong actual full validation allocation",
    )
    for key in ("ReqTRES", "AllocTRES"):
        tres = dict(item.split("=", 1) for item in fields.get(key, "").split(",") if "=" in item)
        require(
            tres.get("gres/gpu:pro6000") == "1"
            and tres.get("gres/gpu", "1") == "1"
            and tres.get("cpu") == "4"
            and tres.get("mem") in ("33G", "33792M"),
            "Wrong actual full validation resources",
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
    with legacy.lock(OPS / "validate_LAUNCH.lock"):
        require(
            not any(
                (OPS / ("validate_" + s + ".json")).exists()
                for s in ("STARTED", "COMPLETE", "FAILURE", "WATCHDOG_ATTEMPT", "WATCHDOG_RESULT")
            )
            and not (OPS / "VALIDATION").exists(),
            "Prior launch cannot repeat",
        )
        try:
            deadline = time.monotonic() + 30
            while not (OPS / "validate_SUBMISSION.json").exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            receipt = read(OPS / "validate_SUBMISSION.json")
            intent = read(OPS / "validate_ATTEMPT.json")
            require(
                receipt.get("status") == "SUBMITTED"
                and receipt.get("job_id") == job
                and receipt.get("source") == actual
                and receipt.get("attempt_sha256") == sha(OPS / "validate_ATTEMPT.json")
                and intent.get("source") == actual
                and intent.get("contract") == dict(path=str(contract_path), sha256=contract_sha256)
                and intent.get("argv") == argv_for(actual, contract_path, contract_sha256),
                "Wrong/partial full validation submission",
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
                OPS / "validate_STARTED.json",
                dict(at=old.now(), job_id=job, source=actual, allocation=hardware),
            )
            bootstrap = (
                "import runpy,sys;sys.path[:0]="
                + repr([str(CODE / "src"), str(Path(actual["deployment"]) / "src")])
                + ";sys.argv=['sr_f6_memory_repair.validation_full']+sys.argv[1:];"
                "runpy.run_module('sr_f6_memory_repair.validation_full',run_name='__main__')"
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
                str(OPS / "VALIDATION"),
            ]
            process = subprocess.run(argv, env=env, check=False)
            require(
                process.returncode == 0,
                "Full validation child failed; preserved, no automatic retry",
            )
            result_path = OPS / "VALIDATION/RESULT.json"
            ref = dict(path=str(result_path), sha256=sha(result_path))
            proof = contract_module(actual).verify_validation_metadata(
                contract_path, contract_sha256, ref
            )
            require(
                proof["status"] == "VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
                "Full numerical validation proof missing",
            )
            complete = dict(
                status="COMPLETED_FULL_VALIDATION_NO_TRAINING",
                at=old.now(),
                job_id=job,
                source=actual,
                result_ref=ref,
                cold_verification_ref=proof["cold_verification_ref"],
                contract_ref=dict(path=str(contract_path), sha256=contract_sha256),
            )
            old.write_once(OPS / "validate_COMPLETE.json", complete)
            return complete
        except BaseException as exc:
            old.write_once(
                OPS / "validate_FAILURE.json",
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


def validation_terminal(old, actual, contract_path, contract_sha256, result_ref):
    """Metadata-only delivery gate; never rerun tensor mathematics on a login node."""
    metadata(actual, contract_path, contract_sha256)
    for name in (
        "validate_FAILURE.json",
        "VALIDATION/FAILURE.json",
        "validate_WATCHDOG_ATTEMPT.json",
        "validate_WATCHDOG_RESULT.json",
    ):
        require(not (OPS / name).exists(), "Failure/unknown watchdog blocks delivery")
    intent = read(OPS / "validate_ATTEMPT.json")
    submission = read(OPS / "validate_SUBMISSION.json")
    complete = read(OPS / "validate_COMPLETE.json")
    job = submission.get("job_id", "")
    cr = dict(path=str(contract_path), sha256=contract_sha256)
    require(
        re.fullmatch(r"[0-9]+", job) is not None
        and submission.get("status") == "SUBMITTED"
        and submission.get("source") == actual
        and intent.get("source") == actual
        and submission.get("attempt_sha256") == sha(OPS / "validate_ATTEMPT.json")
        and intent.get("contract") == cr
        and intent.get("argv") == argv_for(actual, contract_path, contract_sha256),
        "Unique full submission differs",
    )
    require(
        complete.get("status") == "COMPLETED_FULL_VALIDATION_NO_TRAINING"
        and complete.get("job_id") == job
        and complete.get("source") == actual
        and complete.get("contract_ref") == cr
        and complete.get("result_ref") == result_ref,
        "Full completion differs",
    )
    proof = contract_module(actual).verify_validation_metadata(
        contract_path, contract_sha256, result_ref
    )
    require(
        proof.get("status") == "VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH"
        and complete.get("cold_verification_ref") == proof["cold_verification_ref"],
        "Complete must bind actual cold proof",
    )
    rows = [
        line.split("|")
        for line in old.query(
            ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"]
        ).splitlines()
    ]
    rows = [row for row in rows if row and row[0] == job]
    require(
        len(rows) == 1 and rows[0][1:3] == ["COMPLETED", "0:0"],
        "Full job must be actual COMPLETED 0:0",
    )
    require(
        job not in old.query(["squeue", "-h", "-u", USER, "-o", "%i"]).split(),
        "Full job still queued",
    )
    return dict(
        job_id=job,
        state="COMPLETED",
        exit_code="0:0",
        queue_gone=True,
        result_ref=result_ref,
        cold_verification_ref=proof["cold_verification_ref"],
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
