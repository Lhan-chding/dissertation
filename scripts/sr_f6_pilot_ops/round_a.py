"""One outer Round A allocation for the already frozen B7 PILOT_V3 producer.

This module does not register sources, evaluate scientific gates, sample, train,
resume, or choose tasks. Only the unchanged B7 pair launcher runs model work.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
CODE = PROJECT / "sr_f6_code_b7e7b06"
PRODUCER = "b7e7b067f237b360b1dfb2c3075d3f14be7c281f"
OPS = PROJECT / "sr_f6_operations_20261027/PILOT_V3_GPU_EXECUTION_20261029"
PREPARE = "sr_f6_operations_20261027/PILOT_V3_CLOSURE_EXECUTION_20261029"
USER = "varun024"
QOS = "soujanya-poria-startfund-2026-03"
JOB = "LH-SRF6-PilotV3-A"
PYTHON = PROJECT / "envs/ssvc-py312/bin/python3.12"
MEMBERS = ("scripts/sr_f6_pilot_ops/round_a.py", "scripts/sr_f6_pilot_ops/round_a.sbatch")
PHASE = "d9f061d1cf08e14c485083bbed421a412a29868fd8305ad4c265fb6404562284"
SPEC = "4936b29e3a897334c1c4e67aad436bf46bdea9a105354d0155484d2acc20994c"
ADMISSION = "b65831ad6ec57c829a69a9ad0a87a6aaf6a4a173e2d921cf953462ca7519b2a6"
PLAN = "56102cd5b10d93bde1996527e39fa1ec30943fc2618cab2838f03d5fd550d4f4"
ROOT_PINS = {
    "PHASES/PILOT_V3/IDENTITY.json": PHASE,
    "PHASES/PILOT_V3/SOURCE_MANIFEST.json": (
        "c6f694582b152e5ccb3e0b303f4ace50c4432ec2ac884fccfd42b37de704bb87"
    ),
    "PHASES/PILOT_V3/ADMISSION.json": ADMISSION,
    "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json": SPEC,
    "PHASE_RUNS/PILOT_V3/JOBS/PLAN.json": PLAN,
    "PHASE_RUNS/PILOT_V3/INPUTS_V3/OVERLAY_MANIFEST.json": (
        "ec20d79b104ccf7930e584a43fc2cb9a22a49fc0c9bda0d957bcdc4efdcf9316"
    ),
}
PREPARE_PINS = {
    "PILOT_PREPARE_RESULT.json": "f1ef683cfc0dc303d86d0d97b3be136a47607a68a49986fb8a353747b7cefca4",
    "CPU_PREPARE_SUBMISSION.json": (
        "7f042155b07c18575e5d658c1b9600516431ef7cb639f96f1fd3421f8ef1625f"
    ),
}
QUEUES = [
    ["PRIMARY-b0-U11", "PRIMARY-b0-C_SHUF", "PRIMARY-b0-U10"],
    ["PRIMARY-b1-U10", "PRIMARY-b1-C_SHUF", "PRIMARY-b1-U11"],
]


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(path, *, exists=True):
    path = Path(path)
    require(
        path.is_absolute() and path.is_relative_to(PROJECT) and ".." not in path.parts,
        "Canonical project path required",
    )
    require(path.resolve(strict=exists) == path, "Noncanonical project path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink forbidden")
    return path


def read_bytes(path):
    fd = os.open(canonical(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Regular file required")
        return stream.read()


def bound(path, pin):
    require(isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin), "Exact SHA256 required")
    raw = read_bytes(path)
    require(hashlib.sha256(raw).hexdigest() == pin, "Changed sealed bytes: " + str(path))
    return json.loads(raw)


def write_once(path, value):
    path = canonical(path, exists=False)
    require(not path.exists(), "Immutable operation record already exists")
    staging = canonical(
        path.parent / (".pending-" + path.name + "-" + uuid.uuid4().hex), exists=False
    )
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # A final filename always denotes a complete fsynced record. A crash leaves
    # the private staging file as evidence; a concurrent writer cannot clobber.
    os.link(staging, path, follow_symlinks=False)
    staging.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def lock(name):
    canonical(OPS, exists=False).mkdir(mode=0o700, exist_ok=True)
    fd = os.open(canonical(OPS / name, exists=False), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def operation_identity(revision, deployment, binding_sha256):
    require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "Actual full operation commit required",
    )
    expected = PROJECT / ("sr_f6_pilot_ops_code_" + revision[:7])
    require(Path(deployment) == expected, "Wrong operation deployment")
    canonical(expected)
    require(Path(__file__).resolve() == expected / MEMBERS[0], "Wrong loaded operation origin")
    value = bound(expected / "OPERATION_BINDING.json", binding_sha256)
    require(
        set(value) == {"schema", "revision", "deployment", "producer_revision", "files"}
        and value["schema"] == "srf6-pilot-round-a-operations-v1"
        and value["revision"] == revision
        and value["deployment"] == str(expected)
        and value["producer_revision"] == PRODUCER
        and set(value["files"]) == set(MEMBERS),
        "Actual separately frozen operation identity required",
    )
    for name, record in value["files"].items():
        raw = read_bytes(expected / name)
        require(
            set(record) == {"bytes", "sha256"}
            and len(raw) == record["bytes"]
            and hashlib.sha256(raw).hexdigest() == record["sha256"],
            "Operation bytes changed",
        )
    return dict(revision=revision, deployment=str(expected), binding_sha256=binding_sha256)


def metadata():
    """Small sealed metadata only; B7 performs full admission inside allocation."""
    values = {name: bound(ROOT / name, pin) for name, pin in ROOT_PINS.items()}
    prepared = {name: bound(PROJECT / PREPARE / name, pin) for name, pin in PREPARE_PINS.items()}
    identity = values["PHASES/PILOT_V3/IDENTITY.json"]
    plan = values["PHASE_RUNS/PILOT_V3/JOBS/PLAN.json"]
    spec = values["PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json"]
    admission = values["PHASES/PILOT_V3/ADMISSION.json"]
    require(
        identity["revision"] == PRODUCER
        and identity["deployment"] == str(CODE)
        and identity["root"] == str(ROOT)
        and identity["phase"] == "PILOT_V3",
        "Frozen B7 identity required",
    )
    require(
        prepared["PILOT_PREPARE_RESULT.json"]["status"] == "READY_FOR_REGISTERED_PILOT_ONLY"
        and prepared["CPU_PREPARE_SUBMISSION.json"]["job_id"] == "233935"
        and prepared["CPU_PREPARE_SUBMISSION.json"]["status"] == "SUBMITTED",
        "Actual unique preparation required",
    )
    require(
        admission["status"] == "PASS"
        and admission["scope"] == "PILOT_V3_REGISTERED_ONLY"
        and admission["phase_identity_sha256"] == PHASE
        and admission["pilot_spec_sha256"] == SPEC
        and admission["training_authorized"] is True
        and admission["formal_authorized"] is False
        and admission["test_authorized"] is False,
        "Actual bounded pilot admission required",
    )
    require(
        plan["phase"] == spec["phase"] == "PILOT_V3"
        and plan["queues"]["A"] == QUEUES
        and plan["workers"] == 2
        and plan["gpus_per_worker"] == 1,
        "Original Round A queues required",
    )
    return dict(spec=spec, plan=plan, manifest=values["PHASES/PILOT_V3/SOURCE_MANIFEST.json"])


def pristine(facts):
    jobs = canonical(ROOT / "PHASE_RUNS/PILOT_V3/JOBS")
    require(
        {p.name for p in jobs.iterdir()} == {"PLAN.json"},
        "Prior worker/task/job output forbids new Round A submission",
    )
    for item in facts["spec"]["paths"]:
        directory = canonical(ROOT / item["audit_root"])
        require(
            {p.name for p in directory.iterdir()}
            == {"ROLLOUT_REGISTRATION.json", "RUN_IDENTITY.json"},
            "Prior trainer/checkpoint/partial output forbids new Round A submission",
        )


def scheduler(facts):
    # Import only source files sealed by B7, never the new operation checkout's src.
    require(
        not any(n == "sr_f6" or n.startswith("sr_f6.") for n in sys.modules),
        "Runtime package loaded before fixed B7 selection",
    )
    members = {row["path"]: row for row in facts["manifest"]["files"]}
    required = [
        "src/sr_f6/" + name
        for name in (
            "__init__.py",
            "common.py",
            "source_binding.py",
            "interface_screen_dispatch.py",
            "confirm_dispatch.py",
        )
    ]
    required.append("scripts/sr_f6/run_pilot_v3_pair.sh")
    for name in required:
        raw = read_bytes(CODE / name)
        require(
            len(raw) == members[name]["bytes"]
            and hashlib.sha256(raw).hexdigest() == members[name]["sha256"],
            "Changed B7 entry source",
        )
    sys.path.insert(0, str(CODE / "src"))
    capacity_module = importlib.import_module("sr_f6.interface_screen_dispatch")
    state_module = importlib.import_module("sr_f6.confirm_dispatch")
    for name, module in tuple(sys.modules.items()):
        if name == "sr_f6" or name.startswith("sr_f6."):
            relative = (
                "src/" + name.replace(".", "/") + ("/__init__.py" if name == "sr_f6" else ".py")
            )
            require(Path(module.__file__).resolve() == CODE / relative, "Wrong B7 module origin")
    return capacity_module, state_module


def query(argv):
    result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=15)
    require(result.returncode == 0, "Scheduler query failed; reservation remains UNKNOWN")
    return result.stdout.strip()


def clean_submission_environment():
    forbidden = [k for k in os.environ if k.startswith(("SBATCH_", "SLURM_", "SRF6_"))]
    forbidden += [
        k
        for k in (
            "BASH_ENV",
            "ENV",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONSTARTUP",
            "PYTHONINSPECT",
        )
        if os.environ.get(k)
    ]
    require(not forbidden, "Injected scheduler/resume/process environment: " + ",".join(forbidden))
    require(os.environ.get("PYTHONOPTIMIZE", "0") in ("", "0"), "Python optimization forbidden")


def argv_for(identity):
    export = (
        "PATH,SRF6_OPS_REVISION="
        + identity["revision"]
        + ",SRF6_OPS_DEPLOYMENT="
        + identity["deployment"]
    )
    export += ",SRF6_OPS_BINDING_SHA256=" + identity["binding_sha256"]
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--job-name=" + JOB,
        "--account=rose",
        "--partition=cluster02",
        "--qos=" + QOS,
        "--gres=gpu:pro6000:2",
        "--cpus-per-task=8",
        "--mem=66G",
        "--time=23:00:00",
        "--nodes=1",
        "--chdir=" + str(CODE),
        "--output=" + str(OPS / "round-a-%j.log"),
        "--export=" + export,
        str(Path(identity["deployment"]) / MEMBERS[1]),
    ]


def submit(identity):
    require(getpass.getuser() == USER, "Wrong scheduler account")
    clean_submission_environment()
    facts = metadata()
    capacity_module, state_module = scheduler(facts)
    require(
        state_module._job_states({"233935"}) == {"233935": ("COMPLETED", "0:0")},
        "Preparation not terminal",
    )
    live = query(["squeue", "-h", "-u", USER, "-o", "%i"])
    require("233935" not in live.split(), "Preparation still present in queue")
    with lock("A_SUBMIT.lock"):
        require(
            {p.name for p in OPS.iterdir()} <= {"A_SUBMIT.lock"},
            "Existing/partial Round A operation forbids retry",
        )
        pristine(facts)
        snapshot = capacity_module.capacity(query=query)
        require(snapshot["gpu_limit"] == 8 and snapshot["submit_limit"] == 5, "Changed limits")
        if (
            not snapshot["can_submit_two_gpu_job"]
            or snapshot["reserved_gpus"] + 2 > 8
            or snapshot["submitted_by_user"] + 1 > 5
        ):
            return dict(status="NO_LEGAL_CAPACITY", capacity=snapshot)
        argv = argv_for(identity)
        intent = dict(
            status="SUBMISSION_INTENT_NOT_COMPLETION",
            at=now(),
            operation=identity,
            producer_revision=PRODUCER,
            round="A",
            requested_gpus=2,
            argv=argv,
            capacity=snapshot,
            phase_identity_sha256=PHASE,
            pilot_spec_sha256=SPEC,
            admission_sha256=ADMISSION,
            plan_sha256=PLAN,
            resume_tasks=[],
            worker_attempt_template="A-{job_id}",
        )
        write_once(OPS / "A_ATTEMPT.json", intent)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=25)
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
            at=now(),
            operation=identity,
            round="A",
            requested_gpus=2,
            attempt_sha256=hashlib.sha256(read_bytes(OPS / "A_ATTEMPT.json")).hexdigest(),
        )
        write_once(OPS / "A_SUBMISSION.json", receipt)
        return receipt


def allocation(job_id):
    detail = query(["scontrol", "show", "job", "-o", job_id])
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", detail))
    require(
        fields.get("JobId") == job_id
        and fields.get("JobName") == JOB
        and fields.get("QOS") == QOS
        and fields.get("Partition") == "cluster02"
        and fields.get("Account") == "rose"
        and fields.get("NumCPUs") == "8"
        and fields.get("UserId", "").startswith(USER + "("),
        "Wrong actual allocation",
    )
    for key in ("ReqTRES", "AllocTRES"):
        tres = dict(item.split("=", 1) for item in fields.get(key, "").split(",") if "=" in item)
        require(
            tres.get("cpu") == "8"
            and tres.get("mem") in ("66G", "67584M")
            and tres.get("gres/gpu:pro6000") == "2"
            and tres.get("gres/gpu", "2") == "2",
            "Wrong or unknown two-GPU allocation",
        )
    return detail


def launch(identity):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    require(
        re.fullmatch(r"[0-9]+", job_id)
        and os.environ.get("SLURM_JOB_NAME") == JOB
        and os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
        "Fresh actual Round A job required",
    )
    require(
        len("LH-SRF6-A-0-A-" + job_id + "-PRIMARY-b0-C_SHUF") <= 64,
        "Registered execution ID limit exceeded",
    )
    require(not any(k.startswith("SRF6_RESUME") for k in os.environ), "Resume is not authorized")
    with lock("A_LAUNCH.lock"):
        require(
            not any(
                canonical(OPS / name, exists=False).exists()
                for name in ("A_STARTED.json", "A_LAUNCH_FAILURE.json")
            ),
            "No repeated launch",
        )
        try:
            # The scheduler may start a job before sbatch has returned its receipt.
            deadline = time.monotonic() + 30
            while (
                not canonical(OPS / "A_SUBMISSION.json", exists=False).exists()
                and time.monotonic() < deadline
            ):
                time.sleep(0.2)
            receipt = json.loads(read_bytes(OPS / "A_SUBMISSION.json"))
            intent = bound(OPS / "A_ATTEMPT.json", receipt["attempt_sha256"])
            require(
                receipt["status"] == "SUBMITTED"
                and receipt["job_id"] == job_id
                and receipt["operation"] == intent["operation"] == identity
                and intent["argv"] == argv_for(identity)
                and intent["resume_tasks"] == [],
                "Actual unique Round A submission closure required",
            )
            require(
                intent["worker_attempt_template"] == "A-{job_id}", "Worker attempt identity changed"
            )
            detail = allocation(job_id)
            facts = metadata()
            pristine(facts)
            # Verify the unchanged launcher against its already sealed source inventory.
            member = next(
                r
                for r in facts["manifest"]["files"]
                if r["path"] == "scripts/sr_f6/run_pilot_v3_pair.sh"
            )
            launcher = CODE / member["path"]
            require(
                hashlib.sha256(read_bytes(launcher)).hexdigest() == member["sha256"],
                "Changed B7 pair launcher",
            )
            cache = canonical(OPS / ("job-" + job_id), exists=False)
            cache.mkdir(mode=0o700)
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("SRF6_", "PYTHON"))
                and k not in ("BASH_ENV", "ENV", "LD_PRELOAD", "LD_LIBRARY_PATH")
            }
            for key, relative in {
                "TMPDIR": "tmp",
                "TMP": "tmp",
                "TEMP": "tmp",
                "XDG_CACHE_HOME": "cache",
                "HF_HOME": "cache/huggingface",
                "HF_HUB_CACHE": "cache/huggingface/hub",
                "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
                "TRANSFORMERS_CACHE": "cache/huggingface/hub",
                "HF_MODULES_CACHE": "cache/huggingface/modules",
                "HF_DATASETS_CACHE": "cache/huggingface/datasets",
                "HF_ASSETS_CACHE": "cache/huggingface/assets",
                "HF_XET_CACHE": "cache/huggingface/xet",
                "TORCH_HOME": "cache/torch",
                "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
                "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
                "TRITON_CACHE_DIR": "cache/triton",
                "CUDA_CACHE_PATH": "cache/cuda",
                "NUMBA_CACHE_DIR": "cache/numba",
                "MPLCONFIGDIR": "cache/mpl",
            }.items():
                destination = canonical(cache / relative, exists=False)
                destination.mkdir(parents=True, exist_ok=True)
                env[key] = str(destination)
            env.update(
                SRF6_CODE=str(CODE),
                SRF6_ROOT=str(ROOT),
                SRF6_PLAN_SHA256=PLAN,
                SRF6_PHASE_IDENTITY_SHA256=PHASE,
                SRF6_PILOT_SPEC_SHA256=SPEC,
                SRF6_ADMISSION_SHA256=ADMISSION,
                SRF6_ROUND="A",
                SRF6_ATTEMPT="A-" + job_id,
                PYTHONPATH=str(CODE / "src"),
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONOPTIMIZE="0",
                # srun must inherit this newly cleaned/fixed environment, not
                # the restricted variable list used for the outer sbatch.
                SLURM_EXPORT_ENV="ALL",
            )
            write_once(
                OPS / "A_STARTED.json",
                dict(
                    at=now(),
                    job_id=job_id,
                    operation=identity,
                    producer_revision=PRODUCER,
                    allocation=detail,
                    round="A",
                    resume_tasks=[],
                    cache=str(cache),
                    pair_launcher=str(launcher),
                    status="LAUNCH_NOT_SCIENTIFIC_PASS",
                    worker_attempt_id="A-" + job_id,
                ),
            )
        except BaseException as exc:
            write_once(
                OPS / "A_LAUNCH_FAILURE.json", dict(at=now(), job_id=job_id, error=repr(exc))
            )
            raise
    try:
        os.execve("/bin/bash", ["/bin/bash", str(launcher)], env)
    except OSError as exc:
        write_once(OPS / "A_LAUNCH_FAILURE.json", dict(at=now(), job_id=job_id, error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "launch"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--binding-sha256", required=True)
    args = parser.parse_args()
    identity = operation_identity(args.revision, args.deployment, args.binding_sha256)
    if args.action == "launch":
        launch(identity)
    else:
        result = submit(identity)
        print(json.dumps(result))
        if result["status"] not in ("SUBMITTED", "NO_LEGAL_CAPACITY"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
