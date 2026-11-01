"""One user-authorized fixed DEV evaluation; no delivery or historical audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

BASE = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = BASE / "sr_f6_semantic_factorial_20261027"
DEPLOYMENT = BASE / "sr_f6_code_b7e7b06"
NEW_OPS = BASE / "sr_f6_operations_20261027/PILOT_DEV_EVAL_V2_20261101"
PLAN = ROOT / "PHASE_RUNS/PILOT_V3/JOBS/PLAN.json"
PLAN_SHA = "56102cd5b10d93bde1996527e39fa1ec30943fc2618cab2838f03d5fd550d4f4"
SPEC = ROOT / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json"
SPEC_SHA = "4936b29e3a897334c1c4e67aad436bf46bdea9a105354d0155484d2acc20994c"
B_COMPLETE = BASE / "sr_f6_operations_20261027/PILOT_ROUND_B_20261031/round_b_COMPLETE.json"
B_COMPLETE_SHA = "0047dabc1db4484e9949d1f367896e02df07b3d4ae6e99d9636d25b7c87bd77f"
EVALUATION = ROOT / "PHASE_RUNS/PILOT_V3/EVALUATION"
PAIRS = {f"{p}/parent" for p in ("b0", "b1", "b2")} | {
    f"{p}/{a}/step12" for p in ("b0", "b1", "b2") for a in ("U10", "U11", "C_SHUF")
}
SCHEMA = "srf6-dev-eval-v2"
JOB_NAME = "LH-SRF6-PilotV3-DEVV2"


def require(value, message):
    if not value:
        raise PermissionError(message)


def plain(path):
    p = Path(path)
    require(p.is_absolute() and ".." not in p.parts, "Absolute plain path required")
    for x in (*p.parents, p):
        require(not x.is_symlink(), "Symlink forbidden")
    return p


def data(path):
    p = plain(path)
    require(p.is_file() and p.stat().st_size <= 32 * 2**20, "Bounded file required")
    return p.read_bytes()


def sha(path):
    return hashlib.sha256(data(path)).hexdigest()


def read(path):
    return json.loads(data(path))


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def pinned(path, digest):
    require(sha(path) == digest, "Pinned input changed: " + str(path))
    return read(path)


def write(path, value):
    p = plain(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(p)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def local_module(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("dev_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_plan():
    plan = pinned(PLAN, PLAN_SHA)
    spec = pinned(SPEC, SPEC_SHA)
    endpoints = spec["evaluation"]["endpoints"]
    require(
        len(endpoints) == 12 and {x["endpoint_id"] for x in endpoints} == PAIRS,
        "Fixed twelve endpoints required",
    )
    tasks = {t["task_id"]: t for t in plan["tasks"] if t["kind"] == "EVALUATION"}
    queues = [[tasks[t]["endpoint_id"] for t in lane] for lane in plan["queues"]["EVAL"]]
    require(
        [len(q) for q in queues] == [6, 6]
        and {item for queue in queues for item in queue} == PAIRS,
        "Fixed two lanes required",
    )
    require(spec["budgets"]["dev_eval_responses"] == 2304, "Fixed response budget required")
    return plan, spec, queues


def pristine(endpoint=None):
    folders = [EVALUATION / endpoint] if endpoint else [EVALUATION]
    if endpoint is None:
        jobs = ROOT / "PHASE_RUNS/PILOT_V3/JOBS"
        folders.extend((jobs / "TASKS").glob("EVAL-*"))
        folders.append(jobs / "WORKERS/EVAL")
    for folder in folders:
        plain(folder)
        if folder.exists():
            require(
                not any(p.is_file() or p.is_symlink() for p in folder.rglob("*")),
                "Prior evaluation attempt forbids replacement: " + str(folder),
            )


def check_preflight():
    operator = local_module("operations")
    technical_predecessor = operator.verify_prior_technical_failure()
    for value in operator.USER_AUTHORIZATION.values():
        if isinstance(value, dict) and set(value) == {"path", "sha256"}:
            pinned(value["path"], value["sha256"])
    _plan, spec, queues = fixed_plan()
    b = pinned(B_COMPLETE, B_COMPLETE_SHA)
    require(b.get("status") == "ROUND_B_COMPLETE_NOT_GATE_SIGNED", "B completion required")
    pristine()
    sources = []
    for endpoint in spec["evaluation"]["endpoints"]:
        if endpoint["kind"] == "PILOT_ENDPOINT":
            path = ROOT / endpoint["expected_commit_path"]
            value = read(path)
            require(value.get("step") == 12, "Original step12 required")
            sources.append(ref(path))
        else:
            value = endpoint["reference"]
            pinned(ROOT / value["path"], value["sha256"])
            sources.append(value)
    return dict(
        at=now(),
        status="REGISTERED_DEV_INPUTS_PRESENT_NOT_DELIVERY_AUDITED",
        plan_ref=ref(PLAN),
        spec_ref=ref(SPEC),
        endpoint_refs=sources,
        queues=queues,
        new_responses=0,
        full_delivery_verified=False,
        delivery_audit_waived_by_user=True,
        technical_predecessor=technical_predecessor,
    )


def contract(path, digest):
    require(Path(path) == NEW_OPS / "CONTRACT.json", "Fixed DEV contract required")
    value = pinned(path, digest)
    operator = local_module("operations")
    expected = {
        "schema",
        "output_directory",
        "driver_sha256",
        "source_deployment",
        "b_job_id",
        "round_b_complete_sha256",
        "user_authorization",
        "delivery_audit_waived_by_user",
        "full_delivery_verified",
        "formal_authorized",
        "test_authorized",
        "execution_source",
    }
    require(set(value) == expected and value["schema"] == SCHEMA, "DEV contract fields differ")
    require(
        value["output_directory"] == str(NEW_OPS / "RESULT")
        and value["source_deployment"] == str(DEPLOYMENT)
        and value["driver_sha256"] == sha(Path(__file__).resolve()),
        "Driver/source/output differ",
    )
    require(
        value["b_job_id"] == "237289" and value["round_b_complete_sha256"] == B_COMPLETE_SHA,
        "B predecessor differs",
    )
    require(
        value["user_authorization"] == operator.USER_AUTHORIZATION, "User DEV authorization differs"
    )
    require(
        value["delivery_audit_waived_by_user"] is True
        and all(
            value[k] is False
            for k in ("full_delivery_verified", "formal_authorized", "test_authorized")
        ),
        "Do not invent delivery or scientific admission",
    )
    source = value["execution_source"]
    deployment = BASE / ("sr_f6_dev_eval_v2_code_" + source["revision"][:7])
    require(
        Path(source["deployment"]) == deployment
        and Path(__file__).resolve() == deployment / "scripts/sr_f6_dev_eval_v2/driver.py",
        "Actual new execution source differs",
    )
    manifest = pinned(deployment / "DEV_SOURCE.json", source["binding_sha256"])
    require(
        manifest["revision"] == source["revision"] and manifest["deployment"] == str(deployment),
        "Manifest identity differs",
    )
    require(
        set(manifest["files"])
        == {
            "scripts/sr_f6_dev_eval_v2/" + n
            for n in ("operations.py", "driver.py", "runtime_adapter.py", "launch.sbatch")
        },
        "Runtime source membership differs",
    )
    for name, item in manifest["files"].items():
        raw = data(deployment / name)
        require(
            len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
            "Runtime source bytes changed",
        )
    return value


def allocation():
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        job.isdigit()
        and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
        and os.environ.get("OMP_NUM_THREADS") == "1",
        "Unique Slurm OMP1 execution required",
    )
    out = subprocess.run(
        ["scontrol", "show", "job", "-o", job],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", out))
    require(
        all(
            fields.get(k) == v
            for k, v in dict(
                JobId=job,
                JobName=JOB_NAME,
                JobState="RUNNING",
                QOS="soujanya-poria-startfund-2026-03",
                Partition="cluster02",
                Account="rose",
                Requeue="0",
                TimeLimit="23:00:00",
            ).items()
        ),
        "Actual DEV allocation differs",
    )
    require(fields.get("UserId", "").startswith("varun024("), "Wrong scheduler user")
    for key in ("ReqTRES", "AllocTRES"):
        tres = dict(x.split("=", 1) for x in fields.get(key, "").split(",") if "=" in x)
        require(
            tres.get("gres/gpu:pro6000") == "2"
            and tres.get("gres/gpu", "2") == "2"
            and tres.get("cpu") == "8"
            and tres.get("mem") in ("66G", "67584M"),
            "Wrong DEV resources",
        )
    return fields


def arguments(action, path, digest):
    return [
        sys.executable,
        "-I",
        "-B",
        "-u",
        str(Path(__file__).resolve()),
        action,
        "--contract",
        str(path),
        "--contract-sha256",
        digest,
    ]


def adapter():
    sys.path.insert(0, str(DEPLOYMENT / "src"))
    return local_module("runtime_adapter")


def endpoint(value, path, digest, identifier):
    require(identifier in PAIRS, "Unregistered endpoint")
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
        and "," not in os.environ["CUDA_VISIBLE_DEVICES"],
        "One visible GPU required",
    )
    pristine(identifier)
    task = "LH-SRF6-DEVV2-" + os.environ["SLURM_JOB_ID"] + "-" + identifier.replace("/", "-")
    result = adapter().run_endpoint(
        ROOT,
        endpoint_id=identifier,
        task_id=task,
        execution_source=value["execution_source"],
        authorization_ref=dict(path=str(path), sha256=digest),
    )
    require(
        result["status"] == "COMPLETE_NOT_GATE_SIGNED"
        and result["responses"] == 192
        and result["optimizer_updates"] == 0,
        "Endpoint incomplete",
    )


def lane(value, path, digest, worker):
    require(worker in (0, 1), "Fixed lane required")
    folder = NEW_OPS / "RESULT/LANES" / str(worker)
    plain(folder).mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    queue = fixed_plan()[2][worker]
    write(folder / "STARTED.json", dict(at=now(), worker=worker, queue=queue))
    refs = []
    try:
        for identifier in queue:
            cmd = [*arguments("endpoint", path, digest), "--endpoint-id", identifier]
            with (folder / (identifier.replace("/", "-") + ".log")).open("x") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
            require(
                result.returncode == 0,
                "Endpoint failed; preserve attempt, never resample: " + identifier,
            )
            refs.append(ref(EVALUATION / identifier / "COMPLETE.json"))
        write(
            folder / "COMPLETE.json",
            dict(
                at=now(),
                worker=worker,
                endpoint_refs=refs,
                elapsed_seconds=time.monotonic() - start,
                optimizer_updates=0,
            ),
        )
    except BaseException as exc:
        write(
            folder / "FAILURE.json",
            dict(
                at=now(),
                error_type=type(exc).__name__,
                error=str(exc),
                completed_endpoint_refs=refs,
                scientific_gate_status="NOT_EVALUATED",
            ),
        )
        raise


def launch(value, path, digest):
    allocated = allocation()
    preflight = check_preflight()
    output = plain(NEW_OPS / "RESULT")
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    write(
        output / "STARTED.json",
        dict(
            at=now(),
            allocation=allocated,
            preflight=preflight,
            authorization_ref=ref(path),
            execution_source=value["execution_source"],
        ),
    )
    children = []
    waited = set()
    streams = []
    try:
        for worker in (0, 1):
            log = (output / f"lane-{worker}.log").open("x")
            streams.append(log)
            cmd = [
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
                *arguments("lane", path, digest),
                "--worker",
                str(worker),
            ]
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        codes = []
        for index, child in enumerate(children):
            codes.append(child.wait())
            waited.add(index)
        require(codes == [0, 0], "One or more DEV lanes failed; peer allowed to finish")
        summary = adapter().aggregate(
            ROOT,
            execution_source=value["execution_source"],
            authorization_ref=dict(path=str(path), sha256=digest),
        )
        summary_ref = write(output / "SUMMARY.json", summary)
        write(
            output / "COMPLETE.json",
            dict(
                at=now(),
                status="COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED",
                execution_source=value["execution_source"],
                authorization_ref=dict(path=str(path), sha256=digest),
                summary_ref=summary_ref,
                lanes=[ref(output / "LANES" / str(w) / "COMPLETE.json") for w in (0, 1)],
                elapsed_seconds=time.monotonic() - start,
                delivery_audit_waived_by_user=True,
                full_delivery_verified=False,
                formal_authorized=False,
                test_authorized=False,
                optimizer_updates=0,
            ),
        )
    except BaseException as exc:
        for index, child in enumerate(children):
            if index not in waited:
                child.wait()
        write(
            output / "FAILURE.json",
            dict(
                at=now(),
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                elapsed_seconds=time.monotonic() - start,
                scientific_gate_status="NOT_EVALUATED",
                automatic_retry_authorized=False,
            ),
        )
        raise
    finally:
        for stream in streams:
            stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "lane", "endpoint"))
    p.add_argument("--contract", required=True)
    p.add_argument("--contract-sha256", required=True)
    p.add_argument("--worker", type=int)
    p.add_argument("--endpoint-id")
    args = p.parse_args()
    value = contract(args.contract, args.contract_sha256)
    require(os.environ.get("OMP_NUM_THREADS") == "1", "OMP1 required")
    if args.action == "launch":
        launch(value, args.contract, args.contract_sha256)
    elif args.action == "lane":
        lane(value, args.contract, args.contract_sha256, args.worker)
    else:
        endpoint(value, args.contract, args.contract_sha256, args.endpoint_id)


if __name__ == "__main__":
    main()
