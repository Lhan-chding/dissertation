#!/usr/bin/env python3
"""Finish only the interrupted CPU preparation for 11031/T1 DEV16.

Default verifies the old local receipt. --execute starts one detached CPU
worker on the cluster; --collect reads the same attempt after reconnection.
The original preparation transport remains SSH255. This operator has no
scheduler mutation, model invocation, optimizer, or new production request.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1")
CODE = ROOT / "train_code_71018e5"
PUB = ROOT / "execution_releases"
PREFIX = "TRAINED_DEV_71018e5_20261116_start-11031_T1_step-016_"
SIDE = PUB / "dev16_cpu_recovery_11031_T1_016_v1"
PYTHON = "/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
PROJECT = Path("/Users/louis/Documents/ChatGPT/dissertation-ntu")
OLD = PROJECT / "artifacts/sr_satdose_20261112/trained_dev_20261124/start-11031_T1_step-016"
PERMISSION = "NO_MODEL_CALLS_OR_GPU_SUBMISSION"
JOB = "LH-SATDOSE-DEV-11031-T1-s16"
PROCESS = {
    "boot_id": "54e9a9a5-0b43-419a-bc5b-010b70d2acc9",
    "host": "login-1.cluster02.eee.ntu.edu.sg",
    "pid": 1946848,
    "start_ticks": 345906020,
}
PRIOR = {
    "PREPARATION.json": "3e25317235b1414895af07dfc1689ff5edad77b135caefb03f88337ae6341ed9",
    "PREPARATION.json.intent.json": (
        "59c22aa67e618d018503c5f2735bf5d56bf1045b3522344ecd2607ab2fc9c1ae"
    ),
    "PREPARATION.json.source.py": (
        "188b510417dd648971ee1216d6468a0fb5b0d4297fb5f95158cd1ac65b468b88"
    ),
    "PREPARATION.json.stderr.log": (
        "27b8e25ec24ea1acc2777e8c1dac47538e844d41e856496ed237e4a1d346c44f"
    ),
    "PREPARATION.json.stdout.log": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
}
PRIOR_REMOTE = {
    "OBSERVED_RISK": "30de55d57398a9a8a60f5fbeac4a5095028dcd977f8eb900e5ad894ea1109ec8",
    "PREPARATION_CLAIM": "b4eb6325dd73f08dd242090f91632cdc87be077ee38d236fc01cfb5331b4c6f5",
    "RELEASE": "8d174c1e71b0a69a125afae3cc2723b3ee73f7c2ba07324c982fd0a00db23078",
    "STORAGE_BUDGET": "bae245d3ec2e8434ddde50760bfd504b178cf5fdfa52595a6dadf496b1cf7129",
}
SELECTION = (11031, "T1", 16)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file(path):
    require(
        path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
        "Noncanonical file: " + str(path),
    )
    return path.read_bytes()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def write(path, value):
    require(path.parent.resolve(strict=True) == path.parent, "Noncanonical output")
    data = value if isinstance(value, bytes) else canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_EXCL | os.O_CREAT, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    syncdir(path.parent)


def syncdir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def bind_intent(raw):
    require(
        digest(raw) == PRIOR["PREPARATION.json.intent.json"], "Old transport intent SHA differs"
    )
    i = json.loads(raw)
    p = i["request"]
    require(
        i["status"] == "REMOTE_PREPARATION_ATTEMPTED_UNKNOWN_UNTIL_RECEIPT"
        and i["source"]
        == {
            "path": str(OLD / "PREPARATION.json.source.py"),
            "sha256": PRIOR["PREPARATION.json.source.py"],
        }
        and p["preparation_script_sha256"] == i["source"]["sha256"]
        and (p["start_seed"], p["arm"], p["step"]) == SELECTION
        and p["root"] == str(ROOT)
        and p["code"] == str(CODE)
        and p["tag"] == "71018e5_20261116",
        "Wrong original preparation",
    )
    return p


def old_local():
    files = {k: file(OLD / k) for k in PRIOR}
    require(
        {k: digest(v) for k, v in files.items()} == PRIOR, "Original local attempt bytes changed"
    )
    r = json.loads(files["PREPARATION.json"])
    require(
        r["status"] == "REMOTE_PREPARATION_UNKNOWN_NO_RETRY"
        and r["returncode"] == 255
        and r["response"] is None
        and not files["PREPARATION.json.stdout.log"],
        "Unexpected original response",
    )
    bind_intent(files["PREPARATION.json.intent.json"])
    return files["PREPARATION.json.intent.json"]


def artifact(key):
    return PUB / (PREFIX + key + ".json")


def prior_ref(key):
    return {"path": str(artifact(key).relative_to(ROOT)), "sha256": PRIOR_REMOTE[key]}


def verified_inputs(p):
    originals = {k: file(artifact(k)) for k in PRIOR_REMOTE}
    require(
        {k: digest(v) for k, v in originals.items()} == PRIOR_REMOTE,
        "Original remote input hashes changed",
    )
    claim, risk, budget, release = [
        json.loads(originals[k])
        for k in ("PREPARATION_CLAIM", "OBSERVED_RISK", "STORAGE_BUDGET", "RELEASE")
    ]
    require(
        claim["status"] == "PREPARATION_IN_PROGRESS_NO_AUTOMATIC_RETRY"
        and claim["permission"] == PERMISSION
        and claim["request"] == p
        and claim["process"] == PROCESS,
        "Foreign original claim or process",
    )
    require(
        release["schema"] == "sr-satdose-trained-dev-release-v1"
        and release["release_root"] == str(CODE)
        and release["evidence_root"] == str(ROOT)
        and release["storage_budget"] == prior_ref("STORAGE_BUDGET")
        and all(
            release[k] == p[k]
            for k in (
                "sources",
                "admission_spec",
                "admission",
                "risk_clarification",
                "original_train_release",
                "canary_spec",
            )
        ),
        "Foreign release lineage",
    )
    expected_paths = {f"{seed}/{arm}" for seed in (11031, 11032) for arm in ("T0", "T1")}
    expected_devs = (
        {f"{s}/START/0" for s in (11031, 11032)}
        | {f"{k}/{st}" for k in expected_paths for st in (2, 8)}
        | {"11031/T0/16"}
    )
    require(
        set(release["trusted_commits"]) == expected_paths
        and all(len(v) == 16 for v in release["trusted_commits"].values())
        and set(release["continuation_results"])
        == {k + "/" + chunk for k in expected_paths for chunk in ("3-8", "9-16")}
        and set(release["dev_endpoints"]) == expected_devs,
        "Expected 64 updates, eight chunks, eleven prior DEV endpoints",
    )
    require(
        risk["status"] == "PENDING_REQUIRED_DEV"
        and risk["global_hold"] is False
        and set(risk["pending_required_dev"])
        == {k + "/16" for k in expected_paths if k != "11031/T0"},
        "Unexpected risk decision",
    )
    require(
        budget["scope"] == "TRAINED_DEV_ONLY"
        and budget["created_at_utc"] == claim["started_at_utc"]
        and budget["planned_responses"] == 108
        and budget["observed_completed_response_count"] == 11 * 108
        and {r["key"]: r["receipt"] for r in budget["observed_endpoints"]}
        == {k: v["receipt"] for k, v in release["dev_endpoints"].items()}
        and budget["required_bytes"]
        == 108 * budget["per_response_allowance_bytes"]
        + budget["raw_accounting_allowance_bytes"]
        + budget["metadata_allowance_bytes"],
        "Budget sizing lineage differs",
    )
    return release, budget, risk


def stable_inventory(p):
    actual = {}
    require(CODE.resolve(strict=True) == CODE, "Noncanonical training source root")
    for f in CODE.rglob("*"):
        require(f.resolve(strict=True) == f, "Noncanonical source")
        if f.is_file():
            actual[str(f.relative_to(CODE))] = digest(file(f))
        else:
            require(f.is_dir(), "Nonordinary source entry")
    require(
        len(actual) == 84 and actual == p["sources"] and len(p["execution_sources"]) == 19,
        "Original source inventory differs",
    )
    for value in p["execution_sources"].values():
        require(
            actual[str(Path(value["path"]).relative_to(CODE))] == value["sha256"],
            "Execution source pin differs",
        )


def unoccupied():
    for path in (
        ROOT / "trained_dev/start-11031/T1/step-016",
        ROOT.parent / "scheduling/trained_dev_71018e5/start-11031/T1/step-016",
    ):
        for ancestor in reversed(path.parents):
            if os.path.lexists(ancestor):
                require(
                    ancestor.resolve(strict=True) == ancestor and ancestor.is_dir(),
                    "Unsafe slot ancestor",
                )
        require(not os.path.lexists(path), "Target already attempted: " + str(path))
    expected = {artifact(k) for k in PRIOR_REMOTE}
    require(
        set(PUB.glob("TRAINED_DEV_*_start-11031_T1_step-016_*.json")) == expected,
        "Other or incomplete preparation evidence",
    )
    reads = {}
    for k, argv in (
        (
            "queue",
            ["squeue", "--user=varun024", "--name=" + JOB, "--noheader", "--format=%i|%200j|%T"],
        ),
        (
            "history",
            [
                "sacct",
                "--user=varun024",
                "--starttime=2026-11-12T00:00:00",
                "--name=" + JOB,
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,JobName%160,State%40",
            ],
        ),
    ):
        r = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env={k: v for k, v in os.environ.items() if k != "SLURM_CLUSTERS"},
        )
        require(not r.stderr.strip(), "Scheduler read warning")
        rows = [line.split("|") for line in r.stdout.splitlines() if line.strip()]
        require(all(len(row) == 3 for row in rows), "Malformed scheduler response")
        require(not rows, f"Matching or unexpected {k} record for {JOB}")
        reads[k] = {"argv": argv, "matching": 0, "observed_at_utc": now()}
    return reads


def prior_process_observation():
    host = socket.gethostname()
    if host != PROCESS["host"]:
        return dict(
            observed_host=host, status="OTHER_HOST_ORIGINAL_CLAIM_PUBLISHED_UNDER_SAME_SHARED_LEASE"
        )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    stat = Path("/proc") / str(PROCESS["pid"]) / "stat"
    if boot != PROCESS["boot_id"]:
        return dict(observed_host=host, status="DIFFERENT_BOOT", boot_id=boot)
    try:
        fields = stat.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return dict(observed_host=host, status="ORIGINAL_PID_ABSENT", boot_id=boot)
    require(
        int(fields[19]) != PROCESS["start_ticks"] or fields[0] == "Z", "Original process still live"
    )
    return dict(
        observed_host=host,
        status="REUSED_PID_OR_ZOMBIE",
        state=fields[0],
        start_ticks=int(fields[19]),
        boot_id=boot,
    )


def run_worker():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and ROOT.resolve(strict=True) == ROOT
        and PUB.resolve(strict=True) == PUB
        and SIDE.resolve(strict=True) == SIDE,
        "Canonical CPU-only environment required",
    )
    meta = json.loads(file(SIDE / "INTENT.json"))
    source = file(SIDE / "RECOVERY_SOURCE.py")
    require(
        Path(__file__) == SIDE / "RECOVERY_SOURCE.py" and digest(source) == meta["source_sha256"],
        "Worker source identity differs",
    )
    p = bind_intent(file(SIDE / "ORIGINAL_INTENT.json"))
    with ExitStack() as stack:
        leases = []
        for key, mode in [
            ("FIRST_TWO_LAUNCHER.lock", fcntl.LOCK_SH),
            ("DIAGNOSTIC_WORKER.lock", fcntl.LOCK_EX),
        ]:
            path = ROOT / key
            require(path.resolve(strict=True) == path and path.is_file(), "Noncanonical lease")
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
            leases.append((fd, path))

        def stable():
            for fd, path in leases:
                require(
                    path.resolve(strict=True) == path
                    and os.path.samestat(path.stat(), os.fstat(fd)),
                    "Lease replaced",
                )

        stable()
        stable_inventory(p)
        before = unoccupied()
        prior_process = prior_process_observation()
        release, budget, risk = verified_inputs(p)
        sys.path.insert(0, str(CODE))
        from execution import trained_dev_control as c

        require(
            c.ROOT == ROOT
            and c.PACKAGE == CODE
            and Path(c.__file__) == CODE / "execution/trained_dev_control.py",
            "Foreign original controller",
        )
        identity = c._process_identity()
        require(identity != PROCESS, "Recovery retained old process identity")
        write(
            SIDE / "CPU_STARTED.json",
            dict(
                process=identity,
                locks=[{"name": path.name, "inode": os.fstat(fd).st_ino} for fd, path in leases],
                scheduler=before,
                prior_process_observation=prior_process,
                observed_at_utc=now(),
            ),
        )
        ctx = c.preflight(artifact("RELEASE"), PRIOR_REMOTE["RELEASE"], *SELECTION)
        require(
            ctx["release"] == release
            and ctx["risk"] == risk
            and ctx["budget"] == budget
            and ctx["original"]["execution_sources"] == p["execution_sources"]
            and len(ctx["manifest"]) == len(ctx["requests"]) == 108,
            "Original CPU preflight differs from pinned preparation",
        )
        c._hold_free()
        ctx["base"]._observed.stable()
        ctx["canary"].stable()
        stable()
        stable_inventory(p)
        storage = c.storage_check(PUB, release["storage_budget"])
        require(
            storage["schema"] == "sr-satdose-trained-dev-storage-v1"
            and storage["budget"] == release["storage_budget"],
            "Storage result differs from fixed budget",
        )
        c._hold_free()
        stable()
        stable_inventory(p)
        verified_inputs(p)
        after = unoccupied()
        measured_at = datetime.fromisoformat(storage["checked_at_utc"])
        observed_at = c.storage._now()
        require(
            c.storage._budget(release["storage_budget"], observed_at) == budget
            and 0
            <= (observed_at - measured_at).total_seconds()
            <= budget["max_observation_age_seconds"],
            "Storage observation expired during final checks",
        )
        result = dict(
            status="TRAINED_DEV_RELEASE_CPU_PREFLIGHT_AND_STORAGE_OBSERVED",
            permission=PERMISSION,
            start_seed=11031,
            arm="T1",
            step=16,
            responses=108,
            release=prior_ref("RELEASE"),
            storage_budget=prior_ref("STORAGE_BUDGET"),
            observed_risk=prior_ref("OBSERVED_RISK"),
            storage=storage,
            storage_observation_directory=str(PUB),
            storage_limit=(
                "Snapshot, not reservation. Future controller must rerun "
                "on actual endpoint directory."
            ),
            original_source_count=84,
            original_execution_source_count=19,
            committed_updates=64,
            completed_dev_endpoints=11,
            pending_required_dev=ctx["risk"]["pending_required_dev"],
            completed_at_utc=now(),
        )
        proof = dict(
            status="RECOVERED_CPU_PREFLIGHT_AND_STORAGE_OBSERVED",
            permission=PERMISSION,
            original_local_sha256=PRIOR,
            original_remote_sha256=PRIOR_REMOTE,
            original_process=PROCESS,
            recovery_process=identity,
            recovery_source_sha256=digest(source),
            scheduler=after,
            intended_result=result,
            original_transport_exit=255,
        )
        write(SIDE / "PROVENANCE.json", proof)
        c.tx.write_once(artifact("PREPARATION_RESULT"), result)
        require(
            file(artifact("PREPARATION_RESULT")) == c.tx.canonical(result) + b"\n",
            "Written result bytes differ",
        )
        write(
            SIDE / "COMPLETE.json",
            dict(
                status="CPU_RECOVERY_COMPLETE",
                provenance=c.ref(SIDE / "PROVENANCE.json"),
                result=c.ref(artifact("PREPARATION_RESULT")),
                finished_at_utc=now(),
            ),
        )


def remote_start(source, raw):
    p = bind_intent(raw)
    require(
        PUB.resolve(strict=True) == PUB and not os.path.lexists(SIDE),
        "Recovery already exists, inspect with --collect",
    )
    verified_inputs(p)
    unoccupied()
    SIDE.mkdir(mode=0o700)
    syncdir(PUB)
    write(SIDE / "ORIGINAL_INTENT.json", raw)
    write(SIDE / "RECOVERY_SOURCE.py", source)
    write(
        SIDE / "INTENT.json",
        dict(
            status="CPU_RECOVERY_ATTEMPT_UNKNOWN_UNTIL_COMMITTED",
            source_sha256=digest(source),
            original_local_sha256=PRIOR,
            original_remote_sha256=PRIOR_REMOTE,
            created_at_utc=now(),
        ),
    )
    with (SIDE / "WORKER.stdout").open("xb") as out, (SIDE / "WORKER.stderr").open("xb") as err:
        proc = subprocess.Popen(
            [PYTHON, "-I", "-B", str(SIDE / "RECOVERY_SOURCE.py"), "--worker"],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
            close_fds=True,
        )
    value = dict(
        status="CPU_WORKER_STARTED_NOT_VALIDATED",
        pid=proc.pid,
        host=socket.gethostname(),
        permission=PERMISSION,
    )
    write(SIDE / "STARTED.json", value)
    return value


def remote_collect(source, raw):
    bind_intent(raw)
    require(
        file(SIDE / "ORIGINAL_INTENT.json") == raw and file(SIDE / "RECOVERY_SOURCE.py") == source,
        "Different recovery attempt",
    )
    require(
        not os.path.lexists(SIDE / "ERROR.json"),
        "Inspect retained recovery ERROR and worker stderr before further action",
    )
    if not (SIDE / "COMPLETE.json").is_file():
        return dict(
            status="CPU_RECOVERY_INCOMPLETE_OR_UNKNOWN",
            permission=PERMISSION,
            started=json.loads(file(SIDE / "STARTED.json"))
            if (SIDE / "STARTED.json").is_file()
            else None,
        )
    complete = json.loads(file(SIDE / "COMPLETE.json"))
    result = file(artifact("PREPARATION_RESULT"))
    proof_raw = file(SIDE / "PROVENANCE.json")
    proof = json.loads(proof_raw)
    require(
        complete["status"] == "CPU_RECOVERY_COMPLETE"
        and complete["provenance"]
        == {"path": str((SIDE / "PROVENANCE.json").relative_to(ROOT)), "sha256": digest(proof_raw)}
        and complete["result"]
        == {"path": str(artifact("PREPARATION_RESULT").relative_to(ROOT)), "sha256": digest(result)}
        and proof["status"] == "RECOVERED_CPU_PREFLIGHT_AND_STORAGE_OBSERVED"
        and proof["original_remote_sha256"] == PRIOR_REMOTE
        and proof["original_local_sha256"] == PRIOR
        and proof["recovery_source_sha256"] == digest(source)
        and canonical(proof["intended_result"]) == result,
        "Recovery evidence chain or result differs",
    )
    verified_inputs(bind_intent(raw))
    return dict(
        status="CPU_RECOVERY_INDEPENDENT_BYTES_VERIFIED",
        complete=complete,
        provenance=proof,
        permission=PERMISSION,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--collect", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--local-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            run_worker()
        except Exception as error:
            write(
                SIDE / "ERROR.json",
                dict(
                    status="RECOVERY_FAILED_OR_UNKNOWN_NO_RETRY",
                    error=str(error),
                    error_type=type(error).__name__,
                    at=now(),
                ),
            )
            raise
        return 0
    raw = old_local()
    source = file(Path(__file__).resolve(strict=True))
    if not args.execute and not args.collect:
        print(
            json.dumps(
                dict(
                    status="LOCAL_ORIGINAL_SSH255_AND_INTENT_VERIFIED",
                    permission=PERMISSION,
                    original_local_sha256=PRIOR,
                    expected_remote_sha256=PRIOR_REMOTE,
                )
            )
        )
        return 0
    local = args.local_dir
    require(
        local is not None
        and local.is_absolute()
        and local.parent.resolve(strict=True) == local.parent,
        "Canonical absolute local directory required",
    )
    if args.execute:
        local.mkdir(mode=0o700)
        write(local / "RECOVERY_SOURCE.py", source)
        write(local / "ORIGINAL_INTENT.json", raw)
    else:
        require(
            file(local / "RECOVERY_SOURCE.py") == source
            and file(local / "ORIGINAL_INTENT.json") == raw,
            "Different local recovery attempt",
        )
    action = "remote_start" if args.execute else "remote_collect"
    bootstrap = (
        "import base64,hashlib,json,sys; data=json.load(sys.stdin); "
        "src=base64.b64decode(data['source'],validate=True); "
        "assert hashlib.sha256(src).hexdigest()==sys.argv[1]; ns={'__name__':'_recovery'}; "
        "exec(compile(src,'<recovery>','exec'),ns); "
        "v=ns[sys.argv[2]](src,base64.b64decode(data['intent'],validate=True)); "
        "print(json.dumps(v,sort_keys=True))"
    )
    command = shlex.join(
        [
            "env",
            "CUDA_VISIBLE_DEVICES=",
            "PYTHONDONTWRITEBYTECODE=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            PYTHON,
            "-I",
            "-B",
            "-c",
            bootstrap,
            digest(source),
            action,
        ]
    )
    label = (
        "START"
        if args.execute
        else "OBSERVE_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    with (
        (local / (label + ".stdout")).open("xb") as out,
        (local / (label + ".stderr")).open("xb") as err,
    ):
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "eee-cluster", command],
            input=canonical(
                {
                    "source": base64.b64encode(source).decode(),
                    "intent": base64.b64encode(raw).decode(),
                }
            ),
            stdout=out,
            stderr=err,
        )
        out.flush()
        err.flush()
        os.fsync(out.fileno())
        os.fsync(err.fileno())
    if result.returncode:
        print(
            json.dumps(
                dict(status="TRANSPORT_OR_WORKER_UNKNOWN", returncode=result.returncode, logs=label)
            )
        )
        return result.returncode
    response = json.loads(file(local / (label + ".stdout")))
    write(local / (label + ".json"), response)
    print(json.dumps({k: v for k, v in response.items() if k != "provenance"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
