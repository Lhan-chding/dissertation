#!/usr/bin/env python3
"""Run the confirmed interrupted DEV16 CPU preparation in one CPU-only Slurm job.

Default is a local evidence check. --execute sends at most one held sbatch.
--collect is read-only and validates the original CPU provenance plus Slurm
accounting. No GPU, training, deletion, replacement, or uncertain retry.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import types
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

V1_SHA = "957c611fd0d9f1ed9824903b17d87324f4d82cc042be95f932b64648c5efc111"
SLURM_HELPER_SHA = "28d3e0785d834e35a3119d8af4f5e5d7fffdc70b1ad8e277b476cb0d0efd0989"
PROBE_SHA = "924fe4bebd1aeca4fbb6273cc09bcac29468da279d72bfc0613098bb501bb794"
V1_NAME = "satdose_resume_dev16_preflight.py"
HELPER_NAME = "satdose_launch_parallel_main.py"
ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1")
BASE = ROOT / "execution_releases/dev16_cpu_recovery_11031_T1_016_slurm_v2"
OLD = ROOT / "execution_releases/dev16_cpu_recovery_11031_T1_016_v1"
PY = "/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
QOS = "soujanya-poria-startfund-2026-03"
CPU_JOB = "LH-SATDOSE-CPU-DEV16-11031-T1"
OLD_PROC = dict(
    host="login-2.cluster02.eee.ntu.edu.sg",
    boot_id="887548b6-32fd-4f48-b04b-42341e5921b2",
    pid=316867,
    start_ticks=447262966,
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def module(data, expected, filename):
    need(sha(data) == expected, "Source pin differs: " + filename.name)
    m = types.ModuleType("_authenticated_" + filename.stem)
    m.__file__ = str(filename)
    exec(compile(data, str(filename), "exec"), m.__dict__)
    return m


def load_dependencies(old, helper):
    return module(old, V1_SHA, BASE / "RECOVERY_SOURCE.py"), module(
        helper, SLURM_HELPER_SHA, BASE / HELPER_NAME
    )


def observation(v, raw):
    need(sha(raw) == PROBE_SHA, "Wrong original-login-node disappearance proof")
    proof = json.loads(raw)
    need(
        proof["host"] == OLD_PROC["host"]
        and proof["boot_id"] == OLD_PROC["boot_id"]
        and proof["process"] == OLD_PROC
        and proof["proc_pid_exists"] is False
        and proof["exclusive_worker_idle"] is True
        and proof["slot_exists"] is False
        and proof["original_result_exists"] is False,
        "Old worker absence is unresolved",
    )
    need(
        proof["old_publication_sha256"]
        == {key + ".json": value for key, value in v.PRIOR_REMOTE.items()},
        "Original freeze references differ",
    )
    return proof


def old_side(v, proof):
    need(
        OLD.resolve(strict=True) == OLD
        and {p.name for p in OLD.iterdir()} == set(proof["old_side_files"]),
        "Old CPU attempt acquired new or unexpected evidence",
    )
    for name, meta in proof["old_side_files"].items():
        raw = v.file(OLD / name)
        need(meta == {"sha256": sha(raw), "bytes": len(raw)}, "Changed old CPU evidence: " + name)
    need(
        json.loads(v.file(OLD / "CPU_STARTED.json"))["process"] == OLD_PROC,
        "Old CPU claim identity differs",
    )


@contextmanager
def original_leases(v):
    with ExitStack() as stack:
        opened = []
        for name, mode in [
            ("FIRST_TWO_LAUNCHER.lock", fcntl.LOCK_SH),
            ("DIAGNOSTIC_WORKER.lock", fcntl.LOCK_EX),
        ]:
            path = ROOT / name
            v.file(path)
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
            opened.append((fd, path))

        def stable():
            for fd, path in opened:
                need(
                    path.resolve(strict=True) == path
                    and os.path.samestat(os.fstat(fd), path.stat()),
                    "Original lease changed",
                )

        stable()
        yield stable


def no_cpu_job(s):
    queue = s.observed(
        ["squeue", "--user=varun024", "--name=" + CPU_JOB, "-h", "-o", "%i|%200j|%T"]
    )
    history = s.observed(
        [
            "sacct",
            "--user=varun024",
            "--starttime=2026-11-24T00:00:00",
            "--name=" + CPU_JOB,
            "-nP",
            "-o",
            "JobIDRaw,JobName%160,State%40",
        ]
    )
    need(
        not queue.strip() and not history.strip(),
        "Existing CPU recovery job; inspect it without resubmitting",
    )


def resources(s, value):
    r = s.tres(value)
    need(
        r.get("cpu") == "2"
        and r.get("mem") in ("6G", "6144M")
        and r.get("node") == "1"
        and all(
            int(n) == 0 for key, n in r.items() if key == "gres/gpu" or key.startswith("gres/gpu:")
        ),
        "Expected two CPUs, 6144 MiB, one node and zero GPUs",
    )


def own(s, job, held):
    need(type(job) is int and job > 0, "Invalid CPU job ID")
    fields = s.fields_for(job)
    expected = dict(
        JobId=str(job),
        JobName=CPU_JOB,
        Account="rose",
        QOS=QOS,
        Partition="cluster02",
        NumCPUs="2",
        NumTasks="1",
        Requeue="0",
        TimeLimit="04:00:00",
        WorkDir=str(BASE),
        Command=str(BASE / "RUN.sh"),
        StdOut=str(BASE / f"slurm-{job}.out"),
        StdErr=str(BASE / f"slurm-{job}.err"),
    )
    need(
        all(fields.get(k) == v for k, v in expected.items())
        and fields.get("NumNodes") in ("1", "1-1")
        and fields.get("UserId", "").startswith("varun024("),
        "Wrong CPU allocation identity",
    )
    resources(s, fields["ReqTRES"])
    if held:
        need(
            fields["JobState"] == "PENDING"
            and fields["Reason"] == "JobHeldUser"
            and fields["Priority"] == "0",
            "Unverified CPU job hold",
        )
    else:
        need(
            fields["JobState"] in ("RUNNING", "PENDING", "COMPLETING", "COMPLETED", "CONFIGURING")
            and fields.get("Reason") not in ("JobHeldUser", "JobHeldAdmin"),
            "Unverified CPU job release",
        )
        if fields["JobState"] == "RUNNING":
            resources(s, fields["AllocTRES"])
    return fields


def command(operator_sha):
    runner = (
        "import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        "assert p.resolve(strict=True)==p; b=p.read_bytes(); "
        "assert hashlib.sha256(b).hexdigest()==sys.argv[2]; "
        'm={"__name__":"_slurm_cpu_worker","__file__":str(p)}; '
        'exec(compile(b,str(p),"exec"),m); m["worker"]()'
    )
    launch = shlex.join(
        [PY, "-I", "-B", "-c", runner, str(BASE / "SLURM_OPERATOR.py"), operator_sha]
    )
    return (
        '#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 '
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "OPENBLAS_NUM_THREADS=2\n"
        "unset PYTHONPATH PYTHONHOME\nexec " + launch + "\n"
    ).encode()


def sbatch():
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--account=rose",
        "--qos=" + QOS,
        "--partition=cluster02",
        "--job-name=" + CPU_JOB,
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=6G",
        "--time=04:00:00",
        "--no-requeue",
        "--export=ALL",
        "--chdir=" + str(BASE),
        "--output=" + str(BASE / "slurm-%j.out"),
        "--error=" + str(BASE / "slurm-%j.err"),
        str(BASE / "RUN.sh"),
    ]


def submit(v, s, operator, old, helper, raw, probe):
    p = v.bind_intent(raw)
    proof = observation(v, probe)
    need(not os.path.lexists(BASE), "CPU submission location occupied; collect existing attempt")
    with original_leases(v) as stable:
        old_side(v, proof)
        v.unoccupied()
        v.verified_inputs(p)
        v.stable_inventory(p)
        no_cpu_job(s)
        stable()
        BASE.mkdir(mode=0o700)
        v.syncdir(BASE.parent)
        for name, data in [
            ("SLURM_OPERATOR.py", operator),
            ("RECOVERY_SOURCE.py", old),
            (HELPER_NAME, helper),
            ("V1_ABSENCE.json", probe),
            ("ORIGINAL_INTENT.json", raw),
            ("RUN.sh", command(sha(operator))),
        ]:
            v.write(BASE / name, data)
        (BASE / "RUN.sh").chmod(0o700)
        v.write(
            BASE / "INTENT.json",
            dict(
                status="CPU_RECOVERY_SLURM_INTENT_NO_AUTOMATIC_RETRY",
                source_sha256=V1_SHA,
                operator_sha256=sha(operator),
                helper_sha256=SLURM_HELPER_SHA,
                old_side=str(OLD),
                old_probe_sha256=PROBE_SHA,
                permission=v.PERMISSION,
                command=sbatch(),
                observed_at_utc=v.now(),
            ),
        )
        old_side(v, proof)
        v.unoccupied()
        no_cpu_job(s)
        stable()
        response = s.run(sbatch())
        v.write(BASE / "SUBMIT_RESPONSE.json", response)
        need(
            response["returncode"] == 0 and re.fullmatch("[1-9][0-9]*", response["stdout"].strip()),
            "sbatch failed/unknown; no retry",
        )
        job = int(response["stdout"].strip())
        fields = own(s, job, True)
        v.write(
            BASE / "ACCEPTED.json",
            dict(
                status="CPU_RECOVERY_HELD_VERIFIED",
                job_id=job,
                held=fields,
                operator_sha256=sha(operator),
            ),
        )
        own(s, job, True)
        stable()
    args = ["scontrol", "release", str(job)]
    v.write(BASE / "RELEASE_INTENT.json", dict(job_id=job, argv=args, at=v.now()))
    released = s.run(args)
    v.write(BASE / "RELEASE_RESPONSE.json", released)
    need(released["returncode"] == 0, "CPU release failed/unknown; no resubmission")
    own(s, job, False)
    result = dict(
        status="CPU_RECOVERY_SLURM_RELEASE_CONFIRMED",
        job_id=job,
        gpus=0,
        cpus=2,
        memory_mib=6144,
        observed_at_utc=v.now(),
    )
    v.write(BASE / "RELEASE_RESULT.json", result)
    return result


def process_identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return dict(
        host=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid=os.getpid(),
        start_ticks=int(fields[19]),
    )


def worker():
    v, s = load_dependencies(
        (BASE / "RECOVERY_SOURCE.py").read_bytes(), (BASE / HELPER_NAME).read_bytes()
    )
    try:
        ack = json.loads(v.file(BASE / "ACCEPTED.json"))
        job = ack["job_id"]
        need(
            os.environ.get("SLURM_JOB_ID") == str(job)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
            "Must run in original CPU-only allocation",
        )
        fields = own(s, job, False)
        need(
            fields["JobState"] == "RUNNING"
            and ack["operator_sha256"] == sha(v.file(BASE / "SLURM_OPERATOR.py")),
            "Job context or operator changed",
        )
        proof = observation(v, v.file(BASE / "V1_ABSENCE.json"))
        old_side(v, proof)
        v.write(
            BASE / "WORKER_BINDING.json",
            dict(
                job_id=job,
                process=process_identity(),
                resources=fields,
                operator_sha256=ack["operator_sha256"],
                at=v.now(),
            ),
        )
        v.SIDE = BASE
        v.run_worker()
    except Exception as exc:
        v.write(
            BASE / "ERROR.json",
            dict(
                status="CPU_SLURM_RECOVERY_FAILED_NO_RETRY",
                error_type=type(exc).__name__,
                error=str(exc),
                observed_at_utc=v.now(),
            ),
        )
        raise


def account(s, job):
    query = [
        "sacct",
        "-nP",
        "-j",
        str(job),
        "-o",
        "JobIDRaw,JobName%120,Account%40,QOS%80,Partition,State%40,ExitCode,ReqTRES%256,AllocTRES%256",
    ]
    rows = [line.split("|") for line in s.observed(query).splitlines() if line.strip()]
    jid = str(job)
    need(
        all(len(r) == 9 and r[0] in (jid, jid + ".batch", jid + ".extern") for r in rows)
        and len({r[0] for r in rows}) == len(rows),
        "Unknown or duplicate accounting record",
    )
    parents = [r for r in rows if r[0] == jid]
    if not parents:
        live = own(s, job, False)
        return False, {"live": live, "accounting": rows}
    parent = parents[0]
    need(parent[1:5] == [CPU_JOB, "rose", QOS, "cluster02"], "CPU accounting identity differs")
    resources(s, parent[7])
    if parent[5] in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"):
        if parent[8] and parent[5] == "RUNNING":
            resources(s, parent[8])
        return False, {"accounting": rows}
    need(parent[5:7] == ["COMPLETED", "0:0"], "Unsuccessful CPU job: " + repr(parent[5:7]))
    resources(s, parent[8])
    need(
        jid + ".batch" in {r[0] for r in rows}
        and all(r[5:7] == ["COMPLETED", "0:0"] for r in rows),
        "CPU batch step not confirmed successful",
    )
    return True, {"accounting": rows}


def collect(v, s, operator, old, raw, probe):
    need(
        v.file(BASE / "SLURM_OPERATOR.py") == operator
        and v.file(BASE / "RECOVERY_SOURCE.py") == old
        and v.file(BASE / "RUN.sh") == command(sha(operator))
        and v.file(BASE / "V1_ABSENCE.json") == probe
        and sha(v.file(BASE / HELPER_NAME)) == SLURM_HELPER_SHA
        and v.file(BASE / "ORIGINAL_INTENT.json") == raw,
        "Wrong CPU recovery source, inputs, or batch script",
    )
    proof = observation(v, probe)
    old_side(v, proof)
    meta = json.loads(v.file(BASE / "INTENT.json"))
    ack = json.loads(v.file(BASE / "ACCEPTED.json"))
    sub = json.loads(v.file(BASE / "SUBMIT_RESPONSE.json"))
    job = ack["job_id"]
    need(
        type(job) is int
        and job > 0
        and ack["status"] == "CPU_RECOVERY_HELD_VERIFIED"
        and meta["command"] == sub["argv"] == sbatch()
        and sub["returncode"] == 0
        and sub["stdout"].strip() == str(job)
        and meta["operator_sha256"] == ack["operator_sha256"] == sha(operator)
        and meta["source_sha256"] == V1_SHA
        and meta["old_probe_sha256"] == PROBE_SHA
        and ack["held"]["JobId"] == str(job)
        and ack["held"]["JobName"] == CPU_JOB
        and ack["held"]["Reason"] == "JobHeldUser",
        "CPU submission evidence differs",
    )
    resources(s, ack["held"]["ReqTRES"])
    finished, states = account(s, job)
    if not finished:
        return dict(status="CPU_JOB_NOT_YET_VERIFIED", job_id=job, **states)
    intent = json.loads(v.file(BASE / "RELEASE_INTENT.json"))
    need(
        intent["argv"] == ["scontrol", "release", str(job)] and intent["job_id"] == job,
        "CPU job release intent differs",
    )
    release_response = BASE / "RELEASE_RESPONSE.json"
    if os.path.lexists(release_response):
        reply = json.loads(v.file(release_response))
        need(
            reply["argv"] == intent["argv"] and reply["returncode"] == 0,
            "CPU release response differs",
        )
    # A lost login transport can omit the release response even after Slurm starts
    # the job. The held identity, release intent, actual binding and accounting
    # prove that the exact job executed; never manufacture a missing response.
    binding = json.loads(v.file(BASE / "WORKER_BINDING.json"))
    started = json.loads(v.file(BASE / "CPU_STARTED.json"))
    need(
        binding["job_id"] == job
        and binding["resources"]["JobId"] == str(job)
        and binding["process"] == started["process"]
        and binding["operator_sha256"] == sha(operator),
        "CPU worker identity differs",
    )
    resources(s, binding["resources"]["AllocTRES"])
    v.SIDE = BASE
    receipt = v.remote_collect(old, raw)
    need(
        receipt["status"] == "CPU_RECOVERY_INDEPENDENT_BYTES_VERIFIED"
        and receipt["provenance"]["recovery_process"] == binding["process"],
        "Frozen recovery has no verified completion",
    )
    return dict(
        status="SLURM_CPU_RECOVERY_VERIFIED",
        job_id=job,
        gpus=0,
        complete=receipt["complete"],
        recovery_receipt=receipt,
        release_reply_present=(BASE / "RELEASE_RESPONSE.json").is_file(),
        **states,
    )


def remote(op, mode, encoded):
    old, helper, raw, probe = [
        base64.b64decode(encoded[k], validate=True)
        for k in ("v1", "slurm", "intent", "observation")
    ]
    v, s = load_dependencies(old, helper)
    need(v.SIDE == OLD and v.ROOT == ROOT, "Wrong original frozen namespace")
    v.bind_intent(raw)
    observation(v, probe)
    if mode == "execute":
        return submit(v, s, op, old, helper, raw, probe)
    need(mode == "collect", "Invalid mode")
    return collect(v, s, op, old, raw, probe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--local-dir", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve()
    old = (here.parent / V1_NAME).read_bytes()
    helper = (here.parent / HELPER_NAME).read_bytes()
    v, _ = load_dependencies(old, helper)
    raw = v.old_local()
    probe = v.file(v.OLD / "interruption_20261125/SLURM_RECOVERY_INSPECTION.json")
    observation(v, probe)
    op = v.file(here)
    if not args.execute and not args.collect:
        print(
            json.dumps(
                dict(
                    status="LOCAL_CPU_SLURM_PLAN_ONLY",
                    sbatch=sbatch(),
                    operator_sha256=sha(op),
                    v1_sha256=V1_SHA,
                    old_absence_sha256=PROBE_SHA,
                )
            )
        )
        return 0
    dest = args.local_dir
    need(
        dest is not None and dest.is_absolute() and dest.parent.resolve(strict=True) == dest.parent,
        "Use canonical absolute local dir",
    )
    if args.execute:
        dest.mkdir(mode=0o700)
        v.write(dest / "SLURM_OPERATOR.py", op)
        v.write(
            dest / "LOCAL_INTENT.json",
            dict(
                mode="execute",
                prior_source=V1_SHA,
                helper=SLURM_HELPER_SHA,
                probe=PROBE_SHA,
                at=v.now(),
            ),
        )
    else:
        need(v.file(dest / "SLURM_OPERATOR.py") == op, "Different local submission operator")
    label = (
        "SUBMIT"
        if args.execute
        else "COLLECT_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    action = "execute" if args.execute else "collect"
    bootstrap = (
        "import sys,json,hashlib,base64; x=json.load(sys.stdin); "
        "b=base64.b64decode(x.pop('operator'),validate=True); "
        "assert hashlib.sha256(b).hexdigest()==sys.argv[1]; n={'__name__':'_remote_launcher'}; "
        "exec(compile(b,'<launcher>','exec'),n); "
        "print(json.dumps(n['remote'](b,sys.argv[2],x),sort_keys=True))"
    )
    cmd = shlex.join(
        [
            "env",
            "CUDA_VISIBLE_DEVICES=",
            "PYTHONDONTWRITEBYTECODE=1",
            PY,
            "-I",
            "-B",
            "-c",
            bootstrap,
            sha(op),
            action,
        ]
    )
    payload = {
        k: base64.b64encode(b).decode()
        for k, b in {
            "operator": op,
            "v1": old,
            "slurm": helper,
            "intent": raw,
            "observation": probe,
        }.items()
    }
    with (
        (dest / (label + ".stdout")).open("xb") as out,
        (dest / (label + ".stderr")).open("xb") as err,
    ):
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "eee-cluster", cmd],
            input=v.canonical(payload),
            stdout=out,
            stderr=err,
        )
        out.flush()
        err.flush()
        os.fsync(out.fileno())
        os.fsync(err.fileno())
    if r.returncode:
        print(
            json.dumps(
                dict(
                    status="TRANSPORT_OR_REMOTE_STATE_UNKNOWN", returncode=r.returncode, logs=label
                )
            )
        )
        return r.returncode
    value = json.loads(v.file(dest / (label + ".stdout")))
    v.write(dest / (label + ".json"), value)
    print(
        json.dumps(
            {k: v for k, v in value.items() if k not in ("recovery_receipt", "accounting", "live")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
