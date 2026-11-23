"""Supervise the seven pairs of MAIN endpoints after independently verified training.

With no arguments, print the offline plan. --execute runs the supervisor for this
experiment. It only observes while training is incomplete. Submission intents,
returned job IDs, and audit attempts survive supervisor restarts.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OPS = Path(__file__).resolve().parent
WORKSPACE = OPS.parents[2]
ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1")
EXTERNAL = ROOT / "external_parallel_main_v1"
TERMINAL = ROOT / "execution_releases/terminal_4190bc3"
REMOTE_PYTHON = "/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
DEFAULT_STATE = WORKSPACE / "artifacts/sr_satdose_20261112/parallel_main/run"
WAVES = [("START", 0), ("T0", 8), ("T0", 16), ("T0", 24), ("T1", 8), ("T1", 16), ("T1", 24)]
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
PUBLISHER = OPS / "satdose_publish_actual_terminal.py"
LAUNCHER = OPS / "satdose_launch_parallel_main.py"
ADMISSION = OPS / "satdose_parallel_main_admission.py"
AUDITOR = OPS / "satdose_audit_parallel_main.py"
COVERAGE = DEFAULT_STATE.parent / "COVERAGE.json"
PUBLISHER_SHA = "49e1188ea76d5945ee21faad394b0ea06064bd18f1f8fe0b836e43940c783511"


class TransportUnavailable(Exception):
    """A read-only observation could not reach the cluster."""


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    require(path.resolve(strict=True) == path and path.is_file(), f"Noncanonical file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    sha(path)
    return json.loads(path.read_bytes())


def save(path, data, *, replace=False):
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(path.parent.resolve(strict=True) == path.parent, "Noncanonical evidence directory")
    if replace:
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(dir=path.parent, delete=False) as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        os.replace(out.name, path)
    else:
        with path.open("xb") as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def network_failure(path):
    return path.is_file() and any(
        message in path.read_text(errors="replace")
        for message in (
            "Network is unreachable",
            "Operation timed out",
            "Connection timed out",
            "Broken pipe",
            "No route to host",
        )
    )


def ssh(argv):
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=12",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "eee-cluster",
            shlex.join(argv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 255:
        raise TransportUnavailable(result.stderr[-600:])
    require(
        result.returncode == 0,
        f"Remote observation failed ({result.returncode}): {result.stderr[-800:]}",
    )
    return result.stdout


def readiness():
    # Authenticate the exact deployed read-only readiness program before running it.
    program = """import hashlib,sys,pathlib
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
if p.resolve(strict=True)!=p or not p.is_file(): raise ValueError('Admission source relocated')
data=p.read_bytes()
if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('Admission source changed')
sys.argv=[str(p),'--readiness']
exec(compile(data,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
"""
    answer = json.loads(
        ssh(
            [
                REMOTE_PYTHON,
                "-I",
                "-B",
                "-c",
                program,
                str(EXTERNAL / "operations" / ADMISSION.name),
                sha(ADMISSION),
            ]
        )
    )
    require(
        answer["schema"] == "sr-satdose-parallel-main-readiness-v1"
        and answer["status"] in ("WAIT_FOR_TRAINING_AND_AUDITS", "READY_FOR_TERMINAL_PUBLICATION"),
        "Unrecognized terminal readiness response",
    )
    return answer


def local(script, arguments, logdir):
    with (logdir / "stdout.log").open("xb") as out, (logdir / "stderr.log").open("xb") as err:
        return subprocess.run(
            [sys.executable, "-B", str(script), *map(str, arguments)],
            stdout=out,
            stderr=err,
            check=False,
        ).returncode


def terminal_publication(state):
    manifest, download = "REMOTE_MANIFEST.json", "LOCAL_DOWNLOAD_RECEIPT.json"
    downloads = sorted(state.glob("terminal-download-*"))
    for attempt in [state / "terminal-publish", *downloads]:
        folder = attempt / "bundle"
        if (folder / download).is_file() and not (folder / "LOCAL_ERROR.json").exists():
            receipt = read(folder / download)
            require(
                receipt["publication_success"] is True
                and receipt["remote_returncode"] == 0
                and receipt["status"] == "EXACT_TERMINAL_ARTIFACT_BYTES_DOWNLOADED",
                "Terminal evidence was not successfully verified",
            )
            return [
                "--terminal-dir",
                str(folder),
                "--terminal-manifest-sha256",
                sha(folder / manifest),
                "--terminal-download-receipt-sha256",
                sha(folder / download),
            ]
    probe = """import os, pathlib, sys
p=pathlib.Path(sys.argv[1])
if p.resolve()!=p: raise ValueError('Terminal path relocated')
if os.path.lexists(p/'ERROR.json'): raise ValueError('Terminal publisher retained ERROR')
if (p/'PUBLICATION_RECEIPT.json').is_file(): print('PUBLISHED')
else: print('OCCUPIED' if os.path.lexists(p) else 'ABSENT')
"""
    status = ssh([REMOTE_PYTHON, "-I", "-B", "-c", probe, str(TERMINAL)]).strip()
    require(
        status in ("ABSENT", "PUBLISHED"), "Existing terminal attempt requires read-only resolution"
    )
    mode = "publish" if status == "ABSENT" else "download"
    attempt = state / (
        "terminal-publish" if mode == "publish" else f"terminal-download-{len(downloads) + 1:04d}"
    )
    require(
        not os.path.lexists(attempt), "Previous terminal invocation requires read-only recovery"
    )
    attempt.mkdir()
    rc = local(
        PUBLISHER,
        [
            "--execute" if mode == "publish" else "--download",
            "--ssh-host",
            "eee-cluster",
            "--local-dir",
            attempt / "bundle",
        ],
        attempt,
    )
    if rc:
        logs = [
            *list((attempt / "bundle").glob("*.DOWNLOAD_STDERR")),
            attempt / "bundle" / "REMOTE_STDERR.log",
        ]
        if any(network_failure(path) for path in logs):
            raise TransportUnavailable(
                f"Terminal {mode} transport interrupted; evidence at {attempt}"
            )
    require(
        rc == 0 and (attempt / "bundle" / download).is_file(),
        f"Inspect terminal {mode} evidence at {attempt}",
    )
    return terminal_publication(state)


def pair_state(job_ids, accounting):
    expected = {str(job) for job in job_ids}
    require(
        len(expected) == 2 and all(j.isdecimal() and int(j) > 0 for j in expected),
        "Invalid pair identities",
    )
    observed = {}
    for line in accounting.splitlines():
        fields = line.strip().split("|")
        require(
            len(fields) == 3 and fields[0] in expected and fields[0] not in observed,
            "Unexpected Slurm accounting row",
        )
        job, status, exitcode = fields
        require(
            status in ACTIVE or (status == "COMPLETED" and exitcode == "0:0"),
            f"Inspect MAIN job {job}, state {status}, exit {exitcode}",
        )
        observed[job] = status
    return (
        "COMPLETE"
        if set(observed) == expected and all(s == "COMPLETED" for s in observed.values())
        else "WAIT"
    )


def verify_result(result, index):
    require(
        isinstance(result, dict)
        and result.get("status") == "MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED"
        and type(result.get("wave")) is int
        and result["wave"] == index,
        "Wrong wave release",
    )
    jobs = result["jobs"]
    require(
        set(jobs) == {"11031", "11032"}
        and len(set(jobs.values())) == 2
        and all(type(j) is int and j > 0 for j in jobs.values()),
        "Invalid pair identities",
    )
    require(
        result["release"]["path"] == "execution_releases/parallel_main_v1/RELEASE.json"
        and re.fullmatch(r"[0-9a-f]{64}", result["release"]["sha256"]),
        "Wrong shared release reference",
    )
    return result


def transport_receipt(path, index):
    receipt = read(path)
    source_sha = sha(LAUNCHER)
    intent = read(path.with_name(path.name + ".intent.json"))
    require(
        intent.get("mode") == "launch"
        and intent.get("wave") == index
        and intent.get("source_sha256") == receipt.get("source_sha256") == source_sha,
        "Saved transport intent does not match this launcher and wave",
    )
    for stream in ("stdout", "stderr"):
        target = path.with_name(path.name + "." + stream + ".log")
        require(
            receipt["raw_logs"][stream] == dict(path=str(target), sha256=sha(target)),
            "Submission transport log changed",
        )
    prefix = b"SATDOSE_PARALLEL_MAIN="
    markers = [
        line[len(prefix) :]
        for line in path.with_name(path.name + ".stdout.log").read_bytes().splitlines()
        if line.startswith(prefix)
    ]
    if receipt.get("status") == "PARALLEL_MAIN_OPERATOR_CAPTURED":
        require(
            len(markers) == 1 and json.loads(markers[0]) == receipt.get("response"),
            "Saved response differs from the original SSH output",
        )
    return receipt


def accepted(receipt, index):
    require(
        receipt.get("status") == "PARALLEL_MAIN_OPERATOR_CAPTURED"
        and receipt.get("returncode") == 0,
        "Submission did not return an authenticated acceptance",
    )
    return verify_result(receipt["response"]["result"], index)


def launcher_query(arguments):
    response = subprocess.run(
        [sys.executable, "-B", str(LAUNCHER), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if response.returncode == 255:
        raise TransportUnavailable(response.stderr[-600:] or "Read-only SSH unavailable")
    return response


def inspect(index):
    response = launcher_query(["--inspect-wave", "--wave", str(index)])
    require(
        response.returncode == 0,
        f"No verified remote release for interrupted wave {index}; inspect original intent",
    )
    envelope = json.loads(response.stdout)
    require(envelope.get("status") == "WAVE_RELEASE_CONFIRMED", "No confirmed remote wave")
    return verify_result(envelope["result"], index)


def deferred(receipt, index):
    value = (receipt.get("response") or {}).get("deferred")
    return (
        receipt.get("status") == "PARALLEL_MAIN_OPERATOR_CAPTURED"
        and receipt.get("returncode") == 3
        and isinstance(value, dict)
        and value.get("status") == "CAPACITY_DEFERRED"
        and type(value.get("wave")) is int
        and value["wave"] == index
        and type(value.get("submission_mutations")) is int
        and value["submission_mutations"] == 0
    )


def launch(state, index, terminal_args):
    prefix = f"wave-{index:02d}"
    previous = sorted(state.glob(prefix + "-attempt-*"))
    for attempt in previous:
        receipt_path = attempt / "SUBMISSION.json"
        if not receipt_path.exists():
            return inspect(index)
        receipt = transport_receipt(receipt_path, index)
        if deferred(receipt, index):
            continue
        if receipt.get("returncode") == 0 and (receipt.get("response") or {}).get("result"):
            return accepted(receipt, index)
        return inspect(index)
    capacity = launcher_query(["--check-capacity"])
    require(capacity.returncode in (0, 3), "Capacity probe failed: " + capacity.stderr[-600:])
    value = json.loads(capacity.stdout)
    if value["status"] == "CAPACITY_DEFERRED" and capacity.returncode == 3:
        return None
    require(
        value["status"] == "CAPACITY_READY" and capacity.returncode == 0,
        "Unrecognized capacity response",
    )
    attempt = state / f"{prefix}-attempt-{len(previous) + 1:04d}"
    attempt.mkdir()
    receipt_path = attempt / "SUBMISSION.json"
    rc = local(
        LAUNCHER, [*terminal_args, "--wave", index, "--receipt", receipt_path, "--execute"], attempt
    )
    if not receipt_path.exists():
        return inspect(index)
    receipt = transport_receipt(receipt_path, index)
    if rc == 3 and deferred(receipt, index):
        return None
    if rc or not (receipt.get("response") or {}).get("result"):
        return inspect(index)
    return accepted(receipt, index)


def advance(state):
    terminal = state / "TERMINAL_BINDING.json"
    if not terminal.exists():
        check = readiness()
        if check["status"] == "WAIT_FOR_TRAINING_AND_AUDITS":
            return dict(status=check["status"], missing=check["missing"])
        args = terminal_publication(state)
        save(terminal, args)
    args = read(terminal)
    used = set()
    shared_release = None
    for index in range(len(WAVES)):
        receipt = state / f"wave-{index:02d}-accepted.json"
        if receipt.exists():
            result = verify_result(read(receipt), index)
        else:
            result = launch(state, index, args)
            if result is None:
                return dict(status="WAIT_FOR_TWO_JOB_SLOTS_AND_FOUR_GPUS", wave=index)
            save(receipt, result)
        if shared_release is None:
            shared_release = result["release"]
        require(result["release"] == shared_release, "MAIN release changed between pairs")
        jobs = [str(v) for v in result["jobs"].values()]
        require(used.isdisjoint(jobs), "Job reused across MAIN waves")
        used.update(jobs)
        acct = ssh(["sacct", "-X", "-nP", "-j", ",".join(jobs), "-o", "JobIDRaw,State%40,ExitCode"])
        if pair_state(jobs, acct) != "COMPLETE":
            return dict(
                status="MAIN_PAIR_SUBMITTED", wave=index, jobs=jobs, accounting=acct.splitlines()
            )
    return finish_audit(state, shared_release, used)


def finish_audit(state, release, job_ids):
    # A verification/download attempt is read-only on the cluster. Preserve a
    # partial transfer and use a new directory to verify the same release.
    prior = sorted(state.glob("final-audit-*"))
    for folder in prior:
        output = folder / "bundle"
        receipt_path = output / "AUDIT_RECEIPT.json"
        if receipt_path.exists():
            r = read(receipt_path)
            require(
                r.get("release") == release
                and r.get("auditor_sha256") == sha(AUDITOR)
                and r.get("status") == "PARALLEL_MAIN_ALL_13760_CAPTURES_INDEPENDENTLY_VERIFIED"
                and r.get("responses") == 13760
                and r.get("lanes") == {"11031": 6880, "11032": 6880}
                and (r.get("waves"), r.get("jobs"), r.get("endpoints")) == (7, 14, 14)
                and r["bundle"]
                == {"path": "BUNDLE.json.gz", "sha256": sha(output / "BUNDLE.json.gz")},
                "Final audit receipt does not match this release and verified bundle",
            )
            return dict(
                status="MAIN_EVALUATION_AND_MERGE_VERIFIED",
                jobs=sorted(job_ids),
                audit=str(receipt_path),
            )
    attempt = state / f"final-audit-{len(prior) + 1:04d}"
    attempt.mkdir()
    output = attempt / "bundle"
    rc = local(
        AUDITOR,
        [
            "--release-sha256",
            release["sha256"],
            "--execute",
            "--ssh-host",
            "eee-cluster",
            "--output-dir",
            output,
        ],
        attempt,
    )
    if rc:
        network_log = output / "SSH_STDERR.log"
        if network_failure(network_log):
            raise TransportUnavailable(
                f"Final read-only audit connection interrupted; retained {attempt}"
            )
        raise RuntimeError(f"Final audit failed; inspect {attempt}")
    require((output / "AUDIT_RECEIPT.json").is_file(), "Final audit has no success receipt")
    return finish_audit(state, release, job_ids)


def run(args):
    state = args.state_dir.absolute()
    require(
        state.parent.resolve(strict=True) == state.parent and state.resolve() == state,
        "Noncanonical supervisor state directory",
    )
    state.mkdir(exist_ok=True)
    lock = os.open(state / "supervisor.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = [Path(__file__), ADMISSION, PUBLISHER, LAUNCHER, AUDITOR, COVERAGE]
        binding = {str(path): sha(path) for path in files}
        require(binding[str(PUBLISHER)] == PUBLISHER_SHA, "Original terminal publisher pin changed")
        pinfile = state / "SOURCE_BINDING.json"
        if not pinfile.exists():
            save(pinfile, binding)
        while True:
            require(
                read(pinfile) == {str(path): sha(path) for path in files},
                "Armed source or coverage binding changed",
            )
            try:
                value = advance(state)
            except TransportUnavailable as exc:
                value = dict(status="READ_ONLY_CONNECTION_UNAVAILABLE", error=str(exc))
            except Exception as exc:
                value = dict(status="NEEDS_REVIEW", error=str(exc))
                save(
                    state / "LATEST.json",
                    dict(value, utc=datetime.now(timezone.utc).isoformat()),
                    replace=True,
                )
                raise
            value["utc"] = datetime.now(timezone.utc).isoformat()
            save(state / "LATEST.json", value, replace=True)
            print(json.dumps(value), flush=True)
            if args.once or value["status"] == "MAIN_EVALUATION_AND_MERGE_VERIFIED":
                return
            time.sleep(3600)
    finally:
        os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--once", action="store_true", help="Make one supervisory observation/advance and exit"
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    if args.execute:
        run(args)
    else:
        print(
            json.dumps(
                dict(
                    status="LOCAL_PLAN_ONLY",
                    waves=WAVES,
                    concurrent_jobs=2,
                    gpus_per_job=2,
                    manifest_sha256=read(COVERAGE)["main_manifest_sha256"],
                )
            )
        )


if __name__ == "__main__":
    main()
