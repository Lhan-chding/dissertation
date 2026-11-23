#!/usr/bin/env python3
"""Deploy the parallel MAIN runner, inspect capacity, or release one verified pair.

Default is a local source inventory. Submission needs authenticated terminal
publication, exact independent audits, both CPU preflights, and two slots.
The fixed wave directory is the at-most-once submission record.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import types
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1")
CODE = ROOT / "external_parallel_main_v1"
BATCH = ROOT / "execution_releases/parallel_main_v1"
LEGACY = ROOT.parent / "scheduling/main_4190bc3"
LEGACY_GUARD = LEGACY / "PARALLEL_MAIN_OWNER.json"
PYTHON = "/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
QOS = "soujanya-poria-startfund-2026-03"
USER = "varun024"
PREPARER_SHA = "6e40ae7735fc5d6e2386cf0e996a4db4a3fa90d6eea79d13a8d14d03325302b5"
PLAN_CODE_SHA = "3956432c29633b002b2245dabe1941118eac7afb6a7611040f8652d92bef48a7"
PLAN_SHA = "7df3f350e7230d670168c22d5b93d9c0dfd647c10987b873643684f2b4965df3"
SELF = "operations/satdose_launch_parallel_main.py"
CONTROL = "execution/parallel_main_control.py"
LAUNCHER = "scheduling/run_parallel_main.sh"
ADMISSION = "operations/satdose_parallel_main_admission.py"
PLAN = "operations/satdose_parallel_main_plan.py"
PREPARER = "operations/satdose_prepare_main.py"
ASSETS = (
    SELF,
    CONTROL,
    LAUNCHER,
    ADMISSION,
    PLAN,
    PREPARER,
    "operations/satdose_publish_actual_terminal.py",
    "operations/satdose_audit_parallel_main.py",
)
WAVES = (("START", 0), ("T0", 8), ("T0", 16), ("T0", 24), ("T1", 8), ("T1", 16), ("T1", 24))
SEEDS = (11031, 11032)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path):
    require(
        path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
        "Noncanonical file: " + str(path),
    )
    return path.read_bytes()


def read(path):
    return json.loads(checked(path))


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(checked(path))}


def sync_parent(path):
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def put(path, value):
    require(path.parent.resolve(strict=True) == path.parent, "Noncanonical output directory")
    # Publish complete bytes atomically. Concurrent workers can poll a receipt as
    # soon as the name appears; a partial JSON file must never become visible.
    raw = value if isinstance(value, bytes) else canonical(value)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        sync_parent(path)
    finally:
        os.unlink(temporary)


def make_dir(path):
    require(path.parent.resolve(strict=True) == path.parent, "Noncanonical directory parent")
    path.mkdir(mode=0o700)
    sync_parent(path)


def inventory(directory):
    require(
        directory.resolve(strict=True) == directory and directory.is_dir(),
        "Noncanonical code directory",
    )
    found = {}
    for p in directory.rglob("*"):
        require(
            p.resolve(strict=True) == p and not p.is_symlink(), "Noncanonical deployed code path"
        )
        if p.is_file():
            found[str(p.relative_to(directory))] = digest(checked(p))
        else:
            require(p.is_dir(), "Nonordinary code entry")
    require(set(found) == set(ASSETS), "Incomplete/extra parallel source package")
    return found


def load(path, expected, name):
    source = checked(path)
    require(digest(source) == expected, "Changed dependency: " + path.name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def run(argv):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_CLUSTERS"}
    if argv[0] == "sbatch":
        env = {
            k: v
            for k, v in env.items()
            if not k.startswith("SBATCH_") and k != "CUDA_VISIBLE_DEVICES"
        }
    proc = subprocess.run(argv, capture_output=True, text=True, env=env)
    return dict(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        observed_at_utc=now(),
    )


def observed(argv):
    r = run(argv)
    require(r["returncode"] == 0 and not r["stderr"].strip(), "Scheduler query failed: " + argv[0])
    return r["stdout"]


def fields_for(job):
    value = observed(["scontrol", "show", "job", "--oneliner", str(job)])
    # Slurm emits Key=value pairs, not shell-escaped words. Unquoted values
    # can contain spaces, apostrophes, empty strings and additional '=' signs.
    required = {
        "JobId",
        "JobName",
        "Account",
        "QOS",
        "Partition",
        "UserId",
        "WorkDir",
        "Command",
        "StdOut",
        "StdErr",
        "NumCPUs",
        "NumTasks",
        "NumNodes",
        "Requeue",
        "TimeLimit",
        "JobState",
        "Reason",
        "Priority",
        "ReqTRES",
        "AllocTRES",
        "ExitCode",
    }
    fields = {}
    tokens = list(re.finditer(r"(?<!\S)([A-Z][a-zA-Z0-9_:/.-]*)=", value))
    for i, match in enumerate(tokens):
        key = match[1]
        if key not in required:
            continue
        require(key not in fields, "Ambiguous scheduler field: " + key)
        end = tokens[i + 1].start() if i + 1 < len(tokens) else len(value)
        raw = value[match.end() : end].strip()
        if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
            raw = raw[1:-1]
        fields[key] = raw
    require(fields.get("JobId") == str(job), "Scheduler job identity mismatch")
    return fields


def tres(value):
    parts = [v.split("=", 1) for v in value.split(",")]
    require(
        all(len(v) == 2 for v in parts) and len(dict(parts)) == len(parts),
        "Unknown scheduler resource request",
    )
    return dict(parts)


def measure_capacity():
    rows = observed(
        [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "where",
            "name=" + QOS,
            "format=Name%80,GrpTRES%256,MaxSubmitPU",
        ]
    ).splitlines()
    require(len(rows) == 1, "Cannot identify unique teacher QoS")
    name, limits, submit_limit = rows[0].strip().split("|")
    require(name == QOS and submit_limit.isdigit(), "Unknown teacher QoS job limit")
    gpu_limit = int(tres(limits)["gres/gpu:pro6000"])
    jobs = observed(["squeue", "-r", "-h", "--qos=" + QOS, "-o", "%i|%u|%T"]).splitlines()
    used = runnable = held = slots = 0
    seen = set()
    for line in jobs:
        jid, user, _ = line.strip().split("|")
        require(
            re.fullmatch(r"[1-9][0-9]*", jid) and jid not in seen,
            "Unresolved or duplicate queue job",
        )
        seen.add(jid)
        fields = fields_for(jid)
        require(
            fields.get("QOS") == QOS and fields["UserId"].startswith(user + "("),
            "Queue changed during observation",
        )
        slots += int(user == USER)
        state = fields.get("JobState")
        if (
            state == "PENDING"
            and fields.get("Reason") == "JobHeldUser"
            and fields.get("Priority") == "0"
        ):
            held += 1
            continue
        active = state in ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED")
        require(active or state == "PENDING", "Cannot account for a changing queue job")
        resources = tres(fields["AllocTRES"] if active else fields["ReqTRES"])
        generic = int(resources.get("gres/gpu", "0"))
        typed = int(resources.get("gres/gpu:pro6000", "0"))
        # Any untyped request could be satisfied with the requested PRO6000 type.
        known = sum(int(n) for k, n in resources.items() if k.startswith("gres/gpu:"))
        count = typed + max(0, generic - known)
        if active:
            used += count
        else:
            runnable += count
    free_slots = int(submit_limit) - slots
    headroom = gpu_limit - used - runnable
    return dict(
        status="CAPACITY_READY" if free_slots >= 2 and headroom >= 4 else "CAPACITY_DEFERRED",
        qos=QOS,
        gpu_limit=gpu_limit,
        running_gpu=used,
        runnable_gpu=runnable,
        gpu_headroom=headroom,
        held_jobs=held,
        submitted_jobs=slots,
        submission_limit=int(submit_limit),
        free_slots=free_slots,
        observed_at_utc=now(),
    )


def capacity():
    try:
        return measure_capacity()
    except (ValueError, KeyError, OSError) as exc:
        # No capacity query ever mutates Slurm. An unobservable or changing
        # shared queue must not become permission to submit or a fatal launch.
        return dict(
            status="CAPACITY_DEFERRED",
            reason="CAPACITY_UNOBSERVED",
            detail=str(exc),
            qos=QOS,
            observed_at_utc=now(),
        )


def deploy(payload):
    require(ROOT.resolve(strict=True) == ROOT, "Invalid deployment parent")
    sources = payload["sources"]
    require(set(sources) == set(ASSETS) == set(payload["blobs"]), "Invalid deployment file set")
    if os.path.lexists(CODE):
        require(inventory(CODE) == sources, "Existing code deployment differs; no mutation")
        return {"status": "PARALLEL_MAIN_DEPLOYED_VERIFIED", "root": str(CODE), "sources": sources}
    stage = Path(tempfile.mkdtemp(prefix=".parallel-main-deploy-", dir=ROOT))
    for name in ASSETS:
        raw = base64.b64decode(payload["blobs"][name], validate=True)
        require(digest(raw) == sources[name], "Invalid shipped source: " + name)
        path = stage / name
        path.parent.mkdir(exist_ok=True)
        put(path, raw)
        if name == LAUNCHER:
            path.chmod(0o700)
    require(inventory(stage) == sources, "Incomplete staged deployment")
    os.rename(stage, CODE)
    sync_parent(CODE)
    require(inventory(CODE) == sources, "Deployed source verification failed")
    return {"status": "PARALLEL_MAIN_DEPLOYED_VERIFIED", "root": str(CODE), "sources": sources}


def name_for(seed, wave):
    arm, step = WAVES[wave]
    return f"LH-SATDOSE-MAIN-{seed}-{arm}-s{step:02d}"


def wave_dir(wave):
    return BATCH / f"wave-{wave:02d}"


def walltime(wave):
    return "24:00:00" if WAVES[wave][1] in (0, 24) else "06:00:00"


def intended_command(seed, wave, release):
    d = wave_dir(wave)
    arm, step = WAVES[wave]
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--account=rose",
        "--qos=" + QOS,
        "--job-name=" + name_for(seed, wave),
        "--partition=cluster02",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=66G",
        "--gpus=pro6000:2",
        "--time=" + walltime(wave),
        "--no-requeue",
        "--export=ALL",
        "--chdir=" + str(CODE),
        "--output=" + str(d / f"{seed}-%j.out"),
        "--error=" + str(d / f"{seed}-%j.err"),
        str(CODE / LAUNCHER),
        str(BATCH / "RELEASE.json"),
        release["sha256"],
        str(seed),
        arm,
        str(step),
        str(wave),
    ]


def own_job(job, seed, wave, held):
    fields = fields_for(job)
    d = wave_dir(wave)
    required = dict(
        JobName=name_for(seed, wave),
        Account="rose",
        QOS=QOS,
        Partition="cluster02",
        NumCPUs="8",
        NumTasks="1",
        Requeue="0",
        Command=str(CODE / LAUNCHER),
        WorkDir=str(CODE),
        StdOut=str(d / f"{seed}-{job}.out"),
        StdErr=str(d / f"{seed}-{job}.err"),
    )
    require(
        all(fields.get(k) == v for k, v in required.items())
        and fields.get("NumNodes") in ("1", "1-1")
        and fields.get("UserId", "").startswith(USER + "("),
        "Newly submitted job is not the expected lane",
    )
    require(
        fields["TimeLimit"]
        in (("24:00:00", "1-00:00:00") if walltime(wave) == "24:00:00" else ("06:00:00",)),
        "Unexpected scheduler time limit",
    )
    r = tres(fields["ReqTRES"])
    require(
        all(
            r.get(k) == v
            for k, v in {
                "cpu": "8",
                "node": "1",
                "mem": "66G",
                "gres/gpu": "2",
                "gres/gpu:pro6000": "2",
            }.items()
        ),
        "Scheduler allocated wrong resource type/count",
    )
    if held:
        require(
            fields.get("JobState") == "PENDING"
            and fields.get("Reason") == "JobHeldUser"
            and fields.get("Priority") == "0",
            "Job is not user-held",
        )
    else:
        require(
            fields.get("JobState")
            in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "COMPLETED")
            and fields.get("Reason") not in ("JobHeldAdmin", "JobHeldUser")
            and (fields["JobState"] != "PENDING" or int(fields["Priority"]) > 0),
            "Job release is not confirmed",
        )
        require(
            fields["JobState"] != "COMPLETED" or fields.get("ExitCode") == "0:0",
            "Released job failed",
        )
    return {"fields": fields, "observed_at_utc": now()}


def no_scheduler_duplicates(wave):
    names = [name_for(s, wave) for s in SEEDS]
    live = observed(["squeue", "--user=" + USER, "-h", "-o", "%i|%200j|%T"]).splitlines()
    for line in live:
        row = line.split("|")
        require(len(row) == 3 and row[1].strip() not in names, "Existing matching live MAIN job")
    history = observed(
        [
            "sacct",
            "--user=" + USER,
            "--starttime=2026-11-12T00:00:00",
            "--name=" + ",".join(names),
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName%160,State%40",
        ]
    )
    require(not history.strip(), "Existing matching MAIN accounting record")


def verify_finished_jobs(wave, release):
    all_ids = set()
    for index in range(wave):
        directory = wave_dir(index)
        require(
            directory.resolve(strict=True) == directory
            and not any("ERROR" in p.name or "HOLD" in p.name for p in directory.iterdir()),
            "Prior pair unresolved",
        )
        pair = read(directory / "PAIR.json")
        result = read(directory / "RELEASE_RESULT.json")
        require(
            pair["schema"] == "sr-satdose-main-pair-v1"
            and pair["wave"] == index
            and (pair["arm"], pair["step"]) == WAVES[index]
            and pair["release"] == release
            and result["status"] == "MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED"
            and result["pair"] == ref(directory / "PAIR.json")
            and result["jobs"] == pair["jobs"]
            and result["wave"] == index
            and set(pair["jobs"]) == {str(s) for s in SEEDS},
            "Prior pair admission identity differs",
        )
        for seed in SEEDS:
            job = pair["jobs"][str(seed)]
            require(
                type(job) is int and job > 0 and job not in all_ids,
                "Invalid or repeated prior Slurm ID",
            )
            all_ids.add(job)
            response = observed(
                [
                    "sacct",
                    "-j",
                    str(job),
                    "-nP",
                    "-o",
                    "JobIDRaw,JobName%160,State%40,ExitCode,Account%50,QOS%80,Partition",
                ]
            )
            rows = [line.strip().split("|") for line in response.splitlines() if line.strip()]
            jobstr = str(job)
            require(
                len(rows) == 4
                and all(len(r) == 7 and r[2:4] == ["COMPLETED", "0:0"] for r in rows)
                and {r[0] for r in rows}
                == {jobstr, jobstr + ".extern", jobstr + ".batch", jobstr + ".0"}
                and next(r for r in rows if r[0] == jobstr)[1:]
                == [name_for(seed, index), "COMPLETED", "0:0", "rose", QOS, "cluster02"],
                "Prior pair accounting incomplete or failed",
            )


def dispatch(wave, release, previous_results, admission_check, unlock):
    """Mutate Slurm at most once per lane, from one durable exclusive wave claim."""
    no_scheduler_duplicates(wave)
    resources = capacity()
    if resources["status"] != "CAPACITY_READY":
        return {"deferred": resources}
    d = wave_dir(wave)
    make_dir(d)
    jobs, held_refs = {}, {}
    try:
        put(
            d / "INTENT.json",
            dict(
                wave=wave,
                release=release,
                previous_results=previous_results,
                capacity=resources,
                commands={str(s): intended_command(s, wave, release) for s in SEEDS},
                observed_at_utc=now(),
            ),
        )
        for seed in SEEDS:
            response = run(intended_command(seed, wave, release))
            key = str(seed)
            put(d / f"RESPONSE_{key}.json", response)
            require(
                response["returncode"] == 0
                and re.fullmatch(r"[1-9][0-9]*", response["stdout"].strip()),
                "Held submission failed or ambiguous; inspect this wave without submitting again",
            )
            job = int(response["stdout"].strip())
            require(job not in jobs.values(), "Same job ID returned for both lanes")
            jobs[key] = job
            held = own_job(job, seed, wave, True)
            held_path = d / f"HELD_{key}.json"
            put(held_path, held)
            held_refs[key] = ref(held_path)
        admission_check()
        for seed in SEEDS:
            own_job(jobs[str(seed)], seed, wave, True)
        arm, step = WAVES[wave]
        put(
            d / "PAIR.json",
            dict(
                schema="sr-satdose-main-pair-v1",
                wave=wave,
                arm=arm,
                step=step,
                jobs=jobs,
                release=release,
                held_refs=held_refs,
                previous_results=previous_results,
                observed_at_utc=now(),
            ),
        )
        command = ["scontrol", "release", ",".join(str(jobs[str(s)]) for s in SEEDS)]
        put(
            d / "RELEASE_INTENT.json",
            dict(pair=ref(d / "PAIR.json"), argv=command, observed_at_utc=now()),
        )
        response = run(command)
        put(d / "RELEASE_RESPONSE.json", response)
        require(
            response["returncode"] == 0,
            "Paired release failed/ambiguous; runners cannot generate without RELEASE_RESULT",
        )
        after = {str(seed): own_job(jobs[str(seed)], seed, wave, False) for seed in SEEDS}
        result = dict(
            status="MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED",
            wave=wave,
            jobs=jobs,
            pair=ref(d / "PAIR.json"),
            release=release,
            post_release=after,
            release_response=ref(d / "RELEASE_RESPONSE.json"),
            observed_at_utc=now(),
        )
        unlock()
        put(d / "RELEASE_RESULT.json", result)
        return {"result": result}
    except Exception as exc:
        put(
            d / "ERROR.json",
            dict(
                status="PARALLEL_MAIN_DISPATCH_UNKNOWN_OR_FAILED_NO_RETRY",
                known_jobs=jobs,
                detail=str(exc),
                observed_at_utc=now(),
            ),
        )
        raise


def prepare_budget(P, terminal, controller):
    maximum_triplet = maximum_aux = responses = 0
    for item in terminal.audit_spec["dev_endpoints"].values():
        receipt = P.read(P.bound(item["receipt"]))
        count = receipt["responses"]
        require(
            type(count) is int and count == len(receipt["complete_files"]) == 108,
            "DEV sizing endpoint incomplete",
        )
        responses += count
        for reference in receipt["complete_files"]:
            path = P.bound(reference)
            stem = path.name.removesuffix(".COMPLETE.json")
            size = sum(
                P.ordinary(path.with_name(stem + suffix)).stat().st_size
                for suffix in (".BEGIN.json", ".CAPTURE.json", ".COMPLETE.json")
            )
            maximum_triplet = max(maximum_triplet, size)
        aux = sum(P.bound(receipt[k]).stat().st_size for k in ("raw", "accounting"))
        maximum_aux = max(maximum_aux, math.ceil(aux / count))
    require(responses == 1512 and maximum_triplet > 0, "Sizing needs all 14 verified DEV endpoints")
    at = datetime.now(timezone.utc)
    # Both full endpoints may write simultaneously. Each subsequent wave must
    # satisfy the same live quota/filesystem check before either lane is sent.
    required = 2 * (1952 * (max(1 << 20, 4 * maximum_triplet) + 4 * maximum_aux) + (128 << 20))
    return dict(
        schema="sr-satdose-storage-budget-v1",
        scope="MAIN_ONLY",
        experiment_id=controller.EXPERIMENT,
        evidence_root=str(ROOT),
        design_freeze_sha256=controller.FREEZE_SHA,
        required_bytes=required,
        max_observation_age_seconds=300,
        created_at_utc=at.isoformat(),
        expires_at_utc=(at + timedelta(days=30)).isoformat(),
        combined_wave_responses=3904,
        observed_dev_responses=responses,
        largest_triplet_bytes=maximum_triplet,
        largest_aux_bytes_per_response=maximum_aux,
    )


def batch_release(payload, P, c, terminal, publication, gate_module, wave):
    plan_module = load(CODE / PLAN, PLAN_CODE_SHA, "parallel_plan")
    expected_plan = plan_module.build_plan(
        P.bound(c.read(ROOT / "DESIGN_FREEZE.json")["artifacts"]["main_manifest"])
    )
    plan_bytes = (
        json.dumps(expected_plan, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    require(digest(plan_bytes) == PLAN_SHA, "Frozen two-lane coverage or request order changed")
    gate = gate_module.verify_admission(terminal)
    manifest = payload["terminal_manifest"]
    if not os.path.lexists(BATCH):
        require(
            wave == 0 and not c._existing_slots(),
            "Parallel MAIN starts from untouched MAIN outputs",
        )
        for legacy in (P.PREPARATIONS, LEGACY):
            if os.path.lexists(legacy):
                require(
                    legacy.resolve(strict=True) == legacy
                    and legacy.is_dir()
                    and not list(legacy.iterdir()),
                    "Existing serial MAIN preparation or scheduling attempt",
                )
        budget = prepare_budget(P, terminal, c)
        make_dir(BATCH)
        put(BATCH / "TERMINAL_MANIFEST.json", manifest)
        put(BATCH / "TERMINAL_ADMISSION.json", gate)
        put(BATCH / "PLAN.json", plan_bytes)
        put(BATCH / "STORAGE_BUDGET.json", budget)
        for s in SEEDS:
            put(BATCH / f"LANE_{s}.lock", b"")
        if not LEGACY.exists():
            make_dir(LEGACY)
        put(
            LEGACY_GUARD,
            dict(
                status="PARALLEL_MAIN_EXCLUSIVE_OWNER",
                directory=str(BATCH),
                sources=payload["parallel_sources"],
            ),
        )
    require(
        BATCH.resolve(strict=True) == BATCH
        and LEGACY.resolve(strict=True) == LEGACY
        and set(LEGACY.iterdir()) == {LEGACY_GUARD}
        and read(LEGACY_GUARD)
        == dict(
            status="PARALLEL_MAIN_EXCLUSIVE_OWNER",
            directory=str(BATCH),
            sources=payload["parallel_sources"],
        ),
        "Parallel ownership or original serial submitter exclusion differs",
    )
    require(
        read(BATCH / "TERMINAL_ADMISSION.json") == gate
        and read(BATCH / "TERMINAL_MANIFEST.json") == manifest
        and checked(BATCH / "PLAN.json") == plan_bytes,
        "Parallel admission evidence changed",
    )
    for seed in SEEDS:
        require(checked(BATCH / f"LANE_{seed}.lock") == b"", "Missing or changed lane lock")
    path = BATCH / "RELEASE.json"
    rel = dict(
        schema="sr-satdose-main-parallel-release-v1",
        status="AUDITED_FOR_FIXED_MAIN_PARALLEL_ONLY",
        permission="FIXED_MAIN_ONLY_NO_TRAINING",
        experiment_id=c.EXPERIMENT,
        evidence_root=str(ROOT),
        design_freeze_sha256=c.FREEZE_SHA,
        config_sha256=c.CONFIG_SHA,
        external_root=str(P.EXTERNAL),
        external_sources=payload["external_sources"],
        original_train_release=P.ORIGINAL,
        terminal_authority=dict(
            path=str(P.EXTERNAL / "execution/terminal_authority.py"),
            sha256=payload["external_sources"]["execution/terminal_authority.py"],
        ),
        terminal_spec=publication["terminal_spec"],
        terminal_publication=ref(P.TERMINAL / "PUBLICATION_RECEIPT.json"),
        terminal_manifest=ref(BATCH / "TERMINAL_MANIFEST.json"),
        terminal_admission=ref(BATCH / "TERMINAL_ADMISSION.json"),
        storage_budget=ref(BATCH / "STORAGE_BUDGET.json"),
        parallel_root=str(CODE),
        parallel_sources=payload["parallel_sources"],
        plan=ref(BATCH / "PLAN.json"),
    )
    if not os.path.lexists(path):
        require(
            wave == 0 and not any(BATCH.glob("wave-*")),
            "Missing release for previously attempted wave",
        )
        put(path, rel)
    require(
        read(path) == rel, "Shared parallel release no longer matches independently derived inputs"
    )
    # The original prep verifier loaded and authenticated terminal.modules; the
    # original storage probe consults current disk and user quota, not cached capacity.
    c._configure(sys.modules["_satdose_main_operator_terminal_preverify"], terminal)
    c.storage_check(BATCH, rel["storage_budget"])
    return ref(path), gate


def endpoint_preflight(seed, wave, release):
    arm, step = WAVES[wave]
    command = [
        PYTHON,
        "-I",
        "-B",
        str(CODE / CONTROL),
        "--release-receipt",
        str(BATCH / "RELEASE.json"),
        "--release-sha256",
        release["sha256"],
        "--start-seed",
        str(seed),
        "--arm",
        arm,
        "--step",
        str(step),
    ]
    r = run(command)
    require(
        r["returncode"] == 0,
        "Parallel CPU endpoint preflight failed: " + r["stdout"][-900:] + r["stderr"][-300:],
    )
    report = json.loads(r["stdout"])
    require(
        report.get("status") == "MAIN_PARALLEL_PREFLIGHT_COMPLETE"
        and (report.get("start_seed"), report.get("arm"), report.get("step")) == (seed, arm, step)
        and report.get("permission") == "NO_MODEL_CALLS"
        and report.get("responses") == (1952 if step in (0, 24) else 256)
        and isinstance(report.get("previous_results"), dict),
        "Incomplete or mismatched CPU endpoint preflight",
    )
    return report


def remote_launch(payload):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and ROOT.resolve(strict=True) == ROOT,
        "CPU-only fixed bootstrap required",
    )
    require(
        inventory(CODE) == payload["parallel_sources"], "Unverified parallel operator deployment"
    )
    wave = payload["wave"]
    require(type(wave) is int and 0 <= wave < len(WAVES), "Invalid parallel MAIN wave")
    P = load(CODE / PREPARER, PREPARER_SHA, "parallel_main_baseline_preparer")
    gate_module = load(
        CODE / ADMISSION, payload["parallel_sources"][ADMISSION], "parallel_main_operator_admission"
    )
    require(
        not os.path.lexists(wave_dir(wave)), "Wave already attempted; inspect without resubmission"
    )
    resources = capacity()
    if resources["status"] != "CAPACITY_READY":
        resources.update(
            wave=wave,
            submission_mutations=0,
            release=ref(BATCH / "RELEASE.json") if (BATCH / "RELEASE.json").is_file() else None,
        )
        return {"deferred": resources}
    with ExitStack() as stack:
        stable_l = stack.enter_context(P.existing_lease("FIRST_TWO_LAUNCHER.lock", False))
        stable_w = stack.enter_context(P.existing_lease("DIAGNOSTIC_WORKER.lock", True))
        c, terminal, publication = P.authenticate_remote(payload)
        release, gate = batch_release(payload, P, c, terminal, publication, gate_module, wave)
        expect = {wave_dir(i) for i in range(wave)}
        require(set(BATCH.glob("wave-*")) == expect, "Unknown, missing, or extra wave attempt")
        verify_finished_jobs(wave, release)
        preflights = [endpoint_preflight(s, wave, release) for s in SEEDS]
        previous = preflights[0]["previous_results"]
        require(
            preflights[1]["previous_results"] == previous
            and set(previous) == {f"{s}/{arm}/{step}" for s in SEEDS for arm, step in WAVES[:wave]},
            "Two CPU preflights disagree with the fixed wave prefix",
        )

        def recheck():
            require(
                inventory(CODE) == payload["parallel_sources"]
                and ref(BATCH / "RELEASE.json") == release,
                "Deployment or release changed",
            )
            P.validate_terminal(
                payload["terminal_manifest"],
                P.TERMINAL,
                P.TERMINAL / "PUBLISHER_SOURCE.py",
                remote=True,
            )
            require(
                gate_module.verify_admission(terminal) == gate,
                "Terminal audit proof changed during submission",
            )
            terminal.stable()
            c._hold_free()
            stable_l()
            stable_w()

        recheck()
        outcome = dispatch(wave, release, previous, recheck, stack.close)
        if "deferred" in outcome:
            outcome["deferred"].update(wave=wave, submission_mutations=0, release=release)
        return outcome


def inspect_wave(wave):
    require(type(wave) is int and 0 <= wave < len(WAVES), "Invalid inspection wave")
    d = wave_dir(wave)
    if not os.path.lexists(d):
        return {"status": "WAVE_ABSENT_NOT_RETRY_AUTHORITY", "wave": wave}
    require(d.resolve(strict=True) == d, "Noncanonical wave root")
    files = {p.name: ref(p) for p in d.iterdir() if p.suffix == ".json"}
    if "ERROR.json" in files or "RELEASE_RESULT.json" not in files:
        return {"status": "WAVE_INCOMPLETE_UNKNOWN_NO_RETRY", "wave": wave, "files": files}
    release = ref(BATCH / "RELEASE.json")
    pair = read(d / "PAIR.json")
    r = read(d / "RELEASE_RESULT.json")
    require(
        pair["schema"] == "sr-satdose-main-pair-v1"
        and pair["wave"] == wave
        and (pair["arm"], pair["step"]) == WAVES[wave]
        and pair["release"] == release
        and set(pair["jobs"]) == set(map(str, SEEDS))
        and r["status"] == "MAIN_PARALLEL_PAIR_RELEASE_CONFIRMED"
        and r["wave"] == wave
        and r["release"] == release
        and r["jobs"] == pair["jobs"]
        and r["pair"] == ref(d / "PAIR.json")
        and r["release_response"] == ref(d / "RELEASE_RESPONSE.json"),
        "Wave evidence is inconsistent",
    )
    jobs = pair["jobs"]
    require(
        all(type(j) is int and j > 0 for j in jobs.values()) and len(set(jobs.values())) == 2,
        "Ambiguous job IDs",
    )
    intent = read(d / "INTENT.json")
    require(
        intent["release"] == release
        and intent["wave"] == wave
        and intent["previous_results"] == pair["previous_results"]
        and read(d / "RELEASE_INTENT.json")["pair"] == ref(d / "PAIR.json"),
        "Wave intent proof differs",
    )
    release_cmd = ["scontrol", "release", ",".join(str(jobs[str(s)]) for s in SEEDS)]
    rel = read(d / "RELEASE_RESPONSE.json")
    require(
        read(d / "RELEASE_INTENT.json")["argv"] == release_cmd
        and rel["argv"] == release_cmd
        and rel["returncode"] == 0,
        "Cannot verify pair release response",
    )
    for seed in SEEDS:
        k = str(seed)
        j = jobs[k]
        sub = read(d / f"RESPONSE_{k}.json")
        require(
            intent["commands"][k] == intended_command(seed, wave, release)
            and sub["argv"] == intent["commands"][k]
            and sub["returncode"] == 0
            and sub["stdout"].strip() == str(j)
            and pair["held_refs"][k] == ref(d / f"HELD_{k}.json"),
            "Held job submission proof differs",
        )
        for obs in (read(d / f"HELD_{k}.json"), r["post_release"][k]):
            f = obs["fields"]
            require(
                f["JobId"] == str(j)
                and f["JobName"] == name_for(seed, wave)
                and f["Account"] == "rose"
                and f["QOS"] == QOS
                and f["Partition"] == "cluster02"
                and tres(f["ReqTRES"]).get("gres/gpu:pro6000") == "2",
                "Wrong historical job",
            )
        require(
            read(d / f"HELD_{k}.json")["fields"]["Reason"] == "JobHeldUser"
            and r["post_release"][k]["fields"]["Reason"] not in ("JobHeldAdmin", "JobHeldUser"),
            "Release transition unverified",
        )
    accounting = {}
    for seed in SEEDS:
        job = jobs[str(seed)]
        output = observed(
            [
                "sacct",
                "-X",
                "-nP",
                "-j",
                str(job),
                "-o",
                "JobIDRaw,JobName%160,Account%40,QOS%80,Partition,ReqTRES%256,State%40,ExitCode",
            ]
        )
        rows = [line.strip().split("|") for line in output.splitlines() if line.strip()]
        if rows:
            require(
                len(rows) == 1
                and len(rows[0]) == 8
                and rows[0][:5] == [str(job), name_for(seed, wave), "rose", QOS, "cluster02"]
                and tres(rows[0][5]).get("gres/gpu:pro6000") == "2",
                "Accounting identity differs",
            )
            accounting[str(seed)] = rows[0]
        else:
            accounting[str(seed)] = own_job(job, seed, wave, False)["fields"]
    return {"status": "WAVE_RELEASE_CONFIRMED", "result": r, "files": files, "history": accounting}


def on_server(request):
    mode = request["mode"]
    if mode == "capacity":
        return capacity()
    if mode == "inspect":
        return inspect_wave(request["wave"])
    if mode == "deploy":
        return {"result": deploy(request)}
    require(mode == "launch", "Unknown operator mode")
    return remote_launch(request)


def ssh_action(request, source):
    # Inputs use stdin to avoid Linux's individual command-argument size limit.
    bootstrap = (
        "import hashlib,json,sys,base64; x=json.load(sys.stdin); "
        "src=base64.b64decode(x.pop('operator'),validate=True); "
        "assert hashlib.sha256(src).hexdigest()==sys.argv[1]; "
        "scope={'__name__':'_parallel_main_operator','__file__':sys.argv[2]}; "
        "exec(compile(src,sys.argv[2],'exec'),scope); r=scope['on_server'](x); "
        "print('SATDOSE_PARALLEL_MAIN='+json.dumps(r,sort_keys=True,allow_nan=False)); "
        "sys.exit(3 if 'deferred' in r or r.get('status')=='CAPACITY_DEFERRED' else 0)"
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
            str(CODE / SELF),
        ]
    )
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            "eee-cluster",
            command,
        ],
        input=canonical(dict(request, operator=base64.b64encode(source).decode())),
        capture_output=True,
    )


def transport(request, source, destination):
    if destination is not None:
        destination = destination.absolute()
        require(
            destination.parent.resolve(strict=True) == destination.parent,
            "Invalid local receipt directory",
        )
        intent = destination.with_name(destination.name + ".intent.json")
        stdout = destination.with_name(destination.name + ".stdout.log")
        stderr = destination.with_name(destination.name + ".stderr.log")
        require(
            not any(os.path.lexists(q) for q in (destination, intent, stdout, stderr)),
            "Local attempt occupied",
        )
        put(
            intent,
            dict(
                status="PARALLEL_MAIN_SSH_ATTEMPT_UNKNOWN",
                mode=request["mode"],
                wave=request.get("wave"),
                source_sha256=digest(source),
                payload_sha256=digest(canonical(request)),
                observed_at_utc=now(),
            ),
        )
    result = ssh_action(request, source)
    marks = [v for v in result.stdout.splitlines() if v.startswith(b"SATDOSE_PARALLEL_MAIN=")]
    reply = json.loads(marks[0].split(b"=", 1)[1]) if len(marks) == 1 else None
    if destination is not None:
        put(stdout, result.stdout)
        put(stderr, result.stderr)
        receipt = dict(
            status="PARALLEL_MAIN_OPERATOR_CAPTURED"
            if reply is not None
            else "PARALLEL_MAIN_OPERATOR_UNKNOWN",
            returncode=result.returncode,
            response=reply,
            source_sha256=digest(source),
            raw_logs={
                k: dict(path=str(f), sha256=digest(checked(f)))
                for k, f in [("stdout", stdout), ("stderr", stderr)]
            },
        )
        put(destination, receipt)
        print(json.dumps(receipt, sort_keys=True))
    elif reply is not None:
        print(json.dumps(reply, sort_keys=True))
    else:
        print(
            json.dumps(
                dict(
                    status="NETWORK_UNAVAILABLE"
                    if result.returncode == 255
                    else "OPERATOR_UNKNOWN",
                    returncode=result.returncode,
                    error=result.stderr.decode(errors="replace")[-700:],
                )
            )
        )
    if result.returncode != 0:
        return result.returncode
    if reply is None or (
        request["mode"] == "inspect" and reply.get("status") != "WAVE_RELEASE_CONFIRMED"
    ):
        return 2
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for flag in ("deploy", "check-capacity", "inspect-wave", "execute"):
        modes.add_argument("--" + flag, action="store_true")
    parser.add_argument("--wave", type=int, choices=range(7))
    parser.add_argument("--terminal-dir", type=Path)
    parser.add_argument("--terminal-manifest-sha256")
    parser.add_argument("--terminal-download-receipt-sha256")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    source = checked(Path(__file__).resolve())
    if args.check_capacity:
        return transport({"mode": "capacity"}, source, None)
    if args.inspect_wave:
        require(args.wave is not None, "Inspection needs wave index")
        return transport({"mode": "inspect", "wave": args.wave}, source, None)
    local = Path(__file__).resolve().parent.parent
    assets = {n: checked(local / n) for n in ASSETS}
    sources = {n: digest(raw) for n, raw in assets.items()}
    require(
        sources[PREPARER] == PREPARER_SHA and sources[PLAN] == PLAN_CODE_SHA,
        "Local original or coverage helper changed",
    )
    if not args.deploy and not args.execute:
        print(
            json.dumps(
                dict(
                    status="PARALLEL_MAIN_LOCAL_ONLY",
                    external_root=str(CODE),
                    sources=sources,
                    waves=WAVES,
                    concurrency=2,
                    gpus_per_job=2,
                    total_requests=13760,
                ),
                sort_keys=True,
            )
        )
        return 0
    require(
        args.receipt is not None and args.receipt.is_absolute(),
        "Fresh absolute output receipt required",
    )
    if args.deploy:
        return transport(
            dict(
                mode="deploy",
                sources=sources,
                blobs={k: base64.b64encode(v).decode() for k, v in assets.items()},
            ),
            source,
            args.receipt,
        )
    require(
        args.wave is not None and args.terminal_dir is not None,
        "Submission needs wave and authenticated terminal publication",
    )
    prep = load(local / PREPARER, PREPARER_SHA, "parallel_main_local_preparer")
    manifest = prep.local_terminal(
        args.terminal_dir, args.terminal_manifest_sha256, args.terminal_download_receipt_sha256
    )
    request = dict(
        mode="launch",
        wave=args.wave,
        parallel_sources=sources,
        terminal_manifest=manifest,
        **prep.local_pins(),
    )
    return transport(request, source, args.receipt)


if __name__ == "__main__":
    raise SystemExit(main())
