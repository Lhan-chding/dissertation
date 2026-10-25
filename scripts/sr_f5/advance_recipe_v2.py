"""Operations-only v2 G2-to-G3 bridge; imports the pinned scientific deployment.

No CPU tests, model calls, new recipe, or formal-training transition are performed
here. The science revision is independent of this operations script's location.
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
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f5_semantic_j_recipe_v2_20261026"
V1 = PROJECT / "sr_f5_semantic_j_20261025"
ACCOUNT = "rose"
QOS = "soujanya-poria-startfund-2026-03"
VERIFIED = "TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED"
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}


def canonical(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or not path.is_relative_to(PROJECT)
    ):
        raise PermissionError("Canonical paths inside our project are required")
    return path


def read(path):
    return json.loads(canonical(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical(path.parent)
    if path.is_symlink():
        raise PermissionError("Operation receipt cannot be a symlink")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dependencies(code):
    sys.path[:0] = [str(code / "src"), str(code)]
    modules = {}
    for name in (
        "sr_f4.common",
        "sr_f4.pool",
        "sr_f5.calibration",
        "sr_f5.pilot",
        "sr_f5.pilot_pool",
        "sr_f5.runtime",
        "sr_f5.training",
    ):
        item = importlib.import_module(name)
        if not Path(item.__file__).resolve(strict=True).is_relative_to(code):
            raise PermissionError("Scientific module imported from another deployment: " + name)
        modules[name] = item
    submission = runpy.run_path(str(code / "scripts/sr_f5/submit_g3.py"))
    return SimpleNamespace(
        aggregate=modules["sr_f5.calibration"].aggregate,
        freeze=modules["sr_f5.pilot"].freeze,
        require_frozen=modules["sr_f5.pilot"].require_frozen,
        require_plan=modules["sr_f5.pilot_pool"].require_plan,
        science_digest=modules["sr_f5.calibration"].digest,
        submit=submission["submit"],
        capacity=submission["capacity"],
        gpu_count=modules["sr_f4.pool"].gpu_count,
        slurm=modules["sr_f4.pool"].Slurm(),
        now=modules["sr_f4.common"].now,
    )


def acceptance(code, revision):
    registration = read(V1 / "CONTINUATION/RECIPE_V2.json")
    identity = registration.get("identity", {})
    config = read(ROOT / "CONFIG.json")
    ready, tests = (read(ROOT / name) for name in ("CPU_READY.json", "CPU_TESTS.json"))
    if (
        identity.get("new_source_revision") != revision
        or identity.get("new_root") != str(ROOT)
        or identity.get("recipe_version") != 2
        or identity.get("new_config_digest") != digest(config)
        or read(V1 / "CONTINUATION/RECIPE_V2_CONFIG.json") != config
        or config.get("numerics") != identity.get("numerics")
        or ready.get("status") != "CPU_READY_FOR_G2_PREFLIGHT"
        or ready.get("code_revision") != revision
        or tests.get("status") != "PASS"
        or tests.get("returncode") != 0
        or tests.get("code_revision") != revision
        or tests.get("model_calls") != 0
        or "tests/sr_f5" not in tests.get("test_paths", [])
        or read(ROOT / "G2_SPEC_FROZEN.json").get("code_revision") != revision
        or read(ROOT / "DATA/HISTORICAL_WORLD_AUDIT.json").get("status") != "PASS"
        or read(ROOT / "ENVIRONMENT_LOCK.json").get("code_revision") != revision
    ):
        raise PermissionError("Registered v2 source/config/actual CPU acceptance differs")
    return registration, dict(
        source_code=str(code),
        science_revision=revision,
        registration_digest=digest(registration),
        config_digest=digest(config),
        cpu_tests_digest=digest(tests),
        cpu_tests_reused="CPU_TESTS.json; no rerun",
    )


def pairs(code, revision, registration, deps, user):
    results, missing = [], []
    for pair in (0, 1):
        path = ROOT / f"G2_SUBMISSION_PAIR{pair}.json"
        if not path.exists():
            missing.append(pair)
            continue
        item = read(path)
        if item.get("status") != "RELEASED_VERIFIED":
            return "G2_SUBMISSION_REQUIRES_DIAGNOSIS", []
        job = item.get("job_id", "")
        actual = item.get("allocation", {})
        if (
            not isinstance(job, str)
            or not job.isdecimal()
            or item.get("pair") != pair
            or item.get("shards") != [2 * pair, 2 * pair + 1]
            or item.get("expected_answers") != 2304
            or item.get("science_revision") != revision
            or item.get("source_code") != str(code)
            or item.get("registered_recipe") != registration
            or actual.get("JobId") != job
            or actual.get("JobName") != f"LH-SRF5-G2-P{pair}"
            or actual.get("UserId", "").split("(")[0] != user
            or actual.get("Account") != ACCOUNT
            or actual.get("QOS") != QOS
            or deps.gpu_count(actual.get("ReqTRES")) != 2
            or actual.get("WorkDir") != str(code)
            or actual.get("Command") != str(code / "scripts/sr_f5/calibration.sbatch")
            or actual.get("StdOut") != str(ROOT / "slurm" / f"G2-P{pair}-{job}.log")
        ):
            raise PermissionError("G2 pair allocation is not bound to this recipe/root")
        results.append(item)
    if missing:
        return "WAITING_FOR_G2_PAIR_RECEIPT", results
    if len({row["job_id"] for row in results}) != 2:
        raise PermissionError("Duplicate G2 allocation identity")
    return "PAIR_RECEIPTS_VERIFIED", results


def terminal(deps, records, user):
    active = deps.slurm.user_job_ids(user)
    if any(row["job_id"] in active for row in records):
        return "WAITING_FOR_G2", []
    states = [deps.slurm.terminal(row["job_id"]) for row in records]
    if any(row is None for row in states):
        return "WAITING_FOR_ACCOUNTING", []
    for actual, registered in zip(states, records, strict=True):
        if actual.get("job_id") != registered["job_id"] or (
            actual.get("user"),
            actual.get("account"),
            actual.get("qos"),
        ) != (user, ACCOUNT, QOS):
            raise PermissionError("Terminal allocation owner or identity differs")
    if any(row.get("state") not in TERMINAL_STATES for row in states):
        return "WAITING_FOR_ACCOUNTING", states
    if any(row.get("state") != "COMPLETED" or row.get("exit_code") != "0:0" for row in states):
        return "G2_TECHNICAL_FAILURE_REQUIRES_DIAGNOSIS", states
    return "G2_COMPLETE", states


def advance(args, *, deps=None, user=None, launcher=None):
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Full registered science revision required")
    code = canonical(args.code)
    canonical(ROOT)
    launcher = canonical(Path(launcher or __file__))
    deps, user = deps or dependencies(code), user or getpass.getuser()
    registration, accepted = acceptance(code, args.revision)
    decision = ROOT / "CONTINUATION/RECIPE_V2_ADVANCE.json"
    aggregation = ROOT / "CONTINUATION/RECIPE_V2_G2_AGGREGATION.json"
    provenance = dict(
        **accepted,
        operations_script=str(launcher),
        operations_script_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
    )

    def finish(status, **details):
        result = dict(at=deps.now(), status=status, **provenance, **details)
        write(decision, result)
        return result

    lock_path = ROOT / "DEVELOPMENT_ADVANCE.lock"
    if lock_path.is_symlink():
        raise PermissionError("Development lock cannot be a symlink")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submission_path = ROOT / "G3_SUBMISSION_DUAL_GPU.json"
        frozen_path = ROOT / "G3/FROZEN.json"
        # This branch MUST precede aggregation: G2's timestamp is in G3's identity.
        if submission_path.exists():
            prior = read(submission_path)
            if prior.get("status") != VERIFIED:
                return finish("G3_SUBMISSION_REQUIRES_DIAGNOSIS_NO_RETRY", submission=prior)
            frozen = deps.require_frozen(ROOT)
            plan = deps.require_plan(ROOT, args.revision)
            if (
                frozen.get("code_revision") != args.revision
                or prior.get("code_revision") != args.revision
                or prior.get("source_code") != str(code)
                or prior.get("pool_plan_digest") != deps.science_digest(plan)
                or len(prior.get("jobs", [])) != 2
                or {row.get("pair") for row in prior["jobs"]} != {0, 1}
                or len({row.get("job_id") for row in prior["jobs"]}) != 2
            ):
                raise PermissionError("Existing G3 submission identity differs")
            return finish("G3_ALREADY_SUBMITTED_DO_NOT_DUPLICATE", submission=prior)
        status, receipts = pairs(code, args.revision, registration, deps, user)
        if status != "PAIR_RECEIPTS_VERIFIED":
            return finish(status)
        status, states = terminal(deps, receipts, user)
        if status != "G2_COMPLETE":
            return finish(status, terminals=states)
        bindings = dict(**accepted, pair_receipt_digests=[digest(r) for r in receipts])
        if frozen_path.exists():
            frozen = deps.require_frozen(ROOT)
            if frozen.get("code_revision") != args.revision:
                raise PermissionError("Existing G3 frozen science revision differs")
        else:
            if aggregation.exists():
                previous = read(aggregation)
                if previous.get("status") != "COMPLETE":
                    return finish("G2_AGGREGATION_REQUIRES_DIAGNOSIS", aggregation=previous)
                if (
                    previous.get("bindings") != bindings
                    or previous.get("gate_digest") != digest(read(ROOT / "GATES/G2.json"))
                    or previous.get("metrics_digest") != digest(read(ROOT / "G2/METRICS.json"))
                ):
                    raise PermissionError("Completed aggregation evidence changed")
                gate = previous["gate"]
            else:
                gate_path, metrics_path = ROOT / "GATES/G2.json", ROOT / "G2/METRICS.json"
                if metrics_path.exists() or (
                    gate_path.exists() and read(gate_path).get("status") != "NOT_RUN"
                ):
                    return finish("G2_ARTIFACTS_WITHOUT_AGGREGATION_RECEIPT_REQUIRES_DIAGNOSIS")
                write(aggregation, dict(status="STARTED", bindings=bindings, at=deps.now()))
                report = deps.aggregate(ROOT)
                gate = report["gate"]
                if report != read(metrics_path):
                    raise PermissionError("Aggregate return and persisted metrics differ")
                write(
                    aggregation,
                    dict(
                        status="COMPLETE",
                        bindings=bindings,
                        gate=gate,
                        gate_digest=digest(read(gate_path)),
                        metrics_digest=digest(report),
                        at=deps.now(),
                    ),
                )
            if gate.get("status") == "FAIL":
                return finish("STOP_RECIPE_V2_FEASIBILITY", g2_gate=gate, terminals=states)
            if gate.get("status") != "PASS":
                return finish("G2_AGGREGATION_NOT_COMPLETE_REQUIRES_DIAGNOSIS", g2_gate=gate)
            deps.freeze(ROOT, args.revision)
        jobs, gpus, _ = deps.capacity(deps.slurm, user, set())
        if jobs + 2 > 5 or gpus + 4 > 5:
            return finish("WAITING_FOR_G3_CAPACITY", observed_jobs=jobs, reserved_gpus=gpus)
        try:
            submitted = deps.submit(
                SimpleNamespace(root=ROOT, code=code, revision=args.revision),
                slurm=deps.slurm,
                user=user,
            )
        except RuntimeError:
            capacity_path = ROOT / "G3/CAPACITY_LAST_CHECK.json"
            if (
                not submission_path.exists()
                and capacity_path.exists()
                and read(capacity_path).get("status") == "WAITING_FOR_CAPACITY"
            ):
                return finish("WAITING_FOR_G3_CAPACITY", capacity=read(capacity_path))
            raise
        return finish("G3_SUBMITTED_AFTER_REAL_G2_PASS", submission=submitted, terminals=states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    print(json.dumps(advance(parser.parse_args()), indent=2), flush=True)
