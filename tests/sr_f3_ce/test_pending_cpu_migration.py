"""CPU-only regressions for exact-job CPU relocation; no real Slurm calls."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

CHECKOUT = Path(__file__).resolve().parents[2]
QOS = "soujanya-poria-startfund-2026-03"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class FakeSlurm:
    """Observable mutation boundary, including successful but unknown updates."""

    def __init__(self, intent):
        self.calls = []
        self.intent = intent
        self.fields = dict(
            JobId="215520",
            JobName=intent["job_name"],
            UserId="varun024(1001)",
            Account="rose",
            QOS=QOS,
            Partition="cluster02",
            Command=intent["script"],
            Comment=intent["comment"],
            NumCPUs="4",
            NumTasks="1",
            **{"CPUs/Task": "4"},
            NumNodes="1",
            ReqNodeList="cpu-2",
            NodeList="(null)",
            ReqTRES="cpu=4,mem=12G,node=1,billing=4",
            AllocTRES="(null)",
            JobState="PENDING",
            Reason="Resources",
            TimeLimit="3-00:00:00",
            Requeue="0",
            RunTime="00:00:00",
            Features="cpu_ok",
            StartTime="Unknown",
            EndTime="Unknown",
            Priority="100",
            Dependency="(null)",
            Reservation="(null)",
            ExcNodeList="(null)",
        )
        self.node = dict(
            NodeName="cpu-1",
            CPUAlloc="56",
            CPUTot="96",
            RealMemory="339968",
            AllocMem="172032",
            FreeMem="30000",
            MemSpecLimit="5120",
            State="MIXED",
            Partitions="cluster02",
            AvailableFeatures="cpu_ok",
            ActiveFeatures="cpu_ok",
        )
        self.start = "Unknown"
        self.elapsed = "0"
        self.alloc = ""
        self.on_mutation = None
        self.responses = {}

    @property
    def mutations(self):
        return [
            v
            for v in self.calls
            if v[:2] in (["scontrol", "hold"], ["scontrol", "update"], ["scontrol", "release"])
        ]

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        assert argv[0] != "sbatch", "Migration must never submit a replacement"
        if argv[:2] in (["scontrol", "hold"], ["scontrol", "update"], ["scontrol", "release"]):
            action = argv[1]
            if action == "hold":
                self.fields.update(Reason="JobHeldUser", Priority="0")
            elif action == "update":
                assert "ReqNodeList=cpu-1" in argv
                self.fields["ReqNodeList"] = "cpu-1"
            else:
                self.fields.update(Reason="Resources", Priority="100")
            if self.on_mutation:
                self.on_mutation(action, self)
            if action in self.responses:
                response = self.responses[action]
                if isinstance(response, BaseException):
                    raise response
                return subprocess.CompletedProcess(argv, *response)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "sacct":
            state = self.fields["JobState"]
            output = (
                "|".join(
                    [
                        "215520",
                        state,
                        "0:0",
                        self.start,
                        self.elapsed,
                        self.alloc,
                        self.fields["NodeList"],
                        "varun024",
                        "rose",
                        QOS,
                        self.intent["job_name"],
                    ]
                )
                + "\n"
            )
        elif argv[:3] == ["scontrol", "show", "job"]:
            output = " ".join(f"{k}={v}" for k, v in self.fields.items()) + "\n"
        elif argv[:3] == ["scontrol", "show", "node"]:
            output = " ".join(f"{k}={v}" for k, v in self.node.items()) + "\n"
        elif argv[:3] == ["scontrol", "write", "batch_script"]:
            output = Path(self.intent["script"]).read_text()
            if argv[-1] not in ("-", "/dev/stdout", "215520"):
                Path(argv[-1]).write_text(output)
        elif argv[0] == "squeue":
            output = ""
        else:
            raise AssertionError(f"Unexpected Slurm command {argv}")
        return subprocess.CompletedProcess(argv, 0, output, "")


@pytest.fixture
def migration_env(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "exact_cpu_migration_tests", CHECKOUT / "scripts/sr_f3_ce/migrate_cpu_node.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(module, "TARGET_ROOT", str(root))
    key = "job-164520339dd44ae6"
    claim = "20225757473db3656141775e6e3071850f4ba14e1514c545a80af0368e00e4cd"
    jobdir = root / "orchestration/jobs" / key
    jobdir.mkdir(parents=True)
    script = jobdir / "run.sbatch"
    script.write_text("#!/bin/bash\n# frozen original worker\nexit 0\n")
    intent = dict(
        root=str(root),
        key=key,
        job_id="215520",
        claim_id=claim,
        kind="release",
        task_ids=["release"],
        cpu_node="cpu-2",
        gpu_count=0,
        cpus=4,
        memory_gib=12,
        qos=QOS,
        partition="cluster02",
        user="varun024",
        job_name="SRF3CE-release-N0001-LH",
        script=str(script),
        script_sha256=digest(script),
        comment=claim,
        execution_entry_schema=2,
        worker_entry_commands={"release": ["python", "worker", "release"]},
    )
    save(jobdir / "INTENT.json", intent)
    save(jobdir / "STATE.json", dict(key=key, job_id="215520", status="SUBMITTED", claim_id=claim))
    slurm = FakeSlurm(intent)
    save(jobdir / "ALLOCATION_VERIFIED.json", dict(job_id="215520", fields=slurm.fields))
    save(jobdir / "RELEASE_INTENT.json", dict(job_id="215520", claim_id=claim))
    save(jobdir / "RELEASE.json", dict(job_id="215520", returncode=0))
    old = "35800b02b1ef1eff2378e4c933393b4ea2668e4d"
    for relative in [
        "orchestration/CONFIG.json",
        "EXECUTION_FREEZE.json",
        "RUN_IDENTITY.json",
        f"operations/{old}/OPERATION_IDENTITY.json",
    ]:
        save(
            root / relative,
            dict(user="varun024", identity="preserved", cpu_policy=dict(cpu_node="cpu-2")),
        )
    relatives = [
        f"orchestration/jobs/{key}/{name}"
        for name in [
            "INTENT.json",
            "run.sbatch",
            "ALLOCATION_VERIFIED.json",
            "RELEASE_INTENT.json",
            "RELEASE.json",
        ]
    ]
    relatives += [
        "orchestration/CONFIG.json",
        "EXECUTION_FREEZE.json",
        "RUN_IDENTITY.json",
        f"operations/{old}/OPERATION_IDENTITY.json",
    ]
    intent.update(
        run_identity_sha256=digest(root / "RUN_IDENTITY.json"),
        execution_freeze_sha256=digest(root / "EXECUTION_FREEZE.json"),
        pool_operation_identity_sha256=digest(root / f"operations/{old}/OPERATION_IDENTITY.json"),
    )
    save(jobdir / "INTENT.json", intent)
    state = json.loads((jobdir / "STATE.json").read_text())
    state["intent_sha256"] = digest(jobdir / "INTENT.json")
    save(jobdir / "STATE.json", state)
    request = "Migrate exact pending 215520 cpu-2 to cpu-1 without resubmission or science changes."
    authorization = dict(
        schema_version=1,
        scope="SRF3CE_EXACT_CPU_NODE_MIGRATION",
        root=str(root),
        job_id="215520",
        job_key=key,
        claim_id=claim,
        from_node="cpu-2",
        to_node="cpu-1",
        scientific_settings_changed=False,
        preserve_history=True,
        no_resubmit=True,
        source_sha256=digest(module.__file__),
        operation_commit="e" * 40,
        user_request=request,
        user_request_sha256=hashlib.sha256(request.encode()).hexdigest(),
        files={r: digest(root / r) for r in relatives},
    )
    auth = root / "operations" / ("e" * 40) / "CPU_MIGRATION_AUTHORIZATION.json"
    auth.parent.mkdir(parents=True)
    deployed = auth.parent / "migrate_cpu_node.py"
    shutil.copyfile(module.__file__, deployed)
    monkeypatch.setattr(module, "__file__", str(deployed))
    guard = auth.parent / "cpu_dispatcher_pause.py"
    shutil.copyfile(CHECKOUT / "scripts/sr_f3_ce/cpu_dispatcher_pause.py", guard)
    authorization["guard_source_sha256"] = digest(guard)
    save(auth, authorization)
    pause = root / "operations/cpu_node_migration_guard/PAUSED.json"
    save(
        pause,
        dict(
            root=str(root),
            job_id="211333",
            source_sha256=digest(guard),
            process=dict(
                state="T", pid=123, startticks="456", argv=["python", "dispatch_registered.py"]
            ),
        ),
    )
    monkeypatch.setattr(module, "verify_paused_process", lambda process: None)

    def make():
        obj = module.Migration(root, auth, runner=slurm)
        original = obj.migrate
        obj.migrate = lambda: original(pause_receipt=pause)
        return obj

    baseline = {r: (root / r).read_bytes() for r in relatives}
    return SimpleNamespace(
        module=module,
        root=root,
        auth=auth,
        authorization=authorization,
        intent=intent,
        jobdir=jobdir,
        slurm=slurm,
        baseline=baseline,
        pause=pause,
        guard=guard,
        make=make,
    )


def assert_preserved(env):
    assert {r: (env.root / r).read_bytes() for r in env.baseline} == env.baseline
    assert not any(c[0] == "sbatch" for c in env.slurm.calls)


def test_original_job_migrates_once_and_preserves_all_original_bytes(migration_env):
    env = migration_env
    env.make().migrate()
    assert [c[1] for c in env.slurm.mutations] == ["hold", "update", "release"]
    assert env.slurm.fields["ReqNodeList"] == "cpu-1"
    assert_preserved(env)
    previous = list(env.slurm.mutations)
    assert env.make().migrate()["status"] == "MIGRATED"
    assert env.slurm.mutations == previous
    assert_preserved(env)


@pytest.mark.parametrize(
    "field,value",
    [
        ("JobState", "RUNNING"),
        ("NodeList", "cpu-2"),
        ("AllocTRES", "cpu=4,mem=12G,node=1"),
        ("RunTime", "00:00:01"),
    ],
)
def test_started_or_allocated_job_is_never_mutated(migration_env, field, value):
    env = migration_env
    env.slurm.fields[field] = value
    assert env.make().migrate()["status"] == "ABORTED_ALREADY_STARTED"
    assert env.slurm.mutations == []
    assert_preserved(env)


def test_hold_race_start_stops_before_update_or_release(migration_env):
    env = migration_env

    def start(action, slurm):
        if action == "hold":
            slurm.fields.update(
                JobState="RUNNING", NodeList="cpu-2", AllocTRES="cpu=4,mem=12G,node=1"
            )
            slurm.start = "2026-10-20T14:00:00"
            slurm.elapsed = "1"
            slurm.alloc = "cpu=4,mem=12G,node=1"

    env.slurm.on_mutation = start
    assert env.make().migrate()["status"] == "ABORTED_ALREADY_STARTED"
    assert [c[1] for c in env.slurm.mutations] == ["hold"]
    assert_preserved(env)


@pytest.mark.parametrize(
    "field,value",
    [
        ("JobId", "999999"),
        ("ReqNodeList", "cpu-3"),
        ("NumCPUs", "8"),
        ("ReqTRES", "cpu=4,mem=24G,node=1"),
        ("QOS", "rose"),
        ("Command", "/different/script"),
        ("Comment", "different-claim"),
        ("TimeLimit", "1-00:00:00"),
        ("Requeue", "1"),
        ("Partition", "other"),
    ],
)
def test_scheduler_identity_and_resource_changes_are_refused(migration_env, field, value):
    env = migration_env
    env.slurm.fields[field] = value
    with pytest.raises((PermissionError, RuntimeError)):
        env.make().migrate()
    assert env.slurm.mutations == []
    assert_preserved(env)


@pytest.mark.parametrize("action", ["hold", "update", "release"])
def test_unknown_scheduler_result_is_never_retried(migration_env, action):
    env = migration_env
    env.slurm.responses[action] = subprocess.TimeoutExpired(action, 60)
    with pytest.raises((PermissionError, RuntimeError, subprocess.TimeoutExpired)):
        env.make().migrate()
    assert [c[1] for c in env.slurm.mutations] == ["hold", "update", "release"][
        : ["hold", "update", "release"].index(action) + 1
    ]
    first = list(env.slurm.mutations)
    with pytest.raises((PermissionError, RuntimeError, FileExistsError)):
        env.make().migrate()
    assert env.slurm.mutations == first
    assert_preserved(env)


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "215521"),
        ("job_key", "job-other"),
        ("claim_id", "0" * 64),
        ("to_node", "cpu-3"),
        ("scientific_settings_changed", True),
        ("no_resubmit", False),
        ("source_sha256", "0" * 64),
    ],
)
def test_authorization_mismatch_is_refused_before_any_scheduler_mutation(
    migration_env, field, value
):
    env = migration_env
    auth = copy.deepcopy(env.authorization)
    auth[field] = value
    save(env.auth, auth)
    with pytest.raises((PermissionError, RuntimeError)):
        env.make().migrate()
    assert env.slurm.mutations == []


def test_bound_original_file_tampering_is_refused(migration_env):
    env = migration_env
    (env.jobdir / "RELEASE.json").write_text("{}\n")
    with pytest.raises((PermissionError, RuntimeError)):
        env.make().migrate()
    assert env.slurm.mutations == []


def test_resource_change_after_hold_never_releases(migration_env):
    env = migration_env

    def change(action, slurm):
        if action == "update":
            slurm.fields["ReqTRES"] = "cpu=4,mem=24G,node=1"

    env.slurm.on_mutation = change
    with pytest.raises((PermissionError, RuntimeError)):
        env.make().migrate()
    assert [c[1] for c in env.slurm.mutations] == ["hold", "update"]
    assert_preserved(env)


@pytest.fixture
def efficient_migration_env(tmp_path, monkeypatch):
    from test_dispatch_efficiency import efficient_env

    return efficient_env.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("scheduler_state", ["PENDING", "RUNNING"])
def test_real_fast_dispatcher_waits_for_cpu1_release_without_strict_live_refresh(
    efficient_migration_env, monkeypatch, scheduler_state
):
    """Keep the actual preexisting wait implementation; only ledger payloads are toy."""
    from test_dispatch_efficiency import fast_driver

    env = efficient_migration_env
    env.efficiency.start()
    original_scientific = {
        p: p.read_bytes()
        for p in [env.root / "EXECUTION_FREEZE.json", env.root / "orchestration/CONFIG.json"]
    }
    claimed = {t for intent, state in env.helper.state_rows(env.root) for t in intent["task_ids"]}
    for task, definition in env.api.orchestration.TASKS.items():
        if task in claimed:
            continue
        key = "job-" + hashlib.sha256(task.encode()).hexdigest()[:16]
        directory = env.root / "orchestration/jobs" / key
        active = task == "release"
        env.helper.save(
            directory / "INTENT.json",
            dict(
                key=key,
                kind=definition["kind"],
                task_ids=[task],
                qos=QOS,
                gpu_count=0,
                run_identity_sha256=env.value["run_identity_sha256"],
            ),
        )
        env.helper.save(
            directory / "STATE.json",
            dict(
                key=key,
                status="SUBMITTED" if active else "COMPLETE",
                job_id="215520" if active else None,
                gpu_count=0,
                qos=QOS,
                run_identity_sha256=env.value["run_identity_sha256"],
                intent_sha256=env.helper.sha(directory / "INTENT.json"),
            ),
        )
    env.raw.rows["215520"] = dict(
        JobId="215520",
        JobName="SRF3CE-release-LH",
        QOS=QOS,
        UserId="alice(1)",
        ReqTRES="cpu=4,mem=12G,node=1",
        AllocTRES="cpu=4,mem=12G,node=1" if scheduler_state == "RUNNING" else "(null)",
        ReqNodeList="cpu-1",
        NodeList="cpu-1" if scheduler_state == "RUNNING" else "(null)",
        JobState=scheduler_state,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Active relocated release must not trigger full status, refresh, or submission"
        )

    for method in ("status", "tick", "refresh", "submit"):
        monkeypatch.setattr(env.api.controller, method, forbidden)
    driver = fast_driver(env)
    assert driver.poll() == "WAIT_FIXED_TASKS"
    assert driver.poll() == "UNCHANGED"
    assert env.transport.submitted == []
    assert all(p.read_bytes() == body for p, body in original_scientific.items())


def test_frozen_allocation_policy_stays_strict_and_old_receipt_unchanged(migration_env):
    """Migration is an explicit extra gate, never a global old-policy relaxation."""
    from sr_f3_ce.orchestration import verify_allocation

    env = migration_env
    original = copy.deepcopy(env.slurm.fields)
    verify_allocation(original, env.intent, "varun024")
    relocated = dict(original, ReqNodeList="cpu-1")
    with pytest.raises(PermissionError, match="CPU allocation policy differs"):
        verify_allocation(relocated, env.intent, "varun024")
    assert env.intent["cpu_node"] == "cpu-2"
    assert original["ReqNodeList"] == "cpu-2"
    verify_allocation(original, env.intent, "varun024")


@pytest.fixture
def pooled_migration_env(tmp_path, monkeypatch):
    from test_pooled_execution import pooled_env

    return pooled_env.__wrapped__(tmp_path, monkeypatch)


def test_original_pooled_release_entry_and_terminal_verification_keep_historical_receipts(
    pooled_migration_env, monkeypatch
):
    """Run old release entry with toy data, using real old history verifiers."""
    from test_pooled_execution import complete_pool, intent_for

    from sr_f3_ce import endpoint_runner, orchestration

    env = pooled_migration_env
    for kind, total in [("train", 12), ("eval", 15)]:
        done = 0
        while done < total:
            state = env.controller.submit(kind)
            complete_pool(env, state, monkeypatch)
            done += len(intent_for(env, state)["task_ids"])
    state = env.controller.submit("release")
    directory = env.root / "orchestration/jobs" / state["key"]
    original = {
        name: (directory / name).read_bytes()
        for name in [
            "INTENT.json",
            "run.sbatch",
            "ALLOCATION_VERIFIED.json",
            "RELEASE_INTENT.json",
            "RELEASE.json",
        ]
    }
    env.slurm.rows[state["job_id"]].update(ReqNodeList="cpu-1", NodeList="cpu-1")
    before = endpoint_runner._completed(env.root)
    assert "release" not in before
    assert before == set(orchestration.TASKS) - {"release"}
    status, launches = complete_pool(env, state, monkeypatch)
    assert launches and status["all_complete"] and not status["blocked"]
    assert env.controller.verify_job(state) == ["release"]
    assert endpoint_runner._completed(env.root) == set(orchestration.TASKS)
    assert {name: (directory / name).read_bytes() for name in original} == original


def test_shared_submit_lock_contention_prevents_any_scheduler_action(migration_env):
    env = migration_env
    with (env.root / "orchestration/SUBMIT.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(BlockingIOError):
                env.make().migrate()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    assert env.slurm.calls == []
    assert_preserved(env)


@pytest.mark.parametrize(
    "marker",
    ["STARTED.json", "release.LAUNCH.json", "release.EXIT.json", "POOL_END.json", "COMPLETE.json"],
)
def test_execution_receipts_stop_even_if_scheduler_looks_pending(migration_env, marker):
    env = migration_env
    save(env.jobdir / marker, dict(job_id="215520"))
    assert env.make().migrate()["status"] == "ABORTED_ALREADY_STARTED"
    assert env.slurm.mutations == []


@pytest.fixture
def final_migration_env(migration_env, monkeypatch):
    env = migration_env
    deployed = env.auth.parent / "migrate_cpu_node.py"
    env.make().migrate()
    env.slurm.fields.update(
        JobState="COMPLETED", NodeList="cpu-1", AllocTRES="cpu=4,mem=12G,node=1"
    )
    env.slurm.start = "2026-10-20T15:00:00"
    env.slurm.elapsed = "60"
    env.slurm.alloc = "cpu=4,mem=12G,node=1"
    claim = env.authorization["claim_id"]
    save(
        env.jobdir / "COMPLETE.json",
        dict(status="COMPLETE", job_id="215520", task_ids=["release"], claim_id=claim),
    )
    save(
        env.jobdir / "POOL_END.json",
        dict(status="COMPLETE", job_id="215520", verified_tasks=["release"], exits={"release": 0}),
    )
    save(env.jobdir / "STARTED.json", dict(job_id="215520", devices=[]))
    save(
        env.jobdir / "release.LAUNCH.json",
        dict(claim_id=claim, command=["python", "worker", "release"], device=""),
    )
    save(env.jobdir / "release.EXIT.json", dict(returncode=0))
    task = env.root / "tasks/release/COMPLETE.json"
    save(
        task,
        dict(
            job_id="215520",
            claim_id=claim,
            status="COMPLETE",
            run_identity_sha256=env.intent["run_identity_sha256"],
            execution_freeze_sha256=env.intent["execution_freeze_sha256"],
        ),
    )
    observer = (
        env.root / "accounting/observability" / env.authorization["job_key"] / "release/FINAL.json"
    )
    save(observer, dict(status="COMPLETE", task_receipt_sha256=digest(task)))
    env.archive = env.root / "delivery/SR_F3_CE_20261016.zip"
    env.builder = Path(str(env.archive) + ".verification.json")
    env.dispatcher = env.root / "operations/dispatcher/attempts/0007/DELIVERY_VERIFIED.json"
    env.deployed = deployed
    save(env.dispatcher.parent / "END.json", dict(status="DONE"))
    migrated = (
        env.root / "operations/cpu_node_migration" / env.authorization["job_key"] / "MIGRATED.json"
    )
    save(
        env.pause.parent / "ACTION_RESULT.json",
        dict(status="RETURNED", result=json.loads(migrated.read_text())),
    )
    save(
        env.pause.parent / "RESUMED.json",
        dict(
            job_id="211333",
            process=dict(
                state="S", pid=123, startticks="456", argv=["python", "dispatch_registered.py"]
            ),
            worker_reaped_before_resume=True,
        ),
    )

    def package(omit=None, corrupt=None):
        env.archive.parent.mkdir(parents=True, exist_ok=True)
        files = [p for p in (env.root / "operations/cpu_node_migration").rglob("*") if p.is_file()]
        files += [
            env.auth,
            deployed,
            env.guard,
            env.pause,
            env.pause.parent / "RESUMED.json",
            env.pause.parent / "ACTION_RESULT.json",
        ]
        files += [env.root / relative for relative in env.authorization["files"]]
        with zipfile.ZipFile(env.archive, "w") as zipped:
            for p in files:
                if omit and p.name == omit:
                    continue
                body = p.read_bytes()
                if corrupt and p.name == corrupt:
                    body += b"\n"
                zipped.writestr("evidence/" + str(p.relative_to(env.root)), body)
        save(
            env.builder,
            dict(
                status="VERIFIED_DELIVERY",
                checks={k: "PASS" for k in env.module.DELIVERY_CHECKS},
                run_identity_sha256=digest(env.root / "RUN_IDENTITY.json"),
                archive=dict(
                    path=str(env.archive),
                    sha256=digest(env.archive),
                    bytes=env.archive.stat().st_size,
                    entries=len(zipped.namelist()),
                ),
            ),
        )
        save(env.dispatcher, dict(receipt=str(env.builder), sha256=digest(env.builder)))

    env.package = package
    package()
    return env


def test_final_acceptance_requires_both_science_and_archived_migration_chain(final_migration_env):
    env = final_migration_env
    result = env.make().verify_final()
    assert result["status"] == "FINAL_MIGRATION_VERIFIED"
    assert result["actual_node"] == "cpu-1"
    assert result["archive"]["sha256"] == digest(env.archive)
    assert result["archived_migration_files"]
    assert (env.root / "delivery/FINAL_MIGRATION_VERIFIED.json").exists()
    assert_preserved(env)


@pytest.mark.parametrize(
    "filename",
    [
        "MIGRATED.json",
        "RELEASE_GATE.json",
        "CPU_MIGRATION_AUTHORIZATION.json",
        "migrate_cpu_node.py",
    ],
)
def test_old_scientific_pass_cannot_accept_archive_missing_migration(final_migration_env, filename):
    env = final_migration_env
    env.package(omit=filename)
    with pytest.raises(PermissionError, match="absent from delivery"):
        env.make().verify_final()
    assert not (env.root / "delivery/FINAL_MIGRATION_VERIFIED.json").exists()


def test_archived_migration_tamper_refused_even_with_rehashed_builder(final_migration_env):
    env = final_migration_env
    env.package(corrupt="RELEASE_GATE.json")
    with pytest.raises(PermissionError, match="Archived migration evidence differs"):
        env.make().verify_final()


@pytest.mark.parametrize("field,value", [("JobState", "RUNNING"), ("NodeList", "cpu-2")])
def test_final_wrong_or_incomplete_execution_refused(final_migration_env, field, value):
    env = final_migration_env
    env.slurm.fields[field] = value
    with pytest.raises(PermissionError, match="terminal execution differs"):
        env.make().verify_final()


@pytest.mark.parametrize("target", ["COMPLETE.json", "release.EXIT.json", "POOL_END.json"])
def test_final_incomplete_release_receipt_refused(final_migration_env, target):
    env = final_migration_env
    save(env.jobdir / target, {})
    with pytest.raises(PermissionError):
        env.make().verify_final()


def test_final_missing_migration_receipt_refused(final_migration_env):
    env = final_migration_env
    (
        env.root / "operations/cpu_node_migration" / env.authorization["job_key"] / "MIGRATED.json"
    ).unlink()
    with pytest.raises((FileNotFoundError, PermissionError)):
        env.make().verify_final()


def test_final_mutated_live_journal_refused(final_migration_env):
    env = final_migration_env
    gate = (
        env.root
        / "operations/cpu_node_migration"
        / env.authorization["job_key"]
        / "migrate/RELEASE_GATE.json"
    )
    gate.write_bytes(gate.read_bytes() + b"\n")
    with pytest.raises(PermissionError, match="journal missing, changed, or extended"):
        env.make().verify_final()


def test_final_original_science_gate_failure_cannot_be_hidden_by_migration(final_migration_env):
    env = final_migration_env
    data = json.loads(env.builder.read_text())
    data["checks"]["raw_rescoring_chain"] = "FAIL"
    save(env.builder, data)
    save(env.dispatcher, dict(receipt=str(env.builder), sha256=digest(env.builder)))
    with pytest.raises(PermissionError, match="scientific delivery verification incomplete"):
        env.make().verify_final()


def test_missing_live_dispatcher_pause_refuses_before_scheduler_commands(migration_env):
    env = migration_env
    with pytest.raises(PermissionError, match="pause"):
        env.module.Migration(env.root, env.auth, runner=env.slurm).migrate()
    assert env.slurm.calls == []


def test_pause_receipt_cannot_substitute_for_live_process_verification(migration_env, monkeypatch):
    env = migration_env

    def resumed_process(process):
        raise PermissionError("Dispatcher has resumed")

    monkeypatch.setattr(env.module, "verify_paused_process", resumed_process)
    with pytest.raises(PermissionError, match="resumed"):
        env.make().migrate()
    assert env.slurm.calls == []


@pytest.mark.parametrize(
    "field,value", [("claim_id", "wrong"), ("device", "0"), ("command", ["unregistered", "worker"])]
)
def test_final_release_launch_identity_must_match_frozen_intent(final_migration_env, field, value):
    env = final_migration_env
    path = env.jobdir / "release.LAUNCH.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    save(path, receipt)
    with pytest.raises(PermissionError, match="launch binding"):
        env.make().verify_final()


def test_unrecognized_site_request_field_change_is_not_whitelisted(migration_env):
    env = migration_env
    env.slurm.fields["SitePolicy"] = "original"

    def change(action, slurm):
        if action == "update":
            slurm.fields["SitePolicy"] = "changed"

    env.slurm.on_mutation = change
    with pytest.raises(PermissionError, match="other than the authorized node"):
        env.make().migrate()
    assert [c[1] for c in env.slurm.mutations] == ["hold", "update"]


@pytest.mark.parametrize("status", ["FAILED", "RUNNING", "PASS", None])
def test_final_requires_original_dispatcher_done(final_migration_env, status):
    env = final_migration_env
    save(env.dispatcher.parent / "END.json", dict(status=status))
    with pytest.raises(PermissionError, match="dispatcher has not completed"):
        env.make().verify_final()


@pytest.mark.parametrize("change", ["failed", "forged_result"])
def test_final_rejects_unsuccessful_or_unbound_guard_action(final_migration_env, change):
    env = final_migration_env
    path = env.pause.parent / "ACTION_RESULT.json"
    action = json.loads(path.read_text())
    if change == "failed":
        action["status"] = "FAILED"
    else:
        action["result"]["job_id"] = "215521"
    save(path, action)
    env.package()
    with pytest.raises(PermissionError, match="Guard action"):
        env.make().verify_final()


@pytest.mark.parametrize("change", ["wrong_pid", "worker_not_reaped", "still_paused"])
def test_final_requires_same_dispatcher_safely_resumed(final_migration_env, change):
    env = final_migration_env
    path = env.pause.parent / "RESUMED.json"
    resume = json.loads(path.read_text())
    if change == "wrong_pid":
        resume["process"]["pid"] += 1
    elif change == "worker_not_reaped":
        resume["worker_reaped_before_resume"] = False
    else:
        resume["process"]["state"] = "T"
    save(path, resume)
    env.package()
    with pytest.raises(PermissionError, match="Dispatcher resume"):
        env.make().verify_final()
