"""CPU-only fixtures for the separate operational dispatcher, never live Slurm."""

from __future__ import annotations

import copy
import importlib.util
import json
import py_compile
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f1.chartqa import PROMPT
from sr_f3_ce import external, orchestration


@pytest.fixture(scope="module")
def helper():
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/dispatch_registered.py"
    spec = importlib.util.spec_from_file_location("srf3_dispatch_test_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registered(helper, tmp_path, monkeypatch):
    checkout = Path(__file__).resolve().parents[2]
    root = tmp_path / "run"
    operation = root / "operations" / ("a" * 40)
    operation.mkdir(parents=True)
    shutil.copyfile(helper.__file__, operation / "dispatch_registered.py")
    for name in helper.HELPERS[1:]:
        (operation / name).write_text("# synthetic operational source fixture\n")
    original_preparer = root / "operations" / helper.TYPED_COMMIT / "finalize_external_types.py"
    original_preparer.parent.mkdir()
    shutil.copyfile(checkout / "scripts/sr_f3_ce/finalize_external_types.py", original_preparer)
    monkeypatch.setattr(helper, "__file__", str(operation / "dispatch_registered.py"))
    python = str(Path(sys.executable).resolve())
    helper.save(
        root / "RUN_IDENTITY.json",
        dict(experiment_id=helper.EXPERIMENT, code_commit=helper.FROZEN_COMMIT),
    )
    helper.save(root / "deployment/SOURCE_MANIFEST.json", {"fixture": True})
    helper.save(
        root / "orchestration/CONFIG.json",
        dict(
            pool_size=1,
            python=python,
            user="alice",
            cpu_policy=dict(partition="cluster02", qos=helper.QOS, cpu_node="cpu-2"),
        ),
    )
    extension = root / "native/observability" / ("b" * 40)
    extension.mkdir(parents=True)
    for name in ("instrumented_execution.py", "observe_runtime.py"):
        shutil.copyfile(checkout / "scripts/sr_f3_ce" / name, extension / name)
    sources = [
        dict(
            path=name, sha256=helper.sha(extension / name), bytes=(extension / name).stat().st_size
        )
        for name in ("instrumented_execution.py", "observe_runtime.py")
    ]
    helper.save(
        extension / "CPU_VALIDATION.json",
        dict(
            status="PASS",
            scope="CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION",
            tests_passed=1,
            gpu_validation_performed=False,
            new_model_generations=0,
            new_optimizer_steps=0,
            source_sha256={row["path"]: row["sha256"] for row in sources},
        ),
    )
    helper.save(
        extension / "SOURCE_MANIFEST.json",
        dict(
            schema_version=1,
            experiment_id=helper.EXPERIMENT,
            status="FROZEN_OBSERVABILITY_EXTENSION",
            extension_commit="b" * 40,
            base_code_commit=helper.FROZEN_COMMIT,
            run_identity_sha256=helper.sha(root / "RUN_IDENTITY.json"),
            base_source_manifest_sha256=helper.sha(root / "deployment/SOURCE_MANIFEST.json"),
            sources=sources,
            executed_sources=["instrumented_execution.py", "observe_runtime.py"],
            cpu_validation=dict(
                path="CPU_VALIDATION.json",
                sha256=helper.sha(extension / "CPU_VALIDATION.json"),
                bytes=(extension / "CPU_VALIDATION.json").stat().st_size,
            ),
        ),
    )
    path = helper.register(root, python, "alice", "a" * 40, extension / "SOURCE_MANIFEST.json")
    value = helper.identity(path)
    value["identity_path"] = str(path)
    return value


class FakeSlurm:
    def __init__(self):
        self.rows, self.scripts, self.accounting, self.calls = {}, {}, {}, []
        self.next_job = 100
        self.ambiguous_submit = self.alter_script = False

    def run(self, command):
        self.calls.append(command)
        if command[0] == "sbatch":
            args = dict(part[2:].split("=", 1) for part in command[1:-1] if "=" in part)
            job = str(self.next_job)
            self.next_job += 1
            gpu_count = int(args["gres"].rsplit(":", 1)[1]) if "gres" in args else 0
            tres = f"cpu={args['cpus-per-task']},mem={args['mem']},node=1"
            if gpu_count:
                tres += f",gres/gpu={gpu_count},gres/gpu:pro6000={gpu_count}"
            self.rows[job] = dict(
                JobId=job,
                JobName=args["job-name"],
                Account=args["account"],
                Partition=args["partition"],
                QOS=args["qos"],
                Command=command[-1],
                Comment=args["comment"],
                NumCPUs=args["cpus-per-task"],
                UserId="alice(1)",
                NumNodes="1",
                ReqTRES=tres,
                ReqNodeList=args.get("nodelist", "(null)"),
                Features="highmem" if gpu_count else "(null)",
                JobState="PENDING",
                Reason="JobHeldUser",
                TimeLimit="3-00:00:00",
                Requeue="0",
                RunTime="00:00:00",
                NodeList="gpu-node" if gpu_count else "cpu-2",
                AllocTRES="(null)",
            )
            self.scripts[job] = Path(command[-1]).read_text()
            return SimpleNamespace(
                returncode=int(self.ambiguous_submit),
                stdout="" if self.ambiguous_submit else job + "\n",
                stderr="",
            )
        if command[:2] == ["scontrol", "release"]:
            self.rows[command[2]].update(
                JobState="RUNNING", Reason="None", AllocTRES=self.rows[command[2]]["ReqTRES"]
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    def show(self, job):
        return dict(self.rows[job])

    def queue(self, user):
        return copy.deepcopy(self.rows)

    def batch_script(self, job):
        return self.scripts[job] + ("# unapproved change\n" if self.alter_script else "")

    def terminal(self, job):
        return self.accounting.get(job)

    def finish(self, job, state="COMPLETED"):
        fields = self.rows.pop(job)
        self.accounting[job] = dict(
            job_id=job,
            state=state,
            exit_code="0:0",
            user="alice",
            account="rose",
            qos=fields["QOS"],
            name=fields["JobName"],
        )


class FakeController:
    def __init__(self, completed=()):
        self.result = dict(
            completed=list(completed), ready=[], blocked=False, all_complete=False, attempts=[]
        )
        self.status_calls = self.tick_calls = 0
        self.resumes = []

    def status(self):
        self.status_calls += 1
        return copy.deepcopy(self.result)

    def tick(self):
        self.tick_calls += 1
        return self.status()

    def blocking_attempts(self, states):
        return [state for state in states if state["status"] in orchestration.BLOCKING]

    def verify_job(self, state):
        assert state["key"] == "fixture-p0" and state["status"] == "COMPLETE"
        return ["p0"]

    def resume(self, key):
        self.resumes.append(key)
        self.result.update(blocked=False, attempts=[])
        return dict(status="SUBMITTED", key=key + "--resume-01")


def api_fixture(helper, registered, *, completed=()):
    root = Path(registered["root"])
    api = SimpleNamespace(
        orchestration=orchestration,
        external=external,
        protocol=SimpleNamespace(verify_run=lambda *args: None),
        slurm=FakeSlurm(),
        controller=FakeController(completed),
        freezes=[],
        amendments=[],
        lifecycle=[],
    )
    if "p0" in completed:
        state = dict(key="fixture-p0", status="COMPLETE", gpu_count=0, qos=helper.QOS, job_id="1")
        directory = root / "orchestration/jobs" / state["key"]
        helper.save(directory / "INTENT.json", dict(kind="p0", task_ids=["p0"]))
        helper.save(directory / "STATE.json", state)
        terminal = dict(
            job_id="1",
            state="COMPLETED",
            exit_code="0:0",
            user="alice",
            account="rose",
            qos=helper.QOS,
            name="SRF3CE-P0-LH",
        )
        helper.save(directory / "TERMINAL.json", terminal)
        api.slurm.accounting["1"] = terminal
        api.controller.result["attempts"] = [state]

    def freeze(run):
        api.lifecycle.append("freeze")
        api.freezes.append(str(run))
        helper.save(root / "EXECUTION_FREEZE.json", {"fixture_science": True})

    api.bootstrap = SimpleNamespace(freeze_science=freeze)

    def seal(run, manifest, *, typed_verified):
        assert not (root / "EXECUTION_FREEZE.json").exists()
        api.lifecycle.append("amendment")
        api.amendments.append(dict(manifest=str(manifest), typed=typed_verified))

    def verify(run, manifest, *, science=False):
        api.lifecycle.append("extension_verify" if science else "static_verify")
        return dict(
            manifest_path=str(manifest),
            manifest_sha256=registered["extension_manifest_sha256"],
            extension_commit="b" * 40,
            base_code_commit=helper.FROZEN_COMMIT,
            observer_source_sha256=helper.sha(Path(manifest).with_name("observe_runtime.py")),
            amendment_sha256="d" * 64,
            cpu_validation_sha256=helper.sha(Path(manifest).with_name("CPU_VALIDATION.json")),
        )

    api.extension = SimpleNamespace(seal_amendment=seal, verify_extension=verify)
    return api


def typed_stub(helper, registered):
    return dict(
        status="VERIFIED_TYPED_V1",
        auxiliary_code_commit=helper.TYPED_COMMIT,
        preparation_script_sha256=helper.TYPED_SCRIPT_SHA,
    )


def make_driver(helper, registered, api, **kwargs):
    directory = helper.attempts_root(registered) / "0001"
    return helper.Driver(registered, api, directory, **kwargs)


@pytest.fixture
def engine_api(helper, registered, monkeypatch):
    """Use the unchanged Controller, its real receipts, and fake Slurm transport."""
    root = Path(registered["root"])
    code = root / "code"
    scripts = code / "scripts/sr_f3_ce"
    scripts.mkdir(parents=True)
    for name in ("submit.py", "run_worker.py"):
        (scripts / name).write_text("# synthetic worker, never executed\n")
    # The general operational fixture has a minimal CONFIG. Replace it only in
    # this scheduler integration fixture with the actual constructor's schema.
    (root / "orchestration/CONFIG.json").unlink()
    monkeypatch.setattr("sr_f3_ce.protocol.verify_run", lambda *args, **kwargs: ({}, {}))
    api = api_fixture(helper, registered)
    api.controller = orchestration.Controller(
        root,
        code,
        python=registered["python"],
        slurm=api.slurm,
        user="alice",
        pool_size=1,
        cpu_qos=helper.QOS,
    )
    path = Path(__file__).with_name("test_orchestration.py")
    spec = importlib.util.spec_from_file_location("engine_capacity_receipt_fixtures", path)
    receipts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receipts)
    api.finish = lambda state: receipts.complete_allocation(api.controller, state, monkeypatch)
    api.local_p0 = lambda: receipts.complete_local_p0(api.controller)
    return api


def finish_p0(api):
    state = api.controller.submit("p0")
    result = api.finish(state)
    assert "p0" in result["completed"] and not result["blocked"]
    return state


def finish_all_engines(api):
    for _ in range(4):
        state = api.controller.submit("engine")
        assert state["status"] == "SUBMITTED"
        api.finish(state)


def engine_intents(helper, registered):
    paths = (Path(registered["root"]) / "orchestration/jobs").glob("*/INTENT.json")
    return [row for path in paths if (row := helper.read(path))["kind"] == "engine"]


def cpu_queue_row(helper):
    return dict(
        QOS=helper.QOS,
        UserId="alice(1)",
        ReqTRES="cpu=4,mem=12G,node=1",
        AllocTRES="cpu=4,mem=12G,node=1",
        JobState="RUNNING",
    )


def test_real_p0_receipt_without_completed_allocation_cannot_dispatch_engine(
    helper,
    registered,
    engine_api,
):
    api = engine_api
    assert helper.read(Path(registered["root"]) / "RUN_IDENTITY.json")["code_commit"] == (
        helper.FROZEN_COMMIT
    )  # The legacy v1 waiting boundary remains valid.
    state = api.controller.submit("p0")
    helper.save(
        orchestration.receipt_path(Path(registered["root"]), "p0"),
        dict(status="PASS", job_id=state["job_id"]),
    )
    driver = make_driver(helper, registered, api)
    assert driver.poll() == "WAIT_ENGINE"
    assert driver.poll() == "UNCHANGED"
    assert not engine_intents(helper, registered)
    assert not api.amendments and not api.freezes
    assert [call[0] for call in api.slurm.calls] == ["sbatch", "scontrol"]


@pytest.mark.parametrize("all_engines_complete", [False, True])
def test_local_p0_cannot_dispatch_or_start_freeze(
    helper, registered, engine_api, all_engines_complete
):
    api = engine_api
    api.local_p0()
    if all_engines_complete:
        # The unchanged controller intentionally permits this legacy local path.
        finish_all_engines(api)
    status = api.controller.status()
    assert "p0" in status["completed"] and not status["blocked"]
    if all_engines_complete:
        assert set(status["completed"]) == helper.ENGINES
    before = len(api.slurm.calls)
    driver = make_driver(helper, registered, api)
    with pytest.raises(PermissionError, match="real completed P0 allocation"):
        driver.act()
    assert len(api.slurm.calls) == before
    assert not api.amendments and not api.freezes
    assert not (driver.directory.parent.parent / "FREEZE_INTENT.json").exists()


@pytest.mark.parametrize("all_engines_complete", [False, True])
@pytest.mark.parametrize(
    "change", [None, {"state": "FAILED"}, {"exit_code": "1:0"}, {"user": "mallory"}]
)
def test_p0_live_accounting_must_match_authenticated_completion_before_dispatch_or_freeze(
    helper, registered, engine_api, all_engines_complete, change
):
    api = engine_api
    p0 = finish_p0(api)
    if all_engines_complete:
        finish_all_engines(api)
    if change is None:
        del api.slurm.accounting[p0["job_id"]]
    else:
        api.slurm.accounting[p0["job_id"]] = dict(api.slurm.accounting[p0["job_id"]], **change)
    # Saved receipts remain valid; the new operational guard must query accounting.
    status = api.controller.status()
    assert "p0" in status["completed"] and not status["blocked"]
    before = len(api.slurm.calls)
    driver = make_driver(helper, registered, api)
    with pytest.raises(PermissionError, match="P0 live accounting"):
        driver.act()
    assert len(api.slurm.calls) == before
    assert not api.amendments and not api.freezes
    assert not (driver.directory.parent.parent / "FREEZE_INTENT.json").exists()


def test_real_engine_dispatch_waits_for_capacity_then_fills_only_fixed_unclaimed_tasks(
    helper,
    registered,
    engine_api,
):
    api = engine_api
    finish_p0(api)
    manual = api.controller.submit("engine")
    assert helper.read(
        Path(registered["root"]) / "orchestration/jobs" / manual["key"] / "INTENT.json"
    )["task_ids"] == ["engine--b0"]
    api.slurm.rows["cpu-dispatcher"] = cpu_queue_row(helper)
    for index in range(3):
        api.slurm.rows[f"unrelated-{index}"] = dict(
            QOS=helper.QOS,
            UserId="alice(1)",
            ReqTRES="cpu=4,gres/gpu=1",
            JobState="RUNNING" if index == 0 else "PENDING",
            Reason="Dependency",
        )
    original = copy.deepcopy(api.slurm.rows)
    calls_before = len(api.slurm.calls)

    def forbidden(*args):
        raise AssertionError("Typed/science cannot run while ENGINE is incomplete")

    driver = make_driver(helper, registered, api, typed=forbidden)
    assert driver.poll() == "WAIT_ENGINE"
    assert len(api.slurm.calls) == calls_before
    assert api.slurm.rows == original
    api.slurm.rows.pop("unrelated-0")
    assert driver.poll() == "DISPATCHING_ENGINE"
    assert len(engine_intents(helper, registered)) == 2
    assert orchestration.teacher_usage(
        api.slurm.queue("alice"), api.controller.attempts(), "alice"
    ) == dict(gpus=4, submitted=5)
    assert driver.poll() == "WAIT_ENGINE"
    assert driver.poll() == "UNCHANGED"
    assert len(engine_intents(helper, registered)) == 2
    api.slurm.rows.pop("unrelated-1")
    api.slurm.rows.pop("unrelated-2")
    assert driver.poll() == "DISPATCHING_ENGINE"
    intents = engine_intents(helper, registered)
    assert len(intents) == 4
    assert {task for intent in intents for task in intent["task_ids"]} == (helper.ENGINES - {"p0"})
    assert all(
        len(intent["task_ids"]) == intent["gpu_count"] == 1 and intent["job_name"].endswith("-LH")
        for intent in intents
    )
    for intent in intents:
        job_dir = Path(intent["script"]).parent
        assert (
            helper.read(job_dir / "ALLOCATION_VERIFIED.json")["script_sha256"]
            == (intent["script_sha256"])
        )
        assert helper.read(job_dir / "RELEASE.json")["returncode"] == 0
    assert not api.amendments and not api.freezes
    assert not (Path(registered["root"]) / "EXECUTION_FREEZE.json").exists()


def test_real_engine_dispatch_respects_other_users_physical_gpu_reservations(
    helper,
    registered,
    engine_api,
):
    api = engine_api
    finish_p0(api)
    api.slurm.rows["cpu-dispatcher"] = cpu_queue_row(helper)
    api.slurm.rows["other-user"] = dict(
        QOS=helper.QOS,
        UserId="bob(2)",
        ReqTRES="cpu=4,gres/gpu=5",
        AllocTRES="cpu=4,gres/gpu=5",
    )
    assert make_driver(helper, registered, api).poll() == "WAIT_ENGINE"
    assert not engine_intents(helper, registered)
    assert not api.freezes


def test_real_engine_sbatch_unknown_is_preserved_and_never_retried(
    helper,
    registered,
    engine_api,
):
    api = engine_api
    finish_p0(api)
    api.slurm.ambiguous_submit = True
    driver = make_driver(helper, registered, api)
    with pytest.raises(RuntimeError, match="ENGINE submission outcome"):
        driver.act()
    attempts = api.controller.attempts()
    assert [row["status"] for row in attempts if row["gpu_count"]] == ["SUBMISSION_UNKNOWN"]
    calls = len(api.slurm.calls)
    with pytest.raises(RuntimeError, match="non-recoverable"):
        driver.act()
    assert len(api.slurm.calls) == calls
    assert not api.amendments and not api.freezes


def test_real_engine_changed_slurm_script_stays_held_and_cannot_continue(
    helper,
    registered,
    engine_api,
):
    api = engine_api
    finish_p0(api)
    api.slurm.alter_script = True
    driver = make_driver(helper, registered, api)
    with pytest.raises(PermissionError, match="stored batch script"):
        driver.act()
    assert [row["status"] for row in api.controller.attempts() if row["gpu_count"]] == ["HELD"]
    calls = len(api.slurm.calls)
    with pytest.raises(RuntimeError, match="non-recoverable"):
        driver.act()
    assert len(api.slurm.calls) == calls
    assert not api.freezes


@pytest.mark.parametrize(
    "returned_status", ["SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN", "HELD", "FAILED"]
)
def test_engine_submission_must_return_confirmed_submitted_before_next_task(
    helper,
    registered,
    engine_api,
    returned_status,
):
    api = engine_api
    finish_p0(api)
    submitted = []

    def submit(kind):
        submitted.append(kind)
        return dict(status=returned_status)

    api.controller.submit = submit
    with pytest.raises(RuntimeError, match="ENGINE submission outcome"):
        make_driver(helper, registered, api).act()
    assert submitted == ["engine"]
    assert not engine_intents(helper, registered) and not api.amendments and not api.freezes


def test_engine_failure_observed_between_submissions_stops_without_retry(
    helper,
    registered,
    engine_api,
):
    api = engine_api
    finish_p0(api)
    original_status = api.controller.status

    def fail_new_engine():
        for state in api.controller.attempts():
            if state["gpu_count"] and state.get("job_id") in api.slurm.rows:
                api.slurm.finish(state["job_id"], "FAILED")
                break
        return original_status()

    api.controller.status = fail_new_engine
    driver = make_driver(helper, registered, api)
    with pytest.raises(RuntimeError, match="ENGINE submission left"):
        driver.act()
    assert len(engine_intents(helper, registered)) == 1
    calls = len(api.slurm.calls)
    with pytest.raises(RuntimeError, match="non-recoverable"):
        driver.act()
    assert len(api.slurm.calls) == calls
    assert not api.amendments and not api.freezes


def test_only_all_real_engine_completions_unlock_amendment_and_freeze(
    helper,
    registered,
    engine_api,
):
    api = engine_api
    finish_p0(api)
    api.slurm.rows["cpu-dispatcher"] = cpu_queue_row(helper)
    driver = make_driver(
        helper, registered, api, typed=lambda *args: typed_stub(helper, registered)
    )
    assert driver.act() == "DISPATCHING_ENGINE"
    states = [state for state in api.controller.attempts() if state["gpu_count"]]
    assert len(states) == 4
    for state in states[:-1]:
        api.finish(state)
        assert driver.act() == "WAIT_ENGINE"
        assert not api.amendments and not api.freezes
    api.finish(states[-1])
    status = api.controller.status()
    assert set(status["completed"]) == helper.ENGINES and not status["blocked"]
    # Test only the engineering boundary, not a scientific training allocation.
    ticks = []

    def scientific_tick():
        assert (Path(registered["root"]) / "EXECUTION_FREEZE.json").is_file()
        ticks.append(True)
        return dict(blocked=False, all_complete=False)

    api.controller.tick = scientific_tick
    assert driver.act() == "DISPATCHING_FIXED_TASKS"
    assert len(api.amendments) == len(api.freezes) == len(ticks) == 1
    assert api.lifecycle[:3] == ["amendment", "freeze", "extension_verify"]


def test_identity_binds_scientific_config_and_all_operational_sources(helper, registered):
    assert helper.identity(registered["identity_path"])["pool_size"] == 1
    manifest = helper.read(Path(registered["identity_path"]).with_name("OPERATION_SOURCES.json"))
    assert set(manifest["files"]) == set(helper.HELPERS)
    Path(registered["identity_path"]).with_name("build_delivery.py").write_text("changed")
    with pytest.raises(PermissionError, match="source changed"):
        helper.identity(registered["identity_path"])


def test_operation_identity_rejects_changed_run_or_config(helper, registered):
    helper.save(Path(registered["root"]) / "RUN_IDENTITY.json", {"changed": True})
    with pytest.raises(PermissionError, match="identity changed"):
        helper.identity(registered["identity_path"])


def test_driver_waits_for_actual_engine_completion_before_typed_or_science(helper, registered):
    api = api_fixture(helper, registered, completed=["p0", "engine--b0"])

    def forbidden(*args):
        raise AssertionError("Typed audit/science was accessed before ENGINE completion")

    driver = make_driver(helper, registered, api, typed=forbidden)
    assert driver.poll() == "WAIT_ENGINE"
    assert not api.freezes and api.controller.tick_calls == 0


def test_missing_typed_publication_waits_without_freeze_or_tick(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    driver = make_driver(helper, registered, api)
    assert driver.poll() == "WAIT_TYPED_REVIEW"
    assert driver.poll() == "UNCHANGED"
    assert api.controller.status_calls == 1 and api.controller.tick_calls == 0
    assert not api.freezes


def test_stable_snapshot_does_not_rehash_and_new_publication_freezes_once(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    publication = Path(registered["root"]) / helper.REVISION / "PUBLICATION.json"
    verified = typed_stub(helper, registered)
    driver = make_driver(
        helper, registered, api, typed=lambda *args: verified if publication.exists() else None
    )
    assert driver.poll() == "WAIT_TYPED_REVIEW"
    assert driver.poll() == "UNCHANGED"
    helper.save(publication, {"fixture_ready": True})
    assert driver.poll() == "DISPATCHING_FIXED_TASKS"
    calls = api.controller.status_calls
    for _ in range(3):
        assert driver.poll() == "UNCHANGED"
    assert api.controller.status_calls == calls
    assert len(api.freezes) == api.controller.tick_calls == 1
    assert len(api.amendments) == 1
    assert api.lifecycle[:3] == ["amendment", "freeze", "extension_verify"]
    api.slurm.rows["other"] = dict(QOS=helper.QOS, UserId="bob(2)", ReqTRES="cpu=1,gres/gpu=1")
    assert driver.poll() == "DISPATCHING_FIXED_TASKS"
    assert len(api.freezes) == 1 and api.controller.tick_calls == 2


def test_clock_and_progress_fields_do_not_trigger_full_scans(helper, registered):
    api = api_fixture(helper, registered)
    api.slurm.rows["99"] = dict(
        QOS=helper.QOS, UserId="alice(1)", ReqTRES="cpu=4", RunTime="00:00:01", StartTime="today"
    )
    driver = make_driver(helper, registered, api)
    assert driver.poll() == "WAIT_ENGINE"
    api.slurm.rows["99"].update(RunTime="02:00:01", StartTime="changed")
    assert driver.poll() == "UNCHANGED" and api.controller.status_calls == 1


def test_terminal_transition_during_full_validation_is_not_lost(helper, registered):
    api = api_fixture(helper, registered)
    path = Path(registered["root"]) / "orchestration/jobs/engine--b0"
    helper.save(path / "INTENT.json", dict(kind="engine", gpu_count=1, qos=helper.QOS))
    helper.save(
        path / "STATE.json",
        dict(key="engine--b0", status="SUBMITTED", gpu_count=1, qos=helper.QOS, job_id="77"),
    )
    original = api.controller.status

    def terminal_arrives():
        result = original()
        api.slurm.accounting["77"] = dict(state="COMPLETED", job_id="77")
        return result

    api.controller.status = terminal_arrives
    driver = make_driver(helper, registered, api)
    assert driver.poll() == "WAIT_ENGINE"
    assert driver.poll() == "WAIT_ENGINE"
    assert driver.poll() == "UNCHANGED"
    assert api.controller.status_calls == 2


def test_unrelated_qos_does_not_consume_teacher_capacity(helper, registered):
    api = api_fixture(helper, registered)
    api.slurm.rows["other-qos"] = dict(QOS="rose", UserId="alice(1)", ReqTRES="gres/gpu=99")
    assert make_driver(helper, registered, api).snapshot()["teacher_usage"] == dict(
        gpus=0, submitted=0
    )


def test_fixed_controller_progression_never_reads_score_files(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    helper.save(Path(registered["root"]) / "EXECUTION_FREEZE.json", {"fixture": True})
    scores = Path(registered["root"]) / "sealed/scores.json"
    scores.parent.mkdir()
    scores.write_text("deliberately invalid score JSON that the dispatcher must never open")
    driver = make_driver(
        helper, registered, api, typed=lambda *args: typed_stub(helper, registered)
    )
    assert driver.act() == "DISPATCHING_FIXED_TASKS"
    assert api.controller.tick_calls == 1


def test_failed_freeze_intent_is_not_retried_after_restart(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)

    def fail(*args):
        api.freezes.append("attempt")
        raise OSError("fixture disk failure")

    api.bootstrap.freeze_science = fail
    verified = typed_stub(helper, registered)
    with pytest.raises(OSError):
        make_driver(helper, registered, api, typed=lambda *args: verified).act()
    with pytest.raises(RuntimeError, match="previous scientific freeze"):
        make_driver(helper, registered, api, typed=lambda *args: verified).act()
    assert api.freezes == ["attempt"]


def add_failed_attempt(
    helper, registered, api, *, status="FAILED", terminal="TIMEOUT", kind="train"
):
    task = "train--SRF3CE_RL_b0" if kind == "train" else "engine--resume"
    key = "fixture-science"
    path = Path(registered["root"]) / "orchestration/jobs" / key
    state = dict(key=key, status=status, gpu_count=1, qos=helper.QOS, job_id="88")
    helper.save(path / "INTENT.json", dict(kind=kind, gpu_count=1, qos=helper.QOS, task_ids=[task]))
    helper.save(path / "STATE.json", state)
    if terminal:
        helper.save(path / "TERMINAL.json", dict(state=terminal))
    api.controller.result.update(blocked=True, attempts=[state])
    return key, task


@pytest.mark.parametrize(
    "status,terminal,kind",
    [
        ("SUBMISSION_UNKNOWN", None, "train"),
        ("ACCOUNTING_UNKNOWN", None, "train"),
        ("HELD", None, "train"),
        ("FAILED", "FAILED", "train"),
        ("FAILED", "PREEMPTED", "train"),
        ("FAILED", "TIMEOUT", "engine"),
    ],
)
def test_unknown_non_timeout_and_engine_failures_never_resume(
    helper, registered, status, terminal, kind
):
    api = api_fixture(helper, registered)
    add_failed_attempt(helper, registered, api, status=status, terminal=terminal, kind=kind)
    with pytest.raises(RuntimeError, match="non-recoverable"):
        make_driver(helper, registered, api).act()
    assert not api.controller.resumes and not api.freezes


def test_scientific_single_gpu_timeout_calls_frozen_exact_resume(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    key, _ = add_failed_attempt(helper, registered, api)
    helper.save(Path(registered["root"]) / "EXECUTION_FREEZE.json", {"fixture": True})
    driver = make_driver(
        helper, registered, api, typed=lambda *args: typed_stub(helper, registered)
    )
    assert driver.act() == "DISPATCHING_FIXED_TASKS"
    assert api.controller.resumes == [key]
    assert (
        helper.read(helper.attempts_root(registered) / "0001/recoveries" / (key + ".json"))[
            "previous"
        ]
        == key
    )


def test_complete_receipt_on_failed_job_blocks_resampling(helper, registered):
    api = api_fixture(helper, registered)
    _, task = add_failed_attempt(helper, registered, api)
    helper.save(orchestration.receipt_path(registered["root"], task), {"status": "COMPLETE"})
    with pytest.raises(RuntimeError, match="receipt-only"):
        make_driver(helper, registered, api).act()
    assert api.controller.resumes == []


def test_failed_cpu_recovery_validation_is_not_retried(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    add_failed_attempt(helper, registered, api)
    helper.save(Path(registered["root"]) / "EXECUTION_FREEZE.json", {"fixture": True})

    def fail(key):
        api.controller.resumes.append(key)
        raise PermissionError("checkpoint cached chain changed")

    api.controller.resume = fail
    with pytest.raises(PermissionError, match="checkpoint"):
        make_driver(
            helper, registered, api, typed=lambda *args: typed_stub(helper, registered)
        ).poll()
    assert len(api.controller.resumes) == 1 and api.controller.tick_calls == 0


def test_timeout_cannot_resume_before_completed_independent_typed_gate(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    add_failed_attempt(helper, registered, api)
    helper.save(Path(registered["root"]) / "EXECUTION_FREEZE.json", {"fixture": True})
    with pytest.raises(PermissionError, match="independent type review"):
        make_driver(helper, registered, api).act()
    assert not api.controller.resumes


def test_terminal_timeout_arriving_inside_tick_uses_exact_recovery(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    helper.save(Path(registered["root"]) / "EXECUTION_FREEZE.json", {"fixture": True})

    def late_timeout():
        api.controller.tick_calls += 1
        add_failed_attempt(helper, registered, api)
        return api.controller.status()

    api.controller.tick = late_timeout
    driver = make_driver(
        helper, registered, api, typed=lambda *args: typed_stub(helper, registered)
    )
    assert driver.act() == "DISPATCHING_FIXED_TASKS"
    assert api.controller.resumes == ["fixture-science"]


def test_all_blockers_are_classified_before_any_science_resume(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    add_failed_attempt(helper, registered, api)
    path = Path(registered["root"]) / "orchestration/jobs/other-failure"
    helper.save(
        path / "INTENT.json",
        dict(kind="train", gpu_count=1, qos=helper.QOS, task_ids=["train--SRF3CE_RL_b1"]),
    )
    helper.save(path / "TERMINAL.json", dict(state="PREEMPTED"))
    api.controller.result["attempts"].append(dict(key="other-failure", status="FAILED"))
    with pytest.raises(RuntimeError, match="non-recoverable"):
        make_driver(helper, registered, api).act()
    assert not api.controller.resumes


def test_cpu_submission_and_frozen_scientific_submission_share_lock(helper, registered):
    api = api_fixture(helper, registered)
    with (
        orchestration.lease(Path(registered["root"]) / "orchestration"),
        pytest.raises(RuntimeError, match="Another SR-F3-CE submitter"),
    ):
        helper.submit_driver(registered, api, 1)
    assert not api.slurm.calls


def test_submit_cpu_driver_is_held_verified_and_lh(helper, registered, monkeypatch):
    api = api_fixture(helper, registered)
    state = helper.submit_driver(registered, api, 1)
    assert state["status"] == "SUBMITTED"
    args = api.slurm.calls[0]
    assert args[1:3] == ["--hold", "--parsable"]
    assert "--job-name=SRF3CE-DISPATCH-N0001-LH" in args
    assert "--qos=" + helper.QOS in args and "--no-requeue" in args
    assert not any(arg.startswith("--gres") for arg in args)
    assert api.slurm.calls[1] == ["scontrol", "release", state["job_id"]]
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    directory = helper.attempts_root(registered) / "0001"
    assert helper.runtime_allocation(registered, api, directory)["NodeList"] == "cpu-2"
    api.slurm.rows[state["job_id"]]["AllocTRES"] += ",gres/gpu=1"
    with pytest.raises(PermissionError, match="CPU-only"):
        helper.runtime_allocation(registered, api, directory)


def test_altered_actual_script_stays_held(helper, registered):
    api = api_fixture(helper, registered)
    api.slurm.alter_script = True
    with pytest.raises(PermissionError, match="stored dispatcher script"):
        helper.submit_driver(registered, api, 1)
    assert len(api.slurm.calls) == 1
    assert helper.read(helper.attempts_root(registered) / "0001/STATE.json")["status"] == "HELD"


def test_unknown_cpu_submission_preserved_and_cannot_fork_or_retry(helper, registered):
    api = api_fixture(helper, registered)
    api.slurm.ambiguous_submit = True
    with pytest.raises(RuntimeError, match="outcome unknown"):
        helper.submit_driver(registered, api, 1)
    with pytest.raises(RuntimeError, match="already attempted"):
        helper.submit_driver(registered, api, 1)
    with pytest.raises(RuntimeError, match="active or its submission is unknown"):
        helper.submit_driver(registered, api, 2, 1)
    assert len(api.slurm.calls) == 1


def test_cpu_driver_counts_toward_five_teacher_submissions(helper, registered):
    api = api_fixture(helper, registered)
    for i in range(5):
        api.slurm.rows[str(i)] = dict(QOS=helper.QOS, UserId="alice(1)", ReqTRES="cpu=1")
    with pytest.raises(RuntimeError, match="No submission slot in the dispatcher QoS"):
        helper.submit_driver(registered, api, 1)
    assert not api.slurm.calls
    api.slurm.rows.pop("4")
    state = helper.submit_driver(registered, api, 1)
    assert state["status"] == "SUBMITTED"
    assert orchestration.teacher_usage(api.slurm.rows, [], "alice") == dict(gpus=0, submitted=5)


def test_other_users_and_unknown_gpu_reservations_are_counted(helper, registered):
    api = api_fixture(helper, registered)
    api.slurm.rows["other"] = dict(QOS=helper.QOS, UserId="bob(2)", ReqTRES="gres/gpu=5")
    add_failed_attempt(helper, registered, api, status="SUBMISSION_UNKNOWN", terminal=None)
    with pytest.raises(PermissionError, match="capacity exceeds"):
        make_driver(helper, registered, api).snapshot()


def test_explicit_cpu_lease_handoff_keeps_predecessor_and_new_identity(helper, registered):
    api = api_fixture(helper, registered)
    first = helper.submit_driver(registered, api, 1)
    with pytest.raises(RuntimeError, match="active"):
        helper.submit_driver(registered, api, 2, 1)
    api.slurm.finish(first["job_id"])
    helper.save(helper.attempts_root(registered) / "0001/END.json", {"status": "HANDOFF_REQUIRED"})
    second = helper.submit_driver(registered, api, 2, 1)
    assert second["job_id"] != first["job_id"]
    intent = helper.read(helper.attempts_root(registered) / "0002/INTENT.json")
    assert intent["previous"] == 1 and intent["job_name"] == "SRF3CE-DISPATCH-N0002-LH"
    assert (helper.attempts_root(registered) / "0001/TERMINAL.json").exists()


def test_driver_lock_excludes_a_second_process(helper, tmp_path):
    with (
        helper.lock(tmp_path / "driver.lock"),
        pytest.raises(RuntimeError, match="Another dispatcher"),
        helper.lock(tmp_path / "driver.lock"),
    ):
        raise AssertionError("Lock must exclude a second owner")


def make_typed_fixture(helper, registered):
    root = Path(registered["root"])
    revision = root / helper.REVISION
    old, new = root / "external/chartqa", root / "external/chartqa_typed_v1"
    old.mkdir(parents=True)
    (old / "image.bin").write_bytes(b"synthetic visible chart fixture")
    image_hash = helper.sha(old / "image.bin")
    items = [
        dict(qid=f"q{i:04d}", question=f"Fixture question {i}", image_sha256=image_hash)
        for i in range(2500)
    ]
    inputs = [
        dict(
            qid=item["qid"],
            prompt=PROMPT + "\n" + item["question"],
            image_file="image.bin",
            image_sha256=image_hash,
        )
        for item in items
    ]
    (old / "MODEL_INPUTS.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inputs))
    (old / "TASKS_AUDIT_ONLY.jsonl").write_bytes(
        b"opaque original audit sidecar: never parsed here\n"
    )
    original_registry = external.build_type_registry(items, [], outputs_unseen=True)
    helper.save(old / "TYPE_REGISTRY.json", original_registry)
    freeze = dict(
        status="FROZEN",
        image_count=2500,
        image_root=str(old),
        protocol="chartqa_plain",
        inputs_sha256=helper.sha(old / "MODEL_INPUTS.jsonl"),
        audit_tasks_sha256=helper.sha(old / "TASKS_AUDIT_ONLY.jsonl"),
        type_registry_sha256=helper.digest(original_registry),
        typed_unresolved_count=2500,
    )
    helper.save(old / "FREEZE.json", freeze)
    helper.save(
        root / "external/chartqapro/FREEZE.json",
        dict(status="UNAVAILABLE", image_count=0, reason="Synthetic fixture only"),
    )
    panels = dict(status="FROZEN", experiment_id=helper.EXPERIMENT)
    for pool, path in (("CHARTQA", old), ("CHARTQAPRO", root / "external/chartqapro")):
        panels[pool] = dict(
            root=str(path.relative_to(root)), freeze_sha256=helper.sha(path / "FREEZE.json")
        )
        for name in ("FREEZE.json", "TYPE_REGISTRY.json"):
            if (path / name).exists():
                target = revision / "v0" / pool / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path / name, target)
    helper.save(revision / "PANELS.v0.json", panels)
    review_sets = []
    audit = dict(
        status="TWO_INDEPENDENT_REVIEW_PASSES_COMPLETED",
        experiment_id=helper.EXPERIMENT,
        run_identity_sha256=helper.sha(root / "RUN_IDENTITY.json"),
        code_commit=helper.FROZEN_COMMIT,
        scientific_source_changed=False,
        model_calls=0,
        generation_budget_changed=False,
        before_execution_freeze=True,
        before_any_new_test=True,
        original_panels_sha256=helper.sha(revision / "PANELS.v0.json"),
        reviewer_ids=["A", "B"],
        review_files={},
        reviewer_evidence_manifests={},
        auxiliary_code_commit=helper.TYPED_COMMIT,
        auxiliary_commit_identity_supplied_by_operator=True,
        preparation_script_sha256=helper.TYPED_SCRIPT_SHA,
    )
    for label in ("A", "B"):
        rows = [
            dict(
                qid=item["qid"],
                reviewer_id=label,
                input_identity=external.blind_item_identity(item),
                method="independent_contract_rule_check",
                kind="UNRESOLVED",
                comparison=None,
                evidence={"reason": "synthetic fixture retains ambiguity"},
                used_gold=False,
                used_model_output=False,
            )
            for item in items
        ]
        wrapper = dict(
            experiment_id=helper.EXPERIMENT,
            reviewer_id=label,
            blind_inputs_only=True,
            used_gold=False,
            used_model_output=False,
            saw_other_review=False,
            reviews=rows,
        )
        helper.save(revision / f"REVIEW_{label}.json", wrapper)
        audit["review_files"][f"REVIEW_{label}.json"] = helper.sha(
            revision / f"REVIEW_{label}.json"
        )
        evidence = revision / "reviewer_evidence" / label / "rules.txt"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("synthetic independent evidence fixture\n")
        manifest = dict(
            reviewer_id=label,
            files=[
                dict(path="rules.txt", sha256=helper.sha(evidence), bytes=evidence.stat().st_size)
            ],
            total_bytes=evidence.stat().st_size,
        )
        manifest_path = revision / "reviewer_evidence" / f"{label}_MANIFEST.json"
        helper.save(manifest_path, manifest)
        audit["reviewer_evidence_manifests"][label] = helper.sha(manifest_path)
        review_sets.append(rows)
    merged = [row for pair in zip(*review_sets, strict=True) for row in pair]
    registry = external.build_type_registry(items, merged, outputs_unseen=True)
    shutil.copytree(old, new)
    helper.save(new / "TYPE_REGISTRY.json", registry)
    new_freeze = dict(freeze, type_registry_sha256=helper.digest(registry))
    helper.save(new / "FREEZE.json", new_freeze)
    shutil.copytree(root / "external/chartqapro", root / "external/chartqapro_typed_v1")
    scope = {item["qid"]: external.blind_item_identity(item) for item in items}
    audit["annotation_scope_sha256"] = helper.digest(scope)
    helper.save(
        revision / "BLIND_SCOPE.json",
        dict(
            experiment_id=helper.EXPERIMENT,
            inputs=scope,
            annotation_scope_sha256=helper.digest(scope),
            gold_accessed_for_type_assignment=False,
            model_outputs_accessed=False,
        ),
    )
    helper.save(
        revision / "RECONCILIATION.json",
        dict(
            CHARTQA=[
                dict(
                    qid=item["qid"],
                    status="UNRESOLVED",
                    kind="UNRESOLVED",
                    comparison=None,
                    reason="both_reviewers_unresolved",
                )
                for item in items
            ],
            CHARTQAPRO=[],
        ),
    )
    audit["identities"] = dict(
        CHARTQA=dict(
            questions=2500,
            inputs_sha256=freeze["inputs_sha256"],
            audit_tasks_sha256=freeze["audit_tasks_sha256"],
            type_registry_sha256=new_freeze["type_registry_sha256"],
            image_and_prompt_identity_unchanged=True,
            selection_unchanged=True,
        ),
        CHARTQAPRO=dict(status="UNAVAILABLE", reason="Synthetic fixture only"),
    )
    helper.save(revision / "TYPE_AUDIT_RECEIPT.json", audit)
    for pool in ("CHARTQA", "CHARTQAPRO"):
        directory = root / f"external/{pool.lower()}_typed_v1"
        panels[pool].update(
            root=str(directory.relative_to(root)),
            freeze_sha256=helper.sha(directory / "FREEZE.json"),
        )
    panels.update(
        typed_audit_revision="typed_v1",
        typed_audit_receipt=helper.REVISION + "/TYPE_AUDIT_RECEIPT.json",
        typed_audit_receipt_sha256=helper.sha(revision / "TYPE_AUDIT_RECEIPT.json"),
    )
    helper.save(root / "external/PANELS.json", panels)
    helper.save(revision / "PANELS.v1.json", panels)
    helper.save(
        revision / "PUBLICATION.json",
        dict(
            status="PUBLISHED_TYPED_V1_BEFORE_SCIENCE",
            previous_panels_sha256=helper.sha(revision / "PANELS.v0.json"),
            current_panels_sha256=helper.sha(root / "external/PANELS.json"),
            typed_audit_receipt_sha256=helper.sha(revision / "TYPE_AUDIT_RECEIPT.json"),
        ),
    )
    return root, revision


def reseal_review_fixture(helper, root, revision):
    audit = helper.read(revision / "TYPE_AUDIT_RECEIPT.json")
    for label in ("A", "B"):
        audit["review_files"][f"REVIEW_{label}.json"] = helper.sha(
            revision / f"REVIEW_{label}.json"
        )
    helper.save(revision / "TYPE_AUDIT_RECEIPT.json", audit)
    panels = helper.read(root / "external/PANELS.json")
    panels["typed_audit_receipt_sha256"] = helper.sha(revision / "TYPE_AUDIT_RECEIPT.json")
    helper.save(root / "external/PANELS.json", panels)
    helper.save(revision / "PANELS.v1.json", panels)
    publication = helper.read(revision / "PUBLICATION.json")
    publication.update(
        current_panels_sha256=helper.sha(root / "external/PANELS.json"),
        typed_audit_receipt_sha256=helper.sha(revision / "TYPE_AUDIT_RECEIPT.json"),
    )
    helper.save(revision / "PUBLICATION.json", publication)


def test_complete_real_schema_blind_publication_validates_without_parsing_gold(helper, registered):
    root, _ = make_typed_fixture(helper, registered)
    result = helper.verify_typed(root, external)
    assert result["status"] == "VERIFIED_TYPED_V1" and result["questions"] == 2500
    assert result["reviewer_ids"] == ["A", "B"]


@pytest.mark.parametrize(
    "change", ["missing_question", "saw_other", "nested_gold", "wrong_input", "empty_evidence"]
)
def test_incomplete_or_nonblind_review_cannot_be_promoted_by_resealing_hashes(
    helper, registered, change
):
    root, revision = make_typed_fixture(helper, registered)
    wrapper = helper.read(revision / "REVIEW_A.json")
    if change == "missing_question":
        wrapper["reviews"].pop()
    elif change == "saw_other":
        wrapper["saw_other_review"] = True
    elif change == "nested_gold":
        wrapper["reviews"][0]["evidence"] = {"gold": "forbidden"}
    elif change == "wrong_input":
        wrapper["reviews"][0]["input_identity"] = "incorrect"
    else:
        wrapper["reviews"][0]["evidence"] = " "
    helper.save(revision / "REVIEW_A.json", wrapper)
    reseal_review_fixture(helper, root, revision)
    with pytest.raises(PermissionError):
        helper.verify_typed(root, external)


@pytest.mark.parametrize("target", ["TYPE_REGISTRY.json", "MODEL_INPUTS.jsonl", "image.bin"])
def test_registry_original_inputs_and_visible_images_are_hash_bound(helper, registered, target):
    root, _ = make_typed_fixture(helper, registered)
    directory = root / (
        "external/chartqa" if target == "image.bin" else "external/chartqa_typed_v1"
    )
    with (directory / target).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises((PermissionError, ValueError)):
        helper.verify_typed(root, external)


def test_numerical_completion_without_builder_receipt_is_not_done(helper, registered):
    api = api_fixture(helper, registered, completed=orchestration.TASKS)
    api.controller.result["all_complete"] = True
    helper.save(Path(registered["root"]) / "EXECUTION_FREEZE.json", {"fixture": True})
    driver = make_driver(
        helper,
        registered,
        api,
        typed=lambda *args: typed_stub(helper, registered),
        execute=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="success", stderr=""),
    )
    with pytest.raises(FileNotFoundError):
        driver.act()
    assert not (driver.directory / "DELIVERY_VERIFIED.json").exists()
    assert (
        helper.read(driver.directory / "DELIVERY_INTENT.json")["status"]
        == "NUMERICAL_RELEASE_COMPLETE"
    )


def test_delivery_requires_exact_success_checks_hashes_and_registered_coverage(helper, registered):
    root = Path(registered["root"])
    helper.save(root / "external/PANELS.json", dict(CHARTQAPRO={"root": "external/pro"}))
    helper.save(root / "external/pro/FREEZE.json", dict(status="UNAVAILABLE", image_count=0))
    api = api_fixture(helper, registered, completed=orchestration.TASKS)
    api.controller.result["all_complete"] = True
    helper.save(root / "EXECUTION_FREEZE.json", {"fixture": True})
    calls = []

    def builder(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"synthetic builder verification fixture; not a real delivery ZIP")
        manifest = Path(command[command.index("--helper-manifest") + 1])
        receipt = dict(
            schema_version=1,
            status="VERIFIED_DELIVERY",
            run_identity_sha256=registered["run_identity_sha256"],
            scientific_source_manifest_sha256=registered["source_manifest_sha256"],
            helper_manifest_sha256=helper.sha(manifest),
            observability_extension=api.extension.verify_extension(
                root, registered["extension_manifest_path"], science=True
            ),
            archive=dict(
                path=str(output), sha256=helper.sha(output), bytes=output.stat().st_size, entries=1
            ),
            checks=dict.fromkeys(
                (
                    "crc",
                    "safe_unique_names",
                    "payload_hashes",
                    "manifest_hashes",
                    "entry_links",
                    "gzip_read_to_eof",
                    "raw_rescoring_chain",
                ),
                "PASS",
            ),
            coverage=dict(
                verified_tasks=33,
                training_paths=12,
                evaluated_models=15,
                logical_updates=768,
                training_records=98304,
                support_examples=2304,
                monitor_records=26112,
                engine_diagnostic_records=2880,
                engine_new_rollouts=512,
                engine_replayed_rollouts=256,
                engine_physical_updates=6,
                records=229500,
            ),
        )
        receipt["observability_extension"]["manifest_path"] = str(
            Path(registered["extension_manifest_path"]).relative_to(root)
        )
        helper.save(Path(str(output) + ".verification.json"), receipt)
        return SimpleNamespace(returncode=0, stdout="fixture only", stderr="")

    driver = make_driver(
        helper, registered, api, typed=lambda *args: typed_stub(helper, registered), execute=builder
    )
    assert driver.act() == "DONE"
    assert len(calls) == 1 and "--helper-manifest" in calls[0]
    assert (
        calls[0][calls[0].index("--extension-manifest") + 1]
        == registered["extension_manifest_path"]
    )
    assert (driver.directory / "DELIVERY_VERIFIED.json").exists()
    assert api.controller.tick_calls == 0


def test_running_driver_near_lease_end_records_handoff_without_science_stop(
    helper, registered, monkeypatch
):
    api = api_fixture(helper, registered)
    state = helper.submit_driver(registered, api, 1)
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    api.slurm.rows[state["job_id"]]["RunTime"] = "2-23:50:00"
    assert helper.run(registered, api, 1) == "HANDOFF_REQUIRED"
    end = helper.read(helper.attempts_root(registered) / "0001/END.json")
    assert end["scientific_time_budget_changed"] is False
    assert api.controller.status_calls == 0 and not (Path(registered["root"]) / "STOP").exists()


def test_running_driver_persists_unknown_failure_and_exits(helper, registered, monkeypatch):
    api = api_fixture(helper, registered)
    add_failed_attempt(helper, registered, api, status="SUBMISSION_UNKNOWN", terminal=None)
    state = helper.submit_driver(registered, api, 1)
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    with pytest.raises(RuntimeError, match="non-recoverable"):
        helper.run(registered, api, 1)
    end = helper.read(helper.attempts_root(registered) / "0001/END.json")
    assert end["status"] == "FAILED" and not api.controller.resumes


def test_operation_binds_extension_manifest_and_rejects_changed_source_before_import(
    helper, registered
):
    manifest = Path(registered["extension_manifest_path"])
    assert registered["extension_manifest_sha256"] == helper.sha(manifest)
    manifest.with_name("observe_runtime.py").write_text("raise AssertionError('must not import')")
    with pytest.raises(PermissionError, match="before import"):
        helper.identity(registered["identity_path"])


def test_new_operations_retain_the_exact_previously_published_typed_preparer(helper, registered):
    api = api_fixture(helper, registered)
    driver = make_driver(helper, registered, api)
    assert registered["operation_commit"] != helper.TYPED_COMMIT
    driver.check_typed_source(typed_stub(helper, registered))
    changed = dict(
        typed_stub(helper, registered), auxiliary_code_commit=registered["operation_commit"]
    )
    with pytest.raises(PermissionError, match="retained published preparer"):
        driver.check_typed_source(changed)


def test_bad_observation_freeze_blocks_resume_before_any_submission(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    add_failed_attempt(helper, registered, api)
    helper.save(Path(registered["root"]) / "EXECUTION_FREEZE.json", {"fixture": True})

    def reject(*args, **kwargs):
        raise PermissionError("missing observation static chain")

    api.extension.verify_extension = reject
    with pytest.raises(PermissionError, match="observation static chain"):
        make_driver(
            helper, registered, api, typed=lambda *args: typed_stub(helper, registered)
        ).act()
    assert not api.controller.resumes and api.controller.tick_calls == 0


def test_amendment_failure_cannot_fall_through_to_freeze_or_science(helper, registered):
    api = api_fixture(helper, registered, completed=helper.ENGINES)

    def reject(*args, **kwargs):
        raise PermissionError("CPU evidence does not bind this observer")

    api.extension.seal_amendment = reject
    with pytest.raises(PermissionError, match="CPU evidence"):
        make_driver(
            helper, registered, api, typed=lambda *args: typed_stub(helper, registered)
        ).act()
    assert not api.freezes and api.controller.tick_calls == 0


def test_dispatch_loader_ignores_unmanifested_unchecked_hash_pyc(helper, registered, tmp_path):
    manifest = Path(registered["extension_manifest_path"])
    source = manifest.with_name("instrumented_execution.py")
    marker = tmp_path / "unregistered-dispatch-cache-executed"
    payload = tmp_path / "bad_cache.py"
    payload.write_text(f"open({str(marker)!r}, 'w').write('unregistered')\n")
    cached = Path(importlib.util.cache_from_source(str(source)))
    cached.parent.mkdir()
    py_compile.compile(
        str(payload),
        cfile=str(cached),
        dfile=str(source),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    helper._EXTENSIONS.clear()
    loaded = helper.load_extension(
        Path(registered["root"]), manifest, expected_sha=registered["extension_manifest_sha256"]
    )
    assert callable(loaded.install) and not marker.exists()


def test_dispatch_loader_rechecks_executable_bytes_after_inventory_scan(
    helper, registered, monkeypatch, tmp_path
):
    manifest = Path(registered["extension_manifest_path"])
    source = manifest.with_name("instrumented_execution.py")
    marker = tmp_path / "unregistered-dispatch-source-executed"
    original = helper.sha

    def sha(path):
        value = original(path)
        if Path(path) == manifest.with_name("CPU_VALIDATION.json"):
            source.write_text(f"open({str(marker)!r}, 'w').write('unregistered')\n")
        return value

    helper._EXTENSIONS.clear()
    monkeypatch.setattr(helper, "sha", sha)
    with pytest.raises(PermissionError, match="verified-byte import"):
        helper.load_extension(
            Path(registered["root"]), manifest, expected_sha=registered["extension_manifest_sha256"]
        )
    assert not marker.exists()


def test_dispatcher_rejects_base_package_cache_before_any_scientific_import(
    helper, registered, monkeypatch
):
    code = Path(registered["code_root"])
    code.mkdir()
    (code / "injected.pyc").write_bytes(b"unmanifested scientific cache")

    def forbidden(*args, **kwargs):
        raise AssertionError("Scientific imports must not run with an unbound cache")

    monkeypatch.setattr(helper.importlib, "import_module", forbidden)
    with pytest.raises(PermissionError, match="scientific bytecode"):
        helper.load_science(registered)


@pytest.mark.parametrize("revision", ["v2", "c2"])
def test_revision_guard_runs_only_after_complete_source_authentication(
    helper, tmp_path, monkeypatch, revision
):
    root = tmp_path / "selected"
    source = root / "code/src/sr_f3_ce/protocol.py"
    source.parent.mkdir(parents=True)
    source.write_text("# authenticated source guard fixture, no model import\n")
    helper.save(root / "RUN_IDENTITY.json", dict(code_commit="c" * 40))
    helper.save(root / "plan/config/SR_F3_CE.json", dict(protocol_revision=revision))
    manifest = dict(
        code_commit="c" * 40,
        files=[
            dict(
                path="src/sr_f3_ce/protocol.py",
                bytes=source.stat().st_size,
                sha256=helper.sha(source),
            )
        ],
    )
    helper.save(root / "deployment/SOURCE_MANIFEST.json", manifest)
    checked = dict(
        code_commit="c" * 40,
        protocol_revision=revision,
        protocol_selection_sha256="d" * 64,
        request_sha256="e" * 64,
        ablation_root="/separate/ablation",
    )
    calls = []

    def guard(given_root, source_manifest):
        calls.append((given_root, source_manifest))
        return checked

    module = SimpleNamespace(__file__=str(source), verify_run_source_revision=guard)
    monkeypatch.setattr(helper.importlib, "import_module", lambda name: module)
    assert helper.source_revision(root) == checked
    assert calls == [(root.resolve(), manifest)]
    assert helper.revision_binding(checked) == {"protocol_source_guard": checked}
    calls.clear()
    source.write_text("# corrupt source must not be imported\n")
    with pytest.raises(PermissionError, match="before guard import"):
        helper.source_revision(root)
    assert calls == []


def test_original_source_guard_cannot_adopt_new_commit_without_revision(helper, registered):
    root = Path(registered["root"])
    helper.save(root / "RUN_IDENTITY.json", dict(code_commit="f" * 40))
    with pytest.raises(PermissionError, match="frozen scientific commit"):
        helper.source_revision(root)


def test_revised_guard_failure_cannot_be_replaced_by_valid_commit_metadata(
    helper, tmp_path, monkeypatch
):
    root = tmp_path / "selected"
    source = root / "code/src/sr_f3_ce/protocol.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source with rejected selection\n")
    helper.save(root / "RUN_IDENTITY.json", dict(code_commit="c" * 40))
    helper.save(root / "plan/config/SR_F3_CE.json", dict(protocol_revision="v2"))
    helper.save(
        root / "deployment/SOURCE_MANIFEST.json",
        dict(
            files=[
                dict(
                    path="src/sr_f3_ce/protocol.py",
                    bytes=source.stat().st_size,
                    sha256=helper.sha(source),
                )
            ]
        ),
    )

    def guard(*args, **kwargs):
        raise PermissionError("Protocol request/selection is not bound by the run")

    monkeypatch.setattr(
        helper.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(source), verify_run_source_revision=guard),
    )
    with pytest.raises(PermissionError, match="request/selection"):
        helper.source_revision(root)


@pytest.mark.parametrize("terminal", ["TIMEOUT", "FAILED", "CANCELLED"])
@pytest.mark.parametrize("entry_schema", [2, 3])
def test_pooled_failure_preserves_sibling_receipt_and_never_replays(
    helper, registered, terminal, entry_schema
):
    api = api_fixture(helper, registered, completed=helper.ENGINES)
    key, task = add_failed_attempt(helper, registered, api, terminal=terminal)
    root = Path(registered["root"])
    path = root / "orchestration/jobs" / key
    intent = helper.read(path / "INTENT.json")
    sibling = "train--SRF3CE_GOLD_b0"
    intent.update(execution_entry_schema=entry_schema, gpu_count=2, task_ids=[task, sibling])
    helper.save(path / "INTENT.json", intent)
    receipt = orchestration.receipt_path(root, sibling)
    helper.save(receipt, dict(status="PASS", job_id="88"))
    before = receipt.read_bytes()
    driver = make_driver(helper, registered, api)
    assert driver.act() == "HANDOFF_REQUIRED"
    handoff = helper.read(driver.directory / "POOL_HANDOFF_REQUIRED.json")
    assert handoff["reason"] == "POOLED_ALLOCATION_REQUIRES_MEMBER_RECONCILIATION"
    assert handoff["completed_task_receipts"] == {sibling: helper.sha(receipt)}
    assert handoff["resubmitted"] is False
    assert receipt.read_bytes() == before
    assert not api.controller.resumes and not api.slurm.calls
    assert not (root / "STOP").exists()


def test_pooled_failure_handoff_cannot_be_mistaken_for_cpu_lease_continuation(helper, registered):
    api = api_fixture(helper, registered)
    first = helper.submit_driver(registered, api, 1)
    api.slurm.finish(first["job_id"])
    previous = helper.attempts_root(registered) / "0001"
    helper.save(previous / "END.json", dict(status="HANDOFF_REQUIRED"))
    helper.save(previous / "POOL_HANDOFF_REQUIRED.json", dict(reason="member failure"))
    before = list(api.slurm.calls)
    with pytest.raises(PermissionError, match="member failure requires reconciliation"):
        helper.submit_driver(registered, api, 2, 1)
    assert api.slurm.calls == before
    assert not (helper.attempts_root(registered) / "0002/INTENT.json").exists()


def test_running_driver_ends_on_pool_handoff_instead_of_polling_or_resubmitting(
    helper, registered, monkeypatch
):
    api = api_fixture(helper, registered)
    state = helper.submit_driver(registered, api, 1)
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    calls = []
    monkeypatch.setattr(
        helper.Driver, "poll", lambda self: calls.append("poll") or "HANDOFF_REQUIRED"
    )
    monkeypatch.setattr(helper.time, "sleep", lambda _: pytest.fail("must exit on pool handoff"))
    assert helper.run(registered, api, 1) == "HANDOFF_REQUIRED"
    assert calls == ["poll"]
    end = helper.read(helper.attempts_root(registered) / "0001/END.json")
    assert end["status"] == "HANDOFF_REQUIRED"


def test_pool_source_checked_before_executing_import(helper, tmp_path):
    source = tmp_path / "pooled_execution.py"
    source.write_text("raise AssertionError('unverified pool source executed')\n")
    with pytest.raises(PermissionError, match="before import"):
        helper.load_pool_source(tmp_path, expected_sha="0" * 64)


class ActualCapacity:
    def __init__(self, gpu=8, submitted=5):
        self.gpu, self.submitted, self.fresh = gpu, submitted, []

    def limits(self, fresh=False):
        self.fresh.append(fresh)
        return dict(
            gpu_limit=self.gpu,
            submitted_limit=self.submitted,
            max_tasks_per_job=self.gpu,
            gpu_per_user_limit=None,
        )

    def headroom(self, usage, limits=None):
        limits = self.limits() if limits is None else limits
        return dict(
            gpus=limits["gpu_limit"] - usage["gpus"],
            submitted=limits["submitted_limit"] - usage["submitted"],
        )


@pytest.mark.parametrize("occupied,available", [(5, True), (7, True), (8, False)])
def test_live_capacity_driver_has_no_artificial_five_gpu_ceiling(
    helper, registered, occupied, available
):
    api = api_fixture(helper, registered)
    api.capacity = ActualCapacity()
    api.slurm.rows["99"] = dict(QOS=helper.QOS, UserId="bob(2)", ReqTRES=f"gres/gpu={occupied}")
    driver = make_driver(helper, registered, api)
    snapshot = driver.snapshot()
    assert snapshot["teacher_usage"]["gpus"] == occupied
    assert driver.has_capacity(snapshot, "train") is available
    assert driver.has_capacity(snapshot, "release")
    evidence = helper.read(driver.directory / "QUEUE_SNAPSHOT.json")
    assert evidence["capacity_limits"]["gpu_limit"] == 8


def test_cpu_handoff_uses_actual_limits_and_preserves_frozen_configuration(helper, registered):
    api = api_fixture(helper, registered)
    api.capacity = ActualCapacity()
    api.slurm.rows["99"] = dict(QOS=helper.QOS, UserId="alice(1)", ReqTRES="gres/gpu=7")
    config = Path(registered["root"]) / "orchestration/CONFIG.json"
    before = config.read_bytes()
    state = helper.submit_driver(registered, api, 1)
    assert state["status"] == "SUBMITTED"
    assert api.capacity.fresh == [True, True]
    assert config.read_bytes() == before
    directory = helper.attempts_root(registered) / "0001"
    assert helper.read(directory / "PRE_SUBMIT_CAPACITY.json")["limits"]["gpu_limit"] == 8
    assert helper.read(directory / "PRE_RELEASE_CAPACITY.json")["usage"]["gpus"] == 7


def test_real_cap_drop_blocks_cpu_submit_without_sbatch(helper, registered):
    api = api_fixture(helper, registered)
    api.capacity = ActualCapacity(gpu=6)
    api.slurm.rows["99"] = dict(QOS=helper.QOS, UserId="alice(1)", ReqTRES="gres/gpu=7")
    with pytest.raises(RuntimeError, match="live scheduler limits"):
        helper.submit_driver(registered, api, 1)
    assert not api.slurm.calls


def test_authorized_pool_cannot_silently_use_legacy_capacity(helper, registered):
    api = api_fixture(helper, registered)
    registered["task_pool"] = dict(capacity_policy="LIVE_SLURM_QOS_AND_ASSOCIATION")
    with pytest.raises(PermissionError, match="lacks live scheduler capacity"):
        helper.capacity_limits(registered, api)


def test_capacity_source_checked_before_execution(helper, tmp_path):
    (tmp_path / "scheduler_capacity.py").write_text("raise AssertionError('not verified')")
    with pytest.raises(PermissionError, match="before import"):
        helper.load_capacity_source(tmp_path, expected_sha="0" * 64)


class OrdinaryCPUCapacity:
    def __init__(self, limit=5):
        self.limit, self.queries = limit, []

    def dispatcher_limits(self, qos):
        assert qos == "rose"
        self.queries.append(qos)
        return {"qos": qos, "submitted_limit": self.limit}

    def dispatcher_headroom(self, queue, limits):
        count = sum(row.get("QOS") == "rose" for row in queue.values())
        return {"submitted": limits["submitted_limit"] - count, "used": count}


def test_ordinary_cpu_ignores_full_teacher_and_preserves_science_config(helper, registered):
    registered["dispatcher_qos"] = "rose"
    api = api_fixture(helper, registered)
    api.capacity = OrdinaryCPUCapacity()
    for i in range(5):
        api.slurm.rows[str(i)] = dict(QOS=helper.QOS, UserId="alice(1)", ReqTRES="gres/gpu=2")
    config = Path(registered["root"]) / "orchestration/CONFIG.json"
    before = config.read_bytes()
    state = helper.submit_driver(registered, api, 1)
    directory = helper.attempts_root(registered) / "0001"
    intent = helper.read(directory / "INTENT.json")
    assert state["status"] == "SUBMITTED" and intent["qos"] == "rose"
    assert "--qos=rose" in intent["sbatch"] and intent["gpu_count"] == 0
    assert registered["qos"] == helper.QOS and config.read_bytes() == before
    assert api.capacity.queries == ["rose", "rose"]
    with pytest.raises(RuntimeError, match="already attempted"):
        helper.submit_driver(registered, api, 1)
    assert sum(cmd[0] == "sbatch" for cmd in api.slurm.calls) == 1


def test_ordinary_cpu_full_rose_prevents_submission(helper, registered):
    registered["dispatcher_qos"] = "rose"
    api = api_fixture(helper, registered)
    api.capacity = OrdinaryCPUCapacity(limit=0)
    with pytest.raises(RuntimeError, match="dispatcher QoS"):
        helper.submit_driver(registered, api, 1)
    assert not api.slurm.calls


def test_ordinary_cpu_lease_continuation_uses_retained_qos(helper, registered):
    registered["dispatcher_qos"] = "rose"
    api = api_fixture(helper, registered)
    api.capacity = OrdinaryCPUCapacity()
    first = helper.submit_driver(registered, api, 1)
    api.slurm.finish(first["job_id"], state="TIMEOUT")
    second = helper.submit_driver(registered, api, 2, 1)
    assert second["status"] == "SUBMITTED"
    assert helper.read(helper.attempts_root(registered) / "0001/TERMINAL.json")["qos"] == "rose"


def test_cpu_qos_registration_requires_explicit_bound_authorization(helper, registered):
    with pytest.raises(PermissionError, match="bound operational handoff"):
        helper.register(
            registered["root"],
            registered["python"],
            registered["user"],
            registered["operation_commit"],
            registered["extension_manifest_path"],
            dispatcher_qos="rose",
        )


def test_dispatcher_active_rows_delegate_bound_split_filter_without_changing_history(
    helper, registered
):
    api = api_fixture(helper, registered)
    root = Path(registered["root"])
    directory = root / "orchestration/jobs/job-deadbeefdeadbeef"
    helper.save(directory / "INTENT.json", dict(kind="eval", gpu_count=7, qos=helper.QOS))
    helper.save(
        directory / "STATE.json",
        dict(key=directory.name, status="SUBMITTED", job_id="211093", gpu_count=7, qos=helper.QOS),
    )
    before = (directory / "STATE.json").read_bytes()
    seen = []

    def filtered(rows):
        seen.extend(rows)
        return [(i, s) for i, s in rows if s["key"] != directory.name]

    api.task_pool = SimpleNamespace(filter_rows=filtered)
    snapshot = make_driver(helper, registered, api).snapshot()
    assert seen and not snapshot["jobs"]
    assert snapshot["teacher_usage"] == dict(gpus=0, submitted=0)
    assert (directory / "STATE.json").read_bytes() == before

    def invalid(rows):
        raise PermissionError("Superseded pending ledger changed")

    api.task_pool.filter_rows = invalid
    with pytest.raises(PermissionError, match="ledger changed"):
        make_driver(helper, registered, api).snapshot()


@pytest.fixture
def failed_split_driver(tmp_path, monkeypatch):
    from test_dispatch_efficiency import retired_driver

    env = retired_driver.__wrapped__(tmp_path, monkeypatch)
    prior = env.retired
    env.helper.save(
        prior.directory / "END.json",
        dict(
            status="FAILED",
            error="PermissionError('Actual teacher capacity exceeds the live scheduler limit')",
        ),
    )
    terminal = dict(prior.terminal, state="FAILED", exit_code="1:0")
    env.raw.accounting[prior.state["job_id"]] = terminal
    receipt = env.helper.read(prior.transition)
    receipt.update(status="OPERATOR_HANDOFF_FAILED", terminal=terminal)
    receipt.pop("cancel_returncode")
    env.helper.save(prior.transition, receipt)
    auth = env.helper.read(prior.authorization)
    auth.update(
        terminal=terminal,
        transition_sha256=env.helper.sha(prior.transition),
        pending_split={"fixture": "validated split"},
    )
    auth["predecessor"]["end_sha256"] = env.helper.sha(prior.directory / "END.json")
    env.helper.save(prior.authorization, auth)

    def check_split(root, binding):
        if binding != {"fixture": "validated split"}:
            raise PermissionError("Pending split binding differs")

    monkeypatch.setattr(
        env.helper,
        "load_pool_source",
        lambda *a, **k: SimpleNamespace(check_pending_split=check_split),
    )
    env.value["task_pool"] = dict(pending_split=auth["pending_split"])
    env.value["dispatch_efficiency"] = env.helper.check_efficiency_authorization(
        env.root, prior.authorization
    )
    return env


def test_exact_failed_cpu_handoff_is_bound_to_split_and_live_terminal(failed_split_driver):
    env = failed_split_driver
    p = env.retired
    original = {f.name: f.read_bytes() for f in p.directory.iterdir() if f.is_file()}
    terminal = env.helper.verify_efficiency_predecessor(
        env.value, env.api, p.directory, p.intent, p.state, env.raw.queue("alice")
    )
    assert terminal["state"] == "FAILED" and terminal["exit_code"] == "1:0"
    assert original == {f.name: f.read_bytes() for f in p.directory.iterdir() if f.is_file()}
    env.raw.accounting[p.state["job_id"]] = dict(terminal, exit_code="2:0")
    with pytest.raises(PermissionError, match="live accounting"):
        env.helper.verify_efficiency_predecessor(
            env.value, env.api, p.directory, p.intent, p.state, {}
        )


@pytest.mark.parametrize(
    "mutation", ["end_error", "end_hash", "exit", "split", "operation_split", "lease"]
)
def test_failed_cpu_handoff_rejects_unbound_or_different_failure(failed_split_driver, mutation):
    env = failed_split_driver
    p = env.retired
    auth = env.helper.read(p.authorization)
    receipt = env.helper.read(p.transition)
    if mutation in {"end_error", "end_hash"}:
        env.helper.save(p.directory / "END.json", dict(status="FAILED", error="different failure"))
        if mutation == "end_error":
            auth["predecessor"]["end_sha256"] = env.helper.sha(p.directory / "END.json")
    elif mutation == "exit":
        auth["terminal"]["exit_code"] = "2:0"
        receipt["terminal"] = auth["terminal"]
    elif mutation == "split":
        auth.pop("pending_split")
    elif mutation == "operation_split":
        env.value["task_pool"]["pending_split"] = {"fixture": "other split"}
    else:
        receipt["orchestration_lease_held"] = False
    env.helper.save(p.transition, receipt)
    auth["transition_sha256"] = env.helper.sha(p.transition)
    env.helper.save(p.authorization, auth)
    with pytest.raises(PermissionError):
        env.value["dispatch_efficiency"] = env.helper.check_efficiency_authorization(
            env.root, p.authorization
        )
        env.helper.verify_efficiency_predecessor(
            env.value, env.api, p.directory, p.intent, p.state, {}
        )


@pytest.mark.parametrize("room,maximum,expected", [(2, 8, False), (3, 8, True), (3, 2, False)])
def test_split_capacity_waits_for_whole_remaining_group(
    helper, registered, room, maximum, expected
):
    api = api_fixture(helper, registered)
    api.capacity = ActualCapacity()
    evaluations = [k for k, t in orchestration.TASKS.items() if t["kind"] == "eval"]
    registered["task_pool"] = {
        "pending_split": {
            "groups": [
                dict(gpu_count=4, task_ids=evaluations[:8]),
                dict(gpu_count=3, task_ids=evaluations[8:]),
            ]
        }
    }
    snapshot = dict(
        teacher_usage=dict(gpus=8 - room, submitted=2),
        capacity_limits=dict(gpu_limit=8, submitted_limit=5, max_tasks_per_job=maximum),
        jobs=[dict(task_ids=evaluations[:8], status="SUBMITTED")],
    )
    driver = make_driver(helper, registered, api)
    assert driver.has_capacity(snapshot, "eval") is expected
    snapshot["jobs"][0]["task_ids"] = evaluations[:7]
    with pytest.raises(PermissionError, match="partially claimed"):
        driver.has_capacity(snapshot, "eval")


@pytest.mark.parametrize("room", [2, 3])
def test_fast_split_wait_skips_expensive_submission_until_group_fits(helper, registered, room):
    api = api_fixture(helper, registered)
    api.capacity = ActualCapacity()
    api.efficiency = SimpleNamespace(check=lambda: None)
    evaluations = [k for k, t in orchestration.TASKS.items() if t["kind"] == "eval"]
    prerequisites = [k for k, t in orchestration.TASKS.items() if t["kind"] in {"engine", "train"}]
    registered["task_pool"] = {
        "pending_split": {
            "groups": [
                dict(gpu_count=4, task_ids=evaluations[:8]),
                dict(gpu_count=3, task_ids=evaluations[8:]),
            ]
        }
    }
    snapshot = dict(
        teacher_usage=dict(gpus=8 - room, submitted=2),
        capacity_limits=dict(gpu_limit=8, submitted_limit=5, max_tasks_per_job=8),
        jobs=[
            dict(task_ids=prerequisites, status="COMPLETE", terminal=None),
            dict(task_ids=evaluations[:8], status="SUBMITTED", terminal=None),
        ],
    )
    driver = make_driver(helper, registered, api)
    events = []
    driver.require_typed = lambda: events.append("typed")
    driver.phase = lambda phase, **kw: events.append(phase)
    api.controller.submit = lambda kind: events.append("submit")
    assert driver.fast_science(snapshot) == (
        "WAIT_CAPACITY" if room == 2 else "WAIT_CAPACITY_RECHECK"
    )
    assert events == (
        [] if room == 2 else ["typed", "SUBMIT_FIXED_TASK", "submit", "SUBMIT_RETURNED"]
    )
    assert api.controller.status_calls == api.controller.tick_calls == 0


@pytest.mark.parametrize("room", [2, 3])
def test_claimed_rejected_split_group_uses_exact_group_capacity(helper, registered, room):
    api = api_fixture(helper, registered)
    api.capacity = ActualCapacity()
    api.efficiency = SimpleNamespace(check=lambda: None)
    evaluations = [k for k, t in orchestration.TASKS.items() if t["kind"] == "eval"]
    registered["task_pool"] = {
        "pending_split": {
            "groups": [
                dict(gpu_count=4, task_ids=evaluations[:8]),
                dict(gpu_count=3, task_ids=evaluations[8:]),
            ]
        }
    }
    snapshot = dict(
        teacher_usage=dict(gpus=8 - room, submitted=2),
        capacity_limits=dict(gpu_limit=8, submitted_limit=5, max_tasks_per_job=8),
        jobs=[
            dict(
                task_ids=evaluations[:8],
                status="SUBMITTED",
                kind="eval",
                terminal=None,
                unknown=None,
            ),
            dict(
                task_ids=evaluations[8:],
                status="SUBMISSION_UNKNOWN",
                kind="eval",
                terminal=None,
                unknown="SUBMISSION_UNKNOWN",
            ),
        ],
    )
    directory = Path(registered["root"]) / "orchestration/jobs/job-abcdefabcdefabcd"
    helper.save(directory / "INTENT.json", dict(kind="eval", task_ids=evaluations[8:]))
    helper.save(directory / "STATE.json", dict(key=directory.name, status="SUBMISSION_UNKNOWN"))
    retries = []
    api.repair = SimpleNamespace(
        known_rejections_only=lambda states: True,
        retry_pending=lambda: retries.append("retry") or "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION",
    )
    api.controller.submit = lambda kind: pytest.fail(
        "Must repair the existing claim, never create a new one"
    )
    driver = make_driver(helper, registered, api)
    assert driver.has_capacity(snapshot, "eval") is (room == 3)
    assert driver.fast_science(snapshot) == "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
    assert retries == ([] if room == 2 else ["retry"])
    snapshot["jobs"][1]["task_ids"] = evaluations[8:-1]
    with pytest.raises(PermissionError, match="differs from pending split"):
        driver.has_capacity(snapshot, "eval")


def test_rejected_four_gpu_group_takes_priority_over_unclaimed_three(helper, registered):
    api = api_fixture(helper, registered)
    api.capacity = ActualCapacity()
    evaluations = [k for k, t in orchestration.TASKS.items() if t["kind"] == "eval"]
    registered["task_pool"] = {
        "pending_split": {
            "groups": [
                dict(gpu_count=4, task_ids=evaluations[:8]),
                dict(gpu_count=3, task_ids=evaluations[8:]),
            ]
        }
    }
    snapshot = dict(
        teacher_usage=dict(gpus=5, submitted=2),
        capacity_limits=dict(gpu_limit=8, submitted_limit=5, max_tasks_per_job=8),
        jobs=[
            dict(
                task_ids=evaluations[:8],
                status="SUBMISSION_UNKNOWN",
                kind="eval",
                unknown="SUBMISSION_UNKNOWN",
            )
        ],
    )
    driver = make_driver(helper, registered, api)
    assert driver.has_capacity(snapshot, "eval") is False
    snapshot["teacher_usage"]["gpus"] = 4
    assert driver.has_capacity(snapshot, "eval") is True
