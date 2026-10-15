"""CPU-only fixtures for the separate operational dispatcher, never live Slurm."""

from __future__ import annotations

import copy
import importlib.util
import json
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
    root = tmp_path / "run"
    operation = root / "operations" / ("a" * 40)
    operation.mkdir(parents=True)
    shutil.copyfile(helper.__file__, operation / "dispatch_registered.py")
    for name in helper.HELPERS[1:]:
        (operation / name).write_text("# synthetic operational source fixture\n")
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
    path = helper.register(root, python, "alice", "a" * 40)
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
                ReqTRES="cpu=4,mem=12G,node=1",
                ReqNodeList=args["nodelist"],
                JobState="PENDING",
                Reason="JobHeldUser",
                TimeLimit="3-00:00:00",
                Requeue="0",
                RunTime="00:00:00",
                NodeList="cpu-2",
                AllocTRES="cpu=4,mem=12G,node=1",
            )
            self.scripts[job] = Path(command[-1]).read_text()
            return SimpleNamespace(
                returncode=int(self.ambiguous_submit),
                stdout="" if self.ambiguous_submit else job + "\n",
                stderr="",
            )
        if command[:2] == ["scontrol", "release"]:
            self.rows[command[2]].update(JobState="RUNNING", Reason="None")
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
            completed=list(completed), blocked=False, all_complete=False, attempts=[]
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
    )

    def freeze(run):
        api.freezes.append(str(run))
        helper.save(root / "EXECUTION_FREEZE.json", {"fixture_science": True})

    api.bootstrap = SimpleNamespace(freeze_science=freeze)
    return api


def typed_stub(helper, registered):
    return dict(
        status="VERIFIED_TYPED_V1",
        auxiliary_code_commit=registered["operation_commit"],
        preparation_script_sha256=helper.sha(
            Path(registered["identity_path"]).parent / "finalize_external_types.py"
        ),
    )


def make_driver(helper, registered, api, **kwargs):
    directory = helper.attempts_root(registered) / "0001"
    return helper.Driver(registered, api, directory, **kwargs)


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
    with pytest.raises(RuntimeError, match="No teacher submission slot"):
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
        auxiliary_code_commit=registered["operation_commit"],
        auxiliary_commit_identity_supplied_by_operator=True,
        preparation_script_sha256=helper.sha(
            Path(registered["identity_path"]).parent / "finalize_external_types.py"
        ),
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
        helper.save(Path(str(output) + ".verification.json"), receipt)
        return SimpleNamespace(returncode=0, stdout="fixture only", stderr="")

    driver = make_driver(
        helper, registered, api, typed=lambda *args: typed_stub(helper, registered), execute=builder
    )
    assert driver.act() == "DONE"
    assert len(calls) == 1 and "--helper-manifest" in calls[0]
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
