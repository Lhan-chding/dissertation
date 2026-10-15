"""No-model Slurm fixtures for explicit observation launch/source identity."""

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

from sr_f3_ce import endpoint_runner, orchestration, protocol


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path, monkeypatch):
    checkout = Path(__file__).resolve().parents[2]
    old_tests = load_module(
        Path(__file__).with_name("test_orchestration.py"), "old_scheduler_fixtures"
    )
    extension = load_module(
        checkout / "scripts/sr_f3_ce/instrumented_execution.py", "new_launch_fixture"
    )
    root, code = tmp_path / "run", tmp_path / "run/code"
    scripts = code / "scripts/sr_f3_ce"
    scripts.mkdir(parents=True)
    for name in ("submit.py", "run_worker.py"):
        (scripts / name).write_text("# fake scientific script for no-model scheduling tests\n")
    original_methods = code / "src/fake_runtime.py"
    original_methods.parent.mkdir()
    original_methods.write_text("# original method source fixture\n")
    orchestration.save_json(
        root / "RUN_IDENTITY.json",
        dict(experiment_id=extension.EXPERIMENT, code_commit=extension.BASE_COMMIT, root=str(root)),
    )
    orchestration.save_json(
        root / "deployment/SOURCE_MANIFEST.json",
        dict(
            files=[
                dict(
                    path=str(p.relative_to(code)), sha256=extension._sha(p), bytes=p.stat().st_size
                )
                for p in (original_methods, scripts / "submit.py", scripts / "run_worker.py")
            ]
        ),
    )
    orchestration.save_json(root / "plan/config/SR_F3_CE.json", {"synthetic_fixed_config": True})
    monkeypatch.setattr(protocol, "verify_run", lambda *args, **kwargs: ({}, {}))
    slurm = old_tests.FakeSlurm()
    monkeypatch.setattr(orchestration, "Slurm", lambda: slurm)
    original_class = orchestration.Controller
    # Restore the process-global explicit class replacement at fixture teardown.
    monkeypatch.setattr(orchestration, "Controller", original_class)
    original = original_class(
        root,
        code,
        python=sys.executable,
        user="alice",
        pool_size=1,
        slurm=slurm,
        cpu_qos=orchestration.TEACHER_QOS,
    )
    for kind in ("p0", "engine", "engine", "engine", "engine"):
        state = original.submit(kind)
        old_tests.complete_allocation(original, state, monkeypatch)
    baseline = {
        name: extension._sha(root / name)
        for name in (
            "RUN_IDENTITY.json",
            "deployment/SOURCE_MANIFEST.json",
            "orchestration/CONFIG.json",
        )
    }
    typed = root / "external/revisions/typed_v1"
    orchestration.save_json(
        typed / "TYPE_AUDIT_RECEIPT.json",
        dict(
            status="TWO_INDEPENDENT_REVIEW_PASSES_COMPLETED",
            auxiliary_code_commit=extension.TYPED_COMMIT,
            preparation_script_sha256=extension.TYPED_SCRIPT_SHA,
        ),
    )
    orchestration.save_json(typed / "PUBLICATION.json", {"synthetic_typed_proof": True})
    typed_verified = dict(
        status="VERIFIED_TYPED_V1",
        auxiliary_code_commit=extension.TYPED_COMMIT,
        preparation_script_sha256=extension.TYPED_SCRIPT_SHA,
        receipt_sha256=extension._sha(typed / "TYPE_AUDIT_RECEIPT.json"),
        publication_sha256=extension._sha(typed / "PUBLICATION.json"),
    )
    directory = root / "native/observability" / ("b" * 40)
    directory.mkdir(parents=True)
    for name in ("instrumented_execution.py", "observe_runtime.py"):
        shutil.copyfile(checkout / "scripts/sr_f3_ce" / name, directory / name)
    monkeypatch.setattr(extension, "__file__", str(directory / "instrumented_execution.py"))
    sources = [
        dict(
            path=name,
            sha256=extension._sha(directory / name),
            bytes=(directory / name).stat().st_size,
        )
        for name in ("instrumented_execution.py", "observe_runtime.py")
    ]
    cpu = dict(
        status="PASS",
        scope="CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION",
        tests_passed=1,
        gpu_validation_performed=False,
        new_model_generations=0,
        new_optimizer_steps=0,
        source_sha256={r["path"]: r["sha256"] for r in sources},
    )
    orchestration.save_json(directory / "CPU_VALIDATION.json", cpu)
    manifest = dict(
        schema_version=1,
        experiment_id=extension.EXPERIMENT,
        status="FROZEN_OBSERVABILITY_EXTENSION",
        extension_commit="b" * 40,
        base_code_commit=extension.BASE_COMMIT,
        run_identity_sha256=baseline["RUN_IDENTITY.json"],
        base_source_manifest_sha256=baseline["deployment/SOURCE_MANIFEST.json"],
        sources=sources,
        executed_sources=["instrumented_execution.py", "observe_runtime.py"],
        cpu_validation=dict(
            path="CPU_VALIDATION.json",
            sha256=extension._sha(directory / "CPU_VALIDATION.json"),
            bytes=(directory / "CPU_VALIDATION.json").stat().st_size,
        ),
    )
    manifest_path = directory / "SOURCE_MANIFEST.json"
    orchestration.save_json(manifest_path, manifest)
    monkeypatch.setattr(
        extension,
        "_import_base",
        lambda *args: SimpleNamespace(orchestration=orchestration, protocol=protocol),
    )
    cls = extension.install(root, manifest_path)
    controller = cls(
        root,
        code,
        python=sys.executable,
        user="alice",
        pool_size=1,
        slurm=slurm,
        cpu_qos=orchestration.TEACHER_QOS,
    )
    return SimpleNamespace(
        module=extension,
        root=root,
        code=code,
        manifest=manifest_path,
        controller=controller,
        original=original,
        baseline=baseline,
        old_tests=old_tests,
        typed_verified=typed_verified,
        slurm=slurm,
    )


def seal_and_freeze(setup):
    e, root = setup.module, setup.root
    amendment = e.seal_amendment(root, setup.manifest, typed_verified=setup.typed_verified)
    paths = [
        setup.manifest,
        setup.manifest.with_name("CPU_VALIDATION.json"),
        setup.manifest.with_name("AMENDMENT.json"),
    ]
    freeze = dict(
        status="FROZEN_BEFORE_SCIENCE",
        run_identity_sha256=e._sha(root / "RUN_IDENTITY.json"),
        engine_receipts={
            t: e._sha(root / "tasks" / t / "COMPLETE.json") for t in e.ENGINE_IDS if t != "p0"
        },
        engineering_files={str(p.relative_to(root)): e._sha(p) for p in paths},
    )
    orchestration.save_json(root / "EXECUTION_FREEZE.json", freeze)
    return amendment


def observation_receipt(setup, intent, task_id, job):
    e, root = setup.module, setup.root
    binding = e._binding(root, intent, task_id, job)
    directory = root / "accounting/observability" / intent["key"] / task_id
    directory.mkdir(parents=True, exist_ok=True)
    method_path = root / "code/src/fake_runtime.py"
    observer = dict(
        observer_source_sha256=e._sha(setup.manifest.with_name("observe_runtime.py")),
        runtime_class="sr_f3_ce.runtime.SRF3CERuntime",
        installation_status="INSTALLED",
        installation_id="fixture-observation",
        original_methods={
            name: dict(file=str(method_path), file_sha256=e._sha(method_path))
            for name in ("__init__", "generate_group", "_native_forward")
        },
    )
    orchestration.save_json(
        directory / "INSTALLATION.json",
        dict(schema_version=1, status="INSTALLED", execution_binding=binding, observer=observer),
    )
    common = dict(
        execution_binding=binding,
        installation_id="fixture-observation",
        runtime_id="r1",
        scope_id="s1",
    )
    rows = [
        dict(common, event="SCOPE_BEGIN"),
        dict(common, event="SCOPE_END", coverage_status="COMPLETE", outcome="RETURNED"),
    ]
    events = directory / "events.jsonl"
    events.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = dict(
        e._events_summary(events, binding, observer["installation_id"]),
        path=str(events.relative_to(root)),
    )
    snapshot = dict(
        sealed=True,
        installation_id="fixture-observation",
        active_scopes=[],
        write_errors=[],
        events_written={"SCOPE_BEGIN": 1, "SCOPE_END": 1},
    )
    orchestration.save_json(
        directory / "FINAL.json",
        dict(
            schema_version=1,
            status="COMPLETE",
            execution_binding=binding,
            installation_sha256=e._sha(directory / "INSTALLATION.json"),
            events=summary,
            observer_snapshot=snapshot,
            task_receipt_sha256=e._sha(root / "tasks" / task_id / "COMPLETE.json"),
        ),
    )
    return directory


def complete_new(setup, state, monkeypatch):
    root, e = setup.root, setup.module
    job = state["job_id"]
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    intent = orchestration.read_json(root / "orchestration/jobs" / state["key"] / "INTENT.json")

    def popen(command, env):
        task_id = command[-1]
        assert command == e._worker_command(root, task_id, intent["python"])
        with monkeypatch.context() as patch:
            for key, value in env.items():
                patch.setenv(key, value)
            output = root / (task_id + ".raw")
            output.write_text("synthetic registered output fixture, no model called")
            evidence = setup.old_tests.evidence(task_id)
            evidence["checkpoint_sha256"] = e._sha(output)
            orchestration.write_task_receipt(
                root, task_id, artifacts=[output.name], evidence=evidence
            )
            observation_receipt(setup, intent, task_id, job)
        return SimpleNamespace(wait=lambda: 0)

    e.run_job(root, state["key"], popen=popen)
    fields = setup.slurm.rows.pop(job)
    setup.slurm.accounting[job] = dict(
        job_id=job,
        state="COMPLETED",
        exit_code="0:0",
        user="alice",
        account="rose",
        qos=fields["QOS"],
        name=fields["JobName"],
    )
    return setup.controller.status()


def test_install_and_amendment_preserve_original_identities_and_engine_evidence(setup):
    assert set(setup.controller.completed()) == setup.module.ENGINE_IDS
    amendment = seal_and_freeze(setup)
    assert amendment["repeat_engine"] is False and amendment["new_model_generations"] == 0
    assert setup.module.verify_extension(setup.root, setup.manifest, science=True)[
        "amendment_sha256"
    ]
    assert all(setup.module._sha(setup.root / name) == sha for name, sha in setup.baseline.items())
    assert setup.module.install(setup.root, setup.manifest) is type(setup.controller)


def test_science_cannot_submit_without_amendment_or_final_freeze(setup):
    calls = len(setup.slurm.calls)
    with pytest.raises(PermissionError, match="sealed observation amendment"):
        setup.controller.submit("train")
    assert len(setup.slurm.calls) == calls
    setup.module.seal_amendment(setup.root, setup.manifest, typed_verified=setup.typed_verified)
    with pytest.raises(FileNotFoundError):
        setup.controller.submit("train")
    assert len(setup.slurm.calls) == calls


@pytest.mark.parametrize(
    "name", ["instrumented_execution.py", "observe_runtime.py", "CPU_VALIDATION.json"]
)
def test_changed_extension_or_cpu_acceptance_bytes_are_rejected(setup, name):
    with setup.manifest.with_name(name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(PermissionError, match="artifact identity changed"):
        setup.module.verify_extension(setup.root, setup.manifest)


@pytest.mark.parametrize("name", ["SOURCE_MANIFEST.json", "CPU_VALIDATION.json", "AMENDMENT.json"])
def test_freeze_requires_entire_static_extension_chain(setup, name):
    seal_and_freeze(setup)
    freeze_path = setup.root / "EXECUTION_FREEZE.json"
    freeze = orchestration.read_json(freeze_path)
    del freeze["engineering_files"][str(setup.manifest.with_name(name).relative_to(setup.root))]
    orchestration.save_json(freeze_path, freeze)
    with pytest.raises(PermissionError, match="full extension static chain"):
        setup.controller.submit("train")


def test_real_wrapper_commands_are_bound_and_old_entry_is_not_impersonated(setup, monkeypatch):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    intent = orchestration.read_json(
        setup.root / "orchestration/jobs" / state["key"] / "INTENT.json"
    )
    assert intent["job_entry_command"][1] == str(
        setup.manifest.with_name("instrumented_execution.py")
    )
    assert "--mode job" in Path(intent["script"]).read_text()
    assert intent["job_name"].endswith("-LH") and intent["gpu_count"] == 1
    status = complete_new(setup, state, monkeypatch)
    task = intent["task_ids"][0]
    assert task in status["completed"]
    launch = orchestration.read_json(
        setup.root / "orchestration/jobs" / state["key"] / (task + ".LAUNCH.json")
    )
    assert launch["command"] == intent["worker_entry_commands"][task]
    assert "run_worker.py" not in launch["command"][1]
    assert task in endpoint_runner._completed(setup.root)


def test_command_tampering_in_immutable_intent_is_rejected_even_with_state_rehash(setup):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    directory = setup.root / "orchestration/jobs" / state["key"]
    intent = orchestration.read_json(directory / "INTENT.json")
    intent["worker_entry_commands"][intent["task_ids"][0]][1] = str(
        setup.code / "scripts/sr_f3_ce/run_worker.py"
    )
    orchestration.save_json(directory / "INTENT.json", intent)
    state["intent_sha256"] = setup.module._sha(directory / "INTENT.json")
    orchestration.save_json(directory / "STATE.json", state)
    with pytest.raises(PermissionError, match="entry/source/amendment"):
        setup.controller.status()


def test_false_old_launch_record_is_not_accepted_for_new_science(setup, monkeypatch):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    complete_new(setup, state, monkeypatch)
    directory = setup.root / "orchestration/jobs" / state["key"]
    task = orchestration.read_json(directory / "INTENT.json")["task_ids"][0]
    launch = orchestration.read_json(directory / (task + ".LAUNCH.json"))
    launch["command"] = [
        str(Path(sys.executable).resolve()),
        str(setup.code / "scripts/sr_f3_ce/run_worker.py"),
        "--root",
        str(setup.root),
        "--task-id",
        task,
    ]
    orchestration.save_json(directory / (task + ".LAUNCH.json"), launch)
    with pytest.raises(PermissionError, match="process identity"):
        setup.controller.completed()


def test_observation_partial_or_mutated_ledger_never_certifies_completion(setup, monkeypatch):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    complete_new(setup, state, monkeypatch)
    intent = orchestration.read_json(
        setup.root / "orchestration/jobs" / state["key"] / "INTENT.json"
    )
    task = intent["task_ids"][0]
    directory = setup.root / "accounting/observability" / state["key"] / task
    rows = (directory / "events.jsonl").read_text().splitlines()
    (directory / "events.jsonl").write_text(rows[0] + "\n")
    with pytest.raises(PermissionError, match="Unfinished observation"):
        setup.controller.completed()


def test_future_resume_keeps_original_raw_verifier_and_real_wrapper_command(setup, monkeypatch):
    seal_and_freeze(setup)
    first = setup.controller.submit("train")
    directory = setup.root / "orchestration/jobs" / first["key"]
    intent = orchestration.read_json(directory / "INTENT.json")
    fields = setup.slurm.rows.pop(first["job_id"])
    terminal = dict(
        job_id=first["job_id"],
        state="TIMEOUT",
        exit_code="0:0",
        user="alice",
        account="rose",
        qos=fields["QOS"],
        name=fields["JobName"],
    )
    setup.slurm.accounting[first["job_id"]] = terminal
    setup.controller.status()
    checkpoint = setup.root / "resume-fixture.json"
    checkpoint.write_text("synthetic immutable checkpoint proof")

    def verifier(root, task, previous):
        assert task == intent["task_ids"][0] and previous == first["key"]
        return dict(
            status="PASS",
            task_id=task,
            previous_job_key=previous,
            previous_job_id=first["job_id"],
            run_identity_sha256=setup.module._sha(root / "RUN_IDENTITY.json"),
            resume_kind="checkpoint",
            resume_state_sha256="c" * 64,
            execution_freeze_sha256=setup.module._sha(root / "EXECUTION_FREEZE.json"),
            artifacts=[orchestration.artifact_record(root, checkpoint.name)],
        )

    second = setup.controller.resume(first["key"], verifier=verifier)
    resumed = orchestration.read_json(
        setup.root / "orchestration/jobs" / second["key"] / "INTENT.json"
    )
    assert resumed["resume_of"] == first["key"]
    assert resumed["worker_entry_commands"] == intent["worker_entry_commands"]
    assert resumed["amendment_sha256"] == intent["amendment_sha256"]
    assert setup.controller.resume(first["key"], verifier=verifier) == second
    assert (directory / "TERMINAL.json").exists()


def test_terminal_checks_still_reject_wrong_account(setup, monkeypatch):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    complete_new(setup, state, monkeypatch)
    directory = setup.root / "orchestration/jobs" / state["key"]
    terminal = orchestration.read_json(directory / "TERMINAL.json")
    terminal["account"] = "other"
    orchestration.save_json(directory / "TERMINAL.json", terminal)
    with pytest.raises(PermissionError, match="accounting identity"):
        setup.controller.completed()


def test_existing_science_or_wrong_typed_preparer_blocks_prospective_amendment(setup):
    bad = dict(setup.typed_verified, auxiliary_code_commit="c" * 40)
    with pytest.raises(PermissionError, match="existing independently verified"):
        setup.module.seal_amendment(setup.root, setup.manifest, typed_verified=bad)
    (setup.root / "runs").mkdir()
    with pytest.raises(PermissionError, match="precede all science"):
        setup.module.seal_amendment(setup.root, setup.manifest, typed_verified=setup.typed_verified)


def test_process_cannot_switch_extension_root_or_accept_unbound_manifest(setup):
    changed = copy.deepcopy(orchestration.read_json(setup.manifest))
    changed["extension_commit"] = "c" * 40
    orchestration.save_json(setup.manifest, changed)
    with pytest.raises(PermissionError, match="explicit immutable deployment"):
        setup.module.install(setup.root, setup.manifest)


def test_science_rejects_ambiguous_freeze_extension_identity(setup):
    seal_and_freeze(setup)
    path = setup.root / "EXECUTION_FREEZE.json"
    freeze = orchestration.read_json(path)
    freeze["engineering_files"]["native/observability/" + "c" * 40 + "/SOURCE_MANIFEST.json"] = (
        "d" * 64
    )
    orchestration.save_json(path, freeze)
    with pytest.raises(PermissionError, match="exactly one"):
        setup.controller.submit("train")


@pytest.mark.parametrize("field", ["python", "code_root", "job_name", "cpus", "memory_gib", "qos"])
def test_resealed_intent_cannot_change_frozen_launcher_or_resource_fields(setup, field):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    directory = setup.root / "orchestration/jobs" / state["key"]
    intent = orchestration.read_json(directory / "INTENT.json")
    intent[field] = 999 if isinstance(intent[field], int) else "unregistered-value"
    orchestration.save_json(directory / "INTENT.json", intent)
    state["intent_sha256"] = setup.module._sha(directory / "INTENT.json")
    if field == "qos":
        state["qos"] = intent["qos"]
    orchestration.save_json(directory / "STATE.json", state)
    with pytest.raises(PermissionError, match="entry/source/amendment"):
        setup.controller.attempts()


def test_rehashed_batch_script_still_requires_the_actual_registered_command(setup):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    directory = setup.root / "orchestration/jobs" / state["key"]
    intent = orchestration.read_json(directory / "INTENT.json")
    Path(intent["script"]).write_text("#!/bin/bash\nexec unregistered-entry\n")
    intent["script_sha256"] = setup.module._sha(intent["script"])
    orchestration.save_json(directory / "INTENT.json", intent)
    state["intent_sha256"] = setup.module._sha(directory / "INTENT.json")
    orchestration.save_json(directory / "STATE.json", state)
    with pytest.raises(PermissionError, match="batch script"):
        setup.controller.attempts()


def test_unknown_or_unsealed_observation_is_not_a_completed_task(setup, monkeypatch):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    complete_new(setup, state, monkeypatch)
    intent = orchestration.read_json(
        setup.root / "orchestration/jobs" / state["key"] / "INTENT.json"
    )
    final_path = (
        setup.root
        / "accounting/observability"
        / state["key"]
        / intent["task_ids"][0]
        / "FINAL.json"
    )
    final = orchestration.read_json(final_path)
    final["observer_snapshot"]["sealed"] = False
    orchestration.save_json(final_path, final)
    with pytest.raises(PermissionError, match="final snapshot"):
        setup.controller.completed()


@pytest.mark.parametrize("missing_ledger", [False, True])
def test_worker_failure_seals_and_preserves_snapshot_without_retrying_main(
    setup, monkeypatch, missing_ledger
):
    seal_and_freeze(setup)
    state = setup.controller.submit("train")
    intent = orchestration.read_json(
        setup.root / "orchestration/jobs" / state["key"] / "INTENT.json"
    )
    task = intent["task_ids"][0]
    for key, value in dict(
        SLURM_JOB_ID=state["job_id"],
        SRF3CE_JOB_KEY=state["key"],
        SRF3CE_CLAIM_ID=intent["claim_id"],
    ).items():
        monkeypatch.setenv(key, value)
    calls = []
    snapshot = dict(sealed=True, active_scopes=["unfinished"], write_errors=[])
    observer = SimpleNamespace(
        install=lambda *args: {"installation_id": "fixture"},
        seal=lambda cls: calls.append("seal") or snapshot,
    )
    monkeypatch.setattr(setup.module, "_load_observer", lambda root: observer)
    original_import = setup.module.importlib.import_module
    monkeypatch.setattr(
        setup.module.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(SRF3CERuntime=type("Toy", (), {}))
            if name == "sr_f3_ce.runtime"
            else original_import(name)
        ),
    )

    def main(args):
        calls.append("main")
        if missing_ledger:
            (
                setup.root / "accounting/observability" / state["key"] / task / "events.jsonl"
            ).unlink()
        raise KeyboardInterrupt("synthetic scientific interruption")

    monkeypatch.setattr(setup.module, "_load_source", lambda *args: SimpleNamespace(main=main))
    with pytest.raises(KeyboardInterrupt, match="scientific interruption"):
        setup.module.run_worker(setup.root, task)
    directory = setup.root / "accounting/observability" / state["key"] / task
    assert calls == ["main", "seal"]
    failure = orchestration.read_json(directory / "FAILURE.json")
    assert failure["observer_snapshot"] == snapshot
    if missing_ledger:
        assert (
            failure["events_sha256"] == "UNKNOWN"
            and "FileNotFoundError" in failure["events_hash_error"]
        )
    assert not (directory / "FINAL.json").exists()


def test_observer_loader_never_executes_unchecked_hash_pyc(setup, tmp_path):
    seal_and_freeze(setup)
    source = setup.manifest.with_name("observe_runtime.py")
    marker = tmp_path / "unregistered-cache-executed"
    payload = tmp_path / "cache_payload.py"
    payload.write_text(f"open({str(marker)!r}, 'w').write('unregistered')\n")
    cached = Path(importlib.util.cache_from_source(str(source)))
    cached.parent.mkdir()
    py_compile.compile(
        str(payload),
        cfile=str(cached),
        dfile=str(source),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    observer = setup.module._load_observer(setup.root)
    assert callable(observer.install) and not marker.exists()


def test_observer_rejects_source_changed_after_static_verification(setup, monkeypatch, tmp_path):
    seal_and_freeze(setup)
    source = setup.manifest.with_name("observe_runtime.py")
    marker = tmp_path / "unregistered-source-executed"
    original = setup.module.verify_extension

    def verify(*args, **kwargs):
        checked = original(*args, **kwargs)
        source.write_text(f"open({str(marker)!r}, 'w').write('unregistered')\n")
        return checked

    monkeypatch.setattr(setup.module, "verify_extension", verify)
    with pytest.raises(PermissionError, match="verified-byte import"):
        setup.module._load_observer(setup.root)
    assert not marker.exists()


def test_source_loader_executes_only_the_bytes_actually_hashed(setup, monkeypatch, tmp_path):
    source = tmp_path / "captured_source.py"
    source.write_text("VALUE = 'verified'\n")
    record = dict(bytes=source.stat().st_size, sha256=setup.module._sha(source))
    original = Path.read_bytes

    def read(path):
        body = original(path)
        if path == source:
            source.write_text("raise AssertionError('unverified second read')\n")
        return body

    monkeypatch.setattr(Path, "read_bytes", read)
    loaded = setup.module._load_source(source, record, "captured_verified_fixture")
    assert loaded.VALUE == "verified"


def test_existing_scientific_package_bytecode_is_rejected_before_import(setup):
    cached = setup.code / "src/__pycache__/fake_runtime.cpython-312.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"unbound cache fixture")
    with pytest.raises(PermissionError, match="scientific bytecode"):
        setup.module.verify_extension(setup.root, setup.manifest)
