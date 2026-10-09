from types import SimpleNamespace

import pytest

from sr_f1 import engine, freeze
from sr_f1.runtime import atomic_json, file_hash, read_json


@pytest.fixture
def recovery_case(tmp_path, monkeypatch):
    relative = "engineering/engine/natural/continuous"
    staged = tmp_path / relative
    checkpoint = staged / "checkpoints/step-00-original.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"authenticated-full-state-zero-update-fixture")
    receipt = dict(
        step=0,
        path=checkpoint.name,
        sha256=file_hash(checkpoint),
        state_hash="a" * 64,
        field_hashes={"validated_by_archived_cpu_state_review": "b" * 64},
    )
    atomic_json(staged / "checkpoints/LATEST.json", receipt)
    atomic_json(staged / "checkpoints/commit-00.json", receipt)
    atomic_json(staged / "RUN_MANIFEST.json", {"immutable": "original"})
    for slot in range(16):
        for index in range(8):
            atomic_json(
                staged / f"rollouts/01-{slot:02d}-{index}.json",
                {"slot": slot, "sample_index": index, "original": True},
            )
    recovery = dict(
        staged_track_relative_path=relative,
        archived_track_relative_path=(
            "technical_incidents/engine_memory_20261010/evidence/" + relative
        ),
        original_checkpoint_state_hash="a" * 64,
        original_raw_count=128,
        engine_mode="natural",
        resume_step=0,
        artifact_hashes={
            str(p.relative_to(tmp_path)): file_hash(p) for p in staged.rglob("*") if p.is_file()
        },
    )
    atomic_json(tmp_path / "ENGINE_MEMORY_REPAIR.json", {"zero_update_recovery": recovery})
    # Historical full-state/source validation is independently tested in
    # test_engine_memory_freeze.py; this suite exercises live activation only.
    monkeypatch.setattr(
        freeze,
        "verify_engine_memory_repair",
        lambda root: {"repair_sha256": file_hash(root / "ENGINE_MEMORY_REPAIR.json")},
    )
    return tmp_path, staged, recovery


def child_of_activation(root, monkeypatch):
    activation, _ = engine._zero_update_recovery_markers(root)
    parent = read_json(activation)["activating_pid"]
    monkeypatch.setattr(engine.os, "getppid", lambda: parent)
    monkeypatch.setattr(engine.os, "getpid", lambda: parent + 100)


def test_step_zero_reuse_is_consumed_before_the_first_new_process(recovery_case, monkeypatch):
    root, staged, recovery = recovery_case
    before = dict(recovery["artifact_hashes"])
    assert engine._preserve_interrupted_mode(root, "natural") is None
    activation, process = engine._zero_update_recovery_markers(root)
    assert activation.is_file() and not process.exists()
    assert not list(staged.glob("PROCESS-*.json"))
    assert all(file_hash(root / name) == expected for name, expected in before.items())
    child_of_activation(root, monkeypatch)
    claim = engine._claim_zero_update_recovery_process(root, "natural", "continuous")
    assert read_json(process) == claim
    assert claim["activation_sha256"] == file_hash(activation)
    assert claim["resume_step"] == 0
    with pytest.raises(PermissionError, match="already consumed"):
        engine._claim_zero_update_recovery_process(root, "natural", "continuous")


@pytest.mark.parametrize("child_started", [False, True])
def test_second_wrapper_archives_even_if_the_first_child_never_started(
    recovery_case, monkeypatch, child_started
):
    root, staged, _ = recovery_case
    engine._preserve_interrupted_mode(root, "natural")
    if child_started:
        child_of_activation(root, monkeypatch)
        engine._claim_zero_update_recovery_process(root, "natural", "continuous")
        atomic_json(staged / "PROCESS-failed.json", {"pid": engine.os.getpid()})
    activation, process = engine._zero_update_recovery_markers(root)
    activation_hash = file_hash(activation)
    archived = engine._preserve_interrupted_mode(root, "natural")
    assert not staged.exists()
    assert (archived / "continuous/checkpoints/LATEST.json").exists()
    assert len(list((archived / "continuous/rollouts").glob("*.json"))) == 128
    assert (archived / "INTERRUPTION_PRESERVED.json").exists()
    assert file_hash(activation) == activation_hash
    assert engine._preserve_interrupted_mode(root, "natural") is None
    assert engine._claim_zero_update_recovery_process(root, "natural", "continuous") is None
    assert process.exists() == child_started


@pytest.mark.parametrize(
    "change",
    ["extra_process", "two_processes", "extra_raw", "missing_raw", "changed_raw", "missing_latest"],
)
def test_activation_rejects_any_difference_in_the_complete_staged_inventory(recovery_case, change):
    root, staged, _ = recovery_case
    if change in {"extra_process", "two_processes"}:
        atomic_json(staged / "PROCESS-one.json", {"pid": 1})
        if change == "two_processes":
            atomic_json(staged / "PROCESS-two.json", {"pid": 2})
    elif change == "extra_raw":
        atomic_json(staged / "rollouts/02-00-0.json", {"unexpected": True})
    elif change == "missing_raw":
        (staged / "rollouts/01-00-0.json").unlink()
    elif change == "changed_raw":
        atomic_json(staged / "rollouts/01-00-0.json", {"changed": True})
    else:
        (staged / "checkpoints/LATEST.json").unlink()
    with pytest.raises((PermissionError, FileNotFoundError)):
        engine._preserve_interrupted_mode(root, "natural")
    assert staged.exists()
    assert not engine._zero_update_recovery_markers(root)[0].exists()
    assert not (root / "engineering/engine_interrupted").exists()


@pytest.mark.parametrize("change", ["latest_step", "committed_step", "state_hash", "another_track"])
def test_recovery_cannot_reuse_an_updated_or_conflicting_boundary(recovery_case, change):
    root, staged, _ = recovery_case
    if change == "another_track":
        (staged.parent / "split").mkdir()
    else:
        target = staged / (
            "checkpoints/commit-00.json"
            if change == "committed_step"
            else "checkpoints/LATEST.json"
        )
        receipt = read_json(target)
        receipt["state_hash" if change == "state_hash" else "step"] = (
            "changed" if (change == "state_hash") else 1
        )
        atomic_json(target, receipt)
    with pytest.raises(PermissionError):
        engine._preserve_interrupted_mode(root, "natural")
    assert not engine._zero_update_recovery_markers(root)[0].exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("resume_step", 1),
        ("original_raw_count", 129),
        ("engine_mode", "stress"),
        ("staged_track_relative_path", "../escape"),
        ("archived_track_relative_path", "unverified_history"),
    ],
)
def test_recovery_authorization_must_have_the_exact_registered_boundary(recovery_case, key, value):
    root, _, recovery = recovery_case
    atomic_json(
        root / "ENGINE_MEMORY_REPAIR.json", {"zero_update_recovery": {**recovery, key: value}}
    )
    with pytest.raises(PermissionError, match="boundary differs"):
        engine._preserve_interrupted_mode(root, "natural")


def test_permanent_history_authentication_precedes_activation(recovery_case, monkeypatch):
    root, staged, _ = recovery_case

    def invalid_history(root):
        raise PermissionError("historical evidence changed")

    monkeypatch.setattr(freeze, "verify_engine_memory_repair", invalid_history)
    with pytest.raises(PermissionError, match="historical evidence changed"):
        engine._preserve_interrupted_mode(root, "natural")
    assert staged.exists()
    assert not engine._zero_update_recovery_markers(root)[0].exists()


@pytest.mark.parametrize("change", ["no_activation", "wrong_parent", "changed_after_activation"])
def test_first_child_revalidates_before_loading_the_model(recovery_case, monkeypatch, change):
    root, staged, _ = recovery_case
    if change != "no_activation":
        engine._preserve_interrupted_mode(root, "natural")
        child_of_activation(root, monkeypatch)
        if change == "wrong_parent":
            monkeypatch.setattr(engine.os, "getppid", lambda: -1)
        else:
            atomic_json(staged / "rollouts/01-00-0.json", {"changed": True})

    def forbidden_load(*args, **kwargs):
        pytest.fail("Model loading must follow authenticated once-only activation")

    monkeypatch.setattr(engine, "load_runtime", forbidden_load)
    with pytest.raises(PermissionError):
        engine._run_segment({}, root, mode="natural", segment="continuous", boundary=None)
    assert not engine._zero_update_recovery_markers(root)[1].exists()


def test_continuous_completion_retains_the_claim_identity(recovery_case, monkeypatch):
    root, staged, _ = recovery_case
    engine._preserve_interrupted_mode(root, "natural")
    child_of_activation(root, monkeypatch)
    runtime = SimpleNamespace(identity={"hardware": {"gpu": "fixture"}, "determinism": {}})
    monkeypatch.setattr(engine, "load_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(engine, "runtime_account", lambda *args: None)
    monkeypatch.setattr(engine, "engine_schedule", lambda root: [])
    monkeypatch.setattr(engine, "load_inputs", lambda root: {})
    monkeypatch.setattr(engine, "load_tasks", lambda root: {})
    monkeypatch.setattr(engine, "execute_path", lambda *args, **kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr(engine, "verified_adapter", lambda *args: {"trainable_state_hash": "hash"})
    engine._run_segment({}, root, mode="natural", segment="continuous", boundary=None)
    completed = read_json(staged / "CONTINUOUS_COMPLETE.json")
    claim = read_json(engine._zero_update_recovery_markers(root)[1])
    assert completed["zero_update_recovery"] == claim
    assert completed["pid"] == claim["pid"]
    assert engine._claim_zero_update_recovery_process(root, "natural", "first") is None
    assert engine._claim_zero_update_recovery_process(root, "stress", "continuous") is None
