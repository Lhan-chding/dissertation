"""Regression coverage for explicit, source-bound R4 diagnostic continuation."""

import copy
import json
from pathlib import Path

import pytest

from src import r4_runtime
from src.core import canonical_hash, file_hash


def _prompt():
    return {
        "base_scene_id": "scene",
        "prompt_id": "prompt",
        "family": "trend",
        "interface": "SYMBOLIC_FRESH",
        "split": "train",
        "track": "N",
        "prompt_hash": "prompt-hash",
        "scene": {"truth": [1, 2, 3, 4]},
        "max_new_tokens": 64,
    }


def test_execution_source_migration_preserves_logical_requests_and_seed():
    logical = {
        "phase": "R4",
        "model_hash": "m",
        "data_hash": "d",
        "config_hash": "c",
        "source_hash": "old",
    }
    execution = {**logical, "source_hash": "new", "continuation_hash": "review"}
    expected = r4_runtime.build_requests([_prompt()], logical, "X_BASE", 35, "train", "state")
    actual = r4_runtime.build_requests(
        [_prompt()], execution, "X_BASE", 35, "train", "state", sampling_identity=logical
    )
    assert actual == expected
    assert execution["source_hash"] == "new"
    with pytest.raises(ValueError, match="sampling identity"):
        r4_runtime.build_requests(
            [_prompt()],
            {**execution, "data_hash": "other"},
            "X_BASE",
            35,
            "train",
            "state",
            sampling_identity=logical,
        )


def test_reviewed_warning_does_not_waive_mean_kl_or_nonfinite_values():
    from src.r4_continuation import REVIEWED_POLICY

    context = {"continuation_hash": "review", "decision": {"policy": REVIEWED_POLICY}}
    diagnostic = {
        "status": "STOP_DIAGNOSE",
        "should_stop": True,
        "mean_token_kl": 0.02,
        "sequence_log_ratio_p99_abs": 2.1,
        "alarms": {"mean_token_kl": False, "sequence_log_ratio_p99_abs": True},
        "thresholds": {
            "mean_token_kl": 0.1,
            "sequence_log_ratio_p99_abs": 2.0,
            "comparison": "strict_greater_than",
        },
    }
    before = copy.deepcopy(diagnostic)
    assert r4_runtime._reviewed_sequence_warning(diagnostic, context)
    assert diagnostic == before
    assert not r4_runtime._reviewed_sequence_warning(diagnostic, None)
    for mean in (0.10001, float("nan"), float("inf")):
        assert not r4_runtime._reviewed_sequence_warning(
            {**diagnostic, "mean_token_kl": mean}, context
        )
    assert not r4_runtime._reviewed_sequence_warning(
        {**diagnostic, "sequence_log_ratio_p99_abs": float("inf")}, context
    )
    assert not r4_runtime._reviewed_sequence_warning(
        {
            **diagnostic,
            "thresholds": {**diagnostic["thresholds"], "sequence_log_ratio_p99_abs": 3.0},
        },
        context,
    )


def _stopped_fixture(tmp_path, monkeypatch, *, stop_step=1, stop_arm="X_BASE"):
    from test_r4_runtime import _fixture

    from src import r4_gate, r4_metrics
    from src.r4_continuation import REVIEWED_POLICY

    args, adapter, instances = _fixture(tmp_path, monkeypatch)
    original_diagnostic = r4_metrics.control_kl_diagnostic
    diagnostic_calls = []
    stopped_updates = stop_step + (64 if stop_arm == "X_VALID" else 0)

    def warning(*a, **k):
        measured = original_diagnostic(*a, **k)
        diagnostic_calls.append(measured)
        if len(diagnostic_calls) < stopped_updates:
            return measured
        return {
            **measured,
            "status": "STOP_DIAGNOSE",
            "should_stop": True,
            "mean_token_kl": 0.02,
            "sequence_log_ratio_p99_abs": 2.1,
            "alarms": {"mean_token_kl": False, "sequence_log_ratio_p99_abs": True},
        }

    monkeypatch.setattr(r4_metrics, "control_kl_diagnostic", warning)
    result = r4_runtime.run_r4(*args, allow_training=True, _adapter_factory=adapter)
    assert result["status"] == "BLOCKED", result["details"]
    assert (
        result["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == stopped_updates
    )
    parent = args[2]
    runtime = json.loads((parent / "runtime_lock.json").read_text())
    checkpoints = json.loads((parent / "checkpoint_manifest.json").read_text())
    alarm = json.loads((parent / "alarm_stop.json").read_text())
    raw = _raw_rows(parent)
    binding = {
        "status": "PASS_STOPPED_AUDIT",
        "root": str(parent.resolve()),
        "files": {
            p.relative_to(parent).as_posix(): file_hash(p) for p in parent.rglob("*") if p.is_file()
        },
        "manifest_sha256": file_hash(parent / "manifest.json"),
        "runtime_lock_sha256": file_hash(parent / "runtime_lock.json"),
        "alarm_stop_sha256": file_hash(parent / "alarm_stop.json"),
        "source": runtime["source"],
        "identity": runtime["identity"],
        "stop": {
            "arm": stop_arm,
            "step": stop_step,
            "checkpoint": next(
                c
                for c in checkpoints["checkpoints"]
                if c["arm"] == stop_arm and c["step"] == stop_step
            ),
            "diagnostic": alarm["control_diagnostic"],
        },
        "checkpoint_manifest": checkpoints,
        "inherited_counts": {
            "optimizer_updates": stopped_updates,
            "training_rollouts": 32 * stopped_updates,
            "shared_step0_outputs": 576,
            "step32_outputs": sum(
                r["bank_role"] == "evaluation" and r["checkpoint_step"] == 32 for r in raw
            ),
            "step64_outputs": sum(
                r["bank_role"] == "evaluation" and r["checkpoint_step"] == 64 for r in raw
            ),
            "new_outputs": len(raw),
        },
    }
    binding["audit_hash"] = canonical_hash({k: v for k, v in binding.items() if k != "root"})

    def audited_snapshot(root, *unused, _recorded_root=None):
        root = Path(root)
        assert root == parent or root.name == "inherited_parent"
        assert _recorded_root in (None, str(parent.resolve()))
        # The tiny adapter is not CUDA evidence. Its mock audit nevertheless
        # verifies all actual bytes on each call; preparation remains unmocked.
        for name, digest in binding["files"].items():
            assert file_hash(root / name) == digest
        return copy.deepcopy(binding)

    monkeypatch.setattr(r4_gate, "audit_stopped_r4", audited_snapshot)
    new_source = {"source_commit": "CPU-FIXTURE-CONTINUATION"}
    monkeypatch.setattr(r4_runtime, "_source", lambda: copy.deepcopy(new_source))
    decision = {
        "schema_version": 1,
        "authorization": {
            "user_request": "Analyze, fix and continue training according to the plan.",
            "reason": "A reviewed sequence-only stop; all hard stops remain active.",
        },
        "parent_audit_hash": binding["audit_hash"],
        "execution_source_hash": canonical_hash(new_source),
        "policy": copy.deepcopy(REVIEWED_POLICY),
    }
    child_args = (*args[:2], tmp_path / "continued", *args[3:])
    return child_args, adapter, instances, parent, binding, decision, original_diagnostic


def _raw_rows(root):
    return [
        row
        for path in root.rglob("samples.jsonl")
        for line in path.read_text().splitlines()
        if "raw_text" in (row := json.loads(line))
    ]


def test_second_reviewed_recovery_preserves_ancestor_state_and_budget(tmp_path, monkeypatch):
    from src import r4_gate, r4_metrics
    from src.optimizer_fork import state_hash
    from src.r4_continuation import REVIEWED_CUMULATIVE_POLICY

    args, adapter, instances, ancestor, binding, decision, original = _stopped_fixture(
        tmp_path, monkeypatch, stop_step=35
    )
    diagnostic_calls = 0

    def mean_stop(*a, **k):
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        mean = 0.02 if diagnostic_calls < 16 else 0.11
        return {
            **original(*a, **k),
            "status": "STOP_DIAGNOSE",
            "should_stop": True,
            "mean_token_kl": mean,
            "sequence_log_ratio_p99_abs": 3.0,
            "alarms": {"mean_token_kl": mean > 0.1, "sequence_log_ratio_p99_abs": True},
        }

    monkeypatch.setattr(r4_metrics, "control_kl_diagnostic", mean_stop)
    stopped = r4_runtime.run_r4(
        *args,
        allow_training=True,
        continuation_parent=ancestor,
        continuation_decision=decision,
        _adapter_factory=adapter,
    )
    assert stopped["status"] == "BLOCKED", stopped["details"]
    parent = args[2]
    runtime = json.loads((parent / "runtime_lock.json").read_text())
    manifest = json.loads((parent / "checkpoint_manifest.json").read_text())
    alarm = json.loads((parent / "alarm_stop.json").read_text())
    stop_checkpoint = next(c for c in manifest["checkpoints"] if c["step"] == 51)
    raw = _raw_rows(parent)
    assert len(raw) == 2784
    parent_binding = {
        **binding,
        "root": str(parent.resolve()),
        "files": {
            p.relative_to(parent).as_posix(): file_hash(p) for p in parent.rglob("*") if p.is_file()
        },
        "manifest_sha256": file_hash(parent / "manifest.json"),
        "runtime_lock_sha256": file_hash(parent / "runtime_lock.json"),
        "alarm_stop_sha256": file_hash(parent / "alarm_stop.json"),
        "source": runtime["source"],
        "identity": runtime["identity"],
        "logical_sampling_identity": binding["identity"],
        "continuation": runtime["continuation"],
        "stop": {
            "arm": "X_BASE",
            "step": 51,
            "checkpoint": stop_checkpoint,
            "diagnostic": alarm["control_diagnostic"],
        },
        "checkpoint_manifest": manifest,
        "inherited_counts": {
            **binding["inherited_counts"],
            "optimizer_updates": 51,
            "training_rollouts": 1632,
            "new_outputs": len(raw),
        },
    }
    parent_binding["audit_hash"] = canonical_hash(
        {k: v for k, v in parent_binding.items() if k not in ("root", "audit_hash")}
    )

    def audit(root, *unused, _recorded_root=None):
        root = Path(root)
        actual = (
            parent_binding
            if file_hash(root / "runtime_lock.json") == parent_binding["runtime_lock_sha256"]
            else binding
        )
        for name, digest in actual["files"].items():
            assert file_hash(root / name) == digest
        return copy.deepcopy(actual)

    monkeypatch.setattr(r4_gate, "audit_stopped_r4", audit)
    source = {"source_commit": "CPU-FIXTURE-SECOND-CONTINUATION"}
    monkeypatch.setattr(r4_runtime, "_source", lambda: copy.deepcopy(source))
    decision2 = {
        **decision,
        "schema_version": 2,
        "parent_audit_hash": parent_binding["audit_hash"],
        "execution_source_hash": canonical_hash(source),
        "policy": copy.deepcopy(REVIEWED_CUMULATIVE_POLICY),
    }
    calls = []
    original_step = r4_runtime._step

    def observe(*a, **k):
        calls.append((a[9], a[10], state_hash(a[2])))
        return original_step(*a, **k)

    monkeypatch.setattr(r4_runtime, "_step", observe)
    out = tmp_path / "second_continuation"
    final_args = (*args[:2], out, *args[3:])
    completed = r4_runtime.run_r4(
        *final_args,
        allow_training=True,
        continuation_parent=parent,
        continuation_decision=decision2,
        _adapter_factory=adapter,
    )
    assert completed["status"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS", completed["details"]
    assert calls[0] == ("X_BASE", 52, stop_checkpoint["state_hash"])
    assert len(calls) == 77
    assert sum(a.generation_calls for a in instances) == 14400
    assert instances[-1].generation_calls == 14400 - len(raw)
    assert completed["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 128
    assert completed["details"]["runtime_counts"]["backward_calls_observed_at_least"] == 4096
    history = completed["details"]["diagnostic_stop_history"]
    assert [(s["step"], s["status"]) for s in history] == [
        (35, "DIAGNOSTIC_STOP"),
        (51, "DIAGNOSTIC_STOP"),
    ]
    all_rows = _raw_rows(out)
    assert len(all_rows) == len({r["sample_key"] for r in all_rows}) == 14400
    assert all(r["run_id"] == canonical_hash(binding["identity"]) for r in all_rows)
    assert json.loads((out / "logical_sampling_identity.json").read_text()) == binding["identity"]
    for name, digest in parent_binding["files"].items():
        assert file_hash(parent / name) == digest
    before = sum(a.generation_calls for a in instances)
    resumed = r4_runtime.run_r4(
        *final_args,
        allow_training=True,
        resume=True,
        _adapter_factory=adapter,
    )
    assert resumed["status"] == completed["status"], resumed["details"]
    assert sum(a.generation_calls for a in instances) == before
    cost = json.loads((out / "runtime_profile.json").read_text())
    assert cost["optimizer_steps_observed_at_least"] == 128
    assert cost["invocations"][-1]["observed"]["optimizer_step_calls_observed"] == 0
    assert cost["invocations"][-1]["observed"]["backward_calls_observed"] == 0


def test_real_torch_stop_continuation_and_completed_resume_preserve_all_state(
    tmp_path, monkeypatch
):
    from src.optimizer_fork import load_checkpoint, state_hash

    args, adapter, instances, parent, binding, decision, _ = _stopped_fixture(tmp_path, monkeypatch)
    preserved = {
        p.relative_to(parent).as_posix(): (file_hash(p), p.stat().st_mtime_ns)
        for p in parent.rglob("*")
        if p.is_file()
    }
    stop = binding["stop"]["checkpoint"]
    stopped_state = load_checkpoint(stop["checkpoint_path"], stop["checkpoint_identity"])
    assert len(stopped_state["metadata"]["completed_sample_keys"]) == 32
    assert stopped_state["optimizer"]["state"]
    assert all(int(state["step"]) == 1 for state in stopped_state["optimizer"]["state"].values())
    calls = []
    original_step = r4_runtime._step

    def observed_step(*positional, **kwargs):
        prestate, arm, step = positional[2], positional[9], positional[10]
        calls.append((arm, step, state_hash(prestate)))
        return original_step(*positional, **kwargs)

    monkeypatch.setattr(r4_runtime, "_step", observed_step)
    completed = r4_runtime.run_r4(
        *args,
        allow_training=True,
        continuation_parent=parent,
        continuation_decision=decision,
        _adapter_factory=adapter,
    )
    assert completed["status"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS", completed["details"]
    out = args[2]
    assert len(calls) == 127
    assert calls[0] == ("X_BASE", 2, state_hash(stopped_state))
    assert ("X_BASE", 1) not in [(arm, step) for arm, step, _ in calls]
    assert not (out / "X_BASE/step_01").exists()
    assert not (out / "alarm_stop.json").exists()
    assert (out / "inherited_parent/alarm_stop.json").exists()
    assert sum(a.generation_calls for a in instances) == 14400
    assert instances[-1].generation_calls == 14400 - 608
    assert completed["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 128
    cost = json.loads((out / "runtime_profile.json").read_text())
    assert sum(i["observed"]["optimizer_step_calls_observed"] for i in cost["invocations"]) == 127
    rows = _raw_rows(out)
    assert len(rows) == len({r["sample_key"] for r in rows}) == 14400
    inherited = _raw_rows(out / "inherited_parent")
    inherited_keys = {r["sample_key"] for r in inherited}
    assert len(inherited) == 608
    logical = binding["identity"]
    identity = json.loads((out / "identity.json").read_text())
    assert identity["source_hash"] != logical["source_hash"]
    assert json.loads((out / "logical_sampling_identity.json").read_text()) == logical
    current = [r for r in rows if r["sample_key"] not in inherited_keys]
    assert all(r["run_id"] == canonical_hash(logical) for r in rows)
    assert all(r["execution_identity_hash"] == canonical_hash(identity) for r in current)
    assert all("execution_identity_hash" not in r for r in inherited)
    plan = json.loads((out / "two_arm_training_config.json").read_text())["plan"]
    prompts = {p["prompt_id"]: p for p in plan["train_prompts"]}
    second = [
        r
        for r in current
        if r["arm"] == "X_BASE" and r["checkpoint_step"] == 1 and r["bank_role"] == "train"
    ]
    expected = r4_runtime.build_requests(
        [prompts[p] for p in plan["train_steps"][1]],
        logical,
        "X_BASE",
        1,
        "train",
        state_hash(stopped_state),
    )
    assert {r["sample_key"]: r["sample_seed"] for r in second} == {
        r["sample_key"]: r["sample_seed"] for r in expected
    }
    checkpoints = json.loads((out / "checkpoint_manifest.json").read_text())["checkpoints"]
    assert len(checkpoints) == 130
    assert sum(c["source_segment"] == "parent" for c in checkpoints) == 3
    for arm in ("X_BASE", "X_VALID"):
        last = next(c for c in checkpoints if c["arm"] == arm and c["step"] == 64)
        state = load_checkpoint(out / last["checkpoint_path"], last["checkpoint_identity"])
        assert state_hash(state) == last["state_hash"]
        assert len(state["metadata"]["completed_sample_keys"]) == 2048
        assert len(set(state["metadata"]["completed_sample_keys"])) == 2048
        assert all(int(item["step"]) == 64 for item in state["optimizer"]["state"].values())
        if arm == "X_BASE":
            assert (
                state["metadata"]["completed_sample_keys"][:32]
                == stopped_state["metadata"]["completed_sample_keys"]
            )
    warnings = json.loads((out / "diagnostic_warnings.json").read_text())
    assert len(warnings) == 128
    assert {w["arm"] for w in warnings} == {"X_BASE", "X_VALID"}
    assert warnings[0]["status"] == "DIAGNOSTIC_STOP"
    assert all(w["status"] == "DIAGNOSTIC_WARNING" for w in warnings[1:])
    assert all(w["control_diagnostic"]["should_stop"] for w in warnings)
    assert all(
        w["control_diagnostic"]["thresholds"] == decision["policy"]["thresholds"] for w in warnings
    )
    assert preserved == {
        p.relative_to(parent).as_posix(): (file_hash(p), p.stat().st_mtime_ns)
        for p in parent.rglob("*")
        if p.is_file()
    }
    generations = sum(a.generation_calls for a in instances)
    resumed = r4_runtime.run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    assert resumed["status"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS", resumed["details"]
    assert sum(a.generation_calls for a in instances) == generations
    assert instances[-1].forward_calls == 0
    assert resumed["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 128
    assert file_hash(out / "inherited_parent/alarm_stop.json") == preserved["alarm_stop.json"][0]


@pytest.mark.parametrize("fault", ["mean", "nonfinite"])
def test_real_torch_reviewed_continuation_still_hard_stops(tmp_path, monkeypatch, fault):
    from src import r4_metrics

    args, adapter, instances, parent, _, decision, diagnostic = _stopped_fixture(
        tmp_path, monkeypatch
    )

    def hard_fault(*a, **k):
        if fault == "nonfinite":
            raise FloatingPointError("Nonfinite KL or sequence log-ratio")
        return {
            **diagnostic(*a, **k),
            "status": "STOP_DIAGNOSE",
            "should_stop": True,
            "mean_token_kl": 0.10001,
            "sequence_log_ratio_p99_abs": 2.1,
            "alarms": {"mean_token_kl": True, "sequence_log_ratio_p99_abs": True},
        }

    monkeypatch.setattr(r4_metrics, "control_kl_diagnostic", hard_fault)
    failed = r4_runtime.run_r4(
        *args,
        allow_training=True,
        continuation_parent=parent,
        continuation_decision=decision,
        _adapter_factory=adapter,
    )
    assert failed["status"] == ("BLOCKED" if fault == "mean" else "FAIL"), failed["details"]
    out = args[2]
    assert not (out / "X_BASE/step_03").exists()
    assert not (out / "X_VALID").exists()
    assert (out / "X_BASE/step_02/updates/attempt_0000/checkpoint.pt").exists()
    assert (out / ("alarm_stop.json" if fault == "mean" else "measurement_fault.json")).exists()
    before = sum(a.generation_calls for a in instances)
    if fault == "nonfinite":
        with pytest.raises(r4_runtime.PilotBlocked, match="failed measurement"):
            r4_runtime.run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    else:
        again = r4_runtime.run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
        assert again["status"] == "BLOCKED"
    assert sum(a.generation_calls for a in instances) == before


@pytest.mark.parametrize("path_kind", ["absolute", "alias", "relative"])
def test_stop_at_step32_generates_missing_panel_before_next_update(
    tmp_path, monkeypatch, path_kind
):
    args, adapter, instances, parent, _, decision, _ = _stopped_fixture(
        tmp_path, monkeypatch, stop_step=32
    )
    assert not (parent / "evaluation/X_BASE/step_32/N").exists()
    if path_kind == "alias":
        alias = tmp_path / "storage-alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        args = (*args[:2], alias / "continued", *args[3:])
    elif path_kind == "relative":
        monkeypatch.chdir(tmp_path)
        args = (*args[:2], Path("continued"), *args[3:])

    def stop_before_update(*positional, **kwargs):
        assert positional[9:11] == ("X_BASE", 33)
        raise RuntimeError("test interruption before step33 update")

    monkeypatch.setattr(r4_runtime, "_step", stop_before_update)
    result = r4_runtime.run_r4(
        *args,
        allow_training=True,
        continuation_parent=parent,
        continuation_decision=decision,
        _adapter_factory=adapter,
    )
    assert result["status"] == "FAIL", result["details"]
    assert "test interruption before step33" in result["details"]["error"]["message"]
    out = args[2]
    panel = _raw_rows(out / "evaluation/X_BASE/step_32/N")
    assert len(panel) == 576
    assert {r["checkpoint_step"] for r in panel} == {32}
    assert {r["arm"] for r in panel} == {"X_BASE"}
    assert not (out / "inherited_parent/evaluation/X_BASE/step_32/N").exists()
    assert instances[-1].generation_calls == 576 + 32
    assert result["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 32
    assert not (out / "X_BASE/step_32").exists()


def test_parent_stopped_in_valid_arm_inherits_completed_base_endpoints(tmp_path, monkeypatch):
    args, adapter, instances, parent, binding, decision, _ = _stopped_fixture(
        tmp_path, monkeypatch, stop_arm="X_VALID", stop_step=1
    )
    old_endpoints = _raw_rows(parent / "evaluation/X_BASE/step_64")
    assert len(old_endpoints) == 4288
    assert {r["track"] for r in old_endpoints} == {"N", "L", "OOD"}
    assert binding["inherited_counts"]["optimizer_updates"] == 65
    assert binding["inherited_counts"]["step64_outputs"] == 4288
    preserved = {
        p.relative_to(parent).as_posix(): (file_hash(p), p.stat().st_mtime_ns)
        for p in parent.rglob("*")
        if p.is_file()
    }
    next_updates = []

    def stop_before_valid_update(*positional, **kwargs):
        next_updates.append(positional[9:11])
        raise RuntimeError("test interruption before X_VALID step2 update")

    monkeypatch.setattr(r4_runtime, "_step", stop_before_valid_update)
    result = r4_runtime.run_r4(
        *args,
        allow_training=True,
        continuation_parent=parent,
        continuation_decision=decision,
        _adapter_factory=adapter,
    )
    assert result["status"] == "FAIL", result["details"]
    assert "test interruption before X_VALID step2" in result["details"]["error"]["message"]
    assert next_updates == [("X_VALID", 2)]
    out = args[2]
    assert not (out / "X_BASE").exists()
    assert not (out / "evaluation/X_BASE").exists()
    assert instances[-1].generation_calls == 32
    assert result["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 65
    cost = json.loads((out / "runtime_profile.json").read_text())
    assert sum(i["observed"]["optimizer_step_calls_observed"] for i in cost["invocations"]) == 0
    copied_endpoints = _raw_rows(out / "inherited_parent/evaluation/X_BASE/step_64")
    assert copied_endpoints == old_endpoints
    assert preserved == {
        p.relative_to(parent).as_posix(): (file_hash(p), p.stat().st_mtime_ns)
        for p in parent.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("resume_parent", ["explicit", "recorded"])
def test_interrupted_preparation_resumes_before_first_execution_identity(
    tmp_path, monkeypatch, resume_parent
):
    from src import r4_continuation

    args, adapter, instances, parent, _, decision, _ = _stopped_fixture(tmp_path, monkeypatch)
    original_copy = r4_continuation._copy_snapshot
    copy_calls = []

    def interrupted_copy(parent_root, destination, snapshot, inventory):
        copy_calls.append(parent_root)
        if len(copy_calls) == 1:
            copying = destination / ".inherited_parent.copying"
            copying.mkdir()
            relative = next(iter(inventory))
            target = copying / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((parent_root / relative).read_bytes())
            (destination / ".inherited_parent.file.part").write_bytes(b"unfinished transfer")
            raise RuntimeError("test interrupted immutable parent copy")
        return original_copy(parent_root, destination, snapshot, inventory)

    monkeypatch.setattr(r4_continuation, "_copy_snapshot", interrupted_copy)
    before = sum(a.generation_calls for a in instances)
    loaded = len(instances)
    with pytest.raises(RuntimeError, match="interrupted immutable parent copy"):
        r4_runtime.run_r4(
            *args,
            allow_training=True,
            continuation_parent=parent,
            continuation_decision=decision,
            _adapter_factory=adapter,
        )
    out = args[2]
    assert (out / "continuation_binding.json").exists()
    assert (out / "continuation_decision.json").exists()
    assert not (out / "identity.json").exists()
    assert not (out / "inherited_parent").exists()
    assert (out / ".inherited_parent.copying").exists()
    assert len(instances) == loaded
    assert sum(a.generation_calls for a in instances) == before
    actual_steps = []
    original_step = r4_runtime._step

    def stop_after_one_real_update(*positional, **kwargs):
        assert positional[9] == "X_BASE"
        if positional[10] == 3:
            raise RuntimeError("test interruption after resumed real step2")
        result = original_step(*positional, **kwargs)
        actual_steps.append(positional[10])
        return result

    monkeypatch.setattr(r4_runtime, "_step", stop_after_one_real_update)
    selection = (
        {"continuation_parent": parent, "continuation_decision": decision}
        if resume_parent == "explicit"
        else {}
    )
    resumed = r4_runtime.run_r4(
        *args, allow_training=True, resume=True, _adapter_factory=adapter, **selection
    )
    assert resumed["status"] == "FAIL", resumed["details"]
    assert "after resumed real step2" in resumed["details"]["error"]["message"]
    assert len(copy_calls) == 2
    assert actual_steps == [2]
    assert (out / "identity.json").exists()
    assert (out / "inherited_parent").is_dir()
    assert not (out / ".inherited_parent.copying").exists()
    assert not (out / ".inherited_parent.file.part").exists()
    assert (out / "X_BASE/step_02/updates/attempt_0000/checkpoint.pt").exists()
    assert instances[-1].generation_calls == 64
    assert resumed["details"]["runtime_counts"]["optimizer_steps_observed_at_least"] == 2
    assert file_hash(out / "inherited_parent/alarm_stop.json") == file_hash(
        parent / "alarm_stop.json"
    )
    # The preparation-only exception must not recreate a deleted identity once
    # generation or optimizer evidence exists in the new execution segment.
    (out / "identity.json").unlink()
    loaded = len(instances)
    with pytest.raises(ValueError, match="Missing continuation identity after runtime evidence"):
        r4_runtime.run_r4(*args, allow_training=True, resume=True, _adapter_factory=adapter)
    assert len(instances) == loaded
    assert not (out / "identity.json").exists()
