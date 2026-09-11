"""An authorized continuation preserves the stopped run and its sampling identity."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.core import canonical_hash, file_hash, write_json


@pytest.fixture
def stopped_parent(tmp_path):
    parent = tmp_path / "parent"
    old_source = {"source_commit": "a" * 40, "source_files": {"src/old.py": "a" * 64}}
    new_source = {"source_commit": "b" * 40, "source_files": {"src/new.py": "b" * 64}}
    identity = {
        "phase": "R4",
        "model_hash": "m" * 64,
        "data_hash": "d" * 64,
        "config_hash": "c" * 64,
        "source_hash": canonical_hash(old_source),
    }
    diagnostic = {
        "mean_token_kl": 0.019,
        "sequence_log_ratio_p99_abs": 2.079,
        "should_stop": True,
        "alarms": {"mean_token_kl": False, "sequence_log_ratio_p99_abs": True},
        "thresholds": {
            "mean_token_kl": 0.1,
            "sequence_log_ratio_p99_abs": 2.0,
            "comparison": "strict_greater_than",
        },
    }
    write_json(parent / "identity.json", identity)
    write_json(parent / "runtime_lock.json", {"identity": identity, "source": old_source})
    write_json(parent / "alarm_stop.json", {"control_diagnostic": diagnostic})
    write_json(parent / "checkpoint_manifest.json", {"checkpoints": []})
    (parent / "checkpoint.pt").write_bytes(b"full Adam and RNG evidence")
    write_json(parent / "manifest.json", {"files": []})
    binding = {
        "status": "PASS_STOPPED_AUDIT",
        "root": str(parent),
        "files": {p.name: file_hash(p) for p in parent.iterdir() if p.is_file()},
        "manifest_sha256": file_hash(parent / "manifest.json"),
        "runtime_lock_sha256": file_hash(parent / "runtime_lock.json"),
        "alarm_stop_sha256": file_hash(parent / "alarm_stop.json"),
        "source": old_source,
        "identity": identity,
        "stop": {
            "arm": "X_BASE",
            "step": 35,
            "checkpoint": {
                "checkpoint_path": str(parent / "checkpoint.pt"),
                "checkpoint_sha256": file_hash(parent / "checkpoint.pt"),
            },
            "diagnostic": diagnostic,
        },
        "checkpoint_manifest": {"checkpoints": []},
    }
    binding["audit_hash"] = canonical_hash({k: v for k, v in binding.items() if k != "root"})
    return parent, binding, new_source


def _decision(binding, source):
    from src.r4_continuation import REVIEWED_POLICY

    return {
        "schema_version": 1,
        "authorization": {
            "user_request": "Analyze, fix the discovered issue, then resume training.",
            "reason": "The audited stop is a cumulative sequence warning only.",
        },
        "parent_audit_hash": binding["audit_hash"],
        "execution_source_hash": canonical_hash(source),
        "policy": copy.deepcopy(REVIEWED_POLICY),
    }


def test_preparation_copies_all_bytes_and_keeps_parent_alarm(stopped_parent, tmp_path):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    (parent / "extra-preserved.log").write_text("diagnostic beyond the phase manifest\n")
    before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    decision = _decision(binding, source)
    inputs = copy.deepcopy((decision, binding, source))
    out = tmp_path / "child"
    context = prepare_continuation(parent, out, decision, binding, source, allow_training=True)
    snapshot = out / "inherited_parent"
    assert context["parent_root"] == snapshot
    assert {
        str(p.relative_to(snapshot)): p.read_bytes() for p in snapshot.rglob("*") if p.is_file()
    } == before
    assert context["logical_identity"] == binding["identity"]
    assert context["contract"]["reviewed_warning_policy"] == "sequence_p99_only_two_arms_to_step64"
    assert context["contract"]["parent_binding"] == binding
    assert (decision, binding, source) == inputs
    assert (parent / "alarm_stop.json").read_bytes() == before["alarm_stop.json"]


def test_preparation_requires_explicit_authorization_before_any_write(stopped_parent, tmp_path):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    with pytest.raises(ValueError, match="allow_training"):
        prepare_continuation(parent, out, _decision(binding, source), binding, source)
    assert not out.exists()


def test_resume_uses_snapshot_when_external_parent_is_unavailable(stopped_parent, tmp_path):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    decision = _decision(binding, source)
    first = prepare_continuation(parent, out, decision, binding, source, allow_training=True)
    parent.rename(tmp_path / "external-parent-no-longer-mounted")
    snapshot = out / "inherited_parent"
    before = {str(p.relative_to(out)): p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    second = prepare_continuation(
        snapshot,
        out,
        out / "continuation_decision.json",
        binding,
        source,
        allow_training=True,
        resume=True,
    )
    assert first == second
    assert before == {
        str(p.relative_to(out)): p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "change",
    [
        "wrong_parent",
        "wrong_source",
        "blank_request",
        "blank_reason",
        "missing_authorization",
        "extra_authorization",
        "single_arm",
        "fewer_steps",
        "bool_steps",
        "changed_mean_threshold",
        "changed_p99_threshold",
        "drop_hard_stop",
        "clip_warning",
        "extra_decision_key",
    ],
)
def test_decision_rejects_scope_source_and_policy_changes(stopped_parent, change):
    from src.r4_continuation import validate_continuation_decision

    _, binding, source = stopped_parent
    decision = _decision(binding, source)
    if change == "wrong_parent":
        decision["parent_audit_hash"] = "0" * 64
    elif change == "wrong_source":
        decision["execution_source_hash"] = "0" * 64
    elif change == "blank_request":
        decision["authorization"]["user_request"] = " "
    elif change == "blank_reason":
        decision["authorization"]["reason"] = "\n"
    elif change == "missing_authorization":
        del decision["authorization"]
    elif change == "extra_authorization":
        decision["authorization"]["override_all_stops"] = True
    elif change == "single_arm":
        decision["policy"]["arms"] = ["X_BASE"]
    elif change == "fewer_steps":
        decision["policy"]["steps"] = 63
    elif change == "bool_steps":
        decision["policy"]["steps"] = True
    elif change == "changed_mean_threshold":
        decision["policy"]["thresholds"]["mean_token_kl"] = 0.2
    elif change == "changed_p99_threshold":
        decision["policy"]["thresholds"]["sequence_log_ratio_p99_abs"] = 3.0
    elif change == "drop_hard_stop":
        decision["policy"]["hard_stops"].remove("measurement_fault")
    elif change == "clip_warning":
        decision["policy"]["sequence_p99"] = "clip_and_continue"
    elif change == "extra_decision_key":
        decision["unreviewed_override"] = True
    with pytest.raises(ValueError):
        validate_continuation_decision(decision, binding, source)


def _rehash(binding):
    binding["audit_hash"] = canonical_hash(
        {k: v for k, v in binding.items() if k not in ("root", "audit_hash")}
    )


@pytest.mark.parametrize(
    "fault",
    ["mean_alarm", "mean_over_limit", "wrong_threshold", "wrong_status", "no_alarm", "step64"],
)
def test_stopped_audit_only_permits_sequence_only_stop(stopped_parent, fault):
    from src.r4_continuation import validate_continuation_decision

    _, binding, source = stopped_parent
    diagnostic = binding["stop"]["diagnostic"]
    if fault == "mean_alarm":
        diagnostic["alarms"]["mean_token_kl"] = True
    elif fault == "mean_over_limit":
        diagnostic["mean_token_kl"] = 0.101
    elif fault == "wrong_threshold":
        diagnostic["thresholds"]["sequence_log_ratio_p99_abs"] = 3.0
    elif fault == "wrong_status":
        binding["status"] = "PASS"
    elif fault == "no_alarm":
        diagnostic["should_stop"] = False
    elif fault == "step64":
        binding["stop"]["step"] = 64
    _rehash(binding)
    with pytest.raises(ValueError):
        validate_continuation_decision(_decision(binding, source), binding, source)


@pytest.mark.parametrize("fault", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_diagnostic_never_enters_a_continuation(stopped_parent, fault):
    from src.r4_continuation import validate_continuation_decision

    _, binding, source = stopped_parent
    decision = _decision(binding, source)
    binding["stop"]["diagnostic"]["mean_token_kl"] = float(fault)
    with pytest.raises(ValueError):
        validate_continuation_decision(decision, binding, source)


@pytest.mark.parametrize(
    "name", ["manifest.json", "runtime_lock.json", "alarm_stop.json", "checkpoint.pt"]
)
def test_parent_tampering_is_rejected_before_snapshot_write(stopped_parent, tmp_path, name):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    (parent / name).write_bytes(b"changed")
    out = tmp_path / "child"
    with pytest.raises(ValueError, match="changed"):
        prepare_continuation(
            parent, out, _decision(binding, source), binding, source, allow_training=True
        )
    assert not out.exists()


@pytest.mark.parametrize("name", ["measurement_fault.json", "policy_contamination.json"])
def test_new_preserved_fault_cannot_be_bypassed(stopped_parent, tmp_path, name):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    write_json(parent / "nested" / name, {"reason": "preserved"})
    with pytest.raises(ValueError, match="fault"):
        prepare_continuation(
            parent,
            tmp_path / "child",
            _decision(binding, source),
            binding,
            source,
            allow_training=True,
        )


@pytest.mark.parametrize(
    "kind",
    [
        "file_symlink",
        "directory_symlink",
        "root_symlink",
        "destination_symlink",
        "inside_parent",
        "path_escape",
    ],
)
def test_paths_cannot_escape_or_alias_parent_evidence(stopped_parent, tmp_path, kind):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    if kind == "file_symlink":
        (parent / "aliased.pt").symlink_to(parent / "checkpoint.pt")
    elif kind == "directory_symlink":
        (parent / "aliased-directory").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "root_symlink":
        alias = tmp_path / "parent-alias"
        alias.symlink_to(parent, target_is_directory=True)
        parent = alias
    elif kind == "destination_symlink":
        out.symlink_to(parent, target_is_directory=True)
    elif kind == "inside_parent":
        out = parent / "child"
    elif kind == "path_escape":
        binding["files"]["../outside.pt"] = "a" * 64
        _rehash(binding)
    with pytest.raises(ValueError):
        prepare_continuation(
            parent, out, _decision(binding, source), binding, source, allow_training=True
        )


@pytest.mark.parametrize("target", ["alarm_stop.json", "extra-preserved.log"])
def test_resume_refuses_to_repair_a_changed_snapshot(stopped_parent, tmp_path, target):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    (parent / "extra-preserved.log").write_text("outside original manifest\n")
    out = tmp_path / "child"
    decision = _decision(binding, source)
    prepare_continuation(parent, out, decision, binding, source, allow_training=True)
    copied = out / "inherited_parent" / target
    copied.write_bytes(b"changed snapshot")
    with pytest.raises(ValueError):
        prepare_continuation(
            out / "inherited_parent",
            out,
            decision,
            binding,
            source,
            allow_training=True,
            resume=True,
        )
    assert copied.read_bytes() == b"changed snapshot"


def test_parent_artifacts_are_bound_even_when_original_manifest_omits_them(
    stopped_parent, tmp_path
):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    (parent / "extra-preserved.log").write_text("not in original manifest\n")
    context = prepare_continuation(
        parent, tmp_path / "child", _decision(binding, source), binding, source, allow_training=True
    )
    record = json.loads((tmp_path / "child" / "continuation_binding.json").read_text())
    assert record["snapshot_files"]["extra-preserved.log"]["sha256"] == file_hash(
        parent / "extra-preserved.log"
    )
    assert context["continuation_hash"] == canonical_hash(
        {
            k: v
            for k, v in context["contract"].items()
            if k not in ("continuation_hash", "parent_dir", "parent_binding")
        }
    )


def test_read_only_verification_reuses_snapshot_and_never_writes(
    stopped_parent, tmp_path, monkeypatch
):
    from src.r4_continuation import prepare_continuation, verify_continuation

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    first = prepare_continuation(
        parent, out, _decision(binding, source), binding, source, allow_training=True
    )
    parent.rename(tmp_path / "not-mounted")

    def forbid_write(*args, **kwargs):
        pytest.fail("Read-only continuation verifier attempted a write")

    monkeypatch.setattr(Path, "mkdir", forbid_write)
    monkeypatch.setattr(Path, "write_text", forbid_write)
    monkeypatch.setattr(Path, "write_bytes", forbid_write)
    assert verify_continuation(out, binding, source) == first


@pytest.mark.parametrize(
    "filename",
    ["continuation_decision.json", "continuation_binding.json", "logical_sampling_identity.json"],
)
def test_read_only_verifier_refuses_changed_contract_documents(stopped_parent, tmp_path, filename):
    from src.r4_continuation import prepare_continuation, verify_continuation

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    prepare_continuation(
        parent, out, _decision(binding, source), binding, source, allow_training=True
    )
    changed = json.loads((out / filename).read_text())
    changed["unexpected"] = "tamper"
    write_json(out / filename, changed)
    with pytest.raises(ValueError):
        verify_continuation(out, binding, source)


def _interrupt_snapshot_copy(monkeypatch, parent, out, binding, source):
    from src import r4_continuation as module

    original = module._open_regular
    selected_reads = 0

    class InterruptedFile:
        def __init__(self, stream):
            self.stream = stream
            self.read_once = False

        def read(self, _size):
            if self.read_once:
                raise OSError("simulated copy interruption")
            self.read_once = True
            return self.stream.read(5)

    @contextmanager
    def interrupted(root, relative):
        nonlocal selected_reads
        with original(root, relative) as stream:
            if root == parent and relative == "checkpoint.pt":
                selected_reads += 1
                if selected_reads == 2:
                    yield InterruptedFile(stream)
                    return
            yield stream

    with monkeypatch.context() as local:
        local.setattr(module, "_open_regular", interrupted)
        with pytest.raises(OSError, match="simulated copy interruption"):
            module.prepare_continuation(
                parent, out, _decision(binding, source), binding, source, allow_training=True
            )


def test_interrupted_snapshot_copy_resumes_without_publishing_or_rewriting_files(
    stopped_parent, tmp_path, monkeypatch
):
    from src.r4_continuation import prepare_continuation, verify_continuation

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    original_bytes = {p.name: p.read_bytes() for p in parent.iterdir()}
    _interrupt_snapshot_copy(monkeypatch, parent, out, binding, source)
    assert not (out / "inherited_parent").exists()
    copying = out / ".inherited_parent.copying"
    complete = copying / "alarm_stop.json"
    before = complete.stat().st_mtime_ns
    assert complete.read_bytes() == original_bytes["alarm_stop.json"]
    assert (out / ".inherited_parent.file.part").read_bytes() == b"full "
    decision = _decision(binding, source)
    result = prepare_continuation(
        parent, out, decision, binding, source, allow_training=True, resume=True
    )
    assert (out / "inherited_parent/alarm_stop.json").stat().st_mtime_ns == before
    assert not copying.exists()
    assert not (out / ".inherited_parent.file.part").exists()
    assert {p.name: p.read_bytes() for p in parent.iterdir()} == original_bytes
    assert verify_continuation(out, binding, source) == result


@pytest.mark.parametrize("mutation", ["changed-file", "extra-file", "extra-directory"])
def test_resumed_copy_rejects_changed_or_extra_complete_scratch_without_overwriting(
    stopped_parent, tmp_path, monkeypatch, mutation
):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    _interrupt_snapshot_copy(monkeypatch, parent, out, binding, source)
    copying = out / ".inherited_parent.copying"
    if mutation == "changed-file":
        (copying / "alarm_stop.json").write_bytes(b"changed complete copied evidence")
    elif mutation == "extra-file":
        (copying / "unaccounted.bin").write_bytes(b"extra")
    else:
        (copying / "unaccounted").mkdir()
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=r"[Pp]artial"):
        prepare_continuation(
            parent,
            out,
            _decision(binding, source),
            binding,
            source,
            allow_training=True,
            resume=True,
        )
    assert {
        str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()
    } == before
    assert not (out / "inherited_parent").exists()


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_partial_transfer_scratch_cannot_alias_and_truncate_parent(
    stopped_parent, tmp_path, monkeypatch, alias
):
    from src.r4_continuation import prepare_continuation

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    _interrupt_snapshot_copy(monkeypatch, parent, out, binding, source)
    scratch = out / ".inherited_parent.file.part"
    scratch.unlink()
    if alias == "symlink":
        scratch.symlink_to(parent / "checkpoint.pt")
    else:
        scratch.hardlink_to(parent / "checkpoint.pt")
    before = (parent / "checkpoint.pt").read_bytes()
    with pytest.raises((OSError, ValueError)):
        prepare_continuation(
            parent,
            out,
            _decision(binding, source),
            binding,
            source,
            allow_training=True,
            resume=True,
        )
    assert (parent / "checkpoint.pt").read_bytes() == before
    assert not (out / "inherited_parent").exists()


def test_completed_copy_can_resume_atomic_publication_without_recopying(
    stopped_parent, tmp_path, monkeypatch
):
    from src import r4_continuation as module

    parent, binding, source = stopped_parent
    out = tmp_path / "child"
    decision = _decision(binding, source)
    original_replace = module.os.replace

    def fail_publication(old, new):
        if Path(old).name == ".inherited_parent.copying":
            raise OSError("simulated publication interruption")
        return original_replace(old, new)

    with monkeypatch.context() as local:
        local.setattr(module.os, "replace", fail_publication)
        with pytest.raises(OSError, match="publication interruption"):
            module.prepare_continuation(parent, out, decision, binding, source, allow_training=True)
    copying = out / ".inherited_parent.copying"
    before = {p.name: p.stat().st_mtime_ns for p in copying.iterdir()}
    assert not (out / "inherited_parent").exists()
    module.prepare_continuation(
        parent, out, decision, binding, source, allow_training=True, resume=True
    )
    assert {p.name: p.stat().st_mtime_ns for p in (out / "inherited_parent").iterdir()} == before
    assert not copying.exists()
