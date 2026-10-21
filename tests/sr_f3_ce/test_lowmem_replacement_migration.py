"""Recovered delivery adds a gate; it must never erase the original migration/failure."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_pending_cpu_migration import digest, final_migration_env, migration_env, save

CHECKOUT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recovered_migration(tmp_path, monkeypatch):
    base = migration_env.__wrapped__(tmp_path, monkeypatch)
    original_auth = base.auth
    base.auth = original_auth.with_name("MIGRATION_AUTHORIZATION.json")
    shutil.copyfile(original_auth, base.auth)

    def make():
        operation = base.module.Migration(base.root, base.auth, runner=base.slurm)
        original = operation.migrate
        operation.migrate = lambda: original(pause_receipt=base.pause)
        return operation

    base.make = make
    env = final_migration_env.__wrapped__(base, monkeypatch)
    failure = env.dispatcher.parent / "END.json"
    save(failure, dict(status="FAILED", error="original OOM -9", job_id="211333"))
    original_failure = failure.read_bytes()
    deploy = env.root / "operations" / ("f" * 40)
    deploy.mkdir()
    source = deploy / "verify_recovered_migration.py"
    shutil.copyfile(
        CHECKOUT / "scripts/sr_f3_ce/lowmem_replacement/verify_recovered_migration.py", source
    )
    recovery_source = deploy / "recover_delivery.py"
    recovery_source.write_text("# fake validated recovery gate; tested separately\n")
    spec = importlib.util.spec_from_file_location("recovered_migration_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "TARGET_ROOT", str(env.root))
    monkeypatch.setattr(module, "MIGRATION_COMMIT", env.auth.parent.name)
    auth_path = deploy / "RECOVERY_AUTHORIZATION.json"
    authorization = dict(
        source_sha256=digest(recovery_source),
        files={
            str(env.deployed.relative_to(env.root)): digest(env.deployed),
            str(failure.relative_to(env.root)): digest(failure),
        },
        extra_sources={str(source.relative_to(env.root)): digest(source)},
    )
    save(auth_path, authorization)
    profile_path = deploy / "MEMORY_PROFILE.json"
    save(profile_path, {"fixture_only": True})
    revision_identity = {"fixture_revision": "new-lowmem-builder"}

    def bind_revision():
        builder = json.loads(env.builder.read_text())
        builder["delivery_revision"] = revision_identity
        save(env.builder, builder)

    bind_revision()
    original_package = env.package

    def package(**kwargs):
        result = original_package(**kwargs)
        bind_revision()
        return result

    env.package = package
    proof_path = env.root / "delivery/FINAL_RECOVERY_VERIFIED.json"
    control = SimpleNamespace(calls=0, fail=None, write_proof=True, mutate=lambda value: value)

    class RecoveryGate:
        def __init__(self, root, authorization, *, runner):
            assert Path(root) == env.root and Path(authorization) == auth_path
            assert runner is env.slurm
            self.profile_path = profile_path

        def verify_final(self):
            control.calls += 1
            if control.fail:
                raise control.fail
            current = json.loads(auth_path.read_text())
            for relative, expected in current["files"].items():
                if digest(env.root / relative) != expected:
                    raise PermissionError("Recovery original evidence changed")
            result = dict(
                status="FINAL_RECOVERY_VERIFIED",
                job_id="216001",
                archive=json.loads(env.builder.read_text())["archive"],
                delivery_revision=revision_identity,
                memory_profile_sha256=digest(profile_path),
            )
            result = control.mutate(copy.deepcopy(result))
            if control.write_proof:
                save(proof_path, result)
            return result

    actual_load = module.load_bound

    def load_bound(path, expected, name):
        path = Path(path)
        if path == recovery_source:
            if digest(path) != expected:
                raise PermissionError("Bound verifier source changed")
            return SimpleNamespace(Recovery=RecoveryGate)
        loaded = actual_load(path, expected, name)
        monkeypatch.setattr(loaded, "TARGET_ROOT", str(env.root))
        return loaded

    monkeypatch.setattr(module, "load_bound", load_bound)
    env.verify_recovery = lambda: module.verify(env.root, auth_path, runner=env.slurm)
    env.recovery_control = control
    env.recovery_auth = auth_path
    env.recovery_source = recovery_source
    env.recovery_verifier_source = source
    env.recovery_module = module
    env.failure = failure
    env.failure_bytes = original_failure
    env.proof_path = proof_path
    return env


def test_recovered_acceptance_preserves_original_failure_and_migration(recovered_migration):
    env = recovered_migration
    result = env.verify_recovery()
    assert result["status"] == "FINAL_MIGRATION_VERIFIED"
    assert result["job_id"] == "215520"
    assert result["recovery_job_id"] == "216001"
    assert result["actual_node"] == "cpu-1"
    assert result["original_failed_dispatcher_end_sha256"] == digest(env.failure)
    assert result["recovery_final_sha256"] == digest(env.proof_path)
    assert env.failure.read_bytes() == env.failure_bytes
    assert env.recovery_control.calls == 1
    assert not any(call[0] == "sbatch" for call in env.slurm.calls)
    assert env.verify_recovery() == result


def test_missing_recovery_proof_cannot_use_old_builder_pass(recovered_migration):
    env = recovered_migration
    env.recovery_control.write_proof = False
    with pytest.raises((FileNotFoundError, PermissionError)):
        env.verify_recovery()
    assert not (env.root / "delivery/FINAL_MIGRATION_VERIFIED.json").exists()


def test_recovery_gate_failure_propagates_before_migration_acceptance(recovered_migration):
    env = recovered_migration
    env.recovery_control.fail = PermissionError("Recovery terminal not accepted")
    with pytest.raises(PermissionError, match="Recovery terminal"):
        env.verify_recovery()
    assert env.failure.read_bytes() == env.failure_bytes


def test_recovery_archive_must_match_original_builder_archive(recovered_migration):
    env = recovered_migration

    def different_archive(value):
        value["archive"]["sha256"] = "0" * 64
        return value

    env.recovery_control.mutate = different_archive
    with pytest.raises(PermissionError, match="Recovery archive binding"):
        env.verify_recovery()


@pytest.mark.parametrize("target", ["recovery_source", "deployed", "recovery_verifier_source"])
def test_changed_verifier_source_is_rejected(recovered_migration, target):
    env = recovered_migration
    path = getattr(env, target)
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    with pytest.raises(PermissionError):
        env.verify_recovery()


@pytest.mark.parametrize("state", ["DONE", "FAILED"])
def test_old_failure_cannot_be_rewritten_even_to_another_failure(recovered_migration, state):
    env = recovered_migration
    save(env.failure, dict(status=state, error="rewritten"))
    with pytest.raises(PermissionError, match="original evidence changed"):
        env.verify_recovery()


@pytest.mark.parametrize(
    "filename", ["MIGRATED.json", "RELEASE_GATE.json", "RESUMED.json", "ACTION_RESULT.json"]
)
def test_original_migration_and_guard_archive_evidence_remains_mandatory(
    recovered_migration, filename
):
    env = recovered_migration
    env.package(omit=filename)
    with pytest.raises(PermissionError, match="absent from delivery"):
        env.verify_recovery()


@pytest.mark.parametrize("filename", ["RELEASE_GATE.json", "RESUMED.json", "ACTION_RESULT.json"])
def test_rehashed_recovery_archive_cannot_hide_corrupt_migration_evidence(
    recovered_migration, filename
):
    env = recovered_migration
    env.package(corrupt=filename)
    with pytest.raises(PermissionError, match="Archived migration evidence differs"):
        env.verify_recovery()


@pytest.mark.parametrize("field,value", [("JobState", "FAILED"), ("NodeList", "cpu-2")])
def test_original_release_terminal_state_and_node_remain_required(
    recovered_migration, field, value
):
    env = recovered_migration
    env.slurm.fields[field] = value
    with pytest.raises(PermissionError, match="Actual terminal execution differs"):
        env.verify_recovery()


def test_original_release_claim_cannot_change_during_delivery_recovery(recovered_migration):
    env = recovered_migration
    task = env.root / "tasks/release/COMPLETE.json"
    receipt = json.loads(task.read_text())
    receipt["claim_id"] = "wrong"
    save(task, receipt)
    with pytest.raises(PermissionError, match="Release task claim differs"):
        env.verify_recovery()


def test_archive_changed_after_builder_acceptance_is_rejected(recovered_migration):
    env = recovered_migration
    with env.archive.open("ab") as stream:
        stream.write(b"unexpected bytes")
    with pytest.raises(PermissionError, match="Archive identity differs"):
        env.verify_recovery()


def test_persisted_recovery_proof_must_equal_gate_return_value(recovered_migration):
    env = recovered_migration
    env.recovery_control.write_proof = False
    save(env.proof_path, dict(status="FINAL_RECOVERY_VERIFIED", job_id="other"))
    with pytest.raises(PermissionError, match="Recovery final receipt changed"):
        env.verify_recovery()


def test_recovered_gate_itself_refuses_synthetic_old_dispatcher_done(recovered_migration):
    env = recovered_migration
    save(env.failure, dict(status="DONE"))
    # Even a stubbed recovery gate that accepts these bytes cannot weaken the
    # independent verifier's requirement to retain the original failed status.
    auth = json.loads(env.recovery_auth.read_text())
    auth["files"][str(env.failure.relative_to(env.root))] = digest(env.failure)
    save(env.recovery_auth, auth)
    with pytest.raises(PermissionError, match="Original failure was rewritten"):
        env.verify_recovery()


def test_lowmem_migration_requires_matching_delivery_revision(recovered_migration):
    env = recovered_migration
    builder = json.loads(env.builder.read_text())
    builder["delivery_revision"] = {"wrong": True}
    save(env.builder, builder)
    with pytest.raises(PermissionError, match="Builder run identity differs"):
        env.verify_recovery()


def test_lowmem_migration_requires_bound_memory_receipt(recovered_migration):
    env = recovered_migration

    def changed(value):
        value["memory_profile_sha256"] = "0" * 64
        return value

    env.recovery_control.mutate = changed
    with pytest.raises(PermissionError, match="Builder run identity differs"):
        env.verify_recovery()
