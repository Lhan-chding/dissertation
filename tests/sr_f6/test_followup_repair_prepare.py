"""Synthetic single-process preparation and durable failure-boundary checks."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6_followup import repair_prepare, review_package, source


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = tmp_path.resolve()
    root = project / "new-root"
    deployment = project / "sr_f6_followup_code_abcdef1"
    operations = project / "operations"
    deployment.mkdir()
    operations.mkdir()
    monkeypatch.setattr(source, "PROJECT", project)
    monkeypatch.setattr(source, "ROOT", root)
    monkeypatch.setattr(source, "_actual_deployment", lambda: deployment)
    monkeypatch.setattr(repair_prepare, "OPERATIONS", operations)
    calls = []
    consumer = dict(
        deployment=str(deployment),
        revision="abcdef1" + "0" * 33,
        registration_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
    )
    producer = dict(revision=source.PRODUCER_REVISION, scope="SYNTHETIC_ONLY")

    def register(actual_root, revision):
        assert actual_root == root and revision == consumer["revision"]
        assert not root.exists()
        assert (operations / repair_prepare.DIRECTORY / "REGISTER.ATTEMPT.json").exists()
        calls.append("REGISTER")
        root.mkdir()
        source._write(root / "REGISTRATION.json", {"fixture": "SYNTHETIC_ONLY"})
        return dict(
            consumer=copy.deepcopy(consumer),
            registration=dict(
                status="BOUND_NOT_AUTHORIZED",
                source_manifest=dict(path=str(root / "SOURCE_MANIFEST.json"), sha256="b" * 64),
                spec=dict(path=str(root / "SPEC.json"), sha256="c" * 64),
            ),
        )

    def support(actual_root):
        assert actual_root == root and calls == ["REGISTER"]
        assert (operations / repair_prepare.DIRECTORY / "BIND_SUPPORT.ATTEMPT.json").exists()
        calls.append("BIND_SUPPORT")
        source._write(root / "SUPPORT_BINDING.attempt.json", {"fixture": "SYNTHETIC_ONLY"})
        return dict(
            consumer=copy.deepcopy(consumer),
            support_binding=dict(status="VERIFIED_SUPPORT_PASS_REVIEW_PENDING"),
            support_binding_ref=dict(path=str(root / "SUPPORT_BINDING.json"), sha256="d" * 64),
        )

    def package(actual_root):
        assert actual_root == root and calls == ["REGISTER", "BIND_SUPPORT"]
        assert (operations / repair_prepare.DIRECTORY / "REVIEW_PACKAGE.ATTEMPT.json").exists()
        calls.append("REVIEW_PACKAGE")
        return dict(
            consumer=copy.deepcopy(consumer),
            input_producer=producer,
            status="BLIND_PACKAGE_ONLY",
            review_status="REVIEW_NOT_RUN",
            training_allowed=False,
            gate_status="NOT_SIGNED",
            package=dict(path="REVIEW/PACKAGE/BINDING.json", sha256="e" * 64),
        )

    monkeypatch.setattr(source, "register", register)
    monkeypatch.setattr(source, "bind_support", support)
    monkeypatch.setattr(review_package, "write_package", package)
    return SimpleNamespace(
        root=root,
        operations=operations,
        deployment=deployment,
        calls=calls,
        consumer=consumer,
        producer=producer,
        monkeypatch=monkeypatch,
        directory=operations / repair_prepare.DIRECTORY,
    )


def test_three_original_operations_run_once_in_order(setup):
    result = repair_prepare.prepare(setup.root, setup.consumer["revision"])
    assert setup.calls == ["REGISTER", "BIND_SUPPORT", "REVIEW_PACKAGE"]
    assert result["status"] == "PREPARED_NOT_AUTHORIZED"
    assert result["review_status"] == "REVIEW_NOT_RUN"
    assert result["consumer"] == setup.consumer and result["input_producer"] == setup.producer
    assert result["new_model_calls"] == result["optimizer_updates"] == 0
    assert all(
        result[key] is False
        for key in ("training_allowed", "generation_allowed", "evaluation_allowed")
    )
    assert {path.name for path in setup.directory.iterdir()} == {
        "ATTEMPT.json",
        "RESULT.json",
        "REGISTER.ATTEMPT.json",
        "REGISTER.RESULT.json",
        "BIND_SUPPORT.ATTEMPT.json",
        "BIND_SUPPORT.RESULT.json",
        "REVIEW_PACKAGE.ATTEMPT.json",
        "REVIEW_PACKAGE.RESULT.json",
    }
    before = {path.name: path.read_bytes() for path in setup.directory.iterdir()}
    with pytest.raises(PermissionError, match="existing or partial"):
        repair_prepare.prepare(setup.root, setup.consumer["revision"])
    assert before == {path.name: path.read_bytes() for path in setup.directory.iterdir()}
    assert len(setup.calls) == 3


@pytest.mark.parametrize("kind", ["empty", "partial", "file", "symlink", "unknown"])
def test_existing_partial_or_unknown_root_rejected_without_new_artifacts(setup, kind):
    actual_root = setup.root
    if kind == "unknown":
        actual_root = setup.root.parent / "unregistered-root"
    elif kind == "file":
        actual_root.write_text("SYNTHETIC_ONLY preserved")
    elif kind == "symlink":
        actual_root.symlink_to(setup.root.parent / "missing")
    else:
        actual_root.mkdir()
        if kind == "partial":
            (actual_root / "SOURCE_MANIFEST.json").write_text("SYNTHETIC_ONLY partial")
    with pytest.raises(PermissionError):
        repair_prepare.prepare(actual_root, setup.consumer["revision"])
    assert setup.calls == [] and not setup.directory.exists()


@pytest.mark.parametrize("stage", ["REGISTER", "BIND_SUPPORT", "REVIEW_PACKAGE"])
def test_failed_stage_preserves_attempt_and_prior_results_without_retry(setup, stage):
    target, name = {
        "REGISTER": (source, "register"),
        "BIND_SUPPORT": (source, "bind_support"),
        "REVIEW_PACKAGE": (review_package, "write_package"),
    }[stage]

    def fail(*args):
        assert (setup.directory / (stage + ".ATTEMPT.json")).exists()
        raise PermissionError("SYNTHETIC_ONLY stage failure")

    setup.monkeypatch.setattr(target, name, fail)
    with pytest.raises(PermissionError, match="stage failure"):
        repair_prepare.prepare(setup.root, setup.consumer["revision"])
    failure = json.loads((setup.directory / "FAILURE.json").read_text())
    assert failure["stage"] == stage and failure["status"] == "STOP_PRESERVE"
    assert not (setup.directory / "RESULT.json").exists()
    assert not (setup.directory / (stage + ".RESULT.json")).exists()
    before = {path.name: path.read_bytes() for path in setup.directory.iterdir()}
    with pytest.raises(PermissionError, match="no automatic retry"):
        repair_prepare.prepare(setup.root, setup.consumer["revision"])
    assert before == {path.name: path.read_bytes() for path in setup.directory.iterdir()}


def test_interrupted_registration_intent_is_not_retried(setup):
    setup.monkeypatch.setattr(
        source, "register", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        repair_prepare.prepare(setup.root, setup.consumer["revision"])
    assert (setup.directory / "REGISTER.ATTEMPT.json").exists()
    assert not setup.root.exists()
    assert not (setup.directory / "FAILURE.json").exists()
    with pytest.raises(PermissionError, match="Prior or unknown"):
        repair_prepare.prepare(setup.root, setup.consumer["revision"])


def test_partial_operation_directory_prevents_any_registration(setup):
    setup.directory.mkdir()
    with pytest.raises(PermissionError, match="Prior or unknown"):
        repair_prepare.prepare(setup.root, setup.consumer["revision"])
    assert not setup.root.exists() and setup.calls == []


def test_invalid_revision_has_no_side_effects(setup):
    with pytest.raises(PermissionError, match="revision"):
        repair_prepare.prepare(setup.root, "bad")
    assert not setup.root.exists() and not setup.directory.exists()


def test_repair_prepare_launcher_remains_zero_gpu_and_checks_new_root():
    script = (
        Path(source.__file__).resolve().parents[2] / "scripts/sr_f6_followup/followup_cpu.sbatch"
    )
    text = script.read_text()
    assert "register|repair-prepare)" in text
    assert (
        '-m sr_f6_followup.repair_prepare --root "$SRF6_ROOT" --revision "$SRF6_REVISION"' in text
    )
    assert "CUDA_VISIBLE_DEVICES=''" in text and "#SBATCH --gres" not in text
    assert "sr_f6_confirm_followup_v2_20261029" in text
    assert "CONFIRM_FOLLOWUP_V2_EXECUTION_20261029" in text
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_original_support_reconstruction_runs_once_across_all_three_stages(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "prepare_source_fixtures", Path(__file__).with_name("test_followup_source.py")
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    setup = fixtures.setup.__wrapped__(tmp_path, monkeypatch)
    saved = fixtures.save_analysis(setup)
    operations = setup.project / "operations"
    operations.mkdir()
    monkeypatch.setattr(repair_prepare, "OPERATIONS", operations)

    def package(root):
        # The real package begins with this same source gate. Package content
        # and selection are covered by its dedicated full-matrix tests.
        context = source.require_support(root, purpose="review")
        assert len(saved.scans) == 1
        return dict(
            consumer=context["consumer"],
            input_producer=context["evidence"]["input_producer"],
            status="BLIND_PACKAGE_ONLY",
            review_status="REVIEW_NOT_RUN",
            training_allowed=False,
            gate_status="NOT_SIGNED",
            package=dict(path="REVIEW/PACKAGE/BINDING.json", sha256="e" * 64),
        )

    monkeypatch.setattr(review_package, "write_package", package)
    result = repair_prepare.prepare(setup.root, "abcdef1" + "0" * 33)
    assert saved.scans == ["full"]
    assert result["status"] == "PREPARED_NOT_AUTHORIZED"
    assert (setup.root / "REGISTRATION.json").exists()
    assert (setup.root / "SUPPORT_BINDING.attempt.json").exists()
    assert (setup.root / "SUPPORT_BINDING.json").exists()
