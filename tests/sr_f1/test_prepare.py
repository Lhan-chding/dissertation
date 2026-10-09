"""CPU boundary regressions; these tests make no claims about actual Qwen outputs."""

import json
from pathlib import Path

import pytest

from sr_f1.contract import PACKAGE, file_hash
from sr_f1.prepare import (
    _copy_registered,
    _run_reference_checks,
    isolation_tests,
    render_inputs,
    verify_package,
)


def test_uploaded_package_is_authentic_and_copy_is_immutable(tmp_path):
    before = file_hash(PACKAGE / "PACKAGE_SHA256.json")
    assert verify_package()["checked_files"] == 76
    _copy_registered(tmp_path)
    _copy_registered(tmp_path)
    assert file_hash(tmp_path / "config/SR_F1.json") == file_hash(PACKAGE / "config/SR_F1.json")
    (tmp_path / "config/SR_F1.json").write_text("{}")
    with pytest.raises(PermissionError, match="Existing registered"):
        _copy_registered(tmp_path)
    assert file_hash(PACKAGE / "PACKAGE_SHA256.json") == before


def test_package_hash_manifest_cannot_be_rewritten_to_bless_changed_bytes(tmp_path, monkeypatch):
    import sr_f1.prepare as module

    (tmp_path / "PACKAGE_SHA256.json").write_text("{}")
    monkeypatch.setattr(module, "PACKAGE", tmp_path)
    with pytest.raises(PermissionError, match="inventory itself changed"):
        module.verify_package()


def test_no_git_source_receipt_is_checked_against_actual_code(tmp_path, monkeypatch):
    import subprocess

    import sr_f1.prepare as module
    from mm_core.execution import object_hash

    monkeypatch.setattr(module, "SOURCE_ROOT", tmp_path)

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(module.subprocess, "check_output", unavailable)
    receipt = {"source_commit": "a" * 40, "source_tree_sha256": object_hash({})}
    (tmp_path / "SOURCE_DEPLOYMENT.json").write_text(json.dumps(receipt))
    assert module.source_identity()["source_commit"] == "a" * 40
    (tmp_path / "pyproject.toml").write_text("[project]\nname='changed'\n")
    with pytest.raises(PermissionError, match="source-tree receipt"):
        module.source_identity()


def test_model_revision_name_does_not_substitute_for_weight_byte_identity(tmp_path):
    from types import SimpleNamespace

    from sr_f1.contract import load_config
    from sr_f1.prepare import model_identity

    snapshot = tmp_path / load_config()["model"]["revision"]
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"not actual Qwen weights")
    with pytest.raises(PermissionError, match="composite model hash"):
        model_identity(snapshot, SimpleNamespace(identity={}))


def test_independent_rebuild_and_validator_do_not_modify_package(tmp_path):
    before = {str(p): file_hash(p) for p in PACKAGE.rglob("*") if p.is_file()}
    _copy_registered(tmp_path)
    result = _run_reference_checks(tmp_path)
    assert result["status"] == "PASS"
    assert all(result["byte_identical"].values())
    assert result["independent_validation"]["gold_outputs_scored_J1"] == 4800
    assert before == {str(p): file_hash(p) for p in PACKAGE.rglob("*") if p.is_file()}


def test_wrong_font_fails_before_rendering(tmp_path):
    font = tmp_path / "wrong.ttf"
    font.write_bytes(b"not the uploaded font")
    with pytest.raises(PermissionError, match="Renderer font"):
        render_inputs(tmp_path, font, font)
    assert not (tmp_path / "images").exists()


class TensorBoundary:
    """Control-only processor substitute for checking the isolation assertion logic."""

    def __init__(self, leak=False, ignore_image=False, ignore_question=False):
        self.leak, self.ignore_image, self.ignore_question = leak, ignore_image, ignore_question

    def prepare(self, row, root):
        from sr_f1.data import model_input

        visible = model_input(row)
        image = "same" if self.ignore_image else visible["image_file"]
        text = "same" if self.ignore_question else visible["text"]
        payload = [image, text, row.get("answer") if self.leak else None]
        return dict(
            chat_text=text,
            routing=dict(input_tensor_hash=json.dumps(payload), processed_pixel_sha256=image),
        )


def sample_row():
    return dict(
        qid="unobservable-opaque-id",
        image_file="images/ENGINE/a.png",
        text="Count Alpha entries",
        plain_text="How many Alpha entries?",
    )


def test_isolation_requires_both_negative_and_positive_controls(tmp_path):
    result = isolation_tests(TensorBoundary(), sample_row(), tmp_path)
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    "runtime",
    [
        TensorBoundary(leak=True),
        TensorBoundary(ignore_image=True),
        TensorBoundary(ignore_question=True),
    ],
)
def test_isolation_rejects_leakage_or_noop_processor(tmp_path, runtime):
    with pytest.raises(PermissionError, match="isolation control"):
        isolation_tests(runtime, sample_row(), tmp_path)


def test_local_preparation_is_not_gpu_certification(tmp_path, monkeypatch):
    import sr_f1.prepare as module

    monkeypatch.setattr(module, "_run_reference_checks", lambda _: {"status": "PASS"})

    def fake_render(root, *_):
        (root / "RENDER_RECEIPT.json").write_text("{}")
        return {}

    monkeypatch.setattr(module, "render_inputs", fake_render)
    result = module.prepare_run(
        tmp_path,
        font_path=Path("unused"),
        bold_path=Path("unused"),
        operator="test",
        local_only=True,
    )
    assert result["status"] == "LOCAL_PREPARED"
    assert result["gpu_execution_authorized"] is False
    assert result["processor_verified"] is False
    assert not (tmp_path / "MODEL_ENVIRONMENT_IDENTITY.json").exists()
