"""Safety gates for large-model entry points; these tests never start a model."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from dataclasses import FrozenInstanceError, is_dataclass, replace
from pathlib import Path

import pytest

from compbias.models.qwen_vl import (
    DEFAULT_MODEL_NAME,
    PINNED_MODEL_REVISION,
    PINNED_TRANSFORMERS_REVISION,
    PINNED_VERL_REVISION,
    PINNED_VLLM_REVISION,
    ModelSnapshotEvidence,
    VLMPreflightConfig,
    VLMPreflightReport,
    probe_local_cuda_devices,
    require_frozen_qwen_stack,
    require_large_gpu_acknowledgement,
    validate_preflight,
    verify_model_snapshot,
)


def _config(
    *,
    acknowledged: bool,
    api_audited: bool = True,
) -> VLMPreflightConfig:
    return VLMPreflightConfig(
        model_name=DEFAULT_MODEL_NAME,
        model_revision=PINNED_MODEL_REVISION,
        transformers_revision=PINNED_TRANSFORMERS_REVISION,
        verl_revision=PINNED_VERL_REVISION,
        vllm_revision=PINNED_VLLM_REVISION,
        acknowledge_large_gpu_run=acknowledged,
        verl_api_audited=api_audited,
    )


@pytest.fixture
def verified_snapshot(tmp_path: Path) -> ModelSnapshotEvidence:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    files = {
        "config.json": b"{}",
        "preprocessor_config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors": b"fixture-weights",
    }
    entries = []
    for relative, content in files.items():
        (snapshot / relative).write_bytes(content)
        entries.append(
            {
                "path": relative,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = tmp_path / "snapshot-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete_snapshot": True,
                "model_name": DEFAULT_MODEL_NAME,
                "revision": PINNED_MODEL_REVISION,
                "files": entries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return verify_model_snapshot(
        snapshot,
        manifest,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )


def test_preflight_configuration_is_immutable_and_requires_pinned_revisions() -> None:
    config = _config(acknowledged=False)

    assert is_dataclass(config)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.model_revision = "different"  # type: ignore[misc]
    with pytest.raises(ValueError, match="revision"):
        VLMPreflightConfig(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            model_revision="",
            transformers_revision="transformers-commit-fixture",
            verl_revision="verl-commit-fixture",
            vllm_revision="vllm-commit-fixture",
            acknowledge_large_gpu_run=False,
            verl_api_audited=True,
        )


def test_missing_acknowledgement_fails_before_heavy_framework_import(monkeypatch) -> None:
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers", "verl", "vllm"}:
            imported.append(name)
            raise AssertionError(f"preflight imported heavy dependency {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(RuntimeError, match="acknowledg"):
        require_large_gpu_acknowledgement(_config(acknowledged=False))

    assert imported == []


def test_gpu_is_mandatory_even_after_explicit_acknowledgement() -> None:
    config = _config(acknowledged=True)

    with pytest.raises(RuntimeError, match=r"GPU|CUDA"):
        validate_preflight(config, cuda_available=False, gpu_devices=())


def test_validate_preflight_itself_rejects_a_nonfrozen_stack() -> None:
    config = replace(_config(acknowledged=True), model_revision="different-fixed-revision")

    with pytest.raises(RuntimeError, match=r"frozen|model.*revision"):
        validate_preflight(
            config,
            cuda_available=True,
            gpu_devices=("GPU-fixture",),
        )


def test_unaudited_verl_api_is_rejected_instead_of_guessing_configuration_keys() -> None:
    config = _config(acknowledged=True, api_audited=False)

    with pytest.raises(RuntimeError, match=r"veRL|audit|revision"):
        validate_preflight(config, cuda_available=True, gpu_devices=("fixture-gpu",))


def test_metadata_only_preflight_does_not_import_training_frameworks(monkeypatch) -> None:
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers", "verl", "vllm"}:
            imported.append(name)
            raise AssertionError(f"preflight imported heavy dependency {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    report = validate_preflight(
        _config(acknowledged=True),
        cuda_available=True,
        gpu_devices=("fixture-gpu",),
    )

    assert report is not None
    assert imported == []


def test_execution_plan_rejects_a_different_but_nonmoving_stack_revision() -> None:
    config = replace(_config(acknowledged=True), model_revision="different-fixed-revision")

    with pytest.raises(RuntimeError, match=r"frozen|model.*revision"):
        require_frozen_qwen_stack(config)


def test_low_level_grpo_config_builder_is_not_publicly_exported() -> None:
    import compbias.models.qwen_vl as module

    assert "build_verl_grpo_config" not in module.__all__
    assert not hasattr(module, "build_verl_grpo_config")


def test_verified_snapshot_cannot_be_constructed_without_manifest_verification() -> None:
    with pytest.raises(TypeError):
        ModelSnapshotEvidence(
            path=(
                f"/srv/hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/{PINNED_MODEL_REVISION}"
            ),
            revision=PINNED_MODEL_REVISION,
            manifest_sha256="a" * 64,
            verified_file_count=4,
        )


def test_snapshot_verifier_rejects_custom_code_even_when_self_hashed(
    verified_snapshot: ModelSnapshotEvidence,
) -> None:
    snapshot = Path(verified_snapshot.path)
    manifest = Path(verified_snapshot.manifest_path)
    custom_code = snapshot / "modeling_qwen.py"
    custom_code.write_text("raise RuntimeError('untrusted custom code')\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"].append(
        {
            "path": custom_code.name,
            "size_bytes": custom_code.stat().st_size,
            "sha256": hashlib.sha256(custom_code.read_bytes()).hexdigest(),
        }
    )
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"unsafe|custom|code|extension"):
        verify_model_snapshot(
            snapshot,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_snapshot_verifier_rejects_auto_map_even_when_self_hashed(
    verified_snapshot: ModelSnapshotEvidence,
) -> None:
    snapshot = Path(verified_snapshot.path)
    manifest = Path(verified_snapshot.manifest_path)
    config = snapshot / "config.json"
    config.write_text(
        json.dumps({"auto_map": {"AutoModel": "modeling_qwen.UntrustedModel"}}),
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entry = next(item for item in payload["files"] if item["path"] == "config.json")
    entry["size_bytes"] = config.stat().st_size
    entry["sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"auto_map|custom code"):
        verify_model_snapshot(
            snapshot,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("member_kind", ("hidden-directory", "fifo"))
def test_snapshot_verifier_rejects_hidden_directories_and_nonregular_nodes(
    verified_snapshot: ModelSnapshotEvidence,
    member_kind: str,
) -> None:
    snapshot = Path(verified_snapshot.path)
    manifest = Path(verified_snapshot.manifest_path)
    if member_kind == "hidden-directory":
        (snapshot / ".unreviewed-cache").mkdir()
    else:
        os.mkfifo(snapshot / "unreviewed.pipe")

    with pytest.raises(RuntimeError, match=r"hidden|regular|node|FIFO"):
        verify_model_snapshot(
            snapshot,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_snapshot_verifier_rejects_an_index_with_missing_weight_shards(
    verified_snapshot: ModelSnapshotEvidence,
) -> None:
    snapshot = Path(verified_snapshot.path)
    manifest = Path(verified_snapshot.manifest_path)
    weights = snapshot / "model.safetensors"
    weights.unlink()
    index = snapshot / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {"model.layer": "model-00001-of-00002.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"] = [item for item in payload["files"] if item["path"] != weights.name]
    payload["files"].append(
        {
            "path": index.name,
            "size_bytes": index.stat().st_size,
            "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        }
    )
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"shard|weight|index"):
        verify_model_snapshot(
            snapshot,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_preflight_report_cannot_be_constructed_without_validation() -> None:
    with pytest.raises(TypeError):
        VLMPreflightReport(
            model_name=DEFAULT_MODEL_NAME,
            model_revision=PINNED_MODEL_REVISION,
            transformers_revision=PINNED_TRANSFORMERS_REVISION,
            verl_revision=PINNED_VERL_REVISION,
            vllm_revision=PINNED_VLLM_REVISION,
            gpu_devices=("GPU-fake",),
            verl_api_audited=True,
        )


def test_cuda_probe_uses_machine_output_and_rejects_unparseable_rows(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = "GPU-a123\nnot-a-gpu\nGPU-b456\n"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")

    assert probe_local_cuda_devices() == ("GPU-a123", "GPU-b456")


def test_snapshot_manifest_rejects_excessive_json_depth(
    verified_snapshot: ModelSnapshotEvidence,
) -> None:
    manifest = Path(verified_snapshot.manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(70):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    payload["irrelevant"] = nested
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"depth|complex"):
        verify_model_snapshot(
            verified_snapshot.path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )
