from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _paths_payload(root: Path, model: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_root": str(root),
        "model": {"qwen25vl3b": {"path": str(model)}},
        "storage": {
            "data": str(root / "data"),
            "outputs": str(root / "outputs"),
            "checkpoints": str(root / "checkpoints"),
            "trajectories": str(root / "trajectories"),
            "cache": str(root / "cache"),
        },
    }


def test_paths_config_is_closed_and_supports_environment_override(tmp_path: Path) -> None:
    from compbias.gpu_pilot.config import load_pilot_paths

    model = tmp_path / "model"
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump(_paths_payload(tmp_path, model)), encoding="utf-8")

    overridden = tmp_path / "override-model"
    paths = load_pilot_paths(
        config,
        environ={"COMPBIAS_MODEL_PATH": str(overridden)},
    )

    assert paths.project_root == tmp_path.resolve()
    assert paths.model_path == overridden.resolve()
    assert paths.outputs == (tmp_path / "outputs").resolve()

    payload = _paths_payload(tmp_path, model)
    payload["unknown"] = True
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_pilot_paths(config, environ={})


def test_pilot_data_contract_freezes_registered_counts_and_tasks() -> None:
    from compbias.gpu_pilot.config import load_pilot_data_config

    config = load_pilot_data_config(Path("configs/data/cva_chart_pilot.yaml"))

    assert config.dataset_id == "CVA-Chart-Pilot-v0.1"
    assert config.split_counts == {
        "calibration": 200,
        "smoke_train": 600,
        "pilot_train": 1200,
        "dev": 200,
        "iid_test": 300,
        "mechanism_ood": 300,
    }
    assert config.chart_types == ("grouped_bar", "line")
    assert config.operations == ("difference", "sum", "max_minus_min")
    assert config.counterfactual_pairs == 150
    assert config.natural_audit == 150


def test_preflight_rejects_low_vram_and_accepts_valid_fixture(tmp_path: Path) -> None:
    from compbias.gpu_pilot.config import PilotPaths
    from compbias.gpu_pilot.preflight import HardwareSnapshot, audit_server

    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ):
        (model / name).write_bytes(b"fixture")
    paths = PilotPaths(
        project_root=tmp_path,
        model_path=model,
        data=tmp_path / "data",
        outputs=tmp_path / "outputs",
        checkpoints=tmp_path / "checkpoints",
        trajectories=tmp_path / "trajectories",
        cache=tmp_path / "cache",
    )

    low = HardwareSnapshot(
        cuda_available=True,
        device_name="fixture",
        total_vram_gib=24.0,
        bf16_supported=True,
        torch_version="2.8.0+cu128",
        torch_cuda_runtime="12.8",
    )
    with pytest.raises(RuntimeError, match="VRAM"):
        audit_server(paths, hardware=low, free_disk_gib=180.0)

    valid = HardwareSnapshot(
        cuda_available=True,
        device_name="NVIDIA GeForce RTX 4090",
        total_vram_gib=47.37,
        bf16_supported=True,
        torch_version="2.8.0+cu128",
        torch_cuda_runtime="12.8",
    )
    report = audit_server(paths, hardware=valid, free_disk_gib=180.0)
    assert report["ready"] is True
    assert report["large_gpu_started"] is False
    assert report["model_path"] == str(model.resolve())


def test_qwen_loader_is_offline_and_disables_remote_code(tmp_path: Path) -> None:
    from compbias.gpu_pilot.qwen_smoke import load_local_qwen

    calls: list[tuple[str, str, dict[str, object]]] = []

    class StubModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> str:
            calls.append(("model", path, kwargs))
            return "model"

    class StubProcessor:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> str:
            calls.append(("processor", path, kwargs))
            return "processor"

    model_path = tmp_path / "model"
    model_path.mkdir()
    model, processor = load_local_qwen(
        model_path,
        model_class=StubModel,
        processor_class=StubProcessor,
        torch_dtype="bf16-fixture",
    )

    assert (model, processor) == ("model", "processor")
    assert {name for name, _path, _kwargs in calls} == {"model", "processor"}
    for _name, path, kwargs in calls:
        assert path == str(model_path.resolve())
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
    model_kwargs = next(kwargs for name, _path, kwargs in calls if name == "model")
    assert model_kwargs["torch_dtype"] == "bf16-fixture"
    assert model_kwargs["device_map"] == "cuda:0"


def test_gpu_entrypoints_exist_and_do_not_enable_training_by_default() -> None:
    required = (
        "00_preflight.py",
        "01_smoke_qwen.py",
        "02_generate_pilot_data.py",
        "03_base_calibration.py",
        "04_collect_natural.py",
        "05_pilot_a.py",
        "06_pilot_b.py",
        "07_analyze.py",
    )
    root = Path("experiments/gpu_pilot")
    for name in required:
        assert (root / name).is_file()
    for name, config in (
        ("05_pilot_a.py", "configs/train/pilot_a.yaml"),
        ("06_pilot_b.py", "configs/train/pilot_b_lm_only.yaml"),
    ):
        completed = subprocess.run(
            [sys.executable, str(root / name), "--config", config],
            check=False,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": ".:src"},
        )
        assert completed.returncode == 2
        assert "BLOCKED" in completed.stdout


def test_public_configuration_contains_no_online_model_identifier_as_path() -> None:
    model = yaml.safe_load(Path("configs/model/qwen25vl3b.yaml").read_text(encoding="utf-8"))
    assert model["local_files_only"] is True
    assert model["trust_remote_code"] is False
    assert model["path"] == "/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct"
    forbidden = {"token", "api_key", "password", "secret"}
    assert not forbidden.intersection(model)
