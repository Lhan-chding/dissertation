from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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

    config = load_pilot_data_config(Path("configs/data/cva_chart_pilot_v0_3.yaml"))
    previous = load_pilot_data_config(Path("configs/data/cva_chart_pilot_v0_2.yaml"))
    legacy = load_pilot_data_config(Path("configs/data/cva_chart_pilot.yaml"))

    assert config.dataset_id == "CVA-Chart-Pilot-v0.3"
    assert config.seed == 20260815
    assert config.output_slug == "cva_chart_pilot_v0_3"
    assert config.render_mode == "axis_scale_v0_3"
    assert previous.dataset_id == "CVA-Chart-Pilot-v0.2"
    assert previous.seed == 20260814
    assert previous.output_slug == "cva_chart_pilot_v0_2"
    assert previous.render_mode == "axis_scale_v0_2"
    assert legacy.dataset_id == "CVA-Chart-Pilot-v0.1"
    assert legacy.output_slug == "cva_chart_pilot_v0_1"
    assert legacy.render_mode == "direct_labels_v0_1"
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


def test_active_manifest_rejects_legacy_v0_1_dataset() -> None:
    from compbias.gpu_pilot.execution_gate import _validate_manifest

    legacy_manifest = {
        "schema_version": 1,
        "dataset_id": "CVA-Chart-Pilot-v0.1",
        "record_count": 2_800,
        "split_counts": {
            "calibration": 200,
            "smoke_train": 600,
            "pilot_train": 1_200,
            "dev": 200,
            "iid_test": 300,
            "mechanism_ood": 300,
        },
        "counterfactual_pairs": 150,
        "natural_audit_ids": [f"calibration-{index:06d}" for index in range(150)],
        "records_path": "records.jsonl",
        "records_sha256": "0" * 64,
        "counterfactual_path": "counterfactual_pairs.jsonl",
        "counterfactual_sha256": "1" * 64,
        "images_generated": 2_950,
        "images_sha256": "2" * 64,
    }

    with pytest.raises(RuntimeError, match="registered contract"):
        _validate_manifest(legacy_manifest)


def test_preflight_rejects_low_vram_and_accepts_valid_fixture(tmp_path: Path) -> None:
    from compbias.gpu_pilot.config import PilotPaths
    from compbias.gpu_pilot.preflight import HardwareSnapshot, audit_server

    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "chat_template.json",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ):
        (model / name).write_bytes(b"fixture")
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
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
    first_model_hash = report["model_snapshot_sha256"]
    (model / "extra-loadable.json").write_text("{}", encoding="utf-8")
    changed = audit_server(paths, hardware=valid, free_disk_gib=180.0)
    assert changed["model_snapshot_sha256"] != first_model_hash


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


@pytest.mark.parametrize(
    ("stage", "path"),
    [
        ("pilot_a", Path("configs/train/pilot_a.yaml")),
        ("pilot_b_lm_only", Path("configs/train/pilot_b_lm_only.yaml")),
    ],
)
def test_gpu_stage_configs_bind_only_the_active_v0_3_dataset(stage: str, path: Path) -> None:
    from compbias.gpu_pilot.stages import load_stage_config

    config = load_stage_config(path, stage)

    assert config["data_config"] == "configs/data/cva_chart_pilot_v0_3.yaml"
    assert config["dataset_manifest"] == "data/generated/cva_chart_pilot_v0_3/manifest.json"


def test_gpu_stage_rejects_escaping_output_subdirectory(tmp_path: Path) -> None:
    from compbias.gpu_pilot.stages import load_stage_config

    source = yaml.safe_load(Path("configs/train/pilot_a.yaml").read_text(encoding="utf-8"))
    source["output_subdir"] = "../../outside"
    config = tmp_path / "pilot_a.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ValueError, match="output_subdir"):
        load_stage_config(config, "pilot_a")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_steps", 1_000_000_000, "registered budget"),
        ("learning_rate", float("nan"), "registered budget"),
        ("num_generations", True, "registered budget"),
    ],
)
def test_gpu_stage_rejects_unregistered_training_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    from compbias.gpu_pilot.stages import load_stage_config

    source = yaml.safe_load(Path("configs/train/pilot_a.yaml").read_text(encoding="utf-8"))
    source["training"][field] = value
    config = tmp_path / "pilot_a.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_stage_config(config, "pilot_a")


def test_gpu_stage_cannot_execute_without_smoke_and_calibration_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from compbias.gpu_pilot import stages, training

    config = {
        "schema_version": 1,
        "stage": "pilot_a",
        "paths_config": "configs/paths.yaml",
        "model_config": "configs/model/qwen25vl3b.yaml",
        "data_config": "configs/data/cva_chart_pilot_v0_2.yaml",
        "dataset_manifest": "data/generated/cva_chart_pilot_v0_2/manifest.json",
        "natural_records": "trajectories/natural/pilot_train_records.jsonl",
        "output_subdir": "pilot_a",
        "training": {},
        "freeze": {},
        "claims": {},
    }
    paths = SimpleNamespace(
        project_root=tmp_path,
        model_path=tmp_path / "model",
        outputs=tmp_path / "outputs",
        trajectories=tmp_path / "trajectories",
    )
    called = False

    def fake_training(_config: dict[str, object]) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(stages, "load_stage_config", lambda _path, _stage: config)
    monkeypatch.setattr(stages, "load_pilot_paths", lambda _path: paths, raising=False)
    monkeypatch.setattr(training, "run_grpo_stage", fake_training)
    monkeypatch.setenv(
        "COMPBIAS_GPU_EXECUTION_ACK",
        "I_UNDERSTAND_THIS_STARTS_GPU_TRAINING",
    )

    exit_code = stages.main_for_stage(
        "pilot_a",
        ["--config", str(tmp_path / "pilot_a.yaml"), "--execute"],
    )

    assert exit_code == 3
    assert called is False


def test_gpu_execution_gate_accepts_only_complete_bound_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from compbias.gpu_pilot import execution_gate
    from compbias.gpu_pilot.config import PilotPaths
    from compbias.gpu_pilot.preflight import model_snapshot_sha256
    from compbias.models.structured_parser import parse_trajectory

    paths = PilotPaths(
        project_root=tmp_path,
        model_path=(tmp_path / "model").resolve(),
        data=(tmp_path / "data").resolve(),
        outputs=(tmp_path / "outputs").resolve(),
        checkpoints=(tmp_path / "checkpoints").resolve(),
        trajectories=(tmp_path / "trajectories").resolve(),
        cache=(tmp_path / "cache").resolve(),
    )
    for path in paths.to_mapping().values():
        Path(path).mkdir(parents=True, exist_ok=True)
    for name in (
        "chat_template.json",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ):
        (paths.model_path / name).write_bytes(b"fixture")
    (paths.model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    model_hash = model_snapshot_sha256(paths.model_path)

    raw = (
        '<perception>{"values":[3,7,5]}</perception>'
        '<reasoning>{"operation":"max_minus_min"}</reasoning><answer>4</answer>'
    )
    parsed = parse_trajectory(raw, sample_id="smoke-000001").to_mapping()
    reports = {
        paths.outputs / "preflight" / "report.json": {
            "schema_version": 1,
            "artifact_type": "compbias_gpu_pilot_preflight",
            "ready": True,
            "large_gpu_started": False,
            "hardware": {
                "cuda_available": True,
                "device_name": "NVIDIA GeForce RTX 4090",
                "total_vram_gib": 47.37,
                "bf16_supported": True,
                "torch_version": "2.8.0+cu128",
                "torch_cuda_runtime": "12.8",
            },
            "free_disk_gib": 180.0,
            "model_path": str(paths.model_path),
            "model_snapshot_sha256": model_hash,
            "storage": paths.to_mapping(),
        },
        paths.outputs / "smoke" / "smoke_report.json": {
            "schema_version": 1,
            "artifact_type": "qwen25vl3b_offline_smoke",
            "training_invoked": False,
            "model_path": str(paths.model_path),
            "model_snapshot_sha256": model_hash,
            "expected_answer": 4,
            "raw_response": raw,
            "parsed": parsed,
            "format_attempts": [
                {
                    "attempt_index": 0,
                    "raw_text": raw,
                    "status": "ok",
                    "error_code": None,
                }
            ],
            "format_retries": 0,
            "format_passed": True,
            "smoke_passed": True,
            "answer_correct": True,
            "latency_seconds": 3.0,
            "peak_memory_gib": 8.0,
        },
        paths.trajectories / "natural" / "calibration_records.summary.json": {
            "schema_version": 1,
            "split": "calibration",
            "records": 200,
            "answer_accuracy": 0.55,
            "parse_rate": 1.0,
            "natural_perception_error_rate": 0.4,
            "error_counts": {
                "none": 80,
                "visual_error": 40,
                "reasoning_error": 40,
                "compensated_visual_error": 40,
            },
            "output": str(paths.trajectories / "natural" / "calibration_records.jsonl"),
            "gate_failures": [],
            "gate_passed": True,
        },
        tmp_path / "data" / "generated" / "cva_chart_pilot_v0_2" / "manifest.json": {
            "schema_version": 1,
            "dataset_id": "CVA-Chart-Pilot-v0.2",
            "record_count": 2_800,
            "split_counts": {
                "calibration": 200,
                "smoke_train": 600,
                "pilot_train": 1_200,
                "dev": 200,
                "iid_test": 300,
                "mechanism_ood": 300,
            },
            "counterfactual_pairs": 150,
            "natural_audit_ids": [f"calibration-{index:06d}" for index in range(150)],
            "records_path": "records.jsonl",
            "records_sha256": "0" * 64,
            "counterfactual_path": "counterfactual_pairs.jsonl",
            "counterfactual_sha256": "1" * 64,
            "images_generated": 2_950,
            "images_sha256": "2" * 64,
        },
    }
    for path, report in reports.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    manifest_path = tmp_path / "data" / "generated" / "cva_chart_pilot_v0_2" / "manifest.json"
    import hashlib

    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    provenance = {
        "model_snapshot_sha256": model_hash,
        "dataset_manifest_sha256": manifest_hash,
        "dataset_images_sha256": "2" * 64,
    }
    calibration_path = (
        paths.outputs.parent / "trajectories" / "natural" / "calibration_records.summary.json"
    )
    calibration_report = {**reports[calibration_path], **provenance}
    calibration_path.write_text(json.dumps(calibration_report), encoding="utf-8")
    (paths.trajectories / "natural" / "calibration_records.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (paths.trajectories / "natural" / "pilot_train_records.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    pilot_summary = {
        **calibration_report,
        "split": "pilot_train",
        "records": 1_200,
        "output": str(paths.trajectories / "natural" / "pilot_train_records.jsonl"),
    }
    (paths.trajectories / "natural" / "pilot_train_records.summary.json").write_text(
        json.dumps(pilot_summary), encoding="utf-8"
    )
    dataset_root = tmp_path / "data" / "generated" / "cva_chart_pilot_v0_2"
    (dataset_root / "records.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset_root / "counterfactual_pairs.jsonl").write_text("{}\n", encoding="utf-8")
    stage_config = tmp_path / "configs" / "train" / "pilot_a.yaml"
    paths_config = tmp_path / "configs" / "paths.yaml"
    model_config = tmp_path / "configs" / "model" / "qwen25vl3b.yaml"
    data_config = tmp_path / "configs" / "data" / "cva_chart_pilot_v0_2.yaml"
    for path in (stage_config, paths_config, model_config, data_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("schema_version: 1\n", encoding="utf-8")
    config = {
        "stage": "pilot_a",
        "paths_config": "configs/paths.yaml",
        "model_config": "configs/model/qwen25vl3b.yaml",
        "data_config": "configs/data/cva_chart_pilot_v0_2.yaml",
        "dataset_manifest": "data/generated/cva_chart_pilot_v0_2/manifest.json",
        "natural_records": "trajectories/natural/pilot_train_records.jsonl",
    }
    derived = {
        "records": 200,
        "answer_accuracy": 0.55,
        "parse_rate": 1.0,
        "natural_perception_error_rate": 0.4,
        "error_counts": {
            "none": 80,
            "visual_error": 40,
            "reasoning_error": 40,
            "compensated_visual_error": 40,
        },
    }
    monkeypatch.setattr(execution_gate, "_validate_dataset_bundle", lambda *_args: {})
    monkeypatch.setattr(execution_gate, "_validate_canonical_dataset", lambda *_args: None)
    monkeypatch.setattr(execution_gate, "_validate_registered_data_config", lambda *_args: None)
    monkeypatch.setattr(execution_gate, "_validate_model_config", lambda *_args: None)
    monkeypatch.setattr(
        execution_gate,
        "_validate_natural_records",
        lambda *_args, **_kwargs: derived,
    )

    hashes = execution_gate.validate_execution_evidence(
        config,
        paths,
        stage_config_path=stage_config,
        paths_config_path=paths_config,
    )
    assert set(hashes) == {
        "preflight",
        "smoke",
        "calibration",
        "dataset_manifest",
        "stage_config",
        "paths_config",
        "model_config",
        "model_snapshot",
        "data_config",
        "natural_records",
        "natural_records_summary",
        "dataset_records",
        "dataset_counterfactuals",
        "calibration_records",
    }
    assert all(len(value) == 64 for value in hashes.values())

    smoke_path = paths.outputs / "smoke" / "smoke_report.json"
    failed_smoke = {**reports[smoke_path], "smoke_passed": False}
    smoke_path.write_text(json.dumps(failed_smoke), encoding="utf-8")
    with pytest.raises(RuntimeError, match="successful known-answer"):
        execution_gate.validate_execution_evidence(
            config,
            paths,
            stage_config_path=stage_config,
            paths_config_path=paths_config,
        )


def test_public_configuration_contains_no_online_model_identifier_as_path() -> None:
    model = yaml.safe_load(Path("configs/model/qwen25vl3b.yaml").read_text(encoding="utf-8"))
    assert model["local_files_only"] is True
    assert model["trust_remote_code"] is False
    assert model["path"] == "/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct"
    forbidden = {"token", "api_key", "password", "secret"}
    assert not forbidden.intersection(model)
