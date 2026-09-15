import hashlib
import json

import pytest

from src.modeling_contrast import server_cli


def test_explicit_cpu_allocation_required(monkeypatch):
    monkeypatch.setattr(server_cli.sys, "platform", "linux")
    monkeypatch.setattr(server_cli.platform, "node", lambda: "cpu-2")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="allocation"):
        server_cli.require_cpu_allocation()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    server_cli.require_cpu_allocation()
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(ValueError, match="GPU"):
        server_cli.require_cpu_allocation()
    monkeypatch.delenv("SLURM_JOB_GPUS")
    monkeypatch.setattr(server_cli.platform, "node", lambda: "login-1")
    with pytest.raises(ValueError, match="CPU compute"):
        server_cli.require_cpu_allocation()


def test_server_forecast_retains_slowest_rate():
    budget = {"forecast": {"wall_seconds": 100, "peak_ram_gib": 2, "added_output_bytes": 99}}
    old = {"observation_wall_seconds": 2, "fit_wall_seconds": 1}
    server = {
        "observation_wall_seconds": 4,
        "fit_wall_seconds": 0.5,
        "forecast": {"peak_ram_gib": 3},
    }
    actual = server_cli.server_forecast(budget, old, server)
    assert actual["wall_seconds"] == 200
    assert actual["peak_ram_gib"] == 3
    assert actual["added_output_bytes"] == 99
    assert budget["forecast"]["wall_seconds"] == 100
    with pytest.raises(ValueError, match="positive"):
        server_cli.server_forecast(budget, {}, server)


def test_preflight_completion_requires_matching_selection_and_positive_time(tmp_path):
    stage = tmp_path / "run/N3_server"
    control = tmp_path / "control/preflight_123"
    stage.mkdir(parents=True)
    control.mkdir(parents=True)
    (stage / "SERVER_RESOURCE_PREFLIGHT.json").write_text('{"job_id":"123"}')
    (stage / "RUN_MANIFEST.json").write_bytes(b"frozen manifest")
    receipt = {
        "status": "COMPLETE",
        "wall_seconds": 87.0,
        "selection_manifest_sha256": hashlib.sha256(b"frozen manifest").hexdigest(),
    }
    (control / "COMPLETE.json").write_text(json.dumps(receipt))
    settings = {"control_root": str(tmp_path / "control")}
    assert server_cli._preflight_completion(settings, tmp_path / "run")["wall_seconds"] == 87
    (stage / "RUN_MANIFEST.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="preflight timing"):
        server_cli._preflight_completion(settings, tmp_path / "run")


def test_source_catalog_covers_dataset_generation_and_requirements(tmp_path, monkeypatch):
    root = tmp_path / "ssvc_flow"
    (root / "src/modeling_contrast").mkdir(parents=True)
    (root / "src/generate_worlds.py").write_text("frozen dataset generator")
    (root / "src/modeling_contrast/server_cli.py").write_text("frozen driver")
    (root / "requirements-cpu.txt").write_text("numpy")
    monkeypatch.setattr(server_cli, "ROOT", root)
    hashes = server_cli._source_catalog({"source_hashes": {}}, root)
    assert set(hashes) == {
        str(root / "src/generate_worlds.py"),
        str(root / "src/modeling_contrast/server_cli.py"),
        str(root / "requirements-cpu.txt"),
    }
