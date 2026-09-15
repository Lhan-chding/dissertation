"""Frozen plan, metadata integrity, and CPU boundary regressions."""

import copy
from pathlib import Path

import pytest

from src.modeling_qualification.io import canonical_hash, new_output, source_hashes
from src.modeling_qualification.protocol import cpu_environment, load_config, validate_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/modeling_qualification/protocol.json"


def test_locked_protocol_and_derived_workload():
    config = load_config(CONFIG)
    assert validate_config(config)["total_optimizer_steps"] == 11400
    assert validate_config(config)["total_sampled_finite_actions"] == 203520


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("scope", "new_qwen_calls", True),
        ("scope", "online_detector", True),
        ("toy_model", "dtype", "float32"),
        ("resources", "cpu_threads", 8),
        ("model_selection", "response_nrmse_max", 9.0),
    ],
)
def test_protocol_rejects_silent_plan_drift(section, key, value):
    config = copy.deepcopy(load_config(CONFIG))
    config[section][key] = value
    with pytest.raises(ValueError, match="protocol"):
        validate_config(config)


def test_reject_cross_split_seed_and_event_column_drift():
    config = copy.deepcopy(load_config(CONFIG))
    config["profiles"]["core"]["seeds_by_role"]["locked_test"][0] = 201
    with pytest.raises(ValueError):
        validate_config(config)
    config = copy.deepcopy(load_config(CONFIG))
    config["events"] = ["S", "X", "W", "I"]
    with pytest.raises(ValueError):
        validate_config(config)


def test_cpu_environment_is_explicit(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cpu_environment()
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"


def test_no_clobber_and_hash_contracts(tmp_path):
    target = tmp_path / "run"
    new_output(target)
    (target / "existing").write_text("preserved")
    with pytest.raises(FileExistsError):
        new_output(target)
    assert (target / "existing").read_text() == "preserved"
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        canonical_hash({"nan": float("nan")})
    assert source_hashes(tmp_path)["run/existing"]


def test_core_requires_measured_unchanged_preflight(tmp_path):
    from src.modeling_qualification.cli import _preflight
    from src.modeling_qualification.io import sha256_file, write_json

    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="RESOURCE_REVIEW_REQUIRED"):
        _preflight(config, tmp_path / "M2")
    smoke = tmp_path / "smoke.json"
    write_json(smoke, {"wall_seconds": 2.0})
    preflight = {
        "status": "PASS",
        "config_hash": canonical_hash(config),
        "estimated_total_wall_seconds": 120,
        "estimated_peak_ram_gib": 1,
        "estimated_total_output_bytes": 1000,
        "measurement_sources": {"smoke.json": sha256_file(smoke)},
    }
    write_json(tmp_path / "preflight.json", preflight)
    assert _preflight(config, tmp_path / "M2")["status"] == "PASS"
    preflight["estimated_peak_ram_gib"] = 9
    write_json(tmp_path / "preflight.json", preflight)
    with pytest.raises(ValueError, match="exceeds budget"):
        _preflight(config, tmp_path / "M2")
    preflight["estimated_peak_ram_gib"] = 1
    write_json(tmp_path / "preflight.json", preflight)
    write_json(smoke, {"wall_seconds": 3.0})
    with pytest.raises(ValueError, match="changed smoke"):
        _preflight(config, tmp_path / "M2")
