import copy
from pathlib import Path

import pytest

from src.modeling_v4.config import atomic_json, digest, load_config, plan, validate_config

CONFIG = Path(__file__).resolve().parents[2] / "configs/modeling_v4/protocol.json"


def test_plan_has_four_real_development_origins_without_cpu_gate():
    config = load_config(CONFIG)
    before = copy.deepcopy(config)
    result = plan(config)
    assert [(r["seed"], r["step"]) for r in result["first_map_origins"]] == [
        (41001, 32), (41001, 96), (41002, 32), (41002, 96)
    ]
    assert result["stage_dependencies"]["B"] == []
    assert result["gpu_executed"] is False and result["measured_runtime"] is None
    assert config == before


@pytest.mark.parametrize("change", ["overlap", "gpu", "online", "rho", "reference", "query_fit"])
def test_config_rejects_identity_or_scope_violations(change):
    config = load_config(CONFIG)
    if change == "overlap":
        config["qwen"]["seed_roles"]["test"][0] = 41001
    elif change == "gpu":
        config["operations"]["max_concurrent_gpus"] = 4
    elif change == "online":
        config["operations"]["online_ssvc"] = True
    elif change == "rho":
        config["statistics"]["fixed_rho_gate"] = 0.05
    elif change == "reference":
        config["observations"]["separate_reference_rng"] = False
    else:
        config["models"]["query_outcomes_for_fit"] = True
    with pytest.raises(ValueError):
        validate_config(config)


def test_atomic_json_failure_preserves_previous_document(tmp_path):
    path = tmp_path / "run.json"
    atomic_json(path, {"complete": True})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {"not_finite": float("nan")})
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.pending")) == []
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
