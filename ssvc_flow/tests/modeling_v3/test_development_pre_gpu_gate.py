import pytest

from src.modeling_v3.cpu_campaign import _binding, _finish
from src.modeling_v3.server_preparation import _verify_development_runs


def roots(tmp_path, *, pilot=False):
    config = {
        "cpu": {"repeat_measurements_dev": 200},
        "observation": {
            "total_draws_grid": [64, 256, 1024, 4096],
            "proposals": ["origin", "equal_pair_mixture"],
        },
    }
    q1, q2 = tmp_path / "q1", tmp_path / "q2"
    q1.mkdir()
    q2.mkdir()
    units = [
        {"n": n, "proposal": p}
        for n in config["observation"]["total_draws_grid"]
        for p in config["observation"]["proposals"]
    ]
    _finish(
        q1,
        _binding(config, stage="Q1"),
        {
            "stage": "Q1",
            "pilot": pilot,
            "scientific_status": "DEVELOPMENT_ONLY",
            "units": units,
            "unit_count": len(units),
            "repeat_count_per_unit": 200,
        },
    )
    _finish(
        q2,
        _binding(config, stage="Q2"),
        {
            "stage": "Q2",
            "pilot": False,
            "scientific_status": "DEVELOPMENT_ONLY",
            "results": [{"model_beats_direct": False}],
            "fit_count": 1,
        },
    )
    return config, [q1, q2]


def test_actual_development_receipts_required_and_pilot_rejected(tmp_path):
    with pytest.raises(ValueError, match="Q1 and Q2"):
        _verify_development_runs({}, None)
    config, paths = roots(tmp_path, pilot=True)
    with pytest.raises(ValueError, match="actual development"):
        _verify_development_runs(config, paths)


def test_development_gate_does_not_require_model_to_beat_direct_baseline(tmp_path):
    config, paths = roots(tmp_path)
    result = _verify_development_runs(config, paths)
    assert set(result) == {"Q1", "Q2"}
    assert not result["Q2"]["comparison_success_required"]
    assert result["Q1"]["sha256"]
