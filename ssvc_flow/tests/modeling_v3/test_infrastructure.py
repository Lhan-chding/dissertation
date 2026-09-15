import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3.io import AttemptStore, verify_manifest
from src.modeling_v3.statistics import conformal_radius, error_metrics, holm, seed_bootstrap


def test_immutable_attempt_resume_and_tamper(tmp_path):
    out = tmp_path / "unit"
    binding = {"source": "s", "config": "c", "data": "d"}
    with AttemptStore(out, binding) as store:
        store.record("request/one", {"value": 3})
    with pytest.raises(FileExistsError), AttemptStore(out, binding):
        pass
    with (
        pytest.raises(ValueError, match="binding"),
        AttemptStore(out, {**binding, "source": "changed"}, resume=True),
    ):
        pass
    with AttemptStore(out, binding, resume=True) as store:
        assert store.record("request/one", {"value": 3}) is False
        with pytest.raises(ValueError, match="conflict"):
            store.record("request/one", {"value": 4})
        store.finish(expected_keys=["request/one"])
    assert verify_manifest(out)["status"] == "COMPLETE"
    path = next((out / "records").glob("*.json"))
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        verify_manifest(out)


def test_worker_and_incomplete_keys_rejected(tmp_path):
    with AttemptStore(tmp_path / "unit", {"a": 1}) as store:
        with (
            pytest.raises(RuntimeError, match="writer"),
            AttemptStore(tmp_path / "unit", {"a": 1}, resume=True),
        ):
            pass
        with pytest.raises(ValueError, match="request"):
            store.finish(expected_keys=["missing"])


def test_statistics_keep_undefined_and_seed_units():
    assert error_metrics([1, 2], [0, 0])["nrmse"] is None
    score = error_metrics([2, 4], [1, 2])
    assert score["nrmse"] == 1.0
    assert score["q95_absolute_residual"] == pytest.approx(1.95)
    assert conformal_radius([1, 2, 3], nominal=0.95)["status"] == "INSUFFICIENT_CALIBRATION_SEEDS"
    assert conformal_radius(list(range(20)), nominal=0.95)["radius"] == 19
    np.testing.assert_allclose(holm([0.01, 0.03, 0.2, 0.04]), [0.04, 0.09, 0.2, 0.09])
    result = seed_bootstrap([1, 1, 2, 2], [1.0, 1.0, 3.0, 3.0], reps=30, seed=4)
    assert result["independent_seeds"] == 2
    assert result["estimate"] == 2.0
    assert result["unit"] == "training_seed"


def test_resume_after_manifest_before_complete(tmp_path, monkeypatch):
    import src.modeling_v3.io as mod

    original = mod.atomic_json

    def fail_completion(path, value):
        if str(path).endswith("COMPLETE.json"):
            raise OSError("injected process interruption")
        return original(path, value)

    monkeypatch.setattr(mod, "atomic_json", fail_completion)
    with AttemptStore(tmp_path / "unit", {"source": "same"}) as store:
        store.record("one", {"n": 1})
        with pytest.raises(OSError, match="interruption"):
            store.finish(["one"])
    monkeypatch.setattr(mod, "atomic_json", original)
    with AttemptStore(tmp_path / "unit", {"source": "same"}, resume=True) as store:
        store.finish(["one"])
    assert (tmp_path / "unit/COMPLETE.json").is_file()


def test_unknown_string_is_not_an_acceptance_mask():
    from src.modeling_v3.statistics import risk_coverage

    with pytest.raises(ValueError, match="boolean"):
        risk_coverage([1, 2], ["UNKNOWN", "IDENTIFIED"])


def test_selection_cannot_freeze_without_completed_development(tmp_path):
    from src.modeling_v3.schema import freeze_selection

    with pytest.raises(ValueError, match=r"Q1.*Q2"):
        freeze_selection({}, [], {}, tmp_path / "lock")


@pytest.mark.parametrize("fault", ["different_config", "unregistered_file", "crossfit_gls"])
def test_selection_verifies_protocol_inventory_and_supported_combination(tmp_path, fault):
    from src.modeling_v3.cpu_campaign import _binding, _finish
    from src.modeling_v3.schema import freeze_selection

    config = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    selected = {
        "observation_methods": ["PRESERVE_XI"],
        "selection_rules": ["BLOCK_LOGDET"],
        "models": ["FULL_RIDGE"],
        "rank_caps": [1, "FULL"],
        "alpha": 0.01,
        "rho_threshold": 0.1,
        "leverage_threshold": 1.0,
    }
    roots = []
    for stage in ("Q1", "Q2"):
        root = tmp_path / stage
        root.mkdir()
        binding_config = (
            {**config, "changed_protocol": True} if fault == "different_config" else config
        )
        _finish(
            root,
            _binding(binding_config, stage=stage),
            {"stage": stage, "scientific_status": "DEVELOPMENT_ONLY"},
        )
        roots.append(root)
    if fault == "unregistered_file":
        (roots[0] / "unregistered.json").write_text("{}")
    if fault == "crossfit_gls":
        selected["observation_methods"] = ["CROSSFIT_COV_ZERO_SUM"]
        selected["models"] = ["FULL_GLS"]
    with pytest.raises(ValueError):
        freeze_selection(config, roots, selected, tmp_path / "lock")
