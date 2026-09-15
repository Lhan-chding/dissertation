import json
from types import SimpleNamespace

import numpy as np

from src.modeling_contrast.observation_study import event_validity_view
from src.modeling_contrast.study import metric_rows


def test_raw_validity_keeps_mass_residual_instead_of_silently_using_control_variate():
    raw = np.array([[[0.1, 0.2, 0.3, -0.4]]])
    assert event_validity_view(raw, "raw_event").item() == 0.1 + 0.2 + 0.3
    assert event_validity_view(raw, "helmert").item() == 0.4


def test_active_population_is_separate_for_three_frozen_fit_targets():
    truth = np.zeros((4, 2, 72, 4))
    truth[:, :, :, 0] = 0.001
    truth[:, :, :, 3] = -0.001
    groups = np.repeat(np.arange(6), 12)
    active = np.broadcast_to([False, True, False], (4, 3))
    rows = metric_rows(truth, truth, groups, {}, active)
    assert {r["target"] for r in rows if r["population"] == "active"} == {
        "no_x_off_1_minus_joint_0"
    }
    assert len([r for r in rows if r["population"] == "all"]) == 3


def test_finite_unit_seals_before_oracle_and_does_not_select_on_query_excitation(
    tmp_path, monkeypatch
):
    import src.modeling_contrast.parent as parent_module
    from src.modeling_contrast.model_study import fit_unit

    entry = {
        "id": "fixture",
        "seed": 201,
        "arm": "X_BASE",
        "split": "model_selection",
        "sha256": {"observations_file": "a" * 64},
    }
    theta = np.zeros((3, 14, 3, 737))
    theta[:, 10:14, 1:, 0] = 0.1
    observation = {"anchors": np.array([8, 24, 40]), "branch_theta": theta}
    monkeypatch.setattr(
        parent_module, "load_parent", lambda *args: SimpleNamespace(observations=observation)
    )
    meta = [{"prompt_id": str(i), "group": i // 12, "weight": 1 / 72} for i in range(72)]
    (tmp_path / "probe_metadata.json").write_text(json.dumps(meta))
    out = tmp_path / "fit"

    def trap(*args):
        assert (out / "predictions.npz").exists()
        assert (out / "freeze.json").exists()
        p = np.full((3, 14, 3, 72, 4), 0.25)
        p[:, 10:14, 1:, :, 0] += 0.001
        p[:, 10:14, 1:, :, 3] -= 0.001
        return {"branch_p": p}

    monkeypatch.setattr(parent_module, "load_parent_oracle", trap)
    packet = {
        "helmert": np.zeros((8, 14, 2, 72, 3)),
        "covariance": np.zeros((8, 14, 72, 6, 6)),
        "metadata": {
            "method": "O_CRN",
            "n": 64,
            "packets": [
                {
                    "bank": b,
                    "noise_replica": noise,
                    "packet_id": f"{b}_{noise}",
                    "policy_fingerprints": [(str(b), str(b)), (str(b), str(b))],
                }
                for noise in range(8)
                for b in range(8)
            ],
        },
    }
    cfg = {
        "stage": "N2B",
        "method": "C2",
        "observation": "O_CRN",
        "n": 64,
        "rank": "FULL",
        "alpha": 1e-5,
        "eta": 0.01,
        "fit_banks": 8,
    }
    rows = fit_unit(tmp_path, entry, 0, [cfg], {("O_CRN", 64): packet}, out)
    assert all(r["model_status"] == "NO_CONTRAST_EXCITATION" for r in rows)
    assert all(r["population"] == "all" for r in rows)
    assert all(r["predicted_difference_norm"] == 0 for r in rows)
