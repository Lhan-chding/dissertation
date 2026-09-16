import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3.io import atomic_json, atomic_npz, canonical_hash, finalize_run, sha256_file
from src.modeling_v4.signature_fit import (
    fit_signature_development,
    freeze_signature_models,
    load_signature_model,
    predict_signature_models,
    save_signature_model,
    verify_selected_signature_model,
)


def bound(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def originals(tmp_path):
    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v4/protocol.json").read_text()
    )
    inputs = []
    rng = np.random.default_rng(14)
    for seed in (41001, 41002, 41003):
        for step in (32, 96):
            origin = f"{seed}_X_BASE_{step}"
            root = tmp_path / origin
            root.mkdir()
            units = []
            provenance = []
            for bank in range(2):
                for j in range(3):
                    row = {
                        "origin_id": origin,
                        "bank_id": f"b{bank}",
                        "contrast_id": f"c{j}",
                        "candidate": "joint_1",
                        "baseline": "joint_0",
                        "candidate_fingerprint": f"{origin}-b{bank}-c{j}",
                        "baseline_fingerprint": origin + "-base",
                        "pair_id": f"b{bank}-c{j}",
                        "seed": seed,
                        "arm": "X_BASE",
                        "step": step,
                        "role": "development",
                        "bank_role": "calibration",
                        "is_alias": False,
                    }
                    units.append(row)
                    provenance.append(
                        {
                            "native_width": 576,
                            "prompts": 36,
                            "draws_per_prompt": 16,
                            "origin_id": origin,
                            "panel_role": "signature",
                            "panel_id": origin,
                            "split_guard": "VERIFIED",
                            "score_definition": "FULL_SEQUENCE_LOGPROB_DIFFERENCE",
                            "target_outcomes_used": False,
                            "paid_feature": True,
                            "sample_identity_sha256": "a" * 64,
                            "split_identity_sha256": "b" * 64,
                            "candidate_fingerprint": row["candidate_fingerprint"],
                            "baseline_fingerprint": row["baseline_fingerprint"],
                        }
                    )
            e = rng.normal(size=(6, 576)) * 0.1
            y = np.zeros((6, 2, 4))
            y[:, :, 0] = e[:, :2]
            y[:, :, 3] = -e[:, :2]
            runtime = {
                "fixture": True,
                "source_hash": "fixture",
                "config_hash": canonical_hash(config),
            }
            sig = root / "signature"
            sig.mkdir()
            atomic_npz(
                sig / "SIGNATURE_FEATURES.npz",
                {
                    "native_signatures": e,
                    "EVENT_ONLY": np.full((36, 4), 0.25),
                    "EVENT_PLUS_DIAGNOSTICS": np.full((36, 11), 0.25),
                },
            )
            atomic_json(sig / "SIGNATURE_PROVENANCE.json", provenance)
            atomic_json(
                sig / "ORIGIN_STATE.json",
                {
                    "stage": "ORIGIN",
                    "origin_id": origin,
                    "panel_role": "signature",
                    "panel_id": origin,
                },
            )
            receipt = {
                "kind": "V4_NATIVE_SIGNATURE_FEATURES",
                "origin_id": origin,
                "runtime_identity": runtime,
                "units": [
                    {k: r[k] for k in ("bank_id", "contrast_id", "candidate", "baseline")}
                    for r in units
                ],
                "arrays": bound(sig / "SIGNATURE_FEATURES.npz"),
                "origin_state": bound(sig / "ORIGIN_STATE.json"),
                "query_endpoint_target_labels_read": False,
                "reference_labels_read": False,
                "artifact_hashes": {p.name: sha256_file(p) for p in sig.iterdir()},
            }
            atomic_json(sig / "COMPLETE.json", receipt)
            label = root / "labels"
            label.mkdir()
            atomic_npz(label / "LABELS.npz", {"RAW4": y})
            labels = {
                "kind": "V4_CALIBRATION_LABELS",
                "config_hash": canonical_hash(config),
                "source_hash": "fixture",
                "origin_id": origin,
                "seed": seed,
                "arm": "X_BASE",
                "step": step,
                "role": "development",
                "bank_role_scope": "calibration",
                "calibration_bank_count": 2,
                "draws": 1024,
                "arrays": bound(label / "LABELS.npz"),
                "units": units,
                "prompt_ids": ["p0", "p1"],
                "probe_groups": ["g0", "g1"],
                "observation_methods": ["RAW4"],
                "event_order": ["X", "S", "W", "I"],
                "query_labels_read": False,
                "reference_labels_read": False,
            }
            atomic_json(label / "LABELS.json", labels)
            finalize_run(label, {"fixture": True})
            inputs.append(
                {"signature": bound(sig / "COMPLETE.json"), "labels": bound(label / "LABELS.json")}
            )
    return config, inputs


def test_actual_fold_receipts_gate_and_frozen_safe_independent_predictions(tmp_path):
    config, inputs = originals(tmp_path)
    specs = [
        {"id": "linear", "model": "FULL_DUAL_RIDGE", "alpha": 1e-5},
        {"id": "rbf", "model": "FULL_RBF_RAW", "alpha": 1e-5, "bandwidth_multiplier": 1.0},
        {
            "id": "mlp256",
            "model": "NONLINEAR_SIGNATURE_RESIDUAL",
            "hidden_width": 256,
            "state_mode": "EVENT_ONLY",
        },
    ]
    report = fit_signature_development(
        config,
        inputs,
        out=tmp_path / "fit",
        model_specs=specs,
        observation_methods=["RAW4"],
        epochs=1,
        fixture=True,
    )
    assert len(report["whole_seed_cv"]) == 3
    assert all(
        len(r["train_seeds"]) == 2 and len(r["validation_seeds"]) == 1
        for r in report["whole_seed_cv"]
    )
    statuses = {r["id"]: r["status"] for r in report["models"]}
    assert statuses["RAW4/mlp256"] == "NOT_ELIGIBLE"
    assert statuses["RAW4/linear"] == "AVAILABLE"
    assert report["query_labels_read"] is False and report["reference_labels_read"] is False
    lock = freeze_signature_models(
        bound(tmp_path / "fit" / "SIGNATURE_CV.json"),
        ["RAW4/linear", "RAW4/rbf"],
        out=tmp_path / "lock",
        fixture=True,
    )
    prediction = predict_signature_models(
        bound(tmp_path / "lock" / "SIGNATURE_SELECTION.json"),
        inputs[0]["signature"],
        out=tmp_path / "prediction",
        fixture=True,
    )
    assert np.load(prediction["arrays"]["path"])["predictions"].shape == (2, 6, 2, 4)
    assert lock["fit_scope"] == "ALL_REGISTERED_DEVELOPMENT_ORIGINS"
    with pytest.raises(ValueError, match="AVAILABLE"):
        freeze_signature_models(
            bound(tmp_path / "fit" / "SIGNATURE_CV.json"),
            ["RAW4/mlp256"],
            out=tmp_path / "badlock",
            fixture=True,
        )


def test_production_requires_server_and_calibration_test_labels_cannot_train(tmp_path, monkeypatch):
    config, inputs = originals(tmp_path)
    with monkeypatch.context() as scope:
        scope.delenv("SLURM_JOB_ID", raising=False)
        with pytest.raises(RuntimeError, match=r"server|Slurm"):
            fit_signature_development(config, inputs, out=tmp_path / "prod")
    labels = json.loads(Path(inputs[0]["labels"]["path"]).read_text())
    labels["role"] = "test"
    path = tmp_path / "forged.json"
    atomic_json(path, labels)
    inputs[0]["labels"] = bound(path)
    with pytest.raises(ValueError, match=r"development|complete|manifest"):
        fit_signature_development(config, inputs, out=tmp_path / "bad", epochs=1, fixture=True)


def test_selected_model_label_and_settings_must_match_actual_saved_method():
    selected = {
        "id": "RAW4/model",
        "spec": {"model": "FULL_DUAL_RIDGE", "alpha": 1e-5},
        "observation_method": "RAW4",
    }
    model = {
        "metadata": {
            "method": "FULL_DUAL_RIDGE",
            "metadata": {
                "method": "FULL_DUAL_RIDGE",
                "alpha": 1e-5,
                "ridge_lambda": 1e-5,
                "rank_cap": "FULL",
                "ridge_override": False,
            },
        }
    }
    verify_selected_signature_model(selected, model)
    wrong = copy.deepcopy(selected)
    wrong["spec"]["model"] = "NONLINEAR_SIGNATURE_RESIDUAL"
    with pytest.raises(ValueError, match="actual frozen"):
        verify_selected_signature_model(wrong, model)
    wrong = copy.deepcopy(selected)
    wrong["spec"]["alpha"] = 0.01
    with pytest.raises(ValueError, match="actual frozen"):
        verify_selected_signature_model(wrong, model)


def test_full_pool_refit_uses_only_development_and_safe_tensor_reload(tmp_path):
    import torch

    from src.modeling_v4.nonlinear import refit_signature_residual

    path = Path(__file__).with_name("test_nonlinear.py")
    spec = importlib.util.spec_from_file_location("nonlinear_test_data", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    e, z, y, records, kwargs = fixture.dataset()
    kwargs.pop("validation_seed")
    model = refit_signature_residual(e, z, y, records, epochs=1, **kwargs)
    assert model["partition"]["training_indices"].size == 768
    assert model["partition"]["validation_indices"].size == 0
    assert model["metadata"]["source_training_seed_count"] == 3
    save_signature_model(model, tmp_path / "model", output_shape=y.shape[1:])
    payload = torch.load(tmp_path / "model" / "WEIGHTS.pt", map_location="cpu", weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in payload["networks"][0].values())
    restored = load_signature_model(tmp_path / "model")
    np.testing.assert_allclose(
        restored["predict"](e[:3], z[:3]), model["predict"](e[:3], z[:3]), atol=1e-12
    )
    bad = copy.deepcopy(records)
    bad[0]["role"] = "test"
    with pytest.raises(ValueError, match="development"):
        refit_signature_residual(e, z, y, bad, epochs=1, **kwargs)


def test_reversed_same_endpoints_cannot_increase_unique_pair_training_gate():
    from src.modeling_v4.nonlinear import validate_training_partition

    path = Path(__file__).with_name("test_nonlinear.py")
    spec = importlib.util.spec_from_file_location("reverse_pair_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    e, z, y, records, kwargs = fixture.dataset()
    first, second = kwargs["signature_provenance"][:2]
    second["baseline_fingerprint"] = first["candidate_fingerprint"]
    second["candidate_fingerprint"] = first["baseline_fingerprint"]
    e[1] = -e[0]
    y[1] = -y[0]
    with pytest.raises(ValueError, match="duplicate endpoint pair"):
        validate_training_partition(e, z, y, records, **kwargs)
