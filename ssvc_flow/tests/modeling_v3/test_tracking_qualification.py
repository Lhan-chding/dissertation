"""Tiny entrance tests. Mock verifiers never constitute real Q6 qualification."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3 import tracking_offline as tracking
from src.modeling_v3.io import atomic_json, atomic_npz, canonical_hash, finalize_run, sha256_file


def config():
    path = Path(__file__).parents[2] / "configs/modeling_v3/protocol.json"
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(tracking.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SLURM_JOB_ID", "tiny-mocked-qualification-test")


def test_real_entry_requires_config_and_server_cpu(server, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="protocol config"):
        tracking.track_offline({}, tmp_path / "out")
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError, match="server CPU"):
        tracking.track_offline({}, tmp_path / "out", config=config())
    assert not (tmp_path / "out").exists()


def test_handwritten_qualification_and_arrays_cannot_enter_real_tracking(server, tmp_path):
    for key in ("qualification", "basis", "coefficients", "updates", "anchor_probability"):
        with pytest.raises(ValueError, match="handwritten"):
            tracking.track_offline({key: {"pointwise_qualified": True}}, config=config())
    with pytest.raises(ValueError, match="bound pointwise"):
        tracking.track_offline({}, tmp_path / "out", config=config())


def test_verifier_receives_exact_bindings_and_failure_is_not_swallowed(server, monkeypatch):
    seen = []
    payload = {
        "qualification_binding": {"path": "q", "sha256": "qhash"},
        "fit_binding": {"path": "fit", "sha256": "fhash"},
        "reference_binding": {"path": "never-open-reference", "sha256": "bad"},
    }

    def verify(c, q, *, fit_binding):
        seen.append((c, q, fit_binding))
        return {"kind": "V3_VLM_POINTWISE_QUALIFICATION", "pointwise_qualified": False}

    monkeypatch.setattr(tracking, "_verify_qualification", verify)
    result = tracking.track_offline(payload, config=config())
    assert result["status"] == "NOT_QUALIFIED_FOR_TRACKING"
    assert result["reference_inputs_consumed"] is False
    assert seen == [(config(), payload["qualification_binding"], payload["fit_binding"])]

    def invalid(*args, **kwargs):
        raise ValueError("original qualification hash mismatch")

    monkeypatch.setattr(tracking, "_verify_qualification", invalid)
    with pytest.raises(ValueError, match="original qualification hash mismatch"):
        tracking.track_offline(payload, config=config())


def test_no_95_claim_or_wrong_fit_binding_passes_verified_gate(server, monkeypatch):
    payload = {
        "qualification_binding": {"path": "q", "sha256": "qhash"},
        "fit_binding": {"path": "fit", "sha256": "fhash"},
    }
    record = {
        "kind": "V3_VLM_POINTWISE_QUALIFICATION",
        "status": "QUALIFIED_EMPIRICALLY",
        "pointwise_qualified": True,
        "fit_hash": "fhash",
        "qualified_design_ids": ["d"],
        "design_id": "d",
        "cross_seed_95": "NOT_CERTIFIED",
        "online_ssvc": "NOT_CERTIFIED",
    }
    for key, value in (
        ("fit_hash", "other"),
        ("design_id", "other"),
        ("cross_seed_95", "CERTIFIED"),
        ("online_ssvc", "CERTIFIED"),
    ):
        invalid = {**record, key: value}
        monkeypatch.setattr(tracking, "_verify_qualification", lambda *a, _r=invalid, **k: _r)
        with pytest.raises(ValueError, match="fit/design/certification"):
            tracking.track_offline(payload, config=config())


def model_original(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "source_identity", lambda: {"sha256": "fixed-source"})
    c = config()
    fit_root, model_root = tmp_path / "fit", tmp_path / "model"
    fit_root.mkdir()
    model_root.mkdir()
    fit = {
        "kind": "V3_Q5_RESPONSE_FIT",
        "fixture": False,
        "source_hash": "fixed-source",
        "config_hash": canonical_hash(c),
        "arrays": {"path": "fit-arrays", "sha256": "actual-calibration"},
        "vector_identity": {"dimension": 2, "parameter_order": ["weight"]},
        "probe_ids": ["probe"],
        "origin_id": "101_X_BASE_64",
    }
    atomic_json(fit_root / "FIT_SPEC.json", fit)
    fit_binding = binding(fit_root / "FIT_SPEC.json")
    inputs = {"spec_sha256": fit_binding["sha256"], "arrays_sha256": "actual-calibration"}
    atomic_npz(
        model_root / "MODEL_ARRAYS.npz",
        {
            "Q": np.eye(2),
            "directions": np.array([[0.0], [1.0]]),
            "coefficients": np.array([[0.1, -0.1, 0.0, 0.0]]),
        },
    )
    model = {
        "input_binding": inputs,
        "fit_specification": fit,
        "method": "PCA",
        "k": 2,
        "r": 1,
        "array_payload": {
            "path": "MODEL_ARRAYS.npz",
            "sha256": sha256_file(model_root / "MODEL_ARRAYS.npz"),
        },
    }
    atomic_json(model_root / "MODEL.json", model)
    atomic_json(
        model_root / "RECEIPT.json",
        {"command": "fit", "input_binding": inputs, "config_sha256": canonical_hash(c)},
    )
    finalize_run(model_root, {**inputs, "source": "fixed-source", "config": canonical_hash(c)})
    return c, fit_binding, binding(model_root / "MODEL.json")


def test_actual_model_directions_and_fit_bytes_are_used(tmp_path, monkeypatch):
    c, fit, model = model_original(tmp_path, monkeypatch)
    spec, basis, coef = tracking._tracking_model(c, fit, model)
    np.testing.assert_array_equal(basis, [[0.0], [1.0]])
    np.testing.assert_array_equal(coef, [[0.1, -0.1, 0.0, 0.0]])
    assert spec["origin_id"].endswith("_64")
    altered = copy.deepcopy(fit)
    altered["sha256"] = "other-fit"
    with pytest.raises(ValueError, match="missing or changed"):
        tracking._tracking_model(c, altered, model)
    arrays = Path(model["path"]).parent / "MODEL_ARRAYS.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        tracking._tracking_model(c, fit, model)


def test_qualified_pointwise_does_not_mean_dense_tracking_is_available(server, monkeypatch):
    payload = {
        key: {"path": key, "sha256": key + "hash"}
        for key in ("qualification_binding", "fit_binding", "model_binding", "source_binding")
    }
    qualification = {
        "kind": "V3_VLM_POINTWISE_QUALIFICATION",
        "status": "QUALIFIED_EMPIRICALLY",
        "pointwise_qualified": True,
        "fit_hash": payload["fit_binding"]["sha256"],
        "qualified_design_ids": ["d"],
        "tracking_qualified": True,
        "tracking_eligible_design_ids": ["d"],
        "design_id": "d",
        "cross_seed_95": "NOT_CERTIFIED",
        "online_ssvc": "NOT_CERTIFIED",
    }
    monkeypatch.setattr(tracking, "_verify_qualification", lambda *a, **k: qualification)
    fit = {
        "origin_id": "101_X_BASE_64",
        "probe_ids": ["p"],
        "vector_identity": {"parameter_order": ["weight"]},
    }
    monkeypatch.setattr(tracking, "_tracking_model", lambda *a: (fit, np.eye(2), np.zeros((2, 4))))
    monkeypatch.setattr(tracking, "_source_window", lambda *a: (np.zeros((16, 2)), []))
    result = tracking.track_offline(payload, config=config())
    assert result["status"] == "Q6_ANCHOR_OBSERVATION_REQUIRED"
    assert not result["tracking_executed"]
    assert result["next_command"] == "prepare-q6 --phase anchor"
    payload["reference_binding"] = {"path": "handwritten", "sha256": "not-a-producer"}
    with pytest.raises(ValueError, match="Freeze the Q6 prediction"):
        tracking.track_offline(payload, config=config())


@pytest.mark.parametrize("half_width", [None, 0, [0.01, 0, 0.01, 0.01]])
def test_null_or_zero_real_uncertainty_cannot_masquerade_as_exact(tmp_path, half_width):
    payload = {
        "anchor_probability": [0.25] * 4,
        "updates": [[0.1]],
        "basis": [[1.0]],
        "coefficients": [[0.1, -0.1, 0, 0]],
        "qualification": {
            "pointwise_qualified": True,
            "fit_hash": "fixture",
            "source_hash": "fixture",
        },
    }
    path = tmp_path / "ref.json"
    atomic_json(
        path, {"kind": "REAL_VLM_ESTIMATE", "probabilities": [[0.25] * 4], "half_width": half_width}
    )
    payload["reference_binding"] = binding(path)
    with pytest.raises(ValueError, match="uncertainty"):
        tracking.track_offline(payload, fixture=True)


def test_fixture_cannot_open_production_qualification():
    with pytest.raises(ValueError, match="cannot consume production"):
        tracking.track_offline({"qualification_binding": {"path": "real"}}, fixture=True)


def test_dense_source_uses_all_actual_steps_and_rejects_wrong_origin(tmp_path, monkeypatch):
    import torch

    from src import followup_updates
    from src.modeling_v3 import vlm_campaign

    c = config()
    seed = c["qwen"]["seed_roles"]["locked_test"][0]
    states = {
        step: {
            "parameters": {"weight": torch.tensor([float(step), -float(step)])},
            "metadata": {"seed": seed, "arm": "X_BASE", "checkpoint_step": step},
        }
        for step in range(64, 81)
    }
    records = [
        {
            "step": step,
            "path": str(tmp_path / f"{step}.pt"),
            "checkpoint_identity": {"step": step},
            "checkpoint_sha256": str(step),
            "state_hash": followup_updates.state_hash(states[step]) if step in states else "other",
            "parameter_hash": followup_updates.state_hash(states[step]["parameters"])
            if step in states
            else "other",
            "inference_fingerprint": f"fp{step}",
            "update_norm": np.sqrt(2),
        }
        for step in range(1, 129)
    ]
    source = {
        "identity": {"seed": seed, "arm": "X_BASE"},
        "steps": 128,
        "source_optimizer_steps": 128,
        "retained_all_intermediate_states": True,
        "online_feedback": False,
        "checkpoints": records,
    }
    source_path = tmp_path / "source.json"
    atomic_json(source_path, source)
    source_binding = binding(source_path)
    fork_path = tmp_path / "forks.json"
    atomic_json(fork_path, {"identity": {"config_hash": canonical_hash(c)}})
    atomic_json(
        tmp_path / "bank_plan.json", {"origin_identity": {"state_hash": records[63]["state_hash"]}}
    )
    fit = {
        "origin_id": f"{seed}_X_BASE_64",
        "forks": binding(fork_path),
        "vector_identity": {"parameter_order": ["weight"], "dimension": 2},
    }
    loaded = []

    def load(path, identity, *, expected_file_sha256):
        step = identity["step"]
        assert path == str(tmp_path / f"{step}.pt") and expected_file_sha256 == str(step)
        loaded.append(step)
        return states[step]

    monkeypatch.setattr(followup_updates, "load_checkpoint", load)
    monkeypatch.setattr(vlm_campaign, "_verified_execution", lambda *a, **kw: source)
    updates, ledger = tracking._source_window(c, source_binding, fit, [1, 2, 4, 8, 16])
    np.testing.assert_array_equal(updates, np.tile([1.0, -1.0], (16, 1)))
    assert loaded == list(range(64, 81))
    assert [record["step"] for record in ledger] == loaded
    with pytest.raises(ValueError, match="calibration at step64"):
        tracking._source_window(c, source_binding, {**fit, "origin_id": f"{seed}_X_BASE_96"}, [16])
    states[66]["metadata"]["checkpoint_step"] = 67
    with pytest.raises(ValueError, match="checkpoint identity"):
        tracking._source_window(c, source_binding, fit, [16])
