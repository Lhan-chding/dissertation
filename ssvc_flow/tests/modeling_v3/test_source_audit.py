import gzip
import hashlib
import json

import numpy as np
import pytest

from src.modeling_v3.source_audit import (
    H,
    _canonical,
    _inside,
    _pooled,
    _sha,
    audit_parent,
    full_rank_equivalence,
    independent_model_statistics,
    independent_observation_statistics,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(tmp_path):
    """Tiny genuine primitive packets, no training and no oracle reconstruction."""
    from src.modeling_contrast.observations import ToyWorldService, measure
    from src.modeling_contrast.packet_codec import encode_arrays

    root = tmp_path / "old"
    raw = root / "N4/raw"
    raw.mkdir(parents=True)
    config = {"fixture": True}
    write_json(raw / "resolved_config.json", config)
    metadata = [{"prompt_id": "p0", "group": "g0", "weight": 1.0}]
    for name, value in (
        ("probe_metadata.json", metadata),
        ("train_metadata.json", [{"prompt_id": "t0"}]),
        ("parameter_layout.json", []),
    ):
        write_json(raw / name, value)
    np.savez(raw / "dataset.npz", probe_categories=[[0, 1, 2, 3]])
    theta = np.zeros((1, 14, 3, 2))
    theta[:, :, 2, 0] = 0.001
    aliases = np.arange(42).reshape(1, 14, 3)
    fields = dict(branch_theta=theta, branch_alias=aliases)
    fields.update(
        {
            key: np.zeros(1)
            for key in (
                "branch_prompt_indices",
                "branch_actions",
                "branch_categories",
                "branch_old_logp",
                "adam_m",
                "adam_v",
                "adam_step",
                "branch_adam_m",
                "branch_adam_v",
                "branch_adam_step",
                "checkpoint_grad",
                "checkpoint_grad_is_none",
            )
        }
    )
    np.savez(raw / "observations.npz", **fields)
    p = np.array([[0.2, 0.3, 0.2, 0.3]])
    u = p + np.array([[0.001, -0.001, 0, 0]])
    v = p + np.array([[-0.001, 0, 0.001, 0]])
    oracle = np.broadcast_to(np.stack([p, u, v]), (1, 14, 3, 1, 4)).copy()
    np.savez(raw / "oracle.npz", branch_p=oracle)
    np.savez(raw / "counts.npz", counts=np.ones((1, 4)))
    write_json(raw / "rng.json", {"state": "fixture"})
    entry = {
        "id": "seed501_X_BASE",
        "seed": 501,
        "arm": "X_BASE",
        "split": "fresh_calibration",
        "observations_file": "observations.npz",
        "oracle_file": "oracle.npz",
        "counts_file": "counts.npz",
        "rng_file": "rng.json",
    }
    entry["sha256"] = {
        key: _sha(raw / entry[key])
        for key in ("observations_file", "oracle_file", "counts_file", "rng_file")
    }
    manifest = {
        "events": ["X", "S", "W", "I"],
        "operation_names": ["joint_0", "joint_1", "no_x_off_1"],
        "anchors": [8],
        "bank_roles": {"fit": list(range(8)), "diagnostic": [8, 9], "evaluation": [10, 11, 12, 13]},
        "trajectories": [entry],
        "source_hashes": {},
        "config_sha256": _canonical(config),
    }
    manifest.update(
        {
            field: _sha(raw / name)
            for name, field in (
                ("dataset.npz", "dataset_sha256"),
                ("probe_metadata.json", "probe_identity_sha256"),
                ("train_metadata.json", "train_identity_sha256"),
                ("parameter_layout.json", "parameter_layout_sha256"),
            )
        }
    )
    write_json(raw / "manifest.json", manifest)
    unit = root / "N4_validation/fresh_calibration/seed501_X_BASE/a0"
    packet_root = unit / "packets/O_LR_ORIGIN_n16"
    packet_root.mkdir(parents=True)
    primitives, rows = {}, []
    for replica in range(2):
        packet = measure(
            ToyWorldService.from_action_probabilities(
                {"b": p, "u": u, "v": v, "o": p}, [[0, 1, 2, 3]]
            ),
            [("b", "u"), ("b", "v")],
            origin_id="o",
            method="O_LR_ORIGIN",
            n=16,
            seed=replica,
        )
        prefix = f"r{replica}b0_"
        primitives.update(
            {
                prefix + key: value
                for key, value in packet.arrays(include_covariance=False).items()
                if key.startswith("sample_")
            }
        )
        row = packet.metadata()
        row.update(
            noise_replica=replica, bank=0, array_prefix=prefix, sample_names=list(packet.samples)
        )
        rows.append(row)
    np.savez(packet_root / "packet_arrays.npz", **encode_arrays(primitives))
    packet_meta = {
        "trajectory_id": entry["id"],
        "method": "O_LR_ORIGIN",
        "n": 16,
        "replicas": 2,
        "banks": [0],
        "packets": rows,
        "anchor_index": 0,
        "input_parameters_sha256": entry["sha256"]["observations_file"],
        "packet_arrays_sha256": _sha(packet_root / "packet_arrays.npz"),
    }
    write_json(packet_root / "packets.json", packet_meta)
    model_root = unit / "models"
    model_root.mkdir()
    predictions = np.zeros((2, 4, 2, 1, 3), dtype=np.float64)
    np.savez(model_root / "predictions.npz", prediction=predictions)
    models = []
    for method, prediction in zip(("C3", "C5"), predictions, strict=True):
        digest = hashlib.sha256(
            str(prediction.dtype).encode() + str(prediction.shape).encode() + prediction.tobytes()
        ).hexdigest()
        models.append(
            {
                "method": method,
                "observation": "O_LR_ORIGIN",
                "n": 16,
                "rank": 2,
                "actual_rank": 2,
                "k": 2,
                "alpha": 0.01,
                "eta": 0.01,
                "fit_banks": 2,
                "noise_replica": 0,
                "seed": 501,
                "arm": "X_BASE",
                "anchor": 8,
                "configuration_id": method,
                "stage": "N4",
                "prediction_hash": digest,
            }
        )
    write_json(
        model_root / "freeze.json",
        {
            "models": models,
            "prediction_sha256": _sha(model_root / "predictions.npz"),
            "parameters_sha256": entry["sha256"]["observations_file"],
        },
    )
    with gzip.open(model_root / "metrics.jsonl.gz", "wt") as stream:
        for prediction, model in zip(predictions, models, strict=True):
            for row in independent_model_statistics(
                prediction @ H.T,
                oracle[0, 10:14, 1:] - oracle[0, 10:14, :1],
                metadata,
                theta[0, :2, 1:] - theta[0, :2, :1],
                model,
            ):
                stream.write(json.dumps(row) + "\n")
    for stage in (root / "N4", root / "N4_validation"):
        write_json(
            stage / "RUN_MANIFEST.json",
            {
                "status": "COMPLETE",
                "outputs": {
                    str(path.relative_to(stage)): _sha(path)
                    for path in stage.rglob("*")
                    if path.is_file()
                },
            },
        )
    return root


def test_missing_input_stays_missing_without_summary_reconstruction(tmp_path):
    parent = tmp_path / "summaries"
    parent.mkdir()
    write_json(parent / "N4_FACTUAL_SUMMARY.json", {"status": "PASS", "n": 999})
    result = audit_parent({}, parent, tmp_path / "audit")
    assert result["status"] == "MISSING_INPUTS"
    assert result["packet_units_replayed"] == 0
    assert result["q1_self_contained_math_may_continue"]
    assert not result["summary_reconstruction_used"]
    assert all((tmp_path / "audit" / name).is_file() for name in result["output_hashes"])


def test_original_packet_replay_all_active_and_r_equals_k(tmp_path, monkeypatch):
    root = fixture_run(tmp_path)
    import src.modeling_contrast.observations as observations

    monkeypatch.setattr(observations, "measure", lambda *a, **kw: pytest.fail("audit sampled"))
    before = {str(path): _sha(path) for path in root.rglob("*") if path.is_file()}
    result = audit_parent({}, root, tmp_path / "audit")
    assert result["status"] == "MISSING_INPUTS"  # V2 did not save Q/U/C or V3 pilot roles.
    assert not result["errors"]
    assert result["packet_units_replayed"] == 1
    assert result["primitive_arrays_replayed"] > 0
    assert result["raw_helmert_rows"] == 4
    assert result["stored_metric_rows_matched"] == 10
    assert result["rank_equivalence_rows"] == 1
    assert result["origin_count"] == 1
    assert all(_sha(path) == digest for path, digest in before.items())
    assert (
        "inactive_all_minus_active"
        in (tmp_path / "audit/Q0_ALL_ACTIVE_SQUARED_SUMS.csv").read_text()
    )
    with pytest.raises(FileExistsError):
        audit_parent({}, root, tmp_path / "audit")


def test_changed_original_does_not_pass_and_no_original_is_overwritten(tmp_path):
    root = fixture_run(tmp_path)
    (root / "N4/raw/observations.npz").write_bytes(b"corrupted")
    result = audit_parent({}, root, tmp_path / "audit")
    assert result["status"] == "FAIL"
    assert any("hash mismatch" in row["detail"] for row in result["errors"])
    assert (root / "N4/raw/observations.npz").read_bytes() == b"corrupted"


def test_missing_primitive_never_falls_back_to_cached_summary_arrays(tmp_path):
    from src.modeling_contrast.packet_codec import decode_arrays, encode_arrays
    from src.modeling_contrast.packet_reconstruction import reconstruct_measurements

    root = fixture_run(tmp_path)
    packet_path = next((root / "N4_validation").rglob("packets.json"))
    packet_meta = json.loads(packet_path.read_text())
    arrays_path = packet_path.parent / "packet_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as packed:
        decoded = decode_arrays(packed)
    decoded.update(reconstruct_measurements(decoded, packet_meta, force=True))
    missing = next(name for name in decoded if name.endswith("candidate_logp"))
    del decoded[missing]
    np.savez(arrays_path, **encode_arrays(decoded))
    packet_meta["packet_arrays_sha256"] = _sha(arrays_path)
    write_json(packet_path, packet_meta)
    stage = root / "N4_validation"
    manifest_path = stage / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for path in (arrays_path, packet_path):
        manifest["outputs"][str(path.relative_to(stage))] = _sha(path)
    write_json(manifest_path, manifest)
    result = audit_parent({}, root, tmp_path / "audit")
    assert result["status"] == "MISSING_INPUTS"
    assert result["packet_units_replayed"] == 0
    assert any(missing in row["detail"] for row in result["missing_inputs"])


def test_paths_and_output_never_escape_or_overwrite_inputs(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        _inside(tmp_path, "../outside")
    with pytest.raises(ValueError, match="unsafe"):
        _inside(tmp_path, "/outside")
    with pytest.raises(ValueError, match="overwrite"):
        audit_parent({}, tmp_path, tmp_path / "nested")


def test_raw_validity_must_not_silently_become_minus_invalid():
    raw = np.array([[[0.10, 0.0, 0.0, -0.01]], [[0.12, 0.0, 0.0, -0.02]]])
    truth = np.zeros((1, 4))
    metadata = [{"prompt_id": "p", "group": "g", "weight": 1.0}]
    r = independent_observation_statistics(raw, raw, truth, metadata, "raw_event")
    assert r["v_error_ss"] == pytest.approx(0.10**2 + 0.12**2)
    projected = raw @ H @ H.T
    h = independent_observation_statistics(projected, raw, truth, metadata, "helmert")
    assert h["v_error_ss"] == pytest.approx(float(np.square(projected[..., 3]).sum()))


def test_equivalence_requires_same_calibration_and_preserves_nontruncation():
    base = {
        "observation": "O_LR_ORIGIN",
        "n": 16,
        "fit_banks": 2,
        "actual_rank": 2,
        "k": 2,
        "alpha": 0.01,
        "eta": 0.01,
    }
    models = [
        dict(base, method="C3"),
        dict(base, method="C5"),
        dict(base, method="C6", fit_banks=4),
        dict(base, method="C5", actual_rank=1),
    ]
    pred = np.zeros((4, 2))
    rows = full_rank_equivalence(pred, models)
    assert [row["status"] for row in rows] == [
        "PASS",
        "MISSING_MATCHED_C3",
        "NONDEGENERATE_RANK_NOT_EQUIVALENCE_CASE",
    ]
    assert not rows[0]["genuine_truncation"] and rows[2]["genuine_truncation"]
    pred[1, 0] = 1
    assert full_rank_equivalence(pred, models)[0]["status"] == "FAIL"


def test_inactive_squared_sums_retain_unknown_denominator():
    base = {"method": "C5", "target": "joint_1_minus_joint_0", "population": "all", "case_count": 4}
    base.update(
        {
            f"{metric}_{kind}_ss": 4.0
            for metric in ("event", "pX", "v")
            for kind in ("error", "truth")
        }
    )
    active = dict(base, population="active", case_count=2)
    active.update({key: 1.0 for key in active if key.endswith("_ss")})
    inactive = next(
        row for row in _pooled([base, active]) if row["population"] == "inactive_all_minus_active"
    )
    assert inactive["case_count"] == 2
    assert inactive["pX_error_ss"] == 3.0
    assert inactive["pX_truth_ss"] == 3.0
