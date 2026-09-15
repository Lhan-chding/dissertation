"""N3 decisions must derive paired evidence from actual metric rows."""

import copy
import gzip
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from src.modeling_contrast.io import RunWriter, verify_run_manifest
from src.modeling_contrast.selection_driver import (
    build_selection,
    evaluate_rows,
    frozen_direction_curves,
    write_selection,
)

CONFIG = json.loads(
    (Path(__file__).parents[2] / "configs/modeling_contrast/protocol.json").read_text()
)


def rows():
    result = []
    for seed, arm, anchor in product((201, 202, 203, 204), ("X_BASE", "X_VALID"), (8, 24, 40)):
        for noise in range(8):
            for method, error in (("C2", 0.04), ("C0", 1.0), ("C1_LEGACY_EXACT_RULE", 0.64)):
                result.append(
                    {
                        "configuration_id": f"{method}|O_CRN|64|FULL|1e-05|0.01|8",
                        "method": method,
                        "observation": "O_CRN",
                        "n": 64,
                        "rank": "FULL",
                        "alpha": 1e-5,
                        "eta": 0.01,
                        "fit_banks": 8,
                        "stage": "N2C",
                        "seed": seed,
                        "arm": arm,
                        "anchor": anchor,
                        "noise_replica": noise,
                        "target": "joint_1_minus_joint_0",
                        "population": "all",
                        "case_count": 4,
                        "actual_rank": 2,
                        "k": 3,
                        "predicted_difference_norm": 0.001,
                        "pX_error_ss": error,
                        "pX_truth_ss": 1.0,
                        "v_error_ss": error,
                        "v_truth_ss": 1.0,
                        "event_error_ss": error,
                        "event_truth_ss": 1.0,
                        "event_mae": error,
                        "pX_abs_errors": [error**0.5] * 24,
                        "v_abs_errors": [error**0.5] * 24,
                    }
                )
    return result


def test_stability_comes_from_four_seed_paired_bootstrap_for_both_targets():
    evidence = evaluate_rows(rows(), CONFIG, {(s, "X_BASE", 8): 4 for s in range(201, 205)})
    candidate = evidence["candidates"][0]
    assert candidate["stable_improvement_C0"] is True
    assert candidate["stable_improvement_C1"] is True
    assert candidate["pX_nrmse"] == 0.2
    assert len(evidence["bootstrap"]) == 4
    assert all(row["cluster_count"] == 4 for row in evidence["bootstrap"])
    assert all(row["bootstrap_replicates"] == 5000 for row in evidence["bootstrap"])
    assert candidate["nonalias_eval_units"] == 16


def test_one_target_or_missing_paired_rows_cannot_manufacture_stability():
    data = rows()
    for row in data:
        if row["method"] == "C2":
            row["v_error_ss"] = 1.1
    candidate = evaluate_rows(data, CONFIG, {})["candidates"][0]
    assert candidate["stable_improvement_C1"] is False
    data = rows()
    data.pop()
    evidence = evaluate_rows(data, CONFIG, {})
    assert evidence["candidates"][0]["stable_improvement_C1"] is False
    assert any(row["status"] == "INCOMPLETE_PAIRED_BLOCKS" for row in evidence["bootstrap"])


def test_duplicate_stage_rows_do_not_double_seed_counts():
    data = rows()
    duplicates = copy.deepcopy(data)
    for row in duplicates:
        row["stage"] = "N2B"
    evidence = evaluate_rows(data + duplicates, CONFIG, {})
    assert evidence["deduplicated_rows"] == len(data)
    assert evidence["bootstrap"][0]["per_seed"][0]["row_count"] == 48


def test_build_and_write_preserve_bound_evidence_and_fail_unknown_cost(tmp_path):
    parent, run = tmp_path / "parent", tmp_path / "runs/modeling_contrast_v2"
    parent.mkdir()
    for name in ("N0", "N1", "N2", "smoke"):
        (run / name).mkdir(parents=True)
    for name in ("dataset.npz", "probe_metadata.json", "train_metadata.json"):
        (parent / name).write_bytes(b"fixture")
    trajectories = []
    for seed in range(201, 205):
        name = f"seed{seed}.npz"
        theta = np.zeros((1, 14, 3, 2))
        theta[:, :, 1] = 1
        np.savez(parent / name, branch_theta=theta, anchors=np.array([8]))
        trajectories.append(
            {"id": f"seed{seed}", "seed": seed, "arm": "X_BASE", "observations_file": name}
        )
    (parent / "manifest.json").write_text(json.dumps({"trajectories": trajectories}))
    audit = {
        "status": "PASS",
        "complete_trajectories": 60,
        "expected_trajectories": 60,
        "files": [],
        "source_inventory": [],
        "archives": [],
        "parent_archive_verified": True,
        "seed_collision_audit": {"status": "PASS", "colliding_seeds": []},
    }
    (run / "N0/parent_integrity_audit.json").write_text(json.dumps(audit))
    packet = run / "N1/packet_arrays.npz"
    packet.write_bytes(b"frozen fixture packet")
    (run / "N1/packets.json").write_text(
        json.dumps(
            {
                "packet_arrays_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
            }
        )
    )
    with gzip.open(run / "N2/METRICS.jsonl.gz", "wt") as stream:
        for row in rows():
            stream.write(json.dumps(row) + "\n")
    (run / "smoke/smoke_result.json").write_text(
        json.dumps(
            {
                "forecast": {
                    "wall_seconds": 60,
                    "peak_ram_gib": 1,
                    "added_output_bytes": 1000,
                    "temporary_bytes": 1000,
                },
            }
        )
    )
    payload = build_selection(CONFIG, parent, run)
    assert payload["lock"]["allow_fresh_cpu"] is False
    assert payload["lock"]["candidate_audit"][0]["stable_improvement_C1"] is True
    assert (
        payload["lock"]["packet_hashes"][str(packet)]
        == hashlib.sha256(packet.read_bytes()).hexdigest()
    )
    with RunWriter(run / "N3", payload["lock"]["binding"], project_root=tmp_path) as writer:
        write_selection(writer, payload)
    manifest = verify_run_manifest(run / "N3", payload["lock"]["binding"])
    assert "MODEL_SELECTION_LOCK.json" in manifest["outputs"]
    assert "PAIRED_BOOTSTRAP.json" in manifest["outputs"]


def test_raw_direction_curves_keep_replica_units_and_four_seed_limit(tmp_path, monkeypatch):
    from src.modeling_contrast import parent as parent_module
    from src.modeling_qualification.math_contracts import helmert

    parent, run = tmp_path / "parent", tmp_path / "run"
    parent.mkdir()
    (run / "N2/N2C").mkdir(parents=True)
    metadata = [{"group": i // 12, "weight": 1} for i in range(72)]
    (parent / "probe_metadata.json").write_text(json.dumps(metadata))
    rule = {
        key: value
        for key, value in rows()[0].items()
        if key
        in (
            "configuration_id",
            "method",
            "observation",
            "n",
            "rank",
            "alpha",
            "eta",
            "fit_banks",
        )
    }
    (run / "N2/frozen_nonzero_rules.json").write_text(json.dumps([rule]))
    event = np.zeros((4, 2, 72, 4))
    event[..., 0] = 0.001
    event[..., 3] = -0.001
    prediction = np.repeat((event @ helmert())[None], 8, axis=0)
    branch = np.zeros((1, 14, 3, 72, 4)) + 0.25
    branch[0, 10:14, 1:] += event
    monkeypatch.setattr(parent_module, "load_parent_oracle", lambda *_: {"branch_p": branch})
    for seed in (101, 201, 202, 203, 204):
        out = run / f"N2/N2C/seed{seed}_X_BASE/a0"
        out.mkdir(parents=True)
        np.savez_compressed(out / "predictions.npz", prediction=prediction)
        (out / "freeze.json").write_text(
            json.dumps(
                {
                    "frozen_before_scoring": True,
                    "prediction_sha256": hashlib.sha256(
                        (out / "predictions.npz").read_bytes()
                    ).hexdigest(),
                    "models": [
                        {
                            **rule,
                            "seed": seed,
                            "arm": "X_BASE",
                            "anchor": 8,
                            "noise_replica": noise,
                        }
                        for noise in range(8)
                    ],
                }
            )
        )
        if seed == 101:
            full = json.loads((out / "freeze.json").read_text())
            metadata_path = out / "freeze_metadata.json.gz"
            with gzip.open(metadata_path, "wt") as stream:
                json.dump(full, stream)
            (out / "freeze.json").write_text(json.dumps({
                "frozen_before_scoring": True,
                "prediction_sha256": full["prediction_sha256"],
                "metadata_file": metadata_path.name,
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            }))
    result = frozen_direction_curves(CONFIG, parent, run)
    primary = [
        row
        for row in result["rows"]
        if row["target"] == "joint_1_minus_joint_0"
        and row["quantity"] == "pX"
        and row["delta"] == 0.00025
    ]
    full, selection = primary
    assert full["effective_anchor_bank_units"] == 20
    assert full["training_seed_count"] == 5
    assert full["accuracy"] == 1
    assert selection["effective_anchor_bank_units"] == 16
    assert selection["training_seed_count"] == 4
    assert selection["status"] == "INSUFFICIENT_EFFECTIVE_CASES"
    assert selection["accuracy"] is None
    assert str(metadata_path) in result["source_hashes"]
    metadata_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed frozen model metadata"):
        frozen_direction_curves(CONFIG, parent, run)
