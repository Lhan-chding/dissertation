"""Tiny synthetic tests for the separate, model-free N4 audit script."""

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/audit_modeling_n4_results.py"
SPEC = importlib.util.spec_from_file_location("independent_n4_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_group_contrasts_use_original_signal_energy_and_unfloored_ratios():
    truth = np.tile([0.02, -0.01, -0.02, 0.01], (4, 72, 1))
    groups = np.repeat(np.arange(6), 12)
    row = audit.contrast_stats(truth * 0.5, truth, groups)
    summary = audit.aggregate([{"seed": 601, **row}])
    assert summary["pX_nrmse"] == pytest.approx(0.5)
    assert summary["v_nrmse"] == pytest.approx(0.5)
    assert summary["pX_error_q95"] == pytest.approx(0.01)
    assert summary["v_error_q95"] == pytest.approx(0.005)
    tiny = audit.contrast_stats(truth * 1e-30 * 0.5, truth * 1e-30, groups)
    assert audit.aggregate([{"seed": 601, **tiny}])["pX_nrmse"] == pytest.approx(0.5)
    empty_signal = audit.contrast_stats(truth * 0, truth * 0, groups)
    assert audit.aggregate([{"seed": 601, **empty_signal}])["pX_nrmse"] is None


def bootstrap_row(seed, error, energy, method):
    return {
        "configuration_id": method,
        "seed": seed,
        "arm": "X_BASE",
        "anchor": 8,
        "noise_replica": 0,
        "pX_error_ss": error,
        "pX_truth_ss": energy,
    }


def test_bootstrap_recomputes_energy_weighted_pooled_ratios_not_seed_ratio_mean():
    method = [bootstrap_row(601, 9, 9, "selected"), bootstrap_row(602, 0, 1, "selected")]
    baseline = [bootstrap_row(601, 9, 9, "C0"), bootstrap_row(602, 1, 1, "C0")]
    result = audit.bootstrap(method, baseline, "pX")
    assert result["paired_mean"] == pytest.approx(np.sqrt(0.9) - 1)
    assert result["paired_mean"] != pytest.approx(-0.5)
    assert result["ci025"] == -1
    assert result["ci975"] == 0
    assert result["bootstrap_replicates"] == 5000
    with pytest.raises(ValueError, match="complete unique"):
        audit.bootstrap(method, baseline[:1], "pX")
    baseline[1]["pX_truth_ss"] = 2
    with pytest.raises(ValueError, match="truth energy"):
        audit.bootstrap(method, baseline, "pX")


def stage(tmp_path):
    root = tmp_path / "N4"
    root.mkdir()
    (root / "value.json").write_text('{"value": 1}')
    manifest = {
        "status": "COMPLETE",
        "binding": dict.fromkeys(audit.BINDINGS, "a" * 64),
        "outputs": {"value.json": audit.sha(root / "value.json")},
    }
    (root / "RUN_MANIFEST.json").write_text(json.dumps(manifest))
    return root, manifest


def test_stage_verifies_hashes_completeness_and_rejects_traversal(tmp_path):
    root, manifest = stage(tmp_path)
    assert audit.Integrity().stage(root)["status"] == "COMPLETE"
    (root / "value.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.Integrity().stage(root)
    manifest["status"] = "RUNNING"
    (root / "RUN_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not COMPLETE"):
        audit.Integrity().stage(root)
    with pytest.raises(ValueError, match="unsafe"):
        audit.inside(root, "../escape")


def prediction_unit(tmp_path):
    truth = np.tile([0.02, -0.01, -0.02, 0.01], (4, 72, 1))
    encoded = truth * 0.5 @ audit.helmert()
    prediction = np.stack([encoded, encoded], axis=1)[None]
    np.savez_compressed(tmp_path / "predictions.npz", prediction=prediction)
    value = np.ascontiguousarray(prediction[0])
    array_hash = hashlib.sha256(
        str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
    ).hexdigest()
    entry = {"seed": 501, "arm": "X_BASE", "split": "fresh_calibration"}
    model = {
        "configuration_id": "C6|O_LR_MIX|64|4|1e-05|0.01|2",
        "observation": "O_LR_MIX",
        "rank": 4,
        "alpha": 1e-5,
        "eta": 0.01,
        "fit_banks": 2,
        "seed": 501,
        "arm": "X_BASE",
        "anchor": 8,
        "noise_replica": 0,
        "method": "C6",
        "n": 64,
        "original_role": "fresh_calibration",
        "v2_role": "fresh_calibration",
        "prediction_hash": array_hash,
    }
    metadata = {
        "frozen_before_scoring": True,
        "oracle_diagnostic": False,
        "prediction_sha256": audit.sha(tmp_path / "predictions.npz"),
        "models": [model],
    }
    with gzip.open(tmp_path / "freeze_metadata.json.gz", "wt") as stream:
        json.dump(metadata, stream)
    mini = {key: value for key, value in metadata.items() if key != "models"}
    mini.update(
        metadata_file="freeze_metadata.json.gz",
        metadata_sha256=audit.sha(tmp_path / "freeze_metadata.json.gz"),
    )
    (tmp_path / "freeze.json").write_text(json.dumps(mini))
    oracle = np.full((3, 14, 3, 72, 4), 0.25)
    oracle[0, 10:14, 1] += truth
    return entry, oracle, np.repeat(np.arange(6), 12)


def test_sealed_raw_prediction_decodes_without_model_or_production_statistics(tmp_path):
    entry, oracle, groups = prediction_unit(tmp_path)
    rows = audit.read_prediction_unit(
        tmp_path, entry, 0, oracle, groups, {"C6|O_LR_MIX|64|4|1e-05|0.01|2"}, audit.Integrity()
    )
    result = audit.aggregate(rows)
    assert result["pX_nrmse"] == pytest.approx(0.5)
    assert result["v_nrmse"] == pytest.approx(0.5)
    source = SCRIPT.read_text()
    assert "from src." not in source and "import torch" not in source
    assert "load_parent_oracle" not in source
    (tmp_path / "freeze_metadata.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.read_prediction_unit(
            tmp_path, entry, 0, oracle, groups, {"C6|O_LR_MIX|64|4|1e-05|0.01|2"}, audit.Integrity()
        )


def test_bad_role_and_changed_numeric_scores_are_rejected(tmp_path):
    entry, oracle, groups = prediction_unit(tmp_path)
    entry["split"] = "fresh_locked_test"
    with pytest.raises(ValueError, match="identity disagrees"):
        audit.read_prediction_unit(
            tmp_path, entry, 0, oracle, groups, {"C6|O_LR_MIX|64|4|1e-05|0.01|2"}, audit.Integrity()
        )
    with pytest.raises(ValueError, match="statistical mismatch"):
        audit.near(0.65, 0.64, "pooled")


def test_cli_records_failure_without_touching_execution_or_overwriting(tmp_path):
    out = tmp_path / "posthoc/audit.json"
    assert audit.main(["--run-root", str(tmp_path), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["status"] == "FAIL"
    with pytest.raises(FileExistsError):
        audit.main(["--run-root", str(tmp_path), "--out", str(out)])
    with pytest.raises(SystemExit):
        audit.main(["--run-root", str(tmp_path), "--out", str(tmp_path / "N4/audit.json")])


def test_server_selection_cannot_change_original_candidates_even_with_valid_new_hash(tmp_path):
    directory = tmp_path / "N3_server"
    directory.mkdir()

    def lock(rank):
        result = {
            "source_hashes": {},
            "input_hashes": {},
            "packet_hashes": {},
            "selected": [{"configuration_id": "fixed", "rank": rank}],
            "candidate_audit": [{"configuration_id": "fixed", "rank": rank}],
            "protocol_sha256": "a" * 64,
        }
        result["binding"] = {
            "source": audit.canonical({}),
            "data": audit.canonical({}),
            "packet": audit.canonical({}),
            "selector": audit.canonical(result["selected"]),
            "config": "a" * 64,
        }
        result["lock_sha256"] = audit.canonical(result)
        return result

    (directory / "MODEL_SELECTION_LOCK.json").write_text(json.dumps(lock(1)))
    (directory / "ORIGINAL_MODEL_SELECTION_LOCK.json").write_text(json.dumps(lock(4)))
    with pytest.raises(ValueError, match="immutable scientific selection"):
        audit.verify_selection(tmp_path, {}, audit.Integrity())


def test_wrong_total_science_pass_is_rejected():
    results = [{"prediction_status": "PREDICTION_PASS"}, {"prediction_status": "PREDICTION_FAIL"}]
    with pytest.raises(ValueError, match="science_status"):
        audit.verify_science_status({"science_status": "PREDICTION_PASS"}, results)
    assert (
        audit.verify_science_status({"science_status": "PREDICTION_FAIL"}, results)
        == "PREDICTION_FAIL"
    )


def test_configuration_identity_is_rebuilt_from_frozen_fields(tmp_path):
    entry, oracle, groups = prediction_unit(tmp_path)
    path = tmp_path / "freeze_metadata.json.gz"
    metadata = audit.read_json(path)
    metadata["models"][0]["rank"] = 1
    with gzip.open(path, "wt") as stream:
        json.dump(metadata, stream)
    mini = audit.read_json(tmp_path / "freeze.json")
    mini["metadata_sha256"] = audit.sha(path)
    (tmp_path / "freeze.json").write_text(json.dumps(mini))
    with pytest.raises(ValueError, match="configuration identity"):
        audit.read_prediction_unit(
            tmp_path, entry, 0, oracle, groups, {"C6|O_LR_MIX|64|4|1e-05|0.01|2"}, audit.Integrity()
        )


def test_extra_by_seed_rows_are_not_silently_ignored(tmp_path):
    rows = []
    recorded_seed = []
    for seed in range(501, 507):
        row = {
            "configuration_id": "C6|O_LR_MIX|64|4|1e-05|0.01|2",
            "method": "C6",
            "observation": "O_LR_MIX",
            "n": 64,
            "rank": 4,
            "alpha": 1e-5,
            "eta": 0.01,
            "fit_banks": 2,
            "seed": seed,
            "arm": "X_BASE",
            "anchor": 8,
            "noise_replica": 0,
            "target": audit.TARGET,
            "population": "all",
        }
        summary = {
            "configuration_id": "C6|O_LR_MIX|64|4|1e-05|0.01|2",
            "seed": seed,
            "target": audit.TARGET,
            "population": "all",
            "seed_count": 1,
        }
        for quantity in audit.QUANTITIES:
            row.update(
                {
                    quantity + "_error_ss": 6.0,
                    quantity + "_truth_ss": 24.0,
                    quantity + "_abs_errors": [0.5] * 24,
                }
            )
            summary.update(
                {
                    quantity + "_error_ss": 6.0,
                    quantity + "_truth_ss": 24.0,
                    quantity + "_nrmse": 0.5,
                    quantity + "_error_q95": 0.5,
                    quantity + "_mae": 0.5,
                }
            )
        rows.append(row)
        recorded_seed.append(summary)
    with gzip.open(tmp_path / "fresh_calibration_metrics.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    recorded_seed.append({**recorded_seed[0], "seed": 999})
    with (tmp_path / "fresh_calibration_BY_SEED.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(recorded_seed[0]))
        writer.writeheader()
        writer.writerows(recorded_seed)
    with pytest.raises(ValueError, match=r"BY_SEED.*coverage"):
        audit.verify_scores(tmp_path, "fresh_calibration", rows, {"C6|O_LR_MIX|64|4|1e-05|0.01|2"})


def test_group_energy_sums_to_pooled_and_group_precision_retains_seed_support():
    groups = np.repeat([f"group_{index}" for index in range(6)], 12)
    truth = np.zeros((4, 72, 4))
    for index in range(6):
        truth[:, groups == f"group_{index}"] = np.array([0.01, -0.01, -0.02, 0.02]) * index
    rows = [
        {
            "configuration_id": "selected",
            "method": "C6",
            "seed": seed,
            **audit.contrast_stats(truth * scale * 0.5, truth * scale, groups),
        }
        for seed, scale in ((501, 1), (502, 2))
    ]
    table = audit.group_table({"fresh_calibration": rows})
    pooled = audit.aggregate(rows)
    assert len(table) == 6
    assert {row["group"] for row in table} == set(groups)
    assert all(
        row["seed_count"] == 2 and row["row_count"] == 2 and row["case_count"] == 8 for row in table
    )
    for quantity in audit.QUANTITIES:
        for suffix in ("error_ss", "truth_ss"):
            field = quantity + "_" + suffix
            assert sum(row[field] for row in table) == pytest.approx(pooled[field])
    zero = next(row for row in table if row["group"] == "group_0")
    assert zero["pX_nrmse"] is None and zero["v_nrmse"] is None
    worst = max(table, key=lambda row: row["pX_error_q95"])
    assert worst["group"] == "group_5"
    assert worst["pX_nrmse"] == pytest.approx(0.5)
    assert worst["pX_mae"] == pytest.approx(0.0375)
    assert worst["pX_error_q95"] == pytest.approx(0.05)
    assert len(json.dumps(table).encode()) < 3 * 2**20
