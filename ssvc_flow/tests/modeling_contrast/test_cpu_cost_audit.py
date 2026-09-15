import gzip
import hashlib
import json

import numpy as np
import pytest

from src.modeling_contrast.cpu_cost_audit import (
    audit_fit_query,
    discover_sealed_configs,
    instrument_cpu_components,
    operation_slack,
    run_cpu_cost_audit,
)


def test_forward_timing_overlap_is_method_specific():
    ledger = {"sampling_seconds": 2.0, "score_seconds": 3.0, "forward_seconds": 1.0}
    assert instrument_cpu_components(ledger, "O_LR_ORIGIN")["timed_cpu_seconds"] == 6
    assert instrument_cpu_components(ledger, "O_LR_MIX")["timed_cpu_seconds"] == 5
    assert instrument_cpu_components(ledger, "D_KNOWN_EVENT_LOGP")["timed_cpu_seconds"] == 5
    old = instrument_cpu_components(
        {"reused_actions": 100, "historical_unique_policy_count": 2}, "O_IND"
    )
    assert old["cold_acquisition_cpu_seconds"] is None
    assert old["status"] == "HISTORICAL_ACQUISITION_CPU_UNMEASURED"


def test_operation_slack_keeps_seconds_separate_from_generation_units():
    row = operation_slack(
        {"generated_actions": 100}, {"generated_actions": 140}, 0.25, fit_query_cpu_seconds=2.0
    )
    assert row["remaining_fit_plus_four_queries_generation_units"] == 40
    assert row["max_cpu_second_price_generation_units"] == 20
    assert row["cpu_price_conversion_is_assumed"] is False


def test_audit_performs_real_737_fit_and_nonzero_query_timing_without_truth():
    rng = np.random.default_rng(7)
    theta = np.zeros((14, 3, 737))
    theta[:, 1:] = rng.normal(size=(14, 2, 737)) * 0.001
    response = rng.normal(size=(1, 14, 2, 2, 3)) * 0.001
    response[:, 10:] = np.nan  # evaluation behavior must not be used by fitting/timing
    rows = [
        {
            "bank": b,
            "noise_replica": 0,
            "old_counts_reused": False,
            "packet_id": f"p{b}",
            "policy_fingerprints": [(f"b{b}", f"u{b}"), (f"b{b}", f"v{b}")],
        }
        for b in range(14)
    ]
    packet = {
        "helmert": response,
        "covariance": np.tile(np.eye(6), (1, 14, 2, 1, 1)),
        "metadata": {"method": "O_IND", "n": 64, "packets": rows},
    }

    class Trap:
        def __getattr__(self, name):
            raise AssertionError("oracle access forbidden")

    packet["oracle"] = Trap()
    config = {
        "method": "C2",
        "observation": "O_IND",
        "n": 64,
        "rank": "FULL",
        "alpha": 1e-5,
        "eta": 0.01,
        "fit_banks": 1,
    }
    first = audit_fit_query(theta, packet, config, np.zeros((2, 6)), query_repeats=3)
    second = audit_fit_query(
        theta,
        packet,
        config,
        np.zeros((2, 6)),
        query_repeats=3,
        expected_prediction_hash=first["prediction_hash"],
    )
    assert first["parameter_count"] == 737
    assert first["setup_cpu_seconds"] > 0 and first["fit_cpu_seconds"] > 0
    assert first["query_first_cpu_seconds"] > 0
    assert len(first["query_warm_cpu_seconds"]) == 3
    assert second["prediction_matches_sealed"] is True
    assert first["cross_configuration_preparation_cache_reused"] is False


def write_seal(root, stage, model):
    out = root / stage / "seed201_X_BASE/a0"
    out.mkdir(parents=True)
    with gzip.open(out / "freeze_metadata.json.gz", "wt") as stream:
        json.dump({"frozen_before_scoring": True, "models": [model]}, stream)
    digest = hashlib.sha256((out / "freeze_metadata.json.gz").read_bytes()).hexdigest()
    (out / "freeze.json").write_text(
        json.dumps(
            {
                "frozen_before_scoring": True,
                "metadata_file": "freeze_metadata.json.gz",
                "metadata_sha256": digest,
            }
        )
    )


def test_sealed_configs_deduplicate_stages_but_keep_input_hash_receipts(tmp_path):
    model = {
        "method": "C2",
        "observation": "O_IND",
        "n": 64,
        "rank": "FULL",
        "alpha": 1e-5,
        "eta": 0.01,
        "fit_banks": 8,
        "noise_replica": 0,
        "prediction_hash": "a" * 64,
        "seed": 201,
        "arm": "X_BASE",
        "anchor": 8,
    }
    write_seal(tmp_path, "N2B", model)
    write_seal(tmp_path, "N2C", model)
    plan = discover_sealed_configs(tmp_path)
    assert len(plan["units"]) == 1 and len(plan["units"][0]["configs"]) == 1
    assert len(plan["input_files"]) == 4
    assert plan["units"][0]["configs"][0]["sealed_stages"] == ["N2B", "N2C"]


def test_unsealed_or_changed_metadata_is_rejected(tmp_path):
    model = {
        "method": "C2",
        "observation": "O_IND",
        "n": 64,
        "rank": 1,
        "alpha": 1e-5,
        "eta": 0.01,
        "fit_banks": 8,
        "noise_replica": 0,
        "prediction_hash": "a" * 64,
        "seed": 201,
        "arm": "X_BASE",
        "anchor": 8,
    }
    write_seal(tmp_path, "N2B", model)
    path = tmp_path / "N2B/seed201_X_BASE/a0/freeze_metadata.json.gz"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        discover_sealed_configs(tmp_path)


def test_main_audit_waits_for_complete_n2(tmp_path):
    (tmp_path / "N2").mkdir()
    (tmp_path / "N2/RUN_MANIFEST.json").write_text('{"status":"RUNNING"}')
    with pytest.raises(ValueError, match="N2_NOT_COMPLETE"):
        run_cpu_cost_audit(tmp_path / "parent", tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
