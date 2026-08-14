from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from scripts.run_small_natural_replay import main


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "experiment": "small_neural_natural_replay_v2",
        "data": {
            "samples_per_family_per_split": 1,
            "realizations_per_semantic": 2,
            "data_seed": 20260814,
            "splits": ["train", "calibration", "val", "iid_test", "ood_test"],
            "task_families": [
                "digit_offset",
                "count_transform",
                "gauge_calibration",
                "bar_chart_aggregate",
                "relation_rule",
            ],
        },
        "replay": {"n_mediators": 2, "n_forks": 2, "bootstrap_draws": 1000},
        "training": {
            "seeds": [0],
            "steps": 1,
            "batch_size": 16,
            "image_size": 16,
            "hidden_dim": 4,
            "learning_rate": 0.2,
            "device": "cpu",
        },
        "synthetic": {"error_mass": 0.8, "role": "off_support_stress_test"},
        "confirmatory": False,
        "outputs": {
            "summary": str(tmp_path / "summary.json"),
            "compensabilities": str(tmp_path / "compensabilities.csv"),
            "crossed_risks": str(tmp_path / "crossed_risks.csv"),
            "selection": str(tmp_path / "selection.csv"),
            "manifest": str(tmp_path / "manifest.json"),
        },
    }


def test_small_replay_cli_writes_an_atomic_auditable_bundle(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(tmp_path), sort_keys=False), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    with (tmp_path / "compensabilities.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["schema_version"] == 2
    assert summary["experiment"] == "small_neural_natural_replay_v2"
    assert summary["status"] == "PILOT_COMPLETE"
    assert summary["claim_scope"] == "small_neural_operational_mechanism_only"
    assert summary["counts"]["semantic_states"] == 25
    assert len(summary["family_estimates"]) == 5
    assert {
        "c_sel",
        "c_fork",
        "c_syn",
        "mediator_gap",
        "transport_gap",
    }.issubset(summary["family_estimates"][0])
    assert set(summary["family_estimates"][0]["c_sel"]) == {"mean", "ci_low", "ci_high"}
    assert len(rows) == 5
    assert {"c_sel", "c_fork", "c_syn", "mediator_gap", "transport_gap"}.issubset(rows[0])
    assert manifest["natural_mediators"]["materialization"] == "cluster_aggregated"
    assert manifest["forked_continuations"]["independence_unit"] == "sample_id"
    assert len(manifest["files"]) == 4


def test_small_replay_cli_rejects_unknown_config_fields_before_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["unknown"] = True
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 2
    assert not (tmp_path / "summary.json").exists()
