import copy
import json

import pytest

from src.decision_modeling import cli
from src.decision_modeling.runtime import read_config
from src.modeling_v3.io import canonical_hash


def test_frozen_config_plan_and_portable_panels(tmp_path):
    config = read_config(cli.DEFAULT_CONFIG)
    panels = json.loads(cli.DEFAULT_PANELS.read_text())
    assert panels["config_hash"] == canonical_hash(config)
    assert panels["inputs_hash"] == canonical_hash(
        {k: v for k, v in panels.items() if k != "inputs_hash"}
    )
    result = cli.main(["plan", "--out", str(tmp_path / "plan.json")])
    assert result["D2_updates"] == 512 and result["maximum_D2_D4_updates"] == 768
    assert not result["automatic_successor"]


@pytest.mark.parametrize(
    "command,extra",
    [
        ("bridge", []),
        ("block", ["--origin", "O1", "--recipe", "R4", "--steps", "8"]),
        (
            "observe",
            [
                "--checkpoint",
                "missing.json",
                "--origin",
                "O1",
                "--candidate",
                "R4",
                "--horizon",
                "8",
            ],
        ),
    ],
)
def test_gpu_commands_default_to_no_model_no_server_reads(tmp_path, monkeypatch, command, extra):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not load models or read private runtime")

    monkeypatch.setattr(cli, "load_runtime", forbidden)
    result = cli.main([command, "--runtime", "missing.json", "--out", str(tmp_path), *extra])
    assert result["status"] == "DRY_RUN_NO_MODEL_LOAD"


def test_endpoint_labels_cannot_mix_origins_or_horizons():
    config = read_config(cli.DEFAULT_CONFIG)
    spec = {
        "identity": {
            "kind": "DECISION_MODELING_BLOCK",
            "origin_id": "O1",
            "recipe": "R4",
            "step": 8,
        }
    }
    cli.validate_endpoint(spec, origin="O1", candidate="R4", horizon=8, config=config)
    with pytest.raises(ValueError, match="label"):
        cli.validate_endpoint(spec, origin="O2", candidate="R4", horizon=8, config=config)
    with pytest.raises(ValueError, match="label"):
        cli.validate_endpoint(spec, origin="O1", candidate="R4", horizon=32, config=config)
    origin = {"identity": config["blocks"]["origins"]["O1"]}
    cli.validate_endpoint(origin, origin="O1", candidate="ORIGIN", horizon=0, config=config)


def test_no_reference_extension_or_independent_eval_before_required_evidence(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Reject unsupported execution before loading model")

    monkeypatch.setattr(cli, "load_runtime", forbidden)
    base = [
        "observe",
        "--runtime",
        "missing",
        "--checkpoint",
        "missing",
        "--out",
        str(tmp_path),
        "--origin",
        "O1",
        "--candidate",
        "R4",
        "--horizon",
        "32",
        "--execute-gpu",
    ]
    with pytest.raises(ValueError, match="preceding"):
        cli.main([*base, "--look", "128", "--append-reason", "choice unresolved"])
    with pytest.raises(ValueError, match="freeze"):
        cli.main([*base, "--panel", "E"])
    with pytest.raises(ValueError, match="combination"):
        cli.main([*base, "--role", "reference", "--look", "32"])


def test_source_model_settings_cannot_silently_change(tmp_path):
    config = copy.deepcopy(read_config(cli.DEFAULT_CONFIG))
    config["model"]["generation"]["max_new_tokens"] = 65
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="settings"):
        read_config(path)


def test_development_freeze_requires_all_16_branches_and_strong_baselines(tmp_path):
    config = read_config(cli.DEFAULT_CONFIG)
    reports = []
    for origin in ("O1", "O2"):
        for recipe in config["blocks"][f"{origin}_recipes"]:
            path = tmp_path / f"{origin}_{recipe}.json"
            cli.gpu._publish(
                path,
                {
                    "origin_id": origin,
                    "recipe": recipe,
                    "status": "BLOCK_COMPLETE",
                    "steps": 32,
                    "checkpoint": {"identity": {"decision_config_hash": canonical_hash(config)}},
                },
            )
            reports.append(cli.gpu._binding(path))
    lock = {
        "config_hash": canonical_hash(config),
        "selected_exact_recipe": "R4",
        "simple_fixed_recipe": "R2",
        "readout_rule": "w0_frozen_intervals",
        "E_candidates": ["R0", "R2", "R4", "GDPO_R4", "SAW_R4"],
        "development_reports": reports,
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(lock))
    assert cli._freeze(path, config)["selected_exact_recipe"] == "R4"
    path.write_text(json.dumps(lock | {"development_reports": reports[:-1]}))
    with pytest.raises(ValueError, match="every registered"):
        cli._freeze(path, config)
    path.write_text(json.dumps(lock | {"E_candidates": ["R0", "R4"]}))
    with pytest.raises(ValueError, match="strong baselines"):
        cli._freeze(path, config)
