"""The real VLM CPU analysis chain is reachable without GPU runtime imports."""

import json
import sys
import types
from pathlib import Path

import pytest

from src.modeling_v3 import cli

CONFIG = Path(__file__).resolve().parents[2] / "configs/modeling_v3/protocol.json"


@pytest.mark.parametrize("command", ["freeze-vlm", "calibrate-vlm", "analyze-vlm"])
def test_vlm_analysis_dispatch_keeps_original_bindings(tmp_path, monkeypatch, command):
    seen = {}

    def run(config, lock, **kwargs):
        seen.update(lock=lock, **kwargs)
        return {"status": "DISPATCH_FIXTURE_ONLY"}

    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.vlm_response",
        types.SimpleNamespace(freeze_vlm_selection=run),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.vlm_results",
        types.SimpleNamespace(analyze_vlm_calibration=run, analyze_vlm_test=run),
    )
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"original": True}))
    argv = [command, "--config", str(CONFIG), "--out", str(tmp_path / "out")]
    argv += ["--inputs", str(original)]
    if command == "freeze-vlm":
        argv += [
            "--parent-cpu-lock",
            str(original),
            "--development-completion",
            str(original),
            "--selection",
            str(original),
        ]
    else:
        argv += ["--lock", str(original), "--stage-completion", str(original)]
    if command == "analyze-vlm":
        argv += ["--calibration-receipt", str(original)]
    assert cli.main(argv) == 0
    assert seen["lock"] == str(original)
    assert seen["evaluation_inputs"] == [str(original)]
    assert not seen.get("fixture", False)
    if command == "analyze-vlm":
        assert seen["calibration_receipt"] == str(original)


def test_test_prediction_dispatch_preserves_calibration_path(tmp_path, monkeypatch):
    seen = {}

    def run(config, **kwargs):
        seen.update(kwargs)
        return {"status": "DISPATCH_FIXTURE_ONLY"}

    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.vlm_response",
        types.SimpleNamespace(freeze_q5_predictions=run),
    )
    assert (
        cli.main(
            [
                "freeze-predictions",
                "--config",
                str(CONFIG),
                "--out",
                str(tmp_path / "out"),
                "--fit-spec",
                "fit.json",
                "--fit-root",
                "fit",
                "--calibration-receipt",
                "cal.json",
            ]
        )
        == 0
    )
    assert seen["calibration_receipt"] == "cal.json"


def test_tracking_cli_passes_protocol_to_production_verifier(tmp_path, monkeypatch):
    seen = {}

    def run(payload, out, **kwargs):
        seen.update(kwargs)
        return {"status": "DISPATCH_FIXTURE_ONLY"}

    monkeypatch.setitem(
        sys.modules, "src.modeling_v3.tracking_offline", types.SimpleNamespace(track_offline=run)
    )
    payload = tmp_path / "tracking.json"
    payload.write_text("{}")
    assert (
        cli.main(
            [
                "track-offline",
                "--config",
                str(CONFIG),
                "--out",
                str(tmp_path / "out"),
                "--inputs",
                str(payload),
            ]
        )
        == 0
    )
    assert seen["config"]["scope"]["online_ssvc"] is False
    assert not seen.get("fixture", False)


@pytest.mark.parametrize("command", ["validate-cpu", "analyze-cpu"])
def test_cpu_confirmation_preserves_calibration_original_path(tmp_path, monkeypatch, command):
    seen = {}

    def run(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "DISPATCH_FIXTURE_ONLY"}

    monkeypatch.setattr(cli, "verify_selection_lock", lambda *args: {"selected": {}})
    monkeypatch.setitem(
        sys.modules, "src.modeling_v3.cpu_campaign", types.SimpleNamespace(validate_cpu=run)
    )
    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.cpu_results",
        types.SimpleNamespace(analyze_test=run, analyze_calibration=run),
    )
    original = tmp_path / "calibration.json"
    original.write_text("{}")
    argv = [
        command,
        "--config",
        str(CONFIG),
        "--out",
        str(tmp_path / "out"),
        "--lock",
        "lock.json",
        "--calibration-receipt",
        str(original),
    ]
    if command == "validate-cpu":
        argv += ["--role", "locked_test", "--existing-run-roots", "prior_runs"]
    else:
        argv += ["--inputs", "test_response"]
    assert cli.main(argv) == 0
    assert seen["calibration_receipt"] == str(original)
