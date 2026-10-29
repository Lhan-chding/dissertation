"""Local request assembly guards only; no scheduler, evidence or scientific PASS."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6.phases import source_delta


@pytest.fixture
def builder(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts/sr_f6/pilot_v3_operations/build_prepare_submission.py"
    spec = importlib.util.spec_from_file_location("pilot_ops_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path / "pilot"
    base.mkdir()
    before = {"files": [{"path": "src/x.py", "bytes": 1, "sha256": "a" * 64}]}
    after = {"files": [{"path": "src/x.py", "bytes": 2, "sha256": "b" * 64}]}
    reasons = {"src/x.py": "Synthetic local operations test; no actual evidence"}
    values = {
        "ORIGINAL_SOURCE_MANIFEST.json": before,
        "EXPECTED_SOURCE_MANIFEST.json": after,
        "SOURCE_CHANGE_REASONS.json": reasons,
        "EXPECTED_DELTA.json": source_delta(before, after, reasons=reasons),
    }
    for name, value in values.items():
        (base / name).write_text(json.dumps(value))
    monkeypatch.setattr(module, "base", base)
    calls = []

    def local_command(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] in (["bash", "-n"], ["git", "ls-files"], ["git", "diff"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", local_command)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "c" * 40)
    return module, base, calls


def test_validate_only_cannot_activate_or_contact_scheduler(builder, monkeypatch, capsys):
    module, base, calls = builder
    monkeypatch.setattr(module.sys, "argv", ["builder", "--validate-only"])
    module.main()
    assert json.loads(capsys.readouterr().out)["status"].endswith("NOT_ACTIVATED")
    assert calls == [["bash", "-n"]]
    assert not (base / "submit_prepare_remote.py").exists()
    assert not (base / "ACTIVATED_PREPARE_REQUEST.json").exists()


def test_missing_confirmation_is_not_synthesized(builder, monkeypatch):
    module, base, _ = builder
    monkeypatch.setattr(module.sys, "argv", ["builder"])
    with pytest.raises(FileNotFoundError, match="CONFIRMATION_GATE_DELIVERY"):
        module.main()
    assert not (base / "submit_prepare_remote.py").exists()


@pytest.mark.parametrize("status", ["FAIL", "UNKNOWN", "REVIEW_COMPLETE_NOT_GATE_SIGNED"])
def test_nonconfirmation_status_cannot_activate(builder, monkeypatch, status):
    module, base, _ = builder
    upstream = base.parent / "confirm_followup_v2_20261029"
    upstream.mkdir()
    (upstream / "CONFIRMATION_GATE_DELIVERY.json").write_text(json.dumps({"status": status}))
    monkeypatch.setattr(module.sys, "argv", ["builder"])
    with pytest.raises(AssertionError):
        module.main()
    assert not (base / "submit_prepare_remote.py").exists()


def test_unexplained_source_delta_stops_before_activation(builder, monkeypatch):
    module, base, calls = builder
    (base / "SOURCE_CHANGE_REASONS.json").write_text("{}")
    monkeypatch.setattr(module.sys, "argv", ["builder"])
    with pytest.raises(PermissionError, match="Every source difference"):
        module.main()
    assert calls == [["bash", "-n"]]
    assert not (base / "submit_prepare_remote.py").exists()
