"""Cross-module CLI analysis and immutable server authorization boundaries."""

import copy
import json

import pytest

from src.followup_cli import _s1_gate, _smoke_gate, analyze_run, main
from src.followup_protocol import canonical_hash, source_files


def test_analyze_actual_six_bank_cpu_bundle_and_refuse_gpu_upgrade(tmp_path):
    from test_followup_runtime import fixture_context

    from src.followup_runtime import run_followup_bank

    plan, adapter, optimizer, origin = fixture_context()
    out = tmp_path / "S1"
    run_followup_bank(
        plan,
        output_root=out,
        adapter=adapter,
        optimizer=optimizer,
        origin=origin,
        data_root=".",
        fixture=True,
    )
    result = analyze_run(out, tmp_path / "analysis")
    assert result["status"] == "CPU_TESTED"
    assert result["s1_measurement_chain_passed"] is False
    assert len(result["bank_manifests"]) == 6
    with pytest.raises(ValueError, match="measurement"):
        _s1_gate(plan, out / "s1_measurement_chain.json")
    with pytest.raises(FileExistsError):
        analyze_run(out, tmp_path / "analysis")


def test_analyze_rejects_unmeasured_bank_without_writing_output(tmp_path):
    bank = tmp_path / "S1" / "bank00"
    bank.mkdir(parents=True)
    (bank / "status.json").write_text('{"status":"RUNNING"}')
    (bank / "identity.json").write_text("{}")
    with pytest.raises(ValueError, match="not completed"):
        analyze_run(tmp_path / "S1", tmp_path / "analysis")
    assert not (tmp_path / "analysis").exists()


def test_smoke_binding_rejects_plan_source_and_fake_kind_changes(tmp_path):
    plan = {"design": {"version": "test"}}
    evidence = {
        "status": "PASS",
        "execution_kind": "REAL_CUDA_FOLLOWUP_SMOKE",
        "gpu_smoke_passed": True,
        "plan_hash": canonical_hash(plan),
        "source_files": source_files(),
    }
    path = tmp_path / "smoke.json"
    for field, value in [
        ("plan_hash", "other"),
        ("source_files", {}),
        ("execution_kind", "CPU_FAKE_TORCH"),
        ("gpu_smoke_passed", False),
    ]:
        bad = copy.deepcopy(evidence)
        bad[field] = value
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="exact plan and source"):
            _smoke_gate(plan, path)


def test_dry_run_accepts_valid_design_without_gpu_or_output(tmp_path):
    from src.followup_protocol import build_plan, load_design

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(build_plan(load_design("configs/mechanism_followup.yaml"))))
    assert (
        main(
            [
                "run-s1",
                "--validated-plan",
                str(plan),
                "--bank",
                "bank03",
                "--out",
                str(tmp_path / "unused"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not (tmp_path / "unused").exists()


def test_analyze_rejects_mixed_model_origin_or_source_bank_bundles(tmp_path):
    from src.followup_protocol import BANK_IDS, file_hash

    root = tmp_path / "S1"
    for index, bank in enumerate(BANK_IDS):
        path = root / bank
        path.mkdir(parents=True)
        identity = {
            "execution_kind": "CPU_FAKE_TORCH",
            "design_hash": "same-design",
            "origin_state_hash": "same-origin",
            "model_hash": "model-a" if index == 0 else "model-b",
            "data_hash": "same-data",
            "config_hash": "same-config",
            "parser_version_hash": "same-parser",
            "protocol_version": "same-protocol",
            "source_hash": "same-source",
            "validated_plan_hash": "same-plan",
            "bank_id": bank,
        }
        files = {
            "identity.json": identity,
            "status.json": {"status": "CPU_TESTED"},
            "paired_response.json": [],
        }
        for name, value in files.items():
            (path / name).write_text(json.dumps(value))
        (path / "manifest.json").write_text(
            json.dumps(
                {"files": [{"path": name, "sha256": file_hash(path / name)} for name in files]}
            )
        )
    with pytest.raises(ValueError, match=r"identity|binding"):
        analyze_run(root, tmp_path / "analysis")
    assert not (root / "s1_measurement_chain.json").exists()


def test_parent_audit_cannot_write_inside_original_directory(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    with pytest.raises(SystemExit) as result:
        main(
            [
                "audit-parent",
                "--design",
                "configs/mechanism_followup.yaml",
                "--parent",
                str(parent),
                "--out",
                str(parent / "new-audit.json"),
            ]
        )
    assert result.value.code == 2
    assert not list(parent.iterdir())


def test_prepare_cli_cannot_publish_plan_inside_parent(tmp_path, monkeypatch):
    from src import followup_backend as backend

    parent = tmp_path / "parent"
    parent.mkdir()
    paths = {key: str(parent) for key in backend.DIRECTORY_KEYS}
    pathfile = tmp_path / "paths.json"
    pathfile.write_text(json.dumps(paths))
    monkeypatch.setattr(backend, "prepare_server_plan", lambda *a, **k: {"paths": paths})
    with pytest.raises(SystemExit) as result:
        main(
            [
                "prepare-server",
                "--design",
                "configs/mechanism_followup.yaml",
                "--paths",
                str(pathfile),
                "--out",
                str(parent / "validated.json"),
            ]
        )
    assert result.value.code == 2
    assert not list(parent.iterdir())
