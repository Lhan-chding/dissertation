"""Portable N3 transfer verifies bytes and never recomputes scientific selection."""

import copy
import json
from pathlib import Path

import pytest

from src.modeling_contrast.io import RunWriter, sha256_file, verify_run_manifest
from src.modeling_contrast.selection import canonical_hash, verify_selection_lock
from src.modeling_contrast.server_transfer import build_server_transfer, write_server_transfer


def fixture(tmp_path, monkeypatch):
    from src.modeling_contrast import server_transfer as module

    old = Path("/original/project")
    project = tmp_path / "server"
    evidence = project / "evidence"
    snapshot = project / "original_source"
    for folder in (evidence, snapshot, project / "src"):
        folder.mkdir(parents=True, exist_ok=True)
    for name, data in (("data.npz", b"data"), ("packet.npz", b"packet")):
        (evidence / name).write_bytes(data)
    (snapshot / "source.py").write_text("original = True\n")
    (project / "src/source.py").write_text("server = True\n")
    original_config = {"resources": {"max_added_output_gib": 3}, "science": {"seeds": [501, 601]}}
    amended = {**copy.deepcopy(original_config), "execution_amendment": {"scope": "resource_only"}}
    amended["resources"]["max_added_output_gib"] = 6
    amendment = project / "protocol_server.json"
    amendment.write_text(json.dumps(amended))
    monkeypatch.setattr(module, "validate_config", lambda config: None)
    monkeypatch.setattr(module, "load_config", lambda: original_config)
    monkeypatch.setattr(module, "config_sha256", lambda config: sha256_file(amendment))
    monkeypatch.setattr(module, "CONFIG_SHA256", "1" * 64)
    monkeypatch.setattr(
        module,
        "resource_gate",
        lambda forecast, config: {
            "status": "PASS"
            if forecast["added_output_bytes"] <= 6 * 2**30
            else "RESOURCE_REVIEW_REQUIRED",
            "passed": forecast["added_output_bytes"] <= 6 * 2**30,
            "forecast": forecast,
            "reasons": [] if forecast["added_output_bytes"] <= 6 * 2**30 else ["disk"],
        },
    )
    source = {str(old / "src/source.py"): sha256_file(snapshot / "source.py")}
    inputs = {str(old / "evidence/data.npz"): sha256_file(evidence / "data.npz")}
    packets = {str(old / "evidence/packet.npz"): sha256_file(evidence / "packet.npz")}
    candidate = {
        "configuration_id": "fixed",
        "actual_rank": 4,
        "n": 64,
        "science_pass": True,
        "cost_pass": True,
        "pX_nrmse": 0.64,
        "v_nrmse": 0.42,
        "stable_improvement_C0": True,
        "stable_improvement_C1": True,
    }
    lock = {
        "status": "RESOURCE_REVIEW_REQUIRED",
        "allow_fresh_cpu": False,
        "selected": [candidate],
        "candidate_audit": [candidate],
        "statistics": {"bootstrap_seed": 77},
        "best_nonzero_diagnostic": candidate,
        "cost_frontier": [{"frozen": True}],
        "science_gate": {"status": "PASS"},
        "cost_gate": {"status": "PASS"},
        "input_identity_gate": {"status": "PASS"},
        "data_gate": {"status": "PASS", "fresh_seed_inventory_roots": ["/original"]},
        "resource_gate": {"status": "FAIL", "forecast": {"added_output_gib": 4}},
        "protocol_sha256": "1" * 64,
        "source_hashes": source,
        "input_hashes": inputs,
        "packet_hashes": packets,
        "selector_source_hashes": source,
    }
    lock["binding"] = {
        "source": canonical_hash(source),
        "data": canonical_hash(inputs),
        "packet": canonical_hash(packets),
        "selector": canonical_hash(lock["selected"]),
        "config": lock["protocol_sha256"],
    }
    lock["lock_sha256"] = canonical_hash(lock)
    original = project / "runs/modeling_contrast_v2/N3"
    with RunWriter(original, lock["binding"], project_root=project) as writer:
        writer.write_json("MODEL_SELECTION_LOCK.json", lock)
        writer.write_json("PAIRED_BOOTSTRAP.json", [{"ci975": -0.2, "seed": 77}])
    amended["execution_amendment"]["original_selection_lock_sha256"] = sha256_file(
        original / "MODEL_SELECTION_LOCK.json"
    )
    amendment.write_text(json.dumps(amended))
    kwargs = dict(
        prefix_mappings={str(old): str(project)},
        original_project_root=str(old),
        original_source_prefix_mappings={str(old / "src"): str(snapshot)},
        amended_config=amended,
        amendment_path=amendment,
        source_hashes={str(project / "src/source.py"): sha256_file(project / "src/source.py")},
        resource_forecast={
            "wall_seconds": 100,
            "peak_ram_gib": 1,
            "added_output_bytes": 4 * 2**30,
            "temporary_bytes": 0,
        },
        server_seed_inventory_roots=[str(project)],
    )
    return project, original / "MODEL_SELECTION_LOCK.json", lock, kwargs


def test_rejects_another_internally_valid_lock_outside_pinned_amendment(tmp_path, monkeypatch):
    _, original, lock, kwargs = fixture(tmp_path, monkeypatch)
    lock["statistics"]["bootstrap_seed"] = 78
    lock["lock_sha256"] = canonical_hash(
        {key: value for key, value in lock.items() if key != "lock_sha256"}
    )
    original.write_text(json.dumps(lock))
    manifest_path = original.parent / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][original.name] = sha256_file(original)
    manifest_path.write_text(json.dumps(manifest))
    assert verify_selection_lock(lock)
    assert verify_run_manifest(original.parent)["status"] == "COMPLETE"
    with pytest.raises(ValueError, match=r"original selection file.*amendment"):
        build_server_transfer(original, **kwargs)


def test_transfer_preserves_science_bytes_and_verifies_distinct_completed_writer(
    tmp_path, monkeypatch
):
    project, original, lock, kwargs = fixture(tmp_path, monkeypatch)
    original_bytes = original.read_bytes()
    payload = build_server_transfer(original, **kwargs)
    transferred = payload["lock"]
    for field in (
        "selected",
        "candidate_audit",
        "statistics",
        "science_gate",
        "cost_gate",
        "best_nonzero_diagnostic",
        "cost_frontier",
    ):
        assert transferred[field] == lock[field]
    assert transferred["resource_gate"]["status"] == "PASS"
    assert transferred["allow_fresh_cpu"] is True
    assert transferred["data_gate"]["fresh_seed_inventory_roots"] == [str(project)]
    assert verify_selection_lock(transferred)
    out = project / "runs/modeling_contrast_v2/N3_server"
    with RunWriter(out, transferred["binding"], project_root=project) as writer:
        write_server_transfer(writer, payload)
    assert verify_run_manifest(out)["status"] == "COMPLETE"
    assert (out / "ORIGINAL_MODEL_SELECTION_LOCK.json").read_bytes() == original_bytes
    assert original.read_bytes() == original_bytes
    assert (out / "original_N3/PAIRED_BOOTSTRAP.json").read_bytes() == (
        original.parent / "PAIRED_BOOTSTRAP.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "file", ["evidence/data.npz", "evidence/packet.npz", "original_source/source.py"]
)
def test_any_changed_original_evidence_is_rejected(tmp_path, monkeypatch, file):
    project, original, _, kwargs = fixture(tmp_path, monkeypatch)
    (project / file).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        build_server_transfer(original, **kwargs)


def test_unmapped_paths_and_changed_scientific_config_fail_closed(tmp_path, monkeypatch):
    _, original, _, kwargs = fixture(tmp_path, monkeypatch)
    kwargs["prefix_mappings"] = {"/unrelated": "/nowhere"}
    with pytest.raises(ValueError, match="unmapped"):
        build_server_transfer(original, **kwargs)
    _, original, _, kwargs = fixture(tmp_path / "second", monkeypatch)
    kwargs["amended_config"]["science"]["seeds"] = [502]
    Path(kwargs["amendment_path"]).write_text(json.dumps(kwargs["amended_config"]))
    with pytest.raises(ValueError, match=r"resource.only"):
        build_server_transfer(original, **kwargs)


def test_resource_failure_does_not_authorize_and_bootstrap_tamper_is_rejected(
    tmp_path, monkeypatch
):
    _, original, _, kwargs = fixture(tmp_path, monkeypatch)
    kwargs["resource_forecast"]["added_output_bytes"] = 7 * 2**30
    result = build_server_transfer(original, **kwargs)
    assert result["lock"]["allow_fresh_cpu"] is False
    assert result["lock"]["status"] == "RESOURCE_REVIEW_REQUIRED"
    (original.parent / "PAIRED_BOOTSTRAP.json").write_text("[]")
    with pytest.raises(ValueError, match="hash"):
        build_server_transfer(original, **kwargs)


def test_writer_rejects_other_stage_and_changed_prepared_payload(tmp_path, monkeypatch):
    project, original, _, kwargs = fixture(tmp_path, monkeypatch)
    payload = build_server_transfer(original, **kwargs)
    with (
        RunWriter(
            project / "runs/modeling_contrast_v2/wrong",
            payload["lock"]["binding"],
            project_root=project,
        ) as writer,
        pytest.raises(ValueError, match="N3_server"),
    ):
        write_server_transfer(writer, payload)
    payload["lock"]["selected"][0]["actual_rank"] = 1
    with (
        RunWriter(
            project / "runs/modeling_contrast_v2/N3_server",
            payload["lock"]["binding"],
            project_root=project,
        ) as writer,
        pytest.raises(ValueError, match=r"payload|lock"),
    ):
        write_server_transfer(writer, payload)


def test_longest_component_prefix_wins_and_collapsed_paths_are_rejected(tmp_path, monkeypatch):
    project, original, _, kwargs = fixture(tmp_path, monkeypatch)
    kwargs["prefix_mappings"] = {
        "/original": str(tmp_path / "wrong"),
        "/original/project": str(project),
    }
    assert build_server_transfer(original, **kwargs)["lock"]["allow_fresh_cpu"]
    kwargs["prefix_mappings"] = {"/original/projected": str(project)}
    with pytest.raises(ValueError, match="unmapped"):
        build_server_transfer(original, **kwargs)
    from src.modeling_contrast import server_transfer as module

    same = {"/a/data": "a" * 64, "/b/data": "a" * 64}
    # Same-content aliasing is still an ambiguous many-to-one transfer.
    path = project / "alias"
    path.write_bytes(b"aliased")
    same = {key: sha256_file(path) for key in same}
    with pytest.raises(ValueError, match="collapse"):
        module._verified_mapping(
            same, module._mappings({"/a/data": str(path), "/b/data": str(path)}), Path("/"), "test"
        )


def test_symlink_escape_and_changed_current_source_are_rejected(tmp_path, monkeypatch):
    project, original, _, kwargs = fixture(tmp_path, monkeypatch)
    data = project / "evidence/data.npz"
    outside = tmp_path / "outside"
    outside.write_bytes(data.read_bytes())
    data.unlink()
    data.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        build_server_transfer(original, **kwargs)
    project, original, _, kwargs = fixture(tmp_path / "second", monkeypatch)
    (project / "src/source.py").write_text("changed")
    with pytest.raises(ValueError, match="server source hash"):
        build_server_transfer(original, **kwargs)


def test_recheck_between_prepare_and_write_detects_changed_evidence(tmp_path, monkeypatch):
    project, original, _, kwargs = fixture(tmp_path, monkeypatch)
    payload = build_server_transfer(original, **kwargs)
    (project / "evidence/data.npz").write_bytes(b"changed after preparation")
    with RunWriter(
        project / "runs/modeling_contrast_v2/N3_server",
        payload["lock"]["binding"],
        project_root=project,
    ) as writer:
        with pytest.raises(ValueError, match="hash"):
            write_server_transfer(writer, payload)
        assert writer.output_hashes == {}
