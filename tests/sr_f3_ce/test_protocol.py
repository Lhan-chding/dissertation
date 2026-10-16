import json

import pytest

from sr_f3_ce.protocol import contained, stable_seed, verify_source


def test_registered_paths_reject_escape_and_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("unrelated")
    root = tmp_path / "run"
    root.mkdir()
    (root / "link").symlink_to(outside)
    for path in ("../outside", str(outside), "link"):
        with pytest.raises(PermissionError):
            contained(root, path)


def test_source_manifest_detects_changed_bytes(tmp_path):
    from sr_f3_ce.protocol import file_hash

    source = tmp_path / "trainer.py"
    source.write_text("fixed")
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "trainer.py", "bytes": 5, "sha256": file_hash(source)}]})
    )
    assert verify_source(manifest, tmp_path)
    source.write_text("other")
    with pytest.raises(PermissionError):
        verify_source(manifest, tmp_path)


def test_stream_seeds_are_separate_and_reproducible():
    args = ("train", 73101, 1, "qid", 0)
    assert stable_seed(*args) == stable_seed(*args)
    assert stable_seed(*args) != stable_seed("monitor", *args[1:])


@pytest.fixture
def selected_source(tmp_path, monkeypatch):
    from sr_f3_ce import ablation, protocol, protocol_revision

    root, original = tmp_path / "selected", tmp_path / "ablation"
    original.mkdir()
    root.mkdir()

    def write(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    request = root / "authorization/REQUEST.md"
    request.parent.mkdir()
    request.write_text("adopted request fixture")
    monkeypatch.setattr(protocol_revision, "REQUEST_SHA256", protocol.file_hash(request))
    config = protocol_revision.revision_config({"generation": {"prefill": "{"}}, "v2")
    write("plan/config/SR_F3_CE.json", config)
    write("plan/PACKAGE_SHA256.json", {"fixture": True})
    write("plan/manifests/MODEL_INPUTS.jsonl", {"fixture": True})
    manifest = write("deployment/SOURCE_MANIFEST.json", {"code_commit": "a" * 40, "files": []})
    selection = {
        "status": "PROTOCOL_SELECTED",
        "selected_revision": "v2",
        "ablation_root": str(original),
        "candidate_identities": {
            "P2": {
                "protocol_revision": "v2",
                "package_sha256": protocol.file_hash(root / "plan/PACKAGE_SHA256.json"),
                "config_sha256": protocol.file_hash(root / "plan/config/SR_F3_CE.json"),
                "model_inputs_sha256": protocol.file_hash(
                    root / "plan/manifests/MODEL_INPUTS.jsonl"
                ),
            }
        },
    }
    selected = write("PROTOCOL_SELECTION.json", selection)
    (original / selected.name).write_bytes(selected.read_bytes())
    monkeypatch.setattr(
        ablation, "verify_selection", lambda p: selection if p == original else None
    )
    identity = {
        "experiment_id": protocol.EXPERIMENT_ID,
        "root": str(root),
        "code_commit": "a" * 40,
        "run_stage": "SELECTED_PROTOCOL_REVISION",
        "protocol_revision": "v2",
        "protocol_revision_request_path": "authorization/REQUEST.md",
        "protocol_selection_path": "PROTOCOL_SELECTION.json",
        "ablation_root": str(original),
        "bound_files": {
            "deployment/SOURCE_MANIFEST.json": protocol.file_hash(manifest),
            "authorization/REQUEST.md": protocol.file_hash(request),
            "PROTOCOL_SELECTION.json": protocol.file_hash(selected),
        },
    }
    write("RUN_IDENTITY.json", identity)
    return root, original, identity, selection, write


def test_adopted_revision_source_requires_complete_fixed_selection(selected_source):
    from sr_f3_ce.protocol import file_hash, verify_run_source_revision

    root, original, *_ = selected_source
    result = verify_run_source_revision(root)
    assert result["protocol_revision"] == "v2"
    assert result["code_commit"] == "a" * 40
    assert result["protocol_selection_sha256"] == file_hash(root / "PROTOCOL_SELECTION.json")
    assert result["ablation_root"] == str(original)


@pytest.mark.parametrize(
    "mutation", ["stage", "request", "selection", "candidate", "source", "stop"]
)
def test_selected_source_fails_closed_on_binding_changes(selected_source, mutation):
    from sr_f3_ce.protocol import verify_run_source_revision

    root, original, identity, _, write = selected_source
    if mutation == "stage":
        write("RUN_IDENTITY.json", dict(identity, run_stage="PROTOCOL_ABLATION"))
    elif mutation == "request":
        (root / "authorization/REQUEST.md").write_text("different authorization")
    elif mutation == "selection":
        (original / "PROTOCOL_SELECTION.json").write_text("{}")
    elif mutation == "candidate":
        write("plan/manifests/MODEL_INPUTS.jsonl", {"changed": True})
    elif mutation == "source":
        write("deployment/SOURCE_MANIFEST.json", {"code_commit": "b" * 40, "files": []})
    else:
        (root / "STOP").write_text("stop")
    with pytest.raises(PermissionError):
        verify_run_source_revision(root)


@pytest.mark.parametrize(
    "status,revision",
    [
        ("STACK_REPRODUCTION_FAILURE", None),
        ("STOP_PROTOCOL_FAILURE", None),
        ("PROTOCOL_SELECTED", "c2"),
    ],
)
def test_nonselected_or_wrong_revision_cannot_authorize_new_science(
    selected_source, status, revision
):
    from sr_f3_ce.protocol import file_hash, verify_run_source_revision

    root, original, identity, selection, write = selected_source
    selection.update(status=status, selected_revision=revision)
    path = write("PROTOCOL_SELECTION.json", selection)
    (original / path.name).write_bytes(path.read_bytes())
    identity["bound_files"][path.name] = file_hash(path)
    write("RUN_IDENTITY.json", identity)
    with pytest.raises(PermissionError, match="complete successful"):
        verify_run_source_revision(root)


def test_original_operations_keep_original_source_pin(selected_source):
    from sr_f3_ce.protocol import file_hash, verify_run_source_revision

    root, _, identity, _, write = selected_source
    write("plan/config/SR_F3_CE.json", {"generation": {"prefill": "{"}})
    with pytest.raises(PermissionError, match="pinned to v1"):
        verify_run_source_revision(root)
    commit = "71483fbf31d57213fdc7e6f3c881848397b45329"
    manifest = write("deployment/SOURCE_MANIFEST.json", {"code_commit": commit, "files": []})
    identity.update(code_commit=commit, run_stage="ORIGINAL")
    identity["bound_files"]["deployment/SOURCE_MANIFEST.json"] = file_hash(manifest)
    write("RUN_IDENTITY.json", identity)
    assert verify_run_source_revision(root)["protocol_revision"] == "v1"
