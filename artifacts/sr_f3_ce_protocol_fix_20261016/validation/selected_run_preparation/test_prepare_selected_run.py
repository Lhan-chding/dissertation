"""CPU-only failure and immutable copy checks; no Slurm call or selected root."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]
spec = importlib.util.spec_from_file_location(
    "tested_selected_preparation", HERE / "prepare_selected_run.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(WT / "src"))


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("revision", ["v2", "c2"])
def test_actual_tested_candidate_package_can_be_copied_exactly(revision):
    candidates = WT / "artifacts/sr_f3_ce_protocol_fix_20261016/candidates"
    p = candidates / revision / "plan"
    row = dict(
        plan=revision + "/plan",
        package_sha256=m.sha(p / "PACKAGE_SHA256.json"),
        config_sha256=m.sha(p / "config/SR_F3_CE.json"),
        model_inputs_sha256=m.sha(p / "manifests/MODEL_INPUTS.jsonl"),
    )
    selection = dict(
        selected_revision=revision, candidate_identities={dict(v2="P2", c2="P3")[revision]: row}
    )
    plan, names = m.candidate_plan(candidates, selection)
    assert plan == p and len(names) == 104
    assert (
        json.loads((plan / "PROTOCOL_REVISION.json").read_text())["protocol_revision"] == revision
    )


def test_reviewed_prep_files_and_cpu_guard_match_pins():
    for name, digest in m.PREP_FILES.items():
        assert m.sha(HERE / name) == digest
    assert (
        m.sha(HERE.parent / "selected_data_preparation/prepare_selected_data.py")
        == m.DATA_HELPER_SHA
    )


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside", "a/../../outside"])
def test_unsafe_copy_path_rejected_before_creation(tmp_path, relative):
    with pytest.raises(PermissionError):
        m.bounded(tmp_path, relative, exists=False)
    assert list(tmp_path.iterdir()) == []


def test_symlink_copy_path_rejected(tmp_path):
    (tmp_path / "real").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(PermissionError):
        m.bounded(tmp_path, "link")


def test_failed_or_duplicate_write_never_overwrites(tmp_path):
    path = tmp_path / "receipt.json"
    m.write_once(path, b"first")
    with pytest.raises(PermissionError):
        m.write_once(path, b"second")
    assert path.read_bytes() == b"first"


def test_copy_checks_source_before_creating_target(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    expected = m.file_record(source)
    source.write_bytes(b"changed")
    with pytest.raises(PermissionError):
        m.copy_exact(source, tmp_path / "new", expected, [])
    assert not (tmp_path / "new").exists()


def test_copy_preserves_bytes_and_refuses_existing_target(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"exact")
    target = tmp_path / "child/target"
    ledger = []
    m.copy_exact(source, target, m.file_record(source), ledger)
    assert target.read_bytes() == b"exact" and len(ledger) == 1
    with pytest.raises(PermissionError):
        m.copy_exact(source, target, m.file_record(source), ledger)
    assert len(ledger) == 1


def source_fixture(tmp_path, monkeypatch):
    p = tmp_path / "code/src/a.py"
    p.parent.mkdir(parents=True)
    p.write_text("VALUE = 1\n")
    manifest = tmp_path / "deployment/SOURCE_MANIFEST.json"
    json_file(
        manifest, dict(code_commit=m.COMMIT, files=[dict(path="src/a.py", **m.file_record(p))])
    )
    monkeypatch.setattr(m, "SOURCE_SHA", m.sha(manifest))
    return p


def test_complete_scientific_inventory_rejects_extra_native_library(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    assert m.verify_code(tmp_path)["code_commit"] == m.COMMIT
    (tmp_path / "code/extra.so").write_bytes(b"not registered")
    with pytest.raises(PermissionError, match="inventory"):
        m.verify_code(tmp_path)


def test_complete_scientific_inventory_rejects_changed_source(tmp_path, monkeypatch):
    p = source_fixture(tmp_path, monkeypatch)
    p.write_text("VALUE = 2\n")
    with pytest.raises(PermissionError, match="identity"):
        m.verify_code(tmp_path)


def test_existing_selected_root_is_not_reused_or_cleared(tmp_path):
    new = tmp_path / "selected_v2"
    new.mkdir()
    (new / "partial").write_text("retain")
    with pytest.raises(PermissionError, match="never-created"):
        m.stage_static(None, None, None, new, None, None)
    assert (new / "partial").read_text() == "retain"


def test_external_phase_refuses_existing_identity_before_loading_any_runtime(tmp_path):
    (tmp_path / "RUN_IDENTITY.json").write_text("{}")
    with pytest.raises(PermissionError, match="single pre-identity"):
        m.stage_external(tmp_path, {}, {})


@pytest.mark.parametrize("native_revision", ["v1", "c2"])
def test_seal_rejects_native_receipt_from_another_revision(tmp_path, native_revision):
    prep = tmp_path / "prep"
    json_file(prep / "BOUND_FILES_POLICY.json", dict(proposed_stable_paths=[]))
    new = tmp_path / "new"
    for pool in ("chartqa", "chartqapro"):
        (new / "external" / pool).mkdir(parents=True)
    json_file(new / "native/PROCESSOR_RECEIPT.json", dict(protocol_revision=native_revision))
    json_file(
        new / "native/SUPPORT_MASK_RECEIPT.json",
        dict(protocol_revision=native_revision, serialization="indent_2"),
    )
    config = json.loads(
        (
            WT / "artifacts/sr_f3_ce_protocol_fix_20261016/candidates/v2/plan/config/SR_F3_CE.json"
        ).read_text()
    )
    protocol = SimpleNamespace(verify_package=lambda path: dict(config=config))
    with pytest.raises(PermissionError, match="different protocol revision"):
        m.stage_seal(
            SimpleNamespace(prep_dir=str(prep)), new, dict(selected_revision="v2"), protocol, {}
        )
    assert not (new / "RUN_IDENTITY.json").exists()


def test_auxiliary_import_uses_same_authenticated_bytes_despite_later_disk_change(
    tmp_path, monkeypatch
):
    path = tmp_path / "aux.py"
    path.write_text("VALUE = 1\n")
    expected = m.sha(path)
    original = Path.read_bytes
    calls = []

    def intercepted(self):
        content = original(self)
        if self == path:
            calls.append(self)
            self.write_text("VALUE = 2\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", intercepted)
    module = m.import_file("selected_import_fixture", path, expected)
    assert module.VALUE == 1 and len(calls) == 1


def test_auxiliary_import_rejects_changed_bytes_before_execution(tmp_path):
    path = tmp_path / "aux.py"
    path.write_text('raise RuntimeError("must never execute")\n')
    with pytest.raises(PermissionError, match="changed before import"):
        m.import_file("selected_import_negative", path, "0" * 64)
