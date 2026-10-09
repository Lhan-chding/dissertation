"""Bounded source amendments preserve the original freeze and all prior evidence."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from mm_core.execution import object_hash
from sr_f1.contract import PACKAGE, PLAN_ID, file_hash
from sr_f1.freeze import (
    TECHNICAL_REPAIR_ID,
    _verify_execution_source,
    verify_technical_repair,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def hashes(contents):
    import hashlib

    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in contents.items()}


def refresh_change_ledger(repair, actual):
    before, after = repair["original_source_file_hashes"], actual["source_file_hashes"]
    actual["source_tree_sha256"] = object_hash(after)
    for key in ("source_file_hashes", "source_tree_sha256", "source_commit", "source_dirty_files"):
        repair[key] = copy.deepcopy(actual[key])
    repair["changed_files"] = [
        {"path": name, "before_sha256": before.get(name), "after_sha256": after[name]}
        for name in sorted(after)
        if before.get(name) != after[name]
    ]


@pytest.fixture
def repair_case(tmp_path, monkeypatch):
    before_text = {
        "src/sr_f1/freeze.py": "original source gate\n",
        "src/sr_f1/runtime.py": "original format guard\n",
        "src/sr_f1/orchestration.py": "original transitions\n",
        "scripts/sr_f1/submit_matrix.py": "original CLI\n",
        "src/sr_f1/training.py": "scientific training must stay identical\n",
        "src/sr_f1/data.py": "model-visible inputs must stay identical\n",
        "pyproject.toml": "[project]\nname='frozen'\n",
    }
    for name, content in before_text.items():
        path = tmp_path / "code_before_format_guard_repair" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    before = hashes(before_text)
    input_path = tmp_path / "config/SR_F1.json"
    input_path.parent.mkdir()
    input_path.write_bytes((PACKAGE / "config/SR_F1.json").read_bytes())
    original = {
        "status": "CPU_VERIFIED",
        "source_file_hashes": before,
        "source_tree_sha256": object_hash(before),
        "source_commit": "a" * 40,
        "package_input_hashes": {"config/SR_F1.json": file_hash(input_path)},
    }
    source_path = tmp_path / "SOURCE_AND_RENDER_MANIFEST.json"
    write(source_path, original)
    freeze = {
        "plan_id": PLAN_ID,
        "status": "FROZEN",
        "run_root": str(tmp_path),
        "source_file_hashes": before,
        "source_tree_sha256": object_hash(before),
        "source_commit": "a" * 40,
        "artifact_hashes": {"SOURCE_AND_RENDER_MANIFEST.json": file_hash(source_path)},
    }
    freeze_path = tmp_path / "EXECUTION_FREEZE.json"
    write(freeze_path, freeze)
    review_path = tmp_path / "FORMAT_TECHNICAL_REVIEW.json"
    write(
        review_path,
        {
            "plan_id": PLAN_ID,
            "freeze_sha256": file_hash(freeze_path),
            "status": "VERIFIED_PROTOCOL_NONADHERENCE",
        },
    )
    calls = []

    def verify_review(root):
        calls.append(root)
        return {"status": "VERIFIED_PROTOCOL_NONADHERENCE"}

    monkeypatch.setitem(
        sys.modules, "sr_f1.format_review", SimpleNamespace(verify_format_review=verify_review)
    )
    after_text = dict(before_text)
    for name in (
        "src/sr_f1/freeze.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/orchestration.py",
        "scripts/sr_f1/submit_matrix.py",
    ):
        after_text[name] += "scoped technical repair\n"
    after_text["src/sr_f1/format_review.py"] = "new CPU-only technical review\n"
    after = hashes(after_text)
    actual = {
        "source_file_hashes": after,
        "source_tree_sha256": object_hash(after),
        "source_commit": "b" * 40,
        "source_dirty_files": [],
    }
    repair = {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "repair_id": TECHNICAL_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_REPAIR",
        "reason": "Audit preserved FORMAT failures before the registered one-time bridge.",
        "original_freeze_sha256": file_hash(freeze_path),
        "original_source_manifest_sha256": file_hash(source_path),
        "original_source_tree_sha256": original["source_tree_sha256"],
        "original_source_commit": "a" * 40,
        "original_source_file_hashes": before,
        "preserved_source_root": "code_before_format_guard_repair",
        "scientific_contract_changes": [],
        "format_review_sha256": file_hash(review_path),
    }
    refresh_change_ledger(repair, actual)
    write(tmp_path / "TECHNICAL_REPAIR.json", repair)
    return SimpleNamespace(
        root=tmp_path, original=original, actual=actual, repair=repair, calls=calls
    )


def test_original_source_still_passes_without_any_amendment(repair_case):
    case = repair_case
    (case.root / "TECHNICAL_REPAIR.json").unlink()
    original_actual = {**case.actual, **case.original}
    assert _verify_execution_source(case.root, case.original, original_actual) is None
    assert case.calls == []


def test_scoped_repair_preserves_original_freeze_and_dispatches_review_verifier(repair_case):
    case = repair_case
    freeze_path = case.root / "EXECUTION_FREEZE.json"
    source_path = case.root / "SOURCE_AND_RENDER_MANIFEST.json"
    before = freeze_path.read_bytes(), source_path.read_bytes()
    result = _verify_execution_source(case.root, case.original, case.actual)
    assert result["status"] == "VERIFIED_TECHNICAL_REPAIR"
    assert result["original_freeze_sha256"] == file_hash(freeze_path)
    assert result["repair_sha256"] == file_hash(case.root / "TECHNICAL_REPAIR.json")
    assert result["review_sha256"] == file_hash(case.root / "FORMAT_TECHNICAL_REVIEW.json")
    assert case.calls == [case.root]
    assert before == (freeze_path.read_bytes(), source_path.read_bytes())


def test_repair_is_not_invalidated_by_later_authorized_progress(repair_case):
    case = repair_case
    write(case.root / "COMMON_START.json", {"freeze_sha256": case.repair["original_freeze_sha256"]})
    write(case.root / "training/SRF1_A_s71001/progress.json", {"step": 8})
    assert verify_technical_repair(case.root, case.actual)["status"] == "VERIFIED_TECHNICAL_REPAIR"


def test_changed_source_without_amendment_is_rejected(repair_case):
    case = repair_case
    (case.root / "TECHNICAL_REPAIR.json").unlink()
    with pytest.raises(FileNotFoundError):
        _verify_execution_source(case.root, case.original, case.actual)


@pytest.mark.parametrize(
    "field,value",
    [
        ("original_freeze_sha256", "0" * 64),
        ("original_source_manifest_sha256", "0" * 64),
        ("original_source_tree_sha256", "0" * 64),
        ("original_source_commit", "c" * 40),
        ("source_commit", "c" * 40),
        ("source_tree_sha256", "0" * 64),
        ("source_dirty_files", [" M src/sr_f1/runtime.py"]),
        ("scientific_contract_changes", ["change the reward"]),
        ("preserved_source_root", "code_precommit_attempt01"),
        ("format_review_sha256", "0" * 64),
    ],
)
def test_amendment_cannot_substitute_identities(repair_case, field, value):
    case = repair_case
    case.repair[field] = value
    write(case.root / "TECHNICAL_REPAIR.json", case.repair)
    with pytest.raises(PermissionError):
        verify_technical_repair(case.root, case.actual)


@pytest.mark.parametrize(
    "name",
    [
        "src/sr_f1/training.py",
        "src/sr_f1/data.py",
        "src/mm_core/vl_runtime.py",
        "scripts/sr_f1/run_worker.py",
        "pyproject.toml",
    ],
)
def test_explicit_allowlist_rejects_other_source_changes(repair_case, name):
    case = repair_case
    case.actual["source_file_hashes"][name] = "c" * 64
    refresh_change_ledger(case.repair, case.actual)
    write(case.root / "TECHNICAL_REPAIR.json", case.repair)
    with pytest.raises(PermissionError, match="explicit source-file allowlist"):
        verify_technical_repair(case.root, case.actual)


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "wrong_before", "wrong_after"])
def test_change_ledger_must_exactly_equal_actual_before_after_diff(repair_case, mutation):
    case = repair_case
    if mutation == "omit":
        case.repair["changed_files"].pop()
    elif mutation == "duplicate":
        case.repair["changed_files"].append(case.repair["changed_files"][0])
    else:
        field = "before_sha256" if mutation == "wrong_before" else "after_sha256"
        case.repair["changed_files"][0][field] = "0" * 64
    write(case.root / "TECHNICAL_REPAIR.json", case.repair)
    with pytest.raises(PermissionError, match="file changes are not exact"):
        verify_technical_repair(case.root, case.actual)


def test_snapshot_must_match_frozen_source_not_an_old_precommit_copy(repair_case):
    case = repair_case
    path = case.root / "code_before_format_guard_repair/src/sr_f1/runtime.py"
    path.write_text("old or corrupted source copy")
    with pytest.raises(PermissionError, match="Preserved source differs"):
        verify_technical_repair(case.root, case.actual)


def test_registered_inputs_cannot_change_in_a_source_repair(repair_case):
    case = repair_case
    write(case.root / "config/SR_F1.json", {"training": {"updates": 97}})
    with pytest.raises(PermissionError, match="changed a registered input"):
        verify_technical_repair(case.root, case.actual)


def test_source_deletion_is_not_an_allowed_technical_repair(repair_case):
    case = repair_case
    del case.actual["source_file_hashes"]["src/sr_f1/runtime.py"]
    refresh_change_ledger(case.repair, case.actual)
    write(case.root / "TECHNICAL_REPAIR.json", case.repair)
    with pytest.raises(PermissionError, match="cannot remove frozen source"):
        verify_technical_repair(case.root, case.actual)


def test_dirty_actual_source_cannot_claim_a_clean_repair(repair_case):
    case = repair_case
    case.actual["source_dirty_files"] = [" M src/sr_f1/runtime.py"]
    with pytest.raises(PermissionError, match="Commit the technical repair"):
        verify_technical_repair(case.root, case.actual)


def test_review_artifact_verifier_failure_is_not_bypassed(repair_case, monkeypatch):
    case = repair_case

    def invalid_review(root):
        raise PermissionError("Original FORMAT raw evidence changed")

    monkeypatch.setitem(
        sys.modules, "sr_f1.format_review", SimpleNamespace(verify_format_review=invalid_review)
    )
    with pytest.raises(PermissionError, match="Original FORMAT raw evidence changed"):
        verify_technical_repair(case.root, case.actual)
