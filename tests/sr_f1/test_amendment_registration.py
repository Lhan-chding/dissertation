"""The protocol revision is separately registered; unrelated changes fail closed."""

import copy
import json
from pathlib import Path

import pytest

from mm_core.execution import atomic_json
from sr_f1.amendment import (
    AMENDMENT_ID,
    EFFECTIVE_CONFIG,
    ORIGINAL_FREEZE_SHA256,
    approved_changes,
    build_amended_config,
    effective_plan,
    protocol_amendment,
    register_amendment,
    validate_plan,
    verify_amendment,
    verify_protocol_route,
)
from sr_f1.contract import PACKAGE, PLAN_ID, file_hash, load_config


@pytest.fixture
def amended(tmp_path, monkeypatch):
    import sr_f1.amendment as module

    documents = tmp_path / "source_documents"
    documents.mkdir()
    (documents / "USER_SOLUTION_zh.md").write_bytes(
        (module.AMENDMENT_SOURCE_DIR / "USER_SOLUTION_zh.md").read_bytes()
    )
    (documents / "AMENDMENT_zh.md").write_text("SR-F1.1 test registration\n")
    monkeypatch.setattr(module, "AMENDMENT_SOURCE_DIR", documents)
    root = tmp_path / "new_run"
    root.mkdir()
    plan = build_amended_config(1e-4)
    register_amendment(root, plan)
    return root, plan


def test_amendment_keeps_original_configuration_and_only_changes_registered_fields():
    original_hash = file_hash(PACKAGE / "config/SR_F1.json")
    original = load_config()
    amended = build_amended_config()
    assert amended["version"] == PLAN_ID
    assert amended["training"]["lr"] == 1e-4
    assert amended["protocol_amendment"]["id"] == AMENDMENT_ID
    assert amended["protocol_amendment"]["confirmation_samples"] == 512
    assert amended["protocol_amendment"]["minimum_covered_samples"] == 461
    assert amended["protocol_amendment"]["standard_covered_samples"] == 487
    back = copy.deepcopy(amended)
    back.pop("protocol_amendment")
    back["training"]["lr"] = 1e-5
    assert back == original
    amended["training"]["seeds"].append(123)
    assert load_config() == original
    assert file_hash(PACKAGE / "config/SR_F1.json") == original_hash


@pytest.mark.parametrize("lr", [1e-5, 1e-3, True, "0.0001", None])
def test_unselected_learning_rate_options_are_not_execution_overrides(lr):
    with pytest.raises(PermissionError, match="option A"):
        build_amended_config(lr)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "updates", 97),
        ("training", "max_new_tokens", 1024),
        ("training", "lr", 1e-5),
        ("training", "seeds", [71001]),
        ("protocol_amendment", "assistant_prefill", '{"evidence":['),
        ("protocol_amendment", "minimum_covered_samples", 460),
        ("protocol_amendment", "allow_original_bridge", True),
        ("protocol_amendment", "append_synthetic_eos", True),
    ],
)
def test_amendment_identifier_does_not_authorize_unregistered_changes(section, key, value):
    plan = build_amended_config()
    plan[section][key] = value
    with pytest.raises(PermissionError, match="authenticated contract/amendment"):
        validate_plan(plan)


def test_original_remains_explicitly_supported(tmp_path):
    assert validate_plan(load_config()) == load_config()
    assert effective_plan(tmp_path) == load_config()
    assert protocol_amendment(load_config()) is None
    assert approved_changes(load_config()) == []
    assert register_amendment(tmp_path, load_config()) is None


def test_registration_is_immutable_and_preserves_original_freeze_reference(amended):
    root, plan = amended
    first = (root / "AMENDMENT.json").read_bytes()
    receipt = register_amendment(root, plan)
    assert (root / "AMENDMENT.json").read_bytes() == first
    assert receipt["original_freeze_sha256"] == ORIGINAL_FREEZE_SHA256
    assert receipt["formal_test_results_seen_before_freeze"] is False
    assert receipt["inherited_or_prior_run_results_used_to_modify_protocol"] is True
    assert receipt["user_selected_learning_rate_option"] == "A"
    assert receipt["prior_bridged_adapter_used_for_scientific_start"] is False
    assert effective_plan(root) == plan
    assert not (root / "EXECUTION_FREEZE.json").exists()
    with pytest.raises(PermissionError, match="original plan"):
        register_amendment(root, load_config())
    with pytest.raises(PermissionError, match="conflicts"):
        verify_amendment(root, load_config())


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_freeze_sha256", "0" * 64),
        ("formal_test_results_seen_before_freeze", True),
        ("inherited_or_prior_run_results_used_to_modify_protocol", False),
        ("user_selected_learning_rate_option", "B"),
        ("prior_bridged_adapter_used_for_scientific_start", True),
    ],
)
def test_registration_rejects_false_provenance(amended, key, value):
    root, plan = amended
    receipt = json.loads((root / "AMENDMENT.json").read_text())
    receipt[key] = value
    atomic_json(root / "AMENDMENT.json", receipt)
    with pytest.raises(PermissionError, match="Registered amendment differs"):
        verify_amendment(root, plan)


@pytest.mark.parametrize("relative", ["amendment/AMENDMENT_zh.md", "amendment/USER_SOLUTION_zh.md"])
def test_registration_verifies_actual_document_bytes(amended, relative):
    root, plan = amended
    (root / relative).write_text("Changed document")
    with pytest.raises(PermissionError, match="document changed"):
        verify_amendment(root, plan)


def test_registration_rejects_effective_config_tampering(amended):
    root, plan = amended
    tampered = copy.deepcopy(plan)
    tampered["training"]["updates"] = 95
    atomic_json(root / EFFECTIVE_CONFIG, tampered)
    with pytest.raises(PermissionError, match="Registered amendment differs"):
        verify_amendment(root, plan)
    with pytest.raises(PermissionError, match="authenticated contract/amendment"):
        effective_plan(root)


def test_amendment_must_not_be_installed_over_previous_model_work(tmp_path):
    (tmp_path / "engineering/format").mkdir(parents=True)
    (tmp_path / "engineering/format/sample.json").write_text("{}")
    with pytest.raises(PermissionError, match="before model generation"):
        register_amendment(tmp_path, build_amended_config())
    assert not (tmp_path / "AMENDMENT.json").exists()


def test_existing_frozen_execution_is_never_overwritten(tmp_path):
    atomic_json(tmp_path / "EXECUTION_FREEZE.json", {"status": "FROZEN"})
    before = file_hash(tmp_path / "EXECUTION_FREEZE.json")
    with pytest.raises(PermissionError, match="original execution freeze"):
        register_amendment(tmp_path, build_amended_config())
    assert file_hash(tmp_path / "EXECUTION_FREEZE.json") == before


def test_prefill_identity_is_required_for_every_prepared_amended_route():
    plan = build_amended_config()
    route = {
        "protocol_amendment": protocol_amendment(plan),
        "protocol_amendment_id": AMENDMENT_ID,
        "assistant_prefill": "{",
        "assistant_prefill_token_ids": [90],
    }
    verify_protocol_route(route, plan)
    for field in route:
        incomplete = copy.deepcopy(route)
        incomplete.pop(field)
        with pytest.raises(PermissionError, match="prefill protocol"):
            verify_protocol_route(incomplete, plan)
    verify_protocol_route({}, load_config())
    with pytest.raises(PermissionError, match="Original processor route"):
        verify_protocol_route(route, load_config())


@pytest.mark.parametrize(
    "field", ["protocol_amendment", "amendment_sha256", "original_freeze_sha256"]
)
def test_execution_freeze_binds_registered_amendment_identity(amended, field):
    from sr_f1.freeze import verify_execution

    root, plan = amended
    freeze = {
        "plan_id": PLAN_ID,
        "status": "FROZEN",
        "run_root": str(root),
        "config_sha256": file_hash(root / EFFECTIVE_CONFIG),
        "protocol_amendment": plan["protocol_amendment"],
        "amendment_sha256": file_hash(root / "AMENDMENT.json"),
        "original_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "inherited_or_prior_run_results_used_to_modify_protocol": True,
        "uploaded_config_sha256": file_hash(PACKAGE / "config/SR_F1.json"),
    }
    freeze[field] = None
    atomic_json(root / "EXECUTION_FREEZE.json", freeze)
    with pytest.raises(PermissionError, match="Frozen amendment identity differs"):
        verify_execution(plan, root)


def test_local_amended_preparation_registers_but_does_not_claim_gpu_acceptance(
    amended, monkeypatch
):
    import sr_f1.prepare as module

    root, plan = amended
    monkeypatch.setattr(module, "_run_reference_checks", lambda _: {"status": "PASS"})

    def fake_render(root, *_):
        (root / "RENDER_RECEIPT.json").write_text("{}")
        return {}

    monkeypatch.setattr(module, "render_inputs", fake_render)
    result = module.prepare_run(
        root,
        plan=plan,
        font_path=Path("unused"),
        bold_path=Path("unused"),
        operator="test",
        local_only=True,
    )
    assert result["status"] == "LOCAL_PREPARED"
    assert result["gpu_execution_authorized"] is False
    assert effective_plan(root) == plan
    assert file_hash(root / "config/SR_F1.json") == file_hash(PACKAGE / "config/SR_F1.json")
