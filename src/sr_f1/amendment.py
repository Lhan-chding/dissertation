"""Explicit, immutable SR-F1.1 registration separate from the uploaded contract.

The original plan version and seeds remain unchanged. The amendment identity is
an additional required input, never a global/environment-controlled override.
"""

from __future__ import annotations

import copy
from pathlib import Path

from mm_core.execution import atomic_json, object_hash, read_json, utc_now

from .contract import PACKAGE, PLAN_ID, file_hash, load_config

AMENDMENT_ID = "SR-F1.1-20261010"
ORIGINAL_FREEZE_SHA256 = "b84d6d1ad26761cb07e3db706be0b8ca2b4e4a8d436473b0ad3e964a81f463f9"
USER_SOLUTION_SHA256 = "113efeaf2a5f3ad367e3f16577a9ca0be5abdc7d1658604a85a6eb3f075854b5"
AMENDMENT_SOURCE_DIR = PACKAGE.parent / "amendments/SR_F1_1_20261010"
EFFECTIVE_CONFIG = "config/SR_F1_1.json"
REGISTERED_DOCUMENTS = {
    "amendment/USER_SOLUTION_zh.md": "USER_SOLUTION_zh.md",
    "amendment/AMENDMENT_zh.md": "AMENDMENT_zh.md",
}


def build_amended_config(learning_rate=1e-4):
    """Build only the user's selected option A; all other base settings survive."""
    if isinstance(learning_rate, bool) or learning_rate != 1e-4:
        raise PermissionError("SR-F1.1 option A requires learning_rate=1e-4")
    plan = copy.deepcopy(load_config())
    plan["training"]["lr"] = 1e-4
    plan["protocol_amendment"] = {
        "id": AMENDMENT_ID,
        "applies_to": ["evidence_answer"],
        "assistant_prefill": "{",
        "assistant_prefill_token_ids": [90],
        "stopping": "first_balanced_top_level_json_object",
        "score_prefill": True,
        "reject_nonwhitespace_suffix": True,
        "include_balancing_token_in_trajectory": True,
        "append_synthetic_eos": False,
        "common_start": "zero_lora_base",
        "format_samples_per_prompt": 8,
        "confirmation_samples_per_prompt": 16,
        "confirmation_questions": 32,
        "confirmation_samples": 512,
        "minimum_covered_samples": 461,
        "standard_covered_samples": 487,
        "required_field_coverage": 0.90,
        "nominal_field_coverage": 0.95,
        "report_format_failure_per_arm_and_round": True,
        "below_minimum_action": "STOP_PENDING_SECOND_AMENDMENT",
        "allow_original_bridge": False,
        "learning_rate_option": "A",
        "engine_learning_rate": 1e-4,
    }
    return plan


def validate_plan(plan):
    """Reject undeclared changes even when an amendment identifier is present."""
    if isinstance(plan, (str, Path)):
        plan = read_json(plan)
    if not isinstance(plan, dict):
        raise PermissionError("Execution plan differs from authenticated contract")
    if object_hash(plan) not in (object_hash(load_config()), object_hash(build_amended_config())):
        raise PermissionError("Execution plan differs from authenticated contract/amendment")
    return copy.deepcopy(plan)


def protocol_amendment(plan):
    return validate_plan(plan).get("protocol_amendment")


def approved_changes(plan):
    plan = validate_plan(plan)
    if not plan.get("protocol_amendment"):
        return []
    return [
        {"path": "training.lr", "before": 1e-5, "after": 1e-4},
        {"path": "protocol_amendment", "before": None, "after": plan["protocol_amendment"]},
    ]


def _immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise PermissionError("Existing amendment artifact differs: " + str(path))
    else:
        atomic_json(path, value)


def register_amendment(root, plan):
    """Register documents and option A before native input preparation/generation."""
    root = Path(root).resolve(strict=True)
    plan = validate_plan(plan)
    amendment = plan.get("protocol_amendment")
    if amendment is None:
        if (root / "AMENDMENT.json").exists() or (root / EFFECTIVE_CONFIG).exists():
            raise PermissionError("An amended run cannot be prepared using the original plan")
        return None
    destination = root / "AMENDMENT.json"
    if destination.exists():
        return verify_amendment(root, plan)
    for folder in ("engineering", "training", "runs", "evaluation", "sealed", "released"):
        directory = root / folder
        if directory.exists() and any(p.is_file() for p in directory.rglob("*")):
            raise PermissionError("Amendment must be registered before model generation")
    if (root / "COMMON_START.json").exists():
        raise PermissionError("Amendment requires a new isolated execution root")
    freeze_path = root / "EXECUTION_FREEZE.json"
    if freeze_path.exists() and read_json(freeze_path).get("status") == "FROZEN":
        raise PermissionError("Preserve the original execution freeze")
    source = AMENDMENT_SOURCE_DIR / "USER_SOLUTION_zh.md"
    if file_hash(source) != USER_SOLUTION_SHA256:
        raise PermissionError("The user-provided solution document changed")
    hashes = {}
    for relative, source_name in REGISTERED_DOCUMENTS.items():
        original, target = AMENDMENT_SOURCE_DIR / source_name, root / relative
        data = original.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise PermissionError("Existing amendment document differs: " + relative)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        hashes[relative] = file_hash(target)
    _immutable_json(root / EFFECTIVE_CONFIG, plan)
    receipt = {
        "plan_id": PLAN_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "REGISTERED_BEFORE_GENERATION",
        "registered_at": utc_now(),
        "run_root": str(root),
        "original_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "original_freeze_preserved": True,
        "original_config_sha256": file_hash(PACKAGE / "config/SR_F1.json"),
        "approved_solution_sha256": USER_SOLUTION_SHA256,
        "document_hashes": hashes,
        "effective_config": EFFECTIVE_CONFIG,
        "effective_config_sha256": file_hash(root / EFFECTIVE_CONFIG),
        "effective_config_object_sha256": object_hash(plan),
        "protocol_amendment": amendment,
        "changes_from_uploaded_contract": approved_changes(plan),
        "inherited_or_prior_run_results_used_to_modify_protocol": True,
        "inherited_or_prior_run_results_used_to_select_parameters": True,
        "formal_test_results_seen_before_freeze": False,
        "prior_bridged_adapter_used_for_scientific_start": False,
        "user_selected_learning_rate_option": "A",
    }
    _immutable_json(destination, receipt)
    return verify_amendment(root, plan)


def verify_amendment(root, plan):
    root = Path(root).resolve(strict=True)
    plan = validate_plan(plan)
    amendment = plan.get("protocol_amendment")
    if amendment is None:
        if (root / "AMENDMENT.json").exists() or (root / EFFECTIVE_CONFIG).exists():
            raise PermissionError("Original plan conflicts with the registered amendment")
        return None
    receipt = read_json(root / "AMENDMENT.json")
    expected = {
        "plan_id": PLAN_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "REGISTERED_BEFORE_GENERATION",
        "run_root": str(root),
        "original_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "original_freeze_preserved": True,
        "original_config_sha256": file_hash(PACKAGE / "config/SR_F1.json"),
        "approved_solution_sha256": USER_SOLUTION_SHA256,
        "effective_config": EFFECTIVE_CONFIG,
        "effective_config_sha256": file_hash(root / EFFECTIVE_CONFIG),
        "effective_config_object_sha256": object_hash(plan),
        "protocol_amendment": amendment,
        "changes_from_uploaded_contract": approved_changes(plan),
        "inherited_or_prior_run_results_used_to_modify_protocol": True,
        "inherited_or_prior_run_results_used_to_select_parameters": True,
        "formal_test_results_seen_before_freeze": False,
        "prior_bridged_adapter_used_for_scientific_start": False,
        "user_selected_learning_rate_option": "A",
    }
    for key, value in expected.items():
        if object_hash(receipt.get(key)) != object_hash(value):
            raise PermissionError("Registered amendment differs: " + key)
    if not receipt.get("registered_at") or read_json(root / EFFECTIVE_CONFIG) != plan:
        raise PermissionError("Registered amendment config or timestamp is missing")
    hashes = receipt.get("document_hashes", {})
    if set(hashes) != set(REGISTERED_DOCUMENTS):
        raise PermissionError("Amendment document inventory is incomplete")
    for relative, source_name in REGISTERED_DOCUMENTS.items():
        if hashes[relative] != file_hash(root / relative):
            raise PermissionError("Frozen amendment document changed: " + relative)
        if hashes[relative] != file_hash(AMENDMENT_SOURCE_DIR / source_name):
            raise PermissionError("Amendment source document differs: " + relative)
    if hashes["amendment/USER_SOLUTION_zh.md"] != USER_SOLUTION_SHA256:
        raise PermissionError("The registered user solution document changed")
    return receipt


def effective_plan(root):
    """Load a root's explicit registration, with no process-global override."""
    root = Path(root).resolve(strict=True)
    amended = root / EFFECTIVE_CONFIG
    if amended.exists() or (root / "AMENDMENT.json").exists():
        plan = validate_plan(read_json(amended))
        verify_amendment(root, plan)
        return plan
    return load_config()


def verify_protocol_route(route, plan):
    amendment = protocol_amendment(plan)
    if amendment is None and any(
        key in route
        for key in (
            "protocol_amendment",
            "protocol_amendment_id",
            "assistant_prefill",
            "assistant_prefill_token_ids",
        )
    ):
        raise PermissionError("Original processor route contains an unregistered prefill protocol")
    if amendment is not None and (
        route.get("protocol_amendment") != amendment
        or route.get("protocol_amendment_id") != AMENDMENT_ID
        or route.get("assistant_prefill") != "{"
        or route.get("assistant_prefill_token_ids") != [90]
    ):
        raise PermissionError("Amended processor route lacks the frozen prefill protocol")
