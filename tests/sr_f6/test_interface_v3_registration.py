"""Boundaries of the prospective v3 namespace, both plans and CPU evidence."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sr_f6 import interface_v3_registration as registration
from sr_f6.common import digest, file_hash, write_once
from sr_f6.interface_v3_plan import _questions, screen_plan
from sr_f6.interface_v3_protocol import VERSION, WIRE_PROTOCOL


@pytest.fixture
def project(tmp_path, monkeypatch):
    path = tmp_path.resolve()
    monkeypatch.setattr(registration, "PROJECT", path)
    monkeypatch.setattr(registration.v2, "PROJECT", path)
    monkeypatch.setattr(registration, "ROOT", path / "screen-v3")
    monkeypatch.setattr(registration, "SOURCE_ROOT", path / "original")
    monkeypatch.setattr(registration, "PRIOR_ROOT", path / "screen-v2")
    return path


def test_only_new_canonical_root_is_accepted(project):
    old = project / "original"
    old.mkdir()
    with pytest.raises(PermissionError, match="independent v3"):
        registration._root(old)
    with pytest.raises(PermissionError, match="outside"):
        registration._path(project.parent)
    (project / "screen-v3").symlink_to(old, target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlinked"):
        registration._root(project / "screen-v3")


@pytest.mark.parametrize("stage", [True, False, 0, 3, "1", None, 1.0])
def test_unregistered_stages_rejected_before_source_access(stage):
    with pytest.raises(PermissionError, match="stages one and two"):
        registration.verify_registered("/nonexistent", stage=stage)


@pytest.mark.parametrize("parent", [None, "b3", True, [], {}])
def test_stage2_requires_exact_parent_before_source_access(parent):
    with pytest.raises(PermissionError):
        registration.verify_registered("/nonexistent", stage=2, parent=parent)


def test_stage1_context_and_cpu_preflight_do_not_grant_stage2():
    assert registration._stage(1, None, require_preflight=True) == 1
    assert registration._stage(2, None, require_preflight=False) == 2
    assert registration._stage(2, "b1", require_preflight=True) == 2


def test_deployment_must_match_full_revision(project, monkeypatch):
    deployment = project / "sr_f6_code_abcdef1"
    deployment.mkdir()
    monkeypatch.setattr(registration, "_actual_deployment", lambda: deployment)
    assert registration._deployment("abcdef1" + "0" * 33) == deployment
    for wrong in ("abcdef1", "0" * 40, "g" * 40, False):
        with pytest.raises(PermissionError):
            registration._deployment(wrong)


def test_no_existing_namespace_can_be_rebound(project):
    (project / "original").mkdir()
    (project / "screen-v3").mkdir()
    with pytest.raises(PermissionError, match="Preserve"):
        registration.register(
            project / "screen-v3", source_root=project / "original", revision="a" * 40
        )


def test_missing_inherited_asset_blocks_inventory(project):
    deployment = project / "sr_f6_code_abcdef1"
    deployment.mkdir()
    for name in registration.DEPLOYMENT_ASSETS[:3]:
        (deployment / name).mkdir(parents=True)
    for name in registration.DEPLOYMENT_ASSETS[3:]:
        path = deployment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (deployment / "docs/sr_f1/package/reference/semantic_contract.py").unlink()
    with pytest.raises(FileNotFoundError):
        registration._source_paths(deployment)


def test_isolated_imports_use_actual_v3_modules_and_inherited_resources(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    isolated = tmp_path / "isolated"
    shutil.copytree(
        repository / "src", isolated / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    for relative in registration.DEPLOYMENT_ASSETS[3:]:
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    result = registration.check_deployment_imports(isolated)
    assert result["status"] == "PASS"
    assert result["model_calls"] == result["optimizer_updates"] == 0
    assert "V3_PRODUCTION_IMPORTS_RESOLVED" in result["stdout"]
    (isolated / "docs/sr_f1/package/PACKAGE_SHA256.json").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        registration.check_deployment_imports(isolated)


@pytest.fixture
def prior(project):
    root = project / "screen-v2"
    root.mkdir()
    old = {
        "schema": registration.v2.REGISTRATION_SCHEMA,
        "phase": "INTERFACE_SCREEN_V2",
        "prompt_version": "srf6-interface-v2",
        "root": str(root),
        "revision": registration.PRIOR_REVISION,
        "training_allowed": False,
    }
    write_once(root / "REGISTRATION.json", old)
    assessment = {
        "status": "SCREEN_FAIL_STOP",
        "phase": "INTERFACE_SCREEN_V2",
        "responses": 576,
        "optimizer_updates": 0,
        "training_allowed": False,
        "registration_sha256": file_hash(root / "REGISTRATION.json"),
        "accounting": {"actual_completion_attempts": 576},
        "saved_token_validation": {
            "status": "PASS",
            "checked_responses": 576,
            "new_model_calls": 0,
        },
        "records_sha256": "a" * 64,
    }
    write_once(root / "SCREEN/SCREEN_ASSESSMENT.json", assessment)
    return root, old, assessment


def test_prior_failed_screen_is_design_evidence_only(prior):
    result = registration._prior_design_basis()
    assert result["prior_status"] == "SCREEN_FAIL_STOP_UNCHANGED"
    assert result["old_authorization_reused"] is False
    assert result["old_results_rescored"] is False
    assert result["records_sha256"] == "a" * 64
    assert "DELETED" in result["older_9216_raw_status"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "SCREEN_PASS"),
        ("responses", 575),
        ("phase", "CENSUS_S0"),
        ("registration_sha256", "b" * 64),
        ("records_sha256", ""),
        ("training_allowed", True),
        ("optimizer_updates", 1),
        ("accounting", {"actual_completion_attempts": 577}),
        (
            "saved_token_validation",
            {"status": "PASS", "checked_responses": 575, "new_model_calls": 0},
        ),
    ],
)
def test_fabricated_or_partial_prior_assessment_rejected(prior, field, value):
    root, _, assessment = prior
    assessment[field] = value
    (root / "SCREEN/SCREEN_ASSESSMENT.json").write_text(json.dumps(assessment))
    with pytest.raises(PermissionError, match="complete preserved failed"):
        registration._prior_design_basis()


@pytest.fixture
def stage_inputs():
    stages = {}
    for number in (1, 2):
        rows = []
        for qid, identity in sorted(_questions(number).items()):
            prompt = "synthetic public test prompt " + qid
            rows.append(
                {
                    "qid": qid,
                    **identity,
                    "screen_stage": number,
                    "protocol": WIRE_PROTOCOL,
                    "prompt_version": VERSION,
                    "source_prompt_sha256": "b" * 64,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "model_input": {
                        "prompt": prompt,
                        "image_file": "DATA/images/test.png",
                        "image_sha256": "a" * 64,
                    },
                }
            )
        stages[number] = rows
    return stages


@pytest.mark.parametrize("stage", [1, 2])
def test_exact_full_stage_plan_required(stage_inputs, stage):
    public = stage_inputs[stage]
    plan = screen_plan(public, stage=stage)
    registration._validate_plan(plan, public, stage)
    plan["attempts"][-1]["seed"] += 1
    with pytest.raises(PermissionError, match="plan differs"):
        registration._validate_plan(plan, public, stage)


def test_plan_stage_switch_cannot_reuse_first_stage_inputs(stage_inputs):
    with pytest.raises(ValueError):
        registration._validate_plan(screen_plan(stage_inputs[1]), stage_inputs[1], 2)


@pytest.fixture
def actual_preflight(stage_inputs):
    public = stage_inputs[1] + stage_inputs[2]
    route = {
        "native_input_tensor_hashes": dict.fromkeys(
            ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"),
            "1" * 64,
        ),
        "input_tensor_hash": "2" * 64,
        "chat_text_sha256": "3" * 64,
        "source_image_sha256": "a" * 64,
        "processor_hash": "b" * 64,
        "chat_template_hash": "c" * 64,
        "source_size": [1280, 896],
        "input_float_dtype": "float32",
        "probability_dtype": "float32",
        "stop_rule": "first_balanced_json",
        "assistant_prefill": "{",
        "sr_f6_protocol_version": "structured_v4-srf6-1.0",
        "image_grid_thw": [[1, 46, 66]],
        "processed_size": [1056, 736],
        "image_token_count": 759,
    }
    receipt = {
        "schema": registration.PREFLIGHT_SCHEMA,
        "phase": registration.PHASE,
        "status": "READY_FOR_V3_SCREEN",
        "execution_origin": "SERVER_ACTUAL",
        "registration_sha256": "f" * 64,
        "model_calls": 0,
        "optimizer_updates": 0,
        "training_allowed": False,
        "confirmation_allowed": False,
        "new_visual_review_performed": False,
        "checks": dict.fromkeys(
            ("source", "historical_processor_visual", "new_prompt_processor"), True
        ),
        "identity": {"processor_hash": "b" * 64, "chat_template_hash": "c" * 64},
        "inputs": [
            {
                "qid": item["qid"],
                "screen_stage": item["screen_stage"],
                "model_input_sha256": digest(item["model_input"]),
                "routing": copy.deepcopy(route),
            }
            for item in public
        ],
        "visual_pixel_rechecks": [{"qid": f"v{i}", "identical_pixels": True} for i in range(48)],
        "verified_dependencies": [{"path": "unit-test-only-no-authorization"}],
        "tokenizer_fixtures": [
            {"text": text, "token_ids": [1, 2], "exact_roundtrip": True}
            for text in registration.TOKENIZER_FIXTURES
        ],
    }
    return receipt, public


def test_preflight_checks_both_stages_without_new_visual_claim(actual_preflight):
    receipt, public = actual_preflight
    registration._validate_preflight(receipt, "f" * 64, public)
    assert len(receipt["inputs"]) == 192
    assert receipt["new_visual_review_performed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "only_stage1",
        "duplicate",
        "prompt",
        "pin",
        "visual",
        "failure",
        "checks",
        "training",
        "dependencies",
        "fixture",
        "new_visual",
        "bool_model_calls",
        "route",
        "screen_stage",
    ],
)
def test_incomplete_or_different_actual_preflight_rejected(actual_preflight, mutation):
    receipt, public = actual_preflight
    if mutation == "only_stage1":
        receipt["inputs"] = receipt["inputs"][:96]
    elif mutation == "duplicate":
        receipt["inputs"][191] = receipt["inputs"][0]
    elif mutation == "prompt":
        receipt["inputs"][150]["model_input_sha256"] = "e" * 64
    elif mutation == "pin":
        receipt["registration_sha256"] = "e" * 64
    elif mutation == "visual":
        receipt["visual_pixel_rechecks"][0]["identical_pixels"] = False
    elif mutation == "failure":
        receipt["status"] = "FAIL"
    elif mutation == "checks":
        receipt["checks"]["source"] = False
    elif mutation == "training":
        receipt["training_allowed"] = True
    elif mutation == "dependencies":
        receipt["verified_dependencies"] = []
    elif mutation == "fixture":
        receipt["tokenizer_fixtures"][0]["token_ids"][0] = True
    elif mutation == "new_visual":
        receipt["new_visual_review_performed"] = True
    elif mutation == "bool_model_calls":
        receipt["model_calls"] = False
    elif mutation == "screen_stage":
        receipt["inputs"][100]["screen_stage"] = 1
    else:
        receipt["inputs"][100]["routing"]["stop_rule"] = "unregistered_stop"
    with pytest.raises(PermissionError):
        registration._validate_preflight(receipt, "f" * 64, public)


def test_failed_actual_preflight_preserved_and_not_retried(project, monkeypatch):
    root = project / "screen-v3"
    root.mkdir()
    write_once(root / "REGISTRATION.json", {"unit_test": True})

    def fail(*args, **kwargs):
        raise PermissionError("source changed")

    monkeypatch.setattr(registration, "verify_registered", fail)
    with pytest.raises(PermissionError, match="source changed"):
        registration.preflight(root)
    receipt = json.loads((root / "PREFLIGHT.json").read_text())
    assert receipt["status"] == "FAIL" and receipt["model_calls"] == 0
    assert receipt["training_allowed"] is False
    assert not (root / "PREFLIGHT_SEAL.json").exists()
    with pytest.raises(PermissionError, match="Preserve"):
        registration.preflight(root)


def test_bound_content_change_is_not_a_new_registration(project):
    write_once(project / "input.json", {"value": 1})
    ref = registration._reference(project, "input.json")
    assert registration._read_reference(project, ref) == {"value": 1}
    (project / "input.json").write_text('{"value":2}')
    with pytest.raises(PermissionError, match="content changed"):
        registration._read_reference(project, ref)


@pytest.fixture
def registered_namespace(project, prior, stage_inputs, monkeypatch):
    """Exercise real immutable files with explicitly stubbed external inventories.

    This unit fixture is not real processor, source, or model acceptance evidence.
    Their independent validation is tested separately and run on the server.
    """
    from sr_f6 import contract, runtime, source_binding

    original = project / "original"
    original.mkdir()
    image = original / "DATA/images/test.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic fixture image, never sent to a processor")
    image_sha = file_hash(image)
    stages = {}
    for stage in (1, 2):
        public = copy.deepcopy(stage_inputs[stage])
        for row in public:
            row["model_input"]["image_sha256"] = image_sha
        stages[str(stage)] = {
            "public": public,
            "private": {row["qid"]: {"qid": row["qid"]} for row in public},
            "plan": screen_plan(public, stage=stage),
        }
    binding = {"parents": {p: {"content_sha256": p * 32} for p in registration.PARENTS}}
    for relative, value in (
        ("CONFIG.json", {"unit_test_external_dependencies_stubbed": True}),
        ("IDENTITY.json", {"unit_test_identity": "origin"}),
        ("PHASES/PILOT_V1/IDENTITY.json", {"unit_test_identity": "pilot"}),
        ("SOURCE_BINDING.json", binding),
    ):
        write_once(original / relative, value)
    deployment = project / "sr_f6_code_abcdef1"
    deployment.mkdir()
    source = deployment / "source.py"
    source.write_text("# unit test only\n")
    monkeypatch.setattr(registration, "_actual_deployment", lambda: deployment)
    monkeypatch.setattr(registration, "_source_paths", lambda _: ["source.py"])
    monkeypatch.setattr(registration, "SOURCE_REFERENCES", ("CONFIG.json", "IDENTITY.json"))
    monkeypatch.setattr(
        registration, "ORIGIN_IDENTITY_SHA256", file_hash(original / "IDENTITY.json")
    )
    monkeypatch.setattr(
        registration,
        "PILOT_IDENTITY_SHA256",
        file_hash(original / "PHASES/PILOT_V1/IDENTITY.json"),
    )
    monkeypatch.setattr(registration, "validate_deployment_assets", lambda _: {})
    monkeypatch.setattr(registration, "check_deployment_imports", lambda _: {"status": "PASS"})
    monkeypatch.setattr(registration, "_source_data", lambda _: (copy.deepcopy(stages), [], {}))
    monkeypatch.setattr(registration, "_validate_producer", lambda *args: None)
    monkeypatch.setattr(contract, "validate_config", lambda _: None)
    monkeypatch.setattr(runtime, "_verify_module_origins", lambda *args: None)
    monkeypatch.setattr(runtime, "verify_binding_inputs", lambda *args: (binding, {}))
    monkeypatch.setattr(
        registration,
        "inventory_file",
        lambda path: source_binding.inventory_file(path, boundary=project),
    )
    monkeypatch.setattr(
        registration,
        "verify_bound_files",
        lambda files: source_binding.verify_bound_files(files, boundary=project),
    )
    root = project / "screen-v3"
    record = registration.register(root, source_root=original, revision="abcdef1" + "0" * 33)
    return root, record, image_sha


def test_registration_freezes_both_plans_before_any_preflight(registered_namespace):
    root, record, _ = registered_namespace
    assert set(record["stages"]) == {"1", "2"}
    assert record["generation_allowed"] is False
    assert not (root / "PREFLIGHT.json").exists()
    for stage in (1, 2):
        context = registration.verify_registered(root, stage=stage, require_preflight=False)
        assert context["root"] == root / f"STAGE{stage}"
        assert context["registration_root"] == root
        assert len(context["public"]) == 96 and len(context["all_public"]) == 192
        assert len(context["plan"]["attempts"]) == 576
    with pytest.raises(PermissionError, match="Evidence is missing"):
        registration.verify_registered(root, stage=1)


def test_stage1_also_rejects_mutated_unused_stage2_plan(registered_namespace):
    root, record, _ = registered_namespace
    path = root / record["stages"]["2"]["plan"]["path"]
    plan = json.loads(path.read_text())
    plan["attempts"][0]["seed"] += 1
    path.write_text(json.dumps(plan))
    # Even updating the local reference cannot change the preregistered formula.
    record["stages"]["2"]["plan"]["sha256"] = file_hash(path)
    record["data_sha256"] = digest(record["stages"])
    (root / "REGISTRATION.json").write_text(json.dumps(record))
    with pytest.raises(PermissionError, match="plan differs"):
        registration.verify_registered(root, stage=1, require_preflight=False)


@pytest.mark.parametrize("authorization", [True, False, "raise"])
def test_stage2_context_requires_complete_stage1_helper(
    registered_namespace, actual_preflight, monkeypatch, authorization
):
    from sr_f6 import interface_v3_screen

    root, record, image_sha = registered_namespace
    context = registration.verify_registered(root, require_preflight=False)
    receipt, _ = actual_preflight
    pin = file_hash(root / "REGISTRATION.json")
    receipt["registration_sha256"] = pin
    by_qid = {row["qid"]: row for row in context["all_public"]}
    for row in receipt["inputs"]:
        row["model_input_sha256"] = digest(by_qid[row["qid"]]["model_input"])
        row["routing"]["source_image_sha256"] = image_sha
    receipt["verified_dependencies"] = record["image_files"]
    write_once(root / "PREFLIGHT.json", receipt)
    write_once(
        root / "PREFLIGHT_SEAL.json",
        {
            "schema": registration.SEAL_SCHEMA,
            "registration_sha256": pin,
            "preflight": registration._reference(root, "PREFLIGHT.json"),
            "training_allowed": False,
        },
    )
    calls = []

    def authorize(base, parent):
        calls.append((base, parent))
        if authorization == "raise":
            raise PermissionError("stage-one failure is preserved")
        return authorization

    monkeypatch.setattr(interface_v3_screen, "stage2_authorized", authorize)
    assert registration.verify_registered(root)["stage"] == 1
    assert calls == []
    if authorization is True:
        context = registration.verify_registered(root, stage=2, parent="b1")
        assert context["root"] == root / "STAGE2"
        assert context["preflight"] == receipt
    else:
        with pytest.raises(PermissionError):
            registration.verify_registered(root, stage=2, parent="b1")
    assert calls == [(root, "b1")]
