"""Fail-closed source, input and actual-processor boundaries for the new screen."""

import copy
import json

import pytest

from sr_f6 import interface_screen_registration as registration
from sr_f6.common import digest, file_hash, write_once
from sr_f6.interface_repair import screen_plan


@pytest.fixture
def project(tmp_path, monkeypatch):
    path = tmp_path.resolve()
    monkeypatch.setattr(registration, "PROJECT", path)
    monkeypatch.setattr(registration, "ROOT", path / "screen")
    monkeypatch.setattr(registration, "SOURCE_ROOT", path / "old")
    return path


def test_paths_reject_original_root_external_and_symlink(project):
    (project / "old").mkdir()
    with pytest.raises(PermissionError, match="independent screen"):
        registration._root(project / "old")
    with pytest.raises(PermissionError, match="outside"):
        registration._path(project.parent)
    (project / "screen").symlink_to(project / "old", target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlinked"):
        registration._root(project / "screen")


def test_actual_deployment_requires_full_matching_revision(project, monkeypatch):
    path = project / "sr_f6_code_abcdef1"
    path.mkdir()
    monkeypatch.setattr(registration, "_actual_deployment", lambda: path)
    assert registration._deployment("abcdef1" + "0" * 33) == path
    for revision in ("abcdef1", "1" * 40, "g" * 40, True):
        with pytest.raises(PermissionError):
            registration._deployment(revision)


def test_source_inventory_covers_recursive_entrypoints_and_rejects_symlinks(project):
    deployment = project / "code"
    for name in ("src/core.py", "tests/sr_f6/test_example.py", "scripts/sr_f6/nested/run.sh"):
        path = deployment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# bound source\n")
    assert "scripts/sr_f6/nested/run.sh" in registration._source_paths(deployment)
    (deployment / "src/escape.py").symlink_to(deployment / "src/core.py")
    with pytest.raises(PermissionError, match="Symlinked"):
        registration._source_paths(deployment)


@pytest.fixture
def public():
    # Plan validation tests do not claim these minimal records are task inputs.
    return [
        {
            "qid": f"q{i}",
            "root_id": f"r{i // 8}",
            "factor": "P0" if i % 8 < 4 else "P1",
            "variant": i % 4,
            "model_input": {
                "prompt": f"visible {i}",
                "image_file": "image.png",
                "image_sha256": "a" * 64,
            },
        }
        for i in range(96)
    ]


@pytest.mark.parametrize(
    "mutation", ["seed", "count", "threshold", "candidate", "generation", "training"]
)
def test_fixed_plan_rejects_any_recipe_or_scope_change(public, mutation):
    plan = screen_plan(public)
    registration._validate_plan(plan, public)
    if mutation == "seed":
        plan["attempts"][0]["seed"] += 1
    elif mutation == "count":
        plan["attempts"].pop()
    elif mutation == "threshold":
        plan["required_counts"]["V"] = 86
    elif mutation == "candidate":
        plan["max_prompt_candidates"] = 2
    else:
        plan[mutation + "_allowed"] = True
    with pytest.raises(PermissionError):
        registration._validate_plan(plan, public)


@pytest.fixture
def route():
    return {
        "native_input_tensor_hashes": dict.fromkeys(
            ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"),
            "1" * 64,
        ),
        "source_image_sha256": "a" * 64,
        "input_tensor_hash": "1" * 64,
        "chat_text_sha256": "2" * 64,
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_image_sha256", "d" * 64),
        ("processor_hash", "d" * 64),
        ("chat_template_hash", "d" * 64),
        ("input_float_dtype", "bfloat16"),
        ("probability_dtype", "bfloat16"),
        ("stop_rule", "constrained_json"),
        ("assistant_prefill", '{"selected":'),
        ("source_size", [1024, 768]),
        ("processed_size", [1056, 738]),
        ("image_grid_thw", [[True, 46, 66]]),
        ("image_token_count", 760),
        ("native_input_tensor_hashes", {"input_ids": "1" * 64}),
        ("input_tensor_hash", None),
        ("chat_text_sha256", ""),
    ],
)
def test_actual_native_route_rejects_mismatch(route, field, value):
    model_input = {"image_sha256": "a" * 64}
    identity = {"processor_hash": "b" * 64, "chat_template_hash": "c" * 64}
    registration._validate_route(route, model_input, identity)
    route[field] = value
    with pytest.raises(PermissionError):
        registration._validate_route(route, model_input, identity)


@pytest.fixture
def preflight(public, route):
    return {
        "schema": registration.PREFLIGHT_SCHEMA,
        "phase": registration.PHASE,
        "status": "READY_FOR_576_SCREEN",
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
                "model_input_sha256": digest(item["model_input"]),
                "routing": copy.deepcopy(route),
            }
            for item in public
        ],
        "visual_pixel_rechecks": [{"qid": f"v{i}", "identical_pixels": True} for i in range(48)],
        "verified_dependencies": [{"path": "test-only-placeholder"}],
        "tokenizer_fixtures": [
            {"text": text, "token_ids": [1, 2], "exact_roundtrip": True}
            for text in registration.TOKENIZER_FIXTURES
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "prompt",
        "pin",
        "visual",
        "failure",
        "checks",
        "training",
        "dependencies",
    ],
)
def test_preflight_cannot_upgrade_partial_or_different_evidence(preflight, public, mutation):
    registration._validate_preflight(preflight, "f" * 64, public)
    if mutation == "missing":
        preflight["inputs"].pop()
    elif mutation == "duplicate":
        preflight["inputs"][1] = preflight["inputs"][0]
    elif mutation == "prompt":
        preflight["inputs"][0]["model_input_sha256"] = "e" * 64
    elif mutation == "pin":
        preflight["registration_sha256"] = "e" * 64
    elif mutation == "visual":
        preflight["visual_pixel_rechecks"][0]["identical_pixels"] = False
    elif mutation == "failure":
        preflight["status"] = "FAIL"
    elif mutation == "checks":
        preflight["checks"]["new_prompt_processor"] = False
    elif mutation == "training":
        preflight["training_allowed"] = True
    else:
        preflight["verified_dependencies"] = []
    with pytest.raises(PermissionError):
        registration._validate_preflight(preflight, "f" * 64, public)


@pytest.mark.parametrize(
    "mutation", ["missing", "text", "bool_id", "empty_ids", "negative_id", "roundtrip"]
)
def test_preflight_requires_exact_actual_tokenizer_fixtures(preflight, public, mutation):
    fixture = preflight["tokenizer_fixtures"][0]
    if mutation == "missing":
        preflight["tokenizer_fixtures"].pop()
    elif mutation == "text":
        fixture["text"] = "different text"
    elif mutation == "bool_id":
        fixture["token_ids"][0] = True
    elif mutation == "empty_ids":
        fixture["token_ids"] = []
    elif mutation == "negative_id":
        fixture["token_ids"][0] = -1
    else:
        fixture["exact_roundtrip"] = False
    with pytest.raises(PermissionError, match="tokenizer fixtures"):
        registration._validate_preflight(preflight, "f" * 64, public)


def test_content_reference_rejects_changed_bytes(project):
    write_once(project / "input.json", {"frozen": True})
    ref = registration._reference(project, "input.json")
    assert registration._read_reference(project, ref) == {"frozen": True}
    (project / "input.json").write_text('{"frozen":false}')
    with pytest.raises(PermissionError, match="content changed"):
        registration._read_reference(project, ref)


def test_failed_preflight_preserved_and_cannot_silently_retry(project, monkeypatch):
    root = project / "screen"
    root.mkdir()
    write_once(root / "REGISTRATION.json", {"test_fixture": True})

    def failure(*args, **kwargs):
        raise PermissionError("actual old source changed")

    monkeypatch.setattr(registration, "verify_registered", failure)
    with pytest.raises(PermissionError, match="actual old source"):
        registration.preflight(root)
    result = json.loads((root / "PREFLIGHT.json").read_text())
    assert result["status"] == "FAIL"
    assert result["checks"] == {}
    assert result["model_calls"] == result["optimizer_updates"] == 0
    assert result["training_allowed"] is False
    with pytest.raises(PermissionError, match="Preserve"):
        registration.preflight(root)


def test_old_data_cannot_be_refrozen_after_its_content_changes(project, monkeypatch):
    source = project / "old"
    source.mkdir()
    write_once(source / "CONFIG.json", {})
    inputs = source / "DATA/DEV_CAL/MODEL_INPUTS.jsonl"
    inputs.parent.mkdir(parents=True)
    inputs.write_text('{"qid":"old"}\n')
    tasks = inputs.with_name("TASKS_PRIVATE.jsonl")
    tasks.write_text('{"qid":"old"}\n')
    manifest = {
        "files": [
            {"path": str(path.relative_to(source)), "sha256": file_hash(path)}
            for path in (inputs, tasks)
        ]
    }
    write_once(source / "DATA/MANIFEST.json", manifest)
    write_once(
        source / "IDENTITY.json",
        {
            "config_sha256": file_hash(source / "CONFIG.json"),
            "data_manifest": registration._reference(source, "DATA/MANIFEST.json"),
        },
    )
    inputs.write_text('{"qid":"replacement"}\n')
    with pytest.raises(PermissionError, match="content changed"):
        registration._validate_producer(source, project / "deployment")


def test_register_refuses_existing_namespace_before_reading_sources(project):
    (project / "old").mkdir()
    (project / "screen").mkdir()
    with pytest.raises(PermissionError, match="Preserve"):
        registration.register(project / "screen", source_root=project / "old", revision="a" * 40)
