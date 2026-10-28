"""Prospectively bind both bounded interface-v3 stages and actual CPU preflight.

The previous screen is design evidence, never an authorization. All inputs and
potential seeds for both stages are frozen before generation; stage two also
requires the validated, complete stage-one decision. Original scoring and
sampling remain unchanged. No model weights are loaded by this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

from . import interface_screen_registration as v2
from .common import digest, file_hash, now, read, safe_evidence, write_once
from .compatibility import CONSUMER_VERSION
from .interface_v3_plan import PHASE, screen_inputs, screen_plan
from .interface_v3_protocol import VERSION, public_prompt
from .source_binding import PROJECT, inventory_file, verify_bound_files

ROOT = PROJECT / "sr_f6_interface_v3_20261028"
SOURCE_ROOT = v2.SOURCE_ROOT
PRIOR_ROOT = PROJECT / "sr_f6_interface_v2_20261028_r1"
PRIOR_REVISION = "0c46a38d630206fb4bf230c5d17a9504c6f77c77"
REGISTRATION_SCHEMA = "srf6-interface-v3-registration-v1"
PREFLIGHT_SCHEMA = "srf6-interface-v3-preflight-v1"
SEAL_SCHEMA = "srf6-interface-v3-preflight-seal-v1"
DEPLOYMENT_ASSETS = v2.DEPLOYMENT_ASSETS
TOKENIZER_FIXTURES = v2.TOKENIZER_FIXTURES
SOURCE_REFERENCES = v2.SOURCE_REFERENCES
ORIGIN_IDENTITY_SHA256 = v2.ORIGIN_IDENTITY_SHA256
PILOT_IDENTITY_SHA256 = v2.PILOT_IDENTITY_SHA256
PARENTS = ("b0", "b1", "b2")


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _path(path, *, exists=True):
    path = Path(path)
    _require(
        path.is_absolute() and path.is_relative_to(PROJECT) and ".." not in path.parts,
        "Screen path is outside the authorized project",
    )
    _require(PROJECT.resolve(strict=True) == PROJECT, "Project must be canonical")
    for candidate in (path, *path.parents):
        _require(not candidate.is_symlink(), "Symlinked screen paths are forbidden")
        if candidate == PROJECT:
            break
    _require(path.resolve(strict=exists) == path, "Screen path must be canonical")
    return path


def _root(root, *, exists=True):
    root = _path(root, exists=exists)
    _require(root == ROOT, "Only the registered independent v3 root is allowed")
    return root


def _stage(stage, parent=None, *, require_preflight=False):
    _require(type(stage) is int and stage in (1, 2), "Only frozen stages one and two exist")
    _require(parent is None or parent in PARENTS, "Unknown registered parent")
    if stage == 2 and require_preflight:
        _require(parent in PARENTS, "Stage two requires an explicitly authorized parent")
    return stage


def _actual_deployment():
    return Path(__file__).resolve(strict=True).parents[2]


def _deployment(revision):
    _require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "Full immutable Git revision required",
    )
    deployment = _path(_actual_deployment())
    match = re.fullmatch(r"sr_f6_code_([0-9a-f]{7,40})", deployment.name)
    _require(
        deployment.parent == PROJECT and match and revision.startswith(match.group(1)),
        "Actual source deployment must match its declared revision",
    )
    return deployment


def _source_paths(deployment):
    return v2._source_paths(deployment)


def validate_deployment_assets(deployment):
    return v2.validate_deployment_assets(deployment)


def check_deployment_imports(deployment):
    """Check the actual isolated production chain, including the new entrypoints."""
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "root = Path(sys.argv[1]).resolve(strict=True)",
            "sys.path.insert(0, str(root / 'src'))",
            "import sr_f6.admission",
            "import sr_f6.interface_v3_registration",
            "import sr_f6.interface_v3_screen",
            "import sr_f6.interface_v3_runtime",
            "import sr_f6.interface_v3_dispatch",
            "import sr_f6.interface_v3_audit",
            "from sr_f1.contract import PACKAGE, load_config, _reference",
            "assert PACKAGE == root / 'docs/sr_f1/package'",
            "assert Path(_reference.__file__).is_relative_to(PACKAGE)",
            "assert load_config()",
            "packages = {p.name for p in (root / 'src').iterdir() if p.is_dir()}",
            "for name, module in tuple(sys.modules.items()):",
            "    if name.split('.')[0] in packages:",
            "        path = getattr(module, '__file__', None)",
            "        if path: assert Path(path).resolve().is_relative_to(root / 'src'), name",
            "print('V3_PRODUCTION_IMPORTS_RESOLVED; model_calls=0; optimizer_updates=0')",
        )
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(deployment)],
        cwd=deployment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return dict(
        status="PASS",
        method="isolated actual production imports; no model construction",
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
        model_calls=0,
        optimizer_updates=0,
    )


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _reference(root, relative):
    path = safe_evidence(root, relative)
    return {"path": relative, "sha256": file_hash(path)}


def _read_reference(root, reference, *, lines=False):
    return v2._read_reference(root, reference, lines=lines)


def _source_data(source_root):
    old_public = _jsonl(safe_evidence(source_root, "DATA/DEV_CAL/MODEL_INPUTS.jsonl"))
    old_private = _jsonl(safe_evidence(source_root, "DATA/DEV_CAL/TASKS_PRIVATE.jsonl"))
    tasks = {row["qid"]: row for row in old_private}
    stages = {}
    for stage in (1, 2):
        public = screen_inputs(old_public, old_private, stage=stage)
        stages[str(stage)] = {
            "public": public,
            "private": {row["qid"]: tasks[row["qid"]] for row in public},
            "plan": screen_plan(public, stage=stage),
        }
    all_public = [row for stage in stages.values() for row in stage["public"]]
    _require(
        len(all_public) == len({row["qid"] for row in all_public}) == 192,
        "Two disjoint 96-input stages must be frozen",
    )
    return stages, old_public, tasks


def _validate_plan(plan, public, stage):
    _stage(stage)
    _require(plan == screen_plan(public, stage=stage), "Frozen prospective stage plan differs")
    _require(
        plan.get("generation_allowed") is False and plan.get("training_allowed") is False,
        "An input plan cannot grant execution",
    )


def _validate_producer(source_root, deployment, stages):
    # Original scorer, inherited implementation and native sampler checks are
    # shared verbatim. This does not reuse v2 admission or its failed gate.
    v2._validate_producer(source_root, deployment)
    origin = read(safe_evidence(source_root, "IDENTITY.json"))
    manifest = _read_reference(source_root, origin["data_manifest"])
    members = {item["path"]: item for item in manifest["files"]}
    for stage in stages.values():
        for item in stage["public"]:
            model = item["model_input"]
            task = stage["private"][item["qid"]]
            _require(
                model["image_file"] == f"DATA/images/{digest(task['world'])}.png"
                and members[model["image_file"]]["sha256"] == model["image_sha256"],
                "Frozen v3 task/world/image lineage differs from its producer",
            )


def _prior_design_basis():
    """Bind the preserved failed v2 evidence without rescoring or authorizing it."""
    prior = _path(PRIOR_ROOT)
    registration_path = safe_evidence(prior, "REGISTRATION.json")
    assessment_path = safe_evidence(prior, "SCREEN/SCREEN_ASSESSMENT.json")
    prior_registration = read(registration_path)
    assessment = read(assessment_path)
    _require(
        prior_registration.get("schema") == v2.REGISTRATION_SCHEMA
        and prior_registration.get("phase") == "INTERFACE_SCREEN_V2"
        and prior_registration.get("prompt_version") == "srf6-interface-v2"
        and prior_registration.get("root") == str(prior)
        and prior_registration.get("revision") == PRIOR_REVISION
        and prior_registration.get("training_allowed") is False,
        "Prior v2 registration identity differs",
    )
    _require(
        assessment.get("status") == "SCREEN_FAIL_STOP"
        and assessment.get("phase") == "INTERFACE_SCREEN_V2"
        and assessment.get("responses") == 576
        and type(assessment.get("optimizer_updates")) is int
        and assessment["optimizer_updates"] == 0
        and assessment.get("training_allowed") is False
        and assessment.get("registration_sha256") == file_hash(registration_path)
        and assessment.get("accounting", {}).get("actual_completion_attempts") == 576
        and assessment.get("saved_token_validation", {}).get("status") == "PASS"
        and assessment.get("saved_token_validation", {}).get("checked_responses") == 576
        and type(assessment.get("saved_token_validation", {}).get("new_model_calls")) is int
        and assessment["saved_token_validation"]["new_model_calls"] == 0
        and isinstance(assessment.get("records_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", assessment["records_sha256"]),
        "A complete preserved failed v2 screen is required as design evidence",
    )
    return {
        "kind": "NEW_PROSPECTIVE_VERSION_AFTER_SCIENTIFIC_STOP",
        "prior_root": str(prior),
        "prior_status": "SCREEN_FAIL_STOP_UNCHANGED",
        "prior_revision": PRIOR_REVISION,
        "registration": _reference(prior, "REGISTRATION.json"),
        "assessment": _reference(prior, "SCREEN/SCREEN_ASSESSMENT.json"),
        "records_sha256": assessment["records_sha256"],
        "old_authorization_reused": False,
        "old_results_rescored": False,
        "older_9216_raw_status": "DELETED_BY_USER_NOT_AVAILABLE_FOR_REGRESSION",
    }


def register(root, *, source_root, revision):
    from .contract import validate_config

    root, source_root = _root(root, exists=False), _path(source_root)
    _require(source_root == SOURCE_ROOT, "Only the original frozen producer may supply inputs")
    _require(not root.exists(), "Preserve the existing v3 namespace")
    deployment = _deployment(revision)
    resources = validate_deployment_assets(deployment)
    imports = check_deployment_imports(deployment)
    config = read(safe_evidence(source_root, "CONFIG.json"))
    validate_config(config)
    _require(
        file_hash(source_root / "IDENTITY.json") == ORIGIN_IDENTITY_SHA256
        and file_hash(source_root / "PHASES/PILOT_V1/IDENTITY.json") == PILOT_IDENTITY_SHA256,
        "Historical producer identity changed",
    )
    stages, _, _ = _source_data(source_root)
    _validate_producer(source_root, deployment, stages)
    basis = _prior_design_basis()
    binding = read(safe_evidence(source_root, "SOURCE_BINDING.json"))
    _require(set(binding["parents"]) == set(PARENTS), "All three frozen parents required")
    source_files = [inventory_file(deployment / item) for item in _source_paths(deployment)]
    upstream = [inventory_file(source_root / item) for item in SOURCE_REFERENCES]
    images = {}
    for stage in stages.values():
        for item in stage["public"]:
            path = safe_evidence(source_root, item["model_input"]["image_file"])
            if str(path) not in images:
                images[str(path)] = inventory_file(path)
            _require(
                images[str(path)]["sha256"] == item["model_input"]["image_sha256"],
                "New source binding differs from the original image",
            )
    root.mkdir()
    write_once(root / "CONFIG.json", config)
    write_once(root / "SOURCE_MANIFEST.json", {"revision": revision, "files": source_files})
    stage_refs = {}
    for number, stage in stages.items():
        directory = root / "INPUT_PLAN" / f"STAGE{number}"
        directory.mkdir(parents=True)
        (root / f"STAGE{number}").mkdir()
        _write_jsonl(directory / "MODEL_INPUTS.jsonl", stage["public"])
        _write_jsonl(
            directory / "TASKS_PRIVATE.jsonl",
            [stage["private"][row["qid"]] for row in stage["public"]],
        )
        write_once(directory / "SCREEN_INPUT_PLAN.json", stage["plan"])
        stage_refs[number] = {
            name: _reference(root, f"INPUT_PLAN/STAGE{number}/{filename}")
            for name, filename in (
                ("public", "MODEL_INPUTS.jsonl"),
                ("private", "TASKS_PRIVATE.jsonl"),
                ("plan", "SCREEN_INPUT_PLAN.json"),
            )
        }
    registration = {
        "schema": REGISTRATION_SCHEMA,
        "phase": PHASE,
        "status": "BOUND_AWAITING_ACTUAL_PREFLIGHT",
        "root": str(root),
        "source_root": str(source_root),
        "deployment": str(deployment),
        "revision": revision,
        "prompt_version": VERSION,
        "consumer_version": CONSUMER_VERSION,
        "deployment_resources": resources,
        "deployment_import_check": imports,
        "design_basis": basis,
        "historical_status": "FAIL_UNCHANGED_OLD_RAW_DELETED_BY_USER",
        "original_identity_sha256": ORIGIN_IDENTITY_SHA256,
        "historical_pilot_identity_sha256": PILOT_IDENTITY_SHA256,
        "source_manifest": _reference(root, "SOURCE_MANIFEST.json"),
        "source_manifest_sha256": file_hash(root / "SOURCE_MANIFEST.json"),
        "config": _reference(root, "CONFIG.json"),
        "config_canonical_sha256": digest(config),
        "stages": stage_refs,
        "data_sha256": digest(stage_refs),
        "parent_content_hashes": {
            parent: value["content_sha256"] for parent, value in binding["parents"].items()
        },
        "upstream_files": upstream,
        "image_files": list(images.values()),
        "expected_stage1_answers": 576,
        "maximum_new_answers": 1152,
        "all_potential_inputs_frozen_before_first_output": True,
        "optimizer_updates": 0,
        "training_allowed": False,
        "generation_allowed": False,
        "confirmation_allowed": False,
        "authorization_scope": (
            "576 stage-one draws; at most 192 per authorized gray parent in stage two"
        ),
    }
    write_once(root / "REGISTRATION.json", registration)
    return registration


def verify_registered(root, *, stage=1, parent=None, require_preflight=True):
    from .contract import validate_config
    from .runtime import _verify_module_origins, verify_binding_inputs

    _stage(stage, parent, require_preflight=require_preflight)
    root = _root(root)
    registration_path = safe_evidence(root, "REGISTRATION.json")
    registration = read(registration_path)
    _require(
        registration.get("schema") == REGISTRATION_SCHEMA
        and registration.get("phase") == PHASE
        and registration.get("root") == str(root)
        and registration.get("prompt_version") == VERSION
        and registration.get("consumer_version") == CONSUMER_VERSION
        and registration.get("training_allowed") is False
        and registration.get("generation_allowed") is False
        and registration.get("confirmation_allowed") is False
        and type(registration.get("optimizer_updates")) is int
        and registration["optimizer_updates"] == 0
        and registration.get("expected_stage1_answers") == 576
        and registration.get("maximum_new_answers") == 1152
        and registration.get("all_potential_inputs_frozen_before_first_output") is True,
        "Prospective v3 registration scope differs",
    )
    source_root = _path(registration["source_root"])
    _require(source_root == SOURCE_ROOT, "Original input producer differs")
    deployment = _deployment(registration["revision"])
    _require(registration["deployment"] == str(deployment), "Executing deployment differs")
    _require(
        registration.get("deployment_resources") == validate_deployment_assets(deployment)
        and registration.get("deployment_import_check", {}).get("status") == "PASS",
        "Actual deployment assets or imports are missing",
    )
    manifest = _read_reference(root, registration["source_manifest"])
    _require(
        registration["source_manifest_sha256"] == registration["source_manifest"]["sha256"]
        and manifest["revision"] == registration["revision"]
        and len(manifest["files"]) == len({item["path"] for item in manifest["files"]})
        and {item["path"] for item in manifest["files"]}
        == {str(deployment / item) for item in _source_paths(deployment)},
        "Complete actual executing source inventory differs",
    )
    source_check = verify_bound_files(manifest["files"])
    _verify_module_origins(
        deployment,
        {
            "files": [
                {"path": str(Path(row["path"]).relative_to(deployment))}
                for row in manifest["files"]
            ]
        },
        tuple(sys.modules.values()),
    )
    _require(
        len(registration["upstream_files"]) == len(SOURCE_REFERENCES)
        and {row["path"] for row in registration["upstream_files"]}
        == {str(source_root / name) for name in SOURCE_REFERENCES},
        "Upstream provenance inventory differs",
    )
    verify_bound_files(registration["upstream_files"])
    _require(
        registration.get("original_identity_sha256") == ORIGIN_IDENTITY_SHA256
        and file_hash(source_root / "IDENTITY.json") == ORIGIN_IDENTITY_SHA256
        and registration.get("historical_pilot_identity_sha256") == PILOT_IDENTITY_SHA256
        and file_hash(source_root / "PHASES/PILOT_V1/IDENTITY.json") == PILOT_IDENTITY_SHA256,
        "Historical producer identity differs",
    )
    config = _read_reference(root, registration["config"])
    validate_config(config)
    _require(
        config == read(source_root / "CONFIG.json")
        and digest(config) == registration["config_canonical_sha256"],
        "Original sampling/model configuration changed",
    )
    expected_stages, _, _ = _source_data(source_root)
    _validate_producer(source_root, deployment, expected_stages)
    refs = registration.get("stages")
    _require(isinstance(refs, dict) and set(refs) == {"1", "2"}, "Both frozen stages required")
    stages = {}
    for number in ("1", "2"):
        _require(
            isinstance(refs[number], dict) and set(refs[number]) == {"public", "private", "plan"},
            "Exact stage references required",
        )
        public = _read_reference(root, refs[number]["public"], lines=True)
        private_rows = _read_reference(root, refs[number]["private"], lines=True)
        private = {row["qid"]: row for row in private_rows}
        plan = _read_reference(root, refs[number]["plan"])
        expected = expected_stages[number]
        _require(
            public == expected["public"]
            and private == expected["private"]
            and len(private_rows) == len(private) == 96,
            "A frozen stage differs from its deterministic source inputs",
        )
        _validate_plan(plan, public, int(number))
        stages[number] = dict(public=public, private=private, plan=plan)
        _path(root / f"STAGE{number}")
    _require(registration["data_sha256"] == digest(refs), "Two-stage data identity differs")
    _require(
        registration.get("design_basis") == _prior_design_basis(),
        "Preserved failed-v2 design evidence changed",
    )
    all_public = [row for entry in stages.values() for row in entry["public"]]
    image_paths = {str(source_root / row["model_input"]["image_file"]) for row in all_public}
    _require(
        len(registration["image_files"]) == len(image_paths)
        and {row["path"] for row in registration["image_files"]} == image_paths,
        "Exact image inventory for both frozen stages required",
    )
    image_check = verify_bound_files(registration["image_files"])
    pins = {row["path"]: row["sha256"] for row in registration["image_files"]}
    _require(
        all(
            pins[str(source_root / row["model_input"]["image_file"])]
            == row["model_input"]["image_sha256"]
            for row in all_public
        ),
        "Frozen stage image content differs",
    )
    binding, model_check = verify_binding_inputs(source_root, config)
    _require(
        registration["parent_content_hashes"]
        == {p: item["content_sha256"] for p, item in binding["parents"].items()},
        "Frozen parent identities differ",
    )
    for selected_parent in PARENTS:
        verify_binding_inputs(source_root, config, selected_parent)
    pin = file_hash(registration_path)
    preflight = None
    if require_preflight:
        seal = read(safe_evidence(root, "PREFLIGHT_SEAL.json"))
        _require(
            seal
            == dict(
                schema=SEAL_SCHEMA,
                registration_sha256=pin,
                preflight=_reference(root, "PREFLIGHT.json"),
                training_allowed=False,
            ),
            "Actual v3 preflight seal is absent or differs",
        )
        preflight = _read_reference(root, seal["preflight"])
        _validate_preflight(preflight, pin, all_public)
        verify_bound_files(preflight["verified_dependencies"])
        if stage == 2:
            from .interface_v3_screen import stage2_authorized

            _require(
                stage2_authorized(root, parent) is True,
                "Stage two is permitted only for an explicitly validated gray parent",
            )
    selected = stages[str(stage)]
    return {
        "root": root / f"STAGE{stage}",
        "registration_root": root,
        "stage": stage,
        "registration": registration,
        "registration_sha256": pin,
        **selected,
        "all_public": all_public,
        "config": config,
        "source_root": source_root,
        "deployment": deployment,
        "source_manifest": manifest,
        "source_binding": binding,
        "preflight": preflight,
        "verification": {"source": source_check, "images": image_check, "model": model_check},
    }


def _validate_preflight(receipt, pin, public):
    _require(
        receipt.get("schema") == PREFLIGHT_SCHEMA
        and receipt.get("phase") == PHASE
        and receipt.get("status") == "READY_FOR_V3_SCREEN"
        and receipt.get("execution_origin") == "SERVER_ACTUAL"
        and receipt.get("registration_sha256") == pin
        and type(receipt.get("model_calls")) is int
        and receipt["model_calls"] == 0
        and type(receipt.get("optimizer_updates")) is int
        and receipt["optimizer_updates"] == 0
        and receipt.get("training_allowed") is False
        and receipt.get("confirmation_allowed") is False
        and receipt.get("new_visual_review_performed") is False
        and receipt.get("checks")
        == dict.fromkeys(("source", "historical_processor_visual", "new_prompt_processor"), True),
        "Complete actual v3 preflight required",
    )
    rows = receipt.get("inputs", [])
    _require(
        len(public) == len(rows) == len({item["qid"] for item in rows}) == 192
        and {item["qid"] for item in rows} == {item["qid"] for item in public},
        "Actual processor checks require all 192 frozen inputs",
    )
    by_qid = {row["qid"]: row for row in rows}
    for item in public:
        actual = by_qid[item["qid"]]
        _require(
            actual["model_input_sha256"] == digest(item["model_input"])
            and type(actual.get("screen_stage")) is int
            and actual["screen_stage"] == item["screen_stage"],
            "Actual processor input differs from the frozen prompt",
        )
        v2._validate_route(actual["routing"], item["model_input"], receipt["identity"])
    visual = receipt.get("visual_pixel_rechecks", [])
    _require(
        len(visual) == len({row["qid"] for row in visual}) == 48
        and all(row.get("identical_pixels") is True for row in visual),
        "Reused visual review requires 48 current pixel rechecks",
    )
    _require(bool(receipt.get("verified_dependencies")), "Actual dependency inventory required")
    fixtures = receipt.get("tokenizer_fixtures", [])
    _require(
        len(fixtures) == len(TOKENIZER_FIXTURES)
        and all(
            row.get("text") == text
            and row.get("exact_roundtrip") is True
            and isinstance(row.get("token_ids"), list)
            and bool(row["token_ids"])
            and all(type(n) is int and n >= 0 for n in row["token_ids"])
            for row, text in zip(fixtures, TOKENIZER_FIXTURES, strict=True)
        ),
        "Complete actual tokenizer fixtures required",
    )


def _processor(context, facts):
    from mm_dev.data import reconstruct_rgb
    from sr_f5.runtime import NUMERICS_V2

    from .runtime import Runtime

    runtime = Runtime.processor_only(
        context["config"]["model"]["model_path"], protocol_revision="v2"
    )
    runtime._srf5_numerics = dict(NUMERICS_V2)
    old_report = facts.read("PREFLIGHT/PROCESSOR.json")
    for key in ("processor_hash", "chat_template_hash", "chat_template_kwargs_hash"):
        _require(
            runtime.identity.get(key) == old_report["identity"].get(key),
            "Current processor differs from the visual evidence producer: " + key,
        )
    rows, prepared_by_qid = [], {}

    def prepare(model_input):
        prepared = runtime.prepare_input(
            model_input, context["source_root"], protocol="structured_v4"
        )
        v2._validate_route(prepared["routing"], model_input, runtime.identity)
        ids = prepared["inputs"]["input_ids"]
        _require(
            ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[-1] > 0,
            "Actual prompt token tensor missing",
        )
        rgb = reconstruct_rgb(
            prepared["inputs"]["pixel_values"],
            prepared["routing"]["image_grid_thw"],
            runtime.processor.image_processor,
        )
        return dict(
            routing=copy.deepcopy(prepared["routing"]),
            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        )

    for item in context["all_public"]:
        current = prepare(item["model_input"])
        prepared_by_qid[item["qid"]] = current
        rows.append(
            dict(
                qid=item["qid"],
                screen_stage=item["screen_stage"],
                model_input_sha256=digest(item["model_input"]),
                routing=current["routing"],
            )
        )
    _, old_public, old_private = _source_data(context["source_root"])
    old_public = {row["qid"]: row for row in old_public}
    visual = []
    for item in old_report["images"]:
        qid = item["qid"]
        current = prepared_by_qid.get(qid)
        if current is None:
            model_input = copy.deepcopy(old_public[qid]["model_input"])
            task = old_private[qid]
            model_input["prompt"] = public_prompt(
                list(task["world"]["categories"]), dict(task["query"])
            )
            current = prepare(model_input)
        _require(
            current["rgb_sha256"] == item["processed_rgb_sha256"]
            and all(
                current["routing"]["native_input_tensor_hashes"][name]
                == item["routing"]["native_input_tensor_hashes"][name]
                for name in ("pixel_values", "image_grid_thw")
            ),
            "Current preprocessing differs from the original visual review",
        )
        visual.append(
            dict(qid=qid, identical_pixels=True, processed_rgb_sha256=current["rgb_sha256"])
        )
    fixtures = []
    tokenizer = runtime.processor.tokenizer
    for text in TOKENIZER_FIXTURES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        _require(
            ids
            and tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            == text,
            "Actual tokenizer roundtrip failed",
        )
        fixtures.append(dict(text=text, token_ids=ids, exact_roundtrip=True))
    return dict(
        inputs=rows,
        identity=copy.deepcopy(runtime.identity),
        visual_pixel_rechecks=visual,
        tokenizer_fixtures=fixtures,
    )


def preflight(root):
    root = _root(root)
    _require(not (root / "PREFLIGHT.json").exists(), "Preserve the existing preflight result")
    receipt = dict(
        schema=PREFLIGHT_SCHEMA,
        phase=PHASE,
        status="FAIL",
        execution_origin="SERVER_ACTUAL",
        at=now(),
        host=socket.gethostname(),
        registration_sha256=file_hash(safe_evidence(root, "REGISTRATION.json")),
        model_calls=0,
        optimizer_updates=0,
        training_allowed=False,
        confirmation_allowed=False,
        new_visual_review_performed=False,
        checks={},
    )
    try:
        from .admission import _Facts, _source
        from .admission import _processor as historical_processor

        context = verify_registered(root, require_preflight=False)
        facts = _Facts(context["source_root"])
        origin = facts.read("IDENTITY.json")
        historical_phase = {"identity": facts.read("PHASES/PILOT_V1/IDENTITY.json")}
        receipt["historical_source_consumer"] = _source(
            facts, historical_phase, origin, context["config"], consumer_version=CONSUMER_VERSION
        )
        receipt["checks"]["source"] = True
        receipt["historical_visual_consumer"] = historical_processor(
            facts, origin, consumer_version=CONSUMER_VERSION
        )
        receipt["checks"]["historical_processor_visual"] = True
        receipt.update(_processor(context, facts))
        receipt["checks"]["new_prompt_processor"] = True
        records = {item["path"]: item for item in context["registration"]["upstream_files"]}
        binding = context["source_binding"]
        for items in (
            binding["model_files"],
            binding["source_files"],
            [binding["previous_config"], *binding["retention"].values()],
            *(entry["files"] for entry in binding["parents"].values()),
            *binding["retention_images"].values(),
        ):
            for item in items:
                records.setdefault(item["path"], {k: v for k, v in item.items() if k != "qid"})
        for reference in facts.inputs.values():
            path = context["source_root"] / reference["path"]
            if str(path) not in records:
                records[str(path)] = inventory_file(path)
            _require(records[str(path)]["sha256"] == reference["sha256"], "Consumed input changed")
        for reference in facts.external.values():
            path = Path(reference["path"])
            if str(path) not in records:
                records[str(path)] = inventory_file(path)
            _require(
                records[str(path)]["sha256"] == reference["sha256"],
                "Consumed external input changed",
            )
        receipt["verified_dependencies"] = list(records.values())
        verify_bound_files(receipt["verified_dependencies"])
        _require(not facts.synthetic, "Synthetic evidence cannot authorize actual sampling")
        receipt["status"] = "READY_FOR_V3_SCREEN"
        _validate_preflight(receipt, context["registration_sha256"], context["all_public"])
        after = verify_registered(root, require_preflight=False)
        _require(
            after["registration_sha256"] == context["registration_sha256"],
            "Registration changed during actual preflight",
        )
    except Exception as exc:
        receipt.update(status="FAIL", error=dict(type=type(exc).__name__, message=str(exc)))
        write_once(root / "PREFLIGHT.json", receipt)
        raise
    write_once(root / "PREFLIGHT.json", receipt)
    write_once(
        root / "PREFLIGHT_SEAL.json",
        dict(
            schema=SEAL_SCHEMA,
            registration_sha256=receipt["registration_sha256"],
            preflight=_reference(root, "PREFLIGHT.json"),
            training_allowed=False,
        ),
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("register")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--revision", required=True)
    sub.add_parser("preflight").add_argument("--root", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p.add_argument("--parent", choices=PARENTS)
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.root, source_root=args.source_root, revision=args.revision)
    elif args.command == "preflight":
        result = preflight(args.root)
    else:
        context = verify_registered(args.root, stage=args.stage, parent=args.parent)
        result = dict(
            status="VERIFIED_READY_FOR_V3_SCREEN",
            phase=PHASE,
            registration_sha256=context["registration_sha256"],
        )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "phase", "registration_sha256")
                if key in result
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
