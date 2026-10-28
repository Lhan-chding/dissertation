"""Bind one interface-v2 engineering screen without reviving the old admission.

The old source and visual receipts retain their actual producer. New prompts,
processor tensors and executing source get independent, immutable identities.
No model is loaded here, and a successful preflight authorizes only 576 draws.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import socket
import sys
from pathlib import Path

from .common import digest, file_hash, now, read, safe_evidence, write_once
from .compatibility import CONSUMER_VERSION
from .interface_protocol import VERSION
from .interface_repair import prepare_screen, screen_inputs, screen_plan
from .source_binding import PROJECT, inventory_file, verify_bound_files

ROOT = PROJECT / "sr_f6_interface_v2_20261028"
SOURCE_ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
PHASE = "INTERFACE_SCREEN_V2"
REGISTRATION_SCHEMA = "srf6-interface-screen-registration-v1"
PREFLIGHT_SCHEMA = "srf6-interface-screen-preflight-v1"
TOKENIZER_FIXTURES = (
    '{"evidence":[],"selected":["Example_A"],"bindings":[{"category":"Example_A","operand":-12}],"aggregate":"sum","answer":-12}',
    '{"work":"选择与计算","evidence":[],"selected":[],"bindings":[],"aggregate":"mean","answer":"1/2"}',
    '{"evidence":[],"selected":["Example_\\u0041"],"bindings":[],"aggregate":"sum","answer":1}',
)
ORIGIN_IDENTITY_SHA256 = "25728b59d2838a7d280813d579f980d3718150b6a453123a01b0059e43a2b52a"
PILOT_IDENTITY_SHA256 = "856dd5e5e16afd4275b98aa2b7b3bac4b80046acbfabac5cc10ba3adf470253a"
SOURCE_REFERENCES = (
    "CONFIG.json",
    "IDENTITY.json",
    "SOURCE_MANIFEST.json",
    "SOURCE_BINDING.json",
    "DATA/MANIFEST.json",
    "DATA/DEV_CAL/MODEL_INPUTS.jsonl",
    "DATA/DEV_CAL/TASKS_PRIVATE.jsonl",
    "PHASES/PILOT_V1/IDENTITY.json",
    "PREFLIGHT/CENSUS_READY.json",
    "PREFLIGHT/PROCESSOR.json",
    "PREFLIGHT/VISUAL_REVIEW.json",
)


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
    _require(root == ROOT, "Only the registered independent screen root is allowed")
    return root


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
    members = set()
    for relative, extensions in (
        ("src", {".py"}),
        ("tests/sr_f6", {".py"}),
        ("scripts/sr_f6", {".py", ".sh", ".sbatch"}),
    ):
        base = _path(deployment / relative)
        for path in base.rglob("*"):
            _path(path)
            if path.is_file() and path.suffix in extensions:
                members.add(str(path.relative_to(deployment)))
    members.add("docs/sr_f6/package/configs/experiment.json")
    _require(bool(members), "Executing source inventory is empty")
    return sorted(members)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _reference(root, relative):
    path = safe_evidence(root, relative)
    return {"path": relative, "sha256": file_hash(path)}


def _read_reference(root, reference, *, lines=False):
    _require(
        isinstance(reference, dict) and set(reference) == {"path", "sha256"},
        "Exact evidence reference required",
    )
    path = safe_evidence(root, reference["path"])
    _require(file_hash(path) == reference["sha256"], "Screen evidence content changed")
    return _jsonl(path) if lines else read(path)


def _validate_plan(plan, public):
    expected = screen_plan(public)
    _require(
        all(plan.get(key) == value for key, value in expected.items()),
        "Screen plan differs from the single fixed 576-draw candidate",
    )
    _require(
        plan.get("generation_allowed") is False and plan.get("training_allowed") is False,
        "An input plan cannot grant execution",
    )


def _source_data(source_root):
    old_public = _jsonl(safe_evidence(source_root, "DATA/DEV_CAL/MODEL_INPUTS.jsonl"))
    old_private = _jsonl(safe_evidence(source_root, "DATA/DEV_CAL/TASKS_PRIVATE.jsonl"))
    public = screen_inputs(old_public, old_private)
    qids = {row["qid"] for row in public}
    private = {row["qid"]: row for row in old_private if row["qid"] in qids}
    return public, private, old_public, {row["qid"]: row for row in old_private}


def _validate_producer(source_root, deployment):
    """Check frozen old DEV_CAL bytes and the unchanged scorer/native sampler."""
    from .source_binding import SOURCE_FILES

    origin = read(safe_evidence(source_root, "IDENTITY.json"))
    _require(
        file_hash(source_root / "CONFIG.json") == origin["config_sha256"],
        "Original config bytes changed",
    )
    data_manifest = _read_reference(source_root, origin["data_manifest"])
    members = {item["path"]: item for item in data_manifest["files"]}
    _require(len(members) == len(data_manifest["files"]), "Duplicate original data manifest member")
    for name in ("DATA/DEV_CAL/MODEL_INPUTS.jsonl", "DATA/DEV_CAL/TASKS_PRIVATE.jsonl"):
        _read_reference(
            source_root, {key: members[name][key] for key in ("path", "sha256")}, lines=True
        )
    public, private, _, _ = _source_data(source_root)
    for item in public:
        model = item["model_input"]
        _require(
            model["image_file"] == f"DATA/images/{digest(private[item['qid']]['world'])}.png"
            and members[model["image_file"]]["sha256"] == model["image_sha256"],
            "Original task/world/image manifest lineage differs",
        )
    old_manifest = _read_reference(source_root, origin["source_manifest"])
    sources = {row["path"]: row for row in old_manifest["files"]}
    for relative in (*SOURCE_FILES, "src/sr_f6/measurement.py"):
        _require(
            file_hash(safe_evidence(deployment, relative)) == sources[relative]["sha256"],
            "Inherited native implementation or strict scorer changed: " + relative,
        )
    pilot = read(source_root / "PHASES/PILOT_V1/IDENTITY.json")
    old_deployment = _path(pilot["origin"]["deployment"])
    old_runtime = safe_evidence(old_deployment, "src/sr_f6/runtime.py")
    _require(
        file_hash(old_runtime) == sources["src/sr_f6/runtime.py"]["sha256"],
        "Historical sampler source changed",
    )

    def sampler_nodes(path):
        parsed = ast.parse(path.read_text())
        cls = next(
            node
            for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == "Runtime"
        )
        native = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_generate_native_group"
        )
        recipe = next(
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name == "generation_recipe"
        )
        return [ast.dump(node, include_attributes=False) for node in (native, recipe)]

    _require(
        sampler_nodes(old_runtime) == sampler_nodes(deployment / "src/sr_f6/runtime.py"),
        "Native sampling or generation recipe changed",
    )


def register(root, *, source_root, revision):
    """Create a new immutable namespace; never overwrite the original run."""
    from .contract import validate_config

    root, source_root = _root(root, exists=False), _path(source_root)
    _require(source_root == SOURCE_ROOT, "Only the actual original input producer is allowed")
    _require(not root.exists(), "Preserve an existing screen namespace")
    deployment = _deployment(revision)
    config = read(safe_evidence(source_root, "CONFIG.json"))
    validate_config(config)
    _require(
        file_hash(safe_evidence(source_root, "IDENTITY.json")) == ORIGIN_IDENTITY_SHA256
        and file_hash(safe_evidence(source_root, "PHASES/PILOT_V1/IDENTITY.json"))
        == PILOT_IDENTITY_SHA256,
        "Historical producer identity changed",
    )
    public, private, _, _ = _source_data(source_root)
    _validate_producer(source_root, deployment)
    binding = read(safe_evidence(source_root, "SOURCE_BINDING.json"))
    _require(set(binding["parents"]) == {"b0", "b1", "b2"}, "Three frozen parents required")
    # Content hashing occurs once at this new source/input freeze boundary.
    source_files = [inventory_file(deployment / item) for item in _source_paths(deployment)]
    upstream_files = [inventory_file(source_root / item) for item in SOURCE_REFERENCES]
    images = {}
    for item in public:
        path = safe_evidence(source_root, item["model_input"]["image_file"])
        if str(path) not in images:
            images[str(path)] = inventory_file(path)
        _require(
            images[str(path)]["sha256"] == item["model_input"]["image_sha256"],
            "Screen source image differs from original binding",
        )
    root.mkdir()
    plan = prepare_screen(source_root, root / "INPUT_PLAN")
    write_once(root / "CONFIG.json", config)
    write_once(root / "SOURCE_MANIFEST.json", {"revision": revision, "files": source_files})
    with (root / "TASKS_PRIVATE.jsonl").open("x") as stream:
        for item in public:
            stream.write(json.dumps(private[item["qid"]], ensure_ascii=False) + "\n")
    inputs_ref = _reference(root, "INPUT_PLAN/MODEL_INPUTS.jsonl")
    private_ref = _reference(root, "TASKS_PRIVATE.jsonl")
    plan_ref = _reference(root, "INPUT_PLAN/SCREEN_INPUT_PLAN.json")
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
        "historical_status": "FAIL_UNCHANGED_OLD_RAW_DELETED_BY_USER",
        "original_identity_sha256": ORIGIN_IDENTITY_SHA256,
        "historical_pilot_identity_sha256": PILOT_IDENTITY_SHA256,
        "source_manifest": _reference(root, "SOURCE_MANIFEST.json"),
        "source_manifest_sha256": file_hash(root / "SOURCE_MANIFEST.json"),
        "config": _reference(root, "CONFIG.json"),
        "config_canonical_sha256": digest(config),
        "public": inputs_ref,
        "private": private_ref,
        "plan": plan_ref,
        "data_sha256": digest({"public": inputs_ref, "private": private_ref, "plan": plan_ref}),
        "parent_content_hashes": {
            parent: value["content_sha256"] for parent, value in binding["parents"].items()
        },
        "upstream_files": upstream_files,
        "image_files": list(images.values()),
        "expected_answers": plan["expected_new_answers"],
        "optimizer_updates": 0,
        "training_allowed": False,
        "generation_allowed": False,
        "confirmation_allowed": False,
        "authorization_scope": "576 engineering-screen draws only after actual preflight",
    }
    write_once(root / "REGISTRATION.json", registration)
    return registration


def verify_registered(root, *, require_preflight=True):
    """Verify actual loaded code and inputs; never grant training or confirmation."""
    from .contract import validate_config
    from .runtime import _verify_module_origins, verify_binding_inputs

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
        and registration.get("optimizer_updates") == 0
        and registration.get("expected_answers") == 576,
        "Screen registration scope differs",
    )
    source_root = _path(registration["source_root"])
    _require(source_root == SOURCE_ROOT, "Original producer root differs")
    deployment = _deployment(registration["revision"])
    _require(registration["deployment"] == str(deployment), "Actual executing deployment differs")
    manifest = _read_reference(root, registration["source_manifest"])
    _require(
        registration["source_manifest_sha256"] == registration["source_manifest"]["sha256"]
        and manifest["revision"] == registration["revision"],
        "Source manifest identity differs",
    )
    _require(
        {item["path"] for item in manifest["files"]}
        == {str(deployment / item) for item in _source_paths(deployment)},
        "Complete actual source inventory differs",
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
        {row["path"] for row in registration["upstream_files"]}
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
    _validate_producer(source_root, deployment)
    config = _read_reference(root, registration["config"])
    validate_config(config)
    _require(
        config == read(source_root / "CONFIG.json")
        and digest(config) == registration["config_canonical_sha256"],
        "Original sampling/model recipe changed",
    )
    public = _read_reference(root, registration["public"], lines=True)
    private_rows = _read_reference(root, registration["private"], lines=True)
    private = {row["qid"]: row for row in private_rows}
    expected_public, expected_private, _, _ = _source_data(source_root)
    _require(
        public == expected_public and private == expected_private and len(private_rows) == 96,
        "Screen data differs from exactly the first two registered roots",
    )
    plan = _read_reference(root, registration["plan"])
    _validate_plan(plan, public)
    _require(
        registration["data_sha256"]
        == digest({key: registration[key] for key in ("public", "private", "plan")}),
        "Screen data identity differs",
    )
    image_paths = {str(source_root / row["model_input"]["image_file"]) for row in public}
    _require(
        {row["path"] for row in registration["image_files"]} == image_paths,
        "Exact screen image inventory required",
    )
    image_check = verify_bound_files(registration["image_files"])
    pins = {row["path"]: row["sha256"] for row in registration["image_files"]}
    _require(
        all(
            pins[str(source_root / row["model_input"]["image_file"])]
            == row["model_input"]["image_sha256"]
            for row in public
        ),
        "Screen image content identity differs",
    )
    binding, model_check = verify_binding_inputs(source_root, config)
    _require(
        registration["parent_content_hashes"]
        == {parent: item["content_sha256"] for parent, item in binding["parents"].items()},
        "Frozen parent identities differ",
    )
    for parent in ("b0", "b1", "b2"):
        verify_binding_inputs(source_root, config, parent)
    pin = file_hash(registration_path)
    preflight = None
    if require_preflight:
        seal = read(safe_evidence(root, "PREFLIGHT_SEAL.json"))
        _require(
            seal
            == {
                "schema": "srf6-interface-screen-preflight-seal-v1",
                "registration_sha256": pin,
                "preflight": _reference(root, "PREFLIGHT.json"),
                "training_allowed": False,
            },
            "Actual preflight seal is absent or differs",
        )
        preflight = _read_reference(root, seal["preflight"])
        _validate_preflight(preflight, pin, public)
        verify_bound_files(preflight["verified_dependencies"])
    return {
        "root": root,
        "registration": registration,
        "registration_sha256": pin,
        "plan": plan,
        "public": public,
        "private": private,
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
        and receipt.get("status") == "READY_FOR_576_SCREEN"
        and receipt.get("execution_origin") == "SERVER_ACTUAL"
        and receipt.get("registration_sha256") == pin
        and receipt.get("model_calls") == receipt.get("optimizer_updates") == 0
        and receipt.get("training_allowed") is False
        and receipt.get("confirmation_allowed") is False
        and receipt.get("new_visual_review_performed") is False
        and receipt.get("checks")
        == dict.fromkeys(("source", "historical_processor_visual", "new_prompt_processor"), True),
        "Complete actual preflight for this registration required",
    )
    rows = receipt.get("inputs", [])
    _require(
        len(rows) == 96 and {item["qid"] for item in rows} == {item["qid"] for item in public},
        "Actual processor checks require all 96 new inputs",
    )
    by_qid = {item["qid"]: item for item in rows}
    for item in public:
        actual = by_qid[item["qid"]]
        _require(
            actual["model_input_sha256"] == digest(item["model_input"]),
            "Actual processor input differs from frozen new prompt",
        )
        _validate_route(actual["routing"], item["model_input"], receipt["identity"])
    _require(
        len(receipt.get("visual_pixel_rechecks", [])) == 48
        and len({row["qid"] for row in receipt["visual_pixel_rechecks"]}) == 48
        and all(row.get("identical_pixels") is True for row in receipt["visual_pixel_rechecks"]),
        "Reused visual review requires all 48 current processor pixel checks",
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
            and all(type(token) is int and token >= 0 for token in row["token_ids"])
            for row, text in zip(fixtures, TOKENIZER_FIXTURES, strict=True)
        ),
        "Complete actual tokenizer fixtures required",
    )


def _validate_route(route, model_input, identity):
    hashes = route.get("native_input_tensor_hashes", {})
    _require(
        set(hashes)
        == {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
        and all(
            isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin) for pin in hashes.values()
        )
        and all(
            isinstance(route.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", route[key])
            for key in ("input_tensor_hash", "chat_text_sha256")
        )
        and route.get("source_image_sha256") == model_input["image_sha256"]
        and route.get("processor_hash") == identity["processor_hash"]
        and route.get("chat_template_hash") == identity["chat_template_hash"]
        and route.get("source_size") == [1280, 896]
        and route.get("input_float_dtype") == route.get("probability_dtype") == "float32"
        and route.get("stop_rule") == "first_balanced_json"
        and route.get("assistant_prefill") == "{"
        and route.get("sr_f6_protocol_version") == "structured_v4-srf6-1.0",
        "Actual native processor route is incomplete or differs",
    )
    grid = route.get("image_grid_thw", [])
    _require(
        len(grid) == 1
        and len(grid[0]) == 3
        and grid[0][0] == 1
        and all(type(n) is int and n > 0 for n in grid[0])
        and route.get("processed_size") == [16 * grid[0][2], 16 * grid[0][1]]
        and type(route.get("image_token_count")) is int
        and route["image_token_count"] * 4 == grid[0][1] * grid[0][2],
        "Actual image-token geometry differs",
    )


def _processor(context, facts):
    from mm_dev.data import reconstruct_rgb
    from sr_f5.runtime import NUMERICS_V2

    from .data import public_prompt
    from .runtime import Runtime

    runtime = Runtime.processor_only(
        context["config"]["model"]["model_path"], protocol_revision="v2"
    )
    runtime._srf5_numerics = dict(NUMERICS_V2)
    old_report = facts.read("PREFLIGHT/PROCESSOR.json")
    for key in ("processor_hash", "chat_template_hash", "chat_template_kwargs_hash"):
        _require(
            runtime.identity.get(key) == old_report["identity"].get(key),
            "Current processor identity differs from the reviewed producer: " + key,
        )
    rows, prepared_by_qid = [], {}
    for item in context["public"]:
        prepared = runtime.prepare_input(
            item["model_input"], context["source_root"], protocol="structured_v4"
        )
        _validate_route(prepared["routing"], item["model_input"], runtime.identity)
        _require(
            prepared["inputs"]["input_ids"].ndim == 2
            and prepared["inputs"]["input_ids"].shape[0] == 1
            and prepared["inputs"]["input_ids"].shape[-1] > 0,
            "Actual prompt token tensor missing",
        )
        rows.append(
            {
                "qid": item["qid"],
                "model_input_sha256": digest(item["model_input"]),
                "routing": copy.deepcopy(prepared["routing"]),
            }
        )
        # Keep only routing and reconstructed RGB identity, not all image tensors.
        rgb = reconstruct_rgb(
            prepared["inputs"]["pixel_values"],
            prepared["routing"]["image_grid_thw"],
            runtime.processor.image_processor,
        )
        prepared_by_qid[item["qid"]] = {
            "routing": copy.deepcopy(prepared["routing"]),
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        }
    _, _, old_public, old_private = _source_data(context["source_root"])
    old_public = {row["qid"]: row for row in old_public}
    visual = []
    for item in old_report["images"]:
        qid = item["qid"]
        current = prepared_by_qid.get(qid)
        if current is None:
            model_input = copy.deepcopy(old_public[qid]["model_input"])
            model_input["prompt"] = public_prompt(old_private[qid], prompt_version=VERSION)
            prepared = runtime.prepare_input(
                model_input, context["source_root"], protocol="structured_v4"
            )
            _validate_route(prepared["routing"], model_input, runtime.identity)
            rgb = reconstruct_rgb(
                prepared["inputs"]["pixel_values"],
                prepared["routing"]["image_grid_thw"],
                runtime.processor.image_processor,
            )
            current = {
                "routing": prepared["routing"],
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
        _require(
            current["rgb_sha256"] == item["processed_rgb_sha256"]
            and all(
                current["routing"]["native_input_tensor_hashes"][name]
                == item["routing"]["native_input_tensor_hashes"][name]
                for name in ("pixel_values", "image_grid_thw")
            ),
            "Current image preprocessing differs from the original visual review",
        )
        visual.append(
            {"qid": qid, "identical_pixels": True, "processed_rgb_sha256": current["rgb_sha256"]}
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
        fixtures.append({"text": text, "token_ids": ids, "exact_roundtrip": True})
    return {
        "inputs": rows,
        "identity": copy.deepcopy(runtime.identity),
        "visual_pixel_rechecks": visual,
        "tokenizer_fixtures": fixtures,
    }


def preflight(root):
    """Run retained source consumers and the real processor, recording failures."""
    from .admission import _Facts, _source
    from .admission import _processor as historical_processor

    root = _root(root)
    _require(not (root / "PREFLIGHT.json").exists(), "Preserve the existing preflight result")
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "phase": PHASE,
        "status": "FAIL",
        "execution_origin": "SERVER_ACTUAL",
        "at": now(),
        "host": socket.gethostname(),
        "registration_sha256": file_hash(safe_evidence(root, "REGISTRATION.json")),
        "model_calls": 0,
        "optimizer_updates": 0,
        "training_allowed": False,
        "confirmation_allowed": False,
        "new_visual_review_performed": False,
        "checks": {},
    }
    try:
        context = verify_registered(root, require_preflight=False)
        facts = _Facts(context["source_root"])
        origin = facts.read("IDENTITY.json")
        # This is the actual saved historical identity, not a fabricated new phase.
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
        # Pin every actual consumed dependency. Existing verified source-bound
        # model/parent assets keep their hashes/stat records; do not rehash weights.
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
                records.setdefault(
                    item["path"], {key: value for key, value in item.items() if key != "qid"}
                )
        for reference in facts.inputs.values():
            path = context["source_root"] / reference["path"]
            if str(path) not in records:
                records[str(path)] = inventory_file(path)
            _require(
                records[str(path)]["sha256"] == reference["sha256"],
                "Consumed input changed before preflight seal",
            )
        for reference in facts.external.values():
            path = Path(reference["path"])
            if str(path) not in records:
                records[str(path)] = inventory_file(path)
            _require(
                records[str(path)]["sha256"] == reference["sha256"],
                "Consumed external source changed before preflight seal",
            )
        receipt["verified_dependencies"] = list(records.values())
        verify_bound_files(receipt["verified_dependencies"])
        _require(not facts.synthetic, "Synthetic evidence cannot authorize actual sampling")
        receipt["status"] = "READY_FOR_576_SCREEN"
        _validate_preflight(receipt, context["registration_sha256"], context["public"])
        final_context = verify_registered(root, require_preflight=False)
        _require(
            final_context["registration_sha256"] == context["registration_sha256"],
            "Registration changed during processor verification",
        )
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_once(root / "PREFLIGHT.json", receipt)
        raise
    write_once(root / "PREFLIGHT.json", receipt)
    write_once(
        root / "PREFLIGHT_SEAL.json",
        {
            "schema": "srf6-interface-screen-preflight-seal-v1",
            "registration_sha256": receipt["registration_sha256"],
            "preflight": _reference(root, "PREFLIGHT.json"),
            "training_allowed": False,
        },
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--root", type=Path, required=True)
    register_parser.add_argument("--source-root", type=Path, required=True)
    register_parser.add_argument("--revision", required=True)
    for name in ("preflight", "verify"):
        sub.add_parser(name).add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.root, source_root=args.source_root, revision=args.revision)
    elif args.command == "preflight":
        result = preflight(args.root)
    else:
        context = verify_registered(args.root)
        result = {
            "status": "VERIFIED_READY_FOR_576_SCREEN",
            "registration_sha256": context["registration_sha256"],
        }
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
