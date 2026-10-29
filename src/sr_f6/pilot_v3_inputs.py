"""Immutable, public-only PILOT_V3 inputs over the original frozen DATA.

Only DEV_PILOT (384 paired rows) and DEV_EVAL (96 P0 rows) are projected.
Source/data provenance and every row are reconstructed on verification. These
CPU artifacts grant no model access and never rewrite historical inputs.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import os
import sys
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from srf6_ref.core import FAMILIES

from . import interface_v3_protocol, phases, registration
from .common import digest
from .data import public_prompt as original_prompt
from .planning import root_schedule
from .rollouts import _bound, _path
from .transactions import _fsync_dir, _plain_path, _read_bytes, _require, _sha

PHASE = "PILOT_V3"
DIRECTORY = "PHASE_RUNS/PILOT_V3/INPUTS_V3"
MANIFEST_FILE = "OVERLAY_MANIFEST.json"
SCHEMA = "srf6-pilot-v3-input-overlay-v1"
PROMPT_VERSION = interface_v3_protocol.VERSION
SPLITS = ("DEV_PILOT", "DEV_EVAL")
IDENTITY_FIELDS = ("qid", "root_id", "family", "factor", "variant", "split")
PUBLIC_FIELDS = frozenset(
    (*IDENTITY_FIELDS, "protocol", "model_input", "prompt_sha256", "source_image_status")
)


def _actual_deployment():
    return Path(__file__).resolve(strict=True).parents[2]


def _source_members(root, context, modules):
    """Bind the actually imported generators to this complete phase manifest."""
    manifest = _bound(SimpleNamespace(), root, context["identity"]["source_manifest"])
    inventory = {item["path"]: item for item in manifest["files"]}
    deployment = Path(context["deployment"])
    result = {}
    for module in modules:
        path = _plain_path(module.__file__)
        relative = "src/" + module.__name__.replace(".", "/") + ".py"
        _require(
            path == deployment / relative and path.resolve(strict=True) == path,
            "Imported PILOT_V3 implementation is outside its declared deployment",
        )
        item = inventory.get(relative)
        _require(
            isinstance(item, dict)
            and item.get("sha256") == hashlib.sha256(_read_bytes(path)).hexdigest(),
            "Actual PILOT_V3 implementation differs from its source manifest",
        )
        result[relative] = item["sha256"]
    return result


def _context(root, supplied):
    _require(
        isinstance(supplied, dict)
        and supplied.get("phase") == PHASE
        and _sha(supplied.get("identity_sha256")),
        "Explicit pinned PILOT_V3 phase context required",
    )
    deployment = _actual_deployment()
    _require(supplied.get("deployment") == str(deployment), "Actual PILOT_V3 source required")
    context = phases.verify_phase(
        root,
        PHASE,
        deployment=deployment,
        expected_identity_sha256=supplied["identity_sha256"],
    )
    _require(context == supplied, "PILOT_V3 context differs from fresh provenance verification")
    _require(
        context.get("status") == "BOUND_NOT_AUTHORIZED"
        and context.get("training_allowed") is False
        and context["identity"]["source_manifest"]["path"] == f"PHASES/{PHASE}/SOURCE_MANIFEST.json"
        and context["identity"]["source_manifest"]["sha256"]
        != context["identity"]["origin"]["source_manifest"]["sha256"],
        "PILOT_V3 provenance is separate from historical source and authorization",
    )
    _source_members(root, context, (sys.modules[__name__], interface_v3_protocol, registration))
    return context


def _project_public(item, task):
    """Copy all original fields; only prompt and its version/digest may change."""
    result = copy.deepcopy(item)
    prompt = interface_v3_protocol.public_prompt(task["world"]["categories"], task["query"])
    result["model_input"]["prompt"] = prompt
    result["prompt_version"] = PROMPT_VERSION
    if "prompt_sha256" in item:
        result["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    return result


def _split(reader, root, config, data_ref, split):
    _require(split in SPLITS, "Only the two registered development splits have v3 overlays")
    count, factors = (8, ("P0", "P1")) if split == "DEV_PILOT" else (4, ("P0",))
    _require(config["data"]["roots_per_family"][split] == count, "Development split budget differs")
    public_ref = data_ref(f"DATA/{split}/MODEL_INPUTS.jsonl")
    private_ref = data_ref(f"DATA/{split}/TASKS_PRIVATE.jsonl")
    public = _bound(reader, root, public_ref, kind="jsonl")
    private = _bound(reader, root, private_ref, kind="jsonl")
    qids = [
        f"{split}_{family}_{index:04d}_{factor}_v{view}"
        for family in FAMILIES
        for index in range(count)
        for factor in factors
        for view in range(4)
    ]
    _require(
        len(public) == len(private) == len(qids)
        and [item.get("qid") for item in public] == qids
        and [task.get("qid") for task in private] == qids,
        "Complete ordered development qids required; duplicates/subsets are forbidden",
    )
    rows, reconstruction = [], []
    for item, task in zip(public, private, strict=True):
        _require(
            set(IDENTITY_FIELDS) | {"protocol", "model_input"} <= set(item)
            and set(item) <= PUBLIC_FIELDS
            and all(item[key] == task.get(key) for key in IDENTITY_FIELDS)
            and item["split"] == split
            and item["factor"] in factors
            and item["family"] in FAMILIES
            and type(item["variant"]) is int
            and item["variant"] in range(4)
            and item["qid"] == f"{item['root_id']}_{item['factor']}_v{item['variant']}"
            and item["root_id"].startswith(f"{split}_{item['family']}_")
            and item["protocol"] == interface_v3_protocol.WIRE_PROTOCOL,
            "Original public/private identity or public metadata boundary differs",
        )
        safe = item["model_input"]
        _require(
            isinstance(safe, dict)
            and set(safe) == {"prompt", "image_file", "image_sha256"}
            and safe["prompt"] == original_prompt(task),
            "Original public prompt/model-input whitelist differs",
        )
        _require(
            "prompt_sha256" not in item
            or item["prompt_sha256"] == hashlib.sha256(safe["prompt"].encode()).hexdigest(),
            "Original public prompt digest differs",
        )
        image = safe["image_file"]
        _require(
            isinstance(image, str)
            and image.startswith("DATA/images/")
            and PurePosixPath(image).as_posix() == image
            and ".." not in PurePosixPath(image).parts
            and _sha(safe["image_sha256"])
            and data_ref(image)["sha256"] == safe["image_sha256"],
            "Original image is outside the immutable DATA inventory",
        )
        row = _project_public(item, task)
        rows.append(row)
        reconstruction.append(
            dict(
                qid=item["qid"],
                original_public_sha256=digest(item),
                private_sha256=digest(task),
                projected_public_sha256=digest(row),
            )
        )
    payload = b"".join(registration._payload(row) for row in rows)
    reference = {
        "path": f"{DIRECTORY}/{split}/MODEL_INPUTS.jsonl",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    entry = dict(
        public=reference,
        original_public=public_ref,
        private=private_ref,
        rows=len(rows),
        paired_factors=list(factors),
        qid_order_sha256=digest(qids),
        row_reconstruction_sha256=digest(reconstruction),
    )
    return entry, payload


def build_overlay(root, *, phase_context):
    """Reconstruct both complete public projections, without publishing anything."""
    root = _plain_path(root)
    context = _context(root, phase_context)
    identity = context["identity"]
    reader = SimpleNamespace()
    config = _bound(reader, root, {"path": "CONFIG.json", "sha256": identity["config_sha256"]})
    data = _bound(reader, root, identity["data_manifest"])
    _require(data.get("config_sha256") == digest(config), "Original DATA/config differs")
    files = data.get("files")
    _require(
        isinstance(files, list) and all(isinstance(item, dict) for item in files),
        "Original DATA inventory required",
    )
    inventory = {item["path"]: item["sha256"] for item in files}
    _require(len(inventory) == len(files), "Duplicate original DATA member")

    def data_ref(relative):
        _require(_sha(inventory.get(relative)), "Missing original DATA member")
        return {"path": relative, "sha256": inventory[relative]}

    schedules = {}
    for parent in registration.PARENTS:
        reference = data_ref(f"DATA/SCHEDULES/DEV_PILOT-{parent}-r0.json")
        _require(
            _bound(reader, root, reference)
            == root_schedule(8, config["master_seed"], parent, 0, split="DEV_PILOT"),
            "Original pilot schedule differs from frozen seed generator",
        )
        schedules[parent] = reference
    projected, artifacts = {}, {}
    for split in SPLITS:
        projected[split], payload = _split(reader, root, config, data_ref, split)
        artifacts[f"{split}/MODEL_INPUTS.jsonl"] = payload
    manifest = dict(
        schema=SCHEMA,
        phase=PHASE,
        status="INPUT_OVERLAY_NOT_AUTHORIZED",
        phase_identity={
            "path": f"PHASES/{PHASE}/IDENTITY.json",
            "sha256": context["identity_sha256"],
        },
        actual_source_manifest=identity["source_manifest"],
        origin_identity=identity["origin"]["identity"],
        data_manifest=identity["data_manifest"],
        config_sha256=identity["config_sha256"],
        prompt_version=PROMPT_VERSION,
        wire_protocol=interface_v3_protocol.WIRE_PROTOCOL,
        constructor="sr_f6.interface_v3_protocol.public_prompt(categories, query)",
        public_inputs=["world.categories", "query"],
        generator_sources=_source_members(
            root, context, (sys.modules[__name__], interface_v3_protocol, registration)
        ),
        splits=projected,
        schedules=schedules,
        permitted_changes=["model_input.prompt", "prompt_version", "prompt_sha256_if_present"],
        old_data_modified=False,
        training_authorized=False,
        generation_authorized=False,
        formal_test_access=False,
        execution_status="NOT_RUN",
    )
    artifacts[MANIFEST_FILE] = registration._payload(manifest)
    reference = registration._reference(f"{DIRECTORY}/{MANIFEST_FILE}", manifest)
    return dict(
        manifest_ref=reference,
        manifest=manifest,
        inputs_by_split={
            split: {name: projected[split][name] for name in ("public", "private")}
            for split in SPLITS
        },
        files=artifacts,
    )


def _check_files(destination, bundle):
    expected = set(bundle["files"])
    actual = set()
    directories = {"DEV_PILOT", "DEV_EVAL"}
    for path in destination.rglob("*"):
        _plain_path(path)
        relative = path.relative_to(destination).as_posix()
        if path.is_dir():
            _require(relative in directories, "Unexpected overlay directory")
        else:
            _require(path.is_file(), "Overlay member is not a regular file")
            actual.add(relative)
    _require(actual == expected, "Overlay file membership differs")
    for relative, payload in bundle["files"].items():
        _require(
            _read_bytes(_path(destination, relative)) == payload,
            "Immutable overlay reconstruction differs",
        )


def verify_overlay(root, *, phase_context, overlay_ref):
    """Verify the manifest pin, exact membership and all original row projections."""
    root = _plain_path(root)
    _require(
        isinstance(overlay_ref, dict)
        and set(overlay_ref) == {"path", "sha256"}
        and overlay_ref.get("path") == f"{DIRECTORY}/{MANIFEST_FILE}"
        and _sha(overlay_ref.get("sha256")),
        "Pinned PILOT_V3 overlay manifest required",
    )
    _bound(SimpleNamespace(), root, overlay_ref)
    bundle = build_overlay(root, phase_context=phase_context)
    _require(bundle["manifest_ref"] == overlay_ref, "Overlay differs from fixed reconstruction")
    _check_files(_plain_path(root / DIRECTORY), bundle)
    return {key: bundle[key] for key in ("manifest_ref", "manifest", "inputs_by_split")}


def freeze_overlay(root, *, phase_context):
    """Publish all three immutable files together; preserve partial attempts."""
    root = _plain_path(root)
    bundle = build_overlay(root, phase_context=phase_context)
    destination = _plain_path(root / DIRECTORY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        _plain_path(destination.parent / ".pilot-v3-overlay.lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            _check_files(destination, bundle)
        else:
            staging = destination.with_name(f".pilot-v3-overlay-{uuid.uuid4().hex}")
            staging.mkdir()
            for relative, payload in bundle["files"].items():
                path = _path(staging, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            for directory, _, _ in os.walk(staging, topdown=False):
                _fsync_dir(directory)
            _require(
                not destination.exists() and not destination.is_symlink(),
                "Never replace an existing overlay",
            )
            os.rename(staging, destination)
            _fsync_dir(destination.parent)
        _check_files(destination, bundle)
    finally:
        os.close(fd)
    return {key: bundle[key] for key in ("manifest_ref", "manifest", "inputs_by_split")}
