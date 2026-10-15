"""CPU-only provenance checks for the three frozen SR-F2 J6 parents.

The package's released authorization is the trust anchor. A pickle checkpoint
is opened only after its byte hash has been authenticated through that chain.
This module never loads a model or mutates a parent artifact.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PARENTS = tuple(f"SRF2_J6_s{seed}" for seed in (71001, 71002, 71003))
ADAPTER_FILES = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
CHECKPOINT_FIELDS = {
    "plan_id",
    "run_identity",
    "committed_logical_step",
    "parameters",
    "optimizer",
    "scheduler",
    "rng",
    "reference",
    "reference_hash",
    "input_stream_hash",
    "sampling_hash",
    "cursor",
    "diagnostics_hash",
    "token_path_hash",
    "training_recipe",
    "adaptive_state",
    "adaptive_state_hash",
    "monitor_due_step",
    "monitor_definition_hash",
}


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def bounded_file(root, relative, *, allow_internal_symlink=True):
    root, relative = Path(root).resolve(strict=True), Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PermissionError(f"Unsafe relative path: {relative}")
    lexical = root / relative
    target = lexical.resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise PermissionError(f"File escapes its root: {relative}")
    if not allow_internal_symlink and lexical.is_symlink():
        raise PermissionError(f"Symlink not allowed: {relative}")
    return target


def verify_file(root, relative, checksum, size=None):
    target = bounded_file(root, relative)
    if file_hash(target) != checksum or (size is not None and target.stat().st_size != size):
        raise PermissionError(f"Artifact bytes changed: {relative}")
    return target


def _destination(root, relative):
    root, relative = Path(root).resolve(strict=True), Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise PermissionError("Unsafe output path")
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise PermissionError("Output path escapes the new root")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _publish_temporary(temp, destination, expected_hash):
    """Atomically publish without replacing an existing artifact."""
    if file_hash(temp) != expected_hash:
        raise PermissionError("Copied artifact differs before publication")
    try:
        os.link(temp, destination)
    except FileExistsError:
        if destination.is_symlink() or file_hash(destination) != expected_hash:
            raise PermissionError(f"Existing output differs: {destination}") from None


def copy_verified(source, output_root, relative, expected_hash=None):
    source = Path(source).resolve(strict=True)
    expected_hash = expected_hash or file_hash(source)
    destination = _destination(output_root, relative)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".copy-", delete=False) as f:
        temporary = Path(f.name)
        try:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, f, length=8 << 20)
            f.flush()
            os.fsync(f.fileno())
            _publish_temporary(temporary, destination, expected_hash)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(relative), "sha256": expected_hash, "bytes": destination.stat().st_size}


def write_once(output_root, relative, value):
    body = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    destination = _destination(output_root, relative)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".json-", delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
            _publish_temporary(temporary, destination, hashlib.sha256(body).hexdigest())
        finally:
            temporary.unlink(missing_ok=True)
    return value


def verify_plan(plan_dir):
    plan_dir = Path(plan_dir).resolve(strict=True)
    inventory = read_json(plan_dir / "PACKAGE_SHA256.json")
    for relative, checksum in inventory.items():
        verify_file(plan_dir, relative, checksum)
    config = read_json(plan_dir / "config/SR_F3_CE.json")
    if (
        config["model"]["id"] != MODEL_ID
        or config["model"]["revision"] != REVISION
        or tuple(config["parents"]) != PARENTS
    ):
        raise PermissionError("The frozen SR-F3-CE parent/base contract differs")
    return config


def verify_release(parent_root, plan_dir):
    root = Path(parent_root).resolve(strict=True)
    authorization_path = bounded_file(root, "released/AUTHORIZATION.json")
    packaged = Path(plan_dir) / "sources/ORIGINAL_SRF2_AUTHORIZATION.json"
    if file_hash(authorization_path) != file_hash(packaged):
        raise PermissionError(
            "Released authorization differs from the supplied package trust anchor"
        )
    authorization = read_json(authorization_path)
    complete = read_json(bounded_file(root, "released/COMPLETE.json"))
    registration = read_json(bounded_file(root, "REGISTRATION.json"))
    if (
        authorization.get("status") != "AUTHORIZED_AFTER_ALL_GPU_TASKS_VERIFIED"
        or authorization.get("experiment_id") != "SR-F2-20261013"
        or authorization.get("registration_sha256") != object_hash(registration)
        or complete.get("status") != "COMPLETE_COVERAGE_VERIFIED"
        or complete.get("experiment_complete") is not True
        or complete.get("authorization_sha256") != object_hash(authorization)
    ):
        raise PermissionError("SR-F2 release/registration chain differs")
    auth_entries = [
        x for x in complete.get("files", []) if x.get("path") == "released/AUTHORIZATION.json"
    ]
    if len(auth_entries) != 1:
        raise PermissionError("Release completion omits its authorization file identity")
    verify_file(
        root, auth_entries[0]["path"], auth_entries[0]["sha256"], auth_entries[0].get("bytes")
    )
    if (
        registration.get("version") != "SR-F2-20261013"
        or registration.get("status") != "FROZEN_BEFORE_GPU"
        or Path(registration["root"]).resolve(strict=True) != root
    ):
        raise PermissionError("Parent registration has a different identity/root")
    prior_config = read_json(bounded_file(root, "config/SR_F2.json"))
    if object_hash(prior_config) != registration["config_sha256"]:
        raise PermissionError("Parent configuration differs from registration")
    for relative, checksum in registration["bound_files"].items():
        verify_file(root, relative, checksum)
    return authorization, registration


def verify_parent_receipt(root, parent_id, authorization, registration):
    task = read_json(bounded_file(root, f"tasks/train/{parent_id}/COMPLETE.json"))
    expected = {
        "kind": "train",
        "model_id": parent_id,
        "status": "COMPLETE",
        "step": 96,
        "registration_sha256": object_hash(registration),
    }
    if any(task.get(k) != v for k, v in expected.items()):
        raise PermissionError(f"Parent worker receipt differs: {parent_id}")
    refs = [
        x
        for x in authorization["tasks"]
        if x.get("kind") == "train" and x.get("model_id") == parent_id
    ]
    if len(refs) != 1 or refs[0]["receipt_sha256"] != object_hash(task):
        raise PermissionError(f"Parent worker is not the released training task: {parent_id}")
    expected_names = {
        f"runs/{parent_id}/{name}"
        for name in ("COMPLETE.json", "checkpoints/commit-96.json", "monitor_state/step-96.json")
    }
    artifacts = task.get("artifacts", [])
    if len(artifacts) != 3 or {a["path"] for a in artifacts} != expected_names:
        raise PermissionError("Parent worker evidence set differs")
    for artifact in artifacts:
        verify_file(root, artifact["path"], artifact["sha256"], artifact.get("bytes"))
    return task


def expected_language_modules(base_config):
    text = base_config["text_config"]
    count = text["num_hidden_layers"]
    layer_types = text.get("layer_types")
    if not layer_types or len(layer_types) != count:
        raise PermissionError("Base configuration has no complete native decoder layer types")
    modules = []
    for index, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{index}"
        modules.extend(f"{prefix}.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj"))
        if layer_type == "full_attention":
            branch, suffixes = "self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")
        elif layer_type == "linear_attention":
            branch, suffixes = (
                "linear_attn",
                ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"),
            )
        else:
            raise PermissionError(f"Unknown native layer type: {layer_type}")
        modules.extend(f"{prefix}.{branch}.{name}" for name in suffixes)
    return sorted(modules)


def verify_tensor_inventory(parameters, adapter_tensors, full_modules, adapter_config):
    import torch

    from mm_core.training import state_hash

    cfg = adapter_config
    if (
        (cfg.get("r"), cfg.get("lora_alpha"), cfg.get("lora_dropout"), cfg.get("bias"))
        != (8, 16, 0, "none")
        or cfg.get("peft_type") != "LORA"
        or cfg.get("use_dora")
        or cfg.get("use_rslora")
        or cfg.get("use_qalora")
        or cfg.get("use_bdlora")
        or cfg.get("lora_bias")
        or cfg.get("fan_in_fan_out")
        or cfg.get("layer_replication")
        or cfg.get("layers_to_transform")
        or cfg.get("target_parameters")
        or cfg.get("trainable_token_indices")
        or cfg.get("exclude_modules")
        or cfg.get("rank_pattern")
        or cfg.get("alpha_pattern")
        or cfg.get("modules_to_save")
    ):
        raise PermissionError("Adapter does not have the frozen inherited LoRA recipe")
    full_modules = sorted(full_modules)
    if len(full_modules) != len(set(full_modules)) or not full_modules:
        raise PermissionError("Duplicate or absent full module identities")
    targets = set(cfg.get("target_modules", []))
    suffixes = {name.rsplit(".", 1)[1] for name in full_modules}
    if targets != suffixes and targets != set(full_modules):
        raise PermissionError("Adapter target suffixes differ from complete language targets")
    expected = {
        f"base_model.model.{name}.lora_{side}.default.weight"
        for name in full_modules
        for side in ("A", "B")
    }
    if set(parameters) != expected:
        raise PermissionError("Checkpoint parameters omit or add a language LoRA tensor")
    mapped = {name.replace(".default.weight", ".weight"): name for name in expected}
    if len(mapped) != len(expected) or set(adapter_tensors) != set(mapped):
        raise PermissionError("Safetensors/default key mapping is not one-to-one and complete")
    inventory = []
    for saved_name, checkpoint_name in sorted(mapped.items()):
        p, a = parameters[checkpoint_name], adapter_tensors[saved_name]
        if (
            not isinstance(p, torch.Tensor)
            or not isinstance(a, torch.Tensor)
            or p.dtype != torch.float32
            or a.dtype != p.dtype
            or p.shape != a.shape
            or p.ndim != 2
            or not torch.isfinite(p).all()
            or not torch.equal(p, a)
        ):
            raise PermissionError(f"Adapter/checkpoint tensor differs: {checkpoint_name}")
        rank_axis = 0 if ".lora_A." in checkpoint_name else 1
        if p.shape[rank_axis] != 8:
            raise PermissionError(f"Wrong LoRA rank: {checkpoint_name}")
        inventory.append(
            {
                "checkpoint_key": checkpoint_name,
                "safetensors_key": saved_name,
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "state_hash": state_hash(p),
                "numel": p.numel(),
            }
        )
    return inventory


def load_verified_checkpoint(directory, marker, manifest, *, loader=None):
    from mm_core.training import state_hash

    # Only reached after released task -> COMPLETE/marker byte verification.
    path = verify_file(directory, marker["path"], marker["sha256"])
    if path.parent != Path(directory).resolve(strict=True):
        raise PermissionError("Checkpoint must be a direct child of its checkpoint directory")
    if loader is None:
        import torch

        def loader(p):
            return torch.load(p, map_location="cpu", weights_only=False)

    state = loader(path)
    if (
        set(state) != CHECKPOINT_FIELDS
        or state["plan_id"] != "SR-F2"
        or state["committed_logical_step"] != 96
        or state["run_identity"] != manifest
        or state["cursor"] != {"next_logical_step": 97, "next_slot": 0, "next_sample_index": 0}
        or state_hash(state) != marker["state_hash"]
        or {k: state_hash(v) for k, v in state.items()} != marker["field_hashes"]
        or state_hash(state["reference"]) != state["reference_hash"]
        or manifest.get("reference_hash") != state["reference_hash"]
    ):
        raise PermissionError("Parent checkpoint state/field/reference identity differs")
    if not state["optimizer"]["state"] or any(
        int(x["step"]) != 96 for x in state["optimizer"]["state"].values()
    ):
        raise PermissionError("Parent Adam does not contain exactly 96 committed updates")
    if file_hash(path) != marker["sha256"]:
        raise PermissionError("Parent checkpoint mutated during read")
    return state


def audit_parent(root, parent_id, authorization, registration, model_identity, output_root):
    from safetensors.torch import load_file

    task_receipt = verify_parent_receipt(root, parent_id, authorization, registration)
    directory = bounded_file(root, f"runs/{parent_id}/COMPLETE.json").parent
    receipt_names = (
        "RUN_MANIFEST.json",
        "COMPLETE.json",
        "checkpoints/commit-96.json",
        "checkpoints/LATEST.json",
        "monitor_state/step-96.json",
    )
    receipt_hashes = {name: file_hash(bounded_file(directory, name)) for name in receipt_names}
    for artifact in task_receipt["artifacts"]:
        name = str(Path(artifact["path"]).relative_to(Path("runs") / parent_id))
        if receipt_hashes[name] != artifact["sha256"]:
            raise PermissionError("Parent receipt changed after release verification")
    done = read_json(directory / "COMPLETE.json")
    marker = read_json(bounded_file(directory, "checkpoints/commit-96.json"))
    latest = read_json(bounded_file(directory, "checkpoints/LATEST.json"))
    manifest = read_json(bounded_file(directory, "RUN_MANIFEST.json"))
    registered = [x for x in registration["main_matrix"] if x.get("model_id") == parent_id]
    if (
        len(registered) != 1
        or any(manifest.get(k) != v for k, v in registered[0].items())
        or manifest.get("technical_only") is not False
        or manifest.get("arm") != "J6"
        or manifest.get("config_sha256") != registration["config_sha256"]
        or manifest.get("source_manifest_sha256") != registration["source_manifest_sha256"]
        or done.get("status") != "COMPLETE"
        or done.get("model_id") != parent_id
        or done.get("step") != 96
        or done.get("technical_only") is not False
        or done.get("last_monitor_step") != 96
        or marker.get("step") != 96
        or done.get("checkpoint") != marker
        or latest != marker
    ):
        raise PermissionError(
            f"Parent COMPLETE/LATEST/commit/manifest binding differs: {parent_id}"
        )
    state = load_verified_checkpoint(directory / "checkpoints", marker, manifest)
    adapter_dir = bounded_file(directory, "adapters/step-96/IDENTITY.json").parent
    adapter_identity_sha256 = file_hash(adapter_dir / "IDENTITY.json")
    identity = read_json(adapter_dir / "IDENTITY.json")
    if (
        identity.get("step") != 96
        or identity.get("run_identity_sha256") != object_hash(manifest)
        or identity.get("policy_hash") != marker["field_hashes"]["parameters"]
        or set(identity.get("files", {})) != ADAPTER_FILES
    ):
        raise PermissionError("Published adapter is not the exact final parent policy")
    for name, checksum in identity["files"].items():
        verify_file(adapter_dir, name, checksum)
    cfg = read_json(adapter_dir / "adapter_config.json")
    base_path = Path(model_identity["model_path"]).resolve(strict=True)
    if Path(cfg["base_model_name_or_path"]).resolve(strict=True) != base_path:
        raise PermissionError("Adapter references another base snapshot")
    modules = manifest["training_modules"]["target_modules"]
    if sorted(modules) != expected_language_modules(
        read_json(base_path / "config.json")
    ) or manifest["training_modules"].get("target_module_count") != len(modules):
        raise PermissionError("Parent did not cover every native language linear module")
    inventory = verify_tensor_inventory(
        state["parameters"],
        load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu"),
        modules,
        cfg,
    )
    if set(manifest["training_modules"].get("trainable_names", [])) != set(state["parameters"]):
        raise PermissionError("Declared trainable tensor inventory differs")
    monitor = read_json(bounded_file(directory, "monitor_state/step-96.json"))
    if (
        monitor["identity"].get("checkpoint_sha256") != marker["sha256"]
        or monitor["identity"].get("checkpoint_state_hash") != marker["state_hash"]
        or monitor.get("measurement_hash") != object_hash(monitor["measurement"])
    ):
        raise PermissionError("Final monitor is not bound to the same parent checkpoint")
    destination = Path("parents") / parent_id
    copies = []
    for name in sorted(ADAPTER_FILES | {"IDENTITY.json"}):
        copies.append(
            copy_verified(
                adapter_dir / name,
                output_root,
                destination / "adapter" / name,
                adapter_identity_sha256 if name == "IDENTITY.json" else identity["files"][name],
            )
        )
    for relative in receipt_names:
        copies.append(
            copy_verified(
                directory / relative,
                output_root,
                destination / "source_receipts" / relative,
                receipt_hashes[relative],
            )
        )
    write_once(output_root, destination / "TENSOR_INVENTORY.json", inventory)
    return {
        "parent_id": parent_id,
        "adapter_path": str(destination / "adapter"),
        "policy_hash": identity["policy_hash"],
        "checkpoint_sha256": marker["sha256"],
        "checkpoint_state_hash": marker["state_hash"],
        "target_modules_full": sorted(modules),
        "adapter_config_sha256": identity["files"]["adapter_config.json"],
        "checkpoint_path": str(directory / "checkpoints" / marker["path"]),
        "checkpoint_field_hashes": marker["field_hashes"],
        "tensor_count": len(inventory),
        "parameter_count": sum(x["numel"] for x in inventory),
        "copied_files": copies,
        "optimizer_inherited": False,
        "source_root_read_only": True,
    }


def audit_environment(root, output_root, *, version_lookup=importlib.metadata.version):
    path = bounded_file(root, "MODEL_ENVIRONMENT_IDENTITY.json")
    inputs = read_json(bounded_file(root, "INPUT_IDENTITY.json"))
    bound_environment = [
        item for item in inputs["files"] if item["path"] == "MODEL_ENVIRONMENT_IDENTITY.json"
    ]
    if inputs.get("status") != "PASS" or len(bound_environment) != 1:
        raise PermissionError("Registered input identity does not bind the model environment")
    entry = bound_environment[0]
    verify_file(root, entry["path"], entry["sha256"], entry.get("bytes"))
    identity = read_json(path)
    if (
        identity.get("model_id") != MODEL_ID
        or identity.get("model_revision") != REVISION
        or identity.get("revision") != REVISION
        or identity.get("model_calls") != 0
    ):
        raise PermissionError("Historical base model identity differs")
    model = Path(identity["model_path"]).resolve(strict=True)
    if (
        identity["model_files"] != inputs["model_files"]
        or identity["model_weights_hash"] != inputs["model_weights_hash"]
        or Path(inputs["model_path"]).resolve(strict=True) != model
    ):
        raise PermissionError("Base snapshot differs from the registered input identity")
    destination = Path(output_root).resolve(strict=True)
    if destination.is_relative_to(model) or model.is_relative_to(destination):
        raise PermissionError("Output must be independent of the base snapshot")
    if model.name != REVISION:
        raise PermissionError("Base snapshot revision directory differs")
    files = identity["model_files"]
    if len(files) != 13 or len({x["name"] for x in files}) != 13:
        raise PermissionError("Base snapshot must contain the 13 frozen files")
    verified = []
    for item in files:
        # HF snapshots use symlinks to immutable blobs outside the snapshot.
        name = Path(item["name"])
        if name.is_absolute() or ".." in name.parts or len(name.parts) != 1:
            raise PermissionError("Unsafe base snapshot filename")
        target = (model / name).resolve(strict=True)
        hub_model_root = model.parent.parent
        if not target.is_relative_to(hub_model_root):
            raise PermissionError("Base snapshot symlink escapes its model cache")
        if file_hash(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise PermissionError(f"Base model bytes differ: {name}")
        verified.append({**item, "resolved_path": str(target)})
    actual_versions = {name: version_lookup(name) for name in identity["packages"]}
    if actual_versions != identity["packages"] or platform.python_version() != identity["python"]:
        raise PermissionError(f"Historical dependency versions differ: {actual_versions}")
    copy_verified(path, output_root, "prior/MODEL_ENVIRONMENT_IDENTITY.json", entry["sha256"])
    report = {
        "status": "PASS",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_path": str(model),
        "files": verified,
        "packages": actual_versions,
        "python": platform.python_version(),
        "model_calls": 0,
        "base_weights_loaded": False,
        "cuda_context_initialized": False,
    }
    write_once(output_root, "prior/BASE_ENVIRONMENT_AUDIT.json", report)
    return identity


def audit_prior_sources(root, plan_dir, registration, output_root):
    root = Path(root).resolve(strict=True)
    source_path = Path(registration["source_manifest"]).resolve(strict=True)
    code_root = Path(registration["code_root"]).resolve(strict=True)
    if not source_path.is_relative_to(root) or not code_root.is_relative_to(root):
        raise PermissionError("Registered source locations escape the parent root")
    manifest = read_json(source_path)
    if object_hash(manifest) != registration["source_manifest_sha256"]:
        raise PermissionError("Registered source manifest content hash differs")
    entries = manifest["files"]
    if isinstance(entries, dict):
        entries = [{"path": p, "sha256": h} for p, h in entries.items()]
    if not entries or len({x["path"] for x in entries}) != len(entries):
        raise PermissionError("Source inventory is absent or duplicates paths")
    copies = []
    for entry in entries:
        path = verify_file(code_root, entry["path"], entry["sha256"], entry.get("bytes"))
        copies.append(
            copy_verified(path, output_root, Path("prior/source") / entry["path"], entry["sha256"])
        )
    small = {
        "REGISTRATION.json",
        "config/SR_F2.json",
        "MODEL_ENVIRONMENT_IDENTITY.json",
        "released/AUTHORIZATION.json",
        "released/COMPLETE.json",
        "INPUT_IDENTITY.json",
        "DIAGNOSTIC_MANIFEST.json",
        "NATIVE_DIAGNOSTIC_PROMPTS.json",
    }
    small.update(registration["bound_files"])
    for relative in sorted(small):
        copies.append(
            copy_verified(bounded_file(root, relative), output_root, Path("prior") / relative)
        )
    copies.append(copy_verified(source_path, output_root, "prior/SOURCE_MANIFEST.json"))
    comparison = []
    for actual, packaged in (
        ("src/sr_f12/training.py", "HISTORICAL_training.py"),
        ("src/sr_f1/training.py", "HISTORICAL_parent_training.py"),
        ("src/sr_f12/protocol.py", "HISTORICAL_protocol.py"),
    ):
        left, right = bounded_file(code_root, actual), Path(plan_dir) / "sources" / packaged
        diff = "".join(
            difflib.unified_diff(
                right.read_text().splitlines(keepends=True),
                left.read_text().splitlines(keepends=True),
                fromfile=f"package/{packaged}",
                tofile=f"parent/{actual}",
            )
        )
        record = {
            "parent_path": actual,
            "package_path": f"sources/{packaged}",
            "parent_sha256": file_hash(left),
            "package_sha256": file_hash(right),
            "identical": not diff,
            "unified_diff": diff,
        }
        comparison.append(record)
    report = {
        "status": "PASS",
        "source_files_verified": len(entries),
        "copies": copies,
        "package_comparisons": comparison,
        "old_scores_modified": False,
    }
    write_once(output_root, "prior/SOURCE_AUDIT.json", report)
    return report


def prior_raw_status(root):
    """Inventory bounded old-only locations; absence cannot block the new science."""
    root = Path(root)
    endpoints = [
        f"SRF2_{arm}_s{seed}"
        for seed in (71001, 71002, 71003)
        for arm in ("EARLY", "J6", "DEC_PRIME", "ADAPT")
    ]
    return {
        "critical": False,
        "status": "INVENTORIED_NOT_EXPORTED",
        "model_calls": 0,
        "chartqa_endpoint_sources": [
            {
                "model_id": name,
                "completion_present": (
                    root / "sealed/evaluation" / name / "COMPLETE.json"
                ).is_file(),
            }
            for name in endpoints
        ],
        "diagnostic_completion_sources": [
            str(p.relative_to(root)) for p in sorted((root / "diagnostics").glob("*/COMPLETE.json"))
        ],
        "note": "FREE/LIST and V0-V4 raw export remains a noncritical P0 delivery task; "
        "inventory does not certify complete raw recovery.",
    }


def audit_parents(parent_root, plan_dir, output_root):
    parent_root, plan_dir = (
        Path(parent_root).resolve(strict=True),
        Path(plan_dir).resolve(strict=True),
    )
    output_root = Path(output_root).resolve()
    if (
        output_root.is_relative_to(parent_root)
        or parent_root.is_relative_to(output_root)
        or output_root.is_relative_to(plan_dir)
    ):
        raise PermissionError("Output must not overlap the parent or be inside the frozen plan")
    output_root.mkdir(parents=True, exist_ok=True)
    verify_plan(plan_dir)
    authorization, registration = verify_release(parent_root, plan_dir)
    sources = audit_prior_sources(parent_root, plan_dir, registration, output_root)
    environment = audit_environment(parent_root, output_root)
    parents = [
        audit_parent(parent_root, name, authorization, registration, environment, output_root)
        for name in PARENTS
    ]
    raw_status = prior_raw_status(parent_root)
    write_once(output_root, "prior/NONCRITICAL_RAW_STATUS.json", raw_status)
    report = {
        "status": "PASS",
        "stage": "P0_A_PARENT_IDENTITY",
        "parents": parents,
        "parent_root": str(parent_root),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_environment_identity": "prior/MODEL_ENVIRONMENT_IDENTITY.json",
        "package_manifest_sha256": file_hash(plan_dir / "PACKAGE_SHA256.json"),
        "authorization_sha256": file_hash(parent_root / "released/AUTHORIZATION.json"),
        "registration_object_hash": object_hash(registration),
        "source_files_verified": sources["source_files_verified"],
        "model_calls": 0,
        "base_weights_loaded": False,
        "cuda_context_initialized": False,
        "fresh_optimizer_required": True,
        "reference": "fixed_parent_per_block",
        "scientific_training_authorized_by_this_receipt": False,
        "remaining_gates": ["rendering_processor", "native_masks", "probabilities", "GPU_resume"],
    }
    write_once(output_root, "parents/PARENT_AUDIT.json", report)
    return report
