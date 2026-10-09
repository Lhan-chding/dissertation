"""CPU-only model/environment revalidation and immutable F2 freeze construction."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

from mm_core.execution import exclusive_json, object_hash, read_json, sha256_file, utc_now
from mm_core.vl_runtime import QwenRuntime, model_file_path

from .common import PLAN_SHA256, load_plan, verify_execution
from .contract import PLAN_ID
from .data import POOLS, load_history, validate_materialization


def validate_data_receipt(plan, root, data, historical_root):
    expected_counts = dict(root_families=292, numeric_sources=584, images=2336, questions=7008)
    if any(data.get(key) != value for key, value in expected_counts.items()):
        raise PermissionError("F2 materialization counts differ from registered scope")
    if data.get("root_counts") != POOLS or data.get("panel_question_counts") != {
        p: n * 24 for p, n in POOLS.items()
    }:
        raise PermissionError("F2 materialization pool counts changed")
    historical = load_history(Path(historical_root))
    saved_history = read_json(root / "data/history_exclusions.json")
    if object_hash(historical) != object_hash(saved_history):
        raise PermissionError("Historical exclusion evidence changed")
    if data.get("renderer") != historical["renderer"]:
        raise PermissionError("Materialized renderer/font differs from historical identity")
    for section, required in (
        ("source_audit", ("status", "historical_exclusion")),
        ("image_audit", ("status", "processed_image_audit", "historical_image_exclusion")),
    ):
        if any(data.get(section, {}).get(key) != "PASS" for key in required):
            raise PermissionError("Incomplete materialization audit: " + section)
    identity = data.get("processor_identity", {})
    for key in (
        "model_id",
        "processor_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
    ):
        if identity.get(key) != plan["model"][key]:
            raise PermissionError("Materialized processor identity mismatch: " + key)
    if identity.get("chat_template_kwargs") != {"enable_thinking": False} or (
        identity.get("processor_geometry")
        != {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}
    ):
        raise PermissionError("Materialized processor kwargs or patch geometry changed")
    expected_samples = {
        (pool, index, chart, visual, difficulty)
        for pool, count in POOLS.items()
        for index in (0, count - 1)
        for chart in ("grouped_bar", "line")
        for visual in ("low", "high")
        for difficulty in ("low", "high")
    }
    if set(data.get("qa", {})) != {"original", "processed"}:
        raise PermissionError("Both fixed QA modes are required")
    for mode, qa in data["qa"].items():
        samples = qa.get("samples", [])
        observed = {(x["pool"], x["root_index"], x["chart_type"], x["V"], x["D"]) for x in samples}
        expected_sheets = {f"data/qa/{mode}_{pool.lower()}_first_last.png" for pool in POOLS}
        if (
            qa.get("selected_image_count") != 80
            or len(samples) != 80
            or observed != expected_samples
            or len(qa.get("contact_sheets", [])) != 5
            or {x["path"] for x in qa["contact_sheets"]} != expected_sheets
        ):
            raise PermissionError("Fixed first/last-root visual QA coverage is incomplete")
        for sample in samples:
            image_id = (
                f"mmdev-f2-{sample['pool'].lower()}-r{sample['root_index']:04d}"
                f"-d{sample['D'][0]}-v{sample['V'][0]}-{sample['chart_type']}"
            )
            folder = "processed_images" if mode == "processed" else "images"
            if sample.get("image_id") != image_id or sample.get("image_path") != (
                f"data/{folder}/{image_id}.png"
            ):
                raise PermissionError("Fixed visual sample image identity differs")


def inspect_model(plan_path, historical_root):
    """Hash actual snapshot bytes and use the pinned CPU processor without model weights."""
    import torch

    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    original_path = Path(historical_root) / "manifests/PRE_INFERENCE_FREEZE.json"
    identity = read_json(original_path)
    expected_env = read_json(
        plan_path.parent.parent / "evidence/E06_PROCESSOR_ENVIRONMENT_LOCK.json"
    )
    expected_fields = (
        "model_id",
        "model_revision",
        "model_weights_hash",
        "processor_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
    )
    for field in expected_fields:
        if identity.get(field) != plan["model"][field]:
            raise PermissionError("Historical model identity conflict: " + field)
    actual_files = []
    for entry in identity["model_files"]:
        path = model_file_path(identity["model_path"], entry["name"])
        actual = dict(name=entry["name"], bytes=path.stat().st_size, sha256=sha256_file(path))
        if actual != entry:
            raise PermissionError("Actual model file differs: " + entry["name"])
        actual_files.append(actual)
    weights = {
        f["name"]: f["sha256"]
        for f in actual_files
        if f["name"].endswith(".safetensors")
        or f["name"] == "config.json"
        or f["name"].endswith("index.json")
    }
    if object_hash(weights) != plan["model"]["model_weights_hash"]:
        raise PermissionError("Actual model aggregate differs")
    packages = {}
    for name, expected in expected_env["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "NOT_INSTALLED"
        packages[name] = actual
        if actual != expected:
            raise PermissionError("Pinned package changed: " + name)
    if platform.python_version() != expected_env["python"]:
        raise PermissionError("Pinned Python version changed")
    runtime = QwenRuntime.processor_only(identity["model_path"])
    for name in (
        "processor_hash",
        "chat_template_hash",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
    ):
        if runtime.identity[name] != identity[name]:
            raise PermissionError("Current CPU processor/kernel identity differs: " + name)
    if torch.cuda.is_initialized():
        raise PermissionError("CPU identity inspection unexpectedly initialized CUDA")
    return dict(
        plan_id=PLAN_ID,
        status="PASS",
        checked_at=utc_now(),
        historical_freeze_sha256=sha256_file(original_path),
        model_identity=identity,
        actual_model_files=actual_files,
        packages=packages,
        python=platform.python_version(),
        runtime_processor_identity=runtime.identity,
        model_weights_loaded=False,
        model_calls=0,
        cuda_context_initialized=False,
    )


def freeze_run(plan_path, run_root, historical_root, *, code_commit):
    root = Path(run_root).resolve(strict=True)
    plan_path = Path(plan_path).resolve(strict=True)
    plan = load_plan(plan_path)
    destination = root / "manifests/F2_FREEZE.json"
    if destination.exists():
        raise FileExistsError("Preserve existing F2 freeze; validate it instead")
    validation = validate_materialization(plan_path, root)
    data_path = root / "manifests/DATA_MATERIALIZATION.json"
    data = read_json(data_path)
    validate_data_receipt(plan, root, data, historical_root)
    if (
        validation.get("status") != "PASS"
        or data.get("processor_status") != "ACTUAL_CPU_PROCESSOR_VERIFIED"
        or data.get("processed_image_count") != 2336
        or data.get("processed_question_count") != 7008
    ):
        raise PermissionError("Complete native CPU processor materialization is required")
    visual_path = root / "manifests/VISUAL_REVIEW.json"
    visual = read_json(visual_path)
    if (
        visual.get("status") != "PASS"
        or visual.get("data_materialization_sha256") != sha256_file(data_path)
        or visual.get("model_outcomes_available") is not False
    ):
        raise PermissionError("Frozen prereveal visual review receipt is required")
    expected_sheets = {
        x["path"]: x["sha256"] for mode in data["qa"].values() for x in mode["contact_sheets"]
    }
    if visual.get("reviewed_contact_sheets") != expected_sheets:
        raise PermissionError("All fixed original and processed QA sheets must be reviewed")
    for relative, expected in expected_sheets.items():
        if sha256_file(root / relative) != expected:
            raise PermissionError("Reviewed image sheet changed")
    permission_path = root / "manifests/ALLOCATION_PERMISSION.json"
    permission = read_json(permission_path)
    if (
        permission.get("plan_id") != PLAN_ID
        or permission.get("authorized") is not True
        or permission.get("run_root") != str(root)
        or permission.get("max_gpus_concurrent") != 5
        or permission.get("gpus_per_worker") != 1
        or permission.get("gres") != "gpu:pro6000:1"
    ):
        raise PermissionError("Current resource permission receipt is required before freeze")
    model = inspect_model(plan_path, historical_root)
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {}
    for directory in ("src/mm_core", "src/mm_dev", "scripts/mm_dev"):
        for path in sorted((source_root / directory).rglob("*.py")):
            if path.is_symlink():
                raise PermissionError("Source symlinks are forbidden")
            source_hashes[str(path.relative_to(source_root))] = sha256_file(path)
    if len(code_commit) != 40 or any(x not in "0123456789abcdef" for x in code_commit):
        raise ValueError("Full committed code identity is required")
    input_hashes = {}
    for directory in ("data", "qa"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                if path.is_symlink():
                    raise PermissionError("Input symlinks are forbidden")
                input_hashes[str(path.relative_to(root))] = sha256_file(path)
    input_hashes["manifests/DATA_MATERIALIZATION.json"] = sha256_file(data_path)
    input_hashes["manifests/VISUAL_REVIEW.json"] = sha256_file(visual_path)
    input_hashes["manifests/ALLOCATION_PERMISSION.json"] = sha256_file(permission_path)
    model_path = root / "manifests/MODEL_AND_ENVIRONMENT_IDENTITY.json"
    exclusive_json(model_path, model)
    input_hashes[str(model_path.relative_to(root))] = sha256_file(model_path)
    freeze = dict(
        plan_id=PLAN_ID,
        status="FROZEN",
        authorized=True,
        authorized_stage="MM-DEV",
        human_authorization="2026-10-09 user: 完整执行吧",
        run_root=str(root),
        plan_sha256=PLAN_SHA256,
        code_commit=code_commit,
        source_hashes=source_hashes,
        input_hashes=input_hashes,
        model_identity=model["model_identity"],
        model_path=model["model_identity"]["model_path"],
        resource_policy=plan["resource"],
        created_at=utc_now(),
        common_lora_binding="COMMON_LORA.json bound to this freeze SHA",
        common_initialization_seed=plan["common_start"]["zero_output_lora_initialization_seed"],
        expected_counts=plan["expected_counts"],
        scientific_configuration_fixed=True,
        inherited_old_resource_caps=False,
    )
    exclusive_json(destination, freeze)
    verify_execution(plan_path, root, full_hashes=True)
    return freeze
