"""Fail-closed S0/S1 identity gates, separate from GPU engine certification."""

from __future__ import annotations

import importlib.metadata
import platform
import re
from pathlib import Path

from mm_core.execution import atomic_json, object_hash, read_json, utc_now

from .contract import PACKAGE, PLAN_ID, file_hash, load_config
from .data import bounded_path, read_jsonl

TECHNICAL_REPAIR_ID = "SR_F1_FORMAT_GUARD_REPAIR_20261009"
TECHNICAL_REPAIR_ALLOWED_FILES = frozenset(
    {
        "src/sr_f1/freeze.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/format_review.py",
        "src/sr_f1/orchestration.py",
        "scripts/sr_f1/submit_matrix.py",
    }
)


def _required(value, expected, message):
    if value != expected:
        raise PermissionError(message)


def _checked_source_files(value, name):
    if not isinstance(value, dict) or not value:
        raise PermissionError(name + " source-file inventory is missing")
    for relative, expected in value.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or str(Path(relative)) != relative
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise PermissionError(name + " source-file inventory is invalid")
    return value


def _preserved_source_files(snapshot):
    """Apply the original source_identity inventory policy to the preserved copy."""
    files = {}
    for folder in ("src/sr_f1", "scripts/sr_f1", "src/mm_core", "src/mm_dev"):
        for path in sorted((snapshot / folder).rglob("*")):
            if path.is_symlink():
                raise PermissionError("Preserved original source contains unsafe links")
            if not path.is_file() or path.suffix not in (".py", ".sh", ".sbatch"):
                continue
            if not path.resolve().is_relative_to(snapshot):
                raise PermissionError("Preserved original source contains unsafe links")
            files[str(path.relative_to(snapshot))] = file_hash(path)
    for name in ("pyproject.toml", "uv.lock"):
        path = snapshot / name
        if path.is_file():
            if path.is_symlink():
                raise PermissionError("Preserved original dependency identity is a symlink")
            files[name] = file_hash(path)
    return files


def verify_technical_repair(root, actual_source=None):
    """Authenticate the one scoped source amendment without rewriting the freeze.

    The original model/data/processor gates remain in verify_execution. This
    additional source gate stays valid after the one-time resume has progressed;
    the scheduler separately enforces the before-bridge/science resume boundary.
    """
    from .prepare import source_identity, verify_package

    root = Path(root).resolve(strict=True)
    verify_package()
    freeze_path = root / "EXECUTION_FREEZE.json"
    source_path = root / "SOURCE_AND_RENDER_MANIFEST.json"
    repair_path = root / "TECHNICAL_REPAIR.json"
    freeze, original, repair = map(read_json, (freeze_path, source_path, repair_path))
    _required(freeze.get("plan_id"), PLAN_ID, "Technical repair has the wrong frozen plan")
    _required(freeze.get("status"), "FROZEN", "Technical repair requires the original freeze")
    _required(freeze.get("run_root"), str(root), "Technical repair root differs from the freeze")
    for key, expected in {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "repair_id": TECHNICAL_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_REPAIR",
        "original_freeze_sha256": file_hash(freeze_path),
        "original_source_manifest_sha256": file_hash(source_path),
        "original_source_tree_sha256": freeze.get("source_tree_sha256"),
        "original_source_commit": freeze.get("source_commit"),
        "preserved_source_root": "code_before_format_guard_repair",
        "scientific_contract_changes": [],
        "source_dirty_files": [],
    }.items():
        _required(repair.get(key), expected, "Technical repair identity differs: " + key)
    if not isinstance(repair.get("reason"), str) or not repair["reason"].strip():
        raise PermissionError("Technical repair requires an explicit implementation-only reason")
    if not re.fullmatch(r"[0-9a-f]{40}", freeze.get("source_commit") or ""):
        raise PermissionError("Technical repair lacks the original committed source identity")
    for relative, expected in freeze.get("artifact_hashes", {}).items():
        _required(
            file_hash(bounded_path(root, relative)),
            expected,
            "Technical repair changed an original frozen artifact: " + relative,
        )
    _required(
        freeze.get("artifact_hashes", {}).get("SOURCE_AND_RENDER_MANIFEST.json"),
        file_hash(source_path),
        "Original source manifest is not bound to the execution freeze",
    )
    before = _checked_source_files(original.get("source_file_hashes"), "Original")
    for candidate in (freeze.get("source_file_hashes"), repair.get("original_source_file_hashes")):
        _required(candidate, before, "Technical repair substituted the original file inventory")
    for candidate in (original.get("source_tree_sha256"), freeze.get("source_tree_sha256")):
        _required(candidate, object_hash(before), "Original source aggregate is inconsistent")
    for relative, expected in original["package_input_hashes"].items():
        _required(
            file_hash(bounded_path(root, relative)),
            expected,
            "Technical repair changed a registered input: " + relative,
        )
        _required(
            file_hash(bounded_path(PACKAGE, relative)),
            expected,
            "Technical repair substituted the uploaded input contract: " + relative,
        )
    declared_snapshot = root / repair["preserved_source_root"]
    snapshot = bounded_path(root, repair["preserved_source_root"])
    if not snapshot.is_dir() or declared_snapshot.is_symlink():
        raise PermissionError("Preserve the actual original source directory before repair")
    _required(
        _preserved_source_files(snapshot),
        before,
        "Preserved source differs from the original freeze",
    )
    actual = source_identity() if actual_source is None else actual_source
    after = _checked_source_files(actual.get("source_file_hashes"), "Actual repaired")
    _required(
        actual.get("source_tree_sha256"), object_hash(after), "Repaired source hash is invalid"
    )
    _required(actual.get("source_dirty_files"), [], "Commit the technical repair before deployment")
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit") or ""):
        raise PermissionError("Technical repair requires its actual full clean commit identity")
    if actual["source_commit"] == freeze["source_commit"]:
        raise PermissionError("Changed technical source cannot retain the original commit identity")
    for key in ("source_commit", "source_tree_sha256", "source_file_hashes"):
        _required(repair.get(key), actual[key], "Actual repaired source differs: " + key)
    if set(before) - set(after):
        raise PermissionError("Technical repair cannot remove frozen source files")
    changed = [
        {"path": name, "before_sha256": before.get(name), "after_sha256": after[name]}
        for name in sorted(after)
        if before.get(name) != after[name]
    ]
    if not changed or any(row["path"] not in TECHNICAL_REPAIR_ALLOWED_FILES for row in changed):
        raise PermissionError("Technical repair exceeds the explicit source-file allowlist")
    _required(repair.get("changed_files"), changed, "Technical repair file changes are not exact")
    review_path = root / "FORMAT_TECHNICAL_REVIEW.json"
    _required(
        repair.get("format_review_sha256"),
        file_hash(review_path),
        "Technical repair references a different FORMAT technical review",
    )
    review = read_json(review_path)
    for key, expected in {
        "plan_id": PLAN_ID,
        "freeze_sha256": file_hash(freeze_path),
        "status": "VERIFIED_PROTOCOL_NONADHERENCE",
    }.items():
        _required(review.get(key), expected, "FORMAT technical review identity differs: " + key)
    # Imported only after the repair module's bytes, original freeze and review
    # identity have been checked; no circular module import or scientific state read.
    from .format_review import verify_format_review

    verify_format_review(root)
    return {
        "status": "VERIFIED_TECHNICAL_REPAIR",
        "repair_id": TECHNICAL_REPAIR_ID,
        "repair_sha256": file_hash(repair_path),
        "review_sha256": file_hash(review_path),
        "original_freeze_sha256": file_hash(freeze_path),
        "source_commit": actual["source_commit"],
        "source_tree_sha256": actual["source_tree_sha256"],
        "changed_files": changed,
    }


def _verify_execution_source(root, original_source, actual_source=None):
    from .prepare import source_identity

    actual = source_identity() if actual_source is None else actual_source
    files = _checked_source_files(actual.get("source_file_hashes"), "Actual")
    _required(actual.get("source_tree_sha256"), object_hash(files), "Actual source hash is invalid")
    if original_source.get("source_tree_sha256") == actual.get("source_tree_sha256"):
        _required(original_source.get("source_file_hashes"), files, "Original source files differ")
        return None
    return verify_technical_repair(root, actual_source=actual)


def verify_engine_receipt(root):
    root = Path(root).resolve(strict=True)
    engine = read_json(root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json")
    _required(engine.get("plan_id"), PLAN_ID, "ENGINE plan differs")
    _required(
        engine.get("freeze_sha256"),
        file_hash(root / "EXECUTION_FREEZE.json"),
        "ENGINE belongs to a different execution freeze",
    )
    _required(
        engine.get("common_start_sha256"),
        file_hash(root / "COMMON_START.json"),
        "ENGINE belongs to a different common start",
    )
    common = read_json(root / "COMMON_START.json")
    for key, expected in dict(
        plan_id=PLAN_ID,
        status="VERIFIED",
        model_id="SRF1_COMMON_START",
        freeze_sha256=engine["freeze_sha256"],
    ).items():
        _required(common.get(key), expected, "ENGINE common start is not verified")
    if not common.get("adapter_file_hashes"):
        raise PermissionError("ENGINE common start lacks actual adapter files")
    adapter = bounded_path(root, common["adapter_path"])
    for relative, expected in common["adapter_file_hashes"].items():
        _required(
            file_hash(bounded_path(adapter, relative)), expected, "Common adapter bytes changed"
        )
    status = engine.get("status")
    if status not in ("PASS_NATURAL_GRPO_RESUME", "PASS_NONZERO_SURROGATE_KERNEL_RESUME"):
        raise PermissionError("GPU probability/gradient/resume certification missing")
    natural, stress = engine.get("natural"), engine.get("stress")
    if not isinstance(natural, dict):
        raise PermissionError("Actual natural ENGINE trace is missing")
    expected_natural = (
        "EXACT_NONZERO_TRACE"
        if status == "PASS_NATURAL_GRPO_RESUME"
        else "EXACT_CONSTANT_REWARD_TRACE"
    )
    _required(natural.get("status"), expected_natural, "Natural ENGINE result differs")
    if status == "PASS_NATURAL_GRPO_RESUME" and stress is not None:
        raise PermissionError("Unexpected surrogate trace for nonconstant natural ENGINE")
    if status == "PASS_NONZERO_SURROGATE_KERNEL_RESUME":
        if not natural.get("stress_allowed") or not isinstance(stress, dict):
            raise PermissionError("Missing permitted nonzero surrogate ENGINE qualification")
        _required(stress.get("status"), "EXACT_NONZERO_TRACE", "Surrogate ENGINE gradient failed")
    artifacts = engine.get("artifact_hashes", {})
    if not artifacts:
        raise PermissionError("ENGINE is missing its actual trace artifact identities")
    for mode, trace in (("natural", natural), ("stress", stress)):
        if trace is None:
            continue
        for key, expected in dict(
            plan_id=PLAN_ID,
            mode=mode,
            logical_steps=4,
            valid_trace_updates=8,
            valid_trace_completions=1024,
            exact_state_and_rng=True,
            exact_token_and_slot_paths=True,
            all_probability_gates_passed=True,
        ).items():
            _required(trace.get(key), expected, "Incomplete ENGINE evidence: " + key)
        if trace.get("physical_updates", 0) < 8 or trace.get("completion_attempts", 0) < 1024:
            raise PermissionError("ENGINE physical attempts omit the valid trace cost")
        if not trace.get("artifact_hashes"):
            raise PermissionError("ENGINE trace has no underlying evidence")
        comparison = f"engineering/engine/{mode}/COMPARISON.json"
        if (
            artifacts.get(comparison) != file_hash(root / comparison)
            or read_json(root / comparison) != trace
        ):
            raise PermissionError("ENGINE comparison receipt does not match its trace")
        for relative, expected in trace["artifact_hashes"].items():
            _required(artifacts.get(relative), expected, "ENGINE omitted underlying trace evidence")
    for relative, expected in artifacts.items():
        _required(
            file_hash(bounded_path(root, relative)), expected, "ENGINE trace artifact changed"
        )
    return engine


def verify_execution(plan, root, require_engine=False):
    """Verify actual artifacts before GPU work, not just a self-reported PASS."""
    from .prepare import BOLD_HASH, FONT_HASH, verify_package

    root = Path(root).resolve(strict=True)
    if isinstance(plan, (str, Path)):
        plan = read_json(plan)
    _required(plan, load_config(), "Execution plan differs from authenticated contract")
    verify_package()
    freeze = read_json(root / "EXECUTION_FREEZE.json")
    _required(freeze.get("plan_id"), PLAN_ID, "Wrong execution freeze plan")
    _required(freeze.get("status"), "FROZEN", "CPU preparation has not been frozen")
    _required(freeze.get("run_root"), str(root), "Execution root identity differs")
    _required(
        freeze.get("allocation_permission_sha256"),
        file_hash(root / "manifests/ALLOCATION_PERMISSION.json"),
        "Allocation permission differs from the execution freeze",
    )
    for key in (
        "formal_test_results_seen_before_freeze",
        "inherited_or_prior_run_results_used_to_select_parameters",
    ):
        _required(freeze.get(key), False, "Prereveal declaration is missing or invalid")
    _required(freeze.get("changes_from_uploaded_contract"), [], "Unregistered scientific changes")
    for relative, expected in freeze.get("artifact_hashes", {}).items():
        if file_hash(bounded_path(root, relative)) != expected:
            raise PermissionError("Frozen artifact changed: " + relative)
    required = {
        "MODEL_ENVIRONMENT_IDENTITY.json",
        "SOURCE_AND_RENDER_MANIFEST.json",
        "RENDER_RECEIPT.json",
        "PROCESSOR_PREFLIGHT.json",
        "PROCESSOR_INPUT_ROUTES.jsonl",
        "PROCESSED_IMAGES.jsonl",
    }
    if not required.issubset(freeze.get("artifact_hashes", {})):
        raise PermissionError("Freeze lacks the complete CPU evidence chain")
    source = read_json(root / "SOURCE_AND_RENDER_MANIFEST.json")
    _required(source.get("status"), "CPU_VERIFIED", "Source/render preparation incomplete")
    _verify_execution_source(root, source)
    for relative, expected in source["package_input_hashes"].items():
        if file_hash(bounded_path(root, relative)) != expected:
            raise PermissionError("Registered copied input changed: " + relative)
        if file_hash(bounded_path(PACKAGE, relative)) != expected:
            raise PermissionError("Copied input differs from authenticated package")
    render = read_json(root / "RENDER_RECEIPT.json")
    from PIL import ImageFont
    from PIL import __version__ as pillow_version

    _required(
        render.get("renderer_environment"),
        {
            "Pillow": pillow_version,
            "FreeType": ImageFont.core.freetype2_version,
            "renderer_sha256": file_hash(PACKAGE / "reference/render_charts.py"),
        },
        "Actual renderer/Pillow/FreeType environment changed",
    )
    _required(render.get("chart_count"), 2890, "Incomplete chart materialization")
    _required(render.get("image_count"), 2891, "Incomplete blank/chart materialization")
    for key, expected in (("normal", FONT_HASH), ("bold", BOLD_HASH)):
        entry = render["fonts"][key]
        _required(entry.get("sha256"), expected, "Wrong renderer font")
        _required(file_hash(entry["path"]), expected, "Renderer font bytes changed")
    images = {row["file"]: row for row in render["images"]}
    if len(images) != 2891:
        raise PermissionError("Missing or duplicated rendered image identity")
    for relative, row in images.items():
        _required(file_hash(bounded_path(root, relative)), row["sha256"], "Rendered image changed")
    processor = read_json(root / "PROCESSOR_PREFLIGHT.json")
    _required(processor.get("status"), "PASS", "Actual CPU processor preflight did not pass")
    _required(
        processor.get("processor_status"),
        "ACTUAL_CPU_PROCESSOR_VERIFIED",
        "Processor evidence is not native execution",
    )
    _required(processor.get("question_count"), 4800, "Incomplete processor question coverage")
    _required(processor.get("chart_image_count"), 2890, "Incomplete processor image coverage")
    _required(processor.get("blank_image_verified"), True, "Blank visual channel was not checked")
    _required(
        processor.get("no_crop_pixel_reconstruction_verified"),
        True,
        "Native pixel reconstruction was not verified",
    )
    checks = processor.get("isolation", {}).get("checks", {})
    if len(checks) != 7 or not all(value is True for value in checks.values()):
        raise PermissionError("Model-input isolation controls are incomplete")
    if processor.get("readability", {}).get("minimum_label_height_pixels", 0) < 9:
        raise PermissionError("Native label readability failed")
    routes = read_jsonl(root / "PROCESSOR_INPUT_ROUTES.jsonl")
    processed = read_jsonl(root / "PROCESSED_IMAGES.jsonl")
    _required(len(routes), 4800, "Input route records are incomplete")
    _required(len({r["qid"] for r in routes}), 4800, "Duplicate processor questions")
    _required(len(processed), 2890, "Processed image records are incomplete")
    for row in routes:
        if (
            row.get("processed_size") != [1024, 768]
            or row.get("pixels_per_count", 0) < 3
            or row.get("image_token_count", 0) <= 0
            or not row.get("input_tensor_hash")
            or row.get("source_image_sha256") != images[row["image_file"]]["sha256"]
        ):
            raise PermissionError("Processor route violates image identity or readability")
    for row in processed:
        if (
            row.get("processed_size") != [1024, 768]
            or row.get("pixels_per_count", 0) < 3
            or row.get("minimum_label_height_pixels", 0) < 9
            or row.get("max_inverse_rgb_error", 999) > 1
        ):
            raise PermissionError("Processed image readability evidence failed")
    for row in processor["qa_samples"] + processor["qa_contact_sheets"]:
        _required(file_hash(bounded_path(root, row["path"])), row["sha256"], "QA artifact changed")
    expected_qa = {
        (r["pool"], r["family"], r["chart"], r["style"], f"{r['series_count']}series")
        for r in render["images"]
        if "pool" in r
    }
    if not expected_qa or {tuple(r["key"]) for r in processor["qa_samples"]} != expected_qa:
        raise PermissionError(
            "Fixed pool/family/chart/style/series visual QA coverage is incomplete"
        )
    if not processor["qa_contact_sheets"]:
        raise PermissionError("Actual native processor QA contact sheets are missing")
    identity = read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")
    _required(freeze.get("model_identity"), identity, "Frozen model identity evidence changed")
    _required(identity.get("status"), "ACTUAL_CPU_VERIFIED", "Actual model inspection missing")
    _required(identity.get("model_id"), plan["model"]["id"], "Model family/size differs")
    _required(identity.get("model_revision"), plan["model"]["revision"], "Model revision differs")
    _required(
        identity.get("model_weights_hash"),
        plan["model"]["prior_composite_weight_hash"],
        "Frozen model aggregate differs",
    )
    _required(identity.get("python"), platform.python_version(), "Python environment changed")
    for name, version in identity.get("packages", {}).items():
        _required(importlib.metadata.version(name), version, "Runtime package changed: " + name)
    from mm_core.vl_runtime import model_file_path

    weights = {}
    for row in identity.get("model_files", []):
        path = model_file_path(identity["model_path"], row["name"])
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise PermissionError("Actual model/processor bytes changed: " + row["name"])
        if row["name"].endswith((".safetensors", "index.json")) or row["name"] == "config.json":
            weights[row["name"]] = row["sha256"]
    _required(
        object_hash(weights), identity["model_weights_hash"], "Actual model aggregate differs"
    )
    for key in (
        "processor_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
        "generation_config_expanded",
    ):
        if not identity.get(key) or identity[key] != processor["processor_identity"].get(key):
            raise PermissionError("Processor/model environment identity conflict: " + key)
    _required(
        identity.get("chat_template_kwargs"), {"enable_thinking": False}, "Thinking mode changed"
    )
    _required(identity.get("canvas_pixels"), [1024, 768], "Synthetic canvas identity changed")
    _required(identity.get("processor_target_pixels"), 786432, "Native pixel budget changed")
    if require_engine:
        verify_engine_receipt(root)
    return freeze


def freeze_execution(root, *, operator):
    from .prepare import source_identity

    root = Path(root).resolve(strict=True)
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError("Actual execution operator is required")
    destination = root / "EXECUTION_FREEZE.json"
    if destination.exists() and read_json(destination).get("status") == "FROZEN":
        raise FileExistsError("Preserve existing execution freeze")
    source = source_identity()
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("source_commit") or ""):
        raise PermissionError("A full actual source commit/deployment receipt is required")
    if source.get("source_dirty_files"):
        raise PermissionError("Commit the scoped source before the execution freeze")
    receipt = read_json(root / "SOURCE_AND_RENDER_MANIFEST.json")
    if source["source_tree_sha256"] != receipt["source_tree_sha256"]:
        raise PermissionError("Source changed during CPU preparation")
    names = (
        "MODEL_ENVIRONMENT_IDENTITY.json",
        "SOURCE_AND_RENDER_MANIFEST.json",
        "RENDER_RECEIPT.json",
        "PROCESSOR_PREFLIGHT.json",
        "PROCESSOR_INPUT_ROUTES.jsonl",
        "PROCESSED_IMAGES.jsonl",
    )
    identity = read_json(root / names[0])
    external = root / "EXTERNAL_DATASET_AVAILABILITY.json"
    if not external.exists():
        external = root / "external/chartqa/IDENTITY.json"
    allocation = root / "manifests/ALLOCATION_PERMISSION.json"
    if not allocation.exists():
        raise PermissionError("Current allocation permission receipt is required before freezing")
    permission = read_json(allocation)
    for key, value in dict(
        plan_id=PLAN_ID,
        authorized=True,
        run_root=str(root),
        max_gpus_concurrent=5,
        gpus_per_worker=1,
    ).items():
        _required(permission.get(key), value, "Allocation permission is missing or differs")
    inventory = root / "manifests/ACTIVE_PROJECT_JOBS.json"
    payload = dict(
        plan_id=PLAN_ID,
        experiment=PLAN_ID,
        status="FROZEN",
        run_root=str(root),
        server_utc_time=utc_now(),
        operator=operator,
        plan_sha256=file_hash(PACKAGE / "CODEX_NEXT_PLAN_SR_F1_zh.md"),
        config_sha256=file_hash(PACKAGE / "config/SR_F1.json"),
        **source,
        artifact_hashes={name: file_hash(root / name) for name in names},
        allocation_permission_sha256=file_hash(allocation),
        model_revision_verified=identity["model_revision"],
        model_identity=identity,
        model_file_hashes=identity["model_files"],
        model_path=identity["model_path"],
        processor_and_template_hashes={
            k: identity[k] for k in ("processor_hash", "tokenizer_hash", "chat_template_hash")
        },
        renderer_font_hash=read_json(root / "RENDER_RECEIPT.json")["fonts"],
        rendered_dataset_manifest="RENDER_RECEIPT.json",
        runtime_environment=identity["packages"],
        adapter_initialisation_sha256=None,
        common_start_identity=None,
        bridge_used=None,
        common_start_status="PENDING_GPU_FORMAT_AND_ENGINE",
        formal_test_results_seen_before_freeze=False,
        inherited_or_prior_run_results_used_to_select_parameters=False,
        active_F2_jobs_read_only_inventory=read_json(inventory) if inventory.exists() else [],
        total_authorised_project_workers=5,
        research_gpu_hours_limit=None,
        external_dataset_availability_and_revision=read_json(external)
        if external.exists()
        else {"status": "PENDING", "main_experiment_may_continue": True},
        changes_from_uploaded_contract=[],
    )
    # Validate against a candidate, deleting only this unaccepted candidate on error.
    atomic_json(destination, payload)
    try:
        return verify_execution(load_config(), root)
    except Exception:
        payload["status"] = "CPU_FREEZE_REJECTED"
        atomic_json(destination, payload)
        raise
