"""Native processor and teacher-token audits without loading model weights."""

from __future__ import annotations

import importlib
from pathlib import Path

from mm_core.training import state_hash
from sr_f12.evaluation import durable_json

from .contract import support_target
from .protocol import digest, file_hash, read_json

CANVAS = (1024, 768)
TARGET_PIXELS = 786432
SCIENCE_SUPPORT_EXAMPLES = 256
ENGINE_SUPPORT_EXAMPLES = 16


def direct_source_identity():
    """Bind loaded F3 modules; inherited helpers are bound by the outer snapshot."""
    result = {}
    for name in ("runtime", "masks", "data", "contract", "prompts", "native_audit"):
        module = importlib.import_module("sr_f3_ce." + name)
        path = Path(module.__file__).resolve(strict=True)
        result["src/sr_f3_ce/" + name + ".py"] = dict(
            sha256=file_hash(path), bytes=path.stat().st_size, loaded_from=str(path)
        )
    return result


def processor_pixel_budget(runtime):
    """Inspect effective native configuration, including old/new Qwen aliases."""
    processor = runtime.processor.image_processor
    config = processor.to_dict()
    size = config.get("size", {})
    observations = {}
    for side, aliases in (
        ("min", ("min_pixels", "shortest_edge")),
        ("max", ("max_pixels", "longest_edge")),
    ):
        values = []
        for key in aliases:
            for source, mapping in (("config", config), ("size", size)):
                if isinstance(mapping, dict) and mapping.get(key) is not None:
                    values.append({"source": source + "." + key, "value": mapping[key]})
        value = getattr(processor, side + "_pixels", None)
        if value is not None:
            values.append({"source": "attribute." + side + "_pixels", "value": value})
        if not values or any(item["value"] != TARGET_PIXELS for item in values):
            raise PermissionError("Native processor does not fix " + side + " pixels to 786432")
        observations[side] = values
    if runtime.identity.get("processor_target_pixels") != TARGET_PIXELS:
        raise PermissionError("Runtime identity does not match native pixel budget")
    return dict(
        status="PASS",
        canvas=list(CANVAS),
        target_pixels=TARGET_PIXELS,
        effective_min_pixels=TARGET_PIXELS,
        effective_max_pixels=TARGET_PIXELS,
        configuration_observations=observations,
        image_processor_config_sha256=digest(config),
    )


def audit_sidecar_independence(runtime, dataset, store, qid):
    """Poison the actual audit row, observe real tensors, then restore that row.

    BatchFeature is a UserDict, not a dict: convert it before state_hash so the
    hash covers complete tensor bytes instead of its potentially truncated repr.
    Capture the first hash before another processor invocation can reuse memory.
    """
    row = dataset.tasks[qid]
    original = dict(row)
    before_sidecar_hash = digest(row)
    safe_before = dict(store.model_input(qid))
    tensors = runtime.prepare_input(safe_before, store.image_root)["inputs"]
    before_tensor_hash = state_hash(dict(tensors))
    poison = dict(
        qid="POISON_QID",
        answer="POISON_GOLD",
        certificate="POISON_CERTIFICATE",
        pool="POISON_POOL",
        root_id="POISON_ROOT",
        world={"POISON_WORLD": True},
        query={"POISON_QUERY": True},
        teacher={"POISON_TEACHER": True},
        question="POISON_QUESTION",
        prompt="POISON_AUDIT_PROMPT",
    )
    try:
        row.update(poison)
        poisoned_sidecar_hash = digest(dataset.tasks[qid])
        if poisoned_sidecar_hash == before_sidecar_hash:
            raise RuntimeError("Sidecar negative test did not alter the real audit row")
        safe_after = dict(store.model_input(qid))
        tensors = runtime.prepare_input(safe_after, store.image_root)["inputs"]
        after_tensor_hash = state_hash(dict(tensors))
        if safe_before != safe_after or before_tensor_hash != after_tensor_hash:
            raise PermissionError("Audit sidecar changed native model inputs")
    finally:
        row.clear()
        row.update(original)
        if dataset.tasks[qid] is not row or digest(row) != before_sidecar_hash:
            raise RuntimeError("Sidecar negative test failed to restore the actual audit row")
    return dict(
        status="PASS",
        qid=qid,
        poisoned_fields=sorted(poison),
        original_sidecar_sha256=before_sidecar_hash,
        poisoned_sidecar_sha256=poisoned_sidecar_hash,
        restored_sidecar_sha256=digest(row),
        input_tensor_sha256_before=before_tensor_hash,
        input_tensor_sha256_after=after_tensor_hash,
        model_input_sha256=digest(safe_before),
        tensor_hash_method="state_hash_dict_of_complete_native_tensors",
        actual_sidecar_mutated=True,
        actual_sidecar_restored=True,
    )


def audit_native_inputs(root):
    import torch

    from .runner import dataset_and_store, engine_schedule
    from .runtime import SRF3CERuntime

    root = Path(root).resolve(strict=True)
    dataset, store = dataset_and_store(root)
    prior = read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    runtime = SRF3CERuntime.processor_only(prior["model_path"])
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit must not initialize CUDA")
    for key in ("processor_hash", "tokenizer_hash", "chat_template_hash"):
        if runtime.identity[key] != prior[key]:
            raise PermissionError(f"Native processor identity differs: {key}")
    pixel_budget = processor_pixel_budget(runtime)
    render = read_json(root / "data/RENDER_AUDIT.json")
    selection = read_json(root / "data/SEMANTIC_RENDER_SPOTCHECK.json")
    checked = []
    patch = runtime.processor.image_processor.patch_size
    for selected in selection["items"]:
        qid = selected["qid"]
        for protocol in ("structured", "natural_work", "answer_only"):
            safe = store.model_input(qid, protocol)
            prepared = runtime.prepare_input(safe, store.image_root, protocol=protocol)
            grid = prepared["inputs"]["image_grid_thw"].cpu().tolist()
            if len(grid) != 1:
                raise PermissionError("One native image per task required")
            frames, height, width = grid[0]
            actual_size = [width * patch, height * patch]
            if (
                frames != 1
                or actual_size != list(CANVAS)
                or prepared["routing"].get("source_size") != list(CANVAS)
                or prepared["routing"].get("processed_size") != actual_size
                or prepared["routing"].get("processor_target_pixels") != TARGET_PIXELS
            ):
                raise PermissionError("Native image route changed the fixed 1024x768 canvas")
            scale_y, scale_x = height * patch / 768, width * patch / 1024
            unit_spacing = render["pixels_per_value_unit"] * scale_y
            label_height = render["minimum_major_label_ink_height_px"] * min(scale_x, scale_y)
            if unit_spacing < 3 or label_height < 9:
                raise PermissionError("Native resized image violates registered readability")
            checked.append(
                dict(
                    qid=qid,
                    protocol=protocol,
                    grid=grid,
                    actual_pixel_size=actual_size,
                    effective_unit_spacing_px=unit_spacing,
                    effective_major_label_height_px=label_height,
                    input_tensor_hash=digest(
                        {k: state_hash(v) for k, v in prepared["inputs"].items()}
                    ),
                    routing=prepared["routing"],
                )
            )
    if not selection["items"]:
        raise PermissionError("No frozen native image spotchecks")
    # Mutate the real audit sidecar while the independent input route is exercised.
    qid = selection["items"][0]["qid"]
    sidecar_audit = audit_sidecar_independence(runtime, dataset, store, qid)
    cases = []
    support_qids = sorted(q for q, t in dataset.tasks.items() if t["pool"] == "SUPPORT")
    engine_qids = [q for item in engine_schedule(dataset.tasks) for q in item["support_qids"]]
    if (
        len(support_qids) != SCIENCE_SUPPORT_EXAMPLES
        or len(engine_qids) != ENGINE_SUPPORT_EXAMPLES
        or len(set(engine_qids)) != len(engine_qids)
        or set(support_qids).intersection(engine_qids)
    ):
        raise PermissionError(
            "Registered native support audit must cover 256 SUPPORT + 16 ENGINE tasks"
        )
    for qid in support_qids + engine_qids:
        task = dataset.tasks[qid]
        for arm in ("GOLD", "PREFIX", "EXEC"):
            target = (
                task["teacher"][arm]
                if task["teacher"]
                else support_target(task["world"], task["query"], arm != "GOLD")
            )
            cases.append(dict(qid=qid, arm=arm, target=target, model_input=store.model_input(qid)))
    serialization = runtime.freeze_support_serialization(cases, store.image_root)
    serialized = serialization["serialization"]
    pair = {}
    audit_files = []
    for case in cases:
        record = runtime.prepare_support(
            case["model_input"],
            store.image_root,
            case["target"],
            case["arm"],
            serialization=serialized,
        )
        metadata = {
            k: record[k]
            for k in (
                "tokens",
                "answer_mask",
                "work_mask",
                "mask_audit",
                "serialized_teacher_target",
            )
        }
        metadata.update(
            qid=case["qid"],
            arm=case["arm"],
            model_input=case["model_input"],
            original_target=case["target"],
        )
        path = root / "native/support_masks" / f"{case['qid']}-{case['arm']}.json"
        durable_json(path, metadata)
        audit_files.append(dict(path=str(path.relative_to(root)), sha256=file_hash(path)))
        if case["arm"] == "PREFIX":
            pair[case["qid"]] = metadata
        elif case["arm"] == "EXEC":
            prefix = pair[case["qid"]]
            if any(
                prefix[key] != metadata[key]
                for key in ("tokens", "answer_mask", "work_mask", "serialized_teacher_target")
            ):
                raise PermissionError("PREFIX/EXEC teacher token or answer-mask identity differs")
    mask_receipt = dict(
        serialization,
        status="PASS",
        science_support_examples=len(support_qids),
        engineering_support_examples=len(engine_qids),
        target_cases=len(cases),
        prefix_exec_pairs=len(pair),
        files=audit_files,
        full_native_offsets=True,
        generated_model_outputs=0,
        native_processor_sha256=runtime.identity["processor_hash"],
    )
    durable_json(root / "native/SUPPORT_MASK_RECEIPT.json", mask_receipt)
    sources = direct_source_identity()
    receipt = dict(
        status="PASS",
        model_calls=0,
        weights_loaded=False,
        cuda_initialized=torch.cuda.is_initialized(),
        native_processor_identity=runtime.identity,
        checked_images=len(selection["items"]),
        protocol_routes=len(checked),
        checks=checked,
        input_sidecar_negative_test="PASS",
        input_tensors_identical=True,
        input_sidecar_audit=sidecar_audit,
        native_pixel_budget=pixel_budget,
        direct_source_files=sources,
        direct_source_sha256={name: item["sha256"] for name, item in sources.items()},
        prefix_views="observed_full_trajectory_only_no_counterfactual_budget_claim",
        image_manifest_sha256=file_hash(root / "data/IMAGE_MANIFEST.jsonl"),
        mask_receipt_sha256=file_hash(root / "native/SUPPORT_MASK_RECEIPT.json"),
    )
    durable_json(root / "native/PROCESSOR_RECEIPT.json", receipt)
    return receipt
