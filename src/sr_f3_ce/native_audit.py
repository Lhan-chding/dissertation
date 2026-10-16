"""Native processor and teacher-token audits without loading model weights."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from mm_core.training import state_hash
from mm_core.vl_runtime import hash_json
from sr_f12.evaluation import durable_json

from .contract import support_target
from .protocol import PARENTS, digest, file_hash, read_json

CANVAS = (1024, 768)
TARGET_PIXELS = 786432
SCIENCE_SUPPORT_EXAMPLES = 256
ENGINE_SUPPORT_EXAMPLES = 16


def direct_source_identity():
    """Bind loaded F3 modules; inherited helpers are bound by the outer snapshot."""
    result = {}
    for name in (
        "runtime",
        "masks",
        "data",
        "contract",
        "prompts",
        "protocol_revision",
        "native_audit",
    ):
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


def audit_native_inputs(root, *, protocol_revision=None):
    import torch

    from .prompts import prefill_for
    from .protocol_revision import revision_from_config, serialize_teacher_target
    from .runner import dataset_and_store, engine_schedule
    from .runtime import SRF3CERuntime

    root = Path(root).resolve(strict=True)
    config_path = root / "plan/config/SR_F3_CE.json"
    configured_revision = revision_from_config(
        read_json(config_path) if config_path.exists() else {}
    )
    if protocol_revision is not None and protocol_revision != configured_revision:
        raise PermissionError("Native audit revision must match its frozen plan")
    protocol_revision = configured_revision
    dataset, store = dataset_and_store(root)
    prior = read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    runtime = SRF3CERuntime.processor_only(prior["model_path"], protocol_revision=protocol_revision)
    if runtime.identity.get("protocol_revision", "v1") != protocol_revision:
        raise PermissionError("Native processor revision differs from frozen audit plan")
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
            target = serialize_teacher_target(target, protocol_revision)
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
            protocol_revision=protocol_revision,
            assistant_prefill=prefill_for("structured", protocol_revision),
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
        protocol_revision=protocol_revision,
    )
    durable_json(root / "native/SUPPORT_MASK_RECEIPT.json", mask_receipt)
    sources = direct_source_identity()
    receipt = dict(
        status="PASS",
        protocol_revision=protocol_revision,
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


def _context_path(root, revision, output_path=None):
    from .prompts import prefill_for

    prefill_for("structured", revision)
    expected = root / "native/protocol_contexts" / (revision + ".json")
    output = Path(output_path) if output_path is not None else expected
    output = output if output.is_absolute() else root / output
    if output.resolve() != expected or expected.is_symlink():
        raise PermissionError("Protocol context receipt must use its registered native path")
    return expected


def _verify_context_sources(root, recorded):
    """Permit another deployment of identical code, retaining original CPU provenance."""
    current = direct_source_identity()
    if set(recorded) != set(current):
        raise PermissionError("Protocol context direct-source inventory differs")
    code = (root / "code").resolve(strict=True)
    for relative, item in recorded.items():
        path = (code / relative).resolve(strict=True)
        if (
            not path.is_relative_to(code)
            or item.get("loaded_from") != str(path)
            or item.get("sha256") != file_hash(path)
            or item.get("bytes") != path.stat().st_size
            or any(current[relative][key] != item[key] for key in ("sha256", "bytes"))
        ):
            raise PermissionError("Protocol context direct-source provenance differs: " + relative)


def _context_registration(root, revision):
    """Use original ENGINE slots and independent model-input stores, never teacher text."""
    from .data import ModelInputStore
    from .evaluation import engine_slots
    from .prompts import prompt_for_revision
    from .protocol_revision import verify_revision_package
    from .runner import dataset_and_store

    dataset, original = dataset_and_store(root)
    slots = [
        s
        for s in engine_slots(dataset.tasks)
        if s["protocol"] == "structured" and s["view"] is None
    ]
    panels = [
        dict(
            block=block,
            parent_id=parent,
            qids=sorted(
                {s["qid"] for s in slots if s["block"] == block and s["parent_id"] == parent}
            ),
        )
        for block, parent in enumerate(PARENTS)
    ]
    qids = sorted({s["qid"] for s in slots})
    if len(slots) != 768 or len(qids) != 64 or any(panel["qids"] != qids for panel in panels):
        raise PermissionError("Protocol contexts must cover the exact 64 by 3 ENGINE panel")
    plan = root / "plan" if revision == "v1" else root / "candidates" / revision / "plan"
    if revision != "v1":
        verified = verify_revision_package(plan, source_plan_dir=root / "plan")
        if verified["protocol_revision"] != revision:
            raise PermissionError("Candidate package revision differs from requested context audit")
    store = ModelInputStore.from_files(
        plan / "manifests/MODEL_INPUTS.jsonl",
        root / "data/IMAGE_MANIFEST.jsonl",
        image_root=root / "data",
    )
    requests = {}
    for qid in qids:
        safe = store.model_input(qid)
        expected = original.model_input(qid)
        expected["prompt"] = prompt_for_revision(expected["prompt"], revision)
        if safe != expected or safe["image_file"] is None:
            raise PermissionError("Protocol context differs from registered ENGINE prompt/image")
        requests[qid] = safe
    files = {
        root / "plan/PACKAGE_SHA256.json",
        root / "plan/manifests/TASKS_AUDIT_ONLY.jsonl",
        root / "data/IMAGE_MANIFEST.jsonl",
        root / "prior/MODEL_ENVIRONMENT_IDENTITY.json",
        plan / "PACKAGE_SHA256.json",
        plan / "config/SR_F3_CE.json",
        plan / "manifests/MODEL_INPUTS.jsonl",
    }
    if revision != "v1":
        files.add(plan / "PROTOCOL_REVISION.json")
    sources = {}
    for path in sorted(files):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise PermissionError("Protocol context source escapes its registered root")
        sources[str(path.relative_to(root))] = dict(
            sha256=file_hash(path), bytes=path.stat().st_size
        )
    return store, requests, panels, sources


def _audit_protocol_context(runtime, safe, image_root):
    """Audit complete actual processor IDs/offsets and their full native text round trip."""
    from .prompts import prefill_for

    prepared = runtime.prepare_input(safe, image_root, protocol="structured")
    encoded = runtime.processor(
        **runtime._processor_arguments(safe, image_root, prepared["chat_text"]),
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise PermissionError("Complete native context offsets are required")
    offsets = encoded.pop("offset_mapping")
    offsets = offsets.tolist() if hasattr(offsets, "tolist") else offsets
    if "mm_token_type_ids" not in encoded and safe["image_file"] is None:
        encoded["mm_token_type_ids"] = runtime.torch.zeros_like(encoded["input_ids"])
    if set(encoded) != set(prepared["inputs"]) or any(
        not runtime.torch.equal(value.cpu(), prepared["inputs"][key].cpu())
        for key, value in encoded.items()
    ):
        raise PermissionError("Offset-bearing native context changed model input tensors")
    ids = encoded["input_ids"][0].tolist()
    if len(offsets) != 1 or len(offsets[0]) != len(ids):
        raise PermissionError("Native context offsets and input IDs differ")
    offsets = [list(pair) for pair in offsets[0]]
    route, tokenizer = prepared["routing"], runtime.processor.tokenizer
    expanded = prepared["chat_text"]
    if safe["image_file"] is not None:
        if (
            route.get("source_size") != list(CANVAS)
            or route.get("processed_size") != list(CANVAS)
            or route.get("processor_target_pixels") != TARGET_PIXELS
        ):
            raise PermissionError("Native protocol context changed fixed image geometry")
        image_token = runtime.processor.image_token
        if expanded.count(image_token) != 1 or route["image_token_count"] <= 0:
            raise PermissionError("Expected one independently routed native image placeholder")
        expanded = expanded.replace(image_token, image_token * route["image_token_count"])
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    whole = tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    if (
        text != expanded
        or whole["input_ids"] != ids
        or [list(pair) for pair in whole["offset_mapping"]] != offsets
    ):
        raise PermissionError("Full native context tokenizer/text/offset identity differs")
    prefill = prefill_for("structured", runtime.protocol_revision)
    start = len(expanded) - len(prefill)
    indices = [i for i, (lo, hi) in enumerate(offsets) if hi > start]
    if (
        not expanded.endswith(prefill)
        or not indices
        or offsets[indices[0]][0] != start
        or offsets[-1][1] != len(expanded)
        or tokenizer.decode(
            [ids[i] for i in indices], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        != prefill
    ):
        raise PermissionError("Registered prefill must be entirely present in native context")
    tensor_hashes = {key: state_hash(value) for key, value in encoded.items()}
    if hash_json(tensor_hashes) != route["input_tensor_hash"]:
        raise PermissionError("Native context tensor identity differs from runtime routing")
    return dict(
        model_input=safe,
        model_input_sha256=digest(safe),
        assistant_prefill=prefill,
        protocol_revision=runtime.protocol_revision,
        chat_text=prepared["chat_text"],
        chat_text_sha256=hashlib.sha256(prepared["chat_text"].encode()).hexdigest(),
        native_expanded_chat_text=expanded,
        native_expanded_chat_text_sha256=hashlib.sha256(expanded.encode()).hexdigest(),
        input_ids=ids,
        offsets=offsets,
        offset_coordinate_system="native_processor_expanded_chat_text",
        prompt_tensor_hash=route["input_tensor_hash"],
        input_tensor_hashes=tensor_hashes,
        prefill_token_indices=indices,
        prefill_token_ids=[ids[i] for i in indices],
        prefill_native_char_span=[start, len(expanded)],
        prefill_is_context_not_loss_target=True,
        routing=route,
    )


def audit_protocol_contexts(root, revision, output_path=None):
    """Publish CPU-only native ENGINE contexts before any ablation outputs exist."""
    import torch

    from .prompts import prefill_for
    from .runtime import SRF3CERuntime

    root = Path(root).resolve(strict=True)
    path = _context_path(root, revision, output_path)
    if (root / "ABLATION_MANIFEST.json").exists() or (root / "ablation/raw").exists():
        raise PermissionError("Context acceptance must precede frozen ablation and model outputs")
    if torch.cuda.is_initialized():
        raise RuntimeError("Protocol context audit must not initialize CUDA")
    store, requests, panels, sources = _context_registration(root, revision)
    prior = read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    direct = direct_source_identity()
    _verify_context_sources(root, direct)
    runtime = SRF3CERuntime.processor_only(prior["model_path"], protocol_revision=revision)
    if runtime.identity.get("protocol_revision") != revision or hasattr(
        runtime.model, "parameters"
    ):
        raise PermissionError("Only the exact revision processor without weights may be audited")
    for key in ("processor_hash", "tokenizer_hash", "chat_template_hash"):
        if runtime.identity[key] != prior[key]:
            raise PermissionError("Native protocol context environment differs: " + key)
    pixel_budget = processor_pixel_budget(runtime)
    contexts = {
        qid: _audit_protocol_context(runtime, safe, store.image_root)
        for qid, safe in requests.items()
    }
    if (
        torch.cuda.is_initialized()
        or direct_source_identity() != direct
        or any(file_hash(root / name) != item["sha256"] for name, item in sources.items())
    ):
        raise PermissionError("CPU protocol context sources or execution boundary changed")
    receipt = dict(
        schema_version=1,
        status="PASS",
        protocol="structured",
        protocol_revision=revision,
        assistant_prefill=prefill_for("structured", revision),
        model_calls=0,
        teacher_forward_calls=0,
        generated_model_outputs=0,
        weights_loaded=False,
        cuda_initialized=False,
        unique_contexts=len(contexts),
        parent_panel_contexts=sum(len(panel["qids"]) for panel in panels),
        parent_panels=panels,
        contexts=contexts,
        contexts_sha256=digest(contexts),
        requests_sha256=digest(requests),
        input_sources=sources,
        native_processor_identity=runtime.identity,
        native_pixel_budget=pixel_budget,
        direct_source_files=direct,
        prefill_is_context_not_loss_target=True,
        claim="actual_CPU_native_input_and_full_sequence_offset_identity_only",
    )
    durable_json(path, receipt)
    return verify_protocol_contexts(root, revision)


def verify_protocol_contexts(root, revision):
    """Read and authenticate a saved CPU context receipt without loading a processor."""
    from .prompts import prefill_for

    root = Path(root).resolve(strict=True)
    receipt = read_json(_context_path(root, revision))
    _store, requests, panels, sources = _context_registration(root, revision)
    expected = dict(
        schema_version=1,
        status="PASS",
        protocol="structured",
        protocol_revision=revision,
        assistant_prefill=prefill_for("structured", revision),
        model_calls=0,
        teacher_forward_calls=0,
        generated_model_outputs=0,
        weights_loaded=False,
        cuda_initialized=False,
        unique_contexts=64,
        parent_panel_contexts=192,
        parent_panels=panels,
        input_sources=sources,
        requests_sha256=digest(requests),
        prefill_is_context_not_loss_target=True,
        claim="actual_CPU_native_input_and_full_sequence_offset_identity_only",
    )
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise PermissionError("Native protocol context acceptance differs from registered inputs")
    _verify_context_sources(root, receipt["direct_source_files"])
    prior = read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    native = receipt["native_processor_identity"]
    if native.get("protocol_revision") != revision or any(
        native[key] != prior[key]
        for key in ("processor_hash", "tokenizer_hash", "chat_template_hash")
    ):
        raise PermissionError("Native protocol context processor identity differs")
    contexts = receipt["contexts"]
    if set(contexts) != set(requests) or receipt["contexts_sha256"] != digest(contexts):
        raise PermissionError("Native context inventory or content checksum differs")
    for qid, row in contexts.items():
        route = row["routing"]
        if (
            row["model_input"] != requests[qid]
            or row["model_input_sha256"] != digest(requests[qid])
            or row["protocol_revision"] != revision
            or row["assistant_prefill"] != expected["assistant_prefill"]
            or route["protocol_revision"] != revision
            or route["assistant_prefill"] != expected["assistant_prefill"]
            or row["prompt_tensor_hash"] != route["input_tensor_hash"]
            or row["prompt_tensor_hash"] != hash_json(row["input_tensor_hashes"])
            or route["input_ids_sha256"] != hash_json([row["input_ids"]])
            or len(row["offsets"]) != len(row["input_ids"])
            or not row["prefill_is_context_not_loss_target"]
            or row["prefill_token_ids"]
            != [row["input_ids"][i] for i in row["prefill_token_indices"]]
            or row["chat_text_sha256"] != hashlib.sha256(row["chat_text"].encode()).hexdigest()
            or row["native_expanded_chat_text_sha256"]
            != hashlib.sha256(row["native_expanded_chat_text"].encode()).hexdigest()
            or route["source_image_sha256"] != requests[qid]["image_sha256"]
        ):
            raise PermissionError("Native context prompt/prefill/image identity differs: " + qid)
    return receipt
