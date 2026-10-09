"""CPU-only historical review of the first SR-F1 FORMAT truncation guard.

This reviews the existing 32 x 8 records and never generates, trains, repairs a
response, drops a truncation, changes the 768-token channel, or relaxes 95%.
A bridge is permitted only if even making EVERY truncated response valid cannot
bring the original full panel to 95%, after the input/stop/parser audit passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from mm_core.vl_runtime import hash_json

from .contract import (
    PACKAGE,
    PLAN_ID,
    digest,
    file_hash,
    gold_output,
    load_config,
    score,
    stable_seed,
)
from .data import bounded_path, load_inputs, load_tasks, read_jsonl

RECEIPT = "FORMAT_TECHNICAL_REVIEW.json"
ARCHIVE = "technical_incidents/format_guard_20261009/FORMAT_AND_BRIDGE_RECEIPT.json"
COVERAGE = "engineering/format/before/COVERAGE.json"
CONTRACT_SOURCE = "source:src/sr_f1/contract.py"
REFERENCE_SOURCE = "package:reference/semantic_contract.py"
CHECK_NAMES = (
    "original_block_archived_unchanged",
    "no_prior_bridge_or_scientific_progress_at_review",
    "frozen_config_and_parser_unchanged",
    "frozen_model_and_processor_identity",
    "all_256_original_slots_seeds_inputs_and_policy",
    "all_record_hashes_valid",
    "all_32_native_cpu_routes_match_saved_routes",
    "sampling_parameters_and_hashes_unchanged",
    "raw_tokens_decode_exactly_without_repair",
    "eos_and_length_stopping_coherent",
    "finite_equal_actual_sampling_logprobs",
    "generation_complete_no_technical_errors_and_visual_forward",
    "original_coverage_receipt_reproduced_without_accuracy",
    "all_truncations_retained_in_full_denominator",
    "nontruncated_protocol_deficiency_independently_established",
    "gold_completions_fit_768_including_eos_all_three_pools",
    "strict_parser_accepts_gold_and_rejects_malformed_wrappers",
    "cpu_only_no_weights_no_model_calls_no_cuda_context",
)
MODEL_KEYS = (
    "processor_hash",
    "tokenizer_hash",
    "chat_template_hash",
    "chat_template_kwargs",
    "chat_template_kwargs_hash",
    "generation_config_expanded",
    "canvas_pixels",
    "processor_target_pixels",
    "model_type",
    "architecture_verified",
    "full_attention_layers",
    "torch_version",
    "transformers_version",
    "linear_kernel_identity_hash",
)


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _read(root, name):
    return json.loads(bounded_path(root, name).read_text())


def _source_path(key):
    if key == CONTRACT_SOURCE:
        return Path(__file__).with_name("contract.py")
    if key == REFERENCE_SOURCE:
        return PACKAGE / "reference/semantic_contract.py"
    raise PermissionError("Unregistered external artifact reference")


def _artifact_path(root, name):
    return (
        _source_path(name)
        if name in (CONTRACT_SOURCE, REFERENCE_SOURCE)
        else bounded_path(root, name)
    )


def _write_new(path, payload):
    from mm_dev.runtime import atomic_json

    atomic_json(path, payload, exclusive=True)


def _archive_once(path, destination):
    payload = Path(path).read_bytes()
    if destination.exists():
        _require(
            destination.read_bytes() == payload,
            "Existing incident archive differs from original block",
        )
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending-block-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def _assert_cpu(runtime=None):
    import torch

    _require(not torch.cuda.is_initialized(), "Review must not initialize or use a CUDA context")
    if runtime is not None:
        _require(runtime.device == "cpu", "Review processor must run only on CPU")
        _require(isinstance(runtime.model, SimpleNamespace), "Review must not load model weights")


def _load_processor(model_path):
    from .runtime import SRRuntime

    _assert_cpu()
    runtime = SRRuntime.processor_only(model_path)
    _assert_cpu(runtime)
    return runtime


def coverage_evidence(counts):
    """Prove deficiency remains even under the most favorable truncation repair."""
    keys = {"responses", "covered", "truncated", "untruncated", "untruncated_covered"}
    _require(set(counts) == keys, "Coverage count schema differs")
    _require(all(type(v) is int and v >= 0 for v in counts.values()), "Invalid coverage counts")
    n, c, t, u, uc = (
        counts[k]
        for k in ("responses", "covered", "truncated", "untruncated", "untruncated_covered")
    )
    _require(
        n == 256 and u + t == n and 0 <= uc <= u and uc <= c <= uc + t,
        "Coverage counts violate the complete 32 x 8 panel",
    )
    _require(c / n < 0.95, "The original protocol already satisfies the bridge threshold")
    upper = (uc + t) / n
    _require(upper < 0.95, "Truncation alone could explain the failure; bridge remains blocked")
    return dict(
        upper_bound_coverage_if_all_truncated_valid=upper,
        untruncated_coverage=uc / u if u else None,
        required_coverage=0.95,
        full_denominator=n,
        decision_uses_answer_or_evidence_accuracy=False,
    )


def _flags(raw_text, task):
    scored = score(raw_text, task["world"], task["query"])
    return {k: scored[k] for k in ("L_json", "L_answer", "L_evidence")}


def _coverage_counts(records, tasks):
    covered = untruncated_covered = truncated = 0
    failures, eos_failures, rows = Counter(), Counter(), []
    for record in records:
        flags = _flags(record["raw_text"], tasks[record["qid"]])
        valid = all(flags.values())
        is_truncated = record["truncated"]
        covered += valid
        truncated += is_truncated
        untruncated_covered += valid and not is_truncated
        if not flags["L_json"]:
            category = "invalid_whole_json_or_top_level_contract"
        elif not flags["L_answer"] and not flags["L_evidence"]:
            category = "answer_and_evidence_not_scorable"
        elif not flags["L_answer"]:
            category = "answer_not_scorable"
        elif not flags["L_evidence"]:
            category = "evidence_not_scorable"
        else:
            category = "covered"
        failures[category] += 1
        if not is_truncated:
            eos_failures[category] += 1
        rows.append(
            dict(
                qid=record["qid"],
                sample_index=record["sample_index"],
                truncated=is_truncated,
                field_flags=flags,
                category=category,
            )
        )
    counts = dict(
        responses=len(records),
        covered=covered,
        truncated=truncated,
        untruncated=len(records) - truncated,
        untruncated_covered=untruncated_covered,
    )
    return counts, dict(all_responses=dict(failures), eos_responses=dict(eos_failures), rows=rows)


def _validate_tokens(record, tokenizer, generation):
    tokens = record.get("tokens")
    _require(
        isinstance(tokens, list) and 0 < len(tokens) <= 768,
        "Raw completion length violates the registered channel",
    )
    _require(all(type(t) is int and t >= 0 for t in tokens), "Invalid raw token identifiers")
    _require(
        record.get("raw_tokens") == tokens and record.get("completion_token_count") == len(tokens),
        "Raw token streams or completion count differ",
    )
    eos = generation["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    _require(eos and all(type(t) is int for t in eos), "Missing native EOS identity")
    _require(
        not any(t in eos for t in tokens[:-1]), "EOS token occurs before final completion token"
    )
    ended = tokens[-1] in eos
    _require(ended or len(tokens) == 768, "Non-EOS completion ended before the fixed token cap")
    _require(
        type(record.get("truncated")) is bool and record["truncated"] == (not ended),
        "Truncation flag contradicts actual EOS/length stopping",
    )
    _require(
        record.get("finish_reason") == ("eos" if ended else "length"),
        "Finish reason contradicts actual EOS/length stopping",
    )
    old, sampler = record.get("old_logprobs"), record.get("sampler_logprobs")
    _require(
        isinstance(old, list) and len(old) == len(tokens) and old == sampler,
        "Actual raw and sampler log probabilities differ or have wrong lengths",
    )
    _require(
        all(type(v) in (float, int) and math.isfinite(v) and v <= 0 for v in old),
        "Nonfinite or invalid actual sampling log probability",
    )
    _require(
        record.get("old_logprob_source") == "actual_generate_raw_logits_selected_token",
        "Probability source is not actual raw generation",
    )
    decoded = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    _require(decoded == record.get("raw_text"), "Token decoding differs from preserved raw text")


def _raw_paths(tasks):
    qids = sorted(q for q, t in tasks.items() if t["pool"] == "FORMAT")
    _require(len(qids) == 32, "FORMAT must contain exactly 32 registered questions")
    return qids, [f"engineering/format/before/{q}-{i}.json" for q in qids for i in range(8)]


def _verify_record_identity(record, qid, draw, inputs, zero, generation):
    expected = dict(
        qid=qid,
        sample_index=draw,
        seed=stable_seed(PLAN_ID, "FORMAT", "FORMAT", qid, draw),
        policy_hash=zero["trainable_state_hash"],
        input_hash=digest(inputs[qid]),
    )
    _require(
        all(record.get(k) == v for k, v in expected.items()),
        "Original FORMAT slot identity changed",
    )
    _require(
        record.get("record_hash")
        == digest({k: v for k, v in record.items() if k != "record_hash"}),
        "Original raw record hash changed",
    )
    _require(
        record.get("generation_status") == "COMPLETE"
        and record.get("technical_validation_errors") == [],
        "FORMAT has an unresolved technical generation failure",
    )
    _require(
        record.get("sampling_parameters") == generation
        and record.get("sampling_hash") == hash_json(generation),
        "Raw sampler differs from fixed native generation configuration",
    )
    _require(
        record.get("image_routing") == record.get("input_routing"),
        "Stored input/image routes disagree",
    )
    _require(
        type(record["input_routing"].get("generation_vision_forward_calls")) is int
        and record["input_routing"]["generation_vision_forward_calls"] > 0,
        "Recorded generation did not use the native visual encoder",
    )
    model = record.get("model_identity", {})
    _require(
        all(model.get(k) == zero[k] for k in zero), "Raw model differs from common zero adapter"
    )
    adapter = record.get("adapter_identity", {})
    _require(
        adapter.get("adapter_parameter_hash") == zero["trainable_state_hash"]
        and adapter.get("published_adapter_hash") == zero["adapter_hash"]
        and adapter.get("base_model_weights_hash") == zero["base_model_weights_hash"]
        and adapter.get("model_id") == "SRF1_COMMON_ZERO"
        and adapter.get("step") is None,
        "Actual raw adapter identity differs from verified common zero",
    )


def _base_artifacts(root, tasks, freeze, zero):
    _qids, raw_paths = _raw_paths(tasks)
    names = set(freeze["artifact_hashes"])
    names.update(
        (
            "EXECUTION_FREEZE.json",
            "config/SR_F1.json",
            "COMMON_ZERO_LORA.json",
            "manifests/MODEL_INPUTS.jsonl",
            "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl",
            COVERAGE,
            ARCHIVE,
            CONTRACT_SOURCE,
            REFERENCE_SOURCE,
        )
    )
    names.update(raw_paths)
    for name in zero["adapter_file_hashes"]:
        names.add(str(Path(zero["adapter_path"]) / name))
    adapter_marker = str(Path(zero["adapter_path"]) / "IDENTITY.json")
    if bounded_path(root, adapter_marker).exists():
        names.add(adapter_marker)
    return {name: file_hash(_artifact_path(root, name)) for name in sorted(names)}


def _historical_inputs(root):
    """Check the frozen record chain without requiring today's source tree hash."""
    tasks, inputs = load_tasks(root), load_inputs(root)
    freeze = _read(root, "EXECUTION_FREEZE.json")
    _require(
        freeze.get("plan_id") == PLAN_ID
        and freeze.get("status") == "FROZEN"
        and freeze.get("run_root") == str(root),
        "Wrong original execution freeze",
    )
    config = _read(root, "config/SR_F1.json")
    _require(config == load_config(), "Frozen SR-F1 contract changed")
    _require(
        file_hash(root / "config/SR_F1.json") == freeze["config_sha256"],
        "Frozen config bytes changed",
    )
    for name, expected in freeze["artifact_hashes"].items():
        _require(
            file_hash(bounded_path(root, name)) == expected,
            "Original frozen CPU artifact changed: " + name,
        )
    required = {
        "MODEL_ENVIRONMENT_IDENTITY.json",
        "PROCESSOR_PREFLIGHT.json",
        "PROCESSOR_INPUT_ROUTES.jsonl",
    }
    _require(
        required <= freeze["artifact_hashes"].keys(), "Frozen processor evidence chain missing"
    )
    model = _read(root, "MODEL_ENVIRONMENT_IDENTITY.json")
    _require(model == freeze["model_identity"], "Model environment differs from original freeze")
    _require(
        model["model_id"] == config["model"]["id"]
        and model["model_revision"] == config["model"]["revision"]
        and model["model_weights_hash"] == config["model"]["prior_composite_weight_hash"],
        "Wrong frozen model identity",
    )
    _require(
        _read(root, "PROCESSOR_PREFLIGHT.json")["status"] == "PASS",
        "Native processor preflight did not pass",
    )
    _require(
        file_hash(_source_path(CONTRACT_SOURCE))
        == freeze["source_file_hashes"]["src/sr_f1/contract.py"],
        "Strict parser wrapper differs from frozen source",
    )
    package_hashes = json.loads((PACKAGE / "PACKAGE_SHA256.json").read_text())
    _require(
        file_hash(_source_path(REFERENCE_SOURCE))
        == package_hashes["reference/semantic_contract.py"],
        "Archived strict parser reference changed",
    )
    zero = _read(root, "COMMON_ZERO_LORA.json")
    _require(
        zero.get("plan_id") == PLAN_ID
        and zero.get("status") == "VERIFIED"
        and zero.get("model_id") == "SRF1_COMMON_ZERO"
        and zero.get("zero_output_exact") is True
        and zero.get("freeze_sha256") == file_hash(root / "EXECUTION_FREEZE.json")
        and zero.get("base_model_weights_hash") == model["model_weights_hash"],
        "Common zero LoRA is not the verified original",
    )
    _require(
        zero.get("initialization_seed") == config["model"]["lora"]["init_seed"],
        "Common zero initialization seed changed",
    )
    files = zero.get("adapter_file_hashes", {})
    _require(
        {"adapter_config.json", "adapter_model.safetensors"} <= files.keys()
        and digest(files) == zero["adapter_hash"],
        "Common zero adapter manifest/hash missing",
    )
    for name, expected in files.items():
        path = bounded_path(bounded_path(root, zero["adapter_path"]), name)
        _require(file_hash(path) == expected, "Common zero adapter bytes changed")
    marker = bounded_path(root, str(Path(zero["adapter_path"]) / "IDENTITY.json"))
    if marker.exists():
        _require(json.loads(marker.read_text()) == zero, "Common zero adapter marker differs")
    return tasks, inputs, freeze, model, zero


def _original_coverage(root, records, tasks, zero):
    original = _read(root, COVERAGE)
    counts, failures = _coverage_counts(records, tasks)
    expected = dict(
        panel="FORMAT",
        label="before",
        policy_hash=zero["trainable_state_hash"],
        responses=counts["responses"],
        covered=counts["covered"],
        truncated=counts["truncated"],
        coverage=counts["covered"] / 256,
        coverage_uses_gold_accuracy=False,
    )
    _require(
        all(original.get(k) == v for k, v in expected.items()),
        "Original COVERAGE field counts do not reproduce",
    )
    block = _read(root, ARCHIVE)
    _require(
        block.get("status") == "PROTOCOL_BLOCKED"
        and block.get("reason") == "FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW"
        and block.get("bridge_executed") is False
        and block.get("format_before") == original,
        "Archived original block is not the pre-bridge guard incident",
    )
    return counts, failures, coverage_evidence(counts)


def _gold_channel_audit(runtime, tasks):
    rows = []
    pool_counts = Counter()
    for qid, task in sorted(tasks.items()):
        if task["pool"] not in ("FORMAT", "FORMAT_CONFIRM", "BRIDGE"):
            continue
        text = json.dumps(
            gold_output(task["world"], task["query"]), ensure_ascii=False, separators=(",", ":")
        )
        tokens = runtime.encode_completion(text)
        _require(
            0 < len(tokens) <= 768
            and tokens[-1] in runtime.eos_ids
            and not any(t in runtime.eos_ids for t in tokens[:-1]),
            "Registered gold completion does not fit the frozen EOS channel",
        )
        _require(
            all(_flags(text, task).values()), "Strict parser rejects a registered gold completion"
        )
        for invalid in (
            "Answer: " + text,
            "```json\n" + text + "\n```",
            text + " trailing",
            "{",
            '{"evidence":false,"answer":true}',
        ):
            _require(
                not all(_flags(invalid, task).values()),
                "Strict parser accepted a malformed/wrapped response",
            )
        pool_counts[task["pool"]] += 1
        rows.append(
            dict(
                qid=qid,
                pool=task["pool"],
                completion_tokens_including_eos=len(tokens),
                gold_text_sha256=digest(text),
                completion_token_hash=digest(tokens),
            )
        )
    _require(
        pool_counts == {"FORMAT": 32, "FORMAT_CONFIRM": 32, "BRIDGE": 128},
        "Gold audit omitted a registered prompt",
    )
    return dict(
        pool_counts=dict(pool_counts),
        maximum_completion_tokens_including_eos=max(
            r["completion_tokens_including_eos"] for r in rows
        ),
        output_token_limit=768,
        same_encoding_as_registered_bridge=True,
        rows=rows,
    )


def build_format_review(root):
    root = Path(root).resolve(strict=True)
    if (root / RECEIPT).exists():
        return verify_format_review(root)
    _assert_cpu()
    _require(not (root / "STOP").exists(), "STOP is active; technical review must not continue")
    for name in ("COMMON_START.json", "states/SRF1_COMMON_START", "training", "engineering/bridge"):
        _require(
            not (root / name).exists(),
            "Review requires no prior bridge or scientific progress: " + name,
        )
    block_path = root / "FORMAT_AND_BRIDGE_RECEIPT.json"
    block = json.loads(block_path.read_text())
    _require(
        block.get("status") == "PROTOCOL_BLOCKED"
        and block.get("bridge_executed") is False
        and block.get("reason") == "FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW",
        "Only the original pre-bridge truncation guard may be reviewed",
    )
    archive = bounded_path(root, ARCHIVE)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _archive_once(block_path, archive)
    tasks, inputs, freeze, model, zero = _historical_inputs(root)
    runtime = _load_processor(model["model_path"])
    for key in MODEL_KEYS:
        _require(
            runtime.identity.get(key) == model.get(key),
            "Actual CPU processor/model identity changed: " + key,
        )
    generation = runtime.generation_config.to_dict()
    _require(
        generation == model["generation_config_expanded"]
        and generation.get("max_new_tokens") == 768,
        "Actual native generation configuration differs from freeze",
    )
    routes = read_jsonl(root / "PROCESSOR_INPUT_ROUTES.jsonl")
    route_map = {r["qid"]: r for r in routes}
    _require(
        len(route_map) == len(routes) == 4800,
        "Frozen processor routes are incomplete or duplicated",
    )
    qids, expected_paths = _raw_paths(tasks)
    actual_paths = {
        str(p.relative_to(root))
        for p in (root / "engineering/format/before").glob("*.json")
        if p.name != "COVERAGE.json"
    }
    _require(
        actual_paths == set(expected_paths), "Original FORMAT panel has missing or extra raw slots"
    )
    records = []
    cpu_routes = []
    for qid in qids:
        _require(not (root / "STOP").exists(), "STOP was requested during CPU review")
        prepared = runtime.prepare(inputs[qid], root)
        actual = prepared["routing"]
        frozen = {k: v for k, v in route_map[qid].items() if k not in ("qid", "image_file")}
        _require(
            actual == frozen and route_map[qid]["image_file"] == inputs[qid]["image_file"],
            "Native CPU prompt/image routing differs from frozen processor routes",
        )
        _require(
            actual["image_token_count"] > 0
            and actual["processed_size"] == [1024, 768]
            and actual["processor_target_pixels"] == 786432,
            "Native visual route violates fixed canvas/channel",
        )
        cpu_routes.append(
            dict(
                qid=qid,
                routing_hash=digest(actual),
                prompt_token_count=prepared["inputs"]["input_ids"].shape[-1],
            )
        )
        for draw in range(8):
            record = _read(root, f"engineering/format/before/{qid}-{draw}.json")
            _verify_record_identity(record, qid, draw, inputs, zero, generation)
            for key in MODEL_KEYS:
                _require(
                    record["model_identity"].get(key) == model.get(key),
                    "Historical model/template identity differs: " + key,
                )
            saved = {
                k: v
                for k, v in record["input_routing"].items()
                if k != "generation_vision_forward_calls"
            }
            _require(
                saved == actual
                and record["prompt_token_count"] == prepared["inputs"]["input_ids"].shape[-1],
                "Actual original generation input differs from audited native CPU input",
            )
            _validate_tokens(record, runtime.processor.tokenizer, generation)
            records.append(record)
    counts, failures, evidence = _original_coverage(root, records, tasks, zero)
    gold = _gold_channel_audit(runtime, tasks)
    _assert_cpu(runtime)
    receipt = dict(
        plan_id=PLAN_ID,
        status="VERIFIED_PROTOCOL_NONADHERENCE",
        freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        permit_bridge=True,
        model_calls=0,
        model_weights_loaded=False,
        cuda_context_initialized=False,
        prior_bridge_executed=False,
        counts=counts,
        **evidence,
        field_coverage_diagnostics=failures,
        native_cpu_format_routes=cpu_routes,
        gold_completion_channel_audit=gold,
        checks=dict.fromkeys(CHECK_NAMES, True),
        artifact_hashes=_base_artifacts(root, tasks, freeze, zero),
    )
    receipt["receipt_hash"] = digest(receipt)
    _write_new(root / RECEIPT, receipt)
    return verify_format_review(root)


def verify_format_review(root):
    """Verify the historic permit after bridge/science too; never consult their results."""
    root = Path(root).resolve(strict=True)
    receipt = _read(root, RECEIPT)
    _require(
        receipt.get("receipt_hash")
        == digest({k: v for k, v in receipt.items() if k != "receipt_hash"}),
        "Format technical review receipt hash changed",
    )
    expected = dict(
        plan_id=PLAN_ID,
        status="VERIFIED_PROTOCOL_NONADHERENCE",
        permit_bridge=True,
        model_calls=0,
        model_weights_loaded=False,
        cuda_context_initialized=False,
        prior_bridge_executed=False,
    )
    _require(
        all(receipt.get(k) == v for k, v in expected.items()),
        "Review does not certify the registered CPU-only permit",
    )
    _require(
        receipt.get("checks") == dict.fromkeys(CHECK_NAMES, True),
        "Review checks are missing or not all verified",
    )
    tasks, inputs, freeze, model, zero = _historical_inputs(root)
    _require(
        receipt["freeze_sha256"] == file_hash(root / "EXECUTION_FREEZE.json"),
        "Review belongs to another freeze",
    )
    artifacts = _base_artifacts(root, tasks, freeze, zero)
    _require(
        receipt.get("artifact_hashes") == artifacts,
        "Bound historical artifact bytes changed or were omitted",
    )
    qids, paths = _raw_paths(tasks)
    records = []
    for qid in qids:
        for draw in range(8):
            record = _read(root, f"engineering/format/before/{qid}-{draw}.json")
            _verify_record_identity(
                record, qid, draw, inputs, zero, model["generation_config_expanded"]
            )
            records.append(record)
    _require(len(paths) == len(records) == 256, "Historical review lost original responses")
    counts, failures, evidence = _original_coverage(root, records, tasks, zero)
    _require(
        receipt["counts"] == counts and receipt["field_coverage_diagnostics"] == failures,
        "Review field-coverage counts or failure classes changed",
    )
    _require(
        all(receipt.get(k) == v for k, v in evidence.items()), "Review coverage upper bound changed"
    )
    gold = receipt["gold_completion_channel_audit"]
    _require(
        gold.get("pool_counts") == {"FORMAT": 32, "FORMAT_CONFIRM": 32, "BRIDGE": 128}
        and gold.get("output_token_limit") == 768
        and gold.get("same_encoding_as_registered_bridge") is True,
        "Registered gold-channel audit missing",
    )
    expected_gold = {
        q for q, t in tasks.items() if t["pool"] in ("FORMAT", "FORMAT_CONFIRM", "BRIDGE")
    }
    gold_rows = gold.get("rows", [])
    _require(
        len(gold_rows) == len(expected_gold)
        and {r["qid"] for r in gold_rows} == expected_gold
        and all(
            type(r["completion_tokens_including_eos"]) is int
            and 0 < r["completion_tokens_including_eos"] <= 768
            for r in gold_rows
        ),
        "Gold-channel audit length or completeness changed",
    )
    _require(
        gold["maximum_completion_tokens_including_eos"]
        == max(r["completion_tokens_including_eos"] for r in gold_rows),
        "Gold-channel audit maximum changed",
    )
    _require(
        len(receipt.get("native_cpu_format_routes", [])) == 32
        and {r["qid"] for r in receipt["native_cpu_format_routes"]} == set(qids),
        "Native CPU review does not cover exactly 32 FORMAT inputs",
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    receipt = build_format_review(args.root)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "plan_id",
                    "status",
                    "permit_bridge",
                    "model_calls",
                    "counts",
                    "upper_bound_coverage_if_all_truncated_valid",
                    "receipt_hash",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
