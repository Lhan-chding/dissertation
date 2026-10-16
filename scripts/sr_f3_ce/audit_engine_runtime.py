"""Read-only ENGINE prompt/termination audit. No model/optimizer APIs are called."""

from __future__ import annotations

import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import collections
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from transformers import AutoTokenizer


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def object_hash(value):
    return sha_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    )


def runtime_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, default=str).encode())


def load_json(path):
    return json.loads(Path(path).read_text())


def tensor_hash_int64(values):
    a = np.asarray(values, dtype=np.int64)
    h = hashlib.sha256(str(("torch.int64", tuple(a.shape))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def engine_manifest(path):
    # Do not deserialize or inspect task/world rows outside ENGINE and MONITOR.
    found = {}
    with Path(path).open() as stream:
        for line in stream:
            if '"ENGINE_' not in line and '"MONITOR_' not in line:
                continue
            row = json.loads(line)
            if row["qid"].startswith(("ENGINE_", "MONITOR_")):
                found[row["qid"]] = row
    return found


def boundary(text, protocol):
    if protocol == "natural_work":
        found = text.find("</final>")
        return None if found < 0 else found + len("</final>")
    if not text.startswith("{"):
        return None
    depth, quoted, escape = 0, False, False
    for i, c in enumerate(text):
        if quoted:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def json_state(text):
    depth, quoted, escape = 0, False, False
    for c in text:
        if quoted:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
    return {"brace_depth": depth, "inside_string": quoted, "escape_pending": escape}


def main():
    if not __debug__:
        raise RuntimeError("Optimized Python is unsupported: audit assertions must remain enabled")
    # Environment changes affect child interpreters, not this active importer.
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser()
    ap.add_argument("--repository", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--plan",
        type=Path,
        help="Frozen plan directory; defaults to repository artifacts/sr_f3_ce_20261016/plan",
    )
    ap.add_argument(
        "--prepared",
        type=Path,
        help=(
            "Prepared image directory; defaults to repository artifacts/sr_f3_ce_20261016/prepared"
        ),
    )
    args = ap.parse_args()
    repo, root, out = args.repository.resolve(), args.audit_root.resolve(), args.output.resolve()
    snapshot = root / "snapshot"
    if out == snapshot or snapshot in out.parents:
        raise RuntimeError("Output must never be inside the read-only snapshot")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source_manifest = load_json(snapshot / "deployment/SOURCE_MANIFEST.json")
    source_by_name = {r["path"]: r["sha256"] for r in source_manifest["files"]}
    source_names = [
        "src/sr_f3_ce/prompts.py",
        "src/sr_f3_ce/contract.py",
        "src/sr_f3_ce/runtime.py",
        "src/sr_f3_ce/data.py",
        "src/sr_f3_ce/evaluation.py",
        "src/sr_f3_ce/runner.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/json_protocol.py",
        "src/sr_f12/runtime.py",
        "src/mm_core/vl_runtime.py",
    ]
    source_hashes = {name: sha_file(repo / name) for name in source_names}
    assert all(source_by_name[n] == s for n, s in source_hashes.items())
    modules = {}
    for name in ("contract", "prompts"):
        path = repo / f"src/sr_f3_ce/{name}.py"
        spec = importlib.util.spec_from_file_location(f"audit_frozen_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    contract, prompts = modules["contract"], modules["prompts"]
    plan = (
        args.plan.resolve() if args.plan is not None else repo / "artifacts/sr_f3_ce_20261016/plan"
    )
    prepared = (
        args.prepared.resolve()
        if args.prepared is not None
        else repo / "artifacts/sr_f3_ce_20261016/prepared"
    )
    tasks = engine_manifest(plan / "manifests/TASKS_AUDIT_ONLY.jsonl")
    inputs = engine_manifest(plan / "manifests/MODEL_INPUTS.jsonl")
    images = engine_manifest(prepared / "IMAGE_MANIFEST.jsonl")
    assert len(tasks) == len(inputs) == len(images) == 128
    run = load_json(snapshot / "RUN_IDENTITY.json")
    assert sha_file(plan / "PACKAGE_SHA256.json") == run["package_manifest_sha256"]
    package_manifest = load_json(plan / "PACKAGE_SHA256.json")
    for name in (
        "manifests/MODEL_INPUTS.jsonl",
        "manifests/TASKS_AUDIT_ONLY.jsonl",
        "reference/prompts.py",
        "reference/build_manifest.py",
        "CODEX_EXECUTION_PLAN_SR_F3_CE_zh.md",
    ):
        assert sha_file(plan / name) == package_manifest[name]
    assert (
        sha_file(prepared / "IMAGE_MANIFEST.jsonl")
        == run["bound_files"]["data/IMAGE_MANIFEST.jsonl"]
    )
    image_checks = {
        qid: sha_file(prepared / row["image_file"]) == row["image_sha256"]
        for qid, row in images.items()
    }
    assert all(image_checks.values())
    tokenizer_dir = root / "supplement/frozen_tokenizer"
    prior = load_json(root / "supplement/prior/MODEL_ENVIRONMENT_IDENTITY.json")
    prior_by_name = {r["name"]: r["sha256"] for r in prior["model_files"]}
    tokenizer_file_hashes = {p.name: sha_file(p) for p in tokenizer_dir.iterdir() if p.is_file()}
    assert all(prior_by_name[n] == s for n, s in tokenizer_file_hashes.items())
    assert (
        sha_file(root / "supplement/prior/MODEL_ENVIRONMENT_IDENTITY.json")
        == run["bound_files"]["prior/MODEL_ENVIRONMENT_IDENTITY.json"]
    )
    tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    auto = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    assert auto.encode("{", add_special_tokens=False) == [90]
    assert tok.encode("{", add_special_tokens=False).ids == [90]
    assert runtime_hash(auto.get_vocab()) == prior["tokenizer_hash"]
    assert runtime_hash(auto.chat_template) == prior["chat_template_hash"]
    native = load_json(snapshot / "native/PROCESSOR_RECEIPT.json")["native_processor_identity"]
    frozen_gen = native["generation_config_expanded"]
    assert (
        frozen_gen["bos_token_id"] is None
        and frozen_gen["eos_token_id"] == [248044]
        and frozen_gen["pad_token_id"] == 248044
    )
    model_config = load_json(tokenizer_dir / "config.json")
    assert model_config["text_config"]["eos_token_id"] == 248044
    prompt_cache, routing_consistency, errors = {}, {}, []
    row_checks = collections.Counter()
    aggregates = collections.defaultdict(lambda: collections.Counter())
    group_hashes, eos_sensitivity, failing_rows = [], [], []
    seen_slots, seen_groups = set(), set()
    record_total = 0
    row_file = (out / "RECORD_AUDIT.jsonl").open("w")
    failure_file = (out / "UNPARSEABLE_RECORDS.jsonl").open("w")
    specifications = [
        (b, "ENGINE_PANEL", sorted((snapshot / "engine" / b / "groups").glob("*.json")), 240, 4)
        for b in ("b0", "b1", "b2")
    ]
    specifications += [
        (
            "monitor-b1",
            "MONITOR_BASELINE",
            sorted((snapshot / "monitor/SRF2_J6_s71002/step-00/groups").glob("*.json")),
            64,
            8,
        )
    ]
    specifications += [
        (
            "resume",
            "ENGINE_CACHED_ROLLOUT",
            sorted(
                p
                for p in (snapshot / "engine/resume/continuous/rollouts").glob("*.json")
                if not p.name.endswith(".request.json")
            ),
            64,
            8,
        )
    ]
    for block, origin, files, expected_groups, expected_rows in specifications:
        assert len(files) == expected_groups
        for file_index, path in enumerate(files):
            rel = str(path.relative_to(snapshot))
            group = load_json(path)
            assert len(group["records"]) == expected_rows
            assert object_hash(group["records"]) == group["records_sha256"]
            if origin != "ENGINE_CACHED_ROLLOUT":
                assert group["group_id"] == path.stem
                gid = group["group_id"]
            else:
                gid = object_hash({"source_path": rel, "identity": group["identity"]})
                assert load_json(path.with_suffix(".request.json")) == group["identity"]
                pending = [
                    json.loads(line)
                    for line in path.with_suffix(".pending.jsonl").read_text().splitlines()
                ]
                assert len(pending) == expected_rows
                assert all(
                    {k: v for k, v in r.items() if k not in ("qid", "score", "reward")}
                    == pending[i]
                    for i, r in enumerate(group["records"])
                )
            assert gid not in seen_groups
            seen_groups.add(gid)
            group_hashes.append(
                {"path": rel, "sha256": sha_file(path), "records_sha256": group["records_sha256"]}
            )
            for row_index, r in enumerate(group["records"]):
                if origin != "ENGINE_CACHED_ROLLOUT":
                    expected_pool = "ENGINE" if origin == "ENGINE_PANEL" else "MONITOR"
                    assert r["pool"] == r["phase"] == expected_pool and not r["sealed"]
                    row_identity = r["slot_id"]
                else:
                    assert r["qid"] == group["identity"]["qid"] and r["qid"].startswith("ENGINE_")
                    assert (
                        r["adapter_identity"]["adapter_parameter_hash"]
                        == group["identity"]["policy_hash"]
                    )
                    row_identity = object_hash({"cached_group": gid, "row": row_index})
                qid, proto, view = r["qid"], r["protocol"], r.get("view")
                assert qid in tasks and row_identity not in seen_slots
                seen_slots.add(row_identity)
                row_errors = []

                def check(name, value, *, failures=row_errors):
                    row_checks[name] += 1
                    if not value:
                        failures.append(name)

                task = tasks[qid]
                key = (qid, proto, view)
                if key not in prompt_cache:
                    text = prompts.prompt_from_frozen(inputs[qid]["prompt"], proto)
                    if view:
                        if view == "true_table":
                            extra = "The exact chart table is: " + contract.canonical(task["world"])
                        else:
                            inds = contract.selected(task["world"], task["query"])
                            vals = [
                                contract.encode(contract.operand(task["world"], task["query"], i))
                                for i in inds
                            ]
                            extra = (
                                (
                                    "The selected operands for the requested aggregate are: "
                                    + contract.canonical(vals)
                                )
                                if view == "selected_operands"
                                else (
                                    "The unevaluated expression is: "
                                    + task["query"]["aggregate"]
                                    + "("
                                    + contract.canonical(vals)
                                    + ")"
                                )
                            )
                        text += "\n" + extra
                    prefill = "{" if proto in ("structured", "answer_only") else ""
                    base_chat = auto.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": [{"type": "image"}, {"type": "text", "text": text}],
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    assert base_chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
                    chat = base_chat + prefill
                    # Native Qwen processor expands this sole image placeholder to 768 tokens.
                    # The persisted actual full input IDs hash independently
                    # verifies this reconstruction.
                    ids = auto(chat.replace("<|image_pad|>", "<|image_pad|>" * 768))["input_ids"]
                    base_ids = auto(base_chat.replace("<|image_pad|>", "<|image_pad|>" * 768))[
                        "input_ids"
                    ]
                    if prefill:
                        assert ids == [*base_ids, 90]
                    request = {
                        "prompt": text,
                        "image_file": images[qid]["image_file"],
                        "image_sha256": images[qid]["image_sha256"],
                    }
                    prompt_cache[key] = {
                        "request": request,
                        "prefill": prefill,
                        "chat_sha256": sha_bytes(chat.encode()),
                        "input_ids_sha256": runtime_hash([ids]),
                        "input_ids_tensor_sha256": tensor_hash_int64([ids]),
                        "attention_mask_tensor_sha256": tensor_hash_int64([[1] * len(ids)]),
                        "mm_token_type_ids_tensor_sha256": tensor_hash_int64(
                            [[int(t == 248056) for t in ids]]
                        ),
                        "prompt_tokens": len(ids),
                        "base_prompt_tokens": len(base_ids),
                    }
                expected = prompt_cache[key]
                routing = r["input_routing"]
                check(
                    "frozen_model_input",
                    r["model_input"] == expected["request"]
                    and r["prompt"] == expected["request"]["prompt"],
                )
                check(
                    "input_object_hash",
                    (
                        r["input_hash"]
                        if origin != "ENGINE_CACHED_ROLLOUT"
                        else group["identity"]["input_hash"]
                    )
                    == object_hash(expected["request"]),
                )
                check("full_chat_text_hash", routing["chat_text_sha256"] == expected["chat_sha256"])
                check(
                    "expanded_prompt_input_ids_hash",
                    routing["input_ids_sha256"] == expected["input_ids_sha256"],
                )
                check("prompt_token_count", r["prompt_token_count"] == expected["prompt_tokens"])
                for field in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    check(
                        field + "_tensor_hash",
                        routing["native_input_tensor_hashes"][field]
                        == expected[field + "_tensor_sha256"],
                    )
                check("image_bytes", routing["source_image_sha256"] == images[qid]["image_sha256"])
                check(
                    "native_image_geometry",
                    routing["source_size"] == [1024, 768]
                    and routing["processed_size"] == [1024, 768]
                    and routing["image_grid_thw"] == [[1, 48, 64]]
                    and routing["image_token_count"] == 768
                    and routing["processor_target_pixels"] == 786432,
                )
                check("image_routing_alias", r["image_routing"] == routing)
                check(
                    "prompt_tensor_hash",
                    r["prompt_tensor_hash"]
                    == routing["input_tensor_hash"]
                    == runtime_hash(routing["native_input_tensor_hashes"]),
                )
                check(
                    "processor_identity",
                    routing["processor_hash"] == native["processor_hash"]
                    and routing["chat_template_hash"] == native["chat_template_hash"]
                    and routing["chat_template_kwargs"] == {"enable_thinking": False},
                )
                check(
                    "prefill_context",
                    r["assistant_prefill"]
                    == r["prefill"]
                    == routing["assistant_prefill"]
                    == expected["prefill"]
                    and routing["prefill_is_context_not_loss_target"] is True,
                )
                if expected["prefill"]:
                    check(
                        "independent_brace_token",
                        routing["assistant_prefill_token_ids"] == [90]
                        and routing["base_prompt_token_count"] == expected["base_prompt_tokens"],
                    )
                else:
                    check(
                        "no_brace_in_natural_route",
                        "assistant_prefill_token_ids" not in routing
                        and "base_prompt_token_count" not in routing,
                    )
                check(
                    "sampling_config",
                    r["sampling_parameters"] == frozen_gen
                    and r["max_new_tokens"] == 1024
                    and (
                        origin == "ENGINE_CACHED_ROLLOUT"
                        or r["generation"]
                        == {
                            "max_new_tokens": 1024,
                            "do_sample": True,
                            "num_return_sequences": 1,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": 0,
                        }
                    ),
                )
                check("sampling_hash", r["sampling_hash"] == runtime_hash(r["sampling_parameters"]))
                if origin != "ENGINE_CACHED_ROLLOUT":
                    check(
                        "run_identity",
                        r["source_sha256"] == sha_file(snapshot / "RUN_IDENTITY.json"),
                    )
                else:
                    check(
                        "cached_task_identity", group["identity"]["task_hash"] == object_hash(task)
                    )
                check(
                    "technical_status",
                    r["generation_status"] == "COMPLETE" and not r["technical_validation_errors"],
                )
                if origin != "ENGINE_CACHED_ROLLOUT":
                    check(
                        "batch_slot_fields",
                        r["group_size"] == r["runtime_group_size"] == expected_rows
                        and r["group_row_index"]
                        == r["runtime_group_row_index"]
                        == r["draw"]
                        == row_index
                        and r["group_seed"]
                        == r["runtime_group_seed"]
                        == group["records"][0]["sampling_seed"]
                        and r["runtime_seed"] == r["sampling_seed"]
                        and r["protocol"] == r["runtime_protocol"],
                    )
                else:
                    check(
                        "batch_slot_fields",
                        r["group_size"] == expected_rows
                        and r["group_row_index"] == row_index
                        and r["group_seed"] == group["identity"]["seeds"][0]
                        and r["seed"] == group["identity"]["seeds"][row_index],
                    )
                route_projection = {
                    k: routing[k]
                    for k in ("native_input_tensor_hashes", "chat_text_sha256", "input_tensor_hash")
                }
                if key in routing_consistency:
                    check(
                        "cross_parent_route_consistency",
                        routing_consistency[key] == route_projection,
                    )
                else:
                    routing_consistency[key] = route_projection
                tokens, backend = r["tokens"], r["backend_sequence_tokens"]
                n = len(tokens)
                check(
                    "token_trajectory_lengths",
                    tokens == r["raw_tokens"] == backend[:n]
                    and r["completion_token_count"] == n
                    and 0 < n <= 1024
                    and r["backend_padding_count"] == len(backend) - n,
                )
                check("only_pad_after_stop", all(t == 248044 for t in backend[n:]))
                check(
                    "finite_selected_logprobs",
                    len(r["sampler_logprobs"]) == n
                    and all(
                        isinstance(v, (int, float)) and math.isfinite(v) and v <= 1e-5
                        for v in r["sampler_logprobs"]
                    ),
                )
                decoded = tok.decode(tokens, skip_special_tokens=True)
                special = tok.decode(tokens, skip_special_tokens=False)
                full = expected["prefill"] + decoded
                check(
                    "raw_token_decode",
                    decoded == r["generated_text"]
                    and special == r["generated_text_with_special_tokens"],
                )
                check(
                    "full_text_aliases",
                    r["raw_text"] == r["full_text"] == r["full_decoded_text"] == full,
                )
                first_n, first_char, previous_text, stopping_text = None, None, None, None
                prefix_ends = []
                stable = True
                views = {}
                for j in range(1, n + 1):
                    prefix = expected["prefill"] + tok.decode(tokens[:j], skip_special_tokens=True)
                    prefix_ends.append(len(prefix))
                    stable = stable and full.startswith(prefix)
                    cut = boundary(prefix, proto)
                    if first_n is None and cut is not None:
                        first_n, first_char = j, cut
                        stopping_text = prefix
                        previous_text = expected["prefill"] + tok.decode(
                            tokens[: j - 1], skip_special_tokens=True
                        )
                    if j in (128, 256, 512, 1024):
                        views[str(j)] = prefix
                for budget in (128, 256, 512, 1024):
                    if n < budget:
                        views[str(budget)] = full
                check(
                    "all_token_prefix_character_ends",
                    prefix_ends == r["token_prefix_character_ends"],
                )
                check("prefix_stability_flag", stable == r["prefix_character_offsets_stable"])
                check("saved_prefix_views", views == r["decoded_prefix_views"])
                eos_n = next((i + 1 for i, t in enumerate(backend) if t == 248044), None)
                expected_n = min(v for v in (first_n, eos_n, len(backend)) if v is not None)
                finish = (
                    ("final_tag" if proto == "natural_work" else "balanced")
                    if first_n == expected_n
                    else ("eos" if eos_n == expected_n else "length")
                )
                check("first_actual_stop", n == expected_n and r["finish_reason"] == finish)
                check(
                    "truncation_and_cap",
                    r["truncated"] == (finish == "length") and (finish != "length" or n == 1024),
                )
                cut = boundary(full, proto)
                check(
                    "character_boundary",
                    r["boundary_char"] == cut
                    and r["balanced_cut_char"] == (cut if proto != "natural_work" else None)
                    and r["final_tag_cut_char"] == (cut if proto == "natural_work" else None),
                )
                check(
                    "token_boundary",
                    r["boundary_token_count"] == (n if first_n == n else None)
                    and r["balanced_token_count"]
                    == (n if first_n == n and proto != "natural_work" else None),
                )
                if proto != "natural_work":
                    format_error = (
                        "unbalanced"
                        if cut is None
                        else ("trailing_after_close" if full[cut:].strip() else None)
                    )
                else:
                    format_error = (
                        "missing_closed_final_tag"
                        if cut is None
                        else (
                            "trailing_after_close"
                            if full[cut:].strip()
                            else (
                                "invalid_final_tag_count_or_order"
                                if full.count("<final>") != 1
                                or full.count("</final>") != 1
                                or full.index("<final>") > full.index("</final>")
                                else None
                            )
                        )
                    )
                check("format_error", r["format_protocol_error"] == format_error)
                score = contract.score(
                    full, task["world"], task["query"], finish_reason=finish, protocol=proto
                )
                if score["value_parseable"]:
                    parse_class, parse_error, keys = "parseable", None, None
                elif proto == "natural_work":
                    parse_class = (
                        "missing_final_tag" if cut is None else "invalid_final_scalar_or_tags"
                    )
                    parse_error = None
                    keys = None
                elif cut is None:
                    parse_class, parse_error, keys = "unclosed_object", None, None
                else:
                    try:
                        obj = contract.loads(full[:cut])
                        keys = list(obj) if isinstance(obj, dict) else None
                        if "answer" not in obj:
                            parse_class, parse_error = "missing_answer", None
                        else:
                            try:
                                contract.rational(obj["answer"])
                            except Exception as exc:
                                parse_class, parse_error = "invalid_answer_scalar", str(exc)
                            else:
                                raise RuntimeError("Parse classifier disagrees with frozen scorer")
                    except (ValueError, TypeError, RecursionError) as exc:
                        parse_class, parse_error, keys = "invalid_json", str(exc), None
                summary_key = "|".join((block, proto, view or "MAIN"))
                aggregate = aggregates[summary_key]
                aggregate["records"] += 1
                aggregate["parse:" + parse_class] += 1
                aggregate["stop:" + finish] += 1
                aggregate["suffix_nonwhitespace"] += int(
                    cut is not None and bool(full[cut:].strip())
                )
                aggregate["token_total"] += n
                item = {
                    "source_path": rel,
                    "source_file_sha256": group_hashes[-1]["sha256"],
                    "record_index": row_index,
                    "slot_id": r.get("slot_id"),
                    "audit_row_id": row_identity,
                    "origin": origin,
                    "qid": qid,
                    "block": block,
                    "protocol": proto,
                    "view": view,
                    "completion_tokens": n,
                    "backend_sequence_tokens": len(backend),
                    "first_boundary_token": first_n,
                    "first_boundary_character": first_char,
                    "actual_first_eos_token": eos_n,
                    "recorded_finish_reason": r["finish_reason"],
                    "replayed_finish_reason": finish,
                    "recorded_stop_token_id": tokens[-1],
                    "decoded_text_sha256": sha_bytes(decoded.encode()),
                    "full_text_sha256": sha_bytes(full.encode()),
                    "chat_text_sha256": expected["chat_sha256"],
                    "expanded_input_ids_sha256": expected["input_ids_sha256"],
                    "prefix_before_first_boundary_sha256": None
                    if previous_text is None
                    else sha_bytes(previous_text.encode()),
                    "first_boundary_prefix_sha256": None
                    if stopping_text is None
                    else sha_bytes(stopping_text.encode()),
                    "retained_suffix": None if cut is None else full[cut:],
                    "frozen_value_parseable": score["value_parseable"],
                    "parse_class": parse_class,
                    "parse_error": parse_error,
                    "top_level_keys": keys,
                    "raw_final_json_state": None if proto == "natural_work" else json_state(full),
                    "all_runtime_checks_pass": not row_errors,
                    "failed_checks": row_errors,
                }
                if row_errors:
                    errors.append(
                        {"source_path": rel, "record_index": row_index, "failed_checks": row_errors}
                    )
                row_file.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
                if not score["value_parseable"]:
                    failure = dict(item, prompt=r["prompt"], raw_text=full)
                    failure_file.write(
                        json.dumps(failure, sort_keys=True, ensure_ascii=False) + "\n"
                    )
                    failing_rows.append(
                        {
                            k: item[k]
                            for k in (
                                "source_path",
                                "record_index",
                                "slot_id",
                                "block",
                                "protocol",
                                "view",
                                "parse_class",
                            )
                        }
                    )
                if 248046 in tokens:
                    first_im_end = tokens.index(248046) + 1
                    altered = expected["prefill"] + tok.decode(
                        tokens[:first_im_end], skip_special_tokens=True
                    )
                    altered_score = contract.score(
                        altered, task["world"], task["query"], finish_reason="eos", protocol=proto
                    )
                    eos_sensitivity.append(
                        {
                            "source_path": rel,
                            "source_file_sha256": group_hashes[-1]["sha256"],
                            "record_index": row_index,
                            "slot_id": r.get("slot_id"),
                            "audit_row_id": row_identity,
                            "origin": origin,
                            "qid": qid,
                            "block": block,
                            "protocol": proto,
                            "view": view,
                            "actual_stop": finish,
                            "actual_token_count": n,
                            "first_tokenizer_eos_position": first_im_end,
                            "actual_model_eos_position": eos_n,
                            "actual_tail_token_ids": tokens[first_im_end - 1 :],
                            "tail_tokens_after_first_tokenizer_eos": tokens[first_im_end:],
                            "retained_text_delta": full[len(altered) :]
                            if full.startswith(altered)
                            else None,
                            "original_value_parseable": score["value_parseable"],
                            "at_first_248046_value_parseable": altered_score["value_parseable"],
                            "original_text_sha256": sha_bytes(full.encode()),
                            "alternate_text_sha256": sha_bytes(altered.encode()),
                            "original_final_json_state": json_state(full),
                            "alternate_final_json_state": json_state(altered),
                            "no_new_generation": True,
                            "read_only_boundary_sensitivity_not_replacement_score": True,
                        }
                    )
                record_total += 1
            if file_index % 60 == 59:
                print(
                    json.dumps(
                        {
                            "progress_block": block,
                            "groups_completed_in_block": file_index + 1,
                            "records_checked": record_total,
                            "runtime_check_failures": len(errors),
                        }
                    ),
                    flush=True,
                )
    row_file.close()
    failure_file.close()
    assert record_total == 3904 and len(seen_slots) == 3904 and len(seen_groups) == 848
    gates = {}
    for block in ("b0", "b1", "b2"):
        gate = load_json(snapshot / "engine" / block / "PROTOCOL_GATE.json")
        gates[block] = gate
        for proto in ("structured", "natural_work", "answer_only"):
            actual = aggregates[f"{block}|{proto}|MAIN"]
            assert (
                actual["records"] == 256
                and actual["parse:parseable"] == gate["protocols"][proto]["parseable"]
            )
    sensitivity_summary = {}
    for block in ("b0", "b1", "b2"):
        rows = [r for r in eos_sensitivity if r["block"] == block and r["view"] is None]
        baseline = gates[block]["protocols"]["structured"]["parseable"]
        delta = sum(
            r["at_first_248046_value_parseable"] - r["original_value_parseable"] for r in rows
        )
        sensitivity_summary[block] = {
            "main_affected_rows": len(rows),
            "baseline_parseable": baseline,
            "actual_read_only_boundary_delta": delta,
            "boundary_sensitive_parseable": baseline + delta,
            "hypothetical_upper_bound_if_every_affected_row_rescued": baseline + len(rows),
            "required_for_90_percent_gate": 231,
        }
    summary = {
        "schema_version": 1,
        "status": "PASS_NO_REPRODUCIBLE_INPUT_OR_STOP_RECORD_FAILURE"
        if not errors
        else "RUNTIME_RECORD_MISMATCH",
        "scope": (
            "2880 ENGINE panel rows, 512 parent-b1 MONITOR baseline rows, 512 unique "
            "cached ENGINE training rows; no new TEST output read"
        ),
        "source_commit": source_manifest["code_commit"],
        "source_files_verified_against_frozen_manifest": source_hashes,
        "snapshot_manifest_sha256": sha_file(snapshot / "ENGINE_SNAPSHOT_MANIFEST.json"),
        "run_identity_sha256": sha_file(snapshot / "RUN_IDENTITY.json"),
        "tokenizer_files_verified_against_prior": tokenizer_file_hashes,
        "input_manifest_sha256": sha_file(plan / "manifests/MODEL_INPUTS.jsonl"),
        "authorized_task_manifest_scope_counts": {"ENGINE": 64, "MONITOR": 64},
        "image_manifest_sha256": sha_file(prepared / "IMAGE_MANIFEST.jsonl"),
        "authorized_image_bytes_verified": sum(image_checks.values()),
        "record_count": record_total,
        "group_count": len(seen_groups),
        "distinct_prompt_contexts": len(prompt_cache),
        "checks_run": dict(row_checks),
        "runtime_check_failures": errors,
        "parse_and_stop_counts": {k: dict(v) for k, v in aggregates.items()},
        "registered_gates": gates,
        "inherited_eos_difference": {
            "model_text_config_eos_token_id": model_config["text_config"]["eos_token_id"],
            "generation_eos_token_ids": frozen_gen["eos_token_id"],
            "tokenizer_eos_token_id": auto.eos_token_id,
            "tokenizer_pad_token_id": auto.pad_token_id,
            "bos_token_id": auto.bos_token_id,
            "token_labels": {str(i): tok.id_to_token(i) for i in (90, 248044, 248045, 248046)},
            "rows_with_tokenizer_eos": len(eos_sensitivity),
            "ENGINE_panel_rows_with_tokenizer_eos": sum(
                r["origin"] == "ENGINE_PANEL" for r in eos_sensitivity
            ),
            "sensitivity_summary": sensitivity_summary,
        },
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("tokenizers", "transformers", "jinja2", "numpy", "torch")
        },
        "server_versions": {
            "transformers": prior["transformers_version"],
            "torch": prior["torch_version"],
            "packages": prior["packages"],
        },
        "python": sys.version,
        "new_model_calls": 0,
        "new_optimizer_steps": 0,
        "weights_loaded": False,
        "new_sampling": False,
        "runtime_source_changed": False,
        "snapshot_written": False,
        "limits": [
            (
                "Exact original model inputs were checked by reconstructing the frozen "
                "native chat and 768 image placeholders, and matching persisted "
                "input_ids/tensor hashes; no GPU forward was replayed."
            ),
            (
                "Image bytes and recorded geometry/pixel-route consistency were checked, but "
                "pixel tensors were not independently rerun through the image processor in "
                "this audit."
            ),
            (
                "Sampler selected log probabilities were checked for preserved length/finite "
                "values; raw logits are not present for independent recomputation."
            ),
            (
                "Recorded full-trajectory prefixes were decoded; no short-budget "
                "counterfactual generation is claimed."
            ),
            (
                "EOS boundary sensitivity only truncates existing token IDs and rescoring "
                "unchanged ENGINE text; original scores and outputs remain immutable."
            ),
        ],
        "audit_elapsed_host_seconds": time.time() - started,
    }
    (out / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    (out / "EOS_BOUNDARY_SENSITIVITY.json").write_text(
        json.dumps(
            {
                "inherited_config_difference": summary["inherited_eos_difference"],
                "records": eos_sensitivity,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    (out / "SOURCE_GROUPS.json").write_text(
        json.dumps(group_hashes, indent=2, sort_keys=True) + "\n"
    )
    shutil.copyfile(__file__, out / "audit_engine_runtime.py")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "records": record_total,
                "groups": len(seen_groups),
                "errors": len(errors),
                "output": str(out),
                "eos_boundary_delta": sensitivity_summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
