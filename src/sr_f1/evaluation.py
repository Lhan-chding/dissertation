"""Fixed evaluation slots and append-only raw inference for SR-F1.

This module does not expose aggregate test scores to training. Every answer-only
slot invokes generation independently; structural P/J are never assigned to it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

ARMS = ("A", "J", "PART", "DEC", "GATE")
SEEDS = (71001, 71002, 71003)
BASELINE = "SRF1_COMMON_START"
VIEWS = ("keys_hint", "gold_values", "text_values", "neutral_hint", "blank_image")
TEST_POOLS = ("TEST_ID", "TEST_COMPOSITION", "TEST_LANGUAGE", "TEST_RENDER")


def encoded(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed(*parts):
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big"
    ) % (2**63 - 1)


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def model_ids():
    return [BASELINE] + [f"SRF1_{arm}_s{seed}" for seed in SEEDS for arm in ARMS]


def generation_config(protocol, samples=4):
    if protocol not in ("evidence_answer", "answer_only", "plain_answer"):
        raise ValueError("Unknown evaluation protocol")
    return dict(
        do_sample=protocol != "plain_answer",
        temperature=1,
        top_p=1,
        top_k=0,
        max_new_tokens=768 if protocol == "evidence_answer" else 128,
        num_return_sequences=1,
    )


def iter_model_slots(tasks, train_fit_qids, model_id, *, stage="final", step=None):
    """Order is frozen by qid; MONITOR blocks are precisely 16 prompts x 8."""
    if model_id not in model_ids() or stage not in ("baseline", "final", "train_fit"):
        raise ValueError("Unregistered model/stage")
    step = (0 if model_id == BASELINE else 96) if step is None else step
    if model_id == BASELINE and step != 0:
        raise ValueError("Common start has only checkpoint 0")
    if model_id != BASELINE and step not in (32, 64, 96):
        raise ValueError("Evaluation checkpoint is not registered")
    if stage == "baseline" and model_id != BASELINE:
        raise ValueError("Only common start may run baseline stage")
    if stage != "train_fit" and model_id != BASELINE and step != 96:
        raise ValueError("Only TRAIN_FIT is allowed at intermediate checkpoints")
    rows = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
    task_map = {t["qid"]: t for t in rows}
    if len(task_map) != len(rows) or len(set(train_fit_qids)) != 32:
        raise ValueError("Duplicate tasks or incorrect frozen TRAIN_FIT panel")
    panels = []
    if stage != "train_fit":
        monitor = sorted((t for t in rows if t["pool"] == "MONITOR"), key=lambda t: t["qid"])
        if len(monitor) != 512:
            raise ValueError("MONITOR must contain all 512 fixed prompts")
        panels.append(("MONITOR", monitor, ("evidence_answer",), 8, None))
        v0 = [t for t in monitor if t["variant"] == "v0"]
        if len(v0) != 64:
            raise ValueError("Diagnostic panel must contain 64 original prompts")
        for view in VIEWS:
            panels.append(("MONITOR_DIAGNOSTIC_ONLY", v0, ("evidence_answer",), 8, view))
    fit = [task_map[qid] for qid in train_fit_qids]
    if any(t["pool"] != "TRAIN" for t in fit):
        raise ValueError("TRAIN_FIT must use the frozen TRAIN qids")
    panels.append(("TRAIN_FIT", fit, ("evidence_answer",), 4, None))
    if stage == "final":
        for pool in TEST_POOLS:
            panel = sorted((t for t in rows if t["pool"] == pool), key=lambda t: t["qid"])
            if len(panel) != (2048 if pool == "TEST_ID" else 128):
                raise ValueError(f"Incomplete {pool}")
            panels.append((pool, panel, ("evidence_answer", "answer_only"), 4, None))
    for pool, panel, protocols, k, view in panels:
        for prompt_index, task in enumerate(panel):
            for protocol in protocols:
                for draw in range(k):
                    identity = [
                        model_id,
                        step,
                        pool,
                        protocol,
                        view or "original",
                        task["qid"],
                        draw,
                    ]
                    yield dict(
                        slot_id="|".join(map(str, identity)),
                        model_id=model_id,
                        step=step,
                        pool=pool,
                        protocol=protocol,
                        view=view,
                        qid=task["qid"],
                        root_id=task["root_id"],
                        family=task["family"],
                        chart=task["chart"],
                        variant=task["variant"],
                        draw=draw,
                        samples_per_prompt=k,
                        monitor_block=prompt_index // 16 if pool == "MONITOR" else None,
                        sampling_seed=_seed(
                            "SRF1_EVALUATION", step, pool, protocol, view, task["qid"], draw
                        ),
                        generation=generation_config(protocol),
                    )


def iter_evaluation_slots(tasks, train_fit_qids):
    for model_id in model_ids():
        yield from iter_model_slots(tasks, train_fit_qids, model_id)
        if model_id != BASELINE:
            for step in (32, 64):
                yield from iter_model_slots(
                    tasks, train_fit_qids, model_id, stage="train_fit", step=step
                )


def neutral_padding(original, target_text, candidate, tokenizer):
    """Match extra tokenizer tokens in the actual complete prompt, never by words.

    Candidate prefixes are selected deterministically using only token counts;
    there are no model responses or gold correctness decisions in this routine.
    """
    if not candidate or not isinstance(candidate, str):
        raise ValueError("Missing frozen neutral template")

    def tokenize(text):
        return list(tokenizer.encode(text, add_special_tokens=False))

    base = len(tokenize(original))
    target = len(tokenize(original + "\n\n" + target_text)) - base
    if target < 0:
        raise ValueError("Invalid tokenizer target")
    repeated = candidate
    while len(tokenize(original + "\n\n" + repeated)) - base < target + 4:
        repeated += candidate
        if len(repeated) > 65536:
            raise ValueError("Unable to construct bounded neutral padding")
    # Decode token prefixes, then re-tokenize the COMPLETE prompt. BPE boundaries
    # can change at the insertion point; comparing only suffix encodings is wrong.
    ids = tokenize(repeated)
    best = None
    for n in range(max(0, target - 8), min(len(ids), target + 8) + 1):
        suffix = tokenizer.decode(ids[:n], skip_special_tokens=False)
        if not repeated.startswith(suffix):
            continue
        count = len(tokenize(original + "\n\n" + suffix)) - base
        item = (abs(count - target), n, suffix, count)
        if best is None or item[:2] < best[:2]:
            best = item
    if best is None or best[0] > 1:
        raise ValueError("Neutral hint failed the frozen <=1 token matching condition")
    return original + "\n\n" + best[2], dict(
        target_added_tokens=target,
        neutral_added_tokens=best[3],
        token_difference=best[3] - target,
        selection="fixed_template_token_prefix_no_model_based_selection",
    )


def prepare_input(slot, inputs, diagnostic_inputs, tokenizer=None):
    row = inputs[slot["qid"]]
    if slot["view"]:
        diagnostic = diagnostic_inputs[(slot["qid"], slot["view"])]
        if diagnostic["allowed_in_training"] is not False:
            raise ValueError("Diagnostic training isolation violated")
        output = {key: diagnostic[key] for key in ("text", "image_file", "information_granted")}
        if slot["view"] == "neutral_hint":
            if tokenizer is None:
                raise ValueError("Native frozen tokenizer is required for neutral_hint")
            pad = diagnostic["runtime_neutral_padding"]
            output["text"], output["token_matching"] = neutral_padding(
                diagnostic["text"], pad["target_text"], pad["candidate"], tokenizer
            )
        return output
    return dict(
        text=row["plain_text"] if slot["protocol"] == "answer_only" else row["text"],
        image_file=row["image_file"],
        information_granted="original_chart_and_question",
    )


def score_response(raw_text, task, protocol):
    from .contract import execute, fraction_of, load_json_strict, score

    if protocol == "evidence_answer":
        return score(raw_text, task["world"], task["query"])
    if protocol != "answer_only":
        raise ValueError("ChartQA uses its separate author-compatible scorer")
    result = dict(L_json=0, L_answer=0, A=0, reason="invalid_json")
    try:
        obj = load_json_strict(raw_text)
        if not isinstance(obj, dict) or not set(obj).issubset({"answer"}):
            return result
        result["L_json"] = 1
        result["reason"] = "missing_answer"
        if "answer" in obj:
            answer = None if obj["answer"] is None else fraction_of(obj["answer"], text=True)
            result.update(
                L_answer=1, A=int(answer == execute(task["world"], task["query"])), reason="scored"
            )
    except (ValueError, TypeError, OverflowError, RecursionError, ZeroDivisionError):
        pass
    return result


def validate_raw(row, slot):
    for key, value in slot.items():
        if row.get(key) != value:
            raise ValueError(f"Raw identity mismatch: {key}")
    if (
        row.get("status") != "GENERATED"
        or row.get("generation_status", "COMPLETE") != "COMPLETE"
        or row.get("technical_validation_errors")
        or not isinstance(row.get("raw_text"), str)
    ):
        raise ValueError("Technical missing response is not a generated answer")
    tokens, logprobs = row.get("tokens"), row.get("old_logprobs")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
        raise ValueError("Missing raw generated token ids")
    if not isinstance(logprobs, list) or len(tokens) != len(logprobs):
        raise ValueError("Missing token-aligned sampling log probabilities")
    if any(p is None or not math.isfinite(float(p)) for p in logprobs):
        raise ValueError("Nonfinite sampling probabilities")
    if type(row.get("truncated")) is not bool:
        raise ValueError("Missing truncation classification")
    for field in ("input_hash", "model_identity", "adapter_identity"):
        if not row.get(field):
            raise ValueError(f"Missing raw provenance: {field}")
    if row.get("image_file") is not None and not row.get("image_hash"):
        raise ValueError("Missing image hash")


def evaluation_model_identity(identity):
    """Compare model/software identity while retaining hardware per attempt.

    Slurm may move a lease to another card of the same audited class. Physical
    UUID, bus and hostname do not change the model or sampling contract.
    """
    if not isinstance(identity, dict):
        return identity
    fixed = (
        "base_model_weights_hash",
        "model_hash",
        "adapter_hash",
        "adapter_file_hashes",
        "trainable_state_hash",
        "model_id",
        "step",
        "freeze_sha256",
        "processor_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "transformers_version",
        "torch_version",
        "dtype",
        "attention_backend",
        "linear_kernel_identity_hash",
        "generation_config_expanded",
        "canvas_pixels",
        "processor_target_pixels",
        "determinism",
    )
    result = {k: identity[k] for k in fixed if k in identity}
    if not result:
        result = {k: v for k, v in identity.items() if k != "hardware"}
    if "hardware" in identity:
        hardware = identity["hardware"]
        result["hardware_class_and_software"] = {
            k: hardware[k]
            for k in (
                "cuda_visible_device_count",
                "cuda_name",
                "cuda_total_memory_bytes",
                "compute_capability",
                "cuda_runtime",
                "driver_version",
                "memory_mib",
                "name",
            )
            if k in hardware
        }
    return result


def evaluation_checkpoint(root, raw_path, completed, reason="PREEMPTION"):
    """Immutable raw prefix is a full evaluation state; each future slot has its own seed."""
    root, raw_path = Path(root), Path(raw_path)
    checksum = sha_file(raw_path)
    snapshot = raw_path.parent / "checkpoints" / (raw_path.stem + "-" + checksum + ".jsonl")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copyfile(raw_path, snapshot)
    if sha_file(snapshot) != checksum:
        raise ValueError("Evaluation checkpoint snapshot mismatch")
    return dict(
        status="CHECKPOINTED",
        reason=reason,
        artifacts=[str(snapshot.relative_to(root))],
        metadata=dict(
            full_state=True,
            identity_verified=True,
            generated_slots=completed,
            raw_prefix_sha256=checksum,
            resume_raw_path=str(raw_path.relative_to(root)),
        ),
    )


def evaluate_slots(
    runtime, slots: Iterable[dict], inputs, diagnostic_inputs, root, raw_path, boundary=None
):
    """Use an already loaded checkpoint; write raw only and fsync each completion."""
    root, raw_path = Path(root).resolve(), Path(raw_path).resolve()
    if not raw_path.is_relative_to(root):
        raise ValueError("Raw output must be inside run root")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    if raw_path.exists():
        for row in read_jsonl(raw_path):
            if row["slot_id"] in seen:
                raise ValueError("Duplicate persisted evaluation slot")
            seen[row["slot_id"]] = row
    completed = 0
    identity = getattr(runtime, "stable_model_identity", None)
    if identity is None:
        identity = getattr(runtime, "identity", None)
    if callable(identity):
        identity = identity()
    tokenizer = getattr(runtime, "tokenizer", None)
    if tokenizer is None and getattr(runtime, "processor", None) is not None:
        tokenizer = runtime.processor.tokenizer
    with raw_path.open("a") as stream:
        for slot in slots:
            if (boundary and boundary["requested"]) or (root / "STOP").exists():
                reason = "STOP_REQUESTED" if (root / "STOP").exists() else "PREEMPTION"
                return evaluation_checkpoint(root, raw_path, completed + len(seen), reason)
            request = prepare_input(slot, inputs, diagnostic_inputs, tokenizer)
            image_path = None
            if request["image_file"] is not None:
                image_path = (root / request["image_file"]).resolve()
                if not image_path.is_relative_to(root):
                    raise ValueError("Image path escaped run root")
            image_hash = sha_file(image_path) if image_path else None
            input_hash = hashlib.sha256(
                encoded(dict(text=request["text"], image_hash=image_hash)).encode()
            ).hexdigest()
            if slot["slot_id"] in seen:
                previous = seen.pop(slot["slot_id"])
                validate_raw(previous, slot)
                if previous["input_hash"] != input_hash:
                    raise ValueError("Resumed input identity changed")
                if identity and evaluation_model_identity(
                    previous["model_identity"]
                ) != evaluation_model_identity(identity):
                    raise ValueError("Resumed model identity changed")
                completed += 1
                continue
            persisted = []

            def persist(
                completion,
                slot=slot,
                request=request,
                input_hash=input_hash,
                image_hash=image_hash,
                persisted=persisted,
            ):
                row = dict(completion)
                row.update(slot)
                row.update(request)
                row.update(status="GENERATED", input_hash=input_hash, image_hash=image_hash)
                if "model_identity" not in row and identity:
                    row["model_identity"] = identity
                if "adapter_identity" not in row:
                    adapter = getattr(runtime, "adapter_identity", None)
                    row["adapter_identity"] = adapter() if callable(adapter) else adapter
                # Durability precedes validation. A real truncated/invalid response
                # or broken sampling probability remains in the raw ledger.
                stream.write(encoded(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                persisted.append(row)

            runtime.generate(
                text=request["text"],
                image_path=image_path,
                run_root=root,
                seed=slot["sampling_seed"],
                generation=slot["generation"],
                on_completion=persist,
            )
            if not persisted:
                raise RuntimeError("Runtime did not durably deliver the actual generation callback")
            if len(persisted) != 1:
                raise RuntimeError("Runtime delivered multiple generations for one slot")
            validate_raw(persisted[0], slot)
            completed += 1
            if (boundary and boundary["requested"]) or (root / "STOP").exists():
                reason = "STOP_REQUESTED" if (root / "STOP").exists() else "PREEMPTION"
                return evaluation_checkpoint(root, raw_path, completed, reason)
    if seen:
        raise ValueError("Raw file contains unregistered or out-of-stage slots")
    return dict(
        status="COMPLETE",
        artifacts=[str(raw_path.relative_to(root))],
        metadata=dict(
            generated_slots=completed, scores_released=False, raw_sha256=sha_file(raw_path)
        ),
    )


def evaluate_model(runtime, root, model_id, stage="final", step=None, boundary=None):
    root = Path(root)
    tasks = {r["qid"]: r for r in read_jsonl(root / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")}
    inputs = {r["qid"]: r for r in read_jsonl(root / "manifests/MODEL_INPUTS.jsonl")}
    diagnostics = {
        (r["qid"], r["view"]): r
        for r in read_jsonl(root / "manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl")
    }
    fit = json.loads((root / "manifests/TRAIN_FIT_QIDS.json").read_text())
    step = (0 if model_id == BASELINE else 96) if step is None else step
    # Stages intentionally use separate files; final common-start slots that were
    # already measured in baseline are omitted and verified at unified release.
    slots = iter_model_slots(tasks, fit, model_id, stage=stage, step=step)
    if stage == "final":
        slots = (
            s
            for s in slots
            if (s["pool"] in TEST_POOLS if model_id == BASELINE else s["pool"] != "TRAIN_FIT")
        )
    path = root / "raw/evaluation" / f"{model_id}_step{step}_{stage}.jsonl"
    if boundary is not None:
        return evaluate_slots(runtime, slots, inputs, diagnostics, root, path, boundary)
    from .training import stop_at_committed_boundary

    with stop_at_committed_boundary() as boundary:
        return evaluate_slots(runtime, slots, inputs, diagnostics, root, path, boundary)
