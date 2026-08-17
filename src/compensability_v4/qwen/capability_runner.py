"""Deterministic Qwen execution and immutable artifact writing for Phase 1."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from pathlib import Path

from compensability_v4.diagnostics.capability_chain import (
    CapabilityCall,
    CapabilityExecutionRecord,
    evaluate_capability_call,
)


def _prepared_batch(processor: object, messages: tuple[Mapping[str, str], ...]) -> object:
    template = getattr(processor, "apply_chat_template", None)
    if not callable(template) or not callable(processor):
        raise TypeError("processor must support chat templating and tokenization")
    rendered = template(list(messages), tokenize=False, add_generation_prompt=True)
    return processor(text=[rendered], padding=True, return_tensors="pt")


def _mapping(batch: object) -> dict[str, object]:
    if isinstance(batch, Mapping):
        return dict(batch)
    keys = getattr(batch, "keys", None)
    if callable(keys):
        return {key: batch[key] for key in keys()}  # type: ignore[index]
    raise TypeError("prepared processor batch must be mapping-like")


def _decode_one(
    model: object,
    processor: object,
    call: CapabilityCall,
    *,
    max_new_tokens: int,
) -> str:
    shortcut = getattr(model, "complete_text", None)
    if callable(shortcut):
        output = shortcut(call.messages)
        if not isinstance(output, str):
            raise RuntimeError("test runtime complete_text() must return text")
        return output
    if max_new_tokens != 32:
        raise RuntimeError("Phase 1 max_new_tokens must remain frozen at 32")
    batch = _prepared_batch(processor, call.messages)
    device = getattr(model, "device", None)
    if device is not None and callable(getattr(batch, "to", None)):
        batch = batch.to(device)
    inputs = _mapping(batch)
    input_ids = inputs.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape"):
        raise RuntimeError("prepared Qwen batch has no tensor input_ids")
    prompt_length = int(input_ids.shape[-1])
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise RuntimeError("loaded Qwen runtime has no generate() method")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - server dependency
        raise RuntimeError("PyTorch is required for Phase 1 execution") from error
    with torch.inference_mode():
        output_ids = generate(**inputs, max_new_tokens=32, do_sample=False)
    continuation = output_ids[:, prompt_length:]
    decoder = getattr(processor, "batch_decode", None)
    if not callable(decoder):
        tokenizer = getattr(processor, "tokenizer", None)
        decoder = getattr(tokenizer, "batch_decode", None)
    if not callable(decoder):
        raise RuntimeError("processor runtime has no batch_decode() method")
    decoded = decoder(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
        raise RuntimeError("Qwen decoder did not return exactly one text completion")
    return decoded[0]


def execute_capability_calls(
    model: object,
    processor: object,
    calls: Iterable[CapabilityCall],
    *,
    max_new_tokens: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[CapabilityExecutionRecord, ...]:
    """Execute every frozen call exactly once, without retries or result gating."""

    frozen = tuple(calls)
    if not frozen or len({call.call_id for call in frozen}) != len(frozen):
        raise ValueError("capability calls must be non-empty with unique identifiers")
    rows: list[CapabilityExecutionRecord] = []
    for completed, call in enumerate(frozen, start=1):
        raw_output = _decode_one(
            model,
            processor,
            call,
            max_new_tokens=max_new_tokens,
        )
        rows.append(evaluate_capability_call(call, raw_output))
        if progress is not None:
            progress(completed, len(frozen))
    return tuple(rows)


def _parsed_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def write_capability_outputs(
    output_directory: Path,
    *,
    records: Iterable[CapabilityExecutionRecord],
    summaries: Iterable[Mapping[str, object]],
    gaps: Mapping[str, object],
) -> None:
    """Write exactly the three frozen Phase-1 artifacts and never overwrite."""

    frozen_records = tuple(records)
    frozen_summaries = tuple(dict(row) for row in summaries)
    if not frozen_records or not frozen_summaries:
        raise ValueError("Phase 1 records and summaries must be non-empty")
    paths = {
        "per_scene": output_directory / "per_scene.csv",
        "summary": output_directory / "summary_by_family.csv",
        "gaps": output_directory / "paired_gaps.json",
    }
    if output_directory.is_symlink() or any(
        path.exists() or path.is_symlink() for path in paths.values()
    ):
        raise FileExistsError("refusing to overwrite an existing Phase 1 artifact")
    output_directory.mkdir(parents=True, exist_ok=True)
    per_scene = []
    for record in frozen_records:
        row = asdict(record)
        row["task_type"] = record.task_type.value
        row["parsed_output"] = _parsed_text(record.parsed_output)
        per_scene.append(row)
    with paths["per_scene"].open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_scene[0]))
        writer.writeheader()
        writer.writerows(per_scene)
    with paths["summary"].open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frozen_summaries[0]))
        writer.writeheader()
        writer.writerows(frozen_summaries)
    with paths["gaps"].open("x", encoding="utf-8") as stream:
        json.dump(dict(gaps), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


__all__ = ["execute_capability_calls", "write_capability_outputs"]
