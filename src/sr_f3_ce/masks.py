"""Audited field masks from the tokenizer's *actual complete sequence* offsets.

No length obtained by separately encoding a prefix or a field is used here.
The only boundary relaxation is ownership of JSON punctuation/whitespace outside
string content. Tokens containing semantic content from two fields fail closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence

SERIALIZATIONS = ("canonical", "field_newlines_v1")


class MaskAlignmentError(ValueError):
    """A full native sequence cannot express the registered field separation."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate teacher JSON key")
        result[key] = value
    return result


def _decoder():
    return json.JSONDecoder(
        object_pairs_hook=_unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite teacher value")),
    )


def target_spans(text):
    """Read exact root-field boundaries without searching inside work strings."""
    if not isinstance(text, str) or not text.startswith("{"):
        raise ValueError("Teacher must be a complete JSON object")
    decoder = _decoder()
    obj, end = decoder.raw_decode(text)
    if end != len(text) or not isinstance(obj, dict):
        raise ValueError("Teacher has trailing data or is not an object")
    keys = list(obj)
    if keys not in (["evidence", "answer"], ["evidence", "work", "answer"]):
        raise ValueError("Teacher order must be evidence, optional work, answer")
    if not isinstance(obj["evidence"], list) or (
        "work" in obj and not isinstance(obj["work"], str)
    ):
        raise ValueError("Invalid teacher evidence/work type")
    spans, cursor = {}, 1
    for expected in keys:
        while text[cursor].isspace():
            cursor += 1
        key_start = cursor
        key, cursor = decoder.raw_decode(text, cursor)
        if key != expected:
            raise ValueError("Teacher root field mismatch")
        while text[cursor].isspace():
            cursor += 1
        if text[cursor] != ":":
            raise ValueError("Teacher field lacks colon")
        cursor += 1
        while text[cursor].isspace():
            cursor += 1
        value_start = cursor
        _, cursor = decoder.raw_decode(text, cursor)
        if key == "work":
            spans["work"] = [value_start, cursor]
        elif key == "answer":
            spans["answer"] = [key_start, len(text)]
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if key != "answer":
            if text[cursor] != ",":
                raise ValueError("Teacher field lacks separator")
            cursor += 1
    return obj, spans


def serialize_target(target, serialization="canonical"):
    """Validate original bytes, or apply one uniform, explicitly named fallback."""
    if serialization not in SERIALIZATIONS:
        raise ValueError("Unknown support serialization")
    if not isinstance(target, Mapping) or not {"text", "spans"} <= set(target):
        raise ValueError("Teacher text and character spans are required")
    obj, spans = target_spans(target["text"])
    if dict(target["spans"]) != spans or ("object" in target and target["object"] != obj):
        raise ValueError("Teacher object/character spans differ from its text")
    if serialization == "canonical":
        text = target["text"]
    else:
        text = "{" + ",\n".join(_canonical(k) + ":" + _canonical(v) for k, v in obj.items()) + "}"
    _, selected_spans = target_spans(text)
    return {
        "text": text,
        "spans": selected_spans,
        "object": obj,
        "serialization": serialization,
        "source_text_sha256": hashlib.sha256(target["text"].encode()).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def structural_characters(text):
    """Mark only actual JSON delimiters, never punctuation inside string values."""
    structural = [False] * len(text)
    quoted, escaped = False, False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
                structural[index] = True
        elif char == '"':
            quoted = True
            structural[index] = True
        elif char in "{}[]:," or char.isspace():
            structural[index] = True
    if quoted:
        raise ValueError("Unclosed teacher JSON string")
    return structural


def field_masks(target, offsets, *, target_start=0, token_ids=None):
    """Assign every actual full-sequence token exactly once, retaining audit data.

    ``target_start`` is in the native processor's expanded-text coordinates.
    ``offsets`` must be returned with the same ``input_ids`` sent to the model.
    Zero offsets (special/padding tokens) are always context. Overlapping byte
    fallback offsets are allowed; offsets may never go backwards.
    """
    text, spans = target["text"], target["spans"]
    _, actual_spans = target_spans(text)
    if dict(spans) != actual_spans or type(target_start) is not int or target_start < 0:
        raise ValueError("Invalid target spans or native character origin")
    offsets = [tuple(pair) for pair in offsets]
    if token_ids is not None and len(token_ids) != len(offsets):
        raise ValueError("Native input IDs and offsets differ in length")
    structural = structural_characters(text)
    owners, adjustments, previous = [], [], (0, 0)
    for token_index, (start, end) in enumerate(offsets):
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise MaskAlignmentError("Invalid native token offset")
        if (start, end) != (0, 0):
            if start < previous[0] or end < previous[1]:
                raise MaskAlignmentError("Native offsets are not monotone")
            previous = start, end
        if end > target_start + len(text):
            raise MaskAlignmentError("Token extends beyond the complete teacher")
        lo, hi = max(0, start - target_start), min(len(text), end - target_start)
        candidates = {
            field for field, (a, b) in spans.items() if hi > lo and max(a, lo) < min(b, hi)
        }
        if not candidates:
            owners.append("context")
            continue
        semantic = set()
        if start < target_start:
            semantic.add("context")
        for position in range(lo, hi):
            if structural[position]:
                continue
            semantic.add(
                next(
                    (field for field, (a, b) in spans.items() if a <= position < b),
                    "context",
                )
            )
        if len(semantic) > 1:
            raise MaskAlignmentError(
                f"Token {token_index} crosses semantic fields {sorted(semantic)} at {start}:{end}"
            )
        owner = (
            next(iter(semantic)) if semantic else ("answer" if "answer" in candidates else "work")
        )
        owners.append(owner)
        contained = owner in spans and (
            spans[owner][0] <= start - target_start <= end - target_start <= spans[owner][1]
        )
        if not contained:
            adjustments.append(
                {
                    "token_index": token_index,
                    "offset": [start, end],
                    "owner": owner,
                    "reason": "only_json_structure_crosses_field_boundary",
                    "text": text[lo:hi],
                }
            )
    answer = [int(owner == "answer") for owner in owners]
    work = [int(owner == "work") for owner in owners]
    if not any(answer) or ("work" in spans and not any(work)):
        raise MaskAlignmentError("Empty answer/work mask is not a zero loss")
    return {
        "answer_mask": answer,
        "work_mask": work,
        "token_owners": owners,
        "offsets": [list(pair) for pair in offsets],
        "token_ids": list(token_ids) if token_ids is not None else None,
        "target_start": target_start,
        "target_text": text,
        "spans": dict(spans),
        "boundary_adjustments": adjustments,
        "offset_source": "actual_complete_native_sequence",
    }


def tokenize_with_offsets(tokenizer, target, *, prefix=""):
    """CPU audit adapter for real fast tokenizers, with no separately encoded fields."""
    encoded = tokenizer(
        prefix + target["text"],
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    if "offset_mapping" not in encoded:
        raise MaskAlignmentError("A real fast-tokenizer offset mapping is required")
    return field_masks(
        target,
        encoded["offset_mapping"],
        target_start=len(prefix),
        token_ids=encoded["input_ids"],
    )


def freeze_serialization(targets: Sequence, encode: Callable):
    """Choose once across *all* targets; never fall back differently per arm/item.

    ``encode(serialized_target, index)`` must audit the complete native sequence
    and return a mask receipt. A failed canonical alignment is retained in the
    freeze receipt. Failure of the uniform fallback is a technical blocker.
    """
    if not targets:
        raise ValueError("Cannot freeze an empty support corpus")
    failures = []
    for serialization in SERIALIZATIONS:
        receipts = []
        for index, target in enumerate(targets):
            selected = serialize_target(target, serialization)
            try:
                receipt = encode(selected, index)
            except MaskAlignmentError as error:
                failures.append(
                    {"serialization": serialization, "index": index, "error": str(error)}
                )
                break
            receipts.append(
                {
                    "index": index,
                    "target_text_sha256": selected["text_sha256"],
                    "mask_audit_sha256": hashlib.sha256(_canonical(receipt).encode()).hexdigest(),
                }
            )
        else:
            return {
                "status": "PASS",
                "serialization": serialization,
                "target_count": len(targets),
                "canonical_failures": failures,
                "targets": receipts,
                "selection_basis": "token_alignment_only_no_model_scores",
            }
    raise MaskAlignmentError("Both uniform serializations failed: " + _canonical(failures))
