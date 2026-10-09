"""SR-F1.1 prompt prefill and token-boundary JSON termination.

This module never extracts objects from prose or changes the semantic scorer.
The closing character is an offset in the full prefill-plus-generation text.
"""

from __future__ import annotations

AMENDMENT_ID = "SR-F1.1-20261010"
PREFILL = "{"
PREFILL_TOKEN_ID = 90
AMENDMENT_RECORD_FIELDS = (
    "protocol_amendment_id",
    "assistant_prefill",
    "generated_text",
    "full_decoded_text",
    "balanced_cut_char",
    "balanced_token_count",
    "format_protocol_error",
)


def amendment_enabled(amendment):
    if amendment is None:
        return False
    if not isinstance(amendment, dict) or amendment.get("id") != AMENDMENT_ID:
        raise PermissionError("Unregistered generation protocol amendment")
    return True


def balanced_cut(text):
    """Return the first closing top-level object offset, aware of JSON strings."""
    if not isinstance(text, str) or not text.startswith(PREFILL):
        return None
    depth, in_string, escaped = 0, False, False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def decoded_protocol(generated_text):
    """Preserve invalid suffixes and classify them before strict scoring."""
    if not isinstance(generated_text, str):
        raise TypeError("Decoded generated text must be a string")
    full = PREFILL + generated_text
    cut = balanced_cut(full)
    error = "unbalanced" if cut is None else "trailing_after_close" if full[cut:].strip() else None
    return dict(
        protocol_amendment_id=AMENDMENT_ID,
        assistant_prefill=PREFILL,
        generated_text=generated_text,
        full_decoded_text=full,
        balanced_cut_char=cut,
        format_protocol_error=error,
        raw_text=full if error else full[:cut],
    )


def decode_generated(tokenizer, tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def first_balanced_token(tokenizer, tokens):
    """Independently replay every token prefix; never stop between token IDs."""
    for count in range(1, len(tokens) + 1):
        cut = balanced_cut(PREFILL + decode_generated(tokenizer, tokens[:count]))
        if cut is not None:
            return count, cut
    return None, None


class BalancedJSONStop:
    """A Transformers-compatible stopping callable for one independent context."""

    def __init__(self, tokenizer, prompt_len):
        if type(prompt_len) is not int or prompt_len <= 0:
            raise ValueError("Positive prompt length required")
        self.tokenizer, self.prompt_len = tokenizer, prompt_len
        self.balanced_token_count = None
        self.balanced_cut_char = None

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        if input_ids.shape[0] != 1:
            raise PermissionError("Balanced JSON stopping requires one independent sequence")
        tokens = input_ids[0, self.prompt_len :].tolist()
        cut = balanced_cut(PREFILL + decode_generated(self.tokenizer, tokens))
        if cut is not None:
            if self.balanced_token_count is not None and (
                self.balanced_token_count != len(tokens) or self.balanced_cut_char != cut
            ):
                raise RuntimeError("Generation continued after its balanced stopping token")
            self.balanced_token_count, self.balanced_cut_char = len(tokens), cut
        # The HF criteria list combines a per-batch bool tensor with EOS/length.
        return input_ids.new_full((1,), cut is not None, dtype=torch.bool)


def validate_record(record, expected_amendment_id=None):
    """Validate durable text/trajectory metadata; malformed records are technical errors.

    No tokenization is performed here. Generation independently verifies the first
    balanced token with the native tokenizer before publishing the record.
    """
    present = set(AMENDMENT_RECORD_FIELDS).intersection(record) or any(
        "protocol_amendment_id" in record.get(key, {}) for key in ("image_routing", "input_routing")
    )
    if not present and expected_amendment_id is None:
        return False
    if expected_amendment_id not in (None, AMENDMENT_ID):
        raise PermissionError("Unknown expected protocol amendment")
    if set(AMENDMENT_RECORD_FIELDS) - set(record):
        raise PermissionError("Amended generation record metadata is incomplete")
    if record["protocol_amendment_id"] != AMENDMENT_ID:
        raise PermissionError("Generation protocol amendment identity differs")
    expected = decoded_protocol(record["generated_text"])
    for key, value in expected.items():
        if record.get(key) != value:
            raise PermissionError("Amended generation record differs: " + key)
    tokens = record.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or record.get("raw_tokens") != tokens
        or record.get("completion_token_count") != len(tokens)
        or len(record.get("old_logprobs", ())) != len(tokens)
        or len(record.get("sampler_logprobs", ())) != len(tokens)
    ):
        raise PermissionError("Amended generated-token trajectory is incomplete")
    balanced = expected["balanced_cut_char"] is not None
    if balanced:
        if (
            record["balanced_token_count"] != len(tokens)
            or record.get("finish_reason") != "balanced"
            or record.get("truncated") is not False
        ):
            raise PermissionError("Balanced stopping token differs from trajectory boundary")
    elif (
        record["balanced_token_count"] is not None
        or record.get("finish_reason") not in ("eos", "length")
        or record.get("truncated") is not (record.get("finish_reason") == "length")
    ):
        raise PermissionError("Unbalanced termination metadata differs")
    for name in ("input_routing", "image_routing"):
        routing = record.get(name, {})
        if (
            routing.get("protocol_amendment_id") != AMENDMENT_ID
            or routing.get("protocol_amendment", {}).get("id") != AMENDMENT_ID
            or routing.get("protocol") != "evidence_answer"
            or routing.get("assistant_prefill") != PREFILL
            or routing.get("assistant_prefill_token_ids") != [PREFILL_TOKEN_ID]
            or routing.get("base_prompt_token_count") != record.get("prompt_token_count", 0) - 1
        ):
            raise PermissionError("Amended prompt routing differs: " + name)
    return True


def score_record(record, task):
    """Call the unchanged strict scorer, enforcing the amended protocol boundary."""
    from .contract import score

    amended = validate_record(record)
    error = record["format_protocol_error"] if amended else None
    result = score("" if error else record["raw_text"], task["world"], task["query"])
    if error:
        result["reason"] = error
    return result
