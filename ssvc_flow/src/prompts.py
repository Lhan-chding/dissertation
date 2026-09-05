"""Versioned literal N-track templates. Only task inputs are read from scenes."""

import hashlib
import json

SYSTEM_PROMPT = (
    "You are repairing a four-integer chart record. Return only one JSON array\n"
    "containing four integers from 0 to 99. Do not include an explanation."
)
USER_TEMPLATE = """The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is {observed_json}.
Exactly one value in this observed record is wrong.
The following relationships are reliable:
{cue_text}
The downstream calculation is: {operation_expression}.
Recover the correct record. Return [a,b,c,d], not the downstream answer."""
OPERATIONS = {
    "sum4": "a+b+c+d",
    "difference_pairs": "(a+b)-(c+d)",
    "range4": "max(a,b,c,d)-min(a,b,c,d)",
}
TEMPLATE_VERSION = "N-v1-2026-09-05"


def cue_text(cue):
    family = cue["family"]
    if family == "duplicate_encoding":
        return f"{'abcd'[cue['known_index']]} = {cue['known_value']}"
    if family == "cross_series":
        return "\n".join(
            f"{'abcd'[i]} + {'abcd'[j]} = {total}" for i, j, total in sorted(cue["edges"])
        )
    if family == "trend":
        return "b - a = c - b = d - c"
    raise ValueError(f"unknown constraint family: {family}")


def build_prompt(scene, interface):
    if interface not in {"SYMBOLIC_FRESH", "IMAGE_CUE_FRESH"}:
        raise ValueError("only the two controlled fresh interfaces use this template")
    user = USER_TEMPLATE.format(
        observed_json=json.dumps(scene["observed_world"], separators=(",", ":")),
        cue_text=cue_text(scene["cue"]),
        operation_expression=OPERATIONS[scene["operation"]],
    )
    if interface == "IMAGE_CUE_FRESH":
        user = "The attached chart also shows the true record.\n" + user
    content = {"system": SYSTEM_PROMPT, "user": user}
    hash_inputs = {
        **content,
        "template_version": TEMPLATE_VERSION,
        "image_hash": scene.get("image_hash") if interface == "IMAGE_CUE_FRESH" else None,
    }
    prompt_hash = hashlib.sha256(json.dumps(hash_inputs, sort_keys=True).encode()).hexdigest()
    return {
        **content,
        "prompt_hash": prompt_hash,
        **({"image_path": scene["image_path"]} if interface == "IMAGE_CUE_FRESH" else {}),
    }
