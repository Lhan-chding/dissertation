"""Fixed R2 interventions and isolated, hash-bound thinking input preparation."""

from __future__ import annotations

import copy
import difflib
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

from .core import canonical_hash, file_hash, write_json
from .prompts import OPERATIONS, SYSTEM_PROMPT, USER_TEMPLATE, build_prompt, cue_text

CONDITIONS = (
    "SYM_ORIGINAL",
    "SYM_CLEAR",
    "SYM_NO_OPERATION",
    "IMAGE_CUE",
    "IMAGE_ONLY",
    "ORACLE_INDEX",
)
LONG_CONDITIONS = ("SYM_LONG", "SYM_THINKING")
FAMILIES = ("duplicate_encoding", "cross_series", "trend")
CHARTS = ("grouped_bar", "line")
TEMPLATE_VERSION = "R2-fixed-inputs-v1-2026-09-10"
SELECTION_VERSION = "R2-calibration-hash-v1"
SAMPLE_SEED_ROOT = 20260909
PINNED_CHAT_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
THINK_OPEN_ID = 248068
THINK_CLOSE_ID = 248069
CLEAR_REPLACEMENT = (
    "Exactly one value in this observed record is wrong; the other three values are correct. "
    "Replace exactly one value, preserving those other three values."
)
IMAGE_ONLY_SYSTEM = SYSTEM_PROMPT.replace("repairing", "reading")
IMAGE_ONLY_USER = (
    "The attached chart shows the true record.\n"
    + USER_TEMPLATE.split("The observed record", 1)[0]
    + "The downstream calculation is: {operation_expression}.\n"
    + "Read the correct record directly from the chart. "
    "Return [a,b,c,d], not the downstream answer."
)


def _cells(scenes):
    cells = defaultdict(list)
    seen = set()
    allowed = set(itertools.product(FAMILIES, CHARTS, OPERATIONS))
    for scene in scenes:
        if scene.get("split") != "calibration":
            raise ValueError("R2 panel requires calibration scenes only")
        sid = scene.get("base_scene_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("R2 scene requires a nonempty base_scene_id")
        if sid in seen:
            raise ValueError("duplicate R2 base_scene_id")
        seen.add(sid)
        cell = (scene.get("constraint_family"), scene.get("chart_type"), scene.get("operation"))
        if cell not in allowed:
            raise ValueError("unknown R2 family/chart/operation cell")
        cells[cell].append(scene)
    return cells


def select_panel(scenes):
    """Select four independent scenes per fixed cell, never consulting model outcomes."""
    cells = _cells(scenes)
    selected = []
    for cell in sorted(itertools.product(FAMILIES, CHARTS, OPERATIONS)):
        ranked = sorted(
            cells[cell], key=lambda s: canonical_hash([SELECTION_VERSION, s["base_scene_id"]])
        )
        if len(ranked) < 4:
            raise ValueError(f"Insufficient calibration scenes for R2 cell {cell}")
        selected.extend(ranked[:4])
    return copy.deepcopy(selected)


def _validate_panel(panel):
    cells = _cells(panel)
    if len(cells) != 18 or any(len(group) != 4 for group in cells.values()):
        raise ValueError("R2 fixed panel must contain exactly four scenes in all 18 cells")
    return cells


def select_long_panel(panel):
    cells = _validate_panel(panel)
    return copy.deepcopy(
        [
            min(group, key=lambda s: canonical_hash(["R2-long-hash-v1", s["base_scene_id"]]))
            for cell, group in sorted(cells.items())
            if cell[0] in {"cross_series", "trend"}
        ]
    )


def variant_prompt(scene, condition):
    if condition not in (*CONDITIONS, *LONG_CONDITIONS):
        raise ValueError(f"unknown R2 condition: {condition}")
    if condition in {"SYM_ORIGINAL", *LONG_CONDITIONS}:
        return build_prompt(scene, "SYMBOLIC_FRESH")
    if condition == "IMAGE_CUE":
        return build_prompt(scene, "IMAGE_CUE_FRESH")
    base = build_prompt(scene, "SYMBOLIC_FRESH")
    user, system = base["user"], base["system"]
    if condition == "SYM_CLEAR":
        clear_cue = (
            "b - a = c - b\nc - b = d - c"
            if scene["cue"]["family"] == "trend"
            else cue_text(scene["cue"])
        )
        user = USER_TEMPLATE.format(
            observed_json=json.dumps(scene["observed_world"], separators=(",", ":")),
            cue_text=clear_cue,
            operation_expression=OPERATIONS[scene["operation"]],
        ).replace("Exactly one value in this observed record is wrong.", CLEAR_REPLACEMENT)
    elif condition == "SYM_NO_OPERATION":
        line = f"The downstream calculation is: {OPERATIONS[scene['operation']]}.\n"
        if user.count(line) != 1:
            raise ValueError("original operation template drift")
        user = user.replace(line, "", 1)
    elif condition == "IMAGE_ONLY":
        system = IMAGE_ONLY_SYSTEM
        user = IMAGE_ONLY_USER.format(operation_expression=OPERATIONS[scene["operation"]])
    elif condition == "ORACLE_INDEX":
        index = scene["changed_index"]
        if type(index) is not int or not 0 <= index < 4:
            raise ValueError("ORACLE_INDEX requires an integer changed_index in 0..3")
        user += (
            f"\nThe wrong value is coordinate {'abcd'[index]} "
            f"(index {index} in zero-based [a,b,c,d]); this hint supplies no replacement value."
        )
    content = {"system": system, "user": user}
    return {
        **content,
        "prompt_hash": canonical_hash(
            {
                **content,
                "condition": condition,
                "template_version": TEMPLATE_VERSION,
                "image_hash": scene.get("image_hash") if condition == "IMAGE_ONLY" else None,
            }
        ),
        **({"image_path": scene["image_path"]} if condition == "IMAGE_ONLY" else {}),
    }


def build_requests(panel):
    _validate_panel(panel)
    long_ids = {s["base_scene_id"] for s in select_long_panel(panel)}
    requests = []
    for scene in sorted(panel, key=lambda s: s["base_scene_id"]):
        conditions = (*CONDITIONS, *(LONG_CONDITIONS if scene["base_scene_id"] in long_ids else ()))
        for condition in conditions:
            length = 256 if condition == "SYM_LONG" else 1024 if condition == "SYM_THINKING" else 64
            prompt = variant_prompt(scene, condition)
            identity = {
                "base_scene_id": scene["base_scene_id"],
                "condition": condition,
                "max_new_tokens": length,
                "enable_thinking": condition == "SYM_THINKING",
                "prompt_hash": prompt["prompt_hash"],
            }
            prompt_id = canonical_hash({"phase": "R2", **identity})
            for mode, count in (
                ("sample", 2 if condition in LONG_CONDITIONS else 4),
                ("greedy", 1),
            ):
                for index in range(count):
                    sample_identity = {
                        "phase": "R2",
                        "prompt_id": prompt_id,
                        "decode_mode": mode,
                        "rollout_index": index,
                        "sample_seed_root": SAMPLE_SEED_ROOT,
                    }
                    key = canonical_hash(sample_identity)
                    requests.append(
                        {
                            **identity,
                            "prompt_id": prompt_id,
                            "decode_mode": mode,
                            "rollout_index": index,
                            "sample_key": key,
                            "sample_seed": int(key[:8], 16) % (2**31),
                        }
                    )
    return requests


def write_panel_artifacts(panel, out):
    """Write literal prompt evidence only; the caller owns execution status and gates."""
    requests = build_requests(panel)
    by_scene = {s["base_scene_id"]: s for s in panel}
    prompt_rows, diffs = [], ["# R2 fixed prompt interventions\n"]
    seen = set()
    for request in requests:
        if request["prompt_id"] in seen:
            continue
        seen.add(request["prompt_id"])
        scene = by_scene[request["base_scene_id"]]
        prompt = variant_prompt(scene, request["condition"])
        original = build_prompt(scene, "SYMBOLIC_FRESH")
        prompt_rows.append(
            {
                **{
                    k: v
                    for k, v in request.items()
                    if k not in {"sample_key", "sample_seed", "decode_mode", "rollout_index"}
                },
                "family": scene["constraint_family"],
                "chart_type": scene["chart_type"],
                "operation": scene["operation"],
                "prompt": prompt,
                "image_hash": scene.get("image_hash") if prompt.get("image_path") else None,
                "changed_index_in_prompt": request["condition"] == "ORACLE_INDEX",
            }
        )
        old = [
            f"SYSTEM: {original['system']}\n",
            *(line + "\n" for line in original["user"].splitlines()),
        ]
        new = [
            f"SYSTEM: {prompt['system']}\n",
            *(line + "\n" for line in prompt["user"].splitlines()),
        ]
        diff = "".join(
            difflib.unified_diff(old, new, fromfile="SYM_ORIGINAL", tofile=request["condition"])
        )
        diffs.append(
            f"\n## {scene['base_scene_id']} / {request['condition']}\n\n"
            f"Original prompt hash: `{original['prompt_hash']}`; "
            f"condition prompt hash: `{prompt['prompt_hash']}`. "
            f"max_new_tokens={request['max_new_tokens']}; "
            f"enable_thinking={request['enable_thinking']}.\n\n"
            + (
                f"```diff\n{diff}\n```\n"
                if diff
                else "Text unchanged; execution condition recorded above.\n"
            )
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    diff_text = "".join(diffs)
    diff_path = out / "prompt_diffs.md"
    if diff_path.exists() and diff_path.read_text() != diff_text:
        raise ValueError("existing R2 prompt diffs changed")
    diff_path.write_text(diff_text, encoding="utf-8")
    manifest = {
        "status": "READY_FOR_SERVER_GENERATION",
        "selection_version": SELECTION_VERSION,
        "template_version": TEMPLATE_VERSION,
        "base_scenes": len(panel),
        "base_scene_ids": sorted(by_scene),
        "long_base_scene_ids": sorted(s["base_scene_id"] for s in select_long_panel(panel)),
        "panel_hash": canonical_hash(sorted(panel, key=lambda s: s["base_scene_id"])),
        "conditions": list(CONDITIONS),
        "long_conditions": list(LONG_CONDITIONS),
        "request_count": len(requests),
        "requests_hash": canonical_hash(requests),
        "prompt_diffs_sha256": file_hash(diff_path),
        "rows": prompt_rows,
    }
    path = out / "panel_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("existing R2 panel manifest changed")
    write_json(path, manifest)
    return manifest


def _thinking_ids(tokenizer):
    for text, expected in (("<think>", THINK_OPEN_ID), ("</think>", THINK_CLOSE_ID)):
        if tokenizer.convert_tokens_to_ids(text) != expected or tokenizer.encode(
            text, add_special_tokens=False
        ) != [expected]:
            raise ValueError("pinned Qwen3.5 thinking token identity drift")
    return THINK_OPEN_ID, THINK_CLOSE_ID


def prepare_r2(adapter, prompt, data_root, enable_thinking=False):
    """Keep all non-thinking inputs on the audited adapter path without mutation."""
    if type(enable_thinking) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    if not enable_thinking:
        return adapter.prepare(prompt, data_root)
    if (
        prompt.get("image_path")
        or "messages" in prompt
        or not isinstance(prompt.get("system"), str)
    ):
        raise ValueError("R2 thinking is restricted to the original symbolic interface")
    from .model_adapters.base import _hash_json
    from .optimizer_fork import state_hash

    processor = adapter.processor
    template = processor.chat_template
    if (
        not isinstance(template, str)
        or hashlib.sha256(template.encode()).hexdigest() != PINNED_CHAT_TEMPLATE_SHA256
    ):
        raise ValueError("pinned Qwen3.5 chat template drift")
    open_id, close_id = _thinking_ids(processor.tokenizer)
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": [{"type": "text", "text": prompt["user"]}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if not text.endswith("<think>\n"):
        raise ValueError("thinking prompt lacks the pinned unclosed opener")
    inputs = dict(processor(text=[text], return_tensors="pt", padding=False))
    ids = inputs["input_ids"][0].tolist()
    last_open = max((i for i, token in enumerate(ids) if token == open_id), default=-1)
    if (
        last_open < 0
        or close_id in ids[last_open + 1 :]
        or processor.tokenizer.decode(
            ids[last_open:], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        != "<think>\n"
    ):
        raise ValueError("tokenized thinking prompt lacks the pinned unclosed opener")
    return {
        "inputs": inputs,
        "audit": {
            "final_prompt": text,
            "final_prompt_token_ids": ids,
            "final_prompt_hash": _hash_json(text),
            "tokenized_prompt_hash": state_hash(inputs["input_ids"]),
            "prompt_token_count": len(ids),
            "image_token_count": 0,
            "enable_thinking": True,
            "thinking_template_changed": True,
            "input_tensor_hash": state_hash(inputs),
            "pixel_values_hash": None,
            "original_image_size": None,
            "image_grid_thw": None,
            "processor_height": None,
            "processor_width": None,
            "token_count_status": "R2_PROCESSOR_MEASURED",
            "thinking_open_token_id": open_id,
            "thinking_close_token_id": close_id,
            "thinking_start_in_prompt": True,
            "chat_template_file_sha256": PINNED_CHAT_TEMPLATE_SHA256,
            "numeric_token_count": sum(
                len(processor.tokenizer.encode(v, add_special_tokens=False))
                for v in re.findall(r"\d+", prompt["user"])
            ),
            "numeric_token_count_definition": (
                "sum of standalone tokenizer lengths of every decimal numeral in user text; "
                "excludes system and template"
            ),
        },
    }


def split_thinking_completion(token_ids, tokenizer, eos_ids):
    """Split only at pinned token delimiters; generated text is never searched for answers."""
    open_id, close_id = _thinking_ids(tokenizer)
    ids = list(token_ids)
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("completion token IDs must be nonnegative integers")
    terminal = bool(ids and ids[-1] in eos_ids)
    body = ids[:-1] if terminal else ids[:]
    leading_open = bool(body and body[0] == open_id)
    work = body[1:] if leading_open else body[:]
    valid = (
        work.count(close_id) == 1 and open_id not in work and not any(t in eos_ids for t in work)
    )
    if valid:
        boundary = work.index(close_id)
        reasoning, final = work[:boundary], work[boundary + 1 :]
        reason = "unique_pinned_closing_delimiter"
        delimiters = int(leading_open) + 1
    else:
        reasoning, final = work, []
        reason = (
            "missing_closing_delimiter"
            if close_id not in work
            else "ambiguous_or_invalid_delimiter_sequence"
        )
        delimiters = int(leading_open)

    def decode(values):
        return tokenizer.decode(
            values, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    return {
        "reasoning_text": decode(reasoning),
        "final_text": decode(final) if valid else None,
        "reasoning_token_ids": reasoning,
        "final_token_ids": final,
        "reasoning_token_count": len(reasoning),
        "final_token_count": len(final),
        "delimiter_token_count": delimiters,
        "eos_token_count": int(terminal),
        "completion_token_count": len(ids),
        "final_status": "resolved_final" if valid else "unresolved_final",
        "unresolved_final": not valid,
        "segmentation_reason": reason,
    }
