"""Immutable follow-up bank assembly and seed-explicit R4 prompt schedules.

No selection function accepts control outcomes. Model input tensors are only
hashed when the caller explicitly supplies already prepared inputs.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict

from .core import canonical_hash
from .followup_parent import CATEGORIES, SELECTED_BANKS, _sha, validate_parent_metadata

COMPOSITE_ID = "composite_no_x_plus_three_level"
COMPOSITE_SOURCES = ((3, 2), (11, 2), (0, 0), (0, 1))


def build_training_schedule(train_scenes, *, seed, preserve_legacy17=True, data_root=None):
    """Reuse exact legacy seed17; parameterize only the other seeds' hash order.

    The return shape matches R4. Generation must still occur on each arm's own
    current policy; this shared prompt schedule does not share training outputs.
    """
    from .r4_inputs import STRATA, TRAIN_NAMESPACE, _n_record, _validate_n, build_train_schedule

    if type(seed) is not int or seed not in (17, 29, 41):
        raise ValueError("Follow-up sampler seed must be one of 17, 29, 41")
    if preserve_legacy17 is not True:
        raise ValueError("Follow-up must preserve the legacy17 schedule")
    if seed == 17:
        return build_train_schedule(train_scenes, data_root=data_root)
    scenes = _validate_n(train_scenes, "train", 576)
    groups = defaultdict(list)
    for scene in scenes:
        groups[(scene["constraint_family"], scene["interface"])].append(scene)
    if any(len(groups[stratum]) != 96 for stratum in STRATA):
        raise ValueError("Follow-up train requires 96 fixed prompts per stratum")
    for stratum in STRATA:
        groups[stratum].sort(
            key=lambda s: canonical_hash(
                [TRAIN_NAMESPACE, seed, s["base_scene_id"], s["interface"]]
            )
        )
    prompts = [
        _n_record(groups[stratum][index], stratum[1], data_root)
        for index in range(96)
        for stratum in STRATA
    ]
    ids = [p["prompt_id"] for p in prompts]
    return {
        "sampler_seed": seed,
        "train_prompts": prompts,
        "train_steps": [ids[i : i + 4] for i in range(0, 256, 4)],
    }


def _source_groups(bank_spec):
    if not isinstance(bank_spec, dict):
        raise ValueError("A frozen bank specification is required")
    if bank_spec.get("kind") == "parent_bank":
        index = bank_spec.get("parent_bank_index")
        if type(index) is not int or index not in SELECTED_BANKS:
            raise ValueError("Bank must be one of the five predeclared parent banks")
        if bank_spec.get("id") != f"bank{index:02d}":
            raise ValueError("Parent bank ID does not match the fixed source")
        return [(index, group) for group in range(4)]
    if bank_spec.get("kind") != "composite_diagnostic_bank" or bank_spec.get("id") != COMPOSITE_ID:
        raise ValueError("Unknown follow-up bank specification")
    source = bank_spec.get("source_groups_zero_based", [])
    if len(source) != 4 or any(not isinstance(g, dict) for g in source):
        raise ValueError("Composite requires four fixed source groups")
    pairs = [(g.get("bank"), g.get("group")) for g in source]
    if tuple(pairs) != COMPOSITE_SOURCES or any(type(n) is not int for pair in pairs for n in pair):
        raise ValueError("Composite source bank/group selection cannot change")
    return pairs


def _request_index(manifest):
    """Verify legacy sample keys, including source/model/data/parser lock identity."""
    identity, plan = manifest["request_identity"], manifest["plan"]
    prompts = {p["prompt_id"]: p for p in [*plan["train_prompts"], *plan["control_prompts"]]}
    result, locations = {}, set()
    requests = manifest["requests"]
    if not isinstance(requests, list) or len(requests) != 1152:
        raise ValueError("Parent requires complete 384 train / 768 control request metadata")
    for item in requests:
        if not isinstance(item, dict):
            raise ValueError("Parent request must be an object")
        prompt = prompts.get(item.get("prompt_id"))
        if prompt is None:
            raise ValueError("Request prompt absent from parent plan")
        index = item.get("sample_index")
        role = "train" if prompt["split"] == "train" else "control_proposal"
        if type(index) is not int or not 0 <= index < (8 if role == "train" else 16):
            raise ValueError("Parent sample index is invalid")
        bank_index = next(
            (i for i, bank in enumerate(plan["banks"]) if prompt["prompt_id"] in bank), None
        )
        expected = {
            "phase": identity["phase"],
            "bank_role": role,
            "bank_index": bank_index,
            "base_scene_id": prompt["base_scene_id"],
            "prompt_id": prompt["prompt_id"],
            "family": prompt["family"],
            "interface": prompt["interface"],
            "split": prompt["split"],
            "prompt_hash": prompt["prompt_hash"],
            "scene_hash": prompt["scene_hash"],
            "group_id": prompt["prompt_id"],
            "sample_index": index,
            "rollout_index": index,
            "decode_mode": "sample",
            "max_new_tokens": 64,
            "enable_thinking": False,
            "origin_state_hash": identity["origin_hash"],
            "checkpoint_step": 64,
        }
        rng = {
            "seed_root": 20260909,
            "phase": identity["phase"],
            "checkpoint_step": 64,
            "model_hash": identity["model_hash"],
            "adapter_hash": identity["initial_adapter_hash"],
            "origin_hash": identity["origin_hash"],
            "plan_hash": plan["plan_hash"],
            "prompt_id": prompt["prompt_id"],
            "role": role,
            "sample_index": index,
        }
        rng_hash = canonical_hash(rng)
        expected.update(
            {
                "sample_rng_key": rng_hash,
                "sample_seed": int(rng_hash[:8], 16) % 2**31,
                "sample_key": canonical_hash(
                    {"identity": identity, "request": expected.copy(), "rng": rng}
                ),
            }
        )
        if item != expected:
            raise ValueError("Parent request origin/model/parser/data identity or hash mismatch")
        key, location = item["sample_key"], (item["prompt_id"], index)
        if key in result or location in locations:
            raise ValueError("Duplicate parent request sample identity")
        result[key] = item
        locations.add(location)
    return result


def _validate_raw(row, request, prompt, runtime):
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in request.items()):
        raise ValueError("Raw sample request identity mismatch")
    if type(row["sample_index"]) is not int or type(row["rollout_index"]) is not int:
        raise ValueError("Raw sample index must be an integer")
    try:
        digest = canonical_hash({key: value for key, value in row.items() if key != "record_hash"})
    except (TypeError, ValueError) as error:
        raise ValueError("Raw sample is not finite JSON") from error
    if row.get("record_hash") != digest:
        raise ValueError("Raw sample content hash mismatch")
    identity = runtime["identity"]
    expected = {
        "model_id": runtime["config"]["model"]["id"],
        "model_revision": runtime["config"]["model"]["revision"],
        "adapter_hash": runtime["initial_adapter_hash"],
        "protocol_version": identity["protocol_version"],
        "train_seed": 17,
        "optimizer_state_hash": runtime["optimizer_initial_hash"],
        "execution_kind": identity["execution_kind"],
        "generation_config_hash": canonical_hash(runtime["config"]["generation_proposed_N"]),
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Raw model/parser/origin policy binding differs from parent")
    # Historic rows bind parser via protocol + source in their sample request key.
    # New explicit aliases, if supplied, must not contradict the historical lock.
    for key, value in (
        ("data_hash", identity["data_hash"]),
        ("model_hash", identity["model_hash"]),
        ("source_hash", identity["source_hash"]),
        ("parser_source_hash", runtime["source"]["source_files"]["src/r2_runtime.py"]),
    ):
        if key in row and row[key] != value:
            raise ValueError("Raw source/parser/data hash alias differs from parent")
    tokens, scores = row.get("token_ids"), row.get("old_logprobs")
    if (
        not isinstance(tokens, list)
        or not 1 <= len(tokens) <= 64
        or any(type(t) is not int or t < 0 for t in tokens)
        or not isinstance(scores, list)
        or len(scores) != len(tokens)
        or any(type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in scores)
    ):
        raise ValueError("Raw token IDs and finite old behavior logprobs must align")
    if (
        row.get("raw_token_ids") != tokens
        or row.get("n_generated_tokens") != len(tokens)
        or row.get("behavior_token_logprobs") != scores
        or row.get("per_token_logprob_behavior") != scores
    ):
        raise ValueError("Raw token / old probability aliases changed")
    if (
        row.get("category") not in CATEGORIES
        or row.get("execution_checks", {}).get("passed") is not True
    ):
        raise ValueError("Raw category or execution validity missing")
    if not isinstance(row.get("raw_text"), str) or row["raw_text"] != row.get("raw_completion"):
        raise ValueError("Raw text aliases changed")
    for key in (
        "final_prompt_hash",
        "tokenized_prompt_hash",
        "input_ids_hash",
        "input_tensor_hash",
        "prepared_hash",
    ):
        _sha(row.get(key))
    if row["input_ids_hash"] != row["tokenized_prompt_hash"]:
        raise ValueError("Raw tokenized input hash aliases changed")
    image = prompt["scene"]["image_hash"] if prompt["interface"] == "IMAGE_CUE_FRESH" else None
    if row.get("image_hash") != image:
        raise ValueError("Raw image binding differs from original scene")


def build_followup_bank(
    parent_manifest, raw_rows, bank_spec, *, candidate_manifest, runtime_lock, prepared_inputs=None
):
    """Reassemble selected original groups, preserving every original sample byte.

    ``raw_rows`` may contain the full train/control ledger. All supplied records
    are checked, while complete source banks are required for the original
    record-hash commitments. ``prepared_inputs`` is a prompt-to-prepared mapping;
    attaching it requires exact original prepared state and audit hashes.
    """
    validate_parent_metadata(runtime_lock, parent_manifest, candidate_manifest)
    sources = _source_groups(bank_spec)
    requests = _request_index(parent_manifest)
    plan = parent_manifest["plan"]
    prompt_map = {p["prompt_id"]: p for p in [*plan["train_prompts"], *plan["control_prompts"]]}
    if isinstance(raw_rows, dict):
        if any(key != value.get("sample_key") for key, value in raw_rows.items()):
            raise ValueError("Raw ledger key mismatch")
        rows = list(raw_rows.values())
    else:
        rows = list(raw_rows)
    if not rows:
        raise ValueError("Missing parent raw samples; compact metadata is insufficient")
    keys, grouped = set(), defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Raw sample must be a JSON object")
        key = row.get("sample_key")
        if not isinstance(key, str) or key in keys:
            raise ValueError("Duplicate or missing raw sample key")
        if key not in requests:
            raise ValueError("Raw sample key is not part of the original parent manifest")
        _validate_raw(row, requests[key], prompt_map[row["prompt_id"]], runtime_lock)
        keys.add(key)
        grouped[row["prompt_id"]].append(row)
    source_banks = sorted({bank for bank, _ in sources})
    for index in source_banks:
        ordered = []
        for pid in plan["banks"][index]:
            group = sorted(grouped[pid], key=lambda r: r["sample_index"])
            if len(group) != 8 or [r["sample_index"] for r in group] != list(range(8)):
                raise ValueError("Incomplete parent source bank: every group requires exactly K8")
            if any(
                r["split"] != "train" or r["bank_role"] != "train" or r["bank_index"] != index
                for r in group
            ):
                raise ValueError("Parent training/control bank separation failure")
            ordered.extend(group)
            grouped[pid] = group
        summary = candidate_manifest["banks"][str(index)]
        if (
            canonical_hash([r["record_hash"] for r in ordered])
            != summary["identity"]["sample_hash"]
        ):
            raise ValueError("Original source bank sample hash commitment mismatch")
        for candidate in summary["candidates"]:
            if (
                candidate["checkpoint_identity"]["sample_hash"]
                != summary["identity"]["sample_hash"]
            ):
                raise ValueError("Candidate source sample hash commitment mismatch")
    result_groups, descriptors, prompt_ids = [], [], []
    for position, (index, group_index) in enumerate(sources):
        pid = plan["banks"][index][group_index]
        group = grouped[pid]
        expected_counts = candidate_manifest["banks"][str(index)]["group_composition"]["prompts"][
            group_index
        ]["counts"]
        counts = {c: sum(row["category"] == c for row in group) for c in CATEGORIES}
        if counts != expected_counts:
            raise ValueError("Raw sample categories differ from committed group composition")
        if (
            bank_spec["id"] == COMPOSITE_ID
            and bank_spec["source_groups_zero_based"][position].get("expected_counts") != counts
        ):
            raise ValueError("Composite fixed group count mismatch")
        fields = (
            "final_prompt_hash",
            "tokenized_prompt_hash",
            "input_tensor_hash",
            "prepared_hash",
            "image_hash",
        )
        if any(any(row[field] != group[0][field] for field in fields) for row in group):
            raise ValueError("Same prompt has inconsistent prepared/input/image bindings")
        output = copy.deepcopy(group)
        if prepared_inputs is not None:
            from .optimizer_fork import state_hash

            if pid not in prepared_inputs:
                raise ValueError("Original prepared model input missing")
            prepared = prepared_inputs[pid]
            if state_hash(prepared) != group[0]["prepared_hash"]:
                raise ValueError("Original prepared input tensor hash changed")
            audit = prepared.get("audit", {})
            if any(audit.get(field) != group[0][field] for field in fields[:3]):
                raise ValueError("Original prepared input audit binding changed")
            copied = copy.deepcopy(prepared)
            for row in output:
                row["prepared"] = copied
        result_groups.append(output)
        prompt_ids.append(pid)
        descriptors.append(
            {"bank": index, "group": group_index, "prompt_id": pid, "counts": counts}
        )
    if len(set(prompt_ids)) != 4:
        raise ValueError("Follow-up bank contains duplicate prompts")
    binding = {
        "bank_id": bank_spec["id"],
        "source_groups": descriptors,
        "origin_hash": runtime_lock["origin_hash"],
        "sample_record_hashes": [[r["record_hash"] for r in group] for group in result_groups],
    }
    return {
        "bank_id": bank_spec["id"],
        "groups": result_groups,
        "source_groups": descriptors,
        "prompt_ids": prompt_ids,
        "bank_hash": canonical_hash(binding),
        "execution_kind": runtime_lock["identity"]["execution_kind"],
        "parent_origin_hash": runtime_lock["origin_hash"],
        "selection_scope": "PREDECLARED_STRUCTURAL_DIAGNOSTIC",
        "prepared_inputs_verified": prepared_inputs is not None,
    }
