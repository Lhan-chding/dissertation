"""Equal-task canonical O0 targets and strict training-semantic alias identities."""

import copy
import json
import random
from collections import defaultdict

from .config import digest
from .public_tasks import compile_case, public_verifier, validate_public


def canonical_json(world):
    if len(world) != 4 or any(type(v) is not int or not 0 <= v <= 99 for v in world):
        raise ValueError("Canonical target requires four legal integers")
    return json.dumps(list(world), separators=(",", ":"))


def build_training_view(tasks, targets, *, tokenizer=None, eos_id=None, source="self"):
    by_id = {t["task_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task identity")
    if set(targets) - set(by_id):
        raise ValueError("Targets contain unknown tasks")
    rows = []
    for tid in sorted(targets):
        task, item = by_id[tid], targets[tid]
        validate_public(task)
        if task["split"] not in ("T_train", "R_replay"):
            raise PermissionError("Only T/replay may become training examples")
        world = item["canonical_vector"]
        if not public_verifier(task, world):
            raise ValueError("Target failed public verification")
        if source == "self" and (
            item.get("label_source_provenance") != "self_public_verifier"
            or not item.get("discovery_request_ids")
        ):
            raise ValueError("SELF target requires accepted teacher provenance, never solver fill")
        if source not in ("self", "gold", "replay"):
            raise ValueError("Explicit registered label source required")
        target = canonical_json(world)
        case = compile_case(task, "O0")
        row = {
            "task_id": tid,
            "O0_prompt_id": case["prompt_identity"],
            "prompt": case["prompt"],
            "target": target,
            "canonical_target_string": target,
            "weight": 1,
            "split": task["split"],
            "family": task["family"],
            "label_source_provenance": copy.deepcopy(item.get("label_source_provenance", source)),
            "discovery_request_ids": sorted(item.get("discovery_request_ids", [])),
        }
        if tokenizer is not None:
            if type(eos_id) is not int or eos_id < 0:
                raise ValueError("One explicit EOS token ID required")
            tokens = list(tokenizer.encode(target, add_special_tokens=False))
            if not tokens or eos_id in tokens:
                raise ValueError("Canonical payload must not contain EOS")
            row.update(target_token_ids=tokens, EOS_id=eos_id)
        rows.append(row)
    return rows


def gold_targets(tasks, audit_rows):
    by_id = {t["task_id"]: t for t in tasks}
    audits = {a["task_id"]: a for a in audit_rows}
    if set(audits) != set(by_id) or len(audits) != len(audit_rows):
        raise ValueError("GOLD requires exactly matching T audit rows")
    targets = {}
    for tid, task in by_id.items():
        if task["split"] != "T_train" or audits[tid]["split"] != "T_train":
            raise PermissionError("GOLD can read only T labels")
        world = audits[tid]["true_world"]
        if not public_verifier(task, world):
            raise ValueError("GOLD audit/public contract disagreement")
        targets[tid] = {
            "canonical_vector": world,
            "label_source_provenance": "gold_solver_privileged_T",
        }
    return targets


def matched_gold_ids(tasks, audit_rows, mix_ids, seed):
    audits = {a["task_id"]: a for a in audit_rows}
    groups, quotas = defaultdict(list), defaultdict(int)
    task_ids = {t["task_id"] for t in tasks}
    if set(mix_ids) - task_ids:
        raise ValueError("MIX contains non-T tasks")
    for task in tasks:
        tid = task["task_id"]
        if task["split"] != "T_train" or audits[tid]["split"] != "T_train":
            raise PermissionError("Matching reads T audit only")
        cell = (task["family"], audits[tid]["corrupted_index"])
        groups[cell].append(tid)
        quotas[cell] += tid in mix_ids
    rng, selected, cells = random.Random(seed), [], {}
    for cell in sorted(groups):
        pool = sorted(groups[cell])
        chosen = rng.sample(pool, quotas[cell])
        selected.extend(chosen)
        cells[":".join(map(str, cell))] = {"N": len(pool), "quota": quotas[cell]}
    union = set(selected) | set(mix_ids)
    return sorted(selected), {
        "seed": seed,
        "cells": cells,
        "jaccard_with_mix": len(set(selected) & set(mix_ids)) / len(union) if union else 1.0,
    }


def view_semantics(rows):
    """Provenance never changes gradients, shuffle seeds, or alias identities."""
    keys = ("task_id", "O0_prompt_id", "prompt", "target", "weight", "target_token_ids", "EOS_id")
    result = []
    seen = set()
    for row in sorted(rows, key=lambda r: r["task_id"]):
        if row["task_id"] in seen:
            raise ValueError("Training view must contain one target per task")
        seen.add(row["task_id"])
        result.append({k: copy.deepcopy(row[k]) for k in keys if k in row})
    return result


def alias_key(parent, repeat, view, replay, settings):
    required = {
        "lora_initial_state",
        "tokenizer_identity",
        "chat_template_identity",
        "eos_id",
        "optimizer",
        "scheduler",
        "batch",
        "seed",
        "steps",
        "runtime_identity",
    }
    if set(settings) < required or not required.issubset(settings):
        raise ValueError(
            "Alias requires full model/token/optimizer/scheduler/batch/runtime settings"
        )
    if parent not in ("S96", "REP96") or repeat not in (0, 1):
        raise ValueError("Registered parent/repeat required")
    if settings["seed"] != (73101, 73102)[repeat]:
        raise ValueError("Wrong pipeline repeat seed")
    return digest(
        {
            "parent": parent,
            "repeat": repeat,
            "view": view_semantics(view),
            "replay": view_semantics(replay),
            "settings": settings,
        }
    )


def empty_parent_alias(parent, repeat):
    return {
        "status": "NO_VERIFIED_TARGETS_RETURN_PARENT",
        "parent_id": parent,
        "pipeline_repeat": repeat,
        "SFT_updates": 0,
        "E_draw_indices": list(range(8)),
        "G_draw_indices": list(range(4)),
        "no_oracle_fill": True,
    }
