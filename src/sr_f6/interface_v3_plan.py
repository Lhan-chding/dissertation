"""Prospective two-stage interface-v3 engineering screen, without execution.

Both stages reuse predetermined original DEV_CAL roots. They are development
inputs, not independent population samples. Stage 2 is a frozen conditional
budget, never replacement sampling or permission to reopen the v2 failure.
The strict scorer, complete confirmation gates, and training contract stay put.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import PurePosixPath

from srf6_ref.core import FAMILIES, seed_for

from .common import digest
from .interface_repair import FACTORS, PARENTS
from .interface_repair import screen_inputs as validate_v2_inputs
from .interface_v3_protocol import VERSION, WIRE_PROTOCOL, public_prompt

PHASE = "INTERFACE_SCREEN_V3"
MASTER_SEED = 20261028
METRICS = ("V", "S_known", "B_known")


def _stage(stage):
    if type(stage) is not int or stage not in (1, 2):
        raise ValueError("Only the two preregistered integer stages exist")
    return stage


def _indices(stage):
    return (0, 1) if _stage(stage) == 1 else (2, 3)


def _questions(stage):
    return {
        f"DEV_CAL_{family}_{index:04d}_{factor}_v{variant}": {
            "split": "DEV_CAL",
            "root_id": f"DEV_CAL_{family}_{index:04d}",
            "family": family,
            "factor": factor,
            "variant": variant,
        }
        for family in FAMILIES
        for index in _indices(stage)
        for factor in FACTORS
        for variant in range(4)
    }


def screen_inputs(public, private, stage=1):
    """Check all 384 original rows before deriving this stage's fixed 96."""
    stage = _stage(stage)
    # This validates all original public/private lineage, prompt, query, and
    # bound image references, including rows outside the selected stage.
    validate_v2_inputs(public, private)
    tasks = {task["qid"]: task for task in private}
    expected = _questions(stage)
    selected = []
    for row in sorted(public, key=lambda item: item["qid"]):
        if row["qid"] not in expected:
            continue
        task = tasks[row["qid"]]
        prompt = public_prompt(task["world"]["categories"], task["query"])
        new = copy.deepcopy(row)
        new.update(
            screen_stage=stage,
            prompt_version=VERSION,
            source_prompt_sha256=row["prompt_sha256"],
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        )
        new["model_input"]["prompt"] = prompt
        selected.append(new)
    if len(selected) != 96:
        raise ValueError("Exactly 96 fixed inputs required per potential stage")
    return selected


def _validate_inputs(inputs, stage):
    expected = _questions(stage)
    if (
        not isinstance(inputs, list)
        or len(inputs) != 96
        or any(not isinstance(row, dict) for row in inputs)
        or any(not isinstance(row.get("qid"), str) for row in inputs)
        or {row.get("qid") for row in inputs} != set(expected)
    ):
        raise ValueError("Exactly the 96 distinct registered stage inputs required")
    for row in inputs:
        if (
            any(row.get(key) != value for key, value in expected[row["qid"]].items())
            or type(row.get("variant")) is not int
            or type(row.get("screen_stage")) is not int
            or row["screen_stage"] != stage
            or row.get("prompt_version") != VERSION
            or row.get("protocol") != WIRE_PROTOCOL
        ):
            raise ValueError("Stage input lineage or prompt protocol differs")
        model_input = row.get("model_input")
        if not isinstance(model_input, dict) or set(model_input) != {
            "prompt",
            "image_file",
            "image_sha256",
        }:
            raise ValueError("Bound structured model input required")
        prompt, image = model_input["prompt"], model_input["image_file"]
        if (
            not isinstance(prompt, str)
            or not isinstance(image, str)
            or PurePosixPath(image).is_absolute()
            or ".." in PurePosixPath(image).parts
            or not image.startswith("DATA/images/")
            or row.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
        ):
            raise ValueError("Bound prompt or image reference differs")
        for pin in (row.get("source_prompt_sha256"), model_input["image_sha256"]):
            if not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
                raise ValueError("Historical prompt and image content pins required")


def screen_plan(inputs, stage=1):
    """Freeze all 576 potential rows, including conditionally unused stage 2."""
    stage = _stage(stage)
    _validate_inputs(inputs, stage)
    attempts = [
        {
            "stage": stage,
            "parent": parent,
            "qid": row["qid"],
            "root_id": row["root_id"],
            "factor": row["factor"],
            "variant": row["variant"],
            "draw": draw,
            "seed": seed_for(
                MASTER_SEED,
                VERSION,
                "screen",
                stage,
                parent,
                row["root_id"],
                row["variant"],
                draw,
            ),
            "model_input_sha256": digest(row["model_input"]),
        }
        for parent in PARENTS
        for row in sorted(inputs, key=lambda item: item["qid"])
        for draw in (0, 1)
    ]
    return {
        "schema": "srf6-interface-v3-input-plan-v1",
        "status": "INPUT_PLAN_ONLY_NOT_DEPLOYED",
        "phase": PHASE,
        "stage": stage,
        "prompt_version": VERSION,
        "wire_protocol": WIRE_PROTOCOL,
        "old_run_status": "FAIL_UNCHANGED",
        "source_split": "DEV_CAL",
        "root_indices": list(_indices(stage)),
        "source_scope": "fixed original development roots, not independent confirmation",
        "parents": list(PARENTS),
        "factors": list(FACTORS),
        "master_seed": MASTER_SEED,
        "seed_excludes_factor": True,
        "seed_semantics": [
            "MASTER_SEED",
            "VERSION",
            "screen",
            "stage",
            "parent",
            "root_id",
            "variant",
            "draw",
        ],
        "draws_per_question": 2,
        "expected_new_answers": 576,
        "per_parent_n": 192,
        "per_parent_factor_n": 96,
        "max_global_answers": 1152,
        "both_stage_plans_required_before_first_output": True,
        "stage2_parent_selection": "only frozen stage1 GRAY parents, with no parent FAIL",
        "stage1_gate": {"pass_each_at_least": 173, "fail_any_at_most": 163},
        "stage2_gate": {"combined_pass_each_at_least": 346, "combined_parent_n": 384},
        "gate_metrics": list(METRICS),
        "threshold_interpretation": "engineering counts, not an IID population confidence bound",
        "complete_G8_credit": "NOT_APPLICABLE",
        "max_prompt_candidates": 1,
        "on_failure": "STOP_PRESERVE_ALL_ROWS_NO_REPLACEMENT",
        "finish_policy": "balanced_only_unchanged",
        "scorer_changed": False,
        "parser_changed": False,
        "constrained_decoding": False,
        "generation_allowed": False,
        "training_allowed": False,
        "gate_status": "NOT_SIGNED",
        "model_calls": 0,
        "optimizer_updates": 0,
        "measurement_valid": "NOT_RUN",
        "comparison_supported": "NOT_RUN",
        "effect_estimated": "NOT_RUN",
        "confirmation": {
            "status": "NOT_PREPARED_NOT_AUTHORIZED",
            "source_split": "DEV_SCHEMA_CONFIRM",
            "master_seed": MASTER_SEED,
            "root_indices": list(range(8)),
            "requires": "independent roots and source/data/processor registration before outputs",
            "expected_new_answers": 9216,
            "per_parent_factor_n": 1536,
            "required_counts": {key: 1383 for key in METRICS},
            "registered_observability_checks": 18,
            "other_support_conditions": "all original F6 conditions unchanged",
            "new_independent_response_review": 360,
        },
        "attempts": attempts,
    }


def _counts(counts):
    if not isinstance(counts, dict) or set(counts) != set(METRICS):
        raise ValueError("Exactly V, S_known, and B_known counts required")
    if any(type(value) is not int or not 0 <= value <= 192 for value in counts.values()):
        raise ValueError("Each count must be an integer from 0 through 192")
    return {key: counts[key] for key in METRICS}


def stage1_gate(counts):
    counts = _counts(counts)
    if all(value >= 173 for value in counts.values()):
        return "PASS"
    if any(value <= 163 for value in counts.values()):
        return "FAIL"
    return "GRAY"


def stage2_gate(first_counts, second_counts):
    first, second = _counts(first_counts), _counts(second_counts)
    if stage1_gate(first) != "GRAY":
        raise ValueError("Stage 2 is only registered for a GRAY first stage")
    return "PASS" if all(first[key] + second[key] >= 346 for key in METRICS) else "FAIL"


def _decision(stage, parents, completed=()):
    statuses = {parent: row["status"] for parent, row in parents.items()}
    if "FAIL" in statuses.values():
        status = "SCREEN_FAIL_STOP"
    elif all(value == "PASS" for value in statuses.values()):
        status = "SCREEN_PASS"
    else:
        status = "STAGE2_REQUIRED"
    pending = sorted(parent for parent, state in statuses.items() if state == "GRAY")
    if status != "STAGE2_REQUIRED":
        pending = []
    return {
        "schema": "srf6-interface-v3-stage-decision-v1",
        "stage": stage,
        "status": status,
        "parents": copy.deepcopy(parents),
        "stage2_parents": pending,
        "completed_stage2_parents": sorted(completed),
        "budget": {
            "stage1_answers": 576,
            "stage2_answers_per_parent": 192,
            "max_stage2_answers": 576,
            "max_total_answers": 1152,
            "required_stage2_answers": 192 * len(pending),
        },
        "generation_allowed": False,
        "training_allowed": False,
        "gate_status": "NOT_SIGNED",
        "complete_G8_credit": "NOT_APPLICABLE",
        "effect_estimated": "NOT_RUN",
        "interpretation": "count decision only; requires separate raw/token/ledger validation",
    }


def stage_decision(parent_counts, stage=1, prior=None):
    """Apply frozen branches without mutating caller counts or prior decisions."""
    stage = _stage(stage)
    if not isinstance(parent_counts, dict):
        raise ValueError("A registered parent-to-count mapping is required")
    if stage == 1:
        if prior is not None or set(parent_counts) != set(PARENTS):
            raise ValueError("Complete three-parent stage 1 without a prior is required")
        parents = {}
        for parent in PARENTS:
            counts = _counts(parent_counts[parent])
            parents[parent] = {
                "status": stage1_gate(counts),
                "stage1_counts": counts,
                "stage2_counts": None,
                "combined_counts": dict(counts),
                "combined_n": 192,
            }
        return _decision(stage, parents)
    if not isinstance(prior, dict) or type(prior.get("stage")) is not int or prior["stage"] != 1:
        raise ValueError("Stage 2 requires a complete stage-1 decision")
    try:
        first = {parent: prior["parents"][parent]["stage1_counts"] for parent in PARENTS}
        rebuilt = stage_decision(first)
        unchanged = json.dumps(prior, sort_keys=True, allow_nan=False) == json.dumps(
            rebuilt, sort_keys=True, allow_nan=False
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid stage-1 decision") from exc
    if not unchanged or prior["status"] != "STAGE2_REQUIRED":
        raise ValueError("Altered, PASS, or FAIL prior cannot authorize stage 2")
    pending = prior["stage2_parents"]
    if set(parent_counts) != set(pending):
        raise ValueError("Exactly the prior GRAY parents must supply stage-2 counts")
    parents = copy.deepcopy(prior["parents"])
    for parent in pending:
        second = _counts(parent_counts[parent])
        first = parents[parent]["stage1_counts"]
        parents[parent].update(
            status=stage2_gate(first, second),
            stage2_counts=second,
            combined_counts={key: first[key] + second[key] for key in METRICS},
            combined_n=384,
        )
    return _decision(stage, parents, completed=pending)
