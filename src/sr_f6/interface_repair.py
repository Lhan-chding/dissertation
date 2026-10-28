"""Prepare a single interface-v2 screen from existing DEV_CAL, without sampling.

The output is a reviewed input/seed plan, not a deployment, launch permission or
evidence of passing measurement. Old run data, prompts and FAIL remain intact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from srf6_ref.core import FAMILIES, seed_for

from .common import digest, file_hash, safe_evidence, write_once
from .data import public_prompt
from .interface_protocol import VERSION, WIRE_PROTOCOL, format_contract, response_shape

MASTER_SEED = 20261028
PARENTS = ("b0", "b1", "b2")
FACTORS = ("P0", "P1")


def screen_inputs(public, private):
    """Validate complete original DEV_CAL; derive only the fixed first two roots."""
    if not isinstance(public, list) or not isinstance(private, list):
        raise ValueError("Complete public/private DEV_CAL lists required")
    expected = {
        f"DEV_CAL_{family}_{index:04d}_{factor}_v{variant}": (
            family,
            index,
            factor,
            variant,
        )
        for family in FAMILIES
        for index in range(8)
        for factor in FACTORS
        for variant in range(4)
    }
    if any(not isinstance(row, dict) for row in public + private):
        raise ValueError("Question objects required")
    tasks = {row.get("qid"): row for row in private}
    if (
        len(public) != 384
        or len(private) != 384
        or set(tasks) != set(expected)
        or {row.get("qid") for row in public} != set(expected)
    ):
        raise ValueError("Exactly 384 unique registered DEV_CAL questions required")
    selected = []
    for row in sorted(public, key=lambda item: item["qid"]):
        family, index, factor, variant = expected[row["qid"]]
        task = tasks[row["qid"]]
        identity = dict(
            split="DEV_CAL",
            family=family,
            factor=factor,
            variant=variant,
            root_id=f"DEV_CAL_{family}_{index:04d}",
        )
        if any(row.get(key) != value or task.get(key) != value for key, value in identity.items()):
            raise ValueError("Public/private DEV_CAL lineage differs")
        model_input = row.get("model_input")
        if (
            row.get("protocol") != WIRE_PROTOCOL
            or not isinstance(model_input, dict)
            or set(model_input) != {"prompt", "image_file", "image_sha256"}
        ):
            raise ValueError("Original structured model-input boundary required")
        image = model_input["image_file"]
        pin = model_input["image_sha256"]
        if (
            not isinstance(image, str)
            or PurePosixPath(image).is_absolute()
            or ".." in PurePosixPath(image).parts
            or not image.startswith("DATA/images/")
            or not isinstance(pin, str)
            or re.fullmatch(r"[0-9a-f]{64}", pin) is None
        ):
            raise ValueError("Bound source image reference required")
        old_prompt = public_prompt(task)
        if (
            model_input["prompt"] != old_prompt
            or row.get("prompt_sha256") != hashlib.sha256(old_prompt.encode()).hexdigest()
        ):
            raise ValueError("Source prompt differs from the historical protocol")
        # Validate public query grammar for every task, without reading its truth.
        prompt = public_prompt(task, prompt_version=VERSION)
        if index >= 2:
            continue
        new = copy.deepcopy(row)
        new.update(
            prompt_version=VERSION,
            source_prompt_sha256=row["prompt_sha256"],
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        )
        new["model_input"]["prompt"] = prompt
        selected.append(new)
    if len(selected) != 96:
        raise ValueError("Fixed screen must have 96 distinct question/factor inputs")
    return selected


def screen_plan(inputs):
    """No effect-dependent choice, replacement draws, or complete-G8 claim."""
    if len(inputs) != 96 or len({row["qid"] for row in inputs}) != 96:
        raise ValueError("Exactly 96 distinct screen inputs required")
    attempts = [
        {
            "parent": parent,
            "qid": row["qid"],
            "root_id": row["root_id"],
            "factor": row["factor"],
            "variant": row["variant"],
            "draw": draw,
            "seed": seed_for(
                MASTER_SEED, VERSION, "screen", parent, row["root_id"], row["variant"], draw
            ),
            "model_input_sha256": digest(row["model_input"]),
        }
        for parent in PARENTS
        for row in inputs
        for draw in range(2)
    ]
    return {
        "schema": "srf6-interface-screen-input-plan-v1",
        "status": "INPUT_PLAN_ONLY_NOT_DEPLOYED",
        "prompt_version": VERSION,
        "wire_protocol": WIRE_PROTOCOL,
        "old_run_status": "FAIL_UNCHANGED",
        "source_split": "DEV_CAL",
        "root_indices": [0, 1],
        "parents": list(PARENTS),
        "factors": list(FACTORS),
        "master_seed": MASTER_SEED,
        "seed_excludes_factor": True,
        "draws_per_question": 2,
        "complete_G8_credit": "NOT_APPLICABLE",
        "expected_new_answers": 576,
        "per_parent_factor_n": 96,
        "required_counts": {"V": 87, "S_known": 87, "B_known": 87},
        "threshold_interpretation": "engineering screen, not a population confidence bound",
        "max_prompt_candidates": 1,
        "on_failure": "STOP_PRESERVE_ALL_ROWS",
        "finish_policy": "balanced_only_unchanged",
        "scorer_changed": False,
        "constrained_decoding": False,
        "generation_allowed": False,
        "training_allowed": False,
        "gate_status": "NOT_SIGNED",
        "model_calls": 0,
        "optimizer_updates": 0,
        "engineering_valid": "PARTIAL_LOCAL_CPU_ONLY",
        "measurement_valid": "NOT_RUN",
        "comparison_supported": "NOT_RUN",
        "effect_estimated": "NOT_RUN",
        "confirmation": {
            "status": "NOT_PREPARED_NOT_AUTHORIZED",
            "requires": "new roots and source/data/processor registration before first output",
            "expected_new_answers": 9216,
            "per_parent_factor_n": 1536,
            "required_counts": {"V": 1383, "S_known": 1383, "B_known": 1383},
            "other_support_conditions": "all original F6 conditions unchanged",
        },
        "attempts": attempts,
    }


def prepare_screen(source_root, output):
    """Write to a fresh sibling destination; never copy or change source images."""
    source_root = Path(source_root).resolve(strict=True)
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Preserve the existing repair plan")
    if destination.resolve().is_relative_to(source_root):
        raise PermissionError("Repair output must be outside the original run")
    if any(parent.is_symlink() for parent in destination.parents):
        raise PermissionError("Repair output cannot use symlinked parents")
    paths = {
        name: safe_evidence(source_root, f"DATA/DEV_CAL/{name}.jsonl")
        for name in ("MODEL_INPUTS", "TASKS_PRIVATE")
    }
    source_rows = {
        name: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for name, path in paths.items()
    }
    inputs = screen_inputs(source_rows["MODEL_INPUTS"], source_rows["TASKS_PRIVATE"])
    plan = screen_plan(inputs)
    plan["source_root"] = str(source_root)
    plan["source_files"] = {
        name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()
    }
    plan["image_verification"] = "REFERENCES_PRESERVED_CONTENT_NOT_RECHECKED"
    plan["source_code_sha256"] = {
        name: file_hash(Path(__file__).with_name(name))
        for name in ("interface_repair.py", "interface_protocol.py", "data.py", "measurement.py")
    }
    destination.mkdir(parents=True)
    with (destination / "MODEL_INPUTS.jsonl").open("x") as stream:
        for row in inputs:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    plan["model_inputs_sha256"] = file_hash(destination / "MODEL_INPUTS.jsonl")
    write_once(destination / "SCREEN_INPUT_PLAN.json", plan)
    write_once(destination / "RESPONSE_SHAPE.json", response_shape())
    (destination / "FORMAT_CONTRACT.txt").write_text(format_contract() + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_screen(args.source_root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
