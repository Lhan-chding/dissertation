"""Frozen SR-F6 schedules; parents have independent order random streams."""

from __future__ import annotations

import random

from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.planning import ARMS
from srf6_ref.planning import draw_seed as _draw_seed

VERSION = "srf6-planning-v1-parent-order-independent"


def root_schedule(roots_per_family, master, parent, repeat, *, split="TRAIN"):
    """One exposure per root, paired across all arms of one parent/repeat.

    The reference omitted parent from its order RNG. This separate production
    stream implements independent training repeats across fixed parents without
    changing the shared within-parent arm pairing.
    """
    if type(roots_per_family) is not int or roots_per_family <= 0 or roots_per_family % 8:
        raise ValueError("Positive multiple of eight roots per family required")
    if parent not in {"b0", "b1", "b2"} or type(repeat) is not int or repeat not in {0, 1, 2}:
        raise ValueError("Unregistered parent or repeat")
    if split not in {"TRAIN", "DEV_PILOT"}:
        raise ValueError("Only training and pilot roots enter training schedules")
    if split == "DEV_PILOT" and (repeat != 0 or roots_per_family != 8):
        raise ValueError("Pilot has one repeat and eight roots per family")
    rng = random.Random(seed_for(master, "task_order", parent, repeat, split))
    updates = []
    for window in range(roots_per_family // 8):
        roots = [
            f"{split}_{family}_{index:04d}"
            for family in FAMILIES
            for index in range(window * 8, (window + 1) * 8)
        ]
        rng.shuffle(roots)
        updates.extend(roots[index : index + 4] for index in range(0, len(roots), 4))
    return updates


def draw_seed(master, parent, repeat, step, slot, variant, draw):
    """Retain the registered draw stream exactly, without arm or factor."""
    if parent not in {"b0", "b1", "b2"}:
        raise ValueError("Unregistered parent")
    for name, value, lower, upper in (
        ("repeat", repeat, 0, 2),
        ("step", step, 1, 72),
        ("slot", slot, 0, 3),
        ("variant", variant, 0, 3),
        ("draw", draw, 0, 7),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Invalid {name}")
    return _draw_seed(master, parent, repeat, step, slot, variant, draw)


def formal_paths(config):
    """Randomize six-arm execution within each fixed parent/repeat block."""
    if config["arms"] != list(ARMS):
        raise ValueError("All six registered arms are required")
    output = []
    for parent in config["model"]["parents"]:
        for repeat in config["training"]["repeats"]:
            arms = list(ARMS)
            order_seed = seed_for(config["master_seed"], "arm_order", parent, repeat)
            random.Random(order_seed).shuffle(arms)
            for block_order, arm in enumerate(arms):
                factor, credit = ARMS[arm]
                output.append(
                    {
                        "parent": parent,
                        "repeat": repeat,
                        "arm": arm,
                        "factor": factor,
                        "credit": credit,
                        "run_id": f"{config['experiment_id']}/{parent}/r{repeat}/{arm}",
                        "block_order": block_order,
                        "arm_order_seed": order_seed,
                        "status": "NOT_RUN",
                        "phase": "FORMAL",
                    }
                )
    return output
