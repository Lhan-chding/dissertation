"""Reference-exact deterministic roots; extra exclusions are applied before freeze."""

from __future__ import annotations

import random
from collections import Counter

from .schema import ARMS, canonical_target, public_prompt, stable_id, valid_vector, validate_public


def make_task(world, observed, center, j, root_id, split, ordinal, family="cross_series"):
    if (
        not valid_vector(world)
        or not valid_vector(observed)
        or [k for k in range(4) if world[k] != observed[k]] != [j]
    ):
        raise ValueError("Exactly one corruption in the legal numeric domain required")
    if family == "cross_series":
        if center not in range(4):
            raise ValueError("Cross hub required")
        pairs = sorted(tuple(sorted((center, k))) for k in range(4) if k != center)
        h, b = [], []
        for left, right in pairs:
            row = [0] * 4
            row[left] = row[right] = 1
            h.append(row)
            b.append(world[left] + world[right])
    elif family == "trend":
        h, b = [[1, -2, 1, 0], [0, 1, -2, 1]], [0, 0]
    elif family == "duplicate_encoding":
        h, b = [[int(k == j) for k in range(4)]], [world[j]]
    else:
        raise ValueError("Unknown task family")
    tid = stable_id("SER-J2", root_id, family, center, j)
    task = dict(
        task_id=tid,
        root_id=root_id,
        base_instance_id=root_id,
        split=split,
        family=family,
        observed=list(observed),
        H_original=h,
        b_original=b,
        legal_domain=[0, 99],
        operation=("sum4", "difference_pairs", "range4")[ordinal % 3],
        template_version="historical-O0-text-SER-J2-v1",
        chart_type="grouped_bar" if ordinal % 2 == 0 else "line",
        interface="SYMBOLIC_FRESH",
        image_path=None,
        image_sha256=None,
    )
    validate_public(task)
    audit = dict(
        task_id=tid,
        root_id=root_id,
        split=split,
        true_world=list(world),
        corrupted_index=j,
        center=center,
        truth_orbit_key=sorted(world),
    )
    return task, audit


def build_new_tasks(exclusion_orbits):
    seen = {tuple(sorted(orbit)) for orbit in exclusion_orbits}
    starting_count = len(seen)
    roots = []

    def get_root(rng, split, index, family):
        if family == "trend":
            choices = [
                (a, a + d, a + 2 * d, a + 3 * d)
                for a in range(100)
                for d in range(1, 34)
                if a + 3 * d <= 99 and (a, a + d, a + 2 * d, a + 3 * d) not in seen
            ]
            if not choices:
                raise ValueError("No unused trend orbits remain; do not expand the domain")
            world = list(rng.choice(choices))
            if rng.randrange(2):
                world.reverse()
        else:
            for _ in range(100000):
                world = rng.sample(range(100), 4)
                if tuple(sorted(world)) not in seen:
                    break
            else:
                raise ValueError("Unable to sample a new root")
        seen.add(tuple(sorted(world)))
        replacements = []
        for value in world:
            alternate = rng.randrange(99)
            replacements.append(alternate + (alternate >= value))
        rid = stable_id("SER-J2-root", split, index, family, world, replacements)
        root = dict(
            root_id=rid, split=split, family=family, true_world=world, replacements=replacements
        )
        roots.append(root)
        return root

    def from_root(root, center, j, split, index):
        observed = list(root["true_world"])
        observed[j] = root["replacements"][j]
        return make_task(
            root["true_world"], observed, center, j, root["root_id"], split, index, root["family"]
        )

    files = {}
    donors, donor_audit, donor_targets = [], [], []
    rng = random.Random(107071)
    for index in range(32):
        root = get_root(rng, "DONOR_TRAIN", index, "cross_series")
        for arm, center in ARMS.items():
            task, audit = from_root(root, center, 1, "DONOR_TRAIN", index)
            donors.append(task)
            donor_audit.append(audit)
            donor_targets.append(
                dict(
                    arm=arm,
                    task_id=task["task_id"],
                    root_id=root["root_id"],
                    target=canonical_target(root["true_world"]),
                    label_source="generated_gold_for_controlled_intervention",
                )
            )
    files.update(
        {
            "donors_public.jsonl": donors,
            "donors_audit.jsonl": donor_audit,
            "donor_targets.jsonl": donor_targets,
        }
    )
    diagnostic, diag_audit = [], []
    rng = random.Random(107072)
    for index in range(2):
        root = get_root(rng, "E_DIAG", index, "cross_series")
        for center in range(4):
            for j in range(4):
                task, audit = from_root(root, center, j, "E_DIAG", index)
                diagnostic.append(task)
                diag_audit.append(audit)
    for family in ("trend", "duplicate_encoding"):
        for index in range(8):
            root = get_root(rng, "E_DIAG", index, family)
            task, audit = from_root(root, None, index % 4, "E_DIAG", index)
            diagnostic.append(task)
            diag_audit.append(audit)
    confirm, confirm_audit = [], []
    rng = random.Random(107073)
    for index in range(128):
        root = get_root(rng, "E_CONFIRM", index, "cross_series")
        cells = (
            [(center, j) for center in range(4) for j in range(4)]
            if index < 32
            else [(2, 2), (3, 3)]
        )
        for center, j in cells:
            task, audit = from_root(root, center, j, "E_CONFIRM", index)
            task["root_cohort"] = audit["root_cohort"] = "CORE32" if index < 32 else "EXTRA96"
            confirm.append(task)
            confirm_audit.append(audit)
    for family, count in (("trend", 64), ("duplicate_encoding", 32)):
        for index in range(count):
            root = get_root(rng, "E_CONFIRM", index, family)
            task, audit = from_root(root, None, index % 4, "E_CONFIRM", index)
            task["root_cohort"] = audit["root_cohort"] = family
            confirm.append(task)
            confirm_audit.append(audit)
    files.update(
        {
            "E_DIAG/tasks_public.jsonl": diagnostic,
            "E_DIAG/audit_only.jsonl": diag_audit,
            "E_CONFIRM/tasks_public.jsonl": confirm,
            "E_CONFIRM/audit_only.jsonl": confirm_audit,
            "new_root_registry_AUDIT_ONLY.jsonl": roots,
        }
    )
    files["new_prompts_public.jsonl"] = [
        dict(task_id=task["task_id"], split=task["split"], prompt=public_prompt(task))
        for task in donors + diagnostic + confirm
    ]
    return files, {
        "historical_exclusion_orbits": starting_count,
        "new_roots": len(roots),
        "new_unique_repair_checks": len(donors + diagnostic + confirm),
        "confirm_cross_cells": dict(
            Counter(
                f"c{row['center'] + 1}j{row['corrupted_index'] + 1}"
                for row in confirm_audit
                if row["center"] is not None
            )
        ),
    }
