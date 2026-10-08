"""J23 schedules inherit legacy slots exactly; no ID sorting or resampling."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

from ..exposure_substitution.schedule import validate_schedule as validate_legacy_schedule
from .schema import ARMS, MICROBATCH_SLOTS, SCHEDULE_SEEDS, digest, read_jsonl, read_training_bound


def inherit_schedule(legacy_rows, task_id_map, root_id_map, donor_ids):
    """Map all four arms to one verified historical slot order, preserving bytes' meaning."""
    legacy = validate_legacy_schedule(legacy_rows)
    rows = []
    for update in range(1, 257):
        for arm in ARMS:
            for slot in range(16):
                original = legacy[("A_LOCAL_C1", update, slot)]
                row = {k: copy.deepcopy(v) for k, v in original.items()}
                row["arm"] = arm
                if slot == 11:
                    root = root_id_map[original["donor_root"]]
                    row["donor_root"] = root
                    row["task_id"] = donor_ids[(root, arm)]
                else:
                    row["task_id"] = task_id_map[original["task_id"]]
                rows.append(row)
    validate_schedule(rows)
    return rows


def validate_schedule(rows):
    if len(rows) != 256 * 16 * 4 or {row.get("block_seed") for row in rows} not in [
        {s} for s in SCHEDULE_SEEDS
    ]:
        raise ValueError("One complete registered four-arm schedule required")
    index = {}
    for row in rows:
        expected = {"block_seed", "arm", "update", "slot", "role", "task_id"}
        if row.get("role") == "donor":
            expected.add("donor_root")
        if set(row) != expected or row["arm"] not in ARMS:
            raise ValueError("Unexpected schedule fields/arm")
        if (
            type(row["update"]) is not int
            or type(row["slot"]) is not int
            or not 1 <= row["update"] <= 256
            or not 0 <= row["slot"] < 16
        ):
            raise ValueError("Invalid update/slot")
        role = "common" if row["slot"] < 11 else "donor" if row["slot"] == 11 else "replay"
        key = row["arm"], row["update"], row["slot"]
        if (
            row["role"] != role
            or key in index
            or not isinstance(row["task_id"], str)
            or not row["task_id"]
        ):
            raise ValueError("Duplicate slot or role mismatch")
        index[key] = row
    for update in range(1, 257):
        for slot in range(16):
            matched = [index[(arm, update, slot)] for arm in ARMS]
            field = "donor_root" if slot == 11 else "task_id"
            if len({row[field] for row in matched}) != 1:
                raise ValueError("Cross-arm common/replay/donor-root slot mismatch")
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if Counter(row["role"] for row in selected) != {
            "common": 2816,
            "donor": 256,
            "replay": 1024,
        }:
            raise ValueError("Exposure count mismatch")
        donors = Counter(row["donor_root"] for row in selected if row["role"] == "donor")
        if len(donors) != 32 or set(donors.values()) != {8}:
            raise ValueError("Every donor root must receive eight exposures")
        if (
            len({row["task_id"] for row in selected if row["role"] == "common"}) != 182
            or len({row["task_id"] for row in selected if row["role"] == "replay"}) != 64
        ):
            raise ValueError("Inherited common/replay pool sizes changed")
        root_tasks = {}
        for row in selected:
            if row["role"] == "donor":
                root_tasks.setdefault(row["donor_root"], set()).add(row["task_id"])
        if (
            any(len(tasks) != 1 for tasks in root_tasks.values())
            or len(set().union(*root_tasks.values())) != 32
        ):
            raise ValueError("Donor root/task mapping must be one-to-one per arm")
    return index


def grouped_microbatches():
    return [list(group) for group in MICROBATCH_SLOTS]


class ExplicitScheduleSampler:
    def __init__(self, rows_or_path, arm, *, committed_step=0):
        if arm not in ARMS:
            raise ValueError("Unregistered J23 arm")
        rows = (
            read_jsonl(rows_or_path)
            if isinstance(rows_or_path, (str, Path))
            else copy.deepcopy(rows_or_path)
        )
        index = validate_schedule(rows)
        self.arm, self.schedule_id = arm, digest(rows)
        self._steps = tuple(
            tuple(copy.deepcopy(index[(arm, step, slot)]) for slot in range(16))
            for step in range(1, 257)
        )
        self._step = 0
        self.load_state_dict(
            dict(
                schema="SER-J23-explicit-cursor-v1",
                schedule_id=self.schedule_id,
                arm=arm,
                committed_step=committed_step,
            )
        )

    @classmethod
    def from_run(cls, run, block, arm, *, committed_step=0, technical=False):
        if type(block) is not int or block not in range(3):
            raise ValueError("Registered block 0,1,2 required")
        return cls(
            read_training_bound(
                run, f"manifests/schedule_{SCHEDULE_SEEDS[block]}.jsonl", technical=technical
            ),
            arm,
            committed_step=committed_step,
        )

    @property
    def step(self):
        return self._step

    def peek(self):
        if self._step >= 256:
            raise StopIteration("All 256 registered updates committed")
        return copy.deepcopy(list(self._steps[self._step]))

    def commit(self, update=None):
        if self._step >= 256 or (update is not None and update != self._step + 1):
            raise ValueError("Only the next scheduled update may commit")
        self._step += 1

    def state_dict(self):
        return dict(
            schema="SER-J23-explicit-cursor-v1",
            schedule_id=self.schedule_id,
            arm=self.arm,
            committed_step=self._step,
        )

    def load_state_dict(self, state):
        if (
            set(state) != {"schema", "schedule_id", "arm", "committed_step"}
            or state["schema"] != "SER-J23-explicit-cursor-v1"
            or state["schedule_id"] != self.schedule_id
            or state["arm"] != self.arm
        ):
            raise ValueError("Resume cursor belongs to different arm/schedule")
        if type(state["committed_step"]) is not int or not 0 <= state["committed_step"] <= 256:
            raise ValueError("Invalid committed step")
        self._step = state["committed_step"]
