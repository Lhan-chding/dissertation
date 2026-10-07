"""Three frozen randomized schedules with shared common/replay slots in all arms."""

from __future__ import annotations

import copy
import random
from collections import Counter
from pathlib import Path

from .schema import ARMS, MICROBATCH_SLOTS, SCHEDULE_SEEDS, digest, read_bound, read_jsonl


def cycle_items(ids, count, seed):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty pool required")
    rng, result = random.Random(seed), []
    while len(result) < count:
        order = sorted(ids)
        rng.shuffle(order)
        result.extend(order)
    return result[:count]


def build_schedule(common_ids, replay_ids, donor_roots, donor_ids, seed):
    if seed not in SCHEDULE_SEEDS or (len(common_ids), len(replay_ids), len(donor_roots)) != (
        182,
        64,
        32,
    ):
        raise ValueError("Frozen pool sizes and one of three registered seeds required")
    common = cycle_items(common_ids, 2816, seed + 10)
    replay = cycle_items(replay_ids, 1024, seed + 20)
    donors = cycle_items(donor_roots, 256, seed + 30)
    result = []
    for step in range(256):
        for arm in ARMS:
            for slot, tid in enumerate(common[11 * step : 11 * (step + 1)]):
                result.append(
                    dict(
                        block_seed=seed,
                        arm=arm,
                        update=step + 1,
                        slot=slot,
                        role="common",
                        task_id=tid,
                    )
                )
            result.append(
                dict(
                    block_seed=seed,
                    arm=arm,
                    update=step + 1,
                    slot=11,
                    role="donor",
                    task_id=donor_ids[(donors[step], arm)],
                    donor_root=donors[step],
                )
            )
            for slot, tid in enumerate(replay[4 * step : 4 * (step + 1)], 12):
                result.append(
                    dict(
                        block_seed=seed,
                        arm=arm,
                        update=step + 1,
                        slot=slot,
                        role="replay",
                        task_id=tid,
                    )
                )
    validate_schedule(result)
    return result


def validate_schedule(rows):
    if len(rows) != 256 * 16 * 3 or {row.get("block_seed") for row in rows} not in [
        {seed} for seed in SCHEDULE_SEEDS
    ]:
        raise ValueError("One full registered all-arm schedule required")
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
        if row["role"] != role or key in index or not isinstance(row["task_id"], str):
            raise ValueError("Duplicate slot or role mismatch")
        index[key] = row
    for step in range(1, 257):
        for slot in range(16):
            matched = [index[(arm, step, slot)] for arm in ARMS]
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
    return index


def grouped_microbatches():
    return [list(group) for group in MICROBATCH_SLOTS]


class ExplicitScheduleSampler:
    """Cursor advances only after an applied update, and is bound to schedule bytes."""

    def __init__(self, rows_or_path, arm, *, committed_step=0):
        if arm not in ARMS:
            raise ValueError("Unregistered arm")
        rows = (
            read_jsonl(rows_or_path)
            if isinstance(rows_or_path, (str, Path))
            else copy.deepcopy(rows_or_path)
        )
        index = validate_schedule(rows)
        self.arm = arm
        self.schedule_id = digest(rows)
        self._steps = tuple(
            tuple(copy.deepcopy(index[(arm, step, slot)]) for slot in range(16))
            for step in range(1, 257)
        )
        self._step = 0
        self.load_state_dict(
            {
                "schema": "SER-J2-explicit-cursor-v1",
                "schedule_id": self.schedule_id,
                "arm": arm,
                "committed_step": committed_step,
            }
        )

    @classmethod
    def from_run(cls, run, block, arm, *, committed_step=0):
        if type(block) is not int or block not in range(3):
            raise ValueError("Registered block 0,1,2 required")
        return cls(
            read_bound(run, f"manifests/schedule_{SCHEDULE_SEEDS[block]}.jsonl"),
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
        return {
            "schema": "SER-J2-explicit-cursor-v1",
            "schedule_id": self.schedule_id,
            "arm": self.arm,
            "committed_step": self._step,
        }

    def load_state_dict(self, state):
        expected = {"schema", "schedule_id", "arm", "committed_step"}
        if (
            set(state) != expected
            or state["schema"] != "SER-J2-explicit-cursor-v1"
            or state["schedule_id"] != self.schedule_id
            or state["arm"] != self.arm
        ):
            raise ValueError("Resume cursor belongs to different arm/schedule")
        if type(state["committed_step"]) is not int or not 0 <= state["committed_step"] <= 256:
            raise ValueError("Invalid committed step")
        self._step = state["committed_step"]
