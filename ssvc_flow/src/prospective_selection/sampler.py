"""Common metadata schedules and independent role-bound random streams."""

from __future__ import annotations

from collections import defaultdict

from ..core import canonical_hash
from ..r4_inputs import STRATA


def build_schedule(prompts, *, seed, steps, batch_size=4, role, repeat=0):
    if role not in {"source", "continuation"}:
        raise ValueError("Unknown training schedule role")
    groups, seen = defaultdict(list), set()
    for prompt in prompts:
        pid = prompt["prompt_id"]
        if pid in seen:
            raise ValueError("Duplicate prompt in training pool")
        seen.add(pid)
        stratum = (prompt["family"], prompt["interface"])
        if stratum not in STRATA:
            raise ValueError("Unknown training stratum")
        groups[stratum].append(pid)
    lengths = {len(groups[s]) for s in STRATA}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("Training pool must have six balanced nonempty strata")
    for stratum in STRATA:
        groups[stratum].sort(
            key=lambda pid: (canonical_hash(["prospective-schedule-v1", seed, role, pid]), pid)
        )
    order = [groups[s][i] for i in range(next(iter(lengths))) for s in STRATA]
    count = steps * batch_size
    if count > len(order) or min(steps, batch_size) < 1:
        raise ValueError("Schedule requires enough prompts without replacement")
    result = {
        "sampler_seed": seed,
        "role": role,
        "repeat": repeat,
        "train_steps": [order[i : i + batch_size] for i in range(0, count, batch_size)],
    }
    result["schedule_hash"] = canonical_hash(result)
    result["schedule_id"] = f"{role}-{repeat}-{result['schedule_hash'][:16]}"
    return result


def _seed(parts):
    return int(canonical_hash(parts)[:16], 16) % (2**63 - 1)


def training_seed(*, experiment_seed, lineage, repeat, step, prompt_id, draw):
    """No recipe/origin/selector key: paired recipes share random-number starts."""
    return _seed(
        ["prospective-training-v1", experiment_seed, str(lineage), repeat, step, prompt_id, draw]
    )


def evaluation_seed(
    *,
    experiment_seed,
    lineage,
    repeat,
    policy_id,
    panel_id,
    horizon,
    prompt_id,
    draw,
    role="evaluation",
):
    if role not in {"evaluation", "predecision", "historical_E", "diagnostic"}:
        raise ValueError("Unknown evaluation role")
    if not policy_id:
        raise ValueError("Independent evaluation RNG requires actual policy identity")
    return _seed(
        [
            "prospective-evaluation-v1",
            role,
            experiment_seed,
            str(lineage),
            repeat,
            policy_id,
            panel_id,
            horizon,
            prompt_id,
            draw,
        ]
    )
