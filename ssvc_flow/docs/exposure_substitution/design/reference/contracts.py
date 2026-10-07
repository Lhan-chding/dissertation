"""CPU-only mathematical and schedule contracts; no model calls or GPU imports."""
from __future__ import annotations
import hashlib
import json
import math
import random
from collections import Counter
from typing import Sequence

ARMS = {"A_LOCAL_C1": 0, "B_FORWARD_C4": 3, "C_FORWARD_C3": 2}
SYSTEM = ("You are repairing a four-integer chart record. Return only one JSON array\n"
          "containing four integers from 0 to 99. Do not include an explanation.")
USER = ("The record order is [a,b,c,d]. For the chart, a and b are the two series\n"
        "at the first x position; c and d are the same two series at the second position.\n"
        "The observed record is {observed_json}.\n"
        "Exactly one value in this observed record is wrong.\n"
        "The following relationships are reliable:\n{cue_text}\n"
        "The downstream calculation is: {operation_expression}.\n"
        "Recover the correct record. Return [a,b,c,d], not the downstream answer.")
OPS = {"sum4": "a+b+c+d", "difference_pairs": "(a+b)-(c+d)",
       "range4": "max(a,b,c,d)-min(a,b,c,d)"}


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def valid(y):
    return isinstance(y, list) and len(y) == 4 and all(type(x) is int and 0 <= x <= 99 for x in y)


def parse(text):
    try:
        y = json.loads(text)
    except (ValueError, TypeError):
        return None
    return y if isinstance(y, list) and len(y) == 4 and all(type(x) is int for x in y) else None


def canonical_target(x):
    if not valid(x):
        raise ValueError("Target must contain exactly four legal integer values")
    return json.dumps(x, separators=(",", ":"))


def star(world, center):
    if center not in range(4) or not valid(list(world)):
        raise ValueError("Invalid star")
    pairs = sorted(tuple(sorted((center, k))) for k in range(4) if k != center)
    H, b = [], []
    for i, j in pairs:
        row = [0] * 4
        row[i] = row[j] = 1
        H.append(row)
        b.append(world[i] + world[j])
    return H, b


def satisfy(y, H, b):
    return y is not None and all(sum(v * t for v, t in zip(row, y)) == rhs for row, rhs in zip(H, b))


def verify(task, y):
    return bool(valid(y) and sum(a != b for a, b in zip(y, task["observed"])) == 1
                and satisfy(y, task["H_original"], task["b_original"]))


def solve_public(task):
    result = []
    for j in range(4):
        for v in range(100):
            if v == task["observed"][j]:
                continue
            y = list(task["observed"])
            y[j] = v
            if verify(task, y):
                result.append(y)
    return result


def deletion_projection(task, y):
    """Only keep one proposed edit; never query the gold world or invent a new value."""
    if y is None or len(y) != 4 or any(type(v) is not int for v in y):
        return None
    found = set()
    for k, value in enumerate(y):
        z = list(task["observed"])
        z[k] = value
        if verify(task, z):
            found.add(tuple(z))
    if len(found) > 1:
        raise ValueError("Unique-repair contract violated")
    return list(next(iter(found))) if found else None


def public_prompt(task):
    """Preserves historical O0 text; no audit fields are read."""
    if task["family"] == "trend":
        cue = "b - a = c - b = d - c"
    elif task["family"] == "cross_series":
        lines = []
        for row, rhs in zip(task["H_original"], task["b_original"]):
            ids = [i for i, value in enumerate(row) if value]
            if len(ids) != 2 or any(row[i] != 1 for i in ids):
                raise ValueError("Main O0 only accepts pair sums")
            lines.append(f"{'abcd'[ids[0]]} + {'abcd'[ids[1]]} = {rhs}")
        cue = "\n".join(lines)
    elif task["family"] == "duplicate_encoding":
        row = task["H_original"][0]
        ids = [i for i, value in enumerate(row) if value]
        if len(ids) != 1 or row[ids[0]] != 1:
            raise ValueError("Invalid duplicate cue")
        cue = f"{'abcd'[ids[0]]} = {task['b_original'][0]}"
    else:
        raise ValueError("Unregistered family")
    return {"system": SYSTEM, "user": USER.format(
        observed_json=json.dumps(task["observed"], separators=(",", ":")),
        cue_text=cue, operation_expression=OPS[task["operation"]])}


def make_task(world, observed, center, j, root_id, split, ordinal, family="cross_series"):
    if not valid(list(world)) or not valid(list(observed)):
        raise ValueError("Invalid numeric domain")
    if [k for k in range(4) if world[k] != observed[k]] != [j]:
        raise ValueError("Exactly one corruption required")
    if family == "cross_series":
        H, b = star(world, center)
    elif family == "trend":
        H, b = [[1, -2, 1, 0], [0, 1, -2, 1]], [0, 0]
    elif family == "duplicate_encoding":
        H = [[int(k == j) for k in range(4)]]
        b = [world[j]]
    else:
        raise ValueError("Unknown family")
    tid = stable_id("SER-J2", root_id, family, center, j)
    task = dict(task_id=tid, root_id=root_id, base_instance_id=root_id, split=split,
                family=family, observed=list(observed), H_original=H, b_original=b,
                legal_domain=[0, 99], operation=list(OPS)[ordinal % 3],
                template_version="historical-O0-text-SER-J2-v1", chart_type="grouped_bar" if ordinal % 2 == 0 else "line",
                interface="SYMBOLIC_FRESH", image_path=None, image_sha256=None)
    audit = dict(task_id=tid, root_id=root_id, split=split, true_world=list(world),
                 corrupted_index=j, center=center, truth_orbit_key=sorted(world))
    return task, audit


def cycle_items(ids: Sequence[str], n: int, seed: int):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty pool required")
    rng = random.Random(seed)
    result = []
    while len(result) < n:
        order = sorted(ids)
        rng.shuffle(order)
        result.extend(order)
    return result[:n]


def schedule(common_ids, replay_ids, donor_roots, donor_ids, seed):
    """One actual randomized schedule shared across A/B/C; donor only at slot 11."""
    if len(donor_roots) != 32:
        raise ValueError("32 matched donor roots required")
    common = cycle_items(common_ids, 2816, seed + 10)
    replay = cycle_items(replay_ids, 1024, seed + 20)
    donor_order = cycle_items(donor_roots, 256, seed + 30)
    result = []
    for s in range(256):
        for arm in ARMS:
            for slot, tid in enumerate(common[11 * s:11 * (s + 1)]):
                result.append(dict(block_seed=seed, arm=arm, update=s + 1, slot=slot, role="common", task_id=tid))
            result.append(dict(block_seed=seed, arm=arm, update=s + 1, slot=11, role="donor",
                               task_id=donor_ids[(donor_order[s], arm)], donor_root=donor_order[s]))
            for slot, tid in enumerate(replay[4 * s:4 * (s + 1)], 12):
                result.append(dict(block_seed=seed, arm=arm, update=s + 1, slot=slot, role="replay", task_id=tid))
    return result


def grouped_microbatches():
    # Donor alone: variant prompt lengths never alter padding of common examples.
    return [list(range(0, 4)), list(range(4, 8)), list(range(8, 11)), [11], list(range(12, 16))]


def lr_at(update):
    if not 1 <= update <= 256:
        raise ValueError("Update must be in [1,256]")
    return 1e-5 * min(update / 8, 1.0)


def specificity(B4, C4, B3, C3):
    return ((B4 - C4) - (B3 - C3)) / 2


def pass_at_k(n, c, k):
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("Invalid finite-sample pass@k parameters")
    return 1.0 if n-c < k else 1 - math.comb(n-c, k) / math.comb(n, k)


def wilson(c, n, z=1.959963984540054):
    if n < 1 or not 0 <= c <= n:
        raise ValueError("Invalid binomial count")
    p = c/n; den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0., mid-half), min(1., mid+half)


def safe_ratio(num, den):
    if den < 0 or num < 0 or num > den:
        raise ValueError("Invalid conditional counts")
    return num/den if den else None


def experiment_workload():
    return {
        "formal_sft_runs": 18,
        "formal_sft_updates": 18 * 256,
        "formal_sft_target_exposures": 18 * 256 * 16,
        "per_run_common_exposures": 2816,
        "per_run_donor_exposures": 256,
        "per_run_replay_exposures": 1024,
        "confirm_student_generations": 18 * 800 * 8,
        "confirm_parent_generations": 2 * 800 * 8,
        "diagnostic_parent_and_base_generations": 3 * 48 * 8,
        "diagnostic_student_generations": 18 * 2 * 48 * 8,
        "training_fit_sentinel_generations": 18 * 2 * 16 * 4,
        "formal_generation_total": 18*800*8 + 2*800*8 + 3*48*8 + 18*2*48*8 + 18*2*16*4,
        "additional_model_scoring": 0,
        "new_rl_updates": 0,
        "new_discovery_campaign": 0,
    }
