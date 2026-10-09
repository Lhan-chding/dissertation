"""F2 deterministic contracts, derived from the unchanged dispatched reference.

Production hardening: detect exact constant rewards before floating reduction.
The original reference and its manifest remain unchanged in docs/mm_dev_f2/design.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from fractions import Fraction

PLAN_ID = "MM-DEV-F2-QWEN35-9B-20261009"
CHARTS = ("grouped_bar", "line")
OPS = ("sum", "difference", "range")
STRATA = tuple((c, o) for c in CHARTS for o in OPS)
CELLS = ("LL", "HL", "LH", "HH")
CELL_COUNTS = {"a0": (2, 2, 2, 2), "aP": (1, 3, 1, 3), "aC": (1, 1, 3, 3)}


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def seed(namespace, *parts):
    return int.from_bytes(
        hashlib.sha256(canonical([PLAN_ID, namespace, *parts])).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def rank(items, namespace, *parts):
    return sorted(items, key=lambda x: (seed(namespace, *parts, x), canonical(x)))


def qid(pool, root, cell, chart, op):
    v, d = cell[0], cell[1]
    return f"mmdev-f2-{pool.lower()}-r{root:04d}-d{d.lower()}-v{v.lower()}-{chart}-{op}"


def slots(pool, action, repeat, roots, steps=32, batch=24):
    """Exact per-stratum 8-slot quotas per two updates. Same ranks across actions."""
    if steps not in (4, 32) or batch != 24 or action not in CELL_COUNTS:
        raise ValueError("unregistered schedule")
    if roots < max(CELL_COUNTS[action]) * (steps // 2):
        raise ValueError("pool too small for no-repeat streams")
    scope = "prep" if pool == "PREP" else "continue"
    counts = CELL_COUNTS[action]
    cells = [c for c, n in zip(CELLS, counts, strict=True) for _ in range(n)]
    cursors = Counter()
    ordered_roots = {}
    for chart, op in STRATA:
        for cell in CELLS:
            ordered_roots[(chart, op, cell)] = rank(
                list(range(roots)), "root-order", scope, repeat, chart, op, cell
            )
    result = []
    for block in range(steps // 2):
        halves = [[], []]
        for chart, op in STRATA:
            indices = rank(list(range(8)), "cell-position", scope, repeat, block, chart, op)
            for j, index in enumerate(indices):
                cell = cells[index]
                key = (chart, op, cell)
                root = ordered_roots[key][cursors[key]]
                cursors[key] += 1
                halves[j // 4].append(
                    {
                        "question_id": qid(pool, root, cell, chart, op),
                        "root_index": root,
                        "cell": cell,
                        "chart_type": chart,
                        "operation": op,
                    }
                )
        for half, items in enumerate(halves):
            # Shared index ordering; question identity differs through the quota mapping.
            positions = rank(list(range(24)), "within-update", scope, repeat, block, half)
            for slot, index in enumerate(positions):
                result.append({"logical_step": 2 * block + half + 1, "slot": slot, **items[index]})
    assert len(result) == steps * batch
    assert len({x["question_id"] for x in result}) == len(result)
    return result


def advantages(rewards, eps=1e-8):
    if len(rewards) != 8 or any(
        isinstance(r, bool) or not math.isfinite(float(r)) or not 0 <= r <= 1 for r in rewards
    ):
        raise ValueError("eight finite rewards in [0,1] are required")
    if len(set(rewards)) == 1:
        return [0.0] * 8
    vals = list(map(float, rewards))
    m = sum(vals) / 8
    v = sum((r - m) ** 2 for r in vals) / 8
    if v == 0:
        return [0.0] * 8
    return [(r - m) / (math.sqrt(v) + eps) for r in vals]


def preparation_reward(answer_correct, readings, truth, beta=Fraction(1, 2)):
    if not 0 <= beta <= 1:
        raise ValueError("beta out of range")
    if readings is None:
        read = Fraction(0)
    else:
        if len(readings) != len(truth):
            raise ValueError("parser must mark wrong arity as missing")
        read = Fraction(sum(r == v for r, v in zip(readings, truth, strict=True)), len(truth))
    return (1 - beta) * int(answer_correct) + beta * read


def shape(e):
    e = list(map(Fraction, e))
    if len(e) < 2:
        raise ValueError("arity >= 2")
    denom = sum(v * v for v in e)
    return None if denom == 0 else sum(e) ** 2 / (len(e) * denom)


def utility(dA, dP, lam=1.0, epsilon=0.02):
    if len(dA) != 6 or len(dP) != 6 or not all(math.isfinite(x) for x in list(dA) + list(dP)):
        raise ValueError("six finite strata")
    g = sum(dA) / 6
    interference = sum(max(0, -p) for p in dP) / 6
    return {
        "G": g,
        "I": interference,
        "U": g - lam * interference,
        "feasible_point_estimate": all(p >= -epsilon for p in dP),
    }


def allocation_counts():
    prep = 4 * 32 * 24 * 8
    response = 30 * 32 * 24 * 8
    probe = 5 * 64 * 24 * 4
    evaluate = 35 * 128 * 24 * 4
    return {
        "logical_scientific_updates": 1088,
        "training_prompt_slots": 1088 * 24,
        "preparation_rollouts": prep,
        "response_rollouts": response,
        "training_rollouts": prep + response,
        "probe_completions": probe,
        "evaluation_completions": evaluate,
        "scientific_completions": prep + response + probe + evaluate,
        "probe_self_forwards": probe,
        "probe_gold_forwards": 5 * 64 * 24,
        "training_reference_forwards": prep + response,
        "training_gradient_forwards": prep + response,
        "engine_natural_physical_updates": 8,
        "engine_natural_completions": 8 * 24 * 8,
        "optional_stress_physical_updates": 8,
        "optional_stress_completions": 8 * 24 * 8,
    }


def run_matrix():
    runs = []
    for rep in (0, 1):
        for recipe in ("A", "AP"):
            state = f"S_{recipe}_{rep}"
            runs.append(
                {
                    "run_id": f"prep_{recipe}_{rep}",
                    "phase": "PREP",
                    "start_state": "S0",
                    "output_state": state,
                    "preparation_recipe": recipe,
                    "repeat": rep,
                    "schedule_id": f"PREP_p{rep}",
                    "reference_state": "S0",
                    "steps": 32,
                    "optimizer_start": "fresh",
                    "preparation_master_seed": seed("preparation-master", rep),
                }
            )
    for state in ("S0", "S_A_0", "S_AP_0", "S_A_1", "S_AP_1"):
        for rep in (0, 1):
            for action in ("a0", "aP", "aC"):
                runs.append(
                    {
                        "run_id": f"cont_{state}_{action}_f{rep}",
                        "phase": "CONTINUE",
                        "start_state": state,
                        "action": action,
                        "repeat": rep,
                        "schedule_id": f"CONT_f{rep}_{action}",
                        "reference_state": state,
                        "steps": 32,
                        "optimizer_start": "fresh",
                        "future_master_seed": seed("future-master", rep),
                    }
                )
    return runs


def seed_usage():
    return {
        "common_lora_initialization": seed("common-lora", 0),
        "preparation_master_seeds": [seed("preparation-master", r) for r in (0, 1)],
        "future_master_seeds": [seed("future-master", r) for r in (0, 1)],
        "source_master_seed": seed("numeric-sources", 0),
        "engine_master_seed": seed("engine", 0),
        "bootstrap_seed": seed("bootstrap", 0),
        "definition": (
            "int.from_bytes(SHA256(UTF8 canonical JSON [PLAN_ID, namespace, "
            "*parts])[:8], big) & (2^63-1)"
        ),
        "canonical_json": "ensure_ascii=False, sort_keys=True, separators=(comma,colon)",
        "rollout_seed": (
            "seed(rollout, phase, repeat, question_id, within_question_sample_index); "
            "excludes action/state/worker/job IDs"
        ),
        "evaluation_seed": (
            "seed(evaluation, panel, question_id, sample_index); shared across states/endpoints"
        ),
        "request_id": (
            "digest([PLAN_ID,phase,run_id,logical_step,slot,sample_index,checkpoint_hash])"
        ),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {"counts": allocation_counts(), "seeds": seed_usage()}, ensure_ascii=False, indent=2
        )
    )
