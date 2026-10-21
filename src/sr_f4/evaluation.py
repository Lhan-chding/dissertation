"""Frozen A/B endpoint panels, independent of observed model scores."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from sr_f3_ce.prompts import prompt_for_revision, prompt_from_frozen
from srf4_ref.sampling import seed_for

EVALUATION_SEED = 2026102224
FAMILIES = ("CROSS", "INTERVAL", "THRESHOLD", "TOPK")
ARMS = ("PARENT", "J_UNIFORM_NEW_REPEAT", "LZ")


def _as_rows(tasks):
    return list(tasks.values()) if isinstance(tasks, dict) else list(tasks)


def answer_only_panel(canonical):
    """16 independently selected roots/family, one chart/root, fixed before scores."""
    by_family = defaultdict(lambda: defaultdict(list))
    for task in canonical:
        by_family[task["family"]][task["root_id"]].append(task)
    selected = []
    for family in FAMILIES:
        roots = sorted(by_family[family])
        if len(roots) < 16:
            raise ValueError("answer_only needs at least 16 old TEST_ID roots per family")
        random.Random(seed_for(EVALUATION_SEED, "answer_only_roots", family)).shuffle(roots)
        for index, root in enumerate(roots[:16]):
            charts = sorted(by_family[family][root], key=lambda row: row["chart"])
            selected.append(charts[index % len(charts)])
    return sorted(selected, key=lambda row: row["qid"])


def build_a_eval_groups(tasks, block, external_tasks=(), *, arm=None):
    """Frozen panels with paired seeds and the registered arm-specific budgets.

    The caller may shard this immutable list. Parent step zero is evaluated once,
    with eight B_CONFIRM draws for B measurement; A uses the same first four.
    Only the two b0 A endpoints receive ChartQA. A workers consume step 64.
    """
    if block not in (0, 1, 2):
        raise ValueError("Only the three registered parent blocks are allowed")
    if arm not in ARMS:
        raise ValueError("An explicit registered evaluation arm is required")
    tasks = _as_rows(tasks)
    pools = defaultdict(list)
    for task in tasks:
        pools[task["pool"]].append(task)
    for pool in pools:
        pools[pool].sort(key=lambda row: row["qid"])
    expected = {
        "TEST_ID": 2048,
        "TEST_COMPOSITION": 256,
        "TEST_ARITY": 128,
        "TEST_LANGUAGE": 128,
        "TEST_RENDER": 128,
        "B_CONFIRM": 56,
    }
    if any(len(pools[pool]) != count for pool, count in expected.items()):
        raise ValueError("A endpoint inventory does not match fixed panel counts")
    canonical = [row for row in pools["TEST_ID"] if row["mode"] == "repair" and row["variant"] == 2]
    if len(canonical) != 256 or set(Counter(row["root_id"] for row in canonical).values()) != {2}:
        raise ValueError("Natural work requires both charts of all 128 canonical roots")
    panels = [
        (name, pools[name], "structured", 8 if name == "B_CONFIRM" and arm == "PARENT" else 4)
        for name in ("TEST_ID", "TEST_COMPOSITION", "TEST_ARITY", "B_CONFIRM")
    ]
    panels += [
        ("NATURAL_WORK", canonical, "natural_work", 2),
        ("ANSWER_ONLY", answer_only_panel(canonical), "answer_only", 4),
        ("TEST_LANGUAGE", pools["TEST_LANGUAGE"], "structured", 2),
        ("TEST_RENDER", pools["TEST_RENDER"], "structured", 2),
    ]
    if block == 0 and arm != "PARENT":
        external = sorted(_as_rows(external_tasks), key=lambda row: row["qid"])
        if len(external) != 2500 or any(row.get("pool") != "CHARTQA" for row in external):
            raise ValueError(
                "b0 A endpoints require the frozen complete 2500-question ChartQA inventory"
            )
        panels.append(("CHARTQA", external, "chartqa_plain", 1))
    groups = []
    for panel, selected, protocol, draws in panels:
        for task in selected:
            index = len(groups)
            groups.append(
                dict(
                    qid=task["qid"],
                    group_index=index,
                    G=draws,
                    protocol=protocol,
                    panel=panel,
                    group_id=f"EVAL_{panel}_{task['qid']}",
                    seeds=[
                        seed_for(EVALUATION_SEED, "eval", panel, task["qid"], draw)
                        for draw in range(draws)
                    ],
                )
            )
    if len({(g["qid"], g["protocol"], g["panel"]) for g in groups}) != len(groups):
        raise ValueError("Duplicate endpoint groups")
    return groups


def shard_groups(groups, shard, shards):
    if type(shards) is not int or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError("Invalid evaluation shard")
    return [group for group in groups if group["group_index"] % shards == shard]


def model_input_for(frozen_input, image, protocol="structured"):
    """Build the runtime's three-field input from model-only manifest rows.

    Audit task/gold sidecars are deliberately rejected. ChartQA uses its own
    already frozen official prompt, not the synthetic output contract.
    """
    allowed = {"qid", "pool", "chart", "prompt", "render_style"}
    if set(frozen_input) - allowed:
        raise ValueError("Use the model-input manifest, not an audit task record")
    prompt = frozen_input["prompt"]
    if protocol == "chartqa_plain":
        if frozen_input.get("pool") != "CHARTQA":
            raise ValueError("External protocol requires a ChartQA model input")
    elif protocol in ("structured", "natural_work", "answer_only"):
        prompt = prompt_from_frozen(prompt_for_revision(prompt, "v2"), protocol)
    else:
        raise ValueError("Unregistered A evaluation protocol")
    return dict(prompt=prompt, image_file=image["image_file"], image_sha256=image["image_sha256"])
