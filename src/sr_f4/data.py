"""Deterministic SR-F4 world splits and fixed exposure, without model calls.

Only small canonical fact tables are hashed for split isolation. Images are
rendered once per world/style and reused; existing old images are hard-linked
where possible. This module does not rehash old models, archives or datasets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from sr_f3_ce.prompts import CONTRACT_V2
from srf4_ref.sampling import seed_for

from . import task_contract as oracle

CELLS = ("I-L", "I-R", "I-B", "T-S", "T-Q", "T-C")
FAMILIES = ("CROSS", "INTERVAL", "THRESHOLD", "TOPK")
NEW_COUNTS = {
    "B_DEV": (128, 200),
    "B_CONFIRM": (32, 56),
    "C_TRAIN": (384, 768),
    "C_DEV": (96, 192),
    "C_CONFIRM_ID": (120, 240),
    "C_CONFIRM_TRANSFER": (120, 240),
}
CATEGORIES = ("January", "February", "March", "April", "May", "June")
GENERATION_SEED = 2026102221


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        )
    )


def base_key(world):
    return json.dumps(
        [world["categories"], world["series"]["Alpha"], world["series"]["Beta"]],
        separators=(",", ":"),
    )


def world_digest(world):
    return hashlib.sha256(base_key(world).encode()).hexdigest()


def _world(rng, n):
    return {
        "categories": list(CATEGORIES[:n]),
        "series": {name: [rng.randint(10, 90) for _ in range(n)] for name in ("Alpha", "Beta")},
        "domain": [10, 90],
    }


def question(world, query, wording="standard"):
    cats = world["categories"]
    family = query["family"]
    if family == "INTERVAL":
        left = "including" if query.get("include_start", True) else "excluding"
        right = "including" if query.get("include_end", True) else "excluding"
        selection = (
            f"the categories from {cats[query['start']]} to {cats[query['end']]}, "
            f"{left} the start and {right} the end"
        )
        if query.get("positive_only", False):
            selection += ", retaining only categories where Alpha exceeds Beta"
    elif family == "TOPK":
        selection = (
            f"the {query.get('k', 2)} categories with the "
            f"{query.get('direction', 'largest')} {query.get('selection_series', 'Alpha')} values"
        )
    elif family == "CROSS":
        comparison = "exceeds" if query.get("comparator", "gt") == "gt" else "is at least"
        selection = f"the categories where Alpha {comparison} Beta"
    else:
        comparison = "greater than" if query.get("comparator", "ge") == "gt" else "at least"
        selection = f"the categories whose Alpha is {comparison} {query['threshold']}"
    value = query.get(
        "value", "Beta" if family == "TOPK" else "diff" if family == "INTERVAL" else "Alpha"
    )
    quantity = "the Alpha minus Beta differences" if value == "diff" else f"the {value} values"
    op = query["aggregate"]
    phrase = {
        "sum": "sum",
        "mean": "arithmetic mean",
        "range": "range (maximum minus minimum)",
        "min": "minimum",
        "max": "maximum",
        "count": "count",
    }[op]
    if op == "count":
        return f"Count {selection}."
    if wording == "transfer_wording":
        return (
            f"Restrict attention to {selection}. "
            f"Using {quantity} for that selection, report their {phrase}."
        )
    if wording == "b_paraphrase":
        return f"For {selection}, determine the {phrase} of {quantity}."
    return f"Select {selection}. What is the {phrase} of {quantity} over the selected categories?"


def _query_pair(cell, n, index, *, transfer=None):
    aggregate = "sum" if index % 2 == 0 else "mean"
    if transfer == "composition":
        aggregate = "range"
    if cell.startswith("I"):
        end = 5 if transfer == "scale" else 3
        first = dict(
            family="INTERVAL",
            start=0,
            end=end,
            include_start=True,
            include_end=True,
            value="diff",
            aggregate=aggregate,
        )
        second = dict(first)
        if cell in ("I-L", "I-B"):
            second["include_start"] = False
        if cell in ("I-R", "I-B"):
            second["include_end"] = False
    else:
        k = 2 if n == 4 or transfer == "scale" else 3
        first = dict(
            family="TOPK",
            k=k,
            direction="largest",
            selection_series="Alpha",
            value="Beta",
            aggregate=aggregate,
        )
        second = dict(first)
        if cell in ("T-S", "T-C"):
            second["selection_series"] = "Beta"
        if cell in ("T-Q", "T-C"):
            first["value"] = "Alpha" if cell == "T-Q" else "Beta"
            second["value"] = "Beta" if cell == "T-Q" else "Alpha"
    return [first, second]


def _representative_query(family, rng, index):
    op = ("sum", "mean", "range", "count")[(index // 4) % 4]
    if family == "INTERVAL":
        return dict(
            family=family,
            start=0,
            end=3,
            include_start=bool(index % 2),
            include_end=True,
            positive_only=bool(index % 3),
            value="diff",
            aggregate=op,
        )
    if family == "TOPK":
        return dict(
            family=family,
            selection_series="Alpha" if index % 2 else "Beta",
            k=2,
            direction="largest",
            value="Beta",
            aggregate=op,
        )
    if family == "CROSS":
        return dict(family=family, comparator="gt", value="Alpha", aggregate=op)
    return dict(
        family=family,
        comparator="ge",
        threshold=rng.choice((30, 60, 90)),
        value="Alpha",
        aggregate=op,
    )


def _relation(world, queries, answers, purpose):
    if len(queries) != 2:
        return None
    sets = [[world["categories"][i] for i in oracle.selected(world, q)] for q in queries]
    first, second = answers
    return {
        "kind": "equal"
        if purpose == "presentation"
        else "sum_equals_count_times_mean"
        if purpose == "operation"
        else "fixed_program_delta",
        "answers_exact": [oracle.encode(first), oracle.encode(second)],
        "answer_0_minus_answer_1": oracle.encode(first - second)
        if first is not None and second is not None
        else None,
        "selection_0_only": sorted(set(sets[0]) - set(sets[1])),
        "selection_1_only": sorted(set(sets[1]) - set(sets[0])),
        "selected_count": [len(x) for x in sets],
        "programs": queries,
        "evaluation_rule": (
            "Check both exact answers separately; also check the stated equality/delta. "
            "Merely unequal outputs do not satisfy this relation."
        ),
    }


def _make_pair(split, cell, index, used, *, purpose="selection", transfer=None, n=4, single=False):
    key = f"{split}_{cell.replace('-', '_')}_{index:03d}"
    rng = random.Random(seed_for(GENERATION_SEED, key))
    queries = _query_pair(cell, n, index, transfer=transfer) if not single else None
    if single:
        queries = [_representative_query(cell, rng, index)]
    elif purpose == "operation":
        queries[1] = dict(queries[0], aggregate="mean")
        queries[0] = dict(queries[0], aggregate="sum")
    elif purpose == "presentation":
        queries[1] = dict(queries[0])
    for _attempt in range(10000):
        world = _world(rng, n)
        fingerprint = world_digest(world)
        if fingerprint in used:
            continue
        answers = [oracle.execute(world, q) for q in queries]
        if (
            split.startswith("B_")
            and not single
            and purpose != "presentation"
            and answers[0] == answers[1]
        ):
            continue
        if split.startswith("C_") and any(
            answer is None or len(oracle.selected(world, query)) < 2
            for answer, query in zip(answers, queries, strict=True)
        ):
            continue
        break
    else:
        raise RuntimeError("Could not draw a registered non-collision world: " + key)
    used.add(fingerprint)
    relation = _relation(world, queries, answers, purpose)
    changed = [
        f"query.{k}"
        for k in sorted(set().union(*(set(q) for q in queries)))
        if len(queries) == 2 and queries[0].get(k) != queries[1].get(k)
    ]
    chart = "grouped_bar" if index % 2 == 0 else "line"
    default_style = "novel" if transfer == "render" else "main"
    default_wording = "transfer_wording" if transfer == "wording" else "standard"
    rows = []
    for branch, (query, answer) in enumerate(zip(queries, answers, strict=True)):
        style, wording = default_style, default_wording
        if purpose == "presentation" and branch == 1:
            if index % 2:
                style = "novel"
            else:
                wording = "b_paraphrase"
        independent = oracle.independently_answer(world, query)
        if independent != answer:
            raise AssertionError("Independent gold mismatch")
        qid = f"SRF4_{key}_{branch}"
        text = question(world, query, wording)
        rows.append(
            {
                "qid": qid,
                "root_id": f"SRF4_{key}",
                "world_id": f"SRF4_{key}",
                "base_content_sha256": fingerprint,
                "split": split,
                "pool": split,
                "family": query["family"],
                "cell": cell,
                "branch": branch,
                "purpose": purpose,
                "variant": index % 8,
                "world": world,
                "query": query,
                "question": text,
                "prompt": text + "\n" + CONTRACT_V2,
                "chart": chart,
                "render_style": style,
                "wording_template": wording,
                "transfer_stratum": transfer,
                "query_template": "range_combination"
                if transfer == "composition"
                else "sum_mean_combination",
                "gold_answer": oracle.encode(answer),
                "gold_independent": oracle.encode(independent),
                "gold_program": {
                    "engine": "sr_f4.task_contract.execute",
                    "version": oracle.VERSION,
                    "query": query,
                },
                "independent_gold_program": {
                    "engine": "sr_f4.task_contract.independently_answer",
                    "version": oracle.VERSION,
                    "query": query,
                },
                "definition_status": "UNDEFINED" if answer is None else "DEFINED",
                "target_sets": [world["categories"][i] for i in oracle.selected(world, query)],
                "target_indices": oracle.selected(world, query),
                "pair_id": None if single else f"SRF4_{key}",
                "relation_id": None if single else f"SRF4_{key}_relation",
                "changed_fields": changed
                + (
                    ["render_style"]
                    if purpose == "presentation" and index % 2
                    else ["wording_template"]
                    if purpose == "presentation"
                    else []
                ),
                "invariant_fields": ["world", "chart"]
                + (
                    ["query"] if purpose == "presentation" else ["render_style", "wording_template"]
                ),
                "expected_relation": relation,
                "generation_attempt": _attempt,
            }
        )
    return rows


def generate_new_tasks(old_tasks=()):
    used = {world_digest(row["world"]) for row in old_tasks}
    rows = []
    for split, diagnostic_count, representative_count in (("B_DEV", 24, 56), ("B_CONFIRM", 8, 8)):
        for purpose in ("selection", "operation", "presentation"):
            for index in range(diagnostic_count):
                cell = (
                    ("I-L" if index % 2 == 0 else "I-R")
                    if index < diagnostic_count // 2
                    else ("T-S" if index % 2 == 0 else "T-Q")
                )
                # Separate purposes need distinct stable IDs and seed streams.
                pair = _make_pair(
                    split,
                    cell,
                    index
                    + (0 if purpose == "selection" else 100 if purpose == "operation" else 200),
                    used,
                    purpose=purpose,
                )
                rows.extend(pair)
        for index in range(representative_count):
            rows.extend(
                _make_pair(
                    split, FAMILIES[index % 4], index, used, purpose="representative", single=True
                )
            )
    for split, count in (
        ("C_TRAIN", 64),
        ("C_DEV", 16),
        ("C_CONFIRM_ID", 20),
        ("C_CONFIRM_TRANSFER", 20),
    ):
        for cell in CELLS:
            for index in range(count):
                transfer = (
                    ("wording", "render", "scale", "composition")[index // 5]
                    if split == "C_CONFIRM_TRANSFER"
                    else None
                )
                n = 6 if transfer == "scale" else 4 if index < count // 2 else 6
                rows.extend(_make_pair(split, cell, index, used, transfer=transfer, n=n))
    validate_splits(rows, old_tasks)
    return rows


def validate_splits(rows, old_tasks=()):
    by_split = defaultdict(list)
    seen_world, seen_content, seen_qid = {}, {}, set()
    old = {world_digest(t["world"]) for t in old_tasks}
    for row in rows:
        split = row["split"]
        by_split[split].append(row)
        if row["qid"] in seen_qid:
            raise ValueError("Duplicate qid")
        seen_qid.add(row["qid"])
        for key, seen in (
            (row["world_id"], seen_world),
            (row["base_content_sha256"], seen_content),
        ):
            if seen.setdefault(key, split) != split:
                raise ValueError("Cross-split world leakage")
        if row["base_content_sha256"] in old:
            raise ValueError("New world repeats an old fact table")
    for split, (worlds, prompts) in NEW_COUNTS.items():
        actual = by_split[split]
        if (len({r["world_id"] for r in actual}), len(actual)) != (worlds, prompts):
            raise ValueError("Wrong frozen counts: " + split)
    train = by_split["C_TRAIN"]
    for cell in CELLS:
        population = [r for r in train if r["cell"] == cell and r["branch"] == 0]
        if Counter(len(r["world"]["categories"]) for r in population) != {4: 32, 6: 32}:
            raise ValueError("C_TRAIN category balance differs")
    train_wording = {r["wording_template"] for r in train}
    train_styles = {r["render_style"] for r in train}
    train_templates = {r["query_template"] for r in train}
    train_scale = {
        (
            r["cell"],
            len(r["world"]["categories"]),
            r["query"].get("k"),
            r["query"].get("end", 0) - r["query"].get("start", 0),
        )
        for r in train
    }
    for row in by_split["C_CONFIRM_TRANSFER"]:
        kind = row["transfer_stratum"]
        heldout = (
            row["wording_template"] not in train_wording
            if kind == "wording"
            else row["render_style"] not in train_styles
            if kind == "render"
            else row["query_template"] not in train_templates
            if kind == "composition"
            else (
                row["cell"],
                len(row["world"]["categories"]),
                row["query"].get("k"),
                row["query"].get("end", 0) - row["query"].get("start", 0),
            )
            not in train_scale
        )
        if not heldout:
            raise ValueError("Transfer attribute appeared in training: " + row["qid"])
    return {
        split: {"worlds": len({r["world_id"] for r in items}), "prompts": len(items)}
        for split, items in by_split.items()
    }


def inventories(rows):
    new = {cell: {} for cell in CELLS}
    old = {family: {str(v): [] for v in range(8)} for family in FAMILIES}
    for row in rows:
        if row["split"] == "C_TRAIN":
            new[row["cell"]].setdefault(row["world_id"], [None, None])[row["branch"]] = row["qid"]
        if row["split"] == "OLD_TRAIN":
            old[row["family"]][str(row["variant"])].append(row["qid"])
    return new, old


def build_c_schedule(new, old, repeat):
    if repeat not in (1, 2):
        raise ValueError("repeat must be 1 or 2")
    seed = 2026102210 + repeat
    ordered = {}
    for cell in CELLS:
        roots = sorted(new[cell])
        if len(roots) != 64 or any(
            len(set(new[cell][root])) != 2 or None in new[cell][root] for root in roots
        ):
            raise ValueError("Need 64 paired worlds per cell")
        random.Random(seed_for(seed, "root_order", cell)).shuffle(roots)
        ordered[cell] = roots
    out = []
    for step in range(32):
        items = []
        for cell in CELLS:
            for root in ordered[cell][2 * step : 2 * step + 2]:
                branch = (seed_for(2026102230, "branch", root) % 2 + repeat - 1) % 2
                items.append(
                    dict(
                        qid=new[cell][root][branch], world_id=root, cell=cell, source="C_TRAIN", G=8
                    )
                )
        for fi, family in enumerate(FAMILIES):
            variant = str((step + fi) % 8)
            choices = sorted(old[family][variant])
            if not choices:
                raise ValueError("Missing old training variant")
            qid = random.Random(seed_for(seed, "old", step, family)).choice(choices)
            items.append(dict(qid=qid, family=family, variant=variant, source="OLD_TRAIN", G=8))
        random.Random(seed_for(seed, "order", step)).shuffle(items)
        out.append(
            dict(
                step=step + 1,
                repeat=repeat,
                assignments=[dict(item, group_index=i) for i, item in enumerate(items)],
                answers=128,
            )
        )
    return out


def b0_groups(old_tasks):
    by_variant = defaultdict(lambda: defaultdict(list))
    for task in old_tasks:
        if task.get("pool") == "TRAIN_RL":
            by_variant[(task["family"], task["variant"])][task["root_id"]].append(task)
    groups = []
    for family in FAMILIES:
        for variant in range(8):
            roots = by_variant[(family, variant)]
            ordered = sorted(roots)
            random.Random(seed_for(GENERATION_SEED, "B0", family, variant)).shuffle(ordered)
            if len(ordered) < 4:
                raise ValueError("B0 requires four different training roots per variant")
            for i, root in enumerate(ordered[:4]):
                choices = sorted(roots[root], key=lambda t: t["qid"])
                task = choices[i % len(choices)]
                groups.append(
                    dict(
                        qid=task["qid"],
                        world_id=root,
                        family=family,
                        variant=variant,
                        variant_key=f"{family}_v{variant}",
                        G=8,
                        probe="base",
                    )
                )
    roots = by_variant[("INTERVAL", 7)]
    ordered = sorted(roots)
    random.Random(seed_for(GENERATION_SEED, "B0", "extra_v7")).shuffle(ordered)
    if len(ordered) < 8:
        raise ValueError("Extra v7 probe requires eight different training roots")
    for i, root in enumerate(ordered[:8]):
        choices = sorted(roots[root], key=lambda t: t["qid"])
        groups.append(
            dict(
                qid=choices[i % len(choices)]["qid"],
                world_id=root,
                family="INTERVAL",
                variant=7,
                variant_key="INTERVAL_v7",
                G=32,
                probe="extra_v7",
            )
        )
    return [dict(row, group_index=i, group_id=f"B0_{i:03d}") for i, row in enumerate(groups)]


def _copy_old(tasks):
    rows = []
    for task in tasks:
        pool = task["pool"]
        if pool != "TRAIN_RL" and not pool.startswith("TEST_"):
            continue
        row = copy.deepcopy(task)
        row.update(
            split="OLD_TRAIN" if pool == "TRAIN_RL" else "OLD_REGRESSION",
            world_id=task["root_id"],
            base_content_sha256=world_digest(task["world"]),
            cell="OLD",
            branch=0,
            pair_id=None,
            expected_relation=None,
            task_contract="legacy_native",
        )
        answer = oracle.execute(row["world"], row["query"])
        row.update(
            gold_answer=oracle.encode(answer),
            gold_program={"engine": "sr_f3_ce.contract.execute", "query": row["query"]},
            definition_status="UNDEFINED" if answer is None else "DEFINED",
        )
        rows.append(row)
    return rows


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def build_dataset(old_plan, output_root, *, old_image_root=None, render=True):
    old_plan, root = Path(old_plan), Path(output_root)
    source = old_plan / "manifests/TASKS_AUDIT_ONLY.jsonl"
    old_tasks = read_jsonl(source)
    new = generate_new_tasks(old_tasks)
    rows = _copy_old(old_tasks) + new
    if len({t["qid"] for t in rows}) != len(rows):
        raise ValueError("Duplicate combined qid")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "DATA_READY.json").exists() and json.loads(
        (root / "DATA_READY.json").read_text()
    ).get("status") == "DATA_READY":
        raise FileExistsError("Dataset already built; reuse its frozen manifests")
    for split in ("OLD_TRAIN", "OLD_REGRESSION", *NEW_COUNTS):
        selected_rows = [t for t in rows if t["split"] == split]
        write_jsonl(root / split / "TASKS.jsonl", selected_rows)
        write_jsonl(
            root / split / "MODEL_INPUTS.jsonl",
            [
                {k: t[k] for k in ("qid", "pool", "chart", "prompt", "render_style")}
                for t in selected_rows
            ],
        )
    write_jsonl(root / "TASKS.jsonl", rows)
    write_jsonl(
        root / "MODEL_INPUTS.jsonl",
        [{k: t[k] for k in ("qid", "pool", "chart", "prompt", "render_style")} for t in rows],
    )
    new_inventory, old_inventory = inventories(rows)
    for name, value in (
        ("C_NEW_INVENTORY.json", new_inventory),
        ("OLD_TRAIN_INVENTORY.json", old_inventory),
    ):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    for repeat in (1, 2):
        write_jsonl(
            root / f"C_SCHEDULE_r{repeat}.jsonl",
            build_c_schedule(new_inventory, old_inventory, repeat),
        )
    write_jsonl(root / "B0_GROUPS.jsonl", b0_groups(old_tasks))
    write_jsonl(
        root / "B2_DEV_GROUPS.jsonl",
        [
            dict(qid=t["qid"], world_id=t["world_id"], G=8, group_index=i)
            for i, t in enumerate(new)
            if t["split"] == "B_DEV"
        ],
    )
    probes = []
    for cell in CELLS:
        worlds = sorted(new_inventory[cell])
        random.Random(seed_for(GENERATION_SEED, "C_TRAIN_PROBE", cell)).shuffle(worlds)
        for world in worlds[:4]:
            probes.extend(
                dict(qid=qid, world_id=world, cell=cell, G=8) for qid in new_inventory[cell][world]
            )
    write_jsonl(
        root / "C_TRAIN_PROBE_GROUPS.jsonl", [dict(t, group_index=i) for i, t in enumerate(probes)]
    )
    image_rows = []
    if render:
        from sr_f3_ce.rendering import render_chart

        (root / "images").mkdir(exist_ok=True)
        old_image_root = Path(old_image_root) if old_image_root is not None else None
        old_images = {}
        if old_image_root is not None and (old_image_root / "IMAGE_MANIFEST.jsonl").is_file():
            old_images = {
                row["qid"]: row for row in read_jsonl(old_image_root / "IMAGE_MANIFEST.jsonl")
            }
        cached = {}
        for task in rows:
            qid = task["qid"]
            relative = f"images/{qid}.png"
            destination = root / relative
            if destination.exists():
                destination.unlink()
            if qid in old_images:
                old = old_images[qid]
                _link_or_copy(old_image_root / old["image_file"], destination)
                digest = old["image_sha256"]
            else:
                key = json.dumps(
                    [task["world"], task["chart"], task["render_style"]], sort_keys=True
                )
                if key in cached:
                    source_path, digest = cached[key]
                    _link_or_copy(source_path, destination)
                else:
                    payload, _ = render_chart(task)
                    destination.write_bytes(payload)
                    digest = hashlib.sha256(payload).hexdigest()
                    cached[key] = (destination, digest)
            image_rows.append(
                dict(
                    qid=qid,
                    image_file=relative,
                    image_sha256=digest,
                    chart=task["chart"],
                    render_style=task["render_style"],
                    width=1024,
                    height=768,
                )
            )
        write_jsonl(root / "IMAGE_MANIFEST.jsonl", image_rows)
    result = dict(
        status="DATA_READY" if render else "MANIFESTS_READY_IMAGES_NOT_RENDERED",
        source_plan=str(old_plan.resolve()),
        query_contract=oracle.VERSION,
        splits=validate_splits(new, old_tasks),
        old_counts=dict(Counter(t["split"] for t in rows if t["split"].startswith("OLD"))),
        total_prompts=len(rows),
        image_routes=len(image_rows),
        B0_answers_per_parent=1280,
        B2_DEV_answers_all_parents=4800,
        C_TRAIN_PROBE_answers_all_parents=1152,
        confirm_model_calls=0,
        transfer_composition="SUPPORTED_NATIVE_RANGE_AGGREGATE_HELD_OUT_FROM_C_TRAIN",
        generation_seed=GENERATION_SEED,
    )
    (root / "DATA_READY.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-image-root", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.old_plan,
                args.output_root,
                old_image_root=args.old_image_root,
                render=not args.no_render,
            ),
            indent=2,
        )
    )
