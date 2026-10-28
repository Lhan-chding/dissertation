"""Model-blind SR-F6 construction, independent test sampling and source images.

This module never calls a model. Its CPU receipts cannot certify processor
readability, actual labels, parent identities, or authorization to train.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from sr_f4 import task_contract as native
from srf6_ref import design
from srf6_ref.core import (
    FAMILIES,
    LABELS,
    enc,
    fingerprint,
    finite_decimal,
    heldout_tags,
    oracle_a,
    oracle_b,
    seed_for,
)

from .planning import formal_paths, root_schedule

VERSION = "srf6-data-v1-independent-composition"
PAIRED_SPLITS = frozenset(("DEV_CAL", "DEV_PILOT", "TRAIN"))
CANVAS = (1280, 896)
PLOT = (110, 125, 1225, 755)
NATURAL_SUFFIX = (
    "Give brief working and end with exactly one line: Answer: <exact number>. Do not output JSON."
)


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _write(path, value, *, rows=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preparation has one exclusive output tree. Partial builds stay visible;
    # never overwrite them with a different set of generated questions.
    with path.open("x", encoding="utf-8") as stream:
        if rows:
            for row in value:
                stream.write(canonical(row) + "\n")
        else:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _ordinary_views(world, query, rng, *, family):
    """The exact original P0 draw stream, with no P1 maximization at all."""
    pool = design.edit_pool(world, query, True, family)
    bins = {}
    for edit in pool:
        bins.setdefault(edit["descriptor"], []).append(edit)
    if len(bins) < 3:
        raise ValueError("fewer than three edit bins")
    keys = sorted(bins, key=str)
    rng.shuffle(keys)
    keys = keys[:3]
    choices = []
    for key in keys:
        values = bins[key][:]
        rng.shuffle(values)
        choices.append(values)
    edits = [rng.choice(values) for values in choices]
    all_mask = 0
    for edit in pool:
        all_mask |= edit["response_mask"]
    rules = design.wrong_rules(query)
    active = [rule for index, rule in enumerate(rules) if (all_mask >> index) & 1]
    return edits, active, rules, keys, len(pool)


def make_composition_root(family, index, master):
    """Accept only the four sampled test views, never hypothetical P1 views."""
    root_id = f"TEST_COMPOSITION_{family}_{index:04d}"
    errors = []
    for attempt in range(256):
        rng = random.Random(seed_for(master, root_id, attempt))
        count = 8 if index % 2 or family in ("SIGNED", "INTERVAL", "K_DIRECTION") else 6
        categories = list(LABELS[:count])
        rng.shuffle(categories)
        world = {
            "categories": categories,
            "series": {series: rng.sample(design.GRID, count) for series in ("Alpha", "Beta")},
            "domain": [10, 90],
        }
        query = design.base_query(family, index, count, True)
        if family == "TIES":
            values = [12, 30, 30, 30, 60, 72, 78, 84][:count]
            rng.shuffle(values)
            world["series"]["Alpha"] = values
        if family == "INTERVAL":
            middle = count // 2
            world["series"]["Alpha"][middle] = 72
            world["series"]["Beta"][middle] = 30
            world["series"]["Alpha"][middle - 1] = 24
            world["series"]["Beta"][middle - 1] = 60
        try:
            if not design.valid_task(world, query, True, family):
                raise ValueError("base validity or heldout constraints")
            edits, active, rules, descriptors, pool_size = _ordinary_views(
                world, query, rng, family=family
            )
            if not active:
                raise ValueError("no active executable candidate")
            views = [
                {"world": world, "edit_id": "anchor", "descriptor": None, "answer_delta": "0"},
                *edits,
            ]
            tasks = []
            for variant, edit in enumerate(views):
                current = edit["world"]
                gold = oracle_a(current, query)
                if gold != oracle_b(current, query):
                    raise AssertionError("dual oracle disagreement")
                if any(
                    design.wrong_value(current, query, rule) == gold["answer"] for rule in active
                ):
                    raise ValueError("candidate alias at actual test view")
                tasks.append(
                    {
                        "qid": f"{root_id}_P0_v{variant}",
                        "root_id": root_id,
                        "split": "TEST_COMPOSITION",
                        "family": family,
                        "factor": "P0",
                        "variant": variant,
                        "world": current,
                        "query": query,
                        "gold_answer": enc(gold["answer"]),
                        "gold_selected": gold["selected"],
                        "gold_bindings": dict(
                            zip(gold["selected"], map(enc, gold["operands"]), strict=True)
                        ),
                        "edit_id": edit["edit_id"],
                        "edit_descriptor": edit["descriptor"],
                        "answer_delta": edit["answer_delta"],
                        "heldout_tags": heldout_tags(current, query),
                        "engineered_collision_anchor": False,
                        "attempt": attempt,
                    }
                )
            return {
                "root_id": root_id,
                "tasks": {"P0": tasks},
                "certificate": {
                    "root_id": root_id,
                    "attempt": attempt,
                    "rejections": errors,
                    "rules": rules,
                    "active_rules": active,
                    "uncovered_by_pool": [rule for rule in rules if rule not in active],
                    "descriptors": descriptors,
                    "pool_size": pool_size,
                    "kind": "ordinary_P0_no_P1_optimization_or_rejection",
                    "engineered_collision_anchor": False,
                },
            }
        except ValueError as error:
            errors.append(str(error))
    raise RuntimeError(f"construction exhausted {root_id}: {errors[-8:]}")


def construct_banks(config):
    banks = {}
    for split, count in config["data"]["roots_per_family"].items():
        master = config["data"]["seeds"][split]
        if split == "TEST_COMPOSITION":
            roots = [
                make_composition_root(family, index, master)
                for family in FAMILIES
                for index in range(count)
            ]
        elif split == "TEST_CLASSOUT":
            roots = [
                design.make_classout_root(family, index, master)
                for family in FAMILIES
                for index in range(count)
            ]
        else:
            roots = design.build_bank(split, count, master)
        if split not in PAIRED_SPLITS:
            # DEV_EVAL and every test panel use one fixed four-view question set.
            roots = [{**root, "tasks": {"P0": root["tasks"]["P0"]}} for root in roots]
        banks[split] = roots
    return banks


def all_tasks(roots):
    return [task for root in roots for tasks in root["tasks"].values() for task in tasks]


def public_prompt(task, *, natural=False, prompt_version=None):
    # An explicit opt-in preserves every frozen v1 caller and natural panel.
    # The wire grammar and all scorers stay unchanged in interface-v2.
    from .interface_protocol import VERSION as INTERFACE_VERSION
    from .interface_protocol import public_prompt as clarified_prompt

    if prompt_version not in (None, INTERFACE_VERSION):
        raise ValueError("Unknown SR-F6 prompt revision")
    if prompt_version == INTERFACE_VERSION and not natural:
        return clarified_prompt(list(task["world"]["categories"]), dict(task["query"]))
    safe = {
        "world": {"categories": list(task["world"]["categories"])},
        "query": dict(task["query"]),
    }
    prompt = design.visible_prompt(safe)
    if natural:
        prompt = prompt.split("Return exactly one JSON object", 1)[0] + NATURAL_SUFFIX
    return prompt


def natural_interface_tasks(id_roots, master):
    tasks = sorted(all_tasks(id_roots), key=lambda task: task["qid"])
    selected = []
    for family in FAMILIES:
        selected.extend([task for task in tasks if task["family"] == family][:16])
    chosen = {task["qid"] for task in selected}
    remaining = [task for task in tasks if task["qid"] not in chosen]
    remaining.sort(
        key=lambda task: (seed_for(master, "natural_interface", task["qid"]), task["qid"])
    )
    selected.extend(remaining[:32])
    if len(selected) != 128:
        raise ValueError("Natural interface requires exactly 128 paired ID questions")
    return [
        dict(
            task,
            qid=f"NATURAL_{task['qid']}",
            paired_id_qid=task["qid"],
            split="NATURAL_INTERFACE",
            protocol="natural_answer",
        )
        for task in selected
    ]


def audit_banks(banks, config):
    counts = config["data"]["roots_per_family"]
    if set(banks) != set(counts):
        raise ValueError("Missing or unregistered split")
    seen_qids, world_splits = set(), {}
    splits = {}
    comparisons = 0
    for split, roots in banks.items():
        expected_roots = {
            f"{split}_{family}_{index:04d}" for family in FAMILIES for index in range(counts[split])
        }
        if (
            len(roots) != len(expected_roots)
            or {root["root_id"] for root in roots} != expected_roots
        ):
            raise ValueError(f"Incomplete or duplicate roots: {split}")
        expected_factors = {"P0", "P1"} if split in PAIRED_SPLITS else {"P0"}
        for root in roots:
            if set(root["tasks"]) != expected_factors:
                raise ValueError(f"Wrong factor matrix: {root['root_id']}")
            anchor = root["tasks"]["P0"][0]
            anchor_world, query = anchor["world"], anchor["query"]
            if split != "TEST_CLASSOUT":
                pool = design.edit_pool(
                    anchor_world, query, split == "TEST_COMPOSITION", anchor["family"]
                )
                edits = {edit["edit_id"]: edit for edit in pool}
                all_mask = 0
                for edit in pool:
                    all_mask |= edit["response_mask"]
                rules = design.wrong_rules(query)
                active = [rule for index, rule in enumerate(rules) if (all_mask >> index) & 1]
                if root["certificate"]["active_rules"] != active:
                    raise ValueError("Stored active-candidate inventory differs")
                for factor, tasks in root["tasks"].items():
                    mask = 0
                    for task in tasks[1:]:
                        edit = edits.get(task["edit_id"])
                        if (
                            edit is None
                            or edit["world"] != task["world"]
                            or list(edit["descriptor"]) != list(task["edit_descriptor"])
                        ):
                            raise ValueError("Stored edit-cost descriptor or edited world differs")
                        mask |= edit["response_mask"]
                    if (
                        split in PAIRED_SPLITS
                        and abs(
                            root["certificate"][f"coverage_{factor}"]
                            - mask.bit_count() / len(active)
                        )
                        > 1e-12
                    ):
                        raise ValueError("Stored coverage differs from actual edited worlds")
            if split in PAIRED_SPLITS:
                left, right = root["tasks"]["P0"], root["tasks"]["P1"]
                if left[0]["world"] != right[0]["world"]:
                    raise ValueError("Paired anchors differ")
                if [t["edit_descriptor"] for t in left] != [t["edit_descriptor"] for t in right]:
                    raise ValueError("Paired edit-cost bins differ")
            for factor, tasks in root["tasks"].items():
                if len(tasks) != 4 or [task["variant"] for task in tasks] != list(range(4)):
                    raise ValueError("Each root/factor requires ordered four views")
                if len({digest(task["world"]) for task in tasks}) != 4:
                    raise ValueError("Each root/factor requires four distinct actual worlds")
                for task in tasks:
                    qid = task["qid"]
                    if (
                        qid in seen_qids
                        or task["split"] != split
                        or task["root_id"] != root["root_id"]
                    ):
                        raise ValueError("Duplicate question or inconsistent lineage")
                    if (
                        task["factor"] != factor
                        or qid != f"{root['root_id']}_{factor}_v{task['variant']}"
                    ):
                        raise ValueError("Question factor identity differs")
                    seen_qids.add(qid)
                    world, query = task["world"], task["query"]
                    if (
                        task["family"] != anchor["family"]
                        or query != anchor["query"]
                        or not root["root_id"].startswith(f"{split}_{task['family']}_")
                    ):
                        raise ValueError("Root query or family changed across views")
                    if world["categories"] != anchor_world["categories"]:
                        raise ValueError("Edits may not change printed category order")
                    if any(
                        value not in design.GRID
                        for column in world["series"].values()
                        for value in column
                    ):
                        raise ValueError("Chart value outside registered grid")
                    if split == "TEST_CLASSOUT" and task["variant"]:
                        deltas = [
                            (series, index, value - anchor_world["series"][series][index])
                            for series in ("Alpha", "Beta")
                            for index, value in enumerate(world["series"][series])
                            if value != anchor_world["series"][series][index]
                        ]
                        if (
                            len(deltas) != 2
                            or len({index for _, index, _ in deltas}) != 2
                            or any(abs(delta) != 12 for _, _, delta in deltas)
                        ):
                            raise ValueError(
                                "CLASSOUT requires two distinct-category twelve-unit edits"
                            )
                        if task["edit_descriptor"] != [list(delta) for delta in deltas]:
                            raise ValueError("CLASSOUT descriptor differs from actual world edits")
                    gold = oracle_a(world, query)
                    if task["answer_delta"] != enc(
                        gold["answer"] - oracle_a(anchor_world, anchor["query"])["answer"]
                    ):
                        raise ValueError("Stored answer delta differs")
                    if gold != oracle_b(world, query):
                        raise ValueError("Independent oracle disagreement: " + qid)
                    if (
                        native.execute(world, query) != gold["answer"]
                        or (native.independently_answer(world, query) != gold["answer"])
                        or set(native.selected(world, query)) != set(gold["indices"])
                    ):
                        raise ValueError("Native F5 oracle disagreement: " + qid)
                    if (
                        task["gold_answer"] != enc(gold["answer"])
                        or (task["gold_selected"] != gold["selected"])
                        or task["gold_bindings"]
                        != dict(zip(gold["selected"], map(enc, gold["operands"]), strict=True))
                    ):
                        raise ValueError("Stored gold mismatch: " + qid)
                    if not finite_decimal(gold["answer"]) or not 0 < len(gold["indices"]) < len(
                        world["categories"]
                    ):
                        raise ValueError("Illegal task result: " + qid)
                    tags = heldout_tags(world, query)
                    if split == "TEST_COMPOSITION":
                        if design.TARGET[task["family"]] not in tags:
                            raise ValueError("Missing assigned composition holdout: " + qid)
                        for rule in root["certificate"]["active_rules"]:
                            if design.wrong_value(world, query, rule) == gold["answer"]:
                                raise ValueError("Active-library answer alias: " + qid)
                    elif tags:
                        raise ValueError("Global holdout leaked outside COMPOSITION: " + qid)
                    if task["heldout_tags"] != tags:
                        raise ValueError("Stored heldout tags mismatch: " + qid)
                    shape = fingerprint(world)
                    previous = world_splits.setdefault(shape, split)
                    if previous != split:
                        raise ValueError(
                            f"Cross-split renamed/reordered world: {previous}/{split}/{qid}"
                        )
                    comparisons += 1
        tasks = all_tasks(roots)
        splits[split] = {
            "roots": len(roots),
            "questions": len(tasks),
            "factors": sorted(expected_factors),
            "family_questions": dict(Counter(task["family"] for task in tasks)),
        }
    train = banks["TRAIN"]
    gaps = [
        root["certificate"]["coverage_P1"] - root["certificate"]["coverage_P0"] for root in train
    ]
    coverage = {
        "mean_gap": sum(gaps) / len(gaps),
        "strict_gain_fraction": sum(gap > 1e-12 for gap in gaps) / len(gaps),
        "roots": len(gaps),
        "model_outputs_used": False,
    }
    coverage["status"] = (
        "PASS"
        if (
            coverage["mean_gap"] >= config["feasibility"]["coverage_mean_gap_min"]
            and coverage["strict_gain_fraction"]
            >= config["feasibility"]["coverage_fraction_roots_strict_gain_min"]
        )
        else "INSUFFICIENT_REGISTERED_EXPOSURE"
    )
    forced = [root for root in train if root["certificate"]["engineered_collision_anchor"]]
    if len(forced) != len(train) // 3:
        raise ValueError("Training engineered-alias fraction is not one third")
    return {
        "status": "PASS",
        "scope": "CPU mathematical and identity checks only",
        "splits": splits,
        "question_oracle_comparisons": comparisons,
        "independent_native_oracles": 2,
        "cross_split_world_duplicates": 0,
        "training_manipulation": coverage,
        "engineered_train_anchors": len(forced),
        "processor_readability": "NOT_RUN",
        "model_calls": 0,
    }


def composition_correction_report(roots, master):
    """Compare preserved v1 arithmetic inputs, without accessing model outcomes."""
    changed = []
    for root in roots:
        first = root["tasks"]["P0"][0]
        index = int(root["root_id"].rsplit("_", 1)[1])
        original = design.make_root(
            "TEST_COMPOSITION", first["family"], index, master, ood=True, zero_alias=True
        )
        old_tasks = original["tasks"]["P0"]
        new_tasks = root["tasks"]["P0"]
        if digest(old_tasks) != digest(new_tasks):
            changed.append(
                {
                    "root_id": root["root_id"],
                    "original_attempt": original["certificate"]["attempt"],
                    "corrected_attempt": root["certificate"]["attempt"],
                    "original_tasks_sha256": digest(old_tasks),
                    "corrected_tasks_sha256": digest(new_tasks),
                }
            )
    return {
        "kind": "UNUSED_P1_TEST_ACCEPTANCE_REMOVED",
        "changed_roots": changed,
        "changed_root_count": len(changed),
        "roots_compared": len(roots),
        "seed_changed": False,
        "model_outputs_used": False,
    }


def render_source(world, *, font_path=None):
    """F6 geometry; no global mutation of the historical 1024x768 renderer."""
    import PIL
    from PIL import Image, ImageDraw, ImageFont, features

    from sr_f3_ce.rendering import _font_path

    native.native.validate_world(world)
    path = _font_path(font_path)
    fonts = {size: ImageFont.truetype(str(path), size) for size in (21, 22, 23, 28)}
    image = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(image)
    labels = []

    def text(xy, value, size=22, anchor="la"):
        bounds = draw.textbbox(xy, value, font=fonts[size], anchor=anchor)
        if min(bounds[:2]) < 0 or bounds[2] > CANVAS[0] or bounds[3] > CANVAS[1]:
            raise ValueError("Source label clipping")
        labels.append({"text": value, "bounding_box": list(bounds), "font_size": size})
        draw.text(xy, value, font=fonts[size], anchor=anchor, fill="#202833")

    left, top, right, bottom = PLOT
    units = (bottom - top) / 100
    if units * 6 < 30:
        raise ValueError("Six chart units require at least thirty source pixels")
    text((left, 35), "Product measurements", 28)
    colors = {"Alpha": (63, 110, 156), "Beta": (184, 112, 48)}
    for index, (series, color) in enumerate(colors.items()):
        x = 840 + index * 170
        draw.rectangle((x, 55, x + 24, 79), fill=color)
        text((x + 35, 50), series, 23)
    for value in range(0, 101, 10):
        y = bottom - units * value
        draw.line((left, y, right, y), fill=(215, 215, 215), width=1)
        text((left - 15, y), str(value), 21, "rm")
    spacing = (right - left) / len(world["categories"])
    bar_width = 37
    bars = []
    for index, category in enumerate(world["categories"]):
        center = left + spacing * (index + 0.5)
        for column, series in enumerate(colors):
            x = center + (column - 0.5) * bar_width * 1.12
            value = world["series"][series][index]
            box = (
                round(x - bar_width / 2),
                round(bottom - units * value),
                round(x + bar_width / 2),
                bottom,
            )
            draw.rectangle(box, fill=colors[series])
            bars.append(
                {"category": category, "series": series, "value": value, "bounding_box": list(box)}
            )
        text((center, bottom + 25), category, 22, "mt")
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    text((left, 825), "Counts | Read categories in printed left-to-right order", 22)
    target = io.BytesIO()
    image.save(target, format="PNG", optimize=False, compress_level=6)
    payload = target.getvalue()
    return payload, {
        "width": CANVAS[0],
        "height": CANVAS[1],
        "pixels_per_value_unit": units,
        "pixels_per_six_units": units * 6,
        "bar_width_px": bar_width,
        "category_center_spacing_px": spacing,
        "labels": labels,
        "bars": bars,
        "numeric_value_labels": False,
        "font_path": str(path),
        "font_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pillow_version": PIL.__version__,
        "freetype_version": features.version_module("freetype2"),
        "reviewer_type": "PROGRAM",
        "processor_readability": "NOT_RUN",
    }


def build(root, config, *, font_path=None, render=True):
    """Create a new DATA tree; source images are not a processor release gate."""
    root = Path(root)
    destination = root / "DATA"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Preserve existing DATA; choose a new preparation root")
    banks = construct_banks(config)
    audit = audit_banks(banks, config)
    if audit["training_manipulation"]["status"] != "PASS":
        raise ValueError("INSUFFICIENT_REGISTERED_EXPOSURE: training coverage manipulation")
    correction = composition_correction_report(
        banks["TEST_COMPOSITION"], config["data"]["seeds"]["TEST_COMPOSITION"]
    )
    destination.mkdir(parents=True)
    _write(destination / "CPU_AUDIT.json", audit)
    _write(destination / "COMPOSITION_CORRECTION.json", correction)
    task_sets = {split: all_tasks(roots) for split, roots in banks.items()}
    task_sets["NATURAL_INTERFACE"] = natural_interface_tasks(
        banks["TEST_ID"], config["master_seed"]
    )
    images, geometry, public_rows = {}, [], []
    for split, tasks in task_sets.items():
        inputs = []
        for task in tasks:
            world_hash = digest(task["world"])
            if render and world_hash not in images:
                payload, metrics = render_source(task["world"], font_path=font_path)
                image_file = f"DATA/images/{world_hash}.png"
                path = root / image_file
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(payload)
                images[world_hash] = {
                    "image_file": image_file,
                    "image_sha256": hashlib.sha256(payload).hexdigest(),
                }
                geometry.append({"world_sha256": world_hash, **images[world_hash], **metrics})
            identity = {
                key: task[key] for key in ("qid", "root_id", "split", "family", "factor", "variant")
            }
            prompt = public_prompt(task, natural=split == "NATURAL_INTERFACE")
            model_input = {"prompt": prompt, **images.get(world_hash, {})}
            inputs.append(
                {
                    **identity,
                    "protocol": "natural_answer"
                    if split == "NATURAL_INTERFACE"
                    else "structured_v4",
                    "model_input": model_input,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "source_image_status": "GENERATED" if render else "NOT_RUN",
                }
            )
        _write(destination / split / "TASKS_PRIVATE.jsonl", tasks, rows=True)
        _write(destination / split / "MODEL_INPUTS.jsonl", inputs, rows=True)
        public_rows.extend(inputs)
    _write(
        destination / "ROOT_CERTIFICATES_PRIVATE.jsonl",
        [root for roots in banks.values() for root in roots],
        rows=True,
    )
    _write(destination / "MODEL_INPUTS.jsonl", public_rows, rows=True)
    _write(destination / "RENDER_GEOMETRY.jsonl", geometry, rows=True)
    for parent in config["model"]["parents"]:
        for repeat in config["training"]["repeats"]:
            _write(
                destination / "SCHEDULES" / f"TRAIN-{parent}-r{repeat}.json",
                root_schedule(48, config["master_seed"], parent, repeat),
            )
        _write(
            destination / "SCHEDULES" / f"DEV_PILOT-{parent}-r0.json",
            root_schedule(8, config["master_seed"], parent, 0, split="DEV_PILOT"),
        )
    _write(destination / "FORMAL_PATH_ORDER.json", formal_paths(config))
    review = []
    for family in FAMILIES:
        candidates = [
            task
            for task in task_sets["DEV_CAL"]
            if task["family"] == family and task["factor"] == "P1" and task["variant"] == 0
        ]
        for task in candidates:
            review.append(
                {
                    "qid": task["qid"],
                    "family": family,
                    **images.get(digest(task["world"]), {}),
                    "processor_image": None,
                    "visual_review": "NOT_RUN",
                }
            )
    _write(
        destination / "PROCESSOR_REVIEW_PLAN.json",
        {
            "items": review,
            "count": len(review),
            "selection": "All eight DEV_CAL anchors per family; deterministic and model-blind",
            "source_visual_review": "NOT_RUN",
            "processor_visual_review": "NOT_RUN",
        },
    )
    manifest = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    _write(destination / "MANIFEST.json", {"config_sha256": digest(config), "files": manifest})
    receipt = {
        "status": "CPU_PREPARED_NOT_RELEASED",
        "data_version": VERSION,
        "question_rows": len(public_rows),
        "unique_source_images": len(images),
        "config_sha256": digest(config),
        "manifest_sha256": hashlib.sha256((destination / "MANIFEST.json").read_bytes()).hexdigest(),
        "cpu_audit": audit,
        "composition_correction": correction,
        "source_images": "GENERATED" if render else "NOT_RUN",
        "source_visual_review": "NOT_RUN",
        "processor_readability": "NOT_RUN",
        "historical_world_exclusion": "NOT_RUN",
        "external_retention_ids": "NOT_RUN",
        "formal_model_access": "LOCKED",
        "model_calls": 0,
    }
    _write(destination / "BUILD_RECEIPT.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument(
        "--no-render", action="store_true", help="CPU task audit only; cannot release data"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.root,
                json.loads(args.config.read_text()),
                font_path=args.font_path,
                render=not args.no_render,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
