"""Build the frozen SR-F5 banks with the inherited production renderer.

Construction consumes only registered seeds and mathematical constraints. Model
outputs are never inputs to this module. Images are rendered once per distinct
world/style, and their new bytes are hashed once for the native input boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from srf5_ref.generator import (
    FAMILIES,
    TARGET,
    audit_splits,
    make_bank,
    make_interface_bank,
    terminate_decimal,
    world_shape,
)
from srf5_ref.oracle import boundary_tie, encode, evaluate_a, evaluate_b, heldout_tags
from srf5_ref.prompts import task_prompt
from srf5_ref.schedule import build_schedule

RENDER_STYLES = {"r0": "main", "r1": "novel"}
VERSION = "srf5-native-data-v1"
SPLIT_COUNTS = {
    "dev": "dev_roots_per_family",
    "pilot": "pilot_roots_per_family",
    "train": "train_roots_per_family",
    "test_id": "test_id_roots_per_family",
    "test_composition": "test_composition_roots_per_family",
    "test_relation": "test_relation_roots_per_family",
    "test_interface": "test_interface_roots_per_family",
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write(path, value, *, rows=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        if rows:
            for row in value:
                stream.write(_canonical(row) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def split_specs(config):
    counts = {split: config["data"][key] for split, key in SPLIT_COUNTS.items()}
    counts["dev_monitor"] = config["evaluation"]["dev_monitor_roots_per_family"]
    if config["data"]["variants_per_root"] != 4:
        raise ValueError("The registered design requires four variants per root")
    if any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("Root counts must be positive integers")
    for split in counts:
        if type(config["data"]["seeds"][split]) is not int:
            raise ValueError("Each split needs its registered integer seed")
    return counts


def construct_banks(config):
    banks = {}
    for split, count in split_specs(config).items():
        seed = config["data"]["seeds"][split]
        banks[split] = (
            make_interface_bank(count, seed)
            if split == "test_interface"
            else make_bank(split, count, seed, ood=split == "test_composition")
        )
    return banks


def public_prompt(task):
    """Only printed labels and the public query can influence model text."""
    public = {
        "world": {"categories": list(task["world"]["categories"])},
        "query": dict(task["query"]),
        "protocol": task.get("protocol", "structured_v3"),
    }
    return task_prompt(public)


def _factor_counts(tasks):
    atoms, pairs, conjunctions = Counter(), Counter(), Counter()
    for task in tasks:
        query = task["query"]
        factors = [f"{key}={value}" for key, value in sorted(query.items())]
        factors += [
            f"boundary_tie={boundary_tie(task['world'], query)}",
            f"category_count={len(task['world']['categories'])}",
        ]
        atoms.update(factors)
        pairs.update(" & ".join(pair) for pair in combinations(sorted(factors), 2))
        conjunctions.update(heldout_tags(task["world"], query))
    return {
        "question_denominator": len(tasks),
        "atoms": dict(sorted(atoms.items())),
        "pairs": dict(sorted(pairs.items())),
        "registered_heldout_conjunctions": dict(sorted(conjunctions.items())),
    }


def audit_banks(banks, config):
    """Validate every question before any image rendering or model execution."""
    specs = split_specs(config)
    if set(banks) != set(specs):
        raise ValueError("Missing or unregistered split")
    split_report = audit_splits(banks)
    seen_qids = set()
    checked = 0
    reports = {}
    for split, tasks in banks.items():
        roots = defaultdict(list)
        family_counts = Counter()
        for task in tasks:
            if task["qid"] in seen_qids or task["split"] != split:
                raise ValueError("Duplicate question identity or split mismatch")
            seen_qids.add(task["qid"])
            roots[task["root_id"]].append(task)
            family_counts[task["cell"]] += 1
            oracle_a = evaluate_a(task["world"], task["query"])
            oracle_b = evaluate_b(task["world"], task["query"])
            if oracle_a != oracle_b:
                raise ValueError("Independent oracle disagreement: " + task["qid"])
            if (
                task["gold_answer"] != encode(oracle_a["answer"])
                or task["gold_selected"] != oracle_a["selected"]
            ):
                raise ValueError("Stored oracle label differs: " + task["qid"])
            if not (split == "test_interface" and task["variant"] == 3) and not terminate_decimal(
                oracle_a["answer"]
            ):
                raise ValueError("Nonterminating primary answer: " + task["qid"])
            tags = heldout_tags(task["world"], task["query"])
            if split == "test_composition":
                if TARGET[task["cell"]] not in tags:
                    raise ValueError("Missing assigned holdout conjunction: " + task["qid"])
            elif split != "test_interface" and tags:
                raise ValueError("Heldout conjunction leaked into support: " + task["qid"])
            if task["render_style"] not in RENDER_STYLES:
                raise ValueError("Unregistered render style")
            checked += 1
        if set(family_counts) != set(FAMILIES) or any(
            count != 4 * specs[split] for count in family_counts.values()
        ):
            raise ValueError("Incomplete family matrix: " + split)
        for root_id, descendants in roots.items():
            if len(descendants) != 4 or {t["variant"] for t in descendants} != set(range(4)):
                raise ValueError("Incomplete root lineage: " + root_id)
            if len({t["cell"] for t in descendants}) != 1:
                raise ValueError("Root assigned to multiple families")
            base = next(t for t in descendants if t["variant"] == 0)
            base_answer = evaluate_a(base["world"], base["query"])["answer"]
            for task in descendants:
                answer = evaluate_a(task["world"], task["query"])["answer"]
                relation = task["relation_to_v0"]
                if relation["answer_delta"] != encode(answer - base_answer) or (
                    relation["selected_same_ids"]
                    != (set(task["gold_selected"]) == set(base["gold_selected"]))
                ):
                    raise ValueError("Invalid stored root relation: " + task["qid"])
        reports[split] = {
            "roots": len(roots),
            "queries": len(tasks),
            "family_questions": dict(sorted(family_counts.items())),
            "max_construction_attempt": max(t["construction_attempt"] for t in tasks),
        }
    split_report.update(status="PASS", splits=reports, checked_lineages=True)
    return {
        "split_audit": split_report,
        "oracle_check": {
            "status": "PASS",
            "questions_compared": checked,
            "oracle_a": "srf5_ref.oracle.evaluate_a: sorted ranks",
            "oracle_b": "srf5_ref.oracle.evaluate_b: enumerated admissible subsets",
            "main_answers_finite_decimal": True,
            "interface_v3_numeric_stress_exempt": True,
            "stored_gold_and_root_relations_verified": True,
        },
        "factor_coverage": {split: _factor_counts(tasks) for split, tasks in banks.items()},
    }


def audit_historical_worlds(root, historical_paths):
    """Exclude all new worlds from required historical task/example sources.

    JSONL task manifests and single-object JSON fixtures are supported. Only
    explicitly identified ChartQA rows may omit a synthetic world; these are
    counted separately rather than silently discarded. Missing sources fail.
    """
    root = Path(root)
    data_root = root / "DATA"
    receipt_path = data_root / "BUILD_RECEIPT.json"
    build_receipt = json.loads(receipt_path.read_text())
    report = {
        "status": "FAIL",
        "sources": [],
        "new_questions_checked": 0,
        "historical_world_rows": 0,
        "external_chartqa_rows_without_world": 0,
        "comparison_keys": ["exact_ordered_world", "printed_world", "renamed_reordered_world"],
        "collisions": [],
    }
    try:
        if not historical_paths:
            raise ValueError("At least one historical world source is required")
        tables = {key: {} for key in report["comparison_keys"]}

        def keys(world):
            # Historical worlds may have an additional series. Alpha/Beta
            # projection is deliberately conservative for overlap detection.
            if not isinstance(world, dict) or "categories" not in world:
                raise ValueError("Historical synthetic world is missing its category inventory")
            for series in ("Alpha", "Beta"):
                if len(world["series"][series]) != len(world["categories"]):
                    raise ValueError("Historical world has inconsistent column lengths")
            shape = world_shape(world)
            return _canonical(world), shape, tuple(sorted(shape))

        for source in historical_paths:
            source = Path(source)
            raw = source.read_text(encoding="utf-8")
            rows = (
                [json.loads(raw)]
                if source.suffix == ".json"
                else [json.loads(line) for line in raw.splitlines() if line.strip()]
            )
            if not rows:
                raise ValueError("Historical source is empty: " + str(source))
            source_report = {"path": str(source.resolve()), "rows": len(rows), "world_rows": 0}
            report["sources"].append(source_report)
            for index, task in enumerate(rows, 1):
                if "world" not in task:
                    if task.get("pool") == "CHARTQA":
                        report["external_chartqa_rows_without_world"] += 1
                        continue
                    raise ValueError(f"Unclassified row without world: {source}:{index}")
                identity = {"source": str(source), "row": index, "qid": task.get("qid")}
                for kind, key in zip(report["comparison_keys"], keys(task["world"]), strict=True):
                    tables[kind].setdefault(key, identity)
                source_report["world_rows"] += 1
                report["historical_world_rows"] += 1
            if not source_report["world_rows"]:
                raise ValueError("Required source contains no synthetic worlds: " + str(source))
        new_path = data_root / "TASKS_PRIVATE.jsonl"
        with new_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                task = json.loads(line)
                report["new_questions_checked"] += 1
                for kind, key in zip(report["comparison_keys"], keys(task["world"]), strict=True):
                    if key in tables[kind]:
                        report["collisions"].append({
                            "kind": kind,
                            "qid": task["qid"],
                            "split": task["split"],
                            "root_id": task["root_id"],
                            "historical": tables[kind][key],
                        })
        if not report["new_questions_checked"]:
            raise ValueError("No new worlds available for historical exclusion")
        if report["collisions"]:
            raise ValueError(
                "New world overlaps a historical source; preserve and revise construction"
            )
        report["status"] = "PASS"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report["error"] = str(exc)
        raise
    finally:
        _write(data_root / "HISTORICAL_WORLD_AUDIT.json", report)
        build_receipt["historical_world_exclusion"] = report["status"]
        build_receipt["historical_world_audit"] = "DATA/HISTORICAL_WORLD_AUDIT.json"
        _write(receipt_path, build_receipt)
    return report


def build(root, config, *, font_path=None):
    """Create all registered data once; existing DATA requires explicit recovery.

    Model-input rows hold identity outside the three-field ``model_input`` object
    expected by the native runtime. Image paths are relative to the run root.
    """
    from sr_f3_ce.rendering import PLOT, render_chart

    root = Path(root)
    destination = root / "DATA"
    if destination.exists():
        raise FileExistsError("Preserve existing DATA; do not silently rebuild a frozen bank")
    banks = construct_banks(config)
    audits = audit_banks(banks, config)
    destination.mkdir(parents=True)
    _write(destination / "SPLIT_AUDIT.json", audits["split_audit"])
    _write(destination / "ORACLE_CHECK.json", audits["oracle_check"])
    _write(destination / "FACTOR_COVERAGE.json", audits["factor_coverage"])
    rendered, geometry = {}, []
    all_inputs, all_images, all_relations, all_tasks = [], [], [], []
    for split, tasks in banks.items():
        inputs, images, relations = [], [], []
        for task in tasks:
            key = _canonical([task["world"], task["render_style"]])
            if key not in rendered:
                # The old renderer only sees world/chart/style, never task labels.
                render_task = {
                    "world": task["world"],
                    "chart": "grouped_bar",
                    "render_style": RENDER_STYLES[task["render_style"]],
                }
                payload, metrics = render_chart(render_task, font_path=font_path)
                world_id = f"world_{len(rendered):05d}"
                image_file = f"DATA/images/{world_id}.png"
                path = root / image_file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                rendered[key] = {
                    "world_id": world_id,
                    "image_file": image_file,
                    "image_sha256": hashlib.sha256(payload).hexdigest(),
                }
                spacing = (PLOT[2] - PLOT[0]) / len(task["world"]["categories"])
                width = min(37, spacing * 0.74 / 2)
                geometry.append({
                    **rendered[key],
                    **metrics,
                    "registered_style": task["render_style"],
                    "category_center_spacing_px": spacing,
                    "bar_nominal_width_px": width,
                    "between_category_group_gap_px": spacing - 2 * width,
                    "reviewer_type": "program",
                    "processor_readability": "NOT_RUN",
                    "claim": "source_geometry_only; actual processor requires separate validation",
                })
            image = rendered[key]
            model_input = {
                "prompt": public_prompt(task),
                "image_file": image["image_file"],
                "image_sha256": image["image_sha256"],
            }
            identity = {key: task[key] for key in ("qid", "root_id", "split", "cell")}
            inputs.append({
                **identity,
                "protocol": task.get("protocol", "structured_v3"),
                "model_input": model_input,
            })
            images.append({**identity, **image, "render_style": task["render_style"]})
            relations.append({
                **identity,
                "variant": task["variant"],
                "anchor_qid": f"{task['root_id']}_v0",
                **task["relation_to_v0"],
            })
        for name, values in (
            ("TASKS_PRIVATE", tasks),
            ("MODEL_INPUTS", inputs),
            ("IMAGES", images),
            ("ROOT_RELATIONS", relations),
        ):
            _write(destination / split / f"{name}.jsonl", values, rows=True)
        all_tasks.extend(tasks)
        all_inputs.extend(inputs)
        all_images.extend(images)
        all_relations.extend(relations)
    for name, values in (
        ("TASKS_PRIVATE", all_tasks),
        ("MODEL_INPUTS", all_inputs),
        ("IMAGES", all_images),
        ("ROOT_RELATIONS", all_relations),
        ("RENDER_GEOMETRY", geometry),
    ):
        _write(destination / f"{name}.jsonl", values, rows=True)
    for parent in config["parents"]:
        for repeat in config["repeats"]:
            schedule = build_schedule(
                banks["train"], config["master_seed"], parent, repeat,
                expected_roots_per_family=config["data"]["train_roots_per_family"],
            )
            _write(destination / "schedules" / f"train_{parent}_r{repeat}.json", schedule)
        pilot = build_schedule(
            banks["pilot"], config["master_seed"], parent, 0,
            expected_roots_per_family=config["data"]["pilot_roots_per_family"],
        )
        _write(destination / "schedules" / f"pilot_{parent}_r0.json", pilot)
    report = {
        "status": "PASS",
        "scope": "CPU data construction, oracle, split and source rendering only",
        "version": VERSION,
        "questions": len(all_tasks),
        "unique_rendered_images": len(rendered),
        "renderer": "sr_f3_ce.rendering.render_chart",
        "style_mapping": RENDER_STYLES,
        "chart": "grouped_bar",
        "processor_readability": "NOT_RUN",
        "historical_world_exclusion": "NOT_RUN; compare historical F4-C worlds before G1 release",
        "image_hash_policy": "once per unique newly rendered image; no weight hashing",
        "split_audit": audits["split_audit"],
    }
    _write(destination / "BUILD_RECEIPT.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    print(json.dumps(build(args.root, config, font_path=args.font_path), indent=2))


if __name__ == "__main__":
    main()
