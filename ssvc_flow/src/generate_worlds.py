"""Seeded N-track datasets, independently solved and globally deduplicated."""

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from .constraint_solver import solve
from .prompts import TEMPLATE_VERSION, build_prompt
from .render_charts import render_chart
from .verifiers import executor, fiber_size

DEFAULT_SIZES = {
    "calibration": 144,
    "train": 576,
    "control": 144,
    "dev": 144,
    "confirm": 288,
    "ood": 288,
    "natural_pool": 2048,
}
FAMILIES = ("duplicate_encoding", "cross_series", "trend")
CHARTS = ("grouped_bar", "line")
OPERATIONS = ("sum4", "difference_pairs", "range4")
INTERFACES = ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")
SCHEMA_VERSION = "ssvc-scene-v1"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _validate_sizes(sizes):
    if not sizes or set(sizes) - set(DEFAULT_SIZES):
        raise ValueError("sizes must name known splits")
    for split, size in sizes.items():
        if type(size) is not int or size < 0:
            raise ValueError("split sizes must be nonnegative integers")
        divisor = 36 if split == "train" else 3 if split == "ood" else 18
        if split != "natural_pool" and size % divisor:
            raise ValueError(f"{split} size must be divisible by {divisor}")
        if split == "natural_pool" and size > 2048:
            raise ValueError("natural_pool cannot exceed the predeclared 2048-scene cap")
    trend_count = sum(
        size // 3 for split, size in sizes.items() if split not in {"ood", "natural_pool"}
    )
    if (
        trend_count + min(sizes.get("natural_pool", 0) // 3, 288) + sizes.get("ood", 0) // 9 + 4
        > 784
    ):
        raise ValueError("requested unique ID trend worlds exceed the finite domain capacity")


def _cell_specs(split, size, families=FAMILIES, subtype=None):
    cells = tuple(itertools.product(families, CHARTS, OPERATIONS))
    return [
        {
            "split": split,
            "constraint_family": family,
            "chart_type": chart,
            "operation": operation,
            "ood_subtype": subtype,
            "interface": INTERFACES[(index // len(cells)) % 2] if split == "train" else None,
            "graph_topology": ("path", "cycle")[(index // len(cells)) % 2]
            if subtype == "graph_structure"
            else "star"
            if family == "cross_series"
            else None,
        }
        for index in range(size)
        for family, chart, operation in (cells[index % len(cells)],)
    ]


def _specifications(sizes):
    specs = []
    for split, size in sizes.items():
        if split == "ood":
            addition = [
                spec
                for subtype in ("graph_structure", "numeric_shift", "render_expression")
                for spec in _cell_specs(
                    split,
                    size // 3,
                    ("cross_series",) if subtype == "graph_structure" else FAMILIES,
                    subtype,
                )
            ]
        elif split == "natural_pool":
            # Only 784 nonconstant progressions exist in 0..49. Balance is not mandated
            # for this pool; reserve 288 trends and disclose the actual composition.
            trends = min(size // 3, 288)
            duplicate = (size - trends) // 2
            allocation = (duplicate, size - trends - duplicate, trends)
            addition = [
                spec
                for family, count in zip(FAMILIES, allocation, strict=True)
                for spec in _cell_specs(split, count, (family,))
            ]
        else:
            addition = _cell_specs(split, size)
        specs = [*specs, *addition]
    return specs


def _trend_worlds(lower, rng):
    worlds = tuple(
        (a, a + d, a + 2 * d, a + 3 * d)
        for a in range(lower, lower + 50)
        for d in range(-16, 17)
        if d and lower <= a + 3 * d < lower + 50
    )
    return iter(rng.sample(worlds, len(worlds)))


def _choose_world(spec, rng, trend_pools, seen):
    lower = 50 if spec["ood_subtype"] == "numeric_shift" else 0
    for _ in range(100000):
        if spec["constraint_family"] == "trend":
            try:
                world = list(next(trend_pools[lower]))
            except StopIteration as error:
                raise ValueError("unique trend world capacity exhausted") from error
        else:
            world = [rng.randrange(lower, lower + 50) for _ in range(4)]
        if _digest(world) not in seen:
            return world
    raise ValueError("could not sample a globally unique truth world")


def _cue(world, family, topology, changed, rng):
    if family == "duplicate_encoding":
        return {"family": family, "known_index": changed, "known_value": world[changed]}
    if family == "trend":
        return {"family": family}
    if topology == "star":
        center = rng.randrange(4)
        edges = [tuple(sorted((center, other))) for other in range(4) if other != center]
    else:
        nodes = rng.sample(range(4), 4)
        pairs = list(itertools.pairwise(nodes)) + (
            [(nodes[-1], nodes[0])] if topology == "cycle" else []
        )
        edges = [tuple(sorted(pair)) for pair in pairs]
    return {"family": family, "edges": [[i, j, world[i] + world[j]] for i, j in sorted(edges)]}


def _scene(spec, world, index, seed, rng, out, render):
    changed = rng.randrange(4)
    replacement = rng.randrange(99)
    replacement += replacement >= world[changed]
    observed = [replacement if i == changed else value for i, value in enumerate(world)]
    cue = _cue(world, spec["constraint_family"], spec["graph_topology"], changed, rng)
    solutions = solve(observed, cue)
    if solutions != [world]:
        raise ValueError("main-task world failed independent unique-solution verification")
    opaque_id = _digest({"seed": seed, "ordinal": index, "namespace": SCHEMA_VERSION})[:32]
    image_path = f"images/{opaque_id}.png"
    variant = "alternate_palette" if spec["ood_subtype"] == "render_expression" else "standard"
    image = render_chart(world, spec["chart_type"], out / image_path, variant) if render else {}
    answer = executor(world, spec["operation"])
    scene = {
        **spec,
        "schema_version": SCHEMA_VERSION,
        "base_scene_id": opaque_id,
        "truth_structure_hash": _digest(world),
        "truth_world": world,
        "observed_world": observed,
        "changed_index": changed,
        "cue": cue,
        "solution_count": len(solutions),
        "fiber_size": fiber_size(answer, spec["operation"]),
        "image_path": image_path,
        "image_hash": image.get("sha256"),
        "image_width": 768,
        "image_height": 512,
        "image_token_count": None,
        "processor_width": None,
        "processor_height": None,
        "render_variant": variant,
        "expression_variant": "fixed_N_v1",
        "error_magnitude": abs(world[changed] - replacement),
        "relation_count": len(cue["edges"])
        if "edges" in cue
        else 2
        if spec["constraint_family"] == "trend"
        else 1,
        "numeric_character_count": sum(len(str(value)) for value in observed),
        "numeric_token_count": None,
        "token_count_status": "P1_TOKENIZER_REQUIRED",
    }
    interfaces = (scene["interface"],) if scene["interface"] else INTERFACES
    return {
        **scene,
        "prompt_hashes": {
            interface: build_prompt(scene, interface)["prompt_hash"] for interface in interfaces
        },
    }


def _diagnostics(seed):
    cases = (
        ("no_solution", {"family": "cross_series", "edges": [[0, 1, 1], [0, 1, 2]]}),
        (
            "multiple_solutions",
            {"family": "duplicate_encoding", "known_index": 0, "known_value": 0},
        ),
        ("no_cue", None),
    )
    return [
        {
            "base_scene_id": _digest([seed, "diagnostic", name])[:32],
            "split": "diagnostic",
            "identifiability": name,
            "observed_world": [0, 0, 0, 0],
            "cue": cue,
            "solution_count": len(solve([0, 0, 0, 0], cue)),
            "truth_world": None,
            "main_classification_applicable": False,
        }
        for name, cue in cases
    ]


def _schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": SCHEMA_VERSION,
        "type": "object",
        "required": [
            "base_scene_id",
            "split",
            "truth_world",
            "observed_world",
            "cue",
            "solution_count",
            "fiber_size",
            "image_path",
            "image_hash",
        ],
        "properties": {
            key: {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0, "maximum": 99},
            }
            for key in ("truth_world", "observed_world")
        },
        "path_convention": (
            "image_path is relative to dataset directory; never concatenate truth values"
        ),
    }


def _manifest(out, sizes, seed, scenes, render):
    files = {
        path.name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(out.glob("*.jsonl"))
    }
    schema_path = out / "schema.json"
    files = {
        **files,
        "schema.json": {
            "sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
            "bytes": schema_path.stat().st_size,
        },
    }
    cells = Counter(
        "|".join(
            str(row.get(key))
            for key in (
                "split",
                "ood_subtype",
                "constraint_family",
                "chart_type",
                "operation",
                "interface",
            )
        )
        for row in scenes
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "split_sizes": sizes,
        "template_version": TEMPLATE_VERSION,
        "files": files,
        "images_rendered": render,
        "dataset_hash": _digest(files),
        "path_base": "dataset_directory",
        "solver_candidates_per_scene": 396,
        "unique_main_solutions": len(scenes),
        "truth_structure_hash_definition": "SHA256(canonical JSON four-integer truth array)",
        "global_truth_uniqueness": len({row["truth_structure_hash"] for row in scenes})
        == len(scenes),
        "allocation_by_cell": dict(sorted(cells.items())),
        "confirm_evaluated": False,
        "confirm_policy": (
            "Created and structurally checked only; "
            "every later access must be logged by split loader."
        ),
        "natural_pool_allocation_reason": (
            "ID domain has only 784 nonconstant arithmetic progressions; "
            "natural_pool balancing is not mandated; trend pool capped at 288 "
            "to preserve global uniqueness."
        ),
        "human_spot_check": {
            "required_images": 36,
            "status": "PENDING",
            "coverage": "3 constraint families x 2 charts x 3 operations x 2 images",
        },
        "processor_audit": (
            "PENDING_P1; original 768x512, actual processor dimensions/tokens not yet measured"
        ),
        "numeric_token_count": "PENDING_P1; numeric_character_count is a separate character metric",
    }


def generate_dataset(out, seed=17, sizes=None, render=True):
    sizes = dict(DEFAULT_SIZES if sizes is None else sizes)
    _validate_sizes(sizes)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() or any(out.glob("*.jsonl")):
        raise FileExistsError(
            "dataset directory contains generated artifacts; choose a fresh output directory"
        )
    rng = random.Random(seed)
    trend_pools = {lower: _trend_worlds(lower, rng) for lower in (0, 50)}
    specs = _specifications(sizes)
    # Reserve all scarce trend worlds before independent random families can collide.
    order = sorted(enumerate(specs), key=lambda item: item[1]["constraint_family"] != "trend")
    completed, seen = [], frozenset()
    for index, spec in order:
        world = _choose_world(spec, rng, trend_pools, seen)
        row = _scene(spec, world, index, seed, rng, out, render)
        seen = seen | {row["truth_structure_hash"]}
        completed = [*completed, (index, row)]
    scenes = [row for _, row in sorted(completed)]
    for split in sizes:
        _write_jsonl(out / f"{split}.jsonl", [row for row in scenes if row["split"] == split])
    _write_jsonl(out / "diagnostic.jsonl", _diagnostics(seed))
    _write_json(out / "schema.json", _schema())
    manifest = _manifest(out, sizes, seed, scenes, render)
    _write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--sizes", type=Path, help="JSON object overriding the complete split-size mapping"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="CPU schema checks only; images unavailable"
    )
    args = parser.parse_args()
    sizes = json.loads(args.sizes.read_text()) if args.sizes else None
    manifest = generate_dataset(args.out, args.seed, sizes, render=not args.no_render)
    print(
        json.dumps(
            {"dataset_hash": manifest["dataset_hash"], "split_sizes": manifest["split_sizes"]}
        )
    )


if __name__ == "__main__":
    main()
