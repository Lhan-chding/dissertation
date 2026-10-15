"""Read-only frozen SR-F3-CE data, independent answers and model-input isolation.

The audit dataset contains truth and teachers. ModelInputStore deliberately does
not accept that dataset: its only sources are frozen input strings and images.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import contract
from .prompts import prompt_from_frozen, question_from_prompt

PACKAGE_MANIFEST_SHA256 = (
    "0e75e71f5e746cdb33a6d3254447a8224325efac966c4b34e5fa24eab41747bb"
)
MODEL_INPUTS_SHA256 = "329f8490345f47a812de3921947695f63e6d793e83e8462adcfead72991f4f74"
ROOT_COUNTS = {
    "ENGINE": 16,
    "TRAIN_RL": 64,
    "SUPPORT": 64,
    "MONITOR": 32,
    "TEST_ID": 128,
    "TEST_COMPOSITION": 128,
    "TEST_LANGUAGE": 64,
    "TEST_RENDER": 64,
    "TEST_ARITY": 64,
}
TASK_COUNTS = {
    "ENGINE": 64,
    "TRAIN_RL": 1024,
    "SUPPORT": 256,
    "MONITOR": 64,
    "TEST_ID": 2048,
    "TEST_COMPOSITION": 256,
    "TEST_LANGUAGE": 128,
    "TEST_RENDER": 128,
    "TEST_ARITY": 128,
}
ARMS = ("RL", "GOLD", "PREFIX", "EXEC")
FAMILIES = contract.FAMILIES
PARENTS = ("SRF2_J6_s71001", "SRF2_J6_s71002", "SRF2_J6_s71003")
FUTURE_SEEDS = (73101, 73102, 73103)
INPUT_FIELDS = {"qid", "pool", "chart", "prompt", "render_style"}
_QID = re.compile(r"[A-Za-z0-9_]+\Z")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(contract.canonical(value).encode()).hexdigest()


def bounded_path(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve(strict=True)
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected a nonempty relative path")
    parts = Path(relative).parts
    if any(part in (".", "..") for part in parts):
        raise ValueError("Path traversal is forbidden")
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("File escapes the input root or is not a file")
    return result


def read_json(path: str | Path):
    # Public artifacts exceed the model-output parser's 64K limit, but still
    # reject duplicate keys and non-finite numbers at the data boundary.
    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=contract._pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
    )


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{number}")
            row = json.loads(
                line,
                object_pairs_hook=contract._pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
            )
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(row)
    return rows


def verify_package(plan_dir: str | Path, *, verify_all: bool = True) -> dict:
    """Authenticate the inventory itself, then the frozen files without writing."""
    root = Path(plan_dir).resolve(strict=True)
    inventory_path = root / "PACKAGE_SHA256.json"
    if file_sha256(inventory_path) != PACKAGE_MANIFEST_SHA256:
        raise ValueError("Uploaded package hash inventory differs from frozen identity")
    inventory = read_json(inventory_path)
    checked = {}
    for name, expected in sorted(inventory.items()):
        if not verify_all and not name.startswith(("manifests/", "config/")):
            continue
        path = bounded_path(root, name)
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError("Frozen package changed: " + name)
        checked[name] = actual
    return {
        "status": "PASS",
        "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "checked_files": checked,
        "read_only": True,
    }


def independently_answer(world: dict, query: dict) -> Fraction | None:
    """Second evaluator: no contract execution, selection or aggregate calls."""
    rows = [
        dict(position=i, a=world["series"]["Alpha"][i], b=world["series"]["Beta"][i])
        for i in range(len(world["categories"]))
    ]
    family = query["family"]
    if family == "CROSS":
        rows = [
            r
            for r in rows
            if r["a"] >= r["b"] + (query.get("comparator", "gt") == "gt")
        ]
    elif family == "THRESHOLD":
        rows = [
            r
            for r in rows
            if r["a"] >= query["threshold"] + (query.get("comparator", "ge") == "gt")
        ]
    elif family == "TOPK":
        direction = -1 if query.get("direction", "largest") == "largest" else 1
        rows = sorted(rows, key=lambda r: (direction * r["a"], r["position"]))[
            : query.get("k", 2)
        ]
    elif family == "INTERVAL":
        lower = query["start"] + (not query.get("include_start", True))
        rows = [r for r in rows if lower <= r["position"] <= query["end"]]
        if query.get("positive_only", False):
            rows = [r for r in rows if r["a"] > r["b"]]
    else:
        raise ValueError("Unknown query family")
    value = query.get(
        "value",
        "Beta" if family == "TOPK" else "diff" if family == "INTERVAL" else "Alpha",
    )
    if value not in ("Alpha", "Beta", "diff"):
        raise ValueError("Unknown operand")
    values = [
        Fraction(
            r["a"]
            if value == "Alpha"
            else r["b"]
            if value == "Beta"
            else r["a"] - r["b"]
        )
        for r in rows
    ]
    operation = query["aggregate"]
    if operation == "count":
        return Fraction(len(values))
    if operation == "sum":
        return sum(values, Fraction())
    if not values:
        return None
    ordered = sorted(values)
    if operation == "mean":
        return sum(values, Fraction()) / len(values)
    if operation == "range":
        return ordered[-1] - ordered[0]
    if operation == "max":
        return ordered[-1]
    if operation == "min":
        return ordered[0]
    raise ValueError("Unknown aggregate")


def world_key(world: dict) -> str:
    """Full Alpha/Beta world identity; Gamma is deliberately irrelevant."""
    return contract.canonical(
        [world["categories"], world["series"]["Alpha"], world["series"]["Beta"]]
    )


@dataclass(frozen=True)
class FrozenDataset:
    plan_dir: Path
    roots: dict[str, dict]
    tasks: dict[str, dict]
    inputs: dict[str, dict]
    runs: list[dict]
    schedules: dict[int, list[dict]]
    audit: dict


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = row[key]
        if not isinstance(identifier, str) or not _QID.fullmatch(identifier):
            raise ValueError("Invalid identifier")
        if identifier in result:
            raise ValueError("Duplicate identifier: " + identifier)
        result[identifier] = row
    return result


def validate_manifest_rows(roots, tasks, inputs, runs, schedules) -> dict:
    """Validate exact frozen cardinalities, stream pairing and truth contracts."""
    if len(roots) != 624 or len(tasks) != 4096 or set(tasks) != set(inputs):
        raise ValueError("Expected 624 roots and exactly 4096 paired tasks/inputs")
    if Counter(r["pool"] for r in roots.values()) != ROOT_COUNTS:
        raise ValueError("Frozen root pool counts changed")
    if Counter(t["pool"] for t in tasks.values()) != TASK_COUNTS:
        raise ValueError("Frozen task pool counts changed")
    worlds = defaultdict(set)
    mismatch = []
    for qid, task in tasks.items():
        source = inputs[qid]
        if set(source) != INPUT_FIELDS:
            raise ValueError("MODEL_INPUTS schema changed")
        if any(source[k] != task[k] for k in INPUT_FIELDS):
            raise ValueError("Frozen model input differs from audited task: " + qid)
        root = roots[task["root_id"]]
        if any(root[k] != task[k] for k in ("pool", "family")):
            raise ValueError("Cross-pool or cross-family root leakage")
        world, query = task["world"], task["query"]
        contract.validate_world(world)
        if world["domain"] != [10, 90] or query["family"] != task["family"]:
            raise ValueError("Registered world/query identity changed")
        expected_n = 6 if task["pool"] == "TEST_ARITY" else 4
        if (
            world["categories"]
            != ["January", "February", "March", "April", "May", "June"][:expected_n]
        ):
            raise ValueError("Registered category order changed")
        expected_gamma = task["pool"] in ("TRAIN_RL", "TEST_ID") and task[
            "variant"
        ] in (4, 5)
        if set(world["series"]) != (
            {"Alpha", "Beta", "Gamma"} if expected_gamma else {"Alpha", "Beta"}
        ):
            raise ValueError("Registered series identities changed")
        expected_style = "novel" if task["pool"] == "TEST_RENDER" else "main"
        if task["render_style"] != expected_style or task["chart"] not in (
            "grouped_bar",
            "line",
        ):
            raise ValueError("Registered chart/style identity changed")
        if task["question"] != question_from_prompt(source["prompt"]):
            raise ValueError("Frozen prompt does not match its audited question")
        if qid in source["prompt"] or task["root_id"] in source["prompt"]:
            raise ValueError("Identifier leaked into model prompt")
        answer = None if task["answer"] is None else Fraction(task["answer"])
        if independently_answer(world, query) != answer:
            mismatch.append(qid)
        candidate = contract.canonical(
            {"evidence": task["certificate"], "answer": task["answer"]}
        )
        if contract.score(candidate, world, query)["J"] != 1:
            raise ValueError(
                "Frozen gold certificate does not satisfy the registered contract"
            )
        if task["pool"] == "SUPPORT":
            teacher = task["teacher"]
            if teacher["PREFIX"] != teacher["EXEC"]:
                raise ValueError("PREFIX and EXEC teacher sequences differ")
            for arm in ("GOLD", "PREFIX", "EXEC"):
                if teacher[arm] != contract.support_target(world, query, arm != "GOLD"):
                    raise ValueError(
                        "Frozen teacher text/spans differ from program trace"
                    )
        elif task["teacher"] is not None:
            raise ValueError("Teachers outside SUPPORT pool")
        worlds[world_key(world)].add(task["root_id"])
    collisions = [sorted(values) for values in worlds.values() if len(values) > 1]
    if mismatch or collisions:
        raise ValueError(
            f"Independent answers/world identity failed: {mismatch}, {collisions}"
        )
    expected_pairs = {(block, arm) for block in range(3) for arm in ARMS}
    if len(runs) != 12 or {(r["block"], r["arm"]) for r in runs} != expected_pairs:
        raise ValueError("Expected the complete 3-block, 4-arm run matrix")
    if len({r["run_id"] for r in runs}) != 12:
        raise ValueError("Repeated run ID")
    for run in runs:
        block = run["block"]
        expected = {
            "run_id": f"SRF3CE_{run['arm']}_b{block}",
            "block": block,
            "parent_id": PARENTS[block],
            "future_seed": FUTURE_SEEDS[block],
            "arm": run["arm"],
            "updates": 64,
            "schedule": f"SCHEDULE_b{block}.json",
        }
        if run != expected:
            raise ValueError("Registered run identity changed")
    if set(schedules) != {0, 1, 2}:
        raise ValueError("Expected three fixed schedules")
    for schedule in schedules.values():
        if len(schedule) != 64 or [row["step"] for row in schedule] != list(
            range(1, 65)
        ):
            raise ValueError("Training update schedule changed")
        for key, pool, count in (
            ("rl_qids", "TRAIN_RL", 16),
            ("support_qids", "SUPPORT", 4),
        ):
            stream = [qid for step in schedule for qid in step[key]]
            if len(set(stream)) != TASK_COUNTS[pool] or set(stream) != {
                qid for qid, task in tasks.items() if task["pool"] == pool
            }:
                raise ValueError("Scheduled tasks repeated, omitted or cross-pool")
            for step in schedule:
                if len(step[key]) != count:
                    raise ValueError("Effective batch size changed")
                selected = [tasks[qid] for qid in step[key]]
                if Counter(t["family"] for t in selected) != {
                    f: count // 4 for f in FAMILIES
                }:
                    raise ValueError("Per-step family balance changed")
                if key == "rl_qids":
                    strata = Counter(
                        (
                            t["family"],
                            t["chart"],
                            "legacy" if t["mode"].startswith("legacy") else t["mode"],
                        )
                        for t in selected
                    )
                    expected = {
                        (f, c, m): 1
                        for f in FAMILIES
                        for c in ("grouped_bar", "line")
                        for m in ("legacy", "repair")
                    }
                    if strata != expected:
                        raise ValueError(
                            "Per-step family/chart/old-repair strata changed"
                        )
    train = {
        contract.canonical(t["query"])
        for t in tasks.values()
        if t["pool"] in ("TRAIN_RL", "SUPPORT")
    }
    transfer = {
        contract.canonical(t["query"])
        for t in tasks.values()
        if t["pool"] == "TEST_COMPOSITION"
    }
    if train & transfer:
        raise ValueError("Transfer AST entered a training pool")
    return {
        "status": "PASS",
        "roots": 624,
        "tasks": 4096,
        "runs": 12,
        "schedules": 3,
        "root_counts": ROOT_COUNTS,
        "task_counts": TASK_COUNTS,
        "independent_answer_mismatches": [],
        "cross_root_world_collisions": [],
        "transfer_ast_overlap": [],
        "model_calls": 0,
    }


def load_frozen_dataset(plan_dir: str | Path) -> FrozenDataset:
    root = Path(plan_dir).resolve(strict=True)
    package = verify_package(root)
    manifests = root / "manifests"
    roots = _index(read_jsonl(manifests / "ROOTS_AUDIT_ONLY.jsonl"), "root_id")
    tasks = _index(read_jsonl(manifests / "TASKS_AUDIT_ONLY.jsonl"), "qid")
    inputs = _index(read_jsonl(manifests / "MODEL_INPUTS.jsonl"), "qid")
    runs = read_json(manifests / "RUN_MATRIX.json")
    schedules = {b: read_json(manifests / f"SCHEDULE_b{b}.json") for b in range(3)}
    audit = validate_manifest_rows(roots, tasks, inputs, runs, schedules)
    audit["package"] = package
    return FrozenDataset(root, roots, tasks, inputs, runs, schedules, audit)


def compare_historical_worlds(tasks: dict[str, dict], paths: list[str | Path]) -> dict:
    """Compare every new task world, including value interventions, with old worlds."""
    if not paths:
        return {
            "status": "NOT_CHECKED",
            "sources": [],
            "reason": "No historical world sources supplied",
        }
    historical = set()
    sources = []
    for source in paths:
        path = Path(source).resolve(strict=True)
        rows = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                "Historical source must be a nonempty list/JSONL of world rows"
            )
        source_keys = set()
        for row in rows:
            world = row.get("world", row)
            # Historical SR-F1/F1.2 worlds carry ``unit=count`` rather than a
            # domain field. Identity comparison needs their actual complete
            # Alpha/Beta arrays, not an invented modern schema or new values.
            categories = world.get("categories")
            series = world.get("series")
            if (
                not isinstance(categories, list)
                or not categories
                or any(not isinstance(category, str) for category in categories)
                or len(set(categories)) != len(categories)
                or not isinstance(series, dict)
                or not {"Alpha", "Beta"} <= set(series)
                or any(
                    not isinstance(series[name], list)
                    or len(series[name]) != len(categories)
                    or any(type(value) is not int for value in series[name])
                    for name in ("Alpha", "Beta")
                )
            ):
                raise ValueError("Malformed historical numerical-world identity")
            source_keys.add(world_key(world))
        historical.update(source_keys)
        sources.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "unique_alpha_beta_worlds": len(source_keys),
            }
        )
    overlap = sorted(
        qid for qid, task in tasks.items() if world_key(task["world"]) in historical
    )
    return {
        "status": "PASS" if not overlap else "FAIL",
        "sources": sources,
        "historical_unique_worlds": len(historical),
        "new_tasks_checked": len(tasks),
        "overlapping_qids": overlap,
        "scope": "All supplied historical worlds; source completeness is a separate gate",
    }


class ModelInputStore:
    """The only supported production route from a qid to prompt and visible image.

    This class never reads TASKS_AUDIT_ONLY or ROOTS_AUDIT_ONLY, even to derive
    natural-work prompts. Runtime must use its exact three-field return value.
    """

    def __init__(self, inputs: dict, images: dict, image_root: str | Path):
        self._inputs = copy.deepcopy(inputs)
        self._images = copy.deepcopy(images)
        self.image_root = Path(image_root).resolve(strict=True)
        if set(inputs) != set(images):
            raise ValueError(
                "Every frozen input requires exactly one rendered image mapping"
            )
        for qid, row in inputs.items():
            if set(row) != INPUT_FIELDS or row["qid"] != qid:
                raise ValueError("Unexpected model input schema")
            question_from_prompt(row["prompt"])
        for qid, row in images.items():
            if row["qid"] != qid or row["image_file"] != f"images/{qid}.png":
                raise ValueError("Unexpected qid-to-image route")
            if not re.fullmatch(r"[a-f0-9]{64}", row["image_sha256"]):
                raise ValueError("Malformed image hash")

    @classmethod
    def from_files(cls, inputs_path, image_manifest_path, *, image_root):
        if file_sha256(inputs_path) != MODEL_INPUTS_SHA256:
            raise ValueError("Frozen MODEL_INPUTS identity changed")
        return cls(
            _index(read_jsonl(inputs_path), "qid"),
            _index(read_jsonl(image_manifest_path), "qid"),
            image_root,
        )

    def model_input(self, qid: str, protocol: str = "structured") -> dict:
        source, image = self._inputs[qid], self._images[qid]
        path = bounded_path(self.image_root, image["image_file"])
        if file_sha256(path) != image["image_sha256"]:
            raise ValueError("Rendered image changed: " + qid)
        return {
            "prompt": prompt_from_frozen(source["prompt"], protocol),
            "image_file": image["image_file"],
            "image_sha256": image["image_sha256"],
        }
