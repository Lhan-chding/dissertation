"""Read-only C3 frozen retest, retaining original prompts, parser and executor.

This qualifies today's source functions; missing historical weights and effective
library defaults still preclude claiming exact reproduction of the old run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCES = {
    "parser": "src/compensability/study_c3/semantic_action_parser.py",
    "executor": "src/compensability_v5/data/common_action_schema.py",
    "taxonomy": "src/compensability/study_c3/action_taxonomy.py",
    "evaluation": "src/compensability/study_c3/evaluation_runtime.py",
    "prompt_renderer": "src/compensability/study_c3/qwen_backend.py",
}
EVALUATION_SPLITS = frozenset({"dev", "test", "positive_control"})


def _regular_path(path):
    path = Path(path).absolute()
    # NTU storage directories have legitimate /projects aliases; reject leaf links.
    if path.is_symlink():
        raise ValueError("legacy input/source symlink is not allowed")
    if not path.is_file():
        raise ValueError(f"legacy input/source is not a regular file: {path}")
    return path.resolve()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=2)
def _source_module(kind):
    """Load only two audited stdlib-only files, never package/training entrypoints."""
    if kind not in {"parser", "executor"}:
        raise ValueError("not an allowlisted pure-function source")
    path = _regular_path(REPOSITORY / SOURCES[kind])
    name = f"_ssvc_legacy_{kind}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their defining module during class creation.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _world(values):
    if (
        not isinstance(values, list)
        or len(values) != 4
        or any(type(v) is not int or not 2 <= v <= 18 for v in values)
    ):
        raise ValueError("legacy world must have four integers in [2,18]")
    return tuple(values)


def _answer(world, operation):
    source = _source_module("executor")
    return source.apply_answer_operation(source.WorldAction(world), operation)


def load_legacy_scenes(data_path, manifest_path):
    """Verify the original data manifest, returning all original eval scenes."""
    data_path, manifest_path = _regular_path(data_path), _regular_path(manifest_path)
    raw = data_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("fiber_rows_sha256") != _sha(raw):
        raise ValueError("legacy data hash does not match manifest")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if manifest.get("prompt_count") != len(rows):
        raise ValueError("legacy manifest prompt count mismatch")
    seen, selected, pairs = set(), [], defaultdict(list)
    pair_splits = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("legacy row must be an object")
        for key in ("scene_id", "pair_id", "family", "split", "prompt"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"legacy row missing {key}")
        if row["scene_id"] in seen:
            raise ValueError("duplicate legacy scene identity")
        seen.add(row["scene_id"])
        if pair_splits.setdefault(row["pair_id"], row["split"]) != row["split"]:
            raise ValueError("legacy base pair split leakage")
        if row.get("condition") not in {"collision", "separating"}:
            raise ValueError("unknown legacy condition")
        if row.get("prompt_sha256") != _sha(row["prompt"].encode()):
            raise ValueError("legacy prompt hash mismatch")
        truth = _world(row.get("truth"))
        _world(row.get("observation"))
        answer = _answer(truth, row.get("operation"))
        if type(row.get("gold_answer")) is not int or row["gold_answer"] != answer:
            raise ValueError("legacy gold answer disagrees with original executor")
        if row["split"] in EVALUATION_SPLITS:
            selected.append(
                {
                    **row,
                    "base_scene_id": row["pair_id"],
                    "track": "L",
                    "interface": row["condition"],
                    "constraint_family": row["family"],
                    "legacy_exact_reproduction": False,
                }
            )
            pairs[row["pair_id"]].append(row)
    for pair in pairs.values():
        if (
            len(pair) != 2
            or {r["condition"] for r in pair} != {"collision", "separating"}
            or any(
                pair[0][key] != pair[1][key]
                for key in ("truth", "observation", "split", "family", "facts")
            )
        ):
            raise ValueError("legacy evaluation pair is not two matched conditions")
    if not selected:
        raise ValueError("legacy evaluation contains no scene pairs")
    return selected


def build_legacy_prompt(scene):
    """Preserve C3's single user message; do not inject N-track system instructions."""
    user = scene["prompt"]
    if not isinstance(user, str) or _sha(user.encode()) != scene["prompt_sha256"]:
        raise ValueError("legacy prompt hash mismatch")
    return {
        "system": None,
        "user": user,
        "prompt_hash": scene["prompt_sha256"],
        "messages": [{"role": "user", "content": user}],
        "template_version": "L-C3-user-only",
        "add_generation_prompt": True,
    }


def annotate_legacy(raw, scene):
    """Use C3 semantic P1 primary labels; retain canonical P0 as a separate audit."""
    parser = _source_module("parser")
    candidate = parser.parse_p1(raw)
    truth = _world(scene["truth"])
    category = (
        "I"
        if candidate is None
        else "X"
        if candidate == truth
        else "S"
        if _answer(candidate, scene["operation"]) == _answer(truth, scene["operation"])
        else "W"
    )
    return {
        "category": category,
        "parsed_world": None if candidate is None else list(candidate),
        "syntax_valid": parser.parse_complete_action_any_domain(raw) is not None,
        "copy_observation": candidate is not None and list(candidate) == scene["observation"],
        # Legacy validity is parser + numeric domain, not N's constraint solver.
        "constraint_satisfaction": None,
        "canonical_parse_success": parser.parse_p0(raw) is not None,
        "semantic_parse_success": candidate is not None,
        "parser_id": "C3.semantic_action_parser.parse_p1",
    }


def legacy_qualification():
    """Small executable qualification, with original source hashes for run identity."""
    scene = {
        "truth": [3, 8, 5, 18],
        "observation": [3, 8, 5, 17],
        "operation": {"operator": "sum", "indices": [0, 1]},
    }
    cases = [
        ("3,8,5,18", "X"),
        ("[3,8,5,18]", "X"),
        ("v1=3,v2=8,v3=5,v4=18", "X"),
        ("3,8,5,17", "S"),
        ("4,8,5,17", "W"),
        ("3,8,5,18\nexplanation", "I"),
        ("1,8,5,18", "I"),
    ]
    outcomes = [
        {"raw": raw, "expected": expected, "observed": annotate_legacy(raw, scene)["category"]}
        for raw, expected in cases
    ]
    operations = [
        ({"operator": "sum", "indices": [0, 1]}, 11),
        ({"operator": "difference", "indices": [0, 1]}, -5),
        ({"operator": "max_minus_min", "indices": [0, 1, 2, 3]}, 15),
    ]
    return {
        "passed": all(r["expected"] == r["observed"] for r in outcomes)
        and all(_answer(tuple(scene["truth"]), op) == expected for op, expected in operations),
        "fixture_cases": outcomes,
        "source_hashes": {
            relative: _sha(_regular_path(REPOSITORY / relative).read_bytes())
            for relative in SOURCES.values()
        },
        "parser_id": "C3.semantic_action_parser.parse_p1",
        "executor_id": "common_action_schema.apply_answer_operation",
        "prompt_renderer": "single user message, add_generation_prompt=True",
        "legacy_exact_reproduction": False,
        "qualification_scope": "current read-only source functions; not historic effective runtime",
        "constraint_satisfaction": "unavailable; no new constraint semantics introduced",
    }
