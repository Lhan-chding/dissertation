"""Append-proof Phase 2a freeze of semantic, image, prompt, and orbit inputs."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from .constraint_basis import Matrix, apply_matrix, graph_signature, matrix_rank
from .orbit_transforms import transform_linear_system

PLAN_SHA256 = "e6a560e3db4a90353ed6e894cb531e9a85ef81b3157bc9c14eeb1f60a988f234"
FACT_SHA256 = "8b838f4148e02c02d3f6f55efcb1e0293fd6d247fd2969b197f533e5101ae9d6"
RAW_SHA256 = "f0ccb4d56415eecf90a2c456bfd7c92a33fc96a581f3603115edbcb253ba8c84"
GRAPH_AXES = (
    "familiar",
    "variable_permuted",
    "fact_order_permuted",
    "equivalent_basis",
    "sparse_mixed_ood",
)
FAMILIES = ("known_value", "pair_sum", "trend")
PROMPT_VERSION = "v5-common-world-output-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _matrix_for_family(family: str) -> Matrix:
    if family == "known_value":
        return ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    if family == "pair_sum":
        return ((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1), (1, 0, 1, 0))
    if family == "trend":
        return ((1, -2, 1, 0), (0, 1, -2, 1), (1, 1, 0, 0), (0, 0, 1, 1))
    raise ValueError("family is not registered")


def _render_chart(path: Path, truth: tuple[int, int, int, int]) -> None:
    image = Image.new("RGB", (280, 280), "white")
    draw = ImageDraw.Draw(image)
    draw.line((36, 230, 250, 230), fill="black", width=2)
    draw.line((36, 30, 36, 230), fill="black", width=2)
    colors = ("#3b82f6", "#ef4444", "#10b981", "#f59e0b")
    for index, (value, color) in enumerate(zip(truth, colors, strict=True)):
        left = 55 + index * 46
        height = round(value / 18 * 170)
        draw.rectangle((left, 230 - height, left + 25, 230), fill=color, outline="black")
        draw.text((left + 8, 236), chr(65 + index), fill="black")
    image.save(path, format="PNG", optimize=False)


def _prompt(matrix: Matrix, targets: tuple[int, ...]) -> str:
    equations = "\n".join(
        f"{','.join(map(str, row))} = {target}"
        for row, target in zip(matrix, targets, strict=True)
    )
    return (
        "Observed values: {observed_world}\n"
        f"Constraint rows (A | b):\n{equations}\n"
        "Recover the true world. Return exactly four comma-separated integers only.\n"
    )


def freeze_pre_model_factorial(
    output_root: Path, *, seed: int, canonical_per_family: int
) -> dict[str, object]:
    """Atomically freeze Phase 2a without a model call or natural observation."""

    if type(seed) is not int or type(canonical_per_family) is not int or canonical_per_family <= 0:
        raise ValueError("seed and canonical_per_family must be positive integers")
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Phase 2a output already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        images = temporary / "images"
        prompts = temporary / "prompts"
        images.mkdir()
        prompts.mkdir()
        rng = random.Random(seed)
        rows: list[dict[str, object]] = []
        for family in FAMILIES:
            base_matrix = _matrix_for_family(family)
            if matrix_rank(base_matrix) != 4:
                raise AssertionError("registered constraint family is not identifiable")
            for parent_index in range(canonical_per_family):
                parent_id = f"v5-{family}-{parent_index:04d}"
                base_truth = tuple(rng.randint(2, 18) for _ in range(4))
                base_targets = apply_matrix(base_matrix, base_truth)  # type: ignore[arg-type]
                for graph_axis in GRAPH_AXES:
                    truth, matrix, targets, transformation = transform_linear_system(
                        world=base_truth,  # type: ignore[arg-type]
                        matrix=base_matrix,
                        targets=base_targets,
                        graph_axis=graph_axis,
                    )
                    if matrix_rank(matrix) != 4 or apply_matrix(matrix, truth) != targets:
                        raise AssertionError("orbit transform changed the unique world")
                    scene_id = f"{parent_id}-{graph_axis.replace('_', '-')}"
                    image_relative = Path("images") / f"{scene_id}.png"
                    prompt_relative = Path("prompts") / f"{scene_id}.txt"
                    _render_chart(temporary / image_relative, truth)
                    (temporary / prompt_relative).write_text(
                        _prompt(matrix, targets), encoding="utf-8"
                    )
                    operation = ("sum", "difference", "max_minus_min")[len(rows) % 3]
                    if operation == "sum":
                        answer = truth[0] + truth[1]
                    elif operation == "difference":
                        answer = truth[0] - truth[1]
                    else:
                        answer = max(truth) - min(truth)
                    rows.append({
                        "schema_version": 1,
                        "scene_id": scene_id,
                        "semantic_scene_id": parent_id,
                        "split": "v5_pre_model_candidate",
                        "role": "phase2a_parent_orbit",
                        "truth": list(truth),
                        "family": family,
                        "capability_type": "direct" if family == "known_value" else "relational",
                        "constraint_matrix": [list(row) for row in matrix],
                        "constraint_targets": list(targets),
                        "matrix_rank": 4,
                        "unique_sparse_decoder": True,
                        "graph_signature": graph_signature(matrix),
                        "graph_axis": graph_axis,
                        "relation_depth": {
                            "known_value": 0,
                            "pair_sum": 1,
                            "trend": 2,
                        }[family],
                        "answer_operation": {"operator": operation, "indices": [0, 1]},
                        "correct_answer": answer,
                        "orbit_parent_id": parent_id,
                        "transformation": transformation,
                        "image_path": image_relative.as_posix(),
                        "image_sha256": _sha256(temporary / image_relative),
                        "prompt_path": prompt_relative.as_posix(),
                        "prompt_template_version": PROMPT_VERSION,
                        "prompt_sha256": _sha256(temporary / prompt_relative),
                        "observation_status": "pending_server_capture",
                    })
        rows_text = "".join(_canonical_json(row) + "\n" for row in rows)
        rows_path = temporary / "pre_model_rows.jsonl"
        rows_path.write_text(rows_text, encoding="utf-8")
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "PHASE_2A_PRE_MODEL_FROZEN",
            "dataset_id": f"qwen-v5-factorial-seed-{seed}",
            "plan_sha256": PLAN_SHA256,
            "fact_document_sha256": FACT_SHA256,
            "raw_archive_sha256": RAW_SHA256,
            "seed": seed,
            "value_domain": [2, 18],
            "image_size": [280, 280],
            "prompt_template_version": PROMPT_VERSION,
            "candidate_neighborhood": {"edit_order": 1, "value_domain": [2, 18]},
            "primary_predicate": "exactly_one_in_domain_natural_error",
            "stress_predicate": "multi_error_or_out_of_domain",
            "model_calls": 0,
            "observation_capture_required": True,
            "row_count": len(rows),
            "rows_sha256": _sha256(rows_path),
            "parent_manifest_sha256": None,
        }
        (temporary / "parent_manifest.json").write_text(
            _canonical_json(manifest) + "\n", encoding="utf-8"
        )
        temporary.rename(output_root)
        return manifest
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
