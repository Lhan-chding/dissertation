"""Materialize the supplied F2 sources using the unchanged audited chart renderer.

This module has no model generation or weight-loading entrypoint. Truth-bearing
questions are private; ``model_input`` is the sole projection sent to processors.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from mm_core import generator as gen

PLAN_ID = "MM-DEV-F2-QWEN35-9B-20261009"
POOLS = {"ENGINE_F2": 4, "PREP": 32, "CONTINUE": 64, "PROBE": 64, "DEV_EVAL": 128}
INPUT_KEYS = ("question_id", "image_id", "image_path", "image_sha256", "prompt", "prompt_sha256")
REPORT = "manifests/DATA_MATERIALIZATION.json"


def digest(value: Any) -> str:
    return hashlib.sha256(gen._canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(b"".join(gen._canonical(row) + b"\n" for row in rows))
    temporary.replace(path)


def relative_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its registered root: {name}")
    return path


def model_input(row: dict[str, Any]) -> dict[str, Any]:
    """Ignore every sidecar field, including adversarial truth/reward metadata."""
    result = {key: row[key] for key in INPUT_KEYS}
    if row.get("processed_pixel_sha256"):
        result["processed_pixel_sha256"] = row["processed_pixel_sha256"]
    return result


def _mutable(root: Path) -> None:
    if any((root / p).exists() for p in ("F2_FREEZE.json", "manifests/F2_FREEZE.json")):
        raise PermissionError("Data mutation is forbidden after F2 freeze")


def load_inputs(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Authenticate the supplied config and sources; never regenerate candidates."""
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("plan_id") != PLAN_ID or plan["data"]["root_counts"] != POOLS:
        raise ValueError("Unregistered F2 plan or pool counts")
    if file_hash(Path(gen.__file__)) != plan["data"]["generator_module_sha256"]:
        raise ValueError("Audited generator source identity changed")
    design = plan_path.parent.parent
    entries = json.loads((design / "BUNDLE_MANIFEST.json").read_text())["files"]
    registered = {entry["path"]: entry["sha256"] for entry in entries}
    files = [plan_path, relative_file(plan_path.parent, plan["data"]["numeric_sources"])]
    files += [plan_path.parent / "MANIFEST_BUILD_RECEIPT.json"]
    files += sorted((plan_path.parent / "schedules").glob("*.jsonl"))
    hashes = {}
    for path in files:
        name = str(path.relative_to(design))
        actual = file_hash(path)
        if registered.get(name) != actual:
            raise ValueError(f"Supplied bundle identity mismatch: {name}")
        hashes[name] = actual
    receipt = json.loads(files[2].read_text())
    if receipt["numeric_sources_sha256"] != file_hash(files[1]):
        raise ValueError("Source construction receipt mismatch")
    if len(files[3:]) != 9:
        raise ValueError("All nine registered schedules are required")
    return plan, read_jsonl(files[1]), hashes


def load_history(historical_root: Path | None) -> dict[str, Any]:
    """Require actual frozen audit data, including original and processed hashes."""
    from PIL import Image

    if historical_root is None:
        raise ValueError("--historical-root is required; reconstructed history is not evidence")
    root = historical_root.resolve()
    preflight_path = root / "manifests/PROCESSOR_PREFLIGHT.json"
    freeze = json.loads((root / "manifests/PRE_INFERENCE_FREEZE.json").read_text())
    if freeze["processor_preflight_hash"] != file_hash(preflight_path):
        raise ValueError("Historical processor receipt is not bound to its freeze")
    preflight = json.loads(preflight_path.read_text())
    artifacts = preflight["artifact_sha256"]
    paths = ("data/questions.jsonl", "data/sources.jsonl", "data/images.jsonl")
    for name in paths:
        if artifacts[name] != file_hash(root / name):
            raise ValueError(f"Historical data hash mismatch: {name}")
    questions, sources, images = (read_jsonl(root / p) for p in paths)
    if (len(questions), len(sources), len(images)) != (864, 72, 288):
        raise ValueError("All historical audit panels are required for exclusion")
    original_pixels = []
    for row in images:
        path = relative_file(root, row["image_path"])
        if file_hash(path) != row["image_sha256"]:
            raise ValueError("Historical original image changed")
        if not row.get("processed_image_hash") or not row.get("processed_pixel_sha256"):
            raise ValueError("Historical processed-image evidence is incomplete")
        with Image.open(path) as image:
            original_pixels.append(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
    report_path = root / "manifests/GENERATOR_REPORT.json"
    if preflight["generator_report_sha256"] != file_hash(report_path):
        raise ValueError("Historical renderer receipt changed")
    renderer = json.loads(report_path.read_text())["renderer"]
    return {
        "status": "VERIFIED_FROZEN_HISTORICAL_INPUTS",
        "evidence_sha256": {
            **{p: artifacts[p] for p in paths},
            "manifests/PROCESSOR_PREFLIGHT.json": file_hash(preflight_path),
            "manifests/PRE_INFERENCE_FREEZE.json": file_hash(
                root / "manifests/PRE_INFERENCE_FREEZE.json"
            ),
            "manifests/GENERATOR_REPORT.json": file_hash(report_path),
        },
        "renderer": renderer,
        "ordered_pairs": {
            d: sorted({tuple(q["true_values"][:2]) for q in questions if q["D"] == d})
            for d in gen.LEVELS
        },
        "range_sets": sorted(
            {tuple(sorted(q["true_values"])) for q in questions if q["operation"] == "range"}
        ),
        "root_families": sorted({r["root_family_id"] for r in sources}),
        "source_hashes": sorted({r["source_graph_hash"] for r in sources}),
        "visible_source_hashes": sorted({r["visible_source_hash"] for r in images}),
        "image_hashes": sorted({r["image_sha256"] for r in images}),
        "original_pixel_hashes": sorted(set(original_pixels)),
        "processed_image_hashes": sorted({r["processed_image_hash"] for r in images}),
        "processed_pixel_hashes": sorted({r["processed_pixel_sha256"] for r in images}),
    }


def validate_sources(sources: list[dict[str, Any]], history: dict[str, Any]) -> dict[str, Any]:
    seen_pairs = {d: {tuple(v) for v in history["ordered_pairs"][d]} for d in gen.LEVELS}
    triples = {tuple(v) for v in history["range_sets"]}
    graphs = set(history["source_hashes"])
    identities = set()
    for row in sources:
        pool, index, level = row["pool"], row["root_index"], row["numeric_level"]
        if pool not in POOLS or isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Invalid source pool/root index")
        if not 0 <= index < POOLS[pool] or level not in gen.LEVELS:
            raise ValueError("Source outside the registered root/level range")
        family = f"mmdev-f2-{pool.lower()}-r{index:04d}"
        if row["plan_id"] != PLAN_ID or row["root_family_id"] != family:
            raise ValueError("Source family identity mismatch")
        if (family, level) in identities or family in history["root_families"]:
            raise ValueError("Duplicate or historical source family")
        identities.add((family, level))
        source = row["source"]
        fixed = {
            "axis": [0, 100],
            "categories": list(gen.CATEGORIES),
            "series": list(gen.SERIES),
            "target_columns": list(gen.TARGET_COLUMNS),
            "target_series_index": 0,
            "units": "count",
            "delta": 1,
        }
        if any(source.get(k) != v for k, v in fixed.items()):
            raise ValueError("Source chart contract changed")
        values = source["values"]
        if len(values) != 4 or any(len(v) != 6 for v in values):
            raise ValueError("Expected four series by six categories")
        if any(type(v) is not int or not 10 <= v <= 89 for series in values for v in series):
            raise ValueError("Source values violate registered integer domain")
        targets = [values[0][i] for i in gen.TARGET_COLUMNS]
        if (
            targets != row["truth_targets"]
            or min(abs(a - b) for i, a in enumerate(targets) for b in targets[i + 1 :]) < 5
        ):
            raise ValueError("Source truth/gap mismatch")
        if any(
            gen.decimal_difficulty(op, targets[: 3 if op == "range" else 2]) != level
            for op in gen.OPERATIONS
        ):
            raise ValueError("D label does not satisfy all three operations")
        graph_hash = digest(source)
        if graph_hash != row["source_graph_hash_f2"] or graph_hash in graphs:
            raise ValueError("Full-source hash mismatch or collision")
        pair, triple = tuple(targets[:2]), tuple(sorted(targets))
        if pair in seen_pairs[level] or triple in triples:
            raise ValueError("Numeric-content duplicate within new pools or historical audit")
        seen_pairs[level].add(pair)
        triples.add(triple)
        graphs.add(graph_hash)
    expected = {
        (f"mmdev-f2-{p.lower()}-r{i:04d}", d)
        for p, n in POOLS.items()
        for i in range(n)
        for d in gen.LEVELS
    }
    if identities != expected:
        raise ValueError("The full registered 584-source panel is required")
    return {
        "status": "PASS",
        "numeric_sources": len(sources),
        "root_families": len(expected) // 2,
        "historical_exclusion": "PASS",
        "numeric_content_duplicates": 0,
    }


def _question_rows(source_row: dict[str, Any], image: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    protocol = digest({"template": gen.PROMPT_TEMPLATE, "operation_text": gen.OPERATION_TEXT})
    for operation in gen.OPERATIONS:
        values = source_row["truth_targets"][: 3 if operation == "range" else 2]
        answer = (
            sum(values)
            if operation == "sum"
            else values[0] - values[1]
            if operation == "difference"
            else max(values) - min(values)
        )
        prompt = gen.make_prompt(operation)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        rows.append(
            {
                **image,
                "question_id": f"{image['image_id']}-{operation}",
                "operation": operation,
                "readset_id": f"{image['image_id']}-readset-{len(values)}",
                "ordered_item_ids": [
                    f"Amber:{gen.CATEGORIES[c]}" for c in gen.TARGET_COLUMNS[: len(values)]
                ],
                "true_values": values,
                "true_values_decimal": [str(v) for v in values],
                "gold_answer_decimal": str(answer),
                "delta": 1,
                "delta_decimal": "1",
                "delta_source": "integer_grid_min_raster_gap_4_pixels",
                "axis_min": "0",
                "axis_max": "100",
                "quantity_units": "count",
                "protocol_hash": protocol,
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "actual_model_prompt_hash": prompt_hash,
                "actual_model_tokenized_prompt_hash": None,
                "prompt_token_count": None,
                "prompt_characters": len(prompt),
                "target_digit_count": [2] * len(values),
                "target_magnitude": max(values),
                "minimum_target_value_gap": min(
                    abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :]
                ),
                "extrema_positions": {
                    "min": values.index(min(values)),
                    "max": values.index(max(values)),
                },
                "gold_response_characters": len(json.dumps({"readings": values, "answer": answer})),
            }
        )
    return rows


def render_source(source_row: dict[str, Any], root: Path) -> tuple[list[dict], list[dict]]:
    """Render exactly the four V/chart variants of a supplied numeric source."""
    from PIL import Image

    images, questions = [], []
    source, level, family = (
        source_row["source"],
        source_row["numeric_level"],
        source_row["root_family_id"],
    )
    numeric_id = f"{family}-d{level[0]}"
    for visual in gen.LEVELS:
        count = 2 if visual == "low" else 4
        visible = {**source, "series": source["series"][:count], "values": source["values"][:count]}
        for chart in gen.CHART_TYPES:
            variant = f"v{visual[0]}-{chart}"
            image_id = f"{numeric_id}-{variant}"
            path = f"data/images/{image_id}.png"
            qa = gen._render(source, chart, visual, root / path)
            if qa["original_geometry_status"] != "PASS":
                raise ValueError(f"Original geometry failed: {image_id}")
            image_hash = file_hash(root / path)
            with Image.open(root / path) as rgb:
                pixels_hash = hashlib.sha256(rgb.convert("RGB").tobytes()).hexdigest()
            image = {
                "plan_id": PLAN_ID,
                "pool": source_row["pool"],
                "split": source_row["pool"],
                "root_index": source_row["root_index"],
                "root_family_id": family,
                "numeric_root_id": numeric_id,
                "image_id": image_id,
                "render_variant_id": variant,
                "chart_type": chart,
                "visual_level": visual,
                "numeric_level": level,
                "V": visual,
                "D": level,
                "source_graph_hash": digest(source),
                "visible_source_hash": digest(visible),
                "image_path": path,
                "image_hash": image_hash,
                "image_sha256": image_hash,
                "original_pixel_sha256": pixels_hash,
                **qa,
            }
            images.append(image)
            questions.extend(_question_rows(source_row, image))
    return images, questions


def _image_audit(images: list[dict], history: dict, *, processed: bool = False) -> dict:
    checks = {"image_sha256": "image_hashes", "original_pixel_sha256": "original_pixel_hashes"}
    if processed:
        checks.update(
            processed_image_hash="processed_image_hashes",
            processed_pixel_sha256="processed_pixel_hashes",
        )
    for field, old_key in checks.items():
        seen = set(history[old_key])
        for row in images:
            identity = row.get(field)
            if not identity or identity in seen:
                raise ValueError(f"Missing, duplicate or historical image identity: {field}")
            seen.add(identity)
    for field, old_key in (
        ("source_graph_hash", "source_hashes"),
        ("visible_source_hash", "visible_source_hashes"),
    ):
        owners = {}
        for row in images:
            value, owner = row[field], row["numeric_root_id"]
            if value in history[old_key] or (value in owners and owners[value] != owner):
                raise ValueError(f"Cross-source content collision: {field}")
            owners[value] = owner
    return {
        "status": "PASS" if processed else "PASS_ORIGINAL_PENDING_PROCESSOR",
        "images": len(images),
        "historical_image_exclusion": "PASS",
        "processed_image_audit": "PASS" if processed else "PENDING_PROCESSOR",
        "findings": [],
    }


def qa_contact_sheets(root: Path, images: list[dict], *, processed: bool = False) -> dict:
    from PIL import Image, ImageDraw, ImageFont

    prefix = "processed" if processed else "original"
    path_key = "processed_image_path" if processed else "image_path"
    selected, sheets = [], []
    for pool, count in POOLS.items():
        rows = sorted(
            (r for r in images if r["pool"] == pool and r["root_index"] in (0, count - 1)),
            key=lambda r: (r["root_index"], r["chart_type"], r["V"], r["D"]),
        )
        if len(rows) != 16:
            raise ValueError("QA requires first and last root and all eight image cells per pool")
        sheet = Image.new("RGB", (1792, 1472), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            x, y = (i % 4) * 448, (i // 4) * 368
            label = (
                f"r{row['root_index']:04d} / {row['chart_type']} / V{row['V'][0]} D{row['D'][0]}"
            )
            draw.text((x + 6, y + 4), label, fill="black", font=ImageFont.load_default(size=15))
            with Image.open(root / row[path_key]) as image:
                sheet.paste(image.resize((448, 336)), (x, y + 30))
            selected.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row[path_key],
                    "pool": pool,
                    "root_index": row["root_index"],
                    "chart_type": row["chart_type"],
                    "V": row["V"],
                    "D": row["D"],
                }
            )
        path = f"data/qa/{prefix}_{pool.lower()}_first_last.png"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(root / path)
        sheets.append({"path": path, "sha256": file_hash(root / path)})
    return {
        "selection_rule": "first_and_last_registered_root_per_pool_all_chart_V_D_cells",
        "selected_image_count": len(selected),
        "samples": selected,
        "contact_sheets": sheets,
        "thumbnails_do_not_replace_full_resolution_review": True,
    }


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): file_hash(p)
        for pattern in ("*.jsonl", "*.json")
        for p in sorted((root / "data").glob(pattern))
    }


def validate_processed_manifests(
    root: Path, questions: list[dict], images: list[dict], report: dict
) -> dict[str, int]:
    """Check full saved native routing coverage and bind every annotation.

    This is read-only and does not retokenize, load weights, or change the
    materializer. Actual tensors are rechecked by F2Runtime before model calls.
    """
    routing_path = root / "data/processor_routing.jsonl"
    processed_path = root / "data/processed_images.jsonl"
    if not routing_path.is_file() or not processed_path.is_file():
        raise ValueError("Processed state requires both native processor manifests")
    routes, processed = read_jsonl(routing_path), read_jsonl(processed_path)

    def unique(rows: list[dict], key: str) -> dict[str, dict]:
        if any(not isinstance(row.get(key), str) or not row[key] for row in rows):
            raise ValueError(f"Missing processor manifest identity: {key}")
        result = {row[key]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f"Duplicate processor manifest identity: {key}")
        return result

    route_map, processed_map = unique(routes, "question_id"), unique(processed, "image_id")
    question_map, image_map = unique(questions, "question_id"), unique(images, "image_id")
    if set(route_map) != set(question_map) or set(processed_map) != set(image_map):
        raise ValueError("Native processor manifests do not exactly cover all questions/images")
    if (
        type(report.get("processed_question_count")) is not int
        or type(report.get("processed_image_count")) is not int
        or report["processed_question_count"] != len(routes)
        or report["processed_image_count"] != len(processed)
    ):
        raise ValueError("Reported processor counts disagree with actual manifest rows")
    identity = report.get("processor_identity", {})
    geometry = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}
    if identity.get("processor_geometry") != geometry:
        raise ValueError("Saved processor geometry is not the native Qwen3.5 contract")
    image_fields = (
        "processed_image_path",
        "processed_image_hash",
        "processed_pixel_sha256",
        "processed_rgb_sha256",
        "processed_size",
        "processed_pixels_per_delta",
        "minimum_processed_raster_grid_gap_pixels",
        "max_inverse_rgb_error",
        "processor_hash",
        "processor_status",
        "readability_status",
    )
    sha_fields = (
        "processed_pixel_sha256",
        "source_image_sha256",
        "mm_token_type_ids_sha256",
        "processor_hash",
        "chat_template_hash",
        "chat_template_kwargs_hash",
        "chat_text_sha256",
        "input_ids_sha256",
    )
    image_token_ids = set()
    for image_id, image in image_map.items():
        saved = processed_map[image_id]
        if saved.get("image_path") != image["image_path"]:
            raise ValueError("Processed manifest source-image path differs")
        if any(key not in saved or saved[key] != image.get(key) for key in image_fields):
            raise ValueError("Processed image manifest differs from registered image annotations")
        if (
            saved["processor_status"] != "ACTUAL_CPU_PROCESSOR_VERIFIED"
            or saved["processed_size"] != list(gen.CANVAS)
            or saved["processed_pixels_per_delta"] != 4.88
            or saved["minimum_processed_raster_grid_gap_pixels"] < 4
            or not 0 <= saved["max_inverse_rgb_error"] <= 1
            or saved["processor_hash"] != identity.get("processor_hash")
        ):
            raise ValueError("Processed image QA or processor identity is inconsistent")
    for question_id, question in question_map.items():
        route = route_map[question_id]
        image_id = question["image_id"]
        image = image_map.get(image_id)
        if image is None or route.get("image_id") != image_id:
            raise ValueError("Processor route refers to a different question image")
        if any(question.get(key) != image.get(key) for key in image_fields):
            raise ValueError("Question processed annotations differ from its image")
        for key in sha_fields:
            value = route.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"Native processor hash is missing or malformed: {key}")
        expected = {
            "processed_pixel_sha256": image["processed_pixel_sha256"],
            "source_image_sha256": image["image_sha256"],
            "source_size": image["original_size"],
            "processed_size": image["processed_size"],
            "pixels_per_delta": image["processed_pixels_per_delta"],
            "input_ids_sha256": question.get("actual_model_tokenized_prompt_hash"),
            "chat_text_sha256": question.get("actual_chat_text_sha256"),
            "image_grid_thw": [[1, 42, 56]],
            "pixel_values_shape": [2352, 1536],
            "image_token_count": 588,
            **{
                key: identity.get(key)
                for key in (
                    "processor_hash",
                    "chat_template_hash",
                    "chat_template_kwargs_hash",
                    "chat_template_kwargs",
                )
            },
        }
        if any(route.get(key) != value for key, value in expected.items()):
            raise ValueError("Native processor routing differs from question/image annotations")
        count = question.get("prompt_token_count")
        if type(count) is not int or count <= 588:
            raise ValueError("Question lacks actual positive text/image token accounting")
        if route.get("input_ids_shape") != [1, count] or route.get("mm_token_type_ids_shape") != [
            1,
            count,
        ]:
            raise ValueError("Native input/MM token shapes do not match question token count")
        token_id = route.get("image_token_id")
        if type(token_id) is not int or token_id <= 0:
            raise ValueError("Native image-token identity is missing")
        image_token_ids.add(token_id)
    if len(image_token_ids) != 1:
        raise ValueError("Native image-token identity differs across questions")
    return {"processed_question_count": len(routes), "processed_image_count": len(processed)}


def validate_materialization(plan_path: Path, run_root: Path) -> dict[str, Any]:
    """Read-only integrity gate; an existing report alone never establishes PASS."""
    root = run_root.resolve()
    _, sources, inputs = load_inputs(plan_path)
    report = json.loads((root / REPORT).read_text())
    if report["input_sha256"] != inputs:
        raise ValueError("Materialized source/config identity differs")
    if report["artifact_sha256"] != _artifact_hashes(root):
        raise ValueError("Materialized manifest bytes changed")
    history = json.loads((root / "data/history_exclusions.json").read_text())
    validate_sources(sources, history)
    questions, sidecars, images, model_rows = (
        read_jsonl(root / f"data/{p}.jsonl")
        for p in ("questions", "sidecars", "images", "model_inputs")
    )
    if read_jsonl(root / "data/sources.jsonl") != sources:
        raise ValueError("Materialized sources differ from the supplied source manifest")
    if (len(questions), len(images)) != (7008, 2336) or questions != sidecars:
        raise ValueError("Incomplete questions/images or inconsistent sidecars")
    if [model_input(row) for row in questions] != model_rows:
        raise ValueError("Model-input manifest must be the exact safe projection")
    ids = {q["question_id"] for q in questions}
    if len(ids) != 7008:
        raise ValueError("Question identities are not unique")
    image_map = {r["image_id"]: r for r in images}
    if len(image_map) != 2336:
        raise ValueError("Image identities are not unique")
    source_map = {f"{r['root_family_id']}-d{r['numeric_level'][0]}": r for r in sources}
    expected_question_map = {}
    for image in images:
        source = source_map.get(image["numeric_root_id"])
        if source is None:
            raise ValueError("Image numeric source is not registered")
        visual, chart = image["V"], image["chart_type"]
        if visual not in gen.LEVELS or chart not in gen.CHART_TYPES:
            raise ValueError("Unregistered image variant")
        expected_id = f"{image['numeric_root_id']}-v{visual[0]}-{chart}"
        if (
            image["image_id"] != expected_id
            or image["D"] != source["numeric_level"]
            or image["pool"] != source["pool"]
            or image["split"] != source["pool"]
            or image["root_index"] != source["root_index"]
            or image["root_family_id"] != source["root_family_id"]
            or image["source_graph_hash"] != digest(source["source"])
        ):
            raise ValueError("Image metadata differs from its registered source")
        for question in _question_rows(source, image):
            expected_question_map[question["question_id"]] = question
    if ids != set(expected_question_map):
        raise ValueError("Question identities do not cover exactly all image/operation variants")
    for row in questions:
        expected = expected_question_map[row["question_id"]]
        token_fields = {"actual_model_tokenized_prompt_hash", "prompt_token_count"}
        if any(row.get(k) != value for k, value in expected.items() if k not in token_fields):
            raise ValueError("Question truth or metadata differs from its source/image")
        if (
            row["prompt"] != gen.make_prompt(row["operation"])
            or hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]
        ):
            raise ValueError("Visible prompt changed or contains sidecar content")
        if row["image_sha256"] != image_map[row["image_id"]]["image_sha256"]:
            raise ValueError("Question image identity mismatch")
    for row in images:
        if file_hash(relative_file(root, row["image_path"])) != row["image_sha256"]:
            raise ValueError("Original image bytes changed")
        if (
            row.get("processed_image_path")
            and file_hash(relative_file(root, row["processed_image_path"]))
            != row["processed_image_hash"]
        ):
            raise ValueError("Processed image bytes changed")
    for path in (plan_path.parent / "schedules").glob("*.jsonl"):
        if any(row["question_id"] not in ids for row in read_jsonl(path)):
            raise ValueError("A frozen schedule refers to an unmaterialized question")
    has_processor = report["processor_status"] == "ACTUAL_CPU_PROCESSOR_VERIFIED"
    if has_processor:
        validate_processed_manifests(root, questions, images, report)
    audit = _image_audit(images, history, processed=has_processor)
    for qa in report["qa"].values():
        for sheet in qa["contact_sheets"]:
            if file_hash(relative_file(root, sheet["path"])) != sheet["sha256"]:
                raise ValueError("QA contact sheet bytes changed")
    return {
        "status": "PASS",
        "questions": len(questions),
        "images": len(images),
        "image_audit": audit,
    }


def materialize_data(
    plan_path: Path, run_root: Path, *, historical_root: Path | None = None
) -> dict[str, Any]:
    import PIL

    root, plan_path = run_root.resolve(), plan_path.resolve()
    _mutable(root)
    if (root / "data").exists() and any((root / "data").iterdir()):
        raise FileExistsError("Existing data must be preserved; validate or preflight it")
    plan, sources, input_hashes = load_inputs(plan_path)
    history = load_history(historical_root)
    source_audit = validate_sources(sources, history)
    renderer = {"library": "Pillow", "version": PIL.__version__, "font": gen._font_identity()}
    if renderer != history["renderer"]:
        raise ValueError(
            "Renderer/Pillow/font differs from the audited renderer; do not silently rerender"
        )
    images, questions = [], []
    for index, source in enumerate(sources):
        rendered, rows = render_source(source, root)
        images.extend(rendered)
        questions.extend(rows)
        if (index + 1) % 64 == 0:
            print(
                json.dumps({"numeric_sources_rendered": index + 1, "images": len(images)}),
                flush=True,
            )
    image_audit = _image_audit(images, history)
    for name, rows in (
        ("sources", sources),
        ("images", images),
        ("questions", questions),
        ("sidecars", questions),
        ("model_inputs", [model_input(q) for q in questions]),
    ):
        write_jsonl(root / f"data/{name}.jsonl", rows)
    write_json(root / "data/history_exclusions.json", history)
    qa = qa_contact_sheets(root, images)
    report = {
        "plan_id": plan["plan_id"],
        "status": "RENDERED_PENDING_PROCESSOR",
        "processor_status": "NOT_EXECUTED",
        "visual_review_status": "READABILITY_REVIEW_REQUIRED",
        "input_sha256": input_hashes,
        "renderer": renderer,
        "code_sha256": {
            "src/mm_core/generator.py": file_hash(Path(gen.__file__)),
            "src/mm_dev/data.py": file_hash(Path(__file__)),
            "scripts/mm_dev/materialize_data.py": file_hash(
                Path(__file__).resolve().parents[2] / "scripts/mm_dev/materialize_data.py"
            ),
        },
        "protocol_hash": questions[0]["protocol_hash"],
        "root_counts": POOLS,
        "root_families": 292,
        "numeric_sources": len(sources),
        "images": len(images),
        "questions": len(questions),
        "panel_question_counts": dict(Counter(q["pool"] for q in questions)),
        "source_audit": source_audit,
        "image_audit": image_audit,
        "qa": {"original": qa},
        "model_input_manifest": "data/model_inputs.jsonl",
        "private_truth_files": [
            "data/questions.jsonl",
            "data/sidecars.jsonl",
            "data/sources.jsonl",
            "data/history_exclusions.json",
        ],
        "artifact_sha256": _artifact_hashes(root),
        "model_weights_loaded": False,
        "new_model_calls": 0,
        "new_training_runs": 0,
    }
    write_json(root / REPORT, report)
    report["validation"] = validate_materialization(plan_path, root)
    write_json(root / REPORT, report)
    return report


def reconstruct_rgb(pixel_values: Any, grid: list[list[int]], image_processor: Any) -> Any:
    """Invert native Qwen merged patches, checking duplicated still-image planes."""
    import numpy as np
    from PIL import Image

    if len(grid) != 1 or grid[0][0] != 1:
        raise ValueError("Exactly one still-image grid is required")
    _, height, width = grid[0]
    patch, merge, temporal = (
        int(getattr(image_processor, k))
        for k in ("patch_size", "merge_size", "temporal_patch_size")
    )
    if (patch, merge, temporal) != (16, 2, 2) or height % merge or width % merge:
        raise ValueError("Unregistered native Qwen3.5 patch geometry")
    array = (
        pixel_values.detach().cpu().numpy()
        if hasattr(pixel_values, "detach")
        else np.asarray(pixel_values)
    )
    if array.shape != (height * width, 3 * temporal * patch * patch):
        raise ValueError("Unexpected processor pixel shape")
    patches = array.reshape(
        height // merge, width // merge, merge, merge, 3, temporal, patch, patch
    )
    if not np.array_equal(patches[:, :, :, :, :, 0], patches[:, :, :, :, :, 1]):
        raise ValueError("Still-image temporal planes differ")
    chw = (
        patches[:, :, :, :, :, 0]
        .transpose(4, 0, 2, 5, 1, 3, 6)
        .reshape(3, height * patch, width * patch)
    )
    rgb = chw.transpose(1, 2, 0).astype(np.float64)
    if image_processor.do_normalize:
        rgb = rgb * np.asarray(image_processor.image_std) + np.asarray(image_processor.image_mean)
    if image_processor.do_rescale:
        rgb /= float(image_processor.rescale_factor)
    if not np.isfinite(rgb).all() or rgb.min() < -0.01 or rgb.max() > 255.01:
        raise ValueError("Reconstructed RGB values are outside the image domain")
    return Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8))


def verify_processor_identity(plan: dict, runtime: Any) -> dict:
    from mm_core.vl_runtime import hash_json

    identity = {
        **runtime.identity,
        "tokenizer_hash": hash_json(runtime.processor.tokenizer.get_vocab()),
    }
    for key in (
        "model_id",
        "processor_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "chat_template_kwargs",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
    ):
        if identity.get(key) != plan["model"][key]:
            raise ValueError(f"Processor identity differs from F2: {key}")
    if identity.get("model_type") != "qwen3_5" or identity.get("architecture_verified") is not True:
        raise ValueError("Unverified Qwen3.5 architecture")
    geometry = {
        key: int(getattr(runtime.processor.image_processor, key))
        for key in ("patch_size", "merge_size", "temporal_patch_size")
    }
    if tuple(geometry.values()) != (16, 2, 2):
        raise ValueError("Processor geometry changed")
    return {**identity, "processor_geometry": geometry}


def processor_preflight(plan_path: Path, run_root: Path, model_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    from mm_core.vl_runtime import QwenRuntime

    root = run_root.resolve()
    _mutable(root)
    validate_materialization(plan_path, root)
    report = json.loads((root / REPORT).read_text())
    if report["processor_status"] != "NOT_EXECUTED":
        raise FileExistsError("Completed CPU preflight must be preserved")
    if torch.cuda.is_initialized():
        raise RuntimeError("Processor preflight must not initialize CUDA")
    torch.set_num_threads(2)
    plan, _, _ = load_inputs(plan_path)
    runtime = QwenRuntime.processor_only(model_path)
    identity = verify_processor_identity(plan, runtime)
    questions = read_jsonl(root / "data/questions.jsonl")
    images = read_jsonl(root / "data/images.jsonl")
    image_map = {r["image_id"]: r for r in images}
    history = json.loads((root / "data/history_exclusions.json").read_text())
    processed, routing_rows = {}, []
    maximum_error = 0
    for index, row in enumerate(questions):
        prepared = runtime.prepare(model_input(row), root)
        routing = prepared["routing"]
        image_id = row["image_id"]
        if image_id not in processed:
            rgb = reconstruct_rgb(
                prepared["inputs"]["pixel_values"],
                routing["image_grid_thw"],
                runtime.processor.image_processor,
            )
            if rgb.size != gen.CANVAS or list(rgb.size) != routing["processed_size"]:
                raise ValueError("Actual processed resolution changed")
            with Image.open(root / row["image_path"]) as original:
                error = int(
                    np.abs(
                        np.asarray(rgb, dtype=np.int16)
                        - np.asarray(original.convert("RGB"), dtype=np.int16)
                    ).max()
                )
            if error > 1:
                raise ValueError("Processor inverse differs by more than one RGB quantization unit")
            maximum_error = max(maximum_error, error)
            path = f"data/processed_images/{image_id}.png"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            rgb.save(root / path)
            ys = [
                round(
                    (gen.PLOT[3] - v * (gen.PLOT[3] - gen.PLOT[1]) / 100)
                    * rgb.height
                    / gen.CANVAS[1]
                )
                for v in range(101)
            ]
            minimum_gap = min(abs(a - b) for a, b in pairwise(ys))
            if minimum_gap < 4 or routing["pixels_per_delta"] < 4:
                raise ValueError("Actual processor violates four pixels per delta")
            processed[image_id] = {
                "processed_image_path": path,
                "processed_image_hash": file_hash(root / path),
                "processed_pixel_sha256": routing["processed_pixel_sha256"],
                "processed_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "processed_size": list(rgb.size),
                "processed_pixels_per_delta": routing["pixels_per_delta"],
                "minimum_processed_raster_grid_gap_pixels": minimum_gap,
                "max_inverse_rgb_error": error,
                "processor_hash": identity["processor_hash"],
                "processor_status": "ACTUAL_CPU_PROCESSOR_VERIFIED",
                "readability_status": "CPU_PASS_PENDING_VISUAL",
            }
            image_map[image_id].update(processed[image_id])
        if processed[image_id]["processed_pixel_sha256"] != routing["processed_pixel_sha256"]:
            raise ValueError("Image processor output changed across operation prompts")
        if routing["chat_template_kwargs_hash"] != identity["chat_template_kwargs_hash"]:
            raise ValueError("Prepared template kwargs identity differs")
        row.update(processed[image_id])
        row.update(
            actual_model_tokenized_prompt_hash=routing["input_ids_sha256"],
            actual_chat_text_sha256=routing["chat_text_sha256"],
            prompt_token_count=routing["input_ids_shape"][1],
        )
        routing_rows.append({"question_id": row["question_id"], "image_id": image_id, **routing})
        if (index + 1) % 192 == 0:
            print(
                json.dumps({"questions_processed": index + 1, "images_processed": len(processed)}),
                flush=True,
            )
    if len(processed) != 2336 or len(routing_rows) != 7008 or torch.cuda.is_initialized():
        raise ValueError("Incomplete CPU-only processor coverage")
    audit = _image_audit(images, history, processed=True)
    qa = qa_contact_sheets(root, images, processed=True)
    for name, rows in (
        ("images", images),
        ("questions", questions),
        ("sidecars", questions),
        ("model_inputs", [model_input(q) for q in questions]),
        ("processor_routing", routing_rows),
        (
            "processed_images",
            [
                {
                    "image_id": r["image_id"],
                    "image_path": r["image_path"],
                    **processed[r["image_id"]],
                }
                for r in images
            ],
        ),
    ):
        write_jsonl(root / f"data/{name}.jsonl", rows)
    processor_class = type(runtime.processor.image_processor)
    environment = {
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "transformers", "Pillow", "numpy", "tokenizers")
        },
        "processor_class": f"{processor_class.__module__}.{processor_class.__qualname__}",
        "processor_preprocess_source_sha256": hashlib.sha256(
            inspect.getsource(type(runtime.processor.image_processor)._preprocess).encode()
        ).hexdigest(),
        "torch_cuda_build": torch.version.cuda,
        "cuda_context_initialized": False,
        "model_weights_loaded": False,
        "new_model_calls": 0,
    }
    report.update(
        status="CPU_PASS_PENDING_VISUAL",
        processor_status="ACTUAL_CPU_PROCESSOR_VERIFIED",
        visual_review_status="CPU_PASS_PENDING_VISUAL",
        image_audit=audit,
        processor_identity=identity,
        processor_environment=environment,
        maximum_inverse_rgb_error=maximum_error,
        processed_image_count=len(processed),
        processed_question_count=len(routing_rows),
        artifact_sha256=_artifact_hashes(root),
    )
    report["qa"]["processed"] = qa
    report["code_sha256"]["src/mm_core/vl_runtime.py"] = file_hash(
        Path(inspect.getfile(QwenRuntime))
    )
    report["processor_preflight_code_sha256"] = file_hash(Path(__file__))
    write_json(root / REPORT, report)
    report["validation"] = validate_materialization(plan_path, root)
    write_json(root / REPORT, report)
    return report
