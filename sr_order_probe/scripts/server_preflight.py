#!/usr/bin/env python3
"""Finite, read-only CPU checks. No model loading, generation, or authorization.

Native 8-category scoring fixtures exercise the unchanged production scorer.
The new 10-category probe uses its frozen scorer; native J is not redefined.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from order_probe.common import digest, file_sha, read_jsonl, write_json
from order_probe.data import oracle_sort, world_key
from order_probe.inputs import assistant_prefill
from order_probe.measurement import measure


def image_selection(inputs):
    """All smoke views, first two stored MAIN roots per registered structure."""
    chosen = []
    for split in ("SMOKE", "MAIN"):
        counts = Counter()
        roots = json.loads((Path(inputs) / f"{split}_ROOTS_PRIVATE.json").read_text())
        for root in roots:
            if split == "MAIN" and counts[root["structure"]] >= 2:
                continue
            counts[root["structure"]] += 1
            chosen.extend((split, task["qid"]) for task in root["views"])
    if len(chosen) != 24 or len(set(chosen)) != 24:
        raise ValueError("Expected exactly 8 smoke and 16 MAIN images")
    return chosen


def reconstruct_pixels(pixels, grid, image_processor):
    """Invert Qwen's merged-patch layout, temporal repeat and normalization.

    Returned image derives from the actual processor pixel tensor, never a
    separate approximation of its resize. Preserve the raw tensor hash too.
    """
    import numpy as np

    data = np.asarray(pixels)
    t, h, w = map(int, grid)
    patch = int(image_processor.patch_size)
    merge = int(image_processor.merge_size)
    temporal = int(image_processor.temporal_patch_size)
    channels = 3
    if t != 1 or h % merge or w % merge:
        raise ValueError("Expected one still image with complete merged patches")
    if data.shape != (t * h * w, channels * temporal * patch * patch):
        raise ValueError("Unknown Qwen pixel tensor shape")
    frames = data.reshape(t, h // merge, w // merge, merge, merge,
                          channels, temporal, patch, patch)
    frames = frames.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)
    frames = frames.reshape(t * temporal, channels, h * patch, w * patch)
    if not all(np.array_equal(frames[0], frame) for frame in frames[1:]):
        raise ValueError("A still image unexpectedly contains distinct temporal frames")
    image = frames[0].transpose(1, 2, 0).astype(np.float64)
    if getattr(image_processor, "do_normalize", True):
        image = image * np.asarray(image_processor.image_std) + np.asarray(image_processor.image_mean)
    if getattr(image_processor, "do_rescale", True):
        image = image / float(image_processor.rescale_factor)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _history_rows(path):
    path = Path(path)
    # Check before opening any content; never traverse or discover sealed data.
    if any(part.upper().startswith(("TEST", "FORMAL")) for part in path.parts):
        raise PermissionError("Formal/test sources are outside this preflight")
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):
        rows = rows.get("tasks", rows.get("rows"))
    if not isinstance(rows, list):
        raise ValueError("History must be JSONL, a task list, or {'tasks': [...]} index")
    return rows


def history_check(inputs, histories):
    named, positional, sources = set(), set(), []
    for path in histories:
        path = Path(path)
        rows = _history_rows(path)
        if not rows:
            raise ValueError("Empty historical task index: " + str(path))
        for row in rows:
            if "world" in row and "query" in row:
                named.add(world_key(row["world"], row["query"]))
                positional.add(world_key(row["world"], row["query"], strip_names=True))
            elif all(k in row for k in ("world_query_key", "unlabeled_world_query_key")):
                for key in ("world_query_key", "unlabeled_world_query_key"):
                    value = row[key]
                    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                        raise ValueError("Invalid sealed exclusion fingerprint")
                named.add(row["world_query_key"])
                positional.add(row["unlabeled_world_query_key"])
            else:
                raise ValueError("Unsupported historical row; do not silently skip: " + str(path))
        sources.append(dict(path=str(path.resolve()), sha256=file_sha(path), rows=len(rows)))
    matches = []
    count = 0
    for split in ("SMOKE", "MAIN"):
        for task in read_jsonl(Path(inputs) / f"{split}_TASKS_PRIVATE.jsonl"):
            count += 1
            exact_match = world_key(task["world"], task["query"]) in named
            positional_match = world_key(task["world"], task["query"], strip_names=True) in positional
            if exact_match or positional_match:
                matches.append(dict(qid=task["qid"], exact=exact_match, positional=positional_match))
    return dict(status="FAIL_OVERLAP" if matches else ("PASS_RECORDED_SCOPE" if sources else "NOT_RUN"),
                new_views=count, historical_sources=sources, exact_unique=len(named),
                positional_unique=len(positional), matches=matches,
                scope="Only explicitly named previously-used train/development indices; no unseen or sealed test coverage claim")


def scoring_fixtures(native_code_root):
    sys.path.insert(0, str(Path(native_code_root) / "src"))
    native = importlib.import_module("sr_f6.measurement")
    if Path(native.__file__).resolve() != (Path(native_code_root) / "src/sr_f6/measurement.py").resolve():
        raise RuntimeError("Native scorer imported from an unexpected deployment")
    world = dict(categories=["Book", "Bowl", "Chair", "Clock", "Desk", "Frame", "Lamp", "Mug"],
                 series=dict(Alpha=[72, 66, 48, 48, 30, 30, 18, 12],
                             Beta=[12, 18, 24, 30, 36, 42, 48, 54]), domain=[10, 90])
    query = dict(family="TOPK", selection_series="Alpha", value="Beta", k=3,
                 direction="largest", aggregate="sum")
    task = dict(world=world, query=query, gold=oracle_sort(world, query),
                boundary_indices=[2, 3], required_from_boundary=1,
                strictly_better_indices=[0, 1], placebo_indices=[4, 5],
                enum_permutation=[3, 2, 0, 1, 4, 5, 6, 7])
    obj = dict(evidence=[dict(series=s, category=name, value=world["series"][s][i])
                         for s in ("Alpha", "Beta") for i, name in enumerate(world["categories"])],
               selected=task["gold"]["selected"],
               bindings=[dict(category=n, operand=v) for n, v in zip(task["gold"]["selected"], task["gold"]["operands"])],
               aggregate="sum", answer=task["gold"]["answer"])
    dumps = lambda value: json.dumps(value, separators=(",", ":"))
    base = dumps(obj)
    continuation = dict(obj)
    continuation.pop("evidence")
    fixtures = [("correct", base, "balanced"),
                ("true_prefix", assistant_prefill(task, "P_STABLE") + dumps(continuation)[1:], "balanced"),
                ("incomplete", base, "length"),
                ("truncated", base[:-2], "length"),
                ("duplicate_key", base[:-1] + ',"answer":"54"}', "balanced"),
                ("prefix_duplicate_evidence", assistant_prefill(task, "P_STABLE") + base[1:], "balanced"),
                ("two_objects", base + base, "balanced"),
                ("markdown", "```json\n" + base + "\n```", "eos")]
    mutations = [("wrong_selection", {"selected": ["Book", "Bowl", "Clock"]}),
                 ("selection_unknown", {"selected": "Book"}),
                 ("duplicate_selection", {"selected": ["Book", "Book", "Chair"]}),
                 ("wrong_answer", {"answer": "999"}),
                 ("answer_not_gated", {"selected": "invalid", "answer": task["gold"]["answer"]}),
                 ("work_cannot_repair", {"answer": "999", "work": "Actually the answer is " + task["gold"]["answer"]}),
                 ("wrong_binding", {"bindings": [dict(category="Book", operand=999)]}),
                 ("binding_unknown", {"bindings": "bad"}),
                 ("nonfinite", {"answer": float("nan")})]
    fixtures.extend((name, dumps(dict(copy.deepcopy(obj), **change)), "balanced") for name, change in mutations)
    results = []
    for name, text, finish in fixtures:
        old = native.score_response(text, world, query, finished=finish != "length", finish_reason=finish)
        new = measure(text, task, "C0", finish)
        differences = {k: dict(native=old[k], probe=new[k]) for k in ("A", "S", "B") if old[k] != new[k]}
        results.append(dict(fixture=name, text=text, finish_reason=finish,
                            native={k: old[k] for k in ("A", "S", "B", "V", "J")},
                            probe={k: new[k] for k in ("A", "S", "B", "V_probe")},
                            differences=differences))
    return dict(status="PASS_FIXED_FIXTURES" if not any(r["differences"] for r in results) else "FAIL_SCORER_MISMATCH",
                native_source=str(Path(native.__file__).resolve()), native_sha256=file_sha(native.__file__),
                fixtures=results, compared_fields=["A", "S", "B"],
                native_J_for_ten_categories="NOT_APPLICABLE_OLD_WORLD_SCHEMA_MAXIMUM_8",
                scope="8-category fixed fixtures validate shared extraction/scoring semantics. Frozen probe scorer supports the new 10-category worlds. V_probe and order diagnostics never replace native J.")


def processor_check(inputs, model_path, native_code_root, out):
    from PIL import Image
    sys.path.insert(0, str(Path(native_code_root) / "src"))
    sys.path.insert(0, str(ROOT))
    from sr_f5.runtime import Runtime as FP32Runtime
    from backend.native_adapter import prepare_public_input

    runtime = FP32Runtime.processor_only(model_path, protocol_revision="v2")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    requests = {}
    for split in ("SMOKE", "MAIN"):
        for req in read_jsonl(Path(inputs) / f"{split}_REQUESTS_PUBLIC.jsonl"):
            if req["condition"] == "C0":
                requests.setdefault((split, req["qid"]), req)
    records = []
    for index, key in enumerate(image_selection(inputs)):
        req = requests[key]
        prepared = prepare_public_input(runtime, req, inputs)
        pixels = prepared["inputs"]["pixel_values"].detach().cpu().numpy()
        grid = prepared["inputs"]["image_grid_thw"].tolist()[0]
        preview = reconstruct_pixels(pixels, grid, runtime.processor.image_processor)
        target = out / f"{index:02d}_processed.png"
        Image.fromarray(preview).save(target)
        source = Path(inputs) / req["image_file"]
        source_copy = out / f"{index:02d}_source.png"
        shutil.copyfile(source, source_copy)
        records.append(dict(index=index, split=key[0], qid=key[1], source_path=str(source.resolve()),
                            source_sha256=file_sha(source), source_copy=str(source_copy.resolve()),
                            processed_path=str(target.resolve()),
                            processed_png_sha256=file_sha(target),
                            actual_pixel_tensor_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                            routing=prepared["routing"]))
    return dict(status="PROCESSED_IMAGES_READY_FOR_VISUAL_REVIEW", visual_review="NOT_RUN",
                method="Inverse actual Qwen merged patch tensor, denormalized and rounded to RGB uint8; no OCR",
                model_path=str(model_path), runtime_identity=runtime.identity,
                source_prepare_input_sha256=hashlib.sha256(__import__("inspect").getsource(prepare_public_input).encode()).hexdigest(),
                images=records, image_count=len(records), model_weight_loads=0, model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--native-code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--history", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    history = history_check(args.inputs, args.history)
    write_json(args.out / "HISTORY_EXCLUSION.json", history)
    scorer = scoring_fixtures(args.native_code_root)
    write_json(args.out / "SCORER_FIXTURES.json", scorer)
    if history["matches"] or scorer["status"].startswith("FAIL"):
        raise SystemExit("Preflight failed: preserve receipts and resolve before generation")
    if args.model_path:
        processor = processor_check(args.inputs, args.model_path, args.native_code_root, args.out / "PROCESSOR_IMAGES")
    else:
        processor = dict(status="NOT_RUN", reason="No model path supplied")
    write_json(args.out / "PROCESSOR_CHECK.json", processor)
    write_json(args.out / "PREFLIGHT_CPU.json", dict(status="CPU_CHECKS_COMPLETE_NOT_GPU_GATE",
               history_status=history["status"], scorer_status=scorer["status"], processor_status=processor["status"],
               visual_review="NOT_RUN", gpu_smoke="NOT_RUN", optimizer_updates=0, model_calls=0))
    print(json.dumps(dict(history=history["status"], scorer=scorer["status"], processor=processor["status"])))


if __name__ == "__main__":
    main()
