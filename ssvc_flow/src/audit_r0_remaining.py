"""Complete the deterministic R0 checks that do not generate new model output.

The module is deliberately evidence producing: missing tokenizer or processor
inputs are reported as pending, never inferred as a pass.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .core import file_hash, write_json
from .prompts import build_prompt


def decode_raw_completion(token_ids, tokenizer, eos_ids=()):
    """Decode raw ids, removing only trailing true EOS ids.

    In particular, this does not use ``skip_special_tokens``: think text and
    malformed output remain visible to the parser audit.
    """
    ids = list(token_ids or [])
    eos = set(int(x) for x in eos_ids)
    end = len(ids)
    while end and ids[end - 1] in eos:
        end -= 1
    return tokenizer.decode(ids[:end], skip_special_tokens=False), ids[:end]


def audit_token_rows(rows, tokenizer, eos_ids=(248044, 248046)):
    results = []
    for row in rows:
        ids = row.get("token_ids")
        if not isinstance(ids, list):
            results.append({"sample_key": row.get("sample_key"), "status": "MISSING_TOKEN_IDS"})
            continue
        decoded, stripped = decode_raw_completion(ids, tokenizer, eos_ids)
        recorded = row.get("raw_completion")
        results.append(
            {
                "sample_key": row.get("sample_key"),
                "recorded_raw": recorded,
                "decoded_raw": decoded,
                "token_ids_without_trailing_eos": stripped,
                "exact_match": decoded == recorded if isinstance(recorded, str) else None,
                "status": "PASS" if isinstance(recorded, str) and decoded == recorded else "FAIL",
            }
        )
    return {
        "status": "PASS" if results and all(x.get("status") == "PASS" for x in results) else "FAIL",
        "checked": len(results),
        "passed": sum(x.get("status") == "PASS" for x in results),
        "failed": [x for x in results if x.get("status") == "FAIL"],
        "missing": [x for x in results if x.get("status") == "MISSING_TOKEN_IDS"],
    }


def audit_cross_split(dataset_root):
    root = Path(dataset_root)
    by_id = defaultdict(list)
    by_truth = defaultdict(list)
    manifests = {}
    for p in sorted(root.glob("*.jsonl")):
        if p.name.startswith(".") or p.name == "access.jsonl":
            continue
        split = p.stem
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        for row in rows:
            bid = row.get("base_scene_id")
            th = row.get("truth_structure_hash")
            if bid is not None:
                by_id[bid].append(split)
            if th is not None:
                by_truth[th].append(split)
        manifests[split] = {
            "rows": len(rows),
            "sha256": file_hash(p),
            "base_scene_count": len({x.get("base_scene_id") for x in rows}),
        }
    id_collisions = {k: sorted(set(v)) for k, v in by_id.items() if len(set(v)) > 1}
    truth_collisions = {k: sorted(set(v)) for k, v in by_truth.items() if len(set(v)) > 1}
    # The manifest's global uniqueness is itself a binding claim; do not silently
    # weaken it when a source manifest is present.
    manifest_path = root / "manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    expected_unique = None if manifest is None else manifest.get("global_truth_uniqueness")
    return {
        "status": "PASS"
        if not id_collisions and (not expected_unique or not truth_collisions)
        else "FAIL",
        "base_scene_collisions": id_collisions,
        "truth_structure_collisions": truth_collisions,
        "manifest_global_truth_uniqueness": expected_unique,
        "splits": manifests,
        "manifest_sha256": file_hash(manifest_path) if manifest_path.exists() else None,
    }


def select_36(dataset_root):
    """Select two lowest opaque IDs for each family x chart x operation cell."""
    root = Path(dataset_root)
    rows = [
        json.loads(x) for x in (root / "calibration.jsonl").read_text().splitlines() if x.strip()
    ]
    cells = defaultdict(list)
    for row in rows:
        cells[(row.get("constraint_family"), row.get("chart_type"), row.get("operation"))].append(
            row
        )
    expected = [
        (f, c, o)
        for f in ("duplicate_encoding", "cross_series", "trend")
        for c in ("grouped_bar", "line")
        for o in ("sum4", "difference_pairs", "range4")
    ]
    selected = []
    missing = []
    for cell in expected:
        values = sorted(cells[cell], key=lambda x: x["base_scene_id"])
        if len(values) < 2:
            missing.append(cell)
        selected.extend(values[:2])
    if len({r["base_scene_id"] for r in selected}) != 36:
        raise ValueError("36 selection has duplicate IDs")
    return selected, missing


def compare_contact_manifest(dataset_root, contact_manifest):
    selected, missing = select_36(dataset_root)
    cm = json.loads(Path(contact_manifest).read_text())
    listed = [r for sheet in cm.get("sheets", []) for r in sheet.get("scenes", [])]
    got = {r["base_scene_id"]: r for r in selected}
    old = {r["base_scene_id"]: r for r in listed}
    same = (
        not missing
        and len(listed) == 36
        and set(got) == set(old)
        and all(
            all(
                got[k].get(f) == old[k].get(f)
                for f in (
                    "image_hash",
                    "image_path",
                    "constraint_family",
                    "chart_type",
                    "operation",
                )
            )
            for k in got
        )
    )
    return {
        "status": "PASS" if same else "PENDING_HUMAN_REVIEW",
        "can_reuse_human_review": same,
        "selected_count": len(selected),
        "missing_cells": missing,
        "contact_manifest_sha256": file_hash(contact_manifest),
        "selected_scene_ids": [r["base_scene_id"] for r in selected],
        "mismatches": [
            k
            for k in set(got) | set(old)
            if k not in got
            or k not in old
            or any(
                got.get(k, {}).get(f) != old.get(k, {}).get(f)
                for f in (
                    "image_hash",
                    "image_path",
                    "constraint_family",
                    "chart_type",
                    "operation",
                )
            )
        ],
    }


def audit_images(dataset_root, selected):
    root = Path(dataset_root)
    out = []
    for s in selected:
        p = (root / s["image_path"]).resolve()
        item = {
            "base_scene_id": s["base_scene_id"],
            "image_path": s["image_path"],
            "declared_hash": s.get("image_hash"),
        }
        if not p.is_relative_to(root.resolve()) or not p.is_file():
            item.update(status="MISSING")
        else:
            from PIL import Image

            with Image.open(p) as im:
                item.update(
                    size=list(im.size),
                    mode=im.mode,
                    actual_hash=file_hash(p),
                    status="PASS"
                    if file_hash(p) == s.get("image_hash") and im.size == (768, 512)
                    else "FAIL",
                )
        out.append(item)
    return {"status": "PASS" if all(x["status"] == "PASS" for x in out) else "FAIL", "images": out}


def audit_symbolic_prompts(dataset_root):
    rows = [
        json.loads(x)
        for x in (Path(dataset_root) / "calibration.jsonl").read_text().splitlines()
        if x.strip()
    ]
    findings = []
    for scene in rows:
        p = build_prompt(scene, "SYMBOLIC_FRESH")
        text = p["system"] + "\n" + p["user"]
        # Values in observed/cues are declared inputs. Reject explicit truth,
        # image payloads, answer labels, or oracle changed-index annotations.
        forbidden = []
        if (
            scene["truth_world"] != scene["observed_world"]
            and json.dumps(scene["truth_world"], separators=(",", ":")) in text
        ):
            forbidden.append("truth_json")
        if scene.get("image_path") and scene["image_path"] in text:
            forbidden.append("image_path")
        if re.search(r"(?:changed[_ ]index|correct answer|oracle|truth world)\s*[:=]", text, re.I):
            forbidden.append("undeclared_label")
        findings.append(
            {
                "base_scene_id": scene["base_scene_id"],
                "status": "PASS" if not forbidden else "FAIL",
                "forbidden": forbidden,
                "prompt_hash": p["prompt_hash"],
            }
        )
    return {
        "status": "PASS" if all(x["status"] == "PASS" for x in findings) else "FAIL",
        "checked": len(findings),
        "failures": [x for x in findings if x["status"] == "FAIL"],
    }


def processor_audit(rows, processor, data_root, selected_ids):
    """Re-run processor on selected image prompts and compare P3 metadata."""
    from .model_adapters.qwen35 import Qwen35Adapter

    adapter = Qwen35Adapter(
        None, processor, "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    results = []
    for scene in rows:
        if scene.get("base_scene_id") not in selected_ids:
            continue
        prompt = build_prompt(scene, "IMAGE_CUE_FRESH")
        try:
            got = adapter.prepare(prompt, data_root)["audit"]
        except Exception as exc:
            results.append(
                {"base_scene_id": scene["base_scene_id"], "status": "ERROR", "error": repr(exc)}
            )
            continue
        results.append(
            {
                "base_scene_id": scene["base_scene_id"],
                "image_grid_thw": got.get("image_grid_thw"),
                "processor_width": got.get("processor_width"),
                "processor_height": got.get("processor_height"),
                "image_token_count": got.get("image_token_count"),
                "p3_image_grid_thw": scene.get("image_grid_thw"),
                "p3_processor_width": scene.get("processor_width"),
                "p3_processor_height": scene.get("processor_height"),
                "p3_image_token_count": scene.get("image_token_count"),
                "status": (
                    "MEASURED"
                    if scene.get("image_grid_thw") is None
                    and scene.get("processor_width") is None
                    and scene.get("processor_height") is None
                    else "PASS"
                    if got.get("image_grid_thw") == scene.get("image_grid_thw")
                    and got.get("processor_width") == scene.get("processor_width")
                    and got.get("processor_height") == scene.get("processor_height")
                    else "FAIL"
                ),
            }
        )
    return {
        "status": "PASS"
        if results and all(x["status"] in {"PASS", "MEASURED"} for x in results)
        else "PENDING_RUNTIME",
        "measured_only": bool(results) and all(x["status"] == "MEASURED" for x in results),
        "results": results,
    }


def run_audit(dataset_root, contact_manifest, out, tokenizer=None, processor=None):
    root = Path(dataset_root)
    selected, _ = select_36(root)
    result = {
        "phase": "R0",
        "checks": {},
        "selection": compare_contact_manifest(root, contact_manifest),
        "images": audit_images(root, selected),
        "cross_split": audit_cross_split(root),
        "symbolic_prompts": audit_symbolic_prompts(root),
    }
    if tokenizer is not None:
        rows = (
            [
                json.loads(x)
                for x in (root / "../runs/NEXT_20260909/raw_P3/N/samples.jsonl")
                .resolve()
                .read_text()
                .splitlines()
                if x.strip()
            ]
            if (root / "../runs/NEXT_20260909/raw_P3/N/samples.jsonl").exists()
            else []
        )
        result["tokenizer"] = audit_token_rows(rows, tokenizer)
    else:
        result["tokenizer"] = {"status": "PENDING_TOKENIZER_RUNTIME"}
    if processor is not None:
        result["processor"] = processor_audit(
            [
                json.loads(x)
                for x in (root / "calibration.jsonl").read_text().splitlines()
                if x.strip()
            ],
            processor,
            root,
            [x["base_scene_id"] for x in selected],
        )
    else:
        result["processor"] = {"status": "PENDING_PROCESSOR_RUNTIME"}
    result["status"] = (
        "PASS"
        if all(
            v.get("status") == "PASS"
            for v in result.values()
            if isinstance(v, dict) and "status" in v
        )
        else "INCONCLUSIVE"
    )
    write_json(Path(out) / "r0_remaining.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--contact-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--processor", type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    selected, _ = select_36(args.dataset)
    result = {
        "phase": "R0",
        "selection": compare_contact_manifest(args.dataset, args.contact_manifest),
        "images": audit_images(args.dataset, selected),
        "cross_split": audit_cross_split(args.dataset),
        "symbolic_prompts": audit_symbolic_prompts(args.dataset),
    }
    if args.samples and args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=False
        )
        rows = [json.loads(x) for x in args.samples.read_text().splitlines() if x.strip()]
        result["tokenizer"] = audit_token_rows(rows, tok)
    else:
        result["tokenizer"] = {"status": "PENDING_TOKENIZER_RUNTIME"}
    result["status"] = (
        "PASS"
        if all(
            x.get("status") == "PASS"
            for x in result.values()
            if isinstance(x, dict) and "status" in x
        )
        else "INCONCLUSIVE"
    )
    write_json(args.out / "r0_remaining.json", result)
    (args.out / "report_zh.md").write_text(
        "# R0 剩余审计\n\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    main()
