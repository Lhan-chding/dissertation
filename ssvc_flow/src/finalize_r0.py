"""Close R0 with immutable prior audits and fresh, model-free input replay.

The calibration image panel and historical dev inputs are different populations.
This runner checks both and writes a new gate; it never edits earlier evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from .audit_r0_remaining import (
    audit_cross_split,
    audit_images,
    audit_symbolic_prompts,
    audit_token_rows,
    compare_contact_manifest,
    processor_audit,
    select_36,
)
from .core import canonical_hash, file_hash, write_json
from .legacy_frozen import build_legacy_prompt, load_legacy_scenes
from .next_stage_common import bind_run, stage_status
from .prompts import build_prompt

INPUT_FIELDS = (
    "final_prompt",
    "final_prompt_token_ids",
    "final_prompt_hash",
    "tokenized_prompt_hash",
    "input_tensor_hash",
    "pixel_values_hash",
    "original_image_size",
    "image_grid_thw",
    "processor_height",
    "processor_width",
    "image_token_count",
    "prompt_token_count",
    "enable_thinking",
    "thinking_template_changed",
)
REMAINING = {
    "raw token-to-text decoding audit with pinned tokenizer",
    "36 hash-selected image/processor spot checks",
    "cross-split leakage recheck",
}
REQUIRED_BASE = {
    "identity.json",
    "baseline_lock.json",
    "audit_findings.json",
    "audit_rollouts.jsonl",
    "data_solver_checks.json",
    "recomputed_metrics.json",
    "paired_CI.json",
    "per_prompt_counts.parquet",
    "group_support_K.json",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_manifest(root, required=()):
    root = Path(root).resolve()
    files = {}
    for item in read_json(root / "manifest.json")["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or item["path"] in files:
            raise ValueError("missing, escaping or duplicate manifest artifact")
        digest = file_hash(path)
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"manifest content mismatch: {path}")
        files[item["path"]] = digest
    if not set(required).issubset(files):
        raise ValueError("manifest is missing required artifacts")
    return files


def verify_dataset_manifest(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    sizes = manifest["split_sizes"]
    required = {"train", "control", "calibration", "dev", "confirm", "ood", "natural_pool"}
    if not required.issubset(sizes) or not {f"{s}.jsonl" for s in sizes}.issubset(
        manifest["files"]
    ):
        raise ValueError("dataset manifest is missing a required split")
    checked = {}
    for name, item in manifest["files"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_hash(path) != item["sha256"]
        ):
            raise ValueError(f"missing or changed dataset artifact: {name}")
        checked[name] = dict(item)
        if path.suffix == ".jsonl" and path.stem in sizes:
            with path.open() as stream:
                count = sum(bool(line.strip()) for line in stream)
            if type(sizes[path.stem]) is not int or count != sizes[path.stem]:
                raise ValueError(f"dataset split row count mismatch: {path.stem}")
            checked[name]["rows"] = count
    return {
        "status": "PASS",
        "manifest_sha256": file_hash(root / "manifest.json"),
        "files": checked,
        "split_sizes": sizes,
    }


def verify_base_audit(base, source, scenes, track):
    base, source = Path(base), Path(source)
    audit_files = verify_manifest(base, REQUIRED_BASE)
    raw_files = verify_manifest(
        source, ("samples.jsonl", "runtime_lock.json", "data_manifest.json")
    )
    identity = read_json(base / "identity.json")
    lock = read_json(base / "baseline_lock.json")
    runtime = read_json(source / "runtime_lock.json")
    if (
        identity["track"] != track
        or identity["raw_files"] != raw_files
        or identity["raw_manifest_hash"] != file_hash(source / "manifest.json")
        or identity["data_hash"] != canonical_hash(scenes)
        or read_json(source / "data_manifest.json")["data_hash"] != canonical_hash(scenes)
    ):
        raise ValueError("prior audit no longer binds these P3 files and scenes")
    bindings = {
        "model": "model_spec",
        "generation": "generation_protocol",
        "actual_max_new_tokens": "max_new_tokens",
        "model_audit": "model_audit",
        "source_commit": "source_commit",
        "source_files": "source_files",
        "runtime_versions": "versions",
    }
    if any(lock.get(a) != runtime.get(b) or runtime.get(b) is None for a, b in bindings.items()):
        raise ValueError("baseline lock does not bind actual P3 runtime")
    if (
        lock["model"]["id"] != "Qwen/Qwen3.5-9B"
        or lock["model"]["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        or lock.get("inference_pass") is not True
    ):
        raise ValueError("unqualified historical model lock")
    findings = read_json(base / "audit_findings.json")
    pending = {x["check"] for x in findings if x["status"] != "PASS"}
    required_pass = {
        "raw_manifest_and_record_hashes",
        "sampled_greedy_counts_and_indices",
        "classification_reproduction",
    }
    if pending != REMAINING or not required_pass.issubset(
        {x["check"] for x in findings if x["status"] == "PASS"}
    ):
        raise ValueError("prior audit has unresolved findings outside the supplement scope")
    raw = read_rows(source / "samples.jsonl")
    keys, groups = set(), defaultdict(list)
    for row in raw:
        if row["sample_key"] in keys or row.get("track") != track:
            raise ValueError("duplicate sample key or mixed tracks")
        keys.add(row["sample_key"])
        if row.get("record_hash") != canonical_hash(
            {k: v for k, v in row.items() if k != "record_hash"}
        ):
            raise ValueError("raw record hash mismatch")
        groups[row["prompt_id"], row["decode_mode"]].append(row["rollout_index"])
    counts = Counter(x["decode_mode"] for x in raw)
    expected = {"N": {"sample": 4608, "greedy": 288}, "L": {"sample": 2816, "greedy": 176}}[track]
    if counts != expected or any(
        sorted(v) != list(range(16 if k[1] == "sample" else 1)) for k, v in groups.items()
    ):
        raise ValueError("incomplete original P3 bank")
    metrics = read_json(base / "recomputed_metrics.json")
    if metrics["category_disagreements"] != 0:
        raise ValueError("unresolved classification disagreements")
    if track == "N":
        checks = read_json(base / "data_solver_checks.json")["checks"]
        if (
            {x["base_scene_id"] for x in checks} != {x["base_scene_id"] for x in scenes}
            or len(checks) != len(scenes)
            or not all(x["passed"] for x in checks)
        ):
            raise ValueError("independent solver evidence does not cover the dev scenes")
    return (
        {
            "status": "PASS",
            "track": track,
            "audit_root": str(base),
            "source_root": str(source),
            "audit_manifest_sha256": file_hash(base / "manifest.json"),
            "audit_files": audit_files,
            "raw_manifest_sha256": file_hash(source / "manifest.json"),
            "raw_files": raw_files,
            "data_hash": canonical_hash(scenes),
            "counts": dict(counts),
            "old_status_preserved": read_json(base / "status.json")["status"],
        },
        raw,
        lock,
        runtime,
    )


def replay_historical_inputs(scenes, rows, adapter, data_root, track):
    scene_map = (
        {(s["base_scene_id"], i): s for s in scenes for i in ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")}
        if track == "N"
        else {(s["base_scene_id"], s["interface"]): s for s in scenes}
    )
    measurements, failures = {}, []
    for row in rows:
        key = (row["base_scene_id"], row["interface"])
        if key not in scene_map:
            failures.append({"sample_key": row["sample_key"], "mismatch_fields": ["scene_key"]})
            continue
        if key not in measurements:
            scene = scene_map[key]
            prompt = build_prompt(scene, key[1]) if track == "N" else build_legacy_prompt(scene)
            measurements[key] = adapter.prepare(prompt, data_root)["audit"]
        got = measurements[key]
        differences = [k for k in INPUT_FIELDS if k not in row or k not in got or got[k] != row[k]]
        if differences:
            failures.append({"sample_key": row["sample_key"], "mismatch_fields": differences})
    return {
        "status": "PASS" if rows and not failures else "FAIL",
        "checked_records": len(rows),
        "checked_prompts": len(measurements),
        "failures": failures,
        "fields": list(INPUT_FIELDS),
        "measurements": [
            {
                "base_scene_id": k[0],
                "interface": k[1],
                **{
                    f: v.get(f)
                    for f in INPUT_FIELDS
                    if f not in {"final_prompt", "final_prompt_token_ids"}
                },
            }
            for k, v in sorted(measurements.items())
        ],
    }


def load_processor(snapshot, lock):
    from transformers import AutoProcessor

    from .model_adapters.base import _hash_json
    from .model_adapters.qwen35 import Qwen35Adapter

    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    audit = lock["model_audit"]
    ip = processor.image_processor
    pixels = audit["processor_max_pixels"]
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = pixels
    if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
        ip.size = {**ip.size, "longest_edge": pixels}
    elif dataclasses.is_dataclass(getattr(ip, "size", None)):
        ip.size = dataclasses.replace(ip.size, longest_edge=pixels)
    hashes = {
        "processor_hash": _hash_json(processor.to_dict()),
        "tokenizer_hash": _hash_json(processor.tokenizer.get_vocab()),
        "chat_template_hash": _hash_json(processor.chat_template),
    }
    if any(audit.get(k) != v for k, v in hashes.items()):
        raise ValueError("pinned processor/tokenizer/template hashes disagree with P3 lock")
    adapter = Qwen35Adapter(
        SimpleNamespace(config=SimpleNamespace(image_token_id=248056)),
        processor,
        lock["model"]["id"],
        lock["model"]["revision"],
        device="cpu",
    )
    adapter.audit = dict(audit)
    return adapter, hashes


def human_review_binding(selection, review_path, runtime):
    review = read_json(review_path)
    evidence = runtime["P1_evidence"]["human_review"]
    digest = file_hash(review_path)
    checked = review.get("checked_scene_ids", [])
    passed = (
        selection["can_reuse_human_review"]
        and review.get("status") == "PASS"
        and evidence.get("status") == "PASS"
        and digest == evidence.get("review_sha256")
        and len(checked) == 36
        and len(set(checked)) == 36
        and set(checked) == set(selection["selected_scene_ids"])
        and review.get("contact_manifest_sha256") == selection["contact_manifest_sha256"]
        and evidence.get("contact_manifest_sha256") == selection["contact_manifest_sha256"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "review_sha256": digest,
        "contact_manifest_sha256": selection["contact_manifest_sha256"],
        "checked_scene_count": len(checked),
        "reviewer": review.get("reviewer"),
    }


def close_r0(args):
    out = args.out.resolve()
    inputs = [
        args.dataset,
        args.legacy_dataset,
        args.base_n,
        args.base_l,
        args.p3_n,
        args.p3_l,
        args.snapshot,
    ]
    if any(out == p.resolve() or out.is_relative_to(p.resolve()) for p in inputs):
        raise ValueError("new closure output must be outside original data and evidence")
    identity = {
        "phase": "R0",
        "inputs": {
            str(p): file_hash(p / "manifest.json") if (p / "manifest.json").exists() else None
            for p in inputs
        },
        "human_review_sha256": file_hash(args.human_review)
        if args.human_review.is_file()
        else None,
        "contact_manifest_sha256": file_hash(args.contact_manifest)
        if args.contact_manifest.is_file()
        else None,
    }
    bind_run(out, identity)
    checks = {}
    try:
        checks["dataset_manifest"] = verify_dataset_manifest(args.dataset)
        selected, _ = select_36(args.dataset)
        checks["selection"] = compare_contact_manifest(args.dataset, args.contact_manifest)
        checks["images"] = audit_images(args.dataset, selected)
        checks["cross_split"] = audit_cross_split(args.dataset)
        checks["symbolic_prompts"] = audit_symbolic_prompts(args.dataset)
        calibration = read_rows(args.dataset / "calibration.jsonl")
        for track, base, source, root in [
            ("N", args.base_n, args.p3_n, args.dataset),
            ("L", args.base_l, args.p3_l, args.legacy_dataset),
        ]:
            scenes = (
                read_rows(root / "dev.jsonl")
                if track == "N"
                else load_legacy_scenes(
                    root / "reward_fibers.jsonl", root / "reward_fibers_manifest.json"
                )
            )
            checks[f"base_{track}"], rows, lock, runtime = verify_base_audit(
                base, source, scenes, track
            )
            adapter, hashes = load_processor(args.snapshot, lock)
            write_json(out / f"baseline_lock_{track}.json", lock)
            checks[f"processor_lock_{track}"] = {"status": "PASS", **hashes}
            checks[f"token_decode_{track}"] = audit_token_rows(
                rows, adapter.processor.tokenizer, lock["model_audit"]["eos_token_ids"]
            )
            checks[f"inputs_{track}"] = replay_historical_inputs(scenes, rows, adapter, root, track)
            if track == "N":
                checks["human_review"] = human_review_binding(
                    checks["selection"], args.human_review, runtime
                )
                checks["calibration_processor"] = processor_audit(
                    calibration, adapter.processor, root, [x["base_scene_id"] for x in selected]
                )
        status = "PASS" if all(v["status"] == "PASS" for v in checks.values()) else "BLOCKED"
    except Exception as exc:
        checks["exception"] = {"status": "BLOCKED", "type": type(exc).__name__, "reason": str(exc)}
        status = "BLOCKED"
    write_json(out / "closure_checks.json", checks)
    details = {
        "R0_gate_passed": status == "PASS",
        "generation_count": 0,
        "prior_artifacts_modified": False,
        "checks": {k: v["status"] for k, v in checks.items()},
        "scope": "N primary audit; L legacy-derived frozen diagnostic, not exact C3 reproduction",
    }
    return stage_status(out, "R0", status, "CPU_AUDIT", details, sorted(out.glob("*.json")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "legacy-dataset",
        "base-n",
        "base-l",
        "p3-n",
        "p3-l",
        "snapshot",
        "contact-manifest",
        "human-review",
        "out",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    result = close_r0(parser.parse_args(argv))
    print(json.dumps({"status": result["status"], **result["details"]}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
