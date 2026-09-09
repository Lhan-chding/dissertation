"""R0 read-only audit of frozen P3 artifacts.

The auditor follows manifest paths and refuses to reconstruct raw rollouts from
summary tables.  Missing server paths therefore produce an explicit BLOCKED
stage record while still writing a useful protocol lock and findings report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .constraint_solver import solve
from .core import canonical_hash, write_json
from .group_support_audit import audit_categories
from .next_stage_common import dry_run_plan, load_yaml, stage_status
from .statistics_scene_cluster import scene_cluster_intervals, weighted_prompt_metrics
from .verifiers import annotate, strict_parse


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reclassify_row(row, track="N"):
    raw = row.get("raw_completion", row.get("raw_text", row.get("completion", "")))
    truth = row.get("truth_world")
    operation = row.get("operation")
    cue = row.get("cue")
    if track == "L":
        from .legacy_frozen import annotate_legacy

        annotation = annotate_legacy(
            raw,
            {
                "truth": row["truth"],
                "observation": row["observation"],
                "operation": row["operation"],
            },
        )
        truth, _observed = row["truth"], row["observation"]
        return {
            **row,
            "reclassified_category": annotation["category"],
            "reclassified_parsed_world": annotation["parsed_world"],
            "reclassified_syntax_valid": annotation["semantic_parse_success"],
            "reclassified_constraint_satisfaction": None,
            "feasible_repair_count": None,
            "parsed_in_unique_repair_set": None,
            "malformed_type": None
            if annotation["semantic_parse_success"]
            else "legacy_parser_invalid",
            "hamming_to_observed": None,
            "hamming_to_truth": None,
            "copy_observation": annotation["copy_observation"],
        }
    if truth is None or operation is None:
        raise ValueError("raw row lacks truth_world or operation; cannot audit parser")
    parsed = strict_parse(raw)
    annotation = annotate(raw, truth, operation, cue)
    malformed = None
    if parsed is None:
        if not raw:
            malformed = "empty"
        elif raw.lstrip().startswith("``") or not raw.strip().startswith("["):
            malformed = "extra_text_or_thinking"
        elif raw.count("[") > raw.count("]"):
            malformed = "truncated_or_unclosed"
        else:
            malformed = "invalid_json_or_domain"
    candidates = solve(row["observed_world"], cue) if cue is not None else []
    parsed_in_candidates = parsed in candidates if parsed is not None else False
    return {
        **row,
        "reclassified_category": annotation["category"],
        "reclassified_parsed_world": parsed,
        "reclassified_syntax_valid": parsed is not None,
        "reclassified_constraint_satisfaction": annotation["constraint_satisfaction"],
        "feasible_repair_count": len(candidates),
        "parsed_in_unique_repair_set": parsed_in_candidates,
        "malformed_type": malformed,
        "hamming_to_observed": sum(
            a != b for a, b in zip(parsed, row["observed_world"], strict=False)
        )
        if parsed is not None
        else None,
        "hamming_to_truth": sum(a != b for a, b in zip(parsed, truth, strict=False))
        if parsed is not None
        else None,
        "copy_observation": parsed == row["observed_world"] if parsed is not None else False,
    }


def audit_rollouts(rows, expected_per_prompt=16, track="N"):
    audited = [reclassify_row(row, track) for row in rows]
    by_prompt = defaultdict(list)
    for row in audited:
        by_prompt[(row["prompt_id"], row.get("decode_mode", "sample"))].append(row)
    counts = []
    for (prompt, mode), group in sorted(by_prompt.items()):
        c = Counter(r["reclassified_category"] for r in group)
        counts.append(
            {
                "prompt_id": prompt,
                "decode_mode": mode,
                "base_scene_id": group[0].get("base_scene_id"),
                "family": group[0].get("constraint_family", group[0].get("family")),
                "interface": group[0].get("interface"),
                "n_X": c["X"],
                "n_S": c["S"],
                "n_W": c["W"],
                "n_I": c["I"],
                "n_total": len(group),
                "n_valid": len(group) - c["I"],
                "expected_per_prompt": expected_per_prompt if mode == "sample" else 1,
                "complete": len(group) == (expected_per_prompt if mode == "sample" else 1),
                "feasible_repair_count": group[0]["feasible_repair_count"],
            }
        )
    metrics = weighted_prompt_metrics(
        [
            {
                "prompt_id": r["prompt_id"],
                "category": r["reclassified_category"],
                "base_scene_id": r.get("base_scene_id", r["prompt_id"]),
            }
            for r in audited
            if r.get("decode_mode", "sample") == "sample"
        ]
    )
    return audited, counts, metrics


def _write_counts(path, counts):
    path = Path(path)
    fields = sorted({key for row in counts for key in row}) if counts else ["prompt_id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(counts)


def build_baseline_lock(config, track):
    model = config.get("model", {})
    generation = config.get("generation_proposed_N", {}) if track == "N" else None
    return {
        "protocol_version": config.get("protocol_version"),
        "track": track,
        "model_id": model.get("id"),
        "model_revision": model.get("revision"),
        "generation": generation,
        "parser": "strict_parse from raw completion; unresolved until raw artifacts are present",
        "verifier": "executor + independent constraint_solver",
        "prompt_weights": config.get("statistics", {}).get("prompt_weights"),
        "unknown_required_fields": [
            "dataset_manifest",
            "raw_completion",
            "token_ids",
            "actual_generation_config",
        ],
        "lock_status": "PARTIAL_UNTIL_SERVER_ARTIFACTS",
        "config_hash": canonical_hash(config),
    }


def audit_phase(
    config, out, rows_path=None, read_only=True, data_root=None, track="N", resume=False
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .core import PROJECT_ROOT, file_hash
    from .next_stage_common import bind_run

    out = Path(out).resolve()
    source = Path(rows_path).resolve().parent if rows_path else None
    if source and (out == source or out.is_relative_to(source)):
        raise ValueError("new output must be outside old P3 directory")
    identities = {
        "config_hash": canonical_hash(config),
        "track": track,
        "source_files": {
            str(p.relative_to(PROJECT_ROOT)): file_hash(p)
            for p in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        },
    }
    raw_rows, scenes = [], []
    checked_files, findings = {}, []
    if rows_path:
        manifest = json.loads((source / "manifest.json").read_text())
        for entry in manifest["files"]:
            path = (source / entry["path"]).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise ValueError("missing or escaping P3 manifest artifact")
            digest = file_hash(path)
            if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise ValueError("P3 manifest content mismatch: " + str(path))
            checked_files[entry["path"]] = digest
        if checked_files.get(Path(rows_path).name) != file_hash(rows_path):
            raise ValueError("ledger must be bound by P3 manifest")
        lock = json.loads((source / "runtime_lock.json").read_text())
        if (
            lock["model_spec"]["id"] != config["model"]["id"]
            or lock["model_spec"]["revision"] != config["model"]["revision"]
        ):
            raise ValueError("P3 model/revision mismatch")
        if track == "N":
            scenes = _rows(Path(data_root) / "dev.jsonl")
            scene_by_id = {
                (s["base_scene_id"], i): s
                for s in scenes
                for i in ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")
            }
        else:
            from .legacy_frozen import load_legacy_scenes

            scenes = load_legacy_scenes(
                Path(data_root) / "reward_fibers.jsonl",
                Path(data_root) / "reward_fibers_manifest.json",
            )
            scene_by_id = {(s["base_scene_id"], s["interface"]): s for s in scenes}
        if (
            canonical_hash(scenes)
            != json.loads((source / "data_manifest.json").read_text())["data_hash"]
        ):
            raise ValueError("scene data does not match frozen P3 data hash")
        raw_rows = _rows(rows_path)
        keys = set()
        for row in raw_rows:
            if row["sample_key"] in keys:
                raise ValueError("duplicate sample_key")
            keys.add(row["sample_key"])
            if row.get("record_hash") != canonical_hash(
                {k: v for k, v in row.items() if k != "record_hash"}
            ):
                raise ValueError("raw record hash mismatch")
            if row.get("decode_mode") not in ("sample", "greedy") or row.get("track") != track:
                raise ValueError("invalid decode mode or mixed tracks")
            if (row["base_scene_id"], row["interface"]) not in scene_by_id:
                raise ValueError("unmatched scene/interface key")
        raw_rows = [{**scene_by_id[(r["base_scene_id"], r["interface"])], **r} for r in raw_rows]
        identities.update(
            {
                "raw_manifest_hash": file_hash(source / "manifest.json"),
                "raw_files": checked_files,
                "data_hash": canonical_hash(scenes),
                "model_hash": canonical_hash(lock["model_spec"]),
            }
        )
    bind_run(out, identities, resume)
    if not rows_path:
        stage_status(
            out, "R0", "BLOCKED", "CPU_AUDIT", {"reason": "MISSING_ARTIFACT", "track": track}
        )
        return "BLOCKED"
    audited, counts, metrics = audit_rollouts(raw_rows, track=track)
    sample_rows = [r for r in audited if r["decode_mode"] == "sample"]
    greedy_rows = [r for r in audited if r["decode_mode"] == "greedy"]
    expected_sampled = config["R0"]["expected_sampled"][track]
    expected_greedy = config["R0"]["expected_greedy"][track]
    if (
        len(sample_rows) != expected_sampled
        or len(greedy_rows) != expected_greedy
        or not all(r["complete"] for r in counts)
    ):
        raise ValueError("incomplete sampled/greedy prompt banks")
    for mode in ("sample", "greedy"):
        for prompt in {r["prompt_id"] for r in audited}:
            indices = sorted(
                r["rollout_index"]
                for r in audited
                if r["prompt_id"] == prompt and r["decode_mode"] == mode
            )
            if indices != list(range(16 if mode == "sample" else 1)):
                raise ValueError("duplicate/missing rollout indices")
    disagreements = sum(r["category"] != r["reclassified_category"] for r in audited)
    unique_checks = []
    if track == "N":
        for scene in scenes:
            solutions = solve(scene["observed_world"], scene["cue"])
            changed = [
                i
                for i, (a, b) in enumerate(
                    zip(scene["truth_world"], scene["observed_world"], strict=True)
                )
                if a != b
            ]
            unique_checks.append(
                {
                    "base_scene_id": scene["base_scene_id"],
                    "solutions": solutions,
                    "passed": solutions == [scene["truth_world"]]
                    and changed == [scene["changed_index"]],
                }
            )
        if not all(c["passed"] for c in unique_checks):
            raise ValueError("independent unique repair or changed-index audit failed")
    actual = {
        "track": track,
        "model": lock["model_spec"],
        "generation": lock["generation_protocol"],
        "actual_max_new_tokens": lock["max_new_tokens"],
        "source_commit": lock["source_commit"],
        "source_files": lock["source_files"],
        "runtime_versions": lock["versions"],
        "model_audit": lock["model_audit"],
        "data_hash": canonical_hash(scenes),
        "raw_manifest_sha256": file_hash(source / "manifest.json"),
        "config_hash": canonical_hash(config),
        "inference_pass": True,
        "training_smoke_pass": None,
        "parser": "N strict whole JSON array"
        if track == "N"
        else "C3.semantic_action_parser.parse_p1",
    }
    write_json(out / "baseline_lock.json", actual)
    write_json(
        out / "data_solver_checks.json",
        {
            "track": track,
            "checks": unique_checks,
            "legacy_solver_scope": "not replaced by N solver",
        },
    )
    _write_counts(out / "per_prompt_counts.csv", counts)
    pq.write_table(pa.Table.from_pylist(counts), out / "per_prompt_counts.parquet")
    (out / "audit_rollouts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audited), encoding="utf-8"
    )
    write_json(out / "paired_CI.json", scene_cluster_intervals(counts, repeats=5000, seed=20260909))
    write_json(
        out / "recomputed_metrics.json",
        {
            "metrics_reclassified": metrics,
            "metrics_original_labels": weighted_prompt_metrics(sample_rows),
            "greedy_descriptive": weighted_prompt_metrics(greedy_rows),
            "category_disagreements": disagreements,
            "sampled_rows": len(sample_rows),
            "greedy_rows": len(greedy_rows),
        },
    )
    support = {}
    for prompt in sorted({r["prompt_id"] for r in sample_rows}):
        support[prompt] = audit_categories(
            [r["reclassified_category"] for r in sample_rows if r["prompt_id"] == prompt],
            tuple(config["R0"]["K_bank_audit"]),
            tuple(config["R0"]["lambda_advantage_grid"]),
            config["reward"]["epsilon"],
        )
    write_json(out / "group_support_K.json", support)
    _write_counts(
        out / "group_support_K.csv",
        [{"prompt_id": p, **r} for p, rows in support.items() for r in rows],
    )
    findings.append({"check": "raw_manifest_and_record_hashes", "status": "PASS"})
    findings.append({"check": "sampled_greedy_counts_and_indices", "status": "PASS"})
    findings.append(
        {"check": "classification_reproduction", "status": "PASS" if not disagreements else "FAIL"}
    )
    # Token IDs are preserved, but tokenizer decoding and the requested new image
    # inspection are still separate R0 requirements, not inferred from old labels.
    unresolved = [
        "raw token-to-text decoding audit with pinned tokenizer",
        "36 hash-selected image/processor spot checks",
        "cross-split leakage recheck",
    ]
    findings.extend({"check": item, "status": "NOT_STARTED"} for item in unresolved)
    _write_counts(out / "audit_findings.csv", findings)
    write_json(out / "audit_findings.json", findings)
    (out / "protocol_diff.md").write_text(
        "# 实际协议\n\n"
        + json.dumps(actual["generation"], indent=2)
        + "\n\nN/L 保留各自实际长度与 parser。\n",
        encoding="utf-8",
    )
    status = "BLOCKED" if disagreements else "INCONCLUSIVE"
    details = {
        "track": track,
        "read_only": True,
        "sampled_rows": len(sample_rows),
        "greedy_rows": len(greedy_rows),
        "category_disagreements": disagreements,
        "manifest_files_verified": len(checked_files),
        "remaining_R0_checks": unresolved,
        "R0_gate_passed": False,
    }
    artifacts = sorted(
        p
        for p in out.iterdir()
        if p.is_file()
        and p.name not in ("status.json", "manifest.json", "report.md", "report_zh.md")
    )
    stage_status(out, "R0", status, "CPU_AUDIT", details, artifacts)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    parser.add_argument("--phase", default="R0", choices=["R0"])
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--track", choices=["N", "L"], default="N")
    parser.add_argument("--out", type=Path, default=Path("runs/NEXT_20260909/R0"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {"phase": "R0", **dry_run_plan(config), "read_only": True},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return (
        0
        if audit_phase(
            config, args.out, args.rows, args.read_only, args.data_root, args.track, args.resume
        )
        in {"PASS", "BLOCKED", "INCONCLUSIVE"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
