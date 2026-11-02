"""Assemble a private analysis package from saved exports, without scoring again."""

import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / ".local/review_package_20261102"
OUT = BASE / "SR_F6_PILOT_DEV_FOR_GPT_CLAUDE_20261102"
DEV = REPO / ".local/pilot_dev_eval_v2_20261101"
METRICS = ("A", "J", "V", "S", "B")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def table(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy(source, name):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert target.read_bytes() == Path(source).read_bytes()


def safe(name):
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def collect():
    OUT.mkdir(parents=True, exist_ok=False)
    for part in ("data", "b0", "b1", "b2", "train_b0", "train_b1", "train_b2", "source"):
        with zipfile.ZipFile(BASE / f"{part}_export.zip") as archive:
            assert archive.testzip() is None
            assert len(set(archive.namelist())) == len(archive.namelist())
            for name in archive.namelist():
                assert safe(name)
                target = OUT / name
                assert not target.exists(), name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    for p in (BASE / "editorial").glob("*.md"):
        copy(p, f"00_READ_FIRST/{p.name}")
    for name in ("SAVED_SUMMARY.json", "SAVED_COMPLETE.json", "DEV_RESULTS_zh.md"):
        copy(DEV / name, f"01_RESULTS/{name}")
    for name, expected in {
        "SAVED_SUMMARY.json": "ef290b7c2441bcc99591ff4dad5d4ccf0cccd28fe0bb1b6542cb9113fbaf711c",
        "SAVED_COMPLETE.json": "abd8b7fadf8d3a15b3b11c9410c8d0b919e78c6c05d4ce2a2a17051e87820e2e",
    }.items():
        assert sha((DEV / name).read_bytes()) == expected
    copy(REPO / "configs/sr_f6.json", "04_DESIGN/sr_f6.json")
    assert (
        sha((OUT / "04_DESIGN/sr_f6.json").read_bytes())
        == "59052e828fecec1a5cf15255cb0ef11f269406a087ee373688bd977012e319d9"
    )
    copy(
        REPO / "docs/sr_f6/package/IMPLEMENTATION_SPEC_zh.md",
        "04_DESIGN/ARCHIVED_FULL_DESIGN_NOT_EXECUTED_zh.md",
    )
    for src, name in (
        (
            ".local/pilot_round_b_20261031/B_CLOSURE_SMALL_READ_20261101T071440.json",
            "B_CLOSURE_SAVED_READ.json",
        ),
        (
            ".local/pilot_round_b_20261031/RECOVERY_CLOSURE_READ_20261031T142346.json",
            "RECOVERY_SAVED_READ.json",
        ),
        (".local/pilot_dev_eval_20261101/CONTINUE_zh.md", "OLD_DEV_HISTORY_zh.md"),
        (".local/pilot_dev_eval_v2_20261101/CONTINUE_zh.md", "DEV_V2_HISTORY_zh.md"),
        (
            ".local/pilot_post_b_delivery_v2_20261101/CONTINUE_zh.md",
            "DELIVERY_V2_HISTORICAL_NOT_CURRENT_zh.md",
        ),
    ):
        copy(REPO / src, f"05_PROVENANCE/{name}")
    redact_shared_snapshot()
    original = json.loads((DEV / "READ_20261102T045614716793_RESULT.json").read_text())
    read = json.loads(original["stdout"])
    # Avoid exporting shared-account queue snapshots or unrelated users' jobs.
    save(
        "05_PROVENANCE/DEV_EXECUTION_SAVED_RECEIPTS.json",
        {
            k: read[k]
            for k in (
                "at",
                "accounting",
                "execution",
                "source",
                "authorization_basis",
                "full_delivery_verified",
                "evaluation_authorized",
                "formal_authorized",
                "test_authorized",
            )
        },
    )
    save(
        "05_PROVENANCE/LOCAL_MONITOR_SOURCE.json",
        {
            "source_name": "READ_20261102T045614716793_RESULT.json",
            "sha256": sha((DEV / "READ_20261102T045614716793_RESULT.json").read_bytes()),
            "projection": (
                "Only own-job status, execution receipts and authorization fields included."
            ),
        },
    )


def redact_shared_snapshot():
    source = REPO / ".local/pilot_round_b_20261031/B_CLOSURE_SMALL_READ_20261101T071440.json"
    wrapper = json.loads(source.read_text())
    saved = json.loads(wrapper["stdout"])
    own = {k: saved[k] for k in ("at", "outer", "refs", "updates")}
    own["accounting"] = saved["commands"]["accounting"]
    own["projection_note"] = "Shared-account queue/process snapshot and remote script omitted."
    own["original_monitor_sha256"] = sha(source.read_bytes())
    save("05_PROVENANCE/B_CLOSURE_SAVED_READ.json", own)


def validate_sources():
    included = {sha(p.read_bytes()) for p in OUT.rglob("*") if p.is_file()}
    originals = 0
    for index in (OUT / "05_PROVENANCE").glob("*_SOURCE_INDEX.json"):
        for entry in json.loads(index.read_text()):
            # DEV raw is a declared text projection; its original byte hash is
            # carried in every row, rather than claiming raw reconstruction.
            if "/draw-" in entry["source"]:
                continue
            assert entry["sha256"] in included, entry["source"]
            originals += 1
    public = {}
    for panel in ("DEV_EVAL", "DEV_PILOT"):
        for line in (OUT / f"04_DESIGN/{panel}_MODEL_INPUTS.jsonl").read_text().splitlines():
            row = json.loads(line)
            model_input = row["model_input"]
            image = OUT / "04_DESIGN" / model_input["image_file"]
            assert sha(image.read_bytes()) == model_input["image_sha256"]
            if panel == "DEV_EVAL":
                public[row["qid"]] = model_input
    for path in (OUT / "02_EVALUATION").glob("*_answers.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["answer"]["model_input"] == public[row["record"]["qid"]]
    return originals


def evaluation_tables():
    summary = json.loads((OUT / "01_RESULTS/SAVED_SUMMARY.json").read_text())
    all_rows, endpoint_rows = [], []
    for path in sorted((OUT / "02_EVALUATION").glob("*_answers.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 192
        report = json.loads(
            path.with_name(path.name.replace("_answers.jsonl", "_COMPLETE.json")).read_text()
        )
        assert [r["record"] for r in rows] == report["records"]
        assert len({(r["record"]["qid"], r["record"]["draw"]) for r in rows}) == 192
        for row in rows:
            endpoint = row["endpoint"]
            parent, arm = endpoint.split("/")[:2]
            record = row["record"]
            all_rows.append(
                {
                    "endpoint": endpoint,
                    "parent": parent,
                    "arm": arm,
                    "qid": record["qid"],
                    "root_id": record["root_id"],
                    "family": record["family"],
                    "draw": record["draw"],
                    "seed": record["seed"],
                    **{m: record["score"][m] for m in METRICS},
                }
            )
        for metric in METRICS:
            values = [r["record"]["score"][metric] for r in rows]
            known = sum(v in (0, 1) for v in values)
            passed = sum(v == 1 for v in values)
            saved = summary["endpoints"][endpoint][metric]
            assert (passed, known, len(values) - known) == (
                saved["passed"],
                saved["known"],
                saved["unknown"],
            )
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "parent": parent,
                    "arm": arm,
                    "metric": metric,
                    "passed": passed,
                    "failed": known - passed,
                    "known": known,
                    "unknown": len(values) - known,
                    "total": len(values),
                    "pass_rate_given_known": passed / known if known else None,
                }
            )
    assert len(all_rows) == 2304
    table("01_RESULTS/answer_scores.csv", all_rows)
    table("01_RESULTS/endpoint_metrics.csv", endpoint_rows)
    for dimension in ("family", "root_id"):
        groups = defaultdict(list)
        for row in all_rows:
            groups[(row["endpoint"], row[dimension])].append(row)
        result = []
        for (endpoint, value), rows in sorted(groups.items()):
            for metric in METRICS:
                values = [r[metric] for r in rows]
                known = sum(v in (0, 1) for v in values)
                result.append(
                    {
                        "endpoint": endpoint,
                        dimension: value,
                        "metric": metric,
                        "passed": sum(v == 1 for v in values),
                        "known": known,
                        "unknown": len(values) - known,
                        "total": len(values),
                    }
                )
        table(f"01_RESULTS/{'family' if dimension == 'family' else 'root'}_metrics.csv", result)
    paired = []
    indexed = {(r["parent"], r["arm"], r["qid"], r["draw"]): r for r in all_rows}
    for parent in ("b0", "b1", "b2"):
        for other in ("parent", "U10", "C_SHUF"):
            rows = [r for r in all_rows if r["parent"] == parent and r["arm"] == "U11"]
            for metric in METRICS:
                for level in ("ALL", "family", "root_id"):
                    labels = ["ALL"] if level == "ALL" else sorted({r[level] for r in rows})
                    for label in labels:
                        chosen = rows if level == "ALL" else [r for r in rows if r[level] == label]
                        pairs = [
                            (r[metric], indexed[(parent, other, r["qid"], r["draw"])][metric])
                            for r in chosen
                        ]
                        known = [(a, b) for a, b in pairs if a in (0, 1) and b in (0, 1)]
                        wins = sum(a > b for a, b in known)
                        losses = sum(a < b for a, b in known)
                        paired.append(
                            {
                                "parent": parent,
                                "target": "U11",
                                "comparator": other,
                                "metric": metric,
                                "level": level,
                                "group": label,
                                "total_pairs": len(pairs),
                                "known_pairs": len(known),
                                "unknown_pairs": len(pairs) - len(known),
                                "target_only_pass": wins,
                                "comparator_only_pass": losses,
                                "both_pass": sum(a == b == 1 for a, b in known),
                                "both_fail": sum(a == b == 0 for a, b in known),
                                "net_pass_difference": wins - losses,
                                "paired_rate_difference": (wins - losses) / len(known)
                                if known
                                else None,
                            }
                        )
    table("01_RESULTS/paired_arm_differences.csv", paired)
    return len(all_rows)


def training_tables():
    group_rows, scored_rows, updates, totals = [], [], [], []
    for directory in sorted((OUT / "03_TRAINING").iterdir()):
        parent, arm = directory.name.split("_", 1)
        groups = []
        for path in sorted(directory.glob("*_credit.json")):
            credit = json.loads(path.read_text())
            assert len(credit["groups"]) == 16
            for group in credit["groups"]:
                assert (
                    len(group["scored_rows"]) == len(group["applied"]) == len(group["native"]) == 8
                )
                groups.append(group)
                group_rows.append(
                    {
                        "parent": parent,
                        "arm": arm,
                        "step": group["step"],
                        "group_id": group["group_id"],
                        "qid": group["qid"],
                        "root_id": group["root_id"],
                        "family": group["family"],
                        "candidate_nonzero": any(v != 0 for v in group["candidate"]),
                        "applied_nonzero": any(v != 0 for v in group["applied"]),
                        "candidate_l1": sum(abs(v) for v in group["candidate"]),
                        "applied_l1": sum(abs(v) for v in group["applied"]),
                        "J_passed": sum(r["J"] == 1 for r in group["scored_rows"]),
                        "rows": 8,
                    }
                )
                for i, r in enumerate(group["scored_rows"]):
                    scored_rows.append(
                        {
                            "parent": parent,
                            "arm": arm,
                            "step": group["step"],
                            "group_id": group["group_id"],
                            "row_id": r["row_id"],
                            "qid": group["qid"],
                            "root_id": group["root_id"],
                            "family": group["family"],
                            "draw": r["draw"],
                            **{m: r[m] for m in ("J", "V", "S", "B")},
                            "native": group["native"][i],
                            "candidate": group["candidate"][i],
                            "applied": group["applied"][i],
                        }
                    )
        assert len(groups) == 192
        nonzero = [g for g in groups if any(v != 0 for v in g["applied"])]
        totals.append(
            {
                "parent": parent,
                "arm": arm,
                "steps": 12,
                "groups": len(groups),
                "rows": len(groups) * 8,
                "J_passed": sum(r["J"] == 1 for g in groups for r in g["scored_rows"]),
                "applied_nonzero_groups": len(nonzero),
                "applied_nonzero_roots": len({g["root_id"] for g in nonzero}),
                "applied_nonzero_families": len({g["family"] for g in nonzero}),
            }
        )
        for path in sorted(directory.glob("*_UPDATE.json")):
            u = json.loads(path.read_text())
            updates.append(
                {
                    "parent": parent,
                    "arm": arm,
                    "step": int(path.name[5:8]),
                    **{
                        k: u[k]
                        for k in (
                            "status",
                            "optimizer_updates",
                            "sequence_count",
                            "token_count",
                            "training_seconds",
                            "loss",
                            "native",
                            "aux",
                            "kl",
                            "parameter_update_norm",
                            "gradient_norm_before_clip",
                            "gradient_norm_after_clip",
                        )
                    },
                    "channel_gradient_status": u["channel_gradients"].get("status"),
                    "channel_gradient_norms": json.dumps(
                        u["channel_gradients"].get("norms"), sort_keys=True
                    ),
                    "channel_gradient_cosines": json.dumps(
                        u["channel_gradients"].get("cosines"), sort_keys=True
                    ),
                }
            )
    assert len(scored_rows) == 13824 and len(updates) == 108
    table("01_RESULTS/training_path_summary.csv", totals)
    table("01_RESULTS/training_groups.csv", group_rows)
    table("01_RESULTS/training_scored_rows.csv", scored_rows)
    table("01_RESULTS/training_updates.csv", updates)
    return len(scored_rows), len(updates)


def finish(counts):
    originals = validate_sources()
    omissions = {
        "purpose": (
            "Saved-evidence analysis projection for GPT and Claude; no new generation or rescoring."
        ),
        "scope": (
            "9 primary paths x 12 updates; 12 DEV endpoints x 192 answers; unique "
            "recovery is separate."
        ),
        "counts": counts,
        "not_included": [
            "Training raw full answer texts (13824 primary and 1024 recovery)",
            (
                "DEV raw token IDs, sampler logprobs, prefix views and other fields "
                "explicitly named in omitted_raw_fields"
            ),
            "Model weights, checkpoint/state/optimizer/RNG tensors, GPU gradient tensors",
            "Full historical census/confirmation raw and legacy failed-attempt diagnostic tensors",
            "A complete executable dependency environment or full reproduction archive",
        ],
        "included_exception": (
            "Original full UPDATE JSON is preserved, including its stored probability "
            "vectors and scalar/gradient diagnostics."
        ),
        "original_data_root": (
            "/projects/_ssd/varunssd/louis-ssvc/sr_f6_semantic_factorial_20261027"
        ),
        "status": "COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED",
        "gate_status": "NOT_SIGNED",
        "full_delivery_verified": False,
        "delivery_audit_waived_by_user": True,
        "formal_test_executed": False,
        "provenance_note": (
            "Old CONTINUE documents are historical. Current completion is in 01_RESULTS"
            " and DEV_EXECUTION_SAVED_RECEIPTS. Frozen spec status records registration"
            " time."
        ),
        "derivation_note": (
            "CSV counts/paired discordance use saved labels unchanged, no new tests or "
            "bootstrap. Empty CSV metric cells mean unknown, not zero."
        ),
        "input_path_map": (
            "Original model_input.image_file DATA/images/X resolves to "
            "04_DESIGN/DATA/images/X inside this package."
        ),
    }
    save("OMISSIONS_AND_COVERAGE.json", omissions)
    (OUT / "START_HERE_zh.md").write_text(
        "# SR-F6 分析包\n\n先读 [阅读指南](00_READ_FIRST/START_HERE_zh.md), 再把 "
        "[分析提示](00_READ_FIRST/ANALYSIS_PROMPT_zh.md) 发给 GPT / Claude。\n\n"
        "当前结果见 [结果说明](01_RESULTS/DEV_RESULTS_zh.md)。全部 2304 条 DEV 回答完整保留; "
        "训练信用和更新记录完整收入, 训练回答原文及模型大张量省略。"
        "具体见 [覆盖与遗漏](OMISSIONS_AND_COVERAGE.json) 和 [文件索引](FILE_INDEX.csv)。\n"
    )
    brief = (
        (OUT / "00_READ_FIRST/ANALYSIS_PROMPT_zh.md").read_text()
        + "\n\n---\n\n"
        + (OUT / "01_RESULTS/DEV_RESULTS_zh.md").read_text()
    )
    (OUT / "ANALYSIS_BRIEF_zh.md").write_text(brief)
    # Validate portable links in newly written entry documents. Historical documents
    # retain original source references, resolved by the source index where included.
    for path in [OUT / "START_HERE_zh.md", *sorted((OUT / "00_READ_FIRST").glob("*.md"))]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            assert (
                target == "FILE_INDEX.csv"
                or "://" in target
                or (path.parent / target.split("#")[0]).exists()
            ), (path, target)
    excluded = {"FILE_INDEX.csv", "MANIFEST.json", "SHA256SUMS.txt"}
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in excluded)
    rows = [
        {"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
        for p in files
    ]
    table("FILE_INDEX.csv", rows)
    rows.append(
        {
            "path": "FILE_INDEX.csv",
            "bytes": (OUT / "FILE_INDEX.csv").stat().st_size,
            "sha256": sha((OUT / "FILE_INDEX.csv").read_bytes()),
        }
    )
    save(
        "MANIFEST.json",
        {
            "format": "SRF6_REVIEW_PACKAGE_V1",
            "files": rows,
            "hash_scope": (
                "All payloads except MANIFEST.json and SHA256SUMS.txt; archive checksum is "
                "external."
            ),
        },
    )
    manifest_data = (OUT / "MANIFEST.json").read_bytes()
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in rows)
        + f"{sha(manifest_data)}  MANIFEST.json\n"
    )
    target = BASE / (OUT.name + ".zip")
    temporary = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, str(p.relative_to(OUT)))
    with zipfile.ZipFile(temporary) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and all(safe(n) for n in names)
        assert z.testzip() is None
        for r in rows:
            b = z.read(r["path"])
            assert len(b) == r["bytes"] and sha(b) == r["sha256"]
        for line in z.read("SHA256SUMS.txt").decode().splitlines():
            expected, name = line.split("  ", 1)
            assert sha(z.read(name)) == expected
        assert set(names) == {r["path"] for r in rows} | {"MANIFEST.json", "SHA256SUMS.txt"}
    temporary.replace(target)
    checksum = sha(target.read_bytes())
    receipt = {
        "archive": target.name,
        "bytes": target.stat().st_size,
        "sha256": checksum,
        "members": len(names),
        "checks": [
            "ZIP_CRC",
            "UNIQUE_SAFE_PATHS",
            "ALL_PAYLOAD_HASHES",
            "MANIFEST",
            "SHA256SUMS",
            "NEW_ENTRY_LINKS",
            "SAVED_METRIC_COUNTS",
            "ORIGINAL_REPORT_BYTE_IDENTITY",
        ],
        "counts": counts,
        "source_byte_identity_members": originals,
        "scope": "Package transport/content checks, not a scientific or full-delivery audit.",
    }
    target.with_suffix(".zip.sha256").write_text(f"{checksum}  {target.name}\n")
    target.with_suffix(".zip.verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    collect()
    evaluation_count = evaluation_tables()
    training_count, update_count = training_tables()
    finish(
        {
            "DEV_answers": evaluation_count,
            "primary_training_scored_rows": training_count,
            "primary_updates": update_count,
            "DEV_endpoints": 12,
        }
    )
