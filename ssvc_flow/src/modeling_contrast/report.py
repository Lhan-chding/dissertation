"""Render factual Chinese decisions from completed, hash-bound CPU artifacts.

Rendering performs no measurements or fitting. Missing evidence stays missing;
N3 authorization is verified separately from N4 execution. The output directory
may already hold unrelated evidence, while every report-owned file is immutable.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from .io import checked_output_path, sha256_file, write_json, write_text
from .protocol import ROOT
from .selection import verify_selection_lock

REPORTS = (
    "MODELING_DECISION_V2_zh.md",
    "REVISED_IDEA_zh.md",
    "REAL_DATA_REQUIREMENTS_zh.md",
    "IMPLEMENTATION_REPORT_zh.md",
    "machine_readable_readiness.json",
    "CPU_ACCEPTANCE_RESULTS_zh.md",
    "REPORT_SOURCE_BINDING.json",
    "DIAGNOSTIC_ISOLATION_SUMMARY.csv",
    "MEASUREMENT_COVARIANCE_AUDIT.json",
    "REPORT_MANIFEST.json",
)


class Evidence:
    def __init__(self):
        self.inputs = {}
        self.missing = set()

    def read(self, path, kind="json"):
        path = Path(path).resolve()
        if not path.is_file():
            self.missing.add(str(path))
            return [] if kind == "csv" else None
        self.inputs[str(path)] = sha256_file(path)
        with path.open(encoding="utf-8", newline="") as stream:
            if kind == "csv":
                return list(csv.DictReader(stream))
            if kind == "text":
                return stream.read()
            return json.load(stream)

    def reference(self, path):
        path = Path(path).resolve()
        digest = self.inputs.get(str(path))
        return f"[{path.name}]({path})；SHA-256 `{digest}`" if digest else f"`{path}`：MISSING"

    def verify(self, path, expected):
        """Bind actual bytes, including binary evidence, before trusting a receipt."""
        path = Path(path).resolve()
        if not path.is_file():
            self.missing.add(str(path))
            return False
        actual = sha256_file(path)
        previous = self.inputs.setdefault(str(path), actual)
        return isinstance(expected, str) and actual == expected and previous == actual


def _bound_files(evidence, files, base):
    if not isinstance(files, dict) or not files:
        return False
    matched = []
    for name, digest in files.items():
        path = Path(name)
        matched.append(evidence.verify(path if path.is_absolute() else base / path, digest))
    return all(matched)


def _supplemental_audits(evidence, run_root, config):
    """Receipt PASS is necessary but not sufficient for checklist acceptance."""
    root = Path(run_root).resolve()
    specifications = {
        "A4": ("git_scope_review.json", "contrast-git-scope-v1"),
        "F2": ("TOTAL_COST_AUDIT.json", "contrast-total-cost-v1"),
        "F9": ("delivery_coverage.json", "contrast-delivery-coverage-v1"),
    }
    result = {}
    for ident, (name, schema) in specifications.items():
        path = root / "implementation" / name
        receipt = evidence.read(path)
        row = {"status": "NOT_RUN", "sources": [str(path)], "detail": "缺少实际审计回执"}
        if ident == "F2":
            row["actual_cpu_saving_claim_verified"] = False
        result[ident] = row
        if not isinstance(receipt, dict):
            continue
        row["receipt"] = receipt
        if not _bound_files(evidence, receipt.get("evidence_files"), path.parent):
            row.update(status="FAIL", detail="回执缺少哈希绑定，或绑定文件缺失/字节不匹配")
            continue
        base_valid = receipt.get("schema_version") == schema and receipt.get("status") == "PASS"
        if ident == "A4":
            valid = (
                base_valid
                and receipt.get("parent_commit") == config["repository"]["parent_commit"]
                and Path(receipt.get("worktree", "")).is_dir()
                and bool(receipt.get("branch"))
                and bool(receipt.get("reviewed_paths"))
                and all(
                    receipt.get(key) is True
                    for key in (
                        "independent_worktree",
                        "unrelated_changes_preserved",
                        "historical_sources_unchanged",
                    )
                )
            )
            row.update(
                status="PASS" if valid else "BLOCKED",
                detail="已核验实际git范围审查回执和引用文件字节；工作树隔离及历史源码保留按回执逐项检查",
            )
        elif ident == "F2":
            coverage = receipt.get("coverage", {})
            counts, times = (
                receipt.get("operation_counts", {}),
                receipt.get("wall_seconds_by_stage", {}),
            )

            def numeric(values):
                return (
                    isinstance(values, dict)
                    and bool(values)
                    and all(
                        _number(value) is not None and float(value) >= 0
                        for value in values.values()
                    )
                )

            valid = (
                base_valid
                and all(
                    coverage.get(key) is True
                    for key in (
                        "generation",
                        "labels",
                        "scoring",
                        "calibration",
                        "oracle",
                        "fit",
                        "query",
                        "candidate_updates",
                    )
                )
                and numeric(counts)
                and numeric(times)
                and receipt.get("scenario_costs_are_conditional") is True
            )
            measured_saving = (
                receipt.get("actual_cpu_saving_claim") is True
                and receipt.get("cpu_timing_comparison_status") == "PASS"
                and receipt.get("same_conditions_cpu_comparison_verified") is True
                and _bound_files(evidence, receipt.get("cpu_timing_evidence_files"), path.parent)
            )
            row["actual_cpu_saving_claim_verified"] = bool(valid and measured_saving)
            valid &= receipt.get("actual_cpu_saving_claim") is False or measured_saving
            row.update(
                status="PASS" if valid else "BLOCKED",
                detail="全阶段实际操作计数及walltime覆盖核验；条件价格/零查询成本下界不等于实测CPU节约",
            )
        else:
            included, excluded = receipt.get("included", []), receipt.get("excluded", [])
            entries = included + excluded
            valid = (
                base_valid
                and bool(included)
                and receipt.get("delivery_mode") == "git_and_local_artifacts"
                and receipt.get("npz_allow_pickle_false_verified") is True
            )
            if any(not isinstance(item, dict) or not item.get("path") for item in entries):
                valid = False
            else:
                valid &= all(
                    _bound_files(evidence, {item["path"]: item.get("sha256")}, path.parent)
                    for item in entries
                )
            valid &= all(item.get("location") in ("git", "local") for item in included)
            valid &= all(bool(item.get("reason")) for item in excluded)
            row.update(
                status="PASS" if valid else "BLOCKED",
                detail="git与本地交付coverage逐文件重新哈希；收录/排除项及NPZ读取验证按回执记录，新报告另由REPORT_MANIFEST绑定",
            )
    result.update(_recomputation_audits(evidence, root))
    return result


def _recomputation_audits(evidence, root):
    base = root / "recompute"
    summary_path, projection_path = (
        base / "RECOMPUTATION_SUMMARY.json",
        base / "PROJECTION_AUDIT_SUMMARY.json",
    )
    summary, projection = evidence.read(summary_path), evidence.read(projection_path)
    result = {
        "D10": {
            "status": "NOT_RUN",
            "sources": [str(projection_path)],
            "detail": "缺少raw/投影独立保存与复算回执",
        },
        "F10": {
            "status": "NOT_RUN",
            "sources": [str(summary_path)],
            "detail": "缺少最终表独立raw复算回执",
        },
    }
    if isinstance(projection, dict):
        bindings = sorted((base / "projection").glob("*/*/a*/PROJECTION_BINDING.json"))
        valid = (
            projection.get("status") == "PASS"
            and projection.get("complete_for_materialized_N2A_B_C_E_units") is True
            and projection.get("main_ranking_eligible") is False
            and len(bindings)
            == projection.get("units")
            == projection.get("materialized_N2A_B_C_E_units")
            and len(bindings) > 0
        )
        for path in bindings:
            value = evidence.read(path) or {}
            files = {
                value.get("raw_prediction_file", "MISSING_RAW"): value.get(
                    "raw_prediction_file_sha256"
                ),
                value.get("oracle_baseline_file", "MISSING_ORACLE"): value.get(
                    "oracle_baseline_file_sha256"
                ),
                "projected_sparse.npz": value.get("sparse_sha256"),
                "raw_projected_errors.jsonl.gz": value.get("error_table_sha256"),
            }
            valid &= _bound_files(evidence, files, path.parent)
            valid &= (
                value.get("main_ranking_eligible") is False
                and value.get("all_projected_arrays_bitwise_reconstructed_from_sparse_archive")
                is True
            )
        result["D10"].update(
            status="PASS" if valid else "BLOCKED",
            detail=f"{len(bindings)}个materialized单元的raw、oracle baseline、"
            "稀疏投影、误差表逐hash；"
            "投影仅posthoc诊断，不参加主排名",
            receipt=projection,
        )
    if isinstance(summary, dict):
        manifest_path, replay_path = (
            base / "COVERAGE_MANIFEST.json",
            base / "FORWARD_REPLAY_RECEIPTS.json",
        )
        manifest, replay = evidence.read(manifest_path) or {}, evidence.read(replay_path) or {}
        covariance = summary.get("N1_covariance_recovery", {})
        valid = evidence.verify(
            manifest_path, summary.get("coverage_manifest_sha256")
        ) and evidence.verify(replay_path, covariance.get("forward_replay_receipts_sha256"))
        files = manifest.get("read_files", [])
        valid &= bool(files) and len(files) == manifest.get("file_count")
        catalog_changes = []
        for item in files:
            matches = evidence.verify(item["path"], item.get("sha256"))
            source_path = Path(item["path"]).resolve()
            provenance_only = (
                set(item.get("roles", [])) == {"source_catalog"}
                and source_path.parent == Path(__file__).resolve().parent
                and source_path.name
                in {
                    "report.py",
                    "selection_recompute.py",
                    "frozen_validation.py",
                    "cli.py",
                    "ledger_audit.py",
                }
                and source_path.is_file()
            )
            if not matches and provenance_only:
                catalog_changes.append(
                    {
                        "path": str(source_path),
                        "replay_catalog_sha256": item.get("sha256"),
                        "current_sha256": sha256_file(source_path),
                        "status": "SOURCE_CATALOG_CHANGED_AFTER_REPLAY",
                        "scope": "已知未被recompute调用的报告/编排模块，目录清单仅为provenance；"
                        "实际复算依赖仍须原hash匹配",
                    }
                )
            else:
                valid &= matches
        table = replay.get("tables_artifact") or {}
        valid &= _bound_files(
            evidence, {table.get("path", "MISSING_TABLE"): table.get("sha256")}, base
        )
        valid &= table.get("all_saved_array_hashes_verified") is True
        required = (
            "N1",
            "N1_final",
            "N2A",
            "N2B",
            "N2C",
            "N2E",
            "N2_final_metrics",
            "N2_aggregate",
            "N2_by_seed",
            "N2D",
        )
        valid &= all(
            summary.get("checks", {}).get(name, {}).get("status") == "PASS" for name in required
        )
        valid &= (
            summary.get("status") == "PASS"
            and not summary.get("errors")
            and covariance.get("status") == "PASS"
            and covariance.get("independent_contribution_moment_formulas") is True
            and covariance.get("action_distribution_inferred_from_four_events") is False
        )
        tolerance, maximum = (
            _number(summary.get("absolute_tolerance")),
            _number(summary.get("max_absolute_difference")),
        )
        valid &= (
            tolerance is not None and maximum is not None and 0 <= maximum <= tolerance <= 1e-12
        )
        result["F10"].update(
            status="PASS" if valid else "BLOCKED",
            detail="独立复算N1 raw/Helmert方差及N2 raw预测/最终聚合/逐seed表；"
            "完整动作表来自计费冻结CPU forward，未由四事件反推。"
            "N3锁与bootstrap推断另列，不声称此回执重算N3推断",
            receipt=summary,
            source_catalog_changes_after_replay=catalog_changes,
        )
        result["F10"]["sources"].extend([str(manifest_path), str(replay_path)])
    selection_path = root / "selection_recompute/RECOMPUTATION_SUMMARY.json"
    selection = evidence.read(selection_path)
    selection_valid = isinstance(selection, dict) and selection.get("status") == "PASS"
    if isinstance(selection, dict):
        selection_valid &= _bound_files(
            evidence, selection.get("evidence_files"), selection_path.parent
        )
        selection_valid &= all(
            selection.get("checks", {}).get(name, {}).get("status") == "PASS"
            for name in ("N3_BOOTSTRAP", "N3_POOLED", "N3_BINDINGS", "N3_GATES")
        )
    n12_status = result["F10"]["status"]
    result["F10"].update(
        status="PASS"
        if n12_status == "PASS" and selection_valid
        else "BLOCKED"
        if summary or selection
        else "NOT_RUN",
        N1_N2_recompute_status=n12_status,
        N3_recompute_status="PASS" if selection_valid else "BLOCKED" if selection else "NOT_RUN",
        N3_receipt=selection,
        detail="N1/N2独立raw复算、N1完整动作协方差公式及N3独立汇总/bootstrap/门禁分开核验；"
        "每项均须实际回执及源文件hash匹配，任何缺项不记全阶段PASS。",
    )
    result["F10"]["sources"].append(str(selection_path))
    if result["F10"].get("source_catalog_changes_after_replay"):
        result["F10"]["detail"] += (
            " 已知未执行的报告/编排模块目录清单发生后续变化，"
            "SOURCE_CATALOG_CHANGED_AFTER_REPLAY及旧/新hash见readiness；实际依赖未放宽。"
        )
    return result


def _covariance_audit(evidence, run_root, parent_root):
    """Supersede an overbroad immutable metadata statement with bound evidence."""
    import numpy as np

    root, parent = Path(run_root), Path(parent_root)
    n1_manifest_path = root / "N1/RUN_MANIFEST.json"
    n1_manifest = evidence.read(n1_manifest_path) or {}
    n1_outputs = n1_manifest.get("outputs", {})
    original_statements = []
    for path in sorted((root / "N1/oracle").glob("*/a*/covariance_audit.json")):
        value = evidence.read(path) or {}
        original_statements.append(
            {
                "path": str(path.resolve()),
                "sha256": evidence.inputs[str(path.resolve())],
                "bank_independence": value.get("bank_independence"),
                "matches_original_N1_manifest": evidence.verify(
                    path, n1_outputs.get(path.relative_to(root / "N1").as_posix())
                ),
            }
        )
    witness_path = root / "implementation/joint_packets_actual_audit.json"
    records = evidence.read(witness_path) or []
    manifest = evidence.read(parent / "manifest.json") if records else {}
    entries = {row["id"]: row for row in (manifest or {}).get("trajectories", [])}
    witnesses = []
    for row in records:
        banks = row.get("banks", [])
        if len(banks) != 2:
            witnesses.append({"status": "FAIL", "reason": "witness requires two banks"})
            continue
        path = (
            root
            / "implementation"
            / f"joint_packets_{row['trajectory_id']}_a{row['anchor']}_b{banks[0]}_{banks[1]}.npz"
        )
        entry = entries.get(row.get("trajectory_id"), {})
        valid = True
        for key in ("counts", "observations"):
            member = entry.get(f"{key}_file")
            valid &= bool(member) and evidence.verify(parent / member, row.get(f"{key}_sha256"))
        result = {
            "trajectory_id": row.get("trajectory_id"),
            "anchor": row.get("anchor"),
            "banks": banks,
            "path": str(path.resolve()),
        }
        if path.is_file():
            evidence.verify(path, sha256_file(path))
            with np.load(path, allow_pickle=False) as arrays:
                joint, block = arrays["joint"], arrays["old_block_diagonal"]
                offdiag = float(np.linalg.norm(joint[:, :6, 6:]))
                difference = float(np.linalg.norm(joint - block))
            valid &= bool(
                np.isclose(offdiag, row.get("offdiag_frobenius"), rtol=0, atol=1e-12)
            ) and bool(
                np.isclose(
                    difference, row.get("joint_minus_blockdiag_frobenius"), rtol=0, atol=1e-12
                )
            )
            result.update(
                sha256=evidence.inputs[str(path.resolve())],
                offdiag_frobenius_recomputed=offdiag,
                joint_minus_blockdiag_frobenius_recomputed=difference,
            )
        else:
            evidence.missing.add(str(path.resolve()))
            valid = False
        result["status"] = "PASS" if valid else "FAIL"
        witnesses.append(result)
    overlays = []
    for path in sorted((root / "N2/C1_auxiliary").glob("*/a*/O_IND_n*/receipt.json")):
        value = evidence.read(path) or {}
        source, auxiliary = (
            Path(value.get("source_packet_path", "MISSING")),
            path.parent / "C1_auxiliary.npz",
        )
        hashes_match = _bound_files(
            evidence,
            {
                str(source): value.get("source_packet_sha256"),
                str(auxiliary): value.get("auxiliary_sha256"),
            },
            path.parent,
        )
        if source.is_relative_to(root / "N1"):
            hashes_match &= evidence.verify(
                source, n1_outputs.get(source.relative_to(root / "N1").as_posix())
            )
        else:
            hashes_match = False
        indices = value.get("historical_replicas_preserved", [])
        valid = (
            hashes_match
            and value.get("version") == "C1_SAME_BANK_COUNTS_V1"
            and value.get("status") == "REPAIRED_WITHOUT_MUTATING_N1"
            and value.get("origin_counts_preserved") is True
            and value.get("fit_evaluation_independent") is True
        )
        preserved = False
        if hashes_match:
            with (
                np.load(source, allow_pickle=False) as original,
                np.load(auxiliary, allow_pickle=False) as corrected,
            ):
                preserved = np.array_equal(
                    original["legacy_origin_counts"], corrected["legacy_origin_counts"]
                ) and np.array_equal(
                    original["legacy_total_levels"][indices],
                    corrected["legacy_total_levels"][indices],
                )
                valid &= preserved
        overlays.append(
            {
                "receipt_path": str(path.resolve()),
                "receipt_sha256": evidence.inputs[str(path.resolve())],
                "source_packet_path": str(source),
                "source_packet_sha256": value.get("source_packet_sha256"),
                "auxiliary_path": str(auxiliary.resolve()),
                "auxiliary_sha256": value.get("auxiliary_sha256"),
                "historical_replicas_preserved": indices,
                "original_counts_and_historical_replicas_equal": bool(preserved),
                "fit_evaluation_independence": "DECLARED_IN_BOUND_BANK_LOCAL_REPAIR_RECEIPT",
                "status": "PASS" if valid else "FAIL",
            }
        )
    witness_ok = (
        bool(witnesses)
        and all(row["status"] == "PASS" for row in witnesses)
        and any(row.get("offdiag_frobenius_recomputed", 0) > 0 for row in witnesses)
    )
    overlay_ok = bool(overlays) and all(row["status"] == "PASS" for row in overlays)
    originals_match = (
        n1_manifest.get("status") == "COMPLETE"
        and bool(original_statements)
        and all(row["matches_original_N1_manifest"] for row in original_statements)
    )
    return {
        "schema_version": "measurement-covariance-audit-v1",
        "status": "PASS"
        if witness_ok and overlay_ok and originals_match
        else "FAIL"
        if any(row["status"] == "FAIL" for row in witnesses + overlays)
        else "NOT_RUN",
        "N1_metadata_erratum": {
            "status": "VERIFIED_AND_SUPERSEDED" if witness_ok else "NOT_VERIFIED",
            "superseded_field": "bank_independence: true",
            "correction": "N1 oracle/covariance_audit.json的bank_independence:true声明过宽。"
            "旧O_IND noise 0-4的fit计数曾按全局参数alias复用，跨bank协方差一般非零；"
            "N2联合观测保留这些cross-bank项。新evaluation采用独立新packet，"
            "不能将旧历史计数的相关性或该笼统元数据直接移用到新eval。",
            "N1_original_files_modified": False if originals_match else None,
            "N1_manifest_original_hashes_verified": originals_match,
            "N1_manifest_path": str(n1_manifest_path.resolve()),
            "N1_manifest_sha256": evidence.inputs.get(str(n1_manifest_path.resolve())),
            "original_statements": original_statements,
            "witnesses": witnesses,
            "evidence_path": str(witness_path.resolve()),
            "evidence_sha256": evidence.inputs.get(str(witness_path.resolve())),
        },
        "C1_auxiliary_repair": {
            "status": "PASS" if overlay_ok else "FAIL" if overlays else "NOT_RUN",
            "scope": "N2/C1_auxiliary只替换C1拟合的辅助total levels。按各自bank的计数补充；"
            "主contrast观测和N1原件不变。保存5份原历史副本及origin counts；"
            "新fit/eval独立性由bank-local修正回执及独立测试核验。",
            "verified_overlay_count": sum(row["status"] == "PASS" for row in overlays),
            "overlays": overlays,
        },
        "verification_scope": "只读取并重新哈希原件，独立计算示例cross-bank范数及C1保留数组相等性；"
        "不执行新forward/采样/拟合，不把示例当全部协方差逐元素复算。"
        "完整N1方差复算另见recompute回执。",
    }


def _validation_audit(evidence, value, path, selection_path, selected):
    if not isinstance(value, dict):
        return {"status": "NOT_RUN", "recommendation_qualified": False}
    valid = all(
        _bound_files(evidence, value.get(key), path.parent)
        for key in ("input_hashes", "prediction_hashes", "scoring_hashes")
    )
    valid &= evidence.verify(selection_path, value.get("selection_sha256"))
    valid &= value.get("input_hashes", {}).get(str(selection_path.resolve())) == value.get(
        "selection_sha256"
    )
    identities = set()
    for candidate in selected:
        fields = ("model", "method", "n", "rank", "alpha", "eta", "fit_banks")
        if not all(key in candidate for key in fields):
            valid = False
            continue
        identities.add("|".join(str(candidate[key]) for key in fields))
    rows = value.get("selected_results", [])
    valid &= bool(identities) and {row.get("configuration_id") for row in rows} == identities
    valid &= len(rows) == len(identities)
    complete = value.get("status") == "INDEPENDENT_VALIDATION_COMPLETE"
    prediction_pass = value.get("science_status") == "PREDICTION_PASS" and all(
        row.get("prediction_status") == "PREDICTION_PASS" for row in rows
    )
    resolution_pass = all(row.get("resolution_status") == "RESOLUTION_PASS" for row in rows)
    return {
        "status": "PASS" if valid and complete else "FAIL",
        "independent_science_status": value.get("science_status", "UNKNOWN"),
        "resolution_status": "PASS" if rows and resolution_pass else "FAIL",
        "recommendation_qualified": bool(
            valid and complete and prediction_pass and resolution_pass
        ),
        "source": str(path),
        "scope": value.get("scope", "UNKNOWN"),
        "selected_configuration_ids": sorted(identities),
        "qualification_does_not_imply_online_or_safety_certification": True,
    }


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value):
    if value is None or value == "":
        return "UNKNOWN"
    numeric = _number(value)
    if numeric is None:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:.6g}"


def _table(headers, rows):
    if not rows:
        return "未产生可用结果（NOT_RUN / MISSING）。\n"
    return (
        "\n".join(
            (
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *("| " + " | ".join(_fmt(x) for x in row) + " |" for row in rows),
            )
        )
        + "\n"
    )


def _gate(lock, name):
    return lock.get(name, {"status": "NOT_RUN", "reasons": ["NO_SELECTION_LOCK"]})


def _parse_tests(evidence, run_root):
    path = run_root / "implementation/final_tests.json"
    receipt = evidence.read(path)
    result = {
        "status": "UNKNOWN",
        "passed": None,
        "failures": None,
        "errors": None,
        "source": str(path.resolve()),
        "executed_passed_nodes": [],
        "junit_sources": [],
    }
    if receipt is None:
        return result
    summary = receipt.get("summary", receipt)
    result.update(
        status=summary.get("status", receipt.get("status", "UNKNOWN")),
        passed=summary.get("passed", summary.get("tests_passed")),
        failures=summary.get("failures"),
        errors=summary.get("errors"),
    )
    paths = []

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("junit", "junit_path", "junit_xml") and isinstance(item, str):
                    candidate = Path(item)
                    if not candidate.is_absolute():
                        candidate = next(
                            (
                                base / candidate
                                for base in (ROOT, run_root, path.parent)
                                if (base / candidate).is_file()
                            ),
                            path.parent / candidate,
                        )
                    paths.append(candidate)
                elif isinstance(item, (dict, list)):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(receipt)
    passed_nodes, failures, errors, skips = [], 0, 0, 0
    for candidate in sorted(set(paths)):
        text = evidence.read(candidate, "text")
        if text is None:
            continue
        result["junit_sources"].append(str(candidate.resolve()))
        for node in ET.fromstring(text).iter("testcase"):
            failures += int(node.find("failure") is not None)
            errors += int(node.find("error") is not None)
            skips += int(node.find("skipped") is not None)
            if all(node.find(tag) is None for tag in ("failure", "error", "skipped")):
                passed_nodes.append(f"{node.get('classname', '')}::{node.get('name', '')}")
    if paths and result["junit_sources"]:
        result.update(
            passed=len(passed_nodes),
            failures=failures,
            errors=errors,
            skipped=skips,
            status="PASS" if not failures and not errors else "FAIL",
        )
    result["executed_passed_nodes"] = passed_nodes
    return result


def _observation_rows(rows, primary):
    groups = defaultdict(list)
    for row in rows:
        if row.get("target") == primary and row.get("view") == "helmert":
            groups[(row.get("method"), row.get("n"))].append(row)
    result = []
    for (method, n), block in sorted(groups.items()):
        metrics = []
        for target in ("pX", "v"):
            error = sum(_number(row.get(f"{target}_error_ss")) or 0 for row in block)
            truth = sum(_number(row.get(f"{target}_truth_ss")) or 0 for row in block)
            metrics.append(math.sqrt(error / truth) if truth > 0 else None)

        def mean(field, block=block):
            values = [_number(row.get(field)) for row in block]
            values = [value for value in values if value is not None]
            return sum(values) / len(values) if values else None

        theory, empirical = mean("theoretical_variance"), mean("empirical_variance")
        result.append(
            [
                method,
                n,
                *metrics,
                theory,
                empirical,
                empirical / theory if theory and empirical is not None else None,
                sum(_number(row.get("unobserved_X_prompt_replicas")) or 0 for row in block),
            ]
        )
    return result


def _model_rows(rows, primary):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("target") == primary and row.get("population") == "all":
            grouped[(row.get("method"), row.get("observation"), row.get("n"))].append(row)
    display = []
    for key, block in sorted(grouped.items(), key=lambda item: str(item[0])):

        def score(row):
            values = [_number(row.get(f"{x}_nrmse")) for x in ("pX", "v")]
            return max(values) if all(x is not None for x in values) else math.inf

        row = min(block, key=score)
        display.append(
            [
                *key,
                row.get("fit_banks"),
                row.get("rank"),
                row.get("actual_rank_min"),
                row.get("actual_rank_max"),
                row.get("k_min"),
                row.get("k_max"),
                row.get("pX_nrmse"),
                row.get("v_nrmse"),
                row.get("pX_error_q95"),
                row.get("v_error_q95"),
                row.get("configuration_id"),
            ]
        )
    return display


def _cost_rows(rows, selected_ids, best_id):
    relevant = [row for row in rows if row.get("configuration_id") in selected_ids | {best_id}]
    return [
        [
            row.get(key)
            for key in (
                "configuration_id",
                "access_regime",
                "score_to_generation_ratio",
                "comparison_baseline",
                "calibration_cost",
                "direct_cost_four_queries",
                "break_even_queries",
                "tested_query_domain",
                "within_domain",
            )
        ]
        for row in relevant
    ]


def _known_summary(ledger):
    records = ledger.get("records", [])
    values = [_number(row.get("known_event_exact_error_max")) for row in records]
    values = [value for value in values if value is not None]
    return {
        "audited_blocks": len(values),
        "max_pX_v_absolute_error": max(values) if values else None,
        "direct_replay_verified_blocks": sum(
            row.get("direct_replay_matches_paid_packet") is True for row in records
        ),
        "physical_cost_fields": sorted(
            {key for row in records for key in row.get("known_event_four_evaluation_banks", {})}
        ),
    }


def _isolation_rows(evidence, run_root, parent_root):
    """Summarize sealed development-only predictions, with no forward or fit."""
    import numpy as np

    from src.modeling_qualification.math_contracts import helmert

    files = sorted((run_root / "N2/N2D").rglob("diagnostics.json"))
    if not files:
        return []
    metadata = evidence.read(parent_root / "probe_metadata.json")
    if metadata is None:
        return []
    groups = np.asarray([row["group"] for row in metadata])
    weights = np.asarray([row["weight"] for row in metadata])
    accumulated = {}
    for path in files:
        record = evidence.read(path)
        prediction_path = path.parent / "predictions.npz"
        digest = sha256_file(prediction_path)
        if (
            digest != record.get("prediction_sha256")
            or record.get("main_ranking_eligible") is not False
        ):
            raise ValueError("unbound or main-ranking N2D diagnostic predictions")
        evidence.inputs[str(prediction_path.resolve())] = digest
        with np.load(prediction_path, allow_pickle=False) as arrays:
            predictions, truth = arrays["prediction"], arrays["truth"]
        if truth.shape != (4, 2, 72, 4):
            raise ValueError("N2D truth shape mismatch")
        for item in record["records"]:
            if (
                item.get("development_only") is not True
                or item.get("main_ranking_eligible") is not False
            ):
                raise ValueError("N2D record lacks development-only scope")
            prediction = predictions[item["array_index"]].reshape(4, 2, 72, 3) @ helmert().T
            key = tuple(
                item[name]
                for name in ("observation", "method", "rank_cap", "diagnostic", "covariance_source")
            )
            aggregate = accumulated.setdefault(
                key,
                {
                    "blocks": 0,
                    "pX_error_ss": 0.0,
                    "v_error_ss": 0.0,
                    "pX_truth_ss": 0.0,
                    "v_truth_ss": 0.0,
                    "projector_distance_sum": 0.0,
                    "training_seeds": set(),
                },
            )
            aggregate["blocks"] += 1
            aggregate["training_seeds"].add(item["seed"])
            aggregate["projector_distance_sum"] += item["subspace_to_exact"][
                "projector_frobenius_distance"
            ]
            for group in range(6):
                mask = groups == group
                w = weights[mask] / weights[mask].sum()
                # Primary contrast only; aggregate prompts before squaring.
                pred = (prediction[:, 0, mask] * w[None, :, None]).sum(1)
                true = (truth[:, 0, mask] * w[None, :, None]).sum(1)
                for quantity, event in (("pX", 0), ("v", 3)):
                    aggregate[f"{quantity}_error_ss"] += float(
                        np.sum((pred[:, event] - true[:, event]) ** 2)
                    )
                    aggregate[f"{quantity}_truth_ss"] += float(np.sum(true[:, event] ** 2))
    rows = []
    for key, values in sorted(accumulated.items(), key=lambda item: str(item[0])):
        row = dict(
            zip(
                ("observation", "method", "rank_cap", "diagnostic", "covariance_source"),
                key,
                strict=True,
            )
        )
        row.update(
            blocks=values["blocks"],
            training_seed_count=len(values["training_seeds"]),
            projector_distance_mean=values["projector_distance_sum"] / values["blocks"],
            scope="PRIMARY_CONTRAST_LEGACY_DEVELOPMENT_ORACLE_ISOLATION_ONLY",
        )
        for quantity in ("pX", "v"):
            error, truth = values[f"{quantity}_error_ss"], values[f"{quantity}_truth_ss"]
            row[f"{quantity}_nrmse"] = math.sqrt(error / truth) if truth > 0 else None
        rows.append(row)
    return rows


# A passed test is usable only when its actual JUnit node is present in the final
# receipt; merely having this name in source is not execution evidence.
TEST_EVIDENCE = {
    "A5": [
        "test_output_root_and_symlink_escape",
        "test_run_writer_lock_resume_hash_and_no_clobber",
    ],
    "A6": ["test_cpu_environment"],
    "B1": ["test_parent_frozen_forward_and_known_scores_match_exact_parent"],
    "B2": ["test_helmert_orthogonality_roundtrip_and_distance", "test_nested_boundaries_roundtrip"],
    "B3": [
        "test_targets_cancel_origin_and_preserve_consistency",
        "test_contrast_inputs_use_same_bank_main_endpoint",
    ],
    "B4": [
        "test_shared_baseline_and_origin_cancellation",
        "test_actual_legacy_shared_baseline_produces_cross_bank_covariance",
    ],
    "B5": ["test_targets_cancel_origin_and_preserve_consistency"],
    "B6": ["test_state_hash_needs_full_state_and_never_merges_adam_histories"],
    "B7": [
        "test_parameter_equality_is_not_norm_equality",
        "test_state_hash_needs_full_state_and_never_merges_adam_histories",
    ],
    "C1": ["test_empirical_estimator_mean_and_covariance_match_finite_oracle"],
    "C2": ["test_crn_fixed_order_marginals_and_can_increase_variance"],
    "C3": ["test_crn_fixed_order_marginals_and_can_increase_variance"],
    "C4": ["test_packets_are_finite_and_aliases_exact_zero"],
    "C5": [
        "test_lr_exact_unbiased_quadratic_scaling_and_support",
        "test_missing_support_is_a_packet_failure",
    ],
    "C6": ["test_packet_identity_binds_labels_and_fit_evaluation_rng"],
    "C7": ["test_oracle_mixture_redundant_and_reverse_covariance_matches_packet_contract"],
    "C8": [
        "test_likelihood_raw_and_helmert_are_distinct_views",
        "test_raw_validity_keeps_mass_residual_instead_of_silently_using_control_variate",
    ],
    "C9": ["test_stable_difference_zero_near_equality_and_failure"],
    "C10": [
        "test_empirical_covariance_joint_order",
        "test_joint_origin_covariance_retains_shared_packet_terms",
    ],
    "C11": [
        "test_multinomial_jeffreys_is_covariance_only",
        "test_oracle_covariance_remains_separately_labeled_and_alias_rows_supported",
    ],
    "C12": ["test_zero_empirical_noise_never_gets_zero_width"],
    "C13": ["test_known_event_exact_scores_cached_and_no_generation"],
    "C14": [
        "test_same_access_cost_and_full_accounting_required",
        "test_logp_claim_requires_known_event_baseline_and_reports_conditional_cost",
    ],
    "C15": [
        "test_packet_reuse_does_not_increase_information_or_cost",
        "test_known_event_exact_scores_cached_and_no_generation",
    ],
    "D1": [
        "test_fit_only_uncentered_subspace_and_opposite_updates",
        "test_unexcited_fit_is_explicit_and_never_uses_test_updates",
    ],
    "D2": [
        "test_spherical_gls_is_scaled_ordinary_ridge",
        "test_joint_gls_matches_row_major_kronecker_objective",
    ],
    "D3": ["test_explicit_alias_row_selection_precedes_q_and_covariance_shrinkage"],
    "D4": [
        "test_zero_response_retains_nonzero_requested_direction",
        "test_grid_has_nonzero_ablations_and_no_full_cartesian_stage",
    ],
    "D5": ["test_c0_only_allows_zero_rank_and_nonzero_models_reject_it"],
    "D6": ["test_finite_unit_seals_before_oracle_and_does_not_select_on_query_excitation"],
    "D7": [
        "test_group_aware_spectrum_preserves_fixed_target_block_weights",
        "test_group_xv_map_matches_event_differences_with_fixed_weights",
    ],
    "D8": ["test_common_response_rank_one_witness_does_not_imply_contrast_fit"],
    "D9": ["test_error_isolation_returns_all_nonzero_ranks_and_three_distinct_sources"],
    "E1": ["test_bridge_preserves_split_and_bounds_parameter_only_inputs"],
    "E3": ["test_seed_bootstrap_pairs_whole_clusters_and_is_reproducible"],
    "E4": ["test_n3_freezes_at_most_two_with_content_hash_and_fixed_fresh_seeds"],
    "E5": [
        "test_prediction_is_saved_before_oracle_is_accessed",
        "test_finite_unit_seals_before_oracle_and_does_not_select_on_query_excitation",
    ],
    "E6": ["test_grid_has_nonzero_ablations_and_no_full_cartesian_stage"],
    "E7": ["test_pooled_nrmse_keeps_original_tiny_energy_and_separate_targets"],
    "E8": [
        "test_direction_replicas_do_not_manufacture_seed_or_unit_support",
        "test_direction_requires_both_support_thresholds",
    ],
    "E9": ["test_six_seed_calibration_cannot_claim_distribution_free_95_percent"],
    "E10": ["test_good_prediction_does_not_make_an_unknown_interval_resolved"],
    "F3": ["test_config_byte_and_semantic_freeze"],
    "F4": [
        "test_boundary_validation_and_input_arrays_are_not_modified",
        "test_autograd_jacobian_matches_parent_analytic_and_preserves_parameters",
    ],
    "F5": [
        "test_new_independent_evaluation_is_not_correlated_with_legacy_fit",
        "test_packet_identity_binds_labels_and_fit_evaluation_rng",
    ],
    "F6": ["test_run_writer_lock_resume_hash_and_no_clobber"],
}


def _acceptance(evidence, paths, audit, lock, tests, smoke, regression, config, supplemental):
    checklist = ROOT / "docs/modeling_contrast/design/CPU_ACCEPTANCE_CHECKLIST_zh.md"
    source = evidence.read(checklist, "text")
    group, index, rows = "", 0, []
    passed_nodes = list(tests["executed_passed_nodes"])
    parent_junit_sources = {}
    parent_receipt = evidence.read(paths["parent_regression"]) or {}
    for record in parent_receipt.get("records", []):
        if record.get("pytest_exit_code", record.get("exit_code")) != 0 or not record.get("junit"):
            continue
        path = Path(record["junit"])
        path = path if path.is_absolute() else ROOT / path
        source_text = evidence.read(path, "text")
        if source_text is None:
            continue
        for node in ET.fromstring(source_text).iter("testcase"):
            if all(node.find(tag) is None for tag in ("failure", "error", "skipped")):
                node_name = f"{node.get('classname', '')}::{node.get('name', '')}"
                passed_nodes.append(node_name)
                parent_junit_sources[node_name.split("::")[-1].split("[")[0]] = str(path)
    for line in source.splitlines():
        if line.startswith("## "):
            group, index = line[3], 0
        if not line.startswith("- [ ]"):
            continue
        index += 1
        ident, statement = f"{group}{index}", line[6:]
        names = TEST_EVIDENCE.get(ident, [])
        matches = [
            name
            for name in names
            if any(node.split("::")[-1].split("[")[0] == name for node in passed_nodes)
        ]
        status = "PASS" if names and len(matches) == len(names) else "NOT_RUN"
        detail = "; ".join(matches) if status == "PASS" else "缺少对应已执行测试节点回执"
        sources = [tests["source"], *tests["junit_sources"]] if names else []
        sources.extend(
            parent_junit_sources[name] for name in matches if name in parent_junit_sources
        )
        if ident in ("A1", "A2", "A3"):
            ok = audit.get("status") == "PASS"
            if ident == "A1":
                ok &= audit.get("parent_archive_verified") is True
            if ident == "A2":
                ok &= (
                    audit.get("complete_trajectories")
                    == config["parent_input"]["expected_legacy_trajectories"]
                )
            if ident == "A3":
                ok &= audit.get("historical_trajectories_regenerated") == 0
            status = "PASS" if ok else "BLOCKED"
            detail, sources = "原件逐文件哈希、身份、缺项与再生成计数", [paths["audit"]]
        elif ident in supplemental:
            item = supplemental[ident]
            status, detail, sources = item["status"], item["detail"], item["sources"]
        elif ident == "E2":
            seed = audit.get("seed_collision_audit", {})
            status = (
                "PASS"
                if seed.get("status") == "PASS" and not seed.get("colliding_seeds")
                else "BLOCKED"
            )
            detail, sources = "实际运行manifest的新seed冲突审计", [paths["audit"]]
        elif ident == "F1":
            status = "PASS" if smoke.get("resource_gate", {}).get("passed") is True else "BLOCKED"
            detail, sources = "实际CPU smoke及其资源外推；外推不替代全程实测", [paths["smoke"]]
        elif ident == "F7":
            status = (
                "PASS"
                if regression.get("all_scratch_removed") is True
                and regression.get("stop_reason") is None
                else "BLOCKED"
            )
            detail, sources = "分批回归临时目录和资源回执；RSS测量限制见原件", [paths["regression"]]
        elif ident == "F8":
            status = "PASS" if regression else "NOT_RUN"
            detail = "原失败/fixture错误保留，工程状态不标全绿；有范围排除须单列"
            sources = [paths["regression"]]
        if ident in ("E4", "E5", "F3") and not lock.get("allow_fresh_cpu"):
            detail += "；N4实际新轨迹未运行，不将单元测试称为新CPU独立验证"
        refs = "<br>".join(evidence.reference(path) for path in sources)
        rows.append([ident, statement, status, detail, refs or "未有独立证据"])
    heading = (
        "# CPU 验收逐项结果\n\n状态针对该条证据，不能把公式测试包或代码存在替代执行。"
        "PASS仅表示对应已记录检查通过；科学选择、独立验证、全仓库回归分别记录。\n\n"
    )
    return heading + _table(
        ["ID", "主规范原要求", "状态", "实际证据/限制", "evidence_path 与 SHA-256"], rows
    ), Counter(row[2] for row in rows)


def run_report(config, run_root, out, parent_root, *, project_root=ROOT):
    """Create report-owned files only; no overwrite and no N4 work is triggered."""
    run_root, parent_root = Path(run_root).resolve(), Path(parent_root).resolve()
    out = checked_output_path(out, project_root=project_root)
    # Preflight every owned destination before publishing even the first file.
    for filename in REPORTS:
        if (out / filename).exists():
            raise FileExistsError(f"report artifact already exists: {out / filename}")
    evidence = Evidence()
    paths = {
        "audit": run_root / "N0/parent_integrity_audit.json",
        "observation": run_root / "N1/OBSERVATION_COMPARISON.csv",
        "aggregate": run_root / "N2/AGGREGATE.csv",
        "by_seed": run_root / "N2/CONTRAST_PREDICTION_BY_SEED.csv",
        "cost": run_root / "N2/COST_FRONTIER.csv",
        "ledger": run_root / "N2/COST_LEDGER.json",
        "rules": run_root / "N2/frozen_nonzero_rules.json",
        "lock": run_root / "N3/MODEL_SELECTION_LOCK.json",
        "indicators": run_root / "N3/DECISION_INDICATORS.json",
        "pooled": run_root / "N3/CONTRAST_SELECTION_POOLED.csv",
        "bootstrap": run_root / "N3/PAIRED_BOOTSTRAP.json",
        "resolution": run_root / "N3/RESOLUTION.csv",
        "directions": run_root / "N3/DIRECTION_CURVES.csv",
        "smoke": run_root / "smoke/smoke_result.json",
        "parent_regression": run_root
        / "implementation/regression/statistics_regression_receipt.json",
        "regression": run_root / "implementation/regression_remaining/SUMMARY.json",
        "n1": run_root / "N1/stage_result.json",
        "n2": run_root / "N2/stage_result.json",
        "jacobian": run_root / "oracle_diagnostics/ORACLE_JACOBIAN_POOLED.csv",
        "cpu_cost": run_root / "cpu_cost_audit/CPU_COST_SUMMARY.json",
    }
    data = {
        key: evidence.read(path, "csv" if path.suffix == ".csv" else "json")
        for key, path in paths.items()
    }
    audit, lock, indicators = (data[key] or {} for key in ("audit", "lock", "indicators"))
    smoke, ledger = data["smoke"] or {}, data["ledger"] or {}
    cpu_cost = data["cpu_cost"] or {}
    regression, parent_regression = data["regression"] or {}, data["parent_regression"] or {}
    tests = _parse_tests(evidence, run_root)
    supplemental = _supplemental_audits(evidence, run_root, config)
    covariance_audit = _covariance_audit(evidence, run_root, parent_root)
    integrity = "PASS" if lock and verify_selection_lock(lock) else "FAIL" if lock else "NOT_RUN"
    authorized = (
        integrity == "PASS"
        and lock.get("allow_fresh_cpu") is True
        and all(
            _gate(lock, name).get("status") == "PASS"
            for name in (
                "science_gate",
                "cost_gate",
                "resource_gate",
                "input_identity_gate",
                "data_gate",
            )
        )
    )
    fresh = evidence.read(run_root / "N4/FRESH_COLLECTION_SUMMARY.json") if authorized else None
    validation_path = run_root / "N4_validation/INDEPENDENT_VALIDATION.json"
    if not validation_path.is_file():
        validation_path = run_root / "N4/INDEPENDENT_VALIDATION.json"
    validation = evidence.read(validation_path) if fresh else None
    validation_audit = _validation_audit(
        evidence, validation, validation_path, paths["lock"], lock.get("selected", [])
    )
    fresh_status = (
        (
            validation.get("status", "UNKNOWN")
            if validation and validation_audit["status"] == "PASS"
            else "BLOCKED_VALIDATION_INTEGRITY"
            if validation
            else "COLLECTED_NOT_EVALUATED"
        )
        if fresh
        else "NOT_RUN"
    )
    decision = lock.get("status", "NOT_RUN") if integrity != "FAIL" else "BLOCKED_LOCK_INTEGRITY"
    selected = lock.get("selected", []) if authorized else []
    best = lock.get("best_nonzero_diagnostic") or {}
    known = _known_summary(ledger)
    isolation = _isolation_rows(evidence, run_root, parent_root)
    isolation_stream = io.StringIO()
    isolation_writer = csv.DictWriter(
        isolation_stream, fieldnames=list(isolation[0]) if isolation else ["status"]
    )
    isolation_writer.writeheader()
    isolation_writer.writerows(isolation or [{"status": "NOT_RUN"}])
    isolation_text = isolation_stream.getvalue()
    isolation_digest = hashlib.sha256(isolation_text.encode()).hexdigest()
    all_regression_green = (
        parent_regression.get("status") == "PASS"
        and all(regression.get(key) == 0 for key in ("failures", "errors"))
        and regression.get("stop_reason") is None
        and regression.get("executed_files", 0) > 0
    )
    engineering = "PASS" if tests["status"] == "PASS" and all_regression_green else "BLOCKED"
    readiness = {
        "schema_version": "modeling-contrast-readiness-v2",
        "overall_decision": decision,
        "engineering_status": engineering,
        "new_module_tests": tests,
        "selection_lock_integrity": integrity,
        "fresh_cpu_status": fresh_status,
        "fresh_cpu_authorized": authorized,
        "fresh_cpu_results": fresh,
        "fresh_collection_status": fresh.get("status", "UNKNOWN") if fresh else "NOT_RUN",
        "recommended_surrogates": selected if validation_audit["recommendation_qualified"] else [],
        "frozen_development_candidates": selected,
        "independent_validation": validation,
        "independent_validation_audit": validation_audit,
        "recommendation_scope": "LEGACY_DEVELOPMENT_ONLY"
        if fresh_status == "NOT_RUN"
        else "SEE_FRESH_RECEIPT_SCOPE",
        "best_nonzero_diagnostic": best or None,
        "known_event_baseline": known,
        "oracle_isolation_summary_rows": len(isolation),
        "supplemental_acceptance_audits": supplemental,
        "measurement_covariance_audit_status": covariance_audit["status"],
        "actual_cpu_saving_claim_verified": supplemental["F2"]["actual_cpu_saving_claim_verified"],
        "recorded_cpu_components_audit": cpu_cost,
        **{
            name: _gate(lock, name)
            for name in (
                "science_gate",
                "cost_gate",
                "resource_gate",
                "input_identity_gate",
                "data_gate",
            )
        },
        "new_qwen_calls": indicators.get("new_qwen_calls"),
        "new_gpu_calls": indicators.get("new_gpu_calls"),
        "online_ssvc": indicators.get("online_ssvc"),
        "N1": data["n1"],
        "N2": data["n2"],
        "legacy_original_roles_preserved": True if audit.get("status") == "PASS" else None,
        "distribution_free_95_prediction_coverage_claim": False,
        "simultaneous_safety_certification": False,
    }
    primary = config["primary_target"]["name"]
    gates = [
        [name, readiness[name].get("status"), "; ".join(readiness[name].get("reasons", []))]
        for name in (
            "science_gate",
            "cost_gate",
            "resource_gate",
            "input_identity_gate",
            "data_gate",
        )
    ]
    selected_table = [
        [
            row.get("configuration_id"),
            row.get("model"),
            row.get("method"),
            row.get("n"),
            row.get("fit_banks"),
            row.get("actual_rank"),
            row.get("pX_nrmse"),
            row.get("v_nrmse"),
            row.get("actual_cost_saving_claim"),
        ]
        for row in selected
    ]
    scope = (
        f"原件完整轨迹 {_fmt(audit.get('complete_trajectories'))}/60；"
        f"N1 状态 `{(data['n1'] or {}).get('status', 'NOT_RUN')}`，"
        f"轨迹 {_fmt((data['n1'] or {}).get('trajectories'))}；"
        f"N2 状态 `{(data['n2'] or {}).get('status', 'NOT_RUN')}`，"
        f"exact 轨迹 {_fmt((data['n2'] or {}).get('exact_legacy_trajectories'))}，"
        f"finite 轨迹 {_fmt((data['n2'] or {}).get('finite_development_trajectories'))}。"
        "所有旧seed均为legacy_diagnostic，旧locked-test只作公开开发/复现，原role保留。"
    )
    design = f"""## 固定目标、语义及计费范围

主目标 `{primary}`；次目标 `no_x_off_1_minus_joint_0`；一致性目标是两者之差。
72个固定题目、X/S/W/I四事件存储，拟合216个Helmert响应坐标。
参数空间737维；局部输入子空间维数k及所选r分别记录，不能将3或216解释为动态秩。
同bank的输入为候选终点减joint_0终点的e；T、B、C分别记录。
共同起点计数在C中消去；共享baseline、跨bank历史alias和同packet相关项仍保留。

fit banks 0-7、diagnostic 8-9、evaluation 10-13绑定所有候选。
主n=64，16/256是冻结稳健性；8个noise replicas不增加训练seed数。
O-IND历史fit计数已有5份原副本，其余3份为新增观测；evaluation使用独立新观测。
其余测量的RNG按seed/arm/anchor/bank/n/replica/purpose固定分离。
训练bank样本与control观测分开，eval语义概率在预测文件封存后才用于评分。

成本列包含旧观测购置成本、生成、标签、指定动作评分、缓存/alias复用与拟合；
scenario中评分/生成比为0.05/0.25/1，标签/生成比为0.05。CPU时间另列，不等于真实VLM时间。
surrogate新增查询成本按0计是乐观下界；收支平衡仅在已验证的4个evaluation bank内评价。
"""
    recommendation = (
        "冻结的有限观测候选如下。N3是开发门禁；独立CPU泛化须另看N4实际回执。\n\n"
        + _table(
            [
                "配置",
                "模型",
                "观测",
                "n",
                "fit banks",
                "r",
                "pX NRMSE",
                "v NRMSE",
                "操作计数情景收益标记",
            ],
            selected_table,
        )
        if selected
        else "**没有获准部署的局部surrogate配置；不将best_nonzero_diagnostic当成合格推荐。**\n"
    )
    if known["audited_blocks"]:
        recommendation += (
            f"\n在允许已知完整动作归一化logp评分的当前toy，保留直接"
            f" `D-KNOWN-EVENT-LOGP` 作为pX/v测量建议：每策略每题最多3个标量评分，"
            f"同一次16动作forward可复用，生成样本n=0、无需拟合k/r。"
            f"实际检查{known['audited_blocks']}块，最大pX/v绝对误差"
            f"{_fmt(known['max_pX_v_absolute_error'])}。"
            "此建议只适用于已知X/I动作集合的有限toy，不扩展为真实VLM事件概率等式。\n"
        )
    else:
        recommendation += (
            "\n已知事件直接评分基线的实际成本/误差回执尚缺，不能宣称该基线已通过验证。\n"
        )
    if best:
        recommendation += (
            "\n最优非零开发诊断（不是部署推荐）："
            f"配置 `{best.get('configuration_id')}`，模型 `{best.get('model')}`，"
            f"观测 `{best.get('method')}`，n={_fmt(best.get('n'))}，"
            f"fit banks={_fmt(best.get('fit_banks'))}，r={_fmt(best.get('actual_rank'))}，"
            f"k范围={_fmt(best.get('k_min'))}-{_fmt(best.get('k_max'))}，"
            f"pX/v NRMSE={_fmt(best.get('pX_nrmse'))}/{_fmt(best.get('v_nrmse'))}。\n"
        )
    finite_rows = data["pooled"] or data["aggregate"] or []
    limit = """## 失效范围与不能推出的结论

- 当前证据固定数据seed20260915、初始化7001、737参数、16动作、两个奖励臂；
  只覆盖旧训练RNG与固定bank查询范围。
- SAMPLE_ONLY中CRN额外要求可控制采样随机数；普通黑盒API不自动具备该权限。
  共同随机数可能提高方差，必须看实测行。
- LR依赖完整归一化动作概率、正确proposal和支持覆盖；
  未观察到X、经验零方差及大权重不能证明事件无变化。
- 无fit差异激励、未覆盖更新方向、局部非线性及低信号保留失效状态；active-only结果不替代all-bank结果。
- n16/256及fit-bank消融只改冻结的一项；所有非零消融保留实际rank/norm/hash；r0仅为C0基线。
- N3选模仅有4个旧selection训练seed，方向门槛要求至少5个seed/20个anchor-bank单元；
  不足则为UNKNOWN/INSUFFICIENT。
- 新校准规划仅6个seed，不支持95%分布无关seed级预测覆盖；
  noise replicas、题目、锚点均不能补充训练独立性。
- 点估计、点态正态近似区间、群体误差分位、同时区间、预测区间和工程门禁分别命名；
  没有安全认证或在线SSVC结论。
"""
    decision_text = (
        f"# 建模第二轮决定\n\n**N3决定：`{decision}`。工程总状态：`{engineering}`。"
        f"冻结锁完整性：`{integrity}`；N4实际状态：`{fresh_status}`。**\n\n"
        + scope
        + "\n\n"
        + _table(["门禁", "状态", "原因"], gates)
        + "\n来源："
        + evidence.reference(paths["lock"])
        + "；"
        + evidence.reference(paths["indicators"])
        + "\n\n## 推荐与有限条件\n\n"
        + recommendation
        + "\n"
        + design
        + "\n## 四种观测的实际比较\n\n主目标、Helmert视图，按误差/真值平方和汇总群体NRMSE；"
        "原始事件LR另保留在完整表。\n\n"
        + _table(
            [
                "观测",
                "n",
                "pX NRMSE",
                "v NRMSE",
                "理论方差均值",
                "经验方差均值",
                "经验/理论",
                "未观测X次数",
            ],
            _observation_rows(data["observation"] or [], primary),
        )
        + "\n来源："
        + evidence.reference(paths["observation"])
        + "\n\n## 目标与方向模型的实际比较\n\n"
        "展示每模型/观测/n在公开开发资料中max(pX,v NRMSE)最小的代表行；"
        "这张诊断表不改变N3冻结选择。所有被淘汰配置仍在完整表，r_min/r_max包含无激励状态。\n\n"
        + _table(
            [
                "模型",
                "观测",
                "n",
                "fit banks",
                "rank cap",
                "r_min",
                "r_max",
                "k_min",
                "k_max",
                "pX NRMSE",
                "v NRMSE",
                "pX q95",
                "v q95",
                "配置ID",
            ],
            _model_rows(finite_rows, primary),
        )
        + "\n来源："
        + evidence.reference(paths["pooled"] if data["pooled"] else paths["aggregate"])
        + "；逐seed："
        + evidence.reference(paths["by_seed"])
        + "\n\n## 收支平衡与权限\n\n仅展示冻结候选及best_nonzero诊断；"
        "完整成本前沿保留全部配置。\n\n"
        + _table(
            [
                "配置",
                "权限",
                "评分/生成",
                "直接基线",
                "校准成本",
                "4查询直接成本",
                "平衡查询数",
                "受测域",
                "在域内",
            ],
            _cost_rows(
                data["cost"] or [],
                {row.get("configuration_id") for row in selected},
                best.get("configuration_id"),
            ),
        )
        + "\n来源："
        + evidence.reference(paths["cost"])
        + "；"
        + evidence.reference(paths["ledger"])
        + "\n\n上述成本前沿按操作计数、给定评分/生成价格比及零增量查询成本下界比较；"
        "它们是有条件的收支情景，不能直接称为实际CPU或费用节约。"
        + f"实际同条件CPU节约证据通过核验：{readiness['actual_cpu_saving_claim_verified']}。"
        + f"全阶段成本审计 `{supplemental['F2']['status']}`。\n\n来源："
        + evidence.reference(run_root / "implementation/TOTAL_COST_AUDIT.json")
        + "\n\n### 已记录CPU组件的实际比较\n\n"
        + f"实测审计 `{cpu_cost.get('status', 'NOT_RUN')}`，"
        + f"selection单元 {_fmt(cpu_cost.get('unit_count'))}，"
        + f"冷拟合次数 {_fmt(cpu_cost.get('unique_fit_evaluations'))}，"
        + f"比较行 {_fmt(cpu_cost.get('comparison_rows'))}，"
        + f"冷拟合CPU合计 {_fmt(cpu_cost.get('cold_fit_cpu_seconds_total'))} 秒，"
        + f"首次query CPU合计 {_fmt(cpu_cost.get('first_query_cpu_seconds_total'))} 秒。"
        + "以下仅列所选及best_nonzero配置；单位为秒，全部是737参数toy。"
        "缺少历史O_IND获取CPU及部分packet构建调度成本，不能据此主张完整CPU节省。\n\n"
        + _table(
            ["配置", "单元", "冷拟合CPU均值", "4查询首次CPU均值", "4查询warm CPU中位数"],
            [
                [
                    row.get(key)
                    for key in (
                        "configuration_id",
                        "unit_count",
                        "cold_fit_cpu_seconds_mean",
                        "four_query_first_cpu_seconds_mean",
                        "four_query_warm_cpu_seconds_median",
                    )
                ]
                for row in cpu_cost.get("by_configuration", [])
                if row.get("configuration_id")
                in {
                    best.get("configuration_id"),
                    *(candidate.get("configuration_id") for candidate in selected),
                }
            ],
        )
        + "\n来源："
        + evidence.reference(paths["cpu_cost"])
        + "\n\n## 方向、区间与独立验证\n\n"
        + f"N4 `{fresh_status}`；许可为{authorized}，许可不代表实验已执行。"
        + "旧结果不能充当新独立验证。方向曲线、bootstrap与分辨率表不用于事后更换测试规则。\n\n"
        + _table(
            ["独立验证项目", "实际状态"],
            [
                ["回执/预测/评分文件hash及selected配置覆盖", validation_audit["status"]],
                ["独立预测门禁", validation_audit.get("independent_science_status", "NOT_RUN")],
                ["独立分辨率门禁", validation_audit.get("resolution_status", "NOT_RUN")],
                ["满足独立推荐条件", validation_audit["recommendation_qualified"]],
            ],
        )
        + ("\n来源：" + evidence.reference(validation_path) + "\n\n" if validation else "\n")
        + "；".join(
            evidence.reference(paths[name]) for name in ("bootstrap", "resolution", "directions")
        )
        + "\n\n"
        + "## 观测、方向、系数与局部非线性的隔离\n\n"
        + "下表仅使用旧development数据的已封存预测，分别置换精确方向、精确系数与精确协方差。"
        "它们属于oracle诊断，不能作为有限观测配置的成绩或实际运行推荐。\n\n"
        + _table(
            [
                "观测",
                "模型",
                "r",
                "方向/系数来源",
                "协方差来源",
                "pX NRMSE",
                "v NRMSE",
                "投影矩阵距离均值",
                "训练seed数",
            ],
            [
                [
                    row.get(key)
                    for key in (
                        "observation",
                        "method",
                        "rank_cap",
                        "diagnostic",
                        "covariance_source",
                        "pX_nrmse",
                        "v_nrmse",
                        "projector_distance_mean",
                        "training_seed_count",
                    )
                ]
                for row in isolation
            ],
        )
        + "\n派生表：[DIAGNOSTIC_ISOLATION_SUMMARY.csv]"
        + f"({out / 'DIAGNOSTIC_ISOLATION_SUMMARY.csv'})；"
        + f"SHA-256 `{isolation_digest}`。"
        + "每个输入diagnostics.json及raw预测的路径/hash见REPORT_SOURCE_BINDING.json。\n\n"
        + "完整Jacobian在起点和baseline终点的一阶预测只评估局部非线性，均为oracle。\n\n"
        + _table(
            ["位置", "pX NRMSE", "v NRMSE", "训练seed数"],
            [
                [row.get("method"), row.get("pX_nrmse"), row.get("v_nrmse"), row.get("seed_count")]
                for row in (data["jacobian"] or [])
                if row.get("target") == primary
                and row.get("bank_role") == "heldout"
                and row.get("population") == "all"
            ],
        )
        + "\n来源："
        + evidence.reference(paths["jacobian"])
        + "\n\n"
        + limit
    )
    implementation = (
        f"# 实现与验证记录\n\n新模块测试状态 `{tests['status']}`，通过数 {_fmt(tests['passed'])}，"
        f"失败 {_fmt(tests['failures'])}，错误 {_fmt(tests['errors'])}。"
        f"工程总状态 `{engineering}`。\n\n"
        f"父建模/数学回归状态 `{parent_regression.get('status', 'UNKNOWN')}`，"
        f"总测试 {_fmt(parent_regression.get('total_tests'))}。其他旧回归：通过"
        f" {_fmt(regression.get('tests_passed'))}，失败 {_fmt(regression.get('failures'))}，"
        f"错误 {_fmt(regression.get('errors'))}；没有把旧失败删掉后声称全套通过。\n\n"
        + "来源："
        + evidence.reference(tests["source"])
        + "；"
        + evidence.reference(paths["parent_regression"])
        + "；"
        + evidence.reference(paths["regression"])
        + "\n\n## 明确排除和缺项\n\n"
        + "实际排除文件/理由：`"
        + json.dumps(regression.get("excluded_files"), ensure_ascii=False)
        + "`。\n\n"
        + "实际deselected节点：`"
        + json.dumps(regression.get("deselected_nodes"), ensure_ascii=False)
        + "`。"
        + "两个tiny Qwen测试涉及真实新模型调用，sealed-confirm节点不读封存真实测试响应；"
        "其他委派回归由独立回执说明。\n\n"
        + "旧回归sealed-confirm实际读取："
        + f"{regression.get('actual_sealed_confirm_read', 'UNKNOWN')}。"
        + "错误原因须以原始日志为准；缺fixture不是本轮方法失败，也不是工程全绿。\n\n"
        + "## CPU实际运行及外推\n\n"
        + _table(
            ["项目", "实际回执值"],
            [
                ["smoke观测秒", smoke.get("observation_wall_seconds")],
                ["smoke拟合秒", smoke.get("fit_wall_seconds")],
                ["smoke packet bytes", smoke.get("packet_bytes")],
                ["smoke fit bytes", smoke.get("fit_bytes")],
                ["N1 wall seconds", (data["n1"] or {}).get("wall_seconds")],
                ["N2 wall seconds", (data["n2"] or {}).get("wall_seconds")],
                ["N2新增optimizer updates", (data["n2"] or {}).get("new_optimizer_updates")],
            ],
        )
        + "\n资源外推（不是全程实测）：`"
        + json.dumps(smoke.get("forecast"), ensure_ascii=False)
        + "`。\n\n"
        + "来源："
        + evidence.reference(paths["smoke"])
        + "；"
        + evidence.reference(paths["n1"])
        + "；"
        + evidence.reference(paths["n2"])
        + "\n\n## 数据与代码边界\n\n"
        + scope
        + "\n\n父来源："
        + evidence.reference(paths["audit"])
        + "\n\nparent服务、有限packet、拟合输入和oracle评分分开。"
        "联合packet保留共享baseline、alias跨bank与CRN/LR共同贡献协方差。"
        + "原5份O-IND计数不改写；补充副本和evaluation独立观测保留来源字段。"
        + "旧源码保留，完整状态hash不可得时输出缺项，不把参数hash代替Adam/RNG状态。\n\n"
        + "## N1元数据勘误及C1辅助修正\n\n"
        + f"协方差勘误核验 `{covariance_audit['N1_metadata_erratum']['status']}`；"
        + covariance_audit["N1_metadata_erratum"]["correction"]
        + f" C1辅助修正 `{covariance_audit['C1_auxiliary_repair']['status']}`，"
        + f"验证overlay数 {covariance_audit['C1_auxiliary_repair']['verified_overlay_count']}。"
        + covariance_audit["C1_auxiliary_repair"]["scope"]
        + " N1原始hash保持，逐文件及103 X_VALID anchor24 bank0/1的NPZ证据见"
        + f"[MEASUREMENT_COVARIANCE_AUDIT.json]({out / 'MEASUREMENT_COVARIANCE_AUDIT.json'})。\n\n"
        + "所有本报告读取文件及实际SHA-256见REPORT_SOURCE_BINDING.json。"
        "报告文件manifest仅包含报告自产文件；源码commit、push和源码总manifest由最终交付步骤提供。\n"
        + "\n## 复算源码清单的范围\n\n"
        + "独立raw复算后仅目录provenance发生变化的已知未执行报告/编排模块数："
        + str(len(supplemental["F10"].get("source_catalog_changes_after_replay", [])))
        + "。若非零，readiness逐项保留SOURCE_CATALOG_CHANGED_AFTER_REPLAY及旧/新hash；"
        "recompute、packet、parent、observations、targets等真实依赖与所有raw文件仍严格比对原hash，"
        "这项例外不扩大科学复算覆盖范围。\n"
    )
    idea = (
        "# 修订后的建模对象\n\n本轮把总体响应T、主响应B及候选差异C分别命名，"
        "直接用e拟合C，同时保留父d→T后相减、零基线及同权限直接测量。"
        "候选差异共享起点噪声严格消去，但共同baseline和测量packet相关项仍需联合协方差。\n\n"
        "O-IND、CRN、共同样本LR和候选混合是既有统计工具；C5/C6为透明两阶段投影，"
        "不将IS/CRN/GLS/RRR/PCA认领为新理论。总体低秩好不保证低能量关键差异好；"
        "216是观测坐标数，k/r由固定fit bank激励决定。\n\n"
        f"本轮实际开发决定 `{decision}`；N4 `{fresh_status}`。"
        "方法是否值得部署依据MODELING_DECISION_V2_zh.md及N3锁，"
        "不预设需要更复杂的预测器。\n\n来源：" + evidence.reference(paths["lock"]) + "\n"
    )
    requirements = """# 下一阶段最小真实资料需求

当前没有发起新Qwen/GPU调用、训练、原点补采样或在线SSVC。以下是资料需求，不是执行授权。

1. 固定prompt/场景/接口/答案映射/parser/解码配置与版本hash，以及逐题权重、
   完整动作身份与X/S/W/I标签规则。
2. 同一起点、同训练bank下joint_0/joint_1/no_x_off_1的真实参数增量e与d，
   以及参数布局、参数hash、Adam/gradient/RNG完整状态指纹或明确缺项。
3. 独立于训练bank的control样本，完整输出序列、实际采样分布、proposal_id、draw_id、
   source_policy、normalized sequence logp、候选交叉评分和支持检查。
4. 明确普通采样、可控制共同随机数、指定序列logp三种访问权限；不能从四类计数反推出动作似然，
   也不能假定真实VLM支持toy的逆CDF耦合。
5. 真实X/I事件通常含多个输出序列：先核验事件集合是否已知且可枚举，
   规范答案字符串的单一概率不能直接当pX。
6. 原始训练seed/arm/anchor/bank角色、测量replica与fit/eval packet独立性；
   测试行为标签在预测封存后才可评分。
7. 实测生成、交叉评分、标签验证、候选更新、归一化forward及缓存成本；
   给出新bank查询域，不把同一packet重复读取当独立信息。

新数据、初始化和真实模型的泛化不在本轮已验证范围内。任何后续真实模型调用须另行授权并制定独立协议。
"""
    acceptance, counts = _acceptance(
        evidence, paths, audit, lock, tests, smoke, regression, config, supplemental
    )
    readiness["acceptance_status_counts"] = dict(counts)
    output_text = {
        "MODELING_DECISION_V2_zh.md": decision_text,
        "IMPLEMENTATION_REPORT_zh.md": implementation,
        "REVISED_IDEA_zh.md": idea,
        "REAL_DATA_REQUIREMENTS_zh.md": requirements,
        "CPU_ACCEPTANCE_RESULTS_zh.md": acceptance,
    }
    for filename, text in output_text.items():
        write_text(out / filename, text, project_root=project_root)
    write_text(out / "DIAGNOSTIC_ISOLATION_SUMMARY.csv", isolation_text, project_root=project_root)
    write_json(
        out / "MEASUREMENT_COVARIANCE_AUDIT.json", covariance_audit, project_root=project_root
    )
    write_json(out / "machine_readable_readiness.json", readiness, project_root=project_root)
    binding = {
        "inputs": dict(sorted(evidence.inputs.items())),
        "missing_inputs": sorted(evidence.missing),
        "parent_root": str(parent_root),
        "run_root": str(run_root),
        "rendering_performs_no_experiments": True,
        "new_module_tests_source": str((run_root / "implementation/final_tests.json").resolve()),
    }
    write_json(out / "REPORT_SOURCE_BINDING.json", binding, project_root=project_root)
    output_hashes = {
        name: sha256_file(out / name) for name in REPORTS if name != "REPORT_MANIFEST.json"
    }
    result = {
        "status": "REPORT_GENERATED",
        "overall_decision": decision,
        "fresh_cpu_status": fresh_status,
        "engineering_status": engineering,
        "output_root": str(out),
        "outputs": output_hashes,
        "missing_input_count": len(evidence.missing),
    }
    write_json(out / "REPORT_MANIFEST.json", result, project_root=project_root)
    return result
