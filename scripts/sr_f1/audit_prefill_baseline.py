#!/usr/bin/env python3
"""Audit the immutable Claude handoff using its authenticated original scorer.

This reads saved engineering responses only. It does not load a tokenizer/model,
generate responses, or score any TEST records. The character-level truncation is
a descriptive replay of old outputs, not a measurement of the prefill protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "SR_F1_CLAUDE_FACTS_AND_CODE_20261010"
DEFAULT_BUNDLE = PROJECT / "artifacts/sr_f1_claude_handoff/stage_02" / BUNDLE_NAME
DEFAULT_OUTPUT = (
    PROJECT / "artifacts/sr_f1_amendment_20261010/PREFILL_BASELINE_REGRESSION_FULL.json"
)
DEFAULT_SUMMARY = PROJECT / "docs/sr_f1/validation/PREFILL_BASELINE_REGRESSION_20261010.json"
PANELS = {"before": 256, "confirm": 256, "after": 113}
COVERAGE_FIELDS = ("L_json", "L_answer", "L_evidence")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_cut(text):
    """The solution's literal string/escape-aware brace scanner, without repair."""
    if not text.startswith("{"):
        return None
    depth, in_string, escaped = 0, False, False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def scanner_self_checks():
    cases = [
        ("", None),
        (" {}", None),
        ("{}", 2),
        ("{} trailing", 2),
        ('{"nested":{"x":1}}', 18),
        ('{"text":"} {"}', 14),
        ('{"text":"\\"}"}', 14),
        ('{"text":"\\\\"}', 13),
        ('{"x":{"y":1}', None),
        ('{"x":"unterminated}', None),
    ]
    for text, expected in cases:
        observed = balanced_cut(text)
        if observed != expected:
            raise AssertionError((text, expected, observed))
    return len(cases)


def covered(score):
    return all(score[field] == 1 for field in COVERAGE_FIELDS)


def category(score):
    for field, name in (
        ("L_json", "whole_json_invalid"),
        ("L_answer", "answer_not_scorable"),
        ("L_evidence", "evidence_not_scorable"),
    ):
        if not score[field]:
            return name
    return "covered"


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2))
    margin /= 1 + z * z / total
    return [center - margin, center + margin]


def failure_details(contract, text, result):
    """Add observed parse/type/duplicate details; leave scorer outputs unchanged."""
    details = {}
    try:
        obj = contract.load_json_strict(text)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        return {"parser_exception_type": type(exc).__name__, "parser_exception": str(exc)}
    if not isinstance(obj, dict):
        return {"root_type": type(obj).__name__}
    if not result["L_answer"] and "answer" in obj:
        answer = obj["answer"]
        details["answer_type"] = type(answer).__name__
        details["answer_repr"] = repr(answer)
        try:
            contract.fraction_of(answer, text=True)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            details["answer_exception_type"] = type(exc).__name__
            details["answer_exception"] = str(exc)
    evidence = obj.get("evidence")
    if isinstance(evidence, list) and not result["L_evidence"]:
        seen = {}
        duplicates = []
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                continue
            key = (item.get("series"), item.get("category"))
            if not all(isinstance(part, str) for part in key):
                continue
            if key in seen:
                duplicates.append(
                    {"entity": list(key), "first_index": seen[key], "repeated_index": index}
                )
            else:
                seen[key] = index
        if duplicates:
            details["duplicate_evidence_entities"] = duplicates
            details["official_duplicate_count_until_first_rejection"] = result["duplicate_count"]
    return details


def audit(bundle, solution):
    bundle = bundle.resolve()
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())["files"]
    checked = {}

    def checked_file(relative):
        path = bundle / relative
        actual_hash = sha256(path)
        expected = manifest[relative]
        if path.stat().st_size != expected["bytes"] or actual_hash != expected["sha256"]:
            raise ValueError("Snapshot manifest mismatch: " + relative)
        checked[relative] = actual_hash
        return path

    wrapper_relative = "code/src/sr_f1/contract.py"
    reference_relative = "code/docs/sr_f1/package/reference/semantic_contract.py"
    package_manifest_relative = "code/docs/sr_f1/package/PACKAGE_SHA256.json"
    wrapper_path = checked_file(wrapper_relative)
    checked_file(reference_relative)
    package_manifest_path = checked_file(package_manifest_relative)
    package_manifest = json.loads(package_manifest_path.read_text())
    if checked[reference_relative] != package_manifest["reference/semantic_contract.py"]:
        raise ValueError("Reference scorer differs from uploaded authenticated package")
    spec = importlib.util.spec_from_file_location("prefill_audit_frozen_contract", wrapper_path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    self_check_count = scanner_self_checks()

    panel_records = {}
    needed_qids = set()
    for panel, expected_count in PANELS.items():
        prefix = f"evidence/server/engineering/format/{panel}/"
        records = []
        for relative in sorted(manifest):
            if not relative.startswith(prefix) or not relative.endswith(".json"):
                continue
            stem = Path(relative).stem
            if not stem.rsplit("-", 1)[-1].isdigit():
                continue
            value = json.loads(checked_file(relative).read_text())
            payload = {key: item for key, item in value.items() if key != "record_hash"}
            if value["record_hash"] != contract.digest(payload):
                raise ValueError("Raw record hash mismatch: " + relative)
            if value["sample_index"] not in range(8):
                raise ValueError("Original snapshot draw outside [0, 8): " + relative)
            records.append((relative, value))
            needed_qids.add(value["qid"])
        if len(records) != expected_count:
            raise ValueError(f"Immutable snapshot count mismatch: {panel}: {len(records)}")
        panel_records[panel] = records

    tasks_relative = "code/docs/sr_f1/package/manifests/TASKS_GOLD_AUDIT_ONLY.jsonl"
    tasks_path = checked_file(tasks_relative)
    if checked[tasks_relative] != package_manifest["manifests/TASKS_GOLD_AUDIT_ONLY.jsonl"]:
        raise ValueError("Tasks differ from uploaded authenticated package")
    tasks = {}
    # The manifest is an original bundled contract artifact. Only engineering qids
    # referenced by the saved FORMAT/FORMAT_CONFIRM records enter the audit.
    with tasks_path.open() as handle:
        for line in handle:
            task = json.loads(line)
            if task["qid"] in needed_qids:
                if task["qid"] in tasks:
                    raise ValueError("Duplicate engineering qid in task manifest")
                tasks[task["qid"]] = task
    if set(tasks) != needed_qids:
        raise ValueError("Missing engineering tasks in authenticated manifest")
    if any(not qid.startswith(("format-", "format_confirm-")) for qid in tasks):
        raise ValueError("Non-FORMAT task in regression")

    panels = {}
    rows = []
    raw_record_index = []
    for panel, records in panel_records.items():
        count = Counter()
        original_categories = Counter()
        cut_categories = Counter()
        suffix_categories = Counter()
        for relative, record in records:
            text = record["raw_text"]
            task = tasks[record["qid"]]
            original = contract.score(text, task["world"], task["query"])
            count["responses"] += 1
            count["original_raw_covered"] += int(covered(original))
            count["truncated"] += int(record["truncated"])
            original_categories[category(original)] += 1
            common = {
                "panel": panel,
                "file": relative,
                "file_sha256": checked[relative],
                "record_hash": record["record_hash"],
                "qid": record["qid"],
                "sample_index": record["sample_index"],
                "raw_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            raw_record_index.append(
                {
                    **common,
                    "starts_with_brace": text.startswith("{"),
                    "original_category": category(original),
                }
            )
            if not text.startswith("{"):
                continue
            count["brace_start_responses"] += 1
            cut = balanced_cut(text)
            cut_text = None if cut is None else text[:cut]
            suffix = None if cut is None else text[cut:]
            nonwhite = suffix is not None and bool(suffix.strip())
            cut_score = (
                None if cut is None else contract.score(cut_text, task["world"], task["query"])
            )
            cut_category = "unbalanced" if cut is None else category(cut_score)
            suffix_category = "trailing_after_close" if nonwhite else cut_category
            cut_ok = cut_score is not None and covered(cut_score)
            count["balanced"] += int(cut is not None)
            count["unbalanced"] += int(cut is None)
            count["balanced_object_only_covered"] += int(cut_ok)
            count["nonwhite_suffix_in_entire_legacy_output"] += int(nonwhite)
            count["balanced_and_entire_legacy_suffix_clean_covered"] += int(cut_ok and not nonwhite)
            cut_categories[cut_category] += 1
            suffix_categories[suffix_category] += 1
            rows.append(
                {
                    **common,
                    "task_sha256": contract.digest(task),
                    "raw_text": text,
                    "finish_reason": record["finish_reason"],
                    "truncated": record["truncated"],
                    "original_score": original,
                    "original_category": category(original),
                    "balanced_cut_char": cut,
                    "balanced_text": cut_text,
                    "legacy_suffix": suffix,
                    "legacy_suffix_has_nonwhitespace": nonwhite,
                    "balanced_object_only_score": cut_score,
                    "balanced_object_only_category": cut_category,
                    "entire_legacy_suffix_check_category": suffix_category,
                    "failure_details": (
                        {"balanced_scanner_result": None}
                        if cut is None
                        else failure_details(contract, cut_text, cut_score)
                        if not cut_ok
                        else {}
                    ),
                    "new_protocol_final_token_suffix_check": "NOT_EVALUATED_FROM_LEGACY_FULL_TEXT",
                }
            )
        panels[panel] = {
            **dict(count),
            "original_coverage": count["original_raw_covered"] / count["responses"],
            "conditional_balanced_coverage": count["balanced_object_only_covered"]
            / count["brace_start_responses"],
            "original_categories": dict(sorted(original_categories.items())),
            "balanced_object_only_categories": dict(sorted(cut_categories.items())),
            "entire_legacy_suffix_check_categories": dict(sorted(suffix_categories.items())),
        }

    denominator = len(rows)
    numerator = sum(
        covered(row["balanced_object_only_score"])
        for row in rows
        if row["balanced_object_only_score"] is not None
    )
    failing = [row["file"] for row in rows if row["balanced_object_only_category"] != "covered"]
    solution_hash = sha256(solution)
    return {
        "schema_version": 1,
        "status": "AUDIT_COMPLETE_DOCUMENT_NUMERATOR_NOT_REPRODUCED",
        "source_bundle": bundle.name,
        "source_bundle_manifest_sha256": sha256(manifest_path),
        "solution_document": {
            "file_name": solution.name,
            "sha256": solution_hash,
            "claimed_numerator": 239,
            "claimed_denominator": 247,
        },
        "audit_script_sha256": sha256(__file__),
        "execution": {
            "read_only_inputs": True,
            "model_calls": 0,
            "tokenizer_loads": 0,
            "new_generations": 0,
            "test_records_scored": 0,
            "scanner_self_checks_passed": self_check_count,
        },
        "scorer": {
            "wrapper_file": wrapper_relative,
            "wrapper_sha256": checked[wrapper_relative],
            "reference_file": reference_relative,
            "reference_sha256": checked[reference_relative],
            "uploaded_reference_authentication": "PASS",
            "coverage_rule": "L_json == 1 and L_answer == 1 and L_evidence == 1",
            "answer_type_and_duplicate_evidence_rules": "UNCHANGED",
        },
        "snapshot_counts": PANELS,
        "panels": panels,
        "combined_brace_start": {
            "responses": denominator,
            "balanced_object_only_covered": numerator,
            "coverage": numerator / denominator,
            "wilson_95_interval": wilson(numerator, denominator),
            "entire_legacy_suffix_clean_covered": sum(
                row["entire_legacy_suffix_check_category"] == "covered" for row in rows
            ),
            "unbalanced": sum(row["balanced_cut_char"] is None for row in rows),
            "failed_record_files": failing,
        },
        "claim_comparison": {
            "denominator_reproduced": denominator == 247,
            "numerator_reproduced": numerator == 239,
            "observed_minus_claimed_numerator": numerator - 239,
            "answer_type_failure": (
                "evidence/server/engineering/format/before/format-0011-line-v0-6.json"
            ),
            "duplicate_evidence_failure": (
                "evidence/server/engineering/format/confirm/format_confirm-0008-line-v0-1.json"
            ),
            "author_analysis_script_available_at_named_download_path": (
                solution.parent / "f1_analyze.py"
            ).is_file(),
            "author_stopping_script_available_at_named_download_path": (
                solution.parent / "balanced_json_stop.py"
            ).is_file(),
            "author_approximate_scorer_not_reexecuted": True,
        },
        "scope_notes": [
            "These are existing conditional brace-start outputs, "
            "not newly generated prefill outputs.",
            "Before has 256 records, confirm has 256, after is the delivered partial snapshot "
            "of 113; later after records are excluded.",
            "No leading prose or code fence is removed; only strings starting literally "
            "with { enter the 247-record subset.",
            "The entire legacy suffix check is reported separately. It cannot identify "
            "the proposed stop-completing token's suffix without tokenizer boundary replay "
            "and is not new-protocol certification.",
            "A null brace cut is unbalanced. A nonnull brace cut alone does not establish "
            "valid JSON or strict format coverage.",
            "The original task manifest hash is authenticated; only selected "
            "FORMAT/FORMAT_CONFIRM tasks are scored.",
        ],
        "checked_input_files": checked,
        "raw_record_index": raw_record_index,
        "brace_start_rows": rows,
    }


def compact_summary(result, full_output):
    """Keep raw answers in the ignored full receipt, not the tracked summary."""
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"checked_input_files", "raw_record_index", "brace_start_rows"}
    }
    try:
        full_name = full_output.resolve().relative_to(PROJECT).as_posix()
    except ValueError:
        full_name = str(full_output.resolve())
    summary["full_receipt"] = {
        "path": full_name,
        "sha256": sha256(full_output),
        "bytes": full_output.stat().st_size,
        "contains_all_625_raw_record_identifiers": True,
        "contains_247_brace_start_raw_answers": True,
        "raw_answers_in_this_summary": False,
    }
    summary["verified_input_file_count"] = len(result["checked_input_files"])
    rows = []
    identity_fields = (
        "panel",
        "file",
        "file_sha256",
        "record_hash",
        "qid",
        "sample_index",
        "raw_text_sha256",
        "finish_reason",
        "truncated",
        "original_category",
        "balanced_cut_char",
        "legacy_suffix_has_nonwhitespace",
        "balanced_object_only_category",
        "entire_legacy_suffix_check_category",
        "new_protocol_final_token_suffix_check",
    )
    flags = (*COVERAGE_FIELDS, "duplicate_count", "reason")
    for row in result["brace_start_rows"]:
        compact = {key: row[key] for key in identity_fields}
        compact["original_official_flags"] = {key: row["original_score"][key] for key in flags}
        cut_score = row["balanced_object_only_score"]
        compact["balanced_official_flags"] = (
            None if cut_score is None else {key: cut_score[key] for key in flags}
        )
        compact["error_messages"] = {
            key: value
            for key, value in row["failure_details"].items()
            if key
            in {
                "parser_exception_type",
                "parser_exception",
                "answer_exception_type",
                "answer_exception",
                "official_duplicate_count_until_first_rejection",
                "balanced_scanner_result",
            }
        }
        rows.append(compact)
    summary["brace_start_rows"] = rows
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--solution", type=Path, default=Path("/Users/louis/Downloads/SR-F1问题分析与解决方案.md")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    result = audit(args.bundle, args.solution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    summary = compact_summary(result, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": str(args.summary),
                "status": result["status"],
                "combined_brace_start": result["combined_brace_start"],
                "panels": result["panels"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
