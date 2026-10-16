#!/usr/bin/env python3
"""Read-only audit of the frozen SR-F3-CE ENGINE and existing MONITOR observations.

Reproduce the original scorer and independently extract exact scalar values.
No model loading, inference, changed thresholds, or raw-file writes.

For an offline bundle, --repo names its frozen_source directory (no Git checkout
required). Supply --plan, --prepared, and --frozen-source-manifest to use the
bundle's original plan, rendered images, and deployment manifest. Every manifest
file is checked against the run-bound original inventory before source imports.
The Git-backed invocation and its original plan/image defaults remain available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from decimal import Decimal, DecimalException
from fractions import Fraction
from pathlib import Path, PurePosixPath

COMMIT = "71483fbf31d57213fdc7e6f3c881848397b45329"
SOURCE_FILES = (
    "src/sr_f3_ce/contract.py",
    "src/sr_f3_ce/evaluation.py",
    "src/sr_f3_ce/runtime.py",
    "src/sr_f3_ce/runner.py",
    "src/sr_f3_ce/prompts.py",
    "src/sr_f3_ce/data.py",
    "src/sr_f12/evaluation.py",
    "src/sr_f12/protocol.py",
    "src/sr_f1/json_protocol.py",
    "src/mm_core/vl_runtime.py",
    "src/mm_core/training.py",
    "src/sr_f3_ce/checkpoint.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), sha(raw)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def manifest_inventory(document):
    """Validate portable paths and identities without trusting last-key-wins maps."""
    if not isinstance(document, dict) or document.get("code_commit") != COMMIT:
        raise PermissionError("Source manifest must identify the frozen 714 commit")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source manifest requires a nonempty file inventory")
    inventory = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("Source inventory entries require path, bytes, and sha256")
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or "\x00" in relative
            or PurePosixPath(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))
        ):
            raise PermissionError("Source inventory path escapes its root")
        if relative in inventory:
            raise ValueError("Duplicate source inventory path")
        if (
            type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is None
        ):
            raise ValueError("Malformed source file identity")
        inventory[relative] = row
    return inventory


def source_bytes(repo, relative):
    path = (repo / relative).resolve(strict=True)
    if not path.is_relative_to(repo) or not path.is_file():
        raise PermissionError("Source file escapes its registered root")
    return path.read_bytes()


def verify_sources(repo, snapshot, frozen_source_manifest=None):
    """Bind imported source to the original deployment, with or without Git."""
    source_manifest, source_manifest_sha = read_json(snapshot / "deployment/SOURCE_MANIFEST.json")
    source_items = manifest_inventory(source_manifest)
    run_identity, _ = read_json(snapshot / "RUN_IDENTITY.json")
    if (
        run_identity.get("code_commit") != COMMIT
        or run_identity.get("bound_files", {}).get("deployment/SOURCE_MANIFEST.json")
        != source_manifest_sha
    ):
        raise PermissionError("Original source manifest is not bound by the frozen run identity")
    if not set(SOURCE_FILES).issubset(source_items):
        raise PermissionError("Source inventory omits an audit dependency")
    sources = {}
    if frozen_source_manifest is None:
        for relative in SOURCE_FILES:
            frozen = subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=repo)
            live = source_bytes(repo, relative)
            if frozen != live:
                raise PermissionError("Working source differs from original 714: " + relative)
            if source_items[relative]["sha256"] != sha(frozen) or source_items[relative][
                "bytes"
            ] != len(frozen):
                raise PermissionError("Original source manifest differs: " + relative)
            sources[relative] = sha(frozen)
        verification = dict(mode="git_show_fixed_commit", checked_files=len(sources))
    else:
        portable_manifest, portable_sha = read_json(frozen_source_manifest)
        portable_items = manifest_inventory(portable_manifest)
        # The optional manifest is not its own authority. It must be the same
        # original inventory already bound by snapshot/RUN_IDENTITY.json.
        if portable_sha != source_manifest_sha or portable_manifest != source_manifest:
            raise PermissionError("Portable source manifest differs from original deployment")
        for relative, expected in portable_items.items():
            raw = source_bytes(repo, relative)
            if len(raw) != expected["bytes"] or sha(raw) != expected["sha256"]:
                raise PermissionError("Frozen source file identity differs: " + relative)
            if relative in SOURCE_FILES:
                sources[relative] = sha(raw)
        if any(
            path.is_file() for suffix in ("*.pyc", "*.pyo") for path in (repo / "src").rglob(suffix)
        ):
            raise PermissionError("Frozen source contains unbound Python bytecode")
        actual_source_files = {
            str(path.relative_to(repo)) for path in (repo / "src").rglob("*") if path.is_file()
        }
        if not actual_source_files.issubset(portable_items):
            raise PermissionError("Frozen source contains an unregistered source file")
        verification = dict(
            mode="run_bound_frozen_source_manifest",
            checked_files=len(portable_items),
            frozen_source_manifest_sha256=portable_sha,
        )
    return source_manifest_sha, sources, verification


def audit_paths(args):
    repo, snapshot = args.repo.resolve(strict=True), args.snapshot.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(snapshot):
        raise PermissionError("Audit output must not alter the original snapshot")
    plan = (args.plan or repo / "artifacts/sr_f3_ce_20261016/plan").resolve()
    prepared = (args.prepared or repo / "artifacts/sr_f3_ce_20261016/prepared").resolve()
    return repo, snapshot, output, plan, prepared


class ParseFailure(ValueError):
    def __init__(self, category, detail):
        super().__init__(detail)
        self.category = category
        self.detail = detail


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ParseFailure("duplicate_json_key", f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def strict_json(text):
    def invalid_constant(value):
        raise ParseFailure("nonfinite_json_number", value)

    if len(text) > 64000:
        raise ParseFailure("json_text_bound", "more than 64000 characters")
    try:
        return json.loads(
            text,
            parse_float=Decimal,
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except json.JSONDecodeError as exc:
        raise ParseFailure("json_syntax", f"{exc.msg} at character {exc.pos}") from exc
    except (DecimalException, OverflowError, RecursionError) as exc:
        raise ParseFailure("json_numeric_or_recursion_bound", str(exc)) from exc


DECIMAL_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,3})?\Z")
FRACTION_PATTERN = re.compile(r"([+-]?[0-9]+)\s*/\s*([+-]?[0-9]+)\Z")


def exact_scalar(value):
    """Independent exact-domain extraction; intentionally does not salvage values."""
    if value is None:
        return None
    if type(value) is bool:
        raise ParseFailure("invalid_answer_scalar", "boolean is not a scalar")
    if isinstance(value, int):
        number = Fraction(value)
    elif isinstance(value, Decimal):
        if not value.is_finite() or abs(value.adjusted()) > 30:
            raise ParseFailure("invalid_answer_scalar", "JSON decimal outside bounds")
        number = Fraction(value)
    elif isinstance(value, str):
        if len(value) > 64:
            raise ParseFailure("invalid_answer_scalar", "scalar string exceeds 64 characters")
        token = value.strip().replace("\u2212", "-")
        matched = FRACTION_PATTERN.fullmatch(token)
        if matched:
            denominator = int(matched[2])
            if not denominator:
                raise ParseFailure("invalid_answer_scalar", "zero denominator")
            number = Fraction(int(matched[1]), denominator)
        elif DECIMAL_PATTERN.fullmatch(token):
            decimal = Decimal(token)
            if decimal and abs(decimal.adjusted()) > 30:
                raise ParseFailure("invalid_answer_scalar", "string decimal outside bounds")
            number = Fraction(decimal)
        else:
            raise ParseFailure("invalid_answer_scalar", "string is not one exact scalar")
    else:
        raise ParseFailure(
            "invalid_answer_scalar", f"unsupported scalar type: {type(value).__name__}"
        )
    if abs(number.numerator) > 10**12 or number.denominator > 10**9:
        raise ParseFailure("invalid_answer_scalar", "rational outside registered bounds")
    return number


def object_boundary(text):
    """Independent lexical brace machine: first initial object only, escaped quotes aware."""
    offset = len(text) - len(text.lstrip())
    if offset >= len(text) or text[offset] != "{":
        return None
    stack, inside, escaped = 0, False, False
    for position in range(offset, len(text)):
        char = text[position]
        if inside:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                inside = False
        else:
            if char == '"':
                inside = True
            elif char == "{":
                stack += 1
            elif char == "}":
                stack -= 1
                if stack == 0:
                    return position + 1
    return None


def independent_extract(text, protocol):
    result = dict(
        value_parseable=0,
        answer=None,
        failure_category=None,
        failure_detail=None,
        object_keys=None,
        boundary=None,
    )
    try:
        if protocol == "natural_work":
            if text.count("<final>") != 1 or text.count("</final>") != 1:
                raise ParseFailure(
                    "natural_tag_count", "exactly one opening and closing final tag required"
                )
            start = text.index("<final>") + len("<final>")
            finish = text.index("</final>")
            if finish < start:
                raise ParseFailure("natural_tag_order", "closing tag precedes designated value")
            result["boundary"] = finish + len("</final>")
            value = text[start:finish].strip()
            if value.startswith('"'):
                value = strict_json(value)
                if not isinstance(value, str):
                    raise ParseFailure(
                        "invalid_answer_scalar", "quoted final is not a scalar string"
                    )
            elif value == "null":
                value = None
            answer = exact_scalar(value)
        else:
            end = object_boundary(text)
            result["boundary"] = end
            if end is None:
                initial = text.lstrip().startswith("{")
                raise ParseFailure(
                    "unclosed_initial_object" if initial else "missing_initial_object",
                    "no complete object at the beginning of the preserved output",
                )
            obj = strict_json(text[:end])
            if not isinstance(obj, dict):
                raise ParseFailure("nonobject_root", "JSON root is not an object")
            result["object_keys"] = list(obj)
            if "answer" not in obj:
                raise ParseFailure(
                    "missing_answer_field", "first complete object has no answer field"
                )
            answer = exact_scalar(obj["answer"])
        result.update(value_parseable=1, answer=answer)
    except ParseFailure as exc:
        result.update(failure_category=exc.category, failure_detail=exc.detail)
    return result


def self_check():
    cases = [
        ('{"answer":2}', "structured", 1, Fraction(2), None),
        ('{"answer":"98/3"}', "structured", 1, Fraction(98, 3), None),
        ('{"answer":null}', "answer_only", 1, None, None),
        ('{"answer":1e-2}', "answer_only", 1, Fraction(1, 100), None),
        ('{"answer":1,"work":"}\\"x"}tail', "structured", 1, Fraction(1), None),
        ('{"answer":2,"answer":3}', "structured", 0, None, "duplicate_json_key"),
        ('{"mean":2}', "structured", 0, None, "missing_answer_field"),
        ('{"answer":2', "structured", 0, None, "unclosed_initial_object"),
        ('junk {"answer":2}', "structured", 0, None, "missing_initial_object"),
        ('{"answer":98/3}', "structured", 0, None, "json_syntax"),
        ('{"answer":true}', "structured", 0, None, "invalid_answer_scalar"),
        ('{"answer":"2 apples"}', "structured", 0, None, "invalid_answer_scalar"),
        ('{"answer":"1/0"}', "structured", 0, None, "invalid_answer_scalar"),
        ('{"answer":NaN}', "structured", 0, None, "nonfinite_json_number"),
        ('work <final>"98/3"</final>tail', "natural_work", 1, Fraction(98, 3), None),
        ("<final>null</final>", "natural_work", 1, None, None),
        ("<final>2</final><final>2</final>", "natural_work", 0, None, "natural_tag_count"),
        ("</final><final>2", "natural_work", 0, None, "natural_tag_order"),
        ("<final>2 cats</final>", "natural_work", 0, None, "invalid_answer_scalar"),
    ]
    for text, protocol, parseable, answer, failure in cases:
        actual = independent_extract(text, protocol)
        assert (actual["value_parseable"], actual["answer"], actual["failure_category"]) == (
            parseable,
            answer,
            failure,
        ), (text, actual)
    return dict(status="PASS", cases=len(cases))


def stop_audit(row):
    """Check original stored token/decoded metadata; this is not a fresh tokenizer replay."""
    issues, text, protocol = [], row["raw_text"], row["protocol"]
    if protocol == "natural_work":
        position = text.find("</final>")
        boundary = position + len("</final>") if position >= 0 else None
    else:
        boundary = object_boundary(text)
    for field in ("full_text", "full_decoded_text"):
        if row[field] != text:
            issues.append(field + "_differs")
    if row["boundary_char"] != boundary:
        issues.append("boundary_char_differs")
    tokens, backend = row["tokens"], row["backend_sequence_tokens"]
    n, offsets = len(tokens), row["token_prefix_character_ends"]
    if backend[:n] != tokens or row["backend_padding_count"] != len(backend) - n:
        issues.append("backend_sequence_prefix_differs")
    config = row["sampling_parameters"]
    eos_ids = config["eos_token_id"]
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    eos_at = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), None)
    if eos_at is not None and eos_at != n:
        issues.append("token_after_eos")
    if any(token != config["pad_token_id"] for token in backend[n:]):
        issues.append("nonpad_backend_after_kept")
    if len(offsets) != n or offsets[-1] != len(text):
        issues.append("prefix_offset_count_or_final_length_differs")
    prefix_replay = bool(row["prefix_character_offsets_stable"])
    first_n = None
    if prefix_replay:
        for i, end in enumerate(offsets):
            fragment = text[:end]
            found = (
                "</final>" in fragment
                if protocol == "natural_work"
                else object_boundary(fragment) is not None
            )
            if found:
                first_n = i + 1
                break
        expected_finish = (
            ("final_tag" if protocol == "natural_work" else "balanced")
            if first_n is not None
            else "eos"
            if eos_at is not None
            else "length"
        )
        if first_n is not None and first_n != n:
            issues.append("continued_after_first_boundary_token")
        if row["finish_reason"] != expected_finish:
            issues.append("finish_reason_differs_from_stored_token_prefix")
        expected_n = n if first_n == n else None
        if row["boundary_token_count"] != expected_n:
            issues.append("boundary_token_count_differs")
        for budget in (128, 256, 512, 1024):
            view = text[: offsets[min(budget, n) - 1]]
            if row["decoded_prefix_views"][str(budget)] != view:
                issues.append(f"decoded_prefix_view_{budget}_differs")
    if row["finish_reason"] == "eos" and eos_at != n:
        issues.append("eos_reason_without_eos_token")
    if row["finish_reason"] == "length" and n != 1024:
        issues.append("length_stop_not_1024")
    # The inherited native runtime hash uses JSON's default spaced separators.
    sampling_hash = sha(json.dumps(config, sort_keys=True, default=str).encode())
    if row["sampling_hash"] != sampling_hash:
        issues.append("sampling_hash_differs")
    for key, expected in dict(
        do_sample=True,
        temperature=1,
        top_p=1,
        top_k=0,
        max_new_tokens=1024,
        num_beams=1,
        repetition_penalty=1,
        output_logits=True,
        output_scores=True,
    ).items():
        if config.get(key) != expected:
            issues.append("sampling_recipe_" + key + "_differs")
    return dict(
        issues=issues,
        stored_prefix_replay=prefix_replay,
        first_boundary_token=first_n,
        eos_token_position=eos_at,
    )


def summarize(rows):
    lengths = sorted(row["tokens"] for row in rows)

    def counters(key):
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    invalid = [r for r in rows if not r["independent_value_parseable"]]
    return dict(
        records=len(rows),
        value_parseable=sum(r["independent_value_parseable"] for r in rows),
        value_correct=sum(r["independent_value_correct"] for r in rows),
        score_totals={
            key: sum(row["score"][key] for row in rows)
            for key in (
                "presentation",
                "finish",
                "certificate",
                "certificate_truth",
                "certificate_sufficient",
                "J",
            )
        },
        failure_categories=dict(sorted(Counter(r["failure_category"] for r in invalid).items())),
        failure_details=dict(sorted(Counter(r["failure_detail"] for r in invalid).items())),
        failure_by_finish=dict(sorted(Counter(r["finish_reason"] for r in invalid).items())),
        finish_reasons=counters("finish_reason"),
        format_protocol_errors=counters("format_protocol_error"),
        token_count=dict(
            min=lengths[0],
            median=statistics.median(lengths),
            max=lengths[-1],
            total=sum(lengths),
            at_1024=lengths.count(1024),
        ),
        stored_prefix_replay_rows=sum(r["stop_audit"]["stored_prefix_replay"] for r in rows),
    )


def audit_existing_monitor_and_cache(
    snapshot, output, tasks, store, identity_sha, reference_hash, evaluation, contract, data
):
    """Audit existing auxiliary engineering observations, not additional experiments."""
    from mm_core.training import state_hash

    differences, stops, monitor_rows, cache_rows, files = [], [], [], [], []
    gold = {qid: data.independently_answer(t["world"], t["query"]) for qid, t in tasks.items()}

    def check_row(row, task, score, saved, path, index, *, phase, group_id, block, step, input_sha):
        relative = str(path.relative_to(snapshot))
        if score != saved:
            differences.append(
                dict(
                    kind="original_scorer_vs_saved",
                    raw_file=relative,
                    row_index=index,
                    computed=score,
                    saved=saved,
                )
            )
        extracted = independent_extract(row["raw_text"], "structured")
        correct = int(
            bool(extracted["value_parseable"]) and extracted["answer"] == gold[task["qid"]]
        )
        if (extracted["value_parseable"], correct) != (
            score["value_parseable"],
            score["value_correct"],
        ):
            differences.append(
                dict(kind="independent_value_vs_original", raw_file=relative, row_index=index)
            )
        stop = stop_audit(row)
        if stop["issues"]:
            stops.append(dict(raw_file=relative, row_index=index, **stop))
        return dict(
            phase=phase,
            block=block,
            step=step,
            protocol="structured",
            view=None,
            group_id=group_id,
            row_index=index,
            qid=task["qid"],
            family=task["family"],
            raw_file=relative,
            raw_text_sha256=sha(row["raw_text"].encode()),
            input_hash=input_sha,
            tokens=len(row["tokens"]),
            finish_reason=row["finish_reason"],
            format_protocol_error=row["format_protocol_error"],
            score=score,
            independent_value_parseable=extracted["value_parseable"],
            independent_value_correct=correct,
            failure_category=extracted["failure_category"],
            failure_detail=extracted["failure_detail"],
            object_keys=extracted["object_keys"],
            stop_audit=stop,
        )

    monitor_root = snapshot / "monitor"
    assert sorted(str(p.relative_to(monitor_root)) for p in monitor_root.glob("*/step-*")) == [
        "SRF2_J6_s71002/step-00"
    ]
    panel = monitor_root / "SRF2_J6_s71002/step-00"
    slots = evaluation.monitor_slots(tasks, "SRF2_J6_s71002", 0)
    groups, hashes, lightweight_rows = evaluation.group_slots(slots), [], []
    coverage, coverage_sha = read_json(panel / "COVERAGE.json")
    complete, complete_sha = read_json(panel / "COMPLETE.json")
    assert len(slots) == 512 and len(groups) == 64
    assert {p.name for p in (panel / "groups").glob("*") if p.is_file()} == {
        g["group_id"] + ".json" for g in groups
    }
    assert {p.name for p in (panel / "scores").glob("*") if p.is_file()} == {
        g["group_id"] + ".json" for g in groups
    }
    for group in groups:
        path = panel / "groups" / (group["group_id"] + ".json")
        raw, raw_sha = read_json(path)
        saved, score_sha = read_json(panel / "scores" / path.name)
        assert raw["group_id"] == group["group_id"] and len(raw["records"]) == 8
        assert raw["records_sha256"] == canonical_hash(raw["records"])
        assert (
            saved["raw_sha256"] == raw_sha and saved["scorer_version"] == evaluation.EXPERIMENT_ID
        )
        assert len(saved["scores"]) == 8
        saved_by_slot = {r["slot_id"]: r["score"] for r in saved["scores"]}
        assert len(saved_by_slot) == 8
        request = store.model_input(group["slots"][0]["qid"], "structured")
        input_sha = canonical_hash(request)
        assert len({r["prompt_tensor_hash"] for r in raw["records"]}) == 1
        for index, (row, slot) in enumerate(zip(raw["records"], group["slots"], strict=True)):
            assert row["pool"] == "MONITOR" and row["phase"] == "MONITOR" and row["sealed"] is False
            assert row["model_input"] == request and row["prompt"] == request["prompt"]
            evaluation.validate_record(
                row,
                slot,
                input_hash=input_sha,
                model_identity=row["model_identity"],
                adapter_identity=row["adapter_identity"],
                source_sha256=identity_sha,
                index=index,
                count=8,
            )
            assert row["model_identity"]["parent_id"] == "SRF2_J6_s71002"
            assert row["adapter_identity"]["adapter_parameter_hash"] == coverage["policy_hash"]
            score = evaluation.score_record(row, tasks[row["qid"]])
            monitor_rows.append(
                check_row(
                    row,
                    tasks[row["qid"]],
                    score,
                    saved_by_slot[row["slot_id"]],
                    path,
                    index,
                    phase="MONITOR_STEP0",
                    group_id=group["group_id"],
                    block=1,
                    step=0,
                    input_sha=input_sha,
                )
            )
            lightweight_rows.append(
                dict(
                    slot,
                    generation_status=row["generation_status"],
                    technical_validation_errors=row["technical_validation_errors"],
                    score=score,
                )
            )
        hashes.append(dict(group_id=group["group_id"], sha256=raw_sha))
        files.append(
            dict(path=str(path.relative_to(snapshot)), sha256=raw_sha, score_sha256=score_sha)
        )
    valid = evaluation.validate_coverage(lightweight_rows, slots)
    assert all(coverage[key] == valid[key] for key in valid)
    assert coverage["groups"] == hashes and coverage["records_sha256"] == canonical_hash(hashes)
    assert all(complete[key] == value for key, value in coverage.items())
    assert complete["families"] == evaluation.summarize_scores(lightweight_rows)
    monitor_receipt = dict(
        status="PASS",
        new_output_rows=512,
        groups=64,
        coverage_sha256=coverage_sha,
        complete_sha256=complete_sha,
        score_schema="evaluation.score_record",
        summary=summarize(monitor_rows),
    )

    resume = snapshot / "engine/resume"
    rollout_dir = resume / "continuous/rollouts"
    expected_names = {
        f"{step:02d}-{slot:02d}{suffix}"
        for step in range(1, 5)
        for slot in range(16)
        for suffix in (".json", ".request.json", ".pending.jsonl")
    }
    assert {p.name for p in rollout_dir.iterdir() if p.is_file()} == expected_names
    assert not list((resume / "split/rollouts").glob("*"))
    run = dict(
        run_id="SRF3CE_ENGINE_EXEC_b0",
        arm="EXEC",
        block=0,
        future_seed=2026101601,
        parent_id="SRF2_J6_s71001",
        updates=4,
        stage="ENGINE",
        experiment_id="SR-F3-CE-20261016",
        run_identity_sha256=identity_sha,
        reference_hash=reference_hash,
    )
    chain = canonical_hash([])
    step3and4_rows, pending_duplicates = 0, 0
    for step in range(1, 5):
        scheduled = sorted(
            (
                t
                for t in tasks.values()
                if t["pool"] == "ENGINE" and t["root_id"].endswith(f"_{step - 1:04d}")
            ),
            key=lambda t: t["qid"],
        )
        assert len(scheduled) == 16
        support_qids = [
            t["qid"] for t in scheduled if t["mode"] == "repair" and t["chart"] == "grouped_bar"
        ]
        assert len(support_qids) == 4
        group_hashes = []
        for slot, task in enumerate(scheduled):
            path = rollout_dir / f"{step:02d}-{slot:02d}.json"
            raw, raw_sha = read_json(path)
            request_identity, request_sha = read_json(path.with_suffix(".request.json"))
            pending_path = path.with_suffix(".pending.jsonl")
            pending_raw = pending_path.read_bytes()
            pending = [json.loads(line) for line in pending_raw.splitlines()]
            assert len(raw["records"]) == len(pending) == 8
            assert raw["records_sha256"] == canonical_hash(raw["records"])
            assert raw["identity"] == request_identity
            request = store.model_input(task["qid"], "structured")
            input_sha = canonical_hash(request)
            seeds = [
                int(
                    canonical_hash(
                        ["SR-F3-CE-20261016", "engine", 2026101601, step, task["qid"], i]
                    )[:16],
                    16,
                )
                % (2**31 - 1)
                for i in range(8)
            ]
            expected = dict(
                run_identity_sha256=canonical_hash(run),
                step=step,
                slot=slot,
                qid=task["qid"],
                seeds=seeds,
                input_hash=input_sha,
                task_hash=canonical_hash(task),
            )
            assert all(request_identity[k] == v for k, v in expected.items())
            for index, (row, pending_row) in enumerate(zip(raw["records"], pending, strict=True)):
                assert row["qid"] == task["qid"] and row["protocol"] == "structured"
                assert row["model_input"] == request and row["prompt"] == request["prompt"]
                assert (
                    row["generation_status"] == "COMPLETE"
                    and not row["technical_validation_errors"]
                )
                assert row["group_row_index"] == index and row["seed"] == seeds[index]
                assert row["group_seed"] == seeds[0] and row["group_size"] == 8
                assert (
                    row["tokens"] == row["raw_tokens"]
                    and len(row["tokens"])
                    == len(row["sampler_logprobs"])
                    == row["completion_token_count"]
                )
                assert 0 < len(row["tokens"]) <= 1024 and all(
                    type(t) is int and t >= 0 for t in row["tokens"]
                )
                assert all(
                    isinstance(p, (int, float))
                    and not isinstance(p, bool)
                    and math.isfinite(p)
                    and p <= 1e-5
                    for p in row["sampler_logprobs"]
                )
                assert (
                    row["truncated"] is (row["finish_reason"] == "length")
                    and row["prompt_tensor_hash"]
                )
                assert (
                    row["assistant_prefill"] == "{"
                    and row["raw_text"] == "{" + row["generated_text"]
                )
                assert row["model_identity"]["parent_id"] == "SRF2_J6_s71001"
                assert (
                    row["adapter_identity"]["adapter_parameter_hash"]
                    == request_identity["policy_hash"]
                )
                score = contract.score(
                    row["full_text"],
                    task["world"],
                    task["query"],
                    finish_reason=row["finish_reason"],
                    protocol="structured",
                )
                expected_committed = dict(
                    pending_row,
                    model_input=request,
                    root=row["root"],
                    qid=task["qid"],
                    score=score,
                    reward=score["J"],
                )
                assert expected_committed == row
                assert row["reward"] == row["score"]["J"]
                cache_rows.append(
                    check_row(
                        row,
                        task,
                        score,
                        row["score"],
                        path,
                        index,
                        phase="ENGINE_RECOVERY_CACHE",
                        group_id=f"{step:02d}-{slot:02d}",
                        block=0,
                        step=step,
                        input_sha=input_sha,
                    )
                )
                pending_duplicates += 1
                step3and4_rows += int(step >= 3)
            group_hashes.append(raw_sha)
            files.append(
                dict(
                    path=str(path.relative_to(snapshot)),
                    sha256=raw_sha,
                    request_sha256=request_sha,
                    pending_sha256=sha(pending_raw),
                    pending_rows=8,
                    pending_is_same_observation=True,
                )
            )
        support_hashes = []
        for qid in support_qids:
            support_path = resume / "continuous/support" / f"{step:02d}-{qid}.json"
            support, support_sha = read_json(support_path)
            assert support["qid"] == qid and support["step"] == step and support["arm"] == "EXEC"
            support_hashes.append(support_sha)
        chain = canonical_hash([chain, step, group_hashes, support_hashes])
        if step in (2, 4):
            for segment in ("continuous", "split"):
                commit, _ = read_json(resume / segment / "checkpoints" / f"commit-{step:02d}.json")
                assert commit["field_hashes"]["committed_records_hash"] == state_hash(chain)
                assert commit["field_hashes"]["run_identity"] == state_hash(run)
    evidence, evidence_sha = read_json(snapshot / "tasks/engine--resume/EVIDENCE.json")
    assert evidence["evidence"]["new_rollouts"] == len(cache_rows) == 512
    assert evidence["evidence"]["replayed_rollouts"] == step3and4_rows == 256
    initialization, initialization_sha = read_json(resume / "split/INITIALIZATION.json")
    continuous2, _ = read_json(resume / "continuous/checkpoints/commit-02.json")
    assert (
        initialization["source_receipt"] == continuous2
        and initialization["copied_logical_step"] == 2
    )
    cache_receipt = dict(
        status="PASS",
        new_output_rows=512,
        groups=64,
        pending_copy_rows_not_new_output=pending_duplicates,
        replay_rows_not_new_output=step3and4_rows,
        orphan_pending_rows=0,
        no_split_generation_files=True,
        committed_records_chain=chain,
        evidence_sha256=evidence_sha,
        initialization_sha256=initialization_sha,
        score_schema="contract.score (no certificate_applicable wrapper field)",
        summary=summarize(cache_rows),
    )
    with (output / "AUXILIARY_ROW_AUDIT.jsonl").open("w") as stream:
        for row in monitor_rows + cache_rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
    write_json(output / "AUXILIARY_RAW_FILES_VERIFIED.json", files)
    write_json(
        output / "AUXILIARY_DIFFERENCES.json", dict(scoring=differences, stop_metadata=stops)
    )
    return dict(
        status="AUDIT_COMPLETE_NO_DIFFERENCES"
        if not differences and not stops
        else "AUDIT_COMPLETE_WITH_DIFFERENCES",
        monitor=monitor_receipt,
        recovery_cache=cache_receipt,
        score_differences=len(differences),
        stop_metadata_differences=len(stops),
        actual_new_output_rows=1024,
        cached_replay_rows_counted_as_new=0,
    )


def main(argv=None):
    if not __debug__:
        raise RuntimeError("Audit validation requires Python without -O or -OO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, help="Original plan directory; defaults to the repo artifact path"
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Frozen rendered-image directory; defaults to the repo artifact path",
    )
    parser.add_argument(
        "--frozen-source-manifest",
        type=Path,
        help="Use the run-bound original deployment manifest instead of git show",
    )
    args = parser.parse_args(argv)
    repo, snapshot, output, plan, prepared = audit_paths(args)
    source_manifest_sha, sources, source_verification = verify_sources(
        repo, snapshot, args.frozen_source_manifest
    )
    output.mkdir(parents=True, exist_ok=False)
    checks = self_check()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "src"))
    from sr_f3_ce import contract, data, evaluation

    package = data.verify_package(plan)
    tasks = {}
    for line in (plan / "manifests/TASKS_AUDIT_ONLY.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["pool"] in ("ENGINE", "MONITOR"):
            assert row["qid"] not in tasks
            tasks[row["qid"]] = row
    assert len(tasks) == 128
    store = data.ModelInputStore.from_files(
        plan / "manifests/MODEL_INPUTS.jsonl",
        prepared / "IMAGE_MANIFEST.jsonl",
        image_root=prepared,
    )
    slots = evaluation.engine_slots(tasks)
    assert len(slots) == 2880 and all(s["pool"] == "ENGINE" and not s["sealed"] for s in slots)
    gold = {qid: data.independently_answer(t["world"], t["query"]) for qid, t in tasks.items()}
    assert all(gold[qid] == contract.execute(t["world"], t["query"]) for qid, t in tasks.items())
    identity, identity_sha = read_json(snapshot / "RUN_IDENTITY.json")
    assert identity["code_commit"] == COMMIT
    assert identity["package_manifest_sha256"] == package["package_manifest_sha256"]
    assert identity["bound_files"]["deployment/SOURCE_MANIFEST.json"] == source_manifest_sha
    assert identity["bound_files"]["data/IMAGE_MANIFEST.jsonl"] == sha(
        (prepared / "IMAGE_MANIFEST.jsonl").read_bytes()
    )
    all_audit, diffs, stop_issues, input_hashes, raw_files = [], [], [], {}, []
    model_hashes, adapter_hashes = defaultdict(set), defaultdict(set)
    gates, coverage_receipts, examples = {}, {}, defaultdict(list)
    for block, model_id in enumerate(evaluation.PARENTS):
        panel = snapshot / "engine" / f"b{block}"
        panel_slots = [s for s in slots if s["model_id"] == model_id]
        groups = evaluation.group_slots(panel_slots)
        assert len(groups) == 240 and len(panel_slots) == 960
        assert {p.name for p in (panel / "groups").glob("*") if p.is_file()} == {
            g["group_id"] + ".json" for g in groups
        }
        assert {p.name for p in (panel / "scores").glob("*") if p.is_file()} == {
            g["group_id"] + ".json" for g in groups
        }
        coverage, coverage_sha = read_json(panel / "COVERAGE.json")
        hashes, observed_slots, panel_audit = [], [], []
        for group in groups:
            path = panel / "groups" / (group["group_id"] + ".json")
            raw, raw_sha = read_json(path)
            relative = str(path.relative_to(snapshot))
            assert raw["group_id"] == group["group_id"] and len(raw["records"]) == 4
            assert raw["records_sha256"] == canonical_hash(raw["records"])
            saved, score_sha = read_json(panel / "scores" / path.name)
            assert (
                saved["raw_sha256"] == raw_sha
                and saved["scorer_version"] == evaluation.EXPERIMENT_ID
            )
            assert len(saved["scores"]) == 4
            saved_by_slot = {r["slot_id"]: r["score"] for r in saved["scores"]}
            assert len(saved_by_slot) == 4
            first = group["slots"][0]
            request = store.model_input(first["qid"], first["protocol"])
            if first["view"]:
                request = evaluation.diagnostic_input(tasks[first["qid"]], request, first["view"])
            input_sha = canonical_hash(request)
            input_hashes[group["group_id"]] = input_sha
            assert len({r["prompt_tensor_hash"] for r in raw["records"]}) == 1
            for index, (row, slot) in enumerate(zip(raw["records"], group["slots"], strict=True)):
                assert (
                    row["pool"] == "ENGINE" and row["phase"] == "ENGINE" and row["sealed"] is False
                )
                assert row["model_input"] == request and row["prompt"] == request["prompt"]
                assert row["group_id"] == group["group_id"]
                evaluation.validate_record(
                    row,
                    slot,
                    input_hash=input_sha,
                    model_identity=row["model_identity"],
                    adapter_identity=row["adapter_identity"],
                    source_sha256=identity_sha,
                    index=index,
                    count=4,
                )
                model_hashes[model_id].add(canonical_hash(row["model_identity"]))
                adapter_hashes[model_id].add(canonical_hash(row["adapter_identity"]))
                assert row["model_identity"]["parent_id"] == model_id
                assert row["adapter_identity"]["adapter_parameter_hash"] == coverage["policy_hash"]
                scored = evaluation.score_record(row, tasks[row["qid"]])
                previous = saved_by_slot[row["slot_id"]]
                if scored != previous:
                    diffs.append(
                        dict(
                            slot_id=row["slot_id"],
                            raw_file=relative,
                            kind="original_scorer_vs_saved",
                            computed=scored,
                            saved=previous,
                        )
                    )
                parsed = independent_extract(row["raw_text"], row["protocol"])
                correct = int(
                    bool(parsed["value_parseable"]) and parsed["answer"] == gold[row["qid"]]
                )
                if (parsed["value_parseable"], correct) != (
                    scored["value_parseable"],
                    scored["value_correct"],
                ):
                    diffs.append(
                        dict(
                            slot_id=row["slot_id"],
                            raw_file=relative,
                            kind="independent_value_vs_original",
                            independent_parseable=parsed["value_parseable"],
                            independent_correct=correct,
                        )
                    )
                stop = stop_audit(row)
                if stop["issues"]:
                    stop_issues.append(
                        dict(slot_id=row["slot_id"], raw_file=relative, row_index=index, **stop)
                    )
                audit = dict(
                    slot_id=row["slot_id"],
                    group_id=group["group_id"],
                    raw_file=relative,
                    raw_file_sha256=raw_sha,
                    row_index=index,
                    block=block,
                    model_id=model_id,
                    qid=row["qid"],
                    family=row["family"],
                    protocol=row["protocol"],
                    view=row["view"],
                    draw=row["draw"],
                    input_hash=input_sha,
                    raw_text_sha256=sha(row["raw_text"].encode()),
                    tokens=len(row["tokens"]),
                    finish_reason=row["finish_reason"],
                    format_protocol_error=row["format_protocol_error"],
                    score=scored,
                    independent_value_parseable=parsed["value_parseable"],
                    independent_value_correct=correct,
                    failure_category=parsed["failure_category"],
                    failure_detail=parsed["failure_detail"],
                    object_keys=parsed["object_keys"],
                    stop_audit=stop,
                )
                panel_audit.append(audit)
                if not parsed["value_parseable"]:
                    example_key = (
                        block,
                        row["view"] or row["protocol"],
                        parsed["failure_category"],
                    )
                    if len(examples[example_key]) < 2:
                        examples[example_key].append(
                            dict(
                                audit,
                                raw_text=row["raw_text"],
                                expected_exact_answer=str(gold[row["qid"]]),
                            )
                        )
                observed_slots.append(
                    dict(
                        slot,
                        generation_status=row["generation_status"],
                        technical_validation_errors=row["technical_validation_errors"],
                    )
                )
            hashes.append(dict(group_id=group["group_id"], sha256=raw_sha))
            raw_files.append(dict(path=relative, sha256=raw_sha, score_sha256=score_sha))
        validated = evaluation.validate_coverage(observed_slots, panel_slots)
        assert all(coverage[k] == validated[k] for k in validated)
        assert coverage["groups"] == hashes and coverage["records_sha256"] == canonical_hash(hashes)
        assert coverage["rng_isolated"] is True
        assert len(model_hashes[model_id]) == len(adapter_hashes[model_id]) == 1
        original_gate, gate_sha = read_json(panel / "PROTOCOL_GATE.json")
        reproduced = {}
        for protocol in evaluation.PROTOCOLS:
            selected = [r for r in panel_audit if r["protocol"] == protocol and r["view"] is None]
            assert len(selected) == 256
            n = sum(r["independent_value_parseable"] for r in selected)
            reproduced[protocol] = dict(
                records=256,
                parseable=n,
                fraction=n / 256,
                status="PASS" if n >= 231 else "PROTOCOL_FAILURE",
            )
        assert original_gate == dict(parent_id=model_id, protocols=reproduced)
        gates[f"b{block}"] = reproduced
        coverage_receipts[f"b{block}"] = dict(
            status="PASS",
            records=960,
            groups=240,
            coverage_sha256=coverage_sha,
            protocol_gate_sha256=gate_sha,
            slots_sha256=validated["slots_sha256"],
            records_sha256=coverage["records_sha256"],
            policy_hash=coverage["policy_hash"],
        )
        all_audit.extend(panel_audit)
    assert len(all_audit) == 2880 and len({r["slot_id"] for r in all_audit}) == 2880
    auxiliary = audit_existing_monitor_and_cache(
        snapshot,
        output,
        tasks,
        store,
        identity_sha,
        coverage_receipts["b0"]["policy_hash"],
        evaluation,
        contract,
        data,
    )
    dimensions = {}
    for names in [("block", "protocol", "view"), ("block", "protocol", "view", "family")]:
        buckets = defaultdict(list)
        for row in all_audit:
            buckets[tuple(row[n] for n in names)].append(row)
        dimensions["_".join(names)] = [
            dict(zip(names, key, strict=True), **summarize(rows)) for key, rows in buckets.items()
        ]
    with (output / "ROW_AUDIT.jsonl").open("w") as stream:
        for row in all_audit:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
    write_json(
        output / "FAILURE_EXAMPLES.json", [item for items in examples.values() for item in items]
    )
    write_json(output / "RAW_FILES_VERIFIED.json", raw_files)
    write_json(output / "DIFFERENCES.json", dict(scoring=diffs, stop_metadata=stop_issues))
    snapshot_manifest, snapshot_manifest_sha = read_json(snapshot / "ENGINE_SNAPSHOT_MANIFEST.json")
    snapshot_files = {item["path"]: item for item in snapshot_manifest["files"]}
    assert snapshot_files["RUN_IDENTITY.json"]["sha256"] == identity_sha
    for item in raw_files + json.loads((output / "AUXILIARY_RAW_FILES_VERIFIED.json").read_text()):
        assert snapshot_files[item["path"]]["sha256"] == item["sha256"]
    result = dict(
        status="AUDIT_COMPLETE_NO_DIFFERENCES"
        if not diffs and not stop_issues and auxiliary["status"] == "AUDIT_COMPLETE_NO_DIFFERENCES"
        else "AUDIT_COMPLETE_WITH_DIFFERENCES",
        scope=(
            "All 2880 original ENGINE diagnostic rows plus existing 512 MONITOR and "
            "512 engineering recovery cache rows; fixed original 714 scorers and "
            "independent exact scalar extractor"
        ),
        new_model_calls=0,
        new_optimizer_steps=0,
        new_test_outputs_read=0,
        source_commit=COMMIT,
        source_files=sources,
        deployment_source_manifest_sha256=source_manifest_sha,
        source_verification=source_verification,
        run_identity_sha256=identity_sha,
        snapshot_manifest_sha256=snapshot_manifest_sha,
        snapshot_declares_no_science_freeze=snapshot_manifest["science_freeze_exists"] is False,
        package_manifest_sha256=package["package_manifest_sha256"],
        input_slots_sha256=canonical_hash(slots),
        exact_gold_crosscheck_tasks=128,
        independent_parser_self_checks=checks,
        panels=coverage_receipts,
        gates=gates,
        protocol_summaries=dimensions["block_protocol_view"],
        family_summaries=dimensions["block_protocol_view_family"],
        score_differences=len(diffs),
        stop_metadata_differences=len(stop_issues),
        auxiliary=auxiliary,
        actual_new_output_rows=2880 + auxiliary["actual_new_output_rows"],
        all_scored_row_differences=len(diffs) + auxiliary["score_differences"],
        all_stop_metadata_differences=len(stop_issues) + auxiliary["stop_metadata_differences"],
        model_identity_hashes={k: sorted(v) for k, v in model_hashes.items()},
        adapter_identity_hashes={k: sorted(v) for k, v in adapter_hashes.items()},
        caveats=[
            "Stored token-prefix metadata is checked for internal consistency; "
            "this is not a fresh native tokenizer/BOS/EOS/model routing replay.",
            "Model/adapter identity metadata and unchanged policy hashes are cross-checked; "
            "model weights are absent from the snapshot.",
            "Diagnostic-view changes are input interventions; "
            "their outcomes cannot certify pure module capability.",
            "No permissive salvage, threshold change, answer search, task selection, "
            "or new generation is performed.",
        ],
    )
    result["audit_script_sha256"] = sha(Path(__file__).read_bytes())
    result["artifacts"] = {
        p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "EVALUATION_REVIEW.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                records=len(all_audit),
                groups=len(raw_files),
                total_scored_records=result["actual_new_output_rows"],
                gates=gates,
                score_differences=len(diffs),
                stop_metadata_differences=len(stop_issues),
                output=str(output),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
