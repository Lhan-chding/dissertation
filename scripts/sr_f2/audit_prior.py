#!/usr/bin/env python3
"""Read-only prior inventory, blind syntax census and v2 sidecar; no model calls."""
# ruff: noqa: E402, RUF001

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sr_f1.chartqa import score_chartqa
from sr_f1.contract import gold_evidence, load_json_strict, reference_module
from sr_f1.evaluation import read_jsonl
from sr_f12.evaluation import score_record
from sr_f12.protocol import BASELINE, object_hash, scientific_matrix

EXPECTED_ENDPOINT_RECORDS = 227700
FIELDS = (
    "slot_id",
    "model_id",
    "step",
    "pool",
    "protocol",
    "view",
    "qid",
    "root_id",
    "family",
    "chart",
    "variant",
    "draw",
)
GATE_GROUPS = ((39, 11, "train-0011-line-v6"), (77, 4, "train-0091-line-v6"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(directory, relative):
    directory, relative = Path(directory).resolve(strict=True), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Artifact path must be relative and contained")
    result = (directory / relative).resolve(strict=True)
    if not result.is_relative_to(directory) or not result.is_file():
        raise PermissionError("Artifact escapes its authenticated directory")
    return result


def new_output(path, protected):
    path = Path(path).resolve()
    for root in protected:
        root = Path(root).resolve()
        if path == root or path.is_relative_to(root):
            raise PermissionError(f"Output must be outside prior source root: {root}")
    path.mkdir(parents=True, exist_ok=False)
    return path


class Sources:
    """Per-file identities; checkpoint bytes are streamed, never unpickled."""

    def __init__(self):
        self.files = {}

    def add(self, path, *, role, expected=None):
        source = Path(path).absolute()
        canonical = source.resolve(strict=True)
        require(canonical.is_file(), f"Not a source file: {source}")
        stat = canonical.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        previous = self.files.get(str(source))
        if previous:
            require(tuple(previous["stat_signature"]) == signature, f"Source changed: {source}")
            require(previous["canonical_path"] == str(canonical), f"Source retargeted: {source}")
            actual = previous["sha256"]
        else:
            actual = sha_file(canonical)
            after = canonical.stat()
            require(
                signature == (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino),
                f"Source changed during hashing: {source}",
            )
            self.files[str(source)] = dict(
                source_path=str(source),
                canonical_path=str(canonical),
                sha256=actual,
                bytes=stat.st_size,
                stat_signature=list(signature),
                roles=[],
            )
        if expected is not None:
            require(actual == expected, f"Source SHA-256 differs: {source}")
        entry = self.files[str(source)]
        if role not in entry["roles"]:
            entry["roles"].append(role)
        return entry

    def read(self, path, *, role, expected=None):
        self.add(path, role=role, expected=expected)
        return json.loads(Path(path).read_text())

    def stable(self):
        for entry in self.files.values():
            path = Path(entry["source_path"])
            require(
                str(path.resolve(strict=True)) == entry["canonical_path"],
                f"Source retargeted during audit: {path}",
            )
            stat = path.stat()
            require(
                [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
                == entry["stat_signature"],
                f"Source changed during audit: {path}",
            )


def verify_registration(root, sources):
    registration = sources.read(root / "EARLY_ENDPOINT_IMPLEMENTATION.json", role="registration")
    protected = [root]
    for prefix in ("", "training_"):
        manifest_path = Path(registration[prefix + "source_manifest"])
        code_root = Path(registration[prefix + "code_root"]).resolve(strict=True)
        protected.append(code_root)
        manifest = sources.read(manifest_path, role="prior_code_manifest")
        require(
            object_hash(manifest) == registration[prefix + "source_manifest_sha256"],
            "Prior source manifest object hash differs",
        )
        entries = manifest["files"]
        if isinstance(entries, dict):
            entries = [dict(path=k, sha256=v) for k, v in entries.items()]
        require(
            bool(entries) and len({e["path"] for e in entries}) == len(entries),
            "Source manifest is empty or has duplicate paths",
        )
        for entry in entries:
            evidence = sources.add(
                contained(code_root, entry["path"]), role="prior_code", expected=entry["sha256"]
            )
            require(entry.get("bytes") in (None, evidence["bytes"]), "Source byte count differs")
    for relative, expected in registration["frozen_files"].items():
        sources.add(contained(root, relative), role="prior_frozen", expected=expected)
    mapping = sources.read(
        root / "deployment/checkpoint_storage_20261010/APPLIED.json", role="hdd_mapping"
    )
    require(len(mapping["mappings"]) == 8, "Expected eight registered HDD checkpoint roots")
    for source, target in mapping["mappings"].items():
        source, target = Path(source), Path(target).resolve()
        require(
            source.is_symlink() and source.resolve() == target,
            f"HDD mapping differs: {source}",
        )
        require(
            source.parent.resolve().parent == (root / "runs").resolve(),
            "HDD checkpoint source is outside prior runs",
        )
        protected.append(target)
    return registration, protected


def released_state(root, sources):
    complete = sources.read(root / "endpoint_orchestration/COMPLETE.json", role="endpoint_release")
    state = sources.read(root / "endpoint_orchestration/STATE.json", role="endpoint_state")
    require(
        complete["status"] == "EVALUATIONS_COMPLETE_REPORT_PENDING"
        and complete["endpoint_models"] == 13
        and complete["endpoint_records"] == EXPECTED_ENDPOINT_RECORDS,
        "Prior endpoint matrix is not completely released",
    )
    require(
        state["scores_sealed"] is False
        and state["endpoints_complete"] == 13
        and not state["failures"]
        and not state["unknown_submissions"],
        "Prior endpoint release still has sealed/failed/unknown work",
    )
    expected = {r["model_id"] for r in scientific_matrix()} | {BASELINE}
    require(
        len(complete["releases"]) == 13
        and {r["model_id"] for r in complete["releases"]} == expected,
        "Released model matrix differs",
    )
    acceptance = sources.read(
        root / "endpoint_orchestration/MAIN_ACCEPTANCE.json", role="prior_training_acceptance"
    )
    authorization = acceptance["main_completion"]
    require(
        object_hash(authorization) == acceptance["main_completion_sha256"],
        "Prior training authorization hash differs",
    )
    require(
        {r["model_id"] for r in authorization["paths"]} == expected - {BASELINE}
        and len(authorization["paths"]) == 12,
        "Prior authorized training matrix differs",
    )
    return complete, authorization


def endpoint_rows(root, sources, complete, authorization):
    """Yield raw rows and v1 scores, retaining at most one model's score table."""
    total = 0
    for ref in complete["releases"]:
        model = ref["model_id"]
        release = sources.read(root / "released/evaluation" / f"{model}.json", role="v1_release")
        require(object_hash(release) == ref["release_sha256"], "Release object hash differs")
        require(
            release["status"] == "RELEASED_AFTER_ALL_MAIN_COMPLETE"
            and release["authorization_sha256"] == object_hash(authorization),
            "Endpoint was not authorized by the complete training matrix",
        )
        require(
            release["count"] == len(release["records"]) == ref["records"]
            and object_hash(release["records"]) == release["records_sha256"],
            "Released record count/hash differs",
        )
        released = {r["slot_id"]: r for r in release["records"]}
        require(len(released) == len(release["records"]), "Duplicate released slot")
        directory = root / "sealed/evaluation" / model
        done = sources.read(directory / "COMPLETE.json", role="endpoint_complete")
        require(
            done["status"] == "COMPLETE" and object_hash(done) == release["evaluation_sha256"],
            "Endpoint completion hash differs",
        )
        if done.get("generation_gate") == "MODEL_TERMINAL_SEALED":
            generation = sources.read(
                directory / "GENERATION_AUTHORIZATION.json", role="early_generation_authorization"
            )
            accepted = sources.read(
                root / "endpoint_orchestration/model_acceptance" / f"{model}.json",
                role="early_model_acceptance",
            )
            require(
                generation == accepted
                and object_hash(generation) == done["generation_authorization_sha256"],
                "Early generation authorization differs",
            )
            selected = generation["model_completion"]
            require(
                object_hash(selected) == generation["model_completion_sha256"]
                and selected["paths"]
                == [p for p in authorization["paths"] if p["model_id"] == model],
                "Early generation checkpoint differs from released checkpoint",
            )
        else:
            require(
                done["main_completion_sha256"] == object_hash(authorization),
                "Endpoint matrix authorization differs",
            )
        seen = set()
        for panel in done["panels"]:
            panel_name = Path(panel["panel"])
            require(
                not panel_name.is_absolute() and ".." not in panel_name.parts,
                "Panel path escapes endpoint directory",
            )
            panel_dir = directory / panel_name
            require(
                panel_dir.resolve(strict=True).is_relative_to(directory.resolve()),
                "Panel directory escapes endpoint root",
            )
            finish = sources.read(
                contained(panel_dir, "COMPLETE.json"),
                role="panel_complete",
                expected=panel["completion_sha256"],
            )
            plan = sources.read(contained(panel_dir, "PLAN.json"), role="panel_plan")
            require(finish["plan_sha256"] == object_hash(plan), "Panel plan hash differs")
            groups = finish["groups"]
            require(
                len(groups) == len(plan["assigned_groups"]) == len({g["group_id"] for g in groups})
                and {g["group_id"] for g in groups} == set(plan["assigned_groups"]),
                "Panel group membership differs",
            )
            panel_count = 0
            for group in groups:
                raw_path = contained(panel_dir, "groups/" + group["group_id"] + ".json")
                raw = sources.read(raw_path, role="endpoint_raw", expected=group["raw_sha256"])
                require(
                    len(raw["records"]) == group["records"]
                    and object_hash(raw["records"]) == raw["records_sha256"],
                    "Raw endpoint group count/hash differs",
                )
                for index, row in enumerate(raw["records"]):
                    key = row["slot_id"]
                    require(
                        key in released and key not in seen and row["model_id"] == model,
                        "Unexpected, duplicate or mismatched raw endpoint slot",
                    )
                    require(
                        {k: row[k] for k in FIELDS} == {k: released[key][k] for k in FIELDS},
                        "Raw/released endpoint identity differs",
                    )
                    seen.add(key)
                    location = dict(path=str(raw_path), sha256=group["raw_sha256"], row_index=index)
                    yield row, released[key]["score"], location
                    panel_count += 1
            require(
                panel_count
                == panel["records"]
                == finish["generated_records"]
                == plan["expected_records"],
                "Endpoint panel record count differs",
            )
        require(seen == set(released), "Raw endpoint coverage differs from released scores")
        total += len(seen)
        print(
            json_text(dict(event="endpoint_verified", model_id=model, records=len(seen))),
            flush=True,
        )
    require(total == EXPECTED_ENDPOINT_RECORDS, "Endpoint total is not 227700")


def checkpoint_inventory(root, sources, authorization, write):
    from mm_core.training import state_hash

    failures, count = [], 0
    for run in scientific_matrix():
        directory = root / "runs" / run["model_id"]
        manifest = sources.read(directory / "RUN_MANIFEST.json", role="training_manifest")
        require(
            all(manifest.get(k) == v for k, v in run.items())
            and manifest["technical_only"] is False,
            "Training manifest differs",
        )
        done = sources.read(directory / "COMPLETE.json", role="training_complete")
        selected = next(p for p in authorization["paths"] if p["model_id"] == run["model_id"])
        require(
            sources.add(directory / "COMPLETE.json", role="training_complete")["sha256"]
            == selected["complete_sha256"],
            "Training completion bytes differ",
        )
        for step in range(0, 97, 8):
            record = dict(kind="checkpoint", model_id=run["model_id"], step=step)
            try:
                cp_dir = directory / "checkpoints"
                commit_path = cp_dir / f"commit-{step:02d}.json"
                commit = sources.read(commit_path, role="checkpoint_commit")
                require(
                    commit["step"] == step
                    and bool(commit["state_hash"])
                    and bool(commit["field_hashes"]),
                    "Malformed checkpoint commit",
                )
                require(
                    commit["field_hashes"]["run_identity"] == state_hash(manifest),
                    "Checkpoint run-identity hash differs",
                )
                path = contained(cp_dir, commit["path"])
                entry = sources.add(path, role="checkpoint_bytes", expected=commit["sha256"])
                if step == 96:
                    latest = sources.read(cp_dir / "LATEST.json", role="checkpoint_latest")
                    require(
                        latest == commit == done["checkpoint"]
                        and commit["sha256"] == selected["checkpoint_sha256"],
                        "Final checkpoint differs from authorized state",
                    )
                record.update(
                    status="BYTE_HASH_VERIFIED",
                    commit_path=str(commit_path),
                    resolved_directory=str(cp_dir.resolve()),
                    commit=commit,
                    checkpoint=entry,
                )
                count += 1
            except (OSError, ValueError, KeyError) as exc:
                record.update(status="MISSING_OR_INVALID", error=f"{type(exc).__name__}: {exc}")
                failures.append({k: record[k] for k in ("model_id", "step", "error")})
            write(record)
        print(
            json_text(
                dict(
                    event="checkpoints_checked",
                    model_id=run["model_id"],
                    verified=count,
                    failures=len(failures),
                )
            ),
            flush=True,
        )
    return dict(
        expected=156,
        verified=count,
        failures=failures,
        state_scope="All bytes hashed against original commits; tensors not unpickled or rehashed",
    )


def gate_evidence(group, metrics, task, *, location):
    """Derive float64 intermediates with the authenticated ORIGINAL implementation."""
    old = reference_module("semantic_contract")
    scores = [r["score"] for r in group["records"]]
    j = np.array([s["J"] for s in scores], dtype=np.float64)
    e = np.array([s["E"] for s in scores], dtype=np.float64)
    p = np.array([s["p_read"] for s in scores], dtype=np.float64)
    h = 0.5 * e * (1 + p)
    failed = j == 0
    require(np.any(failed) and np.any(j), "Requested GATE anomaly is not a mixed group")
    binary = old.zscore(j)
    centered = h[failed] - h[failed].mean()
    scale = float(np.max(np.abs(centered)))
    advantage, branch = old.gate_one(j, e, p)
    slot = group["identity"]["slot"]
    actual = np.array(metrics["reward_advantage_audit"]["final_advantage"][slot])
    require(
        np.array_equal(actual, advantage),
        "Original GATE recomputation differs from saved advantages",
    )
    require(branch == metrics["reward_advantage_audit"]["branches"][slot], "GATE branch differs")
    records = []
    truth = gold_evidence(task["world"], task["query"])
    for row in group["records"]:
        expected = score_record({**row, "protocol": "evidence_answer"}, task)
        # Training's scorer did not override reason for format_protocol_error.
        expected["reason"] = row["score"]["reason"]
        require(expected == row["score"], "Anomalous GATE component rescore differs")
        numerator = round(row["score"]["p_read"] * len(truth))
        records.append(
            dict(
                raw_text=row["raw_text"],
                tokens=row["tokens"],
                score=row["score"],
                p_read_numerator=numerator,
                p_read_denominator=len(truth),
            )
        )
    return dict(
        kind="gate_original_group",
        identity=group["identity"],
        source=location,
        original_records=records,
        derived_from_original_components=True,
        dtype=str(h.dtype),
        J=j.tolist(),
        E=e.tolist(),
        p_read=p.tolist(),
        h=h.tolist(),
        binary_J_advantage=binary.tolist(),
        failed_centered_h=centered.tolist(),
        failed_h_exactly_equal=bool(np.all(h[failed] == h[failed][0])),
        failed_scale=scale,
        branch=branch,
        recomputed_advantage=advantage.tolist(),
        saved_advantage=actual.tolist(),
        saved_advantage_sum=float(actual.sum()),
        exact_reproduction=True,
    )


def training_inventory(root, sources, write, tasks):
    count, responses, anomaly_count = 0, 0, 0
    old = reference_module("semantic_contract")
    for run in scientific_matrix():
        directory = root / "runs" / run["model_id"]
        for step in range(1, 97):
            metrics = sources.read(directory / "steps" / f"{step:02d}.json", role="training_step")
            require(
                metrics["step"] == step and metrics["arm"] == run["arm"],
                "Training step identity differs",
            )
            records, groups = [], []
            for slot in range(16):
                path = directory / "rollouts" / f"{step:02d}-{slot:02d}.json"
                group = sources.read(path, role="training_rollout")
                identity = group["identity"]
                require(
                    identity["run_id"] == run["model_id"]
                    and identity["step"] == step
                    and identity["slot"] == slot
                    and len(group["records"]) == 8
                    and object_hash(group["records"]) == group["sha256"],
                    "Training rollout identity/count/hash differs",
                )
                require(
                    all(
                        r["qid"] == identity["qid"]
                        and bool(r["tokens"])
                        and len(r["tokens"]) == len(r["sampler_logprobs"])
                        for r in group["records"]
                    ),
                    "Training raw text/token records incomplete",
                )
                groups.append([r["score"] for r in group["records"]])
                records.extend(group["records"])
                if (
                    run["model_id"] == "SRF1_2_GATE_s71002"
                    and (step, slot, identity["qid"]) in GATE_GROUPS
                ):
                    write(
                        gate_evidence(
                            group,
                            metrics,
                            tasks[identity["qid"]],
                            location=dict(
                                path=str(path), sha256=sources.files[str(path.absolute())]["sha256"]
                            ),
                        )
                    )
                    anomaly_count += 1
            require(
                object_hash(records) == metrics["raw_records_sha256"],
                "Committed training step no longer binds original rollout records",
            )
            _, audit = old.reward_advantages(run["arm"], groups)
            require(audit == metrics["reward_advantage_audit"], "Original advantage audit differs")
            count += 1
            responses += len(records)
            write(
                dict(
                    kind="rollout_step",
                    model_id=run["model_id"],
                    step=step,
                    records=len(records),
                    raw_records_sha256=object_hash(records),
                    original_advantages_reproduced=True,
                )
            )
        print(
            json_text(dict(event="training_verified", model_id=run["model_id"], steps=96)),
            flush=True,
        )
    require(
        count == 1152 and responses == 147456 and anomaly_count == 2,
        "Training inventory or requested GATE groups incomplete",
    )
    return dict(steps=count, raw_responses=responses, gate_groups=anomaly_count)


def load_tasks(root, sources):
    path = root / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl"
    sources.add(path, role="gold_tasks")
    rows = list(read_jsonl(path))
    require(len({r["qid"] for r in rows}) == len(rows), "Duplicate gold task qid")
    return {r["qid"]: r for r in rows}


def raw_selection(row):
    keys = (
        "raw_text",
        "tokens",
        "generated_text",
        "completion_token_count",
        "finish_reason",
        "truncated",
        "format_protocol_error",
        "generation_status",
    )
    return {k: row.get(k) for k in keys}


def syntax_census(root, sources, complete, authorization, write, classifier):
    counts, unparseable = Counter(), Counter()
    total = 0
    for row, old, location in endpoint_rows(root, sources, complete, authorization):
        if row["pool"] != "TEST_COMPOSITION" or row["model_id"] == BASELINE:
            continue
        # Neither task, gold, correctness score nor model identity enters classifier.
        syntax = classifier(row["raw_text"], protocol=row["protocol"])
        key = "|".join((row["model_id"], row["family"], syntax["answer_syntax"]))
        counts[key] += 1
        if old["L_answer"] == 0:
            unparseable[key] += 1
        write(
            dict(
                kind="blind_syntax",
                **{k: row[k] for k in FIELDS},
                syntax=syntax,
                v1_L_answer=old["L_answer"],
                source=location,
                raw=raw_selection(row),
            )
        )
        total += 1
    return dict(
        rows=total,
        category_counts=dict(counts),
        v1_unparseable_counts=dict(unparseable),
        gold_read=False,
        classifier_input="raw_text and protocol only",
        raw_examples="all student composition records",
    )


def evidence_pattern(raw_text, task):
    old = reference_module("semantic_contract")
    required = set(old.required_keys(task["world"], task["query"]))
    selected = {
        task["world"]["categories"][i] for i in old.selected_indices(task["world"], task["query"])
    }
    try:
        obj = load_json_strict(raw_text)
        entries = obj["evidence"]
        require(isinstance(entries, list), "Evidence is not a list")
        keys = [(v["series"], v["category"]) for v in entries]
        require(
            all(isinstance(s, str) and isinstance(c, str) for s, c in keys),
            "Malformed evidence keys",
        )
        reported = set(keys)
        omitted = required - reported
        labels = [f"{s}:{'selected' if c in selected else 'unselected'}" for s, c in omitted]
        return dict(
            status="KEYS_EXTRACTED",
            reported_keys=sorted(reported),
            missing_keys=sorted(omitted),
            extra_keys=sorted(reported - required),
            duplicated_keys=sorted(k for k, n in Counter(keys).items() if n > 1),
            omission_pattern=dict(sorted(Counter(labels).items())),
        )
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return dict(status="UNEXTRACTABLE_KEYS")


def gold_type(value):
    if value is None:
        return "null"
    try:
        number = Decimal(str(value).strip())
        if number.is_finite():
            return "integer" if number == number.to_integral_value() else "decimal"
        return "nonfinite_literal"
    except ArithmeticError:
        return "fraction_string" if isinstance(value, str) and "/" in value else "nonnumeric_text"


def verify_blind_receipt(path, root, sources, classifier):
    directory = Path(path).resolve(strict=True)
    summary = sources.read(directory / "SUMMARY.json", role="blind_phase_receipt")
    require(
        summary["status"] == "PASS"
        and summary["phase"] == "blind-classify"
        and summary["root"] == str(root)
        and summary["result"]["gold_read"] is False,
        "A successful gold-blind classification receipt from the same root is required",
    )
    classifier_hash = hashlib.sha256(inspect.getsource(classifier).encode()).hexdigest()
    require(
        summary["classifier_function_sha256"] == classifier_hash
        and summary["classifier_module_sha256"] == sha_file(inspect.getfile(classifier)),
        "Classifier rule changed after blind census",
    )
    previous = sources.read(
        directory / "SOURCE_MANIFEST.json",
        role="blind_source_manifest",
        expected=summary["source_manifest_sha256"],
    )
    for row in previous["files"]:
        if "endpoint_raw" in row["roles"]:
            sources.add(row["source_path"], role="blind_raw_identity", expected=row["sha256"])
    sources.add(
        directory / "AUDIT.jsonl", role="blind_syntax_rows", expected=summary["audit_jsonl_sha256"]
    )
    return dict(
        path=str(directory),
        summary_sha256=sha_file(directory / "SUMMARY.json"),
        classifier_function_sha256=classifier_hash,
    )


def rescore(root, sources, complete, authorization, write, scorer, classifier, examples):
    tasks = load_tasks(root, sources)
    chart_receipt = sources.read(root / "external/chartqa/IDENTITY.json", role="chartqa_identity")
    require(chart_receipt["status"] == "AVAILABLE_VERIFIED", "ChartQA not verified")
    chart_path = root / "external/chartqa/TEST_MANIFEST.jsonl"
    sources.add(chart_path, role="chartqa_gold", expected=chart_receipt["manifest_sha256"])
    chart = {r["qid"]: r for r in read_jsonl(chart_path)}
    require(len(chart) == 2500, "ChartQA gold count differs")
    counts, patterns, selected, composition = Counter(), Counter(), Counter(), Counter()
    chart_rows = defaultdict(dict)
    metrics = defaultdict(Counter)
    for row, released, location in endpoint_rows(root, sources, complete, authorization):
        output = dict(kind="endpoint_score", **{k: row[k] for k in FIELDS}, source=location)
        selections = []
        if row["pool"] == "CHARTQA":
            gold = chart[row["qid"]]["answer"]
            old = score_chartqa(gold, row["raw_text"])
            new = score_chartqa(str(gold).strip(), row["raw_text"].strip())
            output["v2_scope"] = "both_sides_strip_only"
            chart_rows[row["qid"]][row["model_id"]] = dict(
                score_v1=old, score_v2=new, source=location, raw=raw_selection(row)
            )
        else:
            task = tasks[row["qid"]]
            old = score_record(row, task)
            new = scorer(row, task)
            syntax = classifier(row["raw_text"], protocol=row["protocol"])
            output["answer_syntax"] = syntax
            if (
                row["pool"] == "TEST_COMPOSITION"
                and row["model_id"] != BASELINE
                and not old["L_answer"]
            ):
                selections.append("composition_v1_unparseable")
                output["gold_answer"] = task["answer"]
                output["gold_type"] = gold_type(task["answer"])
                composition[
                    "|".join(
                        (
                            row["model_id"],
                            row["family"],
                            output["gold_type"],
                            syntax["answer_syntax"],
                        )
                    )
                ] += 1
            if (
                row["model_id"].startswith("SRF1_2_A_s")
                and row["pool"] == "TEST_ID"
                and row["protocol"] == "evidence_answer"
                and row["family"] == "TOPK"
            ):
                pattern = evidence_pattern(row["raw_text"], task)
                output["evidence_key_pattern"] = pattern
                stratum = "|".join(map(str, (row["model_id"], old["missing_count"], old["A"])))
                patterns[
                    "|".join(
                        (
                            stratum,
                            json_text(pattern.get("omission_pattern", {})),
                            str(old["extra_count"]),
                            pattern["status"],
                        )
                    )
                ] += 1
                if old["missing_count"] in (0, 2, 4) and selected[stratum] < examples:
                    selections.append("A_TOPK_fixed_stratum_example")
                    selected[stratum] += 1
                    output["task_world"] = task["world"]
                    output["task_query"] = task["query"]
            output["v2_scope"] = "decimal_answer_strings_and_parallel_E6_P6_J6"
        require(
            old == released, f"Recomputed v1 differs from official released score: {row['slot_id']}"
        )
        output.update(score_v1=old, score_v2=new, evidence_selections=selections)
        if selections:
            output["raw"] = raw_selection(row)
        write(output)
        counts["records"] += 1
        counts["v1_score_mismatches"] += old != released
        counts["answer_score_changed"] += old.get("A") != new.get("A")
        key = "|".join((row["model_id"], row["pool"], row["protocol"], row["family"]))
        metrics[key]["n"] += 1
        for version, score in (("v1", old), ("v2", new)):
            for name, value in score.items():
                if name != "C" and isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[key][version + ":" + name] += value
    models = {r["model_id"] for r in scientific_matrix()} | {BASELINE}
    common, types, candidates = [], Counter(), []
    for qid, values in sorted(chart_rows.items()):
        require(set(values) == models, "ChartQA comparison has missing model outputs")
        baseline_correct = values[BASELINE]["score_v2"]["relaxed_accuracy"] == 1
        wrong_models = sorted(
            m for m, v in values.items() if m != BASELINE and v["score_v2"]["relaxed_accuracy"] == 0
        )
        if baseline_correct and wrong_models:
            candidates.append(qid)
            label = gold_type(chart[qid]["answer"])
            all_students_wrong = len(wrong_models) == len(models) - 1
            if all_students_wrong:
                types[label] += 1
                common.append(qid)
            write(
                dict(
                    kind="chartqa_loss_candidate",
                    qid=qid,
                    gold=chart[qid]["answer"],
                    gold_type=label,
                    image_id=chart[qid]["image_id"],
                    outputs=values,
                    wrong_model_ids=wrong_models,
                    all_students_wrong=all_students_wrong,
                    scope="DIAGNOSTIC_NOT_THE_REQUESTED_UNRESOLVED_122_COHORT",
                )
            )
    # Include both requested original training groups in this same delivery JSONL.
    directory = root / "runs/SRF1_2_GATE_s71002"
    for step, slot, qid in GATE_GROUPS:
        path = directory / "rollouts" / f"{step:02d}-{slot:02d}.json"
        group = sources.read(path, role="requested_gate_rollout")
        require(
            group["identity"]["qid"] == qid and object_hash(group["records"]) == group["sha256"],
            "Requested original GATE group differs",
        )
        metrics_row = sources.read(
            directory / "steps" / f"{step:02d}.json", role="requested_gate_step"
        )
        write(
            gate_evidence(
                group,
                metrics_row,
                tasks[qid],
                location=dict(path=str(path), sha256=sources.files[str(path.absolute())]["sha256"]),
            )
        )
    return dict(
        **counts,
        metrics_sums=dict(metrics),
        A_TOPK_omission_patterns=dict(patterns),
        A_TOPK_example_counts=dict(selected),
        composition_unparseable_by_gold_type=dict(composition),
        requested_chartqa_cohort=dict(
            status="UNRESOLVED",
            planned_count=122,
            reason="Plan gives no qids or exact selector; no cohort is chosen to match 122",
        ),
        chartqa_diagnostic_all12_loss=dict(
            count=len(common),
            qids=common,
            gold_types=dict(types),
            definition="baseline strip-correct; every one of 12 student endpoints strip-incorrect",
            scope="DIAGNOSTIC_NOT_REQUESTED_COHORT",
        ),
        chartqa_loss_candidates=dict(
            count=len(candidates),
            qids=candidates,
            definition="baseline strip-correct; at least one student endpoint strip-incorrect",
            raw_scope="All 13 outputs retained for every candidate without selecting to match 122",
        ),
        historical_official_scores_changed=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Original SR-F1.2 server root")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory outside all prior roots"
    )
    parser.add_argument(
        "--phase", choices=("inventory", "blind-classify", "rescore"), required=True
    )
    parser.add_argument(
        "--blind-receipt", type=Path, help="Successful blind-classify output directory"
    )
    parser.add_argument("--examples-per-stratum", type=int, default=3)
    args = parser.parse_args(argv)
    require(1 <= args.examples_per_stratum <= 20, "Examples per stratum must be in [1,20]")
    root = args.root.resolve(strict=True)
    sources = Sources()
    # Read external-root declarations before creating anything, including for failed audits.
    registration = sources.read(root / "EARLY_ENDPOINT_IMPLEMENTATION.json", role="registration")
    mapping = sources.read(
        root / "deployment/checkpoint_storage_20261010/APPLIED.json", role="hdd_mapping"
    )
    protected = [
        root,
        registration["code_root"],
        registration["training_code_root"],
        *mapping["mappings"].values(),
    ]
    output = new_output(args.output, protected)
    summary = dict(
        phase=args.phase,
        root=str(root),
        status="RUNNING",
        started_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha_file(__file__),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        model_calls=0,
        source_mutations=0,
    )
    try:
        verify_registration(root, sources)
        complete, authorization = released_state(root, sources)
        with (output / "AUDIT.jsonl").open("x") as stream:

            def write(value):
                stream.write(json_text(value) + "\n")

            if args.phase == "inventory":
                checkpoints = checkpoint_inventory(root, sources, authorization, write)
                training = training_inventory(root, sources, write, load_tasks(root, sources))
                endpoints = sum(1 for _ in endpoint_rows(root, sources, complete, authorization))
                summary["result"] = dict(
                    checkpoints=checkpoints, training=training, endpoints=endpoints
                )
                summary["status"] = (
                    "PASS" if not checkpoints["failures"] else "INCOMPLETE_RESOURCES"
                )
            else:
                from sr_f2.scoring import classify_answer_syntax

                summary["classifier_function_sha256"] = hashlib.sha256(
                    inspect.getsource(classify_answer_syntax).encode()
                ).hexdigest()
                summary["classifier_module_sha256"] = sha_file(
                    inspect.getfile(classify_answer_syntax)
                )
                sources.add(inspect.getfile(classify_answer_syntax), role="v2_classifier_source")
                if args.phase == "blind-classify":
                    summary["result"] = syntax_census(
                        root, sources, complete, authorization, write, classify_answer_syntax
                    )
                else:
                    from sr_f2.scoring import score_record_v2

                    require(
                        args.blind_receipt is not None, "--blind-receipt is required before rescore"
                    )
                    summary["blind_receipt"] = verify_blind_receipt(
                        args.blind_receipt, root, sources, classify_answer_syntax
                    )
                    summary["result"] = rescore(
                        root,
                        sources,
                        complete,
                        authorization,
                        write,
                        score_record_v2,
                        classify_answer_syntax,
                        args.examples_per_stratum,
                    )
                summary["status"] = "PASS"
        sources.stable()
    except Exception:
        summary.update(status="FAIL", error=traceback.format_exc())
        raise
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest = dict(
            root=str(root),
            files=list(sources.files.values()),
            excluded=[
                "No new model generation",
                "No checkpoint tensor deserialization",
                "Raw text/tokens copied only for requested evidence selections",
                "No original official score or frozen source is changed",
            ],
        )
        (output / "SOURCE_MANIFEST.json").write_text(json_text(manifest) + "\n")
        summary["source_manifest_sha256"] = sha_file(output / "SOURCE_MANIFEST.json")
        if (output / "AUDIT.jsonl").exists():
            summary["audit_jsonl_sha256"] = sha_file(output / "AUDIT.jsonl")
        (output / "SUMMARY.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )
        (output / "README_zh.md").write_text(
            f"# SR-F2 既有证据 CPU 审计\n\n阶段：`{args.phase}`；状态：`{summary['status']}`。\n\n"
            "输入只读；未调用模型，未重训，未修改 SR-F1.2 正式成绩。\n\n"
            "AUDIT.jsonl 保存逐条记录，SOURCE_MANIFEST.json 保存源路径、解析后路径及 SHA-256，"
            "SUMMARY.json 保存检查数量、异常及分层统计。检查点完整字节对原提交核验，不反序列化"
            "执行 pickle；字节校验不称为新的训练科学结论。\n\n"
            "盲分类不读取金标任务或ChartQA金标清单；分类函数仅接收 raw_text 和协议。"
            "重评分才读金标，"
            "旧v1逐条复算对原发布值，新v2并列。ChartQA v2仅两侧strip后调用原评分器。\n\n"
            "TOPK每种子×缺项0/2/4×原A0/1固定分层，以原冻结遍历顺序取前N条，保留全量键统计。"
            "计划的122题未提供题号/选择器，状态UNRESOLVED。另列起点正确且至少一学生错误的全部候选，"
            "及12学生全错子集；二者明确是诊断，不替代请求的122题。"
            "缺失或失败均不补造。\n"
        )
        print(
            json_text({k: summary[k] for k in ("phase", "status", "started_at", "finished_at")}),
            flush=True,
        )
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
