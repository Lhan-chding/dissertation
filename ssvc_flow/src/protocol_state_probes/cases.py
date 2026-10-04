"""Import and independently recompile frozen cases before any model is loaded.

The main renderer receives public fields only. Labels are kept in a separate
array and clean controls explicitly disclose their correct observation.
"""

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .ptlc import anchors, dpe, transform_star
from .transforms import (
    FWD,
    REV,
    SYSTEM,
    predict,
    public_from_record,
    render,
    replace_once,
    valid_domain,
)


class BundleValidationError(ValueError):
    def __init__(self, differences):
        self.differences = differences
        super().__init__(
            "Frozen bundle recompile mismatch: " + json.dumps(differences, ensure_ascii=False)
        )


def _hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _unique(rows, key, name):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError(f"Duplicate {name} identity: {identity}")
        result[identity] = row
    return result


def _compare(expected, actual, path, differences):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                differences.append(
                    {
                        "path": f"{path}.{key}",
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                        "missing": key not in actual,
                    }
                )
            else:
                _compare(expected[key], actual[key], f"{path}.{key}", differences)
    elif isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{path}[{index}]", differences)
    elif type(expected) is not type(actual) or expected != actual:
        differences.append({"path": path, "expected": expected, "actual": actual})


def compile_case(record, panel, protocol):
    """Rebuild one registered public case and its separate privileged audit."""
    public = public_from_record(record)
    truth = record["scene"]["truth_world"]
    observed = public["observed"]
    j = record["scene"]["changed_index"]
    if type(j) is not int or j not in FWD:
        raise ValueError("Source corruption coordinate must be an integer in 0..3")
    if (
        record["base_scene_id"] != record["scene"]["base_scene_id"]
        or record["family"] != record["scene"]["cue"]["family"]
        or record["operation"] != record["scene"]["operation"]
    ):
        raise ValueError("Source record metadata disagrees with its scene")
    if not valid_domain(truth) or [k for k in FWD if truth[k] != observed[k]] != [j]:
        raise ValueError("Source record must have exactly its registered corruption")
    if any(
        sum(a * x for a, x in zip(row, truth, strict=True)) != rhs
        for row, rhs in zip(public["H"], public["b"], strict=True)
    ):
        raise ValueError("Source truth violates a reliable relation")
    if record["interface"] != "SYMBOLIC_FRESH":
        raise ValueError("Only the registered symbolic interface is supported")
    kind, expected = "repair", None
    if panel == "CONTROL_COPY":
        if protocol not in ("O0", "A1") or len(set(observed)) != 4:
            raise ValueError("Copy controls require four distinct observations and O0/A1")
        order = FWD if protocol == "O0" else REV
        names = "[a,b,c,d]" if order == FWD else "[d,c,b,a]"
        rendered = {
            "system": SYSTEM,
            "user": "The record order is [a,b,c,d].\nThe observed record is "
            + json.dumps(observed, separators=(",", ":"))
            + ".\nCopy the observed values without changing any number.\nReturn "
            + names
            + ", not the downstream answer.",
            "output_order": list(order),
            "H_display": [],
            "b_display": [],
        }
        kind, expected = "copy_control", list(observed)
    elif panel == "CONTROL_CLEAN_REL":
        if protocol not in ("B0", "B1"):
            raise ValueError("Clean controls use B0/B1 only")
        clean_public = copy.deepcopy(public)
        clean_public["observed"] = list(truth)
        clean_public["original_user"] = replace_once(
            clean_public["original_user"],
            "The observed record is " + json.dumps(observed, separators=(",", ":")) + ".",
            "The observed record is " + json.dumps(truth, separators=(",", ":")) + ".",
        )
        clean_public["original_user"] = replace_once(
            clean_public["original_user"],
            "Exactly one value in this observed record is wrong.",
            "All four values are correct and satisfy the relationships. Do not change any value.",
        )
        rendered = render(clean_public, protocol)
        observed, kind, expected = list(truth), "clean_relation_control", list(truth)
    else:
        if panel not in ("D48", "U22", "CONTROL_DUPLICATE"):
            raise ValueError("Unknown registered panel")
        if panel == "CONTROL_DUPLICATE" and (
            record["family"] != "duplicate_encoding" or protocol not in ("O0", "A1")
        ):
            raise ValueError("Duplicate controls require duplicate O0/A1 tasks")
        if panel in ("D48", "U22") and record["family"] not in ("cross_series", "trend"):
            raise ValueError("Core panels require cross/trend")
        if panel == "U22" and (record["split"] != "train" or protocol.startswith("L")):
            raise ValueError("U22 must preserve train origin and its core-only grid")
        rendered = render(public, protocol)
    case_id = f"{panel}:{record['base_scene_id']}:{protocol}"
    prompt = {"system": rendered["system"], "user": rendered["user"]}
    case = {
        "case_id": case_id,
        "panel": panel,
        "kind": kind,
        "base_scene_id": record["base_scene_id"],
        "parent_prompt_id": record["prompt_id"],
        "family": record["family"],
        "chart_type": record["chart_type"],
        "operation": record["operation"],
        "protocol": protocol,
        "interface": record["interface"],
        "original_split": record["split"],
        "split_role": "untouched_outcome_replication"
        if panel == "U22"
        else "development_diagnostic",
        "prompt": prompt,
        "output_order": rendered["output_order"],
        "H_display": rendered["H_display"],
        "b_display": rendered["b_display"],
        "H_original": public["H"],
        "b_original": public["b"],
        "observed_world_public": observed,
        "prompt_identity": _hash(prompt),
        "max_new_tokens": 64,
        "enable_thinking": False,
    }
    audit = {
        "case_id": case_id,
        "truth_world": list(truth),
        "corrupted_coordinate": j,
        "control_expected_canonical": expected,
        "DPE1_old": dpe(public["H"], j, FWD) if kind == "repair" else None,
        "DPE1_new": dpe(rendered["H_display"], j, rendered["output_order"])
        if kind == "repair"
        else None,
        "audit_fields_must_not_enter_model": True,
    }
    prediction = None
    if kind == "repair":
        prediction = {
            "case_id": case_id,
            "anchor_coordinates": anchors(rendered["H_display"], rendered["output_order"]),
            "copy_canonical": list(observed),
            "programs": predict(public, protocol),
        }
    return case, audit, prediction


def b1_diagnostic(public, corrupted_coordinate):
    """Both registered vectors are tested on either protocol's raw observations."""
    old, new = predict(public, "B0"), predict(public, "B1")
    _, _, meta = transform_star(public["H"], public["b"], FWD)
    changed = {variant: old[variant]["canonical"] != new[variant]["canonical"] for variant in old}
    category = {
        (False, False): "unchanged",
        (True, False): "V1_only",
        (False, True): "V2_only",
        (True, True): "both",
    }[(changed["V1"], changed["V2"])]
    return {
        "old": old,
        "new": new,
        "changed": changed,
        "change_category": category,
        "first_leaf_risk": corrupted_coordinate == meta["first_leaf"]
        and meta["center"] > meta["first_leaf"],
        **meta,
    }


def _prepared_records(prepared):
    if not isinstance(prepared, dict) or not isinstance(prepared.get("panels"), dict):
        raise ValueError("Expected historical prepared.json with panels and continuation_prompts")
    rows = []
    for name, panel_rows in prepared["panels"].items():
        if name in ("D", "P"):
            rows.extend(panel_rows)
    rows.extend(prepared.get("continuation_prompts", []))
    indexed = {}
    for row in rows:
        identity = row["prompt_id"]
        if identity in indexed:
            raise ValueError(f"Duplicate prepared prompt_id: {identity}")
        indexed[identity] = row
    return indexed


def _expected_jobs(cases, alias_map, protocol):
    rows = []
    grid = protocol["inference_grid"]
    main = grid["T1_core"] + grid["T2_core"]
    for checkpoint in main + grid["robustness_core"]:
        for case in cases:
            if case["case_id"] in alias_map:
                continue
            panel, transformation = case["panel"], case["protocol"]
            if transformation.startswith("L"):
                if checkpoint not in grid["input_output_factorial"]:
                    continue
                phase = "input_output_factorial"
            elif panel == "U22":
                if checkpoint not in main:
                    continue
                phase = "holdout_replication"
            else:
                phase = (
                    "primary_core_and_controls"
                    if checkpoint in main
                    else "replication_existing_checkpoint"
                )
            rows.append(
                {
                    "checkpoint_id": checkpoint,
                    "case_id": case["case_id"],
                    "panel": panel,
                    "protocol": transformation,
                    "phase": phase,
                    "draws": 64 if panel == "U22" else 32,
                    "checkpoint_kind": "branch" if checkpoint in grid["T2_core"] else "source",
                    "gpu_count": 1,
                }
            )
    return rows


def _verify_scope(cases, snapshot, protocol, differences):
    """Every fixed scene must retain every registered arm, independent of effects."""
    groups = defaultdict(set)
    for case in cases:
        groups[(case["panel"], case["base_scene_id"], case["family"])].add(case["protocol"])
    grid = protocol["inference_grid"]
    for (panel, scene, family), present in groups.items():
        if panel in ("D48", "U22"):
            required = set(
                grid[
                    "primary_protocols_cross"
                    if family == "cross_series"
                    else "primary_protocols_trend"
                ]
            )
            if panel == "D48":
                required.update(grid["factorial"])
        else:
            required = {"B0", "B1"} if panel == "CONTROL_CLEAN_REL" else {"O0", "A1"}
        _compare(sorted(required), sorted(present), f"scope.{panel}.{scene}.protocols", differences)
    for panel, source_panel, families in (
        ("D48", "D", ("cross_series", "trend")),
        ("U22", None, ("cross_series", "trend")),
        ("CONTROL_DUPLICATE", "D", ("duplicate_encoding",)),
    ):
        expected = sorted(
            r["base_scene_id"]
            for r in snapshot.values()
            if r.get("panel") == source_panel and r["family"] in families
        )
        actual = sorted(scene for p, scene, _ in groups if p == panel)
        _compare(expected, actual, f"scope.{panel}.scene_ids", differences)


def load_bundle(design_dir, prepared_path=None):
    """Recompile every manifest field and return independent public/audit arrays.

    Without prepared_path the selected source snapshot is verified; the returned
    receipt explicitly does not claim the server's prepared data was checked.
    Any mismatch raises BundleValidationError with machine-readable differences.
    """
    root = Path(design_dir)
    manifest = root / "manifests"
    protocol = json.loads((root / "protocol.json").read_text())
    cases = _read_jsonl(manifest / "cases.jsonl")
    audits = _read_jsonl(manifest / "audit_labels_NOT_FOR_MODEL.jsonl")
    predictions = _read_jsonl(manifest / "program_predictions.jsonl")
    aliases = _read_jsonl(manifest / "prompt_aliases.jsonl")
    jobs = _read_jsonl(manifest / "logical_jobs.jsonl")
    snapshot = _unique(
        _read_jsonl(root / "sources/selected_original_records.jsonl"), "prompt_id", "source"
    )
    records = snapshot
    if prepared_path is not None:
        records = _prepared_records(json.loads(Path(prepared_path).read_text()))
    by_case = _unique(cases, "case_id", "case")
    by_audit = _unique(audits, "case_id", "audit")
    by_prediction = _unique(predictions, "case_id", "prediction")
    alias_map = {a["case_id"]: a["owner_case_id"] for a in aliases}
    if len(alias_map) != len(aliases):
        raise ValueError("Duplicate prompt alias")
    differences, rebuilt_predictions = [], []
    _verify_scope(cases, snapshot, protocol, differences)
    for case in cases:
        parent = case["parent_prompt_id"]
        if parent not in records:
            differences.append({"path": case["case_id"], "missing_prepared_parent": parent})
            continue
        rebuilt, audit, prediction = compile_case(records[parent], case["panel"], case["protocol"])
        _compare(case, rebuilt, f"cases.{case['case_id']}", differences)
        _compare(by_audit.get(case["case_id"]), audit, f"audits.{case['case_id']}", differences)
        if prediction is not None:
            _compare(
                by_prediction.get(case["case_id"]),
                prediction,
                f"predictions.{case['case_id']}",
                differences,
            )
            if case["family"] == "cross_series":
                prediction["b1_diagnostic"] = b1_diagnostic(
                    public_from_record(records[parent]), audit["corrupted_coordinate"]
                )
            rebuilt_predictions.append(prediction)
    if set(by_audit) != set(by_case) or set(by_prediction) != {
        c["case_id"] for c in cases if c["kind"] == "repair"
    }:
        raise ValueError("Manifest audit/prediction identities do not cover the expected cases")
    # Alias authority requires byte-equivalent public text, order and semantic contract.
    for alias in aliases:
        source, owner = by_case[alias["case_id"]], by_case[alias["owner_case_id"]]
        if alias["independent_replicate"] is not False or owner["case_id"] in alias_map:
            raise ValueError("Aliases must point directly to one shared evidence owner")
        for key in (
            "prompt",
            "output_order",
            "H_display",
            "b_display",
            "observed_world_public",
            "operation",
            "panel",
            "base_scene_id",
        ):
            _compare(owner[key], source[key], f"aliases.{source['case_id']}.{key}", differences)
        _compare(
            by_audit[source["case_id"]]["truth_world"],
            by_audit[owner["case_id"]]["truth_world"],
            "alias_truth",
            differences,
        )
    inferred_aliases = {}
    owners = {}
    for case in cases:
        identity = (
            case["panel"],
            case["base_scene_id"],
            case["prompt_identity"],
            tuple(case["output_order"]),
        )
        if identity in owners:
            inferred_aliases[case["case_id"]] = owners[identity]
        else:
            owners[identity] = case["case_id"]
    _compare(inferred_aliases, alias_map, "alias_map", differences)
    expected_jobs = _expected_jobs(cases, alias_map, protocol)

    def job_key(row):
        return row["checkpoint_id"] + ":" + row["case_id"]

    keyed_jobs = {job_key(j): j for j in jobs}
    if len(keyed_jobs) != len(jobs):
        raise ValueError("Duplicate scheduled checkpoint/case cell")
    _compare({job_key(j): j for j in expected_jobs}, keyed_jobs, "jobs", differences)
    workload = protocol["workload"]
    actual = {
        "total_cases": len(cases),
        "aliases": len(aliases),
        "unique_case_prompts": len(owners),
        "job_cells": len(jobs),
        "scheduled_completions_after_exact_prompt_alias_reuse": sum(j["draws"] for j in jobs),
    }
    for key, value in actual.items():
        _compare(workload[key], value, f"workload.{key}", differences)
    _compare(
        workload, json.loads((manifest / "workload.json").read_text()), "workload_file", differences
    )
    for category, key in (("by_phase", "phase"), ("by_checkpoint", "checkpoint_id")):
        groups = defaultdict(lambda: {"cells": 0, "completions": 0})
        for job in jobs:
            groups[job[key]]["cells"] += 1
            groups[job[key]]["completions"] += job["draws"]
        _compare(workload[category], dict(groups), "workload." + category, differences)
    for panel, expected in (
        ("D48", {"cross_series": 24, "trend": 24}),
        ("U22", {"cross_series": 11, "trend": 11}),
    ):
        scenes = {(c["base_scene_id"], c["family"]) for c in cases if c["panel"] == panel}
        _compare(
            expected, dict(Counter(f for _, f in scenes)), "panel_counts." + panel, differences
        )
    if differences:
        raise BundleValidationError(differences)
    files = [
        root / "protocol.json",
        root / "sources/selected_original_records.jsonl",
        *sorted(manifest.glob("*")),
    ]
    verification = {
        "status": "PASS",
        "all_fields_match": True,
        "recompiled_from_prepared": prepared_path is not None,
        "prepared_path": str(Path(prepared_path).resolve()) if prepared_path is not None else None,
        "prepared_sha256": hashlib.sha256(Path(prepared_path).read_bytes()).hexdigest()
        if prepared_path is not None
        else None,
        "source_snapshot_only": prepared_path is None,
        "counts": actual,
        "program_predictions_verified": len(predictions),
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "u22_exposure_status": "REQUIRES_EXECUTED_JOB_AUDIT",
        "truth_used_by_main_renderer": False,
        "new_model_outputs": 0,
    }
    return {
        "cases": cases,
        "audits": audits,
        "predictions": rebuilt_predictions,
        "aliases": aliases,
        "jobs": jobs,
        "protocol": protocol,
        "verification": verification,
    }
