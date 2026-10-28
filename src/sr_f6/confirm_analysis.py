"""Analyze verified independent G8 confirmation; never authorize training.

Original TRAIN coverage belongs to its historical producer. The new confirmation
has its own data audit, source and raw records; neither is renamed DEV_CAL.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from srf6_ref.census import summarize
from srf6_ref.core import FAMILIES, exact

from . import census_support, confirm_evidence, confirm_registration, group_diagnostics
from .census_evidence import _strict_json
from .common import digest, safe_evidence, write_once
from .confirm_data import MASTER_SEED, SPLIT
from .group_interface_audit import (
    candidate_certificates,
    cross_product_pair,
    matched_oracle_control,
)
from .interface_v3_audit import j_gap_breakdown

SCHEMA = "srf6-confirm-analysis-v1"
DIRECTORY = "CONFIRM/ANALYSIS"
PARENTS = ("b0", "b1", "b2")
FACTORS = ("P0", "P1")


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _read(path, pin, *, lines=False):
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == pin, "Bound analysis input changed")
    return (
        [_strict_json(line) for line in raw.splitlines() if line.strip()]
        if lines
        else _strict_json(raw)
    )


class _Inputs:
    def __init__(self, root, evidence):
        self.root = root
        self.local = {r["path"]: r["sha256"] for r in evidence["inputs"]}
        self.external = {r["path"]: r["sha256"] for r in evidence["external_inputs"]}
        _require(
            len(self.local) == len(evidence["inputs"])
            and len(self.external) == len(evidence["external_inputs"]),
            "Duplicate evidence reference",
        )

    def read(self, relative, *, lines=False):
        return _read(safe_evidence(self.root, relative), self.local[relative], lines=lines)

    def original(self, registration, relative, *, lines=False):
        origin = Path(registration["source_root"])
        _require(
            origin.is_absolute() and origin.resolve(strict=True) == origin,
            "Original producer root must be canonical",
        )
        path = safe_evidence(origin, relative)
        entries = [r for r in registration["upstream_files"] if r["path"] == str(path)]
        _require(
            len(entries) == 1 and self.external[str(path)] == entries[0]["sha256"],
            "Original TRAIN source is not bound by this confirmation",
        )
        return _read(path, entries[0]["sha256"], lines=lines), dict(
            path=str(path), sha256=entries[0]["sha256"]
        )


def _matrix(records, tasks):
    expected_qids = {
        f"{SPLIT}_{family}_{index:04d}_{factor}_v{view}"
        for family in FAMILIES
        for index in range(8)
        for factor in FACTORS
        for view in range(4)
    }
    _require(
        len(tasks) == 384
        and {t["qid"] for t in tasks} == expected_qids
        and all(t["split"] == SPLIT for t in tasks),
        "Exact independent confirmation task matrix required",
    )
    expected_rows = {(p, q, d) for p in PARENTS for q in expected_qids for d in range(8)}
    _require(
        len(records) == 9216
        and {(r["parent"], r["qid"], r["draw"]) for r in records} == expected_rows
        and all(type(r["draw"]) is int and r.get("split") == SPLIT for r in records),
        "All 9216 new confirmation rows required",
    )


def _followups(plan, tasks):
    _require(
        plan["gates"]["review_seed"] == 2026102701
        and plan["gates"]["new_independent_response_review"] == 360,
        "Original independent-review sampling rule changed",
    )
    _require(
        plan["followup_requirements"]["independent_response_review"]
        == dict(
            count=360,
            seed=2026102701,
            new_original_answers_only=True,
            authorization="SEPARATE_INDEPENDENT_REVIEW_REQUIRED",
        ),
        "New independent review was not prospectively required",
    )
    selection = plan["followup_requirements"]["numerical_replay_selection"]
    expected = dict(
        qid=min(t["qid"] for t in tasks),
        parents=list(PARENTS),
        draws=list(range(8)),
        sequences=24,
        new_generated_responses=0,
        optimizer_updates=0,
        authorization="SEPARATE_ACTUAL_NUMERICAL_ADMISSION_REQUIRED",
    )
    _require(
        digest(selection) == digest(expected),
        "Numerical replay selection was not frozen prospectively",
    )
    return dict(
        independent_review=dict(
            status="REQUIRED_NOT_RUN",
            seed=2026102701,
            responses=360,
            per_parent_factor=60,
            per_family_in_cell=10,
            root_cap_per_parent_factor=4,
            new_generated_responses=0,
        ),
        actual_numerical_gate=dict(status="REQUIRED_NOT_RUN", selection=copy.deepcopy(selection)),
        pilot_admission="NOT_SIGNED",
        formal_training="NOT_AUTHORIZED",
    )


def _interface(records, roots, census):
    """Reuse split-neutral mathematics; do not relabel tasks to call a DEV_CAL API."""
    tasks = {t["qid"]: t for root in roots for views in root["tasks"].values() for t in views}
    answers = {
        (r["parent"], r["qid"], r["draw"]): group_diagnostics._parsed_answer(
            r.get("raw_text", r.get("full_text"))
        )
        for r in records
    }
    finishes = {(r["parent"], r["qid"], r["draw"]): r["finished"] for r in records}
    certificates, pairs = [], []
    for root in roots:
        for factor in FACTORS:
            views = root["tasks"][factor]
            anchor = views[0]
            forced = anchor["engineered_collision_anchor"]
            certificates.append(dict(candidate_certificates(views), forced=forced))
            for parent in PARENTS:
                left = [answers[parent, anchor["qid"], draw] for draw in range(8)]
                for edit in views[1:]:
                    right = [answers[parent, edit["qid"], draw] for draw in range(8)]
                    both_finished = sum(
                        bool(finishes[parent, anchor["qid"], a])
                        and bool(finishes[parent, edit["qid"], b])
                        for a in range(8)
                        for b in range(8)
                    )
                    pairs.append(
                        dict(
                            parent=parent,
                            factor=factor,
                            root_id=root["root_id"],
                            family=anchor["family"],
                            forced=forced,
                            variant=edit["variant"],
                            both_finished_pairs=both_finished,
                            at_least_one_unfinished_pairs=64 - both_finished,
                            **cross_product_pair(
                                left,
                                right,
                                exact(anchor["gold_answer"]),
                                exact(edit["gold_answer"]),
                            ),
                        )
                    )
    strata = defaultdict(list)
    for row in pairs:
        for family in ("ALL", row["family"]):
            for population in ("ALL", "FORCED" if row["forced"] else "NONFORCED"):
                strata[f"{row['parent']}/{row['factor']}/{family}/{population}"].append(row)
    summaries = {}
    for key, rows in sorted(strata.items()):
        denominator = 64 * len(rows)
        counts = {k: sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]}
        summaries[key] = dict(
            roots=len({r["root_id"] for r in rows}),
            view_pairs=len(rows),
            cross_product_denominator=denominator,
            counts=counts,
            rates_all_pairs={k: v / denominator for k, v in counts.items()},
            both_finished_pairs=sum(r["both_finished_pairs"] for r in rows),
            at_least_one_unfinished_pairs=sum(r["at_least_one_unfinished_pairs"] for r in rows),
        )
    return dict(
        status="DESCRIPTIVE_COMPLETE_CONFIRMATION",
        split=SPLIT,
        source_rows=len(records),
        roots=len(roots),
        tasks=len(tasks),
        forced_roots=sum(r["tasks"]["P0"][0]["engineered_collision_anchor"] for r in roots),
        root_certificates=certificates,
        all_draw_pairs=pairs,
        pair_summaries=summaries,
        matched_oracle_control=matched_oracle_control(
            census["signature_predictions"], tasks, answers=answers
        ),
        model_calls=0,
        optimizer_updates=0,
        training_allowed=False,
        additional_scientific_gate=False,
        claim_limit=(
            "Finite-library coverage, actual errors and credit are distinct. "
            "G8 cross products are not 64 independent observations. Matched signatures "
            "are descriptive, not internal-program identification or learning effects."
        ),
    )


def _analyze(root, evidence):
    _require(
        evidence["status"] == "VERIFIED_STORED_CONFIRMATION_NOT_GATE_SIGNED"
        and evidence["split"] == SPLIT
        and evidence["source_rows"] == 9216
        and evidence["actual_recorded_calls"] == 9216,
        "Complete original raw confirmation validation required",
    )
    token = evidence["saved_token_validation"]
    _require(
        token["status"] == "PASS"
        and token["checked_responses"] == 9216
        and token["checked_tokens"] == evidence["accounting"]["generated_tokens"]
        and token["new_model_calls"] == 0,
        "Complete original-token validation required",
    )
    inputs = _Inputs(root, evidence)
    registration = inputs.read("REGISTRATION.json")
    _require(
        inputs.local["REGISTRATION.json"] == evidence["registration_sha256"],
        "Analysis registration identity differs",
    )
    producer, predecessor = evidence["input_producer"], evidence["predecessor_screen"]
    _require(
        producer == evidence["consumer_source"]
        and producer["source_manifest"] == registration["source_manifest"]
        and producer["revision"] == registration["revision"]
        and producer["deployment"] == registration["deployment"]
        and predecessor == registration["predecessor_screen"]
        and producer["deployment"] != predecessor["deployment"]
        and producer["revision"] != predecessor["revision"],
        "New confirmation and predecessor source identities must remain distinct",
    )
    plan = inputs.read(registration["spec"]["path"])
    _require(
        plan["source_split"] == SPLIT and plan["master_seed"] == MASTER_SEED,
        "Independent confirmation plan differs",
    )
    tasks = inputs.read(f"DATA/{SPLIT}/TASKS_PRIVATE.jsonl", lines=True)
    roots = inputs.read("DATA/ROOTS.jsonl", lines=True)
    _require(
        len(roots) == 48 and [t for r in roots for v in r["tasks"].values() for t in v] == tasks,
        "New confirmation roots and task manifest differ",
    )
    new_audit = inputs.read("DATA/AUDIT.json")
    _require(
        new_audit["status"] == "PASS"
        and new_audit["split"] == SPLIT
        and new_audit["master_seed"] == MASTER_SEED
        and new_audit["roots"] == 48
        and new_audit["questions"] == 384
        and new_audit["forced_anchors"] == 18
        and new_audit["historical_world_exclusion"]["status"] == "PASS"
        and new_audit["historical_world_exclusion"]["collisions"] == [],
        "Actual independent confirmation data audit did not pass",
    )
    records = evidence["records"]
    _matrix(records, tasks)
    followups = _followups(plan, tasks)
    old_audit, audit_ref = inputs.original(registration, "DATA/CPU_AUDIT.json")
    old_roots, roots_ref = inputs.original(
        registration, "DATA/ROOT_CERTIFICATES_PRIVATE.jsonl", lines=True
    )
    train = [r for r in old_roots if r["tasks"]["P0"][0]["split"] == "TRAIN"]
    expected_train = {f"TRAIN_{f}_{i:04d}" for f in FAMILIES for i in range(48)}
    _require(
        len(train) == 288 and {r["root_id"] for r in train} == expected_train,
        "Exactly the original 288 TRAIN roots required",
    )
    summary = summarize(records, MASTER_SEED)
    summary.update(
        status="COMPLETE_NOT_GATE_SIGNED",
        source_rows=9216,
        frozen_spec_sha256=digest(plan),
        frozen_spec_file_sha256=registration["spec"]["sha256"],
        optimizer_updates=0,
        split=SPLIT,
        real_measurement_review="NOT_RUN",
    )
    support = census_support.assess_support(summary, old_audit, train_roots=train)
    support.update(
        evidence_origin="VERIFIED_NEW_CONFIRMATION_RAW_AND_BOUND_ORIGINAL_TRAIN",
        input_producer=copy.deepcopy(producer),
        data_audit_scope=dict(
            confirmation=dict(
                path="DATA/AUDIT.json",
                sha256=inputs.local["DATA/AUDIT.json"],
                split=SPLIT,
                meaning="ACTUAL_NEW_CONFIRMATION_DATA",
            ),
            original=dict(
                producer_identity_sha256=registration["original_identity_sha256"],
                cpu_audit=audit_ref,
                train_roots=roots_ref,
                meaning="ORIGINAL_TRAIN_COVERAGE_AND_HISTORICAL_DEV_CAL_ONLY",
            ),
            historical_dev_cal_is_new_confirmation=False,
        ),
    )
    census = group_diagnostics.census_diagnostics(records, tasks, master=MASTER_SEED, split=SPLIT)
    _require(
        census["status"] == "DERIVED_COMPLETE_SUPPLIED_MATRIX"
        and census["full_registered_task_manifest"] is True
        and len(census["groups"]) == 1152
        and all(g["credit_status"] == "COMPLETE_G8" for g in census["groups"]),
        "Complete new G8 diagnostic matrix required",
    )
    status = {
        "PASS": "CONFIRM_SUPPORT_PASS_REVIEW_PENDING",
        "INSUFFICIENT_REGISTERED_EXPOSURE": "CONFIRM_FAIL_STOP",
        "INVALID_SUPPORT_EVIDENCE": "TECHNICAL_INVALID",
    }[support["status"]]
    files = {
        "VALIDATION.json": {
            k: v for k, v in evidence.items() if k not in ("records", "raw_by_row")
        },
        "CENSUS_SUMMARY.json": summary,
        "SUPPORT.json": support,
        "GROUP_MATH.json": group_diagnostics.mathematical_diagnostics(roots),
        "GROUP_DIAGNOSTICS.json": census,
        "GROUP_INTERFACE.json": _interface(records, roots, census),
        "J_GAP.json": j_gap_breakdown(records),
    }
    report = dict(
        schema=SCHEMA,
        status=status,
        execution_origin=evidence.get("execution_origin", "ACTUAL_STORED_CONFIRMATION_ANALYSIS"),
        split=SPLIT,
        master_seed=MASTER_SEED,
        responses=9216,
        registration_sha256=evidence["registration_sha256"],
        input_producer=copy.deepcopy(producer),
        predecessor_screen=copy.deepcopy(predecessor),
        original_producer_identity_sha256=registration["original_identity_sha256"],
        records_sha256=digest(records),
        support_status=support["status"],
        support_failures=copy.deepcopy(support["failures"]),
        followup_requirements=followups,
        raw_finish_counts=dict(Counter(r["finish_reason"] for r in records)),
        accounting=copy.deepcopy(evidence["accounting"]),
        saved_token_validation=copy.deepcopy(token),
        complete_G8_credit="DESCRIPTIVELY_RECOMPUTED_NOT_LEARNING_VALIDATED",
        effect_estimated="NOT_RUN",
        on_scientific_failure="STOP_PRESERVE_NO_TUNING",
        on_technical_failure="PRESERVE_NO_AUTOMATIC_RESAMPLING",
    )
    return report, files


def assess(root):
    """Validate saved raw facts first, then write immutable non-admission reports."""
    # Reject unauthorized destinations before the technical-error writer exists.
    root = confirm_registration._root(root)
    _require(
        root.is_absolute()
        and root.resolve(strict=True) == root
        and root.is_dir()
        and not any(p.is_symlink() for p in (root, *root.parents)),
        "Canonical existing confirmation root required",
    )
    output = root / DIRECTORY
    _require(
        not any(p.is_symlink() for p in (output, output.parent)),
        "Symlinked analysis output is forbidden",
    )
    if output.exists():
        raise FileExistsError("Preserve existing or partial confirmation analysis; no rescan")
    # Reserve the analysis once, before the expensive complete raw replay.
    output.mkdir(parents=True, exist_ok=False)
    try:
        evidence = confirm_evidence.validate_saved_confirm(root)
        report, files = _analyze(root, evidence)
    except (OSError, ValueError, TypeError, KeyError, ImportError, PermissionError) as exc:
        report = dict(
            schema=SCHEMA,
            status="TECHNICAL_INVALID",
            split=SPLIT,
            error=type(exc).__name__ + ": " + str(exc),
            interpretation="No support or admission signed from incomplete evidence",
            on_technical_failure="PRESERVE_NO_AUTOMATIC_RESAMPLING",
        )
        files = {}
    report.update(
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        gate_status="NOT_SIGNED",
        new_model_calls=0,
        optimizer_updates=0,
    )
    refs = {}
    for name, value in files.items():
        write_once(output / name, value)
        payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        refs[name] = dict(path=f"{DIRECTORY}/{name}", sha256=hashlib.sha256(payload).hexdigest())
    report["reports"] = refs
    write_once(output / "ASSESSMENT.json", report)
    payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    write_once(
        output / "SEAL.json",
        dict(
            schema="srf6-confirm-analysis-seal-v1",
            status=report["status"],
            assessment=dict(
                path=f"{DIRECTORY}/ASSESSMENT.json", sha256=hashlib.sha256(payload).hexdigest()
            ),
            registration_sha256=report.get("registration_sha256"),
            gate_status="NOT_SIGNED",
            training_allowed=False,
            generation_allowed=False,
        ),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.root)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "training_allowed", "new_model_calls")},
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "CONFIRM_SUPPORT_PASS_REVIEW_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
