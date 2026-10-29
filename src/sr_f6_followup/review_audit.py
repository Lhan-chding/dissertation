"""Verify actual independent 360-answer review; never manufacture judgments.

Use prepare_execution before a real independent task. Each execution lives in
REVIEW/EXECUTIONS/<task_id> with ATTEMPT.json and exactly one SUBMISSION.json or
ADJUDICATION.json. All old attempts, failures, and partial records remain visible.
The producer's two-stage review schemas/rules are unchanged. Typed traces bind
actual visible request/response contents; declared identities and hidden system
context cannot be authenticated by hashes. MODEL_TEXT is never relabeled HUMAN.

aggregate_review is a read-only interim assessment for possible adjudication.
seal is a one-time terminal publication, including negative results. It accepts
no report/status argument. verify_completed replays all inputs and compares the
sealed report. None of these functions authorize training or final evaluation.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

from sr_f6 import review_audit as rules
from sr_f6.common import now, safe_evidence
from sr_f6.review_audit import (
    LABELS,
    _adjudicate,
    _artifact,
    _constraints,
    _event,
    _index,
    _oracle,
    _require,
    _submission,
    _time,
)
from sr_f6.review_package import _directory, _fsync_directory, _payload

from . import review_package, sampling, source

VERSION = "srf6-followup-independent-review-audit-v2"
TRACE_SCHEMA = "srf6-review-execution-trace-v1"
REVIEW_DIRECTORY = "REVIEW/EXECUTIONS"
DESTINATION = "REVIEW/AUDIT"
_CACHE = {}


def _synthetic(value):
    if isinstance(value, dict):
        return any(
            (
                key in ("scope", "execution_origin", "status")
                and isinstance(item, str)
                and item.startswith("SYNTHETIC")
            )
            or _synthetic(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(_synthetic(item) for item in value)


class _Facts(rules._ArtifactReader):
    def __init__(self, root):
        super().__init__(root)
        self.synthetic = False

    def bound(self, reference, *, lines=False):
        value = _artifact(self, reference, lines=lines)
        self.synthetic |= _synthetic(value)
        return value

    def read(self, relative, *, reference=None, opaque=False):
        path = safe_evidence(self.root, relative)
        raw = path.read_bytes()
        ref = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest())
        _require(reference is None or ref == reference, "Evidence identity changed")
        _require(self.references.setdefault(relative, ref) == ref, "Conflicting evidence pin")
        if path.suffix == ".json":
            self.synthetic |= _synthetic(rules._json(raw))
        return _artifact(self, ref, opaque=opaque)


def _visible(facts, reference):
    return facts.bound(reference, lines=reference["path"].endswith(".jsonl"))


def _method_refs(package, *, adjudication=False):
    return [package["review_contract"], *([package["review_method"]] if adjudication else [])]


def _method_materials(facts, package, *, adjudication=False):
    materials = []
    for ref in _method_refs(package, adjudication=adjudication):
        facts.read(ref["path"], reference=ref, opaque=True)
        raw = safe_evidence(facts.root, ref["path"]).read_text()
        materials.append(
            dict(reference=ref, content=json.loads(raw) if ref["path"].endswith(".json") else raw)
        )
    return materials


def _trace(facts, submission, package, *, adjudication=False):
    reviewer = submission["reviewer"]
    _require(
        submission.get("execution_origin") == "ACTUAL_REVIEW"
        and reviewer.get("type") in ("HUMAN", "MODEL_TEXT"),
        "Actual independent HUMAN/MODEL_TEXT execution required",
    )
    trace = facts.bound(reviewer["transcript"])
    source = facts.bound(reviewer["source"])
    method = package["adjudication_method" if adjudication else "review_method"]
    _require(reviewer.get("method") == method, "Reviewer method differs from preregistration")
    facts.read(reviewer["method"]["path"], reference=reviewer["method"], opaque=True)
    _require(
        trace.get("method_content") == safe_evidence(facts.root, method["path"]).read_text(),
        "Actual trace must include the complete registered method",
    )
    _require(
        trace.get("method_attachments")
        == _method_materials(facts, package, adjudication=adjudication),
        "Actual trace must include all registered method attachments",
    )
    _require(
        trace.get("schema") == TRACE_SCHEMA and trace.get("execution_origin") == "ACTUAL_REVIEW",
        "Complete typed actual execution trace required",
    )
    for key, expected in (
        ("reviewer_id", reviewer["id"]),
        ("reviewer_type", reviewer["type"]),
        ("identity", reviewer["identity"]),
        ("model_identity", reviewer.get("model_identity")),
        ("method", reviewer["method"]),
        ("source", reviewer["source"]),
    ):
        _require(
            trace.get(key) == expected, "Actual trace reviewer/method identity differs: " + key
        )
    for key in ("instance_id", "task_id"):
        _require(
            isinstance(trace.get(key), str)
            and bool(trace[key].strip())
            and source.get(key) == trace[key],
            "Actual review instance/task identity required",
        )
    _require(
        source.get("reviewer_type") == reviewer["type"]
        and source.get("execution_origin") == "ACTUAL_REVIEW",
        "Reviewer source is not an actual independent execution",
    )
    if reviewer["type"] == "HUMAN":
        _require(trace.get("model_identity") is None, "Human trace cannot invent a model identity")
    else:
        _require(
            bool(trace.get("model_identity")),
            "Actual model identity or explicit NOT_EXPOSED required",
        )
    if adjudication:
        response = facts.bound(submission["decision_response"])
        _require(
            response == {k: submission[k] for k in ("items", "completed_at")},
            "Actual adjudication response differs from final manifest",
        )
        stages = [("ADJUDICATION", submission["input_references"], submission["decision_response"])]
    else:
        stages = [
            ("BLIND_EXTRACTION", [submission["blind_input"]], submission["blind_extraction"]),
            (
                "INDEPENDENT_JUDGMENT",
                [
                    submission["blind_input"],
                    package["judgment_tasks"],
                    submission["blind_extraction"],
                ],
                submission["judgments"],
            ),
        ]
    events = trace.get("events", [])
    _require(len(events) == len(stages), "Complete visible execution stage trace required")
    previous = None
    for event, (stage, inputs, output) in zip(events, stages, strict=True):
        _require(
            set(event) == {"stage", "started_at", "completed_at", "inputs", "output"}
            and event["stage"] == stage,
            "Unregistered review trace stage/fields",
        )
        started, completed = (
            rules._time(event["started_at"]),
            rules._time(event["completed_at"]),
        )
        _require(
            started <= completed and (previous is None or previous < started),
            "Review execution chronology differs",
        )
        previous = completed
        expected = [dict(reference=ref, content=_visible(facts, ref)) for ref in inputs]
        _require(
            event["inputs"] == expected,
            "Trace inputs omit/change the actual payload or expose extra material",
        )
        actual_output = _visible(facts, output)
        _require(
            event["output"] == dict(reference=output, content=actual_output),
            "Trace output differs from actual recorded response",
        )
        _require(
            actual_output.get("completed_at") == event["completed_at"],
            "Trace completion differs from sealed response",
        )
        if not adjudication and stage == "INDEPENDENT_JUDGMENT":
            _require(
                actual_output.get("started_at") == event["started_at"],
                "Judgment began before its recorded actual request",
            )
    return dict(
        reviewer_id=reviewer["id"],
        reviewer_type=reviewer["type"],
        instance_id=trace["instance_id"],
        task_id=trace["task_id"],
        transcript=reviewer["transcript"],
        visible_payloads_verified=True,
    )


def _inventory(reader, verification):
    directory = _directory(reader.root, REVIEW_DIRECTORY)
    _require(directory.is_dir(), "Actual review execution inventory missing")
    reviews, adjudications, attempts = [], [], {}
    for child in sorted(directory.iterdir()):
        _require(child.is_dir() and not child.is_symlink(), "Unexpected review execution entry")
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", child.name), "Unsafe task ID")
        relative = child.relative_to(reader.root).as_posix()
        attempt = reader.read(relative + "/ATTEMPT.json")
        _require(
            attempt.get("schema") == "srf6-followup-review-attempt-v1"
            and attempt.get("task_id") == child.name
            and attempt.get("consumer") == verification["consumer"]
            and attempt.get("input_producer") == verification["input_producer"]
            and attempt.get("package") == verification["package_ref"]
            and attempt.get("method")
            == verification["artifacts"]["package"][
                "adjudication_method" if attempt.get("kind") == "ADJUDICATION" else "review_method"
            ]
            and attempt.get("execution_origin") == "ACTUAL_REVIEW",
            "Execution attempt differs from actual consumer/producer/package",
        )
        _require(
            attempt.get("method_attachments")
            == _method_refs(
                verification["artifacts"]["package"],
                adjudication=attempt.get("kind") == "ADJUDICATION",
            ),
            "Attempt method attachments differ from preregistration",
        )
        for path in sorted(child.rglob("*")):
            _require(not path.is_symlink(), "Symlinked review evidence forbidden")
            if path.is_dir():
                continue
            _require(path.is_file(), "Unexpected review evidence object")
            _require("FAILURE" not in path.name.upper(), "Unresolved review failure preserved")
            reader.read(path.relative_to(reader.root).as_posix(), opaque=True)
        candidates = [
            name for name in ("SUBMISSION.json", "ADJUDICATION.json") if (child / name).exists()
        ]
        _require(len(candidates) == 1, "Incomplete or ambiguous review execution preserved")
        name = candidates[0]
        _require(
            attempt.get("kind") == ("REVIEW" if name == "SUBMISSION.json" else "ADJUDICATION"),
            "Attempt kind differs from completed execution",
        )
        ref = reader.references[relative + "/" + name]
        (reviews if name == "SUBMISSION.json" else adjudications).append(ref)
        attempts[child.name] = attempt
    return reviews, adjudications, attempts


def _membership(root):
    directory = _directory(root, REVIEW_DIRECTORY)
    _require(directory.is_dir(), "Actual review execution inventory missing")
    result = []
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), "Symlinked review evidence forbidden")
        stat = path.stat()
        result.append(
            (
                str(path.relative_to(root)),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_mode,
            )
        )
    return result


def _adjudication_inputs(package, review_refs, reader, findings_ref, program_ref):
    refs = [package["blind"], package["judgment_tasks"]]
    for ref in review_refs:
        submission = reader.bound(ref)
        refs.extend(
            [
                ref,
                submission["blind_extraction"],
                submission["judgments"],
                submission["reviewer"]["source"],
                submission["reviewer"]["transcript"],
            ]
        )
    refs.extend([findings_ref, program_ref])
    return list({ref["path"]: ref for ref in refs}.values())


def _program_packet(package, claims, findings, package_ref):
    """Reveal complete stored PROGRAM claims even for ambiguity-only findings."""
    index = _index(claims)
    ids = sorted({row["anonymous_id"] for row in findings if row["anonymous_id"] is not None})
    _require(set(ids) <= set(index), "Finding lacks its original program source row")
    return dict(
        schema="srf6-followup-adjudication-program-claims-v1",
        package=package_ref,
        source_program_claims=package["program_claims"],
        status="ORIGINAL_PROGRAM_CLAIMS_NOT_ADJUDICATION_DECISIONS",
        global_finding_ids=sorted(
            row["finding_id"] for row in findings if row["anonymous_id"] is None
        ),
        items=[
            dict(
                index[key],
                finding_ids=sorted(
                    row["finding_id"] for row in findings if row["anonymous_id"] == key
                ),
            )
            for key in ids
        ],
    )


def _check_adjudications(reader, refs, review_refs, package_ref, package, findings, latest, traces):
    originals = {t["instance_id"] for t in traces if t["kind"] == "REVIEW"}
    for ref in refs:
        value = reader.bound(ref)
        finding_ref = value.get("findings")
        _require(isinstance(finding_ref, dict), "Fixed original findings packet required")
        _require(
            reader.bound(finding_ref)
            == dict(
                schema="srf6-followup-review-findings-v1",
                package=package_ref,
                reviews=review_refs,
                findings=findings,
            ),
            "Adjudication findings packet differs",
        )
        program_ref = value.get("program_claims")
        _require(isinstance(program_ref, dict), "Complete program claims packet required")
        _require(
            reader.bound(program_ref)
            == _program_packet(
                package, reader.bound(package["program_claims"]), findings, package_ref
            ),
            "Adjudication program claims differ from original recorded evidence",
        )
        inputs = _adjudication_inputs(package, review_refs, reader, finding_ref, program_ref)
        _require(
            value.get("input_references") == inputs,
            "Adjudication must receive all original claims, texts, tasks and findings",
        )
        trace = reader.bound(value["reviewer"]["transcript"])
        _require(
            _time(trace["events"][0]["started_at"]) > latest,
            "Adjudication must start after all original judgments complete",
        )
        _require(
            trace["instance_id"] not in originals,
            "Adjudication requires a separate instance from every original reviewer",
        )
        for decision in value["items"]:
            _require(
                all(item in inputs for item in decision.get("evidence", [])),
                "Adjudication evidence must have been in its actual visible inputs",
            )


def aggregate_review(root):
    """Read all executions, recompute findings, preserve every original claim."""
    verification = review_package.verify_package(root)
    root = source._root(root)
    reader = _Facts(root)
    reader.references.update({ref["path"]: ref for ref in verification["inputs"]})
    artifacts = verification["artifacts"]
    package, ledger, tasks = (artifacts[k] for k in ("package", "ledger", "tasks"))
    package_ref = verification["package_ref"]
    public, selected = _index(artifacts["blind"]), _index(ledger["selected"])
    before = _membership(root)
    review_refs, adjudication_refs, attempts = _inventory(reader, verification)
    traces = []
    for ref in [*review_refs, *adjudication_refs]:
        submission = reader.bound(ref)
        trace = _trace(reader, submission, package, adjudication=ref in adjudication_refs)
        _require(trace["task_id"] in attempts, "Trace task is outside actual inventory")
        attempt = attempts[trace["task_id"]]
        _require(
            ref["path"]
            == f"{REVIEW_DIRECTORY}/{trace['task_id']}/"
            + ("ADJUDICATION.json" if ref in adjudication_refs else "SUBMISSION.json")
            and attempt.get("instance_id") == trace["instance_id"]
            and attempt.get("reviewer_id") == trace["reviewer_id"],
            "Trace differs from its pre-execution task/instance binding",
        )
        _require(
            _time(attempt["started_at"])
            <= _time(reader.bound(trace["transcript"])["events"][0]["started_at"]),
            "Attempt must precede the actual visible execution",
        )
        if ref in adjudication_refs:
            _require(
                submission.get("findings") == attempt.get("findings")
                and submission.get("program_claims") == attempt.get("program_claims")
                and submission.get("input_references") == attempt.get("input_references"),
                "Adjudication inputs differ from their pre-execution binding",
            )
        trace["kind"] = "ADJUDICATION" if ref in adjudication_refs else "REVIEW"
        traces.append(trace)
    _require(len({r["task_id"] for r in traces}) == len(traces), "Repeated execution task identity")
    _require(
        isinstance(review_refs, (list, tuple)) and isinstance(adjudication_refs, (list, tuple)),
        "Review/adjudication references must be explicit lists",
    )
    submissions, reviewers, findings, labels, oracles = [], {}, [], {}, {}
    latest = datetime.min.replace(tzinfo=timezone.utc)
    independent_coverage, all_coverage, seen_submissions = set(), set(), set()
    for ref in review_refs:
        _require(ref["sha256"] not in seen_submissions, "Duplicate review submission")
        seen_submissions.add(ref["sha256"])
        submission, items, issues = _submission(reader, ref, package_ref, package, public)
        reviewer_id = submission["reviewer"]["id"]
        if reviewer_id in reviewers:
            identity_keys = ("id", "type", "identity", "model_identity", "method", "source")
            _require(
                all(
                    reviewers[reviewer_id].get(k) == submission["reviewer"].get(k)
                    for k in identity_keys
                ),
                "Same reviewer ID changed its method or identity",
            )
        reviewers[reviewer_id] = submission["reviewer"]
        claim_id = reviewer_id + "@" + ref["sha256"]
        submissions.append(
            dict(
                reference=ref,
                claim_id=claim_id,
                reviewer=submission["reviewer"],
                anonymous_ids=sorted(items),
                execution_origin=submission["execution_origin"],
                independence=submission["independence"],
            )
        )
        findings.extend(issues)
        all_coverage.update(items)
        if all(submission["independence"].values()):
            independent_coverage.update(items)
        labels[claim_id] = {key: value["labels"] for key, value in items.items()}
        oracles[claim_id] = {key: value["oracle"] for key, value in items.items()}
        judgment = _artifact(reader, submission["judgments"])
        latest = max(latest, _time(judgment["completed_at"]))
    agreements = {key: Counter() for key in LABELS}
    for anonymous_id, row in selected.items():
        if len(independent_coverage) != 360:
            # Partial inventory is progress only. No PROGRAM labels/oracles may
            # be compared or disclosed before every blind judgment is sealed.
            continue
        program = row["program_labels"]
        for error in _constraints(program):
            findings.append(
                _event("PROGRAM_CONTRACT_VIOLATION", anonymous_id, error, {"PROGRAM": program})
            )
        assigned = {name: values for name, values in labels.items() if anonymous_id in values}
        if not assigned:
            continue
        for field in LABELS:
            claims = {
                "PROGRAM": program[field],
                **{name: values[anonymous_id][field] for name, values in assigned.items()},
            }
            for name in assigned:
                agreements[field]["comparisons"] += 1
                agreements[field]["agreements"] += int(claims[name] == claims["PROGRAM"])
                agreements[field]["program_unknown"] += int(claims["PROGRAM"] is None)
                agreements[field]["review_unknown"] += int(claims[name] is None)
            if any(value != claims["PROGRAM"] for value in claims.values()):
                findings.append(
                    _event(
                        "LABEL_DISAGREEMENT",
                        anonymous_id,
                        field,
                        claims,
                        suspected_false_pass=claims["PROGRAM"] == 1
                        and any(v == 0 for k, v in claims.items() if k != "PROGRAM"),
                    )
                )
        task = tasks[row["qid"]]
        native_oracle = _oracle(
            dict(
                answer=task["gold_answer"],
                selected=task["gold_selected"],
                bindings=task["gold_bindings"],
            )
        )
        claims = {
            "PROGRAM": native_oracle,
            **{
                name: values[anonymous_id]
                for name, values in oracles.items()
                if anonymous_id in values
            },
        }
        if any(value != native_oracle for value in claims.values()):
            findings.append(_event("ORACLE_DISAGREEMENT", anonymous_id, "oracle", claims))
    findings.sort(key=lambda event: event["finding_id"])
    _check_adjudications(
        reader, adjudication_refs, review_refs, package_ref, package, findings, latest, traces
    )
    decisions, adjudicators = _adjudicate(
        reader, adjudication_refs, package_ref, findings, reviewers, latest
    )
    unresolved, blockers = [], []
    for event in findings:
        decision = decisions.get(event["finding_id"])
        event["adjudication"] = decision
        if event["kind"] in ("PROGRAM_CONTRACT_VIOLATION", "INDEPENDENCE_NOT_ESTABLISHED") or (
            decision and decision["decision"] in ("PRODUCTION_INCORRECT", "REVIEW_PROCESS_INVALID")
        ):
            blockers.append(event["finding_id"])
        elif decision is None or decision["decision"] != "PRODUCTION_CORRECT":
            unresolved.append(event["finding_id"])
    workflow = (
        "REVIEW_NOT_RUN"
        if not reviewers
        else "BLOCKING_FINDINGS"
        if blockers
        else "ADJUDICATION_REQUIRED"
        if unresolved
        else "REVIEW_INCOMPLETE"
        if len(independent_coverage) != 360
        else "REVIEW_COMPLETE_NOT_GATE_SIGNED"
    )
    synthetic = (
        reader.synthetic
        or package["execution_origin"] == "SYNTHETIC_TEST"
        or any(r["execution_origin"] == "SYNTHETIC_TEST" for r in (*submissions, *adjudicators))
    )
    report = dict(
        schema=VERSION,
        scope=dict(phase=sampling.PHASE, split=sampling.SPLIT),
        consumer=verification["consumer"],
        execution_origin="SYNTHETIC_TEST" if synthetic else "ACTUAL_REVIEW",
        synthetic=synthetic,
        completed=(not synthetic and workflow == "REVIEW_COMPLETE_NOT_GATE_SIGNED"),
        generation_allowed=False,
        evaluation_allowed=False,
        status="SYNTHETIC_ONLY" if synthetic else workflow,
        workflow_state=workflow,
        training_allowed=False,
        gate_status="NOT_SIGNED",
        reviewer_type="PROGRAM",
        input_producer=verification["input_producer"],
        inputs=[reader.references[key] for key in sorted(reader.references)],
        named_inputs={
            "package": package_ref,
            "reviews": list(review_refs),
            "adjudications": list(adjudication_refs),
        },
        selection_provenance=review_package.SELECTION,
        selection_verification_required_before_admission=[],
        reviewed_original_answers=len(all_coverage),
        independently_reviewed_original_answers=len(independent_coverage),
        missing_independent_review_ids=sorted(set(public) - independent_coverage),
        reviewers=submissions,
        adjudicators=adjudicators,
        independence_status="ATTESTATIONS_AND_BOUND_TRACE_ONLY_NOT_AUTOMATIC_SCIENTIFIC_PROOF",
        reviewer_identity_status="DECLARED_WITH_BOUND_EVIDENCE_NOT_AUTHENTICATED_BY_HASH",
        model_review_gate_authority="NOT_SPECIFIED_IN_ORIGINAL_CONTRACT",
        required_coverage={
            "unique_answers": 360,
            "per_parent_factor": 60,
            "per_family_in_cell": 10,
            "root_cap_in_cell": 4,
        },
        agreement_threshold=None,
        agreement_threshold_status="NOT_SPECIFIED_DO_NOT_INVENT",
        census_observability_threshold={
            "value": 0.90,
            "applies_to": "FULL_DEV_SCHEMA_CONFIRM_CENSUS_ONLY",
            "applied_to_review_sample": False,
        },
        agreement={
            key: dict(
                counts,
                rate=counts["agreements"] / counts["comparisons"]
                if counts["comparisons"]
                else None,
            )
            for key, counts in agreements.items()
        },
        findings=findings,
        blocking_finding_ids=blockers,
        adjudication_required_ids=unresolved,
        sampling_coverage=ledger["blocks"] if len(independent_coverage) == 360 else None,
        rare_cases=ledger["rare_cases"] if len(independent_coverage) == 360 else None,
        program_comparison_status=(
            "FULL_INDEPENDENT_COVERAGE"
            if len(independent_coverage) == 360
            else "WITHHELD_UNTIL_ALL_360_JUDGMENTS"
        ),
        interpretation=(
            "Coverage review only; no population error-rate claim or scientific authorization"
        ),
    )
    allowed_base = {ref["path"] for ref in verification["inputs"]}
    _require(
        all(path in allowed_base or path.startswith("REVIEW/") for path in reader.references),
        "Review artifacts must stay inside the complete REVIEW closure",
    )
    _require(_membership(root) == before, "Review evidence changed during verification")
    report.update(
        package_ref=package_ref,
        support_binding_sha256=package["ledger_source_identity"]["support_binding_sha256"],
        external_inputs=verification["external_inputs"],
        actual_traces=traces,
        execution_inventory=dict(
            executions=len(traces), instances=len({r["instance_id"] for r in traces})
        ),
        blockers=dict(
            blocking_finding_ids=blockers,
            unresolved_finding_ids=unresolved,
            missing_independent_review_ids=report["missing_independent_review_ids"],
            synthetic=synthetic,
        ),
    )
    return report


def _lock(root):
    parent = _directory(root, "REVIEW")
    _require(parent.is_dir(), "Fixed review package must exist first")
    fd = os.open(parent / ".audit.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _write_once(path, value):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(_payload(value))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def prepare_execution(root, *, task_id, instance_id, reviewer_id, kind="REVIEW"):
    """Reserve the real task before extraction/adjudication; never reuse an ID."""
    verification = review_package.verify_package(root)
    root = source._root(root)
    _require(kind in ("REVIEW", "ADJUDICATION"), "Unknown review execution kind")
    _require(
        isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", task_id),
        "Unsafe task ID",
    )
    for value in (instance_id, reviewer_id):
        _require(isinstance(value, str) and value.strip(), "Actual task/reviewer identity required")
    fd = _lock(root)
    try:
        _require(
            not (root / DESTINATION).exists()
            and not (root / "REVIEW/.audit-attempt.json").exists(),
            "Terminal audit already started; additional executions forbidden",
        )
        parent = _directory(root, REVIEW_DIRECTORY)
        parent.mkdir(mode=0o700, exist_ok=True)
        prior = aggregate_review(root) if kind == "ADJUDICATION" else None
        if prior is not None:
            _require(
                prior["independently_reviewed_original_answers"] == 360,
                "All 360 independent judgments must finish before program disclosure",
            )
        directory = parent / task_id
        directory.mkdir(mode=0o700)  # Preserve collisions/partials without retry.
        _fsync_directory(parent)
        attempt = dict(
            schema="srf6-followup-review-attempt-v1",
            execution_origin="ACTUAL_REVIEW",
            consumer=verification["consumer"],
            input_producer=verification["input_producer"],
            package=verification["package_ref"],
            method=verification["artifacts"]["package"][
                "adjudication_method" if kind == "ADJUDICATION" else "review_method"
            ],
            method_attachments=_method_refs(
                verification["artifacts"]["package"], adjudication=kind == "ADJUDICATION"
            ),
            task_id=task_id,
            instance_id=instance_id,
            reviewer_id=reviewer_id,
            kind=kind,
            started_at=now(),
        )
        if prior is not None:
            findings = [
                {k: v for k, v in row.items() if k != "adjudication"} for row in prior["findings"]
            ]
            _require(bool(findings), "No original findings need adjudication")
            packet = dict(
                schema="srf6-followup-review-findings-v1",
                package=verification["package_ref"],
                reviews=prior["named_inputs"]["reviews"],
                findings=findings,
            )
            _write_once(directory / "FINDINGS.json", packet)
            finding_ref = review_package._ref(
                f"{REVIEW_DIRECTORY}/{task_id}/FINDINGS.json", _payload(packet)
            )
            attempt["findings"] = finding_ref
            program_packet = _program_packet(
                verification["artifacts"]["package"],
                verification["artifacts"]["program_claims"],
                findings,
                verification["package_ref"],
            )
            _write_once(directory / "PROGRAM_CLAIMS.json", program_packet)
            program_ref = review_package._ref(
                f"{REVIEW_DIRECTORY}/{task_id}/PROGRAM_CLAIMS.json", _payload(program_packet)
            )
            attempt["program_claims"] = program_ref
            attempt["input_references"] = _adjudication_inputs(
                verification["artifacts"]["package"],
                prior["named_inputs"]["reviews"],
                _Facts(root),
                finding_ref,
                program_ref,
            )
        _write_once(directory / "ATTEMPT.json", attempt)
        return attempt
    finally:
        os.close(fd)


def _audit_files(report):
    payload = _payload(report)
    reference = review_package._ref(f"{DESTINATION}/SUMMARY.json", payload)
    binding = dict(
        schema="srf6-followup-review-seal-v1",
        summary=reference,
        consumer=report["consumer"],
        input_producer=report["input_producer"],
        package=report["package_ref"],
        status=report["status"],
        completed=report["completed"],
        training_allowed=False,
        gate_status="NOT_SIGNED",
    )
    return {"SUMMARY.json": payload, "BINDING.json": _payload(binding)}, reference


def seal(root):
    """One terminal assessment; a negative or partial result is never overwritten."""
    root = source._root(root)
    fd = _lock(root)
    try:
        _require(not (root / "REVIEW/.audit-failure.json").exists(), "Failed audit preserved")
        if (root / DESTINATION).exists():
            return verify_completed(root)
        attempt_path = root / "REVIEW/.audit-attempt.json"
        _write_once(
            attempt_path, dict(schema="srf6-followup-review-seal-attempt-v1", started_at=now())
        )
        try:
            report = aggregate_review(root)
            files, _ = _audit_files(report)
            review_package._store(root, DESTINATION, files)
        except Exception as exc:
            _write_once(
                root / "REVIEW/.audit-failure.json",
                dict(
                    status="TECHNICAL_UNKNOWN",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    training_allowed=False,
                    gate_status="NOT_SIGNED",
                ),
            )
            raise
        return verify_completed(root)
    finally:
        os.close(fd)


def _review_state(root):
    directory = _directory(root, "REVIEW")
    result = []
    for path in [directory, *sorted(directory.rglob("*"))]:
        _require(not path.is_symlink(), "Symlinked review evidence forbidden")
        stat = path.stat()
        _require(path.is_file() or path.is_dir(), "Unsupported review evidence object")
        result.append(
            (
                str(path.relative_to(root)),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_mode,
            )
        )
    return result


def verify_completed(root):
    """Recompute actual traces and compare the immutable terminal evidence."""
    root = source._root(root)
    cached = _CACHE.get((os.getpid(), str(root)))
    before = _review_state(root)
    if cached is not None:
        _require(before == cached[0], "Review closure changed after complete verification")
        context = source.require_support(root, purpose="review")
        _require(
            context["consumer"] == cached[1]["consumer"]
            and context["support_binding_ref"]["sha256"] == cached[1]["support_binding_sha256"],
            "Review upstream binding changed",
        )
        _require(_review_state(root) == before, "Review changed during warm verification")
        return copy.deepcopy(cached[1])
    _require(not (root / "REVIEW/.audit-failure.json").exists(), "Failed audit preserved")
    attempt_path = safe_evidence(root, "REVIEW/.audit-attempt.json")
    attempt_raw = attempt_path.read_bytes()
    attempt = rules._json(attempt_raw)
    _require(
        attempt.get("schema") == "srf6-followup-review-seal-attempt-v1", "Audit attempt required"
    )
    _time(attempt["started_at"])
    report = aggregate_review(root)
    files, summary_ref = _audit_files(report)
    review_package._verify_publication(root, DESTINATION, files)
    seal_ref = review_package._ref(f"{DESTINATION}/BINDING.json", files["BINDING.json"])
    result = dict(
        report,
        summary_ref=summary_ref,
        audit_ref=seal_ref,
        inputs=[
            *report["inputs"],
            summary_ref,
            seal_ref,
            review_package._ref("REVIEW/.audit-attempt.json", attempt_raw),
        ],
    )
    _require(_review_state(root) == before, "Review closure changed during cold verification")
    _CACHE[(os.getpid(), str(root))] = (before, copy.deepcopy(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("assess", "seal", "verify", "prepare"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--instance-id")
    parser.add_argument("--reviewer-id")
    parser.add_argument("--kind", choices=("REVIEW", "ADJUDICATION"), default="REVIEW")
    args = parser.parse_args()
    if args.operation == "prepare":
        result = prepare_execution(
            args.root,
            task_id=args.task_id,
            instance_id=args.instance_id,
            reviewer_id=args.reviewer_id,
            kind=args.kind,
        )
    else:
        result = {"assess": aggregate_review, "seal": seal, "verify": verify_completed}[
            args.operation
        ](args.root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
