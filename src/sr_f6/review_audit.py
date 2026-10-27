"""Aggregate independently submitted reviews, never manufacture their judgments.

All inputs are explicit path/SHA256 references under the run root. The package
manifest binds context to the frozen CENSUS_REVIEW_V1 producer, ledger, blind,
task_source, judgment_tasks and ledger_source_identity. judgment_tasks contains
only anonymous_id/world/query, checked against frozen DEV_CAL TASKS_PRIVATE.

A review manifest binds package_sha256, reviewer, execution_origin,
independence, blind_input, blind_extraction and judgments. Reviewer metadata contains id,
type (HUMAN/MODEL_TEXT/MODEL_VISUAL/PROGRAM), identity, model_identity, method,
source and transcript references. Independence is an attestation backed by
those references, not a scientific proof. Each shard covers its bound blind
subset; the union of independent judgments must cover 360. Reviewers may submit
multiple shards. Overlapping judgments remain separate source-bound claims.

Blind extraction receives only the blind artifact, never tasks or program
labels; it records exact source spans for the five protocol fields. Judgment
receives the sealed extraction, blind and frozen world/query packet. It records
A/J/V/finish/S/B, UNKNOWN reasons, an independently computed oracle and reasons.
Extraction must precede judgment. Neither stage may see program labels. The
aggregator reads the private ledger only after both submitted stages are sealed.
READABLE means semantically checkable in that field's schema, not merely visible
characters. A locatable bindings literal with null/nonnumeric operands remains
UNREADABLE for B; other readable fields retain their own labels.
Each extraction also declares object_status: UNAMBIGUOUS_JSON_OBJECT or
INVALID_OR_AMBIGUOUS_JSON. Duplicate keys/invalid JSON cannot provide readable
semantic claims merely because selected/bindings fragments can be located.

Adjudications reference exact finding IDs and independent reviewer provenance.
They preserve original disagreements; confirmed false PASS/oracle errors and
gold-guided extraction block completion. No agreement percentage is a gate.
The 90% observability requirement belongs to the FULL census, not this sample.
Outputs never sign PASS or authorize training. MODEL_TEXT is a truthful reviewer
type, not an assertion that the contract lets a model sign the scientific gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from .common import digest, file_hash, safe_evidence

VERSION = "srf6-independent-review-audit-v1"
PHASE = "CENSUS_REVIEW_V1"
FIELDS = ("evidence", "selected", "bindings", "aggregate", "answer")
LABELS = ("A", "J", "V", "finish", "S", "B")
INDEPENDENCE = (
    "separate_judgment_from_production",
    "no_shared_production_parser",
    "no_shared_production_scorer",
    "no_gold_guided_extraction",
    "no_program_labels_before_judgment",
)
TYPES = {"HUMAN", "MODEL_TEXT", "MODEL_VISUAL", "PROGRAM"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "SHA256 required")
    return value


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), name + " must be explicit")
    return value


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "Duplicate JSON key")
        value[key] = item
    return value


def _json(raw):
    def nonfinite(_):
        raise ValueError("Nonfinite JSON")

    return json.loads(raw, object_pairs_hook=_unique, parse_constant=nonfinite)


class _ArtifactReader:
    def __init__(self, root):
        self.root = root
        self.references = {}


def _artifact(reader, ref, *, lines=False, opaque=False):
    _require(
        isinstance(ref, dict) and set(ref) == {"path", "sha256"},
        "Exact artifact reference required",
    )
    _sha(ref["sha256"])
    path = safe_evidence(reader.root, ref["path"])
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "Bound artifact bytes changed")
    previous = reader.references.setdefault(ref["path"], dict(ref))
    _require(previous == ref, "Same evidence path has conflicting identities")
    if opaque:
        return None
    return [_json(line) for line in raw.splitlines() if line.strip()] if lines else _json(raw)


def _index(items, expected=None):
    _require(isinstance(items, list), "Review items must be a list")
    result = {}
    for item in items:
        _require(isinstance(item, dict), "Review item must be an object")
        key = _text(item.get("anonymous_id"), "anonymous_id")
        _require(key not in result, "Duplicate anonymous review ID")
        result[key] = item
    if expected is not None:
        _require(
            set(result) == set(expected), "Reviewer must cover exactly its bound anonymous subset"
        )
    return result


def _labels(value):
    _require(isinstance(value, dict) and set(value) == set(LABELS), "All six labels required")
    for key, label in value.items():
        _require(
            (key in ("S", "B") and label is None) or (type(label) is int and label in (0, 1)),
            "Only S/B may use None; A/J/V/finish must remain binary",
        )
    return dict(value)


def _number(value):
    _require(
        isinstance(value, str) and 0 < len(value) <= 256, "Exact oracle numeric string required"
    )
    _require(
        re.fullmatch(r"[+-]?(?:\d+/\d+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?)", value),
        "Bounded exact oracle number required",
    )
    try:
        number = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid exact oracle number") from exc
    _require(
        number.numerator.bit_length() <= 4096 and number.denominator.bit_length() <= 4096,
        "Oracle number too large",
    )
    return str(number)


def _oracle(value):
    _require(
        isinstance(value, dict) and set(value) == {"answer", "selected", "bindings"},
        "Independent oracle fields required",
    )
    selected, bindings = value["selected"], value["bindings"]
    _require(
        isinstance(selected, list) and all(isinstance(s, str) and s for s in selected),
        "Oracle selected names required",
    )
    _require(len(set(selected)) == len(selected), "Oracle selection contains duplicates")
    _require(
        isinstance(bindings, dict) and set(bindings) == set(selected), "Oracle binding keys differ"
    )
    return dict(
        answer=_number(value["answer"]),
        selected=sorted(selected),
        bindings={k: _number(v) for k, v in sorted(bindings.items())},
    )


def _time(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("Explicit review timestamp required") from exc
    _require(parsed.tzinfo is not None, "Review timestamp requires timezone")
    return parsed


def _event(kind, anonymous_id, field, claims, **extra):
    value = dict(kind=kind, anonymous_id=anonymous_id, field=field, claims=claims, **extra)
    return dict(finding_id=digest(value), **value)


def _reviewer(root, value):
    _require(isinstance(value, dict), "Explicit reviewer identity required")
    identifier = _text(value.get("id"), "reviewer ID")
    _require(identifier != "PROGRAM", "PROGRAM is reserved for the production-label claim")
    kind = value.get("type")
    _require(kind in TYPES, "Reviewer type must be HUMAN/MODEL_TEXT/MODEL_VISUAL/PROGRAM")
    _text(value.get("identity"), "reviewer identity")
    if kind.startswith("MODEL_"):
        _text(value.get("model_identity"), "actual model identity")
    else:
        _require(
            value.get("model_identity") is None,
            "Human/program identity cannot disguise a model reviewer",
        )
    for key in ("method", "source", "transcript"):
        _artifact(root, value.get(key), opaque=True)
    return identifier


def _package(root, producer, ref):
    package = _artifact(root, ref)
    _require(
        package.get("schema") == "srf6-review-package-binding-v1", "Review package schema differs"
    )
    _require(
        package.get("context") == producer["identity"]
        and package.get("phase_identity_sha256") == producer["identity_sha256"],
        "Review package producer identity differs",
    )
    _require(
        package.get("execution_origin") in ("ACTUAL_CENSUS", "SYNTHETIC_TEST"),
        "Package execution origin required",
    )
    ledger = _artifact(root, package["ledger"])
    blind = _artifact(root, package["blind"], lines=True)
    _require(
        ledger.get("source_identity") == package.get("ledger_source_identity")
        and isinstance(ledger.get("source_identity"), dict)
        and bool(ledger["source_identity"]),
        "Ledger source identity binding differs",
    )
    for value in ledger["source_identity"].values():
        _sha(value)
    _require(
        ledger.get("version") == "srf6-blind-review-v1.1"
        and ledger.get("sampling_status") == "COMPLETE_360"
        and ledger.get("selected_count") == 360
        and ledger.get("required_count") == 360
        and ledger.get("shortfall") == 0
        and ledger.get("status") == "BLIND_PACKAGE_ONLY"
        and ledger.get("review_status") == "REVIEW_NOT_RUN"
        and ledger.get("seed") == 2026102701
        and ledger.get("root_cap") == 4
        and ledger.get("root_cap_scope") == "parent_factor",
        "Frozen 360 sampling contract differs",
    )
    _require(
        ledger.get("blind_items_sha256") == digest(blind),
        "Blind canonical content differs from ledger",
    )
    selected, public = _index(ledger["selected"]), _index(blind)
    _require(
        len(public) == 360 and set(selected) == set(public),
        "Exactly 360 matched original answers required",
    )
    from srf6_ref.core import FAMILIES

    expected = Counter(
        {
            (p, f, family): 10
            for p in ("b0", "b1", "b2")
            for f in ("P0", "P1")
            for family in FAMILIES
        }
    )
    _require(
        Counter((r["parent"], r["factor"], r["family"]) for r in selected.values()) == expected,
        "Family/cell review quotas differ",
    )
    _require(
        max(Counter((r["parent"], r["factor"], r["root_id"]) for r in selected.values()).values())
        <= 4,
        "Root review cap exceeded",
    )
    for key in ("row_id", "origin_id", "unit_id"):
        _require(
            len({r[key] for r in selected.values()}) == 360, "Duplicate original response identity"
        )
    _require(
        len(
            {
                (r["parent"], r["factor"], r["qid"], public[a]["raw_text"])
                for a, r in selected.items()
            }
        )
        == 360,
        "Duplicate cell/qid/text review unit",
    )
    for anonymous_id, row in selected.items():
        item = public[anonymous_id]
        _require(
            set(item)
            == {"anonymous_id", "raw_text", "protocol_version", "generation_finish_reason"},
            "Unexpected blind payload fields",
        )
        _require(
            isinstance(item["raw_text"], str)
            and hashlib.sha256(item["raw_text"].encode()).hexdigest() == row["raw_text_sha256"],
            "Original blind text differs",
        )
        _require(
            item["protocol_version"] == "structured_v4-srf6-1.0"
            and item["generation_finish_reason"] in ("balanced", "eos", "length"),
            "Blind protocol or raw finish metadata differs",
        )
        _require(
            item["generation_finish_reason"] == row["generation_finish_reason"]
            and row.get("generation_finish_reason_source") == "CENSUS_RAW_BY_ROW",
            "Raw sampler finish provenance differs",
        )
        _labels(row["program_labels"])
    manifest = _artifact(root, producer["identity"]["data_manifest"])
    members = {m["path"]: m["sha256"] for m in manifest["files"]}
    task_ref = package["task_source"]
    _require(
        task_ref["path"] == "DATA/DEV_CAL/TASKS_PRIVATE.jsonl"
        and members.get(task_ref["path"]) == task_ref["sha256"],
        "Judgment task source is not frozen DEV_CAL",
    )
    tasks = _artifact(root, task_ref, lines=True)
    task_map = {t["qid"]: t for t in tasks}
    _require(len(task_map) == len(tasks), "Duplicate frozen task qid")
    packet = _index(_artifact(root, package["judgment_tasks"], lines=True), public)
    for anonymous_id, item in packet.items():
        task = task_map[selected[anonymous_id]["qid"]]
        _require(
            task.get("split") == "DEV_CAL"
            and all(
                selected[anonymous_id][key] == task[key] for key in ("root_id", "family", "factor")
            ),
            "Ledger task lineage differs from frozen DEV_CAL",
        )
        _require(
            item == {"anonymous_id": anonymous_id, "world": task["world"], "query": task["query"]},
            "Judgment packet must contain only frozen world/query, no gold or program labels",
        )
    return package, ledger, public, selected, task_map


def _extraction(root, reference, reviewer_id, blind_input, public):
    value = _artifact(root, reference)
    _require(
        value.get("stage") == "BLIND_EXTRACTION" and value.get("reviewer_id") == reviewer_id,
        "Blind extraction stage/reviewer differs",
    )
    _require(
        value.get("inputs") == [blind_input], "Blind extraction may see only raw blind artifact"
    )
    exposure = value.get("exposure")
    _require(
        isinstance(exposure, dict)
        and set(exposure) == {"program_labels", "oracle", "private_ledger"}
        and all(v is False for v in exposure.values()),
        "Gold/program-guided extraction is forbidden",
    )
    items = _index(value["items"], public)
    for key, item in items.items():
        raw = public[key]["raw_text"]
        _require(
            item.get("raw_text_sha256") == hashlib.sha256(raw.encode()).hexdigest(),
            "Extractor reviewed different raw text",
        )
        fields = item.get("fields")
        _require(
            item.get("object_status") in ("UNAMBIGUOUS_JSON_OBJECT", "INVALID_OR_AMBIGUOUS_JSON"),
            "Explicit whole-object JSON ambiguity status required",
        )
        _require(
            isinstance(fields, dict) and set(fields) == set(FIELDS),
            "Every protocol field needs independent extraction",
        )
        _text(item.get("rationale"), "extraction rationale")
        for field in fields.values():
            _require(
                isinstance(field, dict)
                and field.get("state") in ("READABLE", "MISSING", "UNREADABLE"),
                "Explicit extraction readability required",
            )
            span = field.get("span")
            if field["state"] == "READABLE":
                _require(
                    isinstance(span, dict) and set(span) == {"start", "end", "text"},
                    "Readable claim needs a raw value span",
                )
            if span is not None:
                start, end = span.get("start"), span.get("end")
                _require(
                    type(start) is int
                    and type(end) is int
                    and 0 <= start < end <= len(raw)
                    and raw[start:end] == span.get("text"),
                    "Extracted span is not literal original text",
                )
            if field["state"] == "MISSING":
                _require(span is None, "Missing field cannot claim an observed span")
    return value, items


def _constraints(labels, fields=None, object_status="UNAMBIGUOUS_JSON_OBJECT"):
    errors = []
    if labels["J"] == 1 and any(labels[k] != 1 for k in ("A", "V", "finish")):
        errors.append("J_PASS_REQUIRES_A_V_FINISH")
    if labels["finish"] == 0 and any(labels[k] is not None for k in ("S", "B")):
        errors.append("INCOMPLETE_SEMANTIC_CLAIMS_MUST_BE_UNKNOWN")
    if object_status == "INVALID_OR_AMBIGUOUS_JSON":
        if labels["V"] != 0 or labels["A"] != 0:
            errors.append("INVALID_OR_AMBIGUOUS_JSON_REQUIRES_A_V_FAIL")
        if any(labels[k] is not None for k in ("S", "B")):
            errors.append("INVALID_OR_AMBIGUOUS_JSON_SEMANTICS_MUST_BE_UNKNOWN")
        if fields:
            for label, field in (("S", "selected"), ("B", "bindings")):
                if fields[field]["state"] == "READABLE":
                    errors.append(label + "_AMBIGUOUS_OBJECT_CANNOT_PROVIDE_READABLE_CLAIM")
        return errors
    if fields and labels["finish"] == 1:
        for label, field in (("S", "selected"), ("B", "bindings")):
            extracted = fields[field]
            if extracted["state"] != "READABLE" and labels[label] is not None:
                errors.append(label + "_UNREADABLE_CLAIM_NEEDS_UNKNOWN")
            if extracted["state"] == "READABLE" and labels[label] is None:
                errors.append(label + "_READABLE_CLAIM_CANNOT_BE_UNKNOWN")
            if (
                extracted["state"] == "READABLE"
                and extracted["span"]["text"].strip() == "[]"
                and labels[label] != 0
            ):
                errors.append(label + "_EMPTY_READABLE_CLAIM_IS_FAIL")
    return errors


def _submission(root, ref, package_ref, package, public):
    value = _artifact(root, ref)
    _require(
        value.get("schema") == "srf6-independent-review-v1"
        and value.get("package_sha256") == package_ref["sha256"],
        "Reviewer package binding differs",
    )
    reviewer_id = _reviewer(root, value["reviewer"])
    claim_id = reviewer_id + "@" + ref["sha256"]
    _require(
        value.get("execution_origin") in ("ACTUAL_REVIEW", "SYNTHETIC_TEST"),
        "Review execution origin required",
    )
    independence = value.get("independence")
    _require(
        isinstance(independence, dict) and set(independence) == set(INDEPENDENCE),
        "Explicit independent-method attestation required",
    )
    _require(
        all(type(v) is bool for v in independence.values()),
        "Independence attestations must be boolean",
    )
    subset = _index(_artifact(root, value["blind_input"], lines=True))
    _require(
        bool(subset) and set(subset) <= set(public),
        "Review shard has unknown or empty blind subset",
    )
    _require(
        all(item == public[key] for key, item in subset.items()),
        "Blind shard differs from frozen original text/protocol",
    )
    extracted, extractions = _extraction(
        root, value["blind_extraction"], reviewer_id, value["blind_input"], subset
    )
    judged = _artifact(root, value["judgments"])
    _require(
        judged.get("stage") == "INDEPENDENT_JUDGMENT" and judged.get("reviewer_id") == reviewer_id,
        "Judgment stage/reviewer differs",
    )
    _require(
        judged.get("inputs")
        == [value["blind_input"], package["judgment_tasks"], value["blind_extraction"]],
        "Judgment may only use sealed extraction and frozen task packet",
    )
    _require(
        judged.get("program_labels_seen") is False, "Independent judgment cannot see program labels"
    )
    _require(
        _time(extracted.get("completed_at"))
        < _time(judged.get("started_at"))
        <= _time(judged.get("completed_at")),
        "Blind extraction must be sealed before judgment",
    )
    items = _index(judged["items"], subset)
    findings = []
    for key, item in items.items():
        labels = _labels(item["labels"])
        unknown = item.get("unknown_reasons")
        _require(
            isinstance(unknown, dict)
            and set(unknown) == {k for k in ("S", "B") if labels[k] is None},
            "UNKNOWN requires exactly its field-specific reason",
        )
        for reason in unknown.values():
            _text(reason, "UNKNOWN reason")
        _text(item.get("rationale"), "independent judgment rationale")
        item["oracle"] = _oracle(item["oracle"])
        for error in _constraints(
            labels, extractions[key]["fields"], extractions[key]["object_status"]
        ):
            findings.append(_event("REVIEW_CONTRACT_VIOLATION", key, error, {claim_id: labels}))
        ambiguities = item.get("ambiguities", [])
        _require(
            isinstance(ambiguities, list)
            and all(isinstance(reason, str) and reason.strip() for reason in ambiguities),
            "Review ambiguities must be explicit text",
        )
        if ambiguities:
            findings.append(_event("REVIEW_AMBIGUITY", key, "judgment", {claim_id: ambiguities}))
    if not all(independence.values()):
        findings.append(
            _event("INDEPENDENCE_NOT_ESTABLISHED", None, "method", {claim_id: independence})
        )
    return value, items, findings


def _adjudicate(root, refs, package_ref, findings, reviewers, latest_judgment):
    by_id = {event["finding_id"]: event for event in findings}
    decisions, sources = {}, []
    for ref in refs:
        value = _artifact(root, ref)
        _require(
            value.get("schema") == "srf6-review-adjudication-v1"
            and value.get("package_sha256") == package_ref["sha256"],
            "Adjudication package differs",
        )
        reviewer_id = _reviewer(root, value["reviewer"])
        _require(
            reviewer_id not in reviewers, "Adjudicator must be independent of original reviewers"
        )
        _require(
            value.get("execution_origin") in ("ACTUAL_REVIEW", "SYNTHETIC_TEST"),
            "Adjudication execution origin required",
        )
        _require(
            _time(value["completed_at"]) > latest_judgment,
            "Adjudication must follow all independent judgments",
        )
        _require(
            value.get("independent_of_reviewers") is True,
            "Independent adjudication attestation required",
        )
        for decision in value["items"]:
            key = decision["finding_id"]
            _require(
                key in by_id and key not in decisions,
                "Unknown or duplicate adjudication finding ID",
            )
            _require(
                decision.get("decision")
                in (
                    "PRODUCTION_CORRECT",
                    "PRODUCTION_INCORRECT",
                    "UNRESOLVED",
                    "REVIEW_PROCESS_INVALID",
                ),
                "Explicit adjudication decision required",
            )
            _text(decision.get("rationale"), "adjudication rationale")
            evidence = decision.get("evidence")
            _require(
                isinstance(evidence, list) and bool(evidence), "Adjudication evidence required"
            )
            for reference in evidence:
                _artifact(root, reference, opaque=True)
            decisions[key] = dict(
                decision, reviewer_id=reviewer_id, execution_origin=value["execution_origin"]
            )
        sources.append(
            dict(
                reference=ref,
                reviewer=value["reviewer"],
                execution_origin=value["execution_origin"],
            )
        )
    return decisions, sources


def aggregate_review(
    root, *, phase_deployment, phase_identity_sha256, package_ref, review_refs, adjudication_refs=()
):
    """Verify frozen inputs and aggregate submitted independent work, read-only.

    verify_phase authenticates the INPUT PRODUCER only. This new validator is
    not part of that frozen deployment and never adopts its output context.
    Signatures/identity authenticity and the truth of independence attestations
    require external review; file hashes alone cannot establish either.
    """
    from .phases import verify_phase

    _sha(phase_identity_sha256)
    producer = verify_phase(
        root, PHASE, deployment=phase_deployment, expected_identity_sha256=phase_identity_sha256
    )
    _require(
        producer.get("phase") == PHASE
        and producer.get("identity_sha256") == phase_identity_sha256
        and producer.get("training_allowed") is False
        and producer.get("status") == "BOUND_NOT_AUTHORIZED",
        "Input producer binding is not its registered review phase",
    )
    reader = _ArtifactReader(root)
    phase_identity = _artifact(
        reader, {"path": f"PHASES/{PHASE}/IDENTITY.json", "sha256": phase_identity_sha256}
    )
    _require(phase_identity == producer["identity"], "Input producer identity bytes differ")
    package, ledger, public, selected, tasks = _package(reader, producer, package_ref)
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
    synthetic = package["execution_origin"] == "SYNTHETIC_TEST" or any(
        r["execution_origin"] == "SYNTHETIC_TEST" for r in (*submissions, *adjudicators)
    )
    return dict(
        schema=VERSION,
        status="SYNTHETIC_ONLY" if synthetic else workflow,
        workflow_state=workflow,
        training_allowed=False,
        gate_status="NOT_SIGNED",
        reviewer_type="PROGRAM",
        input_producer=producer,
        validator={
            "module": str(Path(__file__).resolve()),
            "source_sha256": file_hash(__file__),
            "phase": None,
            "identity_scope": "CURRENT_VALIDATOR_BYTES_NOT_INPUT_PRODUCER_PHASE",
        },
        inputs=[reader.references[key] for key in sorted(reader.references)],
        named_inputs={
            "package": package_ref,
            "reviews": list(review_refs),
            "adjudications": list(adjudication_refs),
        },
        selection_provenance="NOT_RECOMPUTED_FROM_FULL_CENSUS",
        selection_verification_required_before_admission=[
            "bind_selected_rows_to_complete_census_evidence",
            "replay_frozen_sampler_with_seed_2026102701",
        ],
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
            "applies_to": "FULL_DEV_CAL_CENSUS_ONLY",
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
        sampling_coverage=ledger["blocks"],
        rare_cases=ledger["rare_cases"],
        interpretation=(
            "Coverage review only; no population error-rate claim or scientific authorization"
        ),
    )
