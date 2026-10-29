"""Prospective, case-free instructions for an independently executed review.

This module publishes a contract, not a parser, scorer, oracle or review engine.
Its source anchors refer to the frozen answer producer. No answer or prior
review is read when deriving these deterministic payloads.
"""

from __future__ import annotations

import hashlib
import json
from textwrap import dedent

VERSION = "srf6-review-repair-contract-v1"
PRODUCER_REVISION = "6e5d36351b8c1f6f18c61661246191772ba6a6dd"

_SOURCES = (
    (
        "src/sr_f6/confirm_census.py",
        "e10fedcc3b7e370cc6b28e94addb7e8e25f8aae5b29cf427815f79fe565e7b0c",
        "18;76-79;152-169: confirmation reuses annotate_group and preserves raw stop",
    ),
    (
        "src/sr_f6/census.py",
        "2984457cbed82eb2312a13a12a9cba734bbe1f967112a27ead17a7d03f00a7a8",
        "188-194: finished is raw finish_reason == balanced",
    ),
    (
        "src/sr_f6/measurement.py",
        "90c26410af3f5526bce5b697138c72e0bf2d0cb58f3557d67fe42cf35db29a04",
        "23-101: whole JSON and V;135-229: A/J/finish;232-263: field-local S/B",
    ),
    (
        "src/srf6_ref/core.py",
        "9b0086b5cd4eea8ddd88669e0c2223c43b7b3a0aeb5c53cf27912caa887cf8bd",
        "18-34: exact numeric grammar/bounds;47-98: task arithmetic",
    ),
    (
        "src/sr_f5/measurement.py",
        "1b8e8cb7749f213b3b1e993f7e881ba4016ab6a76e520de750cd6b76ad0a6adb",
        "97-101;120-155: native projection and joint J",
    ),
    (
        "src/srf5_ref/measurement.py",
        "487c572e446c5ac48991477469cf6fef004ecdfc5c2b73eff67ada819f2e9621",
        "39-76: projected F5 format and 64000-character bound",
    ),
    (
        "src/sr_f4/task_contract.py",
        "a98e73540165cc7c0daeb0f96375341184fd345d0010a052213dec17bbf1be72",
        "51-55;126-136;151-200: interval/Beta-ranking native evidence extensions",
    ),
    (
        "src/sr_f3_ce/contract.py",
        "ef8862f340da0d932d878cddc76f398f25b53471218746d3120970ad4dbcce75",
        "21-57: native numbers;212-272: sufficiency;293-312: facts;354-460: native J",
    ),
    (
        "src/sr_f6/review_audit.py",
        "5732c1d3e6cb0f8daa6c7f30bca110e919b48c6e98485560cb38afc1bb56d27a",
        "140-187;353-589: review labels, extraction, UNKNOWN and adjudication",
    ),
)


def _policy():
    """Return fresh machine-readable boundaries; these do not compute labels."""
    return {
        "schema": VERSION,
        "producer_revision": PRODUCER_REVISION,
        "scope": {
            "phase": "SCHEMA_CONFIRM_V3",
            "split": "DEV_SCHEMA_CONFIRM",
            "existing_answer_count": 9216,
            "review_count": 360,
            "selection_seed": 2026102701,
            "new_answers": 0,
            "optimizer_updates": 0,
            "scientific_gate_authority": "NOT_GRANTED_BY_THIS_CONTRACT",
        },
        "completion": {
            "raw_metadata_field": "generation_finish_reason",
            "scored_finish_one_iff": "raw reason equals balanced",
            "raw_eos_is_scored_complete": False,
            "preserve_raw_metadata": True,
        },
        "numbers": {
            "fields": ["evidence[*].value", "bindings[*].operand", "answer"],
            "answer_null_valid_for_V_only": True,
            "boolean_is_number": False,
            "reduced_numerator_abs_max": 10**12,
            "reduced_denominator_max": 10**9,
            "bounds_inclusive": True,
            "numeric_string_stripped_max_characters": 64,
            "numeric_string_grammar": (
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,2})?"
                r"|[+-]?\d+\s*/\s*[+-]?\d+"
            ),
            "numeric_string_conversion": "Fraction(stripped_string.replace(' ', ''))",
            "json_decimal_adjusted_exponent_abs_max": 24,
            "whole_response_max_characters": 100_000,
            "tolerance": 0,
            "native_J_additional_limits": {
                "projected_response_max_characters": 64_000,
                "evidence_entries_max": 32,
                "numeric_string_unstripped_max_characters": 64,
                "fact_values": "integers within world.domain",
                "legacy_presentation_is_additional_J_gate": False,
            },
        },
        "labels": {
            "binary": ["A", "J", "V", "finish"],
            "nullable": ["S", "B"],
            "UNKNOWN": "null with exactly its field-specific reason",
            "finish_zero_semantics": "S=null and B=null; A/V remain independently checked",
            "invalid_whole_object": "A=V=J=0; S=B=null; finish uses raw metadata",
            "V_zero_does_not_erase_readable_fields": True,
            "J_checks_selected_or_binding_truth": False,
            "S_B_label_normalization": "none; exact strings and order-independent sets",
        },
        "exposure": {
            "extraction": "registered method plus anonymous raw text/protocol/raw stop only",
            "judgment": "sealed extraction plus same raw text and original world/query",
            "program_claims": "adjudication only, after all 360 judgments are sealed",
            "prior_review_material": "never supplied to new blind reviewers",
            "production_parser_or_scorer_execution_by_reviewer": "forbidden",
            "implementation_auditor_as_blind_reviewer": "forbidden",
            "adjudicator_instance": "different from every original review instance",
        },
        "adjudication": {
            "program_labels_required_for_every_finding_row": ["A", "J", "V", "finish", "S", "B"],
            "program_oracle_required": ["answer", "selected", "bindings"],
            "preserve_original_claims": True,
            "no_inferred_program_claims": True,
            "decisions": [
                "PRODUCTION_CORRECT",
                "PRODUCTION_INCORRECT",
                "UNRESOLVED",
                "REVIEW_PROCESS_INVALID",
            ],
            "agreement_threshold": None,
        },
        "source_anchors": [
            {"path": path, "sha256": sha256, "anchors": anchors}
            for path, sha256, anchors in _SOURCES
        ],
        "limitations": [
            "Instructions and synthetic tests do not constitute an actual independent review.",
            "Hashes bind bytes, not hidden context, authentic identity or actual independence.",
            "A MODEL_TEXT review is not HUMAN review and cannot itself sign a scientific gate.",
            "Unresolved semantics are recorded; no alternate parser, repair or favorable default.",
        ],
    }


_REVIEW_METHOD = r"""
# Frozen MODEL_TEXT review method

Version: srf6-review-repair-contract-v1.
Answer producer: 6e5d36351b8c1f6f18c61661246191772ba6a6dd.
Scope: the registered 360-answer sample of the same 9,216 SCHEMA_CONFIRM_V3 /
DEV_SCHEMA_CONFIRM answers. No resampling, text repair, scoring change, answer
generation, training or scientific-gate permission is supplied by this method.
CONTRACT.json records exact source anchors and byte identities. Read and follow
this entire method before the first extraction; retain its registered hash.

## Independence and exposure order

The actual reviewer is MODEL_TEXT, never HUMAN. Record the actual task and
instance, actual exposed model identity (NOT_EXPOSED if the platform does not
expose it), method reference, input/output contents, hashes and stage times.
Declare all five independence attestations truthfully. A reviewer must be a
fresh independent instance without answer-production, implementation-audit or
prior-review context. It must not read any earlier review, private selection
ledger, stored gold, program label, another reviewer's answer, result summary or
adjudication. It must not import, call, copy or execute the production parser,
scorer or oracle to produce its judgments. Mechanical hash/span validation and
serialization may check a submitted review; they must not supply its labels.
Independently read each original answer and compute each task result.

Stage 1, BLIND_EXTRACTION: see only this registered method/contract and the
bound anonymous raw text, protocol version and original generation_finish_reason.
Do not receive world/query yet. For each of evidence, selected, bindings,
aggregate and answer, report READABLE, MISSING or UNREADABLE. READABLE means the
value satisfies that field's structural/numeric grammar below, not that it is
true. Keep its exact original value span: zero-based Unicode character offsets,
end exclusive, and literal raw_text[start:end]. MISSING has no span. An invalid
present value may retain a literal span but remains UNREADABLE. Give the whole
object status and rationale. Do not infer a field from work or a nearby fragment.
Seal the complete extraction before receiving the Stage 2 task packet.

Stage 2, INDEPENDENT_JUDGMENT: see the same blind text, sealed extraction and
bound anonymous world/query only, in addition to the registered method. Never
alter the sealed extraction. Independently derive the oracle answer, selected
set and category-to-operand map from world/query; record exact numeric strings.
Compute all six labels A,J,V,finish,S,B under the rules below, with rationale.
Only S and B can be null. unknown_reasons must have exactly the null fields as
keys, each with its own nonempty reason. Record every unresolved ambiguity in
ambiguities. If a sealed extraction is wrong, record that conflict for independent
adjudication; do not silently rewrite it or pretend the stages were independent.
Seal all 360 judgments before any program-label comparison or adjudication.

## Whole-object boundary and exact numbers

The entire text must be one JSON object, at most 100,000 characters. Leading
and trailing whitespace are permitted; fences, prose, a second object, a
truncated object or a non-object top level are not. Duplicate JSON keys at ANY
nesting depth and nonfinite JSON constants invalidate the whole object.
Do not choose the last duplicate, salvage a complete prefix, repair JSON, use
work as the answer, or select a fragment using the task. When the whole object
is invalid/ambiguous, A=V=J=0 and S=B=null; selected/bindings cannot be READABLE.
Its finish label still follows raw stop metadata, independently of JSON validity.

Use exact rationals, never float tolerance or rounded decimals. The same exact
number rules apply to EVERY evidence.value, EVERY bindings.operand and answer;
they are not an answer-only check. The exception is answer:null: allowed by V's
syntax but not an exact answer (A=0 for this nonempty-answer task bank).
Booleans, null operands/facts, arrays, objects, nonfinite values, expressions,
units and unparsable strings are not numbers. Integers and JSON decimal/scientific
number literals are exact; JSON decimals additionally require the absolute
Decimal adjusted exponent (position of the most significant digit) <=24,
including signed/zero decimals. This is distinct from the written exponent.

For quoted numbers, strip outer whitespace first; the resulting length must be
<=64 characters, and the entire string must match the frozen Python fullmatch:
`[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,2})?|[+-]?\d+\s*/\s*[+-]?\d+`.
Convert that stripped string to an exact rational after removing ASCII spaces;
conversion must succeed and the denominator must be nonzero. Matching the regex
alone is insufficient. Do not widen grammar or normalize minus signs/case/units.
The exponent in a numeric STRING has at most two digits. For ALL accepted exact
values, after fraction reduction require abs(numerator)<=10^12 and positive
denominator<=10^9, inclusively, regardless of integer/decimal/fraction spelling.
Trailing decimal zeros do not themselves invalidate a value; use the reduced
rational and the applicable syntax/decimal bounds. Bounds apply to numeric fields,
not digits inside free work or identifier strings. A numerical defect in one
field does not invalidate the JSON object or erase other readable fields.

## The six labels

finish: 1 if and only if original generation_finish_reason is exactly balanced;
0 for eos or length. Preserve the original raw reason. The actual confirmation
entrypoint passes finished=(raw_reason == balanced); standalone scorer defaults
are NOT this execution contract. A complete-looking EOS text is still finish=0.
Do not infer completion from readable JSON or generation_status=COMPLETE. A raw
balanced stop can have finish=1 even when its JSON is malformed. Never relabel
raw EOS as balanced or replace raw metadata with scorer-normalized length.

A: 1 exactly when the sole whole JSON object's answer is a valid bounded exact
number equal to the independently computed task answer; otherwise 0. A does
not require V, finish, evidence truth or selected/bindings/aggregate correctness.
Missing/invalid/extra other fields do not erase a readable exact answer.

V: 1 exactly when the whole object has all five required fields evidence,
selected, bindings, aggregate, answer; only optional work may be additional.
evidence is a list of objects with exactly series, category, value; the first
two are strings and value passes the exact-number rules. selected is a list
of strings. bindings is a list of objects with exactly category and operand;
category is a string and operand passes the exact-number rules. aggregate is
exactly the string sum or mean. answer is null or passes the exact-number rules.
Optional work is a string. Otherwise V=0. V imposes NO truth, cardinality,
nonempty, category membership, duplicate-list-entry or task-aggregate condition.
The suggested field order is not enforced. Evidence-cell duplicates and wrong
identifiers can satisfy this V schema but fail native evidence for J.

S: if finish=0 or the whole object is invalid/ambiguous, null. Otherwise inspect
only selected: missing, non-list or any non-string member means null. For a
readable list, S=1 exactly when it has no duplicate strings and its set equals
the independently derived final selected set; otherwise 0. Order is irrelevant.
String spelling is exact: no trimming, case-folding, aliases or corrections.
Empty, extra, missing, wrong or duplicate category declarations are readable
semantic failures in this task bank, not UNKNOWN. Invalid unrelated fields/V=0
do not erase S. Do not use evidence or bindings to repair selected.

B: if finish=0 or the whole object is invalid/ambiguous, null. Otherwise inspect
only bindings: missing/non-list, malformed entry, missing/extra entry key,
non-string category, or any nonnumeric/out-of-bounds operand means null for B.
For a readable list, B=1 exactly when category keys are unique, their set equals
the TRUE final selected set, and every exact operand equals the requested Alpha,
Beta or signed Alpha-minus-Beta value for that category; otherwise 0. Compare
against the true set, not the submitted selected list. Empty, incomplete, extra,
duplicate or incorrect but readable mappings are failures. One unreadable
operand makes B UNKNOWN even if another entry has a readable semantic error.
Invalid unrelated fields/V=0 do not erase B. S and B need not agree.

J: the frozen native F5 joint reward intersected with F6 V. It requires V=1,
finish=1, the projected F5 format, the exact native answer and a truthful
sufficient native evidence certificate. It does NOT test semantic truth of selected or
bindings, nor whether the declared aggregate matches the query. Removing only
bindings/aggregate, then selected, produces the native view without rewriting
remaining numeric/work text. Independently judge that view; do not invoke it
as a production scoring engine. J=0 does not mean A=0.

The projected F5 format retains its 64,000-character text bound. The frozen F5
joint reward is F5_V AND native_finish AND native_value_correct AND certificate;
the legacy presentation diagnostic is NOT an additional gate. In particular,
do not import its separate 16,000-character work restriction into F6 V or J.
The native fact list has at most 32 entries, each exactly series/category/value;
identifiers must name actual world cells and each (series,category) may occur
only once, including identical repeats. Every fact value is an exact INTEGER
within world.domain and equals that original cell; any false extra fact fails
truth. A missing, invalid or duplicate fact certificate cannot earn J. Native
numeric strings additionally have <=64 characters BEFORE stripping whitespace
and must pass the native ASCII-digit numeric grammar: a signed decimal (optional
1-to-3-digit exponent), or signed-integer numerator/denominator with whitespace
around the slash; denominator nonzero. Native strings replace the Unicode minus
with ASCII minus, but this cannot rescue a value already rejected by F6 V.
Native exact rationals obey the same reduced numerator/denominator bounds;
native Decimal/nonzero decimal strings have adjusted-exponent absolute bound 30.
These are additional native-J checks, not reasons to redefine F6 A/V/S/B.

The sufficient certificate is the registered template, not an intuitive claim
that the answer happens to be right. For TOPK, evidence must contain the query's
ranking-series cell for EVERY category. Rank those reported cells in the stated
direction, resolving ties by earlier original category position; then require
the requested operand cells for every category selected by that ranking. A diff
operand needs BOTH original Alpha and Beta cells; a computed difference in
evidence cannot replace them. For INTERVAL, apply BOTH endpoint inclusion flags.
Without positive_only, require operand cells for every remaining interval
category. With positive_only, require BOTH Alpha and Beta cells for EVERY
pre-filter interval category, including subsequently excluded ones; select
strictly Alpha>Beta and then require the requested operands. The fact union
contains each cell once. Values appearing only in work or bindings do not
satisfy evidence. True extra cells are allowed. Check truth and sufficiency
separately; do not infer sufficiency from the submitted selected list.

## Independent oracle and unresolved conditions

Derive selection from the supplied task, not the answer text. TOPK uses the
specified selection_series, direction and k, breaking ties by earlier original
category position. INTERVAL uses printed zero-based positions and both endpoint
flags; positive_only is strictly Alpha>Beta. Derive each requested Alpha, Beta
or signed Alpha-minus-Beta operand and use the query's sum or arithmetic mean.
Return the final set and exact per-category operands as well as the exact answer.
This registered task bank has nonempty true final selections. If a packet violates
that invariant, has inconsistent source metadata, or needs semantics not resolved
by this contract, preserve the item and state the ambiguity; do not invent a
new task, favorable label, extra sampling or altered scoring rule.

The structural implications J=1 => A=V=finish=1 and finish=0 => S=B=null are
consistency checks, never substitutes for reading and independent judgment.
UNKNOWN is never filled with 0/1 or removed from coverage. There is no agreement
percentage threshold, and the full-census observability threshold is not a
review-sample gate. The review's provenance is an attested execution record;
hashes alone cannot authenticate hidden context or actual independence.
"""

_ADJUDICATION_METHOD = """
# Independent adjudication method

Version: srf6-review-repair-contract-v1.
Use the exact registered REVIEW_METHOD.md and CONTRACT.json for producer
6e5d36351b8c1f6f18c61661246191772ba6a6dd. This is adjudication after the complete
360 independent judgments are sealed; it is never a blind extraction stage.
Use a new actual MODEL_TEXT or HUMAN instance distinct from every original
reviewer, with truthful identity, method, source and full visible execution trace.
Do not invent a model identity, impersonate a human, or relabel the original
review instance as a new adjudicator. Do not run the production parser/scorer
to replace your own reading and decision.

The bound adjudication packet must contain original anonymous raw texts and raw
stop metadata, original world/query, sealed extractions and judgments, exact
review manifests, all original findings, and a separately source-verified PROGRAM
statement for EVERY finding-associated answer. Each PROGRAM statement must show
all six labels A,J,V,finish,S,B (including nulls) and its answer/selected/bindings
oracle, with source references. The statement must match the frozen private
selection ledger and original task identity exactly. Do not assume a production
label from a finding's presence, a reviewer's text, or absence of disagreement.
REVIEW_AMBIGUITY and REVIEW_CONTRACT_VIOLATION findings need the same full PROGRAM
statement as LABEL_DISAGREEMENT and ORACLE_DISAGREEMENT findings. No required
label may be omitted merely because the finding names another field.

Read every supplied claim and its rationale before deciding. For each exact
finding_id, state what the production claim says, what the reviewer says, the
applicable registered rule, the decisive raw span/task arithmetic or procedural
evidence, and why that supports the decision. Keep the original claims immutable.
Resolve the finding's actual question; do not replace it with an easier question
or dismiss a substantive oracle/false-PASS defect as a stylistic difference.

Use exactly one decision per finding: PRODUCTION_CORRECT when the supplied
production statement is correct under the frozen contract and the concern is
resolved; PRODUCTION_INCORRECT when it is not; UNRESOLVED when required evidence
or a definite interpretation is missing; REVIEW_PROCESS_INVALID for invalid
extraction/judgment procedure or independence. No majority vote, agreement-rate
threshold, default agreement with PROGRAM, or desired gate outcome is a rule.
If the program statement is absent/incomplete or its binding cannot be checked,
do not infer it and do not issue PRODUCTION_CORRECT; retain an unresolved or
process-invalid decision with the precise missing evidence. If a finding has no
answer ID (for example an independence finding), use its complete procedural
claims and traces; do not invent an answer-level PROGRAM claim.

Every decision requires a concrete nonempty rationale and references only to
evidence actually supplied in this adjudication's recorded visible inputs.
Record method/source/trace identities, start and completion times and the actual
decision response. Adjudication starts after every original judgment completes.
Record concerns about the method or packet explicitly; do not rewrite raw text,
earlier reviews, findings, or production labels. Do not alter scorer semantics,
generate answers, add samples, loosen a gate or choose a favorable subset.
Negative and unresolved decisions remain evidence. Completion of this record
does not authorize sampling, training, numerical replay, evaluation or scientific
gate signing. Hashes establish byte binding, not authentic reviewer independence.
"""


def _text(value):
    return (dedent(value).strip() + "\n").encode("utf-8")


def _json(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def payloads() -> dict[str, bytes]:
    """Return the exact case-free publication; no filesystem or model access."""
    methods = {
        "REVIEW_METHOD.md": _text(_REVIEW_METHOD),
        "ADJUDICATION_METHOD.md": _text(_ADJUDICATION_METHOD),
    }
    contract = _policy()
    contract["method_sha256"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in methods.items()
    }
    return {**methods, "CONTRACT.json": _json(contract)}


def specification() -> dict:
    """Freeze policy and all three byte hashes without a self-referential hash."""
    return {
        **_policy(),
        "payload_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in payloads().items()
        },
    }
