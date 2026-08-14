"""Recursive leakage and split-isolation checks for recoverability data."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_FORBIDDEN_TOKENS = (
    "recoverable",
    "gold_answer",
    "gold_target",
    "gold_scene",
    "gold_reasoning",
    "compensation_category",
    "answer_source",
)
_ANSWER_CODED_ID = re.compile(r"(?:^|[_-])answer[_-]?-?\d+(?:$|[_-])", re.IGNORECASE)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def reject_forbidden_payload_content(payload: object) -> None:
    """Reject hidden labels recursively, including nested keys and string values."""

    def visit(value: object, *, key_hint: str | None = None) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise ValueError("payload keys must be strings")
                normalized = _normalized(key)
                if any(token in normalized for token in _FORBIDDEN_TOKENS):
                    raise ValueError(f"forbidden payload field: {key}")
                visit(nested, key_hint=normalized)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested, key_hint=key_hint)
            return
        if isinstance(value, str):
            normalized = _normalized(value)
            if any(token in normalized for token in _FORBIDDEN_TOKENS):
                raise ValueError("forbidden payload value")
            if key_hint == "scene_id" and _ANSWER_CODED_ID.search(value):
                raise ValueError("answer-coded scene identifier")
            return
        if value is None or type(value) in {bool, int, float}:
            return
        raise TypeError("payload contains an unsupported value type")

    visit(payload)


@dataclass(frozen=True, slots=True)
class CueAuditRecord:
    cue_signature: str
    question_signature: str
    answer: int
    count: int
    cue_only_correct_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.cue_signature, str) or not self.cue_signature:
            raise ValueError("cue_signature must be non-empty")
        if not isinstance(self.question_signature, str) or not self.question_signature:
            raise ValueError("question_signature must be non-empty")
        if type(self.answer) is not int:
            raise TypeError("answer must be an exact integer")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("count must be a positive integer")
        if (
            type(self.cue_only_correct_count) is not int
            or not 0 <= self.cue_only_correct_count <= self.count
        ):
            raise ValueError("cue_only_correct_count must lie between zero and count")


def build_cue_audit_record(
    *,
    public_cue: Mapping[str, object],
    question_signature: str,
    answer: int,
    count: int,
    cue_only_correct_count: int,
) -> CueAuditRecord:
    """Derive the grouping signature from the actual public cue, not a template label."""

    if not isinstance(public_cue, Mapping) or not public_cue:
        raise ValueError("public_cue must be a non-empty mapping")
    reject_forbidden_payload_content(public_cue)
    if not isinstance(question_signature, str) or not question_signature:
        raise ValueError("question_signature must be non-empty")
    try:
        signature = json.dumps(
            public_cue,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("public_cue must be canonical JSON data") from error
    return CueAuditRecord(
        cue_signature=signature,
        question_signature=question_signature,
        answer=answer,
        count=count,
        cue_only_correct_count=cue_only_correct_count,
    )


@dataclass(frozen=True, slots=True)
class CueGroupAudit:
    cue_signature: str
    question_signature: str
    distinct_answers: int
    total_records: int
    answer_imbalance: int
    cue_only_accuracy_upper: float


def assert_cue_group_nonleaking(
    records: tuple[CueAuditRecord, ...],
    *,
    min_distinct_answers: int,
    max_answer_imbalance: int,
    max_cue_only_accuracy_upper: float,
) -> CueGroupAudit:
    if not isinstance(records, tuple) or not records:
        raise ValueError("records must be a non-empty tuple")
    if type(min_distinct_answers) is not int or min_distinct_answers < 2:
        raise ValueError("min_distinct_answers must be at least two")
    if type(max_answer_imbalance) is not int or max_answer_imbalance < 0:
        raise ValueError("max_answer_imbalance must be non-negative")
    if (
        isinstance(max_cue_only_accuracy_upper, bool)
        or not isinstance(max_cue_only_accuracy_upper, (int, float))
        or not 0 < float(max_cue_only_accuracy_upper) < 1
    ):
        raise ValueError("max_cue_only_accuracy_upper must lie strictly between zero and one")
    if any(not isinstance(item, CueAuditRecord) for item in records):
        raise TypeError("records must contain CueAuditRecord instances")
    groups = {(item.cue_signature, item.question_signature) for item in records}
    if len(groups) != 1:
        raise ValueError("cue audit records must share the same full cue and question")
    counts: Counter[int] = Counter()
    for item in records:
        counts[item.answer] += item.count
    if len(counts) < min_distinct_answers:
        raise ValueError("cue group has too few distinct answers")
    imbalance = max(counts.values()) - min(counts.values())
    if imbalance > max_answer_imbalance:
        raise ValueError("cue group answer distribution is imbalanced")
    total = sum(item.count for item in records)
    successes = sum(item.cue_only_correct_count for item in records)
    if successes == total:
        upper = 1.0
    else:
        from scipy.stats import beta

        upper = float(beta.ppf(0.95, successes + 1, total - successes))
    if upper >= float(max_cue_only_accuracy_upper):
        raise ValueError("cue-only accuracy upper confidence bound is too high")
    cue_signature, question_signature = next(iter(groups))
    return CueGroupAudit(
        cue_signature=cue_signature,
        question_signature=question_signature,
        distinct_answers=len(counts),
        total_records=total,
        answer_imbalance=imbalance,
        cue_only_accuracy_upper=upper,
    )


def assert_disjoint_numeric_tables(
    tables_by_split: Mapping[str, Sequence[tuple[int, int, int, int]]],
) -> None:
    if not isinstance(tables_by_split, Mapping) or not tables_by_split:
        raise ValueError("tables_by_split must be a non-empty mapping")
    owner: dict[tuple[int, int, int, int], str] = {}
    for split, tables in tables_by_split.items():
        if not isinstance(split, str) or not split:
            raise ValueError("split identifiers must be non-empty strings")
        for table in tables:
            if not isinstance(table, tuple) or len(table) != 4:
                raise ValueError("numeric tables must contain exactly four values")
            if any(type(value) is not int for value in table):
                raise TypeError("numeric tables must contain exact integers")
            previous = owner.get(table)
            if previous is not None and previous != split:
                raise ValueError(f"numeric table overlap between {previous} and {split}")
            owner[table] = split
