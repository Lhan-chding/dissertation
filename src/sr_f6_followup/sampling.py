"""Fixed coverage sampling of original SCHEMA_CONFIRM_V3 answers.

The frozen producer supplies the neutral selection/capacity/ranking algorithms.
Only the registered confirmation namespace is accepted; no DEV_CAL alias or
producer rewrite is introduced. This module never performs independent review.
"""

from __future__ import annotations

import hashlib

from sr_f6.review import (
    BLOCKS,
    DOMAINS,
    PER_FAMILY,
    PROTOCOL,
    RARE,
    REVIEW_SEED,
    ROOT_CAP,
    TARGETS,
    TOTAL,
    _alias,
    _coverage,
    _digest,
    _labels,
    _rank,
    _raw_evidence,
    _select,
    _state,
    _targets,
)

VERSION = "srf6-followup-blind-review-v1"
PHASE = "SCHEMA_CONFIRM_V3"
SPLIT = "DEV_SCHEMA_CONFIRM"


def _pool(records, seed, raw_by_row):
    by_id = {}
    for source in records:
        row = dict(source)
        parent, factor, family = (row.get(key) for key in ("parent", "factor", "family"))
        if (parent, factor, family) not in BLOCKS or (
            row.get("phase") != PHASE or row.get("split") != SPLIT
        ):
            raise PermissionError("Review may only sample registered confirmation cells")
        root, qid, draw = row.get("root_id"), row.get("qid"), row.get("draw")
        roots = {f"DEV_SCHEMA_CONFIRM_{family}_{index:04d}" for index in range(8)}
        if root not in roots or qid not in {f"{root}_{factor}_v{v}" for v in range(4)}:
            raise PermissionError("Review question is outside registered confirmation")
        if type(draw) is not int or draw not in range(8):
            raise ValueError("Registered census draw must be 0 through 7")
        row_id = row.get("row_id")
        if row_id != f"{parent}/{qid}/{draw}" or row_id in by_id:
            raise ValueError("Invalid or duplicate original row identity")
        text = row.get("full_text")
        if not isinstance(text, str) or row.get("raw_text", text) != text:
            raise ValueError("Original full_text must be present and unmodified")
        if row.get("finish_reason") not in ("balanced", "eos", "length"):
            raise ValueError("Annotated program finish reason is required")
        raw_reason, raw_hash = _raw_evidence(row, raw_by_row)
        labels = _labels(row)
        key = _digest([parent, factor, qid, text])
        by_id[row_id] = dict(
            row_id=row_id,
            block=(parent, factor, family),
            root_id=root,
            qid=qid,
            full_text=text,
            finish_reason=raw_reason,
            finish_reason_source="SCHEMA_CONFIRM_V3_RAW_BY_ROW",
            program_finish_reason=row["finish_reason"],
            raw_sampling_evidence_sha256=raw_hash,
            labels=labels,
            unit_id=key,
            alias=_alias(row),
            rank=_rank(seed, "representative", row_id),
        )
    if set(raw_by_row) != set(by_id):
        raise PermissionError("Raw mapping must match the exact annotated census row set")
    for row in by_id.values():
        if row["alias"] is not None and row["alias"] not in by_id:
            raise PermissionError("Alias must reference this original confirmation matrix")
        visited = {row["row_id"]}
        origin = row["alias"] or row["row_id"]
        while origin in by_id and by_id[origin]["alias"]:
            if origin in visited:
                raise ValueError("Cyclic source-row provenance")
            visited.add(origin)
            if by_id[origin]["full_text"] != row["full_text"]:
                raise ValueError("Explicit aliases disagree on original text")
            origin = by_id[origin]["alias"]
        if origin in by_id and by_id[origin]["full_text"] != row["full_text"]:
            raise ValueError("Explicit aliases disagree on original text")
        row["origin_id"] = origin
    units = {}
    for row in sorted(by_id.values(), key=lambda item: (item["rank"], item["row_id"])):
        key = row["unit_id"]
        if key not in units:
            labels = row["labels"]
            units[key] = dict(
                unit_id=key,
                block=row["block"],
                root_id=row["root_id"],
                qid=row["qid"],
                text=row["full_text"],
                labels=labels,
                stratum=(labels["A"], labels["J"], _state(labels["S"]), _state(labels["B"])),
                targets=_targets(labels),
                rank=_rank(seed, "candidate", key),
                rows=[],
                options={},
            )
        unit = units[key]
        if unit["labels"] != row["labels"]:
            raise ValueError("Identical cell/qid/text has inconsistent labels")
        unit["rows"].append(row)
        unit["options"].setdefault(row["origin_id"], row)
    return units, by_id


def build_review(records, *, raw_by_row, source_identity, seed=REVIEW_SEED):
    """Return ``(private_ledger, blind_items)``; neither is a completed review.

    ``records`` are original annotated census rows, not oracle/task objects.
    ``raw_by_row`` maps each registered row_id to its original sampling row.
    The caller adds qid from the raw source filename when the sampler did not
    store it; qid, seed, tokens, full_text and model_input must match. Do not
    construct this mapping from annotated rows: their finish_reason may have
    been normalized by the scorer. No scorer reason is used as a fallback.
    ``source_identity`` names externally verified source SHA256 identities.
    Incomplete/duplicate-constrained pools produce an explicit insufficient
    result and only their actual unique selections, never fabricated fillers.
    """
    if type(seed) is not int or seed != REVIEW_SEED:
        raise ValueError("Review seed is frozen at 2026102701; post hoc reselection is prohibited")
    if not isinstance(raw_by_row, dict):
        raise ValueError("An explicit original raw sampling row mapping is required")
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("Explicit source identities required")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for name, value in source_identity.items()
    ):
        raise ValueError("Source identities must be named lowercase SHA256 values")
    units, rows = _pool(records, seed, raw_by_row)
    selected, reasons, maximum = _select(units)
    order = sorted(selected, key=lambda key: (_rank(seed, "blind_order", key), key))
    blind, mapping = [], []
    for index, key in enumerate(order):
        row, unit = selected[key], units[key]
        anonymous_id = f"R{index + 1:04d}"
        blind.append(
            dict(
                anonymous_id=anonymous_id,
                raw_text=row["full_text"],
                protocol_version=PROTOCOL,
                generation_finish_reason=row["finish_reason"],
            )
        )
        mapping.append(
            dict(
                anonymous_id=anonymous_id,
                row_id=row["row_id"],
                origin_id=row["origin_id"],
                unit_id=key,
                parent=unit["block"][0],
                factor=unit["block"][1],
                family=unit["block"][2],
                root_id=unit["root_id"],
                qid=unit["qid"],
                raw_text_sha256=hashlib.sha256(row["full_text"].encode("utf-8")).hexdigest(),
                program_labels=dict(unit["labels"]),
                program_finish_reason=row["program_finish_reason"],
                generation_finish_reason=row["finish_reason"],
                generation_finish_reason_source=row["finish_reason_source"],
                raw_sampling_evidence_sha256=row["raw_sampling_evidence_sha256"],
                stratum=list(unit["stratum"]),
                selection_reason=reasons[key],
                source_row_ids=[r["row_id"] for r in unit["rows"]],
            )
        )
    rare_cases = []
    for key, unit in sorted(units.items()):
        patterns = [target for target in RARE if target in unit["targets"]]
        if patterns:
            rare_cases.append(
                dict(
                    unit_id=key,
                    parent=unit["block"][0],
                    factor=unit["block"][1],
                    family=unit["block"][2],
                    root_id=unit["root_id"],
                    qid=unit["qid"],
                    patterns=patterns,
                    program_labels=dict(unit["labels"]),
                    source_row_ids=[r["row_id"] for r in unit["rows"]],
                    selected=key in selected,
                )
            )
    ledger = dict(
        version=VERSION,
        status="BLIND_PACKAGE_ONLY",
        review_status="REVIEW_NOT_RUN",
        sampling_status="COMPLETE_360" if maximum == TOTAL else "INSUFFICIENT_UNIQUE_RESPONSES",
        reviewer_type=None,
        execution_origin="OFFLINE_SAMPLING_ONLY",
        seed=seed,
        generation_finish_reason_source="SCHEMA_CONFIRM_V3_RAW_BY_ROW",
        seed_domains=dict(DOMAINS),
        source_identity=dict(source_identity),
        selection_inputs_sha256=_digest([rows[key] for key in sorted(rows)]),
        blind_items_sha256=_digest(blind),
        raw_count=len(rows),
        unique_units=len(units),
        selected_count=len(selected),
        required_count=TOTAL,
        shortfall=TOTAL - len(selected),
        root_cap=ROOT_CAP,
        root_cap_scope="parent_factor",
        uniqueness_scope="parent_factor_qid_exact_full_text",
        alias_policy="explicit_source_row_provenance_only",
        coverage_priority=list(TARGETS),
        per_family=PER_FAMILY,
        stratum_tie_break="selected_count, remaining_unique_supply, lexical_A_J_S_B",
        candidate_tie_break=(
            "candidate_domain_sha256, unit_id; representative_domain_sha256, row_id"
        ),
        unmet_coverage_policy="report; never duplicate, relax caps, or alter source labels",
        interpretation="Coverage sample, not a population rate or independent correctness review",
        selected=mapping,
        blocks=_coverage(units, selected),
        rare_cases=rare_cases,
    )
    return ledger, blind
