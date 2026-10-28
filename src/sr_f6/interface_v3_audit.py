"""Existing-response J/A decomposition without converting missing evidence to truth.

This diagnostic supplies no reward, gate, repair or hidden-program attribution.
Native parser defaults are retained separately from content-level knownness.
"""

from collections import Counter

CATEGORIES = (
    "INVALID_INTERFACE",
    "UNREADABLE_CERTIFICATE",
    "TRUE_BUT_INSUFFICIENT",
    "FALSE_CERTIFICATE",
    "OTHER_J_REQUIREMENT",
)


def _classification(row):
    score, diagnostic = row["wire_score"], row["diagnostic"]
    if score["V"] == 0:
        return "INVALID_INTERFACE"
    certificate = diagnostic["certificate"]
    if certificate.get("known") is not True:
        return "UNREADABLE_CERTIFICATE"
    if certificate.get("truth") == 1 and certificate.get("sufficient") == 0:
        return "TRUE_BUT_INSUFFICIENT"
    if certificate.get("truth") == 0:
        return "FALSE_CERTIFICATE"
    return "OTHER_J_REQUIREMENT"


def _group(rows):
    gap = [r for r in rows if r["wire_score"]["A"] == 1 and r["wire_score"]["J"] == 0]
    partition = Counter(_classification(r) for r in gap)
    eligible_correct = sum(r["wire_score"]["A"] == r["wire_score"]["V"] == 1 for r in rows)
    content = {}
    for key in ("truth", "sufficient"):
        values = [
            r["diagnostic"]["certificate"].get(key)
            if r["diagnostic"]["certificate"].get("known") is True
            else None
            for r in gap
        ]
        content[key] = {
            "true": values.count(1),
            "false": values.count(0),
            "unknown": values.count(None),
        }
    native = {}
    for key in ("certificate_truth", "certificate_sufficient", "presentation", "finish"):
        native[key] = dict(
            Counter(
                "MISSING" if key not in r["wire_score"] else str(r["wire_score"][key]) for r in gap
            )
        )
    return {
        "n": len(rows),
        "A_correct": sum(r["wire_score"]["A"] for r in rows),
        "J_correct": sum(r["wire_score"]["J"] for r in rows),
        "A_correct_V_valid": eligible_correct,
        "A_correct_J_zero": len(gap),
        "exclusive_gap_partition": {key: partition[key] for key in CATEGORIES},
        "content_certificate_labels": content,
        "native_flag_counts_not_semantic_labels": native,
        "true_insufficient_fraction_of_A_correct_V_valid": partition["TRUE_BUT_INSUFFICIENT"]
        / eligible_correct
        if eligible_correct
        else None,
        "raw_finish_reasons_in_gap": dict(Counter(r["finish_reason"] for r in gap)),
    }


def j_gap_breakdown(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Nonempty existing responses required")
    if len({r["row_id"] for r in records}) != len(records):
        raise ValueError("Duplicate existing response")
    for row in records:
        score = row["wire_score"]
        if any(
            type(score.get(key)) is not int or score[key] not in (0, 1) for key in ("A", "J", "V")
        ):
            raise ValueError("Explicit binary original labels required")
        if score["J"] and not (score["A"] and score["V"]):
            raise ValueError("Original J requires A and V")
    gap_rows = []
    for row in records:
        if row["wire_score"]["A"] == 1 and row["wire_score"]["J"] == 0:
            gap_rows.append(
                {
                    "row_id": row["row_id"],
                    "parent": row["parent"],
                    "factor": row["factor"],
                    "family": row["family"],
                    "category": _classification(row),
                    "certificate_content": row["diagnostic"]["certificate"],
                    "native_reason": row["wire_score"].get("native_reason"),
                    "core_reason": row["diagnostic"].get("core_score", {}).get("reason"),
                    "raw_finish_reason": row["finish_reason"],
                }
            )
    return {
        "schema": "srf6-interface-v3-j-gap-diagnostic-v1",
        "all": _group(records),
        "by_parent_factor": {
            f"{p}/{f}": _group([r for r in records if (r["parent"], r["factor"]) == (p, f)])
            for p, f in sorted({(r["parent"], r["factor"]) for r in records})
        },
        "by_family": {
            name: _group([r for r in records if r["family"] == name])
            for name in sorted({r["family"] for r in records})
        },
        "gap_rows": gap_rows,
        "new_model_calls": 0,
        "optimizer_updates": 0,
        "additional_admission_gate": False,
        "interpretation": (
            "Descriptive decomposition; native default zeros are not known semantic "
            "failures. No 20-percent causal or stopping rule."
        ),
    }
