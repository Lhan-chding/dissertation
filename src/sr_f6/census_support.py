"""Read-only checks of the fixed section 5.4 census feasibility conditions.

This is not a stage-gate signer. ``assess_support`` checks a complete_summary
projection, not the truth of arbitrary caller-supplied labels. ``check_support``
first recomputes that projection through census.complete_summary, including its
persisted annotated-G8 validation. Independent original-raw, worker and actual
call-accounting verification requires census_evidence.validate_saved_census;
scientific admission must assess that verified summary. Coverage certificates
remain dependent on the earlier mathematical audit; neither entry point executes
a model or optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from srf6_ref.core import FAMILIES

PARENTS = ("b0", "b1", "b2")
FACTORS = ("P0", "P1")
CELL_ROWS = 1536
CENSUS_ROWS = 9216
TRAIN_ROOTS = 288
_CROSS = re.compile(r"A([01])/J([01])/S(None|[01])/B(None|[01])\Z")


class InvalidEvidence(ValueError):
    """Missing, inconsistent, or nonfinite input is not feasibility evidence."""


def _require(condition, path, message):
    if not condition:
        raise InvalidEvidence(f"{path}: {message}")


def _mapping(value, path):
    _require(isinstance(value, dict), path, "object required")
    return value


def _integer(value, path, *, minimum=0):
    _require(type(value) is int and value >= minimum, path, "nonnegative integer required")
    return value


def _number(value, path, *, lower=0, upper=1):
    _require(
        type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper,
        path,
        f"finite number in [{lower}, {upper}] required",
    )
    return value


def _count_from_rate(rate, denominator, path):
    rate = _number(rate, path)
    count = round(rate * denominator)
    _require(abs(rate * denominator - count) <= 1e-9, path, "not an integer sample count")
    return count


def _same_number(actual, expected, path):
    _number(actual, path, lower=-1, upper=max(1, abs(expected)))
    _require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), path, "recomputed mismatch")


def _cells(summary):
    cells = _mapping(summary.get("cells"), "cells")
    _require(
        set(cells) == {f"{p}/{f}" for p in PARENTS for f in FACTORS},
        "cells",
        "six exact parent/factor cells required",
    )
    observed = {}
    for key, cell in cells.items():
        cell = _mapping(cell, f"cells.{key}")
        n = _integer(cell.get("n"), f"cells.{key}.n", minimum=1)
        _require(n == CELL_ROWS, f"cells.{key}.n", "full 1536-row denominator required")
        counts = {
            name: _count_from_rate(cell.get(f"{name}_rate"), n, f"cells.{key}.{name}_rate")
            for name in ("V", "S_known", "B_known", "A", "J")
        }
        cross = _mapping(cell.get("A_J_S_B_cross"), f"cells.{key}.A_J_S_B_cross")
        rebuilt = Counter()
        for label, count in cross.items():
            match = _CROSS.fullmatch(label) if isinstance(label, str) else None
            _require(
                match is not None,
                f"cells.{key}.cross",
                "binary A/J and literal None/0/1 S/B required",
            )
            count = _integer(count, f"cells.{key}.cross.{label}", minimum=1)
            a, j, s, b = match.groups()
            rebuilt.update(n=count, A=count * int(a), J=count * int(j))
            rebuilt["S_known"] += count * (s != "None")
            rebuilt["B_known"] += count * (b != "None")
        _require(rebuilt["n"] == n, f"cells.{key}.cross", "UNKNOWN-inclusive denominator mismatch")
        for name in ("A", "J", "S_known", "B_known"):
            _require(
                rebuilt[name] == counts[name], f"cells.{key}.{name}", "cross-table count mismatch"
            )
        observed[key] = {
            "n": n,
            **counts,
            "S_unknown": n - counts["S_known"],
            "B_unknown": n - counts["B_known"],
        }
    return observed


def _groups(summary, cells):
    groups = summary.get("groups")
    _require(
        isinstance(groups, list) and len(groups) == 1152,
        "groups",
        "complete 1152 G8 facts required",
    )
    seen, root_families, geometry = set(), {}, Counter()
    by_factor = defaultdict(list)
    for index, group in enumerate(groups):
        path = f"groups[{index}]"
        group = _mapping(group, path)
        parent, factor = group.get("parent"), group.get("factor")
        _require(parent in PARENTS and factor in FACTORS, path, "unregistered parent/factor")
        root, family, gid = group.get("root_id"), group.get("family"), group.get("group_id")
        _require(isinstance(root, str) and bool(root), path, "root identity required")
        _require(isinstance(gid, str) and bool(gid), path, "group identity required")
        _require(family in FAMILIES, path, "unregistered family")
        key = (parent, factor, gid)
        _require(key not in seen, path, "duplicate G8 identity")
        seen.add(key)
        _require(root_families.setdefault(root, family) == family, path, "root changed family")
        geometry[(parent, factor, root)] += 1
        for name in ("nonzero", "exchangeable", "changed"):
            _require(type(group.get(name)) is bool, f"{path}.{name}", "literal boolean required")
        l1 = _number(group.get("credit_L1"), f"{path}.credit_L1", upper=2)
        _require(group["nonzero"] == (l1 > 1e-15), path, "COND nonzero/L1 mismatch")
        _require(
            group["exchangeable"] == group["nonzero"],
            path,
            "centered COND exchangeability mismatch",
        )
        _require(
            not group["changed"] or group["exchangeable"],
            path,
            "SHUF change requires exchangeable credit",
        )
        _require(group.get("J_class") in ("all_zero", "all_one", "mixed"), path, "J-class required")
        by_factor[factor].append(group)
    _require(
        Counter(root_families.values()) == Counter({f: 8 for f in FAMILIES}),
        "groups",
        "48 distinct roots, eight per family required",
    )
    expected_geometry = Counter(
        {(p, f, root): 4 for p in PARENTS for f in FACTORS for root in root_families}
    )
    _require(
        geometry == expected_geometry,
        "groups",
        "each parent/factor/root requires four distinct views",
    )
    for parent in PARENTS:
        for factor in FACTORS:
            classes = Counter(g["J_class"] for g in by_factor[factor] if g["parent"] == parent)
            lower = 8 * classes["all_one"] + classes["mixed"]
            upper = 8 * classes["all_one"] + 7 * classes["mixed"]
            _require(
                lower <= cells[f"{parent}/{factor}"]["J"] <= upper,
                f"groups.{parent}/{factor}.J_class",
                "G8 classes disagree with cell J count",
            )
    factors = {}
    for factor in FACTORS:
        all_groups = by_factor[factor]
        nonzero = [group for group in all_groups if group["nonzero"]]
        factors[factor] = {
            "groups": len(all_groups),
            "nonzero_groups": len(nonzero),
            "roots": len({g["root_id"] for g in nonzero}),
            "families": len({g["family"] for g in nonzero}),
            "parents": len({g["parent"] for g in nonzero}),
            "exchangeable_groups": sum(g["exchangeable"] for g in all_groups),
            "actually_changed_groups": sum(g["changed"] for g in all_groups),
            "nonzero_by_J_class": dict(Counter(g["J_class"] for g in nonzero)),
        }
    changed = [group for group in groups if group["changed"]]
    supplied = _mapping(summary.get("factors"), "factors")
    _require(set(supplied) == set(FACTORS), "factors", "both factors required")
    for factor, actual in factors.items():
        old = _mapping(supplied[factor], f"factors.{factor}")
        for name, expected in actual.items():
            if name == "nonzero_by_J_class":
                table = _mapping(old.get(name), f"factors.{factor}.{name}")
                for key, value in table.items():
                    _require(key in ("all_zero", "all_one", "mixed"), name, "unknown J class")
                    _integer(value, name, minimum=1)
            else:
                _integer(old.get(name), f"factors.{factor}.{name}")
            _require(
                old.get(name) == expected,
                f"factors.{factor}.{name}",
                "full group facts disagree with supplied aggregate",
            )
    totals = {
        "total_changed_groups": len(changed),
        "total_changed_roots": len({g["root_id"] for g in changed}),
    }
    for name, expected in totals.items():
        _integer(summary.get(name), name)
        _require(
            summary.get(name) == expected, name, "full group facts disagree with supplied aggregate"
        )
    return factors, totals


def _coverage(audit, train_roots):
    audit = _mapping(audit, "cpu_audit")
    _require(
        audit.get("status") == "PASS", "cpu_audit.status", "completed mathematical audit required"
    )
    for name, value in (
        ("model_calls", 0),
        ("independent_native_oracles", 2),
        ("cross_split_world_duplicates", 0),
    ):
        _integer(audit.get(name), f"cpu_audit.{name}")
        _require(audit[name] == value, f"cpu_audit.{name}", "registered CPU-audit value required")
    splits = _mapping(audit.get("splits"), "cpu_audit.splits")
    for split, roots, questions, per_family in (
        ("DEV_CAL", 48, 384, 64),
        ("TRAIN", 288, 2304, 384),
    ):
        entry = _mapping(splits.get(split), f"cpu_audit.splits.{split}")
        for name, expected in (("roots", roots), ("questions", questions)):
            _integer(entry.get(name), f"{split}.{name}")
            _require(entry[name] == expected, f"{split}.{name}", "registered denominator required")
        _require(entry.get("factors") == list(FACTORS), split, "both registered factors required")
        family_counts = _mapping(entry.get("family_questions"), f"{split}.family_questions")
        _require(set(family_counts) == set(FAMILIES), split, "six families required")
        for family in FAMILIES:
            _integer(family_counts[family], f"{split}.{family}")
            _require(family_counts[family] == per_family, split, "family question count mismatch")
    coverage = _mapping(audit.get("training_manipulation"), "training_manipulation")
    _require(
        coverage.get("model_outputs_used") is False,
        "training_manipulation",
        "coverage must be model-blind",
    )
    _integer(coverage.get("roots"), "training_manipulation.roots")
    _require(
        coverage["roots"] == TRAIN_ROOTS, "training_manipulation.roots", "288 TRAIN roots required"
    )
    mean = _number(coverage.get("mean_gap"), "training_manipulation.mean_gap", lower=-1)
    gains = _count_from_rate(
        coverage.get("strict_gain_fraction"),
        TRAIN_ROOTS,
        "training_manipulation.strict_gain_fraction",
    )
    if train_roots is not None:
        _require(
            isinstance(train_roots, list) and len(train_roots) == TRAIN_ROOTS,
            "train_roots",
            "288 full TRAIN certificates required",
        )
        seen, gaps = set(), []
        for index, root in enumerate(train_roots):
            root = _mapping(root, f"train_roots[{index}]")
            rid = root.get("root_id")
            _require(
                isinstance(rid, str) and rid.startswith("TRAIN_") and rid not in seen,
                "train_roots",
                "distinct TRAIN root identities required",
            )
            seen.add(rid)
            certificate = _mapping(root.get("certificate"), f"train_roots[{index}].certificate")
            _require(
                certificate.get("root_id") == rid,
                "train_roots",
                "root/certificate identity mismatch",
            )
            p0 = _number(certificate.get("coverage_P0"), f"{rid}.coverage_P0")
            p1 = _number(certificate.get("coverage_P1"), f"{rid}.coverage_P1")
            gaps.append(p1 - p0)
        # Match DATA.audit_banks' registered aggregation and its root order.
        recomputed_mean = sum(gaps) / TRAIN_ROOTS
        recomputed_gains = sum(gap > 0 for gap in gaps)
        _same_number(mean, recomputed_mean, "training_manipulation.mean_gap")
        _require(
            gains == recomputed_gains,
            "training_manipulation.strict_gain_fraction",
            "strict-positive certificate count differs from audit",
        )
        mean, gains = recomputed_mean, recomputed_gains
    expected_status = (
        "PASS" if mean >= 0.05 and gains * 3 >= TRAIN_ROOTS else "INSUFFICIENT_REGISTERED_EXPOSURE"
    )
    _require(
        coverage.get("status") == expected_status,
        "training_manipulation.status",
        "stored status disagrees with fixed thresholds",
    )
    return {
        "roots": TRAIN_ROOTS,
        "mean_gap": mean,
        "strict_gain_roots": gains,
        "strict_gain_fraction": gains / TRAIN_ROOTS,
        "model_outputs_used": False,
        "certificate_recomputed": train_roots is not None,
        "truth_dependency": (
            "Earlier DATA mathematical audit; certificates are not independently regenerated here."
        ),
    }


def _report():
    return {
        "status": "INVALID_SUPPORT_EVIDENCE",
        "scope": "registered_feasibility_only",
        "reviewer_type": "PROGRAM",
        "gate_status": "NOT_SIGNED",
        "training_authorized": False,
        "model_calls": 0,
        "optimizer_updates": 0,
        "checks": [],
        "failures": [],
        "pending_verification": {
            "independent_360_response_measurement_review": "NOT_VERIFIED_BY_THIS_CHECK",
            "real_update_numerical_and_recovery_replay": "NOT_VERIFIED_BY_THIS_CHECK",
            "SOURCE_BINDING_final_gate": "NOT_SIGNED",
            "DATA_AND_MEASUREMENT_final_gate": "NOT_SIGNED",
        },
        "claim_limit": (
            "Minimum registered exposure only; no learning-effect, power, "
            "independence, or training-authorization conclusion."
        ),
    }


def assess_support(census_summary, cpu_audit, *, train_roots=None):
    """Validate a complete-summary projection, recompute support, return a report.

    Inputs are never mutated. UNKNOWN remains in the 1536-row denominators.
    Invalid structure/tampered aggregates fail closed; all legitimate threshold
    deficits are collected as INSUFFICIENT_REGISTERED_EXPOSURE. The optional
    TRAIN certificates verify coverage aggregates, not certificate mathematics.
    """
    report = _report()
    report["evidence_origin"] = "CALLER_SUPPLIED_COMPLETE_SUMMARY_PROJECTION"
    try:
        summary = _mapping(census_summary, "census_summary")
        _require(
            summary.get("status") == "COMPLETE_NOT_GATE_SIGNED",
            "status",
            "complete_summary result required",
        )
        for name, expected in (
            ("rows", CENSUS_ROWS),
            ("source_rows", CENSUS_ROWS),
            ("optimizer_updates", 0),
        ):
            _integer(summary.get(name), name)
            _require(summary[name] == expected, name, "full registered untrained census required")
        identity = summary.get("frozen_spec_sha256")
        _require(
            isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity) is not None,
            "frozen_spec_sha256",
            "frozen census identity required",
        )
        cells = _cells(summary)
        factors, totals = _groups(summary, cells)
        coverage = _coverage(cpu_audit, train_roots)
        report.update(
            frozen_spec_sha256=identity,
            observed={
                "cells": cells,
                "factors": factors,
                **totals,
                "training_manipulation": coverage,
            },
        )
        checks = report["checks"]

        def minimum(name, observed, threshold, *, denominator=None):
            item = {
                "name": name,
                "observed": observed,
                "minimum": threshold,
                "status": "PASS" if observed >= threshold else "FAIL",
            }
            if denominator is not None:
                item["denominator"] = denominator
            checks.append(item)

        for cell, counts in cells.items():
            for label in ("V", "S_known", "B_known"):
                # 90% of 1536 requires 1383 observed rows; never condition on V.
                minimum(
                    f"{cell}.{label}_count",
                    counts[label],
                    math.ceil(0.9 * CELL_ROWS),
                    denominator=CELL_ROWS,
                )
        for factor, values in factors.items():
            for key, threshold in (
                ("groups", 576),
                ("nonzero_groups", 24),
                ("roots", 12),
                ("families", 3),
                ("parents", 2),
            ):
                minimum(f"{factor}.{key}", values[key], threshold)
        minimum("SHUF.total_changed_groups", totals["total_changed_groups"], 12)
        minimum("SHUF.total_changed_roots", totals["total_changed_roots"], 6)
        minimum("TRAIN.coverage_mean_gap", coverage["mean_gap"], 0.05, denominator=TRAIN_ROOTS)
        minimum(
            "TRAIN.strict_gain_roots", coverage["strict_gain_roots"], 96, denominator=TRAIN_ROOTS
        )
        report["failures"] = [dict(item) for item in checks if item["status"] != "PASS"]
        report["status"] = "INSUFFICIENT_REGISTERED_EXPOSURE" if report["failures"] else "PASS"
    except InvalidEvidence as exc:
        report["failures"] = [{"name": "evidence_integrity", "status": "FAIL", "detail": str(exc)}]
    return report


def check_support(root):
    """Read actual saved census/coverage evidence; never write or sign a gate."""
    from .census import complete_summary

    root = Path(root)
    try:
        summary = complete_summary(root)
        with (root / "DATA/CPU_AUDIT.json").open() as stream:
            audit = json.load(stream)
        train_roots = []
        with (root / "DATA/ROOT_CERTIFICATES_PRIVATE.jsonl").open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                _mapping(item, "root_certificate")
                rid = item.get("root_id")
                _require(isinstance(rid, str), "root_certificate", "root_id required")
                if rid.startswith("TRAIN_"):
                    train_roots.append(item)
        report = assess_support(summary, audit, train_roots=train_roots)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report = _report()
        report["failures"] = [
            {
                "name": "evidence_integrity",
                "status": "FAIL",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        ]
    report["evidence_origin"] = "LOCAL_RECOMPUTATION_OF_PERSISTED_CENSUS_AND_TRAIN_CERTIFICATES"
    report["run_root"] = str(root.resolve())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    report = check_support(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
