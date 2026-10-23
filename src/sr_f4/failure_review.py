"""Join blinded AI extractions to targets, then replay existing C failures only.

AI judgments are supplied records, never manufactured by this module. Reviewers
see response evidence/work and the world, but not query, target, answer or J.
This is developmental AI review, not independent-provider or human certification.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from .development_review import read_rows

VERSION = "srf4-c-failure-ai-development-r1"
STATUSES = {"PASS", "FAIL", "UNKNOWN"}


def _indexed(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def _validate(row, blind, source):
    if row.get("annotation_source") != source or not row.get("rationale", "").strip():
        raise ValueError("Explicit AI provenance and rationale required")
    if row.get("P") not in STATUSES or row.get("extraction_status") not in {"KNOWN", "UNKNOWN"}:
        raise ValueError("Invalid AI judgment")
    if row.get("selection_series") not in {None, "Alpha", "Beta"}:
        raise ValueError("Invalid observed ranking series")
    categories = row.get("adopted_categories")
    quotes = row.get("evidence_quote")
    if not isinstance(quotes, list) or any(
        not isinstance(q, str) or not q or q not in (blind.get("work") or "") for q in quotes
    ):
        raise ValueError("Evidence quotations must occur verbatim in response work")
    if row["extraction_status"] == "UNKNOWN":
        if categories is not None:
            raise ValueError("Unknown selection cannot assert a category set")
    elif (
        not isinstance(categories, list)
        or any(not isinstance(c, str) for c in categories)
        or len(set(categories)) != len(categories)
        or not set(categories).issubset(blind["world"]["categories"])
        or not quotes
    ):
        raise ValueError("Known selection needs canonical unique categories and evidence")


def _identity(row):
    return (
        row["P"], row["extraction_status"],
        tuple(sorted(row["adopted_categories"])) if row["adopted_categories"] is not None else None,
        row.get("selection_series"),
    )


def resolve(keys, blinded, reviews, adjudications=()):
    """Require complete review and explicitly adjudicate all extraction conflicts."""
    keyed, inputs = _indexed(keys, "review_id"), _indexed(blinded, "review_id")
    if set(keyed) != set(inputs):
        raise ValueError("Blind inputs and private key must match exactly")
    grouped, seen = defaultdict(list), set()
    for r in reviews:
        identity = (r["review_id"], r.get("reviewer"))
        if not identity[1] or identity in seen or r["review_id"] not in keyed:
            raise ValueError("Unknown, missing or duplicate reviewer identity")
        _validate(r, inputs[r["review_id"]], "AI_REVIEW")
        seen.add(identity)
        grouped[r["review_id"]].append(r)
    if set(grouped) != set(keyed):
        raise ValueError("Every candidate failure must have an AI review")
    decisions = _indexed(list(adjudications), "review_id")
    if not set(decisions).issubset(keyed):
        raise ValueError("Unknown adjudication identity")
    for rid, r in decisions.items():
        _validate(r, inputs[rid], "AI_ADJUDICATION")
    labels, conflicts = [], []
    for rid, rs in grouped.items():
        disagreement = len({_identity(r) for r in rs}) > 1
        if disagreement:
            conflicts.append(rid)
            if rid not in decisions:
                raise ValueError(f"Unresolved AI extraction disagreement: {rid}")
        chosen = decisions.get(rid, rs[0])
        key = keyed[rid]
        status = "UNKNOWN"
        if chosen["extraction_status"] == "KNOWN":
            matches = set(chosen["adopted_categories"]) == set(key["target_categories"])
            ranking = chosen.get("selection_series")
            matches = matches and (ranking is None or ranking == key["selection_series"])
            status = "PASS" if matches else "FAIL"
        labels.append(dict(
            review_id=rid, row_id=key["row_id"], P=chosen["P"], S=status,
            extraction=chosen, reviewer_count=len(rs), adjudicated=rid in decisions,
            had_disagreement=disagreement,
        ))
    return labels, dict(
        status="AI_DEVELOPMENT_REVIEW_NOT_HUMAN_CERTIFICATION", reviewed_rows=len(labels),
        second_review_rows=sum(len(rs) > 1 for rs in grouped.values()),
        adjudicated_rows=len(decisions), disagreements=conflicts,
        P_counts=dict(Counter(r["P"] for r in labels)),
        S_counts=dict(Counter(r["S"] for r in labels)),
        human_review_count=0, independent_provider_review=False,
        chart_images_reviewed=False, production_gate_modified=False,
        limitations="All existing C candidate failures, not a random validation set. "
        "Reviewers share an AI platform; targeted second review is not certification.",
    )


def overlay(groups, resolved):
    """Make fresh minimal replay inputs; never retain stale derived residuals."""
    judgments = _indexed(resolved, "row_id")
    output, candidates, seen, versions = [], set(), set(), set()
    for group in groups:
        if group["source"] != "C_TRAIN_PROBE":
            continue
        version = group.get("verifier_version")
        if not isinstance(version, str) or not version:
            raise ValueError("Missing input verifier version")
        versions.add(version)
        current = {k: copy.deepcopy(group[k]) for k in (
            "group_id", "source", "block", "family", "world_id", "step", "cell", "A_J"
        )}
        current.update(
            verifier_version=VERSION, development_only=True, frozen_gate_eligible=False,
            measurement_mode="AI_REVIEWED_C_FAILURES_OTHER_LABELS_UNCHANGED", labels=[],
        )
        for original in group["labels"]:
            if original.get("verifier_version") != version:
                raise ValueError("Input label/group verifier versions disagree")
            row_id, n = original["row_id"], original["normalization"]
            if row_id in seen:
                raise ValueError("Duplicate replay row")
            seen.add(row_id)
            candidate = n["j"] == 0 and all(n[k] for k in ("protocol_valid", "finished", "defined"))
            if candidate:
                candidates.add(row_id)
                if row_id not in judgments:
                    raise ValueError("Missing candidate review")
            label = copy.deepcopy(original)
            label["input_verifier_version"] = label["verifier_version"]
            label["verifier_version"] = VERSION
            label["frozen_gate_eligible"] = False
            label["human_audited"] = False
            if candidate:
                review = judgments[row_id]
                label["review"] = copy.deepcopy(review)
                label["measurement_mode"] = "AI_REVIEWED_FAILURE"
                label["annotation_source"] = "AI_DEVELOPMENT_REVIEW"
                for d in "PS":
                    label["dimensions"][d] = dict(
                        status=review[d], reason="AI_REVIEW_OBSERVED_RESPONSE_ONLY",
                        observed=copy.deepcopy(review["extraction"]),
                        annotation_source=(
                            "AI_ADJUDICATION" if review["adjudicated"] else "AI_REVIEW"
                        ),
                    )
                    label.setdefault("dimensions_source_versions", {})[d] = VERSION
            current["labels"].append(label)
        output.append(current)
    if len(versions) != 1:
        raise ValueError("Compare different input verifier versions separately")
    if not output or candidates != set(judgments):
        raise ValueError("Review rows must exactly equal C F-prime candidate failures")
    return output


def validate_bindings(keys, blinded, groups, compact, tasks):
    """Bind reviews to the small original cache by contents, without hash scans."""
    inputs = _indexed(blinded, "review_id")
    task_by_qid = _indexed(tasks, "qid")
    raw = _indexed([r for g in compact for r in g["rows"]], "row_id")
    labels = _indexed([r for g in groups if g["source"] == "C_TRAIN_PROBE"
                       for r in g["labels"]], "row_id")
    for key in keys:
        record = raw[key["row_id"]]
        task, label = task_by_qid[record["qid"]], labels[key["row_id"]]
        if task.get("pool") != "C_TRAIN":
            raise ValueError("Only original C_TRAIN tasks may enter this review")
        obj = json.loads(record["full_text"])
        expected = dict(review_id=key["review_id"], world=task["world"],
                        evidence=obj.get("evidence"), work=obj.get("work"))
        metadata = dict(qid=task["qid"], world_id=task["world_id"], query=task["query"],
                        target_categories=task["target_sets"],
                        selection_series=task["query"].get("selection_series", "Alpha"))
        if inputs[key["review_id"]] != expected or any(key.get(k) != v
                                                       for k, v in metadata.items()):
            raise ValueError("Private key/blind content does not match original response/task")
        if (label["qid"] != task["qid"] or label["world_id"] != task["world_id"]
                or label["native_score"] != record["score"]
                or label["normalization"]["j"] != record["score"]["J"]):
            raise ValueError("Replay label identity/native score does not match original cache")


def run(review_root, replay, output):
    root, output = Path(review_root), Path(output)
    blinded = [r for p in sorted(root.glob("BLIND_BATCH_*.jsonl")) for r in read_rows(p)]
    reviews = [r for p in sorted(root.glob("REVIEWER_*.jsonl")) for r in read_rows(p)]
    adjudications = root / "ADJUDICATION.jsonl"
    keys = read_rows(root / "KEY_PRIVATE.jsonl")
    resolved, summary = resolve(
        keys, blinded, reviews,
        read_rows(adjudications) if adjudications.exists() else (),
    )
    original_groups = read_rows(replay)
    groups = overlay(original_groups, resolved)
    identities = {(g["source"], g["block"], g["group_id"]) for g in groups}
    answers = sum(len(g["labels"]) for g in groups)
    if (len(groups) != 144 or len(identities) != 144 or answers != 1152
            or len(resolved) != 243 or any(len(g["labels"]) != 8 for g in groups)):
        raise ValueError("Require the complete frozen C cache: 144 G8 groups / 1152 answers / "
                         "243 F-prime candidate failures")
    validate_bindings(
        keys, blinded, original_groups, read_rows(root.parent / "C_PROBE_COMPACT.jsonl"),
        read_rows(root.parent / "C_TASKS.jsonl"),
    )
    summary.update(verifier_version=VERSION, original_replay=str(replay),
                   groups=len(groups), answers=answers)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("RESOLVED.jsonl", resolved), ("GROUP_REPLAY.jsonl", groups)):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                          for r in rows))
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.review_root, args.replay, args.output)


if __name__ == "__main__":
    main()
