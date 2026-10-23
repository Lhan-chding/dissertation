"""Separate developmental v2 replay and truthful AI-review accounting.

Never writes a production gate or calls a model. AI judgments are development
measurements, not human certification. Frozen v1 outputs remain untouched.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .c_diagnosis import summarize
from .replay import replay_group
from .training import native_advantages


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def developmental_replay(compact, tasks, output):
    from .verification_v2 import VERIFIER_VERSION, verify_semantics

    task_rows = read_rows(tasks)
    if any(t.get("pool") != "C_TRAIN" for t in task_rows):
        raise ValueError("This developmental run only accepts C_TRAIN tasks")
    by_qid = {t["qid"]: t for t in task_rows}
    if len(by_qid) != len(task_rows):
        raise ValueError("Duplicate task IDs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, seen = [], set()
    with (output / "GROUP_REPLAY_V2.jsonl").open("w") as stream:
        for group in read_rows(compact):
            rows = []
            if not group["rows"]:
                raise ValueError("Empty probe group")
            for record in group["rows"]:
                if record["row_id"] in seen:
                    raise ValueError("Duplicate answer ID")
                seen.add(record["row_id"])
                task = by_qid[record["qid"]]
                label = verify_semantics(task, record, record["score"])
                rows.append(dict(task=task, record=record, score=record["score"], label=label))
            if len({r["task"]["qid"] for r in rows}) != 1:
                raise ValueError("Multiple tasks in one group")
            task = rows[0]["task"]
            result = replay_group(
                dict(
                    group_id=group["group_id"],
                    source="C_TRAIN_PROBE",
                    step=0,
                    block="b" + str(rows[0]["record"]["block"]).removeprefix("b"),
                    family=task["family"],
                    world_id=task["world_id"],
                    rows=rows,
                ),
                native_advantages,
            )
            # replay_group's default metadata is v1; injected labels are v2.
            result["verifier_version"] = VERIFIER_VERSION
            result["development_only"] = True
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            results.append(result)
    summary = {
        "status": "DEVELOPMENT_ONLY_NOT_C_AUTHORIZATION",
        "verifier_version": VERIFIER_VERSION,
        "all": summarize(results),
        "by_family": {
            f: summarize([r for r in results if r["family"] == f])
            for f in sorted({r["family"] for r in results})
        },
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def assess_ai_reviews(key_rows, review_rows, labels_by_row, adjudications=()):
    keys = {r["review_id"]: r for r in key_rows}
    if len(keys) != len(key_rows):
        raise ValueError("Duplicate review keys")
    reviews = defaultdict(list)
    seen = set()
    for row in review_rows:
        identity = (row["review_id"], row["reviewer"])
        if identity in seen or row["review_id"] not in keys:
            raise ValueError("Duplicate or unknown review identity")
        if row["annotation_source"] != "AI_REVIEW":
            raise ValueError("This summary is exclusively for explicitly labeled AI reviews")
        if any(row[d] not in {"PASS", "FAIL", "UNKNOWN"} for d in "PS"):
            raise ValueError("Invalid review label")
        if not row.get("rationale", "").strip():
            raise ValueError("Review evidence/rationale required")
        seen.add(identity)
        reviews[row["review_id"]].append(row)
    decisions = {}
    for decision in adjudications:
        key = decision["review_id"]
        if key not in reviews or key in decisions:
            raise ValueError("Unknown or duplicate adjudication")
        if decision.get("annotation_source") != "AI_ADJUDICATION" or not decision.get("reason"):
            raise ValueError("Explicit AI adjudication provenance and rationale required")
        if any(decision[d] not in {"PASS", "FAIL", "UNKNOWN"} for d in "PS"):
            raise ValueError("Invalid adjudication label")
        decisions[key] = decision
    disagreements, resolved, mismatches = [], [], []
    counts = {d: Counter() for d in "PS"}
    for review_id, records in reviews.items():
        key = keys[review_id]
        automatic = labels_by_row[key["row_id"]]
        for dimension in "PS":
            judgments = {r[dimension] for r in records}
            if len(judgments) != 1:
                conflict = dict(review_id=review_id, dimension=dimension, labels=sorted(judgments))
                if review_id not in decisions:
                    disagreements.append(conflict)
                    counts[dimension]["unresolved_ai_disagreement"] += 1
                    continue
                resolved.append(dict(conflict, adjudication=decisions[review_id]))
            value = (
                decisions[review_id][dimension] if review_id in decisions else next(iter(judgments))
            )
            auto = automatic["dimensions"][dimension]["status"]
            counts[dimension][f"auto_{auto}__ai_{value}"] += 1
            if value != auto:
                mismatches.append(
                    dict(
                        review_id=review_id,
                        dimension=dimension,
                        automatic=auto,
                        ai=value,
                        row_id=key["row_id"],
                    )
                )
    return dict(
        status="AI_DEVELOPMENT_REVIEW_NOT_HUMAN_CERTIFICATION",
        annotation_source="AI_REVIEW",
        expected_rows=len(keys),
        reviewed_rows=len(reviews),
        complete=len(reviews) == len(keys),
        independently_double_reviewed=sum(len(r) >= 2 for r in reviews.values()),
        human_review_count=0,
        human_audit_complete=False,
        production_gate_modified=False,
        chart_images_reviewed=False,
        counts=counts,
        unresolved_disagreements=disagreements,
        resolved_disagreements=resolved,
        parser_ai_mismatches=mismatches,
        limitations="Stratified development sample; shared-model AI reviewers are not independent "
        "human truth. No population precision or reward-positive Wilson certification.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    developmental_replay(args.compact, args.tasks, args.output)


if __name__ == "__main__":
    main()
