"""Relabel the existing C/B0/B2 development compact files without model calls.

This entry point is for the fixed 9,792-answer development cache. It does not
fetch data, read confirmation splits, modify old labels, or authorize training.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from .development_review import read_rows
from .replay import replay_group
from .training import native_advantages


def run(evidence_root, output, verifier="v3"):
    if verifier == "v3":
        from .verification_v3 import VERIFIER_VERSION, verify_semantics
    elif verifier == "v2":
        from .verification_v2 import VERIFIER_VERSION, verify_semantics
    else:
        raise ValueError("Only the separate v2/v3 developmental verifiers are supported")
    root, output = Path(evidence_root), Path(output)
    task_rows = read_rows(root / "C_TASKS.jsonl")
    tasks = {t["qid"]: t for t in task_rows}
    if len(tasks) != len(task_rows) or any(t.get("pool") != "C_TRAIN" for t in task_rows):
        raise ValueError("Unique C_TRAIN development tasks required")
    groups = []
    for g in read_rows(root / "C_PROBE_COMPACT.jsonl"):
        task = tasks[g["rows"][0]["qid"]]
        groups.append(
            dict(
                g,
                source="C_TRAIN_PROBE",
                task=task,
                block="b" + str(g["rows"][0]["block"]).removeprefix("b"),
                family=task["family"],
                world_id=task["world_id"],
                cell=task.get("cell"),
                step=0,
            )
        )
    with gzip.open(root / "REDESIGN_DEV_INPUTS/B0_B2_DEV_COMPACT.jsonl.gz", "rt") as stream:
        for line in stream:
            g = json.loads(line)
            allowed = {"B0": "OLD_TRAIN", "B2_DEV": "B_DEV"}
            if g.get("source") not in allowed or g["task"].get("split") != allowed[g["source"]]:
                raise ValueError("Only existing OLD_TRAIN/B_DEV compact groups are allowed")
            groups.append(g)
    seen, seen_groups, counts, coverage = set(), set(), Counter(), Counter()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "GROUP_REPLAY.jsonl").open("x") as stream:
        for g in groups:
            task, joined = g["task"], []
            identity = (g["source"], g["block"], g["group_id"])
            if identity in seen_groups:
                raise ValueError("Duplicate development group")
            seen_groups.add(identity)
            for rec in g["rows"]:
                if rec["row_id"] in seen or rec["qid"] != task["qid"]:
                    raise ValueError("Duplicate or mismatched development answer")
                seen.add(rec["row_id"])
                label = verify_semantics(task, rec, rec["score"])
                joined.append(dict(task=task, record=rec, score=rec["score"], label=label))
                coverage[(g["source"], g["family"], label["dimensions"]["S"]["status"])] += 1
                counts[g["source"]] += 1
            inputs = {
                k: g[k] for k in ("group_id", "source", "block", "family", "world_id", "step")
            }
            result = replay_group(dict(inputs, rows=joined), native_advantages)
            result.update(
                verifier_version=VERIFIER_VERSION,
                development_only=True,
                frozen_gate_eligible=False,
                cell=g.get("cell"),
                measurement_mode="PROGRAMMATIC_ONLY",
            )
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    if counts != {"C_TRAIN_PROBE": 1152, "B0": 3840, "B2_DEV": 4800}:
        raise ValueError(f"Incomplete development cache: {dict(counts)}; partial output retained")
    summary = dict(
        status="DEVELOPMENT_PROGRAMMATIC_ONLY",
        verifier_version=VERIFIER_VERSION,
        groups=len(seen_groups),
        answers=dict(counts),
        new_model_calls=0,
        confirmation_access=False,
        coverage=[
            dict(source=k[0], family=k[1], S=k[2], count=v) for k, v in sorted(coverage.items())
        ],
    )
    (output / "MEASUREMENT_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verifier", choices=("v2", "v3"), default="v3")
    args = parser.parse_args()
    run(args.evidence_root, args.output, args.verifier)


if __name__ == "__main__":
    main()
