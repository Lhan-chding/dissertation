"""Read-only diagnosis of frozen development replay; never relabel or open C.

The opportunity bound relaxes ALL P/S labels. Even an ideal parser cannot
produce SELECT/EVID differences in a group with fewer than two eligible rows.
The tighter bound additionally retains measured P and permits arbitrary S.
Neither bound is an observed signal or a human audit.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def eligible(label):
    n = label["normalization"]
    return n["j"] == 0 and all(
        n[key] is True for key in ("protocol_valid", "finished", "sufficient", "defined")
    )


def summarize(groups):
    """Aggregate existing labels without selecting on confirmation outcomes."""
    counts = Counter()
    dimensions = {key: Counter() for key in "PSOE"}
    unknown_reasons = Counter()
    reasons = Counter()
    sizes = Counter()
    opportunity = defaultdict(set)
    fixed_p_opportunity = defaultdict(set)
    eligible_groups = []
    for group in groups:
        if group["source"] not in {"B0", "B2_DEV", "C_TRAIN_PROBE", "HISTORICAL_RL"}:
            raise ValueError("Only registered development/probe/replay sources are allowed")
        counts["groups"] += 1
        counts["observed_changed_groups"] += bool(group["select_vs_evid_changed"])
        candidates = [r for r in group["labels"] if eligible(r)]
        sizes[len(candidates)] += 1
        if len(candidates) < 2:
            reasons["TOO_FEW_ELIGIBLE_FAILURES"] += 1
        elif any(r["dimensions"][k]["status"] == "UNKNOWN" for r in candidates for k in "PS"):
            reasons["ABSTAIN_UNKNOWN_IN_FAILURE_STRATUM"] += 1
        else:
            reasons["ACTIVE"] += 1
        if len(candidates) >= 2:
            opportunity[group["block"]].add(group["world_id"])
            counts["eligibility_only_group_upper_bound"] += 1
            p = [r["dimensions"]["P"]["status"] for r in candidates]
            if any(status != "FAIL" for status in p):
                fixed_p_opportunity[group["block"]].add(group["world_id"])
                counts["fixed_p_arbitrary_s_group_upper_bound"] += 1
            eligible_groups.append(
                {
                    "group_id": group["group_id"],
                    "world_id": group["world_id"],
                    "block": group["block"],
                    "eligible_rows": len(candidates),
                    "P": p,
                    "S": [r["dimensions"]["S"]["status"] for r in candidates],
                }
            )
        for row in group["labels"]:
            counts["answers"] += 1
            for key in "PSOE":
                dimensions[key][row["dimensions"][key]["status"]] += 1
            if row["dimensions"]["S"]["status"] == "UNKNOWN":
                unknown_reasons[row["dimensions"]["S"]["reason"]] += 1
            n = row["normalization"]
            stages = [
                n["j"] == 0,
                n["protocol_valid"],
                n["finished"],
                n["sufficient"] is True,
                n["defined"],
                row["dimensions"]["P"]["status"] != "UNKNOWN",
                row["dimensions"]["S"]["status"] != "UNKNOWN",
            ]
            names = [
                "j0",
                "protocol_valid",
                "finished",
                "sufficient",
                "defined",
                "p_known",
                "s_known",
            ]
            for index, name in enumerate(names):
                counts["funnel_" + name] += all(stages[: index + 1])
            if all(stages[:3]) and not stages[3]:
                counts["removed_by_sufficiency"] += 1
                counts["removed_by_sufficiency_answer_correct"] += row["native_score"][
                    "value_correct"
                ]
    return {
        "counts": dict(counts),
        "dimensions": {k: dict(v) for k, v in dimensions.items()},
        "s_unknown_reasons": dict(unknown_reasons),
        "group_gate_reasons": dict(reasons),
        "eligible_group_size_distribution": dict(sorted(sizes.items())),
        "eligibility_only_world_upper_bound": len(set().union(*opportunity.values())),
        "fixed_p_arbitrary_s_world_upper_bound": len(set().union(*fixed_p_opportunity.values())),
        "eligible_groups": eligible_groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.replay.suffix == ".gz" else open
    by_source = defaultdict(list)
    with opener(args.replay, "rt", encoding="utf-8") as handle:
        for line in handle:
            group = json.loads(line)
            by_source[group["source"]].append(group)
    probe = by_source["C_TRAIN_PROBE"]
    result = {
        "status": "DIAGNOSTIC_ONLY_NO_RELABELING_NO_HUMAN_AUDIT",
        "sources": {name: summarize(groups) for name, groups in by_source.items()},
        "c_by_family": {
            family: summarize([g for g in probe if g["family"] == family])
            for family in sorted({g["family"] for g in probe})
        },
        "c_by_block": {
            block: summarize([g for g in probe if g["block"] == block])
            for block in sorted({g["block"] for g in probe})
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
