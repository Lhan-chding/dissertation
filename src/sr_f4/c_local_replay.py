"""Fixed old-training replay: legacy-label cohort counterfactual + strict abstention.

Never creates a new declaration, fills UNKNOWN, reads endpoints, or calls a model.
Legacy-label arithmetic is explicitly distinct from new-protocol measurements.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from srf4_ref.sampling import seed_for

from .c_local_credit import VERIFIER_VERSION, measure
from .common import rows, write


def replay(source):
    totals = Counter()
    reasons, strict = Counter(), Counter()
    seen = set()
    signal_qids, signal_roots, conditional_roots = set(), set(), set()
    by_block = {str(b): Counter() for b in range(3)}
    for group in rows(source):
        b, step, gi = (group[k] for k in ("block", "step", "group_index"))
        key = (b, step, gi)
        task, records, obs = group["task"], group["records"], group["observations"]
        if (group["source"] != "COMPLETED_FPRIME_SELECT_TRAIN_ONLY" or key in seen
                or type(b) is not int or b not in range(3)
                or type(step) is not int or step not in range(1, 17)
                or type(gi) is not int or gi not in range(16)
                or task["split"] != "C_FP_TRAIN_R1" or len(records) != 8 or len(obs) != 8):
            raise ValueError("Replay requires the complete fixed old SELECT training corpus")
        seen.add(key)
        group_id = f"CPF-T-b{b}-SELECT/s{step:03d}/g{gi:05d}"
        for draw, (r, o) in enumerate(zip(records, obs, strict=True)):
            expected = dict(block=b, step=step, arm="SELECT", group_id=group_id,
                            row_id=f"{group_id}/d{draw:02d}", draw=draw, qid=task["qid"])
            if any(r.get(k) != v for k, v in expected.items()) or (
                r["row_id"] != o["row_id"] or r["score"]["J"] != o["j"]
            ):
                raise ValueError("Replay raw-record/legacy-observation identity mismatch")
            label = measure(task, r)
            strict[label["S"]] += 1
        candidates = [i for i, o in enumerate(obs)
                      if o["protocol_valid"] and o["finished"] and o["defined"]]
        known = [i for i in candidates if obs[i]["perception"] != "UNKNOWN"
                 and obs[i]["selection"] != "UNKNOWN"]
        reason = ("CANDIDATE_LT2" if len(candidates) < 2 else "KNOWN_LT2" if len(known) < 2
                  else "COVERAGE_LT75" if len(known) * 4 < len(candidates) * 3 else "ACTIVE")
        reasons[reason] += 1
        values = Counter(groups=1, answers=8, candidate_rows=len(candidates), known_rows=len(known),
                         old_active_groups=int(group["old_diagnostic"]["active"]),
                         old_selection_signal_groups=int(group["old_diagnostic"]["selection_signal"]))
        if reason == "ACTIVE":
            values["expanded_active_groups"] += 1
            signal = {obs[i]["selection"] for i in known if obs[i]["perception"] == "PASS"}
            values["expanded_selection_signal_groups"] += int(signal == {"PASS", "FAIL"})
            if signal == {"PASS", "FAIL"}:
                signal_qids.add(task["qid"])
                signal_roots.add(task["root_id"])
            rng = random.Random(seed_for(2026102401, "offline-legacy-shuffle", b, step, gi))
            changed, swappable = 0, False
            for j in (0, 1):
                for p in ("FAIL", "PASS"):
                    indices = [i for i in known if obs[i]["j"] == j and obs[i]["perception"] == p]
                    source_indices = list(indices)
                    rng.shuffle(source_indices)
                    ss = {obs[i]["selection"] for i in indices}
                    if p == "PASS" and ss == {"PASS", "FAIL"}:
                        swappable = True
                    if p == "PASS":
                        changed += sum(obs[i]["selection"] != obs[k]["selection"]
                                       for i, k in zip(indices, source_indices, strict=True))
            values["joint_J_P_swappable_groups"] += int(swappable)
            values["joint_J_P_changed_groups"] += int(changed > 0)
            values["joint_J_P_changed_rows"] += changed
            if swappable:
                conditional_roots.add(task["root_id"])
        totals.update(values)
        by_block[str(b)].update(values)
    expected = {(b, s, g) for b in range(3) for s in range(1, 17) for g in range(16)}
    if seen != expected or totals["answers"] != 6144:
        raise ValueError("Incomplete replay corpus; do not report a complete diagnostic")
    return dict(
        status="COMPLETE_OFFLINE_DIAGNOSTIC_NOT_SCIENTIFIC_GATE",
        source=str(Path(source)), corpus="old_Fprime_all_three_SELECT_training_paths_16_steps",
        counts=dict(totals), by_block={k: dict(v) for k, v in by_block.items()},
        expanded_eligibility_reasons=dict(reasons),
        expanded_signal_unique_questions=len(signal_qids),
        expanded_signal_unique_roots=len(signal_roots),
        conditional_swappable_unique_roots=len(conditional_roots),
        legacy_measurement="frozen_r6_labels_only_for_cohort_arithmetic",
        strict_new_measurement=dict(version=VERIFIER_VERSION, selection=dict(strict)),
        limitations=[
            "No SELECT/COMPUTE statements fabricated; old unsupported responses remain UNKNOWN.",
            "Old-label replay is not an evaluation of the new explicit declaration parser.",
            "No old endpoint, B_CONFIRM or new model generation is read or used.",
            "Signal counts do not demonstrate causal learning benefit or new-task observability.",
            "Local gradients are checked separately on CPU; old text is not relabeled as new.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, replay(args.input))


if __name__ == "__main__":
    main()
