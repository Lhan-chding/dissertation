"""Post-hoc finite-support bounds using already scored complete token actions.

These are conditional arithmetic bounds, not statistical confidence intervals.
They do not bound BF16 or scoring-system error. No model is loaded or scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from .bridge_analysis import EVENTS, Reader, aligned_scores


def support_bounds(records):
    """Deduplicate complete actions; retain the full unseen probability tail.

    Selecting the finite action set from observations does not invalidate this
    pointwise containment. Probabilities must belong to one normalized policy
    per endpoint. Any differing duplicate scores invalidate that premise here.
    """
    actions = {}
    for row in records:
        tokens = tuple(row["token_ids"])
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Complete nonnegative token action required")
        if row["max_new_tokens"] != 64 or len(tokens) > 64:
            raise ValueError("Changed finite action horizon")
        if not row["eos_seen"] and len(tokens) != 64:
            raise ValueError("Action ends before EOS or truncation")
        if row["category"] not in EVENTS:
            raise ValueError("Unknown event category")
        logs = [float(row["left_logp"]), float(row["right_logp"])]
        if not all(math.isfinite(v) and v <= 0 for v in logs):
            raise ValueError("Finite normalized log probabilities required")
        item = actions.setdefault(
            tokens,
            {
                "category": row["category"],
                "eos_seen": bool(row["eos_seen"]),
                "logs": [[], []],
            },
        )
        if item["category"] != row["category"] or item["eos_seen"] != bool(row["eos_seen"]):
            raise ValueError("Same complete action has conflicting event or termination")
        for endpoint, value in enumerate(logs):
            item["logs"][endpoint].append(value)
    if not actions:
        raise ValueError("No scored actions")
    spans = [max(max(a["logs"][e]) - min(a["logs"][e]) for a in actions.values()) for e in range(2)]
    mass = [
        [
            math.fsum(math.exp(a["logs"][e][0]) for a in actions.values() if a["category"] == event)
            for event in EVENTS
        ]
        for e in range(2)
    ]
    totals = [math.fsum(v) for v in mass]
    tails = [1 - value for value in totals]
    reasons = []
    if any(span > 0 for span in spans):
        reasons.append("DUPLICATE_ACTION_SCORE_DISAGREEMENT")
    if any(total > 1 for total in totals):
        reasons.append("OBSERVED_MASS_EXCEEDS_ONE")
    valid = not reasons
    bounds = [
        {
            "event": event,
            "seen_mass_left": mass[0][i],
            "seen_mass_right": mass[1][i],
            "known_mass_difference": mass[0][i] - mass[1][i],
            "lower": mass[0][i] - mass[1][i] - tails[1] if valid else None,
            "upper": mass[0][i] - mass[1][i] + tails[0] if valid else None,
        }
        for i, event in enumerate(EVENTS)
    ]
    return {
        "valid_conditional_bound": valid,
        "invalid_reasons": reasons,
        "conditional_on_recorded_numerical_scores": True,
        "numerical_systematic_error_bounded": False,
        "statistical_confidence_interval": False,
        "scored_rows": len(records),
        "unique_complete_actions": len(actions),
        "duplicate_rows_removed": len(records) - len(actions),
        "duplicate_max_sequence_logp_span_left": spans[0],
        "duplicate_max_sequence_logp_span_right": spans[1],
        "seen_mass_left": totals[0],
        "seen_mass_right": totals[1],
        "unseen_mass_left": tails[0],
        "unseen_mass_right": tails[1],
        "events": bounds,
    }


def _read(path):
    return json.loads(Path(path).read_text())


def analyze(campaign, out):
    campaign, out = Path(campaign).resolve(), Path(out).resolve()
    if out == campaign or out.is_relative_to(campaign / "tasks"):
        raise ValueError("Cannot write analysis over measured tasks")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    reader = Reader(campaign)
    result = {
        "scope": "POSTHOC_COMPLETE_TOKEN_SUPPORT_DIAGNOSTIC",
        "conditional_on_recorded_numerical_scores": True,
        "no_training": True,
        "no_new_scores": True,
        "new_generated_samples": 0,
        "model_calls": 0,
        "scientific_status": "NOT_CERTIFIED",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "origins": {},
        "limitations": [
            "Bounds assume recorded sequence probabilities define exact normalized policies.",
            "BF16 and systematic numerical scoring errors have not been bounded.",
            "Duplicate score disagreement or mass above one invalidates bounds; no clipping.",
            "Float arithmetic is diagnostic, not outward-rounded interval arithmetic.",
            "Adaptive action-set selection is allowed; no training-seed inference follows.",
            "Actions retain EOS or 64-token truncation; decoded text is never deduplicated.",
            "No prediction model accuracy or online SSVC validity is established.",
        ],
    }
    for origin in ("B_origin_0", "B_origin_1"):
        complete = _read(campaign / "tasks" / origin / "COMPLETE.json")
        response = _read(complete["response"]["path"])
        forks = _read(complete["forks"]["path"])
        (check,) = response["pair_checks"]
        left, right = check["left_candidate_id"], check["right_candidate_id"]
        fingerprints = [response["policies"][p]["inference_fingerprint"] for p in (left, right)]
        origin_fp = forks["origin_policy"]["inference_fingerprint"]
        by_prompt, packet_counts = defaultdict(list), {}
        for role in ("work", "reference", "direct_work", "mixture"):
            receipt = check["mixture"] if role == "mixture" else response[role]
            rows = reader.rows(receipt)
            expected_counts = {pid: 64 for pid in response["work"]["identity"]["prompt_ids"]}
            if dict(Counter(r["prompt_id"] for r in rows)) != expected_counts:
                raise ValueError("Expected complete 64-draw bridge packet per prompt")
            packet_counts[role] = len(rows)
            mapping = (
                check["mixture_scores"]
                if role == "mixture"
                else response[
                    {
                        "work": "work_scores",
                        "reference": "reference_scores",
                        "direct_work": "direct_scores",
                    }[role]
                ]
            )
            scores = []
            for index, candidate in enumerate((left, right)):
                score_receipt = mapping[candidate]
                if role != "mixture":
                    score_receipt = score_receipt["score_receipt"]
                if (
                    score_receipt["identity"]["policy"]["inference_fingerprint"]
                    != fingerprints[index]
                ):
                    raise ValueError("Candidate score binding changed")
                scores.append(aligned_scores(rows, score_receipt, reader))
            for index, row in enumerate(rows):
                expected = fingerprints if role == "mixture" else [origin_fp]
                if row["proposal_fingerprint"] not in expected:
                    raise ValueError("Proposal policy changed")
                if (
                    row["probability_execution"] != "uncached_prefix_recompute"
                    or not row["generation_parity"]["passed"]
                ):
                    raise ValueError("Unverified generation probability path")
                by_prompt[row["prompt_id"]].append(
                    {
                        **row,
                        "left_logp": scores[0][index],
                        "right_logp": scores[1][index],
                    }
                )
        units = {}
        for pid, rows in by_prompt.items():
            if len({r["input_hash"] for r in rows}) != 1:
                raise ValueError("One prompt has different prepared inputs")
            units[pid] = support_bounds(rows)
        result["origins"][origin] = {
            "bank_id": check["bank_id"],
            "contrast_id": check["contrast_id"],
            "left_candidate": left,
            "right_candidate": right,
            "endpoint_fingerprints": fingerprints,
            "packet_rows": packet_counts,
            "prompt_count": len(units),
            "units": units,
            "all_conditional_bounds_valid": all(
                v["valid_conditional_bound"] for v in units.values()
            ),
            "complete_receipt_path": str(campaign / "tasks" / origin / "COMPLETE.json"),
        }
        reader.cache.clear()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, value in (("SUPPORT_BOUNDS.json", result), ("INPUT_INDEX.json", reader.manifest)):
        (out / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    (out / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "POSTHOC_SUPPORT_DIAGNOSTIC_COMPLETE_NOT_CERTIFIED",
                "no_training": True,
                "no_new_scores": True,
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in out.iterdir()
                    if p.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        parser.error("Actual bridge analysis requires a server CPU Slurm allocation")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        parser.error("Explicitly hide CUDA for CPU-only analysis")
    analyze(args.campaign, args.out)


if __name__ == "__main__":
    main()
