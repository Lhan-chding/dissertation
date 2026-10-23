"""Offline eligibility ablations for C redesign; never opens or changes a gate.

Consumes existing development replay labels. Four policies separate removing
sufficiency from allowing a known subset. No model calls, relabeling, schedule
extrapolation, or production trainer imports. Frozen F uses reference credit.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from srf4_ref.credit import Observation, semantic_residual

from .replay import SHUFFLE_SEEDS
from .training import native_advantages

SOURCES = {"B0", "B2_DEV", "C_TRAIN_PROBE", "HISTORICAL_RL"}
ARMS = ("J", "EVID", "SELECT", "SELECT_SHUF")
BETA = 0.25


@dataclass(frozen=True)
class Policy:
    name: str
    require_sufficient: bool
    known_subset: bool


POLICIES = (
    Policy("F_FROZEN", True, False),
    Policy("F_KNOWN_75", True, True),
    Policy("F_PRIME_STRICT", False, False),
    Policy("F_PRIME_KNOWN_75", False, True),
)


def eligibility(rows, policy):
    candidates = [
        i
        for i, r in enumerate(rows)
        if r.j == 0
        and r.protocol_valid
        and r.finished
        and r.defined
        and (not policy.require_sufficient or r.sufficient is True)
    ]
    known = [
        i for i in candidates if rows[i].perception != "UNKNOWN" and rows[i].selection != "UNKNOWN"
    ]
    if len(candidates) < 2:
        reason = "TOO_FEW_ELIGIBLE_FAILURES"
    elif not policy.known_subset and len(known) != len(candidates):
        reason = "ABSTAIN_UNKNOWN_IN_FAILURE_STRATUM"
    elif len(known) < 2:
        reason = "TOO_FEW_KNOWN_FAILURES"
    elif policy.known_subset and 4 * len(known) < 3 * len(candidates):
        reason = "KNOWN_COVERAGE_BELOW_75_PERCENT"
    else:
        reason = "ACTIVE"
    return candidates, known, reason


def proposed_residual(rows, arm, policy, *, shuffle_seed=0):
    """Return beta*d, with shared eligibility for every arm on these same rows."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    if len({r.row_id for r in rows}) != len(rows):
        raise ValueError("Duplicate row IDs")
    candidates, known, reason = eligibility(rows, policy)
    residual = [0.0] * len(rows)
    if reason != "ACTIVE" or arm == "J":
        return residual
    ids = known if policy.known_subset else candidates
    p = {i: float(rows[i].perception == "PASS") for i in ids}
    s = {i: float(rows[i].selection == "PASS") for i in ids}
    if arm == "SELECT_SHUF":
        rng = random.Random(shuffle_seed)
        for pv in (0.0, 1.0):
            stratum = [i for i in ids if p[i] == pv]
            values = [s[i] for i in stratum]
            rng.shuffle(values)
            s.update(zip(stratum, values, strict=True))
    scores = {i: 0.5 * p[i] * (1 + s[i] if arm != "EVID" else 1) for i in ids}
    mean = sum(scores.values()) / len(ids)
    for i in ids:
        residual[i] = BETA * (scores[i] - mean)
    return residual


def shuffle_change_probability(n, k):
    """Exact uniform-permutation probability in the P=PASS binary-S stratum."""
    if not isinstance(n, int) or not isinstance(k, int) or not 0 <= k <= n:
        raise ValueError("Require integer 0 <= k <= n")
    return 1 - 1 / math.comb(n, k)


def _observations(group):
    for key in ("source", "group_id", "world_id", "block", "family", "verifier_version"):
        if not isinstance(group.get(key), str) or not group[key].strip():
            raise ValueError(f"Missing or invalid group identity: {key}")
    if group["source"] not in SOURCES:
        raise ValueError("Only registered development sources may enter this diagnostic")
    labels = group["labels"]
    if not labels:
        raise ValueError("Empty group")
    rows = []
    for label in labels:
        if label.get("verifier_version") != group["verifier_version"]:
            raise ValueError("Label and group verifier versions disagree")
        n = label["normalization"]
        if any(type(n[key]) is not bool for key in ("protocol_valid", "finished", "defined")):
            raise ValueError("Eligibility flags must be booleans")
        if n["sufficient"] is not None and type(n["sufficient"]) is not bool:
            raise ValueError("Sufficiency must be boolean or unknown")
        rows.append(
            Observation(
                label["row_id"],
                n["j"],
                n["protocol_valid"],
                n["finished"],
                n["sufficient"],
                n["defined"],
                label["dimensions"]["P"]["status"],
                label["dimensions"]["S"]["status"],
            )
        )
    if len({r.row_id for r in rows}) != len(rows):
        raise ValueError("Duplicate answer identity")
    base = native_advantages([r.j for r in rows])
    stored = group.get("A_J")
    if stored is not None and (
        len(stored) != len(base)
        or any(
            not math.isfinite(a) or abs(a - b) > 1e-12 for a, b in zip(stored, base, strict=True)
        )
    ):
        raise ValueError("Stored advantages disagree with the native J calculation")
    return rows, base


def _l1(values):
    return sum(abs(v) for v in values)


def diagnose_group(group):
    rows, base = _observations(group)
    result = {
        key: group.get(key)
        for key in (
            "source",
            "group_id",
            "world_id",
            "block",
            "family",
            "cell",
            "step",
            "verifier_version",
        )
    }
    result.update(row_ids=[r.row_id for r in rows], answers=len(rows), A_J=base, policies={})
    for policy in POLICIES:
        candidates, known, reason = eligibility(rows, policy)
        active = reason == "ACTIVE"
        p_pass = [i for i in known if rows[i].perception == "PASS"]
        n = len(p_pass)
        k = sum(rows[i].selection == "PASS" for i in p_pass)
        signal = active and 0 < k < n
        residuals = {}
        for arm in ("J", "EVID", "SELECT"):
            # Use the unchanged reference implementation for the frozen policy.
            residuals[arm] = (
                semantic_residual(rows, arm)[0]
                if policy.name == "F_FROZEN"
                else proposed_residual(rows, arm, policy)
            )
        shuffled = [
            (
                semantic_residual(rows, "SELECT_SHUF", shuffle_seed=seed)[0]
                if policy.name == "F_FROZEN"
                else proposed_residual(rows, "SELECT_SHUF", policy, shuffle_seed=seed)
            )
            for seed in SHUFFLE_SEEDS
        ]
        increment = [a - b for a, b in zip(residuals["SELECT"], residuals["EVID"], strict=True)]
        differences = [
            [a - b for a, b in zip(residuals["SELECT"], s, strict=True)] for s in shuffled
        ]
        changed = sum(any(abs(v) > 1e-8 for v in d) for d in differences)
        positive = [i for i, v in enumerate(residuals["SELECT"]) if v > 1e-12]
        # These descriptors overlap; they are not exclusive causal error classes.
        components = dict(
            rows=len(positive),
            evidence_incomplete=sum(rows[i].sufficient is False for i in positive),
            value_correct=sum(
                group["labels"][i]["native_score"]["value_correct"] == 1 for i in positive
            ),
            execution_fail=sum(
                group["labels"][i]["dimensions"]["E"]["status"] == "FAIL" for i in positive
            ),
            execution_unknown=sum(
                group["labels"][i]["dimensions"]["E"]["status"] == "UNKNOWN" for i in positive
            ),
            selection_fail=sum(rows[i].selection == "FAIL" for i in positive),
        )
        numerators = dict(
            evidence=_l1(residuals["EVID"]),
            select_total=_l1(residuals["SELECT"]),
            select_minus_evid=_l1(increment),
            select_minus_shuffle_mean=sum(_l1(d) for d in differences) / len(differences),
            select_minus_shuffle_exact=BETA * k * (n - k) / n if signal else 0.0,
        )
        result["policies"][policy.name] = dict(
            reason=reason,
            active=active,
            signal=signal,
            candidate_indices=candidates,
            known_indices=known,
            known_coverage=len(known) / len(candidates) if candidates else None,
            eligibility_only_opportunity=len(candidates) >= 2,
            fixed_p_arbitrary_s_opportunity=sum(rows[i].perception == "PASS" for i in candidates)
            >= 2,
            p_pass_count=n,
            s_pass_within_p_pass=k,
            select_minus_evid_changed=any(abs(v) > 1e-8 for v in increment),
            expected_shuffle_change_probability=shuffle_change_probability(n, k) if active else 0,
            shuffle_changed=changed,
            shuffle_trials=len(SHUFFLE_SEEDS),
            residuals=residuals,
            shuffle_residuals=shuffled,
            positive_components_overlapping=components,
            l1_numerators=numerators,
            native_l1=_l1(base),
        )
    return result


def summarize(groups, policy_name):
    policies = [g["policies"][policy_name] for g in groups]
    signals = [g for g in groups if g["policies"][policy_name]["signal"]]
    denom = sum(p["native_l1"] for p in policies)
    numerators = Counter()
    components = Counter()
    for p in policies:
        numerators.update(p["l1_numerators"])
        components.update(p["positive_components_overlapping"])
    trials = sum(g["policies"][policy_name]["shuffle_trials"] for g in signals)
    return dict(
        groups=len(groups),
        answers=sum(g["answers"] for g in groups),
        group_size_distribution=dict(Counter(g["answers"] for g in groups)),
        candidate_rows=sum(len(p["candidate_indices"]) for p in policies),
        known_rows=sum(len(p["known_indices"]) for p in policies),
        N_act=sum(p["active"] for p in policies),
        N_sig=len(signals),
        N_sig_density=len(signals) / len(groups) if groups else None,
        W_sig=len({(g["source"], g["world_id"]) for g in signals}),
        select_minus_evid_changed_groups=sum(p["select_minus_evid_changed"] for p in policies),
        eligibility_only_group_upper_bound=sum(p["eligibility_only_opportunity"] for p in policies),
        fixed_measured_p_arbitrary_s_group_upper_bound=sum(
            p["fixed_p_arbitrary_s_opportunity"] for p in policies
        ),
        reasons=dict(Counter(p["reason"] for p in policies)),
        positive_components_overlapping=dict(components),
        native_l1=denom,
        l1_numerators=dict(numerators),
        R={key: value / denom if denom else None for key, value in numerators.items()},
        zero_native_l1_groups=sum(p["native_l1"] == 0 for p in policies),
        zero_native_l1_select_signal=sum(
            p["l1_numerators"]["select_total"] for p in policies if p["native_l1"] == 0
        ),
        shuffle_changed_in_signal_groups=sum(
            g["policies"][policy_name]["shuffle_changed"] for g in signals
        ),
        shuffle_trials_in_signal_groups=trials,
        expected_shuffle_change_fraction=(
            sum(g["policies"][policy_name]["expected_shuffle_change_probability"] for g in signals)
            / len(signals)
            if signals
            else None
        ),
    )


def run(paths, output):
    results, seen_groups, seen_rows = [], set(), set()
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                result = diagnose_group(json.loads(line))
                identity = (result["source"], result["block"], result["group_id"])
                if identity in seen_groups or seen_rows.intersection(result["row_ids"]):
                    raise ValueError(
                        "Duplicate group/answer across inputs; compare versions separately"
                    )
                seen_groups.add(identity)
                seen_rows.update(result["row_ids"])
                results.append(result)
    if not results:
        raise ValueError("No development groups supplied")
    if len({g["verifier_version"] for g in results}) != 1:
        raise ValueError("Compare different verifier versions in separate runs")
    strata = defaultdict(list)
    for g in results:
        strata[(g["source"], None, None, None)].append(g)
        strata[(g["source"], g["family"], None, None)].append(g)
        strata[(g["source"], g["family"], g["block"], g["cell"])].append(g)
    summary = dict(
        status="OFFLINE_DESIGN_DIAGNOSTIC_NOT_TRAINING_AUTHORIZATION",
        beta=BETA,
        known_subset_coverage=0.75,
        shuffle_seeds=list(SHUFFLE_SEEDS),
        verifier_versions=sorted({g["verifier_version"] for g in results}),
        limitations=[
            "Existing fixed responses only; labels are not repaired by eligibility changes.",
            "Unknown cell stays null; sources stay separate and group size mixtures are explicit.",
            "R is descriptive L1 mass, not gradient strength, power, or a path forecast.",
            "R with zero denominator is null; zero-baseline added signal is separately retained.",
            "Positive descriptors overlap. E=FAIL is not necessarily an arithmetic mistake.",
            "Opportunity bounds relax S labels and coverage, not observed usable groups.",
            "The same ten historical replay seeds are used; no production seed policy is changed.",
        ],
        strata=[
            dict(
                source=key[0],
                family=key[1],
                block=key[2],
                cell=key[3],
                policies={p.name: summarize(groups, p.name) for p in POLICIES},
            )
            for key, groups in strata.items()
        ],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "GROUP_DIAGNOSTICS.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in results)
    )
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.replay, args.output)


if __name__ == "__main__":
    main()
