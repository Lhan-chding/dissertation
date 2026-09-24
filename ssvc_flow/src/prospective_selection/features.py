"""Information-boundary packets. Raw completions never enter selector objects."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .semantics import reward_bin, reward_vector, validate_repair

LEVELS = (
    "Z0_MOMENTS_GLOBAL",
    "Z1_MOMENTS_STRATIFIED",
    "Z2_REWARD_DISTRIBUTION",
    "Z3_REPAIR_STRUCTURE",
)
SCHEMA = "prospective-features-v1"
METADATA_FIELDS = frozenset(
    (
        "gradient_norm_mean",
        "gradient_norm_std",
        "effective_group_fraction",
        "zero_advantage_group_fraction",
        "loss_mean",
    )
)
FORBIDDEN = frozenset(
    (
        "outcomes",
        "outcome",
        "outcomestore",
        "outcome_record",
        "future",
        "test_reward",
        "post_update_parameters",
        "endpoint",
        "run_path",
        "path",
        "H8",
        "H32",
        "horizon",
        "continuation_repeat",
        "recipe_final_score",
        "test_return",
    )
)


def reject_future_fields(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            s = str(key).lower()
            if s in {k.lower() for k in FORBIDDEN} or s.startswith(
                ("future_", "outcome_", "post_update_")
            ):
                raise ValueError(f"Forbidden future/outcome field: {key}")
            reject_future_fields(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            reject_future_fields(value)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _moments(rows):
    if not rows:
        return None
    a = np.asarray([reward_vector(row) for row in rows], dtype=float)
    mean = a.mean(axis=0)
    second = a.T @ a / len(a)
    return [
        *mean.tolist(),
        *second.ravel().tolist(),
        *(second - np.outer(mean, mean)).ravel().tolist(),
    ]


def _hist(counts):
    return {"n": sum(counts.values()), "counts": dict(sorted(counts.items()))}


def _snapshot(rows, level):
    index = LEVELS.index(level)
    result = {"n": len(rows), "moments_global": _moments(rows)}
    grouped = defaultdict(list)
    prompts = defaultdict(list)
    for row in rows:
        reward_vector(row)
        if not all(
            isinstance(row[k], str) and row[k] for k in ("prompt_id", "family", "interface")
        ):
            raise ValueError("Prompt ID and fixed stratum labels required")
        grouped[f"{row['family']}|{row['interface']}"].append(row)
        prompts[row["prompt_id"]].append(row)
    if index >= 1:
        result["moments_stratified"] = {
            key: _moments(value) for key, value in sorted(grouped.items())
        }
    if index >= 2:
        result["reward_histograms"] = {}
        result["prompt_strata"] = {}
        for prompt, values in sorted(prompts.items()):
            strata = {f"{r['family']}|{r['interface']}" for r in values}
            if len(strata) != 1:
                raise ValueError("Prompt changes stratum")
            result["prompt_strata"][prompt] = strata.pop()
            result["reward_histograms"][prompt] = _hist(
                Counter(
                    reward_bin(r["event"], r["relation_numerator"], r["relation_denominator"])
                    for r in values
                )
            )
    if index >= 3:
        result["repair_histograms"] = {}
        result["repair_marginals"] = {}
        for prompt, values in sorted(prompts.items()):
            joint, marginals = Counter(), {k: Counter() for k in ("F", "B", "M")}
            for row in values:
                validate_repair(row)
                invalid = row["event"] == "I"
                rbin = reward_bin(
                    row["event"], row["relation_numerator"], row["relation_denominator"]
                )
                atom = "INVALID" if invalid else f"{rbin}|{row['F']}|{row['B']}|{row['M']}"
                joint[atom] += 1
                for key, counts in marginals.items():
                    counts["INVALID" if invalid else str(row[key])] += 1
            result["repair_histograms"][prompt] = _hist(joint)
            result["repair_marginals"][prompt] = {k: _hist(c) for k, c in marginals.items()}
    return result


def _validate_hist(hist):
    if not isinstance(hist, dict) or set(hist) != {"n", "counts"}:
        raise ValueError("Histogram schema mismatch")
    if type(hist["n"]) is not int or hist["n"] <= 0 or not isinstance(hist["counts"], dict):
        raise ValueError("Positive histogram sample count required")
    if not hist["counts"] or any(
        not isinstance(k, str) or type(v) is not int or v <= 0 for k, v in hist["counts"].items()
    ):
        raise ValueError("Sparse histograms store positive observed counts")
    if sum(hist["counts"].values()) != hist["n"]:
        raise ValueError("Histogram count mismatch")


def _reward_atom(atom):
    match = re.fullmatch(r"([XSWI]):(0|[1-9][0-9]*)/([1-9][0-9]*)", atom)
    if match is None:
        raise ValueError("Unregistered reward atom; raw information is forbidden")
    event, numerator, denominator = match.groups()
    row = dict(
        event=event, relation_numerator=int(numerator), relation_denominator=int(denominator)
    )
    if reward_bin(event, int(numerator), int(denominator)) != atom:
        raise ValueError("Reward atoms must use reduced exact rational keys")
    return row


def _histogram_moments(histograms):
    total = sum(h["n"] for h in histograms)
    if not total:
        return None
    mean, second = np.zeros(4), np.zeros((4, 4))
    for hist in histograms:
        for atom, count in hist["counts"].items():
            vector = np.asarray(reward_vector(_reward_atom(atom)))
            mean += count * vector / total
            second += count * np.outer(vector, vector) / total
    return [*mean, *second.ravel(), *(second - np.outer(mean, mean)).ravel()]


def _validate_snapshot(snapshot, level):
    required = {"n", "moments_global"}
    i = LEVELS.index(level)
    if i >= 1:
        required.add("moments_stratified")
    if i >= 2:
        required.update(("reward_histograms", "prompt_strata"))
    if i >= 3:
        required.update(("repair_histograms", "repair_marginals"))
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ValueError("Feature layer permissions/schema mismatch")
    if type(snapshot["n"]) is not int or snapshot["n"] < 0:
        raise ValueError("Nonnegative sample count required")
    moments = [snapshot["moments_global"]]
    if i >= 1:
        moments.extend(snapshot["moments_stratified"].values())
    for vec in moments:
        if vec is None:
            if snapshot["n"]:
                raise ValueError("Missing observed moments")
        elif not isinstance(vec, list) or len(vec) != 36 or not all(_finite(v) for v in vec):
            raise ValueError("Invalid moment vector")
    if i >= 2:
        hist = snapshot["reward_histograms"]
        if set(hist) != set(snapshot["prompt_strata"]):
            raise ValueError("Fixed prompt alignment missing")
        for h in hist.values():
            _validate_hist(h)
            for atom in h["counts"]:
                _reward_atom(atom)
        if sum(h["n"] for h in hist.values()) != snapshot["n"]:
            raise ValueError("Prompt total mismatch")
        if snapshot["n"] and not np.allclose(
            _histogram_moments(list(hist.values())), snapshot["moments_global"], atol=1e-12
        ):
            raise ValueError("Reward histograms do not match global moments")
        strata = snapshot["prompt_strata"]
        if set(strata.values()) != set(snapshot["moments_stratified"]):
            raise ValueError("Reward histograms do not match moment strata")
        for stratum, vector in snapshot["moments_stratified"].items():
            expected = _histogram_moments([h for p, h in hist.items() if strata[p] == stratum])
            if not np.allclose(expected, vector, atol=1e-12):
                raise ValueError("Reward histograms do not match stratified moments")
    if i >= 3:
        for key in ("repair_histograms", "repair_marginals"):
            if set(snapshot[key]) != set(snapshot["reward_histograms"]):
                raise ValueError("Repair and reward samples differ")
        for prompt, h in snapshot["repair_histograms"].items():
            _validate_hist(h)
            if h["n"] != snapshot["reward_histograms"][prompt]["n"]:
                raise ValueError("Repair and reward sample sizes differ")
            if set(snapshot["repair_marginals"][prompt]) != {"F", "B", "M"}:
                raise ValueError("Repair marginal schema mismatch")
            for marginal in snapshot["repair_marginals"][prompt].values():
                _validate_hist(marginal)
                if marginal["n"] != h["n"]:
                    raise ValueError("Repair marginal sample mismatch")
            projected_reward = Counter()
            projected_marginals = {k: Counter() for k in ("F", "B", "M")}
            for atom, count in h["counts"].items():
                if atom == "INVALID":
                    projected_reward["I:0/1"] += count
                    for counts in projected_marginals.values():
                        counts["INVALID"] += count
                    continue
                parts = atom.split("|")
                if len(parts) != 4 or any(not re.fullmatch(r"[0-4]", x) for x in parts[1:]):
                    raise ValueError("Unregistered repair atom")
                row = _reward_atom(parts[0])
                row.update(dict(zip(("F", "B", "M"), map(int, parts[1:]), strict=True)))
                validate_repair(row)
                projected_reward[parts[0]] += count
                for key in projected_marginals:
                    projected_marginals[key][str(row[key])] += count
            if dict(projected_reward) != snapshot["reward_histograms"][prompt]["counts"]:
                raise ValueError("Repair joint does not project to the identical reward samples")
            for key, counts in projected_marginals.items():
                if dict(counts) != snapshot["repair_marginals"][prompt][key]["counts"]:
                    raise ValueError("Repair marginals do not project from the joint")


@dataclass(frozen=True)
class PreDecisionPacket:
    """Immutable JSON value; lineage is audit metadata and never a model input."""

    payload_json: str

    def __post_init__(self):
        d = json.loads(self.payload_json)
        reject_future_fields(d)
        required = {
            "schema",
            "origin_id",
            "lineage_id",
            "source_recipe",
            "step",
            "history_step",
            "feature_level",
            "current",
            "history",
            "known_training_metadata",
        }
        if set(d) != required or d["schema"] != SCHEMA or d["feature_level"] not in LEVELS:
            raise ValueError("PreDecisionPacket schema mismatch")
        if not all(isinstance(d[k], str) and d[k] for k in ("origin_id", "lineage_id")):
            raise ValueError("Audit identifiers required")
        if d["source_recipe"] not in ("R0", "R1") or type(d["step"]) is not int or d["step"] < 8:
            raise ValueError("Registered source and checkpoint required")
        if type(d["history_step"]) is not int or d["history_step"] != d["step"] - 8:
            raise ValueError("History must precede current state by eight steps")
        meta = d["known_training_metadata"]
        if (
            not isinstance(meta, dict)
            or set(meta) - METADATA_FIELDS
            or not all(_finite(v) for v in meta.values())
        ):
            raise ValueError("Unregistered training metadata")
        for key in ("current", "history"):
            _validate_snapshot(d[key], d["feature_level"])
        if (
            LEVELS.index(d["feature_level"]) >= 2
            and d["current"]["n"]
            and d["history"]["n"]
            and d["current"]["prompt_strata"] != d["history"]["prompt_strata"]
        ):
            raise ValueError("History/current prompt panel must be identical")

    @classmethod
    def from_dict(cls, data):
        return cls(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_dict(self):
        return json.loads(self.payload_json)

    @property
    def feature_level(self):
        return self.to_dict()["feature_level"]

    @property
    def origin_id(self):
        return self.to_dict()["origin_id"]

    @property
    def lineage_id(self):
        return self.to_dict()["lineage_id"]

    @property
    def valid_observations(self):
        return self.to_dict()["current"]["n"] > 0


def build_predecision_packet(
    *,
    origin_id,
    lineage_id,
    source_recipe,
    step,
    current_rows,
    history_rows,
    feature_level,
    known_training_metadata=None,
    history_step=None,
):
    if feature_level not in LEVELS:
        raise ValueError("Unknown feature level")
    # The ingestion boundary can see raw labels; the returned packet cannot.
    current_rows, history_rows = list(current_rows), list(history_rows)
    reject_future_fields(current_rows)
    reject_future_fields(history_rows)
    return PreDecisionPacket.from_dict(
        dict(
            schema=SCHEMA,
            origin_id=origin_id,
            lineage_id=lineage_id,
            source_recipe=source_recipe,
            step=step,
            history_step=step - 8 if history_step is None else history_step,
            feature_level=feature_level,
            current=_snapshot(current_rows, feature_level),
            history=_snapshot(history_rows, feature_level),
            known_training_metadata=known_training_metadata or {},
        )
    )
