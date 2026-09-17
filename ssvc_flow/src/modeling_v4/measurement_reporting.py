"""Numerical and interval reporting for finite measurements; no model calls."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def stable_joint_covariance(values):
    """Sample covariance of whole draws, flattened in existing contrast order.

    Subtracting a represented row before centering makes identical floating
    draws exactly constant. Off-diagonal event/contrast covariance is retained.
    This is sample covariance, not covariance of a mean; callers divide by the
    number of iid draws once, or never for bootstrap replicate estimates.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim < 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite vector draws required")
    flat = values.reshape(len(values), -1)
    centered = flat - flat[0]
    centered -= centered.mean(axis=0)
    return centered.T @ centered / (len(values) - 1)


def count_difference_intervals(left, right, *, alpha=0.05):
    """Conservative 1-alpha CP interval for each independent endpoint difference.

    Each endpoint receives a two-sided 1-alpha/2 interval. The union bound
    gives at least 1-alpha coverage for one event difference under iid sampling.
    Pass .05/M for a prespecified family of M event differences.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    packets = [np.asarray(v) for v in (left, right)]
    for values in packets:
        if (
            values.ndim != 2
            or values.shape[1] != 4
            or not len(values)
            or not np.isfinite(values).all()
            or not np.isin(values, [0, 1]).all()
            or not (values.sum(axis=1) == 1).all()
        ):
            raise ValueError("Complete one-hot X/S/W/I endpoint draws required")
    intervals = []
    for event in range(4):
        bounds = [
            binomtest(int(v[:, event].sum()), len(v)).proportion_ci(
                confidence_level=1 - alpha / 2, method="exact"
            )
            for v in packets
        ]
        intervals.append(
            [max(-1, bounds[0].low - bounds[1].high), min(1, bounds[0].high - bounds[1].low)]
        )
    return np.asarray(intervals)


def interval_resolution(intervals, estimate, *, rules):
    """Apply frozen width budgets to asymmetric intervals around an estimate."""
    intervals, estimate = np.asarray(intervals), np.asarray(estimate)
    radius = np.maximum(estimate - intervals[:, 0], intervals[:, 1] - estimate)
    spec = rules["reference_precision"]
    by_scale = {
        str(scale): (radius <= spec["max_half_width_as_fraction_of_scale"] * scale).tolist()
        for scale in spec["probability_scales"]
    }
    return {
        "maximum_distance_from_estimate": radius.tolist(),
        "resolved_at_scale": by_scale,
        "reference_resolved": by_scale[str(spec["primary_resolved_scale"])],
        "primary_scale": spec["primary_resolved_scale"],
    }


def validate_diagnostic_packets(packets):
    """Validate raw packet semantics and independence before reporting means.

    Each item is (receipt, raw_rows, [(score_receipt, score_rows, fingerprint)]).
    This does not infer global support or certify numerical model accuracy.
    """
    import math

    from ..modeling_v3.io import canonical_hash
    from ..modeling_v3.vlm_observation import validate_action

    events = ("X", "S", "W", "I")
    used_keys, used_ids, used_seeds, used_streams = set(), set(), set(), set()
    inputs = {}
    for receipt, raw_rows, scored in packets:
        identity = receipt["identity"]
        expected = {
            (pid, draw) for pid in identity["prompt_ids"] for draw in range(identity["draws"])
        }
        keys, ids, seeds, draws = set(), set(), set(), set()
        stream = identity["rng_namespace"]
        if stream in used_streams:
            raise ValueError("Independent diagnostic packets reuse RNG namespace")
        if identity["proposal"] not in ("ORIGIN", "DIRECT", "MIX") or len(identity["policies"]) != (
            2 if identity["proposal"] == "MIX" else 1
        ):
            raise ValueError("Diagnostic proposal must be registered ORIGIN/DIRECT/ordered MIX")
        runtime = canonical_hash(identity["runtime"])
        for row in raw_rows:
            key, sid, seed = row["sample_key"], row["sample_id"], row["sample_seed"]
            index = (row["prompt_id"], row["draw_index"])
            expected_key = canonical_hash([stream, *index])
            chosen_index = (
                int(canonical_hash([expected_key, "mixture_source"])[0], 16) % 2
                if identity["proposal"] == "MIX"
                else 0
            )
            chosen = identity["policies"][chosen_index]
            if (
                key != expected_key
                or sid != expected_key
                or seed != int(expected_key[:16], 16) % (2**63)
                or row["origin_id"] != identity["origin_id"]
                or row["proposal"] != identity["proposal"]
                or row["proposal_policy_id"] != chosen["candidate_id"]
            ):
                raise ValueError("Diagnostic seeded draw/proposal identity differs")
            if (
                key in keys
                or sid in ids
                or seed in seeds
                or index in draws
                or key in used_keys
                or sid in used_ids
                or seed in used_seeds
            ):
                raise ValueError("Duplicate or reused diagnostic sample identity/seed/draw")
            if (
                index not in expected
                or row["rng_namespace"] != stream
                or row["role"] != identity["role"]
                or row["runtime_identity"] != runtime
                or row["proposal_fingerprint"] != chosen["inference_fingerprint"]
                or row["probability_execution"] != "uncached_prefix_recompute"
                or not row["generation_parity"]["passed"]
            ):
                raise ValueError("Diagnostic raw packet/policy/execution identity differs")
            if (
                row["category"] not in events
                or row["event_onehot"] != [int(row["category"] == e) for e in events]
                or row["shared_token_identity"] != canonical_hash(row["token_ids"])
            ):
                raise ValueError("Diagnostic event or token identity differs")
            flags = validate_action(row, eos_ids=row["eos_token_ids"], max_new_tokens=64)
            if (
                row["max_new_tokens"] != 64
                or flags["eos"] != row["eos_seen"]
                or flags["truncated"] != row["truncated"]
            ):
                raise ValueError("Diagnostic EOS/truncation contract differs")
            logs = np.asarray(row["behavior_token_logprobs"], dtype=float)
            if (
                logs.shape != (len(row["token_ids"]),)
                or not np.isfinite(logs).all()
                or np.any(logs > 0)
                or not math.isclose(
                    math.fsum(logs), row["generation_sequence_logp"], rel_tol=1e-12, abs_tol=1e-10
                )
            ):
                raise ValueError("Diagnostic generation probability sum differs")
            prior_input = inputs.setdefault(row["prompt_id"], row["input_hash"])
            if prior_input != row["input_hash"]:
                raise ValueError("Diagnostic prompt input differs between packets")
            keys.add(key)
            ids.add(sid)
            seeds.add(seed)
            draws.add(index)
        if draws != expected or len(raw_rows) != receipt["count"]:
            raise ValueError("Diagnostic packet lacks complete frozen prompt/draw coverage")
        for score_receipt, score_rows, fingerprint in scored:
            sid = score_receipt["identity"]
            if (
                sid["policy"]["inference_fingerprint"] != fingerprint
                or sid["runtime"] != identity["runtime"]
                or sid["sample_identity"] != canonical_hash(identity)
                or sid["sample_chunks"] != receipt["chunks"]
            ):
                raise ValueError("Diagnostic score receipt binds another policy/packet")
            lookup = {row["sample_key"]: row for row in score_rows}
            if len(lookup) != len(score_rows) or set(lookup) != keys:
                raise ValueError("Duplicate or missing diagnostic score sample")
            for raw in raw_rows:
                row = lookup[raw["sample_key"]]
                if (
                    any(
                        row[field] != raw[field]
                        for field in (
                            "sample_id",
                            "prompt_id",
                            "input_hash",
                            "shared_token_identity",
                            "role",
                            "rng_namespace",
                            "runtime_identity",
                        )
                    )
                    or row["inference_fingerprint"] != fingerprint
                    or row["probability_execution"] != "uncached_prefix_recompute"
                ):
                    raise ValueError("Diagnostic score/raw token/input/policy identity differs")
                logs = np.asarray(row["token_logprobs"], dtype=float)
                if (
                    logs.shape != (len(raw["token_ids"]),)
                    or not np.isfinite(logs).all()
                    or np.any(logs > 0)
                    or not math.isclose(
                        math.fsum(logs), row["sequence_logp"], rel_tol=1e-12, abs_tol=1e-10
                    )
                ):
                    raise ValueError("Diagnostic score probability sum differs")
        used_keys.update(keys)
        used_ids.update(ids)
        used_seeds.update(seeds)
        used_streams.add(stream)
