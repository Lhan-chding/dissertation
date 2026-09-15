"""Recompute measurement summaries from retained paid primitives only.

The arithmetic matches observations.measure, including its summation order.
Reconstruction invokes no sampling, policy scoring, label verification, model,
or oracle. SHA-bound primitive arrays and packet metadata are the source data;
raw/Helmert estimates and covariance need not be duplicated on disk.
"""

from __future__ import annotations

import numpy as np

from .covariance import contrast_covariance, counts_covariance, stable_exp_difference
from .observations import METHODS, _joint_covariance
from .targets import contrast_incidence, helmert

CORE = ("raw", "helmert", "covariance")


def _samples(arrays, row):
    prefix = row["array_prefix"]
    names = row["sample_names"]
    layouts = row.get("sample_layout")
    if layouts is None:
        raise ValueError("Missing primitive sample_layout metadata")
    if [item["name"] for item in layouts] != names or len(names) != len(set(names)):
        raise ValueError("Primitive sample names and layout differ")
    samples = {}
    for index, (name, layout) in enumerate(zip(names, layouts, strict=True)):
        sample = {}
        for field in layout["fields"]:
            key = f"{prefix}sample_{index}_{field}"
            if key not in arrays:
                raise ValueError(f"Missing paid primitive array: {key}")
            sample[field] = np.asarray(arrays[key])
        samples[name] = sample
    return samples


def _labels(sample, prompts, n):
    value = sample["labels"]
    if (
        value.shape != (prompts, n)
        or value.dtype.kind not in "iu"
        or (value < 0).any()
        or (value > 3).any()
    ):
        raise ValueError("Invalid retained verifier labels")
    return value


def _logp(sample, field, prompts, n):
    value = sample[field]
    if (
        value.shape != (prompts, n)
        or value.dtype != np.dtype(np.float64)
        or np.isnan(value).any()
        or np.isposinf(value).any()
    ):
        raise ValueError("Invalid retained normalized action log probabilities")
    return value


def _reconstruct_row(arrays, row):
    method = row["method"].replace("-", "_")
    if method not in METHODS:
        raise ValueError("Unknown observation method")
    if row.get("status") == "BLOCKED_SUPPORT_MISMATCH":
        raise ValueError("Blocked support packet is not a usable measurement")
    n = row["n"]
    canonical = tuple(tuple(pair) for pair in row["policy_fingerprints"])
    if len(canonical) != 2 or any(len(pair) != 2 for pair in canonical):
        raise ValueError("Exactly two candidate contrasts required by unit schema")
    p = len(row["prompt_ids"])
    m = len(canonical)
    if type(n) is not int or n < 2 or p < 1:
        raise ValueError("Positive prompt count and n >= 2 required")
    active = [
        index for index, (baseline, candidate) in enumerate(canonical) if baseline != candidate
    ]
    samples = _samples(arrays, row)
    raw = np.zeros((m, p, 4))
    contributions = np.zeros((n, m, p, 4))
    raw_covariance = np.zeros((p, 4 * m, 4 * m))
    if method == "O_IND":
        policy_ids, incidence = contrast_incidence(canonical)
        used = {policy for j in active for policy in canonical[j]}
        counts = {}
        for policy in used:
            if policy not in samples or "counts" not in samples[policy]:
                raise ValueError("Missing paid O_IND counts")
            value = samples[policy]["counts"]
            if (
                value.shape != (p, 4)
                or value.dtype.kind not in "iu"
                or (value < 0).any()
                or not (value.sum(-1) == n).all()
            ):
                raise ValueError("Invalid retained multinomial counts")
            counts[policy] = value
        for j in active:
            baseline, candidate = canonical[j]
            raw[j] = (
                counts[candidate].astype(np.float64) - counts[baseline].astype(np.float64)
            ) / n
        for i in range(p):
            level_cov = np.zeros((4 * len(policy_ids), 4 * len(policy_ids)))
            for j, policy in enumerate(policy_ids):
                if policy in used:
                    level_cov[j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = counts_covariance(
                        counts[policy][i]
                    )
            raw_covariance[i] = contrast_covariance(level_cov, incidence, coordinates=4)
    elif method == "O_CRN":
        for j in active:
            baseline, candidate = canonical[j]
            left = _labels(samples[baseline], p, n)
            right = _labels(samples[candidate], p, n)
            contributions[:, j] = (np.eye(4)[right] - np.eye(4)[left]).transpose(1, 0, 2)
        raw = contributions.mean(0)
        raw_covariance = _joint_covariance(contributions)
    elif method == "O_LR_ORIGIN" and active:
        labels = _labels(samples["origin"], p, n)
        logrho = _logp(samples["origin"], "logrho", p, n)
        for j in active:
            sample = samples[f"contrast_{j}"]
            logb = _logp(sample, "baseline_logp", p, n)
            logu = _logp(sample, "candidate_logp", p, n)
            weights = stable_exp_difference(logu - logrho, logb - logrho)
            contributions[:, j] = (weights[..., None] * np.eye(4)[labels]).transpose(1, 0, 2)
        raw = contributions.mean(0)
        raw_covariance = _joint_covariance(contributions)
    elif method == "O_LR_MIX":
        independent_groups = [None] * m
        references = row["diagnostics"]["sample_refs"]
        for j in active:
            baseline, candidate = canonical[j]
            independent_groups[j] = tuple(sorted((baseline, candidate)))
            key = references[f"contrast_{j}"]
            sample = samples[key]
            first_pair = int(key.removeprefix("contrast_"))
            first_baseline = canonical[first_pair][0]
            labels = _labels(sample, p, n)
            logrho = _logp(sample, "logrho", p, n)
            first_logb = _logp(sample, "baseline_logp", p, n)
            first_logu = _logp(sample, "candidate_logp", p, n)
            logb, logu = (
                (first_logb, first_logu) if baseline == first_baseline else (first_logu, first_logb)
            )
            weights = stable_exp_difference(logu - logrho, logb - logrho)
            contributions[:, j] = (weights[..., None] * np.eye(4)[labels]).transpose(1, 0, 2)
        raw = contributions.mean(0)
        raw_covariance = _joint_covariance(contributions, independent_groups=independent_groups)
    h = helmert()
    transform = np.kron(np.eye(m), h)
    covariance = np.stack([transform.T @ cov @ transform for cov in raw_covariance])
    return raw, raw @ h, covariance


def reconstruct_measurements(decoded_arrays, metadata, *, force=False):
    """Return raw, Helmert and covariance summaries; legacy stored cores load fast.

    Input is the mapping from packet_codec.decode_arrays and the complete unit
    metadata. Output layout is [replica,14,2,prompt,event/Helmert] and
    covariance [replica,14,prompt,6,6]. A bank absent from the declared banks list
    remains the original zero placeholder; a missing declared packet fails.
    `force=True` ignores stored summaries for reproducibility audits.
    """
    if not force and all(key in decoded_arrays for key in CORE):
        return {key: np.asarray(decoded_arrays[key]) for key in CORE}
    rows = metadata["packets"]
    replicas = metadata["replicas"]
    banks = tuple(metadata["banks"])
    if (
        type(replicas) is not int
        or replicas < 1
        or not rows
        or len(banks) != len(set(banks))
        or any(type(b) is not int or not 0 <= b < 14 for b in banks)
    ):
        raise ValueError("Invalid unit replica/bank metadata")
    prompts = len(rows[0]["prompt_ids"])
    result = {
        "raw": np.zeros((replicas, 14, 2, prompts, 4)),
        "helmert": np.zeros((replicas, 14, 2, prompts, 3)),
        "covariance": np.zeros((replicas, 14, prompts, 6, 6)),
    }
    seen = set()
    for row in rows:
        noise, bank = row["noise_replica"], row["bank"]
        if (noise, bank) in seen:
            raise ValueError("Duplicate unit packet")
        if not 0 <= noise < replicas or bank not in banks:
            raise ValueError("Packet is outside declared unit indices")
        if (
            row["method"] != metadata["method"]
            or row["n"] != metadata["n"]
            or len(row["prompt_ids"]) != prompts
        ):
            raise ValueError("Packet method/sample size/prompts differ from unit")
        seen.add((noise, bank))
        raw, z, cov = _reconstruct_row(decoded_arrays, row)
        result["raw"][noise, bank] = raw
        result["helmert"][noise, bank] = z
        result["covariance"][noise, bank] = cov
    expected = {(noise, bank) for noise in range(replicas) for bank in banks}
    if seen != expected:
        raise ValueError("Missing declared packet; no fabricated zero replacement")
    return result
