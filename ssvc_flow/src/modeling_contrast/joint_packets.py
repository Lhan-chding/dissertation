"""Joint paid covariance across banks that reuse historical inference aliases.

Input schema: packet['covariance'] is [noise,bank,prompt,6,6]; metadata has
method, n, and a packets list containing bank, noise_replica, old_counts_reused,
packet_id, policy_fingerprints=[(baseline,joint),(baseline,gate)]. Historical
O_IND uses legacy_total_levels [noise,bank,operation,prompt,event] in parent
operation order. Each requested bank contributes joint/gate rows, each with
three Helmert coordinates. Thus output is [prompt,6*len(banks),6*len(banks)].

Within one historical archive and noise replica, equal policy fingerprints
refer to the same paid count vector; incidence propagates that dependency
across banks. A newly sampled packet is independent of other bank packets even
when its policy parameters match. This function reads no oracle or model.
"""

from __future__ import annotations

import numpy as np

from .targets import helmert


def _context(packet, noise, banks):
    banks = tuple(banks)
    if any(isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)) for b in banks):
        raise ValueError("Integer bank indices required")
    banks = tuple(int(b) for b in banks)
    if isinstance(noise, (bool, np.bool_)) or not isinstance(noise, (int, np.integer)):
        raise ValueError("Integer noise replica required")
    noise = int(noise)
    if (
        not banks
        or len(banks) != len(set(banks))
        or any(type(b) is not int or b < 0 for b in banks)
    ):
        raise ValueError("Unique nonnegative integer banks required")
    if type(noise) is not int or noise < 0:
        raise ValueError("Nonnegative integer noise replica required")
    metadata = packet["metadata"]
    method = metadata["method"].replace("-", "_")
    n = metadata["n"]
    if type(n) is not int or n < 1:
        raise ValueError("Positive integer sample count required")
    rows = {}
    for row in metadata["packets"]:
        bank = row["bank"]
        if row["noise_replica"] == noise and bank in banks:
            if bank in rows:
                raise ValueError("Duplicate bank packet metadata")
            pairs = tuple(tuple(pair) for pair in row["policy_fingerprints"])
            if (
                len(pairs) != 2
                or any(len(pair) != 2 for pair in pairs)
                or pairs[0][0] != pairs[1][0]
            ):
                raise ValueError("Expected joint/gate pairs sharing a baseline identity")
            rows[bank] = dict(row, policy_fingerprints=pairs)
    if set(rows) != set(banks):
        raise ValueError("Missing selected bank packet metadata")
    return banks, method, n, rows


def fit_packet_covariance(packet, noise, banks):
    """Combine within-packet covariance and genuine legacy cross-bank reuse."""
    banks, method, n, rows = _context(packet, noise, banks)
    covariance = np.asarray(packet["covariance"], dtype=np.float64)
    if (
        covariance.ndim != 5
        or covariance.shape[-2:] != (6, 6)
        or noise >= len(covariance)
        or max(banks) >= covariance.shape[1]
    ):
        raise ValueError("Covariance must be noise/bank/prompt/6/6 with requested indices")
    selected = covariance[noise, list(banks)]
    if not np.isfinite(selected).all():
        raise ValueError("Finite selected packet covariance required")
    prompts = covariance.shape[2]
    result = np.zeros((prompts, 6 * len(banks), 6 * len(banks)))
    historical = []
    for index, bank in enumerate(banks):
        if method == "O_IND" and rows[bank].get("old_counts_reused") is True:
            historical.append((index, bank))
        else:
            result[:, 6 * index : 6 * index + 6, 6 * index : 6 * index + 6] = covariance[
                noise, bank
            ]
    if not historical:
        return result
    if "legacy_total_levels" not in packet:
        raise ValueError("Legacy alias covariance requires paid legacy_total_levels")
    levels = np.asarray(packet["legacy_total_levels"], dtype=np.float64)
    if (
        levels.ndim != 5
        or levels.shape[2:] != (3, prompts, 4)
        or noise >= len(levels)
        or max(banks) >= levels.shape[1]
    ):
        raise ValueError("Legacy levels must be noise/bank/operation/prompt/event")
    policy_counts = {}
    for _index, bank in historical:
        scaled = levels[noise, bank] * n
        rounded = np.rint(scaled)
        if (
            not np.isfinite(scaled).all()
            or (scaled < 0).any()
            or not np.array_equal(scaled, rounded)
            or not (rounded.sum(-1) == n).all()
        ):
            raise ValueError("Legacy levels must recover nonnegative integer counts exactly")
        counts = rounded.astype(np.int64)
        pairs = rows[bank]["policy_fingerprints"]
        identities = (pairs[0][0], pairs[0][1], pairs[1][1])
        for policy, count in zip(identities, counts, strict=True):
            if policy in policy_counts and not np.array_equal(policy_counts[policy], count):
                raise ValueError("Shared legacy inference alias has inconsistent paid counts")
            policy_counts[policy] = count
    policy_ids = tuple(policy_counts)
    incidence = np.zeros((2 * len(banks), len(policy_ids)))
    for index, bank in historical:
        for pair_index, (baseline, candidate) in enumerate(rows[bank]["policy_fingerprints"]):
            incidence[2 * index + pair_index, policy_ids.index(baseline)] -= 1
            incidence[2 * index + pair_index, policy_ids.index(candidate)] += 1
    h = helmert()
    counts = np.stack([policy_counts[policy] for policy in policy_ids])
    probabilities = (counts + 0.5) / (n + 2.0)
    mean = probabilities @ h
    per_policy = (
        np.einsum("upe,ei,ej->upij", probabilities, h, h) - mean[..., :, None] * mean[..., None, :]
    ) / n
    # axes: prompt,contrast,Helmert,contrast,Helmert; no independence assumption
    # is introduced between historical rows with a nonzero shared incidence.
    historical_covariance = np.einsum(
        "au,bu,uijk->iajbk", incidence, incidence, per_policy, optimize=True
    )
    result += historical_covariance.reshape(prompts, 6 * len(banks), 6 * len(banks))
    return result


def retained_fit_rows(packet, noise, banks, theta=None):
    """Indices in bank-major joint/gate rows after exact sample redundancy removal.

    Inference alias contrasts are known zero and omitted. Historical identical or
    reversed pairs sharing the same counts are retained once across banks. New
    independent measurements of the same policy pair remain separate rows; only
    reused draws within the same packet can be deduplicated. `theta` is accepted
    for caller compatibility but is not consulted or inferred from observations.
    """
    banks, method, _n, rows = _context(packet, noise, banks)
    seen, keep = set(), []
    for index, bank in enumerate(banks):
        row = rows[bank]
        historical = method == "O_IND" and row.get("old_counts_reused") is True
        source = (
            ("legacy_count_replica", noise)
            if historical
            else ("independent_packet", row.get("packet_id", bank))
        )
        for pair_index, (baseline, candidate) in enumerate(row["policy_fingerprints"]):
            if baseline == candidate:
                continue
            key = (source, tuple(sorted((baseline, candidate))))
            if key not in seen:
                seen.add(key)
                keep.append(2 * index + pair_index)
    return np.asarray(keep, dtype=np.int64)
