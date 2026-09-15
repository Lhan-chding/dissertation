"""Paid finite observations and isolated exact audits for the fixed CPU action space.

The service owns complete action distributions. A packet contains only sampled
actions/labels, paid scalar scores, estimates and covariance. No model, callback,
path, full table or exact covariance is returned to finite estimators.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

from .cost import CostLedger
from .covariance import (
    contrast_covariance,
    counts_covariance,
    covariance_of_mean,
    lr_moments,
    multinomial_covariance,
    probability,
    stable_exp_difference,
    weighted_moments,
)
from .targets import contrast_incidence, helmert, inference_fingerprint

METHODS = ("O_IND", "O_CRN", "O_LR_ORIGIN", "O_LR_MIX")


def _readonly(value, dtype=None):
    value = np.array(value, dtype=dtype, copy=True)
    value.flags.writeable = False
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _method(method):
    method = method.replace("-", "_")
    if method not in METHODS:
        raise ValueError("Unknown observation estimator")
    return method


def _seed(*parts):
    return int(_digest(parts)[:16], 16)


def _validate_categories(categories):
    categories = np.asarray(categories)
    if (
        categories.ndim != 2
        or categories.dtype.kind not in "iu"
        or (categories < 0).any()
        or (categories > 3).any()
    ):
        raise ValueError("Prompt-by-action integer labels in X/S/W/I order required")
    return _readonly(categories, np.int8)


class ToyWorldService:
    """Frozen inference only. Registration never trains or mutates parent models."""

    def __init__(self, categories, *, prompt_ids=None, ledger=None):
        self._categories = _validate_categories(categories)
        self.prompt_count, self.action_count = self._categories.shape
        self.prompt_ids = tuple(
            prompt_ids if prompt_ids is not None else map(str, range(self.prompt_count))
        )
        if (
            len(self.prompt_ids) != self.prompt_count
            or len(set(self.prompt_ids)) != self.prompt_count
        ):
            raise ValueError("Unique aligned prompt identities required")
        self.ledger = ledger if ledger is not None else CostLedger()
        self._alias = {}
        self._tables = {}
        self._parameters = {}
        self._features = None
        self._score_cache = {}
        self._label_cache = set()
        self._sample_evaluation_cache = set()
        self._last_packet = None
        self.execution_kind = "UNINITIALIZED"
        self.dataset_fingerprint = _digest([self.prompt_ids, self._categories.tolist()])

    @classmethod
    def from_action_probabilities(cls, tables, categories, *, prompt_ids=None, ledger=None):
        """Analytic fixtures only; no fabricated action tables from event counts."""
        service = cls(categories, prompt_ids=prompt_ids, ledger=ledger)
        for name, value in tables.items():
            value = np.asarray(value, dtype=np.float64)
            if value.shape != service._categories.shape:
                raise ValueError("Aligned prompt-by-action probability tables required")
            for row in value:
                probability(row)
            fingerprint = inference_fingerprint(value, "analytic_action_table")
            service._alias[name] = fingerprint
            service._tables[fingerprint] = _readonly(value)
        service.execution_kind = "ANALYTIC_FIXTURE"
        return service

    @classmethod
    def from_parameters(cls, theta_by_id, features, categories, *, prompt_ids=None, ledger=None):
        service = cls(categories, prompt_ids=prompt_ids, ledger=ledger)
        features = np.asarray(features, dtype=np.float64)
        if (
            features.shape != (service.prompt_count, 16, 44)
            or service.action_count != 16
            or not np.isfinite(features).all()
        ):
            raise ValueError("Parent finite CPU feature contract required")
        service._features = _readonly(features)
        service.dataset_fingerprint = _digest(
            [service.dataset_fingerprint, inference_fingerprint(features, "parent_features")]
        )
        for name, theta in theta_by_id.items():
            theta = np.asarray(theta, dtype=np.float64)
            if theta.shape != (737,) or not np.isfinite(theta).all():
                raise ValueError("Frozen parent 737 parameter vector required")
            fingerprint = inference_fingerprint(theta)
            service._alias[name] = fingerprint
            service._parameters[fingerprint] = _readonly(theta)
        service.execution_kind = "FROZEN_PARENT_CPU_FORWARD"
        return service

    def fingerprint(self, policy):
        if policy in self._alias:
            return self._alias[policy]
        if policy in self._tables or policy in self._parameters:
            return policy
        raise ValueError("Unknown policy identity")

    def _table(self, policy):
        fingerprint = self.fingerprint(policy)
        if fingerprint not in self._tables:
            from src.modeling_qualification.toy import _numpy_forward

            started = time.perf_counter()
            action = _numpy_forward(
                self._parameters[fingerprint], self._features, self._categories
            )[1]
            self._tables[fingerprint] = _readonly(action)
            self.ledger.add("physical_policy_forwards", self.prompt_count)
            self.ledger.add("forward_seconds", time.perf_counter() - started)
        return self._tables[fingerprint]

    def support_covers(self, origin, pairs):
        """Support status only. Full distributions remain internal to the service."""
        rho = self._table(origin)
        return all(not ((rho == 0) & (self._table(u) != self._table(b))).any() for b, u in pairs)

    def verify(self, prompt, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if (
            not 0 <= prompt < self.prompt_count
            or (actions < 0).any()
            or (actions >= self.action_count).any()
        ):
            raise ValueError("Valid specified action IDs required")
        for action in actions.ravel():
            key = (prompt, int(action))
            self.ledger.add("label_cache_hits" if key in self._label_cache else "verified_labels")
            self._label_cache.add(key)
        return self._categories[prompt, actions].copy()

    def sample(self, policy, n, rng, *, uniforms=None, return_logp=False):
        if type(n) is not int or n < 1:
            raise ValueError("Positive integer n required")
        started = time.perf_counter()
        table = self._table(policy)
        fingerprint = self.fingerprint(policy)
        if uniforms is None:
            uniforms = rng.random((self.prompt_count, n))
        uniforms = np.asarray(uniforms, dtype=np.float64)
        if (
            uniforms.shape != (self.prompt_count, n)
            or not np.isfinite(uniforms).all()
            or (uniforms < 0).any()
            or (uniforms >= 1).any()
        ):
            raise ValueError("Aligned shared U values in [0,1) required")
        actions = np.empty((self.prompt_count, n), dtype=np.int16)
        labels = np.empty_like(actions, dtype=np.int8)
        logs = np.empty_like(actions, dtype=np.float64) if return_logp else None
        for i, row in enumerate(table):
            key = (fingerprint, i)
            if key not in self._sample_evaluation_cache:
                self.ledger.add("sample_policy_evaluations")
                self._sample_evaluation_cache.add(key)
            cdf = np.cumsum(row)
            cdf[-1] = 1.0
            # The original action_id order is the sole ordering used here.
            actions[i] = np.searchsorted(cdf, uniforms[i], side="right")
            labels[i] = self.verify(i, actions[i])
            if return_logp:
                logs[i] = np.log(row[actions[i]])
                for action, logp in zip(actions[i], logs[i], strict=True):
                    self._score_cache[(fingerprint, i, int(action))] = float(logp)
        self.ledger.add("generated_actions", self.prompt_count * n)
        self.ledger.add("sampling_seconds", time.perf_counter() - started)
        return actions, labels, logs

    def sample_mixture(self, baseline, candidate, n, rng):
        """iid Bernoulli source per draw; never fixed source quotas."""
        started = time.perf_counter()
        tables = [self._table(baseline), self._table(candidate)]
        source = rng.integers(0, 2, size=(self.prompt_count, n), dtype=np.int8)
        uniforms = rng.random((self.prompt_count, n))
        actions = np.empty((self.prompt_count, n), dtype=np.int16)
        labels = np.empty_like(actions, dtype=np.int8)
        for i in range(self.prompt_count):
            for which, policy in enumerate((baseline, candidate)):
                indices = source[i] == which
                if not indices.any():
                    continue
                fingerprint = self.fingerprint(policy)
                key = (fingerprint, i)
                if key not in self._sample_evaluation_cache:
                    self.ledger.add("sample_policy_evaluations")
                    self._sample_evaluation_cache.add(key)
                cdf = np.cumsum(tables[which][i])
                cdf[-1] = 1.0
                actions[i, indices] = np.searchsorted(cdf, uniforms[i, indices], side="right")
                for action in actions[i, indices]:
                    self._score_cache[(fingerprint, i, int(action))] = float(
                        np.log(tables[which][i, action])
                    )
            labels[i] = self.verify(i, actions[i])
        self.ledger.add("generated_actions", self.prompt_count * n)
        self.ledger.add("sampling_seconds", time.perf_counter() - started)
        return actions, labels, source

    def score(self, policy, actions):
        started = time.perf_counter()
        actions = np.asarray(actions, dtype=np.int64)
        if (
            actions.ndim != 2
            or len(actions) != self.prompt_count
            or (actions < 0).any()
            or (actions >= self.action_count).any()
        ):
            raise ValueError("Aligned prompt-by-specified-action queries required")
        fingerprint = self.fingerprint(policy)
        table = self._table(policy)
        result = np.empty(actions.shape, dtype=np.float64)
        for i in range(self.prompt_count):
            for j, action in enumerate(actions[i]):
                key = (fingerprint, i, int(action))
                self.ledger.add("score_requests")
                if key in self._score_cache:
                    self.ledger.add("score_cache_hits")
                else:
                    self.ledger.add("cross_policy_action_scores")
                    self._score_cache[key] = (
                        float(np.log(table[i, action])) if table[i, action] > 0 else -np.inf
                    )
                result[i, j] = self._score_cache[key]
        self.ledger.add("score_seconds", time.perf_counter() - started)
        return result

    def known_event_actions(self):
        correct = [np.flatnonzero(row == 0) for row in self._categories]
        invalid = [np.flatnonzero(row == 3) for row in self._categories]
        if any(len(x) != 1 for x in correct) or any(len(x) != 2 for x in invalid):
            raise ValueError("Known-event baseline requires exactly one X and two I per prompt")
        return np.asarray([np.r_[x, bad] for x, bad in zip(correct, invalid, strict=True)])


@dataclass(frozen=True)
class ObservationPacket:
    packet_id: str
    method: str
    n: int
    purpose: str
    seed: int
    pairs: tuple
    policy_fingerprints: tuple
    prompt_ids: tuple
    event_estimate: np.ndarray
    raw_event_estimate: np.ndarray
    helmert_estimate: np.ndarray
    covariance: np.ndarray
    raw_covariance: np.ndarray
    contributions: np.ndarray
    mass_residual: np.ndarray
    samples: dict
    sample_ids: tuple
    cost: dict
    status: str
    diagnostics: dict

    def arrays(self, *, include_contributions=False, include_covariance=True):
        """Compact NPZ payload; contributions can be rebuilt from actions/labels/logp."""
        result = {
            name: getattr(self, name)
            for name in (
                "event_estimate",
                "raw_event_estimate",
                "helmert_estimate",
                "mass_residual",
            )
        }
        if include_covariance:
            result.update(covariance=self.covariance, raw_covariance=self.raw_covariance)
        if include_contributions:
            result["contributions"] = self.contributions
        for index, (_, sample) in enumerate(self.samples.items()):
            for field, value in sample.items():
                if isinstance(value, np.ndarray):
                    result[f"sample_{index}_{field}"] = value
        return result

    def metadata(self):
        result = {
            key: getattr(self, key)
            for key in (
                "packet_id",
                "method",
                "n",
                "purpose",
                "seed",
                "pairs",
                "policy_fingerprints",
                "prompt_ids",
                "sample_ids",
                "cost",
                "status",
                "diagnostics",
            )
        }
        result["sample_layout"] = [
            {"name": name, "fields": list(value)} for name, value in self.samples.items()
        ]
        return result


def _joint_covariance(contributions, independent_groups=None):
    n, m, p, e = contributions.shape
    result = np.zeros((p, m * e, m * e))
    for i in range(p):
        result[i] = covariance_of_mean(contributions[:, :, i, :].reshape(n, m * e))
        if independent_groups is not None:
            for j in range(m):
                for k in range(m):
                    if independent_groups[j] != independent_groups[k]:
                        result[i, j * e : (j + 1) * e, k * e : (k + 1) * e] = 0
    return result


def measure(
    service,
    pairs,
    *,
    origin_id=None,
    method="O_IND",
    n=64,
    seed=0,
    purpose="fit",
    legacy_counts=None,
):
    method = _method(method)
    if type(n) is not int or n < 2 or type(seed) is not int:
        raise ValueError("n >= 2 and integer RNG seed required")
    if purpose not in ("fit", "diagnostic", "evaluation", "observation_study"):
        raise ValueError("Explicit independent packet purpose required")
    pairs = tuple(tuple(pair) for pair in pairs)
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("Nonempty baseline/candidate pair list required")
    canonical = tuple((service.fingerprint(b), service.fingerprint(u)) for b, u in pairs)
    active = [i for i, (b, u) in enumerate(canonical) if b != u]
    origin = service.fingerprint(origin_id) if origin_id is not None else None
    count_hash = (
        None
        if legacy_counts is None
        else _digest(
            {
                str(k): hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
                for k, v in legacy_counts.items()
            }
        )
    )
    packet_key = _digest(
        [method, n, seed, purpose, canonical, origin, service.dataset_fingerprint, count_hash]
    )
    if service._last_packet is not None and service._last_packet.packet_id == packet_key:
        return service._last_packet
    before = service.ledger.snapshot()
    rng = np.random.default_rng(
        _seed(seed, purpose, method, canonical, origin, service.dataset_fingerprint)
    )
    m, p = len(pairs), service.prompt_count
    contributions = np.zeros((n, m, p, 4))
    raw = np.zeros((m, p, 4))
    samples = {}
    sample_refs = {}
    actual_samples = []
    diagnostics = {
        "action_order": "parent_action_id_not_verifier_label",
        "access_regime": "SAMPLE_AND_LOGP" if "LR" in method else "SAMPLE_ONLY",
        "alias_contrasts": [i for i in range(m) if i not in active],
        "covariance_scope": "fixed_policy_observation_noise_only",
        "execution_kind": service.execution_kind,
        "dataset_fingerprint": service.dataset_fingerprint,
        "proposal_ids": [],
        "n_is_per_unique_policy_or_iid_proposal": True,
    }
    raw_covariance = np.zeros((p, 4 * m, 4 * m))
    status = "ALIAS_EXACT_ZERO" if not active else "OBSERVED"
    if method == "O_LR_ORIGIN" and active:
        if origin is None:
            raise ValueError("Origin proposal identity required")
        if not service.support_covers(origin, [canonical[i] for i in active]):
            status = "BLOCKED_SUPPORT_MISMATCH"
    if status == "BLOCKED_SUPPORT_MISMATCH":
        diagnostics["support"] = status
    elif method == "O_IND":
        policy_ids, lmat = contrast_incidence(canonical)
        used = {policy for i in active for policy in canonical[i]}
        counts = {}
        if legacy_counts is not None:
            canonical_counts = {}
            for key, value in legacy_counts.items():
                identity = service.fingerprint(key)
                value = np.asarray(value)
                if (
                    value.shape != (p, 4)
                    or value.dtype.kind not in "iu"
                    or (value < 0).any()
                    or not (value.sum(-1) == n).all()
                ):
                    raise ValueError("Aligned unmodified legacy counts with n total required")
                if identity in canonical_counts and not np.array_equal(
                    canonical_counts[identity], value
                ):
                    raise ValueError("Alias policies have inconsistent legacy counts")
                canonical_counts[identity] = value
            for policy in used:
                if policy not in canonical_counts:
                    raise ValueError("MISSING_LEGACY_COUNTS")
                counts[policy] = canonical_counts[policy]
                service.ledger.add("reused_actions", p * n)
                samples[policy] = {"counts": _readonly(counts[policy])}
            contributions = np.zeros((0, m, p, 4))
        else:
            for policy in policy_ids:
                if policy not in used:
                    continue
                actions, labels, _ = service.sample(policy, n, rng)
                counts[policy] = np.stack([np.bincount(row, minlength=4) for row in labels])
                samples[policy] = {"actions": actions, "labels": labels, "counts": counts[policy]}
            for j in active:
                b, u = canonical[j]
                contributions[:, j] = (
                    np.eye(4)[samples[u]["labels"]] - np.eye(4)[samples[b]["labels"]]
                ).transpose(1, 0, 2)
        for j in active:
            b, u = canonical[j]
            # Historical count archives use uint16; cast before subtracting.
            raw[j] = (counts[u].astype(np.float64) - counts[b].astype(np.float64)) / n
        for i in range(p):
            level_cov = np.zeros((4 * len(policy_ids), 4 * len(policy_ids)))
            for j, policy in enumerate(policy_ids):
                if policy in used:
                    level_cov[j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = counts_covariance(
                        counts[policy][i]
                    )
            raw_covariance[i] = contrast_covariance(level_cov, lmat, coordinates=4)
        diagnostics["covariance_estimator"] = "multinomial_Jeffreys_half_covariance_only"
        actual_samples = list(samples)
    elif method == "O_CRN":
        uniforms = rng.random((p, n))
        for policy in dict.fromkeys(policy for j in active for policy in canonical[j]):
            actions, labels, _ = service.sample(policy, n, rng, uniforms=uniforms)
            samples[policy] = {"actions": actions, "labels": labels}
        for j in active:
            b, u = canonical[j]
            contributions[:, j] = (
                np.eye(4)[samples[u]["labels"]] - np.eye(4)[samples[b]["labels"]]
            ).transpose(1, 0, 2)
        raw = contributions.mean(0)
        raw_covariance = _joint_covariance(contributions)
        diagnostics["event_discordance"] = (contributions**2).mean(0).tolist()
        actual_samples = list(samples)
    elif method == "O_LR_ORIGIN" and active:
        actions, labels, logrho = service.sample(origin, n, rng, return_logp=True)
        samples["origin"] = {"actions": actions, "labels": labels, "logrho": logrho}
        scores = {
            policy: service.score(policy, actions)
            for policy in dict.fromkeys(policy for j in active for policy in canonical[j])
        }
        for j in active:
            b, u = canonical[j]
            weights = stable_exp_difference(scores[u] - logrho, scores[b] - logrho)
            contributions[:, j] = (weights[..., None] * np.eye(4)[labels]).transpose(1, 0, 2)
            samples[f"contrast_{j}"] = {"baseline_logp": scores[b], "candidate_logp": scores[u]}
        raw = contributions.mean(0)
        raw_covariance = _joint_covariance(contributions)
        diagnostics["proposal_ids"] = [_digest(["origin", origin])]
        actual_samples = ["origin"]
    elif method == "O_LR_MIX":
        unique_samples = {}
        independent_groups = [None] * m
        for j in active:
            b, u = canonical[j]
            group = tuple(sorted((b, u)))
            independent_groups[j] = group
            if group not in unique_samples:
                actions, labels, source = service.sample_mixture(b, u, n, rng)
                logb, logu = service.score(b, actions), service.score(u, actions)
                logrho = np.logaddexp(logb, logu) - np.log(2.0)
                key = f"contrast_{j}"
                samples[key] = {
                    "actions": actions,
                    "labels": labels,
                    "source": source,
                    "baseline_logp": logb,
                    "candidate_logp": logu,
                    "logrho": logrho,
                }
                unique_samples[group] = (key, b, u)
                actual_samples.append(key)
                diagnostics["proposal_ids"].append(_digest(["iid_bernoulli_mixture", group]))
            key, first_b, _first_u = unique_samples[group]
            sample_refs[f"contrast_{j}"] = key
            sample = samples[key]
            actions, labels, logrho = sample["actions"], sample["labels"], sample["logrho"]
            logb, logu = (
                (sample["baseline_logp"], sample["candidate_logp"])
                if b == first_b
                else (sample["candidate_logp"], sample["baseline_logp"])
            )
            weights = stable_exp_difference(logu - logrho, logb - logrho)
            if (np.abs(weights) > 2 + 1e-12).any():
                raise FloatingPointError("Mixture analytic weight bound violated")
            contributions[:, j] = (weights[..., None] * np.eye(4)[labels]).transpose(1, 0, 2)
        raw = contributions.mean(0)
        raw_covariance = _joint_covariance(contributions, independent_groups=independent_groups)
        diagnostics["analytic_weight_bound"] = 2.0
        diagnostics["mixture_sampling"] = "iid_bernoulli_source_per_draw"
        diagnostics["cross_pair_covariance"] = (
            "independent_unique_proposals; redundant_or_reversed_pairs_share_draws"
        )
        diagnostics["sample_refs"] = sample_refs
    h = helmert()
    transform = np.kron(np.eye(m), h)
    covariance = np.stack([transform.T @ cov @ transform for cov in raw_covariance])
    z = raw @ h
    event = z @ h.T if "LR" in method else raw.copy()
    unobserved = []
    for j in active:
        if method == "O_IND":
            b, u = canonical[j]
            missing = (counts[b][:, 0] + counts[u][:, 0]) == 0
        elif method == "O_CRN":
            b, u = canonical[j]
            missing = ~((samples[b]["labels"] == 0).any(1) | (samples[u]["labels"] == 0).any(1))
        elif status == "BLOCKED_SUPPORT_MISMATCH":
            missing = np.ones(p, dtype=bool)
        else:
            sample = (
                samples["origin"]
                if method == "O_LR_ORIGIN"
                else samples[sample_refs[f"contrast_{j}"]]
            )
            missing = ~(sample["labels"] == 0).any(1)
        unobserved.extend([[j, int(i)] for i in np.flatnonzero(missing)])
    diagnostics.update(
        unobserved_X=unobserved,
        unobserved_X_status="UNOBSERVED_X",
        covariance_status="EMPIRICAL_DEGENERATE"
        if active and np.all(covariance == 0)
        else "ESTIMATED",
        mass_residual_view="raw_event_LR",
        known_zero_alias_only=True,
        scalar_scores_are_normalized_action_log_probabilities=True,
    )
    packet = ObservationPacket(
        packet_key,
        method,
        n,
        purpose,
        seed,
        pairs,
        canonical,
        service.prompt_ids,
        _readonly(event),
        _readonly(raw),
        _readonly(z),
        _readonly(covariance),
        _readonly(raw_covariance),
        _readonly(contributions),
        _readonly(raw.sum(-1)),
        {
            name: {key: _readonly(value) for key, value in sample.items()}
            for name, sample in samples.items()
        },
        tuple(_digest([packet_key, name]) for name in actual_samples),
        service.ledger.delta(before),
        status,
        diagnostics,
    )
    service._last_packet = packet  # bounded cache: avoids retaining all noise replicas
    return packet


def known_event_logp(service, policies):
    before = service.ledger.snapshot()
    actions = service.known_event_actions()
    for i, row in enumerate(actions):
        service.verify(i, row)
    scores = np.stack([np.exp(service.score(policy, actions)) for policy in policies])
    return {
        "method": "D_KNOWN_EVENT_LOGP",
        "status": "EXACT_FINITE_EVENT_IDENTITY",
        "pX": scores[:, :, 0],
        "v": 1 - scores[:, :, 1:].sum(-1),
        "actions": actions,
        "cost": service.ledger.delta(before),
        "access_regime": "SAMPLE_AND_LOGP",
        "correct_actions_per_prompt": 1,
        "invalid_actions_per_prompt": 2,
        "physical_forward_reuse": True,
        "scope": "parent_finite_action_toy_only",
    }


def full_enumeration_logp(service, policies):
    """Paid all-action scoring baseline; a diagnostic for this finite support only."""
    before = service.ledger.snapshot()
    actions = np.tile(np.arange(service.action_count), (service.prompt_count, 1))
    labels = np.stack([service.verify(i, row) for i, row in enumerate(actions)])
    probabilities = np.stack([np.exp(service.score(policy, actions)) for policy in policies])
    events = np.einsum("kpa,pae->kpe", probabilities, np.eye(4)[labels])
    return {
        "method": "D_FULL_FINITE_ENUMERATION",
        "status": "EXACT_FINITE_EVENT_ENUMERATION_DIAGNOSTIC",
        "event_probabilities": events,
        "pX": events[:, :, 0],
        "v": events[:, :, :3].sum(-1),
        "cost": service.ledger.delta(before),
        "access_regime": "SAMPLE_AND_LOGP",
        "actions_per_prompt": service.action_count,
        "scope": "parent_finite_action_toy_only",
    }


class OracleAudit:
    """Explicit exact-score object; never pass it to finite fitting/prediction."""

    def __init__(self, service):
        self._service = service

    @classmethod
    def from_action_probabilities(cls, tables, categories, **kwargs):
        return cls(ToyWorldService.from_action_probabilities(tables, categories, **kwargs))

    @classmethod
    def from_parameters(cls, theta_by_id, features, categories, **kwargs):
        return cls(ToyWorldService.from_parameters(theta_by_id, features, categories, **kwargs))

    def moments(self, pairs, *, origin_id=None, method="O_IND", n=64):
        method = _method(method)
        if type(n) is not int or n < 1:
            raise ValueError("Positive n required")
        service = self._service
        before = service.ledger.snapshot()
        canonical = tuple((service.fingerprint(b), service.fingerprint(u)) for b, u in pairs)
        policies, lmat = contrast_incidence(canonical)
        tables = {policy: service._table(policy) for policy in policies}
        if method == "O_LR_ORIGIN":
            if origin_id is None:
                raise ValueError("Origin required")
            origin = service._table(origin_id)
        p, m = service.prompt_count, len(pairs)
        mean = np.zeros((m, p, 4))
        raw_cov = np.zeros((p, 4 * m, 4 * m))
        for i in range(p):
            features = np.eye(4)[service._categories[i]]
            for j, (b, u) in enumerate(canonical):
                mean[j, i] = (tables[u][i] - tables[b][i]) @ features
            if method == "O_IND":
                level = np.zeros((4 * len(policies), 4 * len(policies)))
                for j, policy in enumerate(policies):
                    level[4 * j : 4 * j + 4, 4 * j : 4 * j + 4] = multinomial_covariance(
                        tables[policy][i] @ features, n
                    )
                raw_cov[i] = contrast_covariance(level, lmat, coordinates=4)
            elif method == "O_CRN":
                cdfs = {policy: np.cumsum(tables[policy][i]) for policy in policies}
                edges = np.unique(
                    np.clip(np.concatenate([np.array([0.0, 1.0]), *cdfs.values()]), 0, 1)
                )
                masses = np.diff(edges)
                mid = (edges[1:] + edges[:-1]) / 2
                sampled = {
                    policy: np.searchsorted(cdf, mid, side="right").clip(
                        0, service.action_count - 1
                    )
                    for policy, cdf in cdfs.items()
                }
                contribution = np.concatenate(
                    [features[sampled[u]] - features[sampled[b]] for b, u in canonical], axis=1
                )
                _, cov = weighted_moments(contribution, masses)
                raw_cov[i] = cov / n
            elif method == "O_LR_ORIGIN":
                rho = origin[i]
                columns = []
                for b, u in canonical:
                    _, _, weights = lr_moments(tables[b][i], tables[u][i], rho, features)
                    columns.append(weights[:, None] * features)
                _, cov = weighted_moments(np.concatenate(columns, axis=1), rho)
                raw_cov[i] = cov / n
            else:
                groups = {}
                for j, (b, u) in enumerate(canonical):
                    groups.setdefault(tuple(sorted((b, u))), []).append(j)
                for (left, right), indices in groups.items():
                    rho = (tables[left][i] + tables[right][i]) / 2
                    columns = []
                    for j in indices:
                        b, u = canonical[j]
                        _, _, weights = lr_moments(tables[b][i], tables[u][i], rho, features)
                        columns.append(weights[:, None] * features)
                    _, cov = weighted_moments(np.concatenate(columns, axis=1), rho)
                    coordinates = np.concatenate([np.arange(4 * j, 4 * j + 4) for j in indices])
                    raw_cov[i][np.ix_(coordinates, coordinates)] = cov / n
        service.ledger.add("exact_probability_calls", len(policies) * p)
        transform = np.kron(np.eye(m), helmert())
        covariance = np.stack([transform.T @ cov @ transform for cov in raw_cov])
        return {
            "status": "ORACLE_DIAGNOSTIC_ONLY",
            "event_mean": mean,
            "raw_covariance": raw_cov,
            "covariance": covariance,
            "n": n,
            "method": method,
            "cost": service.ledger.delta(before),
            "event_helmert_covariance": np.stack(
                [
                    np.kron(np.eye(m), helmert()) @ cov @ np.kron(np.eye(m), helmert()).T
                    for cov in covariance
                ]
            ),
        }

    def event_probabilities(self, policies):
        service = self._service
        result = np.stack(
            [
                np.einsum("pa,pae->pe", service._table(policy), np.eye(4)[service._categories])
                for policy in policies
            ]
        )
        service.ledger.add("exact_probability_calls", len(policies) * service.prompt_count)
        return result
