"""PSD empirical-distribution kernels with fit-only preprocessing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import LEVELS, METADATA_FIELDS, PreDecisionPacket


def _add(dest, source, weight=1.0):
    for key, value in source.items():
        dest[key] = dest.get(key, 0.0) + weight * value
    return dest


def _dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return math.fsum(v * b.get(k, 0.0) for k, v in a.items())


def _moments(snapshot, level):
    result = {f"global:{i}": v for i, v in enumerate(snapshot["moments_global"] or [0.0] * 36)}
    if LEVELS.index(level) >= 1:
        for stratum, values in snapshot["moments_stratified"].items():
            result.update({f"stratum:{stratum}:{i}": v for i, v in enumerate(values)})
    return result


def _hist_embedding(histograms, prefix, weight=1.0):
    result = {}
    if not histograms:
        return result
    for prompt, hist in histograms.items():
        for atom, count in hist["counts"].items():
            result[f"{prefix}:{prompt}:{atom}"] = math.sqrt(
                weight * count / hist["n"] / len(histograms)
            )
    return result


def _distribution(snapshot, repair=False):
    # Joint and three marginal distributions contribute equal shares to K_E.
    if not repair:
        return _hist_embedding(snapshot["reward_histograms"], "reward")
    result = _hist_embedding(snapshot["repair_histograms"], "joint", 1 / 4)
    if repair:
        for coordinate in ("F", "B", "M"):
            marg = {p: h[coordinate] for p, h in snapshot["repair_marginals"].items()}
            _add(result, _hist_embedding(marg, coordinate, 1 / 4))
    return result


def _nuisance(data):
    result = {
        "source_R0": float(data["source_recipe"] == "R0"),
        "source_R1": float(data["source_recipe"] == "R1"),
        "step": float(data["step"]),
    }
    for key in sorted(METADATA_FIELDS):
        result[key] = float(data["known_training_metadata"].get(key, 0.0))
        result[key + ":observed"] = float(key in data["known_training_metadata"])
    return result


def _fit_standardizer(vectors):
    keys = sorted(set().union(*(v.keys() for v in vectors)))
    a = np.array([[v.get(k, 0.0) for k in keys] for v in vectors])
    mean, std = a.mean(axis=0), a.std(axis=0)
    return {
        k: [float(mu), float(sd)] for k, mu, sd in zip(keys, mean, std, strict=True) if sd > 1e-12
    }


def _standardize(vector, scaler, center=True):
    return {
        k: (vector.get(k, 0.0) - (mu if center else 0.0)) / sd for k, (mu, sd) in scaler.items()
    }


@dataclass(frozen=True)
class FeatureKernel:
    feature_level: str
    moment_scaler: dict
    nuisance_scaler: dict
    scales: dict
    panel: dict
    strata: tuple
    history_weight: float = 0.25
    nuisance_weight: float = 0.25

    @classmethod
    def fit(cls, packets, *, history_weight=0.25, nuisance_weight=0.25):
        packets = list(packets)
        if not packets or any(
            not isinstance(p, PreDecisionPacket) or not p.valid_observations for p in packets
        ):
            raise ValueError("Nonempty valid fit packets required")
        levels = {p.feature_level for p in packets}
        if len(levels) != 1:
            raise ValueError("Mixed feature permissions")
        if not all(math.isfinite(w) and w >= 0 for w in (history_weight, nuisance_weight)):
            raise ValueError("Kernel weights must be nonnegative")
        level = levels.pop()
        data = [p.to_dict() for p in packets]
        moment_scaler = (
            _fit_standardizer([_moments(d["current"], level) for d in data])
            if LEVELS.index(level) < 2
            else {}
        )
        nuisance_scaler = _fit_standardizer([_nuisance(d) for d in data])
        panel = data[0]["current"].get("prompt_strata", {})
        strata = tuple(sorted(data[0]["current"].get("moments_stratified", {})))
        initial = cls(
            level,
            moment_scaler,
            nuisance_scaler,
            {},
            panel,
            strata,
            history_weight,
            nuisance_weight,
        )
        embeddings = [initial.embedding(p) for p in packets]
        scales = {
            key: math.fsum(_dot(e[key], e[key]) for e in embeddings) / len(embeddings)
            for key in initial.weights()
        }
        # A zero component remains exactly zero; never divide by a tiny empirical diagonal.
        scales = {key: (value if value > 1e-15 else 1.0) for key, value in scales.items()}
        return cls(
            level,
            moment_scaler,
            nuisance_scaler,
            scales,
            panel,
            strata,
            history_weight,
            nuisance_weight,
        )

    def embedding(self, packet):
        if not isinstance(packet, PreDecisionPacket) or packet.feature_level != self.feature_level:
            raise ValueError("Inference requires a matching PreDecisionPacket")
        d = packet.to_dict()
        if not packet.valid_observations:
            raise ValueError("No current observations")
        for key in ("current", "history"):
            snap = d[key]
            if not snap["n"]:
                raise ValueError("Missing registered history observations")
            if (
                snap.get("prompt_strata", {}) != self.panel
                or tuple(sorted(snap.get("moments_stratified", {}))) != self.strata
            ):
                raise ValueError("Packet uses a different fixed probe panel")
        if LEVELS.index(self.feature_level) < 2:
            current = _standardize(_moments(d["current"], self.feature_level), self.moment_scaler)
            previous = _standardize(_moments(d["history"], self.feature_level), self.moment_scaler)
            result = dict(current=current, history=_add(dict(current), previous, -1.0))
        else:
            current, previous = _distribution(d["current"]), _distribution(d["history"])
            result = dict(current=current, history=_add(dict(current), previous, -1.0))
            if self.feature_level == LEVELS[3]:
                current_e, previous_e = (
                    _distribution(d["current"], True),
                    _distribution(d["history"], True),
                )
                result.update(
                    repair=current_e, repair_history=_add(dict(current_e), previous_e, -1.0)
                )
        result["nuisance"] = _standardize(_nuisance(d), self.nuisance_scaler)
        return result

    def weights(self):
        weights = {"current": 1.0, "history": self.history_weight, "nuisance": self.nuisance_weight}
        if self.feature_level == LEVELS[3]:
            weights.update(
                current=0.5,
                repair=0.5,
                history=0.5 * self.history_weight,
                repair_history=0.5 * self.history_weight,
            )
        return weights

    def gram(self, left, right=None):
        left = [self.embedding(p) for p in left]
        right = left if right is None else [self.embedding(p) for p in right]
        weights = self.weights()
        return np.array(
            [
                [
                    math.fsum(weights[k] * _dot(a[k], b[k]) / self.scales[k] for k in weights)
                    for b in right
                ]
                for a in left
            ],
            dtype=float,
        )

    def diagnostics(self, packet):
        e = self.embedding(packet)
        return {
            k: {"coordinates": len(v), "nonzero": sum(x != 0 for x in v.values())}
            for k, v in e.items()
        }

    def to_dict(self):
        return dict(
            feature_level=self.feature_level,
            moment_scaler=self.moment_scaler,
            nuisance_scaler=self.nuisance_scaler,
            scales=self.scales,
            panel=self.panel,
            strata=list(self.strata),
            history_weight=self.history_weight,
            nuisance_weight=self.nuisance_weight,
        )

    @classmethod
    def from_dict(cls, data):
        if set(data) != {
            "feature_level",
            "moment_scaler",
            "nuisance_scaler",
            "scales",
            "panel",
            "strata",
            "history_weight",
            "nuisance_weight",
        }:
            raise ValueError("Frozen kernel schema mismatch")
        keys = {"current", "history", "nuisance"}
        if data["feature_level"] == LEVELS[3]:
            keys.update(("repair", "repair_history"))
        if data["feature_level"] not in LEVELS or set(data["scales"]) != keys:
            raise ValueError("Frozen kernel scale mismatch")
        if any(not math.isfinite(v) or v <= 0 for v in data["scales"].values()):
            raise ValueError("Invalid frozen kernel scales")
        return cls(**{**data, "strata": tuple(data["strata"])})
