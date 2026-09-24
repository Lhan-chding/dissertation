"""Small frozen KRR selectors. Only fitting accepts development utility labels."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .features import LEVELS, PreDecisionPacket
from .kernels import FeatureKernel

ACTIONS = tuple(f"R{i}" for i in range(8))
ALPHA_GRID = (1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def choose(predictions, default, tau=0.005):
    if (
        set(predictions) != set(ACTIONS)
        or default not in ACTIONS
        or not math.isfinite(tau)
        or tau < 0
    ):
        raise ValueError("Registered actions, default and finite tolerance required")
    if not all(math.isfinite(x) for x in predictions.values()):
        raise ValueError("Nonfinite utility prediction")
    top = min(ACTIONS, key=lambda k: (-predictions[k], k))
    margin = predictions[top] - predictions[default]
    return dict(
        recipe=default if margin <= tau else top,
        predicted_best=top,
        margin_to_default=margin,
        predictions=dict(predictions),
        status="POINT_DECISION_NOT_SAFETY_CERTIFICATE",
    )


def krr_fit(kernel, targets, alpha):
    k, y = np.asarray(kernel, float), np.asarray(targets, float)
    if (
        k.ndim != 2
        or k.shape[0] != k.shape[1]
        or not len(k)
        or y.ndim not in (1, 2)
        or len(y) != len(k)
    ):
        raise ValueError("Aligned square kernel and targets required")
    if not np.isfinite(k).all() or not np.isfinite(y).all() or not np.allclose(k, k.T, atol=1e-10):
        raise ValueError("Finite symmetric kernel required")
    if not math.isfinite(alpha) or alpha <= 0 or np.linalg.eigvalsh(k).min() < -1e-9:
        raise ValueError("PSD kernel and positive regularization required")
    mean = y.mean(axis=0)
    return mean, np.linalg.solve(k + len(k) * alpha * np.eye(len(k)), y - mean)


def _utilities(packets, values):
    if isinstance(values, dict):
        if set(values) != {p.origin_id for p in packets}:
            raise ValueError("Utilities must match the complete origin blocks")
        if any(set(values[p.origin_id]) != set(ACTIONS) for p in packets):
            raise ValueError("All eight actions required per origin")
        values = [[values[p.origin_id][action] for action in ACTIONS] for p in packets]
    a = np.asarray(values, float)
    if (
        a.shape != (len(packets), len(ACTIONS))
        or not np.isfinite(a).all()
        or (a < 0).any()
        or (a > 1).any()
    ):
        raise ValueError("Complete finite macro-pX utility matrix required")
    return a


def _check_packets(packets):
    if not packets or any(not isinstance(p, PreDecisionPacket) for p in packets):
        raise ValueError("Fitting requires typed predecision packets")
    if len({p.origin_id for p in packets}) != len(packets):
        raise ValueError("Duplicate origin is not an independent fit context")
    if len({p.feature_level for p in packets}) != 1:
        raise ValueError("Mixed feature layers")


def best_static(packets, utilities):
    packets = list(packets)
    _check_packets(packets)
    utility = _utilities(packets, utilities)
    # Weight independent source lineages equally, and their registered anchors equally.
    lineages = sorted({p.lineage_id for p in packets})
    means = [
        utility[[i for i, p in enumerate(packets) if p.lineage_id == lineage]].mean(axis=0)
        for lineage in lineages
    ]
    score = np.asarray(means).mean(axis=0)
    return min(ACTIONS, key=lambda a: (-score[ACTIONS.index(a)], a))


@dataclass(frozen=True)
class FrozenSelector:
    """Serialized immutable model: no training/update or outcome-loading method."""

    payload_json: str

    def __post_init__(self):
        data = json.loads(self.payload_json)
        required = {
            "schema",
            "selector_id",
            "feature_level",
            "best_static",
            "alpha",
            "practical_tie",
            "kernel",
            "training_packets",
            "mean",
            "beta",
            "tuning_scores",
            "status",
        }
        if (
            set(data) != required
            or data["schema"] != "prospective-selector-v1"
            or data["status"] != "FROZEN"
        ):
            raise ValueError("Frozen selector schema mismatch")
        content = {k: v for k, v in data.items() if k != "selector_id"}
        if data["selector_id"] != _identifier(content):
            raise ValueError("Frozen selector identity mismatch")
        if (
            data["feature_level"] not in LEVELS
            or data["best_static"] not in ACTIONS
            or data["alpha"] not in ALPHA_GRID
        ):
            raise ValueError("Unregistered frozen rule")
        if data["practical_tie"] != 0.005:
            raise ValueError("Preregistered practical tie must remain 0.005")
        kernel = FeatureKernel.from_dict(data["kernel"])
        packets = [PreDecisionPacket.from_dict(p) for p in data["training_packets"]]
        _check_packets(packets)
        if kernel.feature_level != data["feature_level"] or any(
            p.feature_level != data["feature_level"] for p in packets
        ):
            raise ValueError("Frozen selector feature layer mismatch")
        mean, beta = np.asarray(data["mean"]), np.asarray(data["beta"])
        if (
            mean.shape != (8,)
            or beta.shape != (len(packets), 8)
            or not np.isfinite(mean).all()
            or not np.isfinite(beta).all()
        ):
            raise ValueError("Frozen coefficient shape/nonfinite error")
        if mean[0] != 0 or np.any(beta[:, 0] != 0):
            raise ValueError("R0 target must be identically zero")

    @property
    def selector_id(self):
        return self.to_dict()["selector_id"]

    @property
    def feature_level(self):
        return self.to_dict()["feature_level"]

    def to_dict(self):
        return json.loads(self.payload_json)

    @classmethod
    def from_dict(cls, data):
        return cls(_canonical(data))

    @classmethod
    def fit(cls, packets, utilities, *, alpha, default_action, tuning_scores=None):
        packets = list(packets)
        _check_packets(packets)
        if alpha not in ALPHA_GRID or default_action not in ACTIONS:
            raise ValueError("Unregistered alpha/default")
        utility = _utilities(packets, utilities)
        kernel = FeatureKernel.fit(packets)
        targets = utility - utility[:, :1]
        mean, beta = krr_fit(kernel.gram(packets), targets, alpha)
        data = dict(
            schema="prospective-selector-v1",
            feature_level=packets[0].feature_level,
            best_static=default_action,
            alpha=alpha,
            practical_tie=0.005,
            kernel=kernel.to_dict(),
            training_packets=[p.to_dict() for p in packets],
            mean=mean.tolist(),
            beta=beta.tolist(),
            tuning_scores=tuning_scores or [],
            status="FROZEN",
        )
        data["selector_id"] = _identifier(data)
        return cls.from_dict(data)

    def predict(self, packet):
        if not isinstance(packet, PreDecisionPacket):
            raise TypeError("Inference accepts only PreDecisionPacket, never paths/outcomes")
        data = self.to_dict()
        kernel = FeatureKernel.from_dict(data["kernel"])
        train = [PreDecisionPacket.from_dict(d) for d in data["training_packets"]]
        prediction = np.asarray(data["mean"]) + kernel.gram([packet], train)[0] @ np.asarray(
            data["beta"]
        )
        return dict(zip(ACTIONS, prediction.tolist(), strict=True))

    def choose_action(self, packet):
        if not isinstance(packet, PreDecisionPacket):
            raise TypeError("Inference accepts only PreDecisionPacket, never paths/outcomes")
        data = self.to_dict()
        try:
            predictions = self.predict(packet)
        except ValueError as exc:
            # A broken or unobserved registered panel cannot be repaired with future labels.
            return dict(
                selector_id=self.selector_id,
                origin_id=packet.origin_id,
                recipe=data["best_static"],
                predictions=None,
                predicted_best=None,
                margin_to_default=None,
                status="FALLBACK_BEST_STATIC",
                reason=str(exc),
            )
        return dict(
            selector_id=self.selector_id,
            origin_id=packet.origin_id,
            **choose(predictions, data["best_static"], data["practical_tie"]),
            reason=None,
        )


def tune_selector(
    fit_packets, fit_utilities, tuning_packets, tuning_utilities, *, default_action=None, refit=True
):
    """Tune only on held-out development lineages' selected-action J, not R²."""
    fit_packets, tuning_packets = list(fit_packets), list(tuning_packets)
    _check_packets(fit_packets)
    _check_packets(tuning_packets)
    if {p.lineage_id for p in fit_packets} & {p.lineage_id for p in tuning_packets}:
        raise ValueError("Lineage leakage between fit and tuning")
    if fit_packets[0].feature_level != tuning_packets[0].feature_level:
        raise ValueError("Feature permission mismatch")
    fit_u, tune_u = (
        _utilities(fit_packets, fit_utilities),
        _utilities(tuning_packets, tuning_utilities),
    )
    all_packets = fit_packets + tuning_packets
    all_u = np.concatenate((fit_u, tune_u))
    default = default_action or best_static(all_packets, all_u)
    scores = []
    models = {}
    for alpha in ALPHA_GRID:
        model = FrozenSelector.fit(fit_packets, fit_u, alpha=alpha, default_action=default)
        selected = [model.choose_action(p)["recipe"] for p in tuning_packets]
        gains = [tune_u[i, ACTIONS.index(action)] for i, action in enumerate(selected)]
        lineage_scores = [
            np.mean(
                [g for g, p in zip(gains, tuning_packets, strict=True) if p.lineage_id == lineage]
            )
            for lineage in sorted({p.lineage_id for p in tuning_packets})
        ]
        score = float(np.mean(lineage_scores))
        scores.append(dict(alpha=alpha, selected_action_utility=score, selected=selected))
        models[alpha] = model
    peak = max(s["selected_action_utility"] for s in scores)
    alpha = max(s["alpha"] for s in scores if peak - s["selected_action_utility"] < 0.001 + 1e-12)
    return FrozenSelector.fit(
        all_packets if refit else fit_packets,
        all_u if refit else fit_u,
        alpha=alpha,
        default_action=default,
        tuning_scores=scores,
    )


def leave_lineage_out(packets, utilities, *, alpha, default_action):
    """Development diagnostic with fixed rules and whole-lineage holdout.

    No alpha/default retuning uses the held-out outcomes. This is not an
    independent prospective test, and both anchors stay in the same fold.
    """
    packets = list(packets)
    _check_packets(packets)
    values = _utilities(packets, utilities)
    lineages = sorted({p.lineage_id for p in packets})
    if len(lineages) < 2:
        raise ValueError("At least two independent lineages required")
    records = []
    for held in lineages:
        fit_indices = [i for i, p in enumerate(packets) if p.lineage_id != held]
        held_indices = [i for i, p in enumerate(packets) if p.lineage_id == held]
        model = FrozenSelector.fit(
            [packets[i] for i in fit_indices],
            values[fit_indices],
            alpha=alpha,
            default_action=default_action,
        )
        for i in held_indices:
            decision = model.choose_action(packets[i])
            records.append(
                dict(
                    origin_id=packets[i].origin_id,
                    lineage_id=held,
                    origin_step=packets[i].to_dict()["step"],
                    source_recipe=packets[i].to_dict()["source_recipe"],
                    recipe=decision["recipe"],
                    predictions=decision["predictions"],
                    selected_utility=float(values[i, ACTIONS.index(decision["recipe"])]),
                    best_static_utility=float(values[i, ACTIONS.index(default_action)]),
                    fit_lineages=[lineage for lineage in lineages if lineage != held],
                    alpha=alpha,
                    status="DEVELOPMENT_LEAVE_LINEAGE_OUT_DIAGNOSTIC",
                )
            )
    return records
