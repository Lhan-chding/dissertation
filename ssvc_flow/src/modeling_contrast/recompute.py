"""Read-only reconstruction of experiment tables from hash-bound raw arrays.

This verifier does not call production scoring/metric helpers. It reuses the
lossless paid-packet decoder/reconstructor and, only for independent N1 exact
covariance recovery, the unchanged frozen CPU forward. New full-action scores
are separately metered and retained. No sampling, Jacobian or optimizer is used.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from contextlib import suppress
from pathlib import Path

import numpy as np

TARGETS = ("joint_1_minus_joint_0", "no_x_off_1_minus_joint_0", "joint_1_minus_no_x_off_1")
H = np.array(
    [
        [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
        [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
        [0, -2 / np.sqrt(6), 1 / np.sqrt(12)],
        [0, 0, -3 / np.sqrt(12)],
    ]
)
MODEL_FIELDS = (
    "case_count",
    "event_error_ss",
    "event_truth_ss",
    "event_mae",
    "predicted_difference_norm",
    "pX_error_ss",
    "pX_truth_ss",
    "v_error_ss",
    "v_truth_ss",
    "pX_abs_errors",
    "v_abs_errors",
    "event_nrmse",
    "pX_nrmse",
    "v_nrmse",
)
MODEL_KEYS = (
    "stage",
    "configuration_id",
    "seed",
    "arm",
    "anchor",
    "noise_replica",
    "target",
    "population",
)
OBS_KEYS = ("trajectory_id", "anchor", "bank", "method", "n", "target", "view")
OBS_FIELDS = (
    "replicas",
    "bias",
    "rms_bias",
    "rmse",
    "empirical_variance",
    "pX_error_ss",
    "pX_truth_ss",
    "v_error_ss",
    "v_truth_ss",
    "mass_residual_rms",
)


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _key(row, fields=MODEL_KEYS):
    return tuple(str(row.get(name, "")) for name in fields)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            )


def _group_mean(values, metadata):
    values = np.asarray(values)
    groups = np.asarray([m["group"] for m in metadata])
    weights = np.asarray([m["weight"] for m in metadata], dtype=np.float64)
    if values.shape[-1] != len(groups) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("fixed prompt identities/weights do not match response coordinates")
    means = []
    for group in sorted(set(groups.tolist())):
        selected = weights[groups == group]
        if selected.sum() <= 0:
            raise ValueError("group has no positive fixed weight")
        part = values[..., groups == group]
        means.append(
            part.mean(axis=-1)
            if np.all(selected == selected[0])
            else np.sum(part * (selected / selected.sum()), axis=-1)
        )
    return np.stack(means, axis=-1)


def independent_model_statistics(prediction, truth, metadata, fit_updates, identity):
    """Compute all/active scores from probability-unit arrays independently."""
    pred, truth = np.asarray(prediction), np.asarray(truth)
    fit = np.asarray(fit_updates)
    if (
        pred.shape != truth.shape
        or pred.ndim != 4
        or pred.shape[1] != 2
        or pred.shape[-1] != 4
        or fit.ndim != 3
        or fit.shape[1] != 2
        or not np.isfinite(pred).all()
        or not np.isfinite(truth).all()
        or not np.isfinite(fit).all()
    ):
        raise ValueError(
            "aligned bank/two-target/prompt/event arrays and fit-only increments required"
        )
    excitation = [bool(np.any(fit[:, i] != 0)) for i in range(2)]
    excitation.append(bool(np.any(fit[:, 0] - fit[:, 1] != 0)))
    result = []
    for i, target in enumerate(TARGETS):
        p = pred[:, i] if i < 2 else pred[:, 0] - pred[:, 1]
        t = truth[:, i] if i < 2 else truth[:, 0] - truth[:, 1]
        error = p - t
        x_error, v_error = (
            _group_mean(error[..., 0], metadata),
            _group_mean(-error[..., 3], metadata),
        )
        x_truth, v_truth = _group_mean(t[..., 0], metadata), _group_mean(-t[..., 3], metadata)
        scores = {
            "case_count": len(p),
            "event_error_ss": float(np.sum(error * error)),
            "event_truth_ss": float(np.sum(t * t)),
            "event_mae": float(np.mean(np.abs(error))),
            "predicted_difference_norm": float(np.linalg.norm(p)),
            "pX_error_ss": float(np.sum(x_error * x_error)),
            "pX_truth_ss": float(np.sum(x_truth * x_truth)),
            "v_error_ss": float(np.sum(v_error * v_error)),
            "v_truth_ss": float(np.sum(v_truth * v_truth)),
            "pX_abs_errors": np.abs(x_error).ravel().tolist(),
            "v_abs_errors": np.abs(v_error).ravel().tolist(),
        }
        for metric in ("event", "pX", "v"):
            energy = scores[f"{metric}_truth_ss"]
            scores[f"{metric}_nrmse"] = (
                float(np.sqrt(scores[f"{metric}_error_ss"] / energy)) if energy > 0 else None
            )
        for population in ["all", "active"] if excitation[i] else ["all"]:
            result.append(dict(identity, target=target, population=population, **scores))
    return result


def independent_observation_statistics(prediction, raw, truth, metadata, view):
    """Across-replica bias, MSE and variance with the correct raw-LR validity map."""
    pred, raw, truth = map(np.asarray, (prediction, raw, truth))
    if (
        pred.ndim != 3
        or pred.shape != raw.shape
        or truth.shape != pred.shape[1:]
        or pred.shape[-1] != 4
    ):
        raise ValueError("replica/prompt/four-event predictions and aligned truth required")
    if view not in ("raw_event", "helmert") or len(pred) < 2:
        raise ValueError("recognized event view and at least two independent replicas required")
    error = pred - truth
    bias = pred.mean(axis=0) - truth
    gx = _group_mean(pred[..., 0], metadata)
    valid = pred[..., :3].sum(axis=-1) if view == "raw_event" else -pred[..., 3]
    gv = _group_mean(valid, metadata)
    tx, tv = _group_mean(truth[..., 0], metadata), _group_mean(-truth[..., 3], metadata)
    mse = float(np.mean(error**2))
    return {
        "replicas": len(pred),
        "bias": float(np.mean(bias)),
        "rms_bias": float(np.sqrt(np.mean(bias**2))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "empirical_variance": float(np.var(pred, axis=0, ddof=1).mean()),
        "pX_error_ss": float(np.sum((gx - tx) ** 2)),
        "pX_truth_ss": float(len(pred) * np.sum(tx**2)),
        "v_error_ss": float(np.sum((gv - tv) ** 2)),
        "v_truth_ss": float(len(pred) * np.sum(tv**2)),
        "mass_residual_rms": float(np.sqrt(np.mean(raw.sum(axis=-1) ** 2))),
    }


def project_contrast_diagnostic(raw_contrast, baseline_probability, *, tolerance=1e-12):
    """Euclidean simplex projection of invalid evaluation levels, diagnostic only.

    Numerically valid rows remain bitwise unchanged. No projection is used to
    select an estimator, compute its fit, or replace its sealed raw prediction.
    """
    raw = np.asarray(raw_contrast, dtype=np.float64)
    baseline = np.asarray(baseline_probability, dtype=np.float64)
    if (
        raw.ndim != 4
        or raw.shape[1] != 2
        or raw.shape[-1] != 4
        or baseline.shape != (raw.shape[0], raw.shape[2], 4)
        or not np.isfinite(raw).all()
        or not np.isfinite(baseline).all()
        or np.any(baseline < 0)
        or not np.allclose(baseline.sum(-1), 1, atol=tolerance, rtol=0)
    ):
        raise ValueError("finite raw contrasts and aligned valid oracle baseline levels required")
    levels = raw + baseline[:, None]
    negative = levels < 0
    changed = np.any(negative, axis=-1) | (np.abs(levels.sum(-1) - 1) > tolerance)
    indices = np.argwhere(changed)
    projected = raw.copy()
    if len(indices):
        selected = levels[changed]
        ordered = np.sort(selected, axis=-1)[:, ::-1]
        cumulative = np.cumsum(ordered, axis=-1) - 1
        dimension = np.arange(1, 5)
        active = (ordered - cumulative / dimension) > 0
        rho = active.sum(-1) - 1
        threshold = cumulative[np.arange(len(selected)), rho] / (rho + 1)
        simplex = np.maximum(selected - threshold[:, None], 0)
        changed_baseline = baseline[indices[:, 0], indices[:, 2]]
        projected[changed] = simplex - changed_baseline
        clipped = int(np.count_nonzero(simplex == 0))
    else:
        clipped = 0
    return {
        "projected": projected,
        "changed_indices": indices,
        "projected_contrast_events": projected[changed],
        "invalid_level_row_count": int(changed.sum()),
        "negative_component_count": int(negative.sum()),
        "zero_components_after_projection": clipped,
        "simplex_tolerance": tolerance,
        "main_ranking_eligible": False,
        "evaluation_oracle_baseline_used": True,
    }


def _counts_raw_covariance(events, policy_ids, n):
    events = np.asarray(events, dtype=float)
    result = np.zeros((events.shape[1], 8, 8))
    for policy in dict.fromkeys(policy_ids):
        first = policy_ids.index(policy)
        coefficients = np.array(
            [int(policy_ids[j] == policy) - int(policy_ids[0] == policy) for j in (1, 2)]
        )
        for prompt, prob in enumerate(events[first]):
            level = np.diag(prob) - prob[:, None] * prob[None, :]
            result[prompt] += np.kron(np.outer(coefficients, coefficients), level) / n
    return result


def independent_raw_covariance(
    action_tables, labels, method, n, *, policy_ids=("b", "u", "v"), origin=None
):
    """Exact joint contribution covariance from genuine full action laws.

    CRN integrates the shared uniform over the union of ordered action CDFs.
    LR computes centered second moments under the stated proposal, and mixture
    packets share contributions only for an identical unordered policy pair.
    """
    tables = np.asarray(action_tables, dtype=float)
    labels = np.asarray(labels)
    if (
        tables.ndim != 3
        or tables.shape[0] != 3
        or labels.shape != tables.shape[1:]
        or len(policy_ids) != 3
        or not np.isfinite(tables).all()
        or np.any(tables < 0)
        or not np.allclose(tables.sum(-1), 1, atol=1e-12, rtol=0)
        or not isinstance(n, int)
        or n < 1
        or labels.dtype.kind not in "iu"
        or np.any((labels < 0) | (labels > 3))
    ):
        raise ValueError(
            "three complete action probability laws and aligned event identities required"
        )
    policy_ids = tuple(policy_ids)
    features = np.eye(4)[labels]
    if method == "O_IND":
        return _counts_raw_covariance(np.einsum("kpa,pae->kpe", tables, features), policy_ids, n)
    result = np.zeros((tables.shape[1], 8, 8))
    if method == "O_LR_ORIGIN":
        origin = np.asarray(origin, dtype=float)
        if (
            origin.shape != labels.shape
            or not np.isfinite(origin).all()
            or np.any(origin < 0)
            or not np.allclose(origin.sum(-1), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError("complete origin proposal probability law required")
    for prompt in range(tables.shape[1]):
        probs = tables[:, prompt]
        if method == "O_CRN":
            cdf = np.cumsum(probs, axis=-1)
            edges = np.unique(np.clip(np.r_[0.0, 1.0, cdf.ravel()], 0, 1))
            mass = np.diff(edges)
            mid = (edges[1:] + edges[:-1]) / 2
            actions = np.stack(
                [np.searchsorted(c, mid, side="right").clip(0, tables.shape[-1] - 1) for c in cdf]
            )
            sampled = features[prompt][actions]
            contribution = np.concatenate(
                (sampled[1] - sampled[0], sampled[2] - sampled[0]), axis=-1
            )
            centered = contribution - mass @ contribution
            result[prompt] = (centered.T * mass) @ centered / n
        elif method in ("O_LR_ORIGIN", "O_LR_MIX"):
            if method == "O_LR_ORIGIN":
                groups = {"origin": [0, 1]}
            else:
                groups = {}
                for j in range(2):
                    groups.setdefault(tuple(sorted((policy_ids[0], policy_ids[j + 1]))), []).append(
                        j
                    )
            for indices in groups.values():
                rho = (
                    origin[prompt]
                    if method == "O_LR_ORIGIN"
                    else (probs[0] + probs[indices[0] + 1]) / 2
                )
                contribution = np.zeros((tables.shape[-1], 8))
                for j in indices:
                    difference = probs[j + 1] - probs[0]
                    if np.any((rho == 0) & (difference != 0)):
                        raise ValueError("BLOCKED_SUPPORT_MISMATCH")
                    weights = np.divide(difference, rho, out=np.zeros_like(rho), where=rho > 0)
                    contribution[:, 4 * j : 4 * j + 4] = weights[:, None] * features[prompt]
                centered = contribution - rho @ contribution
                result[prompt] += (centered.T * rho) @ centered / n
        else:
            raise ValueError("unknown observation covariance method")
    return result


class _Verifier:
    def __init__(self, run_root, parent_root, out):
        self.run, self.parent, self.out = map(
            lambda p: Path(p).resolve(), (run_root, parent_root, out)
        )
        if self.out == self.run or self.parent == self.out or self.parent in self.out.parents:
            raise ValueError("verification output must not overwrite the run root or parent inputs")
        self.out.mkdir(parents=True, exist_ok=False)
        self.coverage, self.errors, self.checks = {}, [], {}
        self.scalar_comparisons, self.max_diff = 0, 0.0
        self.expected_n1, self.expected_n2 = {}, {}
        self.parent_cache, self.packet_cache = {}, None
        self.action_cache, self.raw_cov_cache, self.origin_cache = {}, {}, {}
        self.dataset = None
        self.forward_cost = {
            "batched_model_forward_calls": 0,
            "physical_policy_forwards": 0,
            "complete_action_score_values": 0,
            "forward_seconds": 0.0,
            "optimizer_updates": 0,
        }
        self.projection_totals = {
            "units": 0,
            "raw_prediction_arrays": 0,
            "prompt_contrast_rows": 0,
            "invalid_level_rows": 0,
            "negative_components": 0,
            "sparse_artifact_bytes": 0,
        }
        self.manifest = self.json(self.parent / "manifest.json", "parent_manifest")
        self.metadata = self.json(self.parent / "probe_metadata.json", "fixed_probe_metadata")
        if len({m["prompt_id"] for m in self.metadata}) != len(self.metadata):
            raise ValueError("duplicate fixed prompt identity")
        self.entries = {e["id"]: e for e in self.manifest["trajectories"]}

    def cover(self, path, role, expected=None):
        path = Path(path).resolve()
        name = str(path)
        if name not in self.coverage:
            stat = path.stat()
            self.coverage[name] = {
                "path": name,
                "bytes": stat.st_size,
                "sha256": _hash(path),
                "mtime_ns": stat.st_mtime_ns,
                "roles": [],
            }
        item = self.coverage[name]
        if role not in item["roles"]:
            item["roles"].append(role)
        if expected is not None and expected != item["sha256"]:
            raise ValueError(f"hash mismatch for {path}")
        return item["sha256"]

    def json(self, path, role, expected=None):
        self.cover(path, role, expected)
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as stream:
            return json.load(stream)

    def records(self, path, role):
        self.cover(path, role)
        if str(path).endswith(".csv"):
            with open(path, newline="") as stream:
                yield from csv.DictReader(stream)
        else:
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    yield json.loads(line)

    def bound_metadata(self, mini_path, default_file, role):
        mini = self.json(mini_path, role + "_binding")
        relative = mini.get("metadata_file", default_file)
        extra = (mini_path.parent / relative).resolve()
        if extra.parent != mini_path.parent.resolve():
            raise ValueError("metadata pointer escapes unit directory")
        if extra.exists():
            if "metadata_sha256" not in mini:
                raise ValueError("compressed metadata lacks hash binding")
            return self.json(extra, role, mini["metadata_sha256"])
        if "metadata_sha256" in mini:
            raise ValueError("hash-bound compressed metadata is missing")
        return mini

    def parent_arrays(self, trajectory):
        if trajectory in self.parent_cache:
            return self.parent_cache[trajectory]
        entry = self.entries[trajectory]
        result = []
        for field, name in (("observations_file", "branch_theta"), ("oracle_file", "branch_p")):
            path = (self.parent / entry[field]).resolve()
            if self.parent not in path.parents:
                raise ValueError("historical artifact escapes parent root")
            self.cover(path, field, entry["sha256"][field])
            with np.load(path, allow_pickle=False) as arrays:
                result.append(arrays[name].copy())
        self.parent_cache[trajectory] = tuple(result)
        return tuple(result)

    def packet(self, path):
        path = Path(path).resolve()
        if self.packet_cache is not None and self.packet_cache[0] == path:
            return self.packet_cache[1]
        from .packet_codec import decode_arrays
        from .packet_reconstruction import reconstruct_measurements

        meta = self.bound_metadata(
            path / "packets.json", "packet_metadata.json.gz", "paid_packet_metadata"
        )
        if "trajectory_id" in meta:
            entry = self.entries[meta["trajectory_id"]]
            if meta.get("input_parameters_sha256") != entry["sha256"]["observations_file"]:
                raise ValueError("paid packet parameter identity differs from parent manifest")
            if meta.get("input_counts_sha256") is not None:
                count_path = (self.parent / entry["counts_file"]).resolve()
                if self.parent not in count_path.parents:
                    raise ValueError("counts path escapes parent root")
                self.cover(count_path, "historical_paid_counts_source", meta["input_counts_sha256"])
        sha = self.cover(
            path / "packet_arrays.npz", "paid_packet_primitives", meta["packet_arrays_sha256"]
        )
        with np.load(path / "packet_arrays.npz", allow_pickle=False) as arrays:
            decoded = decode_arrays(arrays)
        # Force primitive replay even if a legacy archive also stores summaries.
        can_replay = bool(meta.get("packets")) and all(
            "sample_layout" in row for row in meta["packets"]
        )
        if not can_replay:
            raise ValueError(
                "paid primitive layout missing; derived summaries alone "
                "are not independent evidence"
            )
        value = reconstruct_measurements(decoded, meta, force=True)
        value.update(metadata=meta, artifact_sha256=sha)
        self.packet_cache = path, value
        return value

    def action_table(self, theta, identity):
        fingerprint = _array_hash(np.asarray(theta))
        if fingerprint in self.action_cache:
            return self.action_cache[fingerprint]["table"]
        from src.modeling_qualification.toy import _numpy_forward

        self.cover(
            Path(__file__).resolve().parents[1] / "modeling_qualification/toy.py",
            "frozen_forward_source",
        )
        if self.dataset is None:
            path = self.parent / "dataset.npz"
            self.cover(path, "frozen_forward_features_labels", self.manifest["dataset_sha256"])
            with np.load(path, allow_pickle=False) as arrays:
                self.dataset = arrays["probe_features"].copy(), arrays["probe_categories"].copy()
        features, labels = self.dataset
        started = time.perf_counter()
        table = _numpy_forward(np.asarray(theta), features, labels)[1]
        seconds = time.perf_counter() - started
        self.forward_cost["batched_model_forward_calls"] += 1
        self.forward_cost["physical_policy_forwards"] += len(features)
        self.forward_cost["complete_action_score_values"] += int(table.size)
        self.forward_cost["forward_seconds"] += seconds
        self.action_cache[fingerprint] = {
            "table": table,
            "array_name": f"policy_{len(self.action_cache):05d}",
            "theta_sha256": fingerprint,
            "table_sha256": _array_hash(table),
            "first_source_identity": identity,
            "seconds": seconds,
        }
        return table

    def exact_raw_covariance(self, trajectory, ai, bank, method, n):
        key = trajectory, ai, bank, method
        if key not in self.raw_cov_cache:
            theta, events = self.parent_arrays(trajectory)
            parameters = theta[ai, bank]
            ids = tuple(_array_hash(t) for t in parameters)
            if method == "O_IND":
                covariance = _counts_raw_covariance(events[ai, bank], ids, 1)
            else:
                tables = np.stack(
                    [
                        self.action_table(
                            t,
                            {
                                "trajectory": trajectory,
                                "anchor_index": ai,
                                "bank": bank,
                                "operation_index": op,
                            },
                        )
                        for op, t in enumerate(parameters)
                    ]
                )
                origin = None
                if method == "O_LR_ORIGIN":
                    if trajectory not in self.origin_cache:
                        with np.load(
                            self.parent / self.entries[trajectory]["observations_file"],
                            allow_pickle=False,
                        ) as arrays:
                            self.origin_cache[trajectory] = arrays["theta"].copy()
                    step = self.manifest["anchors"][ai]
                    origin = self.action_table(
                        self.origin_cache[trajectory][step],
                        {"trajectory": trajectory, "origin_step": step},
                    )
                covariance = independent_raw_covariance(
                    tables, self.dataset[1], method, 1, policy_ids=ids, origin=origin
                )
            self.raw_cov_cache[key] = covariance
        return self.raw_cov_cache[key] / n

    def compare(self, expected, actual, fields, scope):
        matched = True
        for field in fields:
            if field not in actual:
                self.error(scope, f"missing field {field}")
                matched = False
                continue
            left, right = expected[field], actual[field]
            if isinstance(right, str):
                if right == "":
                    right = None
                elif isinstance(left, list):
                    right = json.loads(right)
                elif left is not None:
                    with suppress(ValueError):
                        right = float(right)
            if left is None or right is None:
                good = left is None and right is None
                difference = 0.0 if good else float("inf")
            else:
                try:
                    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
                    difference = (
                        float(np.max(np.abs(a - b)))
                        if a.size and a.shape == b.shape
                        else (0.0 if a.shape == b.shape else float("inf"))
                    )
                    good = (
                        a.shape == b.shape
                        and np.isfinite(a).all()
                        and np.isfinite(b).all()
                        and difference <= 1e-12
                    )
                except (TypeError, ValueError):
                    good, difference = left == right, 0.0
            self.scalar_comparisons += 1
            if np.isfinite(difference):
                self.max_diff = max(self.max_diff, difference)
            if not good:
                self.error(scope, f"{field} differs; absolute_difference={difference}")
                matched = False
        return matched

    def error(self, scope, detail):
        if len(self.errors) < 200:
            self.errors.append({"scope": str(scope), "detail": str(detail)})

    def phase(self, name, paths, callback):
        if not paths:
            self.checks[name] = {"status": "NOT_RUN", "matched_rows": 0}
            return
        before, matched = len(self.errors), 0
        try:
            matched = callback(paths)
        except (
            ValueError,
            KeyError,
            OSError,
            AssertionError,
            StopIteration,
            IndexError,
            TypeError,
        ) as exc:
            self.error(name, str(exc))
        self.checks[name] = {
            "status": "PASS" if len(self.errors) == before else "FAIL",
            "matched_rows": matched,
        }

    def n1(self, paths):
        matched = 0
        for comparison in paths:
            unit = comparison.parent
            trajectory, ai = unit.parent.name, int(unit.name.removeprefix("a"))
            _theta, parent_p = self.parent_arrays(trajectory)
            audit_meta = self.json(unit / "covariance_audit.json", "oracle_covariance_binding")
            audit_file = unit / "oracle_audit.npz"
            self.cover(audit_file, "oracle_truth_covariance", audit_meta["oracle_sha256"])
            with np.load(audit_file, allow_pickle=False) as oracle:
                rows = list(self.records(comparison, "N1_unit_comparison"))
                packets = {}
                for row in rows:
                    method, n, bank = row["method"], int(row["n"]), int(row["bank"])
                    if (
                        row["trajectory_id"] != trajectory
                        or int(row["anchor"]) != self.manifest["anchors"][ai]
                        or int(row["seed"]) != self.entries[trajectory]["seed"]
                        or row["arm"] != self.entries[trajectory]["arm"]
                    ):
                        raise ValueError("N1 table identity differs from its source unit")
                    pair = method, n
                    if pair not in packets:
                        packets[pair] = (
                            self.run / "N1/packets" / trajectory / f"a{ai}" / f"{method}_n{n}"
                        )
                    packet = self.packet(packets[pair])
                    prefix = f"{method}_n{n}_b{bank}"
                    truth = oracle[prefix + "_truth"]
                    exact = parent_p[ai, bank, 1:] - parent_p[ai, bank, :1]
                    if not np.allclose(truth, exact, atol=1e-12, rtol=0):
                        raise ValueError(f"N1 oracle truth differs from parent endpoints: {prefix}")
                    target = TARGETS.index(row["target"])
                    raw = packet["raw"][:, bank, target]
                    pred = (
                        packet["helmert"][:, bank, target] @ H.T
                        if row["view"] == "helmert"
                        else raw
                    )
                    expected = independent_observation_statistics(
                        pred, raw, truth[target], self.metadata, row["view"]
                    )
                    fields = list(OBS_FIELDS)
                    raw_covariance = self.exact_raw_covariance(trajectory, ai, bank, method, n)
                    sl = slice(4 * target, 4 * target + 4)
                    if row["view"] == "helmert":
                        transform = np.kron(np.eye(2), H)
                        covariance = oracle[prefix + "_covariance"]
                        if (
                            covariance.shape != (len(self.metadata), 6, 6)
                            or not np.isfinite(covariance).all()
                        ):
                            raise ValueError("invalid oracle joint covariance shape or values")
                        restored_helmert = np.stack(
                            [transform.T @ c @ transform for c in raw_covariance]
                        )
                        if not np.allclose(covariance, restored_helmert, atol=1e-12, rtol=0):
                            raise ValueError(
                                "saved oracle covariance differs from "
                                "independent raw contribution moments"
                            )
                        event_cov = np.stack(
                            [transform @ c @ transform.T for c in restored_helmert]
                        )
                        theory = float(
                            np.trace(event_cov[:, sl, sl], axis1=-2, axis2=-1).mean() / 4
                        )
                    else:
                        theory = float(
                            np.trace(raw_covariance[:, sl, sl], axis1=-2, axis2=-1).mean() / 4
                        )
                    expected["theoretical_variance"] = theory
                    expected["empirical_to_theoretical_variance"] = (
                        expected["empirical_variance"] / theory if theory > 0 else None
                    )
                    fields.extend(("theoretical_variance", "empirical_to_theoretical_variance"))
                    if row.get("packet_sha256") != packet["artifact_sha256"]:
                        raise ValueError("N1 row paid packet hash mismatch")
                    matched += self.compare(expected, row, fields, comparison)
                    self.expected_n1[_key(row, OBS_KEYS)] = expected
        return matched

    def n1_final(self, paths):
        matched, seen = 0, set()
        for row in self.records(paths[0], "N1_final_comparison"):
            key = _key(row, OBS_KEYS)
            if key not in self.expected_n1:
                self.error(paths[0], f"row lacks independently replayed source: {key}")
                continue
            expected = self.expected_n1[key]
            fields = [
                *OBS_FIELDS,
                "theoretical_variance",
                "empirical_to_theoretical_variance",
            ]
            matched += self.compare(expected, row, fields, paths[0])
            seen.add(key)
        if seen != set(self.expected_n1):
            self.error(paths[0], "N1 final rows omit independently replayed unit rows")
        return matched

    def n2(self, paths):
        matched = 0
        for mini_path in paths:
            unit = mini_path.parent
            trajectory, ai = unit.parent.name, int(unit.name.removeprefix("a"))
            theta, parent_p = self.parent_arrays(trajectory)
            meta = self.bound_metadata(mini_path, "freeze_metadata.json.gz", "prediction_freeze")
            self.cover(unit / "predictions.npz", "frozen_predictions", meta["prediction_sha256"])
            if meta["parameters_sha256"] != self.entries[trajectory]["sha256"]["observations_file"]:
                raise ValueError("prediction parameter source hash mismatch")
            with np.load(unit / "predictions.npz", allow_pickle=False) as data:
                predictions = data["prediction"]
            if len(predictions) != len(meta["models"]):
                raise ValueError("prediction/model identity count differs")
            truth = parent_p[ai, 10:14, 1:] - parent_p[ai, 10:14, :1]
            expected_unit = {}
            changed_indices, changed_values, projection_rows, projection_models = [], [], [], []
            for model_index, (pred, ident) in enumerate(
                zip(predictions, meta["models"], strict=True)
            ):
                if pred.dtype != np.float64 or _array_hash(pred) != ident["prediction_hash"]:
                    raise ValueError("frozen prediction array hash mismatch")
                if (
                    ident["seed"] != self.entries[trajectory]["seed"]
                    or ident["arm"] != self.entries[trajectory]["arm"]
                    or ident["anchor"] != self.manifest["anchors"][ai]
                ):
                    raise ValueError("N2 model identity differs from its source unit")
                bank_count = int(ident["fit_banks"])
                fit = theta[ai, :bank_count, 1:] - theta[ai, :bank_count, :1]
                raw_events = pred @ H.T
                raw_rows = independent_model_statistics(
                    raw_events, truth, self.metadata, fit, ident
                )
                for row in raw_rows:
                    expected_unit[_key(row)] = row
                projection = project_contrast_diagnostic(raw_events, parent_p[ai, 10:14, 0])
                count = projection["invalid_level_row_count"]
                if count:
                    changed_indices.append(
                        np.column_stack(
                            (np.full(count, model_index), projection["changed_indices"])
                        )
                    )
                    changed_values.append(projection["projected_contrast_events"])
                    projected_rows = independent_model_statistics(
                        projection["projected"], truth, self.metadata, fit, ident
                    )
                else:
                    projected_rows = raw_rows
                for raw_row, projected_row in zip(raw_rows, projected_rows, strict=True):
                    compact = {k: raw_row[k] for k in MODEL_KEYS}
                    compact.update(
                        method=ident["method"],
                        observation=ident["observation"],
                        n=ident["n"],
                        main_ranking_eligible=False,
                        evaluation_oracle_baseline_used=True,
                    )
                    for metric in ("event", "pX", "v"):
                        for field in ("error_ss", "truth_ss", "nrmse"):
                            compact[f"raw_{metric}_{field}"] = raw_row[f"{metric}_{field}"]
                            compact[f"projected_{metric}_{field}"] = projected_row[
                                f"{metric}_{field}"
                            ]
                    projection_rows.append(compact)
                projection_models.append(
                    {
                        "model_index": model_index,
                        "raw_prediction_sha256": ident["prediction_hash"],
                        "raw_event_array_sha256": _array_hash(raw_events),
                        "projected_event_array_sha256": _array_hash(projection["projected"]),
                        "invalid_level_rows": count,
                        "negative_components": projection["negative_component_count"],
                        "zero_components_after_projection": projection[
                            "zero_components_after_projection"
                        ],
                    }
                )
                self.projection_totals["raw_prediction_arrays"] += 1
                self.projection_totals["prompt_contrast_rows"] += int(
                    np.prod(raw_events.shape[:-1])
                )
                self.projection_totals["invalid_level_rows"] += count
                self.projection_totals["negative_components"] += projection[
                    "negative_component_count"
                ]
            projection_out = self.out / "projection" / unit.relative_to(self.run / "N2")
            projection_out.mkdir(parents=True, exist_ok=False)
            sparse = projection_out / "projected_sparse.npz"
            np.savez_compressed(
                sparse,
                changed_indices=np.vstack(changed_indices).astype(np.int64)
                if changed_indices
                else np.empty((0, 4), dtype=np.int64),
                projected_contrast_events=np.vstack(changed_values)
                if changed_values
                else np.empty((0, 4)),
                helmert_to_event_matrix=H.T,
            )
            with np.load(sparse, allow_pickle=False) as saved:
                indices, values = saved["changed_indices"], saved["projected_contrast_events"]
                for record, prediction in zip(projection_models, predictions, strict=True):
                    restored = prediction @ saved["helmert_to_event_matrix"]
                    selected = indices[:, 0] == record["model_index"]
                    if selected.any():
                        restored[tuple(indices[selected, 1:].T)] = values[selected]
                    if _array_hash(restored) != record["projected_event_array_sha256"]:
                        raise ValueError(
                            "sparse projected prediction failed bitwise reconstruction"
                        )
            with gzip.open(projection_out / "raw_projected_errors.jsonl.gz", "wt") as stream:
                for row in projection_rows:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            _write_json(
                projection_out / "PROJECTION_BINDING.json",
                {
                    "main_ranking_eligible": False,
                    "posthoc_oracle_baseline_diagnostic": True,
                    "raw_prediction_file": str(unit / "predictions.npz"),
                    "raw_prediction_file_sha256": meta["prediction_sha256"],
                    "oracle_baseline_file": str(
                        self.parent / self.entries[trajectory]["oracle_file"]
                    ),
                    "oracle_baseline_file_sha256": self.entries[trajectory]["sha256"][
                        "oracle_file"
                    ],
                    "baseline_array": "branch_p",
                    "anchor_index": ai,
                    "baseline_banks": [10, 11, 12, 13],
                    "baseline_operation_index": 0,
                    "changed_indices_order": [
                        "model_index",
                        "evaluation_bank_index",
                        "target_index",
                        "prompt_index",
                    ],
                    "unchanged_rows": (
                        "reconstruct from sealed raw prediction @ stored helmert_to_event_matrix"
                    ),
                    "changed_rows": (
                        "replace contrast event rows with projected_contrast_events "
                        "without any rounding"
                    ),
                    "simplex_tolerance": 1e-12,
                    "valid_rows_preserved_bitwise": True,
                    "models": projection_models,
                    "all_projected_arrays_bitwise_reconstructed_from_sparse_archive": True,
                    "sparse_sha256": _hash(sparse),
                    "error_table_sha256": _hash(projection_out / "raw_projected_errors.jsonl.gz"),
                },
            )
            self.projection_totals["units"] += 1
            self.projection_totals["sparse_artifact_bytes"] += sum(
                p.stat().st_size for p in projection_out.iterdir()
            )
            seen = set()
            for row in self.records(unit / "metrics.jsonl.gz", "N2_unit_metrics"):
                key = _key(row)
                if key not in expected_unit:
                    self.error(unit, f"unexpected metric row or wrong active mask: {key}")
                    continue
                matched += self.compare(expected_unit[key], row, MODEL_FIELDS, unit)
                seen.add(key)
            if seen != set(expected_unit):
                self.error(unit, "missing raw-derived target/population metric rows")
            self.expected_n2.update(expected_unit)
        return matched

    def direct(self, row):
        trajectory = next(
            e["id"]
            for e in self.entries.values()
            if e["seed"] == row["seed"] and e["arm"] == row["arm"]
        )
        ai = self.manifest["anchors"].index(row["anchor"])
        relative = Path(trajectory) / f"a{ai}" / f"{row['observation']}_n{row['n']}"
        packet_path = self.run / "N1/packets" / relative
        if not packet_path.exists():
            packet_path = self.run / "N2/packets" / relative
        packet = self.packet(packet_path)
        theta, parent_p = self.parent_arrays(trajectory)
        if row["prediction_hash"] != packet["artifact_sha256"]:
            raise ValueError("D_DIRECT paid packet hash mismatch")
        fit = theta[ai, :8, 1:] - theta[ai, :8, :1]
        truth = parent_p[ai, 10:14, 1:] - parent_p[ai, 10:14, :1]
        for noise in range(packet["metadata"]["replicas"]):
            ident = dict(row, noise_replica=noise)
            for expected in independent_model_statistics(
                packet["helmert"][noise, 10:14] @ H.T, truth, self.metadata, fit, ident
            ):
                self.expected_n2[_key(expected)] = expected

    def n2_final(self, paths):
        matched, seen = 0, set()
        for row in self.records(paths[0], "N2_final_metrics"):
            key = _key(row)
            if row["method"] == "D_DIRECT" and key not in self.expected_n2:
                self.direct(row)
            if key not in self.expected_n2:
                self.error(paths[0], f"final row lacks raw source: {key}")
                continue
            matched += self.compare(self.expected_n2[key], row, MODEL_FIELDS, paths[0])
            seen.add(key)
        if seen != set(self.expected_n2):
            self.error(paths[0], "final metrics omit independently reconstructed rows")
        return matched

    def aggregate(self, paths, by_seed=False):
        keys = ("configuration_id", "target", "population") + (("seed",) if by_seed else ())
        buckets = {}
        for row in self.expected_n2.values():
            buckets.setdefault(_key(row, keys), []).append(row)
        expected = {}
        for key, rows in buckets.items():
            item = {
                "seed_count": len({r["seed"] for r in rows}),
                "actual_rank_min": min(r.get("actual_rank", 0) for r in rows),
                "actual_rank_max": max(r.get("actual_rank", 0) for r in rows),
                "k_min": min(r.get("k", 0) for r in rows),
                "k_max": max(r.get("k", 0) for r in rows),
                "predicted_difference_norm": float(
                    np.sqrt(sum(r["predicted_difference_norm"] ** 2 for r in rows))
                ),
            }
            for metric in ("event", "pX", "v"):
                err = sum(r[f"{metric}_error_ss"] for r in rows)
                truth = sum(r[f"{metric}_truth_ss"] for r in rows)
                item.update(
                    {
                        f"{metric}_error_ss": err,
                        f"{metric}_truth_ss": truth,
                        f"{metric}_nrmse": float(np.sqrt(err / truth)) if truth > 0 else None,
                    }
                )
                if metric != "event":
                    absolute = np.concatenate([r[f"{metric}_abs_errors"] for r in rows])
                    item[f"{metric}_error_q95"] = float(np.quantile(absolute, 0.95))
                    item[f"{metric}_mae"] = float(absolute.mean())
            expected[key] = item
        matched, seen = 0, set()
        for row in self.records(
            paths[0], "final_aggregate_by_seed" if by_seed else "final_aggregate"
        ):
            key = _key(row, keys)
            if key not in expected:
                self.error(paths[0], f"aggregate lacks raw source: {key}")
                continue
            matched += self.compare(expected[key], row, expected[key].keys(), paths[0])
            seen.add(key)
        if seen != set(expected):
            self.error(paths[0], "aggregate omits raw-source groups")
        return matched

    def n2d(self, paths):
        rows, matched = [], 0
        for path in paths:
            unit = path.parent
            trajectory, ai = unit.parent.parent.name, int(unit.parent.name.removeprefix("a"))
            theta, p = self.parent_arrays(trajectory)
            meta = self.json(path, "N2D_diagnostic_identity")
            if meta.get("main_ranking_eligible") is not False:
                raise ValueError("N2D oracle included in main ranking")
            self.cover(unit / "predictions.npz", "N2D_raw_predictions", meta["prediction_sha256"])
            with np.load(unit / "predictions.npz", allow_pickle=False) as data:
                predictions, truth = data["prediction"], data["truth"]
            parent_truth = p[ai, 10:14, 1:] - p[ai, 10:14, :1]
            if not np.allclose(truth, parent_truth, atol=1e-12, rtol=0):
                raise ValueError("N2D stored truth differs from parent")
            used = []
            fit = theta[ai, :8, 1:] - theta[ai, :8, :1]
            for ident in meta["records"]:
                index = int(ident["array_index"])
                used.append(index)
                prediction = predictions[index].reshape(4, 2, len(self.metadata), 3) @ H.T
                scores = independent_model_statistics(prediction, truth, self.metadata, fit, ident)
                rows.extend(
                    {k: v for k, v in score.items() if not k.endswith("_abs_errors")}
                    for score in scores
                )
                matched += 1
            if sorted(used) != list(range(len(predictions))):
                raise ValueError("N2D prediction identity is missing or repeated")
        _write_csv(self.out / "N2D_RECOMPUTED_BY_SEED.csv", rows)
        groups = {}
        group_keys = (
            "method",
            "rank_cap",
            "diagnostic",
            "covariance_source",
            "observation",
            "target",
            "population",
        )
        for row in rows:
            groups.setdefault(_key(row, group_keys), []).append(row)
        pooled = []
        for key, rr in groups.items():
            record = dict(
                zip(group_keys, key, strict=True),
                seed_count=len({r["seed"] for r in rr}),
                main_ranking_eligible=False,
            )
            for metric in ("event", "pX", "v"):
                numerator = sum(r[f"{metric}_error_ss"] for r in rr)
                denominator = sum(r[f"{metric}_truth_ss"] for r in rr)
                record[f"{metric}_error_ss"] = numerator
                record[f"{metric}_truth_ss"] = denominator
                record[f"{metric}_nrmse"] = (
                    float(np.sqrt(numerator / denominator)) if denominator > 0 else None
                )
            pooled.append(record)
        _write_csv(self.out / "N2D_RECOMPUTED_POOLED.csv", pooled)
        return matched


def run_recompute(run_root, parent_root, out):
    """Verify only materialized phases; missing phases remain explicit NOT_RUN."""
    started = time.perf_counter()
    verifier = _Verifier(run_root, parent_root, out)
    verifier.phase(
        "N1", sorted((verifier.run / "N1/oracle").glob("*/a*/comparison.csv")), verifier.n1
    )
    n1_table = verifier.run / "N1/OBSERVATION_COMPARISON.csv"
    verifier.phase("N1_final", [n1_table] if n1_table.exists() else [], verifier.n1_final)
    materialized_n2_units = 0
    for phase in ("N2A", "N2B", "N2C", "N2E"):
        paths = sorted((verifier.run / "N2" / phase).glob("*/a*/freeze.json"))
        materialized_n2_units += len(paths)
        verifier.phase(phase, paths, verifier.n2)
    final = verifier.run / "N2/METRICS.jsonl.gz"
    verifier.phase("N2_final_metrics", [final] if final.exists() else [], verifier.n2_final)
    aggregate = verifier.run / "N2/AGGREGATE.csv"
    verifier.phase("N2_aggregate", [aggregate] if aggregate.exists() else [], verifier.aggregate)
    seeds = verifier.run / "N2/CONTRAST_PREDICTION_BY_SEED.csv"
    verifier.phase(
        "N2_by_seed",
        [seeds] if seeds.exists() else [],
        lambda paths: verifier.aggregate(paths, True),
    )
    verifier.phase(
        "N2D", sorted((verifier.run / "N2/N2D").glob("*/a*/*/diagnostics.json")), verifier.n2d
    )
    table_artifact = None
    if verifier.action_cache:
        table_path = verifier.out / "raw_covariance_forward_tables.npz"
        np.savez_compressed(
            table_path,
            **{entry["array_name"]: entry["table"] for entry in verifier.action_cache.values()},
        )
        with np.load(table_path, allow_pickle=False) as saved:
            for entry in verifier.action_cache.values():
                if _array_hash(saved[entry["array_name"]]) != entry["table_sha256"]:
                    verifier.error(
                        table_path, "replayed full-action table failed saved-array hash check"
                    )
        table_artifact = {
            "path": table_path.name,
            "sha256": _hash(table_path),
            "bytes": table_path.stat().st_size,
            "all_saved_array_hashes_verified": True,
        }
    replay_receipts = {
        "purpose": (
            "independent raw covariance formula verification on unchanged legacy CPU policies"
        ),
        "main_ranking_eligible": False,
        "optimizer_updates": 0,
        "jacobian_calls": 0,
        "sampling_calls": 0,
        "cost": verifier.forward_cost,
        "tables_artifact": table_artifact,
        "tables": [
            {k: v for k, v in entry.items() if k != "table"}
            for entry in verifier.action_cache.values()
        ],
    }
    _write_json(verifier.out / "FORWARD_REPLAY_RECEIPTS.json", replay_receipts)
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        verifier.cover(path, "source_catalog")
    for path, item in list(verifier.coverage.items()):
        stat = Path(path).stat()
        if (stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]) and _hash(
            path
        ) != item["sha256"]:
            verifier.error(path, "input changed during verification")
    manifest = {
        "read_files": list(verifier.coverage.values()),
        "file_count": len(verifier.coverage),
        "total_covered_bytes": sum(v["bytes"] for v in verifier.coverage.values()),
        "source_catalog_is_provenance_not_a_claim_of_executing_every_module": True,
    }
    _write_json(verifier.out / "COVERAGE_MANIFEST.json", manifest)
    projection_complete = materialized_n2_units == verifier.projection_totals["units"]
    _write_json(
        verifier.out / "PROJECTION_AUDIT_SUMMARY.json",
        dict(
            verifier.projection_totals,
            status="NOT_RUN"
            if not materialized_n2_units
            else ("PASS" if projection_complete else "FAIL"),
            main_ranking_eligible=False,
            posthoc_oracle_baseline_diagnostic=True,
            raw_files_modified=False,
            materialized_N2A_B_C_E_units=materialized_n2_units,
            complete_for_materialized_N2A_B_C_E_units=projection_complete,
        ),
    )
    executed = any(check["status"] != "NOT_RUN" for check in verifier.checks.values())
    result = {
        "status": "FAIL" if verifier.errors else ("PASS" if executed else "NOT_RUN"),
        "checks": verifier.checks,
        "not_run": [k for k, v in verifier.checks.items() if v["status"] == "NOT_RUN"],
        "errors": verifier.errors,
        "maximum_reported_errors": 200,
        "absolute_tolerance": 1e-12,
        "scalar_or_array_comparisons": verifier.scalar_comparisons,
        "max_absolute_difference": verifier.max_diff,
        "model_forward_calls": verifier.forward_cost["batched_model_forward_calls"],
        "jacobian_calls": 0,
        "sampling_calls": 0,
        "optimizer_updates": 0,
        "additional_frozen_cpu_verification_cost": verifier.forward_cost,
        "elapsed_seconds": time.perf_counter() - started,
        "coverage_manifest_sha256": _hash(verifier.out / "COVERAGE_MANIFEST.json"),
        "projection_audit": verifier.projection_totals,
        "N1_covariance_recovery": {
            "status": verifier.checks["N1"]["status"],
            "unique_unscaled_joint_raw_covariances": len(verifier.raw_cov_cache),
            "matched_raw_and_helmert_variance_rows": verifier.checks["N1"]["matched_rows"],
            "new_full_action_tables": len(verifier.action_cache),
            "independent_contribution_moment_formulas": True,
            "action_distribution_inferred_from_four_events": False,
            "forward_replay_receipts_sha256": _hash(verifier.out / "FORWARD_REPLAY_RECEIPTS.json"),
        },
        "independence": {
            "production_score_helpers_called": False,
            "paid_packet_reconstruction_reused": True,
            "prediction_files_modified": False,
            "parent_files_modified": False,
        },
        "coverage_limits": [
            "N1 interval/cost/label-status metadata is outside this point-estimate "
            "and covariance-trace table verification.",
            "N2D is independently scored as an oracle diagnostic "
            "and remains outside the main ranking.",
        ],
    }
    _write_json(verifier.out / "RECOMPUTATION_SUMMARY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_recompute(args.run_root, args.parent_root, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
