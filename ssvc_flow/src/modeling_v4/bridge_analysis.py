"""CPU-only, post-hoc analysis of immutable B-stage samples; no model loading.

Intervals describe sampling conditional on the measured policies and prompts.
No method selection, new observations, reference certification or GPU calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

from ..modeling_v3.observation_geometry import ContributionBatch, stable_origin_weight
from .analysis_rules import endpoint_overlap, load_analysis_rules, reference_precision
from .observations import observe_packet

EVENTS = ("X", "S", "W", "I")
RAW_COLUMNS = (
    "sample_key",
    "sample_seed",
    "prompt_id",
    "draw_index",
    "rng_namespace",
    "event_onehot",
    "category",
    "generation_sequence_logp",
    "token_ids",
    "shared_token_identity",
    "input_hash",
    "proposal_fingerprint",
    "eos_seen",
    "generation_parity",
    "probability_execution",
    "max_new_tokens",
)
SCORE_COLUMNS = (
    "sample_key",
    "prompt_id",
    "sequence_logp",
    "shared_token_identity",
    "input_hash",
    "inference_fingerprint",
    "probability_execution",
)


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_plain, allow_nan=False) + "\n"
    )


def moments(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 2:
        raise ValueError("At least two complete four-event draws required")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite contributions")
    # Remove a represented row before centering. Repeated identical floating
    # values otherwise acquire tiny spurious variances through mean rounding.
    covariance = np.cov(values - values[0], rowvar=False, ddof=1) / len(values)
    mass = values.sum(1)
    return {
        "estimate": values.mean(0),
        "covariance_of_mean": covariance,
        "se": np.sqrt(np.maximum(0, np.diag(covariance))),
        "mass_mean": float(mass.mean()),
        "mass_se": float((mass - mass[0]).std(ddof=1) / math.sqrt(len(values))),
    }


def direct_intervals(left, right, *, alpha=0.05):
    """Subtract CP endpoint intervals with a union bound (>=1-alpha coverage).

    Each endpoint gets a two-sided 1-alpha/2 interval. The resulting interval
    is conservative and does not collapse on unobserved events. For a family
    of M differences, pass alpha=.05/M. Assumes iid Bernoulli endpoint draws.
    """
    left, right = np.asarray(left), np.asarray(right)
    intervals = []
    for event in range(4):
        a = binomtest(int(left[:, event].sum()), len(left)).proportion_ci(
            confidence_level=1 - alpha / 2, method="exact"
        )
        b = binomtest(int(right[:, event].sum()), len(right)).proportion_ci(
            confidence_level=1 - alpha / 2, method="exact"
        )
        intervals.append([max(-1, a.low - b.high), min(1, a.high - b.low)])
    return np.asarray(intervals)


def independent_packets(packets):
    ids, seeds, namespaces = set(), set(), set()
    for rows in packets:
        keys = {r["sample_key"] for r in rows}
        sample_seeds = {r["sample_seed"] for r in rows}
        streams = {r["rng_namespace"] for r in rows}
        if len(keys) != len(rows) or len(sample_seeds) != len(rows):
            raise ValueError("Duplicate sample ID or seed within packet")
        if ids & keys or seeds & sample_seeds or namespaces & streams:
            raise ValueError("Independent packets reuse sample IDs, seeds or RNG namespaces")
        ids.update(keys)
        seeds.update(sample_seeds)
        namespaces.update(streams)
    return {
        "unique_sample_keys": len(ids),
        "unique_seeds": len(seeds),
        "rng_namespaces": sorted(namespaces),
        "overlap": 0,
    }


class Reader:
    def __init__(self, campaign):
        self.root = Path(campaign).resolve()
        self.cache = {}
        self.manifest = {}

    def rows(self, receipt, *, score=False):
        columns = SCORE_COLUMNS if score else RAW_COLUMNS
        key = tuple(c["path"] for c in receipt["chunks"])
        if key not in self.cache:
            rows = []
            for chunk in receipt["chunks"]:
                path = Path(chunk["path"]).resolve()
                if not path.is_relative_to(self.root / "tasks"):
                    raise ValueError("Input chunk outside campaign tasks")
                table = pq.read_table(path, columns=list(columns))
                part = table.to_pylist()
                if len(part) != chunk["stop"] - chunk["start"]:
                    raise ValueError("Chunk count differs from recorded receipt")
                rows.extend(part)
                self.manifest[str(path)] = {
                    "recorded_sha256": chunk["sha256"],
                    "bytes": path.stat().st_size,
                    "rows_read": len(part),
                    "content_rehashed": False,
                }
            if len(rows) != receipt["count"] or len({r["sample_key"] for r in rows}) != len(rows):
                raise ValueError("Receipt count or sample uniqueness failed")
            self.cache[key] = rows
        return self.cache[key]


def aligned_scores(rows, receipt, reader):
    scores = {r["sample_key"]: r for r in reader.rows(receipt, score=True)}
    if len(scores) != len(rows):
        raise ValueError("Score/sample count mismatch")
    values = []
    for row in rows:
        other = scores[row["sample_key"]]
        for field in ("prompt_id", "shared_token_identity", "input_hash"):
            if other[field] != row[field]:
                raise ValueError("Score changed prompt, input or token identity")
        if other["inference_fingerprint"] != receipt["identity"]["policy"]["inference_fingerprint"]:
            raise ValueError("Score policy mismatch")
        if other["probability_execution"] != "uncached_prefix_recompute":
            raise ValueError("Unexpected scoring probability execution")
        values.append(other["sequence_logp"])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scores")
    return np.asarray(values)


def _table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(campaign, out, *, bootstrap=1000):
    started = time.monotonic()
    campaign, out = Path(campaign).resolve(), Path(out).resolve()
    if out.is_relative_to(campaign / "tasks") or out == campaign:
        raise ValueError("Analysis output must not overwrite measured tasks")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty analysis output directory")
    out.mkdir(parents=True, exist_ok=True)
    reader, rules = Reader(campaign), load_analysis_rules()
    plan = _read(campaign.parent / "tasks_initial.json")
    prompts = {p["prompt_id"]: p for p in plan["inputs"]["bridge_probes"]}
    # This post-hoc family is declared here, independent of outcome values.
    family = 2 * len(prompts) * 4
    normal_z = NormalDist().inv_cdf(1 - 0.05 / (2 * family))
    old_z = NormalDist().inv_cdf(1 - 0.01 / (8 * len(prompts)))
    output = {
        "origins": {},
        "rules": rules,
        "interval_family_cells": family,
        "normal_family_scope": "96 cells PER fixed method/comparison, not all comparisons jointly",
        "posthoc_normal_z": normal_z,
        "old_diagnostic_z": old_z,
        "bootstrap_repetitions": bootstrap,
        "model_calls": 0,
        "new_generated_samples": 0,
        "analysis_scope": "POSTHOC_FIXED_BRIDGE_PANEL",
        "scientific_status": "NOT_CERTIFIED",
        "model_prediction_accuracy_measured": False,
    }
    cells, packets, comparisons, counts, mass_rows, overlaps = [], [], [], [], [], []
    for origin in ("B_origin_0", "B_origin_1"):
        directory = campaign / "tasks" / origin
        complete = _read(directory / "COMPLETE.json")
        response = _read(complete["response"]["path"])
        forks = _read(complete["forks"]["path"])
        old = _read(complete["pair_diagnostics"]["path"])
        (check,) = response["pair_checks"]
        left, right = check["left_candidate_id"], check["right_candidate_id"]
        streams = {key: reader.rows(response[key]) for key in ("work", "reference", "direct_work")}
        streams["mixture"] = reader.rows(check["mixture"])
        streams["count_left"] = reader.rows(check["direct"][left])
        streams["count_right"] = reader.rows(check["direct"][right])
        independent = independent_packets(streams.values())
        score_arrays = {}
        for role in ("work", "reference", "direct_work", "mixture"):
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
            for candidate in (left, right):
                receipt = (
                    mapping[candidate] if role == "mixture" else mapping[candidate]["score_receipt"]
                )
                if (
                    receipt["identity"]["policy"]["inference_fingerprint"]
                    != response["policies"][candidate]["inference_fingerprint"]
                ):
                    raise ValueError("Scored endpoint differs from selected candidate")
                score_arrays[role, candidate] = aligned_scores(streams[role], receipt, reader)
        origin_result = {
            "complete": {
                "path": str(directory / "COMPLETE.json"),
                "sha256": hashlib.sha256((directory / "COMPLETE.json").read_bytes()).hexdigest(),
            },
            "origin_fingerprint": forks["origin_policy"]["inference_fingerprint"],
            "optimizer_hash": forks["origin_policy"]["optimizer_state_hash"],
            "independence": independent,
            "active_bank": check["bank_id"],
            "left_candidate": left,
            "right_candidate": right,
            "bank_contrasts": [
                {
                    "bank_id": b["bank_id"],
                    "origin_state_restored": b["origin_state_restored"],
                    "contrasts": {
                        k: {x: v[x] for x in ("norm", "exact_inference_alias")}
                        for k, v in b["contrasts"].items()
                    },
                }
                for b in forks["banks"]
            ],
            "scoring_path": complete["scoring_path"],
            "resume_smoke": complete["resume_smoke"],
            "gradient_path": {
                k: complete["gradient"]["gradient_path"].get(k)
                for k in (
                    "status",
                    "passed",
                    "relative_l2",
                    "cosine",
                    "directional_dots",
                    "fallback",
                )
            },
            "gradient_samples": complete["gradient"]["anchor_samples"],
            "old_diagnostic_max_recompute_error": 0.0,
        }
        old_by_prompt = {u["prompt_id"]: u for u in old["units"]}
        for index, pid in enumerate(response["work"]["identity"]["prompt_ids"]):
            base = {
                "origin": origin,
                "probe": index + 1,
                "prompt_id": pid,
                **{k: prompts[pid][k] for k in ("interface", "family", "operation")},
            }
            subsets, masks = {}, {}
            for role, rows in streams.items():
                masks[role] = np.asarray([r["prompt_id"] == pid for r in rows])
                subsets[role] = [r for r in rows if r["prompt_id"] == pid]
                sample = subsets[role]
                expected_fps = {
                    "work": {forks["origin_policy"]["inference_fingerprint"]},
                    "reference": {forks["origin_policy"]["inference_fingerprint"]},
                    "direct_work": {forks["origin_policy"]["inference_fingerprint"]},
                    "count_left": {response["policies"][left]["inference_fingerprint"]},
                    "count_right": {response["policies"][right]["inference_fingerprint"]},
                    "mixture": {
                        response["policies"][c]["inference_fingerprint"] for c in (left, right)
                    },
                }[role]
                if any(
                    r["probability_execution"] != "uncached_prefix_recompute"
                    or r["max_new_tokens"] != 64
                    or r["proposal_fingerprint"] not in expected_fps
                    for r in sample
                ):
                    raise ValueError("Sample probability path, horizon or proposal mismatch")
                if len(sample) != 64 or sorted(r["draw_index"] for r in sample) != list(range(64)):
                    raise ValueError("Bridge must retain all 64 registered draws per prompt")
                events = np.asarray([r["event_onehot"] for r in sample], dtype=float)
                if not np.all((events == 0) | (events == 1)) or not np.all(events.sum(1) == 1):
                    raise ValueError("Invalid event partition")
                if any(EVENTS[int(np.argmax(r["event_onehot"]))] != r["category"] for r in sample):
                    raise ValueError("Event order mismatch")
                if not all(r["generation_parity"]["passed"] for r in sample):
                    raise ValueError("Generation/rescore mismatch in completed packet")
                for e, event in enumerate(EVENTS):
                    counts.append(
                        {
                            **base,
                            "packet": role,
                            "event": event,
                            "count": int(events[:, e].sum()),
                            "draws": len(sample),
                            "eos_count": sum(bool(r["eos_seen"]) for r in sample),
                            "max_tokens": max(len(r["token_ids"]) for r in sample),
                        }
                    )
            estimates = {}
            for role in ("work", "reference", "direct_work", "mixture"):
                sample, take = subsets[role], masks[role]
                logl, logr = score_arrays[role, left][take], score_arrays[role, right][take]
                logq = np.array([r["generation_sequence_logp"] for r in sample])
                weight = (
                    2 * np.tanh((logl - logr) / 2)
                    if role == "mixture"
                    else stable_origin_weight(logl, logr, logq)
                )
                onehot = np.array([r["event_onehot"] for r in sample])
                raw = onehot * weight[:, None]
                stats = moments(raw)
                overlap_usable = True
                if role != "mixture":
                    for endpoint, logs in (("left", logl), ("right", logr)):
                        ov = endpoint_overlap(logs - logq, rules=rules)
                        overlap_usable &= bool(ov["usable"])
                        overlaps.append(
                            {
                                **base,
                                "packet": role,
                                "endpoint": endpoint,
                                "mean_weight": float(np.exp(logs - logq).mean()),
                                **{
                                    k: float(ov[k])
                                    for k in ("ess", "ess_fraction", "max_normalized_weight")
                                },
                                "usable": bool(ov["usable"]),
                            }
                        )
                mass_rows.append(
                    {
                        **base,
                        "packet": role,
                        "mass_mean": stats["mass_mean"],
                        "mass_se": stats["mass_se"],
                        "observed_event_types": int((onehot.sum(0) > 0).sum()),
                        "max_abs_weight": float(np.max(np.abs(weight))),
                        "largest_abs_contribution_fraction": float(
                            np.abs(weight).max() / np.abs(weight).sum()
                        )
                        if np.abs(weight).sum()
                        else 0.0,
                    }
                )
                # Existing collector uses full softmax support, identical vocab,
                # EOS/truncation and prefix scoring; no support claim from ESS.
                batch = ContributionBatch(
                    raw[:, None, :],
                    tuple(r["sample_key"] for r in sample),
                    sample[0]["rng_namespace"],
                    pid,
                    (check["contrast_id"],),
                    (
                        (
                            response["policies"][right]["inference_fingerprint"],
                            response["policies"][left]["inference_fingerprint"],
                        ),
                    ),
                    tuple(tuple(r["token_ids"]) for r in sample),
                    "equal_pair_mixture" if role == "mixture" else "origin",
                )
                observed = observe_packet(
                    batch,
                    bootstrap_repetitions=bootstrap,
                    split_seed=20260917,
                    bootstrap_seed=20260917 + index,
                )
                for method, result in observed["methods"].items():
                    value = np.asarray(result["estimate"]).reshape(4)
                    if method == "RAW4":
                        cov = stats["covariance_of_mean"]
                    elif method == "PRESERVE_XI":
                        transform = np.eye(4) - np.outer([0, 0.5, 0.5, 0], np.ones(4))
                        cov = transform @ stats["covariance_of_mean"] @ transform.T
                    else:
                        replicates = np.asarray(
                            result["uncertainty"]["replicate_estimates"]
                        ).reshape(bootstrap, 4)
                        # Replicates are already estimates of a mean: DO NOT
                        # divide their sampling variance by bootstrap count.
                        cov = np.cov(replicates - replicates[0], rowvar=False, ddof=1)
                    se = np.sqrt(np.maximum(0, np.diag(cov)))
                    label = f"{role}/{method}"
                    estimates[label] = (value, cov)
                    precision = reference_precision(se, overlap_usable=overlap_usable, rules=rules)
                    packets.append(
                        {
                            **base,
                            "packet": role,
                            "method": method,
                            "estimate": value,
                            "covariance_of_mean": cov,
                            "raw_mass_mean": stats["mass_mean"],
                            "raw_mass_se": stats["mass_se"],
                            "uncertainty_status": result["uncertainty"]["status"],
                            "fold_coefficients": result["diagnostics"].get("fold_coefficients"),
                            "bootstrap_percentile_interval": result["uncertainty"].get(
                                "percentile_interval"
                            ),
                        }
                    )
                    for e, event in enumerate(EVENTS):
                        cells.append(
                            {
                                **base,
                                "packet": role,
                                "method": method,
                                "event": event,
                                "estimate": float(value[e]),
                                "se": float(se[e]),
                                "normal95_low": float(value[e] - 1.96 * se[e]),
                                "normal95_high": float(value[e] + 1.96 * se[e]),
                                "zero_empirical_variance": bool(se[e] == 0),
                                "observed_event_count": int(onehot[:, e].sum()),
                                **{
                                    f"resolved_{s}": bool(a[e])
                                    for s, a in precision["resolved_at_scale"].items()
                                },
                                "direct_cp95_low": "",
                                "direct_cp95_high": "",
                                "direct_family95_low": "",
                                "direct_family95_high": "",
                            }
                        )
            a, b = [
                np.asarray([r["event_onehot"] for r in subsets[role]], float)
                for role in ("count_left", "count_right")
            ]
            value = a.mean(0) - b.mean(0)
            cov = moments(a)["covariance_of_mean"] + moments(b)["covariance_of_mean"]
            se = np.sqrt(np.maximum(0, np.diag(cov)))
            cp = direct_intervals(a, b)
            cp_family = direct_intervals(a, b, alpha=0.05 / family)
            estimates["direct/COUNT"] = (value, cov)
            packets.append(
                {
                    **base,
                    "packet": "direct",
                    "method": "COUNT",
                    "estimate": value,
                    "covariance_of_mean": cov,
                    "direct_cp95": cp,
                    "direct_family95": cp_family,
                }
            )
            for e, event in enumerate(EVENTS):
                cells.append(
                    {
                        **base,
                        "packet": "direct",
                        "method": "COUNT",
                        "event": event,
                        "estimate": float(value[e]),
                        "se": float(se[e]),
                        "normal95_low": float(value[e] - 1.96 * se[e]),
                        "normal95_high": float(value[e] + 1.96 * se[e]),
                        "zero_empirical_variance": bool(se[e] == 0),
                        "observed_event_count": int(a[:, e].sum() + b[:, e].sum()),
                        **{
                            f"resolved_{s}": bool(
                                max(value[e] - cp[e, 0], cp[e, 1] - value[e]) <= s / 2
                            )
                            for s in (0.00025, 0.001, 0.005)
                        },
                        "direct_cp95_low": float(cp[e, 0]),
                        "direct_cp95_high": float(cp[e, 1]),
                        "direct_family95_low": float(cp_family[e, 0]),
                        "direct_family95_high": float(cp_family[e, 1]),
                    }
                )
            for original, label in (
                ("ORIGIN", "reference/RAW4"),
                ("MIX", "mixture/RAW4"),
                ("DIRECT", "direct/COUNT"),
            ):
                value, cov = estimates[label]
                error = max(
                    np.max(np.abs(value - old_by_prompt[pid]["means"][original])),
                    np.max(np.abs(np.diag(cov) - old_by_prompt[pid]["variance_of_mean"][original])),
                )
                origin_result["old_diagnostic_max_recompute_error"] = max(
                    origin_result["old_diagnostic_max_recompute_error"], float(error)
                )
                if error > 1e-12:
                    raise ValueError("Recomputation differs from bridge diagnostic")
            for method in ("RAW4", "PRESERVE_XI", "CROSSFIT_COV_ZERO_SUM"):
                ref = f"reference/{method}"
                for other in (
                    f"work/{method}",
                    f"direct_work/{method}",
                    f"mixture/{method}",
                    "direct/COUNT",
                ):
                    gap = estimates[other][0] - estimates[ref][0]
                    se = np.sqrt(np.maximum(0, np.diag(estimates[other][1] + estimates[ref][1])))
                    for e, event in enumerate(EVENTS):
                        comparisons.append(
                            {
                                **base,
                                "reference": ref,
                                "other": other,
                                "event": event,
                                "signed_difference": float(gap[e]),
                                "se": float(se[e]),
                                "abs_z": float(abs(gap[e]) / se[e])
                                if se[e] > 0
                                else "UNRESOLVED_ZERO_SE",
                                "outside_normal95": bool(se[e] > 0 and abs(gap[e]) > 1.96 * se[e]),
                                "outside_old_diagnostic": bool(abs(gap[e]) > old_z * se[e] + 1e-4),
                                "outside_family_normal95": bool(
                                    se[e] > 0 and abs(gap[e]) > normal_z * se[e]
                                ),
                            }
                        )
            # Explicit work/reference checks and closure retain all covariance.
        output["origins"][origin] = origin_result
        reader.cache.clear()
        print(f"{origin}: read and refit complete", flush=True)
    output["cell_summary"] = []
    for origin in output["origins"]:
        for packet, method in sorted({(r["packet"], r["method"]) for r in cells}):
            rows = [
                r
                for r in cells
                if (r["origin"], r["packet"], r["method"]) == (origin, packet, method)
            ]
            positive = [r["se"] for r in rows if r["se"] > 0]
            output["cell_summary"].append(
                {
                    "origin": origin,
                    "packet": packet,
                    "method": method,
                    "cells": len(rows),
                    "zero_variance": sum(r["zero_empirical_variance"] for r in rows),
                    "median_positive_se": float(np.median(positive)) if positive else None,
                    "max_se": max(r["se"] for r in rows),
                    "max_abs_estimate": max(abs(r["estimate"]) for r in rows),
                    **{
                        f"resolved_{s}": sum(r[f"resolved_{s}"] for r in rows)
                        for s in (0.00025, 0.001, 0.005)
                    },
                }
            )
    output["stream_event_counts"] = {
        origin: {
            role: dict(
                Counter(
                    {
                        event: sum(
                            r["count"]
                            for r in counts
                            if r["origin"] == origin and r["packet"] == role and r["event"] == event
                        )
                        for event in EVENTS
                    }
                )
            )
            for role in ("work", "reference", "direct_work", "mixture", "count_left", "count_right")
        }
        for origin in output["origins"]
    }
    output["input_chunk_count"] = len(reader.manifest)
    output["input_bytes_read"] = sum(r["bytes"] for r in reader.manifest.values())
    output["elapsed_seconds"] = time.monotonic() - started
    output["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    output["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output["old_precision_labels_were_unconditional"] = True
    output["covariance_centering"] = (
        "Subtract first represented row before centering; constant draws have exact zero variance"
    )
    output["interval_limitations"] = [
        "Normal and full-refit bootstrap intervals are descriptive, not guaranteed coverage.",
        "Nonparametric bootstrap cannot reveal unobserved events or unobserved importance tails.",
        "CP differences use endpoint exact intervals and a union bound; iid sampling required.",
        "Family CP covers 96 bridge prompt/event differences; no cross-seed population inference.",
        "Frozen empirical precision labels are not accurate-reference certification.",
        "No exact event truth, hence no measured bias/RMSE or model prediction accuracy.",
    ]
    _table(out / "EVENT_ESTIMATES.csv", cells)
    _table(out / "METHOD_DIFFERENCES.csv", comparisons)
    _table(out / "EVENT_COUNTS.csv", counts)
    _table(out / "MASS_RESIDUALS.csv", mass_rows)
    _table(out / "ENDPOINT_OVERLAP.csv", overlaps)
    _write(out / "PACKET_STATISTICS.json", packets)
    _write(out / "INPUT_INDEX.json", reader.manifest)
    _write(out / "SUMMARY.json", output)
    _write(
        out / "COMPLETE.json",
        {
            "status": "POSTHOC_ANALYSIS_COMPLETE_NOT_CERTIFIED",
            "job_id": output["slurm_job_id"],
            "model_calls": 0,
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.iterdir())
                if p.is_file()
            },
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        parser.error("Actual bridge analysis requires a server CPU Slurm allocation")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        parser.error("Explicitly hide CUDA devices for CPU-only analysis")
    if args.bootstrap < 2:
        parser.error("At least two full-refit bootstrap repetitions required")
    analyze(args.campaign, args.out, bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
