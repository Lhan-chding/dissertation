"""Derived Q1 summaries from immutable repeated observations, never group proxies.

Full analysis runs on server CPUs. Per-unit arrays preserve each measurement
repeat and prompt. Pooled quantiles are recomputed from those residuals, never
averaged from unit/group quantiles. Missing timing/fit receipts stay missing.
"""

from __future__ import annotations

import csv
import io
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import (
    atomic_bytes,
    atomic_json,
    atomic_npz,
    canonical_hash,
    finalize_run,
    sha256_file,
    verify_manifest,
)

EVENTS = ("X", "S", "W", "I")
QUANTILES = (0.5, 0.9, 0.95, 0.99)
TIMING_FIELDS = (
    "packet_sampling_cached_scoring_label_seconds",
    "estimator_seconds",
    "independent_count_seconds",
    "sufficient_statistics_seconds",
    "artifact_io_seconds",
    "repeat_wall_seconds_before_completion_manifest",
)
DEFINITIONS = {
    "signed_bias": "mean(prediction - exact truth) over repeat-prompt cells",
    "pooled_residual_population_variance": (
        "ddof=0 over repeat-prompt cells; includes heterogeneous prompt biases"
    ),
    "mean_within_prompt_repeat_sample_variance": (
        "ddof=1 across independent repeats separately for each fixed prompt, "
        "then equal prompt average"
    ),
    "mse_bias_variance_identity": (
        "MSE = mean(prompt_bias^2) + (R-1)/R * mean(within_prompt_repeat_sample_variance)"
    ),
    "delta_v": "-delta_pI; signed bias and signed quantiles reverse the I channel",
    "coefficients": (
        "actual applied coefficients; variance across repeats at fixed prompt; "
        "crossfit folds reported separately"
    ),
    "pooled_quantiles": (
        "recomputed from concatenated original residual cells within method/n/proposal/case; "
        "never mean unit quantiles"
    ),
    "intervals": "residual quantiles are descriptive errors, not confidence intervals",
    "timings": (
        "observed blocks, not inferred per-method shares; "
        "packet block combines cached scoring, sampling and labels"
    ),
}


def _quantiles(values):
    return dict(
        zip(("q50", "q90", "q95", "q99"), map(float, np.quantile(values, QUANTILES)), strict=True)
    )


def _flat_summary(values):
    a = np.asarray(values, dtype=float).ravel()
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Finite nonempty original residuals required")
    return {
        "count": len(a),
        "signed_bias": float(a.mean()),
        "pooled_residual_population_variance": float(np.var(a)),
        "mse": float(np.mean(a * a)),
        "mae": float(np.mean(abs(a))),
        "signed_residual_quantiles": _quantiles(a),
        "absolute_residual_quantiles": _quantiles(abs(a)),
        "max_absolute": float(abs(a).max()),
    }


def residual_summary(residual, truth):
    """[repeat,prompt,4] residuals with explicitly separate variance definitions."""
    error, truth = np.asarray(residual, float), np.asarray(truth, float)
    if (
        error.ndim != 3
        or error.shape[-1] != 4
        or not error.shape[0]
        or not error.shape[1]
        or truth.shape != error.shape[1:]
        or not np.isfinite(error).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("Aligned finite repeat/prompt/event residuals and exact truth required")
    result = {}
    for name, index, sign in [*((e, i, 1) for i, e in enumerate(EVENTS)), ("delta_v", 3, -1)]:
        value = sign * error[..., index]
        record = _flat_summary(value)
        within = float(np.var(value, axis=0, ddof=1).mean()) if len(error) > 1 else None
        bias_energy = float(np.mean(value.mean(0) ** 2))
        signal = float(np.sum(truth[..., index] ** 2)) * len(error)
        result[name] = {
            **record,
            "mean_within_prompt_repeat_sample_variance": within,
            "repeat_variance_status": "AVAILABLE" if within is not None else "MISSING_INPUTS",
            "mean_squared_prompt_bias": bias_energy,
            "mse_bias_variance_identity_residual": None
            if within is None
            else record["mse"] - bias_energy - (len(error) - 1) / len(error) * within,
            "nrmse": float(np.sqrt(np.sum(value**2) / signal)) if signal else None,
            "signal_energy": signal,
            "nrmse_status": "AVAILABLE" if signal else "UNDEFINED_ZERO_SIGNAL",
        }
    return result


def _missing(metric, reason):
    return {"status": "MISSING_INPUTS", "metric": metric, "reason": reason}


def _publish_json(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Immutable derived output differs: " + str(path))
    else:
        atomic_json(path, value)


def _publish_npz(path, values):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(values) or any(
                not np.array_equal(existing[k], v, equal_nan=True) for k, v in values.items()
            ):
                raise ValueError("Immutable derived exact records differ")
    else:
        atomic_npz(path, values)


def _coefficient_report(method, records, diagnostics, expected_repeats, prompts):
    if method in ("INDEPENDENT_COUNT", "KNOWN_EVENT_SCORE"):
        return {"status": "NOT_APPLICABLE_NO_CONTROL_VARIATE"}
    crossfit = method == "CROSSFIT_COV_ZERO_SUM"
    names = [f"b_fold0_{method}", f"b_fold1_{method}"] if crossfit else [f"b_{method}"]
    blocks = {}
    for name in names:
        values = [r[name] for r in records if name in r]
        if len(values) != expected_repeats:
            blocks[name] = _missing(
                "coefficient", "Missing actual applied coefficient in one or more repeats"
            )
            continue
        b = np.asarray(values)
        if b.shape != (expected_repeats, prompts, 4) or not np.isfinite(b).all():
            raise ValueError("Applied coefficient ledger has invalid shape or nonfinite values")
        l1 = abs(b).sum(-1)
        blocks[name] = {
            "status": "AVAILABLE",
            "coefficient_fits": int(np.prod(b.shape[:-1])),
            "event_order": EVENTS,
            "mean_coefficient": b.mean(axis=(0, 1)).tolist(),
            "mean_within_prompt_repeat_sample_variance": np.var(b, axis=0, ddof=1).mean(0).tolist()
            if len(b) > 1
            else None,
            "pooled_population_variance": np.var(b, axis=(0, 1)).tolist(),
            "l1_mean": float(l1.mean()),
            "l1_max": float(l1.max()),
            "l1_quantiles": _quantiles(l1),
            "coefficient_sum_max_error": float(
                abs(b.sum(-1) - (0 if method == "RAW4" else 1)).max()
            ),
        }
    receipts = []
    for diagnostic in diagnostics:
        for item in diagnostic.get(method, []):
            receipts.extend([item.get("fold0", {}), item.get("fold1", {})] if crossfit else [item])
    if method.startswith("PILOT_") or crossfit:
        expected = expected_repeats * prompts * (2 if crossfit else 1)
        observed = [
            r
            for r in receipts
            if r.get("status")
            in ("PILOT_ESTIMATED", "PILOT_UNINFORMATIVE", "PILOT_INSUFFICIENT_DRAWS")
        ]
        if len(observed) != expected:
            fallback = {
                **_missing("fallback_rate", "Missing fitted-pilot status receipts"),
                "rate": None,
                "expected_fits": expected,
                "observed_fits": len(observed),
            }
        else:
            if any(
                key in receipt and type(receipt[key]) is not bool
                for receipt in observed
                for key in ("fallback_used", "l1_limited")
            ):
                raise ValueError("Pilot fallback/L1-limit receipts must be booleans")
            count = sum(
                r.get("fallback_used", r["status"] != "PILOT_ESTIMATED") is True for r in observed
            )
            fallback = {
                "status": "AVAILABLE",
                "count": count,
                "fits": expected,
                "rate": count / expected,
                "l1_limited_count": sum(r.get("l1_limited") is True for r in observed),
                "status_counts": {
                    status: sum(r["status"] == status for r in observed)
                    for status in sorted({r["status"] for r in observed})
                },
            }
    elif method == "KNOWN_COV_ORACLE":
        fallback = {
            **_missing("oracle_fallback_rate", "Original oracle fallback receipts unavailable"),
            "rate": None,
        }
    else:
        fallback = {"status": "NOT_APPLICABLE_FIXED_COEFFICIENT", "rate": None}
    return {
        "single_coefficient_status": "NOT_APPLICABLE_TWO_FOLDS"
        if crossfit
        else "ONE_APPLIED_COEFFICIENT_PER_PROMPT_REPEAT",
        "blocks": blocks,
        "fallback": fallback,
    }


def _timings(repeats, expected):
    result = {}
    for field in TIMING_FIELDS:
        values = [record.get("timings", {}).get(field) for record in repeats]
        available = [
            value
            for value in values
            if type(value) in (int, float) and math.isfinite(value) and value >= 0
        ]
        result[field] = {
            "status": "AVAILABLE" if len(available) == expected else "MISSING_INPUTS",
            "seconds": float(sum(available)) if len(available) == expected else None,
            "observed_partial_seconds": float(sum(available)) if available else None,
            "expected_repeats": expected,
            "observed_repeats": len(available),
        }
    for field in (
        "generation_only_seconds",
        "scoring_only_seconds",
        "labels_only_seconds",
        "per_estimator_fit_seconds",
    ):
        result[field] = {
            **_missing(field, "Acquisition retained combined timing blocks only"),
            "seconds": None,
        }
    return result


def _groups_from_original(identities, prompt_count, explicit):
    if explicit is not None:
        path = Path(explicit)
    else:
        source = next(
            (r.get("source_policy_files") for r in identities if r.get("source_policy_files")), None
        )
        if source is None or len(Path(source).parents) < 3:
            return None
        path = Path(source).parents[2] / "dataset/probe_metadata.json"
    hashes = {
        r.get("prompt_metadata_sha256") for r in identities if r.get("prompt_metadata_sha256")
    }
    if not path.is_file() or len(hashes) != 1:
        return None
    if sha256_file(path) != next(iter(hashes)):
        raise ValueError("Original fixed prompt metadata hash changed")
    rows = json.loads(path.read_text())
    if len(rows) != prompt_count or len({r["prompt_id"] for r in rows}) != prompt_count:
        raise ValueError("Prompt metadata does not match exact record axis")
    return rows


def _analyze_unit(unit, declared, output, *, fixture, prompt_metadata):
    output.mkdir(parents=True, exist_ok=True)
    expected = int(declared["repeats"])
    complete = json.loads((unit / "COMPLETE.json").read_text())
    if (
        complete["summary"] != declared
        or json.loads((unit / "RESULTS.json").read_text()) != declared
    ):
        raise ValueError("Q1 unit summary differs from its original completed manifest")
    records, identities, timing_receipts, inputs = [], [], [], []
    missing = []
    raw_verified = 0
    for repeat in range(expected):
        root = unit / f"repeat{repeat:03d}"
        stats = root / "statistics.npz"
        if not stats.is_file():
            missing.append(_missing("repeat_statistics", str(stats)))
            continue
        with np.load(stats, allow_pickle=False) as values:
            record = {key: values[key].copy() for key in values.files}
        if "truth" not in record:
            missing.append(_missing("exact_truth", str(stats)))
            continue
        if records and not np.array_equal(record["truth"], records[0]["truth"]):
            raise ValueError("Exact Q1 truth changed across repeated fixed-policy measurements")
        records.append(record)
        identity_path = root / "sampling_identity.json"
        identity = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
        identities.append(identity)
        receipt_path = root / "COMPLETE.json"
        timing_receipts.append(
            json.loads(receipt_path.read_text()).get("summary", {})
            if receipt_path.is_file()
            else {}
        )
        inputs.append(
            {
                "repeat": repeat,
                "statistics_path": str(stats),
                "statistics_sha256": sha256_file(stats),
                "sampling_identity_sha256": sha256_file(identity_path)
                if identity_path.is_file()
                else None,
                "completion_sha256": sha256_file(receipt_path) if receipt_path.is_file() else None,
            }
        )
        raw_path = root / "raw_packet.npz"
        if raw_path.is_file():
            with np.load(raw_path, allow_pickle=False) as raw:
                z = raw["z_raw"]
                if z.shape != (declared["n"], *record["truth"].shape) or not np.isfinite(z).all():
                    raise ValueError("Raw packet has invalid original draw/prompt/event shape")
                if "raw_sum" not in record or not np.allclose(
                    z.sum(0), record["raw_sum"], rtol=1e-12, atol=1e-14
                ):
                    raise ValueError(
                        "Raw draw ledger disagrees with saved raw sufficient statistics"
                    )
                if "raw_second_moment" in record and not np.allclose(
                    np.einsum("npe,npf->pef", z, z),
                    record["raw_second_moment"],
                    rtol=1e-12,
                    atol=1e-14,
                ):
                    raise ValueError(
                        "Raw draw second moment differs from saved sufficient statistics"
                    )
                raw_verified += 1
            inputs[-1]["raw_packet_sha256"] = sha256_file(raw_path)
        elif declared["n"] == 1024:
            missing.append(_missing("n1024_raw_original", str(raw_path)))
    if not records:
        result = {"id": declared["id"], "status": "MISSING_INPUTS", "missing_inputs": missing}
        _publish_json(output / "SUMMARY.json", result)
        return result
    truth = records[0]["truth"]
    if truth.ndim != 2 or truth.shape[-1] != 4 or not np.isfinite(truth).all():
        raise ValueError("Finite original prompt-by-four exact response required")
    if not fixture and truth.shape != (72, 4):
        raise ValueError("Full Q1 expects the independent 72-prompt control panel")
    panel = _groups_from_original(identities, len(truth), prompt_metadata)
    arrays = {"truth": truth}
    methods = {}
    paired = {}
    mass = {}
    names = list(declared["metrics"])
    for method in names:
        key = "estimate_" + method
        if len(records) != expected or any(key not in r for r in records):
            methods[method] = _missing(
                method, "Complete repeated original estimates are unavailable"
            )
            missing.append(methods[method])
            continue
        estimate = np.asarray([record[key] for record in records])
        if estimate.shape != (expected, *truth.shape) or not np.isfinite(estimate).all():
            raise ValueError("Original estimate ledger has invalid shape/nonfinite values")
        error = estimate - truth
        arrays["estimate_" + method] = estimate
        arrays["residual_" + method] = error
        coefficients = _coefficient_report(
            method, records, [r.get("diagnostics", {}) for r in identities], expected, len(truth)
        )
        for name in ("b_" + method, "b_fold0_" + method, "b_fold1_" + method):
            if all(name in r for r in records):
                arrays[name] = np.asarray([r[name] for r in records])
        report = {
            "status": "RECOMPUTED_FROM_ORIGINAL_ESTIMATES",
            "events": residual_summary(error, truth),
            "four_event_total_mse": float(np.mean(np.sum(error**2, axis=-1))),
            "coefficients": coefficients,
        }
        if panel is not None:
            groups = list(dict.fromkeys(r["group"] for r in panel))
            weights = np.asarray([r.get("weight", 1.0) for r in panel], float)
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("Fixed prompt weights must be finite/nonnegative")
            grouped_error, grouped_truth = [], []
            for group in groups:
                idx = [i for i, r in enumerate(panel) if r["group"] == group]
                w = weights[idx]
                if w.sum() <= 0:
                    raise ValueError("Fixed group has no positive prompt weight")
                w = w / w.sum()
                grouped_error.append(np.einsum("rpe,p->re", error[:, idx], w))
                grouped_truth.append(w @ truth[idx])
            grouped_error = np.stack(grouped_error, axis=1)
            report["groups"] = {
                "group_order": groups,
                "targets": residual_summary(grouped_error, np.asarray(grouped_truth)),
            }
            arrays["group_residual_" + method] = grouped_error
        else:
            report["groups"] = _missing(
                "fixed_group_metrics", "Hash-bound original prompt metadata unavailable"
            )
        methods[method] = report
        mass[method] = {"post_correction": _flat_summary(estimate.sum(-1))}
        if "main_subset_estimate_" + method in records[0] and all(
            "main_subset_estimate_" + method in r for r in records
        ):
            subset = np.asarray([r["main_subset_estimate_" + method] for r in records])
            arrays["paired_main_residual_" + method] = subset - truth
            paired[method] = {
                "scope": "MAIN_DRAWS_ONLY_DIAGNOSTIC_NOT_COST_MATCHED_BASELINE",
                "events": residual_summary(subset - truth, truth),
            }
    if all("raw_mass_residual" in r for r in records):
        raw_mass = np.asarray([r["raw_mass_residual"] for r in records])
        arrays["raw_mass_residual"] = raw_mass
        for item in mass.values():
            item["pre_correction_all_draws"] = _flat_summary(raw_mass)
    else:
        missing.append(_missing("raw_mass_residual", "Raw four-channel mass original missing"))
    if all("main_subset_estimate_RAW4" in r for r in records):
        arrays["raw_main_mass_residual"] = np.asarray(
            [r["main_subset_estimate_RAW4"].sum(-1) for r in records]
        )
        for name, item in mass.items():
            if name.startswith("PILOT_"):
                item["pre_correction_main_draws"] = _flat_summary(arrays["raw_main_mass_residual"])
    result = {
        "id": declared["id"],
        "n": declared["n"],
        "proposal": declared["proposal"],
        "case": declared["case"],
        "seed": declared.get("seed"),
        "arm": declared.get("arm"),
        "anchor": declared.get("anchor"),
        "bank": declared.get("bank"),
        "expected_repeats": expected,
        "observed_repeats": len(records),
        "prompt_count": len(truth),
        "prompt_axis_metadata": panel,
        "methods": methods,
        "paired_main_only": paired,
        "mass": mass,
        "timings": _timings(timing_receipts, expected),
        "original_unit_wall_seconds": declared.get("wall_seconds"),
        "raw_draw_moment_verified_repeats": raw_verified,
        "missing_inputs": missing,
        "data_ownership": "IMMUTABLE_ORIGINALS_READ_ONLY",
        "definitions": DEFINITIONS,
    }
    _publish_npz(output / "EXACT_RECORDS.npz", arrays)
    _publish_json(output / "ORIGINALS.json", inputs)
    _publish_json(output / "SUMMARY.json", result)
    return result


def _all_missing(value):
    missing = []
    if isinstance(value, dict):
        if value.get("status") == "MISSING_INPUTS":
            missing.append(value)
        for key, child in value.items():
            if key != "definitions":
                missing.extend(_all_missing(child))
    elif isinstance(value, list):
        for child in value:
            missing.extend(_all_missing(child))
    return missing


def analyze_observation(
    config, observation_roots, out, *, fixture=False, resume=False, prompt_metadata=None
):
    """Publish Q1 descriptive tables and exact repeated records from complete roots.

    fixture=True is only for tiny engineering tests and marks every report as
    unsuitable for scientific or launch authorization. Full roots must be
    nonpilot Q1 development evidence; group/timing omissions remain explicit.
    """
    if not fixture and platform.system() != "Linux":
        raise RuntimeError("Full Q1 analysis must run on server CPUs")
    from ..core import frozen_writer
    from .cpu_campaign import _verify_complete

    roots = [Path(p).resolve() for p in observation_roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Distinct completed Q1 source roots required")
    out = Path(out).resolve()
    if any(out.is_relative_to(root) or root.is_relative_to(out) for root in roots):
        raise ValueError("Derived output must not overlap immutable original roots")
    manifests = []
    for root in roots:
        manifest = _verify_complete(root)
        summary = manifest["summary"]
        if (
            summary.get("stage") != "Q1"
            or summary.get("pilot") is not False
            or summary.get("scientific_status") != "DEVELOPMENT_ONLY"
        ):
            raise ValueError("Completed nonpilot Q1 development evidence required")
        if manifest["binding"].get("config_sha256") != canonical_hash(config):
            raise ValueError("Q1 original config does not match requested analysis")
        manifests.append(manifest)
    binding = {
        "kind": "Q1_ORIGINALS_DERIVED_SUMMARY",
        "fixture": fixture,
        "config_sha256": canonical_hash(config),
        "analysis_source_sha256": sha256_file(__file__),
        "inputs": {str(root): sha256_file(root / "COMPLETE.json") for root in roots},
        "prompt_metadata": str(Path(prompt_metadata).resolve()) if prompt_metadata else None,
        "prompt_metadata_sha256": sha256_file(prompt_metadata)
        if prompt_metadata and Path(prompt_metadata).is_file()
        else None,
    }
    with frozen_writer(out):
        if (out / "COMPLETE.json").exists():
            if not resume:
                raise FileExistsError("Derived Q1 output exists; explicit resume required")
            verify_manifest(out, binding)
            return json.loads((out / "Q1_ANALYSIS.json").read_text())
        if (out / "BINDING.json").exists() and not resume:
            raise FileExistsError("Partial derived Q1 output exists; explicit resume required")
        _publish_json(out / "BINDING.json", binding)
        results = []
        seen = set()
        pooled = defaultdict(list)
        for root, manifest in zip(roots, manifests, strict=True):
            summary = manifest["summary"]
            if summary.get("unit_count") != len(summary["units"]):
                raise ValueError("Q1 declared unit count is inconsistent")
            for declared in summary["units"]:
                name = declared["id"]
                if name in seen or Path(name).name != name or name in (".", ".."):
                    raise ValueError("Duplicate or unsafe Q1 unit identity")
                seen.add(name)
                unit_result = _analyze_unit(
                    root / "units" / name,
                    declared,
                    out / "units" / name,
                    fixture=fixture,
                    prompt_metadata=prompt_metadata,
                )
                results.append(unit_result)
                for method, report in unit_result.get("methods", {}).items():
                    if report.get("status") == "RECOMPUTED_FROM_ORIGINAL_ESTIMATES":
                        pooled[
                            (method, declared["n"], declared["proposal"], declared["case"])
                        ].append(unit_result)
        pooled_rows = []
        for (method, n, proposal, case), units in sorted(pooled.items()):
            residuals = []
            for unit in units:
                with np.load(
                    out / "units" / unit["id"] / "EXACT_RECORDS.npz", allow_pickle=False
                ) as arrays:
                    residuals.append(arrays["residual_" + method].reshape(-1, 4))
            all_residuals = np.concatenate(residuals, axis=0)
            total_mse = float(np.mean(np.sum(all_residuals**2, axis=-1)))
            for event, index, sign in [
                *((e, i, 1) for i, e in enumerate(EVENTS)),
                ("delta_v", 3, -1),
            ]:
                metrics = _flat_summary(sign * all_residuals[:, index])
                within = [
                    (
                        u["methods"][method]["events"][event][
                            "mean_within_prompt_repeat_sample_variance"
                        ],
                        u["prompt_count"],
                    )
                    for u in units
                ]
                metrics["mean_within_unit_prompt_repeat_sample_variance"] = (
                    sum(v * p for v, p in within) / sum(p for _, p in within)
                    if all(v is not None for v, _ in within)
                    else None
                )
                pooled_rows.append(
                    {
                        "method": method,
                        "n": n,
                        "proposal": proposal,
                        "case": case,
                        "event": event,
                        "units": len(units),
                        "four_event_total_mse": total_mse,
                        **metrics,
                    }
                )
        missing = [{"unit": unit["id"], **item} for unit in results for item in _all_missing(unit)]
        result = {
            "stage": "Q1_DESCRIPTIVE_ANALYSIS",
            "fixture": fixture,
            "scientific_status": "FIXTURE_NOT_AUTHORIZATION"
            if fixture
            else "DEVELOPMENT_ONLY_NOT_CERTIFIED",
            "unit_count": len(results),
            "pooled_rows": pooled_rows,
            "missing_input_count": len(missing),
            "definitions": DEFINITIONS,
            "source_bindings": binding["inputs"],
            "originals_modified": False,
            "new_samples_generated": 0,
            "uncertainty_intervals_certified": False,
            "missing_inputs": missing,
        }
        _publish_json(out / "Q1_ANALYSIS.json", result)
        _publish_json(out / "MISSING_INPUTS.json", missing)
        flat = []
        for row in pooled_rows:
            flat.append(
                {key: value for key, value in row.items() if not isinstance(value, dict)}
                | {
                    f"signed_{key}": value
                    for key, value in row["signed_residual_quantiles"].items()
                }
                | {
                    f"absolute_{key}": value
                    for key, value in row["absolute_residual_quantiles"].items()
                }
            )
        stream = io.StringIO(newline="")
        if flat:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        csv_path = out / "Q1_POOLED_ERRORS.csv"
        payload = stream.getvalue().encode()
        if csv_path.exists():
            if csv_path.read_bytes() != payload:
                raise ValueError("Immutable pooled table differs")
        else:
            atomic_bytes(csv_path, payload)
        lines = [
            "# Q1 观测统计汇总",
            "",
            f"状态：{result['scientific_status']}。",
            "",
            (
                f"已读取 {len(results)} 个完整观测单元；缺失指标记录 {len(missing)} 条。"
                "原件未修改，未生成新样本。"
            ),
            "",
            "## 指标定义",
            "",
            (
                "- 逐事件与 delta_v=-delta_pI 的偏差、MSE、MAE和分位数"
                "均从原始重复估计减去精确真值重算。"
            ),
            "- pooled 方差含不同题目的偏差差异；测量方差逐题在重复间计算，再平均。二者分别报告。",
            "- 跨单元分位数从原始残差拼接重算；残差分位数不是置信区间。",
            (
                "- 系数方差、L1、fallback 与修正前后质量残差保存在各单元 SUMMARY.json；"
                "crossfit 两折系数分别保留。"
            ),
            (
                "- 计时按实际保存的组合块汇总，不拆分推测的生成、评分或标签时间。"
                "缺少原始回执时标 MISSING_INPUTS。"
            ),
            "",
            "## 原始记录",
            "",
            "Q1_POOLED_ERRORS.csv：按方法、样本量、proposal、策略情形分别汇总。",
            "",
            (
                "units/*/EXACT_RECORDS.npz：完整 repeat x prompt x event "
                "估计、真值、残差、系数及质量残差。"
            ),
            "units/*/ORIGINALS.json：被读取的原始文件与 SHA-256。",
            "",
            "本报告不认证在线 SSVC 有效，也不将工程完成等同于统计有效。",
            "",
        ]
        report_path = out / "Q1_ANALYSIS_zh.md"
        report = "\n".join(lines).encode()
        if report_path.exists():
            if report_path.read_bytes() != report:
                raise ValueError("Immutable Q1 text report differs")
        else:
            atomic_bytes(report_path, report)
        finalize_run(
            out,
            binding,
            metadata={
                "stage": "Q1_DESCRIPTIVE_ANALYSIS",
                "fixture": fixture,
                "unit_count": len(results),
            },
        )
        return result
