"""Read-only CPU acceptance of completed R3-warm evidence, without model weights.

The library accepts already verified upstream bindings. The CLI either rebuilds
them recursively or verifies the immutable source, bound files and exact warm
binding of a previously completed recursive preflight before reusing it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tempfile
from pathlib import Path

from .core import canonical_hash, file_hash, write_json
from .finalize_r0 import verify_manifest
from .next_stage_runtime import _stage
from .optimizer_fork import load_checkpoint, state_hash
from .r3_gate import (
    _bank_bindings,
    _check_optimizer_state,
    _check_raw_rows,
    _check_reuse,
    _check_reuse_bindings,
    _check_scores,
    _ledger,
)
from .r3_report_gate import (
    _compare_response,
    _csv_value,
    _json,
    _same_json,
    validate_response_artifacts,
)
from .r3_runtime import LAMBDAS, _load_plan, build_requests, validate_sample_ledger
from .r3_warm_response import analyze_direct_validation
from .r3_warm_runtime import COUPLING, build_direct_requests
from .r4_gate import _check_finished_source, _check_source, _committed_sources

KIND = "REAL_CUDA_FORK"
REQUIRED = (
    "identity.json",
    "runtime_lock.json",
    "gate_binding.json",
    "bank_manifest.json",
    "samples.jsonl",
    "origin.pt",
    "reuse_decision.json",
    "candidate_manifest.json",
    "joint_advantage_checks.json",
    "gradients_summary.parquet",
    "paired_response.csv",
    "support_control_report.md",
    "response_bank_00.json",
    "response_bank_06.json",
    "runtime_profile.json",
    "report_zh.md",
    "warm_report.md",
    "direct_coupling.json",
    "direct_manifest.json",
    "direct_validation_bank_00.json",
    "direct_validation_bank_06.json",
    "direct_validation_effects.csv",
)


def _eq(actual, expected, name):
    _same_json(actual, expected, "R3-warm " + name)


def _path(root, files, value):
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root) or str(path.relative_to(root)) not in files:
        raise ValueError("Warm evidence path is absent from its manifest or escapes stage")
    return path


def _parameters(state):
    return state_hash({name: state_hash(tensor) for name, tensor in state["parameters"].items()})


def _atomic(root, files, parent, identity, result):
    completed = [p for p in parent.glob("attempt_*") if (p / "completed.json").exists()]
    if len(completed) != 1:
        raise ValueError("Warm atomic bank must have one successful attempt")
    attempt = completed[0].resolve()
    for name in ("identity.json", "manifest.json", "result.json", "completed.json"):
        _path(root, files, attempt / name)
    _eq(_json(attempt, "identity.json"), identity, "atomic identity")
    _eq(_json(attempt, "result.json"), result, "atomic saved result")
    _eq(
        _json(attempt, "completed.json"),
        {
            "status": "PASS",
            "identity_hash": canonical_hash(identity),
            "manifest_sha256": file_hash(attempt / "manifest.json"),
        },
        "atomic completion",
    )
    verify_manifest(attempt)
    return attempt


def _state_schema(value):
    if isinstance(value, dict):
        return {k: _state_schema(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, [_state_schema(v) for v in value])
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return (str(type(value)), list(value.shape), str(value.dtype))
    return str(type(value))


def _check_mature(state, *, step, origin=None):
    """Check actual finite Adam tensors, not a reported checkpoint number."""
    import torch

    _check_optimizer_state(state, step=step)
    if origin is not None:
        _eq(sorted(state), sorted(origin), "candidate full state fields")
        _eq(state["metadata"], origin["metadata"], "candidate origin metadata")
        _eq(_state_schema(state["rng"]), _state_schema(origin["rng"]), "candidate RNG schema")
        _eq(
            _state_schema(state["parameters"]),
            _state_schema(origin["parameters"]),
            "candidate parameter names/shape/dtype",
        )
        _eq(
            state["optimizer"]["param_groups"],
            origin["optimizer"]["param_groups"],
            "candidate optimizer topology/options",
        )
    if state["scheduler"] is not None:
        raise ValueError("Warm fork scheduler must remain absent")
    indices = [i for group in state["optimizer"]["param_groups"] for i in group["params"]]
    for i, parameter in zip(indices, state["parameters"].values(), strict=True):
        moments = state["optimizer"]["state"][i]
        if any(
            moments[k].shape != parameter.shape or moments[k].dtype != parameter.dtype
            for k in ("exp_avg", "exp_avg_sq")
        ) or torch.any(moments["exp_avg_sq"] < 0):
            raise ValueError("Warm Adam moment shape/dtype/second moment changed")


def _origin(root, files, identity, runtime, checkpoint):
    source = Path(checkpoint["path"]).resolve()
    if file_hash(source) != checkpoint["file_sha256"]:
        raise ValueError("R4 warm source checkpoint bytes changed")
    parent = load_checkpoint(source, checkpoint["identity"])
    origin = load_checkpoint(
        _path(root, files, "origin.pt"), {**identity, "unit": "initial_origin"}
    )
    _check_mature(origin, step=64)
    _eq(state_hash(parent), checkpoint["state_hash"], "R4 origin state")
    _eq(state_hash(origin), checkpoint["state_hash"], "warm origin full state/RNG")
    _eq(_parameters(origin), checkpoint["parameter_hash"], "warm origin parameters")
    _eq(state_hash(origin["optimizer"]), checkpoint["optimizer_state_hash"], "warm origin Adam")
    metadata = origin["metadata"]
    keys = metadata.get("completed_sample_keys", [])
    if (
        metadata.get("arm") != "X_BASE"
        or metadata.get("checkpoint_step") != 64
        or metadata.get("sampler", {}).get("position") != 64
        or not isinstance(keys, list)
        or len(keys) != 2048
        or len(set(keys)) != 2048
    ):
        raise ValueError("Warm origin is not complete X_BASE step64 with its sampler ledger")
    _eq(runtime["origin_hash"], state_hash(origin), "runtime origin hash")
    _eq(runtime["origin_file_sha256"], files["origin.pt"], "origin file")
    _eq(runtime["optimizer_initial_hash"], state_hash(origin["optimizer"]), "initial optimizer")
    return origin


def _raw_inputs(root, records, plan, identity, config, state, runtime, gate, inputs):
    _check_raw_rows(records, plan, identity, config, state, runtime["model_audit"])
    prompts = {p["prompt_id"]: p for p in [*plan["train_prompts"], *plan["control_prompts"]]}
    expected_step = 64 if identity.get("unit") != "direct_control" else 65
    for row in records.values():
        if (
            type(row.get("elapsed")) not in (int, float)
            or not math.isfinite(row["elapsed"])
            or row["elapsed"] < 0
            or row.get("elapsed_unit") != "seconds"
            or row.get("elapsed_scope") != "generation_meter_scope_including_synchronization"
            or row.get("memory_measurement_status") != "CUDA_MEASURED"
            or not isinstance(row.get("peak_memory"), dict)
            or any(
                type(row["peak_memory"].get(k)) is not int or row["peak_memory"][k] <= 0
                for k in ("peak_cuda_bytes", "peak_cuda_reserved_bytes", "peak_cpu_rss_bytes")
            )
            or type(row.get("runtime_forward_by_reason", {}).get("generation")) is not int
            or row["runtime_forward_by_reason"]["generation"] < 1
        ):
            raise ValueError("Warm measured generation telemetry missing or simulated")
        if (
            row.get("checkpoint_step") != 64
            or row.get("origin_checkpoint_step") != 64
            or row.get("optimizer_step") != expected_step
        ):
            raise ValueError("Warm generation actual optimizer/checkpoint step changed")
        inputs.check(prompts[row["prompt_id"]], row)


def _processor_inputs(runtime, gate):
    from .r2_result_audit import load_processor_adapter, validate_processor_certificate
    from .r4_gate import _PreparedInputs

    validate_processor_certificate(runtime, gate)
    adapter = load_processor_adapter(runtime)
    return _PreparedInputs(adapter, gate["config"]["data_root"])


def _forks(root, files, identity, origin, bank_bindings, proposal):
    origin_hash = state_hash(origin)
    reuse = _json(root, "reuse_decision.json")
    allowed = _check_reuse(reuse, origin_hash)
    _check_reuse_bindings(root, files, reuse, identity, origin_hash, bank_bindings)
    for index in (1, 11):
        value = reuse["validation"][str(index)]
        unit = {
            **identity,
            "unit": "reuse_validation",
            "bank_index": index,
            "origin_hash": origin_hash,
            "sample_hash": bank_bindings[str(index)]["sample_hash"],
        }
        _atomic(
            root,
            files,
            root / "validation" / f"bank_{index:02d}",
            unit,
            {k: v for k, v in value.items() if k not in {"attempt", "reused_completed_unit"}},
        )
    manifest = _json(root, "candidate_manifest.json")
    banks = manifest["banks"]
    joint = _json(root, "joint_advantage_checks.json")
    if (
        manifest.get("distinct_candidates") != 60
        or manifest.get("committed_candidates") != 0
        or manifest.get("response_bank_indices") != [0, 6]
        or set(banks) != {str(i) for i in range(12)}
    ):
        raise ValueError("Warm candidate manifest is incomplete")
    _eq(joint["reuse_validation"], reuse["validation"], "reuse report")
    _eq(joint["bank_advantage_and_gradient_audits"], banks, "gradient report")
    selected, ids = {}, set()
    for index, summary in banks.items():
        unit = {
            **identity,
            "unit": "bank_forks",
            "bank_index": int(index),
            "origin_hash": origin_hash,
            "reuse_authorized": allowed,
            "sample_hash": bank_bindings[index]["sample_hash"],
        }
        _eq(summary["identity"], unit, "bank identity")
        attempt = _atomic(root, files, root / "forks" / f"bank_{int(index):02d}", unit, summary)
        if (
            summary.get("passed") is not True
            or summary.get("status") != "PASS"
            or summary.get("origin_state_hash") != origin_hash
            or summary.get("origin_restored") is not True
            or summary.get("candidate_optimizer_updates") != 5
            or summary.get("replay_optimizer_updates") != 1
            or summary.get("lambda_zero_replay", {}).get("passed") is not True
            or summary.get("lambda_zero_replay", {}).get("bitwise_equal") is not True
            or (summary.get("reuse_adopted") is True and not allowed)
            or any(
                summary.get("answer_diagnostics", {}).get(a, {}).get("optimizer_updates") != 0
                for a in ("A_BASE", "A_VALID")
            )
            or (
                summary.get("all_valid_invariant", {}).get("applicable") is True
                and summary["all_valid_invariant"].get("passed") is not True
            )
            or sorted(c["lambda"] for c in summary["candidates"]) != list(LAMBDAS)
        ):
            raise ValueError("Warm fork restoration/replay/gradient controls incomplete")
        for candidate in summary["candidates"]:
            cid = candidate["candidate_id"]
            if cid in ids or cid != f"bank_{int(index):02d}_lambda_{candidate['lambda']:g}":
                raise ValueError("Warm candidate identity is duplicate or changed")
            ids.add(cid)
            path = _path(root, files, candidate["checkpoint_path"])
            if path.parent != attempt / "engine":
                raise ValueError("Warm candidate is outside its successful bank attempt")
            if file_hash(path) != candidate["checkpoint_sha256"]:
                raise ValueError("Warm candidate checkpoint bytes changed")
            expected = {
                **unit,
                "candidate_id": cid,
                "lambda": candidate["lambda"],
                "origin_state_hash": origin_hash,
                "bank_hash": bank_bindings[index]["bank_hash"],
            }
            ci = candidate["checkpoint_identity"]
            if any(ci.get(k) != v for k, v in expected.items()):
                raise ValueError("Warm candidate checkpoint identity changed")
            state = load_checkpoint(path, ci)
            _check_mature(state, step=65, origin=origin)
            _eq(state_hash(state), candidate["state_hash"], "candidate full state")
            _eq(_parameters(state), candidate["parameter_hash"], "candidate parameters")
            _eq(state_hash(state["optimizer"]), candidate["optimizer_state_hash"], "candidate Adam")
            if int(index) in (0, 6):
                _check_scores(root, identity, candidate, proposal)
                if candidate["lambda"] in (0, 1):
                    selected[cid] = (candidate, attempt)
            del state
    if len(ids) != 60 or len(selected) != 4:
        raise ValueError("Warm fork/direct candidate coverage incomplete")
    return banks, selected, allowed


def _direct_rows(results):
    rows = []
    for index, result in results.items():
        for comparison, value in result["comparisons"].items():
            for scope, metrics in value["responses"].items():
                for metric, measurement in metrics.items():
                    row = {
                        "bank_index": index,
                        "comparison": comparison,
                        "scope": scope,
                        "metric": metric,
                        **{k: v for k, v in measurement.items() if k != "ci"},
                    }
                    for kind, interval in measurement["ci"].items():
                        row.update({f"{kind}_CI_{k}": v for k, v in interval.items()})
                    rows.append(row)
    return rows


def _check_direct_csv(root, results):
    rows = _direct_rows(results)
    keys = ("bank_index", "comparison", "scope", "metric")
    expected = {tuple(row[k] for k in keys): row for row in rows}
    if len(rows) != 168 or len(expected) != 168:
        raise ValueError("Warm direct report must contain 168 distinct measured cells")
    fields = set(rows[0])
    seen = set()
    with (root / "direct_validation_effects.csv").open(newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if (
            not reader.fieldnames
            or len(reader.fieldnames) != len(fields)
            or set(reader.fieldnames) != fields
        ):
            raise ValueError("Warm direct CSV columns changed")
        for row in reader:
            key = (int(row["bank_index"]), *(row[k] for k in keys[1:]))
            if set(row) != fields or key not in expected or key in seen:
                raise ValueError("Warm direct CSV has missing/duplicate/unknown cells")
            for name, value in expected[key].items():
                # csv.DictWriter serializes diagnostic lists with Python repr.
                wanted = str(value) if isinstance(value, (list, dict)) else value
                if _csv_value(row[name], wanted) != wanted:
                    raise ValueError(f"Warm direct CSV differs from saved JSON at {key}/{name}")
            seen.add(key)
    if seen != set(expected):
        raise ValueError("Warm direct CSV is incomplete")
    return len(seen)


def _compare_direct(stored, recomputed):
    """Check primitive estimates with the established float64 rounding contract.

    Width/effect ratios can amplify tiny cancellation errors. Validate these
    independently against the accepted saved primitive numerator/denominator,
    rather than relaxing the tolerance for the primitive measurements.
    """
    for comparison in stored["comparisons"].values():
        for metrics in comparison["responses"].values():
            for measurement in metrics.values():
                for interval in measurement["ci"].values():
                    low, high, width = (interval[k] for k in ("low", "high", "half_width"))
                    if width is None:
                        if low is not None or high is not None:
                            raise ValueError("Warm direct CI undefined bounds disagree")
                    elif (
                        type(width) is not float
                        or not math.isfinite(width)
                        or width < 0
                        or type(low) is not float
                        or type(high) is not float
                        or not math.isfinite(low)
                        or not math.isfinite(high)
                        or low > high
                        or abs(width - (high - low) / 2)
                        > 64 * sys.float_info.epsilon * max(1, abs(width), abs(high), abs(low))
                    ):
                        raise ValueError("Warm direct CI width/domain/bounds relationship changed")
                    for name in (
                        "half_width_to_abs_point_estimate",
                        "half_width_to_abs_predicted_delta",
                    ):
                        ratio = interval[name]
                        if ratio is not None and (
                            type(ratio) is not float or not math.isfinite(ratio) or ratio < 0
                        ):
                            raise ValueError("Warm direct CI ratio must be nonnegative and finite")
    observation = {
        "control_IS": _compare_response(
            stored["control_IS"], recomputed["control_IS"], "warm direct IS"
        )
    }
    differences = []

    def compare(a, b, path=()):
        if path == ("control_IS",):
            return
        if type(a) is not type(b):
            raise ValueError(f"Warm direct response type changed at {path}")
        if isinstance(a, dict):
            if set(a) != set(b):
                raise ValueError(f"Warm direct response fields changed at {path}")
            for key in sorted(a):
                compare(a[key], b[key], (*path, key))
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError("Warm direct response list coverage changed")
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                compare(left, right, (*path, index))
        else:
            derived = path[:1] == ("comparisons",) and (
                (len(path) == 6 and path[-1] in {"predicted_delta", "observed_delta", "residual"})
                or (
                    len(path) == 8
                    and path[5] == "ci"
                    and path[-1]
                    in {
                        "low",
                        "high",
                        "half_width",
                        "half_width_to_abs_point_estimate",
                        "half_width_to_abs_predicted_delta",
                    }
                )
            )
            if derived and len(path) == 8 and path[-1].startswith("half_width_to_abs_"):
                metric = stored["comparisons"][path[1]]["responses"][path[3]][path[4]]
                width = metric["ci"][path[6]]["half_width"]
                denominator = (
                    metric["predicted_delta"]
                    if path[-1].endswith("predicted_delta")
                    else metric[path[6]]
                )
                b = None if width is None or denominator in (None, 0) else width / abs(denominator)
                if type(a) is not type(b):
                    raise ValueError("Warm direct CI ratio undefined status changed")
            if canonical_hash(a) == canonical_hash(b):
                return
            bound = (
                64 * sys.float_info.epsilon * max(1, abs(a), abs(b))
                if derived and type(a) is float and math.isfinite(a) and math.isfinite(b)
                else 0
            )
            if not bound or abs(a - b) > bound:
                raise ValueError(f"Warm direct response differs at {path}")
            differences.append(
                {
                    "path": list(path),
                    "stored": a,
                    "recomputed": b,
                    "absolute_difference": abs(a - b),
                    "allowed_bound": bound,
                }
            )

    compare(stored, recomputed)
    return {**observation, "differences": differences}


def _direct(root, files, identity, runtime, origin, gate, plan, proposal, banks, selected, inputs):
    request_identity = {**identity, "origin_hash": state_hash(origin)}
    coupling = {
        **COUPLING,
        "identity": request_identity,
        "bank_indices": [0, 6],
        "lambdas": [0.0, 1.0],
        "total_direct_outputs": 3072,
    }
    _eq(_json(root, "direct_coupling.json"), coupling, "direct coupling")
    manifest = _json(root, "direct_manifest.json")
    _eq(manifest["identity"], request_identity, "direct manifest identity")
    _eq(manifest["coupling"], coupling, "direct manifest coupling")
    if manifest.get("direct_outputs") != 3072 or manifest.get("candidate_commit") is not False:
        raise ValueError("Warm direct manifest count/commit changed")
    listed = manifest["candidates"]
    by_id = {c["candidate_id"]: c for c in listed}
    if len(listed) != 4 or set(by_id) != set(selected):
        raise ValueError("Warm direct candidate manifest coverage changed")
    collected, sample_keys = {0: [], 6: []}, set()
    for cid, (candidate, attempt) in selected.items():
        state = load_checkpoint(candidate["checkpoint_path"], candidate["checkpoint_identity"])
        index = int(cid.split("_")[1])
        direct_identity = {
            **request_identity,
            "unit": "direct_control",
            "candidate_id": cid,
            "initial_adapter_hash": candidate["parameter_hash"],
            "candidate_state_hash": state_hash(state),
            "candidate_optimizer_state_hash": candidate["optimizer_state_hash"],
            "coupling_hash": canonical_hash(coupling),
        }
        requests = build_direct_requests(
            plan, request_identity, {**candidate, "optimizer_step": 65}, index, state_hash(state)
        )
        prefix = Path("direct_samples") / cid
        for name in ("identity.json", "request_manifest.json", "samples.jsonl"):
            _path(root, files, prefix / name)
        _eq(_json(root, prefix / "identity.json"), direct_identity, "direct policy identity")
        _eq(
            _json(root, prefix / "request_manifest.json"),
            {"identity": direct_identity, "requests": requests},
            "direct requests/CRN",
        )
        rows = _ledger(root / prefix / "samples.jsonl")
        validate_sample_ledger(rows, requests)
        if (
            len(rows) != 768
            or set(rows) != {r["sample_key"] for r in requests}
            or sample_keys & set(rows)
        ):
            raise ValueError("Warm direct sample coverage is incomplete or duplicate")
        sample_keys.update(rows)
        _raw_inputs(root, rows, plan, direct_identity, gate["config"], state, runtime, gate, inputs)
        expected = {
            "bank_index": index,
            "candidate_id": cid,
            "lambda": candidate["lambda"],
            "candidate_state_hash": state_hash(state),
            "origin_checkpoint_step": 64,
            "candidate_optimizer_step": 65,
            "candidate_parameter_hash": candidate["parameter_hash"],
            "candidate_optimizer_state_hash": candidate["optimizer_state_hash"],
            "source_fork_attempt": str(attempt),
            "new_outputs": 768,
            "sample_file": str(root / prefix / "samples.jsonl"),
            "sample_file_sha256": files[str(prefix / "samples.jsonl")],
        }
        for key in ("source_fork_attempt", "sample_file"):
            if Path(by_id[cid][key]).resolve() != Path(expected[key]).resolve():
                raise ValueError("Warm direct source path differs from accepted evidence")
            expected[key] = by_id[cid][key]
        _eq(by_id[cid], expected, "direct candidate provenance")
        collected[index].extend(rows[r["sample_key"]] for r in requests)
        del state
    results, roundoff = {}, {}
    for index in (0, 6):
        scores = {}
        for candidate in banks[str(index)]["candidates"]:
            cid = candidate["candidate_id"]
            scores[cid] = {
                r["proposal_sample_key"]: r["candidate_token_logprobs"]
                for r in _ledger(root / "control_scores" / cid / "samples.jsonl").values()
            }
        result = analyze_direct_validation(
            list(proposal.values()),
            scores,
            collected[index],
            bank_index=index,
            baseline_key=f"bank_{index:02d}_lambda_0",
            candidate_key=f"bank_{index:02d}_lambda_1",
        )
        stored = _json(root, f"direct_validation_bank_{index:02d}.json")
        roundoff[str(index)] = _compare_direct(stored, result)
        results[index] = stored
    _eq(
        manifest["statistical_status"],
        {str(i): r["status"] for i, r in results.items()},
        "direct status",
    )
    return {
        "direct_outputs": len(sample_keys),
        "direct_csv_rows": _check_direct_csv(root, results),
        "statistical_status": manifest["statistical_status"],
        "roundoff": roundoff,
    }


def _profile_check(profile):
    if (
        profile.get("language_layers_instrumented") != 32
        or profile.get("top_hook_installed") is not True
        or profile.get("vision_hooks_installed") != 1
        or profile.get("internal_recompute_status") != "MEASURED"
        or any(
            type(profile.get(k)) is not int or profile[k] < 0
            for k in ("optimizer_step_calls_observed", "backward_calls_observed")
        )
    ):
        raise ValueError("Warm forward/backward instrumentation is incomplete")


def _cost(root, files, status):
    cost = _json(root, "runtime_profile.json")
    histories = []
    for item in sorted((root / "invocations").iterdir()):
        if not item.is_dir():
            raise ValueError("Warm invocation entry is not a directory")
        relative = item.relative_to(root)
        profile = (
            _json(root, relative / "runtime_profile.json")
            if (item / "runtime_profile.json").exists()
            else {}
        )
        if profile:
            _profile_check(profile)
        complete = (item / "completion.json").exists()
        for name in ("runtime_profile.json", "completion.json"):
            if (item / name).exists():
                _path(root, files, relative / name)
        histories.append(
            {
                "invocation": item.name,
                "completed": complete,
                "observed": profile,
                "unrecorded_work_possible": not complete,
            }
        )
    if (
        not histories
        or _json(root, Path("invocations") / histories[-1]["invocation"] / "completion.json").get(
            "status"
        )
        != "PASS"
    ):
        raise ValueError("Warm final invocation is not completed")
    _eq(cost["invocations"], histories, "runtime invocation records")
    for key, field in (
        ("optimizer_steps_observed_at_least", "optimizer_step_calls_observed"),
        ("backward_calls_observed_at_least", "backward_calls_observed"),
    ):
        _eq(cost[key], sum(x["observed"].get(field, 0) for x in histories), "observed counts")
    _eq(
        cost["incomplete_invocations_with_unknown_extra_cost"],
        sum(x["unrecorded_work_possible"] for x in histories),
        "unknown cost intervals",
    )
    if (
        cost["optimizer_steps_observed_at_least"] < 80
        or cost["backward_calls_observed_at_least"] <= 0
    ):
        raise ValueError("Warm observed update/backward budget incomplete")
    _eq(
        status["details"]["runtime_counts"],
        {k: v for k, v in cost.items() if k != "invocations"},
        "completion runtime counts",
    )
    return {k: v for k, v in cost.items() if k != "invocations"}


def _reports(root, banks, status):
    from .r3_runtime import _write_reports

    responses = {i: _json(root, f"response_bank_{i:02d}.json") for i in (0, 6)}
    validation = _json(root, "reuse_decision.json")["validation"]
    direct = _json(root, "direct_manifest.json")
    original_root = Path(direct["candidates"][0]["sample_file"]).parent.parent.parent
    if original_root.resolve() != root:
        raise ValueError("Warm report output path is not the accepted stage")
    # Independent temporary report regeneration, never write the source stage.
    with tempfile.TemporaryDirectory(prefix="ssvc-warm-report-", dir=root.parent.parent) as temp:
        temporary = Path(temp)
        _write_reports(temporary, banks, responses, validation, status["details"])
        for name in ("report_zh.md", "warm_report.md", "support_control_report.md"):
            expected = (temporary / name).read_text().replace(str(temporary), str(original_root))
            stored = (root / name).read_text()
            if name != "support_control_report.md":
                # The original writer used insertion order inside its JSON fence;
                # status.json uses sorted keys. Bind JSON values, not key order.
                pattern = r"(?s)\n```json\n(.*?)\n```\n"
                left, right = re.search(pattern, stored), re.search(pattern, expected)
                if left is None or right is None:
                    raise ValueError("Warm prose report is missing its completion facts")
                _eq(json.loads(left.group(1)), status["details"], "prose completion details")
                stored = stored[: left.start(1)] + "<verified JSON>" + stored[left.end(1) :]
                expected = expected[: right.start(1)] + "<verified JSON>" + expected[right.end(1) :]
            if stored != expected:
                raise ValueError(f"Warm prose report differs from verified facts: {name}")
    return ["report_zh.md", "warm_report.md", "support_control_report.md"]


def validate_r3_warm_gate(warm_dir, gate, r2_binding, r3_binding, r4_binding):
    """Verify completed warm evidence against already accepted upstream bindings."""
    root = Path(warm_dir).resolve()
    bookends = {name: file_hash(root / name) for name in ("status.json", "manifest.json")}
    status, files = _stage(root, "R3-warm", KIND, REQUIRED)
    if any(
        (root / name).exists() for name in ("measurement_fault.json", "policy_contamination.json")
    ):
        raise ValueError("Warm unresolved measurement/policy fault")
    runtime, identity, bank = (
        _json(root, p) for p in ("runtime_lock.json", "identity.json", "bank_manifest.json")
    )
    binding = {"r0_r1": gate["binding"], "r2": r2_binding, "r3_cold": r3_binding, "r4": r4_binding}
    _eq(_json(root, "gate_binding.json"), binding, "recursive prerequisite binding")
    _eq(runtime["config"], gate["config"], "locked config")
    for upstream in (r2_binding, r3_binding, r4_binding):
        _eq(runtime["environment"], upstream["environment"], "measured environment")
    _check_source(runtime, gate)
    _check_finished_source(status, runtime)
    checkpoint = r4_binding["warm_checkpoint"]
    expected_identity = {
        "phase": "R3-warm",
        "protocol_version": gate["config"]["protocol_version"],
        "checkpoint_step": 64,
        "model_hash": canonical_hash(gate["config"]["model"]),
        "config_hash": canonical_hash(gate["config"]),
        "data_hash": canonical_hash(bank["data_binding"]),
        "initial_adapter_hash": checkpoint["parameter_hash"],
        "gate_hash": canonical_hash(gate["binding"]),
        "r2_gate_hash": canonical_hash(r2_binding),
        "source_hash": canonical_hash(runtime["source"]),
        "plan_hash": bank["plan"]["plan_hash"],
        "execution_kind": KIND,
        "r3_cold_gate_hash": canonical_hash(r3_binding),
        "r4_gate_hash": canonical_hash(r4_binding),
        "warm_checkpoint_sha256": checkpoint["file_sha256"],
    }
    for value in (identity, runtime["identity"], bank["identity"]):
        _eq(value, expected_identity, "warm identity")
    if (
        runtime.get("selected_probability_path") != "uncached_prefix_recompute"
        or runtime.get("initial_adapter_hash") != checkpoint["parameter_hash"]
        or runtime.get("frozen_base_hash")
        != gate["certificate"]["model_audit"]["frozen_parameter_hash"]
        or runtime.get("gradient_scaler") is not None
        or runtime.get("scheduler") is not None
    ):
        raise ValueError("Warm model/probability-path binding changed")
    origin = _origin(root, files, identity, runtime, checkpoint)
    plan, data = _load_plan(gate["config"]["data_root"], gate)
    _eq(bank["plan"], plan, "fixed original prompt plan")
    _eq(bank["data_binding"], data, "original dataset")
    _eq(plan["plan_hash"], r3_binding["plan_hash"], "cold/warm prompt IDs")
    request_identity = {**identity, "origin_hash": state_hash(origin)}
    requests = build_requests(plan, request_identity)
    _eq(bank["request_identity"], request_identity, "origin request identity")
    _eq(bank["requests"], requests, "on-policy request bank")
    records = _ledger(root / "samples.jsonl")
    validate_sample_ledger(records, requests)
    if len(records) != 1152 or set(records) != {r["sample_key"] for r in requests}:
        raise ValueError("Warm on-policy output coverage is incomplete")
    inputs = _processor_inputs(runtime, gate)
    _raw_inputs(root, records, plan, identity, gate["config"], origin, runtime, gate, inputs)
    proposal = {
        r["sample_key"]: records[r["sample_key"]]
        for r in requests
        if r["bank_role"] == "control_proposal"
    }
    banks, selected, allowed = _forks(
        root, files, identity, origin, _bank_bindings(plan, records), proposal
    )
    observed_roundoff = []
    report = validate_response_artifacts(
        root,
        proposal_records=list(proposal.values()),
        bank_summaries=banks,
        roundoff_observer=observed_roundoff.append,
    )
    direct = _direct(
        root, files, identity, runtime, origin, gate, plan, proposal, banks, selected, inputs
    )
    details = status["details"]
    expected_counts = {
        "raw_sample_count": 4224,
        "train_rollouts": 384,
        "control_proposal_rollouts": 768,
        "distinct_candidate_optimizer_updates": 60,
        "distinct_lambda_zero_replays": 12,
        "response_candidates": 10,
        "control_candidate_sequences_scored": 7680,
        "direct_rollouts": 3072,
        "direct_candidates": 4,
    }
    if any(type(details.get(k)) is not int or details[k] != v for k, v in expected_counts.items()):
        raise ValueError("Warm final measured counts incomplete")
    if (
        details.get("candidate_commit") is not False
        or details.get("direct_resample_cold") is not False
        or details.get("direct_resample_warm") is not True
        or details.get("reuse_authorized") is not allowed
        or any(
            details.get("scratch_restoration", {}).get(k) is not True
            for k in (
                "full_origin_state_and_rng",
                "initial_adapter",
                "warm_adam_and_sampler",
                "frozen_base",
            )
        )
    ):
        raise ValueError("Warm final origin restoration/direct policy incomplete")
    _eq(details["warm_checkpoint"], checkpoint, "final R4 checkpoint binding")
    _eq(details["direct_validation_status"], direct["statistical_status"], "final direct status")
    counts = _cost(root, files, status)
    prose = _reports(root, banks, status)
    if (
        file_hash(checkpoint["path"]) != checkpoint["file_sha256"]
        or verify_manifest(root, REQUIRED) != files
        or any(file_hash(root / name) != digest for name, digest in bookends.items())
    ):
        raise ValueError("Warm evidence changed during read-only audit")
    return {
        "status": "PASS",
        "execution_kind": "CPU_AUDIT",
        "model_weights_loaded": False,
        "gpu_work_requested": False,
        "r3_warm_dir": str(root),
        "source": runtime["source"],
        "upstream_binding_sha256": canonical_hash(binding),
        "files": files,
        "status_sha256": bookends["status.json"],
        "manifest_sha256": bookends["manifest.json"],
        "origin_state_hash": state_hash(origin),
        "warm_checkpoint": checkpoint,
        "raw_counts": expected_counts,
        "runtime_counts": counts,
        "prepared_prompt_count": len(inputs),
        "response_artifact_audit": report,
        "response_roundoff": observed_roundoff,
        "direct_validation": direct,
        "prose_reports_verified": prose,
        "safety_status": "NOT_CERTIFIED",
    }


def _cached_preflight(path, warm_dir, gate, config, data_root, r2_dir, r3_dir, r4_dir):
    from .next_stage_runtime import validate_runtime_environment

    preflight = _json(Path(path).resolve().parent, Path(path).name)
    if (
        preflight.get("status") != "PASS"
        or preflight.get("phase") != "R3-warm_PREFLIGHT"
        or preflight.get("execution_kind") != "CPU_AUDIT"
        or preflight.get("model_loaded") is not False
    ):
        raise ValueError("Cached warm preflight is not a completed read-only recursive audit")
    source = preflight["source"]
    if not isinstance(source.get("source_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", source["source_commit"]
    ):
        raise ValueError("Cached preflight source must name an immutable full Git commit")
    _eq(source["source_files"], _committed_sources(source["source_commit"]), "preflight Git source")
    _eq(preflight["config_hash"], canonical_hash(config), "preflight config")
    binding = preflight["gate_binding"]
    _eq(binding["r0_r1"], gate["binding"], "preflight R0/R1")
    _eq(
        binding,
        _json(Path(warm_dir).resolve(), "gate_binding.json"),
        "preflight vs original warm prerequisite binding",
    )
    for key, directory, path_key, files_key in (
        ("r2", r2_dir, "r2_dir", "r2_files"),
        ("r3_cold", r3_dir, "r3_cold_dir", "r3_files"),
        ("r4", r4_dir, "r4_dir", "r4_files"),
    ):
        upstream = binding[key]
        if (
            upstream.get("status") != "PASS"
            or Path(upstream[path_key]).resolve() != Path(directory).resolve()
        ):
            raise ValueError("Cached preflight upstream stage/path changed")
        _eq(verify_manifest(directory), upstream[files_key], "cached upstream file bytes")
        directory = Path(directory).resolve()
        if any(
            next(directory.rglob(name), None) is not None
            for name in ("measurement_fault.json", "policy_contamination.json")
        ):
            raise ValueError("Cached upstream contains an unresolved measurement/policy fault")
        terminal = _json(directory, "status.json")
        expected_phase = {"r2": "R2", "r3_cold": "R3-cold", "r4": "R4"}[key]
        expected_kind = {
            "r2": "REAL_CUDA_INFERENCE",
            "r3_cold": "REAL_CUDA_FORK",
            "r4": "REAL_CUDA_TRAINING",
        }[key]
        if (
            terminal.get("phase") != expected_phase
            or terminal.get("execution_kind") != expected_kind
            or terminal.get("status") != upstream.get("completion_status", "PASS")
        ):
            raise ValueError("Cached upstream final status/phase/execution kind changed")
        runtime = _json(directory, "runtime_lock.json")
        if key == "r2":
            runtime = {
                **runtime,
                "source": {k: runtime[k] for k in ("source_commit", "source_files")},
            }
        _check_source(runtime, gate)
        _check_finished_source(terminal, runtime)
    plan, data = _load_plan(data_root, gate)
    _eq(data, preflight["data_binding"], "preflight dataset")
    _eq(plan["plan_hash"], preflight["plan_hash"], "preflight plan")
    _eq(validate_runtime_environment(gate), preflight["environment"], "current environment")
    return binding


def main(argv=None):
    from .next_stage_common import load_yaml
    from .next_stage_preflight import preflight
    from .next_stage_runtime import validate_config_against_gate, validate_prerequisites

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    for name in (
        "warm-dir",
        "data-root",
        "r0-dir",
        "r1-run",
        "supplement-dir",
        "r2-dir",
        "r3-dir",
        "r4-dir",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.resolve().is_relative_to(args.warm_dir.resolve()):
        raise ValueError("Warm postflight must write a new audit outside immutable evidence")
    gate = validate_prerequisites(args.r0_dir, args.r1_run, args.supplement_dir)
    config = validate_config_against_gate(load_yaml(args.config), gate)
    if args.data_root.resolve() != Path(config["data_root"]).resolve():
        raise ValueError("Warm postflight data root differs from locked configuration")
    if args.preflight:
        binding = _cached_preflight(
            args.preflight,
            args.warm_dir,
            gate,
            config,
            args.data_root,
            args.r2_dir,
            args.r3_dir,
            args.r4_dir,
        )
    else:
        binding = preflight(
            config,
            args.data_root,
            args.r0_dir,
            args.r1_run,
            args.supplement_dir,
            args.r2_dir,
            phase="R3-warm",
            r3_dir=args.r3_dir,
            r4_dir=args.r4_dir,
        )["gate_binding"]
    result = validate_r3_warm_gate(
        args.warm_dir, gate, binding["r2"], binding["r3_cold"], binding["r4"]
    )
    result["preflight_sha256"] = file_hash(args.preflight) if args.preflight else None
    write_json(args.out, result)
    print(f"PASS: warm postflight audit written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
