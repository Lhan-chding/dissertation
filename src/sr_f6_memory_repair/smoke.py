"""Two retained rows, four paired measurements; never a full-batch admission.

Imports at module scope are standard-library only, permitting bounded metadata
checks on a login node without importing a tensor library or loading a model.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import re
import signal
import time
from pathlib import Path

SCHEMA = "srf6-memory-smoke-contract-v1"
RESULT_SCHEMA = "srf6-memory-smoke-result-v1"
STATUS = "SMOKE_NUMERICAL_PERFORMANCE_MEASURED"
ADMISSION = "NOT_FULL_BATCH_ADMISSION"
PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
PRIOR_SHA = "dfd8d792aed9fb2dc0d9a33ed4962afefb410609b974c92a34659efccc444a34"
HOST_BYTES, DISK_BYTES, DEVICE_BYTES = 8 * 2**30, 96 * 2**30, 24 * 2**30
SMOKE_PLAN = dict(
    measurements=[
        dict(name="row0_eager", row=0, storage="eager"),
        dict(name="row0_candidate", row=0, storage="candidate"),
        dict(name="row26_allspill", row=26, storage="allspill"),
        dict(name="row26_candidate", row=26, storage="candidate"),
    ],
    loss_denominator=128,
    atol=1e-6,
    rtol=1e-5,
    candidate_max_seconds=120.0,
    candidate_allocated_headroom_bytes=2 * 2**30,
    timing_scope="FIXED_ORDER_OBSERVATION_CACHE_EFFECTS_NOT_CAUSAL_SPEEDUP",
    free_memory_scope="SAMPLED_POINTS_NOT_PROCESS_MINIMUM",
    total_max_seconds=1800.0,
    controlled_interrupt_seconds=1500.0,
    restoration_grace_seconds=300.0,
    diagnostic_gradient_calls=0,
    generated_responses=0,
    optimizer_updates=0,
    archived_forward_rows=[0, 26],
    archived_forward_bound_count=27,
    row0_must_pass_before_row26=True,
    full_batch_admission=False,
    row26_eager_gradient_verified=False,
)


def _cancel_owned_alarm():
    if getattr(signal.getsignal(signal.SIGALRM), "_srf6_smoke_alarm", False):
        signal.setitimer(signal.ITIMER_REAL, 0)


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def _ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def _read(root, ref):
    # Pure reference helpers do not call the full pilot/history verifier.
    from .execution import bound_ref

    return bound_ref(Path(root), ref)


def validate_contract(path, expected_sha, *, source=True):
    """Check only the new small contract's anchored inputs, never full88 admission."""
    from .execution import strict_ref

    path = Path(path)
    contract = json.loads(strict_ref(path, expected_sha))
    require(contract.get("schema") == SCHEMA, "Wrong small-test contract")
    require(contract.get("smoke_plan") == SMOKE_PLAN, "Frozen smoke plan differs")
    require(
        (
            contract.get("max_host_bytes"),
            contract.get("max_disk_bytes"),
            contract.get("max_device_bytes"),
            contract.get("storage_backend"),
        )
        == (HOST_BYTES, DISK_BYTES, DEVICE_BYTES, "arena_v3"),
        "Frozen smoke storage bounds differ",
    )
    root = Path(contract["experiment_root"])
    prior_ref = contract["prior_v4_contract_ref"]
    require(prior_ref["sha256"] == PRIOR_SHA, "Wrong original v4 contract anchor")
    prior = _read(root, prior_ref)
    for key in (
        "experiment_root",
        "producer",
        "original_ready_ref",
        "original_audit_ref",
        "original_failure_ref",
        "step0_commit_ref",
    ):
        require(contract[key] == prior[key], "Original immutable binding differs: " + key)
    for key in ("original_ready_ref", "original_failure_ref", "step0_commit_ref"):
        _read(root, contract[key])
    # Original audit is JSONL, not a JSON document.
    ref = contract["original_audit_ref"]
    strict_ref(root / ref["path"], ref["sha256"])
    disposition = _read(root, contract["predecessor_disposition_ref"])
    require(bool(disposition), "Missing preserved predecessor disposition")
    spec_ref = contract["pilot_spec_ref"]
    require(spec_ref["sha256"] == prior["pins"]["pilot_spec_sha256"], "Wrong pilot spec")
    spec = _read(root, spec_ref)
    require(contract["source_binding_ref"] == spec["source_binding"], "Wrong original binding")
    binding = _read(root, contract["source_binding_ref"])
    if source:
        versions = binding["versions"]
        require(
            bool(versions)
            and all(v and importlib.metadata.version(k) == v for k, v in versions.items()),
            "Original numerical library versions differ",
        )
    config = _read(root, contract["config_ref"])
    require(contract["config_ref"]["sha256"] == binding["config_file_sha256"], "Wrong CONFIG")
    require(bool(config), "Empty CONFIG")
    if source:
        producer = contract["producer"]
        manifest = _read(root, producer["source_manifest_ref"])
        files = {v["path"]: v for v in manifest["files"]}
        require(len(files) == len(manifest["files"]), "Duplicate producer source path")
        selected = [v for p, v in files.items() if p.startswith("src/") and p.endswith(".py")]
        require(bool(selected), "Missing anchored producer source")
        for ref in selected:
            strict_ref(Path(producer["deployment"]) / ref["path"], ref["sha256"])
        repair = contract["repair_source"]
        require(re.fullmatch(r"[0-9a-f]{40}", repair["revision"]), "Missing actual repair revision")
        listed = {v["path"]: v for v in repair["files"]}
        require(len(listed) == len(repair["files"]), "Duplicate repair source path")
        own = Path(__file__).resolve().relative_to(Path(repair["deployment"]))
        require(own.as_posix() in listed, "Actual smoke verifier source missing")
        for ref in listed.values():
            strict_ref(Path(repair["deployment"]) / ref["path"], ref["sha256"])
        for name in ("__init__", "validation", "execution", "offload", "arena", "residency"):
            require(
                "src/sr_f6_memory_repair/" + name + ".py" in listed, "Missing repair dependency"
            )
    return contract


def _cuda_snapshot(torch, *, fixture=False):
    if fixture:
        return dict(scope="SYNTHETIC_CPU_ONLY")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(
        scope="ACTUAL_CUDA_MEASUREMENT",
        device_index=int(torch.cuda.current_device()),
        allocated=int(torch.cuda.memory_allocated()),
        reserved=int(torch.cuda.memory_reserved()),
        max_allocated=int(torch.cuda.max_memory_allocated()),
        max_reserved=int(torch.cuda.max_memory_reserved()),
        free=int(free),
        total=int(total),
    )


def _pair(evidence, tensors, originals, row):
    import torch

    from .validation import _compare_trace, _metric

    names = ("row0_eager", "row0_candidate") if row == 0 else ("row26_allspill", "row26_candidate")
    a, b = (evidence["measurements"][name] for name in names)
    forward = [_compare_trace(a["trace"], b["trace"])]
    forward += [_compare_trace(value["trace"], originals[row]) for value in (a, b)]
    left, right = (tensors[name] for name in names)
    require(
        bool(left) and left.keys() == right.keys() == evidence["parameter_shapes"].keys(),
        "Full adapter gradient inventory required",
    )
    for values in (left, right):
        require(
            all(
                v is None
                or (
                    v.dtype == torch.float32
                    and bool(torch.isfinite(v).all())
                    and list(v.shape) == evidence["parameter_shapes"][k]
                )
                for k, v in values.items()
            ),
            "Actual full FP32 gradient shape/values differ",
        )
    gradients = {k: _metric(left[k], right[k], evidence["parameter_shapes"][k]) for k in left}
    metrics = list(gradients.values()) + [v for comparison in forward for v in comparison.values()]
    require(
        all(v["finite"] and v["violations"] == 0 for v in metrics),
        "Fixed paired numerical gate failed",
    )
    for key in ("loss", "native", "aux", "kl"):
        x, y = a["loss"][key], b["loss"][key]
        require(
            math.isfinite(x) and math.isfinite(y) and abs(x - y) <= 1e-6 + 1e-5 * abs(y),
            "Fixed paired loss tolerance failed",
        )
    require(
        0 <= b["seconds"] <= SMOKE_PLAN["candidate_max_seconds"],
        "Candidate technical time limit exceeded",
    )
    if evidence["scope"] == "ACTUAL_CUDA_ONLY":
        cuda = b["cuda"]["finished"]
        require(
            cuda["max_allocated"]
            <= cuda["total"] - SMOKE_PLAN["candidate_allocated_headroom_bytes"],
            "Candidate CUDA allocated headroom below frozen 2 GiB",
        )
    from sr_f12.training import enforce_probability_gate

    from .validation import _numerical_audit

    audit = _numerical_audit([a["trace"], b["trace"]])
    require(audit["status"] == "FINITE", "Nonfinite paired probabilities")
    enforce_probability_gate(audit)
    require(
        audit["sampler_train_max_absolute_difference"] <= 5,
        "Original paired probability gate failed",
    )
    return dict(
        gradient_comparisons=gradients,
        forward_comparisons=forward,
        reference_seconds=a["seconds"],
        candidate_seconds=b["seconds"],
        reference_over_candidate=a["seconds"] / b["seconds"] if b["seconds"] else None,
    )


def recompute(evidence, tensors, originals):
    from sr_f12.training import enforce_probability_gate

    from .validation import RESTORE_KEYS, _numerical_audit

    require(evidence["plan"] == SMOKE_PLAN and len(originals) == 27, "Small-test coverage differs")
    expected_order = [m["name"] for m in SMOKE_PLAN["measurements"]]
    require(
        evidence["measurement_order"] == expected_order
        and set(evidence["measurements"]) == set(expected_order),
        "Measurement order differs",
    )
    for item in SMOKE_PLAN["measurements"]:
        actual = evidence["measurements"][item["name"]]
        require(
            actual["row"] == item["row"] and actual["storage_mode"] == item["storage"],
            "Measurement identity differs",
        )
        require(
            type(actual["seconds"]) in (int, float)
            and math.isfinite(actual["seconds"])
            and actual["seconds"] >= 0,
            "Invalid measurement duration",
        )
        phases = actual["phase_seconds"]
        require(
            set(phases)
            == {
                "prepare",
                "reference_forward",
                "policy_forward_and_loss",
                "backward",
                "copy_cleanup",
            }
            and all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in phases.values()
            )
            and abs(sum(phases.values()) - actual["seconds"]) < 1e-6,
            "Actual phase timing incomplete",
        )
        require(
            (actual["storage"] is None) == (item["storage"] == "eager"),
            "Missing actual storage evidence",
        )
    require(
        evidence["counts"]
        == dict(
            reference_forwards=4,
            policy_forwards=4,
            backward_calls=4,
            diagnostic_gradient_calls=0,
            generated_responses=0,
            optimizer_updates=0,
        ),
        "Small-test call counts differ",
    )
    restoration = evidence["restoration"]
    require(
        all(restoration.get(k) is True for k in RESTORE_KEYS)
        and restoration["restoration_errors"] == []
        and restoration["runtime_requires_reload"] is False,
        "Runtime restoration failed",
    )
    require(
        0 <= evidence["total_seconds"] <= SMOKE_PLAN["total_max_seconds"],
        "Overall technical time limit exceeded",
    )
    require(evidence["row0_checked_before_row26"] is True, "Ordinary row must pass first")
    comparisons = {str(row): _pair(evidence, tensors, originals, row) for row in (0, 26)}
    traces = [m["trace"] for m in evidence["measurements"].values()]
    audit = _numerical_audit(traces)
    require(audit["status"] == "FINITE", "Nonfinite small-test probabilities")
    enforce_probability_gate(audit)
    require(audit["sampler_train_max_absolute_difference"] <= 5, "Original probability gate failed")
    return dict(
        pairs=comparisons,
        probability_audit=audit,
        row26_scope="ALLSPILL_VS_RESIDENCY_NOT_EAGER_GRADIENT",
        admission=ADMISSION,
    )


def measure(
    runtime,
    records,
    credits,
    originals,
    output,
    *,
    fixture=False,
    progress=None,
    started=None,
    clock=time.monotonic,
):
    import torch

    from mm_core.training import capture_rng, restore_rng, trainable_state
    from sr_f6.numerical_replay import _Preserve
    from sr_f6.objective import sequence_loss

    from .offload import saved_activation_offload
    from .validation import _trace, _validate_records, _verify_prepared, verify_storage

    started = clock() if started is None else started
    root = Path(runtime.identity["run_root"])
    require(len(originals) == 27, "Exactly 27 archived forwards required")
    coefficients = _validate_records(
        runtime, records, [8] * 16, runtime.reference, credits, root, trainable_state(runtime.model)
    )
    params = {k: p for k, p in runtime.model.named_parameters() if p.requires_grad}
    preserved = _Preserve(runtime, runtime.reference)
    evidence = dict(
        plan=copy.deepcopy(SMOKE_PLAN),
        scope="SYNTHETIC_CPU_ONLY" if fixture else "ACTUAL_CUDA_ONLY",
        measurements={},
        measurement_order=[],
        counts=dict(
            reference_forwards=0,
            policy_forwards=0,
            backward_calls=0,
            diagnostic_gradient_calls=0,
            generated_responses=0,
            optimizer_updates=0,
        ),
        parameter_shapes={k: list(p.shape) for k, p in params.items()},
        row0_checked_before_row26=False,
    )
    tensors = {}
    counts = evidence["counts"]
    try:
        runtime.model.eval()
        pair_rng = None
        for item in SMOKE_PLAN["measurements"]:
            name, row, mode = item["name"], item["row"], item["storage"]
            require(
                clock() - started <= SMOKE_PLAN["total_max_seconds"],
                "Overall technical time limit exceeded",
            )
            runtime.model.zero_grad(set_to_none=True)
            gc.collect()
            if not fixture:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            if mode == "candidate":
                restore_rng(pair_rng)
            else:
                pair_rng = capture_rng()
            observation = dict(before=_cuda_snapshot(torch, fixture=fixture))
            if progress:
                progress(name + "_started", evidence, tensors)
            begin = clock()
            phase_at = begin
            phase_seconds = {}

            def timed_phase(name, durations=phase_seconds):
                nonlocal phase_at
                if not fixture:
                    torch.cuda.synchronize()
                now = clock()
                durations[name] = now - phase_at
                phase_at = now

            scratch = Path(output) / (name + "_STORAGE")
            context = (
                contextlib.nullcontext()
                if mode == "eager"
                else saved_activation_offload(
                    runtime.model,
                    max_host_bytes=HOST_BYTES,
                    max_disk_bytes=DISK_BYTES,
                    scratch_directory=scratch,
                    disk_backend="arena_v3",
                    fixture=fixture,
                    max_device_bytes=DEVICE_BYTES if mode == "candidate" else 0,
                )
            )
            with context as manager:
                record = records[row]
                prepared = runtime.prepare_input(
                    record["model_input"], root, protocol=record["protocol"]
                )
                _verify_prepared(record, prepared)
                actual = dict(record, prepared=prepared)
                timed_phase("prepare")
                with torch.no_grad():
                    reference = runtime.batch_reference_forward([actual], runtime.reference)[0]
                counts["reference_forwards"] += 1
                timed_phase("reference_forward")
                current = runtime.batch_sequence_forward(
                    [actual], purpose="training_policy", grad=True
                )[0]
                counts["policy_forwards"] += 1
                sampler = torch.tensor(
                    record["sampler_logprobs"], dtype=torch.float32, device=current.device
                )
                result = sequence_loss(
                    current,
                    sampler,
                    reference,
                    coefficients[row]["native"],
                    coefficients[row]["aux"],
                )
                trace = _trace(record, current, reference)
                losses = {k: float(result[k].detach()) for k in ("loss", "native", "aux", "kl")}
                observation["after_forward"] = _cuda_snapshot(torch, fixture=fixture)
                timed_phase("policy_forward_and_loss")
                (result["loss"] / 128).backward()
                counts["backward_calls"] += 1
                observation["after_backward"] = _cuda_snapshot(torch, fixture=fixture)
                timed_phase("backward")
                tensors[name] = {
                    k: None if p.grad is None else p.grad.detach().cpu().clone()
                    for k, p in params.items()
                }
                del result, current, reference, sampler, actual, prepared
                gc.collect()
            observation["finished"] = _cuda_snapshot(torch, fixture=fixture)
            timed_phase("copy_cleanup")
            seconds = phase_at - begin
            storage = manager.summary() if manager is not None else None
            if storage is not None:
                verify_storage(
                    storage,
                    scratch,
                    HOST_BYTES,
                    DISK_BYTES,
                    fixture=fixture,
                    disk_backend="arena_v3",
                    max_device_bytes=DEVICE_BYTES if mode == "candidate" else 0,
                )
            evidence["measurements"][name] = dict(
                row=row,
                storage_mode=mode,
                trace=trace,
                loss=losses,
                seconds=seconds,
                phase_seconds=phase_seconds,
                cuda=observation,
                storage=storage,
            )
            evidence["measurement_order"].append(name)
            if progress:
                progress(name + "_finished", evidence, tensors)
            if mode == "candidate":
                _pair(evidence, tensors, originals, row)
                if row == 0:
                    evidence["row0_checked_before_row26"] = True
        evidence["total_seconds"] = clock() - started
    except BaseException as exc:
        _cancel_owned_alarm()
        exc.smoke_partial = (evidence, tensors)
        raise
    finally:
        evidence["restoration"] = preserved.restore()
        evidence["total_seconds"] = clock() - started
    try:
        evidence["comparisons"] = recompute(evidence, tensors, originals)
    except BaseException as exc:
        _cancel_owned_alarm()
        exc.smoke_partial = (evidence, tensors)
        raise
    return evidence, tensors


def _verify_cuda(evidence, hardware):
    require(
        hardware["cuda_visible_device_count"] == 1
        and "PRO 6000" in hardware["cuda_name"]
        and bool(hardware["cuda_uuid"]),
        "Actual hardware identity missing",
    )
    for measurement in evidence["measurements"].values():
        observations = measurement["cuda"]
        order = ["before", "after_forward", "after_backward", "finished"]
        require(set(observations) == set(order), "Incomplete isolated CUDA observations")
        previous = (0, 0)
        for name in order:
            value = observations[name]
            require(value["scope"] == "ACTUAL_CUDA_MEASUREMENT", "Synthetic CUDA observation")
            keys = (
                "allocated",
                "reserved",
                "max_allocated",
                "max_reserved",
                "free",
                "total",
                "device_index",
            )
            require(
                all(type(value[k]) is int and value[k] >= 0 for k in keys),
                "Invalid actual CUDA counters",
            )
            require(
                value["allocated"] <= value["reserved"] <= value["max_reserved"]
                and value["allocated"] <= value["max_allocated"] <= value["max_reserved"]
                and value["free"] <= value["total"]
                and value["total"] == hardware["cuda_total_memory_bytes"]
                and value["device_index"] == 0
                and value["max_allocated"] >= previous[0]
                and value["max_reserved"] >= previous[1],
                "CUDA counters inconsistent inside isolated measurement",
            )
            previous = value["max_allocated"], value["max_reserved"]


def _safe_partial(value):
    """Preserve nonfinite diagnostic meaning without publishing invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {k: _safe_partial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_partial(v) for v in value]
    return value


def _write(path, value):
    from sr_f6.common import write_once

    write_once(path, value)


def _run_smoke(contract_path, expected_sha, output):
    started = time.monotonic()
    contract = validate_contract(contract_path, expected_sha)
    output = Path(output)
    require(
        output == Path(contract_path).parent / "SMOKE" and not output.exists(),
        "Fresh registered smoke output required",
    )
    require(
        output.parent == output.parent.resolve() and output.is_relative_to(PROJECT),
        "Canonical smoke output required",
    )
    output.mkdir()
    _write(
        output / "ATTEMPT.json",
        dict(
            schema=RESULT_SCHEMA, contract_sha256=expected_sha, plan=SMOKE_PLAN, admission=ADMISSION
        ),
    )
    evidence, tensors, ledger, stages = {}, {}, [], []

    def stage(name, seconds):
        stages.append(dict(stage=name, seconds=seconds, elapsed_seconds=time.monotonic() - started))
        _write(output / "STAGES" / (str(len(stages)).zfill(3) + ".json"), stages[-1])
        require(time.monotonic() - started <= 1800, "Overall technical time limit exceeded")

    def account(kind, count, metadata):
        require(kind == "extra_forward_sequences" and count == 1, "Unregistered smoke model call")
        ledger.append(dict(kind=kind, count=count, metadata=metadata))

    def progress(name, partial, values):
        stage(name, 0.0)
        _write(output / "MEASUREMENTS" / (name + ".json"), _safe_partial(partial))
        if name.endswith("_finished"):
            import torch

            key = name.removesuffix("_finished")
            with (output / "MEASUREMENTS" / (key + ".pt")).open("xb") as stream:
                torch.save(values[key], stream)

    try:
        import torch

        from sr_f6.contract import PACKAGE_CONFIG_SHA256

        from . import validation as v

        require(
            contract["config_ref"]["sha256"] == PACKAGE_CONFIG_SHA256,
            "Original package CONFIG differs",
        )
        stage("contract_verified", time.monotonic() - started)
        ready, records, credits, originals = v._originals(contract)
        stage("original_retained_evidence_loaded", 0.0)
        loader_view = dict(
            experiment_root=contract["experiment_root"], producer=contract["producer"]
        )
        runtime = v._technical_runtime(loader_view, ready, records, account, stage_callback=stage)
        runtime.identity["reused_loader_consumer"] = runtime.identity["technical_consumer"]
        runtime.identity["technical_consumer"] = RESULT_SCHEMA
        evidence, tensors = measure(
            runtime, records, credits, originals, output, progress=progress, started=started
        )
        evidence.update(identity=runtime.identity, ledger=ledger, preflight_stages=stages)
        _verify_cuda(evidence, runtime.identity["hardware"])
        require(len(ledger) == 8, "Actual eight-forward ledger incomplete")
        _write(output / "EVIDENCE.json", evidence)
        with (output / "TENSORS.pt").open("xb") as stream:
            torch.save(tensors, stream)
        result = dict(
            schema=RESULT_SCHEMA,
            status=STATUS,
            admission=ADMISSION,
            contract_sha256=expected_sha,
            producer_revision=contract["producer"]["revision"],
            repair_revision=contract["repair_source"]["revision"],
            generated_responses=0,
            optimizer_updates=0,
            evidence_ref=_ref(output / "EVIDENCE.json"),
            tensors_ref=_ref(output / "TENSORS.pt"),
        )
        proof = _cold_evidence(contract, result, output, expected_sha)
        _write(output / "COLD_VERIFICATION.json", proof)
        result["cold_verification_ref"] = _ref(output / "COLD_VERIFICATION.json")
        _write(output / "RESULT.json", result)
        return result
    except BaseException as exc:
        _cancel_owned_alarm()
        import torch

        evidence, tensors = getattr(exc, "smoke_partial", (evidence, tensors))
        _write(
            output / "PARTIAL_EVIDENCE.json",
            _safe_partial(dict(evidence=evidence, ledger=ledger, stages=stages)),
        )
        if tensors:
            with (output / "PARTIAL_TENSORS.pt").open("xb") as stream:
                torch.save(tensors, stream)
        _write(
            output / "FAILURE.json",
            dict(
                schema=RESULT_SCHEMA,
                status="TECHNICAL_FAILURE_PRESERVED",
                error_type=type(exc).__name__,
                error=str(exc),
                admission=ADMISSION,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


def run_smoke(contract_path, expected_sha, output):
    """25-minute controlled interruption leaves five minutes for restoration."""
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Existing alarm cannot be replaced")
    old = signal.getsignal(signal.SIGALRM)

    def interrupt(signum, frame):
        raise TimeoutError("Frozen 1500-second controlled smoke interruption")

    interrupt._srf6_smoke_alarm = True
    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, SMOKE_PLAN["controlled_interrupt_seconds"])
    try:
        return _run_smoke(contract_path, expected_sha, output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def verify_smoke_metadata(contract_path, expected_sha, result_ref):
    """Read-only metadata/hash check, deliberately not a numerical admission."""
    from .execution import strict_ref

    contract = validate_contract(contract_path, expected_sha, source=False)
    expected = Path(contract_path).parent / "SMOKE" / "RESULT.json"
    require(Path(result_ref["path"]) == expected, "Wrong smoke result path")
    result = _read(PROJECT, result_ref)
    require(
        result["schema"] == RESULT_SCHEMA
        and result["status"] == STATUS
        and result["admission"] == ADMISSION
        and result["contract_sha256"] == expected_sha,
        "Small-test result identity differs",
    )
    require(
        result["producer_revision"] == contract["producer"]["revision"]
        and result["repair_revision"] == contract["repair_source"]["revision"]
        and result["generated_responses"] == result["optimizer_updates"] == 0,
        "Small-test source/budget differs",
    )
    for key, filename in (("evidence_ref", "EVIDENCE.json"), ("tensors_ref", "TENSORS.pt")):
        require(
            Path(result[key]["path"]) == expected.parent / filename, "Wrong smoke evidence path"
        )
        if key == "evidence_ref":
            strict_ref(result[key]["path"], result[key]["sha256"])
        else:
            require(
                re.fullmatch(r"[0-9a-f]{64}", result[key]["sha256"])
                and Path(result[key]["path"]).is_file()
                and not Path(result[key]["path"]).is_symlink(),
                "Missing bound tensor evidence",
            )
    proof = _read(PROJECT, result["cold_verification_ref"])
    require(
        proof["status"] == "VERIFIED_ACTUAL_TWO_ROW_SMOKE"
        and proof["admission"] == ADMISSION
        and proof["evidence_ref"] == result["evidence_ref"]
        and proof["tensors_ref"] == result["tensors_ref"]
        and proof["contract_sha256"] == expected_sha,
        "Missing cold numerical verification",
    )
    require(not (expected.parent / "FAILURE.json").exists(), "Failed smoke cannot authorize")
    return dict(
        status=STATUS,
        verification_scope="METADATA_AND_SAVED_COLD_PROOF_ONLY",
        admission=ADMISSION,
        contract=contract,
        result=result,
    )


def _cold_evidence(contract, result, output, expected_sha):
    import torch

    from sr_f5.runtime import NUMERICS_V2

    from . import validation as v
    from .execution import strict_ref

    evidence = _read(PROJECT, result["evidence_ref"])
    tensors = torch.load(
        io.BytesIO(strict_ref(result["tensors_ref"]["path"], result["tensors_ref"]["sha256"])),
        map_location="cpu",
        weights_only=True,
    )
    require(evidence["scope"] == "ACTUAL_CUDA_ONLY", "Fixture is not actual smoke evidence")
    ready, records, _, originals = v._originals(contract)
    identity = evidence["identity"]
    require(
        identity["technical_consumer"] == RESULT_SCHEMA
        and identity["dtype"] == "float32"
        and identity["numerics"] == NUMERICS_V2
        and identity["parent_id"] == "b1"
        and identity["step"] == 0
        and identity["reference_hash"] == ready["spec"]["policy_before_sha256"]
        and identity["source_sha256"] == records[0]["source_sha256"],
        "Actual smoke model identity differs",
    )
    require(
        identity["forward_parameter_dtypes"] == ["float32"]
        and all(v == "float32" for v in identity["forward_float_buffer_dtypes"])
        and identity["input_float_dtype"] == identity["probability_dtype"] == "float32"
        and identity["tf32_matmul"] is False
        and identity["tf32_cudnn"] is False
        and identity["gradient_checkpointing"] is False
        and identity["trainable_parameters"] == evidence["parameter_shapes"],
        "Actual numerical recipe differs",
    )
    require(
        identity["backend_identity"]
        == dict(
            deterministic_algorithms=True,
            deterministic_warn_only=False,
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
            cudnn_benchmark=False,
            cublas_workspace_config=":4096:8",
        ),
        "Actual backend differs",
    )
    require(
        bool(identity["loaded_producer_modules"])
        and all(
            Path(p).is_relative_to(Path(contract["producer"]["deployment"]) / "src")
            for p in identity["loaded_producer_modules"].values()
        ),
        "Loaded producer closure differs",
    )
    _verify_cuda(evidence, identity["hardware"])
    for item in SMOKE_PLAN["measurements"]:
        name = item["name"]
        measurement = evidence["measurements"][name]
        if item["storage"] != "eager":
            storage = measurement["storage"]
            expected_device = DEVICE_BYTES if item["storage"] == "candidate" else 0
            require(
                storage.get("max_device_bytes", 0) == expected_device,
                "Reference/candidate storage differs",
            )
            if expected_device == 0:
                require(
                    not storage.get("device_residency_schema"),
                    "Allspill reference cannot use device residency",
                )
            v.verify_storage(
                measurement["storage"],
                Path(output) / (name + "_STORAGE"),
                HOST_BYTES,
                DISK_BYTES,
                disk_backend="arena_v3",
                max_device_bytes=DEVICE_BYTES if item["storage"] == "candidate" else 0,
            )
    require(
        len(evidence["ledger"]) == 8
        and all(
            e["kind"] == "extra_forward_sequences" and e["count"] == 1 for e in evidence["ledger"]
        ),
        "Actual forward ledger differs",
    )
    computed = recompute(evidence, tensors, originals)
    require(computed == evidence["comparisons"], "Stored smoke comparison differs")
    return dict(
        status="VERIFIED_ACTUAL_TWO_ROW_SMOKE",
        admission=ADMISSION,
        comparisons=computed,
        generated_responses=0,
        optimizer_updates=0,
        contract_sha256=expected_sha,
        evidence_ref=result["evidence_ref"],
        tensors_ref=result["tensors_ref"],
    )


def verify_smoke(contract_path, expected_sha, result_ref):
    checked = verify_smoke_metadata(contract_path, expected_sha, result_ref)
    return _cold_evidence(
        checked["contract"], checked["result"], Path(result_ref["path"]).parent, expected_sha
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "verify"])
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output")
    parser.add_argument("--result-sha256")
    args = parser.parse_args()
    if args.action == "run":
        result = run_smoke(args.contract, args.contract_sha256, args.output)
    else:
        result = verify_smoke(
            args.contract,
            args.contract_sha256,
            dict(
                path=str(Path(args.contract).parent / "SMOKE/RESULT.json"),
                sha256=args.result_sha256,
            ),
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
