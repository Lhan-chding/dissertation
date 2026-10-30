"""Fixed 44 GiB two-row candidate, using immutable completed v5 references.

Metadata entry points never load tensor payloads. Actual/cold GPU entry points
hash and recheck complete predecessor evidence before relying on its gradients.
Historical observations are never counted as calls in this execution.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import io
import json
import math
import re
import signal
import time
from pathlib import Path

SCHEMA = "srf6-memory-smoke-reuse-contract-v1"
RESULT_SCHEMA = "srf6-memory-smoke-reuse-result-v1"
STATUS = "SMOKE_NUMERICAL_PERFORMANCE_MEASURED"
ADMISSION = "NOT_FULL_BATCH_ADMISSION"
PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
PRIOR_ROOT = PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V5_20261030"
PRIOR_CONTRACT_SHA = "d8031ca63eab7cb17ed6b9272b03dab72b3fd42b9ca057cee853680fcba5deb7"
PRIOR_RESULT_SHA = "be1f794c8ce608ac9c6b1943e14ddae9086c486322308edcbadba81900fc9d37"
HOST_BYTES, DISK_BYTES, DEVICE_BYTES = 8 * 2**30, 96 * 2**30, 44 * 2**30
SOURCE_NAMES = (
    "__init__",
    "validation",
    "execution",
    "offload",
    "arena",
    "residency",
    "smoke",
    "smoke_reuse",
)
INHERITED_KEYS = (
    "experiment_root",
    "producer",
    "original_ready_ref",
    "original_audit_ref",
    "original_failure_ref",
    "step0_commit_ref",
    "pilot_spec_ref",
    "source_binding_ref",
    "config_ref",
)
REFERENCES = {"row0_candidate": "row0_eager", "row26_candidate": "row26_allspill"}
SMOKE_PLAN = dict(
    measurements=[
        dict(name="row0_candidate", row=0, storage="candidate"),
        dict(name="row26_candidate", row=26, storage="candidate"),
    ],
    reused_references=REFERENCES,
    reference_scope="V5_COMPLETE_HASH_BOUND_HISTORICAL_TENSORS_NOT_CURRENT_EXECUTION",
    historical_per_row_rng_state="NOT_RECORDED",
    rng_scope="MATCH_HISTORICAL_GLOBAL_INITIAL_STATE_AND_REQUIRE_CURRENT_CANDIDATES_UNCHANGED_NOT_HISTORICAL_PER_ROW_REPLAY",
    loss_denominator=128,
    atol=1e-6,
    rtol=1e-5,
    candidate_max_seconds=120.0,
    candidate_allocated_headroom_bytes=2 * 2**30,
    timing_scope="CURRENT_CANDIDATE_VS_HISTORICAL_REFERENCE_NOT_CAUSAL_SPEEDUP",
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


def _old():
    from . import smoke

    return smoke


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def _read(root, ref):
    from .execution import bound_ref

    return bound_ref(Path(root), ref)


def _ref(path):
    return _old()._ref(path)


def _write(path, value):
    return _old()._write(path, value)


def _cancel_owned_alarm():
    return _old()._cancel_owned_alarm()


def _safe_partial(value):
    return _old()._safe_partial(value)


def _cuda_snapshot(torch, *, fixture=False):
    return _old()._cuda_snapshot(torch, fixture=fixture)


def validate_contract(path, expected_sha, *, source=True):
    from .execution import strict_ref

    path = Path(path)
    contract = json.loads(strict_ref(path, expected_sha))
    require(
        contract.get("schema") == SCHEMA and contract.get("smoke_plan") == SMOKE_PLAN,
        "Frozen reuse contract/plan differs",
    )
    require(
        (
            contract.get("max_host_bytes"),
            contract.get("max_disk_bytes"),
            contract.get("max_device_bytes"),
            contract.get("storage_backend"),
        )
        == (HOST_BYTES, DISK_BYTES, DEVICE_BYTES, "arena_v3"),
        "Frozen reuse storage bounds differ",
    )
    require(
        contract.get("prior_smoke_contract_ref")
        == dict(path=str(PRIOR_ROOT / "CONTRACT.json"), sha256=PRIOR_CONTRACT_SHA),
        "Wrong v5 contract reference",
    )
    require(
        contract.get("prior_smoke_result_ref")
        == dict(path=str(PRIOR_ROOT / "SMOKE/RESULT.json"), sha256=PRIOR_RESULT_SHA),
        "Wrong v5 result reference",
    )
    checked = _old().verify_smoke_metadata(
        PRIOR_ROOT / "CONTRACT.json", PRIOR_CONTRACT_SHA, contract["prior_smoke_result_ref"]
    )
    prior = checked["contract"]
    for key in INHERITED_KEYS:
        require(contract[key] == prior[key], "Original immutable binding differs: " + key)
    root = Path(contract["experiment_root"])
    # Explicitly bind the saved cold proof path, not only its hash payload.
    result = checked["result"]
    require(
        Path(result["cold_verification_ref"]["path"])
        == PRIOR_ROOT / "SMOKE/COLD_VERIFICATION.json",
        "Wrong v5 cold proof path",
    )
    evidence = _read(root, result["evidence_ref"])
    proof = _read(root, result["cold_verification_ref"])
    require(evidence["comparisons"] == proof["comparisons"], "V5 saved comparison/proof differs")
    if source:
        binding = _read(root, contract["source_binding_ref"])
        require(
            bool(binding["versions"])
            and all(
                v and importlib.metadata.version(k) == v for k, v in binding["versions"].items()
            ),
            "Original numerical library versions differ",
        )
        producer = contract["producer"]
        manifest = _read(root, producer["source_manifest_ref"])
        files = {v["path"]: v for v in manifest["files"]}
        require(len(files) == len(manifest["files"]), "Duplicate producer source")
        selected = [v for p, v in files.items() if p.startswith("src/") and p.endswith(".py")]
        require(bool(selected), "Missing producer closure")
        for ref in selected:
            strict_ref(Path(producer["deployment"]) / ref["path"], ref["sha256"])
        for repair in (prior["repair_source"], contract["repair_source"]):
            require(
                re.fullmatch(r"[0-9a-f]{40}", repair["revision"]), "Missing actual repair revision"
            )
            listed = {v["path"]: v for v in repair["files"]}
            require(len(listed) == len(repair["files"]), "Duplicate repair source")
            for ref in listed.values():
                strict_ref(Path(repair["deployment"]) / ref["path"], ref["sha256"])
        repair = contract["repair_source"]
        listed = {v["path"]: v for v in repair["files"]}
        require(
            Path(__file__).resolve().relative_to(Path(repair["deployment"])).as_posix() in listed,
            "Actual reuse verifier missing",
        )
        require(
            all("src/sr_f6_memory_repair/" + n + ".py" in listed for n in SOURCE_NAMES),
            "Missing reuse dependency closure",
        )
        # The helpers validating historical evidence must be the same frozen v5
        # implementations, not merely similarly named successor helpers.
        old_files = {v["path"]: v["sha256"] for v in prior["repair_source"]["files"]}
        for name in SOURCE_NAMES[:-1]:
            key = "src/sr_f6_memory_repair/" + name + ".py"
            require(
                listed[key]["sha256"] == old_files[key], "Historical verifier dependency changed"
            )
    return contract


def _load_prior(contract, originals):
    """GPU/cold path only: full predecessor numerical verification, then payload."""
    import torch

    from .execution import strict_ref

    result = _read(PROJECT, contract["prior_smoke_result_ref"])
    checked = _old().verify_smoke(
        PRIOR_ROOT / "CONTRACT.json", PRIOR_CONTRACT_SHA, contract["prior_smoke_result_ref"]
    )
    require(
        checked == _read(PROJECT, result["cold_verification_ref"]),
        "V5 cold proof recomputation differs",
    )
    evidence = _read(PROJECT, result["evidence_ref"])
    tensors = torch.load(
        io.BytesIO(strict_ref(result["tensors_ref"]["path"], result["tensors_ref"]["sha256"])),
        map_location="cpu",
        weights_only=True,
    )
    require(
        _old().recompute(evidence, tensors, originals) == evidence["comparisons"],
        "V5 reference payload differs",
    )
    return evidence, tensors


def _file_identity(value):
    return {
        kind: sorted((row["path"], row["sha256"]) for row in section["files"])
        for kind, section in value.items()
    }


def _identity_match(current, previous, *, fixture=False):
    excluded = {
        "technical_consumer",
        "reused_loader_consumer",
        "hardware",
        "bound_files_before",
        "bound_files_after",
    }
    require(
        {k: v for k, v in current.items() if k not in excluded}
        == {k: v for k, v in previous.items() if k not in excluded},
        "Historical model/adapter/environment identity differs",
    )
    if not fixture:
        require(
            current["technical_consumer"] == RESULT_SCHEMA
            and previous["technical_consumer"] == _old().RESULT_SCHEMA,
            "Reference consumer provenance differs",
        )
        for name in ("bound_files_before", "bound_files_after"):
            require(
                _file_identity(current[name]) == _file_identity(previous[name]),
                "Historical model/parent content differs",
            )
        # GPU UUID/host/free memory are historical observations, not identity of
        # today's device; architecture and numerical runtime must still match.
        for key in (
            "cuda_visible_device_count",
            "cuda_name",
            "cuda_total_memory_bytes",
            "compute_capability",
            "cuda_runtime",
            "driver_version",
        ):
            require(
                current["hardware"][key] == previous["hardware"][key],
                "Historical hardware/runtime compatibility differs: " + key,
            )


def _pair(evidence, tensors, originals, row, prior_evidence, prior_tensors):
    name = "row" + str(row) + "_candidate"
    reference = REFERENCES[name]
    require(
        reference in prior_evidence["measurements"] and reference in prior_tensors,
        "Missing complete historical reference",
    )
    projection = dict(
        evidence,
        measurements={
            name: evidence["measurements"][name],
            reference: prior_evidence["measurements"][reference],
        },
    )
    values = {name: tensors[name], reference: prior_tensors[reference]}
    result = _old()._pair(projection, values, originals, row)
    result["reference_observation_scope"] = "HISTORICAL_V5_NOT_EXECUTED_HERE"
    return result


def recompute(evidence, tensors, originals, prior_evidence, prior_tensors):
    from .validation import RESTORE_KEYS

    require(
        _old().recompute(prior_evidence, prior_tensors, originals) == prior_evidence["comparisons"],
        "Historical reference cold comparison differs",
    )
    require(
        evidence["plan"] == SMOKE_PLAN and len(originals) == 27, "Reuse measurement plan differs"
    )
    order = [v["name"] for v in SMOKE_PLAN["measurements"]]
    require(
        evidence["measurement_order"] == order
        and set(evidence["measurements"]) == set(tensors) == set(order),
        "Current execution cannot include historical measurements",
    )
    require(
        evidence["counts"]
        == dict(
            reference_forwards=2,
            policy_forwards=2,
            backward_calls=2,
            diagnostic_gradient_calls=0,
            generated_responses=0,
            optimizer_updates=0,
        ),
        "Reuse actual call counts differ",
    )
    require(
        evidence["reference_reuse"]
        == dict(measurements=REFERENCES, scope="HISTORICAL_V5_NOT_EXECUTED_HERE"),
        "Reference provenance differs",
    )
    require(
        evidence["parameter_shapes"] == prior_evidence["parameter_shapes"],
        "Historical parameter shapes differ",
    )
    require(set(evidence["candidate_rng_observations"]) == set(order), "Missing RNG observations")
    for item in SMOKE_PLAN["measurements"]:
        measured = evidence["measurements"][item["name"]]
        require(
            measured["row"] == item["row"]
            and measured["storage_mode"] == "candidate"
            and measured["storage"] is not None,
            "Candidate identity differs",
        )
        require(
            measured["storage"].get("max_device_bytes") == DEVICE_BYTES,
            "Candidate residency budget differs",
        )
        phases = measured["phase_seconds"]
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
            and abs(sum(phases.values()) - measured["seconds"]) < 1e-6,
            "Actual candidate timing incomplete",
        )
        require(
            measured["rng_after_sha256"] == measured["rng_before_sha256"], "Candidate RNG advanced"
        )
        require(
            evidence["candidate_rng_observations"][item["name"]]
            == dict(
                before_sha256=measured["rng_before_sha256"],
                after_sha256=measured["rng_after_sha256"],
                unchanged=True,
            ),
            "Candidate RNG observation differs",
        )
        require(
            measured["rng_before_sha256"] == prior_evidence["restoration"]["rng_before_sha256"],
            "Historical RNG pre-state differs",
        )
    require(
        evidence["historical_per_row_rng_state"] == "NOT_RECORDED"
        and evidence["global_initial_rng_match"] is True
        and evidence["candidate_rng_unchanged"] is True,
        "RNG reuse scope/state differs",
    )
    restoration = evidence["restoration"]
    require(
        all(restoration.get(k) is True for k in RESTORE_KEYS)
        and restoration["restoration_errors"] == []
        and restoration["runtime_requires_reload"] is False,
        "Reuse runtime restoration failed",
    )
    for key in ("rng_before_sha256", "policy_before_sha256", "reference_sha256"):
        require(
            restoration[key] == prior_evidence["restoration"][key],
            "Historical runtime pre-state differs: " + key,
        )
    require(
        0 <= evidence["total_seconds"] <= SMOKE_PLAN["total_max_seconds"]
        and evidence["row0_checked_before_row26"] is True,
        "Reuse order/time gate failed",
    )
    pairs = {
        str(row): _pair(evidence, tensors, originals, row, prior_evidence, prior_tensors)
        for row in (0, 26)
    }
    return dict(
        pairs=pairs,
        reference_scope="HISTORICAL_V5_NOT_EXECUTED_HERE",
        row26_scope="ALLSPILL_VS_RESIDENCY_NOT_EAGER_GRADIENT",
        admission=ADMISSION,
    )


def measure(
    runtime,
    records,
    credits,
    originals,
    output,
    prior_evidence,
    prior_tensors,
    *,
    fixture=False,
    progress=None,
    started=None,
    clock=time.monotonic,
):
    import torch

    from mm_core.training import capture_rng, state_hash, trainable_state
    from sr_f6.numerical_replay import _Preserve
    from sr_f6.objective import sequence_loss

    from .offload import saved_activation_offload
    from .validation import _trace, _validate_records, _verify_prepared, verify_storage

    require(
        _old().recompute(prior_evidence, prior_tensors, originals) == prior_evidence["comparisons"],
        "Historical reference incomplete or changed",
    )
    _identity_match(runtime.identity, prior_evidence["identity"], fixture=fixture)
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
        historical_per_row_rng_state="NOT_RECORDED",
        global_initial_rng_match=False,
        candidate_rng_unchanged=False,
        candidate_rng_observations={},
        reference_reuse=dict(
            measurements=copy.deepcopy(REFERENCES), scope="HISTORICAL_V5_NOT_EXECUTED_HERE"
        ),
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
        require(
            preserved.rng_hash == prior_evidence["restoration"]["rng_before_sha256"],
            "Historical RNG pre-state differs",
        )
        require(
            evidence["parameter_shapes"] == prior_evidence["parameter_shapes"],
            "Historical parameter shapes differ",
        )
        evidence["global_initial_rng_match"] = True
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
            rng_before = state_hash(capture_rng())
            require(
                rng_before == prior_evidence["restoration"]["rng_before_sha256"],
                "Historical RNG pre-state differs",
            )
            evidence["candidate_rng_observations"][name] = dict(
                before_sha256=rng_before, after_sha256=None, unchanged=False
            )
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
            context = saved_activation_offload(
                runtime.model,
                max_host_bytes=HOST_BYTES,
                max_disk_bytes=DISK_BYTES,
                scratch_directory=scratch,
                disk_backend="arena_v3",
                fixture=fixture,
                max_device_bytes=DEVICE_BYTES,
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
            rng_after = state_hash(capture_rng())
            evidence["candidate_rng_observations"][name].update(
                after_sha256=rng_after, unchanged=rng_before == rng_after
            )
            require(
                rng_after == rng_before,
                "Candidate RNG advanced; historical per-row RNG is unavailable",
            )
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
                rng_before_sha256=rng_before,
                rng_after_sha256=rng_after,
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
                _pair(evidence, tensors, originals, row, prior_evidence, prior_tensors)
                if row == 0:
                    evidence["row0_checked_before_row26"] = True
        evidence["candidate_rng_unchanged"] = True
        evidence["total_seconds"] = clock() - started
    except BaseException as exc:
        _cancel_owned_alarm()
        exc.smoke_partial = (evidence, tensors)
        raise
    finally:
        evidence["restoration"] = preserved.restore()
        evidence["total_seconds"] = clock() - started
    try:
        evidence["comparisons"] = recompute(
            evidence, tensors, originals, prior_evidence, prior_tensors
        )
    except BaseException as exc:
        _cancel_owned_alarm()
        exc.smoke_partial = (evidence, tensors)
        raise
    return evidence, tensors


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
        prior_evidence, prior_tensors = _load_prior(contract, originals)
        stage("historical_v5_full_payload_cold_verified", 0.0)
        runtime = v._technical_runtime(loader_view, ready, records, account, stage_callback=stage)
        runtime.identity["reused_loader_consumer"] = runtime.identity["technical_consumer"]
        runtime.identity["technical_consumer"] = RESULT_SCHEMA
        evidence, tensors = measure(
            runtime,
            records,
            credits,
            originals,
            output,
            prior_evidence,
            prior_tensors,
            progress=progress,
            started=started,
        )
        evidence.update(identity=runtime.identity, ledger=ledger, preflight_stages=stages)
        _old()._verify_cuda(evidence, runtime.identity["hardware"])
        _verify_ledger(ledger)
        _write(output / "EVIDENCE.json", evidence)
        with (output / "TENSORS.pt").open("xb") as stream:
            torch.save(tensors, stream)
        result = dict(
            schema=RESULT_SCHEMA,
            status=STATUS,
            admission=ADMISSION,
            contract_sha256=expected_sha,
            prior_smoke_contract_ref=contract["prior_smoke_contract_ref"],
            prior_smoke_result_ref=contract["prior_smoke_result_ref"],
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


def _verify_ledger(ledger, records=None):
    require(len(ledger) == 4, "Actual four-forward ledger incomplete")
    for index, entry in enumerate(ledger):
        require(
            entry["kind"] == "extra_forward_sequences"
            and entry["count"] == 1
            and entry["metadata"]["purpose"]
            == ("training_reference" if index % 2 == 0 else "training_policy"),
            "Actual reference/policy ledger differs",
        )
        if records is not None:
            row = 0 if index < 2 else 26
            require(
                entry["metadata"]["completion_tokens"] == len(records[row]["tokens"]),
                "Actual ledger row length differs",
            )


def verify_smoke_metadata(contract_path, expected_sha, result_ref):
    """No tensor loads or numerical recomputation on the login node."""
    from .execution import strict_ref

    contract = validate_contract(contract_path, expected_sha, source=False)
    expected = Path(contract_path).parent / "SMOKE/RESULT.json"
    require(Path(result_ref["path"]) == expected, "Wrong reuse result path")
    result = _read(PROJECT, result_ref)
    require(
        result["schema"] == RESULT_SCHEMA
        and result["status"] == STATUS
        and result["admission"] == ADMISSION
        and result["contract_sha256"] == expected_sha,
        "Reuse result identity differs",
    )
    require(
        result["producer_revision"] == contract["producer"]["revision"]
        and result["repair_revision"] == contract["repair_source"]["revision"]
        and result["generated_responses"] == result["optimizer_updates"] == 0,
        "Reuse source/budget differs",
    )
    for key in ("prior_smoke_contract_ref", "prior_smoke_result_ref"):
        require(result[key] == contract[key], "Historical result binding differs")
    for key, filename in (
        ("evidence_ref", "EVIDENCE.json"),
        ("tensors_ref", "TENSORS.pt"),
        ("cold_verification_ref", "COLD_VERIFICATION.json"),
    ):
        require(
            Path(result[key]["path"]) == expected.parent / filename, "Wrong reuse evidence path"
        )
        if key != "tensors_ref":
            strict_ref(result[key]["path"], result[key]["sha256"])
        else:
            require(
                re.fullmatch(r"[0-9a-f]{64}", result[key]["sha256"])
                and Path(result[key]["path"]).is_file()
                and not Path(result[key]["path"]).is_symlink(),
                "Missing bound reuse tensors",
            )
    evidence = _read(PROJECT, result["evidence_ref"])
    proof = _read(PROJECT, result["cold_verification_ref"])
    require(
        proof["status"] == "VERIFIED_ACTUAL_TWO_ROW_REUSED_REFERENCE_SMOKE"
        and proof["admission"] == ADMISSION
        and proof["evidence_ref"] == result["evidence_ref"]
        and proof["tensors_ref"] == result["tensors_ref"]
        and proof["contract_sha256"] == expected_sha
        and proof["prior_smoke_result_ref"] == contract["prior_smoke_result_ref"]
        and proof["comparisons"] == evidence["comparisons"],
        "Missing bound reuse cold proof",
    )
    require(not (expected.parent / "FAILURE.json").exists(), "Failed reuse smoke cannot authorize")
    return dict(
        status=STATUS,
        verification_scope="METADATA_AND_SAVED_COLD_PROOF_ONLY",
        admission=ADMISSION,
        contract=contract,
        result=result,
    )


def _cold_evidence(contract, result, output, expected_sha):
    import torch

    from . import validation as v
    from .execution import strict_ref

    evidence = _read(PROJECT, result["evidence_ref"])
    tensors = torch.load(
        io.BytesIO(strict_ref(result["tensors_ref"]["path"], result["tensors_ref"]["sha256"])),
        map_location="cpu",
        weights_only=True,
    )
    require(
        evidence["scope"] == "ACTUAL_CUDA_ONLY", "Fixture is not actual reused-reference evidence"
    )
    _, records, _, originals = v._originals(contract)
    previous, old_tensors = _load_prior(contract, originals)
    _identity_match(evidence["identity"], previous["identity"])
    _old()._verify_cuda(evidence, evidence["identity"]["hardware"])
    for item in SMOKE_PLAN["measurements"]:
        v.verify_storage(
            evidence["measurements"][item["name"]]["storage"],
            Path(output) / (item["name"] + "_STORAGE"),
            HOST_BYTES,
            DISK_BYTES,
            disk_backend="arena_v3",
            max_device_bytes=DEVICE_BYTES,
        )
    _verify_ledger(evidence["ledger"], records)
    computed = recompute(evidence, tensors, originals, previous, old_tensors)
    require(computed == evidence["comparisons"], "Stored reused-reference comparison differs")
    return dict(
        status="VERIFIED_ACTUAL_TWO_ROW_REUSED_REFERENCE_SMOKE",
        admission=ADMISSION,
        comparisons=computed,
        generated_responses=0,
        optimizer_updates=0,
        contract_sha256=expected_sha,
        prior_smoke_result_ref=contract["prior_smoke_result_ref"],
        evidence_ref=result["evidence_ref"],
        tensors_ref=result["tensors_ref"],
    )


def verify_smoke(contract_path, expected_sha, result_ref):
    checked = verify_smoke_metadata(contract_path, expected_sha, result_ref)
    validate_contract(contract_path, expected_sha, source=True)
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
