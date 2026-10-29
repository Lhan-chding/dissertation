"""Fixed original-token storage validation; no generation or optimizer is exposed.

This is a separately identified technical consumer, not a legacy replay permit.
It compares row zero both ways, the original 27 forward vectors, and executes
all 128 retained rows with the original 64 diagnostic rows. It does not claim
paired baseline gradients for all 128 rows or scientific learning value.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from sr_f6.common import write_once
from sr_f6.numerical_replay import ATOL, RTOL, _metric, _Preserve
from sr_f6.objective import sequence_loss
from sr_f6.training import (
    _diagnostic_indices,
    _numerical_audit,
    _validate_records,
    _verify_prepared,
)
from sr_f12.training import enforce_probability_gate

from .offload import saved_activation_offload

SCHEMA = "srf6-memory-storage-validation-v1"
PLAN = dict(
    baseline_rows=[0],
    offload_rows=list(range(128)),
    archived_forward_rows=list(range(27)),
    failing_row=26,
    diagnostic_rows="original_registered_64",
    loss_denominator=128,
    atol=ATOL,
    rtol=RTOL,
    generated_responses=0,
    optimizer_updates=0,
    full_128_paired_gradient_equivalence=False,
)
RESTORE_KEYS = (
    "adapter_unchanged_during_replay",
    "reference_unchanged_during_replay",
    "adapter_restored",
    "rng_restored",
    "gradients_restored",
    "requires_grad_restored",
    "module_modes_restored",
    "frozen_base_buffer_versions_unchanged",
    "parameter_objects_unchanged",
)


def require(value, message):
    if not value:
        raise PermissionError(message)


def _trace(record, current, ref):
    return dict(
        row_id=record["row_id"],
        qid=record["qid"],
        tokens=record["tokens"],
        sampler_logprobs=record["sampler_logprobs"],
        train_logprobs=current.detach().cpu().tolist(),
        reference_logprobs=ref.detach().cpu().tolist(),
        token_path_sha256=record["token_path_sha256"],
    )


def _compare_trace(left, right):
    import torch

    for key in ("row_id", "qid", "tokens", "sampler_logprobs", "token_path_sha256"):
        require(left[key] == right[key], "Original trajectory differs: " + key)
    result = {}
    for key in ("train_logprobs", "reference_logprobs"):
        a, b = (torch.tensor(row[key], dtype=torch.float32) for row in (left, right))
        require(a.shape == b.shape == (len(left["tokens"]),), "Original token vector shape differs")
        result[key] = _metric(a, b, a.shape)
    return result


def recompute(evidence, tensors, originals):
    """Recompute comparisons from complete vectors and tensors, never status alone."""
    import torch

    require(evidence["plan"] == PLAN, "Validation plan changed")
    traces = evidence["traces"]
    require(len(traces) == 128 and len(originals) == 27, "Incomplete original coverage")
    require(len({row["row_id"] for row in traces}) == 128, "Duplicated original rows")
    comparisons = [_compare_trace(a, b) for a, b in zip(traces[:27], originals, strict=True)]
    comparisons.append(_compare_trace(evidence["baseline_trace"], traces[0]))
    baseline, offload = tensors["baseline"], tensors["offload_row0"]
    require(
        bool(baseline) and baseline.keys() == offload.keys() == tensors["accumulated"].keys(),
        "Adapter gradient inventory differs",
    )
    gradients = {
        key: _metric(baseline[key], offload[key], evidence["parameter_shapes"][key])
        for key in baseline
    }
    metrics = list(gradients.values()) + [v for row in comparisons for v in row.values()]
    require(
        all(v["finite"] and v["violations"] == 0 for v in metrics),
        "Fixed numerical tolerance failed",
    )
    for name in ("loss", "native", "aux", "kl"):
        a, b = evidence["baseline_loss"][name], evidence["offload_row0_loss"][name]
        require(abs(a - b) <= ATOL + RTOL * abs(b), "Fixed row-zero loss tolerance failed")
    require(any(v is not None for v in tensors["accumulated"].values()), "No accumulated gradient")
    for values in (tensors["accumulated"], *tensors["channels"].values()):
        require(values.keys() == baseline.keys(), "Diagnostic gradient inventory differs")
        require(
            all(
                v is None or (v.dtype == torch.float32 and bool(torch.isfinite(v).all()))
                for v in values.values()
            ),
            "Nonfinite or non-FP32 adapter gradient",
        )
        require(
            all(
                v is None or list(v.shape) == evidence["parameter_shapes"][name]
                for name, v in values.items()
            ),
            "Actual adapter gradient shape differs",
        )
    require(set(tensors["channels"]) == {"native", "aux", "kl"}, "Missing diagnostic channels")
    require(
        evidence["counts"]
        == dict(
            reference_forwards=129,
            policy_forwards=129,
            backward_calls=129,
            diagnostic_gradient_calls=192,
            generated_responses=0,
            optimizer_updates=0,
        ),
        "Actual call counts differ",
    )
    require(
        len(evidence["diagnostic_indices"]) == len(set(evidence["diagnostic_indices"])) == 64,
        "Incomplete diagnostic rows",
    )
    require(
        all(evidence["restoration"].get(k) is True for k in RESTORE_KEYS)
        and not evidence["restoration"]["restoration_errors"]
        and evidence["restoration"]["runtime_requires_reload"] is False,
        "Runtime restoration failed",
    )
    audit = _numerical_audit(traces)
    require(audit["status"] == "FINITE", "Nonfinite probability audit")
    enforce_probability_gate(audit)
    require(audit["sampler_train_max_absolute_difference"] <= 5, "Original probability gate failed")
    return dict(
        gradient_comparisons=gradients, forward_comparisons=comparisons, probability_audit=audit
    )


def _measure(runtime, records, credits, reference, originals, *, max_host_bytes, fixture=False):
    import torch

    root = Path(runtime.identity["run_root"])
    coefficients = _validate_records(
        runtime, records, [8] * 16, reference, credits, root, trainable_state(runtime.model)
    )
    indices = _diagnostic_indices(runtime, records, runtime.identity["diagnostic_row_indices"])
    require(len(originals) == 27, "Exactly 27 preserved original forwards required")
    params = {
        name: value for name, value in runtime.model.named_parameters() if value.requires_grad
    }
    preserved = _Preserve(runtime, reference)
    counts = dict(
        reference_forwards=0,
        policy_forwards=0,
        backward_calls=0,
        diagnostic_gradient_calls=0,
        generated_responses=0,
        optimizer_updates=0,
    )
    evidence = dict(
        plan=copy.deepcopy(PLAN),
        traces=[],
        counts=counts,
        diagnostic_indices=list(indices),
        parameter_shapes={name: list(p.shape) for name, p in params.items()},
    )
    tensors = {
        "channels": {
            kind: {name: torch.zeros_like(p, device="cpu") for name, p in params.items()}
            for kind in ("native", "aux", "kl")
        }
    }

    def gradients():
        return {
            name: None if p.grad is None else p.grad.detach().cpu().clone()
            for name, p in params.items()
        }

    def forward(index):
        row = records[index]
        prepared = runtime.prepare_input(row["model_input"], root, protocol=row["protocol"])
        _verify_prepared(row, prepared)
        item = dict(row, prepared=prepared)
        with torch.no_grad():
            ref = runtime.batch_reference_forward([item], reference)[0]
        counts["reference_forwards"] += 1
        current = runtime.batch_sequence_forward([item], purpose="training_policy", grad=True)[0]
        counts["policy_forwards"] += 1
        sampler = torch.tensor(row["sampler_logprobs"], dtype=torch.float32, device=current.device)
        require(
            current.shape == ref.shape == sampler.shape == (len(row["tokens"]),),
            "Full token path required",
        )
        result = sequence_loss(
            current, sampler, ref, coefficients[index]["native"], coefficients[index]["aux"]
        )
        return result, _trace(row, current, ref)

    try:
        runtime.model.eval()
        if not fixture:
            torch.cuda.reset_peak_memory_stats()
        runtime.model.zero_grad(set_to_none=True)
        baseline_rng = capture_rng()
        result, evidence["baseline_trace"] = forward(0)
        evidence["baseline_loss"] = {
            k: float(v.detach()) for k, v in result.items() if k in ("loss", "native", "aux", "kl")
        }
        (result["loss"] / 128).backward()
        counts["backward_calls"] += 1
        tensors["baseline"] = gradients()
        del result
        if not fixture:
            evidence["baseline_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        runtime.model.zero_grad(set_to_none=True)
        restore_rng(baseline_rng)
        with saved_activation_offload(
            runtime.model, max_host_bytes=max_host_bytes, fixture=fixture
        ) as offload:
            for index in range(128):
                result, trace = forward(index)
                evidence["traces"].append(trace)
                if index in indices:
                    rng = capture_rng()
                    try:
                        for name, buffers in tensors["channels"].items():
                            values = torch.autograd.grad(
                                result[name] / 128,
                                tuple(params.values()),
                                retain_graph=True,
                                allow_unused=True,
                            )
                            counts["diagnostic_gradient_calls"] += 1
                            for key, gradient in zip(params, values, strict=True):
                                if gradient is not None:
                                    require(
                                        bool(torch.isfinite(gradient).all()),
                                        "Nonfinite diagnostic gradient",
                                    )
                                    buffers[key].add_(gradient.detach().cpu())
                    finally:
                        restore_rng(rng)
                (result["loss"] / 128).backward()
                counts["backward_calls"] += 1
                if index == 0:
                    tensors["offload_row0"] = gradients()
                    evidence["offload_row0_loss"] = {
                        k: float(v.detach())
                        for k, v in result.items()
                        if k in ("loss", "native", "aux", "kl")
                    }
                del result
        evidence["offload"] = offload.summary()
        if not fixture:
            evidence["offload_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        tensors["accumulated"] = gradients()
    except BaseException as exc:
        exc.validation_partial = (evidence, tensors)
        raise
    finally:
        evidence["restoration"] = preserved.restore()
    try:
        evidence["comparisons"] = recompute(evidence, tensors, originals)
    except BaseException as exc:
        exc.validation_partial = (evidence, tensors)
        raise
    evidence["scope"] = "SYNTHETIC_CPU_ONLY" if fixture else "ACTUAL_CUDA_STORAGE_VALIDATION"
    return evidence, tensors


def measure_fixture(runtime, records, credits, originals, *, max_host_bytes=10_000_000):
    require(all(p.device.type == "cpu" for p in runtime.model.parameters()), "CPU fixture only")
    return _measure(
        runtime,
        records,
        credits,
        runtime.reference,
        originals,
        max_host_bytes=max_host_bytes,
        fixture=True,
    )


def _originals(contract):
    from .execution import bound_ref, strict_ref

    root = Path(contract["experiment_root"])
    ready = bound_ref(root, contract["original_ready_ref"])
    run = (root / contract["original_ready_ref"]["path"]).parent.parent.parent
    rollout = bound_ref(run, ready["references"]["rollout"])
    credit = bound_ref(run, ready["references"]["credit"])
    audit_ref = contract["original_audit_ref"]
    path = Path(audit_ref["path"])
    if not path.is_absolute():
        path = root / path
    events = [
        json.loads(line)
        for line in strict_ref(path, audit_ref["sha256"]).splitlines()
        if line.strip()
    ]
    selected = [e for e in events if e.get("status") == "UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP"]
    require(len(selected) == 1, "Original failure event ambiguous")
    event = selected[0]
    require(
        event["optimizer_started"] is False and event["parameter_state_changed"] is False,
        "Original update may have started",
    )
    records = rollout["records"]
    require(
        len(records) == 128 and len(event["original_trajectories"]) == 128,
        "Original READY coverage incomplete",
    )
    for row, original in zip(records, event["original_trajectories"], strict=True):
        require(
            all(row.get(k) == v for k, v in original.items()), "Original failure trajectory differs"
        )
    return ready, records, credit["groups"], event["probability_vectors"]


def _technical_runtime(contract, ready, records, account):
    import torch

    from mm_core.vl_runtime import seed_all
    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend, reference_parameters
    from sr_f3_ce.runtime import configure_parent_training
    from sr_f5.runtime import NUMERICS_V2
    from sr_f5.runtime import Runtime as FP32Runtime
    from sr_f6.runtime import (
        PROTOCOL_VERSION,
        backend_protocol,
        validate_generation_config,
        verify_binding_inputs,
    )
    from srf6_ref.core import seed_for

    class TechnicalRuntime(FP32Runtime):
        def generate_group(self, *args, **kwargs):
            raise PermissionError("Technical validator cannot sample")

        def _generate_native_group(self, *args, **kwargs):
            raise PermissionError("Technical validator cannot sample")

        def prepare_input(self, model_input, root, *, protocol):
            prepared = super().prepare_input(model_input, root, protocol=backend_protocol(protocol))
            prepared["routing"]["sr_f6_protocol_version"] = PROTOCOL_VERSION
            return prepared

        def batch_reference_forward(self, rows, reference):
            with reference_parameters(self.model, reference):
                return self.batch_sequence_forward(rows, purpose="training_reference", grad=False)

        def batch_sequence_forward(self, rows, *, purpose, grad):
            require(
                len(rows) == 1 and purpose in ("training_reference", "training_policy"),
                "Fixed microbatch required",
            )
            require(grad is (purpose == "training_policy"), "Unexpected forward mode")
            row = rows[0]
            require(
                row["row_id"] in self.allowed
                and self.allowed[row["row_id"]]
                == {k: v for k, v in row.items() if k != "prepared"},
                "Unregistered retained trajectory",
            )
            prepared = row["prepared"]
            _verify_prepared(row, prepared)
            inputs = dict(prepared["inputs"])
            length = inputs["input_ids"].shape[-1]
            completion = inputs["input_ids"].new_tensor([row["tokens"]])
            require(1 <= completion.shape[1] <= 1536, "Original completion boundary differs")
            inputs["input_ids"] = torch.cat([inputs["input_ids"], completion], -1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(completion)], -1
            )
            inputs["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(completion)], -1
            )
            return self._native_forward(
                inputs,
                completion_start=length,
                lengths=[completion.shape[1]],
                purpose=purpose,
                grad=grad,
            )

    producer_src = Path(contract["producer"]["deployment"]) / "src"
    loaded_sources = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("sr_f", "srf6_ref", "mm_core", "mm_dev")) and not name.startswith(
            "sr_f6_memory_repair"
        ):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                require(
                    path.is_relative_to(producer_src),
                    "Technical computation loaded outside original producer: " + name,
                )
                loaded_sources[name] = str(path)
    root = Path(contract["experiment_root"])
    config = json.loads((root / "CONFIG.json").read_bytes())
    validate_generation_config(config)
    bound, before = verify_binding_inputs(root, config, "b1")
    backend_identity = configure_audited_backend()
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "One actual CUDA device required",
    )
    seed_all(seed_for(config["master_seed"], "load", "b1"))
    adapter = Path(config["model"]["parents"]["b1"])
    runtime = TechnicalRuntime(
        config["model"]["model_path"],
        adapter_path=str(adapter),
        output_root=root,
        protocol_revision="v2",
        account=account,
    )
    configure_parent_training(
        runtime, parent_adapter_config=json.loads((adapter / "adapter_config.json").read_bytes())
    )
    source_dtype = runtime.identity["dtype"]
    runtime.model.float()
    require(
        all(
            p.dtype == torch.float32 and p.device.type == "cuda" for p in runtime.model.parameters()
        ),
        "Actual model must be full CUDA FP32",
    )
    runtime._srf5_numerics = dict(NUMERICS_V2)
    runtime.allowed = {r["row_id"]: r for r in records}
    runtime.reference = trainable_state(runtime.model)
    require(
        state_hash(runtime.reference) == ready["spec"]["policy_before_sha256"],
        "Original step-zero parent tensors differ",
    )
    runtime.identity.update(
        run_root=str(root),
        source_sha256=records[0]["source_sha256"],
        reference_hash=state_hash(runtime.reference),
        reference_parent="b1",
        parent_id="b1",
        run_id=records[0]["run_id"],
        step=0,
        diagnostic_row_indices=ready["spec"]["identity"]["diagnostic_row_indices"],
        technical_consumer=SCHEMA,
        loaded_producer_modules=loaded_sources,
        numerics=dict(NUMERICS_V2),
        model_cast_from_dtype=source_dtype,
        forward_parameter_dtypes=sorted(
            {str(p.dtype).removeprefix("torch.") for p in runtime.model.parameters()}
        ),
        forward_float_buffer_dtypes=sorted(
            {
                str(p.dtype).removeprefix("torch.")
                for p in runtime.model.buffers()
                if p.is_floating_point()
            }
        ),
        input_float_dtype="float32",
        probability_dtype="float32",
        backend_identity=backend_identity,
        gradient_checkpointing=bool(getattr(runtime, "_sr_f12_checkpointing", False)),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        trainable_parameters={
            name: list(p.shape) for name, p in runtime.model.named_parameters() if p.requires_grad
        },
        dtype="float32",
        hardware=actual_cuda_identity(),
        parent_content_hash=bound["parents"]["b1"]["content_sha256"],
        bound_files_before=before,
    )
    _, runtime.identity["bound_files_after"] = verify_binding_inputs(root, config, "b1")
    return runtime


def _ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_validation(contract_path, expected_contract_sha256, output):
    import torch

    from .execution import verify_contract

    contract = verify_contract(contract_path, expected_contract_sha256)
    output = Path(output)
    canonical = Path("/projects/_ssd/varunssd/louis-ssvc")
    require(
        output.is_absolute() and output.parent.resolve().is_relative_to(canonical),
        "Technical output must be canonical",
    )
    require(not output.exists(), "Preserve existing validation attempt")
    output.mkdir()
    write_once(
        output / "ATTEMPT.json",
        dict(schema=SCHEMA, contract_sha256=expected_contract_sha256, plan=PLAN),
    )
    ledger = []

    def account(kind, count, metadata):
        require(
            kind in ("extra_forward_sequences", "diagnostic_gradient_calls"),
            "Unregistered model call: " + kind,
        )
        ledger.append(dict(kind=kind, count=count, metadata=metadata))

    try:
        ready, records, credits, originals = _originals(contract)
        runtime = _technical_runtime(contract, ready, records, account)
        evidence, tensors = _measure(
            runtime,
            records,
            credits,
            runtime.reference,
            originals,
            max_host_bytes=contract["max_host_bytes"],
        )
        evidence.update(
            runtime_identity=runtime.identity,
            ledger=ledger,
            contract_sha256=expected_contract_sha256,
            original_ready_ref=contract["original_ready_ref"],
            producer_revision=contract["producer"]["revision"],
            repair_revision=contract["repair_source"]["revision"],
        )
        verify_contract(contract_path, expected_contract_sha256)
        torch.save(tensors, output / "TENSORS.pt")
        write_once(output / "EVIDENCE.json", evidence)
        result = dict(
            schema=SCHEMA,
            status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
            contract_sha256=expected_contract_sha256,
            evidence_ref=_ref(output / "EVIDENCE.json"),
            tensors_ref=_ref(output / "TENSORS.pt"),
            verifier_source_ref=_ref(Path(__file__).resolve()),
            plan=PLAN,
        )
        write_once(output / "RESULT.json", result)
        return verify_validation(
            contract_path, expected_contract_sha256, _ref(output / "RESULT.json")
        )
    except BaseException as exc:
        if hasattr(exc, "validation_partial"):
            from sr_f6.training import _safe

            partial, partial_tensors = exc.validation_partial
            write_once(output / "PARTIAL_EVIDENCE.json", _safe(partial))
            torch.save(partial_tensors, output / "PARTIAL_TENSORS.pt")
        write_once(
            output / "FAILURE.json",
            dict(
                status="FAILED_PRESERVED",
                error_type=type(exc).__name__,
                error=str(exc),
                ledger=ledger,
            ),
        )
        raise


def verify_validation(contract_path, expected_contract_sha256, validation_ref):
    import torch

    from .execution import strict_ref, verify_contract

    contract = verify_contract(contract_path, expected_contract_sha256)
    result_path = Path(validation_ref["path"])
    result = json.loads(strict_ref(result_path, validation_ref["sha256"]))
    require(not (result_path.parent / "FAILURE.json").exists(), "Failed validation preserved")
    require(
        result["schema"] == SCHEMA
        and result["status"] == "VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH"
        and result["plan"] == PLAN
        and result["contract_sha256"] == expected_contract_sha256,
        "Validation identity differs",
    )
    require(
        result["verifier_source_ref"] == _ref(Path(__file__).resolve()),
        "Actual verifier source differs",
    )
    require(
        result_path.name == "RESULT.json"
        and Path(result["evidence_ref"]["path"]) == result_path.parent / "EVIDENCE.json"
        and Path(result["tensors_ref"]["path"]) == result_path.parent / "TENSORS.pt",
        "Validation evidence must be one complete attempt",
    )
    evidence = json.loads(
        strict_ref(Path(result["evidence_ref"]["path"]), result["evidence_ref"]["sha256"])
    )
    tensor_path = Path(result["tensors_ref"]["path"])
    tensor_bytes = strict_ref(tensor_path, result["tensors_ref"]["sha256"])
    tensors = torch.load(io.BytesIO(tensor_bytes), map_location="cpu", weights_only=True)
    ready, records, _, originals = _originals(contract)
    require(
        evidence["scope"] == "ACTUAL_CUDA_STORAGE_VALIDATION",
        "Synthetic evidence cannot authorize recovery",
    )
    require(
        evidence["contract_sha256"] == expected_contract_sha256
        and evidence["producer_revision"] == contract["producer"]["revision"]
        and evidence["repair_revision"] == contract["repair_source"]["revision"],
        "Execution source binding differs",
    )
    require(
        evidence["diagnostic_indices"] == ready["spec"]["identity"]["diagnostic_row_indices"],
        "Diagnostic selection changed",
    )
    for trace, row in zip(evidence["traces"], records, strict=True):
        require(
            all(
                trace[key] == row[key]
                for key in ("row_id", "qid", "tokens", "sampler_logprobs", "token_path_sha256")
            ),
            "Actual original row binding differs",
        )
    from sr_f5.runtime import NUMERICS_V2

    identity = evidence["runtime_identity"]
    require(
        identity["technical_consumer"] == SCHEMA
        and identity["dtype"] == "float32"
        and identity["numerics"] == NUMERICS_V2
        and identity["parent_id"] == "b1"
        and identity["step"] == 0
        and identity["reference_hash"] == ready["spec"]["policy_before_sha256"]
        and identity["source_sha256"] == records[0]["source_sha256"],
        "Actual runtime identity differs",
    )
    require(
        identity["forward_parameter_dtypes"] == ["float32"]
        and all(v == "float32" for v in identity["forward_float_buffer_dtypes"])
        and identity["input_float_dtype"] == identity["probability_dtype"] == "float32"
        and identity["tf32_matmul"] is False
        and identity["tf32_cudnn"] is False
        and identity["gradient_checkpointing"] is False
        and identity["trainable_parameters"] == evidence["parameter_shapes"],
        "Actual numerical recipe changed",
    )
    backend = identity["backend_identity"]
    require(
        backend
        == dict(
            deterministic_algorithms=True,
            deterministic_warn_only=False,
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
            cudnn_benchmark=False,
            cublas_workspace_config=":4096:8",
        ),
        "Actual deterministic backend differs",
    )
    hardware = identity["hardware"]
    require(
        hardware["cuda_visible_device_count"] == 1
        and "PRO 6000" in hardware["cuda_name"]
        and bool(hardware["cuda_uuid"]),
        "Actual registered GPU identity missing",
    )
    loaded = identity["loaded_producer_modules"]
    require(
        bool(loaded)
        and all(
            Path(p).is_relative_to(Path(contract["producer"]["deployment"]) / "src")
            for p in loaded.values()
        ),
        "Loaded producer closure differs",
    )
    ledger = evidence["ledger"]
    require(
        len(ledger) == 258
        and all(e["kind"] == "extra_forward_sequences" and e["count"] == 1 for e in ledger),
        "Native call ledger incomplete",
    )
    require(
        evidence["offload"]["scope"] == "CUDA_SAVED_ACTIVATION_STORAGE"
        and evidence["offload"]["closed"]
        and evidence["offload"]["pack_count"] > 0
        and evidence["offload"]["unpack_count"] > 0
        and evidence["offload"]["peak_host_bytes"] <= contract["max_host_bytes"],
        "Offload was not actually measured",
    )
    computed = recompute(evidence, tensors, originals)
    require(
        computed == evidence["comparisons"], "Stored comparison differs from full raw recomputation"
    )
    return dict(
        status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
        contract_sha256=expected_contract_sha256,
        producer_revision=contract["producer"]["revision"],
        repair_revision=contract["repair_source"]["revision"],
        zero_new_samples=True,
        zero_optimizer_updates=True,
        generated_responses=0,
        optimizer_updates=0,
        compared_gradient_rows=[0],
        offload_rows=128,
        original_ready_ref=contract["original_ready_ref"],
        verifier_source_ref=result["verifier_source_ref"],
        validation_ref=validation_ref,
        scope=PLAN,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(run_validation(args.contract, args.contract_sha256, args.output), sort_keys=True)
    )


if __name__ == "__main__":
    main()
