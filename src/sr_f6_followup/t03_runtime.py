"""New-consumer FP32 replay runtime over immutable producer numerical code."""

from __future__ import annotations

import copy
import weakref
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from mm_core.vl_runtime import seed_all
from sr_f3_ce.runtime import configure_parent_training
from sr_f4.common import append
from sr_f5.runtime import NUMERICS_V2
from sr_f5.runtime import Runtime as FP32Runtime
from sr_f6.common import digest, now, read, write_once
from sr_f6.replay import _RUNTIME_ACCESS
from sr_f6.runtime import PROTOCOL_VERSION, generation_recipe, validate_generation_config
from sr_f6.runtime import Runtime as ProducerRuntime
from srf6_ref.core import seed_for

from .t03_access import active_session, require

PHASE = "CONFIRM_T03_V1"
_LOADED = weakref.WeakKeyDictionary()


def verify_loaded(runtime):
    loaded = _LOADED.get(runtime)
    require(loaded is not None and runtime.model is loaded["model"], "Unregistered replay runtime")
    require(runtime.identity == loaded["identity"], "Actual replay execution identity changed")
    require(
        runtime.identity["phase"] == PHASE and runtime.identity["execution_mode"] == "REPLAY",
        "Wrong replay execution mode",
    )
    return loaded


class ReplayRuntime(FP32Runtime):
    def _set_recipe(self):
        super()._set_recipe()
        for name, value in generation_recipe().items():
            setattr(self.generation_config, name, value)
        self.identity.update(
            runtime_recipe="SR-F6-confirm-T03-native-1536",
            experiment_id=PHASE,
            generation_config_expanded=self.generation_config.to_dict(),
        )

    def _replay_session(self):
        active_session(self)
        return None  # Borrow only the unchanged non-session batch construction body.

    def execution_mode(self):
        verify_loaded(self)
        return "REPLAY"

    def prepare_input(self, model_input, run_root, *, protocol="structured_v4"):
        session = active_session(self)
        session.check_input(model_input, run_root, protocol)
        # The kernel's run_root is the consumer output root. Original images
        # remain in the separately pinned producer root; original rows stay intact.
        prepared = super().prepare_input(model_input, self._input_root, protocol="structured")
        prepared["routing"]["sr_f6_protocol_version"] = PROTOCOL_VERSION
        session.bind_prepared(prepared, access=_RUNTIME_ACCESS)
        return prepared

    def _validate_record_revision(self, record):
        require(
            record.get("sr_f6_protocol_version") == PROTOCOL_VERSION, "Original protocol changed"
        )
        return super()._validate_record_revision(dict(record, protocol="structured"))

    def batch_sequence_forward(self, records, *, purpose="training_policy", grad=True):
        session = active_session(self)
        with session.forward(records, purpose=purpose, grad=grad, access=_RUNTIME_ACCESS):
            prepared = super()._prepared_records(records)
            session.check_prepared(prepared)
            session._forward["prepared"] = prepared
            session._forward["native_started"] = False
            return self._batch_forward(prepared, [records[0]["tokens"]], purpose=purpose, grad=grad)

    def batch_reference_forward(self, records, reference):
        session = active_session(self)
        session.check_reference(reference)
        return self.batch_sequence_forward(records, purpose="t03_reference", grad=False)

    def _batch_forward(self, prepared, token_rows, *, purpose, grad):
        session = active_session(self)
        session.check_batch(prepared, token_rows)
        permit = session._forward
        require(
            permit is not None
            and prepared is permit.get("prepared")
            and token_rows == [permit["row"]["tokens"]]
            and purpose == permit["purpose"]
            and grad is permit["grad"]
            and permit.get("native_started") is False,
            "Unregistered original-token forward",
        )
        return ProducerRuntime._batch_forward(
            self, prepared, token_rows, purpose=purpose, grad=grad
        )

    def _native_forward(self, inputs, *, completion_start, lengths, purpose, grad):
        session = active_session(self)
        permit = session._forward
        require(
            permit is not None and permit.get("native_started") is False,
            "Native replay requires a fresh permit",
        )
        prepared = permit["prepared"]
        expected = self.torch.cat(
            (
                prepared["inputs"]["input_ids"],
                prepared["inputs"]["input_ids"].new_tensor([permit["row"]["tokens"]]),
            ),
            -1,
        )
        require(
            completion_start == prepared["inputs"]["input_ids"].shape[-1]
            and lengths == [len(permit["row"]["tokens"])]
            and purpose == permit["purpose"]
            and grad is permit["grad"]
            and self.torch.equal(inputs["input_ids"], expected),
            "Native replay differs from original prompt/tokens",
        )
        permit["native_started"] = True
        with session.native_call(
            inputs,
            completion_start=completion_start,
            lengths=lengths,
            purpose=purpose,
            grad=grad,
            access=_RUNTIME_ACCESS,
        ):
            session.check_native(
                inputs,
                completion_start=completion_start,
                lengths=lengths,
                purpose=purpose,
                grad=grad,
            )
            return super()._native_forward(
                inputs,
                completion_start=completion_start,
                lengths=lengths,
                purpose=purpose,
                grad=grad,
            )

    def reserve(self, kind, count, **metadata):
        active_session(self).account(kind, count, metadata)
        return super().reserve(kind, count, **metadata)

    def generate_group(self, *args, **kwargs):
        raise PermissionError("T03 cannot generate new responses")

    _generate_native_group = generate_group
    generate_registered = generate_group


def load(root, parent, task_id, audit, *, spec):
    import torch

    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend

    from . import source
    from .t03 import build_spec, selected_parent, verified_spec

    context = source.require_support(root, purpose="t03")  # Before any model construction.
    require(build_spec(context) == spec, "Actual support/review/source binding changed")
    records, raw, policy = selected_parent(context, parent)
    registered_spec, spec_ref = verified_spec(root, context)
    require(registered_spec == spec, "Caller spec differs from frozen bytes")
    root, audit = Path(context["root"]), Path(audit)
    require(audit == root / "T03/RUNS" / task_id, "Wrong replay audit root")
    path = audit / "runtime_identity" / f"{task_id}.json"
    require(not path.exists(), "Preserve existing runtime receipt")
    producer = context["producer_context"]
    config = producer["config"]
    validate_generation_config(config)
    configure_audited_backend()
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "One visible CUDA device required",
    )
    rng = capture_rng()
    try:
        seed_all(seed_for(config["master_seed"], "load", parent))

        def account(kind, count, metadata):
            session = active_session(runtime)
            row = session._forward["row"]
            append(
                audit / "accounting" / f"{task_id}.jsonl",
                dict(
                    at=now(),
                    task_id=task_id,
                    parent=parent,
                    kind=kind,
                    count=count,
                    metadata=dict(
                        metadata,
                        qid=row["qid"],
                        row_id=row["row_id"],
                        consumer_registration_sha256=context["consumer"]["registration_sha256"],
                        producer_registration_sha256=producer["registration_sha256"],
                    ),
                ),
            )

        adapter = Path(config["model"]["parents"][parent])
        runtime = ReplayRuntime(
            config["model"]["model_path"],
            adapter_path=str(adapter),
            output_root=audit,
            protocol_revision="v2",
            account=account,
        )
        configure_parent_training(
            runtime, parent_adapter_config=read(adapter / "adapter_config.json")
        )
        source_dtypes = sorted(
            {
                str(p.dtype).removeprefix("torch.")
                for p in runtime.model.parameters()
                if p.is_floating_point()
            }
        )
        runtime.model = runtime.model.float()
        require(
            all(
                not p.is_floating_point() or p.dtype == torch.float32
                for p in runtime.model.parameters()
            ),
            "FP32 parameters required",
        )
        require(
            all(
                not p.is_floating_point() or p.dtype == torch.float32
                for p in runtime.model.buffers()
            ),
            "FP32 buffers required",
        )
        runtime._srf5_numerics = dict(NUMERICS_V2)
        runtime._reference = trainable_state(runtime.model)
        runtime._adapter_names = tuple(runtime._reference)
        require(
            state_hash(runtime._reference) == policy,
            "Actual loaded adapter differs from confirmed parent",
        )
        runtime._input_root = Path(producer["root"])
        runtime.identity.update(
            phase=PHASE,
            execution_mode="REPLAY",
            run_root=str(root),
            audit_root=str(audit),
            input_root=str(runtime._input_root),
            task_id=task_id,
            run_id=task_id,
            parent_id=parent,
            reference_parent=parent,
            step=0,
            policy_hash=policy,
            reference_hash=policy,
            parent_content_hash=producer["registration"]["parent_content_hashes"][parent],
            consumer=copy.deepcopy(context["consumer"]),
            producer=copy.deepcopy(spec["context"]["producer"]),
            source_sha256=context["consumer"]["source_manifest_sha256"],
            data_hash=producer["registration"]["data_sha256"],
            sr_f6_protocol_version=PROTOCOL_VERSION,
            dtype="float32",
            probability_dtype="float32",
            numerics=dict(NUMERICS_V2),
            model_cast_from_dtype=source_dtypes,
            forward_parameter_dtypes=["float32"],
            forward_float_buffer_dtypes=["float32"],
            fresh_optimizer=False,
            optimizer_updates=0,
            generated_responses=0,
        )
        after = source.require_support(root, purpose="t03")
        require(build_spec(after) == spec, "Binding changed during model loading")
        _LOADED[runtime] = dict(
            model=runtime.model,
            identity=copy.deepcopy(runtime.identity),
            spec_sha256=digest(spec),
            spec_ref=copy.deepcopy(spec_ref),
            records_sha256=digest(records),
            raw_sha256=digest(raw),
            consumer=copy.deepcopy(context["consumer"]),
            producer=copy.deepcopy(context["registration"]["producer"]),
        )
        write_once(
            path,
            dict(
                runtime.identity,
                hardware=actual_cuda_identity(),
                optimizer=None,
                scheduler=None,
                trainable_parameters=[
                    dict(name=n, shape=list(p.shape), dtype=str(p.dtype))
                    for n, p in runtime.model.named_parameters()
                    if p.requires_grad
                ],
            ),
        )
        return runtime, context
    finally:
        restore_rng(rng)
