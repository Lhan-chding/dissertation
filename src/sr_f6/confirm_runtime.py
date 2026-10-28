"""Independent G8 confirmation with bounded original-token FP32 alignment."""

from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from mm_core.vl_runtime import seed_all
from sr_f3_ce.runtime import configure_parent_training
from sr_f4.common import append
from sr_f4.runtime import independent_draw_rng
from sr_f5.runtime import NUMERICS_V2
from sr_f5.runtime import Runtime as FP32Runtime
from srf6_ref.core import seed_for

from .common import digest, now, read, write_once
from .interface_v3_protocol import VERSION
from .runtime import PROTOCOL_VERSION, generation_recipe, validate_generation_config
from .runtime import Runtime as HistoricalRuntime

PHASE = "SCHEMA_CONFIRM_V3"
CONFIRM_VERSION = "srf6-confirm-v3"
SPLIT = "DEV_SCHEMA_CONFIRM"
ALIGNMENT_PURPOSE = "confirm_token_alignment"


class ConfirmRuntime(FP32Runtime):
    """The permit is created only by generate_registered, for an exact plan unit."""

    def _set_recipe(self):
        super()._set_recipe()
        for key, value in generation_recipe().items():
            setattr(self.generation_config, key, value)
        self.identity.update(
            runtime_recipe="SR-F6-interface-v3-readonly-native-1536",
            experiment_id=PHASE,
            generation_config_expanded=self.generation_config.to_dict(),
        )

    def _readonly(self):
        parameters = dict(self.model.named_parameters())
        current = {name: (id(p), p._version) for name, p in parameters.items()}
        if (
            self.identity.get("phase") != PHASE
            or self.identity.get("prompt_version") != VERSION
            or self.identity.get("confirmation_version") != CONFIRM_VERSION
            or current != self._parameter_versions
            or {name: (id(value), value._version) for name, value in self.model.named_buffers()}
            != self._buffer_versions
            or any(p.requires_grad or p.grad is not None for p in parameters.values())
            or any(self.identity.get(key) != value for key, value in self._bound_identity.items())
        ):
            raise PermissionError("Frozen confirmation parent or execution identity changed")
        return self._policy_hash

    def policy_tensor_identity(self):
        return self._readonly()

    def stable_model_identity(self):
        return dict(
            super().stable_model_identity(),
            phase=PHASE,
            confirmation_version=CONFIRM_VERSION,
            prompt_version=VERSION,
            registration_sha256=self.identity["registration_sha256"],
            sr_f6_protocol_version=PROTOCOL_VERSION,
            policy_tensor_hash=self._readonly(),
        )

    def _active(self):
        self._readonly()
        permit = getattr(self, "_permit", None)
        if permit is None:
            raise PermissionError("An exact registered confirmation unit must be active")
        return permit

    def prepare_input(self, model_input, run_root, *, protocol="structured"):
        permit = self._active()
        if (
            protocol not in ("structured", "structured_v4")
            or Path(run_root).resolve(strict=True) != self._input_root
            or model_input != permit["item"]["model_input"]
        ):
            raise PermissionError("Input differs from the registered confirmation unit")
        prepared = super().prepare_input(model_input, run_root, protocol="structured")
        prepared["routing"]["sr_f6_protocol_version"] = PROTOCOL_VERSION
        expected = self._preflight_inputs[permit["item"]["qid"]]
        if any(
            prepared["routing"].get(key) != expected[key]
            for key in (
                "input_tensor_hash",
                "processor_hash",
                "chat_template_hash",
                "chat_text_sha256",
            )
        ):
            raise PermissionError("Actual native input differs from server processor preflight")
        return prepared

    def reserve(self, kind, count, **metadata):
        permit = self._active()
        if type(count) is not int or count < 0:
            raise ValueError("Invalid confirmation accounting")
        if permit["kind"] == "alignment":
            if (
                kind != "extra_forward_sequences"
                or count != 1
                or metadata
                != dict(purpose=ALIGNMENT_PURPOSE, completion_tokens=len(permit["row"]["tokens"]))
                or not permit["native_started"]
                or permit["accounted"]
            ):
                raise PermissionError("Only one registered alignment accounting event allowed")
            permit["accounted"] = True
            return super().reserve(kind, count, **metadata)
        if kind == "completion_attempts":
            if count != 8 or permit["attempted"]:
                raise PermissionError("Only the original G8 generation is allowed")
            if metadata != dict(group_seed=permit["seeds"][0], protocol="structured"):
                raise PermissionError("Generation accounting differs from registered seeds")
            permit["attempted"] = True
        elif kind == "generated_tokens":
            if not permit["attempted"] or not 1 <= count <= 1536:
                raise PermissionError("Invalid generated-token accounting")
            if metadata != dict(group_seed=permit["seeds"][0], row_index=permit["token_rows"]):
                raise PermissionError("Token accounting differs from original draw order")
            permit["token_rows"] += 1
            if permit["token_rows"] > 8:
                raise PermissionError("Confirmation unit exceeded eight responses")
        else:
            raise PermissionError("Confirmation runtime forbids training and extra forwards")
        return super().reserve(kind, count, **metadata)

    def _generate_native_group(self, model_input, run_root, seeds, **kwargs):
        permit = self._active()
        if (
            permit["kind"] != "generation"
            or list(seeds) != permit["seeds"]
            or model_input != permit["item"]["model_input"]
        ):
            raise PermissionError("Native call differs from frozen confirmation draws")
        return HistoricalRuntime._generate_native_group(
            self, model_input, run_root, seeds, **kwargs
        )

    def generate_group(self, *args, **kwargs):
        raise PermissionError("Use a registered confirmation unit, not unrestricted generation")

    def generate_registered(self, qid, *, on_completion):
        if self._permit is not None or qid in self._attempted_qids:
            raise PermissionError("Nested or repeated confirmation unit")
        item = self._public[qid]
        if item.get("split") != SPLIT:
            raise PermissionError("Question belongs to a different registered confirmation split")
        if qid not in self._assigned_qids:
            raise PermissionError("Question belongs to another registered worker")
        if qid != self._first_qid and not self._alignment_done:
            raise PermissionError(
                "First registered G8 alignment must finish before further sampling"
            )
        seeds = self._seeds[qid]
        if len(seeds) != 8 or any(type(seed) is not int for seed in seeds):
            raise PermissionError("Exactly eight registered integer seeds are required")
        policy_hash = self._readonly()
        self._attempted_qids.add(qid)
        self._permit = dict(
            kind="generation", item=item, seeds=seeds, attempted=False, token_rows=0
        )

        def capture(row):
            self._readonly()
            row.update(
                phase=PHASE,
                confirmation_version=CONFIRM_VERSION,
                protocol="structured_v4",
                backend_protocol="structured",
                prompt_version=VERSION,
                sr_f6_protocol_version=PROTOCOL_VERSION,
                seed_semantics="independent_per_draw_torch_generator",
                policy_hash=policy_hash,
                policy_tensor_hash=policy_hash,
                parent_content_hash=self.identity["parent_content_hash"],
                source_sha256=self.identity["source_sha256"],
                source_hash=self.identity["source_sha256"],
                data_hash=self.identity["data_hash"],
                registration_sha256=self.identity["registration_sha256"],
                run_id=self.identity["run_id"],
                task_id=self.identity["task_id"],
                policy_step=0,
                step=0,
                execution_mode="READONLY_SCHEMA_CONFIRM",
                optimizer_updates=0,
                prompt_sha256=hashlib.sha256(row["model_input"]["prompt"].encode()).hexdigest(),
            )
            inputs = self._srf5_prepared["inputs"]
            row["native_prompt_inputs"] = {
                k: v.detach().cpu().tolist()
                for k, v in inputs.items()
                if k != "pixel_values" and hasattr(v, "detach")
            }
            row["position_ids_semantics"] = (
                "native_generate_mrope_from_saved_input_ids_attention_mask_and_image_grid"
            )
            if qid == self._first_qid:
                self._first_raw.append(copy.deepcopy(row))
            on_completion(row)

        try:
            with independent_draw_rng(self.torch, seeds):
                result = self._generate_native_group(
                    item["model_input"],
                    self._input_root,
                    seeds,
                    protocol="structured",
                    on_completion=capture,
                )
            self._readonly()
            if not self._permit["attempted"] or self._permit["token_rows"] != 8:
                raise RuntimeError("Incomplete confirmation generation accounting")
            return result
        finally:
            self._permit = None

    def _replay_session(self):
        permit = self._active()
        if permit["kind"] != "alignment":
            raise PermissionError("Only the bounded alignment may replay tokens")
        return None

    def execution_mode(self):
        self._readonly()
        return "READONLY_SCHEMA_CONFIRM"

    def _batch_forward(self, prepared, token_rows, *, purpose, grad):
        permit = self._active()
        if (
            permit["kind"] != "alignment"
            or prepared is not permit["prepared"]
            or token_rows != [permit["row"]["tokens"]]
            or purpose != ALIGNMENT_PURPOSE
            or grad is not False
            or permit["forward_started"]
        ):
            raise PermissionError("Only one original alignment sequence may be forwarded")
        permit["forward_started"] = True
        return HistoricalRuntime._batch_forward(
            self, prepared, token_rows, purpose=purpose, grad=grad
        )

    def _native_forward(self, inputs, *, completion_start, lengths, purpose, grad):
        permit = self._active()
        if (
            permit["kind"] != "alignment"
            or not permit["forward_started"]
            or permit["native_started"]
            or purpose != ALIGNMENT_PURPOSE
            or grad is not False
            or lengths != [len(permit["row"]["tokens"])]
        ):
            raise PermissionError("Unregistered native alignment forward")
        original = permit["prepared"]["inputs"]
        expected = self.torch.cat(
            (original["input_ids"], original["input_ids"].new_tensor([permit["row"]["tokens"]])), -1
        )
        if completion_start != original["input_ids"].shape[-1] or not self.torch.equal(
            inputs["input_ids"], expected
        ):
            raise PermissionError("Alignment forward differs from original prompt/token IDs")
        permit["native_started"] = True
        return super()._native_forward(
            inputs, completion_start=completion_start, lengths=lengths, purpose=purpose, grad=grad
        )

    def replay_alignment(self, records, *, on_replay=None):
        """Replay only this worker's first generated raw G8, once, with no updates."""
        self._readonly()
        if self._permit is not None or self._alignment_attempted:
            raise PermissionError("Alignment is exclusive and may only be attempted once")
        self._alignment_attempted = True
        if len(records) != 8 or len(self._first_raw) != 8:
            raise PermissionError("Alignment requires the complete original first G8")
        for draw, (record, raw) in enumerate(zip(records, self._first_raw, strict=True)):
            if (
                any(record.get(k) != value for k, value in raw.items())
                or set(record) - set(raw) - {"row_id", "draw", "qid"}
                or record.get("draw", draw) != draw
                or record.get("qid", self._first_qid) != self._first_qid
                or record.get("row_id", f"{self.identity['parent_id']}/{self._first_qid}/{draw}")
                != f"{self.identity['parent_id']}/{self._first_qid}/{draw}"
                or raw["seed"] != self._seeds[self._first_qid][draw]
                or raw["group_row_index"] != draw
            ):
                raise PermissionError("Alignment evidence differs from original first G8")
        rng = capture_rng()
        output = []
        try:
            for draw, raw in enumerate(self._first_raw):
                self._permit = dict(
                    kind="alignment",
                    item=self._public[self._first_qid],
                    row=raw,
                    prepared=None,
                    forward_started=False,
                    native_started=False,
                    accounted=False,
                )
                prepared = self.prepare_input(raw["model_input"], self._input_root)
                self._permit["prepared"] = prepared
                if prepared["routing"]["input_tensor_hash"] != raw["prompt_tensor_hash"]:
                    raise PermissionError("Alignment prompt differs from sampled input")
                values = self._batch_forward(
                    prepared, [raw["tokens"]], purpose=ALIGNMENT_PURPOSE, grad=False
                )
                if len(values) != 1 or values[0].numel() != len(raw["tokens"]):
                    raise RuntimeError("Original-token alignment output length differs")
                value = values[0].detach().float().cpu()
                if value.ndim != 1 or not bool(self.torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite alignment probabilities")
                if not self._permit["accounted"]:
                    raise RuntimeError("Alignment native forward was not accounted")
                self._readonly()
                output.append(value)
                if on_replay is not None:
                    on_replay(draw, value.clone())
            from sr_f12.training import enforce_probability_gate, probability_diagnostics

            differences, truncated = [], []
            for raw, current in zip(self._first_raw, output, strict=True):
                sampler = self.torch.tensor(raw["sampler_logprobs"], dtype=self.torch.float32)
                delta = current - sampler
                differences.append(delta.abs())
                truncated.append(delta > math.log(2.0))
            diagnostics = probability_diagnostics(differences, truncated)
            enforce_probability_gate(diagnostics)
            if diagnostics["sampler_train_max_absolute_difference"] > 5.0:
                raise RuntimeError("Maximum alignment difference exceeds five nat")
            self._readonly()
            self._alignment_done = True
            return output
        finally:
            self._permit = None
            restore_rng(rng)
            self._readonly()


def load_confirm(context, parent, task_id):
    """Load once, hash the adapter once, freeze all parameters, create no optimizer."""
    import torch

    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend

    from .confirm_registration import verify_registered
    from .interface_screen import output_path

    if parent not in ("b0", "b1", "b2"):
        raise ValueError("Unknown parent")
    if not isinstance(task_id, str) or not re.fullmatch(
        rf"LH-SRF6-Confirm-[0-9]+-{parent}-[01]", task_id
    ):
        raise ValueError("Execution identifier must match parent, job and shard")
    shard = int(task_id.rsplit("-", 1)[1])
    base = Path(context["root"])
    verified = verify_registered(base, parent=parent, require_preflight=True)
    if (
        verified["registration_sha256"] != context["registration_sha256"]
        or Path(verified["root"]) != base
        or Path(verified["input_root"]) != base
        or verified["plan"]["phase"] != PHASE
        or verified["plan"]["version"] != CONFIRM_VERSION
        or verified["plan"]["source_split"] != SPLIT
    ):
        raise PermissionError("Confirmation registration or output root changed before load")
    context = verified
    root = Path(context["root"])
    config = context["config"]
    validate_generation_config(config)
    identity_path = output_path(root, f"CONFIRM/runtime_identity/{task_id}.json")
    if identity_path.exists() or identity_path.is_symlink():
        raise FileExistsError("Preserve runtime identity; use a fresh execution task id")
    configure_audited_backend()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each confirmation worker requires exactly one visible CUDA GPU")
    seed_all(seed_for(config["master_seed"], "load", parent))

    def account(kind, count, metadata):
        permit = runtime._active()
        append(
            output_path(root, f"CONFIRM/accounting/{task_id}.jsonl"),
            dict(
                at=now(),
                task_id=task_id,
                parent=parent,
                kind=kind,
                count=count,
                metadata=dict(
                    metadata,
                    qid=permit["item"]["qid"],
                    phase=PHASE,
                    confirmation_version=CONFIRM_VERSION,
                    registration_sha256=context["registration_sha256"],
                ),
            ),
        )

    adapter = Path(config["model"]["parents"][parent])
    runtime = ConfirmRuntime(
        config["model"]["model_path"],
        adapter_path=str(adapter),
        output_root=root,
        protocol_revision="v2",
        account=account,
    )
    configure_parent_training(runtime, parent_adapter_config=read(adapter / "adapter_config.json"))
    source_dtype = runtime.identity["dtype"]
    runtime.model = runtime.model.float()
    parameter_dtypes = sorted(
        {
            str(v.dtype).removeprefix("torch.")
            for v in runtime.model.parameters()
            if v.is_floating_point()
        }
    )
    buffer_dtypes = sorted(
        {
            str(v.dtype).removeprefix("torch.")
            for v in runtime.model.buffers()
            if v.is_floating_point()
        }
    )
    if parameter_dtypes != ["float32"] or any(v != "float32" for v in buffer_dtypes):
        raise RuntimeError("Full FP32 model conversion did not take effect")
    policy_hash = state_hash(trainable_state(runtime.model))
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    runtime.model.eval()
    runtime._srf5_numerics = dict(NUMERICS_V2)
    registration = context["registration"]
    runtime.identity.update(
        dtype="float32",
        numerics=dict(NUMERICS_V2),
        numerics_recipe_version=2,
        model_cast_from_dtype=source_dtype,
        forward_parameter_dtypes=parameter_dtypes,
        forward_float_buffer_dtypes=buffer_dtypes,
        input_float_dtype="float32",
        probability_dtype="float32",
        parent_id=parent,
        run_id=task_id,
        task_id=task_id,
        step=0,
        phase=PHASE,
        confirmation_version=CONFIRM_VERSION,
        execution_mode="READONLY_SCHEMA_CONFIRM",
        optimizer_updates=0,
        snapshot=config["model"]["snapshot"],
        run_root=str(root),
        registration_root=str(base),
        source_root=str(context["source_root"]),
        prompt_version=VERSION,
        sr_f6_protocol_version=PROTOCOL_VERSION,
        registration_sha256=context["registration_sha256"],
        source_sha256=registration["source_manifest_sha256"],
        data_hash=registration["data_sha256"],
        parent_content_hash=registration["parent_content_hashes"][parent],
        policy_hash=policy_hash,
        fresh_optimizer=False,
    )
    runtime._policy_hash = policy_hash
    runtime._input_root = Path(context["input_root"])
    runtime._public = {r["qid"]: copy.deepcopy(r) for r in context["public"]}
    runtime._seeds = {
        qid: [
            a["seed"]
            for a in context["plan"]["attempts"]
            if a["parent"] == parent and a["qid"] == qid
        ]
        for qid in runtime._public
    }
    runtime._preflight_inputs = {
        item["qid"]: copy.deepcopy(item["routing"]) for item in context["preflight"]["inputs"]
    }
    assignments = [
        worker
        for worker in context["plan"]["workers"]
        if worker["parent"] == parent and worker["shard"] == shard
    ]
    if len(assignments) != 1:
        raise PermissionError("Exactly one frozen worker assignment required")
    assignment = assignments[0]
    runtime._assigned_qids = set(assignment["qids"])
    runtime._first_qid = assignment["first_qid"]
    runtime._first_raw = []
    runtime._alignment_attempted = False
    runtime._alignment_done = False
    runtime._permit = None
    runtime._attempted_qids = set()
    runtime._parameter_versions = {
        n: (id(p), p._version) for n, p in runtime.model.named_parameters()
    }
    runtime._buffer_versions = {n: (id(v), v._version) for n, v in runtime.model.named_buffers()}
    runtime._bound_identity = copy.deepcopy(runtime.identity)
    after = verify_registered(base, parent=parent, require_preflight=True)
    if (
        after["registration_sha256"] != context["registration_sha256"]
        or Path(after["root"]) != root
    ):
        raise PermissionError("Confirmation registration changed while loading")
    runtime._readonly()
    write_once(
        identity_path,
        dict(
            runtime.identity,
            hardware=actual_cuda_identity(),
            trainable_parameters=[],
            source_binding_status="REGISTERED_INDEPENDENT_CONFIRMATION_ONLY",
            input_plan_sha256=digest(context["plan"]),
            worker_id=assignment["worker_id"],
            shard=shard,
            first_qid=runtime._first_qid,
            optimizer=None,
            scheduler=None,
        ),
    )
    return runtime
