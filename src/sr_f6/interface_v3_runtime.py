"""Read-only, explicitly registered interface-v3 sampling over the native FP32 path.

This is a separate execution adapter, not a new name for historical CENSUS_S0.
Only the unchanged native SR-F6 sampling body is reused. No training or replay
entrypoint is provided, and every generation must name one frozen two-draw unit.
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from mm_core.training import state_hash, trainable_state
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

PHASE = "INTERFACE_SCREEN_V3"


class ScreenRuntime(FP32Runtime):
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
            or type(self._stage) is not int
            or self._stage not in (1, 2)
            or self.identity.get("screen_stage") != self._stage
            or current != self._parameter_versions
            or any(p.requires_grad or p.grad is not None for p in parameters.values())
            or any(self.identity.get(key) != value for key, value in self._bound_identity.items())
        ):
            raise PermissionError("Frozen screen parent or execution identity changed")
        return self._policy_hash

    def policy_tensor_identity(self):
        return self._readonly()

    def stable_model_identity(self):
        return dict(
            super().stable_model_identity(),
            phase=PHASE,
            screen_stage=self._stage,
            prompt_version=VERSION,
            registration_sha256=self.identity["registration_sha256"],
            sr_f6_protocol_version=PROTOCOL_VERSION,
            policy_tensor_hash=self._readonly(),
        )

    def _active(self):
        self._readonly()
        permit = getattr(self, "_permit", None)
        if permit is None:
            raise PermissionError("An exact registered screen unit must be active")
        return permit

    def prepare_input(self, model_input, run_root, *, protocol="structured"):
        permit = self._active()
        if (
            protocol not in ("structured", "structured_v4")
            or Path(run_root).resolve(strict=True) != self._source_root
            or model_input != permit["item"]["model_input"]
        ):
            raise PermissionError("Input differs from the registered screen unit")
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
            raise ValueError("Invalid screen accounting")
        if kind == "completion_attempts":
            if count != 2 or permit["attempted"]:
                raise PermissionError("Only the original two-draw generation is allowed")
            permit["attempted"] = True
        elif kind == "generated_tokens":
            if not permit["attempted"] or not 1 <= count <= 1536:
                raise PermissionError("Invalid generated-token accounting")
            permit["token_rows"] += 1
            if permit["token_rows"] > 2:
                raise PermissionError("Screen unit exceeded two responses")
        else:
            raise PermissionError("Screen runtime forbids training and extra forwards")
        return super().reserve(kind, count, **metadata)

    def _generate_native_group(self, model_input, run_root, seeds, **kwargs):
        permit = self._active()
        if list(seeds) != permit["seeds"] or model_input != permit["item"]["model_input"]:
            raise PermissionError("Native call differs from frozen screen draws")
        return HistoricalRuntime._generate_native_group(
            self, model_input, run_root, seeds, **kwargs
        )

    def generate_group(self, *args, **kwargs):
        raise PermissionError("Use a registered screen unit, not unrestricted generation")

    def generate_registered(self, qid, *, on_completion):
        if self._permit is not None or qid in self._attempted_qids:
            raise PermissionError("Nested or repeated screen unit")
        item = self._public[qid]
        if item.get("screen_stage") != self._stage:
            raise PermissionError("Question belongs to a different registered screen stage")
        seeds = self._seeds[qid]
        if len(seeds) != 2 or any(type(seed) is not int for seed in seeds):
            raise PermissionError("Exactly two registered integer seeds are required")
        policy_hash = self._readonly()
        self._attempted_qids.add(qid)
        self._permit = dict(item=item, seeds=seeds, attempted=False, token_rows=0)

        def capture(row):
            self._readonly()
            row.update(
                phase=PHASE,
                screen_stage=self._stage,
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
                execution_mode="READONLY_INTERFACE_SCREEN",
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
            on_completion(row)

        try:
            with independent_draw_rng(self.torch, seeds):
                result = self._generate_native_group(
                    item["model_input"],
                    self._source_root,
                    seeds,
                    protocol="structured",
                    on_completion=capture,
                )
            self._readonly()
            if not self._permit["attempted"] or self._permit["token_rows"] != 2:
                raise RuntimeError("Incomplete screen generation accounting")
            return result
        finally:
            self._permit = None

    def _native_forward(self, *args, **kwargs):
        raise PermissionError("Screen runtime does not permit training or replay forwards")


def load_screen(context, parent, task_id):
    """Load once, hash the adapter once, freeze all parameters, create no optimizer."""
    import torch

    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend

    from .interface_screen import output_path
    from .interface_v3_registration import verify_registered

    stage = context.get("stage")
    if type(stage) is not int or stage not in (1, 2):
        raise PermissionError("Registered stage 1 or 2 required")
    if parent not in ("b0", "b1", "b2"):
        raise ValueError("Unknown parent")
    if not isinstance(task_id, str) or not re.fullmatch(
        rf"LH-SRF6-V3-S{stage}-[0-9]+-{parent}-[01]", task_id
    ):
        raise ValueError("Execution identifier must match stage, parent, job and shard")
    base = Path(context["registration_root"])
    verified = verify_registered(base, stage=stage, parent=parent, require_preflight=True)
    if (
        verified["registration_sha256"] != context["registration_sha256"]
        or Path(context["root"]) != base / f"STAGE{stage}"
        or Path(verified["root"]) != base / f"STAGE{stage}"
        or Path(verified["registration_root"]) != base
        or verified["stage"] != stage
    ):
        raise PermissionError("Screen registration or stage output root changed before load")
    context = verified
    root = Path(context["root"])
    config = context["config"]
    validate_generation_config(config)
    identity_path = output_path(root, f"runtime_identity/{task_id}.json")
    if identity_path.exists() or identity_path.is_symlink():
        raise FileExistsError("Preserve runtime identity; use a fresh execution task id")
    configure_audited_backend()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each screen worker requires exactly one visible CUDA GPU")
    seed_all(seed_for(config["master_seed"], "load", parent))

    def account(kind, count, metadata):
        permit = runtime._active()
        append(
            output_path(root, f"accounting/{task_id}.jsonl"),
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
                    screen_stage=stage,
                    registration_sha256=context["registration_sha256"],
                ),
            ),
        )

    adapter = Path(config["model"]["parents"][parent])
    runtime = ScreenRuntime(
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
        screen_stage=stage,
        execution_mode="READONLY_INTERFACE_SCREEN",
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
    runtime._source_root = Path(context["source_root"])
    runtime._stage = stage
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
    runtime._permit = None
    runtime._attempted_qids = set()
    runtime._parameter_versions = {
        n: (id(p), p._version) for n, p in runtime.model.named_parameters()
    }
    runtime._bound_identity = copy.deepcopy(runtime.identity)
    after = verify_registered(base, stage=stage, parent=parent, require_preflight=True)
    if (
        after["registration_sha256"] != context["registration_sha256"]
        or after["stage"] != stage
        or Path(after["root"]) != root
    ):
        raise PermissionError("Screen registration changed while loading")
    runtime._readonly()
    write_once(
        identity_path,
        dict(
            runtime.identity,
            hardware=actual_cuda_identity(),
            trainable_parameters=[],
            source_binding_status="REGISTERED_V3_STAGE_ONLY",
            input_plan_sha256=digest(context["plan"]),
            optimizer=None,
            scheduler=None,
        ),
    )
    return runtime
