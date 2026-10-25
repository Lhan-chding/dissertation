"""SR-F5 protocol adapter over the preserved native Qwen sampling path.

The backend still uses its audited v2 JSON prefill and stopping rule. Only the
public prompt and declared measurement protocol change. No old module is patched.
"""

from __future__ import annotations

import copy
from pathlib import Path

from sr_f3_ce.runtime import configure_parent_training, validate_model_input
from sr_f4.common import append, now, read, write
from sr_f4.runtime import Runtime as NativeRuntime
from srf5_ref.generator import seed_for

PROTOCOL_VERSION = "structured_v3-1.0"


def backend_protocol(protocol):
    if protocol == "structured_v3":
        return "structured"
    if protocol == "natural_work_v2":
        return "natural_work"
    if protocol in ("structured", "natural_work", "answer_only", "chartqa_plain"):
        return protocol
    raise PermissionError("Unregistered SR-F5 protocol")


class Runtime(NativeRuntime):
    def prepare_input(self, model_input, run_root, *, protocol="structured_v3"):
        safe = validate_model_input(model_input)
        backend = backend_protocol(protocol)
        # One current prompt only. It is never reused across distinct image/prompt
        # identities; a changed file's frozen digest remains part of the key.
        key = (str(Path(run_root).resolve()), backend, tuple(sorted(safe.items())))
        if getattr(self, "_srf5_prepared_key", None) != key:
            prepared = super().prepare_input(safe, run_root, protocol=backend)
            prepared["routing"]["sr_f5_protocol_version"] = PROTOCOL_VERSION
            self._srf5_prepared_key, self._srf5_prepared = key, prepared
        return self._srf5_prepared

    def generate_group(
        self,
        model_input,
        run_root,
        seeds,
        *,
        protocol="structured_v3",
        on_completion=None,
        **kwargs,
    ):
        backend = backend_protocol(protocol)

        def capture(row):
            row["backend_protocol"] = backend
            row["protocol"] = protocol
            row["sr_f5_protocol_version"] = PROTOCOL_VERSION
            inputs = self._srf5_prepared["inputs"]
            row["native_prompt_inputs"] = {
                key: value.detach().cpu().tolist()
                for key, value in inputs.items()
                if key != "pixel_values" and hasattr(value, "detach")
            }
            row["position_ids_semantics"] = (
                "native_generate_mrope_from_saved_input_ids_attention_mask_and_image_grid"
            )
            if on_completion is not None:
                on_completion(row)

        return super().generate_group(
            model_input, run_root, seeds, protocol=backend, on_completion=capture, **kwargs
        )

    def _validate_record_revision(self, record):
        compatible = dict(
            record, protocol=backend_protocol(record.get("protocol", "structured_v3"))
        )
        if record.get("sr_f5_protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
            raise PermissionError("SR-F5 protocol version changed")
        super()._validate_record_revision(compatible)

    def _validate_prepared_revision(self, prepared):
        compatible = dict(prepared)
        compatible["routing"] = dict(prepared.get("routing", {}))
        compatible["routing"]["protocol"] = backend_protocol(
            compatible["routing"].get("protocol", "structured_v3")
        )
        super()._validate_prepared_revision(compatible)

    def stable_model_identity(self):
        return dict(super().stable_model_identity(), sr_f5_protocol_version=PROTOCOL_VERSION)


def load(root, parent, task_id, *, training=False):
    import torch

    from mm_core.vl_runtime import seed_all
    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend
    from sr_f3_ce.training import fresh_optimizer

    root = Path(root)
    config = read(root / "CONFIG.json")
    if parent not in config["parents"] or parent not in config["model"]["parents"]:
        raise ValueError("Unknown registered parent")
    adapter = Path(config["model"]["parents"][parent])
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    configure_audited_backend()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each SR-F5 worker must own exactly one GPU")
    seed_all(seed_for(config["master_seed"], "load", parent))

    def account(kind, count, metadata):
        if type(count) is not int or count < 0:
            raise ValueError("Invalid model-call accounting")
        append(
            root / "accounting" / f"{task_id}.jsonl",
            dict(
                at=now(), task_id=task_id, parent=parent, kind=kind, count=count, metadata=metadata
            ),
        )

    runtime = Runtime(
        config["model"]["model_path"],
        adapter_path=str(adapter),
        output_root=root,
        protocol_revision="v2",
        account=account,
    )
    configure_parent_training(runtime, parent_adapter_config=read(adapter / "adapter_config.json"))
    runtime.identity.update(
        parent_id=parent,
        run_id=task_id,
        step=0,
        snapshot=config["model"]["snapshot"],
        sr_f5_protocol_version=PROTOCOL_VERSION,
    )
    write(
        root / "runtime_identity" / f"{task_id}.json",
        dict(
            copy.deepcopy(runtime.identity),
            hardware=actual_cuda_identity(),
            identity_note="registered snapshot; adapter checked once on load; no base rehash",
        ),
    )
    optimizer, scheduler = fresh_optimizer(runtime.model) if training else (None, None)
    return runtime, optimizer, scheduler
