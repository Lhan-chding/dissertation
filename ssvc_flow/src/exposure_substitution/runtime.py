"""Identity-bound SER-J2 optimizer, explicit cursor and complete resume state."""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
from pathlib import Path

from ..verified_discovery_transfer.sft_runtime import reopen_original_lora
from .training import explicit_backward


class SERRuntime:
    def __init__(
        self, adapter, *, seed, identity, parameter_names, sampler, encoded, metadata=None
    ):
        import numpy as np
        import torch

        self.adapter, self.seed = adapter, seed
        self.identity = copy.deepcopy(identity)
        self.parameter_names = tuple(sorted(parameter_names))
        actual = {n: p for n, p in adapter.model.named_parameters() if p.requires_grad}
        if not actual or set(actual) != set(self.parameter_names):
            raise ValueError("Precise trainable whitelist required")
        if any(p.dtype != torch.float32 for p in actual.values()):
            raise ValueError("LoRA trainable parameters must remain FP32")
        if sampler.step != 0:
            raise ValueError(
                "Fresh runtime requires zero schedule cursor; use restore for students"
            )
        self.sampler, self.encoded = sampler, dict(encoded)
        self.metadata = copy.deepcopy(metadata or {})
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.optimizer = torch.optim.AdamW(
            [actual[n] for n in self.parameter_names],
            lr=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        self.step, self.exposures = 0, {}
        self.role_exposures = {"common": 0, "donor": 0, "replay": 0}
        self.poisoned = False

    @classmethod
    def from_frozen_backend(cls, backend, **kwargs):
        if not backend.receipt.get("checkpoint_full_cpu_state_verified"):
            raise ValueError("Verified original parent restoration required")
        names = reopen_original_lora(backend.adapter)
        runtime = cls(backend.adapter, parameter_names=names, **kwargs)
        runtime.parent_receipt = copy.deepcopy(backend.receipt)
        return runtime

    def update(self):
        import torch

        if self.poisoned:
            raise RuntimeError("A partial update must resume from the last committed checkpoint")
        if self.step >= 256 or self.sampler.step != self.step:
            raise ValueError("Update budget/cursor mismatch")
        slots = self.sampler.peek()
        expected_roles = ["common"] * 11 + ["donor"] + ["replay"] * 4
        if (
            len(slots) != 16
            or [r["slot"] for r in slots] != list(range(16))
            or [r["role"] for r in slots] != expected_roles
        ):
            raise ValueError("Explicit 11/1/4 slot contract violated")
        if any(r["update"] != self.step + 1 for r in slots):
            raise ValueError("Explicit schedule update differs")
        examples = [self.encoded[r["task_id"]] for r in slots]
        self.poisoned = True
        self.adapter.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        stats = explicit_backward(self.adapter, examples)
        parameters = dict(self.adapter.model.named_parameters())
        if {n for n, p in parameters.items() if p.requires_grad} != set(self.parameter_names):
            raise ValueError("Trainable whitelist changed")
        trainable = [parameters[n] for n in self.parameter_names]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise FloatingPointError("Missing or nonfinite LoRA gradients")
        if any(p.grad is not None for p in parameters.values() if not p.requires_grad):
            raise RuntimeError("Frozen base/vision acquired gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        # Explicit zeros remain tensors. Adam history may still move weights.
        before = [p.detach().clone() for p in trainable]
        lr = 1e-5 * min((self.step + 1) / 8, 1)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in trainable):
            raise FloatingPointError("Nonfinite SFT parameters")
        change = (
            sum(
                float((p.detach() - old).double().square().sum())
                for p, old in zip(trainable, before, strict=True)
            )
            ** 0.5
        )
        self.step += 1
        self.sampler.commit(self.step)
        for row in slots:
            task_id, role = row["task_id"], row["role"]
            self.exposures[task_id] = self.exposures.get(task_id, 0) + 1
            self.role_exposures[role] += 1
        self.adapter._reset_positions()
        self.poisoned = False
        slot_records = []
        for index, row in enumerate(slots):
            slot_records.append(
                {
                    **row,
                    "sequence_nll": stats["sequence_nll"][index],
                    "target_token_count": len(examples[index].target_ids),
                    "prompt_token_count": len(examples[index].prompt_ids),
                    "sequence_coefficient": 1 / 16,
                    "microbatch_id": next(
                        i for i, group in enumerate(stats["microbatch_slots"]) if index in group
                    ),
                }
            )
        return {
            **stats,
            "update": self.step,
            "lr": lr,
            "gradient_norm": float(norm),
            "zero_finite_gradient": float(norm) == 0,
            "clip_ratio": min(1.0, 1.0 / (float(norm) + 1e-6)),
            "parameter_delta_norm": change,
            "task_ids": [r["task_id"] for r in slots],
            "slots": slot_records,
            "sequence_roles": expected_roles,
            "sequence_target_tokens": [len(x.target_ids) for x in examples],
            "sequence_prompt_tokens": [len(x.prompt_ids) for x in examples],
            "processed_target_sequences": 16,
            "processed_target_tokens": stats["target_tokens"],
            "role_exposures": copy.deepcopy(self.role_exposures),
            "common_loss_contribution": sum(stats["sequence_nll"][:11]) / 16,
            "donor_loss_contribution": stats["sequence_nll"][11] / 16,
            "replay_loss_contribution": sum(stats["sequence_nll"][12:]) / 16,
        }

    def capture(self):
        from ..followup_updates import _forward_state
        from ..optimizer_fork import capture_state

        if self.poisoned or self.sampler.step != self.step:
            raise RuntimeError("Cannot checkpoint a partial or cursor-inconsistent update")
        return {
            **capture_state(self.adapter.model, self.optimizer),
            **_forward_state(self.adapter.model, self.adapter),
            "schema": "ser-j2-student-state-v1",
            "identity": copy.deepcopy(self.identity),
            "parameter_names": list(self.parameter_names),
            "step": self.step,
            "scheduler": {"kind": "linear8_constant", "completed_updates": self.step},
            "sampler": self.sampler.state_dict(),
            "exposures": copy.deepcopy(self.exposures),
            "role_exposures": copy.deepcopy(self.role_exposures),
            "seed": self.seed,
        }

    def restore(self, state):
        from ..followup_updates import _restore_forward, _safe
        from ..optimizer_fork import restore_state

        _safe(state)
        step = state.get("step")
        if (
            state.get("schema") != "ser-j2-student-state-v1"
            or state.get("identity") != self.identity
            or state.get("seed") != self.seed
            or state.get("parameter_names") != list(self.parameter_names)
            or type(step) is not int
            or not 0 <= step <= 256
            or state.get("scheduler") != {"kind": "linear8_constant", "completed_updates": step}
            or state.get("role_exposures")
            != {"common": 11 * step, "donor": step, "replay": 4 * step}
            or sum(state.get("exposures", {}).values()) != 16 * step
        ):
            raise ValueError("Resume identity, whitelist, scheduler or exposures differ")
        detached = copy.deepcopy(self.sampler)
        detached.load_state_dict(state["sampler"])
        if detached.step != step:
            raise ValueError("Resume schedule cursor differs from completed updates")
        parameters = dict(self.adapter.model.named_parameters())
        if set(state["parameters"]) != set(self.parameter_names):
            raise ValueError("Resume parameter keys differ")
        for name in self.parameter_names:
            if (
                state["parameters"][name].shape != parameters[name].shape
                or state["parameters"][name].dtype != parameters[name].dtype
            ):
                raise ValueError("Resume parameter shape/dtype differs")
        # An interrupted student is never continued with empty/fresh Adam.
        optimizer_states = state["optimizer"]["state"]
        if step and (
            len(optimizer_states) != len(self.parameter_names)
            or any(int(v["step"]) != step for v in optimizer_states.values())
        ):
            raise ValueError("Student Adam state absent or step differs")
        if not step and optimizer_states:
            raise ValueError("New branch checkpoint unexpectedly contains Adam history")
        self.poisoned = True
        restore_state(self.adapter.model, self.optimizer, state)
        _restore_forward(self.adapter.model, state, self.adapter)
        self.sampler.load_state_dict(state["sampler"])
        self.step, self.exposures = step, copy.deepcopy(state["exposures"])
        self.role_exposures = copy.deepcopy(state["role_exposures"])
        self.adapter.model.zero_grad(set_to_none=True)
        self.poisoned = False

    def save(self, path):
        import torch

        from ..modeling_v3.io import sha256_file
        from ..optimizer_fork import state_hash

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.capture()
        payload = {"state": state, "state_hash": state_hash(state)}
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            restored = torch.load(temporary, map_location="cpu", weights_only=True)
            if state_hash(restored["state"]) != payload["state_hash"]:
                raise RuntimeError("Checkpoint serialization roundtrip differs")
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "state_hash": payload["state_hash"],
            "step": self.step,
        }

    def resume(self, path):
        self.restore(read_student_checkpoint(path))


def read_student_checkpoint(path):
    import torch

    from ..optimizer_fork import state_hash

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("state_hash") != state_hash(payload["state"])
        or payload["state"].get("schema") != "ser-j2-student-state-v1"
    ):
        raise ValueError("SER-J2 checkpoint checksum/schema mismatch")
    return payload["state"]


def build_runtime(run, backend, parent, block, arm, *, technical=False):
    from ..modeling_v3.io import canonical_hash
    from .schedule import ExplicitScheduleSampler
    from .schema import training_rows, verify_frozen
    from .training import SCHEDULE_SEEDS, encode_rows

    run = Path(run).resolve()
    rows = training_rows(run, arm)
    encoded = encode_rows(backend, rows)
    sampler = ExplicitScheduleSampler.from_run(run, block, arm)
    frozen = verify_frozen(run)
    identity = {
        "experiment_id": "SER_J2_20261007",
        "parent": parent,
        "block": block,
        "arm": arm,
        "frozen_plan_hash": canonical_hash(frozen),
        "schedule_id": sampler.schedule_id,
        "seed": SCHEDULE_SEEDS[block],
        "encoded_training_hash": canonical_hash({k: v.__dict__ for k, v in encoded.items()}),
        "parent_checkpoint_sha256": backend.receipt["checkpoint"]["sha256"],
        "loss": "completion_sequence_mean_slots16_donor_isolated_44314",
        "steps": 256,
        "technical_only": technical,
    }
    return SERRuntime.from_frozen_backend(
        backend,
        seed=SCHEDULE_SEEDS[block],
        identity=identity,
        sampler=sampler,
        encoded=encoded,
        metadata=rows,
    )


def load_parent_backend(run, parent, machine=None, allow_gpu=False):
    """Load with the historical frozen protocol, then allow caller to reopen LoRA."""
    from ..modeling_v3.io import canonical_hash
    from ..protocol_state_probes.inference import FrozenBackend
    from .schema import file_digest, read_bound, verify_frozen

    if allow_gpu is not True:
        raise PermissionError("Real inherited model execution requires --allow-gpu")
    if parent not in ("S96", "REP96"):
        raise ValueError("Unregistered parent")
    run = Path(run).resolve()
    frozen = verify_frozen(run)
    bound_machine = read_bound(run, "machine.json")
    machine = (
        (
            json.loads(Path(machine).read_text())
            if not isinstance(machine, dict)
            else copy.deepcopy(machine)
        )
        if machine is not None
        else bound_machine
    )
    for key in ("runtime_path", "frozen_protocol_path"):
        external = frozen.get("external_inputs", {}).get(key, {})
        if external.get("status") != "VERIFIED_BYTES" or file_digest(machine[key]) != external.get(
            "sha256"
        ):
            raise ValueError("Unverified or modified historical external input: " + key)
    if frozen.get("later_exclusion_status") != "VERIFIED_EXPLICIT_SCAN":
        raise ValueError(
            "Later local root exclusion must be verified before the first GPU execution"
        )
    records = read_bound(run, "parent_availability.json")["checkpoints"]
    selected = [r for r in records if r["checkpoint_id"] == parent]
    if len(selected) != 1 or selected[0].get("status") != "AVAILABLE":
        raise ValueError("Registered original parent is unavailable")
    # This is intentionally the VDT protocol, never the structurally different SER protocol.
    historical = json.loads(Path(machine["frozen_protocol_path"]).read_text())
    if (
        machine.get("frozen_protocol_hash") is not None
        and canonical_hash(historical) != machine["frozen_protocol_hash"]
    ):
        raise ValueError("Historical frozen protocol binding differs")
    return FrozenBackend(machine["runtime_path"], selected[0], historical, allow_gpu=True)


def load_student_into_backend(backend, checkpoint_path, expected_identity=None):
    import torch

    from ..followup_updates import _restore_forward
    from ..modeling_v3.io import canonical_hash, sha256_file
    from ..protocol_state_probes.inference import _model_guard

    state = read_student_checkpoint(checkpoint_path)
    identity = state["identity"]
    if (expected_identity is not None and identity != expected_identity) or identity[
        "parent_checkpoint_sha256"
    ] != backend.receipt["checkpoint"]["sha256"]:
        raise ValueError("Student evaluation parent/identity mismatch")
    names = reopen_original_lora(backend.adapter)
    if names != state["parameter_names"] or set(names) != set(state["parameters"]):
        raise ValueError("Student evaluation whitelist differs")
    params = dict(backend.adapter.model.named_parameters())
    with torch.no_grad():
        for name in names:
            saved, live = state["parameters"][name], params[name]
            if (
                saved.shape != live.shape
                or saved.dtype != live.dtype
                or not torch.isfinite(saved).all()
            ):
                raise ValueError("Student weight shape/dtype/finite contract differs")
        for name in names:
            params[name].copy_(state["parameters"][name].to(params[name].device))
    _restore_forward(backend.adapter.model, state, backend.adapter)
    for parameter in params.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    backend.adapter.model.eval()
    backend.adapter._reset_positions()
    backend._guard_baseline = _model_guard(backend.adapter.model)
    student_sha256 = sha256_file(checkpoint_path)
    backend.receipt = {
        **backend.receipt,
        "student_checkpoint": str(checkpoint_path),
        "student_step": state["step"],
        "student_checkpoint_sha256": student_sha256,
        "student_identity": identity,
        "inference_fingerprint": canonical_hash(
            {"parent": backend.receipt["inference_fingerprint"], "student_sha256": student_sha256}
        ),
    }
    return backend
