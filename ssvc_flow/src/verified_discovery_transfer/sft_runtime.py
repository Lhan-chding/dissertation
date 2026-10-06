"""Fresh SFT AdamW and complete, identity-bound student resume state."""

from __future__ import annotations

import copy
import os
import random
import tempfile
from pathlib import Path

from .sft_loss import accumulated_backward


class ShuffledCycle:
    def __init__(self, size, seed):
        if type(size) is not int or size <= 0:
            raise ValueError("Sampler needs a nonempty registered view")
        self.size = size
        self.rng = random.Random(seed)
        self.order = list(range(size))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.epochs = 0

    def take(self, count):
        result = []
        for _ in range(count):
            if self.cursor == self.size:
                self.rng.shuffle(self.order)
                self.cursor = 0
                self.epochs += 1
            result.append(self.order[self.cursor])
            self.cursor += 1
        return result

    def state_dict(self):
        return {
            "size": self.size,
            "order": self.order[:],
            "cursor": self.cursor,
            "epochs": self.epochs,
            "rng": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        if (
            state["size"] != self.size
            or sorted(state["order"]) != list(range(self.size))
            or not 0 <= state["cursor"] <= self.size
        ):
            raise ValueError("Sampler checkpoint does not match training view")
        self.order = state["order"][:]
        self.cursor, self.epochs = state["cursor"], state["epochs"]
        self.rng.setstate(state["rng"])


def reopen_original_lora(adapter):
    """Exactly one A/B pair per originally audited module; never add/reset adapters."""
    import torch

    audit = adapter.audit
    modules = audit.get("lora_modules", [])
    if (
        len(modules) != 96
        or len(set(modules)) != 96
        or audit.get("lora_rank") != 8
        or audit.get("lora_alpha") != 16
        or audit.get("lora_dropout") != 0
    ):
        raise ValueError("Original 96-module rank8/alpha16/dropout0 audit required")
    actual = dict(adapter.model.named_parameters())
    names = []
    for module in modules:
        if "visual" in module or not module.endswith((".gate_proj", ".up_proj", ".down_proj")):
            raise ValueError("Non-language-MLP LoRA target")
        for side in ("A", "B"):
            suffix = f"{module}.lora_{side}.default.weight"
            matches = [n for n in actual if n == suffix or n.endswith("." + suffix)]
            if len(matches) != 1:
                raise ValueError("Original LoRA parameter missing or ambiguous: " + suffix)
            name = matches[0]
            if actual[name].dtype != torch.float32:
                raise ValueError("Inherited trainable dtype must remain float32")
            if actual[name].shape[0 if side == "A" else 1] != 8:
                raise ValueError("Inherited LoRA rank differs")
            names.append(name)
    if {n for n in actual if "lora_" in n} != set(names):
        raise ValueError("Additional or stacked LoRA adapters are forbidden")
    if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in adapter.model.modules()):
        raise ValueError("All inherited dropout must remain disabled")
    for name, parameter in actual.items():
        parameter.requires_grad_(name in names)
        parameter.grad = None
    adapter.model.train()
    adapter._reset_positions()
    return sorted(names)


class SFTRuntime:
    def __init__(self, adapter, *, seed, view_identity, parameter_names):
        import numpy as np
        import torch

        self.adapter, self.seed = adapter, seed
        self.identity = copy.deepcopy(view_identity)
        self.parameter_names = tuple(sorted(parameter_names))
        actual = {n: p for n, p in adapter.model.named_parameters() if p.requires_grad}
        if not actual or set(actual) != set(self.parameter_names):
            raise ValueError("Precise trainable whitelist required")
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
        self.step = 0
        self.focus_sampler = self.replay_sampler = None
        self.exposures = {}
        self.poisoned = False

    @classmethod
    def from_frozen_backend(cls, backend, *, seed, view_identity):
        if not backend.receipt.get("checkpoint_full_cpu_state_verified"):
            raise ValueError("Verified parent restoration required")
        names = reopen_original_lora(backend.adapter)
        runtime = cls(
            backend.adapter, seed=seed, view_identity=view_identity, parameter_names=names
        )
        runtime.parent_receipt = copy.deepcopy(backend.receipt)
        return runtime

    def bind_data(self, focus, replay):
        if not replay:
            raise ValueError("Registered common replay is required")
        self.focus, self.replay = list(focus), list(replay)
        self.focus_sampler = ShuffledCycle(len(focus), self.seed) if focus else None
        # Independent of focus dataset size and arm: same parent/repeat replay schedule.
        self.replay_sampler = ShuffledCycle(len(replay), self.seed + 1000003)

    def update(self, *, microbatch_size=4):
        import torch

        if self.poisoned:
            raise RuntimeError("Failed update requires resume from a committed checkpoint")
        if self.replay_sampler is None:
            raise ValueError("Bind registered training rows first")
        if self.step >= 256:
            raise ValueError("Registered 256-step budget exhausted")
        self.poisoned = True
        examples = (
            [self.focus[i] for i in self.focus_sampler.take(12)] if self.focus_sampler else []
        )
        examples += [self.replay[i] for i in self.replay_sampler.take(4)]
        self.adapter.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        stats = accumulated_backward(self.adapter, examples, microbatch_size=microbatch_size)
        parameters = dict(self.adapter.model.named_parameters())
        trainable = [parameters[n] for n in self.parameter_names]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise FloatingPointError("Missing or nonfinite LoRA gradients")
        if any(p.grad is not None for p in parameters.values() if not p.requires_grad):
            raise RuntimeError("Frozen base/vision acquired gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        if float(norm) == 0:
            raise FloatingPointError("All LoRA gradients are zero")
        before = [p.detach().clone() for p in trainable]
        lr = 1e-5 * min((self.step + 1) / 8, 1)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in trainable):
            raise FloatingPointError("Nonfinite SFT weights")
        change = (
            sum(
                float((p.detach() - old).double().square().sum())
                for p, old in zip(trainable, before, strict=True)
            )
            ** 0.5
        )
        if change == 0:
            raise RuntimeError("SFT optimizer did not change any LoRA parameter")
        self.step += 1
        for example in examples:
            self.exposures[example.task_id] = self.exposures.get(example.task_id, 0) + 1
        self.adapter._reset_positions()
        self.poisoned = False
        return {
            **stats,
            "update": self.step,
            "lr": lr,
            "gradient_norm": float(norm),
            "parameter_delta_norm": change,
            "task_ids": [x.task_id for x in examples],
            "unique_exposed": len(self.exposures),
            "sequence_target_tokens": [len(x.target_ids) for x in examples],
            "sequence_roles": (["focus"] * 12 if self.focus_sampler else []) + ["replay"] * 4,
            "focus_mean_nll": sum(stats["sequence_nll"][:12]) / 12 if self.focus_sampler else None,
            "replay_mean_nll": sum(stats["sequence_nll"][-4:]) / 4,
            "focus_loss_contribution": sum(stats["sequence_nll"][:12]) / 16
            if self.focus_sampler
            else 0.0,
            "replay_loss_contribution": sum(stats["sequence_nll"][-4:]) / 16,
            "processed_target_sequences": len(examples),
            "processed_target_tokens": stats["target_tokens"],
            "exposures_per_task": copy.deepcopy(self.exposures),
        }

    def capture(self):
        from ..followup_updates import _forward_state
        from ..optimizer_fork import capture_state

        if self.poisoned:
            raise RuntimeError("Cannot save a failed partial update")
        return {
            **capture_state(self.adapter.model, self.optimizer),
            **_forward_state(self.adapter.model, self.adapter),
            "schema": "verified-discovery-sft-state-v1",
            "identity": self.identity,
            "parameter_names": list(self.parameter_names),
            "step": self.step,
            "scheduler": {"kind": "linear8_constant", "completed_updates": self.step},
            "sampler": {
                "focus": self.focus_sampler.state_dict() if self.focus_sampler else None,
                "replay": self.replay_sampler.state_dict() if self.replay_sampler else None,
            },
            "exposures": copy.deepcopy(self.exposures),
            "seed": self.seed,
        }

    def restore(self, state):
        from ..followup_updates import _restore_forward, _safe
        from ..optimizer_fork import restore_state

        _safe(state)
        if (
            state.get("schema") != "verified-discovery-sft-state-v1"
            or state["identity"] != self.identity
            or state["seed"] != self.seed
            or state["parameter_names"] != list(self.parameter_names)
            or state["scheduler"]
            != {"kind": "linear8_constant", "completed_updates": state["step"]}
            or not 0 <= state["step"] <= 256
        ):
            raise ValueError("Resume identity, trainable whitelist or scheduler mismatch")
        for role in ("focus", "replay"):
            sampler, saved = getattr(self, role + "_sampler"), state["sampler"][role]
            if (sampler is None) != (saved is None):
                raise ValueError("Resume sampler role differs")
            if sampler:
                # Validate on a detached sampler before model mutation.
                copy.deepcopy(sampler).load_state_dict(saved)
        self.poisoned = True
        restore_state(self.adapter.model, self.optimizer, state)
        _restore_forward(self.adapter.model, state, self.adapter)
        for role in ("focus", "replay"):
            sampler = getattr(self, role + "_sampler")
            if sampler:
                sampler.load_state_dict(state["sampler"][role])
        self.step, self.exposures = state["step"], copy.deepcopy(state["exposures"])
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
            restored = torch.load(temporary, map_location="cpu", weights_only=True)
            if state_hash(restored["state"]) != payload["state_hash"]:
                raise RuntimeError("Checkpoint roundtrip differs")
            os.link(temporary, path)  # atomic no-clobber publication
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
    if state_hash(payload["state"]) != payload["state_hash"]:
        raise ValueError("Student checkpoint digest mismatch")
    return payload["state"]


def load_student_into_backend(backend, checkpoint_path, expected_identity):
    """Bind final student weights to the inherited raw-only evaluation backend."""
    import torch

    from ..followup_updates import _restore_forward
    from ..modeling_v3.io import canonical_hash, sha256_file
    from ..protocol_state_probes.inference import _model_guard

    state = read_student_checkpoint(checkpoint_path)
    if state["identity"] != expected_identity:
        raise ValueError("Student evaluation identity mismatch")
    names = reopen_original_lora(backend.adapter)
    if names != state["parameter_names"] or set(names) != set(state["parameters"]):
        raise ValueError("Student evaluation whitelist mismatch")
    params = dict(backend.adapter.model.named_parameters())
    with torch.no_grad():
        for name in names:
            saved, live = state["parameters"][name], params[name]
            if saved.shape != live.shape or saved.dtype != live.dtype:
                raise ValueError("Student weight shape/dtype differs")
        for name in names:
            params[name].copy_(state["parameters"][name].to(params[name].device))
    _restore_forward(backend.adapter.model, state, backend.adapter)
    for parameter in params.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    backend.adapter.model.eval()
    backend.adapter._reset_positions()
    backend._guard_baseline = _model_guard(backend.adapter.model)
    backend.receipt = {
        **backend.receipt,
        "student_checkpoint": str(checkpoint_path),
        "student_step": state["step"],
        "student_identity": expected_identity,
        "inference_fingerprint": canonical_hash(
            {
                "parent": backend.receipt["inference_fingerprint"],
                "student_sha256": sha256_file(checkpoint_path),
            }
        ),
    }
    return backend
