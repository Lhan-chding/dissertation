"""Native runtime with independent draw RNG and inexpensive run identities."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path

from sr_f3_ce.runtime import SRF3CERuntime, configure_parent_training

from .common import append, now, read, write


@contextmanager
def independent_draw_rng(torch, seeds):
    """Keep native batched generation, give every sample its own RNG stream.

    Each worker is a separate single-threaded model process. The scoped override
    affects only multinomial sampling, leaves probability tensors unchanged, and
    is restored even on failure. Shape checks fail closed if the backend changes.
    """
    original = torch.multinomial
    generators = {}

    def sample(probabilities, num_samples, replacement=False, *, generator=None, out=None):
        if probabilities.ndim != 2 or probabilities.shape[0] != len(seeds) or num_samples != 1:
            raise RuntimeError("Unexpected native token sampler shape")
        if generator is not None or out is not None:
            raise RuntimeError("Unexpected native token sampler options")
        device = str(probabilities.device)
        if device not in generators:
            generators[device] = [torch.Generator(device=device).manual_seed(s) for s in seeds]
        return torch.stack(
            [
                original(p, 1, replacement, generator=g)
                for p, g in zip(probabilities, generators[device], strict=True)
            ]
        )

    torch.multinomial = sample
    try:
        yield
    finally:
        torch.multinomial = original


class Runtime(SRF3CERuntime):
    def current_adapter_identity(self):
        return dict(
            parent=self.identity.get("parent_id"),
            run_id=self.identity.get("run_id"),
            step=self.identity.get("step", 0),
            adapter_path=self.adapter_path,
            identity_method="registered_parent_and_committed_step",
        )

    def stable_model_identity(self):
        return {
            key: copy.deepcopy(self.identity.get(key))
            for key in (
                "run_id",
                "parent_id",
                "step",
                "dtype",
                "attention_backend",
                "torch_version",
                "transformers_version",
                "protocol_revision",
            )
        }

    def generate_group(self, model_input, run_root, seeds, **kwargs):
        with independent_draw_rng(self.torch, seeds):
            result = super().generate_group(model_input, run_root, seeds, **kwargs)
        for row in result:
            row["seed_semantics"] = "independent_per_draw_torch_generator"
        return result


def load(root, block, task_id, *, training=False):
    import torch

    from mm_core.vl_runtime import seed_all
    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend
    from sr_f3_ce.training import fresh_optimizer
    from srf4_ref.sampling import seed_for

    root = Path(root)
    config = read(root / "CONFIG.json")
    adapter = Path(config["parents"][str(block)])
    configure_audited_backend()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must own exactly one GPU")
    seed_all(seed_for(2026102201, "load", block))

    def account(kind, count, metadata):
        if not isinstance(count, int) or count < 0:
            raise ValueError("Invalid native execution accounting count")
        append(
            root / "accounting" / f"{task_id}.jsonl",
            dict(at=now(), task_id=task_id, kind=kind, count=count, metadata=metadata),
        )

    runtime = Runtime(
        config["model_path"],
        adapter_path=str(adapter),
        output_root=root,
        protocol_revision="v2",
        account=account,
    )
    configure_parent_training(runtime, parent_adapter_config=read(adapter / "adapter_config.json"))
    runtime.identity.update(parent_id=f"SRF2_J6_s{71001 + block}", run_id=task_id, step=0)
    write(
        root / "runtime_identity" / f"{task_id}.json",
        dict(
            runtime.identity,
            hardware=actual_cuda_identity(),
            identity_note="reuse registered snapshot; no full weight rehash",
        ),
    )
    optimizer, scheduler = fresh_optimizer(runtime.model) if training else (None, None)
    return runtime, optimizer, scheduler
