"""Independent trainable-parameter/Adam/all-RNG forks and strict local checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path


def _cpu_copy(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def capture_state(model, optimizer, metadata=None, scheduler=None):
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "parameters": {
            name: _cpu_copy(p) for name, p in model.named_parameters() if p.requires_grad
        },
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "scheduler": _cpu_copy(scheduler.state_dict()) if scheduler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state().clone(),
            "cuda": [v.cpu().clone() for v in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        },
        "metadata": copy.deepcopy(metadata or {}),
    }


def restore_state(model, optimizer, state, scheduler=None):
    import numpy as np
    import torch

    actual = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(actual) != set(state["parameters"]):
        raise ValueError("Trainable parameter names differ from checkpoint")
    with torch.no_grad():
        for name, parameter in actual.items():
            if parameter.shape != state["parameters"][name].shape:
                raise ValueError(f"Checkpoint parameter shape mismatch: {name}")
            parameter.copy_(state["parameters"][name].to(parameter.device))
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    random.setstate(state["rng"]["python"])
    numpy_state = state["rng"]["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"]:
        if not torch.cuda.is_available() or len(state["rng"]["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device topology differs from checkpoint")
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return copy.deepcopy(state["metadata"])


def state_hash(value):
    """Stable content hash: dtype/shape and every byte, including BF16 tensors."""
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for member in item:
                visit(member)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def parameter_hash(model, *, trainable):
    # Hash each tensor and discard its CPU copy before moving to the next tensor.
    return state_hash(
        {
            name: state_hash(p)
            for name, p in model.named_parameters()
            if p.requires_grad == trainable
        }
    )


def save_checkpoint(path, state, identity):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    payload = {"identity": copy.deepcopy(identity), "state": state, "state_hash": state_hash(state)}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, expected_identity):
    import torch

    # weights_only prevents arbitrary Python code in externally modified files.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload["identity"] != expected_identity:
        raise ValueError("Checkpoint model/data/config identity mismatch; refusing resume")
    if state_hash(payload["state"]) != payload["state_hash"]:
        raise ValueError("Checkpoint state checksum mismatch")
    return payload["state"]


def compare_parameters(left, right, *, atol=1e-6, rtol=1e-5):
    import torch

    if set(left["parameters"]) != set(right["parameters"]):
        raise ValueError("Candidate parameter keys differ")
    differences = [
        float((left["parameters"][name] - right["parameters"][name]).abs().max())
        for name in left["parameters"]
    ]
    return {
        "bitwise_equal": state_hash(left) == state_hash(right),
        "parameters_max_abs_difference": max(differences, default=0.0),
        "parameters_allclose": all(
            torch.allclose(
                left["parameters"][name], right["parameters"][name], atol=atol, rtol=rtol
            )
            for name in left["parameters"]
        ),
        "atol": atol,
        "rtol": rtol,
    }


def main(argv=None):
    """Contract-only cold/warm fork entry point.

    A fork needs a real adapter/checkpoint supplied by the server runner.  The
    CPU command records the requested state and refuses to manufacture an
    optimizer update or CUDA execution result.
    """
    import argparse
    import json
    from pathlib import Path

    from .next_stage_common import dry_run_plan, load_yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    parser.add_argument("--state", choices=("cold", "warm"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": "R3-" + args.state,
                    **dry_run_plan(config),
                    "no_commit": args.no_commit,
                    "training_started": False,
                },
                indent=2,
            )
        )
        return 0
    if args.state == "warm" and args.checkpoint is None:
        parser.error("--checkpoint is required for warm state")
    details = {
        "status": "BLOCKED",
        "execution_kind": "CPU_AUDIT",
        "reason": "real model bank and adapter state are required; no fork was created",
        "no_commit": args.no_commit,
        "training_started": False,
    }
    if args.out:
        from .next_stage_common import stage_status

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bank_manifest.json").write_text(
            json.dumps({"status": "NOT_STARTED"}, indent=2) + "\n"
        )
        (out / "joint_advantage_checks.json").write_text(
            json.dumps(
                {"status": "CPU_REFERENCE_AVAILABLE", "bank": "requires raw on-policy rows"},
                indent=2,
            )
            + "\n"
        )
        (out / "gradients_summary.parquet.status").write_text(
            "NOT_CREATED: real model gradients required\n"
        )
        (out / "candidate_manifest.json").write_text(
            json.dumps({"status": "NOT_STARTED"}, indent=2) + "\n"
        )
        (out / "paired_response.csv").write_text(
            "status,reason\nNOT_MEASURED,real optimizer fork required\n"
        )
        (out / "support_control_report.md").write_text("# R3\n\nNo real model fork was created.\n")
        stage_status(out, "R3-" + args.state, "BLOCKED", "CPU_AUDIT", details)
    print(json.dumps({"phase": "R3-" + args.state, **details}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
