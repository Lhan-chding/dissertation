"""Recompute pure native delta-rule intermediates without checkpointing a cache.

This is a technical execution policy, not a replacement numerical kernel. The
context must cover gradient forward construction; returned autograd graphs keep
their original callable after context exit and support both backward traversals.
Native mutable convolution/cache updates are deliberately never wrapped.
"""

from contextlib import contextmanager

POLICY = "native_pure_delta_rule_nonreentrant_recompute_v1"
KERNEL_NAMES = ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")


def checkpoint_pure_delta_rule(kernel, statistics, name):
    """Wrap an explicitly verified pure native callable, preserving all arguments."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    expected = {
        "chunk_gated_delta_rule": native.torch_chunk_gated_delta_rule,
        "recurrent_gated_delta_rule": native.torch_recurrent_gated_delta_rule,
    }
    if name not in expected or kernel is not expected[name]:
        raise PermissionError("Recomputation requires the inspected pure native torch delta rule")

    def wrapped(*args, **kwargs):
        if not torch.is_grad_enabled():
            statistics["bypass_calls"][name] += 1
            return kernel(*args, **kwargs)
        statistics["checkpoint_calls"][name] += 1
        executions = 0

        def execute(*inner_args, **inner_kwargs):
            nonlocal executions
            if executions:
                statistics["recompute_calls"][name] += 1
            executions += 1
            return kernel(*inner_args, **inner_kwargs)

        return checkpoint(
            execute,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
            determinism_check="default",
            **kwargs,
        )

    # Do not disguise the wrapper as the frozen native callable. The integration
    # must authenticate its own technical policy and validate native identity
    # outside this context before constructing any computation graph.
    wrapped.native_delta_rule = kernel
    wrapped.recompute_policy = POLICY
    return wrapped


@contextmanager
def gated_delta_recompute(model, *, expected_linear_count=None):
    """Temporarily replace only pure kernels; fully restore modules on all exits.

    Intended for one model owned by one process, in frozen eval/dropout-zero
    mode. Unknown/fused implementations and nested use fail closed. The context
    does not touch parameters, tensors, RNG, cache containers, or precision.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    if model.training:
        raise PermissionError("Recomputation requires frozen eval mode")
    modules = [m for m in model.modules() if isinstance(m, native.Qwen3_5GatedDeltaNet)]
    if not modules or (expected_linear_count is not None and len(modules) != expected_linear_count):
        raise PermissionError("Unexpected native linear-attention module count")
    statistics = {
        "policy": POLICY,
        "linear_modules": len(modules),
        "checkpoint_calls": dict.fromkeys(KERNEL_NAMES, 0),
        "recompute_calls": dict.fromkeys(KERNEL_NAMES, 0),
        "bypass_calls": dict.fromkeys(KERNEL_NAMES, 0),
        "restored": False,
    }
    replacements = []
    # Validate every callable before mutating even the first module.
    for module in modules:
        for name in KERNEL_NAMES:
            original = getattr(module, name)
            wrapper = checkpoint_pure_delta_rule(original, statistics, name)
            replacements.append((module, name, original, wrapper))
    try:
        for module, name, _original, wrapper in replacements:
            setattr(module, name, wrapper)
        yield statistics
    finally:
        for module, name, original, _wrapper in replacements:
            setattr(module, name, original)
        statistics["restored"] = True
