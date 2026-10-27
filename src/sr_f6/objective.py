"""SR-F6 whole-sequence objective; this is not an optimizer or GPU runtime.

The auxiliary coefficient already contains lambda exactly once. Native, aux,
and weighted KL are separate graphs so diagnostics need not count KL twice.
All completion tokens (including a sampled EOS) belong in these unpadded vectors.
The caller must enforce the batch probability gate before any optimizer update.
"""

from __future__ import annotations

import math

KL_COEFFICIENT = 0.02
TIS_CAP = 2.0
MAX_G8_AUXILIARY = 7 / 16


def _coefficient(value, *, name, bound):
    import torch

    if isinstance(value, torch.Tensor) and (value.numel() != 1 or value.requires_grad):
        raise ValueError(f"{name} must be one detached scalar")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be one finite scalar") from exc
    if not math.isfinite(number) or abs(number) > bound + 1e-7:
        raise ValueError(f"{name} outside registered G8 bounds")
    return number


def sequence_loss(current, sampler, reference, native_adv, aux_adv=0.0):
    """Return one token mean with detached current-old and fixed cap-2 TIS.

    All three probability vectors must be FP32 and on the same device. There
    is deliberately no configurable dose, KL, clipping, mask, or denominator.
    A full update must weight all 128 returned sequence losses by 1/128.
    """
    import torch

    vectors = (current, sampler, reference)
    if not all(isinstance(value, torch.Tensor) for value in vectors):
        raise TypeError("Original completion probabilities must be tensors")
    if any(value.dtype != torch.float32 for value in vectors):
        raise ValueError("Registered probability dtype is float32")
    if any(value.device != current.device for value in vectors):
        raise ValueError("Probability vectors must share one device")
    if (
        current.ndim != 1
        or current.numel() == 0
        or sampler.shape != current.shape
        or reference.shape != current.shape
    ):
        raise ValueError("Nonempty unpadded original completion vectors must align")
    if not current.requires_grad or sampler.requires_grad or reference.requires_grad:
        raise ValueError("Current needs gradients; sampler/reference must be detached")
    if not all(bool(torch.isfinite(value).all()) for value in vectors):
        raise FloatingPointError("Nonfinite original completion logprob")
    native_value = _coefficient(native_adv, name="native", bound=math.sqrt(7))
    aux_value = _coefficient(aux_adv, name="auxiliary", bound=MAX_G8_AUXILIARY)
    old = current.detach()
    ratio = torch.exp(current - old)
    delta = old - sampler
    # This is exactly min(2, exp(delta)), evaluated without intermediate overflow.
    # It does not clamp the reference/current displacement or hide diagnostics.
    weights = torch.exp(delta.clamp(max=math.log(TIS_CAP))).detach()

    def policy(value):
        coefficient = current.new_tensor(value)
        return -weights * torch.minimum(coefficient * ratio, coefficient * ratio.clamp(0.8, 1.2))

    native_tokens = policy(native_value)
    aux_tokens = policy(aux_value)
    displacement = reference - current
    kl_tokens = torch.expm1(displacement) - displacement
    native = native_tokens.mean()
    auxiliary = aux_tokens.mean()
    kl = KL_COEFFICIENT * kl_tokens.mean()
    total = native + auxiliary + kl
    if not all(
        bool(torch.isfinite(value).all())
        for value in (ratio, weights, native_tokens, aux_tokens, kl_tokens, total)
    ):
        raise FloatingPointError("Nonfinite SR-F6 objective; no silent ratio clamp")
    return {
        "loss": total,
        "native": native,
        "aux": auxiliary,
        "kl": kl,
        "kl_unweighted": kl_tokens.mean(),
        "ratio": ratio,
        "weights": weights,
        "sampler_train_delta": delta.detach(),
        "reference_current_delta": displacement.detach(),
        "absolute_difference": delta.abs().detach(),
        "tis_truncated": (delta > math.log(TIS_CAP)).detach(),
        "token_count": current.numel(),
        "kl_coefficient": KL_COEFFICIENT,
    }
