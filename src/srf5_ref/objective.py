"""CPU-testable reference of the inherited single-update TIS surrogate.
The train-old anchor is current.detach(): its forward ratio is 1. PPO clipping
therefore does not bind in this one-update recipe. Do not claim PPO trust-region
protection from it. This intentionally preserves the existing implementation.
"""
from __future__ import annotations
import math

def sequence_loss(current,sampler,reference,native_adv,aux_adv=0.0,mask=None,scope='none',kl_beta=0.02):
    import torch
    if scope not in ('none','full','selected'):raise ValueError('scope')
    if kl_beta not in (0.0,0.02):raise ValueError('unregistered KL coefficient')
    if scope=='none' and float(aux_adv)!=0:raise ValueError('unused auxiliary coefficient is not allowed')
    if current.ndim!=1 or not current.numel() or current.shape!=sampler.shape or current.shape!=reference.shape:raise ValueError('token shapes')
    if not current.requires_grad or sampler.requires_grad or reference.requires_grad:raise ValueError('detach contract')
    for x in (native_adv,aux_adv):
        if isinstance(x,torch.Tensor) and x.requires_grad:raise ValueError('advantages must be detached')
        if not math.isfinite(float(x)):raise ValueError('non-finite coefficient')
    if abs(float(aux_adv))>0.25+1e-9:raise ValueError('aux bounded once')
    current,sampler,reference=current.float(),sampler.float(),reference.float()
    if not all(bool(torch.isfinite(x).all()) for x in (current,sampler,reference)):raise FloatingPointError('nonfinite logp')
    old=current.detach();ratio=torch.exp(current-old)
    weights=torch.exp((old-sampler).clamp(max=math.log(2))).detach()
    a=torch.as_tensor(native_adv,device=current.device,dtype=torch.float32).detach()
    c=torch.as_tensor(aux_adv,device=current.device,dtype=torch.float32).detach()
    policy=-weights*torch.minimum(a*ratio,a*ratio.clamp(.8,1.2))
    z=reference-current
    kl=torch.expm1(z)-z
    native=(policy+kl_beta*kl).mean()
    candidate=-weights*torch.minimum(c*ratio,c*ratio.clamp(.8,1.2))
    if scope=='full':aux=candidate.mean()
    elif scope=='selected':
        m=torch.as_tensor(mask,device=current.device)
        if m.dtype!=torch.bool or m.shape!=current.shape:raise ValueError('boolean token mask')
        aux=candidate[m].mean() if bool(m.any()) else current.sum()*0
    else:aux=current.sum()*0
    loss=native+aux
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite objective; no silent clamp')
    return {'loss':loss,'native':native,'aux':aux,'kl':kl.mean(),'ratio':ratio,'weights':weights,
            'sampler_train_delta':(old-sampler).detach(),'reference_current_delta':z.detach()}
