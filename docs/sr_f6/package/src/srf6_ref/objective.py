"""Exact scalar contract for the one-update-per-rollout inherited objective.
Torch is imported lazily; CPU tests compare loss and gradients. No GPU trainer.
"""
def loss(current_logp,sampled_logp,reference_logp,a_native,c_aux,kl_beta=.02,tis_cap=2.0):
    import torch
    if current_logp.ndim!=1 or current_logp.numel()==0:raise ValueError('nonempty sequence')
    if sampled_logp.shape!=current_logp.shape or reference_logp.shape!=current_logp.shape:raise ValueError('probability alignment')
    if not all(torch.isfinite(v).all() for v in (current_logp,sampled_logp,reference_logp)):raise ValueError('nonfinite input')
    old=current_logp.detach()
    ratio=torch.exp(current_logp-old)
    w=torch.exp(old-sampled_logp.detach()).clamp(max=tis_cap).detach()
    def policy(a):
        return -w*torch.minimum(a*ratio,a*ratio.clamp(.8,1.2))
    d=reference_logp.detach()-current_logp
    kval=torch.expm1(d)-d
    native=policy(float(a_native)).mean()
    aux=policy(float(c_aux)).mean()
    kl=kl_beta*kval.mean()
    total=native+aux+kl
    if not torch.isfinite(total):raise ValueError('nonfinite objective: do not silently clamp')
    return {'loss':total,'native':native,'aux':aux,'kl':kl}
