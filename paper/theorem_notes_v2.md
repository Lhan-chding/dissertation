# Theorem notes v2: natural trajectories and partial identification

Last updated: 2026-08-14 (Asia/Singapore, UTC+08).

This is the primary mathematical contract for the v2 study. The older
`theorem_notes.md` is retained for the controlled v1 mechanism but does not
override the distinctions below.

## Three estimands

For a pre-registered operational interface `m`, natural error label `E`, and
input `x`, the project keeps three quantities separate:

```text
c_sel(e|x)  = P_base(R=1 | E=e, x)
c_fork(e|x) = E[R | do_m(Z=z), z sampled from natural p(z|E=e,x), x]
c_syn(e|x)  = E[R | do_m(Z=z), z sampled from synthetic q(z|e,x), x]
```

`c_sel` is the exact input to trajectory selection. `c_fork` asks whether a
naturally occurring mediator state remains compensable after the image is cut
and downstream continuations are independently resampled. `c_syn` measures an
artificial stress test. It is never relabelled as natural evidence.

The mediator/coordination gap is `c_sel-c_fork`. The transport gap is
`c_syn-c_fork`. A large transport gap invalidates direct natural-error
interpretation of a synthetic intervention.

## Natural selection law

For the KL-regularized trajectory optimum,

```text
pi*(tau|x) proportional to pi0(tau|x) exp(R(tau)/beta).
```

Marginalizing trajectories gives

```text
mu*(e|x) proportional to mu0(e|x) M_beta(e|x),
M_beta(e|x) = E_pi0[exp(R/beta) | E=e,x].
```

For binary reward, `M_beta=1+(exp(1/beta)-1)c_sel`. Therefore the severity
shift is governed by the prompt-conditional covariance of task-induced
severity and `c_sel`, not by `c_syn`. The 10,000-table randomized property test
uses absolute tolerance below `1e-12`.

## Non-identifiability and operational interfaces

If `F=R o P`, any bijection `T` produces the equivalent factorization
`F=(R o T^-1) o (T o P)`. Final black-box behavior therefore cannot identify a
unique perception/reasoning boundary. The implementation permits only an
`operational compensation certificate under interface family M` for black-box
experiments.

## Task-induced geometry

The primary severity is a pseudometric over answers to diagnostic queries,
not raw Euclidean hidden-state distance. The property tests verify
non-negativity, symmetry, the triangle inequality, and zero distance between
task-equivalent states.

## Crossed risks

For model/oracle perception crossed with model/oracle reasoners:

```text
D_P   = L_MO - L_OO
D_R   = L_OM - L_OO
Gamma = L_MM - L_MO - L_OM + L_OO
L_MM - L_OO = D_P + D_R + Gamma
```

This identity holds for arbitrary bounded scalar loss and nonlinear models.
`Gamma<0` is a compensation interaction only at the selected operational
interface. A spurious-compensation claim additionally requires positive
isolated deficits, strong IID outcome, mechanism-shift failure, natural
trajectory support, and a simultaneous same-sign result across valid
interfaces.

## Frozen regimes

- **F0**: every parameter and sampling step upstream of the mediator is
  frozen. `D_P` is invariant under the same interface and estimator.
- **F1**: the visual tower is frozen while projector/readout or language layers
  can change. Acquisition is fixed, but operational perception can change.
- **F2**: visual acquisition can change. Only operational, multi-interface
  conclusions are permitted; parameter modules are not equated with cognitive
  functions.

## Actual checkpoint selection

The optimizer-independent ratio is `s_t(e|x)=mu_{t+1}/mu_t` on common support.
The exact severity shift equals `Cov_mu_t(severity,s_t)`. New support is
reported separately through `new_support_mass`; it is never hidden by
smoothing. `epsilon_alg` measures deviation of actual selection from the
KL/`c_sel` prediction.

## Current evidence boundary

The theory, schema, and CPU property tests are implemented. The controlled
small-neural natural replay is complete. No Qwen checkpoint was downloaded or
trained by the v2 work, and none of the formulas above is presented as an
observed real-VLM mechanism.
