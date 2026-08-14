# Theorem notes and verification boundary (v1 controlled mechanism)

Last updated: 2026-08-14 (Asia/Singapore, UTC+08).

This file records the mathematical contract implemented by the repository. It
does not substitute for a formal proof. The analytical derivations come from the
research plan; CPU property tests check that code agrees with those derivations
on registered fixtures and randomized finite instances.

The primary natural-trajectory contract is now `theorem_notes_v2.md`. In
particular, v2 distinguishes `c_sel`, `c_fork`, and `c_syn`; where this legacy
note uses a fixed interventional `c(e)`, it describes the controlled
fixed-reasoner construction and must not be substituted for natural VLM
selection evidence.

## Status vocabulary

- **Analytical statement**: an identity or conditional result whose assumptions
  are written below. A paper proof is still required.
- **CPU check implemented**: executable tests exist. A check is called
  **registry-recorded** only when the canonical run and required provenance are
  accepted as `VERIFIED_CPU` in `experiment_registry.md`.
- **Empirical hypothesis**: a falsifiable prediction, not a theorem.
- **Not claimable**: deliberately outside the novelty or evidence boundary.

All selection statements are conditional on a prompt/sample `x`. Dataset-level
effects are averages of prompt-conditional quantities; an unstratified pooled
covariance is not an interchangeable substitute.

## Notation and assumptions

For sample `x`, let `e` index a finite, executable perception-error catalog,
`e=0` denote the truthful state, and `ell_x(e)` be a task-semantic severity fixed
without reference to the final reward. The base trajectory law factors as

```text
pi_0(e, omega | x) = mu_0(e | x) rho_0(omega | e, x).
```

The reward `R_x(e, omega)` lies in `[0, 1]`. The legacy controlled mechanism uses
interventional compensability

```text
c_phi(e | x) = E[R_x(e, omega) | do(E=e), x].
```

Here `phi` identifies the frozen reasoner/checkpoint used for the intervention.
It must be estimated by hiding the original image and injecting a controlled
perceived state into the same reasoner. This is `c_syn` unless the injected
state is an exact replay of a naturally sampled mediator. Natural conditional
success `c_sel`, rather than this interventional quantity, is the exact input
to the v2 trajectory selection law. The current CPU interface check only omits
an explicit image argument; until a separate text-only worker and reviewed
adapter are implemented, it does not establish the required image isolation.

## Proposition 0: ideal-reasoner dominance boundary

**Analytical statement.** Suppose an erroneous state is a stochastic degradation
of the truthful state, `z_e = G_e(z_0)`, and the reasoner class receiving `z_0`
can first apply `G_e` and then simulate every policy available from `z_e`. Then

```text
sup_rho E[R | z_0] >= sup_rho E[R | z_e].
```

**Use.** An observed `c(e) > c(0)` cannot be attributed merely to an abstractly
"strong" reasoner. At least one ideal condition fails: capacity, interface,
optimization, distribution, shared-parameter, or reward alignment.

**Verification boundary.** This is an information-dominance argument, not a
numerical VLM result. The current suite does not certify that a real model class
satisfies its simulation assumption.

## Theorem 1: compensability selection law

For fixed `x`, positive `beta`, and a base distribution with support containing
all candidate trajectories, consider

```text
max_pi E_pi[R_x] - beta KL(pi || pi_0).
```

The unique optimum on that support is

```text
pi_beta*(e, omega | x)
  = pi_0(e, omega | x) exp(R_x(e, omega) / beta) / Z_x.
```

Define the reward-moment multiplier

```text
M_beta,x(e) = E_{omega ~ rho_0(. | e,x)}[exp(R_x(e,omega) / beta)].
```

Marginalizing trajectories gives the exact selection law

```text
mu_beta*(e | x)
  = mu_0(e | x) M_beta,x(e) / E_{mu_0}[M_beta,x(E)].
```

For any finite statistic `f(e)`, its exact change is

```text
E_{mu_beta*}[f] - E_{mu_0}[f]
  = Cov_{mu_0}(f(E), M_beta,x(E)) / E_{mu_0}[M_beta,x(E)].
```

For binary reward, with `alpha_beta = exp(1/beta) - 1`, this becomes

```text
L_P,beta - L_P,0
  = alpha_beta Cov_{mu_0}(ell(E), c_0(E | x))
    / (1 + alpha_beta E_{mu_0}[c_0(E | x)]).
```

Here `c_0` is the natural conditional success probability under the base
trajectory law. It equals an interventional frozen-reasoner success probability
only in the registered controlled construction or after replay sufficiency is
established.

Thus the perception-severity direction is the sign of the prompt-conditional
severity-compensability covariance. Pairwise odds obey

```text
log(mu_beta*(e)/mu_beta*(e'))
  = log(mu_0(e)/mu_0(e')) + log(M_beta(e)/M_beta(e')).
```

**Assumptions.** The optimizer is over distributions, `beta > 0`, the base
support is fixed, the reward moment is finite, and the reasoner conditional is
the one defining `M`. A neural GRPO update is not asserted to solve this optimum
at each step.

**CPU check implemented and registry-recorded.**
`src/compbias/theory/selection.py`,
`src/compbias/rl/exact_kl.py`, `tests/test_selection_identity.py`, and
`tests/test_exact_kl.py` check normalization, covariance and odds identities,
the fixed two-state fixture, extreme inputs, immutability, NumPy/Torch parity,
autograd, and 1,000 seeded random distributions.

**Novelty boundary.** Exponential tilting and the covariance identity are
classical tools. Any contribution must lie in the natural selection/forked
replay distinction, operational interface identification, derived diagnostics,
and empirical validation.

## Theorem 2: reasoner-scaling direction law

Let reasoner scale/checkpoint `kappa` induce positive multipliers
`M_beta,kappa(e)` and define

```text
mu_beta,kappa(e) proportional to mu_0(e) M_beta,kappa(e),
G_kappa(e) = partial_kappa log M_beta,kappa(e).
```

Then

```text
d/dkappa E_{mu_beta,kappa}[ell(E)]
  = Cov_{mu_beta,kappa}(ell(E), G_kappa(E)).
```

For binary reward,

```text
G_kappa(e)
  = alpha_beta partial_kappa c_kappa(e)
    / (1 + alpha_beta c_kappa(e)).
```

**Interpretation.** Overall reasoner accuracy is insufficient to predict the
perception direction. Relative gain concentrated on severe errors yields a
positive derivative; relative gain concentrated on truthful/light errors yields
a negative derivative. If `G_kappa` is monotone in severity, the covariance
sign follows the corresponding order inequality.

**CPU check implemented and registry-recorded.**
`src/compbias/theory/scaling.py` and
`tests/test_scaling_identity.py` compare the covariance formula with autograd
and central finite differences on registered cases and 1,000 seeded random
landscapes.

**Empirical hypothesis.** Real checkpoints with matched overall accuracy gains
but opposite `Cov(ell,G)` should produce opposite post-RL perception shifts.
No VLM checkpoint path has been run.

## Theorem 3: fixed-landscape lock-in

For the repeated update

```text
mu_{t+1}(e) = mu_t(e) M(e) / E_{mu_t}[M(E)],
```

the closed form is

```text
mu_t(e) = mu_0(e) M(e)^t / sum_{e'} mu_0(e') M(e')^t.
```

When `M` has a unique maximum `e*` with positive initial mass,
`mu_t(e*) -> 1`; pairwise ratios decay at the exact multiplier ratio raised to
`t`.

**Assumptions.** The landscape is fixed, finite, positive on relevant support,
and the update is the exact normalized selection operator.

**CPU check implemented and registry-recorded.**
`src/compbias/theory/lockin.py` and
`tests/test_lockin_closed_form.py` cover iteration/closed-form equality, ratio
rates, tied maxima, unique-max concentration, numerical stability, validation,
and randomized distributions.

**Non-extension.** This is not a global convergence theorem for GRPO or a neural
network whose reasoner and representation co-adapt.

## Theorem 4: joint coordination and bifurcation

Let `p=P(T)` be truthful-perception probability and `q=P(C)` be canonical-
reasoning probability under reward matrix

```text
        C          K
T       1        1-delta
E     1-epsilon    1
```

with positive mismatch penalties. The expected reward and natural-gradient /
replicator field are

```text
J(p,q) = 1 - delta p(1-q) - epsilon (1-p)q,
dot p = p(1-p)((delta+epsilon)q-delta),
dot q = q(1-q)((delta+epsilon)p-epsilon).
```

The truthful `(1,1)` and compensatory `(0,0)` corners are locally stable; the
interior point

```text
(p_s,q_s) = (epsilon/(delta+epsilon), delta/(delta+epsilon))
```

is a saddle. Along the field,

```text
dot J = p(1-p)(partial_p J)^2 + q(1-q)(partial_q J)^2 >= 0.
```

For `delta=epsilon=a`, the separatrix is `p+q=1`. With symmetric KL to the
uniform reference and `p=q=(1+m)/2`, stationary branches satisfy

```text
2 beta atanh(m) = a m,
beta_c = a/2.
```

**CPU check implemented and registry-recorded.**
`src/compbias/theory/coordination.py` and tests
`test_coordination_fixed_points.py`, `test_coordination_dynamics.py`, and
`test_bifurcation.py` check fixed points, Jacobians, Lyapunov monotonicity,
separatrix behavior, registered ODE trajectories, and the critical branch.

**Empirical boundary.** A deterministic CPU reduction can demonstrate that the
coordination mechanism survives differentiable optimization. It does not show
that Qwen2.5-VL has these two basins. That remains a preregistered, not-run VLM
hypothesis.

## Theorem 5: outcome-error coupling decomposition

For an additive task, let `y*` be the canonical answer from the true scene,
`y^P` the canonical answer from the perceived scene, and `y_hat` the model
answer. Define

```text
e_P = y^P - y*,
e_R = y_hat - y^P.
```

Then `y_hat-y* = e_P+e_R` and squared outcome loss decomposes exactly as

```text
L_O = L_P + L_R + 2 C,
L_P = E[||e_P||^2],
L_R = E[||e_R||^2],
C   = E[e_P^T e_R].
```

Across checkpoints,

```text
Delta L_O = Delta L_P + Delta L_R + 2 Delta C.
```

For a differentiable strictly convex potential `Phi`, the Bregman three-point
identity gives

```text
D_Phi(y*, y_hat)
  = D_Phi(y*, y^P) + D_Phi(y^P, y_hat) + I_Phi.
```

A negative interaction indicates compensatory coupling under the registered
orientation. The raw components must accompany any normalized contribution
ratio; a ratio is undefined when the denominator does not represent an outcome
improvement.

**CPU check implemented and registry-recorded.**
`src/compbias/eval/decomposition.py`,
`src/compbias/theory/bregman.py`, `tests/test_decomposition.py`, and
`tests/test_bregman_identity.py` check per-sample and aggregate equality, the
fixed cancellation fixture, signed Bregman interaction, nonlinear potentials,
and 1,000 seeded strictly convex quadratics.

**Scope.** The additive equality applies only when the canonical intermediate
answer and local errors are well defined. The Bregman identity does not imply
that a natural-language chain of thought is a faithful internal causal trace.

## Optional local Fisher projection

The constrained quadratic projection splits a local residual correction between
perception and reasoning parameter blocks according to their Fisher/KL cost.
`src/compbias/theory/fisher_projection.py` and
`tests/test_fisher_projection.py` check the KKT solution and randomized
positive-definite fixtures.

This is an appendix diagnostic for credit-cause mismatch. Constrained quadratic
optimization is not a title-level novelty claim.

## Falsifiable empirical hypotheses

The formal tabular approximate path constructs the fixed-reasoner joint law

```text
pi_0(e,r) = mu_0(e) Bernoulli(r; c(e)),  r in {0,1},
```

but trains REINFORCE, PPO-like, and GRPO-like finite policies only over `e`.
For every sampled error, the environment independently draws
`r~Bernoulli(c(e))`; the policy cannot choose `r` or change `P(r|e)`. The KL
penalty is applied to the error marginal, and the learned error policy is
compared with the theorem's reward-moment selection target. This registered
`raw_fixed_reasoner_outcome` path now has an accepted finite CPU record with 20
seeds per algorithm, full error-action coverage, and the registered aggregate
conditional-rate tolerance. It is not a theorem that a neural PPO/GRPO
implementation follows the optimum and supplies no real-VLM evidence.

An `unconstrained_joint_trajectory_diagnostic` separately trains a flattened
policy over `(e,r)` with joint-space KL. Although the exact joint target has the
same theorem marginal after summing over `r`, an approximate joint policy can
alter `P(r|e)`; this view is therefore ineligible for the formal fixed-reasoner
gate. A third calibration path for the same algorithms, plus deterministic
mirror descent, optimizes the collapsed error-marginal reward
`beta * log(moment multiplier)`. Both views are diagnostic-only and must never
be substituted for the formal fixed-reasoner result.

- **H1 Selection sign:** pre-RL prompt-conditional `Cov(ell,c)` predicts the
  direction of post-RL perception severity.
- **H2 Odds calibration:** pairwise error odds changes match multiplier ratios,
  up to the measured approximation gap.
- **H3 Differential scaling:** matched overall reasoner gains with opposite
  `Cov(ell,G)` produce opposite perception shifts.
- **H4 Joint bistability:** initialization, KL, and learning-rate ratio alter
  occupancy of truthful and compensatory endpoints.
- **H5 Coupling contribution:** a repeatable nonzero `2 Delta C` contributes to
  real-VLM answer gain.
- **H6 OOD fragility:** larger coupling contribution predicts a larger shift gap
  when the error mechanism changes but the task rule does not.
- **H7 Control:** an oracle perception/process signal reduces compensatory-basin
  occupancy; it is a control, not a proposed contribution.

H1, H2, and H4 have accepted controlled CPU diagnostic records. H3 has an
accepted finite tabular Experiment C with equal-average-gain `truth_gain`,
`uniform_gain`, and `error_gain` paths, covariance directions, and central
finite-difference checks; the distinct matched real-checkpoint experiment is
still pending. These CPU records validate only their finite registered models.
None of H1--H7 is established for a real VLM, and the project registry records
no MeasureBench replication execution.

## Statements excluded from the paper

The current project must not claim any of the following:

- first discovery of right-answer/wrong-reason behavior or error cancellation;
- first separation of perception and reasoning;
- first observation that outcome reward under-identifies local failure;
- novelty of KL exponential tilting, covariance, Bregman identities, replicator
  dynamics, maximal coupling, PPO, or GRPO;
- that a stronger reasoner necessarily harms perception;
- that all multimodal tasks admit an additive error decomposition;
- that explicit chain of thought reveals the model's true internal causal path;
- causal or general VLM evidence before the registered interventions and GPU
  runs are actually completed;
- external-benchmark validation before the pinned checkout and preregistered
  protocol produce hashed artifacts.
