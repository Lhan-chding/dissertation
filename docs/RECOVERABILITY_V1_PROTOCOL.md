# Recoverability-gated answer-source decomposition v1

## Claim boundary and prior evidence

The single registered `CVA-Chart-Pilot-v0.3` calibration is complete and failed.
It produced 200 first responses, answer accuracy `0.355`, parse rate `0.935`,
natural-perception-error rate `0.225`, and the fixed taxonomy counts
`67/13/75/41/1/3` for none, parse failure, reasoning error, visual error,
compensated visual error, and operator-invariant visual error. The gate failed,
the process exited `3`, and the original Pilot A and Pilot B were terminated
without training. That calibration must not be rerun or selected among repeated
attempts.

This result does not show that the broader research idea is false. It shows
that the original confirmatory path lacked enough naturally compensated visual
errors: only one strict compensated case was observed. The revised study asks a
narrower question: when a frozen erroneous mediator is recoverable from
Stage-2-visible constraints, does a valid redundancy cue causally increase
faithful correction relative to matched controls?

No individual trajectory is labelled “genuine reasoning repair.” A trace can
only be a `faithful_repair_candidate` after strict parsing, deterministic DSL
execution, answer/program agreement, and audited cue-variable data flow. The
phrase *causal repair capacity* is reserved for aggregate effects in randomized
matched arms.

## Frozen protocol

The protocol file is `configs/recoverability/recoverability_v1.yaml`; its
tracked status remains `PREREGISTERED_NOT_RUN`. The model stays
`Qwen2.5-VL-3B-Instruct`. Natural model responses have zero retries.

### Measurement bridge

The bridge uses the frozen 300-scene v0.3 IID split. Each scene has two
protocol trajectories but three model calls:

1. legacy one-stage structured response (one image call);
2. Stage 1 evidence-only response (one image call); and
3. Stage 2 restricted-DSL response from frozen Stage-1 evidence (one text call).

The bridge therefore contains 600 protocol trajectories and at most 900 model
calls. A Stage-1 parse failure remains a failure and is not sent to Stage 2, so
the observed call count is `600 + parsed Stage-1 scenes`.
It is a measurement-equivalence audit, not a training run. The two protocols
are mergeable only if Stage-1 parsing is at least `0.98`, program/answer
consistency is at least `0.95`, and the registered absolute differences stay
within `0.03`. A failure means subsequent results must be reported separately
by protocol; it does not by itself establish or refute causal recoverability.

### Phase N: natural prevalence

Phase N is frozen at 4,000 semantic scenes with exactly one natural Stage-1
trajectory per scene. There is no extension and no “draw until an error.” The
primary denominator is fully parsed, operator-sensitive natural-perception
errors. The primary parameter is

```text
theta = P(strict natural repair candidate | parsed operator-sensitive error).
```

The one-sided exact binomial null is `theta >= 0.05`. The low-prevalence claim
is supported only when the one-sided 95% Clopper-Pearson upper bound is strictly
below `0.05` and at least 800 eligible errors are observed. Otherwise the
result is inconclusive. The report must also include all-attempt prevalence,
parsed prevalence, parse rate, and worst-case parse-failure sensitivity bounds.

### Phase C: randomized causal capacity

Phase C screens a fixed 6,000-scene intake and deterministically selects exactly
800 independent scenes: 267 cross-series, 267 trend, and 266 duplicate-encoding.
Underfilled quotas fail closed; scenes are never redistributed or added.
Cross-series and trend are confirmatory families; duplicate-encoding is
exploratory. Each selected scene receives all six matched arms with eight fixed
forks per arm:

- ablated;
- valid redundancy cue;
- matched sham cue;
- legal coherent counterfactual cue/world;
- oracle-perception diagnostic; and
- operator-swap diagnostic.

Forks are Monte Carlo repeats nested inside scenes, never independent sample
size. The primary outcome is faithful success: correct answer, valid restricted
DSL, successful deterministic execution, executed result equal to final answer,
and required cue variables present on the executor dataflow. Parse and execution
failures score zero. The primary estimand is the equal-stratum scene-level
paired effect `Valid - Ablated` among the frozen eligible mediator population;
all-scene ITT is reported separately.

The preregistered causal gate requires all of the following:

- grammar-parse lower bound at least `0.98`;
- program/answer/dataflow consistency lower bound at least `0.95`;
- actual eligible support at least the power target and the frozen family quotas;
- positive recoverable-effect and recoverability-interaction lower bounds;
- sham, nonrecoverable, and operator-invariant equivalence within `+/-0.02`
  using 90% paired scene-cluster intervals;
- positive counterfactual target shift and original-answer suppression;
- the frozen counterfactual control gate; and
- the same effect direction in both confirmatory redundancy families.

The primary confidence level is `0.95`, alpha is `0.05`, the target effect is
`0.05`, target power is `0.90`, and the scene-cluster bootstrap uses 10,000
resamples. The frozen power artifact selects 800 eligible scenes and is bound
by SHA-256 in the protocol.

Any failed or inconclusive gate leaves `rl_authorized=false`. The design does
not permit optional stopping, extra scenes/forks, adaptive threshold changes,
or confirmatory RL after inspecting a failed result.

## Current execution boundary

All theory, leakage, legal-world, DSL, selection, statistical, power, fixture,
preflight, evidence-capture, and bridge contracts are implemented locally.
The tracked 50-scene fixture is model-free and passed its audit. No new VLM
inference or training result is included in the repository.

The next execution step is the reviewed offline-server sequence in
`docs/SERVER_SETUP.md`: metadata preflight, capture of the already-completed
v0.3 bytes, then the fixed 300-scene bridge. Stop after the bridge and review
its report before authorizing Phase N or Phase C.
