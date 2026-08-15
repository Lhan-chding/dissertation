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
phrase *causal repair capacity* is reserved for aggregate effects in the
registered controlled matched complete-crossover interventions.

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
consistency is at least `0.95`, and both paired 90% scene-level bootstrap
intervals for accuracy and perception differences lie strictly inside
`+/-0.03`. A failure means subsequent results must be reported separately by
protocol; it does not by itself establish or refute causal recoverability.

Bridge v1 has now completed once and is frozen as
`FINAL_FAILED_STAGE1_INTERFACE`. The legacy interface remained operational
(`parse_rate=0.9633333333333334`, `answer_accuracy=0.39666666666666667`),
whereas strict Stage 1 parsed `0/300` outputs and therefore invoked Stage 2
zero times. Strict replay attributes the 300 failures to 192 partial
`target_facts` arrays, 100 Markdown-wrapped objects, and eight schema-key
drifts. Consequently the bridge did not test the recoverability hypothesis;
the zero two-stage accuracy is a mechanical consequence of the Stage-1
measurement failure. The source hashes and this interpretation are frozen in
`configs/recoverability/bridge_v1_failure.yaml`. Bridge v1 must not be rerun.
The shell printed `bridge_exit=3` outside the redirected Bridge console file;
its exit status is therefore derived from the frozen failed report together
with the hash-locked Bridge CLI, while the console bytes remain independently
hash-bound.

The next step is a separate development-only Stage-1 v2 probe on 24 frozen
`dev` scenes (four per chart-type by operation stratum). Its prompt receives no
downstream question and binds a literal four-slot JSON grammar. It makes one
image call per scene, uses no retries, and neither invokes Stage 2 nor tests a
scientific hypothesis. Passing requires strict parsing on all 24 scenes. The
probe may guide a future preregistered Bridge v2 on an untouched split, but its
development results cannot be reported as confirmatory evidence.

That Stage-1 v2 probe is now complete and externally frozen. It parsed all
`24/24` outputs and exactly transcribed `22/24`. The two mismatches were
`dev-000003` and `dev-000019`; both were line charts whose A value was read one
unit low. Direct visual review found clear integer-grid placement, adequate
left padding, and no clipping or mark/axis overlap, so these remain natural
model perception errors rather than renderer defects. The probe still did not
test recoverability and does not authorize Bridge v2.

Stage 2 was never reached in Bridge v1, so its model-facing restricted-DSL
interface remains untested. The next registered step is therefore a separate
one-shot Stage-2 v1 development probe. It reuses the 24 frozen Stage-1 outputs,
makes exactly 24 text-only model calls, exposes no image, question, gold value,
or correct answer, and uses zero retries. Each operation receives one exact
DSL grammar whose variables are bound to the frozen perceived values. Passing
requires `24/24` strict program parses, executions, program-answer matches, and
matches to the deterministic operation applied to those perceived values. It
does not test a scientific hypothesis or authorize confirmatory execution.

That Stage-2 v1 probe is now complete and externally frozen as a failed
development interface. Strict programs parsed and executed for `19/24`
scenes, but only `13/24` had an `answer` equal to the executed result. The
remaining failures were five strict parse failures and six program-answer
mismatches. This is not evidence against recoverability: the probe supplied no
repair cue and tested only whether the base model could satisfy the registered
DSL contract. It must not be rerun. The required zero-model-call replay is now
complete and externally anchored. It verified all 24 raw outputs, the `19/24`
parse/execution count, the `13/24` executed-result matches, and the five/six
failure split. The diagnostic SHA-256 is
`d85510ea829a000bc31002f874e5a0ec795421aadec9f9042438d78337d9e7b4`.

The failure exposes one removable interface defect: v1 asked the model to emit
both an executable graph and a second numeric `answer`, even though those two
fields could disagree. Stage-2 v2 therefore preserves the strict parser,
zero-retry rule, trusted value bindings, and operation graph, but replaces the
numeric answer with a final result pointer (`"return":"result"`). The trusted
executor is the sole source of the numeric final answer. This does not repair,
normalize, or reinterpret model output.

Exactly one 24-scene, text-only Stage-2 v2 development probe is now authorized.
It reuses the externally hash-bound Stage-1 v2 records and the completed v1
diagnostic, performs no image calls or training, and tests only whether this
executor-authoritative interface is mechanically viable. Passing requires
`24/24` strict parses, executions, and correct executor results. It is not a
recoverability test and cannot authorize Bridge v2, Phase N, Phase C, or RL.

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

### Phase C: controlled complete-crossover causal capacity

Phase C screens a fixed 8,000-scene intake and deterministically selects exactly
1,066 independent scenes: 400 cross-series, 400 trend, and 266
duplicate-encoding.
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
resamples. The frozen power artifact powers each confirmatory family at the
Holm-adjusted one-sided alpha `0.025`, selects 1,066 eligible scenes, and is
bound by SHA-256 in the protocol.

Any failed or inconclusive gate leaves `rl_authorized=false`. The design does
not permit optional stopping, extra scenes/forks, adaptive threshold changes,
or confirmatory RL after inspecting a failed result.

## Current execution boundary

The one-shot Bridge v1 failure, successful Stage-1 v2 development probe, failed
Stage-2 v1 development probe, and model-free v1 failure diagnostic are fully
frozen; none is evidence for or against recoverability. Phase N, Phase C, Pilot
A, Pilot B, Bridge v2, any other development probe, and all RL training remain
unauthorized. The only next server action is the one-shot 24-call Stage-2 v2
development probe in `docs/SERVER_SETUP.md`. Stop after it regardless of exit
status and return its immutable evidence for review.
