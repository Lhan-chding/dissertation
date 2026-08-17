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

The one-shot 24-scene, text-only Stage-2 v2 development probe is complete and
must not be rerun. All `24/24` model graphs parsed and executed, and the trusted
executor returned the registered operation result for all `24/24` scenes. The
probe used no retries, image calls, hypothesis test, or training. This establishes
only the mechanical viability of the executor-authoritative interface; it is
not evidence for recoverability and does not authorize Bridge v2, Phase N,
Phase C, or RL. Its externally supplied hashes are frozen in
`configs/recoverability/stage2_v2_frozen_result.yaml`; the remaining evidence
step was a zero-model-call replay of all 24 stored raw graphs. That replay is
now complete: it verified all five source hashes and reproduced 24 parses, 24
executions, and 24 executor-correct results without a model call. Its external
manifest SHA-256 is
`3a9e521cfe718cc3dea9aee4f1591aac761fa47f893c986eb1ba722a44374577`,
anchored in
`configs/recoverability/stage2_v2_external_evidence_anchor.yaml`.

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

### Prospective Phase-C v2 amendment after Phase N

Phase N completed on 4,000 fixed scenes with 836 parsed operator-sensitive
errors and 33 strict natural-repair candidates. The point estimate was
`0.03947`, but the one-sided 95% Clopper-Pearson upper bound was `0.05243`.
Consequently the original `0.05` low-prevalence gate remains failed and the
original result remains inconclusive. Those facts and all seven server hashes
are immutable in `configs/recoverability/phase_n_frozen_result.yaml`.

Before any Phase-C scene, mediator, arm, fork, or outcome was generated, the
continuation rule was prospectively amended to `0.10` in
`configs/recoverability/recoverability_phase_c_v2_amendment.yaml`. This is a
versioned protocol amendment, not a claim that the original gate passed. It
authorizes the original fixed confirmatory Phase-C design under v2: an 8,000
scene screen, exact quotas of 400 cross-series, 400 trend, and 266
duplicate-encoding scenes, followed only after a successful screen by all six
matched arms and eight fixed forks. No extension, rerun, redistribution, RL,
or training is authorized by the amendment or the screen alone.

The one-shot Bridge v1 failure, successful Stage-1 v2 development probe, failed
Stage-2 v1 development probe, model-free v1 failure diagnostic, and successful
Stage-2 v2 development probe are frozen; none is evidence for or against
recoverability. Its zero-model-call evidence capture is also complete and must
not be rerun. Pilot A, Pilot B, Bridge v2, six-arm execution, and all RL
training remain unauthorized. The amended confirmatory Phase-C v2 screen is
the only newly authorized action.

The model-free measurement-qualification dataset generation is complete and
must not be rerun. It contains 300 new semantic scenes, exactly 50 per
chart-type by operation stratum, and has no numeric-table overlap with v0.3.
The fixed seed is `20260817`; records SHA-256 is
`98c1ab1228480b58dc4309f7c64280c347e87ac44547d79e36ab6ceb52adff6d`,
and the image-bundle SHA-256 is
`e01ea67f4b5ace4cec3201018ceed9cb68a5699470711e4d233ce64b5263d760`.
The attempt marker, manifest, records, image bundle, console log, generation
commit, and generation package lock are bound in
`configs/recoverability/measurement_qualification_data_anchor.yaml` and are
replayed before model loading.

The one-shot measurement qualification has completed and passed. Stage 1
parsed `299/300` outputs (`0.9967`; one-sided 95% lower bound `0.9843`). All
299 downstream Stage-2 programs parsed and executed, and the trusted executor
returned the registered answer for all 299 (each one-sided lower bound
`0.9900`). Exact transcription was `0.8533`; as preregistered, this was reported
but not gated because natural perception errors are scientific outcomes. The
five artifacts and exact metrics are frozen in
`configs/recoverability/measurement_qualification_frozen_result.yaml` and the
qualification must not be rerun.

Phase N is complete and must not be rerun. The next server action uses the
qualified Stage-1 v2 transcription interface exactly once on each of 8,000
fixed fresh Phase-C scenes. It only freezes eligible mediators and performs the
deterministic family-quota selection. It cannot invoke any arm, fork, RL, or
training. Exit `0` authorizes packaging the already specified six-arm
experiment; exit `3` is a final quota-underfill stop.

### Prospective Phase-C v3 post-screen amendment

The one-shot v2 screen completed before any arm or fork was evaluated. It
parsed 7,905/8,000 scenes and froze 580 scenes satisfying the full per-scene
eligibility predicate: 208 cross-series, 190 trend, and 182
duplicate-encoding. It exited `3` because the earlier planning quotas
400/400/266 were underfilled. The failed v2 screen result, empty atomic
selection, exit code, and seven server hashes remain unchanged in
`configs/recoverability/phase_c_screen_v2_frozen_result.yaml`.

Before observing any six-arm outcome, v3 withdraws the fixed family quotas as
an execution gate and includes every one of the 580 already-frozen eligible
scenes. This is an availability-based sample-size amendment, not a statement
that the v2 quota gate passed. No screen rerun, top-up, redistribution, or
eligible-scene exclusion is allowed. The original six arms, eight nested forks,
outcome definition, scene-level pairing, 10,000-resample analysis, and
confirmatory/exploratory family labels remain unchanged. The execution therefore
contains exactly `580 * 6 * 8 = 27,840` text-only calls.

The original 0.90 target power is explicitly not met. Replaying the frozen
simulation assumptions at the available family sizes gives estimated power
0.7375 for cross-series and 0.6865 for trend. These design sensitivities were
recorded before arm outcomes and must accompany the confirmatory estimates.
They do not authorize RL or training. The post-screen amendment is frozen in
`configs/recoverability/recoverability_phase_c_v3_postscreen_amendment.yaml`.

### Phase-C prompt-interface qualification after v3 execution

The v3 six-arm execution completed all 27,840 calls, but its strict program
parse rate was zero. The stored raw responses show that the prompt did not
define the three constraint payloads or the exact step-object schema with enough
specificity. Consequently the all-zero strict report is treated as an
interface-invalid measurement of the intended semantic comparison; it is not
relabelled as a scientific null and the costly arm run is not repeated.

The next action is a bounded diagnostic prompt qualification, not a new
confirmatory experiment. It selects one already-frozen eligible scene from each
of the nine family-by-operation cells, compares a no-cue prompt with a valid-cue
prompt, and uses two fixed forks per condition: exactly 36 text-only calls. The
prompt explicitly defines index-to-variable mapping, `known_value`, `pair_sum`,
`arithmetic_progression`, the at-most-one-error rule, and an exact JSON example
for each operation. Reporting separates strict schema parsing from conservative
format-independent recovery of the four inferred integers and the answer
computed from those integers.

This diagnostic has no preregistered pass threshold, performs no hypothesis
test, and cannot authorize a scaled run, RL, or training. Its report always
records `hypothesis_tested=false` and `scale_authorized=false`. Any later scale
decision requires explicit human review of all 36 raw records and a separately
versioned prospective execution plan.

### Phase-C world-recovery-only twelve-call diagnostic

The completed 36-call prompt qualification is immutable and must not be rerun.
Its fair post-hoc audit scores both conditions against the same hidden true
world and shows that the combined recovery-plus-DSL interface remains
unqualified. The next diagnostic therefore removes the operation, result
program, and free-form explanation entirely. Its only requested output is the
four recovered integers.

The prospective plan selects exactly two deterministic held-out cases from each
of `cross_series`, `duplicate_encoding`, and `trend`. Each case is evaluated
once with no facts and once with its registered valid facts, giving exactly
`3 * 2 * 2 = 12` calls. Decoding is greedy, `max_new_tokens=32`, and no retry,
sampling, self-consistency, external solver, RL, or training is allowed. The
system prompt, user templates, configuration, renderer, parser, runner, and
inherited server package are SHA-256 locked before model loading. The selected
hidden/public manifests and exact rendered messages are written before the
model is loaded and are never overwritten.

Both conditions are scored against the same hidden truth. Exact whole-response
CSV compliance is separate from conservative semantic extraction, which may
unwrap one complete Markdown fence and accepts exactly one full-line
four-integer candidate. Pair categories are mutually exclusive and use the
fixed order: format failure, corrected, both correct, ignored, over-edited,
wrong single edit, other. No subjective pass threshold or hypothesis test is
performed.

The existing `duplicate_encoding` generator supplies a trusted `known_value`
for all four positions. Its result is therefore a full-state restatement and
instruction-following control, not a nontrivial recovery estimate, and it is
not pooled into a primary recovery rate. `cross_series` and `trend` are reported
as separate nontrivial diagnostics. With two cases per family, every pattern is
descriptive only and cannot establish stable model capability, authorize a
larger rerun, or authorize RL or training.

The controlled CVA dataset remains the primary mechanistic dataset because it
supports registered redundancy constraints, cue ablations, legal coherent
counterfactuals, and exact deterministic replay. Public benchmarks do not
replace that role. A later external-validity section may evaluate on the
human-authored subset of [ChartQA](https://github.com/vis-nlp/ChartQA) or on
[ChartQAPro](https://github.com/vis-nlp/ChartQAPro), but those results must be
reported as descriptive generalization rather than as evidence for the
controlled causal estimands.
