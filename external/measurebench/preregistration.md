# MeasureBench Synthetic external-validation preregistration

Frozen: 2026-08-14 (Asia/Singapore, UTC+08), before repository retrieval or
model evaluation.

Pinned upstream commit:
`d5bb8652dbde6b1b5507f89d37f73993af28b830` at
https://github.com/flageval-baai/MeasureBench.

Execution status: **NOT RUN**. This document fixes the intended selection and
shift rules. It does not assert that the upstream generator exposes every label
or factor name below; exact API mappings must be recorded after checking the
pinned checkout and before viewing model results.

## Purpose

Test whether at least one controlled-world finding survives on an independent
measurement generator:

1. pre-RL, prompt-conditional severity-compensability covariance predicts the
   direction of post-RL perception shift;
2. final answer gain separates into perception, reasoning, and coupling terms
   when an executable scalar measurement state exists;
3. a compensatory solution is more fragile when the perception-error mechanism
   changes but the measurement rule and answer stay fixed.

Failure on this benchmark must narrow the paper to a controlled-mechanism result.

## Source and immutability rules

- Use only the official synthetic generator at the pinned commit for the first
  external stage.
- Do not change upstream evaluation semantics, tolerance, answer parsing, or
  scoring.
- A local adapter may copy the upstream sample ID and save generator parameters,
  executable numeric state, canonical answer, and content hashes.
- Preserve the upstream license and attribution. Do not redistribute generated
  or source assets until license review is recorded in `pin_manifest.json`.
- Record the checked-out Git tree, dirty state, exact command, environment,
  generator config, source license, evaluator hash, and output manifest.

## Preregistered primary subset

The primary semantic families, in fixed priority order, are:

1. clock;
2. ammeter;
3. gauge or the closest upstream general dial/gauge family.

The exact upstream label mapping is deliberately unresolved until the pinned API
is inspected. Mapping is eligible only if all of these criteria hold:

- one scalar numeric measurement is encoded in the synthetic scene;
- the official generator exposes enough parameters to reconstruct that state;
- the official evaluator computes a deterministic target/tolerance;
- the task supports controlled rerendering or state intervention without
  changing the semantic measurement rule;
- sample identity and every generated asset can be hashed.

If an intended family is absent, do not substitute a family after looking at
model accuracy. Mark it unavailable. Proceed with the remaining preregistered
families only if at least two are eligible; otherwise external validation is a
no-go.

## Fixed exclusions

- real photographs during the initial stage;
- samples lacking reconstructable numeric ground truth;
- tasks requiring subjective text grading;
- categorical tasks without a task-semantic severity measure;
- corrupted or parser-failed records silently dropped from denominators;
- families selected because preliminary model results look favorable.

Every exclusion receives an upstream sample ID and controlled reason code.

## Split policy

Create train, calibration, validation, IID test, and shifted test splits using a
local deterministic hash of the upstream sample/generator identity. Freeze the
hash salt and proportions in the generated manifest before any RL run.

- Calibration alone estimates compensability, chooses no final checkpoint, and
  cannot overlap test generator states.
- Validation may select optimization checkpoints and beta.
- IID and shifted test results remain sealed until the run-selection policy is
  frozen.
- Underlying numeric state, generator template, and paired variants stay in one
  split group to prevent leakage.

## Interventional error catalog

For each eligible scalar state `v`, create an executable, unit-aware catalog:

- `truth`: `z=v`, severity `0`;
- `numeric_offset:+2u`: `z=v+2u`, severity `2`;
- `numeric_offset:-2u`: `z=v-2u`, severity `2`;
- `numeric_offset:+1tick`: one valid instrument tick above truth;
- `numeric_offset:-1tick`: one valid instrument tick below truth.

Here `u` is the official numeric unit and tick offsets use the upstream scale.
Out-of-range transformations are retained as explicit ineligible interventions,
not silently clipped. A valid run must place the reasoner in a separately
isolated text-only worker whose reviewed adapter receives only injected state
and question; natural and interventional views are stored separately. Merely
omitting an explicit image argument does not establish that the original image
is unavailable through closures, object state, caches, or shared-process state.
This isolation gate has not yet been implemented or verified.

## Preregistered shifts

### Primary: signed error-mechanism reversal

- Calibration/training mechanism: `numeric_offset:+2u`.
- Shifted mechanism: `numeric_offset:-2u`.
- Held constant: upstream sample ID, true state, unit, question, task rule,
  canonical answer, model checkpoint, decoding settings, and rollout seeds.
- Primary statistic: compensation generalization gap
  `c_train(e)-c_shift(e)` and paired answer-accuracy difference.

This is a controlled perceived-state shift. It does not by itself show that an
image model naturally makes either signed error.

### Secondary: deterministic offset permutation

Permute nontruth signed offsets with a hash fixed from the generator sample ID.
The mapping is independent of reward and model output. Hold the truth state and
answer distribution fixed. Report it as a sensitivity analysis, not a second
independent benchmark.

### Renderer shift, conditional on upstream support

After API audit, register exactly one official rendering factor among font,
lighting, clutter, tick style/density, or pointer style. Selection follows this
priority order and uses the first factor that the pinned generator exposes as an
independent control. It must preserve the latent measurement, question, rule,
and answer. If independence cannot be established, omit this shift rather than
creating an ad hoc one.

## Models and seeds

- Primary model: `Qwen/Qwen2.5-VL-3B-Instruct` at revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- VLM seeds: `11`, `17`, `23`; expand to `29`, `31` only as a preregistered
  robustness extension, never because the first three are unfavorable.
- No 7B expansion until every 3B internal and external gate passes.
- Compare base, structured-SFT, fixed-reasoner outcome-RL, joint outcome-RL, and
  oracle process/perception control only when all corresponding checkpoints have
  complete provenance.

## Metrics and statistical protocol

Primary metrics:

- official MeasureBench score, unchanged;
- state exact match and task-semantic measurement error;
- prompt-conditional `c(e|x)` and `Cov(ell,c)`;
- predicted versus observed perception shift and pairwise odds residual;
- `L_P`, `L_R`, `2C`, and `L_O` where the executable scalar decomposition holds;
- IID/shifted compensation gap.

Use paired bootstrap 95% confidence intervals, 10,000 resamples, and Holm
correction for the fixed family comparisons. Report means, standard deviations,
raw components, all seeds, parser failures, and exclusions. Do not report only
a normalized coupling ratio or final accuracy.

## Success, null, and stop rules

The external mechanism is supported only if a preregistered family shows a
directionally correct pre-RL covariance prediction and a reproducible post-RL
effect without evaluator changes. Coupling is externally supported only if its
paired interval excludes zero and raw decomposition checks close numerically.

Stop and report a null/limited result if:

- fewer than two intended families meet eligibility;
- the official numeric state cannot be reconstructed;
- the paired shift changes the task rule or answer distribution;
- original-image leakage occurs during state injection;
- parser validity is below 98%;
- hashes/provenance are incomplete;
- the covariance does not predict post-RL direction;
- coupling is absent or does not generalize outside CVA-World.

No outcome permits the wording "first discovery of error cancellation". Before
execution, replace every unresolved upstream API label with a pinned-code path
and line/commit reference, then hash the finalized protocol without changing
the semantic selection rules above.
