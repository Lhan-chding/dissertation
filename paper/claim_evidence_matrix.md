# Claim--evidence matrix

Last updated: 2026-08-14 (Asia/Singapore, UTC+08).

Use the narrowest row supported by the registry. "Implemented" means code
exists; "verified" requires a current accepted execution record; "VLM" and
"external" require actual runs and hashed artifacts. Absence of evidence is not
neutral evidence for the hypothesis.

## Permitted claims

| Claim ID | Candidate statement | Required evidence | Current evidence class | Current wording allowed |
|---|---|---|---|---|
| C-SEL-ID | The finite KL-regularized distribution optimum reweights perception-error marginals by their reward-moment multipliers. | Analytical proof plus exact numerical identity checks | Analytical statement plus accepted `VERIFIED_CPU` Phase-A/B identity records | State as a conditional distribution-space derivation with an accepted finite numerical check, not an originality claim or VLM behavior. |
| C-SEL-SIGN | Under binary reward and the stated fixed-reasoner assumptions, the prompt-conditional severity shift has the covariance sign. | Proof, exact fixtures, prompt-conditional tests | Analytical statement plus accepted `VERIFIED_CPU` Phase-A/B checks | State as a conditional theorem/corollary with a finite CPU check, not observed VLM behavior. |
| C-ODDS | Pairwise error odds at the exact optimum change by the multiplier ratio. | Proof and exact numerical calibration | Analytical statement plus accepted `VERIFIED_CPU` exact numerical calibration | State only for the exact distribution-space optimum; cite the current registry-bound artifact rather than superseded residuals. |
| C-APPROX | Finite-policy error-only updates can be compared with the registered selection target while the reasoner conditional remains fixed. | Formal REINFORCE/PPO-like/GRPO-like error policy with independent `r~Bernoulli(c[e])`, marginal KL, full error-action coverage, and aggregate conditional-rate error <=0.03; separate unconstrained-joint and collapsed diagnostics | `VERIFIED_CPU` for the registered 20-seed-per-algorithm finite tabular protocol | Report `raw_fixed_reasoner_outcome` as the formal finite tabular analogue. Report the unconstrained-joint and collapsed views separately as diagnostics only. None establishes neural PPO/GRPO fidelity or guarantees that real outcome RL follows the exact theorem. |
| C-SCALE | The direction under reasoner scaling is governed by relative compensability gain across error severity, not overall accuracy alone. | Proof; three equal-average-gain tabular paths with covariance/finite-difference checks; matched-checkpoint experiment | Analytical identity plus `VERIFIED_CPU` finite tabular Exp-C; real matched-checkpoint experiment pending | State the conditional identity and the accepted three-path finite diagnostic. Real-checkpoint/VLM direction remains H3. |
| C-LOCK | Repeated exact selection in a fixed landscape concentrates on a unique maximum multiplier. | Closed-form proof and numerical recurrence checks | Analytical statement plus accepted `VERIFIED_CPU` recurrence checks | Restrict to the fixed finite landscape; never call this GRPO convergence. |
| C-BISTABLE | The registered 2x2 coordination model has truthful and compensatory stable corners with an interior saddle. | Stability proof, ODE/basin checks | Analytical statement plus accepted `VERIFIED_CPU` coordination record; no VLM | State as a property of the 2x2 model only; do not extrapolate its basins to Qwen. |
| C-BIFURC | Symmetric KL regularization has critical `beta_c=a/2` in the registered coordination model. | Stationarity/stability proof and root checks | Analytical statement plus accepted `VERIFIED_CPU` root checks | State only for the symmetric reduction and declared reference policy. |
| C-DECOMP | Additive outcome error equals perception error plus reasoning error, and squared loss includes a coupling term. | Exact algebra plus per-sample/aggregate checks | Analytical identity plus accepted `VERIFIED_CPU` numerical checks; no VLM decomposition run | State the identity for tasks with executable canonical intermediate answers; do not describe it as an observed real-VLM effect. |
| C-BREG | A registered Bregman loss admits the three-point interaction decomposition. | Exact identity and strictly convex test potentials | Classical analytical identity plus accepted `VERIFIED_CPU` randomized checks | Call it a classical identity used as a diagnostic, not a new theorem. |
| C-NEURAL | A small differentiable CPU diagnostic can reach truthful or compensatory endpoints and expose a registered OOD gap. | Repeated accepted runs with config/artifact hash agreement | `VERIFIED_CPU`: 50 clean condition--seed runs under the current scalar YAML | Restrict the claim to the two-sigmoid coordination reduction and registered synthetic error permutation; it does not train the PIL/CNN path or establish VLM behavior. |
| C-VIS | A convolutional perceiver can participate in the controlled mechanism on rendered input. | Repeated accepted image-path runs, parameter-delta audit, paired shift, full provenance, and config/artifact hash agreement | `VERIFIED_CPU`: accepted 10-seed controlled image-path run under the current YAML | Support only a single-scene mechanism check with byte-identical paired images, not multi-image CNN generalization, natural visual-error selection, or VLM evidence. |
| C-CVA | CVA-World provides five task families with executable state/error contracts. | Generated manifest, solver/round-trip/split audit, frozen bar-operation coverage, human review of >=200 unique sample IDs and all 73 contact sheets, family-applicable fully crossed visual catalog, and run provenance | `PARTIAL_GATE`: clean local v2 generation and schema-v2 automatic audit recorded 1,820 samples/images, 1,820 solver checks, 4,020 error-solver/round-trip checks, 73 sheets, and 100 OOD pairs; human review/sign-off absent | Report the automatic local audit and its hashes only with `phase_d_ready: false`. Do not carry forward v1, call D `VERIFIED_CPU`, treat ignored/local outputs as an approved release, or substitute agent review for the mandatory human sign-off. |
| C-VLM-H1 | Pre-RL compensability predicts post-RL perception direction in Qwen2.5-VL. | >=3 seeds, preregistered families, calibration-only estimate, significance test | NOT RUN | No affirmative wording. Label as H1. |
| C-VLM-COUP | Real-VLM answer gain contains a repeatable coupling contribution. | Paired checkpoint decomposition with bootstrap CI | NOT RUN | No affirmative wording. Label as H5. |
| C-VLM-BASIN | Joint Qwen training has truthful and compensatory seed attractors. | >=3 seeds, preferably 5, with matched high outcome reward and divergent local metrics | NOT RUN | No affirmative wording. Label as H4. |
| C-VLM-OOD | Coupling contribution predicts OOD error-mechanism fragility. | preregistered paired shift and across-run association | NOT RUN | No affirmative wording. Label as H6. |
| C-EXT | At least one core mechanism replicates on MeasureBench Synthetic. | pinned external checkout, unchanged evaluator, hashed synthetic subset and results | NOT RUN | Say only that validation is preregistered at the pinned commit. |

## Evidence promotion rules

1. A passing unit test may contribute to an implementation identity, but
   `VERIFIED_CPU` additionally requires the registry's accepted canonical run
   and provenance bundle; neither can promote a VLM or causal empirical claim.
2. A generated config or preflight plan does not promote any experiment.
3. Natural conditional success `c_sel` is the exact trajectory-selection input.
   Causal downstream compensability requires image-cut fork replay of the exact
   natural mediator (`c_fork`); synthetic injection (`c_syn`) cannot replace
   either quantity without a transport audit.
4. The CPU interface assertion, even if it passes, shows only that no image is passed as an
   explicit reasoner argument. It does not exclude access through closures,
   members, caches, or shared process state; a separately isolated text-only
   worker and reviewed adapter hash remain pending Phase-E gates.
5. Pooled covariance cannot replace prompt-conditional estimation.
6. Test-set results cannot select beta, checkpoint, error profile, or shift.
7. A single seed cannot establish a stable basin, repeated direction, or
   real-VLM coupling contribution.
8. A positive IID result without a held-constant-task-rule error-mechanism shift
   cannot distinguish structural repair from spurious compensation.
9. Failure to reproduce externally must narrow scope to the controlled setting.

## Permanently disallowed novelty claims

These claims remain disallowed even if experiments pass:

- first discovery of error cancellation or right-answer/wrong-reason behavior;
- first perception/reasoning decomposition;
- first recognition that outcome reward underdetermines local credit;
- invention of exponential tilting, covariance selection, Bregman
  decomposition, replicator dynamics, maximal coupling, PPO, or GRPO;
- a new perception reward, process reward, token-level advantage method,
  architecture, router, or auxiliary head;
- strong reasoners necessarily damage perception;
- explicit chain of thought is the model's true internal causal computation;
- the additive decomposition applies to every multimodal task.

## Required transparent negative statements

Until promoted by the registry, manuscripts and reports must explicitly say:

- no large-GPU Qwen SFT/RL start is recorded in the project registry;
- the VLM configs are safety-gated execution plans, not results;
- any future emitted VLM plan remains private, `not_started`, and
  `execution_permitted: false`; local snapshot hashes establish
  self-consistency, not trusted-upstream authenticity;
- no causal image-hidden gate has passed: the isolated text-only worker and
  reviewed adapter hash are not yet implemented or verified;
- the compensability YAML/CLI specifies only the interventional analysis view;
  a matched natural-view schema/producer and both GPU rollout sets are pending;
- MeasureBench is pinned and preregistered but not downloaded/executed;
- CPU tests validate implementation behavior, not universal VLM behavior or
  mathematical originality;
- `requirements-lock.txt` is an unhashed, platform-specific intended dependency
  snapshot, not a verified clean-environment or cryptographic lock; the current
  `.venv` lacks its pinned `wheel` package and has an unrecorded bootstrap `pip`
  version;
- upstream veRL packaging is internally inconsistent: its stale `.[vllm]` extra
  must not be used. The hash-pinned upstream Dockerfile is a dependency
  reference, not a reproducible build: a vendored hardened descendant, final
  image digest, offline hashed inputs, SBOM, vulnerability-policy result, and
  target-GPU smoke are all still missing. The current schema-v2 audit cannot
  record those requirements as completed; emitted plans hard-code them pending,
  so a future authenticated gate extension is required before execution.
- planning-time GPU selection is restricted to UUIDs shared by the reviewed
  target-container smoke and local `nvidia-smi`; the eventual executor has not
  been bound to those UUIDs, so this remains a separate pending gate.
