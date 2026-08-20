# Qwen v4 Phase 0-5 server handoff

This handoff executes the authorized Phase 0-3 diagnostics, the separately guarded Phase 4
language-only LoRA support injection, and inference-only Phase 5 policy-support measurement. It
stops before all RL. The
archived v4 plan is byte-identical to the supplied specification (SHA-256
`01e532f8c7fe5439c70e8cd8de81ff3448465d8d401dbbf478c76aebaf49641e`). Phase 4 SFT/LoRA is
authorized only by `configs/recoverability/v4_phase_4.yaml`. Phase 5 is measurement-only under
`configs/recoverability/v4_phase_5.yaml`; training and RL are unauthorized in that phase.

## Fixed boundaries

- Project: `/cloud/cloud-ssd1/dissertation`
- Model: `/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct`
- Model snapshot SHA-256:
  `e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87`
- Fixed image size: 280 x 280 pixels (both multiples of 28).
- Existing screen records SHA-256:
  `f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a`.
- The absent hundred-call no-cue/valid-cue raw summary remains
  `awaiting_hash_bound_server_evidence`; do not reconstruct it from prose.

Run inside `tmux`. Stop at the first `BLOCKED` message and return the complete output. Every
script except S0 is inert without `--execute`. S3 is a real no-overwrite, hash-bound
teacher-forced scoring run, S4 is the real layerwise diagnostic, and S5 is the real exact-cache
parity run. S6 is the real hash-bound I0--I4 interface-ladder execution over the frozen S5
artifact. No Phase 0-3 script trains or invokes RL. Phase 4 has its own input, package-lock,
trainability, CUDA/bf16, dependency, acknowledgement, and no-overwrite gates.

```bash
cd /cloud/cloud-ssd1/dissertation
source .venv/bin/activate
export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
git rev-parse HEAD
```

## S0 - frozen legacy audit

```bash
python scripts/v4/00_audit_legacy.py --artifact-root artifacts
sha256sum artifacts/v4/audit/*
```

Exactly four files must exist: the experiment registry, claim-evidence matrix, scoring contract,
and legacy hash manifest. No model is loaded.

## S1 - runtime model introspection

```bash
python scripts/v4/01_introspect_qwen.py --execute
sha256sum artifacts/v4/model_introspection.json artifacts/v4/module_manifest.txt
```

Layer and module counts come from the live snapshot; they are never hard-coded.

## S2 - T1-T6 capability-chain execution

S2 is the actual hard-text Phase 1 runner. It requires the exact August 17, 2026 S1 runtime
evidence before loading Qwen:

- `artifacts/v4/model_introspection.json`
  SHA-256 `ed96d19a238d68497617071e29604313e0aae9a41a9e3bd24dbad451d87a0640`
- `artifacts/v4/module_manifest.txt`
  SHA-256 `1c98fd8ba74fa5c30b8f585ffee5020544baf5be61f23e0c28c61a132973e8f0`
- `model_class = Qwen2_5_VLForConditionalGeneration`
- `language_layers = 36`
- `vision depth = 32`
- `module_count = 839`
- required modules:
  `model.visual.blocks.0`, `model.visual.blocks.31`, `model.visual.merger`,
  `model.language_model.layers.0`, `model.language_model.layers.35`,
  `model.language_model.norm`, `lm_head`

The legacy file contains 580 answer-recoverable scenes, but the frozen v4 audit found that one
`trend` scene admits two fact-supported one-edit worlds. V4 therefore excludes that scene by the
exact-world uniqueness equation before any model call and executes 579 scenes x 6 calls = 3,474
model calls. Included family counts are 208 `cross_series`, 182 `duplicate_encoding`, and 189
`trend`; the excluded-family count is exactly one `trend`. T1 is allocated 290 YES / 289 NO and
the four T5 true-label slots are allocated 145 / 145 / 145 / 144. These are deterministic design
allocations, not empirical success thresholds.

Gates remain objective only: hash-bound screen records, S1 runtime evidence, one eligible natural
error per source scene, exact-world uniqueness, four unique T5 candidates with single-token
labels, no overwrite, and strict minimal-output parsing. The excluded scene and both supported
worlds are recorded in `paired_gaps.json`.

The frozen integer domain `2..18` constrains latent truth and candidate worlds, not the noisy
observation itself. In `o = x* + delta e_j`, `delta` is an exact integer and is not restricted to
`+/-1`; the observation may therefore be arbitrarily far outside the render domain at its single
error coordinate. Its one-edit candidates are still enumerated only inside `2..18`.

```bash
SCREEN=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2/phase_c_screen/screen_records.jsonl
python scripts/v4/02_run_capability_chain.py --execute \
  --input "$SCREEN" \
  --input-sha256 f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a
sha256sum artifacts/v4/capability_chain/per_scene.csv \
  artifacts/v4/capability_chain/summary_by_family.csv \
  artifacts/v4/capability_chain/paired_gaps.json
```

The completed S2 evidence is frozen exactly as follows:

- `per_scene.csv`:
  `d01c391e136ed0e5c0ed52e50fe70f6ec128d221d218e7012c9adbcb4293929f`
- `summary_by_family.csv`:
  `8837a7275915f5a90c91eae8378ff6bd0466819842381996db1be8e4925705c7`
- `paired_gaps.json`:
  `a2a5acb5b203719e7e5225e56643a25a4189277d13704cccd4ec86d61573e256`

These are evidence inputs, not success criteria. Their accuracy values are reported but never
used to permit or block S3.

## S3 - teacher-forced candidate scoring

S3 executes 579 scenes x 4 paired cue conditions = 2,316 standard forward passes. It never calls
generation. It verifies four distinct single-token labels, four unique candidate worlds,
balanced true-label slots (145 / 145 / 145 / 144), exact no-cue/valid-cue payload parity except
for `facts`, and structural agreement between all three S2 artifacts and the selected scene set.
It reports point estimates and scene-clustered confidence intervals without an empirical
pass/fail threshold.

```bash
python scripts/v4/03_score_candidates.py --execute \
  --input "$SCREEN" \
  --input-sha256 f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a \
  --input artifacts/v4/capability_chain/per_scene.csv \
  --input-sha256 d01c391e136ed0e5c0ed52e50fe70f6ec128d221d218e7012c9adbcb4293929f \
  --input artifacts/v4/capability_chain/summary_by_family.csv \
  --input-sha256 8837a7275915f5a90c91eae8378ff6bd0466819842381996db1be8e4925705c7 \
  --input artifacts/v4/capability_chain/paired_gaps.json \
  --input-sha256 a2a5acb5b203719e7e5225e56643a25a4189277d13704cccd4ec86d61573e256
sha256sum artifacts/v4/tokenizer/candidate_labels.json \
  artifacts/v4/candidate_scoring/per_scene.jsonl \
  artifacts/v4/candidate_scoring/summary.json
```

## S4 - layerwise assimilation surface

S3 completed on 579 scenes and 2,316 teacher-forced forwards. Its frozen evidence is:

- `candidate_labels.json`:
  `a7a448f230038698c4127b220362c95d47f57cef90cc7904e71b1dacacc04dbd`
- `candidate_scoring/per_scene.jsonl`:
  `c303731438760a30fb2f78a489d20465a1c1e292b01941795f29351a0e234a62`
- `candidate_scoring/summary.json`:
  `5e366dfecb2a4fd530407f896326bc99d3605385cfdadea34c0f478392280c73`
- labels and token IDs: `A=32`, `B=33`, `C=34`, `D=35`

The S3 valid-minus-no-cue point estimate was positive, but the sham-minus-no-cue estimate was
similar and slightly larger. This is reported evidence, not a stop rule. S4 therefore measures
all four conditions and includes valid-minus-sham and counterfactual-minus-sham layerwise paired
contrasts; no effect magnitude determines whether S4 executes.

S4 executes 579 scenes x 4 conditions = 2,316 hidden-state forwards. Every forward must expose
exactly 36 language-layer states, and its projected final-layer candidate logits must match the
same standard forward. Any parity mismatch is an objective measurement-validity failure.

```bash
python scripts/v4/04_layerwise_assimilation.py --execute \
  --input "$SCREEN" \
  --input-sha256 f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a \
  --input artifacts/v4/capability_chain/per_scene.csv \
  --input-sha256 d01c391e136ed0e5c0ed52e50fe70f6ec128d221d218e7012c9adbcb4293929f \
  --input artifacts/v4/capability_chain/summary_by_family.csv \
  --input-sha256 8837a7275915f5a90c91eae8378ff6bd0466819842381996db1be8e4925705c7 \
  --input artifacts/v4/capability_chain/paired_gaps.json \
  --input-sha256 a2a5acb5b203719e7e5225e56643a25a4189277d13704cccd4ec86d61573e256 \
  --input artifacts/v4/tokenizer/candidate_labels.json \
  --input-sha256 a7a448f230038698c4127b220362c95d47f57cef90cc7904e71b1dacacc04dbd \
  --input artifacts/v4/candidate_scoring/per_scene.jsonl \
  --input-sha256 c303731438760a30fb2f78a489d20465a1c1e292b01941795f29351a0e234a62 \
  --input artifacts/v4/candidate_scoring/summary.json \
  --input-sha256 5e366dfecb2a4fd530407f896326bc99d3605385cfdadea34c0f478392280c73
sha256sum artifacts/v4/layerwise_assimilation/per_scene.jsonl \
  artifacts/v4/layerwise_assimilation/summary.json
```

## S5 - exact-cache parity execution

S4 completed on 579 scenes and 2,316 layerwise forwards. Its frozen evidence is:

- `layerwise_assimilation/per_scene.jsonl`:
  `e696d12bb8cb3e6142a3d6ecc6de9474c3e72e3ac85e0c7334005a249556a4af`
- `layerwise_assimilation/summary.json`:
  `53eab07dcd70fce6970a63ce1831ec6369164e92320f051e717decdeb1b790c0`

S5 regenerates one immutable natural visual observation/cache for each of those 579 scenes, then
compares cached continuation with full-history re-encoding under all four cue conditions. Each
of the 2,316 paired calls uses deterministic greedy decoding.
Generated-token equality remains an exact per-call I4 primary gate. When one call diverges, S5
freezes its first-mismatch token and logit evidence, excludes that call from the I4 primary set,
and continues the remaining S5 calls. No token-divergence count or rate threshold is applied.
Suffix/MRoPE and cache-position structure remain fail-closed execution gates.
Full-vocabulary logit identity is reported, not required:
each step records both dtypes and shapes, the realized-token logits, argmax IDs, maximum absolute
and relative difference, nonzero count, L2 difference, and the token with maximum absolute
difference. No logit-drift magnitude threshold is applied. These are measurement-validity
checks and disclosures, not empirical-effect thresholds.

Before model loading, S5 also verifies the complete 8,000-scene frozen Phase C visual source:

- `manifest.json` SHA-256:
  `bc57389dc3164b6aeba8d4565aecfaea3fa7ba171b4df4843c8ec86cbee8a19f`
- `records.jsonl` SHA-256:
  `36e09f7e15107057fd1b942875d12259b1f281e0354b87c82ed17f420693c766`
- the manifest-bound hash of all 8,000 PNG files.

```bash
python scripts/v4/05_validate_cache_runner.py --execute \
  --input artifacts/v4/layerwise_assimilation/per_scene.jsonl \
  --input-sha256 e696d12bb8cb3e6142a3d6ecc6de9474c3e72e3ac85e0c7334005a249556a4af \
  --input artifacts/v4/layerwise_assimilation/summary.json \
  --input-sha256 53eab07dcd70fce6970a63ce1831ec6369164e92320f051e717decdeb1b790c0
sha256sum artifacts/v4/cache/cache_parity.json
python -m json.tool artifacts/v4/cache/cache_parity.json
```

Only calls with exact generated-token and greedy-decision parity can enter an I4 primary result.
A call-level generated-token divergence is preserved as diagnostic-only evidence and does not stop
the remaining calls. Invalid tensor shapes, non-finite logits, realized tokens that are not their
path's top-1 decision, suffix/MRoPE drift, or cache-position drift remain structural failures that
block S5. Full-vocabulary floating-point differences are preserved in the artifact as diagnostic
evidence. The artifact certifies or diagnoses the measurement interface only—it does not claim
that visual revision succeeded.

## S6 - I0-I4 interface-ladder surface

S5 completed all 2,316 cache/full-history comparisons. Its frozen artifact is:

- `cache/cache_parity.json`:
  `c52cb71d42c83e3a32c57c00e006f5117631b9cab25a2ef8fbe62001ff572351`

S6 produces exactly 17 immutable cells per scene (9,843 total): I0, I2, I3, and I4 under all
four cue conditions, plus one pre-cue/no-cue I1 soft-report diagnostic. I0 is measured by fresh
text-only runtime calls; I1 records top-k relative logits at each of the four generated numeric
positions; I2 consumes the frozen S3 candidate decision; I3 decodes S5 full-history output; and
I4 decodes S5 cached-continuation output. I1 and I2 are intervention diagnostics only.

If any I4 call for a scene has token divergence, every cell is retained but that entire scene is
excluded from the complete-case I0/I3/I4 paired primary estimands. The number or rate of such
scenes is reported and is never used as an empirical execution threshold.

```bash
python scripts/v4/06_run_interface_ladder.py --execute \
  --input "$SCREEN" \
  --input-sha256 f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a \
  --input artifacts/v4/capability_chain/per_scene.csv \
  --input-sha256 d01c391e136ed0e5c0ed52e50fe70f6ec128d221d218e7012c9adbcb4293929f \
  --input artifacts/v4/capability_chain/summary_by_family.csv \
  --input-sha256 8837a7275915f5a90c91eae8378ff6bd0466819842381996db1be8e4925705c7 \
  --input artifacts/v4/capability_chain/paired_gaps.json \
  --input-sha256 a2a5acb5b203719e7e5225e56643a25a4189277d13704cccd4ec86d61573e256 \
  --input artifacts/v4/tokenizer/candidate_labels.json \
  --input-sha256 a7a448f230038698c4127b220362c95d47f57cef90cc7904e71b1dacacc04dbd \
  --input artifacts/v4/candidate_scoring/per_scene.jsonl \
  --input-sha256 c303731438760a30fb2f78a489d20465a1c1e292b01941795f29351a0e234a62 \
  --input artifacts/v4/candidate_scoring/summary.json \
  --input-sha256 5e366dfecb2a4fd530407f896326bc99d3605385cfdadea34c0f478392280c73 \
  --input artifacts/v4/layerwise_assimilation/per_scene.jsonl \
  --input-sha256 e696d12bb8cb3e6142a3d6ecc6de9474c3e72e3ac85e0c7334005a249556a4af \
  --input artifacts/v4/layerwise_assimilation/summary.json \
  --input-sha256 53eab07dcd70fce6970a63ce1831ec6369164e92320f051e717decdeb1b790c0 \
  --input artifacts/v4/cache/cache_parity.json \
  --input-sha256 c52cb71d42c83e3a32c57c00e006f5117631b9cab25a2ef8fbe62001ff572351
sha256sum artifacts/v4/interface_ladder/per_scene.jsonl \
  artifacts/v4/interface_ladder/summary.json
python -m json.tool artifacts/v4/interface_ladder/summary.json
```

## Reporting and stopping discipline

There is no subjective empirical success gate: do not require an 80% visual repair rate, a
minimum recovery accuracy, or any other result-dependent threshold. Hash agreement, unique
candidates, split isolation, S1 runtime-evidence agreement, single-token labels, forward-logit
parity, and cache parity are objective measurement-validity gates and may block execution.

Return point estimates, scene-clustered 95% confidence intervals, paired effects, and
family-stratified effects. When policy support is later measured, report every prompt's `p_i`,
`G_K = 1 - p_i^K - (1-p_i)^K`, and actual within-group reward variance; until then record these
as not measured, never as zero. The engineering requirement of at least 80% automated-test
coverage is software QA only, not a scientific result threshold.

## Phase 4 - language-only LoRA support injection

Phase 4 consumes three hash-bound JSONL inputs:

- symbolic support scenes with split `symbolic_support_train`;
- natural support scenes with split `natural_error_support_train`;
- one frozen-base Stage-1 natural observation per natural support scene, with exactly one error.

The source-preparation command needs no user-supplied paths. It generates 579 deterministic
`symbolic_support_train` scenes with seed `2026081804`, reads the completed S6 17-cell artifact,
reconstructs the 579 frozen I1 Stage-1 observations against the hash-pinned Phase-C visual dataset,
and retains every observation with exactly one in-domain position error. S6 is legacy diagnostic
evidence and is not a confirm split. Malformed, zero-error, multiple-error, and out-of-domain I1
outputs remain in `selection_trace.jsonl` and are not training examples. The command fails if S6
closure, model provenance, visual-dataset hashes, or natural single-error support are absent.

The builder rejects confirm splits, unpaired observations, duplicated provenance, and scenes with
anything other than the declared single-position natural error. The training command checks the
support corpus hash and its summary, package lock, offline environment, GPU dependency
installation, CUDA, bf16 capability, model snapshot, exact `named_modules()` LoRA targets, frozen
base hashes, trainable parameters, and output non-overwrite before constructing an optimizer step.
It trains C0, C1, and T in order.

Execute the following commands from the repository root. No path substitution is required. The
run directory must not exist; the parameter manifests must not already exist under
`artifacts/v4/training`.

```bash
python scripts/v4/07_prepare_phase4_support_sources.py --execute
python scripts/v4/07_build_support_data.py --execute --prepared-sources

sha256sum \
  artifacts/v4/training/sources/symbolic_scenes.jsonl \
  artifacts/v4/training/sources/natural_scenes.jsonl \
  artifacts/v4/training/sources/natural_observations.jsonl \
  artifacts/v4/training/sources/selection_trace.jsonl \
  artifacts/v4/training/sources/source_summary.json \
  artifacts/v4/training/support.jsonl \
  artifacts/v4/training/support_summary.json

python scripts/v4/08_train_phase4_lora.py \
  --execute --preflight-only --prepared-support

export COMPBIAS_V4_TRAINING_ACK=I_UNDERSTAND_THIS_STARTS_PHASE_4_LORA_TRAINING
python scripts/v4/08_train_phase4_lora.py --execute --prepared-support
```

The required output evidence is:

```text
artifacts/v4/training/trainable_parameter_manifest.json
artifacts/v4/training/frozen_hashes.json
artifacts/v4/training/runs/phase4-r1/C0_format_only/final_adapter
artifacts/v4/training/runs/phase4-r1/C1_forward_arithmetic/final_adapter
artifacts/v4/training/runs/phase4-r1/T_constraint_recovery/final_adapter
```

Return S0-S6 and Phase 4 console output, the Git revision, every produced SHA-256 line, and any
`BLOCKED` message.

## Phase 5 - held-out policy-support measurement

Phase 5 first freezes a new `support_dev` intake of 576 scenes from the pinned 8,000-scene visual
dataset. All 579 scenes used by the Phase 4 S6 source-selection trace are excluded before model
execution. The intake is balanced at 192 scenes for each of `cross_series`,
`duplicate_encoding`, and `trend`. The frozen Base model receives exactly one Stage-1 visual call
per scene, with no retry and no sample extension. Correct reads, invalid formats, out-of-domain
values, and multiple-error reads remain in `selection_trace.jsonl`; every in-domain, exactly-one
position error enters the held-out natural-error pool. No empirical success threshold is used.

The second command evaluates the same held-out pool in the fixed order Base, C0, C1, T. Each
checkpoint receives one greedy T6 completion, teacher-forced sequence log probabilities for the
truth and frozen observation, and 16 temperature-0.7 rollouts per scene. The common rollout seed
depends on scene and rollout index, not checkpoint. The registered K values are 1, 2, 4, 8, and
16; the Phase 6 informative-group size is 8. The runtime loads a fresh Base model for every
checkpoint, attaches at most one frozen adapter, disables gradients, and never constructs an
optimizer or Trainer.

Completed checkpoint evidence is written under `artifacts/v4/support_work/phase5-r1`. A matching
rerun resumes those checkpoint files after verifying checkpoint, support-dev, and config hashes;
therefore a later failure does not require repeating completed checkpoint inference. Formal
outputs are not overwritten.

Run the two commands in order:

```bash
python scripts/v4/09_prepare_phase5_support_dev.py --execute

sha256sum \
  artifacts/v4/support_dev/candidates.jsonl \
  artifacts/v4/support_dev/held_out_natural_errors.jsonl \
  artifacts/v4/support_dev/selection_trace.jsonl \
  artifacts/v4/support_dev/summary.json
python -m json.tool artifacts/v4/support_dev/summary.json

python scripts/v4/10_measure_policy_support.py --execute

sha256sum \
  artifacts/v4/support/policy_support_by_scene.parquet \
  artifacts/v4/support/informative_group_rate.json \
  artifacts/v4/support/pass_at_k.csv
python -m json.tool artifacts/v4/support/informative_group_rate.json
```

The formal Phase 5 outputs are:

```text
artifacts/v4/support/policy_support_by_scene.parquet
artifacts/v4/support/informative_group_rate.json
artifacts/v4/support/pass_at_k.csv
```

Return both `READY` lines, all printed SHA-256 lines, the support-dev summary, and any `BLOCKED`
message. Do not begin Phase 6 or any RL stage.
