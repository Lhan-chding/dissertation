# Qwen v4 Phase 0-3 server handoff

This handoff executes the authorized Phase 0-3 diagnostics and stops before training or RL. The
archived v4 plan is byte-identical to the supplied specification (SHA-256
`01e532f8c7fe5439c70e8cd8de81ff3448465d8d401dbbf478c76aebaf49641e`). Training, LoRA,
SFT, and RL are not authorized.

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
parity run. S6 remains a pre-work surface until the S5 artifact hash is returned and frozen. No
Phase 0-3 script trains or invokes RL.

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
of the 2,316 paired calls uses deterministic greedy decoding. Generated token IDs, every
next-token vocabulary-logit tensor, the exact chat-template suffix, all three Qwen MRoPE axes,
and `cache_position` must agree. These are measurement-validity equalities, not empirical-effect
thresholds.

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

I4 cannot enter a primary result unless every objective equality passes. Any prefix, token,
logit, MRoPE, or cache-position difference blocks S5 and produces no parity artifact; it is never
counted as a failed recovery. A successful artifact certifies the measurement interface only—it
does not claim that visual revision succeeded.

## S6 - I0-I4 interface-ladder surface

```bash
python scripts/v4/06_run_interface_ladder.py --execute \
  --input "$SCREEN" \
  --input-sha256 f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a
sha256sum artifacts/v4/server_preflight/*.json
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

Return S0-S6 console output, the Git revision, every produced SHA-256 line, and any `BLOCKED`
message. Do not begin Phase 4, training, LoRA, SFT, or RL.
