# Qwen v4 Phase 0-3 server handoff

This handoff stops before any new Qwen experiment. The archived v4 plan is byte-identical to
the supplied specification (SHA-256
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
script except S0 is inert without `--execute`. S3-S6 remain no-overwrite, hash-bound pre-work
surfaces only after the universal provenance checks pass. They do not execute the named
phase-specific gates or model experiments, do not train, and do not invoke RL. Their manifest
status is `PREWORK_MANIFEST_ONLY_PHASE_NOT_EXECUTED`.

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

## S3 - candidate scoring surface

```bash
python scripts/v4/03_score_candidates.py --execute \
  --input "$SCREEN" \
  --input-sha256 f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a
```

## S4 - layerwise assimilation surface

```bash
python scripts/v4/04_layerwise_assimilation.py --execute \
  --input "$SCREEN" \
  --input-sha256 f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a
```

## S5 - exact-cache parity surface

```bash
python scripts/v4/05_validate_cache_runner.py --execute \
  --input "$SCREEN" \
  --input-sha256 f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a
```

I4 cannot enter a primary result unless greedy cached continuation matches full-history
re-encoding, or every token difference is explicitly explained and the condition remains
diagnostic. If assistant-text decoding and chat-template re-encoding do not preserve the cached
token prefix, stop that sample before continuation and record I4 as diagnostic rather than as a
failed recovery.

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
