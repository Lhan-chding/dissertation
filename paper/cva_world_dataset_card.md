# CVA-World v2 dataset card

Last updated: 2026-08-14 (Asia/Singapore, UTC+08).

## Summary

CVA-World is a fully synthetic, deterministic visual diagnostic dataset for
studying whether an outcome-correct trajectory can contain a controlled local
perception error. The clean local `cva_v2` generation contains 1,820 records. For
the four non-OOD splits, every task-family/split combination has 10 semantic
states crossed with every applicable IID style: nine for digit, gauge, and bar,
and eight for count and relation. Each of the 50 OOD semantic states has two
independent `layout_shifted` realizations. Every record stores the latent scene, canonical
answer and reasoning, an executable error catalog, split keys, and a rendered
PNG path.

The task families are `digit_offset`, `count_transform`, `gauge_calibration`,
`bar_chart_aggregate`, and `relation_rule`. The splits are `train`,
`calibration`, `val`, `iid_test`, and `ood_test`.

## Generation and provenance

The frozen generator and renderer configuration is
`configs/data/cva_v2.yaml`. Reproduce the local snapshot with:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_cva.py \
  --config configs/data/cva_v2.yaml \
  --overwrite

PYTHONPATH=src .venv/bin/python scripts/audit_dataset.py \
  --manifest artifacts/manifests/cva_v2.json \
  --output artifacts/metrics/cva_v2_audit.json \
  --report-root artifacts \
  --overwrite
```

The explicit `--overwrite` authorizes replacement only after any existing
generated snapshot/report passes its provenance binding; omit it when
preserving an existing copy for comparison.
Generation also computes the planned image count and total render pixels before
sample construction, rejecting more than 10,000 images or 256,000,000 pixels;
contact sheets use bounded 25-image batches whose Pillow images are closed.
The audit command is the automatic-only form: a clean exit does not change its
`phase_d_ready: false` status without a bound `--visual-audit` human record.

On 2026-08-14, the generator and automatic audit ran from tracked-clean HEAD
`84eda3bbaeaf4822f9cb238f3080ead3cd6c9c0b`. The generation run is
`generation-20260814T010836108099Z-e0658f1a9850`; the audit run is
`audit-20260814T010913290056Z-6bcc090c91cc`. Their local bindings are:

| Binding | SHA-256 |
|---|---|
| tracked `configs/data/cva_v2.yaml` file bytes | `e0658f1a9850d300d74eeb3e6c445f0ad8dcf702e846b1b1d11fb34e3f6b1071` |
| manifest-internal canonicalized config payload | `0780d12f5817dce4f8ffd4854b331f9bd614679048a30b550a11b3ec940709d6` |
| `artifacts/manifests/cva_v2.json` file bytes | `6bcc090c91cc7f47fa03309a2fb56bbd7e2b0e8a0b8bc8ca21b98ee7399c340a` |
| manifest self-binding | `3f9efc0d0ec76284d7d5dc1a36d0c1b57ec736aa8c4c1b968447e022fca0c108` |
| canonical content | `5010137acc0cfc06238efbf23183aa865080d4882a9abcadfe6d8b365092ad36` |
| raw `dataset.jsonl` file bytes | `d439f621c395a21cb5386f1d769b3fd53cf6195c7b2389a6df7e3fcaa43eaddc` |
| complete image set | `6a1171e8716cd93c644bf771dcd3a6f79eceb9c6cca5120f5812895e57999e4e` |
| `artifacts/metrics/cva_v2_audit.json` file bytes | `f6781e82052463f0e749864ced41b559a680604872f0de72b39fa2955dc8142f` |

The tracked config file hash and manifest-internal canonicalized-config hash
identify different byte domains and must not be interchanged. Generated data,
images, manifests, sheets, audit, and run bundles remain ignored/local: this
record is not an approved dataset release or a `VERIFIED_CPU` metrics bundle.

The clean automatic audit reproduced 1,820 JSONL records and 1,820 PNG images
in 250 semantic groups:

| Dimension | Recorded automatic-audit count |
|---|---:|
| `digit_offset`, `gauge_calibration`, `bar_chart_aggregate` | 380 rows each |
| `count_transform`, `relation_rule` | 340 rows each |
| each of `train`, `calibration`, `val`, `iid_test` | 430 rows |
| `ood_test` | 100 rows |
| `baseline`, `size_compact`, `rotation_tilted`, `contrast_low`, `background_grid`, `occlusion_local`, `blur_mild`, `distractor_marks` | 200 rows each |
| `font_weight_bold` | 120 rows |
| `layout_shifted` | 100 rows |
| bar semantics per split: `sum` / `difference` / `ratio` | 4 / 3 / 3 |
| bar rows per non-OOD split after style crossing | 36 / 27 / 27 |
| bar rows in OOD after two realizations | 8 / 6 / 6 |
| contact sheets at 25 images per sheet | 73 sheets |
| canonical-solver checks | 1,820 |
| registered error round trips | 4,020 |

These observed counts match the values derived from the frozen crossing rules.
The prior 250-record `cva_v1` hashes are deliberately not carried forward: they
are incompatible with the v2 identity/manifest contract and are not current
evidence.

Stale untracked `cva_v1` manifests/data and
`artifacts/figures/cva_contact_sheet_*.png` files are historical local
by-products. They are not part of v2 and must not be published or cited as
current evidence.

Generated data, images, manifests, and contact sheets are reproducible local
artifacts and are ignored by Git. The manifest and versioned contact sheets are
the local provenance surface, but neither should be committed or published
until the release bytes and privacy/licensing review are accepted. No externally
sourced scene image, annotation, or personal data is used; text rendering does
use the font bundled with the installed Pillow build.

## Record contract

Each immutable record includes:

- `sample_id` and `task_family`;
- a structured `scene` sufficient for the canonical solver;
- `question`, `canonical_answer`, and `canonical_reasoning`;
- `error_catalog`, whose registered transforms are reversible;
- `split_keys` containing exactly `semantic_split`, `visual_style`, and
  `error_mechanism`;
- an optional top-level `source_id` that binds an OOD realization to its IID
  source; it is not a split key;
- `image_path` for the deterministic rendering.

For `bar_chart_aggregate`, the closed question contract uses the first two
bars and exactly one of `sum`, `difference`, or `ratio`. Every split has ten bar
semantic states in the fixed 4/3/3 operation distribution. Across the four
non-OOD splits and the OOD split, full rendering yields 152 `sum`, 114
`difference`, and 114 `ratio` rows. Canonical answer supports are `13..22`
(train), `23..32` (calibration), `33..42` (validation), and `43..52` for both
paired IID/OOD states. The ratio positions are constructed to divide exactly;
their answers are `15.0/18.0/21.0`, `25.0/28.0/31.0`,
`35.0/38.0/41.0`, and `45.0/48.0/51.0` for those four supports. They use the
JSON float representation while remaining integer-valued.

Paired error-mechanism counterfactuals are created with
`generate_error_mechanism_counterfactuals`. They retain sample identity and all
non-intervened semantics while changing only the registered error mechanism and
its associated catalog. They are distinct from the five mutually disjoint
dataset splits.

## Validation status

The clean schema-v2 automatic audit passed solver, error-solver/round-trip,
split, visual-factor realization, answer balance, OOD image shift,
style--semantic joint-independence, and deterministic data/image/contact-sheet
replay checks. It recorded 1,820 samples and images, 1,820 canonical-solver
checks, 4,020 registered error-solver/round-trip checks, 73 sheets, and 100 OOD
pairs. The OOD split changes only the preregistered `visual_style` and
`error_mechanism` factors. CPU tests additionally cover same-seed
generator/manifest equality, guarded overwrite, full-crossing/count logic, and
representative applicable-style pixels; those tests do not replace the local
manifest/audit evidence.

Phase D nevertheless remains `PARTIAL_GATE` because the audit records
`human_reviewer_signoff: false` and `phase_d_ready: false`. Promotion requires a
self-reported human reviewer to record a separate review of at
least 200 images, bound to the recorded local manifest and image-set hashes, and the
audit consumes it through `--visual-audit`. Any future
`artifacts/manifests/cva_v2_visual_audit.json` agent record must retain
`human_reviewer_signoff: false`; a human record should use
`artifacts/manifests/cva_v2_human_review.json`.
Human-review metadata must use a bounded public reviewer pseudonym matching
`reviewer-[a-z0-9][a-z0-9-]{0,30}`; the audit rejects names, email addresses,
and other values outside that form. Any real identity, consent record, and
pseudonym mapping stays outside the generated/publishable artifact surface.
The closed record must also list every manifest contact sheet in
`reviewed_contact_sheets` and set `contact_sheets_reviewed` to 73; reviewing at
least 200 unique manifest sample IDs does not waive all-sheet coverage. Its
`images_reviewed` count must equal the unique ID list, and its manifest/image-set
hashes must bind the recorded local v2 bytes.
The audit validates the self-reported record, pseudonym form, and bindings, not
an external signature or the reviewer's real-world identity; consent/identity
evidence must be handled separately from the publishable record.

The v2 renderer now implements a closed ten-style catalog: `baseline`,
`font_weight_bold`, `size_compact`, `rotation_tilted`, `contrast_low`,
`background_grid`, `occlusion_local`, `blur_mild`, `distractor_marks`, and
`layout_shifted`. Non-OOD semantic states are fully crossed with all styles
applicable to their family; OOD states have two independent shifted-layout
realizations. Legacy
`font_a`/`font_b`/`rotated` aliases are compatibility-only and absent from the
frozen v2 configuration. The local automatic audit recorded catalog coverage
and split independence; human review and release approval remain separate.
The font-weight factor uses the Pillow-bundled font plus a fixed stroke, never a
system/PATH font or a distinct typeface, and applies only to digit, gauge, and
bar scenes; it is explicitly nonapplicable to count and relation scenes.
The recorded per-style counts and 73-sheet inventory are listed above.

## Intended use

CVA-World supports controlled parser, solver, corruption, state-injection,
error-coupling, and paired-shift experiments. It is not a benchmark of natural
image understanding, visual diversity, linguistic robustness, or real-world
measurement accuracy. Results on this dataset alone cannot establish a Qwen or
general VLM claim.

## Known limitations

- Scenes use simple programmatic graphics and a compact task vocabulary.
- Each family/split has only ten semantic states; the expanded row count comes
  from visual crossing, not added semantic diversity.
- Font weight is nonapplicable to count/relation, so their non-OOD crossing has
  eight rather than nine IID styles.
- Its error catalog is designed for causal control, not for estimating the
  prevalence of naturally occurring model errors.
- Canonical local v2 generation and automatic-audit evidence is recorded, but
  human sign-off and release review are pending; an optional agent-only review
  cannot satisfy the human gate.
- The project registry records no VLM-checkpoint evaluation on this snapshot.

## License and citation

The repository metadata declares the generator, renderer, and code as MIT; the
distribution terms for a future dataset release must be confirmed before the
synthetic outputs are published, including review of Pillow and its bundled
font's applicable notices. Cite the repository using `CITATION.cff`. The
generated scenes contain no human subjects or externally sourced photographs;
that narrower statement does not waive dependency/font licensing.
