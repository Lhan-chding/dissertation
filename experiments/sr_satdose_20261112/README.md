# SR-SATDOSE-1 implementation

The original package is preserved byte-for-byte in `supplied_package/`. `runtime/`
contains the new experiment namespace. Existing native scoring, rendering,
optimizer and loss sources are read-only dependencies, not replacements.

## Verified so far

The user's historical result ZIP is located in the existing SR-ORD worktree at
`sr_ord_train/execution/final_delivery_20261111/SR_ORD_TRAIN_RESULTS_20261111.zip`.
Its SHA-256 is `b6a560fd0d74e9fbca0d0000b74dcc622b749c400e4f45fb211f6301e29d89d2`.
The historical replay passed locally and remotely. Server tensor audit verified
both R24 checkpoints (496 tensors each) and the original b0 reference.

`data_adapter` has prepared all 28,936 requests and 1,376 images on the server.
The user authorized AI inspection of the fixed 32 images in place of human
inspection. The immutable design freeze was then written before model calls.
Job 1736279 is running zero-update DRY calibration; it grants no training admission.
See the dated admission report for the most recent observed progress.

## Modules and limits

- `registration`: complete native/model source lock and write-once design freeze.
  It rejects absent visual review, changed inputs and pre-existing model attempts.
- `data_adapter`: full data/manifest construction, historical and cross-split
  exclusion, dual oracle, paired list-only changes and original image rendering.
- `fork_loader`: original b0 reference followed by R24 policy; read-only starts
  create no optimizer; training forks use native fresh AdamW. Restore requires
  the exact new path identity and a trusted COMMIT SHA.
- `backend`: finite manifest, public-only boundary, original token capture,
  exclusive attempt creation and no automatic unknown-attempt resampling.
- `native_runtime`: original FP32 runtime and two-GPU capacity48 saved-tensor
  path, restricted to DRY/PREFIX_CALIB and original-token replay. No training.
- `diagnostics`: native loss components, independent witness sum-logprob
  gradients and isolated shadow Adam. Its output is never a GPU admission PASS.
- `dry_driver`: finite zero-update generation entry point, subject to actual
  immutable freeze and source validation. CPU fixtures are not production data.
- `analysis/calibration.py`: CPU analysis of completed DRY/PREFIX captures with
  fixed denominators and native score replay. Partial populations are refused.
  PREFIX uses either complete strict or complete terminal-brace readout passing,
  with no mixing of favorable metrics. A separate pre-PREFIX clarification and
  zero-attempt confirmation are required; no training permission is produced.
- `execution/production.py`: independent native loss and gradient capture for
  sixteen complete G8 groups. It preserves the original numerical checks and
  captures gradients before and after clipping without accepting an optimizer.
  CPU agreement with diagnostics is verified; actual GPU replay is outstanding.
- `execution/diagnostic_inputs.py`: both complete passing DRY populations are
  revalidated and rescored before projection to the frozen probe/witness groups.
  It preserves original request order and uses native-known authorised c.
- `execution/shadow_replay.py`: fixed four-slot, write-once shadow claims. Each
  production slot requires CUDA, original R24 identity, independently recorded
  restore/capture evidence and one fresh native Adam step on isolated tensors.
  Unknown attempts cannot be automatically retried. CPU fixtures are separate.

The formal training transaction publisher/driver, production gradient replay
acceptance, canary/recovery integration and terminal MAIN runner are not yet
implemented end-to-end. Do not invoke the old training CLI as a substitute.
Do not manufacture DESIGN_FREEZE or GPU_VERIFICATION from test fixtures.
The GPU diagnostic driver must bind the frozen root/group/request ordering,
verify saved-tensor storage after backward, and use actual runtime accounting.
Shadow acceptance still needs actual execution of two independent R24 restores
per start, GPU Adam on isolated tensors, and a global maximum of four actual
shadow steps. Existing same-copy shadow repetition alone does not establish
restored replay; a JSON assertion alone does not establish a real reload.

## CPU validation

Use Python 3.12, not macOS system Python 3.9. Run each test suite in a fresh
process because native modules have absolute imports and reject source shadowing.

```sh
SR_HISTORICAL_BUNDLE=/verified/historical/results TMPDIR=/private/tmp \
  .venv/bin/python -m unittest discover \
  -s experiments/sr_satdose_20261112/supplied_package/tests -v
SATDOSE_NATIVE_SRC=/verified/native/src .venv/bin/python -m unittest discover \
  -s experiments/sr_satdose_20261112/tests -v
```

Scientific parameters remain those in `supplied_package/config/experiment.json`.
Current evidence and blockers are recorded in
`artifacts/sr_satdose_20261112/ADMISSION_REPORT_zh.md`.
