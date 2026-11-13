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
Job 1736279 completed zero-update DRY calibration and both starts passed.
PREFIX-only job 1737418 was cancelled while still pending with zero runtime
and zero generation, after explicit user authorization. Its single replacement
1737594 uses the identical fixed 96 requests and two GPUs with a six-hour
scheduling limit; the latest recorded state is pending. Neither result grants
training admission.
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
- `execution/diagnostic_worker.py`: one newly loaded runtime per process/slot,
  a global worker lock, immutable capture relocation, complete runtime source
  validation, measured accounting/storage and independent component replay.
  It refuses HOLD or unresolved slots and emits no combined GPU admission.
- `execution/backend_equivalence.py`: a deterministic shortest original probe
  row, its original G8 advantage and native loss compare single-device activation
  retention against the two-device saved-tensor path. It makes no claim about
  starting with only one visible GPU, full batches, or 1536-token pressure.
- `execution/prefix_control.py` and `scheduling/run_prefix.sh`: both complete
  passing DRY starts, the separately pinned release and the pre-recorded OR
  interpretation are required before exactly 48 PREFIX calls per start. The
  shared lock covers both stages; failed first-start measurement stops the second.

- `scheduling/run_gpu_checks.sh`: a bounded sequence of ten fresh GPU processes
  for the two fixed starts. It verifies the independently pinned release, full
  PREFIX evidence, explicit disk budget and live quota before each stage. It
  stops on failure or an occupied slot and permits at most four isolated shadow
  updates. It does not provide formal training admission.

- `execution/training_state.py`: weights-only checkpoint codec with exact ordered
  optimizer parameter identity, safe native RNG serialization and trusted COMMIT
  restore; its 496-scalar CPU test is not a GPU reload acceptance.
- `execution/training_transaction.py`: fixed-path, exclusive durable transactions,
  mandatory live storage callback, write-ahead update intent and atomic checkpoint
  publication. Unknown attempts block all paths; committed repeats only verify.
- `execution/training_runtime.py`: a separate TRAIN adapter using the original
  runtime, R24 fork and fresh native Adam. Live committed identity guards generation;
  the step-2 canary distinguishes the sampled step-1 and teacher step-2 endpoints.

- `execution/storage_ready.py`: explicit SHA-bound space budget, live Ceph ancestor
  quotas and filesystem minimum, with a scoped write/fsync probe. A point-in-time
  observation does not reserve shared storage; no budget is invented here.
- `execution/training_step.py`: non-CLI composition of exactly 128 original
  captures, native scoring, production gradients and one durable native update.
  Its admission authority must be independently authenticated and source-bound;
  no production authority is provided. Historical validation rechecks original
  CAPTURE/BEGIN bytes and the native update linked to the production capture.

- `execution/canary_replay.py`: fixed baseline/new-process reload phases over all
  128 original step2 rows. A private read-only lease holds the real transaction
  lock, with full-chain verification at both boundaries. No four-path aggregate
  PASS, process launcher or step3 permission is supplied.
- `execution/evaluation_loader.py`: source-bound load/audit-only START and committed
  checkpoint branches. It copies only policy, preserves the original reference
  and caller RNG, constructs no optimizer, and leaves generation/MAIN closed.
- `execution/allzero_check.py`: two fixed GPU technical slots use the actual native
  objective with explicitly zero diagnostic advantage, preserving original J and
  all captured data. Three full128 passes record PG-zero, unweighted KL and total;
  shared diagnostic gates reject failed or unknown slots. No Adam or generation.

- `execution/evaluation_probe.py`: two fixed START/R24 GPU load-only slots after
  both complete DRY gates and passing PREFIX. Actual inventory, source files,
  caller RNG and cleanup are recorded. Shared gates reject failed/unknown slots;
  this does not verify future trained checkpoints or grant admission.

- `execution/gpu_evidence.py`: a separately SHA-pinned, read-only CPU auditor
  outside the immutable GPU release. It verifies the ten completed fixed stages
  against persisted tensors, original trajectories, accounting and source hashes.
  It recomputes native numerical checks, fixed Adam recipe validation, actual
  preclip bounds, backend advantages and replay/direction evidence. It produces
  no training permission and does not claim a second observation of unsaved GPU
  weights or unmeasured production peak memory.

The complete training launcher/admission authority, real production GPU replay
acceptance, canary/recovery integration and terminal MAIN runner are not yet
completed end-to-end. Do not invoke the old training CLI as a substitute.
Do not manufacture DESIGN_FREEZE or GPU_VERIFICATION from test fixtures.
The GPU diagnostic driver must bind the frozen root/group/request ordering,
verify saved-tensor storage after backward, and use actual runtime accounting.
Shadow acceptance still needs actual execution of two independent R24 restores
per start, GPU Adam on isolated tensors, and a global maximum of four actual
shadow steps. Existing same-copy shadow repetition alone does not establish
restored replay; a JSON assertion alone does not establish a real reload.
The complete 11032 DRY population contains original 1536-token trajectory
`SAT1/DRY/11032/g004/d07` inside the frozen probe128. Its actual full-batch
production replay remains to be verified; the 11031 probe maximum is 709.
No extra or synthetic rollout is needed to select a long-capacity input.

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
