# GPU48 production execution version

This explicit version uses the verified fixed 48 GiB per-row GPU snapshot helper
in `diagnostics/gpu_capacity_v2.py`. The original `adapter/`, original
`RUN_BINDING.json`, 16 GiB production version and diagnostic source files remain
immutable. Model, reference, gradients, Adam state and all training arithmetic
remain on CUDA 0 in the frozen FP32/eager configuration. CUDA 1 stores lossless
intermediate snapshots only. There is no CPU activation fallback, UVM, precision
change, resampling, gradient checkpointing, or reduction of the 24-step budget.

The technical basis is isolated job 274417: all 128 original b0-G step-9 rows
reached the native pre-Adam gate without an update, their policy/reference token
probabilities matched independent no-gradient replay, and state/RNG and source
files were preserved. The fixed first original row's 16/48 GiB results matched
all 496 gradients and objective/probability fields. The retained 1536-token
failure row completed with the new capacity. Its unavailable original long-row
single-GPU gradient remains `UNVERIFIED`; diagnostic PASS is not a production
update or a claim of scientific benefit.

A new immutable acceptance binds the execution commit, exact added dependencies,
original binding, diagnostic evidence and allowed path list before loading.
Initial b0-G step 9 must reuse its complete original 128 responses without a new
generation and must exactly match the accepted full native audit and all 496
clipped diagnostic gradients before Adam. Later accepted paths retain their own
original checkpoint, samples, seed and parent, and all native numerical gates.
Every GPU48 step records actual optimizer/scheduler calls, strict per-row
capacity/mode/copy/unpack evidence and execution provenance in UPDATE and sampler
state. A failed attempt is retained and cannot stand in for a committed update.

Original one-GPU checkpoints, fresh two-GPU parents and explicitly verified
historical v1 checkpoints can be restored without changing their scientific
identity. Historical v1 bindings, checkpoints and updates are verified through
the retained v1 implementation, keep their v1/16 GiB labels, and are never
reinterpreted as v2. New v2 provenance names the exact source COMMIT and records
full two-GPU RNG restoration. Healthy v1 workers remain unchanged.

## Validation and execution

Use the deployed Python 3.12/native environment for the complete CPU regression
suite; mocked tests are not a substitute for real GPU recovery evidence:

```sh
PYTHONPATH=. SR_ORD_NATIVE_SRC=/path/to/frozen/native/src python -B -m unittest discover -s adapter_tests -p 'test_gpu*.py' -v
bash -n gpu_production_v2/run_training.sh
```

The launcher takes `PATH COMMIT ACCEPTANCE`, preserves the full 24-step budget
and requires one node with two PRO6000 GPUs, 8 CPUs and 66 GiB host memory. A
48-hour production job limit is only a scheduling boundary. Reconcile active,
pending and unknown jobs and shared target-QoS usage before each submission;
retain write-once BEGIN/RESULT guards and never blindly resubmit an unknown
request. Initial diagnostic and production acceptance do not authorize extra
scientific samples or altered frozen inputs.

## Independent path seal

After a path completes all 24 steps, run:

```sh
python -B -m gpu_production_v2.seal_path --path b0-11031-G
```

The sealer retains the original raw/native-score/token/plan checks and all
checkpoint-state, optimizer, scheduler and RNG checks. It additionally verifies
explicit v1/v2 execution identity and each committed version's actual GPU and
Adam evidence, with strict 48 GiB telemetry for v2. Prior failed executions and
pre-Adam passes remain in the evidence; the committed provenance selects the
actual update. Publication of `PATH_RECEIPT.json` is atomic and last.

All twelve paths must finish 24 steps and be independently sealed before the
26 registered evaluation endpoints are opened. Revalidate the extra GPU bridge
references explicitly before evaluation: the original evaluation validator does
not automatically rehash these additional references. Do not reseal an already
sealed original-only path such as b0-D. Preserve all negative scientific results,
original long-baseline limitations and technical failures in the final report.
