# GPU-only execution version

This is a technical execution bridge for the frozen SR-ORD-TRAIN paths. The
original `adapter/` files, scientific identity and `execution/RUN_BINDING.json`
are unchanged. Model, reference-forward arithmetic, gradients and Adam remain
on CUDA 0 in FP32. The retained saved-tensor helper stores at most 16 GiB of
intermediate snapshots per original row on CUDA 1. There is no CPU activation
fallback, resampling, mixed precision or gradient checkpointing.

Before loading, the derived binding validates the original source inventory,
new execution commit and dependency hashes, exact registered path and immutable
technical acceptance. Each execution records actual two-GPU hardware and the
transition from the original RNG state. Every new checkpoint binds this version
and recovery lineage in its sampler state and UPDATE receipt. Original one-GPU
checkpoints remain intact; subsequent two-GPU checkpoints require provenance.

For the first b0-R recovery update (step 5), all 128 original complete attempts
must exist before any new BEGIN or generation call. The optimizer pre-hook
compares the full native audit (except the separately recorded full RNG hash)
and every clipped adapter gradient against retained diagnostic 268174. A
mismatch stops before Adam. Every step verifies original row order, positive
GPU transfer and unpacking, no copy/version errors, unchanged GPU1 RNG and
exactly one completed Adam/scheduler call. Failed attempts retain their records.
The original long single-GPU baseline OOM remains an unverified comparison;
this bridge does not convert it into a scientific equivalence claim.

Initial acceptance should allow only `b0-11031-R`. Broader execution requires
review of the first actual COMMIT and immutable acceptance for the added paths.
The launcher always retains the full 24-step scientific budget. A Slurm time
limit is only a scheduling boundary. Account for active, pending and unknown
allocations and shared QoS requests before each submission.

After a bridged path completes, run `python -m gpu_production.seal_path --path
<registered-path>` in the deployed package. It retains the original independent
native-score/token/checkpoint checks and additionally verifies GPU provenance,
actual pre-Adam and optimizer accounting. Prior failed pre-Adam passes remain
in evidence; the committed execution is selected through its immutable binding.
The original-only sealer remains valid for original-only paths. No formal
evaluation is permitted until all twelve paths are sealed. Before formal
evaluation, explicitly revalidate the additional GPU bridge artifacts: the
original evaluation validator checks its original required evidence references
and does not automatically rehash the added bridge evidence.

Validation in the server Python 3.12 environment:

```sh
PYTHONPATH=. SR_ORD_NATIVE_SRC=/path/to/frozen/native/src python -B -m unittest discover -s adapter_tests -p 'test_gpu*.py' -v
bash -n gpu_production/run_training.sh
```

CPU fixtures test binding, failure gates, exact tiny native update behavior and
transaction plumbing. They are not a substitute for the actual GPU pre-Adam
comparison and subsequent production COMMIT evidence.
