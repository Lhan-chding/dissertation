# GPU capacity v2 isolated diagnostic

This diagnostic is restricted to `b0-11031-G`, original checkpoint 8 and its
complete original 128-response step 9. It does not generate responses, enter an
Adam body, advance the scheduler, publish a production checkpoint, or authorize
production recovery. Original source files and `RUN_BINDING.json` remain intact.

`gpu_capacity_v2.py` is an explicit version of the original GPU snapshot helper
with a fixed cumulative **48 GiB per original-row forward** limit instead of
16 GiB, a distinct mode name, and a strict telemetry validator. CUDA 1 stores
lossless snapshots only; original model, reference, gradients, optimizer state,
objective and arithmetic remain on CUDA 0. No CPU activation fallback, dtype
change, checkpointing, dynamic retry or sample change is introduced.

The retained job 274335 failure receipt fixes the probe selection before any
model call. The original first row is replayed with both 16 and 48 GiB helpers,
with exact comparisons of all 496 gradient tensors and native objective and
probability fields. The retained failed row `s09/g07/d02` is replayed with 48 GiB;
its original unavailable gradient remains `UNVERIFIED`. After both probes pass,
the unchanged native update consumes all original 128 rows, captures clipped
gradients at the optimizer pre-hook, and exits before Adam. Independent original
no-gradient probability vectors must match all 128 exactly. Every row must have
positive bounded GPU copy/unpack activity and zero copy/version/unpack errors.

The new binding explicitly hashes the old dependency files and all new execution
files, including the launcher. State, full two-GPU RNG, old RNG projection,
original production files, hardware, peak memory and elapsed times are audited.
A diagnostic PASS is engineering evidence, not a production update or a direct
original-long-row gradient equivalence certificate.

## Validation

Run with the cluster's existing Python 3.12 environment and native source:

```sh
cd /projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2
export SR_ORD_NATIVE_SRC=/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_b7e7b06/src
PYTHONPATH=.:"$SR_ORD_NATIVE_SRC" /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -B -m unittest discover -s adapter_tests -p 'test_gpu*.py' -v
bash -n diagnostics/run_gpu_capacity_v2.sh
```

CPU tests use tiny explicitly simulated storage fixtures. They do not establish
GPU equivalence or real memory capacity. The three new suites contain 26 tests.

## Launch after quota and submission-ledger reconciliation

Use one node, one task, 2 PRO6000 GPUs, 8 CPUs and 66 GiB host memory. The launcher
preserves existing offline/deterministic environment settings and uses the full
job's host grant. It has no retry loop:

```sh
sbatch --parsable --job-name=LH-ORD-GPU-CAP48-EQ1 --account=rose --partition=cluster02 \
  --qos=soujanya-poria-startfund-2026-03 --nodes=1 --ntasks=1 \
  --gpus-per-node=pro6000:2 --cpus-per-task=8 --mem=66G --time=02:00:00 \
  --chdir=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2 \
  --output=run/slurm/gpu-capacity-%j.out --error=run/slurm/gpu-capacity-%j.err \
  diagnostics/run_gpu_capacity_v2.sh b0-11031-G
```

Record write-once submission BEGIN/RESULT evidence before/after submission and
count active, pending and unknown requests. Do not rerun an existing submission.
This two-hour limit applies only to the isolated diagnostic, not the frozen
24-step production budget. Outputs are new `run/gpu_capacity_v2_diag_JOB_b0-11031-G`
directories. Preserve all failed outputs. A later production version needs its
own explicit acceptance and checkpoint lineage; do not replace running v1 code.
