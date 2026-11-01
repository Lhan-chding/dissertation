# SR-F6 post-B delivery audit

This independent CPU wrapper verifies the nine completed primary pilot runs
using the frozen b7 audit functions. It reads the existing recovery completion
receipt by hash but does not reverify the full recovery. It does not
generate answers, train, rerun recovery comparison production, evaluate DEV, or
write the original final scientific audit. Checkpoint tensors are loaded on CPU.
The original training raw records are checked with their registered scorer; the
earlier census/review/T03 records are not rescored.

The driver and launcher are preparation for a new, separately bound deployment.
They are not a submission tool. Do not run the launcher against a working copy,
reuse the completed A/B deployments, or call `sbatch` without the submission
checks below. There is currently no deployment or submitted job for this wrapper.

## Execution contract

Use a fresh directory under
`/projects/_ssd/varunssd/louis-ssvc/sr_f6_operations_20261027`.
Commit and push the reviewed source, verify the pushed revision, then create an
immutable deployment manifest and a separately hashed contract. The contract
binds the driver bytes, frozen b7 source, B completion evidence and a new output
directory. Do not overwrite an existing contract, result or attempt record.

The launcher requires these exported values:

- `SRF6_DELIVERY_CONTRACT`: absolute contract file path.
- `SRF6_DELIVERY_CONTRACT_SHA256`: its SHA-256.
- `SRF6_DELIVERY_DRIVER`: absolute path to the immutable audit driver.

It checks contract/driver bytes before running the driver and puts all caches in
the new operations directory. The proposed allocation is 4 CPUs, 12 GiB, two
hours, zero GPUs, no requeue, and `OMP_NUM_THREADS=1`. These limits are an initial
execution budget, not measured guarantees. Preserve partial results and technical
failures; diagnose before any equivalence-preserving continuation.

## Submission prerequisites

An independent, reviewed submission operator is still required. It must:

1. Verify the source deployment manifest, contract and exact requested resources.
2. Verify 237289 is terminal `COMPLETED/0:0` and absent from the queue, with the
   four task/execution/storage closures and original exact recovery evidence.
3. Acquire a unique submission lock and reject existing intents, attempts,
   unknown submissions, results or prior execution of this audit contract.
4. Read the live user submission count and reject submission at five occupied
   slots. Do not modify or cancel any other project jobs. This CPU job does not
   consume a GPU reservation, but still consumes a submitted-job slot.
5. Persist a unique intent and the actual scheduler response. A missing or
   ambiguous response requires reconciliation, never automatic resubmission.

All SSH uses `eee-cluster` with a 50-second connection/command bound. Audit
computation belongs in the scheduled job; SSH only submits or reads persistent
receipts. Do not repeatedly run the whole audit on a login node to fit that bound.

## Acceptance and next stage

The wrapper records delivery facts and the original exposure measurements,
including unfavorable results. A technical exception is not a scientific FAIL.
Even successful completion remains `PRIMARY_FACTS_RECOVERY_RECHECK_PENDING`,
with `full_delivery_verified=false` and no DEV authorization.

The frozen `recovery.verify_recovery` and `verify_comparison` functions transitively
rebuild PILOT_V3 registration. In a fresh process that invokes the historical
confirmation verifier, creates new historical audit files, and can update the
phase verification cache. They must not be called by this wrapper. Neither
`verify_payloads=False` nor moving those writes into an allowlist removes the
historical re-audit. Do not inject caches or monkeypatch frozen functions.

Full delivery still requires a separately reviewed, equivalent read-only recovery
consumer: bind the existing registration/confirmation and original comparison,
map every original comparison assertion, and verify complete comparison history.
Referencing the existing exact PASS is not a claim that this recheck has run.

Only after checking the actual terminal job state, every delivery receipt and the
registered recovery/credit evidence may the independent DEV outer operator be
prepared for admission. The frozen b7 EVAL workers already define twelve
endpoints (three parents plus nine trained endpoints), 2,304 answers, and a new
process per endpoint. Final `pilot_audit.audit` requires that real DEV evidence;
do not fabricate a pre-DEV `RUNTIME_PILOT_AUDIT/FACTS.json`. FORMAL/TEST require
their original scientific admission and real production provenance.
