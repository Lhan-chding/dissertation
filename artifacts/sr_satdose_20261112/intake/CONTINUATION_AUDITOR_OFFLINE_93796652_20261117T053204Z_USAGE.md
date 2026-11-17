# Continuation audit and monitor usage

These operators are local-plan-only by default. Replace every ALL_CAPS argument with an actual pinned artifact or integer; add --execute only for the intended future operation. This archive was produced offline; no SSH, actual risk audit, model construction, optimizer, GPU submission, git commit or push was performed by the implementing agent.

## One completed chunk audit

From the repository root:

```text
.venv/bin/python -B experiments/sr_satdose_20261112/operations/satdose_audit_continuation.py --submission-receipt ABS_ACTUAL_SUBMIT_RECEIPT --submission-sha256 ACTUAL_RECEIPT_SHA256 --job-id ACTUAL_JOB_ID --start-seed 11031 --arm T0 --first-step 3 --local-dir ABS_NEW_LOCAL_AUDIT_DIR
```

A future --execute performs one original CPU audit and then one whole-bundle download. It does not submit or cancel jobs. The exact submitter source must be eceec5aab4de56b53d8e1483d2e49093e28f4048b2ef818302bb3db499fb0f94 and its source/intent/raw/preparation provenance must remain available locally.

Remote write-once directory:
`/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/continuation_audits_71018e5/start-11031/T0/chunk-003-008/`

Inspect LOCAL_RECEIPT.json, artifacts/AUDIT_REPORT.json and the full raw evidence. Successful audit permission is only INDEPENDENT_CHUNK_AUDIT_ONLY_NO_TRAINING_MAIN_OR_SUCCESSOR. No implicit next task is authorized.

## Download-only recovery

Use the SAME submission/job/seed/arm/first-step arguments, a NEW local directory, and:

```text
--recover-manifest ABS_SAVED_REMOTE_MANIFEST_JSON --recover-sha256 EXACT_MANIFEST_SHA256 --execute
```

This invokes only the read-only remote bundle download. It never reruns the audit, modifies the failed download, retries generation, or submits a job. If no trusted manifest returned, preserve the UNKNOWN audit attempt and inspect its existing remote evidence separately; do not re-audit automatically.

## Read-only monitoring

```text
.venv/bin/python -B experiments/sr_satdose_20261112/operations/satdose_monitor_continuation.py --submission-receipt ABS_ACTUAL_SUBMIT_RECEIPT --submission-sha256 ACTUAL_RECEIPT_SHA256 --job-id ACTUAL_JOB_ID --start-seed 11031 --arm T0 --first-step 3 --output-dir ABS_EXISTING_LOCAL_INTAKE
```

Add --execute for an actual read-only snapshot. Every invocation saves a timestamped local directory with source copies, intent, original stdout/stderr bytes and RECEIPT.json. Only this job's scheduler rows, exact chunk records, per-step BEGIN/CAPTURE/COMPLETE, COMMIT/risk records, global project incidents and this job's log tails are read. Counts are non-atomic observations and are never completion admission.

## Required ordering and boundaries

Audit each completed chunk before another path or DEV changes the actual committed population. The final POST_RISK is replayed in full using original risk.assess via ContinuingAdmission with original DEV and the exact full four-path committed prefixes. Intermediate reports are SHA/structure/exact-prefix checked, NOT historical full replays. No truncated current chains or copied historical root are used.

The audit requires the exact parent plus batch/extern/0 scheduler records COMPLETED 0:0, matching original CLAIM allocation/process identity, existing inactive worker/launcher/transaction lease boundaries, source84/identity19, original full admission/canary, complete native checkpoint/state/scoring and 128 captures per added step. A purged job's squeue rc1 is accepted only for empty stdout and the exact known Invalid job id stderr, alongside complete successful sacct evidence. A true HOLD or incomplete/unknown/active successor blocks successful publication and is retained.

Exit evidence is Slurm termination plus cooperative lease boundaries. No direct compute-node /proc process death is claimed, and the original RESULT.process_exit_observed=false remains unchanged. Bundles include the entire new audit publication and observed hashes; large original TRAIN raw/state data remains on the server, where it is actually verified. This is not a standalone retraining archive.

All fixed 3-8 / 9-16 / 17-24 chunks remain unchanged. Required DEV8/16 PENDING is retained; no next chunk until the original four-path DEV barrier is met. No MAIN permission is conferred.

## Offline verification

45 new boundary tests passed: 32 audit + 13 monitor. They use clearly synthetic local fixtures and do not claim a real original model/state/risk audit. The inert 250000-byte bootstrap roundtrip actually runs a local Python process with SSH replaced, proving no large argv is used. The original 44-test log is retained separately. Raw archived test copies preserve exact bytes; rerun using the repository paths and pinned dependencies recorded in REVIEW_INDEX.json rather than relocating the files in isolation.
