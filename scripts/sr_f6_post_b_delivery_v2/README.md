# SR-F6 post-B delivery audit v2

Audit 238732 stopped at b1/U10 because frozen b7 requires every historical
numerical event to be PREUPDATE_PASS. The preserved, already registered 233996
OOM event occurred during backward before Adam, with no parameter change or
published update. That event must remain a failure. This independent v2 audit
reconciles that one exact event rather than deleting it or labeling it PASS.

Ordinary eight primary paths retain the frozen verifier. The b1/U10 consumer
retains the original committed-state, raw/RNG, credit, probability, optimizer,
journal and physical-call checks, with an explicit branch for the fixed OOM.
It binds the original v7/v8 contracts, old failure bytes, all 88 original
step0/READY files, original execution and accounting, backward traceback and
actual successor's restore from step0. Partial diagnostic calls remain separate
from complete committed-update diagnostics. Unknown, changed or additional
failures are rejected. No original source or module is patched, and the original
log is neither filtered nor rewritten. This is an independently sourced audit
adaptation, not a claim that the original all-PASS inventory function passed.

Recovery uses the independently pinned read-only consumer from v1, with unchanged
bytes and original comparison assertions. Historical confirmation consumes its
existing transport proof and bound input bytes; it is not rerun. Existing v7
validation receipts are read as proof, not repeated GPU validation, tensor
loading or fetch. The old audit's three passed primary receipts remain preserved;
they do not replace full verification of nine paths in this new CPU job.

## Immutable execution

Runtime source consists of operations.py, audit.py, primary_readonly.py,
recovery_readonly.py and launch.sbatch, plus a source manifest. Both consumers
are hash-pinned by the driver. The historical OOM proof is a separate private
2364-byte file with SHA-256
`5d235c3885217559562c63cea42123029b1177da4394b7739ed678ea4616c81c`.
It is not a source member and is not committed to Git.

After all tests and independent review, commit/push and verify the source
revision. Prepare using `client.py prepare --revision FULL_COMMIT
--historical-oom-proof ABSOLUTE_PRIVATE_FILE`, then call deploy, freeze and
submit once each. The authenticated deploy payload permits only the exact
fixed proof. Source and proof are validated before any directory creation.
Existing or partial operations cannot be overwritten or retried.

The nine-field contract adds the prior delivery contract, prior failure and
historical OOM proof references to the original six fields. Both B237289
COMPLETED/0:0 and audit238732 FAILED/1:0 must be absent from the queue. Only the
exact recorded technical error and NOT_EVALUATED status are accepted as the
predecessor; a scientific FAIL is not an allowed continuation. Check live five-job
capacity before the unique intent and submit. Never alter another project job.
Unknown responses require reconcile, never blind resubmission.

The new directory is PILOT_POST_B_DELIVERY_V2_20261101 beneath the authorized
project operations root. The CPU budget stays 4 CPUs, 12 GiB, two hours, no GPU,
no requeue, OMP1. All ten existing transaction locks and unchanged inventories
are checked; all new receipts and caches stay in the new operations root. SSH is
bounded at 50 seconds and carries only submission/read operations. Long checks
run in the scheduled job. Resource limits do not guarantee completion.

## Acceptance

This audit generates zero answers and performs zero optimizer updates. Only all
nine complete primary facts, the strict historical failure reconciliation,
exact independently generated original recovery comparison and full comparison
history can produce full_delivery_verified=true. All prior adverse results,
failures, raw data and hashes remain. The output remains
DELIVERY_FACTS_NOT_SCIENTIFIC_GATE, gate NOT_SIGNED, evaluation_authorized=false
and formal_authorized=false. Technical exceptions do not become scientific FAIL.

After actual terminal success and receipt verification, inspect original
exposure requirements (two feedback parents; U11 at least 12 groups, eight roots,
three families) before independently preparing DEV. No U11-win or SHUF-failure
condition is added. DEV remains the original three parents plus nine endpoints,
2304 answers in fresh processes, with new bound production source and contract.
No pre-DEV final pilot audit is fabricated. FORMAL/TEST still require their
original scientific admission.
