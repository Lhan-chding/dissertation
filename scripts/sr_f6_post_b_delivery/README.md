# SR-F6 post-B delivery audit

This independent CPU job verifies nine completed primary pilot runs using the
unchanged frozen b7 `_verify_run`, including CPU checkpoint loads, original raw
scores, RNG, credit, numerical records and execution history. A separately pinned
`recovery_readonly.py` retains the original recovery comparison assertions and
checks the saved comparison against fresh state/raw/RNG facts and all comparison
history. It generates no answers, performs no updates, and publishes no new
recovery comparison. The earlier census/review/T03 records are not rescored.

The consumer reconstructs registration and verifies the existing historical
confirmation transport proof and every bound input byte. It does not rerun that
historical scientific confirmation or issue a new admission. Its seven adapted
functions have AST equivalence tests against the frozen original core, plus real
tensor/RNG differential tests and proof/history mutation tests. No frozen module
is patched and no prevalidated cache is injected. Phase hashing is fresh and
read-only. The original recovery entry points must not be called here because
they can rerun historical confirmation and write into the old experiment root.

## Bound source and execution

The runtime manifest contains `operations.py`, `audit.py`, `launch.sbatch`, and
`recovery_readonly.py`. The driver also pins the consumer SHA-256 and checks its
absolute import origin. The immutable source manifest binds the committed and
verified pushed revision. A separate six-field contract binds driver bytes,
frozen b7 deployment, B completion, and the exclusive output directory:
`/projects/_ssd/varunssd/louis-ssvc/sr_f6_operations_20261027/PILOT_POST_B_DELIVERY_20261101`.

Use the local `client.py` with the project Python environment. After reviewed
source is committed and pushed, execute `prepare --revision FULL_COMMIT`, then
`deploy`, `freeze`, and `submit` in order. `prepare` verifies both remote branch
and local source against immutable Git objects. Transport checks repeat source
object validation; private attempts and raw responses are saved under
`.local/pilot_post_b_delivery_20261101`. Do not call `sbatch` directly or reuse
A/B deployments.

The operator verifies B terminal `COMPLETED/0:0` and queue absence, complete B
receipt hashes, actual source and contract, unique lock/intent, and live five-job
capacity immediately before submission. It preserves unknown submissions and
forbids automatic resubmission. A clean `NO_LEGAL_CAPACITY` response has no remote
intent and may be checked later. Otherwise use `reconcile` for a missing or
ambiguous response; it only reads scheduler evidence and never submits. `read`
checks persistent receipts and scheduler state. Never change other project jobs.

All SSH uses `eee-cluster` with a 50-second bound. Heavy verification runs only in
the scheduled job: 4 CPUs, 12 GiB, two hours, zero GPUs, no requeue, OMP1. These
are execution limits, not completion guarantees. All new output and process
caches stay in the independent operations root. All ten existing transaction
locks remain held through verification and final unchanged-inventory checks.
Original failures remain preserved. Any existing output directory prevents reuse.

## Acceptance and next stage

Only fresh comparison PASS/exact/independent-process/1024 original recovery
answers plus `ALL_SAVED_FACTS_VERIFIED` history can close the delivery audit.
A completed job must also have its actual terminal scheduler state and all
persistent receipts checked. Successful facts are labeled
`DELIVERY_FACTS_NOT_SCIENTIFIC_GATE`, `full_delivery_verified=true`, with
`gate_status=NOT_SIGNED` and `evaluation_authorized=false`. The consumer alone
always reports `full_delivery_verified=false` because it covers only recovery.
Technical exceptions preserve `DELIVERY_VERIFICATION_INCOMPLETE`, not science FAIL.

Original exposure measurements, including unfavorable results, are preserved.
DEV requires separate inspection of these facts and original credit requirements
(two parents and U11 at least 12 groups, eight roots, three families), without
adding U11-win or SHUF-failure conditions. The independent DEV outer operator
still needs bound production source and a new contract. Original EVAL workers
cover twelve endpoints (three parents and nine trained endpoints), 2,304 answers,
each in a fresh process. Final `pilot_audit.audit` requires real DEV results;
never fabricate a pre-DEV `RUNTIME_PILOT_AUDIT/FACTS.json`. FORMAL/TEST remain
subject to their original scientific admission and new production provenance.
