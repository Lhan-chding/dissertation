# Registered DEV evaluation: isolated environment repair v2

The user cancelled the additional delivery audit and explicitly requested immediate
DEV evaluation on 2026-11-01. This independent wrapper records that waiver; it does
not assert completion of the cancelled audit or grant FORMAL/TEST authorization.

The fixed original plan evaluates three parents and nine step-12 endpoints:
96 DEV questions × two draws × 12 endpoints = 2304 new answers. Two GPU lanes
follow the original queue order, each endpoint in a fresh process. No training,
recovery regeneration, historical census rescoring, retry, or extra endpoint is
provided. Original raw records, errors, and unfavorable outcomes remain intact.

`runtime_adapter.py` separates the original evaluator's sampling/runtime machinery
from its recursive historical admission audit. It retains actual frozen-source
identity, model/checkpoint loading, original inputs/seeds/scoring and durable
one-shot ledgers. Historical admission receipts are provenance only; current
execution authority is the independently bound explicit-user contract. It does
not monkeypatch frozen code or inject admission caches.

The local client prepares an immutable four-member source bundle plus manifest
from a pushed Git revision, deploys a new directory, freezes a new contract, and
submits exactly one two-PRO6000 job with 8 CPUs, 66 GiB, a 23-hour maximum and no
requeue. Submission checks the actual eight-GPU teacher quota, five-job user cap,
completed B predecessor and failed DEV 238849 predecessor, both absent from the queue. Other projects are not
modified. Transport has a 50-second bound and saves private receipts. Unknown
submission outcomes require reconciliation, never blind resubmission.

Use the project Python environment with `client.py prepare --revision REVISION`,
then `deploy`, `freeze`, and `submit`, once each. Subsequent `read` is read-only;
`reconcile` resolves an existing unknown intent without submitting. Private
receipts stay in `.local/pilot_dev_eval_v2_20261101` and never enter Git.

Results are written under the new operations directory. The summary is descriptive
DEV evidence, `COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED`; full delivery verification
remains false and FORMAL/TEST remain unauthorized. A 23-hour limit is not a
completion guarantee. No automatic audit or monitor is recreated.

This version is limited to the pinned pre-lane failure of job 238849. Its original
FAILURE, STARTED, contract and two OMP1 failure logs remain unchanged. The old
RESULT directory must contain only those four runtime files, with no LANES or
completion; original evaluation and evaluation-task inventories must remain empty.
This permits the first registered DEV answers after an environment failure; no
previous answers are regenerated. The failure remains technical, not scientific.

Both srun lanes explicitly use --export=ALL, and the launcher sets
SLURM_EXPORT_ENV=ALL, OMP_NUM_THREADS=1 and CUBLAS_WORKSPACE_CONFIG=:4096:8.
Every child Python interpreter uses -I -B -u, independently of environment hints,
so imports cannot create bytecode in the immutable deployment. GPU visibility
comes from srun unchanged. The evaluator numerical implementation is unchanged.
