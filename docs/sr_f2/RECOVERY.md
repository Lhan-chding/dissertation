# External recovery of an interrupted frozen run

The `sr_f2_recovery` package is a separate operational extension. It does not
replace the registered source tree, alter the scientific matrix, erase an old
claim, or treat a cancelled allocation as a successful allocation.

Recovery requires a private, immutable `recovery/<id>/PLAN.json` binding the
original registration, runtime, 53-task matrix, old allocation identities, audit
hash, and separately deployed extension source manifest. The read-only auditor
classifies every original task as inherited completion, safe resume, never
attempted, or blocked by incomplete generation. Generation-attempt accounting
must agree with durable complete groups; the absence of a pending file does not
establish that no generation occurred. Checkpoint and retained raw identities
are verified before a resumed worker starts.

The controller uses three persistent pools of 2, 2, and 1 GPUs. Reservations,
including pending and unknown submissions, count against the five-GPU run cap,
the shared teacher QoS cap, and the user's submission limit. Each worker executes
the original frozen entry point with one visible GPU. A shared dependency queue
fills each free slot independently. Ambiguous submissions and failed attempts
are preserved and are never automatically retried.

The CLI entry points are `scripts/sr_f2/audit_recovery.py`,
`scripts/sr_f2/recovery.py`, and `scripts/sr_f2/release_recovery.py`. See their
`--help` output for arguments. Recovery audit and plan files contain private
experiment evidence and must remain outside the public repository.

The explicit recovery release gate requires all registered tasks, original and
recovery attempt lineage, actual terminal Slurm accounting, and raw-generation
accounting to agree before calling the original scientific reporting functions.
Any missing generation group blocks release. There is no implicit technical
replay exception, score inspection, or replacement of the original release gate.

Validation:

```sh
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/sr_f2_recovery -q
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/sr_f2 tests/sr_f12 tests/sr_f1/test_contract_boundary.py --import-mode=importlib -q
ruff check src/sr_f2_recovery scripts/sr_f2/audit_recovery.py scripts/sr_f2/recovery.py scripts/sr_f2/release_recovery.py tests/sr_f2_recovery
```
