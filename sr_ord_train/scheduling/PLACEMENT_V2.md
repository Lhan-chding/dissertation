# Placement v2: fixed node-count representation

Slurm displayed pending fixed one-node request 275612 as `NumNodes=1-1`,
with `ReqTRES=node=1,gres/gpu=2,gres/gpu:pro6000=2`. V1 rejects this
representation. V2 accepts only equal positive endpoints and still cross-checks
squeue and ReqTRES. True ranges, malformed counts, duplicate fields,
mismatched totals and terminal states still fail closed. Pending requests
continue to count in full.

Run `python -m scheduling.placement_v2 --owner varun024 --unresolved-ledger AUDIT.json`.
The ledger format, 15-minute freshness, complete queue observation, limits
(experiment 6 GPUs, shared QoS 8 GPUs, owner 5 jobs), node exclusion and production
commit are unchanged; see README.md. V1 and bound production/diagnostic sources
remain unchanged. V2 is an explicit separate copy so receipts identify the
parser used. It never submits or changes jobs. Record its source hash and
scheduling commit separately from the production commit.

Validate with `python -B -m unittest discover -s adapter_tests -p 'test_placement_v*.py' -v`.
The full original safety suite also runs against v2, plus equal-range, real-range,
cross-check and end-to-end CLI regressions.
