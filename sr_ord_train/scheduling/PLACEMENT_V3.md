# Placement v3: verified single-node job-total GPU requests

V3 is a separate read-only planner. It fixes the case where `squeue %b` reports
`N/A` or `(null)` because GPUs were requested per job rather than per node.
Neither value means zero GPUs. V1, V2 and all bound production/diagnostic sources
remain unchanged. The production commit is still
`f594e32705581d8bcfdd609560eda87fd11cbc31`.

For every relevant row (target QoS, supplied owner, or `LH-ORD-` experiment),
the CLI reads `scontrol show job -o ID`. It does not discard an owner's other-QoS
job because its per-node GPU field is absent. The resulting report preserves the
raw complete `squeue` observation and each relevant raw `scontrol` observation in
`scheduler_observation`, as well as a `resource_basis` for every counted request.
Reports and audited ledgers are execution evidence and must not be uploaded with
code.

An absent per-node GPU field is accepted only when all of these checks pass:

- The queue says exactly one node, `NumNodes` is fixed one or `1-1`, and
  `ReqTRES` also says one node. Multi-node job-total requests remain unsupported;
  the planner does not infer a per-node distribution.
- There is exactly one typed positive `TresPerJob=gres/gpu:TYPE:COUNT`.
  Compound, untyped, CPU-only and zero-GPU descriptions remain blocked.
- `ReqTRES` has exactly that typed GPU total and an equal explicit generic
  `gres/gpu` total; duplicate fields, other GPU types and malformed totals fail.
- No other `TresPerNode`, `TresPerTask` or `TresPerSocket` field requests GPUs.
  Mixing request modes is rejected rather than assigned an assumed precedence.
- Job ID, name, owner, QoS and live state match the queue observation. A changed
  state, missing/disappeared job, conflicting duplicate or diagnostic error
  blocks the snapshot.
- Pending jobs may have no allocation yet, but their full verified requested
  total counts. A real allocation requires matching generic and typed GPU
  totals not exceeding the request. Non-pending jobs cannot lack allocation
  evidence.

Existing typed per-node rows retain their full node-multiplied request counts
and `scontrol` cross-checks. V3 additionally rejects an available generic
allocation total that differs from the typed allocation total. Unknown ledger
requests are never removed merely because they are absent from the queue;
exact overlaps count once and conflicting overlaps fail closed.

The three separate limits remain experiment running + pending + unknown at
most 6 physical GPUs, shared target QoS at most 8 PRO6000 GPUs, and supplied
owner at most 5 target-QoS submitted jobs. Other-QoS requests remain visible but
do not consume target-QoS limits; experiment jobs in other QoS still consume
the experiment limit. The audited ledger must be at most 15 minutes old.

V3 makes the already-authorized future exclusion explicit:
`--exclude=gpu-pro6000-6,gpu-pro6000-11,gpu-pro6000-12`.
The remaining planned request stays one node, 2 PRO6000, 8 CPUs, 66 GiB and
48 hours. This does not establish that every other node is Server Edition.
Model, FP32 precision, seeds, original requests, 24 x 128 scientific budget,
CPU-offload prohibition and production acceptance are unchanged.

Run on the scheduler host:

```sh
python -m scheduling.placement_v3 --owner varun024 --unresolved-ledger AUDIT.json
```

`READY` (exit 0) is a point-in-time plan for one additional job, never a
submission or training acceptance. `BLOCKED_CAPACITY` exits 1; malformed,
unsupported, missing or changing input exits 2 with `BLOCKED_INPUT`.
Every report says `submission_performed: false`. Before an authorized submit,
audit all saved BEGIN/RESULT records, prior jobs, directories and unique keys;
refresh the ledger and rerun the planner immediately before the single guarded
submission. Record the v3 source hash and scheduling commit separately from the
unchanged production commit. Do not reuse historical submission helpers.

For deterministic callers, `parse_queue` and `plan` accept a `job_details`
mapping from job ID strings to raw `scontrol` text. If supplied, every relevant
row must have matching detail, including typed rows. Calls without details can
only calculate the legacy typed-row arithmetic; they cannot resolve `N/A`.
Only the CLI performs live scheduler reads.

Validation from `sr_ord_train/`:

```sh
python -B -m unittest discover -s adapter_tests -p 'test_placement_v*.py' -v
```
