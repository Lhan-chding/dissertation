# Placement v4: positively verified CPU-only jobs

V4 is a new read-only planner. V1, V2, V3, all production sources and their
bindings remain unchanged. It retains V3's verified typed per-node and
single-node job-total GPU paths and adds one narrowly defined case: an owner or
target-QoS CPU-only job whose complete scheduler evidence proves zero GPUs.
An `N/A` or `(null)` per-node GPU field alone still proves nothing.

## CPU-only evidence gate

For an absent per-node GPU field, all these conditions must hold before the
request can be represented as `gpu_type: none`, `gpus_per_node: 0`:

- The same positive numeric job ID, owner, name, QoS and live state appear in
  `squeue` and `scontrol`. Changed, terminal, missing or ambiguous observations
  fail closed. Array and heterogeneous job IDs remain unsupported.
- The queue and `NumNodes` establish one fixed node (`1` or `1-1` in the latter).
  `ReqTRES` independently contains positive `cpu` and exactly `node=1`.
  An available `NumCPUs` must equal the requested CPU total.
- Every `ReqTRES` and real `AllocTRES` component is well formed and unique.
  Only `cpu`, `mem`, `node` and `billing` are accepted for this CPU-only proof;
  GPU keys, zero GPU declarations, other GRES and unknown resource semantics
  are not silently discarded. CPU, node and billing quantities are positive
  integers. Memory is a positive integer with an optional binary `K/M/G/T/P/E`
  unit. Unsupported representations remain blocked.
- Each of `TresPerJob`, `TresPerNode`, `TresPerTask` and `TresPerSocket` is
  checked, including duplicate fields. It may be absent or explicitly `(null)`;
  otherwise every comma-separated component must be an accepted CPU resource
  in colon form with a value supported by, and no greater than, `ReqTRES`.
  `N/A`, malformed components, GPU requests and unknown resources fail.
  Any unrecognized top-level TRES/GRES field also blocks this narrow proof.
- Pending jobs may have no `AllocTRES` yet, or explicit `(null)`. Any real
  allocation must independently contain positive CPU and exactly one node,
  with every allocated resource present in and no greater than `ReqTRES`.
  Non-pending jobs cannot lack allocation evidence. `N/A` allocation is
  ambiguous and remains blocked.
- An `LH-ORD-` experiment job cannot use this CPU-only classification.
  Unresolved-ledger entries still require positive GPU requests; absent live
  observations never turn an unknown request into zero.

The CPU-only row is retained in `requests`, the full raw scheduler observation
and the other-QoS inventory. Its `resource_basis` is
`SCONTROL_VERIFIED_CPU_ONLY_ZERO_GPU`. A CPU-only target-QoS job still consumes
the owner's target-QoS submitted-job slot, despite consuming zero GPUs. The
proof does not authorize cancelling, modifying or omitting unrelated jobs.

## Unchanged admission limits

Experiment running + pending + unknown requests remain limited to 6 physical
GPUs, shared target QoS to 8 PRO6000 requests, and the owner to 5 target-QoS
submitted jobs. Pending GPU requests count in full; unresolved requests count
conservatively and deduplicate only by matching job identity/resources. The
ledger must have a documented audit no older than 15 minutes.

The hypothetical new request remains 1 node, 2 PRO6000, 8 CPUs, 66 GiB and
48 hours, excluding `gpu-pro6000-6,gpu-pro6000-11,gpu-pro6000-12`.
Production remains `f594e32705581d8bcfdd609560eda87fd11cbc31`. This change does
not modify the model, FP32 arithmetic, seeds, inputs or scientific budget.

```sh
python -m scheduling.placement_v4 --owner varun024 --unresolved-ledger AUDIT.json
```

The CLI reads the full queue and `scontrol` for every relevant owner, target-QoS
or experiment row. It never invokes a submission command. Reports use schema
`SR_ORD_PLACEMENT_V4`; `READY` exits 0, `BLOCKED_CAPACITY` exits 1 and
`BLOCKED_INPUT` exits 2. Available raw queue/details are retained even when
input validation fails. A `READY` report is only a point-in-time hypothetical
capacity calculation. Separately authorized submissions still require their
unique submission audit and an immediate fresh gate. A failed gate is not
permission to filter queue rows or reuse a historical submit helper.

Pure `parse_queue`/`plan` APIs retain V3 signatures. Without `job_details`, only
the legacy typed per-node arithmetic is available; CPU-only and absent
per-node GPU rows require independent detail. Execution ledgers and reports
remain local evidence and must not be uploaded with code.

Run validation from `sr_ord_train/`:

```sh
python -B -m unittest discover -s adapter_tests -p 'test_placement_v*.py' -v
```
