# Read-only placement plan v1

`placement_v1.py` plans **one future** two-GPU job. It never submits, cancels,
restarts, edits a production file, or imports the model runtime. Healthy running
jobs keep their current execution version and node. Its output is a point-in-time
capacity observation, not a submission receipt or an experiment acceptance.

The planned resources remain one node, 2 PRO6000, 8 CPUs, 66 GiB, and 48 hours;
production version remains `f594e32705581d8bcfdd609560eda87fd11cbc31`.
Model, precision, seed, samples, arithmetic, and scientific budget are unchanged.
The sole placement change is `--exclude=gpu-pro6000-12` for future jobs.

Observed slower throughput and Max-Q hardware identity on node 12 are an
association, not a paired causal experiment. This exclusion does not establish
that all other nodes are Server Edition, or prove power throttling. Each future
job still requires an actual hardware receipt and normal numerical/identity
acceptance. No CPU activation transfer is introduced.

## Capacity accounting

The CLI reads `squeue --all --states=all --noheader --format='%i|%u|%q|%j|%T|%b|%D'`.
It removes inherited `SQUEUE_*` filters from a copied subprocess environment;
the caller environment is unchanged. It then reads `scontrol show job` for
every relevant queue job and cross-checks job/name/owner/QoS/state, fixed `NumNodes`, and
typed plus generic GPU totals in `ReqTRES`. If `AllocTRES` is present, typed and
generic allocated GPU totals must not exceed the counted request. Node ranges, resource discrepancies,
missing details, or terminal rows fail closed. A disappearing job during these
reads blocks the snapshot; it is not silently removed. Only fixed node counts
are supported: a pending job's minimum-node display is not sufficient evidence.
For `PENDING` only, `AllocTRES=(null)` or `N/A` means allocation has not happened;
the full verified `ReqTRES` request still counts toward every applicable cap.
Non-pending jobs with missing, null, or malformed allocation totals are rejected.
After checking basic fields, it parses rows belonging to the target QoS, supplied
owner, or `LH-ORD-` experiment. Unrelated users' other-QoS arrays/heterogeneous
requests cannot affect these limits and are ignored. Other QoS reporting covers
the current owner and experiment; this is not a global all-QoS inventory.
It includes every recognized live state, including pending and suspended jobs.
Typed GPU TRES must be exactly `gres/gpu:TYPE:COUNT`; COUNT is **per node** and
is multiplied by `%D`. `%b` only describes TRES per node: `(null)` or `N/A`
does not prove a CPU-only request because GPUs may have been requested as a job
total. These values on relevant rows are therefore rejected. Independently
verified ReqTRES would be needed to accept zero GPU requests; v1 does not perform
that extra query. Untyped, compound, unknown, truncated,
array, or heterogeneous forms are rejected rather than guessed. Numeric
expanded job IDs are required. Identical duplicates are counted once; conflicting
state, identity, or resource rows are rejected.

Three separate checks apply to the hypothetical new job:

* All `LH-ORD-` requests across users/QoS, including unknown requests: at most 6 GPUs.
* All users in the target QoS: at most 8 PRO6000 GPUs.
* The supplied owner in the target QoS: at most 5 submitted jobs.

Other QoS requests remain visible in the report but do not consume target QoS
limits. Experiment requests in another QoS still consume the experiment's six
GPU cap. Jobs owned by others are never modified.

## Required audited unknown-request ledger

Queue absence is **not** a terminal-state receipt. Before each plan, audit saved
submission BEGIN/RESULT records and prior requests against current scheduler
accounting. Include every request whose release is unresolved. Only remove one
after a confirmed terminal-state audit; the CLI itself cannot prove this audit
complete. Never create an empty ledger merely because a job disappeared from
squeue. An empty list also requires the audit timestamp and basis.
The audit must be no more than 15 minutes old and cannot be future-dated; stale
or future ledgers block planning even if the unresolved list is empty.

Exact JSON schema (no extra fields):

```json
{
  "at_utc": "2026-11-05T13:00:00Z",
  "basis": "Audited submission BEGIN/RESULT records and confirmed terminal receipts; job 123 release remains unknown.",
  "unresolved_requests": [
    {
      "job_id": "123",
      "owner": "cluster_user",
      "qos": "soujanya-poria-startfund-2026-03",
      "name": "LH-ORD-EXAMPLE",
      "gpu_type": "pro6000",
      "gpus_per_node": 2,
      "nodes": 1
    }
  ]
}
```

An unknown job also present in the queue is counted only once, after exact
identity/resource matching. UNKNOWN cannot silently be treated as zero. Keep the
ledger and output with local execution evidence; they are not code for GitHub.

Run on a host with read-only access to the scheduler:

```sh
python3 -m scheduling.placement_v1 --owner cluster_user --unresolved-ledger /path/to/audited-ledger.json
```

The CLI only invokes read-only `squeue` and `scontrol show job` commands.
`READY` (exit 0) means this snapshot fits the limits;
`BLOCKED_CAPACITY` (exit 1) means it does not. Invalid input, command error,
diagnostic stderr, or timeout returns `BLOCKED_INPUT` (exit 2), never READY.
Every report says `submission_performed: false`. `planned_resource_args` is
displayed even when blocked and is **not** an executable submission command.
Any separately authorized submitter must recheck fresh scheduler state, audit
the ledger, use unique submission guards, and validate hardware/production
acceptance. No original bound production/diagnostic source needs modification.

Validation from `sr_ord_train/`:

```sh
python3 -B -m unittest discover -s adapter_tests -p 'test_placement_v1.py' -v
```
