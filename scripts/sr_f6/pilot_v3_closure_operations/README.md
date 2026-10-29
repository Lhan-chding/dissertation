# Corrected-closure PILOT_V3 CPU operations

This independent operation directory consumes the actual ten-file corrected
closure delivery from job 233920. It preserves the old UNKNOWN admission and
does not modify or invoke the old `pilot_v3_operations` scripts. It never submits
a GPU job, calls a model, or retries partial preparation.

Before activation, the operator must supply actual private records in
`.local/pilot_v3_closure_20261029/`:

- `CODE_REVISION.json`: the complete 40-character committed producer `revision`
  and actual canonical `deployment` (`sr_f6_code_<first7>`). The old 763 producer
  is rejected. Create this record after the new producer is committed and pushed,
  with its intended canonical deployment. This permits source-only deployment;
  activate the preparation request only after deployment and module-origin checks
  have succeeded. The submission and CPU boundaries verify actual module origins.
- `ORIGINAL_SOURCE_MANIFEST.json`, `EXPECTED_SOURCE_MANIFEST.json`,
  `SOURCE_CHANGE_REASONS.json`, and `EXPECTED_DELTA.json`: the original S0 source
  inventory and the complete new committed inventory with explained differences.
- `PREPARE_REFERENCE_INPUTS.json`: actual new revision/deployment, original root,
  original S0 `origin`/`upstream` references and all three original
  `parent_references`. Old confirmation receipt fields must be absent.
- The existing `.local/closure_repair_20261029/DELIVERY.json` and its exact ten
  delivered files. These are consumed in place; no second scientific evaluation
  or successful reinterpretation of the old UNKNOWN is allowed.

`build_prepare_submission.py --validate-only` performs local template/source
declaration checks and never creates a request. Actual activation verifies that
the selected producer is an ancestor of the existing origin branch, compares the
complete executable inventory with that commit's archive, and binds every local
operation byte used for rendering to that same inventory. It validates the real
corrected closure and exclusively creates `submit_prepare_remote.py` plus
`ACTIVATED_PREPARE_REQUEST.json`. Missing evidence or any partial request blocks
activation; no revision, hash, permission, or scientific result is synthesized.

`run_remote.py submit_prepare_remote.py` is the separate, explicit submission
step. The server verifies actual module locations, unchanged corrected closure,
terminal prerequisite jobs 233723/233849/233873/233920, absent old/new preparation
intents and phase roots, and current teacher QoS capacity. It uses the existing
8-GPU/5-slot checks including all-user reservations. A submission intent precedes
the single CPU submission; failed or unknown outcomes are retained without retry.

The CPU program runs the original native suite plus the new historical-closure
bridge suite, then registers actual source/original provenance, binds the
isolated corrected closure, freezes the fixed 384+96 public input overlay,
registers the nine paths, binds the shared numeric kernel, evaluates independent
PILOT_V3 admission and writes the two-worker plan. Only the real production gates
can produce `READY_FOR_REGISTERED_PILOT_ONLY`. Original S0 failure records and
DATA/CONFIG/SOURCE_BINDING remain unchanged. No model weights are rehashed here.
