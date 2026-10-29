# Single Round A operation

This standalone source is outside the B7 production inventory. It preserves B7
and all existing phase/spec/admission/plan files. It authorizes one outer Round A
submission only; no resume, Round B, evaluation, new registration or model call
is implemented here. Two independent workers run the unchanged B7 pair launcher.

After committing and pushing this operation source, deploy the two executable
files at their repository-relative paths beneath the actual canonical
`sr_f6_pilot_ops_code_<first7>` directory. Generate `OPERATION_BINDING.json` there:

- `schema`: `srf6-pilot-round-a-operations-v1`
- `revision`: complete actual operation commit
- `deployment`: actual canonical operation deployment
- `producer_revision`: `b7e7b067f237b360b1dfb2c3075d3f14be7c281f`
- `files`: exactly `scripts/sr_f6_pilot_ops/round_a.py` and
  `scripts/sr_f6_pilot_ops/round_a.sbatch`, each with actual `bytes` and `sha256`.

Bind the SHA256 of that file in the private request. Invoke the deployed Python
entry with `submit --revision <actual> --deployment <actual>
--binding-sha256 <actual>`, using the existing Python 3.12 with `-I -B`.
Do not run a current checkout with a substituted deployment claim. Missing or
unpublished operation identity must be resolved before creating that request.

The server checks the actual completed preparation, fixed sealed metadata,
absence of all prior worker/task/trainer output, unchanged B7 scheduler source,
and live teacher capacity. A durable exclusive intent precedes sbatch. Submitted,
failed and unknown outcomes are preserved; any prior or partial operation blocks
repetition. The submission requests exactly two PRO6000 GPUs, eight CPUs, 66GiB,
23 hours, one node, and no requeue. Allocation startup verifies actual Slurm
resources and the unique submission closure before establishing project-local
per-job caches and executing B7. Full scientific admission remains enforced by
the unchanged production workers within the GPU allocation.
