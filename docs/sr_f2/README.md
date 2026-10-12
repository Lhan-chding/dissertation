# SR-F2 implementation

This module extends the preserved SR-F1.2 runtime with v2 answer parsing,
sufficient-certificate scores, four registered advantage routes, isolated online
measurements, historical-checkpoint evaluation and two diagnostic panels.
Old outputs and their official scores are read-only inputs.

## Execution and verification

1. Run `scripts/sr_f2/audit_prior.py --phase inventory` on a CPU allocation to
   verify the historical checkpoints, training records and endpoint source chain.
2. Deploy a committed source snapshot with a file-level SHA-256 manifest. Supply
   the private plan and clarification records separately; they are deliberately
   excluded from this public repository.
3. Use `scripts/sr_f2/submit_cpu.py --kind preflight` to copy verified inputs,
   classify answer syntax without gold answers, rescore old endpoints, verify the
   specified ChartQA diagnostic cohort and execute CPU acceptance checks.
   Registration binds source, configuration and evidence hashes before GPU work.
4. Use the same CPU submitter with `--kind controller` after registration. The
   controller runs historical evaluations, ENGINE recovery pilots, the fixed
   twelve-path training matrix and sealed endpoints. Four persistent two-GPU
   pools share a task queue. Each free GPU immediately takes a task whose
   dependencies are complete; endpoints depend on their own completed training
   path, and there is no slowest-branch seed-wave barrier. Each worker runs one
   independent single-GPU process. Capacity includes actual, pending and unknown teacher-QoS
   allocations; other jobs are never cancelled to reclaim capacity.
5. `scripts/sr_f2/release_report.py` requires complete record coverage and terminal
   successful Slurm accounting before releasing new scores and the final report.
   Training, endpoint and delivery completion remain distinct states.

All commands take an explicit independent `--root`. Submission receipts are
written before release, ambiguous submission outcomes fail closed, and raw
technical failures are retained. A stopped or failed run is not an invitation to
resample an existing slot or alter the experiment matrix.

The private deployment requires `docs/sr_f2/CLAUDE_SR_F2_PLAN_zh.md`,
`docs/sr_f2/DECISIONS.json` and
`docs/sr_f2/CHARTQA_122_USER_CLARIFICATION.json`. Run-specific data, receipts,
original documents and clarification text are not public code artifacts.

## Local checks

Run the SR-F2 and inherited SR-F1.2 suites separately because their historical
test module names overlap:

```sh
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/sr_f2 -q
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/sr_f12 -q
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/sr_f1/test_contract_boundary.py -q
ruff check --no-cache src/sr_f2 scripts/sr_f2 tests/sr_f2
```

The cohort verifier distinguishes the supplied post-hoc whitespace sensitivity
selection from the formal score. Its PASS requires exact source-backed membership;
candidate loss sets cannot silently replace the requested set.
