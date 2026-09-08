# SSVC probability-flow diagnostics

This independent experiment implements the supplied 2026-09-05 Chinese plan through
the P3 frozen-evaluation implementation and its explicit GPU gates. Run commands **inside this directory**. The L track loads only allowlisted pure parser/executor modules from the historical
experiment; it preserves their original prompt and parsing semantics.

The CPU implementation covers strict world parsing, independent one-error constraint
solving, exact answer fibers, deterministic charts/splits, finite-group mathematics,
hash-aware evidence storage, and the model-independent parts of P1. The real P1
Qwen3.5-9B smoke must run on an allocated NTU CUDA GPU. No large-model results are
claimed from the local tests.

Real NTU P1 passed on 2026-09-08 (job `144911`, source `73f744b`): all 352
sampling/likelihood checks passed, with two real updates and bitwise-equal resume
replays. The linked run record also documents a subsequent telemetry-only counter
fix; the original GPU evidence is preserved with its exact source revision.

See [the NTU handoff](docs/NTU_GPU_HANDOFF_zh.md),
[the P1 failure diagnosis and rerun evidence](docs/P1_FAILURE_144840_zh.md),
[implementation boundaries](docs/IMPLEMENTATION_STATUS_zh.md), and
[the unchanged input specification](docs/CODEX_EXPERIMENT_PLAN_zh.md).

```bash
python -m pytest tests --cov=src --cov-report=term-missing -q
python -m src.audit_legacy --root .. --out runs/P0
python -m src.generate_worlds --out data/generated --seed 17
python -m src.tabular_flow --config configs/locked.json --out runs/P2
python -m src.rollout --phase smoke --model qwen35_9b --dry-run --out runs/P1_budget
python -m src.report --run-root runs --out reports
```

`configs/locked.json` locks the proposed N protocol and the Qwen3.5-9B revision
used in the first real P1 run. `configs/frozen.json` also pins 3B and 7B revisions;
each requires its own real P1 compatibility evidence.

P3 L/N frozen execution, interruption recovery, metrics, and verified matrix reporting
are implemented; see [the P3 handoff](docs/P3_FROZEN_HANDOFF_zh.md). No P3 GPU bank
has been run as part of this implementation. Human inspection of the 36 calibration
charts is still required for N. P4-P9 real-model execution remains unimplemented.
