# SSVC probability-flow diagnostics

This independent experiment implements the supplied 2026-09-05 Chinese plan through
the first NTU GPU boundary. Run commands **inside this directory**. Historical
experiment code is only audited as data; it is never imported by the new runtime.

The CPU implementation covers strict world parsing, independent one-error constraint
solving, exact answer fibers, deterministic charts/splits, finite-group mathematics,
hash-aware evidence storage, and the model-independent parts of P1. The real P1
Qwen3.5-9B smoke must run on an allocated NTU CUDA GPU. No large-model results are
claimed from the local tests.

See [the NTU handoff](docs/NTU_GPU_HANDOFF_zh.md),
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

`configs/locked.json` locks the proposed N protocol. Its unresolved revisions are
intentional: only P1 smoke can resolve them. The GPU bootstrap requirements are
candidates, and the actual P1 environment/model/processor/template must be recorded
before later experiments are implemented and authorized.

P3-P9 real-model execution is outside this delivery. The frozen/pilot/confirm
entrypoints reject execution with an explicit phase reason; mathematical tools
named after later audits remain CPU fixtures. Human inspection of the 36 generated
calibration charts remains a separately recorded requirement.
