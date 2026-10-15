# SR-F3-CE implementation

This module implements the user-supplied SR-F3-CE-20261016 contract. The original private package, rendered images, raw trajectories, model checkpoints, and execution receipts live under the isolated run root and are excluded from Git. Scientific completion is established by released raw-evidence receipts, never by CPU tests or a submitted job.

The fixed matrix has three J6 parents, four arms (RL, GOLD, PREFIX, EXEC), and 64 updates per path. Each update uses 16 questions with eight actual student samples each. GOLD/PREFIX/EXEC add four separate support sequences; PREFIX and EXEC have identical teacher text and answer masks. The runtime uses the existing native Qwen base, processor, tokenizer, and eager backend. Every server job name ends in `-LH`.

## Stages and entry points

1. `audit_parents.py` authenticates historical registration, committed checkpoint state, all parent adapter tensors, base files, and dependency versions in a CPU Slurm allocation.
2. `prepare_data.py` validates the supplied package and independently checks semantics, historical world deduplication, and rendered image inputs. `audit_native.py` checks the real processor, image readability, hidden-sidecar isolation, complete native teacher token offsets, and field masks without loading model weights.
3. `export_prior.py` preserves available historical ChartQA and diagnostic originals with provenance. Missing noncritical historical evidence is reported without generating replacements. `endpoint_runner.prepare_external_panels` freezes official external inputs and conservative gold-blind type registries.
4. `submit.py --kind p0` performs acceptance against the deployed source identity. `--kind engine` runs the three finite protocol panels and the bounded recovery check. Four continuous ENGINE updates plus two replay updates use 512 new trajectories; gradient probes use cached trajectories and zero Adam updates.
5. `freeze_science.py` binds successful ENGINE jobs and all source/input identities before scientific work. `submit.py --kind tick` dispatches only fixed ready tasks. Twelve training paths finish before the fifteen sealed endpoint evaluations. Release requires all task artifacts and actual Slurm terminal states.

`run_worker.py` is an internal allocation-bound entry point. Each held submission has an immutable intent, a verified actual batch script/allocation, and an explicit release receipt. The scheduler counts actual, pending, and unknown teacher QoS allocations toward the five-GPU limit. CPU QoS must be supplied explicitly after authorization has been verified. No unrelated job is cancelled or changed.

## Recovery and evidence

Every eight scientific updates, atomic checkpoints retain trainable parameters, Adam, scheduler, reference parameters, RNG, source identities, task/support cursors, and committed raw-record hashes. Restore verifies the original records and completes a due monitor with the exact corresponding weights before another update. Invalid answers remain observations; missing technical trajectories are incomplete work.

`submit.py --kind resume --job-key OLD_KEY` is restricted to verified single-GPU TIMEOUT/PREEMPTED attempts. It requires an independently verified checkpoint/cache receipt and creates a new attempt while retaining the old failure. A partial generation group requires an explicit identity-bound replay authorization; identical inputs, weights, seeds and existing token/probability trajectories are checked. ENGINE replay cannot silently exceed its finite generation/update budget. Unknown submission outcomes are never blindly retried.

Raw answers, exact prompt/image identities, token IDs, sampling log probabilities, actual termination, field-mask audits, and technical incidents remain available. New TEST scores, including parent baseline scores, are sealed until release. Statistical reporting keeps repair learning, unseen composition, retention, protocol sensitivity, external regression, uncertainty, and technical state separate.

## Local validation

From the repository root with its existing environment:

```sh
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/sr_f3_ce --import-mode=importlib -q
python -m ruff check --no-cache src/sr_f3_ce scripts/sr_f3_ce tests/sr_f3_ce
```

Tests needing the private package use its isolated local artifact directory. Real-model acceptance still requires the server processor and CUDA ENGINE receipts. Deploy a complete committed Git archive, including `docs/sr_f1/package`, because inherited modules authenticate reference code outside `src/` at import time. Never replace existing deployed sources after their identity has been frozen; record any technical revision and its impact separately.
