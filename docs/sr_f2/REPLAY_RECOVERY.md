# Explicit one-time replay after interrupted generation

`sr_f2_replay` is an operational extension for an explicitly authorized technical
exception. It does not alter the frozen source, registration, scientific matrix,
first recovery audit, or historical authorization. A new private plan binds the
original recovery, separate user authorization, source manifest, and exact
interrupted groups. Run-specific identities and raw evidence are not published.

Each authorized group retains its original model, checkpoint, inputs, sampling
seed and K=4. A durable exclusive attempt record is written before generation;
partial replay, identity mismatch or token mismatch stops the task without an
automatic second generation attempt. Complete staged outputs can be finalized
without generating again after a publication interruption.

The exception covers five groups: twenty physical replay responses supply
seventeen missing canonical responses and three duplicate responses used only
for verification. The existing three records must match their corresponding
replay tokens element by element and retain their original record objects in
the canonical group. The original partial file is archived byte for byte, with
its original path and checksum recorded, before canonical publication. Original
complete groups and attempt logs remain intact. The frozen endpoint worker then
continues previously unattempted groups with its original accounting ledger.

Replay generation uses a separate ledger. Final accounting reconciles original
attempts, original retained rows, replay rows, duplicate verification rows, and
canonical scientific rows. Replay cost is reported separately; unpersisted work
from the interrupted original call remains unknown rather than being assigned
zero cost.

Replacement GPU pools use two, two and one GPUs and share the five endpoint
tasks. Each pool may replace its corresponding first-recovery pool only after
that parent has naturally drained, disappeared from the queue, and has verified
successful terminal accounting. Other pools continue independently. Running,
pending and uncertain reservations from both stages count against the same
five-GPU experiment cap, shared QoS limit and user submission limit. The CPU
dispatcher requires a bound allocation identity and does not silently restart.

The composite release gate checks the original 53 tasks across the disjoint
inherited, first-recovery and explicit-replay task sets. It verifies original
cancelled allocation identities, every actual recovery allocation's terminal
state, worker claim/launch/exit lineage, frozen artifacts, replay provenance and
raw accounting before invoking unchanged scientific reporting functions. An
intermediate 48/53 completion with five historically blocked entries remains an
honest first-recovery record; it is not rewritten to pass the original gate.

Entry points are `scripts/sr_f2/replay_endpoint.py`,
`scripts/sr_f2/replay_recovery.py` and `scripts/sr_f2/release_replay.py`.
Deploy only the new replay package in its extension directory. Import precedence
must be new replay extension, bound first-recovery extension, then frozen source,
so that the original recovery module's source identity remains verifiable.

Validation:

```sh
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/sr_f2_replay tests/sr_f2_recovery --import-mode=importlib -q
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/sr_f2 tests/sr_f12 tests/sr_f1/test_contract_boundary.py --import-mode=importlib -q
ruff check --no-cache src/sr_f2_replay scripts/sr_f2/replay_endpoint.py scripts/sr_f2/replay_recovery.py scripts/sr_f2/release_replay.py tests/sr_f2_replay
```
