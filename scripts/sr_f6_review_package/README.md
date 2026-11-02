# Saved SR-F6 pilot analysis package

This utility exports already-completed evidence for external GPT/Claude analysis. It does not
import model/scoring code, generate answers, submit jobs, or assert a scientific gate.
All artifacts and transport receipts stay under `.local/review_package_20261102/` (ignored by Git).

`export_saved.py` runs on the server through SSH stdin and emits a ZIP on stdout. Fixed sections
are `data`, `b0`, `b1`, `b2`, `train_b0`, `train_b1`, `train_b2`. Use `eee-cluster`, BatchMode,
a 12-second connection timeout and a 50-second local subprocess timeout. Save stdout only after
successful exit and ZIP CRC validation. The exporter writes no server files. DEV exports retain
all answer text and saved scores while listing omitted raw fields; training exports preserve
all original credit and UPDATE JSON bytes. Input images are checked against original hashes.

`build.py` consumes the seven section ZIPs, a read-only `source_export.zip` containing the frozen
b7 `src/sr_f6/*.py` and DEVv2 wrapper source, and the four private editorial Markdown files.
Those sources are indexed with original SHA256 values; they are not current working-copy
substitutions. Existing local saved summary/completion, historical receipts and the matching
frozen config are also required. The fixed output staging directory must not already exist.
Run with the project's Python environment from the SR-F6 worktree.

The builder checks 2304 DEV records against original endpoint/summary metrics and 13824 primary
training scored rows across 108 updates. It creates descriptive paired-discordance and
parent/family/root/credit tables from saved labels without rescoring or inference tests.
Some UPDATE channel-gradient diagnostics are absent by design; CSV nulls preserve that status.
Shared-account queue/process snapshots are excluded from the B closure projection.

Publication checks include all source byte hashes for unmodified included files, input-image
hashes, answer/input correspondence, internal links, ZIP CRC, safe unique names, manifest and
checksum consistency. An atomic final ZIP and external verification/checksum receipts are
written only after checks pass. This is a compact evidence projection, not a retraining bundle.

Actual delivery: 12 DEV endpoints, nine 12-update primary paths, one separately registered
recovery. Original full-delivery audit was waived by the user; `NOT_SIGNED` and FORMAL/TEST
not executed remain explicit. Historical full-design documents are not executed-pilot results.
