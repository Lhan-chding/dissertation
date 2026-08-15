# CompBias

CompBias is an auditable research implementation for *Fixing the Answer, Not the
Error: Compensability Bias and Spurious Compensation Equilibria in Multimodal
Reinforcement Learning*. The repository asks whether outcome-only reinforcement
learning repairs local errors or instead selects perception and reasoning errors
that cancel at the final answer.

The code is a diagnostic test bed, not a proposed model architecture. Exact,
tabular, small-CPU, and controlled-data stages must pass in order before any
large-model work is considered.

The primary revision is now the v2 natural-trajectory and partial-identification
protocol in `paper/theorem_notes_v2.md`, `paper/identification_contract.md`,
`paper/experiment_registry_v2.md`, `paper/claim_evidence_matrix_v2.md`, and
`paper/related_work_audit_v2.md`.
The original controlled mechanism remains available as bounded v1 evidence.

The v2 originality audit deliberately rejects a broad "first to separate
perception and reasoning" claim. Closest 2025--2026 work already studies
outcome-reward coupling, perception-aware rewards, blindfolded reasoning, and
alternating perception/reasoning optimization. The narrower candidate
contribution is the combined natural-selection/fork-replay/synthetic-transport
and multi-interface partial-identification framework; it remains conditional on
real-VLM evidence and a submission-time literature refresh.

## Public repository and RTX 4090 pilot

This repository is the code-and-compact-evidence release. It deliberately
excludes model weights, generated images, trajectories, activations, KV caches,
optimizer states, checkpoints, tokens, and machine-private execution evidence.
The research overview is in `docs/RESEARCH_QUESTION.md`; the server runbook and
ordered pilot are in `docs/SERVER_SETUP.md` and `docs/GPU_PILOT_PROTOCOL.md`.

The single registered Qwen2.5-VL-3B-Instruct v0.3 calibration on the validated
RTX 4090 server is complete and failed its preregistered gate. It produced 200
responses, accuracy `0.355`, parse rate `0.935`, perception-error rate `0.225`,
and only one strict compensated visual error. Pilot A/B were terminated without
training, and v0.3 must not be rerun as a confirmatory attempt. The frozen
negative record is in `configs/recoverability/v0_3_negative_pilot.yaml`.

The subsequent one-shot recoverability measurement Bridge v1 also completed.
Its legacy path remained operational, but strict Stage 1 parsed `0/300`, so
Stage 2 was never called. This is frozen as an interface failure rather than a
test of the recoverability hypothesis; Bridge v1 must not be rerun. The next
target in `docs/RECOVERABILITY_V1_PROTOCOL.md` is only a 24-scene,
development-only Stage-1 v2 prompt probe. The model must still be read from:

```text
/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct
```

and all generated state must remain under:

```text
/cloud/cloud-ssd1/dissertation
```

The Phase N/C protocol remains unrun and unauthorized. The next GPU step makes
exactly 24 question-free Stage-1 image calls on the frozen `dev` split, with no
retries, Stage 2, hypothesis test, or training. Server execution requires an
explicit `--execute`; its separate metadata preflight cannot load the model.

Current explicit non-claims are:

- general VLM compensation has not been established;
- no unique perception/reasoning boundary has been recovered;
- frozen-vision gains cannot be called visual-acquisition improvement;
- synthetic errors cannot stand in for natural errors without a transport
  certificate.

## V2 result and current stop boundary

V2 separates natural selection success `c_sel`, image-cut forked natural-state
success `c_fork`, and artificial-injection success `c_syn`. A clean CPU
confirmatory run used 1,000 semantic states, 8,000 rendered images, 640,000
natural mediators, 20,480,000 forked continuations, 20,000 synthetic mediators,
20 seeds, and five error families. `c_sel` and `c_fork` were close in magnitude,
whereas `c_syn-c_fork` was between about -0.071 and -0.073 by family and every
paired 95% interval excluded zero. This is direct controlled evidence that an
injected error need not behave like a naturally produced error; it is not a
Qwen/VLM result.

The five logical Stage-B operations are deliberately executed by one atomic
runner so that mediator IDs, fork seeds, estimates, and manifest hashes cannot
drift across separate invocations:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/run_small_natural_replay.py \
  --config configs/natural_replay/small_neural.yaml
```

Its tracked publishable summary is
`artifacts/metrics/small_neural_natural_replay_v2.json`, SHA-256
`809e93239126d38ea6f8d4414a6593826a08b113b127f53728ef24ed853fa8ac`.
The larger mediator/fork/transport tables remain ignored local evidence.

The four Qwen2.5-VL-3B regimes are now frozen as private execution plans:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/train_vlm_rl_regime.py \
  --config configs/frozen_regimes/lm_only.yaml
PYTHONPATH=.:src .venv/bin/python scripts/train_vlm_rl_regime.py \
  --config configs/frozen_regimes/projector_lm.yaml
PYTHONPATH=.:src .venv/bin/python scripts/train_vlm_rl_regime.py \
  --config configs/end_to_end/vision_lora.yaml
PYTHONPATH=.:src .venv/bin/python scripts/train_vlm_rl_regime.py \
  --config configs/end_to_end/max_end_to_end.yaml
```

All four commands intentionally return `BLOCKED` with exit code 2 after writing
a private metadata plan. They do not import a training framework, download a
model, or start a GPU job. Planning minima are 24/32/48/80 GB respectively;
practical pilot recommendations are 48 GB, 48--80 GB, 80 GB, and two 80 GB
GPUs. Exact placement remains subject to target-machine CUDA/container smoke.

## Evidence status

Evidence labels used throughout `paper/` are deliberately strict:

- **CPU verified** means the registry accepted the canonical command and all
  provenance required for that row. A focused or unit test alone establishes
  implementation behavior only; it cannot promote a row. CPU verification is
  not a formal proof and does not establish behavior in a real VLM.
- **Not run** means that no result may be reported. A configuration or preflight
  plan is not an experiment.
- **Not claimable** marks a statement excluded by the research scope, even if a
  future experiment happens to be suggestive.

As of 2026-08-14 in Asia/Singapore (UTC+08), the registry accepts clean-HEAD
`VERIFIED_CPU` evidence for the whole-project test/coverage gate and Phases
A--C. Phase D is only `PARTIAL_GATE`: its local automatic audit is clean, but
human review/sign-off is absent and `phase_d_ready` remains false. The project
registry records no Qwen checkpoint download, Qwen training/evaluation,
large-GPU job start, or MeasureBench external-validation execution. See
`paper/experiment_registry.md` for run-level status and
`paper/claim_evidence_matrix.md` before using any result in prose.

## Install and verify

Python 3.10--3.13 is supported by the package metadata. The intended CPU
snapshot targets macOS with Python 3.12.13. `requirements-lock.txt` is an
intended exact-version dependency snapshot for that platform/interpreter only:
it has no artifact hashes, is not a cryptographic lock, and must not be
represented as portable across operating systems or Python versions. A final
environment audit found
that the existing `.venv` does not itself exactly match the file (`wheel` is
absent and the bootstrap `pip` version is not recorded), so the current
workspace is not evidence of a clean snapshot installation.

The current `.venv` does contain Pillow `12.3.0`, and `pip check` currently
reports no broken requirements. The package metadata requires
`Pillow>=12.3.0,<13`; this excludes both Pillow `>=10.3.0,<12.1.1`, affected by
[CVE-2026-25990](https://pillow.readthedocs.io/en/latest/releasenotes/12.1.1.html),
and Pillow `5.2.0` through `12.2.0`, affected by
[CVE-2026-59198](https://github.com/python-pillow/Pillow/security/advisories/GHSA-fj7v-r99m-22gq).
These two local checks establish the installed Pillow version and dependency
consistency only; they do not turn the whole environment into a hash-verified
or clean-install snapshot.

On 2026-08-14, an isolated temporary environment running `pip-audit==2.9.0`
scanned the exact versions listed in `requirements-lock.txt` with `--no-deps`
and reported no known vulnerabilities after the snapshot was updated to
`pytest==9.0.3`; the current `.venv` also reports that pytest version. This is a
time-bounded advisory-database check of the listed versions, not a clean-install
record, transitive-resolution proof, SBOM, or artifact-integrity attestation.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
PYTHONPATH=src .venv/bin/python -m pytest -q
COVERAGE_FILE=/private/tmp/.coverage-compbias \
  .venv/bin/coverage erase
PYTHONPATH=.:src COVERAGE_FILE=/private/tmp/.coverage-compbias \
  .venv/bin/coverage run -m pytest -q
COVERAGE_FILE=/private/tmp/.coverage-compbias \
  .venv/bin/coverage combine
COVERAGE_FILE=/private/tmp/.coverage-compbias \
  .venv/bin/coverage json -o artifacts/metrics/coverage.json
COVERAGE_FILE=/private/tmp/.coverage-compbias \
  .venv/bin/coverage report
```

The editable-install command intentionally disables dependency resolution and
build isolation after the snapshot is expected to install the build backends.
`pyproject.toml` pins those backends to `setuptools==84.0.0` and
`wheel==0.46.3`. Before treating a fresh environment as reproducible, verify
that both backends are actually present, explicitly record the installer
version (the snapshot does not pin `pip`), run `pip check`, compare the
installed inventory with the snapshot, and repeat
the editable-install/import smoke test. If `--no-build-isolation` is omitted,
normal PEP 517 isolation may
obtain even those exact-version artifacts from the configured index without
hash verification. For an offline or artifact-verified build, first supply and
verify local wheels for the snapshot and build backends. No `--require-hashes`
lock is currently shipped, and no clean-environment installation record or
SBOM exists. The isolated `pip-audit` result above does not close those residual
supply-chain limitations.

All commands below that execute repository code include `src` explicitly in
`PYTHONPATH` (`PYTHONPATH=.:src` also exposes the instrumented `scripts`
package during whole-project coverage). Keep this even after editable
installation so each published evidence command resolves the source tree
explicitly instead of depending on prior environment state.

The coverage configuration measures both `compbias` and `scripts`; the combined
total must remain at or above the 80% project threshold. The clean canonical
run at HEAD `84eda3bbaeaf4822f9cb238f3080ead3cd6c9c0b` completed 1,024 tests
with no failures/errors in 887.14 seconds. It covered 10,696/12,340 statements
(86.6774716%) and 3,379/4,708 branches (71.7714528%), for combined coverage
82.5610042% (83% display), and passed `fail_under=80`.
The publishable `artifacts/metrics/coverage_summary.json` has SHA-256
`815a588faba3392a96a5e15fe43674984af7957bca1ac4d6c0b44ca2355962cb`
and binds raw coverage SHA-256
`0cbda5472847078ccb44e625eed1b25a49982fcc218fc0f141d5554b78b3a9c0`.
The historical `compbias`-only record remains superseded; do not confuse it
with this whole-project result or mislabel the combined number as pure branch
coverage. A failed test, missing artifact, or dirty-worktree run blocks
promotion to the next phase.

## Gated workflow

### Phase A: theorem implementations

Run the focused theorem tests, then execute the frozen Phase-A configuration:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_selection_identity.py \
  tests/test_scaling_identity.py \
  tests/test_lockin_closed_form.py \
  tests/test_coordination_fixed_points.py \
  tests/test_bifurcation.py \
  tests/test_bregman_identity.py \
  tests/test_fisher_projection.py \
  tests/test_decomposition.py

PYTHONPATH=src .venv/bin/python scripts/verify_theory.py \
  --config configs/theory/all.yaml \
  --overwrite
```

Exit only when the registered randomized identities, fixed points, basin
boundary, bifurcation threshold, and decomposition checks pass their tolerances.
Numerical tests validate code against the stated identities; they do not make
the underlying classical tools novel.

The accepted clean run `run-ce509068302076dc` used config file SHA-256
`c24587a8a4191815553b2a26358a7cdff09d9bbcd2bfeccc30661cfd2692178b`
and passed 7,000 checks across seven identities (1,000 cases each).
`artifacts/theory_verification/random_property_tests.json` has SHA-256
`0977e4c93d642e286f12a47148c071956dc3a922b4cb257500df277293e6b653`.

### Phase B: exact and approximate tabular diagnostics

Run the frozen selection, approximate-update, reasoner-scaling, coordination,
and bifurcation suite together:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_tabular.py \
  --config configs/tabular/all.yaml \
  --overwrite
```

Proceed only if the exact optimizer matches the analytical distribution, mirror
descent error decreases with step size, approximate updates reproduce the
predicted mean direction, the three matched-average-gain scaling paths match
their covariance/finite-difference directions, and both coordination basins are
repeatable.
The formal approximate path, `raw_fixed_reasoner_outcome`, trains only an error
policy `mu_theta(e)`. After sampling `e`, the environment independently samples
the binary reward `r ~ Bernoulli(c[e])` from the frozen reasoner conditional;
the optimizer cannot select or change `P(r|e)`. REINFORCE, PPO-like, and
GRPO-like apply the KL penalty on the error marginal and compare their learned
policy with the exact reward-moment selection target. This is the registered
finite-policy raw-outcome protocol. The accepted canonical run supplies its
finite CPU evidence. The registered aggregate sampling audit requires every
error action to be observed and the empirical conditional success rates to lie
within `0.03` absolute error of the frozen `c[e]`. Even then, PPO/GRPO-like
remain tabular analogues, not neural-library fidelity claims or guarantees
about real VLM training.

Two additional views are diagnostics only. The
`unconstrained_joint_trajectory_diagnostic` optimizes a flattened policy over
`(e,r)` with joint-space KL and can change `P(r|e)`; it is explicitly ineligible
for the formal fixed-reasoner gate. The
`collapsed_effective_reward_diagnostic` instead optimizes the analytically
collapsed reward `beta * log(moment multiplier)` on the error marginal. It
measures the effect of analytic collapse and must not replace or be conflated
with the formal fixed-reasoner result. Mirror descent remains attached to this
deterministic collapsed target as an optimizer oracle.

The same command also runs the plan's tabular Experiment C scaling diagnostic.
`truth_gain`, `uniform_gain`, and `error_gain` have the same reference-policy
average gain but concentrate it on different severities. The gate compares
`Cov(ell,G)` with a central finite difference and requires negative, near-zero,
and positive severity directions respectively. Its independent outputs are
`artifacts/metrics/tabular_scaling.json` and
`artifacts/predictions/tabular_scaling.csv`. This is a controlled finite path,
not the still-pending matched real-checkpoint H3 experiment.

The accepted clean run `run-5f8370785b97da1d` used config file SHA-256
`bedfe03c315586c8502f5089ff8fd5e2c33a0c7937aa55422cbaa5045f23e145`
and passed all registered gates. Its JSON bindings are selection
`625d5b7894063d5e843f16454acf023116bda35dfdac9e1f125de96b68e40192`,
scaling `4ed720e1c7bde52e33d4625bec5da839397e1a69a252383b87c5234236a03548`,
coordination `4ea154f66656d66599b44de877a79962353aec52035f72c112236ecf5c6683d0`,
and bifurcation
`ade5660fd547a4c5b5e80f64c5644efbbb04042996a410434a82c1133ea6709b`.
The formal fixed-reasoner algorithms used 20 seeds each and passed the action
coverage/conditional-freeze audit; joint and collapsed views remain
diagnostic-only.

### Phase C: deterministic small-neural diagnostics

The scalar single-run YAML files (`flat_perception`, `spurious_perception`,
`truth_aligned_perception`, and the two `joint_*` fixtures) use only fields
consumed by `scripts/train_neural.py` and are convenient for isolated
reproduction. The formal scalar Phase-C gate is a 10-seed batch so that
direction, dual-endpoint, and OOD comparisons are emitted together:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_neural.py \
  --config configs/neural/all.yaml \
  --overwrite
```

For Phases A--C, `--overwrite` is an explicit authorization to replace only the
canonical generated outputs after the scripts validate their scoped artifact
paths. Omit it for a first run or whenever the existing evidence must remain
untouched.

This writes the summary, run table, trajectory table, figure, and per-run logs
under `artifacts/metrics`, `artifacts/predictions`, `artifacts/figures`, and
`artifacts/logs`. A single-value YAML invocation retains the legacy one-run JSON
behavior and is not, by itself, the formal Phase C gate.

The formal YAML sweep above is a two-sigmoid differentiable coordination
diagnostic. Its gate requires at least two correctly predicted perception-shift
directions, seed-dependent truthful and compensatory endpoints, and a larger
registered OOD loss for the compensatory endpoint. The `reasoning_only` rows
are explicit freeze controls: perception parameters must remain unchanged while
reasoning parameters update. It does not train the PIL/CNN path.

The accepted scalar sweep used config file SHA-256
`c82da0555b2a4ef91aed9bc42a92b9a27b354eb25d39c1b4163d1ca3cc4f9569`.
It passed all registered gates across 50 complete clean RunLogger bundles, 50
unique condition--seed combinations, and 2,450 checkpoints.
`artifacts/metrics/neural_summary.json` has SHA-256
`534e4de5a3c8bab2076a517c8ce6d1f55163d1564fa9753abb2b4d35be7339e0`.
These are results for the scalar two-sigmoid diagnostic only.

An independent, narrower image-path smoke experiment consumes its own strict
YAML contract and trains the convolutional perceiver on one fixed 16x16 RGB
scene rendered through CVA/PIL:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/compbias-mpl .venv/bin/python \
  scripts/train_visual_neural.py \
  --config configs/neural/visual_modular.yaml \
  --overwrite
```

This run exercises and updates convolutional parameters, but the IID/OOD pair
uses byte-identical image input while changing only the injected error
mechanism. It is a single-scene mechanism check, not evidence of multi-image CNN
generalization, natural visual-error learning, or Qwen2.5-VL behavior. Its
`reasoning_only` control must leave convolutional parameters fixed while the
reasoner changes.

The accepted 10-seed image-path run
`sweep-20260814T010824107927Z-dacc8f8dec6c` used config file SHA-256
`dacc8f8dec6c64678e8d598d5bf603422c43f965f9f997e64513de1f33a97771`.
`artifacts/metrics/visual_neural_summary.json` has SHA-256
`89a862da7e50215d62a43360235d41b4acf6b4d2db971242e40e06cf5d019419`.

### Phase D: CVA-World generation and audit

`configs/data/cva_v2.yaml` freezes the generator, renderer, output, and manifest
contract. Generation must precede audit:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_cva.py \
  --config configs/data/cva_v2.yaml \
  --overwrite

PYTHONPATH=src .venv/bin/python scripts/audit_dataset.py \
  --manifest artifacts/manifests/cva_v2.json \
  --output artifacts/metrics/cva_v2_audit.json \
  --report-root artifacts \
  --overwrite
```

`--overwrite` authorizes replacement only after the generation and audit
scripts validate any existing provenance-bound targets. Omit it when an
existing snapshot/report must be preserved for comparison.

The clean-HEAD local generation
`generation-20260814T010836108099Z-e0658f1a9850` and automatic audit
`audit-20260814T010913290056Z-6bcc090c91cc` completed on 2026-08-14. Their
bindings are:

- tracked config file SHA-256
  `e0658f1a9850d300d74eeb3e6c445f0ad8dcf702e846b1b1d11fb34e3f6b1071`;
- manifest file SHA-256
  `6bcc090c91cc7f47fa03309a2fb56bbd7e2b0e8a0b8bc8ca21b98ee7399c340a`
  and manifest self SHA-256
  `3f9efc0d0ec76284d7d5dc1a36d0c1b57ec736aa8c4c1b968447e022fca0c108`;
- canonical content SHA-256
  `5010137acc0cfc06238efbf23183aa865080d4882a9abcadfe6d8b365092ad36`,
  raw dataset SHA-256
  `d439f621c395a21cb5386f1d769b3fd53cf6195c7b2389a6df7e3fcaa43eaddc`,
  and image-set SHA-256
  `6a1171e8716cd93c644bf771dcd3a6f79eceb9c6cca5120f5812895e57999e4e`;
- local audit-file SHA-256
  `f6781e82052463f0e749864ced41b559a680604872f0de72b39fa2955dc8142f`.

The manifest-internal canonicalized-config hash is a different byte domain
from the tracked config file hash; the dataset card records both explicitly.
All generated D artifacts remain ignored/local and are not an approved release
or a `VERIFIED_CPU` JSON bundle.

Before constructing samples or writing files, the generator rejects plans over
10,000 images or 256,000,000 total render pixels. Contact sheets are assembled
in bounded 25-image batches and their Pillow images are closed after use; the
frozen v2 plan remains below both limits.

The audit command shown above intentionally omits `--visual-audit`. Its clean
local exit records only that the automatic checks passed; the report records
`automatic_audit_clean: true`, `human_reviewer_signoff: false`, and
`phase_d_ready: false`. Promotion requires rerunning with a separately bound
human record via `--visual-audit`, after which both the binding and sign-off
fields must pass.

Do not pass this gate without 100% canonical-solver self-checks, 100% registered
error round trips, a clean split audit, stable manifest hashes, the registered
visual-factor/answer-balance/OOD-image/style-semantic-independence/deterministic
replay checks (including contact-sheet replay), and a documented human review
of at least 200 rendered images. The human-review requirement
cannot be satisfied by unit tests or an agent-only visual review. The local v2
generation and automatic audit recorded 1,820 samples/images, 1,820 solver
checks, 4,020 error-solver/round-trip checks, 73 sheets, and 100 OOD pairs. The
prior 250-row v1 artifact is not accepted as current evidence. Phase D remains
incomplete until a human reviewer supplies a self-reported review bound to the
recorded local v2
manifest and image-set hashes, then reruns the audit with
`--visual-audit artifacts/manifests/cva_v2_human_review.json`. Do not relabel
any `cva_v2_visual_audit.json`: that filename is reserved for an agent-review
record with `human_reviewer_signoff: false`. The local generation/audit bundles
are stored under `artifacts/logs/cva_v2_generation/` and
`artifacts/logs/cva_v2_audit/`.

A human-review record may contain a review date, but its public reviewer field
must be a bounded pseudonym matching `reviewer-[a-z0-9][a-z0-9-]{0,30}`. The
audit rejects names, email addresses, and other values outside that form before
copying the pseudonym into its report/log. Any real identity, consent record,
and pseudonym mapping must remain private and outside generated artifacts.

The human record uses a closed schema: `schema_version`, `reviewer`,
`reviewer_type`, `review_date`, `review_result`, `human_reviewer_signoff`,
`images_reviewed`, `reviewed_sample_ids`, `contact_sheets_reviewed`,
`reviewed_contact_sheets`, `manifest_sha256`, and `image_set_sha256`. For the
frozen v2 snapshot, `reviewer_type` must be `human`, `review_result` must be
`pass`, the sample-ID list must contain at least 200 unique IDs from the bound
manifest and match `images_reviewed`, and the sheet list must cover every
manifest sheet with `contact_sheets_reviewed: 73`. The two hashes must bind the
recorded local manifest self-hash and complete image set. Unknown or missing fields,
partial sheet coverage, or a self-described agent review are rejected.
The audit checks this self-reported record's schema, pseudonym form, and hash
bindings; it does not authenticate the reviewer's real-world identity, verify
an external signature, or retain consent evidence.

Any stale untracked `cva_v1` manifest/data or
`artifacts/figures/cva_contact_sheet_*.png` files are historical local
by-products, not v2 release assets, and must not be published or cited as
current evidence.

The v2 configuration freezes a closed ten-style renderer catalog: baseline,
bold font weight, compact size, rotation, low contrast, grid background, local
occlusion, mild blur, distractor marks, and shifted layout. In each of the four
non-OOD splits, every semantic state is fully crossed with every style
applicable to its task family: nine IID styles for digit/gauge/bar and eight for
count/relation. Each OOD state instead has two independent `layout_shifted`
realizations. This frozen design targets 1,820 rows. Legacy
`font_a`/`font_b`/`rotated` names remain code aliases only and are absent from
the frozen v2 YAML. The local automatic audit recorded catalog coverage and
split independence; human review and release approval remain separate gates.

The clean local audit reproduced the frozen accounting contract of 250 semantic
groups and 1,820 rendered rows:
380 rows for each of digit/gauge/bar, 340 for each of count/relation, 430 in
each non-OOD split, and 100 in OOD. Expected style totals are 200 each for
`baseline`, `size_compact`, `rotation_tilted`, `contrast_low`,
`background_grid`, `occlusion_local`, `blur_mild`, and `distractor_marks`; 120
for `font_weight_bold`; and 100 for `layout_shifted`. At 25 images per sheet,
the audit recorded 73 contact sheets, 1,820 solver checks, and 4,020 registered
error-solver/round-trip checks.

The bar-chart family now freezes three executable operations in every split.
Among each split's ten bar semantic states, four use `sum`, three use
`difference`, and three use `ratio`. Full IID-style crossing therefore gives
36/27/27 bar rows per non-OOD split, while the two OOD realizations give
8/6/6 rows; over all five splits the totals are 152/114/114. Canonical bar
answers have disjoint support `13..22` for train, `23..32` for calibration,
`33..42` for validation, and the paired support `43..52` for IID/OOD. Ratio
answers are serialized as finite floats but are constructed as exact integers.
This schema change leaves the 250 semantic IDs, 1,820 rows, and 4,020
round-trip target unchanged. The bindings above come from the final local
canonical run rather than an earlier preview.

Implementation-level focused tests verify the 1,820-row crossing/count contract
and distinct applicable-style pixels on one representative scene per family.
The canonical local audit enforced distinct rendered hashes across all 250
semantic groups; no focused-test preview replaces that manifest evidence.

`font_weight_bold` changes weight/style with the Pillow-bundled font and a
fixed stroke; it never searches system/PATH fonts and is not a different
typeface. It is applicable only to the text-bearing digit, gauge, and bar
families; count and relation scenes are explicitly nonapplicable.

### Phase E: Qwen2.5-VL preflight only

First record the pin/configuration/local-hardware audit. This command does not
download a model, acknowledge a GPU run, or invoke training:

```bash
PYTHONPATH=src .venv/bin/python scripts/preflight_vlm.py \
  --sft-config configs/vlm/qwen25vl3b_sft.yaml \
  --rl-config configs/vlm/qwen25vl3b_joint_grpo.yaml \
  --output artifacts/logs/private_vlm_evidence/vlm_preflight_hardened_gate.json
```

The current private preflight metadata records `training_invoked: false`,
`large_gpu_started: false`, and six blockers. It is not an accepted evidence
artifact: no hash is registered or publishable, and it does not promote any
Phase-E row.

The preflight writer is no-clobber: use a new versioned filename for each
rerun. Its local environment and hardware inventory belongs in the ignored
private-evidence subtree and must not be committed or published without a
separate privacy review. The historical `artifacts/manifests/vlm_preflight.json`
predates the hardened-container, SBOM, and vulnerability-policy blockers and is
superseded; its hash is not current preflight evidence.

The two training-named scripts are also metadata-only safety gates. They do not
download a model or start SFT/RL. A config-only invocation is intentionally
rejected because the two reviewed artifact path/hash pairs and a private output
destination are mandatory; use the complete templates below only after those
artifacts exist.

A reviewed operator may ask the RL CLI to emit a private plan only after GPU
cost review and two separately reviewed evidence artifacts exist. The Phase-D
audit must be automatically clean and ready, with a human reviewer identifier, at
least 200 unique reviewed sample IDs, and a binding to the manifest's canonical
self-hash. The `stage: joint_outcome_rl` schema-v2 execution audit must bind the
dataset raw-manifest/content hashes and a `local_files_only` model-snapshot
manifest to an actual SFT
checkpoint, target-container smoke, model-measured parser rate, isolated
state-injection adapter, H1 gate, and exact 16-key veRL audit. The CLI re-hashes
the artifacts, checkpoint, adapter, snapshot, raw/canonical dataset JSONL, the
exact 1,820-image tree, and all 73 contact sheets both when loading evidence and
again for every plan construction. The data-tree replay is confined to its
bound safe root and rejects symlinks, missing or extra files, and byte tampering.
It limits the planning-time GPU set to the intersection of the target-container
smoke UUIDs and local `nvidia-smi` output, and accepts no operator-supplied
device or API-audit flag.
Because this boundary emits a non-executable plan, binding the eventual runtime
executor to those UUIDs remains an explicit pending gate:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_vlm_rl.py \
  --config configs/vlm/qwen25vl3b_joint_grpo.yaml \
  --acknowledge-large-gpu-run \
  --phase-d-audit artifacts/metrics/cva_v2_audit.json \
  --phase-d-audit-sha256 "REPLACE_WITH_64_HEX_PHASE_D_SHA256" \
  --execution-audit artifacts/logs/private_vlm_evidence/vlm_execution_audit.json \
  --execution-audit-sha256 "REPLACE_WITH_64_HEX_EXECUTION_AUDIT_SHA256" \
  --output-config artifacts/logs/private_vlm_plans/qwen25vl3b_joint_grpo.plan.yaml
```

Replace both `REPLACE_WITH_...` tokens with the reviewed 64-character
lowercase hashes of the exact files named beside them; placeholders are
intentionally rejected.

The emitted file remains a private `execution_status: not_started`,
`execution_permitted: false` plan; the command neither invokes training nor
turns that plan into experimental evidence. Snapshot hashing supplies local
self-consistency only, not authentication against a trusted upstream digest
allowlist. The example cannot succeed with the repository's current artifacts:
human Phase-D sign-off, the schema-v2 execution audit, the isolated worker and
reviewed adapter hash, target-GPU smoke, model-measured parser rate, local model
snapshot, and their cross-bindings are all still pending. Even if those inputs
later pass, the current plan records
`previous_phase_a_c_artifacts_verified: false` and external authorization as
not granted, so it cannot authorize execution.

The SFT CLI consumes the same four artifact path/hash flags, but its execution
audit must instead declare `stage: structured_sft`. At that pre-training stage,
the schema forbids invented SFT-checkpoint, parser, state-injection, H1, and
veRL-API evidence. Its veRL configuration remains explicitly deferred and its
private plan is likewise non-executable:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_vlm_sft.py \
  --config configs/vlm/qwen25vl3b_sft.yaml \
  --acknowledge-large-gpu-run \
  --phase-d-audit artifacts/metrics/cva_v2_audit.json \
  --phase-d-audit-sha256 "REPLACE_WITH_64_HEX_PHASE_D_SHA256" \
  --execution-audit artifacts/logs/private_vlm_evidence/vlm_sft_execution_audit.json \
  --execution-audit-sha256 "REPLACE_WITH_64_HEX_EXECUTION_AUDIT_SHA256" \
  --output-config artifacts/logs/private_vlm_plans/qwen25vl3b_sft.plan.yaml
```

The same exact-file hash replacement rule applies to this SFT template.

Plan destinations must be new `.yaml`/`.yml` files under the ignored
`artifacts/logs/private_vlm_plans/` subtree or a system temporary directory;
the CLI rejects other repository destinations, clobbers, symlinks, and
overlapping targets. A plan can contain
local absolute audit/snapshot paths and must not be committed or published.
The execution audit likewise belongs under the ignored
`artifacts/logs/private_vlm_evidence/` subtree: it contains local paths and GPU
UUIDs and must not be committed or published. If public attestation is needed,
derive a separately reviewed hash-only redacted record; never edit or replace
the private evidence bytes whose hash gates the plan.

The fixed provenance in code and YAML is:

- Qwen model revision
  [`66285546d2b821cf421d4f5eb2576359d3770cd3`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3);
- veRL revision
  [`7aed6b230776f963fa09509c10d9c3a767d1102c`](https://github.com/verl-project/verl/commit/7aed6b230776f963fa09509c10d9c3a767d1102c)
  (`0.8.0.dev` at that revision);
- vLLM release `0.20.2`;
- Transformers release `5.3.0` and PyTorch `2.11.0`, matching the pinned veRL
  revision's official stable-vLLM Dockerfile.

The audit sources are the official [veRL v0.8.0 release](https://github.com/verl-project/verl/releases/tag/v0.8.0),
[Qwen2.5-VL GRPO example at the pinned revision](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh),
and [stable-vLLM Dockerfile at that revision](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/docker/Dockerfile.stable.vllm).

The pinned upstream revision has a packaging inconsistency: its
[`setup.py`](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/setup.py)
declares the stale extra `vllm>=0.8.5,<=0.12.0`, while its official
`docker/Dockerfile.stable.vllm` pins vLLM `0.20.2`, PyTorch `2.11.0`, and
Transformers `5.3.0`. Therefore, do **not** run `pip install -e .[vllm]` for the
GPU environment. The pinned Dockerfile is the audited upstream dependency
reference; first verify its bytes against SHA-256
`be8bd117fc415690c2d433e2e3c8832e6a96dd6de4e799be6a4be05c9eb4f300`,
then install the pinned veRL checkout with `pip install --no-deps -e .` only
inside the hardened descendant described below. No training may start before
the target-hardware preflight passes.

The repository records that Dockerfile digest as an expected string but does
not vendor the Dockerfile bytes. Before building, retrieve the file from the
exact veRL commit above and verify its raw bytes against the recorded SHA-256.
That upstream file must not be treated as a reproducible or authenticated build
recipe: it contains a base-image tag without a digest, unhashed downloads,
unversioned apt/pip inputs, Git dependencies without immutable commits, and
dynamic CUDA/cuDNN components. Before GPU use, derive and vendor a hardened
Dockerfile that pins the base digest, download/wheel hashes, exact Git commits,
and CUDA/cuDNN artifacts and installs from an offline verified wheelhouse.

The current schema-v2 execution audit validates only the upstream-reference
container smoke, frozen runtime versions, machine GPU UUIDs, and the stage's
dataset/model/checkpoint bindings. It has no completion fields for the hardened
descendant, offline wheelhouse, SBOM, or vulnerability policy. Every emitted
plan therefore hard-codes those four requirements and runtime-executor GPU UUID
binding as `pending`, and remains `execution_permitted: false`. Before actual
GPU execution, implement and review
a new authenticated gate that binds the hardened Dockerfile hash, final image
digest, offline inputs, SBOM, vulnerability-policy result, and target-hardware
smoke. None of that supply-chain evidence or the required gate extension exists
here; a current schema-v2 audit cannot waive it.

Within the generated plan, the nested `verl` mapping contains only keys emitted
through the sole formal entry point
`compbias.rl.verl_entrypoints.build_grpo_execution_plan`; its configuration
surface was checked against the pinned official veRL revision and official GRPO
documentation. The low-level builder is private. The plan's surrounding
`stage`, status, and preflight metadata are local safety/provenance fields, not
veRL configuration keys.

The implemented CPU state-injection assertion, even when it passes, establishes
only that the reasoner call does not receive an explicit image argument. It does
not rule out image access
through a closure, object member, cache, or other shared process state. A
separately isolated text-only worker and a recorded adapter hash have not yet
been implemented or verified; both are mandatory Phase-E gates before any
causal image-hidden claim or GPU execution.

Stop the large-model stage immediately if any of the following is true:

- an earlier CPU/data gate is incomplete;
- acknowledgement, fixed revisions, or target-GPU evidence is absent;
- for joint RL, the exact veRL-key audit is absent or the post-SFT structured
  parse rate is below 98%;
- the isolated text-only reasoner worker or its reviewed adapter hash is absent;
- fixed-reasoner H1 sign prediction is not above chance on preregistered task
  families;
- no task shows a measurable coupling contribution;
- seed, config snapshot, dataset hash, checkpoint hash, or exact command is
  missing from the run record.

The 7B model is out of scope until every 3B gate passes. A generated preflight
YAML, an allocated GPU, or a successful import is never evidence that training
occurred.

### Phase F: external validation

`external/measurebench/pin_manifest.json` freezes MeasureBench at commit
`d5bb8652dbde6b1b5507f89d37f73993af28b830`.
`external/measurebench/preregistration.md` fixes the intended synthetic-only
subset and error-mechanism shifts. Clone, license review, generator adaptation,
and evaluation are all still **not run**.

External validation may be reported only after the checked-out commit, source
license, generated-data manifest, unchanged official evaluator, paired shift,
and result hashes are recorded. A null external result must reduce the paper's
scope rather than be omitted.

## Evaluation interfaces

Compensability estimation consumes only the paths, hashes, seeds, and statistical
contract frozen in its YAML. When its currently missing GPU artifacts are later
supplied, it requires the exact 430-sample `calibration` split and derives each
sample's exhaustive error catalog from the bound cva_v2 dataset rather than
trusting a rollout-provided catalog. The frozen dataset expands those 430
samples to 950 registered `(sample,error)` entries; across the 32 preregistered
rollout seeds, a complete interventional input therefore contains 30,400 rows.
These are required future input counts, not observed rollouts or estimates:

```bash
PYTHONPATH=src .venv/bin/python scripts/estimate_compensability.py \
  --config configs/eval/compensability_qwen3b.yaml
```

That YAML and CLI cover only the interventional view. The matched natural-view
schema and GPU rollout producer are not implemented or recorded, so no paired
natural-versus-interventional estimate exists and neither view may be inferred
from the other's placeholder configuration.

Paired IID/OOD evaluation likewise consumes the frozen YAML and a separately
named recorded checkpoint. Its frozen scope is the 100 source-linked
`iid_test`/`ood_test` pairs induced by the 100 OOD records, at seeds 11, 17,
and 23, with 10,000 seeded paired-bootstrap resamples, 95% intervals, raw
p-values, and Holm-adjusted comparisons. The full 430-record `iid_test` split
exists in cva_v2, but the remaining 330 IID records are outside this CLI's
registered paired-OOD scope. A complete future prediction input contains 600
rows: 100 pairs times three seeds times the two IID/OOD partitions. No such
prediction set is currently recorded:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_checkpoint.py \
  --config configs/eval/full.yaml \
  --checkpoint artifacts/checkpoints/qwen25vl3b/joint_rl
```

Both commands currently exit `BLOCKED`: the compensability command reports
that GPU-rollout `provenance.checkpoint_sha256` is not recorded, and the
evaluation command reports the missing checkpoint artifact. No reviewed
post-GPU execution audit, rollout/prediction manifest, or human-ready Phase-D
binding exists, and the evaluation hash placeholders remain null. The CLIs
validate those artifacts and the complete local v2 entity (1,820 JSONL rows,
1,820 PNG files, plus
manifest file/self/content/raw-dataset hashes) before analysis; a YAML file
cannot be mistaken for
completed E-COMP or checkpoint-evaluation evidence.

The schema-v3 post-GPU execution audits named by both evaluation YAML files
belong under the ignored `artifacts/logs/private_vlm_evidence/` subtree. They
can contain machine GPU UUIDs and local evidence paths and must not be committed
or published. Their present schema is an unauthenticated draft, and the CLIs
hard-block completion until a future authenticated gate extension exists; a
locally edited JSON file cannot complete either analysis. A public result needs
a separately reviewed, redacted/hash-only attestation; it must not alter the
private bytes whose hash gates analysis.

## Reporting contract

Every accepted run must preserve its applicable provenance. All stages record
the Git commit and dirty state, exact configuration and command, timestamps,
seeds/device, Python, and the relevant package versions. Data stages add
dataset hashes; GPU stages additionally add CUDA/GPU metadata, model and veRL
revisions, and checkpoint hashes. Every formal Phase A/B run must also emit a
complete local `RunLogger`
bundle under `artifacts/logs`, containing config, environment, metrics,
rollouts, predictions, checkpoints, and report entries; its standalone JSON/CSV
outputs do not replace that bundle. Fields that truly do not apply may be
explicitly null/not-applicable, never silently fabricated. Local files under
`runs/` or `artifacts/` are the evidence of record; a dashboard alone is
insufficient.

Before drafting a claim, update these files:

- `paper/theorem_notes.md`: statement, assumptions, and test scope;
- `paper/experiment_registry.md`: actual run and artifact status;
- `paper/claim_evidence_matrix.md`: permitted claim strength;
- `paper/related_work_audit.md`: verified metadata and originality boundary;
- `paper/cva_world_dataset_card.md`: generated-data contract and review status.

Then derive the paper-facing table from the evidence registry into a new,
non-overwriting artifact:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_paper_tables.py \
  --registry paper/experiment_registry.md
```

The default destination is `artifacts/reports/paper_tables.md`; use `--output`
only for another new file. This builder formats registry evidence and does not
promote a blocked, partial, or not-run row.

Perception/process rewards are controls only. The project must not claim a new
reward, token-level advantage method, router, auxiliary head, or architecture as
its contribution.
