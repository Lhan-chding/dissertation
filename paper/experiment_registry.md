# Experiment registry

Last updated: 2026-08-14 (Asia/Singapore, UTC+08).

This is the evidence index of record. Rows distinguish executable code,
completed CPU evidence, planned GPU work, and external validation. A config,
test definition, preflight report, or empty artifact directory is never promoted
to a completed experiment.

## Status values

- `VERIFIED_CPU`: the canonical command and every required provenance artifact
  were accepted for the current code/configuration; a focused test alone is
  insufficient.
- `IMPLEMENTED_NOT_RECORDED`: implementation/tests exist but no registry artifact
  has yet been accepted for this row.
- `PARTIAL_GATE`: one or more recorded checks passed, but the full acceptance
  gate remains incomplete.
- `PREREGISTERED_NOT_RUN`: protocol is frozen; no result exists.
- `BLOCKED_BY_GATE`: execution is forbidden until named prerequisites pass.
- `SUPERSEDED`: historical evidence retained only to explain provenance; it is
  not accepted for the current protocol.
- `NOT_APPLICABLE`: intentionally excluded.

## Registered experiments

| ID | Phase | Question | Primary protocol | Current status | Evidence or blocker |
|---|---|---|---|---|---|
| SYS-CPU | A--D | Does the complete local implementation pass its test and coverage contracts? | full `pytest`; branch measurement over `compbias` and `scripts`; coverage.py combined total >=80% | VERIFIED_CPU | Clean HEAD `84eda3bbaeaf4822f9cb238f3080ead3cd6c9c0b`: 1,024 passed with no failures/errors; 10,696/12,340 statements (86.6774716%), 3,379/4,708 branches (71.7714528%), combined 82.5610042% (83% display), so the 80% gate passed. The summary binds raw coverage SHA-256 `0cbda5472847078ccb44e625eed1b25a49982fcc218fc0f141d5554b78b3a9c0`. `artifacts/metrics/coverage_summary.json` SHA-256 `815a588faba3392a96a5e15fe43674984af7957bca1ac4d6c0b44ca2355962cb`. |
| A-SEL | A | Do selection, covariance, and pairwise-odds identities match finite computation? | Fixed fixture plus 1,000 seeded distributions | VERIFIED_CPU | Clean run `run-ce509068302076dc` passed the registered Phase-A bundle: seven identities, 1,000 cases each, 7,000 checks total. `artifacts/theory_verification/random_property_tests.json` SHA-256 `0977e4c93d642e286f12a47148c071956dc3a922b4cb257500df277293e6b653`. |
| A-SCALE | A | Does the reasoner-scaling derivative equal the weighted covariance? | Autograd, finite difference, 1,000 seeded landscapes | VERIFIED_CPU | The accepted clean Phase-A run passed the registered scaling checks. `artifacts/theory_verification/random_property_tests.json` SHA-256 `0977e4c93d642e286f12a47148c071956dc3a922b4cb257500df277293e6b653`. |
| A-LOCK | A | Does repeated exact selection match its closed form and ratio rate? | Fixed/tied/unique maxima plus randomized cases | VERIFIED_CPU | The accepted clean Phase-A run passed the registered lock-in checks. `artifacts/theory_verification/random_property_tests.json` SHA-256 `0977e4c93d642e286f12a47148c071956dc3a922b4cb257500df277293e6b653`. |
| A-COORD | A | Are the coordination fixed points, basin boundary, and KL bifurcation reproduced? | Analytic Jacobian, ODE fixtures, beta/a grid | VERIFIED_CPU | The accepted clean Phase-A run passed the registered coordination and bifurcation checks. `artifacts/theory_verification/random_property_tests.json` SHA-256 `0977e4c93d642e286f12a47148c071956dc3a922b4cb257500df277293e6b653`. |
| A-DECOMP | A | Does outcome loss equal local losses plus coupling? | Per-sample, aggregate, Bregman, randomized quadratics | VERIFIED_CPU | The accepted clean Phase-A run passed the registered decomposition checks. `artifacts/theory_verification/random_property_tests.json` SHA-256 `0977e4c93d642e286f12a47148c071956dc3a922b4cb257500df277293e6b653`. |
| B-EXACT | B | Does the exact optimizer match the selection-law target across three profiles? | truth-aligned / flat / spurious, beta grid | VERIFIED_CPU | Clean run `run-5f8370785b97da1d` passed the exact-selection gates. `artifacts/metrics/tabular_selection.json` SHA-256 `625d5b7894063d5e843f16454acf023116bda35dfdac9e1f125de96b68e40192`. |
| B-APPROX | B | Do finite-policy approximate updates approach the marginal selection target under fixed-reasoner binary outcomes, and what changes under less constrained diagnostics? | formal `raw_fixed_reasoner_outcome` error policy with independent `r~Bernoulli(c[e])` draws and marginal KL for REINFORCE/PPO-like/GRPO-like; separate unconstrained-joint and collapsed diagnostics; mirror oracle | VERIFIED_CPU | The clean run passed the formal 20-seed-per-algorithm fixed-reasoner gates, including full error-action coverage and aggregate conditional-rate error <=0.03. `unconstrained_joint_trajectory_diagnostic` and `collapsed_effective_reward_diagnostic` remain diagnostic-only and cannot establish neural PPO/GRPO fidelity or real-VLM behavior. `artifacts/metrics/tabular_selection.json` SHA-256 `625d5b7894063d5e843f16454acf023116bda35dfdac9e1f125de96b68e40192`. |
| B-SCALE | B / plan Exp-C | Can equal-average-gain reasoner paths induce opposite perception-severity directions according to `Cov(ell,G)`? | `truth_gain`, `uniform_gain`, `error_gain`; covariance versus central finite difference | VERIFIED_CPU | The clean finite tabular run passed all registered covariance/difference gates. This is not the pending matched real-checkpoint H3 experiment. `artifacts/metrics/tabular_scaling.json` SHA-256 `4ed720e1c7bde52e33d4625bec5da839397e1a69a252383b87c5234236a03548`. |
| B-JOINT | B | Are truthful and compensatory tabular basins reproducible? | `p0,q0,beta,delta,epsilon` grids; >=20 seeds | VERIFIED_CPU | The clean run passed the registered coordination and bifurcation gates. `artifacts/metrics/tabular_coordination.json` SHA-256 `4ea154f66656d66599b44de877a79962353aec52035f72c112236ecf5c6683d0`; `artifacts/metrics/tabular_bifurcation.json` SHA-256 `ade5660fd547a4c5b5e80f64c5644efbbb04042996a410434a82c1133ea6709b`. |
| C-FIXED | C | Are at least two fixed-reasoner perception directions predicted correctly? | `configs/neural/all.yaml`; 10 seeds | VERIFIED_CPU | The accepted clean sweep passed all registered scalar gates and produced 50 complete RunLogger bundles for 50 unique condition--seed combinations and 2,450 checkpoints. This is a two-sigmoid diagnostic only. `artifacts/metrics/neural_summary.json` SHA-256 `534e4de5a3c8bab2076a517c8ce6d1f55163d1564fa9753abb2b4d35be7339e0`. |
| C-JOINT | C | Do registered seeds reach different differentiable coordination endpoints? | 10-seed joint sweep | VERIFIED_CPU | The accepted clean scalar sweep passed the registered endpoint gate; it is evidence only for the two-sigmoid reduction. `artifacts/metrics/neural_summary.json` SHA-256 `534e4de5a3c8bab2076a517c8ce6d1f55163d1564fa9753abb2b4d35be7339e0`. |
| C-OOD | C | Is the compensatory endpoint more fragile to the registered error permutation? | paired IID/OOD metrics from the same runs | VERIFIED_CPU | The accepted clean scalar sweep passed the registered synthetic-permutation OOD gate; it is not a natural-image or VLM result. `artifacts/metrics/neural_summary.json` SHA-256 `534e4de5a3c8bab2076a517c8ce6d1f55163d1564fa9753abb2b4d35be7339e0`. |
| C-VIS | C | Does the convolutional perceiver participate in the controlled mechanism on rendered input? | `configs/neural/visual_modular.yaml`; one fixed CVA/PIL 16x16 RGB scene; strict registered mode sweep | VERIFIED_CPU | Clean run `sweep-20260814T010824107927Z-dacc8f8dec6c` passed the 10-seed controlled PIL/CNN/MLP mechanism gate. Scope remains one fixed 16x16 scene with a byte-identical paired image, not multi-image generalization or VLM evidence. `artifacts/metrics/visual_neural_summary.json` SHA-256 `89a862da7e50215d62a43360235d41b4acf6b4d2db971242e40e06cf5d019419`. |
| D-CVA | D | Does CVA-World satisfy solver, round-trip, split, hash, and visual-review gates? | `configs/data/cva_v2.yaml` | PARTIAL_GATE | Clean local generation/audit recorded 1,820 samples/images, 1,820 solver checks, 4,020 error-solver/round-trip checks, 73 sheets, and 100 OOD pairs with `automatic_audit_clean: true`. `artifacts/metrics/cva_v2_audit.json` local file SHA-256 `f6781e82052463f0e749864ced41b559a680604872f0de72b39fa2955dc8142f`; manifest file SHA-256 `6bcc090c91cc7f47fa03309a2fb56bbd7e2b0e8a0b8bc8ca21b98ee7399c340a`; manifest self SHA-256 `3f9efc0d0ec76284d7d5dc1a36d0c1b57ec736aa8c4c1b968447e022fca0c108`; content SHA-256 `5010137acc0cfc06238efbf23183aa865080d4882a9abcadfe6d8b365092ad36`; raw dataset SHA-256 `d439f621c395a21cb5386f1d769b3fd53cf6195c7b2389a6df7e3fcaa43eaddc`; image-set SHA-256 `6a1171e8716cd93c644bf771dcd3a6f79eceb9c6cca5120f5812895e57999e4e`. Human review/sign-off is absent, so `human_reviewer_signoff: false` and `phase_d_ready: false`; these ignored/local D artifacts are not a `VERIFIED_CPU` JSON bundle or approved release. |
| E-SFT | E | Can Qwen2.5-VL produce >=98% valid structured outputs? | pinned 3B SFT | BLOCKED_BY_GATE | No model download or GPU run is recorded. A private, unaccepted preflight metadata report records `training_invoked: false`, `large_gpu_started: false`, and six blockers; its hash is neither registered nor publishable. Artifact-backed CLI requires a ready human-bound Phase-D audit plus a `stage: structured_sft` schema-v2 audit with machine-matched upstream-reference container smoke, exact v2 data-tree bindings, and self-consistent `local_files_only` model-snapshot evidence. This pre-training schema forbids checkpoint/parser/state/H1/veRL-audit claims; SFT veRL keys remain deferred and Phase-A--C linkage remains unverified. Hardened-container/offline-wheelhouse/SBOM/vulnerability evidence and runtime-executor GPU UUID binding remain pending, so every private plan stays `not_started` and `execution_permitted: false`. |
| E-COMP | E | Does pre-RL interventional compensability predict the fixed-reasoner post-RL shift? | matched natural and exact 430-sample calibration interventional views through a separately isolated text-only worker | BLOCKED_BY_GATE | `configs/eval/compensability_qwen3b.yaml` and its CLI specify only the interventional analysis consumer: frozen scope is 950 dataset-derived `(sample,error)` entries times 32 seeds = 30,400 required rows. The natural-view schema/producer and both GPU rollout sets are not implemented or recorded. The CPU interface assertion, even if it passes, shows only that the current call omits an explicit image argument; an isolated worker and reviewed adapter hash are also pending. The private schema-v3 audit is an unauthenticated draft and cannot complete the gate until an authenticated extension is implemented. |
| E-EVAL | E | Does a recorded checkpoint retain its IID behavior under the paired error-mechanism shift? | exact 100 source-linked IID/OOD pairs; seeds 11/17/23; paired bootstrap and Holm correction | BLOCKED_BY_GATE | Frozen future input scope is 100 pairs times three seeds times two partitions = 600 prediction rows. No checkpoint or prediction set is recorded, a human-ready/authenticated Phase-D binding is absent, and the private schema-v3 execution-audit format remains an unauthenticated draft pending an authenticated gate extension. The other 330 cva_v2 IID-test rows are outside this paired-OOD protocol. |
| E-GRPO | E | Does joint outcome-only VLM RL exhibit coupling or multiple endpoints? | pinned veRL/GRPO, >=3 seeds | BLOCKED_BY_GATE | Config is metadata-only; no large-GPU job start is recorded. The private preflight metadata is unaccepted and records six blockers, `training_invoked: false`, and `large_gpu_started: false`. Its `stage: joint_outcome_rl` schema-v2 audit additionally requires a hashed SFT checkpoint, model-measured parser rate, separate text-only adapter file/reviewed hash, H1 gate, and exact 16-leaf veRL audit, all cross-bound to the snapshot, exact v2 data tree, and post-SFT checkpoint as applicable. The CLI probes GPU UUIDs itself, limits planning-time selection to the target-smoke/local intersection, and may emit only a private non-executable plan; runtime-executor UUID binding remains pending. Snapshot hashing establishes integrity/self-consistency, not trusted-upstream authenticity; hardened-container/offline-wheelhouse/SBOM/vulnerability evidence remains outside the current schema and hard-coded pending in every plan. |
| E-CONTROL | E | Does oracle perception/process supervision reduce compensatory occupancy? | matched control only | BLOCKED_BY_GATE | No VLM baseline, so control is not runnable. |
| F-MB | F | Does a core phenomenon replicate on MeasureBench Synthetic? | pinned commit and preregistered numeric subsets/shifts | PREREGISTERED_NOT_RUN | Repository not retrieved or executed; see `external/measurebench/`. |
| F-VFLIP | F | Does coupling predict failure to update under task-critical visual changes? | paired counterfactual evaluation | PREREGISTERED_NOT_RUN | Dataset/code availability and license not audited. |

## Acceptance gates

### Phase A

- identity error below `1e-8`;
- at least 1,000 registered randomized cases for the core identities;
- ODE behavior consistent with the analytical basin result;
- critical beta error below `1e-3`.
- the formal run has a complete local `RunLogger` bundle plus standalone
  artifacts, with exact config/command/environment/Git provenance.

### Phase B

- exact target matches within the registered numerical tolerance;
- mirror-descent error decreases with step size;
- formal `raw_fixed_reasoner_outcome` updates optimize only the error policy,
  sample binary outcomes independently from the registered frozen `c[e]`, apply
  KL on the error marginal, audit that every error was sampled and aggregate
  empirical conditional rates are within `0.03` of `c[e]`, and reproduce the
  registered direction under their finite-policy gates;
- the unconstrained joint-trajectory and collapsed effective-reward results are
  labeled diagnostic-only and reported separately from formal fixed-reasoner
  evidence;
- equal-average-gain `truth_gain`, `uniform_gain`, and `error_gain` paths have
  negative, near-zero, and positive severity directions matching the covariance
  identity and finite differences;
- both stable coordination endpoints repeat across the preregistered grid.
- the formal run has a complete local `RunLogger` bundle plus standalone
  metrics/predictions, with exact config/command/environment/Git provenance.

### Phase C

- two or more fixed-reasoner profiles have the predicted direction;
- distinct seeds yield truthful and compensatory endpoints;
- registered OOD error permutation hurts the compensatory endpoint more.

### Phase D

- all canonical answers self-check;
- all registered error transforms round-trip;
- no forbidden split overlap;
- manifest/config/content hashes validate;
- visual-factor realizations, answer balance, OOD image shift,
  style--semantic joint independence, and deterministic replay (including
  contact sheets) pass the schema-v2 automatic audit;
- at least 200 unique rendered sample IDs receive documented human review, and
  the same closed record covers all 73 manifest contact sheets;
- the public reviewer field is a bounded `reviewer-*` pseudonym; real identity
  and consent evidence remain outside generated artifacts;
- every non-OOD semantic state is crossed with every family-applicable IID style,
  and this is audited independently of semantic/answer splits;
- bar semantics use only the registered first-two-bar `sum`, `difference`, and
  exact-ratio questions in the frozen 4/3/3 allocation per split;

### Phase E

- every earlier gate passes;
- the SFT plan audit contains no fabricated post-SFT evidence; before RL, an
  actual hashed SFT checkpoint exists and its model-measured structured parse
  rate is >=98%;
- reviewed Phase-D and schema-v2 execution-audit bytes pass their supplied
  SHA-256 values, bind the same dataset/model snapshot/content/checkpoint as
  applicable, and record the frozen upstream-reference runtime/container smoke
  whose GPU UUID matches machine `nvidia-smi` output;
- a future authenticated gate binds the hardened descendant, offline
  wheelhouse, SBOM, and vulnerability-policy evidence; the current schema-v2
  audit cannot complete or waive these plan-level pending requirements;
- the eventual runtime executor is bound to reviewed GPU UUIDs; planning-time
  intersection with target-smoke and local `nvidia-smi` UUIDs does not complete
  that execution-time gate;
- a separately isolated text-only reasoner worker and reviewed adapter hash are
  implemented and verified; the existing no-explicit-image-argument interface
  test is insufficient to establish this gate;
- H1 sign prediction is above chance on preregistered families;
- at least one task has measurable coupling contribution;
- all checkpoint, seed, configuration, hardware, and command provenance exists.

### Phase F

- pinned source and license are recorded;
- the official evaluator is unchanged;
- synthetic generation and the paired shift are hashed;
- pre-RL covariance predicts a post-RL perception shift externally;
- a null external result is reported and reduces claim scope.

## Execution log

Append one row per accepted command. Never erase an older row; if later code or
protocol changes invalidate it, retain its command/hash and mark it
`SUPERSEDED` as done below.

| Date (UTC) | Registry ID | Command / run ID | Result | Artifact hash or test summary | Operator note |
|---|---|---|---|---|---|
| 2026-08-13 | SYS-CPU | `PYTHONPATH=src MPLCONFIGDIR=/tmp/compbias-mpl-coverage .venv/bin/python -m pytest -q --cov=compbias --cov-branch --cov-report=term-missing --cov-report=json:artifacts/metrics/coverage.json --cov-fail-under=80` | SUPERSEDED | stale record: 542 passed in 22.63 s; coverage JSON SHA-256 `714ae78d7b7f39288e6790aa69848d2ff4180e78516e2db21c75f587958a2e63` | Historical code snapshot only; superseded by the accepted 2026-08-14 whole-project coverage row below. |
| 2026-08-13 | A-SEL/A-SCALE/A-LOCK/A-COORD/A-DECOMP | `PYTHONPATH=src .venv/bin/python scripts/verify_theory.py --config configs/theory/all.yaml --overwrite` | SUPERSEDED | stale property artifact SHA-256 `8870705fdef1720166f79e4139831d27fe5ad2084c5f2f1691c77756489d2f17` | Historical code snapshot only; superseded by the accepted 2026-08-14 Phase-A row below. |
| 2026-08-13 | A-DECOMP and focused Phase-A suite | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_selection_identity.py tests/test_scaling_identity.py tests/test_lockin_closed_form.py tests/test_coordination_fixed_points.py tests/test_bifurcation.py tests/test_bregman_identity.py tests/test_fisher_projection.py tests/test_decomposition.py` | SUPERSEDED | stale record: 115 passed in 1.52 s | Historical code snapshot only; it does not establish the final repository state or replace proofs. |
| 2026-08-13 | B-EXACT/B-APPROX/B-SCALE/B-JOINT | `PYTHONPATH=src .venv/bin/python scripts/train_tabular.py --config configs/tabular/all.yaml --overwrite` | SUPERSEDED | stale selection SHA-256 `e104dfbfc3654b2c250b5323b3debb8111b43124d1fd250b4cf94093c4f1c1fa`; coordination `dd1613ca7a9fc8b2814d71923ad2f9de567007c60ff45ca39c67cecbd9df7dfc`; bifurcation `e82cbb12e6d4fda1a6967963fcb2ad4bf39d36d20b3d15af0036dc1831f8cebd` | These bytes predate the formal `raw_fixed_reasoner_outcome` path, its explicit diagnostic split, and the registered scaling output. They are superseded by the accepted 2026-08-14 Phase-B row below. |
| 2026-08-13 | C-FIXED/C-JOINT/C-OOD | `PYTHONPATH=src .venv/bin/python scripts/train_neural.py --config configs/neural/all.yaml --overwrite` | SUPERSEDED | stale 30-run summary SHA-256 `ee1aa34524ccdd41337b3740803a32034dfcdc64a50bf942743aa659f41f87c2` | These bytes exercised only `perception_only` and `joint`; the accepted 2026-08-14 row below replaces them for the current config. |
| 2026-08-13 | C-VIS | `PYTHONPATH=src MPLCONFIGDIR=/tmp/compbias-mpl .venv/bin/python scripts/train_visual_neural.py --config configs/neural/visual_modular.yaml --overwrite` | SUPERSEDED | stale summary SHA-256 `2cb3fa3e03711fc97b7543de6cd3f07328d8ec06e3acfe3e2dfada5d8e7096b4`; recorded config SHA-256 `48004cb9e2aa0ad2014517ea788b16339ffe93f21746304bd03fbc7f69b5997d` | Old metrics remain historical and cannot be attributed to the current config; the accepted 2026-08-14 C-VIS row below replaces them. |
| 2026-08-13 | D-CVA-v1 | Historical v1 generation/audit | SUPERSEDED | 250-row artifact; hashes intentionally omitted from the current evidence surface | The v1 identity/manifest contract is incompatible with frozen v2 and must not support a current claim. |
| 2026-08-13 | E-SFT/E-GRPO preflight | `PYTHONPATH=src .venv/bin/python scripts/preflight_vlm.py --sft-config configs/vlm/qwen25vl3b_sft.yaml --rl-config configs/vlm/qwen25vl3b_joint_grpo.yaml --output artifacts/manifests/vlm_preflight.json` | SUPERSEDED | stale metadata/local-hardware report SHA-256 `8d22e5dc0466f3be9ab20ab6959f1a19d65a6d0f1e630a8326884614f60e1103` | That report recorded `model_download_attempted: false` and `training_invoked: false`, but its bytes predate the hardened-descendant, SBOM, and vulnerability-policy blockers now enforced by the preflight code. Do not use this hash as current preflight evidence. A replacement has not been accepted; future reports must use a new no-clobber path under ignored private evidence and remain distinct from schema-v2 execution audits. |
| 2026-08-13 | DOC-CONTRACT | Documentation/configuration contract | PASS | strict unique-key parse of repository contract YAML and `CITATION.cff`, strict parse of registered JSON, fixed-stack equality, 16-key GRPO surface, commit/hash shape, URL syntax, privacy scan, and `git diff --check` passed | Structural/static validation only; ignored run-log counts are deliberately excluded because reruns add immutable bundles. This does not count as experimental or external evidence. |
| 2026-08-14 | SYS-CPU | `COVERAGE_FILE=/private/tmp/.coverage-compbias .venv/bin/coverage erase && PYTHONPATH=.:src COVERAGE_FILE=/private/tmp/.coverage-compbias .venv/bin/coverage run -m pytest -q && COVERAGE_FILE=/private/tmp/.coverage-compbias .venv/bin/coverage combine && COVERAGE_FILE=/private/tmp/.coverage-compbias .venv/bin/coverage json -o artifacts/metrics/coverage.json && COVERAGE_FILE=/private/tmp/.coverage-compbias .venv/bin/coverage report` | VERIFIED_CPU | clean HEAD `84eda3bbaeaf4822f9cb238f3080ead3cd6c9c0b`; 1,024 passed, no failures/errors, 887.14 s; summary SHA-256 `815a588faba3392a96a5e15fe43674984af7957bca1ac4d6c0b44ca2355962cb`; raw coverage SHA-256 `0cbda5472847078ccb44e625eed1b25a49982fcc218fc0f141d5554b78b3a9c0` | Statements 86.6774716%, branches 71.7714528%, combined 82.5610042% (83% display); `fail_under=80` passed. |
| 2026-08-14 | A-SEL/A-SCALE/A-LOCK/A-COORD/A-DECOMP | `PYTHONPATH=src .venv/bin/python scripts/verify_theory.py --config configs/theory/all.yaml --overwrite` / `run-ce509068302076dc` | VERIFIED_CPU | config file SHA-256 `c24587a8a4191815553b2a26358a7cdff09d9bbcd2bfeccc30661cfd2692178b`; property artifact SHA-256 `0977e4c93d642e286f12a47148c071956dc3a922b4cb257500df277293e6b653` | Seven registered identities, 1,000 cases each and 7,000 checks total, passed with clean Git provenance at the named HEAD. |
| 2026-08-14 | B-EXACT/B-APPROX/B-SCALE/B-JOINT | `PYTHONPATH=src .venv/bin/python scripts/train_tabular.py --config configs/tabular/all.yaml --overwrite` / `run-5f8370785b97da1d` | VERIFIED_CPU | config file SHA-256 `bedfe03c315586c8502f5089ff8fd5e2c33a0c7937aa55422cbaa5045f23e145`; selection `625d5b7894063d5e843f16454acf023116bda35dfdac9e1f125de96b68e40192`; scaling `4ed720e1c7bde52e33d4625bec5da839397e1a69a252383b87c5234236a03548`; coordination `4ea154f66656d66599b44de877a79962353aec52035f72c112236ecf5c6683d0`; bifurcation `ade5660fd547a4c5b5e80f64c5644efbbb04042996a410434a82c1133ea6709b` | All gates passed with clean Git provenance. Formal fixed-reasoner results use 20 seeds per algorithm; unconstrained-joint and collapsed paths remain diagnostic-only. |
| 2026-08-14 | C-FIXED/C-JOINT/C-OOD | `PYTHONPATH=src .venv/bin/python scripts/train_neural.py --config configs/neural/all.yaml --overwrite` | VERIFIED_CPU | config file SHA-256 `c82da0555b2a4ef91aed9bc42a92b9a27b354eb25d39c1b4163d1ca3cc4f9569`; summary SHA-256 `534e4de5a3c8bab2076a517c8ce6d1f55163d1564fa9753abb2b4d35be7339e0` | Fifty complete clean RunLogger bundles, 50 unique condition--seed combinations, 2,450 checkpoints, all gates passed. Scope is the scalar two-sigmoid diagnostic. |
| 2026-08-14 | C-VIS | `PYTHONPATH=src MPLCONFIGDIR=/tmp/compbias-mpl .venv/bin/python scripts/train_visual_neural.py --config configs/neural/visual_modular.yaml --overwrite` / `sweep-20260814T010824107927Z-dacc8f8dec6c` | VERIFIED_CPU | config file SHA-256 `dacc8f8dec6c64678e8d598d5bf603422c43f965f9f997e64513de1f33a97771`; summary SHA-256 `89a862da7e50215d62a43360235d41b4acf6b4d2db971242e40e06cf5d019419` | Ten-seed controlled 16x16 PIL/CNN/MLP mechanism run passed with clean Git provenance; no broader image or VLM claim. |
| 2026-08-14 | D-CVA generation | `PYTHONPATH=src .venv/bin/python scripts/generate_cva.py --config configs/data/cva_v2.yaml --overwrite` / `generation-20260814T010836108099Z-e0658f1a9850` | PARTIAL_GATE | tracked config file SHA-256 `e0658f1a9850d300d74eeb3e6c445f0ad8dcf702e846b1b1d11fb34e3f6b1071`; manifest file SHA-256 `6bcc090c91cc7f47fa03309a2fb56bbd7e2b0e8a0b8bc8ca21b98ee7399c340a`; raw dataset SHA-256 `d439f621c395a21cb5386f1d769b3fd53cf6195c7b2389a6df7e3fcaa43eaddc` | Clean local generation completed; outputs are ignored/local and are not an approved release or complete Phase-D evidence. |
| 2026-08-14 | D-CVA automatic audit | `PYTHONPATH=src .venv/bin/python scripts/audit_dataset.py --manifest artifacts/manifests/cva_v2.json --output artifacts/metrics/cva_v2_audit.json --report-root artifacts --overwrite` / `audit-20260814T010913290056Z-6bcc090c91cc` | PARTIAL_GATE | audit local file SHA-256 `f6781e82052463f0e749864ced41b559a680604872f0de72b39fa2955dc8142f`; manifest self `3f9efc0d0ec76284d7d5dc1a36d0c1b57ec736aa8c4c1b968447e022fca0c108`; content `5010137acc0cfc06238efbf23183aa865080d4882a9abcadfe6d8b365092ad36`; image set `6a1171e8716cd93c644bf771dcd3a6f79eceb9c6cca5120f5812895e57999e4e` | Automatic audit clean for 1,820 samples/images, 1,820 solver checks, 4,020 error-solver/round-trip checks, 73 sheets, and 100 OOD pairs. Human sign-off absent; `phase_d_ready: false`. |
| 2026-08-14 | E-COMP | `PYTHONPATH=src .venv/bin/python scripts/estimate_compensability.py --config configs/eval/compensability_qwen3b.yaml` | BLOCKED_BY_GATE | exit 2: `BLOCKED: GPU rollout provenance.checkpoint_sha256 is not recorded` | Expected fail-closed result; no estimate or accepted preflight evidence was produced. |
| 2026-08-14 | E-EVAL | `PYTHONPATH=src .venv/bin/python scripts/evaluate_checkpoint.py --config configs/eval/full.yaml --checkpoint artifacts/checkpoints/qwen25vl3b/joint_rl` | BLOCKED_BY_GATE | exit 2: checkpoint artifact missing | Expected fail-closed result; no evaluation or accepted preflight evidence was produced. |
