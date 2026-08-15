# Validated pilot server

> **Current boundary (2026-08-15).** The one permitted v0.3 calibration has
> already completed and failed. The older generation/calibration commands below
> are retained only as historical documentation and must not be run again. Pilot
> A/B are terminated. The only authorized next model execution is the fixed
> Recoverability v1 bridge described at the end of this document.

```text
GPU: NVIDIA GeForce RTX 4090, 47.37 GiB VRAM, compute capability 8.9
CPU: 16 cores
RAM: approximately 94 GiB usable
Python: 3.12.11
PyTorch: 2.8.0+cu128
PyTorch CUDA runtime: 12.8
Driver-advertised CUDA compatibility: 13.2
Data disk: /cloud/cloud-ssd1
Project: /cloud/cloud-ssd1/dissertation
Model: /model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct
```

`nvidia-smi` reports CUDA 13.2 because the driver is forward-compatible. The
PyTorch runtime used by this project is CUDA 12.8. Do not replace the validated
PyTorch build merely to match the `nvidia-smi` banner.

The model is loaded from local files with `local_files_only=True` and
`trust_remote_code=False`. The smoke path never downloads a model.
Preflight, smoke, and the training launcher independently hash the complete
allowlisted model snapshot, including both weight shards. These streaming hash
passes can take longer than inference; they are evidence checks, not a hang.
The smoke must exit zero and report both `smoke_passed: true` and
`answer_correct: true`. A load-only result, a `missing_structured_tags` parse,
an open-schema response, or a wrong known answer is not a pass and must not be
followed by training. The runner permits only two bounded format-repair retries
and records all attempts without deriving an answer from malformed output. Natural-data
collection does not use these retries: it preserves the first response so that
parse failures cannot silently resample the natural error distribution.

## Environment policy

`requirements-gpu.in` is the tested upper-layer snapshot, including nine
security-overlay versions that shadow vulnerable packages inherited from the
server image. `constraints-gpu.txt` protects torch 2.8.0. After preflight and
smoke succeed, export the installed inventory without the editable project or
machine-local direct references:

```bash
python -m pip list --format=freeze --exclude-editable \
  | LC_ALL=C sort -f > requirements-gpu.lock.txt
```

The reviewed server inventory contains 125 unique exact versions and has
SHA-256
`d928379a590e5071d9b5042fe99d480f57ab187f0cb3a74e13af219a6048aeb3`.
It matches every exact version in `requirements-gpu.in`. Because this export
uses `pip list` rather than `pip freeze`, it records resolved versions but not
original installation provenance. Treat it as an interpreter/platform
inventory, not a standalone proof that no package originally came from a
direct URL, a resolver-produced lock, or an artifact-hashed cryptographic
lock.

On 2026-08-14, an isolated `pip-audit==2.10.1` scan of the complete inventory
with `--no-deps --disable-pip` reported no known vulnerabilities for the
auditable packages from the PyPI advisory service, but skipped the
local-version `torch`, `torchaudio`, and `torchvision` builds because the
`+cu128` builds are not published on PyPI. The OSV service additionally
reported advisories against the inherited `torch==2.8.0+cu128` build. That
build is retained only as a time-bounded exception for this single-user
offline pilot: the repository has no `torch.load` path, accepts only the
hashed allowlisted local safetensors snapshot, and enforces
`local_files_only=True` plus `trust_remote_code=False`. Never use this
exception for untrusted `.pt` or `.pth` files, external optimizer/resume
state, a network service, or a multi-tenant notebook. The environment must
not be described as vulnerability-clean; a separate torch 2.10+ CUDA 12.8
validation remains post-pilot work.

The official TRL documentation lists Qwen2.5-VL as a supported GRPO VLM, but
also warns that compatibility is not guaranteed for every model/version. The
local known-answer smoke therefore remains mandatory; the first registered
pilot must be monitored and stopped on any runtime incompatibility.

## Historical first server commands — do not rerun

Run these commands in order. They prepare paths, preserve the validated PyTorch
build, and perform inference-only checks. They do not launch Pilot A or B.

```bash
cd /cloud/cloud-ssd1
git clone https://github.com/Lhan-chding/dissertation.git
cd dissertation

bash scripts/server/setup_paths.sh
bash scripts/server/bootstrap_env.sh
bash scripts/server/run_smoke.sh
```

After the smoke report has been reviewed, generate the deterministic pilot data
and run base calibration:

```bash
source .venv/bin/activate
export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python experiments/gpu_pilot/02_generate_pilot_data.py \
  --paths configs/paths.yaml \
  --config configs/data/cva_chart_pilot_v0_3.yaml

test -f data/generated/cva_chart_pilot_v0_3/manifest.json

python experiments/gpu_pilot/03_base_calibration.py \
  --paths configs/paths.yaml \
  --execute
```

Stop if any calibration gate fails. Pilot A/B require a separate reviewed
decision plus both `--execute` and the explicit acknowledgement environment
variable; that training command is deliberately not part of first setup.
The launcher will still refuse execution unless the Git worktree is clean and
the strict dataset/smoke/calibration replay gate succeeds. Do not edit a report
to force a pass: its source records and referenced bytes are rehashed and
recomputed at launch.

The v0.3 generator writes to a new directory and never replaces the existing
v0.1 or v0.2 evidence. Do not rename or delete either historical directory:
their registered seeds and renderers are retained solely for byte-exact replay.
All new calibration and training gates bind to v0.3. Its integer ticks and grid
lines address the v0.2 odd-value interpolation imbalance, while its stricter
taxonomy separates operation-invariant perception errors from genuine
compensation without changing any numeric gate.

Run exactly one complete v0.3 calibration. If it exits nonzero, stop and
archive the new taxonomy rather than making another renderer adjustment or
relaxing the parser. v0.3 publishes versioned calibration filenames and refuses
another run if either one already exists; it never archives or replaces the
active attempt. Historical v0.1/v0.2 files remain in place. An interruption
before final publication leaves neither active records nor a summary and can be
retried; an ambiguous records-only state is deliberately fail-closed.

At each eventual training launch, the server audit and known-answer smoke run
again. The launcher regenerates the complete dataset from the committed seed
and compares every JSONL and PNG byte, verifies that model and dataset bytes
did not change during natural collection, and freezes stage-local copies of
the authorization reports. A previous `ready: true` report is therefore not a
training authorization. The explicit acknowledgement also represents human
review of the final GPU lock and vulnerability status;
these supply-chain approvals are manual rather than cryptographically
authenticated by the launcher.

## Recoverability Stage-1 v2 development-probe handoff

Bridge v1 is complete and must not be rerun. Its strict Stage 1 parsed zero of
300 outputs, so Stage 2 was never called and the recoverability hypothesis was
not tested. The next authorized server work is only a one-shot 24-scene
development probe of a question-free, exact-four-slot Stage-1 prompt. The
probe uses the frozen `dev` split, four scenes per chart-type by operation
stratum. It makes exactly 24 image calls, no retries, no Legacy or Stage-2
calls, and no training.

The sequence first verifies the new probe package without loading the model.
The probe command then revalidates the frozen v0.3 evidence, all five Bridge v1
failure artifacts, the dataset bundle, and the model snapshot before inference.
Both commands refuse to overwrite existing outputs.

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p /cloud/cloud-ssd1/recoverability-v1-evidence

# Retain all earlier preflight and bridge evidence. This is a new probe-only
# preflight and must have a new output path.
python experiments/recoverability_v1/00_stage1_v2_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_stage1_v2.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output /cloud/cloud-ssd1/recoverability-v1-evidence/stage1-v2-preflight.json

python experiments/recoverability_v1/04_stage1_v2_probe.py \
  --paths configs/paths.yaml \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --probe-config configs/recoverability/stage1_v2_probe.yaml \
  --server-package-lock configs/recoverability/server_package_lock_stage1_v2.yaml \
  --preflight-report /cloud/cloud-ssd1/recoverability-v1-evidence/stage1-v2-preflight.json \
  --external-evidence /cloud/cloud-ssd1/recoverability-v1-evidence/v0_3_external_evidence.json \
  --v03-records trajectories/natural/calibration_records_v0_3.jsonl \
  --bridge-v1-records outputs/recoverability_v1/cva_recoverability_bridge_v1/bridge_records.jsonl \
  --bridge-v1-report outputs/recoverability_v1/cva_recoverability_bridge_v1/bridge_report.json \
  --bridge-v1-diagnostic /cloud/cloud-ssd1/recoverability-v1-evidence/bridge-stage1-diagnostic.json \
  --bridge-v1-attempt-marker outputs/recoverability_v1/cva_recoverability_bridge_v1.attempted.json \
  --bridge-v1-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/bridge-console.log \
  --execute
```

Stop after the probe even if it exits zero. Return
`outputs/recoverability_v1/stage1_v2_dev_probe/probe_report.json`, the SHA-256
of `probe_records.jsonl`, and the console output for review. Do not rerun the
probe, run a full Bridge v2, or start Phase N, Phase C, Pilot A, Pilot B, or any
RL training in the same session.

If commit `52cb92f` stopped before model loading with
`Bridge v1 console log does not preserve bridge_exit=3`, preserve that failed
preflight and console as evidence. It did not create the Stage-1 v2 attempt
marker and did not consume the one-shot probe. After pulling the corrected
commit, first prove that both the probe output and attempt marker are absent,
then use new versioned preflight and console filenames; never delete or
overwrite the earlier files.
