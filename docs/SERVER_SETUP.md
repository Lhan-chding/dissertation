# Validated pilot server

> **Current boundary (2026-08-15).** The one permitted v0.3 calibration has
> already completed and failed. The older generation/calibration commands below
> are retained only as historical documentation and must not be run again. Pilot
> A/B are terminated. The Stage-2 v1 development probe has completed and must
> not be rerun. Its model-free diagnostic is complete. The one-shot 24-call,
> text-only Stage-2 v2 development probe also completed successfully and must
> not be rerun. Its zero-model-call evidence replay also completed successfully
> and must not be rerun. The 300-scene measurement qualification subsequently
> passed and is frozen. Phase N then completed and remains inconclusive under
> its original `0.05` rule. A prospective `0.10` continuation amendment was
> frozen before any Phase-C outcome. The 8,000-scene Phase-C v2 screen is now
> the only authorized next server action. Bridge v2, six-arm execution, Pilot
> A/B, and all training remain unauthorized.

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

## Historical: completed Recoverability Stage-1 v2 development probe

Bridge v1 is complete and must not be rerun. Its strict Stage 1 parsed zero of
300 outputs, so Stage 2 was never called and the recoverability hypothesis was
not tested. The one-shot follow-up was a 24-scene development probe of a
question-free, exact-four-slot Stage-1 prompt. It used the frozen `dev` split,
four scenes per chart-type by operation stratum, and made exactly 24 image
calls with no retries, Legacy or Stage-2 calls, or training. That probe is now
complete; the following block is retained only as immutable run history and
must not be executed again.

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

The Stage-1 v2 probe subsequently completed successfully with `24/24` strict
parses and `22/24` exact transcriptions. Its report, records, preflight, and
console hashes are frozen in
`configs/recoverability/stage1_v2_frozen_result.yaml`. Do not execute the
historical commands above again.

## Historical: completed Recoverability Stage-2 v1 development probe

The completed one-shot action was a 24-scene, text-only Stage-2 interface
probe. It reused the frozen Stage-1 v2 raw outputs, performed no image calls,
used no retries, invoked no Legacy protocol, and did not test the recoverability
hypothesis. Each model output was required to be one strict executable DSL
object whose variables exactly equalled the frozen Stage-1 evidence. Passing
would have required all 24 programs to parse, execute, agree with their reported
answers, and equal the registered operation applied to the perceived values.

The preflight was metadata-only. The probe replayed the v0.3 calibration
evidence, the complete deterministic dataset, and all 24 Stage-1 v2 records
before model loading. It refused path overrides, symlinks, an existing attempt
marker, or an existing output directory. The following block is immutable run
history and must not be executed again:

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p /cloud/cloud-ssd1/recoverability-v1-evidence

STAGE2_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v1-preflight.json
STAGE2_CONSOLE=/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v1-console.log

for candidate in \
  "$STAGE2_PREFLIGHT" \
  "$STAGE2_CONSOLE" \
  outputs/recoverability_v1/stage2_v1_dev_probe.attempted.json \
  outputs/recoverability_v1/stage2_v1_dev_probe
do
  test ! -e "$candidate" || {
    echo "BLOCKED: Stage-2 v1 evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/00_stage2_v1_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_stage2_v1.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$STAGE2_PREFLIGHT"

(
  python experiments/recoverability_v1/05_stage2_v1_probe.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --probe-config configs/recoverability/stage2_v1_probe.yaml \
    --stage1-result configs/recoverability/stage1_v2_frozen_result.yaml \
    --server-package-lock configs/recoverability/server_package_lock_stage2_v1.yaml \
    --preflight-report "$STAGE2_PREFLIGHT" \
    --external-evidence /cloud/cloud-ssd1/recoverability-v1-evidence/v0_3_external_evidence.json \
    --v03-records trajectories/natural/calibration_records_v0_3.jsonl \
    --stage1-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/stage1-v2-preflight-v2.json \
    --stage1-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/stage1-v2-console-v2.log \
    --stage1-report outputs/recoverability_v1/stage1_v2_dev_probe/probe_report.json \
    --stage1-records outputs/recoverability_v1/stage1_v2_dev_probe/probe_records.jsonl \
    --execute
  stage2_rc=$?
  echo "stage2_probe_exit=$stage2_rc"
  exit "$stage2_rc"
) 2>&1 | tee "$STAGE2_CONSOLE"

stage2_rc=${PIPESTATUS[0]}
echo "stage2_probe_exit=$stage2_rc"

sha256sum "$STAGE2_PREFLIGHT" "$STAGE2_CONSOLE"
if [ -f outputs/recoverability_v1/stage2_v1_dev_probe/probe_report.json ]; then
  sha256sum \
    outputs/recoverability_v1/stage2_v1_dev_probe/probe_report.json \
    outputs/recoverability_v1/stage2_v1_dev_probe/probe_records.jsonl
fi
```

The probe returned exit `3`: `19/24` programs parsed and executed, `13/24`
answers matched their executed result, five failed strict parsing, and six had
program-answer mismatches. Its preflight, console, report, and raw records are
frozen in `configs/recoverability/stage2_v1_failure.yaml`. The historical block
above must not be executed again.

## Historical: completed Stage-2 v1 failure diagnostic

This deterministic replay of the 24 frozen outputs is complete. It verified
every supplied SHA-256, recomputed all stored parse/execution flags, grouped
failures by operation and raw-output signature, and recorded representative
malformed/mismatched outputs. It made zero model calls. The following block is
immutable run history and must not be executed again.

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src

DIAG_OUTPUT=/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v1-failure-diagnostic.json

test ! -e "$DIAG_OUTPUT" || {
  echo "BLOCKED: Stage-2 v1 diagnostic already exists"
  exit 1
}

python experiments/recoverability_v1/06_diagnose_stage2_v1_failure.py \
  --failure-config configs/recoverability/stage2_v1_failure.yaml \
  --diagnostic-package-lock configs/recoverability/server_package_lock_stage2_v1_diagnostic.yaml \
  --stage1-records outputs/recoverability_v1/stage1_v2_dev_probe/probe_records.jsonl \
  --stage2-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v1-preflight.json \
  --stage2-console /cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v1-console.log \
  --stage2-report outputs/recoverability_v1/stage2_v1_dev_probe/probe_report.json \
  --stage2-records outputs/recoverability_v1/stage2_v1_dev_probe/probe_records.jsonl \
  --output "$DIAG_OUTPUT"

diagnostic_rc=$?
echo "diagnostic_exit=$diagnostic_rc"
sha256sum "$DIAG_OUTPUT"
```

The replay returned `diagnostic_exit=0`, `verified=true`, and SHA-256
`d85510ea829a000bc31002f874e5a0ec795421aadec9f9042438d78337d9e7b4`.

## Historical: completed Recoverability Stage-2 v2 development probe

The completed model-facing action was one 24-scene, text-only development probe
of the executor-authoritative Stage-2 v2 interface. The model returned a
strict graph plus `"return":"result"`; it does not return a numeric answer.
The trusted executor supplies the sole numeric final answer. This probe uses no
images, retries, repair, Legacy calls, cues, or training and does not test the
recoverability hypothesis.

The metadata preflight loaded no model. Both commands verified the exact closed
server package. The probe additionally bound the frozen Stage-1 v2 record hash,
the completed Stage-2 v1 diagnostic hash, and the unchanged model snapshot. An
exclusive attempt marker was created before model loading. The following block
is immutable run history and must not be executed again.

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p /cloud/cloud-ssd1/recoverability-v1-evidence

STAGE2_V2_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-preflight.json
STAGE2_V2_CONSOLE=/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-console.log
STAGE2_V2_OUTPUT=outputs/recoverability_v1/stage2_v2_dev_probe
STAGE2_V2_ATTEMPT=outputs/recoverability_v1/stage2_v2_dev_probe.attempted.json

for candidate in \
  "$STAGE2_V2_PREFLIGHT" \
  "$STAGE2_V2_CONSOLE" \
  "$STAGE2_V2_OUTPUT" \
  "$STAGE2_V2_ATTEMPT"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: Stage-2 v2 evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/00_stage2_v2_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_stage2_v2.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$STAGE2_V2_PREFLIGHT"

stage2_v2_preflight_rc=$?
echo "stage2_v2_preflight_exit=$stage2_v2_preflight_rc"
test "$stage2_v2_preflight_rc" -eq 0 || exit "$stage2_v2_preflight_rc"

set -o pipefail
(
  python experiments/recoverability_v1/07_stage2_v2_probe.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --probe-config configs/recoverability/stage2_v2_probe.yaml \
    --stage1-result configs/recoverability/stage1_v2_frozen_result.yaml \
    --stage2-v1-diagnostic-anchor configs/recoverability/stage2_v1_diagnostic_result.yaml \
    --server-package-lock configs/recoverability/server_package_lock_stage2_v2.yaml \
    --preflight-report "$STAGE2_V2_PREFLIGHT" \
    --stage1-records outputs/recoverability_v1/stage1_v2_dev_probe/probe_records.jsonl \
    --stage2-v1-diagnostic /cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v1-failure-diagnostic.json \
    --execute
  stage2_v2_rc=$?
  echo "stage2_v2_probe_exit=$stage2_v2_rc"
  exit "$stage2_v2_rc"
) 2>&1 | tee "$STAGE2_V2_CONSOLE"
stage2_v2_rc=$?

echo "stage2_v2_probe_exit=$stage2_v2_rc"
sha256sum "$STAGE2_V2_PREFLIGHT" "$STAGE2_V2_CONSOLE"
if [ -f "$STAGE2_V2_OUTPUT/probe_report.json" ]; then
  sha256sum \
    "$STAGE2_V2_OUTPUT/probe_report.json" \
    "$STAGE2_V2_OUTPUT/probe_records.jsonl" \
    "$STAGE2_V2_ATTEMPT"
fi
```

The probe returned exit `0`: all `24/24` strict graphs parsed and executed, and
the trusted executor produced the registered operation result for every scene.
Its parse, execution, and executor-answer rates were all `1.0`; it used zero
retries and did not test a scientific hypothesis. The historical block above
must not be executed again.

## Historical: completed Stage-2 v2 model-free evidence capture

This completed action loaded no model, made no
model or CUDA calls, and did not run a hypothesis test or training. It verified
the exact evidence-capture code package, checks all five externally supplied
artifact hashes, re-parses and executes every stored raw graph, and rejects any
stored flag or aggregate that does not reproduce. It writes one new evidence
manifest exclusively and refuses overwrite or path substitution. The following
block is immutable run history and must not be executed again.

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src

STAGE2_V2_EVIDENCE=/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-external-evidence.json

test ! -e "$STAGE2_V2_EVIDENCE" || {
  echo "BLOCKED: Stage-2 v2 external evidence already exists"
  exit 1
}

python experiments/recoverability_v1/08_capture_stage2_v2_evidence.py \
  --frozen-result configs/recoverability/stage2_v2_frozen_result.yaml \
  --evidence-package-lock configs/recoverability/server_package_lock_stage2_v2_evidence.yaml \
  --paths configs/paths.yaml \
  --stage1-records outputs/recoverability_v1/stage1_v2_dev_probe/probe_records.jsonl \
  --stage2-v2-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-preflight.json \
  --stage2-v2-console /cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-console.log \
  --stage2-v2-report outputs/recoverability_v1/stage2_v2_dev_probe/probe_report.json \
  --stage2-v2-records outputs/recoverability_v1/stage2_v2_dev_probe/probe_records.jsonl \
  --attempt-marker outputs/recoverability_v1/stage2_v2_dev_probe.attempted.json \
  --output "$STAGE2_V2_EVIDENCE"

capture_rc=$?
echo "stage2_v2_capture_exit=$capture_rc"
test "$capture_rc" -eq 0 || exit "$capture_rc"
sha256sum "$STAGE2_V2_EVIDENCE"
```

The capture returned exit `0`, `verified=true`, and reproduced all 24 parses,
executions, and executor-correct results with `model_calls=0`. Its manifest
SHA-256 is
`3a9e521cfe718cc3dea9aee4f1591aac761fa47f893c986eb1ba722a44374577`.
It is anchored in
`configs/recoverability/stage2_v2_external_evidence_anchor.yaml` and must not be
rerun.

## Historical: completed model-free measurement-qualification data

This completed action did not load Qwen, import
Torch, make a model call, test a hypothesis, or authorize training. It verifies
the exact closed package and the frozen Stage-2 v2 external evidence, then
generates 300 new scenes with no numeric-table overlap with v0.3. It creates an
exclusive attempt marker before generation and refuses any rerun. The block
below is immutable run history and must not be executed again.

Pull the reviewed revision and run this block exactly once:

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

QUAL_ROOT=/cloud/cloud-ssd1/dissertation/data/generated/measurement_qualification_v1
QUAL_ATTEMPT=/cloud/cloud-ssd1/dissertation/data/generated/measurement_qualification_v1.attempted.json
QUAL_LOG=/cloud/cloud-ssd1/recoverability-v1-evidence/measurement-qualification-data.log

for candidate in "$QUAL_ROOT" "$QUAL_ATTEMPT" "$QUAL_LOG"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: measurement qualification data evidence already exists: $candidate"
    exit 1
  }
done

set -o pipefail
(
  python experiments/recoverability_v1/09_generate_measurement_qualification_data.py \
    --paths configs/paths.yaml \
    --config configs/recoverability/measurement_qualification_v1.yaml \
    --server-package-lock configs/recoverability/server_package_lock_measurement_qualification_data.yaml \
    --stage2-v2-external-evidence /cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-external-evidence.json \
    --execute
  qualification_data_rc=$?
  echo "qualification_data_exit=$qualification_data_rc"
  exit "$qualification_data_rc"
) 2>&1 | tee "$QUAL_LOG"
qualification_data_rc=$?

echo "qualification_data_exit=$qualification_data_rc"
test "$qualification_data_rc" -eq 0 || exit "$qualification_data_rc"
sha256sum \
  "$QUAL_ATTEMPT" \
  "$QUAL_ROOT/manifest.json" \
  "$QUAL_ROOT/records.jsonl" \
  "$QUAL_LOG"
```

The historical generation package-lock SHA-256 is
`25808fffdf62981163550084c36c4b37428fada7c85bc8b3a5e286b4bc75ec4c`.
The run completed at revision `93bd5f5` with `qualification_data_exit=0`.
Its 300 records, six balanced strata, manifest, images, attempt marker, and
console log are frozen in
`configs/recoverability/measurement_qualification_data_anchor.yaml`.

## Historical: completed one-shot measurement qualification

This completed action was an interface
qualification, not Bridge v2 or a scientific hypothesis test. The metadata
preflight is model-free. The subsequent execution makes one Stage-1 v2 call
per frozen scene and one Stage-2 v2 call only after a strict Stage-1 parse, for
at most 600 calls total and zero retries. An exclusive attempt marker is
created before Qwen is loaded. Exit `0` means the registered interface gate
passed; exit `3` means it failed. Both are final evidence and neither
authorizes Phase N, Phase C, RL, or training.

Pull the reviewed revision and run this block exactly once:

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

QUAL_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/measurement-qualification-preflight.json
QUAL_OUTPUT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/measurement_qualification_v1
QUAL_RUN_ATTEMPT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/measurement_qualification_v1.attempted.json
QUAL_RUN_LOG=/cloud/cloud-ssd1/recoverability-v1-evidence/measurement-qualification-console.log

for candidate in "$QUAL_PREFLIGHT" "$QUAL_OUTPUT" "$QUAL_RUN_ATTEMPT" "$QUAL_RUN_LOG"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: measurement qualification evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/10_measurement_qualification_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_measurement_qualification.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$QUAL_PREFLIGHT"

qualification_preflight_rc=$?
echo "qualification_preflight_exit=$qualification_preflight_rc"
test "$qualification_preflight_rc" -eq 0 || exit "$qualification_preflight_rc"

set -o pipefail
(
  python experiments/recoverability_v1/11_run_measurement_qualification.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --config configs/recoverability/measurement_qualification_v1.yaml \
    --data-anchor configs/recoverability/measurement_qualification_data_anchor.yaml \
    --server-package-lock configs/recoverability/server_package_lock_measurement_qualification.yaml \
    --preflight-report "$QUAL_PREFLIGHT" \
    --dataset-root /cloud/cloud-ssd1/dissertation/data/generated/measurement_qualification_v1 \
    --dataset-attempt-marker /cloud/cloud-ssd1/dissertation/data/generated/measurement_qualification_v1.attempted.json \
    --dataset-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/measurement-qualification-data.log \
    --source-records /cloud/cloud-ssd1/dissertation/data/generated/cva_chart_pilot_v0_3/records.jsonl \
    --stage2-v2-external-evidence /cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-external-evidence.json \
    --execute
  qualification_rc=$?
  echo "measurement_qualification_exit=$qualification_rc"
  exit "$qualification_rc"
) 2>&1 | tee "$QUAL_RUN_LOG"
qualification_rc=$?

echo "measurement_qualification_exit=$qualification_rc"
case "$qualification_rc" in
  0|3) ;;
  *) echo "BLOCKED: unexpected qualification failure"; exit "$qualification_rc" ;;
esac

sha256sum \
  "$QUAL_PREFLIGHT" \
  "$QUAL_RUN_ATTEMPT" \
  "$QUAL_OUTPUT/qualification_report.json" \
  "$QUAL_OUTPUT/qualification_records.jsonl" \
  "$QUAL_RUN_LOG"
```

After exit `0` or `3`, stop. Return the complete printed qualification report,
the five SHA-256 lines, both exit-code lines, and the short Git revision. Do
not run Bridge v2, Phase N, Phase C, Pilot A/B, another qualification attempt,
RL, or any training.

The run passed at revision `8511dbc`: Stage 1 parsed `299/300`; all 299
downstream Stage-2 programs parsed, executed, and returned the correct trusted
executor result. Its exact result is frozen in
`configs/recoverability/measurement_qualification_frozen_result.yaml` and the
block above must not be rerun.

The registered measurement-qualification execution package-lock SHA-256 is
`a4179f3e4c6f90f6730ad15d3f38a4309564b1099459cfd1d3918cc7f36de691`.

## Completed historical action: one-shot Phase N natural-prevalence screen

This completed action generated 4,000 fresh,
fixed, v0.3-rendered scenes that are disjoint from the earlier v0.3 and
qualification numeric tables, then makes exactly one original unified-protocol
image call per scene. There are zero retries, no sample extension, no
draw-until-error loop, no Stage 2, and no training. The qualification artifacts
are hash-verified before Qwen loads. An exclusive Phase N attempt marker is
created before dataset generation or model loading, so the job cannot be
selected among reruns.

Exit `0` means the one-sided registered low-prevalence claim passed both its
confidence and minimum-eligible gates. Exit `3` means the result is
inconclusive. Both are expected final scientific outcomes; after either one,
stop and return the report and hashes.

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PHASE_N_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-n-preflight.json
PHASE_N_DATA=/cloud/cloud-ssd1/dissertation/data/generated/cva_natural_prevalence_v1
PHASE_N_OUTPUT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_natural_prevalence_v1
PHASE_N_ATTEMPT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_natural_prevalence_v1.attempted.json
PHASE_N_LOG=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-n-console.log

for candidate in \
  "$PHASE_N_PREFLIGHT" \
  "$PHASE_N_DATA" \
  "$PHASE_N_OUTPUT" \
  "$PHASE_N_ATTEMPT" \
  "$PHASE_N_LOG"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: Phase N evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/12_phase_n_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_phase_n.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$PHASE_N_PREFLIGHT"

phase_n_preflight_rc=$?
echo "phase_n_preflight_exit=$phase_n_preflight_rc"
test "$phase_n_preflight_rc" -eq 0 || exit "$phase_n_preflight_rc"

set -o pipefail
(
  python experiments/recoverability_v1/13_run_phase_n.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --protocol configs/recoverability/recoverability_v1.yaml \
    --server-package-lock configs/recoverability/server_package_lock_phase_n.yaml \
    --preflight-report "$PHASE_N_PREFLIGHT" \
    --qualification-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/measurement-qualification-preflight.json \
    --qualification-attempt-marker /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/measurement_qualification_v1.attempted.json \
    --qualification-report /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/measurement_qualification_v1/qualification_report.json \
    --qualification-records /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/measurement_qualification_v1/qualification_records.jsonl \
    --qualification-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/measurement-qualification-console.log \
    --source-records /cloud/cloud-ssd1/dissertation/data/generated/cva_chart_pilot_v0_3/records.jsonl \
    --qualification-dataset-root /cloud/cloud-ssd1/dissertation/data/generated/measurement_qualification_v1 \
    --qualification-dataset-attempt-marker /cloud/cloud-ssd1/dissertation/data/generated/measurement_qualification_v1.attempted.json \
    --qualification-dataset-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/measurement-qualification-data.log \
    --execute
  phase_n_rc=$?
  echo "phase_n_exit=$phase_n_rc"
  exit "$phase_n_rc"
) 2>&1 | tee "$PHASE_N_LOG"
phase_n_rc=$?

echo "phase_n_exit=$phase_n_rc"
case "$phase_n_rc" in
  0|3) ;;
  *) echo "BLOCKED: unexpected Phase N execution failure"; exit "$phase_n_rc" ;;
esac

sha256sum \
  "$PHASE_N_PREFLIGHT" \
  "$PHASE_N_ATTEMPT" \
  "$PHASE_N_DATA/manifest.json" \
  "$PHASE_N_DATA/records.jsonl" \
  "$PHASE_N_OUTPUT/phase_n_report.json" \
  "$PHASE_N_OUTPUT/phase_n_records.jsonl" \
  "$PHASE_N_LOG"
```

Do not rerun Phase N. Its original `0.05` decision remains inconclusive and its
evidence is frozen. The prospective Phase-C v2 amendment below was registered
after Phase N and before any Phase-C outcome.

## Next authorized action: amended confirmatory Phase-C v2 screen

This one-shot step generates 8,000 new v0.3-rendered scenes balanced over three
redundancy families, two chart types, and three operations. Numeric tables are
disjoint from the source, qualification, and Phase-N datasets. It makes exactly
8,000 zero-retry Stage-1 v2 calls, retains only parsed one-position,
operator-sensitive errors whose registered cue uniquely recovers the answer,
and deterministically selects 400 cross-series, 400 trend, and 266
duplicate-encoding scenes. It does not execute any arm or fork and cannot train.

The original Phase-N result remains inconclusive at the original `0.05` rule.
The v2 continuation amendment uses `0.10`; the observed one-sided upper bound
`0.05243` passes only that amended continuation rule. Exit `0` means every
frozen family quota was filled and authorizes packaging the six-arm execution.
Exit `3` means final quota underfill; do not rerun or extend the screen.

```bash
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PHASE_C_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-preflight.json
PHASE_C_DATA=/cloud/cloud-ssd1/dissertation/data/generated/cva_recoverability_causal_v2_screen
PHASE_C_OUTPUT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2/phase_c_screen
PHASE_C_ATTEMPT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2.screen.attempted.json
PHASE_C_LOG=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-console.log

for candidate in \
  "$PHASE_C_PREFLIGHT" \
  "$PHASE_C_DATA" \
  "$PHASE_C_OUTPUT" \
  "$PHASE_C_ATTEMPT" \
  "$PHASE_C_LOG"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: Phase C screen evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/14_phase_c_screen_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_phase_c_screen_v2.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$PHASE_C_PREFLIGHT"

phase_c_preflight_rc=$?
echo "phase_c_screen_preflight_exit=$phase_c_preflight_rc"
test "$phase_c_preflight_rc" -eq 0 || exit "$phase_c_preflight_rc"

set -o pipefail
(
  python experiments/recoverability_v1/15_run_phase_c_screen.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --amendment configs/recoverability/recoverability_phase_c_v2_amendment.yaml \
    --phase-n-result configs/recoverability/phase_n_frozen_result.yaml \
    --server-package-lock configs/recoverability/server_package_lock_phase_c_screen_v2.yaml \
    --preflight-report "$PHASE_C_PREFLIGHT" \
    --phase-n-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/phase-n-preflight.json \
    --phase-n-attempt-marker /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_natural_prevalence_v1.attempted.json \
    --phase-n-dataset-root /cloud/cloud-ssd1/dissertation/data/generated/cva_natural_prevalence_v1 \
    --phase-n-output-root /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_natural_prevalence_v1 \
    --phase-n-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/phase-n-console.log \
    --source-records /cloud/cloud-ssd1/dissertation/data/generated/cva_chart_pilot_v0_3/records.jsonl \
    --qualification-records /cloud/cloud-ssd1/dissertation/data/generated/measurement_qualification_v1/records.jsonl \
    --execute
  phase_c_rc=$?
  echo "phase_c_screen_exit=$phase_c_rc"
  exit "$phase_c_rc"
) 2>&1 | tee "$PHASE_C_LOG"
phase_c_rc=$?

echo "phase_c_screen_exit=$phase_c_rc"
case "$phase_c_rc" in
  0|3) ;;
  *) echo "BLOCKED: unexpected Phase C screen failure"; exit "$phase_c_rc" ;;
esac

sha256sum \
  "$PHASE_C_PREFLIGHT" \
  "$PHASE_C_ATTEMPT" \
  "$PHASE_C_DATA/manifest.json" \
  "$PHASE_C_DATA/records.jsonl" \
  "$PHASE_C_OUTPUT/screen_report.json" \
  "$PHASE_C_OUTPUT/screen_records.jsonl" \
  "$PHASE_C_LOG"
```

After exit `0` or `3`, stop and return the full report, seven SHA-256 lines,
both screen exit-code lines, the preflight exit, and the short Git revision.
Do not rerun the screen. Even after exit `0`, do not start training: the next
separately packaged action is the frozen six-arm, eight-fork causal experiment.

## Next authorized action: Phase-C v3 frozen six-arm execution

The v2 screen has already exited `3` and must not be rerun. Its 580 individually
eligible scenes are frozen. The v3 amendment removes only the internally chosen
400/400/266 availability quotas as an execution gate, before any arm outcome is
observed. It preserves the original failed screen report and records that the
0.90 power target is not met. This command runs all six original arms and eight
fixed forks for all 580 scenes: exactly 27,840 text-only model calls. It cannot
train and leaves `rl_authorized=false`.

Run inside `tmux`. The outer subshell ensures that any fail-closed `exit` stops
only this workflow and does not close the interactive tmux shell:

```bash
(
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PHASE_C_ARMS_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-arms-v3r1-preflight.json
PHASE_C_ARMS_OUTPUT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v3/phase_c_arms
PHASE_C_ARMS_ATTEMPT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v3.arms.attempted.json
PHASE_C_ARMS_LOG=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-arms-v3r1-console.log

for candidate in \
  "$PHASE_C_ARMS_PREFLIGHT" \
  "$PHASE_C_ARMS_OUTPUT" \
  "$PHASE_C_ARMS_ATTEMPT" \
  "$PHASE_C_ARMS_LOG"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: Phase C arm evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/16_phase_c_arm_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_phase_c_arms_v3.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$PHASE_C_ARMS_PREFLIGHT"

phase_c_arms_preflight_rc=$?
echo "phase_c_arms_preflight_exit=$phase_c_arms_preflight_rc"
test "$phase_c_arms_preflight_rc" -eq 0 || exit "$phase_c_arms_preflight_rc"

set -o pipefail
(
  python experiments/recoverability_v1/17_run_phase_c_arms.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --postscreen-amendment configs/recoverability/recoverability_phase_c_v3_postscreen_amendment.yaml \
    --screen-result configs/recoverability/phase_c_screen_v2_frozen_result.yaml \
    --server-package-lock configs/recoverability/server_package_lock_phase_c_arms_v3.yaml \
    --preflight-report "$PHASE_C_ARMS_PREFLIGHT" \
    --screen-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-preflight.json \
    --screen-attempt-marker /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2.screen.attempted.json \
    --screen-dataset-root /cloud/cloud-ssd1/dissertation/data/generated/cva_recoverability_causal_v2_screen \
    --screen-output-root /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2/phase_c_screen \
    --screen-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-console.log \
    --execute
  phase_c_arms_rc=$?
  echo "phase_c_arms_exit=$phase_c_arms_rc"
  exit "$phase_c_arms_rc"
) 2>&1 | tee "$PHASE_C_ARMS_LOG"
phase_c_arms_rc=$?

echo "phase_c_arms_exit=$phase_c_arms_rc"
test "$phase_c_arms_rc" -eq 0 || {
  echo "BLOCKED: unexpected Phase C arm execution failure"
  exit "$phase_c_arms_rc"
}

sha256sum \
  "$PHASE_C_ARMS_PREFLIGHT" \
  "$PHASE_C_ARMS_ATTEMPT" \
  "$PHASE_C_ARMS_OUTPUT/arm_report.json" \
  "$PHASE_C_ARMS_OUTPUT/arm_records.jsonl" \
  "$PHASE_C_ARMS_LOG"
)
phase_c_workflow_rc=$?
echo "phase_c_workflow_exit=$phase_c_workflow_rc"
```

Detach from `tmux` with `Ctrl-b`, then `d`; reconnect with `tmux attach -t
test`. Completion means data collection and the frozen paired analysis finished,
not that RL or training was authorized.

## Low-cost Phase-C prompt qualification after the unusable v3 arm interface

The completed v3 arm run must not be rerun. It made all 27,840 calls, but the
strict result-program parser accepted none of them. The raw responses are
preserved; this is an interface failure, not evidence that every model answer
was semantically wrong. Before considering another large run, the repaired
prompt is tested on only nine frozen scenes: one deterministic scene for every
family-by-operation cell. Each scene receives `no_cue` and `valid_cue`, with two
fixed forks per condition, for exactly `3 * 3 * 2 * 2 = 36` text-only calls.

The prompt now defines the index mapping, all three constraint kinds, the
one-error rule, and the exact operation-specific JSON template. The report
separates strict-format parsing from format-independent extraction of the four
inferred values and their computed answer. This diagnostic performs no
hypothesis test and always leaves `scale_authorized=false`, `rl_authorized=false`,
and `training_authorized=false`; the human decision about whether the prompt is
good enough comes only after inspecting its raw 36 records.

Run this once. It uses new paths and does not overwrite the v3 arm evidence:

```bash
(
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PROMPT_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-prompt-qualification-v1-preflight.json
PROMPT_OUTPUT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v3/phase_c_prompt_qualification_v1
PROMPT_ATTEMPT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v3.prompt-qualification-v1.attempted.json
PROMPT_LOG=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-prompt-qualification-v1-console.log

for candidate in \
  "$PROMPT_PREFLIGHT" \
  "$PROMPT_OUTPUT" \
  "$PROMPT_ATTEMPT" \
  "$PROMPT_LOG"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: prompt qualification evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/18_phase_c_prompt_qualification_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_phase_c_prompt_qualification_v1.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$PROMPT_PREFLIGHT"

prompt_preflight_rc=$?
echo "prompt_qualification_preflight_exit=$prompt_preflight_rc"
test "$prompt_preflight_rc" -eq 0 || exit "$prompt_preflight_rc"

set -o pipefail
(
  python experiments/recoverability_v1/19_run_phase_c_prompt_qualification.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --qualification-config configs/recoverability/phase_c_prompt_qualification_v1.yaml \
    --screen-result configs/recoverability/phase_c_screen_v2_frozen_result.yaml \
    --server-package-lock configs/recoverability/server_package_lock_phase_c_prompt_qualification_v1.yaml \
    --preflight-report "$PROMPT_PREFLIGHT" \
    --screen-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-preflight.json \
    --screen-attempt-marker /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2.screen.attempted.json \
    --screen-dataset-root /cloud/cloud-ssd1/dissertation/data/generated/cva_recoverability_causal_v2_screen \
    --screen-output-root /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2/phase_c_screen \
    --screen-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-console.log \
    --execute
  prompt_rc=$?
  echo "prompt_qualification_exit=$prompt_rc"
  exit "$prompt_rc"
) 2>&1 | tee "$PROMPT_LOG"
prompt_rc=$?

echo "prompt_qualification_exit=$prompt_rc"
test "$prompt_rc" -eq 0 || exit "$prompt_rc"

python -m json.tool "$PROMPT_OUTPUT/prompt_qualification_report.json"

sha256sum \
  "$PROMPT_PREFLIGHT" \
  "$PROMPT_ATTEMPT" \
  "$PROMPT_OUTPUT/prompt_qualification_report.json" \
  "$PROMPT_OUTPUT/prompt_qualification_records.jsonl" \
  "$PROMPT_LOG"
)
prompt_workflow_rc=$?
echo "prompt_workflow_exit=$prompt_workflow_rc"
```

Return the full report, five SHA-256 lines, both prompt exit-code lines, the
preflight exit, and the short Git revision. Do not start another large run.

## One-shot Phase-C world-recovery-only 12-call diagnostic v1r1

The completed 36-call prompt qualification must not be rerun. This new workflow
uses separate paths and performs exactly 12 greedy, text-only model calls. It
writes the six-case hidden/public manifests and all rendered messages before
loading the model. The original v1 launch stopped before model loading or any
model call because one source scene was not uniquely recoverable under the
world-only criterion. Revision v1r1 preserves that failed preflight/log evidence,
filters such ambiguous source scenes before deterministic seeded selection, and
uses fresh evidence paths. Run the entire block once inside `tmux`:

```bash
(
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

WORLD_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-world-recovery-v1r1-preflight.json
WORLD_OUTPUT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v3/phase_c_world_recovery_v1r1
WORLD_ATTEMPT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v3.world-recovery-v1r1.attempted.json
WORLD_LOG=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-world-recovery-v1r1-console.log

for candidate in \
  "$WORLD_PREFLIGHT" \
  "$WORLD_OUTPUT" \
  "$WORLD_ATTEMPT" \
  "$WORLD_LOG"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: world recovery evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/20_phase_c_world_recovery_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --qualification-config configs/recoverability/phase_c_world_recovery_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_phase_c_world_recovery_v1.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$WORLD_PREFLIGHT"

world_preflight_rc=$?
echo "world_recovery_preflight_exit=$world_preflight_rc"
test "$world_preflight_rc" -eq 0 || exit "$world_preflight_rc"

set -o pipefail
(
  python experiments/recoverability_v1/21_run_phase_c_world_recovery.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --qualification-config configs/recoverability/phase_c_world_recovery_v1.yaml \
    --system-prompt prompts/world_recovery_v1_main.system.txt \
    --screen-result configs/recoverability/phase_c_screen_v2_frozen_result.yaml \
    --server-package-lock configs/recoverability/server_package_lock_phase_c_world_recovery_v1.yaml \
    --preflight-report "$WORLD_PREFLIGHT" \
    --screen-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-preflight.json \
    --screen-attempt-marker /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2.screen.attempted.json \
    --screen-dataset-root /cloud/cloud-ssd1/dissertation/data/generated/cva_recoverability_causal_v2_screen \
    --screen-output-root /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2/phase_c_screen \
    --screen-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-console.log \
    --execute
  world_rc=$?
  echo "world_recovery_exit=$world_rc"
  exit "$world_rc"
) 2>&1 | tee "$WORLD_LOG"
world_rc=$?

echo "world_recovery_exit=$world_rc"
test "$world_rc" -eq 0 || exit "$world_rc"

python -m json.tool "$WORLD_OUTPUT/world_recovery_report.json"

sha256sum \
  "$WORLD_PREFLIGHT" \
  "$WORLD_ATTEMPT" \
  "$WORLD_OUTPUT/manifest.hidden.jsonl" \
  "$WORLD_OUTPUT/manifest.public.jsonl" \
  "$WORLD_OUTPUT/messages.jsonl" \
  "$WORLD_OUTPUT/world_recovery_report.json" \
  "$WORLD_OUTPUT/world_recovery_records.jsonl" \
  "$WORLD_LOG"
)
world_workflow_rc=$?
echo "world_recovery_workflow_exit=$world_workflow_rc"
```

Return the full report, eight SHA-256 lines, the preflight/runner/workflow exit
codes, and the short Git revision. Do not tune the prompt on these six cases or
add calls after inspecting the result.

## One-shot Phase-C nontrivial world-recovery 100-call audit

The completed v1r1 diagnostic must not be rerun. This user-authorized follow-up
uses the identical frozen prompt and greedy decoder on 25 paired cases from each
nontrivial family (`cross_series` and `trend`). Each scene receives `no_cue` and
`valid_cue`, giving exactly 100 inference calls. It excludes the
`duplicate_encoding` full-state restatement control, performs no training or RL,
and uses fresh exclusive evidence paths. Run this entire block once inside
`tmux`:

```bash
(
cd /cloud/cloud-ssd1/dissertation
git -c http.version=HTTP/1.1 pull --ff-only origin main
git rev-parse --short HEAD
source .venv/bin/activate

export PYTHONPATH=/cloud/cloud-ssd1/dissertation/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

WORLD100_PREFLIGHT=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-world-recovery-100-v1-preflight.json
WORLD100_OUTPUT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v3/phase_c_world_recovery_100_v1
WORLD100_ATTEMPT=/cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v3.world-recovery-100-v1.attempted.json
WORLD100_LOG=/cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-world-recovery-100-v1-console.log

for candidate in \
  "$WORLD100_PREFLIGHT" \
  "$WORLD100_OUTPUT" \
  "$WORLD100_ATTEMPT" \
  "$WORLD100_LOG"
do
  test ! -e "$candidate" || {
    echo "BLOCKED: 100-call world recovery evidence already exists: $candidate"
    exit 1
  }
done

python experiments/recoverability_v1/20_phase_c_world_recovery_preflight.py \
  --runtime configs/recoverability/server_runtime_v1.yaml \
  --qualification-config configs/recoverability/phase_c_world_recovery_100_v1.yaml \
  --server-package-lock configs/recoverability/server_package_lock_phase_c_world_recovery_100_v1.yaml \
  --project-root /cloud/cloud-ssd1/dissertation \
  --output "$WORLD100_PREFLIGHT"

world100_preflight_rc=$?
echo "world_recovery_100_preflight_exit=$world100_preflight_rc"
test "$world100_preflight_rc" -eq 0 || exit "$world100_preflight_rc"

set -o pipefail
(
  python experiments/recoverability_v1/21_run_phase_c_world_recovery.py \
    --paths configs/paths.yaml \
    --runtime configs/recoverability/server_runtime_v1.yaml \
    --qualification-config configs/recoverability/phase_c_world_recovery_100_v1.yaml \
    --system-prompt prompts/world_recovery_v1_main.system.txt \
    --screen-result configs/recoverability/phase_c_screen_v2_frozen_result.yaml \
    --server-package-lock configs/recoverability/server_package_lock_phase_c_world_recovery_100_v1.yaml \
    --preflight-report "$WORLD100_PREFLIGHT" \
    --screen-preflight /cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-preflight.json \
    --screen-attempt-marker /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2.screen.attempted.json \
    --screen-dataset-root /cloud/cloud-ssd1/dissertation/data/generated/cva_recoverability_causal_v2_screen \
    --screen-output-root /cloud/cloud-ssd1/dissertation/outputs/recoverability_v1/cva_recoverability_causal_v2/phase_c_screen \
    --screen-console-log /cloud/cloud-ssd1/recoverability-v1-evidence/phase-c-screen-v2-console.log \
    --execute
  world100_rc=$?
  echo "world_recovery_100_exit=$world100_rc"
  exit "$world100_rc"
) 2>&1 | tee "$WORLD100_LOG"
world100_rc=$?

echo "world_recovery_100_exit=$world100_rc"
test "$world100_rc" -eq 0 || exit "$world100_rc"

python -m json.tool "$WORLD100_OUTPUT/world_recovery_report.json"

sha256sum \
  "$WORLD100_PREFLIGHT" \
  "$WORLD100_ATTEMPT" \
  "$WORLD100_OUTPUT/manifest.hidden.jsonl" \
  "$WORLD100_OUTPUT/manifest.public.jsonl" \
  "$WORLD100_OUTPUT/messages.jsonl" \
  "$WORLD100_OUTPUT/world_recovery_report.json" \
  "$WORLD100_OUTPUT/world_recovery_records.jsonl" \
  "$WORLD100_LOG"
)
world100_workflow_rc=$?
echo "world_recovery_100_workflow_exit=$world100_workflow_rc"
```

Return the full report, eight SHA-256 lines, all three exit codes, and the short
Git revision. Do not tune the prompt, replace cases, or add calls after seeing
the result.
