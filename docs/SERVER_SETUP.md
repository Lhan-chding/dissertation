# Validated pilot server

> **Current boundary (2026-08-15).** The one permitted v0.3 calibration has
> already completed and failed. The older generation/calibration commands below
> are retained only as historical documentation and must not be run again. Pilot
> A/B are terminated. The Stage-2 v1 development probe has completed and must
> not be rerun. Its model-free diagnostic is complete. The one-shot 24-call,
> text-only Stage-2 v2 development probe also completed successfully and must
> not be rerun. The only next server action is a zero-model-call evidence replay
> at the end of this document. Bridge v2, Phase N/C, Pilot A/B, and all training
> remain unauthorized.

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

## Stage-2 v2 model-free evidence capture handoff

This is the only authorized next server action. It loads no model, makes no
model or CUDA calls, and does not run a hypothesis test or training. It verifies
the exact evidence-capture code package, checks all five externally supplied
artifact hashes, re-parses and executes every stored raw graph, and rejects any
stored flag or aggregate that does not reproduce. It writes one new evidence
manifest exclusively and refuses overwrite or path substitution.

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

Stop after this capture. Return the complete manifest,
`stage2_v2_capture_exit`, and its SHA-256. Do not rerun any probe or start
Bridge v2, Phase N, Phase C, Pilot A, Pilot B, or training.
