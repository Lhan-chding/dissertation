# Validated pilot server

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

`requirements-gpu.in` is a candidate upper-layer snapshot based on current
official package releases. `constraints-gpu.txt` protects torch 2.8.0. It is
not yet evidence of server compatibility. After preflight and smoke succeed,
record `pip freeze --all` as `requirements-gpu.lock.txt` and commit it in a
separate reviewed change.

The official TRL documentation lists Qwen2.5-VL as a supported GRPO VLM, but
also warns that compatibility is not guaranteed for every model/version. The
local known-answer smoke therefore remains mandatory; the first registered
pilot must be monitored and stopped on any runtime incompatibility.

## First server commands

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
  --config configs/data/cva_chart_pilot.yaml

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

If calibration exits nonzero, the same command may be run again after the
reviewed task adjustment. The collector archives the prior complete failed
attempt under `trajectories/natural/attempts/failed-*`; a successful attempt is
never replaced. Interrupted collections publish neither final records nor a
summary and can also be rerun without deleting evidence manually.

At each eventual training launch, the server audit and known-answer smoke run
again. The launcher regenerates the complete dataset from the committed seed
and compares every JSONL and PNG byte, verifies that model and dataset bytes
did not change during natural collection, and freezes stage-local copies of
the authorization reports. A previous `ready: true` report is therefore not a
training authorization. The explicit acknowledgement also represents human
review of the final GPU lock and vulnerability status;
these supply-chain approvals are manual rather than cryptographically
authenticated by the launcher.
