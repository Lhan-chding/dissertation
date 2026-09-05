#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=override-limits-but-killable
#SBATCH --job-name=louis-ssvc-p1
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/varunssd/louis-ssvc/dissertation-ssvc/ssvc_flow
#SBATCH --output=/projects/varunssd/louis-ssvc/logs/p1-%j.out
#SBATCH --error=/projects/varunssd/louis-ssvc/logs/p1-%j.err
#SBATCH --open-mode=append

# Submit with sbatch, not bash. Slurm copies this file into its spool, so $0
# cannot locate the repository. The personal environment is already installed.
set -euo pipefail
umask 077
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ || ! ${SLURM_RESTART_COUNT:-0} =~ ^[0-9]+$ ]]; then
  echo 'Submit with sbatch; a numeric Slurm job ID and restart count are required.' >&2
  exit 2
fi
export SSVC_WORK="${SSVC_WORK:-/projects/varunssd/louis-ssvc}"
export SSVC_PYTHON="${SSVC_PYTHON:-$SSVC_WORK/envs/ssvc-py312/bin/python}"
ssvc_root="$SSVC_WORK/dissertation-ssvc/ssvc_flow"
if [[ $SSVC_WORK != /* || ! -x $SSVC_PYTHON || ! -f $ssvc_root/configs/locked.json ]]; then
  echo 'Expected an absolute personal work directory, installed Python and SSVC checkout.' >&2
  exit 2
fi
cd "$ssvc_root"
ssvc_name="slurm-${SLURM_JOB_ID}-attempt-${SLURM_RESTART_COUNT:-0}"
export SSVC_RUN_DIR="$SSVC_WORK/runs/$ssvc_name"
mkdir -p "$SSVC_WORK/runs"
# Atomic claim: never truncate evidence from an earlier invocation.
if ! mkdir "$SSVC_RUN_DIR"; then
  echo "Refusing to reuse existing attempt: $SSVC_RUN_DIR" >&2
  exit 2
fi
mkdir -p "$SSVC_RUN_DIR/checks" "$SSVC_RUN_DIR/runs" "$SSVC_RUN_DIR/tmp"
export TMPDIR="$SSVC_RUN_DIR/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$SSVC_WORK/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda-pkgs"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export PIP_CONFIG_FILE=/dev/null PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" \
  "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"

ssvc_stage=initialization
write_result() {
  printf 'state=%s\nexit_code=%s\nstage=%s\njob_id=%s\nfinished_utc=%s\n' \
    "$1" "$2" "$ssvc_stage" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" \
    > "$SSVC_RUN_DIR/result.txt"
}
finish() {
  local rc=$? report_rc archive_rc status_rc
  trap - EXIT
  set +e
  # Reporting is useful on failure too, but must never hide the original error.
  "$SSVC_PYTHON" -m src.report --run-root "$SSVC_RUN_DIR/runs" \
    --out "$SSVC_RUN_DIR/reports" > "$SSVC_RUN_DIR/checks/report.log" 2>&1
  report_rc=$?
  if (( rc == 0 && report_rc != 0 )); then rc=$report_rc; ssvc_stage=report; fi
  # Do not include changing temp files or the final packaging status in the tar.
  tar --exclude="$ssvc_name/tmp" --exclude="$ssvc_name/result.txt" \
    -czf "$SSVC_RUN_DIR.tar.gz.partial" -C "$SSVC_WORK/runs" "$ssvc_name" \
    > "$SSVC_RUN_DIR.packaging.log" 2>&1
  archive_rc=$?
  if (( archive_rc == 0 )); then
    mv "$SSVC_RUN_DIR.tar.gz.partial" "$SSVC_RUN_DIR.tar.gz"
    archive_rc=$?
  fi
  if (( archive_rc == 0 )); then
    "$SSVC_PYTHON" - "$SSVC_RUN_DIR.tar.gz" <<'PY'
import hashlib
import sys
from pathlib import Path

archive = Path(sys.argv[1])
with archive.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
print(f"Evidence archive: {archive}\nSHA-256: {digest}")
PY
    archive_rc=$?
  fi
  if (( rc == 0 && archive_rc != 0 )); then rc=$archive_rc; ssvc_stage=archive; fi
  if (( rc == 0 )); then write_result PASS 0; else write_result FAILED "$rc"; fi
  status_rc=$?
  if (( status_rc != 0 )); then
    echo 'Could not write final status; consult Slurm accounting and the raw evidence.' >&2
    if (( rc == 0 )); then rc=$status_rc; fi
  fi
  echo "Batch exit=$rc; final status: $SSVC_RUN_DIR/result.txt"
  exit "$rc"
}
trap finish EXIT
# SIGKILL cannot be trapped; Slurm accounting and already-written files remain.
trap 'exit 143' TERM
trap 'exit 130' INT
run_check() {
  ssvc_stage=$1
  shift
  write_result RUNNING pending
  echo "[$(date -u +%FT%TZ)] $ssvc_stage -> $SSVC_RUN_DIR/checks/$ssvc_stage.log"
  "$@" > "$SSVC_RUN_DIR/checks/$ssvc_stage.log" 2>&1
}
echo "P1 evidence directory: $SSVC_RUN_DIR"
cp "$0" "$SSVC_RUN_DIR/submitted.sbatch"
cp configs/locked.json "$SSVC_RUN_DIR/locked-config.json"
git rev-parse HEAD > "$SSVC_RUN_DIR/source-commit.txt"
hostname > "$SSVC_RUN_DIR/hostname.txt"
run_check gpu nvidia-smi
run_check pip-check "$SSVC_PYTHON" -m pip check
run_check pip-freeze "$SSVC_PYTHON" -m pip freeze --all
run_check cuda "$SSVC_PYTHON" - <<'PY'
import importlib.metadata
import json
import sys

import torch
import torchvision
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import LoraConfig
from accelerate import Accelerator

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.cuda.is_available(), "No CUDA device in this allocation"
assert torch.cuda.device_count() == 1, "Expected exactly one allocated visible GPU"
assert torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 is required"
torch.manual_seed(17)
torch.backends.cuda.matmul.allow_tf32 = False
weight = torch.nn.Parameter(torch.randn(128, 128, device="cuda", dtype=torch.float32))
before = weight.detach().clone()
optimizer = torch.optim.AdamW([weight], lr=1e-3)
x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = x @ weight
    loss = output.float().square().mean()
assert output.dtype == torch.bfloat16 and torch.isfinite(loss).item()
loss.backward()
assert weight.grad is not None and torch.isfinite(weight.grad).all().item()
assert weight.grad.norm().item() > 0
optimizer.step()
torch.cuda.synchronize()
assert torch.isfinite(weight).all().item() and not torch.equal(before, weight)
device = torch.cuda.get_device_properties(0)
print(json.dumps({
    "python": sys.executable,
    "versions": {name: importlib.metadata.version(name) for name in
                 ("torch", "torchvision", "transformers", "peft", "accelerate")},
    "cuda_runtime": torch.version.cuda,
    "gpu": device.name,
    "capability": torch.cuda.get_device_capability(0),
    "vram_bytes": device.total_memory,
    "compiled_architectures": torch.cuda.get_arch_list(),
    "native_bf16_forward_backward_optimizer": "PASS",
    "loss": loss.item(),
}, indent=2))
PY
run_check cpu-tests env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$SSVC_PYTHON" -m pytest tests -q \
  --basetemp="$TMPDIR/pytest" -o "cache_dir=$TMPDIR/pytest-cache" \
  --junitxml="$SSVC_RUN_DIR/checks/cpu-tests.xml"
run_check data "$SSVC_PYTHON" -m src.generate_worlds \
  --out "$SSVC_RUN_DIR/data/generated" --seed 17
run_check budget "$SSVC_PYTHON" -m src.rollout --phase smoke --model qwen35_9b \
  --dry-run --out "$SSVC_RUN_DIR/budget"
run_check p1 "$SSVC_PYTHON" -m src.rollout --phase smoke --model qwen35_9b \
  --data-root "$SSVC_RUN_DIR/data/generated" --out "$SSVC_RUN_DIR/runs/P1"
run_check acceptance "$SSVC_PYTHON" - "$SSVC_RUN_DIR/runs/P1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
status = json.loads((root / "status.json").read_text())
audit = json.loads((root / "model_audit.json").read_text())
lock = json.loads((root / "validated_runtime_lock.json").read_text())
assert status["phase"] == "P1" and status["status"] == "PASS", status
assert audit["passed"] is True and audit["execution_kind"] == "REAL_CUDA_MODEL", audit
assert lock["runtime_status"] == "P1_PASSED", lock
assert lock["execution_kind"] == "REAL_CUDA_MODEL", lock
print("Real CUDA P1 acceptance: PASS. Later research phases remain NOT_RUN.")
PY
# EXIT builds the report and checksum; only then can the entire batch be PASS.
