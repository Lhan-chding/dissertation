#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --job-name=louis-ssvc-r1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/varunssd/louis-ssvc/dissertation-ssvc/ssvc_flow
#SBATCH --output=/projects/varunssd/louis-ssvc/logs/r1-%j.out
#SBATCH --error=/projects/varunssd/louis-ssvc/logs/r1-%j.err
#SBATCH --open-mode=append

set -euo pipefail
umask 077
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ || ! ${SLURM_RESTART_COUNT:-0} =~ ^[0-9]+$ ]]; then
  echo 'Submit with sbatch; a numeric Slurm job ID and restart count are required.' >&2
  exit 2
fi
export SSVC_WORK="${SSVC_WORK:-/projects/varunssd/louis-ssvc}"
export SSVC_PYTHON="${SSVC_PYTHON:-$SSVC_WORK/envs/ssvc-py312/bin/python}"
ssvc_root="$SSVC_WORK/dissertation-ssvc/ssvc_flow"
if [[ $SSVC_WORK != /* || ! -x $SSVC_PYTHON || ! -d $ssvc_root ]]; then
  echo 'Expected an absolute personal work directory, installed Python and SSVC checkout.' >&2
  exit 2
fi
cd "$ssvc_root"
export XDG_CACHE_HOME="$SSVC_WORK/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export PIP_CONFIG_FILE=/dev/null PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTEST_ADDOPTS TRANSFORMERS_CACHE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"

run_root="$SSVC_WORK/runs/NEXT_20260909/R1_server_${SLURM_JOB_ID}_attempt_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$(dirname "$run_root")"
if ! mkdir "$run_root"; then
  echo "Refusing to reuse existing attempt: $run_root" >&2
  exit 2
fi
export TMPDIR="$run_root/tmp" TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR" "$run_root/checks"
printf 'state=RUNNING\njob_id=%s\nstarted_utc=%s\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$run_root/result.txt"
cp "$0" "$run_root/submitted.sbatch"
git rev-parse HEAD > "$run_root/source-commit.txt"
hostname > "$run_root/hostname.txt"

rc=0
"$SSVC_PYTHON" -m src.audit_r1_runtime \
  --config configs/next_stage.yaml \
  --data-root "$SSVC_WORK/reviewed-data-20260908/generated" \
  --out "$run_root" \
  --smoke-out "$run_root/smoke" \
  > "$run_root/checks/r1-runtime.log" 2>&1 || rc=$?

tar -czf "$run_root.tar.gz.partial" -C "$(dirname "$run_root")" "$(basename "$run_root")" \
  > "$run_root/packaging.log" 2>&1 || rc=$?
if [[ -f "$run_root.tar.gz.partial" ]]; then
  mv "$run_root.tar.gz.partial" "$run_root.tar.gz"
  "$SSVC_PYTHON" - "$run_root.tar.gz" <<'PY' || rc=$?
import hashlib, sys
from pathlib import Path
p = Path(sys.argv[1])
p.with_name(p.name + '.sha256').write_text(f"{hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()}  {p.name}\n")
PY
fi
if (( rc == 0 )); then
  printf 'state=PASS\nexit_code=0\njob_id=%s\nfinished_utc=%s\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$run_root/result.txt"
else
  printf 'state=FAILED\nexit_code=%s\njob_id=%s\nfinished_utc=%s\n' "$rc" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$run_root/result.txt"
fi
exit "$rc"
