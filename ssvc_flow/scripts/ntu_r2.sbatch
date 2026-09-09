#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --job-name=louis-ssvc-r2
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/varunssd/louis-ssvc/dissertation-ssvc/ssvc_flow
#SBATCH --output=/projects/varunssd/louis-ssvc/logs/r2-%j.out
#SBATCH --error=/projects/varunssd/louis-ssvc/logs/r2-%j.err
#SBATCH --open-mode=append

set -euo pipefail
umask 077
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ || ! ${SLURM_RESTART_COUNT:-0} =~ ^[0-9]+$ ]]; then
  echo 'Submit with sbatch; a numeric job ID and restart count are required.' >&2
  exit 2
fi
export SSVC_WORK="${SSVC_WORK:-/projects/varunssd/louis-ssvc}"
export SSVC_PYTHON="${SSVC_PYTHON:-$SSVC_WORK/envs/ssvc-py312/bin/python}"
ssvc_root="$SSVC_WORK/dissertation-ssvc/ssvc_flow"
if [[ $SSVC_WORK != /* || ! -x $SSVC_PYTHON || ! -d $ssvc_root ]]; then
  echo 'Expected an absolute personal project directory, installed Python and checkout.' >&2
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

run_root="$SSVC_WORK/runs/NEXT_20260909/R2_server_${SLURM_JOB_ID}_attempt_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$(dirname "$run_root")"
if ! mkdir "$run_root"; then
  echo "Refusing to reuse existing attempt wrapper: $run_root" >&2
  exit 2
fi
export TMPDIR="$run_root/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR" "$run_root/checks"
stage_root="$run_root/R2"
resume_args=()
if [[ -n ${SSVC_R2_RESUME_STAGE:-} ]]; then
  stage_root=$(realpath -e "$SSVC_R2_RESUME_STAGE")
  allowed_root=$(realpath -e "$SSVC_WORK/runs/NEXT_20260909")
  if [[ $stage_root != "$allowed_root"/R2_server_*/R2 || ! -f $stage_root/identity.json ]]; then
    echo 'Resume must name an existing R2 stage within this project.' >&2
    exit 2
  fi
  resume_args=(--resume)
  printf '%s\n' "$stage_root" > "$run_root/resumed_stage_path.txt"
fi
printf 'state=RUNNING\njob_id=%s\nstarted_utc=%s\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$run_root/result.txt"
cp "$0" "$run_root/submitted.sbatch"
git rev-parse HEAD > "$run_root/source-commit.txt"
hostname > "$run_root/hostname.txt"

finish() {
  local rc=$? archive_rc=0
  trap - EXIT
  set +e
  local archive_paths=(-C "$(dirname "$run_root")" "$(basename "$run_root")")
  if (( ${#resume_args[@]} )); then
    archive_paths+=(-C "$(dirname "$stage_root")" "$(basename "$stage_root")")
  fi
  tar --exclude="$(basename "$run_root")/tmp" \
    --exclude="$(basename "$run_root")/result.txt" \
    -czf "$run_root.tar.gz.partial" "${archive_paths[@]}" \
    > "$run_root.packaging.log" 2>&1
  archive_rc=$?
  if (( archive_rc == 0 )); then
    mv "$run_root.tar.gz.partial" "$run_root.tar.gz"
    archive_rc=$?
  fi
  if (( archive_rc == 0 )); then
    "$SSVC_PYTHON" - "$run_root.tar.gz" <<'PY'
import hashlib
import sys
from pathlib import Path
p = Path(sys.argv[1])
with p.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
p.with_name(p.name + '.sha256').write_text(f"{digest}  {p.name}\n")
PY
    archive_rc=$?
  fi
  if (( rc == 0 && archive_rc != 0 )); then rc=$archive_rc; fi
  local state=FAILED
  if (( rc == 0 )); then state=PASS; fi
  printf 'state=%s\nexit_code=%s\narchive_exit_code=%s\njob_id=%s\nfinished_utc=%s\n' \
    "$state" "$rc" "$archive_rc" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$run_root/result.txt"
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

"$SSVC_PYTHON" -m src.r2_runtime \
  --config configs/next_stage.yaml \
  --data-root "$SSVC_WORK/reviewed-data-20260908/generated" \
  --r0-dir "$SSVC_WORK/runs/NEXT_20260909/R0_R1_close_146806_attempt_0/R0" \
  --r1-run "$SSVC_WORK/runs/NEXT_20260909/R1_server_146571_attempt_0" \
  --supplement-dir "$SSVC_WORK/runs/NEXT_20260909/R0_R1_close_146806_attempt_0/R1_supplement" \
  --out "$stage_root" "${resume_args[@]}" \
  > "$run_root/checks/r2.log" 2>&1
