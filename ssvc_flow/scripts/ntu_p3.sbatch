#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --job-name=louis-ssvc-p3
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/varunssd/louis-ssvc/dissertation-ssvc/ssvc_flow
#SBATCH --output=/projects/varunssd/louis-ssvc/logs/p3-%j.out
#SBATCH --error=/projects/varunssd/louis-ssvc/logs/p3-%j.err
#SBATCH --open-mode=append

# One frozen model/track only. PASS does not complete the cross-model matrix.
# Create the logs directory before submission; Slurm opens stdout before this runs.
set -euo pipefail
umask 077
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  echo 'Submit with sbatch on an allocated GPU, not on a login node.' >&2
  exit 2
fi
export SSVC_WORK="${SSVC_WORK:-/projects/varunssd/louis-ssvc}"
export SSVC_PYTHON="${SSVC_PYTHON:-$SSVC_WORK/envs/ssvc-py312/bin/python}"
export SSVC_RESUME="${SSVC_RESUME:-0}"
ssvc_root="$SSVC_WORK/dissertation-ssvc/ssvc_flow"
export SSVC_CONTACT_MANIFEST="${SSVC_CONTACT_MANIFEST:-$ssvc_root/docs/local_evidence/P0/contact_sheets_manifest.json}"
if [[ ! -x $SSVC_PYTHON ]]; then
  echo 'SSVC_PYTHON must name an installed executable personal Python.' >&2
  exit 2
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
# Resolve paths before mkdir or any model import, including directory aliases.
"$SSVC_PYTHON" - <<'PY'
import os
from pathlib import Path


def path(name, kind, within=None):
    value = os.environ.get(name, '')
    if not value or not Path(value).is_absolute():
        raise ValueError(f'{name} must be an explicit absolute path')
    target = Path(value).resolve()
    if within is not None and (target == within or not target.is_relative_to(within)):
        raise ValueError(f'{name} must stay inside {within}')
    if kind == 'file' and not target.is_file():
        raise ValueError(f'{name} must exist as a file')
    if kind == 'dir' and not target.is_dir():
        raise ValueError(f'{name} must exist as a directory')
    return target

work = path('SSVC_WORK', 'dir')
for relative in ('runs', 'logs', 'tmp', 'cache', 'cache/huggingface/hub',
                 'cache/huggingface/xet', 'cache/cuda', 'cache/triton',
                 'cache/torch', 'cache/torchinductor', 'cache/matplotlib'):
    target = (work / relative).resolve()
    if target == work or not target.is_relative_to(work):
        raise ValueError(f'output/cache directory escapes SSVC_WORK: {relative}')
repo = work / 'dissertation-ssvc/ssvc_flow'
if not repo.is_dir() or not repo.resolve().is_relative_to(work):
    raise ValueError('SSVC checkout must exist inside SSVC_WORK')
path('SSVC_CONFIG', 'file', work)
path('SSVC_P1_RUN', 'dir', work)
if os.environ.get('SSVC_MODEL') not in {'qwen25vl_3b', 'qwen35_9b', 'qwen25vl_7b'}:
    raise ValueError('SSVC_MODEL must be a registered P3 model')
if os.environ['SSVC_RESUME'] not in {'0', '1'}:
    raise ValueError('SSVC_RESUME must be 0 or 1')
runs = (work / 'runs').resolve()
if not runs.is_relative_to(work):
    raise ValueError('runs directory escapes SSVC_WORK')
out = path('SSVC_FROZEN_OUT', 'new', runs)
if os.environ['SSVC_RESUME'] == '0' and out.exists():
    raise ValueError('existing output requires explicit SSVC_RESUME=1')
if os.environ['SSVC_RESUME'] == '1' and not out.is_dir():
    raise ValueError('resume output must already exist')
track = os.environ.get('SSVC_TRACK')
if track == 'N':
    path('SSVC_DATA_ROOT', 'dir', work)
    path('SSVC_HUMAN_REVIEW', 'file', work)
    path('SSVC_CONTACT_MANIFEST', 'file', work)
elif track == 'L':
    path('SSVC_LEGACY_DATA', 'file', work)
    path('SSVC_LEGACY_MANIFEST', 'file', work)
else:
    raise ValueError('SSVC_TRACK must be L or N')
PY
export TMPDIR="$SSVC_WORK/tmp" TMP="$SSVC_WORK/tmp" TEMP="$SSVC_WORK/tmp"
export XDG_CACHE_HOME="$SSVC_WORK/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
"$SSVC_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('P3 requires exactly one visible CUDA GPU in the Slurm allocation')
print('Frozen GPU:', torch.cuda.get_device_name(0))
PY
mkdir -p "$SSVC_WORK/runs" "$SSVC_WORK/logs" "$TMPDIR" \
  "$HF_HUB_CACHE" "$HF_XET_CACHE" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" \
  "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"
cd "$ssvc_root"
ssvc_args=(-m src.rollout --phase frozen --model "$SSVC_MODEL" --track "$SSVC_TRACK"
  --config "$SSVC_CONFIG" --p1-run-dir "$SSVC_P1_RUN" --out "$SSVC_FROZEN_OUT")
if [[ $SSVC_TRACK == N ]]; then
  ssvc_args+=(--split dev --data-root "$SSVC_DATA_ROOT" --human-review "$SSVC_HUMAN_REVIEW"
    --contact-manifest "$SSVC_CONTACT_MANIFEST")
else
  ssvc_args+=(--legacy-data "$SSVC_LEGACY_DATA" --legacy-manifest "$SSVC_LEGACY_MANIFEST")
fi
if [[ $SSVC_RESUME == 1 ]]; then ssvc_args+=(--resume); fi
echo "P3 single task: model=$SSVC_MODEL track=$SSVC_TRACK out=$SSVC_FROZEN_OUT"
# CLI owns evidence/resume validation. No deletion, training, installation or upload.
exec "$SSVC_PYTHON" "${ssvc_args[@]}"
