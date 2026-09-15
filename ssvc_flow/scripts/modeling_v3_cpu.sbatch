#!/bin/bash
#SBATCH --job-name=ssvc-v3-cpu
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --no-requeue

set -euo pipefail
: "${SSVC_V3_CHECKOUT:?Set an immutable V3 checkout}"
: "${SSVC_V3_PYTHON:?Set the verified production Python}"
: "${SSVC_V3_OUTPUT:?Set a new V3 output directory}"
: "${SSVC_V3_STAGE:?Set pilot, test, audit, historical, collect, observe, coverage, or request}"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export SSVC_WORK=/projects/varunssd/louis-ssvc
export XDG_CACHE_HOME="$SSVC_WORK/cache" HF_HOME="$SSVC_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$SSVC_WORK/modeling_v3_20260915/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$SSVC_V3_CHECKOUT/ssvc_flow"
CFG=configs/modeling_v3/protocol.json
case "$SSVC_V3_STAGE" in
  request)
    : "${SSVC_V3_REQUEST:?Set a frozen CPU request JSON}"
    : "${SSVC_V3_REQUEST_SHA256:?Set its verified SHA256}"
    /usr/bin/time -v "$SSVC_V3_PYTHON" scripts/run_modeling_v3_cpu_request.py --request "$SSVC_V3_REQUEST" --sha256 "$SSVC_V3_REQUEST_SHA256"
    ;;
  pilot)
    /usr/bin/time -v "$SSVC_V3_PYTHON" -m src.modeling_v3.cli collect-toy --config "$CFG" --out "$SSVC_V3_OUTPUT" --pilot
    ;;
  test)
    /usr/bin/time -v "$SSVC_V3_PYTHON" scripts/verify_modeling_v3_cpu.py --out "$SSVC_V3_OUTPUT" -- tests docs/modeling_v3/design/reference/test_math_contracts.py --deselect=tests/test_audit_r0_remaining.py::test_cross_split_passes_generated_dataset
    ;;
  audit)
    : "${SSVC_V3_PARENT:?Set the read-only N4 run root}"
    extra=()
    if [ -n "${SSVC_V3_LEGACY_PARENT:-}" ]; then extra+=(--legacy-parent "$SSVC_V3_LEGACY_PARENT"); fi
    /usr/bin/time -v "$SSVC_V3_PYTHON" -m src.modeling_v3.cli audit-parent --config "$CFG" --parent "$SSVC_V3_PARENT" --out "$SSVC_V3_OUTPUT" "${extra[@]}"
    ;;
  historical)
    : "${SSVC_V3_PARENT:?Set the read-only historical raw collection}"
    /usr/bin/time -v "$SSVC_V3_PYTHON" -m src.modeling_v3.cli prepare-historical --config "$CFG" --parent "$SSVC_V3_PARENT" --out "$SSVC_V3_OUTPUT"
    ;;
  collect)
    /usr/bin/time -v "$SSVC_V3_PYTHON" -m src.modeling_v3.cli collect-toy --config "$CFG" --role development --out "$SSVC_V3_OUTPUT"
    ;;
  observe|coverage)
    : "${SSVC_V3_COLLECTION:?Set the immutable V3 collection}"
    command=observe-toy
    if [ "$SSVC_V3_STAGE" = coverage ]; then command=coverage-study; fi
    /usr/bin/time -v "$SSVC_V3_PYTHON" -m src.modeling_v3.cli "$command" --config "$CFG" --collection "$SSVC_V3_COLLECTION" --out "$SSVC_V3_OUTPUT"
    ;;
  *) echo 'Unknown V3 CPU stage' >&2; exit 2 ;;
esac
