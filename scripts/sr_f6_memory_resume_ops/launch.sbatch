#!/usr/bin/env bash
#SBATCH --no-requeue
set -euo pipefail
: "${SRF6_RESUME_REVISION:?}" "${SRF6_RESUME_DEPLOYMENT:?}" "${SRF6_RESUME_BINDING:?}"
: "${SRF6_RESUME_CONTRACT:?}" "${SRF6_RESUME_CONTRACT_SHA:?}"
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -I -B \
  "$SRF6_RESUME_DEPLOYMENT/scripts/sr_f6_memory_resume_ops/operations.py" launch \
  --revision "$SRF6_RESUME_REVISION" --deployment "$SRF6_RESUME_DEPLOYMENT" \
  --binding-sha256 "$SRF6_RESUME_BINDING" --contract "$SRF6_RESUME_CONTRACT" \
  --contract-sha256 "$SRF6_RESUME_CONTRACT_SHA"
