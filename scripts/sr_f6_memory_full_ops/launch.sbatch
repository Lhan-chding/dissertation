#!/usr/bin/env bash
#SBATCH --no-requeue
set -euo pipefail
: "${SRF6_FULL_REVISION:?}" "${SRF6_FULL_DEPLOYMENT:?}" "${SRF6_FULL_BINDING:?}"
: "${SRF6_FULL_CONTRACT:?}" "${SRF6_FULL_CONTRACT_SHA:?}"
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -I -B \
  "$SRF6_FULL_DEPLOYMENT/scripts/sr_f6_memory_full_ops/operations.py" launch \
  --revision "$SRF6_FULL_REVISION" --deployment "$SRF6_FULL_DEPLOYMENT" \
  --binding-sha256 "$SRF6_FULL_BINDING" --contract "$SRF6_FULL_CONTRACT" \
  --contract-sha256 "$SRF6_FULL_CONTRACT_SHA"
