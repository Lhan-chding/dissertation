#!/usr/bin/env bash
#SBATCH --no-requeue
set -euo pipefail
: "${SRF6_SMOKE_REVISION:?}" "${SRF6_SMOKE_DEPLOYMENT:?}" "${SRF6_SMOKE_BINDING:?}"
: "${SRF6_SMOKE_CONTRACT:?}" "${SRF6_SMOKE_CONTRACT_SHA:?}"
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -I -B \
  "$SRF6_SMOKE_DEPLOYMENT/scripts/sr_f6_memory_smoke_reuse_ops/operations.py" launch \
  --revision "$SRF6_SMOKE_REVISION" --deployment "$SRF6_SMOKE_DEPLOYMENT" \
  --binding-sha256 "$SRF6_SMOKE_BINDING" --contract "$SRF6_SMOKE_CONTRACT" \
  --contract-sha256 "$SRF6_SMOKE_CONTRACT_SHA"
