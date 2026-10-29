#!/usr/bin/env bash
#SBATCH --no-requeue
set -euo pipefail
: "${SRF6_MEMORY_ACTION:?}" "${SRF6_MEMORY_REVISION:?}" "${SRF6_MEMORY_DEPLOYMENT:?}"
: "${SRF6_MEMORY_BINDING:?}" "${SRF6_MEMORY_CONTRACT:?}" "${SRF6_MEMORY_CONTRACT_SHA:?}"
args=()
if [[ -n "${SRF6_MEMORY_VALIDATION:-}" ]]; then
  : "${SRF6_MEMORY_VALIDATION_SHA:?}"
  args+=(--validation-path "$SRF6_MEMORY_VALIDATION" --validation-sha256 "$SRF6_MEMORY_VALIDATION_SHA")
fi
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -I -B \
  "$SRF6_MEMORY_DEPLOYMENT/scripts/sr_f6_memory_repair_ops/operations.py" launch \
  --action "$SRF6_MEMORY_ACTION" --revision "$SRF6_MEMORY_REVISION" \
  --deployment "$SRF6_MEMORY_DEPLOYMENT" --binding-sha256 "$SRF6_MEMORY_BINDING" \
  --contract "$SRF6_MEMORY_CONTRACT" --contract-sha256 "$SRF6_MEMORY_CONTRACT_SHA" "${args[@]}"
