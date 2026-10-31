#!/usr/bin/env bash
#SBATCH --no-requeue
set -euo pipefail
: "${SRF6_B_REVISION:?}" "${SRF6_B_DEPLOYMENT:?}" "${SRF6_B_BINDING:?}"
: "${SRF6_B_CONTRACT:?}" "${SRF6_B_CONTRACT_SHA:?}"
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -I -B \
  "$SRF6_B_DEPLOYMENT/scripts/sr_f6_round_b_ops/operations.py" launch \
  --revision "$SRF6_B_REVISION" --deployment "$SRF6_B_DEPLOYMENT" \
  --binding-sha256 "$SRF6_B_BINDING" --contract "$SRF6_B_CONTRACT" \
  --contract-sha256 "$SRF6_B_CONTRACT_SHA"
