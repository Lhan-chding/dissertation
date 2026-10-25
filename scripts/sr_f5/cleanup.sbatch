#!/bin/bash
#SBATCH --job-name=LH-SRF5-CLEANUP
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SRF5_CLEANUP_CODE:?immutable cleanup source directory}"
export PYTHONDONTWRITEBYTECODE=1
ROOT=/projects/_ssd/varunssd/louis-ssvc
AUDIT="$ROOT/storage_audits/cleanup_20261025_project_only"
cd "$SRF5_CLEANUP_CODE"
exec "$ROOT/envs/ssvc-py312/bin/python3.12" -u scripts/sr_f5/cleanup_project.py \
  --audit-dir "$AUDIT" \
  --allowlist "$AUDIT/retained/srf3_cleanup_allowlist.json" \
  --archive "$AUDIT/retained/SR_F3_CE_COMPLETE_UNDER_400MB.zip" \
  --apply --remove-old-model-cache
