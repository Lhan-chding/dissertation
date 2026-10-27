#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-Preflight
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SRF6_CODE:?immutable deployed code root required}"
: "${SRF6_ROOT:?registered SR-F6 run root required}"
cd "$SRF6_CODE"
export PYTHONPATH="$SRF6_CODE/src"
export PYTHONDONTWRITEBYTECODE=1
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -u \
  -m sr_f6.runtime processor-preflight --root "$SRF6_ROOT"
