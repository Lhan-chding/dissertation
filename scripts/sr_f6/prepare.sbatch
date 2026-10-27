#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-Prepare
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SRF6_CODE:?immutable code root required}"
: "${SRF6_ROOT:?registered run root required}"
: "${SRF6_REVISION:?full revision required}"
cd "$SRF6_CODE"
export PYTHONPATH="$SRF6_CODE/src"
python=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
"$python" -u -m sr_f6.prepare --root "$SRF6_ROOT" --config configs/sr_f6.json --revision "$SRF6_REVISION"
"$python" -u -m sr_f6.runtime processor-preflight --root "$SRF6_ROOT"
"$python" -u -m sr_f6.release native-regression --root "$SRF6_ROOT"
