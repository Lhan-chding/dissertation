#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-T03
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=66G
#SBATCH --time=23:00:00
set -euo pipefail
: "${SRF6_T03_LAUNCHER:?explicit separately bound operations launcher required}"
operations='/projects/_ssd/varunssd/louis-ssvc/sr_f6_operations_20261027'
[[ "$SRF6_T03_LAUNCHER" == "$operations/"* && -f "$SRF6_T03_LAUNCHER" && ! -L "$SRF6_T03_LAUNCHER" ]] || exit 2
parent=$(dirname -- "$SRF6_T03_LAUNCHER")
[[ "$(cd -P -- "$parent" && pwd -P)" == "$parent" ]] || exit 2
exec /bin/bash "$SRF6_T03_LAUNCHER"
