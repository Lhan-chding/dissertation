#!/bin/bash
#SBATCH --job-name=LH-SRF4-PREP
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
SRF4_ROOT=/projects/_ssd/varunssd/louis-ssvc/sr_f4_consolidated_20261022
SRF4_OLD=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2
SRF4_PYTHON=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONPATH="$SRF4_ROOT/code/src:$SRF4_ROOT/code"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$SRF4_ROOT/code"
"$SRF4_PYTHON" -c 'import sys; from sr_f4.deployment import check_imports; print(check_imports(".", sys.executable))'
"$SRF4_PYTHON" -u -m sr_f4.prepare --root "$SRF4_ROOT" --old-root "$SRF4_OLD" --code "$SRF4_ROOT/code"
"$SRF4_PYTHON" -u -m sr_f4.pool init --root "$SRF4_ROOT"
"$SRF4_PYTHON" -u -m sr_f4.pool submit --root "$SRF4_ROOT" --code "$SRF4_ROOT/code" --python "$SRF4_PYTHON"
