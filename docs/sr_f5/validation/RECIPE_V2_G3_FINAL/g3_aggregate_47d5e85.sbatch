#!/bin/bash
#SBATCH --job-name=LH-SRF5-G3-AGG
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/_ssd/varunssd/louis-ssvc/sr_f5_recipe_v2_47d5e85/src:/projects/_ssd/varunssd/louis-ssvc/sr_f5_recipe_v2_47d5e85
cd /projects/_ssd/varunssd/louis-ssvc/sr_f5_recipe_v2_47d5e85
exec 9>/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026/CONTINUATION/G3_AGGREGATION_RUN.lock
flock -n 9
test ! -e /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026/G3/METRICS.json
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -u -m sr_f5.pilot --root /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026 --aggregate
