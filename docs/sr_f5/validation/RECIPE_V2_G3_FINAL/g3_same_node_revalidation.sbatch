#!/bin/bash
#SBATCH --job-name=LH-SRF5-G3-AGG-CHECK
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/_ssd/varunssd/louis-ssvc/sr_f5_recipe_v2_47d5e85/src:/projects/_ssd/varunssd/louis-ssvc/sr_f5_recipe_v2_47d5e85
cd /projects/_ssd/varunssd/louis-ssvc/sr_f5_recipe_v2_47d5e85
exec /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 -u /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026/CONTINUATION/g3_same_node_revalidation.py
