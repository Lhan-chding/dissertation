#!/bin/bash
#SBATCH --job-name=LH-SRF5-G3-DRAIN
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=09:00:00
set -euo pipefail
exec srun --jobid=226282 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12 /projects/_ssd/varunssd/louis-ssvc/sr_f5_operations_139709d/scripts/sr_f5/drain_pair.py --root /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025 --code /projects/_ssd/varunssd/louis-ssvc/sr_f5_development_ea50c51 --revision ea50c51f0f145a183df360ab82025216d6303aca --controller-revision 139709df6fe381b93a3d43cc92d1fdfb7a5f205c --job 226282 --python /projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
