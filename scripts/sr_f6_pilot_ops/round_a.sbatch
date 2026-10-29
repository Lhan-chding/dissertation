#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-PilotV3-A
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=66G
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --no-requeue
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
: "${SRF6_OPS_REVISION:?actual operation revision required}"
: "${SRF6_OPS_DEPLOYMENT:?actual operation deployment required}"
: "${SRF6_OPS_BINDING_SHA256:?actual operation binding required}"
[[ "$SRF6_OPS_REVISION" =~ ^[0-9a-f]{40}$ && "$SRF6_OPS_BINDING_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$SRF6_OPS_DEPLOYMENT" == "$project/sr_f6_pilot_ops_code_${SRF6_OPS_REVISION:0:7}" ]] || exit 2
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ && "${SLURM_JOB_NAME:?}" == LH-SRF6-PilotV3-A ]] || exit 2
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || exit 2
[[ ! ${SRF6_RESUME_TASK_0+x} && ! ${SRF6_RESUME_TASK_1+x} ]] || exit 2
unset BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONSTARTUP PYTHONINSPECT PYTHONPATH
exec "$project/envs/ssvc-py312/bin/python3.12" -I -B \
  "$SRF6_OPS_DEPLOYMENT/scripts/sr_f6_pilot_ops/round_a.py" launch \
  --revision "$SRF6_OPS_REVISION" --deployment "$SRF6_OPS_DEPLOYMENT" \
  --binding-sha256 "$SRF6_OPS_BINDING_SHA256"
