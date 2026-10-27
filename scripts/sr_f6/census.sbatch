#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-Census
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:00:00
set -euo pipefail
: "${SRF6_CODE:?immutable deployed code root required}"
: "${SRF6_ROOT:?registered SR-F6 run root required}"
: "${SRF6_PARENT:?registered parent required}"
cd "$SRF6_CODE"
export PYTHONPATH="$SRF6_CODE/src"
export PYTHONDONTWRITEBYTECODE=1
python=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
# Two independent one-GPU shards. No model parallelism or changed sample order.
srun --exclusive --ntasks=1 --gres=gpu:pro6000:1 --cpus-per-task=4 --mem=96G \
  "$python" -u -m sr_f6.census run --root "$SRF6_ROOT" --parent "$SRF6_PARENT" --shard 0 --shards 2 &
pid0=$!
srun --exclusive --ntasks=1 --gres=gpu:pro6000:1 --cpus-per-task=4 --mem=96G \
  "$python" -u -m sr_f6.census run --root "$SRF6_ROOT" --parent "$SRF6_PARENT" --shard 1 --shards 2 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
exit "$status"
