#!/usr/bin/env bash
#SBATCH --job-name=LH-SRF6-PostB-DeliveryV2
#SBATCH --account=rose
#SBATCH --partition=cluster02
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
project=/projects/_ssd/varunssd/louis-ssvc
contract="${SRF6_DELIVERY_CONTRACT:?immutable audit contract required}"
contract_sha="${SRF6_DELIVERY_CONTRACT_SHA256:?contract hash required}"
driver="${SRF6_DELIVERY_DRIVER:?immutable audit driver required}"
[[ "$contract_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_JOB_NAME:?}" == LH-SRF6-PostB-DeliveryV2 ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:?}" == 4 && "${SLURM_MEM_PER_NODE:?}" == 12288 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || exit 2
ops="$(dirname -- "$contract")"
for file in "$contract" "$driver"; do
  [[ "$file" == "$project/"* && -f "$file" && ! -L "$file" ]] || exit 2
  directory="$(dirname -- "$file")"
  [[ "$(cd -P -- "$directory" && pwd -P)/$(basename -- "$file")" == "$file" ]] || exit 2
done
[[ "$ops" == "$project/sr_f6_operations_20261027/PILOT_POST_B_DELIVERY_V2_20261101" ]] || exit 2
"$project/envs/ssvc-py312/bin/python3.12" -I -B - "$contract" "$contract_sha" "$driver" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(contract).hexdigest() != sys.argv[2]:
    raise SystemExit("Audit contract hash differs")
if hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest() != json.loads(contract)["driver_sha256"]:
    raise SystemExit("Audit driver hash differs")
PY
cache="$ops/cpu_cache_$SLURM_JOB_ID"
[[ ! -e "$cache" && ! -L "$cache" ]] || exit 2
mkdir -- "$cache" "$cache/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
export XDG_CACHE_HOME="$cache/xdg" HF_HOME="$cache/huggingface"
export HF_HUB_CACHE="$cache/huggingface/hub" HUGGINGFACE_HUB_CACHE="$cache/huggingface/hub"
export TRANSFORMERS_CACHE="$cache/huggingface/hub" HF_MODULES_CACHE="$cache/huggingface/modules"
export HF_DATASETS_CACHE="$cache/huggingface/datasets" HF_ASSETS_CACHE="$cache/huggingface/assets"
export TORCH_HOME="$cache/torch" TORCH_EXTENSIONS_DIR="$cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton" CUDA_CACHE_PATH="$cache/cuda"
unset BASH_ENV ENV PYTHONPATH
cd "$ops"
exec "$project/envs/ssvc-py312/bin/python3.12" -I -B -u "$driver" \
  --contract "$contract" --contract-sha256 "$contract_sha"
