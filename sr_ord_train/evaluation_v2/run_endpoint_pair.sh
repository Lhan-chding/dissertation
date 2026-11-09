#!/usr/bin/env bash
# Explicit v2 typed-GPU launcher; registered evaluation implementation unchanged.
set -euo pipefail
root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
[[ $# == 3 ]] || { echo 'Usage: run_endpoint_pair.sh ABSOLUTE_BINDING ENDPOINT_A ENDPOINT_B' >&2; exit 2; }
binding=$1
endpoint_a=$2
endpoint_b=$3
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
cd "$root"
[[ ${SLURM_MEM_PER_NODE:-} == 67584 && ${SLURM_CPUS_ON_NODE:-} == 8 ]] || exit 2
[[ ${CUDA_VISIBLE_DEVICES:-} =~ ^[^,[:space:]]+,[^,[:space:]]+$ ]] || exit 2
[[ ${CUDA_VISIBLE_DEVICES%%,*} != ${CUDA_VISIBLE_DEVICES##*,} ]] || exit 2
[[ -z ${SLURM_GPUS_ON_NODE:-} || ${SLURM_GPUS_ON_NODE} == 2 ]] || exit 2
worker_mem_mb=$((SLURM_MEM_PER_NODE / 2))
worker_cpus=$((SLURM_CPUS_ON_NODE / 2))
# Read only the registration and endpoint lock. Each unchanged worker performs
# complete binding/source/request verification before its model initialization.
"$python_bin" -B - "$root" "$binding" "$endpoint_a" "$endpoint_b" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys
root, binding_path = map(Path, sys.argv[1:3])
endpoints = sys.argv[3:]
def need(ok, reason):
    if not ok:
        raise PermissionError(reason)
def plain(path):
    need(path.is_absolute() and path.is_file() and path.resolve(strict=True) == path
         and not path.is_symlink(), 'Canonical existing registration file required')
    return path
need(root.resolve(strict=True) == root, 'Canonical package required')
need(len(endpoints) == 2 and endpoints[0] != endpoints[1], 'Two distinct endpoints required')
pattern = r'b[01]-(?:PARENT|1103[12]-[RGD]-step-(?:012|024))'
need(all(re.fullmatch(pattern, x) for x in endpoints), 'Invalid endpoint identity')
need(binding_path.is_relative_to(root), 'Binding outside package')
binding = json.loads(plain(binding_path).read_bytes())
need(binding['schema'] == 'sr-ord-eval-binding-v1' and binding['experiment_id'] == 'SR-ORD-TRAIN-1'
     and binding['package_root'] == str(root), 'Evaluation binding identity differs')
need(type(binding['worker_shards']) is int and binding['worker_shards'] == 1
     and binding['endpoint_count'] == 26 and binding['registered_responses'] == 26624
     and binding['status'] == 'REGISTERED_NOT_GENERATED'
     and binding['optimizer_created'] is False and binding['parameter_updates'] == 0,
     'Registration scope differs')
reference = binding['endpoint_lock']
lock_path = plain(Path(reference['path']))
need(lock_path.is_relative_to(root), 'Endpoint lock outside package')
raw = lock_path.read_bytes()
need(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Endpoint lock bytes differ')
lock = json.loads(raw)
need(lock['verified_paths'] == 12 and lock['optimizer_updates'] == 288
     and lock['training_responses'] == 36864 and len(lock['endpoints']) == 26,
     'Incomplete training acceptance')
for endpoint in endpoints:
    need(endpoint in lock['endpoints'] and lock['endpoints'][endpoint]['endpoint_id'] == endpoint,
         'Endpoint not registered')
    stage = root / 'run/evaluation' / endpoint
    need(not stage.exists() and not stage.is_symlink(), 'Endpoint already has output; preserve evidence')
    for ancestor in (root / 'run', root / 'run/evaluation'):
        need(not ancestor.is_symlink() and (not ancestor.exists() or ancestor.resolve() == ancestor),
             'Unsafe output ancestor')
PY
# Slurm requires the step GPU type to agree with the typed job allocation.
# Clear only resource/binding option inputs, never allocation identity or CUDA
# visibility. Explicit srun options below select each exclusive worker's share.
# See https://slurm.schedmd.com/srun.html (INPUT ENVIRONMENT VARIABLES).
step=(env
  -u SLURM_GPUS -u SLURM_GPUS_PER_NODE -u SLURM_GPUS_PER_TASK
  -u SLURM_GRES -u SLURM_STEP_GRES -u SLURM_GRES_FLAGS
  -u SLURM_TRES_PER_TASK -u SLURM_GPU_BIND -u SLURM_TRES_BIND
  -u SLURM_CPUS_PER_GPU -u SLURM_MEM_PER_GPU -u SLURM_MEM_PER_CPU
  -u SLURM_NTASKS_PER_GPU
  srun --exclusive --exact --nodes=1 --ntasks=1
  --cpus-per-task="$worker_cpus" --gpus-per-task=pro6000:1 --mem="${worker_mem_mb}M")
"${step[@]}" "$python_bin" -B -m adapter.evaluation worker --binding "$binding" --endpoint "$endpoint_a" --shard-index 0 &
p0=$!
"${step[@]}" "$python_bin" -B -m adapter.evaluation worker --binding "$binding" --endpoint "$endpoint_b" --shard-index 0 &
p1=$!
# A failed worker must not kill its healthy peer. Preserve both outputs and wait.
status=0
wait "$p0" || status=1
wait "$p1" || status=1
exit "$status"
