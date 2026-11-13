#!/usr/bin/env bash
set -euo pipefail
# Exactly one audited release receipt and its independently recorded SHA.
[[ $# == 2 ]] || exit 2
receipt=$1
receipt_sha=$2
[[ "$receipt" == /* ]] || exit 2
base=/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
unset PYTORCH_CUDA_ALLOC_CONF PYTHONPATH PYTHONHOME
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_NTASKS:-} == 1 && ${SLURM_NNODES:-} == 1 ]] || exit 2
actual_freeze=$(sha256sum "$base/DESIGN_FREEZE.json")
[[ ${actual_freeze%% *} == 6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0 ]] || exit 2
# Validate ALL release files with only stdlib before importing release code.
# The receipt lives outside release_root, avoiding a self-referential digest.
release_root=$("$python_bin" -I -B - "$receipt" "$receipt_sha" "${BASH_SOURCE[0]}" <<'PY'
import hashlib, json, pathlib, re, sys
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok, message):
    if not ok:
        raise SystemExit(message)
def pairs(items):
    obj = dict(items)
    require(len(obj) == len(items), 'Duplicate receipt key')
    return obj
path = pathlib.Path(sys.argv[1]).resolve(strict=True)
require(re.fullmatch('[0-9a-f]{64}', sys.argv[2]) and sha(path) == sys.argv[2], 'Release receipt SHA mismatch')
record = json.loads(path.read_text(), object_pairs_hook=pairs)
require(record.get('schema') == 'sr-satdose-prefix-technical-release-v1' and
        record.get('status') == 'AUDITED_FOR_PREFIX_CALIB_ONLY' and
        record.get('permission') == 'NO_TRAINING', 'Unaudited PREFIX release')
root = pathlib.Path(record['release_root']).resolve(strict=True)
require(not path.is_relative_to(root), 'Receipt must live outside release')
actual = {}
for file in root.rglob('*'):
    if '__pycache__' in file.relative_to(root).parts:
        continue
    require(not file.is_symlink(), 'Release symlink forbidden')
    if file.is_file():
        actual[str(file.relative_to(root))] = sha(file)
require(actual == record['sources'], 'Release inventory/SHA mismatch')
launcher = pathlib.Path(sys.argv[3]).resolve(strict=True)
require(sha(launcher) == actual.get('scheduling/run_prefix.sh'), 'Executing launcher SHA mismatch')
require('execution/prefix_control.py' in actual, 'Missing PREFIX controller')
print(root)
PY
)
cd "$release_root"
exec "$python_bin" -B -m execution.prefix_control --release-receipt "$receipt" --release-sha256 "$receipt_sha" --execute
