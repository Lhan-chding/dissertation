#!/usr/bin/env bash
#SBATCH --job-name=LH-SATDOSE-FIRST-TWO
#SBATCH --account=rose
#SBATCH --qos=soujanya-poria-startfund-2026-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=pro6000:2
#SBATCH --mem=66G
#SBATCH --no-requeue
set -euo pipefail
[[ $# == 2 && $1 == /* && $2 =~ ^[0-9a-f]{64}$ ]] || exit 2
receipt=$1
receipt_sha=$2
python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
launcher=${BASH_SOURCE[0]}
[[ "$launcher" == /* ]] || launcher="$PWD/$launcher"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
unset PYTORCH_CUDA_ALLOC_CONF PYTHONPATH PYTHONHOME
[[ ${SLURM_RESTART_COUNT:-0} == 0 ]] || exit 2
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_NTASKS:-} == 1 && ${SLURM_NNODES:-} == 1 ]] || exit 2
release=$("$python_bin" -I -B - "$receipt" "$receipt_sha" "$launcher" <<'PY'
import hashlib,json,pathlib,sys
base=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
def sha(p):
    assert p.is_file() and p.resolve(strict=True)==p and not p.is_symlink()
    return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=pathlib.Path(sys.argv[1]);assert sha(receipt)==sys.argv[2]
r=json.loads(receipt.read_text());root=pathlib.Path(r['release_root'])
assert r['schema']=='sr-satdose-first-two-release-v1' and r['status']=='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY'
assert r['permission']=='FOUR_PATHS_MAX_STEP_2_NO_STEP3' and r['evidence_root']==str(base)
assert root.is_relative_to(base) and receipt.is_relative_to(base) and not receipt.is_relative_to(root)
assert root.resolve(strict=True)==root and not root.is_symlink()
actual={}
for p in root.rglob('*'):
    assert not p.is_symlink() and p.resolve(strict=True)==p
    if p.is_file():actual[str(p.relative_to(root))]=sha(p)
assert actual==r['sources']
assert sha(pathlib.Path(sys.argv[3]))==actual['scheduling/run_first_two.sh']
assert sha(base/'DESIGN_FREEZE.json')==r['design_freeze_sha256']=='6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
print(root)
PY
)
cd "$release"
env -u SLURM_TRES_PER_TASK srun --exclusive --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --gpus-per-task=pro6000:2 --mem=66G \
  "$python_bin" -B -m scheduling.first_two_launcher \
  --release-receipt "$receipt" --release-sha256 "$receipt_sha"
