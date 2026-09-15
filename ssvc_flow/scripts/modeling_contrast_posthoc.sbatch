#!/bin/bash
#SBATCH --job-name=ssvc-n4-audit
#SBATCH --partition=cluster02
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

set -euo pipefail
: "${SSVC_N4_ROOT:?Set the existing isolated experiment root}"
: "${SSVC_N4_PYTHON:?Set the verified Python executable}"
: "${SSVC_N4_EXECUTION_JOB_ID:?Set the completed N4 execution job}"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${SSVC_N4_ROOT}/tmp/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
cd "${SSVC_N4_ROOT}/checkout/ssvc_flow"
/usr/bin/time -v "${SSVC_N4_PYTHON}" - <<'PY'
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

from src.modeling_contrast.protocol import ROOT, cpu_environment, load_config
from src.modeling_contrast.server_checks import RuntimeResourceWatchdog
from src.modeling_contrast.server_cli import require_cpu_allocation

started = time.perf_counter()
require_cpu_allocation()
cpu_environment()
base = Path(os.environ["SSVC_N4_ROOT"]).resolve()
run_root = ROOT / "runs/modeling_contrast_v2"
execution = base / "control" / f"execute_{os.environ['SSVC_N4_EXECUTION_JOB_ID']}"
read = lambda path: json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


finished = read(execution / "COMPLETE.json")
assert finished["status"] == "COMPLETE" and finished["mode"] == "execute"
assert read(execution / "EXECUTION_SUMMARY.json")["status"] == "COMPLETE"
assert finished["selection_manifest_sha256"] == sha(run_root / "N3_server/RUN_MANIFEST.json")
watch = json.loads((execution / "resource_watchdog.jsonl").read_text().splitlines()[-1])
assert watch["status"] == "PASS"
prior = watch["prior_wall_seconds"] + finished["wall_seconds"]
code_root = base / "posthoc_code"
code = read(code_root / "MANIFEST.json")
for name, expected in code["files"].items():
    assert Path(name).name == name and sha(code_root / name) == expected
assert "torch" not in sys.modules
control = base / "control" / f"posthoc_{os.environ['SLURM_JOB_ID']}"
control.mkdir(parents=True, exist_ok=False)
out = run_root / f"posthoc_{os.environ['SLURM_JOB_ID']}"
out.mkdir(exist_ok=False)
config = load_config(ROOT / "configs/modeling_contrast/protocol_server.json")
with RuntimeResourceWatchdog(config, run_root, control, base / "tmp", prior_wall_seconds=prior):
    sys.argv = [str(code_root / "audit_modeling_n4_results.py"),
                "--run-root", str(run_root), "--out", str(out / "N4_AUDIT.json")]
    audit_main = runpy.run_path(sys.argv[0])["main"]
    assert audit_main() == 0
    assert read(out / "N4_AUDIT.json")["status"] == "PASS"
    sys.argv = [str(code_root / "summarize_modeling_n4_results.py"),
                "--run-root", str(run_root), "--out-dir", str(out / "summary"),
                "--raw-audit-json", str(out / "N4_AUDIT.json")]
    summary_main = runpy.run_path(sys.argv[0])["main"]
    summary_main()
    assert "torch" not in sys.modules
    outputs = {str(path.relative_to(out)): sha(path)
               for path in sorted(out.rglob("*")) if path.is_file()}
    complete = {
        "status": "COMPLETE", "job_id": os.environ["SLURM_JOB_ID"],
        "execution_job_id": os.environ["SSVC_N4_EXECUTION_JOB_ID"],
        "code": code, "outputs": outputs, "output_root": str(out),
        "wall_seconds": time.perf_counter() - started,
        "prior_experiment_wall_seconds": prior,
        "model_library_imported": False, "new_optimizer_updates": 0,
        "new_gpu_calls": 0, "new_qwen_calls": 0, "online_ssvc": False,
    }
(control / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")
print(json.dumps({"status": "POSTHOC_COMPLETE", "output_root": str(out)}))
PY
