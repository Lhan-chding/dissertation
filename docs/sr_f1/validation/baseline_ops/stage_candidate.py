"""Stage a hash-listed baseline candidate and submit CPU validation only."""

import datetime
import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010")
INC = ROOT / "technical_incidents/baseline_parallel_20261010"
CANDIDATE = ROOT / "code_baseline_parallel_candidate_20261010"
PYTHON = "/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12"
FREEZE = "c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8"
PARENT = "fdf036fc9836aa9de7b48d8978b03bb39926ac43cef113ed2b254329b61a21fd"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    assert ROOT.resolve() == ROOT and sha(ROOT / "EXECUTION_FREEZE.json") == FREEZE
    assert sha(ROOT / "ENGINE_MULTIGPU_REPAIR.json") == PARENT
    assert not (ROOT / "BASELINE_PARALLEL_REPAIR.json").exists()
    assert not CANDIDATE.exists() and not (ROOT / "STOP").exists()
    inventory = read(INC / "inventory.json")
    assert sha(INC / "source.tar.gz") == inventory["sha256"]
    production = read(ROOT / "code/SOURCE_DEPLOYMENT.json")
    assert production["source_commit"] == "22735a2def1ee1ed79d69f603e580c67e301c5dd"
    for name, expected in production["source_file_hashes"].items():
        assert sha(ROOT / "code" / name) == expected, name
    CANDIDATE.mkdir()
    with tarfile.open(INC / "source.tar.gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(inventory["files"]) == len({x.name for x in members})
        for member in members:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute() and ".." not in relative.parts
            assert member.name in inventory["files"]
            data = archive.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == inventory["files"][member.name]
            target = CANDIDATE / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    identity = read(CANDIDATE / "SOURCE_DEPLOYMENT.json")
    assert identity["source_commit"] == inventory["source_commit"]
    for name, expected in identity["source_file_hashes"].items():
        assert sha(CANDIDATE / name) == expected, name
    save(
        INC / "STAGING_VERIFIED.json",
        {
            "status": "CANDIDATE_ONLY_PRODUCTION_UNCHANGED",
            "source_commit": identity["source_commit"],
            "source_tree_sha256": identity["source_tree_sha256"],
            "files": len(inventory["files"]),
            "production_source_commit": production["source_commit"],
            "freeze_sha256": FREEZE,
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    script = INC / "candidate_cpu.sbatch"
    with script.open("x") as stream:
        stream.write(f"""#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH={CANDIDATE}/src
cd {CANDIDATE}
{PYTHON} -m pytest -q --import-mode=importlib tests/sr_f1 \\
 tests/mm_dev/test_cached_teacher_forcing.py tests/mm_core/test_training.py \\
 docs/sr_f1/package/tests > {INC}/CANDIDATE_SERVER_TESTS.log 2>&1
{PYTHON} - > {INC}/CANDIDATE_SOURCE_CHECK.json <<'CHECK'
import json
from sr_f1.prepare import verify_package,source_identity
print(json.dumps(dict(package=verify_package(),source=source_identity())))
CHECK
""")
    command = [
        "sbatch",
        "--parsable",
        "--account=rose",
        "--qos=override-limits-but-killable",
        "--partition=cluster02",
        "--cpus-per-task=2",
        "--mem=3G",
        "--time=30",
        "--job-name=srf11-baseline-cpu",
        "--output=" + str(INC / "candidate_cpu_%j.log"),
        str(script),
    ]
    save(INC / "CANDIDATE_CPU_SUBMISSION_INTENT.json", {"command": command})
    result = subprocess.run(command, text=True, capture_output=True, timeout=45)
    save(
        INC / "CANDIDATE_CPU_SUBMISSION.json",
        {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    assert result.returncode == 0, result.stderr
    job = result.stdout.strip().split(";")[0]
    assert re.fullmatch("[0-9]+", job)
    print(
        json.dumps(
            {
                "status": "CANDIDATE_STAGED_CPU_TESTS_SUBMITTED",
                "cpu_job": job,
                "source_commit": identity["source_commit"],
                "production_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
