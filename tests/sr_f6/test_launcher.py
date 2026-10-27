"""Exercise the shell launcher without allocating GPUs or loading a model."""

import json
import os
import subprocess
import sys
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[2] / "scripts/sr_f6/census.sbatch"


def launch(tmp_path, *, child_exit=0):
    binary = tmp_path / "bin"
    binary.mkdir()
    child = binary / "srun"
    child.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "keys = ['CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', "
        "'TOKENIZERS_PARALLELISM', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'SRF6_PARENT']\n"
        "record = {'argv': sys.argv[1:], 'environment': {k: os.environ.get(k) for k in keys}}\n"
        "target = pathlib.Path(os.environ['SRF6_TEST_RECORDS']) / (str(os.getpid()) + '.json')\n"
        "target.write_text(json.dumps(record))\n"
        f"sys.exit({child_exit})\n"
    )
    child.chmod(0o755)
    records = tmp_path / "records"
    records.mkdir()
    environment = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "SRF6_CODE": str(tmp_path),
        "SRF6_ROOT": str(tmp_path),
        "SRF6_PARENT": "b0",
        "SRF6_TEST_RECORDS": str(records),
    }
    for key in (
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        environment.pop(key, None)
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=environment, capture_output=True, text=True, check=False
    )
    return result, [json.loads(p.read_text()) for p in sorted(records.glob("*.json"))]


def test_both_workers_receive_required_deterministic_offline_environment(tmp_path):
    result, children = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(children) == 2
    shards = set()
    for child in children:
        assert child["environment"] == {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SRF6_PARENT": "b0",
        }
        args = child["argv"]
        assert "--mem=33G" in args and "--gres=gpu:pro6000:1" in args
        assert args[args.index("-m") + 1] == "sr_f6.census"
        assert args[args.index("--shards") + 1] == "2"
        shards.add(args[args.index("--shard") + 1])
    assert shards == {"0", "1"}


def test_worker_failure_is_reported_by_launcher(tmp_path):
    result, children = launch(tmp_path, child_exit=3)
    assert len(children) == 2
    assert result.returncode == 1
