"""Bounded transport for prepared project operations; preserves every receipt."""

import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo))
from scripts.sr_f6.pilot_v3_closure_operations import contracts  # noqa: E402

base = Path(__file__).resolve().parents[3] / ".local/pilot_v3_closure_20261029"
name = sys.argv[1]
assert Path(name).name == name and name.endswith(".py") and not name.endswith(".template.py")
project = "/projects/_ssd/varunssd/louis-ssvc"
revision, deployment = contracts.producer(json.loads((base / "CODE_REVISION.json").read_text()))
code = str(deployment)
activation = json.loads((base / "ACTIVATED_PREPARE_REQUEST.json").read_text())
contracts.require(
    activation["producer_revision"] == revision and activation["source_deployment"] == code,
    "Wrong activated source",
)
if name == "submit_prepare_remote.py":
    import hashlib

    contracts.require(
        hashlib.sha256((base / name).read_bytes()).hexdigest() == activation["sha256"],
        "Activated operator changed",
    )
command = shlex.join(
    [
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONOPTIMIZE=0",
        "PYTHONPATH=" + code + "/src:" + code,
        project + "/envs/ssvc-py312/bin/python3.12",
        "-B",
        "-",
    ]
)
argv = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=12",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=2",
    "eee-cluster",
    command,
]
at = datetime.datetime.now(datetime.timezone.utc)
try:
    done = subprocess.run(
        argv, input=(base / name).read_text(), capture_output=True, text=True, timeout=50
    )
    receipt = dict(
        at=at.isoformat(),
        script=name,
        returncode=done.returncode,
        stdout=done.stdout,
        stderr=done.stderr,
    )
except subprocess.TimeoutExpired as exc:
    receipt = dict(at=at.isoformat(), script=name, status="UNKNOWN_TIMEOUT", error=str(exc))
path = base / (Path(name).stem + "_" + at.strftime("%Y%m%dT%H%M%S") + ".json")
with path.open("x") as stream:
    json.dump(receipt, stream, indent=2)
print("Receipt: " + path.name)
if receipt.get("returncode") == 0:
    value = json.loads(receipt["stdout"])
    print(
        json.dumps({k: v for k, v in value.items() if k not in ("capacity", "argv", "files")})[
            :8000
        ]
    )
else:
    print(json.dumps(receipt)[:8000])
sys.exit(receipt.get("returncode", 1))
