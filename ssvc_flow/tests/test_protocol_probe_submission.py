"""Fixed lane resource bounds and durable Slurm submission intent regression."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "probe_submit", Path(__file__).parents[1] / "scripts/protocol_state_probes/submit_lanes.py"
)
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def args(tmp_path, lanes):
    base = ["submit"]
    for key in ["run", "project", "python", "runtime", "cache", "partition", "account", "qos"]:
        base.extend(["--" + key, str(tmp_path / key)])
    return [*base, "--execute", "--lanes", *map(str, lanes)]


def test_staged_submission_never_duplicates_or_exceeds_five(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(submit.subprocess, "check_output", lambda *a, **k: "")

    def sbatch(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(5000 + len(calls)) + "\n", "")

    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    monkeypatch.setattr("sys.argv", args(tmp_path, [1]))
    submit.main()
    monkeypatch.setattr("sys.argv", args(tmp_path, [2, 3, 4, 5]))
    submit.main()
    assert len(calls) == 5
    assert all("--gres=gpu:pro6000:1" in c for c in calls)
    assert all("--mem=33G" in c for c in calls)
    assert calls[0][-1] == "S96"
    assert calls[-1][-2:] == ["REP96", "REP32"]
    monkeypatch.setattr("sys.argv", args(tmp_path, [1]))
    with pytest.raises(RuntimeError, match="never blindly resubmit"):
        submit.main()
    assert len(calls) == 5


def test_ambiguous_prior_submission_blocks_new_lane(tmp_path, monkeypatch):
    root = tmp_path / "run" / "scheduler"
    root.mkdir(parents=True)
    (root / "lane-1.intent.json").write_text(json.dumps({"lane": "lane-1"}))
    monkeypatch.setattr("sys.argv", args(tmp_path, [2]))
    with pytest.raises(RuntimeError, match="Ambiguous prior intent"):
        submit.main()
