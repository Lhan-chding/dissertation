"""Read-only request selection and mocked dispatch; no jobs are submitted."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).parents[2] / "scripts/run_modeling_v3_cpu_array.py"
    spec = importlib.util.spec_from_file_location("cpu_array_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_list(tmp_path, *, rows=None):
    for seed in (21001, 21002):
        path = tmp_path / f"request-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "argv": [
                        "validate-cpu",
                        "--seeds",
                        str(seed),
                        "--arms",
                        "X_BASE",
                        "X_VALID",
                        "--out",
                        str(tmp_path / f"out-{seed}"),
                    ]
                }
            )
        )
    rows = (
        rows
        if rows is not None
        else [
            {"path": f"request-{seed}.json", "sha256": digest(tmp_path / f"request-{seed}.json")}
            for seed in (21001, 21002)
        ]
    )
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(rows))
    return path, digest(path)


def test_frozen_order_index_and_relative_paths_do_not_change_request_bytes(runner, tmp_path):
    path, seal = frozen_list(tmp_path)
    originals = {p: p.read_bytes() for p in tmp_path.iterdir()}
    selected = runner.select_request(path, seal, 1, "1")
    assert selected["request_index"] == 1
    assert selected["request_count"] == 2
    assert selected["request"] == str((tmp_path / "request-21002.json").resolve())
    assert selected["request_sha256"] == digest(tmp_path / "request-21002.json")
    assert selected["request_list_sha256"] == seal
    assert all(p.read_bytes() == contents for p, contents in originals.items())
    assert set(tmp_path.iterdir()) == set(originals)


@pytest.mark.parametrize(
    "index,task",
    [
        (-1, "-1"),
        (2, "2"),
        (0, "1"),
        (True, "1"),
        ("0", ""),
        ("+0", "0"),
        (0, chr(0xFF10)),
        (0, None),
    ],
)
def test_invalid_or_disagreeing_array_indices_are_rejected(runner, tmp_path, index, task):
    path, seal = frozen_list(tmp_path)
    with pytest.raises(ValueError, match=r"index|task"):
        runner.select_request(path, seal, index, task)


def test_outer_seal_checked_before_parsing_and_selected_request_tampering(runner, tmp_path):
    path, seal = frozen_list(tmp_path)
    original = path.read_bytes()
    path.write_bytes(b"not json")
    with pytest.raises(ValueError, match="list hash"):
        runner.select_request(path, seal, 0, "0")
    path.write_bytes(original)
    request = tmp_path / "request-21001.json"
    request.write_bytes(request.read_bytes() + b" ")
    with pytest.raises(ValueError, match="request hash"):
        runner.select_request(path, seal, 0, "0")


@pytest.mark.parametrize(
    "rows",
    [[], {}, [{"path": "x", "sha256": "bad"}], [{"path": "x", "sha256": "a" * 64, "argv": []}]],
)
def test_manifest_schema_is_strict(runner, tmp_path, rows):
    path, seal = frozen_list(tmp_path, rows=rows)
    with pytest.raises(ValueError, match=r"list|entry|sha256"):
        runner.select_request(path, seal, 0, "0")


def test_duplicate_jobs_and_unselected_invalid_entries_are_rejected(runner, tmp_path):
    path, _ = frozen_list(tmp_path)
    rows = json.loads(path.read_text())
    path.write_text(json.dumps([rows[0], rows[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        runner.select_request(path, digest(path), 0, "0")
    rows[1]["path"] = str((tmp_path / "request-21001.json").resolve())
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="duplicate"):
        runner.select_request(path, digest(path), 0, "0")
    rows[1] = {"path": "unused.json", "sha256": "not-a-hash"}
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="sha256"):
        runner.select_request(path, digest(path), 0, "0")


def environment(monkeypatch, runner):
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setenv("SLURM_JOB_ID", "155999")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")


def test_dispatch_calls_existing_wrapper_without_shell_or_argv_changes(
    runner, tmp_path, monkeypatch, capsys
):
    path, seal = frozen_list(tmp_path)
    environment(monkeypatch, runner)
    invocations = []

    def run(argv, **kwargs):
        invocations.append((argv, kwargs))
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["--request-list", str(path), "--sha256", seal, "--index", "1"]) == 17
    args, kwargs = invocations[0]
    request = tmp_path / "request-21002.json"
    assert args == [
        runner.sys.executable,
        str(Path(runner.__file__).with_name("run_modeling_v3_cpu_request.py")),
        "--request",
        str(request),
        "--sha256",
        digest(request),
    ]
    assert kwargs == {"check": False}
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["request_index"] == 1 and receipt["request_list_sha256"] == seal
    assert receipt["slurm_job_id"] == "155999"
    assert not (tmp_path / "out-21002").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("platform", "darwin"),
        ("SLURM_JOB_ID", ""),
        ("SLURM_JOB_ID", "abc"),
        ("SLURM_JOB_ID", chr(0xFF11) + chr(0xFF12)),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("HIP_VISIBLE_DEVICES", "0"),
    ],
)
def test_production_environment_rejected_before_dispatch(
    runner, tmp_path, monkeypatch, field, value
):
    path, seal = frozen_list(tmp_path)
    environment(monkeypatch, runner)
    if field == "platform":
        monkeypatch.setattr(runner.sys, "platform", value)
    else:
        monkeypatch.setenv(field, value)
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("dispatched")
    )
    with pytest.raises(ValueError, match=r"Linux|Slurm|GPU|empty"):
        runner.main(["--request-list", str(path), "--sha256", seal, "--index", "1"])
