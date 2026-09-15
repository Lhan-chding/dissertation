"""Model-free authorization fixtures and mocked child processes only."""

import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from src.modeling_v3.io import atomic_json, canonical_hash, sha256_file, source_identity
from src.modeling_v3.schema import load_config

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_C = "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc"
GPU_D = "GPU-dddddddd-dddd-dddd-dddd-dddddddddddd"


@pytest.fixture
def launcher(monkeypatch):
    path = Path(__file__).parents[2] / "scripts/run_modeling_v3_q4_pair.py"
    spec = importlib.util.spec_from_file_location("q4_pair_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "query_allocated_gpus",
        lambda tokens: [
            {"token": token, "uuid": gpu, "name": "NVIDIA RTX PRO 6000", "index": str(i)}
            for i, (token, gpu) in enumerate(zip(tokens, [GPU_A, GPU_C], strict=True))
        ],
        raising=False,
    )
    return module


def bound(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def authorization(tmp_path, *, pending=False):
    config_path = Path(__file__).parents[2] / "configs/modeling_v3/protocol.json"
    config = load_config(config_path)
    identity = {"source_hash": source_identity()["sha256"], "config_hash": canonical_hash(config)}
    technical = {}
    for phase in ("Q1", "Q2"):
        path = tmp_path / f"{phase}.json"
        atomic_json(path, {**identity, "status": "PASS", "checks": [{"passed": True}]})
        technical[phase] = bound(path)
    plan = {**identity, "origins": [{"id": "initial"}, {"id": "warm"}]}
    plan["plan_hash"] = canonical_hash(plan)
    atomic_json(tmp_path / "bridge.json", plan)
    stage = {
        **identity,
        "phase": "Q4",
        "workers": [0, 1],
        "operations": ["vlm-smoke"],
        "max_concurrent_project_gpus": 2,
        "q4_plan_hash": plan["plan_hash"],
        "technical_receipts": technical,
        "operator_authorization": "PENDING_FIRST_GPU_CONFIRMATION",
    }
    atomic_json(tmp_path / "prepared.json", stage)
    (tmp_path / "handoff.md").write_text("Fixture reviewed commands and runtime. No GPU execution.")
    if not pending:
        stage.update(
            operator_authorization="CONFIRMED_FIRST_GPU_SUBMISSION",
            first_gpu_confirmation={
                "scope": "Q4",
                "confirmation_text": "fixture only",
                "reviewed_handoff": bound(tmp_path / "handoff.md"),
                "prepared_stage": bound(tmp_path / "prepared.json"),
            },
        )
    atomic_json(tmp_path / "authorized.json", stage)
    bindings = {
        "v3_stage_lock": bound(tmp_path / "authorized.json"),
        "q4_bridge_plan": bound(tmp_path / "bridge.json"),
    }
    atomic_json(tmp_path / "bindings.json", bindings)
    return config_path, tmp_path / "bindings.json"


@pytest.mark.parametrize(
    "raw",
    ["", "0", "0,0", "0,00", "0,1,2", "MIG-a,MIG-b", "GPU-a,GPU-a", "GPU-a,", "-1,0", "0,abc"],
)
def test_requires_two_distinct_whole_gpu_allocation_tokens(launcher, raw):
    with pytest.raises(ValueError, match=r"GPU|token"):
        launcher.gpu_tokens(raw)


def test_plan_preserves_original_tokens_and_separates_outputs(launcher, tmp_path):
    config, bindings = authorization(tmp_path)
    plan = launcher.build_plan(config, bindings, tmp_path / "pair", "GPU-abc,7")
    assert plan["allocation_tokens"] == ["GPU-abc", "7"]
    assert plan["distinct_gpu_uuids_verified"] is False
    assert [w["cuda_visible_devices"] for w in plan["workers"]] == ["GPU-abc", "7"]
    for index, worker in enumerate(plan["workers"]):
        assert worker["command"][-2:] == ["--allow-gpu", "--acknowledge-new-experiment"]
        assert worker["command"][worker["command"].index("--device") + 1] == "cuda:0"
        assert worker["command"][worker["command"].index("--worker-index") + 1] == str(index)
        assert worker["out"] == str(tmp_path / "pair" / f"worker_{index}")
    assert not (tmp_path / "pair").exists()


def test_pending_authorization_and_tampered_handoff_refused(launcher, tmp_path):
    pending = tmp_path / "pending"
    pending.mkdir()
    config, bindings = authorization(pending, pending=True)
    with pytest.raises(PermissionError, match="authorization"):
        launcher.build_plan(config, bindings, pending / "out", "0,1")
    approved = tmp_path / "approved"
    approved.mkdir()
    config, bindings = authorization(approved)
    (approved / "handoff.md").write_text("modified")
    with pytest.raises(ValueError, match=r"handoff|changed"):
        launcher.build_plan(config, bindings, approved / "out", "0,1")


def test_source_binding_and_explicit_resume_gate(launcher, tmp_path):
    config, bindings = authorization(tmp_path)
    out = tmp_path / "pair"
    out.mkdir()
    with pytest.raises(FileExistsError, match="resume"):
        launcher.build_plan(config, bindings, out, "0,1")
    plan = launcher.build_plan(config, bindings, out, "0,1", resume=True)
    assert all("--resume" in w["command"] for w in plan["workers"])
    stage_path = tmp_path / "authorized.json"
    stage = json.loads(stage_path.read_text())
    stage["source_hash"] = "modified"
    stage_path.write_text(json.dumps(stage))
    value = json.loads(bindings.read_text())
    value["v3_stage_lock"] = bound(stage_path)
    bindings.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source"):
        launcher.build_plan(config, bindings, out, "0,1", resume=True)


class Child:
    def __init__(self, pid, code=0):
        self.pid, self.code, self.returncode = pid, code, None

    def poll(self):
        self.returncode = self.code
        return self.code

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = self.code
        return self.returncode


def test_two_children_single_visible_gpu_and_failure_receipt(launcher, tmp_path, monkeypatch):
    config, bindings = authorization(tmp_path)
    plan = launcher.build_plan(config, bindings, tmp_path / "pair", "3,8")
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Child(9000 + len(calls), 17 if len(calls) == 1 else 0)

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    result = launcher.execute_pair(plan)
    assert result["exit_code"] != 0
    assert [r["returncode"] for r in result["workers"]] == [17, 0]
    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in calls] == ["3", "8"]
    assert [kwargs["env"]["SSVC_V3_EXPECTED_GPU_UUID"] for _, kwargs in calls] == [GPU_A, GPU_C]
    assert all(kwargs["start_new_session"] is True for _, kwargs in calls)
    assert all("shell" not in kwargs for _, kwargs in calls)
    saved = list((tmp_path / "pair/pair_attempts").glob("*/RESULT.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["distinct_gpu_uuids_verified"] is False


def test_sigterm_only_targets_own_child_groups(launcher, tmp_path, monkeypatch):
    config, bindings = authorization(tmp_path)
    plan = launcher.build_plan(config, bindings, tmp_path / "pair", "0,1")
    handlers, children, killed = {}, [], []

    def install(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    class Running(Child):
        def poll(self):
            if len(children) == 2 and not killed:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return self.returncode

    def popen(*_args, **_kwargs):
        child = Running(9100 + len(children))
        children.append(child)
        return child

    def killpg(pid, signum):
        killed.append((pid, signum))
        next(child for child in children if child.pid == pid).returncode = -signum

    monkeypatch.setattr(launcher.signal, "signal", install)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.os, "killpg", killpg)
    result = launcher.execute_pair(plan)
    assert result["exit_code"] == 128 + signal.SIGTERM
    assert {pid for pid, _ in killed} == {9100, 9101}
    assert all(sig == signal.SIGTERM for _, sig in killed)
    assert all(row["returncode"] == -signal.SIGTERM for row in result["workers"])
    assert handlers[signal.SIGTERM] == signal.SIG_DFL


def test_partial_spawn_failure_reaps_only_started_child_and_preserves_attempt(
    launcher, tmp_path, monkeypatch
):
    config, bindings = authorization(tmp_path)
    plan = launcher.build_plan(config, bindings, tmp_path / "pair", "0,1")
    child = Child(9200)
    child.poll = lambda: child.returncode
    calls = []

    def popen(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise OSError("fixture failed second spawn")
        return child

    def killpg(pid, signum):
        assert pid == 9200
        child.returncode = -signum

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.os, "killpg", killpg)
    result = launcher.execute_pair(plan)
    assert result["exit_code"] != 0
    assert result["workers"][1]["status"] == "NOT_STARTED"
    identity = (tmp_path / "pair/PAIR_IDENTITY.json").read_bytes()
    resumed = launcher.build_plan(config, bindings, tmp_path / "pair", "2,3", resume=True)
    assert resumed["identity"] == plan["identity"]
    assert (tmp_path / "pair/PAIR_IDENTITY.json").read_bytes() == identity


def test_model_free_plan_imports_no_torch_or_transformers(launcher, tmp_path):
    config, bindings = authorization(tmp_path)
    code = (
        "import runpy,sys; module=runpy.run_path(sys.argv[1]); "
        "module['build_plan'](sys.argv[2],sys.argv[3],sys.argv[4],'0,1', "
        "gpu_query=lambda tokens:[dict(token=t,uuid=u,name='PRO 6000',index=str(i)) "
        "for i,(t,u) in enumerate(zip(tokens,sys.argv[5:]))]); "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            launcher.__file__,
            str(config),
            str(bindings),
            str(tmp_path / "pair"),
            GPU_A,
            GPU_C,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "pair").exists()


def completed_worker(root, authorization_identity, index, gpu, *, corrupt_null_hash=False):
    root.mkdir(parents=True)
    receipt = root / "null_invocations/000000/null_receipt.json"
    receipt.parent.mkdir(parents=True)
    atomic_json(receipt, {"gpu_identity": gpu})
    atomic_json(root / "null_receipt.json", {"gpu_identity": gpu})
    null, anchor = bound(receipt), bound(root / "null_receipt.json")
    atomic_json(root / "null_receipt_binding.json", anchor)
    hardware = receipt.with_name("hardware.json")
    atomic_json(hardware, {"allocated_gpu": {"uuid": gpu}})
    comparison = receipt.with_name("comparison.json")
    atomic_json(comparison, {"status": "PASS", "anchor": anchor, "current": null})
    invocation = receipt.with_name("invocation.json")
    atomic_json(invocation, {"null_receipt": null, "hardware": bound(hardware)})
    result = {
        "status": "Q4_WORKER_COMPLETED_PENDING_TWO_GPU_NULL",
        "execution_kind": "REAL_CUDA_MODEL",
        "worker_index": index,
        "config_hash": authorization_identity["config_hash"],
        "source_hash": authorization_identity["source_hash"],
        "Q4_plan_hash": authorization_identity["q4_plan_hash"],
        "null_receipt": {**null, "sha256": "bad"} if corrupt_null_hash else null,
        "null_anchor": anchor,
        "null_comparison": bound(comparison),
        "null_invocation": bound(invocation),
    }
    atomic_json(root / "result.json", result)
    atomic_json(
        root / "manifest.json",
        {
            "files": [
                {
                    "path": str(p.relative_to(root)),
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ]
        },
    )
    atomic_json(
        root / "completed.json",
        {"status": "COMPLETED", "manifest_sha256": sha256_file(root / "manifest.json")},
    )


def allocation(tokens, uuids):
    return [
        {"token": token, "uuid": gpu, "name": "PRO 6000", "index": str(i)}
        for i, (token, gpu) in enumerate(zip(tokens, uuids, strict=True))
    ]


def test_completed_a_and_current_c_a_remaps_incomplete_worker(launcher, tmp_path):
    config, bindings = authorization(tmp_path)
    base = launcher.build_plan(config, bindings, tmp_path / "pair", "0,1")
    completed_worker(tmp_path / "pair/worker_0", base["identity"], 0, GPU_A)
    plan = launcher.build_plan(
        config,
        bindings,
        tmp_path / "pair",
        "4,9",
        resume=True,
        gpu_query=lambda tokens: allocation(tokens, [GPU_C, GPU_A]),
    )
    assert [w["cuda_visible_devices"] for w in plan["workers"]] == ["9", "4"]
    assert plan["expected_effective_null_uuids"] == [GPU_A, GPU_C]
    assert plan["completed_workers"][0]["null_receipt_uuid"] == GPU_A


def test_old_a_not_allocated_new_c_d_preserves_safe_mapping(launcher, tmp_path):
    config, bindings = authorization(tmp_path)
    base = launcher.build_plan(config, bindings, tmp_path / "pair", "0,1")
    completed_worker(tmp_path / "pair/worker_0", base["identity"], 0, GPU_A)
    plan = launcher.build_plan(
        config,
        bindings,
        tmp_path / "pair",
        "4,9",
        resume=True,
        gpu_query=lambda tokens: allocation(tokens, [GPU_C, GPU_D]),
    )
    assert [w["cuda_visible_devices"] for w in plan["workers"]] == ["4", "9"]
    assert plan["expected_effective_null_uuids"] == [GPU_A, GPU_D]


@pytest.mark.parametrize("fault", ["same_uuid", "null_hash", "manifest_hash"])
def test_completed_worker_bad_originals_or_same_uuid_fail_before_launch(launcher, tmp_path, fault):
    config, bindings = authorization(tmp_path)
    base = launcher.build_plan(config, bindings, tmp_path / "pair", "0,1")
    completed_worker(
        tmp_path / "pair/worker_0",
        base["identity"],
        0,
        GPU_A,
        corrupt_null_hash=fault == "null_hash",
    )
    if fault == "same_uuid":
        completed_worker(tmp_path / "pair/worker_1", base["identity"], 1, GPU_A)
    if fault == "manifest_hash":
        (tmp_path / "pair/worker_0/result.json").write_text("modified")
    with pytest.raises(ValueError, match=r"UUID|hash|manifest|bound"):
        launcher.build_plan(config, bindings, tmp_path / "pair", "0,1", resume=True)
    assert not (tmp_path / "pair/pair_attempts").exists()


def test_nvidia_smi_queries_original_tokens_and_rejects_malformed_uuid(launcher):
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        token = command[command.index("--id") + 1]
        return type(
            "Result",
            (),
            {"stdout": f"{token}, {GPU_A if token == '4' else GPU_C}, NVIDIA RTX PRO 6000\n"},
        )()

    # Access the original implementation because the fixture mocks the hardware helper.
    spec = importlib.util.spec_from_file_location("q4_pair_query_fixture", launcher.__file__)
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    values = real.query_allocated_gpus(["4", "9"], run=query)
    assert [v["uuid"] for v in values] == [GPU_A, GPU_C]
    assert all(command[0] == "nvidia-smi" for command in calls)
    with pytest.raises(ValueError, match=r"UUID|query"):
        real.query_allocated_gpus(
            ["4", "9"], run=lambda *_a, **_k: type("Result", (), {"stdout": "malformed"})()
        )


@pytest.mark.parametrize("platform,job", [("darwin", "123"), ("linux", ""), ("linux", "job")])
def test_real_entry_requires_linux_numeric_slurm_before_planning(
    launcher, monkeypatch, platform, job
):
    monkeypatch.setattr(launcher.sys, "platform", platform)
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setattr(launcher, "build_plan", lambda *_a, **_k: pytest.fail("planned"))
    with pytest.raises(ValueError, match=r"Linux|Slurm"):
        launcher.main(["--config", "unused", "--bindings", "unused", "--out", "unused"])
