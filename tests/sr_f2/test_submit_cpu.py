"""CPU submission and partial-phase recovery use fake Slurm, never the cluster."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sr_f2/submit_cpu.py"
SPEC = importlib.util.spec_from_file_location("sr_f2_submit_cpu", SCRIPT)
cpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cpu)


def fields(intent, job="100", *, user="alice"):
    return dict(JobId=job, JobName=intent["job_name"], Account="rose", Partition="cluster02",
                QOS=cpu.CPU_QOS, Command=intent["script"], Comment=intent["comment"],
                ReqNodeList="cpu-2", NumCPUs="4", NumNodes="1", UserId=user + "(1)",
                ReqTRES="cpu=4,mem=12G",
                TimeLimit="04:00:00" if intent["kind"] == "preflight" else "3-00:00:00",
                JobState="PENDING", Reason="JobHeldUser")


class FakeSlurm:
    def __init__(self):
        self.jobs, self.accounting = {}, {}
        self.calls = []
        self.fail_submit = False
        self.fail_release = False

    def queue(self, user):
        return dict(self.jobs)

    def show(self, job):
        return self.jobs[job]

    def terminal(self, job):
        return self.accounting.get(job)

    def run(self, args):
        self.calls.append(args)
        if args[0] == "sbatch":
            intent = cpu.read_json(Path(args[-1]).parent / "INTENT.json")
            assert intent["script"] == args[-1]
            if self.fail_submit:
                return SimpleNamespace(returncode=1, stdout="", stderr="transport unknown")
            job = str(99 + sum(call[0] == "sbatch" for call in self.calls))
            self.jobs[job] = fields(intent, job)
            return SimpleNamespace(returncode=0, stdout=job + "\n", stderr="")
        if args[:2] == ["scontrol", "release"]:
            if self.fail_release:
                return SimpleNamespace(returncode=1, stdout="", stderr="release unknown")
            self.jobs[args[2]].update(JobState="RUNNING", Reason="None")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(args)

    def finish(self, job, *, state="COMPLETED", exit_code="0:0"):
        fields_ = self.jobs.pop(job)
        self.accounting[job] = dict(job_id=job, state=state, exit_code=exit_code,
                                    user="alice", account="rose", qos=cpu.CPU_QOS,
                                    name=fields_["JobName"])


@pytest.fixture
def submitter(tmp_path, monkeypatch):
    root = tmp_path / "new-run"
    code = root / "code"
    scripts = code / "scripts/sr_f2"
    scripts.mkdir(parents=True)
    for name in ("prepare.py", "audit_prior.py", "verify_chartqa_cohort.py",
                 "cpu_preflight.py", "controller.py", "submit_cpu.py"):
        (scripts / name).write_text("# deployed " + name + "\n")
    definition = code / "docs/sr_f2/CHARTQA_122_USER_CLARIFICATION.json"
    cpu.save_json(definition, dict(clarified=True))
    private = root / "private_inputs/audit_chartqa_sensitivity_rows.jsonl"
    private.parent.mkdir()
    private.write_text("{}\n")
    manifest = root / "deployment/SOURCE_MANIFEST.json"
    cpu.save_json(manifest, dict(git_commit="a" * 40, files=[
        dict(path=str(p.relative_to(code)), sha256=cpu.file_hash(p), bytes=p.stat().st_size)
        for p in sorted(code.rglob("*")) if p.is_file()
    ]))
    old = tmp_path / "old-run"
    old.mkdir()
    monkeypatch.setattr(cpu, "PARENT_ROOT", str(old))
    monkeypatch.setattr(cpu, "verify_registration", lambda root: dict(
        source_manifest_sha256=cpu.verify_source_manifest(manifest, code), code_root=str(code)))
    return cpu.CpuSubmitter(root, "preflight", sys.executable, manifest,
                            slurm=FakeSlurm(), user="alice")


def finish_preflight(submitter):
    submitter.submit()
    for name in ("CPU_ACCEPTANCE.json", "prior_audit/inventory/SUMMARY.json",
                 "prior_audit/blind/SUMMARY.json", "prior_audit/rescore/SUMMARY.json"):
        cpu.save_json(submitter.root / name, dict(status="PASS"))
    cpu.save_json(submitter.root / "prior_audit/chartqa122/COHORT_VERIFICATION.json",
                  dict(status="PASS", count=122, raw_evidence_status="RAW_AND_GOLD_REVERIFIED",
                       raw_output_count=1586, cohort_identity_verified=True))
    registration = cpu.verify_registration(submitter.root)
    cpu.save_json(submitter.directory / "WORKER_COMPLETE.json", dict(status="COMPLETE",
                  source_manifest_sha256=submitter.identity(),
                  registration_sha256=cpu.object_hash(registration)))
    submitter.slurm.finish("100")
    assert submitter.submit()["status"] == "COMPLETE"


def phase_receipt(submitter, *, phase="blind-classify", status="PASS"):
    name = "blind" if phase == "blind-classify" else phase
    output = submitter.root / "prior_audit" / name
    artifact = submitter.root / "source.bin"
    artifact.write_bytes(b"retained source")
    source_manifest = output / "SOURCE_MANIFEST.json"
    cpu.save_json(source_manifest, dict(files=[dict(source_path=str(artifact),
        canonical_path=str(artifact.resolve()), sha256=cpu.file_hash(artifact),
        bytes=artifact.stat().st_size)]))
    audit = output / "AUDIT.jsonl"
    audit.write_text('{"retained": true}\n')
    summary = dict(status=status, phase=phase, root=cpu.PARENT_ROOT,
                   source_manifest_sha256=cpu.file_hash(source_manifest),
                   audit_jsonl_sha256=cpu.file_hash(audit),
                   script_sha256=cpu.file_hash(submitter.code / "scripts/sr_f2/audit_prior.py"))
    cpu.save_json(output / "SUMMARY.json", summary)
    return output, artifact


def test_cpu_submission_is_held_bound_and_released_once(submitter):
    state = submitter.submit()
    assert state == dict(kind="preflight", status="SUBMITTED", job_id="100")
    args = submitter.slurm.calls[0]
    assert args[:3] == ["sbatch", "--hold", "--parsable"]
    for option in ("--qos=" + cpu.CPU_QOS, "--nodelist=cpu-2", "--cpus-per-task=4",
                   "--mem=12G", "--time=04:00:00"):
        assert option in args
    assert not any("gres" in arg or "soujanya" in arg for arg in args)
    assert submitter.slurm.calls[1] == ["scontrol", "release", "100"]
    assert submitter.submit()["status"] == "RUNNING"
    assert len(submitter.slurm.calls) == 2
    assert (submitter.directory / "ALLOCATION_VERIFIED.json").exists()
    assert (submitter.directory / "RELEASE_INTENT.json").exists()


def test_preflight_command_order_and_runtime_environment(submitter):
    submitter.submit()
    intent = cpu.read_json(submitter.directory / "INTENT.json")
    commands = intent["commands"]
    assert [Path(row[1]).name for row in commands] == ["prepare.py", "audit_prior.py",
        "audit_prior.py", "verify_chartqa_cohort.py", "cpu_preflight.py"]
    assert commands[1][-1] == str(submitter.root / "prior_audit/blind")
    assert commands[2][-1] == str(submitter.root / "prior_audit/rescore")
    assert commands[2][commands[2].index("--blind-receipt") + 1] == commands[1][-1]
    assert commands[3][-1] == str(submitter.root / "prior_audit/chartqa122")
    script = Path(intent["script"]).read_text()
    assert "PYTHONDONTWRITEBYTECODE=1" in script
    assert "PYTHONPATH=" + str(submitter.code / "src") in script
    assert "--execute-preflight" in script
    assert "scontrol" not in script and "sbatch" not in script


@pytest.mark.parametrize("key,value", [("UserId", "bob(2)"), ("QOS", "teacher"),
    ("Command", "/other/script"), ("Comment", "unrelated"), ("ReqNodeList", "gpu-1"),
    ("NumCPUs", "8"), ("ReqTRES", "cpu=4,mem=12G,gres/gpu=1"),
    ("ReqTRES", "cpu=4,mem=11G"), ("ReqTRES", "cpu=4,mem=13G"),
    ("ReqTRES", "cpu=4,mem=24G"), ("TimeLimit", "03:00:00")])
def test_cpu_exact_allocation_mismatch_cannot_release(key, value):
    intent = dict(kind="preflight", job_id="100", job_name="name", script="/script", comment="id")
    row = fields(intent)
    cpu.verify_cpu_allocation(row, intent, "alice", held=True)
    with pytest.raises(PermissionError):
        cpu.verify_cpu_allocation(dict(row, **{key: value}), intent, "alice", held=True)


@pytest.mark.parametrize("nodes", ["1", "1-1"])
def test_cpu_fixed_one_node_accepts_slurm_scalar_and_range(nodes):
    intent = dict(kind="preflight", job_id="100", job_name="name", script="/script", comment="id")
    row = dict(fields(intent), NumNodes=nodes)
    cpu.verify_cpu_allocation(row, intent, "alice", held=True)


@pytest.mark.parametrize("nodes", [None, "", "0", "2", "1-2", "2-2", "0-1", "1-", "1-1,2"])
def test_cpu_one_node_guard_rejects_multiple_or_unknown_node_counts(nodes):
    intent = dict(kind="preflight", job_id="100", job_name="name", script="/script", comment="id")
    row = dict(fields(intent), NumNodes=nodes)
    with pytest.raises(PermissionError, match="NumNodes"):
        cpu.verify_cpu_allocation(row, intent, "alice", held=True)


def test_unknown_cpu_submission_never_resubmits(submitter):
    submitter.slurm.fail_submit = True
    assert submitter.submit()["status"] == "SUBMISSION_UNKNOWN"
    assert submitter.submit()["status"] == "SUBMISSION_UNKNOWN"
    assert len(submitter.slurm.calls) == 1


def test_unknown_cpu_release_never_repeats_mutation(submitter):
    submitter.slurm.fail_release = True
    assert submitter.submit()["status"] == "RELEASE_UNKNOWN"
    assert submitter.submit()["status"] == "RELEASE_UNKNOWN"
    assert len(submitter.slurm.calls) == 2


def test_failed_cpu_job_is_retained_and_never_retried(submitter):
    submitter.submit()
    submitter.slurm.finish("100", state="FAILED", exit_code="1:0")
    assert submitter.submit()["status"] == "FAILED"
    assert submitter.submit()["status"] == "FAILED"
    assert (submitter.directory / "TERMINAL.json").exists()
    assert len(submitter.slurm.calls) == 2


def test_missing_accounting_is_unknown_without_resubmit(submitter):
    submitter.submit()
    submitter.slurm.jobs.clear()
    assert submitter.submit()["status"] == "ACCOUNTING_UNKNOWN"
    assert len(submitter.slurm.calls) == 2


def test_slurm_success_without_phase_receipts_is_failed(submitter):
    submitter.submit()
    submitter.slurm.finish("100")
    assert submitter.submit()["failure"] == "ARTIFACT_VERIFICATION"
    assert (submitter.directory / "VERIFICATION_FAILURE.json").exists()


def test_unrelated_job_is_never_released_or_cancelled(submitter):
    submitter.slurm.jobs["999"] = dict(Comment="unrelated")
    submitter.submit()
    submitter.submit()
    assert submitter.slurm.jobs["999"] == dict(Comment="unrelated")
    assert all(call[0] != "scancel" and call[-1] != "999" for call in submitter.slurm.calls)


def test_existing_script_changes_block_further_mutation(submitter):
    submitter.submit()
    (submitter.directory / "run.sbatch").write_text("changed")
    with pytest.raises(PermissionError, match="script changed"):
        submitter.submit()
    assert len(submitter.slurm.calls) == 2


def test_controller_requires_preflight_slurm_and_artifacts(submitter):
    controller = cpu.CpuSubmitter(submitter.root, "controller", submitter.python,
                                  submitter.manifest, slurm=submitter.slurm, user="alice")
    with pytest.raises(PermissionError, match="absent"):
        controller.submit()
    assert not submitter.slurm.calls
    finish_preflight(submitter)
    assert controller.submit()["status"] == "SUBMITTED"
    args = [row for row in submitter.slurm.calls if row[0] == "sbatch"][-1]
    assert "--time=3-00:00:00" in args
    assert "--mem=12G" in args
    assert cpu.read_json(controller.directory / "INTENT.json")["commands"] == [[
        submitter.python, str(submitter.code / "scripts/sr_f2/controller.py"),
        "--root", str(submitter.root), "--python", submitter.python,
    ]]


def test_preflight_completion_does_not_submit_controller(submitter):
    finish_preflight(submitter)
    assert sum(row[0] == "sbatch" for row in submitter.slurm.calls) == 1
    assert not (submitter.root / "deployment/cpu_jobs/controller/INTENT.json").exists()


def test_completed_preflight_artifact_mutation_blocks_controller(submitter):
    finish_preflight(submitter)
    cpu.save_json(submitter.root / "prior_audit/blind/SUMMARY.json",
                  dict(status="PASS", mutation=True))
    controller = cpu.CpuSubmitter(submitter.root, "controller", submitter.python,
                                  submitter.manifest, slurm=submitter.slurm, user="alice")
    with pytest.raises(PermissionError, match="evidence changed"):
        controller.submit()
    assert sum(row[0] == "sbatch" for row in submitter.slurm.calls) == 1


def test_partial_success_is_reused_only_with_complete_hash_ledger(submitter):
    output, source = phase_receipt(submitter)
    result = cpu.verify_phase(output, "blind-classify", submitter.code)
    assert result["phase"] == "blind-classify"
    source.write_bytes(b"changed source")
    with pytest.raises(PermissionError, match="source changed"):
        cpu.verify_phase(output, "blind-classify", submitter.code)


def test_failed_phase_is_preserved_without_retry(submitter):
    output, _ = phase_receipt(submitter, status="FAIL")
    before = cpu.file_hash(output / "SUMMARY.json")
    with pytest.raises(PermissionError, match="not PASS"):
        cpu.verify_phase(output, "blind-classify", submitter.code)
    assert cpu.file_hash(output / "SUMMARY.json") == before


def test_preflight_executor_cannot_run_on_login_or_another_job(submitter, monkeypatch):
    submitter.submit()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(PermissionError, match="bound Slurm"):
        submitter.execute_preflight()
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    with pytest.raises(PermissionError, match="bound Slurm"):
        submitter.execute_preflight()


def test_preflight_reuses_data_phases_and_delegates_cpu_acceptance_reuse(submitter, monkeypatch):
    submitter.submit()
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    cpu.save_json(submitter.root / "INPUT_IDENTITY.json", dict(status="PASS"))
    cpu.save_json(submitter.root / "CPU_ACCEPTANCE.json", dict(status="PASS"))
    cpu.save_json(submitter.root / "REGISTRATION.json", dict(status="FROZEN_BEFORE_GPU"))
    for folder in ("blind", "rescore", "chartqa122"):
        (submitter.root / "prior_audit" / folder).mkdir(parents=True)
    seen = []
    monkeypatch.setattr(cpu, "verify_prepared_inputs", lambda root: dict(stage="prepare"))
    monkeypatch.setattr(cpu, "verify_phase", lambda directory, phase, code:
                        seen.append(phase) or dict(phase=phase))

    commands = []

    def final_cpu_stage(command, **kwargs):
        assert Path(command[1]).name == "cpu_preflight.py"
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cpu.subprocess, "run", final_cpu_stage)
    result = submitter.execute_preflight()
    assert result["status"] == "COMPLETE"
    assert seen == ["blind-classify", "rescore", "chartqa122"]
    assert len(commands) == 1


def test_cpu_job_lease_is_exclusive(tmp_path):
    with cpu.job_lease(tmp_path), pytest.raises(RuntimeError), cpu.job_lease(tmp_path):
        pass


def test_prepared_input_reuse_accepts_authenticated_hf_snapshot_blobs(submitter, monkeypatch):
    import sr_f2.protocol as protocol

    cache = submitter.root / "cache"
    model = cache / "snapshots" / "snapshot"
    blobs = cache / "blobs"
    model.mkdir(parents=True)
    blobs.mkdir()
    payload = blobs / "model-hash"
    payload.write_bytes(b"frozen weights")
    (model / "weights.bin").symlink_to("../../blobs/model-hash")
    original = Path(cpu.PARENT_ROOT) / "INPUT_IDENTITY.json"
    cpu.save_json(original, dict(status="PASS"))
    manifest = dict(view="frozen")
    cpu.save_json(submitter.root / "DIAGNOSTIC_MANIFEST.json", manifest)
    identity = dict(status="PASS", root=str(submitter.root), prior_root=cpu.PARENT_ROOT,
                    prior_identity_sha256=cpu.file_hash(original),
                    diagnostic_manifest_sha256=cpu.object_hash(manifest), model_path=str(model),
                    model_files=[dict(name="weights.bin", sha256=cpu.file_hash(payload),
                                      bytes=payload.stat().st_size)])
    cpu.save_json(submitter.root / "INPUT_IDENTITY.json", identity)
    monkeypatch.setattr(protocol, "verify_input_files", lambda root: identity)
    monkeypatch.setattr(protocol, "validate_config", lambda path: {})
    assert cpu.verify_prepared_inputs(submitter.root)["stage"] == "prepare"
    payload.write_bytes(b"modified weights")
    with pytest.raises(PermissionError, match="base-model bytes"):
        cpu.verify_prepared_inputs(submitter.root)


def test_controller_cannot_claim_delivery_from_one_benign_report_file(submitter):
    finish_preflight(submitter)
    controller = cpu.CpuSubmitter(submitter.root, "controller", submitter.python,
                                  submitter.manifest, slurm=submitter.slurm, user="alice")
    controller.submit()
    report = submitter.root / "released/REPORT.md"
    report.parent.mkdir()
    report.write_text("Not a complete evidence delivery")
    cpu.save_json(report.parent / "COMPLETE.json", dict(
        status="COMPLETE_COVERAGE_VERIFIED", experiment_complete=True,
        files=[dict(path="released/REPORT.md", sha256=cpu.file_hash(report),
                    bytes=report.stat().st_size)],
    ))
    submitter.slurm.finish("101")
    assert controller.submit()["failure"] == "ARTIFACT_VERIFICATION"
