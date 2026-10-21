"""Delivery recovery must never create scientific work or alter failed receipts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/lowmem_delivery/recover_delivery.py"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.job = "216001"
        self.state = "COMPLETED"
        self.exit = "0:0"
        self.node = "cpu-1"
        self.builder_rc = 0
        self.mutate_receipt = lambda x: None
        self.drop_archive = None
        self.archive_byte_change = False
        self.oom = False

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        op = self.op
        if command[:3] == ["scontrol", "show", "node"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"NodeName={self.node} Arch={getattr(self, 'node_arch', 'x86_64')} "
                    "ActiveFeatures=cpu_ok"
                ),
                stderr="",
            )
        if command[:3] == ["scontrol", "show", "job"]:
            f = dict(
                JobId=self.job,
                JobState="RUNNING",
                NodeList=self.node,
                ReqNodeList="(null)",
                Features="cpu_ok",
                QOS="soujanya-poria-startfund-2026-03",
                Account="rose",
                UserId="varun024(1001)",
                JobName="delivery-recovery-LH",
                Command=str(op.script),
                TimeLimit="3-00:00:00",
                Requeue="0",
                NumCPUs="4",
                NumNodes="1",
                Partition="cluster02",
                Comment="approved-delivery-recovery",
                AllocTRES="cpu=4,mem=12G,node=1,billing=4",
            )
            f.update(getattr(self, "live_overrides", {}))
            return SimpleNamespace(returncode=0, stdout=" ".join(k + "=" + v for k, v in f.items()))
        if command[:3] == ["scontrol", "write", "batch_script"]:
            return SimpleNamespace(
                returncode=0, stdout=op.script.read_text() + getattr(self, "script_suffix", "")
            )
        if command[0] == "sacct" and command[command.index("-j") + 1] == "216128":
            return SimpleNamespace(
                returncode=0,
                stdout=getattr(self, "superseded_terminal", "216128|CANCELLED|0:0|None|0||None\n"),
            )
        if command[0] == "sacct" and command[command.index("-j") + 1] == "211333":
            return SimpleNamespace(
                returncode=0,
                stdout=getattr(
                    self,
                    "old_terminal",
                    "211333|OUT_OF_MEMORY|0:125|cpu-2|cpu=4,mem=12G,node=1|10000|\n",
                ),
            )
        if command[0] == "sacct":
            return SimpleNamespace(
                returncode=0,
                stdout="|".join(
                    [
                        self.job,
                        self.state,
                        self.exit,
                        self.node,
                        "cpu=4,mem=12G,node=1,billing=4",
                        "varun024",
                        "rose",
                        "soujanya-poria-startfund-2026-03",
                        "delivery-recovery-LH",
                        "100",
                        "3-00:00:00",
                    ]
                )
                + "|\n",
            )
        assert command == op.command
        assert "capture_output" not in kwargs
        kwargs["stdout"].write(b"builder progress\n")
        if self.oom:
            raise OSError("simulated process launch failure")
        if self.builder_rc == 0:
            op.archive.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(op.archive, "w") as package:
                for path in sorted(op.root.rglob("*")):
                    if (
                        path.is_file()
                        and not path.is_relative_to(op.root / "delivery")
                        and path.name != "RECOVERY.lock"
                    ):
                        name = "evidence/" + str(path.relative_to(op.root))
                        if name != self.drop_archive:
                            package.writestr(
                                name,
                                b"changed"
                                if self.archive_byte_change and path == op.source
                                else path.read_bytes(),
                            )
            with zipfile.ZipFile(op.archive) as package:
                count = len(package.namelist())
            receipt = copy.deepcopy(op.auth["expected_builder_identity"])
            receipt.update(
                checks=dict.fromkeys(op_module.CHECKS, "PASS"),
                coverage=op_module.COVERAGE.copy(),
                archive=dict(
                    path=str(op.archive),
                    sha256=digest(op.archive),
                    bytes=op.archive.stat().st_size,
                    entries=count,
                ),
            )
            self.mutate_receipt(receipt)
            save(op.builder_path, receipt)
            profile = dict(
                status="REAL_BUILD_RETURNED",
                exitcode=0,
                science_execution_requested=False,
                output=str(op.archive),
                builder=str(op.source.parent / "build_delivery.py"),
                builder_sha256=digest(op.source.parent / "build_delivery.py"),
                profile_source_sha256=digest(op.source.parent / "profile_delivery_memory.py"),
                result_sha256=hashlib.sha256(
                    json.dumps(
                        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    ).encode()
                ).hexdigest(),
                peak_rss_bytes=100000000,
                measurement_scope="builder child only; not Slurm/cgroup total or resource minimum",
            )
            save(op.profile_path, profile)
        return SimpleNamespace(returncode=self.builder_rc)


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    global op_module
    root = tmp_path / "selected_v2"
    root.mkdir()
    deploy = root / "operations" / "new-code"
    deploy.mkdir(parents=True)
    source = deploy / "recover_delivery.py"
    shutil.copyfile(SOURCE, source)
    spec = importlib.util.spec_from_file_location("delivery_recovery_test_module", source)
    op_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op_module)
    op_module.TARGET_ROOT = str(root)
    m = op_module
    for name in m.REQUIRED_FILES:
        save(root / name, {})
    old = root / "operations" / m.OLD_OPERATION
    extension = root / "native/observability/frozen/SOURCE_MANIFEST.json"
    save(
        extension,
        dict(
            extension_commit="frozen",
            base_code_commit="science",
            protocol_source_guard={"protocol_revision": "v2"},
        ),
    )
    for n in ["observe_runtime.py", "AMENDMENT.json", "CPU_VALIDATION.json"]:
        save(extension.parent / n, {})
    save(root / "deployment/SOURCE_MANIFEST.json", {"files": []})
    identity = dict(
        python=sys.executable,
        extension_manifest_path=str(extension),
        extension_manifest_sha256=digest(extension),
        run_identity_sha256=digest(root / "RUN_IDENTITY.json"),
        source_manifest_sha256=digest(root / "deployment/SOURCE_MANIFEST.json"),
        task_pool={"registered": True},
    )
    save(old / "OPERATION_IDENTITY.json", identity)
    save(root / m.FAILED_ATTEMPT / "END.json", dict(status="FAILED"))
    save(root / m.FAILED_ATTEMPT / "DELIVERY_PROCESS.json", dict(returncode=-9))
    command = [
        sys.executable,
        str(old / "build_delivery.py"),
        "--root",
        str(root),
        "--output",
        str(root / "delivery/SR_F3_CE_20261016.zip"),
        "--helper-manifest",
        str(old / "OPERATION_SOURCES.json"),
        "--extension-manifest",
        str(extension),
        "--operation-identity",
        str(old / "OPERATION_IDENTITY.json"),
    ]
    save(
        root / m.FAILED_ATTEMPT / "DELIVERY_INTENT.json",
        dict(status="NUMERICAL_RELEASE_COMPLETE", command=command),
    )
    save(
        root / "tasks/release/COMPLETE.json",
        dict(
            status="COMPLETE",
            job_id="215520",
            run_identity_sha256=digest(root / "RUN_IDENTITY.json"),
            execution_freeze_sha256=digest(root / "EXECUTION_FREEZE.json"),
        ),
    )
    save(
        root / f"accounting/observability/{m.RELEASE_KEY}/release/FINAL.json",
        dict(status="COMPLETE", task_receipt_sha256=digest(root / "tasks/release/COMPLETE.json")),
    )
    journal = root / "operations/delivery_recovery/20261021_delivery_lowmem"
    journal.mkdir(parents=True)
    script = journal / "run.sbatch"
    script.write_text("#!/bin/bash\nexec authorized recovery\n")
    (root / "operations/delivery_recovery/20261021_delivery_oom_cpu22").mkdir(
        parents=True, exist_ok=True
    )
    (root / "operations/delivery_recovery/20261021_delivery_teacher").mkdir(
        parents=True, exist_ok=True
    )
    history = deploy / "lowmem_history.py"
    history.write_text("def verify_history(root, auth, *, require_retired=False):\n    return []\n")
    # Builder revision authentication itself is covered by test_delivery_revision;
    # this fixture isolates worker binding/lifecycle from expensive scientific execution.
    new_builder = deploy / "build_delivery.py"
    new_builder.write_text(
        "from types import SimpleNamespace\nimport json\n"
        "def load_delivery_revision(root, path, **kwargs):\n"
        "    return SimpleNamespace(identity=json.loads(path.read_text())['fixture_identity'])\n"
    )
    profiler = deploy / "profile_delivery_memory.py"
    profiler.write_text("# profile execution is isolated by FakeRunner\n")
    revision_identity = {"fixture_only": True, "builder_sha256": digest(new_builder)}
    revision_manifest = deploy / "DELIVERY_REVISION.json"
    save(revision_manifest, {"fixture_identity": revision_identity})
    extras = {
        str(path.relative_to(root)): digest(path) for path in (history, new_builder, profiler)
    }
    for n in ("submit_delivery_recovery.py", "verify_recovered_migration.py"):
        p = deploy / n
        p.write_text("# retained auxiliary source\n")
        if n == "submit_delivery_recovery.py":
            shutil.copyfile(SOURCE.parent / n, p)
        extras[str(p.relative_to(root))] = digest(p)
    files = set(m.REQUIRED_FILES) | {
        str(p.relative_to(root))
        for p in [
            extension,
            extension.parent / "observe_runtime.py",
            extension.parent / "AMENDMENT.json",
            extension.parent / "CPU_VALIDATION.json",
            root / "deployment/SOURCE_MANIFEST.json",
        ]
    }
    auth = dict(
        schema_version=1,
        status="AUTHORIZED_DELIVERY_ONLY_RECOVERY",
        root=str(root),
        failed_job_id="211333",
        failed_attempt=m.FAILED_ATTEMPT,
        allowed_nodes=["cpu-1", "cpu-2"],
        qos="soujanya-poria-startfund-2026-03",
        account="rose",
        cpus=4,
        memory_mib=12288,
        time_limit="3-00:00:00",
        user="varun024",
        job_name="delivery-recovery-LH",
        comment="approved-delivery-recovery",
        scope="DELIVERY_PACKAGING_AND_VERIFICATION_ONLY",
        recovery_directory=str(journal.relative_to(root)),
        script=str(script.relative_to(root)),
        script_sha256=digest(script),
        source_sha256=digest(source),
        extra_sources=extras,
        delivery_revision_manifest=dict(
            path=str(revision_manifest.relative_to(root)), sha256=digest(revision_manifest)
        ),
        profile_receipt=str((journal / "MEMORY_PROFILE.json").relative_to(root)),
        files={n: digest(root / n) for n in files},
    )
    auth["expected_builder_identity"] = dict(
        schema_version=1,
        status="VERIFIED_DELIVERY",
        run_identity_sha256=identity["run_identity_sha256"],
        scientific_source_manifest_sha256=identity["source_manifest_sha256"],
        helper_manifest_sha256=digest(old / "OPERATION_SOURCES.json"),
        observability_extension=dict(
            manifest_path=str(extension.relative_to(root)),
            manifest_sha256=digest(extension),
            extension_commit="frozen",
            base_code_commit="science",
            protocol_source_guard={"protocol_revision": "v2"},
            observer_source_sha256=digest(extension.parent / "observe_runtime.py"),
            amendment_sha256=digest(extension.parent / "AMENDMENT.json"),
            cpu_validation_sha256=digest(extension.parent / "CPU_VALIDATION.json"),
        ),
        delivery_revision=revision_identity,
        task_pool_operation=dict(
            operation_identity_path=str((old / "OPERATION_IDENTITY.json").relative_to(root)),
            operation_identity_sha256=digest(old / "OPERATION_IDENTITY.json"),
            operation_commit=m.OLD_OPERATION,
            task_pool=identity["task_pool"],
        ),
    )
    prior_deploy = root / "operations" / m.SUPERSEDED_OPERATION
    prior_deploy.mkdir()
    prior_journal = root / m.SUPERSEDED_DIRECTORY
    prior_journal.mkdir()
    prior_source = prior_deploy / "recover_delivery.py"
    prior_source.write_text("# preserved failed operation83 source\n")
    prior_script = prior_journal / "run.sbatch"
    prior_script.write_text("# original 4CPU64GiB request\n")
    prior_authorization = prior_deploy / "RECOVERY_AUTHORIZATION.json"
    save(
        prior_authorization,
        dict(
            root=str(root),
            recovery_directory=m.SUPERSEDED_DIRECTORY,
            source_sha256=digest(prior_source),
            script_sha256=digest(prior_script),
            cpus=4,
            memory_mib=65536,
        ),
    )
    prior_submitted = prior_journal / "SUBMITTED.json"
    save(
        prior_submitted,
        dict(
            status="SUBMITTED",
            job_id="216128",
            authorization_sha256=digest(prior_authorization),
            script_sha256=digest(prior_script),
        ),
    )
    cancel_receipt = prior_journal / "CANCELLED.json"
    save(cancel_receipt, dict(argv=["scancel", "--state=PENDING", "216128"], returncode=0))
    supersession = dict(
        status="CANCELLED_UNSTARTED_RECOVERY",
        job_id="216128",
        prior_recovery_directory=m.SUPERSEDED_DIRECTORY,
        terminal_accounting=dict(
            JobIDRaw="216128",
            State="CANCELLED",
            ExitCode="0:0",
            Start="None",
            ElapsedRaw="0",
            AllocTRES="",
            NodeList="None",
        ),
        command_receipts={str(cancel_receipt.relative_to(root)): digest(cancel_receipt)},
    )
    for name, path in [
        ("prior_authorization", prior_authorization),
        ("prior_source", prior_source),
        ("prior_script", prior_script),
        ("prior_submitted", prior_submitted),
    ]:
        supersession[name + "_path"] = str(path.relative_to(root))
        supersession[name + "_sha256"] = digest(path)
    supersession_path = prior_journal / "SUPERSESSION.json"
    save(supersession_path, supersession)
    auth["supersession"] = dict(
        path=str(supersession_path.relative_to(root)), sha256=digest(supersession_path)
    )
    authorization = deploy / "RECOVERY_AUTHORIZATION.json"
    save(authorization, auth)
    submit_path = deploy / "submit_delivery_recovery.py"
    submit_spec = importlib.util.spec_from_file_location("recovery_fixture_submit", submit_path)
    submit_module = importlib.util.module_from_spec(submit_spec)
    submit_spec.loader.exec_module(submit_module)
    submitted = False

    def submit_runner(argv, **kwargs):
        nonlocal submitted
        if argv[0] == "sacctmgr" and "qos" in argv:
            text = (
                "Name|GrpSubmit|MaxSubmitPU|GrpJobs|MaxJobsPU|Flags|\n"
                "soujanya-poria-startfund-2026-03||5||||\n"
            )
        elif argv[0] == "sacctmgr":
            text = (
                "Cluster|Account|User|Partition|QOS|GrpSubmit|MaxSubmit|GrpJobs|MaxJobs|\n"
                "x|rose|varun024||soujanya-poria-startfund-2026-03|||||\n"
            )
        elif argv[0] == "squeue":
            text = (
                "216001|varun024|rose|soujanya-poria-startfund-2026-03|cluster02|PENDING\n"
                if submitted
                else ""
            )
        elif argv[:3] == ["scontrol", "show", "node"]:
            text = (
                "NodeName=cpu-1 Arch=x86_64 Partitions=cluster02 State=MIXED ActiveFeatures=cpu_ok "
                "CPUEfctv=96 CPUAlloc=42 RealMemory=339968 AllocMem=129024 MemSpecLimit=5120\n"
                "NodeName=cpu-2 Arch=x86_64 Partitions=cluster02 State=MIXED ActiveFeatures=cpu_ok "
                "CPUEfctv=96 CPUAlloc=42 RealMemory=339968 AllocMem=129024 MemSpecLimit=5120"
            )
        elif argv[0] == "sbatch":
            submitted = True
            text = "216001\n"
        elif argv[:3] == ["scontrol", "show", "job"]:
            text = (
                "JobId=216001 JobName=delivery-recovery-LH "
                "Account=rose QOS=soujanya-poria-startfund-2026-03 Partition=cluster02 "
                f"Command={script} Comment=approved-delivery-recovery "
                "NumCPUs=4 NumTasks=1 CPUs/Task=4 "
                "ReqNodeList=(null) Features=cpu_ok TimeLimit=3-00:00:00 "
                "Requeue=0 JobState=PENDING "
                "Reason=JobHeldUser RunTime=00:00:00 UserId=varun024(1001) NumNodes=1 "
                "NodeList=(null) ReqTRES=cpu=4,mem=12G,node=1"
            )
        elif argv[:3] == ["scontrol", "write", "batch_script"]:
            text = script.read_text()
        elif argv[0] == "sacct":
            text = "216001|PENDING|Unknown|0|\n"
        elif argv[:2] == ["scontrol", "release"]:
            text = ""
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    submission = submit_module.Submission(
        root, authorization, runner=submit_runner, validator=lambda *_: None
    )
    submission.submit()
    submission.continue_job()
    runner = FakeRunner()
    op = m.Recovery(root, authorization, runner=runner)
    runner.op = op
    monkeypatch.setenv("SLURM_JOB_ID", "216001")
    return op, runner, m


def test_success_and_idempotent_final(recovery):
    op, runner, _ = recovery
    originals = {n: digest(op.root / n) for n in op.auth["files"]}
    assert op.run()["status"] == "DONE"
    result = op.verify_final()
    assert result["status"] == "FINAL_RECOVERY_VERIFIED"
    assert op.verify_final() == result
    assert originals == {n: digest(op.root / n) for n in originals}
    assert sum(c == op.command for c in runner.calls) == 1
    assert not any(c[0] in {"sbatch", "scancel", "srun", "scontrol update"} for c in runner.calls)
    assert "migration verification still required" in result["acceptance"]


@pytest.mark.parametrize(
    "override",
    [
        {"NodeList": "gpu-a40-1"},
        {"ReqNodeList": "cpu-2"},
        {"QOS": "other"},
        {"Account": "other"},
        {"UserId": "other(1)"},
        {"JobState": "PENDING"},
        {"JobId": "211333"},
        {"TimeLimit": "1-00:00:00"},
        {"Requeue": "1"},
        {"NumCPUs": "8"},
        {"AllocTRES": "cpu=22,mem=66G,node=1"},
        {"AllocTRES": "cpu=4,mem=12G,node=1,gres/gpu=1"},
        {"Partition": "gpu"},
        {"Comment": "other"},
    ],
)
def test_wrong_allocation_never_executes_builder(recovery, override):
    op, r, _ = recovery
    r.live_overrides = override
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in r.calls


def test_wrong_script_never_executes_builder(recovery):
    op, r, _ = recovery
    r.script_suffix = "changed"
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in r.calls


def test_wrong_environment(recovery, monkeypatch):
    op, r, _ = recovery
    monkeypatch.setenv("SLURM_JOB_ID", "215520")
    with pytest.raises(PermissionError):
        op.run()
    assert r.calls == []


@pytest.mark.parametrize("filename", ["STARTED.json", "PROCESS.json", "END.json"])
def test_unknown_or_prior_attempt_is_not_retried(recovery, filename):
    op, r, _ = recovery
    save(op.directory / filename, {})
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in r.calls


@pytest.mark.parametrize("code", [-9, 1])
def test_builder_failure_preserved_no_retry(recovery, code):
    op, r, m = recovery
    r.builder_rc = code
    with pytest.raises(PermissionError):
        op.run()
    assert m.read(op.directory / "PROCESS.json")["returncode"] == code
    assert m.read(op.directory / "END.json")["status"] == "FAILED"
    with pytest.raises(PermissionError):
        op.run()
    with pytest.raises(PermissionError):
        op.verify_final()


def test_launch_exception_leaves_failed_record(recovery):
    op, r, m = recovery
    r.oom = True
    with pytest.raises(OSError):
        op.run()
    assert m.read(op.directory / "END.json")["status"] == "FAILED"
    with pytest.raises(PermissionError):
        op.run()


@pytest.mark.parametrize(
    "key",
    sorted(
        [
            "crc",
            "safe_unique_names",
            "payload_hashes",
            "manifest_hashes",
            "entry_links",
            "gzip_read_to_eof",
            "raw_rescoring_chain",
        ]
    ),
)
def test_incomplete_builder_check_rejected(recovery, key):
    op, r, m = recovery
    r.mutate_receipt = lambda x: x["checks"].update({key: "FAIL"})
    with pytest.raises(PermissionError):
        op.run()
    assert m.read(op.directory / "END.json")["status"] == "FAILED"


def test_incomplete_coverage_rejected(recovery):
    op, r, _ = recovery
    r.mutate_receipt = lambda x: x["coverage"].update(records=233339)
    with pytest.raises(PermissionError):
        op.run()


@pytest.mark.parametrize("field,value", [("state", "FAILED"), ("exit", "1:0"), ("node", "cpu-2")])
def test_terminal_accounting_gate(recovery, field, value):
    op, r, _ = recovery
    op.run()
    setattr(r, field, value)
    with pytest.raises(PermissionError):
        op.verify_final()


def test_missing_archive_evidence_rejected(recovery):
    op, r, _ = recovery
    r.drop_archive = "evidence/" + str(op.source.relative_to(op.root))
    op.run()
    with pytest.raises(PermissionError, match="absent"):
        op.verify_final()


def test_changed_archived_evidence_rejected(recovery):
    op, r, _ = recovery
    r.archive_byte_change = True
    op.run()
    with pytest.raises(PermissionError, match="Archived"):
        op.verify_final()


@pytest.mark.parametrize(
    "name", ["EXECUTION_FREEZE.json", "RUN_IDENTITY.json", "tasks/release/COMPLETE.json"]
)
def test_original_change_rejected(recovery, name):
    op, r, _ = recovery
    (op.root / name).write_text("{}changed")
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in r.calls


def test_authorized_wrong_builder_identity_rejected(recovery):
    op, r, m = recovery
    a = copy.deepcopy(op.auth)
    a["expected_builder_identity"]["scientific_source_manifest_sha256"] = "0" * 64
    save(op.authorization, a)
    with pytest.raises(PermissionError, match="builder identity"):
        m.Recovery(op.root, op.authorization, runner=r, require_submission=False)


def test_arbitrary_command_even_rehashed_rejected(recovery):
    op, r, m = recovery
    p = op.root / m.FAILED_ATTEMPT / "DELIVERY_INTENT.json"
    save(p, {"command": ["python", "train.py"]})
    a = copy.deepcopy(op.auth)
    a["files"][str(p.relative_to(op.root))] = digest(p)
    save(op.authorization, a)
    with pytest.raises(PermissionError, match="original delivery builder"):
        m.Recovery(op.root, op.authorization, runner=r, require_submission=False)


def test_auth_before_submission_supported(recovery):
    op, r, m = recovery
    for n in ["SUBMITTED.json", "ALLOCATION_VERIFIED.json", "RELEASE.json"]:
        (op.directory / n).unlink()
    prepared = m.Recovery(op.root, op.authorization, runner=r, require_submission=False)
    assert prepared.validate() == op.auth


def test_modified_final_receipt_rejected(recovery):
    op, _, _ = recovery
    op.run()
    result = op.verify_final()
    result["job_id"] = "other"
    save(op.root / "delivery/FINAL_RECOVERY_VERIFIED.json", result)
    with pytest.raises(PermissionError, match="Existing final"):
        op.verify_final()


def test_lock_contention_prevents_run(recovery):
    op, r, _ = recovery
    with op.lock(), pytest.raises(BlockingIOError):
        op.run()
    assert op.command not in r.calls


@pytest.mark.parametrize("filename", ["SUBMITTED.json", "ALLOCATION_VERIFIED.json", "RELEASE.json"])
def test_registration_receipt_rebinding_rejected(recovery, filename):
    op, runner, m = recovery
    path = op.directory / filename
    record = m.read(path)
    record["authorization_sha256"] = "0" * 64
    save(path, record)
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in runner.calls


@pytest.mark.parametrize("kind", ["source", "script", "extra_source", "authorization"])
def test_prelaunch_dependency_change_rejected(recovery, kind):
    op, runner, _ = recovery
    path = {
        "source": op.source,
        "script": op.script,
        "authorization": op.authorization,
        "extra_source": op.root / next(iter(op.auth["extra_sources"])),
    }[kind]
    path.write_text(path.read_text() + " ")
    if kind == "authorization":
        changed = copy.deepcopy(op.auth)
        changed["scope"] = "CHANGED"
        save(path, changed)
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in runner.calls


def test_original_success_cannot_be_recovered(recovery):
    op, runner, m = recovery
    save(op.root / m.FAILED_ATTEMPT / "DELIVERY_VERIFIED.json", {})
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in runner.calls


def test_existing_archive_not_overwritten(recovery):
    op, runner, _ = recovery
    op.archive.parent.mkdir(exist_ok=True)
    op.archive.write_bytes(b"prior partial archive retained")
    with pytest.raises(PermissionError):
        op.run()
    assert op.archive.read_bytes() == b"prior partial archive retained"
    assert op.command not in runner.calls


@pytest.mark.parametrize(
    "old_terminal",
    [
        "211333|RUNNING|0:0|cpu-2|cpu=4,mem=12G,node=1|10000|\n",
        "211333|OUT_OF_MEMORY|1:0|cpu-2|cpu=4,mem=12G,node=1|10000|\n",
        "211333|OUT_OF_MEMORY|0:125|cpu-1|cpu=4,mem=12G,node=1|10000|\n",
        "211333|OUT_OF_MEMORY|0:125|cpu-2|cpu=4,mem=64G,node=1|10000|\n",
    ],
)
def test_wrong_old_failure_terminal_never_runs_builder(recovery, old_terminal):
    op, runner, _ = recovery
    runner.old_terminal = old_terminal
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in runner.calls


def test_old_failure_terminal_rechecked_at_final(recovery):
    op, runner, _ = recovery
    op.run()
    runner.old_terminal = "211333|FAILED|1:0|cpu-2|cpu=4,mem=12G,node=1|10000|\n"
    with pytest.raises(PermissionError):
        op.verify_final()


@pytest.mark.parametrize(
    "field,value",
    [
        ("State", "RUNNING"),
        ("ExitCode", "1:0"),
        ("Start", "2026-10-21T00:00:00"),
        ("ElapsedRaw", "1"),
        ("AllocTRES", "cpu=4,mem=12G,node=1"),
        ("NodeList", "cpu-1"),
    ],
)
def test_supersession_terminal_requires_zero_execution(recovery, field, value):
    op, runner, m = recovery
    proof_path = op.root / op.auth["supersession"]["path"]
    proof = m.read(proof_path)
    proof["terminal_accounting"][field] = value
    save(proof_path, proof)
    op.auth["supersession"]["sha256"] = digest(proof_path)
    save(op.authorization, op.auth)
    with pytest.raises(PermissionError):
        op.validate()
    assert op.command not in runner.calls


@pytest.mark.parametrize("receipt", ["STARTED.json", "RELEASE.json", "END.json", "PROCESS.json"])
def test_started_superseded_attempt_cannot_be_replaced(recovery, receipt):
    op, _, m = recovery
    save(op.root / m.SUPERSEDED_DIRECTORY / receipt, {})
    with pytest.raises(PermissionError, match="launch, release"):
        op.validate()


def test_live_supersession_is_rechecked_before_builder(recovery):
    op, runner, _ = recovery
    runner.superseded_terminal = "216128|RUNNING|0:0|None|0||None\n"
    with pytest.raises(PermissionError):
        op.run()
    assert op.command not in runner.calls


def test_live_supersession_is_rechecked_at_final(recovery):
    op, runner, _ = recovery
    op.run()
    runner.superseded_terminal = "216128|CANCELLED|0:0|None|0|cpu=4|cpu-1\n"
    with pytest.raises(PermissionError):
        op.verify_final()


def test_supersession_sources_are_archived(recovery):
    op, runner, _ = recovery
    runner.drop_archive = "evidence/" + op.auth["supersession"]["path"]
    op.run()
    with pytest.raises(PermissionError, match="absent"):
        op.verify_final()


@pytest.mark.parametrize(
    "argv,code",
    [
        (["scancel", "--state=PENDING", "215520"], 0),
        (["scancel", "216128"], 0),
        (["scancel", "--state=PENDING", "216128"], 1),
        (["scontrol", "requeue", "216128"], 0),
        (["scontrol", "show", "job", "216128"], 0),
    ],
)
def test_supersession_requires_exact_successful_cancellation(recovery, argv, code):
    op, _, m = recovery
    proof_path = op.root / op.auth["supersession"]["path"]
    proof = m.read(proof_path)
    name = next(iter(proof["command_receipts"]))
    path = op.root / name
    save(path, dict(argv=argv, returncode=code))
    proof["command_receipts"][name] = digest(path)
    save(proof_path, proof)
    op.auth["supersession"]["sha256"] = digest(proof_path)
    save(op.authorization, op.auth)
    with pytest.raises(PermissionError):
        op.validate()


def test_duplicate_cancellation_in_supersession_rejected(recovery):
    op, _, m = recovery
    proof_path = op.root / op.auth["supersession"]["path"]
    proof = m.read(proof_path)
    duplicate = proof_path.parent / "DUPLICATE_CANCEL.json"
    save(duplicate, dict(argv=["scancel", "--state=PENDING", "216128"], returncode=0))
    proof["command_receipts"][str(duplicate.relative_to(op.root))] = digest(duplicate)
    save(proof_path, proof)
    op.auth["supersession"]["sha256"] = digest(proof_path)
    save(op.authorization, op.auth)
    with pytest.raises(PermissionError):
        op.validate()


def test_superseded_none_assigned_node_sentinel_retained(recovery):
    op, runner, m = recovery
    row = m.read(op.root / op.auth["supersession"]["path"])["terminal_accounting"]
    row["NodeList"] = "None assigned"
    m.Recovery.check_superseded_accounting(row)
    runner.superseded_terminal = "216128|CANCELLED|0:0|Unknown|0||None assigned\n"
    assert op.superseded_terminal()["accounting"]["NodeList"] == "None assigned"


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "SYNTHETIC_PROFILE_ONLY"),
        ("exitcode", 1),
        ("science_execution_requested", True),
        ("output", "/other.zip"),
        ("builder_sha256", "0" * 64),
        ("profile_source_sha256", "0" * 64),
        ("result_sha256", "0" * 64),
        ("peak_rss_bytes", 0),
        ("peak_rss_bytes", True),
        ("measurement_scope", "minimum resources"),
    ],
)
def test_real_memory_profile_must_bind_same_successful_delivery(recovery, field, value):
    op, _, _ = recovery
    op.run()
    profile = json.loads(op.profile_path.read_text())
    profile[field] = value
    save(op.profile_path, profile)
    with pytest.raises(PermissionError, match="memory measurement"):
        op.verify_final()


def test_original_builder_is_retained_but_revised_builder_is_the_actual_command(recovery):
    op, _runner, _ = recovery
    assert op.command[1:5] == [
        "-B",
        str(op.source.parent / "profile_delivery_memory.py"),
        "--mode",
        "real-build",
    ]
    assert "--delivery-revision-manifest" in op.command
    assert (
        str(op.root / "operations" / op_module.OLD_OPERATION / "build_delivery.py")
        not in op.command
    )
    op.run()
    receipt = op.verify_final()
    assert receipt["memory_profile_sha256"] == digest(op.profile_path)
    assert receipt["delivery_revision"] == op.auth["expected_builder_identity"]["delivery_revision"]
    assert "not cgroup total" in receipt["resource_measurement_scope"]


def test_second_cpu_node_with_same_resources_is_valid(recovery):
    op, runner, _ = recovery
    runner.node = "cpu-2"
    assert op.run()["status"] == "DONE"
    assert op.verify_final()["accounting"]["NodeList"] == "cpu-2"


def test_arm_node_cannot_run_x86_builder(recovery):
    op, runner, _ = recovery
    runner.node_arch = "aarch64"
    with pytest.raises(PermissionError, match="architecture or feature"):
        op.run()
    assert op.command not in runner.calls


def test_unknown_cgroup_is_not_reported_as_measured(recovery, monkeypatch):
    op, _, _ = recovery

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("fixture cgroup unavailable")

    monkeypatch.setattr(Path, "read_text", unavailable)
    result = op.own_cgroup_memory()
    assert result["status"] == "UNKNOWN"
    assert "peak_bytes" not in result
