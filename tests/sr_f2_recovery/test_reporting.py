"""Recovery-only gate fixtures; never load a model or contact a scheduler."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f2_recovery import reporting
from sr_f2_recovery.orchestration import POOL_SPECS
from sr_f12.orchestration import file_hash
from sr_f12.protocol import object_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def change(path, **updates):
    value = json.loads(path.read_text())
    value.update(updates)
    write(path, value)


class Slurm:
    def __init__(self):
        self.live, self.accounting = {}, {}

    def queue(self, user):
        assert user == "alice"
        return self.live

    def terminal(self, job):
        return self.accounting.get(job)


@pytest.fixture
def complete(tmp_path, monkeypatch):
    root = tmp_path / "original"
    root.mkdir()
    directory = root / "recovery/one"
    directory.mkdir(parents=True)
    extension = Path(reporting.__file__).resolve().parents[2]
    code = Path(reporting.frozen.__file__).resolve().parents[2]
    matrix = reporting.global_task_matrix()
    audit = {
        reporting.task_id(task): dict(
            task=task, status="INHERITED_COMPLETE" if i < 35 else "RESUME_ALLOWED"
        )
        for i, task in enumerate(matrix)
    }
    plan = dict(
        registration_sha256="registered",
        python=sys.executable,
        old_cancelled_jobs=[],
        recovery_id="one",
        audit_sha256="audit",
        extension_manifest={
            "src/sr_f2_recovery/reporting.py": "bound",
            "scripts/sr_f2/release_recovery.py": "bound",
        },
    )
    recovery = SimpleNamespace(
        root=root,
        directory=directory,
        code=code,
        extension=extension,
        path=directory / "PLAN.json",
        plan=plan,
        sha256="recovery-plan",
        allowed=audit,
        matrix=matrix,
        tasks={reporting.task_id(task): task for task in matrix},
        verify=lambda: True,
    )
    monkeypatch.setattr(reporting, "Recovery", lambda path: recovery)
    monkeypatch.setattr(
        reporting, "verify_accounting", lambda r, task: dict(task=task, checked=True)
    )

    def task_receipt(root, task, sha):
        assert sha == "registered"
        return dict(task, receipt_sha256="verified-" + reporting.task_id(task))

    monkeypatch.setattr(reporting, "verify_task", task_receipt)
    write(
        root / "orchestration/CONFIG.json",
        dict(
            root=str(root),
            code_root=str(code),
            user="alice",
            pools=reporting.pool_specs(),
            global_tasks=matrix,
            global_task_matrix_sha256=object_hash(matrix),
            strict_seed_waves=False,
        ),
    )
    slurm = Slurm()
    for i, spec in enumerate(reporting.pool_specs()):
        pool = root / "orchestration/pools" / spec["pool_id"]
        job = str(100 + i)
        old_plan = dict(spec, registration_sha256="registered")
        write(pool / "PLAN.json", old_plan)
        write(pool / "STATE.json", dict(spec, registration_sha256="registered", job_id=job))
        write(
            pool / "INTENT.json",
            dict(
                plan_sha256=object_hash(old_plan),
                registration_sha256="registered",
                job_name="old-" + job,
            ),
        )
        write(pool / "STARTED.json", dict(job_id=job, devices=["0", "1"]))
        identity = dict(
            job_id=job, user="alice", account="rose", qos=reporting.TEACHER_QOS, name="old-" + job
        )
        plan["old_cancelled_jobs"].append(identity)
        slurm.accounting[job] = dict(identity, state="CANCELLED", exit_code="0:0")
    cpu = root / "deployment/cpu_jobs/controller"
    write(cpu / "STATE.json", dict(job_id="104"))
    write(
        cpu / "INTENT.json",
        dict(
            kind="controller", user="alice", root=str(root), code_root=str(code), job_name="old-cpu"
        ),
    )
    identity = dict(
        job_id="104",
        user="alice",
        account="rose",
        qos="override-limits-but-killable",
        name="old-cpu",
    )
    plan["old_cancelled_jobs"].append(identity)
    slurm.accounting["104"] = dict(identity, state="CANCELLED", exit_code="0:0")
    for i, (pool_id, count) in enumerate(POOL_SPECS.items()):
        pool, job = directory / "pools" / pool_id, str(200 + i)
        pool.mkdir(parents=True)
        script = pool / "run.sbatch"
        script.write_text("fixture only")
        intent = dict(
            pool_id=pool_id,
            gpu_count=count,
            plan_sha256=recovery.sha256,
            user="alice",
            script=str(script),
            script_sha256=file_hash(script),
            job_name="new-" + job,
            comment="comment-" + job,
        )
        write(pool / "INTENT.json", intent)
        write(
            pool / "STATE.json",
            dict(
                pool_id=pool_id,
                gpu_count=count,
                job_id=job,
                plan_sha256=recovery.sha256,
                status="DRAINED",
            ),
        )
        write(
            pool / "STARTED.json",
            dict(job_id=job, plan_sha256=recovery.sha256, devices=[str(i) for i in range(count)]),
        )
        write(
            pool / "COMPLETE.json",
            dict(job_id=job, plan_sha256=recovery.sha256, pool_id=pool_id, status="DRAINED"),
        )
        fields = dict(
            Account="rose",
            Partition="cluster02",
            QOS=reporting.TEACHER_QOS,
            JobName=intent["job_name"],
            Comment=intent["comment"],
            Command=str(script),
            JobId=job,
            UserId="alice(123)",
            ReqTRES=f"cpu=8,mem={90 * count}G,gres/gpu={count},gres/gpu:pro6000={count}",
            Features="highmem",
            JobState="PENDING",
            Reason="JobHeldUser",
        )
        write(pool / "ALLOCATION_VERIFIED.json", dict(fields=fields))
        write(pool / "RELEASE_INTENT.json", dict(job_id=job))
        write(pool / "RELEASE.json", dict(returncode=0))
        slurm.accounting[job] = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user="alice",
            account="rose",
            qos=reporting.TEACHER_QOS,
            name=intent["job_name"],
        )
    for i, task in enumerate(matrix):
        key = reporting.task_id(task)
        if i < 35:
            pool_id = "GPU_POOL_" + str(i % 4 + 1)
            claim = dict(
                task=task,
                pool_id=pool_id,
                device="0",
                job_id=str(100 + i % 4),
                registration_sha256="registered",
                claim_id=key,
            )
            base = root / "orchestration/queue/tasks" / key
            worker = root / "orchestration/pools" / pool_id / "workers" / key
        else:
            index = (i - 35) % 3
            pool_id = list(POOL_SPECS)[index]
            claim = dict(
                task=task,
                pool_id=pool_id,
                device="0",
                job_id=str(200 + index),
                claim_id=key,
                plan_sha256=recovery.sha256,
                audit_entry_sha256=object_hash(audit[key]),
            )
            base = directory / "queue/tasks" / key
            worker = directory / "pools" / pool_id / "workers" / key
        write(base / "CLAIM.json", claim)
        write(
            base / "STATE.json",
            dict(
                claim,
                status="COMPLETE",
                claim_sha256=object_hash(claim),
                result=task_receipt(root, task, "registered"),
            ),
        )
        write(
            worker / "LAUNCH_INTENT.json",
            dict(
                task=task,
                device="0",
                claim_id=key,
                command=reporting._command(recovery, task),
                plan_sha256=recovery.sha256,
                audit_entry_sha256=object_hash(audit[key]),
            ),
        )
        write(worker / "PROCESS_EXIT.json", dict(returncode=0))
    return recovery, slurm


def test_gate_binds_cancelled_history_successful_recovery_and_all_53_tasks(complete):
    recovery, slurm = complete
    result = reporting.verify_recovery_release_gate(recovery.path, slurm=slurm)
    assert len(result["tasks"]) == 53
    assert len(result["recovery"]["inherited_tasks"]) == 35
    assert len(result["recovery"]["recovered_tasks"]) == 18
    assert result["recovery"]["max_run_gpus"] == 5
    assert {row["state"] for row in result["recovery"]["cancelled_allocations"]} == {"CANCELLED"}
    assert not (recovery.root / "released/AUTHORIZATION.json").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "failed",
        "unknown",
        "old_identity",
        "queued",
        "unregistered_live",
        "old_exit",
        "new_exit",
        "duplicate_worker",
        "missing_source",
        "blocked",
        "accounting_gap",
    ],
)
def test_gate_fails_closed_before_release(complete, monkeypatch, problem):
    recovery, slurm = complete
    key = reporting.task_id(recovery.matrix[35])
    task_dir = recovery.directory / "queue/tasks" / key
    if problem == "missing":
        (task_dir / "STATE.json").unlink()
    elif problem == "failed":
        change(task_dir / "STATE.json", status="FAILED")
    elif problem == "unknown":
        slurm.accounting.pop("200")
    elif problem == "old_identity":
        slurm.accounting["104"]["user"] = "bob"
    elif problem == "queued":
        slurm.live["200"] = dict(Command="fixture")
    elif problem == "unregistered_live":
        slurm.live["999"] = dict(Command=str(recovery.root / "unregistered.sbatch"))
    elif problem == "old_exit":
        key = reporting.task_id(recovery.matrix[0])
        change(
            recovery.root / "orchestration/pools/GPU_POOL_1/workers" / key / "PROCESS_EXIT.json",
            returncode=1,
        )
    elif problem == "new_exit":
        change(
            recovery.directory / "pools/RECOVERY_POOL_1/workers" / key / "PROCESS_EXIT.json",
            pool_aborted=True,
        )
    elif problem == "duplicate_worker":
        (recovery.directory / "pools/RECOVERY_POOL_1/workers/extra").mkdir()
    elif problem == "missing_source":
        recovery.plan["extension_manifest"].pop("src/sr_f2_recovery/reporting.py")
    elif problem == "blocked":
        recovery.allowed[key]["status"] = "BLOCKED_INCOMPLETE_GENERATION"
    else:

        def gap(*args):
            raise PermissionError("Unresolved accounting gap")

        monkeypatch.setattr(reporting, "verify_accounting", gap)
    with pytest.raises((PermissionError, RuntimeError, FileNotFoundError)):
        reporting.verify_recovery_release_gate(recovery.path, slurm=slurm)
    assert not (recovery.root / "released").exists()


@pytest.fixture
def accounted(tmp_path):
    task = dict(kind="endpoint", model_id="model")
    rows = [
        dict(group_seed=12, group_row_index=i, group_size=4, tokens=[3, 4, i]) for i in range(4)
    ]
    write(tmp_path / "sealed/evaluation/model/groups/group.json", dict(records=rows))
    base = dict(experiment_id=reporting.EXPERIMENT_ID, task_id="endpoint-model")
    entries = [dict(base, kind="completion_attempts", count=4, metadata=dict(group_seed=12))]
    entries += [
        dict(base, kind="generated_tokens", count=3, metadata=dict(group_seed=12, row_index=i))
        for i in range(4)
    ]
    ledger = tmp_path / "accounting/endpoint-model.jsonl"
    ledger.parent.mkdir()
    ledger.write_text("".join(json.dumps(row) + "\n" for row in entries))
    return SimpleNamespace(root=tmp_path), task, ledger, entries


def test_accounting_matches_complete_raw_without_scoring(accounted):
    recovery, task, _, _ = accounted
    result = reporting.verify_accounting(recovery, task)
    assert result["groups"] == 1 and result["records"] == 4
    assert result["unresolved_groups"] == result["repeated_attempts"] == 0


@pytest.mark.parametrize(
    "problem", ["reserved_without_raw", "duplicate_attempt", "missing_token", "wrong_identity"]
)
def test_accounting_unresolved_attempts_cannot_be_declared_complete(accounted, problem):
    recovery, task, ledger, entries = accounted
    if problem == "reserved_without_raw":
        entries.append(dict(entries[0], metadata=dict(group_seed=13)))
    elif problem == "duplicate_attempt":
        entries.append(entries[0])
    elif problem == "missing_token":
        entries.pop()
    else:
        entries[0]["task_id"] = "different-model"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in entries))
    with pytest.raises(PermissionError, match=r"Accounting|accounting|repeated"):
        reporting.verify_accounting(recovery, task)


def test_reporting_cannot_score_before_explicit_gate(complete, monkeypatch):
    recovery, _ = complete
    monkeypatch.setattr(
        reporting.frozen,
        "cohort_verification",
        lambda root: dict(
            status="PASS", raw_evidence_status="RAW_AND_GOLD_REVERIFIED", receipt_sha256="cohort"
        ),
    )
    calls = []
    monkeypatch.setattr(
        reporting.frozen, "score_new_endpoints", lambda *args: calls.append("scored")
    )

    def blocked(*args, **kwargs):
        raise PermissionError("Unresolved generation gap")

    monkeypatch.setattr(reporting, "verify_recovery_release_gate", blocked)
    with pytest.raises(PermissionError, match="Unresolved"):
        reporting.release_and_report(recovery.path)
    assert calls == []
    assert not (recovery.root / "released/AUTHORIZATION.json").exists()
