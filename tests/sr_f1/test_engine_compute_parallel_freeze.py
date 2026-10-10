"""Compute source revision keeps old identities and seals interrupted evidence."""

import copy
import json

import pytest

from mm_core.execution import object_hash
from sr_f1 import freeze
from sr_f1.contract import file_hash

PREFIX = freeze.COMPUTE_PARALLEL_PREFIX
TRACK = "engineering/engine/natural/continuous"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source(files, commit):
    return dict(
        source_commit=commit * 40,
        source_tree_sha256=object_hash(files),
        source_file_hashes=files,
        source_dirty_files=[],
    )


@pytest.fixture
def compute_case(tmp_path, monkeypatch):
    root = tmp_path
    saved = root / "code_before_engine_compute_parallel_20261010"
    before_files = {}
    for name in [
        "scripts/sr_f1/run_worker.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/training.py",
        "src/sr_f1/contract.py",
    ]:
        path = saved / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old " + name)
        before_files[name] = file_hash(path)
    before = source(before_files, "a")
    write(saved / "SOURCE_DEPLOYMENT.json", before)
    write(root / "EXECUTION_FREEZE.json", {"status": "FROZEN"})
    write(root / "ENGINE_MULTIGPU_REPAIR.json", {"immutable": True})
    parent = dict(
        repair_sha256=file_hash(root / "ENGINE_MULTIGPU_REPAIR.json"),
        source_commit=before["source_commit"],
        activation_spill_directory="/hdd/private",
        quota_root="/hdd",
        quota_reserve_bytes=10 * 1024**3,
    )

    def original(root, actual_source=None):
        assert actual_source == before
        return copy.deepcopy(parent)

    monkeypatch.setattr(freeze, "_verify_original_engine_multigpu_repair", original)
    current = dict(before_files)
    for name in freeze.ENGINE_COMPUTE_PARALLEL_REQUIRED_FILES:
        current[name] = "b" * 64
    actual = source(current, "b")
    registration = {"immutable": "registered"}
    write(root / "orchestration/REGISTRATION.json", registration)
    evidence = root / PREFIX / "evidence"
    candidate = root / "code_baseline_parallel_candidate_20261010"
    candidate_file = candidate / "src/sr_f1/baseline_parallel.py"
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_file.write_text("baseline")
    write(
        candidate / "SOURCE_DEPLOYMENT.json",
        source({"src/sr_f1/baseline_parallel.py": file_hash(candidate_file)}, "c"),
    )
    for name in ("ZERO_UPDATE_REUSE_ACTIVATED.json", "ZERO_UPDATE_REUSE_PROCESS.json"):
        write(root / "technical_incidents/engine_memory_20261010" / name, {"consumed": True})
    entries = {
        "orchestration/REGISTRATION.json": registration,
        "orchestration/STATE.json": {
            "test_sealed": True,
            "tasks": {
                "ENGINE": {"attempts": [{"attempt_id": "ENGINE_attempt0004", "job_id": "196508"}]},
                "BASELINE": {"attempts": []},
            },
        },
        "orchestration/journal.jsonl": {"journal": True},
        "accounting/ENGINE.jsonl": {"updates": 2},
        "technical_incidents/baseline_parallel_20261010/HANDOFF_AUTHORIZATION.json": {
            "candidate_source_deployment_sha256": file_hash(candidate / "SOURCE_DEPLOYMENT.json")
        },
        "technical_incidents/baseline_parallel_20261010/HANDOFF_SUBMISSION.json": {
            "returncode": 0,
            "stdout": "196654\n",
        },
        TRACK + "/RUN_MANIFEST.json": {"identity": True},
        TRACK + "/checkpoints/step-02.pt": {"optimizer": "adam", "rng": "full"},
        TRACK + "/rollouts/01-00-0.json": {"raw": "preserved"},
    }
    for name in ("ZERO_UPDATE_REUSE_ACTIVATED.json", "ZERO_UPDATE_REUSE_PROCESS.json"):
        entries["technical_incidents/engine_memory_20261010/" + name] = {"consumed": True}
    for n in (1, 2):
        for slot in range(16):
            for sample in range(8):
                entries[TRACK + f"/rollouts/{n:02d}-{slot:02d}-{sample}.json"] = {
                    "raw": "preserved"
                }
    for name, value in entries.items():
        write(evidence / name, value)
    checkpoint = evidence / TRACK / "checkpoints/step-02.pt"
    latest = dict(step=2, path=checkpoint.name, sha256=file_hash(checkpoint), state_hash="c" * 64)
    for name in ("LATEST.json", "commit-02.json"):
        write(checkpoint.parent / name, latest)
    artifacts = {
        str(p.relative_to(evidence)): dict(sha256=file_hash(p), bytes=p.stat().st_size)
        for p in evidence.rglob("*")
        if p.is_file()
    }
    write(
        root / PREFIX / "PRESERVATION.json",
        dict(
            artifact_hashes=artifacts,
            files=len(artifacts),
            bytes=sum(v["bytes"] for v in artifacts.values()),
        ),
    )
    write(
        root / PREFIX / "TERMINAL_JOBS.json",
        dict(
            controller_terminal=True,
            gpu_terminal=True,
            controller_queue_empty=True,
            gpu_queue_empty=True,
            gpu_job_id="196508",
            controller_job_id="196654",
            controller_terminal_state="CANCELLED",
            gpu_terminal_state="CANCELLED",
        ),
    )
    write(
        root / PREFIX / "CPU_STATE_REVIEW.json",
        dict(
            status="PASS_PRESERVED_FULL_STATE",
            committed_logical_step=2,
            checkpoint_state_hash=latest["state_hash"],
            checkpoint_file_sha256=latest["sha256"],
            raw_record_hashes_verified=True,
            optimizer_state_verified=True,
            rng_state_verified=True,
            learning_rate=1e-4,
            controller_identity_verified=True,
            journal_integrity=True,
        ),
    )
    write(
        root / PREFIX / "RECOVERY_REVIEW.json",
        dict(
            status="PASS_FULL_ENGINE_RESTART",
            science_attempts=0,
            test_sealed=True,
            full_engine_restart=True,
            no_partial_gradient_reuse=True,
            no_old_once_reuse=True,
            old_handoff_superseded=True,
            old_candidate_preserved=True,
        ),
    )
    numerical = {
        n: current[n]
        for n in (
            "src/sr_f1/runtime.py",
            "src/sr_f1/training.py",
            "src/sr_f1/compute_parallel.py",
            "src/sr_f1/compute_workers.py",
            "src/sr_f1/recompute.py",
        )
    }
    write(
        root / PREFIX / "REAL_MODEL_PROBE.json",
        dict(
            status="PASS",
            numerical_equivalence=True,
            all_four_actual_compute=True,
            actual_compute_gpu_count=4,
            source_hashes=numerical,
            synthetic_768_smoke={"status": "PASS"},
        ),
    )
    write(
        root / PREFIX / "GPU_PROBE.json",
        dict(
            status="PASS",
            numerical_equivalence=True,
            exact_ordered_gradient_accumulation=True,
            actual_compute_gpu_count=4,
            source_file_hashes=numerical,
            real_model_probe={
                "path": PREFIX + "REAL_MODEL_PROBE.json",
                "sha256": file_hash(root / PREFIX / "REAL_MODEL_PROBE.json"),
            },
        ),
    )
    repair = freeze.build_engine_compute_parallel_repair(
        root,
        actual_source=actual,
        gpu_count=4,
        authorized_at="2026-10-10",
        authorized_user_message="parallel compute, unchanged science",
    )
    write(root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json", repair)
    return root, actual, repair


def test_compute_revision_accepts_preserved_nonzero_updates(compute_case):
    root, actual, _repair = compute_case
    result = freeze.verify_engine_compute_parallel_repair(root, actual_source=actual)
    assert result["maintained_attempt_id"] == "ENGINE_attempt0004"
    assert result["gpu_count"] == 4
    assert (
        freeze.verify_engine_multigpu_repair(root, actual_source=actual)
        == result["engine_multigpu_repair"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_count", 6),
        ("gpu_count", True),
        ("scientific_protocol_unchanged", False),
        ("gradient_reduction", "TREE_REDUCTION"),
        ("baseline_requires_engine_acceptance", False),
        ("original_execution_freeze_sha256", "f" * 64),
        ("previous_source_commit", "f" * 40),
    ],
)
def test_compute_receipt_mutation_rejected(compute_case, field, value):
    root, actual, repair = compute_case
    repair[field] = value
    write(root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json", repair)
    with pytest.raises(PermissionError):
        freeze.verify_engine_compute_parallel_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("GPU_PROBE.json", "numerical_equivalence", False),
        ("GPU_PROBE.json", "actual_compute_gpu_count", 1),
        ("GPU_PROBE.json", "source_file_hashes", {}),
        ("TERMINAL_JOBS.json", "gpu_queue_empty", False),
        ("TERMINAL_JOBS.json", "gpu_job_id", "unrelated"),
        ("TERMINAL_JOBS.json", "controller_job_id", "unrelated"),
        ("CPU_STATE_REVIEW.json", "learning_rate", 1e-5),
        ("CPU_STATE_REVIEW.json", "rng_state_verified", False),
        ("RECOVERY_REVIEW.json", "full_engine_restart", False),
        ("RECOVERY_REVIEW.json", "old_candidate_preserved", False),
    ],
)
def test_rebound_evidence_must_still_satisfy_acceptance(compute_case, name, field, value):
    root, actual, repair = compute_case
    path = root / PREFIX / name
    evidence = json.loads(path.read_text())
    evidence[field] = value
    write(path, evidence)
    repair["historical_artifact_hashes"][PREFIX + name] = file_hash(path)
    write(root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json", repair)
    with pytest.raises(PermissionError):
        freeze.verify_engine_compute_parallel_repair(root, actual_source=actual)


def test_raw_evidence_tamper_rejected(compute_case):
    root, actual, _ = compute_case
    (root / PREFIX / "evidence" / TRACK / "rollouts/01-00-0.json").write_text("changed")
    with pytest.raises(PermissionError, match="evidence changed"):
        freeze.verify_engine_compute_parallel_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "field,value",
    [
        ("numerical_equivalence", False),
        ("actual_compute_gpu_count", 1),
        ("synthetic_768_smoke", {"status": "NOT_RUN"}),
        ("source_hashes", {}),
        ("all_four_actual_compute", False),
    ],
)
def test_real_model_probe_cannot_be_replaced_by_synthetic_pass(compute_case, field, value):
    root, actual, repair = compute_case
    real_path = root / PREFIX / "REAL_MODEL_PROBE.json"
    real = json.loads(real_path.read_text())
    real[field] = value
    write(real_path, real)
    probe_path = root / PREFIX / "GPU_PROBE.json"
    probe = json.loads(probe_path.read_text())
    probe["real_model_probe"]["sha256"] = file_hash(real_path)
    write(probe_path, probe)
    repair["historical_artifact_hashes"][PREFIX + "GPU_PROBE.json"] = file_hash(probe_path)
    write(root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json", repair)
    with pytest.raises(PermissionError):
        freeze.verify_engine_compute_parallel_repair(root, actual_source=actual)
