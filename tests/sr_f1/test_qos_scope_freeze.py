import copy

import pytest

from mm_core.execution import atomic_json, object_hash
from sr_f1.contract import PLAN_ID, file_hash
from sr_f1.freeze import QOS_SCOPE_REPAIR_ID, verify_qos_scope_repair


@pytest.fixture
def repair(tmp_path):
    root = tmp_path
    files = {
        "src/sr_f1/freeze.py": b"original freeze",
        "src/sr_f1/orchestration.py": b"original scheduler",
        "src/sr_f1/runtime.py": b"unchanged scientific runtime",
        "docs/sr_f1/amendments/v1/AMENDMENT.md": b"unchanged amendment",
    }
    snapshot = root / "code_before_qos_scope_20261010"
    for name, data in files.items():
        p = snapshot / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    before = {n: file_hash(snapshot / n) for n in files}
    original = dict(
        source_commit="a" * 40, source_file_hashes=before, source_tree_sha256=object_hash(before)
    )
    atomic_json(root / "SOURCE_AND_RENDER_MANIFEST.json", original)
    atomic_json(
        root / "EXECUTION_FREEZE.json",
        dict(
            experiment="SR-F1.1-20261010",
            status="FROZEN",
            artifact_hashes={
                "SOURCE_AND_RENDER_MANIFEST.json": file_hash(
                    root / "SOURCE_AND_RENDER_MANIFEST.json"
                )
            },
        ),
    )
    atomic_json(
        root / "manifests/ALLOCATION_PERMISSION.json",
        dict(
            qos="soujanya-poria-startfund-2026-03",
            gpus_per_worker=1,
        ),
    )
    prefix = "technical_incidents/qos_scope_20261010/"
    atomic_json(
        root / (prefix + "STATE_BEFORE.json"),
        dict(test_sealed=True, tasks={"COMMON_START": {"attempts": []}}),
    )
    atomic_json(root / (prefix + "CONTROLLER_TERMINAL.json"), dict(old_controller_terminal=True))
    after = dict(before)
    after["src/sr_f1/freeze.py"] = "b" * 64
    after["src/sr_f1/orchestration.py"] = "c" * 64
    actual = dict(
        source_commit="d" * 40,
        source_tree_sha256=object_hash(after),
        source_file_hashes=after,
        source_dirty_files=[],
    )
    receipt = dict(
        repair_id=QOS_SCOPE_REPAIR_ID,
        status="USER_AUTHORIZED_QOS_SCHEDULER_ONLY",
        plan_id=PLAN_ID,
        run_root=str(root),
        original_execution_freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        original_source_tree_sha256=original["source_tree_sha256"],
        original_source_commit=original["source_commit"],
        permission_sha256=file_hash(root / "manifests/ALLOCATION_PERMISSION.json"),
        capacity_mode="TEACHER_QOS_SCHEDULER_ONLY",
        qos="soujanya-poria-startfund-2026-03",
        gpus_per_worker=1,
        scientific_protocol_unchanged=True,
        new_generation_count_before_activation=0,
        no_gpu_attempts_before_activation=True,
        preserved_source_relative_path=snapshot.name,
        authorized_user_message="Use teacher QoS; no extra manual cap",
        authorized_at="2026-10-10",
        changed_files=sorted(["src/sr_f1/freeze.py", "src/sr_f1/orchestration.py"]),
        pre_activation_artifact_hashes={
            prefix + n: file_hash(root / (prefix + n))
            for n in ["STATE_BEFORE.json", "CONTROLLER_TERMINAL.json"]
        },
        **actual,
    )
    atomic_json(root / "QOS_SCOPE_REPAIR.json", receipt)
    return root, actual, receipt


def test_authenticated_scheduler_repair_keeps_original_freeze_and_live_progress(repair):
    root, actual, _ = repair
    frozen = file_hash(root / "EXECUTION_FREEZE.json")
    result = verify_qos_scope_repair(root, actual_source=actual)
    assert result["capacity_mode"] == "TEACHER_QOS_SCHEDULER_ONLY"
    atomic_json(root / "COMMON_ZERO_LORA.json", {"state": "created later"})
    atomic_json(
        root / "orchestration/STATE.json",
        {"tasks": {"COMMON_START": {"attempts": [{"job_id": "new"}]}}},
    )
    assert verify_qos_scope_repair(root, actual_source=actual) == result
    assert file_hash(root / "EXECUTION_FREEZE.json") == frozen


@pytest.mark.parametrize(
    "field,value",
    [
        ("qos", "rose"),
        ("gpus_per_worker", 2),
        ("original_execution_freeze_sha256", "0" * 64),
        ("scientific_protocol_unchanged", False),
        ("authorized_user_message", ""),
    ],
)
def test_scope_authorization_cannot_drift(repair, field, value):
    root, actual, receipt = repair
    receipt[field] = value
    atomic_json(root / "QOS_SCOPE_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_qos_scope_repair(root, actual_source=actual)


def test_scientific_code_or_preserved_amendment_changes_are_rejected(repair):
    root, actual, receipt = repair
    altered = copy.deepcopy(actual)
    altered["source_file_hashes"]["src/sr_f1/runtime.py"] = "9" * 64
    altered["source_tree_sha256"] = object_hash(altered["source_file_hashes"])
    receipt.update(altered)
    atomic_json(root / "QOS_SCOPE_REPAIR.json", receipt)
    with pytest.raises(PermissionError, match="scientific source"):
        verify_qos_scope_repair(root, actual_source=altered)
    (root / "code_before_qos_scope_20261010/docs/sr_f1/amendments/v1/AMENDMENT.md").write_text(
        "changed"
    )
    with pytest.raises(PermissionError, match="Preserved pre-QoS source"):
        verify_qos_scope_repair(root, actual_source=actual)
