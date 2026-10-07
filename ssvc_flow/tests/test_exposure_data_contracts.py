"""SER-J2 data, role, fixed-slot, and actual legacy provenance regressions."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from src.exposure_substitution.import_sources import (
    _validate_later_receipt,
    cpu_check,
    extract_truth_orbits,
    import_legacy,
)
from src.exposure_substitution.matched_tasks import build_new_tasks
from src.exposure_substitution.schedule import ExplicitScheduleSampler, build_schedule
from src.exposure_substitution.schema import (
    EXPERIMENT_ID,
    MICROBATCH_SLOTS,
    SCHEDULE_SEEDS,
    RoleDataset,
    digest,
    file_digest,
    public_prompt,
    read_jsonl,
    solve_public,
    validate_public,
    verify_frozen,
    write_json,
    write_jsonl,
)

PLAN = Path(__file__).resolve().parents[1] / "docs/exposure_substitution/design"


@pytest.fixture(scope="module")
def manifests():
    return {
        path.relative_to(PLAN / "manifests").as_posix(): read_jsonl(path)
        for path in (PLAN / "manifests").rglob("*.jsonl")
    }


def test_three_schedules_reproduce_reference_and_preserve_cross_arm_slots(manifests):
    common = [row["task"]["task_id"] for row in manifests["common_targets.jsonl"]]
    replay = [row["task"]["task_id"] for row in manifests["replay_targets.jsonl"]]
    donors = manifests["donor_targets.jsonl"]
    roots = list(dict.fromkeys(row["root_id"] for row in donors))
    ids = {(row["root_id"], row["arm"]): row["task_id"] for row in donors}
    schedules = []
    for seed in SCHEDULE_SEEDS:
        actual = build_schedule(common, replay, roots, ids, seed)
        assert actual == manifests[f"schedule_{seed}.jsonl"]
        schedules.append(digest(actual))
    assert len(set(schedules)) == 3
    assert list(map(len, MICROBATCH_SLOTS)) == [4, 4, 3, 1, 4]


def test_schedule_rejects_shifted_common_task_even_when_exposure_count_same(manifests):
    rows = copy.deepcopy(manifests["schedule_108701.jsonl"])
    b_rows = [row for row in rows if row["arm"] == "B_FORWARD_C4" and row["role"] == "common"]
    b_rows[0]["task_id"], b_rows[1]["task_id"] = b_rows[1]["task_id"], b_rows[0]["task_id"]
    with pytest.raises(ValueError, match="slot mismatch"):
        ExplicitScheduleSampler(rows, "B_FORWARD_C4")


def test_resume_committed_cursor_and_plan_binding(manifests):
    rows = manifests["schedule_108701.jsonl"]
    sampler = ExplicitScheduleSampler(rows, "A_LOCAL_C1")
    first = sampler.peek()
    first[0]["task_id"] = "caller-owned-change"
    assert sampler.peek()[0]["task_id"] != first[0]["task_id"]
    for update in range(1, 5):
        sampler.commit(update)
    restored = ExplicitScheduleSampler(rows, "A_LOCAL_C1")
    restored.load_state_dict(sampler.state_dict())
    assert restored.peek() == sampler.peek()
    assert restored.peek()[0]["update"] == 5
    with pytest.raises(ValueError, match="different arm/schedule"):
        ExplicitScheduleSampler(manifests["schedule_108702.jsonl"], "A_LOCAL_C1").load_state_dict(
            sampler.state_dict()
        )
    with pytest.raises(ValueError, match="different arm/schedule"):
        ExplicitScheduleSampler(rows, "C_FORWARD_C3").load_state_dict(sampler.state_dict())
    with pytest.raises(ValueError, match="next scheduled"):
        sampler.commit(9)


def test_all_944_new_and_246_reused_have_unique_public_repairs(manifests):
    audits = {
        row["task_id"]: row
        for name in ("donors_audit.jsonl", "E_DIAG/audit_only.jsonl", "E_CONFIRM/audit_only.jsonl")
        for row in manifests[name]
    }
    tasks = [
        task
        for name in (
            "donors_public.jsonl",
            "E_DIAG/tasks_public.jsonl",
            "E_CONFIRM/tasks_public.jsonl",
        )
        for task in manifests[name]
    ]
    assert len(tasks) == 944
    for task in tasks:
        assert solve_public(task) == [audits[task["task_id"]]["true_world"]]
    import json

    reused = manifests["common_targets.jsonl"] + manifests["replay_targets.jsonl"]
    assert len(reused) == 246
    for row in reused:
        assert solve_public(row["task"]) == [json.loads(row["target"])]


def test_new_history_collision_regenerates_from_fixed_seed_before_freeze():
    original, _ = build_new_tasks(set())
    first = original["new_root_registry_AUDIT_ONLY.jsonl"][0]
    excluded = tuple(sorted(first["true_world"]))
    regenerated, _ = build_new_tasks({excluded})
    repeated, _ = build_new_tasks({excluded})
    assert regenerated == repeated
    assert regenerated != original
    roots = regenerated["new_root_registry_AUDIT_ONLY.jsonl"]
    assert len(roots) == 274
    assert excluded not in {tuple(sorted(row["true_world"])) for row in roots}
    assert len({tuple(sorted(row["true_world"])) for row in roots}) == 274


def test_public_prompt_has_only_inherited_text_and_rejects_audit_fields(manifests):
    prompts = {row["task_id"]: row["prompt"] for row in manifests["new_prompts_public.jsonl"]}
    for task in manifests["donors_public.jsonl"]:
        prompt = public_prompt(task)
        assert prompt == prompts[task["task_id"]]
        assert set(prompt) == {"system", "user"}
        assert task["root_id"] not in prompt["user"]
    task = copy.deepcopy(manifests["donors_public.jsonl"][0])
    task["truth"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="audit fields"):
        validate_public(task)


def _frozen_fixture(root, manifests):
    for filename in (
        "donors_public.jsonl",
        "E_CONFIRM/tasks_public.jsonl",
        "E_CONFIRM/audit_only.jsonl",
    ):
        write_jsonl(root / "manifests" / filename, manifests[filename])
    frozen = dict(
        status="FROZEN",
        experiment_id=EXPERIMENT_ID,
        files={
            path.relative_to(root).as_posix(): {"sha256": file_digest(path)}
            for path in root.rglob("*.jsonl")
        },
    )
    frozen["plan_hash"] = digest(frozen)
    write_json(root / "FROZEN_PLAN.json", frozen)
    return frozen


def test_role_boundaries_and_post_freeze_mutation_are_rejected(tmp_path, manifests):
    _frozen_fixture(tmp_path, manifests)
    trainer = RoleDataset(tmp_path, "trainer")
    with pytest.raises(PermissionError):
        trainer.public("E_CONFIRM")
    with pytest.raises(PermissionError):
        trainer.audit("DONOR_TRAIN")
    with pytest.raises(AttributeError):
        trainer.role = "analysis"
    assert len(RoleDataset(tmp_path, "confirm_worker").public("E_CONFIRM")) == 800
    path = tmp_path / "manifests/donors_public.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="modified"):
        verify_frozen(tmp_path)


def test_confirm_audit_release_requires_same_plan_hash(tmp_path, manifests):
    frozen = _frozen_fixture(tmp_path, manifests)
    write_json(
        tmp_path / "FINAL_RELEASE.json",
        dict(plan_hash="other", all_registered_models_terminal=True),
    )
    with pytest.raises(PermissionError, match="sealed"):
        RoleDataset(tmp_path, "analysis").audit("E_CONFIRM")
    write_json(
        tmp_path / "FINAL_RELEASE.json",
        dict(plan_hash=frozen["plan_hash"], all_registered_models_terminal=True),
    )
    assert len(RoleDataset(tmp_path, "analysis").audit("E_CONFIRM")) == 800


def test_history_extraction_never_treats_model_predictions_as_truth():
    assert extract_truth_orbits(
        {"observed": [9, 8, 7, 6], "parsed_vector": [1, 2, 3, 4], "true_world": [4, 3, 2, 1]}
    ) == {(1, 2, 3, 4)}
    with pytest.raises(ValueError, match="Malformed"):
        extract_truth_orbits({"true_world": [True, 2, 3, 4]})


def test_failed_or_unbound_later_history_receipt_cannot_enable_gpu(tmp_path):
    for receipt in ({}, {"schema": "ser-j2-later-history-audit-v1", "status": "NOT_VERIFIED"}):
        with pytest.raises(ValueError, match="incomplete"):
            _validate_later_receipt(tmp_path / "audit.json", receipt, set(), 0)
    receipt = dict(
        schema="ser-j2-later-history-audit-v1",
        status="COMPLETE",
        gpu_calls_before_audit=0,
        local_scanned_roots=["/explicit/scope"],
        new_orbits_after_archive=0,
        server_inventory="inventory.json",
        server_manifest_count=1,
        local_manifests=[{"sha256": "unknown"}],
    )
    write_json(tmp_path / "inventory.json", {"task_manifest_inventory": [{"sha256": "unknown"}]})
    with pytest.raises(ValueError, match="absent from explicit"):
        _validate_later_receipt(tmp_path / "audit.json", receipt, {"known"}, 0)


@pytest.mark.skipif(
    not os.environ.get("SER_J2_LEGACY_RUN_ROOT"),
    reason="Actual legacy archive integration input not configured",
)
def test_actual_original_canonical_views_and_targets():
    common, _, replay, _, _, provenance = import_legacy(os.environ["SER_J2_LEGACY_RUN_ROOT"])
    assert len(common) == 182 and len(replay) == 64
    assert [row["focus_count"] for row in provenance["original_views"]] == [200, 202, 200, 196]
    assert all(row["original_training_signature_verified"] for row in provenance["original_views"])


@pytest.mark.skipif(
    not os.environ.get("SER_J2_PREPARED_RUN"), reason="Actual frozen integration run not configured"
)
def test_actual_frozen_cpu_integration():
    receipt = cpu_check(os.environ["SER_J2_PREPARED_RUN"])
    assert receipt["status"] == "PASS"
    assert receipt["formal_generated_answers"] == 145280
    assert receipt["unique_public_repairs_new"] == 944
    assert receipt["unique_public_repairs_reused"] == 246
