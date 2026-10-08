"""J23 exact inheritance, role boundaries and synthetic confirmation regressions."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from random import Random
from unittest.mock import patch

import pytest

from src.exposure_position.data import (
    build_confirmation,
    build_development_manifests,
    build_training_manifests,
    write_training_manifests,
)
from src.exposure_position.schedule import ExplicitScheduleSampler, validate_schedule
from src.exposure_position.schema import (
    ARMS,
    EXPERIMENT_ID,
    SCHEDULE_SEEDS,
    RoleDataset,
    digest,
    file_digest,
    public_prompt,
    read_training_bound,
    solve_public,
    training_rows,
    validate_public,
    verify_frozen,
    write_json,
    write_jsonl,
)
from src.exposure_substitution.schema import public_prompt as legacy_prompt
from src.exposure_substitution.schema import read_jsonl

LEGACY = Path(__file__).resolve().parents[1] / "docs/exposure_substitution/design"
SYNTHETIC_ONLY_ROOT_SEED = 0x53594E544845544943


@pytest.fixture(scope="module")
def constructed():
    return build_training_manifests(LEGACY)


@pytest.fixture(scope="module")
def development(constructed):
    return build_development_manifests(LEGACY, constructed[0])


def _synthetic_receipt(orbits):
    receipt = dict(
        status="PASS",
        phase_id=EXPERIMENT_ID,
        model_calls=0,
        files=[{"path": "synthetic-fixture-only", "sha256": "0" * 64}],
        coverage={"all_required_categories_verified": True},
        truth_orbit_keys=[list(orbit) for orbit in sorted(orbits)],
    )
    receipt["audit_hash"] = digest(receipt)
    return receipt


@pytest.fixture(scope="module")
def synthetic_confirmation(tmp_path_factory):
    # Substitute an independent RNG before invoking the production builder.
    # The registered scientific seed must never produce candidates in tests.
    orbits = {(1, 2, 3, 4), (11, 12, 13, 14)}
    from src.exposure_position.exclusions import audit_exclusions

    root = tmp_path_factory.mktemp("synthetic-exclusions").resolve()
    truth = root / "truth.json"
    write_json(truth, {"truth_orbit_keys": [list(o) for o in sorted(orbits)]})
    index = root / "history.json"
    write_json(
        index,
        dict(
            schema="vdt-historical-exclusion-v1",
            complete=True,
            errors=[],
            sources=[dict(path=str(truth), scope="synthetic-only", sha256=file_digest(truth))],
            truth_orbits=[list(o) for o in sorted(orbits)],
            exposed_scene_count=len(orbits),
        ),
    )
    receipt = audit_exclusions(
        exact_sources=[
            dict(path=str(index), category="historical_index", kind="history_index"),
            *[
                dict(path=str(truth), category=c, kind="audit")
                for c in ("legacy_j2", "later_local", "manual_fixtures")
            ],
        ]
    )
    assert SYNTHETIC_ONLY_ROOT_SEED != 2026100803
    with patch(
        "src.exposure_position.data.random.Random", return_value=Random(SYNTHETIC_ONLY_ROOT_SEED)
    ) as rng_constructor:
        files, report = build_confirmation(orbits, exclusion_audit=receipt, auditor_authorized=True)
    rng_constructor.assert_called_once_with(2026100803)
    return files, dict(
        report,
        root_seed=SYNTHETIC_ONLY_ROOT_SEED,
        fixture_rng_override=True,
        production_seed_not_used=True,
    )


def test_background_inherited_without_filter_and_four_arm_targets_identical(constructed):
    files, report = constructed
    assert report["common_filter_reapplied"] is False
    assert [r["c4j3_common_exposures_per_arm"] for r in report["schedule_proofs"]] == [15, 16, 16]
    for name in ("common_targets.jsonl", "replay_targets.jsonl"):
        old = read_jsonl(LEGACY / "manifests" / name)
        for source, new in zip(old, files[name], strict=True):
            assert new["source_task_id"] == source["task"]["task_id"]
            assert new["target"] == source["target"]
            assert new["task"]["task_id"] != source["task"]["task_id"]
            assert public_prompt(new["task"]) == legacy_prompt(source["task"])
    by_root = {}
    for row in files["donor_targets.jsonl"]:
        by_root.setdefault(row["root_id"], []).append(row)
    assert len(by_root) == 32
    assert all(len(v) == 4 and len({r["target"] for r in v}) == 1 for v in by_root.values())


def test_j3_uses_registered_replacements_without_delta_clipping(constructed):
    files, _ = constructed
    tasks = {t["task_id"]: t for t in files["donors_public.jsonl"]}
    roots = {r["root_id"]: r for r in files["donor_root_registry_AUDIT_ONLY.jsonl"]}
    for audit in files["donors_audit.jsonl"]:
        task, root = tasks[audit["task_id"]], roots[audit["root_id"]]
        j = ARMS[audit["logical_arm_id"]]["corrupted_index_0based"]
        expected = list(root["true_world"])
        expected[j] = root["replacements"][j]
        assert task["observed"] == expected
        assert solve_public(task) == [root["true_world"]]
        assert audit["signed_delta"] == expected[j] - root["true_world"][j]
    assert len(tasks) == 128


def test_schedule_preserves_every_legacy_slot_and_committed_cursor(constructed):
    files, _ = constructed
    mapping = {
        r["source_task_id"]: r["task_id"]
        for r in files["source_mapping.jsonl"]
        if r["role"] != "donor"
    }
    roots = {
        r["source_root_id"]: r["root_id"] for r in files["donor_root_registry_AUDIT_ONLY.jsonl"]
    }
    for seed in SCHEDULE_SEEDS:
        index = validate_schedule(files[f"schedule_{seed}.jsonl"])
        source = [
            r
            for r in read_jsonl(LEGACY / "manifests" / f"schedule_{seed}.jsonl")
            if r["arm"] == "A_LOCAL_C1"
        ]
        for arm in ARMS:
            counts = Counter()
            for old in source:
                new = index[arm, old["update"], old["slot"]]
                if old["role"] == "donor":
                    assert new["donor_root"] == roots[old["donor_root"]]
                    counts[new["donor_root"]] += 1
                else:
                    assert new["task_id"] == mapping[old["task_id"]]
            assert set(counts.values()) == {8}
    sampler = ExplicitScheduleSampler(files["schedule_108701.jsonl"], "A3_LOCAL_C1_J3")
    copied = sampler.peek()
    copied[0]["task_id"] = "mutated"
    assert sampler.peek()[0]["task_id"] != "mutated"
    sampler.commit(1)
    restored = ExplicitScheduleSampler(files["schedule_108701.jsonl"], "A3_LOCAL_C1_J3")
    restored.load_state_dict(sampler.state_dict())
    assert restored.peek() == sampler.peek()
    with pytest.raises(ValueError, match="different arm/schedule"):
        ExplicitScheduleSampler(files["schedule_108702.jsonl"], "A3_LOCAL_C1_J3").load_state_dict(
            sampler.state_dict()
        )


def test_shifted_common_slot_is_rejected(constructed):
    rows = copy.deepcopy(constructed[0]["schedule_108701.jsonl"])
    selected = [r for r in rows if r["arm"] == "B3_FORWARD_C4_J3" and r["role"] == "common"]
    selected[0]["task_id"], selected[1]["task_id"] = selected[1]["task_id"], selected[0]["task_id"]
    with pytest.raises(ValueError, match="slot mismatch"):
        validate_schedule(rows)


def test_public_prompt_rejects_audit_or_old_phase(constructed):
    task = copy.deepcopy(constructed[0]["donors_public.jsonl"][0])
    for field in ("true_world", "corrupted_index", "center", "signed_delta", "source_task_id"):
        with pytest.raises(ValueError, match="audit fields"):
            public_prompt(dict(task, **{field: "forbidden"}))
    task["phase_id"] = "SER_J2_20261007"
    with pytest.raises(ValueError, match="phase"):
        validate_public(task)


def test_development_panels_fixed_scope_and_illegal_delta_retained(development):
    files, audit = development
    assert audit["delta_tasks_by_cell"] == {"c4j4": 106, "c3j3": 109}
    attempts = files["DEV_DELTA/legality_attempts.jsonl"]
    assert len(attempts) == 256
    assert sum(r["status"] == "REJECTED_OUT_OF_DOMAIN" for r in attempts) == 41
    assert len(files["DEV_TRAJECTORY/tasks_public.jsonl"]) == 128
    assert len(files["TRAIN_FIT_FULL/tasks_public.jsonl"]) == 142
    assert all(
        len(ids) == 46 for ids in files["panel_selection.json"]["TRAIN_FIT_FULL"]["by_arm"].values()
    )
    for panel in ("DEV_DELTA", "DEV_TRAJECTORY", "TRAIN_FIT_FULL"):
        truth = {r["task_id"]: r["true_world"] for r in files[panel + "/audit_only.jsonl"]}
        for task in files[panel + "/tasks_public.jsonl"]:
            assert solve_public(task) == [truth[task["task_id"]]]


def test_confirmation_requires_bound_nonempty_exclusion_and_auditor():
    with pytest.raises(PermissionError, match="auditor"):
        build_confirmation(set(), exclusion_audit={})
    with pytest.raises(ValueError, match="BLOCKED_EXCLUSION_AUDIT"):
        build_confirmation(
            set(), exclusion_audit=_synthetic_receipt(set()), auditor_authorized=True
        )
    receipt = _synthetic_receipt({(1, 2, 3, 4)})
    receipt["coverage"]["all_required_categories_verified"] = False
    with pytest.raises(ValueError, match="BLOCKED_EXCLUSION_AUDIT"):
        build_confirmation({(1, 2, 3, 4)}, exclusion_audit=receipt, auditor_authorized=True)


def test_synthetic_confirmation_counts_single_repair_and_global_exclusion(synthetic_confirmation):
    files, report = synthetic_confirmation
    roots = files["E_CONFIRM2/root_registry_AUDIT_ONLY.jsonl"]
    tasks, audits = files["E_CONFIRM2/tasks_public.jsonl"], files["E_CONFIRM2/audit_only.jsonl"]
    assert report["tasks"] == 992 and report["model_calls"] == 0
    assert report["root_seed"] == SYNTHETIC_ONLY_ROOT_SEED
    assert report["fixture_rng_override"] is report["production_seed_not_used"] is True
    orbits = {tuple(sorted(r["true_world"])) for r in roots}
    assert len(roots) == len(orbits) == 224
    assert not orbits.intersection({(1, 2, 3, 4), (11, 12, 13, 14)})
    for task, audit in zip(tasks, audits, strict=True):
        assert solve_public(task) == [audit["true_world"]]
    for family, count in (("trend", 16), ("duplicate_encoding", 8)):
        assert Counter(
            a["corrupted_index"]
            for t, a in zip(tasks, audits, strict=True)
            if t["family"] == family
        ) == {j: count for j in range(4)}


def _write_run(run, files, audit, confirmation=None):
    write_json(run / "machine.json", {"legacy_run": str(LEGACY)})
    write_training_manifests(run, files, audit)
    for name, rows in (confirmation or {}).items():
        write_jsonl(run / "manifests" / name, rows)
    frozen = dict(
        status="FROZEN",
        phase_id=EXPERIMENT_ID,
        files={
            p.relative_to(run).as_posix(): {"sha256": file_digest(p)}
            for p in run.rglob("*")
            if p.is_file()
        },
    )
    frozen["plan_hash"] = digest(frozen)
    write_json(run / "FROZEN_PLAN.json", frozen)
    return frozen


def test_trainer_never_opens_confirmation_and_confirm_never_opens_truth(
    tmp_path, monkeypatch, constructed, synthetic_confirmation
):
    frozen = _write_run(tmp_path, *constructed, confirmation=synthetic_confirmation[0])
    original, opened = Path.open, []

    def traced(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", traced)
    assert len(training_rows(tmp_path, "A3_LOCAL_C1_J3")) == 278
    ExplicitScheduleSampler.from_run(tmp_path, 0, "A3_LOCAL_C1_J3")
    verify_frozen(tmp_path, role="trainer")
    assert not any("E_CONFIRM2" in p for p in opened)
    trainer = RoleDataset(tmp_path, "trainer")
    for path in ("E_CONFIRM2/tasks_public.jsonl", "E_CONFIRM2/audit_only.jsonl", "../machine.json"):
        with pytest.raises(PermissionError):
            trainer._read(path)
    with pytest.raises(AttributeError):
        trainer.role = "auditor"
    opened.clear()
    verify_frozen(tmp_path, role="confirm_worker")
    assert len(RoleDataset(tmp_path, "confirm_worker").public("E_CONFIRM2")) == 992
    assert not any("audit_only" in p or "targets.jsonl" in p for p in opened)
    write_json(
        tmp_path / "RELEASE_RECEIPT.json",
        {"plan_hash": frozen["plan_hash"], "all_registered_models_terminal": True},
    )
    with pytest.raises(PermissionError, match="sealed"):
        RoleDataset(tmp_path, "analysis").audit("E_CONFIRM2")


def test_source_binding_technical_only_and_post_freeze_mutation(tmp_path, constructed):
    files, audit = constructed
    write_json(tmp_path / "machine.json", {"legacy_run": str(LEGACY)})
    write_training_manifests(tmp_path, files, audit)
    assert len(training_rows(tmp_path, "B3_FORWARD_C4_J3", technical=True)) == 278
    with pytest.raises(PermissionError):
        read_training_bound(tmp_path, "manifests/E_CONFIRM2/tasks_public.jsonl", technical=True)
    with pytest.raises(FileExistsError):
        write_training_manifests(tmp_path, files, audit)
    path = tmp_path / "manifests/common_targets.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="modified"):
        training_rows(tmp_path, "B3_FORWARD_C4_J3", technical=True)


def test_role_paths_reject_symlink_even_when_hash_matches(tmp_path, constructed):
    _write_run(tmp_path, *constructed)
    path = tmp_path / "manifests/donors_public.jsonl"
    target = tmp_path / "elsewhere.jsonl"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(PermissionError, match="Symlinked"):
        RoleDataset(tmp_path, "trainer").public("DONOR_TRAIN")
