"""CPU tests of real data/role/discovery implementation; no model evidence."""

import copy
import json
from collections import Counter

import pytest

from src.protocol_state_probes.transforms import emit
from src.verified_discovery_transfer.canonical_targets import (
    alias_key,
    build_training_view,
    empty_parent_alias,
    gold_targets,
    matched_gold_ids,
)
from src.verified_discovery_transfer.discover import discover_sets
from src.verified_discovery_transfer.fresh_cohort import (
    generate_cohort,
    normalize_replay_candidate,
    prepare_cohort,
    solve_public,
    truth_orbit,
)
from src.verified_discovery_transfer.portfolio import (
    atomic_request,
    index_bank,
    pass_at_k,
    replay_verify_first,
    select_fixed_policies,
)
from src.verified_discovery_transfer.public_tasks import (
    RoleDataset,
    compile_case,
    public_verifier,
    verify_raw,
)


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(history())[0]


def history(orbits=()):
    return {
        "schema": "vdt-historical-exclusion-v1",
        "complete": True,
        "sources": [
            {
                "path": "CPU_SYNTHETIC_FIXTURE",
                "sha256": "0" * 64,
                "scope": "CPU fixture only, not scientific evidence",
            }
        ],
        "truth_orbits": [list(k) for k in orbits],
        "exposed_scene_count": len(orbits),
    }


def replay_candidate(cohort):
    task = copy.deepcopy(cohort["T_train"]["public"][0])
    audit = {a["task_id"]: a for a in cohort["T_train"]["audit"]}[task["task_id"]]
    task["split"] = "R_replay"
    return {
        "task": task,
        "canonical_vector": audit["true_world"],
        "raw_completion": json.dumps(audit["true_world"]),
        "source": {
            "split": "train",
            "interface": "SYMBOLIC_FRESH",
            "protocol": "O0",
            "request_id": "CPU_FIXTURE",
            "base_scene_id": task["base_instance_id"],
            "artifact_path": "CPU_FIXTURE",
            "artifact_sha256": "0" * 64,
        },
    }


def bank(tasks, audit, outcome=None, split="T_train"):
    from src.verified_discovery_transfer.config import PROTOCOLS

    by_id = {r["task_id"]: r for r in audit}
    records = []
    for task in tasks:
        for protocol in PROTOCOLS[task["family"]]:
            for draw in range(16):
                role = "teacher_train" if split == "T_train" else "selector"
                row = atomic_request("S96", task, protocol, role, 0, draw, "revision", "runtime")
                success = outcome(task, protocol, draw) if outcome else False
                raw = (
                    json.dumps(
                        emit(
                            by_id[task["task_id"]]["true_world"],
                            compile_case(task, protocol)["output_order"],
                        )
                    )
                    if success
                    else "failed normally"
                )
                row.update(
                    raw_completion=raw,
                    raw_token_ids=[1],
                    completion_length=1,
                    stop_reason="eos",
                    elapsed_seconds=0.0,
                )
                row.update(verify_raw(task, protocol, raw))
                records.append(row)
    return records


def tiny(cohort, split):
    public = [
        next(t for t in cohort[split]["public"] if t["family"] == f)
        for f in ("trend", "cross_series")
    ]
    ids = {t["task_id"] for t in public}
    return public, [a for a in cohort[split]["audit"] if a["task_id"] in ids]


def test_exact_quotas_capacity_and_orbits(cohort):
    seen = set()
    for split, n in [("T_train", 384), ("V_selection", 96), ("E_test", 384)]:
        tasks, audits = cohort[split]["public"], cohort[split]["audit"]
        assert len(tasks) == n
        audit = {a["task_id"]: a for a in audits}
        cells = Counter((t["family"], audit[t["task_id"]]["corrupted_index"]) for t in tasks)
        assert len(cells) == 8 and set(cells.values()) == {n // 8}
        cross = Counter(
            (a["center"], a["corrupted_index"]) for a in audits if a["center"] is not None
        )
        assert len(cross) == 16 and set(cross.values()) == {n // 32}
        for j in range(4):
            trends = [
                t
                for t in tasks
                if t["family"] == "trend" and audit[t["task_id"]]["corrupted_index"] == j
            ]
            assert len({t["chart_type"] for t in trends}) == 2
        for center in range(4):
            for j in range(4):
                cell = [
                    t
                    for t in tasks
                    if t["family"] == "cross_series"
                    and audit[t["task_id"]]["center"] == center
                    and audit[t["task_id"]]["corrupted_index"] == j
                ]
                assert len({t["operation"] for t in cell}) == 3
                assert len({t["chart_type"] for t in cell}) == 2
        for task in tasks:
            a = audit[task["task_id"]]
            assert solve_public(task) == [a["true_world"]]
            orbit = truth_orbit(a["true_world"])
            assert orbit not in seen
            seen.add(orbit)
    guard = cohort["G_guard"]
    assert Counter(t["interface"] for t in guard["public"]) == {
        "IMAGE_CUE_FRESH": 64,
        "SYMBOLIC_FRESH": 32,
    }
    for task, audit in zip(guard["public"], guard["audit"], strict=True):
        if task["interface"] == "SYMBOLIC_FRESH":
            assert tuple(audit["truth_orbit_key"]) not in seen
            seen.add(tuple(audit["truth_orbit_key"]))
        else:
            assert tuple(audit["truth_orbit_key"]) in seen
            assert task["base_instance_id"] == audit["paired_E_task_id"]


def test_determinism_and_historical_capacity(cohort):
    again, capacity = generate_cohort(history())
    assert again == cohort
    assert capacity["remaining_trend_orbits"] == 1617
    from src.verified_discovery_transfer.fresh_cohort import trend_worlds

    with pytest.raises(ValueError, match="Insufficient"):
        generate_cohort(history(trend_worlds()[:1300]))
    with pytest.raises(ValueError, match="complete"):
        generate_cohort({**history(), "complete": False})


def test_public_parser_permutation_and_no_oracle(cohort):
    task = next(t for t in cohort["T_train"]["public"] if t["family"] == "cross_series")
    audit = {a["task_id"]: a for a in cohort["T_train"]["audit"]}[task["task_id"]]
    truth = audit["true_world"]
    assert verify_raw(task, "L11", json.dumps(truth[::-1]))["canonical_vector"] == truth
    for raw in ["[true,1,2,3]", "[1.0,2,3,4]", "[1,2,3,4] explanation", "[1,2,3]", "[100,2,3,4]"]:
        assert not verify_raw(task, "O0", raw)["public_verifier_pass"]
    with pytest.raises(ValueError, match="audit"):
        compile_case({**task, "true_world": truth})
    assert public_verifier(task, truth)
    case = compile_case(task, "B1")
    assert all(
        sum(x * c for x, c in zip(truth, row, strict=True)) == rhs
        for row, rhs in zip(case["H_display"], case["b_display"], strict=True)
    )
    assert case["output_order"] == [0, 1, 2, 3]


def test_role_manifest_denies_leakage_and_mutation(tmp_path, cohort):
    replay = replay_candidate(cohort)
    out = tmp_path / "cohort"
    # Rendering is exercised once here, not represented as model evaluation.
    manifest = prepare_cohort(out, history([truth_orbit(replay["canonical_vector"])]), [replay])
    assert manifest["status"] == "FROZEN"
    teacher = RoleDataset(out, "teacher_train")
    assert len(teacher.public("T_train")) == 384
    for split in ("V_selection", "E_test", "G_guard"):
        with pytest.raises(PermissionError):
            teacher.public(split)
    with pytest.raises(PermissionError):
        teacher.audit("T_train")
    with pytest.raises(AttributeError):
        teacher.role = "final_eval"
    assert len(RoleDataset(out, "gold").audit("T_train")) == 384
    evaluator = RoleDataset(out, "teacher_eval")
    assert len(evaluator.public("G_guard")) == 96
    with pytest.raises(PermissionError):
        evaluator.audit("G_guard")
    with pytest.raises(FileNotFoundError):
        RoleDataset(out, "final_eval").audit("E_test")
    target = out / "T_train/tasks_public.jsonl"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="modified"):
        teacher.public("T_train")


def test_replay_rejects_wrong_source(cohort):
    item = replay_candidate(cohort)
    assert normalize_replay_candidate(item)["verified"]
    for mutation in [
        {"split": "dev"},
        {"panel": "U22"},
        {"protocol": "L11"},
        {"interface": "IMAGE_CUE_FRESH"},
    ]:
        wrong = copy.deepcopy(item)
        wrong["source"].update(mutation)
        with pytest.raises(ValueError):
            normalize_replay_candidate(wrong)
    wrong = copy.deepcopy(item)
    wrong["raw_completion"] = "[0,0,0,0]"
    with pytest.raises(ValueError):
        normalize_replay_candidate(wrong)


def test_atomic_budget_selection_and_non_nested_discovery(cohort):
    vt, va = tiny(cohort, "V_selection")
    vr = bank(vt, va, lambda t, p, i: p == "L11" and i == 0, "V_selection")
    policy = select_fixed_policies(vt, vr, "S96")
    assert set(policy["best_single_B16"].values()) == {"L11"}
    t, a = tiny(cohort, "T_train")
    # trend succeeds only ninth O0 -> O0 lost in MIX; cross succeeds first B1 -> MIX added.
    rows = bank(
        t,
        a,
        lambda t, p, i: (
            (t["family"] == "trend" and p == "O0" and i == 8)
            or (t["family"] == "cross_series" and p == "B1" and i == 0)
        ),
    )
    result = discover_sets(t, rows, policy, "S96", 0)
    assert len(result["J_O0"]) == len(result["J_MIX"]) == 1
    assert len(result["summary"]["mix_lost"]) == len(result["summary"]["mix_added"]) == 1
    assert result["J_SINGLE"] == {}
    assert result["empty_status"]["J_SINGLE"] == "NO_VERIFIED_TARGETS_RETURN_PARENT"
    assert len({r["sample_seed"] for r in rows}) == len(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        index_bank(t, [*rows, rows[0]], "S96", 0, "T_train")
    with pytest.raises(ValueError, match="Incomplete"):
        index_bank(t, rows[:-1], "S96", 0, "T_train")
    assert pass_at_k(16, 1, 2) == 1 / 8
    assert pass_at_k(16, 1, 2) != 1 - (15 / 16) ** 2
    failed = [r for r in rows if r["task_id"] == t[0]["task_id"] and not r["public_verifier_pass"]][
        :2
    ]
    assert replay_verify_first(t[0], *failed)["canonical_vector"] is None


def test_view_alias_and_gold_match(cohort):
    t, a = tiny(cohort, "T_train")
    targets = gold_targets(t, a)
    view = build_training_view(t, targets, source="gold")
    same = copy.deepcopy(view)
    for row in same:
        row["label_source_provenance"] = "self_public_verifier"
        row["discovery_request_ids"] = ["another-source"]
    settings = {
        "lora_initial_state": "hash",
        "tokenizer_identity": "tok",
        "chat_template_identity": "chat",
        "eos_id": 7,
        "optimizer": {"lr": 1e-5},
        "scheduler": {"warmup": 8},
        "batch": [12, 4],
        "seed": 73101,
        "steps": 256,
        "runtime_identity": "runtime",
    }
    assert alias_key("S96", 0, view, [], settings) == alias_key("S96", 0, same, [], settings)
    assert alias_key("S96", 0, view, [], settings) != alias_key("REP96", 0, view, [], settings)
    changed = copy.deepcopy(view)
    changed[0]["weight"] = 2
    assert alias_key("S96", 0, view, [], settings) != alias_key("S96", 0, changed, [], settings)
    with pytest.raises(ValueError):
        alias_key("S96", 0, view, [], {})
    with pytest.raises(ValueError, match="SELF"):
        build_training_view(t, targets)
    assert empty_parent_alias("S96", 0)["SFT_updates"] == 0
    chosen, report = matched_gold_ids(t, a, set(targets), 42)
    assert chosen == sorted(targets) and report["jaccard_with_mix"] == 1
    bad = copy.deepcopy(t)
    bad[0]["split"] = "E_test"
    with pytest.raises(PermissionError):
        build_training_view(bad, targets, source="gold")


def test_b16_and_b2_select_independently_and_exact_ties(cohort):
    tasks = []
    for family in ("trend", "cross_series"):
        tasks.extend([t for t in cohort["V_selection"]["public"] if t["family"] == family][:2])
    ids = {t["task_id"] for t in tasks}
    audits = [a for a in cohort["V_selection"]["audit"] if a["task_id"] in ids]
    first = {t["family"]: t["task_id"] for t in tasks[::2]}
    records = bank(
        tasks,
        audits,
        lambda t, p, i: (
            (p == "O0" and i == 0) or (p == "L11" and t["task_id"] == first[t["family"]])
        ),
        "V_selection",
    )
    policies = select_fixed_policies(tasks, records, "S96")
    assert set(policies["best_single_B16"].values()) == {"O0"}
    assert set(policies["best_single_B2"].values()) == {"L11"}
    # All-zero ties still select a complete registered policy and retain failures.
    zero = bank(tasks, audits, split="V_selection")
    policies = select_fixed_policies(tasks, zero, "S96")
    assert set(policies["best_single_B16"].values()) == {"O0"}
    assert set(policies["best_single_B2"].values()) == {"O0"}
    assert all(pair == ["O0", "O0"] for pair in policies["best_pair_B2"].values())
