"""Fixed population, root-cluster inference and actual-credit audit regressions."""

import copy

import numpy as np
import pytest

from sr_f4 import c_local_reporting as report
from sr_f4.common import read, write, write_rows


def planned_population():
    tasks = {}
    native = []
    for family in report.FAMILIES:
        for root in range(16):
            for variant in range(4):
                qid = f"{family}-{root}-{variant}"
                tasks[qid] = dict(
                    qid=qid,
                    root_id=f"{family}-{root}",
                    world_id=qid,
                    cell=family,
                    family="TOPK",
                    split=report.NEW_PANEL,
                )
                native.append(dict(qid=qid, panel=report.NEW_PANEL, G=4, protocol="structured"))
    for i in range(128):
        qid = f"old-{i}"
        tasks[qid] = dict(qid=qid, world_id=qid, split="OLD_REGRESSION", pool="TEST_ID")
        native.append(dict(qid=qid, panel=report.OLD_PANEL, G=4, protocol="structured"))
    chart = []
    for i in range(250):
        qid = f"chart-{i}"
        tasks[qid] = dict(qid=qid, pool="CHARTQA", split="CHARTQA")
        chart.append(dict(qid=qid, panel="CHARTQA", G=1, protocol="chartqa_plain"))
    plans = {
        b: [
            dict(a, group_index=i, seeds=[b * 10000 + 4 * i + d for d in range(a["G"])])
            for i, a in enumerate(native + (chart if b == 0 else []))
        ]
        for b in report.BLOCKS
    }
    return tasks, plans


def test_fixed_population_and_lineage_counts_before_outputs():
    tasks, plans = planned_population()
    assert report.validate_plans(tasks, plans) == 31970
    first = plans[0][0]["qid"]
    tasks[first]["root_id"] = "unexpected-root"
    with pytest.raises(ValueError, match="sixteen root lineages"):
        report.validate_plans(tasks, plans)


@pytest.mark.parametrize("change", ["seed", "draw", "protocol", "extra", "pair"])
def test_changed_population_rejected(change):
    tasks, plans = planned_population()
    if change == "seed":
        plans[1][0]["seeds"] = []
    elif change == "draw":
        plans[1][0]["G"] = 8
    elif change == "protocol":
        plans[1][0]["protocol"] = "different"
    elif change == "extra":
        plans[1].append(plans[1][0])
    else:
        plans[1][0]["qid"], plans[1][1]["qid"] = plans[1][1]["qid"], plans[1][0]["qid"]
    with pytest.raises(ValueError):
        report.validate_plans(tasks, plans)


def minimal_groups(root):
    task = dict(
        qid="q",
        world_id="w",
        root_id="root",
        family="TOPK",
        cell="IMPACT",
        variant=0,
        split=report.NEW_PANEL,
    )
    a = dict(
        qid="q",
        group_index=0,
        panel=report.NEW_PANEL,
        G=4,
        protocol="structured",
        seeds=[10, 11, 12, 13],
    )
    paths = []
    for arm in report.ARMS:
        step = 0 if arm == "PARENT" else 12
        group_id = f"CLC-E-b0-{arm}/s{step:03d}/g00000"
        records = [
            dict(
                qid="q",
                world_id="w",
                block=0,
                arm=arm,
                panel=report.NEW_PANEL,
                protocol="structured",
                draw=d,
                seed=10 + d,
                group_id=group_id,
                row_id=f"{group_id}/d{d:02d}",
                step=step,
                split=report.NEW_PANEL,
                generation_status="COMPLETE",
                full_text="invalid model answer",
                finish_reason="eos",
                completion_token_count=3,
                score=dict.fromkeys(report.NATIVE_METRICS, 0),
            )
            for d in range(4)
        ]
        path = root / "EVAL/b0" / arm / "shard0/groups/00000.json"
        write(path, dict(group_id=group_id, block=0, rows=records))
        paths.append(path)
    return {"q": task}, {0: [a]}, paths


def unknown_verifier(task, row):
    from sr_f4.c_local_credit import VERIFIER_VERSION

    return dict(verifier_version=VERIFIER_VERSION, **dict.fromkeys("PSOE", "UNKNOWN"))


def test_invalid_model_answers_keep_denominator_and_read_raw_once(tmp_path, monkeypatch):
    tasks, plans, paths = minimal_groups(tmp_path)
    original, calls = report.read, []

    def count(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(report, "read", count)
    values, coverage = report.scan_evaluations(tmp_path, tasks, plans, verifier=unknown_verifier)
    assert not coverage["issues"] and len(values) == 20
    assert len(calls) == len(paths) == len(set(calls)) == 5
    assert all(r["score"]["value_correct"] == 0 for r in values)
    assert all(
        r["root_id"] == "root" and "tokens" not in r and "full_text" not in r for r in values
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(seed=99),
        dict(arm="SELECT"),
        dict(step=7),
        dict(world_id="other"),
        dict(generation_status="TECHNICAL_INVALID"),
    ],
)
def test_wrong_generation_identity_blocks_complete(tmp_path, change):
    tasks, plans, paths = minimal_groups(tmp_path)
    group = read(paths[0])
    group["rows"][0].update(change)
    write(paths[0], group)
    values, coverage = report.scan_evaluations(tmp_path, tasks, plans, verifier=unknown_verifier)
    assert len(coverage["issues"]) == 1 and len(values) == 16


def test_missing_and_extra_group_cannot_be_ignored(tmp_path):
    tasks, plans, paths = minimal_groups(tmp_path)
    group = read(paths[0])
    write(paths[0].with_name("extra.json"), group)
    group["rows"].pop()
    write(paths[0], group)
    paths[1].unlink()
    _, coverage = report.scan_evaluations(tmp_path, tasks, plans, verifier=unknown_verifier)
    assert len(coverage["issues"]) == 3


def compact(block, arm, root, qid, draw, value, *, family="IMPACT", world=None):
    return dict(
        block=block,
        arm=arm,
        root_id=root,
        world_id=world or root,
        qid=qid,
        draw=draw,
        panel=report.NEW_PANEL,
        cell=family,
        family="TOPK",
        variant=0,
        score=dict.fromkeys(report.NATIVE_METRICS, value),
        semantic=dict.fromkeys("PSOE", "UNKNOWN"),
        truncated=False,
        completion_tokens=10,
    )


def test_shared_root_resampling_groups_different_impact_worlds():
    values = []
    for block in report.BLOCKS:
        for arm in report.ARMS:
            for variant in range(4):
                values.append(
                    compact(
                        block,
                        arm,
                        "root0",
                        f"q{variant}",
                        0,
                        int(arm == "SELECT" and block == 0),
                        world=f"impact{variant}",
                    )
                )
            values.append(compact(block, arm, "root1", "q4", 0, 0))
    result = report.summarize_stratum(values, name="new/all", panel=report.NEW_PANEL)
    primary = next(c for c in result["comparisons"] if c["primary"])
    assert primary["difference"] == pytest.approx(1 / 6)
    assert primary["roots"] == 2
    draws = np.random.default_rng(report.BOOTSTRAP_SEED).integers(0, 2, (2000, 2))
    expected = np.quantile(np.array([1 / 3, 0])[draws].mean(axis=1), [0.025, 0.975])
    assert primary["ci95"] == pytest.approx(expected)
    assert result["measurement"][0]["S"]["pass_given_known"] is None


def test_equal_family_root_query_draw_and_parent_weights():
    values = []
    for block in report.BLOCKS:
        for arm in report.ARMS:
            # Family A has one root, unequal query draw counts, half-success.
            for d in range(8):
                values.append(
                    compact(
                        block,
                        arm,
                        "a",
                        "a0",
                        d,
                        int(arm == "SELECT" and block == 0),
                        family="BINDING",
                    )
                )
            values.append(compact(block, arm, "a", "a1", 0, 0, family="BINDING"))
            # Family B has three zero roots: equal families, not pooled roots.
            for r in range(3):
                values.append(compact(block, arm, f"b{r}", f"b{r}", 0, 0, family="IMPACT"))
    result = report.summarize_stratum(values, name="new/all", panel=report.NEW_PANEL)
    assert next(c for c in result["comparisons"] if c["primary"])["difference"] == pytest.approx(
        1 / 12
    )


def test_unpaired_arm_root_rejected():
    values = [compact(b, a, "root", "q", 0, 0) for b in report.BLOCKS for a in report.ARMS]
    values.pop()
    with pytest.raises(ValueError, match="Unpaired"):
        report.summarize_stratum(values, name="new/all", panel=report.NEW_PANEL)


def mask_fixture():
    records = [dict(row_id="r", tokens=[1, 2, 3, 4])]
    saved = dict(
        evidence_advantages=[0.1],
        selection_advantages=[-0.2],
        record_masks=[
            dict(
                row_id="r",
                evidence_mask=[True, False, False, False],
                selection_mask=[False, True, True, False],
                diagnostics=dict(span_token_counts=dict(evidence=1, selection=2)),
            )
        ],
    )
    return records, saved


def test_token_mass_respects_separate_span_mean_and_boundary_exclusion():
    records, saved = mask_fixture()
    counts = report._mask_diagnostics(records, saved)
    assert counts["selection_mask_tokens"] == 2
    assert counts["selection_normalized_coefficient_l1_before_TIS"] == pytest.approx(0.2)
    assert counts["generated_tokens"] == 4
    # Zero mapped tokens abstain even when a row's auxiliary coefficient exists.
    saved["record_masks"][0]["selection_mask"] = [False] * 4
    saved["record_masks"][0]["diagnostics"]["span_token_counts"]["selection"] = 0
    counts = report._mask_diagnostics(records, saved)
    assert counts["selection_unmapped_nonzero_rows"] == 1
    assert counts["selection_normalized_coefficient_l1_before_TIS"] == 0


@pytest.mark.parametrize("defect", ["identity", "overlap", "length", "count"])
def test_saved_mask_corruption_rejected(defect):
    records, saved = mask_fixture()
    mask = saved["record_masks"][0]
    if defect == "identity":
        mask["row_id"] = "another"
    elif defect == "overlap":
        mask["selection_mask"] = [True, True, False, False]
    elif defect == "length":
        mask["selection_mask"].pop()
    else:
        mask["diagnostics"]["span_token_counts"]["selection"] = 3
    with pytest.raises(ValueError):
        report._mask_diagnostics(records, saved)


def test_completion_requires_every_registered_task_receipt(tmp_path):
    registry = [dict(task_id="CLC-REPORT", kind="c_local_report")]
    for b in report.BLOCKS:
        for arm in report.ARMS:
            if arm != "PARENT":
                registry.append(
                    dict(task_id=f"CLC-T-b{b}-{arm}", kind="c_local_train", block=b, arm=arm)
                )
            for shard in range(4):
                registry.append(
                    dict(
                        task_id=f"CLC-E-b{b}-{arm}-s{shard}",
                        kind="c_local_eval",
                        block=b,
                        arm=arm,
                        shard=shard,
                    )
                )
    write(tmp_path / "TASKS.json", registry)
    for r in registry[1:]:
        write(
            tmp_path / "tasks" / r["task_id"] / "COMPLETE.json",
            dict(status="COMPLETE", task_id=r["task_id"]),
        )
    assert not report.completion_inventory(tmp_path)["issues"]
    (tmp_path / "tasks/CLC-T-b0-SELECT/COMPLETE.json").unlink()
    issues = report.completion_inventory(tmp_path)["issues"]
    assert len(issues) == 1 and issues[0]["task_id"] == "CLC-T-b0-SELECT"


def test_report_incomplete_does_not_publish_partial_science(tmp_path):
    result = dict(status="INCOMPLETE_TECHNICAL", issues=[dict(reason="MISSING_GROUP")])
    report.write_report(tmp_path, result)
    text = (tmp_path / "REPORT_ZH.md").read_text()
    assert "不发布部分结果" in text and "选择分歧" not in text
    manifest = report.delivery_manifest(tmp_path, result["status"])
    assert not manifest["archive_created"]
    assert all("sha256" not in entry for entry in manifest["files"])


def test_unknown_semantics_are_not_scored_as_fail():
    values = [compact(b, a, "root", "q", 0, 0) for b in report.BLOCKS for a in report.ARMS]
    values += [copy.deepcopy(r) for r in values]
    for r in values[len(values) // 2 :]:
        r["draw"] = 1
        r["semantic"]["S"] = "PASS"
    result = report.summarize_stratum(values, name="new/all", panel=report.NEW_PANEL)
    s = result["measurement"][0]["S"]
    assert s["known_coverage"] == 0.5 and s["pass_all"] == 0.5 and s["pass_given_known"] == 1
    assert s["counts"]["UNKNOWN"] == 1 and s["counts"]["FAIL"] == 0


def frozen_fixture(root):
    from sr_f4 import c_local

    tasks = c_local.data.generate_tasks()
    old, _ = planned_population()
    for task in old.values():
        if task.get("pool") in ("TEST_ID", "CHARTQA"):
            tasks.append(dict(task, family="TOPK", variant=0))
    write_rows(root / "DATA/TASKS.jsonl", tasks)
    write_rows(root / "DATA/IMAGE_MANIFEST.jsonl", [])
    write(root / "FROZEN_INPUTS.json", dict(tasks=tasks, images=[]))
    for b in report.BLOCKS:
        write_rows(root / f"DATA/EVAL_GROUPS_b{b}.jsonl", c_local.build_eval_groups(tasks, b))
        write(
            root / f"DATA/TRAIN_PLAN_b{b}.json",
            {str(s): c_local.data.training_assignments(tasks, b, s) for s in range(1, 13)},
        )
    config = dict(
        parents={str(b): f"parent{b}" for b in report.BLOCKS},
        model_path="base",
        code_revision="test-revision",
    )
    write(root / "CONFIG.json", config)
    write(
        root / "FROZEN.json",
        dict(
            c_local.data.plan(),
            **c_local.protocol(),
            parent_references=config["parents"],
            model_path="base",
            code_revision=config["code_revision"],
            prelaunch_gate=c_local.prelaunch_gate(),
        ),
    )
    write(root / "PRELAUNCH_GATE.json", c_local.prelaunch_gate())
    write(root / "TASKS.json", c_local.task_matrix())
    for spec in c_local.task_matrix()[:-1]:
        write(
            root / "tasks" / spec["task_id"] / "COMPLETE.json",
            dict(status="COMPLETE", task_id=spec["task_id"]),
        )
    return tasks


@pytest.mark.parametrize("incomplete", [False, True])
def test_report_integrates_actual_frozen_plan_and_compact_analysis(
    tmp_path, monkeypatch, incomplete
):
    from sr_f4 import c_local

    monkeypatch.setattr(c_local, "prelaunch_gate", lambda: dict(status="PASS", fixture=True))
    tasks = frozen_fixture(tmp_path)
    index = {t["qid"]: t for t in tasks}
    values = []
    for b in report.BLOCKS:
        for assignment in c_local.build_eval_groups(tasks, b):
            task = index[assignment["qid"]]
            for arm in report.ARMS:
                for draw in range(assignment["G"]):
                    row = compact(
                        b,
                        arm,
                        task.get("root_id", task["qid"]),
                        task["qid"],
                        draw,
                        int(arm == "SELECT" and b != 1),
                    )
                    row.update(
                        panel=assignment["panel"],
                        cell=task.get("cell"),
                        family=task["family"],
                        variant=task.get("variant"),
                    )
                    if assignment["panel"] == "CHARTQA":
                        row.update(
                            score={report.CHART_METRICS[0]: 0, report.CHART_METRICS[1]: 1},
                            semantic={},
                        )
                    elif assignment["panel"] == report.OLD_PANEL:
                        row["semantic"] = {}
                    values.append(row)
    assert len(values) == 31970
    monkeypatch.setattr(
        report,
        "scan_evaluations",
        lambda *args: (
            values,
            dict(
                issues=[dict(reason="MISSING_GROUP")] if incomplete else [],
                extracted_answers=len(values),
                expected_answers=31970,
                expected_groups=8930,
                validated_group_files=8930,
            ),
        ),
    )
    monkeypatch.setattr(
        report,
        "training_mechanism",
        lambda *args: dict(issues=[], answers=18432, updates=144, strata={}),
    )
    outcome = report.report(tmp_path)
    summary = read(tmp_path / "ANALYSIS/SUMMARY.json")
    if incomplete:
        assert outcome["status"] == "INCOMPLETE_TECHNICAL"
        assert "comparisons" not in summary
    else:
        assert outcome["status"] == report.COMPLETE, summary["issues"]
        primary = next(c for c in summary["comparisons"] if c["primary"])
        assert primary["difference"] == pytest.approx(2 / 3)
        assert primary["roots"] == 96 and primary["families"] == 6
        assert any(c["difference"] == 0 for c in summary["comparisons"] if c["block"] == 1)
        assert (
            len(
                {
                    r["stratum"]
                    for r in summary["endpoints"]
                    if r["stratum"].startswith("new/family/")
                }
            )
            == 6
        )
    assert "comparisons" not in outcome
    assert not read(tmp_path / "ANALYSIS/DELIVERY_MANIFEST.json")["archive_created"]


def conditional_fixture(*, within_same_j=True, applied=False, changed=False, p="PASS"):
    destinations = [0, 1]
    strata = (
        [
            dict(
                j=0,
                p=p,
                destination_indices=destinations,
                source_indices=[1, 0] if changed else destinations,
            )
        ]
        if within_same_j
        else [dict(j=i, p=p, destination_indices=[i], source_indices=[i]) for i in range(2)]
    )
    return dict(
        active=True,
        labels=[dict(S="PASS"), dict(S="FAIL")],
        shuffle=dict(applied=applied, within_j_p_strata=strata),
    )


def test_conditional_swappability_separates_cross_j_mixture_and_p_fail():
    assert (
        report._conditional_swappability(conditional_fixture(within_same_j=False))[
            "conditional_swappable_groups"
        ]
        == 0
    )
    assert (
        report._conditional_swappability(conditional_fixture(p="FAIL"))[
            "conditional_swappable_groups"
        ]
        == 0
    )


def test_conditional_swappability_independent_of_realized_noop_and_actual_arm():
    potential = report._conditional_swappability(conditional_fixture())
    noop = report._conditional_swappability(conditional_fixture(applied=True))
    changed = report._conditional_swappability(conditional_fixture(applied=True, changed=True))
    for result in (potential, noop, changed):
        assert result["conditional_swappable_groups"] == 1
        assert result["conditional_swappable_strata"] == 1
        assert result["conditional_swappable_rows"] == 2
    assert potential["conditional_realized_changed_rows"] == 0
    assert noop["conditional_realized_changed_rows"] == 0
    assert changed["conditional_realized_changed_rows"] == 2
    assert changed["conditional_realized_changed_groups"] == 1


def test_actual_update_objective_preserves_signed_values_and_rejects_nonfinite():
    receipt = dict(zip(report.OBJECTIVE_METRICS, [-0.2, 0.1, -0.01, 0.05, -0.109], strict=True))
    values = report._objective_diagnostics(receipt)
    assert values["objective_updates"] == 1
    assert values["objective/native_policy_sum"] == -0.2
    assert values["objective/evidence_policy_sum"] == 0.1
    for bad in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="objective diagnostic"):
            report._objective_diagnostics(dict(receipt, loss=bad))
