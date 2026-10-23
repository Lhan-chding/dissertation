"""Pilot inventory and conditional inference regressions; no model calls."""

import numpy as np
import pytest

from sr_f4 import c_pilot
from sr_f4 import c_pilot_data as data
from sr_f4 import c_pilot_reporting as report
from sr_f4.common import read, write, write_rows


def planned_tasks():
    tasks = data.generate_tasks()
    for family in ("CROSS", "INTERVAL", "THRESHOLD", "TOPK"):
        for variant in range(8):
            for root in range(2):
                for chart in ("grouped_bar", "line"):
                    world = f"old-{family}-{variant}-{root}"
                    tasks.append(dict(qid=world + "-" + chart, world_id=world, root_id=world,
                                      family=family, variant=variant, chart=chart,
                                      split="OLD_REGRESSION", pool="TEST_ID"))
    tasks += [dict(qid=f"chart-{i}", world_id=f"chart-{i}", pool="CHARTQA",
                   split="C_FP_CHARTQA_R1") for i in range(250)]
    return tasks


def test_fixed_inventory_uses_pool_for_old_split_and_allows_block_specific_rng():
    tasks = planned_tasks()
    plans = {b: c_pilot.build_eval_groups(tasks, b) for b in range(3)}
    assert plans[0][0]["seeds"] != plans[1][0]["seeds"]
    assert report.validate_plans({t["qid"]: t for t in tasks}, plans) == 24290
    plans[2].pop()
    with pytest.raises(ValueError, match="Wrong panel inventory"):
        report.validate_plans({t["qid"]: t for t in tasks}, plans)


def test_wrong_cell_inventory_rejected_before_results():
    tasks = planned_tasks()
    plans = {b: c_pilot.build_eval_groups(tasks, b) for b in range(3)}
    index = {t["qid"]: t for t in tasks}
    index[plans[0][0]["qid"]]["cell"] = "unregistered"
    with pytest.raises(ValueError, match="TOPK evaluation identity"):
        report.validate_plans(index, plans)


def minimal_groups(tmp_path):
    task = dict(qid="q", world_id="w", family="TOPK", cell=data.CELLS[0],
                variant=0, split=data.TEST_SPLIT)
    assignment = dict(qid="q", group_index=0, panel=data.TEST_SPLIT,
                      G=4, protocol="structured", seeds=[10, 11, 12, 13])
    paths = []
    for arm in report.ARMS:
        step = 0 if arm == "PARENT" else 16
        group_id = f"CPF-E-b0-{arm}/s{step:03d}/g00000"
        records = [dict(qid="q", world_id="w", block=0, arm=arm,
                        panel=data.TEST_SPLIT, protocol="structured", draw=draw,
                        seed=10 + draw, group_id=group_id, row_id=f"{group_id}/d{draw:02d}",
                        step=step, split=data.TEST_SPLIT, generation_status="COMPLETE",
                        full_text="malformed model answer remains in the denominator",
                        finish_reason="eos", completion_token_count=3,
                        score=dict.fromkeys(report.NATIVE_METRICS, 0)) for draw in range(4)]
        path = tmp_path / "EVAL/b0" / arm / "shard0/groups/00000.json"
        write(path, dict(group_id=group_id, block=0, rows=records))
        paths.append(path)
    return {"q": task}, {0: [assignment]}, paths


def fake_verifier(task, row, score):
    return dict(verifier_version=report.VERSION,
                dimensions={d: dict(status="UNKNOWN") for d in "PS"})


def test_invalid_model_answers_keep_all_zero_rows_and_raw_read_once(tmp_path, monkeypatch):
    tasks, plans, paths = minimal_groups(tmp_path)
    original, calls = report.read, []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(report, "read", counted)
    values, coverage = report.scan_evaluations(tmp_path, tasks, plans, verifier=fake_verifier)
    assert coverage["issues"] == []
    assert len(values) == 20 and len(calls) == len(paths) == 5
    assert all(r["score"]["value_correct"] == 0 for r in values)
    assert all("tokens" not in r and "full_text" not in r for r in values)


@pytest.mark.parametrize("change", [dict(seed=99), dict(arm="SELECT"),
                                    dict(generation_status="TECHNICAL_INVALID"),
                                    dict(world_id="another-world"), dict(step=7)])
def test_wrong_or_technical_response_identity_blocks_complete(tmp_path, change):
    tasks, plans, paths = minimal_groups(tmp_path)
    saved = read(paths[0])
    saved["rows"][0].update(change)
    write(paths[0], saved)
    _, coverage = report.scan_evaluations(tmp_path, tasks, plans, verifier=fake_verifier)
    assert coverage["issues"]


def test_missing_extra_draw_and_unexpected_group_are_not_silently_excluded(tmp_path):
    tasks, plans, paths = minimal_groups(tmp_path)
    saved = read(paths[0])
    saved["rows"].pop()
    write(paths[0], saved)
    write(paths[0].with_name("extra.json"), saved)
    paths[1].unlink()
    _, coverage = report.scan_evaluations(tmp_path, tasks, plans, verifier=fake_verifier)
    assert len(coverage["issues"]) == 3
    assert any(i["reason"] == "UNEXPECTED_GROUP_FILE" for i in coverage["issues"])


def compact_row(block, arm, world, qid, draw, value, *, panel=data.TEST_SPLIT):
    return dict(block=block, arm=arm, world_id=world, qid=qid, draw=draw,
                panel=panel, cell=data.CELLS[0], family="TOPK", variant=0,
                score=dict.fromkeys(report.NATIVE_METRICS, value),
                semantic=dict(P="PASS", S="UNKNOWN"), truncated=False,
                completion_tokens=10)


def test_world_query_draw_hierarchy_and_equal_parent_weight():
    values = []
    for block in range(3):
        for arm in report.ARMS:
            # Deliberately unequal draws/query: q0 has 8 successes; q1 one failure.
            for draw in range(8):
                values.append(compact_row(block, arm, "w0", "q0", draw,
                                          int(arm == "SELECT" and block == 0)))
            values.append(compact_row(block, arm, "w0", "q1", 0, 0))
            values.append(compact_row(block, arm, "w1", "q2", 0, 0))
    result = report.summarize_stratum(values, name="new/all", panel=data.TEST_SPLIT)
    primary = next(c for c in result["comparisons"] if c["primary"])
    assert primary["difference"] == pytest.approx(1 / 12)
    # Shared world bootstrap: world deltas are [1/6,0]. Resampling blocks
    # independently would incorrectly shrink this distribution.
    draws = np.random.default_rng(report.BOOTSTRAP_SEED).integers(0, 2, (2000, 2))
    expected = np.quantile(np.array([1/6, 0])[draws].mean(axis=1), [0.025, 0.975])
    assert primary["ci95"] == pytest.approx(expected)
    measured = result["measurement"][0]["S"]
    assert measured["all_denominator"] == 10
    assert measured["known_denominator"] == 0 and measured["pass_given_known"] is None


def test_missing_arm_world_cannot_enter_paired_interval():
    values = [compact_row(b, a, "w", "q", 0, 0)
              for b in range(3) for a in report.ARMS]
    values.pop()
    with pytest.raises(ValueError, match="Unpaired"):
        report.summarize_stratum(values, name="new/all", panel=data.TEST_SPLIT)


def freeze_fixture(root):
    tasks = planned_tasks()
    write_rows(root / "DATA/TASKS.jsonl", tasks)
    write_rows(root / "DATA/IMAGE_MANIFEST.jsonl", [])
    write(root / "FROZEN_INPUTS.json", dict(tasks=tasks, images=[]))
    for b in range(3):
        write_rows(root / f"DATA/EVAL_GROUPS_b{b}.jsonl", c_pilot.build_eval_groups(tasks, b))
        write(root / f"DATA/TRAIN_PLAN_b{b}.json",
              {str(s): data.training_assignments(tasks, b, s) for s in range(1, 17)})
    config = dict(parents={str(b): f"server-parent-{b}" for b in range(3)},
                  model_path="server-model", code_revision="frozen-revision")
    write(root / "CONFIG.json", config)
    write(root / "FROZEN.json", dict(data.plan(), **c_pilot.protocol(),
                                    parent_references=config["parents"],
                                    model_path=config["model_path"],
                                    code_revision=config["code_revision"]))
    write(root / "TASKS.json", c_pilot.task_matrix())
    for spec in c_pilot.task_matrix():
        if spec["kind"] != "c_pilot_report":
            write(root / "tasks" / spec["task_id"] / "COMPLETE.json",
                  dict(status="COMPLETE", task_id=spec["task_id"]))
    return tasks


def test_missing_completion_receipt_remains_technical_incomplete(tmp_path):
    freeze_fixture(tmp_path)
    receipt = tmp_path / "tasks/CPF-T-b0-J/COMPLETE.json"
    receipt.unlink()
    inventory = report.completion_inventory(tmp_path)
    assert len(inventory["issues"]) == 1
    assert inventory["issues"][0]["task_id"] == "CPF-T-b0-J"


def test_report_actual_frozen_plan_integration_and_lightweight_return(tmp_path, monkeypatch):
    tasks = freeze_fixture(tmp_path)
    index = {t["qid"]: t for t in tasks}
    values = []
    for block in range(3):
        for a in c_pilot.build_eval_groups(tasks, block):
            t = index[a["qid"]]
            for arm in report.ARMS:
                for draw in range(a["G"]):
                    value = int(arm == "SELECT" and block != 1)
                    r = compact_row(block, arm, t["world_id"], t["qid"], draw, value,
                                    panel=a["panel"])
                    r.update(cell=t.get("cell"), family=t.get("family", "EXTERNAL"),
                             variant=t.get("variant"))
                    if a["panel"] == "CHARTQA":
                        r.update(score={report.CHART_METRICS[0]: 0, report.CHART_METRICS[1]: 1},
                                 semantic={})
                    values.append(r)
    assert len(values) == 24290
    monkeypatch.setattr(report, "scan_evaluations", lambda *args: (values, dict(
        issues=[], expected_groups=7010, validated_group_files=7010,
        extracted_answers=24290, expected_answers=24290)))
    monkeypatch.setattr(report, "training_mechanism", lambda *args: dict(
        issues=[], answers=24576, strata={}, positive_descriptors_overlap=True))
    write(tmp_path / "C_GATE_REPORT.json", dict(status="BLOCK_C"))
    outcome = report.report(tmp_path)
    assert outcome["status"] == report.COMPLETE
    assert "comparisons" not in outcome  # compact pool COMPLETE receipt
    result = read(tmp_path / "ANALYSIS/SUMMARY.json")
    primary = next(c for c in result["comparisons"] if c["primary"])
    assert primary["difference"] == pytest.approx(2 / 3)
    assert any(c["difference"] == 0 for c in result["comparisons"] if c["block"] == 1)
    strata = {r["stratum"] for r in result["endpoints"]}
    assert len({s for s in strata if s.startswith("new/cell/")}) == 4
    assert len({s for s in strata if s.startswith("old/family_variant/")}) == 32
    assert read(tmp_path / "C_GATE_REPORT.json") == {"status": "BLOCK_C"}
    manifest = read(tmp_path / "ANALYSIS/DELIVERY_MANIFEST.json")
    assert not manifest["archive_created"]
    assert all("sha256" not in f for f in manifest["files"])
    assert "主要发现" in (tmp_path / "REPORT_ZH.md").read_text()


def test_report_does_not_publish_partial_scientific_comparisons(tmp_path, monkeypatch):
    freeze_fixture(tmp_path)
    monkeypatch.setattr(report, "scan_evaluations", lambda *args: ([], dict(
        issues=[dict(reason="MISSING_GROUP")], extracted_answers=0)))
    monkeypatch.setattr(report, "training_mechanism", lambda *args: dict(
        issues=[], answers=24576, strata={}))
    outcome = report.report(tmp_path)
    assert outcome["status"] == "INCOMPLETE_TECHNICAL"
    assert "comparisons" not in read(tmp_path / "ANALYSIS/SUMMARY.json")
    assert "不发布部分结果" in (tmp_path / "REPORT_ZH.md").read_text()
