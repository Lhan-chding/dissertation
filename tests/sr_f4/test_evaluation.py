"""Focused tests for fixed panel coverage and the world-level statistical unit."""

import json
from collections import Counter

import pytest

from sr_f3_ce.prompts import CONTRACT_V2
from sr_f4.evaluation import (
    ARMS,
    build_a_eval_groups,
    evaluation_assignments,
    model_input_for,
    shard_groups,
)
from sr_f4.reporting import coverage, load_evaluations, measurement_summary, paired_world_comparison


def inventory():
    tasks = []
    for family in ("CROSS", "INTERVAL", "THRESHOLD", "TOPK"):
        for i in range(32):
            for chart in ("grouped_bar", "line"):
                for variant in range(8):
                    root = f"TEST_ID_{family}_{i:04d}"
                    tasks.append(
                        dict(
                            qid=f"{root}_{chart}_v{variant}",
                            root_id=root,
                            family=family,
                            chart=chart,
                            variant=variant,
                            mode="repair" if variant else "legacy0",
                            pool="TEST_ID",
                        )
                    )
    for pool, count in (
        ("TEST_COMPOSITION", 256),
        ("TEST_ARITY", 128),
        ("TEST_LANGUAGE", 128),
        ("TEST_RENDER", 128),
        ("B_CONFIRM", 56),
    ):
        tasks.extend(dict(qid=f"{pool}_{i}", pool=pool) for i in range(count))
    return tasks


def test_frozen_counts_shared_random_stream_and_disjoint_shards():
    tasks = inventory()
    b1 = build_a_eval_groups(tasks, 1, arm="LZ")
    b2 = build_a_eval_groups(list(reversed(tasks)), 2, arm="J_UNIFORM_NEW_REPEAT")
    assert b1 == b2
    assert sum(row["G"] for row in b1) == 11232
    by_panel = Counter()
    for row in b1:
        by_panel[row["panel"]] += row["G"]
    assert by_panel == {
        "TEST_ID": 8192,
        "TEST_COMPOSITION": 1024,
        "TEST_ARITY": 512,
        "B_CONFIRM": 224,
        "NATURAL_WORK": 512,
        "ANSWER_ONLY": 256,
        "TEST_LANGUAGE": 256,
        "TEST_RENDER": 256,
    }
    halves = [shard_groups(b1, i, 2) for i in range(2)]
    assert not {r["group_index"] for r in halves[0]} & {r["group_index"] for r in halves[1]}
    assert sorted(halves[0] + halves[1], key=lambda r: r["group_index"]) == b1
    with pytest.raises(ValueError, match="explicit registered"):
        build_a_eval_groups(tasks, 0)
    with pytest.raises(ValueError, match="ChartQA"):
        build_a_eval_groups(tasks, 0, arm="LZ")
    external = [dict(qid=f"CHARTQA_{i}", pool="CHARTQA") for i in range(2500)]
    b0 = build_a_eval_groups(tasks, 0, external, arm="LZ")
    assert sum(row["G"] for row in b0) == 13732
    assert b0[: len(b1)] == b1
    assert b0 == build_a_eval_groups(tasks, 0, external, arm="J_UNIFORM_NEW_REPEAT")
    parent = build_a_eval_groups(tasks, 0, arm="PARENT")
    assert sum(row["G"] for row in parent) == 11456
    assert not any(row["panel"] == "CHARTQA" for row in parent)
    assert parent == build_a_eval_groups(tasks, 1, external, arm="PARENT")
    for parent_group, a_group in zip(parent, b1, strict=True):
        if parent_group["panel"] == "B_CONFIRM":
            assert parent_group["G"] == 8 and a_group["G"] == 4
            assert parent_group["seeds"][:4] == a_group["seeds"]
        else:
            assert parent_group == a_group
    schedules = {
        (block, arm): build_a_eval_groups(tasks, block, external, arm=arm)
        for block in range(3)
        for arm in ARMS
    }
    assert sum(g["G"] for groups in schedules.values() for g in groups) == 106760
    assert (
        sum(
            g["G"]
            for (block, arm), groups in schedules.items()
            if arm == "PARENT"
            for g in groups
            if g["panel"] == "B_CONFIRM"
        )
        == 1344
    )
    parent_values = [
        dict(
            block=0, arm="PARENT", panel=g["panel"], protocol=g["protocol"], qid=g["qid"], draw=draw
        )
        for g in parent
        for draw in range(g["G"])
    ]
    observed = coverage(parent_values, tasks, external)
    assert observed[0] == dict(
        block=0,
        arm="PARENT",
        expected=11456,
        actual=11456,
        missing=0,
        unexpected=0,
        status="COMPLETE",
    )
    assert sum(row["expected"] for row in observed) == 106760
    answer = [r for r in b1 if r["panel"] == "ANSWER_ONLY"]
    by_qid = {row["qid"]: row for row in tasks}
    assert Counter(by_qid[r["qid"]]["family"] for r in answer) == dict.fromkeys(
        ("CROSS", "INTERVAL", "THRESHOLD", "TOPK"), 16
    )
    assert Counter(by_qid[r["qid"]]["chart"] for r in answer) == {"grouped_bar": 32, "line": 32}


def test_input_does_not_accept_audit_sidecars():
    source = dict(
        qid="q",
        pool="TEST_ID",
        chart="line",
        render_style="main",
        prompt="What is the sum?\n" + CONTRACT_V2,
    )
    image = dict(image_file="images/q.png", image_sha256="a" * 64)
    out = model_input_for(source, image, "natural_work")
    assert set(out) == {"prompt", "image_file", "image_sha256"}
    assert "<final>VALUE</final>" in out["prompt"]
    with pytest.raises(ValueError, match="audit task"):
        model_input_for(dict(source, gold_answer=42), image)


def test_pipelined_phases_preserve_every_group_seed_draw_and_shard():
    tasks = inventory()
    external = [dict(qid=f"CHARTQA_{i}", pool="CHARTQA") for i in range(2500)]
    totals = Counter()
    for block in range(3):
        for arm in ARMS:
            groups = build_a_eval_groups(tasks, block, external, arm=arm)
            for shard in range(2):
                phases = {}
                for phase in ("regression", "confirm"):
                    phases[phase] = evaluation_assignments(
                        groups, dict(phase=phase, shard=shard, shards=2)
                    )
                    totals[phase] += sum(g["G"] for g in phases[phase])
                assert not {g["group_index"] for g in phases["regression"]} & {
                    g["group_index"] for g in phases["confirm"]
                }
                assert all(g["panel"] != "B_CONFIRM" for g in phases["regression"])
                assert all(g["panel"] == "B_CONFIRM" for g in phases["confirm"])
                assert sorted(
                    phases["regression"] + phases["confirm"], key=lambda g: g["group_index"]
                ) == shard_groups(groups, shard, 2)
    assert totals == dict(regression=104072, confirm=2688)


def test_regression_runs_early_but_confirmation_remains_guarded(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from sr_f4 import evaluation, run, runtime
    from sr_f4.common import read, write

    groups = [
        dict(qid="old0", panel="TEST_ID", group_index=0, G=4),
        dict(qid="new", panel="B_CONFIRM", group_index=1, G=8),
        dict(qid="old2", panel="TEST_ARITY", group_index=2, G=4),
    ]
    tasks = {g["qid"]: dict(qid=g["qid"]) for g in groups}
    monkeypatch.setattr(run, "task_index", lambda root: tasks)
    monkeypatch.setattr(run, "image_index", lambda root: {key: {} for key in tasks})
    monkeypatch.setattr(evaluation, "build_a_eval_groups", lambda *args, **kwargs: groups)
    loaded = []

    def load(root, block, task_id):
        loaded.append(task_id)
        return SimpleNamespace(identity={}), None, None

    monkeypatch.setattr(runtime, "load", load)
    calls = []

    def group(root, runtime, task, image, assignment, directory, **kwargs):
        calls.append((task["qid"], kwargs["task_id"], kwargs["allow_confirm"]))
        return [{}] * assignment["G"]

    monkeypatch.setattr(run, "group", group)
    write(tmp_path / "CONFIG.json", {})
    original_id = "E-b0-PARENT-s0"
    spec = dict(task_id=original_id, block=0, arm="PARENT", shard=0, shards=1)
    regression = run.run_eval(tmp_path, dict(spec, phase="regression"))
    assert regression["answers"] == 8
    assert calls == [("old0", original_id, False), ("old2", original_id, False)]
    assert not (tmp_path / "CONFIRM_OPENED.json").exists()
    confirm = dict(spec, task_id="EC-b0-PARENT-s0", evaluation_task_id=original_id, phase="confirm")
    with pytest.raises(PermissionError, match="requires completed A-"):
        run.run_eval(tmp_path, confirm)
    assert loaded == [original_id]
    assert not (tmp_path / "CONFIRM_OPENED.json").exists()
    for block in range(3):
        for arm in ARMS:
            if arm != "PARENT":
                write(tmp_path / f"tasks/A-b{block}-{arm}/COMPLETE.json", dict(status="COMPLETE"))
    with pytest.raises(PermissionError, match="B3-replay"):
        run.run_eval(tmp_path, confirm)
    write(tmp_path / "tasks/B3-replay/COMPLETE.json", dict(status="COMPLETE"))
    write(tmp_path / "C_GATE_REPORT.json", dict(decision="BLOCK_C"))
    result = run.run_eval(tmp_path, confirm)
    assert result["answers"] == 8
    assert calls[-1] == ("new", original_id, True)
    directory = tmp_path / "EVAL/b0/PARENT/shard0"
    assert read(directory / "PROGRESS_regression.json")["answers"] == 8
    assert read(directory / "PROGRESS_confirm.json")["answers"] == 8
    assert read(tmp_path / "CONFIRM_OPENED.json")["decisions_frozen"] is True


def paired_fixture(draws=1):
    result = []
    for block in range(3):
        for world in range(4):
            for prompt in range(2):
                for draw in range(draws):
                    for arm in ("J_UNIFORM_NEW_REPEAT", "LZ"):
                        value = int(arm == "LZ" and world < 2)
                        result.append(
                            dict(
                                block=block,
                                arm=arm,
                                panel="B_CONFIRM",
                                protocol="structured",
                                qid=f"q{world}_{prompt}",
                                draw=draw,
                                world_id=f"w{world}",
                                purpose="selection",
                                family="INTERVAL",
                                score={"value_correct": value},
                            )
                        )
    return result


def test_bootstrap_unit_is_world_not_answer_and_preserves_blocks():
    one = paired_world_comparison(
        paired_fixture(), "B_CONFIRM", "LZ", "J_UNIFORM_NEW_REPEAT", replicates=200
    )
    many = paired_world_comparison(
        paired_fixture(8), "B_CONFIRM", "LZ", "J_UNIFORM_NEW_REPEAT", replicates=200
    )
    assert one["status"] == many["status"] == "COMPLETE"
    assert one["worlds"] == many["worlds"] == 4
    assert one["difference"] == many["difference"] == 0.5
    assert one["paired_world_ci95"] == many["paired_world_ci95"]
    assert len(many["per_block"]) == 3
    missing = paired_fixture()[:-1]
    assert (
        paired_world_comparison(missing, "B_CONFIRM", "LZ", "J_UNIFORM_NEW_REPEAT")["status"]
        == "INCOMPLETE_PAIRED_COVERAGE"
    )


def test_b_confirm_parent_comparison_uses_only_four_shared_draws():
    values = paired_fixture(4)
    parents = []
    for row in paired_fixture(8):
        if row["arm"] == "J_UNIFORM_NEW_REPEAT":
            parents.append(dict(row, arm="PARENT", score={"value_correct": int(row["draw"] >= 4)}))
    result = paired_world_comparison(values + parents, "B_CONFIRM", "LZ", "PARENT", replicates=200)
    assert result["status"] == "COMPLETE"
    assert result["difference"] == 0.5
    assert result["answers_per_model"] == 8 * 4
    assert result["paired_draw_indices"] == [0, 1, 2, 3]
    assert paired_world_comparison([], "CHARTQA", "LZ", "PARENT")["status"] == "NOT_REGISTERED"


def test_b_confirmation_measurement_reuses_all_eight_parent_draws_separately(tmp_path):
    directory = tmp_path / "REPLAY"
    directory.mkdir()
    (directory / "SUMMARY.json").write_text(
        json.dumps(
            dict(
                status="REPLAY_COMPLETE",
                answers=10,
                coverage=[dict(source="B2_DEV", dimension="P", status="PASS", count=10)],
            )
        )
    )
    parents = [
        dict(
            block=block,
            arm="PARENT",
            panel="B_CONFIRM",
            qid=f"q{q}",
            draw=draw,
            world_id=f"w{q // 2 if q < 48 else q - 24}",
            family="INTERVAL",
            purpose="selection",
            score={"J": int(draw < 4)},
            semantic_status=dict.fromkeys("PSOE", "PASS" if draw < 4 else "FAIL"),
        )
        for block in range(3)
        for q in range(56)
        for draw in range(8)
    ]
    values = parents + [dict(row, arm="LZ") for row in parents if row["draw"] < 4]
    result = measurement_summary(tmp_path, values)
    assert result["status"] == "MEASUREMENT_COMPLETE"
    assert result["answers"] == 1354 and result["development_and_historical_answers"] == 10
    confirm = result["confirmation"]
    assert confirm["answers"] == confirm["expected_answers"] == 1344
    assert confirm["groups"] == 168 and confirm["draws_per_prompt"] == 8
    assert [row["worlds"] for row in confirm["per_block"]] == [32, 32, 32]
    assert all(row["counts"] == {"PASS": 672, "FAIL": 672} for row in confirm["coverage"])
    assert {row["source"] for row in result["coverage"]} == {"B2_DEV", "B2_CONFIRM_PARENT"}
    assert (
        measurement_summary(tmp_path, parents[:-1])["confirmation"]["status"]
        == "INCOMPLETE_TECHNICAL"
    )


@pytest.mark.parametrize("workers", [1, 8])
def test_loader_ignores_partial_chunks_and_preserves_score_denominators(tmp_path, workers):
    directory = tmp_path / "EVAL/b0/PARENT/shard0/groups"
    directory.mkdir(parents=True)
    record = dict(
        qid="q",
        block=0,
        arm="PARENT",
        draw=0,
        panel="NATURAL_WORK",
        protocol="natural_work",
        score={"J": 0, "value_correct": 1, "certificate_truth": 0, "finish": 1},
        tokens=[1, 2],
    )
    (directory / "00000.json").write_text(json.dumps(dict(rows=[record])))
    (directory / "00000.chunk0.json").write_text(json.dumps([record]))
    loaded = load_evaluations(
        tmp_path, {"q": {"root_id": "w", "family": "INTERVAL"}}, workers=workers
    )
    assert len(loaded) == 1 and "J" not in loaded[0]["score"]
    assert loaded[0]["score"]["value_correct"] == 1
    (directory / "00001.json").write_text(json.dumps(dict(rows=[record])))
    with pytest.raises(ValueError, match="Duplicate"):
        load_evaluations(
            tmp_path, {"q": {"root_id": "w", "family": "INTERVAL"}}, workers=workers
        )


def test_parallel_loader_preserves_all_normalized_fields_and_file_order(tmp_path):
    directory = tmp_path / "EVAL"
    directory.mkdir()
    tasks = {}
    for i in range(35):
        qid = f"q{i}"
        tasks[qid] = dict(world_id=f"w{i}", family="TOPK", variant=i, purpose="confirm")
        record = dict(
            qid=qid, block=i % 3, arm="LZ", draw=0, panel="B_CONFIRM",
            protocol="structured", score=dict(J=i % 2, value_correct=1, work_present=True),
            tokens=[1, 2, 3], completion_token_count=7, finish_reason="length",
            semantic={"dimensions": {"S": {"status": "UNKNOWN"}}},
        )
        if i % 2:
            (directory / f"{i:05d}.jsonl").write_text(json.dumps(record) + "\n")
        else:
            (directory / f"{i:05d}.json").write_text(json.dumps(dict(rows=[record])))
    serial = load_evaluations(tmp_path, tasks, workers=1)
    assert load_evaluations(tmp_path, tasks, workers=8) == serial
    assert [r["qid"] for r in serial] == [
        f"q{i}" for i in [*range(1, 35, 2), *range(0, 35, 2)]
    ]
    assert all(r["completion_tokens"] == 7 and r["truncated"] for r in serial)
    assert all(r["semantic_status"] == {"S": "UNKNOWN"} for r in serial)


def test_parallel_reads_overlap_but_prefetch_is_bounded(monkeypatch):
    from threading import Event, Lock

    from sr_f4 import reporting

    second_started = Event()
    lock = Lock()
    started = []

    def reader(path):
        with lock:
            started.append(path)
        if path == 0:
            assert second_started.wait(5)
        if path == 1:
            second_started.set()
        return [path]

    monkeypatch.setattr(reporting, "_read_evaluation_file", reader)
    stream = reporting._ordered_evaluation_files(list(range(100)), 4)
    assert next(stream) == (0, [0])
    assert len(started) <= 8
    assert list(stream) == [(i, [i]) for i in range(1, 100)]


def test_chartqa_nested_metrics_remain_separate(tmp_path):
    directory = tmp_path / "EVAL/b0/LZ/shard0/groups"
    directory.mkdir(parents=True)
    record = dict(
        qid="q",
        block=0,
        arm="LZ",
        draw=0,
        panel="CHARTQA",
        protocol="chartqa_plain",
        tokens=[1],
        score={
            "historical_code": {"relaxed_accuracy": 0, "exact_match": 0},
            "v2": {"relaxed_accuracy": 1, "exact_match": 0},
            "typed_audit": {"typed_score": None, "status": "UNRESOLVED"},
            "finish": 1,
        },
    )
    (directory / "00000.json").write_text(json.dumps(dict(rows=[record])))
    loaded = load_evaluations(tmp_path, {"q": {"root_id": "same_chart_image"}})
    assert loaded[0]["score"]["v2.relaxed_accuracy"] == 1
    assert loaded[0]["score"]["historical_code.relaxed_accuracy"] == 0
    assert "typed_audit.typed_score" not in loaded[0]["score"]
    assert "value_correct" not in loaded[0]["score"]
