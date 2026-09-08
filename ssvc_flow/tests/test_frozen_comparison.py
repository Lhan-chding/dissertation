"""Fixed support cells preserve overall populations and label missing overlap."""

import copy

import pytest


def subtask(model="qwen35_9b", pairs=((0.0, 0.0),), track="N", data_hash="data"):
    prompts = {
        f"p{i}": {
            "base_scene_id": f"s{i}",
            "interface": "SYMBOLIC_FRESH",
            "rollout_count": 16,
            "pooled": {"pX": px, "v": v, "qX": px / v if v else None, "pS": 0.0},
        }
        for i, (px, v) in enumerate(pairs)
    }
    return {
        "model_key": model,
        "track": track,
        "comparison_identity": {"data_hash": data_hash},
        "path": f"P3/{model}/{track}",
        "metrics": {
            "track": track,
            "overall": {"prompt_count": len(prompts), "unaltered_field": [1, 2]},
            "by_prompt": prompts,
            "greedy": {"overall": {"prompt_count": len(prompts), "pX": 1}},
        },
    }


def test_predeclared_bin_boundaries_include_endpoints_once():
    from src.frozen_comparison import build_support_comparison

    result = build_support_comparison(
        [subtask(pairs=((0, 0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1, 1)))]
    )
    rows = [
        row for row in result["rows"] if row["model_key"] == "qwen35_9b" and row["prompt_count"]
    ]
    assert [row["prompt_count"] for row in rows] == [1, 1, 1, 2]
    assert [row["pX_bin"] for row in rows] == ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1]"]
    assert result["bin_edges"] == [0, 0.25, 0.5, 0.75, 1]


def test_overall_populations_unchanged_and_greedy_never_enters_cells():
    from src.frozen_comparison import build_support_comparison

    task = subtask(pairs=((0.25, 0.5), (0, 0)))
    before = copy.deepcopy(task)
    result = build_support_comparison([task])
    assert task == before
    assert result["overall"][0]["sampling"] == task["metrics"]["overall"]
    assert result["overall"][0]["greedy"] == task["metrics"]["greedy"]["overall"]
    assert sum(row["prompt_count"] for row in result["rows"]) == 2
    empty = next(row for row in result["rows"] if row["prompt_count"] == 0)
    assert empty["mean_prompt_pX"] is None
    undefined = next(
        row for row in result["rows"] if row["prompt_count"] and row["mean_prompt_v"] == 0
    )
    assert undefined["mean_prompt_qX"] is None
    assert undefined["qX_NA_fraction"] == 1


def test_common_cells_require_all_three_models_and_matching_data():
    from src.frozen_comparison import build_support_comparison

    tasks = [
        subtask(model=model, pairs=((0.25, 0.5),))
        for model in ("qwen25vl_3b", "qwen35_9b", "qwen25vl_7b")
    ]
    result = build_support_comparison(tasks)
    observed = [row for row in result["rows"] if row["prompt_count"]]
    assert all(row["coverage_status"] == "COMMON_SUPPORT" for row in observed)
    assert all(row["common_prompt_count"] == 1 for row in observed)
    tasks[-1]["comparison_identity"]["data_hash"] = "different"
    changed = build_support_comparison(tasks)
    assert all(row["coverage_status"] == "INCOMPARABLE_DATA_OR_PROTOCOL" for row in changed["rows"])


def test_missing_model_or_nonoverlapping_cell_is_explicit():
    from src.frozen_comparison import build_support_comparison

    result = build_support_comparison(
        [subtask("qwen25vl_3b", ((0, 0),)), subtask("qwen35_9b", ((1, 1),))]
    )
    assert all(row["coverage_status"] == "NONCOMMON_SUPPORT" for row in result["rows"])
    assert all("qwen25vl_7b" in row["missing_models"] for row in result["rows"])
    assert len(result["rows"]) == 2 * 16 * 3


@pytest.mark.parametrize(
    "pairs", [((float("nan"), 0.5),), ((-1, 0.5),), ((1.1, 1),), ((0.5, None),)]
)
def test_invalid_support_probabilities_rejected(pairs):
    from src.frozen_comparison import build_support_comparison

    task = subtask()
    task["metrics"]["by_prompt"]["p0"]["pooled"].update(pX=pairs[0][0], v=pairs[0][1])
    with pytest.raises(ValueError):
        build_support_comparison([task])


def test_exports_label_auxiliary_scope_and_empty_csv(tmp_path):
    import csv
    import json

    from src.frozen_comparison import write_support_comparison

    paths = write_support_comparison(tmp_path, [subtask()])
    assert len(paths) == 3
    report = (tmp_path / "frozen_support_comparison_zh.md").read_text()
    assert "辅助描述性分析" in report
    assert "不能解释为规模因果效应或训练收益" in report
    assert "模型各自" in report
    document = json.loads((tmp_path / "frozen_support_comparison.json").read_text())
    assert document["overall"][0]["sampling"]["prompt_count"] == 1
    with (tmp_path / "frozen_support_comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["mean_prompt_pX"] == "" for row in rows if row["prompt_count"] == "0")


def test_report_only_compares_verified_imported_metrics(tmp_path, monkeypatch):
    from src import report
    from src.core import write_json

    task = subtask(pairs=((0.25, 0.5),))
    output = tmp_path / "report"
    write_json(output / task["path"] / "metrics.json", task["metrics"])
    verified = {key: value for key, value in task.items() if key != "metrics"}
    monkeypatch.setattr(
        report,
        "_import_frozen",
        lambda root, out: ({"status": "PARTIAL", "details": {}}, [verified]),
    )
    phases = report.build_report(tmp_path / "runs", output)
    assert phases["P3"]["status"] == "PARTIAL"
    assert (output / "frozen_support_comparison.json").exists()
    status = __import__("json").loads((output / "table_status.json").read_text())
    assert status["frozen_support_comparison.csv"]["status"] == "AUXILIARY_DESCRIPTIVE"
    assert status["training_steps.parquet"]["status"] == "NOT_RUN"
