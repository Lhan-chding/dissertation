import csv
import json

from src.modeling_v4.report import summarize


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def test_missing_actual_evidence_is_not_completion(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    write(
        root / "B_MERGED.json", {"status": "TECHNICAL_BRIDGE_COMPLETED", "parity": {"passed": True}}
    )
    result = summarize(root, root / "report")
    body = json.loads((root / "report/RUN_SUMMARY.json").read_text())
    assert result["core_results_rows"] == 0
    assert body["reliability_decision"] == "REQUIRES_FROZEN_CALIBRATION_AND_INDEPENDENT_TEST"
    assert "REAL_9B_EVALUATION" in body["missing_artifacts"]
    assert body["online_ssvc"] == "NOT_RUN_NOT_CERTIFIED"


def test_incomplete_evaluation_is_ignored_and_fixture_is_labelled(tmp_path):
    root = tmp_path / "campaign"
    base = {
        "kind": "V4_EVALUATION",
        "core_results_file": "CORE_RESULTS.csv",
        "binding": {"execution_kind": "FIXTURE"},
        "summary": [],
    }
    write(root / "unfinished/METRICS.json", base)
    write(root / "complete/METRICS.json", base)
    write(root / "complete/COMPLETE.json", {"status": "COMPLETE"})
    (root / "complete/CORE_RESULTS.csv").write_text("method,seed\nZERO,41001\n")
    result = summarize(root, root / "report")
    body = json.loads((root / "report/RUN_SUMMARY.json").read_text())
    assert result["core_results_rows"] == 1
    assert result["real_cuda_evaluation_count"] == 0
    assert body["incomplete_evaluations"][0]["reason"] == "NO_COMPLETE_MARKER"
    with (root / "report/CORE_RESULTS.csv").open() as stream:
        assert next(csv.DictReader(stream))["source_metrics"] == "complete/METRICS.json"
    assert summarize(root, root / "report")["core_results_rows"] == 1


def test_fixture_and_failed_map_markers_do_not_count_as_real_first_maps(tmp_path):
    root = tmp_path / "campaign"
    for name, status, execution in (
        ("fixture", "COMPLETED", "FIXTURE"),
        ("failed", "FAILED", "REAL_CUDA_MODEL"),
        ("real", "COMPLETED", "REAL_CUDA_MODEL"),
    ):
        write(
            root / "tasks" / name / "COMPLETE.json",
            {"kind": "map", "status": status, "execution_kind": execution, "task": {"stage": "C"}},
        )
    summarize(root, root / "report")
    body = json.loads((root / "report/RUN_SUMMARY.json").read_text())
    assert body["completed_first_map_collections"] == 1
