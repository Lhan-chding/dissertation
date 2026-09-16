from pathlib import Path

import numpy as np
import pytest

from src.modeling_v4.cli import main, parser


def test_plan_cli_records_unexecuted_stage_without_model_import(tmp_path):
    config = Path(__file__).resolve().parents[2] / "configs/modeling_v4/protocol.json"
    result = main(["plan", "--config", str(config), "--out", str(tmp_path)])
    assert result["gpu_executed"] is False
    assert (tmp_path / "PLAN.json").is_file()


def test_worker_cli_defaults_to_no_gpu_execution():
    args = parser().parse_args(
        ["run-worker", "--tasks", "tasks.json", "--worker-id", "0", "--workers", "2"]
    )
    assert args.device == "cuda:0" and args.execute_gpu is False


def test_no_implicit_cli_command():
    with pytest.raises(SystemExit):
        parser().parse_args([])


def test_predict_cli_never_opens_extra_reference_array(tmp_path):
    training = tmp_path / "fit.npz"
    np.savez(training, E=np.eye(2), Y=np.array([[1.0, 2.0], [3.0, 4.0]]))
    main(["fit", "--data", str(training), "--out", str(tmp_path / "model")])
    query = tmp_path / "query.npz"
    # Opening this extra array with allow_pickle=False raises. Predict must only
    # read the registered query features, even if unrelated labels are present.
    np.savez(query, queries=np.zeros((1, 2)), reference=np.array([{"unread": True}], dtype=object))
    result = main(
        [
            "predict",
            "--model",
            str(tmp_path / "model"),
            "--data",
            str(query),
            "--out",
            str(tmp_path / "prediction.npz"),
        ]
    )
    assert result["reference_labels_read"] is False
    with np.load(tmp_path / "prediction.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["prediction"], [[0.0, 0.0]])
