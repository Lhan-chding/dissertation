import json

import numpy as np
import pytest

from src.modeling_v3.io import sha256_file
from src.modeling_v3.workflow import run_artifact_command


def test_unbound_acceptance_string_does_not_authorize_predictions(tmp_path):
    arrays = tmp_path / "arrays.npz"
    np.savez(
        arrays,
        predictions=np.zeros((2, 4)),
        reference=np.zeros((2, 4)),
        accepted=np.ones(2, dtype=bool),
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "arrays": {"path": str(arrays), "sha256": sha256_file(arrays)},
                "acceptance_lock_sha256": "arbitrary-nonempty-string",
            }
        )
    )
    with pytest.raises(ValueError, match="acceptance receipt"):
        run_artifact_command("evaluate", {}, [str(spec)], tmp_path / "out")


def test_finite_unaccepted_predictions_remain_unknown(tmp_path):
    arrays = tmp_path / "arrays.npz"
    np.savez(arrays, predictions=np.zeros((2, 4)), reference=np.ones((2, 4)) * 0.1)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"arrays": {"path": str(arrays), "sha256": sha256_file(arrays)}}))
    run_artifact_command("evaluate", {}, [str(spec)], tmp_path / "out")
    result = json.loads((tmp_path / "out/EVALUATION.json").read_text())
    assert result["resolved_cases"] == 2
    assert result["unknown_cases"] == 2
    assert result["risk_coverage"]["accepted_cases"] == 0
    assert result["risk_coverage"]["accepted_risk"] is None


def test_large_parameter_arrays_are_binary_payloads_not_json_lists(tmp_path):
    arrays = tmp_path / "fit.npz"
    np.savez(
        arrays,
        updates=np.eye(2, 4),
        responses=np.array([[0.1, -0.1, 0, 0], [0, 0, 0.1, -0.1]]),
        query_updates=np.ones((1, 4)),
    )
    spec = tmp_path / "fit.json"
    spec.write_text(
        json.dumps(
            {"arrays": {"path": str(arrays), "sha256": sha256_file(arrays)}, "method": "FULL_RIDGE"}
        )
    )
    run_artifact_command("fit", {}, [str(spec)], tmp_path / "model")
    model = json.loads((tmp_path / "model/MODEL.json").read_text())
    assert "Q" not in model and "coefficients" not in model
    with np.load(tmp_path / "model/MODEL_ARRAYS.npz", allow_pickle=False) as payload:
        assert payload["Q"].shape == (4, 2)
        assert payload["predictions"].shape == (1, 4)
    selection_arrays = tmp_path / "pool.npz"
    np.savez(selection_arrays, updates=np.eye(2, 4)[:, None])
    spec.write_text(
        json.dumps(
            {
                "arrays": {"path": str(selection_arrays), "sha256": sha256_file(selection_arrays)},
                "method": "FIRST",
                "n_banks": 1,
            }
        )
    )
    run_artifact_command("select", {}, [str(spec)], tmp_path / "selection")
    selection = json.loads((tmp_path / "selection/SELECTION.json").read_text())
    assert "selected_updates" not in selection
    assert selection["selected_indices"] == [0]
