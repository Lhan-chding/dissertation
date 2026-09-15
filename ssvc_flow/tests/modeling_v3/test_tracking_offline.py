import json

import pytest

from src.core import file_hash
from src.modeling_v3.tracking_offline import evaluate_tracking, fixed_basis_tracking, track_offline


def inputs():
    return dict(
        anchor_probability=[0.25] * 4,
        updates=[[0.1, 0], [0.1, 0.2], [-0.2, -0.2]],
        basis=[[1], [0]],
        coefficients=[[0.1, -0.1, 0, 0]],
        qualification={
            "pointwise_qualified": True,
            "fit_hash": "frozen_fit",
            "source_hash": "frozen_source",
        },
    )


def test_every_step_fixed_basis_and_path_does_not_change_point_prediction():
    result = fixed_basis_tracking(
        **inputs(), interval_half_width=0.01, uncovered_path_coefficient=0.2
    )
    assert len(result["rows"]) == 3
    assert result["first_novel_direction_horizon"] == 2
    assert result["rows"][-1]["full_net_norm"] == 0
    assert result["rows"][-1]["uncovered_path_length"] == pytest.approx(0.4)
    assert all(
        row["projected_net_prediction"] == row["projection_plus_path_point_prediction"]
        for row in result["rows"]
    )
    assert result["rows"][-1]["path_range_width"] > result["rows"][-1]["range_width"]
    assert result["online_feedback"] is False


def test_unknown_not_zero_or_safe_and_horizon_requires_intermediates():
    value = inputs()
    result = fixed_basis_tracking(**value, maximum_uncovered_path=0.1)
    assert [r["status"] for r in result["rows"]] == ["PREDICTED", "UNKNOWN", "UNKNOWN"]
    with pytest.raises(ValueError, match="intervening"):
        evaluate_tracking(result, [[0.25] * 4] * 3, horizons=[4])
    scored = evaluate_tracking(result, [[0.25] * 4] * 3, horizons=[1, 2, 3])
    assert scored["windows"][-1]["identified_coverage"] == pytest.approx(1 / 3)
    assert scored["windows"][-1]["unknown_is_safe"] is False


def test_no_pointwise_qualification_no_references_read(tmp_path):
    value = inputs()
    value["qualification"]["pointwise_qualified"] = False
    value["reference_binding"] = {"path": str(tmp_path / "not-readable.json"), "sha256": "bad"}
    assert track_offline(value, tmp_path / "out")["status"] == "NOT_QUALIFIED_FOR_TRACKING"


def test_intermediate_error_survives_zero_endpoint_and_reference_uncertainty(tmp_path):
    value = inputs()
    reference = {
        "kind": "REAL_VLM_ESTIMATE",
        "probabilities": [[0.3, 0.2, 0.25, 0.25], [0.31, 0.19, 0.25, 0.25], [0.25] * 4],
        "half_width": 0.0001,
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    value.update(
        reference_binding={"path": str(path), "sha256": file_hash(path)},
        tracking_config={"horizons": [1, 2, 3]},
        interval_half_width=0.005,
    )
    result = track_offline(value, tmp_path / "out")
    final = result["assessment"]["windows"][-1]
    assert final["window_maximum_prediction_error"] == pytest.approx(0.04)
    assert final["intermediate"][-1]["maximum_event_error"] == 0
    assert result["assessment"]["reference_is_exact"] is False
    assert (tmp_path / "out/prediction_lock.json").is_file()
    assert final["hindsight_interpolation_online_usable"] is False


def test_basis_change_nonorthogonal_and_missing_vlm_uncertainty_rejected(tmp_path):
    value = inputs()
    with pytest.raises(ValueError, match="changed"):
        fixed_basis_tracking(**value, basis_fingerprint="different")
    value["basis"] = [[2], [0]]
    with pytest.raises(ValueError, match="orthonormal"):
        fixed_basis_tracking(**value)
    value = inputs()
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"kind": "REAL_VLM_ESTIMATE", "probabilities": [[0.25] * 4] * 3}))
    value["reference_binding"] = {"path": str(path), "sha256": file_hash(path)}
    with pytest.raises(ValueError, match="uncertainty"):
        track_offline(value, tmp_path / "out")
