"""Tiny Q1 summary fixtures, never scientific CPU confirmation."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3 import q1_results as q
from src.modeling_v3.cpu_campaign import _digest, _finish, estimate_packet


def config():
    return json.loads((Path(__file__).parents[2] / "configs/modeling_v3/protocol.json").read_text())


def original(
    tmp_path,
    *,
    timings=True,
    diagnostic=True,
    unit_id="unit-one",
    truth_shift=0,
    missing_method=None,
):
    root = tmp_path / "original"
    unit = root / "units" / unit_id
    unit.mkdir(parents=True)
    truth = np.array([[0.1, 0.0, -0.1, 0.0], [-0.1, 0.2, -0.1, 0.0]])
    truth += [truth_shift, -truth_shift, 0, 0]
    methods = ["RAW4", "PRESERVE_XI", "PILOT_SHRINK_ZERO_SUM", "CROSSFIT_COV_ZERO_SUM"]
    expected = []
    for repeat in range(3):
        folder = unit / f"repeat{repeat:03d}"
        folder.mkdir()
        z = np.broadcast_to(np.eye(4)[:, None, :], (4, 2, 4)).copy()
        z[-1] *= 1 + repeat / 10
        packet = {
            "z_raw": z,
            "sample_ids": np.array([f"r{repeat}:{i}" for i in range(4)]),
            "rng_seeds": np.array([repeat * 2, repeat * 2 + 1]),
        }
        outputs, _, diagnostics = estimate_packet(packet, methods, config())
        outputs.update(
            truth=truth, raw_sum=z.sum(0), raw_second_moment=np.einsum("npe,npf->pef", z, z)
        )
        expected.append(outputs["estimate_RAW4"] - truth)
        if missing_method:
            del outputs["estimate_" + missing_method]
        np.savez_compressed(folder / "statistics.npz", **outputs)
        np.savez_compressed(folder / "raw_packet.npz", **packet)
        identity = {"n": 4, "seed": repeat, "diagnostics": diagnostics if diagnostic else {}}
        (folder / "sampling_identity.json").write_text(json.dumps(identity))
        summary = {"repeat": repeat}
        if timings:
            summary["timings"] = {key: (repeat + 1) / 10 for key in q.TIMING_FIELDS}
        _finish(folder, {"repeat": repeat}, summary)
    row = {
        "id": unit_id,
        "n": 4,
        "proposal": "origin",
        "case": "FIXTURE",
        "seed": 999,
        "arm": "X_BASE",
        "anchor": 8,
        "bank": 0,
        "repeats": 3,
        "wall_seconds": 8.0,
        "metrics": {method: {} for method in methods},
    }
    (unit / "RESULTS.json").write_text(json.dumps(row))
    _finish(unit, {"measurement_id": unit_id}, row)
    summary = {
        "stage": "Q1",
        "role": "development",
        "pilot": False,
        "scientific_status": "DEVELOPMENT_ONLY",
        "units": [row],
        "unit_count": 1,
    }
    (root / "OBSERVATION_SUMMARY.json").write_text(json.dumps(summary))
    _finish(root, {"config_sha256": _digest(config())}, summary)
    return root, np.asarray(expected)


def test_recompute_raw_residuals_coefficient_variance_mass_and_timings(tmp_path):
    source, expected = original(tmp_path)
    result = q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True)
    assert result["fixture"] and result["scientific_status"] == "FIXTURE_NOT_AUTHORIZATION"
    unit = json.loads((tmp_path / "analysis/units/unit-one/SUMMARY.json").read_text())
    raw = unit["methods"]["RAW4"]["events"]["X"]
    assert raw["signed_bias"] == pytest.approx(expected[..., 0].mean())
    assert raw["mse"] == pytest.approx(np.mean(expected[..., 0] ** 2))
    assert raw["pooled_residual_population_variance"] == pytest.approx(np.var(expected[..., 0]))
    assert raw["mean_within_prompt_repeat_sample_variance"] == pytest.approx(
        np.var(expected[..., 0], axis=0, ddof=1).mean()
    )
    assert raw["absolute_residual_quantiles"]["q95"] == pytest.approx(
        np.quantile(abs(expected[..., 0]), 0.95)
    )
    assert unit["methods"]["PILOT_SHRINK_ZERO_SUM"]["coefficients"]["fallback"]["rate"] == 1
    assert (
        unit["methods"]["CROSSFIT_COV_ZERO_SUM"]["coefficients"]["single_coefficient_status"]
        == "NOT_APPLICABLE_TWO_FOLDS"
    )
    assert unit["mass"]["RAW4"]["post_correction"]["max_absolute"] > 0.9
    assert unit["mass"]["PRESERVE_XI"]["post_correction"]["max_absolute"] < 1e-14
    assert unit["timings"]["estimator_seconds"]["seconds"] == pytest.approx(0.6)
    assert unit["timings"]["generation_only_seconds"]["status"] == "MISSING_INPUTS"
    npz = np.load(tmp_path / "analysis/units/unit-one/EXACT_RECORDS.npz", allow_pickle=False)
    np.testing.assert_allclose(npz["residual_RAW4"], expected)
    assert (
        q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True, resume=True)
        == result
    )
    with pytest.raises(FileExistsError):
        q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True)


def test_missing_receipts_do_not_turn_into_zero_cost_or_zero_fallback(tmp_path):
    source, _ = original(tmp_path, timings=False, diagnostic=False)
    result = q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True)
    unit = json.loads((tmp_path / "analysis/units/unit-one/SUMMARY.json").read_text())
    assert unit["timings"]["estimator_seconds"]["status"] == "MISSING_INPUTS"
    assert unit["timings"]["estimator_seconds"]["seconds"] is None
    fallback = unit["methods"]["PILOT_SHRINK_ZERO_SUM"]["coefficients"]["fallback"]
    assert fallback["status"] == "MISSING_INPUTS" and fallback["rate"] is None
    assert result["missing_input_count"] > 0


def test_original_tampering_rejected(tmp_path):
    source, _ = original(tmp_path)
    path = source / "units/unit-one/repeat000/statistics.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True)


def test_pooled_quantiles_recomputed_across_distinct_units(tmp_path):
    first, e1 = original(tmp_path / "first")
    second, e2 = original(tmp_path / "second", unit_id="unit-two", truth_shift=0.7)
    result = q.analyze_observation(config(), [first, second], tmp_path / "analysis", fixture=True)
    row = next(r for r in result["pooled_rows"] if r["method"] == "RAW4" and r["event"] == "X")
    original_errors = np.concatenate([e1[..., 0].ravel(), e2[..., 0].ravel()])
    actual = row["absolute_residual_quantiles"]["q95"]
    assert actual == pytest.approx(np.quantile(abs(original_errors), 0.95))
    assert actual != pytest.approx(np.mean([np.quantile(abs(e[..., 0]), 0.95) for e in [e1, e2]]))
    assert row["count"] == original_errors.size


def test_missing_original_estimate_is_not_inferred_from_results(tmp_path):
    source, _ = original(tmp_path, missing_method="RAW4")
    result = q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True)
    unit = json.loads((tmp_path / "analysis/units/unit-one/SUMMARY.json").read_text())
    assert unit["methods"]["RAW4"]["status"] == "MISSING_INPUTS"
    assert not any(row["method"] == "RAW4" for row in result["pooled_rows"])


def test_original_output_overlap_rejected_without_modifying_original(tmp_path):
    source, _ = original(tmp_path)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overlap immutable"):
        q.analyze_observation(config(), [source], source / "analysis", fixture=True)
    assert before == {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }


def test_partial_publication_resume_is_idempotent_and_derived_tamper_rejected(
    tmp_path, monkeypatch
):
    source, _ = original(tmp_path)
    publish = q._publish_json

    def interrupted(path, value):
        if path.name == "Q1_ANALYSIS.json":
            raise RuntimeError("simulated interruption")
        return publish(path, value)

    monkeypatch.setattr(q, "_publish_json", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True)
    monkeypatch.setattr(q, "_publish_json", publish)
    q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True, resume=True)
    path = tmp_path / "analysis/Q1_POOLED_ERRORS.csv"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        q.analyze_observation(config(), [source], tmp_path / "analysis", fixture=True, resume=True)


def test_signed_validity_and_variance_decomposition_are_not_group_averages():
    residual = np.array(
        [[[1.0, 0, 0, -2.0], [3.0, 0, 0, -4.0]], [[2.0, 0, 0, -1.0], [6.0, 0, 0, -3.0]]]
    )
    truth = np.zeros((2, 4))
    report = q.residual_summary(residual, truth)
    assert report["delta_v"]["signed_bias"] == 2.5
    assert report["X"]["signed_bias"] == 3
    assert report["X"]["mean_within_prompt_repeat_sample_variance"] == 2.5
    assert report["X"]["mse_bias_variance_identity_residual"] == pytest.approx(0)
    assert report["X"]["absolute_residual_quantiles"]["q95"] == pytest.approx(
        np.quantile([1, 3, 2, 6], 0.95)
    )


def test_full_analysis_rejects_local_and_pilot_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr(q.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="server"):
        q.analyze_observation(config(), [], tmp_path / "none")
