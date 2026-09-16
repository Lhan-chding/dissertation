import copy

import numpy as np
import pytest

from src.modeling_v4 import cross_prompt


def panels():
    return {
        "cross_prompt": [
            {
                "prompt_id": f"cross_{f}_{i}_{s}",
                "base_scene_id": f"new_{f}_{s}",
                "family": f"f{f}",
                "interface": f"i{i}",
            }
            for f in range(3)
            for i in range(2)
            for s in range(6)
        ],
        "observation": [{"prompt_id": "old", "base_scene_id": "old"}],
        "signature": [{"prompt_id": "signature", "base_scene_id": "signature"}],
        "first_map": [{"prompt_id": "old", "base_scene_id": "old"}],
        "selection_used_outcomes": False,
    }


def banks():
    return [
        {
            "bank_id": f"q{i}",
            "role": "query",
            "policies": {
                "joint_0": {"inference_fingerprint": "a"},
                "joint_1": {"inference_fingerprint": "a" if i == 0 else f"b{i}"},
            },
            "contrasts": {"joint_1_minus_joint_0": {"exact_inference_alias": i == 0}},
        }
        for i in range(4)
    ]


def test_panel_disjoint_selection_is_fingerprint_only_and_stable():
    p = panels()
    chosen = cross_prompt.select_cross_prompt_inputs(p, banks())
    assert [b["bank_id"] for b in chosen[1]] == ["q1", "q2"]
    assert len(chosen[0]) == 36
    bad = copy.deepcopy(p)
    for row in bad["cross_prompt"]:
        if row["base_scene_id"] == "new_0_0":
            row["base_scene_id"] = "old"
    with pytest.raises(ValueError, match="disjoint"):
        cross_prompt.select_cross_prompt_inputs(bad, banks())
    wrong = banks()
    wrong[1]["contrasts"]["joint_1_minus_joint_0"]["exact_inference_alias"] = True
    with pytest.raises(ValueError, match="fingerprint"):
        cross_prompt.select_cross_prompt_inputs(p, wrong)


def test_full_ad_contraction_uses_complete_coordinates_and_fp64_subtraction():
    data = np.array([[[16777216, 2], [-1, 4], [0, 0]]], dtype=np.float32)
    layout = {"a": {"start": 0, "stop": 2, "shape": (2,)}}
    gradient = {
        "parameter_order": ["a"],
        "expansion_point": "ORIGIN",
        "reference_labels_read": False,
        "groups": [["f", "i"]],
        "grouped_gradients": {"a": np.ones((1, 4, 2))},
    }
    actual = cross_prompt.contract_full_gradient(data, layout, gradient, 0, 1)
    np.testing.assert_array_equal(actual, np.full((1, 1, 4), 16777215.0))
    bad = {**gradient, "parameter_order": []}
    with pytest.raises(ValueError, match="complete"):
        cross_prompt.contract_full_gradient(data, layout, bad, 0, 1)


def test_cross_prompt_gate_rejects_wrong_role_before_creating_outputs(tmp_path):
    with pytest.raises(ValueError, match="41001"):
        cross_prompt.run_cross_prompt({}, {}, {}, {"role": "locked_test"}, {}, out=tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_fixed_panel_head_is_unknown_and_unresolved_finite_errors_remain():
    p = panels()["cross_prompt"]
    groups = sorted({(r["family"], r["interface"]) for r in p})
    b = banks()[1]
    predictions = {"DIRECT_MEASURE": np.zeros((6, 4)), "FULL_SCORE_JVP": np.ones((6, 4)) * 0.02}
    reference = [
        {
            "estimate": np.ones((3, 4)) * 0.01,
            "se": np.ones((3, 4)) * 0.001,
            "resolved": np.zeros((3, 4), bool),
        }
        for _ in p
    ]
    rows = cross_prompt.evaluation_rows(
        "origin", b, p, groups, predictions, reference, target_index=0
    )
    unknown = next(r for r in rows if r["method"] == "FIXED_PANEL_OUTPUT_HEAD")
    assert unknown["prediction"] is None and unknown["prediction_status"] == "UNKNOWN"
    from src.modeling_v4.evaluate import evaluate_records

    result = evaluate_records(rows)
    direct = next(
        r for r in result["summary"] if r["method"] == "DIRECT_MEASURE" and r["channel"] == "RAW4"
    )
    assert direct["all_finite_observed_mse"] == pytest.approx(0.0001)
    assert direct["mse"] is None


def test_tiny_completed_pipeline_seals_predictions_before_reference_and_resumes(
    tmp_path, monkeypatch
):
    from src.modeling_v3.io import sha256_file
    from src.modeling_v4 import response_fit

    probes = panels()["cross_prompt"]
    selected = banks()[1:3]
    for bank in selected:
        bank["contrasts"] = {
            target: {"exact_inference_alias": False} for target, _, _ in cross_prompt.CONTRASTS
        }
    forks = {
        "origin_id": "actual_origin:cross_prompt",
        "origin_policy": {"inference_fingerprint": "origin"},
        "banks": selected,
    }
    response = {
        role: {"identity": {"rng_namespace": role}} for role in ("work", "direct_work", "reference")
    }
    opened = []

    def actions(receipt, actual_probes, *, role, **kwargs):
        assert actual_probes == probes
        if role == "reference":
            assert (tmp_path / "PREDICTIONS_FROZEN.json").exists()
            assert (tmp_path / "PREDICTIONS_FROZEN.npz").exists()
            opened.append(True)
        return {
            p["prompt_id"]: [
                {"sample_id": f"{role}:{i}", "sample_seed": i + 1000 * list(response).index(role)}
            ]
            for i, p in enumerate(probes)
        }

    monkeypatch.setattr(response_fit, "_action_rows", actions)

    def observations(bank, resp, samples, raw, actual_probes, methods):
        if methods:
            return {"PRESERVE_XI": np.ones((36, 3, 4)) * 0.01}
        return [
            {
                "estimate": np.ones((3, 4)) * 0.01,
                "se": np.ones((3, 4)) * 1e-5,
                "resolved": np.ones((3, 4), bool),
            }
            for _ in probes
        ]

    monkeypatch.setattr(response_fit, "_observations", observations)
    monkeypatch.setattr(
        response_fit, "_apply_mixture_references", lambda *args, **kwargs: {"fixture": True}
    )
    groups = sorted({(p["family"], p["interface"]) for p in probes})
    gradient = {
        "parameter_order": ["weight"],
        "expansion_point": "ORIGIN",
        "reference_labels_read": False,
        "groups": groups,
        "grouped_gradients": {"weight": np.ones((6, 4, 2))},
    }
    monkeypatch.setattr(response_fit, "_read_gradient", lambda *args: gradient)
    data = np.array([[[0, 0], [0.01, 0.02], [0.01, 0.01]], [[0, 0], [0.02, 0.02], [0.01, 0.02]]])
    layout = {"weight": {"start": 0, "stop": 2, "shape": (2,)}}
    monkeypatch.setattr(response_fit, "_endpoint_memmap", lambda *args: (data, layout, {}))
    runtime = {"identity": {"execution_kind": "CPU_FIXTURE"}, "checkpoint_cache": None}
    result = cross_prompt._evaluate_completed(
        runtime, forks, response, {"fixture": True}, probes, tmp_path
    )
    assert result["status"] == "FIXTURE_EVALUATED"
    assert opened == [True]
    assert len(result["core_results"]) == 18
    with np.load(tmp_path / "PREDICTIONS_FROZEN.npz") as values:
        np.testing.assert_allclose(values["q1__joint_1_minus_joint_0__FULL_SCORE_JVP"], 0.03)
    hashes = {p.name: sha256_file(p) for p in tmp_path.iterdir() if p.is_file()}
    cross_prompt._evaluate_completed(runtime, forks, response, {"fixture": True}, probes, tmp_path)
    assert hashes == {p.name: sha256_file(p) for p in tmp_path.iterdir() if p.is_file()}


def test_packet_roles_reject_shared_actual_generation_seed():
    raw = {k: {"p": [{"sample_id": k, "sample_seed": 12}]} for k in ("work", "reference")}
    response = {k: {"identity": {"rng_namespace": k}} for k in raw}
    with pytest.raises(ValueError, match="overlap"):
        cross_prompt._independent_packets(raw, response)


def test_measurement_counts_include_mix_and_deduplicate_alias_score_receipts():
    def packet(identity, count):
        return {"identity": identity, "count": count, "chunks": [{"start": 0, "stop": count}]}

    work = packet({"role": "work", "proposal": "ORIGIN"}, 36)
    mix = packet({"role": "reference", "proposal": "MIX"}, 12)
    score = packet({"sample_identity": "actual", "policy": "same-full-fingerprint"}, 36)
    result = cross_prompt.measurement_counts(
        {"work": work, "checks": [{"mix": mix}], "scores": [score, copy.deepcopy(score)]}
    )
    assert result["actual_unique_action_rows_total"] == 48
    assert result["unique_logical_score_rows"] == 36
    assert result["logical_score_rows_are_forward_calls"] is False


def test_failed_cross_prompt_collection_restores_full_runtime_state(tmp_path, monkeypatch):
    from src.modeling_v4 import gpu_collect

    monkeypatch.setattr(cross_prompt, "_gate", lambda *args: None)
    monkeypatch.setattr(gpu_collect, "verify_artifact_bindings", lambda *args: None)
    restored = []
    runtime = {
        "identity": {"execution_kind": "CPU_FIXTURE"},
        "capture_complete": lambda: {"full": 1},
        "restore_complete": restored.append,
    }

    def failure(*args, **kwargs):
        raise RuntimeError("failed collection")

    monkeypatch.setattr(gpu_collect, "collect_response_map", failure)
    with pytest.raises(RuntimeError, match="failed collection"):
        cross_prompt.run_cross_prompt(
            {},
            runtime,
            {"panels": panels()},
            {"id": "fixture"},
            {"banks": banks(), "origin_id": "actual"},
            out=tmp_path / "result",
        )
    assert restored == [{"full": 1}]
    assert not (tmp_path / "result/COMPLETE.json").exists()
