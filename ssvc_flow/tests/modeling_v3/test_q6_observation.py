"""Model-free two-worker Q6 samples; no fixture authorizes actual GPU execution."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from src.modeling_v3 import q6_observation as q6
from src.modeling_v3 import vlm_campaign
from src.modeling_v3 import vlm_observation as v
from src.modeling_v3.io import atomic_json, atomic_npz

TOLERANCE = {"mean_abs_token_logp": 1e-6, "max_abs_token_logp": 1e-6, "max_abs_sequence_logp": 2e-6}


class Adapter:
    audit: ClassVar = {"probability_execution": v.PATH}
    eos_ids: ClassVar = {99}
    pad_id = 0
    forward_calls = 0

    def __init__(self):
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(decode=lambda tokens, **kw: str(tokens[0]))
        )

    def prepare(self, prompt, data_root):
        return {
            "audit": {
                "enable_thinking": False,
                "image_token_count": 0,
                "pixel_values_hash": None,
                "final_prompt_hash": v.digest(prompt),
                "input_tensor_hash": v.digest([prompt]),
            }
        }

    def generate(self, prepared, *, seed, **kwargs):
        self.forward_calls += 1
        return {
            "token_ids": [seed % 4, 99],
            "raw_completion": str(seed % 4),
            "completion_length": 2,
            "stop_reason": "eos",
            "behavior_token_logprobs": [-1.0, -1.0],
        }

    def logprobs(self, prepared, tokens, **kwargs):
        self.forward_calls += 1
        return [-1.0] * len(tokens)


def setup_plan(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    runtime = {"fixture": True, "execution_kind": "CPU_FIXTURE"}
    prompts = [
        {"prompt_id": p, "prompt": {"user": p}, "interface": "SYMBOLIC_FRESH", "scene": {}}
        for p in ("p0", "p1")
    ]
    atomic_json(tmp_path / "prompts.json", prompts)
    records = []
    for step in range(64, 81):
        path = tmp_path / f"state{step}.bin"
        path.write_bytes(f"fixture{step}".encode())
        records.append(
            {
                "step": step,
                "path": str(path),
                "checkpoint_sha256": v.file_digest(path),
                "checkpoint_identity": {"step": step},
                "inference_fingerprint": f"state{step}",
                "state_hash": f"full{step}",
                "parameter_hash": f"params{step}",
            }
        )
    atomic_json(tmp_path / "source.json", {"checkpoints": records})
    inputs = {
        key: {"path": str(tmp_path / key), "sha256": key}
        for key in ("qualification_binding", "fit_binding", "model_binding")
    }
    inputs["source_binding"] = q6._binding(tmp_path / "source.json")
    stage = {
        "phase": "Q6",
        "qualification_binding": inputs["qualification_binding"],
        "source_binding": inputs["source_binding"],
        "design_id": "d",
    }
    atomic_json(tmp_path / "stage.json", stage)
    bindings = {
        "v3_stage_lock": q6._binding(tmp_path / "stage.json"),
        "v3_probability_tolerances": TOLERANCE,
    }
    policy = {
        "checkpoint": {"path": records[0]["path"], "sha256": records[0]["checkpoint_sha256"]},
        "inference_fingerprint": records[0]["inference_fingerprint"],
    }
    task = v.freeze_observation_task(
        operation="generate",
        origin_id="123_X_BASE_64",
        candidate_id="origin",
        prompt_file=q6._binding(tmp_path / "prompts.json"),
        prompt_ids=["p0", "p1"],
        role="pilot",
        draw_start=0,
        draw_stop=4,
        rng_namespace="independent-fit",
        proposal="ORIGIN",
        proposal_candidates=["origin"],
    )
    fit_manifest = v.freeze_task_manifest([task], {"origin": policy}, runtime)
    atomic_json(tmp_path / "fit-generation.json", fit_manifest)
    fit = {
        "origin_id": "123_X_BASE_64",
        "probe_ids": ["p0", "p1"],
        "measurement_bundle": {
            "generation_manifest": q6._binding(tmp_path / "fit-generation.json")
        },
    }
    fit["alpha"] = 1e-5
    atomic_npz(tmp_path / "FIT_ARRAYS.npz", {"updates": np.eye(2)})
    fit["arrays"] = q6._binding(tmp_path / "FIT_ARRAYS.npz")
    atomic_json(tmp_path / "FIT_SPEC.json", fit)
    inputs["fit_binding"] = q6._binding(tmp_path / "FIT_SPEC.json")
    basis, coef, updates = np.array([[1.0], [0.0]]), np.zeros((1, 8)), np.zeros((16, 2))
    atomic_npz(
        tmp_path / "MODEL_ARRAYS.npz", {"Q": np.eye(2), "directions": basis, "coefficients": coef}
    )
    atomic_json(tmp_path / "MODEL.json", {"array_payload": {"path": "MODEL_ARRAYS.npz"}})
    inputs["model_binding"] = q6._binding(tmp_path / "MODEL.json")
    atomic_json(
        tmp_path / "SELECTION.json",
        {
            "selected": {
                "pointwise_criteria": {"primary_tolerance": 0.001},
                "rho_threshold": 0.1,
                "leverage_threshold": 10.0,
            }
        },
    )
    qualified = {
        "pointwise_qualified": True,
        "design_id": "d",
        "fit_hash": inputs["fit_binding"]["sha256"],
        "source_hash": "tiny-mocked-source",
        "selection_lock": q6._binding(tmp_path / "SELECTION.json"),
    }
    monkeypatch.setattr(
        q6, "_qualified_inputs", lambda *a: (qualified, fit, basis, coef, updates, records)
    )
    monkeypatch.setattr(vlm_campaign, "planned_runtime_identity", lambda *a: runtime)
    plan = q6.prepare_q6_observation(
        config, bindings, **inputs, out=tmp_path / "plan", fixture=True, fixture_draws=4
    )
    manifest = q6._bound_json(plan["manifest"])
    worker_root = tmp_path / "observations"
    for worker in range(2):
        backend = v.PrefixObservationBackend(
            Adapter(),
            runtime_identity=runtime,
            policy_loader=lambda p: p["inference_fingerprint"],
            annotator=lambda text, scene: {"category": v.EVENTS[int(text)]},
            parity_tolerances=TOLERANCE,
        )
        v.execute_observation_tasks(manifest, backend, out=worker_root, worker_index=worker)
    return config, plan, worker_root


def test_plan_runs_real_executor_fixture_and_counts_retained_actions(tmp_path, monkeypatch):
    config, plan, root = setup_plan(tmp_path, monkeypatch)
    assert not plan["submitted"] and not plan["model_loaded"]
    assert len(plan["commands"]) == 2 and plan["draw_stop"] == 4
    measurement = q6.measure_q6_absolute(
        config,
        plan_binding=plan["plan_binding"],
        worker_root=root,
        out=tmp_path / "measurement",
        fixture=True,
    )
    manifest = q6._bound_json(plan["manifest"])
    rows = list(v.iter_completed_task_rows(root, manifest))
    np.testing.assert_array_equal(
        np.asarray(measurement["probabilities"])[0],
        [
            [sum(r["category"] == e for r in rows if r["prompt_id"] == p) / 4 for e in v.EVENTS]
            for p in plan["probe_ids"]
        ],
    )
    assert np.asarray(measurement["half_width"]).min() > 0
    assert measurement["reference_is_exact"] is False
    assert (
        q6.verify_q6_absolute(config, measurement["receipt"], fixture=True)["probabilities"]
        == measurement["probabilities"]
    )
    assert len(rows) == 8 and v._completed_task_rows(root, manifest) == rows


def test_iterator_validates_manifest_once_and_rejects_modified_original(tmp_path, monkeypatch):
    config, plan, root = setup_plan(tmp_path, monkeypatch)
    validate = v.validate_task_manifest
    calls = []
    monkeypatch.setattr(
        v, "validate_task_manifest", lambda *a, **k: (calls.append(1), validate(*a, **k))[1]
    )
    manifest = q6._bound_json(plan["manifest"])
    assert len(list(v.iter_completed_task_rows(root, manifest))) == 8
    assert len(calls) == 1
    raw = next(root.glob("worker_0/tasks/*/attempt_*/samples.jsonl"))
    raw.write_bytes(raw.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="original bytes changed"):
        q6.measure_q6_absolute(
            config,
            plan_binding=plan["plan_binding"],
            worker_root=root,
            out=tmp_path / "bad",
            fixture=True,
        )


def test_zero_variance_does_not_create_exact_or_zero_uncertainty():
    protocol = {
        "looks": [4, 8],
        "alpha": 0.05,
        "family_cells": 2,
        "primary_half_width_goal": 0.00025,
    }
    result = q6.absolute_statistics(np.array([[[0, 0, 4, 0]]]), 4, protocol)
    assert np.asarray(result["empirical_normal_half_width"]).max() == 0
    assert np.asarray(result["half_width"]).min() > 0
    assert result["status"] == "REFERENCE_EXTEND" and result["next_draws"] == 8
    with pytest.raises(ValueError, match="actual draws"):
        q6.absolute_statistics(np.array([[[0, 0, 3, 0]]]), 4, protocol)


def test_real_reader_rejects_fixture_and_local_cpu(tmp_path, monkeypatch):
    config, plan, root = setup_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(q6.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="server CPU"):
        q6.measure_q6_absolute(
            config, plan_binding=plan["plan_binding"], worker_root=root, out=tmp_path / "bad"
        )
    monkeypatch.setattr(q6.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SLURM_JOB_ID", "mock-cpu-context")
    with pytest.raises(ValueError, match="fixture identity"):
        q6.measure_q6_absolute(
            config, plan_binding=plan["plan_binding"], worker_root=root, out=tmp_path / "bad"
        )


def test_anchor_prediction_reference_assessment_chain_retains_anchor_uncertainty(
    tmp_path, monkeypatch
):
    from src.modeling_v3 import tracking_offline as tracking

    config, anchor_plan, anchor_root = setup_plan(tmp_path / "original", monkeypatch)
    anchor = q6.measure_q6_absolute(
        config,
        plan_binding=anchor_plan["plan_binding"],
        worker_root=anchor_root,
        out=tmp_path / "anchor",
        fixture=True,
    )
    payload = {**anchor_plan["inputs"], "anchor_binding": anchor["receipt"]}
    prepared = q6._qualified_inputs(config, anchor_plan["inputs"], True)
    frozen = tracking._finish_real_tracking(
        config, payload, tmp_path / "predictions", prepared, fixture=True
    )
    lock = frozen["prediction_binding"]
    assert (
        q6._prediction(config, lock, anchor_plan["inputs"], True)["reference_labels_read"] is False
    )
    ref_plan = q6.prepare_q6_observation(
        config,
        q6._bound_json(anchor_plan["bindings"]),
        **anchor_plan["inputs"],
        out=tmp_path / "ref_plan",
        phase="reference",
        prediction_lock=lock,
        fixture=True,
        fixture_draws=4,
    )
    manifest = q6._bound_json(ref_plan["manifest"])
    ref_root = tmp_path / "reference_samples"
    for worker in range(2):
        backend = v.PrefixObservationBackend(
            Adapter(),
            runtime_identity=manifest["runtime_identity"],
            policy_loader=lambda p: p["inference_fingerprint"],
            annotator=lambda text, scene: {"category": v.EVENTS[int(text)]},
            parity_tolerances=TOLERANCE,
        )
        v.execute_observation_tasks(manifest, backend, out=ref_root, worker_index=worker)
    reference = q6.measure_q6_absolute(
        config,
        plan_binding=ref_plan["plan_binding"],
        worker_root=ref_root,
        out=tmp_path / "reference",
        fixture=True,
    )
    result = tracking._finish_real_tracking(
        config,
        {**payload, "prediction_binding": lock, "reference_binding": reference["receipt"]},
        tmp_path / "assessment",
        prepared,
        fixture=True,
    )
    assert result["status"] == "OFFLINE_ASSESSED" and result["fixture"] is True
    assert result["anchor_uncertainty_retained"] and result["reference_uncertainty_retained"]
    assert result["windows"][-1]["all_steps"] == 16
    assert result["windows"][-1]["rejected_steps"] == 16
    assert result["windows"][-1]["reference_unresolved_steps"] == 16
    step = result["windows"][-1]["intermediate"][0]
    assert step["change_maximum_half_width"] >= step["anchor_maximum_half_width"] > 0
    assert result["online_ssvc"] == "NOT_CERTIFIED"


def test_cumulative_extension_reuses_prior_draws_without_resampling(tmp_path, monkeypatch):
    protocol = q6.absolute_protocol
    monkeypatch.setattr(q6, "absolute_protocol", lambda c, p: {**protocol(c, p), "looks": [4, 8]})
    config, first, first_root = setup_plan(tmp_path / "original", monkeypatch)
    measured = q6.measure_q6_absolute(
        config,
        plan_binding=first["plan_binding"],
        worker_root=first_root,
        out=tmp_path / "first_measurement",
        fixture=True,
    )
    second = q6.prepare_q6_observation(
        config,
        q6._bound_json(first["bindings"]),
        **first["inputs"],
        previous_measurement=measured["receipt"],
        out=tmp_path / "extension",
        fixture=True,
    )
    assert (second["draw_start"], second["draw_stop"]) == (4, 8)
    manifest = q6._bound_json(second["manifest"])
    root = tmp_path / "extension_samples"
    for worker in range(2):
        backend = v.PrefixObservationBackend(
            Adapter(),
            runtime_identity=manifest["runtime_identity"],
            policy_loader=lambda p: p["inference_fingerprint"],
            annotator=lambda text, scene: {"category": v.EVENTS[int(text)]},
            parity_tolerances=TOLERANCE,
        )
        v.execute_observation_tasks(manifest, backend, out=root, worker_index=worker)
    result = q6.measure_q6_absolute(
        config,
        plan_binding=second["plan_binding"],
        worker_root=root,
        out=tmp_path / "second_measurement",
        fixture=True,
    )
    with np.load(q6._read_binding(result["arrays"]), allow_pickle=False) as arrays:
        assert np.all(arrays["counts"].sum(-1) == 8)
        assert len(arrays["sample_seeds"]) == 16
    assert result["status"] == "REFERENCE_UNRESOLVED" and result["next_draws"] is None
    with pytest.raises(ValueError, match="same unresolved precision-only"):
        q6.prepare_q6_observation(
            config,
            q6._bound_json(second["bindings"]),
            **second["inputs"],
            previous_measurement=result["receipt"],
            out=tmp_path / "beyond",
            fixture=True,
        )


def test_manifest_cannot_relabel_a_different_policy_as_source64(tmp_path, monkeypatch):
    _, plan, _ = setup_plan(tmp_path, monkeypatch)
    manifest = copy.deepcopy(q6._bound_json(plan["manifest"]))
    manifest["policies"]["state_64"]["inference_fingerprint"] = "wrong-policy"
    manifest["manifest_hash"] = v.digest(
        {k: value for k, value in manifest.items() if k != "manifest_hash"}
    )
    atomic_json(tmp_path / "relabeled.json", manifest)
    with pytest.raises(ValueError, match="actual retained source checkpoint"):
        q6._manifest({**plan, "manifest": q6._binding(tmp_path / "relabeled.json")})


def test_execution_gate_checks_exact_bound_plan_before_model_loading(tmp_path, monkeypatch):
    config, plan, _ = setup_plan(tmp_path, monkeypatch)
    bindings = q6._bound_json(plan["bindings"])
    stage = q6._bound_json(bindings["v3_stage_lock"])
    manifest = q6._bound_json(plan["manifest"])
    checked = []
    # The production _plan rejects fixture originals (covered separately). This
    # isolated wiring test records the exact production flag and checks joins.
    monkeypatch.setattr(q6, "_cpu", lambda f: checked.append(f))
    monkeypatch.setattr(q6, "_plan", lambda c, b, f: (checked.append((b, f)), plan)[1])
    assert q6.verify_q6_observation_stage(config, bindings, stage, manifest) == plan
    assert checked[0] is False and checked[1][1] is False
    with pytest.raises(ValueError, match="bound original bytes"):
        q6.verify_q6_observation_stage(
            config, bindings, {**stage, "fit_binding": {"path": "other"}}, manifest
        )
    altered = copy.deepcopy(manifest)
    altered["runtime_identity"] = {"execution_kind": "wrong-runtime"}
    with pytest.raises(ValueError, match="exact completed"):
        q6.verify_q6_observation_stage(config, bindings, stage, altered)


def test_frozen_draw_ceiling_can_remain_unresolved_despite_zero_empirical_width():
    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    protocol = q6.absolute_protocol(config, config["qwen"]["probe_panel"]["prompts"])
    # Four arithmetic counts, not a 65k-draw experiment.
    n = protocol["looks"][-1]
    result = q6.absolute_statistics(np.array([[[0, 0, n, 0]]]), n, protocol)
    assert np.asarray(result["empirical_normal_half_width"]).max() == 0
    assert result["formal_hoeffding_half_width"] > 0.001
    assert result["status"] == "REFERENCE_UNRESOLVED" and result["next_draws"] is None
    assert not result["formal_precision_met"] and not result["empirical_precision_met"]
