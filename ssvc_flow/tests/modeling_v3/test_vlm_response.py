"""Tiny CPU fixtures for the actual-shard to locked-prediction artifact bridge."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_vlm_observation import backend, binding, prompt

from src.modeling_v3 import vlm_observation as v
from src.modeling_v3 import vlm_response as r
from src.modeling_v3.io import atomic_json, canonical_hash, finalize_run, verify_manifest
from src.modeling_v3.workflow import run_artifact_command


def fixture_campaign(tmp_path):
    import torch

    from src.r3_runtime import _save_tensor_payload

    config = json.loads(Path("configs/modeling_v3/protocol.json").read_text())
    config["qwen"]["observation_n_primary"] = 8
    config["qwen"]["probe_panel"]["prompts"] = 1
    config["qwen"]["training_bank_partition"]["heldout_banks"] = 1
    identity = {
        **r._identity(config),
        "unit": "V3_FORKS",
        "fixture": True,
        "execution_kind": "CPU_FIXTURE",
        "Q4_bridge": False,
    }
    forks = tmp_path / "forks"
    forks.mkdir()
    state = tmp_path / "state.bin"
    state.write_bytes(b"tiny fixture checkpoint")
    state_binding = {**binding(state), "identity": {"kind": "CPU_FIXTURE"}}
    policies = {"origin": {"checkpoint": state_binding, "inference_fingerprint": "A"}}
    banks = []
    for i, bank_id in enumerate(("C0", "C1", "H0")):
        checkpoints = {}
        for candidate, fingerprint in (("joint_0", "A"), ("joint_1", "B"), ("no_x_off_1", "C")):
            checkpoints[candidate] = {**state_binding, "inference_fingerprint": fingerprint}
            policies[bank_id + "/" + candidate] = {
                "checkpoint": state_binding,
                "inference_fingerprint": fingerprint,
            }
        # Realized coordinate vectors have exact contrast closure.
        left = np.array([1.0 + i, 0.25 * i, 0.0])
        right = np.array([0.0, 1.0 + i, 0.5 * i])
        contrasts = {}
        for name, vector in zip(
            [c[0] for c in v.Q4_CONTRASTS] + ["joint_0_minus_origin"],
            (left, right, left - right, np.array([0.1, 0.2, 0.3])),
            strict=True,
        ):
            path = forks / (bank_id + "_" + name + ".pt")
            saved = _save_tensor_payload(path, {"weight": torch.tensor(vector)}, identity)
            contrasts[name] = {
                **saved,
                "path": str(path),
                "parameter_order": ["weight"],
                "dimension": 3,
                "norm": float(np.linalg.norm(vector)),
            }
        banks.append(
            {
                "bank_id": bank_id,
                "role": "heldout" if bank_id == "H0" else "calibration_pool",
                "checkpoints": checkpoints,
                "contrasts": contrasts,
            }
        )
    origin = {"seed": 41001, "arm": "X_BASE", "step": 32}
    atomic_json(forks / "result.json", {"identity": identity, "banks": banks})
    atomic_json(
        forks / "bank_plan.json",
        {
            "origin_identity": origin,
            "banks": [{"bank_id": b["bank_id"], "role": b["role"]} for b in banks],
        },
    )
    finalize_run(forks, r._identity(config))
    prompts = tmp_path / "prompts.json"
    atomic_json(prompts, [prompt()])
    measured, _ = backend()
    measured.runtime_identity = {
        **r._identity(config),
        "fixture": True,
        "execution_kind": "CPU_FIXTURE",
    }
    return config, forks, policies, prompts, measured


def fixture_bundle(
    tmp_path, config, policies, prompts, measured, *, purpose, start=0, stop=8, suffix=""
):
    root = tmp_path / (purpose + suffix)
    root.mkdir()
    if purpose == "measurement":
        kinds = [("pilot", "ORIGIN", ["origin"], 0, 2), ("main", "ORIGIN", ["origin"], 2, 8)]
        kinds += [
            ("main", "DIRECT", ["H0/" + s], 0, 8) for s in ("joint_0", "joint_1", "no_x_off_1")
        ]
    else:
        kinds = [
            ("reference", "ORIGIN", ["origin"], start, stop),
            ("reference", "MIX", ["H0/joint_1", "H0/joint_0"], start, stop),
        ]
    tasks = [
        v.freeze_observation_task(
            operation="generate",
            origin_id="41001_X_BASE_32",
            candidate_id=sources[0],
            prompt_file=binding(prompts),
            prompt_ids=["prompt1"],
            role=role,
            draw_start=start,
            draw_stop=stop,
            rng_namespace="tiny-q5",
            proposal=proposal,
            proposal_candidates=sources,
            worker=0,
        )
        for role, proposal, sources, start, stop in kinds
    ]
    generation = v.freeze_task_manifest(tasks, policies, measured.runtime_identity, workers=1)
    atomic_json(root / "generation.json", generation)
    v.execute_observation_tasks(
        generation, measured, out=root / "generation", worker_index=0, workers=1
    )
    scoring_tasks = []
    for task in tasks:
        if task["proposal"] == "DIRECT":
            continue
        files = [
            binding(p)
            for p in (root / "generation" / "worker_0" / task["output_path"]).glob(
                "attempt_*/samples.jsonl"
            )
        ]
        fields = {
            key: value
            for key, value in task.items()
            if key not in {"task_id", "request_keys", "output_path"}
        }
        candidates = (
            policies if purpose == "measurement" else [p for p in policies if p.startswith("H0/")]
        )
        if task["proposal"] == "MIX":
            candidates = task["proposal_candidates"]
        for candidate in candidates:
            scoring_tasks.append(
                v.freeze_observation_task(
                    **{
                        **fields,
                        "operation": "score",
                        "candidate_id": candidate,
                        "sample_files": files,
                    }
                )
            )
    scoring = v.freeze_task_manifest(scoring_tasks, policies, measured.runtime_identity, workers=1)
    atomic_json(root / "scoring.json", scoring)
    v.execute_observation_tasks(scoring, measured, out=root / "scoring", worker_index=0, workers=1)
    return {
        "generation_manifest": binding(root / "generation.json"),
        "generation_root": str(root / "generation"),
        "scoring_manifest": binding(root / "scoring.json"),
        "scoring_root": str(root / "scoring"),
    }


def prepare_fit(tmp_path, config, forks, bundle, *, n_banks=2):
    geometry = r.prepare_q5_geometry(
        config,
        forks_root=forks,
        n_banks=n_banks,
        selector="FIRST",
        out=tmp_path / "geometry",
        fixture=True,
    )
    run_artifact_command("select", config, [geometry["spec"]["path"]], tmp_path / "selection")
    return r.prepare_q5_fit(
        config,
        forks_root=forks,
        measurement_bundle=bundle,
        geometry_spec=geometry["spec"],
        selection_root=tmp_path / "selection",
        model_spec={},
        out=tmp_path / "fit_input",
        fixture=True,
    )


def test_actual_shards_geometry_fit_prediction_then_independent_evaluation(tmp_path, monkeypatch):
    config, forks, policies, prompts, measured = fixture_campaign(tmp_path)
    bundle = fixture_bundle(tmp_path, config, policies, prompts, measured, purpose="measurement")
    opened = []
    original = r._completed_task_rows

    def tracked(root, manifest, *, allowed_task_ids=None):
        opened.extend(t for t in manifest["tasks"] if t["task_id"] in allowed_task_ids)
        return original(root, manifest, allowed_task_ids=allowed_task_ids)

    monkeypatch.setattr(r, "_completed_task_rows", tracked)
    fit = prepare_fit(tmp_path, config, forks, bundle)
    assert all(
        not t["candidate_id"].startswith("H0/") and t["proposal"] != "DIRECT" for t in opened
    )
    with np.load(r._read(fit["spec"])["arrays"]["path"]) as arrays:
        assert arrays["responses"].shape == (6, 1, 4)
        covariance = arrays["covariance"][0]
        assert covariance.shape == (24, 24)
        assert np.any(covariance[:12, 12:] != 0), "shared draws require cross-bank covariance"
        np.testing.assert_allclose(
            arrays["raw_main_contributions"][:, :, 0], arrays["raw_main_contributions"][:, :, 3]
        )
    run_artifact_command("fit", config, [fit["spec"]["path"]], tmp_path / "fit")
    frozen = r.freeze_q5_predictions(
        config,
        fit_spec=fit["spec"],
        fit_root=tmp_path / "fit",
        out=tmp_path / "predictions",
        fixture=True,
    )
    reference = fixture_bundle(tmp_path, config, policies, prompts, measured, purpose="reference")
    result = r.prepare_q5_evaluation(
        config,
        prediction_lock=frozen["prediction_lock"],
        measurement_bundle=bundle,
        reference_bundle=reference,
        out=tmp_path / "evaluation",
        fixture=True,
    )
    verify_manifest(tmp_path / "evaluation")
    spec = r._read(result["spec"])
    with np.load(spec["arrays"]["path"]) as arrays:
        assert not arrays["accepted"].any()
        assert np.isnan(arrays["reference"]).all()
        assert np.isfinite(arrays["reference_estimate_unmasked"]).all()
        assert np.isfinite(arrays["direct_count_measurement"]).all()
        np.testing.assert_allclose(arrays["delta_v_predictions"], -arrays["predictions"][..., 3])
    receipt = r._read(result["receipt"])
    assert receipt["accepted_count"] == receipt["resolved_reference_cells"] == 0
    assert receipt["reference_status_counts"] == {"REFERENCE_UNRESOLVED": 3}
    precision = r._read(result["precision_receipt"])
    assert precision["current_draws"] == 8 and precision["next_draws"] == 4096
    assert precision["precision_only"] and not precision["predictor_rankings_used"]
    assert "unit_reports" in r._read(precision["decision_evidence"])
    run_artifact_command("evaluate", config, [result["spec"]["path"]], tmp_path / "evaluated")

    def unresolved(*_args, **_kwargs):
        raise r.UnresolvedObservation(
            np.array([[float("inf"), 0.0, 0.0, 0.0]]), {"reasons": ["NONFINITE_ORIGIN_WEIGHT"]}
        )

    monkeypatch.setattr(r, "_observation_estimate", unresolved)
    unresolved_result = r.prepare_q5_evaluation(
        config,
        prediction_lock=frozen["prediction_lock"],
        measurement_bundle=bundle,
        reference_bundle=reference,
        out=tmp_path / "unresolved",
        fixture=True,
    )
    with np.load(r._read(unresolved_result["spec"])["arrays"]["path"]) as arrays:
        assert np.isnan(arrays["query_direct_measurement"]).all()
        assert np.isfinite(arrays["direct_count_measurement"]).all()
        assert np.isinf(arrays["query_unresolved_raw_0"]).any()


def test_geometry_never_reads_heldout_vectors(tmp_path, monkeypatch):
    config, forks, *_ = fixture_campaign(tmp_path)
    from src import r3_runtime

    original = r3_runtime._load_tensor_payload

    def guarded(path, *args):
        assert not path.name.startswith("H0_")
        return original(path, *args)

    monkeypatch.setattr(r3_runtime, "_load_tensor_payload", guarded)
    r.prepare_q5_geometry(
        config, forks_root=forks, n_banks=1, selector="FIRST", out=tmp_path / "geo", fixture=True
    )


def test_unselected_and_heldout_score_shards_unopened_before_freeze(tmp_path):
    config, forks, policies, prompts, measured = fixture_campaign(tmp_path)
    bundle = fixture_bundle(tmp_path, config, policies, prompts, measured, purpose="measurement")
    score = r._read(bundle["scoring_manifest"])
    for task in score["tasks"]:
        if task["candidate_id"].startswith(("C1/", "H0/")):
            for path in (Path(bundle["scoring_root"]) / "worker_0" / task["output_path"]).glob(
                "attempt_*/samples.jsonl"
            ):
                path.write_bytes(b"DO NOT OPEN SEMANTIC LABELS BEFORE LOCK\n")
    prepare_fit(tmp_path, config, forks, bundle, n_banks=1)


def test_missing_prediction_lock_refused_before_reference_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        r,
        "_bundle",
        lambda *_args, **_kwargs: pytest.fail("Reference opened before prediction lock"),
    )
    with pytest.raises(FileNotFoundError):
        r.prepare_q5_evaluation(
            {},
            prediction_lock=tmp_path / "missing.json",
            measurement_bundle={},
            reference_bundle={},
            out=tmp_path / "eval",
            fixture=True,
        )


def test_real_vectors_reject_local_cpu_and_fixture_cannot_launder_real(tmp_path, monkeypatch):
    config, forks, *_ = fixture_campaign(tmp_path)
    monkeypatch.setattr(r.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="server CPU"):
        r.prepare_q5_geometry(
            config, forks_root=forks, n_banks=1, selector="FIRST", out=tmp_path / "geo"
        )
    result = r._read(forks / "result.json")
    result["identity"].update(fixture=False, execution_kind="REAL_CUDA_MODEL")
    (forks / "result.json").write_text(json.dumps(result))
    for name in ("RUN_MANIFEST.json", "COMPLETE.json"):
        (forks / name).unlink()
    finalize_run(forks, r._identity(config))
    with pytest.raises(ValueError, match="cannot consume"):
        r.prepare_q5_geometry(
            config,
            forks_root=forks,
            n_banks=1,
            selector="FIRST",
            out=tmp_path / "geo",
            fixture=True,
        )


def test_vector_payload_identity_cannot_be_rebound_by_file_hash(tmp_path):
    import torch

    config, forks, *_ = fixture_campaign(tmp_path)
    loaded = r._forks(forks, config, fixture=True)
    spec = loaded["banks"]["C0"]["contrasts"][v.Q4_CONTRASTS[0][0]]
    value = torch.load(spec["path"], weights_only=True)
    value["identity"]["config_hash"] = canonical_hash("other config")
    torch.save(value, spec["path"])
    spec["sha256"] = binding(spec["path"])["sha256"]
    with pytest.raises(ValueError, match="state identity"):
        r._vectors(loaded, ["C0"])


def test_reference_batches_merge_only_contiguous_cumulative_streams(tmp_path):
    config, _, policies, prompts, measured = fixture_campaign(tmp_path)
    first = fixture_bundle(tmp_path, config, policies, prompts, measured, purpose="reference")
    second = fixture_bundle(
        tmp_path,
        config,
        policies,
        prompts,
        measured,
        purpose="reference",
        start=8,
        stop=16,
        suffix="2",
    )
    kwargs = {
        "origin_id": "41001_X_BASE_32",
        "policies": policies,
        "runtime_identity": measured.runtime_identity,
        "candidates": set(policies),
        "logs": [],
    }
    _, grouped, _ = r._reference_batches({"batches": [first, second]}, config, **kwargs)
    assert len(grouped["prompt1", "reference", "ORIGIN"]) == 16
    with pytest.raises(ValueError, match="repeat"):
        r._reference_batches({"batches": [first, first]}, config, **kwargs)
    gap = fixture_bundle(
        tmp_path,
        config,
        policies,
        prompts,
        measured,
        purpose="reference",
        start=9,
        stop=16,
        suffix="gap",
    )
    with pytest.raises(ValueError, match="contiguous"):
        r._reference_batches({"batches": [first, gap]}, config, **kwargs)
