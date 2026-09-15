"""Small immutable VLM development-selection and prediction-gate fixtures."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.modeling_v3 import vlm_response as r
from src.modeling_v3.io import atomic_json, atomic_npz, canonical_hash, finalize_run


def _grouped_campaign(tmp_path):
    from test_vlm_response import fixture_campaign

    config, forks, policies, prompts, measured = fixture_campaign(tmp_path)
    plan = r._read(forks / "bank_plan.json")
    train = []
    for index, bank in enumerate(plan["banks"]):
        bank["prompt_ids"] = [f"train-{index}-{j}" for j in range(4)]
        train.extend(
            {"prompt_id": p, "family": f"family-{index % 2}", "interface": "SYMBOLIC_FRESH"}
            for p in bank["prompt_ids"]
        )
    plan["train_prompts"] = train
    grouped_forks = tmp_path / "grouped_forks"
    grouped_forks.mkdir()
    atomic_json(grouped_forks / "result.json", r._read(forks / "result.json"))
    atomic_json(grouped_forks / "bank_plan.json", plan)
    finalize_run(grouped_forks, r._identity(config))
    rows = r._read(prompts)
    rows[0]["family"] = "family-0"
    grouped_prompts = tmp_path / "grouped_prompts.json"
    atomic_json(grouped_prompts, rows)
    return config, grouped_forks, policies, grouped_prompts, measured


def test_actual_vlm_geometry_preserves_bank_composition_strata(tmp_path):
    from src.modeling_v3.workflow import run_artifact_command

    config, forks, *_ = _grouped_campaign(tmp_path)
    result = r.prepare_q5_geometry(
        config,
        forks_root=forks,
        n_banks=1,
        selector="STRATIFIED_RANDOM",
        out=tmp_path / "geometry",
        fixture=True,
    )
    spec = r._read(result["spec"])
    assert len(spec["strata"]) == 2 and len(set(spec["strata"])) == 2
    assert spec["bank_stratification"]["bank_plan"] == r._binding(forks / "bank_plan.json")
    assert spec["bank_stratification"]["bank_groups"][0] == [["family-0", "SYMBOLIC_FRESH"]] * 4
    run_artifact_command("select", config, [result["spec"]["path"]], tmp_path / "selected")
    chosen = r._read(tmp_path / "selected" / "SELECTION.json")
    assert chosen["audit"]["stratification"] == "SUPPLIED_METADATA"


def test_actual_vlm_group_fit_passes_frozen_probe_map(tmp_path):
    from test_vlm_response import fixture_bundle

    from src.modeling_v3.workflow import run_artifact_command

    config, forks, policies, prompts, measured = _grouped_campaign(tmp_path)
    bundle = fixture_bundle(tmp_path, config, policies, prompts, measured, purpose="measurement")
    geometry = r.prepare_q5_geometry(
        config,
        forks_root=forks,
        n_banks=2,
        selector="FIRST",
        out=tmp_path / "geometry",
        fixture=True,
    )
    run_artifact_command("select", config, [geometry["spec"]["path"]], tmp_path / "selected")
    result = r.prepare_q5_fit(
        config,
        forks_root=forks,
        measurement_bundle=bundle,
        geometry_spec=geometry["spec"],
        selection_root=tmp_path / "selected",
        model_spec={"method": "GROUP_WEIGHTED_RESPONSE", "rank_cap": 1},
        out=tmp_path / "fit_input",
        fixture=True,
    )
    spec = r._read(result["spec"])
    with np.load(spec["arrays"]["path"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["group_map"], [[1, 0, 0, 0], [0, 0, 0, -1]])
    assert spec["probe_groups"] == [["family-0", "SYMBOLIC_FRESH"]]
    assert spec["probe_grouping"]["prompt_file"] == r._binding(prompts)
    run_artifact_command("fit", config, [result["spec"]["path"]], tmp_path / "fit")
    model = r._read(tmp_path / "fit" / "MODEL.json")
    assert model["metadata"]["direction"]["group_weighting"] == "CALLER_FIXED_GROUP_MAP"


def test_group_metadata_averages_same_group_before_direction_selection(tmp_path):
    from src.modeling_v3.response_models import fit_response_model

    path = tmp_path / "probes.json"
    atomic_json(
        path,
        [{"prompt_id": p, "family": "one", "interface": "SYMBOLIC_FRESH"} for p in ("p1", "p2")],
    )
    manifest = {"tasks": [{"prompt_file": r._binding(path)}]}
    groups, mapping, metadata = r._probe_group_metadata(manifest, ["p1", "p2"])
    assert groups == [["one", "SYMBOLIC_FRESH"]] * 2
    assert metadata["group_order"] == [["one", "SYMBOLIC_FRESH"]]
    np.testing.assert_array_equal(
        mapping, [[0.5, 0, 0, 0, 0.5, 0, 0, 0], [0, 0, 0, -0.5, 0, 0, 0, -0.5]]
    )
    y = np.zeros((2, 2, 4))
    y[0, :, 0], y[0, :, 3] = [1, -1], [-1, 1]
    y[1, 1, 0], y[1, 1, 3] = 0.5, -0.5
    model = fit_response_model(np.eye(2), y, "GROUP_WEIGHTED_RESPONSE", 1, group_map=mapping)
    np.testing.assert_allclose(abs(model["directions"][:, 0]), [0, 1], atol=1e-12)


@pytest.mark.parametrize("method", ["STRATIFIED_RANDOM", "GROUP_WEIGHTED_RESPONSE"])
def test_required_vlm_group_metadata_missing_is_rejected(tmp_path, method):
    from test_vlm_response import fixture_bundle, fixture_campaign

    from src.modeling_v3.workflow import run_artifact_command

    config, forks, policies, prompts, measured = fixture_campaign(tmp_path)
    if method == "STRATIFIED_RANDOM":
        with pytest.raises(ValueError, match=r"group|metadata"):
            r.prepare_q5_geometry(
                config,
                forks_root=forks,
                n_banks=1,
                selector=method,
                out=tmp_path / "geometry",
                fixture=True,
            )
        return
    geometry = r.prepare_q5_geometry(
        config,
        forks_root=forks,
        n_banks=2,
        selector="FIRST",
        out=tmp_path / "geometry",
        fixture=True,
    )
    run_artifact_command("select", config, [geometry["spec"]["path"]], tmp_path / "selected")
    bundle = fixture_bundle(tmp_path, config, policies, prompts, measured, purpose="measurement")
    with pytest.raises(ValueError, match=r"group|metadata"):
        r.prepare_q5_fit(
            config,
            forks_root=forks,
            measurement_bundle=bundle,
            geometry_spec=geometry["spec"],
            selection_root=tmp_path / "selected",
            model_spec={"method": method, "rank_cap": 1},
            out=tmp_path / "fit_input",
            fixture=True,
        )


def test_production_metadata_producers_match_response_group_contract(tmp_path):
    """Execute the actual prompt/bank producers using metadata only, no VLM."""
    from test_vlm_campaign import original_prompts

    from src.modeling_v3 import vlm_campaign as campaign
    from src.prompts import build_prompt
    from src.r4_inputs import DEV_CELLS, INTERFACES

    train = original_prompts()
    plan = campaign.build_bank_plan(train, origin_identity={"state_hash": "metadata-fixture"})
    atomic_json(tmp_path / "bank_plan.json", plan)
    bank_ids = [bank["bank_id"] for bank in plan["banks"] if bank["role"] == "calibration_pool"]
    strata, metadata = r._bank_stratification({"root": tmp_path, "plan": plan}, bank_ids)
    assert len(strata) == 24 and len(set(strata)) == 3
    assert all(len(group) == 4 for group in metadata["bank_groups"])
    by_family = {row["family"]: row["scene"] for row in train}
    scenes = []
    for family, chart, operation in DEV_CELLS:
        scene = copy.deepcopy(by_family[family])
        scene.update(
            base_scene_id=f"metadata-{family}-{chart}-{operation}",
            split="control",
            interface=None,
            constraint_family=family,
            chart_type=chart,
            operation=operation,
        )
        scene["prompt_hashes"] = {
            interface: build_prompt(scene, interface)["prompt_hash"] for interface in INTERFACES
        }
        scenes.append(scene)
    probes = campaign.build_probe_panel(scenes, excluded_base_scene_ids=[])
    path = tmp_path / "PROBES.json"
    atomic_json(path, probes)
    groups, mapping, metadata = r._probe_group_metadata(
        {"tasks": [{"prompt_file": r._binding(path)}]}, [p["prompt_id"] for p in probes]
    )
    assert len(groups) == 36 and len(metadata["group_order"]) == 6
    assert mapping.shape == (12, 144)
    np.testing.assert_allclose(mapping[::2].sum(axis=1), 1)
    np.testing.assert_allclose(mapping[1::2].sum(axis=1), -1)
    assert all(groups.count(group) == 6 for group in metadata["group_order"])


def test_probe_group_metadata_rejects_different_or_changed_original(tmp_path):
    path = tmp_path / "PROBES.json"
    atomic_json(path, [{"prompt_id": "p", "family": "f", "interface": "i"}])
    bound = r._binding(path)
    with pytest.raises(ValueError, match="identical"):
        r._probe_group_metadata(
            {"tasks": [{"prompt_file": bound}, {"prompt_file": {**bound, "sha256": "changed"}}]},
            ["p"],
        )
    path.write_text("[]")
    with pytest.raises(ValueError, match="hash"):
        r._probe_group_metadata({"tasks": [{"prompt_file": bound}]}, ["p"])


def selection_fixture(tmp_path):
    config = json.loads(Path("configs/modeling_v3/protocol.json").read_text())
    identity = r._identity(config)
    cpu_selected = {
        "models": ["FULL_RIDGE", "PCA"],
        "rank_caps": ["FULL", 1],
        "alpha": 1e-5,
        "observation_methods": ["PILOT_SHRINK_ZERO_SUM", "RAW4"],
        "selection_rules": ["FIRST", "BLOCK_PIVOT_QR"],
        "rho_threshold": 0.5,
        "leverage_threshold": 100.0,
    }
    cpu = tmp_path / "cpu.json"
    atomic_json(
        cpu,
        {
            "schema": "ssvc-v3-selection-lock-1",
            "config_sha256": identity["config_hash"],
            "source_sha256": identity["source_hash"],
            "selected": cpu_selected,
            "selection_hash": canonical_hash(cpu_selected),
            "development_evidence": {},
            "locked_test_opened": False,
        },
    )
    design = {
        "design_id": "full-pilot-first-2",
        "method": "FULL_RIDGE",
        "rank_cap": "FULL",
        "alpha": 1e-5,
        "output_policy": "RAW4",
        "regression": "RIDGE",
        "observation_method": "PILOT_SHRINK_ZERO_SUM",
        "selector": "FIRST",
        "n_banks": 2,
        "selection_seed": 2026091500,
    }
    selected = {
        "designs": [design],
        "rho_threshold": 0.5,
        "leverage_threshold": 100.0,
        "pointwise_criteria": {
            "primary_tolerance": 0.001,
            "reference_half_width_max": 0.00025,
            "minimum_geometry_coverage": 0.2,
            "maximum_q95_absolute_residual": 0.001,
            "maximum_nrmse": None,
        },
    }
    inputs, tasks = [], {}
    for seed in config["qwen"]["seed_roles"]["development"]:
        for step in (32, 96):
            origin = f"{seed}_X_BASE_{step}"
            root = tmp_path / origin
            root.mkdir()
            geometry = root / "geometry.json"
            atomic_json(
                geometry,
                {
                    **identity,
                    "kind": "V3_Q5_CALIBRATION_GEOMETRY",
                    "origin_id": origin,
                    "role": "development",
                    "fixture": True,
                    "method": "FIRST",
                    "n_banks": 2,
                    "seed": 2026091500,
                },
            )
            fit = root / "fit.json"
            atomic_json(
                fit,
                {
                    **identity,
                    **design,
                    "kind": "V3_Q5_RESPONSE_FIT",
                    "origin_id": origin,
                    "role": "development",
                    "fixture": True,
                    "geometry": r._binding(geometry),
                },
            )
            array = root / "arrays.npz"
            atomic_npz(
                array, {"predictions": np.zeros((1, 1, 4)), "reference": np.zeros((1, 1, 4))}
            )
            pred = root / "prediction.json"
            atomic_json(
                pred,
                {
                    **identity,
                    "kind": "V3_FROZEN_PREDICTIONS",
                    "origin_id": origin,
                    "role": "development",
                    "fixture": True,
                    "fit_spec": r._binding(fit),
                    "arrays": r._binding(array),
                    "heldout_labels_read": False,
                    "reference_labels_read": False,
                },
            )
            spec = root / "EVALUATION_INPUT.json"
            atomic_json(
                spec,
                {
                    **identity,
                    "kind": "V3_Q5_REFERENCE_EVALUATION_INPUT",
                    "origin_id": origin,
                    "role": "development",
                    "fixture": True,
                    "prediction_lock": r._binding(pred),
                    "arrays": r._binding(array),
                },
            )
            receipt = root / "RESPONSE_EVALUATION_RECEIPT.json"
            atomic_json(
                receipt,
                {
                    **identity,
                    "kind": "V3_RESPONSE_EVALUATION",
                    "origin_id": origin,
                    "role": "development",
                    "fixture": True,
                    "evaluation_input": r._binding(spec),
                    "fit_spec": r._binding(fit),
                    "prediction_binding": r._binding(pred),
                },
            )
            finalize_run(root, identity)
            inputs.append(r._binding(spec))
            tasks["evaluation_" + origin] = {"kind": "evaluation", "binding": r._binding(receipt)}
    done = tmp_path / "completion"
    done.mkdir()
    atomic_json(
        done / "Q5_STAGE_COMPLETION.json",
        {
            **identity,
            "kind": "V3_Q5_STAGE_COMPLETION",
            "status": "COMPLETE",
            "role": "development",
            "fixture": True,
            "verified_tasks": tasks,
            "expected_task_ids": sorted(tasks),
            "completed_task_ids": sorted(tasks),
            "technical_failures": {},
        },
    )
    finalize_run(done, identity)
    return config, r._binding(cpu), r._binding(done / "Q5_STAGE_COMPLETION.json"), inputs, selected


def freeze(tmp_path, values):
    config, cpu, completion, inputs, selected = values
    return r.freeze_vlm_selection(
        config,
        cpu,
        development_completion=completion,
        evaluation_inputs=inputs,
        selected=selected,
        out=tmp_path / "selection",
        fixture=True,
    )


def test_vlm_selection_freezes_all_three_dev_seeds_and_parent_cpu(tmp_path):
    values = selection_fixture(tmp_path)
    result = freeze(tmp_path, values)
    lock = r.verify_vlm_selection_lock(values[0], result["selection_lock"], fixture=True)
    assert lock["parent_cpu_selection"] == values[1]
    assert lock["development_completion"] == values[2]
    assert lock["development_seeds"] == [41001, 41002, 41003]
    assert not lock["locked_test_opened"] and lock["cross_seed_95"] == "NOT_CERTIFIED"
    assert lock["selection_hash"] == canonical_hash(values[-1])
    altered = copy.deepcopy(lock)
    altered["criteria_scope"]["maximum_nrmse"] = "ignore reference uncertainty"
    with pytest.raises(ValueError, match="mismatch"):
        r.verify_vlm_selection_lock(values[0], altered, fixture=True)
    with pytest.raises(ValueError, match="fixture"):
        r.verify_vlm_selection_lock(values[0], result["selection_lock"])


@pytest.mark.parametrize("method", ["ZERO", "FULL_RIDGE", "FULL_GLS", "RBF_RIDGE"])
def test_full_baseline_rank_matches_actual_cpu_matrix(tmp_path, method):
    from src.modeling_v3.cpu_campaign import development_designs

    config, cpu, _, _, selected = selection_fixture(tmp_path)
    parent = r._read(cpu["path"])
    parent["selected"].update(models=[method, "PCA"], rank_caps=[1, 2])
    selected["designs"][0]["method"] = method
    generated = development_designs(config, selected=parent["selected"])
    assert {d["rank_cap"] for d in generated if d["model"] == method} == {"FULL"}
    r._validate_vlm_selected(config, selected, parent)
    selected["designs"][0]["rank_cap"] = 1
    with pytest.raises(ValueError, match="parent CPU"):
        r._validate_vlm_selected(config, selected, parent)


@pytest.mark.parametrize("method", ["PCA", "RANDOM_Q", "RESPONSE_SVD", "GROUP_WEIGHTED_RESPONSE"])
def test_direction_rank_stays_inside_parent_cpu_selection(tmp_path, method):
    config, cpu, _, _, selected = selection_fixture(tmp_path)
    parent = r._read(cpu["path"])
    parent["selected"].update(models=[method], rank_caps=[1, 2])
    selected["designs"][0].update(method=method, rank_cap=1)
    r._validate_vlm_selected(config, selected, parent)
    for rank in (4, "FULL"):
        selected["designs"][0]["rank_cap"] = rank
        with pytest.raises(ValueError, match="parent CPU"):
            r._validate_vlm_selected(config, selected, parent)


def test_frozen_scope_declares_actual_all_channel_qualification():
    from src.modeling_v3.vlm_results import _metric

    scope = r.VLM_CRITERIA_SCOPE
    assert scope["qualification_channels"] == ["delta_pX", "delta_pS", "delta_pW", "delta_pI"]
    assert scope["primary_reporting_channels"] == ["delta_pX", "delta_v=-delta_pI"]
    reference = np.array([[0.01, 0.02, -0.01, -0.02]])
    prediction = reference + np.array([[0, 0.01, -0.01, 0]])
    metric = _metric(
        prediction, reference, np.zeros((1, 4)), np.full((1, 4), 1e-6), np.array([True])
    )
    assert metric["pX_v"]["q95_absolute_residual"] == 0
    assert metric["accepted_metrics"][
        "q95_absolute_residual_plus_reference_half_width"
    ] == pytest.approx(0.010001)


def test_selection_requires_every_design_on_all_dev_origins(tmp_path):
    values = list(selection_fixture(tmp_path))
    values[3] = values[3][:-1]
    with pytest.raises(ValueError, match=r"origin|development"):
        freeze(tmp_path, values)


@pytest.mark.parametrize("change", ["test_role", "unrun_design", "missing_criterion", "bad_rank"])
def test_selection_rejects_test_evidence_and_undeclared_designs(tmp_path, change):
    values = list(selection_fixture(tmp_path))
    if change == "test_role":
        target = Path(values[3][0]["path"])
        value = r._read(target)
        value["role"] = "locked_test"
        target.write_text(json.dumps(value))
        values[3][0] = r._binding(target)
    elif change == "unrun_design":
        values[-1]["designs"][0]["observation_method"] = "RAW4"
    elif change == "missing_criterion":
        del values[-1]["pointwise_criteria"]["primary_tolerance"]
    else:
        values[-1]["designs"][0]["rank_cap"] = 99
    with pytest.raises((ValueError, PermissionError)):
        freeze(tmp_path, values)


def test_locked_selection_verification_rechecks_original_bytes(tmp_path):
    values = selection_fixture(tmp_path)
    result = freeze(tmp_path, values)
    Path(values[3][0]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        r.verify_vlm_selection_lock(values[0], result["selection_lock"], fixture=True)


def test_nondev_methods_must_match_one_whole_frozen_design(tmp_path):
    values = selection_fixture(tmp_path)
    result = freeze(tmp_path, values)
    config = values[0]
    settings = copy.deepcopy(values[-1]["designs"][0])
    r._method_lock(config, "interval_calibration", settings, result["selection_lock"], fixture=True)
    settings["n_banks"] = 4
    with pytest.raises(PermissionError, match="design"):
        r._method_lock(config, "locked_test", settings, result["selection_lock"], fixture=True)
    with pytest.raises(PermissionError, match="VLM"):
        r._method_lock(config, "interval_calibration", {}, values[1], fixture=True)


def prediction_fixture(tmp_path, config, role):
    identity = r._identity(config)
    inputs, model = tmp_path / "fit_input", tmp_path / "model"
    inputs.mkdir()
    model.mkdir()
    seed = config["qwen"]["seed_roles"][role][0]
    spec = {
        **identity,
        "kind": "V3_Q5_RESPONSE_FIT",
        "fixture": True,
        "role": role,
        "origin_id": f"{seed}_X_BASE_32",
        "query_units": [{"bank_id": "H0"}, {"bank_id": "H1"}],
        "probe_ids": ["p"],
        "method": "FULL_RIDGE",
        "rank_cap": "FULL",
        "alpha": 1e-5,
        "output_policy": "RAW4",
        "regression": "RIDGE",
        "observation_method": "RAW4",
        "selection_lock": None,
        "design_id": "fixture-design",
    }
    atomic_json(inputs / "FIT_SPEC.json", spec)
    finalize_run(inputs, identity)
    binding = r._binding(inputs / "FIT_SPEC.json")
    atomic_npz(model / "MODEL_ARRAYS.npz", {"predictions": np.zeros((2, 1, 4))})
    atomic_json(
        model / "RECEIPT.json",
        {"command": "fit", "input_binding": {"spec_sha256": binding["sha256"]}},
    )
    finalize_run(model, identity)
    return binding, model


def test_calibration_predictions_do_not_depend_on_calibration_receipt(tmp_path):
    config = json.loads(Path("configs/modeling_v3/protocol.json").read_text())
    spec, model = prediction_fixture(tmp_path, config, "interval_calibration")
    result = r.freeze_q5_predictions(
        config, fit_spec=spec, fit_root=model, out=tmp_path / "pred", fixture=True
    )
    lock = r._read(result["prediction_lock"])
    assert lock["accepted_count"] == 0 and lock["calibration_receipt"] is None


def test_test_prediction_requires_calibration_even_for_empty_acceptance(tmp_path):
    config = json.loads(Path("configs/modeling_v3/protocol.json").read_text())
    spec, model = prediction_fixture(tmp_path, config, "locked_test")
    with pytest.raises(PermissionError, match="calibration"):
        r.freeze_q5_predictions(
            config, fit_spec=spec, fit_root=model, out=tmp_path / "pred", fixture=True
        )


def test_prediction_freezes_and_reverifies_geometry_acceptance_without_test_labels(
    tmp_path, monkeypatch
):
    config = json.loads(Path("configs/modeling_v3/protocol.json").read_text())
    spec, model = prediction_fixture(tmp_path, config, "locked_test")
    cal = tmp_path / "cal.json"
    atomic_json(cal, {"kind": "CPU_FIXTURE_CALIBRATION"})
    cal_binding = r._binding(cal)
    accepted = np.array([[True], [False]])
    calls = []

    def derive(_config, selection, calibration, fit, model_arrays, *, fixture):
        calls.append((calibration, fit, model_arrays))
        assert fixture and calibration == cal_binding and fit == spec
        return {
            "accepted": accepted.copy(),
            "rho": np.array([0.1, 0.8]),
            "leverage": np.array([1.0, 2.0]),
            "e_norm": np.array([1.0, 1.0]),
            "selection_binding": selection,
            "calibration_binding": calibration,
            "design_id": "fixture-design",
            "acceptance_kind": "V3_VLM_EMPIRICAL_GEOMETRY_ACCEPTANCE",
            "cross_seed_95": "NOT_CERTIFIED",
        }

    # Dependency seam only; coverage tests the actual empirical calibration rule.
    monkeypatch.setitem(
        sys.modules, "src.modeling_v3.vlm_results", SimpleNamespace(derive_vlm_acceptance=derive)
    )
    result = r.freeze_q5_predictions(
        config,
        fit_spec=spec,
        fit_root=model,
        out=tmp_path / "pred",
        calibration_receipt=cal_binding,
        fixture=True,
    )
    lock = r._read(result["prediction_lock"])
    with np.load(lock["arrays"]["path"]) as arrays:
        np.testing.assert_array_equal(arrays["accepted"], accepted)
        r.verify_vlm_acceptance_receipt(
            config,
            lock["acceptance_receipt"],
            prediction=arrays["predictions"],
            accepted=arrays["accepted"],
            fixture=True,
        )
        with pytest.raises(ValueError, match="identity"):
            r.verify_vlm_acceptance_receipt(
                config,
                lock["acceptance_receipt"],
                prediction=arrays["predictions"],
                accepted=~accepted,
                fixture=True,
            )
    assert len(calls) == 2 and lock["accepted_count"] == 1
    assert not lock["reference_labels_read"] and lock["cross_seed_95"] == "NOT_CERTIFIED"


def test_real_calibration_adapter_integrates_with_prediction_freeze(tmp_path, monkeypatch):
    from test_vlm_results import calibrated, origin

    config, selection, design, calibration, _, _ = calibrated(tmp_path / "calibration")
    evaluated = origin(
        tmp_path / "test",
        config,
        selection,
        design,
        43001,
        "X_BASE",
        32,
        calibration=calibration["receipt"],
    )
    evaluation = r._read(evaluated)
    existing = r._read(evaluation["prediction_lock"])
    model_root = Path(existing["model_arrays"]["path"]).parent
    forbidden = Path(evaluated["path"]).parent
    old_read = Path.read_text

    def no_test_reference_read(path, *args, **kwargs):
        assert forbidden not in path.parents, "test reference opened during acceptance freeze"
        return old_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_test_reference_read)
    frozen = r.freeze_q5_predictions(
        config,
        fit_spec=existing["fit_spec"],
        fit_root=model_root,
        calibration_receipt=calibration["receipt"],
        out=tmp_path / "frozen",
        fixture=True,
    )
    lock = r._read(frozen["prediction_lock"])
    assert lock["accepted_count"] == 3
    with np.load(lock["arrays"]["path"]) as data:
        check = r.verify_vlm_acceptance_receipt(
            config,
            lock["acceptance_receipt"],
            prediction=data["predictions"],
            accepted=data["accepted"],
            fixture=True,
        )
    assert check["cross_seed_95"] == "NOT_CERTIFIED"


def test_reference_authorization_binds_exact_design_prediction_before_labels(tmp_path):
    from src.modeling_v3 import vlm_observation as observation

    values = selection_fixture(tmp_path)
    selection = freeze(tmp_path, values)["selection_lock"]
    config, _, _, _, selected = values
    identity = r._identity(config)
    pred_root = tmp_path / "prediction"
    pred_root.mkdir()
    prediction = {
        "kind": "V3_FROZEN_PREDICTIONS",
        **identity,
        "fixture": True,
        "role": "interval_calibration",
        "origin_id": "42001_X_BASE_32",
        "selection_lock": selection,
        "design_id": selected["designs"][0]["design_id"],
    }
    atomic_json(pred_root / "PREDICTION_LOCK.json", prediction)
    finalize_run(pred_root, identity)
    pred_binding = r._binding(pred_root / "PREDICTION_LOCK.json")
    set_root = tmp_path / "prediction_set"
    set_root.mkdir()
    atomic_json(
        set_root / "PREDICTION_SET.json",
        {
            **prediction,
            "kind": "V3_FROZEN_PREDICTION_SET",
            "predictions": {prediction["design_id"]: pred_binding},
        },
    )
    finalize_run(set_root, identity)
    set_binding = r._binding(set_root / "PREDICTION_SET.json")
    stage = tmp_path / "reference_stage.json"
    atomic_json(stage, {"prediction_binding": set_binding})
    prompt = tmp_path / "prompt.json"
    atomic_json(prompt, [])
    state = tmp_path / "state.bin"
    state.write_bytes(b"CPU fixture state")
    rows = tmp_path / "not-yet-generated.jsonl"
    rows.write_bytes(b"")
    policies = {"origin": {"checkpoint": r._binding(state), "inference_fingerprint": "fixture"}}
    bundle = {}
    for operation in ("generation", "scoring"):
        task = observation.freeze_observation_task(
            operation="generate" if operation == "generation" else "score",
            origin_id=prediction["origin_id"],
            candidate_id="origin",
            prompt_file=r._binding(prompt),
            prompt_ids=["p0"],
            role="reference",
            draw_start=0,
            draw_stop=2,
            rng_namespace="reference-fixture",
            proposal="ORIGIN",
            proposal_candidates=["origin"],
            worker=0,
            sample_files=[] if operation == "generation" else [r._binding(rows)],
        )
        manifest = observation.freeze_task_manifest([task], policies, identity, workers=1)
        target = tmp_path / (operation + ".json")
        atomic_json(target, manifest)
        root = tmp_path / operation
        task_root = root / "worker_0" / task["output_path"]
        task_root.mkdir(parents=True)
        atomic_json(
            task_root / "identity.json",
            {"run_identity": {"execution_stage_binding": r._binding(stage)}},
        )
        bundle[operation + "_manifest"], bundle[operation + "_root"] = r._binding(target), str(root)
    assert (
        r._reference_prediction_set(config, bundle, prediction, pred_binding, fixture=True)
        == set_binding
    )
    replacement = tmp_path / "replacement.json"
    atomic_json(replacement, {**prediction, "later_refit": True})
    with pytest.raises(PermissionError, match="entire frozen design set"):
        r._reference_prediction_set(
            config, bundle, prediction, r._binding(replacement), fixture=True
        )
