"""Model-free role-gate regressions for real-VLM selection and empirical calibration."""

import sys
from types import SimpleNamespace

import pytest
from test_vlm_campaign import bound_json, stage_fixture

from src.modeling_v3 import schema, vlm_response
from src.modeling_v3 import vlm_campaign as c


@pytest.fixture
def gates(tmp_path, monkeypatch):
    config, base = stage_fixture(tmp_path, monkeypatch)
    cpu = bound_json(tmp_path / "cpu.json", {"kind": "CPU_SELECTION_FIXTURE"})
    dev_stage = bound_json(
        tmp_path / "dev_stage.json", {"role": "development", "selection_lock": cpu}
    )
    dev = bound_json(
        tmp_path / "development.json",
        {"status": "COMPLETE", "role": "development", "stage_binding": dev_stage},
    )
    lock = bound_json(
        tmp_path / "vlm_lock.json",
        {
            "kind": "V3_VLM_SELECTION_LOCK",
            "selected": {
                "designs": [
                    {
                        "design_id": design_id,
                        "method": "FULL_RIDGE",
                        "rank_cap": "FULL",
                        "alpha": alpha,
                        "output_policy": "RAW4",
                        "regression": "RIDGE",
                        "observation_method": "PILOT_SHRINK_ZERO_SUM",
                        "selector": "HASH",
                        "n_banks": 4,
                        "selection_seed": 17,
                    }
                    for design_id, alpha in (("design-a", 1e-5), ("design-b", 1e-4))
                ]
            },
            "locked_test_opened": False,
            "parent_cpu_selection": cpu,
            "development_completion": dev,
        },
    )
    cal_stage = bound_json(
        tmp_path / "cal_stage.json", {"role": "interval_calibration", "selection_lock": lock}
    )
    cal = bound_json(
        tmp_path / "calibration_complete.json",
        {"status": "COMPLETE", "role": "interval_calibration", "stage_binding": cal_stage},
    )
    receipt = bound_json(
        tmp_path / "calibration_report.json",
        {
            "kind": "V3_VLM_CALIBRATION",
            "status": "CALIBRATION_ANALYZED",
            "selection_lock": lock,
            "stage_completion": cal,
            "cross_seed_95": "NOT_CERTIFIED",
            "conformal_95": {"radius": None},
        },
    )
    calls = []

    def verify_selection(config, value, **kwargs):
        calls.append(("vlm_selection", value))
        actual = c._bound_json(value) if "path" in value else value
        if actual.get("kind") != "V3_VLM_SELECTION_LOCK":
            raise ValueError("Real VLM development selection required")
        c._bound_json(actual["parent_cpu_selection"])
        return actual

    def verify_calibration(config, selection, value, **kwargs):
        calls.append(("calibration", value))
        actual = c._bound_json(value)
        if actual.get("status") != "CALIBRATION_ANALYZED" or actual["selection_lock"] != selection:
            raise ValueError("Completed calibration on this selection required")
        return actual

    def verify_completion(config, value, role):
        calls.append(("completion", value))
        actual = c._bound_json(value)
        if actual.get("status") != "COMPLETE" or actual.get("role") != role:
            raise ValueError("Preceding stage incomplete")
        return actual

    monkeypatch.setattr(vlm_response, "verify_vlm_selection_lock", verify_selection, raising=False)
    monkeypatch.setattr(schema, "verify_selection_lock", lambda *a, **k: {})
    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.vlm_results",
        SimpleNamespace(verify_vlm_calibration=verify_calibration),
    )
    monkeypatch.setattr(c, "_verify_q5_completion", verify_completion)
    return SimpleNamespace(
        config=config,
        base=base,
        cpu=cpu,
        lock=lock,
        dev=dev,
        cal=cal,
        receipt=receipt,
        calls=calls,
        tmp=tmp_path,
    )


def stage(g, role, **changes):
    value = {
        **g.base,
        "role": role,
        "seeds": list(g.config["qwen"]["seed_roles"][role]),
        "selection_lock": g.lock,
        "development_completion": g.dev,
    }
    if role == "locked_test":
        value.update(interval_calibration_completion=g.cal, calibration_receipt=g.receipt)
    value.update(changes)
    return value


def check(g, value, name="stage.json"):
    binding = bound_json(g.tmp / name, value)
    return c._stage_gate(
        g.config, {"v3_stage_lock": binding}, value["seeds"][0], operation="train-source"
    )


def test_non_development_rejects_cpu_only_selection(gates):
    for role in ("interval_calibration", "locked_test"):
        with pytest.raises((PermissionError, ValueError), match="VLM"):
            check(gates, stage(gates, role, selection_lock=gates.cpu), role + ".json")


def test_interval_stage_uses_vlm_selection_and_its_exact_development_completion(gates):
    assert check(gates, stage(gates, "interval_calibration")) == "interval_calibration"
    assert ("vlm_selection", gates.lock) in gates.calls
    other = bound_json(gates.tmp / "other_dev.json", c._bound_json(gates.dev))
    with pytest.raises(ValueError, match="development"):
        check(gates, stage(gates, "interval_calibration", development_completion=other), "bad.json")


def test_locked_test_requires_analyzed_calibration_but_not_95_percent_certificate(gates):
    missing = stage(gates, "locked_test")
    missing.pop("calibration_receipt")
    with pytest.raises((PermissionError, ValueError), match="calibration"):
        check(gates, missing)
    assert check(gates, stage(gates, "locked_test"), "valid.json") == "locked_test"
    assert ("calibration", gates.receipt) in gates.calls


def test_locked_test_rejects_incomplete_calibration_stage(gates):
    partial = bound_json(
        gates.tmp / "partial.json", {**c._bound_json(gates.cal), "status": "INCOMPLETE"}
    )
    with pytest.raises(ValueError, match="incomplete"):
        check(gates, stage(gates, "locked_test", interval_calibration_completion=partial))


def test_locked_test_binds_calibration_report_to_same_stage_and_selection(gates):
    other = bound_json(gates.tmp / "other_cal.json", c._bound_json(gates.cal))
    with pytest.raises(ValueError, match="calibration"):
        check(gates, stage(gates, "locked_test", interval_calibration_completion=other))
    different_lock = bound_json(gates.tmp / "other_lock.json", c._bound_json(gates.lock))
    prior_stage = bound_json(
        gates.tmp / "different_cal_stage.json", {"selection_lock": different_lock}
    )
    prior = bound_json(
        gates.tmp / "different_cal.json", {**c._bound_json(gates.cal), "stage_binding": prior_stage}
    )
    with pytest.raises(ValueError, match="selection"):
        check(
            gates, stage(gates, "locked_test", interval_calibration_completion=prior), "other.json"
        )


def test_derived_stage_cannot_replace_frozen_selection_dependency(gates):
    parent = bound_json(gates.tmp / "parent.json", stage(gates, "interval_calibration"))
    other = bound_json(gates.tmp / "another_lock.json", c._bound_json(gates.lock))
    with pytest.raises(ValueError, match=r"parent|dependency|selection"):
        check(
            gates,
            stage(gates, "interval_calibration", selection_lock=other, authorization_parent=parent),
        )


def test_prepare_locked_test_refuses_missing_calibration_before_reading_parent(gates, monkeypatch):
    calls = []
    monkeypatch.setattr(c, "audit_v3_compatibility", lambda *a: calls.append("audit"))
    with pytest.raises(ValueError, match="calibration"):
        c.prepare_q5_stage(
            gates.config,
            {},
            q4_receipt={},
            role="locked_test",
            out=gates.tmp / "out",
            selection_lock=gates.lock,
            previous_completion=gates.cal,
        )
    assert not calls


def test_interval_finalization_does_not_require_its_own_calibration_analysis(gates):
    value = stage(gates, "interval_calibration")
    value["required_tasks"] = c._q5_expected_tasks(
        gates.config, "interval_calibration", selection_lock=gates.lock
    )
    value["expected_task_ids"] = sorted(value["required_tasks"])
    binding = bound_json(gates.tmp / "stage.json", value)
    result = c.finalize_q5_stage(
        gates.config, {"v3_stage_lock": binding}, task_receipts={}, out=gates.tmp / "partial"
    )
    assert result["status"] == "INCOMPLETE"
    assert not (gates.tmp / "partial" / "COMPLETE.json").exists()
    assert not any(name == "calibration" for name, _ in gates.calls)


def test_prepare_locked_stage_persists_exact_calibration_receipt(gates, monkeypatch):
    probes = [
        {"prompt_id": f"p{i}", "scene": {"base_scene_id": f"scene{i // 2}"}} for i in range(36)
    ]
    monkeypatch.setattr(c, "audit_v3_compatibility", lambda *args: {})
    monkeypatch.setattr(c, "build_probe_panel", lambda *args, **kwargs: probes)
    warm = bound_json(gates.tmp / "warm" / "bank_manifest.json", {"plan": {"control_prompts": []}})
    control = gates.tmp / "control.jsonl"
    control.write_text("")
    parent = bound_json(
        gates.tmp / "parent.json",
        {
            "paths": {"parent_warm": str(gates.tmp / "warm"), "raw_dataset_root": str(gates.tmp)},
            "warm_binding": {"files": {"bank_manifest.json": warm["sha256"]}},
            "training_data_binding": {"files": {"control.jsonl": c.file_hash(control)}},
        },
    )
    previous = bound_json(gates.tmp / "prior_stage.json", gates.base)
    result = c.prepare_q5_stage(
        gates.config,
        {"parent_validated_plan": parent, "v3_stage_lock": previous},
        q4_receipt=gates.base["v3_gpu_smoke"],
        role="locked_test",
        out=gates.tmp / "prepared",
        selection_lock=gates.lock,
        previous_completion=gates.cal,
        calibration_receipt=gates.receipt,
    )
    prepared = c._bound_json(result["stage_lock"])
    assert prepared["calibration_receipt"] == gates.receipt
    assert prepared["selection_lock"] == gates.lock
    assert prepared["interval_calibration_completion"] == gates.cal
    assert result["submitted"] is False


@pytest.mark.parametrize("kind", ["prediction", "evaluation"])
def test_non_development_task_completion_requires_stage_selected_fit(gates, kind):
    from src.modeling_v3.io import finalize_run

    value = stage(gates, "interval_calibration")
    stage_binding = bound_json(gates.tmp / "stage.json", value)
    expected = c._q5_expected_tasks(gates.config, "interval_calibration", selection_lock=gates.lock)
    origin_id = "42001_X_BASE_32"
    root = gates.tmp / "prediction"
    arrays = bound_json(root / "arrays.json", {"fixture": True})
    fit = bound_json(gates.tmp / "fit.json", {"selection_lock": gates.cpu})
    common = {
        "config_hash": value["config_hash"],
        "source_hash": value["source_hash"],
        "role": "interval_calibration",
        "origin_id": origin_id,
    }
    prediction = bound_json(
        root / "PREDICTION_LOCK.json",
        {**common, "kind": "V3_FROZEN_PREDICTIONS", "arrays": arrays, "fit_spec": fit},
    )
    finalize_run(root, common)
    binding = prediction
    if kind == "evaluation":
        binding = bound_json(
            gates.tmp / "evaluation" / "RESPONSE_EVALUATION_RECEIPT.json",
            {**common, "kind": "V3_RESPONSE_EVALUATION", "prediction_binding": prediction},
        )
        finalize_run(gates.tmp / "evaluation", common)
    with pytest.raises(ValueError, match="selection"):
        c._verify_q5_task(
            gates.config,
            stage_binding,
            next(
                task
                for task in expected.values()
                if task["kind"] == kind and task.get("origin_id") == origin_id
            ),
            binding,
        )


def prediction_member(g, design_id, *, name=None, fit_changes=None, **changes):
    from src.modeling_v3.io import finalize_run

    root = g.tmp / (name or design_id)
    settings = next(
        row for row in c._bound_json(g.lock)["selected"]["designs"] if row["design_id"] == design_id
    )
    geometry = bound_json(
        root / "GEOMETRY_SPEC.json",
        {
            "config_hash": c.canonical_hash(g.config),
            "source_hash": c.canonical_hash(c.source_hashes()),
            "origin_id": "42001_X_BASE_32",
            "role": "interval_calibration",
            "fixture": True,
            "method": settings["selector"],
            "n_banks": settings["n_banks"],
            "seed": settings["selection_seed"],
        },
    )
    fit = bound_json(
        root / "FIT_SPEC.json",
        {
            **settings,
            "geometry": geometry,
            "selection_lock": g.lock,
            "design_id": design_id,
            "config_hash": c.canonical_hash(g.config),
            "source_hash": c.canonical_hash(c.source_hashes()),
            "origin_id": "42001_X_BASE_32",
            "role": "interval_calibration",
            "fixture": True,
            "forks": {"path": "shared-forks", "sha256": "fixture"},
            "measurement_bundle": {"path": "shared-measurement", "sha256": "fixture"},
            **(fit_changes or {}),
        },
    )
    arrays = bound_json(root / "arrays.json", {"fixture": "tiny"})
    value = {
        "kind": "V3_FROZEN_PREDICTIONS",
        "fixture": True,
        "config_hash": c.canonical_hash(g.config),
        "source_hash": c.canonical_hash(c.source_hashes()),
        "origin_id": "42001_X_BASE_32",
        "role": "interval_calibration",
        "selection_lock": g.lock,
        "design_id": design_id,
        "fit_spec": fit,
        "arrays": arrays,
        "model_arrays": arrays,
        "query_units": [{"bank_id": "h0", "contrast_id": "joint"}],
        "probe_ids": ["p0"],
        "reference_labels_read": False,
        "heldout_labels_read": False,
        **changes,
    }
    binding = bound_json(root / "PREDICTION_LOCK.json", value)
    finalize_run(root, {"fixture": True})
    return binding


def test_matrix_covers_each_design_without_duplicating_measurements(gates):
    matrix = c._q5_expected_tasks(gates.config, "interval_calibration", selection_lock=gates.lock)
    origins = {x["origin_id"] for x in matrix.values() if x["kind"] == "forks"}
    for origin in origins:
        for kind in ("prediction", "evaluation"):
            assert {
                x["design_id"]
                for x in matrix.values()
                if x["kind"] == kind and x["origin_id"] == origin
            } == {"design-a", "design-b"}
        assert (
            sum(x["kind"] == "observation" and x["origin_id"] == origin for x in matrix.values())
            == 8
        )
    assert len(matrix) == 6 + 6 * (1 + 4 + 8)


def test_missing_second_design_cannot_complete_stage(gates, monkeypatch):
    value = stage(gates, "interval_calibration")
    value["required_tasks"] = c._q5_expected_tasks(
        gates.config, "interval_calibration", selection_lock=gates.lock
    )
    value["expected_task_ids"] = sorted(value["required_tasks"])
    binding = bound_json(gates.tmp / "stage.json", value)
    monkeypatch.setattr(
        c,
        "_verify_q5_task",
        lambda config, stage, task, receipt: {
            "kind": task["kind"],
            "reference_status": "MEASURED",
            "binding": receipt,
        },
    )
    supplied = {
        key: bound_json(gates.tmp / "tasks" / (key + ".json"), {"fixture": True})
        for key, task in value["required_tasks"].items()
        if task.get("design_id") != "design-b"
    }
    for key, task in value["required_tasks"].items():
        if task["kind"] == "evaluation" and key in supplied:
            supplied[key] = bound_json(
                gates.tmp / "evals" / (key + ".json"),
                {
                    "prediction_binding": supplied["prediction_" + key.removeprefix("evaluation_")],
                },
            )
    result = c.finalize_q5_stage(
        gates.config,
        {"v3_stage_lock": binding},
        task_receipts=supplied,
        out=gates.tmp / "incomplete-design",
    )
    assert result["status"] == "INCOMPLETE"
    assert len(result["missing_task_ids"]) == 12


def test_prediction_set_rejects_missing_design_and_accepts_complete_set(gates):
    a = prediction_member(gates, "design-a")
    b = prediction_member(gates, "design-b")
    with pytest.raises(ValueError, match="all selected designs"):
        c.freeze_q5_prediction_set(gates.config, [a], out=gates.tmp / "missing", fixture=True)
    result = c.freeze_q5_prediction_set(gates.config, [a, b], out=gates.tmp / "set", fixture=True)
    assert set(result["predictions"]) == {"design-a", "design-b"}
    assert (
        c.verify_q5_prediction_set(
            gates.config,
            result["prediction_set"],
            selection_lock=gates.lock,
            role="interval_calibration",
            fixture=True,
        )["predictions"]
        == result["predictions"]
    )
    with pytest.raises(ValueError, match="role"):
        c.verify_q5_prediction_set(
            gates.config, result["prediction_set"], role="locked_test", fixture=True
        )
    with pytest.raises(ValueError, match="source-bound"):
        c.verify_q5_prediction_set(gates.config, result["prediction_set"], fixture=False)


@pytest.mark.parametrize(
    "change",
    [
        {"heldout_labels_read": True},
        {"probe_ids": ["different"]},
        {"origin_id": "42001_X_BASE_96"},
    ],
)
def test_prediction_set_rejects_opened_labels_or_inconsistent_shared_scope(gates, change):
    a = prediction_member(gates, "design-a")
    b = prediction_member(gates, "design-b", **change)
    with pytest.raises(ValueError):
        c.freeze_q5_prediction_set(gates.config, [a, b], out=gates.tmp / "bad", fixture=True)


def test_prediction_set_rechecks_original_prediction_bytes(gates):
    a = prediction_member(gates, "design-a")
    b = prediction_member(gates, "design-b")
    result = c.freeze_q5_prediction_set(gates.config, [a, b], out=gates.tmp / "set", fixture=True)
    from pathlib import Path

    Path(c._bound_json(a)["arrays"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match=r"manifest|changed"):
        c.verify_q5_prediction_set(gates.config, result["prediction_set"], fixture=True)


def test_nondev_reference_execution_rejects_single_prediction(gates, monkeypatch):
    # The executable stage gate runs again before runtime/model loading.
    a = prediction_member(gates, "design-a")
    value = stage(
        gates,
        "interval_calibration",
        observation_purpose="reference",
        prediction_binding=a,
        origin_id="42001_X_BASE_32",
    )
    with pytest.raises(ValueError, match="prediction set"):
        check(gates, value)


@pytest.fixture
def q6(gates, monkeypatch):
    g = gates
    seed = g.config["qwen"]["seed_roles"]["locked_test"][0]
    identity = {
        "execution_kind": "REAL_CUDA_MODEL",
        "fixture": False,
        "config_hash": c.canonical_hash(g.config),
        "source_hash": c.canonical_hash(c.source_hashes()),
        "seed": seed,
        "arm": "X_BASE",
        "step": 64,
    }
    checkpoint = bound_json(g.tmp / "checkpoint.pt", {"tiny": True})
    checkpoint["identity"] = identity
    entries = [{"step": step} for step in range(1, 129)]
    entries[63].update(
        path=checkpoint["path"],
        checkpoint_sha256=checkpoint["sha256"],
        state_hash="actual64",
        checkpoint_identity=identity,
    )
    source = bound_json(
        g.tmp / "source" / "result.json",
        {
            "identity": identity,
            "steps": 128,
            "source_optimizer_steps": 128,
            "checkpoints": entries,
        },
    )
    completion = bound_json(
        g.tmp / "test_complete.json",
        {
            "status": "COMPLETE",
            "role": "locked_test",
            "verified_tasks": {f"source_{seed}_X_BASE": {"binding": source}},
        },
    )
    analysis = bound_json(g.tmp / "test_analysis.json", {"stage_completion": completion})
    qualification = bound_json(
        g.tmp / "qualification.json",
        {
            "pointwise_qualified": True,
            "qualified_design_ids": ["design-a"],
            "tracking_eligible_design_ids": ["design-a"],
            "selection_lock": g.lock,
            "test_analysis": analysis,
        },
    )
    results = sys.modules["src.modeling_v3.vlm_results"]
    monkeypatch.setattr(
        results,
        "verify_vlm_qualification",
        lambda config, binding, **kw: c._bound_json(binding),
        raising=False,
    )
    value = stage(
        g,
        "locked_test",
        phase="Q6",
        seeds=[seed],
        operations=["make-forks", "observe-vlm"],
        qualification_binding=qualification,
        source_binding=source,
        design_id="design-a",
        q6_anchor_checkpoint=checkpoint,
        origin_id=f"{seed}_X_BASE_64",
    )
    return SimpleNamespace(
        g=g, stage=value, checkpoint=checkpoint, source=source, qualification=qualification
    )


def test_q6_requires_qualified_design_and_retained_test_source(q6):
    binding = bound_json(q6.g.tmp / "q6.json", q6.stage)
    assert (
        c._stage_gate(
            q6.g.config, {"v3_stage_lock": binding}, q6.stage["seeds"][0], operation="make-forks"
        )
        == "locked_test"
    )
    with pytest.raises(PermissionError, match="outside"):
        c._stage_gate(
            q6.g.config, {"v3_stage_lock": binding}, q6.stage["seeds"][0], operation="train-source"
        )
    with pytest.raises(PermissionError, match="qualification"):
        c._verify_q6_dependencies(q6.g.config, {**q6.stage, "design_id": "design-b"})
    other = bound_json(q6.g.tmp / "unqualified_source.json", c._bound_json(q6.source))
    with pytest.raises(ValueError, match="qualified independent test"):
        c._verify_q6_dependencies(q6.g.config, {**q6.stage, "source_binding": other})


def test_q6_rejects_incomplete_source_and_wrong_anchor(q6):
    source = c._bound_json(q6.source)
    source["checkpoints"].pop()
    incomplete = bound_json(q6.g.tmp / "incomplete.json", source)
    with pytest.raises(ValueError, match="complete"):
        c._verify_q6_dependencies(q6.g.config, {**q6.stage, "source_binding": incomplete})
    wrong = {**q6.checkpoint, "identity": {**q6.checkpoint["identity"], "step": 96}}
    with pytest.raises(ValueError, match="step-64"):
        c._verify_q6_dependencies(q6.g.config, {**q6.stage, "q6_anchor_checkpoint": wrong})


def test_q5_does_not_accept_step64_forks(q6, monkeypatch):
    binding = bound_json(q6.g.tmp / "q5.json", stage(q6.g, "locked_test"))
    monkeypatch.setattr(
        c, "load_runtime", lambda *a, **k: pytest.fail("must reject before runtime load")
    )
    with pytest.raises(ValueError, match="Q5 response anchor"):
        c.make_forks(
            q6.g.config,
            {"v3_stage_lock": binding},
            checkpoint=q6.checkpoint,
            bank_plan={
                "origin_identity": {"seed": q6.stage["seeds"][0], "arm": "X_BASE", "step": 64}
            },
            device="cuda:0",
            out=q6.g.tmp / "forks",
            allow_gpu=True,
            allow_training=True,
            acknowledge_new_experiment=True,
        )


def test_q6_preparation_no_qualification_does_not_load_source(q6, monkeypatch):
    no = bound_json(
        q6.g.tmp / "no.json", {**c._bound_json(q6.qualification), "pointwise_qualified": False}
    )
    monkeypatch.setattr(
        c, "_verified_execution", lambda *a, **k: pytest.fail("must qualify before source read")
    )
    with pytest.raises(PermissionError, match="qualified"):
        c.prepare_q6_forks(
            q6.g.config,
            {},
            qualification_binding=no,
            design_id="design-a",
            source_root=q6.g.tmp / "source",
            out=q6.g.tmp / "prepare",
        )


def test_q6_preparation_records_new_cost_and_never_trains_source(q6, monkeypatch):
    from pathlib import Path

    from test_vlm_campaign import original_prompts

    g = q6.g
    source = c._bound_json(q6.source)
    template = bound_json(g.tmp / "original_bank_plan.json", {"train_prompts": original_prompts()})
    source["response_plans"] = [
        {"step": 32, "bank_plan": template["path"], "bank_plan_sha256": template["sha256"]}
    ]
    probes = bound_json(g.tmp / "probes.json", [{"prompt_id": str(i)} for i in range(36)])
    parent = bound_json(g.tmp / "parent_q5.json", stage(g, "locked_test", probe_panel=probes))
    monkeypatch.setattr(c, "_verified_execution", lambda *a, **k: source)
    real_resolve = c.resolve_checkpoint_binding
    monkeypatch.setattr(
        c,
        "resolve_checkpoint_binding",
        lambda spec: q6.checkpoint if isinstance(spec, str) else real_resolve(spec),
    )
    checked = []
    monkeypatch.setattr(c, "_verify_q6_dependencies", lambda config, value: checked.append(value))
    result = c.prepare_q6_forks(
        g.config,
        {"v3_stage_lock": parent},
        qualification_binding=q6.qualification,
        design_id="design-a",
        source_root=Path(q6.source["path"]).parent,
        out=g.tmp / "prepared",
    )
    assert checked and checked[0]["q6_anchor_checkpoint"] == q6.checkpoint
    assert result["workload"]["source_optimizer_steps"] == 0
    assert result["workload"]["candidate_adam_calls"] == 108
    assert result["workload"]["measurement_generation_sequences_total"] == 1363968
    assert result["workload"]["measurement_certified_rescore_sequences"] == 1363968
    assert result["workload"]["measurement_explicit_score_sequences_before_alias"] == 4018176
    assert "train-source" not in result["argv"] and result["submitted"] is False
    assert c._bound_json(result["stage_lock"])["operations"] == ["make-forks", "observe-vlm"]


def test_q6_refuses_pointwise_only_design_before_new_fork(q6, monkeypatch):
    no_tracking = bound_json(
        q6.g.tmp / "pointwise_only.json",
        {
            **c._bound_json(q6.qualification),
            "tracking_eligible_design_ids": [],
        },
    )
    monkeypatch.setattr(
        c, "_verified_execution", lambda *a, **k: pytest.fail("must reject before source read")
    )
    with pytest.raises(PermissionError, match="qualified"):
        c.prepare_q6_forks(
            q6.g.config,
            {},
            qualification_binding=no_tracking,
            design_id="design-a",
            source_root=q6.g.tmp / "source",
            out=q6.g.tmp / "prepared",
        )
    with pytest.raises(PermissionError, match="qualification"):
        c._verify_q6_dependencies(q6.g.config, {**q6.stage, "qualification_binding": no_tracking})


def test_prediction_set_does_not_mix_settings_between_designs(gates):
    a = prediction_member(gates, "design-a", fit_changes={"alpha": 1e-4})
    b = prediction_member(gates, "design-b")
    with pytest.raises(PermissionError, match="one complete frozen VLM design"):
        c.freeze_q5_prediction_set(gates.config, [a, b], out=gates.tmp / "mixed", fixture=True)


def test_stage_completion_rejects_evaluation_of_a_different_prediction(gates, monkeypatch):
    value = stage(gates, "interval_calibration")
    expected = c._q5_expected_tasks(gates.config, "interval_calibration", selection_lock=gates.lock)
    value.update(required_tasks=expected, expected_task_ids=sorted(expected))
    binding = bound_json(gates.tmp / "stage.json", value)
    prediction_id = next(key for key, task in expected.items() if task["kind"] == "prediction")
    evaluation_id = "evaluation_" + prediction_id.removeprefix("prediction_")
    prediction = bound_json(gates.tmp / "supplied_prediction.json", {"fixture": "first"})
    replacement = bound_json(gates.tmp / "later_prediction.json", {"fixture": "second"})
    evaluation = bound_json(gates.tmp / "evaluation.json", {"prediction_binding": replacement})
    monkeypatch.setattr(
        c,
        "_verify_q5_task",
        lambda config, stage, task, receipt: {
            "kind": task["kind"],
            "reference_status": "MEASURED",
            "binding": receipt,
        },
    )
    result = c.finalize_q5_stage(
        gates.config,
        {"v3_stage_lock": binding},
        task_receipts={prediction_id: prediction, evaluation_id: evaluation},
        out=gates.tmp / "bad_stage",
    )
    assert result["status"] == "TECHNICAL_FAILURE"
    assert "different predictions" in result["technical_failures"][evaluation_id]["reason"]
