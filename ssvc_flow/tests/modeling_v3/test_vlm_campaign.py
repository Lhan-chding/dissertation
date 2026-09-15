"""Small local fixtures exercise the real source/fork Adam driver, never a VLM."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.modeling_v3 import vlm_campaign as campaign
from src.modeling_v3.io import canonical_hash


def config():
    return json.loads((Path(__file__).parents[2] / "configs/modeling_v3/protocol.json").read_text())


def original_prompts():
    source = (
        Path(__file__).parents[2]
        / "runs/modeling_v3_dev/server_inventory/parent_validated_plan.json"
    )
    if not source.exists():
        # The fixture is generated from original prompt construction contracts,
        # and does not depend on the private server inventory being installed.
        from src.prompts import build_prompt
        from src.r4_inputs import STRATA, _n_record

        rows = []
        for group in STRATA:
            for i in range(96):
                scene = {
                    "base_scene_id": f"{group[0]}-{group[1]}-{i}",
                    "split": "train",
                    "constraint_family": group[0],
                    "interface": group[1],
                    "chart_type": "line",
                    "operation": "sum4",
                    "truth_world": [1, 2, 3, 4],
                    "observed_world": [9, 2, 3, 4],
                    "changed_index": 0,
                    "cue": {"family": "duplicate_encoding", "known_index": 0, "known_value": 1},
                    "image_path": "test.png",
                }
                scene["prompt_hashes"] = {group[1]: build_prompt(scene, group[1])["prompt_hash"]}
                rows.append(_n_record(scene, group[1]))
        return rows
    return json.loads(source.read_text())["training_plan"]["train_prompts"]


def fake_runtime(steps=2):
    import importlib.util

    import torch

    from src.followup_updates import capture_state

    module_path = Path(__file__).parents[1] / "followup_test_adapter.py"
    spec = importlib.util.spec_from_file_location("v3_fixture_adapter", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.make_fake_adapter(91)
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    sampler = {"step": 0, "position": 0}
    state = capture_state(
        adapter.model,
        optimizer,
        {"checkpoint_step": 0, "seed": 41001, "arm": "X_BASE"},
        adapter=adapter,
        sampler=sampler,
    )
    runtime = {
        "adapter": adapter,
        "optimizer": optimizer,
        "sampler": sampler,
        "initial_state": state,
        "identity": {
            "execution_kind": "CPU_FAKE_TORCH",
            "model_hash": "fake",
            "data_hash": "fixture",
            "config_hash": "v3-test",
            "source_hash": "fixture-v1",
            "protocol_version": "V3_FIXTURE",
        },
        "data_root": None,
        "certificate_thresholds": {
            "parity_alarm_mean_abs_token_logp": 0.02,
            "parity_alarm_p99_abs_token_logp": 0.1,
        },
    }
    prompts = module.fake_prompts(steps * 4, split="train")
    for p in prompts:
        p["max_new_tokens"] = 64
    order = [[p["prompt_id"] for p in prompts[i : i + 4]] for i in range(0, steps * 4, 4)]
    schedule = {
        "train_prompts": prompts,
        "train_steps": order,
        "schedule_hash": canonical_hash(order),
        "positions": steps * 4,
    }
    return runtime, schedule


def test_no_model_planner_source_budget_and_two_workers(tmp_path):
    plan = campaign.plan_gpu(config(), tmp_path)
    counts = plan["workload"]["counts"]
    assert counts["source_optimizer_steps"] == 3072
    assert counts["source_training_outputs"] == 98304
    assert counts["response_origins"] == 30
    assert counts["scratch_optimizer_steps"] == 3240
    assert plan["model_loaded"] is False and plan["submitted"] is False
    assert {row["worker"] for row in plan["tasks"]} == {0, 1}
    assert campaign.plan_gpu(config(), tmp_path) == plan
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from src.modeling_v3.vlm_campaign import plan_gpu; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        check=True,
    )


def test_512_balanced_positions_and_unique_bank_partition():
    prompts = original_prompts()
    scenes = [p["scene"] for p in prompts]
    a = campaign.build_training_schedule(scenes, seed=41001)
    b = campaign.build_training_schedule(copy.deepcopy(scenes), seed=41001)
    assert a == b and len(a["train_steps"]) == 128
    used = [p for step in a["train_steps"] for p in step]
    assert len(set(used)) == 512
    groups = {(p["family"], p["interface"]) for p in a["train_prompts"][:6]}
    assert len(groups) == 6
    assert (
        a["schedule_hash"] != campaign.build_training_schedule(scenes, seed=41002)["schedule_hash"]
    )
    banks = campaign.build_bank_plan(a["train_prompts"], origin_identity={"state_hash": "origin"})
    assert len(banks["partitions"]["calibration_pool"]) == 384
    assert len(banks["partitions"]["heldout"]) == 192
    assert len(set([p for r in banks["banks"] for p in r["prompt_ids"]])) == 144
    assert len(banks["banks"]) == 36
    assert all(len(r["prompt_ids"]) == 4 and len(r["candidates"]) == 3 for r in banks["banks"])


def test_source_arms_share_hash_initialization_but_seeds_differ():
    from types import SimpleNamespace

    import torch

    def initialized(seed):
        model = torch.nn.Module()
        model.lora_A = torch.nn.Linear(3, 2, bias=False)
        model.lora_B = torch.nn.Linear(2, 3, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        return campaign.initialize_lora(SimpleNamespace(model=model), optimizer, seed)

    assert initialized(41001) == initialized(41001)
    assert initialized(41001)["parameter_hash"] != initialized(41002)["parameter_hash"]


def test_authorization_is_checked_before_model_import_or_bound_files(tmp_path):
    with pytest.raises(PermissionError, match="authorization"):
        campaign.load_runtime(config(), {}, allow_gpu=False)
    with pytest.raises(PermissionError, match="allow-training"):
        campaign.train_source(
            config(),
            {},
            seed=41001,
            arm="X_BASE",
            device="cuda:0",
            out=tmp_path,
            allow_gpu=True,
            acknowledge_new_experiment=True,
        )
    with pytest.raises(ValueError, match="cuda:0"):
        campaign.load_runtime(
            config(), {}, device="cuda:1", allow_gpu=True, acknowledge_new_experiment=True
        )


def test_actual_cpu_adam_source_restores_resume_and_tamper_detection(tmp_path):
    from src.followup_updates import load_checkpoint
    from src.optimizer_fork import state_hash

    runtime, schedule = fake_runtime(2)
    result = campaign.run_source_trajectory(
        config(), runtime, schedule, seed=41001, arm="X_BASE", out=tmp_path, fixture=True
    )
    assert result["steps"] == 2 and result["training_outputs"] == 64
    assert [r["optimizer_calls"] for r in result["checkpoints"]] == [1, 1]
    final = result["checkpoints"][-1]
    post = load_checkpoint(
        final["path"], final["checkpoint_identity"], expected_file_sha256=final["checkpoint_sha256"]
    )
    assert post["sampler"]["state"]["position"] == 8
    assert all(int(row["step"]) == 2 for row in post["optimizer"]["state"].values())
    assert state_hash(post["parameters"]) != state_hash(runtime["initial_state"]["parameters"])
    calls = runtime["adapter"].generation_calls
    resumed = campaign.run_source_trajectory(
        config(),
        runtime,
        schedule,
        seed=41001,
        arm="X_BASE",
        out=tmp_path,
        fixture=True,
        resume=True,
    )
    assert resumed == result and runtime["adapter"].generation_calls == calls
    Path(final["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash/size"):
        campaign.run_source_trajectory(
            config(),
            runtime,
            schedule,
            seed=41001,
            arm="X_BASE",
            out=tmp_path,
            fixture=True,
            resume=True,
        )


def test_source_failure_preserves_partial_and_resumes_exact_origin(tmp_path):
    from src.followup_updates import capture_state
    from src.optimizer_fork import state_hash

    runtime, schedule = fake_runtime(1)
    runtime["adapter"].fail_after = 4
    with pytest.raises(RuntimeError, match="interruption"):
        campaign.run_source_trajectory(
            config(), runtime, schedule, seed=41001, arm="X_BASE", out=tmp_path, fixture=True
        )
    actual = capture_state(
        runtime["adapter"].model,
        runtime["optimizer"],
        runtime["initial_state"]["metadata"],
        adapter=runtime["adapter"],
        sampler=runtime["sampler"],
    )
    assert state_hash(actual) == state_hash(runtime["initial_state"])
    assert list(tmp_path.rglob("samples.jsonl"))
    runtime["adapter"].fail_after = None
    result = campaign.run_source_trajectory(
        config(),
        runtime,
        schedule,
        seed=41001,
        arm="X_BASE",
        out=tmp_path,
        fixture=True,
        resume=True,
    )
    assert result["steps"] == 1
    assert runtime["adapter"].generation_calls == 32  # preserved four requests were not regenerated


def test_same_full_origin_forks_save_actual_e_and_leave_source_untouched(tmp_path):
    import torch

    from src.followup_updates import capture_state, load_checkpoint
    from src.optimizer_fork import state_hash

    runtime, schedule = fake_runtime(1)
    origin = runtime["initial_state"]
    plan = {
        "origin_identity": {"state_hash": state_hash(origin)},
        "train_prompts": schedule["train_prompts"],
        "banks": [
            {
                "bank_id": "calibration_pool_00",
                "role": "calibration_pool",
                "prompt_ids": schedule["train_steps"][0],
                "candidates": list(campaign.CANDIDATES),
            }
        ],
    }
    plan["plan_hash"] = canonical_hash(plan)
    result = campaign.run_candidate_banks(runtime, origin, plan, out=tmp_path, fixture=True)
    assert result["scratch_optimizer_steps"] == 3 and result["origin_restored"]
    actual = capture_state(
        runtime["adapter"].model,
        runtime["optimizer"],
        origin["metadata"],
        adapter=runtime["adapter"],
        sampler=runtime["sampler"],
    )
    assert state_hash(actual) == state_hash(origin)
    bank = result["banks"][0]
    storage = campaign._read(tmp_path / "storage_preflight.json")
    expected_vectors = (
        4 * 8 * sum(p.numel() for p in runtime["adapter"].model.parameters() if p.requires_grad)
    )
    assert storage["extra_retained_artifact_bytes"] == expected_vectors
    checkpoints = {
        k: load_checkpoint(v["path"], v["identity"], expected_file_sha256=v["sha256"])
        for k, v in bank["checkpoints"].items()
    }
    e = torch.load(bank["contrasts"]["joint_1_minus_joint_0"]["path"], weights_only=True)["payload"]
    assert all(
        torch.equal(
            e[name],
            checkpoints["joint_1"]["parameters"][name].double()
            - checkpoints["joint_0"]["parameters"][name].double(),
        )
        for name in e
    )
    assert bank["training_resume_alias_allowed"] is False


def test_measured_estimate_charges_multitoken_scoring_and_checkpoint_storage():
    measured = {
        "execution_kind": "REAL_CUDA_MODEL",
        "measured": True,
        "seconds_per_generated_sequence": 5,
        "seconds_per_scored_sequence": 2,
        "seconds_per_adam_update_excluding_generation": 100,
        "checkpoint_bytes": 1024,
    }
    estimate = campaign.estimate_workload(config(), measured)
    assert (
        estimate["estimated_gpu_seconds"]
        > estimate["counts"]["main_candidate_score_sequences_before_alias"] * 2
    )
    assert estimate["checkpoint_storage_bytes"] == (129 * 24 + 108 * 30) * 1024
    with pytest.raises(ValueError, match="actual"):
        campaign.estimate_workload(config(), {**measured, "execution_kind": "CPU_FAKE_TORCH"})


def bound_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return campaign._file_binding(path)


def stage_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"src/fixture.py": "v1"})
    cfg = config()
    hashes = {
        "config_hash": canonical_hash(cfg),
        "source_hash": canonical_hash(campaign.source_hashes()),
    }
    receipts = {
        phase: bound_json(
            tmp_path / (phase + ".json"), {**hashes, "status": "PASS", "checks": [{"passed": True}]}
        )
        for phase in ("Q1", "Q2")
    }
    smoke = bound_json(
        tmp_path / "smoke.json",
        {
            **hashes,
            "status": "PASS",
            "execution_kind": "REAL_CUDA_MODEL",
            "two_gpu_null_parity_passed": True,
        },
    )
    stage = {
        **hashes,
        "phase": "Q5",
        "role": "development",
        "seeds": [41001],
        "operations": ["train-source", "make-forks", "observe-vlm"],
        "technical_receipts": receipts,
        "v3_gpu_smoke": smoke,
    }
    return cfg, stage


def test_runtime_identity_has_no_observation_whitelist_hash_cycle(tmp_path, monkeypatch):
    cfg, stage = stage_fixture(tmp_path, monkeypatch)
    parent = {
        "snapshot": {"revision": "test"},
        "training_data_binding": {"files": {}},
        "r4_binding": {"environment": {"torch": "test"}},
    }
    bindings = {
        "parent_validated_plan": bound_json(tmp_path / "parent.json", parent),
        "v3_probability_tolerances": {"max_abs_token_logp": 1e-5},
        "v3_stage_lock": bound_json(tmp_path / "stage.json", stage),
    }
    before = campaign.planned_runtime_identity(cfg, bindings)
    derived = {
        **bindings,
        "v3_stage_lock": bound_json(
            tmp_path / "derived.json", {**stage, "observation_manifest_hashes": ["new manifest"]}
        ),
    }
    assert campaign.planned_runtime_identity(cfg, derived) == before
    changed = {**bindings, "v3_probability_tolerances": {"max_abs_token_logp": 0.01}}
    assert campaign.planned_runtime_identity(cfg, changed) != before


def test_q5_role_and_authorization_ancestry_are_independently_checked(tmp_path, monkeypatch):
    cfg, stage = stage_fixture(tmp_path, monkeypatch)
    parent_binding = bound_json(tmp_path / "parent.json", stage)
    child = {**stage, "authorization_parent": parent_binding}
    binding = bound_json(tmp_path / "child.json", child)
    assert (
        campaign._stage_gate(cfg, {"v3_stage_lock": binding}, 41001, operation="observe-vlm")
        == "development"
    )
    bad = bound_json(tmp_path / "bad.json", {**child, "role": "locked_test"})
    with pytest.raises(PermissionError, match="seed role"):
        campaign._stage_gate(cfg, {"v3_stage_lock": bad}, 41001, operation="observe-vlm")
    runtime = {"execution_stage_binding": binding, "identity": {"config_hash": canonical_hash(cfg)}}
    campaign._verify_runtime_stage(runtime)
    Path(parent_binding["path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        campaign._stage_gate(cfg, {"v3_stage_lock": binding}, 41001, operation="observe-vlm")
    with pytest.raises(ValueError, match="changed"):
        campaign._verify_runtime_stage(runtime)


def test_q5_task_plan_separates_pilot_main_reference_and_charges_direct(tmp_path):
    from src.modeling_v3.vlm_observation import freeze_task_manifest

    cfg = config()
    cfg["qwen"]["observation_n_primary"] = 8
    cfg["qwen"]["reference"]["initial_draws"] = 8
    prompt_binding = bound_json(tmp_path / "prompts.json", [{"prompt_id": "p"}])
    checkpoint = bound_json(tmp_path / "checkpoint.json", {})
    banks = [
        {"bank_id": "cal", "role": "calibration_pool"},
        {"bank_id": "held0", "role": "heldout"},
        {"bank_id": "held1", "role": "heldout"},
    ]
    names = ["origin"] + [b["bank_id"] + "/" + c["id"] for b in banks for c in campaign.CANDIDATES]
    policies = {name: {"checkpoint": checkpoint, "inference_fingerprint": name} for name in names}
    options = {
        "origin_id": "41001_X_BASE_32",
        "policies": policies,
        "banks": banks,
        "probe_binding": prompt_binding,
        "prompt_ids": ["p"],
    }
    measured = campaign._q5_generation_tasks(cfg, **options, purpose="measurement")
    reference = campaign._q5_generation_tasks(cfg, **options, purpose="reference")
    assert [(t["role"], t["draw_start"], t["draw_stop"]) for t in measured[:2]] == [
        ("pilot", 0, 2),
        ("main", 2, 8),
    ]
    assert sum(len(t["request_keys"]) for t in measured) == 8 + 6 * 8
    assert sum(len(t["request_keys"]) for t in reference) == 8 + 2 * 8
    assert {k for t in measured for k in t["request_keys"]}.isdisjoint(
        {k for t in reference for k in t["request_keys"]}
    )
    assert freeze_task_manifest(measured, policies, {"fixture": True})["tasks"] == measured
    counts = campaign.estimate_workload(config())["counts"]
    assert counts["reference_mix_spotcheck_draws_initial"] == 30 * 294912


def test_fresh_probe_panel_excludes_old_s1_and_refuses_insufficient_control():
    from src.prompts import build_prompt
    from src.r4_inputs import DEV_CELLS, INTERFACES

    representatives = {}
    for prompt in original_prompts():
        scene = prompt["scene"]
        representatives.setdefault(
            (scene["constraint_family"], scene["chart_type"], scene["operation"]), scene
        )
    # The generated fallback fixture has one chart/operation; expand only
    # metadata while preserving each family's valid original cue structure.
    by_family = {p["scene"]["constraint_family"]: p["scene"] for p in original_prompts()}
    scenes = []
    for cell in DEV_CELLS:
        for label in ("old", "fresh"):
            scene = copy.deepcopy(representatives.get(cell, by_family[cell[0]]))
            scene.update(
                base_scene_id="_".join((*cell, label)),
                split="control",
                interface=None,
                constraint_family=cell[0],
                chart_type=cell[1],
                operation=cell[2],
            )
            scene["prompt_hashes"] = {i: build_prompt(scene, i)["prompt_hash"] for i in INTERFACES}
            scenes.append(scene)
    excluded = [s["base_scene_id"] for s in scenes if s["base_scene_id"].endswith("_old")]
    probes = campaign.build_probe_panel(scenes, excluded_base_scene_ids=excluded)
    assert len(probes) == 36
    assert all(p["scene"]["base_scene_id"].endswith("_fresh") for p in probes)
    with pytest.raises(ValueError, match="CONTROL_POOL_INSUFFICIENT"):
        campaign.build_probe_panel(
            scenes, excluded_base_scene_ids=[s["base_scene_id"] for s in scenes]
        )


def test_q5_preparation_commands_parse_actual_cli_and_freeze_full_role(tmp_path, monkeypatch):
    from src.modeling_v3.cli import parser

    cfg, stage = stage_fixture(tmp_path, monkeypatch)
    probes = [
        {"prompt_id": f"p{i}", "scene": {"base_scene_id": f"fresh{i // 2}"}} for i in range(36)
    ]
    monkeypatch.setattr(campaign, "audit_v3_compatibility", lambda *args: {})
    monkeypatch.setattr(campaign, "build_probe_panel", lambda *args, **kwargs: probes)
    warm = bound_json(
        tmp_path / "warm" / "bank_manifest.json",
        {"plan": {"control_prompts": [{"scene": {"base_scene_id": "old"}}]}},
    )
    control = tmp_path / "control.jsonl"
    control.write_text("")
    parent = {
        "paths": {"parent_warm": str(tmp_path / "warm"), "raw_dataset_root": str(tmp_path)},
        "warm_binding": {"files": {"bank_manifest.json": warm["sha256"]}},
        "training_data_binding": {"files": {"control.jsonl": campaign.file_hash(control)}},
    }
    bindings = {
        "parent_validated_plan": bound_json(tmp_path / "parent.json", parent),
        "v3_stage_lock": bound_json(tmp_path / "stage.json", stage),
    }
    prepared = campaign.prepare_q5_stage(
        cfg,
        bindings,
        q4_receipt=stage["v3_gpu_smoke"],
        role="development",
        out=tmp_path / "prepared",
    )
    locked = campaign._bound_json(prepared["stage_lock"])
    assert len(locked["expected_task_ids"]) == 72
    assert len(prepared["source_commands"]) == 6
    for command in prepared["source_commands"]:
        assert command["argv"][1:3] == ["-m", "src.modeling_v3.cli"]
        parsed = parser().parse_args(command["argv"][3:])
        assert parsed.command == "train-source" and parsed.allow_training
    # Candidate command preparation only needs saved checkpoint bytes/metadata;
    # no model or optimizer is instantiated by this CPU path.
    checkpoint = tmp_path / "origin.pt"
    checkpoint.write_bytes(b"fixture checkpoint bytes")
    checkpoint_binding = {**campaign._file_binding(checkpoint), "identity": {"fixture": True}}
    bank = bound_json(tmp_path / "bank.json", {"fixture": True})
    source = {
        "identity": {"seed": 41001, "arm": "X_BASE"},
        "response_plans": [
            {
                "step": 32,
                "bank_plan": bank["path"],
                "bank_plan_sha256": bank["sha256"],
                "checkpoint": checkpoint_binding,
                "checkpoint_sha256": checkpoint_binding["sha256"],
            }
        ],
    }
    source_root = tmp_path / "source"
    bound_json(source_root / "result.json", source)
    monkeypatch.setattr(campaign, "_verified_execution", lambda *args, **kwargs: source)
    fork = campaign.prepare_q5_forks(
        cfg,
        campaign._bound_json(prepared["source_bindings"]),
        source_root=source_root,
        out=tmp_path / "forkplan",
    )
    parsed = parser().parse_args(fork["fork_commands"][0]["argv"][3:])
    assert parsed.command == "make-forks" and parsed.allow_training


def test_q5_completion_rejects_manual_pass_and_keeps_partial_matrix_incomplete(
    tmp_path, monkeypatch
):
    cfg, stage = stage_fixture(tmp_path, monkeypatch)
    stage["required_tasks"] = campaign._q5_expected_tasks(cfg, "development")
    stage["expected_task_ids"] = sorted(stage["required_tasks"])
    stage_binding = bound_json(tmp_path / "stage.json", stage)
    bindings = {"v3_stage_lock": stage_binding}
    partial = campaign.finalize_q5_stage(cfg, bindings, task_receipts={}, out=tmp_path / "partial")
    assert partial["status"] == "INCOMPLETE" and len(partial["missing_task_ids"]) == 72
    assert not (tmp_path / "partial" / "COMPLETE.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        campaign._verify_q5_completion(cfg, partial["receipt"], "development")
    pretend = bound_json(tmp_path / "pretend" / "result.json", {"status": "PASS"})
    failed = campaign.finalize_q5_stage(
        cfg, bindings, task_receipts={"source_41001_X_BASE": pretend}, out=tmp_path / "failed"
    )
    assert failed["status"] == "TECHNICAL_FAILURE"
    assert failed["scientific_status"] == "NOT_CERTIFIED"
    assert "source_41001_X_BASE" in failed["technical_failures"]
    with pytest.raises(ValueError, match="only frozen"):
        campaign.finalize_q5_stage(
            cfg, bindings, task_receipts={"unexpected": pretend}, out=tmp_path / "unexpected"
        )


def test_q5_prediction_and_unresolved_evaluation_complete_only_as_engineering(
    tmp_path, monkeypatch
):
    from src.modeling_v3.io import finalize_run

    cfg, stage = stage_fixture(tmp_path, monkeypatch)
    stage_binding = bound_json(tmp_path / "stage.json", stage)
    root = tmp_path / "prediction"
    arrays = root / "arrays.npz"
    root.mkdir()
    arrays.write_bytes(b"small fixture arrays")
    common = {
        "config_hash": canonical_hash(cfg),
        "source_hash": stage["source_hash"],
        "role": "development",
        "origin_id": "41001_X_BASE_32",
    }
    prediction = bound_json(
        root / "PREDICTION_LOCK.json",
        {**common, "kind": "V3_FROZEN_PREDICTIONS", "arrays": campaign._file_binding(arrays)},
    )
    finalize_run(root, common)
    expected = campaign._q5_expected_tasks(cfg, "development")
    result = campaign._verify_q5_task(
        cfg, stage_binding, expected["prediction_41001_X_BASE_32"], prediction
    )
    assert result["scientific_status"] == "NOT_CERTIFIED"
    evaluation = bound_json(
        tmp_path / "eval" / "RESPONSE_EVALUATION_RECEIPT.json",
        {
            **common,
            "kind": "V3_RESPONSE_EVALUATION",
            "prediction_binding": prediction,
            "reference_status": "REFERENCE_UNRESOLVED",
        },
    )
    finalize_run(tmp_path / "eval", common)
    result = campaign._verify_q5_task(
        cfg, stage_binding, expected["evaluation_41001_X_BASE_32"], evaluation
    )
    assert result["reference_status"] == "REFERENCE_UNRESOLVED"
    arrays.write_bytes(b"changed original arrays")
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign._verify_q5_task(
            cfg, stage_binding, expected["prediction_41001_X_BASE_32"], prediction
        )


def test_q5_observation_preparation_binds_completed_generation_and_cli(tmp_path, monkeypatch):
    from src.modeling_v3 import vlm_observation
    from src.modeling_v3.cli import parser

    cfg, stage = stage_fixture(tmp_path, monkeypatch)
    cfg["qwen"]["observation_n_primary"] = 4  # Bounded task-planning fixture only.
    monkeypatch.setattr(campaign, "_stage_gate", lambda *args, **kwargs: "development")
    stage["config_hash"] = canonical_hash(cfg)
    stage["probe_panel"] = bound_json(
        tmp_path / "probes.json", [{"prompt_id": f"p{i}"} for i in range(36)]
    )
    parent = {"snapshot": {}, "training_data_binding": {}, "r4_binding": {"environment": {}}}
    bindings = {
        "v3_stage_lock": bound_json(tmp_path / "stage.json", stage),
        "parent_validated_plan": bound_json(tmp_path / "parent.json", parent),
        "v3_probability_tolerances": {"max_abs_token_logp": 1e-5},
    }
    origin = {"seed": 41001, "arm": "X_BASE", "step": 32, "state_hash": "origin-state"}
    checkpoint = tmp_path / "origin" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture checkpoint")
    checkpoint_spec = {
        **campaign._file_binding(checkpoint),
        "identity": {
            **origin,
            "source_hash": stage["source_hash"],
            "config_hash": canonical_hash(cfg),
        },
    }
    bound_json(
        checkpoint.parent / "result.json",
        {"state_hash": origin["state_hash"], "inference_fingerprint": "origin-fp"},
    )
    banks = [
        {
            "bank_id": f"bank{i}",
            "role": "calibration_pool" if i < 24 else "heldout",
            "checkpoints": {
                c["id"]: {**checkpoint_spec, "inference_fingerprint": f"fp{i}-{c['id']}"}
                for c in campaign.CANDIDATES
            },
        }
        for i in range(36)
    ]
    forks = {"identity": {"Q4_bridge": False}, "banks": banks}
    forks_root = tmp_path / "forks"
    bound_json(forks_root / "result.json", forks)
    bound_json(forks_root / "bank_plan.json", {"origin_identity": origin})
    monkeypatch.setattr(campaign, "_verified_execution", lambda *args, **kwargs: forks)
    options = {
        "forks_root": forks_root,
        "origin_checkpoint": checkpoint_spec,
        "include_direct_count": False,
    }
    generated = campaign.prepare_q5_observation(
        cfg, bindings, **options, out=tmp_path / "generation"
    )
    assert generated["counts"] == {"generate": 36 * 4, "score": 0}
    for row in generated["commands"]:
        assert row["argv"][1:3] == ["-m", "src.modeling_v3.cli"]
        assert parser().parse_args(row["argv"][3:]).command == "observe-vlm"
    with pytest.raises(PermissionError, match="Freeze prediction"):
        campaign.prepare_q5_observation(
            cfg, bindings, **options, purpose="reference", out=tmp_path / "reference"
        )
    with pytest.raises(ValueError, match="completed generation"):
        campaign.prepare_q5_observation(
            cfg, bindings, **options, operation="score", out=tmp_path / "scores"
        )
    # Supply tiny immutable shard paths after the completion-reader boundary;
    # its full record and manifest verification has separate tests.
    manifest = campaign._bound_json(generated["manifest"])
    generation_root = tmp_path / "generation" / "observations"
    for task in manifest["tasks"]:
        worker = vlm_observation._worker_assignment(task, manifest["policies"], 2)
        root = generation_root / f"worker_{worker}" / task["output_path"]
        root.mkdir(parents=True)
        shard = root / "samples.jsonl"
        shard.write_text("{}\n")
        bound_json(root / "COMPLETE.json", {"shards": [{"path": "samples.jsonl"}]})
    checked = []
    monkeypatch.setattr(vlm_observation, "_completed_task_rows", lambda *args: checked.append(args))
    scores = campaign.prepare_q5_observation(
        cfg,
        bindings,
        **options,
        operation="score",
        generation_manifest=generated["manifest"],
        generation_root=generation_root,
        out=tmp_path / "scores",
    )
    assert len(checked) == 1
    assert scores["counts"] == {"generate": 0, "score": 109 * 36 * 4}
    for row in scores["commands"]:
        assert parser().parse_args(row["argv"][3:]).command == "observe-vlm"


def reference_history(end=4096):
    tasks, start = [], 0
    for stop in (4096, 8192, 16384, 32768, 65536):
        tasks.append(
            {
                "operation": "generate",
                "role": "reference",
                "proposal": "ORIGIN",
                "proposal_candidates": ["origin"],
                "origin_id": "41001_X_BASE_32",
                "prompt_ids": ["p0", "p1"],
                "draw_start": start,
                "draw_stop": stop,
                "rng_namespace": "fixed-reference-stream",
            }
        )
        if stop == end:
            break
        start = stop
    return tasks


def reference_units(n=4096, *, status="REFERENCE_EXTEND"):
    return [
        {
            "bank_id": "held0",
            "contrast_id": "joint_1_minus_joint_0",
            "prompt_id": p,
            "origin": {"n": n, "status": status, "empirical_precision_met": False},
            "mix": None,
            "crosscheck_consistent": True,
        }
        for p in ("p0", "p1")
    ]


def test_reference_extensions_follow_registered_looks_and_preserve_independent_stream():
    for current, stop in ((4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536)):
        specs = campaign._reference_extension_specs(
            config(),
            reference_history(current),
            reference_units(current),
            origin_id="41001_X_BASE_32",
        )
        assert specs == [
            {
                "proposal": "ORIGIN",
                "proposal_candidates": ["origin"],
                "candidate_id": "origin",
                "prompt_ids": ["p0", "p1"],
                "draw_start": current,
                "draw_stop": stop,
                "rng_namespace": "fixed-reference-stream",
            }
        ]
    assert (
        campaign._reference_extension_specs(
            config(),
            reference_history(65536),
            reference_units(65536, status="REFERENCE_UNRESOLVED"),
            origin_id="41001_X_BASE_32",
        )
        == []
    )
    broken = reference_history(8192)
    broken[-1]["draw_start"] = 0
    with pytest.raises(ValueError, match="repeated draws"):
        campaign._reference_extension_specs(
            config(), broken, reference_units(8192), origin_id="41001_X_BASE_32"
        )
    with pytest.raises(ValueError, match="ORIGIN stream"):
        campaign._reference_extension_specs(
            config(), reference_history(), reference_units(8192), origin_id="41001_X_BASE_32"
        )


def test_mandatory_mixture_targets_only_failed_units_and_never_redraws_prior_packet():
    reports = reference_units(status="REFERENCE_EMPIRICAL_PRECISION_MET")
    reports[0]["origin"].update(status="REFERENCE_REQUIRES_INDEPENDENT_MIX")
    reports[0]["crosscheck_consistent"] = False
    history = reference_history()
    specs = campaign._reference_extension_specs(
        config(), history, reports, origin_id="41001_X_BASE_32"
    )
    assert len(specs) == 1 and specs[0]["proposal"] == "MIX"
    assert specs[0]["prompt_ids"] == ["p0"]
    assert (specs[0]["draw_start"], specs[0]["draw_stop"]) == (0, 4096)
    prior = {**history[0], **specs[0]}
    history.append(prior)
    reports[0]["mix"] = {"n": 4096, "status": "REFERENCE_EXTEND", "empirical_precision_met": False}
    extension = campaign._reference_extension_specs(
        config(), history, reports, origin_id="41001_X_BASE_32"
    )
    assert (extension[0]["draw_start"], extension[0]["draw_stop"]) == (4096, 8192)
    assert extension[0]["rng_namespace"] == prior["rng_namespace"]
    reports[0]["mix"]["empirical_precision_met"] = True
    assert (
        campaign._reference_extension_specs(config(), history, reports, origin_id="41001_X_BASE_32")
        == []
    )


def test_reference_continuation_rejects_rank_based_or_unfinished_precision_receipt(tmp_path):
    from src.modeling_v3.io import finalize_run

    binding = bound_json(
        tmp_path / "receipt" / "REFERENCE_PRECISION_RECEIPT.json",
        {
            "kind": "V3_REFERENCE_PRECISION",
            "precision_only": False,
            "predictor_rankings_used": True,
        },
    )
    with pytest.raises(ValueError, match="completed evaluator"):
        campaign.prepare_q5_reference_extension(
            config(), {}, precision_receipt=binding, out=tmp_path / "out"
        )

    finalize_run(tmp_path / "receipt", {})
    with pytest.raises(ValueError, match="reference-only"):
        campaign.prepare_q5_reference_extension(
            config(), {}, precision_receipt=binding, out=tmp_path / "out"
        )


def test_reference_extension_cpu_entrypoint_emits_actual_next_range_and_bound_stage(
    tmp_path, monkeypatch
):
    from src.modeling_v3 import vlm_observation
    from src.modeling_v3.cli import parser
    from src.modeling_v3.io import finalize_run

    cfg, stage = stage_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "_stage_gate", lambda *args, **kwargs: "development")
    checkpoint = bound_json(tmp_path / "checkpoint.json", {})
    policies = {
        name: {"checkpoint": {**checkpoint, "identity": {}}, "inference_fingerprint": name}
        for name in ("origin", "held0/joint_0", "held0/joint_1", "held0/no_x_off_1")
    }
    runtime = {"config_hash": canonical_hash(cfg), "source_hash": stage["source_hash"]}
    monkeypatch.setattr(campaign, "planned_runtime_identity", lambda *args: runtime)
    stage["forks_result"] = bound_json(tmp_path / "forks.json", {})
    bindings = {"v3_stage_lock": bound_json(tmp_path / "stage.json", stage)}
    probe = bound_json(tmp_path / "probes.json", [{"prompt_id": "p0"}])
    task = {**reference_history()[0], "prompt_ids": ["p0"], "prompt_file": probe}
    gen = bound_json(
        tmp_path / "prior_gen.json",
        {"tasks": [task], "policies": policies, "runtime_identity": runtime},
    )
    score = bound_json(
        tmp_path / "prior_score.json",
        {
            "tasks": [{**task, "operation": "score"}],
            "policies": policies,
            "runtime_identity": runtime,
        },
    )
    read_history = []
    monkeypatch.setattr(
        vlm_observation, "_completed_task_rows", lambda *args: read_history.append(args)
    )
    origin_id = "41001_X_BASE_32"
    pred_root = tmp_path / "prediction"
    arrays = bound_json(pred_root / "arrays.json", {})
    pred = bound_json(
        pred_root / "PREDICTION_LOCK.json",
        {
            **runtime,
            "kind": "V3_FROZEN_PREDICTIONS",
            "origin_id": origin_id,
            "arrays": arrays,
            "query_units": [{"bank_id": "held0", "contrast_id": "joint_1_minus_joint_0"}],
            "probe_ids": ["p0"],
        },
    )
    finalize_run(pred_root, runtime)
    precision_root = tmp_path / "precision"
    decision = bound_json(
        precision_root / "REFERENCE_PRECISION_DIAGNOSTICS.json",
        {
            "unit_reports": reference_units()[:1],
            "precision_only": True,
            "predictor_rankings_used": False,
        },
    )
    precision = bound_json(
        precision_root / "REFERENCE_PRECISION_RECEIPT.json",
        {
            **runtime,
            "kind": "V3_REFERENCE_PRECISION",
            "origin_id": origin_id,
            "prediction_binding": pred,
            "decision_evidence": decision,
            "precision_only": True,
            "predictor_rankings_used": False,
            "current_draws": 4096,
            "next_draws": 8192,
            "reference_batches": [
                {
                    "generation_manifest": gen,
                    "scoring_manifest": score,
                    "generation_root": str(tmp_path / "prior_gen"),
                    "scoring_root": str(tmp_path / "prior_score"),
                }
            ],
        },
    )
    finalize_run(precision_root, runtime)
    prepared = campaign.prepare_q5_reference_extension(
        cfg, bindings, precision_receipt=precision, out=tmp_path / "next"
    )
    assert prepared["counts"] == {"generate": 4096, "score": 0}
    assert len(read_history) == 2
    manifest = campaign._bound_json(prepared["manifest"])
    assert len(manifest["tasks"]) == 1
    next_task = manifest["tasks"][0]
    assert (next_task["draw_start"], next_task["draw_stop"]) == (4096, 8192)
    assert next_task["rng_namespace"] == task["rng_namespace"]
    resolved = campaign._bound_json(prepared["source_bindings"])
    child = campaign._bound_json(resolved["v3_stage_lock"])
    assert child["reference_precision_receipt"] == precision
    assert child["no_method_rankings_used"] is True
    for command in prepared["commands"]:
        assert parser().parse_args(command["argv"][3:]).command == "observe-vlm"
