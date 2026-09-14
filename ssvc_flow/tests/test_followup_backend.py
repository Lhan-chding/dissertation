"""Fail closed before weights, including mutable prepared-plan inputs."""

import copy
import json
from pathlib import Path

import pytest
from test_followup_inputs import bank_fixture as bank_fixture

from src import followup_backend as backend

DESIGN = Path(__file__).parents[1] / "docs/mechanism_followup/design/configs/followup_design.json"


def test_unresolved_paths_block_before_any_model_load(tmp_path):
    with pytest.raises(ValueError, match="Unresolved server paths"):
        backend.prepare_server_plan(json.loads(DESIGN.read_text()), {})


def test_gpu_opt_in_checked_before_plan_or_model_access(tmp_path):
    with pytest.raises(PermissionError, match="explicit GPU"):
        backend.load_server_context({}, allow_gpu_execution=False)
    with pytest.raises(PermissionError, match="explicit GPU"):
        backend.smoke_server({}, output_root=tmp_path, allow_gpu_execution=False)
    assert not list(tmp_path.iterdir())


def test_output_cannot_replace_or_contain_parent(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    for output in (parent, parent / "child", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            backend._separate_output(output, [parent])
    backend._separate_output(tmp_path / "new", [parent])


def test_snapshot_requires_complete_pinned_local_weights(tmp_path):
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ):
        (snapshot / name).write_text("{}")
    with pytest.raises(ValueError, match="weight"):
        backend._snapshot_binding(snapshot, revision)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "../escape.safetensors"}})
    )
    with pytest.raises(ValueError, match="escapes"):
        backend._snapshot_binding(snapshot, revision)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}})
    )
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"fixture weights")
    result = backend._snapshot_binding(snapshot, revision)
    assert result["revision"] == revision
    assert result["files"]["model-00001-of-00001.safetensors"]
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"changed")
    assert backend._snapshot_binding(snapshot, revision) != result


def test_revalidate_plan_rejects_source_or_parent_drift(monkeypatch):
    validated = {"design": {}, "paths": {}, "source_files": {"x": "old"}, "gate": {}}
    monkeypatch.setattr(
        backend,
        "prepare_server_plan",
        lambda *a, **k: {**copy.deepcopy(validated), "source_files": {"x": "new"}},
    )
    with pytest.raises(ValueError, match="changed"):
        backend._revalidate_plan(validated)


def test_source_bridge_rejects_modified_inherited_code(monkeypatch):
    parent = {"source": {"source_commit": "a" * 40, "source_files": {"src/frozen.py": "one"}}}
    monkeypatch.setattr(backend, "_source_files", lambda: {"src/frozen.py": "two"})
    with pytest.raises(ValueError, match="Inherited production source changed"):
        backend._source_bridge(parent)
    monkeypatch.setattr(
        backend, "_source_files", lambda: {"src/frozen.py": "one", "src/followup.py": "new"}
    )
    bridge = backend._source_bridge(parent)
    assert bridge["added_files"] == {"src/followup.py": "new"}
    assert bridge["gpu_joint_bridge_required"] is True
    assert bridge["gpu_joint_bridge_measured"] is False


def test_actual_adam_joint_bridge_restores_all_state():
    from test_followup_runtime import fixture_context

    from src.followup_updates import capture_state
    from src.optimizer_fork import state_hash

    plan, adapter, optimizer, origin = fixture_context()
    groups = next(u["groups"] for u in plan["units"] if u["bank_id"] == "bank03")
    result = backend._measure_joint_bridge(adapter, optimizer, origin, groups)
    assert set(result["checks"]) == {"0.0", "1.0"}
    assert all(c["comparison"]["passed"] for c in result["checks"].values())
    assert "gpu_smoke_passed" not in result
    assert state_hash(
        capture_state(adapter.model, optimizer, origin["metadata"], adapter=adapter)
    ) == state_hash(origin)


def test_joint_bridge_failure_restores_all_state(monkeypatch):
    from test_followup_runtime import fixture_context

    from src import followup_updates
    from src.followup_updates import capture_state
    from src.optimizer_fork import state_hash

    plan, adapter, optimizer, origin = fixture_context()
    groups = next(u["groups"] for u in plan["units"] if u["bank_id"] == "bank03")

    def fail(*args, **kwargs):
        adapter.model.scale.add_(1)
        raise RuntimeError("injected bridge failure")

    monkeypatch.setattr(followup_updates, "fork_one_candidate", fail)
    with pytest.raises(RuntimeError, match="injected bridge failure"):
        backend._measure_joint_bridge(adapter, optimizer, origin, groups)
    assert state_hash(
        capture_state(adapter.model, optimizer, origin["metadata"], adapter=adapter)
    ) == state_hash(origin)


def test_diagnostic_panel_uses_actual_r2_three_conditions():
    root = Path(__file__).parents[1] / "data/generated"
    prompts, binding = backend._diagnostic_prompts(root)
    assert len(prompts) == 216
    assert len({p["prompt_id"] for p in prompts}) == 216
    assert len({p["base_scene_id"] for p in prompts}) == 72
    assert {p["diagnostic_condition"] for p in prompts} == {
        "SYM_ORIGINAL",
        "IMAGE_CUE",
        "IMAGE_ONLY",
    }
    assert all(p["prompt_hash"] == p["prompt"]["prompt_hash"] for p in prompts)
    assert all(p["max_new_tokens"] == 64 for p in prompts)
    assert binding["panel_hash"]


def _backend_loader_seam(tmp_path, monkeypatch):
    """Explicit CPU wiring seam: actual parent JSON shape, no real gates/weights."""
    import torch
    from followup_test_adapter import make_fake_adapter

    from src import next_stage_runtime, r1_reference_smoke
    from src.followup_inputs import build_training_schedule
    from src.followup_parent import load_parent_metadata
    from src.optimizer_fork import capture_state, save_checkpoint, state_hash

    docs = load_parent_metadata(
        Path(__file__).parents[2] / "reports/SSVC_GPT_PRO_DELIVERY_20260914"
    )
    root = tmp_path / "warm"
    root.mkdir()
    for name, value in docs.items():
        if name in {"runtime_lock", "candidate_manifest", "bank_manifest"}:
            (root / (name + ".json")).write_text(json.dumps(value))
    data_root = Path(__file__).parents[1] / "data/generated"
    scenes = [json.loads(line) for line in (data_root / "train.jsonl").read_text().splitlines()]
    schedule = build_training_schedule(scenes, seed=17, data_root=str(data_root))
    adapter = make_fake_adapter()
    config = copy.deepcopy(docs["runtime_lock"]["config"])
    config["data_root"] = str(data_root)
    opt = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    initial = capture_state(adapter.model, opt, {"checkpoint_step": 0, "arm": "INITIAL"})
    initial_path = tmp_path / "initial.pt"
    save_checkpoint(initial_path, initial, {"unit": "CPU_INITIAL"})
    cold = tmp_path / "cold"
    cold.mkdir()
    (cold / "bank_manifest.json").write_text(json.dumps(docs["bank_manifest"]))
    (cold / "samples.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in _control_rows(docs["bank_manifest"]["plan"]["control_prompts"])
        )
    )
    validated = {
        "design": json.loads(DESIGN.read_text()),
        "design_hash": "fixture-design",
        "paths": {
            "parent_warm": str(root),
            "raw_dataset_root": str(data_root),
            "r3_cold_dir": str(cold),
        },
        "gate": {"certificate": {}},
        "r4_binding": {"environment": {"fixture": "CPU"}, "warm_checkpoint": {}},
        "snapshot": {},
        "config": config,
        "identity": {"model_hash": "fixture"},
        "source_files": {},
        "gates": {"parent_raw_verified": True, "gpu_smoke_passed": False},
        "training_plan": schedule,
        "legacy_lock": {},
        "diagnostic_prompts": [],
        "warning_policy": {"fixture": "CPU"},
        "initial_checkpoint": {
            "path": str(initial_path),
            "identity": {"unit": "CPU_INITIAL"},
            "file_sha256": backend.file_hash(initial_path),
            "state_hash": state_hash(initial),
        },
    }
    monkeypatch.setattr(backend, "_revalidate_plan", lambda plan: plan)
    monkeypatch.setattr(backend, "_load_local_adapter", lambda *a, **k: adapter)
    monkeypatch.setattr(
        next_stage_runtime, "validate_runtime_environment", lambda *a: {"fixture": "CPU"}
    )
    monkeypatch.setattr(r1_reference_smoke, "_certificate_check", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return validated, adapter


def test_s2_load_wires_real_seed17_schedule_and_extended_initial_state(tmp_path, monkeypatch):
    from src.followup_updates import capture_state
    from src.optimizer_fork import state_hash

    validated, adapter = _backend_loader_seam(tmp_path, monkeypatch)
    context = backend.load_server_context(
        validated, stage="S2", seed=17, arm="X_VALID_NO_X_OFF", allow_gpu_execution=True
    )
    assert set(context) == {"plan", "adapter", "optimizer", "initial_state", "data_root"}
    plan = context["plan"]
    assert plan["sampler_seed"] == 17
    assert plan["train_steps"] == validated["training_plan"]["train_steps"]
    assert len(plan["control_probe_rows"]) == 48
    assert len(plan["control_prompts"]) == 48
    assert plan["gates"]["gpu_smoke_passed"] is False
    initial = context["initial_state"]
    assert initial["metadata"]["checkpoint_step"] == 0
    assert not initial["optimizer"]["state"]
    assert "buffers" in initial
    assert state_hash(initial) == state_hash(
        capture_state(adapter.model, context["optimizer"], initial["metadata"], adapter=adapter)
    )


def test_s1_load_wires_actual_selected_bank_api(tmp_path, monkeypatch):
    from src import followup_inputs, r3_gate, r3_runtime, r3_warm_runtime
    from src.optimizer_fork import capture_state

    validated, _adapter = _backend_loader_seam(tmp_path, monkeypatch)
    monkeypatch.setattr(
        r3_warm_runtime,
        "load_warm_origin",
        lambda a, o, c: capture_state(a.model, o, {"checkpoint_step": 64}),
    )
    control = json.loads(
        (Path(validated["paths"]["parent_warm"]) / "bank_manifest.json").read_text()
    )["plan"]["control_prompts"]
    monkeypatch.setattr(
        r3_gate, "_ledger", lambda path: {str(i): r for i, r in enumerate(_control_rows(control))}
    )
    monkeypatch.setattr(
        r3_runtime, "_prepared", lambda a, p, d: {"audit": {"fixture_prompt": p["prompt_id"]}}
    )
    calls = []

    def assemble(manifest, rows, spec, *, candidate_manifest, runtime_lock, prepared_inputs=None):
        assert len(prepared_inputs) == 48
        assert len(rows) == 768
        calls.append(spec["id"])
        return {"bank_id": spec["id"], "groups": [], "bank_hash": "CPU"}

    monkeypatch.setattr(followup_inputs, "build_followup_bank", assemble)
    context = backend.load_server_context(
        validated, stage="S1", bank="bank03", allow_gpu_execution=True
    )
    assert calls == ["bank03"]
    assert set(context) == {"plan", "adapter", "optimizer", "origin", "data_root"}
    assert context["origin"]["metadata"]["checkpoint_step"] == 64
    assert context["plan"]["units"][0]["bank_id"] == "bank03"
    assert len(context["plan"]["control_prompts"]) == 48


def test_prepare_server_plan_positive_wiring_rechecks_parent_and_no_weights(
    bank_fixture, tmp_path, monkeypatch
):
    from src import next_stage_runtime, r3_gate, r3_warm_gate, r4_runtime
    from src.followup_inputs import build_training_schedule
    from src.r4_continuation import REVIEWED_CUMULATIVE_POLICY

    docs, rows = copy.deepcopy(bank_fixture)
    monkeypatch.setattr(r3_gate, "_ledger", lambda path: {row["sample_key"]: row for row in rows})
    design = json.loads(DESIGN.read_text())
    paths = {}
    for key in backend.DIRECTORY_KEYS:
        if key == "raw_dataset_root":
            path = (Path(__file__).parents[1] / "data/generated").resolve()
        elif key == "model_snapshot":
            path = tmp_path / "snapshots" / design["model"]["revision"]
            path.mkdir(parents=True)
        else:
            path = tmp_path / key
            path.mkdir()
        paths[key] = str(path)
    paths["new_run_root"] = str(tmp_path / "new_run")
    snapshot = Path(paths["model_snapshot"])
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ):
        (snapshot / name).write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"explicit CPU fixture, not model weights")
    warm = Path(paths["parent_warm"])
    for key in ("runtime_lock", "bank_manifest", "candidate_manifest"):
        (warm / (key + ".json")).write_text(json.dumps(docs[key]))
    (warm / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (warm / "origin.pt").write_bytes(b"CPU stub of absent real parent tensor")
    decision = Path(paths["parent_r4"]) / "continuation_decision.json"
    decision.write_text(json.dumps({"schema_version": 2, "policy": REVIEWED_CUMULATIVE_POLICY}))
    config = copy.deepcopy(docs["runtime_lock"]["config"])
    config["data_root"] = paths["raw_dataset_root"]
    gate = {"config": config, "binding": {"CPU_TEST_SEAM": True}}
    calls = []

    def prerequisites(r0, r1, supplement):
        assert (r0, r1, supplement) == tuple(
            paths[k] for k in ("r0_dir", "r1_run", "supplement_dir")
        )
        calls.append("prerequisites")
        return gate

    def r2(path, given_gate):
        assert given_gate is gate and path == paths["r2_dir"]
        calls.append("R2")
        return {"phase": "CPU_R2"}

    def cold(path, given_gate, r2_binding):
        assert given_gate is gate and r2_binding == {"phase": "CPU_R2"}
        calls.append("cold")
        return {"phase": "CPU_COLD"}

    def r4(path, given_gate, r2_binding, cold_binding):
        assert path == paths["parent_r4"] and cold_binding == {"phase": "CPU_COLD"}
        calls.append("R4")
        return {"phase": "CPU_R4"}

    def warm_gate(path, given_gate, r2_binding, cold_binding, r4_binding):
        assert path == warm and r4_binding == {"phase": "CPU_R4"}
        calls.append("warm")
        return {"phase": "CPU_WARM"}

    monkeypatch.setattr(next_stage_runtime, "validate_prerequisites", prerequisites)
    monkeypatch.setattr(next_stage_runtime, "validate_r2_gate", r2)
    monkeypatch.setattr(next_stage_runtime, "validate_r3_cold_gate", cold)
    monkeypatch.setattr(next_stage_runtime, "validate_r4_gate", r4)
    monkeypatch.setattr(r3_warm_gate, "validate_r3_warm_gate", warm_gate)
    scenes = [
        json.loads(line)
        for line in (Path(paths["raw_dataset_root"]) / "train.jsonl").read_text().splitlines()
    ]
    train = build_training_schedule(scenes, seed=17, data_root=paths["raw_dataset_root"])
    monkeypatch.setattr(
        r4_runtime, "_load_plan", lambda *args: (train, {"CPU_DATA": True}, {"CPU_LEGACY": True})
    )
    monkeypatch.setattr(backend, "_initial_evidence", lambda path: {"CPU_INITIAL": True})

    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation attempted to load model weights")

    monkeypatch.setattr(backend, "_load_local_adapter", forbidden)
    result = backend.prepare_server_plan(design, paths)
    assert calls == ["prerequisites", "R2", "cold", "R4", "warm"]
    assert len(result["parent_unit_bindings"]) == 6
    assert len(result["diagnostic_prompts"]) == 216
    assert result["warning_policy"] == REVIEWED_CUMULATIVE_POLICY
    assert (
        result["model_weights_loaded"]
        is result["gpu_started"]
        is result["training_started"]
        is False
    )
    assert all("groups" not in unit for unit in result["parent_unit_bindings"])
    assert backend._revalidate_plan(result) == result
    (snapshot / "model.safetensors").write_bytes(b"mutated fixture snapshot")
    with pytest.raises(ValueError, match="changed"):
        backend._revalidate_plan(result)


def _control_rows(prompts):
    return [
        {
            "bank_role": "control_proposal",
            "sample_index": i,
            "prompt_id": p["prompt_id"],
            "final_prompt_hash": "a" * 64,
            "tokenized_prompt_hash": "b" * 64,
            "input_tensor_hash": "c" * 64,
            "pixel_values_hash": None,
            "prepared_hash": "d" * 64,
        }
        for p in prompts
        for i in range(16)
    ]


def test_control_bindings_reject_disagreement_and_incomplete_rows():
    prompts = [{"prompt_id": "one"}]
    rows = _control_rows(prompts)
    assert (
        backend._control_prompt_bindings(prompts, rows)[0]["prepared_binding"]["prepared_hash"]
        == "d" * 64
    )
    with pytest.raises(ValueError, match="sixteen"):
        backend._control_prompt_bindings(prompts, rows[:-1])
    rows[-1]["input_tensor_hash"] = "e" * 64
    with pytest.raises(ValueError, match="disagree"):
        backend._control_prompt_bindings(prompts, rows)


def test_prospective_initialization_same_seed_exact_and_empty_adam():
    from types import SimpleNamespace

    import torch

    from src.optimizer_fork import state_hash

    class LoRAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.zeros(8, 16))
            self.lora_B = torch.nn.Parameter(torch.zeros(16, 8))
            self.base = torch.nn.Parameter(torch.ones(16), requires_grad=False)

    def state(seed):
        adapter = SimpleNamespace(model=LoRAModel())
        optimizer = torch.optim.AdamW(
            [adapter.model.lora_A, adapter.model.lora_B],
            lr=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        value = backend._seed_initial(adapter, optimizer, {}, seed)
        assert not value["optimizer"]["state"]
        assert not value["parameters"]["lora_B"].count_nonzero()
        assert value["parameters"]["lora_A"].count_nonzero()
        assert torch.equal(adapter.model.base, torch.ones(16))
        return value

    seed29 = state(29)
    assert state_hash(seed29) == state_hash(state(29))
    assert state_hash(seed29["parameters"]) != state_hash(state(41)["parameters"])


def test_historical_r2_requires_gpu_opt_in_and_original_arm(tmp_path):
    with pytest.raises(PermissionError, match="explicit GPU"):
        backend.evaluate_historical_r2({}, arm="X_BASE", output_root=tmp_path)
    with pytest.raises(ValueError, match="historical"):
        backend.evaluate_historical_r2(
            {}, arm="X_VALID_NO_X_OFF", output_root=tmp_path, allow_gpu_execution=True
        )


def _historical_r2_fixture():
    from followup_test_adapter import fake_prompts, make_fake_adapter, warm_origin

    adapter = make_fake_adapter()
    optimizer, origin = warm_origin(adapter)
    prompts = []
    for index, p in enumerate(fake_prompts(72, split="calibration")):
        p = {
            **p,
            "base_scene_id": f"historical-{index}",
            "scene": {**p["scene"], "base_scene_id": f"historical-{index}"},
        }
        for condition in ("SYM_ORIGINAL", "IMAGE_CUE", "IMAGE_ONLY"):
            prompts.append(
                {
                    **p,
                    "prompt_id": p["prompt_id"] + "/" + condition,
                    "max_new_tokens": 64,
                    "track": "R2",
                    "diagnostic_condition": condition,
                }
            )
    validated = {
        "design": json.loads(DESIGN.read_text()),
        "design_hash": "cpu-design",
        "identity": {
            "model_hash": "cpu",
            "data_hash": "cpu",
            "config_hash": "cpu",
            "parser_version_hash": "cpu",
            "protocol_version": "cpu",
        },
        "source_files": {},
        "diagnostic_prompts": prompts,
    }
    context = {
        "adapter": adapter,
        "optimizer": optimizer,
        "origin": origin,
        "data_root": ".",
        "endpoint": {"arm": "X_BASE", "step": 64, "state_hash": "cpu-endpoint"},
    }
    return context, validated


def test_historical_endpoint_r2_has_864_samples_zero_updates_exact_resume(tmp_path):
    from src.followup_updates import capture_state
    from src.optimizer_fork import state_hash

    context, validated = _historical_r2_fixture()
    adapter, optimizer, origin = context["adapter"], context["optimizer"], context["origin"]
    steps = []
    original_step = optimizer.step

    def forbidden_step(*args, **kwargs):
        steps.append(True)
        return original_step(*args, **kwargs)

    optimizer.step = forbidden_step
    args = dict(arm="X_BASE", output_root=tmp_path, fixture=True)
    result = backend._run_historical_r2(context, validated, **args)
    assert result["status"] == "CPU_TESTED"
    assert result["evaluation_outputs"] == 864 and result["optimizer_updates"] == 0
    assert result["training_started"] is False and not steps
    assert adapter.generation_calls == 864
    assert state_hash(
        capture_state(adapter.model, optimizer, origin["metadata"], adapter=adapter)
    ) == state_hash(origin)
    hashes = {str(p): backend.file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert backend._run_historical_r2(context, validated, resume=True, **args) == result
    assert adapter.generation_calls == 864 and not steps
    assert hashes == {str(p): backend.file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=r"exist|resume"):
        backend._run_historical_r2(context, validated, **args)
    changed = copy.deepcopy(validated)
    changed["identity"]["parser_version_hash"] = "other"
    with pytest.raises(ValueError, match=r"identity|differs"):
        backend._run_historical_r2(context, changed, resume=True, **args)


def test_historical_loader_verifies_original_checkpoint_before_restoring(tmp_path, monkeypatch):
    from followup_test_adapter import make_fake_adapter, warm_origin

    from src.optimizer_fork import capture_state, parameter_hash, save_checkpoint, state_hash

    adapter = make_fake_adapter()
    optimizer, _ = warm_origin(adapter)
    metadata = {
        "arm": "X_VALID",
        "checkpoint_step": 64,
        "sampler": {"position": 64},
        "completed_sample_keys": [str(i) for i in range(2048)],
    }
    legacy = capture_state(adapter.model, optimizer, metadata)
    runtime_identity = {"phase": "R4", "execution_kind": "CPU_CHECKPOINT_FIXTURE"}
    (tmp_path / "runtime_lock.json").write_text(json.dumps({"identity": runtime_identity}))
    path = tmp_path / "X_VALID" / "step64.pt"
    checkpoint_identity = {**runtime_identity, "arm": "X_VALID", "step": 64}
    save_checkpoint(path, legacy, checkpoint_identity)
    entry = {
        "arm": "X_VALID",
        "step": 64,
        "checkpoint_path": "X_VALID/step64.pt",
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_sha256": backend.file_hash(path),
        "state_hash": state_hash(legacy),
        "optimizer_state_hash": state_hash(legacy["optimizer"]),
        "parameter_hash": parameter_hash(adapter.model, trainable=True),
    }
    manifest = tmp_path / "checkpoint_manifest.json"
    manifest.write_text(json.dumps({"checkpoints": [entry]}))
    validated = {
        "paths": {"parent_r4": str(tmp_path)},
        "r4_binding": {
            "files": {
                "checkpoint_manifest.json": backend.file_hash(manifest),
                "X_VALID/step64.pt": backend.file_hash(path),
            }
        },
    }
    monkeypatch.setattr(
        backend,
        "load_server_context",
        lambda *a, **kw: {"adapter": adapter, "optimizer": optimizer, "data_root": "."},
    )
    result = backend._load_historical_context(validated, "X_VALID")
    assert result["endpoint"]["state_hash"] == state_hash(legacy)
    assert result["origin"]["metadata"] == metadata
    assert "buffers" in result["origin"]
    path.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="bytes changed"):
        backend._load_historical_context(validated, "X_VALID")


@pytest.mark.parametrize("failure_name", ["completion_binding.json", "completed.json"])
def test_historical_completion_publication_recovers_without_extra_samples(
    tmp_path, monkeypatch, failure_name
):
    from src import followup_train
    from src.followup_updates import capture_state
    from src.optimizer_fork import state_hash

    context, validated = _historical_r2_fixture()
    adapter, optimizer, origin = context["adapter"], context["optimizer"], context["origin"]
    original_publish = followup_train._publish

    def fail_publication(path, value):
        if Path(path).name == failure_name:
            raise OSError("injected completion publication interruption")
        return original_publish(path, value)

    def forbidden_step(*args, **kwargs):
        raise AssertionError("Historical evaluation attempted an optimizer step")

    monkeypatch.setattr(followup_train, "_publish", fail_publication)
    monkeypatch.setattr(optimizer, "step", forbidden_step)
    options = dict(arm="X_BASE", output_root=tmp_path, fixture=True)
    with pytest.raises(OSError, match="injected completion publication"):
        backend._run_historical_r2(context, validated, **options)
    assert adapter.generation_calls == 864
    rows_path = tmp_path / "samples" / "samples.jsonl"
    original_rows = rows_path.read_bytes()
    assert len(original_rows.splitlines()) == 864
    assert (tmp_path / "completion_manifest.json").is_file()
    assert not (tmp_path / "completed.json").exists()
    assert (tmp_path / "completion_binding.json").exists() == (failure_name == "completed.json")
    assert state_hash(
        capture_state(adapter.model, optimizer, origin["metadata"], adapter=adapter)
    ) == state_hash(origin)
    monkeypatch.setattr(followup_train, "_publish", original_publish)
    result = backend._run_historical_r2(context, validated, resume=True, **options)
    assert result["evaluation_outputs"] == 864 and result["optimizer_updates"] == 0
    assert adapter.generation_calls == 864
    assert rows_path.read_bytes() == original_rows
    binding = json.loads((tmp_path / "completion_binding.json").read_text())
    assert binding["result_hash"] == backend.canonical_hash(result)
    assert backend._run_historical_r2(context, validated, resume=True, **options) == result
    assert adapter.generation_calls == 864


def test_historical_fixture_rejects_non_cpu_tensors_before_any_output(tmp_path):
    import torch

    context, validated = _historical_r2_fixture()
    context["adapter"].model = torch.nn.Linear(2, 2, device="meta")
    with pytest.raises(ValueError, match="CPU"):
        backend._run_historical_r2(
            context, validated, arm="X_BASE", output_root=tmp_path, fixture=True
        )
    assert not list(tmp_path.iterdir())
