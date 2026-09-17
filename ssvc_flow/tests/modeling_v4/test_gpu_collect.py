import importlib.util
from pathlib import Path

import pytest
import torch

from src.modeling_v4 import gpu_collect as gpu


def tiny_runtime():
    path = Path(__file__).parents[1] / "followup_test_adapter.py"
    spec = importlib.util.spec_from_file_location("v4_fake_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.make_fake_adapter(17)
    adapter.pad_id = 4
    adapter.audit["probability_execution"] = "uncached_prefix_recompute"
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad], lr=1e-3
    )
    runtime = gpu.attach_runtime(
        adapter,
        optimizer,
        identity={
            "execution_kind": "CPU_FAKE_TORCH",
            "model_hash": "tiny",
            "config_hash": "tiny",
            "source_hash": "tiny",
        },
        parity_tolerances={
            "mean_abs_token_logp": 1e-5,
            "max_abs_token_logp": 1e-5,
            "max_abs_sequence_logp": 1e-4,
        },
        allocated_gpu={"uuid": "fixture-only"},
    )
    return runtime, module


def test_compact_complete_restore_adam_rng_buffers_modes_and_frozen_guard(monkeypatch):
    from src.optimizer_fork import state_hash

    runtime, _ = tiny_runtime()
    s = runtime["state"]
    model = runtime["adapter"].model
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    runtime["optimizer"].step()
    runtime["optimizer"].zero_grad()
    origin = s.capture({"step": 1})
    digest = state_hash(origin)

    def forbidden(*args, **kwargs):
        raise AssertionError("Whole frozen model hashing is forbidden")

    monkeypatch.setattr("src.followup_updates.parameter_hash", forbidden)
    with torch.no_grad():
        model.logits.add_(10)
        model.scale.add_(1)
    model.eval()
    runtime["sampler"]["step"] = 99
    torch.rand(5)
    s.restore(origin)
    assert state_hash(s.capture({"step": 1})) == digest
    with torch.no_grad():
        model.frozen.add_(1)
    with pytest.raises(RuntimeError, match="Frozen"):
        s.capture()


def test_checkpoint_import_hash_once_and_reject_changed_file(tmp_path):
    runtime, _ = tiny_runtime()
    state = runtime["state"].capture()
    binding = gpu._save_state(tmp_path / "state.pt", state, {"version": "tiny"})
    cache = gpu.CheckpointCache(max_resident=0)
    cache.load(binding)
    cache.load(binding)
    assert cache.hash_checks == 1
    with open(binding["path"], "ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cache.load(binding)


def test_shared_chunks_resume_without_regenerating_completed_samples(tmp_path):
    calls = []

    def produce(i):
        calls.append(i)
        if i == 3:
            raise RuntimeError("injected")
        return {"sample_id": str(i), "token_ids": [i, 4]}

    with pytest.raises(RuntimeError):
        gpu._chunks(tmp_path / "s", {"role": "work"}, produce, count=5, resume=False, chunk_size=2)
    assert calls == [0, 1, 2, 3]
    calls.clear()
    result = gpu._chunks(
        tmp_path / "s",
        {"role": "work"},
        lambda i: calls.append(i) or {"sample_id": str(i), "token_ids": [i, 4]},
        count=5,
        resume=True,
        chunk_size=2,
    )
    assert calls == [2, 3, 4]
    assert [r["sample_id"] for r in gpu._rows(result)] == list(map(str, range(5)))
    with pytest.raises(ValueError):
        gpu._chunks(tmp_path / "s", {"role": "reference"}, produce, count=5, resume=True)


def test_actual_tiny_adam_forks_restore_and_real_deltas(tmp_path):
    from src.optimizer_fork import state_hash

    runtime, module = tiny_runtime()
    state = runtime["state"].capture({"seed": 41001, "arm": "X_BASE", "checkpoint_step": 32})
    before = state_hash(state)
    prompts = module.fake_prompts(4, split="train")
    plan = {
        "origin_id": "tiny_origin",
        "banks": [
            {
                "bank_id": "calibration_000",
                "role": "calibration",
                "prompt_ids": [p["prompt_id"] for p in prompts],
                "candidates": list(gpu.CANDIDATES),
            }
        ],
    }
    result = gpu.run_forks(runtime, state, plan, prompts, out=tmp_path / "forks", bridge=True)
    assert state_hash(runtime["state"].capture(state["metadata"])) == before
    bank = result["banks"][0]
    assert set(bank["policies"]) == {"joint_0", "joint_1", "no_x_off_1"}
    for p in bank["policies"].values():
        saved = runtime["checkpoint_cache"].load(p)
        assert all(int(v["step"]) == 1 for v in saved["optimizer"]["state"].values())
    assert bank["contrasts"]["joint_1_minus_joint_0"]["norm"] >= 0
    count = runtime["adapter"].generation_calls
    gpu.run_forks(runtime, state, plan, prompts, out=tmp_path / "forks", resume=True, bridge=True)
    assert runtime["adapter"].generation_calls == count


def test_authorization_checked_before_read_or_model_load():
    with pytest.raises(PermissionError):
        gpu.run_worker("missing.json", worker_id=0, workers=2)
    with pytest.raises(PermissionError):
        gpu.load_runtime({}, {}, execute_gpu=False)


def test_fast_path_errors_are_measured_and_fall_back(tmp_path):
    runtime, module = tiny_runtime()
    state = runtime["state"].capture()
    policy = gpu._save_policy(runtime, state, tmp_path / "p.pt", {"candidate_id": "p"})
    prompts = module.fake_prompts(1)
    actions = [{"prompt_id": prompts[0]["prompt_id"], "token_ids": [0, 4]}]

    def score(prepared, tokens, mode):
        return ([0] * len(tokens), [20.0] * len(tokens))

    runtime["adapter"].continuation_scores = score
    result = gpu.measure_scoring_paths(runtime, policy, prompts, actions)
    assert not result["fast_path_available"]
    assert result["mode"] == "uncached_prefix_recompute"
    assert len(result["alternatives"]) == 3
    assert all(not p["passed"] for p in result["alternatives"])


def test_orphan_parquet_commit_resume_keeps_original_and_no_regeneration(tmp_path, monkeypatch):
    original = gpu._publish

    def fail_commit(path, value):
        if Path(path).name == "chunk_00000000.json":
            raise RuntimeError("after parquet before commit")
        return original(path, value)

    monkeypatch.setattr(gpu, "_publish", fail_commit)
    with pytest.raises(RuntimeError):
        gpu._chunks(
            tmp_path / "raw",
            {"role": "work"},
            lambda i: {"sample_id": str(i)},
            count=2,
            resume=False,
        )
    raw = tmp_path / "raw" / "chunk_00000000.parquet"
    before = raw.read_bytes()
    monkeypatch.setattr(gpu, "_publish", original)

    def forbidden(i):
        raise AssertionError("Committed bytes must not be regenerated")

    receipt = gpu._chunks(tmp_path / "raw", {"role": "work"}, forbidden, count=2, resume=True)
    assert raw.read_bytes() == before and receipt["count"] == 2
    raw.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        gpu.verify_artifact_bindings(receipt)


def test_orphan_candidate_checkpoint_replays_exact_state(tmp_path, monkeypatch):
    runtime, module = tiny_runtime()
    origin = runtime["state"].capture({"step": 32})
    prompts = module.fake_prompts(4, split="train")
    plan = {
        "origin_id": "o",
        "banks": [
            {
                "bank_id": "calibration_000",
                "role": "calibration",
                "prompt_ids": [p["prompt_id"] for p in prompts],
            }
        ],
    }
    original = gpu._publish

    def fail(path, value):
        if Path(path).name == "joint_0.json":
            raise RuntimeError("after checkpoint")
        return original(path, value)

    monkeypatch.setattr(gpu, "_publish", fail)
    with pytest.raises(RuntimeError):
        gpu.run_forks(runtime, origin, plan, prompts, out=tmp_path / "forks", bridge=True)
    checkpoint = tmp_path / "forks" / "calibration_000" / "joint_0.pt"
    before = checkpoint.read_bytes()
    monkeypatch.setattr(gpu, "_publish", original)
    result = gpu.run_forks(
        runtime, origin, plan, prompts, out=tmp_path / "forks", bridge=True, resume=True
    )
    assert checkpoint.read_bytes() == before and len(result["banks"]) == 1


def test_actual_producer_to_group_derivative_and_safe_payload(tmp_path):
    from src.r4_inputs import STRATA

    runtime, module = tiny_runtime()
    origin = runtime["state"].capture({"step": 32})
    policy = gpu._save_policy(
        runtime, origin, tmp_path / "origin.pt", {"candidate_id": "origin"}, full=True
    )
    prompts = module.fake_prompts(6)
    for prompt, (family, interface) in zip(prompts, STRATA, strict=True):
        prompt.update(family=family, interface=interface)
    work = gpu.collect_actions(
        runtime, policy, prompts, origin_id="o", role="work", draws=2, out=tmp_path / "work"
    )
    rows = list(gpu._rows(work))
    assert all(
        r["proposal"] == "ORIGIN" and r["proposal_sequence_logp"] == r["generation_sequence_logp"]
        for r in rows
    )
    result = gpu._score_derivative(
        runtime,
        {"origin_id": "o", "origin_policy": policy},
        prompts,
        {"work": work},
        out=tmp_path / "gradient",
        prompt_subset=[p["prompt_id"] for p in prompts],
    )
    saved = torch.load(result["artifact"]["path"], weights_only=True, map_location="cpu")
    assert result["anchor_samples"] == 12
    assert set(saved["grouped_gradients"]) == set(runtime["parameter_order"])
    assert all(tuple(v.shape[:2]) == (6, 4) for v in saved["grouped_gradients"].values())


def test_actual_bridge_interruption_resume_and_noop(tmp_path):
    runtime, module = tiny_runtime()
    state = runtime["state"].capture()
    policy = gpu._save_policy(runtime, state, tmp_path / "p.pt", {"candidate_id": "origin"})
    result = gpu.exercise_bridge_resume(
        runtime, policy, module.fake_prompts(1), origin_id="o", out=tmp_path / "smoke"
    )
    assert result["status"] == "INTERRUPTION_RESUME_MEASURED"
    assert result["actual_generation_calls_after_resume"] == 2
    assert runtime["adapter"].generation_calls == 4
    gpu.exercise_bridge_resume(
        runtime, policy, module.fake_prompts(1), origin_id="o", out=tmp_path / "smoke"
    )
    assert runtime["adapter"].generation_calls == 4


def test_checkpoint_direction_provider_preserves_actual_full_score_products(tmp_path, monkeypatch):
    import numpy as np

    from src.r4_inputs import STRATA

    runtime, module = tiny_runtime()
    origin = runtime["state"].capture({"step": 32})
    train = module.fake_prompts(4, split="train")
    plan = {
        "origin_id": "o",
        "banks": [
            {"bank_id": "query_000", "role": "query", "prompt_ids": [p["prompt_id"] for p in train]}
        ],
    }
    forks = gpu.run_forks(runtime, origin, plan, train, out=tmp_path / "forks", bridge=True)
    prompts = module.fake_prompts(6)
    for prompt, (family, interface) in zip(prompts, STRATA, strict=True):
        prompt.update(family=family, interface=interface)
    work = gpu.collect_actions(
        runtime,
        forks["origin_policy"],
        prompts,
        origin_id="o",
        role="work",
        draws=2,
        out=tmp_path / "work",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SSVC_V4_SCRATCH_DIR", str(scratch))
    receipt = gpu._score_derivative(
        runtime, forks, prompts, {"work": work}, out=tmp_path / "derivative"
    )
    saved = torch.load(receipt["artifact"]["path"], map_location="cpu", weights_only=True)
    assert saved["raw_directional_score_products"].shape == (12, 3)
    assert saved["direction_ids"] == list(gpu._direction_bindings(forks))
    assert not list(scratch.iterdir())
    vectors = []
    for spec in gpu._direction_bindings(forks).values():
        left = runtime["checkpoint_cache"].load(spec["left"])["parameters"]
        right = runtime["checkpoint_cache"].load(spec["right"])["parameters"]
        vectors.append(
            np.concatenate(
                [
                    (left[n].double() - right[n].double()).numpy().reshape(-1)
                    for n in runtime["parameter_order"]
                ]
            )
        )
    vectors = np.stack(vectors)
    np.testing.assert_allclose(saved["raw_parameter_gram"], vectors @ vectors.T, atol=1e-16)
    raw = next(iter(gpu._rows(work)))
    from src.modeling_v4.score_response import score_sample_gradient

    runtime["policy_loader"](forks["origin_policy"])
    actual = score_sample_gradient(runtime, prompts[0], raw)
    gradient = np.concatenate(
        [actual["gradients"][n].reshape(-1) for n in runtime["parameter_order"]]
    )
    np.testing.assert_allclose(
        saved["raw_directional_score_products"][0], vectors @ gradient, atol=1e-16
    )


def test_phase_measurements_bind_task_and_keep_failure_costs():
    runtime, _ = tiny_runtime()
    runtime.update(current_task_id="B_origin_0", current_origin_id="B_origin_0", current_stage="B")

    def fail():
        runtime["adapter"].forward_calls += 3
        raise RuntimeError("deliberate fixture")

    with pytest.raises(RuntimeError):
        gpu._measure_phase(runtime, "score", fail)
    measurement = runtime["phase_measurements"][0]
    assert measurement["task_id"] == "B_origin_0" and measurement["stage"] == "B"
    assert measurement["forward_calls"] == 3 and measurement["status"] == "FAILED"


def test_source_recovers_committed_update_before_latest_without_retraining(tmp_path, monkeypatch):
    runtime, module = tiny_runtime()
    prompts = module.fake_prompts(4, split="train")
    monkeypatch.setattr(
        gpu,
        "build_training_schedule",
        lambda p, s: {"train_steps": [[v["prompt_id"] for v in p]] * 2, "schedule_hash": "fixture"},
    )
    config = {"qwen": {"seed_roles": {"development": [41001]}}}
    original = gpu._replace_json

    def fail(path, value):
        if Path(path).name == "LATEST.json" and value["step"] == 1:
            raise RuntimeError("after UPDATE before cursor")
        return original(path, value)

    monkeypatch.setattr(gpu, "_replace_json", fail)
    with pytest.raises(RuntimeError):
        gpu.run_source_trajectory(
            config,
            runtime,
            prompts,
            seed=41001,
            arm="X_BASE",
            out=tmp_path / "source",
            steps=2,
            fixture=True,
        )
    assert runtime["adapter"].generation_calls == 32
    committed = (tmp_path / "source" / "steps" / "step_001" / "UPDATE.json").read_bytes()
    monkeypatch.setattr(gpu, "_replace_json", original)
    result = gpu.run_source_trajectory(
        config,
        runtime,
        prompts,
        seed=41001,
        arm="X_BASE",
        out=tmp_path / "source",
        steps=2,
        fixture=True,
        resume=True,
    )
    assert result["steps"] == 2 and runtime["adapter"].generation_calls == 64
    assert (tmp_path / "source" / "steps" / "step_001" / "UPDATE.json").read_bytes() == committed
    state = runtime["checkpoint_cache"].load(result["latest"]["checkpoint"])
    assert all(int(v["step"]) == 2 for v in state["optimizer"]["state"].values())


def test_nested_action_and_score_prefix_reuses_original_bytes_and_calls(tmp_path):
    runtime, module = tiny_runtime()
    state = runtime["state"].capture()
    policy = gpu._save_policy(runtime, state, tmp_path / "p.pt", {"candidate_id": "origin"})
    prompts = module.fake_prompts(3)
    first = gpu.collect_actions(
        runtime, policy, prompts[:2], origin_id="o", role="work", draws=32, out=tmp_path / "first"
    )
    scored = gpu.collect_scores(runtime, policy, prompts[:2], first, out=tmp_path / "first_scores")
    before = runtime["adapter"].generation_calls
    full = gpu.collect_actions(
        runtime,
        policy,
        prompts,
        origin_id="o",
        role="work",
        draws=64,
        out=tmp_path / "full",
        prefix=first,
    )
    assert runtime["adapter"].generation_calls - before == 128
    assert {r["sample_key"] for r in gpu._rows(first)} <= {r["sample_key"] for r in gpu._rows(full)}
    assert len({r["sample_key"] for r in gpu._rows(full)}) == 192
    before = runtime["adapter"].forward_calls
    full_scores = gpu.collect_scores(
        runtime, policy, prompts, full, out=tmp_path / "full_scores", prefix=scored
    )
    assert runtime["adapter"].forward_calls - before == 128
    assert full_scores["chunks"][0]["path"] == scored["chunks"][0]["path"]
    assert full["chunks"][0]["path"] == first["chunks"][0]["path"]


def test_source_completed_artifact_corruption_refuses_resume(tmp_path, monkeypatch):
    runtime, module = tiny_runtime()
    prompts = module.fake_prompts(4, split="train")
    monkeypatch.setattr(
        gpu,
        "build_training_schedule",
        lambda p, s: {"train_steps": [[v["prompt_id"] for v in p]], "schedule_hash": "fixture"},
    )
    config = {"qwen": {"seed_roles": {"development": [41001]}}}
    result = gpu.run_source_trajectory(
        config,
        runtime,
        prompts,
        seed=41001,
        arm="X_BASE",
        out=tmp_path / "source",
        steps=1,
        fixture=True,
    )
    Path(result["latest"]["checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        gpu.run_source_trajectory(
            config,
            runtime,
            prompts,
            seed=41001,
            arm="X_BASE",
            out=tmp_path / "source",
            steps=1,
            fixture=True,
            resume=True,
        )


def test_actual_small_response_has_independent_direct_lr_and_pair_diagnostics(tmp_path):
    runtime, module = tiny_runtime()
    state = runtime["state"].capture({"step": 32})
    prompts = module.fake_prompts(4, split="train")
    plan = {
        "origin_id": "o",
        "banks": [
            {
                "bank_id": "query_000",
                "role": "query",
                "prompt_ids": [p["prompt_id"] for p in prompts],
            }
        ],
    }
    forks = gpu.run_forks(runtime, state, plan, prompts, out=tmp_path / "forks", bridge=True)
    probes = module.fake_prompts(2)
    response = gpu.collect_response_map(
        {},
        runtime,
        forks,
        probes,
        out=tmp_path / "response",
        draws=16,
        reference_draws=16,
        bridge=True,
    )
    assert len(response["direct_scores"]) == 3
    namespaces = [
        response[k]["identity"]["rng_namespace"] for k in ("work", "reference", "direct_work")
    ]
    assert len(set(namespaces)) == 3
    assert response["direct_baseline"] == "DIRECT_LR_ORIGIN_INDEPENDENT_PACKET"
    diagnostic = gpu.pair_observation_diagnostics(response)
    assert diagnostic["all_finite"]
    for unit in diagnostic["units"]:
        assert unit["conditional_support_bounds"]["valid_conditional_bound"]
        assert unit["conditional_support_bounds"]["model_calls"] == 0
        assert unit["endpoint_count_intervals"]["single_event_coverage_at_least"] == 0.95
    assert response["scientific_status"] == "NOT_CERTIFIED"
    before = runtime["adapter"].generation_calls
    gpu.collect_response_map(
        {},
        runtime,
        forks,
        probes,
        out=tmp_path / "response",
        draws=16,
        reference_draws=16,
        bridge=True,
        resume=True,
    )
    assert runtime["adapter"].generation_calls == before


def test_alias_endpoints_keep_actual_adam_rng_originals(tmp_path):
    from src.optimizer_fork import state_hash

    runtime, module = tiny_runtime()
    for group in runtime["optimizer"].param_groups:
        group["lr"] = 0.0
    state = runtime["state"].capture({"step": 32})
    prompts = module.fake_prompts(4, split="train")
    plan = {
        "origin_id": "o",
        "banks": [
            {
                "bank_id": "calibration_000",
                "role": "calibration",
                "prompt_ids": [p["prompt_id"] for p in prompts],
            }
        ],
    }
    result = gpu.run_forks(runtime, state, plan, prompts, out=tmp_path / "forks", bridge=False)
    policies = result["banks"][0]["policies"]
    assert len({p["inference_fingerprint"] for p in policies.values()}) == 1
    for policy in policies.values():
        original = runtime["checkpoint_cache"].load(policy["alias_training_state"])
        assert state_hash(original) == policy["training_state_hash"]
        assert all(int(v["step"]) == 1 for v in original["optimizer"]["state"].values())
        assert "rng" in original and policy["complete_training_state_retained"]


def test_gradient_orphan_commit_recovery_reuses_actual_payload(tmp_path, monkeypatch):
    runtime, module = tiny_runtime()
    state = runtime["state"].capture()
    policy = gpu._save_policy(runtime, state, tmp_path / "p.pt", {"candidate_id": "origin"})
    probes = module.fake_prompts(1)
    work = gpu.collect_actions(
        runtime, policy, probes, origin_id="o", role="work", draws=2, out=tmp_path / "work"
    )
    forks = {"origin_id": "o", "origin_policy": policy}
    original = gpu._publish

    def fail(path, value):
        if Path(path) == tmp_path / "gradient" / "COMPLETE.json":
            raise RuntimeError("after payload before commit")
        return original(path, value)

    monkeypatch.setattr(gpu, "_publish", fail)
    with pytest.raises(RuntimeError):
        gpu._score_derivative(
            runtime, forks, probes, {"work": work}, out=tmp_path / "gradient", bridge=True
        )
    before = runtime["adapter"].forward_calls
    monkeypatch.setattr(gpu, "_publish", original)
    result = gpu._score_derivative(
        runtime, forks, probes, {"work": work}, out=tmp_path / "gradient", bridge=True
    )
    assert runtime["adapter"].forward_calls == before and result["status"] == "MEASURED"


def test_registered_nested_draws_and_validation_extend_only_missing_samples(tmp_path):
    runtime, module = tiny_runtime()
    state = runtime["state"].capture({"step": 32})
    train = module.fake_prompts(4, split="train")
    probes = module.fake_prompts(2)
    plan = {
        "origin_id": "o",
        "banks": [
            {"bank_id": "query_000", "role": "query", "prompt_ids": [p["prompt_id"] for p in train]}
        ],
    }
    forks = gpu.run_forks(runtime, state, plan, train, out=tmp_path / "forks", bridge=True)
    response = gpu.collect_response_map(
        {}, runtime, forks, probes, out=tmp_path / "main", draws=32, reference_draws=32, bridge=True
    )
    before = runtime["adapter"].generation_calls
    extended = gpu.collect_nested_measurements(
        runtime,
        forks,
        probes,
        response,
        {
            "n_curve_max_draws": 64,
            "reference_validation": {
                "draws": 64,
                "bank_ids": ["query_000"],
                "prompt_scope": "all_frozen_observation_prompts",
            },
        },
        out=tmp_path / "extended",
    )
    assert runtime["adapter"].generation_calls - before == 192
    assert extended["work"]["identity"]["draws"] == 32
    assert extended["ncurve_work"]["identity"]["draws"] == 64
    assert extended["reference"]["identity"]["draws"] == 32
    assert extended["reference_validation"]["identity"]["draws"] == 64
    assert extended["ncurve_work"]["chunks"][0]["path"] == response["work"]["chunks"][0]["path"]
    assert extended["ncurve_direct_work"]["identity"]["draws"] == 64
    assert (
        extended["ncurve_direct_work"]["chunks"][0]["path"]
        == response["direct_work"]["chunks"][0]["path"]
    )
    assert (
        extended["ncurve_direct_work"]["identity"]["rng_namespace"]
        != extended["ncurve_work"]["identity"]["rng_namespace"]
    )
    assert set(extended["ncurve_direct_scores"]) == set(response["direct_scores"])
    assert all(
        "token_ids" not in r and r["shared_token_identity"]
        for score in extended["ncurve_scores"].values()
        for r in gpu._rows(score["score_receipt"])
    )
