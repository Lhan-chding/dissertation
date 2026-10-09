"""CPU checks of actual F2 objective/state logic; these do not certify 9B CUDA."""

from __future__ import annotations

import copy
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import capture_rng, state_hash, trainable_state
from mm_dev.contract import PLAN_ID, digest
from mm_dev.runtime import F2Runtime, atomic_json, reference_parameters
from mm_dev.training import (
    checkpoint_state,
    commit_checkpoint,
    group_diagnostics,
    load_checkpoint,
    read_json,
    rewards,
    rollout_group,
    sequence_objective,
    update,
)


def test_exact_partial_reading_reward_and_missing_fields():
    row = dict(true_values=[13, 24, 35], operation="range")
    result = rewards('{"readings":[13,24,0],"answer":0}', row)
    assert result["q_read"] == "2/3" and result["rAP"] == "1/3"
    assert rewards('{"answer":22}', row)["rAP"] == "1/2"
    assert rewards('{"readings":[13,24],"answer":22}', row)["q_read"] == "0"
    assert rewards('{"answer":true}', row)["A_full"] == 0
    assert rewards('{"answer":22,"answer":0}', row)["rA"] == "0"


@pytest.mark.parametrize(
    "reward", sorted({Fraction(i, 2 * n) for n in (2, 3) for i in range(2 * n + 1)})
)
def test_all_reachable_constant_fraction_groups_have_exact_zero_advantages(reward):
    rows = [dict(reward=dict(rA="0", rAP=str(reward)))] * 8
    result = group_diagnostics(rows, "AP")
    assert result["zero_contrast"] is True
    assert result["advantages"] == [0.0] * 8
    assert result["reward_population_variance"] == "0"


def test_population_normalization_counterfactuals_and_no_discard():
    rows = [dict(reward=dict(rA="0", rAP="1/3"))] * 4 + [dict(reward=dict(rA="1", rAP="1"))] * 4
    ap, answer = group_diagnostics(rows, "AP"), group_diagnostics(rows, "A")
    assert len(ap["advantages"]) == 8
    assert ap["reward_mean"] == "2/3"
    assert ap["reward_population_variance"] == "1/9"
    assert ap["counterfactual_advantages_A"] == answer["advantages"]


def test_objective_is_sequence_mean_with_separate_policy_and_kl():
    current = torch.tensor([-0.6, -1.1, -0.4], requires_grad=True)
    old = torch.tensor([-0.7, -0.8, -0.5])
    reference = torch.tensor([-0.2, -0.9, -0.8])
    result = sequence_objective(current, old, reference, -0.5)
    ratio = (current - old).exp()
    expected_policy = -torch.minimum(-0.5 * ratio, -0.5 * ratio.clamp(0.8, 1.2)).mean()
    expected_kl = ((reference - current).expm1() - (reference - current)).mean()
    assert torch.equal(result["loss"], expected_policy + 0.02 * expected_kl)
    grad = torch.autograd.grad(result["policy"], current, retain_graph=True)[0]
    total = torch.autograd.grad(result["loss"], current)[0]
    assert grad.norm() > 0 and not torch.equal(grad, total)
    with pytest.raises(ValueError, match="detached"):
        sequence_objective(current, current, reference, 1)
    with pytest.raises(FloatingPointError):
        sequence_objective(
            torch.tensor([1000.0], requires_grad=True), torch.tensor([0.0]), torch.tensor([0.0]), 1
        )


def test_reference_switch_preserves_optimizer_identity_and_restores_after_exception():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    current = trainable_state(model)
    reference = {n: torch.zeros_like(v) for n, v in current.items()}
    parameter_id = id(optimizer.param_groups[0]["params"][0])
    with pytest.raises(RuntimeError, match="fixture"), reference_parameters(model, reference):
        assert model.weight.count_nonzero() == 0
        raise RuntimeError("fixture")
    assert id(model.weight) == parameter_id
    assert state_hash(trainable_state(model)) == state_hash(current)


class TinyRuntime:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[0.1, -0.2], [0.3, 0.1]]))
        self.events = []

    def prepare(self, row, _root):
        return dict(x=torch.tensor(row["x"]))

    def reserve(self, kind, count, **metadata):
        self.events.append((kind, count, metadata))

    def sequence_forward(self, prepared, tokens, *, purpose, grad=False):
        self.reserve("gradient_forward_sequences" if grad else "reference_forward_sequences", 1)
        with torch.enable_grad() if grad else torch.no_grad():
            logits = self.model(prepared["x"])
            lp = logits.log_softmax(-1)[torch.tensor(tokens)]
        return dict(logprobs=lp)

    def reference_forward(self, prepared, tokens, reference):
        with reference_parameters(self.model, reference):
            return self.sequence_forward(prepared, tokens, purpose="reference")


def examples(runtime):
    result = []
    for i in range(192):
        row = dict(x=[1.0, 0.3 + (i % 3) * 0.1])
        tokens = [i % 2] * (1 + i % 3)
        old = runtime.sequence_forward(runtime.prepare(row, None), tokens, purpose="rollout")
        result.append(
            dict(
                row=row,
                root=None,
                advantage=1.0 if i % 2 == 0 else -1.0,
                record=dict(tokens=tokens, old_logprobs=old["logprobs"].tolist()),
            )
        )
    return result


def optimizer_for(runtime):
    optimizer = torch.optim.AdamW(
        runtime.model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def test_full_192_microbatches_nonzero_pg_and_reference_remains_fixed():
    runtime = TinyRuntime()
    optimizer, scheduler = optimizer_for(runtime)
    ref = trainable_state(runtime.model)
    examples_ = examples(runtime)
    runtime.events.clear()
    metrics = update(runtime, optimizer, scheduler, ref, examples_, probability_gate=True)
    assert metrics["policy_gradient_norm"] > 0
    assert metrics["parameter_changed"] and metrics["parameter_max_absolute_delta"] > 0
    assert metrics["effective_sequences"] == 192
    assert metrics["reference_hash"] == state_hash(ref)
    assert sum(n for kind, n, _ in runtime.events if kind == "reference_forward_sequences") == 192
    assert sum(n for kind, n, _ in runtime.events if kind == "gradient_forward_sequences") == 192
    assert sum(n for kind, n, _ in runtime.events if kind == "physical_optimizer_updates") == 1
    with pytest.raises(ValueError, match="24 x 8"):
        update(runtime, optimizer, scheduler, ref, examples_[:-1])


def test_constant_advantages_distinguish_kl_gradient_from_policy_gradient():
    runtime = TinyRuntime()
    optimizer, scheduler = optimizer_for(runtime)
    ref = {n: v + 0.2 * torch.eye(2) for n, v in trainable_state(runtime.model).items()}
    batch = examples(runtime)
    for item in batch:
        item["advantage"] = 0.0
    metrics = update(runtime, optimizer, scheduler, ref, batch)
    assert metrics["policy_gradient_norm"] == 0
    assert metrics["total_gradient_norm_before_clip"] > 0
    assert metrics["parameter_changed"]


def test_four_versus_restored_two_plus_two_exact_adam_rng_and_reference(tmp_path):
    def trajectory(path, steps, checkpoint=None):
        runtime = TinyRuntime()
        optimizer, scheduler = optimizer_for(runtime)
        ref = trainable_state(runtime.model)
        identity = dict(plan_id=PLAN_ID, fixture="CPU_actual_torch_GRPO")
        if checkpoint:
            initial = load_checkpoint(
                checkpoint,
                runtime,
                optimizer,
                scheduler,
                run_identity=identity,
                reference=ref,
                stream_hash="stream",
                sampling_hash="sampling",
                step=2,
            )
            start = initial["committed_logical_step"]
        else:
            from mm_core.vl_runtime import seed_all

            seed_all(12345)
            start = 0
        for step in range(start + 1, steps + 1):
            metrics = update(runtime, optimizer, scheduler, ref, examples(runtime))
            torch.rand(3)  # Proves restoration of a changing RNG, not only a constant hash.
            state = checkpoint_state(
                runtime,
                optimizer,
                scheduler,
                run_identity=identity,
                step=step,
                reference=ref,
                stream_hash="stream",
                sampling_hash="sampling",
                diagnostics_hash=digest(metrics),
                token_path_hash="actual_fixture",
            )
            commit_checkpoint(path, state, retain_all=True)
        return state

    continuous = trajectory(tmp_path / "continuous", 4)
    trajectory(tmp_path / "split", 2)
    torch.rand(100)
    resumed = trajectory(tmp_path / "split", 4, tmp_path / "split")
    assert state_hash(continuous) == state_hash(resumed)
    assert all(value["exp_avg"].count_nonzero() for value in resumed["optimizer"]["state"].values())
    receipt = read_json(tmp_path / "split/LATEST.json")
    assert receipt["field_hashes"]["rng"] == state_hash(capture_rng())


def test_rollout_resume_reuses_raw_sampler_record_and_rejects_identity_changes(tmp_path):
    row = dict(
        question_id="q",
        image_path="image.png",
        image_sha256="imghash",
        prompt="visible",
        true_values=[1, 2],
        operation="sum",
    )
    run = dict(phase="PREP", repeat=0, run_id="prep_A_0")
    slot = dict(logical_step=1, slot=0)
    runtime = SimpleNamespace(calls=0)

    def generate(_row, _root, sample_seed, on_completion):
        runtime.calls += 1
        on_completion(
            dict(
                raw_text='{"readings":[1,2],"answer":3}',
                tokens=[1, 2],
                generation_status="COMPLETE",
                old_logprobs=[-0.5, -0.6],
                seed=sample_seed,
                sampling_hash="sample",
            )
        )

    runtime.generate_training = generate
    first = rollout_group(runtime, tmp_path, tmp_path, run, slot, row, "policy", "sample")
    second = rollout_group(runtime, tmp_path, tmp_path, run, slot, row, "policy", "sample")
    assert first == second and runtime.calls == 8
    with pytest.raises(PermissionError, match="identity"):
        rollout_group(runtime, tmp_path, tmp_path, run, slot, row, "different", "sample")
    raw = tmp_path / "rollouts/01-00-0.json"
    value = read_json(raw)
    value["tokens"] = [99]
    atomic_json(raw, value)
    with pytest.raises(PermissionError, match="changed"):
        rollout_group(runtime, tmp_path, tmp_path, run, slot, row, "policy", "sample")


def test_actual_sampler_logits_are_preserved_before_scoring_and_rng_isolated():
    from transformers import GenerationConfig

    runtime = F2Runtime.__new__(F2Runtime)
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.eos_ids = [2]
    runtime.account = lambda *_args: None
    runtime.generation_config = GenerationConfig(
        do_sample=True, max_new_tokens=192, temperature=0.7, top_p=0.9, top_k=0
    )
    logits = (torch.tensor([[0.2, 0.7, -0.4]]), torch.tensor([[0.1, -0.5, 0.9]]))

    def generate(**kwargs):
        assert kwargs["generation_config"].temperature == 1
        assert kwargs["generation_config"].top_p == 1
        runtime.image_calls += 1
        torch.rand(10)
        return SimpleNamespace(
            sequences=torch.tensor([[9, 1, 2]]),
            logits=logits,
            scores=tuple(t.clone() for t in logits),
        )

    runtime.model = SimpleNamespace(generate=generate, eval=lambda: None)
    runtime.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *_a, **_kw: "raw"))
    runtime.prepare = lambda *_a: dict(inputs=dict(input_ids=torch.tensor([[9]])), routing={})
    saved = []
    before = state_hash(capture_rng())
    result = runtime.generate_training(
        dict(question_id="fixture"), None, 34, on_completion=saved.append
    )
    assert saved == [result]
    assert result["old_logprobs"] == [
        float(t[0].log_softmax(-1)[i]) for t, i in zip(logits, (1, 2), strict=True)
    ]
    assert result["old_logprob_source"] == "actual_generate_raw_logits_selected_token"
    assert before == state_hash(capture_rng())
    assert runtime.generation_config.temperature == 0.7


def test_checkpoint_receipt_tampering_blocks_restore(tmp_path):
    runtime = TinyRuntime()
    optimizer, scheduler = optimizer_for(runtime)
    ref = trainable_state(runtime.model)
    state = checkpoint_state(
        runtime,
        optimizer,
        scheduler,
        run_identity={},
        step=0,
        reference=ref,
        stream_hash="s",
        sampling_hash="g",
    )
    commit_checkpoint(tmp_path, state)
    changed = copy.deepcopy(read_json(tmp_path / "LATEST.json"))
    changed["field_hashes"]["reference"] = "wrong"
    atomic_json(tmp_path / "LATEST.json", changed)
    with pytest.raises(PermissionError, match="hashes"):
        load_checkpoint(
            tmp_path,
            runtime,
            optimizer,
            scheduler,
            run_identity={},
            reference=ref,
            stream_hash="s",
            sampling_hash="g",
        )


def test_committed_metrics_recovery_uses_checkpoint_bound_attempt(tmp_path):
    from mm_dev.training import recover_committed_metrics

    metrics = dict(logical_step=2, policy_gradient_norm=1.2, token_path_hash="tokens")
    state = dict(committed_logical_step=2, diagnostics_hash=digest(metrics))
    atomic_json(tmp_path / "update_attempts/02-attempt.json", metrics)
    recover_committed_metrics(tmp_path, state)
    assert read_json(tmp_path / "steps/02.json") == metrics
    atomic_json(tmp_path / "steps/02.json", {**metrics, "policy_gradient_norm": 9})
    with pytest.raises(PermissionError, match="changed"):
        recover_committed_metrics(tmp_path, state)


def test_live_prepare_checks_all_frozen_native_routing_and_projects_inputs(tmp_path, monkeypatch):
    from mm_core.vl_runtime import QwenRuntime
    from mm_dev.runtime import file_hash

    routing = dict(
        processed_pixel_sha256="pixels",
        input_ids_sha256="input",
        image_token_count=588,
        image_grid_thw=[[1, 42, 56]],
        mm_token_type_ids_sha256="mm",
        processed_size=[896, 672],
    )
    row = dict(
        question_id="q",
        image_id="i",
        image_path="i.png",
        image_sha256="bytes",
        prompt="visible",
        prompt_sha256="prompt",
        true_values=[10, 20],
        reward=1,
    )
    path = tmp_path / "data/processor_routing.jsonl"
    path.parent.mkdir()
    import json

    path.write_text(json.dumps(dict(question_id="q", image_id="i", **routing)) + "\n")
    atomic_json(
        tmp_path / "manifests/F2_FREEZE.json",
        dict(input_hashes={"data/processor_routing.jsonl": file_hash(path)}),
    )
    captured = []

    def fake_prepare(_self, safe, _root):
        captured.append(safe)
        return dict(inputs={}, routing=copy.deepcopy(routing))

    monkeypatch.setattr(QwenRuntime, "prepare", fake_prepare)
    runtime = F2Runtime.__new__(F2Runtime)
    assert runtime.prepare(row, tmp_path)["routing"] == routing
    assert "true_values" not in captured[0] and "reward" not in captured[0]
    routing["image_token_count"] = 587
    with pytest.raises(PermissionError, match="image_token_count"):
        runtime.prepare(row, tmp_path)


def test_probability_gate_rejects_before_optimizer_mutation():
    runtime = TinyRuntime()
    optimizer, scheduler = optimizer_for(runtime)
    ref = trainable_state(runtime.model)
    batch = examples(runtime)
    batch[0]["record"]["old_logprobs"][0] += 0.051
    with pytest.raises(RuntimeError, match="NUMERIC_GATE_FAILED"):
        update(runtime, optimizer, scheduler, ref, batch, probability_gate=True)
    assert state_hash(trainable_state(runtime.model)) == state_hash(ref)
    assert not optimizer.state


def test_engine_interrupted_continuous_trace_is_archived_not_resumed(tmp_path, monkeypatch):
    from mm_dev.engine import prepare_engine_track

    old = tmp_path / "engineering/natural/continuous"
    atomic_json(old / "checkpoints/LATEST.json", dict(step=1))
    monkeypatch.setenv("MM_DEV_ATTEMPT_ID", "registered-attempt-2")
    prepare_engine_track(tmp_path, "natural", "continuous")
    assert not old.exists()
    archived = tmp_path / "engineering/interrupted/natural/continuous-registered-attempt-2"
    assert read_json(archived / "checkpoints/LATEST.json") == dict(step=1)


def test_invalid_sampler_transform_preserves_raw_before_rejecting():
    from transformers import GenerationConfig

    runtime = F2Runtime.__new__(F2Runtime)
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.eos_ids = [2]
    runtime.account = lambda *_a: None
    runtime.generation_config = GenerationConfig(do_sample=True, max_new_tokens=192)

    def generate(**_kwargs):
        runtime.image_calls += 1
        return SimpleNamespace(
            sequences=torch.tensor([[9, 2]]),
            logits=(torch.tensor([[0.2, 0.3, 0.7]]),),
            scores=(torch.tensor([[0.2, 0.3, 0.8]]),),
        )

    runtime.model = SimpleNamespace(generate=generate, eval=lambda: None)
    runtime.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *_a, **_kw: "raw"))
    runtime.prepare = lambda *_a: dict(inputs=dict(input_ids=torch.tensor([[9]])), routing={})
    saved = []
    with pytest.raises(RuntimeError, match="Recorded invalid"):
        runtime.generate_training(dict(question_id="fixture"), None, 34, on_completion=saved.append)
    assert saved[0]["tokens"] == [2]
    assert saved[0]["generation_status"] == "TECHNICAL_INVALID"
    assert saved[0]["technical_validation_errors"] == ["UNREGISTERED_LOGITS_TRANSFORM"]


def test_pinned_lease_checkpoint_survives_rolling_cleanup(tmp_path, monkeypatch):
    from mm_dev.training import pin_checkpoint

    runtime = TinyRuntime()
    optimizer, scheduler = optimizer_for(runtime)
    ref = trainable_state(runtime.model)
    directory = tmp_path / "training/fixture"
    monkeypatch.setenv("MM_DEV_ATTEMPT_ID", "attempt1")
    for step in (0, 1):
        state = checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            run_identity={},
            step=step,
            reference=ref,
            stream_hash="s",
            sampling_hash="g",
        )
        commit_checkpoint(directory / "checkpoints", state)
    artifacts = pin_checkpoint(tmp_path, directory, "fixture")
    for step in (2, 3):
        state = {
            **state,
            "committed_logical_step": step,
            "cursor": dict(next_logical_step=step + 1, next_slot=0, next_sample_index=0),
        }
        commit_checkpoint(directory / "checkpoints", state)
    assert all((tmp_path / relative).is_file() for relative in artifacts)
    assert not list((directory / "checkpoints").glob("step-01-*.pt"))
    assert list((directory / "checkpoints").glob("step-00-*.pt"))


def test_native_hybrid_qwen_sampler_exposes_real_raw_selected_logits():
    import transformers

    config = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            rope_parameters=dict(
                rope_type="default",
                mrope_section=[1, 1, 2],
                mrope_interleaved=True,
                partial_rotary_factor=1.0,
            ),
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
    )
    torch.manual_seed(981)
    runtime = F2Runtime.__new__(F2Runtime)
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.model = transformers.Qwen3_5ForConditionalGeneration(config).eval()
    runtime.account = lambda *_a: None
    runtime.eos_ids = [2]
    runtime.generation_config = transformers.GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=192,
        eos_token_id=2,
        pad_token_id=0,
    )
    config_generation = runtime.training_generation_config()
    config_generation.max_new_tokens = 3  # Deliberately small CPU architecture fixture only.
    runtime.training_generation_config = lambda: config_generation
    runtime.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda tokens, **_kw: str(tokens))
    )
    runtime._vision_hook = runtime._visual_module().register_forward_pre_hook(runtime._mark_vision)
    ids = torch.tensor([[3, 58, 60, 59, 4, 5]])
    inputs = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        pixel_values=torch.full((4, 24), 0.5),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]]),
    )
    runtime.prepare = lambda *_a: dict(inputs=inputs, routing={})
    record = runtime.generate_training(dict(question_id="native-cpu-fixture"), None, 19)
    assert record["generation_status"] == "COMPLETE"
    assert record["old_logprobs"] == record["sampler_logprobs"]
    assert record["image_routing"]["generation_vision_forward_calls"] >= 1
    assert len(record["old_logprobs"]) == len(record["tokens"]) > 0
    scored = runtime.sequence_forward(dict(inputs=inputs), record["tokens"], purpose="CPU_FIXTURE")
    assert torch.allclose(
        scored["logprobs"], torch.tensor(record["old_logprobs"]), atol=1e-5, rtol=1e-5
    )
    runtime._vision_hook.remove()


def test_wrong_actual_sampler_seed_is_saved_not_relabelled(tmp_path):
    row = dict(
        question_id="q",
        image_path="image.png",
        image_sha256="imghash",
        prompt="visible",
        true_values=[1, 2],
        operation="sum",
    )

    def generate(_row, _root, sample_seed, on_completion):
        on_completion(
            dict(
                raw_text="raw",
                tokens=[1],
                old_logprobs=[-0.1],
                generation_status="COMPLETE",
                seed=sample_seed + 1,
                sampling_hash="sample",
            )
        )

    runtime = SimpleNamespace(generate_training=generate)
    with pytest.raises(PermissionError, match="Actual sampler identity"):
        rollout_group(
            runtime,
            tmp_path,
            tmp_path,
            dict(phase="PREP", repeat=0, run_id="prep_A_0"),
            dict(logical_step=1, slot=0),
            row,
            "policy",
            "sample",
        )
    saved = read_json(tmp_path / "rollouts/01-00-0.json")
    assert saved["generation_status"] == "TECHNICAL_INVALID"
    assert saved["seed"] == saved["expected_identity"]["seed"] + 1


def test_full_cpu_engine_evidence_chain_rejects_raw_tampering(tmp_path, monkeypatch):
    """Exercise 1536 real tiny-model samples; CPU fixture is never a production gate."""
    import json
    from pathlib import Path

    import mm_dev.training as training
    from mm_dev.common import CostLedger
    from mm_dev.engine import compare_engine, engine_run, verify_comparison_artifacts
    from mm_dev.runtime import file_hash

    plan_path = Path(__file__).resolve().parents[2] / "docs/mm_dev_f2/design/config/MM_DEV_F2.json"
    schedule = [
        json.loads(line)
        for line in (plan_path.parent / "schedules/ENGINE_e0.jsonl").read_text().splitlines()
    ]
    questions = {
        s["question_id"]: dict(
            question_id=s["question_id"],
            image_path="fixture.png",
            image_sha256="fixture-image",
            prompt="CPU visible fixture",
            x=[1.0, 0.5],
            true_values=[12, 20] if s["operation"] != "range" else [12, 20, 35],
            operation=s["operation"],
        )
        for s in schedule
    }
    (tmp_path / "data").mkdir()
    (tmp_path / "data/questions.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in questions.values())
    )
    atomic_json(
        tmp_path / "manifests/F2_FREEZE.json", dict(plan_id=PLAN_ID, fixture="CPU_NOT_CUDA")
    )
    initial = TinyRuntime()
    atomic_json(
        tmp_path / "manifests/COMMON_LORA.json",
        dict(trainable_state_hash=state_hash(trainable_state(initial.model))),
    )

    def configure(runtime):
        optimizer, scheduler = optimizer_for(runtime)
        return optimizer, scheduler, dict(fixture="CPU_TORCH_GRPO")

    monkeypatch.setattr(training, "configure_training", configure)

    # The CPU fixture has no production source freeze. Exercise the independent
    # comparator against its real tiny checkpoints; its fresh-process CLI and
    # production freeze checks are covered separately.
    from mm_dev.checkpoint_recount import publish_recount, verify_engine_checkpoints

    def fixture_recount(root, mode, _plan):
        path = publish_recount(root, verify_engine_checkpoints(root, mode))
        return str(path.relative_to(root))

    monkeypatch.setattr("mm_dev.engine.independent_checkpoint_recount", fixture_recount)

    class SamplingRuntime(TinyRuntime):
        def __init__(self):
            super().__init__()
            self.identity = dict(
                fixture="CPU_TORCH", trainable_state_hash=state_hash(trainable_state(self.model))
            )
            self.ledger = CostLedger(tmp_path, "ENGINE_F2")

        def training_generation_config(self):
            return SimpleNamespace(to_dict=lambda: dict(fixture="CPU_2_TOKEN_DISTRIBUTION"))

        def reserve(self, kind, count, **metadata):
            super().reserve(kind, count, **metadata)
            self.ledger.reserve(kind, count, {**metadata, "engine_mode": "natural"})

        def generate_training(self, row, root, sample_seed, *, on_completion):
            from mm_core.contracts import operate, rational

            self.reserve("completion_attempts", 1)
            with torch.no_grad():
                lp = self.model(torch.tensor(row["x"])).log_softmax(-1)
            generator = torch.Generator().manual_seed(sample_seed)
            token = int(torch.multinomial(lp.exp(), 1, generator=generator))
            truth = int(operate([rational(v) for v in row["true_values"]], row["operation"]))
            raw = json.dumps(dict(readings=row["true_values"], answer=truth if token == 0 else 999))
            logp = float(lp[token])
            on_completion(
                dict(
                    tokens=[token],
                    raw_tokens=[token],
                    raw_text=raw,
                    old_logprobs=[logp],
                    sampler_logprobs=[logp],
                    seed=sample_seed,
                    sampling_hash=digest(self.training_generation_config().to_dict()),
                    generation_status="COMPLETE",
                    truncated=False,
                    image_routing={},
                )
            )

    run = engine_run(False)
    for segment, pid, steps in (
        ("continuous", 10001, 4),
        ("first", 10002, 2),
        ("resume", 10003, 4),
    ):
        monkeypatch.setattr(training.os, "getpid", lambda pid=pid: pid)
        runtime = SamplingRuntime()
        directory = (
            tmp_path
            / "engineering/natural"
            / ("continuous" if segment == "continuous" else "split")
        )
        result = training.execute_path(
            runtime,
            tmp_path,
            directory,
            run,
            schedule,
            questions,
            stop_step=steps,
            engine=True,
            resume_step=2 if segment == "resume" else None,
        )
        result.update(
            freeze_sha256=file_hash(tmp_path / "manifests/F2_FREEZE.json"),
            runtime_identity=runtime.identity,
            frozen_base_hash="CPU_FIXED_BASE",
            mode="natural",
            segment=segment,
        )
        atomic_json(directory / (segment.upper() + "_COMPLETE.json"), result)
    result = compare_engine(tmp_path, "natural", plan_path)
    assert result["valid_trace_completions"] == result["completions"] == 1536
    assert result["valid_trace_updates"] == result["physical_updates"] == 8
    assert result["natural_nonconstant_groups"] > 0
    assert not result["stress_allowed"]
    verify_comparison_artifacts(tmp_path, result)
    raw = tmp_path / "engineering/natural/continuous/rollouts/01-00-0.json"
    changed = read_json(raw)
    changed["seed"] += 1
    atomic_json(raw, changed)
    with pytest.raises(PermissionError, match="artifact changed"):
        verify_comparison_artifacts(tmp_path, result)
    with pytest.raises(PermissionError, match="raw slot"):
        compare_engine(tmp_path, "natural", plan_path)


def test_engine_parent_forwards_real_lease_signal_to_child(tmp_path):
    import os
    import signal
    import sys
    import threading
    import time

    from mm_dev.engine import forwarding_lease_signals, wait_engine_child

    ready = tmp_path / "ready"
    child_code = (
        "import signal,sys,time,pathlib; "
        "signal.signal(signal.SIGUSR1, lambda *_: sys.exit(75)); "
        "pathlib.Path(sys.argv[1]).write_text('ready'); "
        "time.sleep(5); sys.exit(87)"
    )

    def request_lease_end():
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if ready.exists():
            os.kill(os.getpid(), signal.SIGUSR1)

    with forwarding_lease_signals() as lease, (tmp_path / "child.log").open("w") as output:
        requester = threading.Thread(target=request_lease_end)
        requester.start()
        result = wait_engine_child([sys.executable, "-c", child_code, str(ready)], output, lease)
        requester.join(timeout=5)
        assert lease["requested"] and lease["signal"] == signal.SIGUSR1
        assert result.returncode == 75
        assert lease["child"] is None


def test_actual_hardware_receipt_uses_cuda_uuid_and_rejects_other_class(monkeypatch):
    import mm_dev.runtime as runtime_module

    properties = SimpleNamespace(
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        uuid="GPU-aabb",
        total_memory=96 << 30,
        major=12,
        minor=0,
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: properties)
    output = (
        "NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-ccdd, 01:00.0, 580.1, 96000\n"
        "NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-aabb, 02:00.0, 580.1, 96000\n"
    )
    monkeypatch.setattr(
        runtime_module.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(stdout=output)
    )
    identity = runtime_module.actual_cuda_identity()
    assert identity["uuid"] == "GPU-aabb" and identity["pci_bus_id"] == "02:00.0"
    assert identity["driver_version"] == "580.1"
    properties.name = "NVIDIA RTX 4090"
    with pytest.raises(PermissionError, match="audited RTX PRO 6000"):
        runtime_module.actual_cuda_identity()


def test_engine_keeps_completed_same_hardware_and_archives_changed_hardware(tmp_path, monkeypatch):
    from mm_dev.engine import prepare_engine_track

    directory = tmp_path / "engineering/natural/continuous"
    hardware = dict(uuid="GPU-original", driver_version="audited")
    atomic_json(
        directory / "CONTINUOUS_COMPLETE.json", dict(runtime_identity=dict(hardware=hardware))
    )
    monkeypatch.setenv("MM_DEV_ATTEMPT_ID", "attempt-after-preemption")
    prepare_engine_track(tmp_path, "natural", "continuous", hardware)
    assert directory.exists()
    prepare_engine_track(tmp_path, "natural", "continuous", {**hardware, "uuid": "GPU-next"})
    assert not directory.exists()
    archived = tmp_path / "engineering/interrupted/natural/continuous-attempt-after-preemption"
    assert (archived / "CONTINUOUS_COMPLETE.json").exists()


def test_audited_backend_flags_are_explicit_and_wrong_environment_rejected(monkeypatch):
    from mm_dev.runtime import configure_audited_backend

    original = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    try:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        with pytest.raises(PermissionError, match="CUBLAS"):
            configure_audited_backend()
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        result = configure_audited_backend()
        assert result == dict(
            deterministic_algorithms=True,
            deterministic_warn_only=False,
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
            cudnn_benchmark=False,
            cublas_workspace_config=":4096:8",
        )
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
        torch.backends.cuda.matmul.allow_tf32 = original[2]
        torch.backends.cudnn.allow_tf32 = original[3]
        torch.backends.cudnn.benchmark = original[4]


def test_parent_cache_release_collects_locals_before_releasing_allocator(monkeypatch):
    import mm_dev.engine as engine_module

    events = []
    monkeypatch.setattr(engine_module.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    engine_module.release_parent_cuda_cache()
    assert events == ["collect", "empty_cache"]
