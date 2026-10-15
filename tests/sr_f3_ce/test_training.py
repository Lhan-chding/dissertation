"""CPU numerical/transaction tests, never a claim of real-model GPU acceptance."""

import copy
import random

import numpy as np
import pytest

from mm_core.training import capture_rng, state_hash, trainable_state
from sr_f3_ce.training import (
    ProbabilityMismatch,
    fresh_optimizer,
    group_advantages,
    masked_mean_nll,
    probe_gradients,
    sequence_objective,
    update,
)

torch = pytest.importorskip("torch")


class ToyRuntime:
    """Small differentiable sequence model with explicit token-dependent lengths."""

    def __init__(self, theta=None):
        self.model = torch.nn.Module()
        self.model.register_parameter(
            "theta", torch.nn.Parameter(torch.tensor(theta or [0.13, -0.07, 0.02]))
        )
        self.support_calls = 0
        self.task_calls = 0
        self.observed_parameters = []
        self.oom_at_call = None
        self.consume_rng = False

    def logp(self, record, theta=None):
        values = torch.tensor(record["tokens"], dtype=torch.float32)
        features = torch.stack((torch.ones_like(values), values / 20, (values % 3) / 5), -1)
        return torch.nn.functional.logsigmoid(
            features @ (self.model.theta if theta is None else theta)
        )

    def batch_sequence_forward(self, records, *, purpose, grad):
        assert purpose == "training_policy" and grad
        self.task_calls += 1
        self.observed_parameters.append(self.model.theta.detach().clone())
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(2)
        if self.task_calls == self.oom_at_call:
            raise torch.cuda.OutOfMemoryError("injected CPU test of CUDA OOM rollback")
        return [self.logp(record) for record in records]

    def batch_reference_forward(self, records, reference):
        return [self.logp(record, reference["theta"]).detach() for record in records]

    def batch_support_forward(self, records, *, purpose, grad):
        assert purpose == "training_support" and grad
        self.support_calls += 1
        self.observed_parameters.append(self.model.theta.detach().clone())
        return [self.logp(record) for record in records]


def task_records(runtime, *, offset=0, constant=None):
    records = []
    for group in range(16):
        for sample in range(8):
            record = {
                "qid": f"task-{offset + group}",
                "tokens": [sample + 1, group % 5 + 1] * (1 + sample % 3),
                "reward": int(sample % 3 == 0) if constant is None else constant,
            }
            record["sampler_logprobs"] = runtime.logp(record).detach().tolist()
            records.append(record)
    return records


def supports(*, arm="EXEC", offset=0):
    return [
        {
            "qid": f"support-{offset + index}",
            "tokens": [1, 3 + index, 5, 9, 11, 13][: 5 + index % 2],
            "answer_mask": [False, False, False, True, True, True][: 5 + index % 2],
            "work_mask": [False, True, True, False, False, False][: 5 + index % 2]
            if arm != "GOLD"
            else [False] * (5 + index % 2),
        }
        for index in range(4)
    ]


def model_state(runtime, optimizer, scheduler):
    return state_hash(
        {
            "parameters": trainable_state(runtime.model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng(),
        }
    )


def test_binary_population_advantages_keep_constant_groups():
    rewards = np.array([[0] * 8, [1] * 8] + [[0, 0, 0, 0, 0, 0, 0, 1]] * 14)
    actual = group_advantages(rewards)
    assert np.array_equal(actual[:2], np.zeros((2, 8)))
    assert np.allclose(actual[2], (rewards[2] - rewards[2].mean()) / (rewards[2].std() + 1e-8))
    assert actual[2, -1] == pytest.approx(np.sqrt(7), rel=1e-7)
    with pytest.raises(ValueError, match="binary"):
        group_advantages(np.full((16, 8), 0.5))
    with pytest.raises(ValueError):
        group_advantages(np.zeros((16, 7)))


@pytest.mark.parametrize("arm", ["RL", "GOLD", "PREFIX", "EXEC"])
def test_combined_update_matches_independent_full_batch_equation(arm):
    actual, expected = ToyRuntime(), ToyRuntime()
    optimizer, scheduler = fresh_optimizer(actual.model)
    expected_optimizer, expected_scheduler = fresh_optimizer(expected.model)
    reference = trainable_state(actual.model)
    records = task_records(actual)
    support = None if arm == "RL" else supports(arm=arm)
    advantages = group_advantages(np.array([r["reward"] for r in records]).reshape(16, 8)).ravel()
    terms = []
    for record, advantage in zip(records, advantages, strict=True):
        terms.append(
            sequence_objective(
                expected.logp(record),
                torch.tensor(record["sampler_logprobs"]),
                expected.logp(record, reference["theta"]).detach(),
                advantage,
            )["loss"]
        )
    task_loss = torch.stack(terms).mean()
    auxiliary = torch.tensor(0.0)
    if support is not None:
        auxiliary_terms = []
        for record in support:
            logp = expected.logp(record)
            answer = masked_mean_nll(logp, record["answer_mask"])
            auxiliary_terms.append(
                answer
                if arm != "EXEC"
                else 0.5 * answer + 0.5 * masked_mean_nll(logp, record["work_mask"])
            )
        auxiliary = torch.stack(auxiliary_terms).mean()
    loss = task_loss + 0.1 * auxiliary
    loss.backward()
    expected_grad_norm = float(torch.nn.utils.clip_grad_norm_(expected.model.parameters(), 1.0))
    expected_optimizer.step()
    expected_scheduler.step()
    result = update(
        actual,
        optimizer,
        scheduler,
        records,
        arm=arm,
        reference=reference,
        support_records=support,
        step=1,
    )
    # Independent fp32 reduction trees differ by a few rounding units near zero.
    assert result["loss"] == pytest.approx(float(loss.detach()), abs=4 * np.finfo(np.float32).eps)
    assert result["gradient_norm_before_clip"] == pytest.approx(expected_grad_norm, rel=2e-6)
    assert torch.allclose(actual.model.theta, expected.model.theta, atol=1e-8, rtol=0)
    assert result["total_cost"]["global_clip_calls"] == 1
    assert result["total_cost"]["optimizer_step_calls_attempted"] == 1
    assert result["optimizer_steps"] == [1]
    assert all(torch.equal(theta, reference["theta"]) for theta in actual.observed_parameters)
    assert actual.support_calls == (0 if arm == "RL" else 1)
    assert result["gradient_diagnostics"]["unweighted_norms"]["policy"] > 0
    if arm != "RL":
        assert result["gradient_diagnostics"]["unweighted_norms"]["aux_ce"] > 0
    if arm == "PREFIX":
        assert all(row["supervised_work_tokens"] == 0 for row in result["support_losses"])
        assert all(row["work_ce"] > 0 for row in result["support_losses"])


def test_microbatch_reduction_is_sequence_equal_and_diagnostics_do_not_change_update():
    states, losses = [], []
    for size in (4, 2, 1):
        runtime = ToyRuntime()
        optimizer, scheduler = fresh_optimizer(runtime.model)
        result = update(
            runtime,
            optimizer,
            scheduler,
            task_records(runtime),
            arm="EXEC",
            reference=trainable_state(runtime.model),
            support_records=supports(),
            step=1,
            microbatch_size=size,
        )
        states.append(runtime.model.theta.detach().clone())
        losses.append(result["loss"])
    assert losses == pytest.approx([losses[0]] * 3, abs=2e-8)
    assert all(torch.equal(state, states[0]) for state in states)
    assert supports(arm="PREFIX") == supports(arm="EXEC")


def test_rl_rejects_hidden_support_and_zero_ce_engine_matches_rl():
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    with pytest.raises(PermissionError, match="no support"):
        update(
            runtime,
            optimizer,
            scheduler,
            task_records(runtime),
            arm="RL",
            reference=trainable_state(runtime.model),
            support_records=[],
            step=1,
        )
    snapshots = []
    for arm in ("RL", "EXEC"):
        runtime = ToyRuntime()
        optimizer, scheduler = fresh_optimizer(runtime.model)
        result = update(
            runtime,
            optimizer,
            scheduler,
            task_records(runtime),
            arm=arm,
            reference=trainable_state(runtime.model),
            support_records=None if arm == "RL" else supports(),
            step=1,
            engineering_zero_aux=arm == "EXEC",
        )
        snapshots.append((trainable_state(runtime.model), optimizer.state_dict(), result))
    assert state_hash(snapshots[0][:2]) == state_hash(snapshots[1][:2])
    assert snapshots[1][2]["scientific_recipe_eligible"] is False


def test_gradient_probes_use_no_physical_updates_and_preserve_complete_state():
    runtime = ToyRuntime()
    runtime.consume_rng = True
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, reference = task_records(runtime), trainable_state(runtime.model)
    before = model_state(runtime, optimizer, scheduler)
    pure = probe_gradients(runtime, optimizer, scheduler, records, arm="RL", reference=reference)
    zero_ce = probe_gradients(
        runtime,
        optimizer,
        scheduler,
        records,
        arm="EXEC",
        reference=reference,
        support_records=supports(),
        engineering_zero_aux=True,
    )
    assert pure["unclipped_gradient_hash"] == zero_ce["unclipped_gradient_hash"]
    assert model_state(runtime, optimizer, scheduler) == before
    assert zero_ce["total_cost"]["optimizer_step_calls_attempted"] == 0
    assert zero_ce["total_cost"]["global_clip_calls"] == 0
    assert zero_ce["optimizer_updates"] == 0
    assert zero_ce["gradient_diagnostics"]["unweighted_norms"]["aux_ce"] > 0
    for size in (4, 2, 1):
        result = probe_gradients(
            runtime,
            optimizer,
            scheduler,
            records,
            arm="EXEC",
            reference=reference,
            support_records=supports(),
            microbatch_size=size,
        )
        assert result["loss"] == pytest.approx(0.06405912339687347, abs=1e-7)
        assert model_state(runtime, optimizer, scheduler) == before


def test_constant_reward_group_keeps_kl_and_all_128_sequences():
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = {"theta": runtime.model.theta.detach().clone() + 0.05}
    result = update(
        runtime,
        optimizer,
        scheduler,
        task_records(runtime, constant=0),
        arm="RL",
        reference=reference,
        step=1,
    )
    assert result["constant_group_fraction"] == 1.0
    assert result["sequence_count"] == 128
    assert result["policy"] == 0.0
    assert result["kl"] > 0.0
    assert result["parameters_changed"]
    assert result["gradient_diagnostics"]["angles"]["policy_vs_kl"]["degrees"] is None


@pytest.mark.parametrize("error", ["mean", "cap", "nan", "empty_mask"])
def test_failure_gates_restore_every_live_state(error):
    runtime = ToyRuntime()
    runtime.consume_rng = True
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, support = task_records(runtime), supports()
    if error == "mean":
        for record in records:
            record["sampler_logprobs"] = [value + 0.011 for value in record["sampler_logprobs"]]
    elif error == "cap":
        # >1% cap hits with pooled mean discrepancy below 0.01 nat.
        for record in records[:2]:
            record["sampler_logprobs"] = [value - 0.7 for value in record["sampler_logprobs"]]
    elif error == "nan":
        records[0]["sampler_logprobs"][0] = float("nan")
    else:
        support[0]["answer_mask"] = [False] * len(support[0]["tokens"])
    before = model_state(runtime, optimizer, scheduler)
    expected = (
        ValueError
        if error == "empty_mask"
        else (ProbabilityMismatch if error in ("mean", "cap") else FloatingPointError)
    )
    with pytest.raises(expected):
        update(
            runtime,
            optimizer,
            scheduler,
            records,
            arm="EXEC",
            reference=trainable_state(runtime.model),
            support_records=support,
            step=1,
        )
    assert model_state(runtime, optimizer, scheduler) == before
    assert runtime.support_calls == 0


def test_cuda_oom_restarts_whole_batch_and_preserves_rng_and_cost():
    runtime, control = ToyRuntime(), ToyRuntime()
    runtime.consume_rng = True
    runtime.oom_at_call = 3
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    records, support = task_records(runtime), supports()
    rng_before = state_hash(capture_rng())
    incidents = []
    result = update(
        runtime,
        optimizer,
        scheduler,
        records,
        arm="EXEC",
        reference=reference,
        support_records=support,
        step=1,
        incident_callback=incidents.append,
    )
    assert result["microbatch_size"] == 2
    assert [attempt["status"] for attempt in result["attempts"]] == [
        "CUDA_OOM_ROLLED_BACK",
        "APPLIED",
    ]
    assert result["attempts"][0]["parameters_optimizer_scheduler_rng_restored"]
    assert len(incidents) == 1
    assert state_hash(capture_rng()) == rng_before
    assert result["total_cost"]["task_policy_sequences_attempted"] == 128 + 12
    control_optimizer, control_scheduler = fresh_optimizer(control.model)
    update(
        control,
        control_optimizer,
        control_scheduler,
        records,
        arm="EXEC",
        reference=reference,
        support_records=support,
        step=1,
        microbatch_size=2,
    )
    assert state_hash(trainable_state(runtime.model)) == state_hash(trainable_state(control.model))
    assert state_hash(optimizer.state_dict()) == state_hash(control_optimizer.state_dict())


def test_oom_after_adam_rolls_back_moments_before_retry(monkeypatch):
    runtime, control = ToyRuntime(), ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    control_optimizer, control_scheduler = fresh_optimizer(control.model)
    records, reference = task_records(runtime), trainable_state(runtime.model)
    original_step = optimizer.step
    called = False

    def step_then_oom(*args, **kwargs):
        nonlocal called
        original_step(*args, **kwargs)
        if not called:
            called = True
            raise torch.cuda.OutOfMemoryError("test partial Adam application")

    monkeypatch.setattr(optimizer, "step", step_then_oom)
    step_then_oom._wrapped_by_lr_sched = True
    result = update(runtime, optimizer, scheduler, records, arm="RL", reference=reference, step=1)
    update(
        control,
        control_optimizer,
        control_scheduler,
        records,
        arm="RL",
        reference=reference,
        step=1,
        microbatch_size=2,
    )
    assert result["total_cost"]["optimizer_step_calls_attempted"] == 2
    assert result["optimizer_steps"] == [1]
    assert state_hash(trainable_state(runtime.model)) == state_hash(trainable_state(control.model))
    assert state_hash(optimizer.state_dict()) == state_hash(control_optimizer.state_dict())


def test_single_sequence_oom_stops_without_auto_checkpointing():
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    runtime.oom_at_call = 1
    before = model_state(runtime, optimizer, scheduler)
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        update(
            runtime,
            optimizer,
            scheduler,
            task_records(runtime),
            arm="RL",
            reference=trainable_state(runtime.model),
            step=1,
            microbatch_size=1,
        )
    assert caught.value.training_attempts[0]["microbatch_size"] == 1
    assert model_state(runtime, optimizer, scheduler) == before


def test_cached_score_mismatch_and_repeated_logical_update_are_rejected():
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records = task_records(runtime)
    reference = trainable_state(runtime.model)
    records[0]["score"] = {"J": 1 - records[0]["reward"]}
    with pytest.raises(ValueError, match="verified J"):
        update(runtime, optimizer, scheduler, records, arm="RL", reference=reference, step=1)
    records[0].pop("score")
    update(runtime, optimizer, scheduler, records, arm="RL", reference=reference, step=1)
    before = model_state(runtime, optimizer, scheduler)
    with pytest.raises(PermissionError, match="cursor"):
        update(runtime, optimizer, scheduler, records, arm="RL", reference=reference, step=1)
    assert model_state(runtime, optimizer, scheduler) == before


def test_unregistered_learning_rate_and_cross_field_masks_rejected():
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    optimizer.param_groups[0]["lr"] = 5e-5
    with pytest.raises(PermissionError, match="recipe"):
        update(
            runtime,
            optimizer,
            scheduler,
            task_records(runtime),
            arm="RL",
            reference=trainable_state(runtime.model),
            step=1,
        )
    optimizer, scheduler = fresh_optimizer(runtime.model)
    bad = copy.deepcopy(supports())
    bad[0]["work_mask"][3] = True
    with pytest.raises(ValueError, match="overlap"):
        update(
            runtime,
            optimizer,
            scheduler,
            task_records(runtime),
            arm="EXEC",
            reference=trainable_state(runtime.model),
            support_records=bad,
            step=1,
        )


def test_optimizer_intent_precedes_mutation_and_budget_rejection_rolls_back():
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    records = task_records(runtime)
    before = model_state(runtime, optimizer, scheduler)
    intents = []

    def deny(metadata):
        assert optimizer.state == {}
        assert state_hash(trainable_state(runtime.model)) == state_hash(reference)
        intents.append(metadata)
        raise PermissionError("ENGINE physical update budget exhausted")

    with pytest.raises(PermissionError, match="budget"):
        update(
            runtime,
            optimizer,
            scheduler,
            records,
            arm="RL",
            reference=reference,
            step=1,
            optimizer_step_callback=deny,
        )
    assert len(intents) == 1 and intents[0]["step"] == 1
    assert model_state(runtime, optimizer, scheduler) == before
    update(
        runtime,
        optimizer,
        scheduler,
        records,
        arm="RL",
        reference=reference,
        step=1,
        optimizer_step_callback=intents.append,
    )
    assert len(intents) == 2
    assert int(next(iter(optimizer.state.values()))["step"]) == 1
