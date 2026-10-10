"""Behavioral checks for SR-F1.2 full-sequence TIS training and recovery."""

import math

import pytest
import torch

from mm_core.training import capture_rng, state_hash, trainable_state
from sr_f12.training import (
    ProbabilityMismatch,
    checkpoint_state,
    commit_checkpoint,
    enforce_probability_gate,
    load_checkpoint,
    probability_diagnostics,
    select_microbatch,
    select_pilot_learning_rate,
    sequence_objective,
    update,
)


def test_detached_old_ratio_one_but_policy_gradient_nonzero():
    cur = torch.tensor([-2.0, -3.0], requires_grad=True)
    out = sequence_objective(cur, cur.detach(), cur.detach(), 3.0, kl_coefficient=0.0)
    assert torch.equal(out["ratio"], torch.ones(2))
    assert out["clip_fraction"].item() == 0
    out["loss"].backward()
    assert torch.equal(cur.grad, torch.full((2,), -1.5))


def test_tis_detached_and_capped_without_overflow():
    cur = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
    sampler = torch.tensor([-1.0 - math.log(4), -1.0 + math.log(2), -1000.0])
    out = sequence_objective(cur, sampler, cur.detach(), 1.0, kl_coefficient=0.0)
    torch.testing.assert_close(out["weights"], torch.tensor([2.0, 0.5, 2.0]))
    assert not out["weights"].requires_grad
    out["loss"].backward()
    torch.testing.assert_close(cur.grad, -torch.tensor([2.0, 0.5, 2.0]) / 3)


@pytest.mark.parametrize("length", [1, 3, 768])
def test_zero_advantage_zero_kl_has_exact_zero_gradient(length):
    cur = torch.linspace(-2, -3, length).requires_grad_()
    out = sequence_objective(cur, cur.detach() - 0.003, cur.detach(), 0.0, kl_coefficient=0.0)
    out["loss"].backward()
    assert torch.equal(cur.grad, torch.zeros_like(cur))


def test_reference_k3_gradient_and_sequence_mean():
    cur = torch.tensor([-1.0, -3.0], requires_grad=True)
    ref = torch.tensor([-1.1, -2.8])
    out = sequence_objective(cur, cur.detach(), ref, 0.0)
    out["loss"].backward()
    torch.testing.assert_close(cur.grad, 0.02 * (1 - (ref - cur.detach()).exp()) / 2)


@pytest.mark.parametrize("field", ["sampler", "reference", "advantage"])
def test_rejects_differentiable_detached_inputs(field):
    cur = torch.tensor([-2.0], requires_grad=True)
    sam, ref, adv = cur.detach(), cur.detach(), torch.tensor(1.0)
    if field == "sampler":
        sam = sam.clone().requires_grad_()
    if field == "reference":
        ref = ref.clone().requires_grad_()
    if field == "advantage":
        adv = adv.requires_grad_()
    with pytest.raises(ValueError):
        sequence_objective(cur, sam, ref, adv)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(bad):
    with pytest.raises(FloatingPointError):
        sequence_objective(
            torch.tensor([bad], requires_grad=True), torch.tensor([-2.0]), torch.tensor([-2.0]), 1.0
        )


class ToyRuntime:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(0.2)
        self.calls = []

    def batch_sequence_forward(self, records, purpose, grad):
        assert purpose == "training_policy" and grad
        self.calls.append(len(records))
        return [self.model.weight.reshape(()) * torch.tensor(r["features"]) - 2 for r in records]

    def batch_reference_forward(self, records, reference):
        return [reference["weight"].reshape(()) * torch.tensor(r["features"]) - 2 for r in records]


def fixture():
    runtime = ToyRuntime()
    opt = torch.optim.AdamW(runtime.model.parameters(), lr=5e-5, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    records = []
    for i in range(128):
        features = [float(i % 3 + 1)] * (1 + i % 4)
        sam = (0.2 * torch.tensor(features) - 2).tolist()
        records.append(
            dict(
                qid=f"q{i // 8}",
                features=features,
                tokens=[3] * len(features),
                sampler_logprobs=sam,
            )
        )
    return runtime, opt, scheduler, records, trainable_state(runtime.model)


@pytest.mark.parametrize("size", [4, 2, 1])
def test_update_equal_sequence_weights_single_adam_and_no_diagnostic_backward(size):
    runtime, opt, sch, records, ref = fixture()
    # Alternating lengths cannot change the answer-level normalization.
    coefficients = [(i % 5 - 2) / 10 for i in range(128)]
    expected = -sum(a * r["features"][0] for a, r in zip(coefficients, records, strict=True)) / 128
    out = update(
        runtime,
        opt,
        sch,
        records,
        coefficients,
        reference=ref,
        microbatch_size=size,
        save_train_logprobs=True,
    )
    assert out["backward_calls"] == 128 // size
    assert out["optimizer_updates"] == 1
    assert runtime.calls == [size] * (128 // size)
    assert out["total_gradient_norm"] == pytest.approx(abs(expected), abs=1e-7)
    assert len(out["train_logprobs"]) == 128
    assert [len(x) for x in out["train_logprobs"]] == [len(r["tokens"]) for r in records]
    assert out["clip_fraction"] == 0
    assert out["sampler_train_mean_absolute_difference"] == 0
    assert out["parameters_changed"] and out["adam_nonzero_moments"]
    assert int(next(iter(opt.state.values()))["step"]) == 1
    assert sch.last_epoch == 1


def test_microbatch_choices_produce_same_optimizer_update():
    outputs = []
    for size in (4, 2, 1):
        runtime, opt, sch, records, ref = fixture()
        update(runtime, opt, sch, records, [1.0] * 128, reference=ref, microbatch_size=size)
        outputs.append(trainable_state(runtime.model)["weight"])
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])


def test_alignment_failure_is_before_optimizer_and_clears_grads():
    runtime, opt, sch, records, ref = fixture()
    for record in records:
        record["sampler_logprobs"] = [v - 0.02 for v in record["sampler_logprobs"]]
    original = state_hash(trainable_state(runtime.model))
    with pytest.raises(ProbabilityMismatch) as exc:
        update(runtime, opt, sch, records, [1.0] * 128, reference=ref)
    assert exc.value.diagnostics["sampler_train_mean_absolute_difference"] > 0.01
    assert state_hash(trainable_state(runtime.model)) == original
    assert not opt.state and sch.last_epoch == 0
    assert all(p.grad is None for p in runtime.model.parameters())


def test_probability_statistics_are_token_weighted_not_response_weighted():
    out = probability_diagnostics(
        [torch.tensor([0.02]), torch.zeros(3)], [torch.tensor([False]), torch.zeros(3).bool()]
    )
    assert out["sampler_train_mean_absolute_difference"] == pytest.approx(0.005)
    enforce_probability_gate(out)
    with pytest.raises(ProbabilityMismatch):
        enforce_probability_gate({**out, "tis_truncated_fraction": 0.010001})


@pytest.mark.parametrize("bad", ["short", "qid", "sampler", "microbatch"])
def test_update_rejects_incomplete_or_ambiguous_inputs(bad):
    runtime, opt, sch, records, ref = fixture()
    if bad == "short":
        records = records[:-1]
    if bad == "qid":
        records[1]["qid"] = "different"
    if bad == "sampler":
        records[1].pop("sampler_logprobs")
    with pytest.raises(ValueError):
        update(
            runtime,
            opt,
            sch,
            records,
            [1.0] * 128,
            reference=ref,
            microbatch_size=3 if bad == "microbatch" else 4,
        )
    assert not opt.state


def test_microbatch_only_oom_falls_back_and_checkpointing_last():
    attempts = []

    def probe(size, checkpointing):
        attempts.append((size, checkpointing))
        if not checkpointing:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return {"memory_fit": True}

    result = select_microbatch(probe)
    assert attempts == [(4, False), (2, False), (1, False), (1, True)]
    assert result["microbatch_size"] == 1 and result["gradient_checkpointing"]
    with pytest.raises(ValueError):
        select_microbatch(lambda *_: (_ for _ in ()).throw(ValueError("bug")))


def metrics():
    return [
        dict(
            step=i, finite=True, loss=0.0, kl=0.01, total_gradient_norm=1.0, format_failure_rate=0.1
        )
        for i in range(1, 9)
    ]


@pytest.mark.parametrize(
    "movement,rate,status", [(0.009, 1e-4, "RETRY_ONCE"), (0.01, 5e-5, "PASS"), (0.1, 5e-5, "PASS")]
)
def test_pilot_movement_rule(movement, rate, status):
    decision = select_pilot_learning_rate(metrics(), movement)
    assert decision["learning_rate"] == rate and decision["status"] == status
    assert not decision["score_metrics_used"]


@pytest.mark.parametrize("failure", ["nonfinite", "format", "kl"])
def test_pilot_instability_has_priority_over_small_movement(failure):
    rows = metrics()
    if failure == "nonfinite":
        rows[2]["loss"] = float("nan")
    if failure == "format":
        for row in rows[4:]:
            row["format_failure_rate"] = 0.20001
    if failure == "kl":
        rows[-1]["kl"] = 0.050001
    assert select_pilot_learning_rate(rows, 0.001)["learning_rate"] == 2e-5
    assert (
        select_pilot_learning_rate(rows, 0.001, attempt=1, learning_rate=2e-5)["status"]
        == "STOP_UNSTABLE"
    )


def test_adjusted_pilot_does_not_readjust_on_small_movement():
    decision = select_pilot_learning_rate(metrics(), 0.0, attempt=1, learning_rate=2e-5)
    assert decision["status"] == "PASS" and decision["learning_rate"] == 2e-5
    with pytest.raises(ValueError):
        select_pilot_learning_rate(metrics(), 0.0, attempt=2)


def test_nonfinite_early_pilot_can_choose_lower_lr_without_eight_completed_steps():
    decision = select_pilot_learning_rate([dict(step=1, finite=False)], None)
    assert decision["learning_rate"] == 2e-5 and decision["status"] == "RETRY_ONCE"


def test_resume_restores_actual_adam_scheduler_cursor_parameters_and_rng(tmp_path):
    runtime, opt, sch, records, ref = fixture()
    for _ in range(4):
        # sampler probabilities reflect this newly sampled policy.
        for record in records:
            record["sampler_logprobs"] = (
                runtime.model.weight.item() * torch.tensor(record["features"]) - 2
            ).tolist()
        update(runtime, opt, sch, records, [1.0] * 128, reference=ref)
    kwargs = dict(
        run_identity={"run": "GATE-pilot"},
        reference=ref,
        stream_hash="stream",
        sampling_hash="sampling",
        microbatch_size=4,
    )
    state = checkpoint_state(runtime, opt, sch, step=4, **kwargs)
    commit_checkpoint(tmp_path, state)
    runtime.model.weight.data.add_(1)
    torch.rand(10)
    restored = load_checkpoint(tmp_path, runtime, opt, sch, learning_rate=5e-5, **kwargs)
    assert restored["cursor"]["next_logical_step"] == 5
    assert state_hash(trainable_state(runtime.model)) == state_hash(state["parameters"])
    assert state_hash(capture_rng()) == state_hash(state["rng"])
    assert int(next(iter(opt.state.values()))["step"]) == 4
    assert sch.last_epoch == 4
    with pytest.raises(PermissionError):
        load_checkpoint(
            tmp_path, runtime, opt, sch, learning_rate=5e-5, **{**kwargs, "microbatch_size": 2}
        )
    path = tmp_path / __import__("json").loads((tmp_path / "LATEST.json").read_text())["path"]
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(PermissionError):
        load_checkpoint(tmp_path, runtime, opt, sch, learning_rate=5e-5, **kwargs)


@pytest.mark.parametrize("arm", ["A", "J", "PART", "DEC", "GATE"])
def test_all_five_unchanged_reward_advantages_share_same_loss(arm):
    from sr_f12.training import reward_advantages

    runtime, opt, sch, records, ref = fixture()
    scores = [
        [
            dict(
                A=int(i % 3 == 0),
                J=int(i == 0),
                E=int(i < 6),
                p_read=1.0 if i == 0 else (i % 5) / 5.0,
            )
            for i in range(8)
        ]
        for _ in range(16)
    ]
    coefficients, _ = reward_advantages(arm, scores)
    coefficients = coefficients.reshape(-1)
    expected = (
        -sum(float(a) * r["features"][0] for a, r in zip(coefficients, records, strict=True)) / 128
    )
    out = update(runtime, opt, sch, records, coefficients, reference=ref)
    assert out["total_gradient_norm"] == pytest.approx(abs(expected), abs=1e-7)
    assert out["finite"]


def test_first_batch_movement_is_token_weighted_and_uses_same_128_sequences():
    from sr_f12.training import first_batch_movement

    before = [[-2.0]] * 128
    after = [[-2.0]] * 127 + [[-1.0]]
    assert first_batch_movement(before, after) == pytest.approx(1 / 128)
    with pytest.raises(ValueError):
        first_batch_movement(before[:-1], after[:-1])
    with pytest.raises(ValueError):
        first_batch_movement(before, [*after[:-1], [-1.0, -2.0]])


def test_pilot_thresholds_are_strict():
    rows = metrics()
    for row in rows[4:]:
        row["format_failure_rate"] = 0.2
    rows[-1]["kl"] = 0.05
    assert select_pilot_learning_rate(rows, 0.01)["status"] == "PASS"


def test_checkpoint_refuses_nonfinite_adam_moments():
    runtime, opt, sch, records, ref = fixture()
    update(runtime, opt, sch, records, [1.0] * 128, reference=ref)
    next(iter(opt.state.values()))["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        checkpoint_state(
            runtime,
            opt,
            sch,
            run_identity={"run": "pilot"},
            step=1,
            reference=ref,
            stream_hash="s",
            sampling_hash="p",
            microbatch_size=4,
        )
