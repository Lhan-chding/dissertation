"""Actual CPU autograd/recovery checks; these do not qualify the 9B GPU backend."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from mm_core.training import state_hash, trainable_state
from mm_core.vl_runtime import seed_all
from sr_f1.contract import PLAN_ID, digest, reward_advantages
from sr_f1.runtime import reference_parameters
from sr_f1.training import (
    checkpoint_state,
    commit_checkpoint,
    load_checkpoint,
    sequence_objective,
    update,
)


class TinyRuntime:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[0.1, -0.2], [0.3, 0.1]]))
        self.events = []

    def prepare(self, row, root):
        return dict(x=torch.tensor(row["x"]))

    def reserve(self, kind, count, **metadata):
        self.events.append((kind, count, metadata))

    def sequence_forward(self, prepared, tokens, *, purpose, grad=False):
        with torch.enable_grad() if grad else torch.no_grad():
            lp = self.model(prepared["x"]).log_softmax(-1)[torch.tensor(tokens)]
        return dict(logprobs=lp)

    def reference_forward(self, prepared, tokens, reference):
        with reference_parameters(self.model, reference):
            return self.sequence_forward(prepared, tokens, purpose="reference")


def optimizer_for(runtime):
    opt = torch.optim.AdamW(
        runtime.model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)


def examples(runtime, coefficients=None):
    result = []
    for i in range(128):
        row = dict(x=[1.0, 0.3 + (i % 3) * 0.1])
        tokens = [i % 2] * (1 + i % 3)
        old = runtime.sequence_forward(runtime.prepare(row, None), tokens, purpose="sampler")
        result.append(
            dict(
                row=row,
                root=None,
                advantage=float(coefficients[i]) if coefficients is not None else (-1.0) ** i,
                record=dict(tokens=tokens, old_logprobs=old["logprobs"].tolist()),
            )
        )
    return result


def scores():
    return [
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


@pytest.mark.parametrize("arm", ["A", "J", "PART", "DEC", "GATE"])
def test_full_128_actual_gradient_matches_vectorized_loss_for_all_final_advantages(arm):
    runtime = TinyRuntime()
    ref = trainable_state(runtime.model)
    coefficients, _ = reward_advantages(arm, scores())
    coefficients = coefficients.reshape(-1)
    batch = examples(runtime, coefficients)
    # Vectorized, padded token tensor and explicit completion mask. EOS-like last
    # tokens participate; prompt and padding positions do not exist in the loss.
    xs = torch.tensor([b["row"]["x"] for b in batch])
    logits = runtime.model(xs).log_softmax(-1)
    tokens = torch.zeros(128, 3, dtype=torch.long)
    mask = torch.zeros(128, 3)
    for i, b in enumerate(batch):
        n = len(b["record"]["tokens"])
        tokens[i, :n] = torch.tensor(b["record"]["tokens"])
        mask[i, :n] = 1
    current = logits.gather(1, tokens)
    old = current.detach().clone()
    adv = torch.tensor(coefficients, dtype=torch.float32)[:, None]
    ratio = (current - old).exp()
    policy = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
    expected_loss = ((policy * mask).sum(1) / mask.sum(1)).double().mean()
    expected_grad = torch.autograd.grad(expected_loss, runtime.model.weight)[0]
    opt, scheduler = optimizer_for(runtime)
    result = update(runtime, opt, scheduler, ref, batch, probability_gate=True)
    assert result["effective_sequences"] == 128
    assert result["final_advantages"] == coefficients.tolist()
    assert result["policy"] == pytest.approx(float(expected_loss.detach()), abs=2e-8)
    assert result["policy_gradient_norm"] == pytest.approx(float(expected_grad.norm()), abs=2e-7)
    # Adam stores the unclipped gradient when its norm is below one.
    actual_grad = opt.state[runtime.model.weight]["exp_avg"] / 0.1
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-6)
    assert sum(n for kind, n, _ in runtime.events if kind == "physical_optimizer_updates") == 1
    with pytest.raises(ValueError, match="16 x 8"):
        update(runtime, opt, scheduler, ref, batch[:-1])


def test_final_advantage_must_be_detached_and_nonfinite_ratio_is_not_clipped_away():
    lp = torch.tensor([-0.2, -0.5], requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        sequence_objective(lp, lp, lp.detach(), 1.0)
    with pytest.raises(ValueError, match="detached"):
        sequence_objective(lp, lp.detach(), lp.detach(), torch.tensor(1.0, requires_grad=True))
    with pytest.raises(FloatingPointError):
        sequence_objective(
            torch.tensor([1000.0], requires_grad=True),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            1.0,
        )


def test_reference_switch_keeps_optimizer_parameter_objects_and_restores_on_exception():
    runtime = TinyRuntime()
    optimizer, _ = optimizer_for(runtime)
    original = trainable_state(runtime.model)
    ids = [id(p) for p in optimizer.param_groups[0]["params"]]
    with (
        pytest.raises(RuntimeError),
        reference_parameters(runtime.model, {n: torch.zeros_like(p) for n, p in original.items()}),
    ):
        assert runtime.model.weight.count_nonzero() == 0
        raise RuntimeError("fixture")
    assert ids == [id(p) for p in runtime.model.parameters()]
    assert state_hash(original) == state_hash(trainable_state(runtime.model))


def test_zero_policy_advantage_can_still_have_kl_or_adam_update():
    runtime = TinyRuntime()
    optimizer, scheduler = optimizer_for(runtime)
    reference = {n: p + 0.2 * torch.eye(2) for n, p in trainable_state(runtime.model).items()}
    result = update(runtime, optimizer, scheduler, reference, examples(runtime, [0.0] * 128))
    assert result["policy_gradient_norm"] == 0
    assert result["total_gradient_norm_before_clip"] > 0
    assert result["parameter_changed"]


def test_real_adam_4_versus_2_plus_2_exact_state_rng_and_next_tokens(tmp_path):
    def path(directory, stop, resume=False):
        runtime = TinyRuntime()
        opt, scheduler = optimizer_for(runtime)
        ref = trainable_state(runtime.model)
        identity = dict(plan_id=PLAN_ID, fixture="CPU_actual_SR_F1_loss")
        if resume:
            state = load_checkpoint(
                directory,
                runtime,
                opt,
                scheduler,
                run_identity=identity,
                reference=ref,
                stream_hash="frozen",
                sampling_hash="raw",
                step=2,
            )
            start = state["committed_logical_step"]
        else:
            seed_all(99)
            start = 0
        for step in range(start + 1, stop + 1):
            coeff, _ = reward_advantages(("DEC", "GATE", "DEC", "GATE")[step - 1], scores())
            result = update(runtime, opt, scheduler, ref, examples(runtime, coeff.reshape(-1)))
            torch.rand(5)
            state = checkpoint_state(
                runtime,
                opt,
                scheduler,
                run_identity=identity,
                step=step,
                reference=ref,
                stream_hash="frozen",
                sampling_hash="raw",
                diagnostics_hash=digest(result),
                token_path_hash="fixture",
            )
            commit_checkpoint(directory, state)
        return state, torch.multinomial(torch.ones(7), 32, replacement=True)

    continuous, next_a = path(tmp_path / "continuous", 4)
    path(tmp_path / "split", 2)
    torch.rand(777)
    resumed, next_b = path(tmp_path / "split", 4, resume=True)
    assert state_hash(continuous) == state_hash(resumed)
    assert torch.equal(next_a, next_b)
    assert all(v["exp_avg"].count_nonzero() for v in resumed["optimizer"]["state"].values())
    changed = copy.deepcopy(resumed)
    changed["run_identity"]["fixture"] = "tampered"
    with pytest.raises(PermissionError):
        load_checkpoint(
            tmp_path / "split",
            TinyRuntime(),
            *optimizer_for(TinyRuntime()),
            run_identity=changed["run_identity"],
            reference=resumed["reference"],
            stream_hash="frozen",
            sampling_hash="raw",
        )


def test_checkpoint_preserves_all_milestones_through_96(tmp_path):
    runtime = TinyRuntime()
    opt, scheduler = optimizer_for(runtime)
    ref = trainable_state(runtime.model)
    for step in (0, 8, 32, 64, 96):
        state = checkpoint_state(
            runtime,
            opt,
            scheduler,
            run_identity={"run_id": "fixture"},
            step=step,
            reference=ref,
            stream_hash="s",
            sampling_hash="p",
        )
        commit_checkpoint(tmp_path, state)
    assert all((tmp_path / f"commit-{step:02d}.json").exists() for step in (0, 8, 32, 64, 96))
    assert json.loads((tmp_path / "LATEST.json").read_text())["step"] == 96


def test_scientific_crash_before_eight_step_commit_replays_identical_slots_without_resampling(
    tmp_path, monkeypatch
):
    """A process crash after update 3 restores step 0, reuses slots, and repeats Adam."""
    from types import SimpleNamespace

    from mm_core.vl_runtime import hash_json
    from sr_f1 import training

    class SamplingRuntime(TinyRuntime):
        generated = 0

        def __init__(self):
            super().__init__()
            self.identity = {"trainable_state_hash": state_hash(trainable_state(self.model))}

        def training_generation_config(self):
            return SimpleNamespace(to_dict=lambda: {"fixture": "raw_t1"})

        def generate_training(self, row, root, seed, on_completion):
            SamplingRuntime.generated += 1
            tokens = [seed % 2] * (1 + seed % 3)
            old = self.sequence_forward(self.prepare(row, root), tokens, purpose="sampler")[
                "logprobs"
            ].tolist()
            raw = json.dumps({"evidence": [], "answer": 4 if seed % 2 else 0})
            on_completion(
                dict(
                    seed=seed,
                    tokens=tokens,
                    raw_text=raw,
                    old_logprobs=old,
                    sampler_logprobs=old,
                    sampling_hash=hash_json({"fixture": "raw_t1"}),
                    generation_status="COMPLETE",
                    truncated=False,
                    image_routing={"fixture": True},
                )
            )

    def config(runtime):
        opt, sched = optimizer_for(runtime)
        return opt, sched, {"fixture": "CPU"}

    monkeypatch.setattr(training, "configure_training", config)
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "images/chart.png").write_bytes(b"fixture-image-identity")
    questions = {
        f"q{i}": dict(
            qid=f"q{i}", image_file="images/chart.png", text="query", x=[1.0, 0.3 + (i % 3) * 0.1]
        )
        for i in range(128)
    }
    tasks = {
        q: dict(
            root_id=q,
            world={
                "categories": ["January", "February", "March", "April"],
                "series": {"Alpha": [60, 70, 80, 90], "Beta": [10, 20, 30, 40]},
            },
            query={"family": "THRESHOLD", "predicate": "ge", "threshold": 50, "op": "count"},
        )
        for q in questions
    }
    # Use the authenticated AST factory rather than duplicating query field names.
    from sr_f1.contract import reference_module

    query = reference_module("semantic_contract").base_query("THRESHOLD")
    for task in tasks.values():
        task["query"] = query
    schedule = [
        dict(
            step=step,
            slot=slot,
            qid=f"q{(step - 1) * 16 + slot}",
            rollout_seeds=[step * 10000 + slot * 8 + i for i in range(8)],
        )
        for step in range(1, 9)
        for slot in range(16)
    ]
    run = {"run_id": "CPU_REPLAY", "paired_seed": 123, "H": 8, "arm": "A"}
    training.execute_path(
        SamplingRuntime(), root, tmp_path / "continuous", run, schedule, questions, tasks
    )
    SamplingRuntime.generated = 0
    real_update = training.update
    count = 0

    def fail_before_fourth(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 4:
            raise RuntimeError("simulated process crash")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(training, "update", fail_before_fourth)
    with pytest.raises(RuntimeError, match="simulated"):
        training.execute_path(
            SamplingRuntime(), root, tmp_path / "resumed", run, schedule, questions, tasks
        )
    assert json.loads((tmp_path / "resumed/checkpoints/LATEST.json").read_text())["step"] == 0
    monkeypatch.setattr(training, "update", real_update)
    training.execute_path(
        SamplingRuntime(), root, tmp_path / "resumed", run, schedule, questions, tasks
    )
    assert SamplingRuntime.generated == 8 * 128
    receipts = [
        json.loads((tmp_path / name / "checkpoints/LATEST.json").read_text())
        for name in ("continuous", "resumed")
    ]
    assert receipts[0]["state_hash"] == receipts[1]["state_hash"]
    assert len(list((tmp_path / "resumed/update_attempts").glob("*.json"))) == 11


def test_lease_interrupts_uncommitted_gradient_between_sequences_and_never_steps_adam():
    from sr_f1.training import LeaseEnding

    runtime = TinyRuntime()
    opt, scheduler = optimizer_for(runtime)
    reference = trainable_state(runtime.model)
    before = state_hash(reference)
    batch = examples(runtime)
    boundary = {"requested": False}
    original_forward = runtime.sequence_forward
    gradients = 0

    def forward(*args, **kwargs):
        nonlocal gradients
        result = original_forward(*args, **kwargs)
        if kwargs.get("grad"):
            gradients += 1
            if gradients == 3:
                boundary["requested"] = True
        return result

    runtime.sequence_forward = forward
    with pytest.raises(LeaseEnding, match="UNCOMMITTED_GRADIENT"):
        update(runtime, opt, scheduler, reference, batch, boundary=boundary)
    assert gradients == 3
    assert state_hash(trainable_state(runtime.model)) == before
    assert not opt.state
    assert not any(kind == "physical_optimizer_updates" for kind, _, _ in runtime.events)
    assert all(p.grad is None for p in runtime.model.parameters())


def test_resumed_milestone_evaluation_checkpoint_stops_before_next_training_update(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from mm_core.vl_runtime import hash_json
    from sr_f1 import training

    runtime = TinyRuntime()
    runtime.identity = {"trainable_state_hash": state_hash(trainable_state(runtime.model))}
    runtime.training_generation_config = lambda: SimpleNamespace(
        to_dict=lambda: {"fixture": "sampling"}
    )

    def configure(runtime):
        opt, sched = optimizer_for(runtime)
        return opt, sched, {"fixture": "CPU"}

    monkeypatch.setattr(training, "configure_training", configure)
    optimizer, scheduler, modules = configure(runtime)
    reference = trainable_state(runtime.model)
    run = {"run_id": "MILESTONE_LEASE", "paired_seed": 1, "H": 33, "arm": "J"}
    identity = {
        **run,
        "training_identity": modules,
        "start_policy_hash": state_hash(reference),
        "stress": False,
    }
    state = checkpoint_state(
        runtime,
        optimizer,
        scheduler,
        run_identity=identity,
        step=32,
        reference=reference,
        stream_hash=digest([]),
        sampling_hash=hash_json({"fixture": "sampling"}),
        diagnostics_hash=digest({"fixture": "prior_step"}),
    )
    commit_checkpoint(tmp_path / "run/checkpoints", state)
    (tmp_path / "run/steps").mkdir()
    (tmp_path / "run/steps/32.json").write_text(json.dumps({"fixture": "prior_step"}))
    calls = []

    def incomplete_fit(actual, step):
        calls.append((step, state_hash(trainable_state(actual.model))))
        return {"status": "CHECKPOINTED", "reason": "PREEMPTION"}

    monkeypatch.setattr(
        training, "update", lambda *a, **kw: pytest.fail("advanced past interrupted FIT")
    )
    result = training.execute_path(
        runtime, tmp_path, tmp_path / "run", run, [], {}, {}, on_milestone=incomplete_fit
    )
    assert result["status"] == "CHECKPOINTED" and result["final_step"] == 32
    assert result["reason"] == "PREEMPTION" and calls == [(32, state_hash(reference))]
