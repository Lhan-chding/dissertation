"""Focused CPU equivalence and recovery tests; no real-model acceptance claim."""

import copy
import random

import numpy as np
import pytest
import torch

from mm_core.training import trainable_state
from sr_f3_ce.training import fresh_optimizer, group_advantages
from sr_f3_ce.training import update as old_update
from sr_f4.training import (
    load_checkpoint,
    native_advantages,
    save_checkpoint,
    update,
    validate_groups,
)
from sr_f12.training import sequence_objective


class ToyRuntime:
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.register_parameter(
            "theta", torch.nn.Parameter(torch.tensor([0.13, -0.07, 0.02]))
        )
        self.prepare_calls = 0

    def prepare_input(self, model_input, root, *, protocol):
        self.prepare_calls += 1
        return {"inputs": model_input}

    def logp(self, record, theta=None):
        values = torch.tensor(record["tokens"], dtype=torch.float32)
        features = torch.stack((torch.ones_like(values), values / 20, (values % 3) / 5), -1)
        return torch.nn.functional.logsigmoid(
            features @ (self.model.theta if theta is None else theta)
        )

    def batch_sequence_forward(self, records, *, purpose, grad):
        assert purpose == "training_policy" and grad
        return [self.logp(record) for record in records]

    def batch_reference_forward(self, records, reference):
        return [self.logp(record, reference["theta"]).detach() for record in records]


def records_for(runtime, sizes, *, repeated_qid=False):
    records = []
    for group, size in enumerate(sizes):
        for sample in range(size):
            record = dict(
                qid="repeated" if repeated_qid else f"q{group}",
                group_id=f"g{group}",
                tokens=[sample % 7 + 1, group % 5 + 1] * (1 + sample % 3),
                score={"J": int(sample % 3 == 0)},
                reward=int(sample % 3 == 0),
                model_input={"prompt": "question", "image_file": None, "image_sha256": None},
                run_root="/unused",
                protocol="structured",
            )
            record["sampler_logprobs"] = runtime.logp(record).detach().tolist()
            records.append(record)
    return records


def test_advantages_match_all_binary_g8_cases():
    patterns = np.array([[(value >> bit) & 1 for bit in range(8)] for value in range(256)])
    for start in range(0, 256, 16):
        native = group_advantages(patterns[start : start + 16])
        adapted = np.asarray([native_advantages(row) for row in patterns[start : start + 16]])
        np.testing.assert_array_equal(adapted, native)


def test_variable_groups_preserve_repeated_qids_and_reject_boundary_mix():
    runtime = ToyRuntime()
    sizes = [32, 32, 8, 8, 8, 8, 8, 8, 8, 8]
    records = records_for(runtime, sizes, repeated_qid=True)
    validate_groups(records, sizes)
    broken = copy.deepcopy(records)
    broken[31]["group_id"] = "g1"
    with pytest.raises(ValueError, match="identity"):
        validate_groups(broken, sizes)


def test_j_update_matches_native_loss_gradient_and_adam_step():
    old, new = ToyRuntime(), ToyRuntime()
    reference = {"theta": old.model.theta.detach().clone() + 0.02}
    records = records_for(old, [8] * 16)
    old_optimizer, old_scheduler = fresh_optimizer(old.model)
    new_optimizer, new_scheduler = fresh_optimizer(new.model)
    old_result = old_update(
        old,
        old_optimizer,
        old_scheduler,
        records,
        arm="RL",
        reference=reference,
        step=1,
        microbatch_size=1,
    )
    new_result = update(new, new_optimizer, new_scheduler, records, [8] * 16, reference)
    for name in ("loss", "policy", "kl", "gradient_norm_before_clip"):
        assert new_result[name] == pytest.approx(old_result[name], abs=1e-7, rel=1e-5)
    torch.testing.assert_close(new.model.theta.grad, old.model.theta.grad, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(new.model.theta, old.model.theta, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(
        new_optimizer.state[new.model.theta]["exp_avg"],
        old_optimizer.state[old.model.theta]["exp_avg"],
        atol=1e-7,
        rtol=1e-5,
    )
    assert new_scheduler.state_dict() == old_scheduler.state_dict()


def test_final_advantages_reach_loss_without_restandardization():
    actual, expected = ToyRuntime(), ToyRuntime()
    sizes = [32] * 4
    records = records_for(actual, sizes, repeated_qid=True)
    reference = {"theta": actual.model.theta.detach().clone() + 0.02}
    final = [(0.125 if index % 2 else -0.125) for index in range(128)]
    optimizer, scheduler = fresh_optimizer(actual.model)
    expected_optimizer, expected_scheduler = fresh_optimizer(expected.model)
    for row, advantage in zip(records, final, strict=True):
        current = expected.logp(row)
        ref = expected.logp(row, reference["theta"]).detach()
        loss = sequence_objective(current, torch.tensor(row["sampler_logprobs"]), ref, advantage)[
            "loss"
        ]
        (loss / 128).backward()
    torch.nn.utils.clip_grad_norm_(expected.model.parameters(), 1.0)
    expected_optimizer.step()
    expected_scheduler.step()
    result = update(actual, optimizer, scheduler, records, sizes, reference, final_advantages=final)
    assert result["final_advantages"] == final
    torch.testing.assert_close(
        actual.model.theta.grad, expected.model.theta.grad, atol=1e-7, rtol=1e-5
    )
    torch.testing.assert_close(actual.model.theta, expected.model.theta, atol=1e-7, rtol=1e-5)


def test_checkpoint_restores_parameters_optimizer_cursor_and_rng(tmp_path):
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    update(runtime, optimizer, scheduler, records_for(runtime, [8] * 16), [8] * 16, reference)
    policy = trainable_state(runtime.model)
    identity = {"run_id": "A_b0_LZ", "parent": "SRF2_J6_s71001"}
    sampler = {"p": {"INTERVAL_v7": 0.125}, "tier": "learning", "next_step": 2}
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    target = tmp_path / "checkpoint.pt"
    save_checkpoint(
        target, runtime, optimizer, scheduler, identity=identity, step=1, sampler_state=sampler
    )
    expected = random.random(), float(np.random.random()), torch.rand(3)
    fresh = ToyRuntime()
    fresh_optimizer_, fresh_scheduler = fresh_optimizer(fresh.model)
    step, restored_sampler = load_checkpoint(
        target, fresh, fresh_optimizer_, fresh_scheduler, identity=identity
    )
    actual = random.random(), float(np.random.random()), torch.rand(3)
    assert (step, restored_sampler) == (1, sampler)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(fresh.model.theta, policy["theta"], rtol=0, atol=0)
    assert fresh_scheduler.state_dict() == scheduler.state_dict()
    torch.testing.assert_close(
        fresh_optimizer_.state[fresh.model.theta]["exp_avg"],
        optimizer.state[runtime.model.theta]["exp_avg"],
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="another registered path"):
        load_checkpoint(
            target, fresh, fresh_optimizer_, fresh_scheduler, identity={"run_id": "other"}
        )


def test_independent_draw_rng_matches_per_draw_streams_and_batch_partition():
    from sr_f4.runtime import independent_draw_rng

    original = torch.multinomial
    seeds = [101, 202, 303, 404]
    probabilities = torch.tensor(
        [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2], [0.6, 0.1, 0.3]]
    )
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    expected = torch.stack(
        [
            torch.stack(
                [
                    original(row, 1, generator=generator)
                    for row, generator in zip(probabilities, generators, strict=True)
                ]
            )
            for _ in range(12)
        ]
    )
    global_rng = torch.get_rng_state().clone()
    with independent_draw_rng(torch, seeds):
        actual = torch.stack([torch.multinomial(probabilities, 1) for _ in range(12)])
    assert torch.multinomial is original
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    with independent_draw_rng(torch, seeds[2:]):
        partition = torch.stack([torch.multinomial(probabilities[2:], 1) for _ in range(12)])
    torch.testing.assert_close(partition, actual[:, 2:], rtol=0, atol=0)


def test_independent_draw_rng_restores_sampler_after_invalid_backend_shape():
    from sr_f4.runtime import independent_draw_rng

    original = torch.multinomial
    with pytest.raises(RuntimeError, match="sampler shape"), independent_draw_rng(torch, [1, 2]):
        torch.multinomial(torch.tensor([0.5, 0.5]), 1)
    assert torch.multinomial is original


@pytest.mark.parametrize("step", [16, 64])
def test_committed_checkpoint_publication_repairs_crash_idempotently(tmp_path, step):
    from pathlib import Path
    from types import SimpleNamespace

    from sr_f4.common import read
    from sr_f4.run import publish_committed_checkpoint

    class Exporter:
        calls = 0

        def save_pretrained(self, path, *, safe_serialization):
            assert safe_serialization is True
            self.calls += 1
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_model.safetensors").write_bytes(
                b"partial" if self.calls == 1 else b"complete"
            )
            if self.calls == 1:
                raise RuntimeError("injected crash during adapter publication")

    model = Exporter()
    runtime = SimpleNamespace(model=model)
    latest = tmp_path / "checkpoints/latest.pt"
    latest.parent.mkdir()
    latest.write_bytes(b"committed optimizer and adapter state")
    with pytest.raises(RuntimeError, match="injected crash"):
        publish_committed_checkpoint(runtime, tmp_path, latest, step)
    endpoint = tmp_path / "checkpoints" / f"step-{step:03d}.pt"
    adapter = tmp_path / "adapters" / f"step-{step:03d}"
    assert endpoint.read_bytes() == latest.read_bytes()
    assert not (adapter / "PUBLISHED.json").exists()
    publish_committed_checkpoint(runtime, tmp_path, latest, step)
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"complete"
    assert read(adapter / "PUBLISHED.json")["step"] == step
    assert read(tmp_path / "PROGRESS.json")["successful_updates"] == step
    publish_committed_checkpoint(runtime, tmp_path, latest, step)
    assert model.calls == 2
    assert read(tmp_path / "PROGRESS.json")["answers"] == step * 128
