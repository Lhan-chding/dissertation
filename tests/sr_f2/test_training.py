import copy
import random

import numpy as np
import pytest
import torch

from mm_core.training import capture_rng, state_hash, trainable_state
from sr_f2 import training
from sr_f2.advantages import reward_advantages
from sr_f12 import training as previous


class ToyRuntime:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(0.2)
        self.calls = []

    def batch_sequence_forward(self, records, purpose, grad):
        self.calls.append(len(records))
        return [self.model.weight.reshape(()) * torch.tensor(r["features"]) - 2 for r in records]

    def batch_reference_forward(self, records, reference):
        return [reference["weight"].reshape(()) * torch.tensor(r["features"]) - 2 for r in records]


def fixture():
    runtime = ToyRuntime()
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=5e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    records = []
    for i in range(128):
        features = [float(i % 3 + 1)] * (1 + i % 4)
        records.append(
            dict(
                qid=f"q{i // 8}",
                features=features,
                tokens=[3] * len(features),
                sampler_logprobs=(0.2 * torch.tensor(features) - 2).tolist(),
            )
        )
    return runtime, optimizer, scheduler, records, trainable_state(runtime.model)


def measurement(step, **changes):
    families = {f: dict(A=0.4, E=0.8, P=0.3) for f in training.FAMILIES}
    for family, values in changes.items():
        families[family].update(values)
    return dict(step=step, families=families)


@pytest.mark.parametrize("arm", ["EARLY", "J6", "DEC_PRIME", "ADAPT"])
def test_all_arms_use_identical_parent_loss_and_updates(arm):
    group = [
        dict(
            A=int(i % 2 == 0),
            E=int(i % 3 != 0),
            p_read=1.0 if i % 6 == 2 else 0.5,
            P=int(i % 3 != 0),
            J=int(i % 6 == 2),
            J6=int(i % 4 == 2),
        )
        for i in range(8)
    ]
    families = list(training.FAMILIES) * 4
    scores = [
        [dict(row, J6=row["J6"] if family == "TOPK" else row["J"]) for row in group]
        for family in families
    ]
    advantages, _ = reward_advantages(
        arm,
        scores,
        step=1,
        families=families,
        active_families=["THRESHOLD"],
    )
    outputs = []
    for update_fn in (training.update, previous.update):
        runtime, optimizer, scheduler, records, reference = fixture()
        result = update_fn(
            runtime,
            optimizer,
            scheduler,
            records,
            advantages.ravel().tolist(),
            reference=reference,
            microbatch_size=1,
        )
        assert result["backward_calls"] == 128
        assert result["optimizer_updates"] == 1
        assert result["clip_fraction"] == 0
        outputs.append((trainable_state(runtime.model), optimizer.state_dict()))
    assert state_hash(outputs[0]) == state_hash(outputs[1])
    assert training.sequence_objective is previous.sequence_objective


def test_fixed_recipe_rejects_old_learning_rate_options_and_microbatch():
    runtime, optimizer, scheduler, records, reference = fixture()
    with pytest.raises(PermissionError, match="microbatch"):
        training.update(
            runtime,
            optimizer,
            scheduler,
            records,
            [1.0] * 128,
            reference=reference,
            microbatch_size=2,
        )
    optimizer.param_groups[0]["lr"] = 2e-5
    with pytest.raises(PermissionError):
        training.update(runtime, optimizer, scheduler, records, [1.0] * 128, reference=reference)


def test_r1_is_strict_r2_inclusive_and_never_actions_or_unlatches():
    initial = training.initial_adaptive_state()
    baseline = training.apply_monitor(
        initial,
        measurement(
            0,
            CROSS=dict(A=0.4, P=0.25),
            THRESHOLD=dict(A=0.6, P=0.2),
        ),
    )
    assert initial == training.initial_adaptive_state()
    assert baseline["active_families"] == ["THRESHOLD"]
    moved = training.apply_monitor(
        baseline,
        measurement(
            8,
            CROSS=dict(A=0.5, E=0.7, P=0.4),
            TOPK=dict(A=0.8, P=0.64),
            THRESHOLD=dict(A=0.1, P=0.9),
        ),
    )
    assert moved["history"][-1]["decisions"]["CROSS"]["R2"]
    assert not moved["history"][-1]["decisions"]["CROSS"]["R1"]
    assert moved["active_families"] == ["THRESHOLD", "TOPK"]
    assert baseline["active_families"] == ["THRESHOLD"]
    training.validate_adaptive_state(moved)
    tampered = copy.deepcopy(moved)
    tampered["active_families"] = ["TOPK"]
    with pytest.raises(PermissionError, match="replay"):
        training.validate_adaptive_state(tampered)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_monitor_probability_never_changes_controller(bad):
    with pytest.raises(PermissionError):
        training.apply_monitor(training.initial_adaptive_state(), measurement(0, CROSS=dict(A=bad)))


def test_missing_or_out_of_order_measurement_rejected():
    with pytest.raises(PermissionError):
        training.apply_monitor(training.initial_adaptive_state(), measurement(8))
    missing = measurement(0)
    missing["families"].pop("TOPK")
    with pytest.raises(PermissionError):
        training.apply_monitor(training.initial_adaptive_state(), missing)


def test_monitor_failure_restores_all_rng_and_model_mode():
    runtime = ToyRuntime()
    runtime.model.eval()
    before = state_hash(capture_rng())
    with pytest.raises(RuntimeError, match="synthetic"), training.isolated_rng(runtime):
        random.random()
        np.random.random()
        torch.rand(3)
        runtime.model.train()
        raise RuntimeError("synthetic monitor failure")
    assert state_hash(capture_rng()) == before
    assert not runtime.model.training


def checkpoint_fixture(tmp_path):
    runtime, optimizer, scheduler, records, reference = fixture()
    adaptive = training.apply_monitor(training.initial_adaptive_state(), measurement(0))
    training.update(runtime, optimizer, scheduler, records, [1.0] * 128, reference=reference)
    kwargs = dict(
        run_identity={"run": "SRF2_TEST"},
        reference=reference,
        stream_hash="stream",
        sampling_hash="sampling",
        monitor_definition_hash="monitor",
    )
    state = training.checkpoint_state(
        runtime, optimizer, scheduler, step=8, adaptive_state=adaptive, **kwargs
    )
    training.commit_checkpoint(tmp_path, state)
    return runtime, optimizer, scheduler, kwargs, state


def test_checkpoint_restores_adam_rng_and_pending_monitor(tmp_path):
    runtime, optimizer, scheduler, kwargs, state = checkpoint_fixture(tmp_path)
    runtime.model.weight.data.add_(1)
    torch.rand(2)
    restored = training.load_checkpoint(tmp_path, runtime, optimizer, scheduler, **kwargs)
    assert restored["monitor_due_step"] == 8
    assert restored["adaptive_state"]["last_monitor_step"] == 0
    assert state_hash(trainable_state(runtime.model)) == state_hash(state["parameters"])
    assert state_hash(capture_rng()) == state_hash(state["rng"])
    with pytest.raises(PermissionError, match="Prior boundary"):
        training.checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            step=9,
            adaptive_state=restored["adaptive_state"],
            **kwargs,
        )


@pytest.mark.parametrize(
    "field",
    [
        "run_identity",
        "stream_hash",
        "sampling_hash",
        "monitor_definition_hash",
        "learning_rate",
        "microbatch_size",
    ],
)
def test_checkpoint_identity_recipe_and_monitor_binding(tmp_path, field):
    runtime, optimizer, scheduler, kwargs, _ = checkpoint_fixture(tmp_path)
    kwargs[field] = 2 if field == "microbatch_size" else "changed"
    with pytest.raises(PermissionError):
        training.load_checkpoint(tmp_path, runtime, optimizer, scheduler, **kwargs)


def test_immutable_commit_recovers_interrupted_latest_pointer(tmp_path):
    runtime, optimizer, scheduler, kwargs, state = checkpoint_fixture(tmp_path)
    (tmp_path / "LATEST.json").unlink()
    runtime.model.weight.data.add_(1)
    restored = training.load_checkpoint(tmp_path, runtime, optimizer, scheduler, **kwargs)
    assert restored["committed_logical_step"] == 8
    assert state_hash(restored) == state_hash(state)
    assert (tmp_path / "LATEST.json").read_text() == (tmp_path / "commit-08.json").read_text()
