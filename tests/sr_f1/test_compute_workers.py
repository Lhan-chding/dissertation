"""Real CPU autograd and spawned-replica parity; no claim of GPU acceptance."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from mm_core.training import capture_rng, state_hash, trainable_state
from sr_f1.compute_workers import ComputeParallelManager, ComputeReplica, encode_tensors
from sr_f1.contract import digest, reward_advantages
from sr_f1.training import update

# Load the sibling by absolute path: the combined suite has another
# tests/mm_core/test_training.py and runs with pytest's importlib mode.
_helper_spec = importlib.util.spec_from_file_location(
    "sr_f1_compute_training_test_helpers", Path(__file__).with_name("test_training.py")
)
_helpers = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helpers)
TinyRuntime = _helpers.TinyRuntime
examples = _helpers.examples
optimizer_for = _helpers.optimizer_for
scores = _helpers.scores


def tiny_factory(rank, config):
    torch.set_num_threads(1)
    return ComputeReplica(TinyRuntime(), rank=rank)


def manager_for(runtime):
    return ComputeParallelManager(
        runtime, [{}, {}], factory=tiny_factory, ready_timeout=30, task_timeout=30
    )


@pytest.mark.parametrize("arm", ["A", "J", "PART", "DEC", "GATE", "ZERO", "STRESS"])
def test_parallel_full_update_matches_serial_all_diagnostics_and_state(arm):
    torch.set_num_threads(1)
    serial, parallel = TinyRuntime(), TinyRuntime()
    ref = trainable_state(serial.model)
    so, ss = optimizer_for(serial)
    po, ps = optimizer_for(parallel)
    values = (
        [0.0] * 128
        if arm in {"ZERO", "STRESS"}
        else reward_advantages(arm, scores())[0].reshape(-1)
    )
    batch = examples(serial, values)
    expected = update(serial, so, ss, ref, batch, probability_gate=True, stress=arm == "STRESS")
    manager = manager_for(parallel)
    parallel.compute_parallel_manager = manager
    try:
        before_rng = state_hash(capture_rng())
        actual = update(parallel, po, ps, ref, batch, probability_gate=True, stress=arm == "STRESS")
        assert state_hash(capture_rng()) == before_rng
        assert actual == expected
        assert state_hash(trainable_state(parallel.model)) == state_hash(
            trainable_state(serial.model)
        )
        assert state_hash(po.state_dict()) == state_hash(so.state_dict())
        assert state_hash(ps.state_dict()) == state_hash(ss.state_dict())
        assert [e for e in parallel.events if e[0] == "physical_optimizer_updates"] == serial.events
        assert (
            len([e for e in parallel.events if e[0] == "parallel_sequence_gradient_completed"])
            == 128
        )
        # A second actual optimizer step forces all persistent replicas to reload.
        batch2 = examples(serial, values)
        expected2 = update(
            serial, so, ss, ref, batch2, probability_gate=True, stress=arm == "STRESS"
        )
        actual2 = update(
            parallel, po, ps, ref, batch2, probability_gate=True, stress=arm == "STRESS"
        )
        assert actual2 == expected2
        assert state_hash(po.state_dict()) == state_hash(so.state_dict())
    finally:
        manager.close()


def snapshot(replica, reference):
    binding = dict(step=1, policy_hash=state_hash(reference), reference_hash=state_hash(reference))
    replica.handle(
        dict(
            operation="synchronize",
            parameters=encode_tensors(reference),
            reference=encode_tensors(reference),
            binding=binding,
        )
    )
    return binding


def test_replica_rejects_unsynchronized_policy_and_changed_sample():
    runtime = TinyRuntime()
    replica = ComputeReplica(runtime)
    ref = trainable_state(runtime.model)
    binding = snapshot(replica, ref)
    example = examples(runtime)[0]
    payload = dict(
        operation="gradient",
        binding=binding,
        example=example,
        sample_hash=digest(example),
        example_index=0,
        stress=False,
    )
    changed = copy.deepcopy(payload)
    changed["binding"]["policy_hash"] = "0" * 64
    with pytest.raises(PermissionError, match="matching"):
        replica.handle(changed)
    changed = copy.deepcopy(payload)
    changed["example"]["advantage"] = 7.0
    with pytest.raises(PermissionError, match="content"):
        replica.handle(changed)


def test_replica_rejects_rng_consumption_and_restores_state():
    runtime = TinyRuntime()
    real = runtime.sequence_forward

    def stochastic(*args, **kwargs):
        torch.rand(1)
        return real(*args, **kwargs)

    replica = ComputeReplica(runtime)
    ref = trainable_state(runtime.model)
    binding = snapshot(replica, ref)
    example = examples(runtime)[0]
    runtime.sequence_forward = stochastic
    before = state_hash(capture_rng())
    with pytest.raises(RuntimeError, match="stochastic RNG"):
        replica.handle(
            dict(
                operation="gradient",
                binding=binding,
                example=example,
                sample_hash=digest(example),
                example_index=0,
                stress=False,
            )
        )
    assert state_hash(capture_rng()) == before
    assert all(p.grad is None for p in runtime.model.parameters())


def test_child_failure_never_commits_optimizer_or_leaves_partial_parent_gradients():
    runtime = TinyRuntime()
    ref = trainable_state(runtime.model)
    opt, sched = optimizer_for(runtime)
    batch = examples(runtime)
    batch[1]["record"]["tokens"][0] = 9
    manager = manager_for(runtime)
    runtime.compute_parallel_manager = manager
    from sr_f1.compute_parallel import ComputeWorkerError

    with pytest.raises(ComputeWorkerError):
        update(runtime, opt, sched, ref, batch)
    assert state_hash(trainable_state(runtime.model)) == state_hash(ref)
    assert not opt.state
    assert all(p.grad is None for p in runtime.model.parameters())
    assert not runtime.events
    assert manager.closed


def test_boundary_does_not_commit_partial_parallel_gradients():
    runtime = TinyRuntime()
    ref = trainable_state(runtime.model)
    opt, sched = optimizer_for(runtime)
    manager = manager_for(runtime)
    runtime.compute_parallel_manager = manager
    from sr_f1.training import LeaseEnding

    with pytest.raises(LeaseEnding):
        update(runtime, opt, sched, ref, examples(runtime), boundary=dict(requested=True))
    assert not opt.state
    assert not runtime.events
    assert manager.closed


@pytest.mark.parametrize("failure", [False, True])
def test_execute_path_closes_owned_replica_pool_on_success_and_failure(monkeypatch, failure):
    import sr_f1.training as training

    class Manager:
        closed = False

        def close(self):
            self.closed = True

    runtime = TinyRuntime()
    manager = Manager()
    runtime.compute_parallel_manager = manager

    def segment(*args, **kwargs):
        if failure:
            raise RuntimeError("segment failure")
        return {"status": "COMPLETE"}

    monkeypatch.setattr(training, "_execute_path", segment)
    if failure:
        with pytest.raises(RuntimeError, match="segment failure"):
            training.execute_path(runtime, None, None, None, None, None, None)
    else:
        assert training.execute_path(runtime, None, None, None, None, None, None) == {
            "status": "COMPLETE"
        }
    assert manager.closed
    assert not hasattr(runtime, "compute_parallel_manager")


@pytest.mark.parametrize(
    "corruption",
    [None, "duplicate_uuid", "wrong_rank", "wrong_adapter", "two_visible", "wrong_topology"],
)
def test_production_readiness_binds_each_single_gpu_to_common_model(corruption):
    from types import SimpleNamespace

    contract = {"gpu_count": 3, "policy": "test-frozen-policy"}
    devices = [{"uuid": f"GPU-{rank:064x}"} for rank in range(3)]
    parent_identity = {
        "compute_parallel": contract,
        "base_model_weights_hash": "base",
        "trainable_state_hash": "zero",
        "adapter_hash": "adapter",
        "freeze_sha256": "freeze",
        "hardware": {"compute_devices": devices, "cuda_uuid": devices[0]["uuid"]},
    }
    identities = {0: {"rank": 0, "runtime": parent_identity}}
    for rank in (1, 2):
        child = copy.deepcopy(parent_identity)
        child["compute_rank"] = rank
        child["hardware"] = {"cuda_uuid": devices[rank]["uuid"], "cuda_visible_device_count": 1}
        identities[rank] = {"rank": rank, "runtime": child}
    configs = [{"visibility": devices[rank]["uuid"]} for rank in (1, 2)]
    if corruption == "duplicate_uuid":
        identities[2]["runtime"]["hardware"]["cuda_uuid"] = devices[1]["uuid"]
    elif corruption == "wrong_rank":
        identities[1]["runtime"]["compute_rank"] = 2
    elif corruption == "wrong_adapter":
        identities[1]["runtime"]["adapter_hash"] = "changed"
    elif corruption == "two_visible":
        identities[1]["runtime"]["hardware"]["cuda_visible_device_count"] = 2
    elif corruption == "wrong_topology":
        identities[1]["runtime"]["compute_parallel"]["gpu_count"] = 5
    manager = object.__new__(ComputeParallelManager)
    manager.runtime = SimpleNamespace(identity=parent_identity, compute_parallel_identity=contract)
    manager.pool = SimpleNamespace(worker_identities=identities)
    if corruption:
        with pytest.raises(PermissionError):
            manager._verify_worker_identities(configs)
    else:
        manager._verify_worker_identities(configs)
