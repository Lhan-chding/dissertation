import copy
import os
import time

import pytest
import torch

from sr_f1.compute_parallel import ComputeWorkerError, OrderedComputePool


def tasks(count, **payload):
    return [
        dict(
            identity=dict(
                step=1,
                policy_hash="a" * 64,
                reference_hash="b" * 64,
                sample_hash=f"{index:064x}",
                example_index=index,
            ),
            payload=dict(index=index, **payload),
        )
        for index in range(count)
    ]


class EchoWorker:
    def __init__(self, rank, config):
        if config == "init_error":
            raise ValueError("initialization failed")
        self.identity = dict(pid=os.getpid(), rank=rank)

    def handle(self, payload):
        if payload.get("exit"):
            os._exit(13)
        if payload.get("error"):
            raise ValueError("sequence failed")
        time.sleep(payload.get("delay", 0) if payload["index"] == 0 else 0)
        return dict(index=payload["index"], pid=os.getpid(), finished=time.monotonic())


def echo_factory(rank, config):
    return EchoWorker(rank, config)


def sequence_losses(parameter, index):
    # FP32 and repeated parameter use exercise the actual autograd accumulation
    # path, including cancellation, variable lengths, and zero policy advantage.
    x = torch.arange(1, 2 + index % 7, dtype=torch.float32) / 7
    current = torch.stack([(parameter.sin() * value).sum() for value in x])
    old = current.detach() + 0.01
    reference = current.detach() - 0.02
    coefficient = (index % 5 - 2) / 3
    ratio = (current - old).exp()
    policy = -torch.minimum(ratio * coefficient, ratio.clamp(0.8, 1.2) * coefficient).mean()
    kl = ((reference - current).expm1() - (reference - current)).mean()
    return policy, policy + 0.02 * kl


class GradientWorker:
    def __init__(self, rank, config):
        torch.set_num_threads(1)
        self.parameter = torch.nn.Parameter(torch.tensor(config, dtype=torch.float32))
        self.identity = dict(rank=rank, pid=os.getpid())

    def handle(self, payload):
        self.parameter.grad = None
        policy, loss = sequence_losses(self.parameter, payload["index"])
        pg = torch.autograd.grad(policy / 128, self.parameter, retain_graph=True)[0]
        (loss / 128).backward()
        # Explicit bytes avoid relying on the lifetime of CUDA IPC or shared
        # tensor buffers while persistent workers start their next sequence.
        return dict(
            policy=pg.detach().numpy().tobytes(), total=self.parameter.grad.numpy().tobytes()
        )


def gradient_factory(rank, config):
    return GradientWorker(rank, config)


def test_parallel_completion_is_yielded_original_order_and_workers_persist():
    with OrderedComputePool(echo_factory, [None, None], ready_timeout=30) as pool:
        identities = pool.worker_identities.copy()
        result = list(pool.map_ordered(tasks(6, delay=0.2)))
        assert [item["identity"]["example_index"] for item in result] == list(range(6))
        assert result[1]["result"]["finished"] < result[0]["result"]["finished"]
        assert len({item["result"]["pid"] for item in result}) == 2
        assert len(list(pool.map_ordered(tasks(3)))) == 3
        assert pool.worker_identities == identities
    assert not any(process.is_alive() for process in pool.processes)


@pytest.mark.parametrize("with_local", [False, True])
def test_full_128_parallel_sequence_gradients_match_original_backward_accumulation_bitwise(
    with_local,
):
    initial = [0.125, -0.75, 2.0, -3.5]
    serial = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32))
    serial_pg = torch.zeros_like(serial)
    for index in range(128):
        policy, loss = sequence_losses(serial, index)
        serial_pg.add_(torch.autograd.grad(policy / 128, serial, retain_graph=True)[0])
        (loss / 128).backward()
    parallel = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32))
    parallel_pg = torch.zeros_like(parallel)
    local = GradientWorker(0, initial) if with_local else None
    with OrderedComputePool(
        gradient_factory, [initial] * 2, ready_timeout=30, local_handler=local
    ) as pool:
        for envelope in pool.map_ordered(tasks(128)):
            result = envelope["result"]
            pg = torch.frombuffer(bytearray(result["policy"]), dtype=torch.float32)
            gradient = torch.frombuffer(bytearray(result["total"]), dtype=torch.float32)
            parallel_pg.add_(pg)
            if parallel.grad is None:
                parallel.grad = gradient.clone()
            else:
                parallel.grad.add_(gradient)
    assert torch.equal(serial_pg, parallel_pg)
    assert torch.equal(serial.grad, parallel.grad)
    optimizers = [torch.optim.AdamW([p], lr=1e-4, weight_decay=0) for p in (serial, parallel)]
    for parameter, optimizer in zip((serial, parallel), optimizers, strict=True):
        torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
        optimizer.step()
    assert torch.equal(serial, parallel)
    for field in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(optimizers[0].state[serial][field], optimizers[1].state[parallel][field])


@pytest.mark.parametrize("payload", [{"error": True}, {"exit": True}, {"delay": 2}])
def test_child_error_exit_and_deadline_abort_only_own_children(payload):
    pool = OrderedComputePool(echo_factory, [None, None], ready_timeout=30, task_timeout=0.2)
    with pytest.raises(ComputeWorkerError):
        list(pool.map_ordered(tasks(2, **payload)))
    assert pool.closed
    assert not any(process.is_alive() for process in pool.processes)


def test_local_gpu_owner_computes_concurrently_with_distinct_child_ranks():
    local = EchoWorker(0, None)
    with OrderedComputePool(
        echo_factory, [None, None], ready_timeout=30, local_handler=local
    ) as pool:
        assert set(pool.worker_identities) == {0, 1, 2}
        result = list(pool.map_ordered(tasks(7, delay=0.2)))
        assert [r["identity"]["example_index"] for r in result] == list(range(7))
        assert result[0]["result"]["pid"] == os.getpid()
        assert result[1]["result"]["finished"] < result[0]["result"]["finished"]
        assert len({r["result"]["pid"] for r in result}) == 3
        assert [r["rank"] for r in result] == [0, 1, 2, 0, 1, 2, 0]
    assert not any(process.is_alive() for process in pool.processes)


def test_local_only_and_local_failure():
    with OrderedComputePool(echo_factory, [], local_handler=EchoWorker(0, None)) as pool:
        assert len(list(pool.map_ordered(tasks(3)))) == 3
    pool = OrderedComputePool(
        echo_factory, [None], ready_timeout=30, local_handler=EchoWorker(0, None)
    )
    with pytest.raises(ValueError, match="sequence failed"):
        list(pool.map_ordered(tasks(2, error=True)))
    assert not any(process.is_alive() for process in pool.processes)


def test_factory_failure_propagates():
    with pytest.raises(ComputeWorkerError, match="initialization failed"):
        OrderedComputePool(echo_factory, ["init_error"], ready_timeout=30)


def test_abandoned_iteration_stops_workers():
    pool = OrderedComputePool(echo_factory, [None, None], ready_timeout=30)
    iterator = pool.map_ordered(tasks(6))
    next(iterator)
    iterator.close()
    assert pool.closed
    assert not any(process.is_alive() for process in pool.processes)


@pytest.mark.parametrize(
    "field,value", [("policy_hash", "c" * 64), ("step", 2), ("example_index", 3)]
)
def test_wrong_update_or_sequence_identity_rejected_before_dispatch(field, value):
    with OrderedComputePool(echo_factory, [None], ready_timeout=30) as pool:
        work = tasks(2)
        work[1]["identity"][field] = value
        with pytest.raises(ValueError):
            list(pool.map_ordered(work))
        assert not pool.busy


def test_result_identity_tamper_rejected(monkeypatch):
    pool = OrderedComputePool(echo_factory, [None], ready_timeout=30)
    receive = pool._receive

    def tamper(deadline):
        result = receive(deadline)
        result["identity"]["sample_hash"] = "f" * 64
        return result

    monkeypatch.setattr(pool, "_receive", tamper)
    with pytest.raises(ComputeWorkerError, match="identity differs"):
        list(pool.map_ordered(tasks(1)))
    assert not any(process.is_alive() for process in pool.processes)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonlinear_recurrent_shared_parameters_serial_backward_vs_isolated_add(device):
    """Shared parameters receive many graph contributions before leaf accumulation.

    This tests whether isolating each sequence changes AccumulateGrad rounding;
    it is separate from native Qwen CUDA acceptance, which needs the real model.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; no claim of real GPU parity")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1842)
        serial = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8))
    serial.to(device)
    isolated = copy.deepcopy(serial)
    serial_pg = [torch.zeros_like(p) for p in serial.parameters()]
    isolated_pg = [torch.zeros_like(p) for p in isolated.parameters()]
    isolated_total = [None for _ in isolated.parameters()]

    def losses(model, index):
        hidden = torch.linspace(-0.5, 0.5, 8, device=device) * ((index % 7) + 1)
        values = []
        for position in range(2 + index % 13):
            hidden = torch.tanh(model(hidden) + hidden * 0.2)
            values.append(hidden.log_softmax(-1)[(position + index) % 8])
        current = torch.stack(values)
        old = current.detach() + (0.19 if index % 3 else -0.3)
        reference = current.detach() - 0.17
        advantage = (index % 5 - 2) / 3
        ratio = (current - old).exp()
        policy = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
        kl = ((reference - current).expm1() - (reference - current)).mean()
        return policy, policy + 0.02 * kl

    for index in range(128):
        for model, accumulator in ((serial, serial_pg), (isolated, isolated_pg)):
            policy, total = losses(model, index)
            pg = torch.autograd.grad(policy / 128, list(model.parameters()), retain_graph=True)
            for target, gradient in zip(accumulator, pg, strict=True):
                target.add_(gradient)
            (total / 128).backward()
        for parameter_index, parameter in enumerate(isolated.parameters()):
            if isolated_total[parameter_index] is None:
                isolated_total[parameter_index] = parameter.grad.clone()
            else:
                isolated_total[parameter_index].add_(parameter.grad)
            parameter.grad = None
    for left, right in zip(serial_pg, isolated_pg, strict=True):
        assert torch.equal(left, right)
    for parameter, total in zip(serial.parameters(), isolated_total, strict=True):
        assert torch.equal(parameter.grad, total)
