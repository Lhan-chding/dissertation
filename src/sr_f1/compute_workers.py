"""Persistent model replicas returning individual, ordered sequence gradients.

Only the coordinator owns the optimizer. Replicas never average or accumulate
across sequences; their immutable step snapshot is verified before computation.
"""

from __future__ import annotations

import contextlib
import copy
import io
import time
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state

from .contract import digest
from .runtime import copy_parameters


def encode_tensors(value):
    """Owned bytes avoid multiprocessing shared-storage lifetime and reuse hazards."""
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def decode_tensors(value):
    import torch

    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)


@contextlib.contextmanager
def unchanged_gradient_rng():
    """Gradient execution must not introduce rank-dependent stochastic state."""
    before = capture_rng()
    try:
        yield
        if state_hash(capture_rng()) != state_hash(before):
            raise RuntimeError("Parallel gradient computation consumed stochastic RNG state")
    finally:
        restore_rng(before)


def sequence_gradient(runtime, example, example_index, *, stress=False):
    """Original two-backward sequence calculation, with no batch reduction."""
    import torch

    from .training import sequence_objective

    started = time.monotonic()
    named = [(name, p) for name, p in runtime.model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    for parameter in parameters:
        parameter.grad = None
    try:
        with unchanged_gradient_rng():
            record = example["record"]
            prepared = runtime.prepare(example["row"], example["root"])
            ref = runtime.reference_forward(prepared, record["tokens"], runtime._compute_reference)[
                "logprobs"
            ].detach()
            current = runtime.sequence_forward(
                prepared, record["tokens"], purpose="training_gradient", grad=True
            )["logprobs"]
            old = torch.tensor(record["old_logprobs"], dtype=torch.float32, device=current.device)
            summaries = {
                key: dict(
                    sum=float(value.sum()), minimum=float(value.min()), maximum=float(value.max())
                )
                for key, value in (("old", old), ("reference", ref), ("current", current.detach()))
            }
            difference = (current.detach().float() - old).abs()
            coefficient = (
                (1.0 if example_index % 2 == 0 else -1.0) if stress else example["advantage"]
            )
            objective = sequence_objective(current, old, ref, coefficient)
            pg = torch.autograd.grad(
                objective["policy"] / 128, parameters, retain_graph=True, allow_unused=False
            )
            if any(not bool(torch.isfinite(value).all()) for value in pg):
                raise FloatingPointError("Nonfinite isolated policy gradient")
            (objective["loss"] / 128).backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Missing or nonfinite total sequence gradient")
            result = dict(
                names=[name for name, _ in named],
                policy_gradients=[value.detach().cpu().clone() for value in pg],
                total_gradients=[p.grad.detach().cpu().clone() for p in parameters],
                totals={
                    key: float(objective[key].detach()) / 128
                    for key in ("loss", "policy", "kl", "clip_fraction")
                },
                difference_sum=float(difference.sum()),
                difference_max=float(difference.max()),
                differences=difference.cpu().tolist(),
                token_count=len(record["tokens"]),
                probability_summaries=summaries,
            )
        # Release both backward graph owners before observing final cleanup counters.
        del current, objective, pg
        result["elapsed_seconds"] = time.monotonic() - started
        result["execution_statistics"] = copy.deepcopy(
            getattr(runtime, "last_gradient_execution", None)
        )
        # CPU copies are complete before graph/temporary activation owners die.
        result["content_hash"] = state_hash(result)
        return result
    finally:
        for parameter in parameters:
            parameter.grad = None


class ComputeReplica:
    """One local or spawned model, bound to one verified immutable update."""

    def __init__(self, runtime, *, rank=0):
        self.runtime = runtime
        self.rank = rank
        self.identity = dict(rank=rank, runtime=getattr(runtime, "identity", {}))
        self.binding = None

    def handle(self, payload):
        if payload["operation"] == "synchronize":
            policy, reference = (
                decode_tensors(payload["parameters"]),
                decode_tensors(payload["reference"]),
            )
            binding = payload["binding"]
            if (
                state_hash(policy) != binding["policy_hash"]
                or state_hash(reference) != binding["reference_hash"]
            ):
                raise PermissionError("Compute snapshot hash differs")
            with unchanged_gradient_rng():
                copy_parameters(self.runtime.model, policy)
                if state_hash(trainable_state(self.runtime.model)) != binding["policy_hash"]:
                    raise PermissionError("Compute replica did not load exact policy snapshot")
            self.runtime._compute_reference = reference
            self.binding = dict(binding)
            return dict(binding=self.binding, names=list(trainable_state(self.runtime.model)))
        if payload["operation"] != "gradient" or self.binding != payload["binding"]:
            raise PermissionError("Compute task has no matching policy/reference snapshot")
        if digest(payload["example"]) != payload["sample_hash"]:
            raise PermissionError("Compute sequence content changed")
        return encode_tensors(
            sequence_gradient(
                self.runtime, payload["example"], payload["example_index"], stress=payload["stress"]
            )
        )


def create_compute_replica(rank, config):
    """Spawn entrypoint: select and verify the device before loading its model."""
    from .runtime import load_compute_replica

    runtime = load_compute_replica(
        config["plan"], Path(config["root"]), rank, config["visibility"], config.get("account")
    )
    return ComputeReplica(runtime, rank=rank)


class ComputeParallelManager:
    """GPU 0 computes locally while persistent other-device replicas compute too."""

    def __init__(
        self,
        runtime,
        worker_configs,
        *,
        factory=create_compute_replica,
        pool_factory=None,
        **timeouts,
    ):
        from .compute_parallel import OrderedComputePool

        self.runtime = runtime
        self.local = ComputeReplica(runtime)
        self.pool = (pool_factory or OrderedComputePool)(
            factory, worker_configs, local_handler=self.local, **timeouts
        )
        self.closed = False
        self.sequence_batches = 0
        try:
            self._verify_worker_identities(worker_configs)
        except BaseException:
            self.close(abort=True)
            raise

    def _verify_worker_identities(self, configs):
        contract = getattr(self.runtime, "compute_parallel_identity", None)
        if contract is None:
            return  # CPU test runtimes have no production CUDA allocation.
        count = contract["gpu_count"]
        identities = self.pool.worker_identities
        if len(configs) != count - 1 or set(identities) != set(range(count)):
            raise PermissionError("Compute readiness does not cover the frozen topology")
        parent = self.runtime.identity
        devices = parent["hardware"]["compute_devices"]
        if len(devices) != count:
            raise PermissionError("Parent compute device inventory differs")

        def normalize(value):
            return str(value).lower().removeprefix("gpu-").replace("-", "")

        expected = [normalize(device["uuid"]) for device in devices]
        if not all(expected) or len(set(expected)) != count:
            raise PermissionError("Compute replicas require distinct physical GPU UUIDs")
        for rank, envelope in identities.items():
            actual = envelope["runtime"]
            if envelope["rank"] != rank or actual.get("compute_parallel") != contract:
                raise PermissionError("Compute replica frozen execution identity differs")
            for key in (
                "base_model_weights_hash",
                "trainable_state_hash",
                "adapter_hash",
                "freeze_sha256",
            ):
                if not parent.get(key) or actual.get(key) != parent[key]:
                    raise PermissionError("Compute replica model/common adapter identity differs")
            hardware = actual["hardware"]
            if normalize(hardware.get("cuda_uuid", "")) != expected[rank]:
                raise PermissionError(
                    "Compute replica physical device differs from parent allocation"
                )
            if rank and (
                hardware.get("cuda_visible_device_count") != 1
                or actual.get("compute_rank") != rank
                or normalize(configs[rank - 1]["visibility"]) != expected[rank]
            ):
                raise PermissionError("Child compute replica must use its unique single GPU")

    @property
    def worker_identities(self):
        return self.pool.worker_identities

    def close(self, *, abort=False):
        self.closed = True
        self.pool.close(abort=abort)

    def iter_gradients(self, reference, examples, *, stress=False, boundary=None):
        if self.closed:
            raise RuntimeError("Parallel compute manager is closed")
        parameters = trainable_state(self.runtime.model)
        binding = dict(
            step=int(examples[0]["record"].get("logical_step", self.sequence_batches)),
            policy_hash=state_hash(parameters),
            reference_hash=state_hash(reference),
        )
        names = list(parameters)
        # One task per worker fills the first wave, binding every persistent model.
        parameter_bytes, reference_bytes = encode_tensors(parameters), encode_tensors(reference)
        snapshots = [
            dict(
                identity={
                    **binding,
                    "example_index": index,
                    "sample_hash": digest(["snapshot", binding]),
                },
                payload=dict(
                    operation="synchronize",
                    binding=binding,
                    parameters=parameter_bytes,
                    reference=reference_bytes,
                ),
            )
            for index in range(len(self.pool.worker_identities))
        ]
        try:
            if boundary and boundary["requested"]:
                from .training import LeaseEnding

                raise LeaseEnding("PREEMPTION_DURING_UNCOMMITTED_GRADIENT")
            synchronized = set()
            for envelope in self.pool.map_ordered(snapshots):
                value = envelope["result"]
                if value != dict(binding=binding, names=names) or envelope["rank"] in synchronized:
                    raise PermissionError("Parallel snapshot acknowledgements differ")
                synchronized.add(envelope["rank"])
            if synchronized != set(self.pool.worker_identities):
                raise PermissionError("Not all compute replicas acknowledged the update")
            tasks = []
            for index, example in enumerate(examples):
                # JSON request identity binds the complete saved raw, input and coefficient.
                serializable = {
                    **example,
                    "root": str(example["root"]) if example["root"] is not None else None,
                }
                sample_hash = digest(serializable)
                tasks.append(
                    dict(
                        identity={**binding, "example_index": index, "sample_hash": sample_hash},
                        payload=dict(
                            operation="gradient",
                            binding=binding,
                            example=serializable,
                            sample_hash=sample_hash,
                            example_index=index,
                            stress=stress,
                        ),
                    )
                )
            for index, envelope in enumerate(self.pool.map_ordered(tasks)):
                if boundary and boundary["requested"]:
                    from .training import LeaseEnding

                    raise LeaseEnding("PREEMPTION_DURING_UNCOMMITTED_GRADIENT")
                result = decode_tensors(envelope["result"])
                if envelope["identity"] != tasks[index]["identity"] or result["names"] != names:
                    raise PermissionError("Parallel sequence result identity differs")
                if result["content_hash"] != state_hash(
                    {k: v for k, v in result.items() if k != "content_hash"}
                ):
                    raise PermissionError("Parallel sequence result bytes changed")
                if result["token_count"] != len(examples[index]["record"]["tokens"]):
                    raise PermissionError("Parallel sequence token count differs")
                self.runtime.reserve(
                    "parallel_sequence_gradient_completed",
                    1,
                    compute_rank=envelope["rank"],
                    example_index=index,
                    logical_step=binding["step"],
                    policy_hash=binding["policy_hash"],
                    reference_hash=binding["reference_hash"],
                    sample_hash=tasks[index]["identity"]["sample_hash"],
                    elapsed_seconds=result["elapsed_seconds"],
                    execution_statistics=result["execution_statistics"],
                )
                yield result
            self.sequence_batches += 1
        except BaseException:
            self.close(abort=True)
            raise
