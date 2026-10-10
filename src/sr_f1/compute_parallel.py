"""Bounded, ordered per-sequence work on persistent, isolated compute processes.

The pool never combines gradients, seeds RNGs, or performs optimizer updates.
The owner applies returned per-sequence results in the original input order.
Factories and their configuration must be spawn-pickleable; each factory owns
device selection, runtime creation, and any device-identity validation.
"""

from __future__ import annotations

import contextlib
import math
import multiprocessing as mp
import queue
import time
import traceback

IDENTITY_FIELDS = frozenset(
    {"step", "policy_hash", "reference_hash", "example_index", "sample_hash"}
)


class ComputeWorkerError(RuntimeError):
    """A worker failed, exited, timed out, or returned an unauthenticated result."""


def validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
        raise ValueError("Per-sequence compute identity fields differ")
    for key in ("step", "example_index"):
        if type(identity[key]) is not int or identity[key] < 0:
            raise ValueError(f"Invalid compute identity {key}")
    for key in ("policy_hash", "reference_hash", "sample_hash"):
        value = identity[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Invalid compute identity {key}")


def _worker_main(rank, config, factory, incoming, outgoing):
    runtime = None
    binding = None
    try:
        runtime = factory(rank, config)
        if not callable(getattr(runtime, "handle", None)):
            raise TypeError("Compute factory must return an object with handle(payload)")
        outgoing.put(
            {"kind": "ready", "rank": rank, "identity": getattr(runtime, "identity", None)}
        )
        while True:
            request = incoming.get()
            if request is None:
                return
            binding = request["identity"]
            validate_identity(binding)
            result = runtime.handle(request["payload"])
            outgoing.put({"kind": "result", "rank": rank, "identity": binding, "result": result})
            binding = None
    except BaseException as error:
        outgoing.put(
            {
                "kind": "error",
                "rank": rank,
                "identity": binding,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if runtime is not None and callable(getattr(runtime, "close", None)):
            runtime.close()


class OrderedComputePool:
    """Persistent spawn workers; at most N submitted/unconsumed sequence results.

    ``factory(rank, config)`` returns an object exposing ``handle(payload)`` and
    optionally ``identity`` and ``close()``. A task is a dict with exactly
    ``identity`` and ``payload``; identity has IDENTITY_FIELDS. map_ordered
    returns envelopes containing rank, the authenticated identity, and result.
    Tasks must be the original contiguous example order starting at zero, all
    belonging to one step/policy/reference. Worker completion order is irrelevant.

    ``local_handler`` optionally reuses the owner process's runtime as rank zero;
    configurations then describe only children, whose ranks start at one. Child
    work is dispatched before the local call, so the owner also computes. The
    pool does not close the owner runtime. Owner code must enforce its own signal
    boundaries; the pool can check a local deadline only after handle returns.

    Workers are terminated on any failure or abandoned map iteration. Every IPC
    wait is bounded. A timeout is an explicit technical error, never scientific data.
    """

    def __init__(
        self,
        factory,
        worker_configs,
        *,
        ready_timeout=600.0,
        task_timeout=3600.0,
        shutdown_timeout=10.0,
        local_handler=None,
    ):
        if not worker_configs and local_handler is None:
            raise ValueError("At least one compute worker is required")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (ready_timeout, task_timeout, shutdown_timeout)
        ):
            raise ValueError("Compute timeouts must be positive")
        if local_handler is not None and not callable(getattr(local_handler, "handle", None)):
            raise TypeError("Local compute handler must expose handle(payload)")
        self.local_handler = local_handler
        self.task_timeout = float(task_timeout)
        self.shutdown_timeout = float(shutdown_timeout)
        self.closed = False
        self.busy = False
        context = mp.get_context("spawn")
        self.outgoing = context.Queue(maxsize=max(1, 2 * len(worker_configs)))
        self.incoming = [context.Queue(maxsize=1) for _ in worker_configs]
        self.processes = []
        self.worker_identities = (
            {0: getattr(local_handler, "identity", None)} if local_handler is not None else {}
        )
        offset = int(local_handler is not None)
        self.child_ranks = list(range(offset, offset + len(worker_configs)))
        self.channels = dict(zip(self.child_ranks, self.incoming, strict=True))
        try:
            for rank, config in zip(self.child_ranks, worker_configs, strict=True):
                process = context.Process(
                    target=_worker_main,
                    args=(rank, config, factory, self.channels[rank], self.outgoing),
                    name=f"sr-f1-compute-{rank}",
                )
                process.start()
                self.processes.append(process)
            deadline = time.monotonic() + ready_timeout
            while len(self.worker_identities) != len(self.processes) + offset:
                message = self._receive(deadline)
                rank = message.get("rank")
                if message.get("kind") != "ready" or rank in self.worker_identities:
                    raise ComputeWorkerError("Invalid or duplicate compute worker readiness")
                self.worker_identities[rank] = message["identity"]
        except BaseException:
            self.close(abort=True)
            raise

    def _receive(self, deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ComputeWorkerError("Compute worker deadline exceeded")
            try:
                message = self.outgoing.get(timeout=min(0.1, remaining))
            except queue.Empty:
                dead = [
                    (rank, p.exitcode)
                    for rank, p in zip(self.child_ranks, self.processes, strict=True)
                    if not p.is_alive()
                ]
                if dead:
                    raise ComputeWorkerError(
                        f"Compute workers exited unexpectedly: {dead}"
                    ) from None
                continue
            if message.get("kind") == "error":
                raise ComputeWorkerError(
                    f"Compute worker {message['rank']} failed: "
                    f"{message['error_type']}: {message['error']}\n{message['traceback']}"
                )
            if type(message.get("rank")) is not int or message["rank"] not in self.child_ranks:
                raise ComputeWorkerError("Compute worker rank differs")
            return message

    def map_ordered(self, tasks):
        """Yield exact per-sequence outputs in the original order, without reduction."""
        if self.closed or self.busy:
            raise RuntimeError("Compute pool is closed or already executing")
        tasks = list(tasks)
        common = None
        for index, task in enumerate(tasks):
            if not isinstance(task, dict) or set(task) != {"identity", "payload"}:
                raise ValueError("Compute task must contain identity and payload")
            validate_identity(task["identity"])
            identity = task["identity"]
            current = tuple(identity[k] for k in ("step", "policy_hash", "reference_hash"))
            if identity["example_index"] != index or (common is not None and current != common):
                raise ValueError("Compute task order or shared update identity differs")
            common = current
        self.busy = True
        complete = False
        pending, buffered = {}, {}
        next_submit = next_yield = 0
        available = list(range(len(self.processes)))
        try:
            if self.local_handler is not None:
                yield from self._map_with_local(tasks)
                complete = True
                return
            while next_yield < len(tasks):
                while (
                    available
                    and next_submit < len(tasks)
                    and next_submit - next_yield < len(self.processes)
                ):
                    rank = available.pop(0)
                    # Copy identity so callers cannot change the expected envelope.
                    request = {
                        **tasks[next_submit],
                        "identity": dict(tasks[next_submit]["identity"]),
                    }
                    self.incoming[rank].put(request, timeout=self.shutdown_timeout)
                    pending[rank] = (
                        next_submit,
                        request["identity"],
                        time.monotonic() + self.task_timeout,
                    )
                    next_submit += 1
                message = self._receive(min(item[2] for item in pending.values()))
                rank = message["rank"]
                if message.get("kind") != "result" or rank not in pending:
                    raise ComputeWorkerError("Unexpected or duplicate compute result")
                index, identity, _ = pending.pop(rank)
                if message.get("identity") != identity:
                    raise ComputeWorkerError("Compute result identity differs")
                buffered[index] = message
                available.append(rank)
                while next_yield in buffered:
                    result = buffered.pop(next_yield)
                    next_yield += 1
                    yield result
            complete = True
        finally:
            self.busy = False
            if not complete:
                self.close(abort=True)

    def _map_with_local(self, tasks):
        """Compute on the owner's GPU concurrently with bounded child waves.

        The local call belongs to the owner and cannot be killed by this pool;
        its runtime must provide its own signal/boundary handling. Its deadline
        is checked when it returns. Child waits and process cleanup stay bounded.
        """
        width = len(self.processes) + 1
        for start in range(0, len(tasks), width):
            wave = tasks[start : start + width]
            pending = {}
            deadline = time.monotonic() + self.task_timeout
            for rank, task in zip(self.child_ranks, wave[1:], strict=False):
                request = {**task, "identity": dict(task["identity"])}
                self.channels[rank].put(request, timeout=self.shutdown_timeout)
                pending[rank] = request["identity"]
            identity = dict(wave[0]["identity"])
            result = self.local_handler.handle(wave[0]["payload"])
            if time.monotonic() > deadline:
                raise ComputeWorkerError("Local compute worker deadline exceeded")
            results = {start: dict(kind="result", rank=0, identity=identity, result=result)}
            while pending:
                message = self._receive(deadline)
                rank = message["rank"]
                if message.get("kind") != "result" or rank not in pending:
                    raise ComputeWorkerError("Unexpected or duplicate compute result")
                if message.get("identity") != pending.pop(rank):
                    raise ComputeWorkerError("Compute result identity differs")
                results[message["identity"]["example_index"]] = message
            for index in range(start, start + len(wave)):
                yield results[index]

    def close(self, *, abort=False):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + self.shutdown_timeout
        if not abort:
            for incoming in self.incoming:
                with contextlib.suppress(queue.Full):
                    incoming.put_nowait(None)
            for process in self.processes:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=self.shutdown_timeout)
            if process.is_alive():
                process.kill()
                process.join(timeout=self.shutdown_timeout)
        for channel in (*self.incoming, self.outgoing):
            channel.cancel_join_thread()
            channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close(abort=exc_type is not None)
