"""One-load V4 workers, shared action shards and genuine same-origin Adam forks.

Planning is CPU-only. The sole execution switch is ``execute_gpu``; neither the
old Q3 result nor a V3 authorization receipt is an input to this campaign.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

from ..modeling_v3.io import atomic_json, canonical_hash, sha256_file, source_identity
from .data_adapter import (
    CANDIDATES,
    CONTRASTS,
    NAMESPACE,
    build_bank_plan,
    build_training_schedule,
    prepare_inputs,
)

_VERIFIED_ARTIFACT_FILES = {}


def _read(path):
    return json.loads(Path(path).read_text())


def _publish(path, value):
    path = Path(path)
    if path.exists():
        if canonical_hash(_read(path)) != canonical_hash(value):
            raise ValueError("Immutable V4 artifact changed: " + str(path))
    else:
        try:
            atomic_json(path, value)
        except FileExistsError:
            if canonical_hash(_read(path)) != canonical_hash(value):
                raise ValueError("Concurrent immutable V4 artifact differs: " + str(path)) from None


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def _verify_bound_file(binding):
    path = Path(binding["path"]).resolve()
    stat = path.stat()
    version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    expected = (binding["sha256"], version)
    prior = _VERIFIED_ARTIFACT_FILES.get(str(path))
    if prior is not None:
        if prior != expected:
            raise ValueError("Completed campaign artifact changed: " + str(path))
    else:
        if sha256_file(path) != binding["sha256"]:
            raise ValueError("Completed campaign artifact changed: " + str(path))
        _VERIFIED_ARTIFACT_FILES[str(path)] = expected


def verify_artifact_bindings(value, *, seen=None):
    """Verify campaign artifacts once per read, never scan unbound base weights."""
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            key = (value["path"], value["sha256"])
            if key not in seen:
                seen.add(key)
                _verify_bound_file(value)
                if Path(value["path"]).suffix == ".json":
                    verify_artifact_bindings(_read(value["path"]), seen=seen)
        for name, item in value.items():
            if (
                name == "checkpoint"
                and value.get("kind") == "V4_SOURCE_STEP_UPDATE"
                and value.get("checkpoint_retention") == "OPERATIONAL_UNTIL_NEXT_COMMIT"
            ):
                continue
            verify_artifact_bindings(item, seen=seen)
    elif isinstance(value, list):
        for item in value:
            verify_artifact_bindings(item, seen=seen)


def _policy_layout(runtime):
    model = runtime["adapter"].model
    scalings = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict):
                if set(scaling) != {"default"}:
                    raise ValueError("V4 expects one default LoRA adapter")
                scaling = scaling["default"]
            scalings[name] = float(scaling)
    return {
        "module_scalings": scalings,
        "parameter_layout": [
            {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
            for n, p in sorted(model.named_parameters())
            if p.requires_grad
        ],
    }


class CheckpointCache:
    """One byte/identity check per immutable file version, two resident payloads.

    A changed inode, size, mtime or ctime invalidates the import instead of
    silently promoting a different file under an already-used identity.
    """

    def __init__(self, max_resident=2):
        self.verified, self.resident = {}, OrderedDict()
        self.max_resident, self.hash_checks = max_resident, 0

    @staticmethod
    def version(path):
        s = Path(path).stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def load(self, spec):
        import torch

        from ..optimizer_fork import state_hash

        spec = spec.get("checkpoint", spec)

        def logical(state):
            if "logical_metadata" in spec:
                return {**state, "metadata": copy.deepcopy(spec["logical_metadata"])}
            return state

        path = str(Path(spec["path"]).resolve())
        key = (path, spec["sha256"], canonical_hash(spec["identity"]))
        version = self.version(path)
        if path in self.verified and self.verified[path] != (key, version):
            raise ValueError("An imported immutable checkpoint changed")
        if key in self.resident:
            self.resident.move_to_end(key)
            return logical(self.resident[key])
        first = path not in self.verified
        if first:
            self.hash_checks += 1
            if sha256_file(path) != spec["sha256"]:
                raise ValueError("Checkpoint file digest mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["identity"] != spec["identity"]:
            raise ValueError("Checkpoint identity mismatch")
        if first and state_hash(payload["state"]) != payload["state_hash"]:
            raise ValueError("Checkpoint content digest mismatch")
        self.verified[path] = (key, version)
        self.resident[key] = payload["state"]
        while len(self.resident) > self.max_resident:
            self.resident.popitem(last=False)
        return logical(payload["state"])


class CompleteState:
    """Reuse the complete historical state schema without copying/hash-scanning base."""

    def __init__(self, adapter, optimizer, sampler):
        self.adapter, self.model, self.optimizer, self.sampler = (
            adapter,
            adapter.model,
            optimizer,
            sampler,
        )
        self.frozen = {
            n: (id(p), p._version) for n, p in self.model.named_parameters() if not p.requires_grad
        }
        self.poisoned = False

    def check_frozen(self):
        now = {
            n: (id(p), p._version) for n, p in self.model.named_parameters() if not p.requires_grad
        }
        if self.poisoned or now != self.frozen:
            raise RuntimeError("Frozen parameters changed or state restoration failed")

    def capture(self, metadata=None):
        from ..followup_updates import _forward_state
        from ..optimizer_fork import capture_state

        self.check_frozen()
        return {
            **capture_state(self.model, self.optimizer, metadata),
            **_forward_state(self.model, self.adapter),
            "sampler": {"kind": "mapping", "state": copy.deepcopy(self.sampler)},
            "schema": "ssvc-v4-complete-trainable-state-1",
        }

    def restore(self, state):
        import torch

        from ..followup_updates import _restore_forward
        from ..optimizer_fork import restore_state

        self.check_frozen()
        try:
            restore_state(self.model, self.optimizer, state)
            _restore_forward(self.model, state, self.adapter)
            saved = state.get("sampler")
            self.sampler.clear()
            if saved is not None:
                if saved["kind"] != "mapping":
                    raise ValueError("V4 expects the inherited mapping sampler")
                self.sampler.update(copy.deepcopy(saved["state"]))
            self.model.zero_grad(set_to_none=True)
            for n, p in self.model.named_parameters():
                if p.requires_grad and not torch.equal(
                    p.detach(), state["parameters"][n].to(p.device)
                ):
                    raise RuntimeError("Trainable restoration differs")
            current = self.optimizer.state_dict()
            if current["param_groups"] != state["optimizer"]["param_groups"]:
                raise RuntimeError("Adam parameter groups differ")

            def equal(a, b):
                if isinstance(a, torch.Tensor):
                    return isinstance(b, torch.Tensor) and torch.equal(a, b.to(a.device))
                if isinstance(a, dict):
                    return (
                        isinstance(b, dict)
                        and a.keys() == b.keys()
                        and all(equal(a[k], b[k]) for k in a)
                    )
                if isinstance(a, (tuple, list)):
                    return (
                        type(a) is type(b)
                        and len(a) == len(b)
                        and all(equal(x, y) for x, y in zip(a, b, strict=True))
                    )
                return a == b

            if not equal(current["state"], state["optimizer"]["state"]):
                raise RuntimeError("Adam moments/counters differ after restoration")
            self.check_frozen()
        except BaseException:
            self.poisoned = True
            raise

    def guard(self):
        """Cheap mutation guard; no tensor materialization or content hashing."""
        self.check_frozen()

        def versions(v):
            if hasattr(v, "_version"):
                return (id(v), v._version)
            if isinstance(v, dict):
                return tuple(
                    (str(k), versions(x))
                    for k, x in sorted(v.items(), key=lambda item: str(item[0]))
                )
            if isinstance(v, (list, tuple)):
                return tuple(versions(x) for x in v)
            return v

        return (
            tuple((n, id(p), p._version) for n, p in self.model.named_parameters()),
            tuple((n, id(b), b._version) for n, b in self.model.named_buffers()),
            tuple((n, m.training) for n, m in self.model.named_modules()),
            versions(self.optimizer.state),
        )


def _save_state(path, state, identity):
    import torch

    from ..optimizer_fork import state_hash

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = state_hash(state)
    if path.exists():
        # A crash can publish the checkpoint before its small JSON commit.
        # Reuse only exact state/identity; never replace the original bytes.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload["identity"] != identity
            or payload["state_hash"] != digest
            or state_hash(payload["state"]) != digest
        ):
            raise ValueError("Orphan checkpoint differs from deterministic replay")
        return {**_binding(path), "identity": identity, "state_hash": digest}
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            torch.save({"identity": identity, "state": state, "state_hash": digest}, stream)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {**_binding(path), "identity": identity, "state_hash": digest}


def _tensor_bytes(value):
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _fingerprint(state, identity):
    from ..optimizer_fork import state_hash

    return state_hash(
        {
            "parameters": state["parameters"],
            "buffers": state["buffers"],
            "model_hash": identity["model_hash"],
            "mode": "eval_positions_reset",
        }
    )


def attach_runtime(
    adapter,
    optimizer,
    *,
    identity,
    data_root=None,
    parent=None,
    parity_tolerances=None,
    allocated_gpu=None,
):
    """Shared real/tiny fixture attachment; does not certify a fixture as CUDA."""
    from ..followup_updates import _forward_state

    sampler = {}
    state = CompleteState(adapter, optimizer, sampler)
    cache = CheckpointCache()
    runtime = {
        "adapter": adapter,
        "optimizer": optimizer,
        "sampler": sampler,
        "identity": identity,
        "data_root": data_root,
        "parent": parent,
        "allocated_gpu": allocated_gpu,
        "state": state,
        "checkpoint_cache": cache,
        "parameter_order": sorted(
            n for n, p in adapter.model.named_parameters() if p.requires_grad
        ),
        "parity_tolerances": parity_tolerances,
        "capture_complete": state.capture,
        "restore_complete": state.restore,
        "state_guard": state.guard,
    }
    runtime["initial_forward"] = _forward_state(adapter.model, adapter)

    def policy_loader(spec):
        import torch

        from ..followup_updates import _restore_forward

        saved = cache.load(spec)
        state.check_frozen()
        # Inference switching deliberately does not transplant another policy's
        # Adam state. Actual scratch/source updates always restore CompleteState.
        with torch.no_grad():
            for n, p in adapter.model.named_parameters():
                if p.requires_grad:
                    p.copy_(saved["parameters"][n].to(device=p.device, dtype=p.dtype))
        _restore_forward(adapter.model, saved, adapter)
        adapter.model.eval()
        adapter._reset_positions()
        # cache.load has already checked the immutable file version. Reuse the
        # verified fingerprint instead of hashing every LoRA tensor per switch.
        checkpoint = spec.get("checkpoint", spec)
        key = (checkpoint["path"], checkpoint["sha256"])
        fingerprints = runtime.setdefault("policy_fingerprints", {})
        if key not in fingerprints:
            fingerprints[key] = _fingerprint(saved, identity)
        fingerprint = fingerprints[key]
        runtime["active_policy_identity"] = {
            "inference_fingerprint": fingerprint,
            "checkpoint": spec.get("checkpoint", spec),
        }
        return fingerprint

    runtime["policy_loader"] = policy_loader
    return runtime


def load_runtime(config, bindings, *, device="cuda:0", execute_gpu=False):
    """Inherited certified loader once; no V3 phase/selection/authorization gate."""
    if execute_gpu is not True:
        raise PermissionError("The V4 campaign requires --execute-gpu")
    if device != "cuda:0":
        raise ValueError("Use cuda:0 inside each one-GPU Slurm worker")
    from ..modeling_v3.vlm_campaign import load_runtime as inherited_loader
    from .config import validate_config

    v3 = _read(Path(__file__).parents[2] / "configs/modeling_v3/protocol.json")
    validate_config(config)
    q = config["qwen"]
    old = v3["qwen"]
    lora = {**q["lora"]}
    lora["expected_matrices"] = lora.pop("expected_modules")
    optimizer = {**q["optimizer"]}
    optimizer["type"] = optimizer.pop("name")
    optimizer["step_explicit_zero_gradients"] = optimizer.pop("explicit_zero_gradients")
    if (
        lora != old["LoRA"]
        or optimizer != old["optimizer"]
        or any(q[k] != old[k] for k in ("revision", "B", "K", "steps", "Lnorm", "reward_epsilon"))
    ):
        raise ValueError("V4 runtime settings differ from the inherited production implementation")
    inherited = inherited_loader(
        v3, bindings, device=device, allow_gpu=True, acknowledge_new_experiment=True
    )
    identity = {
        **inherited["identity"],
        "protocol_version": config["version"],
        "config_hash": canonical_hash(config),
        "source_hash": source_identity()["sha256"],
    }
    return attach_runtime(
        inherited["adapter"],
        inherited["optimizer"],
        identity=identity,
        data_root=inherited["data_root"],
        parent=inherited["parent"],
        parity_tolerances=inherited["parity_tolerances"],
        allocated_gpu=inherited["allocated_gpu"],
    )


def build_task_list(config, bindings, *, out, workers=2, stages=("B", "C"), inputs=None):
    if workers not in (2, 3) or workers > config["operations"]["max_concurrent_gpus"]:
        raise ValueError("V4 uses two or at most three independent workers")
    if not set(stages) <= {"B", "C", "D", "E"}:
        raise ValueError("Unknown V4 campaign stage")
    inputs = prepare_inputs(config, bindings) if inputs is None else copy.deepcopy(inputs)
    from .analysis_rules import load_analysis_rules

    analysis_rules = load_analysis_rules()
    tasks = []
    for i in range(2):
        tasks.append(
            {
                "id": f"B_origin_{i}",
                "stage": "B",
                "kind": "bridge",
                "worker": i,
                "origin_index": i,
                "depends_on": [],
                "reserve_natural_bank_indices": list(range(4, 12)),
                "derivative_prompt_subset": _diagnostic_prompt_ids(inputs["bridge_probes"]),
            }
        )
    for i, seed in enumerate((41001, 41002)):
        sid = f"source_{seed}_X_BASE"
        tasks.append(
            {
                "id": sid,
                "stage": "C",
                "kind": "source",
                "worker": i % workers,
                "seed": seed,
                "arm": "X_BASE",
                "depends_on": ["B_MERGED"],
            }
        )
        for step in (32, 96):
            tasks.append(
                {
                    "id": f"map_{seed}_X_BASE_{step}",
                    "stage": "C",
                    "kind": "map",
                    "worker": i % workers,
                    "seed": seed,
                    "arm": "X_BASE",
                    "step": step,
                    "source_task": sid,
                    "depends_on": [sid],
                    "calibration_banks": 32,
                    "query_banks": 16,
                    "prompts": 24,
                    "draws": 256,
                    "derivative_prompt_subset": _diagnostic_prompt_ids(
                        inputs["panels"]["observation"][:24]
                    ),
                }
            )
    # D is registered but needs development selection for confirmation m and
    # roles. Explicit states prevent an unfinished data matrix being called done.
    later = {
        "D": {
            "status": "REGISTERED_NOT_EXECUTED",
            "requires": "development model/m freeze",
            "seed_roles": config["qwen"]["seed_roles"],
        },
        "E": {"status": "REGISTERED_NOT_EXECUTED", "requires": "development tracking selection"},
    }
    result = {
        "schema": "ssvc-v4-task-list-1",
        "config": copy.deepcopy(config),
        "bindings": copy.deepcopy(bindings),
        "inputs": inputs,
        "workers": workers,
        "stages": list(stages),
        "tasks": tasks,
        "analysis_rules": analysis_rules,
        "analysis_rules_hash": canonical_hash(analysis_rules),
        "later_stages": later,
        "source": source_identity(),
        "online_ssvc": False,
    }
    result["campaign_id"] = canonical_hash(result)
    path = Path(out)
    if path.suffix != ".json":
        path = path / "tasks.json"
    result["root"] = str((path.parent / "campaign").resolve())
    result["task_list_hash"] = canonical_hash(result)
    _publish(path, result)
    return result


def _diagnostic_prompt_ids(prompts):
    groups = {}
    for prompt in prompts:
        groups.setdefault((prompt["family"], prompt["interface"]), prompt["prompt_id"])
    return [groups[g] for g in sorted(groups)]


def _chunk_rows(path):
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _write_chunk(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".chunk-", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temp, compression="zstd")
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return _binding(path)


class ControlledInterruption(RuntimeError):
    """Predeclared bridge recovery exercise, after a durable chunk commit."""


def _chunks(
    root,
    identity,
    produce,
    *,
    count,
    resume,
    chunk_size=32,
    interrupt_after_first_commit=False,
    reuse_chunk=None,
):
    """Each immutable action chunk is committed once; incomplete chunks retry.

    Repeated attempts have deterministic sample keys/seeds and separate failure
    records. Completed chunks are verified and never regenerate their samples.
    """
    root = Path(root)
    if root.exists() and not resume:
        raise FileExistsError("Use explicit resume for existing task")
    root.mkdir(parents=True, exist_ok=True)
    _publish(root / "identity.json", identity)
    bindings = []
    for start in range(0, count, chunk_size):
        meta = root / f"chunk_{start:08d}.json"
        stop = min(start + chunk_size, count)
        if meta.exists():
            binding = _read(meta)
            _verify_bound_file(binding)
        else:
            rows = []
            try:
                parquet = root / f"chunk_{start:08d}.parquet"
                prepared = root / f"chunk_{start:08d}.PREPARED.json"
                shared = reuse_chunk(start, stop) if reuse_chunk is not None else None
                if shared is not None:
                    verify_artifact_bindings(shared)
                    binding = {"path": shared["path"], "sha256": shared["sha256"]}
                elif parquet.exists():
                    if not prepared.exists():
                        raise ValueError("Unbound orphan chunk requires manual inspection")
                    pending = _read(prepared)
                    rows = _chunk_rows(parquet)
                    if pending != {
                        "start": start,
                        "stop": stop,
                        "rows_hash": canonical_hash(rows),
                        "identity_hash": canonical_hash(identity),
                    }:
                        raise ValueError("Orphan chunk content differs from prepared transaction")
                    binding = _binding(parquet)
                else:
                    for i in range(start, stop):
                        rows.append(produce(i))
                    _publish(
                        prepared,
                        {
                            "start": start,
                            "stop": stop,
                            "rows_hash": canonical_hash(rows),
                            "identity_hash": canonical_hash(identity),
                        },
                    )
                    binding = _write_chunk(parquet, rows)
                _publish(meta, {**binding, "start": start, "stop": stop})
                binding = _read(meta)
                if interrupt_after_first_commit and start == 0:
                    raise ControlledInterruption(
                        "Predeclared interruption after first durable chunk"
                    )
            except BaseException as exc:
                from ..modeling_v3.vlm_observation import _fault_json

                attempt = time.time_ns()
                partial = None
                if rows and not parquet.exists():
                    partial = _write_chunk(root / f"failed_partial_{attempt}.parquet", rows)
                atomic_json(
                    root / f"failure_{attempt}.json",
                    {
                        "start": start,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "uncommitted_returned_rows": partial,
                        "counted_as_independent_samples": False,
                        "returned_fault": _fault_json(getattr(exc, "raw_result", None)),
                    },
                )
                raise
        bindings.append(binding)
    receipt = {"identity": identity, "count": count, "chunks": bindings}
    _publish(root / "COMPLETE.json", receipt)
    return receipt


def _rows(receipt):
    for chunk in receipt["chunks"]:
        yield from _chunk_rows(chunk["path"])


def _backend(runtime):
    from ..modeling_v3.vlm_observation import PrefixObservationBackend

    return PrefixObservationBackend(
        runtime["adapter"],
        runtime_identity=runtime["identity"],
        policy_loader=runtime["policy_loader"],
        data_root=runtime["data_root"],
        parity_tolerances=runtime["parity_tolerances"],
        state_guard=runtime["state_guard"],
    )


def collect_actions(
    runtime,
    policy,
    prompts,
    *,
    origin_id,
    role,
    draws,
    out,
    resume=False,
    proposal="ORIGIN",
    other_policy=None,
    namespace=None,
    chunk_size=32,
    interrupt_after_first_commit=False,
    prefix=None,
    checkpoint_retention="IMMUTABLE",
):
    backend = _backend(runtime)
    namespace = namespace or f"{NAMESPACE}:{origin_id}:{role}:{proposal}"
    identity = {
        "origin_id": origin_id,
        "role": role,
        "draws": draws,
        "proposal": proposal,
        "policies": [policy] + ([other_policy] if other_policy else []),
        "prompt_hash": canonical_hash(prompts),
        "rng_namespace": namespace,
        "prompt_ids": [p["prompt_id"] for p in prompts],
        "runtime": runtime["identity"],
    }
    if checkpoint_retention == "OPERATIONAL_UNTIL_NEXT_COMMIT":
        identity["policies"] = [
            {k: v for k, v in p.items() if k != "checkpoint"}
            | {
                "checkpoint_commitment": {k: v for k, v in p["checkpoint"].items() if k != "path"},
                "checkpoint_retention": checkpoint_retention,
            }
            for p in identity["policies"]
        ]
    elif checkpoint_retention != "IMMUTABLE":
        raise ValueError("Unknown checkpoint retention contract")

    def reuse(start, stop):
        if prefix is None:
            return None
        old = prefix["identity"]
        pindex, draw = divmod(start, draws)
        pid = prompts[pindex]["prompt_id"]
        if pid not in old["prompt_ids"] or draw + (stop - start) > old["draws"]:
            return None
        oldstart = old["prompt_ids"].index(pid) * old["draws"] + draw
        return next(
            (
                c
                for c in prefix["chunks"]
                if c["start"] == oldstart and c["stop"] == oldstart + (stop - start)
            ),
            None,
        )

    if prefix is not None:
        old = prefix["identity"]
        if any(
            old[k] != identity[k]
            for k in ("origin_id", "role", "proposal", "policies", "rng_namespace", "runtime")
        ):
            raise ValueError("Nested action prefix policy/RNG/runtime changed")
        current_by_id = {p["prompt_id"]: p for p in prompts}
        if (
            old["draws"] > draws
            or not set(old["prompt_ids"]) <= current_by_id.keys()
            or old["prompt_hash"] != canonical_hash([current_by_id[p] for p in old["prompt_ids"]])
        ):
            raise ValueError("Nested action prefix prompt/draw scope changed")
        verify_artifact_bindings(prefix)
        identity["reused_prefix_identity"] = canonical_hash(old)

    def produce(i):
        prompt = prompts[i // draws]
        draw = i % draws
        key = canonical_hash([namespace, prompt["prompt_id"], draw])
        seed = int(key[:16], 16) % (2**63)
        chosen = policy
        if other_policy is not None:
            chosen = (policy, other_policy)[int(canonical_hash([key, "mixture_source"])[0], 16) % 2]
        backend.activate(chosen)
        value = backend.generate(prompt, seed=seed, max_new_tokens=64)
        prepared = backend._prepared(prompt)
        row = {
            **value,
            "sample_key": key,
            "sample_id": key,
            "origin_id": origin_id,
            "prompt_id": prompt["prompt_id"],
            "base_scene_id": prompt["base_scene_id"],
            "role": role,
            "rng_namespace": namespace,
            "draw_index": draw,
            "proposal": proposal,
            "sample_seed": seed,
            "runtime_identity": canonical_hash(runtime["identity"]),
            "proposal_policy_id": chosen["candidate_id"],
            "proposal_fingerprint": chosen["inference_fingerprint"],
            "action_mask": [True] * len(value["token_ids"]),
            "max_new_tokens": 64,
            "eos_seen": value["eos"],
            "input_hash": prepared["audit"]["input_tensor_hash"],
            "input_tensor_hash": prepared["audit"]["input_tensor_hash"],
            "token_mask": [True] * len(value["token_ids"]),
            "shared_token_identity": canonical_hash(value["token_ids"]),
            "final_prompt_hash": prepared["audit"]["final_prompt_hash"],
        }
        if proposal in ("ORIGIN", "DIRECT"):
            row["proposal_sequence_logp"] = row["generation_sequence_logp"]
        from .functional_features import output_diagnostics

        row["behavior_diagnostics"] = output_diagnostics(row["raw_completion"], prompt["scene"])
        return row

    result = _chunks(
        out,
        identity,
        produce,
        count=len(prompts) * draws,
        resume=resume,
        chunk_size=chunk_size,
        interrupt_after_first_commit=interrupt_after_first_commit,
        reuse_chunk=reuse,
    )
    return {**result, "runtime_counters_this_invocation": backend.counters}


def collect_scores(runtime, policy, prompts, samples, *, out, resume=False, prefix=None):
    verify_artifact_bindings(samples)
    backend = _backend(runtime)
    backend.activate(policy)
    by_id = {p["prompt_id"]: p for p in prompts}
    loaded = {"index": None, "rows": None}

    def sample_at(i):
        import bisect

        index = bisect.bisect_right([c["start"] for c in samples["chunks"]], i) - 1
        if loaded["index"] != index:
            loaded.update(index=index, rows=_chunk_rows(samples["chunks"][index]["path"]))
        return loaded["rows"][i - samples["chunks"][index]["start"]]

    identity = {
        "sample_identity": canonical_hash(samples["identity"]),
        "sample_chunks": samples["chunks"],
        "policy": policy,
        "runtime": runtime["identity"],
    }
    shared = {}
    if prefix is not None:
        old = prefix["identity"]
        if (
            old["runtime"] != runtime["identity"]
            or old["policy"]["inference_fingerprint"] != policy["inference_fingerprint"]
        ):
            raise ValueError("Nested score prefix policy/runtime changed")
        verify_artifact_bindings(prefix)
        scores_by_range = {(c["start"], c["stop"]): c for c in prefix["chunks"]}
        shared = {
            (c["path"], c["sha256"]): scores_by_range[c["start"], c["stop"]]
            for c in old["sample_chunks"]
            if (c["start"], c["stop"]) in scores_by_range
        }
        identity["reused_prefix_identity"] = canonical_hash(old)
    sources_by_range = {(c["start"], c["stop"]): c for c in samples["chunks"]}

    def reuse(start, stop):
        source = sources_by_range.get((start, stop))
        return shared.get((source["path"], source["sha256"])) if source else None

    def produce(i):
        sample = sample_at(i)
        mode = runtime.get("observation_score_mode", "uncached_prefix_recompute")
        if mode == "uncached_prefix_recompute":
            value = backend.score(by_id[sample["prompt_id"]], sample)
        else:
            import torch

            prompt = by_id[sample["prompt_id"]]
            prepared = backend._prepared(prompt)
            if prepared["audit"]["input_tensor_hash"] != sample["input_hash"]:
                raise ValueError("Fast scoring input differs from sampled input")
            started = time.perf_counter()
            before = runtime["adapter"].forward_calls
            with backend._measurement(), torch.no_grad():
                _, scores = runtime["adapter"].continuation_scores(
                    prepared, sample["token_ids"], mode=mode
                )
            if len(scores) != len(sample["token_ids"]) or not all(
                __import__("math").isfinite(v) for v in scores
            ):
                raise ValueError("Fast scoring returned invalid path probabilities")
            backend.counters["scored_sequences"] += 1
            backend.counters["scored_tokens"] += len(scores)
            value = {
                "token_ids": sample["token_ids"],
                "token_logprobs": scores,
                "sequence_logp": __import__("math").fsum(scores),
                "probability_execution": mode,
                "score_path_certificate": canonical_hash(runtime["score_path_certificate"]),
                "elapsed_seconds": time.perf_counter() - started,
                "forward_calls": runtime["adapter"].forward_calls - before,
                "runtime_identity": runtime["identity"],
            }
        if value["token_ids"] != sample["token_ids"]:
            raise ValueError("Scored actions differ from shared original tokens")
        # All policy rows point to the one shared token original. Their actual
        # token log probabilities remain separate physical measurements.
        value = {k: v for k, v in value.items() if k != "token_ids"}
        return {
            **value,
            "shared_token_identity": sample["shared_token_identity"],
            "sample_id": sample["sample_id"],
            "sample_key": sample["sample_key"],
            "prompt_id": sample["prompt_id"],
            "candidate_id": policy["candidate_id"],
            "inference_fingerprint": policy["inference_fingerprint"],
            "input_hash": sample["input_hash"],
            "role": sample["role"],
            "rng_namespace": sample["rng_namespace"],
            "runtime_identity": canonical_hash(runtime["identity"]),
        }

    result = _chunks(
        out, identity, produce, count=samples["count"], resume=resume, reuse_chunk=reuse
    )
    return {**result, "runtime_counters_this_invocation": backend.counters}


def measure_scoring_paths(runtime, policy, prompts, null_actions):
    """Actual eval-only full/chunk/token parity, always against prefix recompute.

    No derivative certificate is inferred. This does not change generation or
    Adam's probability path. Failed alternatives remain explicit in the receipt.
    """
    import math

    import torch

    backend = _backend(runtime)
    backend.activate(policy)
    by_id = {p["prompt_id"]: p for p in prompts}
    measured = []
    reference = []
    for action in null_actions:
        prepared = backend._prepared(by_id[action["prompt_id"]])
        reference.append(
            runtime["adapter"]
            .logprobs(prepared, action["token_ids"], require_grad=False)
            .detach()
            .cpu()
            .double()
            .tolist()
        )
    tolerances = runtime["parity_tolerances"]
    if hasattr(runtime["adapter"], "continuation_scores"):
        for mode in ("full", "chunk", "token"):
            started = time.perf_counter()
            errors = []
            sequences = []
            failure = None
            try:
                for action, expected in zip(null_actions, reference, strict=True):
                    prepared = backend._prepared(by_id[action["prompt_id"]])
                    with torch.no_grad():
                        _, actual = runtime["adapter"].continuation_scores(
                            prepared, action["token_ids"], mode=mode
                        )
                    if len(actual) != len(expected) or not all(math.isfinite(x) for x in actual):
                        raise ValueError("Invalid fast-path log probabilities")
                    errors.extend(abs(a - b) for a, b in zip(actual, expected, strict=True))
                    sequences.append(abs(math.fsum(actual) - math.fsum(expected)))
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                failure = {"type": type(exc).__name__, "error": str(exc)}
            values = {
                "mean_abs_token_logp": math.fsum(errors) / max(1, len(errors)),
                "max_abs_token_logp": max(errors, default=float("inf")),
                "max_abs_sequence_logp": max(sequences, default=float("inf")),
            }
            passed = failure is None and all(values[k] <= tolerances[k] for k in values)
            measured.append(
                {
                    "mode": mode,
                    "passed": passed,
                    "errors": {k: v if math.isfinite(v) else None for k, v in values.items()},
                    "failure": failure,
                    "seconds": time.perf_counter() - started,
                }
            )
    usable = [v for v in measured if v["passed"]]
    selected = (
        min(usable, key=lambda x: x["seconds"])["mode"] if usable else "uncached_prefix_recompute"
    )
    receipt = {
        "reference_path": "uncached_prefix_recompute",
        "mode": selected,
        "passed": True,
        "scope": "evaluation_only_no_derivative_certification",
        "tested_sequences": len(null_actions),
        "inference_fingerprint": policy["inference_fingerprint"],
        "alternatives": measured,
        "tolerances": tolerances,
        "fast_path_available": bool(usable),
        "microbatch": 1,
    }
    runtime["observation_score_mode"] = selected
    runtime["score_path_certificate"] = receipt
    return receipt


def _save_policy(runtime, state, path, identity, *, full=False):
    from ..optimizer_fork import state_hash

    # Candidate LoRA/forward state is stored once. Adam/RNG state hashes remain
    # distinct even for exact inference aliases. Only selected bridge forks and
    # source origins/latest retain full Adam state for replay.
    saved = (
        state
        if full
        else {
            k: state[k]
            for k in (
                "parameters",
                "buffers",
                "module_modes",
                "position_state",
                "adapter_position_state",
                "metadata",
            )
        }
    )
    spec = _save_state(path, saved, identity)
    return {
        "checkpoint": spec,
        "inference_fingerprint": _fingerprint(state, runtime["identity"]),
        "candidate_id": identity["candidate_id"],
        "training_state_hash": state_hash(state),
        "optimizer_state_hash": state_hash(state["optimizer"]),
        "complete_training_state_retained": full,
        **_policy_layout(runtime),
    }


def _training_groups(runtime, prompts, receipt):
    """The behavior/rescore parity was measured when each action was collected."""
    from ..followup_runtime import _prepared

    groups = []
    rows = list(_rows(receipt))
    for prompt in prompts:
        prepared = _prepared(runtime["adapter"], prompt, runtime["data_root"])
        selected = [r for r in rows if r["prompt_id"] == prompt["prompt_id"]]
        if len(selected) != 8:
            raise ValueError("Natural training bank requires K8 per prompt")
        groups.append(
            [
                {
                    **r,
                    "prepared": prepared,
                    "old_logprobs": r["behavior_token_logprobs"],
                    "split": "train",
                }
                for r in selected
            ]
        )
    if len(groups) != 4:
        raise ValueError("Natural training bank requires B4")
    return groups


def _update(runtime, groups, spec):
    from ..followup_updates import apply_gradient_update, direct_loss_gradients

    runtime["state"].check_frozen()
    gradients = direct_loss_gradients(
        runtime["adapter"], groups, policy=spec["policy"], auxiliary_weight=spec["auxiliary_weight"]
    )
    update = apply_gradient_update(
        runtime["adapter"].model, runtime["optimizer"], gradients["gradients"]
    )
    cost = runtime.setdefault("cost", {"adam_updates": 0, "backward_calls": 0})
    cost["adam_updates"] += 1
    cost["backward_calls"] += sum(map(len, groups))
    runtime["state"].check_frozen()
    return {"gradient": gradients["audit"], "update": update}


def run_forks(
    runtime, origin, bank_plan, train_prompts, *, out, resume=False, bridge=False, prefix=None
):
    from ..optimizer_fork import state_hash

    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    _publish(root / "bank_plan.json", bank_plan)
    origin_id = bank_plan["origin_id"]
    by_id = {p["prompt_id"]: p for p in train_prompts}
    origin_file = root / "origin_policy.json"
    if origin_file.exists():
        origin_policy = _read(origin_file)
    elif prefix is not None:
        verify_artifact_bindings(prefix)
        if prefix["origin_id"] != origin_id:
            raise ValueError("Nested fork origin changed")
        origin_policy = prefix["origin_policy"]
        if origin_policy["inference_fingerprint"] != _fingerprint(origin, runtime["identity"]):
            raise ValueError("Nested fork parameters differ from source origin")
        _publish(origin_file, origin_policy)
    else:
        origin_policy = _save_policy(
            runtime,
            origin,
            root / "origin.pt",
            {"origin_id": origin_id, "candidate_id": "origin", **runtime["identity"]},
            full=True,
        )
        _publish(origin_file, origin_policy)
    banks = []
    try:
        for bank in bank_plan["banks"]:
            bdir = root / bank["bank_id"]
            done = bdir / "COMPLETE.json"
            shared = (
                next((b for b in prefix["banks"] if b["bank_id"] == bank["bank_id"]), None)
                if prefix
                else None
            )
            if shared is not None and not done.exists():
                if shared["prompt_ids"] != bank["prompt_ids"] or shared["role"] != bank["role"]:
                    raise ValueError("Nested bank plan changed")
                _publish(done, shared)
                banks.append(shared)
                continue
            if done.exists():
                if not resume:
                    raise FileExistsError(done)
                existing = _read(done)
                verify_artifact_bindings(existing)
                banks.append(existing)
                continue
            prompts = [by_id[p] for p in bank["prompt_ids"]]
            runtime["state"].restore(origin)
            samples = collect_actions(
                runtime,
                origin_policy,
                prompts,
                origin_id=origin_id,
                role="train",
                draws=8,
                out=bdir / "samples",
                resume=resume,
                namespace=f"{NAMESPACE}:{origin_id}:{bank['bank_id']}:train",
            )
            groups = _training_groups(runtime, prompts, samples)
            policies = {}
            parameters = {}
            post_states = {}
            for spec in CANDIDATES:
                runtime["state"].restore(origin)
                record_path = bdir / (spec["id"] + ".json")
                if record_path.exists():
                    if not resume:
                        raise FileExistsError(record_path)
                    record = _read(record_path)
                    post = runtime["checkpoint_cache"].load(record["policy"])
                else:
                    try:
                        started = time.perf_counter()
                        audit = _update(runtime, groups, spec)
                        post = runtime["state"].capture(
                            {
                                **origin["metadata"],
                                "candidate": spec,
                                "permanent_training_commit": False,
                            }
                        )
                        policy = _save_policy(
                            runtime,
                            post,
                            bdir / (spec["id"] + ".pt"),
                            {
                                **runtime["identity"],
                                "origin_id": origin_id,
                                "bank_id": bank["bank_id"],
                                "candidate_id": bank["bank_id"] + "_" + spec["id"],
                            },
                            full=bridge,
                        )
                        record = {
                            "policy": policy,
                            "audit": audit,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                        _publish(record_path, record)
                    finally:
                        runtime["state"].restore(origin)
                policies[spec["id"]] = record["policy"]
                parameters[spec["id"]] = post["parameters"]
                post_states[spec["id"]] = post
            # Exact inference aliases keep their distinct actual optimizer/RNG
            # originals, content-deduplicated independently of inference identity.
            aliases = {
                name
                for name, policy in policies.items()
                if sum(
                    p["inference_fingerprint"] == policy["inference_fingerprint"]
                    for p in policies.values()
                )
                > 1
            }
            for name in sorted(aliases):
                state = post_states[name]
                if "optimizer" not in state:
                    runtime["state"].restore(origin)
                    _update(runtime, groups, next(s for s in CANDIDATES if s["id"] == name))
                    state = runtime["state"].capture(
                        {
                            **origin["metadata"],
                            "candidate": next(s for s in CANDIDATES if s["id"] == name),
                            "permanent_training_commit": False,
                        }
                    )
                digest = state_hash(state)
                if digest != policies[name]["training_state_hash"]:
                    raise ValueError("Alias optimizer replay differs from original state")
                if policies[name]["complete_training_state_retained"]:
                    spec = policies[name]["checkpoint"]
                else:
                    # Candidate labels are separately retained metadata. Exact
                    # equal Adam/RNG/tensor contents share one physical original.
                    stored = {**state, "metadata": {"kind": "content_addressed_alias_state"}}
                    content_hash = state_hash(stored)
                    target = root / "alias_training_states" / (content_hash + ".pt")
                    if not target.exists():
                        from .storage import require_space

                        measured = require_space(
                            root, int(_tensor_bytes(stored) * 1.1) + 2 * 1024**2
                        )
                        runtime.setdefault("alias_storage_checks", []).append(measured)
                    spec = _save_state(
                        target,
                        stored,
                        {"kind": "V4_ALIAS_FULL_TRAINING_STATE", "state_hash": content_hash},
                    )
                    spec = {
                        **spec,
                        "logical_metadata": state["metadata"],
                        "logical_state_hash": digest,
                    }
                policies[name] = {
                    **policies[name],
                    "alias_training_state": spec,
                    "complete_training_state_retained": True,
                }
            vectors = {}
            for name, left, right in CONTRASTS:
                difference = {
                    n: parameters[left][n].double() - parameters[right][n].double()
                    for n in sorted(parameters[left])
                }
                # No duplicate delta tensor file: exact differences are defined
                # by the two immutable policy payloads and loaded lazily.
                norm = sum(float(v.square().sum()) for v in difference.values()) ** 0.5
                vectors[name] = {
                    "left": policies[left]["checkpoint"],
                    "right": policies[right]["checkpoint"],
                    "parameter_order": list(difference),
                    "norm": norm,
                    "delta_hash": state_hash(difference),
                    "exact_inference_alias": policies[left]["inference_fingerprint"]
                    == policies[right]["inference_fingerprint"],
                }
            record = {
                **bank,
                "policies": policies,
                "contrasts": vectors,
                "origin_state_restored": True,
                "storage_policy": (
                    "Alias endpoints retain distinct complete Adam/RNG state bindings; "
                    "nonalias LoRA stores retain origin + natural train samples + "
                    "update audit for reconstruction."
                ),
                "train_samples": {
                    k: v for k, v in samples.items() if k != "runtime_counters_this_invocation"
                },
            }
            _publish(done, record)
            banks.append(record)
    finally:
        runtime["state"].restore(origin)
    result = {
        "origin_id": origin_id,
        "origin_policy": origin_policy,
        "banks": banks,
        "full_origin_shared": True,
    }
    _publish(root / "COMPLETE.json", result)
    return result


def _replace_json(path, value):
    """Only operational cursors use replace; scientific shards are immutable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".cursor-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def run_source_trajectory(
    config,
    runtime,
    train_prompts,
    *,
    seed,
    arm,
    out,
    resume=False,
    steps=128,
    fixture=False,
    bridge_fallback=False,
):
    """Actual on-policy B4/K8 Adam; full origins/latest, light dev64..80 states."""
    from ..followup_updates import _restore_forward
    from ..modeling_v3.vlm_campaign import initialize_lora

    if not fixture and steps != 128 and not (bridge_fallback and steps == 32):
        raise ValueError("Production source needs128 steps or explicit bridge fallback stop32")
    if arm not in ("X_BASE", "X_VALID"):
        raise ValueError("Unknown source arm")
    root = Path(out)
    done = root / "COMPLETE.json"
    retained_steps = (
        (32, 64, 96)
        if arm == "X_BASE" and seed in config["qwen"]["seed_roles"]["development"]
        else (32, 96)
    )
    if done.exists():
        if not resume:
            raise FileExistsError(done)
        result = _read(done)
        verify_artifact_bindings(result)
        return result
    root.mkdir(parents=True, exist_ok=True)
    schedule = build_training_schedule(train_prompts, seed)
    _publish(root / "schedule.json", schedule)
    by_id = {p["prompt_id"]: p for p in train_prompts}
    if (root / "LATEST.json").exists():
        if not resume:
            raise FileExistsError(root)
        latest = _read(root / "LATEST.json")
        state = runtime["checkpoint_cache"].load(latest["checkpoint"])
        start = latest["step"]
    else:
        runtime["optimizer"].state.clear()
        runtime["sampler"].clear()
        _restore_forward(runtime["adapter"].model, runtime["initial_forward"], runtime["adapter"])
        if fixture:
            import torch

            torch.manual_seed(seed)
            with torch.no_grad():
                for p in runtime["adapter"].model.parameters():
                    if p.requires_grad:
                        p.zero_()
            initialization = {"seed": seed, "fixture": True}
        else:
            initialization = initialize_lora(runtime["adapter"], runtime["optimizer"], seed)
        state = runtime["state"].capture(
            {"seed": seed, "arm": arm, "checkpoint_step": 0, "initialization": initialization}
        )
        initial = _save_state(
            root / "initial.pt",
            state,
            {
                **runtime["identity"],
                "candidate_id": "source_initial",
                "seed": seed,
                "arm": arm,
                "step": 0,
            },
        )
        latest = {"step": 0, "checkpoint": initial}
        _replace_json(root / "LATEST.json", latest)
        start = 0
    selected = {}
    for meta in (root / "origins").glob("*.json") if (root / "origins").exists() else []:
        row = _read(meta)
        selected[str(row["step"])] = row["checkpoint"]
    for step in range(start + 1, steps + 1):
        sdir = root / "steps" / f"step_{step:03d}"
        # UPDATE is the durable commit, LATEST is only a recoverable cursor.
        if (sdir / "UPDATE.json").exists():
            committed = _read(sdir / "UPDATE.json")
            verify_artifact_bindings(committed)
            state = runtime["checkpoint_cache"].load(committed["checkpoint"])
            latest = {"step": step, "checkpoint": committed["checkpoint"]}
            _replace_json(root / "LATEST.json", latest)
            if step in retained_steps:
                selected[str(step)] = committed["checkpoint"]
            continue
        runtime["state"].restore(state)
        identity = {
            **runtime["identity"],
            "seed": seed,
            "arm": arm,
            "step": step - 1,
            "candidate_id": "source",
        }
        policy = {
            "checkpoint": latest["checkpoint"],
            "candidate_id": "source",
            "inference_fingerprint": _fingerprint(state, runtime["identity"]),
        }
        panel = [by_id[p] for p in schedule["train_steps"][step - 1]]
        samples = collect_actions(
            runtime,
            policy,
            panel,
            origin_id=f"source_{seed}_{arm}_{step - 1}",
            role="train",
            draws=8,
            out=sdir / "samples",
            resume=resume,
            checkpoint_retention="OPERATIONAL_UNTIL_NEXT_COMMIT",
        )
        groups = _training_groups(runtime, panel, samples)
        runtime["state"].restore(state)
        try:
            started = time.perf_counter()
            audit = _update(
                runtime, groups, {"policy": "joint", "auxiliary_weight": float(arm == "X_VALID")}
            )
            runtime["sampler"].update(
                step=step, position=4 * step, schedule_hash=schedule["schedule_hash"]
            )
            post = runtime["state"].capture({**state["metadata"], "checkpoint_step": step})
            checkpoint = _save_state(
                root / "resume_states" / f"step_{step:03d}.pt",
                post,
                {**identity, "step": step, "candidate_id": "source_after_update"},
            )
            if step in retained_steps:
                selected[str(step)] = checkpoint
                _publish(
                    root / "origins" / f"step_{step:03d}.json",
                    {"step": step, "checkpoint": checkpoint},
                )
            if 64 <= step <= 80 and seed in config["qwen"]["seed_roles"]["development"]:
                light = _save_policy(
                    runtime,
                    post,
                    root / "tracking" / f"step_{step:03d}.pt",
                    {**identity, "step": step, "candidate_id": f"tracking_{step}"},
                )
                _publish(root / "tracking" / f"step_{step:03d}.json", light)
            _publish(
                sdir / "UPDATE.json",
                {
                    "kind": "V4_SOURCE_STEP_UPDATE",
                    "step": step,
                    "audit": audit,
                    "elapsed_seconds": time.perf_counter() - started,
                    "state_hash": checkpoint["state_hash"],
                    "checkpoint": checkpoint,
                    "retained_origin": step in retained_steps,
                    "checkpoint_retention": "IMMUTABLE"
                    if step in retained_steps
                    else "OPERATIONAL_UNTIL_NEXT_COMMIT",
                },
            )
            previous = latest
            latest = {"step": step, "checkpoint": checkpoint}
            _replace_json(root / "LATEST.json", latest)
            # These are replaceable operational resume slots, never samples,
            # origins, tracking states, or failed scientific attempts.
            if previous["step"] not in (0, *retained_steps):
                Path(previous["checkpoint"]["path"]).unlink()
            state = post
        except BaseException:
            runtime["state"].restore(state)
            raise
    result = {
        "kind": "source",
        "seed": seed,
        "arm": arm,
        "steps": steps,
        "origins": selected,
        "latest": latest,
        "initial_checkpoint": _binding(root / "initial.pt"),
        "training_originals": [
            {
                "step": i,
                "samples": _binding(root / "steps" / f"step_{i:03d}" / "samples" / "COMPLETE.json"),
                "update": _binding(root / "steps" / f"step_{i:03d}" / "UPDATE.json"),
            }
            for i in range(1, steps + 1)
        ],
        "tracking_states": [_binding(p) for p in sorted((root / "tracking").glob("step_*.json"))],
        "status": "COMPLETED" if steps == 128 or fixture else "PAUSED_AT_BRIDGE_ORIGIN",
        "config_hash": runtime["identity"]["config_hash"],
        "source_hash": runtime["identity"]["source_hash"],
        "execution_kind": runtime["identity"]["execution_kind"],
    }
    _publish(done if steps == 128 or fixture else root / "BRIDGE_ORIGIN_READY.json", result)
    return result


def collect_response_map(
    config,
    runtime,
    forks,
    probes,
    *,
    out,
    draws,
    reference_draws,
    bridge=False,
    resume=False,
    prefix=None,
):
    """Shared origin work/reference, all candidate work scores, heldout references.

    References are acquired to separate files. Fitting APIs receive only the
    work binding; completion never certifies reference precision or prediction.
    """
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    origin = forks["origin_id"]
    origin_policy = forks["origin_policy"]
    if prefix is not None and prefix["origin_id"] != origin:
        raise ValueError("Nested response origin changed")
    work = collect_actions(
        runtime,
        origin_policy,
        probes,
        origin_id=origin,
        role="work",
        draws=draws,
        out=root / "work",
        resume=resume,
        prefix=prefix["work"] if prefix else None,
    )
    ref = collect_actions(
        runtime,
        origin_policy,
        probes,
        origin_id=origin,
        role="reference",
        draws=reference_draws,
        out=root / "reference",
        resume=resume,
        prefix=prefix["reference"] if prefix else None,
    )
    direct_work = collect_actions(
        runtime,
        origin_policy,
        probes,
        origin_id=origin,
        role="direct_work",
        draws=draws,
        out=root / "direct_work",
        resume=resume,
        prefix=prefix["direct_work"] if prefix else None,
    )
    work_scores = {}
    reference_scores = {}
    direct_scores = {}
    cache = {}
    for bank in forks["banks"]:
        for policy in bank["policies"].values():
            fp = policy["inference_fingerprint"]
            for role, samples, target in (
                ("work", work, work_scores),
                ("reference", ref, reference_scores),
                ("direct_work", direct_work, direct_scores),
            ):
                if role != "work" and bank["role"] != "query" and not bridge:
                    continue
                key = (fp, role)
                if key not in cache:
                    prior = None
                    field = {
                        "work": "work_scores",
                        "reference": "reference_scores",
                        "direct_work": "direct_scores",
                    }[role]
                    if prefix:
                        prior = next(
                            (
                                v["score_receipt"]
                                for v in prefix[field].values()
                                if v["score_receipt"]["identity"]["policy"]["inference_fingerprint"]
                                == fp
                            ),
                            None,
                        )
                    cache[key] = collect_scores(
                        runtime,
                        policy,
                        probes,
                        samples,
                        out=root / "scores" / role / fp,
                        resume=resume,
                        prefix=prior,
                    )
                target[policy["candidate_id"]] = {
                    "score_receipt": {
                        k: v
                        for k, v in cache[key].items()
                        if k != "runtime_counters_this_invocation"
                    },
                    "alias_reuse": cache[key]["identity"]["policy"]["candidate_id"]
                    != policy["candidate_id"],
                }
    # Select nonalias pairs by actual trainable/inference identities only. The
    # query response outcomes are never read to choose these mandatory checks.
    eligible = [
        b
        for b in forks["banks"]
        if (bridge or b["role"] == "query")
        and not b["contrasts"][CONTRASTS[0][0]]["exact_inference_alias"]
        and b["contrasts"][CONTRASTS[0][0]]["norm"] > 0
    ]
    overlap = _reference_overlap(ref, reference_scores, runtime.get("analysis_rules"))
    checks = []
    mandatory = {b["bank_id"] for b in eligible[:2]}
    for bank in forks["banks"]:
        if not bridge and bank["role"] != "query":
            continue
        for contrast_id, left_name, right_name in CONTRASTS:
            if bank["contrasts"][contrast_id]["exact_inference_alias"]:
                continue
            left, right = (bank["policies"][v] for v in (left_name, right_name))
            is_mandatory = bank["bank_id"] in mandatory and contrast_id == CONTRASTS[0][0]
            bad = {
                p["prompt_id"]
                for p in probes
                if not all(
                    overlap[v["candidate_id"]][p["prompt_id"]]["usable"] for v in (left, right)
                )
            }
            prior = (
                next(
                    (
                        c
                        for c in prefix["pair_checks"]
                        if c["bank_id"] == bank["bank_id"] and c["contrast_id"] == contrast_id
                    ),
                    None,
                )
                if prefix
                else None
            )
            needed = set(p["prompt_id"] for p in probes) if is_mandatory else set(bad)
            if prior:
                needed.update(prior["mixture"]["identity"]["prompt_ids"])
            if not needed:
                continue
            selected_probes = [p for p in probes if p["prompt_id"] in needed]
            pairdir = root / "checks" / bank["bank_id"] / contrast_id
            mixture = collect_actions(
                runtime,
                left,
                selected_probes,
                origin_id=origin,
                role="reference",
                draws=reference_draws,
                out=pairdir / "mixture",
                resume=resume,
                proposal="MIX",
                other_policy=right,
                namespace=f"{NAMESPACE}:{origin}:{bank['bank_id']}:{contrast_id}:reference:MIX",
                prefix=prior["mixture"] if prior else None,
            )
            mixscores = {
                p["candidate_id"]: collect_scores(
                    runtime,
                    p,
                    selected_probes,
                    mixture,
                    out=pairdir / "mixture_scores" / p["candidate_id"],
                    resume=resume,
                    prefix=prior["mixture_scores"].get(p["candidate_id"]) if prior else None,
                )
                for p in (left, right)
            }
            direct = (
                {
                    p["candidate_id"]: collect_actions(
                        runtime,
                        p,
                        probes,
                        origin_id=origin,
                        role="direct",
                        draws=draws,
                        out=pairdir / "direct" / p["candidate_id"],
                        resume=resume,
                        proposal="DIRECT",
                        namespace=f"{NAMESPACE}:{origin}:{p['candidate_id']}:direct",
                        prefix=prior["direct"].get(p["candidate_id"]) if prior else None,
                    )
                    for p in (left, right)
                }
                if is_mandatory
                else {}
            )
            checks.append(
                {
                    "bank_id": bank["bank_id"],
                    "contrast_id": contrast_id,
                    "left_candidate_id": left["candidate_id"],
                    "right_candidate_id": right["candidate_id"],
                    "switch_reason": "MANDATORY_FIRST_TWO_PRIMARY"
                    if is_mandatory
                    else "FROZEN_ORIGIN_OVERLAP_SCREEN",
                    "poor_origin_prompt_ids": sorted(bad),
                    "mixture": mixture,
                    "mixture_scores": mixscores,
                    "direct": direct,
                }
            )
    result = {
        "origin_id": origin,
        "work": work,
        "reference": ref,
        "work_scores": work_scores,
        "reference_scores": reference_scores,
        "pair_checks": checks,
        "reference_overlap": overlap,
        "mixture_switch_pairs": sum(
            c["switch_reason"] == "FROZEN_ORIGIN_OVERLAP_SCREEN" for c in checks
        ),
        "mixture_generated_reference_draws": sum(c["mixture"]["count"] for c in checks),
        "direct_work": direct_work,
        "direct_scores": direct_scores,
        "direct_baseline": "DIRECT_LR_ORIGIN_INDEPENDENT_PACKET",
        "count_baseline_scope": "D_COUNT_TWO_ACTIVE_PAIR_DIAGNOSTICS",
        "reference_status": "MEASURED_PRECISION_NOT_YET_EVALUATED",
        "scientific_status": "NOT_CERTIFIED",
        "active_pairs_available": len(eligible),
        "all_queries_retained": True,
        "policies": {p["candidate_id"]: p for b in forks["banks"] for p in b["policies"].values()},
    }

    # Invocation-only counters vary on resume; permanent receipt contains the
    # immutable shard identities. Costs stay in worker attempt records.
    def stable(v):
        if isinstance(v, dict):
            return {k: stable(x) for k, x in v.items() if k != "runtime_counters_this_invocation"}
        if isinstance(v, list):
            return [stable(x) for x in v]
        return v

    result = stable(result)
    _publish(root / "COMPLETE.json", result)
    return result


def _reference_overlap(reference, score_receipts, rules):
    from collections import defaultdict

    import numpy as np

    from .analysis_rules import endpoint_overlap

    proposal = {
        r["sample_key"]: (r["prompt_id"], r["generation_sequence_logp"]) for r in _rows(reference)
    }
    output = {}
    cache = {}
    for candidate, value in score_receipts.items():
        receipt = value["score_receipt"]
        fp = receipt["identity"]["policy"]["inference_fingerprint"]
        if fp not in cache:
            weights = defaultdict(list)
            for row in _rows(receipt):
                pid, logp = proposal[row["sample_key"]]
                weights[pid].append(row["sequence_logp"] - logp)
            report = {}
            for pid, values in weights.items():
                if len(values) < 2 or not np.isfinite(values).all():
                    report[pid] = {
                        "usable": False,
                        "status": "NONFINITE_OR_INSUFFICIENT",
                        "draws": len(values),
                    }
                else:
                    result = endpoint_overlap(values, rules=rules)
                    report[pid] = {
                        "usable": bool(result["usable"]),
                        "draws": result["draws"],
                        "ess_fraction": float(result["ess_fraction"]),
                        "max_normalized_weight": float(result["max_normalized_weight"]),
                        "rules_hash": result["rules_hash"],
                        "accuracy_certified": False,
                    }
            cache[fp] = report
        output[candidate] = cache[fp]
    return output


def collect_nested_measurements(runtime, forks, probes, response, task, *, out, resume=False):
    """Predeclared draw curves and 8192 reference subset, sharing saved prefixes."""
    root = Path(out)
    origin = forks["origin_id"]
    policy = forks["origin_policy"]
    extra = {}

    def clean(value):
        if isinstance(value, dict):
            return {
                k: clean(v) for k, v in value.items() if k != "runtime_counters_this_invocation"
            }
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    def score_all(samples, bank_ids, old_scores, target):
        scores = {}
        cache = {}
        for bank in forks["banks"]:
            if bank_ids is not None and bank["bank_id"] not in bank_ids:
                continue
            for endpoint in bank["policies"].values():
                fp = endpoint["inference_fingerprint"]
                if fp not in cache:
                    previous = next(
                        (
                            s["score_receipt"]
                            for s in old_scores.values()
                            if s["score_receipt"]["identity"]["policy"]["inference_fingerprint"]
                            == fp
                        ),
                        None,
                    )
                    cache[fp] = collect_scores(
                        runtime,
                        endpoint,
                        probes,
                        samples,
                        out=target / fp,
                        resume=resume,
                        prefix=previous,
                    )
                scores[endpoint["candidate_id"]] = {
                    "score_receipt": clean(cache[fp]),
                    "alias_reuse": cache[fp]["identity"]["policy"]["candidate_id"]
                    != endpoint["candidate_id"],
                }
        return scores

    if task.get("n_curve_max_draws"):
        n = task["n_curve_max_draws"]
        if n <= response["work"]["identity"]["draws"]:
            raise ValueError("N-curve extension must add registered draws")
        raw = collect_actions(
            runtime,
            policy,
            probes,
            origin_id=origin,
            role="work",
            draws=n,
            out=root / "ncurve_work",
            resume=resume,
            prefix=response["work"],
        )
        extra["ncurve_work"] = clean(raw)
        extra["ncurve_scores"] = score_all(
            raw, None, response["work_scores"], root / "ncurve_scores"
        )
        direct = collect_actions(
            runtime,
            policy,
            probes,
            origin_id=origin,
            role="direct_work",
            draws=n,
            out=root / "ncurve_direct_work",
            resume=resume,
            prefix=response["direct_work"],
        )
        extra["ncurve_direct_work"] = clean(direct)
        extra["ncurve_direct_scores"] = score_all(
            direct,
            {b["bank_id"] for b in forks["banks"] if b["role"] == "query"},
            response["direct_scores"],
            root / "ncurve_direct_scores",
        )
        extra["ncurve_is_nested_prefix_not_independent_replicates"] = True
    validation = task.get("reference_validation")
    if validation:
        if validation["prompt_scope"] != "all_frozen_observation_prompts" or not set(
            validation["bank_ids"]
        ) <= {b["bank_id"] for b in forks["banks"] if b["role"] == "query"}:
            raise ValueError("Frozen reference validation scope changed")
        raw = collect_actions(
            runtime,
            policy,
            probes,
            origin_id=origin,
            role="reference",
            draws=validation["draws"],
            out=root / "reference_validation",
            resume=resume,
            prefix=response["reference"],
        )
        extra["reference_validation"] = clean(raw)
        extra["reference_validation_scores"] = score_all(
            raw,
            set(validation["bank_ids"]),
            response["reference_scores"],
            root / "reference_validation_scores",
        )
        extra["reference_validation_scope"] = validation
    result = {**response, **extra, "additional_measurements_planned_before_outcomes": True}
    _publish(root / "EXTENDED_RESPONSE.json", result)
    return result


def pair_observation_diagnostics(response):
    """Independent ORIGIN/MIX/DIRECT uncertainty; no method rankings are read."""
    import math
    from statistics import NormalDist

    import numpy as np

    reports = []
    reference = list(_rows(response["reference"]))

    def scores(receipt):
        return {r["sample_key"]: r for r in _rows(receipt)}

    for check in response["pair_checks"]:
        left, right = check["left_candidate_id"], check["right_candidate_id"]
        ls = scores(response["reference_scores"][left]["score_receipt"])
        rs = scores(response["reference_scores"][right]["score_receipt"])
        ml = scores(check["mixture_scores"][left])
        mr = scores(check["mixture_scores"][right])
        mix = list(_rows(check["mixture"]))
        direct = {p: list(_rows(check["direct"][p])) for p in check["direct"]}
        for pid in check["mixture"]["identity"]["prompt_ids"]:
            origin = []
            mixture = []
            for row in reference:
                if row["prompt_id"] != pid:
                    continue
                logl = ls[row["sample_key"]]["sequence_logp"] - row["generation_sequence_logp"]
                logr = rs[row["sample_key"]]["sequence_logp"] - row["generation_sequence_logp"]
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = (
                        np.exp(max(logl, logr))
                        * (-np.expm1(-abs(logl - logr)))
                        * (1 if logl >= logr else -1)
                    )
                origin.append(np.asarray(row["event_onehot"], float) * delta)
            for row in mix:
                if row["prompt_id"] != pid:
                    continue
                gap = (
                    ml[row["sample_key"]]["sequence_logp"] - mr[row["sample_key"]]["sequence_logp"]
                )
                mixture.append(np.asarray(row["event_onehot"], float) * (2 * math.tanh(gap / 2)))
            origin = np.asarray(origin)
            mixture = np.asarray(mixture)
            a = np.asarray(
                [r["event_onehot"] for r in direct.get(left, []) if r["prompt_id"] == pid], float
            )
            b = np.asarray(
                [r["event_onehot"] for r in direct.get(right, []) if r["prompt_id"] == pid], float
            )
            finite = all(
                np.isfinite(v).all() and len(v) > 1
                for v in ((origin, mixture, a, b) if direct else (origin, mixture))
            )
            if not finite:
                reports.append(
                    {
                        "bank_id": check["bank_id"],
                        "prompt_id": pid,
                        "status": "UNRESOLVED_NONFINITE_OR_INSUFFICIENT",
                    }
                )
                continue
            means = {"ORIGIN": origin.mean(0), "MIX": mixture.mean(0)}
            variances = {
                "ORIGIN": origin.var(0, ddof=1) / len(origin),
                "MIX": mixture.var(0, ddof=1) / len(mixture),
            }
            if direct:
                means["DIRECT"] = a.mean(0) - b.mean(0)
                variances["DIRECT"] = a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b)
            # Diagnostic simultaneous normal intervals, explicitly not exact
            # rare-event certification. Zero variances never imply precision.
            z = NormalDist().inv_cdf(
                1 - 0.01 / (8 * max(1, len({r["prompt_id"] for r in reference})))
            )
            comparisons = {}
            for name in ("MIX", "DIRECT") if direct else ("MIX",):
                se = np.sqrt(variances["ORIGIN"] + variances[name])
                gap = np.abs(means["ORIGIN"] - means[name])
                comparisons[name] = {
                    "difference": gap.tolist(),
                    "standard_error": se.tolist(),
                    "within_diagnostic_interval": (gap <= z * se + 1e-4).tolist(),
                }
            reports.append(
                {
                    "bank_id": check["bank_id"],
                    "prompt_id": pid,
                    "status": "MEASURED",
                    "means": {k: v.tolist() for k, v in means.items()},
                    "variance_of_mean": {k: v.tolist() for k, v in variances.items()},
                    "comparisons": comparisons,
                    "reference_precision": "UNRESOLVED_NOT_CERTIFIED",
                }
            )
    return {
        "units": reports,
        "all_finite": bool(reports) and all(r["status"] == "MEASURED" for r in reports),
        "scope": "INDEPENDENT_PROPOSAL_AND_COUNT_DIAGNOSTICS",
        "reference_precision": "NOT_CERTIFIED",
    }


def _direction_bindings(forks):
    return {
        f"{bank['bank_id']}/{name}": {"bank_id": bank["bank_id"], "role": bank["role"], **contrast}
        for bank in forks.get("banks", [])
        for name, contrast in bank["contrasts"].items()
    }


class _CheckpointDirections:
    """Bounded exact-coordinate products from disposable node-local scratch.

    Scratch is derived solely from immutable endpoint originals. It is neither
    a new original nor a persisted per-candidate parameter copy. Float64 keeps
    the exact subtraction used by the fork's recorded delta hash.
    """

    def __init__(self, runtime, bindings, scratch, *, block_coordinates=16384):
        import numpy as np

        from ..optimizer_fork import state_hash
        from .storage import require_space

        self.bindings = bindings
        self.ids = list(bindings)
        self.block_coordinates = block_coordinates
        self.layout = []
        offset = 0
        params = dict(runtime["adapter"].model.named_parameters())
        for name in runtime["parameter_order"]:
            shape = tuple(params[name].shape)
            size = int(np.prod(shape))
            self.layout.append(
                {"name": name, "shape": shape, "start": offset, "stop": offset + size}
            )
            offset += size
        self.storage = require_space(scratch, offset * len(self.ids) * 8)
        self.matrix = np.memmap(
            Path(scratch) / "derived_directions.f64",
            dtype=np.float64,
            mode="w+",
            shape=(offset, len(self.ids)),
        )
        for column, spec in enumerate(bindings.values()):
            if spec["parameter_order"] != runtime["parameter_order"]:
                raise ValueError("Direction parameter order differs from runtime")
            left = runtime["checkpoint_cache"].load(spec["left"])["parameters"]
            right = runtime["checkpoint_cache"].load(spec["right"])["parameters"]
            delta = {n: left[n].double() - right[n].double() for n in runtime["parameter_order"]}
            if state_hash(delta) != spec["delta_hash"]:
                raise ValueError("Actual endpoint difference differs from frozen fork delta")
            for item in self.layout:
                value = delta[item["name"]].numpy()
                if value.shape != item["shape"] or not np.isfinite(value).all():
                    raise ValueError("Actual direction shape or finiteness differs")
                self.matrix[item["start"] : item["stop"], column] = value.reshape(-1)
        self.matrix.flush()
        self.gram = np.zeros((len(self.ids), len(self.ids)), dtype=np.float64)
        for start in range(0, offset, self.block_coordinates):
            block = np.asarray(self.matrix[start : start + self.block_coordinates])
            self.gram += block.T @ block
        self.dot_calls = 0

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def validate(self, params):
        if list(params) != [item["name"] for item in self.layout] or any(
            tuple(params[item["name"]].shape) != item["shape"] for item in self.layout
        ):
            raise ValueError("Direction provider must cover every trainable parameter coordinate")

    def dot_all(self, gradients):
        import numpy as np

        if set(gradients) != {item["name"] for item in self.layout}:
            raise ValueError("Actual gradient must cover the complete direction layout")
        result = np.zeros(len(self.ids), dtype=np.float64)
        for item in self.layout:
            gradient = np.asarray(gradients[item["name"]], dtype=np.float64)
            if gradient.shape != item["shape"] or not np.isfinite(gradient).all():
                raise ValueError("Actual gradient shape or finiteness differs")
            vector = gradient.reshape(-1)
            for start in range(0, len(vector), self.block_coordinates):
                stop = min(len(vector), start + self.block_coordinates)
                block = np.asarray(self.matrix[item["start"] + start : item["start"] + stop])
                result += block.T @ vector[start:stop]
        self.dot_calls += 1
        return result

    def metadata(self):
        import numpy as np

        return {
            "direction_bindings": self.bindings,
            "raw_parameter_gram": self.gram,
            "raw_parameter_norms": np.sqrt(np.maximum(np.diag(self.gram), 0)),
            "direction_layout": self.layout,
            "direction_scratch": {
                "storage_preflight": self.storage,
                "dtype": "float64",
                "block_coordinates": self.block_coordinates,
                "persistent_parameter_copy": False,
                "removed_after_collection": True,
                "actual_gradient_contraction_calls": self.dot_calls,
            },
        }

    def close(self):
        self.matrix._mmap.close()


def _score_derivative(runtime, forks, probes, response, *, out, bridge=False, prompt_subset=()):
    import torch

    from ..optimizer_fork import state_hash
    from .score_response import collect_score_response, score_sample_gradient

    root = Path(out)
    done = root / "COMPLETE.json"
    path = root / "score_response.pt"
    directions = {} if bridge else _direction_bindings(forks)
    input_hash = canonical_hash(
        {
            "policy": forks["origin_policy"],
            "probes": probes,
            "work_identity": response["work"]["identity"],
            "work_chunks": response["work"]["chunks"],
            "bridge": bridge,
            "prompt_subset": list(prompt_subset),
            "direction_bindings": directions,
        }
    )
    if done.exists():
        result = _read(done)
        if result.get("input_hash") != input_hash:
            raise ValueError("Derivative input identity changed")
        verify_artifact_bindings(result)
        return result
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        digest = state_hash(payload)
        prepared = [_read(p) for p in root.glob("gradient_prepared_*.json")]
        match = next(
            (p for p in prepared if p["input_hash"] == input_hash and p["payload_hash"] == digest),
            None,
        )
        if match is None:
            raise ValueError("Orphan gradient has no matching prepared content binding")
        receipt = {**match["receipt"], "artifact": _binding(path)}
        _publish(done, receipt)
        return receipt
    runtime["policy_loader"](forks["origin_policy"])
    samples = [r for r in _rows(response["work"]) if r["draw_index"] < (1 if bridge else 32)]
    # The original main stream is separate from reference; adapting its role
    # does not manufacture independent samples or change the sample identity.
    samples = [{**r, "role": "main"} for r in samples]
    certificate = _measure_derivative_path(
        runtime,
        forks,
        probes[0],
        next(r for r in samples if r["prompt_id"] == probes[0]["prompt_id"]),
    )
    if bridge:
        result = score_sample_gradient(
            runtime, probes[0], next(r for r in samples if r["prompt_id"] == probes[0]["prompt_id"])
        )
    else:
        if directions:
            # Node-local TMPDIR by default, with an explicit alternate scratch
            # directory available when a node's temporary disk is too small.
            scratch_parent = os.environ.get("SSVC_V4_SCRATCH_DIR")
            with tempfile.TemporaryDirectory(
                prefix="ssvc-v4-directions-", dir=scratch_parent
            ) as scratch:
                provider = _CheckpointDirections(runtime, directions, scratch)
                try:
                    result = collect_score_response(
                        runtime,
                        probes,
                        samples,
                        expansion_point="ORIGIN",
                        prompt_subset=prompt_subset,
                        directions=provider,
                    )
                    result.update(provider.metadata())
                finally:
                    provider.close()
        else:
            result = collect_score_response(
                runtime, probes, samples, expansion_point="ORIGIN", prompt_subset=prompt_subset
            )

    def safe_payload(value):
        import numpy as np

        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: safe_payload(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [safe_payload(v) for v in value]
        return value

    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / f"gradient_path_attempt_{time.time_ns()}.json", certificate)
    receipt = {
        "status": "MEASURED",
        "origin_id": forks["origin_id"],
        "input_hash": input_hash,
        "anchor_samples": 1 if bridge else len(samples),
        "reference_used": False,
        "prompt_subset": list(prompt_subset),
        "execution_kind": runtime["identity"]["execution_kind"],
        "gradient_path": certificate,
        "scope": "ONE_SEQUENCE_GRADIENT_SMOKE" if bridge else "ORIGIN_FULL_LORA_GROUP_GRADIENTS",
    }
    payload = safe_payload(result)
    atomic_json(
        root / f"gradient_prepared_{time.time_ns()}.json",
        {"input_hash": input_hash, "payload_hash": state_hash(payload), "receipt": receipt},
    )
    fd, temp = tempfile.mkstemp(prefix=".gradient-", dir=root)
    os.close(fd)
    try:
        with open(temp, "wb") as stream:
            torch.save(payload, stream)
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    receipt = {**receipt, "artifact": _binding(path)}
    _publish(done, receipt)
    return receipt


def _measure_derivative_path(runtime, forks, prompt, sample):
    """Per-policy numerical derivative qualification, distinct from eval scoring."""
    from .score_response import certify_gradient_path, differentiable_full_teacher_logprobs

    runtime.pop("derivative_scorer", None)
    runtime.pop("derivative_certificate", None)
    if not hasattr(runtime["adapter"], "full_inputs"):
        return {
            "status": "UNAVAILABLE_EXPLICIT_PREFIX_FALLBACK",
            "passed": False,
            "fixture": runtime["identity"]["execution_kind"] != "REAL_CUDA_MODEL",
        }
    bank = next(
        (b for b in forks.get("banks", []) if b["contrasts"][CONTRASTS[0][0]]["norm"] > 0), None
    )
    if bank is None:
        return {"status": "NONZERO_DIRECTION_UNAVAILABLE_PREFIX_FALLBACK", "passed": False}
    left = runtime["checkpoint_cache"].load(bank["policies"]["joint_1"])["parameters"]
    right = runtime["checkpoint_cache"].load(bank["policies"]["joint_0"])["parameters"]
    direction = {
        n: (left[n].double() - right[n].double()).numpy() for n in runtime["parameter_order"]
    }
    tolerances = {
        **runtime["parity_tolerances"],
        "relative_l2": 1e-3,
        "absolute_directional_dot": 1e-5,
    }
    try:
        result = certify_gradient_path(
            runtime,
            prompt,
            sample,
            scorer=differentiable_full_teacher_logprobs,
            directions={"actual_joint_1_minus_joint_0": direction},
            tolerances=tolerances,
        )
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        result = {
            "status": "NOT_CERTIFIED",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "tolerances": tolerances,
            "fallback": "uncached_prefix_recompute",
        }
    if result["passed"]:
        runtime["derivative_scorer"] = differentiable_full_teacher_logprobs
        runtime["derivative_certificate"] = result
    return result


def _bridge_legacy_checkpoint(runtime, spec):
    """Explicitly promote historical trainable/Adam/RNG state, once per import.

    As in the V3 bridge, old checkpoint bytes do not claim to contain buffers,
    modes or position caches. New forward state is captured in certified
    eval/reset mode and labelled as a V4 bridge, never as historical evidence.
    """
    key = canonical_hash(spec)
    cache = runtime.setdefault("bridged_origins", {})
    if key in cache:
        # Recheck immutable file version even while the promoted state is cached.
        runtime["checkpoint_cache"].load(spec)
        return cache[key]
    old = runtime["checkpoint_cache"].load(spec)
    if {"buffers", "module_modes", "position_state", "adapter_position_state"} <= old.keys():
        cache[key] = old
        return old
    from ..followup_updates import _restore_forward
    from ..optimizer_fork import restore_state, state_hash

    if spec.get("state_hash") and state_hash(old) != spec["state_hash"]:
        raise ValueError("Historical full recorded state identity differs")
    _restore_forward(runtime["adapter"].model, runtime["initial_forward"], runtime["adapter"])
    restore_state(runtime["adapter"].model, runtime["optimizer"], old)
    runtime["adapter"].model.eval()
    runtime["adapter"]._reset_positions()
    step = spec.get("step", spec["identity"].get("step", 64))
    arm = spec.get("arm", spec["identity"].get("arm"))
    runtime["sampler"].clear()
    runtime["sampler"].update(
        step=step, position=4 * step, legacy_sampler_metadata=old["metadata"].get("sampler")
    )
    promoted = runtime["state"].capture(
        {
            **old["metadata"],
            "seed": 17,
            "arm": arm,
            "checkpoint_step": step,
            "legacy_state_hash": state_hash(old),
            "V4_forward_bridge": (
                "eval/reset forward state captured now; absent in historical schema"
            ),
        }
    )
    cache[key] = promoted
    return promoted


def _null_receipt(runtime, inputs, root):
    spec = inputs["null_origin"]
    state = _bridge_legacy_checkpoint(runtime, spec)
    runtime["state"].restore(state)
    record_path = Path(root) / "null_policy.json"
    if record_path.exists():
        policy = _read(record_path)
    else:
        policy = _save_policy(
            runtime,
            state,
            Path(root) / "null_policy.pt",
            {**runtime["identity"], "candidate_id": "same_policy_null"},
            full=False,
        )
        _publish(record_path, policy)
    backend = _backend(runtime)
    backend.activate(policy)
    by_id = {p["prompt_id"]: p for p in inputs["bridge_probes"]}
    scores = []
    for action in inputs["null_actions"]:
        prompt = by_id[action["prompt_id"]]
        prepared = backend._prepared(prompt)
        values = (
            runtime["adapter"]
            .logprobs(prepared, action["token_ids"], require_grad=False)
            .detach()
            .cpu()
            .double()
            .tolist()
        )
        scores.append(
            {
                "prompt_id": action["prompt_id"],
                "token_ids": action["token_ids"],
                "token_logprobs": values,
                "sequence_logp": __import__("math").fsum(values),
                "inference_fingerprint": policy["inference_fingerprint"],
                "input_audit": prepared["audit"],
                "proposal_sample_key": action["sample_key"],
                "prompt_record_hash": canonical_hash(prompt),
                "probability_execution": "uncached_prefix_recompute",
                "max_new_tokens": 64,
                "eos_token_ids": sorted(runtime["adapter"].eos_ids),
            }
        )
    receipt = {
        "gpu_identity": runtime["allocated_gpu"]["uuid"],
        "runtime_identity": runtime["identity"],
        "tolerances": runtime["parity_tolerances"],
        "scores": scores,
    }
    path = measure_scoring_paths(runtime, policy, inputs["bridge_probes"], inputs["null_actions"])
    anchor = Path(root) / "null_receipt.json"
    if anchor.exists():
        from ..modeling_v3.vlm_observation import _parity_values

        old = _read(anchor)
        if old["runtime_identity"] != receipt["runtime_identity"] or len(old["scores"]) != len(
            scores
        ):
            raise ValueError("Same-policy null identity changed on resume")
        errors = []
        sequence_errors = []
        for before, after in zip(old["scores"], scores, strict=True):
            if {
                k: v for k, v in before.items() if k not in ("token_logprobs", "sequence_logp")
            } != {k: v for k, v in after.items() if k not in ("token_logprobs", "sequence_logp")}:
                raise ValueError("Same-policy null actions changed on resume")
            errors.append(
                [
                    abs(a - b)
                    for a, b in zip(before["token_logprobs"], after["token_logprobs"], strict=True)
                ]
            )
            sequence_errors.append(abs(before["sequence_logp"] - after["sequence_logp"]))
        repeat = _parity_values(
            errors, runtime["parity_tolerances"], sequence_errors=sequence_errors
        )
        atomic_json(
            Path(root) / f"null_repeat_{time.time_ns()}.json",
            {"receipt": receipt, "parity": repeat},
        )
        if not repeat["passed"]:
            raise ValueError("Same-policy null changed beyond registered tolerances")
        receipt = old
    else:
        _publish(anchor, receipt)
    # Measurements can change across invocations; preserve each attempt.
    atomic_json(Path(root) / f"score_path_attempt_{time.time_ns()}.json", path)
    return receipt, path


def exercise_bridge_resume(runtime, policy, probes, *, origin_id, out):
    root = Path(out)
    done = root / "COMPLETE.json"
    if done.exists():
        receipt = _read(done)
        verify_artifact_bindings(receipt)
        return receipt
    start = runtime["adapter"].generation_calls
    interrupted = root / "INTERRUPTED.json"
    try:
        collect_actions(
            runtime,
            policy,
            probes[:1],
            origin_id=origin_id,
            role="recovery_smoke",
            draws=4,
            out=root / "packet",
            resume=(root / "packet").exists(),
            chunk_size=2,
            interrupt_after_first_commit=True,
        )
    except ControlledInterruption:
        _publish(
            interrupted,
            {
                "status": "CONTROLLED_INTERRUPTION_AFTER_DURABLE_CHUNK",
                "committed_chunk": _binding(root / "packet" / "chunk_00000000.parquet"),
                "actual_generation_calls": runtime["adapter"].generation_calls - start,
            },
        )
    if not interrupted.exists():
        raise RuntimeError("Recovery exercise lacks its actual interruption receipt")
    before = _read(interrupted)
    verify_artifact_bindings(before)
    second = runtime["adapter"].generation_calls
    packet = collect_actions(
        runtime,
        policy,
        probes[:1],
        origin_id=origin_id,
        role="recovery_smoke",
        draws=4,
        out=root / "packet",
        resume=True,
        chunk_size=2,
    )
    verify_artifact_bindings(before)
    keys = [row["sample_key"] for row in _rows(packet)]
    if len(set(keys)) != 4:
        raise ValueError("Recovery duplicated an independent sample identity")
    result = {
        "status": "INTERRUPTION_RESUME_MEASURED",
        "interruption": _binding(interrupted),
        "packet": _binding(root / "packet" / "COMPLETE.json"),
        "unique_sample_count": 4,
        "committed_original_unchanged": True,
        "actual_generation_calls_after_resume": runtime["adapter"].generation_calls - second,
        "execution_kind": runtime["identity"]["execution_kind"],
    }
    _publish(done, result)
    return result


def merge_bridge(root):
    """CPU merge is an actual technical barrier, not a positivity test."""
    from ..modeling_v3.vlm_observation import compare_cross_gpu_receipts

    root = Path(root)
    paths = [root / "tasks" / f"B_origin_{i}" / "COMPLETE.json" for i in (0, 1)]
    if not all(p.exists() for p in paths):
        return {"status": "WAITING_FOR_BRIDGE"}
    results = [_read(p) for p in paths]
    for value in results:
        verify_artifact_bindings(value)
    if any(r.get("execution_kind") != "REAL_CUDA_MODEL" for r in results):
        return {"status": "FIXTURE_BRIDGE_NOT_REAL_CUDA_EVIDENCE"}
    parity = compare_cross_gpu_receipts(
        results[0]["null"], results[1]["null"], tolerances=results[0]["null"]["tolerances"]
    )
    if not parity["passed"]:
        return {"status": "BRIDGE_NUMERICAL_FAILURE", "parity": parity}
    if not all(r["gradient"]["status"] == "MEASURED" for r in results):
        raise ValueError("Bridge lacks real derivative smoke")
    if not all(r["active_pair_count"] > 0 for r in results):
        return {"status": "BRIDGE_DEGENERATE_NONZERO_PAIR_MISSING", "parity": parity}
    if not all(_read(r["pair_diagnostics"]["path"])["all_finite"] for r in results):
        return {"status": "BRIDGE_DIRECT_LR_DIAGNOSTICS_UNRESOLVED", "parity": parity}
    if not all(r["resume_smoke"]["status"] == "INTERRUPTION_RESUME_MEASURED" for r in results):
        raise ValueError("Bridge lacks actual interruption/resume evidence")
    result = {
        "status": "TECHNICAL_BRIDGE_COMPLETED",
        "parity": parity,
        "worker_results": [_binding(p) for p in paths],
        "scientific_effectiveness": "NOT_CERTIFIED",
    }
    _publish(root / "B_MERGED.json", result)
    return result


def _measure_phase(runtime, name, function):
    import torch

    cuda = runtime["identity"]["execution_kind"] == "REAL_CUDA_MODEL" and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    adapter = runtime["adapter"]
    started = time.perf_counter()
    forwards = adapter.forward_calls
    generations = adapter.generation_calls
    backward = runtime.get("cost", {}).get("backward_calls", 0)
    derivative_start = len(runtime.get("score_cost_ledger", []))
    success = False
    try:
        result = function()
        success = True
        return result
    finally:
        if cuda:
            torch.cuda.synchronize()
        runtime.setdefault("phase_measurements", []).append(
            {
                "phase": name,
                "task_id": runtime.get("current_task_id"),
                "origin_id": runtime.get("current_origin_id"),
                "stage": runtime.get("current_stage"),
                "status": "COMPLETE" if success else "FAILED",
                "seconds": time.perf_counter() - started,
                "forward_calls": adapter.forward_calls - forwards,
                "generation_calls": adapter.generation_calls - generations,
                "update_backward_calls": runtime.get("cost", {}).get("backward_calls", 0)
                - backward,
                "score_derivative_calls": runtime.get("score_cost_ledger", [])[derivative_start:],
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
                "peak_memory_scope": "this_phase_with_existing_model" if cuda else "CPU_FIXTURE",
            }
        )


def _task_runtime_result(config, runtime, inputs, task, root, *, resume):
    tdir = root / "tasks" / task["id"]
    tdir.mkdir(parents=True, exist_ok=True)
    if task["kind"] == "source":
        return _measure_phase(
            runtime,
            "source_trajectory",
            lambda: run_source_trajectory(
                config,
                runtime,
                inputs["train_prompts"],
                seed=task["seed"],
                arm=task["arm"],
                out=tdir,
                resume=resume or (tdir / "LATEST.json").exists(),
            ),
        )
    if task["kind"] == "offline_track":
        from .tracking import run_tracking_task

        return _measure_phase(
            runtime,
            "offline_tracking",
            lambda: run_tracking_task(config, runtime, inputs, task, root, resume=resume),
        )
    if task["kind"] not in ("bridge", "map"):
        raise ValueError("Unsupported registered task kind")
    prefix = None
    if task.get("reuse_prefix_task"):
        previous = _read(root / "tasks" / task["reuse_prefix_task"] / "COMPLETE.json")
        verify_artifact_bindings(previous)
        if (
            previous["task"]["seed"] != task["seed"]
            or previous["task"]["arm"] != task["arm"]
            or previous["task"]["step"] != task["step"]
        ):
            raise ValueError("Prefix collection uses a different source origin")
        prefix = {
            "forks": _read(previous["forks"]["path"]),
            "response": _read(previous["response"]["path"]),
        }
    if task["kind"] == "bridge":
        spec = inputs["bridge_origins"][task["origin_index"]]
        if "fallback_source" in spec:
            fallback = spec["fallback_source"]
            source = _measure_phase(
                runtime,
                "registered_bridge_fallback_source",
                lambda: run_source_trajectory(
                    config,
                    runtime,
                    inputs["train_prompts"],
                    seed=fallback["seed"],
                    arm=fallback["arm"],
                    out=root / "tasks" / f"source_{fallback['seed']}_{fallback['arm']}",
                    resume=resume,
                    steps=32,
                    bridge_fallback=True,
                ),
            )
            spec = source["origins"]["32"]
        origin = _bridge_legacy_checkpoint(runtime, spec)
        null, path = _measure_phase(
            runtime, "same_policy_and_fast_path", lambda: _null_receipt(runtime, inputs, tdir)
        )
        probes = inputs["bridge_probes"]
        calib, query, draws, reference = 4, 0, 64, 64
    else:
        source = _read(root / "tasks" / task["source_task"] / "COMPLETE.json")
        origin = runtime["checkpoint_cache"].load(source["origins"][str(task["step"])])
        probes = inputs["panels"]["observation"][: task["prompts"]]
        calib, query, draws = task["calibration_banks"], task["query_banks"], task["draws"]
        reference = config["observations"]["reference_primary_n"]
        # Re-establish the eval-only fast path on this worker after a resumed
        # allocation, even if its B task already completed on an earlier GPU.
        null, path = _measure_phase(
            runtime, "same_policy_and_fast_path", lambda: _null_receipt(runtime, inputs, tdir)
        )
    origin_id = prefix["forks"]["origin_id"] if prefix else task["id"]
    banks = build_bank_plan(
        inputs["train_prompts"], origin_id=origin_id, calibration_banks=calib, query_banks=query
    )
    forks = _measure_phase(
        runtime,
        "actual_adam_forks",
        lambda: run_forks(
            runtime,
            origin,
            banks,
            inputs["train_prompts"],
            out=tdir / "forks",
            resume=resume,
            bridge=task["kind"] == "bridge",
            prefix=prefix["forks"] if prefix else None,
        ),
    )
    forks_path = tdir / "forks" / "COMPLETE.json"
    if task["kind"] == "bridge":
        for index in task["reserve_natural_bank_indices"]:
            if any(
                not b["contrasts"][CONTRASTS[0][0]]["exact_inference_alias"]
                and b["contrasts"][CONTRASTS[0][0]]["norm"] > 0
                for b in forks["banks"]
            ):
                break
            reserve = build_bank_plan(
                inputs["train_prompts"],
                origin_id=task["id"],
                calibration_banks=index + 1,
                query_banks=0,
            )
            reserve["banks"] = reserve["banks"][index : index + 1]
            reserve["plan_hash"] = canonical_hash(
                {k: v for k, v in reserve.items() if k != "plan_hash"}
            )
            extra = _measure_phase(
                runtime,
                "registered_reserve_natural_bank_forks",
                lambda reserve=reserve, index=index: run_forks(
                    runtime,
                    origin,
                    reserve,
                    inputs["train_prompts"],
                    out=tdir / f"reserve_{index:03d}",
                    resume=resume,
                    bridge=True,
                ),
            )
            forks = {
                **forks,
                "banks": forks["banks"] + extra["banks"],
                "predeclared_reserve_used": True,
            }
        forks_path = tdir / "ALL_FORKS.json"
        _publish(forks_path, forks)
        recovery = _measure_phase(
            runtime,
            "actual_interruption_resume",
            lambda: exercise_bridge_resume(
                runtime,
                forks["origin_policy"],
                probes,
                origin_id=task["id"],
                out=tdir / "resume_smoke",
            ),
        )
    else:
        recovery = None
    response = _measure_phase(
        runtime,
        "observation_scores_and_generation",
        lambda: collect_response_map(
            config,
            runtime,
            forks,
            probes,
            out=tdir / "response",
            draws=draws,
            reference_draws=reference,
            bridge=task["kind"] == "bridge",
            resume=resume,
            prefix=prefix["response"] if prefix else None,
        ),
    )
    response_path = tdir / "response" / "COMPLETE.json"
    if task["stage"] == "D" and (task.get("n_curve_max_draws") or task.get("reference_validation")):
        response = _measure_phase(
            runtime,
            "registered_n_curve_and_reference_validation",
            lambda: collect_nested_measurements(
                runtime,
                forks,
                probes,
                response,
                task,
                out=tdir / "nested_measurements",
                resume=resume,
            ),
        )
        response_path = tdir / "nested_measurements" / "EXTENDED_RESPONSE.json"
    gradient = _measure_phase(
        runtime,
        "full_lora_derivative",
        lambda: _score_derivative(
            runtime,
            forks,
            probes,
            response,
            out=tdir / "derivative",
            bridge=task["kind"] == "bridge",
            prompt_subset=task.get("derivative_prompt_subset", _diagnostic_prompt_ids(probes)),
        ),
    )
    diagnostics = pair_observation_diagnostics(response)
    _publish(tdir / "PAIR_DIAGNOSTICS.json", diagnostics)
    extras = {}
    if task["stage"] == "D":
        from .signature_collect import collect_development_directional, collect_signature_features

        def split(prompts, namespaces):
            return {
                "base_scene_ids": sorted({p["base_scene_id"] for p in prompts}),
                "prompt_ids": [p["prompt_id"] for p in prompts],
                "sample_ids": [],
                "rng_namespaces": list(namespaces),
            }

        panel_splits = {
            "train": split(
                inputs["train_prompts"],
                [b["train_samples"]["identity"]["rng_namespace"] for b in forks["banks"]],
            ),
            "observation": split(probes, [response["work"]["identity"]["rng_namespace"]]),
            "reference": split(probes, [response["reference"]["identity"]["rng_namespace"]]),
            "signature": split(
                inputs["panels"]["signature"], [f"{NAMESPACE}:{origin_id}:signature:ORIGIN"]
            ),
        }
        _measure_phase(
            runtime,
            "native_signature",
            lambda: collect_signature_features(
                runtime,
                forks,
                inputs["panels"]["signature"],
                panel_splits=panel_splits,
                out=tdir / "signature",
                resume=resume,
            ),
        )
        extras["signature"] = _binding(tdir / "signature" / "COMPLETE.json")
        if task["role"] == "development":
            anchors = [
                {**r, "role": "main"} for r in _rows(response["work"]) if r["draw_index"] < 32
            ]
            _measure_phase(
                runtime,
                "base_midpoint_ad_and_fd",
                lambda: collect_development_directional(
                    runtime,
                    forks,
                    probes,
                    anchors,
                    bank_ids=task["directional_bank_ids"],
                    steps=task["finite_difference_steps"],
                    out=tdir / "directional",
                    prompt_subset=task["derivative_prompt_subset"],
                    sample_limit=32,
                    resume=resume,
                    stage_role="development",
                ),
            )
            extras["development_directional"] = _binding(tdir / "directional" / "COMPLETE.json")
            if task["seed"] == 41001 and task["arm"] == "X_BASE" and task["step"] == 32:
                from .cross_prompt import run_cross_prompt

                _measure_phase(
                    runtime,
                    "registered_different_scene_probe_panel",
                    lambda: run_cross_prompt(
                        config,
                        runtime,
                        inputs,
                        task,
                        forks,
                        out=tdir / "cross_prompt",
                        resume=resume,
                    ),
                )
                extras["cross_prompt"] = _binding(tdir / "cross_prompt" / "COMPLETE.json")
    result = {
        "status": "COMPLETED",
        "kind": task["kind"],
        "task": task,
        "origin_id": origin_id,
        "forks": _binding(forks_path),
        "response": _binding(response_path),
        "gradient": gradient,
        "null": null,
        "scoring_path": path,
        "pair_diagnostics": _binding(tdir / "PAIR_DIAGNOSTICS.json"),
        "active_pair_count": response["active_pairs_available"],
        "resume_smoke": recovery,
        "config_hash": runtime["identity"]["config_hash"],
        "source_hash": runtime["identity"]["source_hash"],
        **extras,
        "scientific_status": "NOT_CERTIFIED",
        "execution_kind": runtime["identity"]["execution_kind"],
    }
    _publish(tdir / "COMPLETE.json", result)
    return result


def run_worker(
    tasks,
    *,
    worker_id,
    workers,
    device="cuda:0",
    execute_gpu=False,
    resume=False,
    runtime_factory=None,
):
    """Run assigned B/C tasks with one model; exit instead of holding an idle GPU."""
    if execute_gpu is not True:
        raise PermissionError("V4 real execution requires --execute-gpu")
    tasks = _read(tasks) if not isinstance(tasks, dict) else copy.deepcopy(tasks)
    if (
        canonical_hash({k: v for k, v in tasks.items() if k != "task_list_hash"})
        != tasks["task_list_hash"]
    ):
        raise ValueError("Task list changed")
    if workers != tasks["workers"] or not 0 <= worker_id < workers or workers not in (2, 3):
        raise ValueError("Worker assignment differs")
    if tasks["source"] != source_identity():
        raise ValueError("Campaign source changed")
    if tasks.get("analysis_rules_hash") != canonical_hash(tasks.get("analysis_rules")):
        raise ValueError("Frozen reference analysis rules changed")
    root = Path(tasks["root"])
    wdir = root / "workers" / f"worker_{worker_id}"
    if wdir.exists() and not resume:
        raise FileExistsError("Worker output exists; explicit --resume required")
    wdir.mkdir(parents=True, exist_ok=True)
    from ..core import frozen_writer

    with frozen_writer(wdir):
        runtime = None
        completed = []
        missing = []
        started = time.perf_counter()
        status = "COMPLETED_REQUESTED_STAGES"
        validated_extensions = {}
        try:
            for task in tasks["tasks"]:
                if task["worker"] != worker_id or task["stage"] not in tasks["stages"]:
                    continue
                extension = None
                if task["stage"] in ("D", "E"):
                    extension = _validate_phase_task(tasks, task, validated_extensions)
                tdir = root / "tasks" / task["id"]
                if (tdir / "COMPLETE.json").exists():
                    previous = _read(tdir / "COMPLETE.json")
                    verify_artifact_bindings(previous)
                    if task["kind"] == "source":
                        if (previous.get("seed"), previous.get("arm"), previous.get("steps")) != (
                            task["seed"],
                            task["arm"],
                            128,
                        ):
                            raise ValueError("Completed source differs from task")
                    elif previous.get("task") != task:
                        raise ValueError("Completed collection differs from task")
                    completed.append(task["id"])
                    continue
                if "B_MERGED" in task["depends_on"]:
                    merged = merge_bridge(root)
                    if merged["status"] != "TECHNICAL_BRIDGE_COMPLETED":
                        missing.append("B_MERGED")
                        status = merged["status"]
                        break
                phase_barriers = (
                    {
                        "C_FIRST_MAP_COMPLETE",
                        "D_SELECTION_FROZEN",
                        "D_CALIBRATION_FROZEN",
                        "D_POINT_RESPONSE_RESOLVED",
                    }
                    if extension
                    else set()
                )
                if any(
                    d != "B_MERGED"
                    and d not in phase_barriers
                    and not (root / "tasks" / d / "COMPLETE.json").exists()
                    for d in task["depends_on"]
                ):
                    missing.extend(task["depends_on"])
                    status = "WAITING_FOR_DEPENDENCIES"
                    break
                if runtime is None:
                    load_started = time.perf_counter()
                    runtime = (runtime_factory or load_runtime)(
                        tasks["config"], tasks["bindings"], device=device, execute_gpu=True
                    )
                    runtime["model_load_seconds"] = time.perf_counter() - load_started
                    runtime["analysis_rules"] = tasks["analysis_rules"]
                    from .storage import storage_status

                    storage = storage_status(root)
                    _publish(wdir / "runtime_identity.json", runtime["identity"])
                    atomic_json(
                        wdir / f"preflight_{time.time_ns()}.json",
                        {"storage": storage, "allocated_gpu": runtime["allocated_gpu"]},
                    )
                if extension is not None:
                    runtime["phase_extension"] = extension
                    if task["kind"] == "offline_track":
                        runtime["tracking_extension"] = extension
                runtime["current_task_id"] = task["id"]
                runtime["current_origin_id"] = task.get("reuse_prefix_task", task["id"])
                runtime["current_stage"] = task["stage"]
                phase_start = len(runtime.get("phase_measurements", []))
                task_success = False
                try:
                    _task_runtime_result(
                        tasks["config"], runtime, tasks["inputs"], task, root, resume=resume
                    )
                    task_success = True
                finally:
                    atomic_json(
                        wdir / f"task_measurement_{task['id']}_{time.time_ns()}.json",
                        {
                            "campaign_id": tasks["campaign_id"],
                            "task_list_hash": tasks["task_list_hash"],
                            "task_id": task["id"],
                            "task_hash": canonical_hash(task),
                            "origin_id": runtime["current_origin_id"],
                            "stage": task["stage"],
                            "status": "COMPLETE" if task_success else "FAILED",
                            "execution_kind": runtime["identity"]["execution_kind"],
                            "phase_measurements": runtime.get("phase_measurements", [])[
                                phase_start:
                            ],
                        },
                    )
                completed.append(task["id"])
            if "B" in tasks["stages"]:
                bridge = merge_bridge(root)
                if bridge["status"] != "TECHNICAL_BRIDGE_COMPLETED":
                    status = bridge["status"]
            for phase in set(tasks["stages"]) & {"D", "E"}:
                if not any(t["stage"] == phase for t in tasks["tasks"]):
                    status = "REGISTERED_LATER_STAGES_NOT_EXECUTED"
        except BaseException as exc:
            receipt = {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "worker_id": worker_id,
                "workers": workers,
                "task_list_hash": tasks["task_list_hash"],
                "campaign_id": tasks["campaign_id"],
                "completed_task_ids": completed,
                "phase_measurements": runtime.get("phase_measurements", []) if runtime else [],
                "elapsed_seconds": time.perf_counter() - started,
                "model_load_seconds": runtime.get("model_load_seconds") if runtime else None,
                "alias_storage_checks": runtime.get("alias_storage_checks", []) if runtime else [],
                "score_derivative_costs": runtime.get("score_cost_ledger", []) if runtime else [],
            }
            if isinstance(exc, OSError):
                from .storage import storage_status

                receipt["storage_after_write_error"] = storage_status(root)
            atomic_json(wdir / f"attempt_{time.time_ns()}.json", receipt)
            _replace_json(wdir / "LATEST.json", receipt)
            raise
        receipt = {
            "status": status,
            "worker_id": worker_id,
            "workers": workers,
            "campaign_id": tasks["campaign_id"],
            "task_list_hash": tasks["task_list_hash"],
            "completed_task_ids": completed,
            "missing_dependencies": missing,
            "model_load_count": int(runtime is not None),
            "elapsed_seconds": time.perf_counter() - started,
            "model_load_seconds": runtime.get("model_load_seconds") if runtime else None,
            "phase_measurements": runtime.get("phase_measurements", []) if runtime else [],
            "forward_calls": runtime["adapter"].forward_calls if runtime else 0,
            "generation_calls": runtime["adapter"].generation_calls if runtime else 0,
            "optimizer_and_backward_costs": runtime.get("cost", {}) if runtime else {},
            "score_derivative_costs": runtime.get("score_cost_ledger", []) if runtime else [],
            "alias_storage_checks": runtime.get("alias_storage_checks", []) if runtime else [],
            "execution_kind": runtime["identity"]["execution_kind"]
            if runtime
            else "NO_MODEL_LOADED",
        }
        atomic_json(wdir / f"attempt_{time.time_ns()}.json", receipt)
        _replace_json(wdir / "LATEST.json", receipt)
        return receipt


def _validate_phase_task(tasks, task, cache):
    """One evidence verification per appended phase, before model loading."""
    from .phase_evidence import verify_phase_evidence
    from .phase_schedule import (
        build_confirmation_extension,
        build_development_extension,
        build_tracking_extension,
        validate_selection,
    )

    matches = [
        e
        for e in tasks.get("phase_extensions", [])
        if any(t["id"] == task["id"] for t in e["tasks"])
    ]
    if len(matches) != 1:
        raise ValueError("Task lacks exactly one registered phase extension")
    extension = matches[0]
    digest = extension["extension_hash"]
    if canonical_hash({k: v for k, v in extension.items() if k != "extension_hash"}) != digest:
        raise ValueError("Phase extension changed")
    if next(t for t in extension["tasks"] if t["id"] == task["id"]) != task:
        raise ValueError("Appended task differs from its extension")
    if digest in cache:
        return extension
    if extension["campaign_id"] != tasks["campaign_id"] or extension[
        "config_hash"
    ] != canonical_hash(tasks["config"]):
        raise ValueError("Phase extension campaign/config differs")
    selected = extension.get("selection")
    if selected:
        validate_selection(tasks["config"], selected)
        verify_phase_evidence(selected["development_evidence"])
    if extension["phase"] == "D":
        first = verify_phase_evidence(extension["first_map_evidence"])
        if extension["role"] == "development":
            build_development_extension(tasks["config"], first, workers=tasks["workers"])
        else:
            calibration = extension.get("calibration")
            if calibration:
                verify_phase_evidence(calibration["calibration_evidence"], selection=selected)
            build_confirmation_extension(
                tasks["config"],
                first,
                selected,
                role=extension["role"],
                workers=tasks["workers"],
                calibration=calibration,
            )
    else:
        point = extension.get("point_response_evidence")
        verify_artifact_bindings(point)
        rebuilt = build_tracking_extension(
            tasks["config"], selected, point, workers=tasks["workers"]
        )
        if rebuilt["status"] != "PLANNED_NOT_EXECUTED":
            raise ValueError("Point response remains unresolved; E cannot execute")
    cache[digest] = True
    return extension


def collect_score_response_worker(
    tasks, *, worker_id, workers, device="cuda:0", execute_gpu=False, resume=False
):
    """Derivative-only continuation on already collected origin/work packets."""
    if execute_gpu is not True:
        raise PermissionError("Derivative collection requires --execute-gpu")
    tasks = _read(tasks) if not isinstance(tasks, dict) else tasks
    if (
        canonical_hash({k: v for k, v in tasks.items() if k != "task_list_hash"})
        != tasks["task_list_hash"]
        or tasks["source"] != source_identity()
    ):
        raise ValueError("Campaign identity changed")
    if workers != tasks["workers"] or not 0 <= worker_id < workers:
        raise ValueError("Worker assignment differs")
    runtime = None
    results = []
    for task in tasks["tasks"]:
        if task["worker"] != worker_id or task["kind"] not in ("bridge", "map"):
            continue
        root = Path(tasks["root"]) / "tasks" / task["id"]
        if not (root / "response" / "COMPLETE.json").exists():
            continue
        if runtime is None:
            runtime = load_runtime(
                tasks["config"], tasks["bindings"], device=device, execute_gpu=True
            )
        probes = (
            tasks["inputs"]["bridge_probes"]
            if task["kind"] == "bridge"
            else tasks["inputs"]["panels"]["observation"][: task["prompts"]]
        )
        forks_path = (
            root / "ALL_FORKS.json"
            if (root / "ALL_FORKS.json").exists()
            else root / "forks" / "COMPLETE.json"
        )
        results.append(
            _score_derivative(
                runtime,
                _read(forks_path),
                probes,
                _read(root / "response" / "COMPLETE.json"),
                out=root / "derivative",
                bridge=task["kind"] == "bridge",
                prompt_subset=task.get("derivative_prompt_subset", _diagnostic_prompt_ids(probes)),
            )
        )
    return {
        "status": "MEASURED_AVAILABLE_ORIGINS" if results else "NO_COLLECTED_ORIGINS",
        "origins": results,
        "model_load_count": int(runtime is not None),
    }
