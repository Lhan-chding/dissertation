"""Executable frozen discovery/SFT workflow. Model outputs are never simulated."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

from .queue import Queue, digest, read_json, write_json


def _code_identity():
    from .config import file_digest

    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*.py"))}


def _context(run):
    run = Path(run).resolve()
    plan = read_json(run / "protocol.json")
    machine = read_json(run / "machine.json")
    frozen = read_json(run / "freeze.json")
    if digest(plan) != frozen["protocol_hash"] or digest(machine) != frozen["machine_hash"]:
        raise ValueError("Frozen plan/machine modified")
    if frozen.get("code_identity") is not None and _code_identity() != frozen["code_identity"]:
        raise ValueError("Frozen scientific runtime source changed")
    return run, plan, machine, frozen


def preflight(plan_path, machine_path, out):
    from ..protocol_state_probes.checkpoint_catalog import check_checkpoints
    from .config import load_protocol

    plan = load_protocol(plan_path)
    machine = read_json(machine_path)
    run = Path(out).resolve()
    maximum = machine.get("max_concurrent_gpus", 5)
    if type(maximum) is not int or not 1 <= maximum <= 5:
        raise ValueError("At most five one-GPU workers")
    if (
        machine.get("automatic_dependency_upgrade", False)
        or machine.get("test_results_sealed", True) is not True
    ):
        raise ValueError("Environment upgrade or unsealed tests forbidden")
    for key in (
        "runtime_path",
        "frozen_protocol_path",
        "checkpoint_catalog",
        "historical_metadata_index",
        "replay_rows_path",
    ):
        if key not in machine or not Path(machine[key]).is_file():
            raise FileNotFoundError("Required verified machine input: " + key)
    catalog = read_json(machine["checkpoint_catalog"])
    sources = [
        row["source"] for row in catalog["checkpoints"] if row["checkpoint_id"] in ("S96", "REP96")
    ]
    if {row["id"] for row in sources} != {"S96", "REP96"}:
        raise ValueError("Both registered parent sources are required")
    refreshed = check_checkpoints(
        sources, machine.get("path_mapping", catalog.get("path_mappings", {}))
    )
    frozen = {
        "code_identity": _code_identity(),
        "protocol_hash": digest(plan),
        "machine_hash": digest(machine),
        "input_hashes": {
            key: digest(read_json(machine[key])) for key in ("runtime_path", "frozen_protocol_path")
        },
        "parent_bindings": {
            row["checkpoint_id"]: row.get("checkpoint") for row in refreshed["checkpoints"]
        },
    }
    frozen["run_identity"] = digest(frozen)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "freeze.json").exists() and read_json(run / "freeze.json") != frozen:
        raise ValueError("Existing run has a different immutable identity")
    write_json(run / "protocol.json", plan)
    write_json(run / "machine.json", machine)
    write_json(run / "parent_availability.json", refreshed)
    write_json(run / "freeze.json", frozen)
    receipt = {
        "status": "PASS" if refreshed["status"] == "ALL_AVAILABLE" else "PARTIAL_AVAILABILITY",
        "run_identity": frozen["run_identity"],
        "checkpoint_tensors_verified": False,
        "gpu_execution": False,
    }
    write_json(run / "preflight.json", receipt)
    return receipt


def prepare_data(run):
    from .fresh_cohort import prepare_cohort

    run, plan, machine, _ = _context(run)
    replay_path = Path(machine["replay_rows_path"])
    replay = (
        read_json(replay_path)
        if replay_path.suffix == ".json"
        else [json.loads(line) for line in replay_path.read_text().splitlines() if line.strip()]
    )
    manifest = prepare_cohort(
        run / "cohort",
        read_json(machine["historical_metadata_index"]),
        replay,
        protocol=plan,
        render_images=True,
    )
    from .r0_reference import freeze_prompt_schedules

    freeze_prompt_schedules(run)
    return manifest


def _backend(run, parent):
    from ..protocol_state_probes.inference import FrozenBackend

    run, _, machine, frozen = _context(run)
    for key in ("runtime_path", "frozen_protocol_path"):
        if digest(read_json(machine[key])) != frozen["input_hashes"][key]:
            raise ValueError("Bound runtime/protocol changed")
    records = read_json(run / "parent_availability.json")["checkpoints"]
    record = next(row for row in records if row["checkpoint_id"] == parent)
    if record.get("checkpoint") != frozen["parent_bindings"][parent]:
        raise ValueError("Parent binding changed")
    return FrozenBackend(
        machine["runtime_path"], record, read_json(machine["frozen_protocol_path"]), allow_gpu=True
    )


def bridge(run, device="cuda:0"):
    from .bridge import run_bridge_from_backend
    from .canonical_targets import build_training_view, gold_targets
    from .public_tasks import RoleDataset

    if device != "cuda:0":
        raise ValueError("One allocated visible GPU, addressed as cuda:0, required")
    run, _, _, _ = _context(run)
    data = RoleDataset(run / "cohort", "gold")
    all_tasks = data.public("T_train")
    tasks = [
        task
        for family in ("cross_series", "trend")
        for task in sorted(
            [t for t in all_tasks if t["family"] == family], key=lambda t: t["task_id"]
        )[:4]
    ]
    audit = {row["task_id"]: row for row in data.audit("T_train")}
    targets = gold_targets(tasks, [audit[task["task_id"]] for task in tasks])
    view = build_training_view(tasks, targets, source="gold")
    results = {}
    for parent in ("S96", "REP96"):
        backend = _backend(run, parent)
        results[parent] = run_bridge_from_backend(
            backend, rows=view, output_dir=run / "bridge" / parent
        )
        del backend
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    receipt = {
        "status": "PASS"
        if all(r.get("status") == "PASS" for r in results.values())
        else "BLOCKED_TECHNICAL",
        "parents": results,
        "train_only": True,
    }
    write_json(run / "bridge.json", receipt)
    return receipt


def build_queue(run):
    from .public_tasks import RoleDataset

    run, plan, machine, frozen = _context(run)
    cohort = read_json(run / "cohort/cohort_manifest.json")
    if cohort.get("status") != "FROZEN":
        raise ValueError("Freeze cohort before queue registration")
    jobs = []

    def add(kind, id, dependencies=(), **values):
        priority = (
            0
            if kind in ("select", "discover")
            else 1
            if kind == "sft"
            else 2
            if kind == "teacher" and values.get("split") == "V_selection"
            else 3
            if kind == "teacher" and values.get("split") == "T_train"
            else 6
        )
        jobs.append(
            {
                "id": id,
                "kind": kind,
                "dependencies": list(dependencies),
                "gpu_count": 1,
                "priority": priority,
                **values,
            }
        )
        return id

    teacher = {}
    for parent in ("S96", "REP96"):
        for split, repeats in (
            ("V_selection", (0,)),
            ("T_train", (0, 1)),
            ("E_test", (0,)),
            ("G_guard", (0,)),
        ):
            role = {
                "T_train": "teacher_train",
                "V_selection": "selector",
                "E_test": "teacher_eval",
                "G_guard": "teacher_eval",
            }[split]
            tasks = RoleDataset(run / "cohort", role).public(split)
            for repeat in repeats:
                ids = []
                for offset in range(0, len(tasks), 24):
                    ids.append(
                        add(
                            "teacher",
                            f"teacher.{parent}.{split}.{repeat}.{offset}",
                            parent=parent,
                            split=split,
                            repeat=repeat,
                            role=role,
                            task_ids=[t["task_id"] for t in tasks[offset : offset + 24]],
                            draws=4 if split == "G_guard" else 16,
                        )
                    )
                teacher[parent, split, repeat] = ids
        selection = add(
            "select", f"select.{parent}", teacher[parent, "V_selection", 0], parent=parent
        )
        for repeat in (0, 1):
            discovery = add(
                "discover",
                f"discover.{parent}.{repeat}",
                [selection, *teacher[parent, "T_train", repeat]],
                parent=parent,
                repeat=repeat,
            )
            arms = (
                ["GOLD_ALL"]
                + (["REPLAY_ONLY"] if parent == "S96" else [])
                + ["SELF_O0", "SELF_MIX", "SELF_SINGLE"]
                + (["GOLD_MATCH_MIX"] if parent == "S96" else [])
            )
            prior = []
            for arm in arms:
                dependencies = [] if arm in ("GOLD_ALL", "REPLAY_ONLY") else [discovery]
                student = add(
                    "sft",
                    f"sft.{parent}.{repeat}.{arm}",
                    dependencies,
                    parent=parent,
                    repeat=repeat,
                    arm=arm,
                    seed=plan["sft"]["seeds"][repeat],
                    steps=256,
                    after=prior[:],
                )
                prior.append(student)
                for split, step, draws in (
                    ("V_selection", 64, 4),
                    ("V_selection", 128, 4),
                    ("E_test", 256, 8),
                    ("G_guard", 256, 4),
                ):
                    add(
                        "evaluate",
                        f"eval.{parent}.{repeat}.{arm}.{split}.{step}",
                        [student, *teacher[parent, split, 0]]
                        if split in ("E_test", "G_guard")
                        else [student],
                        parent=parent,
                        repeat=repeat,
                        arm=arm,
                        source_job=student,
                        split=split,
                        step=step,
                        draws=draws,
                    )
                for step in (64, 128, 256):
                    add(
                        "evaluate",
                        f"sentinel.{parent}.{repeat}.{arm}.{step}",
                        [student],
                        parent=parent,
                        repeat=repeat,
                        arm=arm,
                        source_job=student,
                        split="T_train",
                        role="validation",
                        step=step,
                        draws=1,
                        sentinel=True,
                    )
            r0 = add(
                "r0",
                f"r0.{parent}.{repeat}",
                parent=parent,
                repeat=repeat,
                seed=plan["sft"]["seeds"][repeat],
                steps=32,
            )
            for split, step, draws in (
                ("V_selection", 8, 4),
                ("E_test", 32, 8),
                ("G_guard", 32, 4),
            ):
                add(
                    "evaluate",
                    f"eval.{parent}.{repeat}.R0_RESET32.{split}.{step}",
                    [r0],
                    parent=parent,
                    repeat=repeat,
                    arm="R0_RESET32",
                    source_job=r0,
                    split=split,
                    step=step,
                    draws=draws,
                )
    queue = Queue(run / "queue.sqlite", maximum=machine.get("max_concurrent_gpus", 5))
    identity = digest(
        {"run": frozen["run_identity"], "cohort": cohort["manifest_digest"], "jobs": jobs}
    )
    queue.register(jobs, identity)
    registry = {
        "schema": "verified-discovery-matrix-v1",
        "identity": identity,
        "cohort_manifest_digest": cohort["manifest_digest"],
        "jobs": jobs,
    }
    write_json(run / "REGISTERED_MATRIX.json", registry)
    return {
        "status": "REGISTERED",
        "jobs": len(jobs),
        "identity": identity,
        "sft_jobs": sum(job["kind"] == "sft" for job in jobs),
        "r0_jobs": 4,
    }


def _job_root(run, job):
    return run / ("sealed" if job.get("split") in ("E_test", "G_guard") else "evidence") / job["id"]


def _records(run, parent, split, repeat):
    root = run / ("sealed" if split in ("E_test", "G_guard") else "evidence")
    rows = []
    for path in sorted(root.glob(f"teacher.{parent}.{split}.{repeat}.*/chunks/*.json")):
        chunk = read_json(path)
        if digest(chunk["rows"]) != chunk["rows_hash"]:
            raise ValueError("Committed sample chunk digest mismatch")
        rows.extend(chunk["rows"])
    return rows


def _generate(run, job, backend):
    from .config import PROTOCOLS
    from .evaluate import generate_frozen
    from .portfolio import atomic_request
    from .public_tasks import RoleDataset, compile_case, verify_raw

    role = job.get(
        "role", "validation" if job["split"] in ("V_selection", "T_train") else "teacher_eval"
    )
    tasks = RoleDataset(run / "cohort", role).public(job["split"])
    if job.get("sentinel"):
        tasks = [
            task
            for family in ("cross_series", "trend")
            for task in sorted(
                [t for t in tasks if t["family"] == family], key=lambda t: t["task_id"]
            )[:8]
        ]
    if "task_ids" in job:
        wanted = set(job["task_ids"])
        tasks = [task for task in tasks if task["task_id"] in wanted]
        if len(tasks) != len(wanted):
            raise ValueError("Registered task missing")
    root = _job_root(run, job)
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    committed = []
    runtime = backend.receipt["inference_fingerprint"]
    for task in tasks:
        path = root / "chunks" / (digest(task["task_id"]) + ".json")
        identity = digest({"job": job, "task": task, "runtime": runtime})
        if path.exists():
            chunk = read_json(path)
            if chunk["identity"] != identity or digest(chunk["rows"]) != chunk["rows_hash"]:
                raise ValueError("Committed generation identity changed")
            committed.extend(chunk["rows"])
            continue
        rows = []
        began = time.time()
        try:
            protocols = (
                PROTOCOLS[task["family"]]
                if job["kind"] == "teacher" and job["split"] != "G_guard"
                else ("O0",)
            )
            for protocol in protocols:
                case = compile_case(task, protocol)
                for draw in range(job["draws"]):
                    if job["kind"] == "teacher":
                        request = atomic_request(
                            job["parent"],
                            task,
                            protocol,
                            role,
                            job["repeat"],
                            draw,
                            model_revision=runtime,
                            runtime_version="verified-discovery-v1",
                        )
                    else:
                        request = {
                            "parent_id": job["parent"],
                            "model_revision": runtime,
                            "runtime_version": "verified-discovery-v1",
                            "task_id": task["task_id"],
                            "base_instance_id": task["base_instance_id"],
                            "split": task["split"],
                            "protocol_id": protocol,
                            "role": role,
                            "pipeline_repeat": job["repeat"],
                            "draw_index": draw,
                            "template_version": case["template_version"],
                            "source_job": job["source_job"],
                            "step": job["step"],
                            "greedy": bool(job.get("sentinel")),
                        }
                        key = digest(request)
                        request.update(request_id=key, sample_seed=int(key[:16], 16) % (2**63 - 1))
                    raw_path = root / "atomic" / (request["request_id"] + ".json")
                    raw_identity = digest(
                        {"request": request, "prompt_identity": case["prompt_identity"]}
                    )
                    if raw_path.exists():
                        saved = read_json(raw_path)
                        if (
                            saved["identity"] != raw_identity
                            or digest(saved["raw"]) != saved["raw_hash"]
                        ):
                            raise ValueError("Accepted atomic raw receipt changed")
                        raw = saved["raw"]
                    else:
                        raw = generate_frozen(
                            backend,
                            case["prompt"],
                            request["sample_seed"],
                            data_root=run / "cohort",
                            do_sample=not job.get("sentinel", False),
                        )
                        # Commit raw output before scoring; wrong answers are completed draws.
                        write_json(
                            raw_path,
                            {
                                "identity": raw_identity,
                                "request": request,
                                "raw": raw,
                                "raw_hash": digest(raw),
                            },
                        )
                    rows.append(
                        {
                            **request,
                            **raw,
                            "raw_completion": raw["raw_text"],
                            "family": task["family"],
                            **verify_raw(task, protocol, raw["raw_text"]),
                        }
                    )
        except BaseException as exc:
            write_json(
                root / "attempts" / f"{time.time_ns()}.json",
                {
                    "identity": identity,
                    "rows": rows,
                    "error": repr(exc),
                    "raw_failure": getattr(exc, "raw", None),
                },
            )
            raise
        chunk = {
            "identity": identity,
            "rows": rows,
            "rows_hash": digest(rows),
            "wall_seconds": time.time() - began,
        }
        write_json(path, chunk)
        committed.extend(rows)
    receipt = {
        "status": "COMPLETE",
        "rows": len(committed),
        "generated_tokens": sum(len(row.get("token_ids", [])) for row in committed),
        "generation_seconds": sum(row.get("elapsed_seconds", 0.0) for row in committed),
        "backend": backend.receipt,
        "job_identity": digest(job),
    }
    write_json(root / "receipt.json", receipt)
    return receipt


def _bridge_passed(run):
    from .bridge import assess_bridge_diagnostics

    path = run / "bridge.json"
    if not path.exists():
        raise RuntimeError("SFT bridge not yet executed")
    receipt = read_json(path)
    parents = receipt.get("parents") if isinstance(receipt, dict) else None
    if not isinstance(parents, dict) or set(parents) != {"S96", "REP96"}:
        raise RuntimeError("SFT bridge requires complete evidence for both S96 and REP96")
    if any(not isinstance(parent, dict) for parent in parents.values()):
        raise RuntimeError("SFT bridge parent diagnostics are invalid")
    # Reassess evidence instead of trusting an old or incorrectly written PASS label.
    statuses = {
        name: assess_bridge_diagnostics(parent)["status"] for name, parent in parents.items()
    }
    if any(status not in {"PASS", "NUMERICAL_REVIEW_REQUIRED"} for status in statuses.values()):
        raise RuntimeError(f"SFT bridge has non-overridable diagnostic failures: {statuses}")
    if all(status == "PASS" for status in statuses.values()):
        return
    acceptance = run / "BRIDGE_NUMERICAL_ACCEPTANCE.json"
    if acceptance.exists():
        review = read_json(acceptance)
        if (
            isinstance(review, dict)
            and review.get("bridge_hash") == digest(receipt)
            and review.get("status") == "ACCEPTED"
            and isinstance(review.get("rationale"), str)
            and review["rationale"].strip()
        ):
            return
    raise RuntimeError("Real SFT bridge requires successful causality/resume and numerical review")


def _training_rows(run, job):
    from .canonical_targets import build_training_view, gold_targets, matched_gold_ids
    from .public_tasks import RoleDataset

    data = RoleDataset(run / "cohort", "trainer")
    tasks = data.public("T_train")
    replay = data.replay()
    replay_view = build_training_view(
        [row["task"] for row in replay],
        {row["task"]["task_id"]: {"canonical_vector": row["canonical_vector"]} for row in replay},
        source="replay",
    )
    arm = job["arm"]
    if arm == "REPLAY_ONLY":
        focus = []
    elif arm == "GOLD_ALL":
        audit = RoleDataset(run / "cohort", "gold").audit("T_train")
        focus = build_training_view(tasks, gold_targets(tasks, audit), source="gold")
    else:
        discovery = read_json(
            run / "evidence" / f"discover.{job['parent']}.{job['repeat']}" / "discovery.json"
        )
        if arm == "GOLD_MATCH_MIX":
            audit = RoleDataset(run / "cohort", "gold").audit("T_train")
            ids, matching = matched_gold_ids(
                tasks, audit, discovery["J_MIX"], seed=106070 + job["repeat"]
            )
            all_targets = gold_targets(tasks, audit)
            targets = {tid: all_targets[tid] for tid in ids}
            write_json(run / "evidence" / job["id"] / "gold_matching.json", matching)
        else:
            targets = discovery[
                {"SELF_O0": "J_O0", "SELF_MIX": "J_MIX", "SELF_SINGLE": "J_SINGLE"}[arm]
            ]
        focus = build_training_view(
            tasks, targets, source="gold" if arm == "GOLD_MATCH_MIX" else "self"
        )
    return focus, replay_view


def _dispatch(run, job, queue, backend_cache=None):
    run, plan, machine, frozen = _context(run)
    root = _job_root(run, job)
    root.mkdir(parents=True, exist_ok=True)
    if job["kind"] == "teacher":
        backend_cache = backend_cache if backend_cache is not None else {}
        if backend_cache.get("parent") != job["parent"]:
            backend_cache.clear()
            gc.collect()
            backend_cache.update(parent=job["parent"], backend=_backend(run, job["parent"]))
        return _generate(run, job, backend_cache["backend"])
    if job["kind"] == "select":
        from .portfolio import select_fixed_policies
        from .public_tasks import RoleDataset

        selection = select_fixed_policies(
            RoleDataset(run / "cohort", "selector").public("V_selection"),
            _records(run, job["parent"], "V_selection", 0),
            job["parent"],
        )
        write_json(root / "selection.json", selection)
        return {"status": "COMPLETE", "selection": selection}
    if job["kind"] == "discover":
        from .discover import discover_sets
        from .public_tasks import RoleDataset

        selection = read_json(run / "evidence" / f"select.{job['parent']}" / "selection.json")
        discovered = discover_sets(
            RoleDataset(run / "cohort", "teacher_train").public("T_train"),
            _records(run, job["parent"], "T_train", job["repeat"]),
            selection,
            job["parent"],
            job["repeat"],
        )
        write_json(root / "discovery.json", discovered)
        return {"status": "COMPLETE", "discovery_path": str(root / "discovery.json")}
    if job["kind"] == "sft":
        _bridge_passed(run)
        from .sft_runner import run_sft

        focus, replay = _training_rows(run, job)
        if not focus and job["arm"] != "REPLAY_ONLY":
            result = {
                "status": "NO_VERIFIED_TARGETS_RETURN_PARENT",
                "parent": job["parent"],
                "optimizer_updates": 0,
                "empty_discovery_retained": True,
            }
            write_json(root / "result.json", result)
            return result

        # Label-source provenance is excluded; mathematical view and all settings remain.
        def view(rows):
            return [
                {
                    key: row.get(key)
                    for key in (
                        "task_id",
                        "prompt",
                        "target",
                        "target_token_ids",
                        "EOS_id",
                        "weight",
                    )
                }
                for row in rows
            ]

        identity = {
            "parent": job["parent"],
            "parent_checkpoint": frozen["parent_bindings"][job["parent"]],
            "repeat": job["repeat"],
            "seed": job["seed"],
            "runtime_bindings": frozen["input_hashes"],
            "focus": view(focus),
            "replay": view(replay),
            "sft": plan["sft"],
        }
        signature = digest(identity)
        for previous in queue.rows():
            if (
                previous["status"] not in ("COMPLETE", "ALIAS")
                or previous["payload"]["kind"] != "sft"
            ):
                continue
            result = previous["result"]
            if result and result.get("training_signature") == signature:
                alias = {
                    "status": "ALIAS",
                    "alias_of": result.get("alias_of", previous["id"]),
                    "training_signature": signature,
                    "strict_same_view_and_settings": True,
                }
                write_json(root / "result.json", alias)
                return alias
        backend = _backend(run, job["parent"])
        checkpoints = sorted(root.glob("step*.pt"))
        result = run_sft(
            backend,
            focus=focus,
            replay=replay,
            identity={
                "arm": job["arm"],
                "training_signature": signature,
                "run_identity": frozen["run_identity"],
            },
            seed=job["seed"],
            output_dir=root,
            microbatch_size=machine.get("microbatch_size", 4),
            resume_path=checkpoints[-1] if checkpoints else None,
        )
        result = {
            **result,
            "status": "COMPLETE",
            "runtime_status": result["status"],
            "training_signature": signature,
        }
        write_json(root / "result.json", result)
        return result
    if job["kind"] == "r0":
        _bridge_passed(run)
        from .r0_reference import run_r0

        return run_r0(run, job, _backend(run, job["parent"]))
    if job["kind"] == "evaluate":
        lookup = {row["id"]: row for row in queue.rows()}
        source = lookup[job["source_job"]]
        if source["status"] == "ALIAS":
            canonical = lookup[source["result"]["alias_of"]]["payload"]
            prefix = "sentinel" if job.get("sentinel") else "eval"
            suffix = f"{job['step']}" if job.get("sentinel") else f"{job['split']}.{job['step']}"
            alias_id = (
                f"{prefix}.{canonical['parent']}.{canonical['repeat']}.{canonical['arm']}.{suffix}"
            )
            return {
                "status": "ALIAS",
                "alias_of": alias_id,
                "strict_same_model_and_evaluation": True,
            }
        if source["status"] == "NO_VERIFIED_TARGETS_RETURN_PARENT":
            if job["split"] not in ("E_test", "G_guard"):
                return {
                    "status": "ALIAS",
                    "alias_of": "parent_validation",
                    "reason": "Empty J returns unchanged parent; no SFT curve",
                }
            return {
                "status": "ALIAS",
                "alias_of": f"parent.{job['parent']}.{job['split']}",
                "draw_subset": list(range(job["draws"])),
                "source_records": "teacher committed O0 rows",
                "reason": "NO_VERIFIED_TARGETS_RETURN_PARENT",
            }
        from .sft_runtime import load_student_into_backend

        backend = _backend(run, job["parent"])
        source_root = _job_root(run, source["payload"])
        result = source["result"]
        checkpoint = source_root / f"step{job['step']:03d}.pt"
        load_student_into_backend(backend, checkpoint, result["identity"])
        return _generate(run, job, backend)
    raise ValueError("Unknown registered task kind")


def worker(queue_path, worker_id, device, *, once=False):
    if device != "cuda:0":
        raise ValueError("Workers require one visible allocated GPU mapped to cuda:0")
    run = Path(queue_path).resolve().parent
    _, _, machine, _ = _context(run)
    queue = Queue(queue_path, maximum=machine.get("max_concurrent_gpus", 5))
    import fcntl

    lock_path = run / "worker_locks" / (digest(worker_id) + ".lock")
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backend_cache = {}
        while True:
            try:
                _bridge_passed(run)
                blocked = ()
            except RuntimeError:
                blocked = ("sft", "r0")
            job = queue.claim(worker_id, blocked_kinds=blocked)
            if job is None:
                if once or queue.all_terminal():
                    return {"status": "IDLE_OR_TERMINAL"}
                time.sleep(5)
                continue
            try:
                if job["kind"] in ("sft", "r0", "evaluate"):
                    backend_cache.clear()
                    gc.collect()
                result = _dispatch(run, job, queue, backend_cache)
                status = result.get("status")
                if status not in ("COMPLETE", "ALIAS", "NO_VERIFIED_TARGETS_RETURN_PARENT"):
                    raise RuntimeError("Runtime did not return a validated terminal success")
                queue.finish(job["id"], worker_id, status, result)
            except Exception as exc:
                error = {
                    "status": "BLOCKED_TECHNICAL",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(
                    _job_root(run, job) / "attempts" / f"failure_{time.time_ns()}.json", error
                )
                queue.finish(job["id"], worker_id, "BLOCKED_TECHNICAL", error)
            finally:
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            if once:
                return {"status": "ONE_TASK_EXECUTED", "job_id": job["id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "prepare-data", "bridge", "build-queue"):
        sub = subs.add_parser(command)
        sub.add_argument("--plan", required=True)
        sub.add_argument("--machine", required=True)
        sub.add_argument("--out", required=True)
        if command == "bridge":
            sub.add_argument("--device", default="cuda:0")
    sub = subs.add_parser("worker")
    sub.add_argument("--queue", required=True)
    sub.add_argument("--worker-id", required=True)
    sub.add_argument("--device", required=True)
    sub.add_argument("--once", action="store_true")
    sub = subs.add_parser("report")
    sub.add_argument("--run", required=True)
    sub.add_argument("--reveal-test-after-completion", action="store_true")
    sub = subs.add_parser("retry")
    sub.add_argument("--queue", required=True)
    sub.add_argument("--job-id", required=True)
    sub.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.plan, args.machine, args.out)
    elif args.command in ("prepare-data", "bridge", "build-queue"):
        run, plan, machine, _ = _context(args.out)
        if digest(read_json(args.plan)) != digest(plan) or digest(
            read_json(args.machine)
        ) != digest(machine):
            raise ValueError("CLI inputs differ from frozen run")
        if args.command == "prepare-data":
            result = prepare_data(run)
        elif args.command == "bridge":
            result = bridge(run, args.device)
        else:
            result = build_queue(run)
    elif args.command == "worker":
        result = worker(args.queue, args.worker_id, args.device, once=args.once)
    elif args.command == "retry":
        _, _, machine, _ = _context(Path(args.queue).parent)
        queue = Queue(args.queue, maximum=machine.get("max_concurrent_gpus", 5))
        queue.retry(args.job_id, reason=args.reason)
        result = {"status": "PENDING", "job": args.job_id}
    else:
        from .reports import report

        result = report(args.run, reveal=args.reveal_test_after_completion)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
