"""Registered SR-F4 workers. A task is committed only after its output exists."""

from __future__ import annotations

import argparse
import os
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from srf4_ref.sampling import State, initialize, plan_step, seed_for, update_state

from .common import append, now, read, rows, write
from .training import load_checkpoint, save_checkpoint, update


def task_index(root):
    return {row["qid"]: row for row in rows(Path(root) / "DATA/TASKS.jsonl")}


def image_index(root):
    return {row["qid"]: row for row in rows(Path(root) / "DATA/IMAGE_MANIFEST.jsonl")}


@lru_cache(maxsize=1)
def external_type_registry(root):
    return read(
        Path(read(Path(root) / "CONFIG.json")["old_root"]) / "external/chartqa/TYPE_REGISTRY.json"
    )


def input_for(task, image, protocol="structured"):
    from .evaluation import model_input_for

    frozen = {
        key: task[key] for key in ("qid", "pool", "chart", "prompt", "render_style") if key in task
    }
    return model_input_for(frozen, image, protocol)


def group(
    root,
    runtime,
    task,
    image,
    assignment,
    directory,
    *,
    task_id,
    block,
    master,
    step=0,
    repeat=1,
    arm="PARENT",
    source=None,
    allow_confirm=False,
):
    from .task_contract import score
    from .verification import verify_semantics

    if "CONFIRM" in task["split"] and not allow_confirm:
        raise PermissionError("Confirmation questions cannot enter a development/training worker")
    index = assignment["group_index"]
    group_id = f"{task_id}/s{step:03d}/g{index:05d}"
    size = assignment.get("G", assignment.get("g", 8))
    protocol = assignment.get("protocol", "structured")
    seeds = assignment.get("seeds") or [
        seed_for(master, "rollout", block, repeat, step, index, task["qid"], draw)
        for draw in range(size)
    ]

    def validate_cached(values, offset=0):
        for draw, record in enumerate(values, offset):
            if any(
                record.get(key) != value
                for key, value in {
                    "qid": task["qid"],
                    "group_id": group_id,
                    "draw": draw,
                    "seed": seeds[draw],
                    "protocol": protocol,
                    "block": block,
                    "arm": arm,
                }.items()
            ):
                raise ValueError("Cached rollout task, sample, or policy identity differs")

    path = Path(directory) / f"{index:05d}.json"
    if path.exists():
        saved = read(path)
        if saved["group_id"] != group_id or len(saved["rows"]) != size:
            raise ValueError("Saved rollout identity or draw budget differs")
        validate_cached(saved["rows"])
        return saved["rows"]
    result = []
    for offset in range(0, size, 8):
        chunk_path = Path(directory) / f"{index:05d}.chunk{offset // 8}.json"
        if chunk_path.exists():
            chunk = read(chunk_path)
            if [r["seed"] for r in chunk] != seeds[offset : offset + 8]:
                raise ValueError("Cached rollout RNG stream differs")
            validate_cached(chunk, offset)
        else:
            native_input = input_for(task, image, protocol)
            pending = chunk_path.with_suffix(".pending.jsonl")
            if pending.exists():
                prior_partial = pending.with_name(pending.name + f".interrupted-{time.time_ns()}")
                pending.rename(prior_partial)
                append(
                    Path(root) / "technical_events.jsonl",
                    dict(
                        at=now(),
                        task_id=task_id,
                        group_id=group_id,
                        status="REPLAY_INTERRUPTED_BATCH_SAME_SEEDS_SAME_POLICY",
                        retained_partial=str(prior_partial),
                    ),
                )
            chunk = runtime.generate_group(
                native_input,
                Path(root) / "DATA",
                seeds[offset : offset + 8],
                protocol=protocol,
                on_completion=lambda r, destination=pending: append(destination, r),
            )
            for draw, record in enumerate(chunk, offset):
                record["qid"] = task["qid"]
                if task.get("pool") == "CHARTQA":
                    from sr_f3_ce.external import score_external

                    registry = external_type_registry(str(root))
                    native_score = score_external(record, task, type_registry=registry)
                else:
                    native_score = score(
                        record["full_text"],
                        task["world"],
                        task["query"],
                        finish_reason=record["finish_reason"],
                        protocol=protocol,
                    )
                record.update(
                    row_id=f"{group_id}/d{draw:02d}",
                    group_id=group_id,
                    qid=task["qid"],
                    world_id=task.get("world_id", task.get("root_id")),
                    block=block,
                    repeat=repeat,
                    arm=arm,
                    step=step,
                    draw=draw,
                    source=source,
                    panel=assignment.get("panel", source),
                    family=task.get("family"),
                    variant=task.get("variant"),
                    split=task["split"],
                    score=native_score,
                )
                if task.get("pool") != "CHARTQA":
                    record["semantic"] = verify_semantics(task, record, native_score)
            write(chunk_path, chunk)
            pending.unlink(missing_ok=True)
        result.extend(chunk)
    write(path, dict(group_id=group_id, block=block, source=source, step=step, rows=result))
    # Atomic complete group owns all records. Chunk files are now redundant.
    for offset in range(0, size, 8):
        (Path(directory) / f"{index:05d}.chunk{offset // 8}.json").unlink(missing_ok=True)
    return result


def run_probe(root, spec):
    from .runtime import load

    source, block, task_id = spec["source"], spec["block"], spec["task_id"]
    root = Path(root)
    tasks, images = task_index(root), image_index(root)
    directory = root / "PROBES" / source / f"b{block}"
    assignments = list(rows(root / "DATA" / f"{source}_GROUPS.jsonl"))
    runtime, _, _ = load(root, block, task_id)
    rewards, roots = defaultdict(list), defaultdict(lambda: defaultdict(list))
    total = 0
    for assignment in assignments:
        task = tasks[assignment["qid"]]
        records = group(
            root,
            runtime,
            task,
            images[task["qid"]],
            assignment,
            directory / "groups",
            task_id=task_id,
            block=block,
            master=2026102221,
            source=source,
        )
        total += len(records)
        key = f"{task['family']}_v{task.get('variant', 0)}"
        values = [r["score"]["J"] for r in records]
        rewards[key].extend(values)
        roots[key][task["world_id"]].extend(values)
        write(
            directory / "PROGRESS.json",
            dict(
                at=now(),
                groups_done=assignment["group_index"] + 1,
                answers=total,
                target_groups=len(assignments),
            ),
        )
    summary = dict(
        source=source,
        block=block,
        answers=total,
        groups=len(assignments),
        variants={
            key: dict(
                p=sum(value) / len(value),
                n=len(value),
                root_count=len(roots[key]),
                per_root={q: dict(n=len(v), p=sum(v) / len(v)) for q, v in roots[key].items()},
            )
            for key, value in rewards.items()
        },
    )
    write(directory / "SUMMARY.json", summary)
    return summary


def publish_committed_checkpoint(runtime, directory, latest, step):
    """Repair publication after a crash between checkpoint commit and export."""
    directory = Path(directory)
    if step and step % 16 == 0:
        endpoint = directory / "checkpoints" / f"step-{step:03d}.pt"
        if not endpoint.exists():
            os.link(latest, endpoint)
        adapter = directory / "adapters" / f"step-{step:03d}"
        marker = adapter / "PUBLISHED.json"
        if not marker.exists():
            runtime.model.save_pretrained(adapter, safe_serialization=True)
            write(marker, dict(step=step, at=now(), status="COMPLETE"))
    write(
        directory / "PROGRESS.json",
        dict(at=now(), successful_updates=step, target_updates=64, answers=step * 128),
    )


def run_train(root, spec):
    from mm_core.training import trainable_state

    from .runtime import load

    root = Path(root)
    task_id, block, arm = spec["task_id"], spec["block"], spec["arm"]
    if spec.get("track", "A") != "A":
        raise PermissionError("C requires separate F1 with actual human audit and RUN_C")
    if not (root / "FREEZE_A_F1.json").exists():
        raise PermissionError("A must be frozen before any optimizer update")
    tasks, images = task_index(root), image_index(root)
    by_variant = defaultdict(lambda: defaultdict(list))
    for task in tasks.values():
        if task["split"] == "OLD_TRAIN":
            by_variant[f"{task['family']}_v{task['variant']}"][task["world_id"]].append(task["qid"])
    if len(by_variant) != 32:
        raise ValueError("Expected all 32 old training variants")
    initial = read(root / "PROBES/B0" / f"b{block}/SUMMARY.json")["variants"]
    state = {key: initialize(initial[key]["p"]) for key in by_variant}
    directory = root / "RUNS/A" / f"b{block}" / "r1" / arm
    directory.mkdir(parents=True, exist_ok=True)
    runtime, optimizer, scheduler = load(root, block, task_id, training=True)
    reference = trainable_state(runtime.model)
    identity = dict(
        experiment="SRF4-CONSOLIDATED-20261022",
        task_id=task_id,
        block=block,
        arm=arm,
        master_seed=2026102201,
        repeat=1,
        steps=64,
    )
    write(directory / "run_identity.json", identity)
    latest = directory / "checkpoints/latest.pt"
    step = 0
    if latest.exists():
        step, restored = load_checkpoint(latest, runtime, optimizer, scheduler, identity=identity)
        state = {key: State(**value) for key, value in restored["states"].items()}
    else:
        save_checkpoint(
            latest,
            runtime,
            optimizer,
            scheduler,
            identity=identity,
            step=0,
            sampler_state={"states": {k: asdict(v) for k, v in state.items()}},
        )
    runtime.identity["step"] = step
    publish_committed_checkpoint(runtime, directory, latest, step)
    for update_step in range(step + 1, 65):
        started = time.monotonic()
        runtime.torch.cuda.reset_peak_memory_stats()
        # The reference sampler chooses world roots uniformly, then a chart.
        root_lists = {k: sorted(v) for k, v in by_variant.items()}
        assignments = plan_step(
            state,
            root_lists,
            seed=seed_for(2026102201, "sampler", block, update_step),
            mode="lz" if arm == "LZ" else "uniform",
        )
        selected = []
        for assignment in assignments:
            choices = sorted(by_variant[assignment.variant_id][assignment.qid])
            chart = seed_for(
                2026102201, "chart", block, update_step, assignment.group_index, assignment.qid
            ) % len(choices)
            selected.append(dict(asdict(assignment), qid=choices[chart], world_id=assignment.qid))
        write(
            directory / "assignments" / f"{update_step:03d}.json",
            dict(
                step=update_step,
                states_before={k: asdict(v) for k, v in state.items()},
                groups=selected,
            ),
        )
        records, observed, sizes = [], defaultdict(list), []
        for assignment in selected:
            task = tasks[assignment["qid"]]
            current = group(
                root,
                runtime,
                task,
                images[task["qid"]],
                assignment,
                directory / "rollouts" / f"{update_step:03d}",
                task_id=task_id,
                block=block,
                master=2026102201,
                step=update_step,
                arm=arm,
                source="A_TRAIN",
            )
            records.extend(current)
            sizes.append(assignment["g"])
            observed[assignment["variant_id"]].extend(r["score"]["J"] for r in current)
        receipt = update(runtime, optimizer, scheduler, records, sizes, reference)
        state = {key: update_state(value, observed[key]) for key, value in state.items()}
        receipt.update(
            step=update_step,
            arm=arm,
            block=block,
            at=now(),
            checkpoint_before=dict(task_id=task_id, successful_update=update_step - 1),
            checkpoint_after=dict(task_id=task_id, successful_update=update_step),
            states_after={k: asdict(v) for k, v in state.items()},
            wall_seconds=time.monotonic() - started,
            gpu_hours=(time.monotonic() - started) / 3600,
            peak_gpu_allocated_bytes=runtime.torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=runtime.torch.cuda.max_memory_reserved(),
            generated_tokens=sum(len(r["tokens"]) for r in records),
        )
        write(directory / "updates" / f"{update_step:03d}.json", receipt)
        save_checkpoint(
            latest,
            runtime,
            optimizer,
            scheduler,
            identity=identity,
            step=update_step,
            sampler_state={"states": {k: asdict(v) for k, v in state.items()}},
        )
        runtime.identity["step"] = update_step
        publish_committed_checkpoint(runtime, directory, latest, update_step)
    return dict(
        block=block,
        arm=arm,
        successful_updates=64,
        answers=8192,
        adapter=str(directory / "adapters/step-064"),
    )


def execute(root, spec):
    if spec["kind"] == "probe":
        return run_probe(root, spec)
    if spec["kind"] == "train":
        return run_train(root, spec)
    if spec["kind"] == "replay":
        from .replay import replay_stage

        return replay_stage(root)
    if spec["kind"] == "eval":
        return run_eval(root, spec)
    if spec["kind"] == "report":
        from .reporting import report

        return report(root)
    if spec["kind"] == "package":
        from .delivery import package

        return package(root)
    raise ValueError("Unregistered task kind")


def run_eval(root, spec):
    import torch

    from sr_f1.runtime import copy_parameters

    from .evaluation import build_a_eval_groups
    from .runtime import load

    root = Path(root)
    block, arm, task_id = spec["block"], spec["arm"], spec["task_id"]
    tasks, images = task_index(root), image_index(root)
    config = read(root / "CONFIG.json")
    external = []
    if block == 0 and arm != "PARENT":
        external_root = Path(config["old_root"]) / "external/chartqa"
        external = list(rows(external_root / "TASKS_AUDIT_ONLY.jsonl"))
        # Explicit routes are copied at preparation; no ground-truth sidecars.
        routes = {r["qid"]: r for r in rows(root / "DATA/CHARTQA_IMAGES.jsonl")}
        tasks.update(
            {
                t["qid"]: dict(t, split="OLD_REGRESSION", prompt=routes[t["qid"]]["prompt"])
                for t in external
            }
        )
        images.update({r["qid"]: r for r in rows(root / "DATA/CHARTQA_IMAGES.jsonl")})
    all_groups = build_a_eval_groups(list(tasks.values()), block, arm=arm, external_tasks=external)
    assignments = [
        a for i, a in enumerate(all_groups) if i % spec.get("shards", 1) == spec.get("shard", 0)
    ]
    runtime, _, _ = load(root, block, task_id)
    if arm != "PARENT":
        checkpoint = root / "RUNS/A" / f"b{block}/r1" / arm / "checkpoints/step-064.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        copy_parameters(runtime.model, state["policy"])
        runtime.identity["step"] = 64
    directory = root / "EVAL" / f"b{block}" / arm / f"shard{spec.get('shard', 0)}"
    write(
        root / "CONFIRM_OPENED.json",
        dict(
            at=now(),
            decisions_frozen=True,
            C_status=read(root / "C_GATE_REPORT.json").get("decision", "BLOCK_C"),
        ),
    )
    count = 0
    for assignment in assignments:
        task = tasks[assignment["qid"]]
        values = group(
            root,
            runtime,
            task,
            images[task["qid"]],
            assignment,
            directory / "groups",
            task_id=task_id,
            block=block,
            master=2026102224,
            step=0 if arm == "PARENT" else 64,
            arm=arm,
            source="FINAL_EVAL",
            allow_confirm=True,
        )
        count += len(values)
        write(
            directory / "PROGRESS.json",
            dict(at=now(), answers=count, groups_total=len(assignments)),
        )
    return dict(block=block, arm=arm, groups=len(assignments), answers=count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    root = args.root
    spec = next(t for t in read(root / "TASKS.json") if t["task_id"] == args.task)
    complete = root / "tasks" / args.task / "COMPLETE.json"
    if complete.exists() and read(complete).get("status") == "COMPLETE":
        return
    append(
        root / "RUN_REGISTRY.jsonl",
        dict(at=now(), task_id=args.task, status="RUNNING", job_id=os.environ.get("SLURM_JOB_ID")),
    )
    try:
        result = execute(root, spec)
        if result.get("status") == "INCOMPLETE_TECHNICAL":
            raise RuntimeError("Stage retained technical gaps; see its written stage report")
        receipt = dict(status="COMPLETE", task_id=args.task, at=now(), result=result)
        write(complete, receipt)
        append(root / "RUN_REGISTRY.jsonl", receipt)
    except BaseException as error:
        append(
            root / "technical_events.jsonl",
            dict(
                at=now(),
                task_id=args.task,
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
