"""Registered task execution and bounded first-four development scheduling."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from ..core import frozen_writer
from ..modeling_v4 import gpu_collect as gpu
from .jobs import TaskRegistry
from .protocol import get_lineage
from .runtime import read_json


def register_development(config, root, prepared, *, first_four=True):
    registry = TaskRegistry(root, config)
    smoke = registry.register_task("smoke", {"smoke_id": "two-recipes-two-updates"})
    specs = (
        config["origins"]["development"][:4]
        if first_four
        else (config["origins"]["development"] + config["origins"]["tuning"])
    )
    if not first_four and not (Path(root) / "FIRST_FOUR_DELIVERED.json").is_file():
        raise ValueError(
            "Deliver first four complete lineages and measured timing before expansion"
        )
    tasks = [smoke]
    for spec in specs:
        lid = spec["seed"]
        base = {"lineage_id": lid, "source_recipe": spec["source_recipe"], "role": spec["role"]}
        source = registry.register_task("source", base, [smoke["task_id"]])
        tasks.append(source)
        for step in spec["anchors"]:
            origin = f"{lid}_t{step}"
            common = {**base, "origin_id": origin, "origin_step": step}
            dependency = {
                "task_id": source["task_id"],
                "completed_artifact": str(
                    Path(root).resolve() / "sources" / str(lid) / f"H{step:02d}.json"
                ),
            }
            prestate = registry.register_task(
                "prestate",
                {
                    **common,
                    "policy_id": origin,
                    "panel_id": "P",
                    "snapshot_t": step,
                    "draw_role": "predecision",
                },
                [dependency],
            )
            tasks.append(prestate)
            for recipe in (
                *config["actions"]["selectable"],
                *config["actions"]["external_baselines"],
            ):
                schedule = prepared["branch_schedules"]["1"]
                tasks.append(
                    registry.register_task(
                        "branch",
                        {
                            **common,
                            "recipe_id": recipe,
                            "repeat": 1,
                            "schedule_id": schedule["schedule_id"],
                            "branch_schedule_seed": schedule["sampler_seed"],
                        },
                        [dependency, prestate["task_id"]],
                    )
                )
    return {
        "status": "TASKS_REGISTERED_NOT_RUN",
        "tasks": [t["task_id"] for t in tasks],
        "development_lineages": len(specs),
        "final_test_tasks": 0,
    }


def register_test(config, root, prepared):
    from .workflow import verify_frozen_execution

    registry = TaskRegistry(root, config)
    frozen = registry.load_freeze()
    verify_frozen_execution(frozen, prepared)
    tasks = []
    for spec in config["origins"]["test_pool"][: frozen["test_n"]]:
        lid, step = spec["seed"], spec["anchors"][0]
        origin = f"{lid}_t{step}"
        base = {"lineage_id": lid, "source_recipe": spec["source_recipe"], "role": spec["role"]}
        source = registry.register_task("source", base)
        dependency = {
            "task_id": source["task_id"],
            "completed_artifact": str(
                Path(root).resolve() / "sources" / str(lid) / f"H{step:02d}.json"
            ),
        }
        prestate = registry.register_task(
            "prestate",
            {
                **base,
                "origin_id": origin,
                "origin_step": step,
                "policy_id": origin,
                "panel_id": "P",
                "snapshot_t": step,
                "draw_role": "predecision",
            },
            [dependency],
        )
        tasks.extend((source, prestate))
        if (Path(root) / "decisions" / f"{origin}.json").exists():
            tasks.extend(
                registry.register_test_branches(
                    origin,
                    dependencies=[dependency, prestate["task_id"]],
                    branch_schedules=prepared["branch_schedules"],
                )
            )
    return {"status": "FROZEN_TEST_TASKS_REGISTERED", "tasks": [t["task_id"] for t in tasks]}


def _metadata(source, step):
    updates = []
    for path in sorted(Path(source).glob("segments/*/COMMIT.json")):
        commit = read_json(path)
        for binding in commit["updates"]:
            row = read_json(binding["path"])
            if step - 8 < row["step"] <= step:
                updates.append(row)
    if len(updates) != 8:
        raise ValueError("Eight committed past training updates required")
    norms = [r["grad_norm_preclip"] for r in updates]
    advantages = [r["reward_statistics"]["advantages"] for r in updates]
    flat_groups = [g for bank in advantages for g in bank]
    return {
        "gradient_norm_mean": statistics.mean(norms),
        "gradient_norm_std": statistics.pstdev(norms),
        "loss_mean": statistics.mean(r["loss"] for r in updates),
        "effective_group_fraction": sum(any(a != 0 for a in g) for g in flat_groups)
        / len(flat_groups),
        "zero_advantage_group_fraction": sum(all(a == 0 for a in g) for g in flat_groups)
        / len(flat_groups),
    }


def observe_state(runtime, config, root, payload):
    from .evaluation import collect_evaluation
    from .features import LEVELS, build_predecision_packet

    lid, step, origin = payload["lineage_id"], payload["origin_step"], payload["origin_id"]
    source = Path(root) / "sources" / str(lid)
    output = Path(root) / "prestate" / origin
    snapshots = {}
    for snapshot in (step - 8, step):
        checkpoint = read_json(source / f"H{snapshot:02d}.json")["checkpoint"]
        _, rows = collect_evaluation(
            runtime,
            checkpoint,
            runtime["prepared_data"]["panels"]["P"],
            out=output / f"t{snapshot}",
            lineage_id=lid,
            origin_id=origin,
            policy_id=f"{lid}_t{snapshot}",
            panel_id="P",
            horizon=snapshot,
            draws=32,
            role="predecision",
            resume=True,
        )
        fields = ("event", "relation_numerator", "relation_denominator", "F", "B", "M")
        snapshots[snapshot] = [
            {
                **{k: row[k] for k in ("prompt_id", "family", "interface")},
                **{k: row["semantic"][k] for k in fields},
            }
            for row in rows
        ]
    metadata = _metadata(source, step)
    for level in LEVELS:
        packet = build_predecision_packet(
            origin_id=origin,
            lineage_id=str(lid),
            source_recipe=payload["source_recipe"],
            step=step,
            current_rows=snapshots[step],
            history_rows=snapshots[step - 8],
            feature_level=level,
            known_training_metadata=metadata,
        )
        gpu._publish(output / f"{level}.json", packet.to_dict())
    result = {
        "status": "PRESTATE_COMPLETE",
        "origin_id": origin,
        "shared_generated_outputs": 2 * 72 * 32,
        "four_levels_same_samples": True,
    }
    gpu._publish(output / "COMPLETE.json", result)
    return result


def _diagnostic_subset(prompts):
    # Metadata-only: four paired scenes per family, retaining both interfaces.
    by_family = {}
    for prompt in prompts:
        by_family.setdefault(prompt["family"], set()).add(prompt["base_scene_id"])
    ids = {s for scenes in by_family.values() for s in sorted(scenes)[:4]}
    return [p for p in prompts if p["base_scene_id"] in ids]


def execute_task(config, root, task_id, runtime_path, *, allow_gpu=False):
    from .branch_runtime import run_branch
    from .evaluation import collect_evaluation
    from .origin_runtime import run_smoke, run_source
    from .runtime import load_runtime

    registry = TaskRegistry(root, config)
    task = registry.assert_can_execute(task_id)
    p, kind = task["payload"], task["kind"]
    with frozen_writer(Path(root) / "worker_locks" / task_id):
        registry.assert_can_execute(task_id)
        load_start = time.perf_counter()
        runtime = load_runtime(config, runtime_path, allow_gpu=allow_gpu)
        load_elapsed = time.perf_counter() - load_start
        data = runtime["prepared_data"]
        gpu._publish(
            Path(root) / "startup" / task_id / f"{time.time_ns()}.json",
            {
                "task_id": task_id,
                "allocated_gpu": runtime["allocated_gpu"],
                "load_seconds": load_elapsed,
                "execution_kind": runtime["identity"]["execution_kind"],
            },
        )
        if kind == "smoke":
            result = run_smoke(
                runtime,
                prompts=data["source_prompts"],
                schedule=data["source_schedule"],
                out=Path(root) / "smoke",
            )
        elif kind == "source":
            spec = get_lineage(config, p["lineage_id"])
            result = run_source(
                runtime,
                prompts=data["source_prompts"],
                schedule=data["source_schedule"],
                lineage_id=p["lineage_id"],
                source_recipe=p["source_recipe"],
                steps=spec["source_steps"],
                out=Path(root) / "sources" / str(p["lineage_id"]),
                resume=True,
            )
        elif kind == "prestate":
            result = observe_state(runtime, config, root, p)
        elif kind == "branch":
            origin = read_json(
                Path(root) / "sources" / str(p["lineage_id"]) / f"H{p['origin_step']:02d}.json"
            )["checkpoint"]
            out = registry.branch_output(p["origin_id"], p["recipe_id"], p["repeat"])
            test = p["role"] == "locked_test"
            frozen = registry.load_freeze() if test else None
            if test:
                from .workflow import verify_frozen_execution

                verify_frozen_execution(frozen, data)
            if test and data.get("test_panel_status") != "READY":
                raise ValueError("Test execution requires audited T")

            def evaluate(run, checkpoint, step):
                panel = "T" if test else "D"
                prompts = data["panels"][panel]
                draws = frozen["test_draws"] if test else 16
                if step == 8:
                    panel += "_H8"
                    prompts = data["panels"]["T_H8"] if test else _diagnostic_subset(prompts)
                    draws = 8
                value, _ = collect_evaluation(
                    run,
                    checkpoint,
                    prompts,
                    out=out / "evaluations" / f"H{step:02d}",
                    lineage_id=p["lineage_id"],
                    origin_id=p["origin_id"],
                    policy_id=f"{p['origin_id']}:{p['recipe_id']}:{p['repeat']}:H{step}",
                    panel_id=panel,
                    horizon=step,
                    draws=draws,
                    repeat=p["repeat"],
                    role="diagnostic" if step == 8 else "evaluation",
                    resume=True,
                )
                return value

            result = run_branch(
                runtime,
                origin,
                prompts=data["continuation_prompts"],
                schedule=data["branch_schedules"][str(p["repeat"])],
                lineage_id=p["lineage_id"],
                origin_id=p["origin_id"],
                recipe=p["recipe_id"],
                repeat=p["repeat"],
                out=out,
                resume=True,
                evaluate=evaluate,
                role=p["role"],
                authorization_receipt=task if test else None,
            )
        elif kind == "historical_e":
            private = read_json(runtime_path)
            historical = read_json(private["historical_runtime"])
            panel = read_json(historical["panels"]["path"])["panels"]["E"]
            result, _ = collect_evaluation(
                runtime,
                p["checkpoint"],
                panel,
                out=Path(root) / "historical_E" / p["policy_id"],
                lineage_id=p["lineage_id"],
                origin_id=p["origin_id"],
                policy_id=p["policy_id"],
                panel_id="E_old",
                horizon=32,
                draws=32,
                role="historical_E",
                resume=True,
            )
        else:
            raise ValueError("Unsupported registered task kind: " + kind)
        registry.mark_complete(task_id, {"status": "COMPLETE", "result": result})
        return result
