"""Prospective experiment commands. Metadata commands never load Qwen."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ..modeling_v4 import gpu_collect as gpu
from .protocol import DEFAULT_PATH, load_protocol, workload
from .runtime import read_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in (
        "plan",
        "audit-history",
        "prepare-data",
        "prepare-runtime",
        "register-development",
        "register-test",
        "smoke",
        "source",
        "observe-state",
        "branch",
        "worker",
        "submit-ready",
        "fit-selectors",
        "plan-precision",
        "freeze",
        "decide",
        "analyze",
    ):
        q = sub.add_parser(name)
        q.add_argument("--config", type=Path, default=DEFAULT_PATH)
        q.add_argument("--out", type=Path)
        q.add_argument("--root", type=Path)
        q.add_argument("--runtime", type=Path)
        q.add_argument("--allow-gpu", action="store_true")
        if name == "audit-history":
            q.add_argument("--history-config", type=Path, required=True)
        if name == "prepare-data":
            q.add_argument("--bindings", type=Path, required=True)
            q.add_argument("--panels", type=Path, required=True)
            q.add_argument("--exposure", type=Path)
            q.add_argument("--data-root")
            q.add_argument("--development-only", action="store_true")
        if name in ("prepare-runtime", "register-development", "register-test", "freeze"):
            q.add_argument("--prepared", type=Path, required=True)
        if name == "prepare-runtime":
            q.add_argument("--historical-runtime", type=Path, required=True)
        if name == "register-development":
            q.add_argument("--expand-after-first-four", action="store_true")
        if name in ("source", "observe-state", "branch"):
            q.add_argument("--lineage", type=int, required=True)
        if name in ("observe-state", "branch", "decide"):
            q.add_argument("--origin", required=True)
        if name == "observe-state":
            q.add_argument("--role", choices=("predecision",), default="predecision")
        if name == "branch":
            q.add_argument("--recipe", required=True)
            q.add_argument("--repeat", type=int, required=True)
        if name == "worker":
            q.add_argument("--task", type=Path, required=True)
        if name == "submit-ready":
            q.add_argument("--max-gpu-jobs", type=int, default=5)
            q.add_argument("--available-gpus", type=int, default=5)
            q.add_argument("--python", required=True)
            q.add_argument("--project-root", type=Path, required=True)
            q.add_argument("--cache-root", type=Path, required=True)
        if name == "fit-selectors":
            q.add_argument("--role", choices=("development",), default="development")
        if name in ("plan-precision", "freeze"):
            q.add_argument("--fit", type=Path, required=True)
        if name == "freeze":
            q.add_argument("--precision", type=Path, required=True)
        if name == "analyze":
            q.add_argument("--phase", choices=("first-four", "development", "final"), required=True)
    return p


def _need(args, *fields):
    for key in fields:
        if getattr(args, key) is None:
            raise ValueError("--" + key.replace("_", "-") + " is required")


def submit_ready(args, config):
    from .jobs import TaskRegistry

    _need(args, "root", "runtime")
    registry = TaskRegistry(args.root, config)
    prefix = "ps2-"

    def query():
        raw = subprocess.check_output(
            ["squeue", "--noheader", "--me", "--format=%i|%j|%b"], text=True
        )
        rows = []
        for line in raw.splitlines():
            job, name, tres = line.split("|")
            if name.startswith(prefix):
                resources = [part.strip() for part in tres.split(",")]
                if resources not in (["gres/gpu:pro6000:1"], ["gpu:pro6000:1"]):
                    raise ValueError("Project job GPU request must be verified single PRO6000")
                rows.append({"job_id": job.strip(), "gpus": 1})
        return rows

    def submit(task, path):
        logs = args.root.resolve() / "logs"
        logs.mkdir(exist_ok=True)
        script = args.project_root.resolve() / "scripts/prospective_selection/launch_worker.sbatch"
        command = [
            "sbatch",
            "--parsable",
            "--job-name=" + prefix + task["task_id"][:12],
            "--output=" + str(logs / "%j.out"),
            "--error=" + str(logs / "%j.err"),
            str(script),
            args.python,
            str(args.project_root.resolve()),
            str(args.root.resolve()),
            str(args.cache_root),
            str(path),
            str(args.runtime.resolve()),
        ]
        return subprocess.check_output(command, text=True).strip()

    submitted = registry.submit_ready(
        query, submit, max_gpu_jobs=args.max_gpu_jobs, available_gpus=args.available_gpus
    )
    return {"status": "SUBMISSIONS_RECORDED", "submissions": submitted}


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_protocol(args.config)
    command = args.command
    if command == "plan":
        result = {
            "status": "PLAN_ONLY_NOT_EXECUTED",
            "protocol": config["protocol"],
            "workload": workload(config),
            "first_delivery_lineages": [61001, 61002, 61003, 61004],
        }
    elif command == "audit-history":
        from .history import audit_history

        _need(args, "out")
        result = audit_history(read_json(args.history_config), args.out)
    elif command == "prepare-data":
        from .dataset import prepare_data

        result = prepare_data(
            config,
            read_json(args.bindings),
            read_json(args.panels),
            read_json(args.exposure) if args.exposure else None,
            data_root=args.data_root,
            development_only=args.development_only,
        )
    elif command == "prepare-runtime":
        from .runtime import prepare_runtime

        _need(args, "out")
        result = prepare_runtime(config, args.historical_runtime, args.prepared, args.out)
    elif command == "register-development":
        from .orchestration import register_development

        _need(args, "root")
        result = register_development(
            config, args.root, read_json(args.prepared), first_four=not args.expand_after_first_four
        )
    elif command == "register-test":
        from .orchestration import register_test

        _need(args, "root")
        result = register_test(config, args.root, read_json(args.prepared))
    elif command == "submit-ready":
        result = submit_ready(args, config)
    elif command in ("worker", "smoke", "source", "observe-state", "branch"):
        from .jobs import TaskRegistry
        from .orchestration import execute_task

        _need(args, "root", "runtime")
        registry = TaskRegistry(args.root, config)
        if command == "worker":
            task = read_json(args.task)
            if args.task.resolve() != (args.root.resolve() / "tasks" / f"{task['task_id']}.json"):
                raise ValueError("Worker must consume canonical registered task file")
        else:
            kind = {"observe-state": "prestate"}.get(command, command)
            matches = [
                t
                for t in registry.tasks()
                if t["kind"] == kind
                and (kind == "smoke" or t["payload"]["lineage_id"] == args.lineage)
                and (kind not in ("prestate", "branch") or t["payload"]["origin_id"] == args.origin)
                and (
                    kind != "branch"
                    or (t["payload"]["recipe_id"], t["payload"]["repeat"])
                    == (args.recipe, args.repeat)
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Exactly one pre-registered task must match; no unguarded execution"
                )
            task = matches[0]
        result = execute_task(
            config, args.root, task["task_id"], args.runtime, allow_gpu=args.allow_gpu
        )
    elif command == "fit-selectors":
        from .workflow import fit_selectors

        _need(args, "root", "out")
        result = fit_selectors(config, args.root, args.out)
    elif command == "plan-precision":
        from .workflow import precision_from_development

        _need(args, "root")
        result = precision_from_development(config, args.root, read_json(args.fit))
    elif command == "freeze":
        from .workflow import freeze_selectors, implementation_id

        _need(args, "root")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True, capture_output=True, check=False,
        )
        if status.returncode == 0:
            if status.stdout.strip():
                raise ValueError("Commit reviewed implementation before scientific freeze")
            version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        else:
            receipt = read_json(Path(__file__).parents[2] / 'CODE_VERSION.json')
            if receipt['implementation_id'] != implementation_id():
                raise ValueError('Deployment code differs from reviewed commit receipt')
            version = receipt['commit']
        result = freeze_selectors(
            config,
            args.root,
            read_json(args.fit),
            read_json(args.precision),
            read_json(args.prepared),
            version,
        )
    elif command == "decide":
        from .workflow import decide

        _need(args, "root")
        result = decide(config, args.root, args.origin)
    elif command == "analyze":
        from .reporting import development_report, final_report

        _need(args, "root", "out")
        result = (
            final_report(config, args.root, args.out)
            if args.phase == "final"
            else development_report(
                config, args.root, args.out, first_four=args.phase == "first-four"
            )
        )
    else:
        raise AssertionError(command)
    if args.out is not None and args.out.suffix == ".json" and command != "prepare-runtime":
        gpu._publish(args.out, result)
    print(
        json.dumps(
            {
                "status": result.get("status", "RECORDED"),
                "command": command,
                "out": str(args.out) if args.out else None,
            },
            ensure_ascii=False,
        )
    )
    return result


if __name__ == "__main__":
    main()
