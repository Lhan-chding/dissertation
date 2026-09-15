"""Explicit V3 commands. Planning never imports torch or submits Slurm jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import atomic_json, canonical_hash, finalize_run, sha256_file, source_identity
from .schema import freeze_selection, load_config, verify_selection_lock


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    names = [
        "plan",
        "audit-parent",
        "prepare-historical",
        "test",
        "collect-toy",
        "observe-toy",
        "coverage-study",
        "freeze",
        "validate-cpu",
        "plan-gpu",
        "prepare-gpu",
        "merge-q4",
        "prepare-source",
        "prepare-forks",
        "prepare-observation",
        "prepare-reference-extension",
        "prepare-geometry",
        "prepare-fit",
        "freeze-predictions",
        "freeze-prediction-set",
        "prepare-evaluation",
        "analyze-observation",
        "analyze-coverage",
        "finalize-stage",
        "calibrate-cpu",
        "analyze-cpu",
        "analyze-cpu-generalization",
        "freeze-vlm",
        "calibrate-vlm",
        "analyze-vlm",
        "prepare-q6",
        "prepare-q6-forks",
        "measure-q6-absolute",
        "vlm-smoke",
        "train-source",
        "make-forks",
        "observe-vlm",
        "select",
        "fit",
        "evaluate",
        "track-offline",
        "summarize",
    ]
    for name in names:
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--resume", action="store_true")
        if name == "audit-parent":
            p.add_argument("--parent", required=True)
            p.add_argument("--legacy-parent")
        if name == "prepare-historical":
            p.add_argument("--parent", required=True)
        if name in {"collect-toy", "observe-toy", "coverage-study", "validate-cpu"}:
            p.add_argument(
                "--pilot", action="store_true", help="timing namespace; never confirmation"
            )
            p.add_argument("--role", default="development")
        if name in {"observe-toy", "coverage-study"}:
            p.add_argument("--collection", required=True)
        if name == "collect-toy":
            p.add_argument("--seeds", nargs="+", type=int)
            p.add_argument("--arms", nargs="+", choices=["X_BASE", "X_VALID"])
        if name == "freeze":
            p.add_argument("--inputs", nargs="+", required=True)
            p.add_argument("--selection", required=True)
        if name == "validate-cpu":
            p.add_argument("--lock", required=True)
            p.add_argument("--existing-run-roots", nargs="+", required=True)
            p.add_argument("--calibration-receipt")
            p.add_argument("--seeds", nargs="+", type=int)
            p.add_argument("--arms", nargs="+", choices=["X_BASE", "X_VALID"])
        if name == "prepare-gpu":
            p.add_argument("--bindings", required=True)
            p.add_argument("--cpu-tests", required=True)
            p.add_argument("--development-roots", nargs=2, required=True)
        if name in {"calibrate-cpu", "analyze-cpu", "analyze-cpu-generalization"}:
            p.add_argument("--lock", required=True)
            p.add_argument("--inputs", nargs="+", required=True)
        if name in {"analyze-cpu", "analyze-cpu-generalization"}:
            p.add_argument("--calibration-receipt", required=True)
        if name == "freeze-vlm":
            p.add_argument("--parent-cpu-lock", required=True)
            p.add_argument("--development-completion", required=True)
            p.add_argument("--selection", required=True)
            p.add_argument("--inputs", nargs="+", required=True)
        if name in {"calibrate-vlm", "analyze-vlm"}:
            p.add_argument("--lock", required=True)
            p.add_argument("--stage-completion", required=True)
            p.add_argument("--inputs", nargs="+", required=True)
        if name == "analyze-vlm":
            p.add_argument("--calibration-receipt", required=True)
        if name == "prepare-q6":
            for field in ("bindings", "qualification", "fit-spec", "model", "source-result"):
                p.add_argument("--" + field, required=True)
            p.add_argument("--phase", choices=["anchor", "reference"], default="anchor")
            p.add_argument("--previous-measurement")
            p.add_argument("--prediction-lock")
        if name == "prepare-q6-forks":
            for field in ("bindings", "qualification", "design-id", "source-root"):
                p.add_argument("--" + field, required=True)
        if name == "measure-q6-absolute":
            p.add_argument("--plan", required=True)
            p.add_argument("--worker-root", required=True)
        if name == "merge-q4":
            p.add_argument("--worker-receipts", required=True)
        if name in {
            "prepare-source",
            "prepare-forks",
            "prepare-observation",
            "prepare-reference-extension",
        }:
            p.add_argument("--bindings", required=True)
        if name == "prepare-source":
            p.add_argument("--q4-receipt", required=True)
            p.add_argument(
                "--role",
                required=True,
                choices=["development", "interval_calibration", "locked_test"],
            )
            p.add_argument("--lock")
            p.add_argument("--previous-completion")
            p.add_argument("--calibration-receipt")
        if name == "prepare-forks":
            p.add_argument("--source-root", required=True)
        if name == "prepare-observation":
            p.add_argument("--forks-root", required=True)
            p.add_argument("--origin-checkpoint", required=True)
            p.add_argument("--purpose", choices=["measurement", "reference"], required=True)
            p.add_argument("--operation", choices=["generate", "score"], required=True)
            p.add_argument("--generation-manifest")
            p.add_argument("--generation-root")
            p.add_argument("--prediction-lock")
        if name == "prepare-reference-extension":
            p.add_argument("--precision-receipt", required=True)
            p.add_argument("--operation", choices=["generate", "score"], required=True)
            p.add_argument("--generation-manifest")
            p.add_argument("--generation-root")
        if name == "prepare-geometry":
            p.add_argument("--forks-root", required=True)
            p.add_argument("--n-banks", required=True, type=int)
            p.add_argument("--selector", required=True)
            p.add_argument("--seed", type=int, default=2026091500)
            p.add_argument("--lock")
        if name == "prepare-fit":
            for field in (
                "forks-root",
                "measurement-bundle",
                "geometry-spec",
                "selection-root",
                "model-spec",
            ):
                p.add_argument("--" + field, required=True)
            p.add_argument("--lock")
        if name == "freeze-predictions":
            p.add_argument("--fit-spec", required=True)
            p.add_argument("--fit-root", required=True)
            p.add_argument("--calibration-receipt")
        if name == "freeze-prediction-set":
            p.add_argument("--inputs", nargs="+", required=True)
        if name == "prepare-evaluation":
            for field in ("prediction-lock", "measurement-bundle", "reference-bundle"):
                p.add_argument("--" + field, required=True)
        if name in {"analyze-observation", "analyze-coverage"}:
            p.add_argument("--inputs", nargs="+", required=True)
        if name == "finalize-stage":
            p.add_argument("--bindings", required=True)
            p.add_argument("--task-receipts", required=True)
        if name in {"vlm-smoke", "train-source", "make-forks", "observe-vlm"}:
            p.add_argument("--bindings", required=True)
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--allow-gpu", action="store_true")
            p.add_argument("--allow-training", action="store_true")
            p.add_argument("--acknowledge-new-experiment", action="store_true")
        if name == "train-source":
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--arm", choices=["X_BASE", "X_VALID"], required=True)
        if name == "make-forks":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--bank-plan", required=True)
        if name == "observe-vlm":
            p.add_argument("--task-manifest", required=True)
            p.add_argument("--worker-index", type=int, required=True)
            p.add_argument("--workers", type=int, default=2)
        if name == "vlm-smoke":
            p.add_argument("--worker-index", type=int, choices=[0, 1], default=0)
        if name in {"select", "fit", "evaluate", "track-offline", "summarize"}:
            p.add_argument("--inputs", nargs="+", required=True)
        if name == "test":
            p.add_argument("--selectors", nargs="+", default=["tests/modeling_v3"])
    return result


def _read(path):
    return json.loads(Path(path).read_text())


def _file_binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _plan(args, config):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "PLANNED_NOT_EXECUTED",
        "command": args.command,
        "config_sha256": canonical_hash(config),
        "source": source_identity(),
        "arguments": vars(args),
        "derived_workload": config["derived_workload"],
        "first_gpu_submission_requires_user_authorization": True,
        "large_cpu_execution_location": "server",
        "online_ssvc": False,
    }
    atomic_json(out / "PLAN.json", result)
    finalize_run(out, {"source": result["source"]["sha256"], "config": result["config_sha256"]})
    return result


def dispatch(args):
    config = load_config(args.config)
    if args.dry_run or args.command == "plan":
        return _plan(args, config)
    if args.command == "audit-parent":
        from .source_audit import audit_parent

        if args.legacy_parent:
            config = {**config, "q0": {"legacy_parent_root": args.legacy_parent}}
        return audit_parent(config, args.parent, args.out)
    if args.command == "prepare-historical":
        from .historical_observation import prepare_historical_observation_collection

        return prepare_historical_observation_collection(args.parent, args.out, config)
    if args.command == "collect-toy":
        from .cpu_campaign import collect_toy

        return collect_toy(
            config,
            args.out,
            role=args.role,
            pilot=args.pilot,
            resume=args.resume,
            seed_subset=args.seeds,
            arm_subset=args.arms,
        )
    if args.command == "observe-toy":
        from .cpu_campaign import observe_toy

        return observe_toy(config, args.collection, args.out, pilot=args.pilot, resume=args.resume)
    if args.command == "coverage-study":
        from .cpu_campaign import coverage_study

        return coverage_study(
            config, args.collection, args.out, pilot=args.pilot, resume=args.resume
        )
    if args.command == "freeze":
        return freeze_selection(config, args.inputs, _read(args.selection), args.out)
    if args.command == "validate-cpu":
        from .cpu_campaign import validate_cpu

        lock = verify_selection_lock(config, args.lock)
        return validate_cpu(
            config,
            lock,
            args.out,
            pilot=args.pilot,
            resume=args.resume,
            roles=None if args.role == "development" else [args.role],
            existing_run_roots=args.existing_run_roots,
            seed_subset=args.seeds,
            arm_subset=args.arms,
            calibration_receipt=args.calibration_receipt,
        )
    if args.command == "plan-gpu":
        from .vlm_campaign import plan_gpu

        return plan_gpu(config, args.out)
    if args.command == "prepare-gpu":
        from .server_preparation import prepare_gpu

        return prepare_gpu(
            config,
            _read(args.bindings),
            args.cpu_tests,
            args.out,
            development_roots=args.development_roots,
        )
    if args.command == "merge-q4":
        from .vlm_campaign import finalize_q4_bridge

        return finalize_q4_bridge(config, _read(args.worker_receipts), out=args.out)
    if args.command == "prepare-source":
        from .vlm_campaign import prepare_q5_stage

        return prepare_q5_stage(
            config,
            _read(args.bindings),
            q4_receipt=_file_binding(args.q4_receipt),
            role=args.role,
            out=args.out,
            selection_lock=_file_binding(args.lock) if args.lock else None,
            previous_completion=_file_binding(args.previous_completion)
            if args.previous_completion
            else None,
            calibration_receipt=_file_binding(args.calibration_receipt)
            if args.calibration_receipt
            else None,
        )
    if args.command == "prepare-forks":
        from .vlm_campaign import prepare_q5_forks

        return prepare_q5_forks(
            config, _read(args.bindings), source_root=args.source_root, out=args.out
        )
    if args.command == "prepare-observation":
        from .vlm_campaign import prepare_q5_observation

        return prepare_q5_observation(
            config,
            _read(args.bindings),
            forks_root=args.forks_root,
            origin_checkpoint=args.origin_checkpoint,
            out=args.out,
            purpose=args.purpose,
            operation=args.operation,
            generation_manifest=_file_binding(args.generation_manifest)
            if args.generation_manifest
            else None,
            generation_root=args.generation_root,
            prediction_binding=_file_binding(args.prediction_lock)
            if args.prediction_lock
            else None,
            include_direct_count=True,
        )
    if args.command == "prepare-geometry":
        from .vlm_response import prepare_q5_geometry

        return prepare_q5_geometry(
            config,
            forks_root=args.forks_root,
            n_banks=args.n_banks,
            selector=args.selector,
            seed=args.seed,
            selection_lock=args.lock,
            out=args.out,
        )
    if args.command == "prepare-reference-extension":
        from .vlm_campaign import prepare_q5_reference_extension

        return prepare_q5_reference_extension(
            config,
            _read(args.bindings),
            precision_receipt=_file_binding(args.precision_receipt),
            out=args.out,
            operation=args.operation,
            generation_manifest=_file_binding(args.generation_manifest)
            if args.generation_manifest
            else None,
            generation_root=args.generation_root,
        )
    if args.command == "prepare-fit":
        from .vlm_response import prepare_q5_fit

        return prepare_q5_fit(
            config,
            forks_root=args.forks_root,
            measurement_bundle=args.measurement_bundle,
            geometry_spec=args.geometry_spec,
            selection_root=args.selection_root,
            model_spec=args.model_spec,
            selection_lock=args.lock,
            out=args.out,
        )
    if args.command == "freeze-predictions":
        from .vlm_response import freeze_q5_predictions

        return freeze_q5_predictions(
            config,
            fit_spec=args.fit_spec,
            fit_root=args.fit_root,
            calibration_receipt=args.calibration_receipt,
            out=args.out,
        )
    if args.command == "prepare-evaluation":
        from .vlm_response import prepare_q5_evaluation

        return prepare_q5_evaluation(
            config,
            prediction_lock=args.prediction_lock,
            measurement_bundle=args.measurement_bundle,
            reference_bundle=args.reference_bundle,
            out=args.out,
        )
    if args.command == "freeze-prediction-set":
        from .vlm_campaign import freeze_q5_prediction_set

        return freeze_q5_prediction_set(
            config, [_file_binding(path) for path in args.inputs], out=args.out
        )
    if args.command == "analyze-observation":
        from .q1_results import analyze_observation

        return analyze_observation(config, args.inputs, args.out, resume=args.resume)
    if args.command == "analyze-coverage":
        from .q2_results import analyze_coverage

        return analyze_coverage(config, args.inputs, args.out)
    if args.command == "finalize-stage":
        from .vlm_campaign import finalize_q5_stage

        return finalize_q5_stage(
            config, _read(args.bindings), task_receipts=args.task_receipts, out=args.out
        )
    if args.command in {"calibrate-cpu", "analyze-cpu"}:
        from .cpu_results import analyze_calibration, analyze_test

        lock = verify_selection_lock(config, args.lock)
        if args.command == "calibrate-cpu":
            return analyze_calibration(config, lock, args.inputs, args.out)
        return analyze_test(
            config, lock, args.inputs, args.out, calibration_receipt=args.calibration_receipt
        )
    if args.command == "analyze-cpu-generalization":
        from .cpu_generalization import analyze_generalization

        return analyze_generalization(
            config,
            verify_selection_lock(config, args.lock),
            args.inputs,
            args.out,
            calibration_receipt=args.calibration_receipt,
        )
    if args.command == "freeze-vlm":
        from .vlm_response import freeze_vlm_selection

        return freeze_vlm_selection(
            config,
            args.parent_cpu_lock,
            development_completion=args.development_completion,
            evaluation_inputs=args.inputs,
            selected=_read(args.selection),
            out=args.out,
        )
    if args.command in {"calibrate-vlm", "analyze-vlm"}:
        from .vlm_results import analyze_vlm_calibration, analyze_vlm_test

        kwargs = {
            "stage_completion": args.stage_completion,
            "evaluation_inputs": args.inputs,
            "out": args.out,
        }
        if args.command == "calibrate-vlm":
            return analyze_vlm_calibration(config, args.lock, **kwargs)
        return analyze_vlm_test(
            config, args.lock, calibration_receipt=args.calibration_receipt, **kwargs
        )
    if args.command == "prepare-q6":
        from .q6_observation import prepare_q6_observation

        return prepare_q6_observation(
            config,
            _read(args.bindings),
            qualification_binding=_file_binding(args.qualification),
            fit_binding=_file_binding(args.fit_spec),
            model_binding=_file_binding(args.model),
            source_binding=_file_binding(args.source_result),
            out=args.out,
            phase=args.phase,
            previous_measurement=_file_binding(args.previous_measurement)
            if args.previous_measurement
            else None,
            prediction_lock=_file_binding(args.prediction_lock) if args.prediction_lock else None,
        )
    if args.command == "prepare-q6-forks":
        from .vlm_campaign import prepare_q6_forks

        return prepare_q6_forks(
            config,
            _read(args.bindings),
            qualification_binding=_file_binding(args.qualification),
            design_id=args.design_id,
            source_root=args.source_root,
            out=args.out,
        )
    if args.command == "measure-q6-absolute":
        from .q6_observation import measure_q6_absolute

        return measure_q6_absolute(
            config,
            plan_binding=_file_binding(args.plan),
            worker_root=args.worker_root,
            out=args.out,
        )
    if args.command in {"vlm-smoke", "train-source", "make-forks", "observe-vlm"}:
        if not args.allow_gpu or not args.acknowledge_new_experiment:
            raise PermissionError(
                "explicit GPU and new-experiment execution acknowledgement required"
            )
        bindings = _read(args.bindings)
        if args.command == "train-source":
            from .vlm_campaign import train_source

            return train_source(
                config,
                bindings,
                seed=args.seed,
                arm=args.arm,
                device=args.device,
                out=args.out,
                allow_gpu=args.allow_gpu,
                allow_training=args.allow_training,
                acknowledge_new_experiment=args.acknowledge_new_experiment,
                resume=args.resume,
            )
        if args.command == "make-forks":
            from .vlm_campaign import make_forks

            return make_forks(
                config,
                bindings,
                checkpoint=args.checkpoint,
                bank_plan=_read(args.bank_plan),
                device=args.device,
                out=args.out,
                allow_gpu=args.allow_gpu,
                allow_training=args.allow_training,
                acknowledge_new_experiment=args.acknowledge_new_experiment,
                resume=args.resume,
            )
        if args.command == "vlm-smoke":
            from .vlm_campaign import vlm_smoke

            return vlm_smoke(
                config,
                bindings,
                device=args.device,
                out=args.out,
                allow_gpu=args.allow_gpu,
                acknowledge_new_experiment=args.acknowledge_new_experiment,
                worker_index=args.worker_index,
                resume=args.resume,
            )
        from . import vlm_observation

        return vlm_observation.observe_vlm(
            config,
            bindings,
            task_manifest=args.task_manifest,
            worker_index=args.worker_index,
            workers=args.workers,
            device=args.device,
            out=args.out,
            resume=args.resume,
            allow_gpu=args.allow_gpu,
            acknowledge_new_experiment=args.acknowledge_new_experiment,
        )
    if args.command == "track-offline":
        from .tracking_offline import track_offline

        return track_offline(_read(args.inputs[0]), args.out, config=config)
    if args.command in {"select", "fit", "evaluate", "summarize"}:
        from .workflow import run_artifact_command

        return run_artifact_command(args.command, config, args.inputs, args.out)
    if args.command == "test":
        from .workflow import run_tests

        return run_tests(args.out, args.selectors)
    raise ValueError("unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        value = dispatch(args)
    except (ValueError, PermissionError, FileExistsError, FileNotFoundError) as error:
        print(
            json.dumps({"status": "BLOCKED", "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    # Full results are saved in their output directory; stdout remains compact.
    print(
        json.dumps(
            {
                "command": args.command,
                "out": args.out,
                "status": value.get("status", "RECORDED")
                if isinstance(value, dict)
                else "RECORDED",
            },
            ensure_ascii=False,
        )
    )
    if args.command == "test" and isinstance(value, dict):
        code = value.get("exit_code", 1)
        return code if isinstance(code, int) and 0 <= code <= 255 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
