"""Explicit V4 planning, collection, modeling and factual summary commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import atomic_json, load_config, plan, source_record


def _read(path):
    return json.loads(Path(path).read_text())


def _array_document(path, keys):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in keys}


def _npz(path, **values):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    config_default = "configs/modeling_v4/protocol.json"
    for name in ("plan", "cpu-diagnose", "build-task-list"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=config_default)
        p.add_argument("--out", required=True)
        if name == "cpu-diagnose":
            p.add_argument("--reuse", required=True)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--origin-subset", help="Comma-separated recorded origin IDs")
        elif name == "build-task-list":
            p.add_argument("--bindings", default=os.environ.get("SSVC_V4_BINDINGS"))
            p.add_argument("--workers", type=int, default=2)
            p.add_argument(
                "--phase",
                choices=(
                    "bridge",
                    "first-map",
                    "bridge-and-first-map",
                    "development",
                    "confirmation",
                    "tracking",
                ),
                default="bridge-and-first-map",
            )
    p = sub.add_parser("run-worker")
    p.add_argument("--tasks", required=True)
    p.add_argument("--worker-id", required=True, type=int)
    p.add_argument("--workers", required=True, type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execute-gpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("fit")
    p.add_argument("--data", required=True, help="NPZ arrays E and Y, using fit-role data only")
    p.add_argument("--out", required=True)
    p.add_argument("--method", default="FULL_DUAL_RIDGE")
    p.add_argument("--alpha", type=float, default=1e-5, help="Absolute ridge penalty lambda")
    p.add_argument("--rank-cap", default="FULL")
    p.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    p = sub.add_parser("export-calibration-labels")
    p.add_argument("--tasks", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fit-signature-development")
    p.add_argument("--config", default=config_default)
    p.add_argument(
        "--origin-inputs",
        required=True,
        help="JSON list of six development signature/label bindings",
    )
    p.add_argument(
        "--model-specs", help="JSON list of preregistered development model specifications"
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out", required=True)
    p = sub.add_parser("freeze-signature-models")
    p.add_argument("--report", required=True, help="Completed SIGNATURE_CV.json")
    p.add_argument(
        "--model-ids", required=True, help="JSON list of actually available CV model IDs"
    )
    p.add_argument("--out", required=True)
    p = sub.add_parser("predict-signature-models")
    p.add_argument("--selection", required=True, help="Completed SIGNATURE_SELECTION.json")
    p.add_argument(
        "--signature", required=True, help="Completed signature receipt for the target origin"
    )
    p.add_argument("--out", required=True)
    p = sub.add_parser("extend-task-list")
    p.add_argument("--tasks", required=True)
    p.add_argument(
        "--phase", choices=("development", "calibration", "test", "tracking"), required=True
    )
    p.add_argument("--first-map-evidence")
    p.add_argument("--selection")
    p.add_argument("--calibration")
    p.add_argument("--point-response-evidence")
    p.add_argument("--out", required=True)
    p = sub.add_parser("freeze-selection")
    p.add_argument("--config", default=config_default)
    p.add_argument("--first-map-evidence", required=True)
    p.add_argument("--development-evidence", required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument(
        "--bank-selector", choices=("STRATIFIED_RANDOM", "BLOCK_PIVOT_QR"), required=True
    )
    p.add_argument(
        "--primary-models", required=True, help="JSON list of two actual selected model records"
    )
    p.add_argument("--gradient-reference", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("freeze-calibration")
    p.add_argument("--config", default=config_default)
    p.add_argument("--selection", required=True)
    p.add_argument("--calibration-evidence", required=True)
    p.add_argument(
        "--thresholds", required=True, help="JSON empirical thresholds from calibration only"
    )
    p.add_argument("--out", required=True)
    p = sub.add_parser("phase-evidence")
    p.add_argument("--tasks", required=True)
    p.add_argument(
        "--fit-results", required=True, help="JSON mapping task IDs to completed fit bundles"
    )
    p.add_argument("--phase", choices=("C", "D_DEVELOPMENT", "D_CALIBRATION"), default="C")
    p.add_argument("--selection")
    p.add_argument("--cv-receipts", help="JSON list of actual completed whole-seed CV receipts")
    p.add_argument("--out", required=True)
    p = sub.add_parser("point-response-evidence")
    p.add_argument("--config", default=config_default)
    p.add_argument("--selection", required=True)
    p.add_argument("--calibration-evidence")
    p.add_argument("--out", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="NPZ array queries; no reference labels read")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare-kernels")
    p.add_argument("--data", required=True, help="NPZ E, Y and queries")
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=1e-5)
    p = sub.add_parser("collect-score-response")
    p.add_argument("--tasks", required=True)
    p.add_argument("--worker-id", required=True, type=int)
    p.add_argument("--workers", required=True, type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execute-gpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("summarize")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fit-response-map")
    p.add_argument("--tasks", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=1e-5)
    p.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    p.add_argument("--calibration-banks", type=int, choices=(24, 32, 48, 96))
    p.add_argument("--bank-selector", choices=("STRATIFIED_RANDOM", "BLOCK_PIVOT_QR"))
    p.add_argument("--draw-count", type=int, choices=(256, 1024, 4096))
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "plan":
        result = {**plan(load_config(args.config)), "source": source_record(Path.cwd())}
        target = Path(args.out)
        atomic_json(target if target.suffix == ".json" else target / "PLAN.json", result)
    elif args.command == "cpu-diagnose":
        from .cpu_diagnose import run_cpu_diagnose

        reuse = _read(args.reuse) if Path(args.reuse).is_file() else args.reuse
        subset = args.origin_subset.split(",") if args.origin_subset else None
        result = run_cpu_diagnose(
            load_config(args.config), reuse, args.out, resume=args.resume, origin_subset=subset
        )
    elif args.command == "build-task-list":
        from .gpu_collect import build_task_list

        if not args.bindings:
            raise ValueError("Supply --bindings or SSVC_V4_BINDINGS for actual inherited inputs")
        if args.phase in {"development", "confirmation", "tracking"}:
            raise ValueError("Use extend-task-list with completed phase evidence for later stages")
        stages = {
            "bridge": ("B",),
            "first-map": ("C",),
            "bridge-and-first-map": ("B", "C"),
        }[args.phase]
        result = build_task_list(
            load_config(args.config),
            _read(args.bindings),
            out=args.out,
            workers=args.workers,
            stages=stages,
        )
    elif args.command in {"run-worker", "collect-score-response"}:
        from .gpu_collect import run_worker

        if args.command == "collect-score-response":
            from .gpu_collect import collect_score_response_worker

            operation = collect_score_response_worker
        else:
            operation = run_worker
        result = operation(
            args.tasks,
            worker_id=args.worker_id,
            workers=args.workers,
            device=args.device,
            execute_gpu=args.execute_gpu,
            resume=args.resume,
        )
    elif args.command == "fit":
        from .full_kernels import fit_full_response, save_model

        arrays = _array_document(args.data, ("E", "Y"))
        cap = args.rank_cap if args.rank_cap == "FULL" else int(args.rank_cap)
        model = fit_full_response(
            arrays["E"],
            arrays["Y"],
            method=args.method,
            alpha=args.alpha,
            rank_cap=cap,
            bandwidth_multiplier=args.bandwidth_multiplier,
        )
        save_model(model, args.out)
        result = {
            "status": "FIT_COMPLETE",
            "model": str(Path(args.out).resolve()),
            "method": args.method,
            "alpha_is_absolute_lambda": True,
        }
    elif args.command == "predict":
        from .full_kernels import load_model

        arrays = _array_document(args.data, ("queries",))
        model = load_model(args.model)
        prediction = model["predict"](arrays["queries"])
        _npz(args.out, prediction=prediction)
        result = {
            "status": "PREDICTED",
            "reference_labels_read": False,
            "shape": list(prediction.shape),
        }
    elif args.command == "compare-kernels":
        from .full_kernels import fit_full_response

        arrays = _array_document(args.data, ("E", "Y", "queries"))
        output = {}
        for method in ("FULL_DUAL_RIDGE", "FULL_RBF_RAW", "LEGACY_QPROJECTED_RBF"):
            model = fit_full_response(arrays["E"], arrays["Y"], method=method, alpha=args.alpha)
            output[method] = model["predict"](arrays["queries"])
        _npz(args.out, **output)
        result = {
            "status": "PREDICTED_FOR_COMPARISON",
            "methods": list(output),
            "reference_labels_read": False,
            "accuracy_evaluated": False,
        }
    elif args.command == "summarize":
        from .report import summarize

        result = summarize(args.root, args.out)
    elif args.command == "fit-response-map":
        from .response_fit import fit_response_map

        result = fit_response_map(
            args.tasks,
            task_id=args.task_id,
            out=args.out,
            alpha=args.alpha,
            bandwidth_multiplier=args.bandwidth_multiplier,
            calibration_banks=args.calibration_banks,
            bank_selector=args.bank_selector,
            draw_count=args.draw_count,
        )
    elif args.command == "export-calibration-labels":
        from .response_fit import export_calibration_labels

        result = export_calibration_labels(args.tasks, task_id=args.task_id, out=args.out)
    elif args.command == "fit-signature-development":
        from .signature_fit import fit_signature_development

        result = fit_signature_development(
            load_config(args.config),
            _read(args.origin_inputs),
            out=args.out,
            model_specs=_read(args.model_specs) if args.model_specs else None,
            epochs=args.epochs,
        )
    elif args.command == "freeze-signature-models":
        from .signature_fit import freeze_signature_models

        result = freeze_signature_models(args.report, _read(args.model_ids), out=args.out)
    elif args.command == "predict-signature-models":
        from .signature_fit import predict_signature_models

        result = predict_signature_models(args.selection, args.signature, out=args.out)
    elif args.command in {"freeze-selection", "freeze-calibration"}:
        from ..modeling_v3.io import atomic_json as publish
        from .phase_evidence import verify_phase_evidence
        from .phase_schedule import freeze_calibration, freeze_selection

        config = load_config(args.config)
        if args.command == "freeze-selection":
            result = freeze_selection(
                config,
                verify_phase_evidence(_read(args.first_map_evidence)),
                verify_phase_evidence(_read(args.development_evidence)),
                m=args.m,
                bank_selector=args.bank_selector,
                primary_models=_read(args.primary_models),
                gradient_reference=args.gradient_reference,
            )
        else:
            result = freeze_calibration(
                config,
                _read(args.selection),
                verify_phase_evidence(
                    _read(args.calibration_evidence), selection=_read(args.selection)
                ),
                thresholds=_read(args.thresholds),
            )
        publish(Path(args.out), result)
    elif args.command == "point-response-evidence":
        from .point_response_evidence import build_point_response_evidence

        result = build_point_response_evidence(
            load_config(args.config),
            _read(args.selection),
            _read(args.calibration_evidence) if args.calibration_evidence else None,
            out=args.out,
        )
    elif args.command == "extend-task-list":
        from ..modeling_v3.io import atomic_json as publish
        from .phase_evidence import verify_phase_evidence
        from .phase_schedule import (
            append_phase_extension,
            build_confirmation_extension,
            build_development_extension,
            build_tracking_extension,
        )

        tasks = _read(args.tasks)
        config, workers = tasks["config"], tasks["workers"]
        if args.phase == "tracking":
            if not args.selection:
                raise ValueError("Tracking requires the actual frozen selection")
            extension = build_tracking_extension(
                config,
                _read(args.selection),
                _read(args.point_response_evidence) if args.point_response_evidence else None,
                workers=workers,
            )
        else:
            if not args.first_map_evidence:
                raise ValueError("Completed first-map collection/fit/evaluation evidence required")
            first = verify_phase_evidence(_read(args.first_map_evidence))
            if args.phase == "development":
                extension = build_development_extension(config, first, workers=workers)
            else:
                if not args.selection:
                    raise ValueError("Confirmation requires the frozen development selection")
                extension = build_confirmation_extension(
                    config,
                    first,
                    _read(args.selection),
                    role=args.phase,
                    workers=workers,
                    calibration=_read(args.calibration) if args.calibration else None,
                )
        result = append_phase_extension(tasks, extension)
        publish(Path(args.out), result)
    elif args.command == "phase-evidence":
        from ..modeling_v3.io import atomic_json as publish
        from .phase_evidence import build_phase_evidence

        result = build_phase_evidence(
            args.tasks,
            fit_results=_read(args.fit_results),
            phase=args.phase,
            selection=_read(args.selection) if args.selection else None,
            cv_receipts=_read(args.cv_receipts) if args.cv_receipts else None,
        )
        publish(Path(args.out), result)
    else:
        raise AssertionError(args.command)
    # Scientific arrays remain in their named artifacts, not an enormous terminal dump.
    if isinstance(result, dict):
        short = {
            key: result[key]
            for key in (
                "status",
                "schema",
                "campaign_id",
                "root",
                "out",
                "methods",
                "measured_runtime",
            )
            if key in result
        }
        print(json.dumps(short or {"command": args.command, "completed": True}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
