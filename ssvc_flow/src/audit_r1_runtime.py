"""R1: validate the retained prefix policy; audit but do not adopt faster paths."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import traceback
from pathlib import Path

from .core import canonical_hash, file_hash, load_split, phase_artifacts, source_commit, write_json
from .model_adapters import load_adapter
from .optimizer_fork import parameter_hash
from .r1_parity import (
    ForwardMeter,
    aggregate_parity,
    batch_teacher_scores,
    fixed_sequence_audit,
    parity_comparison,
    production_gate,
    reference_execution_audit,
    reference_panel,
)
from .smoke_runtime import _seed_everything


def _seed_for(scene, interface, index, greedy=False):
    return (
        int(canonical_hash(["R1", scene["base_scene_id"], interface, index, greedy])[:8], 16)
        % 2**31
    )


def _model_spec(config):
    return {
        "id": config["model"]["id"],
        "revision": config["model"]["revision"],
        "expected_layers": 32,
    }


def _progress(out, stage, **details):
    write_json(out / "progress.json", {"status": "RUNNING", "stage": stage, **details})
    print(json.dumps({"stage": stage, **details}), flush=True)


def _run_parity(adapter, scenes, config, out):
    import torch

    from .prompts import build_prompt

    interfaces = ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")
    prepared = {
        (s["base_scene_id"], i): adapter.prepare(build_prompt(s, i), config["data_root"])
        for s in scenes
        for i in interfaces
    }
    initial_hash = parameter_hash(adapter.model, trainable=True)
    meter = ForwardMeter(adapter)
    records, reference_records, generated_rows = [], [], []
    stress_records = []
    reference_by_key = {}

    def record(row, target):
        target.append(row)
        with (out / "parity_records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    try:
        for scene in scenes:
            for interface in interfaces:
                item = prepared[(scene["base_scene_id"], interface)]
                for index in range(5):
                    seed = _seed_for(scene, interface, index, index == 4)
                    with meter.scope("generation"):
                        generated = adapter.generate(
                            item,
                            seed=seed,
                            max_new_tokens=config["max_new_tokens"],
                            do_sample=index != 4,
                        )
                    row = {
                        **generated,
                        "base_scene_id": scene["base_scene_id"],
                        "interface": interface,
                        "sample_index": index,
                        "greedy": index == 4,
                        "sample_seed": seed,
                        "input_audit": item["audit"],
                    }
                    generated_rows.append(row)
                    write_json(out / "fixed_reference_rollouts.json", generated_rows)
                    with meter.scope("audit"):
                        checks = reference_execution_audit(adapter, item, generated, config)
                    for check in checks.values():
                        record({**row, **check, "role": "production_reference"}, reference_records)
                    if not all(c["comparison"]["passed"] for c in checks.values()):
                        raise RuntimeError(
                            "Production reference check failed; no optimizer update allowed"
                        )
                    ref = checks["reference_repeat"]
                    key = (scene["base_scene_id"], interface, index)
                    reference_by_key[key] = (ref["reference_top1"], ref["reference_logp"])
                    adapter.model.eval()
                    for name, mode in (
                        ("teacher_forcing", "full"),
                        ("chunk", "chunk"),
                        ("token", "token"),
                    ):
                        with (
                            torch.no_grad(),
                            meter.scope("teacher_forcing" if mode == "full" else "audit"),
                        ):
                            candidate = adapter.continuation_scores(
                                item, generated["token_ids"], mode=mode
                            )
                        record(
                            {
                                **row,
                                "path": name,
                                "role": "optimization_not_adopted",
                                "reference_top1": ref["reference_top1"],
                                "reference_logp": ref["reference_logp"],
                                "candidate_top1": candidate[0],
                                "candidate_logp": candidate[1],
                                "comparison": parity_comparison(
                                    reference_by_key[key], candidate, config
                                ),
                            },
                            records,
                        )
                    _progress(out, "reference_parity", completed=len(generated_rows), target=120)

        # Adjacent image rows now represent different images; the old prompt-major
        # order could test batch4 without ever placing two distinct images together.
        ordered = sorted(
            generated_rows, key=lambda r: (r["sample_index"], r["interface"], r["base_scene_id"])
        )
        for side in ("left", "right"):
            for size in (1, 2, 4):
                for start in range(0, len(ordered), size):
                    batch = ordered[start : start + size]
                    items = [prepared[(r["base_scene_id"], r["interface"])] for r in batch]
                    with meter.scope("teacher_forcing"):
                        observed = batch_teacher_scores(
                            adapter, items, [r["token_ids"] for r in batch], side
                        )
                    for row, candidate in zip(batch, observed, strict=True):
                        ref = reference_by_key[
                            (row["base_scene_id"], row["interface"], row["sample_index"])
                        ]
                        record(
                            {
                                **row,
                                "path": f"batch{size}_{side}",
                                "role": "optimization_not_adopted",
                                "reference_top1": ref[0],
                                "reference_logp": ref[1],
                                "candidate_top1": candidate[0],
                                "candidate_logp": candidate[1],
                                "comparison": parity_comparison(ref, candidate, config),
                            },
                            records,
                        )
                _progress(out, "batch_parity", batch_size=size, padding=side)

        image_rows = [r for r in ordered if r["interface"] == "IMAGE_CUE_FRESH"][:2]
        for name, batch in (
            ("batch2_reversed_images", image_rows[::-1]),
            ("batch2_repeated_image", [image_rows[0], image_rows[0]]),
        ):
            with meter.scope("teacher_forcing"):
                observed = batch_teacher_scores(
                    adapter,
                    [prepared[(r["base_scene_id"], r["interface"])] for r in batch],
                    [r["token_ids"] for r in batch],
                    "left",
                )
            for row, candidate in zip(batch, observed, strict=True):
                ref = reference_by_key[
                    (row["base_scene_id"], row["interface"], row["sample_index"])
                ]
                record(
                    {
                        **row,
                        "path": name,
                        "role": "optimization_not_adopted",
                        "reference_top1": ref[0],
                        "reference_logp": ref[1],
                        "candidate_top1": candidate[0],
                        "candidate_logp": candidate[1],
                        "comparison": parity_comparison(ref, candidate, config),
                    },
                    records,
                )

        # Explicit boundary cases are fixed audit sequences, never model rollouts
        # or training completions. The 64-token case has no invented terminal EOS.
        tokenizer = adapter.processor.tokenizer
        short = tokenizer.encode("[1,2,3,4]", add_special_tokens=False)
        long = tokenizer.encode("[99,88,77,66]", add_special_tokens=False)
        fixtures = {
            "short_array_with_eos": [*short, min(adapter.eos_ids)],
            "long_array_with_eos": [*long, min(adapter.eos_ids)],
            "truncated_at_64_without_eos": (long * (64 // len(long) + 1))[:64],
        }
        keys = [
            (scenes[0]["base_scene_id"], "SYMBOLIC_FRESH"),
            (scenes[0]["base_scene_id"], "IMAGE_CUE_FRESH"),
            (scenes[1]["base_scene_id"], "IMAGE_CUE_FRESH"),
        ]
        for key in keys:
            for case, tokens in fixtures.items():
                with meter.scope("audit"):
                    checks = fixed_sequence_audit(adapter, prepared[key], tokens, config)
                for check in checks.values():
                    record(
                        {
                            "base_scene_id": key[0],
                            "interface": key[1],
                            "case": case,
                            "token_ids": tokens,
                            "role": "fixed_audit_only_never_training",
                            **check,
                        },
                        stress_records,
                    )
                _progress(out, "fixed_sequence_boundaries", case=case, interface=key[1])
                if not all(c["comparison"]["passed"] for c in checks.values()):
                    raise RuntimeError("Reference fixed-sequence boundary check failed")
    finally:
        meter.handle.remove()
        optimizations = aggregate_parity(records, config)
        references = aggregate_parity(reference_records, config)
        certificate = {
            **production_gate(references, optimizations),
            "model_revision": adapter.revision,
            "initial_adapter_hash": initial_hash,
            "model_audit": adapter.audit,
            "panel_hash": canonical_hash(scenes),
            "source_commit": source_commit(),
            "thresholds": {k: config[k] for k in config if k.startswith("parity_alarm_")},
        }
        certificate["fixed_sequence_boundaries"] = {
            "checks_completed": len(stress_records),
            "checks_expected": 27,
            "passed": len(stress_records) == 27
            and all(r["comparison"]["passed"] for r in stress_records),
            "not_model_generated_not_training": True,
        }
        if not certificate["fixed_sequence_boundaries"]["passed"]:
            certificate["status"] = "FAIL"
        write_json(out / "production_path_validation.json", certificate)
        write_json(out / "forward_meter.json", meter.report())
        write_json(
            out / "likelihood_parity.json",
            {
                "status": "PASS"
                if optimizations and all(x["passed"] for x in optimizations.values())
                else "FAIL",
                "production_path_status": certificate["status"],
                "optimization_paths_adopted": [],
                "paths": optimizations,
                "reference_checks": references,
                "completed_reference_outputs": len(generated_rows),
                "raw_token_evidence": "parity_records.jsonl",
                "I4": "NOT_RUN",
            },
        )
    if certificate["status"] != "PASS":
        raise RuntimeError("Incomplete or failed production path certificate")
    return certificate


def finalize_evidence(out, status, details, execution_kind):
    # Always emit required files on exceptions; NOT_MEASURED never becomes PASS.
    required = (
        "environment_lock.json",
        "runtime_profile.json",
        "likelihood_parity.json",
        "lora_module_manifest.json",
        "training_smoke.json",
        "budget_projection.json",
    )
    for name in required:
        if not (out / name).exists():
            write_json(out / name, {"status": "NOT_MEASURED", "reason": details.get("error")})
    write_json(out / "progress.json", {"status": status, **details})
    document = phase_artifacts(
        out,
        "R1",
        status,
        details,
        [p for p in sorted(out.glob("*.json*")) if p.name not in ("status.json", "manifest.json")],
    )
    document["execution_kind"] = execution_kind
    write_json(out / "status.json", document)
    (out / "report_zh.md").write_text((out / "report.md").read_text(), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke-out", type=Path, required=True)
    args = p.parse_args(argv)
    import torch
    import yaml

    args.out.mkdir(parents=True, exist_ok=True)
    if any(
        (args.out / n).exists()
        for n in ("progress.json", "fixed_reference_rollouts.json", "status.json")
    ):
        raise ValueError("Refusing to overwrite an existing R1 attempt")
    execution_kind, final, error = "CPU_AUDIT", "FAIL", None
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("R1 requires CUDA")
        config = yaml.safe_load(args.config.read_text())
        config["data_root"] = str(args.data_root.resolve())
        config["max_new_tokens"] = int(config["generation_proposed_N"]["max_new_tokens"])
        for key in (
            "parity_alarm_mean_abs_token_logp",
            "parity_alarm_p99_abs_token_logp",
            "parity_alarm_min_top1_agreement",
        ):
            config[key] = config["R1"][key]
        scenes = load_split(
            args.data_root, "calibration", purpose="R1 reference and isolated smoke"
        )
        panel = reference_panel(scenes)
        _seed_everything(17)
        adapter = load_adapter("qwen35_9b", _model_spec(config), image_token_limit=768)
        execution_kind = "REAL_CUDA_INFERENCE"
        write_json(
            args.out / "environment_lock.json",
            {
                "status": "PASS",
                "execution_kind": execution_kind,
                "python": platform.python_version(),
                "model": adapter.audit,
                "source_commit": source_commit(),
                "config": config,
                "config_sha256": file_hash(args.config),
                "calibration_sha256": file_hash(args.data_root / "calibration.jsonl"),
                "device_name": torch.cuda.get_device_name(0),
                "device_memory_total": torch.cuda.get_device_properties(0).total_memory,
            },
        )
        certificate = _run_parity(adapter, panel, config, args.out)
        del adapter
        gc.collect()  # Framework audit hooks hold cycles back to the adapter.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _progress(args.out, "isolated_smoke_initialization")
        _seed_everything(17)
        adapter = load_adapter("qwen35_9b", _model_spec(config), image_token_limit=768)
        from .r1_reference_smoke import run_reference_smoke

        smoke = run_reference_smoke(config, scenes, args.smoke_out, adapter, certificate)
        execution_kind = smoke["execution_kind"]
        write_json(args.out / "training_smoke.json", smoke)
        write_json(
            args.out / "lora_module_manifest.json",
            {
                "status": "PASS" if len(adapter.audit["lora_modules"]) == 96 else "FAIL",
                "modules": adapter.audit["lora_modules"],
                "trainable_parameters": adapter.audit["trainable_parameters"],
            },
        )
        write_json(
            args.out / "runtime_profile.json",
            {
                "status": "MEASURED",
                "parity": json.loads((args.out / "forward_meter.json").read_text()),
                "smoke": {
                    k: v for k, v in smoke.items() if "seconds" in k or "calls" in k or "peak_" in k
                },
            },
        )
        write_json(
            args.out / "budget_projection.json",
            {
                "status": "MEASURED",
                "reference_outputs": 120,
                "smoke_rollouts": smoke.get("raw_sample_count"),
                "optimizer_steps": len(smoke.get("training_steps", [])),
                "generation_tokens_per_second": smoke.get("generation_tokens_per_second"),
                "elapsed_seconds": time.perf_counter() - started,
                "projection_is_not_a_runtime_guarantee": True,
            },
        )
        final = "PASS" if smoke["passed"] else smoke.get("status", "FAIL")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        # Keep measured partial smoke and completed steps when an exception aborts.
        partial = args.smoke_out / "model_audit.json"
        if partial.exists():
            write_json(args.out / "training_smoke.json", json.loads(partial.read_text()))
    details = {
        "R1_gate_passed": final == "PASS",
        "error": error,
        "selected_path": "uncached_prefix_recompute",
        "optimized_paths_adopted": [],
        "later_phases": "NOT_RUN",
        "elapsed_seconds": time.perf_counter() - started,
    }
    finalize_evidence(args.out, final, details, execution_kind)
    return 0 if final == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
