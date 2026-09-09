"""Execute the locked R1 reference parity and isolated four-step smoke on CUDA."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from .core import canonical_hash, load_split, phase_artifacts, write_json
from .model_adapters import load_adapter
from .r1_parity import (
    ForwardMeter,
    aggregate_parity,
    batch_teacher_scores,
    parity_comparison,
    prefix_scores,
    reference_panel,
)


def _seed_for(scene, interface, index, greedy=False):
    return int(canonical_hash(["R1", scene["base_scene_id"], interface, index, greedy])[:8], 16) % (
        2**31
    )


def _model_spec(config):
    return {
        "id": config["model"]["id"],
        "revision": config["model"]["revision"],
        "expected_layers": 32,
    }


def _run_parity(adapter, scenes, config, out):
    from .prompts import build_prompt

    interfaces = ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")
    prepared = {}
    for scene in scenes:
        for interface in interfaces:
            prepared[(scene["base_scene_id"], interface)] = adapter.prepare(
                build_prompt(scene, interface), config["data_root"]
            )
    meter = ForwardMeter(adapter)
    records = []
    generation_rows = []
    try:
        for scene in scenes:
            for interface in interfaces:
                item = prepared[(scene["base_scene_id"], interface)]
                for index in range(5):
                    greedy = index == 4
                    seed = _seed_for(scene, interface, index, greedy)
                    with meter.scope("generation"):
                        generated = adapter.generate(
                            item,
                            seed=seed,
                            max_new_tokens=config["max_new_tokens"],
                            do_sample=not greedy,
                        )
                    completion = generated["token_ids"]
                    # Reference is the same fixed sequence scored through the audited
                    # prefix-recompute path. No generated text is injected into training.
                    with meter.scope("audit"):
                        ref_top, ref_logp = prefix_scores(adapter, item, completion)
                    with meter.scope("teacher_forcing"):
                        paths = {
                            "teacher_forcing": adapter.continuation_scores(
                                item, completion, mode="full"
                            )
                        }
                    for mode in ("chunk", "token"):
                        with meter.scope("audit"):
                            paths[mode] = adapter.continuation_scores(item, completion, mode=mode)
                    for path, candidate in paths.items():
                        comparison = parity_comparison((ref_top, ref_logp), candidate, config)
                        records.append(
                            {
                                "base_scene_id": scene["base_scene_id"],
                                "interface": interface,
                                "sample_index": index,
                                "greedy": greedy,
                                "path": path,
                                "token_ids": completion,
                                "reference_top1": ref_top,
                                "reference_logp": ref_logp,
                                "candidate_top1": candidate[0],
                                "candidate_logp": candidate[1],
                                "comparison": comparison,
                            }
                        )
                    generation_rows.append(
                        {
                            "base_scene_id": scene["base_scene_id"],
                            "interface": interface,
                            "sample_index": index,
                            "greedy": greedy,
                            "sample_seed": seed,
                            "raw_completion": generated["raw_completion"],
                            "token_ids": completion,
                            "stop_reason": generated["stop_reason"],
                            "n_generated_tokens": len(completion),
                            "behavior_token_logprobs": generated["behavior_token_logprobs"],
                            "vision_forward_calls": generated["vision_forward_calls"],
                        }
                    )
        # Compare fixed sequences after generation in batch 1/2/4, with both padding
        # directions. This does not replace the unpadded reference path.
        for side in ("left", "right"):
            for batch_size in (1, 2, 4):
                for start in range(0, len(generation_rows), batch_size):
                    batch = generation_rows[start : start + batch_size]
                    items = [prepared[(r["base_scene_id"], r["interface"])] for r in batch]
                    completions = [r["token_ids"] for r in batch]
                    with meter.scope("teacher_forcing"):
                        observed = batch_teacher_scores(adapter, items, completions, side)
                    for row, candidate in zip(batch, observed, strict=True):
                        ref = next(
                            r
                            for r in records
                            if r["base_scene_id"] == row["base_scene_id"]
                            and r["interface"] == row["interface"]
                            and r["sample_index"] == row["sample_index"]
                            and r["path"] == "teacher_forcing"
                        )
                        comparison = parity_comparison(
                            (ref["reference_top1"], ref["reference_logp"]), candidate, config
                        )
                        records.append(
                            {
                                **row,
                                "path": f"batch{batch_size}_{side}",
                                "reference_top1": ref["reference_top1"],
                                "reference_logp": ref["reference_logp"],
                                "candidate_top1": candidate[0],
                                "candidate_logp": candidate[1],
                                "comparison": comparison,
                            }
                        )
    finally:
        meter.handle.remove()
    write_json(out / "fixed_reference_rollouts.json", generation_rows)
    write_json(out / "forward_meter.json", meter.report())
    summary = aggregate_parity(records, config)
    write_json(
        out / "likelihood_parity.json",
        {
            "status": "PASS" if all(x["passed"] for x in summary.values()) else "FAIL",
            "protocol": {
                "base_scenes": 12,
                "prompts": 24,
                "sampled_per_prompt": 4,
                "greedy_per_prompt": 1,
                "total_reference_outputs": 120,
                "padding_sides": ["left", "right"],
                "batch_sizes": [1, 2, 4],
            },
            "paths": summary,
        },
    )
    return meter.report(), summary


def _run_smoke(config, scenes, out):
    from .smoke_runtime import run_smoke

    locked = json.loads((Path(__file__).parents[1] / "configs" / "locked.json").read_text())
    locked["data_root"] = config["data_root"]
    locked["training"] = {**locked["training"], "smoke_updates": 4}
    return run_smoke(locked, scenes[:18], out, "qwen35_9b")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke-out", type=Path, required=True)
    args = p.parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("R1 requires CUDA")
    import yaml

    config = yaml.safe_load(args.config.read_text())
    config["data_root"] = str(args.data_root.resolve())
    config["max_new_tokens"] = int(config["generation_proposed_N"]["max_new_tokens"])
    for key in (
        "parity_alarm_mean_abs_token_logp",
        "parity_alarm_p99_abs_token_logp",
        "parity_alarm_min_top1_agreement",
    ):
        config[key] = config["R1"][key]
    args.out.mkdir(parents=True, exist_ok=True)
    args.smoke_out.mkdir(parents=True, exist_ok=True)
    scenes = load_split(
        args.data_root, "calibration", purpose="R1 fixed reference panel and isolated smoke"
    )
    panel = reference_panel(scenes)
    adapter = load_adapter("qwen35_9b", _model_spec(config), image_token_limit=768)
    started = time.perf_counter()
    meter, parity = _run_parity(adapter, panel, config, args.out)
    smoke = _run_smoke(config, scenes, args.smoke_out)
    write_json(
        args.out / "environment_lock.json",
        {
            "status": "PASS",
            "execution_kind": "REAL_CUDA_INFERENCE",
            "python": platform.python_version(),
            "model": adapter.audit,
            "source_commit": __import__("src.core", fromlist=["source_commit"]).source_commit(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    write_json(
        args.out / "runtime_profile.json",
        {
            "status": "PASS",
            "execution_kind": "REAL_CUDA_INFERENCE",
            **meter,
            "smoke_execution_kind": smoke.get("execution_kind"),
            "smoke_elapsed_seconds": smoke.get("generation_seconds"),
        },
    )
    write_json(
        args.out / "lora_module_manifest.json",
        {
            "status": "PASS"
            if smoke.get("checks", {}).get("nonempty_gradients_and_measured_step")
            else "FAIL",
            "execution_kind": smoke.get("execution_kind"),
            "modules": smoke.get("lora_modules", []),
            "trainable_parameters": smoke.get("trainable_parameters"),
        },
    )
    write_json(
        args.out / "training_smoke.json",
        {
            "status": "PASS" if smoke.get("passed") else "FAIL",
            "execution_kind": smoke.get("execution_kind"),
            "steps": smoke.get("training_steps", []),
            "raw_sample_count": smoke.get("raw_sample_count"),
            "checks": smoke.get("checks"),
        },
    )
    write_json(
        args.out / "budget_projection.json",
        {
            "status": "MEASURED",
            "reference_outputs": 120,
            "smoke_rollouts": 128,
            "generation_tokens_per_second": smoke.get("generation_tokens_per_second"),
            "mean_update_seconds": smoke.get("budget_estimation", {}).get("mean_update_seconds"),
            "projection_is_not_a_runtime_guarantee": True,
        },
    )
    final = (
        "PASS"
        if parity and all(x["passed"] for x in parity.values()) and smoke.get("passed")
        else "FAIL"
    )
    details = {
        "R1_gate_passed": final == "PASS",
        "reference_outputs": 120,
        "smoke_steps": len(smoke.get("training_steps", [])),
        "parity_paths": len(parity),
        "execution_kind": "REAL_CUDA_INFERENCE",
        "smoke_execution_kind": smoke.get("execution_kind"),
        "later_phases": "NOT_RUN",
    }
    status = phase_artifacts(
        args.out,
        "R1",
        final,
        details,
        [
            args.out / n
            for n in [
                "environment_lock.json",
                "runtime_profile.json",
                "likelihood_parity.json",
                "lora_module_manifest.json",
                "training_smoke.json",
                "budget_projection.json",
                "fixed_reference_rollouts.json",
                "forward_meter.json",
            ]
        ],
    )
    status["execution_kind"] = "REAL_CUDA_INFERENCE"
    write_json(args.out / "status.json", status)
    (args.out / "report_zh.md").write_text((args.out / "report.md").read_text(), encoding="utf-8")
    return 0 if final == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
