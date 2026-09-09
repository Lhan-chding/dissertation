"""Read-only R2 postflight: reconstruct inputs, reclassify raw tokens, recompute reports."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

from .core import canonical_hash, file_hash, write_json
from .r2_inputs import (
    build_requests,
    prepare_r2,
    split_thinking_completion,
    variant_prompt,
    write_panel_artifacts,
)
from .r2_runtime import (
    _load_panel,
    _write_analysis,
    annotate_diagnostic,
    generation_checks,
    request_ledger,
)


def validate_processor_certificate(runtime, gate):
    actual, certified = runtime["model_audit"], gate["certificate"]["model_audit"]
    fields = (
        "processor_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "processor_max_pixels",
        "processor_patch_size",
        "processor_merge_size",
        "requested_image_token_limit",
        "eos_token_ids",
    )
    if any(
        key not in actual
        or key not in certified
        or canonical_hash(actual[key]) != canonical_hash(certified[key])
        for key in fields
    ):
        raise ValueError("R2 does not retain the original R1 processor/EOS binding")


def load_processor_adapter(runtime):
    """Use the certified preparation method with cached processor/config and no weights."""
    from transformers import AutoConfig, AutoProcessor

    from .model_adapters.base import _hash_json
    from .model_adapters.qwen35 import Qwen35Adapter

    audit = runtime["model_audit"]
    model_id, revision = audit["model_id"], audit["model_revision"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or runtime["config"]["model"]["id"] != model_id
        or runtime["config"]["model"]["revision"] != revision
    ):
        raise ValueError("R2 processor reconstruction requires the original pinned revision")
    options = {"revision": revision, "trust_remote_code": False, "local_files_only": True}
    processor = AutoProcessor.from_pretrained(model_id, **options)
    config = AutoConfig.from_pretrained(model_id, **options)
    ip = processor.image_processor
    pixels = audit["processor_max_pixels"]
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = pixels
    if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
        ip.size = {**ip.size, "longest_edge": pixels}
    elif dataclasses.is_dataclass(getattr(ip, "size", None)):
        ip.size = dataclasses.replace(ip.size, longest_edge=pixels)
    measured = {
        "processor_hash": _hash_json(processor.to_dict()),
        "tokenizer_hash": _hash_json(processor.tokenizer.get_vocab()),
        "chat_template_hash": _hash_json(processor.chat_template),
    }
    if any(audit.get(key) != value for key, value in measured.items()):
        raise ValueError("Cached processor/tokenizer/template differs from measured R2")
    adapter = Qwen35Adapter(SimpleNamespace(config=config), processor, model_id, revision, "cpu")
    adapter.audit = dict(audit)
    return adapter


def validate_raw_rows(rows, panel, runtime, adapter):
    """Recompute semantic and input fields; the caller first verifies the complete ledger."""
    config, model_audit = runtime["config"], runtime["model_audit"]
    scenes = {scene["base_scene_id"]: scene for scene in panel}
    prepared_audits = {}
    thinking_count = image_count = 0
    for row in rows:
        scene = scenes[row["base_scene_id"]]
        if row["prompt_id"] not in prepared_audits:
            prepared_audits[row["prompt_id"]] = prepare_r2(
                adapter,
                variant_prompt(scene, row["condition"]),
                config["data_root"],
                enable_thinking=row["enable_thinking"],
            )["audit"]
        audit = prepared_audits[row["prompt_id"]]
        faults = generation_checks(
            row, adapter.processor.tokenizer, model_audit["eos_token_ids"], row["max_new_tokens"]
        )
        if faults or row.get("execution_checks") != {"passed": True, "faults": []}:
            raise ValueError(
                "R2 raw token/EOS/behavior execution check failed: " + "; ".join(faults)
            )
        segments = None
        if row["enable_thinking"]:
            thinking_count += 1
            segments = split_thinking_completion(
                row["token_ids"], adapter.processor.tokenizer, model_audit["eos_token_ids"]
            )
        image = row["condition"] in {"IMAGE_CUE", "IMAGE_ONLY"}
        image_count += int(image)
        expected = {
            **audit,
            **annotate_diagnostic(row["raw_completion"], scene, thinking=segments),
            **{
                key: scene[key]
                for key in (
                    "truth_world",
                    "observed_world",
                    "changed_index",
                    "cue",
                    "chart_type",
                    "operation",
                )
            },
            "run_id": canonical_hash(runtime["identity"]),
            "protocol_version": config["protocol_version"],
            "model_id": model_audit["model_id"],
            "model_revision": model_audit["model_revision"],
            "adapter_hash": runtime["initial_adapter_hash"],
            "optimizer_state_hash": None,
            "checkpoint_step": 0,
            "train_seed": 17,
            "split": "calibration",
            "family": scene["constraint_family"],
            "constraint_family": scene["constraint_family"],
            "interface": "IMAGE_CUE_FRESH" if image else "SYMBOLIC_FRESH",
            "solution_count": 1,
            "image_hash": scene["image_hash"] if image else None,
            "input_ids_hash": audit.get("tokenized_prompt_hash"),
            "actual_image_tokens": audit["image_token_count"],
            "raw_text": row["raw_completion"],
            "raw_token_ids": row["token_ids"],
            "n_generated_tokens": len(row["token_ids"]),
            "per_token_logprob_behavior": row["behavior_token_logprobs"],
            "logprob_sequence": sum(row["behavior_token_logprobs"]),
            "thinking_segments": segments,
            "generation_config_hash": canonical_hash(
                {
                    **config["generation_proposed_N"],
                    "max_new_tokens": row["max_new_tokens"],
                    "enable_thinking": row["enable_thinking"],
                    "do_sample": row["decode_mode"] == "sample",
                }
            ),
        }
        for key, value in expected.items():
            if key not in row or canonical_hash(row[key]) != canonical_hash(value):
                raise ValueError(f"R2 raw field differs from reconstructed evidence: {key}")
        if bool(audit["image_token_count"]) != image or (
            image and row.get("vision_forward_calls", 0) < 1
        ):
            raise ValueError("R2 original image input/vision-forward evidence is missing")
    return {
        "rows": len(rows),
        "prepared_prompts": len(prepared_audits),
        "thinking_rows": thinking_count,
        "image_rows": image_count,
    }


def audit_r2_results(r2_dir, gate, out):
    from .next_stage_runtime import validate_r2_gate, validate_runtime_environment

    root, out = Path(r2_dir).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError("R2 postflight requires a new directory outside the immutable stage")
    binding = validate_r2_gate(root, gate)
    environment = validate_runtime_environment(gate)
    if binding["environment"] != environment:
        raise ValueError("R2 postflight must use the measured environment for exact recomputation")
    runtime = json.loads((root / "runtime_lock.json").read_text())
    validate_processor_certificate(runtime, gate)
    panel, data = _load_panel(gate["config"]["data_root"])
    if data != json.loads((root / "data_binding.json").read_text()):
        raise ValueError("R2 source panel or image/solver binding changed")
    requests = request_ledger(build_requests(panel), runtime["identity"])
    stored_requests = json.loads((root / "request_manifest.json").read_text())["requests"]
    if canonical_hash(requests) != canonical_hash(stored_requests):
        raise ValueError("R2 regenerated fixed request allocation differs from the run")
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    raw_audit = validate_raw_rows(rows, panel, runtime, load_processor_adapter(runtime))
    out.mkdir(parents=True, exist_ok=False)
    derived = out / "recomputed"
    derived.mkdir()
    write_panel_artifacts(panel, derived)
    _write_analysis(derived, rows)
    files = (
        "panel_manifest.json",
        "prompt_diffs.md",
        "condition_metrics.json",
        "condition_metrics.csv",
        "paired_condition_effects.csv",
        "invalid_taxonomy.csv",
    )
    checks = {
        name: {
            "original_sha256": file_hash(root / name),
            "recomputed_sha256": file_hash(derived / name),
        }
        for name in files
    }
    passed = all(v["original_sha256"] == v["recomputed_sha256"] for v in checks.values())
    # Detect any source change during processor reconstruction/statistical recomputation.
    for name, digest in binding["r2_files"].items():
        if file_hash(root / name) != digest:
            raise ValueError("R2 source artifact changed during postflight")
    result = {
        "status": "PASS" if passed else "FAIL",
        "phase": "R2_POSTFLIGHT",
        "execution_kind": "CPU_AUDIT",
        "model_weights_loaded": False,
        "new_model_generations": 0,
        "r2_binding": binding,
        "environment": environment,
        "raw_audit": raw_audit,
        "report_hash_checks": checks,
        "scope": (
            "Original frozen CUDA evidence, independently rebuilt CPU inputs "
            "and raw/statistical analysis"
        ),
    }
    write_json(out / "audit.json", result)
    if not passed:
        raise ValueError("R2 derived report differs from original raw-result recomputation")
    return result


def main(argv=None):
    from .next_stage_runtime import validate_prerequisites

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r2-dir", "r0-dir", "r1-run", "supplement-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    gate = validate_prerequisites(args.r0_dir, args.r1_run, args.supplement_dir)
    result = audit_r2_results(args.r2_dir, gate, args.out)
    print(
        json.dumps(
            {"status": result["status"], "raw_audit": result["raw_audit"], "out": str(args.out)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
