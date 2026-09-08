"""P3 fixed frozen banks; no optimizer, checkpoint loading, or adaptive sampling."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

from .core import (
    PROJECT_ROOT,
    RunStore,
    canonical_hash,
    file_hash,
    frozen_writer,
    phase_artifacts,
    source_commit,
    validate_config,
    write_json,
)
from .optimizer_fork import parameter_hash
from .prompts import build_prompt
from .smoke_runtime import INTERFACES, Telemetry, _seed_everything, _verify_scene_images
from .verifiers import annotate

MODEL_ORDER = ("qwen25vl_3b", "qwen35_9b", "qwen25vl_7b")
FROZEN_PROTOCOL = {
    "model_order": list(MODEL_ORDER),
    "sampled_rollouts_per_prompt": 16,
    "greedy_rollouts_per_prompt": 1,
    "N_max_new_tokens": 64,
    "L_max_new_tokens": 48,
    "N_split": "dev",
    "L_evaluation_splits": ["dev", "test", "positive_control"],
}
P1_CHECKS = {
    "sampling_likelihood",
    "cache",
    "image",
    "frozen_base",
    "nonempty_gradients_and_measured_step",
    "fork_resume",
    "real_optimizer_steps",
}
DEPENDENCIES = (
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "accelerate",
    "numpy",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "pillow",
)


def validate_p1_evidence(root, config, model_key):
    if root is None:
        raise ValueError("P1 evidence directory is required for the selected model")
    root = Path(root)
    try:
        status = json.loads((root / "status.json").read_text())
        audit = json.loads((root / "model_audit.json").read_text())
        lock = json.loads((root / "validated_runtime_lock.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("P1 evidence is missing or malformed") from exc
    if (
        status.get("phase") != "P1"
        or status.get("status") != "PASS"
        or audit.get("passed") is not True
        or audit.get("execution_kind") != "REAL_CUDA_MODEL"
        or lock.get("runtime_status") != "P1_PASSED"
        or lock.get("execution_kind") != "REAL_CUDA_MODEL"
    ):
        raise ValueError("P1 must be passed real CUDA evidence, never a fixture")
    spec = config["models"][model_key]
    if (
        audit.get("model_id") != spec["id"]
        or audit.get("model_revision") != spec["revision"]
        or lock.get("model_revision") != spec["revision"]
        or lock.get("resolved_config", {}).get("_model_key") != model_key
    ):
        raise ValueError("P1 model/revision does not match selected frozen model")
    if not all(audit.get("checks", {}).get(k) is True for k in P1_CHECKS):
        raise ValueError("P1 required compatibility check is missing or failed")
    manifest = json.loads((root / "manifest.json").read_text())
    entry = next((r for r in manifest["files"] if r["path"] == "model_audit.json"), None)
    if entry is None or entry["sha256"] != file_hash(root / "model_audit.json"):
        raise ValueError("P1 audit manifest hash mismatch")
    candidate = {k: v for k, v in lock.items() if k not in {"runtime_status", "execution_kind"}}
    recorded = {k: v for k, v in audit["runtime_lock"].items() if k != "runtime_status"}
    if candidate != recorded or file_hash(root / "pip-freeze.txt") != lock["pip_freeze_sha256"]:
        raise ValueError("P1 validated runtime/dependency lock mismatch")
    versions = {}
    for line in (root / "pip-freeze.txt").read_text().splitlines():
        name, separator, version = line.partition("==")
        if separator:
            versions[name.lower().replace("_", "-")] = version
    for name in DEPENDENCIES:
        if versions.get(name) != importlib.metadata.version(name):
            raise ValueError(f"P1 dependency drift: {name}; rerun compatibility before frozen")
    return {
        "audit_sha256": file_hash(root / "model_audit.json"),
        "lock_sha256": file_hash(root / "validated_runtime_lock.json"),
        "model_audit": audit,
        "versions": {k: versions[k] for k in DEPENDENCIES},
    }


def validate_human_review(review_path, contact_manifest_path, data_root):
    if review_path is None:
        raise PermissionError("P0 independent human review of 36 charts is still required")
    manifest = json.loads(Path(contact_manifest_path).read_text())
    review = json.loads(Path(review_path).read_text())
    selected = [r for sheet in manifest["sheets"] for r in sheet["scenes"]]
    expected = {r["base_scene_id"] for r in selected}
    if (
        len(selected) != 36
        or len(expected) != 36
        or review.get("status") != "PASS"
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not isinstance(review.get("reviewed_at"), str)
        or not review["reviewed_at"].strip()
        or review.get("contact_manifest_sha256") != file_hash(contact_manifest_path)
        or len(review.get("checked_scene_ids", [])) != 36
        or set(review.get("checked_scene_ids", [])) != expected
    ):
        raise PermissionError("P0 human review must cover the exact 36-chart contact manifest")
    if file_hash(Path(data_root) / "calibration.jsonl") != manifest["source_calibration_sha256"]:
        raise ValueError("Human-reviewed calibration data hash changed")
    _verify_scene_images(selected, data_root)
    return {
        "status": "PASS",
        "review_sha256": file_hash(review_path),
        "contact_manifest_sha256": file_hash(contact_manifest_path),
        "image_count": 36,
    }


def load_frozen_dev(data_root):
    from .core import load_split

    root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    if file_hash(root / "dev.jsonl") != manifest["files"]["dev.jsonl"]["sha256"]:
        raise ValueError("dev data manifest hash mismatch")
    scenes = load_split(root, "dev", purpose="P3 fixed frozen dev; no confirm access")
    if (
        len(scenes) != 144
        or len({s["base_scene_id"] for s in scenes}) != 144
        or any(s["split"] != "dev" for s in scenes)
    ):
        raise ValueError("P3 N requires all 144 distinct dev scenes")
    cells = Counter((s["constraint_family"], s["chart_type"], s["operation"]) for s in scenes)
    if len(cells) != 18 or set(cells.values()) != {8}:
        raise ValueError("dev family/chart/operation panel changed")
    _verify_scene_images(scenes, root)
    return scenes


def _source_identity(track):
    sources = {
        str(p.relative_to(PROJECT_ROOT)): file_hash(p)
        for p in sorted((PROJECT_ROOT / "src").rglob("*.py"))
    }
    if track == "L":
        from .legacy_frozen import legacy_qualification

        qualification = legacy_qualification()
        if not qualification["passed"]:
            raise ValueError("Legacy parser/executor qualification failed")
        sources.update({"legacy/" + k: v for k, v in qualification["source_hashes"].items()})
    return {
        "source_commit": source_commit(),
        "source_files": sources,
        "python": platform.python_version(),
        "versions": dict(
            sorted(
                (d.metadata["Name"].lower().replace("_", "-"), d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            )
        ),
    }


def _requests(scenes, track, sample_seed):
    for scene in sorted(scenes, key=lambda s: s.get("scene_id", s["base_scene_id"])):
        interfaces = INTERFACES if track == "N" else (scene["interface"],)
        for interface in interfaces:
            pid = canonical_hash(
                {
                    "scene": scene.get("scene_id", scene["base_scene_id"]),
                    "interface": interface,
                    "track": track,
                }
            )
            for mode, count in (("sample", 16), ("greedy", 1)):
                for index in range(count):
                    identity = {
                        "phase": "P3",
                        "track": track,
                        "prompt_id": pid,
                        "decode_mode": mode,
                        "rollout_index": index,
                        "sample_seed_root": sample_seed,
                    }
                    seed = int(canonical_hash(identity)[:8], 16) % (2**31)
                    yield (
                        scene,
                        interface,
                        {**identity, "sample_seed": seed, "sample_key": canonical_hash(identity)},
                    )


def run_frozen(
    config,
    scenes,
    out,
    model_key,
    resume=False,
    *,
    track="N",
    p1_run_dir=None,
    human_review=None,
    contact_manifest=None,
    _adapter_factory=None,
    _telemetry=None,
):
    import torch

    from .frozen_metrics import build_frozen_metrics
    from .frozen_report import write_frozen_report
    from .model_adapters import load_adapter

    # Validate required evidence before allocating model memory or contacting HF.
    gate = (
        {"fixture_only": True}
        if _adapter_factory
        else validate_p1_evidence(p1_run_dir, config, model_key)
    )
    if track not in ("N", "L"):
        raise ValueError("unsupported frozen track")
    validate_config(config, "frozen", model_key)
    if model_key not in MODEL_ORDER or config.get("frozen", FROZEN_PROTOCOL) != FROZEN_PROTOCOL:
        raise ValueError("P3 requires the fixed three-model, 16-sample plus greedy protocol")
    if track == "N":
        if not scenes or any(s["split"] != "dev" for s in scenes):
            raise ValueError("P3 N frozen uses dev only; confirm is sealed")
        _verify_scene_images(scenes, config["data_root"])
        if not _adapter_factory:
            if canonical_hash(scenes) != canonical_hash(load_frozen_dev(config["data_root"])):
                raise ValueError("P3 requires the exact full fixed dev panel")
            gate["human_review"] = validate_human_review(
                human_review, contact_manifest, config["data_root"]
            )
    else:
        from .legacy_frozen import EVALUATION_SPLITS, legacy_qualification

        if not scenes or any(s["split"] not in EVALUATION_SPLITS for s in scenes):
            raise ValueError("L frozen requires the original complete evaluation panel")
        gate["legacy_qualification"] = legacy_qualification()
        if not _adapter_factory:
            from .legacy_frozen import load_legacy_scenes

            selected = load_legacy_scenes(config["legacy_data"], config["legacy_manifest"])
            if canonical_hash(scenes) != canonical_hash(selected):
                raise ValueError("P3 requires the complete original legacy evaluation panel")
    if not _adapter_factory and not torch.cuda.is_available():
        raise RuntimeError("P3 requires an allocated CUDA GPU; no model download attempted")
    import pyarrow.parquet  # noqa: F401 - export backend preflight before model load

    # Frozen means fresh pretrained base. Do not load P1's updated LoRA checkpoint.
    length = 64 if track == "N" else 48
    if track == "N" and config["generation"]["max_new_tokens"] != 64:
        raise ValueError("P3 N main bank has a fixed 64-token cap")
    spec = config["models"][model_key]
    source = _source_identity(track)
    identity = {
        "model_hash": canonical_hash(spec),
        "model_key": model_key,
        "model_id": spec["id"],
        "model_revision": spec["revision"],
        "data_hash": canonical_hash(scenes),
        "config_hash": canonical_hash(config),
        "phase": "P3",
        "track": track,
        "gate_hash": canonical_hash(gate),
        "source_hash": canonical_hash(source),
        "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE" if _adapter_factory else "REAL_CUDA_MODEL",
    }
    out = Path(out)
    with frozen_writer(out):
        # Strict identity rejection happens before model load; no evidence is rewritten.
        store = RunStore(out, identity, resume=resume)
        requests = list(_requests(scenes, track, config.get("sample_seed", 104729)))
        wanted = {meta["sample_key"]: meta for _, _, meta in requests}
        for key, row in store.records.items():
            if key not in wanted or any(row.get(k) != v for k, v in wanted[key].items()):
                raise ValueError("Unexpected or changed frozen sample ledger identity")
            if row.get("record_hash") != canonical_hash(
                {k: v for k, v in row.items() if k != "record_hash"}
            ):
                raise ValueError("Frozen sample ledger content hash mismatch")
        try:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            _seed_everything(config["training"]["seed"])
            adapter = (_adapter_factory or load_adapter)(model_key, spec, image_token_limit=768)
            if adapter.revision != spec["revision"] or adapter.model_id != spec["id"]:
                raise ValueError("Loaded model revision mismatch")
            if not _adapter_factory:
                for key in (
                    "processor_hash",
                    "tokenizer_hash",
                    "chat_template_hash",
                    "probability_execution",
                ):
                    if not adapter.audit.get(key) or adapter.audit[key] != gate["model_audit"].get(
                        key
                    ):
                        raise ValueError(f"P1 processor/template/numerical path drift: {key}")
            adapter.model.requires_grad_(False)
            adapter.model.eval()
            before = parameter_hash(adapter.model, trainable=False)
            runtime = {
                "identity": identity,
                "model_spec": spec,
                "generation_protocol": {**config["generation"], "max_new_tokens": length},
                **source,
                "model_audit": adapter.audit,
                "frozen_parameter_hash": before,
                "max_new_tokens": length,
                "adapter_state": "fresh base; adapter disabled; no checkpoint loaded",
                "P1_evidence": gate,
                "optimizer_updates": 0,
            }
            lock_path = out / "runtime_lock.json"
            if lock_path.exists() and json.loads(lock_path.read_text()) != runtime:
                # Load time telemetry varies; compare scientific model/processor identity only.
                prior = json.loads(lock_path.read_text())
                if any(
                    prior.get(k) != runtime[k]
                    for k in ("identity", "frozen_parameter_hash", "max_new_tokens")
                ):
                    raise ValueError("Frozen runtime lock mismatch")
            elif not lock_path.exists():
                write_json(lock_path, runtime)
            write_json(
                out / "data_manifest.json",
                {
                    "track": track,
                    "data_hash": identity["data_hash"],
                    "scene_count": len(scenes),
                    "prompt_count": len(requests) // 17,
                },
            )
            write_json(out / "gate_review.json", gate)
            telemetry = _telemetry or Telemetry()
            prepared, current = None, None
            started = time.perf_counter()
            disabled = (
                adapter.model.disable_adapter()
                if hasattr(adapter.model, "disable_adapter")
                else contextlib.nullcontext()
            )
            with disabled:
                for scene, interface, meta in requests:
                    if meta["sample_key"] in store.keys:
                        continue
                    if current != meta["prompt_id"]:
                        if track == "L":
                            from .legacy_frozen import build_legacy_prompt

                            prompt = build_legacy_prompt(scene)
                        else:
                            prompt = build_prompt(scene, interface)
                        prepared = adapter.prepare(prompt, config["data_root"])
                        current = meta["prompt_id"]
                    telemetry.reset()
                    forwards = adapter.forward_calls
                    generation = adapter.generate(
                        prepared,
                        seed=meta["sample_seed"],
                        max_new_tokens=length,
                        do_sample=meta["decode_mode"] == "sample",
                    )
                    tokens = generation["token_ids"]
                    behavior = generation["behavior_token_logprobs"]
                    scored = adapter.logprobs(prepared, tokens).detach().cpu().tolist()
                    if (
                        not tokens
                        or len(tokens) != len(behavior)
                        or len(scored) != len(tokens)
                        or generation["completion_length"] != len(tokens)
                        or len(tokens) > length
                        or any(not math.isfinite(x) for x in behavior + scored)
                    ):
                        raise RuntimeError("Malformed generation/likelihood token scores")
                    error = sum(abs(a - b) for a, b in zip(behavior, scored, strict=True)) / len(
                        tokens
                    )
                    if track == "L":
                        from .legacy_frozen import annotate_legacy

                        annotation = annotate_legacy(generation["raw_completion"], scene)
                    else:
                        annotation = annotate(
                            generation["raw_completion"],
                            scene["truth_world"],
                            scene["operation"],
                            scene["cue"],
                        )
                        annotation["copy_observation"] = (
                            annotation["parsed_world"] == scene["observed_world"]
                        )
                    row = {
                        **meta,
                        "base_scene_id": scene["base_scene_id"],
                        "constraint_family": scene["constraint_family"],
                        "interface": interface,
                        "split": scene["split"],
                        "model_key": model_key,
                        "model_id": adapter.model_id,
                        "model_revision": adapter.revision,
                        "execution_kind": identity["execution_kind"],
                        "train_seed": None,
                        "optimizer_step": 0,
                        **prepared["audit"],
                        **generation,
                        **annotation,
                        **telemetry.snapshot(),
                        "policy_token_logprobs": scored,
                        "policy_base_identical": True,
                        "behavior_teacher_forcing_mean_error": error,
                        "on_policy_likelihood_passed": error <= 0.02,
                        "forward_calls": adapter.forward_calls - forwards,
                        "probability_note": (
                            "base softmax token scores; greedy choices are not probability samples"
                        ),
                    }
                    row["record_hash"] = canonical_hash(row)
                    store.append(row)
                    write_json(
                        out / "progress.json",
                        {
                            "completed": len(store.keys),
                            "total": len(wanted),
                            "last_prompt_id": current,
                            "model_key": model_key,
                            "track": track,
                            "state": "RUNNING",
                        },
                    )
                    if error > 0.02:
                        raise RuntimeError(
                            "P3 generation/likelihood consistency alarm; sample preserved"
                        )
                # PEFT restores trainability flags when leaving disable_adapter(), even
                # without a weight change. Audit the actual frozen inference context.
                after = parameter_hash(adapter.model, trainable=False)
                if before != after or any(p.requires_grad for p in adapter.model.parameters()):
                    raise RuntimeError("Frozen model parameters changed")
            rows = list(store.records.values())
            if not all(r["on_policy_likelihood_passed"] for r in rows):
                raise RuntimeError("Preserved P3 probability alarm cannot be bypassed by resume")
            metrics = build_frozen_metrics(rows, track=track)
            audit = {
                **adapter.audit,
                "phase": "P3",
                "model_key": model_key,
                "track": track,
                "passed": True,
                "execution_kind": identity["execution_kind"],
                "raw_sample_count": len(rows),
                "sampled_rollout_count": len(scenes) * (2 if track == "N" else 1) * 16,
                "greedy_rollout_count": len(requests) // 17,
                "optimizer_updates": 0,
                "backward_calls": 0,
                "frozen_parameter_hash_before": before,
                "frozen_parameter_hash_after": after,
                "forward_calls_total": sum(r["forward_calls"] for r in rows),
                "forward_calls_this_invocation": adapter.forward_calls,
                "elapsed_seconds_this_invocation": time.perf_counter() - started,
                "generated_tokens": sum(r["completion_length"] for r in rows),
                "generation_seconds": sum(r["elapsed_seconds"] for r in rows),
                "max_probability_error": max(
                    r["behavior_teacher_forcing_mean_error"] for r in rows
                ),
                "matrix_status": "SINGLE_MODEL_TRACK_ONLY",
                "legacy_exact_reproduction": False,
                "later_phases": "NOT_RUN; no training-safety inference from frozen observations",
            }
            write_json(out / "model_audit.json", audit)
            files = write_frozen_report(out, rows, metrics, audit)
            files = [
                *files,
                *(
                    out / n
                    for n in (
                        "samples.jsonl",
                        "model_audit.json",
                        "identity.json",
                        "runtime_lock.json",
                        "data_manifest.json",
                        "gate_review.json",
                    )
                ),
            ]
            phase_artifacts(out, "P3", "PASS", audit, files)
            write_json(
                out / "progress.json",
                {"completed": len(rows), "total": len(wanted), "state": "PASS"},
            )
            return audit
        except Exception as exc:
            phase_artifacts(
                out,
                "P3",
                "FAILED",
                {
                    "phase": "P3",
                    "model_key": model_key,
                    "track": track,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
            )
            raise
