#!/usr/bin/env python3
"""Run the frozen 36-call, text-only Phase-C prompt qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_LOCK_RELATIVE = (
    "configs/recoverability/server_package_lock_phase_c_prompt_qualification_v1.yaml"
)
_INHERITED_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml"
_ADDITIONS = frozenset(
    {
        "configs/recoverability/phase_c_prompt_qualification_v1.yaml",
        _INHERITED_LOCK_RELATIVE,
        "experiments/recoverability_v1/18_phase_c_prompt_qualification_preflight.py",
        "experiments/recoverability_v1/19_run_phase_c_prompt_qualification.py",
        "src/compbias/recoverability/phase_c_prompt_qualification.py",
        "src/compbias/recoverability/phase_c_prompt_qualification_execution.py",
    }
)
_EXPECTED_MODEL_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_rows(path: Path) -> tuple[tuple[str, str], ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [
        line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")
    ]
    if len(paths) != len(digests) or not paths or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: prompt qualification package lock is malformed")
    return tuple(zip(paths, digests, strict=True))


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in sys.argv:
        return
    try:
        supplied = Path(sys.argv[sys.argv.index("--server-package-lock") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical prompt qualification lock is required") from error
    root = Path(__file__).resolve().parents[2]
    canonical = root / _LOCK_RELATIVE
    inherited = root / _INHERITED_LOCK_RELATIVE
    if supplied != canonical or canonical.is_symlink() or inherited.is_symlink():
        raise SystemExit("BLOCKED: prompt qualification lock path is not canonical")
    inherited_rows = _lock_rows(inherited)
    rows = _lock_rows(canonical)
    if frozenset(relative for relative, _digest in rows) != _ADDITIONS:
        raise SystemExit("BLOCKED: prompt qualification package closure is incomplete")
    for relative, expected in inherited_rows + rows:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: prompt qualification package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.config import load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import load_local_qwen  # noqa: E402
from compbias.io.strict_json import load_strict_json_mapping  # noqa: E402
from compbias.recoverability.phase_c_prompt_qualification import (  # noqa: E402
    PhaseCPromptQualificationRecord,
    build_phase_c_prompt_qualification_calls,
    evaluate_phase_c_prompt_qualification_call,
    load_phase_c_prompt_qualification_config,
    record_payload,
    summarize_phase_c_prompt_qualification,
)
from compbias.recoverability.phase_c_prompt_qualification_execution import (  # noqa: E402
    verify_phase_c_prompt_qualification_package_lock,
)
from compbias.recoverability.phase_c_screen_result import (  # noqa: E402
    load_phase_c_screen_frozen_result,
    verify_phase_c_screen_artifacts,
)
from compbias.recoverability.preflight import (  # noqa: E402
    load_runtime_spec,
    run_metadata_preflight,
)


def _metadata_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def _pip_check() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env=_metadata_environment(),
    )
    return completed.returncode, completed.stdout + completed.stderr


def _pip_inventory() -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
        env=_metadata_environment(),
    )
    return {row["name"]: row["version"] for row in json.loads(completed.stdout)}


def _exclusive_marker(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to rerun Phase C prompt qualification")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("prompt qualification output parent must be a regular directory")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _validate_preflight(path: Path, *, lock: Path, package_files: list[str]) -> None:
    payload = load_strict_json_mapping(
        path, label="Phase C prompt qualification preflight", max_bytes=512 * 1024
    )
    if payload.get("artifact_type") != (
        "recoverability_phase_c_prompt_qualification_metadata_preflight"
    ):
        raise ValueError("prompt qualification preflight artifact type is invalid")
    if payload.get("ready") is not True or payload.get("model_loaded") is not False:
        raise ValueError("prompt qualification preflight is not ready")
    if payload.get("model_call_cap") != 36 or payload.get("scale_authorized") is not False:
        raise ValueError("prompt qualification preflight call cap differs")
    if payload.get("training_authorized") is not False:
        raise ValueError("prompt qualification preflight must not authorize training")
    if payload.get("server_package_lock_sha256") != _sha256(lock):
        raise ValueError("prompt qualification preflight lock digest differs")
    if payload.get("server_package_files") != package_files:
        raise ValueError("prompt qualification preflight package closure differs")


def decode_text_qwen_seeded_once(
    model: object,
    processor: object,
    messages: tuple[dict[str, object], ...],
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Sample one reproducible text-only fork with no image input."""

    import torch

    text = processor.apply_chat_template(  # type: ignore[attr-defined]
        list(messages), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(  # type: ignore[operator]
        text=[text], padding=True, return_tensors="pt"
    ).to("cuda:0")
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            generated = model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
    trimmed = [
        output[len(source) :]
        for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--qualification-config", type=Path)
    parser.add_argument("--screen-result", type=Path)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--screen-preflight", type=Path)
    parser.add_argument("--screen-attempt-marker", type=Path)
    parser.add_argument("--screen-dataset-root", type=Path)
    parser.add_argument("--screen-output-root", type=Path)
    parser.add_argument("--screen-console-log", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: Phase C prompt qualification requires explicit --execute")
        return 2
    required = tuple(value for key, value in vars(args).items() if key != "execute")
    if any(value is None for value in required):
        print("BLOCKED: prompt qualification requires every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("prompt qualification forbids COMPBIAS path overrides")

    root = Path(__file__).resolve().parents[2]
    canonical = {
        "paths": root / "configs/paths.yaml",
        "runtime": root / "configs/recoverability/server_runtime_v1.yaml",
        "qualification_config": (
            root / "configs/recoverability/phase_c_prompt_qualification_v1.yaml"
        ),
        "screen_result": root / "configs/recoverability/phase_c_screen_v2_frozen_result.yaml",
        "server_package_lock": root / _LOCK_RELATIVE,
    }
    for argument, expected in canonical.items():
        supplied = getattr(args, argument)
        if supplied.resolve() != expected or supplied.is_symlink():
            raise ValueError(f"prompt qualification {argument} path is not canonical")
    package = verify_phase_c_prompt_qualification_package_lock(
        canonical["server_package_lock"], repository_root=root
    )
    package_files = [item.relative_path for item in package.files]
    current = run_metadata_preflight(
        load_runtime_spec(canonical["runtime"]),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    if not current.ready or current.large_gpu_started or current.model_loaded:
        raise RuntimeError("prompt qualification runtime preflight failed")
    _validate_preflight(
        args.preflight_report,
        lock=canonical["server_package_lock"],
        package_files=package_files,
    )
    paths = load_pilot_paths(canonical["paths"], environ={})
    evidence_root = Path("/cloud/cloud-ssd1/recoverability-v1-evidence")
    expected_paths = {
        "preflight_report": evidence_root / "phase-c-prompt-qualification-v1-preflight.json",
        "screen_preflight": evidence_root / "phase-c-screen-v2-preflight.json",
        "screen_attempt_marker": (
            paths.outputs / "recoverability_v1/cva_recoverability_causal_v2.screen.attempted.json"
        ),
        "screen_dataset_root": paths.data / "generated/cva_recoverability_causal_v2_screen",
        "screen_output_root": (
            paths.outputs / "recoverability_v1/cva_recoverability_causal_v2/phase_c_screen"
        ),
        "screen_console_log": evidence_root / "phase-c-screen-v2-console.log",
    }
    for label, expected in expected_paths.items():
        if getattr(args, label).resolve() != expected:
            raise ValueError(f"prompt qualification {label} path is not canonical")

    screen = load_phase_c_screen_frozen_result(canonical["screen_result"])
    eligible = verify_phase_c_screen_artifacts(
        screen,
        preflight=args.screen_preflight,
        attempt_marker=args.screen_attempt_marker,
        dataset_manifest=args.screen_dataset_root / "manifest.json",
        dataset_records=args.screen_dataset_root / "records.jsonl",
        screen_report=args.screen_output_root / "screen_report.json",
        screen_records=args.screen_output_root / "screen_records.jsonl",
        console_log=args.screen_console_log,
    )
    config = load_phase_c_prompt_qualification_config(canonical["qualification_config"])
    calls = build_phase_c_prompt_qualification_calls(eligible, config=config)
    if len(calls) != 36 or len({call.call_id for call in calls}) != 36:
        raise RuntimeError("prompt qualification plan must contain exactly 36 calls")
    if config.hypothesis_tested or config.scale_authorized or config.training_authorized:
        raise RuntimeError("prompt qualification must remain diagnostic and low-cost")

    output_parent = paths.outputs / "recoverability_v1"
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / config.output_subdirectory
    attempt = output_parent / "cva_recoverability_causal_v3.prompt-qualification-v1.attempted.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite prompt qualification evidence")
    model_hash = model_snapshot_sha256(paths.model_path)
    if model_hash != _EXPECTED_MODEL_SHA256 or model_hash != screen.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the frozen Phase C screen")
    _exclusive_marker(
        attempt,
        {
            "schema_version": 1,
            "status": "PHASE_C_PROMPT_QUALIFICATION_STARTED_DO_NOT_RERUN",
            "qualification_id": config.qualification_id,
            "selected_scenes": 9,
            "conditions": list(config.conditions),
            "forks_per_condition": 2,
            "model_call_cap": 36,
            "format_retries": 0,
            "hypothesis_tested": False,
            "scale_authorized": False,
            "server_package_lock_sha256": _sha256(canonical["server_package_lock"]),
            "model_snapshot_sha256": model_hash,
            "training_invoked": False,
        },
    )
    max_format_retries = 0
    if max_format_retries != config.format_retries:
        raise RuntimeError("prompt qualification requires max_format_retries = 0")
    model, processor = load_local_qwen(paths.model_path)
    records: list[PhaseCPromptQualificationRecord] = []
    for call in calls:
        raw = decode_text_qwen_seeded_once(
            model,
            processor,
            call.messages,
            seed=call.sampling_seed,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        records.append(evaluate_phase_c_prompt_qualification_call(call, raw))
    if len(records) != config.model_call_cap:
        raise RuntimeError("prompt qualification did not stop at exactly 36 calls")
    if model_snapshot_sha256(paths.model_path) != model_hash:
        raise RuntimeError("model snapshot changed during prompt qualification")
    frozen_records = tuple(records)
    report = {
        "schema_version": 1,
        "artifact_type": "recoverability_phase_c_prompt_qualification_report",
        "status": "COMPLETED_DIAGNOSTIC_PROMPT_QUALIFICATION",
        "qualification_id": config.qualification_id,
        "dataset_id": config.dataset_id,
        "source_eligible_scenes": len(eligible),
        "selected_scene_ids": sorted({record.scene_id for record in frozen_records}),
        "sampling": {
            "seed": config.seed,
            "forks_per_condition": config.forks_per_condition,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        },
        "screen_report_sha256": dict(screen.source_sha256)["screen_report"],
        "server_package_lock_sha256": _sha256(canonical["server_package_lock"]),
        "model_snapshot_sha256": model_hash,
        "analysis": summarize_phase_c_prompt_qualification(frozen_records, config=config),
        "hypothesis_tested": False,
        "scale_authorized": False,
        "training_authorized": False,
        "rl_authorized": False,
        "training_invoked": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phase-c-prompt-qualification-", dir=output.parent
    ) as temp:
        staging = Path(temp) / output.name
        staging.mkdir()
        records_path = staging / "prompt_qualification_records.jsonl"
        with records_path.open("x", encoding="utf-8") as stream:
            for record in frozen_records:
                payload = json.dumps(record_payload(record), sort_keys=True, allow_nan=False)
                stream.write(payload + "\n")
        report["records_sha256"] = _sha256(records_path)
        with (staging / "prompt_qualification_report.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        staging.rename(output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
