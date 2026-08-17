#!/usr/bin/env python3
"""Execute one frozen text-only world-recovery diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

_INHERITED_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml"
_EXPECTED_MODEL_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
_PROFILES = {
    "configs/recoverability/phase_c_world_recovery_v1.yaml": {
        "lock": "configs/recoverability/server_package_lock_phase_c_world_recovery_v1.yaml",
        "preflight": "phase-c-world-recovery-v1r1-preflight.json",
        "attempt": "cva_recoverability_causal_v3.world-recovery-v1r1.attempted.json",
    },
    "configs/recoverability/phase_c_world_recovery_100_v1.yaml": {
        "lock": "configs/recoverability/server_package_lock_phase_c_world_recovery_100_v1.yaml",
        "preflight": "phase-c-world-recovery-100-v1-preflight.json",
        "attempt": "cva_recoverability_causal_v3.world-recovery-100-v1.attempted.json",
    },
}
_COMMON_ADDITIONS = frozenset(
    {
        _INHERITED_LOCK_RELATIVE,
        "experiments/recoverability_v1/20_phase_c_world_recovery_preflight.py",
        "experiments/recoverability_v1/21_run_phase_c_world_recovery.py",
        "prompts/no_cue.user.template.txt",
        "prompts/valid_cue.user.template.txt",
        "prompts/world_recovery_v1_ablation_no_examples.system.txt",
        "prompts/world_recovery_v1_main.system.txt",
        "src/compbias/recoverability/phase_c_world_recovery.py",
        "src/compbias/recoverability/phase_c_world_recovery_execution.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_rows(path: Path) -> tuple[tuple[str, str], ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: world recovery package lock is malformed")
    return tuple(zip(paths, digests, strict=True))


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in sys.argv:
        return
    try:
        supplied_lock = Path(sys.argv[sys.argv.index("--server-package-lock") + 1]).resolve()
        supplied_config = Path(sys.argv[sys.argv.index("--qualification-config") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical world recovery profile is required") from error
    root = Path(__file__).resolve().parents[2]
    inherited = root / _INHERITED_LOCK_RELATIVE
    matched = next(
        (
            (config_relative, profile)
            for config_relative, profile in _PROFILES.items()
            if supplied_config == root / config_relative and supplied_lock == root / profile["lock"]
        ),
        None,
    )
    if matched is None or supplied_config.is_symlink() or supplied_lock.is_symlink():
        raise SystemExit("BLOCKED: world recovery profile paths are not canonical")
    config_relative, _profile = matched
    inherited_rows = _lock_rows(inherited)
    rows = _lock_rows(supplied_lock)
    expected = _COMMON_ADDITIONS | {config_relative}
    if frozenset(relative for relative, _digest in rows) != expected:
        raise SystemExit("BLOCKED: world recovery package closure is incomplete")
    for relative, expected in inherited_rows + rows:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: world recovery package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.config import load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import load_local_qwen  # noqa: E402
from compbias.io.strict_json import load_strict_json_mapping  # noqa: E402
from compbias.recoverability.phase_c_screen_result import (  # noqa: E402
    load_phase_c_screen_frozen_result,
    verify_phase_c_screen_artifacts,
)
from compbias.recoverability.phase_c_world_recovery import (  # noqa: E402
    PhaseCWorldRecoveryCall,
    PhaseCWorldRecoveryRecord,
    build_phase_c_world_recovery_calls,
    evaluate_phase_c_world_recovery_call,
    hidden_manifest_payload,
    load_phase_c_world_recovery_config,
    public_manifest_payload,
    record_payload,
    summarize_phase_c_world_recovery,
)
from compbias.recoverability.phase_c_world_recovery_execution import (  # noqa: E402
    verify_phase_c_world_recovery_package_lock,
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


def _exclusive_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite frozen world recovery evidence")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("world recovery output parent must be a regular directory")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _exclusive_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite frozen world recovery evidence")
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _validate_preflight(
    path: Path,
    *,
    lock: Path,
    package_files: list[str],
    model_call_cap: int,
) -> None:
    payload = load_strict_json_mapping(
        path, label="Phase C world recovery preflight", max_bytes=512 * 1024
    )
    if payload.get("artifact_type") != "recoverability_phase_c_world_recovery_metadata_preflight":
        raise ValueError("world recovery preflight artifact type is invalid")
    if payload.get("ready") is not True or payload.get("model_loaded") is not False:
        raise ValueError("world recovery preflight is not ready")
    if (
        payload.get("model_call_cap") != model_call_cap
        or payload.get("scale_authorized") is not False
    ):
        raise ValueError("world recovery preflight call cap differs")
    if payload.get("training_authorized") is not False:
        raise ValueError("world recovery preflight must not authorize training")
    if payload.get("server_package_lock_sha256") != _sha256(lock):
        raise ValueError("world recovery preflight lock digest differs")
    if payload.get("server_package_files") != package_files:
        raise ValueError("world recovery preflight package closure differs")


def decode_text_qwen_greedy_once(
    model: object,
    processor: object,
    messages: tuple[dict[str, object], ...],
    *,
    max_new_tokens: int,
) -> str:
    """Decode one deterministic text-only response with no image input."""

    import torch

    text = processor.apply_chat_template(  # type: ignore[attr-defined]
        list(messages), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(  # type: ignore[operator]
        text=[text], padding=True, return_tensors="pt"
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(  # type: ignore[attr-defined]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]  # type: ignore[attr-defined]


def _selected_valid_calls(
    calls: tuple[PhaseCWorldRecoveryCall, ...],
) -> tuple[PhaseCWorldRecoveryCall, ...]:
    return tuple(call for call in calls if call.condition == "valid_cue")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--qualification-config", type=Path)
    parser.add_argument("--system-prompt", type=Path)
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
        print("BLOCKED: Phase C world recovery requires explicit --execute")
        return 2
    required = tuple(value for key, value in vars(args).items() if key != "execute")
    if any(value is None for value in required):
        print("BLOCKED: world recovery requires every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("world recovery forbids COMPBIAS path overrides")

    root = Path(__file__).resolve().parents[2]
    matched = next(
        (
            (config_relative, profile)
            for config_relative, profile in _PROFILES.items()
            if args.qualification_config.resolve() == root / config_relative
            and args.server_package_lock.resolve() == root / profile["lock"]
        ),
        None,
    )
    if matched is None:
        raise ValueError("world recovery profile paths are not canonical")
    config_relative, profile = matched
    canonical = {
        "paths": root / "configs/paths.yaml",
        "runtime": root / "configs/recoverability/server_runtime_v1.yaml",
        "qualification_config": root / config_relative,
        "system_prompt": root / "prompts/world_recovery_v1_main.system.txt",
        "screen_result": root / "configs/recoverability/phase_c_screen_v2_frozen_result.yaml",
        "server_package_lock": root / profile["lock"],
    }
    for argument, expected in canonical.items():
        supplied = getattr(args, argument)
        if supplied.resolve() != expected or supplied.is_symlink():
            raise ValueError(f"world recovery {argument} path is not canonical")

    package = verify_phase_c_world_recovery_package_lock(
        canonical["server_package_lock"], repository_root=root
    )
    package_files = [item.relative_path for item in package.files]
    config = load_phase_c_world_recovery_config(canonical["qualification_config"])
    current = run_metadata_preflight(
        load_runtime_spec(canonical["runtime"]),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    if not current.ready or current.large_gpu_started or current.model_loaded:
        raise RuntimeError("world recovery runtime preflight failed")
    _validate_preflight(
        args.preflight_report,
        lock=canonical["server_package_lock"],
        package_files=package_files,
        model_call_cap=config.model_call_cap,
    )

    paths = load_pilot_paths(canonical["paths"], environ={})
    evidence_root = Path("/cloud/cloud-ssd1/recoverability-v1-evidence")
    expected_paths = {
        "preflight_report": evidence_root / profile["preflight"],
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
            raise ValueError(f"world recovery {label} path is not canonical")

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
    system_prompt = canonical["system_prompt"].read_text(encoding="utf-8")
    no_cue_template_path = root / "prompts/no_cue.user.template.txt"
    valid_cue_template_path = root / "prompts/valid_cue.user.template.txt"
    calls = build_phase_c_world_recovery_calls(
        eligible,
        config=config,
        system_prompt=system_prompt,
        no_cue_template=no_cue_template_path.read_text(encoding="utf-8"),
        valid_cue_template=valid_cue_template_path.read_text(encoding="utf-8"),
    )
    if len(calls) != config.model_call_cap or len({call.call_id for call in calls}) != len(calls):
        raise RuntimeError(
            f"world recovery plan must contain exactly {config.model_call_cap} calls"
        )
    if config.hypothesis_tested or config.scale_authorized or config.training_authorized:
        raise RuntimeError("world recovery must remain diagnostic and low-cost")

    model_hash = model_snapshot_sha256(paths.model_path)
    if model_hash != _EXPECTED_MODEL_SHA256 or model_hash != screen.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the frozen Phase C screen")

    output_parent = paths.outputs / "recoverability_v1"
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / config.output_subdirectory
    attempt = output_parent / profile["attempt"]
    if output.exists() or output.is_symlink() or attempt.exists() or attempt.is_symlink():
        raise FileExistsError("refusing to overwrite world recovery evidence")
    output.mkdir()
    valid_calls = _selected_valid_calls(calls)
    hidden_manifest = output / "manifest.hidden.jsonl"
    public_manifest = output / "manifest.public.jsonl"
    messages_path = output / "messages.jsonl"
    _exclusive_jsonl(hidden_manifest, tuple(hidden_manifest_payload(call) for call in valid_calls))
    _exclusive_jsonl(public_manifest, tuple(public_manifest_payload(call) for call in valid_calls))
    _exclusive_jsonl(
        messages_path,
        tuple(
            {
                "call_id": call.call_id,
                "scene_id": call.scene_id,
                "family": call.family,
                "case_index": call.case_index,
                "condition": call.condition,
                "messages": list(call.messages),
            }
            for call in calls
        ),
    )

    _exclusive_json(
        attempt,
        {
            "schema_version": 1,
            "status": "PHASE_C_WORLD_RECOVERY_STARTED_DO_NOT_RERUN",
            "qualification_id": config.qualification_id,
            "selected_cases": config.cases_per_family * len(config.families),
            "conditions": list(config.conditions),
            "model_call_cap": config.model_call_cap,
            "format_retries": 0,
            "do_sample": False,
            "hypothesis_tested": False,
            "scale_authorized": False,
            "training_authorized": False,
            "server_package_lock_sha256": _sha256(canonical["server_package_lock"]),
            "system_prompt_sha256": _sha256(canonical["system_prompt"]),
            "no_cue_template_sha256": _sha256(no_cue_template_path),
            "valid_cue_template_sha256": _sha256(valid_cue_template_path),
            "hidden_manifest_sha256": _sha256(hidden_manifest),
            "public_manifest_sha256": _sha256(public_manifest),
            "messages_sha256": _sha256(messages_path),
            "model_snapshot_sha256": model_hash,
            "training_invoked": False,
        },
    )
    max_format_retries = 0
    if max_format_retries != config.format_retries:
        raise RuntimeError("world recovery requires max_format_retries = 0")

    model, processor = load_local_qwen(paths.model_path)
    records: list[PhaseCWorldRecoveryRecord] = []
    for index, call in enumerate(calls, start=1):
        raw = decode_text_qwen_greedy_once(
            model,
            processor,
            call.messages,
            max_new_tokens=config.max_new_tokens,
        )
        records.append(evaluate_phase_c_world_recovery_call(call, raw))
        if index % 10 == 0 or index == len(calls):
            print(f"world_recovery_progress={index}/{len(calls)}", flush=True)
    if len(records) != config.model_call_cap:
        raise RuntimeError(f"world recovery did not stop at exactly {config.model_call_cap} calls")
    if model_snapshot_sha256(paths.model_path) != model_hash:
        raise RuntimeError("model snapshot changed during world recovery")

    frozen_records = tuple(records)
    records_path = output / "world_recovery_records.jsonl"
    _exclusive_jsonl(records_path, tuple(record_payload(record) for record in frozen_records))
    analysis = summarize_phase_c_world_recovery(frozen_records, config=config)
    report = {
        "schema_version": 1,
        "artifact_type": "recoverability_phase_c_world_recovery_report",
        "status": "COMPLETED_DIAGNOSTIC_WORLD_RECOVERY",
        "qualification_id": config.qualification_id,
        "dataset_id": config.dataset_id,
        "source_eligible_scenes": len(eligible),
        "selected_scene_ids": sorted({record.scene_id for record in frozen_records}),
        "decoding": {
            "do_sample": False,
            "max_new_tokens": config.max_new_tokens,
            "format_retries": 0,
        },
        "system_prompt_sha256": _sha256(canonical["system_prompt"]),
        "no_cue_template_sha256": _sha256(no_cue_template_path),
        "valid_cue_template_sha256": _sha256(valid_cue_template_path),
        "hidden_manifest_sha256": _sha256(hidden_manifest),
        "public_manifest_sha256": _sha256(public_manifest),
        "messages_sha256": _sha256(messages_path),
        "screen_report_sha256": dict(screen.source_sha256)["screen_report"],
        "server_package_lock_sha256": _sha256(canonical["server_package_lock"]),
        "model_snapshot_sha256": model_hash,
        "records_sha256": _sha256(records_path),
        "analysis": analysis,
        "hypothesis_tested": False,
        "scale_authorized": False,
        "training_authorized": False,
        "rl_authorized": False,
        "training_invoked": False,
    }
    _exclusive_json(output / "world_recovery_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
