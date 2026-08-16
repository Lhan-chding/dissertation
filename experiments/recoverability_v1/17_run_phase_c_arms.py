#!/usr/bin/env python3
"""Run the frozen 580-scene, six-arm, eight-fork Phase-C v3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml"
_INHERITED_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_c_screen_v2.yaml"
_ADDITIONS = frozenset(
    {
        "configs/recoverability/phase_c_screen_v2_frozen_result.yaml",
        "configs/recoverability/recoverability_phase_c_v3_postscreen_amendment.yaml",
        _INHERITED_LOCK_RELATIVE,
        "experiments/recoverability_v1/16_phase_c_arm_preflight.py",
        "experiments/recoverability_v1/17_run_phase_c_arms.py",
        "src/compbias/recoverability/paired_effects.py",
        "src/compbias/recoverability/phase_c_arm_execution.py",
        "src/compbias/recoverability/phase_c_arms.py",
        "src/compbias/recoverability/phase_c_postscreen_amendment.py",
        "src/compbias/recoverability/phase_c_screen_result.py",
        "src/compbias/recoverability/power.py",
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
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: Phase C arm package lock is malformed")
    return tuple(zip(paths, digests, strict=True))


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in sys.argv:
        return
    try:
        supplied = Path(sys.argv[sys.argv.index("--server-package-lock") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical Phase C arm lock is required") from error
    root = Path(__file__).resolve().parents[2]
    canonical = root / _LOCK_RELATIVE
    inherited = root / _INHERITED_LOCK_RELATIVE
    if supplied != canonical or canonical.is_symlink() or inherited.is_symlink():
        raise SystemExit("BLOCKED: Phase C arm package lock path is not canonical")
    inherited_paths = frozenset(relative for relative, _digest in _lock_rows(inherited))
    rows = _lock_rows(canonical)
    if frozenset(relative for relative, _digest in rows) != inherited_paths | _ADDITIONS:
        raise SystemExit("BLOCKED: Phase C arm package lock closure is incomplete")
    for relative, expected in rows:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Phase C arm package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.config import load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import load_local_qwen  # noqa: E402
from compbias.io.strict_json import load_strict_json_mapping  # noqa: E402
from compbias.recoverability.paired_effects import (  # noqa: E402
    ArmForks,
    SceneCrossover,
    paired_scene_effect,
)
from compbias.recoverability.phase_c_arm_execution import (  # noqa: E402
    verify_phase_c_arm_execution_package_lock,
)
from compbias.recoverability.phase_c_arms import (  # noqa: E402
    PHASE_C_ARMS,
    PhaseCArmRecord,
    build_phase_c_arm_calls,
    evaluate_phase_c_arm_call,
)
from compbias.recoverability.phase_c_postscreen_amendment import (  # noqa: E402
    load_phase_c_postscreen_amendment,
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
        raise FileExistsError("refusing to rerun Phase C arm execution")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("Phase C arm output parent must be a regular directory")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _validate_preflight(path: Path, *, lock: Path, package_files: list[str]) -> None:
    payload = load_strict_json_mapping(path, label="Phase C arm preflight", max_bytes=512 * 1024)
    if payload.get("artifact_type") != "recoverability_phase_c_v3_arm_metadata_preflight":
        raise ValueError("Phase C arm preflight artifact type is invalid")
    if payload.get("ready") is not True or payload.get("model_loaded") is not False:
        raise ValueError("Phase C arm preflight is not ready")
    if payload.get("training_authorized") is not False:
        raise ValueError("Phase C arm preflight must not authorize training")
    if payload.get("server_package_lock_sha256") != _sha256(lock):
        raise ValueError("Phase C arm preflight lock digest differs")
    if payload.get("server_package_files") != package_files:
        raise ValueError("Phase C arm preflight package closure differs")


def decode_text_qwen_seeded_once(
    model: object,
    processor: object,
    messages: tuple[dict[str, object], ...],
    *,
    seed: int,
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
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
    trimmed = [
        output[len(source) :]
        for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]  # type: ignore[attr-defined]


def _scene_crossovers(records: tuple[PhaseCArmRecord, ...]) -> tuple[SceneCrossover, ...]:
    grouped: dict[str, list[PhaseCArmRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scene_id].append(record)
    scenes: list[SceneCrossover] = []
    for scene_id, rows in grouped.items():
        by_arm: dict[str, list[PhaseCArmRecord]] = defaultdict(list)
        for row in rows:
            by_arm[row.arm].append(row)
        if set(by_arm) != set(PHASE_C_ARMS):
            raise RuntimeError("Phase C scene is missing a registered arm")
        family = rows[0].family
        arms = tuple(
            ArmForks(
                arm,
                tuple(
                    item.faithful_success
                    for item in sorted(by_arm[arm], key=lambda value: value.fork_index)
                ),
            )
            for arm in PHASE_C_ARMS
        )
        scenes.append(SceneCrossover(scene_id, family, family, arms, 8))
    return tuple(sorted(scenes, key=lambda item: item.scene_id))


def _analysis(records: tuple[PhaseCArmRecord, ...]) -> dict[str, object]:
    scenes = _scene_crossovers(records)
    rates = {
        arm: sum(row.faithful_success for row in records if row.arm == arm)
        / sum(row.arm == arm for row in records)
        for arm in PHASE_C_ARMS
    }
    confirmatory = tuple(
        scene for scene in scenes if scene.family in {"cross_series", "trend"}
    )
    effects = {
        "equal_family_confirmatory": asdict(
            paired_scene_effect(
                confirmatory,
                treatment_arm="valid",
                control_arm="ablated",
                bootstrap_resamples=10_000,
                seed=2026081901,
            )
        )
    }
    for offset, family in enumerate(("cross_series", "trend", "duplicate_encoding"), 1):
        effects[family] = asdict(
            paired_scene_effect(
                tuple(scene for scene in scenes if scene.family == family),
                treatment_arm="valid",
                control_arm="ablated",
                bootstrap_resamples=10_000,
                seed=2026081901 + offset,
            )
        )
    return {
        "faithful_rate_by_arm": rates,
        "valid_minus_ablated": effects,
        "parse_rate": sum(row.program_parse_success for row in records) / len(records),
        "execution_rate": sum(row.program_execution_success for row in records) / len(records),
        "faithful_successes": sum(row.faithful_success for row in records),
        "error_counts": dict(sorted(Counter(row.error_code or "none" for row in records).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--postscreen-amendment", type=Path)
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
        print("BLOCKED: Phase C arms require explicit --execute")
        return 2
    required = tuple(value for key, value in vars(args).items() if key != "execute")
    if any(value is None for value in required):
        print("BLOCKED: Phase C arms require every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("Phase C arms forbid COMPBIAS path overrides")

    root = Path(__file__).resolve().parents[2]
    canonical = {
        "paths": root / "configs/paths.yaml",
        "runtime": root / "configs/recoverability/server_runtime_v1.yaml",
        "postscreen_amendment": (
            root / "configs/recoverability/recoverability_phase_c_v3_postscreen_amendment.yaml"
        ),
        "screen_result": root / "configs/recoverability/phase_c_screen_v2_frozen_result.yaml",
        "server_package_lock": root / _LOCK_RELATIVE,
    }
    for argument, expected in canonical.items():
        supplied = getattr(args, argument)
        if supplied.resolve() != expected or supplied.is_symlink():
            raise ValueError(f"Phase C arm {argument} path is not canonical")
    package = verify_phase_c_arm_execution_package_lock(
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
        raise RuntimeError("Phase C arm runtime preflight failed")
    _validate_preflight(
        args.preflight_report,
        lock=canonical["server_package_lock"],
        package_files=package_files,
    )
    paths = load_pilot_paths(canonical["paths"], environ={})
    evidence_root = Path("/cloud/cloud-ssd1/recoverability-v1-evidence")
    expected_paths = {
        "preflight_report": evidence_root / "phase-c-arms-v3-preflight.json",
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
            raise ValueError(f"Phase C arm {label} path is not canonical")

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
    amendment = load_phase_c_postscreen_amendment(
        canonical["postscreen_amendment"], screen=screen
    )
    calls = build_phase_c_arm_calls(eligible, amendment=amendment)
    if (
        len(eligible) != 580
        or len(calls) != 27840
        or len({call.call_id for call in calls}) != 27840
    ):
        raise RuntimeError("Phase C arm plan does not contain the exact frozen crossover")
    if not amendment.confirmatory_arm_execution_authorized or amendment.training_authorized:
        raise RuntimeError("Phase C v3 amendment does not authorize arm execution")

    output_parent = paths.outputs / "recoverability_v1"
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / amendment.output_subdirectory / "phase_c_arms"
    attempt = output_parent / f"{amendment.output_subdirectory}.arms.attempted.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite Phase C arm evidence")
    model_hash = model_snapshot_sha256(paths.model_path)
    if model_hash != _EXPECTED_MODEL_SHA256 or model_hash != screen.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the frozen screen")
    _exclusive_marker(
        attempt,
        {
            "schema_version": 1,
            "status": "PHASE_C_V3_ARMS_STARTED_DO_NOT_RERUN",
            "amendment_id": amendment.amendment_id,
            "original_screen_passed": False,
            "fixed_family_quota_gate_withdrawn": True,
            "frozen_eligible_scenes": 580,
            "arms": list(PHASE_C_ARMS),
            "forks_per_arm": 8,
            "model_call_cap": 27840,
            "format_retries": 0,
            "server_package_lock_sha256": _sha256(canonical["server_package_lock"]),
            "model_snapshot_sha256": model_hash,
            "training_invoked": False,
        },
    )
    max_format_retries = 0
    if max_format_retries != amendment.format_retries:
        raise RuntimeError("Phase C arms require max_format_retries = 0")
    model, processor = load_local_qwen(paths.model_path)
    records: list[PhaseCArmRecord] = []
    for call in calls:
        raw = decode_text_qwen_seeded_once(
            model, processor, call.messages, seed=call.sampling_seed
        )
        records.append(evaluate_phase_c_arm_call(call, raw))
    if len(records) != amendment.model_call_cap:
        raise RuntimeError("Phase C arms did not stop at exactly 27,840 calls")
    if model_snapshot_sha256(paths.model_path) != model_hash:
        raise RuntimeError("model snapshot changed during Phase C arm execution")
    frozen_records = tuple(records)
    report = {
        "schema_version": 1,
        "artifact_type": "recoverability_phase_c_v3_arm_report",
        "status": "COMPLETED_AMENDED_CONFIRMATORY_UNDER_ORIGINAL_POWER_TARGET",
        "amendment_id": amendment.amendment_id,
        "original_screen_passed": False,
        "original_screen_exit": 3,
        "fixed_family_quota_gate_withdrawn": True,
        "original_target_power_met": False,
        "frozen_eligible_scenes": len(eligible),
        "eligible_by_family": dict(amendment.frozen_eligible_by_family),
        "arms": list(PHASE_C_ARMS),
        "forks_per_arm": 8,
        "model_calls": len(records),
        "format_retries": 0,
        "screen_report_sha256": dict(screen.source_sha256)["screen_report"],
        "server_package_lock_sha256": _sha256(canonical["server_package_lock"]),
        "model_snapshot_sha256": model_hash,
        "analysis": _analysis(frozen_records),
        "training_authorized": False,
        "rl_authorized": False,
        "training_invoked": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phase-c-arms-", dir=output.parent) as temp:
        staging = Path(temp) / output.name
        staging.mkdir()
        with (staging / "arm_records.jsonl").open("x", encoding="utf-8") as stream:
            for record in frozen_records:
                stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")
        report["arm_records_sha256"] = _sha256(staging / "arm_records.jsonl")
        with (staging / "arm_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        staging.rename(output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
