"""Execute or preflight the three fail-closed Phase 4 language-only LoRA adapters."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compensability_v4.qwen.model_loader import (  # noqa: E402
    MODEL_PATH,
    load_pinned_qwen,
    require_server_model,
)
from compensability_v4.training.phase4 import (  # noqa: E402
    SupportVariant,
    attach_language_lora,
    discover_language_lora_targets,
    execute_lora_sft,
    freeze_base_parameters,
    load_phase4_config,
    load_support_rows,
    sha256_path,
    trainable_parameter_manifest,
    validate_phase4_preflight,
    verify_phase4_package_lock,
    write_parameter_manifests,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/recoverability/v4_phase_4.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_4.yaml"
_LOCKED_PATHS = (
    "configs/recoverability/v4_phase_4.yaml",
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "docs/QWEN_V4_SERVER_HANDOFF.md",
    "pyproject.toml",
    "requirements-gpu.lock.txt",
    "scripts/v4/07_prepare_phase4_support_sources.py",
    "scripts/v4/07_build_support_data.py",
    "scripts/v4/08_train_phase4_lora.py",
    "src/compensability_v4/training/__init__.py",
    "src/compensability_v4/training/phase4.py",
    "src/compensability_v4/training/phase4_sources.py",
)
_ACK = "I_UNDERSTAND_THIS_STARTS_PHASE_4_LORA_TRAINING"
PREPARED_SUPPORT = ROOT / "artifacts/v4/training/support.jsonl"
PREPARED_SUMMARY = ROOT / "artifacts/v4/training/support_summary.json"
DEFAULT_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"


def _validate_support_summary(path: Path, *, support_path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_path(support_path) != expected_sha256:
        raise RuntimeError("Phase 4 support corpus SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_type") != "phase_4_support_data"
        or payload.get("support_jsonl_sha256") != expected_sha256
        or payload.get("contains_confirmatory_data") is not False
        or payload.get("final_recovery_format") != "a,b,c,d"
    ):
        raise RuntimeError("Phase 4 support summary/provenance is malformed")


def _prepared_support_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Phase 4 prepared support summary is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.get("support_jsonl_sha256") if isinstance(payload, dict) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("Phase 4 prepared support summary/provenance is malformed")
    return digest


def _prepare_model(
    config: object,
) -> tuple[object, object, tuple[str, ...], dict[str, object], dict[str, object]]:
    model, processor = load_pinned_qwen(model_path=Path(MODEL_PATH), device_map="cuda:0")
    targets = discover_language_lora_targets(model)
    frozen = freeze_base_parameters(model)
    adapter = attach_language_lora(model, config=config, targets=targets)  # type: ignore[arg-type]
    manifest = trainable_parameter_manifest(adapter, targets)
    return adapter, processor, targets, frozen, manifest


def _release_model(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--prepared-support", action="store_true")
    parser.add_argument("--support", type=Path)
    parser.add_argument("--support-sha256")
    parser.add_argument("--support-summary", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/v4/training")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 4 LoRA requires explicit --execute.")
        return 2
    if not arguments.preflight_only and os.environ.get("COMPBIAS_V4_TRAINING_ACK") != _ACK:
        print("BLOCKED: COMPBIAS_V4_TRAINING_ACK is required before any optimizer step.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        config = load_phase4_config(arguments.config)
        verify_phase4_package_lock(
            lock_path=arguments.package_lock, repository_root=ROOT, expected_paths=_LOCKED_PATHS
        )
        explicit = (arguments.support, arguments.support_sha256, arguments.support_summary)
        if arguments.prepared_support:
            if any(item is not None for item in explicit):
                raise ValueError("--prepared-support cannot be mixed with explicit support flags")
            support_path = PREPARED_SUPPORT
            support_summary = PREPARED_SUMMARY
            support_sha256 = _prepared_support_sha256(support_summary)
        else:
            if any(item is None for item in explicit):
                raise ValueError(
                    "explicit support path, SHA-256, and summary are required unless "
                    "--prepared-support is used"
                )
            assert arguments.support is not None
            assert arguments.support_sha256 is not None
            assert arguments.support_summary is not None
            support_path = arguments.support
            support_sha256 = arguments.support_sha256
            support_summary = arguments.support_summary
        if len(support_sha256) != 64:
            raise RuntimeError("Phase 4 support SHA-256 is malformed")
        _validate_support_summary(
            support_summary,
            support_path=support_path,
            expected_sha256=support_sha256,
        )
        rows = {
            variant: load_support_rows(support_path, variant=variant) for variant in SupportVariant
        }
        validate_phase4_preflight(config=config, output_root=arguments.output_root)
        require_server_model(Path(MODEL_PATH))
        model, processor, _targets, frozen, manifest = _prepare_model(config)
        if arguments.preflight_only:
            _release_model(model)
            print(
                "READY: Phase 4 model, targets, frozen trainability, inputs, and "
                "output path preflight passed"
            )
            return 0
        write_parameter_manifests(
            artifact_root=arguments.artifact_root,
            trainable_manifest=manifest,
            frozen_hashes=frozen,
        )
        for index, variant in enumerate(SupportVariant):
            if index:
                model, processor, _targets, _frozen, _manifest = _prepare_model(config)
            execute_lora_sft(
                model=model,
                processor=processor,
                rows=rows[variant],
                variant=variant,
                config=config,
                output_root=arguments.output_root,
            )
            _release_model(model)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: Phase 4 C0/C1/T LoRA adapters written below {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
