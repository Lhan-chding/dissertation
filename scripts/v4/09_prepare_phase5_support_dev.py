"""Freeze a disjoint support-dev natural-error pool for Phase 5."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _guards import (  # noqa: E402
    PHASE_C_DATASET_MANIFEST_SHA256,
    PHASE_C_DATASET_RECORDS_SHA256,
    ROOT,
    sha256,
)

from compensability_v4.qwen.manual_generation import generate_observation_with_cache  # noqa: E402
from compensability_v4.qwen.model_loader import (  # noqa: E402
    MODEL_PATH,
    MODEL_SNAPSHOT_SHA256,
    load_pinned_qwen,
)
from compensability_v4.qwen.phase5_runtime import (  # noqa: E402
    load_phase5_config,
    verify_phase5_package_lock,
)
from compensability_v4.qwen.phase5_support import (  # noqa: E402
    build_support_dev_candidates,
    retain_held_out_natural_errors,
    write_support_dev_outputs,
)

CONFIG = ROOT / "configs/recoverability/v4_phase_5.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_5.yaml"
PROMPTS = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"
DATASET_ROOT = ROOT / "data/generated/cva_recoverability_causal_v2_screen"
PHASE4_SOURCE_ROOT = ROOT / "artifacts/v4/training/sources"
OUTPUT_ROOT = ROOT / "artifacts/v4/support_dev"
_LOCKED_PATHS = (
    "configs/recoverability/v4_phase_5.yaml",
    "configs/recoverability/v4/phase_1_3_prompts.yaml",
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "docs/QWEN_V4_SERVER_HANDOFF.md",
    "pyproject.toml",
    "requirements-gpu.lock.txt",
    "scripts/v4/09_prepare_phase5_support_dev.py",
    "scripts/v4/10_measure_policy_support.py",
    "src/compensability_v4/qwen/phase5_runtime.py",
    "src/compensability_v4/qwen/phase5_support.py",
)


def _json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 5 {label} must be a regular JSON file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Phase 5 {label} must contain one JSON object")
    return payload


def _jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 5 {label} must be a regular JSONL file")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Phase 5 {label} is empty or malformed")
    return rows  # type: ignore[return-value]


def _load_dataset(root: Path) -> tuple[tuple[dict[str, object], ...], dict[str, Path]]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError("Phase 5 dataset root must be an absolute regular directory")
    root = root.resolve()
    manifest, records = root / "manifest.json", root / "records.jsonl"
    if (
        sha256(manifest) != PHASE_C_DATASET_MANIFEST_SHA256
        or sha256(records) != PHASE_C_DATASET_RECORDS_SHA256
    ):
        raise RuntimeError("Phase 5 dataset manifest/records SHA-256 drifted")
    metadata, rows = _json(manifest, "dataset manifest"), _jsonl(records, "dataset records")
    if (
        metadata.get("record_count") != 8000
        or metadata.get("records_sha256") != PHASE_C_DATASET_RECORDS_SHA256
        or len(rows) != 8000
    ):
        raise RuntimeError("Phase 5 dataset structure drifted")
    images: dict[str, Path] = {}
    bundle: list[tuple[str, Path]] = []
    for row in rows:
        scene_id, relative = row.get("scene_id"), row.get("image")
        if not isinstance(scene_id, str) or not isinstance(relative, str):
            raise RuntimeError("Phase 5 dataset row identifiers are malformed")
        posix = PurePosixPath(relative)
        image = (root / Path(*posix.parts)).resolve()
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or posix.suffix.lower() != ".png"
            or root not in image.parents
            or image.is_symlink()
            or not image.is_file()
            or scene_id in images
        ):
            raise RuntimeError("Phase 5 dataset image is unsafe, missing, or duplicated")
        images[scene_id] = image
        bundle.append((relative, image))
    image_digest = hashlib.sha256()
    for relative, image in sorted(bundle):
        image_digest.update(relative.encode())
        image_digest.update(b"\0")
        image_digest.update(sha256(image).encode())
        image_digest.update(b"\n")
    if metadata.get("images_sha256") != image_digest.hexdigest():
        raise RuntimeError("Phase 5 dataset image bundle SHA-256 drifted")
    return rows, images


def _phase4_excluded_scenes(root: Path) -> tuple[frozenset[str], dict[str, str]]:
    summary_path, trace_path = root / "source_summary.json", root / "selection_trace.jsonl"
    summary, traces = (
        _json(summary_path, "Phase 4 source summary"),
        _jsonl(trace_path, "Phase 4 selection trace"),
    )
    output_hashes = summary.get("output_hashes")
    counts = summary.get("counts")
    if (
        summary.get("status") != "PHASE_4_SUPPORT_SOURCES_PREPARED_FROM_FROZEN_S6"
        or summary.get("contains_confirmatory_data") is not False
        or not isinstance(output_hashes, dict)
        or output_hashes.get("selection_trace") != sha256(trace_path)
        or not isinstance(counts, dict)
        or counts.get("selection_candidates") != len(traces)
        or len(traces) != 579
    ):
        raise RuntimeError("Phase 5 Phase-4 source exclusion evidence is malformed")
    scene_ids = tuple(trace.get("scene_id") for trace in traces)
    if any(not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids) or len(
        set(scene_ids)
    ) != len(scene_ids):
        raise RuntimeError("Phase 5 Phase-4 exclusion identifiers are malformed")
    return frozenset(scene_ids), {
        "phase4_source_summary": sha256(summary_path),
        "phase4_selection_trace": sha256(trace_path),
    }


def _stage1_prompt(path: Path) -> str:
    import yaml

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Phase 5 prompt config must be a regular file")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    prompt = prompts.get("stage_1_observation") if isinstance(prompts, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Phase 5 Stage-1 prompt is missing")
    return prompt


def _release(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--prompt-config", type=Path, default=PROMPTS)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--phase4-source-root", type=Path, default=PHASE4_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 5 support-dev preparation requires explicit --execute.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        payload, _measurement = load_phase5_config(arguments.config)
        lock_hash = verify_phase5_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=_LOCKED_PATHS,
        )
        support = payload["support_dev"]
        assert isinstance(support, dict)
        dataset_rows, images = _load_dataset(arguments.dataset_root)
        excluded, source_hashes = _phase4_excluded_scenes(arguments.phase4_source_root)
        candidates = build_support_dev_candidates(
            dataset_rows,
            excluded_scene_ids=excluded,
            count=int(support["intake_scene_count"]),
            seed=int(support["selection_seed"]),
        )
        if Counter(_family(scene) for scene in candidates) != Counter(
            {
                family: int(support["scenes_per_family"])
                for family in ("cross_series", "duplicate_encoding", "trend")
            }
        ):
            raise RuntimeError("Phase 5 support-dev family balance drifted")
        prompt = _stage1_prompt(arguments.prompt_config)
        model, processor = load_pinned_qwen(model_path=Path(MODEL_PATH))
        tokenizer = getattr(processor, "tokenizer", processor)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise RuntimeError("Phase 5 tokenizer has no decode()")
        output_by_scene: dict[str, str] = {}
        for index, scene in enumerate(candidates, start=1):
            evidence = generate_observation_with_cache(
                model,
                processor,
                str(images[scene.scene_id]),
                prompt,
                sample_id=scene.scene_id,
                resized_height=int(support["resized_height"]),
                resized_width=int(support["resized_width"]),
                max_new_tokens=int(support["stage1_max_new_tokens"]),
                rng_seed=int(support["selection_seed"]),
            )
            output_by_scene[scene.scene_id] = decode(
                evidence["generated_token_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if index % 25 == 0 or index == len(candidates):
                print(
                    f"PROGRESS: {index}/{len(candidates)} Phase 5 Stage-1 calls complete",
                    flush=True,
                )
        _release(model)
        errors, traces = retain_held_out_natural_errors(
            candidates,
            output_by_scene=output_by_scene,
            stage1_model_sha256=MODEL_SNAPSHOT_SHA256,
            value_domain=range(
                int(support["value_domain_min"]),
                int(support["value_domain_max"]) + 1,
            ),
        )
        if not errors:
            raise RuntimeError("Phase 5 support-dev produced no eligible single-error observations")
        source_hashes.update(
            dataset_manifest=PHASE_C_DATASET_MANIFEST_SHA256,
            dataset_records=PHASE_C_DATASET_RECORDS_SHA256,
            prompt_config=sha256(arguments.prompt_config),
        )
        paths = write_support_dev_outputs(
            output_root=arguments.output_root,
            candidates=candidates,
            errors=errors,
            traces=traces,
            source_sha256=source_hashes,
            config_sha256=sha256(arguments.config),
            package_lock_sha256=lock_hash,
        )
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(
        "READY: Phase 5 support-dev frozen; "
        f"intake={len(candidates)}, held_out_natural_errors={len(errors)}"
    )
    for path in paths.values():
        print(f"SHA256 {sha256(path)}  {path}")
    return 0


def _family(scene: object) -> str:
    facts = tuple(dict(fact) for fact in scene.facts)
    types = {str(fact.get("type")) for fact in facts}
    if "arithmetic_progression" in types:
        return "trend"
    pair_sums = sum(fact.get("type") == "pair_sum" for fact in facts)
    return "cross_series" if pair_sums >= 2 else "duplicate_encoding"


if __name__ == "__main__":
    raise SystemExit(main())
