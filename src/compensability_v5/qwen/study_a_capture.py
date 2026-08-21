"""Phase-2a natural-observation capture and child-manifest publication."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from compensability_v4.qwen.phase5_runtime import freeze_inference_model, phase5_rollout_seed
from compensability_v4.qwen.phase5_support import HeldOutNaturalError, parse_world

from .study_a_execution import (
    CheckpointLoader,
    _append_trace,
    _load_or_create_trace,
    _row_sha256,
    run_study_a,
)
from .study_a_scenarios import (
    _INTEGER,
    BASE_SHA256,
    OBSERVATION_PROMPT,
    OBSERVATION_PROMPT_VERSION,
    PHASE2A_PARENT_MANIFEST_SHA256,
    World,
    _apply_operation,
    _canonical_json,
    _fiber_bin,
    _fiber_size,
    _prompt,
    _sha256_bytes,
    _world,
    build_phase2_study_a_scenarios,
    build_study_a_scenarios,
    sha256_file,
)


def _read_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return payload


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{label} is empty or malformed")
    return rows  # type: ignore[return-value]


def load_phase2a_parents(
    phase2a_root: Path,
    *,
    expected_parent_count: int = 96,
    expected_parent_manifest_sha256: str = PHASE2A_PARENT_MANIFEST_SHA256,
) -> tuple[tuple[dict[str, object], ...], str]:
    """Verify the immutable Phase-2a parent and return one familiar row per semantic scene."""

    if phase2a_root.is_symlink() or not phase2a_root.is_dir():
        raise RuntimeError("Phase-2a root is missing or unsafe")
    manifest_path = phase2a_root / "parent_manifest.json"
    rows_path = phase2a_root / "pre_model_rows.jsonl"
    manifest = _read_json(manifest_path, "Phase-2a parent manifest")
    if sha256_file(manifest_path) != expected_parent_manifest_sha256:
        raise RuntimeError("Phase-2a parent manifest SHA-256 differs from the frozen pilot")
    rows = _read_jsonl(rows_path, "Phase-2a parent rows")
    if (
        manifest.get("status") != "PHASE_2A_PRE_MODEL_FROZEN"
        or manifest.get("row_count") != len(rows)
        or manifest.get("rows_sha256") != sha256_file(rows_path)
        or manifest.get("model_calls") != 0
        or manifest.get("observation_capture_required") is not True
    ):
        raise RuntimeError("Phase-2a parent manifest provenance drifted")
    familiar = tuple(row for row in rows if row.get("graph_axis") == "familiar")
    if (
        len(familiar) != expected_parent_count
        or len({row.get("semantic_scene_id") for row in familiar}) != len(familiar)
        or any(row.get("observation_status") != "pending_server_capture" for row in rows)
    ):
        raise RuntimeError("Phase-2a familiar parent closure drifted")
    for row in familiar:
        relative = row.get("image_path")
        digest = row.get("image_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("Phase-2a familiar image binding is malformed")
        image = phase2a_root / relative
        if image.resolve(strict=False).is_relative_to(phase2a_root.resolve()) is False:
            raise RuntimeError("Phase-2a familiar image path escapes its root")
        if sha256_file(image) != digest:
            raise RuntimeError("Phase-2a familiar image SHA-256 mismatch")
    return tuple(sorted(familiar, key=lambda row: str(row["semantic_scene_id"]))), sha256_file(
        manifest_path
    )


def _parse_natural_observation(raw: str) -> tuple[World | None, bool]:
    strict = parse_world(raw)
    if strict is not None:
        return strict, True
    tokens = _INTEGER.findall(raw)
    if len(tokens) != 4:
        return None, False
    return tuple(int(token) for token in tokens), False  # type: ignore[return-value]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_capture_row(row: Mapping[str, object], semantic_id: str) -> None:
    raw = row.get("raw_output")
    token_ids = row.get("generated_token_ids")
    parsed, strict = _parse_natural_observation(raw) if isinstance(raw, str) else (None, False)
    expected_observation = list(parsed) if parsed is not None else None
    if (
        row.get("row_sha256") != _row_sha256(row)
        or row.get("schema_version") != 1
        or row.get("checkpoint") != "BaseObservation"
        or row.get("scenario_id") != semantic_id
        or row.get("semantic_scene_id") != semantic_id
        or not isinstance(raw, str)
        or not isinstance(token_ids, list)
        or not all(type(token) is int for token in token_ids)
        or row.get("strict_parse_success") is not strict
        or row.get("natural_observation") != expected_observation
        or parsed is None
    ):
        raise RuntimeError("Phase-2a resumed observation row provenance drifted")


def _capture_observation(
    model: object,
    processor: object,
    *,
    image_path: Path,
    sample_id: str,
    seed: int,
) -> tuple[str, tuple[int, ...]]:  # pragma: no cover - real pinned visual runtime
    shortcut = getattr(model, "study_a_observe", None)
    if callable(shortcut):
        raw, token_ids = shortcut(
            image_path=image_path,
            prompt=OBSERVATION_PROMPT,
            sample_id=sample_id,
            seed=seed,
        )
        return str(raw), tuple(int(item) for item in token_ids)
    from PIL import Image

    from compensability_v4.qwen.manual_generation import generate_observation_with_cache

    with Image.open(image_path) as image:
        result = generate_observation_with_cache(
            model,
            processor,
            image.convert("RGB"),
            OBSERVATION_PROMPT,
            sample_id=sample_id,
            resized_height=280,
            resized_width=280,
            max_new_tokens=32,
            rng_seed=seed,
        )
    return str(result["text"]), tuple(result["generated_token_ids"])  # type: ignore[arg-type]


def _observation_label(truth: World, observed: World, *, strict_parse: bool) -> str:
    errors = sum(left != right for left, right in zip(truth, observed, strict=True))
    in_domain = all(2 <= value <= 18 for value in observed)
    if errors == 1 and in_domain:
        return "primary_single_in_domain"
    if errors == 0 and in_domain:
        return "no_error_control"
    if errors > 1 and in_domain:
        return "stress_multiple_error"
    if errors > 1:
        return "stress_multiple_error_out_of_domain"
    if not in_domain:
        return "stress_out_of_domain"
    return "strict" if strict_parse else "relaxed_parse"


def _build_frozen_scene(
    parent: Mapping[str, object], capture: Mapping[str, object]
) -> dict[str, object]:
    observed = _world(capture["natural_observation"], "captured natural observation")
    truth = _world(parent["truth"], "Phase-2a truth")
    operation = parent["answer_operation"]
    if not isinstance(operation, Mapping):
        raise RuntimeError("Phase-2a answer operation drifted")
    indices = tuple(operation["indices"])  # type: ignore[arg-type]
    answer = _apply_operation(truth, str(operation["operator"]), indices)
    fiber_size = _fiber_size(
        observed,
        operation=str(operation["operator"]),
        indices=indices,
        answer=answer,
    )
    return {
        "schema_version": 1,
        "scene_id": str(parent["scene_id"]),
        "semantic_scene_id": str(parent["semantic_scene_id"]),
        "family": parent["family"],
        "prompt": _prompt(
            observed,
            tuple(tuple(row) for row in parent["constraint_matrix"]),  # type: ignore[arg-type]
            tuple(parent["constraint_targets"]),  # type: ignore[arg-type]
        ),
        "truth": list(truth),
        "natural_observation": list(observed),
        "constraint_matrix": parent["constraint_matrix"],
        "constraint_targets": parent["constraint_targets"],
        "answer_operation": dict(operation),
        "correct_answer": answer,
        "transformation": dict(parent["transformation"]),  # type: ignore[arg-type]
        "graph_axis": "canonical",
        "capture_label": _observation_label(
            truth, observed, strict_parse=bool(capture["strict_parse_success"])
        ),
        "observation_in_domain": all(2 <= value <= 18 for value in observed),
        "error_count": sum(left != right for left, right in zip(truth, observed, strict=True)),
        "fiber_size": fiber_size,
        "fiber_bin": _fiber_bin(fiber_size),
        "fiber_definition": {
            "distance": "hamming_at_most_one",
            "value_domain": [2, 18],
        },
        "observation_prompt_version": OBSERVATION_PROMPT_VERSION,
        "observation_raw_output": capture["raw_output"],
        "observation_strict_parse_success": capture["strict_parse_success"],
        "image_path": parent["image_path"],
        "image_sha256": parent["image_sha256"],
    }


def capture_phase2a_natural_observations(
    *,
    phase2a_root: Path,
    output_root: Path,
    work_root: Path,
    model: object,
    processor: object,
    expected_parent_count: int = 96,
    expected_parent_manifest_sha256: str = PHASE2A_PARENT_MANIFEST_SHA256,
    seed: int = 2026082101,
) -> tuple[tuple[dict[str, object], ...], str]:
    """Capture Base observations once and publish an append-only child manifest."""

    parents, parent_manifest_sha256 = load_phase2a_parents(
        phase2a_root,
        expected_parent_count=expected_parent_count,
        expected_parent_manifest_sha256=expected_parent_manifest_sha256,
    )
    parent_rows_sha256 = sha256_file(phase2a_root / "pre_model_rows.jsonl")
    image_bundle_sha256 = _sha256_bytes(
        "".join(
            f"{row['semantic_scene_id']}\0{row['image_path']}\0{row['image_sha256']}\n"
            for row in parents
        ).encode()
    )
    observation_prompt_sha256 = _sha256_bytes(OBSERVATION_PROMPT.encode())
    metadata = {
        "schema_version": 1,
        "status": "V5_PHASE2A_OBSERVATION_TRACE_IN_PROGRESS",
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_rows_sha256": parent_rows_sha256,
        "image_bundle_sha256": image_bundle_sha256,
        "base_sha256": BASE_SHA256,
        "processor_source_sha256": BASE_SHA256,
        "observation_prompt": OBSERVATION_PROMPT,
        "observation_prompt_sha256": observation_prompt_sha256,
        "observation_prompt_version": OBSERVATION_PROMPT_VERSION,
        "seed": seed,
        "semantic_scene_ids": [row["semantic_scene_id"] for row in parents],
    }
    trace_path, completed = _load_or_create_trace(work_root, metadata)
    # The generic trace loader keys by checkpoint/scenario; capture rows use the
    # same two explicit fields to reuse its duplicate and truncation defenses.
    expected_capture_keys = {("BaseObservation", str(row["semantic_scene_id"])) for row in parents}
    if set(completed) - expected_capture_keys:
        raise RuntimeError("Phase-2a observation trace contains unregistered rows")
    for key, row in completed.items():
        _validate_capture_row(row, key[1])
    for parent in parents:
        semantic_id = str(parent["semantic_scene_id"])
        key = ("BaseObservation", semantic_id)
        if key in completed:
            continue
        raw, token_ids = _capture_observation(
            model,
            processor,
            image_path=phase2a_root / str(parent["image_path"]),
            sample_id=semantic_id,
            seed=phase5_rollout_seed(seed, semantic_id, 0),
        )
        observed, strict_parse = _parse_natural_observation(raw)
        row = {
            "schema_version": 1,
            "checkpoint": "BaseObservation",
            "scenario_id": semantic_id,
            "semantic_scene_id": semantic_id,
            "raw_output": raw,
            "generated_token_ids": list(token_ids),
            "strict_parse_success": strict_parse,
            "natural_observation": list(observed) if observed is not None else None,
        }
        completed[key] = _append_trace(trace_path, row)
        if observed is None:
            raise RuntimeError(
                f"Phase-2a neutral observation is not deterministically parseable: {semantic_id}"
            )
    captures = {key[1]: row for key, row in completed.items()}
    if set(captures) != {str(row["semantic_scene_id"]) for row in parents}:
        raise RuntimeError("Phase-2a observation capture closure is incomplete")
    frozen = [
        _build_frozen_scene(parent, captures[str(parent["semantic_scene_id"])])
        for parent in parents
    ]
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Phase-2a child observation publication already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        scenes_path = temporary / "frozen_scenes.jsonl"
        with scenes_path.open("x", encoding="utf-8") as stream:
            for row in frozen:
                stream.write(_canonical_json(row) + "\n")
        trace_copy = temporary / "observation_trace.jsonl"
        trace_copy.write_bytes(trace_path.read_bytes())
        child = {
            "schema_version": 1,
            "status": "V5_PHASE2A_NATURAL_OBSERVATIONS_FROZEN",
            "parent_manifest_sha256": parent_manifest_sha256,
            "parent_rows_sha256": parent_rows_sha256,
            "image_bundle_sha256": image_bundle_sha256,
            "parent_manifest_modified": False,
            "base_sha256": BASE_SHA256,
            "processor_source_sha256": BASE_SHA256,
            "observation_prompt_sha256": observation_prompt_sha256,
            "semantic_scene_count": len(frozen),
            "capture_label_counts": dict(
                sorted(
                    (label, sum(row["capture_label"] == label for row in frozen))
                    for label in {str(row["capture_label"]) for row in frozen}
                )
            ),
            "frozen_scenes_sha256": sha256_file(scenes_path),
            "observation_trace_sha256": sha256_file(trace_copy),
            "prompt_search_invoked": False,
            "training_invoked": False,
            "rl_invoked": False,
        }
        child_path = temporary / "child_manifest.json"
        child_path.write_text(_canonical_json(child) + "\n", encoding="utf-8")
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return tuple(frozen), sha256_file(output_root / "child_manifest.json")


def load_phase2a_child(
    output_root: Path,
    *,
    phase2a_root: Path,
    expected_parent_count: int = 96,
    expected_parent_manifest_sha256: str = PHASE2A_PARENT_MANIFEST_SHA256,
) -> tuple[tuple[dict[str, object], ...], str]:
    manifest_path = output_root / "child_manifest.json"
    scenes_path = output_root / "frozen_scenes.jsonl"
    trace_path = output_root / "observation_trace.jsonl"
    manifest = _read_json(manifest_path, "Phase-2a child manifest")
    scenes = _read_jsonl(scenes_path, "Phase-2a frozen scenes")
    trace = _read_jsonl(trace_path, "Phase-2a observation trace")
    parents, parent_manifest_sha256 = load_phase2a_parents(
        phase2a_root,
        expected_parent_count=expected_parent_count,
        expected_parent_manifest_sha256=expected_parent_manifest_sha256,
    )
    parent_rows_sha256 = sha256_file(phase2a_root / "pre_model_rows.jsonl")
    image_bundle_sha256 = _sha256_bytes(
        "".join(
            f"{row['semantic_scene_id']}\0{row['image_path']}\0{row['image_sha256']}\n"
            for row in parents
        ).encode()
    )
    semantic_ids = {str(row["semantic_scene_id"]) for row in parents}
    scene_ids = {str(row.get("semantic_scene_id")) for row in scenes}
    trace_ids = {str(row.get("semantic_scene_id")) for row in trace}
    for row in trace:
        _validate_capture_row(row, str(row.get("semantic_scene_id")))
    trace_by_id = {str(row["semantic_scene_id"]): row for row in trace}
    expected_scenes = tuple(
        _build_frozen_scene(parent, trace_by_id[str(parent["semantic_scene_id"])])
        for parent in parents
        if str(parent["semantic_scene_id"]) in trace_by_id
    )
    expected_label_counts = dict(
        sorted(
            (label, sum(row["capture_label"] == label for row in expected_scenes))
            for label in {str(row["capture_label"]) for row in expected_scenes}
        )
    )
    if (
        manifest.get("status") != "V5_PHASE2A_NATURAL_OBSERVATIONS_FROZEN"
        or manifest.get("semantic_scene_count") != len(scenes)
        or len(trace) != len(scenes)
        or semantic_ids != scene_ids
        or semantic_ids != trace_ids
        or scenes != expected_scenes
        or manifest.get("capture_label_counts") != expected_label_counts
        or manifest.get("frozen_scenes_sha256") != sha256_file(scenes_path)
        or manifest.get("observation_trace_sha256") != sha256_file(trace_path)
        or manifest.get("parent_manifest_sha256") != parent_manifest_sha256
        or manifest.get("parent_rows_sha256") != parent_rows_sha256
        or manifest.get("image_bundle_sha256") != image_bundle_sha256
        or manifest.get("parent_manifest_modified") is not False
        or manifest.get("base_sha256") != BASE_SHA256
        or manifest.get("processor_source_sha256") != BASE_SHA256
        or manifest.get("observation_prompt_sha256") != _sha256_bytes(OBSERVATION_PROMPT.encode())
        or any(
            not _is_sha256(manifest.get(field))
            for field in (
                "parent_manifest_sha256",
                "parent_rows_sha256",
                "image_bundle_sha256",
                "frozen_scenes_sha256",
                "observation_trace_sha256",
                "observation_prompt_sha256",
            )
        )
        or manifest.get("prompt_search_invoked") is not False
        or manifest.get("training_invoked") is not False
        or manifest.get("rl_invoked") is not False
    ):
        raise RuntimeError("Phase-2a child observation provenance drifted")
    return scenes, sha256_file(manifest_path)


def run_phase2a_study_a(
    *,
    phase2a_root: Path,
    child_root: Path,
    capture_work_root: Path,
    output_root: Path,
    audit_work_root: Path,
    checkpoint_loader: CheckpointLoader,
    legacy_errors: Iterable[HeldOutNaturalError] | None = None,
    legacy_raw_archive_sha256: str | None = None,
    expected_parent_count: int = 96,
    expected_parent_manifest_sha256: str = PHASE2A_PARENT_MANIFEST_SHA256,
    k: int = 8,
    sampling_seed: int = 2026082101,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """Capture all frozen parents, then execute their five-axis Base/T audit."""

    if child_root.exists() or child_root.is_symlink():
        frozen, child_sha256 = load_phase2a_child(
            child_root,
            phase2a_root=phase2a_root,
            expected_parent_count=expected_parent_count,
            expected_parent_manifest_sha256=expected_parent_manifest_sha256,
        )
    else:
        base, processor = checkpoint_loader("Base")
        freeze_inference_model(base)
        try:
            frozen, child_sha256 = capture_phase2a_natural_observations(
                phase2a_root=phase2a_root,
                output_root=child_root,
                work_root=capture_work_root,
                model=base,
                processor=processor,
                expected_parent_count=expected_parent_count,
                expected_parent_manifest_sha256=expected_parent_manifest_sha256,
                seed=sampling_seed,
            )
        finally:
            del base
    phase2_scenarios = tuple(
        scenario for scene in frozen for scenario in build_phase2_study_a_scenarios(scene)
    )
    legacy_scenarios = tuple(
        scenario for error in (legacy_errors or ()) for scenario in build_study_a_scenarios(error)
    )
    if bool(legacy_scenarios) != bool(legacy_raw_archive_sha256):
        raise ValueError("legacy diagnostic errors and raw archive hash must be supplied together")
    summary = run_study_a(
        scenarios=(*phase2_scenarios, *legacy_scenarios),
        source_name="phase2a_child_manifest",
        source_sha256=child_sha256,
        additional_source_sha256=(
            {"raw_archive": legacy_raw_archive_sha256}
            if legacy_raw_archive_sha256 is not None
            else None
        ),
        output_root=output_root,
        work_root=audit_work_root,
        checkpoint_loader=checkpoint_loader,
        k=k,
        sampling_seed=sampling_seed,
        progress=progress,
    )
    return summary
