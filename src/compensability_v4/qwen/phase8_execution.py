"""Shared execution helpers for the frozen Phase 8 confirmatory scripts."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import yaml

from compbias.gpu_pilot.chart_data import _draw_chart
from compbias.recoverability.compatibility import (
    ArithmeticProgressionConstraint,
    KnownValueConstraint,
    PairSumConstraint,
)
from compbias.recoverability.phase_c_screen import build_family_constraints
from compensability_v4.data.splits import DatasetSplit
from compensability_v4.eval.answer_source import classify_answer_source
from compensability_v4.qwen.manual_generation import generate_observation_with_cache
from compensability_v4.qwen.model_loader import MODEL_PATH, MODEL_SNAPSHOT_SHA256, load_pinned_qwen
from compensability_v4.qwen.phase5_runtime import (
    freeze_inference_model,
    generate_completion,
    tree_sha256,
)
from compensability_v4.qwen.phase5_support import HeldOutNaturalError, parse_world
from compensability_v4.schemas.scene import RecoveryScene

PHASE8_CONFIRM_ACK = "I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET"
PHASE8_LOCKED_PATHS = (
    "configs/recoverability/v4_phase_8.yaml",
    "configs/recoverability/v4/phase_1_3_prompts.yaml",
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "docs/QWEN_V4_SERVER_HANDOFF.md",
    "pyproject.toml",
    "requirements-gpu.lock.txt",
    "src/compensability_v4/data/splits.py",
    "src/compensability_v4/qwen/phase8_confirm_runtime.py",
    "src/compensability_v4/qwen/phase8_execution.py",
    "scripts/v4/18_freeze_phase8_confirm_data.py",
    "scripts/v4/19_evaluate_phase8_confirmatory.py",
)
SEVEN_CHECKPOINTS = (
    "Base",
    "C0",
    "C1",
    "T",
    "Base_AnswerOnly_RL",
    "Recovery_LoRA_RecoveryOutcome_RL",
    "Recovery_LoRA_AnswerOnly_RL",
)
PHASE4_PHASES = {
    "C0": "C0_format_only",
    "C1": "C1_forward_arithmetic",
    "T": "T_constraint_recovery",
}
PHASE6_CHECKPOINTS = set(SEVEN_CHECKPOINTS[4:])
OPERATIONS = ("sum", "difference", "max_minus_min")
FAMILIES = ("cross_series", "duplicate_encoding", "trend")
OOD_AXES = (
    "iid",
    "style_ood",
    "constraint_graph_ood",
    "error_mechanism_ood",
)
AXIS_SPLIT = {
    "iid": DatasetSplit.CONFIRM_IID,
    "style_ood": DatasetSplit.CONFIRM_STYLE_OOD,
    "constraint_graph_ood": DatasetSplit.CONFIRM_CONSTRAINT_OOD,
    "error_mechanism_ood": DatasetSplit.CONFIRM_ERROR_MECHANISM_OOD,
}
SOURCE_NAMES = (
    "legacy_diagnostic",
    "symbolic_support_train",
    "natural_error_support_train",
    "support_dev",
    "phase7_evaluation",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ANSWER = re.compile(r"\s*([+-]?\d+)\s*\Z")
_WORLD = re.compile(r"\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\Z")


@dataclass(frozen=True, slots=True)
class Phase8Template:
    source_scene_id: str
    family: str
    truth: tuple[int, int, int, int]
    chart_type: str
    operation: str
    question: str
    answer: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_execute(flag: bool, *, phase: str, action: str) -> None:
    if not flag:
        raise PermissionError(f"BLOCKED: {phase} {action} requires explicit --execute.")


def require_offline_env(*, phase: str) -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError(
            f"BLOCKED: {phase} requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1."
        )


def require_ack(value: str | None, *, phase: str) -> str:
    if value != PHASE8_CONFIRM_ACK:
        raise PermissionError(
            f"BLOCKED: {phase} requires COMPBIAS_V4_PHASE8_CONFIRM_ACK={PHASE8_CONFIRM_ACK}."
        )
    return value


def parse_named_bindings(
    entries: list[str] | None,
    *,
    option: str,
    expected_names: Sequence[str],
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in entries or ():
        name, separator, value = entry.partition("=")
        if not separator or name not in expected_names or not value or name in parsed:
            joined = ",".join(expected_names)
            raise ValueError(
                f"BLOCKED: Phase 8 {option} entries must be unique NAME=VALUE pairs for {joined}."
            )
        parsed[name] = value
    if set(parsed) != set(expected_names):
        joined = ",".join(expected_names)
        raise ValueError(f"BLOCKED: Phase 8 {option} must bind exactly {joined}.")
    return parsed


def require_matching_hashes(
    paths: Mapping[str, Path], *, expected_sha256: Mapping[str, str], phase: str
) -> dict[str, str]:
    observed = {name: sha256(path) for name, path in paths.items()}
    if observed != dict(expected_sha256):
        raise RuntimeError(f"BLOCKED: {phase} input SHA-256 bindings do not match.")
    return observed


def validate_phase7_evaluation(payload: Mapping[str, object]) -> dict[str, str]:
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("status") != "PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED"
        or payload.get("confirmatory_data_used") is not False
        or payload.get("support_dev_diagnostic") is not True
        or payload.get("training_invoked") is not False
    ):
        raise RuntimeError("Phase 7 evidence is not the formal diagnostic artifact")
    hashes = payload.get("checkpoint_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SEVEN_CHECKPOINTS):
        raise RuntimeError("Phase 7 evidence does not close all seven checkpoints")
    validated: dict[str, str] = {}
    for checkpoint in SEVEN_CHECKPOINTS:
        digest = hashes[checkpoint]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise RuntimeError("Phase 7 checkpoint evidence is malformed")
        validated[checkpoint] = digest
    return validated


def atomic_publish_directory(
    *,
    output_root: Path,
    text_files: Mapping[str, str],
    binary_writers: Mapping[str, bytes] | None = None,
    summary_name: str,
) -> dict[str, Path]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {summary_name}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent)))
    try:
        for relative, text in text_files.items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        for relative, payload in (binary_writers or {}).items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {name: output_root / name for name in (*text_files, *((binary_writers or {}).keys()))}


def load_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return payload


def load_jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{label} is malformed")
    return rows  # type: ignore[return-value]


def load_stage1_prompt(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Phase 8 Stage-1 prompt config is missing or unsafe")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    prompt = prompts.get("stage_1_observation") if isinstance(prompts, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Phase 8 Stage-1 observation prompt is missing")
    return prompt


def family_from_scene(scene: RecoveryScene) -> str:
    types = {str(dict(fact).get("type")) for fact in scene.facts}
    if "arithmetic_progression" in types:
        return "trend"
    pair_sum_count = sum(dict(fact).get("type") == "pair_sum" for fact in scene.facts)
    return "cross_series" if pair_sum_count >= 2 else "duplicate_encoding"


def reconstruct_support_dev_scene(error: HeldOutNaturalError) -> RecoveryScene:
    return RecoveryScene(
        scene_id=error.scene_id,
        split=DatasetSplit.SUPPORT_DEV,
        semantic_scene_id=f"phase5-semantic-{error.scene_id}",
        numeric_table_id=f"phase5-numbers-{error.scene_id}",
        constraint_graph_id=f"phase5-graph-{error.scene_id}",
        truth=error.truth,
        facts=tuple(dict(fact) for fact in error.facts),
        resized_height=280,
        resized_width=280,
        image_path=error.image_path,
    )


def load_symbolic_or_natural_scenes(path: Path, label: str) -> tuple[RecoveryScene, ...]:
    return tuple(RecoveryScene.from_mapping(row) for row in load_jsonl(path, label))


def load_support_dev_scenes(path: Path) -> tuple[RecoveryScene, ...]:
    return tuple(
        reconstruct_support_dev_scene(HeldOutNaturalError.from_mapping(row))
        for row in load_jsonl(path, "Phase 8 support-dev errors")
    )


def chart_type_for_scene_id(scene_id: str) -> str:
    return (
        "grouped_bar" if int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 2 == 0 else "line"
    )


def operation_for_scene_id(scene_id: str) -> str:
    return OPERATIONS[
        int(hashlib.sha256(f"op:{scene_id}".encode()).hexdigest(), 16) % len(OPERATIONS)
    ]


def question_for_operation(operation: str) -> str:
    return {
        "sum": "What is the sum of the first two values?",
        "difference": "What is the first value minus the second value?",
        "max_minus_min": "What is the maximum value minus the minimum value?",
    }[operation]


def apply_operation(world: tuple[int, int, int, int], operation: str) -> int:
    if operation == "sum":
        return world[0] + world[1]
    if operation == "difference":
        return world[0] - world[1]
    if operation == "max_minus_min":
        return max(world) - min(world)
    raise ValueError("Phase 8 operation is outside the frozen set")


def constraint_to_fact(constraint: object) -> Mapping[str, object]:
    if isinstance(constraint, PairSumConstraint):
        fact: dict[str, object] = {
            "type": "pair_sum",
            "left_index": constraint.left_index,
            "right_index": constraint.right_index,
            "total": constraint.total,
            "fact_id": constraint.constraint_id,
        }
    elif isinstance(constraint, KnownValueConstraint):
        fact = {
            "type": "known_value",
            "index": constraint.index,
            "value": constraint.value,
            "fact_id": constraint.constraint_id,
        }
    elif isinstance(constraint, ArithmeticProgressionConstraint):
        fact = {
            "type": "arithmetic_progression",
            "indices": constraint.indices,
            "fact_id": constraint.constraint_id,
        }
    else:
        raise TypeError("Phase 8 generated an unregistered visible constraint")
    return MappingProxyType(fact)


def select_confirm_templates(
    *,
    count: int,
    seed: int,
    reserved_truths: set[tuple[int, int, int, int]],
) -> tuple[Phase8Template, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("Phase 8 fixed candidate count must be positive")
    if any(len(world) != 4 for world in reserved_truths):
        raise ValueError("Phase 8 reserved truths are malformed")
    quotas = {family: count // len(FAMILIES) for family in FAMILIES}
    for family in FAMILIES[: count % len(FAMILIES)]:
        quotas[family] += 1
    rng = random.Random(seed)
    chosen = set(reserved_truths)
    selected: list[Phase8Template] = []
    for family, quota in quotas.items():
        for index in range(quota):
            candidate: tuple[int, int, int, int] | None = None
            for _attempt in range(4096):
                candidate = tuple(rng.randint(2, 18) for _ in range(4))
                if candidate in chosen:
                    continue
                try:
                    build_family_constraints(family, candidate)
                except ValueError:
                    continue
                chosen.add(candidate)
                break
            else:
                raise RuntimeError(
                    "Phase 8 candidate search exhausted before satisfying frozen constraints"
                )
            assert candidate is not None
            scene_id = f"phase8-source-{family}-{seed}-{index:04d}"
            operation = operation_for_scene_id(scene_id)
            selected.append(
                Phase8Template(
                    source_scene_id=scene_id,
                    family=family,
                    truth=candidate,
                    chart_type=chart_type_for_scene_id(scene_id),
                    operation=operation,
                    question=question_for_operation(operation),
                    answer=apply_operation(candidate, operation),
                )
            )
    return tuple(sorted(selected, key=lambda item: item.source_scene_id))


def build_scene(
    *,
    template: Phase8Template,
    axis: str,
    index: int,
    facts: Sequence[Mapping[str, object]],
) -> RecoveryScene:
    stem = f"phase8-{axis}-{index:04d}"
    return RecoveryScene(
        scene_id=stem,
        split=AXIS_SPLIT[axis],
        semantic_scene_id=f"phase8-semantic-{stem}",
        numeric_table_id=f"phase8-numbers-{stem}",
        constraint_graph_id=f"phase8-graph-{stem}",
        truth=template.truth,
        facts=tuple(dict(fact) for fact in facts),
        resized_height=280,
        resized_width=280,
        image_path=f"images/{axis}/{stem}.png",
    )


def constraint_ood_facts(template: Phase8Template) -> tuple[Mapping[str, object], ...]:
    truth = template.truth
    family = template.family
    if family == "cross_series":
        return (
            {"type": "pair_sum", "left_index": 0, "right_index": 2, "total": truth[0] + truth[2]},
            {"type": "pair_sum", "left_index": 1, "right_index": 3, "total": truth[1] + truth[3]},
            {"type": "pair_sum", "left_index": 0, "right_index": 3, "total": truth[0] + truth[3]},
            {"type": "pair_sum", "left_index": 1, "right_index": 2, "total": truth[1] + truth[2]},
        )
    if family == "duplicate_encoding":
        return (
            {"type": "known_value", "index": 1, "value": truth[1]},
            {"type": "known_value", "index": 2, "value": truth[2]},
            {"type": "known_value", "index": 3, "value": truth[3]},
            {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": truth[0] + truth[1]},
        )
    constraints = tuple(
        constraint_to_fact(fact) for fact in build_family_constraints("trend", truth)
    )
    return (*constraints[:2], {"type": "known_value", "index": 0, "value": truth[0]})


def render_phase8_image(
    *,
    scene: RecoveryScene,
    template: Phase8Template,
    axis: str,
    output_root: Path,
) -> Path:
    destination = output_root / scene.image_path
    render_mode = {
        "iid": "axis_scale_v0_3",
        "style_ood": "axis_scale_v0_3",
        "constraint_graph_ood": "axis_scale_v0_3",
        "error_mechanism_ood": "axis_scale_v0_2",
    }[axis]
    _draw_chart(
        destination,
        chart_type=template.chart_type,
        values=template.truth,
        size=(scene.resized_width, scene.resized_height),
        ood=axis in {"style_ood", "error_mechanism_ood"},
        render_mode=render_mode,
    )
    return destination


def observation_error_indices(
    *, truth: tuple[int, int, int, int], observed: tuple[int, int, int, int] | None
) -> tuple[int, ...]:
    if observed is None:
        return ()
    return tuple(
        index
        for index, (left, right) in enumerate(zip(observed, truth, strict=True))
        if left != right
    )


def checkpoint_adapter(checkpoint: str, phase4_root: Path, phase6_root: Path) -> Path | None:
    if checkpoint == "Base":
        return None
    if checkpoint in PHASE4_PHASES:
        return phase4_root / PHASE4_PHASES[checkpoint] / "final_adapter"
    return phase6_root / checkpoint / "final_adapter"


def checkpoint_hashes(phase4_root: Path, phase6_root: Path) -> dict[str, str]:
    hashes = {"Base": MODEL_SNAPSHOT_SHA256}
    for checkpoint in SEVEN_CHECKPOINTS[1:]:
        adapter = checkpoint_adapter(checkpoint, phase4_root, phase6_root)
        assert adapter is not None
        hashes[checkpoint] = tree_sha256(adapter)
    return hashes


def load_checkpoint_model(
    checkpoint: str, phase4_root: Path, phase6_root: Path
) -> tuple[object, object]:
    model, processor = load_pinned_qwen(model_path=Path(MODEL_PATH), device_map="cuda:0")
    adapter = checkpoint_adapter(checkpoint, phase4_root, phase6_root)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    freeze_inference_model(model)
    return model, processor


def release_model() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def revision_or_recovery(
    model: object,
    processor: object,
    *,
    observed_raw: str,
    facts: Sequence[Mapping[str, object]],
    seed: int,
    max_new_tokens: int,
) -> tuple[str, tuple[int, ...], tuple[int, int, int, int] | None]:
    prompt = (
        f"Observed values: {observed_raw}\n"
        f"Facts: {json.dumps(list(facts), sort_keys=True, separators=(',', ':'))}\n"
        "Revise the observed values only when required by the facts. "
        "Return exactly four comma-separated integers and no other text."
    )
    raw, token_ids = generate_completion(
        model,
        processor,
        prompt,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    return raw, tuple(token_ids), parse_world(raw)


def chart_operation(
    model: object,
    processor: object,
    *,
    question: str,
    seed: int,
    max_new_tokens: int,
) -> tuple[str, tuple[int, ...], str | None]:
    prompt = (
        f"Question: {question}\nChoose the required chart operation. "
        "Return exactly one of: sum, difference, max_minus_min."
    )
    raw, token_ids = generate_completion(
        model,
        processor,
        prompt,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    parsed = raw.strip() if raw.strip() in OPERATIONS else None
    return raw, tuple(token_ids), parsed


def final_answer(
    model: object,
    processor: object,
    *,
    recovered_raw: str,
    chosen_operation: str | None,
    seed: int,
    max_new_tokens: int,
) -> tuple[str, tuple[int, ...], int | None]:
    operation = chosen_operation if chosen_operation is not None else "INVALID_OPERATION"
    prompt = (
        f"Recovered values: {recovered_raw}\nChart operation: {operation}\n"
        "Apply the chart operation to the recovered values. Return the final integer only."
    )
    raw, token_ids = generate_completion(
        model,
        processor,
        prompt,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    match = _ANSWER.fullmatch(raw)
    return raw, tuple(token_ids), None if match is None else int(match.group(1))


def trace_mismatch(
    *, free_answer_value: int | None, deterministic_answer_value: int | None
) -> bool:
    return (
        free_answer_value is None
        or deterministic_answer_value is None
        or free_answer_value != deterministic_answer_value
    )


def image_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    absolute = (root / Path(*posix.parts)).resolve()
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or posix.suffix.lower() != ".png"
        or root.resolve() not in absolute.parents
        or absolute.is_symlink()
        or not absolute.is_file()
    ):
        raise RuntimeError("Phase 8 image is missing or unsafe")
    return absolute


def cache_checkpoint_rows(
    *,
    root: Path,
    checkpoint: str,
    rows: tuple[dict[str, object], ...] | None,
    expected_scene_ids: frozenset[str],
    checkpoint_sha256: str,
    execution_manifest_sha256: str,
    config_sha256: str,
    package_lock_sha256: str,
) -> tuple[dict[str, object], ...] | None:
    if not expected_scene_ids:
        raise ValueError("Phase 8 expected cache scene closure must not be empty")

    def validate_closure(cached_rows: tuple[dict[str, object], ...]) -> None:
        identities: list[str] = []
        for evidence in cached_rows:
            chain_row = evidence.get("chain_row") if isinstance(evidence, dict) else None
            if not isinstance(chain_row, dict) or chain_row.get("checkpoint") != checkpoint:
                raise RuntimeError(f"Phase 8 {checkpoint} trace cache checkpoint drifted")
            if chain_row.get("checkpoint_sha256") != checkpoint_sha256:
                raise RuntimeError(f"Phase 8 {checkpoint} trace cache checkpoint hash drifted")
            scene_id = chain_row.get("scene_id")
            if not isinstance(scene_id, str):
                raise RuntimeError(f"Phase 8 {checkpoint} trace cache scene closure drifted")
            identities.append(scene_id)
        if len(identities) != len(set(identities)) or set(identities) != set(expected_scene_ids):
            raise RuntimeError(f"Phase 8 {checkpoint} trace cache scene closure drifted")

    destination = root / checkpoint
    if rows is None:
        if not destination.exists():
            return None
        evidence_path = destination / "trace_evidence.jsonl"
        meta_path = destination / "meta.json"
        cached = load_jsonl(evidence_path, f"{checkpoint} trace cache")
        validate_closure(cached)
        meta = load_json(meta_path, f"{checkpoint} trace-cache metadata")
        expected = {
            "schema_version": 1,
            "status": "PHASE_8_CHECKPOINT_EVALUATION_COMPLETE",
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "execution_manifest_sha256": execution_manifest_sha256,
            "config_sha256": config_sha256,
            "package_lock_sha256": package_lock_sha256,
            "row_count": len(cached),
            "trace_evidence_sha256": sha256(evidence_path),
        }
        if meta != expected:
            raise RuntimeError(f"Phase 8 {checkpoint} trace cache drifted")
        return cached
    validate_closure(rows)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite Phase 8 {checkpoint} trace evidence")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint}-", dir=str(root)))
    try:
        evidence_path = temporary / "trace_evidence.jsonl"
        evidence_path.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (temporary / "meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PHASE_8_CHECKPOINT_EVALUATION_COMPLETE",
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": checkpoint_sha256,
                    "execution_manifest_sha256": execution_manifest_sha256,
                    "config_sha256": config_sha256,
                    "package_lock_sha256": package_lock_sha256,
                    "row_count": len(rows),
                    "trace_evidence_sha256": sha256(evidence_path),
                },
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return rows


def free_generation_answer_exact(answer_value: int | None, ground_truth_answer: int) -> bool:
    return answer_value == ground_truth_answer


def deterministic_chain_answer_exact(
    recovered_world: tuple[int, int, int, int] | None,
    chosen_operation: str | None,
    ground_truth_answer: int,
) -> tuple[bool, int | None]:
    if recovered_world is None or chosen_operation is None:
        return False, None
    computed = apply_operation(recovered_world, chosen_operation)
    return computed == ground_truth_answer, computed


def answer_source(
    *,
    answer_correct: bool,
    world_recovered: bool,
    operator_invariant: bool,
    error_cancelled: bool,
    visual_reread_evidence: bool,
) -> str:
    return classify_answer_source(
        answer_correct=answer_correct,
        world_recovered=world_recovered,
        operator_invariant=operator_invariant,
        error_cancelled=error_cancelled,
        visual_reread_evidence=visual_reread_evidence,
    ).value


__all__ = [
    "AXIS_SPLIT",
    "FAMILIES",
    "OOD_AXES",
    "OPERATIONS",
    "PHASE4_PHASES",
    "PHASE8_CONFIRM_ACK",
    "PHASE8_LOCKED_PATHS",
    "SEVEN_CHECKPOINTS",
    "Phase8Template",
    "answer_source",
    "apply_operation",
    "atomic_publish_directory",
    "build_scene",
    "cache_checkpoint_rows",
    "chart_operation",
    "chart_type_for_scene_id",
    "checkpoint_hashes",
    "constraint_ood_facts",
    "constraint_to_fact",
    "deterministic_chain_answer_exact",
    "family_from_scene",
    "final_answer",
    "free_generation_answer_exact",
    "generate_observation_with_cache",
    "image_path",
    "load_checkpoint_model",
    "load_json",
    "load_jsonl",
    "load_stage1_prompt",
    "load_support_dev_scenes",
    "load_symbolic_or_natural_scenes",
    "observation_error_indices",
    "operation_for_scene_id",
    "parse_named_bindings",
    "question_for_operation",
    "reconstruct_support_dev_scene",
    "release_model",
    "render_phase8_image",
    "require_ack",
    "require_execute",
    "require_matching_hashes",
    "require_offline_env",
    "revision_or_recovery",
    "select_confirm_templates",
    "sha256",
    "trace_mismatch",
    "validate_phase7_evaluation",
]
