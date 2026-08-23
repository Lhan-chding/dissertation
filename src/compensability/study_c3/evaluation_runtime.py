"""Shared-seed decoder grids and scene-level Study C3 evaluation summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence

from compensability_v4.qwen.phase5_runtime import phase5_rollout_seed

from .action_taxonomy import ActionClass, classify_action, classify_failure
from .semantic_action_parser import parse_p0, parse_p1

Generated = Mapping[str, object]
Sampler = Callable[[Mapping[str, object], Sequence[int], str], Sequence[Generated]]
SamplerFactory = Callable[[Mapping[str, object]], Sampler]


def _validate_scene(row: Mapping[str, object]) -> None:
    truth = row.get("truth")
    if (
        not isinstance(row.get("scene_id"), str)
        or not isinstance(row.get("pair_id"), str)
        or row.get("condition") not in {"collision", "separating"}
        or not isinstance(row.get("family"), str)
        or not isinstance(row.get("prompt"), str)
        or not isinstance(truth, list)
        or len(truth) != 4
        or any(type(value) is not int for value in truth)
        or not isinstance(row.get("operation"), Mapping)
    ):
        raise ValueError("Study C3 evaluation scene is malformed")


def evaluate_grid(
    *,
    rows: Sequence[Mapping[str, object]],
    checkpoints: Sequence[Mapping[str, object]],
    decoders: Sequence[str],
    rollout_count: int,
    seed_base: int,
    sampler_factory: SamplerFactory,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if not rows or not checkpoints or rollout_count <= 0:
        raise ValueError("Study C3 evaluation grid dimensions must be positive")
    if not decoders or set(decoders) - {"free_16", "free_48", "valid_world_fsa"}:
        raise ValueError("Study C3 evaluation contains an unregistered decoder")
    for row in rows:
        _validate_scene(row)
    checkpoint_ids = [checkpoint.get("checkpoint_id") for checkpoint in checkpoints]
    if any(not isinstance(value, str) for value in checkpoint_ids) or len(set(checkpoint_ids)) != len(
        checkpoint_ids
    ):
        raise ValueError("Study C3 checkpoint identities are malformed or duplicated")
    raw_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        sampler = sampler_factory(checkpoint)
        if not callable(sampler):
            raise ValueError("Study C3 sampler factory returned a non-callable object")
        try:
            for decoder in decoders:
                for scene_index, row in enumerate(rows):
                    scene_id = str(row["scene_id"])
                    seeds = tuple(
                        phase5_rollout_seed(seed_base, scene_id, rollout_index)
                        for rollout_index in range(rollout_count)
                    )
                    completions = tuple(sampler(row, seeds, decoder))
                    if len(completions) != rollout_count:
                        raise ValueError("Study C3 sampler returned the wrong rollout count")
                    truth = tuple(int(value) for value in row["truth"])
                    operation = row["operation"]
                    assert len(truth) == 4 and isinstance(operation, Mapping)
                    for rollout_index, generated in enumerate(completions):
                        completion = generated.get("completion")
                        token_ids = generated.get("token_ids")
                        masks = generated.get("mask_records", [])
                        if (
                            not isinstance(completion, str)
                            or not isinstance(token_ids, list)
                            or any(type(token) is not int for token in token_ids)
                            or not isinstance(masks, list)
                            or any(not isinstance(mask, Mapping) for mask in masks)
                        ):
                            raise ValueError("Study C3 sampler output is malformed")
                        canonical = parse_p0(completion)
                        semantic = parse_p1(completion)
                        label = classify_action(
                            semantic,
                            truth=truth,  # type: ignore[arg-type]
                            operation=operation,
                        )
                        if decoder == "valid_world_fsa" and canonical is None:
                            raise RuntimeError(
                                "Study C3 constrained decoder emitted an invalid action"
                            )
                        result = {
                            "schema_version": 3,
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "checkpoint_step": checkpoint.get("checkpoint_step"),
                            "verifier": checkpoint.get("verifier"),
                            "validity_channel": checkpoint.get("validity_channel"),
                            "adapter_sha256": checkpoint.get("adapter_sha256"),
                            "eval_decoder": decoder,
                            "scene_index": scene_index,
                            "scene_id": scene_id,
                            "pair_id": row["pair_id"],
                            "split": row.get("split"),
                            "condition": row["condition"],
                            "family": row["family"],
                            "prompt_sha256": row.get("prompt_sha256"),
                            "rollout_index": rollout_index,
                            "seed": seeds[rollout_index],
                            "completion": completion,
                            "completion_token_ids": token_ids,
                            "completion_token_length": len(token_ids),
                            "canonical_parse_success": canonical is not None,
                            "semantic_parse_success": semantic is not None,
                            "parsed_world": None if semantic is None else list(semantic),
                            "action_class": label.action_class.value,
                            "valid": label.valid,
                            "exact": label.exact,
                            "answer_correct": label.answer_correct,
                            "failure_category": classify_failure(completion).value,
                            "constrained_generation": decoder == "valid_world_fsa",
                        }
                        raw_rows.append(result)
                        for step, mask in enumerate(masks):
                            mask_rows.append(
                                {
                                    "schema_version": 3,
                                    "checkpoint_id": checkpoint["checkpoint_id"],
                                    "eval_decoder": decoder,
                                    "scene_id": scene_id,
                                    "rollout_index": rollout_index,
                                    "seed": seeds[rollout_index],
                                    "generation_step": step,
                                    **dict(mask),
                                }
                            )
        finally:
            close = getattr(sampler, "close", None)
            if callable(close):
                close()
    return tuple(raw_rows), tuple(mask_rows)


def _summarize_group(group: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = Counter(str(row["action_class"]) for row in group)
    valid = sum(int(row["valid"]) for row in group)
    exact = sum(int(row["exact"]) for row in group)
    answer = sum(int(row["answer_correct"]) for row in group)
    shortcut = counts[ActionClass.S.value]
    taxonomy = Counter(str(row["failure_category"]) for row in group)
    return {
        "rollout_count": len(group),
        "counts": {kind.value: counts[kind.value] for kind in ActionClass},
        "action_validity": valid / len(group),
        "truth_purity_given_valid": None if valid == 0 else exact / valid,
        "exact_recovery": exact / len(group),
        "answer_correctness": answer / len(group),
        "answer_given_valid": None if valid == 0 else answer / valid,
        "shortcut_rate": shortcut / len(group),
        "positive_truth_purity": None if answer == 0 else exact / answer,
        "canonical_parse_rate": sum(row["canonical_parse_success"] is True for row in group)
        / len(group),
        "semantic_parse_rate": valid / len(group),
        "mean_completion_token_length": sum(int(row["completion_token_length"]) for row in group)
        / len(group),
        "failure_taxonomy": dict(sorted(taxonomy.items())),
    }


def summarize_evaluation_rows(
    raw_rows: Sequence[Mapping[str, object]], *, expected_rollouts: int
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    seed_sets: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in raw_rows:
        checkpoint_id = row.get("checkpoint_id")
        decoder = row.get("eval_decoder")
        scene_id = row.get("scene_id")
        rollout_index = row.get("rollout_index")
        seed = row.get("seed")
        if (
            not isinstance(checkpoint_id, str)
            or not isinstance(decoder, str)
            or not isinstance(scene_id, str)
            or type(rollout_index) is not int
            or type(seed) is not int
        ):
            raise ValueError("Study C3 evaluation result row is malformed")
        grouped[(checkpoint_id, decoder, scene_id)].append(row)
        seed_sets[(scene_id, rollout_index)].add(seed)
    per_scene: list[dict[str, object]] = []
    for (checkpoint_id, decoder, scene_id), group in sorted(grouped.items()):
        if len(group) != expected_rollouts:
            raise ValueError("Study C3 scene cell has the wrong rollout count")
        source = group[0]
        per_scene.append(
            {
                "schema_version": 3,
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": source.get("checkpoint_step"),
                "verifier": source.get("verifier"),
                "validity_channel": source.get("validity_channel"),
                "eval_decoder": decoder,
                "scene_id": scene_id,
                "pair_id": source["pair_id"],
                "split": source.get("split"),
                "condition": source["condition"],
                "family": source["family"],
                **_summarize_group(group),
            }
        )
    by_cell: dict[str, object] = {}
    cell_groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in raw_rows:
        cell_groups[(str(row["checkpoint_id"]), str(row["eval_decoder"]))].append(row)
    for (checkpoint_id, decoder), group in sorted(cell_groups.items()):
        by_cell[f"{checkpoint_id}:{decoder}"] = _summarize_group(group)
    same_seeds = all(len(values) == 1 for values in seed_sets.values())
    if not same_seeds:
        raise ValueError("Study C3 evaluation cells do not share scene rollout seeds")
    return tuple(per_scene), {
        "schema_version": 3,
        "status": "STUDY_C3_EVALUATION_GRID_COMPLETE",
        "raw_row_count": len(raw_rows),
        "scene_cell_count": len(per_scene),
        "same_scene_rollout_seeds": True,
        "by_cell": by_cell,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": True,
    }


__all__ = ["evaluate_grid", "summarize_evaluation_rows"]
