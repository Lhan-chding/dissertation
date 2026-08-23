from __future__ import annotations

from collections.abc import Mapping, Sequence

from compensability.study_c3.evaluation_runtime import evaluate_grid, summarize_evaluation_rows


def _scene(condition: str) -> dict[str, object]:
    return {
        "scene_id": f"pair-a-{condition}",
        "pair_id": "pair-a",
        "split": "dev",
        "condition": condition,
        "family": "cross_series",
        "prompt": "Recover the world.",
        "prompt_sha256": ("a" if condition == "collision" else "b") * 64,
        "truth": [2, 3, 4, 5],
        "operation": {"operator": "sum", "indices": [0, 1]},
    }


class FakeSampler:
    def __init__(self) -> None:
        self.closed = False

    def __call__(
        self,
        row: Mapping[str, object],
        seeds: Sequence[int],
        decoder: str,
    ) -> tuple[dict[str, object], ...]:
        completion = "2,3,4,5" if decoder == "valid_world_fsa" else "[2,3,4,5]"
        return tuple(
            {
                "completion": completion,
                "token_ids": [2, 3, 4, 5],
                "mask_records": (
                    [{"step": 0, "allowed_token_ids": [2]}]
                    if decoder == "valid_world_fsa"
                    else []
                ),
            }
            for _seed in seeds
        )

    def close(self) -> None:
        self.closed = True


def test_grid_uses_common_scene_seeds_and_constrained_actions_are_valid() -> None:
    checkpoints = (
        {
            "checkpoint_id": "A_BIN-step192",
            "checkpoint_step": 192,
            "verifier": "answer",
            "validity_channel": "binary",
            "adapter_sha256": "a" * 64,
        },
        {
            "checkpoint_id": "X_LEX-step192",
            "checkpoint_step": 192,
            "verifier": "state",
            "validity_channel": "lex",
            "adapter_sha256": "b" * 64,
        },
    )
    raw, masks = evaluate_grid(
        rows=(_scene("collision"), _scene("separating")),
        checkpoints=checkpoints,
        decoders=("free_48", "valid_world_fsa"),
        rollout_count=2,
        seed_base=2026082501,
        sampler_factory=lambda _checkpoint: FakeSampler(),
    )

    assert len(raw) == 2 * 2 * 2 * 2
    grouped_seeds: dict[tuple[str, int], set[int]] = {}
    for row in raw:
        key = (str(row["scene_id"]), int(row["rollout_index"]))
        grouped_seeds.setdefault(key, set()).add(int(row["seed"]))
        if row["eval_decoder"] == "valid_world_fsa":
            assert row["canonical_parse_success"] is True
            assert row["semantic_parse_success"] is True
    assert all(len(seeds) == 1 for seeds in grouped_seeds.values())
    assert masks and all(row["eval_decoder"] == "valid_world_fsa" for row in masks)

    per_scene, summary = summarize_evaluation_rows(raw, expected_rollouts=2)
    assert len(per_scene) == 2 * 2 * 2
    assert summary["raw_row_count"] == len(raw)
    assert summary["same_scene_rollout_seeds"] is True


def test_grid_releases_each_checkpoint_sampler() -> None:
    samplers: list[FakeSampler] = []

    def factory(_checkpoint: Mapping[str, object]) -> FakeSampler:
        sampler = FakeSampler()
        samplers.append(sampler)
        return sampler

    evaluate_grid(
        rows=(_scene("collision"), _scene("separating")),
        checkpoints=(
            {
                "checkpoint_id": "A_BIN-step192",
                "checkpoint_step": 192,
                "verifier": "answer",
                "validity_channel": "binary",
                "adapter_sha256": "a" * 64,
            },
        ),
        decoders=("free_48",),
        rollout_count=1,
        seed_base=2026082501,
        sampler_factory=factory,
    )

    assert len(samplers) == 1
    assert samplers[0].closed is True
