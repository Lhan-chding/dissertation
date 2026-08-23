"""Read-only C3-A reclassification of frozen Study C2 completions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from .action_taxonomy import classify_action, classify_failure
from .semantic_action_parser import (
    PARSERS,
    World,
    parse_complete_action_any_domain,
    parse_p2,
)

TokenCounter = Callable[[str], int]


def _fiber_index(fiber_rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in fiber_rows:
        scene_id = row.get("scene_id")
        truth = row.get("truth")
        operation = row.get("operation")
        if (
            not isinstance(scene_id, str)
            or not isinstance(truth, list)
            or len(truth) != 4
            or any(type(value) is not int for value in truth)
            or not isinstance(operation, Mapping)
            or scene_id in result
        ):
            raise ValueError("Study C3 fiber source rows are malformed or duplicated")
        result[scene_id] = row
    if not result:
        raise ValueError("Study C3 action audit requires frozen fiber rows")
    return result


def _parser_metrics(
    completion: str,
    *,
    truth: World,
    operation: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for parser_name, parser in PARSERS.items():
        label = classify_action(parser(completion), truth=truth, operation=operation)
        result[parser_name] = {
            "valid": label.valid,
            "exact": label.exact,
            "answer_correct": label.answer_correct,
        }
    return result


def audit_existing_rows(
    raw_rows: Sequence[Mapping[str, object]],
    fiber_rows: Sequence[Mapping[str, object]],
    *,
    token_counter: TokenCounter,
) -> tuple[dict[str, object], ...]:
    fibers = _fiber_index(fiber_rows)
    audited: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for row_index, row in enumerate(raw_rows):
        scene_id = row.get("scene_id")
        arm = row.get("arm")
        rollout_index = row.get("rollout_index")
        completion = row.get("completion")
        if (
            not isinstance(scene_id, str)
            or scene_id not in fibers
            or arm not in {"C2_answer_reward", "C2_exact_state_reward"}
            or type(rollout_index) is not int
            or rollout_index < 0
            or not isinstance(completion, str)
        ):
            raise ValueError(f"Study C2 evaluation row {row_index} is malformed")
        identity = (str(arm), scene_id, rollout_index)
        if identity in seen:
            raise ValueError("Study C2 evaluation rows contain a duplicate rollout")
        seen.add(identity)
        source = fibers[scene_id]
        for key in ("pair_id", "condition", "family", "split"):
            if row.get(key) != source.get(key):
                raise ValueError(f"Study C2 evaluation metadata drifted for {scene_id}: {key}")
        truth = tuple(int(value) for value in source["truth"])
        operation = source["operation"]
        assert len(truth) == 4 and isinstance(operation, Mapping)
        metrics = _parser_metrics(
            completion,
            truth=truth,  # type: ignore[arg-type]
            operation=operation,
        )
        original_parse = row.get("parse_success")
        if (
            type(original_parse) is not bool
            or metrics["P0"]["valid"] != int(original_parse)
            or metrics["P0"]["exact"] != row.get("state_reward")
            or metrics["P0"]["answer_correct"] != row.get("answer_reward")
        ):
            raise ValueError(f"Study C3 P0 does not reproduce frozen Study C2 row {row_index}")
        token_length = token_counter(completion)
        if type(token_length) is not int or token_length < 0:
            raise ValueError("Study C3 token counter returned an invalid length")
        complete = parse_complete_action_any_domain(completion) is not None or parse_p2(
            completion,
            minimum=-(2**31),
            maximum=2**31 - 1,
        ) is not None
        audited.append(
            {
                "schema_version": 3,
                "source_row_index": row_index,
                "arm": arm,
                "scene_id": scene_id,
                "pair_id": row["pair_id"],
                "split": row["split"],
                "condition": row["condition"],
                "family": row["family"],
                "rollout_index": rollout_index,
                "seed": row.get("seed"),
                "completion": completion,
                "completion_character_length": len(completion),
                "completion_token_length": token_length,
                "fourth_value_completed_before_token_budget": complete,
                "failure_category": classify_failure(completion).value,
                "original_parse_success": original_parse,
                "original_state_reward": row["state_reward"],
                "original_answer_reward": row["answer_reward"],
                **metrics,
            }
        )
    if not audited:
        raise ValueError("Study C3 action audit cannot be empty")
    return tuple(audited)


def _summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Study C3 action-audit stratum cannot be empty")
    parser_summaries: dict[str, object] = {}
    for parser_name in PARSERS:
        parser_rows = [row.get(parser_name) for row in rows]
        if any(not isinstance(value, Mapping) for value in parser_rows):
            raise ValueError("Study C3 parser audit row is malformed")
        valid = sum(int(value["valid"]) for value in parser_rows)  # type: ignore[index]
        exact = sum(int(value["exact"]) for value in parser_rows)  # type: ignore[index]
        answer = sum(int(value["answer_correct"]) for value in parser_rows)  # type: ignore[index]
        parser_summaries[parser_name] = {
            "parse_rate": valid / len(rows),
            "exact_rate": exact / len(rows),
            "answer_rate": answer / len(rows),
            "truth_purity_given_valid": None if valid == 0 else exact / valid,
            "answer_purity_given_valid": None if valid == 0 else answer / valid,
        }
    failure_counts = Counter(str(row["failure_category"]) for row in rows)
    lengths = [int(row["completion_character_length"]) for row in rows]
    token_lengths = [int(row["completion_token_length"]) for row in rows]
    return {
        "row_count": len(rows),
        "parsers": parser_summaries,
        "failure_taxonomy": {
            key: {"count": value, "rate": value / len(rows)}
            for key, value in sorted(failure_counts.items())
        },
        "completion_character_length": {
            "mean": sum(lengths) / len(lengths),
            "minimum": min(lengths),
            "maximum": max(lengths),
        },
        "completion_token_length": {
            "mean": sum(token_lengths) / len(token_lengths),
            "minimum": min(token_lengths),
            "maximum": max(token_lengths),
        },
        "fourth_value_completed_before_token_budget_rate": sum(
            row["fourth_value_completed_before_token_budget"] is True for row in rows
        )
        / len(rows),
    }


def summarize_action_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def stratify(key: str) -> dict[str, object]:
        values = sorted({str(row[key]) for row in rows})
        return {
            value: _summarize_rows([row for row in rows if str(row[key]) == value])
            for value in values
        }

    return {
        "schema_version": 3,
        "status": "STUDY_C3_ACTION_CHANNEL_AUDIT_COMPLETE",
        "row_count": len(rows),
        "overall": _summarize_rows(rows),
        "by_arm": stratify("arm"),
        "by_family": stratify("family"),
        "by_condition": stratify("condition"),
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


__all__ = ["audit_existing_rows", "summarize_action_audit"]
