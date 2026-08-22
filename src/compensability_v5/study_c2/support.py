"""Frozen-policy X/S/F/U support summaries and logical identification gates."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from .group_metrics import choose_group_size, summarize_group


def summarize_policy_support(
    rows: Sequence[Mapping[str, object]], *, group_candidates: Sequence[int] = (8, 16, 32)
) -> dict[str, object]:
    by_scene: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        kind = row.get("kind")
        scene_id = row.get("scene_id")
        if kind not in {"X", "S", "F", "U"} or not isinstance(scene_id, str):
            raise ValueError("support rows require scene_id and X/S/F/U kind")
        by_scene[scene_id].append(row)
    scene_rows: list[dict[str, object]] = []
    for scene_id, selected in sorted(by_scene.items()):
        counts = Counter(str(row["kind"]) for row in selected)
        total = len(selected)
        scene_rows.append(
            {
                "scene_id": scene_id,
                "rollout_count": total,
                "p_X": counts["X"] / total,
                "p_S": counts["S"] / total,
                "p_F": counts["F"] / total,
                "p_U": counts["U"] / total,
                "support_class": (
                    "mixed_support"
                    if counts["X"] and counts["S"]
                    else "exact_only"
                    if counts["X"]
                    else "shortcut_only"
                    if counts["S"]
                    else "neither"
                ),
            }
        )
    total_counts = Counter(str(row["kind"]) for row in rows)
    if total_counts["S"] == 0:
        status = "REWARD_CONTRAST_NOT_ESTIMABLE"
    elif total_counts["X"] == 0:
        status = "DISAGREEMENT_PRESENT_BUT_EXACT_CORRECTION_UNEXCITED"
    else:
        status = "REWARD_CONTRAST_IDENTIFIED"
    k_selection = choose_group_size(scene_rows, candidates=group_candidates)
    return {
        "status": status,
        "rollout_count": len(rows),
        "counts": {kind: total_counts[kind] for kind in ("X", "S", "F", "U")},
        "per_scene": scene_rows,
        "k_selection": k_selection,
    }


def summarize_realized_groups(
    rows: Sequence[Mapping[str, object]], *, group_size: int
) -> dict[str, float | int]:
    if len(rows) % group_size:
        raise ValueError("support rows do not divide into complete groups")
    summaries = [
        summarize_group(tuple(str(row["kind"]) for row in rows[start : start + group_size]))
        for start in range(0, len(rows), group_size)
    ]
    return {
        "group_count": len(summaries),
        **{
            metric: sum(summary[metric] is True for summary in summaries) / len(summaries)
            for metric in ("AIGR", "SIGR", "RDGR", "ESGR")
        },
    }


__all__ = ["summarize_policy_support", "summarize_realized_groups"]
