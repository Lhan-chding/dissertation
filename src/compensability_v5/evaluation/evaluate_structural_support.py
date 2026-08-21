"""Scene-level point/orbit support and equivariance metrics for v5."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

REGISTERED_CHECKPOINTS = frozenset({"Base", "T", "B0", "B1", "B2", "B3"})
RELATIONAL_FAMILIES = frozenset({"pair_sum", "trend"})


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def pass_at_k(probability: object, k: int) -> float:
    """Compute exact-world pass@K from a per-rollout exact probability."""

    probability_value = _probability(probability, "exact_probability")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return 1.0 - (1.0 - probability_value) ** k


def _world(value: object, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{label} must contain exactly four integers")
    return tuple(value)  # type: ignore[return-value]


def _equivariant(source: Mapping[str, object]) -> bool:
    explicit = source.get("equivariance_consistent")
    if isinstance(explicit, bool):
        return explicit
    return _world(source.get("decoded_world"), "decoded_world") == _world(
        source.get("pushed_forward_canonical_world"), "pushed_forward_canonical_world"
    )


def evaluate_structural_support(
    records: Iterable[Mapping[str, object]],
    *,
    k: int,
) -> dict[str, object]:
    """Evaluate registered checkpoints with semantic scene as the unit."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    required = {
        "scene_id",
        "orbit_parent",
        "checkpoint",
        "family",
        "graph_axis",
        "is_canonical",
        "exact_probability",
    }
    rows: list[dict[str, object]] = []
    keys: set[tuple[str, str]] = set()
    for source in records:
        missing = required - set(source)
        if missing:
            raise ValueError(f"structural-support record missing fields: {sorted(missing)}")
        scene_id, parent, checkpoint, family, graph_axis = (
            source["scene_id"],
            source["orbit_parent"],
            source["checkpoint"],
            source["family"],
            source["graph_axis"],
        )
        identifiers_to_check = (scene_id, parent, family, graph_axis)
        if any(not isinstance(value, str) or not value for value in identifiers_to_check):
            raise ValueError("structural-support identifiers must be non-empty strings")
        if checkpoint not in REGISTERED_CHECKPOINTS:
            raise ValueError(f"unregistered checkpoint: {checkpoint}")
        if not isinstance(source["is_canonical"], bool):
            raise TypeError("is_canonical must be boolean")
        key = (str(checkpoint), str(scene_id))
        if key in keys:
            raise ValueError("checkpoint/scene rows must be unique")
        keys.add(key)
        probability = _probability(source["exact_probability"], "exact_probability")
        canonical = bool(source["is_canonical"])
        rows.append(
            {
                "scene_id": scene_id,
                "orbit_parent": parent,
                "checkpoint": checkpoint,
                "family": family,
                "graph_axis": graph_axis,
                "is_canonical": canonical,
                "exact_probability": probability,
                "pass_at_k": pass_at_k(probability, k),
                "equivariance_consistent": True if canonical else _equivariant(source),
            }
        )
    if not rows:
        raise ValueError("structural-support records must be non-empty")

    by_checkpoint: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_checkpoint[str(row["checkpoint"])].append(row)
    summaries: list[dict[str, object]] = []
    for checkpoint, group in sorted(by_checkpoint.items()):
        canonical_rows = [row for row in group if row["is_canonical"]]
        transformed_rows = [row for row in group if not row["is_canonical"]]
        parent_counts: dict[str, int] = defaultdict(int)
        for row in canonical_rows:
            parent_counts[str(row["orbit_parent"])] += 1
        represented_parents = {str(row["orbit_parent"]) for row in group}
        if (
            not canonical_rows
            or set(parent_counts) != represented_parents
            or any(count != 1 for count in parent_counts.values())
        ):
            raise ValueError(f"{checkpoint} requires one canonical row per represented orbit")
        relational_ood = [
            row
            for row in group
            if row["family"] in RELATIONAL_FAMILIES and row["graph_axis"] != "familiar"
        ]
        summaries.append(
            {
                "checkpoint": checkpoint,
                "canonical_scene_count": len(canonical_rows),
                "orbit_scene_count": len(group),
                "point_pass_at_k": sum(float(row["pass_at_k"]) for row in canonical_rows)
                / len(canonical_rows),
                "orbit_pass_at_k": sum(float(row["pass_at_k"]) for row in group) / len(group),
                "equivariance_defect": (
                    sum(not bool(row["equivariance_consistent"]) for row in transformed_rows)
                    / len(transformed_rows)
                    if transformed_rows
                    else 0.0
                ),
                "relational_graph_ood_pass_at_k": (
                    sum(float(row["pass_at_k"]) for row in relational_ood) / len(relational_ood)
                    if relational_ood
                    else None
                ),
            }
        )
    by_name = {str(row["checkpoint"]): row for row in summaries}
    b3, b2 = by_name.get("B3"), by_name.get("B2")
    contrast = None
    if (
        b3 is not None
        and b2 is not None
        and b3["relational_graph_ood_pass_at_k"] is not None
        and b2["relational_graph_ood_pass_at_k"] is not None
    ):
        contrast = float(b3["relational_graph_ood_pass_at_k"]) - float(
            b2["relational_graph_ood_pass_at_k"]
        )
    return {
        "schema_version": 1,
        "status": "V5_STRUCTURAL_SUPPORT_EVALUATED",
        "statistical_unit": "semantic_scene",
        "k": k,
        "scene_checkpoint_count": len(rows),
        "checkpoint_summaries": summaries,
        "primary_b3_minus_b2_relational_graph_ood": contrast,
        "rows": rows,
    }


def structural_support_fixture() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for checkpoint, canonical_p, orbit_p in (("B2", 0.75, 0.20), ("B3", 0.75, 0.65)):
        rows.extend(
            (
                {
                    "scene_id": f"{checkpoint}-canonical",
                    "orbit_parent": "fixture-parent",
                    "checkpoint": checkpoint,
                    "family": "pair_sum",
                    "graph_axis": "familiar",
                    "is_canonical": True,
                    "exact_probability": canonical_p,
                },
                {
                    "scene_id": f"{checkpoint}-basis",
                    "orbit_parent": "fixture-parent",
                    "checkpoint": checkpoint,
                    "family": "pair_sum",
                    "graph_axis": "equivalent_basis",
                    "is_canonical": False,
                    "exact_probability": orbit_p,
                    "equivariance_consistent": checkpoint == "B3",
                },
            )
        )
    return tuple(rows)


__all__ = [
    "REGISTERED_CHECKPOINTS",
    "RELATIONAL_FAMILIES",
    "evaluate_structural_support",
    "pass_at_k",
    "structural_support_fixture",
]
