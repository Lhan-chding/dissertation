"""Scene-paired generalization gaps for preregistered OOD conditions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OODGeneralization:
    number_of_pairs: int
    iid_accuracy: float
    ood_accuracy: float
    generalization_gap: float


def paired_ood_generalization(
    iid_by_scene: Mapping[str, bool], ood_by_scene: Mapping[str, bool]
) -> OODGeneralization:
    if not isinstance(iid_by_scene, Mapping) or not isinstance(ood_by_scene, Mapping):
        raise TypeError("IID and OOD results must be mappings")
    if not iid_by_scene or iid_by_scene.keys() != ood_by_scene.keys():
        raise ValueError("IID and OOD results must have the same non-empty paired scene IDs")
    outcomes = (*iid_by_scene.values(), *ood_by_scene.values())
    if any(not isinstance(value, bool) for value in outcomes):
        raise TypeError("IID and OOD outcomes must be boolean")
    count = len(iid_by_scene)
    iid_accuracy = sum(iid_by_scene.values()) / count
    ood_accuracy = sum(ood_by_scene.values()) / count
    return OODGeneralization(
        number_of_pairs=count,
        iid_accuracy=iid_accuracy,
        ood_accuracy=ood_accuracy,
        generalization_gap=iid_accuracy - ood_accuracy,
    )
