"""SELF targets are obtained only from accepted raw teacher outputs."""

from .config import COMPANION, PROTOCOLS, digest
from .portfolio import index_bank


def discover_sets(tasks, records, selection, parent, repeat):
    if selection.get("parent_id") != parent or selection.get("split") != "V_selection":
        raise ValueError("Discovery requires same-parent V-only frozen selection")
    if digest({k: v for k, v in selection.items() if k != "selection_digest"}) != selection.get(
        "selection_digest"
    ):
        raise ValueError("Fixed policy selection was modified")
    bank = index_bank(tasks, records, parent, repeat, "T_train")
    sets = {"J_O0": {}, "J_MIX": {}, "J_SINGLE": {}}
    for task in tasks:
        tid, family = task["task_id"], task["family"]
        single = selection["best_single_B16"][family]
        if single not in PROTOCOLS[family]:
            raise ValueError("Unknown fixed protocol")
        allocations = {
            "J_O0": {"O0": 16},
            "J_MIX": {"O0": 8, COMPANION[family]: 8},
            "J_SINGLE": {single: 16},
        }
        for name, allocation in allocations.items():
            selected = [
                record
                for protocol, count in allocation.items()
                for record in bank[tid][protocol][:count]
            ]
            if len(selected) != 16:
                raise ValueError("Discovery policy must use exactly 16 atomic calls")
            successes = [r for r in selected if r["public_verifier_pass"]]
            if successes:
                worlds = {tuple(r["canonical_vector"]) for r in successes}
                if len(worlds) != 1:
                    raise ValueError(
                        "Public verifier returned conflicting canonical worlds; stop data unit"
                    )
                sets[name][tid] = {
                    "canonical_vector": list(next(iter(worlds))),
                    "discovery_request_ids": [r["request_id"] for r in successes],
                    "label_source_provenance": "self_public_verifier",
                    "policy_allocation": allocation,
                    "weight": 1,
                }
    o0, mix, single = (set(sets[k]) for k in ("J_O0", "J_MIX", "J_SINGLE"))
    return {
        **sets,
        "parent_id": parent,
        "pipeline_repeat": repeat,
        "selection_digest": selection["selection_digest"],
        "summary": {
            "counts": {k: len(v) for k, v in sets.items()},
            "mix_added": sorted(mix - o0),
            "mix_lost": sorted(o0 - mix),
            "mix_o0_intersection": sorted(mix & o0),
            "mix_single_intersection": sorted(mix & single),
            "physical_atomic_calls": len(records),
            "calls_per_task_per_policy": 16,
        },
        "empty_status": {k: "NO_VERIFIED_TARGETS_RETURN_PARENT" for k, v in sets.items() if not v},
    }
