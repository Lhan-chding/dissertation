"""Stable atomic streams and V-only selection at separate B16 and B2 budgets."""

from collections import defaultdict
from fractions import Fraction
from itertools import combinations_with_replacement
from math import comb

from .config import PROTOCOL_ORDER, PROTOCOLS, TEMPLATE_VERSION, digest
from .public_tasks import validate_public, verify_raw


def atomic_request(parent, task, protocol, role, repeat, draw, model_revision, runtime_version):
    validate_public(task)
    if parent not in ("S96", "REP96") or protocol not in PROTOCOLS[task["family"]]:
        raise ValueError("Unregistered parent/protocol")
    if task["split"] == "G_guard" and protocol != "O0":
        raise ValueError("Guard evaluation is O0 only")
    expected_role = {
        "T_train": "teacher_train",
        "V_selection": "selector",
        "E_test": "teacher_eval",
        "G_guard": "teacher_eval",
    }.get(task["split"])
    if (
        role != expected_role
        or type(repeat) is not int
        or repeat not in ((0, 1) if task["split"] == "T_train" else (0,))
    ):
        raise ValueError("Unregistered teacher role/repeat")
    limit = 4 if task["split"] == "G_guard" else 16
    if type(draw) is not int or not 0 <= draw < limit:
        raise ValueError("Draw outside frozen atomic budget")
    if not model_revision or not runtime_version:
        raise ValueError("Runtime/model identity required")
    identity = {
        "parent_id": parent,
        "model_revision": model_revision,
        "runtime_version": runtime_version,
        "task_id": task["task_id"],
        "base_instance_id": task["base_instance_id"],
        "split": task["split"],
        "protocol_id": protocol,
        "role": role,
        "pipeline_repeat": repeat,
        "draw_index": draw,
        "template_version": TEMPLATE_VERSION,
    }
    key = digest(identity)
    return {**identity, "request_id": key, "sample_seed": int(key[:16], 16) % (2**63 - 1)}


def index_bank(tasks, records, parent, repeat, split):
    by_task = {t["task_id"]: t for t in tasks}
    if len(by_task) != len(tasks) or not tasks or any(t["split"] != split for t in tasks):
        raise ValueError("Expected unique tasks from one registered split")
    bank = defaultdict(lambda: defaultdict(dict))
    requests = set()
    runtime_identities = set()
    for record in records:
        tid = record["task_id"]
        if (
            tid not in by_task
            or record["split"] != split
            or record["parent_id"] != parent
            or record["pipeline_repeat"] != repeat
        ):
            raise ValueError("Mixed role/parent/repeat or unexpected task in atomic bank")
        task = by_task[tid]
        expected = atomic_request(
            parent,
            task,
            record["protocol_id"],
            record["role"],
            repeat,
            record["draw_index"],
            record["model_revision"],
            record["runtime_version"],
        )
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Atomic identity/seed disagrees with frozen request")
        runtime_identities.add((record["model_revision"], record["runtime_version"]))
        if len(runtime_identities) > 1:
            raise ValueError("Teacher bank mixes model/runtime identities")
        request = record["request_id"]
        if request in requests or record["draw_index"] in bank[tid][record["protocol_id"]]:
            raise ValueError("Duplicate accepted atomic request; cannot retry a wrong answer")
        requests.add(request)
        scored = verify_raw(task, record["protocol_id"], record["raw_completion"])
        if any(k in record and record[k] != v for k, v in scored.items()):
            raise ValueError("Stored public score disagrees with raw completion")
        bank[tid][record["protocol_id"]][record["draw_index"]] = {**record, **scored}
    for task in tasks:
        for protocol in PROTOCOLS[task["family"]]:
            if set(bank[task["task_id"]][protocol]) != set(range(16)):
                raise ValueError("Incomplete atomic stream: " + task["task_id"] + "/" + protocol)
            bank[task["task_id"]][protocol] = [
                bank[task["task_id"]][protocol][i] for i in range(16)
            ]
    return dict(bank)


def pass_at_k(n, correct, k):
    if any(type(v) is not int for v in (n, correct, k)) or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Invalid pass@k counts")
    return Fraction(1) - Fraction(comb(n - correct, k) if n - correct >= k else 0, comb(n, k))


def pair_utility(left, right, same=False):
    if same:
        if left != right:
            raise ValueError("Repeated protocol must share exactly the same atomic stream")
        return pass_at_k(len(left), sum(left), 2)
    if not left or not right:
        raise ValueError("Nonempty independent protocol streams required")
    return 1 - Fraction(len(left) - sum(left), len(left)) * Fraction(
        len(right) - sum(right), len(right)
    )


def select_fixed_policies(tasks, records, parent):
    bank = index_bank(tasks, records, parent, 0, "V_selection")
    policies = {
        "parent_id": parent,
        "split": "V_selection",
        "best_single_B16": {},
        "best_single_B2": {},
        "best_pair_B2": {},
        "scores": {},
    }
    for family in ("trend", "cross_series"):
        group = [t for t in tasks if t["family"] == family]
        if not group:
            raise ValueError("Missing V family")
        protocols = PROTOCOLS[family]
        flags = {
            t["task_id"]: {
                p: [r["public_verifier_pass"] for r in bank[t["task_id"]][p]] for p in protocols
            }
            for t in group
        }
        scores16, scores2, scores1 = {}, {}, {}
        for p in protocols:
            scores16[p] = sum(
                (pass_at_k(16, sum(flags[t["task_id"]][p]), 16) for t in group), Fraction()
            ) / len(group)
            scores2[p] = sum(
                (pass_at_k(16, sum(flags[t["task_id"]][p]), 2) for t in group), Fraction()
            ) / len(group)
            scores1[p] = sum(
                (Fraction(sum(flags[t["task_id"]][p]), 16) for t in group), Fraction()
            ) / len(group)
        # Iteration order is registered O0,L11,B1; max keeps first exact tie.
        policies["best_single_B16"][family] = max(protocols, key=scores16.get)
        policies["best_single_B2"][family] = max(protocols, key=scores2.get)
        pairs = list(combinations_with_replacement(protocols, 2))
        pair_scores = {
            pair: sum(
                (
                    pair_utility(
                        flags[t["task_id"]][pair[0]],
                        flags[t["task_id"]][pair[1]],
                        pair[0] == pair[1],
                    )
                    for t in group
                ),
                Fraction(),
            )
            / len(group)
            for pair in pairs
        }
        selected = max(pairs, key=pair_scores.get)
        ordered = sorted(selected, key=lambda p: (-scores1[p], PROTOCOL_ORDER.index(p)))
        policies["best_pair_B2"][family] = ordered
        policies["scores"][family] = {
            "B16": {p: str(v) for p, v in scores16.items()},
            "B2": {p: str(v) for p, v in scores2.items()},
            "pairs_B2": {"+".join(p): str(v) for p, v in pair_scores.items()},
        }
    policies["validation_bank_digest"] = digest(sorted(r["request_id"] for r in records))
    policies["selection_digest"] = digest(policies)
    return policies


def replay_verify_first(task, first, second):
    """Offline deploy-policy replay; this function performs no model calls."""
    used = []
    for record in (first, second):
        if record["task_id"] != task["task_id"]:
            raise ValueError("Mismatched deployment replay task")
        used.append(record["request_id"])
        score = verify_raw(task, record["protocol_id"], record["raw_completion"])
        if score["public_verifier_pass"]:
            return {
                "verified": True,
                "canonical_vector": score["canonical_vector"],
                "used_request_ids": used,
                "calls": len(used),
                "offline_replay": True,
            }
    return {
        "verified": False,
        "canonical_vector": None,
        "used_request_ids": used,
        "calls": 2,
        "offline_replay": True,
    }
