"""Strict SER-J23 identity, inherited O0 prompts and role-bound file access.

These capabilities prevent accidental role mixing; operating-system isolation is
still required for model workers. Audit metadata never enters a public prompt.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from ..exposure_substitution.schema import (
    canonical_target as canonical_target,
)
from ..exposure_substitution.schema import (
    digest as digest,
)
from ..exposure_substitution.schema import (
    file_digest as file_digest,
)
from ..exposure_substitution.schema import (
    read_json as read_json,
)
from ..exposure_substitution.schema import (
    read_jsonl as read_jsonl,
)
from ..exposure_substitution.schema import (
    valid_vector as valid_vector,
)
from ..exposure_substitution.schema import (
    write_json as write_json,
)
from ..exposure_substitution.schema import (
    write_jsonl as write_jsonl,
)
from ..prompts import OPERATIONS, SYSTEM_PROMPT, USER_TEMPLATE
from ..protocol_state_probes.transforms import format_linear

EXPERIMENT_ID = PHASE_ID = "SER_J23_20261008"
PARENTS = ("S96", "REP96")
SCHEDULE_SEEDS = (108701, 108702, 108703)
ARMS = {
    "A2_LOCAL_C1_J2": {"center_0based": 0, "corrupted_index_0based": 1, "legacy_arm": "A_LOCAL_C1"},
    "B2_FORWARD_C4_J2": {
        "center_0based": 3,
        "corrupted_index_0based": 1,
        "legacy_arm": "B_FORWARD_C4",
    },
    "A3_LOCAL_C1_J3": {"center_0based": 0, "corrupted_index_0based": 2, "legacy_arm": None},
    "B3_FORWARD_C4_J3": {"center_0based": 3, "corrupted_index_0based": 2, "legacy_arm": None},
}
MICROBATCH_SLOTS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10), (11,), (12, 13, 14, 15))
TRAIN_SPLITS = frozenset({"COMMON_TRAIN", "DONOR_TRAIN", "REPLAY"})
DEV_SPLITS = frozenset({"DEV_TRAJECTORY", "TRAIN_FIT_FULL", "DEV_DELTA"})
SPLITS = TRAIN_SPLITS | DEV_SPLITS | {"E_CONFIRM2"}
PUBLIC_KEYS = frozenset(
    {
        "phase_id",
        "task_id",
        "root_id",
        "base_instance_id",
        "split",
        "family",
        "observed",
        "H_original",
        "b_original",
        "legal_domain",
        "operation",
        "template_version",
        "chart_type",
        "interface",
        "image_path",
        "image_sha256",
    }
)
ROLE_SPLITS = {
    "trainer": TRAIN_SPLITS,
    "diagnostic": TRAIN_SPLITS | DEV_SPLITS,
    "confirm_worker": frozenset({"E_CONFIRM2"}),
    "auditor": SPLITS,
    "analysis": SPLITS,
}
PUBLIC_FILES = {
    "COMMON_TRAIN": "common_targets.jsonl",
    "REPLAY": "replay_targets.jsonl",
    "DONOR_TRAIN": "donors_public.jsonl",
    **{split: split + "/tasks_public.jsonl" for split in DEV_SPLITS | {"E_CONFIRM2"}},
}
AUDIT_FILES = {
    "COMMON_TRAIN": "common_audit.jsonl",
    "REPLAY": "replay_audit.jsonl",
    "DONOR_TRAIN": "donors_audit.jsonl",
    **{split: split + "/audit_only.jsonl" for split in DEV_SPLITS | {"E_CONFIRM2"}},
}


def stable_id(*parts):
    return digest([EXPERIMENT_ID, *parts])[:32]


def validate_public(task):
    if not isinstance(task, dict) or set(task) not in (PUBLIC_KEYS, PUBLIC_KEYS | {"root_cohort"}):
        raise ValueError("Strict J23 public fields required; audit fields are forbidden")
    if task["phase_id"] != EXPERIMENT_ID or task["split"] not in SPLITS:
        raise ValueError("Unregistered J23 phase or split")
    if any(
        not isinstance(task[key], str) or not re.fullmatch(r"[0-9a-f]{32}", task[key])
        for key in ("task_id", "root_id", "base_instance_id")
    ):
        raise ValueError("Opaque lowercase hex J23 identities required")
    if task["root_id"] != task["base_instance_id"]:
        raise ValueError("J23 root and base instance identity must match")
    if task["template_version"] not in (
        "historical-O0-text-SER-J2-v1",
        "inherit-O0-L11-B1-plus-new-cohort-v1",
    ):
        raise ValueError("Only inherited O0 templates are registered")
    if task["legal_domain"] != [0, 99] or not valid_vector(task["observed"]):
        raise ValueError("Invalid observed values/domain")
    if (
        task["family"] not in ("cross_series", "trend", "duplicate_encoding")
        or task["operation"] not in OPERATIONS
    ):
        raise ValueError("Unregistered public family/operation")
    h, b = task["H_original"], task["b_original"]
    if not isinstance(h, list) or not isinstance(b, list) or not h or len(h) != len(b):
        raise ValueError("Invalid relation dimensions")
    if any(not isinstance(row, list) or len(row) != 4 or not any(row) for row in h):
        raise ValueError("Invalid relation row")
    if any(type(v) is not int for row in h for v in row) or any(type(v) is not int for v in b):
        raise ValueError("Relations require exact integers")
    if task["chart_type"] not in ("grouped_bar", "line") or task["interface"] != "SYMBOLIC_FRESH":
        raise ValueError("J23 is SYMBOLIC_FRESH only")
    if task["image_path"] is not None or task["image_sha256"] is not None:
        raise ValueError("J23 tasks must not carry images")
    if "root_cohort" in task and (
        task["split"] != "E_CONFIRM2"
        or task["root_cohort"] not in ("CORE32", "EXTRA96", "trend", "duplicate_encoding")
    ):
        raise ValueError("Invalid confirmation cohort")
    return task


def public_verifier(task, values):
    validate_public(task)
    return bool(
        valid_vector(values)
        and sum(a != b for a, b in zip(values, task["observed"], strict=True)) == 1
        and all(
            sum(a * x for a, x in zip(row, values, strict=True)) == rhs
            for row, rhs in zip(task["H_original"], task["b_original"], strict=True)
        )
    )


def solve_public(task):
    validate_public(task)
    # Enumerate the entire public domain; no generator truth or audit is consulted.
    answers = []
    for j in range(4):
        for value in range(100):
            if value == task["observed"][j]:
                continue
            candidate = list(task["observed"])
            candidate[j] = value
            if all(
                sum(a * x for a, x in zip(row, candidate, strict=True)) == rhs
                for row, rhs in zip(task["H_original"], task["b_original"], strict=True)
            ):
                answers.append(candidate)
    return answers


def public_prompt(task):
    validate_public(task)
    cue = (
        "b - a = c - b = d - c"
        if task["family"] == "trend"
        else "\n".join(
            format_linear(row, rhs)
            for row, rhs in zip(task["H_original"], task["b_original"], strict=True)
        )
    )
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_TEMPLATE.format(
            observed_json=json.dumps(task["observed"], separators=(",", ":")),
            cue_text=cue,
            operation_expression=OPERATIONS[task["operation"]],
        ),
    }


def compile_case(task):
    prompt = public_prompt(task)
    return {
        "phase_id": EXPERIMENT_ID,
        "task_id": task["task_id"],
        "base_instance_id": task["base_instance_id"],
        "root_id": task["root_id"],
        "case_id": task["task_id"] + ":O0",
        "protocol_id": "O0",
        "split": task["split"],
        "family": task["family"],
        "operation": task["operation"],
        "prompt": prompt,
        "prompt_identity": digest([prompt, None, task["template_version"]]),
        "output_order": [0, 1, 2, 3],
        "H_display": copy.deepcopy(task["H_original"]),
        "b_display": list(task["b_original"]),
        "H_original": copy.deepcopy(task["H_original"]),
        "b_original": list(task["b_original"]),
        "observed_world_public": list(task["observed"]),
        "template_version": task["template_version"],
        "interface": "SYMBOLIC_FRESH",
    }


def _frozen_header(run):
    frozen = read_json(Path(run) / "FROZEN_PLAN.json")
    if (
        frozen.get("status") != "FROZEN"
        or frozen.get("experiment_id", frozen.get("phase_id")) != EXPERIMENT_ID
    ):
        raise ValueError("A frozen SER-J23 plan is required")
    if frozen.get("plan_hash") != digest({k: v for k, v in frozen.items() if k != "plan_hash"}):
        raise ValueError("Frozen J23 plan integrity mismatch")
    if not isinstance(frozen.get("files"), dict):
        raise ValueError("Frozen files whitelist required")
    return frozen


def _bound_path(run, relative, frozen):
    root, rel = Path(run).resolve(), Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.as_posix() != str(relative):
        raise PermissionError("Unsafe frozen path")
    path = root / rel
    if any(
        part.is_symlink() for part in (path, *path.parents) if part != root and root in part.parents
    ):
        raise PermissionError("Symlinked frozen data forbidden")
    if not path.resolve().is_relative_to(root):
        raise PermissionError("Frozen path escaped run")
    entry = frozen["files"].get(rel.as_posix())
    if (
        not isinstance(entry, dict)
        or not path.is_file()
        or file_digest(path) != entry.get("sha256")
    ):
        raise ValueError("Unregistered or modified frozen file: " + str(relative))
    return path


def read_bound(run, relative):
    path = _bound_path(run, relative, _frozen_header(run))
    return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)


def verify_frozen(run, *, role="trainer"):
    """Verify only a role's readable bytes; only auditor may open every file."""
    if role not in {*ROLE_SPLITS, "control"}:
        raise PermissionError("Unknown frozen verification role")
    frozen = _frozen_header(run)
    if role == "auditor":
        relatives = set(frozen["files"])
    elif role == "control":
        relatives = set()  # Header binds all hashes without opening sealed files.
    else:
        relatives = {"manifests/" + PUBLIC_FILES[s] for s in ROLE_SPLITS[role]}
        if role in ("trainer", "diagnostic"):
            relatives |= TRAINING_PATHS
        # An analysis capability releases audit files through RoleDataset.audit.
        relatives &= set(frozen["files"])
    for relative in relatives:
        _bound_path(run, relative, frozen)
    return frozen


class RoleDataset:
    def __init__(self, run, role):
        if role not in ROLE_SPLITS:
            raise PermissionError("Unknown J23 data capability")
        object.__setattr__(self, "run", Path(run).resolve())
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "_plan_hash", _frozen_header(run)["plan_hash"])

    def __setattr__(self, name, value):
        if name in ("run", "role", "_plan_hash") and hasattr(self, name):
            raise AttributeError("Role capability cannot change")
        object.__setattr__(self, name, value)

    @property
    def frozen(self):
        frozen = _frozen_header(self.run)
        if frozen["plan_hash"] != self._plan_hash:
            raise ValueError("Role capability is bound to another frozen plan")
        return copy.deepcopy(frozen)

    def _read(self, relative):
        # Even the internal loader enforces path-specific capabilities. Merely
        # knowing a file name must not let a trainer reach sealed confirmation.
        public = {PUBLIC_FILES[s] for s in ROLE_SPLITS[self.role]}
        training = {"donor_targets.jsonl", *[f"schedule_{s}.jsonl" for s in SCHEDULE_SEEDS]}
        allowed = public | (
            training if self.role in ("trainer", "diagnostic", "auditor", "analysis") else set()
        )
        if self.role in ("analysis", "auditor"):
            allowed |= set(AUDIT_FILES.values())
        if relative not in allowed:
            raise PermissionError("Role cannot read this manifest path")
        if relative == AUDIT_FILES["E_CONFIRM2"] and self.role != "auditor":
            release = read_json(self.run / "RELEASE_RECEIPT.json")
            if (
                release.get("plan_hash") != self._plan_hash
                or release.get("all_registered_models_terminal") is not True
                or release.get("all_confirmation_requests_complete") is not True
            ):
                raise PermissionError("Confirmation outcomes are still sealed")
        path = _bound_path(self.run, "manifests/" + relative, self.frozen)
        return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)

    def public(self, split):
        if split not in ROLE_SPLITS[self.role]:
            raise PermissionError(f"{self.role} cannot read {split}")
        rows = self._read(PUBLIC_FILES[split])
        if split in ("COMMON_TRAIN", "REPLAY"):
            rows = [row["task"] for row in rows]
        if len({row["task_id"] for row in rows}) != len(rows) or any(
            row["split"] != split for row in rows
        ):
            raise ValueError("Duplicate task or role mismatch")
        return [validate_public(row) for row in rows]

    def audit(self, split):
        if self.role not in ("analysis", "auditor") or split not in SPLITS:
            raise PermissionError("Audit data requires audit capability")
        return self._read(AUDIT_FILES[split])


TRAINING_PATHS = frozenset(
    {
        "machine.json",
        *(
            "manifests/" + name
            for name in (
                "common_targets.jsonl",
                "replay_targets.jsonl",
                "donors_public.jsonl",
                "donor_targets.jsonl",
                *[f"schedule_{seed}.jsonl" for seed in SCHEDULE_SEEDS],
            )
        ),
    }
)


def verify_source_bound(run, *, role="trainer"):
    if role not in ("trainer", "control"):
        raise PermissionError("Unknown training source binding role")
    bound = read_json(Path(run) / "SOURCE_BOUND.json")
    if (
        bound.get("status") != "SOURCE_BOUND"
        or bound.get("phase_id") != EXPERIMENT_ID
        or bound.get("source_hash")
        != digest({k: v for k, v in bound.items() if k != "source_hash"})
        or set(bound.get("files", {})) != TRAINING_PATHS
    ):
        raise ValueError("Verified J23 training-only source binding required")
    if role == "trainer":
        for relative in bound["files"]:
            _bound_path(run, relative, bound)
    return bound


def read_training_bound(run, relative, *, technical=False):
    if relative not in TRAINING_PATHS:
        raise PermissionError("Training source access forbids this path")
    if not technical:
        return read_bound(run, relative)
    path = _bound_path(run, relative, verify_source_bound(run))
    return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)


def training_rows(run, arm, *, technical=False):
    if arm not in ARMS:
        raise ValueError("Unregistered J23 arm")
    result = {}

    def load(filename):
        return read_training_bound(run, "manifests/" + filename, technical=technical)

    for role, filename in (("common", "common_targets.jsonl"), ("replay", "replay_targets.jsonl")):
        for row in load(filename):
            task = row["task"]
            if task["task_id"] in result:
                raise ValueError("Duplicate training identity")
            if canonical_target(json.loads(row["target"])) != row["target"]:
                raise ValueError("Canonical training target required")
            result[task["task_id"]] = dict(
                task_id=task["task_id"],
                prompt=public_prompt(task),
                target=row["target"],
                role=role,
                root_id=task["root_id"],
                weight=1,
                source_task_id=row["source_task_id"],
                source_root_id=row["source_root_id"],
            )
    donor_tasks = load("donors_public.jsonl")
    tasks = {row["task_id"]: validate_public(row) for row in donor_tasks}
    if len(tasks) != 128 or any(row["split"] != "DONOR_TRAIN" for row in tasks.values()):
        raise ValueError("Exactly 128 unique J23 donor tasks required")
    for row in load("donor_targets.jsonl"):
        if row["logical_arm_id"] == arm:
            task = tasks[row["task_id"]]
            if task["task_id"] in result or row["root_id"] != task["root_id"]:
                raise ValueError("Duplicate or mismatched donor identity")
            if canonical_target(json.loads(row["target"])) != row["target"]:
                raise ValueError("Canonical donor target required")
            result[task["task_id"]] = dict(
                task_id=task["task_id"],
                prompt=public_prompt(task),
                target=row["target"],
                role="donor",
                root_id=task["root_id"],
                weight=1,
                source_task_id=row["source_task_id"],
                source_root_id=row["source_root_id"],
            )
    from collections import Counter

    if Counter(row["role"] for row in result.values()) != {
        "common": 182,
        "replay": 64,
        "donor": 32,
    }:
        raise ValueError("Training corpus must have 182 common,64 replay,32 donor tasks")
    return result
