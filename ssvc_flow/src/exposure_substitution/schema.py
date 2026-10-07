"""Public SER tasks and immutable role-specific manifest access.

Capabilities prevent accidental role mixing; deployment remains responsible for
filesystem isolation. New splits are validated directly, never disguised as T_train.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..prompts import OPERATIONS, SYSTEM_PROMPT, USER_TEMPLATE
from ..protocol_state_probes.transforms import format_linear

ARMS = {"A_LOCAL_C1": 0, "B_FORWARD_C4": 3, "C_FORWARD_C3": 2}
PARENTS = ("S96", "REP96")
SCHEDULE_SEEDS = (108701, 108702, 108703)
EXPERIMENT_ID = "SER_J2_20261007"
MICROBATCH_SLOTS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10), (11,), (12, 13, 14, 15))
SPLITS = {"COMMON_TRAIN", "DONOR_TRAIN", "REPLAY", "E_DIAG", "E_CONFIRM"}
PUBLIC_KEYS = {
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
ROLE_SPLITS = {
    "trainer": {"COMMON_TRAIN", "DONOR_TRAIN", "REPLAY"},
    "diagnostic": {"COMMON_TRAIN", "DONOR_TRAIN", "REPLAY", "E_DIAG"},
    "confirm_worker": {"E_CONFIRM"},
    "analysis": SPLITS,
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def stable_id(*parts):
    return digest(parts)[:32]


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        )
    )


def valid_vector(values):
    return (
        isinstance(values, list)
        and len(values) == 4
        and all(type(value) is int and 0 <= value <= 99 for value in values)
    )


def canonical_target(values):
    if not valid_vector(values):
        raise ValueError("Target requires four exact integers in [0,99]")
    return json.dumps(values, separators=(",", ":"))


def validate_public(task):
    if not isinstance(task, dict) or set(task) not in (PUBLIC_KEYS, PUBLIC_KEYS | {"root_cohort"}):
        raise ValueError("Strict SER public fields required; audit fields are forbidden")
    if task["split"] not in SPLITS:
        raise ValueError("Unregistered SER role")
    if not all(
        isinstance(task[key], str) and task[key]
        for key in ("task_id", "root_id", "base_instance_id")
    ):
        raise ValueError("Opaque nonempty task/root IDs required")
    if task["root_id"] != task["base_instance_id"]:
        raise ValueError("SER root identity must match base instance")
    if task["template_version"] not in (
        "historical-O0-text-SER-J2-v1",
        "inherit-O0-L11-B1-plus-new-cohort-v1",
    ):
        raise ValueError("Unregistered inherited/SER template")
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
        raise ValueError("SER-J2 is SYMBOLIC_FRESH only")
    if task["image_path"] is not None or task["image_sha256"] is not None:
        raise ValueError("SER tasks must not carry image files")
    if "root_cohort" in task and (
        task["split"] != "E_CONFIRM"
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
    answers = []
    for coordinate in range(4):
        for value in range(100):
            if value == task["observed"][coordinate]:
                continue
            candidate = list(task["observed"])
            candidate[coordinate] = value
            if public_verifier(task, candidate):
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
    if frozen.get("status") != "FROZEN" or frozen.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("A frozen SER-J2 plan is required")
    if frozen.get("plan_hash") != digest({k: v for k, v in frozen.items() if k != "plan_hash"}):
        raise ValueError("Frozen plan integrity mismatch")
    return frozen


def _bound_path(run, relative, frozen):
    root = Path(run).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise PermissionError("Unsafe frozen path")
    path = root / rel
    if any(
        part.is_symlink() for part in (path, *path.parents) if part != root and root in part.parents
    ):
        raise PermissionError("Symlinked frozen data forbidden")
    if not path.resolve().is_relative_to(root):
        raise PermissionError("Frozen path escaped run")
    entry = frozen["files"].get(rel.as_posix())
    if not entry or file_digest(path) != entry["sha256"]:
        raise ValueError("Unregistered or modified frozen file: " + str(relative))
    return path


def verify_frozen(run):
    frozen = _frozen_header(run)
    for relative in frozen["files"]:
        _bound_path(run, relative, frozen)
    return frozen


def read_bound(run, relative):
    path = _bound_path(run, relative, _frozen_header(run))
    return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)


class RoleDataset:
    def __init__(self, run, role):
        if role not in ROLE_SPLITS:
            raise PermissionError("Unknown SER data capability")
        object.__setattr__(self, "run", Path(run).resolve())
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "frozen", _frozen_header(run))

    def __setattr__(self, name, value):
        if name in ("run", "role", "frozen") and hasattr(self, name):
            raise AttributeError("Role capability cannot change")
        object.__setattr__(self, name, value)

    def _read(self, relative):
        path = _bound_path(self.run, "manifests/" + relative, self.frozen)
        return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)

    def public(self, split):
        if split not in ROLE_SPLITS[self.role]:
            raise PermissionError(f"{self.role} cannot read {split}")
        if split in ("COMMON_TRAIN", "REPLAY"):
            filename = "common_targets.jsonl" if split == "COMMON_TRAIN" else "replay_targets.jsonl"
            rows = [row["task"] for row in self._read(filename)]
        else:
            filename = (
                "donors_public.jsonl" if split == "DONOR_TRAIN" else split + "/tasks_public.jsonl"
            )
            rows = self._read(filename)
        if len({row["task_id"] for row in rows}) != len(rows) or any(
            row["split"] != split for row in rows
        ):
            raise ValueError("Duplicate task or role mismatch")
        for row in rows:
            validate_public(row)
        return rows

    def audit(self, split):
        if self.role != "analysis":
            raise PermissionError("Audit data requires analysis capability")
        if split == "E_CONFIRM":
            release = read_json(self.run / "FINAL_RELEASE.json")
            if (
                release.get("plan_hash") != self.frozen["plan_hash"]
                or release.get("all_registered_models_terminal") is not True
            ):
                raise PermissionError("Confirmation outcomes are still sealed")
        files = {
            "COMMON_TRAIN": "common_audit.jsonl",
            "DONOR_TRAIN": "donors_audit.jsonl",
            "REPLAY": "replay_audit.jsonl",
            "E_DIAG": "E_DIAG/audit_only.jsonl",
            "E_CONFIRM": "E_CONFIRM/audit_only.jsonl",
        }
        if split not in files:
            raise PermissionError("Unknown audit split")
        return self._read(files[split])


def training_rows(run, arm):
    if arm not in ARMS:
        raise ValueError("Unregistered arm")
    data = RoleDataset(run, "trainer")
    result = {}
    for role, filename in (("common", "common_targets.jsonl"), ("replay", "replay_targets.jsonl")):
        for row in data._read(filename):
            task = row["task"]
            result[task["task_id"]] = {
                "task_id": task["task_id"],
                "prompt": public_prompt(task),
                "target": row["target"],
                "role": role,
                "root_id": task["root_id"],
                "weight": 1,
            }
    tasks = {task["task_id"]: task for task in data.public("DONOR_TRAIN")}
    for row in data._read("donor_targets.jsonl"):
        if row["arm"] == arm:
            task = tasks[row["task_id"]]
            result[task["task_id"]] = {
                "task_id": task["task_id"],
                "prompt": public_prompt(task),
                "target": row["target"],
                "role": "donor",
                "root_id": task["root_id"],
                "weight": 1,
            }
    if len(result) != 182 + 64 + 32:
        raise ValueError("Training corpus must have 182 common,64 replay,32 donor tasks")
    return result
