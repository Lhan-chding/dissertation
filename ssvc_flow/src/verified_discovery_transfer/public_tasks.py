"""Public-only compilation, strict verification, and manifest-bound role loading.

These API checks prevent accidental cross-role reads; they are not an OS sandbox
against hostile Python code. Deployment must grant each worker only its role root.
"""

import copy
import json
from pathlib import Path

from ..prompts import OPERATIONS, SYSTEM_PROMPT, USER_TEMPLATE
from ..protocol_state_probes.semantics import parse_four_ints
from ..protocol_state_probes.transforms import format_linear, render, restore, valid_domain
from .config import PROTOCOLS, TEMPLATE_VERSION, digest, file_digest

PUBLIC_KEYS = {
    "task_id",
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
    "teacher_train": {"T_train"},
    "selector": {"V_selection"},
    "teacher_eval": {"E_test", "G_guard"},
    "trainer": {"T_train", "R_replay"},
    "gold": {"T_train", "R_replay"},
    "validation": {"V_selection", "T_train"},
    "final_eval": {"E_test", "G_guard"},
}


def validate_public(task):
    if not isinstance(task, dict) or set(task) != PUBLIC_KEYS:
        raise ValueError("Strict public task keys required; audit fields forbidden")
    if not all(isinstance(task[k], str) and task[k] for k in ("task_id", "base_instance_id")):
        raise ValueError("Opaque task and base IDs required")
    if task["split"] not in {"T_train", "V_selection", "E_test", "G_guard", "R_replay"}:
        raise ValueError("Unregistered split")
    if task["template_version"] != TEMPLATE_VERSION or task["legal_domain"] != [0, 99]:
        raise ValueError("Unregistered template/domain")
    if (
        not valid_domain(task["observed"])
        or task["family"] not in PROTOCOLS
        or task["operation"] not in OPERATIONS
    ):
        raise ValueError("Invalid public values/family/operation")
    H, b = task["H_original"], task["b_original"]
    if not H or len(H) != len(b) or any(len(row) != 4 or not any(row) for row in H):
        raise ValueError("Invalid relation dimensions")
    if any(type(v) is not int for row in H for v in row) or any(type(v) is not int for v in b):
        raise ValueError("Relations must be exact integers")
    if task["chart_type"] not in ("grouped_bar", "line"):
        raise ValueError("Invalid chart type")
    if task["interface"] not in ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH"):
        raise ValueError("Invalid interface")
    if task["interface"] == "IMAGE_CUE_FRESH":
        if task["split"] != "G_guard" or not task["image_path"] or not task["image_sha256"]:
            raise ValueError("Only guard tasks have public images; rendered image required")
    elif task["image_path"] is not None or task["image_sha256"] is not None:
        raise ValueError("Symbolic task must not contain image information")
    return task


def public_verifier(task, canonical):
    validate_public(task)
    return bool(
        valid_domain(canonical)
        and sum(a != b for a, b in zip(canonical, task["observed"], strict=True)) == 1
        and all(
            sum(a * x for a, x in zip(row, canonical, strict=True)) == rhs
            for row, rhs in zip(task["H_original"], task["b_original"], strict=True)
        )
    )


def compile_case(task, protocol="O0"):
    validate_public(task)
    if protocol not in PROTOCOLS[task["family"]]:
        raise ValueError("Protocol not registered for family")
    # Preserve historical trend wording exactly rather than changing its equations.
    cue = (
        "b - a = c - b = d - c"
        if task["family"] == "trend"
        else "\n".join(
            format_linear(row, rhs)
            for row, rhs in zip(task["H_original"], task["b_original"], strict=True)
        )
    )
    user = USER_TEMPLATE.format(
        observed_json=json.dumps(task["observed"], separators=(",", ":")),
        cue_text=cue,
        operation_expression=OPERATIONS[task["operation"]],
    )
    if task["interface"] == "IMAGE_CUE_FRESH":
        user = "The attached chart also shows the true record.\n" + user
    rendered = render(
        {
            "observed": task["observed"],
            "H": task["H_original"],
            "b": task["b_original"],
            "family": task["family"],
            "operation": task["operation"],
            "original_system": SYSTEM_PROMPT,
            "original_user": user,
        },
        protocol,
    )
    prompt = {"system": rendered["system"], "user": rendered["user"]}
    if task["image_path"]:
        prompt["image_path"] = task["image_path"]
    return {
        "task_id": task["task_id"],
        "base_instance_id": task["base_instance_id"],
        "case_id": task["task_id"] + ":" + protocol,
        "protocol_id": protocol,
        "split": task["split"],
        "family": task["family"],
        "operation": task["operation"],
        "prompt": prompt,
        "prompt_identity": digest([prompt, task["image_sha256"], TEMPLATE_VERSION]),
        "output_order": rendered["output_order"],
        "H_display": rendered["H_display"],
        "b_display": rendered["b_display"],
        "H_original": copy.deepcopy(task["H_original"]),
        "b_original": list(task["b_original"]),
        "observed_world_public": list(task["observed"]),
        "template_version": TEMPLATE_VERSION,
        "interface": task["interface"],
    }


def verify_raw(task, protocol, raw):
    case = compile_case(task, protocol)
    emitted = parse_four_ints(raw)
    canonical = restore(emitted, case["output_order"]) if emitted is not None else None
    return {
        "emitted_vector": emitted,
        "canonical_vector": canonical,
        "strict_parse_status": "PASS" if emitted is not None else "FAIL",
        "domain_status": "PASS" if valid_domain(canonical) else "FAIL",
        "public_verifier_pass": public_verifier(task, canonical),
    }


class RoleDataset:
    """Only registered fixed paths can be read, after verifying their frozen digest."""

    def __init__(self, root, role):
        object.__setattr__(self, "root", Path(root).resolve())
        if role not in ROLE_SPLITS:
            raise PermissionError("Unknown data role")
        object.__setattr__(self, "role", role)
        self.manifest = json.loads((self.root / "cohort_manifest.json").read_text())
        expected = self.manifest.get("manifest_digest")
        if digest({k: v for k, v in self.manifest.items() if k != "manifest_digest"}) != expected:
            raise ValueError("Cohort manifest integrity mismatch")
        if self.manifest.get("status") != "FROZEN":
            raise ValueError("Only frozen cohorts can be loaded")

    def __setattr__(self, name, value):
        if name in ("root", "role") and hasattr(self, name):
            raise AttributeError("Role capability cannot be changed after creation")
        object.__setattr__(self, name, value)

    def _read(self, split, kind):
        if split not in ROLE_SPLITS[self.role]:
            raise PermissionError(f"{self.role} cannot read {split}")
        if kind == "audit" and not (self.role == "gold" and split == "T_train"):
            if self.role != "final_eval":
                raise PermissionError("Audit data forbidden to this role")
            release = json.loads((self.root / "FINAL_RELEASE.json").read_text())
            if (
                release.get("cohort_manifest_digest") != self.manifest["manifest_digest"]
                or release.get("all_registered_models_terminal") is not True
            ):
                raise PermissionError("Final outcomes have not been released")
        key = split + "/" + ("tasks_public.jsonl" if kind == "public" else "audit_only.jsonl")
        unresolved = self.root / key
        if any(p.is_symlink() for p in (unresolved, unresolved.parent)):
            raise PermissionError("Symlinked role paths are forbidden")
        path = unresolved.resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError("Role path escaped cohort root")
        entry = self.manifest["files"].get(key)
        if entry is None or file_digest(path) != entry["sha256"]:
            raise ValueError("Unregistered or modified role data")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != entry["rows"] or any(r["split"] != split for r in rows):
            raise ValueError("Role file count/split mismatch")
        if kind == "public":
            for row in rows:
                validate_public(row)
                if row["image_path"]:
                    image = self.root / row["image_path"]
                    if (
                        image.is_symlink()
                        or image.parent.is_symlink()
                        or not image.resolve().is_relative_to(self.root)
                    ):
                        raise PermissionError("Guard image path escaped cohort root")
                    if file_digest(image) != row["image_sha256"]:
                        raise ValueError("Guard image digest mismatch")
        return rows

    def public(self, split):
        return self._read(split, "public")

    def audit(self, split):
        return self._read(split, "audit")

    def replay(self):
        if self.role not in ("trainer", "gold"):
            raise PermissionError("Replay targets are training only")
        path = self.root / "R_replay/verified_targets.jsonl"
        entry = self.manifest["files"]["R_replay/verified_targets.jsonl"]
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Replay digest mismatch")
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
