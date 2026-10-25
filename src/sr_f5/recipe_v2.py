"""One registered numerical recipe revision, prepared in a separate run root.

This entry point only registers configuration and runs CPU preparation. It never
submits GPU jobs, modifies v1 scientific evidence, or treats diagnostics as PASS.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path

from sr_f4.common import now, read
from srf5_ref.generator import seed_for

from .prepare import digest, prepare

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
V1_ROOT = PROJECT / "sr_f5_semantic_j_20261025"
V2_ROOT = PROJECT / "sr_f5_semantic_j_recipe_v2_20261026"
V1_REVISION = "ea50c51f0f145a183df360ab82025216d6303aca"
NUMERICS = {
    "recipe_version": 2,
    "model_dtype": "float32",
    "input_float_dtype": "float32",
    "probability_dtype": "float32",
}
EVIDENCE = (
    "CONTINUATION/PROBABILITY_226283/bf16.json",
    "CONTINUATION/PROBABILITY_226283/fp32.json",
    "CONTINUATION/PROBABILITY_CACHE_226283/bf16.json",
    "CONTINUATION/PROBABILITY_CACHE_226283/fp32.json",
)


def derive_config(original):
    """Change only the registered precision and prescribed development seeds."""
    if "numerics" in original or original.get("recipe_version", 1) != 1:
        raise PermissionError("Only the original v1 can receive the single recipe revision")
    if original["pilot"]["max_recipe_versions"] != 2:
        raise PermissionError("The contract permits exactly two development versions")
    config = copy.deepcopy(original)
    config["numerics"] = dict(NUMERICS)
    for split in ("dev", "pilot"):
        config["data"]["seeds"][split] = seed_for(config["master_seed"], "recipe_v2", split)
    config["status"] = "RECIPE_V2_REGISTERED_NOT_GPU_VALIDATED"
    config["experiment_id"] = original["experiment_id"] + "-RECIPE-V2"
    config["deployment"]["root"] = str(V2_ROOT)
    return config


def _frozen_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _read_owned(root, relative):
    path = root / relative
    if path.resolve(strict=True) != path or not path.is_file():
        raise PermissionError("Evidence must be a canonical file in the original project")
    return read(path)


def evidence_and_decision(config):
    """A reviewed explicit decision is required in addition to successful diagnostics."""
    original = _read_owned(V1_ROOT, "CONFIG.json")
    frozen = _read_owned(V1_ROOT, "G3/FROZEN.json")
    if (
        frozen.get("code_revision") != V1_REVISION
        or frozen.get("config_digest") != _frozen_digest(original)
        or frozen.get("status") != "FROZEN_BEFORE_PILOT_OUTPUTS"
    ):
        raise PermissionError("Original frozen scientific identity differs")
    decision = _read_owned(V1_ROOT, "CONTINUATION/RECIPE_V2_DECISION.json")
    if (
        decision.get("decision") != "REGISTER_RECIPE_V2"
        or decision.get("recipe_version") != 2
        or decision.get("source_revision") != V1_REVISION
        or decision.get("engineering_mismatch_found") is not False
        or decision.get("preserve_v1") is not True
        or decision.get("uniform_all_arms") is not True
        or decision.get("new_config_digest") != digest(config)
        or not isinstance(decision.get("reason"), str)
        or not decision["reason"].strip()
    ):
        raise PermissionError("Explicit, reviewed numerical recipe decision is missing or changed")
    evidence = {}
    for relative in EVIDENCE:
        item = _read_owned(V1_ROOT, relative)
        if (
            item.get("status") != "DIAGNOSTIC_COMPLETE"
            or item.get("science_revision") != V1_REVISION
            or item.get("run_id") != "G3-b1-SHUF"
            or item.get("checkpoint_step") != 6
            or item.get("optimizer_updates") != 0
            or item.get("gate_decision") != "NOT_AUTHORIZED_BY_DIAGNOSTIC"
        ):
            raise PermissionError("Incomplete or unrelated numerical diagnosis")
        if "PROBABILITY_CACHE_" in relative:
            trials = item.get("trials", [])
            if (
                item.get("kind") != "FORCED_TOKEN_REPLAY_ONLY"
                or item.get("new_scientific_samples") != 0
                or item.get("token_prefix_length") != 163
                or [t.get("name") for t in trials] != ["G8", "single_row3"]
                or [len(t.get("rows", [])) for t in trials] != [8, 1]
                or any(t.get("status") != "REPLAY_COMPLETE" for t in trials)
                or any(
                    len(t.get("raw_logits_score_checks", [])) != 163
                    or not all(
                        check.get("raw_logits_scores_equal") is True
                        for check in t["raw_logits_score_checks"]
                    )
                    for t in trials
                )
            ):
                raise PermissionError("Cached replay evidence is incomplete or transformed")
        elif item.get("new_samples") != 0 or len(item.get("rows", [])) != 8:
            raise PermissionError("Same-token precision diagnosis is incomplete")
        evidence[relative] = digest(item)
    return original, decision, evidence


def _create_once(path, value):
    """Publish a complete file without ever replacing an existing registration."""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def register_and_prepare(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The new immutable deployment needs a full committed revision")
    if V1_ROOT.resolve(strict=True) != V1_ROOT or V2_ROOT.resolve() != V2_ROOT:
        raise PermissionError("Canonical fixed project roots are required")
    if V1_ROOT.parent != PROJECT or V2_ROOT.parent != PROJECT or V1_ROOT == V2_ROOT:
        raise PermissionError("Recipe roots must be separate owned project children")
    continuation = V1_ROOT / "CONTINUATION"
    if continuation.resolve(strict=True) != continuation:
        raise PermissionError("Continuation directory must not be a symlink")
    lock_path = continuation / "RECIPE_V2.lock"
    if lock_path.is_symlink():
        raise PermissionError("Registration lock must not be a symlink")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = derive_config(_read_owned(V1_ROOT, "CONFIG.json"))
        original, decision, evidence = evidence_and_decision(config)
        registration_path = continuation / "RECIPE_V2.json"
        config_path = continuation / "RECIPE_V2_CONFIG.json"
        identity = dict(
            recipe_version=2,
            original_root=str(V1_ROOT),
            new_root=str(V2_ROOT),
            original_science_revision=V1_REVISION,
            new_source_revision=revision,
            original_config_digest=digest(original),
            new_config_digest=digest(config),
            decision_digest=digest(decision),
            evidence=evidence,
            reason=decision["reason"],
            numerics=dict(NUMERICS),
            changed_development_seeds={k: config["data"]["seeds"][k] for k in ("dev", "pilot")},
            all_arms_require_recheck=True,
            v1_evidence_preserved=True,
            failed_v2_action="STOP_FEASIBILITY_NO_FORMAL_MATRIX_NO_RECIPE_V3",
        )
        if registration_path.exists():
            registration = _read_owned(V1_ROOT, "CONTINUATION/RECIPE_V2.json")
            if (
                registration.get("identity") != identity
                or _read_owned(V1_ROOT, "CONTINUATION/RECIPE_V2_CONFIG.json") != config
            ):
                raise PermissionError("Existing revision registration differs; no third recipe")
            # Only a completed exact preparation can be reused. Unknown/partial
            # preparation is preserved for explicit diagnosis, never retried here.
            ready = V2_ROOT / "CPU_READY.json"
            if not ready.exists():
                raise RuntimeError("Registered preparation is incomplete or unknown; diagnose it")
            if _read_owned(V2_ROOT, "CONFIG.json") != config:
                raise PermissionError("Prepared configuration differs from the registration")
            result = _read_owned(V2_ROOT, "CPU_READY.json")
            tests = _read_owned(V2_ROOT, "CPU_TESTS.json")
            if (
                result.get("status") != "CPU_READY_FOR_G2_PREFLIGHT"
                or result.get("code_revision") != revision
                or tests.get("status") != "PASS"
                or tests.get("code_revision") != revision
            ):
                raise PermissionError(
                    "Existing CPU acceptance is incomplete or from another revision"
                )
            return result
        if config_path.exists():
            raise RuntimeError("Partial registration exists; preserve and diagnose it")
        if V2_ROOT.exists():
            entries = list(V2_ROOT.iterdir())
            if any(p.name != "slurm" or p.is_symlink() or not p.is_dir() for p in entries):
                raise PermissionError(
                    "Unregistered new root already contains evidence or configuration"
                )
        _create_once(config_path, config)
        _create_once(
            registration_path,
            dict(
                status="REGISTERED_BEFORE_CPU_PREPARATION", registered_at=now(), identity=identity
            ),
        )
        return prepare(V2_ROOT, config_path, revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(register_and_prepare(args.revision)))


if __name__ == "__main__":
    main()
