"""Conditional, frozen CPU collection; importing this module never trains."""

from __future__ import annotations

import copy
import json
import math
import re
import time
from pathlib import Path

from .io import (
    RunWriter,
    canonical_hash,
    checked_output_path,
    sha256_file,
    validate_binding,
    verify_run_manifest,
)
from .protocol import ROOT, config_sha256, cpu_environment, resource_gate, validate_config


def build_fresh_parent_config(config: dict) -> dict:
    """Thin parent collector adapter, preserving its reward and Adam contract."""
    derived = validate_config(config)
    from src.modeling_qualification.protocol import load_config as load_parent_config

    parent = load_parent_config(ROOT / "configs/modeling_qualification/protocol.json")
    adapted = copy.deepcopy(parent)
    fresh = config["fresh_cpu"]
    branches = config["branches"]
    if (
        parent["dataset"]["seed"] != fresh["dataset_seed"]
        or parent["toy_model"]["model_initialization_seed"] != fresh["initialization_seed"]
        or parent["toy_model"]["parameter_count"] != 737
        or [item["name"] for item in parent["fork_interventions"]] != branches["operations"]
    ):
        raise ValueError("fresh gate: inherited parent contract differs")
    profile = {
        "steps": fresh["steps"],
        "B": fresh["B"],
        "K": fresh["K"],
        "arms": list(fresh["main_arms"]),
        "anchors": list(branches["anchors"]),
        "seeds_by_role": {
            "fresh_calibration": config["data_roles"]["fresh_calibration_seeds"],
            "fresh_locked_test": config["data_roles"]["fresh_locked_test_seeds"],
        },
        "fit_bank_indices": list(branches["fit_banks"]),
        "diagnostic_bank_indices": list(branches["diagnostic_banks"]),
        "evaluation_bank_indices": list(branches["evaluation_banks"]),
    }
    adapted["profiles"]["contrast_fresh"] = copy.deepcopy(profile)
    adapted["measurement"]["samples_per_prompt_grid"] = [16, 64, 256]
    adapted["measurement"]["noise_replicas"] = config["observation"]["noise_replicas"]
    adapted["measurement"]["noise_seed_root"] = config["observation"]["seed_root"]
    # This is a collector adapter, not a rewritten qualification protocol.
    adapted["contrast_adapter"] = {
        "protocol_version": config["protocol_version"],
        "parent_config_preserved_except_profile_and_measurement": True,
        "derived_budget": derived,
    }
    adapted["budget_derived"]["total_optimizer_steps"] = derived["total_optimizer_updates"]
    return adapted


def _is_appledouble_sidecar(path: Path) -> bool:
    """Recognize macOS metadata by both its sidecar name and AppleDouble magic."""
    if not path.name.startswith("._"):
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x00\x05\x16\x07"


def scan_seed_collisions(existing_run_roots, requested_seeds: set[int]) -> list[dict]:
    """Read local run inventories and actual seed-named raw files; no renumbering."""
    if not existing_run_roots:
        raise ValueError("seed inventory roots must be provided")
    collisions = {}
    for root in existing_run_roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"seed inventory root missing: {root}")
        for path in sorted(root.rglob("*")):
            # Exclude descendants of this inventory root, not a pytest or
            # deployment ancestor which happens to be named "fixtures".
            relative_parts = path.relative_to(root).parts
            if not path.is_file() or "fixtures" in relative_parts or ".git" in relative_parts:
                continue
            if _is_appledouble_sidecar(path):
                continue
            # Locks and planned configs are not evidence of already generated seeds.
            if path.suffix in (".npz", ".jsonl"):
                match = re.search(r"(?:^|_)seed(\d+)(?:_|\.)", path.name)
                if match and int(match.group(1)) in requested_seeds:
                    collisions[(int(match.group(1)), str(path))] = {
                        "seed": int(match.group(1)),
                        "evidence": str(path),
                    }
            if "manifest" not in path.name.lower() or path.suffix != ".json":
                continue
            try:
                manifest = json.loads(path.read_text())
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"seed inventory manifest unreadable: {path}") from error
            entries = manifest.get("trajectories", []) if isinstance(manifest, dict) else []
            for item in entries:
                seed = item.get("seed")
                if seed in requested_seeds:
                    collisions[(seed, str(path))] = {"seed": seed, "evidence": str(path)}
    return list(collisions.values())


def _verify_files(mapping, project_root, name):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"fresh gate: missing {name} file hashes")
    for relative, digest in mapping.items():
        path = Path(relative)
        if not path.is_absolute():
            path = Path(project_root) / path
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"fresh gate: {name} hash changed or missing: {path}")


def _stable_improvement(value):
    if value is True:
        return True
    return (
        isinstance(value, dict)
        and value.get("cluster_count", 0) >= 2
        and value.get("ci975", math.inf) < 0
    )


def validate_fresh_gate(
    config,
    selection_path,
    *,
    existing_run_roots,
    resource_forecast,
    project_root=ROOT,
    completed_collection_root=None,
) -> dict:
    """Revalidate frozen inputs, or exempt one verified completed collection.

    Post-collection validation must first prove the collection's COMPLETE
    manifest, selection binding and every registered byte hash. Only its exact
    registered raw-file paths are exempt from the fresh-seed collision scan.
    Calling the pre-collection form never exempts an existing seed.
    """
    validate_config(config)
    try:
        selection_path = checked_output_path(selection_path, project_root=project_root)
        allowed_stages = {"N3", "N3_server"} if "execution_amendment" in config else {"N3"}
        if (
            selection_path.name != "MODEL_SELECTION_LOCK.json"
            or selection_path.parent.name not in allowed_stages
        ):
            raise ValueError("selection must be the approved N3 MODEL_SELECTION_LOCK.json")
        manifest = verify_run_manifest(selection_path.parent)
        if manifest["status"] != "COMPLETE":
            raise ValueError("N3 run is not complete")
        lock = json.loads(selection_path.read_text())
        digest = lock.get("lock_sha256")
        if digest != canonical_hash(
            {key: value for key, value in lock.items() if key != "lock_sha256"}
        ):
            raise ValueError("selection lock hash changed")
        if lock.get("allow_fresh_cpu") is not True:
            raise ValueError("N3 did not authorize fresh CPU validation")
        for gate in (
            "science_gate",
            "cost_gate",
            "resource_gate",
            "input_identity_gate",
            "data_gate",
        ):
            if lock.get(gate, {}).get("status") != "PASS":
                raise ValueError(f"{gate} did not pass")
        expected_config_sha256 = config_sha256(config)
        if lock.get("protocol_sha256") != expected_config_sha256:
            raise ValueError("protocol byte hash changed")
        selected = lock.get("selected", [])
        if not 1 <= len(selected) <= 2:
            raise ValueError("requires one or two frozen nonzero finite configurations")
        audits = {entry["configuration_id"]: entry for entry in lock.get("candidate_audit", [])}
        for item in selected:
            candidate = audits.get(item.get("configuration_id"), item)
            valid = (
                candidate.get("method") in config["observation"]["methods"]
                and candidate.get("n") in (64, 256)
                and candidate.get("actual_rank", 0) > 0
                and candidate.get("nonalias_eval_units", 0) > 0
                and candidate.get("predicted_difference_norm", 0) > 0
                and 0 <= candidate.get("pX_nrmse", math.inf) < 0.75
                and 0 <= candidate.get("v_nrmse", math.inf) < 0.75
                and _stable_improvement(candidate.get("stable_improvement_C0"))
                and _stable_improvement(candidate.get("stable_improvement_C1"))
                and candidate.get("cost_pass") is True
            )
            if not valid:
                raise ValueError("selected candidate lacks finite scientific/cost eligibility")
        for field in ("fresh_calibration_seeds", "fresh_locked_test_seeds"):
            if lock.get(field) != config["data_roles"][field]:
                raise ValueError("frozen fresh seeds changed")
        binding = lock.get("binding", {})
        validate_binding(binding)
        mappings = {
            "source": lock.get("source_hashes"),
            "data": lock.get("input_hashes"),
            "packet": lock.get("packet_hashes"),
        }
        for name, mapping in mappings.items():
            _verify_files(mapping, project_root, name)
            if binding[name] != canonical_hash(mapping):
                raise ValueError(f"{name} binding changed")
        _verify_files(lock.get("selector_source_hashes"), project_root, "selector source")
        if binding["config"] != expected_config_sha256 or binding["selector"] != canonical_hash(
            selected
        ):
            raise ValueError("config/selector binding changed")
        if manifest["binding"] != binding:
            raise ValueError("N3 run/selector binding mismatch")
        resource = resource_gate(resource_forecast, config)
        if not resource["passed"]:
            raise ValueError("RESOURCE_REVIEW_REQUIRED: " + "; ".join(resource["reasons"]))
        seeds = set(
            config["data_roles"]["fresh_calibration_seeds"]
            + config["data_roles"]["fresh_locked_test_seeds"]
        )
        verified_collection = None
        excluded_raw_paths = set()
        if completed_collection_root is not None:
            collection_root = checked_output_path(
                completed_collection_root, project_root=project_root
            )
            if collection_root.name != "N4":
                raise ValueError("completed collection must be the N4 stage")
            collection = verify_run_manifest(collection_root, binding)
            if collection["status"] != "COMPLETE":
                raise ValueError("completed collection exemption requires COMPLETE status")
            raw_root = collection_root / "raw"
            for relative in collection["outputs"]:
                path = (collection_root / relative).resolve()
                if raw_root in path.parents:
                    excluded_raw_paths.add(path)
            if raw_root / "manifest.json" not in excluded_raw_paths:
                raise ValueError("completed collection has no registered raw manifest")
            verified_collection = {
                "root": str(collection_root),
                "status": "COMPLETE",
                "binding": dict(collection["binding"]),
                "outputs": dict(collection["outputs"]),
                "excluded_raw_paths": sorted(map(str, excluded_raw_paths)),
            }
        collisions = [
            row
            for row in scan_seed_collisions(existing_run_roots, seeds)
            if Path(row["evidence"]).resolve() not in excluded_raw_paths
        ]
        if collisions:
            raise ValueError("STOP_FOR_REVIEW seed collision: " + json.dumps(collisions))
        return {
            "status": "PASS",
            "selection_sha256": sha256_file(selection_path),
            "binding": binding,
            "lock": lock,
            "resource_gate": resource,
            "seed_inventory_roots": [str(Path(path).resolve()) for path in existing_run_roots],
            "seed_collisions": [],
            "verified_collection": verified_collection,
        }
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise ValueError(f"fresh gate: {error}") from error


def run_collect_fresh(
    config,
    selection_path,
    out,
    *,
    binding,
    existing_run_roots,
    resource_forecast,
    collector=None,
    project_root=ROOT,
) -> dict:
    """Run exactly 32 frozen trajectories only after all live gates pass.

    ``collector`` is an injectable test seam. The default imports only the
    existing CPU toy collector after the refusal checks; there is no Qwen,
    download, accelerator, scheduler or online-control entry point.
    """
    gate = validate_fresh_gate(
        config,
        selection_path,
        existing_run_roots=existing_run_roots,
        resource_forecast=resource_forecast,
        project_root=project_root,
    )
    if binding != gate["binding"]:
        raise ValueError("fresh gate: requested run binding differs from frozen selection")
    out = checked_output_path(out, project_root=project_root)
    if out.name != "N4":
        raise ValueError("fresh gate: output must be the N4 stage")
    cpu_environment()
    adapted = build_fresh_parent_config(config)
    if collector is None:
        from src.modeling_qualification.collection import run_collect

        collector = run_collect
    started = time.perf_counter()
    with RunWriter(out, binding, project_root=project_root) as writer:
        writer.write_json(
            "FRESH_GATE_RECEIPT.json", {key: value for key, value in gate.items() if key != "lock"}
        )
        writer.write_json("FROZEN_SELECTION_COPY.json", gate["lock"])
        # The parent collector owns a fresh subdirectory and preserves full raw
        # observations/oracle/counts/RNG files. Training is never used for N0-N3.
        result = collector(adapted, "contrast_fresh", writer.out / "raw")
        expected = validate_config(config)
        actual = {
            "expected_trajectories": result.get("trajectory_count"),
            "total_optimizer_updates": result.get("optimizer_steps"),
            "training_and_branch_actions": result.get("sampled_finite_actions"),
        }
        if any(actual[key] != expected[key] for key in actual):
            raise RuntimeError(f"fresh raw collector budget mismatch: {actual}")
        # A changed source/selection during training invalidates verification.
        if sha256_file(selection_path) != gate["selection_sha256"]:
            raise RuntimeError("selection changed during fresh collection")
        _verify_files(gate["lock"]["source_hashes"], project_root, "source")
        import numpy as np

        for path in sorted((writer.out / "raw").rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as arrays:
                    for key in arrays.files:
                        array = arrays[key]
                        if array.dtype.hasobject or (
                            np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array))
                        ):
                            raise RuntimeError(f"invalid raw array: {path}:{key}")
            writer.register_existing(path.relative_to(writer.out))
        summary = {
            **result,
            "v2_role": "fresh_locked_cpu_validation",
            "selection_sha256": gate["selection_sha256"],
            "wall_seconds_including_validation": time.perf_counter() - started,
            "new_qwen_calls": 0,
            "new_gpu_calls": 0,
            "online_ssvc": False,
        }
        writer.write_json("FRESH_COLLECTION_SUMMARY.json", summary)
    return summary
