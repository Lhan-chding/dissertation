"""V3 protocol validation and explicit development/confirmation access locks."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .io import atomic_json, canonical_hash, sha256_file, source_identity

EVENTS = ("X", "S", "W", "I")
ROLES = ("development", "interval_calibration", "locked_test")


def validate_config(config):
    c = copy.deepcopy(config)
    if c.get("protocol_version") != "modeling-v3-observation-coverage-20260915":
        raise ValueError("unknown V3 protocol version")
    if c.get("events") != list(EVENTS):
        raise ValueError("event order must be X/S/W/I")
    scope = c["scope"]
    if (
        scope["max_concurrent_project_gpus"] != 2
        or scope["online_ssvc"]
        or scope["automatic_training_feedback"]
    ):
        raise ValueError("V3 scope is at most two GPUs with offline observation only")
    if (
        c["coverage"]["center_updates"]
        or c["observation"]["weight_clipping_primary"]
        or c["observation"]["renormalize_primary"]
    ):
        raise ValueError(
            "uncentered realized increments and unclipped unbiased observations required"
        )
    qwen = c["qwen"]
    if (
        qwen["model_id"] != "Qwen/Qwen3.5-9B"
        or qwen["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    ):
        raise ValueError("fixed Qwen3.5-9B revision required")
    if qwen["steps"] != 128 or qwen["B"] != 4 or qwen["K"] != 8 or qwen["max_new_tokens"] != 64:
        raise ValueError("V3 source/action schedule changed")
    roles = qwen["seed_roles"]
    sets = [set(roles[role]) for role in ROLES]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("training seed roles overlap")
    if set(c["cpu"]["interval_calibration_seeds"]) & set(c["cpu"]["locked_test_seeds"]):
        raise ValueError("CPU seed roles overlap")
    if (
        not c["execution"]["cpu_walltime_cap_removed"]
        or not c["execution"]["disk_cap_legacy_removed"]
    ):
        raise ValueError("legacy study budget restriction cannot be inherited")
    return c


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def role_for_seed(config, seed, *, domain="qwen"):
    if type(seed) is not int:
        raise ValueError("integer training seed required")
    if domain == "qwen":
        for role, values in config["qwen"]["seed_roles"].items():
            if seed in values:
                return role
    elif domain == "cpu":
        for role, key in [
            ("interval_calibration", "interval_calibration_seeds"),
            ("locked_test", "locked_test_seeds"),
        ]:
            if seed in config["cpu"][key]:
                return role
        if seed in config["cpu"]["orthogonal_generalization"]["train_rng_seeds"]:
            return "generalization"
    raise ValueError("seed absent from registered role matrix")


def freeze_selection(config, inputs, selected, out):
    """Publish a method choice bound to completed development evidence only."""
    if "primary_hypothesis_tests" in selected:
        raise ValueError("legacy four-CPU tests cannot replace the cross-stage primary family")
    if len(inputs) < 2:
        raise ValueError("completed Q1 and Q2 development evidence required")
    source = source_identity()
    paths = [Path(p).resolve() for p in inputs]
    evidence = {}
    stages = set()

    def check_development(value):
        if isinstance(value, dict):
            if value.get("role") in {
                "locked_test",
                "interval_calibration",
                "generalization",
                "orthogonal_generalization",
                "timing_pilot",
            }:
                raise ValueError("selection evidence must be development only")
            if (
                value.get("pilot") is True
                or value.get("scientific_status") == "PILOT_NOT_CONFIRMATION"
            ):
                raise ValueError("timing pilot cannot serve as completed development evidence")
            for child in value.values():
                check_development(child)
        elif isinstance(value, list):
            for child in value:
                check_development(child)

    for path in paths:
        if not path.is_dir() or not (path / "COMPLETE.json").is_file():
            raise ValueError("completed Q1 and Q2 directory manifests required")
        complete = json.loads((path / "COMPLETE.json").read_text())
        if "files" in complete:
            from .cpu_campaign import _digest, _verify_complete

            manifest = _verify_complete(path)
            if manifest.get("binding", {}).get("config_sha256") != _digest(config):
                raise ValueError("development config binding differs from current protocol")
        else:
            from .io import verify_manifest

            manifest = verify_manifest(path)
        if manifest.get("status") != "COMPLETE":
            raise ValueError("development stage manifest is incomplete")
        check_development(manifest)
        summary = manifest.get("summary", manifest.get("metadata", {}))
        stages.add(summary.get("stage"))
        if summary.get("scientific_status") != "DEVELOPMENT_ONLY":
            raise ValueError("completed Q1/Q2 DEVELOPMENT_ONLY scientific scope required")
        bound_sources = manifest.get("binding", {}).get("source_hashes", {})
        if not bound_sources or any(
            source["files"].get(key) != digest
            for key, digest in bound_sources.items()
            if key.startswith("src/")
        ):
            raise ValueError("development source binding differs from current implementation")
        files = sorted(p for p in path.rglob("*") if p.is_file())
        for p in files:
            if p.suffix == ".json" and ("SUMMARY" in p.name.upper() or p.name == "COMPLETE.json"):
                check_development(json.loads(p.read_text()))
            evidence[str(p)] = sha256_file(p)
    if not {"Q1", "Q2"} <= stages:
        raise ValueError("completed Q1 and Q2 development stages required")
    methods = selected.get("observation_methods", [])
    selectors = selected.get("selection_rules", [])
    if not 1 <= len(methods) <= 2 or not set(methods) <= set(config["observation"]["methods"]) - {
        "KNOWN_COV_ORACLE"
    }:
        raise ValueError("freeze one or two nonoracle observation rules")
    if not 1 <= len(selectors) <= 2 or not set(selectors) <= set(
        config["coverage"]["selection_rules"]
    ):
        raise ValueError("freeze one or two registered selectors")
    for field in ("models", "rank_caps", "alpha", "rho_threshold", "leverage_threshold"):
        if field not in selected:
            raise ValueError(f"frozen selection missing {field}")
    if not selected["models"] or not set(selected["models"]) <= set(config["models"]["baselines"]):
        raise ValueError("frozen models must belong to the preregistered baseline matrix")
    if not set(selected["models"]) - {"DIRECT_MEASURE", "KNOWN_EVENT_SCORE"}:
        raise ValueError("at least one fitted response or ZERO model is required")
    if "CROSSFIT_COV_ZERO_SUM" in methods and "FULL_GLS" in selected["models"]:
        raise ValueError(
            "cross-fit with FULL_GLS requires full-refit covariance; unsupported combination"
        )
    if not selected["rank_caps"] or not set(selected["rank_caps"]) <= set(
        config["coverage"]["rank_caps"]
    ):
        raise ValueError("frozen rank rules must belong to the preregistered rank matrix")
    if selected["alpha"] not in config["models"]["ridge_alpha_grid"]:
        raise ValueError("ridge alpha was not preregistered")
    for field in ("rho_threshold", "leverage_threshold"):
        value = selected[field]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative geometry rejection thresholds required")
    if "primary_comparison_family" in selected:
        from .frozen_comparisons import validate_primary_comparison_family

        validate_primary_comparison_family(config, selected)
    lock = {
        "schema": "ssvc-v3-selection-lock-1",
        "config_sha256": canonical_hash(config),
        "source_sha256": source["sha256"],
        "source_files": source["files"],
        "development_evidence": evidence,
        "selected": copy.deepcopy(selected),
        "selection_hash": canonical_hash(selected),
        "locked_test_opened": False,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "SELECTION_LOCK.json", lock)
    return lock


def verify_selection_lock(config, lock, *, verify_source=True, verify_evidence=True):
    if not isinstance(lock, dict):
        path = Path(lock)
        if path.is_dir():
            path /= "SELECTION_LOCK.json"
        lock = json.loads(path.read_text())
    if "primary_hypothesis_tests" in lock["selected"]:
        raise ValueError("legacy four-CPU tests cannot replace the cross-stage primary family")
    if lock["config_sha256"] != canonical_hash(config) or lock["selection_hash"] != canonical_hash(
        lock["selected"]
    ):
        raise ValueError("selection/config hash mismatch")
    if "primary_comparison_family" in lock["selected"]:
        from .frozen_comparisons import validate_primary_comparison_family

        validate_primary_comparison_family(config, lock["selected"])
    if verify_source and lock["source_sha256"] != source_identity()["sha256"]:
        raise ValueError("source changed after selection freeze; new confirmation lock required")
    if verify_evidence:
        for path, digest in lock["development_evidence"].items():
            if sha256_file(path) != digest:
                raise ValueError("development evidence changed after freeze")
    return lock


class LabelAccessGate:
    """Evaluators log accesses after a prediction file has been frozen.

    A gate does not claim OS isolation. The estimator is given only selected
    calibration labels; query labels remain behind this audited evaluator API.
    """

    def __init__(self, store, *, selected_bank_ids, role, lock_hash):
        if role not in (*ROLES, "generalization"):
            raise ValueError("invalid access role")
        self.store, self.selected = store, frozenset(selected_bank_ids)
        self.role, self.lock_hash = role, lock_hash

    def calibration(self, bank_id, loader):
        if bank_id not in self.selected:
            raise PermissionError("unselected bank semantic label is unavailable")
        self.store.record(
            "calibration/" + str(bank_id), {"bank_id": bank_id, "lock_hash": self.lock_hash}
        )
        return loader(bank_id)

    def evaluation(self, bank_id, loader, *, prediction_path, expected_prediction_hash):
        prediction_path = Path(prediction_path)
        if (
            not prediction_path.is_file()
            or sha256_file(prediction_path) != expected_prediction_hash
        ):
            raise PermissionError("prediction must be frozen before query reference access")
        self.store.record(
            "evaluation/" + str(bank_id),
            {
                "bank_id": bank_id,
                "role": self.role,
                "lock_hash": self.lock_hash,
                "prediction_hash": expected_prediction_hash,
            },
        )
        return loader(bank_id)
