"""Read-only, hash-bound adapters for the frozen qualification collection.

This module never collects trajectories. Parameters, paid observations, and
oracle responses have separate loaders so the runner can enforce sealing before
scoring. All legacy seed roles are retained, with a new development-only role.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from src.modeling_qualification.io import canonical_hash, sha256_file

BANK_ROLES = {"fit": list(range(8)), "diagnostic": [8, 9], "evaluation": list(range(10, 14))}
OPERATIONS = ("joint_0", "joint_1", "no_x_off_1")
FRESH_SEEDS = tuple(range(501, 507)) + tuple(range(601, 611))
FILE_FIELDS = ("observations_file", "oracle_file", "counts_file", "rng_file")
STATE_COMPONENTS = (
    "parameters",
    "adam_m",
    "adam_v",
    "adam_step",
    "gradients",
    "gradient_is_none",
    "optimizer_config",
    "numpy_rng",
    "torch_rng",
)


def resolve_member(root, member):
    root, relative = Path(root).resolve(), Path(member)
    if relative.is_absolute():
        raise ValueError("manifest member must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("manifest member escapes input root")
    return path


def parameter_hash(value):
    """Hash exact dtype/shape/bytes; equal norms are irrelevant to identity."""
    value = np.ascontiguousarray(value)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite parameters")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _serializable(value):
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "array_sha256": parameter_hash(value),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return value


def complete_state_fingerprint(components):
    """A partial historical snapshot must never acquire a complete-state hash."""
    missing = [key for key in STATE_COMPONENTS if components.get(key) is None]
    available = {
        key: _serializable(value) for key, value in components.items() if value is not None
    }
    return {
        "complete_state_sha256": None if missing else canonical_hash(available),
        "available_components_sha256": canonical_hash(available),
        "missing_components": missing,
        "status": "INCOMPLETE_HISTORICAL_STATE" if missing else "COMPLETE",
        "dedup_scope": "inference_only",
    }


def audit_aliases(theta, aliases):
    theta = np.asarray(theta)
    flat = theta.reshape(-1, theta.shape[-1])
    aliases = np.asarray(aliases).reshape(-1)
    if not np.issubdtype(aliases.dtype, np.integer) or len(aliases) != len(flat):
        raise ValueError("invalid alias shape/type")
    if np.any(aliases < 0) or np.any(aliases > np.arange(len(flat))):
        raise ValueError("invalid/forward alias reference")
    if not np.array_equal(flat, flat[aliases]):
        raise ValueError("alias parameters differ from canonical output")
    hashes = [parameter_hash(value) for value in flat]
    return {
        "alias_count": int(np.count_nonzero(aliases != np.arange(len(flat)))),
        "unique_parameter_hashes": len(set(hashes)),
        "parameter_hashes": hashes,
        "dedup_scope": "inference_only",
        "optimizer_states_merged": False,
    }


def _freeze(value):
    value = np.asarray(value).copy()
    value.flags.writeable = False
    return value


def _checked_identity(root, trajectory_id):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["bank_roles"] != BANK_ROLES:
        raise ValueError("historical bank roles differ from frozen V2 roles")
    if tuple(manifest["operation_names"]) != OPERATIONS:
        raise ValueError("historical operation order differs")
    if manifest["anchors"] != [8, 24, 40] or manifest["events"] != ["X", "S", "W", "I"]:
        raise ValueError("historical anchor/event semantics differ")
    entries = [entry for entry in manifest["trajectories"] if entry["id"] == trajectory_id]
    if len(entries) != 1:
        raise ValueError("trajectory identity must match exactly once")
    entry = dict(entries[0], original_role=entries[0]["split"], v2_role="legacy_diagnostic")
    return root, manifest, entry


def _checked_artifact(root, entry, field):
    if field not in entry or field not in entry.get("sha256", {}):
        raise ValueError(f"missing manifest binding: {field}")
    path = resolve_member(root, entry[field])
    if sha256_file(path) != entry["sha256"][field]:
        raise ValueError(f"historical artifact hash mismatch: {field}")
    return path


def _load_arrays(path):
    with np.load(path, allow_pickle=False) as handle:
        return MappingProxyType({key: _freeze(handle[key]) for key in handle.files})


@dataclass(frozen=True)
class ParentTrajectory:
    entry: dict
    manifest: dict
    metadata: tuple
    observations: object
    rng: dict

    def anchor_inputs(self, anchor_index, role="fit"):
        if role not in BANK_ROLES:
            raise ValueError("unknown bank role")
        banks = np.asarray(BANK_ROLES[role])
        obs = self.observations
        theta = obs["branch_theta"][anchor_index, banks]
        step = self.manifest["anchors"][anchor_index]
        return {
            "identity": f"{self.entry['id']}:t{step}",
            "anchor_step": step,
            "bank_indices": _freeze(banks),
            "theta_origin": _freeze(obs["theta"][step]),
            "branch_theta": _freeze(theta),
            "d": _freeze(obs["branch_d"][anchor_index, banks]),
            "e": _freeze(theta[:, 1:] - theta[:, :1]),
            "alias": _freeze(obs["branch_alias"][anchor_index, banks]),
            "parameter_hashes": tuple(tuple(parameter_hash(t) for t in row) for row in theta),
            "role": role,
            "v2_role": "legacy_diagnostic",
        }


def load_parent(root, trajectory_id):
    """Load historical service inputs without any measured response arrays.

    Fixed action identities and training logs are retained for the measurement
    service. This whole object is not a finite-observation predictor capability.
    """
    root, manifest, entry = _checked_identity(root, trajectory_id)
    obs = _load_arrays(_checked_artifact(root, entry, "observations_file"))
    audit_aliases(obs["branch_theta"], obs["branch_alias"])
    metadata_path = root / "probe_metadata.json"
    if (
        "probe_identity_sha256" in manifest
        and sha256_file(metadata_path) != manifest["probe_identity_sha256"]
    ):
        raise ValueError("probe metadata hash mismatch")
    metadata = tuple(json.loads(metadata_path.read_text()))
    rng = (
        json.loads(_checked_artifact(root, entry, "rng_file").read_text())
        if "rng_file" in entry
        else {}
    )
    return ParentTrajectory(entry, manifest, metadata, obs, rng)


def load_parent_counts(root, trajectory_id):
    """Measurement-service capability; never pass this whole mapping to a model."""
    root, _, entry = _checked_identity(root, trajectory_id)
    return _load_arrays(_checked_artifact(root, entry, "counts_file"))


def load_parent_oracle(root, trajectory_id):
    """Explicit scoring/development capability; open after prediction sealing."""
    root, _, entry = _checked_identity(root, trajectory_id)
    return _load_arrays(_checked_artifact(root, entry, "oracle_file"))


def audit_seed_collisions(roots):
    """Search actual local trajectory manifests, excluding mere planned seed lists."""
    paths = set()
    excluded_fixture_manifests = set()
    errors = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            errors.append({"path": str(root), "reason": "SCAN_ROOT_MISSING"})
            continue
        for name in ("manifest.json", "collection_manifest.json"):
            for path in root.rglob(name):
                if "fixtures" in path.parts:
                    excluded_fixture_manifests.add(str(path.resolve()))
                elif ".git" not in path.parts:
                    paths.add(path.resolve())
    collisions = []
    used = set()
    for path in sorted(paths):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            errors.append({"path": str(path), "reason": str(error)})
            continue
        rows = value.get("trajectories", []) if isinstance(value, dict) else []
        for entry in rows:
            if not isinstance(entry, dict) or "seed" not in entry:
                continue
            seed = int(entry["seed"])
            used.add(seed)
            if seed in FRESH_SEEDS:
                collisions.append({"seed": seed, "id": entry.get("id"), "manifest": str(path)})
    return {
        "status": "STOP_FOR_REVIEW" if collisions else "SCAN_INCOMPLETE" if errors else "PASS",
        "checked_candidate_seeds": list(FRESH_SEEDS),
        "colliding_seeds": sorted({row["seed"] for row in collisions}),
        "collisions": collisions,
        "observed_trajectory_seeds": sorted(used),
        "manifests_scanned": len(paths),
        "roots": [str(Path(root).resolve()) for root in roots],
        "scope": "local actual trajectory manifests; declared future protocol seeds excluded",
        "excluded_test_fixture_manifest_count": len(excluded_fixture_manifests),
        "scan_errors": errors,
    }


def audit_frozen_predictions(root):
    """Verify every old saved prediction against its original array hashes."""
    root = Path(root)
    path = root / "prediction_freeze.json"
    freeze = json.loads(path.read_text())
    missing, mismatches = [], []
    verified = 0
    for record in freeze["records"]:
        member = resolve_member(root, record["prediction_file"])
        if not member.is_file():
            missing.append({"record": record["record"], "path": str(member)})
            continue
        with np.load(member, allow_pickle=False) as arrays:
            for key in ("raw", "delta"):
                actual = parameter_hash(arrays[key])
                if actual != record[f"{key}_prediction_hash"]:
                    mismatches.append(
                        {
                            "record": record["record"],
                            "field": key,
                            "actual": actual,
                            "expected": record[f"{key}_prediction_hash"],
                        }
                    )
        verified += 1
    selection_hash = sha256_file(root / "selection.json")
    if selection_hash != freeze["selection_sha256"]:
        mismatches.append(
            {
                "field": "selection_sha256",
                "actual": selection_hash,
                "expected": freeze["selection_sha256"],
            }
        )
    return {
        "status": "HASH_MISMATCH" if mismatches else "MISSING_INPUTS" if missing else "PASS",
        "root": str(root),
        "freeze_sha256": sha256_file(path),
        "expected_prediction_files": len(freeze["records"]),
        "verified_prediction_files": verified,
        "missing_inputs": missing,
        "mismatches": mismatches,
        "verification": "original saved raw/delta array dtype, shape and byte hashes",
    }


def audit_parent_collection(
    root, source_root=None, inventory_path=None, archive_paths=(), seed_scan_roots=()
):
    """Enumerate missing originals precisely; no fallback training or mutation."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    missing, mismatches, files, entries = [], [], [], []

    def check(member, expected=None, **context):
        try:
            path = resolve_member(root, member)
        except ValueError as error:
            mismatches.append({"path": str(member), "reason": str(error), **context})
            return False
        if not path.is_file():
            missing.append({"path": str(path), "expected_sha256": expected, **context})
            return False
        actual = sha256_file(path)
        row = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
            "status": "PASS" if expected is None or actual == expected else "HASH_MISMATCH",
            **context,
        }
        files.append(row)
        if row["status"] != "PASS":
            mismatches.append(row)
            return False
        return True

    for filename, key in (
        ("dataset.npz", "dataset_sha256"),
        ("probe_metadata.json", "probe_identity_sha256"),
        ("train_metadata.json", "train_identity_sha256"),
        ("parameter_layout.json", "parameter_layout_sha256"),
    ):
        check(filename, manifest.get(key), field=key)
    config_path = root / "resolved_config.json"
    if check("resolved_config.json"):
        config = json.loads(config_path.read_text())
        actual_config_hash = canonical_hash(config)
        if actual_config_hash != manifest.get("config_sha256"):
            mismatches.append(
                {
                    "field": "config_sha256",
                    "actual": actual_config_hash,
                    "expected": manifest.get("config_sha256"),
                }
            )
    else:
        config = {}
    try:
        _checked_identity(root, manifest["trajectories"][0]["id"])
    except (ValueError, IndexError, KeyError) as error:
        mismatches.append({"field": "collection_schema", "reason": str(error)})
    for entry in manifest["trajectories"]:
        valid = True
        for field in FILE_FIELDS:
            if field not in entry or field not in entry.get("sha256", {}):
                missing.append(
                    {
                        "trajectory_id": entry["id"],
                        "field": field,
                        "reason": "MISSING_MANIFEST_BINDING",
                    }
                )
                valid = False
            else:
                valid = (
                    check(
                        entry[field], entry["sha256"][field], trajectory_id=entry["id"], field=field
                    )
                    and valid
                )
        entries.append(
            {
                "id": entry["id"],
                "seed": entry["seed"],
                "arm": entry["arm"],
                "original_role": entry["split"],
                "v2_role": "legacy_diagnostic",
                "complete": valid,
            }
        )
    if len(entries) != 60:
        missing.append(
            {
                "field": "trajectory_count",
                "expected": 60,
                "actual": len(entries),
                "reason": "EXPECTED_60_LEGACY_TRAJECTORIES",
            }
        )
    if config:
        prof = config["profiles"]["core"]
        expected = {
            (seed, arm, role)
            for role, seeds in prof["seeds_by_role"].items()
            for seed in seeds
            for arm in prof["arms"]
        }
        actual = {(row["seed"], row["arm"], row["original_role"]) for row in entries}
        for seed, arm, role in sorted(expected - actual):
            missing.append(
                {
                    "trajectory_id": f"seed{seed}_{arm}",
                    "original_role": role,
                    "reason": "MISSING_MANIFEST_TRAJECTORY",
                }
            )
        if actual - expected or len(actual) != len(entries):
            mismatches.append(
                {
                    "field": "trajectory_identity_membership",
                    "extra": sorted(actual - expected),
                    "duplicates": len(entries) - len(actual),
                }
            )
    sources = []
    if inventory_path is not None:
        for record in json.loads(Path(inventory_path).read_text()):
            path = Path(source_root) / record["path"]
            actual = sha256_file(path) if path.is_file() else None
            row = dict(
                record,
                actual_sha256=actual,
                actual_bytes=path.stat().st_size if path.is_file() else None,
                status="PASS"
                if actual == record["sha256"] and path.stat().st_size == record["bytes"]
                else "SOURCE_MISMATCH",
            )
            sources.append(row)
            if row["status"] != "PASS":
                mismatches.append(row)
    expected_archive_hash = "c42bf359950de62ab01a22f83fb3ce7b0dfb1fc18ef34719fefcb199551f1ef3"
    archives = [
        {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "bytes": Path(path).stat().st_size,
        }
        for path in archive_paths
        if Path(path).is_file()
    ]
    for row in archives:
        row["matches_parent_attachment"] = row["sha256"] == expected_archive_hash
    generated = root / "generated_dataset/manifest.json"
    if generated.is_file():
        check("generated_dataset/manifest.json")
        for member, record in json.loads(generated.read_text()).get("files", {}).items():
            check(
                str(Path("generated_dataset") / member), record["sha256"], field="generated_dataset"
            )
    status = "HASH_OR_SCHEMA_MISMATCH" if mismatches else "MISSING_INPUTS" if missing else "PASS"
    return {
        "schema_version": "modeling-contrast-parent-audit-v1",
        "status": status,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "expected_trajectories": 60,
        "manifest_trajectories": len(entries),
        "complete_trajectories": sum(row["complete"] for row in entries),
        "trajectories": entries,
        "files": files,
        "missing_inputs": missing,
        "mismatches": mismatches,
        "source_inventory": sources,
        "archives": archives,
        "parent_archive_verified": any(row["matches_parent_attachment"] for row in archives),
        "seed_collision_audit": audit_seed_collisions(seed_scan_roots)
        if seed_scan_roots
        else {"status": "NOT_RUN"},
        "originals_modified": False,
        "historical_trajectories_regenerated": 0,
    }


def audit_historical_state(parent, config):
    """Record inference aliases and optimizer identity without inventing snapshots.

    The parent saves one terminal torch RNG, not checkpoint-indexed torch RNG;
    branch RNG records stop before bank sampling. Thus a full restoration-state
    hash is unavailable even though model/Adam tensors have exact byte hashes.
    """
    obs, rng = parent.observations, parent.rng
    alias = audit_aliases(obs["branch_theta"], obs["branch_alias"])
    rows = []
    for ai, step in enumerate(parent.manifest["anchors"]):
        main = {
            "parameters": obs["theta"][step],
            "optimizer_config": config["optimizer"],
            "numpy_rng": rng.get("main_numpy_rng_states", [None] * 65)[step],
        }
        for destination, source in (
            ("adam_m", "adam_m"),
            ("adam_v", "adam_v"),
            ("adam_step", "adam_step"),
            ("gradients", "checkpoint_grad"),
            ("gradient_is_none", "checkpoint_grad_is_none"),
        ):
            main[destination] = obs[source][step] if source in obs else None
        rows.append(
            {
                "anchor": step,
                "kind": "main",
                "parameter_sha256": parameter_hash(main["parameters"]),
                **complete_state_fingerprint(main),
            }
        )
        for bank in range(14):
            for operation in range(3):
                components = {
                    "parameters": obs["branch_theta"][ai, bank, operation],
                    "optimizer_config": config["optimizer"],
                }
                for destination, source in (
                    ("adam_m", "branch_adam_m"),
                    ("adam_v", "branch_adam_v"),
                    ("adam_step", "branch_adam_step"),
                    ("gradients", "branch_g_clipped"),
                ):
                    components[destination] = (
                        obs[source][ai, bank, operation] if source in obs else None
                    )
                rows.append(
                    {
                        "anchor": step,
                        "bank": bank,
                        "operation": OPERATIONS[operation],
                        "kind": "branch",
                        "parameter_sha256": parameter_hash(components["parameters"]),
                        **complete_state_fingerprint(components),
                    }
                )
    return {
        "id": parent.entry["id"],
        "alias_audit": alias,
        "states": rows,
        "complete_state_fingerprints_available": sum(row["status"] == "COMPLETE" for row in rows),
        "complete_fingerprint_limitation": (
            "No checkpoint-indexed torch RNG; branches lack post-sampling RNG and stored "
            "gradient-None masks. Saved tensors are hashed separately, never treated as "
            "full training-state identity."
        ),
    }


def run_legacy_diagnostics(root, out):
    """Recompute all legacy contrasts and seven nonzero parent-rule ablations.

    All 60 old trajectories are posthoc development. Exact and original n64
    replica 0 are used; no new observation sampling, forward pass or training.
    Each output path is new. Stored predicted differences permit independent
    reconstruction of every diagnostic row and its hash.
    """
    import csv
    import time

    from src.modeling_qualification.evaluation import _predict_one
    from src.modeling_qualification.io import write_json

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions").mkdir()
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "resolved_config.json").read_text())
    start = time.perf_counter()
    ablations = [(8, "per_prompt", "actual_d", "reference_nonzero")]
    ablations += [(b, "per_prompt", "actual_d", f"fit_budget_{b}") for b in (1, 2, 4)]
    ablations += [(8, r, "actual_d", r) for r in ("pooled", "six_group")]
    ablations += [(8, "per_prompt", i, i) for i in ("norm_only", "sgd_proxy")]
    paired, rows, states, input_hashes = [], [], [], []
    for entry in manifest["trajectories"]:
        parent = load_parent(root, entry["id"])
        exact, counts = load_parent_oracle(root, entry["id"]), load_parent_counts(root, entry["id"])
        obs = parent.observations
        groups = np.asarray([row["group"] for row in parent.metadata])
        weights = np.asarray([row["weight"] for row in parent.metadata])
        states.append(audit_historical_state(parent, config))
        predictions = {}
        for ai, step in enumerate(manifest["anchors"]):
            true_branch = exact["branch_p"][ai]
            true_c = true_branch[:, 1:] - true_branch[:, :1]
            for bank in range(14):
                bank_role = next(role for role, banks in BANK_ROLES.items() if bank in banks)
                total = true_branch[bank] - exact["p"][step]
                for ci, target in enumerate(("joint_1_minus_joint_0", "no_x_off_1_minus_joint_0")):
                    e = obs["branch_theta"][ai, bank, ci + 1] - obs["branch_theta"][ai, bank, 0]
                    for group in range(6):
                        mask = groups == group
                        w = weights[mask] / weights[mask].sum()
                        delta = w @ true_c[bank, ci, mask]
                        p0 = w @ true_branch[bank, 0, mask]
                        p1 = w @ true_branch[bank, ci + 1, mask]
                        v0, v1 = float(1 - p0[3]), float(1 - p1[3])
                        paired.append(
                            {
                                "trajectory_id": entry["id"],
                                "seed": entry["seed"],
                                "arm": entry["arm"],
                                "original_role": entry["split"],
                                "v2_role": "legacy_diagnostic",
                                "anchor": step,
                                "bank": bank,
                                "bank_role": bank_role,
                                "contrast": target,
                                "group": group,
                                **{
                                    f"delta_{event}": float(delta[j])
                                    for j, event in enumerate(("X", "S", "W", "I"))
                                },
                                "delta_v": float(-delta[3]),
                                "baseline_v": v0,
                                "candidate_v": v1,
                                "baseline_qX": float(p0[0] / v0) if v0 > 0 else None,
                                "candidate_qX": float(p1[0] / v1) if v1 > 0 else None,
                                "q_denominator_status": "PRESENT"
                                if min(v0, v1) > 0
                                else "UNDEFINED_ZERO_VALID_MASS",
                                "e_norm": float(np.linalg.norm(e)),
                                "inference_alias": bool(np.array_equal(e, np.zeros_like(e))),
                                "T_minus_B_minus_C_maxabs": float(
                                    np.max(np.abs(total[ci + 1] - total[0] - true_c[bank, ci]))
                                ),
                            }
                        )
            inputs = {
                "d": obs["branch_d"][ai, BANK_ROLES["evaluation"]].reshape(-1, 737),
                "g": obs["branch_g"][ai, BANK_ROLES["evaluation"]].reshape(-1, 737),
            }
            truth = true_c[BANK_ROLES["evaluation"]]
            for n in (None, 64):
                p0 = exact["p"][step] if n is None else counts["trajectory_n64"][0, step] / 64
                p1 = true_branch if n is None else counts["branch_n64"][0, ai] / 64
                fit = {
                    "d": obs["branch_d"][ai, BANK_ROLES["fit"]].reshape(-1, 737),
                    "g": obs["branch_g"][ai, BANK_ROLES["fit"]].reshape(-1, 737),
                    "logs": obs["branch_logs"][ai, BANK_ROLES["fit"]].reshape(-1, 11),
                    "p0": p0,
                    "p1": p1[BANK_ROLES["fit"]].reshape(-1, 72, 4),
                }
                for cap in (1, 2, "FULL"):
                    for budget, representation, mode, ablation in ablations:
                        _, pred_t, model, _ = _predict_one(
                            fit,
                            inputs,
                            "B5_WORST_GROUP_RANK",
                            cap,
                            0.01,
                            {},
                            groups,
                            fit_budget=budget,
                            representation=representation,
                            input_mode=mode,
                        )
                        pred_t = pred_t.reshape(4, 3, 72, 4)
                        pred = pred_t[:, 1:] - pred_t[:, :1]
                        key = f"a{step}_n{n}_r{cap}_{ablation}"
                        predictions[key] = pred
                        group_prediction = np.stack(
                            [
                                np.einsum(
                                    "bcpe,p->bce",
                                    pred[:, :, groups == g, :],
                                    weights[groups == g] / weights[groups == g].sum(),
                                )
                                for g in range(6)
                            ],
                            axis=2,
                        )
                        group_truth = np.stack(
                            [
                                np.einsum(
                                    "bcpe,p->bce",
                                    truth[:, :, groups == g, :],
                                    weights[groups == g] / weights[groups == g].sum(),
                                )
                                for g in range(6)
                            ],
                            axis=2,
                        )
                        error = group_prediction - group_truth
                        rows.append(
                            {
                                "trajectory_id": entry["id"],
                                "seed": entry["seed"],
                                "arm": entry["arm"],
                                "original_role": entry["split"],
                                "v2_role": "legacy_diagnostic",
                                "anchor": step,
                                "n": "exact" if n is None else n,
                                "noise_replica": None if n is None else 0,
                                "ablation": ablation,
                                "fit_budget": budget,
                                "representation": representation,
                                "input_mode": mode,
                                "rank_cap": cap,
                                "actual_rank": model["r"],
                                "k": model["k"],
                                "alpha": 0.01,
                                "predicted_difference_norm": float(np.linalg.norm(pred)),
                                "prediction_hash": parameter_hash(pred),
                                "prediction_key": key,
                                "prediction_file": f"predictions/{entry['id']}.npz",
                                "delta_pX_error_ss": float(np.sum(error[..., 0] ** 2)),
                                "delta_pX_truth_ss": float(np.sum(group_truth[..., 0] ** 2)),
                                "delta_v_error_ss": float(np.sum(error[..., 3] ** 2)),
                                "delta_v_truth_ss": float(np.sum(group_truth[..., 3] ** 2)),
                                "full_error_ss": float(np.sum((pred - truth) ** 2)),
                                "full_truth_ss": float(np.sum(truth**2)),
                                "treatment_status": "NO_IDENTIFIED_UPDATE_SUBSPACE"
                                if model["r"] == 0
                                else "NONZERO_RANK_ZERO_CONTRAST"
                                if np.linalg.norm(pred) == 0
                                else "NONZERO_EFFECTIVE_TREATMENT",
                            }
                        )
        prediction_path = out / "predictions" / f"{entry['id']}.npz"
        np.savez_compressed(prediction_path, **predictions)
        input_hashes.append(
            {
                "trajectory_id": entry["id"],
                "parent_sha256": entry["sha256"],
                "prediction_file": str(prediction_path.relative_to(out)),
                "prediction_file_sha256": sha256_file(prediction_path),
            }
        )
    for name, data in (("paired_contrast.csv", paired), ("nondegenerate_ablation.csv", rows)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write_json(out / "state_fingerprint_audit.json", states)
    summary = {
        "status": "PASS",
        "trajectory_count": len(manifest["trajectories"]),
        "paired_group_rows": len(paired),
        "ablation_rows": len(rows),
        "nonzero_rank_rows": sum(row["actual_rank"] > 0 for row in rows),
        "zero_prediction_rows": sum(row["predicted_difference_norm"] == 0 for row in rows),
        "nonzero_ablation_count": 7,
        "rank_caps": [1, 2, "FULL"],
        "alpha": 0.01,
        "measurements": ["exact", "original_n64_replica0"],
        "wall_seconds": time.perf_counter() - start,
        "new_training_updates": 0,
        "new_cpu_forwards": 0,
        "new_observation_samples": 0,
        "source_inputs": input_hashes,
        "scope": "all legacy trajectories, posthoc development and reproduction only",
    }
    write_json(out / "summary.json", summary)
    return summary


def summarize_legacy_targets(root, diagnostics):
    """Recompute separate primary/secondary scores from saved raw predictions."""
    import csv

    root, diagnostics = Path(root), Path(diagnostics)
    source = diagnostics / "nondegenerate_ablation.csv"
    with source.open() as stream:
        records = list(csv.DictReader(stream))
    metadata = json.loads((root / "probe_metadata.json").read_text())
    groups = np.array([row["group"] for row in metadata])
    weights = np.array([row["weight"] for row in metadata])
    results = []
    for tid in sorted({row["trajectory_id"] for row in records}):
        oracle = load_parent_oracle(root, tid)
        with np.load(diagnostics / "predictions" / f"{tid}.npz", allow_pickle=False) as arrays:
            for record in (row for row in records if row["trajectory_id"] == tid):
                prediction = arrays[record["prediction_key"]]
                if parameter_hash(prediction) != record["prediction_hash"]:
                    raise ValueError("saved ablation prediction hash mismatch")
                ai = [8, 24, 40].index(int(record["anchor"]))
                endpoint = oracle["branch_p"][ai, 10:14]
                truth = endpoint[:, 1:] - endpoint[:, :1]
                for ci, target in enumerate(("joint_1_minus_joint_0", "no_x_off_1_minus_joint_0")):
                    p, t = prediction[:, ci], truth[:, ci]
                    gp, gt = [], []
                    for group in range(6):
                        mask = groups == group
                        w = weights[mask] / weights[mask].sum()
                        gp.append((p[:, mask] * w[None, :, None]).sum(1))
                        gt.append((t[:, mask] * w[None, :, None]).sum(1))
                    gp, gt = np.stack(gp, axis=1), np.stack(gt, axis=1)
                    error = gp - gt
                    row = {
                        key: record[key]
                        for key in (
                            "trajectory_id",
                            "seed",
                            "arm",
                            "original_role",
                            "v2_role",
                            "anchor",
                            "n",
                            "noise_replica",
                            "ablation",
                            "fit_budget",
                            "representation",
                            "input_mode",
                            "rank_cap",
                            "actual_rank",
                            "k",
                            "alpha",
                            "prediction_key",
                            "prediction_file",
                        )
                    }
                    row.update(
                        contrast=target,
                        contrast_axis_index=ci,
                        predicted_difference_norm=float(np.linalg.norm(p)),
                        prediction_hash=parameter_hash(p),
                        joint_packet_hash=record["prediction_hash"],
                        full_error_ss=float(np.sum((p - t) ** 2)),
                        full_truth_ss=float(np.sum(t * t)),
                    )
                    for metric, event in (("pX", 0), ("v", 3)):
                        row[f"delta_{metric}_error_ss"] = float(np.sum(error[..., event] ** 2))
                        row[f"delta_{metric}_truth_ss"] = float(np.sum(gt[..., event] ** 2))
                        row[f"delta_{metric}_mae"] = float(np.mean(np.abs(error[..., event])))
                        row[f"delta_{metric}_q95"] = float(
                            np.quantile(np.abs(error[..., event]), 0.95)
                        )
                    results.append(row)
    path = diagnostics / "nondegenerate_ablation_by_target.csv"
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return {
        "status": "PASS",
        "target_rows": len(results),
        "path": str(path),
        "sha256": sha256_file(path),
        "source_table_sha256": sha256_file(source),
    }
