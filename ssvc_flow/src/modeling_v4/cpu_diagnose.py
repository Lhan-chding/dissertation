"""Short Stage A diagnostics on saved V3 toy states, not another Q3 grid.

Only fixture-sized campaigns may run locally. Input paths are registered and
hashed once; restored Adam forks add missing calibration/query banks without
training a new source trajectory. Exact evaluator responses are explicitly
separated from finite-observation data and never used for bank selection.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from src.modeling_v3.io import atomic_bytes, atomic_json, atomic_npz, canonical_hash, sha256_file

from .error_decomposition import (
    decompose_response,
    exact_toy_geometry,
    fisher_direction_diagnostics,
)

METHODS = ("RAW4", "PRESERVE_XI", "CROSSFIT_COV_ZERO_SUM")
TARGETS = ("joint_1_minus_joint_0", "no_x_off_1_minus_joint_0", "joint_1_minus_no_x_off_1")
PAIRS = ((1, 0), (2, 0), (1, 2))


def _json(path):
    return json.loads(Path(path).read_text())


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(x) for x in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path, value):
    path, value = Path(path), _safe(value)
    if path.exists():
        if path.is_symlink() or _json(path) != value:
            raise ValueError(f"Immutable JSON differs during resume: {path}")
        return
    atomic_json(path, value)


def _npz(path, arrays):
    path = Path(path)
    if path.exists():
        if path.is_symlink():
            raise ValueError("Symlink output forbidden")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                saved[k].dtype != np.asarray(v).dtype
                or not np.array_equal(saved[k], v, equal_nan=True)
                for k, v in arrays.items()
            ):
                raise ValueError(f"Immutable arrays differ during resume: {path}")
        return
    atomic_npz(path, arrays)


def _elapsed(root, started):
    path = Path(root) / "TIMING.json"
    if path.exists():
        return _json(path)["completed_attempt_seconds"]
    value = time.perf_counter() - started
    _write(
        path,
        {
            "completed_attempt_seconds": value,
            "scope": "This attempt; interruptions before timing receipt are not included",
        },
    )
    return value


def _bytes(path, value):
    path = Path(path)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != value:
            raise ValueError(f"Immutable output differs during resume: {path}")
        return
    atomic_bytes(path, value)


def _complete(root, summary):
    root = Path(root)
    files = {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "COMPLETE.json" and p.name != ".writer.lock"
    }
    _write(root / "COMPLETE.json", {"status": "COMPLETE", "files": files, "summary": summary})
    return summary


def _verify(root):
    root = Path(root)
    body = _json(root / "COMPLETE.json")
    if body.get("status") != "COMPLETE":
        raise ValueError("Incomplete original or output")
    observed = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "COMPLETE.json" and p.name != ".writer.lock"
    }
    if observed != set(body["files"]):
        raise ValueError("Immutable artifact inventory mismatch")
    for relative, digest in body["files"].items():
        path = root / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Unsafe complete-manifest path")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Original/output hash mismatch: {path}")
    return body


def _checked(root, relative, manifest):
    path = Path(root) / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.is_symlink()
        or not path.resolve().is_relative_to(Path(root).resolve())
    ):
        raise ValueError("Unsafe selected original path")
    expected = manifest["files"].get(str(relative))
    if expected is None or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Original hash mismatch or unregistered input: {path}")
    return path


def _settings(config, fixture):
    if fixture:
        c = config.get("cpu_fixture")
        if (
            not isinstance(c, dict)
            or not c.get("fit_banks")
            or max(c["fit_banks"]) > 3
            or min(c["fit_banks"]) < 1
            or not 1 <= c.get("heldout_banks", 0) <= 2
            or not 1 <= c.get("probe_count", 0) <= 3
            or not 2 <= c.get("measurement_repeats", 0) <= 3
            or not 4 <= c.get("measurement_draws", 0) <= 16
            or max(c.get("observation_n", [1000])) > 16
            or not 4 <= c.get("fisher_draws", 0) <= 16
        ):
            raise ValueError("Explicit small fixture bounds required")
        return {**c, "selectors": ["BLOCK_PIVOT_QR"], "fixture": True}
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Actual Stage A must run on server CPU under a numeric Slurm job")
    if any(
        os.environ.get(key, "") not in ("", "-1", "NoDevFiles")
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "SLURM_JOB_GPUS")
    ):
        raise RuntimeError("CPU diagnostic must not expose GPU devices")
    if (
        config["cpu"]["fit_banks"] != [8, 24, 48, 96]
        or config["cpu"]["heldout_banks"] != 24
        or config["cpu"]["measurement_repeats"] != 32
    ):
        raise ValueError("Production Stage A matrix changed from the registered short design")
    return {
        "fit_banks": [8, 24, 48, 96],
        "heldout_banks": 24,
        "probe_count": 72,
        "fisher_draws": 128,
        "measurement_draws": 1024,
        "measurement_repeats": 32,
        "observation_n": [256, 1024, 4096],
        "selectors": ["STRATIFIED_RANDOM", "BLOCK_PIVOT_QR"],
        "fixture": False,
    }


def resolve_reuse(reuse):
    """Mapping/JSON uses collection_root, optional q2_root/q1_root and SHA pins."""
    if isinstance(reuse, (str, Path)):
        path = Path(reuse).expanduser().resolve()
        if path.is_file():
            reuse = _json(path)
            reuse = {
                k: str((path.parent / v).resolve())
                if k.endswith("_root") and not Path(v).is_absolute()
                else v
                for k, v in reuse.items()
            }
        elif (path / "dataset/dataset.npz").is_file():
            reuse = {"collection_root": str(path)}
        else:
            binding = _json(path / "BINDING.json")
            collection = path.parent / "Q2_development_collection"
            if not (collection / "COMPLETE.json").is_file():
                raise ValueError("Q2 binding has no resolvable saved collection; supply reuse JSON")
            if binding.get("collection_sha256") != sha256_file(collection / "COMPLETE.json"):
                raise ValueError("Sibling collection does not match Q2 binding hash")
            reuse = {"collection_root": str(collection), "q2_root": str(path)}
            q1 = path.parent / "Q1_observation_full"
            if (q1 / "COMPLETE.json").is_file():
                reuse["q1_root"] = str(q1)
    if not isinstance(reuse, dict) or not reuse.get("collection_root"):
        raise ValueError("reuse mapping requires collection_root")
    allowed = {
        f"{stem}_{suffix}"
        for stem in ("collection", "q2", "q1")
        for suffix in ("root", "complete_sha256")
    }
    if set(reuse) - allowed:
        raise ValueError("Unknown reuse mapping fields")
    result = {}
    for stem in ("collection", "q2", "q1"):
        if reuse.get(f"{stem}_root"):
            root = Path(reuse[f"{stem}_root"]).expanduser().resolve()
            digest = sha256_file(root / "COMPLETE.json")
            if reuse.get(f"{stem}_complete_sha256", digest) != digest:
                raise ValueError(f"{stem} COMPLETE SHA hash mismatch")
            result[f"{stem}_root"] = str(root)
            result[f"{stem}_complete_sha256"] = digest
    if "q2_root" in result:
        q2_root = Path(result["q2_root"])
        manifest = _json(q2_root / "COMPLETE.json")
        binding = _json(_checked(q2_root, "BINDING.json", manifest))
        if binding.get("collection_sha256") != result["collection_complete_sha256"]:
            raise ValueError("Q2 and saved collection bindings do not match")
    return result


def recheck_observation_records(
    q1_root, out, *, n_grid=(256, 1024, 4096), repeats=32, fixture=False
):
    """Reuse the first registered independent repeats; no new packet generation.

    For n!=1024 V3 retained estimates/sufficient statistics, not full draw arrays.
    This function reports that distinction and does not manufacture missing raw.
    """
    if not fixture and (list(n_grid) != [256, 1024, 4096] or repeats != 32):
        raise ValueError("A1 requires the registered 32-repeat, three-n subset")
    if fixture and (repeats > 3 or max(n_grid) > 16):
        raise ValueError("A1 fixture is too large")
    out = Path(out)
    if (out / "COMPLETE.json").is_file():
        return _verify(out)["summary"]
    out.mkdir(parents=True, exist_ok=True)
    if q1_root is None:
        return _complete(
            out,
            {
                "status": "MISSING_INPUTS",
                "reason": "q1_root not supplied",
                "new_packets_generated": 0,
            },
        )
    root = Path(q1_root)
    manifest = _json(root / "COMPLETE.json")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Q1 original lacks completion")
    selected = [
        x
        for x in manifest["summary"]["units"]
        if x["case"] == "IDENTITY_HASH_SELECTED"
        and x["n"] in n_grid
        and x.get("observed_case_inventory", {}).get("identical_policy") is False
    ]
    identities = {(x.get("seed"), x.get("arm"), x["anchor"], x["bank"]) for x in selected}
    if not fixture and (len(identities) != 8 or len(selected) != 8 * 2 * 3):
        raise ValueError("A1 requires eight actual nonalias policies, both proposals and all n")
    errors, rows, original_hashes, repeat_records, raw_records = [], [], {}, 0, 0
    for unit in selected:
        per_method = {method: [] for method in METHODS}
        seeds = set()
        for repeat in range(repeats):
            prefix = f"units/{unit['id']}/repeat{repeat:03d}"
            complete = _checked(root, f"{prefix}/COMPLETE.json", manifest)
            original_hashes[str(complete)] = sha256_file(complete)
            record = _json(complete)
            stat_path = _checked(complete.parent, "statistics.npz", record)
            identity_path = _checked(complete.parent, "sampling_identity.json", record)
            identity = _json(identity_path)
            if identity["seed"] in seeds:
                raise ValueError("A1 repeated RNG identity")
            seeds.add(identity["seed"])
            with np.load(stat_path, allow_pickle=False) as data:
                truth = data["truth"].copy()
                for method in METHODS:
                    estimate = data[f"estimate_{method}"]
                    if estimate.shape != truth.shape or not np.isfinite(estimate).all():
                        raise ValueError("A1 estimator record invalid")
                    per_method[method].append(estimate - truth)
            raw_records += int((complete.parent / "raw_packet.npz").is_file())
            repeat_records += 1
        for method, records in per_method.items():
            residual = np.stack(records)
            errors.append(residual)
            rows.append(
                {
                    key: unit.get(key)
                    for key in ("id", "seed", "arm", "anchor", "bank", "n", "proposal")
                }
                | {
                    "method": method,
                    "repeats": repeats,
                    "bias_by_prompt_event": residual.mean(0).tolist(),
                    "mse_by_prompt_event": (residual**2).mean(0).tolist(),
                    "q95_by_prompt_event": np.quantile(abs(residual), 0.95, axis=0).tolist(),
                    "repeat_variance_by_prompt_event": residual.var(0, ddof=1).tolist(),
                    "pooled_event_mse": (residual**2).mean((0, 1)).tolist(),
                    "pooled_event_q95": np.quantile(abs(residual), 0.95, axis=(0, 1)).tolist(),
                }
            )
    if not errors:
        return _complete(
            out,
            {
                "status": "MISSING_INPUTS",
                "reason": "no qualifying nonalias records",
                "new_packets_generated": 0,
            },
        )
    _npz(out / "OBSERVATION_RECHECK.npz", {"errors": np.stack(errors)})
    _write(
        out / "OBSERVATION_RECHECK.json",
        {
            "rows": rows,
            "source_repeat_hashes": original_hashes,
            "error_axes": ["unit_method", "repeat", "prompt", "event"],
            "variance_scope": "Independent complete packets at each fixed policy/prompt; ddof=1",
            "crossfit_interval": "No iid corrected-draw SE; independent packet repeats retained",
        },
    )
    return _complete(
        out,
        {
            "status": "FIXTURE_REUSED" if fixture else "REUSED_FIXED_POLICY_RECORDS",
            "policy_pairs": len(identities),
            "unit_method_rows": len(rows),
            "repeat_records_read": repeat_records,
            "raw_packet_files_present": raw_records,
            "raw_packet_contents_recomputed": False,
            "new_packets_generated": 0,
        },
    )


def _load_dataset(collection):
    from src.modeling_qualification.toy import ToyDataset

    root = Path(collection) / "dataset"
    with np.load(root / "dataset.npz", allow_pickle=False) as arrays:
        data = {
            k: arrays[k].copy()
            for k in ("train_features", "train_categories", "probe_features", "probe_categories")
        }
    return ToyDataset(
        **data,
        train_metadata=_json(root / "train_metadata.json"),
        probe_metadata=_json(root / "probe_metadata.json"),
        scenes={},
    )


def _extend_origin(dataset, trajectory, anchor, settings, out):
    """Reuse saved candidates; restore the full saved Adam state before each new fork."""
    import torch

    from src.modeling_qualification.toy import (
        flatten_parameters,
        make_model,
        perform_step,
        sample_bank,
    )
    from src.modeling_v3 import cpu_campaign as old

    out = Path(out)
    if (out / "COMPLETE.json").is_file():
        summary = _verify(out)["summary"]
        with np.load(out / "CANDIDATES.npz", allow_pickle=False) as arrays:
            return {k: arrays[k].copy() for k in arrays.files}, summary
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    unit = Path(trajectory)
    identity = _json(unit / "identity.json")
    ai = identity["anchors"].index(anchor)
    old_cal, old_query = (identity["bank_counts"][key] for key in ("calibration", "heldout"))
    n_cal, n_query = max(settings["fit_banks"]), settings["heldout_banks"]
    # Fixtures may keep a prefix; production never drops the saved pool.
    if (old_cal > n_cal or old_query > n_query) and not settings["fixture"]:
        raise ValueError("Requested Stage A bank pool would discard saved banks")
    with np.load(unit / "observations.npz", allow_pickle=False) as arrays:
        theta = arrays["theta"][anchor].copy()
        old_theta = arrays["branch_theta"][ai].copy()
        old_indices = arrays["branch_prompt_indices"][ai].copy()
    trees = _json(unit / "restoration_state_trees.json")
    with np.load(unit / "restoration_state_tensors.npz", allow_pickle=False) as arrays:
        state = old.unpack_state(trees[f"source_step{anchor}"], arrays)
    model = make_model(identity["initialization_seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **{
            k: tuple(v) if k == "betas" else v
            for k, v in identity["parent_optimizer"].items()
            if k in ("lr", "betas", "eps", "weight_decay")
        },
    )
    rng = np.random.default_rng(0)
    old._full_restore(model, optimizer, rng, state)
    if not np.array_equal(flatten_parameters(model), theta):
        raise ValueError("Saved Adam snapshot and source parameters disagree")
    origin_hash = old._state_digest(old._full_snapshot(model, optimizer, rng, anchor))
    with np.load(unit / "source_probabilities.npz", allow_pickle=False) as arrays:
        origin_action_p = arrays["origin_action_p"][ai].copy()
    result = {"origin_theta": theta, "origin_action_p": origin_action_p}
    new_states, new_trees, originals, checks = {}, {}, {}, 0
    counts = {"calibration": (old_cal, n_cal, 0), "heldout": (old_query, n_query, old_cal)}
    for role, (old_count, count, offset) in counts.items():
        existing = min(old_count, count)
        params = np.zeros((count, 3, 737))
        prompt_indices = np.zeros((count, 4), dtype=np.int64)
        params[:existing] = old_theta[offset : offset + existing]
        prompt_indices[:existing] = old_indices[offset : offset + existing]
        with np.load(
            unit
            / ("calibration_action_p.npz" if role == "calibration" else "heldout_action_p.npz"),
            allow_pickle=False,
        ) as arrays:
            probabilities = np.zeros((count, 3, len(dataset.probe_metadata), 16))
            probabilities[:existing] = arrays["action_p"][ai, :existing]
        for bank_index in range(existing, count):
            seed = old._seed(
                "V4_NEW_BANK",
                identity["seed"],
                identity["arm"],
                identity["initialization_seed"],
                anchor,
                role,
                bank_index,
            )
            branch_rng = np.random.default_rng(seed)
            old._full_restore(model, optimizer, rng, state)
            indices = old.bank_prompt_indices(
                dataset.train_metadata, identity["bank_partition"][role], bank_index, branch_rng, 4
            )
            bank = sample_bank(
                model,
                dataset.train_features,
                dataset.train_categories,
                branch_rng,
                B=4,
                K=8,
                prompt_indices=indices,
            )
            prompt_indices[bank_index] = indices
            prefix = f"{role}_bank{bank_index}"
            originals[f"{prefix}_actions"] = bank["actions"]
            originals[f"{prefix}_categories"] = bank["categories"]
            originals[f"{prefix}_old_logp"] = bank["old_logp"]
            originals[f"{prefix}_indices"] = indices
            for oi, (name, lam, policy) in enumerate(old.OPERATIONS):
                old._full_restore(model, optimizer, rng, state)
                try:
                    record = perform_step(
                        model, optimizer, bank, lam=lam, policy=policy, operation_index=oi
                    )
                    params[bank_index, oi] = flatten_parameters(model)
                    probabilities[bank_index, oi] = old._action_probabilities(
                        model, dataset.probe_features
                    )
                    for key in ("d", "g", "g_clipped", "advantages", "rewards", "logs"):
                        originals[f"{prefix}_{name}_{key}"] = record[key]
                    # RNG is unchanged during a deterministic update; bind the shared parent state.
                    candidate = {
                        "model": copy.deepcopy(model.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "parent_state_hash": origin_hash,
                        "bank_seed": seed,
                    }
                    new_trees[f"{prefix}_{name}"] = old._pack_tree(
                        candidate, new_states, f"{prefix}_{oi}"
                    )
                finally:
                    old._full_restore(model, optimizer, rng, state)
                if (
                    old._state_digest(old._full_snapshot(model, optimizer, rng, anchor))
                    != origin_hash
                ):
                    raise RuntimeError(
                        "New candidate contaminated full saved source Adam/RNG state"
                    )
                checks += 1
        result[f"{role}_theta"] = params
        result[f"{role}_action_p"] = probabilities
        result[f"{role}_prompt_indices"] = prompt_indices
    if not set(result["calibration_prompt_indices"].ravel()).isdisjoint(
        result["heldout_prompt_indices"].ravel()
    ):
        raise ValueError("Calibration and query training-prompt pools overlap")
    _npz(out / "CANDIDATES.npz", result)
    _npz(out / "NEW_BANK_ORIGINALS.npz", originals)
    _npz(out / "NEW_FORK_STATES.npz", new_states)
    _write(out / "NEW_FORK_STATE_TREES.json", new_trees)
    summary = {
        "source_training_steps": 0,
        "reused_calibration_banks": min(old_cal, n_cal),
        "reused_heldout_banks": min(old_query, n_query),
        "new_calibration_banks": max(0, n_cal - old_cal),
        "new_heldout_banks": max(0, n_query - old_query),
        "new_candidate_optimizer_steps": checks,
        "wall_seconds": _elapsed(out, started),
        "new_exact_probe_policy_scores": checks * len(dataset.probe_metadata),
        "restoration_checks": checks,
        "parent_state_hash": origin_hash,
        "parent_path": str(unit),
        "source_state_key": f"source_step{anchor}",
        "fixture": settings["fixture"],
    }
    return result, _complete(out, summary)


def _query_exact(pool, features, categories, seed, fisher_draws):
    from src.modeling_v3.cpu_campaign import _seed

    origin = exact_toy_geometry(pool["origin_theta"], features, categories)
    n_query = len(pool["heldout_theta"]) * 3
    params = pool["heldout_theta"]
    updates = np.stack([p[u] - p[b] for p in params for u, b in PAIRS])
    mass = np.einsum(
        "bcpa,pae->bcpe", pool["heldout_action_p"][:, :, : len(features)], np.eye(4)[categories]
    )
    truth = np.stack([p[u] - p[b] for p in mass for u, b in PAIRS])
    base_geometries, base_jacobians, base_jvp, midpoint_jvp, fisher_actions = [], [], [], [], []
    base_cache = {}
    aliases, base_offsets = [], []
    for qi in range(n_query):
        bank, ci = divmod(qi, 3)
        u, b = PAIRS[ci]
        cache_key = (bank, b)
        if cache_key not in base_cache:
            base_cache[cache_key] = exact_toy_geometry(params[bank, b], features, categories)
        base = base_cache[cache_key]
        middle = exact_toy_geometry((params[bank, b] + params[bank, u]) / 2, features, categories)
        base_geometries.append(base)
        base_offsets.append(params[bank, b] - pool["origin_theta"])
        base_jacobians.append(base["jacobian"])
        base_jvp.append(base["jacobian"] @ updates[qi])
        midpoint_jvp.append(middle["jacobian"] @ updates[qi])
        rng = np.random.default_rng(_seed("V4_FISHER", seed, bank, ci))
        fisher_actions.append(
            np.stack([rng.choice(16, fisher_draws, p=p) for p in base["action_probabilities"]])
        )
        aliases.append(bool(np.array_equal(params[bank, b], params[bank, u])))
    return {
        "truth": truth,
        "updates": updates,
        "origin_jvp": np.einsum("ped,qd->qpe", origin["jacobian"], updates),
        "base_jvp": np.asarray(base_jvp),
        "midpoint_jvp": np.asarray(midpoint_jvp),
        "base_jacobians": np.asarray(base_jacobians),
        "base_geometries": base_geometries,
        "base_origin_offsets": np.asarray(base_offsets),
        "fisher_actions": np.asarray(fisher_actions),
        "is_alias": np.asarray(aliases),
        "origin_jacobian": origin["jacobian"],
    }


def _model_specs(config, m, fixture):
    """A small set of declared slices, not a full hyperparameter Cartesian grid."""
    base = [
        {"method": "FULL_DUAL_RIDGE", "alpha": 1e-5},
        {"method": "FULL_DUAL_RIDGE", "alpha": 1e-5, "standardize_blocks": True},
        {"method": "FULL_RBF_RAW", "alpha": 1e-5, "bandwidth_multiplier": 1.0},
        {"method": "LEGACY_QPROJECTED_RBF", "alpha": 1e-5, "bandwidth_multiplier": 1.0},
    ]
    if not fixture:
        base += [
            {"method": "FULL_DUAL_RIDGE", "alpha": 1e-5, "rank_cap": rank}
            for rank in config["models"]["compression_ranks"]
            if rank != "FULL"
        ]
        # Alpha and bandwidth sensitivity are restricted to the m8 diagnostic.
        if m == 8:
            base += [
                {"method": "FULL_DUAL_RIDGE", "alpha": alpha}
                for alpha in config["models"]["ridge_alpha"]
                if alpha != 1e-5
            ]
            base += [
                {"method": "FULL_RBF_RAW", "alpha": 1e-5, "bandwidth_multiplier": factor}
                for factor in config["models"]["rbf_bandwidth_multipliers"]
                if factor != 1.0
            ]
    return base


def fit_focused_gls(updates, responses, queries, covariance, *, ridge=1e-5):
    """Small full-row/event GLS with the same absolute parameter ridge penalty.

    The row-space basis is orthonormal in original parameter coordinates. The
    covariance is never replaced by diagonal candidate blocks. Stabilization is
    explicit because event conservation makes covariance singular.
    """
    e, y, x, cov = map(
        lambda z: np.asarray(z, dtype=float), (updates, responses, queries, covariance)
    )
    if (
        e.ndim != 2
        or y.ndim != 3
        or y.shape[0] != len(e)
        or y.shape[2] != 4
        or x.ndim != 2
        or x.shape[1] != e.shape[1]
        or ridge <= 0
        or cov.shape != (y.shape[1], 4 * len(e), 4 * len(e))
        or not all(np.isfinite(z).all() for z in (e, y, x, cov))
    ):
        raise ValueError("Invalid focused GLS arrays or absolute penalty")
    # This is an exact coordinate reduction, not a rank truncation: retain
    # every right-singular vector from the thin SVD, including weak/null modes.
    # A positive ridge penalizes the unexcited modes to zero. Directions outside
    # this span also have the unique minimum-norm ridge coefficient zero.
    _, input_singular_values, right = np.linalg.svd(e, full_matrices=False)
    q = right.T
    k = q.shape[1]
    diagnostic_k = int(np.count_nonzero(input_singular_values > 0))
    if not np.any(e):
        result = np.zeros((len(x), y.shape[1], 4))
        result[np.linalg.norm(x, axis=1) > 0] = np.nan
        return {
            "prediction": result,
            "metadata": {
                "status": "NO_CALIBRATION_EXCITATION",
                "ridge_lambda": ridge,
                "k": 0,
                "r": 0,
                "rank_label": "FULL",
                "positive_modes_truncated": False,
            },
            "Q": q,
        }
    design = np.kron(e @ q, np.eye(4))
    prediction, floors, ranks, conditions = [], [], [], []
    for prompt in range(y.shape[1]):
        matrix = cov[prompt]
        if not np.allclose(matrix, matrix.T, atol=1e-14, rtol=1e-10):
            raise ValueError("GLS covariance must be symmetric")
        eigen, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        scale = max(float(eigen[-1]), 1e-16)
        if eigen[0] < -scale * 1e-8:
            raise ValueError("GLS covariance must be positive semidefinite")
        floor = max(scale * 1e-6, 1e-16)
        # Solve the regularized whitened objective by its SVD, without a
        # relative-rank cutoff or squaring its condition number in a normal solve.
        whitener = (vectors / np.sqrt(np.maximum(eigen, floor))).T
        weighted_design = whitener @ design
        weighted_y = whitener @ y[:, prompt].reshape(-1)
        left, singular, right = np.linalg.svd(weighted_design, full_matrices=False)
        beta = right.T @ ((singular / (singular**2 + ridge)) * (left.T @ weighted_y))
        prediction.append((x @ q) @ beta.reshape(k, 4))
        floors.append(floor)
        ranks.append(int(np.sum(eigen > floor)))
        conditions.append(float((singular[0] ** 2 + ridge) / (singular[-1] ** 2 + ridge)))
    return {
        "prediction": np.stack(prediction, axis=1),
        "Q": q,
        "metadata": {
            "status": "DIAGNOSTIC_ONLY",
            "k": diagnostic_k,
            "r": diagnostic_k,
            "rank_label": "FULL",
            "rank_diagnostic": "STRICTLY_POSITIVE_RETURNED_INPUT_SINGULAR_VALUES",
            "input_singular_values": input_singular_values.tolist(),
            "fit_basis_columns": k,
            "positive_modes_truncated": False,
            "rank_threshold_scope": "NO_INPUT_OR_WHITENED_MODE_CUTOFF",
            "linear_solver": "SVD_RIDGE_FILTER_ALL_WHITENED_MODES",
            "ridge_lambda": ridge,
            "ridge_parameterization": "ABSOLUTE_PARAMETER_PENALTY",
            "covariance_scope": "FULL_CROSS_CALIBRATION_EVENT_BLOCK",
            "covariance_eigenvalue_floors": floors,
            "covariance_ranks": ranks,
            "regularized_normal_condition_numbers": conditions,
            "stabilization": "max(max_eigenvalue*1e-6,1e-16)",
            "query_responses_used": False,
        },
    }


def _finite_packet(pool, categories, indices, n, seed, *, role="calibration", all_targets=False):
    """Cached finite-action scoring, all candidate rows share origin draws.

    This is actual categorical sampling from saved policies, not model-forward
    timing. Oracle covariance is evaluator-only and kept separate from empirical
    covariance and the observed response means.
    """
    from src.modeling_v3.cpu_campaign import _seed

    p = len(categories)
    origin = pool["origin_action_p"][:p]
    probs = pool[role + "_action_p"][indices, :, :p]
    pairs = PAIRS if all_targets else PAIRS[:2]
    weights = np.stack([(bank[u] - bank[b]) / origin for bank in probs for u, b in pairs])
    rng = np.random.default_rng(_seed("V4_FINITE_PACKET", seed, role, n))
    actions = np.stack([rng.choice(16, n, p=row) for row in origin])
    indicator = np.eye(4)[categories]
    table = weights[..., None] * indicator[None]
    raw, corrected, crossfit, fold_coefficients, finite_cov, oracle_cov = [], [], [], [], [], []
    b = np.array([0.0, 0.5, 0.5, 0.0])
    for prompt in range(p):
        draws = table[:, prompt, actions[prompt], :].transpose(1, 0, 2)
        fixed = draws - draws.sum(-1, keepdims=True) * b
        exact_table = table[:, prompt] - table[:, prompt].sum(-1, keepdims=True) * b
        values = exact_table.transpose(1, 0, 2).reshape(16, -1)
        mean = origin[prompt] @ values
        oracle_cov.append(((values.T * origin[prompt]) @ values - np.outer(mean, mean)) / n)
        flat = fixed.reshape(n, -1)
        finite_cov.append(np.cov(flat, rowvar=False, ddof=1) / n)
        if all_targets:
            from src.modeling_v3.covariance_pilot import pilot_coefficient

            half = n // 2
            predictions, coefficients = [], []
            for row in range(len(weights)):
                fold0, fold1 = draws[:half, row], draws[half:, row]
                b0, _ = pilot_coefficient(fold0, shrink=0.0)
                b1, _ = pilot_coefficient(fold1, shrink=0.0)
                # Each coefficient acts on the other fold. No iid SE is claimed.
                c0 = fold0 - fold0.sum(-1, keepdims=True) * b1
                c1 = fold1 - fold1.sum(-1, keepdims=True) * b0
                predictions.append((c0.sum(0) + c1.sum(0)) / n)
                coefficients.append([b0, b1])
            crossfit.append(predictions)
            fold_coefficients.append(coefficients)
        raw.append(draws.mean(0))
        corrected.append(fixed.mean(0))
    return {
        "raw": np.stack(raw, axis=1),
        "preserve_xi": np.stack(corrected, axis=1),
        "finite_covariance": np.asarray(finite_cov),
        "oracle_covariance": np.asarray(oracle_cov),
        "crossfit": np.asarray(crossfit).transpose(1, 0, 2) if all_targets else np.zeros((0, p, 4)),
        "crossfit_fold_coefficients": np.asarray(fold_coefficients),
        "actions": actions,
        "raw_action_contributions": table,
        "draw_count_per_prompt": n,
        "rng_seed": _seed("V4_FINITE_PACKET", seed, role, n),
    }


def _write_parquet(path, records):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    table = pa.Table.from_pylist(records)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    os.close(fd)
    pending = Path(name)
    try:
        pq.write_table(table, pending, compression="zstd")
        if path.exists():
            if path.is_symlink() or sha256_file(path) != sha256_file(pending):
                raise ValueError("Immutable parquet differs during resume")
        else:
            os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _origin_diagnosis(config, data, pool, identity, settings, out):
    from src.modeling_v3.bank_selection import select_banks
    from src.modeling_v3.coverage import coverage_diagnostics

    from .full_kernels import fit_full_response

    out = Path(out)
    if (out / "COMPLETE.json").is_file():
        return _verify(out)["summary"]
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    p = settings["probe_count"]
    features, categories = data.probe_features[:p], data.probe_categories[:p]
    fit_updates = np.stack(
        [
            np.stack((params[1] - params[0], params[2] - params[0]))
            for params in pool["calibration_theta"]
        ]
    )
    fit_mass = np.einsum(
        "bcpa,pae->bcpe", pool["calibration_action_p"][:, :, :p], np.eye(4)[categories]
    )
    fit_truth = np.stack([np.stack((mass[1] - mass[0], mass[2] - mass[0])) for mass in fit_mass])
    exact = _query_exact(
        pool, features, categories, identity["origin_id"], settings["fisher_draws"]
    )
    direct = _finite_packet(
        pool,
        categories,
        np.arange(len(pool["heldout_theta"])),
        settings["measurement_draws"],
        identity["origin_id"],
        role="heldout",
        all_targets=True,
    )
    _npz(out / "FINITE_DIRECT_PACKET.npz", {k: v for k, v in direct.items()})
    _npz(
        out / "EXACT_QUERY_DIAGNOSTICS.npz",
        {
            k: exact[k]
            for k in (
                "truth",
                "updates",
                "origin_jvp",
                "base_jvp",
                "midpoint_jvp",
                "fisher_actions",
                "is_alias",
                "origin_jacobian",
                "base_origin_offsets",
            )
        },
    )
    groups = [str(row["group"]) for row in data.probe_metadata[:p]]
    prompt_ids = [row["prompt_id"] for row in data.probe_metadata[:p]]
    blocks = {
        "first_weight": [0, 704],
        "first_bias": [704, 720],
        "last_weight": [720, 736],
        "last_bias": [736, 737],
    }
    strata = [
        data.train_metadata[int(indices[0])]["group"]
        for indices in pool["calibration_prompt_indices"]
    ]
    records, model_summaries, basis_arrays, selection_audit = [], [], {}, {}
    associations = []
    for selector in settings["selectors"]:
        chosen = select_banks(
            fit_updates, len(fit_updates), selector, strata=strata, seed=2026091600
        )
        order = chosen["selected_indices"]
        selection_audit[selector] = {
            "selected_order": order,
            "audit": chosen["audit"],
            "selection_uses_query_responses": False,
            "subsets_nested": True,
        }
        for m in settings["fit_banks"]:
            indices = np.asarray(order[:m])
            e = fit_updates[indices].reshape(-1, 737)
            y = fit_truth[indices].reshape(-1, p, 4)
            geometry = coverage_diagnostics(e, exact["updates"], alpha=1e-5)
            q = geometry["Q"]
            key = f"{selector}_m{m}"
            basis_arrays[key + "_Q"] = q
            basis_arrays[key + "_indices"] = indices
            residual = geometry["e_perp"]
            exact_fisher, empirical_fisher, bounds = [], [], []
            for qi, base in enumerate(exact["base_geometries"]):
                fisher = fisher_direction_diagnostics(
                    base, residual[qi], actions=exact["fisher_actions"][qi]
                )
                exact_fisher.append(fisher["exact_energy"])
                empirical_fisher.append(fisher["empirical_energy"])
                bounds.append(fisher["exact_local_bound"])
            exact_fisher, empirical_fisher, bounds = map(
                np.asarray, (exact_fisher, empirical_fisher, bounds)
            )
            omitted = np.einsum("qped,qd->qpe", exact["base_jacobians"], residual)
            omitted_rms = np.sqrt(np.mean(omitted**2, axis=(1, 2)))
            for scope, mask in (
                ("all", np.ones(len(omitted), dtype=bool)),
                ("nonalias", ~exact["is_alias"]),
            ):
                for name, values in (
                    ("euclidean_rho", geometry["rho"]),
                    ("exact_fisher", np.sqrt(exact_fisher.mean(1))),
                    ("finite_fisher", np.sqrt(empirical_fisher.mean(1))),
                ):
                    associations.append(
                        {
                            "selector": selector,
                            "m": m,
                            "scope": scope,
                            "diagnostic": name,
                            "response": "RMS_J_BASE_E_PERP_OVER_PROMPT_EVENT",
                            **_association(values[mask], omitted_rms[mask]),
                        }
                    )
            specs = _model_specs(config, m, settings["fixture"])
            predictions = []
            for spec in specs:
                kwargs = {**spec, "blocks": blocks}
                model = fit_full_response(e, y, **kwargs)
                predictions.append(
                    (spec, model["predict"](exact["updates"]), model["metadata"], int(model["r"]))
                )
            # Focused m8/QR slice compares identical finite labels with absolute
            # parameter penalties; covariance truth is a separately named oracle.
            if selector == "BLOCK_PIVOT_QR" and m == min(settings["fit_banks"]):
                packet = _finite_packet(
                    pool, categories, indices, settings["measurement_draws"], identity["origin_id"]
                )
                _npz(out / "FOCUSED_GLS_PACKET.npz", {k: v for k, v in packet.items()})
                for alpha in [1e-5] if settings["fixture"] else config["models"]["ridge_alpha"]:
                    measured = packet["preserve_xi"]
                    ridge_model = fit_full_response(
                        e, measured, method="FULL_DUAL_RIDGE", alpha=alpha
                    )
                    predictions.append(
                        (
                            {
                                "method": "FINITE_FULL_DUAL_RIDGE",
                                "alpha": alpha,
                                "training_information": "FINITE_PRESERVE_XI",
                                "n": settings["measurement_draws"],
                            },
                            ridge_model["predict"](exact["updates"]),
                            ridge_model["metadata"],
                            int(ridge_model["r"]),
                        )
                    )
                    for label, covariance in [
                        ("ORACLE_COV", packet["oracle_covariance"]),
                        ("FINITE_COV", packet["finite_covariance"]),
                    ]:
                        gls = fit_focused_gls(
                            e, measured, exact["updates"], covariance, ridge=alpha
                        )
                        predictions.append(
                            (
                                {
                                    "method": "FINITE_FULL_GLS_" + label,
                                    "alpha": alpha,
                                    "training_information": "FINITE_PRESERVE_XI_" + label,
                                    "n": settings["measurement_draws"],
                                },
                                gls["prediction"],
                                gls["metadata"],
                                gls["metadata"]["r"],
                            )
                        )
            predictions.extend(
                [
                    (
                        {"method": "DIRECT_MEASURE_RAW4", "n": settings["measurement_draws"]},
                        direct["raw"],
                        {"information": "FINITE_SAME_ORIGIN_PROPOSAL_QUERY_MEASUREMENT"},
                        0,
                    ),
                    (
                        {
                            "method": "DIRECT_MEASURE_CROSSFIT_COV_ZERO_SUM",
                            "n": settings["measurement_draws"],
                        },
                        direct["crossfit"],
                        {
                            "information": "FINITE_SAME_ORIGIN_PROPOSAL_QUERY_MEASUREMENT",
                            "iid_corrected_draw_standard_error": "NOT_USED",
                        },
                        0,
                    ),
                    (
                        {
                            "method": "DIRECT_MEASURE_PRESERVE_XI",
                            "n": settings["measurement_draws"],
                        },
                        direct["preserve_xi"],
                        {"information": "FINITE_SAME_ORIGIN_PROPOSAL_QUERY_MEASUREMENT"},
                        0,
                    ),
                    (
                        {"method": "ZERO"},
                        np.zeros_like(exact["truth"]),
                        {"information": "STATISTICAL_BASELINE"},
                        0,
                    ),
                    (
                        {"method": "ORACLE_ORIGIN_JVP"},
                        exact["origin_jvp"],
                        {"information": "EXACT_FULL_ORIGIN_JACOBIAN"},
                        737,
                    ),
                    (
                        {"method": "ORACLE_BASE_JVP"},
                        exact["base_jvp"],
                        {"information": "EXACT_FULL_BANK_BASE_JACOBIAN"},
                        737,
                    ),
                    (
                        {"method": "ORACLE_MIDPOINT_JVP"},
                        exact["midpoint_jvp"],
                        {"information": "EXACT_FULL_MIDPOINT_JACOBIAN"},
                        737,
                    ),
                    (
                        {"method": "KNOWN_EVENT_SCORE"},
                        exact["truth"],
                        {"information": "EXACT_QUERY_EVALUATOR_REFERENCE"},
                        0,
                    ),
                ]
            )
            for spec, prediction, model_metadata, rank in predictions:
                design = {
                    "m": m,
                    "selector": selector,
                    "design_seed": 2026091600,
                    "training_information": "EXACT_CALIBRATION_EVENTS",
                    **spec,
                }
                design_id = canonical_hash(design)[:20]
                split = decompose_response(
                    exact["truth"], exact["base_jacobians"], exact["updates"], q, prediction
                )
                error = prediction - exact["truth"]
                finite = np.isfinite(prediction).all((1, 2))
                rank_fields = _rank_fields(geometry["k"], model_metadata, rank)
                model_summaries.append(
                    {
                        "design_id": design_id,
                        "design": design,
                        "metadata": model_metadata,
                        **rank_fields,
                        "all_queries": len(prediction),
                        "finite_queries": int(finite.sum()),
                        "finite_four_event_mse": float(np.mean(error[finite] ** 2))
                        if finite.any()
                        else None,
                        "all_case_four_event_mse": float(np.mean(error**2))
                        if finite.all()
                        else None,
                        "nonalias_four_event_mse": float(
                            np.mean(error[finite & ~exact["is_alias"]] ** 2)
                        )
                        if np.any(finite & ~exact["is_alias"])
                        else None,
                    }
                )
                for qi in range(len(prediction)):
                    bank, ci = divmod(qi, 3)
                    records.append(
                        {
                            **identity,
                            "design_id": design_id,
                            "method": spec["method"],
                            "design": json.dumps(design, sort_keys=True),
                            "selector": selector,
                            "m": m,
                            **rank_fields,
                            "target": TARGETS[ci],
                            "target_index": ci,
                            "bank_id": f"heldout_{bank}",
                            "prompt_ids": prompt_ids,
                            "groups": groups,
                            "role": "development",
                            "fixture": settings["fixture"],
                            "information_access": model_metadata.get(
                                "information", design["training_information"]
                            ),
                            "information_level": model_metadata.get(
                                "information", design["training_information"]
                            ),
                            "n": spec.get("n"),
                            "prediction": prediction[qi].tolist(),
                            "reference": exact["truth"][qi].tolist(),
                            "reference_se": np.zeros((p, 4)).tolist(),
                            "reference_resolved": True,
                            "reference_kind": "EXACT",
                            "reference_estimator": "EXACT_FINITE_ACTION",
                            "prediction_status": "PREDICTED" if finite[qi] else "UNKNOWN",
                            "is_alias": bool(exact["is_alias"][qi]),
                            "origin_jvp": exact["origin_jvp"][qi].tolist(),
                            "base_origin_offset_norm": float(
                                np.linalg.norm(exact["base_origin_offsets"][qi])
                            ),
                            "base_jvp": exact["base_jvp"][qi].tolist(),
                            "midpoint_jvp": exact["midpoint_jvp"][qi].tolist(),
                            "nonlinearity": split["nonlinearity"][qi].tolist(),
                            "omitted_semantic": split["omitted_semantic"][qi].tolist(),
                            "fit_error": split["fit_error"][qi].tolist(),
                            "component_squared_norms": split["component_squared_norms"][
                                qi
                            ].tolist(),
                            "cross_terms": split["cross_terms"][qi].tolist(),
                            "total_squared_error": split["total_squared_error"][qi].tolist(),
                            "error_decomposition_sign": "REFERENCE_MINUS_PREDICTION",
                            "fisher_exact_energy": exact_fisher[qi].tolist(),
                            "fisher_empirical_energy": empirical_fisher[qi].tolist(),
                            "fisher_exact_local_bound": bounds[qi].tolist(),
                            "fisher_sample_count": settings["fisher_draws"],
                            "diagnostics": {
                                "rho": float(geometry["rho"][qi]),
                                "e_norm": float(geometry["e_norm"][qi]),
                                "e_perp_norm": float(geometry["e_perp_norm"][qi]),
                                "leverage": float(geometry["leverage"][qi]),
                                "fisher_residual": float(np.sqrt(exact_fisher[qi].mean())),
                                "empirical_fisher_residual": float(
                                    np.sqrt(empirical_fisher[qi].mean())
                                ),
                            },
                            "geometry_filters_query": False,
                            "query_labels_used_for_fit": False,
                        }
                    )
    _npz(out / "CALIBRATION_BASES.npz", basis_arrays)
    _write(
        out / "SELECTION_AND_MODELS.json",
        {
            "selections": selection_audit,
            "models": model_summaries,
            "semantic_geometry_associations": associations,
        },
    )
    _write_parquet(out / "QUERY_RESULTS.parquet", records)
    summary = {
        "origin_id": identity["origin_id"],
        "query_rows": len(records),
        "fits_and_evaluator_baselines": len(model_summaries),
        "models": model_summaries,
        "semantic_geometry_associations": associations,
        "wall_seconds": _elapsed(out, started),
        "information_scope": (
            "Exact calibration main; focused finite-label GLS/ridge and direct "
            "finite query baselines; J/Fisher diagnostic only"
        ),
        "finite_measurement_n": settings["measurement_draws"],
        "finite_measurement_proposal": "SHARED_ORIGIN_WITHIN_CALIBRATION_OR_QUERY_PACKET",
        "finite_measurement_packet_independence": (
            "Distinct calibration/query/Fisher RNG namespaces"
        ),
        "sampling_cost": {
            "response_packet_categorical_actions": 2 * p * settings["measurement_draws"],
            "fisher_categorical_actions": len(exact["updates"]) * p * settings["fisher_draws"],
            "model_forward_calls": 0,
            "scoring_kind": "CACHED_FULL_ACTION_TABLE_LOOKUP",
        },
        "new_model_training_sources": 0,
        "query_outcomes_used_for_selection": False,
        "focused_GLS": "MATCHED_ABSOLUTE_PENALTY_FINITE_AND_ORACLE_COVARIANCE_DIAGNOSTIC",
        "scientific_status": "FIXTURE_ONLY" if settings["fixture"] else "DEVELOPMENT_ONLY",
    }
    return _complete(out, summary)


def _combine_parquet(paths, target):
    import pyarrow.parquet as pq

    fd, name = tempfile.mkstemp(prefix=".pending-", dir=target.parent)
    os.close(fd)
    pending = Path(name)
    writer = None
    try:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                if writer is None:
                    writer = pq.ParquetWriter(pending, batch.schema, compression="zstd")
                writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()
    if target.exists():
        if target.is_symlink() or sha256_file(target) != sha256_file(pending):
            raise ValueError("Immutable consolidated parquet differs during resume")
    else:
        os.link(pending, target)
    pending.unlink()


def _run_cpu_diagnose(config, reuse, out, *, fixture=False, resume=False, origin_subset=None):
    """Stage A entrypoint; production is server CPU only, reusable per origin.

    ``reuse`` is a JSON mapping/path or a V3 Q2 root. Mandatory collection_root
    supplies full saved Adam states. Optional q1_root supplies the old independent
    fixed-policy repeats; its absence is an explicit incomplete A1 result.
    """

    settings = _settings(config, fixture)
    registered = resolve_reuse(reuse)
    out = Path(out).expanduser().resolve()
    for key, path in registered.items():
        if key.endswith("_root"):
            root = Path(path)
            if out == root or root in out.parents or out in root.parents:
                raise ValueError("Output and original input roots must not overlap")
    from src.modeling_v3.cpu_campaign import _sources

    # Imported V3/toy numerical dependencies are included; unrelated V4 GPU code is not.
    dependency_names = (
        "cpu_diagnose.py",
        "error_decomposition.py",
        "full_kernels.py",
        "evaluate.py",
    )
    source_hashes = _sources() | {
        "src/modeling_v4/" + name: sha256_file(Path(__file__).with_name(name))
        for name in dependency_names
    }
    binding = {
        "config_sha256": canonical_hash(config),
        "reuse": registered,
        "fixture": fixture,
        "origin_subset": sorted(origin_subset) if origin_subset is not None else None,
        "source_hashes": source_hashes,
    }
    if out.exists():
        if not resume:
            raise FileExistsError(
                "Immutable Stage A output already exists; explicit resume required"
            )
        if _json(out / "BINDING.json") != binding:
            raise ValueError("Stage A resume binding mismatch")
        if (out / "COMPLETE.json").is_file():
            return _verify(out)["summary"]
    else:
        out.mkdir(parents=True)
        _write(out / "BINDING.json", binding)
    started = time.perf_counter()
    collection = Path(registered["collection_root"])
    body = _verify(collection)
    trajectories = body["summary"]["trajectories"]
    if not fixture:
        expected = {(seed, arm, 7001) for seed in (101, 201) for arm in ("X_BASE", "X_VALID")}
        observed = {(int(t["seed"]), t["arm"], int(t["initialization_seed"])) for t in trajectories}
        if (
            body["summary"].get("role") != "development"
            or observed != expected
            or len(trajectories) != 4
        ):
            raise ValueError("Stage A requires the four saved V3 development trajectories")
    elif len(trajectories) != 1:
        raise ValueError("fixture must contain only one saved source trajectory")
    data = _load_dataset(collection)
    origins = []
    for trajectory in trajectories:
        path = collection / trajectory["path"]
        if not path.resolve().is_relative_to(collection) or any(
            token in trajectory["id"] for token in ("/", "\\", "..")
        ):
            raise ValueError("Unsafe saved trajectory identity/path")
        identity = _json(path / "identity.json")
        if not fixture and identity["anchors"] != [8, 24, 40]:
            raise ValueError("Saved production anchors differ from the twelve-origin matrix")
        for anchor in identity["anchors"]:
            name = f"{trajectory['id']}_a{anchor}"
            origins.append(
                (
                    name,
                    path,
                    {key: identity[key] for key in ("seed", "arm", "initialization_seed")}
                    | {"anchor": anchor, "origin_id": name},
                )
            )
    if origin_subset is not None:
        if not set(origin_subset).issubset({x[0] for x in origins}):
            raise ValueError("Unregistered origin subset")
        origins = [x for x in origins if x[0] in origin_subset]
    if not origins:
        raise ValueError("Empty origin selection")
    a1 = recheck_observation_records(
        registered.get("q1_root"),
        out / "A1",
        n_grid=settings["observation_n"],
        repeats=settings["measurement_repeats"],
        fixture=fixture,
    )
    results, receipts, paths = [], [], []
    for name, trajectory, identity in origins:
        origin_root = out / "origins" / name
        pool, acquisition = _extend_origin(
            data, trajectory, identity["anchor"], settings, origin_root / "acquisition"
        )
        report = _origin_diagnosis(
            config, data, pool, identity, settings, origin_root / "diagnosis"
        )
        results.append(report)
        receipts.append(acquisition)
        paths.append(origin_root / "diagnosis/QUERY_RESULTS.parquet")
    _combine_parquet(paths, out / "QUERY_RESULTS.parquet")
    evaluation = _evaluate_parquet(out / "QUERY_RESULTS.parquet")
    for name in ("summary", "risk_coverage", "bootstrap"):
        _write(out / (name.upper() + ".json"), evaluation[name])
    _write(
        out / "EVALUATION_METADATA.json",
        {
            k: v
            for k, v in evaluation.items()
            if k not in ("core_results", "summary", "risk_coverage", "bootstrap")
        },
    )
    summary = {
        "stage": "A",
        "fixture": fixture,
        "status": "FIXTURE_COMPLETE"
        if fixture
        else "DEVELOPMENT_DIAGNOSED"
        if len(origins) == 12 and a1["status"] == "REUSED_FIXED_POLICY_RECORDS"
        else "PARTIAL_STAGE_A",
        "scientific_status": "FIXTURE_ONLY" if fixture else "DEVELOPMENT_ONLY_NOT_CERTIFIED",
        "origins_completed": len(origins),
        "full_twelve_origins": not fixture and len(origins) == 12,
        "A1": a1,
        "A2_A3_complete_for_requested_origins": True,
        "new_source_training_steps": 0,
        "new_candidate_optimizer_steps": sum(x["new_candidate_optimizer_steps"] for x in receipts),
        "restoration_checks": sum(x["restoration_checks"] for x in receipts),
        "query_rows": sum(x["query_rows"] for x in results),
        "fits_and_evaluator_baselines": sum(x["fits_and_evaluator_baselines"] for x in results),
        "wall_seconds": _elapsed(out, started),
        "origin_reports": results,
        "acquisition_receipts": receipts,
        "focused_GLS": "MATCHED_ABSOLUTE_PENALTY_FINITE_AND_ORACLE_COVARIANCE_DIAGNOSTIC",
        "GPU_calls": 0,
        "online_SSVC": "NOT_RUN",
        "local_fixture_is_scientific_evidence": False,
    }
    _write(out / "RUN_SUMMARY.json", summary)
    _bytes(
        out / "CPU_FAILURE_DECOMPOSITION.md",
        (
            "# Stage A CPU exact response diagnosis\n\n"
            f"Status: {summary['scientific_status']}. Origins: {len(origins)}. "
            f"New source steps: 0. New candidate Adam steps: "
            f"{summary['new_candidate_optimizer_steps']}.\n\n"
            "QUERY_RESULTS.parquet has one design/query row containing prompt-by-event vectors. "
            "Signed components are C-J(base)e, J(base)e_perp, and J(base)e_parallel-prediction. "
            "Their sum equals C-prediction; all doubled pairwise inner products are stored. "
            "No component norm subtraction is interpreted as a contribution percentage.\n\n"
            "Fitted models in this diagnostic use EXACT_CALIBRATION_EVENTS; oracle J and Fisher "
            "are evaluator-only. Exact and finite-sample Fisher directional energies are separate. "
            "No rho threshold removes queries. Alias and nonalias flags are retained. "
            "These two source seeds do not supply independent confirmation.\n\n"
            f"A1: {a1['status']}. Focused GLS: m8/QR, finite PRESERVE_XI labels; "
            "n1024 in production; "
            "oracle versus finite joint covariance and matched absolute parameter penalty. "
            "This CPU result is not a gate requiring a method to win before real model bridging.\n"
            + _factual_table(evaluation, results)
        ).encode(),
    )
    return _complete(out, summary)


def _evaluate_parquet(path):
    import pyarrow.parquet as pq

    from .evaluate import evaluate_records

    columns = [
        "method",
        "design_id",
        "seed",
        "origin_id",
        "bank_id",
        "target",
        "prompt_ids",
        "groups",
        "role",
        "prediction",
        "reference",
        "reference_se",
        "reference_resolved",
        "reference_kind",
        "reference_estimator",
        "is_alias",
        "prediction_status",
        "diagnostics",
        "information_level",
        "n",
    ]

    def rows():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()

    return evaluate_records(
        rows(), bootstrap_reps=2000, bootstrap_seed=2026091600, core_sink=lambda row: None
    )


def run_cpu_diagnose(config, reuse, out, *, fixture=False, resume=False, origin_subset=None):
    """Server-only Stage A; fixture is explicitly bounded, resume preserves originals.

    ``reuse`` is a JSON mapping/path or existing Q2 root. See resolve_reuse for
    fields. ``origin_subset`` is an optional list of registered trajectory_aN IDs
    for a timing run; it does not silently reduce banks/prompts for that origin.
    """
    _settings(config, fixture)
    registered = resolve_reuse(reuse)
    out = Path(out).expanduser().resolve()
    for key, value in registered.items():
        if key.endswith("_root"):
            root = Path(value)
            if out == root or out in root.parents or root in out.parents:
                raise ValueError("Output and original input roots must not overlap")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Parent-level lock survives crashes and is outside scientific artifact roots.
    with (out.parent / ("." + out.name + ".writer.lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another writer owns this Stage A output") from exc
        return _run_cpu_diagnose(
            config, registered, out, fixture=fixture, resume=resume, origin_subset=origin_subset
        )


def _association(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "count": len(x),
            "pearson": None,
            "spearman": None,
            "status": "UNDEFINED_EMPTY_OR_CONSTANT",
            "causal_interpretation": False,
        }

    def ranks(a):
        _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[inverse]

    return {
        "count": len(x),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
        "status": "DESCRIPTIVE_WITHIN_ORIGIN",
        "causal_interpretation": False,
    }


def _factual_table(evaluation, results):
    designs = {
        model["design_id"]: model["design"] for result in results for model in result["models"]
    }
    lines = [
        "\n## Fixed primary slice: query errors\n",
        "QR, smallest/largest registered m, alpha=1e-5 and RBF factor=1. "
        "RAW4 statistics average over all four event cells; missing values remain UNKNOWN. "
        "All three targets and group X/v statistics are in SUMMARY.json.\n",
        "| Method | m | rank cap | block scaling | scope | finite/all cells | RMSE | q95 |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    ms = sorted({d["m"] for d in designs.values()})

    def fmt(value):
        return "UNKNOWN" if value is None else f"{value:.8g}"

    for row in evaluation["summary"]:
        design = designs[row["design_id"]]
        if (
            row["target"] != TARGETS[0]
            or row["channel"] != "RAW4"
            or row["scope"] not in ("all", "nonalias")
            or design["selector"] != "BLOCK_PIVOT_QR"
            or design["m"] not in (ms[0], ms[-1])
            or design.get("alpha", 1e-5) != 1e-5
            or design.get("bandwidth_multiplier", 1.0) != 1.0
        ):
            continue
        lines.append(
            f"| {row['method']} | {design['m']} | {design.get('rank_cap', 'FULL')} | "
            f"{design.get('standardize_blocks', False)} | {row['scope']} | "
            f"{row['finite_resolved_count']}/{row['all_count']} | "
            f"{fmt(row['rmse'])} | {fmt(row['q95_absolute'])} |"
        )
    return "\n".join(lines) + "\n"


def _rank_fields(geometry_k, model_metadata, evaluated_dimension):
    """Keep the Euclidean diagnostic rank separate from each model's own rank.

    The compatibility k/r columns are a matched model pair. Baselines have no
    fitted calibration rank; their parameter access dimension is separate.
    """
    model_k = model_metadata.get("k")
    model_r = model_metadata.get("r")
    return {
        "geometry_k": int(geometry_k),
        "geometry_rank_definition": "V3_EUCLIDEAN_CALIBRATION_DIAGNOSTIC",
        "model_k": None if model_k is None else int(model_k),
        "model_r": None if model_r is None else int(model_r),
        "k": None if model_k is None else int(model_k),
        "r": None if model_r is None else int(model_r),
        "rank_label": model_metadata.get("rank_label", "NOT_APPLICABLE_BASELINE"),
        "baseline_parameter_access_dimension": (
            int(evaluated_dimension) if model_k is None else None
        ),
    }
