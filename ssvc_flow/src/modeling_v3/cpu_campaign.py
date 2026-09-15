"""Actual finite-action CPU experiments for V3, with immutable replay originals.

The finite probability tables in this module belong to the experiment evaluator.
Observation estimators receive sampled contributions only. Large campaigns are
server jobs; importing this module never collects data or submits a job.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

EVENTS = ("X", "S", "W", "I")
OPERATIONS = (
    ("joint_0", 0.0, "joint"),
    ("joint_1", 1.0, "joint"),
    ("no_x_off_1", 1.0, "no_x_off_before_joint_normalization"),
)
CONTRASTS = ((1, 0), (2, 0), (1, 2))
CONTRAST_NAMES = ("joint_1_minus_joint_0", "no_x_off_1_minus_joint_0", "joint_1_minus_no_x_off_1")
DEV_SEEDS = (101, 201)
DESIGN_SEEDS = tuple(range(2026091500, 2026091510))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _seed(*parts):
    return int(_digest(["modeling-v3", *parts])[:16], 16)


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    temporary = path.with_name(path.name + f".pending-{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _npz(path, **arrays):
    path = Path(path)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are forbidden")
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    temporary = path.with_name(path.name + f".pending-{os.getpid()}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _sources():
    """Bind the complete CPU dependency set without unrelated GPU/CLI modules.

    The final method-selection lock separately binds the entire source tree.
    This permits preparing unrelated VLM modules during immutable CPU execution.
    """
    root = Path(__file__).resolve().parents[2]
    paths = [
        *(
            root / "src/modeling_v3" / name
            for name in (
                "__init__.py",
                "cpu_campaign.py",
                "observation_geometry.py",
                "covariance_pilot.py",
                "bank_selection.py",
                "coverage.py",
                "response_models.py",
                "io.py",
                "schema.py",
                "historical_observation.py",
                "statistics.py",
                "cpu_results.py",
                "frozen_comparisons.py",
            )
        ),
        root / "src/__init__.py",
        root / "src/modeling_qualification/__init__.py",
        root / "src/modeling_qualification/toy.py",
        root / "src/modeling_contrast/__init__.py",
        root / "src/modeling_contrast/collect_fresh.py",
        root / "src/modeling_contrast/io.py",
        root / "src/modeling_contrast/protocol.py",
        root / "src/generate_worlds.py",
        root / "src/constraint_solver.py",
        root / "src/verifiers.py",
        root / "src/prompts.py",
        root / "src/render_charts.py",
        root / "configs/modeling_qualification/protocol.json",
    ]
    return {str(path.relative_to(root)): _hash(path) for path in paths if path.is_file()}


def _binding(config, **scope):
    return {"config_sha256": _digest(config), "source_hashes": _sources(), **scope}


def _verify_complete(root, binding=None):
    root = Path(root)
    path = root / "COMPLETE.json"
    if not path.is_file():
        raise ValueError(f"incomplete attempt; preserve and use a new output path: {root}")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "COMPLETE" or not isinstance(manifest.get("files"), dict):
        raise ValueError("complete manifest schema/status mismatch")
    if binding is not None and manifest["binding"] != binding:
        raise ValueError("resume source/config/scope binding mismatch")
    for relative, digest in manifest["files"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or _hash(target) != digest:
            raise ValueError(f"immutable artifact hash mismatch: {relative}")
    observed = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "COMPLETE.json" and ".pending-" not in p.name
    }
    if observed != set(manifest["files"]):
        raise ValueError("immutable artifact inventory mismatch")
    return manifest


def _finish(root, binding, summary):
    root = Path(root)
    files = {
        str(path.relative_to(root)): _hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "COMPLETE.json" and ".pending-" not in path.name
    }
    manifest = {
        "schema_version": "modeling-v3-cpu-unit-v1",
        "binding": binding,
        "files": files,
        "summary": summary,
        "status": "COMPLETE",
    }
    _json(root / "COMPLETE.json", manifest)
    return summary


def _prepare(root, binding, resume):
    root = Path(root)
    if root.exists():
        if not resume:
            raise FileExistsError(f"immutable run already exists: {root}")
        if (root / "COMPLETE.json").is_file():
            return _verify_complete(root, binding)["summary"]
        existing = json.loads((root / "BINDING.json").read_text())
        if existing != binding:
            raise ValueError("resume source/config/scope binding mismatch")
    else:
        root.mkdir(parents=True)
        _json(root / "BINDING.json", binding)
    return None


def _require_server(pilot):
    if not pilot and platform.system() != "Linux":
        raise RuntimeError("full CPU campaigns must run on the server CPU; local pilot only")


def trajectory_specs(config, role="development", pilot=False):
    """Frozen seed/arm/init matrix. Pilot identities never enter confirmation."""
    cpu = config["cpu"]
    if pilot:
        return [
            {
                "seed": 1999001,
                "arm": "X_BASE",
                "role": "timing_pilot",
                "initialization_seed": cpu["fixed_parent_initialization"],
            }
        ]
    if role == "development":
        seeds, inits = DEV_SEEDS, [cpu["fixed_parent_initialization"]]
    elif role == "interval_calibration":
        seeds, inits = cpu["interval_calibration_seeds"], [cpu["fixed_parent_initialization"]]
    elif role == "locked_test":
        seeds, inits = cpu["locked_test_seeds"], [cpu["fixed_parent_initialization"]]
    elif role == "orthogonal_generalization":
        seeds = cpu["orthogonal_generalization"]["train_rng_seeds"]
        inits = cpu["orthogonal_generalization"]["init_seeds"]
    else:
        raise ValueError("unknown CPU role")
    return [
        {"seed": int(seed), "arm": arm, "role": role, "initialization_seed": int(init)}
        for init in inits
        for seed in seeds
        for arm in cpu["arms"]
    ]


def partition_training_prompts(metadata):
    """48/24 disjoint prompt identities, with 8/4 from each semantic group."""
    calibration, heldout = [], []
    for group in range(6):
        indices = [i for i, row in enumerate(metadata) if row["group"] == group]
        if len(indices) != 12:
            raise ValueError("parent toy needs twelve training prompts per group")
        indices.sort(key=lambda i: _digest(["training-bank-partition", metadata[i]["prompt_id"]]))
        calibration.extend(indices[:8])
        heldout.extend(indices[8:])
    if set(calibration) & set(heldout):
        raise AssertionError("calibration/query prompt overlap")
    return {"calibration": calibration, "heldout": heldout}


def bank_prompt_indices(metadata, partition, bank_index, rng, B=4):
    """Bank-internal unique prompts, balanced rotating metadata groups."""
    groups = [(bank_index + offset) % 6 for offset in range(B)]
    selected = [
        int(rng.choice([i for i in partition if metadata[i]["group"] == group])) for group in groups
    ]
    if len(set(selected)) != B:
        raise AssertionError("duplicate prompt in candidate bank")
    return np.asarray(selected, dtype=np.int64)


def _full_snapshot(model, optimizer, rng, sampler_position):
    import torch

    from src.modeling_qualification.toy import snapshot

    state = snapshot(model, optimizer, rng)
    state.update(
        python_rng=random.getstate(),
        numpy_global_rng=np.random.get_state(),
        module_modes={name: module.training for name, module in model.named_modules()},
        sampler_position=sampler_position,
        cuda_rng=(torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None),
    )
    return state


def _full_restore(model, optimizer, rng, state):
    import torch

    from src.modeling_qualification.toy import restore

    restore(model, optimizer, rng, state)
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_global_rng"])
    for name, module in model.named_modules():
        module.training = state["module_modes"][name]
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def _pack_tree(value, arrays, prefix):
    """Lossless safe tree serialization; never pickle or arbitrary torch.load."""
    import torch

    if isinstance(value, torch.Tensor):
        arrays[prefix] = value.detach().cpu().numpy().copy()
        return {"kind": "tensor", "key": prefix}
    if isinstance(value, np.ndarray):
        arrays[prefix] = value.copy()
        return {"kind": "array", "key": prefix}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                [
                    _pack_tree(k, arrays, f"{prefix}_key{i}"),
                    _pack_tree(v, arrays, f"{prefix}_value{i}"),
                ]
                for i, (k, v) in enumerate(value.items())
            ],
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_pack_tree(v, arrays, f"{prefix}_{i}") for i, v in enumerate(value)],
        }
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported safe state type: {type(value).__name__}")


def unpack_state(tree, arrays):
    import torch

    if not isinstance(tree, dict):
        return tree
    kind = tree["kind"]
    if kind in ("array", "tensor"):
        array = arrays[tree["key"]].copy()
        return torch.from_numpy(array) if kind == "tensor" else array
    if kind == "dict":
        return {unpack_state(k, arrays): unpack_state(v, arrays) for k, v in tree["items"]}
    values = [unpack_state(v, arrays) for v in tree["items"]]
    return tuple(values) if kind == "tuple" else values


def _action_probabilities(model, features):
    import torch

    with torch.no_grad():
        return (
            torch.softmax(model(torch.as_tensor(features, dtype=torch.float64)).squeeze(-1), dim=-1)
            .numpy()
            .copy()
        )


def _state_digest(state):
    arrays = {}
    tree = _pack_tree(state, arrays, "state")
    return _digest(
        {
            "tree": tree,
            "arrays": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
        }
    )


def _collect_trajectory(dataset, spec, config, out, pilot=False):
    import torch

    from src.modeling_qualification.toy import (
        LOG_FIELDS,
        adam_vectors,
        flatten_parameters,
        make_model,
        perform_step,
        sample_bank,
    )

    started = time.perf_counter()
    cpu = config["cpu"]
    steps = 8 if pilot else cpu["steps"]
    anchors = [8] if pilot else cpu["anchors"]
    counts = {
        "calibration": 2 if pilot else config["coverage"]["pool_banks"],
        "heldout": 2 if pilot else config["coverage"]["heldout_banks"],
    }
    total_banks = sum(counts.values())
    B, K = cpu["B"], cpu["K"]
    model = make_model(spec["initialization_seed"])
    random.seed(_seed(spec["seed"], spec["initialization_seed"], "python-global"))
    np.random.seed(_seed(spec["seed"], spec["initialization_seed"], "numpy-global") % 2**32)
    torch.manual_seed(_seed(spec["seed"], spec["initialization_seed"], "torch-global"))
    parent_config = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/modeling_qualification/protocol.json"
        ).read_text()
    )
    opt = parent_config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt["lr"],
        betas=tuple(opt["betas"]),
        eps=opt["eps"],
        weight_decay=opt["weight_decay"],
    )
    rng = np.random.default_rng(_seed(spec["seed"], "source_action"))
    order_rng = np.random.default_rng(_seed(spec["seed"], "source_prompt_order"))
    order = order_rng.integers(len(dataset.train_metadata), size=(steps, B))
    partition = partition_training_prompts(dataset.train_metadata)
    shape = (len(anchors), total_banks, 3)
    arrays = {
        "theta": np.zeros((steps + 1, 737)),
        "train_prompt_indices": order,
        "d": np.zeros((steps, 737)),
        "g": np.zeros((steps, 737)),
        "g_clipped": np.zeros((steps, 737)),
        "logs": np.zeros((steps, len(LOG_FIELDS))),
        "train_actions": np.zeros((steps, B, K), dtype=np.int8),
        "train_categories": np.zeros((steps, B, K), dtype=np.int8),
        "train_old_logp": np.zeros((steps, B, K)),
        "train_advantages": np.zeros((steps, B, K)),
        "train_rewards": np.zeros((steps, B, K)),
        "branch_theta": np.zeros((*shape, 737)),
        "branch_d": np.zeros((*shape, 737)),
        "branch_g": np.zeros((*shape, 737)),
        "branch_g_clipped": np.zeros((*shape, 737)),
        "branch_logs": np.zeros((*shape, len(LOG_FIELDS))),
        "branch_advantages": np.zeros((*shape, B, K)),
        "branch_rewards": np.zeros((*shape, B, K)),
        "branch_prompt_indices": np.zeros((len(anchors), total_banks, B), dtype=np.int64),
        "branch_actions": np.zeros((len(anchors), total_banks, B, K), dtype=np.int8),
        "branch_categories": np.zeros((len(anchors), total_banks, B, K), dtype=np.int8),
        "branch_old_logp": np.zeros((len(anchors), total_banks, B, K)),
        "branch_alias": np.arange(np.prod(shape), dtype=np.int64).reshape(shape),
        "anchors": np.asarray(anchors),
        "adam_step": np.zeros(steps + 1, dtype=np.int64),
        "adam_m": np.zeros((steps + 1, 737)),
        "adam_v": np.zeros((steps + 1, 737)),
    }
    evaluator = {
        "origin_action_p": np.zeros((len(anchors), 72, 16)),
        "branch_action_p": np.zeros((*shape, 72, 16)),
        "trajectory_event_p": np.zeros((steps + 1, 72, 4)),
    }
    state_arrays, state_trees, bank_rng = {}, {}, []
    aliases, restoration_checks = {}, 0
    costs = {
        "training_seconds": 0.0,
        "fork_update_seconds": 0.0,
        "state_copy_restore_seconds": 0.0,
        "exact_scoring_seconds": 0.0,
        "bank_generation_seconds": 0.0,
    }
    indicators = np.eye(4)[dataset.probe_categories]
    for step in range(steps + 1):
        arrays["theta"][step] = flatten_parameters(model)
        m, v, counter = adam_vectors(model, optimizer)
        arrays["adam_m"][step], arrays["adam_v"][step], arrays["adam_step"][step] = m, v, counter
        copy_start = time.perf_counter()
        state = _full_snapshot(model, optimizer, rng, step)
        state_trees[f"source_step{step}"] = _pack_tree(state, state_arrays, f"source{step}")
        costs["state_copy_restore_seconds"] += time.perf_counter() - copy_start
        score_start = time.perf_counter()
        action_p = _action_probabilities(model, dataset.probe_features)
        evaluator["trajectory_event_p"][step] = np.einsum("pa,pae->pe", action_p, indicators)
        costs["exact_scoring_seconds"] += time.perf_counter() - score_start
        if step in anchors:
            ai = anchors.index(step)
            evaluator["origin_action_p"][ai] = action_p
            origin_hash = _state_digest(state)
            for bank_index in range(total_banks):
                role = "calibration" if bank_index < counts["calibration"] else "heldout"
                bank_seed = _seed(
                    spec["seed"],
                    spec["arm"],
                    spec["initialization_seed"],
                    step,
                    bank_index,
                    "bank_rollout",
                )
                branch_rng = np.random.default_rng(bank_seed)
                before_rng = copy.deepcopy(branch_rng.bit_generator.state)
                indices = bank_prompt_indices(
                    dataset.train_metadata, partition[role], bank_index, branch_rng, B
                )
                generation_start = time.perf_counter()
                bank = sample_bank(
                    model,
                    dataset.train_features,
                    dataset.train_categories,
                    branch_rng,
                    B,
                    K,
                    indices,
                )
                costs["bank_generation_seconds"] += time.perf_counter() - generation_start
                bank_rng.append(
                    {
                        "anchor": step,
                        "bank": bank_index,
                        "role": role,
                        "seed": bank_seed,
                        "before": before_rng,
                        "after": copy.deepcopy(branch_rng.bit_generator.state),
                    }
                )
                for key in ("prompt_indices", "actions", "categories", "old_logp"):
                    arrays[f"branch_{key}"][ai, bank_index] = bank[key]
                for oi, (_, lam, policy) in enumerate(OPERATIONS):
                    where = (ai, bank_index, oi)
                    try:
                        copy_start = time.perf_counter()
                        _full_restore(model, optimizer, rng, state)
                        costs["state_copy_restore_seconds"] += time.perf_counter() - copy_start
                        update_start = time.perf_counter()
                        record = perform_step(
                            model, optimizer, bank, lam=lam, policy=policy, operation_index=oi
                        )
                        costs["fork_update_seconds"] += time.perf_counter() - update_start
                        for key in ("d", "g", "g_clipped", "logs", "advantages", "rewards"):
                            arrays[f"branch_{key}"][where] = record[key]
                        theta = flatten_parameters(model)
                        arrays["branch_theta"][where] = theta
                        candidate_state = _full_snapshot(model, optimizer, rng, step)
                        state_trees[f"anchor{step}_bank{bank_index}_candidate{oi}"] = _pack_tree(
                            candidate_state, state_arrays, f"fork{ai}_{bank_index}_{oi}"
                        )
                        key = hashlib.sha256(theta.tobytes()).hexdigest()
                        flat_index = int(np.ravel_multi_index(where, shape))
                        if key in aliases:
                            arrays["branch_alias"][where] = aliases[key]
                            evaluator["branch_action_p"][where] = evaluator[
                                "branch_action_p"
                            ].reshape(-1, 72, 16)[aliases[key]]
                        else:
                            aliases[key] = flat_index
                            score_start = time.perf_counter()
                            evaluator["branch_action_p"][where] = _action_probabilities(
                                model, dataset.probe_features
                            )
                            costs["exact_scoring_seconds"] += time.perf_counter() - score_start
                    finally:
                        copy_start = time.perf_counter()
                        _full_restore(model, optimizer, rng, state)
                        costs["state_copy_restore_seconds"] += time.perf_counter() - copy_start
                    if _state_digest(_full_snapshot(model, optimizer, rng, step)) != origin_hash:
                        raise RuntimeError("candidate fork contaminated complete source origin")
                    restoration_checks += 1
        if step == steps:
            break
        update_start = time.perf_counter()
        bank = sample_bank(
            model, dataset.train_features, dataset.train_categories, rng, B, K, order[step]
        )
        record = perform_step(
            model,
            optimizer,
            bank,
            lam=float(spec["arm"] == "X_VALID"),
            operation_index=int(spec["arm"] == "X_VALID"),
        )
        costs["training_seconds"] += time.perf_counter() - update_start
        for key in ("d", "g", "g_clipped", "logs"):
            arrays[key][step] = record[key]
        for key in ("actions", "categories", "old_logp"):
            arrays[f"train_{key}"][step] = bank[key]
        for key in ("advantages", "rewards"):
            arrays[f"train_{key}"][step] = record[key]
    io_start = time.perf_counter()
    _npz(out / "observations.npz", **arrays)
    _npz(
        out / "source_probabilities.npz",
        origin_action_p=evaluator["origin_action_p"],
        trajectory_event_p=evaluator["trajectory_event_p"],
    )
    _npz(
        out / "calibration_action_p.npz",
        action_p=evaluator["branch_action_p"][:, : counts["calibration"]],
    )
    _npz(
        out / "heldout_action_p.npz",
        action_p=evaluator["branch_action_p"][:, counts["calibration"] :],
    )
    _npz(out / "restoration_state_tensors.npz", **state_arrays)
    _json(out / "restoration_state_trees.json", state_trees)
    _json(out / "bank_rng_states.json", bank_rng)
    _json(
        out / "identity.json",
        {
            **spec,
            "steps": steps,
            "anchors": anchors,
            "bank_counts": counts,
            "bank_partition": partition,
            "operations": [item[0] for item in OPERATIONS],
            "contrasts": CONTRAST_NAMES,
            "parent_optimizer": opt,
            "log_fields": LOG_FIELDS,
            "training_schedule": "paired seed-bound prompt stream; own on-policy arm sampling",
            "bank_semantics": (
                "unseen banks and disjoint training-prompt sources, not independent seeds"
            ),
        },
    )
    costs["output_io_seconds"] = time.perf_counter() - io_start
    return {
        **spec,
        "steps": steps,
        "anchors": len(anchors),
        "fork_updates": int(np.prod(shape)),
        "source_updates": steps,
        "sampled_training_actions": steps * B * K,
        "sampled_bank_actions": len(anchors) * total_banks * B * K,
        "unique_candidate_policies": len(aliases),
        "alias_count": int(np.prod(shape)) - len(aliases),
        "full_origin_restoration_checks": restoration_checks,
        "actual_probe_panel_forward_calls": steps + 1 + len(aliases),
        "actual_probe_action_scores": (steps + 1 + len(aliases)) * 72 * 16,
        "actual_bank_sampler_action_scores": len(anchors) * total_banks * B * 16,
        "actual_source_sampler_action_scores": steps * B * 16,
        "actual_optimizer_forward_action_scores": (steps + int(np.prod(shape))) * B * 16,
        "costs": costs,
        "wall_seconds": time.perf_counter() - started,
        "scientific_status": "PILOT_NOT_CONFIRMATION" if pilot else "COLLECTED_NOT_EVALUATED",
    }


def collect_toy(
    config,
    out,
    *,
    role="development",
    pilot=False,
    resume=False,
    seed_subset=None,
    arm_subset=None,
    initialization_seed=None,
):
    """Collect actual source trajectories and scratch Adam forks, atomically per unit."""
    _require_server(pilot)
    from src.modeling_qualification.toy import build_toy_dataset, make_model, parameter_layout

    specs = trajectory_specs(config, role, pilot)
    if seed_subset is not None:
        allowed = set(seed_subset)
        if not allowed.issubset({row["seed"] for row in specs}):
            raise ValueError("requested seeds are outside the frozen matrix")
        specs = [row for row in specs if row["seed"] in allowed]
    if arm_subset is not None:
        if not set(arm_subset).issubset(config["cpu"]["arms"]):
            raise ValueError("unknown requested arm")
        specs = [row for row in specs if row["arm"] in arm_subset]
    if initialization_seed is not None:
        specs = [row for row in specs if row["initialization_seed"] == initialization_seed]
    if not specs:
        raise ValueError("empty CPU trajectory matrix")
    out = Path(out)
    binding = _binding(config, stage="toy_collection", role=role, pilot=pilot, specs=specs)
    existing = _prepare(out, binding, resume)
    if existing is not None:
        return existing
    started = time.perf_counter()
    data_root = out / "dataset"
    data_binding = _binding(config, stage="toy_dataset")
    if not data_root.exists():
        data_root.mkdir()
        dataset = build_toy_dataset(
            data_root / "generated", config["cpu"]["fixed_parent_data_seed"]
        )
        _npz(
            data_root / "dataset.npz",
            train_features=dataset.train_features,
            train_categories=dataset.train_categories,
            probe_features=dataset.probe_features,
            probe_categories=dataset.probe_categories,
        )
        _json(data_root / "train_metadata.json", dataset.train_metadata)
        _json(data_root / "probe_metadata.json", dataset.probe_metadata)
        _json(data_root / "parameter_layout.json", parameter_layout(make_model()))
        _json(
            data_root / "training_partition.json",
            partition_training_prompts(dataset.train_metadata),
        )
        _finish(data_root, data_binding, {"status": "DATASET_CREATED"})
    else:
        _verify_complete(data_root, data_binding)
        from src.modeling_qualification.toy import ToyDataset

        with np.load(data_root / "dataset.npz", allow_pickle=False) as data:
            dataset = ToyDataset(
                *(
                    data[key].copy()
                    for key in (
                        "train_features",
                        "train_categories",
                        "probe_features",
                        "probe_categories",
                    )
                ),
                json.loads((data_root / "train_metadata.json").read_text()),
                json.loads((data_root / "probe_metadata.json").read_text()),
                {},
            )
    rows = []
    for spec in specs:
        name = f"seed{spec['seed']}_{spec['arm']}_init{spec['initialization_seed']}"
        unit = out / "trajectories" / name
        unit_binding = _binding(
            config,
            stage="toy_trajectory",
            spec=spec,
            pilot=pilot,
            dataset_manifest_sha256=_hash(data_root / "COMPLETE.json"),
        )
        if unit.exists():
            row = _verify_complete(unit, unit_binding)["summary"]
        else:
            unit.mkdir(parents=True)
            _json(unit / "BINDING.json", unit_binding)
            row = _collect_trajectory(dataset, spec, config, unit, pilot)
            _finish(unit, unit_binding, row)
        rows.append({"id": name, "path": f"trajectories/{name}", **row})
    summary = {
        "stage": "toy_collection",
        "status": "COLLECTED",
        "pilot": pilot,
        "role": role,
        "trajectory_count": len(rows),
        "trajectories": rows,
        "source_updates": sum(row["source_updates"] for row in rows),
        "fork_updates": sum(row["fork_updates"] for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "output_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        "new_gpu_calls": 0,
        "online_ssvc": False,
        "scientific_status": "PILOT_NOT_CONFIRMATION" if pilot else "NOT_YET_EVALUATED",
    }
    _json(out / "COLLECTION_SUMMARY.json", summary)
    return _finish(out, binding, summary)


def _observation_module():
    from types import SimpleNamespace

    from . import covariance_pilot as pilot
    from . import observation_geometry as geometry

    return SimpleNamespace(
        stable_origin_weight=geometry.stable_origin_weight,
        stable_pair_weight=geometry.stable_pair_weight,
        correction=geometry.correction,
        fixed_coefficient=geometry.fixed_coefficient,
        pilot_coefficient=pilot.pilot_coefficient,
        oracle_b=pilot.oracle_b,
    )


def draw_packet(origin, baseline, candidate, categories, n, seed, proposal="origin"):
    """Evaluator-only iid finite actions with separate pilot/main RNG streams.

    Probability tables remain in this sampler. The estimator only receives the
    resulting sampled contributions. A MIX source coin is iid for every draw.
    """
    observation = _observation_module()
    origin, baseline, candidate = map(np.asarray, (origin, baseline, candidate))
    if origin.shape != baseline.shape or baseline.shape != candidate.shape:
        raise ValueError("finite policy shape mismatch")
    if origin.shape != categories.shape or n < 4 or n % 4:
        raise ValueError("n must support an exact quarter pilot; matching category table required")
    if proposal not in ("origin", "equal_pair_mixture"):
        raise ValueError("unknown proposal")
    prompts, _ = origin.shape
    packet = {
        "actions": np.zeros((n, prompts), dtype=np.int8),
        "labels": np.zeros((n, prompts), dtype=np.int8),
        "sourcecoin": np.full((n, prompts), -1, dtype=np.int8),
        "sample_role": np.r_[np.zeros(n // 4, dtype=np.int8), np.ones(n - n // 4, dtype=np.int8)],
        "logp_origin": np.zeros((n, prompts)),
        "logp_baseline": np.zeros((n, prompts)),
        "logp_candidate": np.zeros((n, prompts)),
        "weights": np.zeros((n, prompts)),
        "z_raw": np.zeros((n, prompts, 4)),
        "rng_seeds": np.asarray([_seed(seed, "pilot"), _seed(seed, "main")], dtype=np.uint64),
        "sample_ids": np.asarray(
            [f"{seed}:pilot:{i}" if i < n // 4 else f"{seed}:main:{i - n // 4}" for i in range(n)]
        ),
    }
    for role_index, (start, stop) in enumerate(((0, n // 4), (n // 4, n))):
        rng = np.random.default_rng(_seed(seed, "pilot" if role_index == 0 else "main"))
        count = stop - start
        for prompt in range(prompts):
            if proposal == "origin":
                actions = rng.choice(origin.shape[1], size=count, p=origin[prompt])
            else:
                coins = rng.integers(0, 2, size=count, dtype=np.int8)
                actions = np.empty(count, dtype=np.int64)
                for coin, policy in ((0, baseline), (1, candidate)):
                    mask = coins == coin
                    actions[mask] = rng.choice(
                        origin.shape[1], size=int(mask.sum()), p=policy[prompt]
                    )
                packet["sourcecoin"][start:stop, prompt] = coins
            label = categories[prompt, actions]
            log_origin = np.log(origin[prompt, actions])
            log_b, log_u = np.log(baseline[prompt, actions]), np.log(candidate[prompt, actions])
            weight = (
                observation.stable_origin_weight(log_u, log_b, log_origin)
                if proposal == "origin"
                else observation.stable_pair_weight(log_u, log_b)
            )
            packet["actions"][start:stop, prompt] = actions
            packet["labels"][start:stop, prompt] = label
            packet["logp_origin"][start:stop, prompt] = log_origin
            packet["logp_baseline"][start:stop, prompt] = log_b
            packet["logp_candidate"][start:stop, prompt] = log_u
            packet["weights"][start:stop, prompt] = weight
            packet["z_raw"][start:stop, prompt] = weight[:, None] * np.eye(4)[label]
    return packet


def estimate_packet(packet, methods, config, *, known_covariance=None):
    """Compute finite rules using observed draws only, and named oracle separately."""
    observation = _observation_module()
    z = packet["z_raw"]
    if len(set(map(int, packet["rng_seeds"]))) != 2 or len(np.unique(packet["sample_ids"])) != len(
        z
    ):
        raise ValueError("pilot/main sampling identity or RNG leakage")
    pilot_n = len(z) // 4
    outputs, adjusted, diagnostics = {}, {}, {}
    shrink = config["observation"].get("selected_shrink", 0.1)
    cap = config["observation"]["pilot_b_l1_cap"]
    for method in methods:
        estimate = np.zeros(z.shape[1:])
        coefficients = np.zeros(z.shape[1:])
        main_only = method.startswith("PILOT_")
        corrected = np.zeros_like(z[pilot_n:] if main_only else z)
        method_diagnostics = []
        for prompt in range(z.shape[1]):
            sample = z[:, prompt]
            if method == "RAW4":
                b = np.zeros(4)
            elif method in ("EQUAL_ZERO_SUM", "DROP_W", "PRESERVE_XI"):
                b = observation.fixed_coefficient(method)
            elif method in ("PILOT_COV_ZERO_SUM", "PILOT_SHRINK_ZERO_SUM") or method.startswith(
                "PILOT_SHRINK_ZERO_SUM_ETA_"
            ):
                applied_shrink = 0.0 if method == "PILOT_COV_ZERO_SUM" else shrink
                if method.startswith("PILOT_SHRINK_ZERO_SUM_ETA_"):
                    applied_shrink = float(method.split("_ETA_", 1)[1])
                    if applied_shrink not in config["observation"]["cov_shrink_grid"]:
                        raise ValueError("unregistered development covariance shrinkage")
                b, diag = observation.pilot_coefficient(
                    sample[:pilot_n],
                    shrink=applied_shrink,
                    l1_cap=cap,
                )
                method_diagnostics.append(diag)
            elif method == "KNOWN_COV_ORACLE":
                if known_covariance is None:
                    raise PermissionError(
                        "oracle covariance is evaluator-only and must be explicit"
                    )
                b = observation.oracle_b(known_covariance[prompt])
                if b is None:
                    b = observation.fixed_coefficient("PRESERVE_XI")
            elif method == "CROSSFIT_COV_ZERO_SUM":
                split = len(sample) // 2
                b_a, diag_a = observation.pilot_coefficient(sample[:split], shrink=0.0, l1_cap=cap)
                b_b, diag_b = observation.pilot_coefficient(sample[split:], shrink=0.0, l1_cap=cap)
                corrected[:split, prompt] = observation.correction(sample[:split], b_b)
                corrected[split:, prompt] = observation.correction(sample[split:], b_a)
                outputs[f"b_fold0_{method}"] = outputs.get(
                    f"b_fold0_{method}", np.zeros(z.shape[1:])
                )
                outputs[f"b_fold1_{method}"] = outputs.get(
                    f"b_fold1_{method}", np.zeros(z.shape[1:])
                )
                outputs[f"b_fold0_{method}"][prompt] = b_b
                outputs[f"b_fold1_{method}"][prompt] = b_a
                coefficients[prompt] = np.nan  # No single applied coefficient for crossfit.
                estimate[prompt] = corrected[:, prompt].mean(axis=0)
                method_diagnostics.append(
                    {
                        "fold0": diag_a,
                        "fold1": diag_b,
                        "interval": (
                            "requires complete crossfit resampling; iid SE is not certified"
                        ),
                    }
                )
                continue
            else:
                raise ValueError(f"unsupported observation method: {method}")
            coefficients[prompt] = b
            sample = sample[pilot_n:] if main_only else sample
            corrected[:, prompt] = sample if method == "RAW4" else observation.correction(sample, b)
            estimate[prompt] = corrected[:, prompt].mean(axis=0)
        outputs[f"estimate_{method}"] = estimate
        outputs[f"b_{method}"] = coefficients
        if method in ("RAW4", "EQUAL_ZERO_SUM", "DROP_W", "PRESERVE_XI"):
            outputs[f"main_subset_estimate_{method}"] = corrected[pilot_n:].mean(axis=0)
        adjusted[method] = corrected
        diagnostics[method] = method_diagnostics
    outputs["raw_mass_residual"] = z.sum(-1).mean(axis=0)
    outputs["raw_valid_event_sum"] = z[..., :3].sum(-1).mean(axis=0)
    outputs["raw_delta_v_minus_I"] = -z[..., 3].mean(axis=0)
    return outputs, adjusted, diagnostics


def _exact_contrast(baseline, candidate, categories):
    return np.einsum("pa,pae->pe", candidate - baseline, np.eye(4)[categories])


def _exact_covariance(origin, baseline, candidate, categories, proposal):
    proposal_p = origin if proposal == "origin" else (candidate + baseline) / 2
    if np.any((proposal_p == 0) & (candidate != baseline)):
        raise ValueError("insufficient proposal support")
    difference = candidate - baseline
    weights = np.divide(difference, proposal_p, out=np.zeros_like(difference), where=proposal_p > 0)
    z = weights[..., None] * np.eye(4)[categories]
    mean = np.einsum("pa,pae->pe", proposal_p, z)
    return np.einsum("pa,pae,paf->pef", proposal_p, z, z) - np.einsum("pe,pf->pef", mean, mean)


def _independent_counts(baseline, candidate, categories, total_draws, seed):
    """Strong independent-count comparator with the same total generation budget."""
    rng = np.random.default_rng(_seed(seed, "independent_counts"))
    n0, n1 = total_draws // 2, total_draws - total_draws // 2
    result = np.zeros((len(categories), 4))
    counts = np.zeros((2, len(categories), 4), dtype=np.int64)
    if np.array_equal(baseline, candidate):
        # Exact policy aliases need no independent draw to establish zero contrast.
        # Zero-filled count arrays represent unqueried counts, not multinomial n.
        return result, counts
    for index, (policy, n) in enumerate(((baseline, n0), (candidate, n1))):
        for prompt, probabilities in enumerate(policy):
            actions = rng.choice(len(probabilities), size=n, p=probabilities)
            counts[index, prompt] = np.bincount(categories[prompt, actions], minlength=4)
    result[:] = counts[1] / n1 - counts[0] / n0
    return result, counts


def _error_summary(estimates, truth, groups):
    estimates, truth = np.asarray(estimates), np.asarray(truth)
    error = estimates - truth
    group_error = np.stack(
        [error[:, np.asarray(groups) == group].mean(axis=1) for group in range(6)], 1
    )
    group_truth = np.stack([truth[np.asarray(groups) == group].mean(axis=0) for group in range(6)])
    rows = {}
    for index, name in enumerate(EVENTS):
        residual = error[..., index]
        rows[name] = {
            "bias": float(residual.mean()),
            "variance": float(np.var(residual, axis=0, ddof=1).mean())
            if len(estimates) > 1
            else None,
            "mse": float(np.mean(residual**2)),
            "mae": float(np.mean(np.abs(residual))),
            "absolute_error_quantiles": dict(
                zip(
                    ("q50", "q90", "q95", "q99"),
                    np.quantile(np.abs(residual), [0.5, 0.9, 0.95, 0.99]).tolist(),
                    strict=True,
                )
            ),
        }
    for index, name in ((0, "group_delta_pX"), (3, "group_delta_v")):
        residual = group_error[..., index]
        energy = float(np.sum(group_truth[..., index] ** 2)) * len(estimates)
        rows[name] = {
            "mse": float(np.mean(residual**2)),
            "mae": float(np.mean(np.abs(residual))),
            "nrmse": float(np.sqrt(np.sum(residual**2) / energy)) if energy else None,
            "signal_energy": energy,
            "q95_absolute_error": float(np.quantile(np.abs(residual), 0.95)),
        }
    rows["four_event_total_mse"] = float(np.mean(np.sum(error**2, axis=-1)))
    valid_error = estimates[..., :3].sum(-1) - truth[..., :3].sum(-1)
    rows["valid_event_sum_diagnostic"] = {
        "mse": float(np.mean(valid_error**2)),
        "mae": float(np.mean(abs(valid_error))),
        "definition": "X+S+W; RAW4 differs from primary -I by mass residual",
    }
    rows["primary_v_definition"] = "delta_v = -delta_pI"
    return rows


def observe_toy(config, collection_root, out, *, pilot=False, resume=False):
    """Q1 fixed-identity development policies, all eight rules, both proposals.

    Every n1024 repetition retains sampled actions/logp/weights/raw contributions
    and correction coefficients. Other n values retain sufficient statistics and
    deterministic sampling identities. They can be regenerated from originals.
    """
    _require_server(pilot)
    collection_root, out = Path(collection_root), Path(out)
    collection = _verify_complete(collection_root)
    if collection["summary"]["role"] != "development" and not pilot:
        raise PermissionError("Q1 method exploration is development-only")
    binding = _binding(
        config, stage="Q1", pilot=pilot, collection_sha256=_hash(collection_root / "COMPLETE.json")
    )
    existing = _prepare(out, binding, resume)
    if existing is not None:
        return existing
    started = time.perf_counter()
    with np.load(collection_root / "dataset/dataset.npz", allow_pickle=False) as data:
        categories = data["probe_categories"].copy()
    metadata = json.loads((collection_root / "dataset/probe_metadata.json").read_text())
    groups = [row["group"] for row in metadata]
    methods = list(config["observation"]["methods"])
    methods.extend(
        f"PILOT_SHRINK_ZERO_SUM_ETA_{eta:g}" for eta in config["observation"]["cov_shrink_grid"]
    )
    witnesses = analytic_observation_witnesses(config, out / "analytic_witnesses")
    grid = [64] if pilot else config["observation"]["total_draws_grid"]
    repeats = 2 if pilot else config["cpu"]["repeat_measurements_dev"]
    rows = []
    for trajectory in collection["summary"]["trajectories"]:
        unit = collection_root / trajectory["path"]
        identity = json.loads((unit / "identity.json").read_text())
        with np.load(unit / "source_probabilities.npz", allow_pickle=False) as data:
            origins = data["origin_action_p"].copy()
        with np.load(unit / "calibration_action_p.npz", allow_pickle=False) as data:
            candidates = data["action_p"].copy()
        for ai, anchor in enumerate(identity["anchors"]):
            bank = min(
                range(identity["bank_counts"]["calibration"]),
                key=lambda b: _digest([trajectory["id"], anchor, b, "Q1-fixed-pair"]),
            )
            original_bank = (
                identity.get("bank_original_ids", [])[ai][bank]
                if identity.get("bank_original_ids")
                else bank
            )
            historical = bool(identity.get("historical_reused_fixed_policies", False))
            for candidate_index, baseline_index, case in (
                (1, 0, "IDENTITY_HASH_SELECTED"),
                (0, 0, "IDENTICAL_POLICY"),
            ):
                baseline, candidate = (
                    candidates[ai, bank, baseline_index],
                    candidates[ai, bank, candidate_index],
                )
                truth = _exact_contrast(baseline, candidate, categories)
                observed_cases = {
                    "identical_policy": bool(np.array_equal(baseline, candidate)),
                    "max_absolute_action_probability_difference": float(
                        np.max(abs(candidate - baseline))
                    ),
                    "min_X_endpoint_probability": float(
                        min(
                            np.sum(baseline * (categories == 0), axis=1).min(),
                            np.sum(candidate * (categories == 0), axis=1).min(),
                        )
                    ),
                    "structurally_empty_S_prompts": int(np.sum(~np.any(categories == 1, axis=1))),
                    "all_valid_endpoint_prompts": int(
                        np.sum(
                            (np.sum(baseline * (categories == 3), axis=1) == 0)
                            & (np.sum(candidate * (categories == 3), axis=1) == 0)
                        )
                    ),
                    "source": (
                        "REUSED_HISTORICAL_FIXED_POLICIES_NEW_SCORING"
                        if historical
                        else "ACTUAL_SOURCE_TRAJECTORY_SELECTED_POLICY_PAIR"
                    ),
                    "historical_reused_fixed_policies": historical,
                    "original_trajectory_id": identity.get(
                        "original_trajectory_id", trajectory["id"]
                    ),
                    "original_bank": original_bank,
                }
                for proposal in config["observation"]["proposals"]:
                    covariance = _exact_covariance(
                        origins[ai], baseline, candidate, categories, proposal
                    )
                    for n in grid:
                        name = f"{trajectory['id']}_a{anchor}_b{bank}_{case}_{proposal}_n{n}"
                        measurement_root = out / "units" / name
                        unit_binding = {**binding, "measurement_id": name, "repeats": repeats}
                        if (
                            measurement_root.exists()
                            and (measurement_root / "COMPLETE.json").is_file()
                        ):
                            rows.append(_verify_complete(measurement_root, unit_binding)["summary"])
                            continue
                        _prepare(measurement_root, unit_binding, resume)
                        method_estimates = {
                            method: []
                            for method in [*methods, "INDEPENDENT_COUNT", "KNOWN_EVENT_SCORE"]
                        }
                        measurement_start = time.perf_counter()
                        for repeat in range(repeats):
                            rep_root = measurement_root / f"repeat{repeat:03d}"
                            rep_binding = {**unit_binding, "repeat": repeat}
                            if rep_root.exists():
                                _verify_complete(rep_root, rep_binding)
                                with np.load(
                                    rep_root / "statistics.npz", allow_pickle=False
                                ) as data:
                                    estimates = {
                                        method: data[f"estimate_{method}"].copy()
                                        for method in method_estimates
                                    }
                            else:
                                rep_root.mkdir()
                                repeat_started = time.perf_counter()
                                timings = {}
                                seed = _seed(name, repeat)
                                block_started = time.perf_counter()
                                packet = draw_packet(
                                    origins[ai], baseline, candidate, categories, n, seed, proposal
                                )
                                timings["packet_sampling_cached_scoring_label_seconds"] = (
                                    time.perf_counter() - block_started
                                )
                                block_started = time.perf_counter()
                                outputs, _adjusted, diagnostics = estimate_packet(
                                    packet, methods, config, known_covariance=covariance
                                )
                                timings["estimator_seconds"] = time.perf_counter() - block_started
                                block_started = time.perf_counter()
                                counts_estimate, counts = _independent_counts(
                                    baseline, candidate, categories, n, seed
                                )
                                timings["independent_count_seconds"] = (
                                    time.perf_counter() - block_started
                                )
                                block_started = time.perf_counter()
                                outputs.update(
                                    estimate_INDEPENDENT_COUNT=counts_estimate,
                                    independent_counts=counts,
                                    estimate_KNOWN_EVENT_SCORE=truth,
                                    truth=truth,
                                    raw_sum=packet["z_raw"].sum(0),
                                    raw_second_moment=np.einsum(
                                        "npe,npf->pef", packet["z_raw"], packet["z_raw"]
                                    ),
                                )
                                timings["sufficient_statistics_seconds"] = (
                                    time.perf_counter() - block_started
                                )
                                block_started = time.perf_counter()
                                _npz(rep_root / "statistics.npz", **outputs)
                                if n == 1024 or pilot:
                                    _npz(rep_root / "raw_packet.npz", **packet)
                                _json(
                                    rep_root / "sampling_identity.json",
                                    {
                                        "seed": seed,
                                        "n": n,
                                        "proposal": proposal,
                                        "pilot_rng": _seed(seed, "pilot"),
                                        "main_rng": _seed(seed, "main"),
                                        "record_hash": _digest([name, repeat, seed]),
                                        "diagnostics": diagnostics,
                                        "origin_id": trajectory["id"],
                                        "anchor": anchor,
                                        "bank": original_bank,
                                        "local_bank_index": bank,
                                        "historical_reused_fixed_policies": historical,
                                        "independent_count_status": (
                                            "IDENTICAL_POLICY_ALIAS_ZERO_NO_DRAWS"
                                            if observed_cases["identical_policy"]
                                            else "SAMPLED"
                                        ),
                                        "candidate_index": candidate_index,
                                        "baseline_index": baseline_index,
                                        "prompt_metadata_sha256": _hash(
                                            collection_root / "dataset/probe_metadata.json"
                                        ),
                                        "per_draw_record_identity": (
                                            "measurement id + repeat + prompt id + role-specific "
                                            "sample id; raw archive SHA binds all contributions"
                                        ),
                                        "adjusted_contribution_reconstruction": (
                                            "z_raw - b * sum(z_raw), using independent main for "
                                            "pilot; crossfit opposite fold b"
                                        ),
                                        "source_policy_files": str(
                                            unit / "calibration_action_p.npz"
                                        ),
                                        "source_policy_sha256": _hash(
                                            unit / "calibration_action_p.npz"
                                        ),
                                    },
                                )
                                timings["artifact_io_seconds"] = time.perf_counter() - block_started
                                timings["repeat_wall_seconds_before_completion_manifest"] = (
                                    time.perf_counter() - repeat_started
                                )
                                _finish(
                                    rep_root,
                                    rep_binding,
                                    {
                                        "repeat": repeat,
                                        "timings": timings,
                                        "timing_scope": {
                                            "packet": (
                                                "combined finite sampling, cached probability "
                                                "lookup, log weights and labels; no model forward"
                                            ),
                                            "estimator": "all declared rules on the same packet",
                                            "artifact_io": (
                                                "statistics and optional raw packet serialization "
                                                "and writes, identity metadata and source hashing"
                                            ),
                                            "completion_manifest_self_write_included": False,
                                            "sampling_scoring_label_split_measured": False,
                                        },
                                        "generated_actions": n * 72,
                                        "counterfactual_candidate_and_baseline_action_scores": 2
                                        * n
                                        * 72,
                                        "actual_cached_probability_lookups": 3 * n * 72,
                                        "actual_additional_model_forward_calls": 0,
                                        "independent_count_generated_actions": (
                                            0 if observed_cases["identical_policy"] else n * 72
                                        ),
                                        "counterfactual_known_event_score_actions": 2 * 16 * 72,
                                        "known_event_score_source": (
                                            "reuse exact probabilities charged in collection"
                                        ),
                                    },
                                )
                                estimates = {
                                    method: outputs[f"estimate_{method}"]
                                    for method in method_estimates
                                }
                            for method in method_estimates:
                                method_estimates[method].append(estimates[method])
                        row = {
                            "id": name,
                            "seed": trajectory["seed"],
                            "arm": trajectory["arm"],
                            "anchor": anchor,
                            "bank": original_bank,
                            "local_bank_index": bank,
                            "historical_reused_fixed_policies": historical,
                            "case": case,
                            "n": n,
                            "proposal": proposal,
                            "repeats": repeats,
                            "observed_case_inventory": observed_cases,
                            "metrics": {
                                method: _error_summary(values, truth, groups)
                                for method, values in method_estimates.items()
                            },
                            "wall_seconds": time.perf_counter() - measurement_start,
                        }
                        _json(measurement_root / "RESULTS.json", row)
                        _finish(measurement_root, unit_binding, row)
                        rows.append(row)
    summary = {
        "stage": "Q1",
        "status": "PILOT_COMPLETE" if pilot else "OBSERVED",
        "pilot": pilot,
        "units": rows,
        "unit_count": len(rows),
        "repeat_count_per_unit": repeats,
        "wall_seconds": time.perf_counter() - started,
        "scientific_status": "PILOT_NOT_CONFIRMATION" if pilot else "DEVELOPMENT_ONLY",
        "policy_pair_selection": "identity hash only; explicit identical-policy controls",
        "covariance_shrinkage_grid": config["observation"]["cov_shrink_grid"],
        "named_primary_shrinkage": config["observation"].get("selected_shrink", 0.1),
        "grid_comparison": "same independent pilot/main packet; no extra generation or scoring",
        "analytic_witnesses": witnesses,
        "case_coverage_limit": (
            "analytic witness results are separate from actual source-trajectory policy pairs"
        ),
    }
    _json(out / "OBSERVATION_SUMMARY.json", summary)
    return _finish(out, binding, summary)


def analytic_observation_witnesses(config, out):
    """Explicit constructed finite cases; never claimed as source experiments.

    Each case retains exact finite enumeration and an actually sampled packet.
    These establish boundary correctness, not estimator superiority or new-seed
    prediction quality. Scientific development repeats remain in observe_toy.
    """
    out = Path(out)
    binding = _binding(config, stage="ANALYTIC_WITNESS")
    if out.exists():
        return _verify_complete(out, binding)["summary"]
    out.mkdir()
    cases = (
        "IDENTICAL_POLICY",
        "TINY_CONTRAST",
        "RARE_X",
        "STRUCTURALLY_EMPTY_S",
        "ALL_VALID",
        "NO_CALIBRATION_EXCITATION",
        "THREE_REWARD_LEVELS",
    )
    rows = []
    for case in cases:
        baseline = np.full((1, 16), 1 / 16)
        candidate = baseline.copy()
        categories = np.asarray([[0, 1, *([2] * 12), 3, 3]])
        if case == "RARE_X":
            baseline[0, 0] = 1e-6
            baseline[0, 1:] = (1 - 1e-6) / 15
            candidate[:] = baseline
            candidate[0, 0] += 1e-7
            candidate[0, 1] -= 1e-7
        elif case == "STRUCTURALLY_EMPTY_S":
            categories[categories == 1] = 2
            candidate[0, 0] += 0.01
            candidate[0, 1] -= 0.01
        elif case == "ALL_VALID":
            categories[categories == 3] = 2
            candidate[0, 0] += 0.01
            candidate[0, 1] -= 0.01
        elif case == "TINY_CONTRAST":
            candidate[0, 0] += 1e-10
            candidate[0, 1] -= 1e-10
        elif case == "THREE_REWARD_LEVELS":
            candidate[0, 0] += 0.01
            candidate[0, 15] -= 0.01
        truth = _exact_contrast(baseline, candidate, categories)
        records = []
        for proposal in config["observation"]["proposals"]:
            packet = draw_packet(
                baseline, baseline, candidate, categories, 64, _seed(case, proposal), proposal
            )
            covariance = _exact_covariance(baseline, baseline, candidate, categories, proposal)
            outputs, _, _ = estimate_packet(
                packet, config["observation"]["methods"], config, known_covariance=covariance
            )
            proposal_p = baseline if proposal == "origin" else (baseline + candidate) / 2
            contribution = ((candidate - baseline) / proposal_p)[..., None] * np.eye(4)[categories]
            enumerated = np.einsum("pa,pae->pe", proposal_p, contribution)
            residual = float(np.max(abs(enumerated - truth)))
            if residual > 1e-14:
                raise AssertionError("analytic witness enumeration is biased")
            _npz(
                out / f"{case}_{proposal}.npz",
                baseline=baseline,
                candidate=candidate,
                categories=categories,
                truth=truth,
                known_covariance=covariance,
                **packet,
                **outputs,
            )
            records.append(
                {"proposal": proposal, "draws": 64, "exact_expectation_residual": residual}
            )
        rows.append(
            {
                "case": case,
                "provenance": "ANALYTIC_WITNESS_NOT_SOURCE_TRAJECTORY",
                "records": records,
                "source_training_steps": 0,
                "calibration_excitation_case": case == "NO_CALIBRATION_EXCITATION",
            }
        )
    summary = {
        "status": "ANALYTIC_BOUNDARY_CHECKS_EXECUTED",
        "cases": rows,
        "scientific_confirmation": False,
        "one_sampled_packet_per_case_proposal": True,
    }
    _json(out / "WITNESS_SUMMARY.json", summary)
    return _finish(out, binding, summary)


def development_designs(config, *, pilot=False, selected=None):
    """Matched allocation/rank slices instead of a full Cartesian search."""
    if pilot:
        return [
            {
                "study": "TIMING_PILOT",
                "m": 2,
                "n": 64,
                "selector": "STRATIFIED_RANDOM",
                "design_seed": DESIGN_SEEDS[0],
                "estimator": estimator,
                "model": "FULL_RIDGE",
                "rank_cap": "FULL",
                "alpha": 1e-5,
            }
            for estimator in ("PRESERVE_XI", "PILOT_SHRINK_ZERO_SUM")
        ]
    rows = []
    if selected is not None:
        if (
            "CROSSFIT_COV_ZERO_SUM" in selected["observation_methods"]
            and "FULL_GLS" in selected["models"]
        ):
            raise ValueError(
                "crossfit + FULL_GLS requires full-refit covariance; iid draw covariance is invalid"
            )
        for n in config["observation"]["total_draws_grid"]:
            for estimator in selected["observation_methods"]:
                for selector in selected["selection_rules"]:
                    for model in selected["models"]:
                        if model in ("DIRECT_MEASURE", "KNOWN_EVENT_SCORE"):
                            continue
                        ranks = (
                            ["FULL"]
                            if model in ("ZERO", "FULL_RIDGE", "FULL_GLS", "RBF_RIDGE")
                            else selected["rank_caps"]
                        )
                        for rank in ranks:
                            rows.append(
                                {
                                    "study": "FROZEN_CONFIRMATION",
                                    "m": config["coverage"]["primary_fit_banks"],
                                    "n": n,
                                    "selector": selector,
                                    "design_seed": DESIGN_SEEDS[0],
                                    "estimator": estimator,
                                    "model": model,
                                    "rank_cap": rank,
                                    "alpha": selected["alpha"],
                                }
                            )
        return rows
    estimators = ("PRESERVE_XI", "PILOT_SHRINK_ZERO_SUM")
    for m in config["coverage"]["fit_bank_grid"]:
        allocations = [
            ("FIXED_PER_BANK", n, None) for n in config["observation"]["total_draws_grid"]
        ]
        allocations += [
            ("FIXED_CONTRAST_DRAW_BUDGET", max(4, (budget // m // 4) * 4), budget)
            for budget in (2048, 8192, 24576)
        ]
        for study, n, budget in allocations:
            for selector in config["coverage"]["selection_rules"]:
                for design_seed in (
                    DESIGN_SEEDS if selector == "STRATIFIED_RANDOM" else DESIGN_SEEDS[:1]
                ):
                    for estimator in estimators:
                        rows.append(
                            {
                                "study": study,
                                "m": m,
                                "n": n,
                                "budget": budget,
                                "actual_bank_draws": m * n,
                                "selector": selector,
                                "design_seed": design_seed,
                                "estimator": estimator,
                                "model": "FULL_RIDGE",
                                "rank_cap": "FULL",
                                "alpha": 1e-5,
                            }
                        )
    for selector in ("STRATIFIED_RANDOM", "BLOCK_PIVOT_QR"):
        for estimator in estimators:
            for model in config["models"]["baselines"]:
                if model in ("DIRECT_MEASURE", "KNOWN_EVENT_SCORE"):
                    continue
                ranks = (
                    ["FULL"]
                    if model in ("ZERO", "FULL_RIDGE", "FULL_GLS", "RBF_RIDGE")
                    else config["coverage"]["rank_caps"]
                )
                for rank in ranks:
                    rows.append(
                        {
                            "study": "DIMENSION",
                            "m": 8,
                            "n": 1024,
                            "selector": selector,
                            "design_seed": DESIGN_SEEDS[0],
                            "estimator": estimator,
                            "model": model,
                            "rank_cap": rank,
                            "alpha": 1e-5,
                        }
                    )
            for alpha in config["models"]["ridge_alpha_grid"]:
                rows.append(
                    {
                        "study": "ALPHA",
                        "m": 8,
                        "n": 1024,
                        "selector": selector,
                        "design_seed": DESIGN_SEEDS[0],
                        "estimator": estimator,
                        "model": "FULL_RIDGE",
                        "rank_cap": "FULL",
                        "alpha": alpha,
                    }
                )
    return rows


def random_direction_seed(trajectory_id, anchor, design):
    """One nested random rotation per fixed origin and calibration selection.

    Observation, draw budget, measurement repeat, alpha and rank never change
    the random rotation. Every rank uses a prefix of this same k-by-k rotation.
    """
    return _seed(
        trajectory_id,
        anchor,
        design["selector"],
        design["design_seed"],
        design["m"],
        "shared-response-rotation",
    )


def _safe_json(value):
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (tuple, list)):
        return [_safe_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _score_shared_packet(base_packet, origin, baseline, candidate, categories):
    """Score an existing common-origin action packet without resampling actions."""
    observation = _observation_module()
    actions = base_packet["actions"]
    index = np.arange(len(categories))[None]
    log_o = np.log(origin[index, actions])
    log_b = np.log(baseline[index, actions])
    log_u = np.log(candidate[index, actions])
    weight = observation.stable_origin_weight(log_u, log_b, log_o)
    labels = categories[index, actions]
    return {
        "actions": actions,
        "sample_role": base_packet["sample_role"],
        "rng_seeds": base_packet["rng_seeds"],
        "sample_ids": base_packet["sample_ids"],
        "sourcecoin": base_packet["sourcecoin"],
        "labels": labels,
        "logp_origin": log_o,
        "logp_baseline": log_b,
        "logp_candidate": log_u,
        "weights": weight,
        "z_raw": weight[..., None] * np.eye(4)[labels],
    }


def _measure_banks(origin, candidates, categories, n, seed, methods, config, output, *, save_raw):
    """Evaluator-owned bank label store. Predictor reads selected rows afterward."""
    measurement_binding = {
        "seed": seed,
        "n": n,
        "methods": methods,
        "save_raw": save_raw,
        "config_sha256": _digest(config),
        "source_hashes": _sources(),
        "origin_sha256": hashlib.sha256(origin.tobytes()).hexdigest(),
        "candidates_sha256": hashlib.sha256(candidates.tobytes()).hexdigest(),
        "category_sha256": hashlib.sha256(categories.tobytes()).hexdigest(),
    }
    if output.exists():
        manifest = _verify_complete(output, measurement_binding)
        with np.load(output / "estimates.npz", allow_pickle=False) as arrays:
            estimates = {method: arrays[method].copy() for method in methods}
        with np.load(output / "contributions.npz", allow_pickle=False) as arrays:
            contributions = {method: arrays[method].copy() for method in methods}
        return estimates, contributions, manifest["summary"]
    output.mkdir(parents=True)
    started = time.perf_counter()
    base_packet = draw_packet(origin, origin, origin, categories, n, seed, "origin")
    values = {method: [] for method in methods}
    contributions = {method: [] for method in methods}
    all_b, raw = {}, {}
    diagnostics = []
    for bank in range(len(candidates)):
        bank_values = {method: [] for method in methods}
        bank_contrib = {method: [] for method in methods}
        for ci, (candidate_index, baseline_index) in enumerate(CONTRASTS[:2]):
            packet = _score_shared_packet(
                base_packet,
                origin,
                candidates[bank, baseline_index],
                candidates[bank, candidate_index],
                categories,
            )
            estimates, adjusted, diag = estimate_packet(packet, methods, config)
            for method in methods:
                bank_values[method].append(estimates[f"estimate_{method}"])
                bank_contrib[method].append(adjusted[method])
                all_b[f"b{bank}_c{ci}_{method}"] = estimates[f"b_{method}"]
                if method == "CROSSFIT_COV_ZERO_SUM":
                    all_b[f"b{bank}_c{ci}_fold0_{method}"] = estimates[f"b_fold0_{method}"]
                    all_b[f"b{bank}_c{ci}_fold1_{method}"] = estimates[f"b_fold1_{method}"]
            if save_raw:
                for field in ("logp_baseline", "logp_candidate", "weights", "z_raw"):
                    raw[f"b{bank}_c{ci}_{field}"] = packet[field]
            diagnostics.append({"bank": bank, "contrast": ci, "methods": diag})
        for method in methods:
            values[method].append(bank_values[method])
            contributions[method].append(bank_contrib[method])
    values = {key: np.asarray(value) for key, value in values.items()}
    contributions = {key: np.asarray(value) for key, value in contributions.items()}
    _npz(output / "estimates.npz", **values)
    _npz(output / "contributions.npz", **contributions)
    _npz(output / "coefficients.npz", **all_b)
    _npz(output / "shared_origin_packet.npz", **base_packet)
    if save_raw:
        _npz(output / "raw_scores.npz", **raw)
    _json(output / "diagnostics.json", _safe_json(diagnostics))
    summary = {
        "n": n,
        "seed": seed,
        "bank_count": len(candidates),
        "pilot_seed": _seed(seed, "pilot"),
        "main_seed": _seed(seed, "main"),
        "generated_finite_actions": 72 * n,
        "actual_additional_model_forward_calls": 0,
        "actual_cached_probability_lookups": len(candidates) * 6 * 72 * n,
        "counterfactual_unique_candidate_action_scores_before_alias": len(candidates) * 3 * 72 * n,
        "counterfactual_shared_generation_actions": 72 * n,
        "model_probabilities_acquired_in_source_collection": True,
        "wall_seconds": time.perf_counter() - started,
        "estimator_covariance": "conditional on pilot; outer repetitions include pilot variation",
        "labels_ownership": "EVALUATOR_STORE; selected-bank query log required",
    }
    _finish(output, measurement_binding, summary)
    return values, contributions, summary


def _joint_covariance(contributions):
    # Input bank,contrast,draw,prompt,event; preserve the common draw across banks.
    c = np.asarray(contributions)
    c = c.transpose(3, 2, 0, 1, 4).reshape(c.shape[3], c.shape[2], -1)
    centered = c - c.mean(axis=1, keepdims=True)
    return np.einsum("pni,pnj->pij", centered, centered) / (c.shape[1] * (c.shape[1] - 1))


def _query_metrics(prediction, truth, groups, *, include_targets=True):
    finite = np.isfinite(prediction).all(axis=(1, 2))
    error = prediction - truth
    grouped = np.stack([error[:, np.asarray(groups) == g].mean(axis=1) for g in range(6)], 1)
    signal = np.stack([truth[:, np.asarray(groups) == g].mean(axis=1) for g in range(6)], 1)
    result = {
        "total_queries": len(prediction),
        "finite_queries": int(finite.sum()),
        "unknown_queries": int((~finite).sum()),
        "unknown_fraction": float((~finite).mean()),
        "all_cases_mse": float(np.mean(error**2)) if finite.all() else None,
        "finite_cases_mse": float(np.mean(error[finite] ** 2)) if finite.any() else None,
        "max_absolute_residual": float(np.max(abs(error))) if finite.all() else None,
        "finite_max_absolute_residual": float(np.max(abs(error[finite]))) if finite.any() else None,
        "error_ss": float(np.sum(error[finite] ** 2)),
        "signal_ss": float(np.sum(truth[finite] ** 2)),
        "finite_event_count": int(error[finite].size),
        "unknown_not_dropped_from_denominator": True,
        "primary_v_definition": "delta_v = -delta_pI",
    }
    valid_error = prediction[finite, :, :3].sum(-1) - truth[finite, :, :3].sum(-1)
    result["valid_event_sum_diagnostic"] = {
        "mse": float(np.mean(valid_error**2)) if valid_error.size else None,
        "definition": "X+S+W; retained separately from primary -I",
    }
    for index, name in ((0, "pX"), (3, "v")):
        residual = grouped[finite, :, index]
        energy = float(np.sum(signal[finite, :, index] ** 2))
        result[name] = {
            "mae": float(np.mean(abs(residual))) if residual.size else None,
            "mse": float(np.mean(residual**2)) if residual.size else None,
            "error_ss": float(np.sum(residual**2)),
            "signal_ss": energy,
            "count": int(residual.size),
            "max_absolute_residual": float(np.max(abs(residual))) if residual.size else None,
            "nrmse": float(np.sqrt(np.sum(residual**2) / energy))
            if energy and residual.size
            else None,
            "q95": float(np.quantile(abs(residual), 0.95)) if residual.size else None,
            "signal_energy": energy,
        }
    if include_targets:
        if len(prediction) % 3:
            raise ValueError("bank-major query target triplets required")
        result["by_target"] = {
            name: _query_metrics(
                prediction[index::3], truth[index::3], groups, include_targets=False
            )
            for index, name in enumerate(CONTRAST_NAMES)
        }
        result["primary_target"] = CONTRAST_NAMES[0]
    return result


def persist_error_decomposition(
    root,
    rows,
    predictions,
    truth,
    source_theta,
    fit_pool,
    query_e,
    candidate_theta,
    calibration_count,
    calibration_probabilities,
    features,
    categories,
):
    """Evaluator-only exact diagnostics with each noisy fit's basis held fixed.

    The derivative is evaluated at the complete source origin, so candidate
    baseline displacement is retained explicitly. No response basis is refitted
    from exact labels; that would mix observation error with a basis change.
    """
    from src.modeling_qualification.toy import event_jacobian

    from .response_models import _covariance_blocks, _fit_coefficients, error_decomposition

    selected_rows = [row for row in rows if row["design"]["study"] in ("DIMENSION", "TIMING_PILOT")]
    if not selected_rows:
        return {"status": "NOT_IN_PRESPECIFIED_DIMENSION_SLICE", "fit_count": 0}
    output = Path(root) / "error_decomposition"
    if output.exists():
        return _verify_complete(output)["summary"]
    output.mkdir()
    started = time.perf_counter()
    jacobian = event_jacobian(source_theta, features, categories)
    full_linear = np.einsum("ped,qd->qpe", jacobian, query_e)
    baseline_d0 = candidate_theta[:, 0] - source_theta
    _npz(
        output / "ORIGIN_JACOBIAN_AND_BASELINES.npz",
        origin_theta=source_theta,
        jacobian=jacobian,
        all_bank_baseline_d0=baseline_d0,
        all_bank_baseline_d0_norm=np.linalg.norm(baseline_d0, axis=1),
        all_bank_baseline_theta=candidate_theta[:, 0],
        full_linear=full_linear,
        truth=truth,
        query_e=query_e,
    )
    diagnostics = []
    for row in selected_rows:
        design = row["design"]
        design_id = row["design_id"]
        if design["model"] == "RBF_RIDGE":
            diagnostics.append(
                {
                    "design_id": design_id,
                    "status": "SECONDARY_NONLINEAR_MODEL_EXCLUDED_FROM_LINEAR_DECOMPOSITION",
                }
            )
            continue
        if row["k"] == 0 and design["model"] != "ZERO":
            diagnostics.append(
                {"design_id": design_id, "status": "UNKNOWN_NO_CALIBRATION_EXCITATION"}
            )
            continue
        with np.load(Path(root) / row["predictions_relative"], allow_pickle=False) as arrays:
            Q, directions = arrays["Q"].copy(), arrays["directions"].copy()
            indices = arrays["selected_bank_indices"].copy()
        fit_e = fit_pool[indices].reshape(-1, 737)
        exact_y = np.stack(
            [
                _exact_contrast(policies[b], policies[u], categories)
                for policies in calibration_probabilities[indices]
                for u, b in CONTRASTS[:2]
            ]
        )
        covariance = None
        if design["model"] == "FULL_GLS":
            with np.load(
                Path(root) / f"calibration_n{design['n']}/contributions.npz", allow_pickle=False
            ) as arrays:
                covariance = _joint_covariance(arrays[design["estimator"]][indices])
        if design["model"] == "ZERO":
            exact_fit = np.zeros_like(truth)
            beta = np.zeros((0, 72 * 4))
        else:
            blocks = _covariance_blocks(covariance, len(fit_e), 72)[0]
            beta, _, _ = _fit_coefficients(
                fit_e @ directions,
                exact_y.reshape(len(fit_e), -1),
                design["alpha"],
                row["fit_metadata"]["regression"],
                blocks,
                row["fit_metadata"]["output_policy"],
                1e-10,
                1e-12,
            )
            exact_fit = ((query_e @ directions) @ beta).reshape(-1, 72, 4)
        projected_e = (query_e @ Q) @ Q.T
        projected_linear = np.einsum("ped,qd->qpe", jacobian, projected_e)
        decomposition = error_decomposition(
            truth, full_linear, projected_linear, exact_fit, predictions[design_id]
        )
        payload = {
            "Q": Q,
            "directions": directions,
            "exact_calibration_responses": exact_y,
            "exact_fit_coefficients_fixed_directions": beta,
            "full_local_first_order": full_linear,
            "calibration_projected_first_order": projected_linear,
            "exact_calibration_fit": exact_fit,
            "finite_fit": predictions[design_id],
            "selected_calibration_banks": indices,
            "selected_calibration_baseline_d0": baseline_d0[indices],
            "query_bank_baseline_d0": baseline_d0[calibration_count:],
            "projected_query_e": projected_e,
            "total_error": decomposition["total_error"],
            **decomposition["components"],
            **{
                f"squared_{key}": value
                for key, value in decomposition["per_unit_squared_norms"].items()
            },
            **{
                f"cross_{key}": value
                for key, value in decomposition["per_unit_cross_terms"].items()
            },
        }
        _npz(output / f"{design_id}.npz", **payload)
        diagnostics.append(
            {
                "design_id": design_id,
                "status": "EXACT_DIAGNOSTIC_COMPUTED",
                "k": row["k"],
                "r": row["r"],
                "basis_source": "FROZEN_FINITE_FIT_DIRECTIONS",
                "squared_norms": decomposition["squared_norms"],
                "cross_terms": decomposition["cross_terms"],
                "total_squared_error": decomposition["total_squared_error"],
                "reconstruction_residual": decomposition["reconstruction_residual"],
                "causal_fraction_interpretation": False,
                "prediction_sha256_before_evaluator": row["prediction_sha256"],
            }
        )
    summary = {
        "status": "EVALUATOR_DECOMPOSITION_RECORDED",
        "fit_count": len(diagnostics),
        "diagnostics": diagnostics,
        "wall_seconds": time.perf_counter() - started,
        "additional_exact_jacobian_evaluations": 1,
        "heldout_not_used_for_basis_or_fit": True,
        "new_exact_calibration_fits_are_diagnostic_only": True,
    }
    _json(output / "DECOMPOSITION_SUMMARY.json", _safe_json(summary))
    return _finish(
        output,
        {"prediction_lock_sha256": _hash(Path(root) / "PREDICTION_LOCK.json")},
        _safe_json(summary),
    )


def coverage_study(
    config,
    collection_root,
    out,
    *,
    pilot=False,
    resume=False,
    selection=None,
    calibration_receipt=None,
):
    """Q2/Q3 bank selection and response prediction with sealed query references.

    Q2 scans prespecified matched slices. Q3 uses only supplied frozen rules.
    The source unit retains heldout probabilities separately. All predictions
    for an origin/repetition are published before that file is opened.
    """
    from .bank_selection import select_banks
    from .coverage import classify_coverage, coverage_diagnostics
    from .response_models import fit_response_model, group_xv_map

    _require_server(pilot)
    collection_root, out = Path(collection_root), Path(out)
    collection = _verify_complete(collection_root)
    role = collection["summary"]["role"]
    if not pilot and role != "development" and selection is None:
        raise PermissionError("fresh CPU fitting requires frozen method selection")
    if (
        not pilot
        and role in ("locked_test", "orthogonal_generalization")
        and calibration_receipt is None
    ):
        raise PermissionError("locked response evaluation requires a completed calibration receipt")
    if selection is not None and "selected" in selection:
        selection = selection["selected"]
    if not pilot and role in ("locked_test", "orthogonal_generalization"):
        calibration_receipt = verify_cpu_calibration_receipt(
            config, {"selection_hash": _digest(selection)}, calibration_receipt
        )
    designs = development_designs(config, pilot=pilot, selected=selection)
    binding = _binding(
        config,
        stage="Q3" if selection else "Q2",
        pilot=pilot,
        role=role,
        collection_sha256=_hash(collection_root / "COMPLETE.json"),
        designs=designs,
        calibration_receipt=calibration_receipt,
    )
    existing = _prepare(out, binding, resume)
    if existing is not None:
        return existing
    _json(out / "DESIGNS.json", designs) if not (out / "DESIGNS.json").exists() else None
    metadata = json.loads((collection_root / "dataset/probe_metadata.json").read_text())
    groups = [row["group"] for row in metadata]
    train_metadata = json.loads((collection_root / "dataset/train_metadata.json").read_text())
    with np.load(collection_root / "dataset/dataset.npz", allow_pickle=False) as arrays:
        categories = arrays["probe_categories"].copy()
        features = arrays["probe_features"].copy()
    repeats = 1 if selection is None or pilot else config["cpu"]["repeat_measurements_test"]
    started, result_rows = time.perf_counter(), []
    methods = list(dict.fromkeys(design["estimator"] for design in designs))
    for trajectory in collection["summary"]["trajectories"]:
        unit = collection_root / trajectory["path"]
        identity = json.loads((unit / "identity.json").read_text())
        calibration_count = identity["bank_counts"]["calibration"]
        with np.load(unit / "observations.npz", allow_pickle=False) as arrays:
            candidate_theta = arrays["branch_theta"].copy()
            bank_prompts = arrays["branch_prompt_indices"].copy()
            source_thetas = arrays["theta"].copy()
        with np.load(unit / "source_probabilities.npz", allow_pickle=False) as arrays:
            origins = arrays["origin_action_p"].copy()
        with np.load(unit / "calibration_action_p.npz", allow_pickle=False) as arrays:
            calibration_probabilities = arrays["action_p"].copy()
        for ai, anchor in enumerate(identity["anchors"]):
            fit_pool = (
                candidate_theta[ai, :calibration_count, 1:]
                - candidate_theta[ai, :calibration_count, :1]
            )
            query_e = np.stack(
                [
                    candidate_theta[ai, calibration_count:, u]
                    - candidate_theta[ai, calibration_count:, b]
                    for u, b in CONTRASTS
                ],
                1,
            ).reshape(-1, 737)
            strata = [
                "-".join(map(str, sorted(train_metadata[int(p)]["group"] for p in indices)))
                for indices in bank_prompts[ai, :calibration_count]
            ]
            for repeat in range(repeats):
                name = f"{trajectory['id']}_a{anchor}_repeat{repeat:02d}"
                root = out / "units" / name
                unit_binding = {**binding, "origin": name}
                if root.exists() and (root / "COMPLETE.json").is_file():
                    result_rows.extend(_verify_complete(root, unit_binding)["summary"]["results"])
                    continue
                _prepare(root, unit_binding, resume)
                predictions, predictions_meta = {}, []
                fit_seconds = measurement_seconds = 0.0
                # One n cache at a time bounds resident memory for large draw budgets.
                for n in sorted(set(row["n"] for row in designs)):
                    values, contributions, measurement_cost = _measure_banks(
                        origins[ai],
                        calibration_probabilities[ai],
                        categories,
                        n,
                        _seed(name, n, "calibration"),
                        methods,
                        config,
                        root / f"calibration_n{n}",
                        save_raw=n == 1024 or pilot,
                    )
                    measurement_seconds += measurement_cost["wall_seconds"]
                    for design in [row for row in designs if row["n"] == n]:
                        fit_start = time.perf_counter()
                        design_id = _digest(design)[:20]
                        fit_root = root / "fits" / design_id
                        if fit_root.exists() and (fit_root / "COMPLETE.json").is_file():
                            previous = _verify_complete(fit_root)["summary"]
                            with np.load(
                                fit_root / "PREDICTIONS.npz", allow_pickle=False
                            ) as arrays:
                                predictions[design_id] = arrays["prediction"].copy()
                            predictions_meta.append(previous)
                            continue
                        fit_root.mkdir(parents=True, exist_ok=False)
                        chosen = select_banks(
                            fit_pool,
                            design["m"],
                            design["selector"],
                            strata=strata,
                            seed=design["design_seed"],
                        )
                        indices = chosen["selected_indices"]
                        fit_e = fit_pool[indices].reshape(-1, 737)
                        y = values[design["estimator"]][indices].reshape(-1, 72, 4)
                        covariance = (
                            _joint_covariance(contributions[design["estimator"]][indices])
                            if design["model"] == "FULL_GLS"
                            else None
                        )
                        model = fit_response_model(
                            fit_e,
                            y,
                            design["model"],
                            design["rank_cap"],
                            design["alpha"],
                            covariance=covariance,
                            group_map=group_xv_map(groups),
                            random_seed=random_direction_seed(trajectory["id"], anchor, design),
                        )
                        prediction = model["predict"](query_e)
                        geometry = coverage_diagnostics(fit_e, query_e, alpha=1e-14)
                        # Geometry alone cannot certify a probability tolerance.
                        envelope = (calibration_receipt or {}).get("designs", {}).get(design_id, {})
                        radius, tolerance = envelope.get("radius"), envelope.get("tolerance")
                        empirical_qualified = (
                            role == "locked_test"
                            and radius is not None
                            and tolerance is not None
                            and np.isfinite([radius, tolerance]).all()
                            and 0 <= radius <= tolerance
                        )
                        target_fit_e = np.stack(
                            (
                                fit_pool[indices, 0],
                                fit_pool[indices, 1],
                                fit_pool[indices, 0] - fit_pool[indices, 1],
                            ),
                            axis=1,
                        )
                        excitation_by_target = (
                            np.linalg.norm(target_fit_e, axis=-1).max(axis=0)
                            > config["coverage"]["rank_atol"]
                        )
                        excitation = np.tile(excitation_by_target, len(query_e) // 3)
                        classification = classify_coverage(
                            geometry,
                            rho_threshold=(selection or {}).get("rho_threshold", 0.05),
                            leverage_threshold=(selection or {}).get("leverage_threshold", 10.0),
                            target_excitation=excitation,
                            measurement_resolved=empirical_qualified,
                            validated_tolerance=empirical_qualified,
                        )
                        if design["model"] == "ZERO":
                            classification = {
                                "labels": ["STATISTICAL_BASELINE_ONLY"] * len(query_e),
                                "status": ["STATISTICAL_BASELINE_ONLY"] * len(query_e),
                                "accepted": np.zeros(len(query_e), dtype=bool),
                            }
                        _npz(
                            fit_root / "PREDICTIONS.npz",
                            prediction=prediction,
                            query_e=query_e,
                            rho=geometry["rho"],
                            leverage=geometry["leverage"],
                            e_norm=geometry["e_norm"],
                            e_perp_norm=geometry["e_perp_norm"],
                            e_perp=geometry["e_perp"],
                            singular_values=geometry["singular_values"],
                            Q=model["Q"],
                            directions=model["directions"],
                            coefficients=model["coefficients"],
                            selected_bank_indices=np.asarray(indices),
                            classifications=np.asarray(classification["labels"]),
                        )
                        _json(
                            fit_root / "QUERY_LOG.json",
                            {
                                "selected_calibration_banks": indices,
                                "hidden_query_labels_read": False,
                                "hidden_query_source": "heldout_action_p.npz",
                                "training_seed": trajectory["seed"],
                                "role": role,
                                "observation_store_sha256": _hash(
                                    root / f"calibration_n{n}/COMPLETE.json"
                                ),
                                "prediction_sha256": _hash(fit_root / "PREDICTIONS.npz"),
                            },
                        )
                        meta = {
                            "design_id": design_id,
                            "design": design,
                            "seed": trajectory["seed"],
                            "origin_id": name,
                            "arm": trajectory["arm"],
                            "initialization_seed": trajectory["initialization_seed"],
                            "anchor": anchor,
                            "repeat": repeat,
                            "k": model["k"],
                            "r": model["r"],
                            "rank_comparison_eligible": 0 < model["r"] < model["k"],
                            "condition_number": geometry["condition_number"],
                            "status": model["status"],
                            "target_excitation_from_actual_parameter_contrasts": (
                                excitation_by_target.tolist()
                            ),
                            "selector": _safe_json(chosen["audit"]),
                            "fit_metadata": _safe_json(model["metadata"]),
                            "predictions_relative": str(
                                (fit_root / "PREDICTIONS.npz").relative_to(root)
                            ),
                            "prediction_sha256": _hash(fit_root / "PREDICTIONS.npz"),
                            "all_pool_geometry_acquisition_fork_updates": calibration_count * 3,
                            "calibration_contrast_draw_budget": design["m"] * 2 * n * 72,
                            "counterfactual_selected_bank_score_actions": design["m"] * 3 * n * 72,
                            "counterfactual_shared_generation_actions": n * 72,
                            "geometry_only_does_not_validate_tolerance": True,
                        }
                        _finish(fit_root, {"design": design, "origin": name}, meta)
                        predictions[design_id] = prediction
                        predictions_meta.append(meta)
                        fit_seconds += time.perf_counter() - fit_start
                    del values, contributions
                # This immutable lock precedes the first opening of heldout reference probabilities.
                prediction_lock = {
                    row["design_id"]: row["prediction_sha256"] for row in predictions_meta
                }
                if not (root / "PREDICTION_LOCK.json").exists():
                    _json(root / "PREDICTION_LOCK.json", prediction_lock)
                elif json.loads((root / "PREDICTION_LOCK.json").read_text()) != prediction_lock:
                    raise ValueError("resume prediction lock mismatch")
                for row in predictions_meta:
                    if _hash(root / row["predictions_relative"]) != row["prediction_sha256"]:
                        raise ValueError("prediction changed before query evaluation")
                with np.load(unit / "heldout_action_p.npz", allow_pickle=False) as arrays:
                    heldout = arrays["action_p"][ai].copy()
                truth = np.stack(
                    [
                        _exact_contrast(policies[b], policies[u], categories)
                        for policies in heldout
                        for u, b in CONTRASTS
                    ]
                )
                _npz(root / "QUERY_REFERENCE.npz", truth=truth)
                _json(
                    root / "REFERENCE_ACCESS_RECEIPT.json",
                    {
                        "prediction_lock_sha256": _hash(root / "PREDICTION_LOCK.json"),
                        "reference_source_sha256": _hash(unit / "heldout_action_p.npz"),
                        "reference_status": "KNOWN_EVENT_SCORE_EXACT_FINITE_ACTION_ENUMERATION",
                        "heldout_read_after_prediction_freeze": True,
                        "training_seed": trajectory["seed"],
                    },
                )
                results = []
                for row in predictions_meta:
                    results.append(
                        {
                            **row,
                            "metrics": _query_metrics(predictions[row["design_id"]], truth, groups),
                        }
                    )
                decomposition_receipt = persist_error_decomposition(
                    root,
                    predictions_meta,
                    predictions,
                    truth,
                    source_thetas[anchor],
                    fit_pool,
                    query_e,
                    candidate_theta[ai],
                    calibration_count,
                    calibration_probabilities[ai],
                    features,
                    categories,
                )
                # Strong direct measurement uses all n draws, and the same two observation rules.
                direct_results = []
                for n in sorted(set(row["n"] for row in designs)):
                    direct, _, cost = _measure_banks(
                        origins[ai],
                        heldout,
                        categories,
                        n,
                        _seed(name, n, "direct-query"),
                        methods,
                        config,
                        root / f"direct_n{n}",
                        save_raw=n == 1024 or pilot,
                    )
                    for method in methods:
                        base = direct[method]
                        direct_prediction = np.stack(
                            (base[:, 0], base[:, 1], base[:, 0] - base[:, 1]), 1
                        ).reshape(-1, 72, 4)
                        direct_results.append(
                            {
                                "model": "DIRECT_MEASURE",
                                "estimator": method,
                                "n": n,
                                "metrics": _query_metrics(direct_prediction, truth, groups),
                                "cost": cost,
                            }
                        )
                direct_results.append(
                    {
                        "model": "KNOWN_EVENT_SCORE",
                        "metrics": _query_metrics(truth, truth, groups),
                        "actual_additional_model_scores": 0,
                        "counterfactual_enumerated_action_scores": len(heldout) * 3 * 72 * 16,
                        "source": (
                            "reuse evaluator probabilities already acquired "
                            "and charged in collection"
                        ),
                        "scientific_interpretation": (
                            "finite toy event is exactly enumerable and remains a strong baseline"
                        ),
                    }
                )
                row = {
                    "origin": name,
                    "results": results,
                    "direct_baselines": direct_results,
                    "fit_seconds": fit_seconds,
                    "measurement_seconds": measurement_seconds,
                    "prediction_lock_sha256": _hash(root / "PREDICTION_LOCK.json"),
                    "error_decomposition": decomposition_receipt,
                    "all_pool_scoring_is_research_acquisition_cost": True,
                    "method_counterfactual_cost_uses_selected_banks_only": True,
                    "classification_status": "EMPIRICAL_TOLERANCE_CALIBRATION_PENDING",
                }
                _json(root / "RESULTS.json", _safe_json(row))
                _finish(root, unit_binding, _safe_json(row))
                result_rows.extend(results)
    summary = {
        "stage": "Q3" if selection else "Q2",
        "role": role,
        "pilot": pilot,
        "status": "PILOT_COMPLETE" if pilot else "MEASURED_NOT_SCIENTIFIC_PASS",
        "fit_count": len(result_rows),
        "wall_seconds": time.perf_counter() - started,
        "probe_groups": groups,
        "r_less_k_fit_count": sum(row["rank_comparison_eligible"] for row in result_rows),
        "results": result_rows,
        "scientific_status": (
            "PILOT_NOT_CONFIRMATION"
            if pilot
            else "DEVELOPMENT_ONLY"
            if role == "development"
            else "AWAITING_SEED_CLUSTER_ANALYSIS"
        ),
        "fixed_cost_definition": (
            "equal m*n contrast draws (rounded); generated/scored actions and actual time "
            "reported separately"
        ),
        "refusal_scope": (
            "geometry-only rejection; accepted empirical-error envelopes "
            "require interval calibration"
        ),
    }
    _json(out / "COVERAGE_SUMMARY.json", _safe_json(summary))
    return _finish(out, binding, _safe_json(summary))


def validate_cpu(
    config,
    lock,
    out,
    *,
    pilot=False,
    resume=False,
    roles=None,
    seed_subset=None,
    arm_subset=None,
    existing_run_roots=None,
    calibration_receipt=None,
):
    """Execute the frozen 100-primary +40-secondary trajectory matrix on CPU."""
    from .schema import verify_selection_lock

    _require_server(pilot)
    if pilot:
        raise ValueError("timing pilot must use collect_toy/coverage_study; never Q3 confirmation")
    frozen = verify_selection_lock(config, lock)
    development_designs(config, selected=frozen["selected"])
    roles = roles or ["interval_calibration"]
    if not set(roles).issubset(
        {"interval_calibration", "locked_test", "orthogonal_generalization"}
    ):
        raise ValueError("unknown frozen CPU role")
    if not existing_run_roots:
        raise ValueError(
            "fresh seed collision inventory roots required before new CPU confirmation"
        )
    requested = {
        row["seed"]
        for role in roles
        for row in trajectory_specs(config, role)
        if seed_subset is None or row["seed"] in seed_subset
    }
    if set(roles) & {"locked_test", "orthogonal_generalization"}:
        calibration_receipt = verify_cpu_calibration_receipt(config, frozen, calibration_receipt)
    out = Path(out)
    binding = _binding(
        config,
        stage="Q3_FROZEN_CAMPAIGN",
        selection_hash=frozen["selection_hash"],
        roles=roles,
        seed_subset=seed_subset,
        arm_subset=arm_subset,
        calibration_receipt=calibration_receipt,
    )
    excluded_files = set()
    if resume and out.exists():
        if json.loads((out / "BINDING.json").read_text()) != binding:
            raise ValueError("resume source/config/scope binding mismatch")
        for receipt in out.rglob("COMPLETE.json"):
            verified = _verify_complete(receipt.parent)
            excluded_files.add(str(receipt.resolve()))
            excluded_files.update(
                str((receipt.parent / name).resolve()) for name in verified["files"]
            )
    collisions = scan_cpu_seed_inventory(existing_run_roots, requested)
    collisions = [
        row for row in collisions if str(Path(row["evidence"]).resolve()) not in excluded_files
    ]
    if collisions:
        raise ValueError(
            "new CPU seed collision; register replacement before reading results: "
            + _canonical(collisions)
        )
    existing = _prepare(out, binding, resume)
    if existing is not None:
        return existing
    results = []
    for role in roles:
        verify_selection_lock(config, frozen)
        collection = collect_toy(
            config,
            out / role / "collection",
            role=role,
            resume=resume,
            seed_subset=seed_subset,
            arm_subset=arm_subset,
        )
        study = coverage_study(
            config,
            out / role / "collection",
            out / role / "response",
            resume=resume,
            selection=frozen["selected"],
            calibration_receipt=calibration_receipt,
        )
        results.append({"role": role, "collection": collection, "study": study})
    summary = {
        "stage": "Q3",
        "status": "COLLECTED_AND_MEASURED",
        "results": results,
        "selection_hash": frozen["selection_hash"],
        "scientific_status": "AWAITING_SEED_CLUSTER_ANALYSIS",
        "seed_inventory_roots": list(map(str, existing_run_roots)),
        "seed_collisions": [],
        "online_ssvc": False,
        "new_gpu_calls": 0,
    }
    _json(out / "CPU_VALIDATION_SUMMARY.json", _safe_json(summary))
    return _finish(out, binding, _safe_json(summary))


def verify_cpu_calibration_receipt(config, frozen, receipt):
    """Use the analyzer's identical source/config/evidence gate before collection."""
    from .cpu_results import _verify_calibration

    if receipt is None:
        raise PermissionError("locked CPU test requires completed interval-calibration receipt")
    checked = _verify_calibration(config, frozen, receipt, fixture=False)
    if checked.get("status") != "CALIBRATION_ANALYZED":
        raise ValueError("completed scientific calibration analysis receipt required")
    return checked


def scan_cpu_seed_inventory(existing_run_roots, requested_seeds):
    """Inventory identities only, including V3 directory-named seed originals."""
    import re

    from src.modeling_contrast.collect_fresh import scan_seed_collisions

    rows = scan_seed_collisions(existing_run_roots, set(requested_seeds))
    for root in map(Path, existing_run_roots):
        for path in root.rglob("*"):
            if not path.is_file() or ".git" in path.relative_to(root).parts:
                continue
            if path.name == "identity.json":
                identity = json.loads(path.read_text())
                if identity.get("seed") in requested_seeds and identity.get("steps", 0) > 0:
                    rows.append({"seed": identity["seed"], "evidence": str(path.resolve())})
            elif path.suffix == ".npz":
                # Read path identity only; never open response arrays before collision review.
                for part in path.relative_to(root).parts:
                    match = re.search(r"(?:^|_)seed(\d+)(?:_|\.)", part)
                    if match and int(match.group(1)) in requested_seeds:
                        rows.append({"seed": int(match.group(1)), "evidence": str(path.resolve())})
                        break
    return list({(row["seed"], row["evidence"]): row for row in rows}.values())
