"""Frozen CPU trajectory collection and explicit paid-observation/oracle views."""

from __future__ import annotations

import copy
import hashlib
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .toy import (
    LOG_FIELDS,
    adam_vectors,
    build_toy_dataset,
    event_probabilities,
    flatten_parameters,
    make_model,
    parameter_layout,
    perform_step,
    restore,
    sample_bank,
    snapshot,
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16], 16)


def _bank_roles(profile):
    if "fit_bank_indices" in profile:
        return {
            role: list(profile[f"{role}_bank_indices"])
            for role in ("fit", "diagnostic", "evaluation")
        }
    sizes = [profile[f"{role}_banks"] for role in ("fit", "diagnostic", "evaluation")]
    ends = np.cumsum([0, *sizes])
    return {
        role: list(range(int(ends[i]), int(ends[i + 1])))
        for i, role in enumerate(("fit", "diagnostic", "evaluation"))
    }


def _trajectory_specs(profile, config):
    roles = profile.get("seeds_by_role", {})
    if not roles:
        lookup = {
            seed: role
            for role, seeds in config["profiles"]["core"]["seeds_by_role"].items()
            for seed in seeds
        }
        roles = {}
        for seed in profile["seeds"]:
            roles.setdefault(lookup.get(seed, "smoke"), []).append(seed)
    return [
        (seed, arm, role)
        for role, seeds in roles.items()
        for seed in seeds
        for arm in profile["arms"]
    ]


def _freeze_array(value):
    result = np.asarray(value).copy()
    result.flags.writeable = False
    return result


def _exact_p(model, dataset):
    with torch.no_grad():
        return event_probabilities(model, dataset.probe_features, dataset.probe_categories).numpy()


def _collect_trajectory(dataset, seed, arm, split, profile, config):
    steps, anchors = profile["steps"], profile["anchors"]
    bank_roles = _bank_roles(profile)
    bank_count = sum(map(len, bank_roles.values()))
    interventions = config["fork_interventions"]
    nops, B, K = len(interventions), profile["B"], profile["K"]
    model = make_model(config["toy_model"]["model_initialization_seed"])
    opt = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt["lr"],
        betas=tuple(opt["betas"]),
        eps=opt["eps"],
        weight_decay=opt["weight_decay"],
    )
    # Sampling and prompt order have separate streams. Same seed arms share order.
    rng = np.random.default_rng(_seed(seed, "main_action_sampling"))
    order_rng = np.random.default_rng(_seed(seed, "main_prompt_order"))
    prompt_order = order_rng.integers(len(dataset.train_features), size=(steps, B))
    shape = (len(anchors), bank_count, nops)
    observations = {
        "theta": np.zeros((steps + 1, 737)),
        "d": np.zeros((steps, 737)),
        "g": np.zeros((steps, 737)),
        "g_clipped": np.zeros((steps, 737)),
        "checkpoint_grad": np.zeros((steps + 1, 737)),
        "checkpoint_grad_is_none": np.ones((steps + 1, 4), dtype=bool),
        "adam_m": np.zeros((steps + 1, 737)),
        "adam_v": np.zeros((steps + 1, 737)),
        "adam_step": np.zeros(steps + 1, dtype=np.int64),
        "logs": np.zeros((steps, len(LOG_FIELDS))),
        "train_prompt_indices": prompt_order,
        "train_actions": np.zeros((steps, B, K), dtype=np.int8),
        "train_categories": np.zeros((steps, B, K), dtype=np.int8),
        "train_old_logp": np.zeros((steps, B, K)),
        "train_advantages": np.zeros((steps, B, K)),
        "train_rewards": np.zeros((steps, B, K)),
        "branch_theta": np.zeros((*shape, 737)),
        "branch_d": np.zeros((*shape, 737)),
        "branch_g": np.zeros((*shape, 737)),
        "branch_g_clipped": np.zeros((*shape, 737)),
        "branch_adam_m": np.zeros((*shape, 737)),
        "branch_adam_v": np.zeros((*shape, 737)),
        "branch_adam_step": np.zeros(shape, dtype=np.int64),
        "branch_logs": np.zeros((*shape, len(LOG_FIELDS))),
        "branch_prompt_indices": np.zeros((len(anchors), bank_count, B), dtype=np.int64),
        "branch_actions": np.zeros((len(anchors), bank_count, B, K), dtype=np.int8),
        "branch_categories": np.zeros((len(anchors), bank_count, B, K), dtype=np.int8),
        "branch_old_logp": np.zeros((len(anchors), bank_count, B, K)),
        "branch_advantages": np.zeros((*shape, B, K)),
        "branch_rewards": np.zeros((*shape, B, K)),
        "branch_alias": np.arange(np.prod(shape), dtype=np.int64).reshape(shape),
        "anchors": np.asarray(anchors, dtype=np.int64),
    }
    oracle = {"p": np.zeros((steps + 1, 72, 4)), "branch_p": np.zeros((*shape, 72, 4))}
    rng_states, branch_rng_states = [], []
    alias_hashes = {}
    exact_calls = 0
    train_seconds = branch_seconds = exact_seconds = 0.0
    for step in range(steps + 1):
        observations["theta"][step] = flatten_parameters(model)
        observations["checkpoint_grad"][step] = np.concatenate(
            [
                (np.zeros(p.numel()) if p.grad is None else p.grad.detach().numpy().ravel())
                for p in model.parameters()
            ]
        )
        observations["checkpoint_grad_is_none"][step] = [p.grad is None for p in model.parameters()]
        m, v, optstep = adam_vectors(model, optimizer)
        observations["adam_m"][step], observations["adam_v"][step] = m, v
        observations["adam_step"][step] = optstep
        rng_states.append(copy.deepcopy(rng.bit_generator.state))
        started = time.perf_counter()
        oracle["p"][step] = _exact_p(model, dataset)
        exact_seconds += time.perf_counter() - started
        exact_calls += 1
        if step in anchors:
            ai = anchors.index(step)
            anchor_state = snapshot(model, optimizer, rng)
            before_hash = hashlib.sha256(
                observations["theta"][step].tobytes() + m.tobytes() + v.tobytes()
            ).hexdigest()
            for bank_index in range(bank_count):
                branch_rng = np.random.default_rng(_seed(seed, arm, step, bank_index, "local_bank"))
                branch_rng_states.append(
                    {
                        "anchor": step,
                        "bank": bank_index,
                        "before_sampling": copy.deepcopy(branch_rng.bit_generator.state),
                    }
                )
                started = time.perf_counter()
                bank = sample_bank(
                    model, dataset.train_features, dataset.train_categories, branch_rng, B, K
                )
                branch_seconds += time.perf_counter() - started
                for key, source in (
                    ("branch_prompt_indices", "prompt_indices"),
                    ("branch_actions", "actions"),
                    ("branch_categories", "categories"),
                    ("branch_old_logp", "old_logp"),
                ):
                    observations[key][ai, bank_index] = bank[source]
                for op_index, operation in enumerate(interventions):
                    try:
                        # Every operation starts from the identical main state.
                        restore(model, optimizer, rng, anchor_state)
                        started = time.perf_counter()
                        record = perform_step(
                            model,
                            optimizer,
                            bank,
                            lam=operation["lambda"],
                            policy=operation["policy"],
                            operation_index=op_index,
                        )
                        branch_seconds += time.perf_counter() - started
                        where = (ai, bank_index, op_index)
                        for key in ("d", "g", "g_clipped", "logs", "advantages", "rewards"):
                            observations[f"branch_{key}"][where] = record[key]
                        theta = flatten_parameters(model)
                        observations["branch_theta"][where] = theta
                        bm, bv, bs = adam_vectors(model, optimizer)
                        (
                            observations["branch_adam_m"][where],
                            observations["branch_adam_v"][where],
                        ) = bm, bv
                        observations["branch_adam_step"][where] = bs
                        alias_key = hashlib.sha256(theta.tobytes()).hexdigest()
                        flat_index = np.ravel_multi_index(where, shape)
                        if alias_key in alias_hashes:
                            prior = alias_hashes[alias_key]
                            observations["branch_alias"][where] = prior
                            oracle["branch_p"][where] = oracle["branch_p"].reshape(-1, 72, 4)[prior]
                        else:
                            alias_hashes[alias_key] = flat_index
                            started = time.perf_counter()
                            oracle["branch_p"][where] = _exact_p(model, dataset)
                            exact_seconds += time.perf_counter() - started
                            exact_calls += 1
                    finally:
                        restore(model, optimizer, rng, anchor_state)
                am, av, _ = adam_vectors(model, optimizer)
                after_hash = hashlib.sha256(
                    flatten_parameters(model).tobytes() + am.tobytes() + av.tobytes()
                ).hexdigest()
                if (
                    before_hash != after_hash
                    or rng.bit_generator.state != anchor_state["numpy_rng"]
                ):
                    raise RuntimeError("branch contaminated main trajectory")
        if step == steps:
            break
        started = time.perf_counter()
        bank = sample_bank(
            model, dataset.train_features, dataset.train_categories, rng, B, K, prompt_order[step]
        )
        record = perform_step(
            model,
            optimizer,
            bank,
            lam=float(arm == "X_VALID"),
            operation_index=int(arm == "X_VALID"),
        )
        train_seconds += time.perf_counter() - started
        for key in ("d", "g", "g_clipped", "logs"):
            observations[key][step] = record[key]
        for key in ("actions", "categories", "old_logp"):
            observations[f"train_{key}"][step] = bank[key]
        for key in ("advantages", "rewards"):
            observations[f"train_{key}"][step] = record[key]
    return (
        observations,
        oracle,
        {
            "main_numpy_rng_states": rng_states,
            "branch_sampling_rng_states": branch_rng_states,
            "torch_rng_state": torch.random.get_rng_state().tolist(),
            "seed_derivation": (
                "SHA256(JSON tuple) first 64 bits; separate prompt/main/bank/count streams"
            ),
            "exact_panel_calls": exact_calls,
            "unique_branch_outputs": len(alias_hashes),
            "branch_alias_count": int(np.prod(shape)) - len(alias_hashes),
            "training_seconds": train_seconds,
            "branch_seconds": branch_seconds,
            "exact_seconds": exact_seconds,
            "optimizer_steps": steps + int(np.prod(shape)),
            "sampled_finite_actions": (steps + len(anchors) * bank_count) * B * K,
        },
    )


def _sample_counts(oracle, observations, seed, arm, config):
    result = {}
    measurement = config["measurement"]
    replicas = measurement["noise_replicas"]
    aliases = observations["branch_alias"].reshape(-1)
    branch_p = oracle["branch_p"].reshape(-1, 72, 4)
    for n in measurement["samples_per_prompt_grid"]:
        main = np.zeros((replicas, *oracle["p"].shape), dtype=np.uint16)
        branch = np.zeros((replicas, *oracle["branch_p"].shape), dtype=np.uint16)
        for noise in range(replicas):
            rng = np.random.default_rng(
                _seed(measurement["noise_seed_root"], seed, arm, n, noise, "measurement")
            )
            for step, probabilities in enumerate(oracle["p"]):
                main[noise, step] = [rng.multinomial(n, p / p.sum()) for p in probabilities]
            destination = branch[noise].reshape(-1, 72, 4)
            for index, p in enumerate(branch_p):
                if aliases[index] != index:
                    destination[index] = destination[aliases[index]]
                else:
                    destination[index] = [rng.multinomial(n, row / row.sum()) for row in p]
        result[f"trajectory_n{n}"], result[f"branch_n{n}"] = main, branch
    return result


def run_collect(config: dict, profile: str, out: Path) -> dict:
    """Create a fresh complete raw collection; resource authorization is CLI-owned."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    prof = config["profiles"][profile]
    _write(out / "resolved_config.json", config)
    dataset_started = time.perf_counter()
    dataset = build_toy_dataset(out / "generated_dataset", config["dataset"]["seed"])
    dataset_seconds = time.perf_counter() - dataset_started
    np.savez_compressed(
        out / "dataset.npz",
        train_features=dataset.train_features,
        train_categories=dataset.train_categories,
        probe_features=dataset.probe_features,
        probe_categories=dataset.probe_categories,
    )
    _write(out / "probe_metadata.json", dataset.probe_metadata)
    _write(out / "train_metadata.json", dataset.train_metadata)
    _write(out / "parameter_layout.json", parameter_layout(make_model()))
    for directory in ("observations", "oracle", "counts", "rng"):
        (out / directory).mkdir()
    manifest = {
        "schema_version": "modeling-qualification-collection-v1",
        "profile": profile,
        "events": ["X", "S", "W", "I"],
        "parameter_count": 737,
        "probe_count": 72,
        "log_fields": list(LOG_FIELDS),
        "bank_roles": _bank_roles(prof),
        "operation_names": [x["name"] for x in config["fork_interventions"]],
        "anchors": prof["anchors"],
        "steps": prof["steps"],
        "trajectories": [],
        "probe_identity_sha256": _hash(out / "probe_metadata.json"),
        "train_identity_sha256": _hash(out / "train_metadata.json"),
        "dataset_sha256": _hash(out / "dataset.npz"),
        "parameter_layout_sha256": _hash(out / "parameter_layout.json"),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_hashes": {
            name: _hash(Path(__file__).with_name(name)) for name in ("toy.py", "collection.py")
        },
        "safety_certified": False,
        "new_gpu_started": False,
        "online_ssvc_started": False,
    }
    totals = {
        "optimizer_steps": 0,
        "sampled_finite_actions": 0,
        "exact_panel_calls": 0,
        "unique_branch_outputs": 0,
        "branch_alias_count": 0,
        "training_seconds": 0.0,
        "branch_seconds": 0.0,
        "exact_seconds": 0.0,
        "measurement_seconds": 0.0,
        "output_io_seconds": 0.0,
    }
    for seed, arm, split in _trajectory_specs(prof, config):
        tid = f"seed{seed}_{arm}"
        observations, oracle, accounting = _collect_trajectory(
            dataset, seed, arm, split, prof, config
        )
        entry = {
            "id": tid,
            "seed": seed,
            "arm": arm,
            "split": split,
            "observations_file": f"observations/{tid}.npz",
            "oracle_file": f"oracle/{tid}.npz",
            "counts_file": f"counts/{tid}.npz",
            "rng_file": f"rng/{tid}.json",
        }
        measured = time.perf_counter()
        counts = _sample_counts(oracle, observations, seed, arm, config)
        totals["measurement_seconds"] += time.perf_counter() - measured
        io_started = time.perf_counter()
        np.savez_compressed(out / entry["observations_file"], **observations)
        np.savez_compressed(out / entry["oracle_file"], **oracle)
        np.savez_compressed(out / entry["counts_file"], **counts)
        _write(out / entry["rng_file"], accounting)
        entry["sha256"] = {
            field: _hash(out / entry[field])
            for field in ("observations_file", "oracle_file", "counts_file", "rng_file")
        }
        entry["branch_alias_count"] = accounting["branch_alias_count"]
        manifest["trajectories"].append(entry)
        totals["output_io_seconds"] += time.perf_counter() - io_started
        for key in totals:
            if key in accounting:
                totals[key] += accounting[key]
    _write(out / "manifest.json", manifest)
    wall = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    output_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    core_steps = config["budget_derived"]["total_optimizer_steps"]
    ratio = core_steps / totals["optimizer_steps"]
    result = {
        "status": "PASS",
        "profile": profile,
        "trajectory_count": len(manifest["trajectories"]),
        **totals,
        "dataset_seconds": dataset_seconds,
        "wall_seconds": wall,
        "peak_rss_gib": rss_bytes / 2**30,
        "output_bytes": output_bytes,
        "estimated_core_wall_seconds": wall * ratio,
        "estimated_core_output_bytes": int(output_bytes * ratio),
        "estimated_core_peak_rss_gib": rss_bytes / 2**30 + 0.25,
        "estimation_basis": (
            "optimizer step ratio; conservative additional .25 GiB for larger "
            "per-trajectory core arrays; M3/M4 separate"
        ),
        "manifest_sha256": _hash(out / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "new_gpu_started": False,
        "online_ssvc_started": False,
        "safety_certified": False,
    }
    _write(out / "summary.json", result)
    return result


@dataclass(frozen=True, slots=True)
class ObservedView:
    """Materialized approved inputs; no oracle path, dense labels or model states.

    This is a data capability boundary, not a Python sandbox: callers given the
    dataset root remain responsible for following the separate scoring contract.
    """

    meta: dict
    updates: np.ndarray
    logs: np.ndarray
    _anchors: tuple
    _inputs: tuple

    def anchor(self, index, role="fit"):
        if role != "fit":
            raise PermissionError("only paid local-fit responses are available to a predictor")
        return dict(self._anchors[index])

    def anchor_inputs(self, index, role="evaluation"):
        if role not in ("fit", "diagnostic", "evaluation"):
            raise ValueError("unknown bank role")
        return dict(self._inputs[index][role])

    def trajectory_updates(self):
        return self.updates


def _load_identity(root, trajectory_id):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    found = [x for x in manifest["trajectories"] if x["id"] == trajectory_id]
    if len(found) != 1:
        raise ValueError("trajectory identity must match exactly one manifest entry")
    entry = found[0]
    for field in ("observations_file", "counts_file", "oracle_file"):
        if _hash(root / entry[field]) != entry["sha256"][field]:
            raise ValueError(f"identity/hash mismatch: {field}")
    if _hash(root / "probe_metadata.json") != manifest["probe_identity_sha256"]:
        raise ValueError("probe identity mismatch")
    if _hash(root / "dataset.npz") != manifest["dataset_sha256"]:
        raise ValueError("candidate feature/category dataset identity mismatch")
    if _hash(root / "parameter_layout.json") != manifest["parameter_layout_sha256"]:
        raise ValueError("parameter layout identity mismatch")
    metadata = json.loads((root / "probe_metadata.json").read_text())
    return root, manifest, entry, metadata


def load_observed(root, trajectory_id, n=64, noise=0):
    root, manifest, entry, metadata = _load_identity(root, trajectory_id)
    obs = np.load(root / entry["observations_file"], allow_pickle=False)
    labels = np.load(
        root / entry["oracle_file"] if n is None else root / entry["counts_file"],
        allow_pickle=False,
    )
    if n is not None and (
        f"trajectory_n{n}" not in labels or not 0 <= noise < len(labels[f"trajectory_n{n}"])
    ):
        raise ValueError("unavailable measurement budget/replica")
    groups = _freeze_array([m["group"] for m in metadata])
    safe_meta = {
        "id": entry["id"],
        "seed": entry["seed"],
        "arm": entry["arm"],
        "split": entry["split"],
        "anchors": manifest["anchors"],
        "bank_roles": manifest["bank_roles"],
        "log_fields": manifest["log_fields"],
        "operation_names": manifest["operation_names"],
        "groups": groups,
        "prompt_ids": tuple(m["prompt_id"] for m in metadata),
        "n": n,
        "noise": noise,
        "access_level": "EXACT_PAID_OBSERVATIONS" if n is None else "UPDATE_AWARE",
    }
    anchors, all_inputs = [], []
    nops = len(manifest["operation_names"])
    for ai, step in enumerate(manifest["anchors"]):
        inputs = {}
        for role, banks in manifest["bank_roles"].items():
            indices = np.asarray(banks, dtype=int)
            inputs[role] = {
                "d": _freeze_array(obs["branch_d"][ai, indices].reshape(-1, 737)),
                "g": _freeze_array(obs["branch_g"][ai, indices].reshape(-1, 737)),
                "logs": _freeze_array(obs["branch_logs"][ai, indices].reshape(-1, len(LOG_FIELDS))),
                "bank_indices": _freeze_array(np.repeat(indices, nops)),
                "operation_indices": _freeze_array(np.tile(np.arange(nops), len(indices))),
                "branch_ids": tuple(
                    f"{trajectory_id}:t{step}:b{b}:o{o}" for b in banks for o in range(nops)
                ),
                "alias": _freeze_array(obs["branch_alias"][ai, indices].reshape(-1)),
            }
        banks = manifest["bank_roles"]["fit"]
        anchor_counts = (
            None if n is None else _freeze_array(labels[f"trajectory_n{n}"][noise, step])
        )
        p0 = _freeze_array(labels["p"][step] if n is None else anchor_counts / n)
        pfit = _freeze_array(
            (
                labels["branch_p"][ai, banks]
                if n is None
                else labels[f"branch_n{n}"][noise, ai, banks] / n
            ).reshape(-1, 72, 4)
        )
        row = {
            "identity": f"{trajectory_id}:t{step}",
            "anchor_step": step,
            "groups": groups,
            "p_anchor": p0,
            "p0": p0,
            "p_fit": pfit,
            "p1": pfit,
            "anchor_counts": anchor_counts,
            "anchor_measurement_id": f"{trajectory_id}:t{step}:n{n}:noise{noise}",
            "d_fit": inputs["fit"]["d"],
            "g_fit": inputs["fit"]["g"],
            "logs_fit": inputs["fit"]["logs"],
            "d_eval": inputs["evaluation"]["d"],
            "g_eval": inputs["evaluation"]["g"],
            "logs_eval": inputs["evaluation"]["logs"],
            **inputs["fit"],
        }
        anchors.append(row)
        all_inputs.append(inputs)
    obs.close()
    labels.close()
    return ObservedView(
        safe_meta,
        _freeze_array(_read_key(root / entry["observations_file"], "d")),
        _freeze_array(_read_key(root / entry["observations_file"], "logs")),
        tuple(anchors),
        tuple(all_inputs),
    )


def _read_key(path, key):
    with np.load(path, allow_pickle=False) as handle:
        return handle[key]


def load_checkpoint_parameters(root, trajectory_id):
    """Load CPU checkpoints for analytic oracle directions without opening labels."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = [x for x in manifest["trajectories"] if x["id"] == trajectory_id]
    if len(entries) != 1:
        raise ValueError("trajectory identity must match exactly once")
    entry = entries[0]
    if _hash(root / "parameter_layout.json") != manifest["parameter_layout_sha256"]:
        raise ValueError("parameter layout identity mismatch")
    path = root / entry["observations_file"]
    if _hash(path) != entry["sha256"]["observations_file"]:
        raise ValueError("checkpoint identity mismatch")
    return _freeze_array(_read_key(path, "theta"))


def validate_collection(root):
    """Read-only complete integrity/shape/cost audit of a frozen collection.

    This is a scoring-side audit: it intentionally opens all oracle labels.
    Call it outside model selection or fitting, and retain its returned receipt.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "resolved_config.json").read_text())
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if config_hash != manifest["config_sha256"]:
        raise ValueError("resolved configuration identity mismatch")
    profile = config["profiles"][manifest["profile"]]
    expected_specs = _trajectory_specs(profile, config)
    actual_specs = [(r["seed"], r["arm"], r["split"]) for r in manifest["trajectories"]]
    # Role mapping order changes when resolved_config is serialized with
    # sort_keys=True. Identity membership is fixed; trajectory listing order is
    # not an experimental variable. Sorting lists also preserves multiplicity.
    if sorted(actual_specs) != sorted(expected_specs) or len(
        {r["id"] for r in manifest["trajectories"]}
    ) != len(actual_specs):
        raise ValueError("trajectory identities/splits differ from the frozen protocol")
    if manifest["events"] != ["X", "S", "W", "I"]:
        raise ValueError("event order mismatch")
    if manifest["bank_roles"] != _bank_roles(profile) or manifest["anchors"] != profile["anchors"]:
        raise ValueError("branch/anchor role mismatch")
    for filename, key in (
        ("dataset.npz", "dataset_sha256"),
        ("probe_metadata.json", "probe_identity_sha256"),
        ("parameter_layout.json", "parameter_layout_sha256"),
    ):
        if _hash(root / filename) != manifest[key]:
            raise ValueError(f"{filename} identity mismatch")
    # Early smoke files predate the redundant train metadata hash. The original
    # generated training records and feature/category dataset remain hash-bound.
    if (
        "train_identity_sha256" in manifest
        and _hash(root / "train_metadata.json") != manifest["train_identity_sha256"]
    ):
        raise ValueError("train metadata identity mismatch")
    layout = json.loads((root / "parameter_layout.json").read_text())
    if layout != parameter_layout(make_model()):
        raise ValueError("parameter order/shape/dtype layout differs from the toy model")
    probe_meta = json.loads((root / "probe_metadata.json").read_text())
    train_meta = json.loads((root / "train_metadata.json").read_text())
    if len(probe_meta) != 72 or len(train_meta) != 72:
        raise ValueError("prompt panel size mismatch")
    if len({r["prompt_id"] for r in probe_meta}) != 72:
        raise ValueError("probe prompt identity is duplicated")
    if len({r["base_scene_id"] for r in probe_meta}) != 36:
        raise ValueError("paired probe base-scene identity mismatch")
    for field in ("base_scene_id", "truth_structure_hash"):
        if {r[field] for r in probe_meta} & {r[field] for r in train_meta}:
            raise ValueError(f"train/probe {field} overlap")
    if np.bincount([r["group"] for r in probe_meta]).tolist() != [12] * 6:
        raise ValueError("group balance mismatch")
    generated = root / "generated_dataset"
    generated_manifest = json.loads((generated / "manifest.json").read_text())
    if generated_manifest["images_rendered"] or set(generated_manifest["split_sizes"]) != {
        "train",
        "control",
    }:
        raise ValueError("generator rendered images or generated an undeclared split")
    for filename, record in generated_manifest["files"].items():
        if _hash(generated / filename) != record["sha256"]:
            raise ValueError(f"generated original identity mismatch: {filename}")
    with np.load(root / "dataset.npz", allow_pickle=False) as data:
        for split in ("train", "probe"):
            features, categories = data[f"{split}_features"], data[f"{split}_categories"]
            if features.shape != (72, 16, 44) or features.dtype != np.float64:
                raise ValueError("feature dimension/dtype mismatch")
            if categories.shape != (72, 16) or not np.isin(categories, [0, 1, 2, 3]).all():
                raise ValueError("category shape/domain mismatch")
            if not np.all((categories == 0).sum(-1) == 1) or not np.all(
                (categories == 3).sum(-1) == 2
            ):
                raise ValueError("candidate exact/invalid cardinality mismatch")
    T, B, K = profile["steps"], profile["B"], profile["K"]
    A = len(profile["anchors"])
    banks = sum(map(len, _bank_roles(profile).values()))
    ops = len(config["fork_interventions"])
    shape = (A, banks, ops)
    total_main = total_fork = total_main_actions = total_bank_actions = aliases_total = 0
    first_theta = None
    prompt_order_by_seed = {}
    for entry in manifest["trajectories"]:
        for field, digest in entry["sha256"].items():
            path = (root / entry[field]).resolve()
            if not path.is_relative_to(root.resolve()) or _hash(path) != digest:
                raise ValueError(f"manifest referenced artifact mismatch: {field}")
        with (
            np.load(root / entry["observations_file"], allow_pickle=False) as obs,
            np.load(root / entry["oracle_file"], allow_pickle=False) as oracle,
            np.load(root / entry["counts_file"], allow_pickle=False) as counts,
        ):
            expected_shapes = {
                "theta": (T + 1, 737),
                "d": (T, 737),
                "g": (T, 737),
                "adam_m": (T + 1, 737),
                "adam_v": (T + 1, 737),
                "branch_theta": (*shape, 737),
                "branch_d": (*shape, 737),
                "branch_g": (*shape, 737),
                "branch_adam_m": (*shape, 737),
                "branch_adam_v": (*shape, 737),
                "logs": (T, len(LOG_FIELDS)),
                "branch_logs": (*shape, len(LOG_FIELDS)),
                "train_actions": (T, B, K),
                "branch_actions": (A, banks, B, K),
            }
            for key, expected in expected_shapes.items():
                if obs[key].shape != expected or not np.isfinite(obs[key]).all():
                    raise ValueError(f"trajectory shape/nonfinite failure: {key}")
            theta = obs["theta"]
            if first_theta is None:
                first_theta = theta[0].copy()
            if not np.array_equal(theta[0], first_theta):
                raise ValueError("initial model parameters differ across trajectories")
            if not np.array_equal(obs["d"], np.diff(theta, axis=0)):
                raise ValueError("stored main update differs from checkpoint displacement")
            expected_branch_d = obs["branch_theta"] - theta[profile["anchors"]][:, None, None]
            if not np.array_equal(obs["branch_d"], expected_branch_d):
                raise ValueError("stored branch update differs from checkpoint displacement")
            if not np.array_equal(obs["adam_step"], np.arange(T + 1)):
                raise ValueError("main Adam step mismatch")
            if not np.all(
                obs["branch_adam_step"] == np.asarray(profile["anchors"])[:, None, None] + 1
            ):
                raise ValueError("branch Adam step mismatch")
            order = obs["train_prompt_indices"]
            prior = prompt_order_by_seed.setdefault(entry["seed"], order.copy())
            if not np.array_equal(order, prior):
                raise ValueError("same-seed arms do not share prompt order")
            for key, expected in (("p", (T + 1, 72, 4)), ("branch_p", (*shape, 72, 4))):
                probabilities = oracle[key]
                if (
                    probabilities.shape != expected
                    or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0)
                ):
                    raise ValueError("oracle probability shape/domain mismatch")
                if not np.allclose(probabilities.sum(-1), 1, atol=1e-13, rtol=0):
                    raise ValueError("oracle simplex sum mismatch")
            alias = obs["branch_alias"].reshape(-1)
            if np.any(alias < 0) or np.any(alias > np.arange(alias.size)):
                raise ValueError("invalid or forward alias reference")
            aliases_total += int(np.count_nonzero(alias != np.arange(alias.size)))
            for key, values in (("theta", obs["branch_theta"]), ("p", oracle["branch_p"])):
                flattened = values.reshape(alias.size, *values.shape[3:])
                if not np.array_equal(flattened, flattened[alias]):
                    raise ValueError(f"alias {key} differs from its canonical output")
            for n in config["measurement"]["samples_per_prompt_grid"]:
                replicas = config["measurement"]["noise_replicas"]
                for key, expected in (
                    (f"trajectory_n{n}", (replicas, T + 1, 72, 4)),
                    (f"branch_n{n}", (replicas, *shape, 72, 4)),
                ):
                    values = counts[key]
                    if values.shape != expected or not np.issubdtype(values.dtype, np.integer):
                        raise ValueError("finite count shape/dtype mismatch")
                    if np.any(values < 0) or not np.all(values.sum(-1) == n):
                        raise ValueError("finite count total/domain mismatch")
                flattened = counts[f"branch_n{n}"].reshape(replicas, alias.size, 72, 4)
                if not np.array_equal(flattened, flattened[:, alias]):
                    raise ValueError("alias measurements were independently resampled")
            total_main += T
            total_fork += int(np.prod(shape))
            total_main_actions += T * B * K
            total_bank_actions += A * banks * B * K
    if manifest["profile"] == "core":
        expected = config["budget_derived"]
        for actual, name in (
            (total_main, "main_optimizer_steps"),
            (total_fork, "fork_optimizer_steps"),
            (total_main_actions, "main_sampled_actions"),
            (total_bank_actions, "fork_base_sampled_actions"),
        ):
            if actual != expected[name]:
                raise ValueError(f"core budget mismatch: {name}")
    return {
        "status": "PASS",
        "manifest_sha256": _hash(root / "manifest.json"),
        "trajectory_count": len(actual_specs),
        "main_optimizer_steps": total_main,
        "fork_optimizer_steps": total_fork,
        "total_optimizer_steps": total_main + total_fork,
        "main_sampled_actions": total_main_actions,
        "fork_base_sampled_actions": total_bank_actions,
        "total_sampled_finite_actions": total_main_actions + total_bank_actions,
        "branch_alias_count": aliases_total,
        "all_referenced_hashes_verified": True,
        "split_and_identity_checks": "PASS",
        "shapes_and_counts": "PASS",
        "initialization_and_optimizer_steps": "PASS",
        "alias_reuse": "PASS",
    }


@dataclass(frozen=True, slots=True)
class OracleView:
    """Explicit offline-scoring capability; never passed to predictor functions."""

    meta: dict
    p: np.ndarray
    branch_p: np.ndarray
    theta: np.ndarray
    probe_features: np.ndarray
    probe_categories: np.ndarray

    def anchor(self, index, role="evaluation"):
        banks = self.meta["bank_roles"][role]
        step = self.meta["anchors"][index]
        return {
            "p0": self.p[step],
            "p1": self.branch_p[index, banks].reshape(-1, 72, 4),
            "p_anchor": self.p[step],
            "p_eval": self.branch_p[index, banks].reshape(-1, 72, 4),
            "theta": self.theta[step],
            "anchor_step": step,
        }


def load_oracle(root, trajectory_id):
    root, manifest, entry, metadata = _load_identity(root, trajectory_id)
    with np.load(root / entry["oracle_file"], allow_pickle=False) as handle:
        p, branch_p = _freeze_array(handle["p"]), _freeze_array(handle["branch_p"])
    with np.load(root / "dataset.npz", allow_pickle=False) as handle:
        features, categories = (
            _freeze_array(handle["probe_features"]),
            _freeze_array(handle["probe_categories"]),
        )
    return OracleView(
        {
            "id": entry["id"],
            "seed": entry["seed"],
            "split": entry["split"],
            "anchors": manifest["anchors"],
            "bank_roles": manifest["bank_roles"],
            "groups": _freeze_array([m["group"] for m in metadata]),
        },
        p,
        branch_p,
        _freeze_array(_read_key(root / entry["observations_file"], "theta")),
        features,
        categories,
    )
