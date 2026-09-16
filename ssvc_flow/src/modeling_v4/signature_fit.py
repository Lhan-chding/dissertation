"""Server CPU whole-seed signature learning from bound development calibration rows.

The prediction entrypoint accepts a frozen model and paid signature features
only. It has no query/reference label argument or future-label callback.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from ..modeling_v3.io import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    finalize_run,
    sha256_file,
    source_identity,
    verify_manifest,
)
from .config import validate_config
from .data_adapter import bound_json
from .full_kernels import anchored_rbf_from_gram, fit_full_response
from .nonlinear import fit_signature_residual, refit_signature_residual, validate_training_partition
from .observations import PRIMARY_METHODS

SEEDS = (41001, 41002, 41003)


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def _doc(value):
    binding = value if isinstance(value, dict) else _binding(value)
    return bound_json(binding), binding


def _arrays(binding):
    if sha256_file(binding["path"]) != binding["sha256"]:
        raise ValueError("Array bytes changed")
    with np.load(binding["path"], allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    if any(v.dtype.hasobject or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Finite numeric arrays required")
    return result


def _gate(fixture):
    if type(fixture) is not bool:
        raise ValueError("Explicit boolean fixture scope required")
    if not fixture:
        from .response_fit import _require_server_cpu

        _require_server_cpu()


def _complete_document(binding):
    value, actual = _doc(binding)
    root = Path(actual["path"]).parent
    if not (root / "COMPLETE.json").exists() or not (root / "RUN_MANIFEST.json").exists():
        raise ValueError("Complete immutable manifest required")
    verify_manifest(root)
    return value, actual


def _signature(binding, fixture):
    value, actual = _doc(binding)
    if (
        value.get("kind") != "V4_NATIVE_SIGNATURE_FEATURES"
        or value.get("query_endpoint_target_labels_read") is not False
        or value.get("reference_labels_read") is not False
    ):
        raise ValueError("Actual independent signature feature receipt required")
    runtime = value["runtime_identity"]
    if bool(runtime.get("fixture", False)) != fixture or (
        not fixture and runtime.get("execution_kind") != "REAL_CUDA_MODEL"
    ):
        raise ValueError("Signature fixture/real execution scope differs")
    root = Path(actual["path"]).parent
    for relative, digest in value["artifact_hashes"].items():
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Signature original artifact changed")
    inventory = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != Path(actual["path"])
    }
    if inventory != set(value["artifact_hashes"]):
        raise ValueError("Signature artifact inventory changed")
    provenance = json.loads((root / "SIGNATURE_PROVENANCE.json").read_text())
    state = bound_json(value["origin_state"])
    arrays = _arrays(value["arrays"])
    if (
        state.get("stage") != "ORIGIN"
        or state.get("origin_id") != value["origin_id"]
        or state.get("panel_role") != "signature"
    ):
        raise ValueError("Only original signature-panel state features are permitted")
    if arrays["native_signatures"].shape != (len(value["units"]), 576) or len(provenance) != len(
        value["units"]
    ):
        raise ValueError("Native feature/unit/provenance axes differ")
    if arrays["EVENT_ONLY"].shape != (36, 4) or arrays["EVENT_PLUS_DIAGNOSTICS"].shape != (36, 11):
        raise ValueError("Origin state feature panel differs")
    return value, actual, arrays, provenance, state


def _dataset(config, origin_inputs, methods, fixture):
    if len(origin_inputs) != 6:
        raise ValueError("Exactly six development origins required")
    e, states, ys, records, provenance, state_provenance = (
        [],
        {"EVENT_ONLY": [], "EVENT_PLUS_DIAGNOSTICS": []},
        {m: [] for m in methods},
        [],
        [],
        [],
    )
    source_hash = "fixture" if fixture else source_identity()["sha256"]
    origins, seen_pairs, removed, bindings = set(), set(), [], []
    prompt_ids, probe_groups, campaign_id = None, None, None
    prepared = []
    for item in origin_inputs:
        if set(item) != {"signature", "labels"}:
            raise ValueError("Only bound signature and calibration-label inputs are accepted")
        labels, lb = _complete_document(item["labels"])
        sig, sb, features, prov, state = _signature(item["signature"], fixture)
        if (
            labels.get("kind") != "V4_CALIBRATION_LABELS"
            or labels.get("role") != "development"
            or labels.get("bank_role_scope") != "calibration"
            or labels.get("query_labels_read") is not False
            or labels.get("reference_labels_read") is not False
            or labels.get("config_hash") != canonical_hash(config)
            or labels.get("source_hash") != source_hash
            or labels.get("origin_id") != sig["origin_id"]
            or sig["runtime_identity"].get("config_hash") != canonical_hash(config)
            or sig["runtime_identity"].get("source_hash") != source_hash
        ):
            raise ValueError("Only matching-source development calibration labels may train")
        if labels.get("event_order") != ["X", "S", "W", "I"]:
            raise ValueError("Calibration event order must remain X/S/W/I")
        if not fixture and labels.get("binding", {}).get("execution_kind") != "REAL_CUDA_MODEL":
            raise ValueError("Production calibration labels need actual CUDA acquisition evidence")
        key = (labels["seed"], labels["arm"], labels["step"])
        if (
            key in origins
            or labels["seed"] not in SEEDS
            or labels["arm"] != "X_BASE"
            or labels["step"] not in (32, 96)
        ):
            raise ValueError("Exact development seed/arm/origin matrix required")
        origins.add(key)
        if (
            not fixture
            and (labels["calibration_bank_count"] != 96 or len(labels["prompt_ids"]) != 72)
        ) or labels["draws"] != 1024:
            raise ValueError(
                "Production signature fitting requires 96 calibration banks and 72 prompts at n1024"
            )
        if fixture and labels["calibration_bank_count"] > 2:
            raise ValueError("Driver fixtures are limited to two tiny banks per origin")
        if prompt_ids is None:
            prompt_ids, probe_groups = labels["prompt_ids"], labels["probe_groups"]
        if labels["prompt_ids"] != prompt_ids or labels["probe_groups"] != probe_groups:
            raise ValueError("Development origins must share the frozen output prompt/group axes")
        if len(set(prompt_ids)) != len(prompt_ids) or len(probe_groups) != len(prompt_ids):
            raise ValueError("Output prompt and group metadata invalid")
        if not fixture and len(set(probe_groups)) != 6:
            raise ValueError("Six actual family/interface output groups required")
        ys_local = _arrays(labels["arrays"])
        if fixture:
            current_campaign = canonical_hash(["fixture", canonical_hash(config)])
        else:
            plan = bound_json(labels["binding"]["tasks"])
            if plan["source"]["sha256"] != source_hash or canonical_hash(
                plan["config"]
            ) != canonical_hash(config):
                raise ValueError("Original campaign task identity changed")
            current_campaign = plan["campaign_id"]
        if campaign_id is not None and campaign_id != current_campaign:
            raise ValueError("Development inputs span different campaigns")
        campaign_id = current_campaign
        if set(methods) - set(labels["observation_methods"]) or any(
            ys_local[m].shape != (len(labels["units"]), len(prompt_ids), 4) for m in methods
        ):
            raise ValueError("Calibration response axes/observation methods differ")
        bank_ids = {u["bank_id"] for u in labels["units"]}
        if (
            len(bank_ids) != labels["calibration_bank_count"]
            or len(labels["units"]) != len(bank_ids) * 3
        ):
            raise ValueError("Exactly three actual contrasts per calibration bank required")
        bindings.append({"signature": sb, "labels": lb})
        prepared.append((key, labels, sig, features, prov, state, ys_local))
    for _, labels, sig, features, prov, state, ys_local in sorted(prepared, key=lambda row: row[0]):
        index = {(u["bank_id"], u["contrast_id"]): i for i, u in enumerate(sig["units"])}
        if len(index) != len(sig["units"]):
            raise ValueError("Duplicate signature candidate unit")
        seen_units = set()
        for i, row in enumerate(labels["units"]):
            key = (row["bank_id"], row["contrast_id"])
            if key in seen_units or key not in index:
                raise ValueError("Calibration labels must join unique actual signature units")
            seen_units.add(key)
            j = index[key]
            p = prov[j]
            if (
                any(row[k] != labels[k] for k in ("origin_id", "seed", "arm", "step", "role"))
                or row["bank_role"] != "calibration"
            ):
                raise ValueError("A nondevelopment or query row entered calibration labels")
            if any(row[k] != sig["units"][j][k] for k in ("candidate", "baseline")) or any(
                row[k] != p[k] for k in ("candidate_fingerprint", "baseline_fingerprint")
            ):
                raise ValueError("Actual signature/response endpoint fingerprints differ")
            pair = (row["baseline_fingerprint"], row["candidate_fingerprint"])
            if type(row["is_alias"]) is not bool or row["is_alias"] != (pair[0] == pair[1]):
                raise ValueError("Alias flags must match exact policy identities")
            if row["is_alias"] and (
                np.any(features["native_signatures"][j])
                or any(np.any(ys_local[m][i]) for m in methods)
            ):
                raise ValueError("Exact alias has nonzero measured contrast")
            unordered_pair = tuple(sorted(pair))
            if unordered_pair in seen_pairs:
                removed.append(
                    {
                        "origin_id": row["origin_id"],
                        "bank_id": row["bank_id"],
                        "contrast_id": row["contrast_id"],
                        "reason": "EXACT_UNORDERED_ENDPOINT_PAIR_ALREADY_REPRESENTED",
                    }
                )
                continue
            seen_pairs.add(unordered_pair)
            e.append(features["native_signatures"][j])
            records.append(row)
            provenance.append(p)
            for mode in states:
                states[mode].append(features[mode])
            for m in methods:
                ys[m].append(ys_local[m][i])
            state_provenance.append(
                {
                    "stage": "ORIGIN",
                    "origin_id": row["origin_id"],
                    "panel_role": "signature",
                    "panel_id": state["panel_id"],
                }
            )
    if origins != {(s, "X_BASE", step) for s in SEEDS for step in (32, 96)}:
        raise ValueError("Incomplete six-origin development matrix")
    return {
        "signatures": np.asarray(e),
        "state": {k: np.asarray(v) for k, v in states.items()},
        "responses": {k: np.asarray(v) for k, v in ys.items()},
        "records": records,
        "signature_provenance": provenance,
        "state_provenance": state_provenance,
        "prompt_ids": prompt_ids,
        "probe_groups": probe_groups,
        "original_bindings": bindings,
        "deduplicated_units": removed,
        "source_hash": source_hash,
        "campaign_id": campaign_id,
    }


def _specs(config, requested):
    if requested is None:
        requested = []
        for alpha in config["models"]["ridge_alpha"]:
            requested.append({"id": f"linear_a{alpha}", "model": "FULL_DUAL_RIDGE", "alpha": alpha})
            for bandwidth in config["models"]["rbf_bandwidth_multipliers"]:
                requested.append(
                    {
                        "id": f"rbf_a{alpha}_b{bandwidth}",
                        "model": "FULL_RBF_RAW",
                        "alpha": alpha,
                        "bandwidth_multiplier": bandwidth,
                    }
                )
        for width in (256, 1024):
            for mode in ("EVENT_ONLY", "EVENT_PLUS_DIAGNOSTICS"):
                requested.append(
                    {
                        "id": f"residual_{width}_{mode}",
                        "model": "NONLINEAR_SIGNATURE_RESIDUAL",
                        "hidden_width": width,
                        "state_mode": mode,
                    }
                )
    ids = set()
    result = []
    for spec in requested:
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("id"), str)
            or not spec["id"]
            or spec["id"] in ids
        ):
            raise ValueError("Unique predeclared model IDs required")
        ids.add(spec["id"])
        method = spec.get("model")
        allowed = {"id", "model"}
        if method in {"FULL_DUAL_RIDGE", "FULL_RBF_RAW"}:
            allowed |= {"alpha"}
            if spec.get("alpha") not in config["models"]["ridge_alpha"]:
                raise ValueError("Registered ridge grid required")
            if method == "FULL_RBF_RAW":
                allowed |= {"bandwidth_multiplier"}
                if (
                    spec.get("bandwidth_multiplier")
                    not in config["models"]["rbf_bandwidth_multipliers"]
                ):
                    raise ValueError("Registered bandwidth grid required")
        elif method == "NONLINEAR_SIGNATURE_RESIDUAL":
            allowed |= {"hidden_width", "state_mode"}
            if spec.get("hidden_width") not in (256, 1024) or spec.get("state_mode") not in {
                "EVENT_ONLY",
                "EVENT_PLUS_DIAGNOSTICS",
            }:
                raise ValueError("Registered network width/origin state mode required")
        else:
            raise ValueError("Unknown native signature model")
        if set(spec) != allowed:
            raise ValueError("Complete frozen model fields required; no hidden settings")
        result.append(dict(spec))
    if not result:
        raise ValueError("At least one predeclared model required")
    return result


def _partition_args(data):
    return {
        "development_seeds": list(SEEDS),
        "signature_provenance": data["signature_provenance"],
        "state_provenance": data["state_provenance"],
        "probe_groups": data["probe_groups"],
    }


def _fit(data, spec, observation, *, validation_seed=None, epochs=200):
    e = data["signatures"]
    y = data["responses"][observation]
    train = np.array([r["seed"] != validation_seed for r in data["records"]])
    if spec["model"] == "NONLINEAR_SIGNATURE_RESIDUAL":
        fn = refit_signature_residual if validation_seed is None else fit_signature_residual
        kw = {} if validation_seed is None else {"validation_seed": validation_seed}
        return fn(
            e,
            data["state"][spec["state_mode"]],
            y,
            data["records"],
            hidden_width=spec["hidden_width"],
            epochs=epochs,
            network_seeds=(0, 1, 2),
            **_partition_args(data),
            **kw,
        )
    model = fit_full_response(
        e[train],
        y[train],
        method=spec["model"],
        alpha=spec["alpha"],
        rank_cap="FULL",
        bandwidth_multiplier=spec.get("bandwidth_multiplier", 1.0),
    )
    model["metadata"].update(
        input_representation="NATIVE_SEQUENCE_SIGNATURE_576", paid_feature=True
    )
    return model


def _metrics(prediction, truth, groups):
    residual = prediction - truth
    grouped = np.stack(
        [
            np.stack(
                (
                    residual[:, np.array(groups) == g, 0].mean(1),
                    -residual[:, np.array(groups) == g, 3].mean(1),
                ),
                -1,
            )
            for g in dict.fromkeys(groups)
        ],
        1,
    )
    return {
        "raw4_mse": float(np.mean(residual**2)),
        "group_Xv_mse": float(np.mean(grouped**2)),
        "rows": len(truth),
        "target": "HELDOUT_DEVELOPMENT_SEED_CALIBRATION_OBSERVATION_NOT_REFERENCE_TRUTH",
    }


def _save_tensors(path, payload):
    import torch

    fd, temp = tempfile.mkstemp(prefix=".weights-", dir=path.parent)
    os.close(fd)
    try:
        with open(temp, "wb") as stream:
            torch.save(payload, stream)
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def save_signature_model(model, out, *, output_shape):
    """Persist fitted coefficients or pure CPU network tensors, never closures."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    method = model["metadata"]["method"]
    metadata = {
        "kind": "V4_FROZEN_SIGNATURE_MODEL",
        "method": method,
        "output_shape": list(output_shape),
        "metadata": model["metadata"],
    }
    if method == "NONLINEAR_SIGNATURE_RESIDUAL":
        import torch

        payload = {
            "networks": [
                {k: torch.as_tensor(v).detach().cpu().clone() for k, v in weights.items()}
                for weights in model["state_dicts"]
            ],
            "normalization": {
                k: torch.as_tensor(v).detach().cpu().clone()
                for k, v in model["normalization"].items()
            },
        }
        _save_tensors(root / "WEIGHTS.pt", payload)
        metadata["weights"] = _binding(root / "WEIGHTS.pt")
        metadata["serialization"] = "CPU_TENSORS_WEIGHTS_ONLY"
    else:
        fit = model["_serialization"]["fit"]
        if len(fit) != 1:
            raise ValueError("Native signatures use one unprojected feature block")
        arrays = {
            "fit_signatures": next(iter(fit.values())),
            "dual_coefficients": model["dual_coefficients"],
        }
        atomic_npz(root / "COEFFICIENTS.npz", arrays)
        metadata["arrays"] = _binding(root / "COEFFICIENTS.npz")
        metadata["serialization"] = "FITTED_COEFFICIENTS_NO_REFITTING"
    atomic_json(root / "MODEL.json", metadata)
    finalize_run(root, {"kind": "V4_FROZEN_SIGNATURE_MODEL"})
    return _binding(root / "MODEL.json")


def load_signature_model(root):
    """Load only frozen weights; prediction has no label reader or refit path."""
    root = Path(root)
    verify_manifest(root)
    metadata = json.loads((root / "MODEL.json").read_text())
    if metadata.get("kind") != "V4_FROZEN_SIGNATURE_MODEL":
        raise ValueError("Unknown model artifact")
    shape = tuple(metadata["output_shape"])
    if len(shape) != 2 or shape[-1] != 4:
        raise ValueError("Model output must be prompt by RAW4")
    method = metadata["method"]
    if method == "NONLINEAR_SIGNATURE_RESIDUAL":
        import torch
        from torch.nn import functional as F

        binding = metadata["weights"]
        if sha256_file(binding["path"]) != binding["sha256"]:
            raise ValueError("Weights changed")
        payload = torch.load(binding["path"], map_location="cpu", weights_only=True)
        networks, normalization = payload["networks"], payload["normalization"]
        if len(networks) != 3 or any(
            not isinstance(v, torch.Tensor) or v.device.type != "cpu" or not torch.isfinite(v).all()
            for weights in [*networks, normalization]
            for v in weights.values()
        ):
            raise ValueError("Three finite safe CPU tensor networks required")
        width = metadata["metadata"]["hidden_width"]
        z_width = metadata["metadata"]["state_feature_width"]
        expected_normalization = {
            "signature_scale": (576,),
            "state_mean": (z_width,),
            "state_scale": (z_width,),
        }
        if (
            set(normalization) != set(expected_normalization)
            or any(
                tuple(normalization[k].shape) != shape
                for k, shape in expected_normalization.items()
            )
            or bool((normalization["signature_scale"] <= 0).any())
            or bool((normalization["state_scale"] <= 0).any())
        ):
            raise ValueError("Frozen training-only normalization differs")
        output = int(np.prod(shape))
        expected = {
            "linear.weight": (output, 576),
            "residual.0.weight": (width, z_width + 576),
            "residual.0.bias": (width,),
            "residual.2.weight": (width, width),
            "residual.2.bias": (width,),
            "residual.4.weight": (output, width),
            "residual.4.bias": (output,),
        }
        if any(
            set(w) != set(expected) or any(tuple(w[k].shape) != v for k, v in expected.items())
            for w in networks
        ):
            raise ValueError("Frozen network architecture differs")

        def members(e, z):
            e = np.asarray(e, dtype=np.float64)
            z = np.asarray(z, dtype=np.float64).reshape(len(e), -1)
            if (
                e.shape != (len(e), 576)
                or z.shape != (len(e), z_width)
                or not np.isfinite(e).all()
                or not np.isfinite(z).all()
            ):
                raise ValueError("Signature/state prediction dimensions differ")
            qe = torch.as_tensor(e) / normalization["signature_scale"]
            qz = (torch.as_tensor(z) - normalization["state_mean"]) / normalization["state_scale"]
            result = []
            with torch.no_grad():
                for w in networks:

                    def g(values, w=w):
                        for i in (0, 2):
                            values = torch.tanh(
                                F.linear(values, w[f"residual.{i}.weight"], w[f"residual.{i}.bias"])
                            )
                        return F.linear(values, w["residual.4.weight"], w["residual.4.bias"])

                    value = (
                        F.linear(qe, w["linear.weight"])
                        + g(torch.cat((qz, qe), 1))
                        - g(torch.cat((qz, torch.zeros_like(qe)), 1))
                    )
                    result.append(value.numpy().reshape(len(e), *shape))
            return np.stack(result)

        return {
            "predict": lambda e, z: members(e, z).mean(0),
            "predict_members": members,
            "metadata": metadata,
        }
    if method not in {"FULL_DUAL_RIDGE", "FULL_RBF_RAW"}:
        raise ValueError("Unknown frozen kernel method")
    arrays = _arrays(metadata["arrays"])
    fit, dual = arrays["fit_signatures"], arrays["dual_coefficients"]
    if fit.ndim != 2 or fit.shape[1] != 576 or dual.shape != (len(fit), int(np.prod(shape))):
        raise ValueError("Frozen kernel dimensions differ")

    def predict(e, z=None):
        e = np.asarray(e, dtype=np.float64)
        if e.ndim != 2 or e.shape[1] != 576 or not np.isfinite(e).all():
            raise ValueError("Full native signature required")
        cross = e @ fit.T
        norm = np.sum(e * e, 1)
        fitnorm = np.sum(fit * fit, 1)
        kernel = (
            cross
            if method == "FULL_DUAL_RIDGE"
            else anchored_rbf_from_gram(
                cross, norm, fitnorm, length_scale=metadata["metadata"]["length_scale"]
            )
        )
        y = kernel @ dual
        y[norm == 0] = 0
        if not np.any(fitnorm):
            y[norm > 0] = np.nan
        return y.reshape(len(e), *shape)

    return {"predict": predict, "metadata": metadata}


def fit_signature_development(
    config,
    origin_inputs,
    *,
    out,
    observation_methods=None,
    model_specs=None,
    epochs=200,
    fixture=False,
):
    _gate(fixture)
    config = validate_config(config)
    if type(epochs) is not int or epochs < 1 or (fixture and epochs > 2):
        raise ValueError("Freeze positive epochs; fixtures allow at most two")
    methods = list(observation_methods or PRIMARY_METHODS)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(PRIMARY_METHODS):
        raise ValueError("Registered observation methods required")
    specs = _specs(config, model_specs)
    data = _dataset(config, origin_inputs, methods, fixture)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    metadata = {k: v for k, v in data.items() if k not in {"signatures", "state", "responses"}}
    arrays = {
        "signatures": data["signatures"],
        **{f"state_{k}": v for k, v in data["state"].items()},
        **{f"response_{k}": v for k, v in data["responses"].items()},
    }
    atomic_npz(root / "DATASET.npz", arrays)
    metadata.update(
        arrays=_binding(root / "DATASET.npz"),
        config=config,
        config_hash=canonical_hash(config),
        fixture=fixture,
        epochs=epochs,
        observation_methods=methods,
    )
    atomic_json(root / "DATASET.json", metadata)
    dataset_binding = _binding(root / "DATASET.json")
    model_rows = {
        f"{m}/{s['id']}": {"id": f"{m}/{s['id']}", "spec": s, "observation_method": m, "folds": []}
        for m in methods
        for s in specs
    }
    folds = []
    for heldout in SEEDS:
        foldroot = root / f"fold_{heldout}"
        foldroot.mkdir()
        valid = np.array([r["seed"] == heldout for r in data["records"]])
        train = ~valid
        fold = {
            "schema": "ssvc-v4-whole-seed-cv-1",
            "status": "COMPLETED",
            "kind": "V4_SIGNATURE_WHOLE_SEED_FOLD",
            "config_hash": canonical_hash(config),
            "campaign_id": data["campaign_id"],
            "fixture": fixture,
            "dataset": dataset_binding,
            "train_seeds": [s for s in SEEDS if s != heldout],
            "validation_seeds": [heldout],
            "training_rows": int(train.sum()),
            "validation_rows": int(valid.sum()),
            "query_labels_read": False,
            "reference_labels_read": False,
            "models": [],
        }
        for row in model_rows.values():
            spec = row["spec"]
            m = row["observation_method"]
            record = {"id": row["id"], "status": "AVAILABLE"}
            if (
                not train.any()
                or not valid.any()
                or not any(not r["is_alias"] and r["seed"] == heldout for r in data["records"])
                or not np.any(data["signatures"][train])
            ):
                record.update(
                    status="NOT_ELIGIBLE",
                    reason="UNINFORMATIVE_TRAIN_OR_WHOLE_SEED_VALIDATION_POOL",
                )
                fold["models"].append(record)
                row["folds"].append({"validation_seed": heldout, **record})
                continue
            if spec["model"] == "NONLINEAR_SIGNATURE_RESIDUAL":
                count = sum(not r["is_alias"] and r["seed"] != heldout for r in data["records"])
                origin_count = len({r["origin_id"] for r in data["records"]})
                if count < 512 or origin_count < 6:
                    record.update(
                        status="NOT_ELIGIBLE",
                        reason="INSUFFICIENT_UNIQUE_NONALIAS_TRAINING_PAIRS_OR_DEVELOPMENT_ORIGINS",
                        actual_training_pairs=count,
                        required=512,
                        actual_development_origins=origin_count,
                        required_development_origins=6,
                    )
                    fold["models"].append(record)
                    row["folds"].append({"validation_seed": heldout, **record})
                    continue
                validate_training_partition(
                    data["signatures"],
                    data["state"][spec["state_mode"]],
                    data["responses"][m],
                    data["records"],
                    validation_seed=heldout,
                    **_partition_args(data),
                )
            started = time.perf_counter()
            model = _fit(data, spec, m, validation_seed=heldout, epochs=epochs)
            fit_seconds = time.perf_counter() - started
            prediction = (
                model["predict"](
                    data["signatures"][valid], data["state"][spec["state_mode"]][valid]
                )
                if spec["model"] == "NONLINEAR_SIGNATURE_RESIDUAL"
                else model["predict"](data["signatures"][valid])
            )
            modelroot = foldroot / canonical_hash(row["id"])
            binding = save_signature_model(
                model, modelroot, output_shape=data["responses"][m].shape[1:]
            )
            atomic_npz(
                foldroot / f"{canonical_hash(row['id'])}_PREDICTIONS.npz",
                {
                    "predictions": prediction,
                    "truth": data["responses"][m][valid],
                    "validation_indices": np.flatnonzero(valid),
                },
            )
            record.update(
                model=binding,
                predictions=_binding(foldroot / f"{canonical_hash(row['id'])}_PREDICTIONS.npz"),
                metrics=_metrics(prediction, data["responses"][m][valid], data["probe_groups"]),
                fit_seconds=fit_seconds,
                fit_predict_publish_seconds=time.perf_counter() - started,
            )
            fold["models"].append(record)
            row["folds"].append({"validation_seed": heldout, **record})
            del model
        shared = {
            "dataset": dataset_binding,
            "train_seeds": fold["train_seeds"],
            "validation_seeds": [heldout],
            "config_hash": canonical_hash(config),
            "campaign_id": data["campaign_id"],
            "fixture": fixture,
            "query_labels_read": False,
            "reference_labels_read": False,
        }
        atomic_json(
            foldroot / "FIT_RECEIPT.json",
            {
                "kind": "V4_SIGNATURE_CV_FITS",
                **shared,
                "training_indices": np.flatnonzero(train).tolist(),
                "models": [
                    {k: v for k, v in r.items() if k not in {"predictions", "metrics"}}
                    for r in fold["models"]
                ],
            },
        )
        atomic_json(
            foldroot / "EVALUATION_RECEIPT.json",
            {
                "kind": "V4_SIGNATURE_CV_EVALUATION",
                **shared,
                "validation_indices": np.flatnonzero(valid).tolist(),
                "evaluation_scope": "HELDOUT_DEVELOPMENT_SEED_CALIBRATION_OBSERVATIONS",
                "models": [{k: v for k, v in r.items() if k != "model"} for r in fold["models"]],
            },
        )
        fold.update(
            fit_receipt=_binding(foldroot / "FIT_RECEIPT.json"),
            evaluation_receipt=_binding(foldroot / "EVALUATION_RECEIPT.json"),
        )
        atomic_json(foldroot / "FOLD.json", fold)
        finalize_run(foldroot, {"dataset": dataset_binding, "validation_seed": heldout})
        folds.append(
            {
                "train_seeds": fold["train_seeds"],
                "validation_seeds": [heldout],
                "receipt": _binding(foldroot / "FOLD.json"),
            }
        )
    for row in model_rows.values():
        row["status"] = (
            "AVAILABLE" if all(r["status"] == "AVAILABLE" for r in row["folds"]) else "NOT_ELIGIBLE"
        )
        if row["status"] == "AVAILABLE":
            row["mean_seed_cv"] = {
                key: float(np.mean([r["metrics"][key] for r in row["folds"]]))
                for key in ("raw4_mse", "group_Xv_mse")
            }
    report = {
        "kind": "V4_SIGNATURE_DEVELOPMENT_CV",
        "status": "COMPLETE",
        "fixture": fixture,
        "config_hash": canonical_hash(config),
        "campaign_id": data["campaign_id"],
        "source_hash": data["source_hash"],
        "dataset": dataset_binding,
        "whole_seed_cv": folds,
        "models": list(model_rows.values()),
        "original_bindings": data["original_bindings"],
        "deduplicated_units": data["deduplicated_units"],
        "query_labels_read": False,
        "reference_labels_read": False,
        "cross_seed_95": "NOT_CERTIFIED",
        "network_initializations_are_training_seeds": False,
    }
    atomic_json(root / "SIGNATURE_CV.json", report)
    finalize_run(root, {"dataset": dataset_binding, "fixture": fixture})
    return report


def _reload_dataset(binding):
    meta = bound_json(binding)
    arrays = _arrays(meta["arrays"])
    return {
        **meta,
        "signatures": arrays["signatures"],
        "state": {k: arrays[f"state_{k}"] for k in ("EVENT_ONLY", "EVENT_PLUS_DIAGNOSTICS")},
        "responses": {m: arrays[f"response_{m}"] for m in meta["observation_methods"]},
    }


def freeze_signature_models(report_binding, model_ids, *, out, fixture=False):
    """Refit chosen measured CV specifications on the full six-origin development pool."""
    _gate(fixture)
    report, binding = _complete_document(report_binding)
    if (
        report.get("kind") != "V4_SIGNATURE_DEVELOPMENT_CV"
        or report.get("fixture") is not fixture
        or report.get("query_labels_read") is not False
        or report.get("reference_labels_read") is not False
    ):
        raise ValueError("Actual same-scope development CV report required")
    ids = list(model_ids)
    models = {r["id"]: r for r in report["models"]}
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(k not in models or models[k]["status"] != "AVAILABLE" for k in ids)
    ):
        raise ValueError("Only actually AVAILABLE CV specifications may be frozen")
    data = _reload_dataset(report["dataset"])
    if data["fixture"] is not fixture or (
        not fixture and data["source_hash"] != source_identity()["sha256"]
    ):
        raise ValueError("Frozen development source differs")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    selected = []
    for key in ids:
        row = models[key]
        started = time.perf_counter()
        model = _fit(data, row["spec"], row["observation_method"], epochs=data["epochs"])
        binding_model = save_signature_model(
            model,
            root / canonical_hash(key),
            output_shape=data["responses"][row["observation_method"]].shape[1:],
        )
        selected.append(
            {
                "id": key,
                "spec": row["spec"],
                "observation_method": row["observation_method"],
                "model": binding_model,
                "fit_seconds": time.perf_counter() - started,
                "network_initializations": 3
                if row["spec"]["model"] == "NONLINEAR_SIGNATURE_RESIDUAL"
                else None,
            }
        )
        del model
    lock = {
        "kind": "V4_SIGNATURE_SELECTION",
        "status": "FROZEN",
        "fixture": fixture,
        "config_hash": report["config_hash"],
        "source_hash": report["source_hash"],
        "development_report": binding,
        "dataset": report["dataset"],
        "models": selected,
        "fit_scope": "ALL_REGISTERED_DEVELOPMENT_ORIGINS",
        "development_seeds": list(SEEDS),
        "whole_seed_cv": report["whole_seed_cv"],
        "prompt_ids": data["prompt_ids"],
        "probe_groups": data["probe_groups"],
        "query_labels_read": False,
        "reference_labels_read": False,
        "cross_seed_95": "NOT_CERTIFIED",
    }
    lock["selection_hash"] = canonical_hash(lock)
    atomic_json(root / "SIGNATURE_SELECTION.json", lock)
    finalize_run(root, {"report": binding, "fixture": fixture})
    return lock


def verify_selected_signature_model(selected, model):
    """A selected label cannot relabel different actual frozen coefficients/weights."""
    spec, metadata = selected["spec"], model["metadata"]
    actual = metadata["metadata"]
    method = spec.get("model")
    if (
        method != metadata.get("method")
        or method != actual.get("method")
        or selected.get("observation_method") not in PRIMARY_METHODS
    ):
        raise ValueError("Selected specification differs from actual frozen model")
    if method == "NONLINEAR_SIGNATURE_RESIDUAL":
        widths = {"EVENT_ONLY": 36 * 4, "EVENT_PLUS_DIAGNOSTICS": 36 * 11}
        if spec.get("hidden_width") != actual.get("hidden_width") or widths.get(
            spec.get("state_mode")
        ) != actual.get("state_feature_width"):
            raise ValueError("Selected architecture differs from actual frozen network")
    elif method in {"FULL_DUAL_RIDGE", "FULL_RBF_RAW"}:
        if (
            spec.get("alpha") != actual.get("alpha")
            or spec.get("alpha") != actual.get("ridge_lambda")
            or actual.get("ridge_override") is not False
            or actual.get("rank_cap") != "FULL"
            or (
                method == "FULL_RBF_RAW"
                and spec.get("bandwidth_multiplier") != actual.get("bandwidth_multiplier")
            )
        ):
            raise ValueError("Selected kernel settings differ from actual frozen model")
    else:
        raise ValueError("Unknown actual frozen signature model")
    return {
        "id": selected["id"],
        "spec": spec,
        "observation_method": selected["observation_method"],
        "model": selected.get("model"),
        "actual_metadata": actual,
    }


def predict_signature_models(selection_binding, signature_binding, *, out, fixture=False):
    """No label input: load final development weights and publish immutable predictions."""
    _gate(fixture)
    lock, lb = _complete_document(selection_binding)
    if (
        lock.get("kind") != "V4_SIGNATURE_SELECTION"
        or lock.get("fixture") is not fixture
        or lock.get("selection_hash")
        != canonical_hash({k: v for k, v in lock.items() if k != "selection_hash"})
    ):
        raise ValueError("Frozen signature selection identity differs")
    signature, sb, arrays, _, _ = _signature(signature_binding, fixture)
    if any(signature["runtime_identity"].get(k) != lock[k] for k in ("config_hash", "source_hash")):
        raise ValueError("Independent prediction source/config differs")
    predictions, model_specs = [], []
    for selected in lock["models"]:
        _doc(selected["model"])
        model = load_signature_model(Path(selected["model"]["path"]).parent)
        model_specs.append(verify_selected_signature_model(selected, model))
        state = np.repeat(
            arrays[selected["spec"].get("state_mode", "EVENT_ONLY")][None],
            len(arrays["native_signatures"]),
            axis=0,
        )
        predictions.append(model["predict"](arrays["native_signatures"], state))
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    atomic_npz(root / "PREDICTIONS.npz", {"predictions": np.asarray(predictions)})
    receipt = {
        "kind": "V4_FROZEN_SIGNATURE_PREDICTIONS",
        "fixture": fixture,
        "selection": lb,
        "signature": sb,
        "origin_id": signature["origin_id"],
        "units": signature["units"],
        "model_ids": [m["id"] for m in lock["models"]],
        "model_specs": model_specs,
        "prompt_ids": lock["prompt_ids"],
        "probe_groups": lock["probe_groups"],
        "arrays": _binding(root / "PREDICTIONS.npz"),
        "query_labels_read": False,
        "reference_labels_read": False,
        "fit_or_refit_performed": False,
    }
    atomic_json(root / "PREDICTION_LOCK.json", receipt)
    finalize_run(root, {"selection": lb, "signature": sb})
    return receipt
