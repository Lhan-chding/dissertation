"""Full-input kernels with absolute ridge penalties and no reference access.

Gram matrices are sample-by-sample. Neither raw dual ridge nor the effective
LoRA trace kernel constructs a parameter-by-parameter or dense weight matrix.
Numerical rank is a diagnostic; FULL never discards small positive eigenmodes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite matrix with nonempty feature axis")
    return value


def _blocks(value, blocks=None, *, copy=False):
    if isinstance(value, Mapping):
        if blocks is not None or not value or any(not isinstance(k, str) for k in value):
            raise ValueError(
                "named input blocks must be nonempty and cannot have another partition"
            )
        result = {}
        for name, array in sorted(value.items()):
            array = np.asarray(array, dtype=np.float64)
            if array.ndim < 2:
                raise ValueError("each block requires a sample and feature axis")
            result[name] = _matrix(array.reshape(len(array), int(np.prod(array.shape[1:]))), name)
    else:
        array = _matrix(value, "updates")
        blocks = {"full": [0, array.shape[1]]} if blocks is None else blocks
        if not isinstance(blocks, Mapping) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(type(v) is not int for v in bounds)
            for name, bounds in blocks.items()
        ):
            raise ValueError("named integer block intervals required")
        intervals = sorted((start, stop, name) for name, (start, stop) in blocks.items())
        if (
            not intervals
            or intervals[0][0] != 0
            or intervals[-1][1] != array.shape[1]
            or any(
                start >= stop or (index and start != intervals[index - 1][1])
                for index, (start, stop, _) in enumerate(intervals)
            )
        ):
            raise ValueError("blocks must partition every input coordinate exactly once")
        result = {name: array[:, start:stop] for start, stop, name in intervals}
    if len({len(array) for array in result.values()}) != 1:
        raise ValueError("input block sample axes differ")
    return {name: array.copy() if copy else array for name, array in result.items()}


def fit_block_scales(updates):
    """RMS L2 norm per block from fit rows; preserve zero blocks with scale 1."""
    values = _blocks(updates)
    if not len(next(iter(values.values()))):
        raise ValueError("nonempty fit data required for block scales")
    return {
        name: float(np.sqrt(np.square(x).sum(axis=1).mean())) or 1.0 for name, x in values.items()
    }


def raw_gram(left, right=None, *, scales=None):
    left = _blocks(left)
    right = left if right is None else _blocks(right)
    if set(left) != set(right) or any(left[name].shape[1] != right[name].shape[1] for name in left):
        raise ValueError("input block identities or dimensions differ")
    scales = dict.fromkeys(left, 1.0) if scales is None else scales
    if set(scales) != set(left) or any(not np.isfinite(s) or s <= 0 for s in scales.values()):
        raise ValueError("positive fit-only scales for every block required")
    gram = np.zeros((len(next(iter(left.values()))), len(next(iter(right.values())))))
    for name, array in left.items():
        gram += (array / scales[name]) @ (right[name] / scales[name]).T
    return gram


def _raw_norms(blocks, scales):
    return sum(np.square(value / scales[name]).sum(axis=1) for name, value in blocks.items())


def _spectrum(gram):
    gram = _matrix(gram, "Gram")
    if gram.shape[0] != gram.shape[1] or not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-13):
        raise ValueError("symmetric square Gram required")
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if eigenvalues[0] < -1e-9 * scale:
        raise ValueError("Gram is not positive semidefinite within roundoff")
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    vectors = vectors[:, ::-1]
    threshold = max(1e-28, float(eigenvalues[0]) * 1e-10)
    return eigenvalues, vectors, eigenvalues > threshold, threshold


def anchored_rbf_from_gram(cross, left_norms, right_norms, *, length_scale):
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("positive finite RBF length scale required")
    cross = np.asarray(cross, dtype=np.float64)
    left, right = np.asarray(left_norms, float), np.asarray(right_norms, float)
    if (
        cross.shape != (len(left), len(right))
        or not all(np.isfinite(v).all() for v in (cross, left, right))
        or np.any(left < 0)
        or np.any(right < 0)
    ):
        raise ValueError("aligned finite Gram and nonnegative squared norms required")
    gamma = 0.5 / length_scale**2
    distance = np.maximum(left[:, None] + right[None, :] - 2 * cross, 0)
    kernel = (
        np.exp(-gamma * distance)
        - np.exp(-gamma * left[:, None])
        - np.exp(-gamma * right[None, :])
        + 1
    )
    kernel[left == 0] = 0
    kernel[:, right == 0] = 0
    return kernel


def _fit(gram, responses, query_inner, *, method, alpha, rank_cap, bandwidth_multiplier, ridge):
    y = np.asarray(responses, dtype=np.float64).copy()
    if (
        y.ndim < 2
        or y.shape[0] != len(gram)
        or not len(y)
        or not np.isfinite(y).all()
        or not np.prod(y.shape[1:])
    ):
        raise ValueError("finite responses with one row per fit update required")
    lam = float(alpha if ridge is None else ridge)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("absolute ridge lambda must be positive and finite")
    if rank_cap != "FULL" and (type(rank_cap) is not int or rank_cap <= 0):
        raise ValueError("rank_cap must be a positive integer or FULL")
    eigenvalues, vectors, keep, threshold = _spectrum(gram)
    k = int(keep.sum())
    r = k if rank_cap == "FULL" else min(rank_cap, k)
    linear = method in {"FULL_DUAL_RIDGE", "FULL_EFFECTIVE_WEIGHT_RIDGE"}
    if not linear and rank_cap != "FULL":
        raise ValueError("rank compression is a linear baseline, not the full RBF")
    fit_norms = np.maximum(np.diag(gram), 0)
    metadata = {
        "method": method,
        "alpha": float(alpha),
        "ridge_lambda": lam,
        "ridge_parameterization": "ABSOLUTE_PARAMETER_PENALTY"
        if linear
        else "ABSOLUTE_RKHS_PENALTY",
        "ridge_override": ridge is not None,
        "k": k,
        "r": r,
        "rank_cap": rank_cap,
        "rank_label": "TRUE_COMPRESSION"
        if 0 < r < k
        else "FULL"
        if rank_cap == "FULL"
        else "FULL_EQUIVALENT",
        "rank_eigenvalue_threshold": threshold,
        "rank_threshold_scope": "DIAGNOSTIC_ONLY_FOR_FULL",
        "full_positive_modes_truncated": False,
        "positive_modes_truncated": bool(r < k),
        "compression_basis": "INPUT_GRAM_PRINCIPAL_COMPONENTS" if r < k else "NONE",
        "response_svd_used": False,
        "output_policy": "RAW_UNCONSTRAINED",
        "query_labels_read": False,
        "intercept": False,
    }
    length_scale = None
    if linear:
        solver_gram = gram
        active = np.arange(r) if r < k else np.arange(len(eigenvalues))
        dual = vectors[:, active] @ (
            (vectors[:, active].T @ y.reshape(len(y), -1)) / (eigenvalues[active, None] + lam)
        )
    else:
        if method not in {"FULL_RBF_RAW", "LEGACY_QPROJECTED_RBF", "FULL_RBF_EFFECTIVE_WEIGHT"}:
            raise ValueError("unknown full response model")
        if not np.isfinite(bandwidth_multiplier) or bandwidth_multiplier <= 0:
            raise ValueError("positive bandwidth multiplier required")
        kernel_input_gram = (
            (vectors[:, keep] * eigenvalues[keep]) @ vectors[:, keep].T
            if method == "LEGACY_QPROJECTED_RBF"
            else gram
        )
        kernel_fit_norms = np.maximum(np.diag(kernel_input_gram), 0)
        distances = np.sqrt(
            np.maximum(
                kernel_fit_norms[:, None] + kernel_fit_norms[None, :] - 2 * kernel_input_gram, 0
            )
        )
        positive = distances[np.triu_indices(len(gram), 1)]
        positive = positive[positive > 0]
        fallback = None
        if not len(positive):
            positive = np.sqrt(kernel_fit_norms[kernel_fit_norms > 0])
            fallback = "FIT_TO_ZERO_DISTANCES" if len(positive) else "NO_NONZERO_FIT_DISTANCE"
        length_scale = float(np.median(positive) * bandwidth_multiplier) if len(positive) else 1.0
        solver_gram = anchored_rbf_from_gram(
            kernel_input_gram, kernel_fit_norms, kernel_fit_norms, length_scale=length_scale
        )
        eig_kernel, u_kernel, _, _ = _spectrum(solver_gram)
        dual = u_kernel @ ((u_kernel.T @ y.reshape(len(y), -1)) / (eig_kernel[:, None] + lam))
        metadata.update(
            length_scale=length_scale,
            bandwidth_multiplier=float(bandwidth_multiplier),
            bandwidth_fallback=fallback,
            rbf_definition="exp(-squared_distance/(2*length_scale**2))",
            rank_label="LEGACY_PROJECTED_NONLINEAR"
            if method == "LEGACY_QPROJECTED_RBF"
            else "FULL_INPUT_NONLINEAR",
        )

    def geometry_from(cross, norm):
        coordinates = cross @ vectors[:, keep]
        projected = (
            np.sum(np.square(coordinates) / eigenvalues[keep], axis=1)
            if k
            else np.zeros(len(cross))
        )
        perpendicular = np.maximum(norm - projected, 0)
        return {
            "k": k,
            "r": r,
            "rank_label": metadata["rank_label"],
            "e_norm": np.sqrt(norm),
            "e_perp_norm": np.sqrt(perpendicular),
            "rho": np.sqrt(np.divide(perpendicular, norm, out=np.zeros_like(norm), where=norm > 0)),
            "leverage": np.sum(
                np.square(coordinates) / (eigenvalues[keep] * (eigenvalues[keep] + lam)), axis=1
            )
            if k
            else np.zeros(len(cross)),
            "leverage_scope": "TRAINING_SPAN_RIDGE",
            "geometry_is_acceptance_gate": False,
            "condition_number": float(np.sqrt(eigenvalues[0] / eigenvalues[k - 1])) if k else None,
        }

    def predict(query):
        cross, norm = query_inner(query)
        kernel = cross
        if not linear:
            kernel_norm = norm
            if method == "LEGACY_QPROJECTED_RBF":
                kernel_norm = (
                    np.sum(np.square(cross @ vectors[:, keep]) / eigenvalues[keep], axis=1)
                    if k
                    else np.zeros(len(cross))
                )
                cross = (cross @ vectors[:, keep]) @ vectors[:, keep].T
            kernel = anchored_rbf_from_gram(
                cross, kernel_norm, kernel_fit_norms, length_scale=length_scale
            )
        prediction = kernel @ dual
        if not np.any(fit_norms > 0):
            prediction[norm > 0] = np.nan
        prediction[norm == 0] = 0
        return prediction.reshape((len(cross), *y.shape[1:]))

    return {
        "predict": predict,
        "geometry": lambda query: geometry_from(*query_inner(query)),
        "metadata": metadata,
        "gram": solver_gram,
        "input_gram": gram,
        "dual_coefficients": dual,
        "k": k,
        "r": r,
        "status": "FIT_AVAILABLE_NOT_VALIDATED"
        if np.any(fit_norms > 0)
        else "NO_CALIBRATION_EXCITATION",
    }


def fit_full_response(
    updates,
    responses,
    *,
    method="FULL_DUAL_RIDGE",
    alpha=1e-5,
    rank_cap="FULL",
    blocks=None,
    standardize_blocks=False,
    bandwidth_multiplier=1.0,
    ridge=None,
):
    """Fit from complete arrays or named parameter blocks; alpha is absolute λ."""
    if method not in {"FULL_DUAL_RIDGE", "FULL_RBF_RAW", "LEGACY_QPROJECTED_RBF"}:
        raise ValueError("raw coordinates require an explicitly named raw-input model")
    fit = _blocks(updates, blocks, copy=True)
    blocks = {name: list(bounds) for name, bounds in blocks.items()} if blocks is not None else None
    if not len(next(iter(fit.values()))):
        raise ValueError("nonempty calibration updates required")
    scales = fit_block_scales(fit) if standardize_blocks else dict.fromkeys(fit, 1.0)

    def query_inner(query):
        query = _blocks(query, blocks if not isinstance(query, Mapping) else None)
        return raw_gram(query, fit, scales=scales), _raw_norms(query, scales)

    model = _fit(
        raw_gram(fit, scales=scales),
        responses,
        query_inner,
        method=method,
        alpha=alpha,
        rank_cap=rank_cap,
        bandwidth_multiplier=bandwidth_multiplier,
        ridge=ridge,
    )
    model["metadata"].update(
        input_representation="FULL_RAW_PARAMETER_BLOCKS",
        block_standardized=bool(standardize_blocks),
        block_scales=dict(scales),
        block_scale_definition=(
            "fit-only sqrt(mean squared L2 block norm); zero block scale=1; no centering"
        ),
        input_dimension=sum(x.shape[1] for x in fit.values()),
    )
    model["_serialization"] = {
        "kind": "raw",
        "fit": fit,
        "responses": np.asarray(responses).copy(),
        "settings": {
            "method": method,
            "alpha": alpha,
            "rank_cap": rank_cap,
            "standardize_blocks": standardize_blocks,
            "bandwidth_multiplier": bandwidth_multiplier,
            "ridge": ridge,
        },
        "array_blocks": blocks,
    }
    return model


def product_inner(b1, a1, b2, a2):
    b1, a1, b2, a2 = [_matrix(value, "LoRA factor") for value in (b1, a1, b2, a2)]
    if (
        b1.shape[1] != a1.shape[0]
        or b2.shape[1] != a2.shape[0]
        or b1.shape[0] != b2.shape[0]
        or a1.shape[1] != a2.shape[1]
    ):
        raise ValueError("incompatible low-rank product shapes")
    return float(np.einsum("ij,ji->", b1.T @ b2, a2 @ a1.T))


def _contrast(value, *, copy=False):
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("base_model_id"), str)
        or not value["base_model_id"]
        or not isinstance(value.get("modules"), Mapping)
        or not value["modules"]
    ):
        raise ValueError("effective contrasts require base identity and named modules")
    modules = {}
    for name, layer in sorted(value["modules"].items()):
        scale = float(layer["scaling"])
        if not isinstance(name, str) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("module identity and positive scaling required")
        endpoints = {}
        for endpoint in ("candidate", "baseline"):
            b, a = (_matrix(layer[endpoint][key], "LoRA factor") for key in ("B", "A"))
            if b.shape[1] != a.shape[0]:
                raise ValueError("LoRA factor rank mismatch")
            endpoints[endpoint] = {"B": b.copy() if copy else b, "A": a.copy() if copy else a}
        if any(
            endpoints["candidate"][key].shape != endpoints["baseline"][key].shape
            for key in ("A", "B")
        ):
            raise ValueError("completed endpoints must share module factor shapes")
        modules[name] = {"scaling": scale, **endpoints}
    return {"base_model_id": value["base_model_id"], "modules": modules}


def _signature(value):
    return value["base_model_id"], tuple(
        (name, layer["scaling"], layer["candidate"]["A"].shape, layer["candidate"]["B"].shape)
        for name, layer in value["modules"].items()
    )


def _terms(layer):
    # Stable near identical endpoints; never form the out_features by in_features product.
    c, b, scale = layer["candidate"], layer["baseline"], layer["scaling"]
    return ((scale * (c["B"] - b["B"]), c["A"]), (scale * b["B"], c["A"] - b["A"]))


def _effective_inner(left, right):
    if _signature(left) != _signature(right):
        raise ValueError("base/module/scaling identity differs between effective contrasts")
    return sum(
        product_inner(b1, a1, b2, a2)
        for name in left["modules"]
        for b1, a1 in _terms(left["modules"][name])
        for b2, a2 in _terms(right["modules"][name])
    )


def effective_delta_inner(left, right):
    return _effective_inner(_contrast(left), _contrast(right))


def effective_weight_gram(left, right=None):
    left = [_contrast(value) for value in left]
    right = left if right is None else [_contrast(value) for value in right]
    if not left or not right:
        raise ValueError("nonempty effective contrast samples required")
    return np.asarray([[_effective_inner(a, b) for b in right] for a in left])


def fit_effective_response(
    contrasts,
    responses,
    *,
    method="FULL_EFFECTIVE_WEIGHT_RIDGE",
    alpha=1e-5,
    bandwidth_multiplier=1.0,
    ridge=None,
):
    if method not in {"FULL_EFFECTIVE_WEIGHT_RIDGE", "FULL_RBF_EFFECTIVE_WEIGHT"}:
        raise ValueError("effective weights require explicitly named effective models")
    fit = [_contrast(value, copy=True) for value in contrasts]
    gram = effective_weight_gram(fit)

    def query_inner(query):
        query = [_contrast(value) for value in query]
        cross = effective_weight_gram(query, fit)
        norms = np.asarray([max(_effective_inner(value, value), 0.0) for value in query])
        return cross, norms

    model = _fit(
        gram,
        responses,
        query_inner,
        method=method,
        alpha=alpha,
        rank_cap="FULL",
        bandwidth_multiplier=bandwidth_multiplier,
        ridge=ridge,
    )
    model["metadata"].update(
        input_representation="EFFECTIVE_ADAPTER_WEIGHT_CONTRAST",
        dense_weight_materialized=False,
        gauge_invariant_representation=True,
        training_dynamics_gauge_invariant_claim=False,
    )
    model["_serialization"] = {
        "kind": "effective",
        "fit": fit,
        "responses": np.asarray(responses).copy(),
        "settings": {
            "method": method,
            "alpha": alpha,
            "bandwidth_multiplier": bandwidth_multiplier,
            "ridge": ridge,
        },
    }
    return model


def fit_precomputed_response(
    input_gram,
    responses,
    *,
    query_cross_gram,
    query_norms,
    method="FULL_DUAL_RIDGE",
    alpha=1e-5,
    rank_cap="FULL",
    bandwidth_multiplier=1.0,
    input_metadata=None,
):
    """Fit the same full model from block-streamed sufficient inner products.

    Input Gram is E E^T; cross Gram is query E^T; query_norms are squared
    complete input norms (including directions outside the fit span). No query
    responses are accepted. ``predict(indices=None)`` and ``geometry`` operate
    on the registered query rows. This avoids retaining dense N-by-D LoRA
    arrays for every hyperparameter while retaining every coordinate in D.
    """
    allowed = {
        "FULL_DUAL_RIDGE",
        "FULL_EFFECTIVE_WEIGHT_RIDGE",
        "FULL_RBF_RAW",
        "FULL_RBF_EFFECTIVE_WEIGHT",
        "LEGACY_QPROJECTED_RBF",
    }
    if method not in allowed:
        raise ValueError("An explicit full-input kernel method is required")
    gram = _matrix(input_gram, "input Gram").copy()
    cross = _matrix(query_cross_gram, "query cross Gram").copy()
    norm = np.asarray(query_norms, dtype=np.float64).copy()
    if (
        gram.shape[0] != gram.shape[1]
        or cross.shape[1] != len(gram)
        or norm.shape != (len(cross),)
        or not np.isfinite(norm).all()
        or np.any(norm < 0)
    ):
        raise ValueError("Aligned complete fit/query inner products required")
    if np.any(cross[norm == 0] != 0):
        raise ValueError("Zero query directions require zero inner products")
    metadata = json.loads(json.dumps(input_metadata or {}, allow_nan=False))
    if not isinstance(metadata, dict):
        raise ValueError("Streamed input metadata must be a JSON record")

    def query_inner(indices=None):
        if indices is None:
            return cross, norm
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= len(cross))
        ):
            raise ValueError("Registered query row indices must be nonnegative integers")
        return cross[indices], norm[indices]

    model = _fit(
        gram,
        responses,
        query_inner,
        method=method,
        alpha=alpha,
        rank_cap=rank_cap,
        bandwidth_multiplier=bandwidth_multiplier,
        ridge=None,
    )
    original_predict, original_geometry = model["predict"], model["geometry"]
    model["predict"] = lambda indices=None: original_predict(indices)
    model["geometry"] = lambda indices=None: original_geometry(indices)
    model["metadata"].update(
        input_representation="PRECOMPUTED_COMPLETE_INPUT_INNER_PRODUCTS",
        complete_query_norms_retained=True,
        dense_parameter_arrays_retained=False,
        registered_query_count=len(cross),
        input_metadata=metadata,
    )
    model["_serialization"] = {
        "kind": "precomputed",
        "fit": gram,
        "responses": np.asarray(responses).copy(),
        "query_cross_gram": cross,
        "query_norms": norm,
        "settings": {
            "method": method,
            "alpha": alpha,
            "rank_cap": rank_cap,
            "bandwidth_multiplier": bandwidth_multiplier,
            "input_metadata": metadata,
        },
    }
    return model


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(model, out):
    """Lossless fit-data bundle; load performs declared refitting, without pickle."""
    state = model["_serialization"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arrays, manifest = (
        {"responses": state["responses"]},
        {
            "schema": "ssvc-v4-full-kernel-fit-1",
            "kind": state["kind"],
            "settings": state["settings"],
        },
    )
    if state["kind"] == "raw":
        manifest["blocks"] = list(state["fit"])
        manifest["array_blocks"] = state["array_blocks"]
        arrays.update({f"block_{i}": value for i, value in enumerate(state["fit"].values())})
    elif state["kind"] == "precomputed":
        arrays.update(
            input_gram=state["fit"],
            query_cross_gram=state["query_cross_gram"],
            query_norms=state["query_norms"],
        )
    else:
        manifest["contrasts"] = []
        for i, value in enumerate(state["fit"]):
            item = {"base_model_id": value["base_model_id"], "modules": {}}
            for j, (name, layer) in enumerate(value["modules"].items()):
                item["modules"][name] = {"scaling": layer["scaling"]}
                for endpoint in ("candidate", "baseline"):
                    item["modules"][name][endpoint] = {}
                    for factor in ("A", "B"):
                        key = f"c{i}_m{j}_{endpoint}_{factor}"
                        arrays[key] = layer[endpoint][factor]
                        item["modules"][name][endpoint][factor] = key
            manifest["contrasts"].append(item)
    with (out / "FIT.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest["fit_sha256"] = _file_hash(out / "FIT.npz")
    (out / "MODEL.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_model(root):
    root = Path(root)
    manifest = json.loads((root / "MODEL.json").read_text())
    if (
        manifest.get("schema") != "ssvc-v4-full-kernel-fit-1"
        or _file_hash(root / "FIT.npz") != manifest["fit_sha256"]
    ):
        raise ValueError("saved full kernel fit identity mismatch")
    with np.load(root / "FIT.npz", allow_pickle=False) as arrays:
        y = arrays["responses"].copy()
        if manifest["kind"] == "raw":
            fit = {name: arrays[f"block_{i}"].copy() for i, name in enumerate(manifest["blocks"])}
            if manifest["array_blocks"] is not None:
                original = np.concatenate(
                    [
                        fit[name]
                        for name, _ in sorted(
                            manifest["array_blocks"].items(), key=lambda item: item[1][0]
                        )
                    ],
                    axis=1,
                )
                model = fit_full_response(
                    original, y, blocks=manifest["array_blocks"], **manifest["settings"]
                )
            else:
                model = fit_full_response(
                    fit["full"] if manifest["blocks"] == ["full"] else fit,
                    y,
                    **manifest["settings"],
                )
        elif manifest["kind"] == "precomputed":
            model = fit_precomputed_response(
                arrays["input_gram"],
                y,
                query_cross_gram=arrays["query_cross_gram"],
                query_norms=arrays["query_norms"],
                **manifest["settings"],
            )
        elif manifest["kind"] == "effective":
            fit = manifest["contrasts"]
            for value in fit:
                for layer in value["modules"].values():
                    for endpoint in ("candidate", "baseline"):
                        layer[endpoint] = {
                            factor: arrays[key].copy() for factor, key in layer[endpoint].items()
                        }
            model = fit_effective_response(fit, y, **manifest["settings"])
        else:
            raise ValueError("unknown saved representation")
    model["metadata"]["load_mechanism"] = "REFIT_ON_LOAD"
    return model
