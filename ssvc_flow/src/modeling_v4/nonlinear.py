"""Paid native-signature residual networks with a whole-seed development split.

This module receives origin-only state, independent signature features and local
calibration responses. It never consumes cross-seed calibration or test labels.
Three network initializations are retained; they are not independent source runs.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _array(value, name, ndim=None):
    value = np.asarray(value, dtype=np.float64)
    if (ndim is not None and value.ndim != ndim) or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must have the declared nonempty finite axes")
    return value


def _hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _group_members(groups, prompt_count):
    if not isinstance(groups, (tuple, list)) or len(groups) != prompt_count:
        raise ValueError("one frozen semantic group per output prompt required")
    keys = []
    for group in groups:
        key = tuple(group) if isinstance(group, (list, tuple)) else group
        if (
            not isinstance(key, (str, tuple))
            or not key
            or (isinstance(key, tuple) and any(not isinstance(x, str) or not x for x in key))
        ):
            raise ValueError("semantic groups must be explicit nonempty names")
        keys.append(key)
    order = list(dict.fromkeys(keys))
    return order, [np.flatnonzero([key == group for key in keys]) for group in order]


def validate_training_partition(
    signatures,
    state_features,
    responses,
    records,
    *,
    development_seeds,
    validation_seed,
    signature_provenance,
    state_provenance,
    probe_groups,
    final_development_refit=False,
):
    """Fail before fitting if sample identities, feature scope or seed roles drift.

    The six-origin minimum applies to the entire development pool. The training
    fold must itself retain 512 distinct nonalias endpoint pairs after a whole
    seed is held out. Unique pairs do not imply independent IID observations.
    """
    e = _array(signatures, "signature", 2)
    if e.shape[1] != 576:
        raise ValueError("native signature must retain all 576 sequence scores")
    z = _array(state_features, "origin state")
    y = _array(responses, "response", 3)
    if z.ndim < 2 or y.shape[-1] != 4 or len(z) != len(e) or len(y) != len(e):
        raise ValueError("aligned signature/state/RAW4 response rows required")
    z = z.reshape(len(z), -1)
    if any(len(values) != len(e) for values in (records, signature_provenance, state_provenance)):
        raise ValueError("one actual identity and provenance record per fit row required")
    if (
        not isinstance(development_seeds, (list, tuple))
        or len(development_seeds) < 3
        or any(type(s) is not int for s in development_seeds)
        or len(set(development_seeds)) != len(development_seeds)
        or (
            not final_development_refit
            and (type(validation_seed) is not int or validation_seed not in development_seeds)
        )
        or (final_development_refit and validation_seed is not None)
    ):
        raise ValueError(
            "at least three unique development seeds and a whole heldout seed required"
        )
    _, groups = _group_members(probe_groups, y.shape[1])
    source_seeds, origins, identifiers, endpoint_pairs, aliases = [], {}, set(), set(), []
    panel_identities = {}
    for i, (record, signature, state) in enumerate(
        zip(records, signature_provenance, state_provenance, strict=True)
    ):
        if not all(isinstance(item, Mapping) for item in (record, signature, state)):
            raise ValueError("identity and feature provenance must be records")
        if (
            record.get("role") != "development"
            or record.get("bank_role") != "calibration"
            or type(record.get("seed")) is not int
            or record["seed"] not in development_seeds
        ):
            raise ValueError(
                "only registered development local calibration banks may train or validate"
            )
        origin, pair = record.get("origin_id"), record.get("pair_id")
        if (
            any(not isinstance(v, str) or not v for v in (origin, pair))
            or type(record.get("is_alias")) is not bool
        ):
            raise ValueError("actual origin, pair identity and fingerprint alias flag required")
        if (origin, pair) in identifiers:
            raise ValueError("duplicate candidate pair identity")
        identifiers.add((origin, pair))
        if (
            signature.get("native_width") != 576
            or signature.get("prompts") != 36
            or signature.get("draws_per_prompt") != 16
            or signature.get("origin_id") != origin
            or signature.get("panel_role") != "signature"
            or not signature.get("panel_id")
            or signature.get("split_guard") != "VERIFIED"
            or signature.get("score_definition") != "FULL_SEQUENCE_LOGPROB_DIFFERENCE"
            or signature.get("target_outcomes_used") is not False
            or signature.get("paid_feature") is not True
            or not _hash(signature.get("sample_identity_sha256"))
            or not _hash(signature.get("split_identity_sha256"))
        ):
            raise ValueError("independent native signature provenance is incomplete or unsafe")
        baseline, candidate = (
            signature.get("baseline_fingerprint"),
            signature.get("candidate_fingerprint"),
        )
        if (
            any(not isinstance(v, str) or not v for v in (baseline, candidate))
            or (baseline == candidate) != record["is_alias"]
        ):
            raise ValueError("alias status must come from exact endpoint fingerprints")
        # Reversing one physical endpoint pair changes only the contrast sign.
        # It cannot manufacture another unique training pair for the 512 gate.
        endpoints = tuple(sorted((baseline, candidate)))
        if endpoints in endpoint_pairs:
            raise ValueError("duplicate endpoint pair cannot increase training sample count")
        endpoint_pairs.add(endpoints)
        if record["is_alias"] and np.any(e[i] != 0):
            raise ValueError("identical endpoint signature must be exactly zero")
        paid = signature.get("paid_score_actions")
        if paid is not None and (type(paid) is not int or paid < 0):
            raise ValueError("paid signature actions require actual nonnegative ledger counts")
        if (
            state.get("stage") != "ORIGIN"
            or state.get("origin_id") != origin
            or state.get("panel_role") != "signature"
            or set(state) - {"stage", "origin_id", "panel_role", "panel_id", "feature_names"}
        ):
            raise ValueError("state features require origin-only signature-panel provenance")
        if state.get("panel_id", signature["panel_id"]) != signature["panel_id"]:
            raise ValueError("origin state and signature panels differ")
        panel_identity = (
            signature["panel_id"],
            signature["sample_identity_sha256"],
            signature["split_identity_sha256"],
        )
        if origin in panel_identities and panel_identities[origin] != panel_identity:
            raise ValueError("signature panel must remain fixed within an origin")
        panel_identities[origin] = panel_identity
        if origin in origins:
            first = origins[origin]
            if record["seed"] != records[first]["seed"] or not np.array_equal(z[i], z[first]):
                raise ValueError("origin state or source seed changed within the same origin")
        else:
            origins[origin] = i
        source_seeds.append(record["seed"])
        aliases.append(record["is_alias"])
    if set(source_seeds) != set(development_seeds) or len(origins) < 6:
        raise ValueError(
            "at least six actual development origins across every declared seed required"
        )
    validation = np.asarray(source_seeds) == validation_seed
    nonalias = ~np.asarray(aliases)
    training = ~validation
    if int(np.sum(training & nonalias)) < 512 or (
        not final_development_refit and not np.any(validation & nonalias)
    ):
        raise ValueError(
            "training fold requires 512 distinct nonalias pairs and heldout nonalias validation"
        )
    return {
        "training_indices": np.flatnonzero(training),
        "validation_indices": np.flatnonzero(validation),
        "nonalias_training_pair_count": int(np.sum(training & nonalias)),
        "nonalias_development_pair_count": int(nonalias.sum()),
        "development_origin_count": len(origins),
        "source_training_seeds": sorted(set(np.asarray(source_seeds)[training].tolist())),
        "validation_seed": validation_seed,
        "groups": groups,
    }


def _fit_signature_residual(
    signatures,
    state_features,
    responses,
    records,
    *,
    development_seeds,
    validation_seed,
    signature_provenance,
    state_provenance,
    probe_groups,
    hidden_width=256,
    network_seeds=(0, 1, 2),
    epochs=200,
    learning_rate=1e-3,
    weight_decay=1e-4,
    group_loss_weight=1.0,
    final_development_refit=False,
):
    """Fit linear(e) + g(z,e) - g(z,0); keep all three seed initializations.

    Architecture/epochs are caller-frozen. Validation labels are only evaluated
    after training; they never select epochs or initialization inside this API.
    Returned predictions average the three separately retained networks.
    """
    partition = validate_training_partition(
        signatures,
        state_features,
        responses,
        records,
        development_seeds=development_seeds,
        validation_seed=validation_seed,
        signature_provenance=signature_provenance,
        state_provenance=state_provenance,
        probe_groups=probe_groups,
        final_development_refit=final_development_refit,
    )
    if (
        type(hidden_width) is not int
        or hidden_width not in (256, 1024)
        or type(epochs) is not int
        or epochs < 1
    ):
        raise ValueError("registered hidden width 256/1024 and positive fixed epochs required")
    if (
        not isinstance(network_seeds, (list, tuple))
        or len(network_seeds) != 3
        or any(type(s) is not int or s < 0 for s in network_seeds)
        or len(set(network_seeds)) != 3
    ):
        raise ValueError("exactly three distinct network initialization seeds required")
    if (
        not np.isfinite(learning_rate)
        or learning_rate <= 0
        or weight_decay != 1e-4
        or not np.isfinite(group_loss_weight)
        or group_loss_weight <= 0
    ):
        raise ValueError("positive learning rate/group loss and fixed weight decay 1e-4 required")
    import torch
    from torch import nn

    e = np.asarray(signatures, dtype=np.float64).copy()
    z = np.asarray(state_features, dtype=np.float64).reshape(len(e), -1).copy()
    y = np.asarray(responses, dtype=np.float64).copy()
    train, valid = partition["training_indices"], partition["validation_indices"]
    e_scale = np.sqrt(np.square(e[train]).mean(axis=0))
    e_scale[e_scale == 0] = 1
    z_mean, z_scale = z[train].mean(axis=0), z[train].std(axis=0)
    z_scale[z_scale == 0] = 1
    normalized_e, normalized_z = e / e_scale, (z - z_mean) / z_scale
    tensors = [
        torch.as_tensor(value, dtype=torch.float64, device="cpu")
        for value in (normalized_e, normalized_z, y)
    ]
    group_indices = [torch.as_tensor(index, dtype=torch.long) for index in partition["groups"]]

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(e.shape[1], int(np.prod(y.shape[1:])), bias=False)
            self.residual = nn.Sequential(
                nn.Linear(e.shape[1] + z.shape[1], hidden_width),
                nn.Tanh(),
                nn.Linear(hidden_width, hidden_width),
                nn.Tanh(),
                nn.Linear(hidden_width, int(np.prod(y.shape[1:]))),
            )

        def forward(self, signature, state):
            output = (
                self.linear(signature)
                + self.residual(torch.cat((state, signature), dim=1))
                - self.residual(torch.cat((state, torch.zeros_like(signature)), dim=1))
            )
            return output.reshape(len(signature), *y.shape[1:])

    def losses(prediction, truth):
        residual = prediction - truth
        raw = torch.square(residual).mean()
        grouped = torch.stack(
            [
                torch.stack(
                    (residual[:, indices, 0].mean(dim=1), -residual[:, indices, 3].mean(dim=1)),
                    dim=-1,
                )
                for indices in group_indices
            ],
            dim=1,
        )
        return raw, torch.square(grouped).mean()

    networks, histories, validation_metrics = [], [], []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in network_seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                network = Residual().to(dtype=torch.float64, device="cpu")
                optimizer = torch.optim.AdamW(
                    network.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                history = []
                network.train()
                for _ in range(epochs):
                    optimizer.zero_grad(set_to_none=True)
                    raw, group = losses(
                        network(tensors[0][train], tensors[1][train]), tensors[2][train]
                    )
                    loss = raw + group_loss_weight * group
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite network loss; no trained model returned")
                    loss.backward()
                    optimizer.step()
                    history.append(float(loss.detach()))
                network.eval()
                if len(valid):
                    with torch.no_grad():
                        raw, group = losses(
                            network(tensors[0][valid], tensors[1][valid]), tensors[2][valid]
                        )
                    validation_metrics.append(
                        {
                            "network_seed": seed,
                            "raw4_mse": float(raw),
                            "group_Xv_mse": float(group),
                            "heldout_source_seed": validation_seed,
                            "row_count": len(valid),
                        }
                    )
                histories.append(history)
                networks.append(network)
    finally:
        torch.set_num_threads(previous_threads)

    def predict_members(query_signatures, query_state):
        query_e = _array(query_signatures, "query signature", 2)
        query_z = _array(query_state, "query origin state")
        if query_e.shape[1] != 576 or query_z.ndim < 2 or len(query_e) != len(query_z):
            raise ValueError("query signature and origin state dimensions differ")
        query_z = query_z.reshape(len(query_z), -1)
        if query_z.shape[1] != z.shape[1]:
            raise ValueError("query origin state dimension differs")
        qe = torch.as_tensor(query_e / e_scale, dtype=torch.float64, device="cpu")
        qz = torch.as_tensor((query_z - z_mean) / z_scale, dtype=torch.float64, device="cpu")
        with torch.no_grad():
            return np.stack([network(qe, qz).numpy() for network in networks])

    metadata = {
        "method": "NONLINEAR_SIGNATURE_RESIDUAL",
        "architecture": "linear(e)+g(z,e)-g(z,0)",
        "hidden_layers": 2,
        "hidden_width": hidden_width,
        "native_signature_width": 576,
        "response_svd_used": False,
        "state_feature_width": z.shape[1],
        "output_policy": "RAW_UNCONSTRAINED",
        "group_channels": ["delta_pX", "delta_v=-delta_pI"],
        "group_weighting": "EQUAL_GROUP_MEAN_EQUAL_PROMPT_WITHIN_GROUP",
        "group_loss_weight": group_loss_weight,
        "network_seeds": list(network_seeds),
        "network_initializations_are_source_seeds": False,
        "source_training_seed_count": len(partition["source_training_seeds"]),
        "source_training_seeds": partition["source_training_seeds"],
        "validation_seed": validation_seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dtype": "float64",
        "device": "cpu",
        "zero_anchor": True,
        "paid_feature": True,
        "paid_score_actions_per_record": [
            item.get("paid_score_actions") for item in signature_provenance
        ],
        "paid_counts_deduplicated": False,
        "predictions": "MEAN_OF_THREE_INITIALIZATIONS_NOT_A_CONFIDENCE_INTERVAL",
        "query_labels_read": False,
        "validation_used_for_gradient_or_early_stopping": False,
        "preprocessing_fit_scope": "TRAINING_SEEDS_ONLY",
        "fit_scope": "ALL_REGISTERED_DEVELOPMENT_ORIGINS"
        if final_development_refit
        else "WHOLE_SEED_CV_TRAINING_FOLD",
        "status": "FIT_AVAILABLE_NOT_VALIDATED",
    }
    return {
        "predict": lambda query_e, query_z: predict_members(query_e, query_z).mean(axis=0),
        "predict_members": predict_members,
        "metadata": metadata,
        "partition": partition,
        "validation_metrics": validation_metrics,
        "training_loss_history": histories,
        "normalization": {
            "signature_scale": e_scale.copy(),
            "state_mean": z_mean.copy(),
            "state_scale": z_scale.copy(),
        },
        "state_dicts": [
            {
                name: value.detach().cpu().numpy().copy()
                for name, value in model.state_dict().items()
            }
            for model in networks
        ],
    }


def fit_signature_residual(
    signatures, state_features, responses, records, *, validation_seed, **kwargs
):
    """Whole-seed validation; validation responses never train or stop the network."""
    return _fit_signature_residual(
        signatures,
        state_features,
        responses,
        records,
        validation_seed=validation_seed,
        final_development_refit=False,
        **kwargs,
    )


def refit_signature_residual(signatures, state_features, responses, records, **kwargs):
    """Fit frozen settings on all registered development origins, with three initializations.

    The same identity, origin-only feature, alias, and >=512 distinct nonalias
    training-pair gates apply. No calibration/test seed or future label is accepted.
    """
    return _fit_signature_residual(
        signatures,
        state_features,
        responses,
        records,
        validation_seed=None,
        final_development_refit=True,
        **kwargs,
    )
