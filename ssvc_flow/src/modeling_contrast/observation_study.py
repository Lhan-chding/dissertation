"""Four paid instruments on fixed legacy policies, with isolated oracle audits."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from .study import digest, fwrite, jwrite, load_metadata, rng_seed


def event_validity_view(events, view):
    events = np.asarray(events)
    if view == "raw_event":
        return events[..., :3].sum(-1)
    if view == "helmert":
        return -events[..., 3]
    raise ValueError("unknown event estimator view")


def service_arguments(root, entry, ai):
    from .parent import load_parent

    parent = load_parent(root, entry["id"])
    obs = parent.observations
    theta = {"origin": obs["theta"][obs["anchors"][ai]]}
    for b in range(14):
        for op in range(3):
            theta[f"b{b}o{op}"] = obs["branch_theta"][ai, b, op]
    with np.load(Path(root) / "dataset.npz", allow_pickle=False) as d:
        features, categories = d["probe_features"].copy(), d["probe_categories"].copy()
    metadata, groups, _ = load_metadata(root)
    return theta, features, categories, tuple(m["prompt_id"] for m in metadata), groups


def measure_unit(
    root, entry, ai, method, n, out, *, replicas=8, banks=tuple(range(14)), data_role=None
):
    from .observations import ToyWorldService, measure
    from .parent import load_parent_counts

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    theta, features, categories, prompt_ids, _ = service_arguments(root, entry, ai)
    old_counts = load_parent_counts(root, entry["id"]) if method == "O_IND" else None
    z = np.zeros((replicas, 14, 2, 72, 3))
    raw = np.zeros((replicas, 14, 2, 72, 4))
    covariance = np.zeros((replicas, 14, 72, 6, 6))
    levels = np.zeros((replicas, 14, 3, 72, 4)) if method == "O_IND" else None
    origin_counts = np.zeros((replicas, 72, 4), dtype=np.int64) if method == "O_IND" else None
    supplementary_costs = []
    primitive_arrays, metadata_rows = {}, []
    started = time.perf_counter()
    for noise in range(replicas):
        # Each replica is a separate proposed deployment: do not amortize its
        # score/label caches across independent estimator replications.
        service = ToyWorldService.from_parameters(
            theta, features, categories, prompt_ids=prompt_ids
        )
        historical = old_counts is not None and noise < len(old_counts[f"branch_n{n}"])
        available_levels = {}
        for bank in banks:
            policies = [f"b{bank}o{op}" for op in range(3)]
            # Legacy aliases can cross the fit/evaluation split. Retain old fit
            # counts but draw independent evaluation packets, never relabel a
            # shared historical sample as independently observed.
            reuse_legacy = historical and bank < 10
            legacy = (
                {
                    policy: old_counts[f"branch_n{n}"][noise, ai, bank, op]
                    for op, policy in enumerate(policies)
                }
                if reuse_legacy
                else None
            )
            role = "fit" if bank < 8 else "diagnostic" if bank < 10 else "evaluation"
            seed = rng_seed(
                2026091502, entry["seed"], entry["arm"], int(ai), int(bank), n, noise, role
            )
            packet = measure(
                service,
                [(policies[0], policies[1]), (policies[0], policies[2])],
                origin_id="origin",
                method=method,
                n=n,
                seed=seed,
                purpose=role,
                legacy_counts=legacy,
            )
            if packet.status == "BLOCKED_SUPPORT_MISMATCH":
                raise ValueError("BLOCKED_SUPPORT_MISMATCH; packet retained in failed stage")
            z[noise, bank] = packet.helmert_estimate
            raw[noise, bank] = packet.raw_event_estimate
            covariance[noise, bank] = packet.covariance
            prefix = f"r{noise}b{bank}_"
            arrays = packet.arrays(include_contributions=False, include_covariance=False)
            for key, value in arrays.items():
                if key.startswith("sample_"):
                    primitive_arrays[prefix + key] = value
            meta = packet.metadata()
            meta.update(
                bank=bank,
                noise_replica=noise,
                sample_names=list(packet.samples),
                array_prefix=prefix,
                seed_source="V2_independent_bank_packet_RNG",
                old_counts_reused=reuse_legacy,
                observation_origin=(
                    "FRESH_CPU_COUNTS"
                    if data_role and data_role != "legacy_diagnostic"
                    else "LEGACY_COUNTS"
                )
                if reuse_legacy
                else "NEW_FROZEN_CPU_MEASUREMENT",
                v2_role=data_role or "legacy_diagnostic",
            )
            if method == "O_IND":
                for policy in policies:
                    fp = service.fingerprint(policy)
                    if fp in packet.samples and "counts" in packet.samples[fp]:
                        available_levels[(bank, fp)] = packet.samples[fp]["counts"]
            # Large per-prompt discordance arrays are redundant with raw draws.
            discordance = meta["diagnostics"].pop("event_discordance", None)
            if discordance is not None:
                primitive_arrays[prefix + "event_discordance"] = np.asarray(discordance)
            metadata_rows.append(meta)
        if method == "O_IND":
            before = service.ledger.snapshot()
            if historical:
                levels[noise] = old_counts[f"branch_n{n}"][noise, ai] / n
                origin_counts[noise] = old_counts[f"trajectory_n{n}"][noise, [8, 24, 40][ai]]
            else:
                # C1 additionally needs the main motion and origin counts.
                # Reuse counts paid by the contrast packet when available.
                for bank in banks:
                    for op in range(3):
                        policy = f"b{bank}o{op}"
                        fp = service.fingerprint(policy)
                        if (bank, fp) not in available_levels:
                            from .c1_packets import auxiliary_seed

                            rr = np.random.default_rng(
                                auxiliary_seed(entry["id"], ai, n, noise, bank, fp)
                            )
                            _, labels, _ = service.sample(policy, n, rr)
                            available_levels[(bank, fp)] = np.stack(
                                [np.bincount(v, minlength=4) for v in labels]
                            )
                        levels[noise, bank, op] = available_levels[(bank, fp)] / n
                rr = np.random.default_rng(
                    rng_seed(2026091502, entry["id"], ai, n, noise, "C1_origin")
                )
                _, labels, _ = service.sample("origin", n, rr)
                origin_counts[noise] = np.stack([np.bincount(v, minlength=4) for v in labels])
            supplementary_costs.append(
                {
                    "noise_replica": noise,
                    "cost": service.ledger.delta(before),
                    "purpose": "C1_total_response_additional_levels_and_origin",
                    "legacy_reused": historical,
                }
            )
    if levels is not None:
        primitive_arrays["legacy_total_levels"] = levels
        primitive_arrays["legacy_origin_counts"] = origin_counts
    from .packet_codec import encode_arrays

    np.savez_compressed(out / "packet_arrays.npz", **encode_arrays(primitive_arrays))
    result = {
        "trajectory_id": entry["id"],
        "seed": entry["seed"],
        "arm": entry["arm"],
        "original_role": entry["split"],
        "v2_role": data_role or "legacy_diagnostic",
        "anchor_index": ai,
        "method": method,
        "n": n,
        "replicas": replicas,
        "banks": list(banks),
        "packets": metadata_rows,
        "packet_arrays_sha256": digest(out / "packet_arrays.npz"),
        "input_parameters_sha256": entry["sha256"]["observations_file"],
        "input_counts_sha256": entry["sha256"]["counts_file"] if method == "O_IND" else None,
        "wall_seconds": time.perf_counter() - started,
        "optimizer_updates": 0,
        "C1_supplementary_costs": supplementary_costs,
        "generated_sequences": 0,
        "actions_preserve_parent_order": True,
        "bank_rng_independent": method != "O_IND",
        "legacy_cross_bank_covariance_preserved": method == "O_IND",
        "fit_evaluation_packet_independent": True,
        "c1_auxiliary_fit_evaluation_independent": True,
        "evaluation_source": "NEW_INDEPENDENT_FROZEN_CPU_PACKET",
        "codec": "LOSSLESS_PAID_LOGP_SPARSE_DICTIONARY",
    }
    result["derived_statistics_storage"] = "RECONSTRUCT_FROM_PAID_PRIMITIVES_BITWISE_VERIFIED"
    with gzip.open(out / "packet_metadata.json.gz", "wt") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    summary = {k: v for k, v in result.items() if k not in ("packets", "C1_supplementary_costs")}
    summary["metadata_sha256"] = digest(out / "packet_metadata.json.gz")
    jwrite(out / "packets.json", summary)
    return load_packet(out)


def load_packet(path):
    path = Path(path)
    metadata = json.loads((path / "packets.json").read_text())
    if (path / "packet_metadata.json.gz").exists():
        if digest(path / "packet_metadata.json.gz") != metadata["metadata_sha256"]:
            raise ValueError("packet metadata hash mismatch")
        with gzip.open(path / "packet_metadata.json.gz", "rt") as stream:
            metadata = json.load(stream)
    actual = digest(path / "packet_arrays.npz")
    if actual != metadata["packet_arrays_sha256"]:
        raise ValueError("packet hash mismatch")
    with np.load(path / "packet_arrays.npz", allow_pickle=False) as arrays:
        if all(key in arrays for key in ("helmert", "raw", "covariance")):
            result = {
                key: arrays[key].copy()
                for key in (
                    "helmert",
                    "raw",
                    "covariance",
                    "legacy_total_levels",
                    "legacy_origin_counts",
                )
                if key in arrays
            }
        else:
            from .packet_codec import decode_arrays
            from .packet_reconstruction import reconstruct_measurements

            decoded = decode_arrays({key: arrays[key] for key in arrays.files})
            result = reconstruct_measurements(decoded, metadata)
            result.update(
                {
                    key: decoded[key]
                    for key in ("legacy_total_levels", "legacy_origin_counts")
                    if key in decoded
                }
            )
    result.update(metadata=metadata, artifact_sha256=actual, path=str(path))
    return result


def compare_unit(root, entry, ai, packet_paths, out):
    """Oracle covariance is calculated only after finite packets are on disk."""
    from src.modeling_qualification.math_contracts import helmert

    from .observations import OracleAudit, ToyWorldService, known_event_logp

    theta, features, categories, prompt_ids, groups = service_arguments(root, entry, ai)
    oracle = OracleAudit.from_parameters(theta, features, categories, prompt_ids=prompt_ids)
    rows, oracle_arrays = [], {}
    for (method, n), path in packet_paths.items():
        packet = load_packet(path)
        z, raw, cov = packet["helmert"], packet["raw"], packet["covariance"]
        estimated_events = z @ helmert().T
        for bank in packet["metadata"]["banks"]:
            p = [f"b{bank}o{i}" for i in range(3)]
            moments = oracle.moments(
                [(p[0], p[1]), (p[0], p[2])], origin_id="origin", method=method, n=n
            )
            truth = moments["event_mean"]
            tcov = moments["covariance"]
            prefix = f"{method}_n{n}_b{bank}"
            oracle_arrays[prefix + "_covariance"] = tcov
            oracle_arrays[prefix + "_truth"] = truth
            for ti, target in enumerate(("joint_1_minus_joint_0", "no_x_off_1_minus_joint_0")):
                h = helmert()
                covariance_block = cov[:, bank, :, 3 * ti : 3 * ti + 3, 3 * ti : 3 * ti + 3]
                point_var_x = np.einsum("i,rpij,j->rp", h[0], covariance_block, h[0])
                point_var_v = np.einsum("i,rpij,j->rp", h[3], covariance_block, h[3])
                interval_widths = {}
                for metric, variances in (("pX", point_var_x), ("v", point_var_v)):
                    group_var = np.stack(
                        [
                            variances[:, groups == g].sum(1) / np.count_nonzero(groups == g) ** 2
                            for g in sorted(set(groups))
                        ],
                        1,
                    )
                    positive = group_var[group_var > 0]
                    interval_widths[metric + "_group_pointwise_normal_width_q95"] = (
                        float(np.quantile(3.92 * np.sqrt(positive), 0.95))
                        if len(positive)
                        else None
                    )
                    interval_widths[metric + "_empirical_zero_group_variance_count"] = int(
                        np.count_nonzero(group_var <= 0)
                    )
                bank_meta = [m for m in packet["metadata"]["packets"] if m["bank"] == bank]
                missing = sum(
                    sum(pair[0] == ti for pair in m["diagnostics"]["unobserved_X"])
                    for m in bank_meta
                )
                for view, pred in (
                    ("helmert", estimated_events[:, bank, ti]),
                    ("raw_event", raw[:, bank, ti]),
                ):
                    err = pred - truth[ti]
                    gx = np.stack([pred[:, groups == g, 0].mean(1) for g in sorted(set(groups))], 1)
                    valid = event_validity_view(pred, view)
                    gv = np.stack([valid[:, groups == g].mean(1) for g in sorted(set(groups))], 1)
                    tx = np.array([truth[ti, groups == g, 0].mean() for g in sorted(set(groups))])
                    tv = np.array([-truth[ti, groups == g, 3].mean() for g in sorted(set(groups))])
                    raw_cov = moments.get("raw_covariance")
                    event_cov = (
                        moments.get("event_helmert_covariance") if view == "helmert" else raw_cov
                    )
                    sl = slice(4 * ti, 4 * ti + 4)
                    theory_var = float(
                        np.trace(event_cov[:, sl, sl], axis1=-2, axis2=-1).mean() / 4
                    )
                    empirical_var = float(np.var(pred, axis=0, ddof=1).mean())
                    item = dict(
                        trajectory_id=entry["id"],
                        seed=entry["seed"],
                        arm=entry["arm"],
                        original_role=entry["split"],
                        v2_role="legacy_diagnostic",
                        anchor=[8, 24, 40][ai],
                        bank=bank,
                        method=method,
                        n=n,
                        target=target,
                        view=view,
                        replicas=len(pred),
                        bias=float(np.mean(pred.mean(0) - truth[ti])),
                        rms_bias=float(np.sqrt(np.mean((pred.mean(0) - truth[ti]) ** 2))),
                        rmse=float(np.sqrt(np.mean(err**2))),
                        theoretical_variance=theory_var,
                        empirical_variance=empirical_var,
                        empirical_to_theoretical_variance=empirical_var / theory_var
                        if theory_var > 0
                        else None,
                        pX_error_ss=float(np.sum((gx - tx) ** 2)),
                        pX_truth_ss=float(len(pred) * np.sum(tx**2)),
                        v_error_ss=float(np.sum((gv - tv) ** 2)),
                        v_truth_ss=float(len(pred) * np.sum(tv**2)),
                        mass_residual_rms=float(np.sqrt(np.mean(raw[:, bank, ti].sum(-1) ** 2))),
                        packet_source=str(path),
                        packet_sha256=packet["artifact_sha256"],
                    )
                    if view == "helmert":
                        item.update(interval_widths)
                    item.update(
                        unobserved_X_prompt_replicas=missing,
                        interval_type="EMPIRICAL_POINTWISE_NORMAL_APPROXIMATION_NOT_CERTIFIED",
                        zero_empirical_variance_means_safety=False,
                    )
                    rows.append(item)
    # Direct event scoring is measured on an independent fresh cache, so no
    # earlier oracle or finite scoring can make this strong baseline free.
    direct_service = ToyWorldService.from_parameters(
        theta, features, categories, prompt_ids=prompt_ids
    )
    policies = [f"b{b}o{o}" for b in range(14) for o in range(3)]
    direct = known_event_logp(direct_service, policies)
    oracle_arrays["known_event_pX"] = direct.pop("pX")
    oracle_arrays["known_event_v"] = direct.pop("v")
    oracle_arrays["known_event_actions"] = direct.pop("actions")
    direct.update(trajectory_id=entry["id"], anchor=[8, 24, 40][ai], policies=policies)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "oracle_audit.npz", **oracle_arrays)
    jwrite(out / "direct_known_event.json", direct)
    jwrite(
        out / "covariance_audit.json",
        dict(
            status="ORACLE_DIAGNOSTIC_ONLY",
            oracle_sha256=digest(out / "oracle_audit.npz"),
            predictor_can_access_oracle=False,
            bank_independence=True,
            within_bank_joint_covariance=True,
            oracle_cost=oracle._service.ledger.snapshot(),
        ),
    )
    fwrite(out / "comparison.csv", rows)
    return rows
