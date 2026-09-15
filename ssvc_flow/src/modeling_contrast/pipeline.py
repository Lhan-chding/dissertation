"""N2 staged experiments and cost accounting, without new trajectories."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .model_study import (
    _record_stream,
    baseline_configs,
    best_nonzero,
    choose_observations,
    fit_unit,
)
from .observation_study import load_packet, measure_unit, service_arguments
from .study import (
    METHODS,
    aggregate_rows,
    configuration_id,
    contrast_inputs,
    digest,
    experiment_grid,
    fwrite,
    jwrite,
    load_metadata,
    metric_rows,
)


def _entries(root, seeds=None):
    entries = json.loads((root / "manifest.json").read_text())["trajectories"]
    return [e for e in entries if seeds is None or e["seed"] in seeds]


def packet_for(parent_root, run_root, out, entry, ai, method, n):
    relative = Path(entry["id"]) / f"a{ai}" / f"{method}_n{n}"
    previous = run_root / "N1/packets" / relative
    local = out / "packets" / relative
    if previous.exists():
        packet = load_packet(previous)
    elif local.exists():
        packet = load_packet(local)
    else:
        packet = measure_unit(parent_root, entry, ai, method, n, local)
    if method == "O_IND":
        from .c1_packets import repair_c1_auxiliary

        packet = repair_c1_auxiliary(
            parent_root, entry, ai, packet, out / "C1_auxiliary" / relative
        )
    return packet


def save_summary(out, records):
    _record_stream(out / "METRICS.jsonl.gz", records)
    fwrite(
        out / "CONTRAST_PREDICTION_BY_SEED.csv",
        aggregate_rows(records, keys=("configuration_id", "target", "population", "seed")),
    )
    fwrite(out / "AGGREGATE.csv", aggregate_rows(records))


def _direct_rows(parent_root, entry, ai, packets):
    from src.modeling_qualification.math_contracts import helmert

    from .parent import load_parent, load_parent_oracle

    oracle = load_parent_oracle(parent_root, entry["id"])
    parent = load_parent(parent_root, entry["id"])
    truth = oracle["branch_p"][ai, 10:14, 1:] - oracle["branch_p"][ai, 10:14, :1]
    fit_e = contrast_inputs(parent.observations["branch_theta"][ai, :8])
    active = np.broadcast_to(
        np.r_[np.any(fit_e != 0, axis=(0, 2)), np.any(fit_e[:, 0] - fit_e[:, 1] != 0)], (4, 3)
    )
    _, groups, _ = load_metadata(parent_root)
    rows = []
    for (obs, n), packet in packets.items():
        config = dict(
            stage="DIRECT",
            method="D_DIRECT",
            observation=obs,
            n=n,
            rank=0,
            alpha=0.0,
            eta=0.0,
            fit_banks=0,
        )
        for noise in range(8):
            ident = dict(
                config,
                configuration_id=configuration_id(config),
                seed=entry["seed"],
                arm=entry["arm"],
                anchor=[8, 24, 40][ai],
                noise_replica=noise,
                original_role=entry["split"],
                v2_role="legacy_diagnostic",
                actual_rank=0,
                k=0,
                prediction_hash=packet["artifact_sha256"],
            )
            rows.extend(
                metric_rows(
                    packet["helmert"][noise, 10:14] @ helmert().T, truth, groups, ident, active
                )
            )
    return rows


def _isolation(parent_root, entry, ai, packet, config, out):
    from src.modeling_qualification.math_contracts import helmert

    from .joint_packets import fit_packet_covariance, retained_fit_rows
    from .observations import OracleAudit
    from .oracle_diagnostics import isolate_fit_errors
    from .parent import load_parent, load_parent_oracle
    from .study import assemble_covariance
    from .targets import group_xv_matrix

    p = load_parent(parent_root, entry["id"])
    exact = load_parent_oracle(parent_root, entry["id"])
    theta = p.observations["branch_theta"][ai]
    e = contrast_inputs(theta[:8]).reshape(16, 737)
    query = contrast_inputs(theta[10:14]).reshape(8, 737)
    y = ((exact["branch_p"][ai, :8, 1:] - exact["branch_p"][ai, :8, :1]) @ helmert()).reshape(
        16, 72, 3
    )
    metadata, _, _ = load_metadata(parent_root)
    active = []
    for bank in range(8):
        if not np.array_equal(theta[bank, 1], theta[bank, 0]):
            active.append(2 * bank)
        if not np.array_equal(theta[bank, 2], theta[bank, 0]) and not np.array_equal(
            theta[bank, 2], theta[bank, 1]
        ):
            active.append(2 * bank + 1)
    theta_map, features, categories, prompt_ids, _ = service_arguments(parent_root, entry, ai)
    instrument = OracleAudit.from_parameters(theta_map, features, categories, prompt_ids=prompt_ids)
    oracle_blocks = np.stack(
        [
            instrument.moments(
                [(f"b{b}o0", f"b{b}o1"), (f"b{b}o0", f"b{b}o2")],
                origin_id="origin",
                method=config["observation"],
                n=config["n"],
            )["covariance"]
            for b in range(8)
        ]
    )
    predictions = []
    rows = []
    for noise in range(8):
        oracle_cov = assemble_covariance(oracle_blocks)
        if config["observation"] == "O_IND" and noise < 5:
            oracle_cov = instrument.moments(
                [(f"b{b}o0", f"b{b}o{o}") for b in range(8) for o in (1, 2)],
                origin_id="origin",
                method="O_IND",
                n=config["n"],
            )["covariance"]
        result = isolate_fit_errors(
            e,
            packet["helmert"][noise, :8].reshape(16, 72, 3),
            y,
            fit_packet_covariance(packet, noise, range(8)),
            query,
            group_xv_matrix(metadata),
            config["n"],
            active_rows=retained_fit_rows(packet, noise, range(8)),
            oracle_covariance=oracle_cov,
        )
        offset = len(predictions)
        predictions.extend(result["predictions"])
        for row in result["records"]:
            row.update(
                noise_replica=noise,
                seed=entry["seed"],
                arm=entry["arm"],
                anchor=[8, 24, 40][ai],
                observation=config["observation"],
                array_index=offset + row["prediction_index"],
            )
            rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "predictions.npz",
        prediction=np.asarray(predictions),
        truth=exact["branch_p"][ai, 10:14, 1:] - exact["branch_p"][ai, 10:14, :1],
    )
    jwrite(
        out / "diagnostics.json",
        {
            "main_ranking_eligible": False,
            "records": rows,
            "packet_sha256": packet["artifact_sha256"],
            "prediction_sha256": digest(out / "predictions.npz"),
            "oracle_covariance_cost": instrument._service.ledger.snapshot(),
        },
    )


def run_fits(config, parent_root, run_root, out):
    smoke = json.loads((run_root / "smoke/smoke_result.json").read_text())
    if not smoke["resource_gate"]["passed"]:
        raise ValueError("RESOURCE_REVIEW_REQUIRED")
    from .cost_study import cold_direct_costs, frontier, priced
    from .resources import StageResources

    resources = StageResources(config, run_root, out)
    resources.check("before_N2")
    start = time.perf_counter()
    records = []
    cost_records = []
    development = config["data_roles"]["observation_development_seeds"]
    selection = config["data_roles"]["legacy_selection_seeds"]
    entries = _entries(parent_root, development + selection)
    # 2A exact diagnostics include every old trajectory, retaining original roles.
    grid = experiment_grid("N2A") + baseline_configs(0)
    for entry in _entries(parent_root):
        for ai in range(3):
            records.extend(
                fit_unit(
                    parent_root,
                    entry,
                    ai,
                    grid,
                    {},
                    out / "N2A" / entry["id"] / f"a{ai}",
                    oracle_diagnostics=True,
                )
            )
        resources.check("N2A_" + entry["id"])
        print(
            json.dumps(
                {
                    "phase": "N2A",
                    "id": entry["id"],
                    "seconds": round(time.perf_counter() - start, 2),
                }
            ),
            flush=True,
        )
    # 2B compares all four instruments at fixed full-Q OLS before choosing any.
    b_records = []
    for entry in entries:
        for ai in range(3):
            packets = {
                (m, 64): packet_for(parent_root, run_root, out, entry, ai, m, 64) for m in METHODS
            }
            b_records.extend(
                fit_unit(
                    parent_root,
                    entry,
                    ai,
                    experiment_grid("N2B") + baseline_configs(),
                    packets,
                    out / "N2B" / entry["id"] / f"a{ai}",
                )
            )
            b_records.extend(_direct_rows(parent_root, entry, ai, packets))
            for (obs, n), packet in packets.items():
                cost_records.append(
                    cold_direct_costs(
                        parent_root,
                        entry,
                        ai,
                        obs,
                        n,
                        packet,
                        out / "cost" / entry["id"] / f"a{ai}" / f"{obs}_n{n}.json",
                    )
                )
        resources.check("N2B_" + entry["id"])
        print(
            json.dumps(
                {
                    "phase": "N2B",
                    "id": entry["id"],
                    "seconds": round(time.perf_counter() - start, 2),
                }
            ),
            flush=True,
        )
    calibration_cost = {
        method: float(
            np.mean(
                [
                    priced(row["fit_by_bank_budget"]["8"], 0.25)
                    for row in cost_records
                    if row["observation"] == method and row["seed"] in selection
                ]
            )
        )
        for method in METHODS
    }
    selected, ranking = choose_observations(b_records, calibration_cost)
    jwrite(
        out / "selected_observations.json",
        {
            "selected": selected,
            "selection_seed_ids": selection,
            "selection_uses_test": False,
            "ranking": ranking,
            "legacy_posthoc_development": True,
        },
    )
    records.extend(b_records)
    # 2C only the selected instruments receive rank/alpha/eta sweeps.
    c_records = []
    for entry in entries:
        for ai in range(3):
            packets = {
                (m, 64): packet_for(parent_root, run_root, out, entry, ai, m, 64) for m in selected
            }
            c_records.extend(
                fit_unit(
                    parent_root,
                    entry,
                    ai,
                    experiment_grid("N2C", selected),
                    packets,
                    out / "N2C" / entry["id"] / f"a{ai}",
                )
            )
        resources.check("N2C_" + entry["id"])
        print(
            json.dumps(
                {
                    "phase": "N2C",
                    "id": entry["id"],
                    "seconds": round(time.perf_counter() - start, 2),
                }
            ),
            flush=True,
        )
    records.extend(c_records)
    rules = []
    for obs in selected:
        chosen = best_nonzero(c_records, obs)
        if chosen is not None:
            rules.append(
                {
                    k: chosen[k]
                    for k in ("method", "observation", "n", "rank", "alpha", "eta", "fit_banks")
                }
            )
    jwrite(
        out / "frozen_nonzero_rules.json",
        {
            "rules": rules,
            "status": "BEST_NONZERO_DIAGNOSTIC",
            "selection_seed_ids": selection,
            "n16_n256_retuning": False,
            "rank_zero_fallback": False,
        },
    )
    # 2D direction/coefficient isolation uses only the original development seeds.
    for entry in _entries(parent_root, development):
        for ai in range(3):
            for rule in rules:
                packet = packet_for(parent_root, run_root, out, entry, ai, rule["observation"], 64)
                _isolation(
                    parent_root,
                    entry,
                    ai,
                    packet,
                    rule,
                    out / "N2D" / entry["id"] / f"a{ai}" / rule["observation"],
                )
        resources.check("N2D_" + entry["id"])
        print(
            json.dumps(
                {
                    "phase": "N2D",
                    "id": entry["id"],
                    "seconds": round(time.perf_counter() - start, 2),
                }
            ),
            flush=True,
        )
    # 2E changes one budget coordinate at a time using the already frozen rule.
    egrid = []
    for rule in rules:
        for n in (16, 256):
            egrid.append(dict(rule, stage="N2E", n=n))
        for banks in (1, 2, 4):
            egrid.append(dict(rule, stage="N2E", fit_banks=banks))
    for n in (16, 256):
        egrid.extend(baseline_configs(n))
    for entry in entries:
        for ai in range(3):
            needed = {(c["observation"], c["n"]) for c in egrid}
            packets = {
                key: packet_for(parent_root, run_root, out, entry, ai, *key) for key in needed
            }
            records.extend(
                fit_unit(
                    parent_root, entry, ai, egrid, packets, out / "N2E" / entry["id"] / f"a{ai}"
                )
            )
            records.extend(
                _direct_rows(
                    parent_root,
                    entry,
                    ai,
                    {key: value for key, value in packets.items() if key[1] != 64},
                )
            )
            for (obs, n), packet in packets.items():
                if n != 64:
                    cost_records.append(
                        cold_direct_costs(
                            parent_root,
                            entry,
                            ai,
                            obs,
                            n,
                            packet,
                            out / "cost" / entry["id"] / f"a{ai}" / f"{obs}_n{n}.json",
                        )
                    )
        resources.check("N2E_" + entry["id"])
        print(
            json.dumps(
                {
                    "phase": "N2E",
                    "id": entry["id"],
                    "seconds": round(time.perf_counter() - start, 2),
                }
            ),
            flush=True,
        )
    save_summary(out, records)
    fwrite(out / "COST_FRONTIER.csv", frontier(aggregate_rows(records), cost_records))
    jwrite(
        out / "COST_LEDGER.json",
        {
            "records": cost_records,
            "scenario_label_to_generation_ratio": 0.05,
            "scope": "CPU measurement operations and explicit hypothetical sequence-scoring ratios",
            "candidate_update_cost": (
                "already incurred in legacy corpus; "
                "common to direct and surrogate; zero new optimizer updates"
            ),
            "score_generation_label_counts_separate": True,
        },
    )
    result = {
        "status": "COMPLETE",
        "exact_legacy_trajectories": 60,
        "finite_development_trajectories": 20,
        "selected_observation_methods": selected,
        "new_optimizer_updates": 0,
        "noise_replicas": 8,
        "rows": len(records),
        "wall_seconds": time.perf_counter() - start,
        "stages_completed": ["N2A", "N2B", "N2C", "N2D", "N2E"],
    }
    result["resources"] = resources.check("N2_complete")
    jwrite(out / "stage_result.json", result)
    return result
