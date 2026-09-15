#!/usr/bin/env python3
"""Read-only, NumPy-only audit of sealed N4 primary contrast predictions.

No production modules, model forward passes, fits, or training are imported.
Run with --run-root RUNS/modeling_contrast_v2 --out posthoc/N4_AUDIT.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TARGET = "joint_1_minus_joint_0"
ROLES = {"fresh_calibration": tuple(range(501, 507)), "fresh_locked_test": tuple(range(601, 611))}
ARMS = ("X_BASE", "X_VALID")
ANCHORS = (8, 24, 40)
QUANTITIES = ("pX", "v")
BASELINES = ("C0", "C1_LEGACY_EXACT_RULE")
KEYS = ("configuration_id", "seed", "arm", "anchor", "noise_replica")
BINDINGS = ("source", "config", "data", "selector", "packet")
ORIGINAL_SELECTION_SHA256 = "559a1b513d49c1683e19155c4183cfc0c34daf92d0892c7fbfa46f6709c88645"


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        return json.load(stream)


def inside(root, name):
    root, child = Path(root).resolve(), Path(name)
    if child.is_absolute() or ".." in child.parts:
        raise ValueError(f"unsafe relative artifact: {name}")
    result = (root / child).resolve()
    if root not in result.parents:
        raise ValueError(f"artifact escapes stage: {name}")
    return result


class Integrity:
    def __init__(self):
        self.files = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"missing artifact: {path}")
        value = self.files.get(str(path))
        if value is None:
            value = sha(path)
            self.files[str(path)] = value
        if expected is not None and value != expected:
            raise ValueError(f"artifact hash mismatch: {path}")
        return value

    def stage(self, root):
        root = Path(root).resolve()
        self.check(root / "RUN_MANIFEST.json")
        record = read_json(root / "RUN_MANIFEST.json")
        if record.get("status") != "COMPLETE":
            raise ValueError(f"stage is not COMPLETE: {root.name}")
        if set(record.get("binding", {})) != set(BINDINGS):
            raise ValueError("invalid stage identity binding")
        outputs = record.get("outputs", {})
        actual = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name not in ("RUN_MANIFEST.json", ".writer.lock")
        }
        if actual != set(outputs) or (root / ".writer.lock").exists():
            raise ValueError(f"unregistered or active stage files: {root.name}")
        for name, expected in outputs.items():
            self.check(inside(root, name), expected)
        return record


def _lock(value):
    if value.get("lock_sha256") != canonical(
        {k: v for k, v in value.items() if k != "lock_sha256"}
    ):
        raise ValueError("selection lock canonical hash mismatch")
    for key, field in (
        ("source", "source_hashes"),
        ("data", "input_hashes"),
        ("packet", "packet_hashes"),
        ("selector", "selected"),
    ):
        if value["binding"][key] != canonical(value[field]):
            raise ValueError(f"selection {key} binding mismatch")
    if value["binding"]["config"] != value["protocol_sha256"]:
        raise ValueError("selection config binding mismatch")


def configuration_identity(row, *, selection=False):
    fields = ("rank", "alpha", "eta", "fit_banks")
    values = (
        row["model"] if selection else row["method"],
        row["method"] if selection else row["observation"],
        row["n"],
        *(row[field] for field in fields),
    )
    identity = "|".join(str(value) for value in values)
    if row.get("configuration_id") != identity:
        raise ValueError("configuration identity differs from frozen method and parameters")
    return identity


def verify_science_status(summary, selected_results):
    expected = (
        "PREDICTION_PASS"
        if selected_results
        and all(row["prediction_status"] == "PREDICTION_PASS" for row in selected_results)
        else "PREDICTION_FAIL"
    )
    if summary.get("science_status") != expected:
        raise ValueError("overall science_status differs from recomputed selected results")
    return expected


def verify_selection(root, manifests, integrity):
    stage = root / "N3_server"
    lock = read_json(stage / "MODEL_SELECTION_LOCK.json")
    original = read_json(stage / "ORIGINAL_MODEL_SELECTION_LOCK.json")
    _lock(lock)
    _lock(original)
    if (
        lock["selected"] != original["selected"]
        or lock["candidate_audit"] != original["candidate_audit"]
    ):
        raise ValueError("server transfer changed immutable scientific selection")
    for field in ("statistics", "science_gate", "cost_gate", "cost_frontier"):
        if lock[field] != original[field]:
            raise ValueError(f"server transfer changed {field}")
    if not 1 <= len(lock["selected"]) <= 2 or any(row["n"] != 64 for row in lock["selected"]):
        raise ValueError("audit scope requires one or two frozen n64 selections")
    identities = [configuration_identity(row, selection=True) for row in lock["selected"]]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate frozen selected configuration identities")
    original_sha = integrity.check(stage / "ORIGINAL_MODEL_SELECTION_LOCK.json")
    if original_sha != ORIGINAL_SELECTION_SHA256:
        raise ValueError("original selection differs from the authorized frozen file")
    if lock["server_transfer"]["original_selection_file_sha256"] != original_sha:
        raise ValueError("server transfer original-file binding mismatch")
    receipt = read_json(stage / "SERVER_TRANSFER_RECEIPT.json")
    if canonical(receipt) != lock["server_transfer"]["receipt_sha256"]:
        raise ValueError("server transfer receipt hash mismatch")
    if any(record["binding"] != lock["binding"] for record in manifests.values()):
        raise ValueError("N3/N4/validation identity bindings differ")
    if not lock.get("allow_fresh_cpu") or any(
        lock[gate]["status"] != "PASS"
        for gate in (
            "science_gate",
            "cost_gate",
            "resource_gate",
            "input_identity_gate",
            "data_gate",
        )
    ):
        raise ValueError("fresh execution gates did not pass")
    for field in ("source_hashes", "input_hashes", "packet_hashes", "selector_source_hashes"):
        for path, expected in lock[field].items():
            if not Path(path).is_absolute():
                raise ValueError("server evidence paths must be explicit absolute paths")
            integrity.check(path, expected)
    return lock


def helmert():
    # Explicit orthonormal zero-sum event basis, category order X,S,W,I.
    return np.array(
        [
            [1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
            [-1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
            [0, -2 / math.sqrt(6), 1 / math.sqrt(12)],
            [0, 0, -3 / math.sqrt(12)],
        ]
    )


def contrast_stats(prediction, truth, groups):
    """Independent primary all-population group means from event-space arrays."""
    prediction, truth, groups = np.asarray(prediction), np.asarray(truth), np.asarray(groups)
    if prediction.shape != (4, 72, 4) or truth.shape != prediction.shape or groups.shape != (72,):
        raise ValueError("primary contrast requires four banks, 72 prompts, four events")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("nonfinite sealed prediction or oracle")
    labels = sorted(set(groups))
    if len(labels) != 6 or any(np.count_nonzero(groups == label) != 12 for label in labels):
        raise ValueError("expected six groups with twelve prompts each")
    result = {}
    for quantity, category, sign in (("pX", 0, 1), ("v", 3, -1)):
        errors = np.column_stack(
            [sign * (prediction - truth)[:, groups == label, category].mean(1) for label in labels]
        )
        signal = np.column_stack(
            [sign * truth[:, groups == label, category].mean(1) for label in labels]
        )
        result[quantity + "_error_ss"] = float(np.square(errors).sum())
        result[quantity + "_truth_ss"] = float(np.square(signal).sum())
        result[quantity + "_abs_errors"] = np.abs(errors).ravel().tolist()
    return result


def read_prediction_unit(directory, entry, ai, oracle, groups, selected_ids, integrity):
    directory = Path(directory)
    mini = read_json(directory / "freeze.json")
    if not mini.get("frozen_before_scoring") or mini.get("oracle_diagnostic"):
        raise ValueError("prediction is not sealed finite fresh evidence")
    full = mini
    if mini.get("metadata_file"):
        path = inside(directory, mini["metadata_file"])
        integrity.check(path, mini["metadata_sha256"])
        full = read_json(path)
        for field in ("frozen_before_scoring", "prediction_sha256", "oracle_diagnostic"):
            if full.get(field) != mini.get(field):
                raise ValueError("mini/full frozen metadata disagreement")
    integrity.check(directory / "predictions.npz", full["prediction_sha256"])
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        predictions = data["prediction"]
    models = full["models"]
    if predictions.shape != (len(models), 4, 2, 72, 3):
        raise ValueError("raw prediction and frozen model identity shapes differ")
    if oracle.shape != (3, 14, 3, 72, 4):
        raise ValueError("raw oracle shape differs from frozen 32-run protocol")
    truth = oracle[ai, 10:14, 1] - oracle[ai, 10:14, 0]
    rows = []
    for values, model in zip(predictions, models, strict=True):
        configuration_identity(model)
        if model["configuration_id"] not in selected_ids and not (
            model["method"] in BASELINES and model["n"] == 64
        ):
            continue
        expected = {
            "seed": entry["seed"],
            "arm": entry["arm"],
            "anchor": ANCHORS[ai],
            "original_role": entry["split"],
            "v2_role": entry["split"],
        }
        if any(model.get(key) != value for key, value in expected.items()):
            raise ValueError("frozen model identity disagrees with trajectory role")
        contiguous = np.ascontiguousarray(values)
        digest = hashlib.sha256(
            str(contiguous.dtype).encode() + str(contiguous.shape).encode() + contiguous.tobytes()
        ).hexdigest()
        if model["prediction_hash"] != digest:
            raise ValueError("per-model prediction hash mismatch")
        stats = contrast_stats(values[:, 0] @ helmert().T, truth, groups)
        rows.append({**{key: model[key] for key in KEYS}, "method": model["method"], **stats})
    return rows


def aggregate(rows):
    result = {"seed_count": len({row["seed"] for row in rows}), "row_count": len(rows)}
    if not rows:
        raise ValueError("missing raw prediction rows")
    for quantity in QUANTITIES:
        errors = np.concatenate([row[quantity + "_abs_errors"] for row in rows])
        ss = sum(row[quantity + "_error_ss"] for row in rows)
        energy = sum(row[quantity + "_truth_ss"] for row in rows)
        result.update(
            {
                quantity + "_error_ss": ss,
                quantity + "_truth_ss": energy,
                quantity + "_nrmse": float(np.sqrt(ss / energy)) if energy > 0 else None,
                quantity + "_error_q95": float(np.quantile(errors, 0.95)),
                quantity + "_mae": float(errors.mean()),
            }
        )
    return result


def bootstrap(method, baseline, quantity, *, seed=2026091503, replicates=5000):
    """Resample whole training seeds and recompute ratios from summed energies."""
    keys = KEYS[1:]
    paired = {tuple(row[key] for key in keys): row for row in baseline}
    identities = [tuple(row[key] for key in keys) for row in method]
    if (
        len(paired) != len(baseline)
        or len(set(identities)) != len(identities)
        or set(identities) != set(paired)
    ):
        raise ValueError("bootstrap requires complete unique matched blocks")
    seeds = sorted({row["seed"] for row in method})
    columns = (quantity + "_error_ss", quantity + "_truth_ss")
    left, right = [], []
    for training_seed in seeds:
        block = [row for row in method if row["seed"] == training_seed]
        matches = [paired[tuple(row[key] for key in keys)] for row in block]
        if not np.allclose(
            [row[columns[1]] for row in block],
            [row[columns[1]] for row in matches],
            rtol=1e-12,
            atol=0,
        ):
            raise ValueError("bootstrap paired truth energy differs")
        left.append([sum(row[column] for row in block) for column in columns])
        right.append([sum(row[column] for row in matches) for column in columns])
    left, right = np.asarray(left), np.asarray(right)
    if np.any(left[:, 1] <= 0):
        return {"status": "NO_SIGNAL_IN_SEED_CLUSTER", "cluster_count": len(seeds)}
    draws = np.random.default_rng(seed).integers(0, len(seeds), (replicates, len(seeds)))
    ls, rs = left[draws].sum(1), right[draws].sum(1)
    differences = np.sqrt(ls[:, 0] / ls[:, 1]) - np.sqrt(rs[:, 0] / rs[:, 1])
    lm, rm = left.sum(0), right.sum(0)
    return {
        "cluster_count": len(seeds),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "paired_mean": float(np.sqrt(lm[0] / lm[1]) - np.sqrt(rm[0] / rm[1])),
        "ci025": float(np.quantile(differences, 0.025)),
        "ci975": float(np.quantile(differences, 0.975)),
        "per_seed": [
            {
                "seed": key,
                "paired_difference": float(
                    np.sqrt(left_seed[0] / left_seed[1]) - np.sqrt(right_seed[0] / right_seed[1])
                ),
                "row_count": sum(row["seed"] == key for row in method),
            }
            for key, left_seed, right_seed in zip(seeds, left, right, strict=True)
        ],
    }


def near(actual, expected, name):
    if isinstance(expected, (list, tuple)):
        if not np.allclose(actual, expected, rtol=2e-10, atol=1e-15):
            raise ValueError(f"statistical mismatch: {name}")
    elif expected is None:
        if actual not in (None, ""):
            raise ValueError(f"zero-signal status mismatch: {name}")
    elif actual in (None, "") or not math.isclose(
        float(actual), float(expected), rel_tol=2e-10, abs_tol=1e-15
    ):
        raise ValueError(f"statistical mismatch: {name}: {actual} != {expected}")


def _index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate recorded score: {key}")
        result[key] = row
    return result


def verify_scores(validation, role, rows, selected_ids):
    expected = _index(rows, KEYS)
    recorded = []
    with gzip.open(validation / f"{role}_metrics.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["target"] == TARGET
                and row["population"] == "all"
                and row["configuration_id"] in {key[0] for key in expected}
            ):
                configuration_identity(row)
                recorded.append(row)
    actual = _index(recorded, KEYS)
    if actual.keys() != expected.keys():
        raise ValueError("raw prediction and recorded primary metric coverage differ")
    for key, row in expected.items():
        for quantity in QUANTITIES:
            for suffix in ("error_ss", "truth_ss", "abs_errors"):
                field = quantity + "_" + suffix
                near(actual[key][field], row[field], f"row {key} {field}")
    with (validation / f"{role}_BY_SEED.csv").open(newline="") as stream:
        recorded_seed = [
            row
            for row in csv.DictReader(stream)
            if row["target"] == TARGET
            and row["population"] == "all"
            and row["configuration_id"] in {key[0] for key in expected}
        ]
    actual_seed = _index(recorded_seed, ("configuration_id", "seed"))
    expected_seed_keys = {
        (identity, str(seed)) for identity in {key[0] for key in expected} for seed in ROLES[role]
    }
    if actual_seed.keys() != expected_seed_keys:
        raise ValueError("BY_SEED configuration/seed coverage differs from frozen roles")
    per_seed = []
    for ident in sorted({row["configuration_id"] for row in rows}):
        for seed in ROLES[role]:
            computed = aggregate(
                [row for row in rows if row["configuration_id"] == ident and row["seed"] == seed]
            )
            stored = actual_seed.get((ident, str(seed)))
            if stored is None:
                raise ValueError("missing per-seed score")
            for field, value in computed.items():
                if field not in ("row_count",):
                    near(stored[field], value, f"seed {seed} {ident} {field}")
            if ident in selected_ids:
                per_seed.append({"configuration_id": ident, "seed": seed, **computed})
    return per_seed


def audit_results(run_root):
    root = Path(run_root).resolve()
    integrity = Integrity()
    manifests = {
        stage: integrity.stage(root / stage) for stage in ("N3_server", "N4", "N4_validation")
    }
    lock = verify_selection(root, manifests, integrity)
    selected_ids = {row["configuration_id"] for row in lock["selected"]}
    validation, raw = root / "N4_validation", root / "N4/raw"
    manifest = read_json(raw / "manifest.json")
    expected_roles = {
        (seed, arm, role) for role, seeds in ROLES.items() for seed in seeds for arm in ARMS
    }
    identities = [(row["seed"], row["arm"], row["split"]) for row in manifest["trajectories"]]
    if (
        len(identities) != 32
        or set(identities) != expected_roles
        or manifest.get("profile") != "contrast_fresh"
    ):
        raise ValueError("fresh raw corpus is not the frozen 32-trajectory protocol")
    metadata = read_json(raw / "probe_metadata.json")
    integrity.check(raw / "probe_metadata.json", manifest["probe_identity_sha256"])
    groups = np.asarray([row["group"] for row in metadata])
    folds = {role: [] for role in ROLES}
    for entry in manifest["trajectories"]:
        oracle_path = inside(raw, entry["oracle_file"])
        integrity.check(oracle_path, entry["sha256"]["oracle_file"])
        with np.load(oracle_path, allow_pickle=False) as data:
            oracle = data["branch_p"]
        for ai in range(3):
            directory = inside(validation, f"{entry['split']}/{entry['id']}/a{ai}/models")
            folds[entry["split"]].extend(
                read_prediction_unit(directory, entry, ai, oracle, groups, selected_ids, integrity)
            )
    ids = None
    seed_results = {}
    for role, rows in folds.items():
        unique = _index(rows, KEYS)
        actual_ids = {key[0] for key in unique}
        if ids is None:
            ids = actual_ids
        if ids != actual_ids or not selected_ids <= ids:
            raise ValueError("calibration/test frozen model coverage differs")
        for ident in ids:
            wanted = {
                (ident, seed, arm, anchor, noise)
                for seed in ROLES[role]
                for arm in ARMS
                for anchor in ANCHORS
                for noise in range(8)
            }
            if {key for key in unique if key[0] == ident} != wanted:
                raise ValueError("missing or duplicate fresh model/seed/anchor/replica blocks")
        seed_results[role] = verify_scores(validation, role, rows, selected_ids)
    summary = read_json(validation / "INDEPENDENT_VALIDATION.json")
    if summary["status"] != "INDEPENDENT_VALIDATION_COMPLETE":
        raise ValueError("independent validation was not completed")
    integrity.check(root / "N3_server/MODEL_SELECTION_LOCK.json", summary["selection_sha256"])
    for mapping in ("input_hashes", "prediction_hashes", "scoring_hashes"):
        for path, expected in summary[mapping].items():
            integrity.check(path, expected)
    stored_pooled = _index(summary["selected_results"], ("configuration_id",))
    if {key[0] for key in stored_pooled} != selected_ids:
        raise ValueError("independent result selected configuration coverage differs")
    pooled = []
    recorded_bootstrap = _index(
        read_json(validation / "PAIRED_FROZEN_TEST_BOOTSTRAP.json"),
        ("configuration_id", "baseline", "quantity"),
    )
    comparisons = []
    for ident in sorted(selected_ids):
        rows = [row for row in folds["fresh_locked_test"] if row["configuration_id"] == ident]
        result = aggregate(rows)
        for field, expected in result.items():
            if field != "row_count":
                near(stored_pooled[(ident,)][field], expected, f"pooled {ident} {field}")
        result["prediction_status"] = (
            "PREDICTION_PASS"
            if all(
                result[q + "_nrmse"] is not None and result[q + "_nrmse"] <= 0.75
                for q in QUANTITIES
            )
            else "PREDICTION_FAIL"
        )
        result["resolution_status"] = (
            "RESOLUTION_PASS"
            if all(result[q + "_error_q95"] <= 0.000125 for q in QUANTITIES)
            else "RESOLUTION_FAIL"
        )
        for field in ("prediction_status", "resolution_status"):
            if result[field] != stored_pooled[(ident,)][field]:
                raise ValueError(f"pooled scientific classification mismatch: {field}")
        pooled.append({"configuration_id": ident, **result})
        for baseline in BASELINES:
            baseline_rows = [row for row in folds["fresh_locked_test"] if row["method"] == baseline]
            for quantity in QUANTITIES:
                value = bootstrap(rows, baseline_rows, quantity)
                stored = recorded_bootstrap[(ident, baseline, quantity)]
                for field, expected in value.items():
                    if field == "per_seed":
                        actual = _index(stored[field], ("seed",))
                        for record in expected:
                            for column in ("paired_difference", "row_count"):
                                near(
                                    actual[(record["seed"],)][column],
                                    record[column],
                                    f"bootstrap seed {column}",
                                )
                    elif field != "status":
                        near(
                            stored[field],
                            expected,
                            f"bootstrap {ident} {baseline} {quantity} {field}",
                        )
                comparisons.append(
                    {"configuration_id": ident, "baseline": baseline, "quantity": quantity, **value}
                )
    science_status = verify_science_status(summary, pooled)
    calibration = read_json(validation / "CALIBRATION_LOCK.json")
    if calibration.get("calibration_seed_ids") != list(ROLES["fresh_calibration"]):
        raise ValueError("empirical calibration seed identities differ")
    if calibration["calibration_sha256"] != canonical(
        {k: v for k, v in calibration.items() if k != "calibration_sha256"}
    ):
        raise ValueError("calibration lock canonical hash mismatch")
    cal_index = _index(
        [
            row
            for row in calibration["rows"]
            if row["target"] == TARGET and row["population"] == "all"
        ],
        ("configuration_id", "quantity"),
    )
    coverage = _index(
        [
            row
            for row in read_json(validation / "EMPIRICAL_COVERAGE.json")
            if row["target"] == TARGET and row["population"] == "all"
        ],
        ("configuration_id", "quantity"),
    )
    envelopes = []
    for ident in sorted(selected_ids):
        for quantity in QUANTITIES:
            maxima = {
                role: {
                    seed: max(
                        max(row[quantity + "_abs_errors"])
                        for row in folds[role]
                        if row["configuration_id"] == ident and row["seed"] == seed
                    )
                    for seed in ROLES[role]
                }
                for role in ROLES
            }
            radius = max(maxima["fresh_calibration"].values())
            rate = sum(value <= radius for value in maxima["fresh_locked_test"].values()) / 10
            near(cal_index[(ident, quantity)]["radius"], radius, "calibration radius")
            near(coverage[(ident, quantity)]["radius"], radius, "test envelope radius")
            near(
                coverage[(ident, quantity)]["empirical_seed_coverage"],
                rate,
                "test empirical seed coverage",
            )
            envelopes.append(
                {
                    "configuration_id": ident,
                    "quantity": quantity,
                    "radius": radius,
                    "empirical_seed_coverage": rate,
                }
            )
    return {
        "status": "PASS",
        "scope": (
            "independent primary joint contrast/all-population selected n64 "
            "and C0/C1 raw-prediction audit"
        ),
        "run_root": str(root),
        "auditor_sha256": sha(Path(__file__)),
        "production_statistics_imported": False,
        "model_forwards": 0,
        "training_runs": 0,
        "stage_manifest_hashes": {
            stage: integrity.files[str(root / stage / "RUN_MANIFEST.json")] for stage in manifests
        },
        "verified_file_count": len(integrity.files),
        "verified_file_map_sha256": canonical(integrity.files),
        "selected_pooled": pooled,
        "science_status": science_status,
        "selected_by_seed": seed_results,
        "bootstrap": comparisons,
        "empirical_envelopes": envelopes,
        "trajectory_count": 32,
        "anchor_count": 96,
        "bootstrap_replicates": 5000,
        "bootstrap_cluster": "training_seed",
        "calibration_seed_count": 6,
        "test_seed_count": 10,
        "distribution_free_95_claim": False,
        "simultaneous_safety_certification": False,
        "excluded_from_statistical_recomputation": [
            "n16/n256",
            "secondary contrasts",
            "active population",
            "direction curves",
            "fit algorithm",
            "direct/known-event baselines",
        ],
        "causal_order_scope": (
            "verifies sealed metadata and hashes; "
            "cannot independently prove historical wall-clock order"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output, root = args.out.resolve(), args.run_root.resolve()
    if any(
        stage == output or stage in output.parents
        for stage in (root / "N3_server", root / "N4", root / "N4_validation")
    ):
        parser.error("audit output must be outside the sealed execution stages")
    try:
        result = audit_results(root)
        exit_code = 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {
            "status": "FAIL",
            "reason": str(error),
            "run_root": str(root),
            "auditor_sha256": sha(Path(__file__)),
            "model_forwards": 0,
            "training_runs": 0,
        }
        exit_code = 1
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(content.encode()) > 3 * 2**20:
        raise ValueError("audit JSON exceeded compact 3 MiB output limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(content)
    print(json.dumps({"status": result["status"], "out": str(output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
