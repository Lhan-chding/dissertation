"""Read-only V3 Q0 audit of original V2 packets and frozen predictions.

Summary tables never stand in for absent primitives. This module performs no
sampling, policy scoring, fitting, optimizer update, or source-file mutation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from src.modeling_contrast.recompute import (
    MODEL_FIELDS,
    MODEL_KEYS,
    TARGETS,
    H,
    independent_model_statistics,
    independent_observation_statistics,
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError(f"unsafe manifest member: {relative}")
    return path


def _json(path):
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as stream:
        return json.load(stream)


def _publish(path, text):
    """Atomic publication which never clobbers a prior result."""
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix=".q0-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _csv_text(rows):
    stream = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        )
    return stream.getvalue()


def full_rank_equivalence(predictions, models, *, atol=1e-12):
    """Compare C5/C6 to the matched original C3; r=k is not low-rank evidence."""
    predictions = np.asarray(predictions)
    if len(predictions) != len(models) or not np.isfinite(predictions).all():
        raise ValueError("finite aligned predictions and model identities required")
    for model in models:
        r, k = model.get("actual_rank"), model.get("k")
        if type(r) is not int or type(k) is not int or not 0 <= r <= k:
            raise ValueError("each original model requires integer ranks 0 <= r <= k")
    fields = (
        "observation",
        "n",
        "alpha",
        "eta",
        "fit_banks",
        "noise_replica",
        "seed",
        "arm",
        "anchor",
    )
    index = {}
    for i, model in enumerate(models):
        if model.get("method") == "C3" and model.get("actual_rank") == model.get("k"):
            key = tuple(model.get(field) for field in fields)
            if key in index:
                raise ValueError("duplicate matched C3 full-rank baseline")
            index[key] = i
    rows = []
    for i, model in enumerate(models):
        if model.get("method") not in ("C5", "C6"):
            continue
        row = {field: model.get(field) for field in fields}
        row.update(method=model["method"], actual_rank=model.get("actual_rank"), k=model.get("k"))
        row["genuine_truncation"] = 0 < model.get("actual_rank", 0) < model.get("k", 0)
        key = tuple(model.get(field) for field in fields)
        if model.get("actual_rank") != model.get("k"):
            row.update(
                status="NONDEGENERATE_RANK_NOT_EQUIVALENCE_CASE", max_absolute_difference=None
            )
        elif key not in index:
            row.update(status="MISSING_MATCHED_C3", max_absolute_difference=None)
        else:
            difference = float(np.max(np.abs(predictions[i] - predictions[index[key]])))
            row.update(
                status="PASS" if difference <= atol else "FAIL",
                max_absolute_difference=difference,
                zero_dimensional=model.get("k") == 0,
            )
        rows.append(row)
    return rows


class _Audit:
    def __init__(self, raw, run, out):
        self.raw, self.run, self.out = map(Path, (raw, run, out))
        self.index, self.missing, self.errors = {}, [], []
        self.models, self.observations, self.equivalence, self.origins = [], [], [], []
        self.packet_count, self.primitive_count, self.metric_comparisons = 0, 0, 0
        self.absent_model_matrices = 0

    def missing_input(self, scope, detail):
        item = {"scope": str(scope), "detail": detail, "status": "MISSING_INPUTS"}
        if item not in self.missing:
            self.missing.append(item)

    def cover(self, path, role, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        key = str(path)
        if key not in self.index:
            self.index[key] = {
                "record_type": "file",
                "path": key,
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
                "roles": [],
            }
        row = self.index[key]
        if role not in row["roles"]:
            row["roles"].append(role)
        if expected is not None and row["sha256"] != expected:
            raise ValueError(f"hash mismatch: {path}")
        return row["sha256"]

    def read(self, path, role, expected=None):
        self.cover(path, role, expected)
        return _json(path)

    def attempt(self, scope, callback):
        try:
            return callback()
        except FileNotFoundError as exc:
            self.missing_input(scope, f"original artifact absent: {exc}")
        except (ValueError, KeyError, TypeError, IndexError, OSError, AssertionError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("Missing "):
                self.missing_input(scope, str(exc))
            else:
                self.errors.append({"scope": str(scope), "detail": str(exc)})
        return None

    def stage(self, directory):
        manifest = self.read(directory / "RUN_MANIFEST.json", "stage_manifest")
        if manifest.get("status") != "COMPLETE" or (directory / ".writer.lock").exists():
            raise ValueError("historical stage is incomplete or has an unresolved writer")
        outputs = manifest.get("outputs", {})
        if not outputs:
            raise ValueError("empty historical output inventory")
        for relative, expected in outputs.items():
            self.attempt(
                directory / relative,
                lambda p=relative, h=expected: self.cover(
                    _inside(directory, p), "sealed_stage_output", h
                ),
            )
        actual = {
            str(p.relative_to(directory))
            for p in directory.rglob("*")
            if p.is_file() and p.name not in ("RUN_MANIFEST.json", ".writer.lock")
        }
        extra = actual - set(outputs)
        if extra:
            raise ValueError(f"unregistered historical outputs: {sorted(extra)[:10]}")
        return manifest

    def metadata(self, path, role, default):
        mini = self.read(path, role + "_binding")
        extra = _inside(path.parent, mini.get("metadata_file", default))
        if extra.is_file() or "metadata_sha256" in mini:
            if "metadata_sha256" not in mini:
                raise ValueError("compressed metadata lacks its byte hash")
            full = self.read(extra, role, mini["metadata_sha256"])
            for field in set(mini) & set(full):
                if mini[field] != full[field]:
                    raise ValueError(f"mini/full metadata mismatch: {field}")
            return full
        return mini

    def array(self, entry, field, key):
        path = _inside(self.raw, entry[field])
        self.cover(path, field, entry["sha256"][field])
        with np.load(path, allow_pickle=False) as data:
            return data[key].copy()

    def origin(self, entry, manifest):
        for field in ("observations_file", "oracle_file", "counts_file", "rng_file"):
            self.attempt(
                f"{entry['id']}:{field}",
                lambda f=field: self.cover(_inside(self.raw, entry[f]), f, entry["sha256"][f]),
            )
        theta = self.array(entry, "observations_file", "branch_theta")
        aliases = self.array(entry, "observations_file", "branch_alias").reshape(-1)
        flat = theta.reshape(-1, theta.shape[-1])
        if (
            aliases.dtype.kind not in "iu"
            or len(aliases) != len(flat)
            or np.any(aliases < 0)
            or np.any(aliases > np.arange(len(flat)))
            or not np.array_equal(flat, flat[aliases])
        ):
            raise ValueError("invalid or inconsistent original parameter aliases")
        required = (
            "branch_prompt_indices",
            "branch_actions",
            "branch_categories",
            "branch_old_logp",
            "adam_m",
            "adam_v",
            "adam_step",
            "branch_adam_m",
            "branch_adam_v",
            "branch_adam_step",
            "checkpoint_grad",
            "checkpoint_grad_is_none",
        )
        path = _inside(self.raw, entry["observations_file"])
        with np.load(path, allow_pickle=False) as data:
            absent = [name for name in required if name not in data]
            if absent:
                self.missing_input(
                    entry["id"], "missing original state/bank arrays: " + ", ".join(absent)
                )
        for ai, step in enumerate(manifest["anchors"]):
            updates = theta[ai, :, 1:] - theta[ai, :, :1]
            self.origins.append(
                {
                    "record_type": "origin",
                    "trajectory_id": entry["id"],
                    "seed": entry["seed"],
                    "arm": entry["arm"],
                    "step": step,
                    "original_role": entry["split"],
                    "v3_role": "PUBLIC_HISTORICAL_DEVELOPMENT",
                    "bank_roles": manifest["bank_roles"],
                    "actual_update_shape": list(updates.shape),
                    "zero_contrast_count": int(np.all(updates == 0, axis=-1).sum()),
                    "observations_sha256": entry["sha256"]["observations_file"],
                    "source_config_sha256": manifest.get("config_sha256"),
                    "sample_key_rule": (
                        "trajectory_id:step:bank:operation with original array indices"
                    ),
                    "state_components_status": "MISSING_INPUTS" if absent else "PRESENT",
                }
            )

    def packet(self, path, entries, metadata, manifest):
        from src.modeling_contrast.packet_codec import decode_arrays
        from src.modeling_contrast.packet_reconstruction import reconstruct_measurements

        record = self.metadata(path, "paid_packet", "packet_metadata.json.gz")
        entry = entries[record["trajectory_id"]]
        if record["input_parameters_sha256"] != entry["sha256"]["observations_file"]:
            raise ValueError("packet parameter binding differs from original manifest")
        arrays_path = path.parent / "packet_arrays.npz"
        self.cover(arrays_path, "paid_primitives", record["packet_arrays_sha256"])
        with np.load(arrays_path, allow_pickle=False) as packed:
            decoded = decode_arrays(packed)
        reconstructed = reconstruct_measurements(decoded, record, force=True)
        for key in ("raw", "helmert", "covariance"):
            if key in decoded and not np.array_equal(decoded[key], reconstructed[key]):
                raise ValueError(f"primitive replay differs from saved {key}")
        self.packet_count += 1
        self.primitive_count += sum("sample_" in key for key in decoded)
        if any(
            not all(key in item for key in ("cost", "policy_fingerprints", "sample_layout"))
            for item in record["packets"]
        ):
            self.missing_input(path, "packet ledger, alias identity, or sample layout absent")
        if any(
            "counts" in item.get("fields", []) and "actions" not in item.get("fields", [])
            for row in record["packets"]
            for item in row.get("sample_layout", [])
        ):
            self.missing_input(
                "legacy_count_only_draws",
                "original reused count packets retain counts without individual actions; "
                "no action sequence reconstructed",
            )
        # V2's replica streams are historical measurement streams, never a V3 pilot split.
        if any("pilot_main_ref_role" not in item for item in record["packets"]):
            self.missing_input(
                "legacy_packet_roles",
                "V2 has no explicit V3 pilot/main/reference roles; "
                "do not relabel as independent pilot",
            )
        oracle = self.array(entry, "oracle_file", "branch_p")
        ai = int(record["anchor_index"])
        if not 0 <= ai < len(manifest["anchors"]):
            raise ValueError("packet anchor outside parent manifest")
        for bank in record["banks"]:
            truth = oracle[ai, bank, 1:] - oracle[ai, bank, :1]
            for target in range(2):
                raw = reconstructed["raw"][:, bank, target]
                if len(raw) < 2:
                    self.missing_input(path, "fewer than two replicas; variance is not estimable")
                    continue
                for view, prediction in (
                    ("raw_event", raw),
                    ("helmert", reconstructed["helmert"][:, bank, target] @ H.T),
                ):
                    row = independent_observation_statistics(
                        prediction, raw, truth[target], metadata, view
                    )
                    self.observations.append(
                        dict(
                            row,
                            trajectory_id=entry["id"],
                            seed=entry["seed"],
                            arm=entry["arm"],
                            anchor=manifest["anchors"][ai],
                            bank=bank,
                            target=TARGETS[target],
                            method=record["method"],
                            n=record["n"],
                            view=view,
                            packet_sha256=record["packet_arrays_sha256"],
                            provenance="RAW_PRIMITIVE_REPLAY",
                        )
                    )

    def prediction(self, path, entries, metadata, manifest):
        record = self.metadata(path, "frozen_prediction", "freeze_metadata.json.gz")
        models = record["models"]
        if not models:
            raise ValueError("empty prediction model inventory")
        entries_by_identity = {(entry["seed"], entry["arm"]): entry for entry in entries.values()}
        entry = entries_by_identity[(models[0]["seed"], models[0]["arm"])]
        if record["parameters_sha256"] != entry["sha256"]["observations_file"]:
            raise ValueError("prediction parameters differ from original manifest")
        predpath = path.parent / "predictions.npz"
        self.cover(predpath, "frozen_predictions", record["prediction_sha256"])
        with np.load(predpath, allow_pickle=False) as arrays:
            predictions = arrays["prediction"].copy()
            if not all(key in arrays for key in ("Q", "U", "C")):
                self.absent_model_matrices += 1
        if len(predictions) != len(models):
            raise ValueError("prediction/model count mismatch")
        theta = self.array(entry, "observations_file", "branch_theta")
        oracle = self.array(entry, "oracle_file", "branch_p")
        expected = {}
        for prediction, identity in zip(predictions, models, strict=True):
            array = np.ascontiguousarray(prediction)
            digest = hashlib.sha256(
                str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
            ).hexdigest()
            if digest != identity["prediction_hash"]:
                raise ValueError("raw prediction array hash mismatch")
            if (identity["seed"], identity["arm"]) != (entry["seed"], entry["arm"]):
                raise ValueError("mixed trajectory identities in prediction unit")
            ai = manifest["anchors"].index(identity["anchor"])
            banks = int(identity["fit_banks"])
            truth = oracle[ai, 10:14, 1:] - oracle[ai, 10:14, :1]
            fit = theta[ai, :banks, 1:] - theta[ai, :banks, :1]
            rows = independent_model_statistics(prediction @ H.T, truth, metadata, fit, identity)
            for row in rows:
                key = tuple(str(row.get(field, "")) for field in MODEL_KEYS)
                if key in expected:
                    raise ValueError("duplicate model metric identity")
                expected[key] = row
                self.models.append({k: v for k, v in row.items() if not k.endswith("_abs_errors")})
        metric_path = path.parent / "metrics.jsonl.gz"
        if not metric_path.is_file():
            self.missing_input(
                metric_path, "original unit metric table absent; raw recomputation retained"
            )
        else:
            self.cover(metric_path, "original_metric_crosscheck")
            seen = set()
            with gzip.open(metric_path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    key = tuple(str(row.get(field, "")) for field in MODEL_KEYS)
                    if key not in expected or key in seen:
                        raise ValueError("unexpected or duplicate stored all/active metric row")
                    seen.add(key)
                    for field in MODEL_FIELDS:
                        actual, wanted = row[field], expected[key][field]
                        if wanted is None:
                            if actual is not None:
                                raise ValueError(f"metric mismatch: {field}")
                        elif not np.allclose(actual, wanted, rtol=1e-9, atol=1e-12):
                            raise ValueError(f"metric mismatch: {field}")
                    self.metric_comparisons += 1
            if seen != set(expected):
                raise ValueError("stored table omits recomputed all/active rows")
        rows = full_rank_equivalence(predictions, models)
        for row in rows:
            row["source_prediction_sha256"] = record["prediction_sha256"]
            if row["status"] == "FAIL":
                raise ValueError("r=k original predictions are not equivalent to matched C3")
        self.equivalence.extend(rows)

    def legacy_observations(self, legacy_root):
        """Replay N1 using its own original parent identity, never the N4 parent."""
        previous_raw = self.raw
        self.raw = Path(legacy_root).resolve()
        try:
            manifest = self.read(self.raw / "manifest.json", "legacy_N1_parent_manifest")
            metadata = self.read(
                self.raw / "probe_metadata.json",
                "legacy_N1_probe_identity",
                manifest["probe_identity_sha256"],
            )
            entries = {entry["id"]: entry for entry in manifest["trajectories"]}
            paths = sorted((self.run / "N1/packets").rglob("packets.json"))
            if not paths:
                self.missing_input(
                    "legacy_N1_packets", "no original N1 measurement packet units found"
                )
            start = len(self.observations)
            for path in paths:
                self.attempt(path, lambda p=path: self.packet(p, entries, metadata, manifest))
            keys = ("trajectory_id", "anchor", "bank", "method", "n", "target", "view")
            recovered = {
                tuple(str(row[key]) for key in keys): row for row in self.observations[start:]
            }
            comparison = self.run / "N1/OBSERVATION_COMPARISON.csv"
            if not comparison.is_file():
                self.missing_input(comparison, "original N1 raw/Helmert comparison table absent")
                return
            self.cover(comparison, "legacy_N1_table_crosscheck")
            with comparison.open() as stream:
                for row in csv.DictReader(stream):
                    key = tuple(str(row[field]) for field in keys)
                    if key not in recovered:
                        self.missing_input(
                            comparison, f"table row has no replayed primitive source: {key}"
                        )
                        continue
                    for field in (
                        "replicas",
                        "bias",
                        "rms_bias",
                        "rmse",
                        "empirical_variance",
                        "pX_error_ss",
                        "pX_truth_ss",
                        "v_error_ss",
                        "v_truth_ss",
                        "mass_residual_rms",
                    ):
                        if not np.isclose(
                            float(row[field]), recovered[key][field], rtol=1e-9, atol=1e-12
                        ):
                            raise ValueError(f"N1 raw/Helmert table mismatch: {key}: {field}")
                    self.metric_comparisons += 1
        finally:
            self.raw = previous_raw


def _pooled(rows):
    groups = {}
    for row in rows:
        fields = ("method", "observation", "n", "rank", "fit_banks", "target", "population")
        key = tuple(str(row.get(field, "")) for field in fields)
        item = groups.setdefault(
            key,
            dict(
                zip(fields, key, strict=True),
                rows=0,
                case_count=0,
                **{
                    f"{metric}_{kind}_ss": 0.0
                    for metric in ("event", "pX", "v")
                    for kind in ("error", "truth")
                },
            ),
        )
        item["rows"] += 1
        item["case_count"] += row["case_count"]
        for metric in ("event", "pX", "v"):
            for kind in ("error", "truth"):
                field = f"{metric}_{kind}_ss"
                item[field] += row[field]
    output = list(groups.values())
    lookup = {key: item for key, item in groups.items()}
    for key, row in list(groups.items()):
        if row["population"] != "all":
            continue
        active = lookup.get((*key[:-1], "active"), {})
        inactive = dict(row, population="inactive_all_minus_active")
        for field in (
            "rows",
            "case_count",
            *[f"{m}_{k}_ss" for m in ("event", "pX", "v") for k in ("error", "truth")],
        ):
            inactive[field] -= active.get(field, 0)
        output.append(inactive)
    return output


def audit_parent(config, parent, out):
    """Audit a run root (N4/raw + N4_validation), or explicit raw parent root.

    Optional config['q0']: run_root, legacy_parent_root, source_config_path, source_root. Existing
    output files are immutable. Missing Q0 inputs do not prohibit Q1 math work.
    """
    started = time.perf_counter()
    settings = config.get("q0", {})
    supplied = Path(parent).expanduser().resolve()
    raw = supplied / "N4/raw" if (supplied / "N4/raw/manifest.json").is_file() else supplied
    inferred = raw.parent.parent if raw.name == "raw" and raw.parent.name == "N4" else supplied
    run = Path(settings.get("run_root", inferred)).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    if out in (run, raw) or out.is_relative_to(raw) or out.is_relative_to(run / "N4_validation"):
        raise ValueError("Q0 output must not overwrite historical input roots")
    if settings.get("legacy_parent_root") and out.is_relative_to(
        Path(settings["legacy_parent_root"]).expanduser().resolve()
    ):
        raise ValueError("Q0 output must not overwrite historical input roots")
    names = (
        "Q0_SOURCE_AUDIT.json",
        "MISSING_INPUTS_V3.json",
        "PARENT_RECORD_INDEX.jsonl",
        "Q0_REPRODUCTION_zh.md",
        "Q0_RAW_HELMERT.csv",
        "Q0_ALL_ACTIVE_SQUARED_SUMS.csv",
        "Q0_R_EQUALS_K.csv",
    )
    if any((out / name).exists() for name in names):
        raise FileExistsError("Q0 result exists; choose a new immutable output directory")
    out.mkdir(parents=True, exist_ok=True)
    audit = _Audit(raw, run, out)
    manifest = audit.attempt(
        raw / "manifest.json", lambda: audit.read(raw / "manifest.json", "parent_manifest")
    )
    for stage in ("N4", "N4_validation"):
        audit.attempt(run / stage, lambda s=stage: audit.stage(run / s))
    if manifest is not None:
        try:
            if manifest["events"] != ["X", "S", "W", "I"] or manifest["operation_names"] != [
                "joint_0",
                "joint_1",
                "no_x_off_1",
            ]:
                raise ValueError("historical event/operation order differs")
            entries = {entry["id"]: entry for entry in manifest["trajectories"]}
            if len(entries) != len(manifest["trajectories"]):
                raise ValueError("duplicate parent trajectory identity")
            for name, field in (
                ("dataset.npz", "dataset_sha256"),
                ("probe_metadata.json", "probe_identity_sha256"),
                ("train_metadata.json", "train_identity_sha256"),
                ("parameter_layout.json", "parameter_layout_sha256"),
            ):
                audit.attempt(
                    name,
                    lambda n=name, f=field: audit.cover(
                        raw / n, "parent_global_identity", manifest[f]
                    ),
                )
            source_root = Path(
                settings.get("source_root", Path(__file__).parents[1] / "modeling_qualification")
            )
            for name, digest in manifest.get("source_hashes", {}).items():
                audit.attempt(
                    name,
                    lambda n=name, h=digest: audit.cover(
                        _inside(source_root, n), "collector_source", h
                    ),
                )
            config_path = Path(settings.get("source_config_path", raw / "resolved_config.json"))
            source_config = audit.attempt(
                config_path, lambda: audit.read(config_path, "complete_source_config")
            )
            if source_config is not None and _canonical(source_config) != manifest["config_sha256"]:
                audit.errors.append(
                    {"scope": str(config_path), "detail": "source config canonical hash mismatch"}
                )
            metadata = audit.attempt(
                "probe_metadata",
                lambda: audit.read(
                    raw / "probe_metadata.json", "probe_identity", manifest["probe_identity_sha256"]
                ),
            )
            if metadata is not None and len({item["prompt_id"] for item in metadata}) != len(
                metadata
            ):
                raise ValueError("duplicate probe identity")
            for entry in entries.values():
                audit.attempt(entry["id"], lambda e=entry: audit.origin(e, manifest))
            packet_paths = sorted((run / "N4_validation").rglob("packets.json"))
            model_paths = sorted((run / "N4_validation").rglob("freeze.json"))
            if not packet_paths:
                audit.missing_input(
                    "N4_packets", "no original packets.json/packet_arrays.npz found"
                )
            if not model_paths:
                audit.missing_input("N4_predictions", "no original frozen prediction units found")
            if metadata is not None:
                for path in packet_paths:
                    audit.attempt(path, lambda p=path: audit.packet(p, entries, metadata, manifest))
                for path in model_paths:
                    audit.attempt(
                        path, lambda p=path: audit.prediction(p, entries, metadata, manifest)
                    )
        except (ValueError, KeyError, TypeError) as exc:
            audit.errors.append({"scope": "parent_manifest", "detail": str(exc)})
    if settings.get("legacy_parent_root"):
        audit.attempt(
            "legacy_N1_replay", lambda: audit.legacy_observations(settings["legacy_parent_root"])
        )
    else:
        audit.missing_input(
            "legacy_N1_parent",
            "set q0.legacy_parent_root to the original M2 parent to reproduce N1 tables",
        )
    if audit.absent_model_matrices:
        audit.missing_input(
            "original_Q_U_C",
            f"{audit.absent_model_matrices} original prediction units did not retain Q/U/C arrays; "
            "no reconstructed matrices claimed as originals",
        )
    if not audit.equivalence:
        audit.missing_input(
            "r_equals_k_equivalence",
            "no original C5/C6 matched rank-equivalence records were available",
        )
    status = "FAIL" if audit.errors else "MISSING_INPUTS" if audit.missing else "PASS"
    result = {
        "schema_version": "modeling-v3-q0-v1",
        "status": status,
        "parent_root": str(raw),
        "run_root": str(run),
        "output_root": str(out),
        "source_files": len(audit.index),
        "origin_count": len(audit.origins),
        "packet_units_replayed": audit.packet_count,
        "primitive_arrays_replayed": audit.primitive_count,
        "raw_helmert_rows": len(audit.observations),
        "model_metric_rows_recomputed": len(audit.models),
        "stored_metric_rows_matched": audit.metric_comparisons,
        "rank_equivalence_rows": len(audit.equivalence),
        "missing_inputs": audit.missing,
        "errors": audit.errors,
        "wall_seconds": time.perf_counter() - started,
        "new_samples": 0,
        "policy_scoring_calls": 0,
        "optimizer_updates": 0,
        "summary_reconstruction_used": False,
        "historical_data_role": "PUBLIC_DEVELOPMENT_ONLY",
        "q1_self_contained_math_may_continue": True,
        "online_ssvc_certified": False,
    }
    _publish(
        out / names[1],
        json.dumps(
            {
                "status": "MISSING_INPUTS" if audit.missing else "NONE",
                "missing_inputs": audit.missing,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _publish(
        out / names[2],
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in [*audit.index.values(), *audit.origins]
        ),
    )
    _publish(out / names[4], _csv_text(audit.observations))
    _publish(out / names[5], _csv_text(_pooled(audit.models)))
    _publish(out / names[6], _csv_text(audit.equivalence))
    report = (
        f"# Q0 原件审计与复算\n\n状态：`{status}`。\n\n"
        f"核验文件 {len(audit.index)} 个；原点 {len(audit.origins)} 个；"
        f"从原始付费数组重放 packet {audit.packet_count} 个。"
        f"模型指标复算 {len(audit.models)} 行，已对照原始指标 {audit.metric_comparisons} 行。\n\n"
        "raw 的 v 为 X/S/W 三项之和；Helmert 还原后的 v 为 -I。两种公式只对零和差值等价。"
        "all/active 的活跃判据来自原始 fit 增量；inactive 单列为 all 减 active 的平方和。"
        "r=k 比较只与同观测、n、校准量、正则及噪声副本的 C3 比较，不作为真实截断收益。\n\n"
        "原始来源及哈希见 PARENT_RECORD_INDEX.jsonl。缺失项见 MISSING_INPUTS_V3.json；"
        "任何缺失原件均未从摘要或新采样反造。旧公开 seed 仅属 development。"
        "本次没有训练、概率评分或采样；自包含 Q1 数学验收可继续。\n\n"
        f"完整性错误 {len(audit.errors)} 项；缺失分类 {len(audit.missing)} 项。"
        "Q0 不能认证在线 SSVC 有效。\n"
    )
    _publish(out / names[3], report)
    result["output_hashes"] = {name: _sha(out / name) for name in names[1:]}
    _publish(
        out / names[0], json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return result
