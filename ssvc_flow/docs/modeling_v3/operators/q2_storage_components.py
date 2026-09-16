#!/usr/bin/env python3
"""Read-only Q2 component byte counts inside a Slurm CPU allocation.

Use --root candidate06/Q2_coverage_full --complete-sha256 HASH --out NEW_DIRECTORY.
Deploy the frozen measure_completed_cpu_output.py beside this helper. Apart from
top-level COMPLETE hashing, only directory names and stat metadata are read from
the measured tree. No ZIP headers, arrays or scientific values are opened. Two
metadata traversals must agree; output is exclusive and outside the input tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from contextlib import suppress
from pathlib import Path

MEASUREMENT_HELPER_SHA256 = "2b89329a1e7ac9666d4144737bee43c92a749ee08b562879f62712b875830982"
DEPENDENCY = Path(__file__).with_name("measure_completed_cpu_output.py")
if (
    DEPENDENCY.is_symlink()
    or hashlib.sha256(DEPENDENCY.read_bytes()).hexdigest() != MEASUREMENT_HELPER_SHA256
):
    raise ValueError("frozen adjacent measurement helper identity mismatch")
SPEC = importlib.util.spec_from_file_location("frozen_cpu_output_measurement", DEPENDENCY)
measurement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measurement)
MATCHING_N = (64, 256, 1024, 4096)
METRICS = (
    "regular_file_count",
    "regular_apparent_bytes",
    "regular_allocated_bytes",
    "unique_regular_inode_count",
    "unique_regular_apparent_bytes",
    "unique_regular_allocated_bytes",
)


def empty_counts():
    return dict.fromkeys(METRICS, 0)


def add(table, key, info):
    entry = table.setdefault(key, {"counts": empty_counts(), "seen": set()})
    counts, seen = entry["counts"], entry["seen"]
    counts["regular_file_count"] += 1
    counts["regular_apparent_bytes"] += info.st_size
    counts["regular_allocated_bytes"] += info.st_blocks * 512
    inode = (info.st_dev, info.st_ino)
    if inode not in seen:
        seen.add(inode)
        counts["unique_regular_inode_count"] += 1
        counts["unique_regular_apparent_bytes"] += info.st_size
        counts["unique_regular_allocated_bytes"] += info.st_blocks * 512


def counts(table, key):
    return table[key]["counts"] if key in table else empty_counts()


def classify(relative):
    parts, basename = relative.parts, relative.name
    origin, scope, acquisition, n = None, "stage_root", None, None
    if len(parts) >= 2 and parts[0] == "units":
        origin, scope = parts[1], "unit_root"
    if len(parts) == 4 and origin:
        match = re.fullmatch(r"(calibration|direct)_n([0-9]+)", parts[2])
        if match:
            acquisition, n, scope = match.group(1), int(match.group(2)), "measurement_cache"
        elif parts[2] == "error_decomposition":
            scope = "decomposition"
        else:
            scope = "other_unit_subtree"
    elif len(parts) == 5 and origin and parts[2] == "fits":
        scope = "fit"
    elif len(parts) > 3:
        scope = "other_subtree"
    cache_names = {
        "contributions.npz": "measurement_contributions",
        "raw_scores.npz": "measurement_raw_scores",
        "shared_origin_packet.npz": "shared_origin_packet",
        "estimates.npz": "measurement_estimates",
        "coefficients.npz": "measurement_coefficients",
    }
    if scope == "measurement_cache" and basename in cache_names:
        category = cache_names[basename]
    elif scope == "fit" and basename == "PREDICTIONS.npz":
        category = "fit_predictions"
    elif scope == "unit_root" and basename == "QUERY_REFERENCE.npz":
        category = "query_reference"
    elif scope == "decomposition" and basename == "ORIGIN_JACOBIAN_AND_BASELINES.npz":
        category = "decomposition_origin"
    elif scope == "decomposition" and relative.suffix == ".npz":
        category = "decomposition_design"
    elif relative.suffix in {".json", ".jsonl"}:
        category = "json_or_jsonl"
    else:
        category = "other_regular"
    return category, scope, acquisition, n, origin, basename


def scan_components(root_fd):
    """Stream stat records into mutually exclusive groups; never open data files."""
    overall, categories, groups, origins, caches = {}, {}, {}, {}, {}
    fingerprint = hashlib.sha256()
    directory_count = symlink_count = other_count = pending_count = 0
    root_device = os.fstat(root_fd).st_dev

    def record(relative, info):
        nonlocal directory_count, symlink_count, other_count, pending_count
        fingerprint.update(
            json.dumps(
                [str(relative), *measurement.identity(info), info.st_blocks], separators=(",", ":")
            ).encode()
            + b"\n"
        )
        if relative != Path(".") and "pending" in relative.name.lower():
            pending_count += 1
        if stat.S_ISDIR(info.st_mode):
            directory_count += 1
        elif stat.S_ISLNK(info.st_mode):
            symlink_count += 1
        elif stat.S_ISREG(info.st_mode):
            category, scope, acquisition, n, origin, basename = classify(relative)
            add(overall, "all", info)
            add(categories, category, info)
            add(groups, (category, scope, acquisition, n, origin, basename), info)
            add(origins, origin or "STAGE_ROOT", info)
            if scope == "measurement_cache":
                stratum = "matching" if n in MATCHING_N else "other"
                add(caches, (stratum,), info)
                add(caches, (stratum, n), info)
                add(caches, (stratum, n, acquisition), info)
                add(caches, (stratum, n, acquisition, category), info)
                add(caches, (stratum, n, "category", category), info)
        else:
            other_count += 1

    def walk(fd, relative):
        before = os.fstat(fd)
        if before.st_dev != root_device:
            raise ValueError("nested mount/device outside the measured filesystem")
        record(relative, before)
        with os.scandir(fd) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, measurement.DIRECTORY_FLAGS, dir_fd=fd)
                try:
                    if measurement.identity(os.fstat(child)) != measurement.identity(info):
                        raise ValueError("directory metadata changed before traversal")
                    walk(child, relative / name)
                finally:
                    os.close(child)
            else:
                record(relative / name, info)
        if measurement.identity(os.fstat(fd)) != measurement.identity(before):
            raise ValueError("directory metadata changed during traversal")

    walk(root_fd, Path("."))

    def budget_table(stratum, budgets):
        return {
            "total": counts(caches, (stratum,)),
            "per_n": {
                str(n): {
                    "total": counts(caches, (stratum, n)),
                    "composition": {
                        category: counts(caches, (stratum, n, "category", category))
                        for category in sorted(categories)
                        if (stratum, n, "category", category) in caches
                    },
                    "by_acquisition": {
                        acquisition: {
                            "total": counts(caches, (stratum, n, acquisition)),
                            "composition": {
                                category: counts(caches, (stratum, n, acquisition, category))
                                for category in sorted(categories)
                                if (stratum, n, acquisition, category) in caches
                            },
                        }
                        for acquisition in ("calibration", "direct")
                    },
                }
                for n in budgets
            },
        }

    return {
        "tree_total": {
            **counts(overall, "all"),
            "directory_count": directory_count,
            "symlink_count": symlink_count,
            "other_entry_count": other_count,
        },
        "category_totals": {key: entry["counts"] for key, entry in sorted(categories.items())},
        "group_rows": [
            dict(
                zip(
                    ("category", "scope", "acquisition", "n", "origin", "basename"),
                    key,
                    strict=True,
                ),
                **entry["counts"],
            )
            for key, entry in sorted(groups.items(), key=lambda item: repr(item[0]))
        ],
        "origin_totals": {key: entry["counts"] for key, entry in sorted(origins.items())},
        "q3_matching_cache_n": budget_table("matching", MATCHING_N),
        "q2_other_cache_n": budget_table(
            "other", sorted({key[1] for key in caches if len(key) == 2 and key[0] == "other"})
        ),
        "metadata_inventory_sha256": fingerprint.hexdigest(),
        "symlink_count": symlink_count,
        "pending_name_count": pending_count,
    }


def measure_q2_components(root, complete_sha256, out):
    measurement.enforce_cpu_environment()
    if re.fullmatch(r"[0-9a-f]{64}", complete_sha256) is None:
        raise ValueError("lowercase COMPLETE SHA256 required")
    root, out = measurement.physical_path(root), measurement.physical_path(out)
    expected_root = (
        measurement.PHYSICAL_SSVC_ROOT / measurement.CANDIDATE_RELATIVE / "Q2_coverage_full"
    )
    if root != expected_root:
        raise ValueError("only fixed candidate06/Q2_coverage_full may be measured")
    if out == measurement.PHYSICAL_SSVC_ROOT or out.is_relative_to(root):
        raise ValueError("new output directory must be outside the measured tree")
    root_fd = measurement.open_directory(root)
    out_parent_fd = out_fd = None
    try:
        initial = measurement.complete_identity(root_fd)
        if initial["sha256"] != complete_sha256:
            raise ValueError("initial COMPLETE SHA256 mismatch")
        out_parent_fd = measurement.open_directory(out.parent)
        os.mkdir(out.name, mode=0o700, dir_fd=out_parent_fd)
        out_fd = os.open(out.name, measurement.DIRECTORY_FLAGS, dir_fd=out_parent_fd)
        intent = {
            "schema": "ssvc-candidate06-q2-storage-components-1",
            "status": "STARTED",
            "root": str(root),
            "out": str(out),
            "started_utc": measurement.utc(),
            "expected_complete_sha256": complete_sha256,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "measurement_dependency_sha256": MEASUREMENT_HELPER_SHA256,
        }
        measurement.write_new(out_fd, "INTENT.json", intent)
        components = scan_components(root_fd)
        independent = measurement.scan_tree(root_fd)
        final = measurement.complete_identity(root_fd)
        if initial != final:
            raise ValueError("COMPLETE changed during measurement")
        if components["metadata_inventory_sha256"] != independent["metadata_inventory_sha256"]:
            raise ValueError("tree metadata changed between component and independent scans")
        for key, value in components["tree_total"].items():
            if value != independent["total"][key]:
                raise ValueError("component totals differ from independent full-tree totals")
        for key in ("regular_file_count", "regular_apparent_bytes", "regular_allocated_bytes"):
            if (
                sum(row[key] for row in components["category_totals"].values())
                != independent["total"][key]
            ):
                raise ValueError("mutually exclusive component reconciliation failed")
        fresh_fd = measurement.open_directory(root)
        try:
            if (os.fstat(fresh_fd).st_dev, os.fstat(fresh_fd).st_ino) != (
                os.fstat(root_fd).st_dev,
                os.fstat(root_fd).st_ino,
            ):
                raise ValueError("measured root pathname changed")
        finally:
            os.close(fresh_fd)
        report = {
            **intent,
            **components,
            "status": "MEASURED_COMPONENTS",
            "finished_utc": measurement.utc(),
            "complete_hash_unchanged": True,
            "reconciliation": {
                "matches_independent_full_tree_scan": True,
                "path_count_and_byte_groups_are_additive": True,
                "unique_inode_group_totals_are_additive": False,
                "independent_tree_total": independent["total"],
            },
            "budget_scope": (
                "n grouping is measurement-cache paths only, including their diagnostics/COMPLETE "
                "JSON. Fit n is not inferred from design hashes; fit/error/unit/stage files "
                "remain separate."
            ),
            "space_scope": (
                "Apparent bytes are per regular-file path; allocated bytes are st_blocks*512. "
                "Unique regular inode bytes deduplicate only within each reported group. "
                "No exclusive physical space or shared quota guarantee."
            ),
            "array_parameter_scope": (
                "PREDICTIONS.npz contains predictions, query geometry and model parameters "
                "together; this stat-only receipt does not split ZIP members."
            ),
            "Q3_storage_estimate_bytes": None,
            "interpretation": (
                "Actual Q2 components only. Observed compression/size is not a guaranteed "
                "ratio for a new Q3 seed. No new time, memory or Q3 disk estimate."
            ),
            "zip_headers_read": False,
            "scientific_data_values_read": False,
            "all_original_hashes_verified": False,
            "scientific_originals_modified": False,
            "symlinks_followed": False,
            "pending_names_excluded": False,
        }
        measurement.write_new(out_fd, "COMPONENTS.json", report)
        return report
    except BaseException as exc:
        if out_fd is not None:
            with suppress(OSError):
                measurement.write_new(
                    out_fd,
                    "FAILED.json",
                    {
                        "status": "FAILED_NO_VALID_COMPONENT_MEASUREMENT",
                        "timestamp_utc": measurement.utc(),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
        raise
    finally:
        for fd in (out_fd, out_parent_fd, root_fd):
            if fd is not None:
                os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--complete-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = measure_q2_components(args.root, args.complete_sha256, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "receipt": str(Path(report["out"]) / "COMPONENTS.json"),
                "regular_apparent_bytes": report["tree_total"]["regular_apparent_bytes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            json.dumps({"status": "FAILED", "type": type(error).__name__, "message": str(error)}),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
