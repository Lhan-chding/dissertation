#!/usr/bin/env python3
"""Seal Q3 calibration requests and submission intentions; never submit a job.

Run on the server after the chosen candidate and Q3 selection lock are frozen.
All imports from the checkout follow complete snapshot hash verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path

SERVER_BASE = Path("/projects/varunssd/louis-ssvc/modeling_v3_20260915")
SERVER_PYTHON = Path("/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python")
QOS = "soujanya-poria-startfund-2026-03"
CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected SHA256 must be 64 lowercase hexadecimal digits")
    if Path(path).is_symlink() or not Path(path).is_file() or sha256(path) != expected:
        raise ValueError(f"original file missing, symlinked or changed: {path}")


def unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_fields)


def safe_file(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative:
        raise ValueError("nonempty relative inventory path required")
    parts = Path(relative).parts
    if Path(relative).is_absolute() or any(part in {"..", ".git"} for part in parts):
        raise ValueError(f"unsafe inventory path: {relative}")
    path = root / relative
    if root not in path.resolve().parents:
        raise ValueError(f"inventory path escapes root: {relative}")
    cursor = root
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"symlink inside inventory: {relative}")
    return path


def verify_snapshot(checkout, expected):
    checkout = Path(checkout).resolve()
    path = checkout / "CODE_SNAPSHOT_MANIFEST.json"
    check_hash(path, expected)
    manifest = read_json(path)
    # Deployed CODE_SNAPSHOT_MANIFEST uses a flat map; build receipts wrap it.
    files = manifest.get("files", manifest)
    if not isinstance(files, dict) or not files:
        raise ValueError("snapshot requires a complete nonempty files mapping")
    for relative, digest in files.items():
        check_hash(safe_file(checkout, relative), digest)
    source_root = checkout / "ssvc_flow/src"
    actual = {str(p.relative_to(checkout)) for p in source_root.rglob("*.py")}
    registered = {p for p in files if p.startswith("ssvc_flow/src/") and p.endswith(".py")}
    if not actual or actual != registered:
        raise ValueError("source inventory differs from the verified snapshot")
    required = {
        "ssvc_flow/configs/modeling_v3/protocol.json",
        "ssvc_flow/scripts/modeling_v3_cpu_array.sbatch",
        "ssvc_flow/scripts/run_modeling_v3_cpu_array.py",
        "ssvc_flow/scripts/run_modeling_v3_cpu_request.py",
    }
    if not required <= files.keys():
        raise ValueError("snapshot is missing required CPU protocol or launchers")
    return manifest


def verify_basis(path, expected):
    check_hash(path, expected)
    basis = read_json(path)
    if basis.get("schema") != "ssvc-v3-cpu-seed-inventory-basis-1":
        raise ValueError("unknown CPU seed inventory basis")
    roots = basis.get("roots", [])
    if len(roots) != 4:
        raise ValueError("exactly four independently verified historical roots required")
    normalized = []
    for entry in roots:
        root = Path(entry["root"])
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("historical inventory root missing or relative")
        root = root.resolve()
        if root in normalized or any(
            root in old.parents or old in root.parents for old in normalized
        ):
            raise ValueError("duplicate or overlapping historical inventory roots")
        check_hash(safe_file(root, entry["identity_file"]), entry["sha256"])
        normalized.append(root)
    return basis, normalized


def seal_json(path, value):
    """Atomic publication without overwriting even a partial earlier preparation."""
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def calibration_requests(config, lock_path, config_path, campaign, roots):
    """Only the complete calibration seed list; no subset, pilot or resume option."""
    cpu = config["cpu"]
    seeds, arms = cpu["interval_calibration_seeds"], cpu["arms"]
    if (
        len(seeds) != 20
        or len(set(seeds)) != 20
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(arms) != 2
        or set(arms) != {"X_BASE", "X_VALID"}
        or len(cpu["anchors"]) != 3
        or cpu["repeat_measurements_test"] != 20
    ):
        raise ValueError("complete 20-seed/two-arm/three-anchor/20-repeat calibration required")
    forbidden = set(cpu["locked_test_seeds"]) | set(
        cpu["orthogonal_generalization"]["train_rng_seeds"]
    )
    if set(seeds) & forbidden:
        raise ValueError("calibration seed roles overlap")
    rows = []
    for index, seed in enumerate(seeds):
        output = campaign / f"interval_calibration_seed{seed}"
        rows.append(
            {
                "index": index,
                "seed": seed,
                "output": str(output),
                "response_root": str(output / "interval_calibration/response"),
                "request": {
                    "argv": [
                        "validate-cpu",
                        "--config",
                        str(config_path),
                        "--out",
                        str(output),
                        "--role",
                        "interval_calibration",
                        "--lock",
                        str(lock_path),
                        "--existing-run-roots",
                        *map(str, roots),
                        "--seeds",
                        str(seed),
                        "--arms",
                        *arms,
                    ]
                },
            }
        )
    return rows


def workload(config, designs):
    cpu, coverage = config["cpu"], config["coverage"]
    trajectories = len(cpu["arms"])
    origins = trajectories * len(cpu["anchors"])
    units = origins * cpu["repeat_measurements_test"]
    source = trajectories * cpu["steps"]
    forks = (
        origins
        * (coverage["pool_banks"] + coverage["heldout_banks"])
        * len(cpu["local_candidate_arms"])
    )
    per_seed = {
        "trajectories": trajectories,
        "origins": origins,
        "source_optimizer_updates": source,
        "candidate_optimizer_updates": forks,
        "total_optimizer_updates": source + forks,
        "origin_repeat_units": units,
        "response_fits": units * len(designs),
    }
    return {
        "designs_per_origin_repeat": len(designs),
        "per_seed": per_seed,
        "all_20_seeds": {key: 20 * value for key, value in per_seed.items()},
        "wall_seconds_estimate": None,
        "timing_status": "AWAITING_FIRST_REAL_FULL_CALIBRATION_SEED",
        "timing_caution": "Collection/Q2 pilot timings do not estimate Q3 fit and originals I/O.",
    }


def submission_intents(checkout, candidate, bundle, list_hash):
    batches = [[0], list(range(1, 6)), list(range(6, 11)), list(range(11, 16)), list(range(16, 20))]
    exports = {
        **CPU_ENV,
        "SSVC_V3_CHECKOUT": str(checkout),
        "SSVC_V3_PYTHON": str(SERVER_PYTHON),
        "SSVC_V3_REQUEST_LIST": str(bundle / "REQUEST_LIST.json"),
        "SSVC_V3_REQUEST_LIST_SHA256": list_hash,
    }
    result = []
    for batch, indices in enumerate(batches):
        logs = candidate / "Q3_operator_logs"
        argv = [
            "sbatch",
            "--parsable",
            "--account=rose",
            f"--qos={QOS}",
            "--time=1-00:00:00",
            f"--job-name=ssvc-v3-q3-cal-b{batch}",
            f"--array={indices[0]}-{indices[-1]}%{len(indices)}",
            "--export=ALL",
            f"--output={logs}/%x_%A_%a.out",
            f"--error={logs}/%x_%A_%a.err",
            str(checkout / "ssvc_flow/scripts/modeling_v3_cpu_array.sbatch"),
        ]
        result.append(
            {
                "batch": batch,
                "indices": indices,
                "environment": exports,
                "argv": argv,
                "review_command_only": shlex.join(
                    ["env", *(f"{k}={v}" for k, v in exports.items()), *argv]
                ),
                "submission_status": "NOT_SUBMITTED",
                "operator_gate": (
                    "Verify the sealed preparation and inspect current QOS capacity. "
                    "Submit only this batch when its entire index count fits MaxSubmitPU=5. "
                    "Do not submit all batches with dependencies; pending indices consume quota. "
                    "Inspect first-seed completion, originals and timing before batch 1."
                ),
            }
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "verify"), required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--inventory-basis", required=True)
    parser.add_argument("--inventory-basis-sha256", required=True)
    parser.add_argument(
        "--prepared-sha256", help="Required in verify mode; printed after preparation"
    )
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("operator preparation is server-only; no local experimental execution")
    os.environ.update(CPU_ENV)
    sys.dont_write_bytecode = True
    candidate, checkout = Path(args.candidate_root).resolve(), Path(args.checkout).resolve()
    if (
        candidate.parent != SERVER_BASE.resolve()
        or not re.fullmatch(r"candidate[0-9]+", candidate.name)
        or not candidate.is_dir()
        or checkout != candidate / "checkout"
    ):
        raise ValueError(
            "existing candidate and its checkout must be inside the fixed server experiment root"
        )
    if not SERVER_PYTHON.is_file() or not os.access(SERVER_PYTHON, os.X_OK):
        raise ValueError("verified server Python is unavailable")
    if args.mode == "verify" and not args.prepared_sha256:
        raise ValueError("verify requires the independently recorded preparation SHA256")
    snapshot = verify_snapshot(checkout, args.snapshot_sha256)
    basis_path = Path(args.inventory_basis).resolve()
    basis, historical_roots = verify_basis(basis_path, args.inventory_basis_sha256)
    flow = checkout / "ssvc_flow"
    os.chdir(flow)
    sys.path.insert(0, str(flow))
    from src.modeling_v3.cpu_campaign import development_designs, scan_cpu_seed_inventory
    from src.modeling_v3.schema import load_config, verify_selection_lock

    config_path = flow / "configs/modeling_v3/protocol.json"
    lock_path = candidate / "Q3_SELECTION/SELECTION_LOCK.json"
    if lock_path.is_symlink():
        raise ValueError("selection lock must be an original file")
    config = load_config(config_path)
    lock = verify_selection_lock(config, lock_path)
    designs = development_designs(config, selected=lock["selected"])
    if not designs:
        raise ValueError("frozen calibration design matrix is empty")
    campaign = candidate / "Q3_campaign"
    bundle = candidate / "Q3_OPERATOR_INTERVAL_CALIBRATION"
    if campaign.is_symlink() or bundle.is_symlink():
        raise ValueError("Q3 outputs and preparation cannot be symlinked")
    roots = [*historical_roots, campaign]
    if any(
        root == campaign or root in campaign.parents or campaign in root.parents
        for root in historical_roots
    ):
        raise ValueError("new Q3 root overlaps historical inventory")
    rows = calibration_requests(config, lock_path, config_path, campaign, roots)
    binding = {
        "candidate_root": str(candidate),
        "checkout": str(checkout),
        "snapshot_manifest_sha256": args.snapshot_sha256,
        "snapshot_upstream_commit": snapshot.get("upstream_commit"),
        "config_path": str(config_path),
        "config_file_sha256": sha256(config_path),
        "selection_lock_path": str(lock_path),
        "selection_lock_sha256": sha256(lock_path),
        "selection_hash": lock["selection_hash"],
        "source_sha256": lock["source_sha256"],
        "inventory_basis_path": str(basis_path),
        "inventory_basis_sha256": args.inventory_basis_sha256,
        "historical_roots": list(map(str, historical_roots)),
        "future_q3_root_mapping": {
            "basis_document_path": basis["append_future_real_q3_root"],
            "chosen_path": str(campaign),
        },
    }
    manifest_path = bundle / "PREPARATION_MANIFEST.json"
    if args.mode == "verify":
        check_hash(manifest_path, args.prepared_sha256)
        manifest = read_json(manifest_path)
        if (
            manifest.get("binding") != binding
            or manifest.get("status") != "PREPARED_NOT_SUBMITTED"
            or manifest.get("operator_script_sha256") != sha256(Path(__file__).resolve())
        ):
            raise ValueError("prepared source/config/lock/inventory binding changed")
        for relative, digest in manifest["files"].items():
            check_hash(safe_file(bundle, relative), digest)
        actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {manifest_path.name}:
            raise ValueError("unregistered or missing preparation files")
        print(
            json.dumps(
                {"status": "VERIFIED_NOT_SUBMITTED", "preparation_sha256": args.prepared_sha256}
            )
        )
        return 0
    if args.prepared_sha256 is not None:
        raise ValueError("prepare does not accept an existing preparation hash")
    if bundle.exists() or (campaign.exists() and any(campaign.iterdir())):
        raise FileExistsError(
            "new empty campaign and unused preparation root required; preserve prior originals"
        )
    campaign.mkdir(exist_ok=True)
    collisions = scan_cpu_seed_inventory(roots, set(config["cpu"]["interval_calibration_seeds"]))
    if collisions:
        raise ValueError(
            "calibration seed collision before new collection: " + json.dumps(collisions)
        )
    if any(Path(row["output"]).exists() for row in rows):
        raise FileExistsError("planned calibration result already exists")
    if (candidate / "Q3_operator_logs").is_symlink():
        raise ValueError("operator logs cannot be symlinked outside the candidate")
    bundle.mkdir()
    (bundle / "requests").mkdir()
    (candidate / "Q3_operator_logs").mkdir(exist_ok=True)
    list_rows = []
    for row in rows:
        path = bundle / "requests" / f"{row['index']:02d}_seed{row['seed']}.json"
        seal_json(path, row["request"])
        list_rows.append({"path": str(path), "sha256": sha256(path)})
    seal_json(bundle / "REQUEST_LIST.json", list_rows)
    list_hash = sha256(bundle / "REQUEST_LIST.json")
    seal_json(
        bundle / "SUBMISSION_INTENTS.json",
        submission_intents(checkout, candidate, bundle, list_hash),
    )
    seal_json(
        bundle / "CALIBRATION_MATRIX.json",
        {
            "rows": rows,
            "selected_from_lock": lock["selected"],
            "designs": designs,
            "workload": workload(config, designs),
            "request_list_sha256": list_hash,
            "historical_seed_collisions": [],
            "seed_inventory_roots": list(map(str, roots)),
            "locked_test_requests_created": False,
            "generalization_requests_created": False,
            "new_gpu_calls": 0,
            "submissions": 0,
        },
    )
    files = {
        str(p.relative_to(bundle)): sha256(p) for p in sorted(bundle.rglob("*")) if p.is_file()
    }
    seal_json(
        manifest_path,
        {
            "schema": "ssvc-v3-q3-operator-preparation-1",
            "status": "PREPARED_NOT_SUBMITTED",
            "binding": binding,
            "files": files,
            "request_count": 20,
            "operator_script_sha256": sha256(Path(__file__).resolve()),
            "scientific_results_created": False,
            "gpu_authorization_created": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PREPARED_NOT_SUBMITTED",
                "manifest": str(manifest_path),
                "preparation_sha256": sha256(manifest_path),
                "request_list_sha256": list_hash,
                "requests": 20,
                "submissions": 0,
                "wall_seconds_estimate": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None
