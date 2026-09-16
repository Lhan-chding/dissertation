#!/usr/bin/env python3
"""Recompute Q1 subgroup tails from frozen residuals, on Slurm CPUs only.

Classification authority is the hash-bound Q1 producer inventory: its
``identical_policy`` records np.array_equal on complete endpoint probability
arrays. This helper verifies the sampling-identity chain but does NOT reload
historical probability arrays or infer policy identity from zero event truth.
No frozen product module, training, sampling, test or confirmation output is
modified. Production requires SSVC_V3_CHECKOUT and verifies the pinned candidate06
manifest and all 1025 files before importing src from checkout/ssvc_flow. The
operator itself may live anywhere outside the checkout. CLI has no fixture
bypass; tiny tests explicitly provide fixture=True and fixture_import_root.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

# Product modules are imported only after a verified deployment boundary.
SNAPSHOT_SHA256 = "cd703f28725212aa17e86ac8d0be82890f253c955aba7293fc9d84728d34b52d"
SNAPSHOT_FILE_COUNT = 1025
atomic_bytes = atomic_json = canonical_hash = finalize_run = None
sha256_file = verify_manifest = _flat_summary = None


def _local_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_snapshot_file(checkout, relative):
    if not isinstance(relative, str):
        raise ValueError("Snapshot paths must be strings")
    path = Path(relative)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("Unsafe snapshot path")
    actual = checkout
    for part in path.parts:
        actual /= part
        if actual.is_symlink():
            raise ValueError("Snapshot entries may not traverse symlinks")
    if not actual.is_file() or not actual.resolve().is_relative_to(checkout):
        raise ValueError("Snapshot entry missing or outside checkout")
    return actual


def _verify_snapshot(checkout):
    checkout = Path(checkout).resolve()
    path = _safe_snapshot_file(checkout, "CODE_SNAPSHOT_MANIFEST.json")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("candidate06 CODE_SNAPSHOT_MANIFEST SHA-256 differs")
    files = json.loads(raw)
    if not isinstance(files, dict) or len(files) != SNAPSHOT_FILE_COUNT:
        raise ValueError("candidate06 requires exactly 1025 bound files")
    for relative, expected in files.items():
        if _local_sha256(_safe_snapshot_file(checkout, relative)) != _sha(expected):
            raise ValueError("Frozen candidate06 file changed: " + relative)
    flow = checkout / "ssvc_flow"
    actual_src = {str(p.relative_to(checkout)) for p in (flow / "src").rglob("*.py")}
    expected_src = {p for p in files if p.startswith("ssvc_flow/src/") and p.endswith(".py")}
    if not actual_src or actual_src != expected_src:
        raise ValueError("Frozen candidate06 src inventory differs")
    return checkout, files


def _load_frozen_dependencies(*, fixture_import_root=None):
    """No production path fallback, environment hash override or CLI fixture bypass."""
    import importlib

    if fixture_import_root is None:
        value = os.environ.get("SSVC_V3_CHECKOUT")
        if not value or not Path(value).is_absolute():
            raise ValueError("SSVC_V3_CHECKOUT must name the absolute frozen candidate06 checkout")
        checkout, inventory = _verify_snapshot(value)
        flow = checkout / "ssvc_flow"
        mode = "PINNED_CANDIDATE06"
    else:
        flow = Path(fixture_import_root).resolve()
        if (
            not (flow / "src/modeling_v3/io.py").is_file()
            or not (flow / "src/modeling_v3/q1_results.py").is_file()
        ):
            raise ValueError("Tiny fixture requires an explicit valid src import root")
        checkout, inventory, mode = None, None, "EXPLICIT_TINY_FIXTURE_IMPORT_ROOT"
    for name, module in list(sys.modules.items()):
        if name == "src" or name.startswith("src."):
            filename = getattr(module, "__file__", None)
            if not filename or not Path(filename).resolve().is_relative_to(flow / "src"):
                raise ValueError("Existing src module was imported from a different checkout")
    # Remove every competing directory that exposes src; this is not dependent
    # on the helper's own location or the working directory.
    sys.path[:] = [str(flow)] + [
        p
        for p in sys.path
        if Path(p or Path.cwd()).resolve() != flow and not (Path(p or Path.cwd()) / "src").exists()
    ]
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()
    product_io = importlib.import_module("src.modeling_v3.io")
    product_q1 = importlib.import_module("src.modeling_v3.q1_results")
    for module in (product_io, product_q1):
        filename = Path(module.__file__).resolve()
        if not filename.is_relative_to(flow / "src"):
            raise ValueError("Imported src is outside verified candidate06")
        if inventory is not None and _local_sha256(filename) != inventory.get(
            str(filename.relative_to(checkout))
        ):
            raise ValueError("Imported dependency differs from verified snapshot")
    for name in (
        "atomic_bytes",
        "atomic_json",
        "canonical_hash",
        "finalize_run",
        "sha256_file",
        "verify_manifest",
    ):
        globals()[name] = getattr(product_io, name)
    globals()["_flat_summary"] = product_q1._flat_summary
    actual_source = product_io.source_identity(flow)
    if inventory is not None:
        expected = {
            str(Path(p).relative_to("ssvc_flow")): h
            for p, h in inventory.items()
            if p.startswith("ssvc_flow/src/") and p.endswith(".py")
        }
        if actual_source["files"] != expected:
            raise ValueError("Actual imported src tree differs from pinned inventory")
    return {
        "mode": mode,
        "checkout": str(checkout) if checkout else None,
        "src_import_root": str(flow),
        "snapshot_manifest_sha256": SNAPSHOT_SHA256 if inventory is not None else None,
        "snapshot_verified_file_count": len(inventory) if inventory is not None else None,
        "actual_src_sha256": actual_source["sha256"],
        "actual_src_files": actual_source["files"],
        "dependencies": {
            str(Path(m.__file__).resolve()): _local_sha256(m.__file__)
            for m in (product_io, product_q1)
        },
    }


STRATA = (
    "SELECTED_NONIDENTICAL",
    "SELECTED_EXACT_ALIAS",
    "EXPLICIT_IDENTICAL_CONTROL",
)
HEX = re.compile(r"^[0-9a-f]{64}$")


def _json(path):
    return json.loads(Path(path).read_text())


def _seed(*parts):
    return int(canonical_hash(["modeling-v3", *parts])[:16], 16)


def _sha(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError("Required original SHA-256 is missing or malformed")
    return value


def _verified_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(expected):
        raise ValueError(f"Original JSON hash mismatch: {path}")
    return json.loads(raw)


def _scope(fixture):
    if not fixture and (
        platform.system() != "Linux"
        or not str(os.environ.get("SLURM_JOB_ID", "")).isdigit()
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("SLURM_JOB_GPUS", "").strip()
        or os.environ.get("SLURM_STEP_GPUS", "").strip()
    ):
        raise RuntimeError("Full Q1 subgroup analysis requires Linux Slurm CPU-only execution")


def _classification(declared):
    inventory = declared.get("observed_case_inventory", {})
    same = inventory.get("identical_policy")
    distance = inventory.get("max_absolute_action_probability_difference")
    if (
        type(same) is not bool
        or type(distance) not in (int, float)
        or not math.isfinite(distance)
        or distance < 0
        or same != (distance == 0)
    ):
        raise ValueError("Missing/conflicting complete-policy identity inventory")
    if declared["case"] == "IDENTICAL_POLICY":
        if not same:
            raise ValueError("Explicit identical control conflicts with producer identity")
        return STRATA[2]
    if declared["case"] != "IDENTITY_HASH_SELECTED":
        raise ValueError("Only original Q1 selected pairs and explicit controls are allowed")
    return STRATA[1] if same else STRATA[0]


def _sampling_chain(source_root, complete, declared, originals, fixture):
    """Read small original identities/receipts; bind large statistics by recorded hashes."""
    name = declared["id"]
    repeats = declared["repeats"]
    if len(originals) != repeats or [r.get("repeat") for r in originals] != list(range(repeats)):
        raise ValueError("Complete ordered original repeat identity chain required")
    policy = set()
    prompt_hashes = set()
    chain = []
    strata = _classification(declared)
    same = strata != STRATA[0]
    for record in originals:
        repeat = record["repeat"]
        prefix = f"units/{name}/repeat{repeat:03d}/"
        files = complete["files"]
        for item, field in [
            ("statistics.npz", "statistics_sha256"),
            ("sampling_identity.json", "sampling_identity_sha256"),
            ("COMPLETE.json", "completion_sha256"),
        ]:
            if files.get(prefix + item) != _sha(record.get(field)):
                raise ValueError("Analysis ORIGINALS differs from frozen Q1 file inventory")
        if (
            Path(record["statistics_path"]).resolve()
            != (source_root / prefix / "statistics.npz").resolve()
        ):
            raise ValueError("Original statistics path does not match its frozen unit/repeat")
        identity = _verified_json(
            source_root / prefix / "sampling_identity.json", record["sampling_identity_sha256"]
        )
        receipt = _verified_json(
            source_root / prefix / "COMPLETE.json", record["completion_sha256"]
        )
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("summary", {}).get("repeat") != repeat
            or any(
                receipt.get("files", {}).get(file) != record[field]
                for file, field in [
                    ("statistics.npz", "statistics_sha256"),
                    ("sampling_identity.json", "sampling_identity_sha256"),
                ]
            )
        ):
            raise ValueError("Original repeat completion/identity/statistics chain differs")
        seed = _seed(name, repeat)
        expected = {
            "seed": seed,
            "n": declared["n"],
            "proposal": declared["proposal"],
            "pilot_rng": _seed(seed, "pilot"),
            "main_rng": _seed(seed, "main"),
            "record_hash": canonical_hash([name, repeat, seed]),
            "anchor": declared["anchor"],
            "bank": declared["bank"],
            "local_bank_index": declared["local_bank_index"],
            "candidate_index": 0 if declared["case"] == "IDENTICAL_POLICY" else 1,
            "baseline_index": 0,
            "historical_reused_fixed_policies": declared["historical_reused_fixed_policies"],
            "independent_count_status": "IDENTICAL_POLICY_ALIAS_ZERO_NO_DRAWS"
            if same
            else "SAMPLED",
        }
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("Sampling identity disagrees with repeat/policy/alias/RNG contract")
        expected_name = (
            f"{identity.get('origin_id')}_a{declared['anchor']}_b{declared['local_bank_index']}_"
            f"{declared['case']}_{declared['proposal']}_n{declared['n']}"
        )
        if expected_name != name or identity["pilot_rng"] == identity["main_rng"]:
            raise ValueError("Sampling origin identity or pilot/main independence differs")
        source_policy = identity.get("source_policy_files")
        if not isinstance(source_policy, str) or not source_policy:
            raise ValueError("Sampling identity lacks original full-policy file binding")
        policy.add((source_policy, _sha(identity.get("source_policy_sha256"))))
        prompt_hashes.add(_sha(identity.get("prompt_metadata_sha256")))
        if not fixture:
            expected_count = 0 if same else declared["n"] * 72
            if receipt["summary"].get("independent_count_generated_actions") != expected_count:
                raise ValueError("Original alias versus sampled-count billing differs")
        chain.append([repeat, record["sampling_identity_sha256"], record["completion_sha256"]])
    if len(policy) != 1 or len(prompt_hashes) != 1:
        raise ValueError("Original full-policy/probe binding changed across fixed-policy repeats")
    return {
        "repeats_verified": repeats,
        "identity_chain_sha256": canonical_hash(chain),
        "source_policy_binding_not_reread": dict(
            zip(("path", "sha256"), next(iter(policy)), strict=True)
        ),
        "prompt_metadata_sha256": next(iter(prompt_hashes)),
        "policy_pair_identity": [
            identity["origin_id"],
            *next(iter(policy)),
            declared["anchor"],
            declared["local_bank_index"],
            identity["candidate_index"],
            0,
        ],
    }


def analyze(
    analysis_root, out, *, expected_complete_sha256, fixture=False, fixture_import_root=None
):
    _scope(fixture)
    if (fixture_import_root is not None) != bool(fixture):
        raise ValueError("Only tiny fixtures may supply an explicit fixture import root")
    started = time.perf_counter()
    runtime = _load_frozen_dependencies(fixture_import_root=fixture_import_root)
    analysis_root, out = Path(analysis_root).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(analysis_root) or analysis_root.is_relative_to(out):
        raise ValueError("New derived output must not exist or overlap immutable analysis input")
    _verified_json(analysis_root / "COMPLETE.json", expected_complete_sha256)
    manifest = verify_manifest(analysis_root)
    binding = manifest["binding"]
    report = _json(analysis_root / "Q1_ANALYSIS.json")
    if (
        manifest.get("status") != "COMPLETE"
        or binding.get("kind") != "Q1_ORIGINALS_DERIVED_SUMMARY"
        or binding.get("fixture") is not fixture
        or _json(analysis_root / "BINDING.json") != binding
        or report.get("stage") != "Q1_DESCRIPTIVE_ANALYSIS"
        or report.get("fixture") is not fixture
        or report.get("scientific_status")
        != ("FIXTURE_NOT_AUTHORIZATION" if fixture else "DEVELOPMENT_ONLY_NOT_CERTIFIED")
        or report.get("originals_modified") is not False
        or report.get("new_samples_generated") != 0
        or report.get("source_bindings") != binding.get("inputs")
    ):
        raise ValueError("Complete Q1-only development analysis binding required")
    _sha(binding.get("config_sha256"))
    _sha(binding.get("analysis_source_sha256"))
    if not fixture and binding["analysis_source_sha256"] != sha256_file(
        sys.modules["src.modeling_v3.q1_results"].__file__
    ):
        raise ValueError("Analysis producer differs from frozen candidate06 q1_results")
    if not binding["inputs"]:
        raise ValueError("Q1 original completion binding missing")
    source_units = {}
    source_receipts = []
    for root_name, digest in binding["inputs"].items():
        source_root = Path(root_name).resolve()
        if out.is_relative_to(source_root) or source_root.is_relative_to(out):
            raise ValueError("Derived output overlaps original Q1 source")
        complete = _verified_json(source_root / "COMPLETE.json", digest)
        source = complete.get("summary", {})
        if (
            complete.get("status") != "COMPLETE"
            or source.get("stage") != "Q1"
            or source.get("pilot") is not False
            or source.get("scientific_status") != "DEVELOPMENT_ONLY"
            or complete.get("binding", {}).get("config_sha256") != binding["config_sha256"]
            or source.get("unit_count") != len(source.get("units", []))
        ):
            raise ValueError("Source completion must bind nonpilot Q1 development only")
        if not fixture:
            bound_sources = complete.get("binding", {}).get("source_hashes", {})
            bound_src = {p: h for p, h in bound_sources.items() if p.startswith("src/")}
            if not bound_src or any(
                runtime["actual_src_files"].get(p) != h for p, h in bound_src.items()
            ):
                raise ValueError("Q1 measurement source is not the verified candidate06 code")
        for declared in source["units"]:
            name = declared.get("id")
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                raise ValueError("Unsafe or missing original Q1 unit identity")
            if name in source_units:
                raise ValueError("Duplicate original Q1 unit identity")
            source_units[name] = (source_root, complete, declared)
        source_receipts.append({"path": str(source_root / "COMPLETE.json"), "sha256": digest})
    if report.get("unit_count") != len(source_units):
        raise ValueError("Source and derived unit count differ")
    actual_units = {p.parent.name for p in (analysis_root / "units").glob("*/SUMMARY.json")}
    if actual_units != set(source_units):
        raise ValueError("Derived unit set differs from original Q1 completion")
    if not actual_units:
        raise ValueError("No completed Q1 units available")
    if fixture and len(actual_units) > 8:
        raise ValueError("Tiny fixture limited to at most eight units")
    if not fixture and len(actual_units) != 192:
        raise ValueError("This full Q1 helper requires the completed 192-unit matrix")
    groups = collections.defaultdict(list)
    membership = []
    all_methods = None
    pair_classifications = {}
    for name in sorted(actual_units):
        source_root, complete, declared = source_units[name]
        unit = analysis_root / "units" / name
        summary = _json(unit / "SUMMARY.json")
        if any(
            summary.get(k) != declared.get(k)
            for k in ("id", "n", "proposal", "case", "seed", "arm", "anchor", "bank")
        ):
            raise ValueError("Derived SUMMARY identity differs from original producer inventory")
        repeats, prompts = summary.get("observed_repeats"), summary.get("prompt_count")
        if repeats != declared["repeats"] or summary.get("expected_repeats") != repeats:
            raise ValueError("Incomplete measurement repetitions cannot be silently discarded")
        if fixture:
            if not (1 <= repeats <= 8 and 1 <= prompts <= 8 and declared["n"] <= 64):
                raise ValueError("Fixture arrays must stay tiny")
        elif repeats != 200 or prompts != 72:
            raise ValueError("Production Q1 requires all 200 repeats and 72 original probes")
        methods = sorted(summary["methods"])
        if (
            not methods
            or set(methods) != set(declared["metrics"])
            or any(
                summary["methods"][m].get("status") != "RECOMPUTED_FROM_ORIGINAL_ESTIMATES"
                for m in methods
            )
        ):
            raise ValueError("All original measurement methods must be complete")
        if all_methods is not None and all_methods != methods:
            raise ValueError("Original methods differ across Q1 units")
        all_methods = methods
        strata = _classification(declared)
        chain = _sampling_chain(
            source_root, complete, declared, _json(unit / "ORIGINALS.json"), fixture
        )
        pair = tuple(chain["policy_pair_identity"])
        if pair in pair_classifications and pair_classifications[pair] != strata:
            raise ValueError("Same complete policy identity conflicts across n/proposal units")
        pair_classifications[pair] = strata
        member = {
            "unit_id": name,
            "stratum": strata,
            "n": declared["n"],
            "proposal": declared["proposal"],
            "seed": declared["seed"],
            "arm": declared["arm"],
            "anchor": declared["anchor"],
            "bank": declared["bank"],
            "repeats": repeats,
            "prompts": prompts,
            "producer_inventory": declared["observed_case_inventory"],
            **chain,
        }
        membership.append(member)
        groups[(declared["n"], declared["proposal"], strata)].append((unit, summary))
    if not fixture:
        expected = {
            (n, p, s)
            for n in (64, 256, 1024, 4096)
            for p in ("origin", "equal_pair_mixture")
            for s in STRATA
        }
        if set(groups) != expected:
            raise ValueError("Original subgroup/n/proposal matrix is incomplete")
    rows = []
    for (n, proposal, strata), items in sorted(groups.items()):
        residuals = {method: [] for method in all_methods}
        for unit, summary in items:
            shape = (summary["observed_repeats"], summary["prompt_count"], 4)
            exact_path = unit / "EXACT_RECORDS.npz"
            exact_bytes = exact_path.read_bytes()
            if (
                hashlib.sha256(exact_bytes).hexdigest()
                != manifest["files"][str(exact_path.relative_to(analysis_root))]
            ):
                raise ValueError("Exact residual ledger changed after manifest verification")
            with np.load(io.BytesIO(exact_bytes), allow_pickle=False) as arrays:
                truth = arrays["truth"]
                if truth.shape != shape[1:] or not np.isfinite(truth).all():
                    raise ValueError("Original exact truth shape or finiteness differs")
                for method in all_methods:
                    estimate = arrays["estimate_" + method]
                    residual = arrays["residual_" + method]
                    if (
                        estimate.shape != shape
                        or residual.shape != shape
                        or not np.isfinite(estimate).all()
                        or not np.array_equal(residual, estimate - truth)
                    ):
                        raise ValueError(
                            "Exact residual ledger differs from its estimate/truth originals"
                        )
                    if strata != STRATA[0] and (np.any(truth != 0) or np.any(residual != 0)):
                        raise ValueError(
                            "Producer exact alias contradicts retained truth or residuals"
                        )
                    residuals[method].append(residual.reshape(-1, 4))
        for method, parts in residuals.items():
            errors = np.concatenate(parts, axis=0)
            total_mse = float(np.mean(np.sum(errors**2, axis=1)))
            for event, index, sign in [
                ("X", 0, 1),
                ("S", 1, 1),
                ("W", 2, 1),
                ("I", 3, 1),
                ("delta_v", 3, -1),
            ]:
                rows.append(
                    {
                        "stratum": strata,
                        "n": n,
                        "proposal": proposal,
                        "method": method,
                        "event": event,
                        "unit_count": len(items),
                        "unit_repeat_count": sum(s["observed_repeats"] for _, s in items),
                        "four_event_total_mse": total_mse,
                        **_flat_summary(sign * errors[:, index]),
                    }
                )
    result = {
        "schema": "ssvc-v3-q1-nonalias-original-residuals-1",
        "fixture": fixture,
        "status": "FIXTURE_NOT_AUTHORIZATION" if fixture else "DEVELOPMENT_ONLY_NOT_CERTIFIED",
        "classification_authority": (
            "HASH_BOUND_Q1_PRODUCER_NP_ARRAY_EQUAL_COMPLETE_ENDPOINT_PROBABILITY_ARRAYS"
        ),
        "historical_full_policy_arrays_reread": False,
        "classification_from_zero_truth": False,
        "sampling_identity_chain_verified": True,
        "quantiles_from_pooled_original_residuals": True,
        "unit_quantiles_averaged": False,
        "new_samples_generated": 0,
        "originals_modified": False,
        "confirm_or_test_read": False,
        "selection_frozen": False,
        "online_ssvc": "NOT_CERTIFIED",
        "code_snapshot": {k: v for k, v in runtime.items() if k != "actual_src_files"},
        "analysis_binding": {
            "path": str(analysis_root),
            "complete_sha256": expected_complete_sha256,
            "manifest_sha256": sha256_file(analysis_root / "RUN_MANIFEST.json"),
            "binding": binding,
        },
        "source_completion_bindings": source_receipts,
        "definitions": {
            "signed_bias": "mean(estimate - exact truth)",
            "quantiles": (
                "NumPy linear quantiles of concatenated repeat-prompt residual cells; "
                "not confidence intervals"
            ),
            "variance": "ddof=0 pooled across original residual cells",
            "delta_v": "-delta_pI; opposite signed residuals, identical absolute errors",
            "sample_count_is_not_independent_policy_count": True,
        },
        "unit_count": len(membership),
        "row_count": len(rows),
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "Q1_NONALIAS_ANALYSIS.json", result)
    atomic_json(out / "UNIT_CLASSIFICATION.json", membership)
    flat = []
    for row in rows:
        r = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for label in ("signed", "absolute"):
            r.update(
                {label + "_" + q: value for q, value in row[label + "_residual_quantiles"].items()}
            )
        flat.append(r)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
    atomic_bytes(out / "Q1_NONALIAS_ERRORS.csv", stream.getvalue().encode())
    atomic_json(
        out / "RUN_RECEIPT.json",
        {
            "helper_sha256": sha256_file(__file__),
            "code_snapshot": runtime,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "execution_scope": "TINY_FIXTURE" if fixture else "REAL_SERVER_CPU_READONLY_ANALYSIS",
            "elapsed_seconds_before_manifest": time.perf_counter() - started,
            "original_sampling_identities_verified": sum(r["repeats_verified"] for r in membership),
            "original_repeat_completions_verified": sum(r["repeats_verified"] for r in membership),
            "historical_statistics_and_raw_packets_reread": False,
            "frozen_dependencies": {
                str(Path(p).resolve()): sha256_file(p)
                for p in [
                    sys.modules["src.modeling_v3.io"].__file__,
                    sys.modules["src.modeling_v3.q1_results"].__file__,
                ]
            },
        },
    )
    finalize_run(
        out,
        {
            "analysis_complete_sha256": expected_complete_sha256,
            "helper_sha256": sha256_file(__file__),
            "code_snapshot": runtime,
            "fixture": fixture,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--expected-complete-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(
        args.analysis_root, args.out, expected_complete_sha256=args.expected_complete_sha256
    )
    print(
        json.dumps(
            {"status": result["status"], "units": result["unit_count"], "rows": result["row_count"]}
        )
    )


if __name__ == "__main__":
    main()
