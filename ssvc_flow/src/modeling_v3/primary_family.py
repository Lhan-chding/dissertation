"""Finalize the four predeclared questions from verified CPU and VLM originals."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from .io import (
    atomic_json,
    canonical_hash,
    finalize_run,
    sha256_file,
    source_identity,
    verify_manifest,
)


def _file(path, filename):
    path = Path(path).resolve()
    return path / filename if path.is_dir() else path


def _binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def _completed_analysis(path, fixture):
    path = _file(path, "TEST_ANALYSIS.json")
    if path.name != "TEST_ANALYSIS.json" or not (path.parent / "COMPLETE.json").is_file():
        raise ValueError("complete original TEST_ANALYSIS.json is required")
    manifest = verify_manifest(path.parent)
    if manifest.get("status") != "COMPLETE":
        raise ValueError("analysis manifest is not complete")
    report = json.loads(path.read_text())
    if report.get("fixture") is not fixture:
        raise ValueError("analysis fixture and production scopes differ")
    return _binding(path)


def finalize_primary_family(
    config, cpu_lock, cpu_analysis, vlm_lock, vlm_analysis, out, *, fixture=False
):
    """Recompute stage comparisons before applying the common Holm family.

    Neither a p-value table nor a hand-written completion status is accepted.
    Each stage verifier rereads its original evaluation records. Incomplete or
    unresolved hypotheses retain the family's pending/unknown status.
    """
    if not fixture and (
        platform.system() != "Linux"
        or not str(os.environ.get("SLURM_JOB_ID", "")).isdigit()
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("primary-family recomputation requires server Slurm CPU execution")
    from .cpu_results import verify_cpu_primary_comparisons
    from .frozen_comparisons import summarize_primary_family
    from .schema import verify_selection_lock
    from .vlm_response import verify_vlm_selection_lock
    from .vlm_results import verify_vlm_primary_comparison

    cpu_path = _file(cpu_lock, "SELECTION_LOCK.json")
    vlm_path = _file(vlm_lock, "VLM_SELECTION_LOCK.json")
    cpu = verify_selection_lock(config, cpu_path)
    vlm = verify_vlm_selection_lock(config, vlm_path, fixture=fixture)
    parent = vlm.get("parent_cpu_selection", {})
    if (
        parent.get("sha256") != sha256_file(cpu_path)
        or not parent.get("path")
        or sha256_file(parent["path"]) != parent["sha256"]
    ):
        raise ValueError("VLM parent must bind the exact CPU selection lock")
    family = cpu["selected"].get("primary_comparison_family")
    if family is None:
        raise ValueError("CPU selection has no predeclared four-question family")
    cpu_original = _completed_analysis(cpu_analysis, fixture)
    vlm_original = _completed_analysis(vlm_analysis, fixture)
    cpu_results = verify_cpu_primary_comparisons(config, cpu_path, cpu_original, fixture=fixture)
    vlm_results = verify_vlm_primary_comparison(config, vlm_path, vlm_original, fixture=fixture)
    if cpu_results["family"] != family or vlm_results["family"] != family:
        raise ValueError("CPU and VLM hypotheses must share the unchanged frozen family")
    results = [*cpu_results["results"], *vlm_results["results"]]
    if (
        {r["hypothesis_id"] for r in cpu_results["results"]} != {"RQ1", "RQ2", "RQ3"}
        or len(cpu_results["results"]) != 3
        or [r["hypothesis_id"] for r in vlm_results["results"]] != ["RQ4"]
    ):
        raise ValueError("the final family requires three CPU questions and one VLM question")
    report = {
        "schema": "ssvc-v3-final-primary-family-1",
        "fixture": bool(fixture),
        "config_sha256": canonical_hash(config),
        "source_sha256": source_identity()["sha256"],
        "family": family,
        "family_sha256": canonical_hash(family),
        "selection_locks": {"CPU": _binding(cpu_path), "VLM": _binding(vlm_path)},
        "analyses": {"CPU": cpu_original, "VLM": vlm_original},
        "results": results,
        "summary": summarize_primary_family(family, results),
        "raw_stage_comparisons_recomputed_from_originals": True,
        "online_ssvc": "NOT_CERTIFIED",
    }
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "PRIMARY_FAMILY.json", report)
    finalize_run(
        root,
        {
            "config": report["config_sha256"],
            "source": report["source_sha256"],
            "family": report["family_sha256"],
        },
        metadata={"fixture": bool(fixture), "online_ssvc": "NOT_CERTIFIED"},
    )
    return report
