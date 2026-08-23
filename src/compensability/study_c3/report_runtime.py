"""Fact-only Study C3 report rendering and evidence inventory helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping


def fact_report_markdown(facts: Mapping[str, object]) -> str:
    required = {"git_commit_sha", "model_snapshot_sha256", "single_seed_mechanism_pilot"}
    if not required.issubset(facts) or facts["single_seed_mechanism_pilot"] is not True:
        raise ValueError("Study C3 fact report provenance is incomplete")
    lines = [
        "# Study C3 Result Facts",
        "",
        "## Registered scope",
        "",
        "- model: Qwen2.5-VL-3B-Instruct",
        "- single_seed_mechanism_pilot: true",
        f"- git_commit_sha: {facts['git_commit_sha']}",
        f"- model_snapshot_sha256: {facts['model_snapshot_sha256']}",
        "",
        "## Artifact facts",
        "",
        "```json",
        json.dumps(dict(facts), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False),
        "```",
        "",
    ]
    return "\n".join(lines)


__all__ = ["fact_report_markdown"]
