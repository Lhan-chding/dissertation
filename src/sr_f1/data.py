"""Separate model-visible records from verifier-only worlds and labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .contract import PACKAGE, SEEDS, file_hash


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bounded_path(root, relative):
    root = Path(root).resolve()
    part = Path(relative)
    target = (root / part).resolve()
    if part.is_absolute() or ".." in part.parts or not target.is_relative_to(root):
        raise ValueError("Unsafe run-relative path")
    return target


def _manifest(root, name):
    path = bounded_path(root, "manifests/" + name)
    expected = PACKAGE / "manifests" / name
    if file_hash(path) != file_hash(expected):
        raise PermissionError("Registered manifest bytes changed: " + name)
    return path


def _by_qid(rows):
    result = {r["qid"]: r for r in rows}
    if len(rows) != len(result):
        raise ValueError("Duplicate question identifier")
    return result


def load_inputs(root):
    return _by_qid(read_jsonl(_manifest(root, "MODEL_INPUTS.jsonl")))


def load_tasks(root):
    return _by_qid(read_jsonl(_manifest(root, "TASKS_GOLD_AUDIT_ONLY.jsonl")))


def model_input(row, protocol="evidence_answer"):
    if protocol not in {"evidence_answer", "answer_only"}:
        raise ValueError("Unregistered model-visible protocol")
    # Deliberately reconstruct an allowlist. No verifier field or qid reaches the prompt.
    image = row["image_file"]
    p = Path(image)
    if p.is_absolute() or ".." in p.parts or p.parts[0] != "images":
        raise ValueError("Image must be a registered relative route")
    text = row["text" if protocol == "evidence_answer" else "plain_text"]
    if not isinstance(text, str) or not text or len(text) > 65536:
        raise ValueError("Model text must be a bounded nonempty string")
    return {"image_file": image, "text": text}


def load_schedule(root, seed):
    if seed not in SEEDS:
        raise ValueError("Unregistered training seed")
    rows = read_jsonl(_manifest(root, f"schedules/train_seed_{seed}.jsonl"))
    if len(rows) != 1536 or len({r["qid"] for r in rows}) != 1536:
        raise ValueError("Each training question must occur exactly once")
    for step in range(1, 97):
        batch = [r for r in rows if r["step"] == step]
        if sorted(r["slot"] for r in batch) != list(range(16)):
            raise ValueError("Incomplete update slot sequence")
        if set(Counter((r["family"], r["chart"]) for r in batch).values()) != {2}:
            raise ValueError("Training family/chart quota changed")
        if any(len(r["rollout_seeds"]) != 8 or r["paired_seed"] != seed for r in batch):
            raise ValueError("Paired sampling identity changed")
    return rows
