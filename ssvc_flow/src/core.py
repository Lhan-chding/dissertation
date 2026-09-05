"""Durable, hash-checked run records and explicit research phase boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = frozenset({"model_hash", "data_hash", "config_hash"})


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomic replacement; invalid numeric values cannot enter evidence records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def phase_artifacts(out, phase, status, details, artifacts=()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = [
        {
            "path": os.path.relpath(Path(path), out),
            "sha256": file_hash(path),
            "bytes": Path(path).stat().st_size,
        }
        for path in artifacts
    ]
    document = {
        "phase": phase,
        "status": status,
        "details": details,
        "source_commit": source_commit(),
        "source_files": {
            str(p.relative_to(PROJECT_ROOT)): file_hash(p)
            for p in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "status.json", document)
    write_json(out / "manifest.json", {"phase": phase, "files": files})
    (out / "report.md").write_text(
        f"# {phase}\n\nStatus: `{status}`\n\n"
        + "```json\n"
        + json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n```\n",
        encoding="utf-8",
    )
    return document


def load_config(path=None):
    path = Path(path) if path else PROJECT_ROOT / "configs/locked.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    return config


def validate_config(config, phase="smoke", model_key="qwen35_9b"):
    model = config["models"][model_key]
    if model_key == "qwen35_9b" and model["id"] != "Qwen/Qwen3.5-9B":
        raise ValueError("primary model must be Qwen/Qwen3.5-9B")
    revision = model.get("revision") or ""
    if phase != "smoke" and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("an immutable resolved model revision is required outside smoke")
    expected = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "num_beams": 1,
    }
    if any(config["generation"].get(key) != value for key, value in expected.items()):
        raise ValueError("main sampling distribution must be the untransformed softmax")
    if config["generation"]["max_new_tokens"] not in (48, 64, 128):
        raise ValueError("unsupported completion length")
    train = config["training"]
    locked_values = {
        "epsilon": 1e-4,
        "weight_decay": 0.0,
        "beta_kl": 0.0,
        "alpha": 2.0,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "base_dtype": "bfloat16",
        "adapter_dtype": "float32",
        "microbatch": 1,
        "betas": [0.9, 0.999],
        "adam_eps": 1e-8,
        "clip_epsilon": 0.2,
        "grad_clip": 1.0,
        "lambda": 1.0,
    }
    if any(train.get(key) != value for key, value in locked_values.items()):
        raise ValueError("declared training values differ from the locked P1 implementation")
    if train["B"] != 4 or train["K"] != 8 or train["Lnorm"] != 64:
        raise ValueError("locked main protocol requires B=4, K=8, Lnorm=64")
    if train["smoke_updates"] not in (2, 3, 4):
        raise ValueError("P1 requires 2-4 real optimizer updates")


def sample_identity(run_id, prompt_id, optimizer_step, sample_seed, rollout_index):
    return canonical_hash(
        {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "optimizer_step": optimizer_step,
            "sample_seed": sample_seed,
            "rollout_index": rollout_index,
        }
    )


class RunStore:
    """Single-writer durable JSONL ledger. Corrupt/truncated tails fail visibly."""

    def __init__(self, out, identity, resume=False):
        if not identity.keys() >= IDENTITY_FIELDS:
            raise ValueError("identity needs model_hash, data_hash and config_hash")
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        manifest = self.out / "identity.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("run exists; use --resume with identical hashes")
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("resume identity mismatch")
        elif resume:
            raise FileNotFoundError("cannot resume a run without identity.json")
        else:
            if any(self.out.glob("*.jsonl")) or any(self.out.glob("*.pt")):
                raise ValueError("orphan records/checkpoints without identity; choose a fresh run")
            write_json(manifest, identity)
        self.records = self._read_records()

    @property
    def keys(self):
        return set(self.records)

    def _read_records(self):
        path = self.out / "samples.jsonl"
        records = {}
        if not path.exists():
            return records
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    if not line.endswith("\n"):
                        raise ValueError("incomplete JSONL record")
                    row = json.loads(line)
                    key = row["sample_key"]
                    if key in records:
                        raise ValueError("duplicate sample key in ledger")
                    records = {**records, key: row}
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError("incomplete or malformed sample ledger") from exc
        return records

    def append(self, row):
        key = row.get("sample_key")
        if not isinstance(key, str) or not key:
            raise ValueError("sample_key must be a nonempty string")
        if key in self.records:
            if self.records[key] != row:
                raise ValueError("conflict for existing sample key")
            return False
        with (self.out / "samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records = {**self.records, key: dict(row)}
        return True


def load_split(data_root, split, allow_confirm=False, purpose=None):
    if split not in {"calibration", "train", "control", "dev", "confirm", "ood", "natural_pool"}:
        raise ValueError("unknown split")
    if split == "confirm" and (not allow_confirm or not purpose):
        raise PermissionError("confirm access requires explicit authorization and purpose")
    root = Path(data_root)
    with (root / f"{split}.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    with (root / "access.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "split": split,
                    "purpose": purpose or "local implementation audit",
                    "row_count": len(rows),
                    "source_commit": source_commit(),
                    "accessed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )
    return rows
