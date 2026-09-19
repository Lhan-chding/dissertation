"""Two-level immutable complete-action score cache; never a training-state cache."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

from ..modeling_v3.vlm_observation import digest


def score_key(
    inference_fingerprint,
    input_identity,
    tokens,
    *,
    eos_ids,
    max_new_tokens=64,
    scoring_path="uncached_prefix_recompute",
):
    tokens = list(tokens)
    if not inference_fingerprint or not input_identity or not scoring_path:
        raise ValueError("Inference, input, and scoring identities are required")
    if (
        not tokens
        or len(tokens) > max_new_tokens
        or any(type(t) is not int or t < 0 for t in tokens)
        or any(t in eos_ids for t in tokens[:-1])
    ):
        raise ValueError("Expected one complete token sequence through first EOS")
    stop = "eos" if tokens[-1] in eos_ids else "length"
    if stop == "length" and len(tokens) != max_new_tokens:
        raise ValueError("Unterminated prefix is not a complete action")
    return digest(
        {
            "inference": inference_fingerprint,
            "input": input_identity,
            "tokens": tokens,
            "stop": stop,
            "horizon": max_new_tokens,
            "eos_ids": sorted(eos_ids),
            "path": scoring_path,
        }
    )


class ScoreCache:
    """SQLite transactions publish atomic entries; namespaces preserve old caches.

    Namespace includes runtime precision, implementation, template and generation
    settings. Keys include true forward fingerprints, excluding Adam/RNG state.
    Each returned list is a copy. Duplicate draws retain their statistical weight.
    """

    def __init__(self, path, *, namespace):
        if not isinstance(namespace, dict) or not namespace:
            raise ValueError("Explicit immutable scoring namespace required")
        self.namespace = digest(namespace)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS scores "
            "(namespace TEXT, key TEXT, payload TEXT, "
            "PRIMARY KEY(namespace,key))"
        )
        self.memory = {}
        self.counters = dict(
            logical_scores=0,
            physical_scoring_keys=0,
            memory_hits=0,
            persistent_hits=0,
            scoring_seconds=0.0,
        )

    @staticmethod
    def _validate(value, length):
        values = [float(v) for v in value]
        if len(values) != length or any(not math.isfinite(v) or v > 0 for v in values):
            raise ValueError("Finite nonpositive token log probabilities required")
        return tuple(values)

    def score(self, key, *, length, compute):
        self.counters["logical_scores"] += 1
        if key in self.memory:
            self.counters["memory_hits"] += 1
            return list(self._validate(self.memory[key], length))
        row = self.db.execute(
            "SELECT payload FROM scores WHERE namespace=? AND key=?", (self.namespace, key)
        ).fetchone()
        if row is not None:
            result = self._validate(json.loads(row[0]), length)
            self.counters["persistent_hits"] += 1
        else:
            started = time.perf_counter()
            result = self._validate(compute(), length)
            self.counters["scoring_seconds"] += time.perf_counter() - started
            self.counters["physical_scoring_keys"] += 1
            payload = json.dumps(result, allow_nan=False)
            with self.db:
                self.db.execute(
                    "INSERT OR IGNORE INTO scores VALUES (?,?,?)", (self.namespace, key, payload)
                )
                winner = self.db.execute(
                    "SELECT payload FROM scores WHERE namespace=? AND key=?", (self.namespace, key)
                ).fetchone()[0]
                if self._validate(json.loads(winner), length) != result:
                    raise ValueError("Conflicting scores for an immutable scoring key")
        self.memory[key] = result
        return list(result)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
