"""Bank-local C1 auxiliary observations and immutable repair overlays.

An inference alias may occur in both fit and evaluation banks. The auxiliary
total-response baseline must reuse counts only from its own bank packet. This
adapter preserves completed N1 artifacts and records any additional frozen CPU
sampling needed by an all-alias bank, whose contrast packet contained no draws.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import numpy as np

from .io import _atomic_bytes
from .study import digest, rng_seed

VERSION = "C1_SAME_BANK_COUNTS_V1"


def same_bank_counts(primitives, row):
    """Read only the primitive count keys named by this bank's packet."""
    result = {}
    for index, policy in enumerate(row["sample_names"]):
        key = f"{row['array_prefix']}sample_{index}_counts"
        if key in primitives:
            result[policy] = np.asarray(primitives[key]).copy()
    return result


def auxiliary_seed(trajectory_id, anchor_index, n, noise, bank, policy):
    role = "fit" if bank < 8 else "diagnostic" if bank < 10 else "evaluation"
    return rng_seed(
        2026091502,
        trajectory_id,
        anchor_index,
        n,
        noise,
        bank,
        role,
        policy,
        "C1_same_bank_level_v1",
    )


def _attach(packet, out, receipt):
    path = out / "C1_auxiliary.npz"
    if (
        receipt.get("source_packet_sha256") != packet["artifact_sha256"]
        or receipt.get("version") != VERSION
        or digest(path) != receipt.get("auxiliary_sha256")
    ):
        raise ValueError("C1 auxiliary overlay identity mismatch")
    with np.load(path, allow_pickle=False) as values:
        levels = values["legacy_total_levels"].copy()
        origin = values["legacy_origin_counts"].copy()
    if levels.shape != packet["legacy_total_levels"].shape or not np.isfinite(levels).all():
        raise ValueError("C1 auxiliary level shape or values changed")
    np.testing.assert_array_equal(origin, packet["legacy_origin_counts"])
    metadata = copy.deepcopy(packet["metadata"])
    metadata["c1_auxiliary_fit_evaluation_independent"] = True
    metadata["c1_auxiliary_overlay"] = receipt
    return {
        **packet,
        "metadata": metadata,
        "legacy_total_levels": levels,
        "legacy_origin_counts": origin,
        "c1_auxiliary_sha256": receipt["auxiliary_sha256"],
        "c1_auxiliary_receipt_sha256": digest(out / "receipt.json"),
        "c1_auxiliary_path": str(path.resolve()),
        "c1_auxiliary_receipt_path": str((out / "receipt.json").resolve()),
    }


def repair_c1_auxiliary(parent_root, entry, ai, packet, out):
    """Return a corrected copy, binding a new overlay to the original paid file.

    Historical replicas retain the exact original levels. New replicas recover
    same-bank counts from paid primitives and independently sample missing
    policies. Origin counts were already independent and remain byte-identical.
    Existing overlays are verified and reused without further measurement.
    """
    metadata = packet["metadata"]
    if (
        metadata["method"] != "O_IND"
        or metadata.get("c1_auxiliary_fit_evaluation_independent") is True
    ):
        return packet
    out = Path(out)
    if (out / "receipt.json").is_file():
        receipt = json.loads((out / "receipt.json").read_text())
        if (receipt.get("trajectory_id"), receipt.get("anchor_index"), receipt.get("n")) != (
            entry["id"],
            ai,
            metadata["n"],
        ):
            raise ValueError("C1 auxiliary requested experiment identity mismatch")
        return _attach(packet, out, receipt)
    from .observation_study import service_arguments
    from .observations import ToyWorldService
    from .packet_codec import decode_arrays

    original_path = Path(packet["path"]) / "packet_arrays.npz"
    if digest(original_path) != packet["artifact_sha256"]:
        raise ValueError("C1 source packet changed")
    with np.load(original_path, allow_pickle=False) as values:
        primitives = decode_arrays(values)
    levels = packet["legacy_total_levels"].copy()
    origins = packet["legacy_origin_counts"].copy()
    n = metadata["n"]
    theta, features, categories, prompt_ids, _ = service_arguments(parent_root, entry, ai)
    repairs = []
    total_cost = {}
    rows = {(row["noise_replica"], row["bank"]): row for row in metadata["packets"]}
    legacy_replicas = {
        row["noise_replica"]
        for row in metadata.get("C1_supplementary_costs", [])
        if row.get("legacy_reused") is True
    }
    for noise in range(metadata["replicas"]):
        if noise in legacy_replicas:
            continue
        service = ToyWorldService.from_parameters(
            theta, features, categories, prompt_ids=prompt_ids
        )
        for bank in metadata["banks"]:
            row = rows[(noise, bank)]
            available = same_bank_counts(primitives, row)
            before = service.ledger.snapshot()
            generated, reused = [], []
            for op in range(3):
                policy = f"b{bank}o{op}"
                fp = service.fingerprint(policy)
                if fp not in available:
                    seed = auxiliary_seed(entry["id"], ai, n, noise, bank, fp)
                    _, labels, _ = service.sample(policy, n, np.random.default_rng(seed))
                    available[fp] = np.stack([np.bincount(v, minlength=4) for v in labels])
                    generated.append({"fingerprint": fp, "seed": seed})
                else:
                    reused.append(fp)
                counts = available[fp]
                if counts.shape != (72, 4) or not (counts.sum(-1) == n).all():
                    raise ValueError("C1 same-bank count shape or total differs")
                levels[noise, bank, op] = counts / n
            cost = service.ledger.delta(before)
            repairs.append(
                {
                    "noise_replica": noise,
                    "bank": bank,
                    "role": row["purpose"],
                    "source_packet_id": row["packet_id"],
                    "new_samples": generated,
                    "same_bank_reused_policies": sorted(set(reused)),
                    "additional_cost": cost,
                }
            )
        for key, value in service.ledger.snapshot().items():
            total_cost[key] = total_cost.get(key, 0) + value
    out.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, legacy_total_levels=levels, legacy_origin_counts=origins)
    _atomic_bytes(out / "C1_auxiliary.npz", buffer.getvalue())
    receipt = {
        "version": VERSION,
        "status": "REPAIRED_WITHOUT_MUTATING_N1",
        "trajectory_id": entry["id"],
        "anchor_index": ai,
        "n": n,
        "source_packet_path": str(original_path.resolve()),
        "source_packet_sha256": packet["artifact_sha256"],
        "auxiliary_sha256": digest(out / "C1_auxiliary.npz"),
        "historical_replicas_preserved": sorted(legacy_replicas),
        "origin_counts_preserved": True,
        "fit_evaluation_independent": True,
        "additional_cost": total_cost,
        "repairs": repairs,
        "new_optimizer_updates": 0,
        "new_gpu_calls": 0,
        "new_qwen_calls": 0,
    }
    _atomic_bytes(
        out / "receipt.json", (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    )
    return _attach(packet, out, receipt)
