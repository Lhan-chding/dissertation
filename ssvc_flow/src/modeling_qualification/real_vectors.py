"""Bounded CPU extraction of realized S1 parameter-update geometry.

Only existing completed bank checkpoints are read. Parameters are retained for
one bank at a time; delta tensors are processed one named tensor at a time.
No model is constructed. The output contains hashes/layout/Gram matrices,
never raw parameters, optimizer tensors, raw text or private absolute paths.
Production execution requires a CPU Slurm allocation; preflight reads only
small JSON, file stats and ZIP central directories.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..core import canonical_hash
from ..optimizer_fork import state_hash
from .real_audit import _hash, _read, _safe

BANKS = ("bank00", "bank03", "bank04", "bank05", "bank11")
CANDIDATES = ("joint_0", "joint_1", "no_x_off_1")
MAX_CHECKPOINT_BYTES = 200 * 1024**2
MAX_OUTPUT_BYTES = 20 * 1024**2
MAX_RSS_BYTES = 2 * 1024**3
MAX_SECONDS = 600


def require_cpu_allocation():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Production checkpoint extraction requires a Slurm CPU allocation")
    for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE"):
        allowed = (
            {"", "NoDevFiles", "-1", "0"}
            if key == "SLURM_GPUS_ON_NODE"
            else {"", "NoDevFiles", "-1"}
        )
        if os.environ.get(key, "") not in allowed:
            raise RuntimeError("GPU visibility/allocation is forbidden for checkpoint extraction")


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _budget(started):
    if time.monotonic() - started > MAX_SECONDS or _rss_bytes() > MAX_RSS_BYTES:
        raise RuntimeError("RESOURCE_REVIEW_REQUIRED: CPU time or peak RSS budget exceeded")


def _layout(parameters):
    import torch

    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("nonempty ordered parameter mapping required")
    layout, offset = [], 0
    for name, tensor in parameters.items():
        if (
            not isinstance(name, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("finite named CPU floating parameters required")
        layout.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "offset": offset,
                "numel": tensor.numel(),
            }
        )
        offset += tensor.numel()
    return layout


def parameter_geometry(
    origin: dict, candidates: list[dict], candidate_ids: list[str], *, budget_check=lambda: None
) -> dict:
    """Stream true FP64 displacements and stable tall-skinny QR, without full D.

    The canonical vector concatenates checkpoint parameter insertion order,
    C-flattened tensors, with candidate minus origin in CPU FP64. Its SHA256
    has a layout-hash prefix, then little-endian FP64 delta bytes. QR supplies
    singular values: small Gram eigenvalues would square the conditioning.
    """
    import torch

    if (
        not candidates
        or len(candidates) != len(candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("one distinct candidate ID is required per parameter mapping")
    layout = _layout(origin)
    for parameters in candidates:
        if _layout(parameters) != layout:
            raise ValueError("candidate parameter order/name/shape/dtype layout mismatch")
    layout_hash = canonical_hash(layout)
    digests = [
        hashlib.sha256(("delta-fp64-le-v1:" + layout_hash + ":").encode()) for _ in candidates
    ]
    m = len(candidates)
    contributions = [[[] for _ in candidates] for _ in candidates]
    maxima = [0.0] * m
    reduced = np.empty((0, m), dtype=np.float64)
    for item in layout:
        budget_check()
        name = item["name"]
        baseline = origin[name].detach().to(dtype=torch.float64).numpy().reshape(-1)
        blocks = []
        for index, parameters in enumerate(candidates):
            delta = parameters[name].detach().to(dtype=torch.float64).numpy().reshape(-1) - baseline
            if not np.isfinite(delta).all():
                raise ValueError("nonfinite actual displacement")
            digests[index].update(delta.astype("<f8", copy=False).tobytes(order="C"))
            maxima[index] = max(maxima[index], float(np.max(np.abs(delta), initial=0.0)))
            blocks.append(delta)
        for i in range(m):
            for j in range(i, m):
                contributions[i][j].append(float(np.dot(blocks[i], blocks[j])))
        block = np.column_stack(blocks)
        if len(block):
            reduced = np.linalg.qr(np.vstack((reduced, block)), mode="r")
    gram = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            gram[i, j] = gram[j, i] = math.fsum(contributions[i][j])
    singular = np.linalg.svd(reduced, compute_uv=False)
    threshold = max(1e-14, (float(singular[0]) if len(singular) else 0.0) * 1e-10)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    cosines, angles = [], []
    for i in range(m):
        cosine_row, angle_row = [], []
        for j in range(m):
            cosine = (
                float(np.clip(gram[i, j] / (norms[i] * norms[j]), -1.0, 1.0))
                if norms[i] and norms[j]
                else None
            )
            cosine_row.append(cosine)
            angle_row.append(math.degrees(math.acos(cosine)) if cosine is not None else None)
        cosines.append(cosine_row)
        angles.append(angle_row)
    return {
        "candidate_ids": candidate_ids,
        "parameter_layout": layout,
        "parameter_layout_hash": layout_hash,
        "parameter_order": "CHECKPOINT_PARAMETER_MAPPING_INSERTION_ORDER",
        "parameter_count": sum(item["numel"] for item in layout),
        "delta_hash_definition": (
            "SHA256(UTF8(delta-fp64-le-v1: + layout_sha256 + :) || concat"
            "enated little-endian FP64 candidate-minus-origin bytes)"
        ),
        "delta_hashes": [digest.hexdigest() for digest in digests],
        "update_norms": norms.tolist(),
        "max_abs_update": maxima,
        "gram": gram.tolist(),
        "cosines": cosines,
        "angles_degrees": angles,
        "cosine_null_reason": "ONE_OR_BOTH_VECTORS_HAVE_ZERO_NORM",
        "singular_values_from_streaming_qr": singular.tolist(),
        "numerical_update_rank": int(np.sum(singular > threshold)),
        "rank_rtol": 1e-10,
        "rank_atol": 1e-14,
        "rank_threshold": threshold,
        "gram_qr_max_abs_difference": float(
            np.max(np.abs(gram - reduced.T @ reduced), initial=0.0)
        ),
        "semantic_response_rank": None,
        "semantic_response_rank_reason": (
            "PARAMETER_UPDATE_SPAN_IS_NOT_SEMANTIC_JACOBIAN_RESPONSE_RANK"
        ),
        "accumulation_dtype": "CPU_FP64",
        "full_update_vectors_written": False,
    }


def _stat_checkpoint(path, evidence_id):
    path = _safe(path)
    if not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint absent or exceeds 200 MiB bounded input size: " + evidence_id)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if any(info.flag_bits & 1 for info in entries):
            raise ValueError("encrypted checkpoint archive is not supported")
        total = sum(info.file_size for info in entries)
        if total > MAX_CHECKPOINT_BYTES:
            raise ValueError("uncompressed checkpoint exceeds bounded input size")
    return {
        "evidence_id": evidence_id,
        "bytes": path.stat().st_size,
        "zip_member_count": len(entries),
        "zip_uncompressed_bytes": total,
        "storage_format": "TORCH_SAVE_ZIP; restricted weights_only CPU loader required",
    }


def prepare_vector_extraction(plan_path: Path):
    """Read only JSON/stat/ZIP directory; return public plan plus private paths."""
    plan_path = _safe(plan_path)
    plan = _read(plan_path)
    root = _safe(Path(plan["paths"]["new_run_root"]))
    warm = plan["r4_binding"]["warm_checkpoint"]
    origin = _safe(Path(warm["path"]))
    origin_info = {
        **_stat_checkpoint(origin, "PARENT_R4/WARM_X_BASE_STEP64/checkpoint.pt"),
        "recorded_file_sha256": warm["file_sha256"],
        "expected_legacy_state_hash": warm["state_hash"],
        "hash_verified_during_preflight": False,
    }
    banks, public, missing = [], [], []
    for name in BANKS:
        bank = root / "S1" / name
        identity, completed, status, manifest = [
            _read(bank / file)
            for file in ("identity.json", "completed.json", "status.json", "manifest.json")
        ]
        if (
            not identity
            or not completed
            or not status
            or not manifest
            or completed.get("status") != "MEASURED"
            or status.get("status") != "MEASURED"
        ):
            raise ValueError("selected S1 bank is not completed: " + name)
        if completed.get("identity_hash") != canonical_hash(identity) or completed.get(
            "manifest_sha256"
        ) != _hash(bank / "manifest.json"):
            raise ValueError("S1 completion identity/manifest binding mismatch: " + name)
        if identity.get("validated_plan_hash") != canonical_hash(plan) or identity.get(
            "source_hash"
        ) != canonical_hash(plan["source_files"]):
            raise ValueError("S1 identity differs from frozen plan/source binding: " + name)
        entries = {entry["path"]: entry for entry in manifest["files"]}
        selected, public_selected = [], []
        for cid in CANDIDATES:
            record_path = bank / "candidates" / cid / "candidate.json"
            if not record_path.is_file():
                missing.append(
                    {
                        "bank_id": name,
                        "candidate_id": cid,
                        "reason": "NOT_A_CANDIDATE_IN_COMPLETED_BANK",
                    }
                )
                continue
            record = _read(record_path)
            relative_record = f"candidates/{cid}/candidate.json"
            relative_checkpoint = f"candidates/{cid}/checkpoint.pt"
            checkpoint = _safe(bank / relative_checkpoint)
            if (
                _hash(record_path) != entries[relative_record]["sha256"]
                or record["checkpoint_sha256"] != entries[relative_checkpoint]["sha256"]
            ):
                raise ValueError("candidate/checkpoint metadata differs from completed manifest")
            expected_identity = {**identity, "candidate_spec": record["candidate_spec"]}
            if record["candidate_id"] != cid or record["checkpoint_identity"] != expected_identity:
                raise ValueError("candidate checkpoint identity mismatch")
            info = {
                **_stat_checkpoint(checkpoint, f"S1/{name}/{relative_checkpoint}"),
                "candidate_id": cid,
                "recorded_file_sha256": record["checkpoint_sha256"],
                "expected_state_hash": record["candidate_state_hash"],
                "hash_verified_during_preflight": False,
                "candidate_record_sha256": _hash(record_path),
            }
            if info["bytes"] != entries[relative_checkpoint]["bytes"]:
                raise ValueError("candidate checkpoint stat differs from manifest byte count")
            selected.append((cid, checkpoint, record, info))
            public_selected.append(info)
        aliases = _read(bank / "policy_aliases.json", {})
        banks.append((name, identity, selected, aliases))
        public.append(
            {
                "bank_id": name,
                "identity_sha256": _hash(bank / "identity.json"),
                "manifest_sha256": _hash(bank / "manifest.json"),
                "completed_sha256": _hash(bank / "completed.json"),
                "origin_followup_state_hash_recorded": identity["origin_state_hash"],
                "selected_checkpoints": public_selected,
                "policy_aliases": {key: value["alias_of"] for key, value in aliases.items()},
            }
        )
    input_bytes = origin_info["bytes"] + sum(
        info["bytes"] for bank in public for info in bank["selected_checkpoints"]
    )
    public_plan = {
        "schema_version": "m5-real-vectors-preflight-v1",
        "source_plan_sha256": _hash(plan_path),
        "origin": origin_info,
        "banks": public,
        "missing_selected_candidates": missing,
        "checkpoint_file_count": 1 + sum(len(bank["selected_checkpoints"]) for bank in public),
        "checkpoint_input_bytes": input_bytes,
        "estimated_file_read_bytes": input_bytes * 2,
        "estimated_wall_seconds": 300,
        "estimate_basis": (
            "15 bounded checkpoints, one byte hash and one restricted CPU"
            " load each; streamed tensor geometry; conservative 50 MB/s r"
            "ead estimate plus CPU hashes"
        ),
        "estimated_peak_rss_gib": 1.5,
        "memory_estimate_basis": (
            "origin parameters plus at most three candidate parameter map"
            "s; only one full checkpoint payload loaded at a time; optimi"
            "zer tensors released before next candidate"
        ),
        "limits": {
            "wall_seconds": MAX_SECONDS,
            "rss_bytes": MAX_RSS_BYTES,
            "output_bytes": MAX_OUTPUT_BYTES,
        },
        "new_model_calls": 0,
        "full_update_vectors_exported": False,
        "reconstruction": (
            "Retained source checkpoint minus the bound R4 warm checkpoin"
            "t in exported parameter-layout order"
        ),
        "scope": (
            "Five completed S1 banks; joint_0, joint_1 and no_x_off_1 whe"
            "re actually present; alias checkpoints remain distinct state"
            "s, no samples read or duplicated"
        ),
    }
    return public_plan, warm, origin, banks


def _load_verified(path, expected_identity, expected_file_hash, expected_state_hash, started):
    import torch

    _budget(started)
    if _hash(path) != expected_file_hash:
        raise ValueError("checkpoint bytes differ from frozen manifest before deserialization")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or set(payload) != {"identity", "state", "state_hash"}
        or payload["identity"] != expected_identity
    ):
        raise ValueError("checkpoint restricted payload/identity mismatch")
    if (
        payload["state_hash"] != expected_state_hash
        or state_hash(payload["state"]) != expected_state_hash
    ):
        raise ValueError("checkpoint tensor/state checksum mismatch")
    _layout(payload["state"]["parameters"])
    _budget(started)
    return payload["state"]


def extract_real_vectors(plan_path: Path, out: Path, *, require_slurm=True):
    if require_slurm:
        require_cpu_allocation()
    out = _safe(out)
    if out.exists():
        raise FileExistsError("real-vector output already exists")
    started = time.monotonic()
    public, warm, origin_path, banks = prepare_vector_extraction(plan_path)
    out.mkdir(parents=True, exist_ok=False)
    try:
        import torch

        torch.set_num_threads(1)
        origin = _load_verified(
            origin_path, warm["identity"], warm["file_sha256"], warm["state_hash"], started
        )
        if (
            origin["metadata"].get("arm") != "X_BASE"
            or origin["metadata"].get("checkpoint_step") != 64
        ):
            raise ValueError("warm origin is not the bound X_BASE step64 state")
        origin_optimizer_hash = state_hash(origin["optimizer"])
        origin_parameters = origin["parameters"]
        origin_parameter_hash = state_hash(origin_parameters)
        origin_model_parameter_hash = state_hash(
            {name: state_hash(value) for name, value in origin_parameters.items()}
        )
        origin_layout = _layout(origin_parameters)
        del origin
        gc.collect()
        results = []
        for name, identity, selected, aliases in banks:
            parameters, candidate_ids, verified = [], [], []
            for cid, path, record, source_info in selected:
                state = _load_verified(
                    path,
                    record["checkpoint_identity"],
                    record["checkpoint_sha256"],
                    record["candidate_state_hash"],
                    started,
                )
                audit = record["audit"]
                if (
                    audit["origin_state_hash"] != identity["origin_state_hash"]
                    or audit["optimizer_hash_before"] != origin_optimizer_hash
                    or audit["parameter_hash_before"] != origin_model_parameter_hash
                ):
                    raise ValueError(
                        "candidate origin parameters/optimizer are not the bound lega"
                        "cy warm tensors"
                    )
                if (
                    state_hash(state["parameters"]) != record["candidate_parameter_hash"]
                    or state_hash(state["optimizer"]) != record["candidate_optimizer_state_hash"]
                ):
                    raise ValueError("candidate tensor hashes differ from candidate record")
                if _layout(state["parameters"]) != origin_layout:
                    raise ValueError("actual candidate parameter layout differs from origin")
                optimizer_names = state.get("optimizer_parameter_names")
                if not optimizer_names or [
                    name for group in optimizer_names for name in group
                ] != list(origin_parameters):
                    raise ValueError(
                        "candidate optimizer parameter order differs from origin vector layout"
                    )
                if {int(float(item["step"])) for item in state["optimizer"]["state"].values()} != {
                    65
                }:
                    raise ValueError("candidate optimizer step is not 65")
                verified.append(
                    {
                        **source_info,
                        "byte_sha256_verified": True,
                        "full_state_hash_recomputed": True,
                        "parameter_hash_recomputed": record["candidate_parameter_hash"],
                        "optimizer_hash_recomputed": record["candidate_optimizer_state_hash"],
                        "optimizer_parameter_order_hash": canonical_hash(optimizer_names),
                        "recorded_update_norm": audit.get("actual_step_norm"),
                        "alias_of": aliases.get(cid, {}).get("alias_of"),
                    }
                )
                candidate_ids.append(cid)
                parameters.append(state["parameters"])
                del state
                gc.collect()
                _budget(started)
            geometry = parameter_geometry(
                origin_parameters, parameters, candidate_ids, budget_check=lambda: _budget(started)
            )
            for item, norm in zip(verified, geometry["update_norms"], strict=True):
                expected = item["recorded_update_norm"]
                item["update_norm_absolute_difference_from_recorded"] = (
                    abs(norm - expected) if expected is not None else None
                )
                item["update_norm_matches_recorded_rtol1e6_atol1e10"] = (
                    math.isclose(norm, expected, rel_tol=1e-6, abs_tol=1e-10)
                    if expected is not None
                    else None
                )
            base_index = candidate_ids.index("joint_0")
            aux_ids = [cid for cid in candidate_ids if cid != "joint_0"]
            auxiliary = (
                parameter_geometry(
                    parameters[base_index],
                    [
                        params
                        for cid, params in zip(candidate_ids, parameters, strict=True)
                        if cid != "joint_0"
                    ],
                    aux_ids,
                    budget_check=lambda: _budget(started),
                )
                if aux_ids
                else None
            )
            results.append(
                {
                    "bank_id": name,
                    "checkpoint_verification": verified,
                    "actual_origin_relative_updates": geometry,
                    "candidate_minus_joint_0_geometry": auxiliary,
                    "origin_followup_state_hash_recorded": identity["origin_state_hash"],
                    "origin_followup_full_state_hash_recomputed": False,
                    "origin_binding_note": (
                        "Legacy R4 parameters and optimizer are fully rehashed and eq"
                        "ual the candidate audit before-hashes; the followup recaptur"
                        "e adds state fields so its distinct complete hash is metadat"
                        "a only"
                    ),
                    "new_samples": 0,
                    "semantic_J_computed": False,
                    "cross_checkpoint_fit": False,
                }
            )
            del parameters
            gc.collect()
        result = {
            "schema_version": "m5-real-vectors-v1",
            "status": "ACTUAL_UPDATE_GEOMETRY_CPU_VERIFIED",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "preflight": public,
            "origin_verification": {
                "byte_sha256": warm["file_sha256"],
                "legacy_state_hash_recomputed": warm["state_hash"],
                "parameter_hash_recomputed": origin_parameter_hash,
                "model_parameter_hash_recomputed": origin_model_parameter_hash,
                "optimizer_hash_recomputed": origin_optimizer_hash,
                "checkpoint_step": 64,
                "arm": "X_BASE",
            },
            "banks": results,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": _rss_bytes(),
            "new_qwen_calls": 0,
            "gpu_operations": 0,
            "model_constructed": False,
            "full_update_vectors_written": False,
            "semantic_response_rank": None,
            "semantic_response_rank_reason": (
                "Update-space rank at a single anchor does not identify Jacob"
                "ian rank or out-of-anchor generalization"
            ),
        }
        encoded = (
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        if len(encoded.encode()) > MAX_OUTPUT_BYTES:
            raise RuntimeError("RESOURCE_REVIEW_REQUIRED: output exceeds 20 MiB")
        _budget(started)
        (out / "real_update_geometry.json").write_text(encoded, encoding="utf-8")
        return result
    except BaseException as error:
        # Keep failures reviewable without private paths from a library traceback.
        (out / "failure.json").write_text(
            json.dumps(
                {
                    "status": "FAILED_OR_RESOURCE_REVIEW_REQUIRED",
                    "exception_type": type(error).__name__,
                    "reason": str(error)
                    if "/" not in str(error)
                    else "Exception included a private path; details retained in allocation log",
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_rss_bytes": _rss_bytes(),
                    "new_qwen_calls": 0,
                    "gpu_operations": 0,
                },
                indent=2,
            )
            + "\n"
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        public, _, _, _ = prepare_vector_extraction(args.plan)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "vector_preflight.json").write_text(
            json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_ONLY",
                    "checkpoint_file_count": public["checkpoint_file_count"],
                    "checkpoint_input_bytes": public["checkpoint_input_bytes"],
                }
            )
        )
        return

    def timeout(signum, frame):
        raise TimeoutError("RESOURCE_REVIEW_REQUIRED: 600-second budget expired")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(MAX_SECONDS)
    try:
        result = extract_real_vectors(args.plan, args.out)
        print(
            json.dumps(
                {key: result[key] for key in ("status", "elapsed_seconds", "peak_rss_bytes")}
            )
        )
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
