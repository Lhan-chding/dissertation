"""Read-only, bounded import of existing S1 and historical R3/R4 evidence.

This module never imports a model, loads a checkpoint, or starts a remote task.
Metadata hashes/norms are not parameter vectors. A missing local file is never
reported as a missing server original. Output contains relative evidence IDs,
hashes, counts and status only; raw generations and private paths stay at source.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from ..core import canonical_hash
from ..verifiers import classify

REQUIRED_FIELDS = [
    "run_id",
    "source_commit",
    "origin_state_hash",
    "candidate_state_hash",
    "origin_optimizer_hash",
    "candidate_optimizer_hash",
    "parameter_layout_hash",
    "inference_fingerprint",
    "seed",
    "checkpoint_step",
    "bank_id",
    "candidate_id",
    "intervention",
    "base_scene_id",
    "prompt_id",
    "family",
    "interface",
    "operation",
    "nX",
    "nS",
    "nW",
    "nI",
    "n",
    "sample_ledger_hash",
    "update_vector_path",
    "update_vector_hash",
    "update_norm",
    "source_files",
    "evidence_level",
    "alias_of",
]
EVENTS = ("X", "S", "W", "I")


def _safe(path: Path) -> Path:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink evidence paths are not accepted")
    if any("sealed" in part.lower() or "confirm" in part.lower() for part in path.parts):
        raise ValueError("sealed/confirm evidence is outside M5 authorization")
    return path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with _safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, default=None):
    path = _safe(path)
    if not path.is_file():
        return default

    def bad(value):
        raise ValueError("nonfinite JSON constant")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=bad)

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        if isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def _ref(path: Path, root: Path) -> dict:
    return {
        "evidence_id": path.relative_to(root).as_posix(),
        "sha256": _hash(path),
        "bytes": path.stat().st_size,
    }


def _row(values: dict) -> dict:
    result = {field: values.get(field) for field in REQUIRED_FIELDS}
    result.update({key: value for key, value in values.items() if key not in result})
    reasons = {
        key: "NOT_RECORDED_IN_ACCESSIBLE_ARTIFACT" for key in REQUIRED_FIELDS if result[key] is None
    }
    for key in ("update_vector_path", "update_vector_hash"):
        if result[key] is None:
            reasons[key] = (
                "NO_BOUND_UPDATE_VECTOR_IMPORTED; NORM_AND_HASH_DO_NOT_IDENTIFY_DIRECTION"
            )
    if result["parameter_layout_hash"] is None:
        reasons["parameter_layout_hash"] = "LOCAL_COMPACT_BUNDLE_HAS_NO_ORDERED_PARAMETER_LAYOUT"
    if result["alias_of"] is None:
        reasons["alias_of"] = "NOT_AN_ALIAS"
    if result["prompt_id"] is None:
        for key in (
            "prompt_id",
            "base_scene_id",
            "family",
            "interface",
            "operation",
            "nX",
            "nS",
            "nW",
            "nI",
            "n",
        ):
            if result[key] is None:
                reasons[key] = "NO_DIRECT_PER_PROMPT_MEASUREMENT_FOR_THIS_CANDIDATE"
    result["null_reasons"] = reasons
    return result


def _manifest(bank: Path) -> dict:
    manifest = _read(bank / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("completion manifest missing or invalid")
    present, missing, excluded = [], [], []
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise ValueError("unsafe or duplicate manifest entry")
        seen.add(relative.as_posix())
        path = _safe(bank / relative)
        if path.suffix in {".pt", ".safetensors", ".bin", ".pth"}:
            excluded.append(
                {
                    "evidence_id": relative.as_posix(),
                    "exists_locally": path.is_file(),
                    "sha256_recorded": entry["sha256"],
                }
            )
            continue
        if not path.is_file():
            missing.append(relative.as_posix())
        elif _hash(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("manifest byte/hash mismatch: " + relative.as_posix())
        else:
            present.append(relative.as_posix())
    return {
        "verified_available_files": len(present),
        "missing_local_nonbinary_files": missing,
        "checkpoint_files_not_loaded": excluded,
        "entries": seen,
    }


def _read_ledger(path: Path, identity: dict, candidate: dict, scenes: dict, parser_verified: bool):
    grouped, keys, reparsed = {}, set(), 0
    from ..r2_runtime import annotate_diagnostic

    for line in _safe(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("record_hash") != canonical_hash(
            {k: v for k, v in row.items() if k != "record_hash"}
        ):
            raise ValueError("sample record hash mismatch")
        if row.get("execution_checks", {}).get("passed") is not True or row["execution_checks"].get(
            "faults"
        ):
            raise ValueError("execution fault must not enter semantic I or counts")
        if row.get("category") not in EVENTS:
            raise ValueError("unknown semantic category")
        for name in (
            "run_id",
            "origin_state_hash",
            "bank_id",
            "model_hash",
            "data_hash",
            "parser_version_hash",
            "control_panel_hash",
            "source_hash",
        ):
            if name in identity and row.get(name) != identity[name]:
                raise ValueError("sample identity mismatch: " + name)
        if row.get("candidate_id") != candidate["candidate_id"] or row.get(
            "candidate_policy_fingerprint"
        ) != candidate.get("candidate_policy_fingerprint"):
            raise ValueError("sample candidate identity mismatch")
        sample_key = row.get("sample_key")
        if not isinstance(sample_key, str) or sample_key in keys:
            raise ValueError("missing/duplicate sample key")
        keys.add(sample_key)
        scene = scenes.get(row.get("base_scene_id"))
        if scene is not None and parser_verified and "raw_completion" in row:
            if canonical_hash(scene) != row.get("scene_hash"):
                raise ValueError("local scene hash differs from recorded scene identity")
            annotation = annotate_diagnostic(row["raw_completion"], scene)
            if any(row.get(key) != value for key, value in annotation.items()):
                raise ValueError("raw output semantic reparse mismatch")
            if (
                classify(row["raw_completion"], scene["truth_world"], scene["operation"])
                != row["category"]
            ):
                raise ValueError("independent X/S/W/I classifier mismatch")
            reparsed += 1
        group = grouped.setdefault(
            row["prompt_id"],
            {
                **{
                    key: row.get(key)
                    for key in ("prompt_id", "base_scene_id", "family", "interface", "operation")
                },
                "seed": row.get("train_seed"),
                "counts": Counter(),
                "sample_indices": set(),
                "n": 0,
            },
        )
        for key in ("base_scene_id", "family", "interface", "operation"):
            if group[key] != row.get(key):
                raise ValueError("per-prompt identity changed")
        if (
            group["seed"] != row.get("train_seed")
            or row.get("sample_index") in group["sample_indices"]
        ):
            raise ValueError("inconsistent seed or repeated sample index")
        group["sample_indices"].add(row.get("sample_index"))
        group["counts"][row["category"]] += 1
        group["n"] += 1
    return grouped, keys, reparsed


def _bank(bank: Path, root: Path, scenes: dict, parser_info: dict) -> tuple[list, dict, set]:
    identity = _read(bank / "identity.json", {})
    status = _read(bank / "status.json", {})
    completed = _read(bank / "completed.json")
    info = {
        "bank_id": bank.name,
        "source": _ref(bank / "identity.json", root),
        "recorded_status": status.get("status"),
        "usable": False,
    }
    if not completed or completed.get("status") != "MEASURED" or status.get("status") != "MEASURED":
        return [], {**info, "audit_state": "INCOMPLETE_LOCAL_EVIDENCE"}, set()
    if status.get("execution_kind") != "REAL_CUDA_FOLLOWUP":
        return [], {**info, "audit_state": "NOT_REAL_FOLLOWUP_EVIDENCE"}, set()
    if completed.get("identity_hash") != canonical_hash(identity) or completed.get(
        "manifest_sha256"
    ) != _hash(bank / "manifest.json"):
        raise ValueError("completion identity/manifest binding mismatch")
    manifest = _manifest(bank)
    manifest_entries = manifest.pop("entries")
    aliases = _read(bank / "policy_aliases.json", {})
    origin = _read(bank / "origin_binding.json", {})
    receipts = sorted((bank.parent.parent / "receipts").glob("*integrity*verification*.json"))
    receipt = _read(receipts[0], {}) if receipts else {}
    source_commit = (
        receipt.get("source_commit")
        if receipt.get("identity_hash") == canonical_hash(identity)
        and receipt.get("manifest_sha256") == _hash(bank / "manifest.json")
        else None
    )
    common_refs = [
        _ref(bank / name, root)
        for name in ("identity.json", "origin_binding.json", "manifest.json", "completed.json")
        if (bank / name).exists()
    ]
    if receipts:
        common_refs.append(_ref(receipts[0], root))
    rows, all_keys, panels, candidates, totals, reparsed_total = [], set(), {}, {}, {}, 0
    count_paths = sorted((bank / "candidates").glob("*/counts_by_prompt.json"))
    candidate_paths = sorted((bank / "candidates").glob("*/candidate.json"))
    for cp in candidate_paths:
        if cp.relative_to(bank).as_posix() not in manifest_entries:
            raise ValueError("candidate record is outside completion manifest")
        candidate = _read(cp)
        cid = cp.parent.name
        if candidate.get("candidate_id") != cid:
            raise ValueError("candidate path identity mismatch")
        candidates[cid] = candidate
    for count_path in count_paths:
        cid = count_path.parent.name
        counts = _read(count_path)
        if isinstance(counts, dict):
            continue
        candidate = candidates[cid]
        ledger = count_path.parent / "direct" / "samples.jsonl"
        grouped, keys, reparsed = {}, set(), 0
        level = "METADATA_COUNTS"
        if ledger.exists():
            if ledger.relative_to(bank).as_posix() not in manifest_entries:
                raise ValueError("sample ledger is outside completion manifest")
            grouped, keys, reparsed = _read_ledger(
                ledger,
                identity,
                candidate,
                scenes,
                parser_info.get("parser_version_hash") == identity.get("parser_version_hash")
                and bool(parser_info.get("verified")),
            )
            flat = count_path.parent / "samples.jsonl"
            if flat.exists() and _hash(flat) != _hash(ledger):
                raise ValueError("flat sample copy differs from direct ledger")
            if keys & all_keys:
                raise ValueError("independent candidate ledgers repeat sample keys")
            all_keys |= keys
            if set(grouped) != {item["prompt_id"] for item in counts}:
                raise ValueError("raw/summary prompt panel mismatch")
            level = (
                "RAW_SEMANTICS_RECOMPUTED" if reparsed == len(keys) else "LEDGER_COUNTS_RECOMPUTED"
            )
        panel = {}
        for item in counts:
            n = item["n"]
            if type(n) is not int or n <= 0 or set(item["counts"]) != set(EVENTS):
                raise ValueError("invalid event counts")
            if (
                any(
                    type(item["counts"][event]) is not int or item["counts"][event] < 0
                    for event in EVENTS
                )
                or sum(item["counts"].values()) != n
            ):
                raise ValueError("event counts are not a nonnegative partition")
            pid = item["prompt_id"]
            if pid in panel:
                raise ValueError("duplicate prompt summary")
            panel[pid] = (item["base_scene_id"], item["family"], item["interface"], n)
            if grouped:
                actual = grouped[pid]
                if actual["n"] != n or any(
                    actual["counts"][event] != item["counts"][event] for event in EVENTS
                ):
                    raise ValueError("raw ledger/count summary mismatch")
                if any(
                    actual[key] != item[key] for key in ("base_scene_id", "family", "interface")
                ):
                    raise ValueError("raw/summary prompt identity mismatch")
        panels[cid] = panel
        totals[cid] = {event: sum(item["counts"][event] for item in counts) for event in EVENTS}
        reparsed_total += reparsed
        candidate["_measurement"] = (counts, grouped, ledger if ledger.exists() else None, level)
    if panels and any(panel != next(iter(panels.values())) for panel in panels.values()):
        raise ValueError("candidate probe panels or sample budgets differ")
    for cid, candidate in candidates.items():
        audit = candidate.get("audit", {})
        alias = aliases.get(cid)
        measurement = candidate.get("_measurement")
        if alias:
            source = alias["alias_of"]
            if (
                source not in candidates
                or aliases.get(source)
                or alias.get("additional_samples") != 0
                or alias.get("matching_rule") != "EXACT_COMPLETE_FORWARD_FINGERPRINT"
            ):
                raise ValueError("unsupported or cyclic policy alias")
            if candidate.get("candidate_policy_fingerprint") != candidates[source].get(
                "candidate_policy_fingerprint"
            ) or alias.get("candidate_policy_fingerprint") != candidate.get(
                "candidate_policy_fingerprint"
            ):
                raise ValueError("alias fingerprint mismatch")
            alias_counts = _read(bank / "candidates" / cid / "counts_by_prompt.json", {})
            if (
                alias_counts.get("alias_of") != source
                or alias_counts.get("additional_samples") != 0
                or alias_counts.get("source_counts") != f"candidates/{source}/counts_by_prompt.json"
            ):
                raise ValueError("alias counts binding mismatch")
            measurement = candidates[source].get("_measurement")
            if source in totals:
                totals[cid] = dict(totals[source])
        norm = audit.get("actual_step_norm")
        if norm is not None and (
            not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm < 0
        ):
            raise ValueError("invalid recorded update norm")
        base = {
            "run_id": identity.get("run_id"),
            "source_commit": source_commit,
            "origin_state_hash": identity.get("origin_state_hash"),
            "candidate_state_hash": candidate.get("candidate_state_hash"),
            "origin_optimizer_hash": audit.get("optimizer_hash_before"),
            "candidate_optimizer_hash": candidate.get("candidate_optimizer_state_hash"),
            "inference_fingerprint": candidate.get("candidate_policy_fingerprint"),
            "checkpoint_step": origin.get("origin_checkpoint_step"),
            "bank_id": bank.name,
            "candidate_id": cid,
            "intervention": candidate.get("candidate_spec"),
            "update_norm": norm,
            "alias_of": alias["alias_of"] if alias else None,
            "source_files": [
                *common_refs,
                _ref(bank / "candidates" / cid / "candidate.json", root),
            ],
            "evidence_level": "METADATA_CANDIDATE_NO_DIRECT_COUNTS",
        }
        if not measurement:
            rows.append(_row(base))
            continue
        counts, grouped, ledger, level = measurement
        source_cid = alias["alias_of"] if alias else cid
        refs = base["source_files"] + [
            _ref(bank / "candidates" / source_cid / "counts_by_prompt.json", root)
        ]
        ledger_hash = None
        if ledger:
            refs.append(_ref(ledger, root))
            ledger_hash = refs[-1]["sha256"]
        for item in counts:
            actual = grouped.get(item["prompt_id"], {})
            rows.append(
                _row(
                    {
                        **base,
                        **{
                            key: item.get(key)
                            for key in ("base_scene_id", "prompt_id", "family", "interface")
                        },
                        "operation": actual.get("operation"),
                        "seed": actual.get("seed"),
                        **{"n" + event: item["counts"][event] for event in EVENTS},
                        "n": item["n"],
                        "sample_ledger_hash": ledger_hash,
                        "source_files": refs,
                        "evidence_level": level,
                    }
                )
            )
    origins = {c.get("audit", {}).get("optimizer_hash_before") for c in candidates.values()}
    seeds = sorted({row["seed"] for row in rows if row["seed"] is not None})
    for candidate in candidates.values():
        recorded_origin = candidate.get("audit", {}).get("origin_state_hash")
        if recorded_origin is not None and recorded_origin != identity.get("origin_state_hash"):
            raise ValueError("candidate origin state differs from bank origin")
    if len(origins - {None}) > 1:
        raise ValueError("candidates do not share origin optimizer metadata")
    paired = None
    if "joint_0" in totals and "joint_1" in totals:

        def metrics(counts):
            n = sum(counts.values())
            valid = n - counts["I"]
            return {
                "pX": counts["X"] / n,
                "v": valid / n,
                "qX": counts["X"] / valid if valid else None,
            }

        left, right = metrics(totals["joint_0"]), metrics(totals["joint_1"])
        paired = {
            key: right[key] - left[key]
            if right[key] is not None and left[key] is not None
            else None
            for key in left
        }
    return (
        rows,
        {
            **info,
            **manifest,
            "audit_state": "LOCAL_AVAILABLE_BYTES_AND_COUNTS_VERIFIED",
            "usable": True,
            "metadata_only_candidate_count": sum(
                c.get("_measurement") is None and cid not in aliases
                for cid, c in candidates.items()
            ),
            "raw_semantic_records_recomputed": reparsed_total,
            "unique_sample_count": len(all_keys),
            "same_origin_optimizer_metadata": len(origins) == 1 and None not in origins,
            "same_probe_panel_and_budget": bool(panels),
            "seed_coverage": seeds,
            "origin_state_hash": identity.get("origin_state_hash"),
            "origin_checkpoint_step": origin.get("origin_checkpoint_step"),
            "candidate_counts": totals,
            "joint_1_minus_joint_0": paired,
            "response_reference": "CANDIDATE_JOINT_0_AFTER_ONE_ADAM_STEP; NOT_ORIGIN_DISTRIBUTION",
            "policy_aliases": {cid: value["alias_of"] for cid, value in aliases.items()},
            "source_commit_receipt": source_commit,
            "source_commit_rebuilt_from_git": False,
            "parameter_layout_verified": False,
            "dtype_verified_from_tensors": False,
            "actual_update_vectors_loaded": False,
            "raw_or_weights_exported": False,
        },
        all_keys,
    )


def run_audit_real(readonly_root: Path, out: Path) -> dict[str, Any]:
    """Audit accessible completed artifacts into an exclusively new directory."""
    if not isinstance(readonly_root, Path) or not isinstance(out, Path):
        raise TypeError("readonly_root and out must be pathlib.Path")
    root, out = _safe(readonly_root), _safe(out)
    if out.exists():
        raise FileExistsError("audit output already exists")
    started = perf_counter()
    flow = root / "ssvc_flow" if (root / "ssvc_flow").is_dir() else root
    bank_dirs = set()
    for base in (root, flow):
        if (base / "identity.json").is_file() and (base / "candidates").is_dir():
            bank_dirs.add(base)
        for pattern in (
            "S1/*/identity.json",
            "*_review/evidence/S1/*/identity.json",
            "runs/mechanism_followup_server/*/*/*_review/evidence/S1/*/identity.json",
        ):
            bank_dirs.update(path.parent for path in base.glob(pattern))
    scenes, parser_info = {}, {"verified": False}
    control = flow / "data/generated/control.jsonl"
    if control.is_file():
        for line in _safe(control).read_text().splitlines():
            scene = json.loads(line)
            if scene["base_scene_id"] in scenes and scenes[scene["base_scene_id"]] != scene:
                raise ValueError("duplicate local control scene identity")
            scenes[scene["base_scene_id"]] = scene
    plans = sorted(
        flow.glob("runs/mechanism_followup_server/*/*/pro6000_route/validated_plan.json")
    )
    if plans:
        plan = _read(plans[-1])
        names = ("src/verifiers.py", "src/legacy_frozen.py", "src/r2_runtime.py")
        sources = plan.get("source_files", {})
        parser_sources = {name: sources.get(name) for name in names}
        current_flow = Path(__file__).resolve().parents[2]
        verified = all(
            value is not None
            and (current_flow / name).is_file()
            and _hash(current_flow / name) == value
            for name, value in parser_sources.items()
        )
        parser_info = {
            "verified": verified,
            "parser_version_hash": canonical_hash(parser_sources),
            "plan_source": _ref(plans[-1], root),
            "parser_sources": parser_sources,
            "control_source": _ref(control, root) if control.exists() else None,
        }
    rows, banks, all_keys, failures = [], [], set(), []
    for bank in sorted(bank_dirs):
        try:
            imported, info, keys = _bank(_safe(bank), root, scenes, parser_info)
            if keys & all_keys:
                raise ValueError("duplicate sample keys across completed bank bundles")
            all_keys |= keys
            rows.extend(imported)
            banks.append(info)
        except (ValueError, KeyError, TypeError, OSError) as error:
            failures.append(
                {"bank_id": bank.name, "reason": str(error).replace(str(root), "<evidence_root>")}
            )
    history = []
    historical_root = flow / "runs/NEXT_20260909/final_document_work/server_sources"
    for phase in ("R3_cold", "R3_warm", "R4"):
        path = historical_root / phase / "status.json"
        if path.exists():
            status = _read(path)
            item = {
                key: status.get(key)
                for key in ("phase", "status", "source_commit", "execution_kind")
            }
            item.update(
                {
                    "evidence_level": "METADATA_ONLY",
                    "source": _ref(path, root),
                    "raw_semantics_recomputed": False,
                    "update_vectors_loaded": False,
                }
            )
            manifest = historical_root / phase / "candidate_manifest.json"
            if manifest.exists():
                item["response_bank_indices"] = _read(manifest).get("response_bank_indices")
                item["candidate_manifest_source"] = _ref(manifest, root)
            history.append(item)
    snapshots = sorted(
        flow.glob("runs/mechanism_followup_server/*/*/s1_launch/status_check_*.json")
    )
    snapshot = None
    if snapshots:
        latest = _read(snapshots[-1])
        snapshot = {
            "source": _ref(snapshots[-1], root),
            "checked_utc": latest.get("checked_utc"),
            "jobs": {
                key: {
                    "job_id": value.get("job_id"),
                    "completed_marker": value.get("completed"),
                    "recorded_status": value.get("status"),
                    "completion_verified_here": False,
                }
                for key, value in latest.get("jobs", {}).items()
            },
            "snapshot_only_not_live_server_status": True,
        }
    usable = [bank["bank_id"] for bank in banks if bank.get("usable")]
    summary = {
        "schema_version": "modeling-real-audit-v1",
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "INTEGRITY_FAILURE"
        if failures
        else "PARTIAL_REAL_EVIDENCE_READY"
        if usable
        else "MISSING_REAL_COMPLETION_EVIDENCE",
        "usable_completed_banks": usable,
        "bank_audits": banks,
        "failures": failures,
        "unique_sample_count": len(all_keys),
        "unified_row_count": len(rows),
        "raw_semantic_records_recomputed": sum(
            bank.get("raw_semantic_records_recomputed", 0) for bank in banks
        ),
        "parser_audit": parser_info,
        "historical_metadata": history,
        "latest_local_execution_snapshot": snapshot,
        "real_response_dimension": None,
        "real_response_dimension_reason": (
            "BOUND_UPDATE_VECTORS_NOT_IMPORTED; ONE_WARM_ORIGIN_CANNOT_ID"
            "ENTIFY_CROSS_CHECKPOINT_DYNAMICS"
        ),
        "real_origin_response_available": False,
        "real_origin_response_reason": (
            "NO_MATCHED_ORIGIN_DISTRIBUTION_IMPORTED; JOINT_0_IS_AFTER_ONE_ADAM_STEP"
        ),
        "new_qwen_calls": 0,
        "gpu_operations": 0,
        "remote_operations": 0,
        "checkpoint_loading": False,
        "sealed_confirm_read": False,
        "elapsed_seconds": perf_counter() - started,
        "export_requirements": [
            "Ordered LoRA parameter names/shapes/dtypes and layout hash",
            (
                "Actual realized update vectors or lossless parameter deltas "
                "bound to origin/candidate/optimizer hashes"
            ),
            "Matched origin distribution on identical probe panel and sample budget",
            (
                "Completed composite evidence with manifest and alias-aware p"
                "er-prompt raw/count exports"
            ),
            (
                "Multiple independent training seeds and checkpoints with pre"
                "assigned fit/selection/test roles"
            ),
        ],
        "scope_note": (
            "Local compact bundle omissions do not prove server originals"
            " are absent. Existing server verification receipts remain me"
            "tadata claims unless recomputed above."
        ),
    }
    out.mkdir(parents=True, exist_ok=False)
    with (out / "real_evidence_table.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )
    (out / "audit_summary.json").write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    report = [
        "# 既有真实证据只读适配（M5）",
        "",
        f"状态：`{summary['audit_status']}`。本轮只读本机已取回产物；"
        "不读取 sealed confirm，不加载 checkpoint，不调用 Qwen 或 GPU。",
        "",
        f"本机可复核完成 bank：{', '.join(usable) or '无'}。"
        f"独立样本 {len(all_keys)}；统一表 {len(rows)} 行；"
        f"本轮语义重新计算 {summary['raw_semantic_records_recomputed']} 条。",
        (
            "相同策略 alias 与 direct/flat ledger 副本均不增加独立样本量"
            "。每个缺失字段使用 null + reason。"
        ),
        "",
        "| bank | 独立样本 | 语义复算 | ΔpX (joint_1 − joint_0) | Δv | ΔqX |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bank in banks:
        delta = bank.get("joint_1_minus_joint_0") or {}
        report.append(
            f"| {bank['bank_id']} | {bank.get('unique_sample_count', 0)} | "
            f"{bank.get('raw_semantic_records_recomputed', 0)} | "
            f"{delta.get('pX')} | {delta.get('v')} | {delta.get('qX')} |"
        )
    report += [
        "",
        "## 可支持的结论",
        "",
        (
            "上述为已完成候选间的实测比例差，qX 按所有有效样本加权后取比"
            "。joint_0 自身执行了一次 Adam 更新，不能把它当作原点分布。点"
            "估计没有添加未计算的显著性结论。"
        ),
        (
            "历史 R3-cold/R3-warm 的 bank 0/6 与新 S1 bank00/03/04/05/11 "
            "属于不同实验；历史 R4 是单独训练证据。历史数据在本次仅导入状"
            "态和清单元数据，未声称重算其原始输出。"
        ),
        "",
        "## 缺项与边界",
        "",
        (
            "本机紧凑包没有导入绑定参数布局的实际更新向量；现有范数、梯度"
            "哈希、参数哈希和几何摘要均不能代替向量，故真实 J、夹角和响应"
            "维数保持未知。服务器上可能保有 checkpoint；本次本机审计不能"
            "推断服务器原件缺失。"
        ),
        (
            "原点/候选/Adam 哈希仅从已校验文件读取；未加载原始张量，因此"
            "没有把张量哈希元数据称为独立重算。未核验 LoRA 参数顺序或 ten"
            "sor dtype。"
        ),
        (
            "新 S1 仍只有一个 warm 原点（step64）及 seed17；不支持拆分独"
            "立训练 seed 的拟合/验证，也不支持跨 checkpoint 泛化。"
        ),
        (
            "本机 composite 最新快照仅为状态参考；没有本机完成原件就不导"
            "入部分输出，不以 RUNNING、样本计数或提交回执作为完成。"
        ),
        "",
        "## 最小已有原件导出清单",
        "",
    ]
    report.extend("- " + item for item in summary["export_requirements"])
    report += [
        "",
        "上述是所需数据导出清单，不授权新 GPU 测量、训练、采样或控制。",
        "",
        (
            "文件和字段来源见 `real_evidence_table.jsonl`；字节哈希、各 b"
            "ank 完整性检查、遗漏清单、状态快照和时间见 `audit_summary.js"
            "on`。相对 evidence_id 是来源标识，不包含私有绝对路径。"
        ),
        "",
    ]
    (out / "REAL_EVIDENCE_READINESS_zh.md").write_text("\n".join(report), encoding="utf-8")
    return summary
