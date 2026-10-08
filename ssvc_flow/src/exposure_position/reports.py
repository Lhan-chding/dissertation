"""Release-gated SER-J23 scientific tables and literal execution/accounting status.

All result probabilities remain in [0,1] in machine tables; ``*_pp`` fields and
figure axes explicitly use percentage points. Missing costs never become zero.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import stat
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..verified_discovery_transfer.queue import digest as evidence_digest
from .schema import digest as scientific_digest
from .schema import file_digest, read_json
from .semantics import aggregate_events, score_output
from .statistics import ARMS, PARENTS, TASK_FIELDS, analyze_confirmation, summarize_tasks

PANELS = ("E_CONFIRM2", "TRAIN_FIT_FULL", "DEV_TRAJECTORY", "DEV_DELTA")


def _bytes(path, payload):
    """Replace a derived report atomically while retaining its previous bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = path.read_bytes()
        if previous == payload:
            return
        old = path.parent / ".history" / file_digest(path) / path.name
        old.parent.mkdir(parents=True, exist_ok=True)
        if old.exists() and old.read_bytes() != previous:
            raise ValueError("Derived report history hash collision")
        old.write_bytes(previous)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _json(path, value):
    _bytes(
        path,
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode(),
    )


def _csv(path, rows, *, fields=()):
    rows = list(rows)
    names = list(dict.fromkeys((*fields, *(key for row in rows for key in row))))
    if not names:
        names = ["status", "reason"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=names)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (dict, list, tuple)) or value is None
                else value
                for key, value in row.items()
            }
        )
    _bytes(path, output.getvalue().encode())


def _jsonl(path, rows):
    _bytes(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ).encode(),
    )


def _valid_digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def write_root_manifest(run):
    """Hash a released run after reports/stage receipt, including all local weights.

    External historical bytes remain references to the frozen CPU audit, not a
    claim that this run contains or freshly reverified those external files.
    """
    started, cpu_started = time.perf_counter(), time.process_time()
    run = Path(run).resolve(strict=True)
    manifest_path = run / "FILE_MANIFEST_SHA256.json"

    def fingerprint(path):
        s = path.lstat()
        return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    if manifest_path.is_symlink():
        raise ValueError("Root manifest must not be a symlink")
    previous_stat = None
    # Archive before scanning, so this new history file is itself covered.
    if manifest_path.exists():
        previous_stat = fingerprint(manifest_path)
        previous = manifest_path.read_bytes()
        previous_hash = file_digest(manifest_path)
        if fingerprint(manifest_path) != previous_stat:
            raise ValueError("Root manifest changed while archiving")
        old = run / ".history" / previous_hash / manifest_path.name
        if any(path.is_symlink() for path in (old, old.parent, old.parent.parent)):
            raise ValueError("Root manifest history must not contain symlinks")
        old.parent.mkdir(parents=True, exist_ok=True)
        if old.exists() and old.read_bytes() != previous:
            raise ValueError("Root manifest history hash collision")
        if not old.exists():
            old.write_bytes(previous)

    def inventory():
        files, omissions = {}, []

        def visit(directory):
            for path in sorted(directory.iterdir()):
                relative = path.relative_to(run).as_posix()
                if path == manifest_path:
                    continue
                info = fingerprint(path)
                if stat.S_ISLNK(info[2]):
                    omissions.append(
                        {
                            "path": relative,
                            "reason": "symlink_not_followed",
                            "target": os.readlink(path),
                        }
                    )
                elif path.name.endswith((".lock", ".partial", ".tmp", "-shm")) or path.name in (
                    "tmp",
                    "partial",
                    "locks",
                    ".tmp",
                    ".partial",
                    ".locks",
                ):
                    omissions.append(
                        {
                            "path": relative,
                            "reason": "runtime_lock_or_uncommitted_temporary_artifact",
                            "scope": "directory_subtree" if stat.S_ISDIR(info[2]) else "file",
                        }
                    )
                elif stat.S_ISDIR(info[2]):
                    visit(path)
                elif stat.S_ISREG(info[2]):
                    files[relative] = info
                else:
                    omissions.append({"path": relative, "reason": "non_regular_file_not_read"})

        visit(run)
        return files, omissions

    before, omissions = inventory()
    files = {}
    for relative, initial in before.items():
        path = run / relative
        # Resolve immediately before hashing as well as lstat afterwards: an
        # external symlink substitution must not become included evidence.
        if not path.resolve(strict=True).is_relative_to(run) or fingerprint(path) != initial:
            raise ValueError("Run file changed before manifest hashing: " + relative)
        checksum = file_digest(path)
        if fingerprint(path) != initial:
            raise ValueError("Run file changed during manifest hashing: " + relative)
        files[relative] = {"sha256": checksum, "bytes": initial[3]}
    binding_name = "PARENT_AND_ENDPOINT_BINDINGS.json"
    if not {binding_name, "FROZEN_PLAN.json"} <= files.keys():
        raise ValueError("Root manifest requires regular frozen checkpoint binding files")
    frozen, bindings = read_json(run / "FROZEN_PLAN.json"), read_json(run / binding_name)
    if frozen.get("files", {}).get(binding_name, {}).get("sha256") != files[binding_name]["sha256"]:
        raise ValueError("External checkpoint bindings differ from the frozen plan")
    after, final_omissions = inventory()
    if before != after or omissions != final_omissions:
        raise ValueError("Run files changed during manifest inventory; retry after writers stop")
    result = {
        "schema": "ser-j23-run-file-manifest-v1",
        "status": "PERSISTED_FILES_HASHED_WITH_DECLARED_OMISSIONS",
        "files": files,
        "file_count": len(files),
        "total_hashed_bytes": sum(row["bytes"] for row in files.values()),
        "self_exclusion": {"path": manifest_path.name, "reason": "manifest_self_hash_recursion"},
        "omissions": omissions,
        "local_checkpoint_policy": (
            "All regular run-local checkpoint bytes included; no size exclusion"
        ),
        "external_evidence": {
            "binding_file": binding_name,
            "binding_file_sha256": files[binding_name]["sha256"],
            "frozen_plan_sha256": files["FROZEN_PLAN.json"]["sha256"],
            "checkpoint_lookup": bindings.get("checkpoint_lookup", {}),
            "base_snapshot": bindings.get("base_snapshot"),
            "scope": (
                "External historical files are not copied or read here; locations, hashes, "
                "verification statuses and other evidence remain in the frozen binding file. "
                "This is a binding reference, not fresh verification of external bytes."
            ),
        },
        "timing": {
            "manifest_wall_seconds": time.perf_counter() - started,
            "manifest_process_cpu_seconds": time.process_time() - cpu_started,
            "scope": "Archiving, inventory and file hashing before atomic manifest write",
            "separate_from_report_CPU_analysis_seconds": True,
            "added_to_allocated_gpu_hours": False,
            "allocation_note": (
                "CPU hashing is not converted into GPU hours; sacct allocation envelopes "
                "already account for any concurrent GPU allocation."
            ),
        },
    }
    # History is already archived and indexed; _json safely reuses those bytes.
    if (manifest_path.exists() or manifest_path.is_symlink()) != (previous_stat is not None) or (
        previous_stat is not None and fingerprint(manifest_path) != previous_stat
    ):
        raise ValueError("Root manifest changed before atomic replacement")
    _json(manifest_path, result)
    return result


def _release_context(output_dir, receipt):
    """Never open confirmation audit data before receipt AND queue seal agree."""
    from .evaluate import evaluation_tasks
    from .queue import registered_queue
    from .schema import RoleDataset, read_bound, verify_frozen

    output = Path(output_dir).resolve()
    run = output.parent
    if output.name != "reports":
        raise ValueError("Official reports must be written in the released run/reports directory")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "ser-j23-release-v1"
        or receipt.get("status") != "RELEASED"
        or any(
            receipt.get(name) is not True
            for name in (
                "all_registered_models_terminal",
                "all_confirmation_requests_complete",
                "all_registered_jobs_complete",
            )
        )
        or receipt.get("registered_training_jobs") not in (12, 24)
        or not all(_valid_digest(receipt.get(name)) for name in ("plan_hash", "matrix_digest"))
    ):
        raise PermissionError("Scientific reports require the complete sealed J23 release receipt")
    if read_json(run / "RELEASE_RECEIPT.json") != receipt:
        raise PermissionError("Provided release differs from the immutable on-disk receipt")
    frozen = verify_frozen(run, role="control")
    if receipt["plan_hash"] != frozen["plan_hash"]:
        raise PermissionError("Release names a different frozen plan")
    queue = registered_queue(run)
    try:
        rows = queue.rows()
        seal = queue.db.execute("SELECT value FROM metadata WHERE key='released'").fetchone()
        if not seal or seal[0] != receipt["matrix_digest"] or evidence_digest(rows) != seal[0]:
            raise PermissionError("Scientific reports require an intact queue release seal")
        if any(row["status"] != "COMPLETE" for row in rows):
            raise PermissionError("Incomplete queue cannot enter scientific reporting")
    finally:
        queue.db.close()
    # This is the first confirmation truth access in the report entry point.
    data = RoleDataset(run, "analysis")
    public = {panel: {row["task_id"]: row for row in data.public(panel)} for panel in PANELS}
    audits = {panel: {row["task_id"]: row for row in data.audit(panel)} for panel in PANELS}
    registry = []
    for task_id, task in public["E_CONFIRM2"].items():
        audit = audits["E_CONFIRM2"][task_id]
        combined = {**task, **audit}
        registry.append({field: combined[field] for field in TASK_FIELDS})
    jobs = read_bound(run, "manifests/evaluation_jobs.jsonl")
    history_receipt_path = "HISTORICAL_TRAJECTORY_REUSE.json"
    history_rows_path = "manifests/historical_trajectory_reused_rows.jsonl"
    if history_receipt_path in frozen["files"] and history_rows_path in frozen["files"]:
        historical_receipt = read_bound(run, history_receipt_path)
        historical_rows = read_bound(run, history_rows_path)
    else:
        historical_receipt = {
            "status": "UNAVAILABLE",
            "compatible_models": [],
            "unavailable": [
                {"reason": "compatible_historical_raw_baselines_not_bound_before_freeze"}
            ],
            "new_model_calls": 0,
            "new_logical_generations": 0,
        }
        historical_rows = []

    def training_audit(split):
        tasks = {task["task_id"]: task for task in data.public(split)}
        return [{**tasks[audit["task_id"]], **audit} for audit in data.audit(split)]

    return {
        "run": run,
        "frozen": frozen,
        "queue_rows": rows,
        "public": public,
        "audits": audits,
        "registry": registry,
        "jobs": jobs,
        "job_tasks": {job["job_id"]: evaluation_tasks(run, job) for job in jobs},
        "mode": read_bound(run, "EXECUTION_MODE.json")["mode"],
        "training_audits": {
            split: training_audit(split) for split in ("COMMON_TRAIN", "REPLAY", "DONOR_TRAIN")
        },
        "delta_legality": read_bound(run, "manifests/DEV_DELTA/legality_attempts.jsonl"),
        "diagnostic_availability": read_bound(run, "manifests/diagnostic_availability.json"),
        "historical_receipt": historical_receipt,
        "historical_rows": historical_rows,
    }


def _historical_trajectory_scores(context):
    """Reuse only immutable pre-audited old outputs; retain original seeds/identity."""
    receipt = context.get("historical_receipt") or {"status": "UNAVAILABLE"}
    rows = context.get("historical_rows", [])
    if receipt.get("status") == "UNAVAILABLE":
        if rows:
            raise ValueError("Unavailable historical trajectory cannot contain answers")
        return []
    if (
        receipt.get("status") != "AUDITED"
        or receipt.get("mode") != context["mode"]
        or receipt.get("new_model_calls") != 0
        or receipt.get("new_logical_generations") != 0
        or receipt.get("rows_count") != len(rows)
        or receipt.get("rows_hash") != scientific_digest(rows)
    ):
        raise ValueError("Historical trajectory reuse must be frozen and independently audited")
    models = {}
    fields = ("parent", "block", "logical_arm_id", "step")
    for model in receipt["compatible_models"]:
        key = tuple(model[name] for name in fields)
        if key in models or model.get("compatible") is not True or model.get("samples") != 1024:
            raise ValueError("Historical trajectory model compatibility/count mismatch")
        parent, block, arm, step = key
        valid_parent = parent in PARENTS and block is None and arm == "PARENT" and step == 0
        valid_student = (
            context["mode"] == "REUSE_12"
            and parent in PARENTS
            and type(block) is int
            and block in range(3)
            and arm in ARMS[:2]
            and step == 256
        )
        if not (valid_parent or valid_student):
            raise ValueError("Unbudgeted or incompatible historical trajectory endpoint")
        models[key] = model
    expected_models = {(parent, None, "PARENT", 0) for parent in PARENTS}
    if context["mode"] == "REUSE_12":
        expected_models |= {
            (parent, block, arm, 256)
            for parent in PARENTS
            for block in range(3)
            for arm in ARMS[:2]
        }
    unavailable = [
        tuple(model[name] for name in fields) for model in receipt.get("unavailable", [])
    ]
    if (
        len(unavailable) != len(set(unavailable))
        or set(unavailable) & set(models)
        or set(unavailable) | set(models) != expected_models
    ):
        raise ValueError("Every historical endpoint must be compatible or explicitly unavailable")
    actual, by_model, seen, scored = Counter(), defaultdict(list), set(), []
    for row in rows:
        key = tuple(row[name] for name in fields)
        original = row.get("original_record", {})
        if (
            key not in models
            or row.get("origin") != "historical_reuse"
            or row.get("panel") != "DEV_TRAJECTORY"
            or row.get("sample_id") in seen
            or row.get("sample_id") != original.get("request_id", original.get("sample_id"))
            or any(
                row.get(field) != original.get(field)
                for field in ("sample_seed", "raw_text", "token_ids", "stop_reason")
            )
            or row.get("source_record_sha256") != scientific_digest(original)
            or row.get("compatibility_verified") is not True
        ):
            raise ValueError("Historical raw answer/seed/identity was changed or duplicated")
        seen.add(row["sample_id"])
        actual[key] += 1
        by_model[key].append(row)
        task, audit = (
            context["public"]["DEV_TRAJECTORY"][row["task_id"]],
            context["audits"]["DEV_TRAJECTORY"][row["task_id"]],
        )
        if (
            row["root_id"] != task["root_id"]
            or original.get("task_id") != audit.get("source_task_id")
            or original.get("root_id") != audit.get("source_root_id")
        ):
            raise ValueError("Historical alias changed the frozen source task/root mapping")
        scored.append({**row, **score_output(task, audit, row["raw_text"], row["stop_reason"])})
    if set(actual) != set(models) or any(count != 1024 for count in actual.values()):
        raise ValueError("Incomplete compatible historical trajectory matrix")
    for key, model_rows in by_model.items():
        if scientific_digest(model_rows) != models[key]["alias_records_hash"]:
            raise ValueError("Historical model alias rows differ from their frozen digest")
    # Also reject repeated task/draws despite distinct sample IDs.
    summarize_tasks(scored, expected_draws=8)
    return scored


def _verify_scored_against_release(scored, context, receipt):
    """Bind every supplied score to the immutable raw record, task and receipt."""
    by_job = defaultdict(dict)
    sample_ids = set()
    for row in scored:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or sid in sample_ids:
            raise ValueError("Missing or duplicate supplied scored sample identity")
        sample_ids.add(sid)
        key = row["task_id"], row["draw_index"]
        if key in by_job[row["job_id"]]:
            raise ValueError("Duplicate task/draw in supplied scored matrix")
        by_job[row["job_id"]][key] = row
    if set(by_job) != {job["job_id"] for job in context["jobs"]}:
        raise ValueError(
            "Scored data must include every registered confirmation and diagnostic job"
        )
    completions = {r["job_id"]: r for r in receipt["completed_evaluation_receipts"]}
    for job in context["jobs"]:
        jid, panel = job["job_id"], job["panel"]
        raw_rows = []
        for task in context["job_tasks"][jid]:
            for draw in range(job["draws"]):
                row = by_job[jid].get((task["task_id"], draw))
                if row is None or row.get("execution_kind") != "REAL_FROZEN_GPU":
                    raise ValueError("Missing registered real-model scored answer")
                if not _valid_digest(row["sample_id"]):
                    raise ValueError("Unsafe scored sample identity")
                path = (
                    context["run"]
                    / "evaluations"
                    / jid
                    / "roots"
                    / task["root_id"]
                    / (row["sample_id"] + ".json")
                )
                envelope = read_json(path)
                raw = envelope["record"]
                if (
                    evidence_digest(raw) != envelope.get("record_sha256")
                    or envelope.get("manifest_hash") != completions[jid]["manifest_hash"]
                    or any(
                        row.get(key) != value for key, value in raw.items() if key != "truncated"
                    )
                ):
                    raise ValueError("Scored input differs from its immutable raw envelope")
                derived = score_output(
                    task,
                    context["audits"][panel][task["task_id"]],
                    raw["raw_text"],
                    raw["stop_reason"],
                )
                if any(
                    row.get(key) != value
                    for key, value in derived.items()
                    if key != "projection_cpu_seconds"
                ):
                    raise ValueError(
                        "Supplied scored events differ from independent native scoring"
                    )
                raw_rows.append(raw)
        if (
            len(by_job[jid]) != len(raw_rows)
            or evidence_digest(raw_rows) != completions[jid]["records_hash"]
        ):
            raise ValueError("Scored job coverage differs from the released raw-record digest")


def _enriched_task_rows(scored, context):
    result = []
    for panel in PANELS:
        rows = [row for row in scored if row["panel"] == panel]
        if not rows:
            continue
        tasks = summarize_tasks(
            rows, expected_draws=8 if panel in ("E_CONFIRM2", "DEV_TRAJECTORY") else 4
        )
        origins = defaultdict(set)
        for source in rows:
            origins[source["checkpoint_id"], source["task_id"]].add(source.get("origin"))
        for row in tasks:
            audit = context["audits"][panel][row["task_id"]]
            for field in (
                "signed_delta",
                "source_task_id",
                "source_root_id",
                "source_training_task_id",
                "source_ordinal_0based",
            ):
                if field in audit:
                    row[field] = audit[field]
            row["probability_units"] = "probability_0_to_1"
            source_origins = origins[row["checkpoint_id"], row["task_id"]]
            if len(source_origins) != 1:
                raise ValueError("A task stream cannot mix new and historical generation origins")
            row["historical_reused"] = source_origins == {"historical_reuse"}
            if panel == "TRAIN_FIT_FULL":
                row["fit_subset"] = (
                    "c4j4_training"
                    if (row["center"], row["corrupted_index"]) == (3, 3)
                    else "arm_specific_donor"
                )
        result.extend(tasks)
    return result


def _diagnostic_tables(scored, task_rows, context):
    """Retain each task and fixed path, and compute only same-task model contrasts."""
    output = {
        panel: [
            dict(row, row_type="task", scope="development_only_not_confirmatory")
            for row in task_rows
            if row["panel"] == panel
        ]
        for panel in ("TRAIN_FIT_FULL", "DEV_TRAJECTORY", "DEV_DELTA")
    }
    fields = (
        "panel",
        "parent",
        "block",
        "logical_arm_id",
        "step",
        "family",
        "center",
        "corrupted_index",
        "signed_delta",
    )
    cells = defaultdict(list)
    for row in scored:
        if row["panel"] not in ("DEV_TRAJECTORY", "DEV_DELTA"):
            continue
        audit = context["audits"][row["panel"]][row["task_id"]]
        enriched = {**row, "signed_delta": audit.get("signed_delta")}
        cells[tuple(enriched.get(field) for field in fields)].append(enriched)
    for key, rows in cells.items():
        metadata = dict(zip(fields, key, strict=True))
        anchor_matches, anchor_outside = [], []
        for row in rows:
            task = context["public"][row["panel"]][row["task_id"]]
            center, y = row["center"], row["parsed_vector"]
            matches, outside = [False] * 4, [False] * 4
            if center is not None:
                for relation, rhs in zip(task["H_original"], task["b_original"], strict=True):
                    leaf = next(k for k in range(4) if k != center and relation[k] == 1)
                    compatible = rhs - task["observed"][center]
                    matches[leaf] = y is not None and y[leaf] == compatible
                    outside[leaf] = not 0 <= compatible <= 99
            anchor_matches.append(matches)
            anchor_outside.append(outside)
        output[metadata["panel"]].append(
            {
                **metadata,
                **aggregate_events(rows),
                "row_type": "cell_mean",
                "n_roots": len({row["root_id"] for row in rows}),
                "scope": "development_only_not_confirmatory",
                "old_center_compatible_output_match_by_position": np.mean(
                    anchor_matches, axis=0
                ).tolist(),
                "old_center_compatible_value_out_of_domain_by_position": np.mean(
                    anchor_outside, axis=0
                ).tolist(),
                "mixed_anchor_public_output_match": float(
                    np.mean([row["mixed_anchor_public"] for row in rows])
                ),
                "output_matches_do_not_identify_internal_rules": True,
            }
        )
    streams = defaultdict(dict)
    for row in task_rows:
        if row["panel"] in ("DEV_TRAJECTORY", "DEV_DELTA"):
            key = (
                row["panel"],
                row["parent"],
                row["block"],
                row["logical_arm_id"],
                row["step"],
                row["center"],
                row["corrupted_index"],
                row.get("signed_delta"),
            )
            streams[key][row["task_id"]] = row
    comparisons = defaultdict(list)
    for key, left_tasks in streams.items():
        panel, parent, block, arm, step, center, j, delta = key
        if arm == "PARENT":
            continue
        rights = ["PARENT"] if panel == "DEV_DELTA" else []
        if arm in (ARMS[1], "C_FORWARD_C3"):
            rights.append(ARMS[0])
        for right in rights:
            rhs_key = (
                panel,
                parent,
                None if right == "PARENT" else block,
                right,
                step if right != "PARENT" else 0,
                center,
                j,
                delta,
            )
            right_tasks = streams.get(rhs_key)
            metadata = dict(
                panel=panel,
                parent=parent,
                block=block,
                step=step,
                center=center,
                corrupted_index=j,
                signed_delta=delta,
                left=arm,
                right=right,
                row_type="paired_path_effect",
                scope="development_only_not_confirmatory",
            )
            if right_tasks is None:
                output[panel].append(
                    {
                        **metadata,
                        "status": "UNAVAILABLE",
                        "reason": "comparator_not_registered_or_compatible",
                    }
                )
                continue
            if set(left_tasks) != set(right_tasks):
                raise ValueError("Diagnostic comparator must use the exact same legal tasks")
            for metric in ("X", "F_val", "R", "q1", "q2", "q3"):
                value = float(
                    np.mean(
                        [left_tasks[tid][metric] - right_tasks[tid][metric] for tid in left_tasks]
                    )
                )
                record = {
                    **metadata,
                    "status": "AVAILABLE",
                    "metric": metric,
                    "estimate": value,
                    "estimate_pp": 100 * value,
                    "n_paired_roots": len(left_tasks),
                }
                output[panel].append(record)
                comparisons[panel, parent, step, center, j, delta, arm, right, metric].append(
                    record
                )
    for key, rows in comparisons.items():
        panel, parent, step, center, j, delta, arm, right, metric = key
        blocks = {row["block"] for row in rows}
        output[panel].append(
            dict(
                panel=panel,
                parent=parent,
                step=step,
                center=center,
                corrupted_index=j,
                signed_delta=delta,
                left=arm,
                right=right,
                metric=metric,
                row_type="equal_order_parent_effect",
                available_blocks=sorted(blocks),
                estimate=float(np.mean([r["estimate"] for r in rows]))
                if blocks == {0, 1, 2}
                else None,
                status="AVAILABLE" if blocks == {0, 1, 2} else "INCOMPLETE_FIXED_ORDER_COMPARISON",
            )
        )
    for attempt in context["delta_legality"]:
        if attempt["status"] == "REJECTED_OUT_OF_DOMAIN":
            output["DEV_DELTA"].append(
                {
                    **attempt,
                    "row_type": "infeasible_construction",
                    "status": "INFEASIBLE",
                    "answer_denominator": None,
                    "not_a_model_I_answer": True,
                }
            )
    historical = context.get("historical_receipt", {})
    compatible, unavailable = (
        historical.get("compatible_models", []),
        historical.get("unavailable", []),
    )
    output["DEV_TRAJECTORY"].append(
        {
            "row_type": "baseline_scope",
            "status": "AVAILABLE"
            if compatible and not unavailable
            else "PARTIALLY_AVAILABLE"
            if compatible
            else "UNAVAILABLE",
            "steps_allowed": [0, 256] if context["mode"] == "REUSE_12" else [0],
            "compatible_models": compatible,
            "unavailable": unavailable,
            "mode": context["mode"],
            "new_calls_added": 0,
            "E_CONFIRM2_is_a_separate_panel_not_connected_to_development_trajectory": True,
        }
    )
    output["DEV_DELTA"].append(
        {
            "row_type": "cross_delta_scope",
            "status": "NO_CROSS_DELTA_CAUSAL_COMPARISON",
            "reason": "legal_root_masks_differ_by_delta; comparisons_are_within_delta",
        }
    )
    return output


def training_nll_audit(context):
    """Read committed update 193..256 files, including unchanged legacy sources."""
    from .training import student_directory

    summaries, exposures, errors, source_files = [], [], [], []
    students = [
        job
        for job in context["jobs"]
        if job["panel"] == "E_CONFIRM2" and job["logical_arm_id"] != "PARENT"
    ]
    for job in students:
        arm = job["logical_arm_id"]
        metadata = {name: job[name] for name in ("parent", "block", "logical_arm_id")}
        source = job["source"]
        if source == "LEGACY":
            directory = Path(job["checkpoint_binding"]["path"]).parent
        elif source == "NEW_TRAINING":
            directory = student_directory(context["run"], job["parent"], job["block"], arm)
        else:
            raise ValueError("Unregistered training NLL source")
        audit_index = {}
        for split, audits in context["training_audits"].items():
            for audit in audits:
                if split == "DONOR_TRAIN" and audit.get("logical_arm_id") != arm:
                    continue
                tid = audit.get("source_task_id") if source == "LEGACY" else audit["task_id"]
                if tid in audit_index and audit_index[tid] != audit:
                    raise ValueError("Ambiguous source-task mapping in NLL audit")
                audit_index[tid] = audit
        path_exposures = []
        for step in range(193, 257):
            path = directory / f"update{step:03d}.json"
            try:
                record = read_json(path)
                slots = record["slots"]
                if (
                    record.get("update") != step
                    or record.get("status") != "UPDATE_APPLIED"
                    or record.get("physical_cost_complete") is not True
                    or len(slots) != 16
                    or {slot["slot"] for slot in slots} != set(range(16))
                ):
                    raise ValueError("Committed late-update status/slot allocation mismatch")
                if Counter(slot["role"] for slot in slots) != {
                    "common": 11,
                    "donor": 1,
                    "replay": 4,
                }:
                    raise ValueError("Committed late-update roles changed")
                pending = []
                for slot in slots:
                    loss = slot["sequence_nll"]
                    if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
                        raise ValueError("Invalid per-sequence completion NLL")
                    audit = audit_index[slot["task_id"]]
                    pending.append(
                        {
                            **metadata,
                            "source": source,
                            "source_task_id": slot["task_id"],
                            "task_id": audit["task_id"],
                            "family": audit.get("family", "cross_series"),
                            "center": audit["center"],
                            "corrupted_index": audit["corrupted_index"],
                            "step": step,
                            "slot": slot["slot"],
                            "role": slot["role"],
                            "sequence_nll": float(loss),
                            "source_path": str(path),
                            "measurement": "logged_step_teacher_forcing_not_uniform_step256_eval",
                        }
                    )
                source_files.append({"path": str(path), "sha256": file_digest(path)})
                path_exposures.extend(pending)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(
                    {
                        **metadata,
                        "source": source,
                        "step": step,
                        "path": str(path),
                        "status": "MISSING_OR_INVALID_NLL_EVIDENCE",
                        "error": str(exc),
                    }
                )
        exposures.extend(path_exposures)
        cells, last = defaultdict(list), {}
        for row in path_exposures:
            cells[row["family"], row["center"], row["corrupted_index"]].append(row)
            prior = last.get(row["source_task_id"])
            if prior is None or (row["step"], row["slot"]) > (prior["step"], prior["slot"]):
                last[row["source_task_id"]] = row
        for cell, rows in cells.items():
            values = np.asarray([row["sequence_nll"] for row in rows])
            summaries.append(
                {
                    **metadata,
                    "source": source,
                    "row_type": "late_cell_summary",
                    "family": cell[0],
                    "center": cell[1],
                    "corrupted_index": cell[2],
                    "steps_in_scope": [193, 256],
                    "exposure_count": len(rows),
                    "unique_task_count": len({row["source_task_id"] for row in rows}),
                    "mean_completion_nll": float(values.mean()),
                    "quantiles_min_q25_median_q75_max": np.quantile(
                        values, [0, 0.25, 0.5, 0.75, 1], method="linear"
                    ).tolist(),
                    "status": "COMPLETE_PATH"
                    if len(path_exposures) == 64 * 16
                    else "INCOMPLETE_PATH",
                }
            )
        summaries.extend(
            {**row, "row_type": "last_per_task", "last_observed_step": row["step"]}
            for row in last.values()
        )
    return {
        "status": "COMPLETE" if len(students) == 24 and not errors else "INCOMPLETE",
        "summaries": summaries,
        "exposures": exposures,
        "errors": errors,
        "source_files": source_files,
        "paths": len(students),
    }


def gpu_accounting(run, receipt):
    """Never infer allocated GPU hours from logical dose or model elapsed time.

    An explicit SCHEDULER_ACCOUNTING.json must cover submitted allocations,
    including bridge/failed/cancelled/CPU jobs. Each SLURM allocation appears once;
    job steps (e.g. .batch/.extern) are not separate allocations.
    """
    run = Path(run)
    worker_starts = [read_json(p) for p in (run / "worker_attempts").glob("*/STARTED.json")]
    worker_ids = {str(row["slurm_job_id"]) for row in worker_starts if row.get("slurm_job_id")}
    new_attempts = list((run / "training").glob("*/attempts/*/update*.json"))
    technical_attempts = list((run / "bridge" / "attempts").glob("physical*.json"))
    generation_starts = list((run / "evaluations").glob("*/attempts/*/*/STARTED.json"))
    ledger = []
    for category, paths in (
        ("new_training_update", new_attempts),
        ("technical_bridge_update", technical_attempts),
    ):
        for path in paths:
            record = read_json(path)
            ledger.append(
                {
                    "category": category,
                    "path": str(path),
                    "sha256": file_digest(path),
                    "status": record.get("status", "UNKNOWN"),
                    "physical_cost_complete": record.get("physical_cost_complete") is True,
                    "processed_target_sequences": record.get("processed_target_sequences"),
                    "processed_target_tokens": record.get("processed_target_tokens"),
                }
            )
    for path in generation_starts:
        status = next(
            (
                name
                for name in ("ACCEPTED", "FAILED", "RECOVERY")
                if (path.parent / (name + ".json")).exists()
            ),
            "UNKNOWN",
        )
        ledger.append(
            {
                "category": "generation_attempt",
                "path": str(path),
                "sha256": file_digest(path),
                "status": status,
                "physical_cost_complete": status in ("ACCEPTED", "FAILED"),
                "terminal_records": [
                    {
                        "path": str(path.parent / (name + ".json")),
                        "sha256": file_digest(path.parent / (name + ".json")),
                    }
                    for name in ("ACCEPTED", "FAILED", "RECOVERY")
                    if (path.parent / (name + ".json")).exists()
                ],
            }
        )
    for path in (run / "worker_attempts").glob("*/STARTED.json"):
        status = (
            "COMPLETE"
            if (path.parent / "COMPLETE.json").exists()
            else "FAILURE"
            if (path.parent / "FAILURE.json").exists()
            else "UNKNOWN"
        )
        ledger.append(
            {
                "category": "worker_attempt",
                "path": str(path),
                "sha256": file_digest(path),
                "status": status,
                "terminal_records": [
                    {
                        "path": str(path.parent / (name + ".json")),
                        "sha256": file_digest(path.parent / (name + ".json")),
                    }
                    for name in ("COMPLETE", "FAILURE")
                    if (path.parent / (name + ".json")).exists()
                ],
            }
        )
    for pattern in ("*/attempts/*/FAILURE.json", "*/attempts/*/PROCESS.json"):
        for path in (run / "training").glob(pattern):
            ledger.append(
                {
                    "category": "training_process_history",
                    "path": str(path),
                    "sha256": file_digest(path),
                    "status": read_json(path).get("status", "UNKNOWN"),
                }
            )
    result = {
        "status": "UNVERIFIED",
        "allocated_gpu_hours": None,
        "reason": "SCHEDULER_ACCOUNTING_NOT_IMPORTED",
        "scheduler_jobs": [],
        "logical_new_training_updates": 256 * receipt["registered_training_jobs"],
        "logical_new_training_target_exposures": 4096 * receipt["registered_training_jobs"],
        "logical_committed_generations": sum(
            r["samples"] for r in receipt["completed_evaluation_receipts"]
        ),
        "physical_new_training_updates_started": len(new_attempts),
        "physical_technical_updates_started": len(technical_attempts),
        "physical_generation_attempts_started": len(generation_starts),
        "physical_failed_generation_attempts": sum(
            row["status"] == "FAILED" for row in ledger if row["category"] == "generation_attempt"
        ),
        "physical_generation_unknown_outcomes": sum(
            row["status"] in ("UNKNOWN", "RECOVERY")
            for row in ledger
            if row["category"] == "generation_attempt"
        ),
        "physical_incomplete_update_outcomes": sum(
            not row.get("physical_cost_complete", False)
            for row in ledger
            if row["category"] in ("new_training_update", "technical_bridge_update")
        ),
        "attempt_ledger": ledger,
        "historical_reused_training_cost": "historical_only_not_added_to_this_phase_GPU_hours",
        "allocation_includes_loading_idle_failed_retry_teacher_forcing_work": True,
        "CPU_analysis_seconds": None,
    }
    path = run / "SCHEDULER_ACCOUNTING.json"
    if not path.exists():
        return result
    try:
        account = read_json(path)
        if (
            account.get("status") != "VERIFIED"
            or account.get("source") != "sacct"
            or account.get("plan_hash") != receipt["plan_hash"]
            or account.get("allocation_scope_complete") is not True
        ):
            raise ValueError("Scheduler accounting is not explicitly verified and plan-bound")
        submitted = account["submitted_slurm_job_ids"]
        jobs = account["jobs"]
        ids = [str(job["slurm_job_id"]) for job in jobs]
        if (
            not submitted
            or len(submitted) != len(set(submitted))
            or len(ids) != len(set(ids))
            or set(ids) != set(map(str, submitted))
            or not worker_ids <= set(ids)
        ):
            raise ValueError("Scheduler records do not cover all unique submitted allocations")
        if not account.get("raw_files"):
            raise ValueError("Raw sacct evidence is required")
        for raw in account["raw_files"]:
            raw_path = Path(raw["path"])
            raw_path = raw_path if raw_path.is_absolute() else run / raw_path
            if file_digest(raw_path) != raw["sha256"]:
                raise ValueError("Raw sacct evidence hash mismatch")
        gpu_hours, cpu_jobs = 0.0, []
        terminal = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
        }
        categories = set()
        for job in jobs:
            elapsed, gpus = job["elapsed_seconds"], job["allocated_gpus"]
            if (
                type(gpus) is not int
                or gpus < 0
                or type(elapsed) not in (int, float)
                or not math.isfinite(elapsed)
                or elapsed < 0
                or job["state"].split()[0] not in terminal
                or "." in str(job["slurm_job_id"])
                or not job.get("workload_category")
            ):
                raise ValueError(
                    "Invalid allocation duration/GPU count/state or duplicate job-step entry"
                )
            gpu_hours += elapsed * gpus / 3600
            categories.add(job["workload_category"])
            if gpus == 0:
                cpu_jobs.append(job)
        if technical_attempts and "technical_bridge" not in categories:
            raise ValueError("Technical bridge allocation must be included explicitly")
        if len(technical_attempts) != 32 or any(
            not row["physical_cost_complete"]
            for row in ledger
            if row["category"] == "technical_bridge_update"
        ):
            raise ValueError("The complete 32-update technical bridge journal is required")
        if result["physical_new_training_updates_started"] < result["logical_new_training_updates"]:
            raise ValueError("Physical training journal is incomplete")
        if result["physical_generation_attempts_started"] < result["logical_committed_generations"]:
            raise ValueError("Physical generation journal is incomplete")
        result.update(
            status="VERIFIED",
            allocated_gpu_hours=gpu_hours,
            reason=None,
            scheduler_jobs=jobs,
            cpu_only_allocations=cpu_jobs,
            source_sha256=file_digest(path),
            raw_files=account["raw_files"],
            failure_history_retained=True,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["reason"] = str(exc)
    return result


def _direction_rows(analysis):
    rows = [
        {
            "endpoint": "H3",
            "family": "primary",
            "expected": "negative",
            "estimate_pp": 100 * analysis["primary"]["estimate"],
            "ci_pp": [100 * x for x in analysis["primary"]["ci95_root_conditional"]],
            "supported": analysis["primary"]["harm_supported"],
        }
    ]
    for name, detail in analysis["secondary_family"]["endpoints"].items():
        rows.append(
            {
                "endpoint": name,
                "family": "secondary_Bonferroni_m3",
                "expected": "positive",
                "estimate_pp": 100 * detail["estimate"],
                "ci_pp": [100 * x for x in detail["ci98_333333_bonferroni_three"]],
                "supported": detail["direction_supported"],
            }
        )
    for name, detail in analysis["noninferiority_family"]["endpoints"].items():
        rows.append(
            {
                "endpoint": name,
                "family": "noninferiority_Bonferroni_m2",
                "expected": "one_sided_lower > -3pp",
                "estimate_pp": 100 * detail["estimate"],
                "lower_pp": 100 * detail["one_sided_97_5_lower"],
                "supported": detail["noninferiority_established"],
            }
        )
    gdiff = analysis["descriptive"]["G_3_3_minus_G_3_2"]
    rows.append(
        {
            "endpoint": "G_3_3_minus_G_3_2",
            "family": "descriptive",
            "expected": "positive",
            "estimate_pp": 100 * gdiff["estimate"],
            "supported": gdiff["ci95_root_conditional"][0] > 0,
        }
    )
    return rows


def _plots(output, analysis, diagnostics):
    """Standard standalone scientific PNG/PDF figures; never join different panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    def save(fig, name):
        for suffix in ("png", "pdf"):
            buffer = io.BytesIO()
            fig.savefig(buffer, format=suffix, dpi=160, bbox_inches="tight")
            path = Path(output) / "figures" / f"{name}.{suffix}"
            _bytes(path, buffer.getvalue())
            paths.append(str(path.relative_to(output)))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cells = [
        row
        for row in analysis["all_cell_metrics"]
        if row["family"] == "cross_series" and (row["center"], row["corrupted_index"]) == (3, 3)
    ]
    for i, arm in enumerate(ARMS):
        values = [
            100 * row["X"]
            for row in cells
            if row["logical_arm_id"] == arm and row["parent"] in PARENTS
        ]
        ax.bar(i, np.mean(values), color="#70a6c2", alpha=0.65)
        ax.scatter([i] * len(values), values, color="#263a5e", s=18)
    parents = [
        100 * row["X"]
        for row in cells
        if row["logical_arm_id"] == "PARENT" and row["parent"] in PARENTS
    ]
    ax.axhline(np.mean(parents), ls="--", color="#b8503d", label="Fixed parents, equal mean")
    ax.set(
        xticks=range(4),
        xticklabels=["A2", "B2", "A3", "B3"],
        ylabel="Raw X (%)",
        title="E_CONFIRM2: c4j4, fixed paths\nDots: six parent/order values",
        ylim=(-2, 103),
    )
    ax.legend(fontsize=8)
    save(fig, "CONFIRM_PRIMARY")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, (left, right), title in zip(
        axes, ((ARMS[1], ARMS[0]), (ARMS[3], ARMS[2])), ("B2 - A2", "B3 - A3"), strict=True
    ):
        matrix = np.zeros((4, 4))
        for row in analysis["all_cell_contrasts"]:
            if (
                row["family"] == "cross_series"
                and row["left"] == left
                and row["right"] == right
                and row["metric"] == "X"
            ):
                matrix[row["center"], row["corrupted_index"]] = row["estimate"] * 100
        im = ax.imshow(matrix, vmin=-100, vmax=100, cmap="RdBu")
        for c in range(4):
            for j in range(4):
                ax.text(j, c, f"{matrix[c, j]:+.1f}", ha="center", va="center", fontsize=8)
        ax.set(
            title=title,
            xticks=range(4),
            xticklabels=["j1", "j2", "j3", "j4"],
            yticks=range(4),
            yticklabels=["c1", "c2", "c3", "c4"],
        )
    fig.colorbar(im, ax=axes.tolist(), label="Raw X contrast (pp)", shrink=0.8)
    fig.suptitle("E_CONFIRM2: all cross cells retained")
    save(fig, "CONFIRM_ALL_CELLS")
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, label in ((-0.18, "H2"), (0.18, "H3")):
        parts = analysis["value_extra_decomposition"][label]
        ax.bar(
            np.arange(3) + offset,
            [parts[m]["estimate"] * 100 for m in ("X", "F_val", "R")],
            width=0.36,
            label=label,
        )
    ax.axhline(0, color="black", lw=0.7)
    ax.set(
        xticks=range(3),
        xticklabels=["X", "F_val", "R"],
        ylabel="B - A (pp)",
        title="E_CONFIRM2 c4j4: X = F_val - R; all-answer denominators",
    )
    ax.legend()
    save(fig, "CONFIRM_VALUE_EXTRA")
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, (left, right), label in (
        (-0.18, (ARMS[1], ARMS[0]), "B2 - A2"),
        (0.18, (ARMS[3], ARMS[2]), "B3 - A3"),
    ):
        edits = {
            row["metric"]: row["estimate"] * 100
            for row in analysis["all_cell_contrasts"]
            if row["family"] == "cross_series"
            and (row["center"], row["corrupted_index"]) == (3, 3)
            and row["left"] == left
            and row["right"] == right
            and row["metric"] in ("q0", "q1", "q2", "q3")
        }
        ax.bar(np.arange(4) + offset, [edits[f"q{k}"] for k in range(4)], width=0.36, label=label)
    ax.axhline(0, color="black", lw=0.7)
    ax.set(
        xticks=range(4),
        xticklabels=["j1", "j2", "j3", "j4"],
        ylabel="Unconditional edit-rate difference (pp)",
        title="E_CONFIRM2 c4j4: q uses every answer, including invalid outputs",
    )
    ax.legend()
    save(fig, "CONFIRM_EDITS")
    trajectory = [
        row
        for row in diagnostics["DEV_TRAJECTORY"]
        if row["row_type"] == "equal_order_parent_effect"
        and row["metric"] == "X"
        and row["status"] == "AVAILABLE"
    ]
    if trajectory:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        for ax, cell in zip(axes.flat, ((3, 3), (3, 1), (3, 2), (2, 2)), strict=True):
            for parent in PARENTS:
                rows = sorted(
                    (
                        r
                        for r in trajectory
                        if r["parent"] == parent and (r["center"], r["corrupted_index"]) == cell
                    ),
                    key=lambda r: r["step"],
                )
                if rows:
                    ax.plot(
                        [r["step"] for r in rows],
                        [r["estimate"] * 100 for r in rows],
                        "o-",
                        label=parent,
                    )
            ax.axhline(0, color="grey", lw=0.5)
            ax.set(
                title=f"c{cell[0] + 1}j{cell[1] + 1}",
                xlabel="Optimizer updates",
                ylabel="B2 - A2 (pp)",
            )
        axes.flat[0].legend()
        fig.suptitle("DEV_TRAJECTORY only: checkpoint training progress, not isolated donor dose")
        fig.tight_layout()
        save(fig, "DEV_TRAJECTORY")
    return paths


def _final_text(analysis, scope, directions):
    h3 = analysis["primary"]
    ci = [value * 100 for value in h3["ci95_root_conditional"]]
    wording = {
        "negative": "固定配置下的相对损害得到支持",
        "positive": "固定配置下的相对改善得到支持",
        "uncertain": "区间跨零，方向不确定",
    }[h3["direction95"]]
    lines = [
        "# SER-J23 阶段报告",
        "",
        f"执行状态：`{scope['status']}`。所有数值效应以 pp（百分点）表示。",
        "",
        "研究问题：把供体从登记的 j2 配置换到 j3 配置后，B3 相对 A3 是否仍损害 c4j4。"
        "污染幅度与历史背景并未完全隔离。",
        "",
        f"主确认 H3 = {h3['estimate'] * 100:+.3f} pp，双侧95%根配对区间 "
        f"[{ci[0]:+.3f}, {ci[1]:+.3f}] pp；{wording}。",
        "H3 的相对损害不自动表示相对父模型的绝对下降，父对照必须另看。",
        "",
    ]
    for name, detail in analysis["parent_contrasts"].items():
        bounds = [100 * value for value in detail["ci95_root_conditional"]]
        lines.append(
            f"- {name}：{100 * detail['estimate']:+.3f} pp，"
            f"95%区间 [{bounds[0]:+.3f}, {bounds[1]:+.3f}]。"
        )
    lines.extend(["", "全部预定义次级与非劣比较均公布，不以主终点结果决定是否展示：", ""])
    for row in directions[1:]:
        lines.append(
            f"- {row['endpoint']}：{row['estimate_pp']:+.3f} pp；家族={row['family']}；"
            f"预定义方向/非劣条件支持={row['supported']}。"
        )
    lines.extend(
        [
            "",
            "三次级采用各98.333333%双侧Bonferroni区间。两个非劣比较采用97.5%单侧下限与−3pp界；未建立非劣不自动等于证实伤害，通过也不等于效应为零。各家族不合并声称统一5%总体错误率。",
            "",
            "F_val 包括其它位置域外时的正确值出现；F_legal 保持原生合法回答口径。X=F_val−R。"
            "q/r分母包含所有回答，c以非X回答为分母，零分母保留null。"
            "原始X是主结果，projection与mixed-anchor输出匹配不证明内部算法。",
            "",
            "异质性：PATH_HETEROGENEITY.csv 保留两个父和三个实际题序全部效应。"
            "区间条件于固定父、题序、训练根和实际训练路径；只重采样数值根，不重抽训练块或8次回答。"
            "经验全零/退化区间不证明真实概率为零。",
            "",
            "开发诊断：FULL_TRAIN_FIT.csv 逐题保留14道中心训练题和32道各臂供体题，K=4，"
            "不能认证整个训练池。TRAIN_NLL_AUDIT.csv 的每题最后NLL来自其最后实际出现的step，"
            "不是统一step256重新评估。DEV_TRAJECTORY仅在同一旧开发面板连接兼容的已注册中间点；"
            "没有补调用。DEV_DELTA保留非法构造和合法掩码，非法构造不算模型I；"
            "没有把不同合法根集合的δ变化解释成同根因果效果。",
            "",
            "与预测不符、负向和不确定结果均见DIRECTION_CHECKS.csv及完整cell表；不删cross:j1，不按992题直接平均宏分。后续机制/保护需要独立设计，本阶段不训练guard、奖励、RL或agent，也不自动开启下一实验。",
            "",
            f"成本状态：`{scope['gpu_accounting_status']}`；实际GPU分配小时={scope['allocated_gpu_hours']}。逻辑剂量与所有物理attempt分开记录；未导入合格sacct时不得以生成数或耗时估算冒充实际分配小时。",
            "",
            "未完成或无法核验的交付项：",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in scope["incomplete_reasons"]] or ["- 无。"])
    return "\n".join(lines) + "\n"


def write_reports(output_dir, scored, *, release_receipt):
    """Write all results only for a verified immutable release, retaining prior report bytes."""
    started = time.perf_counter()
    context = _release_context(output_dir, release_receipt)
    output = Path(output_dir)
    scored = list(scored)
    _verify_scored_against_release(scored, context, release_receipt)
    analysis = analyze_confirmation(
        [row for row in scored if row["panel"] == "E_CONFIRM2"], expected_tasks=context["registry"]
    )
    historical_scores = _historical_trajectory_scores(context)
    all_scored = [*scored, *historical_scores]
    tasks = _enriched_task_rows(all_scored, context)
    diagnostics = _diagnostic_tables(all_scored, tasks, context)
    nll = training_nll_audit(context)
    accounting = gpu_accounting(context["run"], release_receipt)
    directions = _direction_rows(analysis)
    _json(output / "INTEGRITY_REPORT.json", release_receipt)
    _json(
        output / "PRIMARY_H3.json",
        {
            **analysis["primary"],
            "H2": analysis["H2"],
            "parent_contrasts": analysis["parent_contrasts"],
            "bootstrap": analysis["bootstrap"],
            "scope": analysis["scope"],
        },
    )
    _json(
        output / "SECONDARY_FAMILY.json",
        {**analysis["secondary_family"], "other_descriptive": analysis["descriptive"]},
    )
    _json(output / "NONINFERIORITY_C3J3.json", analysis["noninferiority_family"])
    _csv(output / "ALL_CELL_METRICS.csv", analysis["all_cell_metrics"])
    _csv(output / "ALL_CELL_CONTRASTS.csv", analysis["all_cell_contrasts"])
    _json(output / "MACRO_METRICS.json", analysis["macro"])
    _json(output / "HISTORICAL_TRAJECTORY_REUSE.json", context.get("historical_receipt", {}))
    edits, decomposition, heterogeneity = [], [], []
    for row in analysis["all_cell_metrics"]:
        identity = {
            key: row[key]
            for key in ("parent", "block", "logical_arm_id", "family", "center", "corrupted_index")
        }
        n = row["n_answers"]
        for k in range(4):
            edits.append(
                {
                    **identity,
                    "coordinate_0based": k,
                    "denominator_all_answers": n,
                    "denominator_not_X": round(n * (1 - row["X"])),
                    "q_numerator": round(n * row["q"][k]),
                    "r_numerator": round(n * row["r"][k]),
                    "q": row["q"][k],
                    "r": row["r"][k],
                    "c": row["c"][k],
                    "c_undefined_reason": row["c_undefined_reason"],
                }
            )
        decomposition.append(
            {
                **identity,
                "X": row["X"],
                "F_val": row["F_val"],
                "F_legal": row["F_legal"],
                "R": row["R"],
                "denominator_all_answers": n,
            }
        )
    for row in analysis["all_cell_contrasts"]:
        for p, parent in enumerate(PARENTS):
            for b in range(3):
                heterogeneity.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key in ("family", "center", "corrupted_index", "left", "right", "metric")
                    }
                    | {
                        "parent": parent,
                        "block": b,
                        "estimate": row["per_parent_order"][p][b],
                        "estimate_pp": 100 * row["per_parent_order"][p][b],
                        "scope": "fixed_path_description_no_six_sample_population_pvalue",
                    }
                )
    _csv(output / "EDIT_DENOMINATORS.csv", edits)
    _csv(output / "VALUE_VS_EXTRA_ERRORS.csv", decomposition)
    _jsonl(output / "ROOT_LEVEL_METRICS.jsonl", analysis["task_metrics"])
    _csv(output / "PATH_HETEROGENEITY.csv", heterogeneity)
    for panel, filename in (
        ("TRAIN_FIT_FULL", "FULL_TRAIN_FIT.csv"),
        ("DEV_TRAJECTORY", "DEV_TRAJECTORY.csv"),
        ("DEV_DELTA", "DEV_DELTA.csv"),
    ):
        _csv(output / filename, diagnostics[panel])
    _csv(output / "TRAIN_NLL_AUDIT.csv", [*nll["summaries"], *nll["errors"]])
    _csv(output / "TRAIN_NLL_EXPOSURES.csv", nll["exposures"])
    _json(output / "TRAIN_NLL_SOURCE_FILES.json", nll["source_files"])
    _csv(output / "DIRECTION_CHECKS.csv", directions)
    plot_error, plots = None, []
    try:
        plots = _plots(output, analysis, diagnostics)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        plot_error = str(exc)
    incomplete = []
    if accounting["status"] != "VERIFIED":
        incomplete.append("Actual allocation accounting UNVERIFIED: " + str(accounting["reason"]))
    if nll["status"] != "COMPLETE":
        incomplete.append("Late training NLL audit INCOMPLETE; all missing/invalid logs retained")
    if plot_error:
        incomplete.append("Scientific figures unavailable: " + plot_error)
    scope = {
        "status": "STAGE_COMPLETE" if not incomplete else "REPORT_COMPLETE_STAGE_INCOMPLETE",
        "scientific_confirmation_status": analysis["status"],
        "mode": context["mode"],
        "all_registered_scientific_tasks_complete": True,
        "probability_units": "probability_0_to_1",
        "effect_display_units": "percentage_points_pp",
        "gpu_accounting_status": accounting["status"],
        "allocated_gpu_hours": accounting["allocated_gpu_hours"],
        "incomplete_reasons": incomplete,
        "preregistered_unavailable_diagnostics": context["diagnostic_availability"],
        "historical_trajectory_answers_reused": len(historical_scores),
        "historical_reuse_new_model_calls": 0,
        "new_guard_training": 0,
        "new_RL": 0,
        "automatic_successor_experiments": False,
        "figures": plots,
        "analysis_identity_bound_by_registered_queue": True,
        "scientific_direction_does_not_control_completion": True,
    }
    accounting["CPU_analysis_seconds"] = time.perf_counter() - started
    accounting["CPU_analysis_seconds_scope"] = (
        "Report analysis/figure wall time through accounting assembly; excludes final report "
        "serialization and root manifest hashing. Root manifest records its separate CPU/wall time."
    )
    _json(output / "FINAL_GPU_ACCOUNTING.json", accounting)
    _json(output / "EXECUTION_SCOPE.json", scope)
    _bytes(output / "FINAL_REPORT_zh.md", _final_text(analysis, scope, directions).encode())
    manifest = {
        str(path.relative_to(output)): {"sha256": file_digest(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and ".history" not in path.parts
        and path.name != "FILE_MANIFEST_SHA256.json"
    }
    _json(output / "FILE_MANIFEST_SHA256.json", manifest)
    if scope["status"] == "STAGE_COMPLETE":
        from .evaluate import write_once

        write_once(
            context["run"] / "STAGE_COMPLETE.json",
            {
                "status": "STAGE_COMPLETE",
                "release_matrix_digest": release_receipt["matrix_digest"],
                "scientific_analysis_sha256": scientific_digest(analysis),
                "automatic_successor_experiments": False,
            },
        )
    write_root_manifest(context["run"])
    return scope


def release_and_analyze(run):
    """The shared gateway validates integrity, seals the queue, writes receipt, then scores."""
    from .evaluate import release_and_analyze as release_gateway

    return release_gateway(run)
