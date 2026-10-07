"""Import actual VDT provenance, rebuild SER manifests, then freeze CPU contracts."""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path

from .matched_tasks import build_new_tasks
from .schedule import ExplicitScheduleSampler, build_schedule, validate_schedule
from .schema import (
    ARMS,
    EXPERIMENT_ID,
    MICROBATCH_SLOTS,
    PARENTS,
    SCHEDULE_SEEDS,
    canonical_target,
    digest,
    file_digest,
    public_prompt,
    public_verifier,
    read_bound,
    read_json,
    read_jsonl,
    solve_public,
    valid_vector,
    verify_frozen,
    write_json,
    write_jsonl,
)


def _record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path), "bytes": path.stat().st_size}


def _index(rows, key="task_id"):
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate source identity")
    return indexed


def _center(task):
    if task["family"] != "cross_series":
        return None
    candidates = [j for j in range(4) if all(row[j] == 1 for row in task["H_original"])]
    if len(candidates) != 1:
        raise ValueError("Cross star must have exactly one center")
    return candidates[0]


def _public_replay_audit(row):
    task, truth = row["task"], json.loads(row["target"])
    changed = [j for j in range(4) if task["observed"][j] != truth[j]]
    if len(changed) != 1:
        raise ValueError("Replay target is not one repair")
    return dict(
        task_id=task["task_id"],
        root_id=task["root_id"],
        split="REPLAY",
        true_world=truth,
        corrupted_index=changed[0],
        center=_center(task),
        truth_orbit_key=sorted(truth),
    )


def import_legacy(legacy_run_root):
    """Reconstruct and bind all four original SELF_O0 training signatures."""
    from ..verified_discovery_transfer.canonical_targets import build_training_view
    from ..verified_discovery_transfer.public_tasks import (
        RoleDataset as LegacyDataset,
    )
    from ..verified_discovery_transfer.public_tasks import (
        compile_case as legacy_case,
    )

    legacy = Path(legacy_run_root).resolve()
    source_files = []

    def load(relative):
        path = legacy / relative
        source_files.append(_record(path))
        return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)

    frozen, protocol = load("freeze.json"), load("protocol.json")
    if digest(protocol) != frozen["protocol_hash"]:
        raise ValueError("Legacy protocol differs from original freeze")
    source_machine = load("machine.json")
    if digest(source_machine) != frozen["machine_hash"]:
        raise ValueError("Legacy machine differs from original freeze")
    load("cohort/cohort_manifest.json")
    original_tasks = LegacyDataset(legacy / "cohort", "trainer").public("T_train")
    task_index = _index(original_tasks)
    load("cohort/T_train/tasks_public.jsonl")
    audits = _index(load("cohort/T_train/audit_only.jsonl"))
    if set(audits) != set(task_index) or any(row["split"] != "T_train" for row in audits.values()):
        raise ValueError("Original T audit identity/role mismatch")
    replay = LegacyDataset(legacy / "cohort", "trainer").replay()
    load("cohort/R_replay/verified_targets.jsonl")
    replay_public = load("cohort/R_replay/tasks_public.jsonl")
    replay_index = {row["task"]["task_id"]: row for row in replay}
    if len(replay_index) != 64 or set(replay_index) != set(_index(replay_public)):
        raise ValueError("Exactly 64 original verified replay IDs required")
    replay_view = build_training_view(
        [row["task"] for row in replay],
        {row["task"]["task_id"]: {"canonical_vector": row["canonical_vector"]} for row in replay},
        source="replay",
    )

    def view(rows):
        return [
            {
                key: row.get(key)
                for key in ("task_id", "prompt", "target", "target_token_ids", "EOS_id", "weight")
            }
            for row in rows
        ]

    views, receipts = [], []
    for parent in PARENTS:
        for repeat in (0, 1):
            discovery = load(f"evidence/discover.{parent}.{repeat}/discovery.json")
            if discovery["parent_id"] != parent or discovery["pipeline_repeat"] != repeat:
                raise ValueError("Legacy discovery source identity mismatch")
            focus = build_training_view(original_tasks, discovery["J_O0"], source="self")
            identity = dict(
                parent=parent,
                parent_checkpoint=frozen["parent_bindings"][parent],
                repeat=repeat,
                seed=73101 + repeat,
                runtime_bindings=frozen["input_hashes"],
                focus=view(focus),
                replay=view(replay_view),
                sft=protocol["sft"],
            )
            result = load(f"evidence/sft.{parent}.{repeat}.SELF_O0/result.json")
            if digest(identity) != result["training_signature"]:
                raise ValueError("Reconstructed original canonical training view signature differs")
            views.append(_index(focus))
            receipts.append(
                dict(
                    parent=parent,
                    repeat=repeat,
                    focus_count=len(focus),
                    training_signature=digest(identity),
                    original_training_signature_verified=True,
                )
            )
    intersection = set.intersection(*(set(view) for view in views))
    removed = {
        tid
        for tid in intersection
        if audits[tid]["center"] in (0, 2, 3) and audits[tid]["corrupted_index"] == 1
    }
    common_ids = intersection - removed
    if (len(intersection), len(removed), len(common_ids)) != (194, 12, 182):
        raise ValueError("Historical four-view intersection/removal must be 194/12/182")
    common, common_audits = [], []
    for tid in sorted(common_ids):
        targets = {view[tid]["target"] for view in views}
        if len(targets) != 1:
            raise ValueError("Original four canonical training targets disagree")
        target = next(iter(targets))
        if target != canonical_target(audits[tid]["true_world"]):
            raise ValueError("Original training target differs from independent audit")
        task = dict(
            task_index[tid], root_id=task_index[tid]["base_instance_id"], split="COMMON_TRAIN"
        )
        if public_prompt(task) != legacy_case(task_index[tid], "O0")["prompt"]:
            raise ValueError("SER renderer changed an inherited common prompt")
        common.append(
            dict(
                task=task,
                target=target,
                source="intersection_of_four_historical_SELF_O0_focus_views",
                original_task_id=tid,
            )
        )
        common_audits.append(dict(audits[tid], split="COMMON_TRAIN", root_id=task["root_id"]))
    replay_rows = []
    for original in replay_public:
        record = replay_index[original["task_id"]]
        if (
            record["task"] != original
            or record.get("verified") is not True
            or not record.get("source")
        ):
            raise ValueError("Original replay public/verified provenance mismatch")
        task = dict(original, root_id=original["base_instance_id"], split="REPLAY")
        if public_prompt(task) != legacy_case(original, "O0")["prompt"]:
            raise ValueError("SER renderer changed an inherited replay prompt")
        replay_rows.append(
            dict(
                task=task,
                target=canonical_target(record["canonical_vector"]),
                source=record["source"],
            )
        )
    parents = load("parent_availability.json")
    records = {row["checkpoint_id"]: row for row in parents["checkpoints"]}
    if set(records) != set(PARENTS):
        raise ValueError("Exactly original S96 and REP96 parent records required")
    parent_receipts = {}
    for parent, lineage in (("S96", 61001), ("REP96", 61003)):
        row = records[parent]
        if (
            row["source"]["lineage"] != lineage
            or row["source"]["source_step"] != 96
            or row.get("checkpoint") != frozen["parent_bindings"][parent]
        ):
            raise ValueError("Original parent catalogue/freeze binding mismatch")
        checks = {}
        for kind in ("checkpoint", "commit", "manifest"):
            binding = row[kind]
            path = Path(binding["path"])
            if path.is_file():
                if file_digest(path) != binding["sha256"]:
                    raise ValueError("Parent source bytes differ: " + parent + "/" + kind)
                checks[kind] = "VERIFIED_BYTES"
            else:
                checks[kind] = "UNAVAILABLE_LOCAL"
        parent_receipts[parent] = dict(
            checks=checks,
            checkpoint_tensor_content_verified=False,
            inherited_checkpoint=row["checkpoint"],
        )
    provenance = dict(
        legacy_run_root=str(legacy),
        original_run_identity=frozen["run_identity"],
        source_files=source_files,
        original_views=receipts,
        common_intersection_before_removal=194,
        common_removed_donor_cells=12,
        removed_task_ids=sorted(removed),
        original_targets_verified_against_training_signatures=True,
        public_renderer_inherited_match=True,
        parent_bindings=parent_receipts,
    )
    return common, common_audits, replay_rows, parents, source_machine, provenance


def extract_truth_orbits(value):
    """Read only explicit truth/canonical target fields, never observed model answers."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("true_world", "canonical_vector"):
                if not valid_vector(item):
                    raise ValueError("Malformed explicit historical truth vector")
                found.add(tuple(sorted(item)))
            elif key == "truth_orbits":
                if not isinstance(item, list) or any(not valid_vector(vector) for vector in item):
                    raise ValueError("Malformed historical truth orbit index")
                found.update(tuple(sorted(vector)) for vector in item)
            elif isinstance(item, (dict, list)):
                found.update(extract_truth_orbits(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                found.update(extract_truth_orbits(item))
    return found


def scan_history(paths):
    """Explicit source files/directories, recorded exactly; no inferred broad disk scan."""
    files = set()
    for entry in paths:
        path = Path(entry).resolve()
        if not path.exists():
            raise FileNotFoundError("History scan input missing: " + str(path))
        if path.is_file():
            files.add(path)
        else:
            files.update(path.rglob("audit_only.jsonl"))
            files.update(path.rglob("historical_exclusion.json"))
            files.update(path.rglob("verified_targets.jsonl"))
    orbits, records = set(), []
    for path in sorted(files):
        if path.suffix not in (".json", ".jsonl") or path.is_symlink():
            raise ValueError("Explicit history file must be nonsymlink JSON/JSONL")
        value = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
        found = extract_truth_orbits(value)
        orbits.update(found)
        records.append({**_record(path), "truth_orbits": len(found)})
    return orbits, records


def _historical_index(legacy, machine, explicit):
    candidates = (
        [Path(explicit)]
        if explicit is not None
        else [
            Path(machine["historical_metadata_index"]),
            legacy.parent / "project_docs/evidence/historical_inputs/historical_exclusion.json",
            legacy.parent / "inputs/historical_exclusion.json",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Explicit archived historical exclusion index required")


def _jobs():
    training = [
        dict(parent=parent, block=block, block_seed=seed, arm=arm, updates=256)
        for parent in PARENTS
        for block, seed in enumerate(SCHEDULE_SEEDS)
        for arm in ARMS
    ]
    evaluation = []
    for job in training:
        for step in (128, 256):
            evaluation.append(
                dict(job, kind="diagnostic_eval", panel="E_DIAG", step=step, tasks=48, draws=8)
            )
            evaluation.append(
                dict(job, kind="fit_sentinel", panel="TRAIN_FIT", step=step, tasks=16, draws=4)
            )
        evaluation.append(
            dict(job, kind="confirm_eval_sealed", panel="E_CONFIRM", step=256, tasks=800, draws=8)
        )
    for parent in PARENTS:
        for panel, count in (("E_DIAG", 48), ("E_CONFIRM", 800)):
            evaluation.append(
                dict(
                    kind="frozen_parent_eval",
                    parent=parent,
                    arm="PARENT",
                    block=None,
                    step=0,
                    panel=panel,
                    tasks=count,
                    draws=8,
                )
            )
    evaluation.append(
        dict(
            kind="frozen_base_diagnostic",
            parent="BASE_NO_ADAPTER",
            arm="BASE",
            block=None,
            step=0,
            panel="E_DIAG",
            tasks=48,
            draws=8,
        )
    )
    return training, evaluation


def _validate_later_receipt(receipt_path, receipt, known_hashes, additional_count):
    if (
        receipt.get("schema") != "ser-j2-later-history-audit-v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("gpu_calls_before_audit") != 0
        or not receipt.get("local_scanned_roots")
        or receipt.get("new_orbits_after_archive") != additional_count
    ):
        raise ValueError(
            "Later-history receipt is incomplete or disagrees with explicit exclusion scan"
        )
    inventory_name = receipt.get("server_inventory")
    if (
        not isinstance(inventory_name, str)
        or Path(inventory_name).is_absolute()
        or ".." in Path(inventory_name).parts
    ):
        raise ValueError("Later-history receipt needs a relative server inventory binding")
    inventory_path = Path(receipt_path).resolve().parent / inventory_name
    inventory = read_json(inventory_path)
    server_rows = inventory.get("task_manifest_inventory", [])
    if len(server_rows) != receipt.get("server_manifest_count"):
        raise ValueError("Server inventory count differs from later-history receipt")
    rows = receipt.get("local_manifests", []) + server_rows
    if not rows or any(row.get("sha256") not in known_hashes for row in rows):
        raise ValueError(
            "Later receipt contains task manifests absent from explicit hashed history inputs"
        )
    return {"binding": _record(inventory_path), "inventory": inventory}


def _validate_protocol(protocol):
    if protocol["experiment_id"] != EXPERIMENT_ID or protocol["model"]["parents"] != list(PARENTS):
        raise ValueError("Unexpected SER experiment/parents")
    required = dict(
        schedule_seeds=list(SCHEDULE_SEEDS),
        steps=256,
        common_per_step=11,
        donor_per_step=1,
        replay_per_step=4,
        loss_denominator=16,
        microbatch_slots=[list(group) for group in MICROBATCH_SLOTS],
        checkpoint_steps=[0, 64, 128, 192, 256],
    )
    if any(protocol["training"].get(key) != value for key, value in required.items()):
        raise ValueError("The fixed 11/1/4 schedule/loss/checkpoint contract changed")
    if (
        protocol["workload"]["formal_sft_runs"] != 18
        or protocol["workload"]["formal_generation_total"] != 145280
    ):
        raise ValueError("Workload exceeds or differs from registered experiment")
    if any(
        protocol["scope"][key]
        for key in (
            "new_rl",
            "new_discovery",
            "fit_controller",
            "auxiliary_reward",
            "extra_scoring",
        )
    ):
        raise ValueError("Unregistered experiment scope")


def prepare(
    legacy_run_root,
    plan_dir,
    out,
    *,
    later_history_paths=(),
    later_audit_receipt=None,
    historical_index_path=None,
    machine=None,
):
    legacy, plan, run = (
        Path(legacy_run_root).resolve(),
        Path(plan_dir).resolve(),
        Path(out).resolve(),
    )
    if (run / "FROZEN_PLAN.json").exists():
        raise FileExistsError("Frozen run already exists; use cpu-check, never silently refreeze")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Preparation requires an empty destination")
    protocol = read_json(plan / "protocol.json")
    _validate_protocol(protocol)
    common, common_audit, replay, parents, inherited_machine, provenance = import_legacy(legacy)
    machine_value = (
        read_json(machine)
        if isinstance(machine, (str, Path))
        else copy.deepcopy(machine or inherited_machine)
    )
    history = _historical_index(legacy, inherited_machine, historical_index_path)
    baseline_orbits, baseline_records = scan_history([history, legacy / "cohort"])
    if len(baseline_orbits) != 4219:
        raise ValueError("Archived historical baseline must reproduce the 4219 truth orbits")
    additional_orbits, later_records = scan_history(later_history_paths)
    if later_audit_receipt is not None:
        receipt = read_json(later_audit_receipt)
        later_receipt_binding = _record(later_audit_receipt)
    else:
        receipt, later_receipt_binding = None, None
    later_verified = bool(later_history_paths and receipt is not None)
    inventory_receipt = None
    if receipt is not None:
        known_hashes = {
            row["sha256"] for row in provenance["source_files"] + baseline_records + later_records
        }
        # Public evaluation task files contain no truth, but their bytes also bind the scan.
        known_hashes.update(
            file_digest(path) for path in (legacy / "cohort").rglob("tasks_public.jsonl")
        )
        inventory_receipt = _validate_later_receipt(
            later_audit_receipt, receipt, known_hashes, len(additional_orbits - baseline_orbits)
        )
    files, generated_report = build_new_tasks(baseline_orbits | additional_orbits)
    files.update(
        {
            "common_targets.jsonl": common,
            "common_audit.jsonl": common_audit,
            "replay_targets.jsonl": replay,
            "replay_audit.jsonl": [_public_replay_audit(row) for row in replay],
        }
    )
    donor_targets = files["donor_targets.jsonl"]
    donor_roots = list(dict.fromkeys(row["root_id"] for row in donor_targets))
    donor_ids = {(row["root_id"], row["arm"]): row["task_id"] for row in donor_targets}
    for seed in SCHEDULE_SEEDS:
        files[f"schedule_{seed}.jsonl"] = build_schedule(
            [row["task"]["task_id"] for row in common],
            [row["task"]["task_id"] for row in replay],
            donor_roots,
            donor_ids,
            seed,
        )
    training, evaluation = _jobs()
    files.update({"training_jobs.jsonl": training, "evaluation_jobs.jsonl": evaluation})
    rng = random.Random(107074)
    sentinels = dict(
        common=rng.sample(sorted(row["task"]["task_id"] for row in common), 8),
        replay=rng.sample([row["task"]["task_id"] for row in replay], 4),
        donor_roots=rng.sample(donor_roots, 4),
    )
    comparison = {}
    for filename, rows in files.items():
        reference = plan / "manifests" / filename
        if not reference.is_file():
            comparison[filename] = "ADDED_AUDIT_METADATA"
            continue
        original = read_jsonl(reference)
        normalized = (
            [{key: value for key, value in row.items() if key != "root_id"} for row in rows]
            if filename == "common_audit.jsonl"
            else rows
        )
        comparison[filename] = "EXACT_RECORD_MATCH" if normalized == original else "DIFFERS"
    comparison["fit_sentinels.json"] = (
        "EXACT_RECORD_MATCH"
        if sentinels == read_json(plan / "manifests/fit_sentinels.json")
        else "DIFFERS"
    )
    differences = [filename for filename, status in comparison.items() if status == "DIFFERS"]
    new_exclusions = additional_orbits - baseline_orbits
    if differences and not new_exclusions:
        raise ValueError(
            "Unexplained difference from supplied reference manifests: " + ",".join(differences)
        )
    # With extra historical roots only deterministic root/schedule changes are permitted.
    for fixed in (
        "common_targets.jsonl",
        "common_audit.jsonl",
        "replay_targets.jsonl",
        "training_jobs.jsonl",
        "evaluation_jobs.jsonl",
    ):
        if comparison[fixed] != "EXACT_RECORD_MATCH":
            raise ValueError("Extra root exclusion changed fixed legacy/job records")
    run.mkdir(parents=True, exist_ok=True)
    for filename, rows in files.items():
        write_jsonl(run / "manifests" / filename, rows)
    write_json(run / "manifests/fit_sentinels.json", sentinels)
    exclusion = dict(
        status="VERIFIED_EXPLICIT_SCAN" if later_verified else "NOT_VERIFIED",
        baseline=baseline_records,
        explicit_later_scan_paths=[str(Path(path).resolve()) for path in later_history_paths],
        later_files=later_records,
        external_scan_receipt=later_receipt_binding,
        additional_unique_orbits=len(new_exclusions),
        final_exclusion_orbits=len(baseline_orbits | additional_orbits),
        regeneration=(
            "fixed seeds; sequential sampling skips all historical collisions before freeze"
        ),
        limitations=(
            "Only explicit scan scope and recorded receipt were audited; no model outcomes used"
        ),
    )
    provenance["historical_exclusion"] = exclusion
    write_json(run / "SOURCE_PROVENANCE.json", provenance)
    write_json(
        run / "LATER_HISTORY_AUDIT.json",
        {"scan": exclusion, "external_receipt": receipt, "server_inventory": inventory_receipt},
    )
    write_json(run / "MANIFEST_COMPARISON.json", comparison)
    write_json(run / "protocol.json", protocol)
    write_json(run / "machine.json", machine_value)
    write_json(run / "parent_availability.json", parents)
    report = {
        **generated_report,
        "common_tasks": 182,
        "replay_tasks": 64,
        "donor_roots": 32,
        "donor_variants": 96,
        "diagnostic_tasks": 48,
        "confirm_tasks": 800,
        "common_intersection_before_removal": 194,
        "common_removed_donor_cells": 12,
        "reused_unique_repair_checks": 246,
        "workload": protocol["workload"],
        "later_exclusion_status": exclusion["status"],
        "changed_manifest_files": differences,
        "checkpoint_tensor_content_verified": False,
        "model_calls": 0,
    }
    write_json(run / "manifests/BUILD_REPORT.json", report)
    external = {}
    for key in ("runtime_path", "frozen_protocol_path", "checkpoint_catalog"):
        path = Path(machine_value[key])
        external[key] = (
            {**_record(path), "status": "VERIFIED_BYTES"}
            if path.is_file()
            else {"path": str(path), "sha256": None, "status": "UNAVAILABLE_LOCAL"}
        )
    frozen = dict(
        schema="SER-J2-frozen-plan-v1",
        status="FROZEN",
        experiment_id=EXPERIMENT_ID,
        files={
            str(path.relative_to(run)): {"sha256": file_digest(path)}
            for path in sorted(run.rglob("*"))
            if path.is_file()
        },
        external_inputs=external,
        later_exclusion_status=exclusion["status"],
        checkpoint_tensors_verified=False,
        source_protocol_sha256=file_digest(plan / "protocol.json"),
    )
    frozen["plan_hash"] = digest(frozen)
    write_json(run / "FROZEN_PLAN.json", frozen)
    checked = cpu_check(run)
    write_json(run / "CPU_CHECK.json", checked)
    return {
        "status": "CPU_INTEGRATION_PASS",
        "run": str(run),
        "plan_hash": frozen["plan_hash"],
        "later_exclusion_status": exclusion["status"],
        "checkpoint_tensors_verified": False,
        "model_calls": 0,
        "report": report,
        "cpu_check": checked,
    }


def cpu_check(run):
    run = Path(run).resolve()
    frozen = verify_frozen(run)
    protocol = read_bound(run, "protocol.json")
    _validate_protocol(protocol)
    source = read_bound(run, "SOURCE_PROVENANCE.json")
    if (
        source.get("original_targets_verified_against_training_signatures") is not True
        or len(source["original_views"]) != 4
    ):
        raise ValueError("Missing original training target provenance")

    def read(relative):
        return read_bound(run, "manifests/" + relative)

    common, replay = read("common_targets.jsonl"), read("replay_targets.jsonl")
    donors, targets, da = (
        read("donors_public.jsonl"),
        read("donor_targets.jsonl"),
        read("donors_audit.jsonl"),
    )
    diag, dia = read("E_DIAG/tasks_public.jsonl"), read("E_DIAG/audit_only.jsonl")
    confirm, ca = read("E_CONFIRM/tasks_public.jsonl"), read("E_CONFIRM/audit_only.jsonl")
    if tuple(map(len, (common, replay, donors, diag, confirm))) != (182, 64, 96, 48, 800):
        raise ValueError("Frozen task counts changed")
    for row in common + replay:
        target = json.loads(row["target"])
        if canonical_target(target) != row["target"] or solve_public(row["task"]) != [target]:
            raise ValueError("Reused canonical target is not the unique public repair")
    audit = _index(da + dia + ca)
    all_new = donors + diag + confirm
    if len(_index(all_new)) != 944 or set(audit) != {task["task_id"] for task in all_new}:
        raise ValueError("New public/audit task identities differ")
    for task in all_new:
        if solve_public(task) != [audit[task["task_id"]]["true_world"]]:
            raise ValueError("New task is not a unique public repair")
    roots = read("new_root_registry_AUDIT_ONLY.jsonl")
    if len(roots) != 274 or len({tuple(sorted(row["true_world"])) for row in roots}) != 274:
        raise ValueError("New root/orbit crosses splits or is reused")
    ri = _index(roots, "root_id")
    for task in all_new:
        root = ri[task["root_id"]]
        if (
            root["split"] != task["split"]
            or root["true_world"] != audit[task["task_id"]]["true_world"]
        ):
            raise ValueError("Root split/true-world mismatch")
    donor_tasks = _index(donors)
    by_root = {}
    for row in targets:
        by_root.setdefault(row["root_id"], []).append(row)
    if len(by_root) != 32:
        raise ValueError("Donor root count mismatch")
    for matched in by_root.values():
        if (
            {row["arm"] for row in matched} != set(ARMS)
            or len(matched) != 3
            or len({row["target"] for row in matched}) != 1
        ):
            raise ValueError("Donor arm/target matching failed")
        tasks = [donor_tasks[row["task_id"]] for row in matched]
        if (
            len(
                {
                    digest([task["observed"], task["operation"], task["chart_type"]])
                    for task in tasks
                }
            )
            != 1
        ):
            raise ValueError("Donor observations/operation differ")
        for row, task in zip(matched, tasks, strict=True):
            truth = json.loads(row["target"])
            if (
                _center(task) != ARMS[row["arm"]]
                or audit[task["task_id"]]["corrupted_index"] != 1
                or canonical_target(truth) != row["target"]
                or not public_verifier(task, truth)
            ):
                raise ValueError(
                    "Donor must vary only center at fixed j2 and correct canonical target"
                )
    public_prompts = read("new_prompts_public.jsonl")
    if public_prompts != [
        dict(task_id=task["task_id"], split=task["split"], prompt=public_prompt(task))
        for task in all_new
    ]:
        raise ValueError("Frozen prompt text differs from inherited renderer")
    signatures = set()
    for seed in SCHEDULE_SEEDS:
        rows = read(f"schedule_{seed}.jsonl")
        validate_schedule(rows)
        regenerated = build_schedule(
            [row["task"]["task_id"] for row in common],
            [row["task"]["task_id"] for row in replay],
            list(by_root),
            {(row["root_id"], row["arm"]): row["task_id"] for row in targets},
            seed,
        )
        if rows != regenerated:
            raise ValueError("Schedule differs from frozen seed construction")
        signatures.add(digest(rows))
        sampler = ExplicitScheduleSampler(rows, "A_LOCAL_C1")
        for update in range(1, 5):
            sampler.commit(update)
        recovered = ExplicitScheduleSampler(rows, "A_LOCAL_C1")
        recovered.load_state_dict(sampler.state_dict())
        if recovered.peek() != sampler.peek():
            raise ValueError("Explicit cursor restore failed")
    if len(signatures) != 3:
        raise ValueError("Three actual different input schedules required")
    training, evaluation = _jobs()
    if read("training_jobs.jsonl") != training or read("evaluation_jobs.jsonl") != evaluation:
        raise ValueError("Registered full matrix changed")
    if len(training) != 18 or sum(row["tasks"] * row["draws"] for row in evaluation) != 145280:
        raise ValueError("Formal run/generation scope changed")
    cells = Counter(
        (row["center"], row["corrupted_index"]) for row in ca if row["center"] is not None
    )
    if any(
        cells[(h, j)] != (128 if (h, j) in ((2, 2), (3, 3)) else 32)
        for h in range(4)
        for j in range(4)
    ):
        raise ValueError("Confirmation cell support changed")
    return dict(
        status="PASS",
        execution_kind="CPU_DATA_INTEGRATION_ONLY",
        plan_hash=frozen["plan_hash"],
        original_training_views=4,
        common_intersection=194,
        common_removed=12,
        common_tasks=182,
        unique_public_repairs_new=944,
        unique_public_repairs_reused=246,
        donor_roots=32,
        root_split_disjoint=True,
        canonical_target_matching=True,
        inherited_prompt_match=True,
        schedule_count=3,
        formal_sft_jobs=18,
        updates=4608,
        formal_generated_answers=145280,
        later_exclusion_status=frozen["later_exclusion_status"],
        gpu_execution=False,
        token_EOS_equality="REQUIRES_TOKENIZER_CHECK_IN_GPU_BRIDGE",
        checkpoint_tensors_verified=False,
    )
