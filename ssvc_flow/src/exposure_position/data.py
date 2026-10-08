"""Auditable CPU-only construction from the exact J2 inputs.

Builders return data without writing or freezing a run. Confirmation construction
requires an explicit auditor call and a complete history exclusion receipt.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path

from ..exposure_substitution.matched_tasks import make_task as legacy_make_task
from ..exposure_substitution.schema import public_prompt as legacy_prompt
from ..exposure_substitution.schema import read_bound as read_legacy_bound
from ..exposure_substitution.schema import validate_public as validate_legacy_public
from .schedule import inherit_schedule, validate_schedule
from .schema import (
    ARMS,
    EXPERIMENT_ID,
    SCHEDULE_SEEDS,
    TRAINING_PATHS,
    canonical_target,
    digest,
    file_digest,
    public_prompt,
    read_json,
    read_jsonl,
    solve_public,
    stable_id,
    valid_vector,
    validate_public,
    write_json,
    write_jsonl,
)


def _index(rows, key="task_id"):
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate source identity")
    return indexed


def _loader(source, records):
    source = Path(source).resolve()

    def load(relative):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise PermissionError("Unsafe source path")
        path = source / rel
        if not path.resolve().is_relative_to(source) or any(
            p.is_symlink() for p in (path, *path.parents) if p != source and source in p.parents
        ):
            raise PermissionError("Symlinked or escaped source path")
        records.append(
            {
                "path": str(path),
                "relative_path": relative,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
        )
        if (source / "FROZEN_PLAN.json").is_file():
            return read_legacy_bound(source, relative)
        return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)

    return load


def _center(task):
    if task["family"] != "cross_series":
        return None
    choices = [j for j in range(4) if all(row[j] == 1 for row in task["H_original"])]
    if len(choices) != 1:
        raise ValueError("A cross task needs exactly one star center")
    return choices[0]


def _audit(task, truth, **extra):
    changed = [j for j in range(4) if truth[j] != task["observed"][j]]
    if len(changed) != 1 or solve_public(task) != [truth]:
        raise ValueError("Target must be the unique public single repair")
    center = _center(task)
    return dict(
        phase_id=EXPERIMENT_ID,
        task_id=task["task_id"],
        root_id=task["root_id"],
        split=task["split"],
        true_world=list(truth),
        truth_orbit_key=sorted(truth),
        center_0based=center,
        corrupted_index_0based=changed[0],
        center=center,
        corrupted_index=changed[0],
        **extra,
    )


def _renamed(task, split=None, *, namespace="inherited", root_id=None):
    result = copy.deepcopy(task)
    result.pop("root_cohort", None)
    result.update(
        phase_id=EXPERIMENT_ID,
        split=split or task["split"],
        task_id=stable_id(namespace, split or task["split"], task["task_id"]),
    )
    result["root_id"] = result["base_instance_id"] = root_id or stable_id(
        "source-root", task["root_id"]
    )
    validate_public(result)
    return result


def make_task(
    world,
    observed,
    center,
    corrupted_index,
    root_id,
    split,
    ordinal,
    family="cross_series",
    *,
    identity_extra=None,
):
    """Keep the original relationship/operation/format generator's exact rules."""
    original, _ = legacy_make_task(
        world, observed, center, corrupted_index, root_id, "DONOR_TRAIN", ordinal, family
    )
    task = _renamed(original, split, namespace="generated", root_id=root_id)
    task["task_id"] = stable_id(
        "task", root_id, split, family, center, corrupted_index, identity_extra
    )
    validate_public(task)
    if public_prompt(task) != legacy_prompt(original):
        raise ValueError("Inherited O0 prompt renderer changed")
    return task, _audit(task, list(world))


def verify_legacy_source_code(legacy_source):
    """Bind the old pure builders/renderers actually imported by the J23 code."""
    old, current = Path(legacy_source).resolve(), Path(__file__).resolve().parents[2]
    names = (
        "src/exposure_substitution/schema.py",
        "src/exposure_substitution/matched_tasks.py",
        "src/exposure_substitution/schedule.py",
        "src/prompts.py",
        "src/protocol_state_probes/transforms.py",
    )
    rows = []
    for relative in names:
        a, b = old / relative, current / relative
        rows.append(
            dict(
                relative_path=relative,
                historical_path=str(a),
                current_path=str(b),
                historical_sha256=file_digest(a),
                current_sha256=file_digest(b),
            )
        )
    if any(row["historical_sha256"] != row["current_sha256"] for row in rows):
        raise ValueError("Imported legacy generator/renderer differs from native archived source")
    return dict(status="EXACT_NATIVE_SOURCE_MATCH", files=rows)


def build_training_manifests(legacy_run, *, legacy_source=None):
    """Read exact old pools and slot files; do not recompute the old common filter."""
    records = []
    candidate = Path(legacy_run).resolve().parent / "code_v1" / "ssvc_flow"
    if legacy_source is None and candidate.is_dir():
        legacy_source = candidate
    source_code = (
        verify_legacy_source_code(legacy_source)
        if legacy_source is not None
        else {"status": "NATIVE_SOURCE_NOT_BOUND", "files": []}
    )
    load = _loader(legacy_run, records)
    common = load("manifests/common_targets.jsonl")
    replay = load("manifests/replay_targets.jsonl")
    roots = [
        r
        for r in load("manifests/new_root_registry_AUDIT_ONLY.jsonl")
        if r["split"] == "DONOR_TRAIN"
    ]
    old_donors = _index(load("manifests/donors_public.jsonl"))
    old_targets = load("manifests/donor_targets.jsonl")
    source_donors = {(r["root_id"], r["arm"]): r for r in old_targets}
    if (len(common), len(replay), len(roots), len(source_donors), len(old_donors)) != (
        182,
        64,
        32,
        96,
        96,
    ):
        raise ValueError("Exact historical 182/64/32 pools and 96 donor variants required")
    _index(roots, "root_id")
    _index([r["task"] for r in common + replay])
    task_map, root_map, files, mappings = {}, {}, {}, []
    for name, pool, split in (("common", common, "COMMON_TRAIN"), ("replay", replay, "REPLAY")):
        out, audits = [], []
        for original in pool:
            task = original["task"]
            validate_legacy_public(task)
            if task["split"] != split:
                raise ValueError("Historical training role mismatch")
            target = json.loads(original["target"])
            if canonical_target(target) != original["target"]:
                raise ValueError("Historical target is not canonical")
            new = _renamed(task)
            if public_prompt(new) != legacy_prompt(task):
                raise ValueError("Inherited common/replay prompt changed")
            task_map[task["task_id"]], root_map[task["root_id"]] = new["task_id"], new["root_id"]
            row = copy.deepcopy(original)
            row.update(task=new, source_task_id=task["task_id"], source_root_id=task["root_id"])
            out.append(row)
            audits.append(
                _audit(new, target, source_task_id=task["task_id"], source_root_id=task["root_id"])
            )
            mappings.append(
                dict(
                    task_id=new["task_id"],
                    root_id=new["root_id"],
                    source_task_id=task["task_id"],
                    source_root_id=task["root_id"],
                    logical_arm_id=None,
                    role=name,
                    prompt_sha256=digest(public_prompt(new)),
                    target_sha256=digest(original["target"]),
                )
            )
        files[name + "_targets.jsonl"], files[name + "_audit.jsonl"] = out, audits
    donors, targets, audits, registered_roots, donor_ids = [], [], [], [], {}
    for ordinal, root in enumerate(roots):
        world, replacements = root["true_world"], root["replacements"]
        if (
            root["family"] != "cross_series"
            or not valid_vector(world)
            or len(set(world)) != 4
            or not valid_vector(replacements)
            or any(a == b for a, b in zip(world, replacements, strict=True))
        ):
            raise ValueError(
                "Historical donor truth/replacements invalid; no clipping or replacement allowed"
            )
        rid = root_map.setdefault(root["root_id"], stable_id("source-root", root["root_id"]))
        registered_roots.append(
            dict(
                phase_id=EXPERIMENT_ID,
                root_id=rid,
                source_root_id=root["root_id"],
                source_ordinal_0based=ordinal,
                split="DONOR_TRAIN",
                family="cross_series",
                true_world=list(world),
                replacements=list(replacements),
            )
        )
        for arm, spec in ARMS.items():
            center, j = spec["center_0based"], spec["corrupted_index_0based"]
            observed = list(world)
            observed[j] = replacements[j]
            task, audit = make_task(world, observed, center, j, rid, "DONOR_TRAIN", ordinal)
            source_arm = "A_LOCAL_C1" if center == 0 else "B_FORWARD_C4"
            source_target = source_donors[(root["root_id"], source_arm)]
            source_task = old_donors[source_target["task_id"]]
            expected_old, _ = legacy_make_task(
                world,
                [world[0], replacements[1], world[2], world[3]],
                center,
                1,
                root["root_id"],
                "DONOR_TRAIN",
                ordinal,
            )
            if source_task != expected_old or source_target["target"] != canonical_target(world):
                raise ValueError(
                    "Historical donor identity/order/target differs from original generator"
                )
            if j == 1 and public_prompt(task) != legacy_prompt(source_task):
                raise ValueError("J2 prompt changed")
            audit.update(
                logical_arm_id=arm,
                source_root_id=root["root_id"],
                source_task_id=source_task["task_id"],
                source_ordinal_0based=ordinal,
                signed_delta=observed[j] - world[j],
            )
            donors.append(task)
            audits.append(audit)
            targets.append(
                dict(
                    logical_arm_id=arm,
                    arm=arm,
                    task_id=task["task_id"],
                    root_id=rid,
                    target=canonical_target(world),
                    source_root_id=root["root_id"],
                    source_task_id=source_task["task_id"],
                    label_source="inherited_donor_root_gold_for_registered_intervention",
                )
            )
            mappings.append(
                dict(
                    task_id=task["task_id"],
                    root_id=rid,
                    source_task_id=source_task["task_id"],
                    source_root_id=root["root_id"],
                    logical_arm_id=arm,
                    role="donor",
                    source_ordinal_0based=ordinal,
                    source_corrupted_index_0based=1,
                    corrupted_index_0based=j,
                    prompt_sha256=digest(public_prompt(task)),
                    target_sha256=digest(canonical_target(world)),
                )
            )
            donor_ids[(rid, arm)] = task["task_id"]
    files.update(
        {
            "donors_public.jsonl": donors,
            "donor_targets.jsonl": targets,
            "donors_audit.jsonl": audits,
            "donor_root_registry_AUDIT_ONLY.jsonl": registered_roots,
            "source_mapping.jsonl": mappings,
        }
    )
    exposure_checks = []
    common_c4j3 = {
        r["task_id"]
        for r in files["common_audit.jsonl"]
        if (r["center"], r["corrupted_index"]) == (3, 2)
    }
    if len(common_c4j3) != 1:
        raise ValueError("Historical common c4j3 single task must be preserved")
    for seed in SCHEDULE_SEEDS:
        original = load(f"manifests/schedule_{seed}.jsonl")
        rows = inherit_schedule(original, task_map, root_map, donor_ids)
        files[f"schedule_{seed}.jsonl"] = rows
        new_index = validate_schedule(rows)
        original_a = {(r["update"], r["slot"]): r for r in original if r["arm"] == "A_LOCAL_C1"}
        for arm in ARMS:
            for key, source in original_a.items():
                new = new_index[(arm, *key)]
                if source["role"] != "donor" and task_map[source["task_id"]] != new["task_id"]:
                    raise ValueError("Inherited common/replay slot changed")
        count = sum(r["task_id"] in common_c4j3 for r in rows if r["arm"] == "A2_LOCAL_C1_J2")
        exposure_checks.append(
            {
                "seed": seed,
                "c4j3_common_exposures_per_arm": count,
                "all_15_background_slots_identical": True,
                "donor_root_order_identical": True,
                "legacy_schedule_sha256": records[-1]["sha256"],
                "new_schedule_digest": digest(rows),
            }
        )
    if [r["c4j3_common_exposures_per_arm"] for r in exposure_checks] != [15, 16, 16]:
        raise ValueError("Historical c4j3 exposure must remain 15/16/16")
    report = dict(
        status="CPU_TRAINING_DATA_PASS",
        phase_id=EXPERIMENT_ID,
        legacy_run=str(Path(legacy_run).resolve()),
        source_files=records,
        source_code=source_code,
        j23_data_code={
            p.name: file_digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("schema.py"),
                Path(__file__).with_name("schedule.py"),
            )
        },
        common_tasks=182,
        replay_tasks=64,
        donor_roots=32,
        donor_variants=128,
        schedule_proofs=exposure_checks,
        inherited_prompt_target_equality=True,
        common_filter_reapplied=False,
        unique_public_repairs=374,
        model_calls=0,
        target_token_equality="REQUIRES_BOUND_TOKENIZER_BRIDGE",
    )
    return files, report


def write_training_manifests(run, files, audit):
    """Commit CPU training inputs once, after the caller has written machine.json."""
    run = Path(run)
    if not (run / "machine.json").is_file():
        raise ValueError("Write explicit machine.json before binding training sources")
    if (run / "SOURCE_BOUND.json").exists() or (run / "FROZEN_PLAN.json").exists():
        raise FileExistsError("Source-bound/frozen run cannot be overwritten")
    if any((run / "manifests" / relative).exists() for relative in files):
        raise FileExistsError("Existing manifests cannot be overwritten")
    for relative in files:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] == "E_CONFIRM2":
            raise PermissionError("Training writer cannot write confirmation or unsafe paths")
    for relative, rows in files.items():
        path = run / "manifests" / relative
        (write_jsonl if path.suffix == ".jsonl" else write_json)(path, rows)
    write_json(run / "TRAINING_DATA_AUDIT.json", audit)
    bound = dict(
        schema="ser-j23-source-bound-v1",
        status="SOURCE_BOUND",
        phase_id=EXPERIMENT_ID,
        files={
            relative: {"sha256": file_digest(run / relative)} for relative in sorted(TRAINING_PATHS)
        },
        training_data_audit_sha256=file_digest(run / "TRAINING_DATA_AUDIT.json"),
    )
    bound["source_hash"] = digest(bound)
    write_json(run / "SOURCE_BOUND.json", bound)
    return bound


def build_development_manifests(legacy_run, training_files):
    """Build fixed developer panels; no historical answers are imported here."""
    records = []
    load = _loader(legacy_run, records)
    registry = load("manifests/new_root_registry_AUDIT_ONLY.jsonl")
    old_public = _index(load("manifests/E_CONFIRM/tasks_public.jsonl"))
    old_audits = load("manifests/E_CONFIRM/audit_only.jsonl")
    cross_roots = [
        r for r in registry if r["split"] == "E_CONFIRM" and r["family"] == "cross_series"
    ]
    if len(cross_roots) != 128:
        raise ValueError("Exactly 128 legacy cross confirmation roots required")
    original_by_cell = {
        (a["root_id"], a["center"], a["corrupted_index"]): a
        for a in old_audits
        if a["center"] is not None
    }
    cells = ((3, 3), (3, 1), (3, 2), (2, 2))
    files, selections, source_mapping = {}, {}, []
    trajectory, trajectory_audit = [], []
    for root in cross_roots[:32]:
        for center, j in cells:
            audit = original_by_cell[(root["root_id"], center, j)]
            old = old_public[audit["task_id"]]
            if old.get("root_cohort") != "CORE32":
                raise ValueError("Legacy CORE32 registry order changed")
            task = _renamed(old, "DEV_TRAJECTORY", namespace="development")
            trajectory.append(task)
            trajectory_audit.append(
                _audit(
                    task,
                    audit["true_world"],
                    source_task_id=old["task_id"],
                    source_root_id=root["root_id"],
                )
            )
            source_mapping.append(
                dict(
                    panel="DEV_TRAJECTORY",
                    task_id=task["task_id"],
                    root_id=task["root_id"],
                    source_task_id=old["task_id"],
                    source_root_id=root["root_id"],
                )
            )
            if public_prompt(task) != legacy_prompt(old):
                raise ValueError("Trajectory prompt changed")
    files.update(
        {
            "DEV_TRAJECTORY/tasks_public.jsonl": trajectory,
            "DEV_TRAJECTORY/audit_only.jsonl": trajectory_audit,
        }
    )
    selections["DEV_TRAJECTORY"] = {"task_ids": [t["task_id"] for t in trajectory]}

    all_training = {
        r["task"]["task_id"]: r["task"]
        for name in ("common_targets.jsonl", "replay_targets.jsonl")
        for r in training_files[name]
    }
    all_training.update({t["task_id"]: t for t in training_files["donors_public.jsonl"]})
    shared = [
        r
        for name in ("common_audit.jsonl", "replay_audit.jsonl")
        for r in training_files[name]
        if (r["center"], r["corrupted_index"]) == (3, 3)
    ]
    if len(shared) != 14:
        raise ValueError("Full fit must preserve 14 c4j4 training tasks")
    fit, fit_audit, fit_ids = [], [], {}
    for audit in shared + training_files["donors_audit.jsonl"]:
        original = all_training[audit["task_id"]]
        task = _renamed(
            original, "TRAIN_FIT_FULL", namespace="train-fit", root_id=original["root_id"]
        )
        fit.append(task)
        fit_ids[original["task_id"]] = task["task_id"]
        fit_audit.append(
            _audit(
                task,
                audit["true_world"],
                source_training_task_id=original["task_id"],
                source_task_id=audit["source_task_id"],
                source_root_id=audit["source_root_id"],
            )
        )
        source_mapping.append(
            dict(
                panel="TRAIN_FIT_FULL",
                task_id=task["task_id"],
                root_id=task["root_id"],
                source_training_task_id=original["task_id"],
                source_task_id=audit["source_task_id"],
                source_root_id=audit["source_root_id"],
            )
        )
    by_arm = {
        arm: [fit_ids[a["task_id"]] for a in shared]
        + [
            fit_ids[a["task_id"]]
            for a in training_files["donors_audit.jsonl"]
            if a["logical_arm_id"] == arm
        ]
        for arm in ARMS
    }
    if any(len(ids) != 46 for ids in by_arm.values()) or len(fit) != 142:
        raise ValueError("Full fit must have 46 registered tasks per arm")
    files.update(
        {"TRAIN_FIT_FULL/tasks_public.jsonl": fit, "TRAIN_FIT_FULL/audit_only.jsonl": fit_audit}
    )
    selections["TRAIN_FIT_FULL"] = {"by_arm": by_arm}

    delta_tasks, delta_audits, attempts, by_cell = [], [], [], {"c4j4": [], "c3j3": []}
    for ordinal, root in enumerate(cross_roots[:16]):
        world = root["true_world"]
        rid = stable_id("source-root", root["root_id"])
        for center, j in ((3, 3), (2, 2)):
            cell = f"c{center + 1}j{j + 1}"
            source_id = original_by_cell[(root["root_id"], center, j)]["task_id"]
            for delta in (-40, -20, -10, -5, 5, 10, 20, 40):
                attempt = dict(
                    source_root_id=root["root_id"],
                    source_ordinal_0based=ordinal,
                    cell=cell,
                    signed_delta=delta,
                )
                if not 0 <= world[j] + delta <= 99:
                    attempts.append(dict(attempt, status="REJECTED_OUT_OF_DOMAIN"))
                    continue
                observed = list(world)
                observed[j] += delta
                task, audit = make_task(
                    world, observed, center, j, rid, "DEV_DELTA", ordinal, identity_extra=delta
                )
                audit.update(
                    source_root_id=root["root_id"],
                    source_task_id=source_id,
                    signed_delta=delta,
                    source_ordinal_0based=ordinal,
                )
                delta_tasks.append(task)
                delta_audits.append(audit)
                by_cell[cell].append(task["task_id"])
                attempts.append(
                    dict(attempt, status="LEGAL_UNIQUE_PUBLIC_REPAIR", task_id=task["task_id"])
                )
                source_mapping.append(
                    dict(
                        panel="DEV_DELTA",
                        task_id=task["task_id"],
                        root_id=rid,
                        source_task_id=source_id,
                        source_root_id=root["root_id"],
                    )
                )
    if (len(by_cell["c4j4"]), len(by_cell["c3j3"])) != (106, 109):
        raise ValueError("Fixed legacy delta legality must be 106 c4j4 / 109 c3j3")
    files.update(
        {
            "DEV_DELTA/tasks_public.jsonl": delta_tasks,
            "DEV_DELTA/audit_only.jsonl": delta_audits,
            "DEV_DELTA/legality_attempts.jsonl": attempts,
            "development_source_mapping.jsonl": source_mapping,
        }
    )
    selections["DEV_DELTA"] = {"by_cell": by_cell}
    files["panel_selection.json"] = selections
    return files, dict(
        status="CPU_DEVELOPMENT_DATA_PASS",
        phase_id=EXPERIMENT_ID,
        trajectory_tasks=128,
        fit_tasks_per_arm=46,
        delta_tasks_by_cell={k: len(v) for k, v in by_cell.items()},
        delta_out_of_domain_rejections=256 - len(delta_tasks),
        source_files=records,
        model_calls=0,
        legacy_endpoint_answer_reuse="REQUIRES_MODEL_AND_INFERENCE_FINGERPRINT_COMPATIBILITY",
    )


def _validate_exclusion_receipt(exclusion_orbits, receipt):
    from .exclusions import verify_exclusion_audit

    try:
        verified = verify_exclusion_audit(receipt, verify_sources=True)
        supplied = set()
        for orbit in exclusion_orbits:
            if not valid_vector(list(orbit)):
                raise ValueError("Malformed exclusion truth orbit")
            supplied.add(tuple(sorted(orbit)))
        if not supplied or supplied != verified:
            raise ValueError("Exclusion argument differs from bound audit")
        return verified
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise ValueError("BLOCKED_EXCLUSION_AUDIT: " + str(error)) from error


def build_confirmation(exclusion_orbits, *, exclusion_audit, auditor_authorized=False):
    """Auditor-only pure CPU generation; callers must establish freeze ordering first."""
    if auditor_authorized is not True:
        raise PermissionError(
            "E_CONFIRM2 generation requires explicit auditor authorization after mode freeze"
        )
    seen = _validate_exclusion_receipt(exclusion_orbits, exclusion_audit)
    starting_count = len(seen)
    rng, roots, tasks, audits, attempts = random.Random(2026100803), [], [], [], []

    def root(family, ordinal):
        if family == "trend":
            choices = []
            for a in range(100):
                for d in range(1, 34):
                    world = (a, a + d, a + 2 * d, a + 3 * d)
                    if world[-1] <= 99:
                        if world in seen:
                            attempts.append(
                                dict(
                                    family=family,
                                    ordinal_0based=ordinal,
                                    status="EXCLUDED_TRUTH_ORBIT",
                                    truth_orbit_key=list(world),
                                    method="enumerated_legal_trend_pool",
                                )
                            )
                        else:
                            choices.append(world)
            if not choices:
                raise ValueError(
                    "BLOCKED_EXCLUSION_AUDIT: no unused trend orbits; domain cannot expand"
                )
            world = list(rng.choice(choices))
            if rng.randrange(2):
                world.reverse()
        else:
            for draw in range(100000):
                world = rng.sample(range(100), 4)
                if tuple(sorted(world)) not in seen:
                    break
                attempts.append(
                    dict(
                        family=family,
                        ordinal_0based=ordinal,
                        draw_0based=draw,
                        status="EXCLUDED_TRUTH_ORBIT",
                        truth_orbit_key=sorted(world),
                    )
                )
            else:
                raise ValueError("No fresh root within registered sampling bound")
        replacements = []
        for value in world:
            alternative = rng.randrange(99)
            replacements.append(alternative + (alternative >= value))
        rid = stable_id("confirm-root", "E_CONFIRM2", ordinal, family, world, replacements)
        record = dict(
            phase_id=EXPERIMENT_ID,
            root_id=rid,
            split="E_CONFIRM2",
            family=family,
            ordinal_0based=ordinal,
            true_world=world,
            replacements=replacements,
        )
        roots.append(record)
        seen.add(tuple(sorted(world)))
        attempts.append(
            dict(
                family=family,
                ordinal_0based=ordinal,
                status="ACCEPTED_FRESH_LEGAL_ROOT",
                root_id=rid,
                truth_orbit_key=sorted(world),
            )
        )
        return record

    def add(record, center, j, ordinal, cohort):
        observed = list(record["true_world"])
        observed[j] = record["replacements"][j]
        task, audit = make_task(
            record["true_world"],
            observed,
            center,
            j,
            record["root_id"],
            "E_CONFIRM2",
            ordinal,
            record["family"],
        )
        task["root_cohort"] = audit["root_cohort"] = cohort
        validate_public(task)
        tasks.append(task)
        audits.append(audit)

    for ordinal in range(128):
        record = root("cross_series", ordinal)
        cells = (
            [(c, j) for c in range(4) for j in range(4)]
            if ordinal < 32
            else [(3, 3), (3, 1), (3, 2), (2, 2)]
        )
        for center, j in cells:
            add(record, center, j, ordinal, "CORE32" if ordinal < 32 else "EXTRA96")
    for family, count in (("trend", 64), ("duplicate_encoding", 32)):
        for ordinal in range(count):
            add(root(family, ordinal), None, ordinal % 4, ordinal, family)
    cell_counts = Counter(
        f"c{a['center'] + 1}j{a['corrupted_index'] + 1}" for a in audits if a["center"] is not None
    )
    if len(tasks) != 992 or len(roots) != 224 or len(seen) - starting_count != 224:
        raise ValueError("Confirmation root/task scope changed")
    if any(
        cell_counts[f"c{c + 1}j{j + 1}"]
        != (128 if (c, j) in ((3, 3), (3, 1), (3, 2), (2, 2)) else 32)
        for c in range(4)
        for j in range(4)
    ):
        raise ValueError("Confirmation cell coverage changed")
    files = {
        "E_CONFIRM2/tasks_public.jsonl": tasks,
        "E_CONFIRM2/audit_only.jsonl": audits,
        "E_CONFIRM2/root_registry_AUDIT_ONLY.jsonl": roots,
        "E_CONFIRM2/rejection_and_acceptance_audit.jsonl": attempts,
    }
    return files, dict(
        status="CPU_CONFIRMATION_GENERATION_PASS",
        phase_id=EXPERIMENT_ID,
        tasks=992,
        roots=224,
        cross_roots=128,
        unique_public_repair_checks=992,
        root_seed=2026100803,
        historical_exclusion_orbits=starting_count,
        exclusion_audit_hash=exclusion_audit["audit_hash"],
        model_calls=0,
        cell_counts=dict(cell_counts),
        confirmation_release=False,
    )


def build_confirmation_from_audit(audit_path, *, auditor_authorized=False):
    from .exclusions import load_verified_exclusions

    orbits = load_verified_exclusions(audit_path, verify_sources=True)
    return build_confirmation(
        orbits, exclusion_audit=read_json(audit_path), auditor_authorized=auditor_authorized
    )
