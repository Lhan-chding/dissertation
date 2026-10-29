"""Read-only, original-evidence audit of the registered twelve-step PILOT.

No supplied PASS/score/credit/report is authoritative. Admission, saved native
trajectories, committed states, every physical attempt, the one recovery and all
DEV_EVAL endpoints are replayed. This module does not load a model or authorize
FORMAL. Its explicit CPU entry can only return SYNTHETIC_ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from sr_f3_ce.training import finite_tree
from sr_f12.training import enforce_probability_gate

from . import pilot_contract, recovery, registration, rollouts, training
from .common import digest, write_once
from .credit import ARMS, build_credit_group
from .transactions import RunTransactions, _plain_path, _read_bytes, _require

FORMAT = "srf6-runtime-pilot-audit-v1"
DIRECTORY = "PHASE_RUNS/PILOT_V1/RUNTIME_PILOT_AUDIT"
TASK_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}"
KNOWN_ACCOUNTING = {
    "completion_attempts",
    "generated_tokens",
    "extra_forward_sequences",
    "diagnostic_gradient_calls",
    "cached_training_model_forward_calls",
}


def _read(path):
    return recovery._json(_read_bytes(path))


def _lines(path):
    data = _read_bytes(path)
    _require(not data or data.endswith(b"\n"), "Truncated append-only evidence")
    return [recovery._json(line) for line in data.splitlines()]


def _members(directory, *, files=False, suffix=None):
    directory = _plain_path(directory)
    if not directory.exists():
        return {}
    result = {}
    for path in directory.iterdir():
        _plain_path(path)
        _require(path.is_file() if files else path.is_dir(), "Unexpected inventory entry")
        _require(suffix is None or path.name.endswith(suffix), "Unexpected inventory suffix")
        name = path.name[: -len(suffix)] if suffix else path.name
        _require(re.fullmatch(TASK_PATTERN, name), "Unsafe physical task identifier")
        _require(name not in result, "Duplicate physical task identifier")
        result[name] = path
    return result


def _snapshot(directory):
    """Metadata-only inventory boundary; no repeated model/image content hashing."""
    result = []
    for path in sorted(directory.rglob("*")):
        _plain_path(path)
        info = path.stat()
        result.append(
            (
                path.relative_to(directory).as_posix(),
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
        )
    return result


def _event_key(event):
    kind, count, meta = event.get("kind"), event.get("count"), event.get("metadata")
    _require(kind in KNOWN_ACCOUNTING, "Unknown accounting kind")
    _require(type(count) is int and count >= 0 and isinstance(meta, dict), "Invalid accounting")
    if kind == "completion_attempts":
        _require(
            set(meta) == {"group_seed", "protocol"}
            and count == 8
            and type(meta["group_seed"]) is int
            and meta["protocol"] == "structured",
            "Native G8 reservation shape differs",
        )
        return (kind, count, meta["group_seed"], meta["protocol"])
    if kind == "generated_tokens":
        _require(
            set(meta) == {"group_seed", "row_index"}
            and 1 <= count <= 1536
            and type(meta["group_seed"]) is int
            and type(meta["row_index"]) is int
            and meta["row_index"] in range(8),
            "Native G8 token event differs",
        )
        return (kind, count, meta["group_seed"], meta["row_index"])
    return None


def _runtime(tx, root, task, *, synthetic):
    value = _read(tx.directory / "runtime_identity" / f"{task}.json")
    for key, expected in {
        "task_id": task,
        "audit_root": str(tx.directory),
        "run_root": str(root),
        "run_id": tx.identity["run_id"],
        "parent_id": tx.identity["parent"],
        "source_sha256": tx.identity["source_sha256"],
        "data_hash": tx.identity["data_sha256"],
        "reference_hash": tx.identity["reference_sha256"],
        "phase": pilot_contract.for_phase(tx.identity["phase"]).phase,
        "execution_mode": "TRAINING",
    }.items():
        _require(value.get(key) == expected, "Physical runtime identity differs: " + key)
    recovery._overlay_identity(value, tx.identity)
    if synthetic:
        _require(value.get("execution_origin") == "SYNTHETIC_TEST", "Explicit CPU origin required")
    else:
        hw = value.get("hardware", {})
        _require(
            value.get("execution_origin") != "SYNTHETIC_TEST"
            and hw.get("cuda_visible_device_count") == 1
            and "RTX PRO 6000" in str(hw.get("cuda_name", "")).upper()
            and all(hw.get(k) for k in ("cuda_uuid", "cuda_runtime", "driver_version"))
            and value.get("dtype") == "float32"
            and value.get("fresh_optimizer") is True
            and value.get("step") == 0
            and value.get("policy_hash") == tx.identity["reference_sha256"]
            and value.get("phase_identity_sha256") == tx.identity["phase_identity_sha256"]
            and value.get("config_sha256") == tx.identity["config_sha256"],
            "Actual CUDA loader/source receipt required; CPU cannot become actual evidence",
        )
    return value


def _numeric(tx, marker, update):
    audit = training._numerical_audit(update["probability_vectors"])
    _require(audit.get("status") == "FINITE", "Nonfinite probability trace")
    enforce_probability_gate(audit)
    _require(audit["sampler_train_max_absolute_difference"] <= 5, "Sampler/train maximum gate")
    _require(
        all(update.get(k) == v for k, v in audit.items() if k != "status"),
        "Saved numerical statistics differ from all original probability vectors",
    )
    evidence = recovery._bound(tx.directory, marker["references"]["evidence"])
    recovery._audit_event(tx.directory, evidence, required=True)
    _require(
        evidence.get("diagnostics") == audit
        and evidence.get("arm") == tx.identity["arm"]
        and evidence.get("audit_root") == str(tx.directory)
        and evidence.get("channel_gradients") == update.get("channel_gradients"),
        "Preupdate audit is not the actual saved update",
    )
    channels = update["channel_gradients"]
    if marker["step"] in (1, 4, 8, 12):
        _require(
            channels.get("status") == "MEASURED"
            and channels.get("row_indices") == list(registration.DIAGNOSTIC_ROWS)
            and channels.get("qid_count") == 8
            and channels.get("parameter_names") == marker["parameter_names"],
            "Fixed eight-G8 diagnostic subset differs",
        )
    else:
        _require(channels == {"status": "NOT_SCHEDULED"}, "Unregistered diagnostic selection")
    _require(finite_tree(channels), "Nonfinite channel gradients")
    norms = {
        k: update.get(k)
        for k in ("gradient_norm_before_clip", "gradient_norm_after_clip", "parameter_update_norm")
    }
    _require(
        all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in norms.values()),
        "Finite gradient/parameter norms required",
    )
    _require(
        update.get("microbatch_size") == 1 and update.get("kl_coefficient") == 0.02,
        "Registered update reduction differs",
    )
    return dict(step=marker["step"], probability=audit, channel_gradients=channels, **norms)


def _verify_run(tx, root, *, synthetic=False, attempt_pins=None):
    """Recheck persisted numerical states and deterministically rescore original raw."""
    tx._locked()
    markers = tx.published()
    _require([m["step"] for m in markers] == list(range(13)), "All thirteen pilot states required")
    pilot_contract.for_phase(tx.identity["phase"])
    _require(
        tx.identity["lane"] in (("PRIMARY", "MAIN") if synthetic else ("PRIMARY",))
        and tx.identity["max_steps"] == 12
        and tx.identity["repeat"] == 0
        and tx.identity["arm"] in registration.ARMS,
        "Only a registered primary pilot run",
    )
    reg = recovery._bound(tx.directory, tx.identity["rollout_registration"])
    _require(
        reg.get("identity")
        == {k: v for k, v in tx.identity.items() if k != "rollout_registration"},
        "Rollout registration identity differs",
    )
    config = recovery._bound(root, dict(path="CONFIG.json", sha256=tx.identity["config_sha256"]))
    manifest = recovery._bound(
        root, dict(path="DATA/MANIFEST.json", sha256=tx.identity["data_sha256"])
    )
    public, tasks = rollouts._registration_inputs(tx, root, reg, config, manifest)
    states = recovery._state(tx, markers[0], gpu=not synthetic)
    _require(markers[0]["policy_sha256"] == tx.identity["reference_sha256"], "Wrong parent start")
    del states
    groups, steps, rows_by_step, expected_events = [], [], {}, defaultdict(Counter)
    inputs, evidence_tasks = [], {}
    for marker in markers[1:]:
        step = marker["step"]
        state = recovery._state(tx, marker, gpu=not synthetic)
        del state
        originals = recovery._originals(tx, marker, synthetic=False)
        rng = originals["rollout_rng"]
        _require(
            recovery.state_hash(rng[0]) == recovery.state_hash(rng[2])
            and recovery.state_hash(rng[1]) == recovery.state_hash(rng[-1])
            and all(
                recovery.state_hash(rng[i]) == recovery.state_hash(rng[i + 1])
                for i in range(3, 33, 2)
            ),
            "Original G8 RNG chain does not connect the committed rollout boundaries",
        )
        rollout = recovery._bound(tx.directory, marker["references"]["rollout"])
        credit = recovery._bound(tx.directory, marker["references"]["credit"])
        update = _read(tx.checkpoints / f"step-{step:03d}" / "UPDATE.json")
        numeric = _numeric(tx, marker, update)
        expected_qids = [f"{rid}_P1_v{v}" for rid in reg["schedule"][step - 1] for v in range(4)]
        _require(
            rollout.get("registration_ref") == tx.identity["rollout_registration"],
            "Unregistered rollout source",
        )
        rows_by_step[step] = rollout["records"]
        for index, qid in enumerate(expected_qids):
            ref = rollout["complete_group_references"][index]
            expected_path = f"rollouts/step-{step:03d}/groups/{index:02d}/COMPLETE.json"
            _require(ref["path"] == expected_path, "Original group path differs")
            complete = recovery._bound(tx.directory, ref)
            raw = [
                recovery._json(line)
                for line in recovery._bound(
                    tx.directory, complete["raw_reference"], json_value=False
                ).splitlines()
            ]
            _require(
                complete["raw_reference"]["path"]
                == expected_path.replace("COMPLETE.json", "raw.jsonl"),
                "Original raw path differs",
            )
            _require(len({r.get("task_id") for r in raw}) == 1, "G8 mixed physical generator")
            task_id = raw[0]["task_id"]
            _require(
                isinstance(task_id, str) and re.fullmatch(TASK_PATTERN, task_id), "Missing raw task"
            )
            runtime = SimpleNamespace(
                identity={**_runtime(tx, root, task_id, synthetic=synthetic), "step": step - 1}
            )
            seeds = reg["draw_seeds"][step - 1][index]
            safe, task = public[qid]["model_input"], tasks[qid]
            step_spec = dict(
                format=rollouts.FORMAT,
                identity=tx.identity,
                step=step,
                registration_ref=tx.identity["rollout_registration"],
                policy_before_sha256=marker["policy_before_sha256"],
                roots=reg["schedule"][step - 1],
            )
            expected_spec = dict(
                step_spec_sha256=digest(step_spec),
                group_index=index,
                qid=qid,
                seeds=seeds,
                model_input_sha256=digest(safe),
                task_sha256=digest(task),
            )
            group_started = _read(
                tx.directory / expected_path.replace("COMPLETE.json", "STARTED.json")
            )
            _require(
                complete.get("spec") == group_started.get("spec") == expected_spec
                and group_started.get("rng_before_reference") == complete["rng_before_reference"],
                "Original group STARTED/COMPLETE registration differs",
            )
            rollouts._validate_raw(raw, runtime, safe, seeds, marker["policy_before_sha256"], root)
            annotated = rollouts._annotate(raw, task, tx, runtime, step, index)
            _require(
                annotated == rollout["records"][index * 8 : (index + 1) * 8],
                "Original raw/scorer/task metadata differs from committed rows",
            )
            actual = build_credit_group(
                annotated, tx.identity["arm"], reg["shuffle_seeds"][step - 1][index]
            )
            _require(
                actual == credit["groups"][index],
                "Applied credit differs from frozen recomputation",
            )
            selected = actual["counterfactuals"][ARMS[tx.identity["arm"]][1]]
            groups.append(
                dict(
                    step=step,
                    group_index=index,
                    qid=qid,
                    root_id=task["root_id"],
                    family=task["family"],
                    nonzero=selected["nonzero"],
                    exchangeable=selected["exchangeable"],
                    changed=selected["permutation_changed"],
                    generator_task_id=task_id,
                    source_step=step - 1,
                    source_commit_sha256=markers[step - 1]["commit_sha256"],
                    commit_sha256=marker["commit_sha256"],
                    original_reference=ref,
                )
            )
            expected_events[task_id][("completion_attempts", 8, seeds[0], "structured")] += 1
            for d, row in enumerate(raw):
                expected_events[task_id][("generated_tokens", len(row["tokens"]), seeds[0], d)] += 1
        evidence_path = Path(marker["references"]["evidence"]["path"])
        _require(
            evidence_path.parent.as_posix() == f"PREUPDATE_EVIDENCE/step-{step:03d}"
            and evidence_path.suffix == ".json",
            "Actual update task evidence path required",
        )
        evidence_tasks[step] = evidence_path.stem
        inputs.append(recovery._ref(root, tx.checkpoints / f"step-{step:03d}/COMMIT.json"))
        steps.append(dict(step=step, committed_update=marker["commit_sha256"], numerical=numeric))
    _require(
        {p.name for p in (tx.directory / "rollouts").iterdir()}
        == {f"step-{s:03d}" for s in range(1, 13)},
        "Extra/partial rollout step preserved; audit blocked",
    )
    for step in range(1, 13):
        directory = _plain_path(tx.directory / "rollouts" / f"step-{step:03d}" / "groups")
        _require(
            {p.name for p in directory.iterdir()} == {f"{g:02d}" for g in range(16)},
            "Extra/partial raw group inventory",
        )
        for path in directory.iterdir():
            _plain_path(path)
            _require(
                {p.name for p in path.iterdir()}
                == {"STARTED.json", "rng-before.pt", "raw.jsonl", "rng-after.pt", "COMPLETE.json"},
                "Extra/partial raw group artifacts",
            )
    inventory = verify_execution_inventory(
        tx,
        root=root,
        markers=markers,
        rows_by_step=rows_by_step,
        expected_events=expected_events,
        evidence_tasks=evidence_tasks,
        synthetic=synthetic,
        attempt_pins=attempt_pins,
    )
    return dict(
        format=FORMAT,
        scope="SYNTHETIC_ONLY" if synthetic else "SERVER_ACTUAL",
        parent=tx.identity["parent"],
        arm=tx.identity["arm"],
        run_id=tx.identity["run_id"],
        updates=12,
        responses=1536,
        groups=groups,
        steps=steps,
        inventory=inventory,
        inputs=inputs + inventory["inputs"],
        training_authorized=False,
    )


def verify_execution_inventory(
    tx,
    *,
    root,
    markers,
    rows_by_step,
    expected_events,
    evidence_tasks,
    synthetic=False,
    attempt_pins=None,
):
    """All PRIMARY attempts, including failures; only original committed G8 closes calls.

    A missing END is retained. It is reconciled only when its positive calls are
    all original complete groups and a later restored process consumed/passed its
    work. Empty unclosed attempts have no such proof and remain UNKNOWN.
    """
    tx._locked()
    attempts = _members(tx.directory / "TRAINER_ATTEMPTS")
    journals = _members(tx.directory / "EXECUTIONS")
    accounts = _members(tx.directory / "accounting", files=True, suffix=".jsonl")
    runtimes = _members(tx.directory / "runtime_identity", files=True, suffix=".json")
    tasks = set(attempts) | set(journals) | set(accounts) | set(runtimes) | set(expected_events)
    _require(tasks, "No physical execution inventory")
    if not synthetic:
        _require(set(attempts) == tasks, "Orphan execution/accounting/runtime without STARTED")
        _require(
            isinstance(attempt_pins, dict)
            and set(attempt_pins)
            == {"context", "phase_identity_sha256", "pilot_spec_sha256", "admission_ref"},
            "Actual admitted attempt provenance required",
        )
    inputs, facts, begins = [], [], {}
    diagnostic_events = defaultdict(Counter)
    for task, folder in journals.items():
        begin = _read(folder / "BEGIN.json")
        _require(
            begin.get("format") == recovery.FORMAT
            and begin.get("status") == "BEGIN"
            and begin.get("recovery_binding_ref") is None
            and begin.get("identity") == tx.identity
            and begin.get("task_id") == task
            and begin.get("scope") == ("SYNTHETIC_ONLY" if synthetic else "SERVER_ACTUAL")
            and begin.get("storage_lane") == "PRIMARY"
            and type(begin.get("start_step")) is int
            and begin["start_step"] in range(13)
            and begin.get("start_commit_sha256") == markers[begin["start_step"]]["commit_sha256"],
            "Execution BEGIN does not bind a real committed source",
        )
        _require(
            recovery._bound(root, begin["runtime_identity_reference"])
            == _runtime(tx, root, task, synthetic=synthetic)
            and begin["runtime_identity_reference"]["path"]
            == (tx.directory / "runtime_identity" / f"{task}.json").relative_to(root).as_posix(),
            "Execution runtime reference differs",
        )
        _require(
            type(begin.get("restored")) is bool and type(begin.get("resumed_after_data")) is bool,
            "Explicit restore facts required",
        )
        if not synthetic:
            _require(
                begin.get("process_identity", {}).get("kind") == "LINUX_PROC",
                "Actual process identity required",
            )
        begins[task] = begin
    for task in sorted(tasks):
        if not synthetic and task in runtimes:
            recovery._overlay_identity(
                _runtime(tx, root, task, synthetic=False),
                tx.identity,
                pilot_spec_sha256=attempt_pins["pilot_spec_sha256"],
            )
        for mapping in (attempts, journals):
            if task in mapping:
                for path in mapping[task].iterdir():
                    _plain_path(path)
                    _require(path.is_file(), "Unexpected attempt/journal object")
                    inputs.append(recovery._ref(root, path))
        for mapping in (accounts, runtimes):
            if task in mapping:
                inputs.append(recovery._ref(root, mapping[task]))
        start, terminal = None, None
        if task in attempts:
            folder = attempts[task]
            _require(
                {p.name for p in folder.iterdir()}
                <= {"STARTED.json", "COMPLETE.json", "FAILURE.json"}
                and (folder / "STARTED.json").exists()
                and not (
                    (folder / "COMPLETE.json").exists() and (folder / "FAILURE.json").exists()
                ),
                "Conflicting/unknown trainer attempt terminal",
            )
            start = _read(folder / "STARTED.json")
            _require(
                start.get("status") == "STARTED"
                and start.get("task_id") == task
                and start.get("run_id") == tx.identity["run_id"]
                and start.get("parent") == tx.identity["parent"]
                and start.get("arm") == tx.identity["arm"]
                and start.get("storage_lane") == "PRIMARY",
                "Trainer STARTED identity differs",
            )
            if not synthetic:
                _require(
                    start.get("execution_origin") == "SERVER_ACTUAL"
                    and start.get("phase_identity_sha256") == tx.identity["phase_identity_sha256"]
                    and all(start.get(k) == v for k, v in attempt_pins.items())
                    and start.get("recovery_binding") is None,
                    "Trainer phase provenance differs",
                )
            if task in begins:
                _require(
                    start.get("initial_published_step") == begins[task]["start_step"]
                    or (
                        start.get("initial_published_step") is None
                        and begins[task]["start_step"] == 0
                    ),
                    "Trainer attempt initial commit differs from actual BEGIN",
                )
            for name in ("COMPLETE", "FAILURE"):
                if (folder / (name + ".json")).exists():
                    terminal = _read(folder / (name + ".json"))
                    _require(terminal.get("task_id") == task, "Attempt terminal identity differs")
                    if name == "FAILURE":
                        _require(
                            terminal.get("status") == "FAILED_PRESERVED"
                            and terminal.get("sampling_retry_authorized") is False
                            and all(
                                terminal.get(k) == v for k, v in start.items() if k != "status"
                            ),
                            "Failure cannot authorize new sampling",
                        )
                    else:
                        _require(
                            terminal.get("status") == "COMPLETED_NOT_GATE_SIGNED"
                            and terminal.get("run_id") == tx.identity["run_id"]
                            and terminal.get("storage_lane") == "PRIMARY"
                            and terminal.get("committed_step")
                            == terminal.get("logical_updates")
                            == 12
                            and terminal.get("training_authorized") is False,
                            "Trainer COMPLETE identity/budget differs",
                        )
        events = _lines(accounts[task]) if task in accounts else []
        actual = Counter()
        counts = Counter()
        for event in events:
            _require(
                event.get("task_id") == task and event.get("parent") == tx.identity["parent"],
                "Accounting belongs to another task/parent",
            )
            key = _event_key(event)
            counts[event["kind"]] += event["count"]
            if key is not None:
                actual[key] += 1
            if event["kind"] == "diagnostic_gradient_calls":
                meta = event["metadata"]
                _require(
                    set(meta) == {"row_id", "step"}
                    and event["count"] == 3
                    and type(meta["step"]) is int
                    and isinstance(meta["row_id"], str),
                    "Fixed three-channel diagnostic call shape differs",
                )
                diagnostic_events[task][(meta["row_id"], meta["step"], 3)] += 1
        _require(
            actual == expected_events.get(task, Counter()),
            "All attempts must match original G8/draw accounting exactly; "
            "extra/unknown calls remain",
        )
        n = counts["completion_attempts"]
        _require(not n or task in begins, "Data call without execution BEGIN")
        if task in runtimes:
            _runtime(tx, root, task, synthetic=synthetic)
        status = (
            "FAILED_BEFORE_DATA"
            if terminal and terminal.get("status") == "FAILED_PRESERVED"
            else "UNKNOWN"
        )
        journal_steps = []
        if task in journals:
            folder, begin = journals[task], begins[task]
            _require(
                {p.name for p in folder.iterdir()}
                <= {"BEGIN.json", "COMMITS.jsonl", "END.json", "FAILURE.json"}
                and not ((folder / "END.json").exists() and (folder / "FAILURE.json").exists()),
                "Conflicting/unknown execution terminal",
            )
            entries = _lines(folder / "COMMITS.jsonl")
            journal_steps = [v["step"] for v in entries]
            _require(
                journal_steps
                == list(range(begin["start_step"] + 1, begin["start_step"] + len(entries) + 1)),
                "Duplicate/non-prefix execution COMMIT journal",
            )
            for entry in entries:
                s = entry["step"]
                _require(
                    s in range(1, 13) and evidence_tasks[s] == task,
                    "Journal claims another updater's COMMIT",
                )
                _require(
                    recovery._bound(root, entry["commit_reference"]) == markers[s]
                    and entry["commit_sha256"] == markers[s]["commit_sha256"]
                    and entry["rollout_reference"] == markers[s]["references"]["rollout"],
                    "Journal commit binding differs",
                )
                new = sum(r["task_id"] == task for r in rows_by_step[s])
                _require(
                    entry["new_model_calls"] == new
                    and entry["reused_complete_groups"] == 16 - new // 8,
                    "Journal new/reused accounting differs",
                )
            if (folder / "END.json").exists():
                recovery._execution(
                    root,
                    recovery._ref(root, folder / "END.json"),
                    tx.directory,
                    tx.identity,
                    pilot_spec_sha256=(attempt_pins or {}).get("pilot_spec_sha256"),
                )
                status = "SEGMENT_COMPLETE"
                if terminal and terminal.get("status") != "FAILED_PRESERVED":
                    _require(
                        terminal.get("execution") == recovery._ref(root, folder / "END.json")
                        and terminal.get("final_commit")
                        == recovery._ref(root, tx.checkpoints / "step-012/COMMIT.json"),
                        "Trainer COMPLETE does not bind actual END/COMMIT",
                    )
            else:
                # Published commits and complete raw are authoritative even if killed
                # between rename and journal append. A later restore is essential.
                data_steps = [
                    s for s, rows in rows_by_step.items() if any(r["task_id"] == task for r in rows)
                ]
                consumed = [s for s, owner in evidence_tasks.items() if owner == task]
                boundary = max(consumed or [begin["start_step"]])
                successor = any(
                    other != task and b["restored"] and b["start_step"] >= boundary
                    for other, b in begins.items()
                )
                update_calls = sum(
                    counts[k]
                    for k in KNOWN_ACCOUNTING
                    if k not in ("completion_attempts", "generated_tokens")
                )
                if successor and ((n and data_steps) or update_calls):
                    status = "INTERRUPTED_RECONCILED_WITH_ORIGINALS"
                elif terminal and terminal.get("status") == "FAILED_PRESERVED" and not events:
                    status = "FAILED_BEFORE_DATA"
                else:
                    status = "UNKNOWN"
        _require(
            status != "UNKNOWN",
            "Unclosed attempt has no proof of complete calls/restore; retain UNKNOWN",
        )
        facts.append(
            dict(
                task_id=task,
                status=status,
                counts=dict(counts),
                journal_steps=journal_steps,
                trainer_terminal=terminal.get("status") if terminal else None,
                execution_terminal=(
                    "END"
                    if task in journals and (journals[task] / "END.json").exists()
                    else "FAILURE"
                    if task in journals and (journals[task] / "FAILURE.json").exists()
                    else None
                ),
            )
        )
    _require(set(evidence_tasks.values()) <= set(begins), "Committed updater has no physical BEGIN")
    for step, task in evidence_tasks.items():
        _require(begins[task]["start_step"] < step, "Update predates its restore boundary")
    # Every numerical event is retained. Exact PREUPDATE bytes identify physical
    # attempts; unreferenced failure events cannot be laundered by a later success.
    audit_path = tx.directory / "NUMERICAL_TOKEN_AUDIT.jsonl"
    audit_events = _lines(audit_path)
    bound_slices = set()
    physical_evidence = []
    for step_dir in (tx.directory / "PREUPDATE_EVIDENCE").iterdir():
        _plain_path(step_dir)
        _require(re.fullmatch(r"step-0(?:0[1-9]|1[0-2])", step_dir.name), "Unknown numerical step")
        for path in step_dir.iterdir():
            _plain_path(path)
            evidence = _read(path)
            _require(path.stem in begins and path.suffix == ".json", "Unknown numerical updater")
            _require(
                evidence.get("task_id") == path.stem,
                "Numerical event physical producer differs from wrapper path",
            )
            recovery._audit_event(tx.directory, evidence, required=True)
            ref = evidence["source_audit_event"]
            _require(ref["byte_offset"] not in bound_slices, "Duplicated original numerical event")
            bound_slices.add(ref["byte_offset"])
            physical_evidence.append(
                dict(
                    task_id=path.stem,
                    step=evidence["committed_source_step"] + 1,
                    reference=recovery._ref(root, path),
                )
            )
            inputs.append(recovery._ref(root, path))
    offset = 0
    physical_pairs = set()
    fact_by_task = {item["task_id"]: item for item in facts}
    for line, event in zip(
        _read_bytes(audit_path).splitlines(keepends=True), audit_events, strict=True
    ):
        _require(
            event.get("status") == "PREUPDATE_PASS",
            "Failed numerical attempt remains in inventory; cannot assert runtime gate",
        )
        owner = event.get("task_id")
        source_step = event.get("committed_source_step")
        _require(
            owner in begins and type(source_step) is int and source_step in range(12),
            "Actual numerical event must name its original physical task and step",
        )
        step = source_step + 1
        _require((owner, step) not in physical_pairs, "Repeated physical task/step numerical event")
        physical_pairs.add((owner, step))
        if offset not in bound_slices:
            # A process can die after Adam returned, before the slice wrapper was
            # written. The old event is not a committed update. Reconcile it only
            # against the exact same original data/policy's later measured event,
            # and its explicitly identified updater with a verified later restore.
            committed = recovery._bound(tx.directory, markers[step]["references"]["evidence"])
            _require(
                {k: v for k, v in event.items() if k not in ("at", "task_id")}
                == {
                    k: v
                    for k, v in committed.items()
                    if k not in ("at", "task_id", "source_audit_event")
                }
                and offset < committed["source_audit_event"]["byte_offset"],
                "Unbound preupdate does not exactly match restored original-data replay",
            )
            fact = fact_by_task[owner]
            _require(
                fact["status"] == "INTERRUPTED_RECONCILED_WITH_ORIGINALS"
                and step == max(fact["journal_steps"] or [begins[owner]["start_step"]]) + 1
                and evidence_tasks[step] != owner,
                "Uncommitted preupdate lacks its task's restore/physical attempt proof",
            )
            physical_evidence.append(
                dict(
                    task_id=owner,
                    step=step,
                    status="UNCOMMITTED_PREUPDATE_RECONCILED",
                    byte_offset=offset,
                    byte_count=len(line),
                    sha256=hashlib.sha256(line).hexdigest(),
                    matched_committed_evidence=markers[step]["references"]["evidence"],
                )
            )
        offset += len(line)
    for fact in facts:
        task = fact["task_id"]
        expected_diagnostics = Counter(
            (rows_by_step[step][row]["row_id"], step, 3)
            for owner, step in physical_pairs
            if owner == task and step in (1, 4, 8, 12)
            for row in registration.DIAGNOSTIC_ROWS
        )
        _require(
            diagnostic_events[task] == expected_diagnostics,
            "Physical diagnostic calls differ from each actual fixed-subset update attempt",
        )
        if fact["status"] == "INTERRUPTED_RECONCILED_WITH_ORIGINALS" and not fact["counts"].get(
            "completion_attempts", 0
        ):
            _require(
                any(owner == fact["task_id"] for owner, _ in physical_pairs),
                "Cache-only interrupted update has no attributable original numerical event",
            )
    inputs.append(recovery._ref(root, audit_path))
    _require(
        sum(f["counts"].get("completion_attempts", 0) for f in facts) == 1536,
        "Primary physical generation total must be exactly1536 without replacement",
    )
    return dict(
        status="ORIGINAL_FACTS_RECONCILED",
        attempts=facts,
        actual_generated_responses=1536,
        physical_preupdate_events=physical_evidence,
        physical_preupdate_count=len(audit_events),
        inputs=inputs,
    )


def verify_synthetic_run(tx, *, root):
    """CPU regression entry; never emits actual experiment or gate certification."""
    result = _verify_run(tx, _plain_path(root), synthetic=True)
    return {**result, "status": "SYNTHETIC_ONLY", "gate_status": "NOT_SIGNED"}


def _exposure(runs):
    selected = [r for r in runs if r["arm"] == "U11"]
    nonzero = [(r["parent"], g) for r in selected for g in r["groups"] if g["nonzero"]]
    witnesses = {}
    for run in selected:
        pairs = []
        ordered = sorted(run["groups"], key=lambda g: (g["step"], g["qid"]))
        for first in ordered:
            if not first["nonzero"] or first["step"] > 10:
                continue
            for later in ordered:
                if (
                    later["nonzero"]
                    and later["step"] > first["step"]
                    and later["source_step"] >= first["step"]
                ):
                    pairs.append(
                        dict(
                            first_step=first["step"],
                            first_qid=first["qid"],
                            committed_update=first["commit_sha256"],
                            later_step=later["step"],
                            later_qid=later["qid"],
                            newly_generated_source_step=later["source_step"],
                            original_reference=later["original_reference"],
                        )
                    )
                    break
            if pairs:
                break
        witnesses[run["parent"]] = pairs
    facts = dict(
        nonzero_groups=len(nonzero),
        roots=sorted({g["root_id"] for _, g in nonzero}),
        families=sorted({g["family"] for _, g in nonzero}),
        parents_with_feedback=sorted(p for p, rows in witnesses.items() if rows),
        feedback_witnesses=witnesses,
    )
    facts["witness_selection"] = (
        "Lexicographically earliest (first_step,first_qid,later_step,later_qid) "
        "per parent; all groups retained"
    )
    facts["registered_exposure_met"] = (
        facts["nonzero_groups"] >= 12
        and len(facts["roots"]) >= 8
        and len(facts["families"]) >= 3
        and len(facts["parents_with_feedback"]) >= 2
    )
    facts["shuf_description"] = {
        run["parent"]: {
            name: sum(g[name] for g in run["groups"])
            for name in ("nonzero", "exchangeable", "changed")
        }
        for run in runs
        if run["arm"] == "C_SHUF"
    }
    return facts


def _comparison_inventory(root, directory, compared):
    """Never select a favorable UUID while concealing an earlier comparison."""
    expected = {
        k: v
        for k, v in compared.items()
        if k not in ("receipt_reference", "verified_from_saved_facts")
    }
    receipts, unknown = [], False
    folder = _plain_path(directory / "COMPARISONS")
    for path in sorted(folder.iterdir()):
        _plain_path(path)
        _require(
            path.is_file() and re.fullmatch(r"[0-9a-f]{32}(?:-FAILURE)?\.json", path.name),
            "Unexpected recovery comparison attempt",
        )
        saved = _read(path)
        if path.name.endswith("-FAILURE.json"):
            _require(
                saved.get("format") == recovery.FORMAT
                and saved.get("status") == "INVALID_OR_INCOMPLETE_EVIDENCE"
                and saved.get("binding_ref") == expected["binding_ref"]
                and saved.get("gate_status") == "NOT_SIGNED"
                and saved.get("training_authorized") is False,
                "Unbound historical comparison failure",
            )
            unknown = True
        else:
            _require(
                saved == expected,
                "Historical recovery comparison differs from current original facts",
            )
        receipts.append(dict(reference=recovery._ref(root, path), evidence=saved))
    _require(
        any(row["reference"] == compared["receipt_reference"] for row in receipts),
        "Selected recovery receipt is absent from full comparison inventory",
    )
    return dict(
        status="HISTORICAL_TECHNICAL_UNKNOWN" if unknown else "ALL_SAVED_FACTS_VERIFIED",
        receipts=receipts,
    )


def verify_runtime_pilot(
    root,
    *,
    phase_identity_sha256,
    pilot_spec_sha256,
    admission_ref,
    recovery_binding_ref,
    recovery_comparison_ref,
    evaluation_summary_ref,
    phase="PILOT_V1",
):
    """Verify all registered actual facts; no caller-provided PASS or data is accepted."""
    from .evaluation import verify_evaluation

    root = _plain_path(root)
    contract = pilot_contract.for_phase(phase)
    admitted = pilot_contract.require_admission(
        root,
        phase=contract.phase,
        phase_identity_sha256=phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        admission_ref=admission_ref,
    )
    spec, context = admitted["pilot_spec"], admitted["phase_context"]
    paths = spec["paths"]
    _require(
        len(paths) == 9
        and {(p["parent"], p["arm"]) for p in paths}
        == {(p, a) for p in registration.PARENTS for a in registration.ARMS},
        "Exact nine primary paths required",
    )
    recovered = recovery.verify_recovery(
        root,
        phase_context=context,
        pilot_spec_sha256=pilot_spec_sha256,
        binding_ref=recovery_binding_ref,
    )
    _require(recovered["scope"] == "SERVER_ACTUAL", "Actual registered recovery required")
    saved_comparison = recovery._bound(root, recovery_comparison_ref)
    results, transactions, snapshots = [], {}, {}
    with ExitStack() as locks:
        for item in paths:
            identity = recovery._bound(
                root / contract.registration_directory, item["transaction_identity"]
            )
            directory = recovery._path(root, item["audit_root"])
            tx = locks.enter_context(RunTransactions(directory, identity))
            transactions[(item["parent"], item["arm"])] = tx
            results.append(
                _verify_run(
                    tx,
                    root,
                    attempt_pins=dict(
                        context=context["identity"],
                        phase_identity_sha256=phase_identity_sha256,
                        pilot_spec_sha256=pilot_spec_sha256,
                        admission_ref=admission_ref,
                    ),
                )
            )
        rec_tx = locks.enter_context(
            RunTransactions(Path(recovered["directory"]), recovered["identity"])
        )
        compared = recovery.verify_comparison(
            transactions[("b0", "U11")],
            rec_tx,
            root=root,
            phase_context=context,
            pilot_spec_sha256=pilot_spec_sha256,
            binding_ref=recovery_binding_ref,
            primary_execution_ref=saved_comparison["primary_execution_ref"],
            recovery_execution_ref=saved_comparison["recovery_execution_ref"],
            comparison_ref=recovery_comparison_ref,
        )
        comparisons = _comparison_inventory(root, rec_tx.directory, compared)
        for tx in [*transactions.values(), rec_tx]:
            snapshots[tx.directory] = _snapshot(tx.directory)
    evaluation = verify_evaluation(
        root,
        phase_identity_sha256=phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        admission_ref=admission_ref,
        summary_ref=evaluation_summary_ref,
        phase=contract.phase,
    )
    _require(
        evaluation["status"] == "COMPLETE_DESCRIPTIVE_NOT_GATE_SIGNED"
        and evaluation["checkpoints"] == 12
        and evaluation["responses"] == 2304,
        "All twelve actual evaluation endpoints required",
    )
    _require(
        all(_snapshot(directory) == value for directory, value in snapshots.items()),
        "Training/recovery inventory changed while evaluation was being verified",
    )
    exposure = _exposure(results)
    exact = compared["exact"] and compared["new_generated_responses"] == 1024
    inputs = (
        [
            admission_ref,
            admitted["pilot_spec_ref"],
            recovery_binding_ref,
            recovery_comparison_ref,
            evaluation_summary_ref,
        ]
        + [ref for run in results for ref in run["inputs"]]
        + [row["reference"] for row in comparisons["receipts"]]
    )
    # These are small final evidence boundaries; images/base weights are not rescanned.
    for ref in inputs:
        recovery._bound(root, ref, json_value=False)
    return dict(
        format=FORMAT,
        scope="SERVER_ACTUAL",
        status=(
            "RECOVERY_COMPARISON_HISTORY_UNKNOWN"
            if comparisons["status"] == "HISTORICAL_TECHNICAL_UNKNOWN"
            else "RECOVERY_MISMATCH"
            if not exact
            else "COMPLETE_FACTS_NOT_GATE_SIGNED"
            if exposure["registered_exposure_met"]
            else "INSUFFICIENT_REGISTERED_EXPOSURE"
        ),
        phase_identity_sha256=phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        **(
            dict(
                phase=contract.phase,
                input_overlay=spec["input_overlay"],
                prompt_version=spec["prompt_version"],
            )
            if contract.phase == "PILOT_V3"
            else {}
        ),
        paths=9,
        logical_updates=108,
        primary_responses=13824,
        primary_actual_generated_responses=sum(
            r["inventory"]["actual_generated_responses"] for r in results
        ),
        recovery_responses=1024,
        evaluation_responses=2304,
        runs=results,
        exposure=exposure,
        recovery=compared,
        recovery_comparison_history=comparisons,
        evaluation=evaluation,
        inputs=inputs,
        gate_status="NOT_SIGNED",
        training_authorized=False,
        formal_authorized=False,
        effectiveness_gate=None,
        interpretation=(
            "Runtime/exposure facts only. One repeat is not training variance, "
            "effect evidence or formal authorization."
        ),
    )


def audit(root, **kwargs):
    """Publish one immutable fact report, never a scientific gate signature."""
    result = verify_runtime_pilot(root, **kwargs)
    destination = (
        _plain_path(root)
        / pilot_contract.for_phase(kwargs.get("phase", "PILOT_V1")).audit_directory
        / "FACTS.json"
    )
    write_once(_plain_path(destination), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--phase", choices=("PILOT_V1", "PILOT_V3"), default="PILOT_V1")
    for name in (
        "phase-identity-sha256",
        "pilot-spec-sha256",
        "admission-sha256",
        "recovery-binding-sha256",
        "recovery-comparison-path",
        "recovery-comparison-sha256",
        "evaluation-summary-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    contract = pilot_contract.for_phase(args.phase)

    result = audit(
        args.root,
        phase=contract.phase,
        phase_identity_sha256=args.phase_identity_sha256,
        pilot_spec_sha256=args.pilot_spec_sha256,
        admission_ref=dict(path=contract.admission_path, sha256=args.admission_sha256),
        recovery_binding_ref=dict(
            path=contract.recovery_directory + "/RECOVERY_BINDING.json",
            sha256=args.recovery_binding_sha256,
        ),
        recovery_comparison_ref=dict(
            path=args.recovery_comparison_path, sha256=args.recovery_comparison_sha256
        ),
        evaluation_summary_ref=dict(
            path=contract.evaluation_directory + "/SUMMARY.json",
            sha256=args.evaluation_summary_sha256,
        ),
    )
    print(result["status"])


if __name__ == "__main__":
    main()
