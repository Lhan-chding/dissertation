"""Independent committed-delivery audit retaining the unique registered old OOM.

Original committed-state, raw, RNG, credit and numerical assertions are unchanged.
Only the fixed pre-optimizer b1/U10 event is reconciled against preserved source
and later restore evidence. Its partial diagnostics never count as a full step.
This consumer writes no evidence and never signs the original runtime gate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from sr_f6 import phases
from sr_f6 import pilot_audit as frozen

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
EXPERIMENT = PROJECT / "sr_f6_semantic_factorial_20261027"
RUN = "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/b1/r0/U10"
OLD_TASK = "LH-SRF6-A-1-A-233996-PRIMARY-b1-U10"
AUDIT_BYTES = 2481675
AUDIT_SHA256 = "1978e39d51e6b3bbbedc0048b4275db048d1f9507d7a7b40c0c8baa504ad0915"
V7 = dict(
    path=str(PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V7_20261030/CONTRACT.json"),
    sha256="e5e14a0ce0036b9b640c930d01fc9f34ce68a7f48f0303eebb44d323b0e28cd9",
)
V8 = dict(
    path=str(PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_RESUME_V8_20261030/CONTRACT.json"),
    sha256="6ae70ea969803febcb8f91c4e2ad6da78c9e901abb30769bd109ddda8e608957",
)
TRACE_SHA256 = "2b393958950be9f9721ee492668ce8a74af88b41f11e6c6692283a9da33dbbfe"


def _cold_reference(reference, *, json_value=True, stream_only=False):
    """Read actual bytes under the project without trusting a content cache."""
    frozen._require(
        isinstance(reference, dict)
        and set(reference) == {"path", "sha256"}
        and frozen.recovery._sha(reference["sha256"]),
        "Exact immutable historical reference required",
    )
    path = phases._path(Path(reference["path"]))
    frozen._require(
        path.is_relative_to(PROJECT) and path != PROJECT, "Historical reference outside project"
    )
    before = frozen.recovery._stamp(path)
    hasher = hashlib.sha256()
    chunks = []
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
            if not stream_only:
                chunks.append(chunk)
    frozen._require(hasher.hexdigest() == reference["sha256"], "Historical reference bytes changed")
    frozen._require(
        frozen.recovery._stamp(path) == before, "Historical reference changed during read"
    )
    if stream_only:
        return None
    payload = b"".join(chunks)
    return frozen.recovery._json(payload) if json_value else payload


def _absolute_ref(path):
    return dict(path=str(path), sha256=frozen.recovery._file_hash(path))


def _oom_proof(reference):
    proof = _cold_reference(reference)
    frozen._require(
        set(proof)
        == {
            "schema",
            "memory_contract_ref",
            "resume_contract_ref",
            "old_execution_refs",
            "traceback_ref",
        }
        and proof["schema"] == "srf6-post-b-registered-oom-proof-v1",
        "Exact registered OOM proof schema required",
    )
    frozen._require(
        proof["memory_contract_ref"] == V7 and proof["resume_contract_ref"] == V8,
        "Only immutable registered v7/v8 lineage allowed",
    )
    contract, resume = _cold_reference(V7), _cold_reference(V8)
    frozen._require(
        contract["schema"] == "srf6-memory-full-contract-v7"
        and resume["schema"] == "srf6-memory-resume-contract-v8"
        and resume["full_contract_ref"] == V7,
        "Original memory repair lineage changed",
    )
    result = _cold_reference(resume["full_result_ref"])
    complete = _cold_reference(resume["full_complete_ref"])
    cold = _cold_reference(resume["full_cold_ref"])
    frozen._require(
        result["status"] == cold["status"] == "VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH"
        and result["contract_sha256"] == cold["contract_sha256"] == V7["sha256"]
        and complete["status"] == "COMPLETED_FULL_VALIDATION_NO_TRAINING"
        and complete["job_id"] == "235864"
        and complete["contract_ref"] == V7
        and complete["result_ref"] == cold["validation_ref"] == resume["full_result_ref"]
        and complete["cold_verification_ref"] == resume["full_cold_ref"],
        "Existing v7 validation proof linkage differs",
    )
    frozen._require(
        cold["zero_new_samples"] is True
        and cold["zero_optimizer_updates"] is True
        and cold["generated_responses"] == cold["optimizer_updates"] == 0
        and cold["original_ready_ref"] == contract["original_ready_ref"],
        "Existing v7 scope differs",
    )
    inputs = [
        reference,
        V7,
        V8,
        resume["full_result_ref"],
        resume["full_complete_ref"],
        resume["full_cold_ref"],
    ]
    for member in (result["evidence_ref"], result["verifier_source_ref"]):
        _cold_reference(member, stream_only=True)
        inputs.append(member)
    frozen._require(
        result["verifier_source_ref"] == cold["verifier_source_ref"],
        "Cold/source verifier proof differs",
    )
    # TENSORS is retained as an existing dependency reference, not loaded or
    # validated again: this operation reconciles the old failed training event.
    return proof, contract, inputs, result["tensors_ref"]


def _reconcile_registered_oom(
    tx,
    *,
    root,
    event,
    line,
    offset,
    markers,
    begins,
    fact_by_task,
    rows_by_step,
    evidence_tasks,
    bound_slices,
    proof_ref,
    synthetic,
):
    """One immutable no-Adam incident, never a generic failed-event exemption."""
    frozen._require(
        not synthetic
        and root == EXPERIMENT
        and tx.directory == root / RUN
        and tx.identity["phase"] == "PILOT_V3"
        and tx.identity["parent"] == "b1"
        and tx.identity["arm"] == "U10"
        and tx.identity["lane"] == "PRIMARY"
        and tx.identity["repeat"] == 0,
        "Registered actual b1/U10 only",
    )
    frozen._require(
        offset == 0
        and len(line) == AUDIT_BYTES
        and hashlib.sha256(line).hexdigest() == AUDIT_SHA256
        and offset not in bound_slices,
        "Only the fixed original OOM prefix can be reconciled",
    )
    frozen._require(
        event.get("status") == "UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP"
        and event.get("task_id") == OLD_TASK
        and event.get("committed_source_step") == 0
        and type(event.get("committed_source_step")) is int
        and event.get("optimizer_started") is False
        and event.get("parameter_state_changed") is False
        and event.get("requires_checkpoint_restore") is False
        and event.get("error_type") == "OutOfMemoryError"
        and "out of memory" in event.get("error", "").lower()
        and event.get("run_id") == tx.identity["run_id"]
        and event.get("arm") == "U10"
        and event.get("audit_root") == str(tx.directory)
        and event.get("policy_before_sha256")
        == markers[0]["policy_sha256"]
        == markers[1]["policy_before_sha256"],
        "Original failure is not a proven unchanged pre-optimizer attempt",
    )
    frozen._require(frozen.recovery._json(line) == event, "Original OOM bytes/event differ")
    proof, contract, inputs, tensors_ref = _oom_proof(proof_ref)
    frozen._require(
        contract["experiment_root"] == str(root)
        and contract["original_execution_id"] == OLD_TASK
        and contract["original_audit_bytes"] == AUDIT_BYTES
        and contract["original_audit_path"] == str(tx.directory / "NUMERICAL_TOKEN_AUDIT.jsonl")
        and contract["original_audit_ref"]["sha256"] == AUDIT_SHA256
        and _cold_reference(contract["original_audit_ref"], json_value=False) == line,
        "Registered original prefix differs",
    )
    inputs.append(contract["original_audit_ref"])
    originals = contract["original_files"]
    paths = [item["path"] for item in originals]
    frozen._require(
        len(paths) == len(set(paths)) == 88
        and all(
            Path(path).is_relative_to(tx.directory / "checkpoints/step-000")
            or Path(path).is_relative_to(tx.directory / "rollouts/step-001")
            for path in paths
        ),
        "Original frozen 88-file dependency closure differs",
    )
    for member in originals:
        _cold_reference(member, stream_only=True)
        inputs.append(member)
    expected_paths = {
        "original_ready_ref": tx.directory / "rollouts/step-001/READY.json",
        "step0_commit_ref": tx.directory / "checkpoints/step-000/COMMIT.json",
        "original_failure_ref": tx.directory / "TRAINER_ATTEMPTS" / OLD_TASK / "FAILURE.json",
        "original_task_failure_ref": root
        / "PHASE_RUNS/PILOT_V3/JOBS/TASKS/PRIMARY-b1-U10/ATTEMPTS"
        / OLD_TASK
        / "FAILURE.json",
        "original_worker_failure_ref": root
        / "PHASE_RUNS/PILOT_V3/JOBS/WORKERS/A/1/ATTEMPTS/A-233996/FAILURE.json",
    }
    old = {}
    for key, expected in expected_paths.items():
        frozen._require(
            contract[key]["path"] == str(expected), "Original evidence location differs"
        )
        old[key] = _cold_reference(contract[key])
        inputs.append(contract[key])
    frozen._require(old["step0_commit_ref"] == markers[0], "Original source checkpoint changed")
    for key in ("original_failure_ref", "original_task_failure_ref", "original_worker_failure_ref"):
        frozen._require(
            old[key].get("status") == "FAILED_PRESERVED", "Preserved original failure missing"
        )
    terminal = old["original_failure_ref"]
    frozen._require(
        terminal.get("task_id") == OLD_TASK
        and terminal.get("error_type") == "OutOfMemoryError"
        and terminal.get("error") == event["error"]
        and terminal.get("sampling_retry_authorized") is False
        and terminal.get("requires_fresh_process_restore") is True,
        "Old trainer failure does not bind this preserved OOM",
    )
    refs = proof["old_execution_refs"]
    expected = {
        "begin_ref": tx.directory / "EXECUTIONS" / OLD_TASK / "BEGIN.json",
        "commits_ref": tx.directory / "EXECUTIONS" / OLD_TASK / "COMMITS.jsonl",
        "failure_ref": tx.directory / "EXECUTIONS" / OLD_TASK / "FAILURE.json",
        "accounting_ref": tx.directory / "accounting" / (OLD_TASK + ".jsonl"),
        "runtime_ref": tx.directory / "runtime_identity" / (OLD_TASK + ".json"),
    }
    frozen._require(set(refs) == set(expected), "Complete original execution proof required")
    for key, path in expected.items():
        frozen._require(
            refs[key]["path"] == str(path), "Original execution reference location differs"
        )
        _cold_reference(refs[key], stream_only=True)
        inputs.append(refs[key])
    begin = _cold_reference(refs["begin_ref"])
    failure = _cold_reference(refs["failure_ref"])
    frozen._require(
        begin == begins[OLD_TASK]
        and begin["start_step"] == 0
        and begin["start_commit_sha256"] == markers[0]["commit_sha256"]
        and _cold_reference(refs["commits_ref"], json_value=False) == b""
        and OLD_TASK not in evidence_tasks.values()
        and not fact_by_task[OLD_TASK]["journal_steps"],
        "Old failed attempt has published updates or lost source binding",
    )
    frozen._require(
        failure.get("status") == "INCOMPLETE_SEGMENT"
        and failure.get("scope") == "SERVER_ACTUAL"
        and failure.get("task_id") == OLD_TASK
        and failure.get("error_type") == "OutOfMemoryError"
        and failure.get("error") == event["error"]
        and frozen.recovery._bound(root, failure["begin_reference"]) == begin,
        "Old execution failure does not bind source BEGIN",
    )
    frozen._require(
        not any(
            path.stem == OLD_TASK for path in (tx.directory / "PREUPDATE_EVIDENCE").glob("*/*.json")
        ),
        "Failed OOM cannot have a PREUPDATE wrapper",
    )
    trace_path = (
        root / "PHASE_RUNS/PILOT_V3/JOBS/TASKS/PRIMARY-b1-U10/ATTEMPTS" / OLD_TASK / "process.log"
    )
    frozen._require(
        proof["traceback_ref"] == dict(path=str(trace_path), sha256=TRACE_SHA256),
        "Only preserved original traceback allowed",
    )
    traceback = _cold_reference(proof["traceback_ref"], json_value=False).decode("utf-8")
    frozen._require(
        'training.py", line 537' in traceback
        and '(result["loss"] / 128).backward()' in traceback
        and "OutOfMemoryError" in traceback,
        "Original failure must be the registered backward OOM",
    )
    inputs.append(proof["traceback_ref"])
    ready = old["original_ready_ref"]
    frozen._require(
        ready.get("status") == "ROLLOUT_READY"
        and ready["references"]["rollout"] == markers[1]["references"]["rollout"]
        and ready["sampler_state"]["rollout_cursor"] == {"step": 1, "groups": 16, "rows": 128},
        "Later first commit did not reuse the original READY",
    )
    rows = rows_by_step[1]
    frozen._require(
        len(rows) == 128 and all(row["task_id"] == OLD_TASK for row in rows),
        "Original 128 answers were replaced or mixed",
    )
    trajectory_keys = (
        "row_id",
        "qid",
        "tokens",
        "sampler_logprobs",
        "token_path_sha256",
        "policy_hash",
        "prompt_tensor_hash",
        "source_sha256",
    )
    frozen._require(
        event["original_trajectories"]
        == [{key: row.get(key) for key in trajectory_keys} for row in rows],
        "Failed original trajectories differ from committed original raw",
    )
    traces = event["probability_vectors"]
    frozen._require(
        len(traces) == 27
        and all(
            trace["row_id"] == rows[index]["row_id"]
            and trace["tokens"] == rows[index]["tokens"]
            and trace["token_path_sha256"] == rows[index]["token_path_sha256"]
            for index, trace in enumerate(traces)
        ),
        "Fixed partial 27-forward diagnostic provenance differs",
    )
    partial = frozen.training._numerical_audit(traces)
    frozen._require(
        partial == event["diagnostics"]
        and frozen.finite_tree(partial)
        and partial["status"] == "FINITE",
        "Saved partial numerical diagnostics changed",
    )
    successor = evidence_tasks[1]
    restored = begins[successor]
    frozen._require(
        successor != OLD_TASK
        and restored["restored"] is True
        and restored["start_step"] == 0
        and restored["start_commit_sha256"] == markers[0]["commit_sha256"]
        and restored["process_identity"] != begin["process_identity"],
        "Later committed update lacks independent restore from original step zero",
    )
    journal_path = tx.directory / "EXECUTIONS" / successor / "COMMITS.jsonl"
    journal = frozen._lines(journal_path)
    frozen._require(
        journal
        and journal[0]["step"] == 1
        and journal[0]["new_model_calls"] == 0
        and journal[0]["reused_complete_groups"] == 16
        and journal[0]["commit_sha256"] == markers[1]["commit_sha256"],
        "Original READY was not consumed without new sampling",
    )
    inputs.extend(
        [
            _absolute_ref(journal_path),
            _absolute_ref(tx.directory / "EXECUTIONS" / successor / "BEGIN.json"),
            _absolute_ref(tx.checkpoints / "step-001/COMMIT.json"),
        ]
    )
    ledger = [
        frozen.recovery._json(raw)
        for raw in _cold_reference(refs["accounting_ref"], json_value=False).splitlines()
    ]
    forwards = [row for row in ledger if row["kind"] == "extra_forward_sequences"]
    frozen._require(
        [(row["count"], row["metadata"]) for row in forwards]
        == [
            (1, {"purpose": purpose, "completion_tokens": len(rows[index]["tokens"])})
            for index in range(27)
            for purpose in ("training_reference", "training_policy")
        ]
        and fact_by_task[OLD_TASK]["counts"]
        == {
            "completion_attempts": 128,
            "generated_tokens": sum(len(row["tokens"]) for row in rows),
            "extra_forward_sequences": 54,
            "diagnostic_gradient_calls": 48,
        },
        "Original partial physical forward accounting differs",
    )
    diagnostic_rows = [index for index in frozen.registration.DIAGNOSTIC_ROWS if index < 27]
    frozen._require(
        diagnostic_rows == list(range(16)), "Frozen partial diagnostic index mapping changed"
    )
    diagnostics = frozen.Counter((rows[index]["row_id"], 1, 3) for index in diagnostic_rows)
    return dict(
        task_id=OLD_TASK,
        step=1,
        status="REGISTERED_OOM_NO_OPTIMIZER_STEP_RECONCILED",
        source_audit_event=dict(
            path=str(tx.directory / "NUMERICAL_TOKEN_AUDIT.jsonl"),
            byte_offset=0,
            byte_count=AUDIT_BYTES,
            sha256=AUDIT_SHA256,
        ),
        original_failure_ref=contract["original_failure_ref"],
        original_ready_ref=contract["original_ready_ref"],
        step0_commit_ref=contract["step0_commit_ref"],
        successor_task_id=successor,
        restored_commit_ref=_absolute_ref(tx.checkpoints / "step-001/COMMIT.json"),
        partial_forward_count=27,
        partial_diagnostic_row_indices=diagnostic_rows,
        partial_diagnostic_calls=48,
        partial_diagnostics=partial,
        complete_numerical_pass=False,
        optimizer_updates=0,
        regenerated_responses=0,
        old_optimizer_tensor_state_observed=False,
        existing_v7_tensors_reference=tensors_ref,
        v7_tensors_reverified=False,
        literal_original_runtime_gate="REJECTED_BY_PRESERVED_OOM_NOT_OVERRIDDEN",
        inputs=inputs,
    ), diagnostics


def _verify_run_readonly(
    tx, root, *, synthetic=False, attempt_pins=None, historical_oom_proof_ref=None
):
    """Recheck persisted numerical states and deterministically rescore original raw."""
    tx._locked()
    markers = tx.published()
    frozen._require(
        [m["step"] for m in markers] == list(range(13)), "All thirteen pilot states required"
    )
    frozen.pilot_contract.for_phase(tx.identity["phase"])
    frozen._require(
        tx.identity["lane"] in (("PRIMARY", "MAIN") if synthetic else ("PRIMARY",))
        and tx.identity["max_steps"] == 12
        and (tx.identity["repeat"] == 0)
        and (tx.identity["arm"] in frozen.registration.ARMS),
        "Only a registered primary pilot run",
    )
    reg = frozen.recovery._bound(tx.directory, tx.identity["rollout_registration"])
    frozen._require(
        reg.get("identity")
        == {k: v for (k, v) in tx.identity.items() if k != "rollout_registration"},
        "Rollout registration identity differs",
    )
    config = frozen.recovery._bound(
        root, dict(path="CONFIG.json", sha256=tx.identity["config_sha256"])
    )
    manifest = frozen.recovery._bound(
        root, dict(path="DATA/MANIFEST.json", sha256=tx.identity["data_sha256"])
    )
    (public, tasks) = frozen.rollouts._registration_inputs(tx, root, reg, config, manifest)
    states = frozen.recovery._state(tx, markers[0], gpu=not synthetic)
    frozen._require(
        markers[0]["policy_sha256"] == tx.identity["reference_sha256"], "Wrong parent start"
    )
    del states
    (groups, steps, rows_by_step, expected_events) = (
        [],
        [],
        {},
        frozen.defaultdict(frozen.Counter),
    )
    (inputs, evidence_tasks) = ([], {})
    for marker in markers[1:]:
        step = marker["step"]
        state = frozen.recovery._state(tx, marker, gpu=not synthetic)
        del state
        originals = frozen.recovery._originals(tx, marker, synthetic=False)
        rng = originals["rollout_rng"]
        frozen._require(
            frozen.recovery.state_hash(rng[0]) == frozen.recovery.state_hash(rng[2])
            and frozen.recovery.state_hash(rng[1]) == frozen.recovery.state_hash(rng[-1])
            and all(
                frozen.recovery.state_hash(rng[i]) == frozen.recovery.state_hash(rng[i + 1])
                for i in range(3, 33, 2)
            ),
            "Original G8 RNG chain does not connect the committed rollout boundaries",
        )
        rollout = frozen.recovery._bound(tx.directory, marker["references"]["rollout"])
        credit = frozen.recovery._bound(tx.directory, marker["references"]["credit"])
        update = frozen._read(tx.checkpoints / f"step-{step:03d}" / "UPDATE.json")
        numeric = frozen._numeric(tx, marker, update)
        expected_qids = [f"{rid}_P1_v{v}" for rid in reg["schedule"][step - 1] for v in range(4)]
        frozen._require(
            rollout.get("registration_ref") == tx.identity["rollout_registration"],
            "Unregistered rollout source",
        )
        rows_by_step[step] = rollout["records"]
        for index, qid in enumerate(expected_qids):
            ref = rollout["complete_group_references"][index]
            expected_path = f"rollouts/step-{step:03d}/groups/{index:02d}/COMPLETE.json"
            frozen._require(ref["path"] == expected_path, "Original group path differs")
            complete = frozen.recovery._bound(tx.directory, ref)
            raw = [
                frozen.recovery._json(line)
                for line in frozen.recovery._bound(
                    tx.directory, complete["raw_reference"], json_value=False
                ).splitlines()
            ]
            frozen._require(
                complete["raw_reference"]["path"]
                == expected_path.replace("COMPLETE.json", "raw.jsonl"),
                "Original raw path differs",
            )
            frozen._require(
                len({r.get("task_id") for r in raw}) == 1, "G8 mixed physical generator"
            )
            task_id = raw[0]["task_id"]
            frozen._require(
                isinstance(task_id, str) and frozen.re.fullmatch(frozen.TASK_PATTERN, task_id),
                "Missing raw task",
            )
            runtime = frozen.SimpleNamespace(
                identity={
                    **frozen._runtime(tx, root, task_id, synthetic=synthetic),
                    "step": step - 1,
                }
            )
            seeds = reg["draw_seeds"][step - 1][index]
            (safe, task) = (public[qid]["model_input"], tasks[qid])
            step_spec = dict(
                format=frozen.rollouts.FORMAT,
                identity=tx.identity,
                step=step,
                registration_ref=tx.identity["rollout_registration"],
                policy_before_sha256=marker["policy_before_sha256"],
                roots=reg["schedule"][step - 1],
            )
            expected_spec = dict(
                step_spec_sha256=frozen.digest(step_spec),
                group_index=index,
                qid=qid,
                seeds=seeds,
                model_input_sha256=frozen.digest(safe),
                task_sha256=frozen.digest(task),
            )
            group_started = frozen._read(
                tx.directory / expected_path.replace("COMPLETE.json", "STARTED.json")
            )
            frozen._require(
                complete.get("spec") == group_started.get("spec") == expected_spec
                and group_started.get("rng_before_reference") == complete["rng_before_reference"],
                "Original group STARTED/COMPLETE registration differs",
            )
            frozen.rollouts._validate_raw(
                raw, runtime, safe, seeds, marker["policy_before_sha256"], root
            )
            annotated = frozen.rollouts._annotate(raw, task, tx, runtime, step, index)
            frozen._require(
                annotated == rollout["records"][index * 8 : (index + 1) * 8],
                "Original raw/scorer/task metadata differs from committed rows",
            )
            actual = frozen.build_credit_group(
                annotated, tx.identity["arm"], reg["shuffle_seeds"][step - 1][index]
            )
            frozen._require(
                actual == credit["groups"][index],
                "Applied credit differs from frozen recomputation",
            )
            selected = actual["counterfactuals"][frozen.ARMS[tx.identity["arm"]][1]]
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
            expected_events[task_id]["completion_attempts", 8, seeds[0], "structured"] += 1
            for d, row in enumerate(raw):
                expected_events[task_id]["generated_tokens", len(row["tokens"]), seeds[0], d] += 1
        evidence_path = frozen.Path(marker["references"]["evidence"]["path"])
        frozen._require(
            evidence_path.parent.as_posix() == f"PREUPDATE_EVIDENCE/step-{step:03d}"
            and evidence_path.suffix == ".json",
            "Actual update task evidence path required",
        )
        evidence_tasks[step] = evidence_path.stem
        inputs.append(frozen.recovery._ref(root, tx.checkpoints / f"step-{step:03d}/COMMIT.json"))
        steps.append(dict(step=step, committed_update=marker["commit_sha256"], numerical=numeric))
    frozen._require(
        {p.name for p in (tx.directory / "rollouts").iterdir()}
        == {f"step-{s:03d}" for s in range(1, 13)},
        "Extra/partial rollout step preserved; audit blocked",
    )
    for step in range(1, 13):
        directory = frozen._plain_path(tx.directory / "rollouts" / f"step-{step:03d}" / "groups")
        frozen._require(
            {p.name for p in directory.iterdir()} == {f"{g:02d}" for g in range(16)},
            "Extra/partial raw group inventory",
        )
        for path in directory.iterdir():
            frozen._plain_path(path)
            frozen._require(
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
        historical_oom_proof_ref=historical_oom_proof_ref,
    )
    return dict(
        format=frozen.FORMAT,
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
    historical_oom_proof_ref=None,
):
    """All PRIMARY attempts, including failures; only original committed G8 closes calls.

    A missing END is retained. It is reconciled only when its positive calls are
    all original complete groups and a later restored process consumed/passed its
    work. Empty unclosed attempts have no such proof and remain UNKNOWN.
    """
    tx._locked()
    attempts = frozen._members(tx.directory / "TRAINER_ATTEMPTS")
    journals = frozen._members(tx.directory / "EXECUTIONS")
    accounts = frozen._members(tx.directory / "accounting", files=True, suffix=".jsonl")
    runtimes = frozen._members(tx.directory / "runtime_identity", files=True, suffix=".json")
    tasks = set(attempts) | set(journals) | set(accounts) | set(runtimes) | set(expected_events)
    frozen._require(tasks, "No physical execution inventory")
    if not synthetic:
        frozen._require(
            set(attempts) == tasks, "Orphan execution/accounting/runtime without STARTED"
        )
        frozen._require(
            isinstance(attempt_pins, dict)
            and set(attempt_pins)
            == {"context", "phase_identity_sha256", "pilot_spec_sha256", "admission_ref"},
            "Actual admitted attempt provenance required",
        )
    (inputs, facts, begins) = ([], [], {})
    diagnostic_events = frozen.defaultdict(frozen.Counter)
    for task, folder in journals.items():
        begin = frozen._read(folder / "BEGIN.json")
        frozen._require(
            begin.get("format") == frozen.recovery.FORMAT
            and begin.get("status") == "BEGIN"
            and (begin.get("recovery_binding_ref") is None)
            and (begin.get("identity") == tx.identity)
            and (begin.get("task_id") == task)
            and (begin.get("scope") == ("SYNTHETIC_ONLY" if synthetic else "SERVER_ACTUAL"))
            and (begin.get("storage_lane") == "PRIMARY")
            and (type(begin.get("start_step")) is int)
            and (begin["start_step"] in range(13))
            and (begin.get("start_commit_sha256") == markers[begin["start_step"]]["commit_sha256"]),
            "Execution BEGIN does not bind a real committed source",
        )
        frozen._require(
            frozen.recovery._bound(root, begin["runtime_identity_reference"])
            == frozen._runtime(tx, root, task, synthetic=synthetic)
            and begin["runtime_identity_reference"]["path"]
            == (tx.directory / "runtime_identity" / f"{task}.json").relative_to(root).as_posix(),
            "Execution runtime reference differs",
        )
        frozen._require(
            type(begin.get("restored")) is bool and type(begin.get("resumed_after_data")) is bool,
            "Explicit restore facts required",
        )
        if not synthetic:
            frozen._require(
                begin.get("process_identity", {}).get("kind") == "LINUX_PROC",
                "Actual process identity required",
            )
        begins[task] = begin
    for task in sorted(tasks):
        if not synthetic and task in runtimes:
            frozen.recovery._overlay_identity(
                frozen._runtime(tx, root, task, synthetic=False),
                tx.identity,
                pilot_spec_sha256=attempt_pins["pilot_spec_sha256"],
            )
        for mapping in (attempts, journals):
            if task in mapping:
                for path in mapping[task].iterdir():
                    frozen._plain_path(path)
                    frozen._require(path.is_file(), "Unexpected attempt/journal object")
                    inputs.append(frozen.recovery._ref(root, path))
        for mapping in (accounts, runtimes):
            if task in mapping:
                inputs.append(frozen.recovery._ref(root, mapping[task]))
        (start, terminal) = (None, None)
        if task in attempts:
            folder = attempts[task]
            frozen._require(
                {p.name for p in folder.iterdir()}
                <= {"STARTED.json", "COMPLETE.json", "FAILURE.json"}
                and (folder / "STARTED.json").exists()
                and (
                    not ((folder / "COMPLETE.json").exists() and (folder / "FAILURE.json").exists())
                ),
                "Conflicting/unknown trainer attempt terminal",
            )
            start = frozen._read(folder / "STARTED.json")
            frozen._require(
                start.get("status") == "STARTED"
                and start.get("task_id") == task
                and (start.get("run_id") == tx.identity["run_id"])
                and (start.get("parent") == tx.identity["parent"])
                and (start.get("arm") == tx.identity["arm"])
                and (start.get("storage_lane") == "PRIMARY"),
                "Trainer STARTED identity differs",
            )
            if not synthetic:
                frozen._require(
                    start.get("execution_origin") == "SERVER_ACTUAL"
                    and start.get("phase_identity_sha256") == tx.identity["phase_identity_sha256"]
                    and all(start.get(k) == v for (k, v) in attempt_pins.items())
                    and (start.get("recovery_binding") is None),
                    "Trainer phase provenance differs",
                )
            if task in begins:
                frozen._require(
                    start.get("initial_published_step") == begins[task]["start_step"]
                    or (
                        start.get("initial_published_step") is None
                        and begins[task]["start_step"] == 0
                    ),
                    "Trainer attempt initial commit differs from actual BEGIN",
                )
            for name in ("COMPLETE", "FAILURE"):
                if (folder / (name + ".json")).exists():
                    terminal = frozen._read(folder / (name + ".json"))
                    frozen._require(
                        terminal.get("task_id") == task, "Attempt terminal identity differs"
                    )
                    if name == "FAILURE":
                        frozen._require(
                            terminal.get("status") == "FAILED_PRESERVED"
                            and terminal.get("sampling_retry_authorized") is False
                            and all(
                                terminal.get(k) == v for (k, v) in start.items() if k != "status"
                            ),
                            "Failure cannot authorize new sampling",
                        )
                    else:
                        frozen._require(
                            terminal.get("status") == "COMPLETED_NOT_GATE_SIGNED"
                            and terminal.get("run_id") == tx.identity["run_id"]
                            and (terminal.get("storage_lane") == "PRIMARY")
                            and (
                                terminal.get("committed_step")
                                == terminal.get("logical_updates")
                                == 12
                            )
                            and (terminal.get("training_authorized") is False),
                            "Trainer COMPLETE identity/budget differs",
                        )
        events = frozen._lines(accounts[task]) if task in accounts else []
        actual = frozen.Counter()
        counts = frozen.Counter()
        for event in events:
            frozen._require(
                event.get("task_id") == task and event.get("parent") == tx.identity["parent"],
                "Accounting belongs to another task/parent",
            )
            key = frozen._event_key(event)
            counts[event["kind"]] += event["count"]
            if key is not None:
                actual[key] += 1
            if event["kind"] == "diagnostic_gradient_calls":
                meta = event["metadata"]
                frozen._require(
                    set(meta) == {"row_id", "step"}
                    and event["count"] == 3
                    and (type(meta["step"]) is int)
                    and isinstance(meta["row_id"], str),
                    "Fixed three-channel diagnostic call shape differs",
                )
                diagnostic_events[task][meta["row_id"], meta["step"], 3] += 1
        frozen._require(
            actual == expected_events.get(task, frozen.Counter()),
            "All attempts must match original G8/draw accounting exactly; "
            "extra/unknown calls remain",
        )
        n = counts["completion_attempts"]
        frozen._require(not n or task in begins, "Data call without execution BEGIN")
        if task in runtimes:
            frozen._runtime(tx, root, task, synthetic=synthetic)
        status = (
            "FAILED_BEFORE_DATA"
            if terminal and terminal.get("status") == "FAILED_PRESERVED"
            else "UNKNOWN"
        )
        journal_steps = []
        if task in journals:
            (folder, begin) = (journals[task], begins[task])
            frozen._require(
                {p.name for p in folder.iterdir()}
                <= {"BEGIN.json", "COMMITS.jsonl", "END.json", "FAILURE.json"}
                and (not ((folder / "END.json").exists() and (folder / "FAILURE.json").exists())),
                "Conflicting/unknown execution terminal",
            )
            entries = frozen._lines(folder / "COMMITS.jsonl")
            journal_steps = [v["step"] for v in entries]
            frozen._require(
                journal_steps
                == list(range(begin["start_step"] + 1, begin["start_step"] + len(entries) + 1)),
                "Duplicate/non-prefix execution COMMIT journal",
            )
            for entry in entries:
                s = entry["step"]
                frozen._require(
                    s in range(1, 13) and evidence_tasks[s] == task,
                    "Journal claims another updater's COMMIT",
                )
                frozen._require(
                    frozen.recovery._bound(root, entry["commit_reference"]) == markers[s]
                    and entry["commit_sha256"] == markers[s]["commit_sha256"]
                    and (entry["rollout_reference"] == markers[s]["references"]["rollout"]),
                    "Journal commit binding differs",
                )
                new = sum(r["task_id"] == task for r in rows_by_step[s])
                frozen._require(
                    entry["new_model_calls"] == new
                    and entry["reused_complete_groups"] == 16 - new // 8,
                    "Journal new/reused accounting differs",
                )
            if (folder / "END.json").exists():
                frozen.recovery._execution(
                    root,
                    frozen.recovery._ref(root, folder / "END.json"),
                    tx.directory,
                    tx.identity,
                    pilot_spec_sha256=(attempt_pins or {}).get("pilot_spec_sha256"),
                )
                status = "SEGMENT_COMPLETE"
                if terminal and terminal.get("status") != "FAILED_PRESERVED":
                    frozen._require(
                        terminal.get("execution") == frozen.recovery._ref(root, folder / "END.json")
                        and terminal.get("final_commit")
                        == frozen.recovery._ref(root, tx.checkpoints / "step-012/COMMIT.json"),
                        "Trainer COMPLETE does not bind actual END/COMMIT",
                    )
            else:
                data_steps = [
                    s
                    for (s, rows) in rows_by_step.items()
                    if any(r["task_id"] == task for r in rows)
                ]
                consumed = [s for (s, owner) in evidence_tasks.items() if owner == task]
                boundary = max(consumed or [begin["start_step"]])
                successor = any(
                    other != task and b["restored"] and (b["start_step"] >= boundary)
                    for (other, b) in begins.items()
                )
                update_calls = sum(
                    counts[k]
                    for k in frozen.KNOWN_ACCOUNTING
                    if k not in ("completion_attempts", "generated_tokens")
                )
                if successor and ((n and data_steps) or update_calls):
                    status = "INTERRUPTED_RECONCILED_WITH_ORIGINALS"
                elif terminal and terminal.get("status") == "FAILED_PRESERVED" and (not events):
                    status = "FAILED_BEFORE_DATA"
                else:
                    status = "UNKNOWN"
        frozen._require(
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
                execution_terminal="END"
                if task in journals and (journals[task] / "END.json").exists()
                else "FAILURE"
                if task in journals and (journals[task] / "FAILURE.json").exists()
                else None,
            )
        )
    frozen._require(
        set(evidence_tasks.values()) <= set(begins), "Committed updater has no physical BEGIN"
    )
    for step, task in evidence_tasks.items():
        frozen._require(begins[task]["start_step"] < step, "Update predates its restore boundary")
    audit_path = tx.directory / "NUMERICAL_TOKEN_AUDIT.jsonl"
    audit_events = frozen._lines(audit_path)
    bound_slices = set()
    physical_evidence = []
    for step_dir in (tx.directory / "PREUPDATE_EVIDENCE").iterdir():
        frozen._plain_path(step_dir)
        frozen._require(
            frozen.re.fullmatch("step-0(?:0[1-9]|1[0-2])", step_dir.name), "Unknown numerical step"
        )
        for path in step_dir.iterdir():
            frozen._plain_path(path)
            evidence = frozen._read(path)
            frozen._require(
                path.stem in begins and path.suffix == ".json", "Unknown numerical updater"
            )
            frozen._require(
                evidence.get("task_id") == path.stem,
                "Numerical event physical producer differs from wrapper path",
            )
            frozen.recovery._audit_event(tx.directory, evidence, required=True)
            ref = evidence["source_audit_event"]
            frozen._require(
                ref["byte_offset"] not in bound_slices, "Duplicated original numerical event"
            )
            bound_slices.add(ref["byte_offset"])
            physical_evidence.append(
                dict(
                    task_id=path.stem,
                    step=evidence["committed_source_step"] + 1,
                    reference=frozen.recovery._ref(root, path),
                )
            )
            inputs.append(frozen.recovery._ref(root, path))
    offset = 0
    physical_pairs = set()
    failed_numeric_events = []
    partial_diagnostics = frozen.defaultdict(frozen.Counter)
    fact_by_task = {item["task_id"]: item for item in facts}
    for line, event in zip(
        frozen._read_bytes(audit_path).splitlines(keepends=True), audit_events, strict=True
    ):
        if event.get("status") != "PREUPDATE_PASS":
            frozen._require(not failed_numeric_events, "Only one registered old OOM exists")
            reconciled, diagnostic_calls = _reconcile_registered_oom(
                tx,
                root=root,
                event=event,
                line=line,
                offset=offset,
                markers=markers,
                begins=begins,
                fact_by_task=fact_by_task,
                rows_by_step=rows_by_step,
                evidence_tasks=evidence_tasks,
                bound_slices=bound_slices,
                proof_ref=historical_oom_proof_ref,
                synthetic=synthetic,
            )
            failed_numeric_events.append(reconciled)
            partial_diagnostics[reconciled["task_id"]].update(diagnostic_calls)
            inputs.extend(reconciled["inputs"])
            offset += len(line)
            continue
        frozen._require(
            event.get("status") == "PREUPDATE_PASS",
            "Failed numerical attempt remains in inventory; cannot assert runtime gate",
        )
        owner = event.get("task_id")
        source_step = event.get("committed_source_step")
        frozen._require(
            owner in begins and type(source_step) is int and (source_step in range(12)),
            "Actual numerical event must name its original physical task and step",
        )
        step = source_step + 1
        frozen._require(
            (owner, step) not in physical_pairs, "Repeated physical task/step numerical event"
        )
        physical_pairs.add((owner, step))
        if offset not in bound_slices:
            committed = frozen.recovery._bound(
                tx.directory, markers[step]["references"]["evidence"]
            )
            frozen._require(
                {k: v for (k, v) in event.items() if k not in ("at", "task_id")}
                == {
                    k: v
                    for (k, v) in committed.items()
                    if k not in ("at", "task_id", "source_audit_event")
                }
                and offset < committed["source_audit_event"]["byte_offset"],
                "Unbound preupdate does not exactly match restored original-data replay",
            )
            fact = fact_by_task[owner]
            frozen._require(
                fact["status"] == "INTERRUPTED_RECONCILED_WITH_ORIGINALS"
                and step == max(fact["journal_steps"] or [begins[owner]["start_step"]]) + 1
                and (evidence_tasks[step] != owner),
                "Uncommitted preupdate lacks its task's restore/physical attempt proof",
            )
            physical_evidence.append(
                dict(
                    task_id=owner,
                    step=step,
                    status="UNCOMMITTED_PREUPDATE_RECONCILED",
                    byte_offset=offset,
                    byte_count=len(line),
                    sha256=frozen.hashlib.sha256(line).hexdigest(),
                    matched_committed_evidence=markers[step]["references"]["evidence"],
                )
            )
        offset += len(line)
    for fact in facts:
        task = fact["task_id"]
        expected_diagnostics = frozen.Counter(
            (rows_by_step[step][row]["row_id"], step, 3)
            for (owner, step) in physical_pairs
            if owner == task and step in (1, 4, 8, 12)
            for row in frozen.registration.DIAGNOSTIC_ROWS
        )
        expected_diagnostics.update(partial_diagnostics[task])
        frozen._require(
            diagnostic_events[task] == expected_diagnostics,
            "Physical diagnostic calls differ from each actual fixed-subset update attempt",
        )
        if fact["status"] == "INTERRUPTED_RECONCILED_WITH_ORIGINALS" and (
            not fact["counts"].get("completion_attempts", 0)
        ):
            frozen._require(
                any(owner == fact["task_id"] for (owner, _) in physical_pairs),
                "Cache-only interrupted update has no attributable original numerical event",
            )
    inputs.append(frozen.recovery._ref(root, audit_path))
    frozen._require(
        sum(f["counts"].get("completion_attempts", 0) for f in facts) == 1536,
        "Primary physical generation total must be exactly1536 without replacement",
    )
    result = dict(
        status="ORIGINAL_FACTS_RECONCILED",
        attempts=facts,
        actual_generated_responses=1536,
        physical_preupdate_events=physical_evidence,
        physical_preupdate_count=len(audit_events),
        inputs=inputs,
    )
    if failed_numeric_events:
        result.update(
            failed_numeric_events=failed_numeric_events,
            physical_numeric_event_count=len(audit_events),
            physical_preupdate_count=len(audit_events) - len(failed_numeric_events),
            partial_diagnostic_calls=sum(
                item["partial_diagnostic_calls"] for item in failed_numeric_events
            ),
            literal_original_runtime_gate="REJECTED_BY_PRESERVED_OOM_NOT_OVERRIDDEN",
            status="COMMITTED_FACTS_RECONCILED_WITH_REGISTERED_OOM_PRESERVED",
        )
    return result


def verify_primary_run(
    tx, root, *, synthetic=False, attempt_pins=None, historical_oom_proof_ref=None
):
    target = (
        tx.identity.get("phase") == "PILOT_V3"
        and tx.identity.get("parent") == "b1"
        and tx.identity.get("arm") == "U10"
    )
    if not target:
        return frozen._verify_run(tx, root, synthetic=synthetic, attempt_pins=attempt_pins)
    result = _verify_run_readonly(
        tx,
        root,
        synthetic=synthetic,
        attempt_pins=attempt_pins,
        historical_oom_proof_ref=historical_oom_proof_ref,
    )
    failures = result["inventory"].get("failed_numeric_events", [])
    frozen._require(
        len(failures) == 1 and failures[0]["task_id"] == OLD_TASK,
        "The fixed registered old OOM must remain present exactly once",
    )
    return result
