"""New-consumer original-24 T03 execution and complete evidence verification."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import re
import stat
from pathlib import Path

from sr_f6 import numerical_replay as kernel
from sr_f6.common import digest, file_hash, now, read, safe_evidence, write_once
from sr_f6.replay import RULES
from sr_f6.transactions import _plain_path

from .t03_access import require

PARENTS = ("b0", "b1", "b2")
SPEC_PATH = "T03/SPEC.json"
SCHEMA = "srf6-confirm-followup-t03-v1"


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _path(root, relative):
    root = _plain_path(root)
    relative = Path(relative)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Relative consumer output required",
    )
    path = _plain_path(root / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ref(root, relative):
    path = safe_evidence(root, relative)
    return dict(path=relative, sha256=file_hash(path))


def selected_parent(context, parent):
    require(parent in PARENTS, "Unregistered T03 parent")
    producer = context["producer_context"]
    selection = producer["spec"]["followup_requirements"]["numerical_replay_selection"]
    qid = sorted(row["qid"] for row in producer["public"])[0]
    require(
        selection["qid"] == qid
        and selection["parents"] == list(PARENTS)
        and selection["draws"] == list(range(8))
        and selection["sequences"] == 24
        and selection["new_generated_responses"] == selection["optimizer_updates"] == 0,
        "Original confirmation T03 selection changed",
    )
    records = sorted(
        (row for row in context["records"] if row["parent"] == parent and row["qid"] == qid),
        key=lambda row: row["draw"],
    )
    require(
        len(records) == 8 and [row["draw"] for row in records] == list(range(8)),
        "Complete original first G8 required",
    )
    raw = [context["raw_by_row"][row["row_id"]] for row in records]
    policy = context["evidence"]["parent_tensor_hashes"][parent]
    for draw, (record, original) in enumerate(zip(records, raw, strict=True)):
        require(
            record["row_id"] == f"{parent}/{qid}/{draw}"
            and original["phase"] == record["phase"] == "SCHEMA_CONFIRM_V3"
            and original["policy_tensor_hash"] == record["policy_tensor_hash"] == policy
            and original["source_sha256"] == producer["registration"]["source_manifest_sha256"]
            and original["registration_sha256"] == producer["registration_sha256"]
            and original["tokens"] == record["tokens"]
            and original["sampler_logprobs"] == record["sampler_logprobs"],
            "Original confirmed trajectory identity differs",
        )
    return copy.deepcopy(records), copy.deepcopy(raw), policy


def build_spec(context):
    producer = context["producer_context"]
    identity = context["registration"]["producer"]
    require(
        Path(kernel.__file__).resolve()
        == Path(identity["deployment"]) / "src/sr_f6/numerical_replay.py",
        "Kernel must come from the frozen producer",
    )
    groups = {}
    for parent in PARENTS:
        records, raw, policy = selected_parent(context, parent)
        groups[parent] = dict(
            qid=records[0]["qid"],
            row_ids=[row["row_id"] for row in records],
            records_sha256=digest(records),
            raw_sha256=digest(raw),
            policy_tensor_hash=policy,
        )
    review = context["review"]
    rules = dict(
        RULES,
        selection="first qid in sorted frozen DEV_SCHEMA_CONFIRM public; all original draws 0..7",
    )
    return _jsonable(
        dict(
            schema=SCHEMA,
            phase="CONFIRM_T03_V1",
            context=dict(
                producer=identity,
                consumer=context["consumer"],
                support_binding=context["support_binding_ref"],
                review_package=review["package_ref"],
                review_audit=review["audit_ref"],
            ),
            rules=rules,
            groups=groups,
            generated_responses=0,
            optimizer_updates=0,
            training_authorized=False,
            generation_authorized=False,
            kernel=dict(
                module="sr_f6.numerical_replay",
                path=str(Path(kernel.__file__).resolve()),
                source_manifest_sha256=producer["registration"]["source_manifest_sha256"],
            ),
            inputs=[
                dict(
                    path=str(Path(context["root"]) / "REGISTRATION.json"),
                    sha256=context["consumer"]["registration_sha256"],
                ),
                context["support_binding_ref"],
                review["package_ref"],
                review["audit_ref"],
            ],
        )
    )


def freeze_spec(root):
    from .source import require_support

    context = require_support(root, purpose="t03")
    spec = build_spec(context)
    write_once(_path(root, SPEC_PATH), spec)
    return spec


def verified_spec(root, context):
    spec = read(safe_evidence(root, SPEC_PATH))
    require(spec == build_spec(context), "Frozen T03 source/input/rules changed")
    return spec, _ref(root, SPEC_PATH)


def run(root, parent, task_id):
    import torch

    from mm_dev.runtime import actual_cuda_identity

    from . import source, t03_access, t03_runtime

    require(
        parent in PARENTS
        and isinstance(task_id, str)
        and re.fullmatch(rf"LH-SRF6-T03-[0-9]+-{parent}", task_id),
        "Registered parent/job task identifier required",
    )
    context = source.require_support(root, purpose="t03")
    root = Path(context["root"])
    spec, spec_ref = verified_spec(root, context)
    from .t03_dispatch import submission_receipt

    receipt = submission_receipt(root, context, spec_ref)
    require(
        task_id == f"LH-SRF6-T03-{receipt['job_id']}-{parent}",
        "Task does not match actual submission",
    )
    failures = _path(root, "T03/FAILURES/.sentinel").parent
    require(not list(failures.iterdir()), "Prior failed T03 attempt blocks further parents")
    slot = _path(root, f"T03/SLOTS/{parent}.json")
    require(not slot.exists(), "Preserve prior parent attempt; no automatic retry")
    audit = _path(root, f"T03/RUNS/{task_id}/.sentinel").parent
    require(not list(audit.iterdir()), "Preserve orphan replay execution")
    with _path(root, f"T03/LOCKS/{parent}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not slot.exists(), "Parent slot already consumed")
        attempt = dict(
            at=now(),
            phase="CONFIRM_T03_V1",
            parent=parent,
            task_id=task_id,
            spec_ref=spec_ref,
            consumer=spec["context"]["consumer"],
            producer=spec["context"]["producer"],
            generated_responses=0,
            optimizer_updates=0,
        )
        write_once(slot, attempt)
        try:
            write_once(audit / "ATTEMPT.json", attempt)
            runtime, actual = t03_runtime.load(root, parent, task_id, audit, spec=spec)
            require(
                torch.cuda.is_available()
                and torch.cuda.device_count() == 1
                and all(p.device.type == "cuda" for p in runtime.model.parameters()),
                "Actual CUDA runtime required",
            )
            attempt_ref = _ref(root, str((audit / "ATTEMPT.json").relative_to(root)))
            with t03_access.open_replay(runtime, actual, spec, spec_ref) as session:
                require(
                    type(session) is t03_access.ReplaySession,
                    "Authenticated new-consumer session required",
                )
                session.assert_active()
                report = kernel._execute(
                    session, hardware=actual_cuda_identity(), attempt_ref=attempt_ref
                )
            if report["status"] != "PASS":
                write_once(
                    failures / f"{task_id}.json",
                    dict(
                        attempt,
                        status=report["status"],
                        result_ref=_ref(root, f"T03/RUNS/{task_id}/T03/RESULT.json"),
                    ),
                )
            return report
        except BaseException as exc:
            failure = dict(
                attempt,
                status="TECHNICAL_FAILURE_STOP",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            write_once(audit / "FAILURE.json", failure)
            write_once(failures / f"{task_id}.json", failure)
            raise


def _validate_boundaries(report, raw, runtime):
    native = runtime["generation_config_expanded"]
    eos = native["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    pad = native["pad_token_id"]
    require(
        report.get("native_special_tokens") == dict(eos_ids=eos, pad_token_id=pad),
        "Native special token identities differ",
    )
    require(len(report.get("boundaries", [])) == 8, "Complete boundary evidence required")
    for boundary, original in zip(report["boundaries"], raw, strict=True):
        tokens, backend = original["tokens"], original["backend_sequence_tokens"]
        count = len(tokens)
        positions = [i for i, token in enumerate(tokens) if token in eos]
        require(
            tokens == original["raw_tokens"]
            and backend[:count] == tokens
            and original["completion_token_count"] == count
            and original["backend_padding_count"] == len(backend) - count
            and all(t == pad for t in backend[count:])
            and (not positions or positions == [count - 1]),
            "Original stop/padding differs",
        )
        finish = original["finish_reason"]
        require(
            finish in ("balanced", "eos", "length")
            and (finish != "eos" or positions)
            and (finish != "length" or count == 1536),
            "Original finish boundary differs",
        )
        expected = dict(
            token_count=count,
            prompt_token_count=original["prompt_token_count"],
            prompt_masked_positions=original["native_prompt_inputs"]["attention_mask"][0].count(0),
            backend_padding_count=len(backend) - count,
            finish_reason=finish,
            eos_positions=positions,
            eos_included=bool(positions),
            native_prompt_inputs_verified=True,
            original_completion_ids_verified=True,
            padding_excluded=True,
            prompt_excluded=True,
            boundary_scope="Exact native prompt and unpadded completion-vector length checked",
        )
        require(boundary == expected, "Reconstructed original boundary differs")


def _validate_result(root, context, spec, spec_ref, task_id, attempt, report):
    parent = attempt["parent"]
    records, raw, policy = selected_parent(context, parent)
    audit_rel = f"T03/RUNS/{task_id}"
    runtime_ref = _ref(root, f"{audit_rel}/runtime_identity/{task_id}.json")
    runtime = read(root / runtime_ref["path"])
    expected = dict(
        phase="CONFIRM_T03_V1",
        execution_mode="REPLAY",
        run_root=str(root),
        audit_root=str(root / audit_rel),
        input_root=str(context["producer_context"]["root"]),
        parent_id=parent,
        reference_parent=parent,
        task_id=task_id,
        run_id=task_id,
        step=0,
        policy_hash=policy,
        reference_hash=policy,
        fresh_optimizer=False,
        optimizer_updates=0,
        generated_responses=0,
        dtype="float32",
        probability_dtype="float32",
        consumer=spec["context"]["consumer"],
        producer=spec["context"]["producer"],
    )
    require(
        all(type(runtime.get(k)) is type(v) and runtime[k] == v for k, v in expected.items()),
        "Actual T03 runtime identity differs",
    )
    require(
        runtime.get("optimizer") is None and runtime.get("scheduler") is None,
        "T03 optimizer forbidden",
    )
    hardware = runtime.get("hardware", {})
    require(
        report.get("hardware") == hardware
        and hardware.get("cuda_visible_device_count") == 1
        and bool(hardware.get("cuda_uuid"))
        and "RTX PRO 6000" in hardware.get("cuda_name", "").upper(),
        "Actual single PRO6000 GPU receipt required",
    )
    expected_report = dict(
        schema=kernel.SCHEMA,
        scope="SERVER_ACTUAL",
        parent=parent,
        task_id=task_id,
        context=spec["context"],
        spec_ref=spec_ref,
        inputs=spec["inputs"],
        selected_records_sha256=digest(records),
        selected_raw_records_sha256=digest(raw),
        qid=records[0]["qid"],
        row_ids=[r["row_id"] for r in records],
        generated_responses=0,
        optimizer_updates=0,
        training_authorized=False,
        gate_status="NOT_SIGNED",
        tolerances=dict(atol=kernel.ATOL, rtol=kernel.RTOL),
        execution_attempt_ref=_ref(root, f"{audit_rel}/ATTEMPT.json"),
    )
    require(
        all(report.get(k) == v for k, v in expected_report.items()),
        "Original T03 result provenance differs",
    )
    _validate_boundaries(report, raw, runtime)
    require(
        report.get("parameter_inventory") == runtime.get("trainable_parameters"),
        "Actual complete adapter inventory differs",
    )
    scores, advantages = kernel._advantages(records)
    require(
        report.get("J") == scores and report.get("native_advantages") == advantages,
        "Native J advantages changed",
    )
    require(len(report.get("probability_vectors", [])) == 8, "Complete original vectors required")
    for trace, row in zip(report["probability_vectors"], raw, strict=True):
        require(
            trace["row_id"] == row["row_id"]
            and trace["qid"] == row["qid"]
            and trace["tokens"] == row["tokens"]
            and trace["sampler_logprobs"] == row["sampler_logprobs"],
            "Original tokens/probabilities changed",
        )
    ledger_ref = _ref(root, f"{audit_rel}/accounting/{task_id}.jsonl")
    events = [kernel._json(line) for line in (root / ledger_ref["path"]).read_bytes().splitlines()]
    require(
        report["accounting"].get("reference") == ledger_ref
        and report["accounting"].get("events") == events,
        "Terminal replay accounting changed",
    )
    require(len(events) == 16, "Complete sixteen-forward accounting required")
    for index, event in enumerate(events):
        require(
            event["metadata"].get("row_id") == records[index // 2]["row_id"]
            and event["metadata"].get("qid") == records[index // 2]["qid"]
            and event["metadata"].get("consumer_registration_sha256")
            == context["consumer"]["registration_sha256"]
            and event["metadata"].get("producer_registration_sha256")
            == context["producer_context"]["registration_sha256"],
            "Replay accounting provenance differs",
        )
    start_ref = _ref(root, f"{audit_rel}/T03/STARTED.json")
    start = read(root / start_ref["path"])
    require(
        all(report.get(k) == value for k, value in start.items() if k != "errors"),
        "Result differs from pre-forward STARTED identity",
    )
    require(
        not report.get("errors"),
        "Incomplete/exceptional numerical execution requires technical review",
    )
    require(
        report["restoration"]["policy_before_sha256"] == policy,
        "Replay policy differs from confirmed parent",
    )
    facts = kernel.assess_facts(report)
    require(
        report.get("assessment") == facts and report.get("status") == facts["status"],
        "Recorded numerical status differs from original kernel recomputation",
    )
    return facts, [runtime_ref, ledger_ref, start_ref]


def _tree_snapshot(root):
    """Stat-only closure; no repeated model/data content scans."""
    directory = _plain_path(Path(root) / "T03")
    snapshot = {}
    if not directory.exists():
        return snapshot
    for path in sorted(directory.rglob("*")):
        _plain_path(path)
        st = path.lstat()
        require(stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode), "Nonregular T03 evidence")
        snapshot[path.relative_to(directory).as_posix()] = (
            "dir" if path.is_dir() else "file",
            st.st_dev,
            st.st_ino,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        )
    return snapshot


def _allowed_members(tasks):
    leaves = {
        "SPEC.json",
        "SUBMISSIONS/dispatch.lock",
        "SUBMISSIONS/t03.attempt.json",
        "SUBMISSIONS/t03.result.json",
    }
    for task, parent in tasks.items():
        leaves.update({f"SLOTS/{parent}.json", f"LOCKS/{parent}.lock", f"FAILURES/{task}.json"})
        leaves.update(
            f"RUNS/{task}/{name}"
            for name in (
                "ATTEMPT.json",
                "FAILURE.json",
                ".session.lock",
                "T03/STARTED.json",
                "T03/RESULT.json",
                f"runtime_identity/{task}.json",
                f"accounting/{task}.jsonl",
            )
        )
    directories = {"SUBMISSIONS", "RUNS", "SLOTS", "LOCKS", "FAILURES"}
    for leaf in leaves:
        directories.update(str(p) for p in Path(leaf).parents if str(p) != ".")
    return leaves, directories


def verify_completed(root):
    """Reconstruct every attempt; technical uncertainty never becomes scientific FAIL."""
    from .source import require_support
    from .t03_dispatch import submission_receipt

    context = require_support(root, purpose="t03")
    root = Path(context["root"])
    spec, spec_ref = verified_spec(root, context)
    parents = {p: dict(status="INCOMPLETE", result_ref=None) for p in PARENTS}
    inputs, errors, attempts, tasks = [spec_ref], [], [], {}
    before = _tree_snapshot(root)
    try:
        receipt = submission_receipt(root, context, spec_ref)
        inputs.extend(
            _ref(root, f"T03/SUBMISSIONS/t03.{kind}.json") for kind in ("attempt", "result")
        )
    except (OSError, KeyError, ValueError, TypeError, PermissionError) as exc:
        receipt = None
        errors.append(dict(error=f"Submission: {type(exc).__name__}: {exc}"))
    runs = root / "T03/RUNS"
    seen = set()
    for audit in sorted(runs.iterdir()) if runs.exists() else []:
        if not audit.is_dir() or audit.is_symlink():
            errors.append(dict(error=f"Unexpected run member: {audit.name}"))
            continue
        task = audit.name
        try:
            attempt_ref = _ref(root, f"T03/RUNS/{task}/ATTEMPT.json")
            attempt = read(audit / "ATTEMPT.json")
            parent = attempt["parent"]
            require(
                parent in PARENTS
                and parent not in seen
                and re.fullmatch(rf"LH-SRF6-T03-[0-9]+-{parent}", task),
                "Duplicate/unregistered T03 parent attempt",
            )
            require(
                receipt is not None and task == f"LH-SRF6-T03-{receipt['job_id']}-{parent}",
                "Task differs from singleton actual submission",
            )
            seen.add(parent)
            tasks[task] = parent
            require(
                attempt["task_id"] == task
                and attempt["spec_ref"] == spec_ref
                and attempt["phase"] == "CONFIRM_T03_V1"
                and attempt["consumer"] == spec["context"]["consumer"]
                and attempt["producer"] == spec["context"]["producer"]
                and type(attempt["generated_responses"]) is int
                and type(attempt["optimizer_updates"]) is int
                and attempt["generated_responses"] == attempt["optimizer_updates"] == 0,
                "Attempt source or zero-call contract differs",
            )
            slot_ref = _ref(root, f"T03/SLOTS/{parent}.json")
            require(read(root / slot_ref["path"]) == attempt, "Original consumed slot differs")
            inputs.extend([attempt_ref, slot_ref])
            if (audit / "FAILURE.json").exists():
                failure_ref = _ref(root, f"T03/RUNS/{task}/FAILURE.json")
                failure = read(root / failure_ref["path"])
                require(
                    all(failure.get(k) == v for k, v in attempt.items()),
                    "Technical failure provenance differs",
                )
                require(
                    failure.get("status") == "TECHNICAL_FAILURE_STOP",
                    "Unknown execution failure status",
                )
                inputs.append(failure_ref)
                parents[parent] = dict(
                    status="TECHNICAL_INVALID", failure_ref=failure_ref, result_ref=None
                )
            elif (audit / "T03/RESULT.json").exists():
                result_ref = _ref(root, f"T03/RUNS/{task}/T03/RESULT.json")
                report = read(root / result_ref["path"])
                facts, leaves = _validate_result(
                    root, context, spec, spec_ref, task, attempt, report
                )
                inputs.extend([result_ref, *leaves])
                parents[parent] = dict(
                    status=facts["status"], result_ref=result_ref, assessment=facts
                )
            attempts.append(dict(task_id=task, parent=parent, status=parents[parent]["status"]))
        except (OSError, KeyError, ValueError, TypeError, PermissionError, IndexError) as exc:
            errors.append(dict(task_id=task, error=f"{type(exc).__name__}: {exc}"))
    leaves, directories = _allowed_members(tasks)
    for name, stamp in before.items():
        if name not in (leaves if stamp[0] == "file" else directories):
            errors.append(dict(error="Unknown T03 evidence member", path=name))
    present_slots = {
        Path(name).stem
        for name, stamp in before.items()
        if name.startswith("SLOTS/") and stamp[0] == "file"
    }
    if present_slots != seen:
        errors.append(dict(error="Orphan/missing consumed slot"))
    for task, parent in tasks.items():
        marker = root / f"T03/FAILURES/{task}.json"
        state = parents[parent]["status"]
        if marker.exists():
            try:
                value = read(marker)
                expected = read(root / f"T03/RUNS/{task}/ATTEMPT.json")
                require(
                    all(value.get(k) == v for k, v in expected.items()),
                    "Failure marker provenance differs",
                )
                expected_state = "TECHNICAL_FAILURE_STOP" if state == "TECHNICAL_INVALID" else state
                require(
                    state in ("FAIL", "INCONCLUSIVE", "TECHNICAL_INVALID")
                    and value.get("status") == expected_state,
                    "Unresolved original failure marker",
                )
                if state in ("FAIL", "INCONCLUSIVE"):
                    require(
                        value.get("result_ref") == parents[parent]["result_ref"],
                        "Failure marker result differs",
                    )
                else:
                    require(
                        value == read(root / f"T03/RUNS/{task}/FAILURE.json"),
                        "Technical failure marker differs",
                    )
                inputs.append(_ref(root, str(marker.relative_to(root))))
            except (OSError, KeyError, ValueError, TypeError, PermissionError) as exc:
                errors.append(dict(task_id=task, error=f"Failure marker: {exc}"))
        elif state in ("FAIL", "INCONCLUSIVE", "TECHNICAL_INVALID"):
            errors.append(dict(task_id=task, error="Missing terminal stop marker"))
    if before != _tree_snapshot(root):
        errors.append(dict(error="T03 evidence changed during verification"))
    after_context = require_support(root, purpose="t03")
    require(build_spec(after_context) == spec, "Support/review/source changed during verification")
    states = [row["status"] for row in parents.values()]
    status = (
        "TECHNICAL_INVALID"
        if errors or "TECHNICAL_INVALID" in states
        else "FAIL"
        if "FAIL" in states
        else "INCOMPLETE"
        if "INCOMPLETE" in states
        else "INCONCLUSIVE"
        if "INCONCLUSIVE" in states
        else "PASS"
    )
    return dict(
        schema=SCHEMA,
        scope="SERVER_ACTUAL",
        status=status,
        parents=parents,
        attempts=attempts,
        errors=errors,
        spec_ref=spec_ref,
        consumer=spec["context"]["consumer"],
        producer=spec["context"]["producer"],
        support_binding=spec["context"]["support_binding"],
        inputs=inputs,
        generated_responses=0,
        optimizer_updates=0,
        training_authorized=False,
        gate_status="NOT_SIGNED",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", choices=PARENTS)
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    result = (
        freeze_spec(args.root)
        if args.action == "freeze"
        else verify_completed(args.root)
        if args.action == "verify"
        else run(args.root, args.parent, args.task_id)
    )
    print(json.dumps(result, ensure_ascii=False))
    if args.action == "run" and result.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
