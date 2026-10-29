"""Read-only projection of the frozen T03 reader; never executes model work.

The two functions below intentionally retain the original control flow. The only
semantic correction takes the trace identity from the scored record while keeping
original token/probability checks attached to the unmodified raw record.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
from pathlib import Path
from types import CodeType

from sr_f6_followup import t03 as original_t03
from sr_f6_followup.t03 import (
    PARENTS,
    SCHEMA,
    _allowed_members,
    _ref,
    _tree_snapshot,
    _validate_boundaries,
    build_spec,
    digest,
    kernel,
    read,
    require,
    selected_parent,
    verified_spec,
)

ORIGINAL_SOURCE_SHA256 = "a3deb4900656ef9a89f39481ff48475afc8b260780f80000f277c084e3144f8d"
TARGETS = ("_validate_result", "verify_completed")

_OLD_VECTOR_LOOP = """for trace, row in zip(report["probability_vectors"], raw, strict=True):
    require(
        trace["row_id"] == row["row_id"]
        and trace["qid"] == row["qid"]
        and trace["tokens"] == row["tokens"]
        and trace["sampler_logprobs"] == row["sampler_logprobs"],
        "Original tokens/probabilities changed",
    )"""
_NEW_VECTOR_LOOP = """for trace, record, original in zip(
    report["probability_vectors"], records, raw, strict=True
):
    require(
        trace["row_id"] == record["row_id"]
        and trace["qid"] == record["qid"] == original["qid"]
        and trace["tokens"] == original["tokens"]
        and trace["sampler_logprobs"] == original["sampler_logprobs"],
        "Original tokens/probabilities changed",
    )"""


def _ast(value):
    return ast.dump(value, include_attributes=False)


def _target(tree, name):
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    require(
        len(found) == 1
        and found[0] in tree.body
        and isinstance(found[0], ast.FunctionDef)
        and not found[0].decorator_list,
        f"Exactly one undecorated top-level {name} required",
    )
    return found[0]


def prove_projection_sources(original_source, projected_source):
    """Prove the exact registered reader correction, without executing source text."""
    require(
        hashlib.sha256(original_source.encode()).hexdigest() == ORIGINAL_SOURCE_SHA256,
        "Frozen T03 reader source changed",
    )
    old_tree, new_tree = ast.parse(original_source), ast.parse(projected_source)
    expected_loop = ast.parse(_OLD_VECTOR_LOOP).body[0]
    replacement_loop = ast.parse(_NEW_VECTOR_LOOP).body[0]
    counts = dict(vector_loop=0, source_import=0, dispatch_import=0)

    class ApprovedRepair(ast.NodeTransformer):
        def visit_For(self, node):
            if _ast(node) == _ast(expected_loop):
                counts["vector_loop"] += 1
                return copy.deepcopy(replacement_loop)
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for module, name, count in (
                ("source", "require_support", "source_import"),
                ("t03_dispatch", "submission_receipt", "dispatch_import"),
            ):
                expected = ast.parse(f"from .{module} import {name}").body[0]
                if _ast(node) == _ast(expected):
                    counts[count] += 1
                    return ast.parse(f"from sr_f6_followup.{module} import {name}").body[0]
            return node

    for name in TARGETS:
        expected = ApprovedRepair().visit(copy.deepcopy(_target(old_tree, name)))
        actual = _target(new_tree, name)
        require(_ast(actual) == _ast(expected), f"Unregistered projection change: {name}")
    require(
        counts == dict(vector_loop=1, source_import=1, dispatch_import=1),
        "Approved repair sites are not unique and complete",
    )
    return dict(
        original_source_sha256=ORIGINAL_SOURCE_SHA256,
        transformations=counts,
        functions=list(TARGETS),
        scientific_predicates="UNCHANGED",
        raw_records="UNCHANGED",
    )


def verify_projection():
    """Check file ASTs, actual loaded functions and original helper identities.

    Compilation is solely a code-object comparison; no source is executed. The
    outer consumer still verifies the complete three-deployment source origins.
    """
    original_path = Path(original_t03.__file__).resolve()
    projected_path = Path(__file__).resolve()
    original_source, projected_source = original_path.read_text(), projected_path.read_text()
    proof = prove_projection_sources(original_source, projected_source)
    compiled = compile(projected_source, str(projected_path), "exec", dont_inherit=True)
    for name in TARGETS:
        fn = globals()[name]
        expected = [
            code
            for code in compiled.co_consts
            if isinstance(code, CodeType) and code.co_name == name
        ]
        require(
            len(expected) == 1
            and fn.__module__ == __name__
            and fn.__name__ == name
            and Path(fn.__code__.co_filename).resolve() == projected_path
            and fn.__code__ == expected[0],
            f"Loaded projection differs from new source: {name}",
        )
    for name in (
        "PARENTS",
        "SCHEMA",
        "_allowed_members",
        "_ref",
        "_tree_snapshot",
        "_validate_boundaries",
        "build_spec",
        "digest",
        "kernel",
        "read",
        "require",
        "selected_parent",
        "verified_spec",
    ):
        require(globals()[name] is getattr(original_t03, name), f"Frozen helper replaced: {name}")
    require(Path is original_t03.Path and re is original_t03.re, "Standard helpers replaced")
    return dict(
        proof,
        original_path=str(original_path),
        projection_path=str(projected_path),
        projection_sha256=hashlib.sha256(projected_source.encode()).hexdigest(),
        generated_responses=0,
        optimizer_updates=0,
    )


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
    for trace, record, original in zip(report["probability_vectors"], records, raw, strict=True):
        require(
            trace["row_id"] == record["row_id"]
            and trace["qid"] == record["qid"] == original["qid"]
            and trace["tokens"] == original["tokens"]
            and trace["sampler_logprobs"] == original["sampler_logprobs"],
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


def verify_completed(root):
    """Reconstruct every attempt; technical uncertainty never becomes scientific FAIL."""
    from sr_f6_followup.source import require_support
    from sr_f6_followup.t03_dispatch import submission_receipt

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
