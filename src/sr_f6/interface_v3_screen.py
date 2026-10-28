"""Execute and assess each fixed stage of the bounded interface-v3 screen.

Whole two-draw question units are immutable. Interrupted partial units and
technical failures stop, without replacement sampling. PASS concerns only this
screen; it does not establish G8 support or authorize training.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
from collections import Counter
from pathlib import Path

from .common import digest, now, read, write_once
from .interface_diagnostics import diagnose_response
from .interface_v3_plan import FACTORS, PARENTS, stage_decision
from .interface_v3_protocol import VERSION
from .interface_v3_runtime import PHASE
from .runtime import PROTOCOL_VERSION, generation_recipe


def output_path(root, relative):
    """Resolve no symlinks, including existing ancestor directories."""
    root = Path(root)
    if root.resolve(strict=True) != root or root.is_symlink():
        raise PermissionError("A canonical registered screen root is required")
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Screen evidence must remain beneath its run root")
    target = root / relative
    for path in (target, *target.parents):
        if path == root:
            break
        if path.is_symlink():
            raise PermissionError("Symlinked screen evidence is forbidden")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def group_spec(context, parent, item):
    if parent not in PARENTS:
        raise ValueError("Unknown registered parent")
    attempts = [
        row
        for row in context["plan"]["attempts"]
        if row["parent"] == parent and row["qid"] == item["qid"]
    ]
    if len(attempts) != 2 or [a["draw"] for a in attempts] != [0, 1]:
        raise PermissionError("Exactly the two registered draws are required")
    return {
        "phase": PHASE,
        "screen_stage": context["stage"],
        "parent": parent,
        **{key: item[key] for key in ("qid", "root_id", "family", "factor", "variant")},
        "registration_sha256": context["registration_sha256"],
        "source_sha256": context["registration"]["source_manifest_sha256"],
        "data_hash": context["registration"]["data_sha256"],
        "parent_content_hash": context["registration"]["parent_content_hashes"][parent],
        "model_input_sha256": digest(item["model_input"]),
        "prompt_sha256": item["prompt_sha256"],
        "private_sha256": digest(context["private"][item["qid"]]),
        "seeds": [a["seed"] for a in attempts],
    }


def annotate(raw, spec, task, *, preflight_route=None):
    """Rescore intact raw text, never coerce malformed answers or missing labels."""
    if len(raw) != 2 or [r.get("seed") for r in raw] != spec["seeds"]:
        raise ValueError("Screen unit must retain exactly its two original draws")
    tensor_hashes = {row.get("policy_tensor_hash") for row in raw}
    if len(tensor_hashes) != 1 or not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in tensor_hashes
    ):
        raise PermissionError("Screen unit requires one actual loaded policy identity")
    result = []
    for draw, row in enumerate(raw):
        if type(row.get("screen_stage")) is not int or any(
            row.get(key) != spec[key]
            for key in (
                "phase",
                "screen_stage",
                "registration_sha256",
                "source_sha256",
                "data_hash",
                "parent_content_hash",
            )
        ):
            raise PermissionError("Raw screen provenance differs from registration")
        if (
            row.get("generation_status") != "COMPLETE"
            or row.get("technical_validation_errors") != []
            or row.get("execution_mode") != "READONLY_INTERFACE_SCREEN"
            or row.get("optimizer_updates") != 0
            or row.get("step") != 0
            or row.get("policy_step") != 0
            or row.get("protocol") != "structured_v4"
            or row.get("prompt_version") != VERSION
            or row.get("sr_f6_protocol_version") != PROTOCOL_VERSION
            or row.get("seed_semantics") != "independent_per_draw_torch_generator"
            or row.get("group_size") != 2
            or row.get("group_row_index") != draw
            or row.get("group_seed") != spec["seeds"][0]
            or row.get("adapter_identity", {}).get("parent") != spec["parent"]
            or row.get("adapter_identity", {}).get("step") != 0
        ):
            raise PermissionError("Invalid raw screen execution or technical failure")
        tokens, logps = row.get("tokens"), row.get("sampler_logprobs")
        if (
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= 1536
            or tokens != row.get("raw_tokens")
            or not isinstance(logps, list)
            or len(tokens) != len(logps)
            or any(type(token) is not int or token < 0 for token in tokens)
            or any(type(p) not in (int, float) or not math.isfinite(p) for p in logps)
            or row.get("completion_token_count") != len(tokens)
            or row.get("max_new_tokens") != 1536
            or row.get("sampler_logprob_source") != "actual_generate_raw_logits_selected_token"
            or row.get("finish_reason") not in ("balanced", "eos", "length")
            or (row.get("finish_reason") == "length" and len(tokens) != 1536)
        ):
            raise ValueError("Incomplete original tokens, probabilities, or termination")
        parameters = row.get("sampling_parameters", {})
        if any(parameters.get(key) != value for key, value in generation_recipe().items()):
            raise PermissionError("Raw screen sampling recipe changed")
        if (
            digest(row.get("model_input")) != spec["model_input_sha256"]
            or row.get("prompt_sha256") != spec["prompt_sha256"]
        ):
            raise PermissionError("Raw screen model input changed")
        if preflight_route is not None and (
            row.get("prompt_tensor_hash") != preflight_route["input_tensor_hash"]
            or any(
                row.get("input_routing", {}).get(key) != preflight_route[key]
                for key in (
                    "input_tensor_hash",
                    "processor_hash",
                    "chat_template_hash",
                    "chat_text_sha256",
                )
            )
        ):
            raise PermissionError("Raw screen processor identity differs from server preflight")
        diagnostic = diagnose_response(
            row["full_text"], task["world"], task["query"], raw_finish_reason=row["finish_reason"]
        )
        result.append(
            dict(
                row,
                row_id=f"{spec['parent']}/{spec['qid']}/draw{draw}",
                draw=draw,
                parent=spec["parent"],
                qid=spec["qid"],
                root_id=spec["root_id"],
                family=spec["family"],
                factor=spec["factor"],
                variant=spec["variant"],
                engineered_collision_anchor=task["engineered_collision_anchor"],
                wire_score=diagnostic["wire_score"],
                diagnostic=diagnostic,
            )
        )
    return result


def validate_group(group, spec, task, *, preflight_route=None):
    if group.get("status") != "COMPLETE" or group.get("spec") != spec:
        raise PermissionError("Saved screen unit is incomplete or has changed provenance")
    records = group.get("records", [])
    rebuilt = annotate(records, spec, task, preflight_route=preflight_route)
    if records != rebuilt:
        raise PermissionError("Saved screen labels differ from strict rescoring")
    return records


def _append_raw(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())


def run(root, parent, shard, task_id, *, stage=1):
    from .interface_v3_registration import verify_registered
    from .interface_v3_runtime import load_screen

    if parent not in PARENTS or type(shard) is not int or shard not in (0, 1):
        raise ValueError("Registered parent and one of two question shards required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task_id):
        raise ValueError("One safe execution task identifier required")
    base = Path(root).resolve(strict=True)
    context = verify_registered(base, stage=stage, parent=parent, require_preflight=True)
    root = context["root"]
    selected = context["public"][shard::2]
    routes = {item["qid"]: item["routing"] for item in context["preflight"]["inputs"]}
    # Fixed shard lock prevents duplicate workers, even with distinct task identifiers.
    lock_path = output_path(root, f"SCREEN/locks/{parent}-{shard}.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt_path = output_path(root, f"SCREEN/executions/{task_id}.json")
        if attempt_path.exists():
            raise FileExistsError("Use a new task identifier; preserve prior execution")
        write_once(
            attempt_path,
            dict(
                status="STARTED",
                at=now(),
                parent=parent,
                shard=shard,
                shards=2,
                task_id=task_id,
                registration_sha256=context["registration_sha256"],
            ),
        )
        output_path(root, f"runtime_identity/{task_id}.json")
        output_path(root, f"accounting/{task_id}.jsonl")
        runtime = None
        generated = recovered = reused = 0
        try:
            for item in selected:
                spec = group_spec(context, parent, item)
                task = context["private"][item["qid"]]
                stem = f"{parent}/{item['qid']}"
                path = output_path(root, f"SCREEN/groups/{stem}.json")
                raw_path = output_path(root, f"SCREEN/raw/{stem}.jsonl")
                unit_attempt = output_path(root, f"SCREEN/attempts/{stem}.json")
                if path.exists():
                    validate_group(read(path), spec, task, preflight_route=routes[item["qid"]])
                    reused += 2
                    continue
                if unit_attempt.exists():
                    if read(unit_attempt).get("spec") != spec:
                        raise PermissionError("Interrupted screen unit identity changed")
                    raw = _rows(raw_path) if raw_path.exists() else []
                    if len(raw) != 2:
                        raise RuntimeError("Partial screen unit preserved; no automatic resampling")
                    recovered += 2
                else:
                    if raw_path.exists():
                        raise RuntimeError("Orphan raw screen evidence requires diagnosis")
                    if runtime is None:
                        runtime = load_screen(context, parent, task_id)
                    write_once(
                        unit_attempt,
                        dict(
                            status="STARTED",
                            at=now(),
                            spec=spec,
                            task_id=task_id,
                        ),
                    )
                    runtime.generate_registered(
                        item["qid"], on_completion=lambda row, p=raw_path: _append_raw(p, row)
                    )
                    generated += 2
                    raw = _rows(raw_path)
                records = annotate(raw, spec, task, preflight_route=routes[item["qid"]])
                group = dict(status="COMPLETE", spec=spec, records=records)
                validate_group(group, spec, task, preflight_route=routes[item["qid"]])
                write_once(path, group)
                print(
                    json.dumps(dict(task_id=task_id, qid=item["qid"], status="COMPLETE")),
                    flush=True,
                )
            if runtime is not None:
                runtime.policy_tensor_identity()
            after = verify_registered(base, stage=stage, parent=parent, require_preflight=True)
            if after["registration_sha256"] != context["registration_sha256"]:
                raise PermissionError("Screen registration changed during execution")
            result = dict(
                status="COMPLETE",
                phase=PHASE,
                screen_stage=stage,
                task_id=task_id,
                parent=parent,
                shard=shard,
                shards=2,
                groups=len(selected),
                responses=2 * len(selected),
                generated_this_process=generated,
                recovered_raw_rows=recovered,
                reused_rows=reused,
                optimizer_updates=0,
                registration_sha256=context["registration_sha256"],
                training_allowed=False,
                gate_status="NOT_SIGNED",
            )
            write_once(output_path(root, f"SCREEN/workers/{task_id}.json"), result)
            return result
        except Exception as exc:
            write_once(
                output_path(root, f"SCREEN/failures/{task_id}.json"),
                dict(
                    status="TECHNICAL_FAILURE_STOP",
                    at=now(),
                    task_id=task_id,
                    parent=parent,
                    shard=shard,
                    error=f"{type(exc).__name__}: {exc}",
                    registration_sha256=context["registration_sha256"],
                    completed_new_rows_before_failure=generated,
                    optimizer_updates=0,
                    on_failure="PRESERVE_ALL_NO_AUTOMATIC_RETRY",
                ),
            )
            raise


def _counts(records):
    return {
        "V": sum(r["wire_score"]["V"] for r in records),
        "S_known": sum(r["wire_score"]["S"] is not None for r in records),
        "B_known": sum(r["wire_score"]["B"] is not None for r in records),
    }


def summarize_records(records, *, stage=1, parents=PARENTS, prior=None):
    from .interface_v3_audit import j_gap_breakdown

    expected_n = 192 * len(parents)
    if len(records) != expected_n or len({r["row_id"] for r in records}) != expected_n:
        raise ValueError("Complete distinct authorized stage responses required")
    if any(r.get("screen_stage") != stage for r in records):
        raise ValueError("Mixed stage responses")
    cells = Counter((r["parent"], r["factor"]) for r in records)
    if cells != Counter({(p, f): 96 for p in parents for f in FACTORS}):
        raise ValueError("Complete authorized parent/factor cells required")
    parent_counts = {p: _counts([r for r in records if r["parent"] == p]) for p in parents}
    decision = stage_decision(parent_counts, stage=stage, prior=prior)
    output, families = [], []
    for parent in parents:
        for factor in FACTORS:
            subset = [r for r in records if (r["parent"], r["factor"]) == (parent, factor)]
            counts = dict(
                _counts(subset),
                A=sum(r["wire_score"]["A"] for r in subset),
                J_wire=sum(r["wire_score"]["J"] for r in subset),
                J_core_diag=sum(r["diagnostic"]["J_core_diag"] == 1 for r in subset),
            )
            output.append(
                dict(
                    parent=parent,
                    factor=factor,
                    n=96,
                    counts=counts,
                    interpretation="FACTOR_DESCRIPTIVE_NOT_POOLED_POPULATION_PROOF",
                )
            )
            for family in sorted({r["family"] for r in subset}):
                sub = [r for r in subset if r["family"] == family]
                families.append(
                    dict(parent=parent, factor=factor, family=family, n=len(sub), **_counts(sub))
                )
    return dict(
        schema="srf6-interface-v3-stage-assessment-v1",
        status=decision["status"],
        phase=PHASE,
        screen_stage=stage,
        parents=list(parents),
        responses=expected_n,
        decision=decision,
        cells=output,
        family_counts=families,
        raw_finish_counts=dict(Counter(r["finish_reason"] for r in records)),
        j_gap=j_gap_breakdown(records),
        completion_policy="original-census-balanced-only",
        scorer_changed=False,
        optimizer_updates=0,
        engineering_valid="STAGE_COMPLETE",
        measurement_valid="ENGINEERING_SCREEN_ONLY",
        comparison_supported="NOT_RUN",
        effect_estimated="NOT_RUN",
        complete_G8_credit="NOT_APPLICABLE",
        training_allowed=False,
        confirmation_status="REQUIRES_INDEPENDENT_REGISTRATION"
        if decision["status"] == "SCREEN_PASS"
        else "NOT_AUTHORIZED",
        interpretation="Two-stage resource screen; no IID, confidence or learning claim",
    )


def collect_records(context, parents):
    root = context["root"]
    stage = context["stage"]
    if list((root / "SCREEN/failures").glob("*.json")):
        raise PermissionError("Preserved technical failure requires diagnosis; no PASS")
    routes = {r["qid"]: r["routing"] for r in context["preflight"]["inputs"]}
    records, expected, expected_raw, expected_attempts = [], set(), set(), set()
    for parent in parents:
        for item in context["public"]:
            path = output_path(root, f"SCREEN/groups/{parent}/{item['qid']}.json")
            expected.add(path)
            spec = group_spec(context, parent, item)
            task = context["private"][item["qid"]]
            group_records = validate_group(
                read(path), spec, task, preflight_route=routes[item["qid"]]
            )
            raw_path = output_path(root, f"SCREEN/raw/{parent}/{item['qid']}.jsonl")
            attempt_path = output_path(root, f"SCREEN/attempts/{parent}/{item['qid']}.json")
            expected_raw.add(raw_path)
            expected_attempts.add(attempt_path)
            if (
                read(attempt_path).get("spec") != spec
                or annotate(_rows(raw_path), spec, task, preflight_route=routes[item["qid"]])
                != group_records
            ):
                raise PermissionError("Saved unit differs from actual raw sampling evidence")
            records.extend(group_records)
    for directory, pattern, wanted in (
        ("groups", "*/*.json", expected),
        ("raw", "*/*.jsonl", expected_raw),
        ("attempts", "*/*.json", expected_attempts),
    ):
        if set((root / "SCREEN" / directory).glob(pattern)) != wanted:
            raise PermissionError("Unexpected or missing stage sampling files")
    for parent in parents:
        if len({r["policy_tensor_hash"] for r in records if r["parent"] == parent}) != 1:
            raise PermissionError("Mixed parent tensors")
    workers = [read(p) for p in (root / "SCREEN/workers").glob("*.json")]
    expected_workers = {(p, s) for p in parents for s in (0, 1)}
    if (
        len(workers) != len(expected_workers)
        or {(w.get("parent"), w.get("shard")) for w in workers} != expected_workers
    ):
        raise PermissionError("Exact authorized stage workers required")
    for w in workers:
        if any(
            w.get(k) != v
            for k, v in dict(
                status="COMPLETE",
                phase=PHASE,
                screen_stage=stage,
                registration_sha256=context["registration_sha256"],
                groups=48,
                responses=96,
                shards=2,
                optimizer_updates=0,
            ).items()
        ):
            raise PermissionError("Incomplete or cross-stage worker receipt")
    return records, verify_accounting(root, records, context)


def _stage1_assessment(base):
    from .common import file_hash
    from .interface_v3_registration import verify_registered

    context = verify_registered(base, stage=1, require_preflight=True)
    root = context["root"]
    report_path = output_path(root, "SCREEN/SCREEN_ASSESSMENT.json")
    report = read(report_path)
    seal = read(output_path(root, "SCREEN/ASSESSMENT_SEAL.json"))
    expected_seal = dict(
        schema="srf6-interface-v3-assessment-seal-v1",
        screen_stage=1,
        registration_sha256=context["registration_sha256"],
        assessment_sha256=file_hash(report_path),
        training_allowed=False,
    )
    if seal != expected_seal:
        raise PermissionError("Prior stage assessment seal differs")
    records, accounting = collect_records(context, PARENTS)
    rebuilt = summarize_records(records, stage=1)
    if (
        any(report.get(k) != v for k, v in rebuilt.items())
        or report.get("records_sha256") != digest(records)
        or report.get("accounting") != accounting
    ):
        raise PermissionError("Prior stage gate differs from complete original records")
    token = report.get("saved_token_validation", {})
    if (
        token.get("status") != "PASS"
        or token.get("checked_responses") != 576
        or token.get("checked_tokens") != accounting["actual_generated_tokens"]
        or token.get("new_model_calls") != 0
    ):
        raise PermissionError("Prior complete original token validation absent")
    return report


def stage2_authorized(base, parent=None):
    report = _stage1_assessment(base)
    if report["decision"]["status"] != "STAGE2_REQUIRED":
        raise PermissionError("Only an all-parent completed non-failing gray stage permits stage2")
    allowed = report["decision"]["stage2_parents"]
    if not allowed or (parent is not None and parent not in allowed):
        raise PermissionError("Parent is not prospectively authorized for stage2")
    return list(allowed) if parent is None else True


def assess(root, *, stage=1):
    from sr_f5.runtime import Runtime as ProcessorRuntime

    from .common import file_hash
    from .interface_v3_groups import audit_screen_groups
    from .interface_v3_registration import verify_registered

    base = Path(root).resolve(strict=True)
    prior = _stage1_assessment(base)["decision"] if stage == 2 else None
    parents = tuple(stage2_authorized(base)) if stage == 2 else PARENTS
    context = verify_registered(base, stage=stage, parent=parents[0], require_preflight=True)
    output = context["root"]
    records, accounting = collect_records(context, parents)
    runtime = ProcessorRuntime.processor_only(
        context["config"]["model"]["model_path"], protocol_revision="v2"
    )
    token_check = verify_saved_tokens(records, runtime.processor.tokenizer, runtime.eos_ids)
    groups = audit_screen_groups(records, context["private"], stage=stage, parents=parents)
    write_once(output_path(output, "SCREEN/GROUP_DESCRIPTIVE.json"), groups)
    report = summarize_records(records, stage=stage, parents=parents, prior=prior)
    report.update(
        registration_sha256=context["registration_sha256"],
        records_sha256=digest(records),
        accounting=accounting,
        saved_token_validation=token_check,
        group_diagnostics=dict(
            path="SCREEN/GROUP_DESCRIPTIVE.json",
            object_sha256=digest(groups),
            interpretation="DESCRIPTIVE_ONLY",
        ),
    )
    path = output_path(output, "SCREEN/SCREEN_ASSESSMENT.json")
    write_once(path, report)
    write_once(
        output_path(output, "SCREEN/ASSESSMENT_SEAL.json"),
        dict(
            schema="srf6-interface-v3-assessment-seal-v1",
            screen_stage=stage,
            registration_sha256=context["registration_sha256"],
            assessment_sha256=file_hash(path),
            training_allowed=False,
        ),
    )
    return report


def verify_accounting(root, records, context):
    """Reconcile actual native reservations with all saved rows and loaded identities."""
    task_ids = {r["task_id"] for r in records}
    expected = {Path(root) / "accounting" / f"{task_id}.jsonl" for task_id in task_ids}
    if set((Path(root) / "accounting").glob("*.jsonl")) != expected:
        raise PermissionError("Actual sampling ledger has missing or additional executions")
    totals = Counter()
    for task_id in sorted(task_ids):
        subset = [r for r in records if r["task_id"] == task_id]
        identity = read(output_path(root, f"runtime_identity/{task_id}.json"))
        if (
            identity.get("registration_sha256") != context["registration_sha256"]
            or identity.get("phase") != PHASE
            or type(identity.get("screen_stage")) is not int
            or identity.get("screen_stage") != context["stage"]
            or identity.get("execution_mode") != "READONLY_INTERFACE_SCREEN"
            or identity.get("fresh_optimizer") is not False
            or identity.get("trainable_parameters") != []
            or identity.get("optimizer_updates") != 0
            or {r["policy_tensor_hash"] for r in subset} != {identity.get("policy_hash")}
            or {r["parent"] for r in subset} != {identity.get("parent_id")}
        ):
            raise PermissionError("Actual loaded parent identity differs from sampled rows")
        ledger = _rows(output_path(root, f"accounting/{task_id}.jsonl"))
        counts = Counter()
        expected_events = Counter()
        for row in subset:
            qid, seed, index = row["qid"], row["group_seed"], row["group_row_index"]
            expected_events[(qid, "generated_tokens", seed, index, len(row["tokens"]))] += 1
            if index == 0:
                expected_events[(qid, "completion_attempts", seed, None, 2)] += 1
        actual_events = Counter()
        for entry in ledger:
            metadata = entry.get("metadata", {})
            if (
                entry.get("task_id") != task_id
                or entry.get("parent") != identity["parent_id"]
                or entry.get("kind") not in ("completion_attempts", "generated_tokens")
                or type(entry.get("count")) is not int
                or entry["count"] <= 0
                or metadata.get("phase") != PHASE
                or type(metadata.get("screen_stage")) is not int
                or metadata.get("screen_stage") != context["stage"]
                or metadata.get("registration_sha256") != context["registration_sha256"]
                or (
                    entry["kind"] == "completion_attempts"
                    and metadata.get("protocol") != "structured"
                )
            ):
                raise PermissionError("Unexpected actual model-call ledger entry")
            counts[entry["kind"]] += entry["count"]
            actual_events[
                (
                    metadata.get("qid"),
                    entry["kind"],
                    metadata.get("group_seed"),
                    metadata.get("row_index"),
                    entry["count"],
                )
            ] += 1
        if actual_events != expected_events:
            raise PermissionError(
                "Actual sampling reservations differ from registered question/draw rows"
            )
        if counts["completion_attempts"] != len(subset) or counts["generated_tokens"] != sum(
            len(r["tokens"]) for r in subset
        ):
            raise PermissionError("Actual sampling reservations do not match saved rows")
        totals.update(counts)
    if totals["completion_attempts"] != len(records):
        raise PermissionError("Actual sampling budget must match the authorized complete stage")
    return dict(
        actual_completion_attempts=totals["completion_attempts"],
        actual_generated_tokens=totals["generated_tokens"],
        optimizer_updates=0,
    )


def verify_saved_tokens(records, tokenizer, eos_ids):
    """Decode original saved tokens and replay stopping on CPU, without any model call."""
    from sr_f1.json_protocol import decode_generated
    from sr_f3_ce.runtime import _prefill, first_boundary_token

    checked_tokens = 0
    for row in records:
        tokens = row["tokens"]
        text = decode_generated(tokenizer, tokens)
        if row["full_text"] != _prefill("structured", "v2") + text or row.get(
            "generated_text_with_special_tokens"
        ) != tokenizer.decode(
            tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ):
            raise PermissionError("Saved response text differs from original sampled token IDs")
        boundary, _ = first_boundary_token(tokenizer, tokens, "structured", revision="v2")
        eos = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), None)
        n = min(value for value in (boundary, eos, len(tokens)) if value is not None)
        finish = "balanced" if boundary == n else "eos" if eos == n else "length"
        if n != len(tokens) or finish != row["finish_reason"]:
            raise PermissionError("Saved response stopping differs from original sampled tokens")
        backend = row.get("backend_sequence_tokens")
        if (
            not isinstance(backend, list)
            or backend[: len(tokens)] != tokens
            or any(token != tokenizer.pad_token_id for token in backend[len(tokens) :])
            or row.get("backend_padding_count") != len(backend) - len(tokens)
        ):
            raise PermissionError("Saved backend tokens or native row padding changed")
        checked_tokens += len(tokens)
    return dict(
        status="PASS",
        checked_responses=len(records),
        checked_tokens=checked_tokens,
        new_model_calls=0,
        validation="CPU original-token decode and native stopping replay",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--parent", choices=PARENTS, required=True)
    run_parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    run_parser.add_argument("--task-id", required=True)
    assess_parser = sub.add_parser("assess")
    assess_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    assess_parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    result = (
        run(args.root, args.parent, args.shard, args.task_id, stage=args.stage)
        if args.command == "run"
        else assess(args.root, stage=args.stage)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
