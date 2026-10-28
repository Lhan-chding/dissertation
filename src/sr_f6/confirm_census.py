"""Independent, immutable G8 confirmation; no training or automatic resampling.

The old census mathematics is reused, with a new split, producer and source
identity. Raw native trajectories are stored before scoring or token replay.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
from pathlib import Path

from srf6_ref.core import seed_for

from .census import annotate_group, counterfactual
from .common import digest, file_hash, now, write_once
from .interface_diagnostics import diagnose_response
from .interface_v3_protocol import VERSION
from .interface_v3_screen import _append_raw, output_path
from .runtime import PROTOCOL_VERSION, generation_recipe

PHASE = "SCHEMA_CONFIRM_V3"
SPLIT = "DEV_SCHEMA_CONFIRM"
MASTER = 20261028
PARENTS = ("b0", "b1", "b2")


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def group_spec(context, parent, item):
    """Recompute the exact eight seeds; never derive them from observed answers."""
    _require(parent in PARENTS and item["split"] == SPLIT, "Unregistered confirmation input")
    spec = context["spec"]
    attempts = [r for r in spec["attempts"] if r["parent"] == parent and r["qid"] == item["qid"]]
    seeds = [
        seed_for(MASTER, "srf6-confirm-v3", "confirm", parent, item["root_id"], item["variant"], d)
        for d in range(8)
    ]
    _require(
        len(attempts) == 8
        and [a["draw"] for a in attempts] == list(range(8))
        and [a["seed"] for a in attempts] == seeds,
        "Frozen confirmation draws differ from the prospective rule",
    )
    registration = context["registration"]
    parent_hash = registration["parent_content_hashes"][parent]
    return dict(
        phase=PHASE,
        split=SPLIT,
        parent=parent,
        **{key: item[key] for key in ("qid", "root_id", "family", "factor", "variant")},
        group_id=f"{parent}/{item['qid']}",
        policy_hash=parent_hash,
        parent_content_hash=parent_hash,
        registration_sha256=context["registration_sha256"],
        frozen_spec_sha256=context["spec_object_sha256"],
        source_sha256=registration["source_manifest_sha256"],
        data_hash=registration["data_sha256"],
        model_input_sha256=digest(item["model_input"]),
        public_sha256=digest(item),
        private_sha256=digest(context["private"][item["qid"]]),
        prompt_sha256=item["prompt_sha256"],
        seeds=seeds,
        shuffle_seed=seed_for(
            MASTER, "census_shuffle", parent, item["factor"], f"{parent}/{item['qid']}"
        ),
    )


def annotate(raw, spec, task, *, preflight_route=None):
    """Keep native termination alongside strict score, UNKNOWN and content diagnosis."""
    _require(spec["phase"] == PHASE and spec["split"] == SPLIT, "Wrong confirmation phase")
    _require(digest(task) == spec["private_sha256"], "Private question identity differs")
    if len(raw) != 8 or [r.get("seed") for r in raw] != spec["seeds"]:
        raise ValueError("Exactly the eight original registered draws are required")
    for draw, row in enumerate(raw):
        _require(
            all(
                row.get(key) == spec[key]
                for key in (
                    "phase",
                    "registration_sha256",
                    "source_sha256",
                    "data_hash",
                    "parent_content_hash",
                )
            ),
            "Raw confirmation provenance differs",
        )
        _require(
            row.get("generation_status") == "COMPLETE"
            and row.get("technical_validation_errors") == []
            and row.get("execution_mode") == "READONLY_SCHEMA_CONFIRM"
            and type(row.get("optimizer_updates")) is int
            and row["optimizer_updates"] == 0
            and row.get("step") == row.get("policy_step") == 0
            and row.get("protocol") == "structured_v4"
            and row.get("prompt_version") == VERSION
            and row.get("sr_f6_protocol_version") == PROTOCOL_VERSION
            and row.get("seed_semantics") == "independent_per_draw_torch_generator"
            and row.get("group_size") == 8
            and row.get("group_row_index") == draw
            and row.get("group_seed") == spec["seeds"][0],
            "Invalid confirmation execution or preserved technical failure",
        )
        tokens, logps = row.get("tokens"), row.get("sampler_logprobs")
        if (
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= 1536
            or tokens != row.get("raw_tokens")
            or any(type(t) is not int or t < 0 for t in tokens)
            or not isinstance(logps, list)
            or len(logps) != len(tokens)
            or any(type(p) not in (int, float) or not math.isfinite(p) or p > 0 for p in logps)
            or row.get("completion_token_count") != len(tokens)
            or row.get("max_new_tokens") != 1536
            or row.get("sampler_logprob_source") != "actual_generate_raw_logits_selected_token"
            or row.get("finish_reason") not in ("balanced", "eos", "length")
            or (row.get("finish_reason") == "length" and len(tokens) != 1536)
        ):
            raise ValueError("Incomplete original confirmation tokens/probabilities/termination")
        _require(
            all(
                row.get("sampling_parameters", {}).get(k) == v
                for k, v in generation_recipe().items()
            )
            and digest(row.get("model_input")) == spec["model_input_sha256"]
            and row.get("prompt_sha256") == spec["prompt_sha256"],
            "Confirmation prompt or sampling recipe differs",
        )
        if preflight_route is not None:
            _require(
                row.get("prompt_tensor_hash") == preflight_route["input_tensor_hash"]
                and all(
                    row.get("input_routing", {}).get(k) == preflight_route[k]
                    for k in (
                        "input_tensor_hash",
                        "processor_hash",
                        "chat_template_hash",
                        "chat_text_sha256",
                    )
                ),
                "Confirmation processor route differs from actual preflight",
            )
    annotated = annotate_group(raw, task, spec)
    for original, row in zip(raw, annotated, strict=True):
        diagnostic = diagnose_response(
            original["full_text"],
            task["world"],
            task["query"],
            raw_finish_reason=original["finish_reason"],
        )
        _require(row["score"] == diagnostic["wire_score"], "Strict confirmation scorers disagree")
        row.update(
            finish_reason=original["finish_reason"],
            raw_finish_reason=original["finish_reason"],
            split=SPLIT,
            variant=task["variant"],
            engineered_collision_anchor=task["engineered_collision_anchor"],
            wire_score=diagnostic["wire_score"],
            diagnostic=diagnostic,
        )
    return annotated


def validate_group(group, spec, task, *, preflight_route=None):
    _require(
        group.get("status") == "COMPLETE" and group.get("spec") == spec,
        "Confirmation group identity differs",
    )
    records = group.get("records", [])
    _require(
        records == annotate(records, spec, task, preflight_route=preflight_route)
        and group.get("credit") == counterfactual(records, spec),
        "Persisted confirmation labels or G8 credit differ",
    )
    return records


def _alignment_identity(spec, raw):
    return dict(
        frozen_spec_sha256=spec["frozen_spec_sha256"],
        group_spec_sha256=digest(spec),
        raw_trajectories_sha256=digest(
            [{k: r[k] for k in ("tokens", "sampler_logprobs", "prompt_tensor_hash")} for r in raw]
        ),
    )


def _diagnostics(traces, records):
    import torch

    from sr_f12.training import enforce_probability_gate, probability_diagnostics

    _require(len(traces) == len(records) == 8, "Eight actual alignment traces required")
    differences, truncated = [], []
    for trace, row in zip(traces, records, strict=True):
        sampler = torch.tensor(row["sampler_logprobs"], dtype=torch.float32)
        _require(
            trace.get("row_id") == row["row_id"]
            and trace.get("token_ids") == row["tokens"]
            and trace.get("sampled_logprobs") == sampler.tolist(),
            "Replay trace differs from original draw",
        )
        current = torch.tensor(trace.get("current_logprobs"), dtype=torch.float32)
        _require(
            current.shape == sampler.shape
            and current.ndim == 1
            and bool(torch.isfinite(current).all()),
            "Invalid saved-token alignment vector",
        )
        delta = current - sampler
        differences.append(delta.abs())
        truncated.append(delta > math.log(2.0))
    diagnostics = probability_diagnostics(differences, truncated)
    enforce_probability_gate(diagnostics)
    _require(
        diagnostics["sampler_train_max_absolute_difference"] <= 5.0,
        "Maximum alignment difference exceeds five nat",
    )
    return diagnostics


def validate_alignment(report, spec, records):
    _require(
        report.get("status") == "PASS"
        and report.get("identity") == _alignment_identity(spec, records)
        and report.get("phase") == PHASE
        and report.get("parent") == spec["parent"]
        and report.get("group_id") == spec["group_id"]
        and report.get("execution_origin") == "SERVER_ACTUAL"
        and report.get("parameter_tensor_sha256") == records[0]["policy_tensor_hash"]
        and report.get("checked_sequences") == report.get("original_registered_responses") == 8
        and type(report.get("new_generated_responses")) is int
        and report["new_generated_responses"] == 0
        and type(report.get("optimizer_updates")) is int
        and report["optimizer_updates"] == 0,
        "Actual first-G8 alignment identity differs or failed",
    )
    _require(
        report.get("diagnostics") == _diagnostics(report.get("traces", []), records),
        "Alignment summary differs from original traces",
    )
    return report


def check_alignment(runtime, raw, records, spec, output):
    """One first-G8 replay, saving failures even when a numerical gate rejects."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve prior alignment; no automatic repeat")
    report = dict(
        status="FAIL",
        phase=PHASE,
        identity=_alignment_identity(spec, raw),
        parent=spec["parent"],
        group_id=spec["group_id"],
        execution_origin="SERVER_ACTUAL",
        reviewer_type="PROGRAM",
        new_generated_responses=0,
        optimizer_updates=0,
        original_registered_responses=8,
        traces=[],
    )

    def capture(draw, current):
        _require(
            type(draw) is int and draw == len(report["traces"]) < 8,
            "Alignment callback order differs",
        )
        row = records[draw]
        sampler = runtime.torch.tensor(row["sampler_logprobs"], dtype=runtime.torch.float32)
        report["traces"].append(
            dict(
                row_id=row["row_id"],
                token_ids=row["tokens"],
                sampled_logprobs=sampler.tolist(),
                current_logprobs=current.detach().float().cpu().tolist(),
            )
        )

    try:
        before = runtime.policy_tensor_identity()
        forwarded = runtime.replay_alignment(raw, on_replay=capture)
        _require(len(forwarded) == 8, "First-G8 alignment returned incomplete vectors")
        _require(
            len(report["traces"]) == 8
            and all(
                row["current_logprobs"] == value.detach().float().cpu().tolist()
                for row, value in zip(report["traces"], forwarded, strict=True)
            ),
            "Returned replay vectors differ from saved callbacks",
        )
        report["diagnostics"] = _diagnostics(report["traces"], records)
        _require(
            before == runtime.policy_tensor_identity(), "Confirmation replay changed parameters"
        )
        report.update(status="PASS", parameter_tensor_sha256=before, checked_sequences=8)
        validate_alignment(report, spec, records)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_once(output, report)
    return report


def run(root, parent, shard, task_id):
    """Run one newly submitted immutable shard. Existing or partial shards stop."""
    from .confirm_registration import verify_registered
    from .confirm_runtime import load_confirm

    if parent not in PARENTS or type(shard) is not int or shard not in (0, 1):
        raise ValueError("One registered parent and one of two shards required")
    if not re.fullmatch(rf"LH-SRF6-Confirm-[0-9]+-{parent}-{shard}", task_id):
        raise ValueError("Task identifier must match the registered job/parent/shard")
    context = verify_registered(root, parent=parent, require_preflight=True)
    root = context["root"]
    _require(
        [r["qid"] for r in context["public"]] == sorted(r["qid"] for r in context["public"]),
        "Confirmation inputs must follow frozen sorted qid order",
    )
    selected = context["public"][shard::2]
    _require(len(selected) == 192, "Complete fixed confirmation shard required")
    assigned = [
        w for w in context["spec"]["workers"] if w["parent"] == parent and w["shard"] == shard
    ]
    _require(
        len(assigned) == 1 and assigned[0]["qids"] == [r["qid"] for r in selected],
        "Confirmation order differs from frozen worker assignment",
    )
    routes = {r["qid"]: r["routing"] for r in context["preflight"]["inputs"]}
    lock_path = output_path(root, f"CONFIRM/locks/{parent}-{shard}.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _require(
            not list((root / "CONFIRM/failures").glob("*.json")),
            "Preserved confirmation failure requires diagnosis",
        )
        # Fixed execution slot is consumed before model construction, also on failure.
        slot = output_path(root, f"CONFIRM/execution_slots/{parent}-{shard}.json")
        _require(not slot.exists(), "Prior confirmation shard exists; no automatic retry")
        execution = dict(
            status="STARTED",
            at=now(),
            phase=PHASE,
            parent=parent,
            shard=shard,
            shards=2,
            task_id=task_id,
            registration_sha256=context["registration_sha256"],
        )
        write_once(slot, execution)
        generated = 0
        try:
            write_once(output_path(root, f"CONFIRM/executions/{task_id}.json"), execution)
            alignment_path = output_path(root, f"CONFIRM/alignment/{task_id}.json")
            runtime = load_confirm(context, parent, task_id)
            for index, item in enumerate(selected):
                spec = group_spec(context, parent, item)
                task = context["private"][item["qid"]]
                stem = f"{parent}/{item['qid']}"
                target = output_path(root, f"CONFIRM/groups/{stem}.json")
                raw_path = output_path(root, f"CONFIRM/raw/{stem}.jsonl")
                attempt = output_path(root, f"CONFIRM/attempts/{stem}.json")
                _require(
                    not any(p.exists() for p in (target, raw_path, attempt)),
                    "Existing confirmation draw must be preserved; no resampling",
                )
                write_once(attempt, dict(status="STARTED", at=now(), spec=spec, task_id=task_id))
                runtime.generate_registered(
                    item["qid"], on_completion=lambda row, p=raw_path: _append_raw(p, row)
                )
                generated += 8
                raw = [
                    json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()
                ]
                records = annotate(raw, spec, task, preflight_route=routes[item["qid"]])
                if index == 0:
                    check_alignment(runtime, raw, records, spec, alignment_path)
                group = dict(
                    status="COMPLETE",
                    spec=spec,
                    records=records,
                    credit=counterfactual(records, spec),
                )
                validate_group(group, spec, task, preflight_route=routes[item["qid"]])
                write_once(target, group)
                print(
                    json.dumps(dict(task_id=task_id, qid=item["qid"], status="COMPLETE")),
                    flush=True,
                )
            runtime.policy_tensor_identity()
            after = verify_registered(root, parent=parent, require_preflight=True)
            _require(
                after["registration_sha256"] == context["registration_sha256"],
                "Confirmation registration changed during execution",
            )
            result = dict(
                status="COMPLETE",
                phase=PHASE,
                split=SPLIT,
                task_id=task_id,
                parent=parent,
                shard=shard,
                shards=2,
                groups=192,
                responses=1536,
                generated_this_process=generated,
                actual_generated_responses_this_process=generated,
                recovered_raw_rows=0,
                reused_rows=0,
                optimizer_updates=0,
                registration_sha256=context["registration_sha256"],
                frozen_spec_sha256=context["spec_object_sha256"],
                generation_alignment_sha256=file_hash(alignment_path),
                training_allowed=False,
                gate_status="NOT_SIGNED",
            )
            write_once(output_path(root, f"CONFIRM/workers/{task_id}.json"), result)
            return result
        except Exception as exc:
            write_once(
                output_path(root, f"CONFIRM/failures/{task_id}.json"),
                dict(
                    status="TECHNICAL_FAILURE_STOP",
                    at=now(),
                    phase=PHASE,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", required=True, choices=PARENTS)
    parser.add_argument("--shard", required=True, type=int, choices=(0, 1))
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.parent, args.shard, args.task_id), ensure_ascii=False))


if __name__ == "__main__":
    main()
