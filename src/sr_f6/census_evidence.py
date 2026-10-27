"""Verify the complete S0 census against independently persisted native raw rows.

The frozen S0 producer is not modified. This later consumer recomputes its
annotations, checks actual worker coverage/accounting, and binds replay traces
to the original tokens. It returns inputs for review and scientific assessment;
it does not sign a gate or authorize training. The caller must bind its own
executing phase when publishing a report, never claim this validation ran in S0.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from srf6_ref.census import summarize

from . import census, phases
from .common import digest, safe_evidence


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _strict_json(payload):
    """Reject ambiguous containers; the model's full_text string stays untouched."""

    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "Duplicate JSON key in census evidence")
            value[key] = item
        return value

    def finite_number(token):
        value = float(token)
        _require(math.isfinite(value), "Nonfinite JSON number in census evidence")
        return value

    def reject_constant(_):
        raise PermissionError("Nonfinite JSON number in census evidence")

    return json.loads(
        payload,
        object_pairs_hook=unique,
        parse_float=finite_number,
        parse_constant=reject_constant,
    )


class _Reader:
    """Read each evidence payload once at the final census binding boundary."""

    def __init__(self, root):
        self.root = root
        self.inputs = {}

    def read(self, relative, *, lines=False):
        path = safe_evidence(self.root, relative)
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        _require(
            (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Census evidence changed while being read",
        )
        reference = {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
        _require(
            self.inputs.setdefault(relative, reference) == reference,
            "Census evidence changed between reads",
        )
        if lines:
            return [_strict_json(line) for line in payload.splitlines() if line.strip()]
        return _strict_json(payload)


def _inventory(root, relative, pattern):
    directory = root / relative
    _require(directory.is_dir() and not directory.is_symlink(), "Missing census evidence tree")
    return {str(path.relative_to(root)) for path in directory.glob(pattern)}


def _worker_partition(workers, public, frozen):
    """Require one complete, nonoverlapping partition for each registered parent."""
    assignments, shard_counts, shards_seen = {}, {}, set()
    for relative, worker in workers.items():
        task_id = worker.get("task_id", "")
        match = re.fullmatch(r"CENSUS-(b[012])-(0|[1-9][0-9]*)-of-([1-9][0-9]*)", task_id)
        _require(match is not None, "Invalid census worker identity")
        parent, shard, count = match.group(1), int(match.group(2)), int(match.group(3))
        _require(
            relative == f"CENSUS/workers/{task_id}.json"
            and 0 <= shard < count <= len(public)
            and shard_counts.setdefault(parent, count) == count
            and (parent, shard) not in shards_seen,
            "Mixed, duplicate or renamed census worker partition",
        )
        shards_seen.add((parent, shard))
        selected = public[shard::count]
        _require(
            worker.get("status") == "COMPLETE"
            and worker.get("phase") == "PARENT_CENSUS"
            and worker.get("frozen_spec_sha256") == digest(frozen)
            and worker.get("optimizer_updates") == 0
            and type(worker.get("optimizer_updates")) is int
            and worker.get("groups") == len(selected)
            and type(worker.get("groups")) is int,
            "Worker did not complete its exact registered partition",
        )
        generated = worker.get("actual_generated_responses_this_process")
        _require(
            type(generated) is int and 0 <= generated <= 8 * len(selected) and generated % 8 == 0,
            "Invalid process-only census call count",
        )
        for item in selected:
            key = parent, item["qid"]
            _require(key not in assignments, "Overlapping census worker coverage")
            assignments[key] = task_id
    _require(
        set(shard_counts) == set(census.PARENTS)
        and shards_seen == {(p, i) for p, count in shard_counts.items() for i in range(count)}
        and set(assignments) == {(p, row["qid"]) for p in census.PARENTS for row in public},
        "Incomplete census worker coverage",
    )
    return assignments


def _runtime_identity(runtime, task_id, parent, root, origin, frozen):
    for name, expected in {
        "run_id": task_id,
        "parent_id": parent,
        "run_root": str(root),
        "step": 0,
        "reference_parent": parent,
        "parent_content_hash": frozen["parent_policy_hashes"][parent],
        "source_sha256": origin["source_manifest"]["sha256"],
        "source_hash": origin["source_manifest"]["sha256"],
        "data_hash": origin["data_manifest"]["sha256"],
        "fresh_optimizer": False,
        "dtype": "float32",
        "probability_dtype": "float32",
        "sr_f6_protocol_version": census.PROTOCOL_VERSION,
    }.items():
        _require(runtime.get(name) == expected, "S0 runtime identity differs: " + name)
    tensor_hash = runtime.get("policy_hash")
    _require(
        isinstance(tensor_hash, str)
        and re.fullmatch(r"[a-f0-9]{64}", tensor_hash) is not None
        and runtime.get("reference_hash") == tensor_hash,
        "Census requires the loaded, unchanged parent tensor identity",
    )
    return tensor_hash


def _raw_identity(raw, runtime, task_id, parent, root, spec):
    expected = {
        "run_id": task_id,
        "run_root": str(root),
        "policy_hash": runtime["policy_hash"],
        "policy_tensor_hash": runtime["policy_hash"],
        "parent_content_hash": runtime["parent_content_hash"],
        "source_sha256": runtime["source_sha256"],
        "source_hash": runtime["source_sha256"],
        "data_hash": runtime["data_hash"],
        "step": 0,
        "policy_step": 0,
        "protocol": "structured_v4",
        "sr_f6_protocol_version": census.PROTOCOL_VERSION,
        "sampler_logprob_source": "actual_generate_raw_logits_selected_token",
    }
    _require(len(raw) == 8, "Original raw must contain exactly eight draws")
    for draw, row in enumerate(raw):
        _require(
            all(row.get(k) == v for k, v in expected.items()), "Original raw source/policy differs"
        )
        adapter = row.get("adapter_identity", {})
        _require(
            adapter.get("run_id") == task_id
            and adapter.get("parent") == parent
            and adapter.get("step") == 0,
            "Original raw adapter does not belong to the generating worker",
        )
        _require(
            isinstance(row.get("prompt_tensor_hash"), str)
            and re.fullmatch(r"[a-f0-9]{64}", row["prompt_tensor_hash"]) is not None
            and isinstance(row.get("native_prompt_inputs"), dict)
            and bool(row["native_prompt_inputs"])
            and row.get("finish_reason") in ("balanced", "eos", "length"),
            "Original native prompt/termination evidence is missing",
        )
        tokens, reason = row.get("tokens"), row["finish_reason"]
        _require(
            isinstance(tokens, list)
            and 1 <= len(tokens) <= 1536
            and row.get("raw_tokens") == tokens
            and row.get("completion_token_count") == len(tokens)
            and row.get("max_new_tokens") == 1536
            and row.get("group_size") == 8
            and row.get("group_row_index") == draw
            and row.get("group_seed") == spec["seeds"][0]
            and row.get("seed") == spec["seeds"][draw]
            and row.get("truncated") is (reason == "length")
            and (reason != "length" or len(tokens) == 1536)
            and row.get("balanced_token_count") == (len(tokens) if reason == "balanced" else None)
            and row.get("boundary_token_count") == (len(tokens) if reason == "balanced" else None),
            "Original raw draw/termination metadata contradicts native generation",
        )
        _require(
            all(
                type(x) in (int, float) and math.isfinite(x) and x <= 0
                for x in row.get("sampler_logprobs", [])
            ),
            "Original sampler log probabilities are invalid",
        )


def _accounting(accounting, task_id, parent, expected):
    """Compare recorded native events, retaining shared-seed multiplicities."""
    observed = Counter()
    for row in accounting:
        _require(
            row.get("task_id") == task_id
            and row.get("parent") == parent
            and type(row.get("count")) is int
            and row["count"] > 0
            and isinstance(row.get("metadata"), dict),
            "Malformed or mixed census call accounting",
        )
        kind, metadata, count = row["kind"], row["metadata"], row["count"]
        if kind == "completion_attempts":
            _require(set(metadata) == {"group_seed", "protocol"}, "Unexpected generation metadata")
            key = kind, metadata["group_seed"], metadata["protocol"], count
        elif kind == "generated_tokens":
            _require(set(metadata) == {"group_seed", "row_index"}, "Unexpected token metadata")
            key = kind, metadata["group_seed"], metadata["row_index"], count
        elif kind == "extra_forward_sequences":
            _require(
                set(metadata) == {"purpose", "completion_tokens"}, "Unexpected forward metadata"
            )
            key = kind, metadata["purpose"], metadata["completion_tokens"], count
        else:
            raise PermissionError("Unresolved census accounting kind: " + str(kind))
        observed[key] += 1
    _require(
        observed == expected,
        "Actual census event accounting differs from original raw/replay; preserve and diagnose",
    )


def _alignment(report, spec, records, tensor_hash):
    """Recompute FP32 diagnostics from saved traces, including the raw binding."""
    import torch

    from sr_f12.training import enforce_probability_gate, probability_diagnostics

    expected = {
        "frozen_spec_sha256": spec["frozen_spec_sha256"],
        "group_spec_sha256": digest(spec),
        "raw_trajectories_sha256": digest(
            [
                {k: row[k] for k in ("tokens", "sampler_logprobs", "prompt_tensor_hash")}
                for row in records
            ]
        ),
    }
    _require(
        report.get("identity") == expected
        and report.get("status") == "PASS"
        and report.get("execution_origin") == "SERVER_ACTUAL"
        and report.get("phase") == "PARENT_CENSUS"
        and report.get("parent") == spec["parent"]
        and report.get("group_id") == spec["group_id"]
        and report.get("parameter_tensor_sha256") == tensor_hash
        and report.get("original_registered_responses") == 8
        and report.get("checked_sequences") == 8
        and report.get("new_generated_responses") == 0
        and report.get("optimizer_updates") == 0,
        "Real first-G8 replay identity is incomplete or changed",
    )
    traces = report.get("traces")
    _require(
        isinstance(traces, list) and len(traces) == 8, "Exactly eight actual replay traces required"
    )
    differences, truncated = [], []
    for row, trace in zip(records, traces, strict=True):
        sampler = torch.tensor(row["sampler_logprobs"], dtype=torch.float32)
        _require(
            trace.get("row_id") == row["row_id"]
            and trace.get("token_ids") == row["tokens"]
            and trace.get("sampled_logprobs") == sampler.tolist(),
            "Replay trace is not the original sampled trajectory",
        )
        current = torch.tensor(trace.get("current_logprobs"), dtype=torch.float32)
        _require(
            current.shape == sampler.shape and bool(torch.isfinite(current).all()),
            "Invalid actual replay probability vector",
        )
        delta = current - sampler
        differences.append(delta.abs())
        truncated.append(delta > math.log(2.0))
    diagnostics = probability_diagnostics(differences, truncated)
    _require(report.get("diagnostics") == diagnostics, "Replay summary differs from actual traces")
    enforce_probability_gate(diagnostics)
    _require(
        diagnostics["sampler_train_max_absolute_difference"] <= 5,
        "Real token replay exceeds the registered maximum difference",
    )


def validate_saved_census(root, *, origin_deployment):
    """Return the full verified raw/annotated matrix for downstream independent review.

    This is deliberately all-or-nothing: never fill absent groups, repair raw,
    recompute a favorable seed, or infer missing calls from missing files. Source
    manifests reuse the existing verified metadata cache. Dynamic census files
    are read once because this is their first final binding boundary.
    """
    root = Path(root)
    origin = phases.verify_origin(root, deployment=origin_deployment)
    reader = _Reader(root)
    context = origin["identity"]
    config = reader.read("CONFIG.json")
    public, tasks = census.census_inputs(root)
    # The DATA manifest was verified above; retain its selected input references.
    reader.read("DATA/DEV_CAL/MODEL_INPUTS.jsonl", lines=True)
    reader.read("DATA/DEV_CAL/TASKS_PRIVATE.jsonl", lines=True)
    frozen = reader.read("CENSUS_SPEC_FROZEN.json")
    _require(
        census.validate_census_spec(root, config, public, tasks) == frozen,
        "Frozen census changed",
    )
    workers = {p: reader.read(p) for p in sorted(_inventory(root, "CENSUS/workers", "*.json"))}
    assignments = _worker_partition(workers, public, frozen)
    expected = {
        category: {
            f"CENSUS/{category}/{p}/{row['qid']}.{extension}"
            for p in census.PARENTS
            for row in public
        }
        for category, extension in (("groups", "json"), ("raw", "jsonl"), ("attempts", "json"))
    }
    for category, paths in expected.items():
        extension = "jsonl" if category == "raw" else "json"
        _require(
            _inventory(root, f"CENSUS/{category}", f"*/*.{extension}") == paths,
            "Unexpected or missing census " + category,
        )
    runtimes, rows_by_worker, first, events = {}, Counter(), {}, {}
    tensors, records, raw_by_row = {}, [], {}
    for parent in census.PARENTS:
        for item in public:
            qid = item["qid"]
            task_id = assignments[parent, qid]
            if task_id not in runtimes:
                runtimes[task_id] = reader.read(f"runtime_identity/{task_id}.json")
                events[task_id] = Counter()
                tensor = _runtime_identity(
                    runtimes[task_id], task_id, parent, root, context, frozen
                )
                _require(
                    tensors.setdefault(parent, tensor) == tensor,
                    "Parent workers loaded different policies",
                )
            spec = census.group_spec(config, parent, item, tasks[qid], frozen)
            attempt = reader.read(f"CENSUS/attempts/{parent}/{qid}.json")
            _require(
                attempt.get("status") == "STARTED"
                and attempt.get("task_id") == task_id
                and attempt.get("spec") == spec,
                "Original generation attempt differs from completed worker coverage",
            )
            raw = reader.read(f"CENSUS/raw/{parent}/{qid}.jsonl", lines=True)
            _raw_identity(raw, runtimes[task_id], task_id, parent, root, spec)
            rebuilt = census.annotate_group(raw, tasks[qid], spec)
            group = reader.read(f"CENSUS/groups/{parent}/{qid}.json")
            _require(
                group.get("status") == "COMPLETE"
                and group.get("spec") == spec
                and group.get("records") == rebuilt
                and group.get("credit") == census.counterfactual(rebuilt, spec),
                "Annotated group differs from the independently persisted original raw",
            )
            first.setdefault(task_id, (spec, rebuilt))
            rows_by_worker[task_id] += len(raw)
            events[task_id]["completion_attempts", spec["seeds"][0], "structured", 8] += 1
            events[task_id].update(
                ("generated_tokens", spec["seeds"][0], draw, len(row["tokens"]))
                for draw, row in enumerate(raw)
            )
            records.extend(rebuilt)
            # qid is derived from the frozen file/spec; original raw stays unchanged.
            raw_by_row.update(
                {
                    row["row_id"]: dict(value, qid=qid)
                    for row, value in zip(rebuilt, raw, strict=True)
                }
            )
    for worker in workers.values():
        task_id = worker["task_id"]
        alignment_path = f"CENSUS/alignment/{task_id}.json"
        alignment = reader.read(alignment_path)
        _require(
            worker.get("generation_alignment_sha256") == reader.inputs[alignment_path]["sha256"],
            "Worker replay content differs",
        )
        spec, first_records = first[task_id]
        _alignment(alignment, spec, first_records, tensors[spec["parent"]])
        accounting = reader.read(f"accounting/{task_id}.jsonl", lines=True)
        events[task_id].update(
            ("extra_forward_sequences", "census_token_alignment", len(row["tokens"]), 1)
            for row in first_records
        )
        _accounting(accounting, task_id, spec["parent"], events[task_id])
    expected_ids = {worker["task_id"] for worker in workers.values()}
    for directory, extension in (
        ("runtime_identity", "json"),
        ("accounting", "jsonl"),
        ("CENSUS/alignment", "json"),
    ):
        actual = _inventory(root, directory, f"CENSUS-*.{extension}")
        _require(
            actual == {f"{directory}/{name}.{extension}" for name in expected_ids},
            "Unresolved census worker evidence outside the completed partition",
        )
    _require(len(records) == len(raw_by_row) == 9216, "Exact full census denominator required")
    summary = summarize(records, config["master_seed"])
    summary.update(
        status="COMPLETE_NOT_GATE_SIGNED",
        frozen_spec_sha256=digest(frozen),
        source_rows=9216,
        optimizer_updates=0,
        real_measurement_review="NOT_RUN",
    )
    return {
        "status": "VERIFIED_STORED_CENSUS_NOT_GATE_SIGNED",
        "training_authorized": False,
        "model_calls_by_this_validator": 0,
        "optimizer_updates_by_this_validator": 0,
        "input_producer": {
            "phase": phases.ORIGIN,
            "identity": context,
            "identity_sha256": origin["identity_sha256"],
        },
        "summary": summary,
        "records": records,
        "raw_by_row": raw_by_row,
        "actual_recorded_calls": sum(rows_by_worker.values()),
        "inputs": [reader.inputs[p] for p in sorted(reader.inputs)],
        "limitation": (
            "Stored provenance and numerical consistency only; "
            "independent review and science gates remain separate."
        ),
    }
