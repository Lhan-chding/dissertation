"""Reconstruct the complete independent confirmation from immutable native facts.

This consumer signs no science gate and grants no model access. The earlier
5ca screen is a distinct producer. No old CENSUS_S0 evidence is consumed here.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from pathlib import Path

from . import confirm_census as census
from .census_evidence import _strict_json
from .common import digest, safe_evidence
from .interface_v3_protocol import VERSION as PROMPT_VERSION
from .interface_v3_screen import verify_saved_tokens
from .runtime import PROTOCOL_VERSION
from .source_binding import stat_cache_key, verify_bound_files

VERSION = "srf6-confirm-v3"
SCHEMA = "srf6-confirm-evidence-v1"
PARENTS = ("b0", "b1", "b2")
KINDS = (
    "raw",
    "groups",
    "attempts",
    "workers",
    "executions",
    "execution_slots",
    "alignment",
    "runtime_identity",
    "accounting",
    "failures",
)


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


class _Reader:
    """One read/hash per fact leaf, with a final unchanged-observation check."""

    def __init__(self, root):
        self.root = root
        self.inputs, self.external, self.values, self.stamps = {}, {}, {}, {}

    def _record(self, path, pin, stamp, resolved):
        relative = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else None
        entries, key = (self.inputs, relative) if relative else (self.external, str(path))
        ref = dict(path=key, sha256=pin)
        _require(entries.setdefault(key, ref) == ref, "Conflicting confirmation evidence pin")
        observation = (str(resolved), stamp)
        _require(
            self.stamps.setdefault(str(path), observation) == observation,
            "Confirmation evidence changed between observations",
        )

    def read(self, relative, *, lines=False, expected=None, opaque=False):
        path = safe_evidence(self.root, relative)
        if relative not in self.values:
            before = stat_cache_key(path.stat())
            payload = path.read_bytes()
            _require(
                stat_cache_key(path.stat()) == before,
                "Confirmation evidence changed while being read",
            )
            pin = hashlib.sha256(payload).hexdigest()
            self._record(path, pin, before, path.resolve(strict=True))
            self.values[relative] = (
                payload
                if opaque
                else (
                    [_strict_json(line) for line in payload.splitlines() if line.strip()]
                    if lines
                    else _strict_json(payload)
                )
            )
        if expected is not None:
            _require(self.inputs[relative]["sha256"] == expected, "Confirmation input pin differs")
        return self.values[relative]

    def reference(self, ref, *, lines=False, opaque=False):
        return self.read(ref["path"], lines=lines, opaque=opaque, expected=ref["sha256"])

    def inventory(self, rows, boundary):
        """Use verified binding metadata; never sweep unchanged model payloads."""
        if not rows:
            return
        unique = {}
        for row in rows:
            key = row["path"]
            _require(unique.setdefault(key, row) == row, "Conflicting bound inventory")
        checked = verify_bound_files(list(unique.values()), boundary=boundary)
        for row in checked["files"]:
            path = Path(row["path"])
            bound = unique[str(path)]
            self._record(path, row["sha256"], row["stat_cache_key"], Path(bound["resolved_path"]))

    def stable(self):
        for name, (resolved, stamp) in self.stamps.items():
            path = Path(name)
            if path.is_relative_to(self.root):
                safe_evidence(self.root, str(path.relative_to(self.root)))
            _require(
                str(path.resolve(strict=True)) == resolved and stat_cache_key(path.stat()) == stamp,
                "Confirmation evidence changed during validation",
            )


def _tree(root, relative, *, optional=False):
    directory = root / relative
    _require(not directory.is_symlink(), "Symlinked confirmation evidence tree")
    if optional and not directory.exists():
        return set()
    _require(directory.is_dir(), "Missing confirmation evidence tree: " + relative)
    result = set()
    for path in directory.rglob("*"):
        _require(not path.is_symlink(), "Symlinked confirmation evidence")
        _require(path.is_file() or path.is_dir(), "Nonregular confirmation evidence")
        if path.is_file():
            result.add(str(path.relative_to(root)))
    return result


def _snapshot(root):
    _require(not (root / "CONFIRM").is_symlink(), "Symlinked confirmation root")
    result = {kind: _tree(root, f"CONFIRM/{kind}", optional=kind == "failures") for kind in KINDS}
    result["submissions"] = _tree(root, "SUBMISSIONS") - {"SUBMISSIONS/dispatch.lock"}
    return result


def _provenance(reader, context):
    """Bind the new producer and its verified upstream inventories separately."""
    from . import confirm_registration as registration

    record = reader.read("REGISTRATION.json", expected=context["registration_sha256"])
    _require(record == context["registration"], "Loaded registration changed")
    for key in ("config", "source_manifest", "spec", "data_manifest"):
        reader.reference(record[key])
    _require(
        reader.values[record["spec"]["path"]] == context["spec"]
        and reader.values[record["config"]["path"]] == context["config"]
        and reader.values[record["source_manifest"]["path"]] == context["source_manifest"],
        "Loaded confirmation inputs differ from their saved bytes",
    )
    manifest = reader.values[record["data_manifest"]["path"]]
    reader.inventory(
        [dict(row, path=str(reader.root / row["path"])) for row in manifest["files"]], reader.root
    )
    # These small receipts close the explicit evidence chain. Their referenced
    # images/model files reuse existing content-bound inventories below.
    seal = reader.read("PREFLIGHT/SEAL.json")
    processor = reader.reference(seal["processor"])
    visual = reader.reference(seal["visual"])
    reader.reference(visual["review_evidence"], opaque=True)
    native = reader.reference(processor["native_regression"])
    reader.reference(native["log"], opaque=True)
    for image in processor["images"]:
        path = safe_evidence(reader.root, image["processed_file"])
        before = stat_cache_key(path.stat())
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            pin == image["processed_file_sha256"] and before == stat_cache_key(path.stat()),
            "Reviewed processor image changed",
        )
        reader._record(path, pin, before, path.resolve(strict=True))
    _require(
        context["preflight"] == dict(processor, visual=visual, seal=seal),
        "Loaded processor/visual evidence differs",
    )
    binding = context["source_binding"]
    inventories = [
        context["source_manifest"]["files"],
        record["upstream_files"],
        binding["model_files"],
        processor["verified_dependencies"],
    ]
    inventories.extend(binding["parents"][parent]["files"] for parent in PARENTS)
    audit = reader.read("DATA/AUDIT.json")
    inventories.append(audit["historical_world_exclusion"]["bound_files"])
    merged = {}
    for rows in inventories:
        for row in rows:
            prior = merged.setdefault(row["path"], row)
            _require(
                all(prior[key] == row[key] for key in ("sha256", "bytes", "resolved_path")),
                "Conflicting original dependency pins",
            )
    reader.inventory(list(merged.values()), registration.PROJECT)
    deployment = Path(context["deployment"])
    _require(
        deployment == Path(__file__).resolve(strict=True).parents[2]
        and deployment != Path(record["predecessor_screen"]["deployment"]),
        "Confirmation consumer must execute its new registered source",
    )
    return dict(
        phase=census.PHASE,
        deployment=str(deployment),
        revision=record["revision"],
        source_manifest=copy.deepcopy(record["source_manifest"]),
    )


def _workers(reader, context, snapshot):
    public, spec = context["public"], context["spec"]
    _require(
        len(public) == 384
        and len({row["qid"] for row in public}) == 384
        and public == sorted(public, key=lambda row: row["qid"]),
        "Exact sorted 384 confirmation questions required",
    )
    workers, assignments, shards = {}, {}, set()
    jobs = {}
    _require(
        snapshot["submissions"]
        == {
            f"SUBMISSIONS/{parent}.{kind}.json"
            for parent in PARENTS
            for kind in ("attempt", "result")
        },
        "Missing or unresolved additional confirmation submission",
    )
    for parent in PARENTS:
        attempt = reader.read(f"SUBMISSIONS/{parent}.attempt.json")
        result = reader.read(f"SUBMISSIONS/{parent}.result.json")
        _require(
            attempt.get("parent") == result.get("parent") == parent
            and attempt.get("registration_sha256") == context["registration_sha256"]
            and attempt.get("status") == "SUBMISSION_INTENT_NOT_COMPLETION"
            and result.get("status") == "SUBMITTED"
            and type(result.get("returncode")) is int
            and result["returncode"] == 0
            and isinstance(result.get("job_id"), str)
            and re.fullmatch(r"[0-9]+", result["job_id"])
            and result.get("stdout", "").strip().split(";", 1)[0] == result["job_id"],
            "Confirmation worker requires its original successful submission receipt",
        )
        deployment, root = Path(context["deployment"]), reader.root
        _require(
            attempt.get("argv")
            == [
                "sbatch",
                "--parsable",
                "--no-requeue",
                f"--job-name=LH-SRF6-Confirm-{parent}",
                f"--chdir={root}",
                f"--output={root / 'slurm'}/confirm-{parent}-%j.log",
                f"--export=ALL,SRF6_CODE={deployment},SRF6_ROOT={root},SRF6_PARENT={parent}",
                str(deployment / "scripts/sr_f6/confirm.sbatch"),
            ],
            "Original submission command differs from registered confirmation deployment",
        )
        jobs[parent] = result["job_id"]
    _require(len(set(jobs.values())) == 3, "Three distinct original parent submissions required")
    for relative in sorted(snapshot["workers"]):
        worker = reader.read(relative)
        task_id = worker.get("task_id", "")
        match = re.fullmatch(r"LH-SRF6-Confirm-([0-9]+)-(b[012])-([01])", task_id)
        _require(match is not None, "Invalid confirmation worker identity")
        job, parent, shard = match.group(1), match.group(2), int(match.group(3))
        _require(
            relative == f"CONFIRM/workers/{task_id}.json"
            and jobs[parent] == job
            and (parent, shard) not in shards,
            "Duplicate, renamed or unsubmitted confirmation worker",
        )
        shards.add((parent, shard))
        selected = public[shard::2]
        planned = [
            row for row in spec["workers"] if row["parent"] == parent and row["shard"] == shard
        ]
        _require(
            len(planned) == 1
            and planned[0]["qids"] == [item["qid"] for item in selected]
            and planned[0]["first_qid"] == selected[0]["qid"],
            "Frozen worker question order differs",
        )
        expected = dict(
            status="COMPLETE",
            phase=census.PHASE,
            split=census.SPLIT,
            parent=parent,
            shard=shard,
            shards=2,
            groups=192,
            responses=1536,
            generated_this_process=1536,
            actual_generated_responses_this_process=1536,
            recovered_raw_rows=0,
            reused_rows=0,
            optimizer_updates=0,
            registration_sha256=context["registration_sha256"],
            frozen_spec_sha256=context["spec_object_sha256"],
            training_allowed=False,
            gate_status="NOT_SIGNED",
        )
        _require(
            all(
                type(worker.get(key)) is type(value) and worker[key] == value
                for key, value in expected.items()
            ),
            "Worker did not complete its exact original confirmation shard",
        )
        execution = reader.read(f"CONFIRM/executions/{task_id}.json")
        slot = reader.read(f"CONFIRM/execution_slots/{parent}-{shard}.json")
        _require(
            execution == slot
            and all(
                type(execution.get(key)) is type(value) and execution[key] == value
                for key, value in dict(
                    status="STARTED",
                    phase=census.PHASE,
                    task_id=task_id,
                    parent=parent,
                    shard=shard,
                    shards=2,
                    registration_sha256=context["registration_sha256"],
                ).items()
            ),
            "Original execution slot or attempt differs",
        )
        workers[task_id] = worker
        assignments.update({(parent, item["qid"]): task_id for item in selected})
    _require(
        shards == {(parent, shard) for parent in PARENTS for shard in (0, 1)} and len(workers) == 6,
        "Exactly six complete confirmation workers required",
    )
    return workers, assignments


def _runtime(value, worker, context):
    parent, shard, task = worker["parent"], worker["shard"], worker["task_id"]
    expected = dict(
        run_id=task,
        task_id=task,
        parent_id=parent,
        phase=census.PHASE,
        confirmation_version=VERSION,
        run_root=str(context["root"]),
        registration_root=str(context["root"]),
        source_root=str(context["source_root"]),
        registration_sha256=context["registration_sha256"],
        source_sha256=context["registration"]["source_manifest_sha256"],
        data_hash=context["registration"]["data_sha256"],
        parent_content_hash=context["registration"]["parent_content_hashes"][parent],
        step=0,
        optimizer_updates=0,
        fresh_optimizer=False,
        optimizer=None,
        scheduler=None,
        trainable_parameters=[],
        dtype="float32",
        probability_dtype="float32",
        input_float_dtype="float32",
        forward_parameter_dtypes=["float32"],
        prompt_version=PROMPT_VERSION,
        sr_f6_protocol_version=PROTOCOL_VERSION,
        execution_mode="READONLY_SCHEMA_CONFIRM",
        input_plan_sha256=context["spec_object_sha256"],
        worker_id=f"{parent}-s{shard}-of-2",
        shard=shard,
        first_qid=context["public"][shard]["qid"],
        source_binding_status="REGISTERED_INDEPENDENT_CONFIRMATION_ONLY",
    )
    _require(
        all(
            type(value.get(key)) is type(item) and value[key] == item
            for key, item in expected.items()
        ),
        "Actual confirmation runtime identity differs",
    )
    _require(
        isinstance(value.get("policy_hash"), str)
        and re.fullmatch(r"[a-f0-9]{64}", value["policy_hash"])
        and isinstance(value.get("forward_float_buffer_dtypes"), list)
        and all(item == "float32" for item in value["forward_float_buffer_dtypes"]),
        "Actual frozen FP32 parent tensor identity required",
    )
    hardware = value.get("hardware", {})
    _require(
        type(hardware.get("cuda_visible_device_count")) is int
        and hardware["cuda_visible_device_count"] == 1
        and isinstance(hardware.get("cuda_uuid"), str)
        and bool(hardware["cuda_uuid"]),
        "Actual single-GPU runtime evidence required",
    )
    return value["policy_hash"]


def _native(raw, runtime, task_id, root, spec, route, native_cache):
    import torch

    from mm_core.training import state_hash

    expected = dict(
        run_id=task_id,
        task_id=task_id,
        run_root=str(root),
        root=str(root),
        policy_hash=runtime["policy_hash"],
        policy_tensor_hash=runtime["policy_hash"],
        source_hash=runtime["source_sha256"],
        confirmation_version=VERSION,
        protocol_revision="v2",
        backend_protocol="structured",
    )
    for row in raw:
        _require(
            all(row.get(key) == value for key, value in expected.items()),
            "Original raw worker/source/tensor identity differs",
        )
        _require(
            row.get("adapter_identity", {}).get("run_id") == task_id,
            "Original adapter belongs to another execution",
        )
        _require(
            all(
                type(row.get(key)) is int
                for key in (
                    "step",
                    "policy_step",
                    "seed",
                    "group_seed",
                    "group_size",
                    "group_row_index",
                    "completion_token_count",
                    "max_new_tokens",
                )
            ),
            "Original native integer identity fields are malformed",
        )
        reason, tokens = row["finish_reason"], row["tokens"]
        _require(
            row.get("truncated") is (reason == "length")
            and row.get("balanced_token_count") == (len(tokens) if reason == "balanced" else None)
            and row.get("boundary_token_count") == (len(tokens) if reason == "balanced" else None),
            "Native termination metadata contradicts original tokens",
        )
        native = row.get("native_prompt_inputs")
        _require(
            isinstance(native, dict)
            and set(native)
            == {"input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw"},
            "Original native prompt tensors are missing",
        )
        key = digest(native)
        cached = native_cache.get(spec["qid"])
        if cached is None:
            for name, values in native.items():
                tensor = torch.tensor(values)
                _require(
                    tensor.dtype == torch.int64
                    and tensor.ndim == 2
                    and state_hash(tensor) == route["native_input_tensor_hashes"][name],
                    "Original native prompt tensor differs from preflight",
                )
            native_cache[spec["qid"]] = key
        else:
            _require(key == cached, "Original native prompt changed across draws/parents")


def _accounting(ledger, task_id, parent, pin, expected):
    observed, totals = [], Counter()
    for entry in ledger:
        metadata = entry.get("metadata", {})
        _require(
            entry.get("task_id") == task_id
            and entry.get("parent") == parent
            and type(entry.get("count")) is int
            and entry["count"] > 0
            and isinstance(metadata, dict)
            and metadata.get("phase") == census.PHASE
            and metadata.get("confirmation_version") == VERSION
            and metadata.get("registration_sha256") == pin,
            "Malformed or mixed confirmation event accounting",
        )
        fields = {"qid", "phase", "confirmation_version", "registration_sha256"}
        kind = entry.get("kind")
        if kind == "completion_attempts":
            fields |= {"group_seed", "protocol"}
            _require(
                type(metadata.get("group_seed")) is int,
                "Original generation seed must be an integer",
            )
            event = (
                metadata["qid"],
                kind,
                metadata.get("group_seed"),
                metadata.get("protocol"),
                entry["count"],
            )
        elif kind == "generated_tokens":
            fields |= {"group_seed", "row_index"}
            _require(
                type(metadata.get("group_seed")) is int and type(metadata.get("row_index")) is int,
                "Original token event indices must be integers",
            )
            event = (
                metadata["qid"],
                kind,
                metadata.get("group_seed"),
                metadata.get("row_index"),
                entry["count"],
            )
        elif kind == "extra_forward_sequences":
            fields |= {"purpose", "completion_tokens"}
            _require(
                type(metadata.get("completion_tokens")) is int,
                "Original replay length must be an integer",
            )
            event = (
                metadata["qid"],
                kind,
                metadata.get("purpose"),
                metadata.get("completion_tokens"),
                entry["count"],
            )
        else:
            raise PermissionError("Unexpected confirmation event: " + str(kind))
        _require(set(metadata) == fields, "Unexpected confirmation event metadata")
        observed.append(event)
        totals[kind] += entry["count"]
    _require(observed == expected, "Actual ordered events differ from original draws/alignment")
    return dict(totals)


def _tokenizer(context):
    from sr_f5.runtime import Runtime

    runtime = Runtime.processor_only(
        context["config"]["model"]["model_path"], protocol_revision="v2"
    )
    for key in ("processor_hash", "chat_template_hash"):
        _require(
            runtime.identity[key] == context["preflight"]["identity"][key],
            "Actual token decoder differs from registered processor",
        )
    return runtime.processor.tokenizer, runtime.eos_ids


def validate_saved_confirm(root):
    """Return all 9,216 reconstructed rows and bound facts, without signing gates."""
    from .confirm_registration import verify_registered

    root = Path(root)
    _require(
        root.is_absolute() and root.resolve(strict=True) == root and not root.is_symlink(),
        "Canonical independent confirmation root required",
    )
    context = verify_registered(root, require_preflight=True)
    _require(
        Path(context["root"]) == root and context["spec"] == context["plan"],
        "Independent confirmation context differs",
    )
    reader = _Reader(root)
    producer = _provenance(reader, context)
    snapshot = _snapshot(root)
    _require(not snapshot["failures"], "Preserved technical failure blocks confirmation completion")
    workers, assignments = _workers(reader, context, snapshot)
    for kind, suffix in (("raw", "jsonl"), ("groups", "json"), ("attempts", "json")):
        wanted = {
            f"CONFIRM/{kind}/{p}/{r['qid']}.{suffix}" for p in PARENTS for r in context["public"]
        }
        _require(snapshot[kind] == wanted, "Missing or extra original confirmation " + kind)
    for kind in ("workers", "executions", "alignment", "runtime_identity", "accounting"):
        suffix = "jsonl" if kind == "accounting" else "json"
        _require(
            snapshot[kind] == {f"CONFIRM/{kind}/{task}.{suffix}" for task in workers},
            "Unresolved confirmation execution evidence: " + kind,
        )
    _require(
        snapshot["execution_slots"]
        == {f"CONFIRM/execution_slots/{p}-{s}.json" for p in PARENTS for s in (0, 1)},
        "Missing or extra consumed confirmation execution slot",
    )
    routes = {row["qid"]: row["routing"] for row in context["preflight"]["inputs"]}
    records, raw_by_row, tensors, native_cache, totals = [], {}, {}, {}, Counter()
    for task_id, worker in workers.items():
        parent, shard = worker["parent"], worker["shard"]
        runtime = reader.read(f"CONFIRM/runtime_identity/{task_id}.json")
        tensor = _runtime(runtime, worker, context)
        _require(
            tensors.setdefault(parent, tensor) == tensor, "Parent shards loaded different tensors"
        )
        events, first = [], None
        for index, item in enumerate(context["public"][shard::2]):
            qid = item["qid"]
            _require(assignments[parent, qid] == task_id, "Overlapping confirmation assignment")
            spec = census.group_spec(context, parent, item)
            task = context["private"][qid]
            stem = f"{parent}/{qid}"
            attempt = reader.read(f"CONFIRM/attempts/{stem}.json")
            _require(
                attempt.get("status") == "STARTED"
                and attempt.get("task_id") == task_id
                and attempt.get("spec") == spec,
                "Original G8 attempt differs",
            )
            raw = reader.read(f"CONFIRM/raw/{stem}.jsonl", lines=True)
            rebuilt = census.annotate(raw, spec, task, preflight_route=routes[qid])
            _native(raw, runtime, task_id, root, spec, routes[qid], native_cache)
            group = reader.read(f"CONFIRM/groups/{stem}.json")
            _require(
                group.get("records") == rebuilt,
                "Saved labels differ from original raw reconstruction",
            )
            census.validate_group(group, spec, task, preflight_route=routes[qid])
            events.append((qid, "completion_attempts", spec["seeds"][0], "structured", 8))
            events.extend(
                (qid, "generated_tokens", spec["seeds"][0], draw, len(row["tokens"]))
                for draw, row in enumerate(raw)
            )
            if index == 0:
                first = spec, rebuilt
                events.extend(
                    (
                        qid,
                        "extra_forward_sequences",
                        "confirm_token_alignment",
                        len(row["tokens"]),
                        1,
                    )
                    for row in raw
                )
            for row, original in zip(rebuilt, raw, strict=True):
                _require(row["row_id"] not in raw_by_row, "Duplicate original confirmation row")
                raw_by_row[row["row_id"]] = dict(original, qid=qid)
            records.extend(rebuilt)
        alignment_path = f"CONFIRM/alignment/{task_id}.json"
        alignment = reader.read(alignment_path, expected=worker["generation_alignment_sha256"])
        census.validate_alignment(alignment, *first)
        ledger = reader.read(f"CONFIRM/accounting/{task_id}.jsonl", lines=True)
        totals.update(_accounting(ledger, task_id, parent, context["registration_sha256"], events))
    _require(
        len(records) == len(raw_by_row) == totals["completion_attempts"] == 9216
        and totals["extra_forward_sequences"] == 48,
        "Exact confirmation denominator required",
    )
    tokenizer, eos_ids = _tokenizer(context)
    token_check = verify_saved_tokens(list(raw_by_row.values()), tokenizer, eos_ids)
    after = verify_registered(root, require_preflight=True)
    _require(
        after["registration_sha256"] == context["registration_sha256"]
        and _snapshot(root) == snapshot,
        "Confirmation context or evidence membership changed",
    )
    reader.stable()
    return dict(
        schema=SCHEMA,
        status="VERIFIED_STORED_CONFIRMATION_NOT_GATE_SIGNED",
        phase=census.PHASE,
        split=census.SPLIT,
        registration_sha256=context["registration_sha256"],
        registration=copy.deepcopy(context["registration"]),
        config=copy.deepcopy(context["config"]),
        source_rows=9216,
        records=records,
        raw_by_row=raw_by_row,
        input_producer=producer,
        consumer_source=copy.deepcopy(producer),
        predecessor_screen=copy.deepcopy(context["registration"]["predecessor_screen"]),
        parent_tensor_hashes=tensors,
        actual_recorded_calls=9216,
        accounting=dict(totals),
        saved_token_validation=token_check,
        inputs=[reader.inputs[key] for key in sorted(reader.inputs)],
        external_inputs=[reader.external[key] for key in sorted(reader.external)],
        training_authorized=False,
        evaluation_authorized=False,
        generation_authorized=False,
        gate_status="NOT_SIGNED",
        model_calls_by_this_validator=0,
        optimizer_updates_by_this_validator=0,
        limitation=(
            "Stored native facts reconstructed; independent review and scientific "
            "admission remain separate."
        ),
    )
