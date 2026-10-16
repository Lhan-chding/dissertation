"""Fixed, prospectively registered zero-training protocol serialization ablation."""

from __future__ import annotations

import getpass
import json
import os
import re
from collections import Counter
from pathlib import Path

from sr_f12.evaluation import durable_json, group_slots, shard_lease

from . import contract
from .evaluation import engine_slots, validate_record
from .protocol import PARENTS, digest, file_hash, read_json, verify_run

ABLATION_ID = "SR-F3-CE-PROTOCOL-FIX-20261016"
VARIANTS = {"P1": "v1", "P2": "v2", "P3": "c2"}
RULE = dict(
    denominator_per_parent_variant=256,
    threshold=231,
    outputs=2304,
    p1_must_fail_blocks=[0, 2],
    priority=["P2", "P3"],
    decision_metric="value_parseable",
)
BASE_FILES = (
    "RUN_IDENTITY.json",
    "deployment/SOURCE_MANIFEST.json",
    "plan/PACKAGE_SHA256.json",
    "plan/config/SR_F3_CE.json",
    "plan/manifests/MODEL_INPUTS.jsonl",
    "plan/manifests/TASKS_AUDIT_ONLY.jsonl",
    "data/IMAGE_MANIFEST.jsonl",
    "parents/PARENT_AUDIT.json",
    "prior/MODEL_ENVIRONMENT_IDENTITY.json",
    "native/PROCESSOR_RECEIPT.json",
    "native/protocol_contexts/v1.json",
    "native/protocol_contexts/v2.json",
    "native/protocol_contexts/c2.json",
)
CANDIDATE_FILES = (
    "PACKAGE_SHA256.json",
    "PROTOCOL_REVISION.json",
    "config/SR_F3_CE.json",
    "manifests/MODEL_INPUTS.jsonl",
    "manifests/TASKS_AUDIT_ONLY.jsonl",
)
REQUIRED_SOURCE_FILES = (
    "scripts/sr_f3_ce/run_protocol_ablation.py",
    *[
        f"src/sr_f3_ce/{name}.py"
        for name in (
            "__init__",
            "ablation",
            "contract",
            "data",
            "evaluation",
            "native_audit",
            "prompts",
            "protocol",
            "protocol_revision",
            "runner",
            "runtime",
        )
    ],
    "src/sr_f12/evaluation.py",
    "src/sr_f12/runtime.py",
    "src/sr_f12/runner.py",
    "src/sr_f1/runtime.py",
    "src/mm_core/vl_runtime.py",
    "src/mm_core/training.py",
    "src/mm_dev/runtime.py",
)


def task_id(block, variant):
    if type(block) is not int or block not in range(3) or variant not in VARIANTS:
        raise ValueError("Unregistered ablation block/variant")
    return f"ablation--{variant}--b{block}"


def job_name(block, variant):
    task_id(block, variant)
    return f"SRF3CE-ABLATION-{variant}-b{block}-LH"


def task_matrix():
    return {
        task_id(block, variant): dict(
            block=block,
            variant=variant,
            parent_id=PARENTS[block],
            protocol_revision=revision,
            job_name=job_name(block, variant),
            outputs=256,
        )
        for variant, revision in VARIANTS.items()
        for block in range(3)
    }


def slots_for(tasks):
    from .prompts import prefill_for

    original = [
        s for s in engine_slots(tasks) if s["protocol"] == "structured" and s["view"] is None
    ]
    if len(original) != 768:
        raise ValueError("Original structured ENGINE coverage changed")
    return [
        dict(
            slot,
            phase="PROTOCOL_ABLATION",
            ablation_variant=variant,
            protocol_revision=revision,
            assistant_prefill=prefill_for("structured", revision),
            original_slot_id=slot["slot_id"],
            slot_id=digest([ABLATION_ID, variant, slot["slot_id"]]),
        )
        for variant, revision in VARIANTS.items()
        for slot in original
    ]


def groups_for(slots):
    result = []
    for variant in VARIANTS:
        for group in group_slots([s for s in slots if s["ablation_variant"] == variant]):
            result.append(
                dict(
                    group,
                    original_group_id=group["group_id"],
                    group_id=digest([ABLATION_ID, variant, group["group_id"]]),
                )
            )
    return result


def _contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Ablation artifact must be a safe relative path")
    path = (Path(root) / relative).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve(strict=True)) or not path.is_file():
        raise PermissionError("Ablation artifact escapes its bound root")
    return path


def _record(root, path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve(strict=True)):
        raise PermissionError("Ablation record escapes its bound root")
    return dict(path=str(path.relative_to(root)), sha256=file_hash(path), bytes=path.stat().st_size)


def _verify_record(root, record):
    path = _contained(root, record["path"])
    if _record(root, path) != record:
        raise PermissionError("Ablation artifact bytes changed: " + record["path"])
    return path


def _registration(root):
    """Build input-only requests; audit truth never supplies a model input."""
    from .native_audit import verify_protocol_contexts
    from .prompts import prefill_for, prompt_for_revision
    from .protocol_revision import verify_revision_package
    from .runner import dataset_and_store

    dataset, store = dataset_and_store(root)
    slots = slots_for(dataset.tasks)
    qids = sorted({slot["qid"] for slot in slots})
    requests, candidates = {}, {}
    for variant, revision in VARIANTS.items():
        context = verify_protocol_contexts(root, revision)
        if any(
            context.get(key) != value
            for key, value in dict(
                status="PASS",
                protocol_revision=revision,
                unique_contexts=64,
                parent_panel_contexts=192,
                model_calls=0,
                weights_loaded=False,
                teacher_forward_calls=0,
                assistant_prefill=prefill_for("structured", revision),
            ).items()
        ):
            raise PermissionError("Exact native context acceptance is incomplete")
        plan = root / "plan" if revision == "v1" else root / "candidates" / revision / "plan"
        if revision != "v1":
            verify_revision_package(plan, source_plan_dir=root / "plan")
        inputs = {
            row["qid"]: row
            for row in map(
                json.loads, (plan / "manifests/MODEL_INPUTS.jsonl").read_text().splitlines()
            )
        }
        request_rows = {}
        for qid in qids:
            original = store.model_input(qid, protocol="structured")
            request = dict(original, prompt=prompt_for_revision(original["prompt"], revision))
            if request["prompt"] != inputs[qid]["prompt"]:
                raise PermissionError("Candidate prompt differs from the registered revision")
            if context["contexts"][qid]["model_input_sha256"] != digest(request):
                raise PermissionError("CPU accepted context differs from the actual model input")
            request_rows[qid] = request
        requests[variant] = request_rows
        candidates[variant] = dict(
            protocol_revision=revision,
            assistant_prefill=prefill_for("structured", revision=revision),
            plan=str(plan.relative_to(root)),
            package_sha256=file_hash(plan / "PACKAGE_SHA256.json"),
            config_sha256=file_hash(plan / "config/SR_F3_CE.json"),
            model_inputs_sha256=file_hash(plan / "manifests/MODEL_INPUTS.jsonl"),
            requests_sha256=digest(request_rows),
            native_contexts={
                qid: dict(
                    prompt_tensor_hash=context["contexts"][qid]["prompt_tensor_hash"],
                    assistant_prefill_token_ids=context["contexts"][qid]["routing"][
                        "assistant_prefill_token_ids"
                    ],
                )
                for qid in qids
            },
        )
    return dataset.tasks, slots, requests, candidates


def _bound_names():
    return [
        *BASE_FILES,
        *[
            f"candidates/{revision}/plan/{name}"
            for revision in ("v2", "c2")
            for name in CANDIDATE_FILES
        ],
    ]


def _ablation_run(root):
    from .protocol_revision import revision_from_config

    identity, config = verify_run(root)
    if identity.get("run_stage") != "PROTOCOL_ABLATION" or revision_from_config(config) != "v1":
        raise PermissionError("A separate original-v1 PROTOCOL_ABLATION run is required")
    code = root / "code"
    inventory = read_json(root / "deployment/SOURCE_MANIFEST.json")
    if (
        not re.fullmatch(r"[a-f0-9]{40}", str(identity.get("code_commit")))
        or identity["code_commit"] != inventory.get("code_commit")
        or identity.get("bound_files", {}).get("deployment/SOURCE_MANIFEST.json")
        != file_hash(root / "deployment/SOURCE_MANIFEST.json")
    ):
        raise PermissionError("Ablation source commit and run-bound inventory differ")
    entries = inventory["files"]
    names = [row["path"] for row in entries]
    actual_python = set()
    for path in code.rglob("*"):
        if path.is_symlink() or path.suffix in (".pyc", ".pyo"):
            raise PermissionError("Ablation source contains a symlink or unmanifested bytecode")
        if path.is_file() and path.suffix == ".py":
            actual_python.add(path.relative_to(code).as_posix())
    if (
        len(set(names)) != len(names)
        or not set(REQUIRED_SOURCE_FILES) <= set(names)
        or (
            actual_python != {name for name in names if name.endswith(".py")}
            or file_hash(__file__) != file_hash(code / "src/sr_f3_ce/ablation.py")
        )
    ):
        raise PermissionError("Complete ablation entry/dependency source must be manifest-bound")
    return identity


def _parent_identity(root, task, model, adapter):
    """Authenticate parent provenance independently of completion self-report."""
    prior = read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    audit = read_json(root / "parents/PARENT_AUDIT.json")
    parents = [row for row in audit["parents"] if row["parent_id"] == task["parent_id"]]
    if audit.get("status") != "PASS" or len(parents) != 1:
        raise PermissionError("Ablation parent audit is incomplete or ambiguous")
    parent = parents[0]
    adapter_path = (root / parent["adapter_path"]).resolve(strict=True)
    if not adapter_path.is_relative_to(root / "parents"):
        raise PermissionError("Parent adapter is outside its frozen copied tree")
    adapter_names = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "IDENTITY.json",
    }
    required_copies = {str(Path(parent["adapter_path"]) / name) for name in adapter_names}
    copies = [row for row in parent.get("copied_files", []) if row["path"] in required_copies]
    if len(copies) != len(adapter_names) or {row["path"] for row in copies} != required_copies:
        raise PermissionError("Frozen parent adapter copy inventory is incomplete")
    for copy in copies:
        _verify_record(root, copy)
    adapter_manifest = read_json(adapter_path / "IDENTITY.json")
    if adapter_manifest.get("policy_hash") != parent["policy_hash"] or (
        adapter_manifest.get("files")
        != {name: file_hash(adapter_path / name) for name in adapter_names - {"IDENTITY.json"}}
    ):
        raise PermissionError("Copied parent adapter differs from its original identity")
    expected_model = dict(
        parent_id=task["parent_id"],
        protocol_revision=task["protocol_revision"],
        base_model_weights_hash=prior["model_weights_hash"],
    )
    expected_model.update(
        {key: prior[key] for key in ("processor_hash", "tokenizer_hash", "chat_template_hash")}
    )
    if any(model.get(k) != v for k, v in expected_model.items()) or (
        adapter.get("base_model_weights_hash") != prior["model_weights_hash"]
        or adapter.get("adapter_parameter_hash") != parent["policy_hash"]
        or Path(adapter.get("adapter_path", "")).resolve() != adapter_path
    ):
        raise PermissionError("Ablation raw model/adapter differs from frozen parent provenance")


def freeze_ablation(root, *, previous_root, instruction, stop_decision):
    from .protocol_revision import REQUEST_SHA256

    root, previous_root = Path(root).resolve(strict=True), Path(previous_root).resolve(strict=True)
    identity = _ablation_run(root)
    if (
        root == previous_root
        or root.is_relative_to(previous_root)
        or previous_root.is_relative_to(root)
    ):
        raise PermissionError("Ablation requires a separate root from its stopped predecessor")
    if any(
        (root / name).exists()
        for name in ("STOP", "runs", "sealed", "EXECUTION_FREEZE.json", "ablation/raw")
    ):
        raise PermissionError("Ablation manifest must precede new generation/training")
    instruction = _contained(root, instruction)
    if instruction != root / "authorization/REQUEST.md" or (
        file_hash(instruction) != REQUEST_SHA256
        or identity.get("bound_files", {}).get("authorization/REQUEST.md") != REQUEST_SHA256
    ):
        raise PermissionError("The adopted protocol instruction must be bound before ablation")
    if str(stop_decision) in ("STOP", "RUN_IDENTITY.json", "deployment/SOURCE_MANIFEST.json"):
        raise PermissionError("A separate original stop-decision artifact is required")
    previous = {
        name: _record(previous_root, _contained(previous_root, name))
        for name in (
            "STOP",
            "RUN_IDENTITY.json",
            "deployment/SOURCE_MANIFEST.json",
            str(stop_decision),
        )
    }
    tasks, slots, requests, candidates = _registration(root)
    value = dict(
        schema_version=1,
        ablation_id=ABLATION_ID,
        status="FROZEN_BEFORE_ABLATION",
        root=str(root),
        decision_rule=RULE,
        variants=VARIANTS,
        slots=slots,
        requests=requests,
        candidates=candidates,
        instruction=_record(root, instruction),
        bound_files=[_record(root, _contained(root, name)) for name in _bound_names()],
        previous_root=str(previous_root),
        previous_files=previous,
        stop_decision=str(stop_decision),
        task_sidecars={qid: digest(tasks[qid]) for qid in sorted(requests["P1"])},
        tasks=task_matrix(),
        optimizer_steps=0,
        teacher_forward_calls=0,
        monitor_calls=0,
        selection_uses_only="value_parseable",
        sampling_semantics=(
            "original ENGINE group_seed initializes one batched RNG; "
            "original draw seeds are identities"
        ),
    )
    durable_json(root / "ABLATION_MANIFEST.json", value)
    return value


def verify_manifest(root):
    from .protocol_revision import REQUEST_SHA256

    root = Path(root).resolve(strict=True)
    identity = _ablation_run(root)
    manifest = read_json(root / "ABLATION_MANIFEST.json")
    expected = dict(
        schema_version=1,
        ablation_id=ABLATION_ID,
        status="FROZEN_BEFORE_ABLATION",
        root=str(root),
        decision_rule=RULE,
        variants=VARIANTS,
        optimizer_steps=0,
        teacher_forward_calls=0,
        monitor_calls=0,
        selection_uses_only="value_parseable",
        tasks=task_matrix(),
    )
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise PermissionError("Prospective ablation contract differs")
    previous = Path(manifest["previous_root"]).resolve(strict=True)
    if root == previous or root.is_relative_to(previous) or previous.is_relative_to(root):
        raise PermissionError("Ablation predecessor is not isolated")
    if set(manifest["previous_files"]) != {
        "STOP",
        "RUN_IDENTITY.json",
        "deployment/SOURCE_MANIFEST.json",
        manifest["stop_decision"],
    }:
        raise PermissionError("Stopped predecessor identities are missing")
    if [row["path"] for row in manifest["bound_files"]] != _bound_names():
        raise PermissionError("Required ablation input/source bindings are missing")
    if manifest["instruction"].get("path") != "authorization/REQUEST.md" or (
        manifest["instruction"].get("sha256") != REQUEST_SHA256
        or identity.get("bound_files", {}).get("authorization/REQUEST.md") != REQUEST_SHA256
        or len(manifest["previous_files"]) != 4
    ):
        raise PermissionError("Adopted instruction or distinct stopped decision is not bound")
    for name, record in manifest["previous_files"].items():
        if record.get("path") != name:
            raise PermissionError("Stopped predecessor artifact path differs")
        _verify_record(previous, record)
    for record in [*manifest["bound_files"], manifest["instruction"]]:
        _verify_record(root, record)
    tasks, slots, requests, candidates = _registration(root)
    if (
        manifest["slots"] != slots
        or manifest["requests"] != requests
        or manifest["candidates"] != candidates
    ):
        raise PermissionError("Registered sampling/input/candidate identity changed")
    if manifest["task_sidecars"] != {qid: digest(tasks[qid]) for qid in sorted(requests["P1"])}:
        raise PermissionError("Ablation task sidecar scope differs")
    return manifest


def runtime_revision_for_task(root, registered_task):
    manifest = verify_manifest(root)
    if registered_task not in manifest["tasks"]:
        raise PermissionError("Inference override is not a registered ablation task")
    return manifest["tasks"][registered_task]["protocol_revision"]


def _allocation(task, *, slurm=None):
    from .orchestration import Slurm

    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise PermissionError("Ablation requires an actual Slurm allocation")
    fields = (slurm or Slurm()).show(job)
    result = dict(job_id=job, user=getpass.getuser(), fields=fields)
    _validate_allocation(task, result)
    return result


def _validate_allocation(task, allocation):
    from .orchestration import TEACHER_QOS, physical_gpu_count

    job, fields = allocation["job_id"], allocation["fields"]
    expected = dict(
        JobId=job,
        JobName=task["job_name"],
        Account="rose",
        Partition="cluster02",
        QOS=TEACHER_QOS,
        JobState="RUNNING",
        Requeue="0",
    )
    if any(fields.get(k) != v for k, v in expected.items()) or (
        not str(job).isdigit()
        or not allocation["user"]
        or fields.get("UserId", "").split("(")[0] != allocation["user"]
        or physical_gpu_count(fields.get("ReqTRES", "")) != 1
        or physical_gpu_count(fields.get("AllocTRES", "")) != 1
        or not re.search(r"(?:^|,)gres/gpu:pro6000=1(?:,|$)", fields.get("ReqTRES", ""))
        or not task["job_name"].endswith("-LH")
    ):
        raise PermissionError("Actual ablation allocation differs from the single GPU task")


def _validate(row, slot, request, task, model, adapter, manifest_sha, index, native_context):
    from mm_core.vl_runtime import hash_json

    from .runtime import decoded_protocol

    validate_record(
        row,
        slot,
        input_hash=digest(request),
        model_identity=model,
        adapter_identity=adapter,
        source_sha256=manifest_sha,
        index=index,
        count=4,
    )
    if (
        row.get("model_input") != request
        or row.get("prompt") != request["prompt"]
        or (
            row.get("full_text") != row.get("raw_text")
            or row.get("task_sidecar_sha256") != digest(task)
            or row.get("runtime_protocol_revision") != slot["protocol_revision"]
            or row.get("runtime_assistant_prefill") != row.get("assistant_prefill")
            or row.get("finish_reason") not in ("balanced", "eos", "length")
        )
    ):
        raise PermissionError("Ablation prompt/native revision/full raw/sidecar differs")
    if row.get("prompt_tensor_hash") != native_context["prompt_tensor_hash"] or (
        row.get("input_routing", {}).get("assistant_prefill_token_ids")
        != native_context["assistant_prefill_token_ids"]
    ):
        raise PermissionError("Generated input differs from the actual accepted native CPU context")
    decoded = decoded_protocol(
        row["generated_text"], "structured", revision=slot["protocol_revision"]
    )
    backend, tokens = row.get("backend_sequence_tokens"), row["tokens"]
    sampling = row.get("sampling_parameters", {})
    ends = row.get("token_prefix_character_ends")
    if any(row.get(key) != val for key, val in decoded.items()) or (
        not isinstance(backend, list)
        or backend[: len(tokens)] != tokens
        or any(type(token) is not int or token < 0 for token in backend)
        or row.get("backend_padding_count") != len(backend) - len(tokens)
        or any(sampling.get(key) != val for key, val in slot["generation"].items())
        or row.get("sampling_hash") != hash_json(sampling)
        or row.get("max_new_tokens") != 1024
        or not isinstance(ends, list)
        or len(ends) != len(tokens)
        or any(type(end) is not int or end < 0 for end in ends)
        or ends[-1] != len(row["raw_text"])
    ):
        raise PermissionError("Native raw token/stop/decoding/generation trajectory differs")
    balanced = row["finish_reason"] == "balanced"
    if row.get("boundary_token_count") != (len(tokens) if balanced else None) or row.get(
        "balanced_token_count"
    ) != (len(tokens) if balanced else None):
        raise PermissionError("Native structured stop boundary differs")


def _group(root, manifest, group, runtime, tasks, directory):
    first = group["slots"][0]
    request = manifest["requests"][first["ablation_variant"]][first["qid"]]
    model, adapter = runtime.stable_model_identity(), runtime.current_adapter_identity()
    source = file_hash(root / "ABLATION_MANIFEST.json")
    path = directory / "groups" / (group["group_id"] + ".json")
    pending = path.with_suffix(".pending.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_json(
        path.with_suffix(".REQUEST.json"),
        dict(
            group=group,
            model_input=request,
            input_hash=digest(request),
            model_identity=model,
            adapter_identity=adapter,
            source_sha256=source,
            task_sidecar_sha256=digest(tasks[first["qid"]]),
        ),
    )
    if path.exists():
        value = read_json(path)
        rows = value["records"]
        if (
            value.get("records_sha256") != digest(rows)
            or value.get("group_id") != group["group_id"]
        ):
            raise PermissionError("Committed ablation group checksum differs")
    elif pending.exists():
        rows = [json.loads(line) for line in pending.read_text().splitlines()]
        if len(rows) != 4:
            raise RuntimeError("Partial ablation preserved; no automatic resampling authorization")
    else:
        rows = []

        def persist(completion):
            index = len(rows)
            if index >= 4:
                raise RuntimeError("Ablation runtime generated excess records")
            row = dict(completion)
            for key in (
                "group_seed",
                "group_row_index",
                "group_size",
                "protocol",
                "seed",
                "protocol_revision",
                "assistant_prefill",
            ):
                row["runtime_" + key] = completion.get(key)
            row.update(group["slots"][index])
            row.update(
                input_hash=digest(request),
                source_sha256=source,
                group_id=group["group_id"],
                original_group_id=group["original_group_id"],
                group_row_index=index,
                group_size=4,
                task_sidecar_sha256=digest(tasks[first["qid"]]),
            )
            with pending.open("a") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)

        runtime.generate_group(
            request,
            root / "data",
            [slot["sampling_seed"] for slot in group["slots"]],
            protocol="structured",
            generation=dict(first["generation"], num_return_sequences=4),
            on_completion=persist,
        )
    if len(rows) != 4:
        raise RuntimeError("Ablation group is incomplete; denominator cannot change")
    for index, (row, slot) in enumerate(zip(rows, group["slots"], strict=True)):
        _validate(
            row,
            slot,
            request,
            tasks[first["qid"]],
            model,
            adapter,
            source,
            index,
            manifest["candidates"][first["ablation_variant"]]["native_contexts"][first["qid"]],
        )
    if len({row["prompt_tensor_hash"] for row in rows}) != 1:
        raise PermissionError("Ablation same-question prompt tensors differ")
    durable_json(path, dict(group_id=group["group_id"], records=rows, records_sha256=digest(rows)))
    if pending.exists() and [json.loads(line) for line in pending.read_text().splitlines()] != rows:
        raise PermissionError("Ablation pending and committed raw bytes disagree")
    return _record(root, path)


def _raw_artifacts(root, directory, groups):
    expected = [
        directory / "groups" / (group["group_id"] + suffix)
        for group in groups
        for suffix in (".json", ".REQUEST.json", ".pending.jsonl")
    ]
    actual = list((directory / "groups").iterdir())
    if any(not path.is_file() or path.is_symlink() for path in actual) or (
        {path.name for path in actual} != {path.name for path in expected}
    ):
        raise PermissionError("Unregistered, missing or extra ablation raw artifact")
    return [
        _record(root, path)
        for path in expected
        if path.name.endswith((".REQUEST.json", ".pending.jsonl"))
    ]


def execute_parent_variant(root, block, variant, *, slurm=None):
    from .protocol import record_incident

    try:
        return _execute_parent_variant(root, block, variant, slurm=slurm)
    except BaseException as error:
        try:
            record_incident(
                root,
                "PROTOCOL_ABLATION",
                error,
                extra=dict(block=block, variant=variant, automatic_retry=False),
            )
        except BaseException as incident_error:
            if hasattr(error, "add_note"):
                error.add_note("Ablation incident persistence also failed: " + repr(incident_error))
        raise


def _execute_parent_variant(root, block, variant, *, slurm=None):
    from .runner import dataset_and_store, load_parent_runtime

    root = Path(root).resolve(strict=True)
    manifest = verify_manifest(root)
    registered_task = task_id(block, variant)
    task = manifest["tasks"][registered_task]
    allocation = _allocation(task, slurm=slurm)
    directory = root / "ablation/raw" / registered_task
    with shard_lease(directory):
        if (directory / "COMPLETE.json").exists():
            raise PermissionError("Existing completion requires receipt-only verification")
        durable_json(directory / "ALLOCATION.json", allocation)
        dataset, _ = dataset_and_store(root)
        runtime = load_parent_runtime(
            root,
            task["parent_id"],
            registered_task,
            protocol_revision=task["protocol_revision"],
            training=False,
        )
        if runtime.identity.get("protocol_revision") != task["protocol_revision"]:
            raise PermissionError("Loaded runtime protocol revision differs")
        before = runtime.current_adapter_identity()
        _parent_identity(root, task, runtime.stable_model_identity(), before)
        groups = [
            g
            for g in groups_for(manifest["slots"])
            if g["slots"][0]["block"] == block and g["slots"][0]["ablation_variant"] == variant
        ]
        records = [
            _group(root, manifest, group, runtime, dataset.tasks, directory) for group in groups
        ]
        if runtime.current_adapter_identity() != before:
            raise PermissionError("Zero-training ablation changed parent parameters")
        raw_artifacts = _raw_artifacts(root, directory, groups)
        receipt = dict(
            status="COMPLETE",
            task_id=registered_task,
            job_id=allocation["job_id"],
            job_name=task["job_name"],
            user=allocation["user"],
            records=256,
            ablation_manifest_sha256=file_hash(root / "ABLATION_MANIFEST.json"),
            model_identity=runtime.stable_model_identity(),
            adapter_identity=before,
            raw_groups=records,
            raw_artifacts=raw_artifacts,
            allocation_sha256=file_hash(directory / "ALLOCATION.json"),
            optimizer_steps=0,
            teacher_forward_calls=0,
            monitor_calls=0,
        )
        durable_json(directory / "COMPLETE.json", receipt)
        return receipt


def select_counts(counts):
    if set(counts) != set(task_matrix()) or any(
        type(count) is not int or not 0 <= count <= 256 for count in counts.values()
    ):
        raise ValueError("Exactly nine complete registered parseability counts are required")
    if any(counts[task_id(block, "P1")] >= 231 for block in (0, 2)):
        return dict(status="STACK_REPRODUCTION_FAILURE", selected_revision=None)
    for variant in ("P2", "P3"):
        if all(counts[task_id(block, variant)] >= 231 for block in range(3)):
            return dict(status="PROTOCOL_SELECTED", selected_revision=VARIANTS[variant])
    return dict(status="STOP_PROTOCOL_FAILURE", selected_revision=None)


def _selection(root, *, slurm=None, live=False):
    from .orchestration import TEACHER_QOS, Slurm
    from .runner import dataset_and_store

    root = Path(root).resolve(strict=True)
    manifest = verify_manifest(root)
    dataset, _ = dataset_and_store(root)
    if live:
        slurm = slurm or Slurm()
    groups = groups_for(manifest["slots"])
    if {path.name for path in (root / "ablation/raw").iterdir()} != set(manifest["tasks"]):
        raise PermissionError("All and only the nine registered ablation tasks must finish")
    counts, descriptions, receipts, observed = {}, {}, [], set()
    for registered_task, task in manifest["tasks"].items():
        directory = root / "ablation/raw" / registered_task
        receipt = read_json(directory / "COMPLETE.json")
        allocation = read_json(directory / "ALLOCATION.json")
        _validate_allocation(task, allocation)
        expected = dict(
            status="COMPLETE",
            task_id=registered_task,
            job_id=allocation["job_id"],
            job_name=task["job_name"],
            user=allocation["user"],
            records=256,
            ablation_manifest_sha256=file_hash(root / "ABLATION_MANIFEST.json"),
            allocation_sha256=file_hash(directory / "ALLOCATION.json"),
            optimizer_steps=0,
            teacher_forward_calls=0,
            monitor_calls=0,
        )
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise PermissionError("Ablation completion identity differs")
        _parent_identity(root, task, receipt["model_identity"], receipt["adapter_identity"])
        terminal_path = directory / "TERMINAL.json"
        terminal = slurm.terminal(receipt["job_id"]) if live else read_json(terminal_path)
        wanted = dict(
            job_id=receipt["job_id"],
            state="COMPLETED",
            exit_code="0:0",
            user=receipt["user"],
            account="rose",
            qos=TEACHER_QOS,
            name=task["job_name"],
        )
        if terminal is None or any(terminal.get(k) != v for k, v in wanted.items()):
            raise PermissionError("Every ablation task requires successful terminal accounting")
        if live:
            durable_json(terminal_path, terminal)
        chosen = [
            g
            for g in groups
            if g["slots"][0]["block"] == task["block"]
            and g["slots"][0]["ablation_variant"] == task["variant"]
        ]
        raw_paths = [
            _record(root, directory / "groups" / (g["group_id"] + ".json")) for g in chosen
        ]
        if receipt["raw_groups"] != raw_paths:
            raise PermissionError("Ablation raw group inventory differs")
        raw_artifacts = _raw_artifacts(root, directory, chosen)
        if receipt.get("raw_artifacts") != raw_artifacts:
            raise PermissionError("Ablation request/pending originals differ")
        metric, keys, token_total, work = Counter(), Counter(), 0, 0
        for group, raw in zip(chosen, raw_paths, strict=True):
            value = read_json(_verify_record(root, raw))
            rows = value["records"]
            if (
                value.get("group_id") != group["group_id"]
                or value.get("records_sha256") != digest(rows)
                or len(rows) != 4
            ):
                raise PermissionError("Ablation raw group identity/coverage differs")
            path = _contained(root, raw["path"])
            pending = [
                json.loads(line)
                for line in path.with_suffix(".pending.jsonl").read_text().splitlines()
            ]
            first = group["slots"][0]
            request = manifest["requests"][task["variant"]][first["qid"]]
            wanted_request = dict(
                group=group,
                model_input=request,
                input_hash=digest(request),
                model_identity=receipt["model_identity"],
                adapter_identity=receipt["adapter_identity"],
                source_sha256=expected["ablation_manifest_sha256"],
                task_sidecar_sha256=digest(dataset.tasks[first["qid"]]),
            )
            if pending != rows or read_json(path.with_suffix(".REQUEST.json")) != wanted_request:
                raise PermissionError("Ablation original request/stream and committed group differ")
            for index, (row, slot) in enumerate(zip(rows, group["slots"], strict=True)):
                if row["slot_id"] in observed:
                    raise PermissionError("Duplicate ablation record")
                observed.add(row["slot_id"])
                request = manifest["requests"][task["variant"]][slot["qid"]]
                sidecar = dataset.tasks[slot["qid"]]
                _validate(
                    row,
                    slot,
                    request,
                    sidecar,
                    receipt["model_identity"],
                    receipt["adapter_identity"],
                    expected["ablation_manifest_sha256"],
                    index,
                    manifest["candidates"][task["variant"]]["native_contexts"][slot["qid"]],
                )
                score = contract.score(
                    row["raw_text"],
                    sidecar["world"],
                    sidecar["query"],
                    finish_reason=row["finish_reason"],
                    protocol="structured",
                )
                metric.update(
                    {
                        key: int(score[key])
                        for key in (
                            "value_parseable",
                            "value_correct",
                            "certificate",
                            "presentation",
                            "finish",
                        )
                    }
                )
                match = re.match(r'^\s*\{?\s*"([^"]+)"', row["raw_text"])
                keys[match[1] if match else "UNPARSEABLE_FIRST_KEY"] += 1
                token_total += len(row["tokens"])
                work += bool(re.search(r'"work"\s*:', row["raw_text"]))
        counts[registered_task] = metric["value_parseable"]
        descriptions[registered_task] = dict(
            records=256,
            metrics=dict(metric),
            first_keys=dict(keys),
            series_first=keys["series"],
            work_field_observed=work,
            completion_tokens=token_total,
            mean_tokens=token_total / 256,
            descriptive_only=True,
        )
        receipts.append(
            dict(
                task_id=registered_task,
                complete=_record(root, directory / "COMPLETE.json"),
                raw_groups=raw_paths,
                raw_artifacts=raw_artifacts,
                allocation=_record(root, directory / "ALLOCATION.json"),
                terminal=_record(root, terminal_path),
            )
        )
    if observed != {slot["slot_id"] for slot in manifest["slots"]} or len(observed) != 2304:
        raise PermissionError("Full unique 2,304-output coverage is required before selection")
    return dict(
        schema_version=1,
        ablation_id=ABLATION_ID,
        **select_counts(counts),
        ablation_root=str(root),
        manifest_sha256=file_hash(root / "ABLATION_MANIFEST.json"),
        ablation_manifest=_record(root, root / "ABLATION_MANIFEST.json"),
        raw_binding=digest(receipts),
        decision_rule=RULE,
        counts=counts,
        denominators=dict.fromkeys(counts, 256),
        candidate_identities=manifest["candidates"],
        raw_evidence=receipts,
        descriptive_statistics=descriptions,
        outputs=2304,
        decision_uses_only="value_parseable",
        scientific_execution_authorized=False,
    )


def select_protocol(root, *, slurm=None):
    root = Path(root).resolve(strict=True)
    with shard_lease(root / "ablation/selection"):
        if (root / "PROTOCOL_SELECTION.json").exists():
            return verify_selection(root)
        result = _selection(root, slurm=slurm, live=True)
        durable_json(root / "PROTOCOL_SELECTION.json", result)
        return result


def verify_selection(root):
    """Read-only recomputation against stored, previously authenticated terminal snapshots."""
    value = read_json(Path(root) / "PROTOCOL_SELECTION.json")
    expected = _selection(root, live=False)
    if value != expected:
        raise PermissionError("Protocol selection differs from its complete immutable evidence")
    return value
