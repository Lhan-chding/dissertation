"""CPU verification for an explicit, identity-preserving science lease recovery.

No model is loaded, no answer is scored, and no failed ENGINE budget is extended.
Mutable pending files are preserved as immutable evidence before a replay can be
authorized. The GPU worker must still reproduce each cached policy and trajectory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mm_core.training import state_hash
from sr_f12.evaluation import durable_json

from .checkpoint import read_checkpoint
from .evaluation import (
    _write_immutable_bytes,
    endpoint_models,
    endpoint_slots,
    monitor_slots,
    validate_record,
    verify_panel_cache,
)
from .orchestration import (
    artifact_record,
    receipt_path,
    registered_task,
    verify_freeze,
    verify_task,
)
from .protocol import (
    EXPERIMENT_ID,
    contained,
    digest,
    file_hash,
    read_json,
    stable_seed,
    verify_run,
)


class Evidence:
    def __init__(self, root, key):
        self.root = root
        self.directory = root / "recovery" / key
        self.files = {}
        self.pending = []
        self.authorizations = []

    def bind(self, path):
        path = Path(path).resolve(strict=True)
        row = artifact_record(self.root, str(path.relative_to(self.root)))
        self.files[row["path"]] = row
        return row

    def snapshot(self, path, *, authorization=None):
        path = Path(path).resolve(strict=True)
        relative = str(path.relative_to(self.root))
        destination = self.directory / "partial" / (digest(relative)[:16] + "-" + path.name)
        _write_immutable_bytes(destination, path.read_bytes())
        row = self.bind(destination)
        if file_hash(path) != row["sha256"]:
            raise PermissionError("Pending output changed during recovery verification")
        self.pending.append(dict(original_path=relative, snapshot=row))
        if authorization is not None:
            if authorization["pending_sha256"] != row["sha256"]:
                raise PermissionError("Partial replay snapshot and authorization differ")
            self.authorizations.append(dict(authorization, status="AUTHORIZED"))
        return row

    def authorize(self):
        if not self.authorizations:
            return None
        entries = sorted(self.authorizations, key=lambda row: (row["task_id"], row["group_id"]))
        if len({(r["task_id"], r["group_id"]) for r in entries}) != len(entries):
            raise PermissionError("Duplicate partial replay authorization")
        entrypoint = self.root / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json"
        expected = dict(
            experiment_id=EXPERIMENT_ID,
            run_identity_sha256=file_hash(self.root / "RUN_IDENTITY.json"),
        )
        if entrypoint.exists():
            value, source = read_json(entrypoint), entrypoint
            if "authorization_file" in value:
                source = contained(self.root, value["authorization_file"])
                if not source.is_relative_to(self.root / "recovery") or (
                    file_hash(source) != value.get("sha256")
                ):
                    raise PermissionError("Partial authorization pointer differs")
                value = read_json(source)
            if any(value.get(k) != v for k, v in expected.items()) or any(
                value.get("authorizations", []).count(row) != 1 for row in entries
            ):
                raise PermissionError(
                    "Existing immutable authorization does not cover this recovery"
                )
        else:
            source = entrypoint
            durable_json(source, dict(expected, authorizations=entries))
        # A pointer may change for a later explicit decision; bind its immutable target only.
        return self.bind(source)


def _previous(root, task_id, key, evidence):
    if not re.fullmatch(r"job-[a-f0-9]{16}", key):
        raise PermissionError("Invalid recovery predecessor")
    task = registered_task(task_id)
    if task["kind"] not in {"train", "eval"}:
        raise PermissionError("ENGINE recovery needs a separate exact physical-budget audit")
    directory = root / "orchestration/jobs" / key
    intent, state, terminal = (
        read_json(directory / name) for name in ("INTENT.json", "STATE.json", "TERMINAL.json")
    )
    expected = dict(
        job_id=state.get("job_id"),
        state="TIMEOUT",
        user=intent.get("user"),
        account="rose",
        qos=intent.get("qos"),
        name=intent.get("job_name"),
    )
    if (
        intent.get("key") != key
        or state.get("key") != key
        or state.get("status") != "FAILED"
        or intent.get("task_ids") != [task_id]
        or intent.get("gpu_count") != 1
        or intent.get("kind") != task["kind"]
        or intent.get("run_identity_sha256") != file_hash(root / "RUN_IDENTITY.json")
        or state.get("intent_sha256") != file_hash(directory / "INTENT.json")
        or not str(state.get("job_id", "")).isdigit()
        or any(terminal.get(k) != v for k, v in expected.items())
    ):
        raise PermissionError("Recovery requires this exact single-GPU science TIMEOUT")
    if receipt_path(root, task_id).exists():
        raise PermissionError("Completed output needs receipt reconciliation, not replay")
    evidence.bind(directory / "INTENT.json")
    evidence.bind(directory / "TERMINAL.json")
    return task, state


def _parent(root, parent_id, block, evidence):
    """Obtain runtime identities from an ENGINE artifact already bound by the freeze."""
    audit = read_json(root / "parents/PARENT_AUDIT.json")
    parents = [p for p in audit["parents"] if p["parent_id"] == parent_id]
    if audit.get("status") != "PASS" or len(parents) != 1:
        raise PermissionError("Parent audit identity differs")
    parent = parents[0]
    task_id = f"engine--b{block}"
    verify_task(root, task_id)
    receipt = read_json(receipt_path(root, task_id))
    prefix = f"engine/b{block}/groups/"
    raw = next(
        (
            row
            for row in receipt["artifacts"]
            if row["path"].startswith(prefix) and row["path"].endswith(".json")
        ),
        None,
    )
    if raw is None:
        raise PermissionError("Frozen ENGINE has no bound original model identity")
    path = contained(root, raw["path"])
    if evidence.bind(path) != raw:
        raise PermissionError("Frozen ENGINE original bytes changed")
    value = read_json(path)
    rows = value["records"]
    if not rows or value.get("records_sha256") != digest(rows):
        raise PermissionError("Frozen ENGINE trajectory checksum differs")
    row = rows[0]
    model, adapter = row["model_identity"], row["adapter_identity"]
    prior = read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    if (
        row.get("model_id") != parent_id
        or model.get("parent_id") != parent_id
        or model.get("base_model_weights_hash") != prior["model_weights_hash"]
        or adapter.get("adapter_parameter_hash") != parent["policy_hash"]
        or Path(adapter["adapter_path"]).resolve() != contained(root, parent["adapter_path"])
    ):
        raise PermissionError("Frozen ENGINE parent/model provenance differs")
    evidence.bind(receipt_path(root, task_id))
    evidence.bind(root / "parents/PARENT_AUDIT.json")
    return parent, model, adapter


def _checkpoint(root, run, dataset, store, config, parent, evidence, *, step=None):
    from .runner import verify_committed_records

    directory = root / "runs" / run["run_id"]
    state, marker = read_checkpoint(directory / "checkpoints", step=step)
    expected_identity = dict(
        run,
        stage="SCIENCE",
        experiment_id=EXPERIMENT_ID,
        run_identity_sha256=file_hash(root / "RUN_IDENTITY.json"),
        reference_hash=parent["policy_hash"],
    )
    expected = dict(
        run_identity=expected_identity,
        reference_hash=parent["policy_hash"],
        input_stream_hash=digest(dataset.schedules[run["block"]]),
        sampling_hash=digest(config["generation"]),
        source_hashes=dict(
            run_identity=file_hash(root / "RUN_IDENTITY.json"),
            masks=file_hash(root / "native/SUPPORT_MASK_RECEIPT.json"),
            config=file_hash(root / "plan/config/SR_F3_CE.json"),
        ),
    )
    if any(state.get(k) != v for k, v in expected.items()) or (
        state["committed_logical_step"] % 8 != 0
    ):
        raise PermissionError(
            "Checkpoint frozen identity, sources, stream, or recovery boundary differs"
        )
    verify_committed_records(
        directory, state, dataset.schedules[run["block"]], dataset.tasks, store
    )
    evidence.bind(directory / "checkpoints" / f"commit-{marker['step']:02d}.json")
    evidence.bind(directory / "checkpoints" / marker["path"])
    return state, marker


def _training_rows(rows, request, model_input, model, adapter, source):
    if not isinstance(rows, list) or len(rows) > 8:
        raise PermissionError("Training cache has excess trajectories")
    expected_adapter = dict(adapter, adapter_parameter_hash=request["policy_hash"])
    for index, row in enumerate(rows):
        expected = dict(
            model_input=model_input,
            protocol="structured",
            seed=request["seeds"][index],
            group_seed=request["seeds"][0],
            group_row_index=index,
            group_size=8,
        )
        if any(row.get(k) != v for k, v in expected.items()) or (
            row.get("technical_validation_errors") != []
        ):
            raise PermissionError("Training cache input, seed, or group identity differs")
        # Translate native runtime fields to the shared technical trajectory validator;
        # none of the original input/model/token/termination fields are overwritten.
        slot = dict(
            protocol="structured",
            generation=dict(max_new_tokens=1024),
            group_seed=request["seeds"][0],
            sampling_seed=request["seeds"][index],
        )
        normalized = dict(
            row,
            generation=slot["generation"],
            sampling_seed=row["seed"],
            input_hash=digest(model_input),
            source_sha256=source,
            runtime_group_seed=row["group_seed"],
            runtime_group_row_index=row["group_row_index"],
            runtime_group_size=row["group_size"],
            runtime_protocol=row["protocol"],
            runtime_seed=row["seed"],
        )
        validate_record(
            normalized,
            slot,
            input_hash=digest(model_input),
            model_identity=model,
            adapter_identity=expected_adapter,
            source_sha256=source,
            index=index,
            count=8,
        )
        if row.get("full_text") != row.get("raw_text"):
            raise PermissionError("Training original full text differs")
    if len({row["prompt_tensor_hash"] for row in rows}) > 1:
        raise PermissionError("Training group prompt tensor identities differ")


def _training_cache(root, run, state, dataset, store, model, adapter, evidence):
    directory = root / "runs" / run["run_id"]
    schedule, committed = dataset.schedules[run["block"]], state["committed_logical_step"]
    expected = {
        f"{item['step']:02d}-{slot:02d}": (item, slot, qid)
        for item in schedule
        for slot, qid in enumerate(item["rl_qids"])
    }
    observed = {}
    for path in sorted((directory / "rollouts").glob("*")):
        match = re.fullmatch(r"(\d{2}-\d{2})(\.request\.json|\.pending\.jsonl|\.json)", path.name)
        if not match or match[1] not in expected or not path.is_file():
            raise PermissionError("Unregistered training rollout cache path")
        observed.setdefault(match[1], {})[match[2]] = path
    policies, complete, partial = {}, 0, 0
    for key, files in observed.items():
        item, slot, qid = expected[key]
        if ".request.json" not in files:
            raise PermissionError("Training output lacks a durable request")
        request = read_json(files[".request.json"])
        model_input = store.model_input(qid)
        required = dict(
            run_identity_sha256=digest(state["run_identity"]),
            step=item["step"],
            slot=slot,
            qid=qid,
            seeds=[
                stable_seed("train", run["future_seed"], item["step"], qid, i) for i in range(8)
            ],
            input_hash=digest(model_input),
            task_hash=digest(dataset.tasks[qid]),
        )
        if (
            set(request) != {*required, "policy_hash"}
            or any(request.get(k) != v for k, v in required.items())
            or not re.fullmatch(r"[a-f0-9]{64}", str(request.get("policy_hash", "")))
        ):
            raise PermissionError("Training request identity, seeds, or inputs changed")
        step = item["step"]
        if (
            step > committed + 8
            or policies.setdefault(step, request["policy_hash"]) != request["policy_hash"]
        ):
            raise PermissionError("Training cache exceeds the next checkpoint or mixes policies")
        if step == committed + 1 and request["policy_hash"] != state_hash(state["parameters"]):
            raise PermissionError("First resumed group differs from the committed policy")
        evidence.bind(files[".request.json"])
        saved = None
        if ".json" in files:
            value = read_json(files[".json"])
            saved = value["records"]
            if (
                value.get("identity") != request
                or len(saved) != 8
                or (value.get("records_sha256") != digest(saved))
                or any(row.get("qid") != qid for row in saved)
            ):
                raise PermissionError("Training cached original checksum/coverage differs")
            _training_rows(saved, request, model_input, model, adapter, evidence.source)
            evidence.bind(files[".json"])
            complete += 1
        if ".pending.jsonl" in files:
            path = files[".pending.jsonl"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            _training_rows(rows, request, model_input, model, adapter, evidence.source)
            if saved is not None and any(
                any(
                    row.get(k) != saved[i].get(k)
                    for k in row
                    if k not in {"root", "score", "reward"}
                )
                for i, row in enumerate(rows)
            ):
                raise PermissionError("Pending and cached original trajectories differ")
            authorization = None
            if saved is None and len(rows) < 8:
                if step <= committed:
                    raise PermissionError("Committed checkpoint contains a partial rollout")
                partial += 1
                authorization = dict(
                    task_id="train--" + run["run_id"],
                    model_id=run["run_id"],
                    group_id=digest(request),
                    pending_sha256=file_hash(path),
                )
            evidence.snapshot(path, authorization=authorization)
    allowed_support = (
        {
            f"{item['step']:02d}-{qid}.json": (item["step"], qid)
            for item in schedule
            for qid in item["support_qids"]
        }
        if run["arm"] != "RL"
        else {}
    )
    for path in sorted((directory / "support").glob("*")):
        if path.name not in allowed_support or not path.is_file():
            raise PermissionError("Unregistered support cache")
        step, qid = allowed_support[path.name]
        value = read_json(path)
        if step > committed + 8 or any(
            value.get(k) != v for k, v in dict(step=step, qid=qid, arm=run["arm"]).items()
        ):
            raise PermissionError("Support cache identity differs")
        evidence.bind(path)
    return dict(
        complete_groups=complete,
        partial_groups=partial,
        live_replay_policy_verification_required=True,
    )


def _panel(root, path, slots, store, tasks, model, adapter, task_id, evidence):
    result = verify_panel_cache(
        root,
        path,
        slots,
        store,
        tasks,
        model_identity=model,
        adapter_identity=adapter,
        source_sha256=evidence.source,
        task_id=task_id,
    )
    for item in result["complete_groups"]:
        source = contained(root, item["path"])
        if source.name.endswith(".pending.jsonl"):
            evidence.snapshot(source)
        else:
            evidence.bind(source)
    for item in result["partial_groups"]:
        authorization = {
            key: item[key] for key in ("task_id", "model_id", "group_id", "pending_sha256")
        }
        evidence.snapshot(contained(root, item["path"]), authorization=authorization)
    return result


def _train(root, task, dataset, store, config, parent, model, adapter, evidence):
    run = next(r for r in dataset.runs if r["run_id"] == task["model_id"])
    state, marker = _checkpoint(root, run, dataset, store, config, parent, evidence)
    cache = _training_cache(root, run, state, dataset, store, model, adapter, evidence)
    monitors = []
    for due in (16, 32, 48, 64):
        path = root / "monitor" / run["run_id"] / f"step-{due:02d}"
        if due > marker["step"]:
            if path.exists():
                raise PermissionError("Monitor exists beyond the committed checkpoint")
            continue
        previous, _ = _checkpoint(root, run, dataset, store, config, parent, evidence, step=due)
        policy = state_hash(previous["parameters"])
        panel = _panel(
            root,
            path,
            monitor_slots(dataset.tasks, run["run_id"], due),
            store,
            dataset.tasks,
            model,
            dict(adapter, adapter_parameter_hash=policy),
            task["task_id"],
            evidence,
        )
        if due < marker["step"] or (path / "COMPLETE.json").exists():
            complete, coverage = (
                read_json(path / "COMPLETE.json"),
                read_json(path / "COVERAGE.json"),
            )
            actual = {
                item["group_id"]: item["sha256"]
                for item in panel["complete_groups"]
                if item["path"].endswith(".json")
            }
            if (
                panel["partial_groups"]
                or panel["missing_groups"]
                or any(
                    value.get("records") != 512
                    or value.get("policy_hash") != policy
                    or {g["group_id"]: g["sha256"] for g in value["groups"]} != actual
                    or value.get("records_sha256") != digest(value["groups"])
                    for value in (complete, coverage)
                )
            ):
                raise PermissionError("Earlier fixed monitor is incomplete or changed")
            evidence.bind(path / "COMPLETE.json")
            evidence.bind(path / "COVERAGE.json")
        monitors.append(dict(step=due, cache=panel))
    return dict(
        checkpoint=marker,
        committed_records_hash=state["committed_records_hash"],
        cache=cache,
        monitors=monitors,
    )


def _eval(root, task, dataset, store, config, parent, model, adapter, evidence):
    from .endpoint_runner import _external

    registered = next(
        row for row in endpoint_models(dataset.runs) if row["model_id"] == task["model_id"]
    )
    destination = root / "sealed/endpoint" / task["model_id"]
    provenance = read_json(destination / "MODEL_IDENTITY.json")
    policy = parent["policy_hash"]
    expected = dict(model=registered, policy_source="verified parent adapter")
    if registered["arm"] != "PARENT":
        run = next(row for row in dataset.runs if row["run_id"] == registered["model_id"])
        state, marker = _checkpoint(root, run, dataset, store, config, parent, evidence, step=64)
        _training_cache(root, run, state, dataset, store, model, adapter, evidence)
        policy = state_hash(state["parameters"])
        exported = root / "runs" / run["run_id"] / "adapters/step-64"
        published = read_json(exported / "IDENTITY.json")
        if (
            published.get("step") != 64
            or published.get("policy_hash") != policy
            or published.get("run_identity_sha256") != digest(state["run_identity"])
        ):
            raise PermissionError("Endpoint published adapter differs from the final checkpoint")
        for relative, expected_hash in published["files"].items():
            path = contained(exported, relative)
            if file_hash(path) != expected_hash:
                raise PermissionError("Endpoint published adapter bytes changed")
            evidence.bind(path)
        evidence.bind(exported / "IDENTITY.json")
        model = dict(model, adapter_hash=file_hash(exported / "IDENTITY.json"))
        adapter = dict(
            adapter,
            adapter_path=str(exported),
            published_adapter_hash=file_hash(exported / "IDENTITY.json"),
        )
        expected.update(
            policy_source="verified committed step-64 parameters",
            checkpoint=marker,
            adapter_export=published,
        )
    model = dict(model, model_id=registered["model_id"], step=registered["step"])
    adapter = dict(
        adapter,
        model_id=registered["model_id"],
        step=registered["step"],
        adapter_parameter_hash=policy,
    )
    expected.update(policy_hash=policy, model_identity=model, adapter_identity=adapter)
    if provenance != expected:
        raise PermissionError(
            "Endpoint model provenance differs from its authenticated parent/checkpoint"
        )
    evidence.bind(destination / "MODEL_IDENTITY.json")
    external_tasks, external_stores, _, _ = _external(root, load_scorer=False)
    tasks = dict(dataset.tasks, **external_tasks)
    slots = endpoint_slots(dataset.tasks, dataset.runs, external_tasks, model_id=task["model_id"])
    pools = sorted({slot["pool"] for slot in slots})
    for path in destination.iterdir():
        if path.is_dir() and path.name not in pools:
            raise PermissionError("Unregistered endpoint cache panel")
    panels = [
        _panel(
            root,
            destination / pool,
            [s for s in slots if s["pool"] == pool],
            external_stores.get(pool, store),
            tasks,
            model,
            adapter,
            task["task_id"],
            evidence,
        )
        for pool in pools
    ]
    return dict(model_id=task["model_id"], policy_hash=policy, panels=panels, scores_accessed=False)


def verify_resume(root, task_id, previous_job_key):
    """Produce the scheduler callback schema only after immutable CPU verification."""
    from .runner import dataset_and_store

    root = Path(root).resolve(strict=True)
    _, config = verify_run(root, science=True)
    freeze = verify_freeze(root)
    evidence = Evidence(root, previous_job_key)
    evidence.source = file_hash(root / "RUN_IDENTITY.json")
    task, previous = _previous(root, task_id, previous_job_key, evidence)
    dataset, store = dataset_and_store(root)
    model = next(
        row for row in endpoint_models(dataset.runs) if row["model_id"] == task["model_id"]
    )
    parent, stable, adapter = _parent(root, model["parent_id"], model["block"], evidence)
    details = (_train if task["kind"] == "train" else _eval)(
        root, task, dataset, store, config, parent, stable, adapter, evidence
    )
    authorization = evidence.authorize()
    manifest = dict(
        experiment_id=EXPERIMENT_ID,
        task_id=task_id,
        previous_job_key=previous_job_key,
        previous_job_id=previous["job_id"],
        run_identity_sha256=evidence.source,
        execution_freeze_sha256=freeze,
        details=details,
        pending=evidence.pending,
        authorization=authorization,
        scores_used_for_recovery=False,
        same_seed_full_group_replay_only=True,
    )
    path = evidence.directory / "VERIFICATION.json"
    durable_json(path, manifest)
    evidence.bind(path)
    return dict(
        status="PASS",
        task_id=task_id,
        previous_job_key=previous_job_key,
        previous_job_id=previous["job_id"],
        run_identity_sha256=evidence.source,
        execution_freeze_sha256=freeze,
        resume_kind="checkpoint" if task["kind"] == "train" else "generation_cache",
        resume_state_sha256=digest(manifest),
        artifacts=sorted(evidence.files.values(), key=lambda row: row["path"]),
    )
