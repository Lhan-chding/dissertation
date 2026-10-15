"""CPU-only recovery tests use real checkpoints and original-record hash chains."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_core.training import state_hash, trainable_state
from sr_f3_ce import endpoint_runner, recovery, runner
from sr_f3_ce.checkpoint import checkpoint_state, commit_checkpoint
from sr_f3_ce.evaluation import _slots, group_slots
from sr_f3_ce.orchestration import TEACHER_QOS, artifact_record, validate_resume_verification
from sr_f3_ce.protocol import EXPERIMENT_ID, PARENTS, digest, file_hash, read_json, stable_seed
from sr_f3_ce.training import fresh_optimizer

torch = pytest.importorskip("torch")
KEY = "job-0123456789abcdef"
RUN = "SRF3CE_RL_b0"
TASK = "train--" + RUN


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return path


def pending(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def previous(root, task_id=TASK, *, terminal="TIMEOUT"):
    directory = root / "orchestration/jobs" / KEY
    intent = dict(
        key=KEY,
        kind=task_id.split("--")[0],
        task_ids=[task_id],
        gpu_count=1,
        run_identity_sha256=file_hash(root / "RUN_IDENTITY.json"),
        user="alice",
        qos=TEACHER_QOS,
        job_name="srf3ce-test-LH",
    )
    write(directory / "INTENT.json", intent)
    state = dict(
        key=KEY, status="FAILED", job_id="123", intent_sha256=file_hash(directory / "INTENT.json")
    )
    write(directory / "STATE.json", state)
    write(
        directory / "TERMINAL.json",
        dict(
            state=terminal,
            job_id="123",
            user="alice",
            account="rose",
            qos=TEACHER_QOS,
            name="srf3ce-test-LH",
            exit_code="0:15",
        ),
    )
    return intent, state


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path
    write(root / "RUN_IDENTITY.json", dict(experiment_id=EXPERIMENT_ID))
    write(root / "EXECUTION_FREEZE.json", dict(status="FROZEN_BEFORE_SCIENCE"))
    config = dict(generation=dict(temperature=1, max_new_tokens=1024))
    write(root / "plan/config/SR_F3_CE.json", config)
    write(root / "native/SUPPORT_MASK_RECEIPT.json", dict(status="PASS"))
    runs = [
        dict(
            run_id=f"SRF3CE_{arm}_b{block}",
            arm=arm,
            block=block,
            parent_id=PARENTS[block],
            future_seed=73101 + block,
            updates=64,
        )
        for block in range(3)
        for arm in ("RL", "GOLD", "PREFIX", "EXEC")
    ]
    run = runs[0]
    schedule = [
        dict(step=i, rl_qids=[f"q-{i}-{j}" for j in range(16)], support_qids=[])
        for i in range(1, 65)
    ]
    tasks = {qid: dict(qid=qid) for step in schedule for qid in step["rl_qids"]}
    data = root / "data"
    data.mkdir()
    store = SimpleNamespace(
        image_root=data,
        model_input=lambda qid, protocol="structured": dict(
            prompt=qid, image_file=qid + ".png", image_sha256=digest(qid)
        ),
    )
    dataset = SimpleNamespace(runs=runs, schedules={0: schedule}, tasks=tasks)
    model = torch.nn.Module()
    model.register_parameter("lora_a", torch.nn.Parameter(torch.tensor([0.1, -0.2])))
    runtime = SimpleNamespace(model=model)
    optimizer, scheduler = fresh_optimizer(model)
    reference = trainable_state(model)
    parent = dict(
        parent_id=PARENTS[0], policy_hash=state_hash(reference), adapter_path="parents/b0"
    )
    (root / parent["adapter_path"]).mkdir(parents=True)
    write(root / "parents/PARENT_AUDIT.json", dict(status="PASS", parents=[parent]))
    write(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json", dict(model_weights_hash=digest("base")))
    model_identity = dict(
        parent_id=PARENTS[0], base_model_weights_hash=digest("base"), model_id="base"
    )
    adapter = dict(
        model_id="base",
        step=None,
        adapter_path=str(root / parent["adapter_path"]),
        base_model_weights_hash=digest("base"),
        adapter_parameter_hash=parent["policy_hash"],
        published_adapter_hash=None,
    )
    rows = [dict(model_id=PARENTS[0], model_identity=model_identity, adapter_identity=adapter)]
    raw = write(
        root / "engine/b0/groups/original.json", dict(records=rows, records_sha256=digest(rows))
    )
    write(
        root / "tasks/engine--b0/COMPLETE.json",
        dict(artifacts=[artifact_record(root, str(raw.relative_to(root)))]),
    )
    identity = dict(
        run,
        stage="SCIENCE",
        experiment_id=EXPERIMENT_ID,
        run_identity_sha256=file_hash(root / "RUN_IDENTITY.json"),
        reference_hash=parent["policy_hash"],
    )
    sources = dict(
        run_identity=file_hash(root / "RUN_IDENTITY.json"),
        masks=file_hash(root / "native/SUPPORT_MASK_RECEIPT.json"),
        config=file_hash(root / "plan/config/SR_F3_CE.json"),
    )
    kwargs = dict(
        run_identity=identity,
        reference=reference,
        stream_hash=digest(schedule),
        sampling_hash=digest(config["generation"]),
        source_hashes=sources,
        microbatch_size=4,
        step_statistics={},
    )
    state = checkpoint_state(
        runtime,
        optimizer,
        scheduler,
        step=0,
        used_task_qids=[],
        used_support_qids=[],
        committed_records_hash=digest([]),
        **kwargs,
    )
    directory = root / "runs" / RUN
    commit_checkpoint(directory / "checkpoints", state)
    monkeypatch.setattr(recovery, "verify_run", lambda *args, **kwargs: ({}, config))
    monkeypatch.setattr(
        recovery, "verify_freeze", lambda root: file_hash(root / "EXECUTION_FREEZE.json")
    )
    monkeypatch.setattr(recovery, "verify_task", lambda *args: None)
    monkeypatch.setattr(runner, "dataset_and_store", lambda root: (dataset, store))
    monkeypatch.setattr(
        runner, "load_runtime", lambda *args, **kwargs: pytest.fail("GPU runtime loaded")
    )
    previous(root)
    return SimpleNamespace(
        root=root,
        run=run,
        runs=runs,
        schedule=schedule,
        tasks=tasks,
        dataset=dataset,
        store=store,
        identity=identity,
        model=model_identity,
        adapter=adapter,
        kwargs=kwargs,
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        reference=reference,
        directory=directory,
        state=state,
        config=config,
        parent=parent,
    )


def group(context, *, step=1, slot=0, count=8, complete=False):
    c = context
    qid = c.schedule[step - 1]["rl_qids"][slot]
    source = c.store.model_input(qid)
    seeds = [stable_seed("train", c.run["future_seed"], step, qid, i) for i in range(8)]
    request = dict(
        run_identity_sha256=digest(c.identity),
        step=step,
        slot=slot,
        qid=qid,
        seeds=seeds,
        policy_hash=state_hash(trainable_state(c.runtime.model)),
        input_hash=digest(source),
        task_hash=digest(c.tasks[qid]),
    )
    rows = [
        dict(
            model_input=source,
            model_identity=c.model,
            adapter_identity=dict(c.adapter, adapter_parameter_hash=request["policy_hash"]),
            seed=seeds[i],
            group_seed=seeds[0],
            group_row_index=i,
            group_size=8,
            protocol="structured",
            tokens=[1, 2],
            raw_tokens=[1, 2],
            sampler_logprobs=[-0.1, -0.2],
            completion_token_count=2,
            prompt_tensor_hash="actual-tensor",
            generation_status="COMPLETE",
            technical_validation_errors=[],
            finish_reason="balanced",
            truncated=False,
            assistant_prefill="{",
            generated_text='"answer":0}',
            full_text='{"answer":0}',
            raw_text='{"answer":0}',
            root=str(c.store.image_root),
        )
        for i in range(count)
    ]
    path = c.directory / "rollouts" / f"{step:02d}-{slot:02d}.json"
    write(path.with_suffix(".request.json"), request)
    if complete:
        saved = [dict(row, qid=qid, score={"J": 0}, reward=0) for row in rows]
        write(path, dict(identity=request, records=saved, records_sha256=digest(saved)))
    else:
        pending(path.with_suffix(".pending.jsonl"), rows)
    return path, request, rows


def committed_eight(c):
    chain, consumed = digest([]), []
    for step in range(1, 9):
        hashes = [file_hash(group(c, step=step, slot=slot, complete=True)[0]) for slot in range(16)]
        c.optimizer.zero_grad()
        c.runtime.model.lora_a.square().sum().backward()
        c.optimizer.step()
        c.scheduler.step()
        consumed.extend(c.schedule[step - 1]["rl_qids"])
        chain = digest([chain, step, hashes, []])
    state = checkpoint_state(
        c.runtime,
        c.optimizer,
        c.scheduler,
        step=8,
        used_task_qids=consumed,
        used_support_qids=[],
        committed_records_hash=chain,
        **c.kwargs,
    )
    return commit_checkpoint(c.directory / "checkpoints", state)


def test_checkpoint_resume_is_cpu_only_idempotent_and_matches_scheduler_schema(
    context, monkeypatch
):
    c = context
    result = recovery.verify_resume(c.root, TASK, KEY)
    assert result == recovery.verify_resume(c.root, TASK, KEY)
    assert result["status"] == "PASS" and result["resume_kind"] == "checkpoint"
    assert all(Path(row["path"]).name != "LATEST.json" for row in result["artifacts"])
    monkeypatch.setattr("sr_f3_ce.orchestration.verify_freeze", recovery.verify_freeze)
    intent, state = previous(c.root)
    assert validate_resume_verification(c.root, result, intent, state) == result


@pytest.mark.parametrize("terminal", ["FAILED", "PREEMPTED", "CANCELLED", "COMPLETED"])
def test_only_verified_science_timeout_is_recoverable(context, terminal):
    previous(context.root, terminal=terminal)
    with pytest.raises(PermissionError, match="TIMEOUT"):
        recovery.verify_resume(context.root, TASK, KEY)


def test_engine_is_never_implicitly_given_an_extra_budget(context):
    with pytest.raises(PermissionError, match="physical-budget"):
        recovery.verify_resume(context.root, "engine--resume", KEY)


def test_partial_authorization_binds_original_snapshot_without_mutable_pending(context):
    c = context
    path, request, rows = group(c, count=3)
    source = path.with_suffix(".pending.jsonl")
    original = source.read_bytes()
    result = recovery.verify_resume(c.root, TASK, KEY)
    auth = read_json(c.root / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json")
    assert auth["authorizations"] == [
        dict(
            task_id=TASK,
            model_id=RUN,
            group_id=digest(request),
            pending_sha256=file_hash(source),
            status="AUTHORIZED",
        )
    ]
    assert str(source.relative_to(c.root)) not in {row["path"] for row in result["artifacts"]}
    snapshots = list((c.root / "recovery" / KEY / "partial").iterdir())
    assert len(snapshots) == 1 and snapshots[0].read_bytes() == original
    pending(source, rows + rows[:1])
    assert snapshots[0].read_bytes() == original
    assert all(artifact_record(c.root, row["path"]) == row for row in result["artifacts"])


def test_full_pending_group_is_snapshotted_without_resampling_authorization(context):
    group(context)
    result = recovery.verify_resume(context.root, TASK, KEY)
    assert not (context.root / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json").exists()
    assert any("/partial/" in row["path"] for row in result["artifacts"])


@pytest.mark.parametrize(
    "field", ["seed", "tokens", "model_input", "technical_validation_errors", "adapter_identity"]
)
def test_partial_trajectory_corruption_blocks_authorization(context, field):
    path, _, rows = group(context, count=2)
    rows[0][field] = {"tampered": True} if isinstance(rows[0][field], dict) else None
    pending(path.with_suffix(".pending.jsonl"), rows)
    with pytest.raises((PermissionError, ValueError, RuntimeError)):
        recovery.verify_resume(context.root, TASK, KEY)
    assert not (context.root / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json").exists()


@pytest.mark.parametrize("field", ["seeds", "policy_hash", "input_hash"])
def test_uncommitted_cache_request_must_match_checkpoint_stream(context, field):
    path, request, _ = group(context, complete=True)
    request[field] = [1] * 8 if field == "seeds" else "0" * 64
    write(path.with_suffix(".request.json"), request)
    with pytest.raises(PermissionError):
        recovery.verify_resume(context.root, TASK, KEY)


def test_committed_original_chain_is_recomputed_and_rejects_a_changed_old_observation(context):
    c = context
    marker = committed_eight(c)
    result = recovery.verify_resume(c.root, TASK, KEY)
    details = read_json(c.root / "recovery" / KEY / "VERIFICATION.json")["details"]
    assert details["checkpoint"] == marker and details["cache"]["complete_groups"] == 128
    assert any(row["sha256"] == marker["sha256"] for row in result["artifacts"])
    path = c.directory / "rollouts/01-00.json"
    value = read_json(path)
    value["records"][0]["tokens"] = [77]
    value["records_sha256"] = digest(value["records"])
    write(path, value)
    with pytest.raises(PermissionError, match="hash chain"):
        recovery.verify_resume(c.root, TASK, KEY)


def test_conflicting_latest_pointer_and_wrong_checkpoint_sources_block(context):
    c = context
    latest = read_json(c.directory / "checkpoints/LATEST.json")
    write(c.directory / "checkpoints/LATEST.json", dict(latest, sha256="0" * 64))
    with pytest.raises(PermissionError, match="Latest pointer"):
        recovery.verify_resume(c.root, TASK, KEY)
    write(c.directory / "checkpoints/LATEST.json", latest)
    write(c.root / "native/SUPPORT_MASK_RECEIPT.json", dict(status="CHANGED"))
    with pytest.raises(PermissionError, match="sources"):
        recovery.verify_resume(c.root, TASK, KEY)


def test_existing_authorization_is_immutable_and_cannot_silently_cover_another_group(context):
    c = context
    group(c, count=2)
    path = c.root / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json"
    existing = dict(
        experiment_id=EXPERIMENT_ID,
        run_identity_sha256=file_hash(c.root / "RUN_IDENTITY.json"),
        authorizations=[],
    )
    write(path, existing)
    with pytest.raises(PermissionError, match="immutable authorization"):
        recovery.verify_resume(c.root, TASK, KEY)
    assert read_json(path) == existing


def test_unregistered_cache_and_frozen_parent_impersonation_block(context):
    c = context
    unknown = write(c.directory / "rollouts/99-00.json", {})
    with pytest.raises(PermissionError, match="Unregistered"):
        recovery.verify_resume(c.root, TASK, KEY)
    unknown.unlink()
    raw = c.root / "engine/b0/groups/original.json"
    value = read_json(raw)
    value["records"][0]["adapter_identity"]["adapter_parameter_hash"] = "0" * 64
    value["records_sha256"] = digest(value["records"])
    write(raw, value)
    with pytest.raises(PermissionError, match="ENGINE original"):
        recovery.verify_resume(c.root, TASK, KEY)


def test_endpoint_cache_recovery_uses_authenticated_parent_and_never_loads_scorer(
    context, monkeypatch
):
    c = context
    model_id, task_id = PARENTS[0], "eval--" + PARENTS[0]
    previous(c.root, task_id)
    model = recovery.endpoint_models(c.runs)[0]
    destination = c.root / "sealed/endpoint" / model_id
    stable = dict(c.model, model_id=model_id, step=0)
    adapter = dict(c.adapter, model_id=model_id, step=0)
    provenance = dict(
        model=model,
        policy_source="verified parent adapter",
        policy_hash=c.parent["policy_hash"],
        model_identity=stable,
        adapter_identity=adapter,
    )
    write(destination / "MODEL_IDENTITY.json", provenance)
    qid = c.schedule[0]["rl_qids"][0]
    task = dict(c.tasks[qid], pool="TEST_ID", family="CROSS", root_id="root")
    slots = _slots(model, [task], "structured", 4)
    grouped = group_slots(slots)[0]
    _, _, raw = group(c, count=2)
    rows = []
    for index, original in enumerate(raw):
        slot = grouped["slots"][index]
        rows.append(
            dict(
                original,
                **slot,
                model_identity=stable,
                adapter_identity=adapter,
                model_input=c.store.model_input(qid),
                input_hash=digest(c.store.model_input(qid)),
                source_sha256=file_hash(c.root / "RUN_IDENTITY.json"),
                group_id=grouped["group_id"],
                group_row_index=index,
                group_size=4,
                runtime_group_seed=slot["group_seed"],
                runtime_group_row_index=index,
                runtime_group_size=4,
                runtime_protocol="structured",
                runtime_seed=slot["sampling_seed"],
            )
        )
    path = destination / "TEST_ID/groups" / (grouped["group_id"] + ".pending.jsonl")
    pending(path, rows)
    monkeypatch.setattr(recovery, "endpoint_slots", lambda *args, **kwargs: slots)
    observed = []

    def external(root, *, load_scorer):
        observed.append(load_scorer)
        return {}, {}, {}, lambda *args: pytest.fail("Scores accessed")

    monkeypatch.setattr(endpoint_runner, "_external", external)
    result = recovery.verify_resume(c.root, task_id, KEY)
    assert result["resume_kind"] == "generation_cache" and observed == [False]
    assert (
        read_json(c.root / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json")["authorizations"][0][
            "group_id"
        ]
        == grouped["group_id"]
    )
    changed = copy.deepcopy(provenance)
    changed["model_identity"]["base_model_weights_hash"] = "0" * 64
    write(destination / "MODEL_IDENTITY.json", changed)
    with pytest.raises(PermissionError, match="model provenance"):
        recovery.verify_resume(c.root, task_id, KEY)
