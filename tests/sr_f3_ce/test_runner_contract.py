"""CPU-only wiring checks against the actual frozen data; no model is constructed."""

import copy
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f3_ce import checkpoint, engine_recovery, evaluation, runner, training
from sr_f3_ce.contract import support_target
from sr_f3_ce.data import FAMILIES, load_frozen_dataset

PLAN = Path(__file__).resolve().parents[2] / "artifacts/sr_f3_ce_20261016/plan"


@pytest.fixture(scope="module")
def dataset():
    if not PLAN.is_dir():
        pytest.skip("Private frozen user package is not installed in this checkout")
    return load_frozen_dataset(PLAN)


class InputStoreSpy:
    def __init__(self, root, dataset):
        self.image_root = root
        self.dataset = dataset

    def model_input(self, qid):
        return {
            "prompt": self.dataset.inputs[qid]["prompt"],
            "image_file": f"images/{qid}.png",
            "image_sha256": "a" * 64,
        }


class SupportRuntimeSpy:
    """Exercises runner's actual calls and validates the native API target shape."""

    def __init__(self, revision="v1"):
        self.protocol_revision = revision
        self.calls = []
        self.model = object()

    def prepare_support(self, model_input, run_root, target, arm, *, serialization):
        assert set(model_input) == {"prompt", "image_file", "image_sha256"}
        assert isinstance(target, dict) and set(target) == {"text", "spans", "object"}
        assert isinstance(target["text"], str)
        assert isinstance(target["spans"], dict)
        assert isinstance(target["object"], dict)
        assert ("work" in target["spans"]) == (arm != "GOLD")
        self.calls.append((copy.deepcopy(target), arm, serialization))
        return {
            "tokens": [1, 2],
            "answer_mask": [0, 1],
            "work_mask": [1, 0] if arm == "EXEC" else [0, 0],
            "mask_audit": {"test_spy": True},
        }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.mark.parametrize("revision", ["v1", "v2", "c2"])
def test_parent_loader_threads_frozen_revision_and_constructs_no_optimizer(
    tmp_path, monkeypatch, revision
):
    """CPU constructor spies, not evidence of a CUDA run or a model generation."""
    import torch

    from mm_dev import runtime as inherited_runtime
    from sr_f3_ce import runtime as native_runtime
    from sr_f3_ce.protocol_revision import revision_config

    config = {"generation": {"prefill": "{"}}
    if revision != "v1":
        config = revision_config(config, revision)
    parent_id = runner.PARENTS[0]
    model = torch.nn.Linear(1, 1)
    policy_hash = runner.state_hash(runner.trainable_state(model))
    adapter = tmp_path / "parents/adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter.bin").write_bytes(b"CPU constructor fixture")
    write_json(
        adapter / "IDENTITY.json",
        {"files": {"adapter.bin": runner.file_hash(adapter / "adapter.bin")}},
    )
    write_json(adapter / "adapter_config.json", {})
    prior = dict(
        model_path="CPU_FIXTURE_NEVER_LOADED",
        model_weights_hash="base",
        processor_hash="processor",
        tokenizer_hash="tokenizer",
        chat_template_hash="template",
    )
    write_json(tmp_path / "prior/MODEL_ENVIRONMENT_IDENTITY.json", prior)
    write_json(
        tmp_path / "parents/PARENT_AUDIT.json",
        dict(
            status="PASS",
            parents=[
                dict(
                    parent_id=parent_id,
                    adapter_path="parents/adapter",
                    policy_hash=policy_hash,
                )
            ],
        ),
    )
    write_json(
        tmp_path / "native/PROCESSOR_RECEIPT.json",
        dict(
            native_processor_identity=dict(
                processor_hash="processor",
                tokenizer_hash="tokenizer",
                chat_template_hash="template",
                generation_config_expanded={"CPU_FIXTURE": True},
            )
        ),
    )
    calls = []

    class RuntimeSpy:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            self.protocol_revision = kwargs["protocol_revision"]
            self.model = model
            self.identity = dict(prior)

        def verify_identity(self, *args):
            return None

    monkeypatch.setattr(runner, "verify_run", lambda root: ({}, config))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner, "seed_all", lambda seed: None)
    monkeypatch.setattr(
        inherited_runtime, "configure_audited_backend", lambda: {"CPU_FIXTURE": True}
    )
    monkeypatch.setattr(inherited_runtime, "actual_cuda_identity", lambda: {"CPU_FIXTURE": True})
    monkeypatch.setattr(native_runtime, "SRF3CERuntime", RuntimeSpy)
    monkeypatch.setattr(
        native_runtime, "configure_parent_training", lambda *a, **kw: {"CPU_FIXTURE": True}
    )

    def forbidden_optimizer(*args):
        raise AssertionError("Parent-only loading must not construct Adam or scheduler")

    monkeypatch.setattr(training, "fresh_optimizer", forbidden_optimizer)
    result = runner.load_parent_runtime(tmp_path, parent_id, "engine--b0", training=False)
    assert result.protocol_revision == revision == calls[0]["protocol_revision"]
    receipt = json.loads(next((tmp_path / "runtime_identity").glob("*.json")).read_text())
    assert receipt["training_enabled"] is False
    assert receipt["optimizer_constructed"] is receipt["scheduler_constructed"] is False
    assert receipt["fresh_optimizer_state_entries"] is None
    with pytest.raises(PermissionError, match="Parent-only"):
        runner.load_parent_runtime(tmp_path, parent_id, "engine--b0", training=True)


def test_ablation_override_requires_authenticated_task_parent_and_zero_training(
    tmp_path, monkeypatch
):
    calls = []

    def expected(root, task):
        calls.append(task)
        if task != "ablation--P3--b2":
            raise PermissionError("Unregistered ablation task")
        return "c2"

    monkeypatch.setitem(
        sys.modules, "sr_f3_ce.ablation", SimpleNamespace(runtime_revision_for_task=expected)
    )
    assert (
        runner._runtime_revision(
            tmp_path,
            runner.PARENTS[2],
            "ablation--P3--b2",
            {},
            "c2",
            training=False,
        )
        == "c2"
    )
    for parent, revision, training_enabled in [
        (runner.PARENTS[0], "c2", False),
        (runner.PARENTS[2], "v2", False),
        (runner.PARENTS[2], "c2", True),
    ]:
        with pytest.raises(PermissionError):
            runner._runtime_revision(
                tmp_path, parent, "ablation--P3--b2", {}, revision, training=training_enabled
            )
    with pytest.raises(PermissionError, match="frozen run"):
        runner._runtime_revision(
            tmp_path, runner.PARENTS[0], "engine--b0", {}, "v2", training=False
        )
    assert (
        runner._runtime_revision(
            tmp_path, runner.PARENTS[0], "engine--b0", {}, "v1", training=False
        )
        == "v1"
    )
    assert calls[0] == "ablation--P3--b2"


def test_ablation_run_stage_blocks_regular_runtime_and_training_before_loading(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        runner, "verify_run", lambda *args, **kwargs: ({"run_stage": "PROTOCOL_ABLATION"}, {})
    )
    with pytest.raises(PermissionError, match="zero-training ablation run stage"):
        runner.load_runtime(tmp_path, runner.PARENTS[0], "engine--resume")
    with pytest.raises(PermissionError, match="cannot execute ENGINE"):
        runner.execute_training_segment(tmp_path, engine=True)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(runner, "verify_run", lambda *args, **kwargs: ({}, {}))
    with pytest.raises(PermissionError, match="zero-training ablation run stage"):
        runner.load_parent_runtime(
            tmp_path, runner.PARENTS[0], "ablation--P1--b0", protocol_revision="v1"
        )


def test_work_occurrence_uses_raw_fields_preserves_invalid_denominator_and_scores():
    texts = [
        '{"evidence":[],"work":"reason","answer":1}',
        '{"evidence":[],"work":"","answer":1}',
        '{"evidence":[],"Work":"case mismatch","answer":1}',
        '{"work":"first","work":"duplicate","answer":1}',
        'junk {"work":"not at beginning","answer":1}',
        '{"work":"missing answer"}',
    ]
    rows = [
        dict(full_text=text, reward=0, score={"value_parseable": int(i < 3)})
        for i, text in enumerate(texts)
    ]
    before = copy.deepcopy(rows)
    summary = runner.work_occurrence(rows)
    assert rows == before
    assert summary["records"] == 6 and summary["invalid_initial_objects"] == 2
    assert summary["work_key_present"] == 3 and summary["work_nonempty_string"] == 2
    assert summary["work_fraction_all_records"] == 0.5
    assert summary["work_fraction_parseable_answers"] == 2 / 3
    assert summary["changes_rewards"] is summary["changes_training_control"] is False


def committed_fixture(directory, schedule, tasks, store, identity, step):
    """Durable original-record fixture with the registered sequential hash chain."""
    chain = runner.digest([])
    used_tasks, used_support = [], []
    for item in schedule[:step]:
        hashes, support_hashes = [], []
        for slot, qid in enumerate(item["rl_qids"]):
            path = directory / "rollouts" / f"{item['step']:02d}-{slot:02d}.json"
            request = dict(
                run_identity_sha256=runner.digest(identity),
                step=item["step"],
                slot=slot,
                qid=qid,
                input_hash=runner.digest(store.model_input(qid)),
                task_hash=runner.digest(tasks[qid]),
                policy_hash="b" * 64,
                seeds=list(range(8)),
            )
            records = [
                dict(qid=qid, raw_text="fixture-only cached response", draw=i) for i in range(8)
            ]
            write_json(path.with_suffix(".request.json"), request)
            write_json(
                path, dict(identity=request, records=records, records_sha256=runner.digest(records))
            )
            hashes.append(runner.file_hash(path))
        used_tasks.extend(item["rl_qids"])
        if identity["arm"] != "RL":
            for qid in item["support_qids"]:
                path = directory / "support" / f"{item['step']:02d}-{qid}.json"
                task = tasks[qid]
                target = (
                    task["teacher"][identity["arm"]]
                    if task["teacher"]
                    else support_target(task["world"], task["query"], identity["arm"] != "GOLD")
                )
                write_json(
                    path, dict(qid=qid, step=item["step"], arm=identity["arm"], target=target)
                )
                support_hashes.append(runner.file_hash(path))
            used_support.extend(item["support_qids"])
        chain = runner.digest([chain, item["step"], hashes, support_hashes])
    return dict(
        committed_logical_step=step,
        microbatch_size=4,
        run_identity=identity,
        used_task_qids=used_tasks,
        used_support_qids=used_support,
        committed_records_hash=chain,
    )


def test_engine_schedule_consumes_all_and_only_64_fixed_engine_tasks(dataset):
    schedule = runner.engine_schedule(dataset.tasks)
    assert [row["step"] for row in schedule] == [1, 2, 3, 4]
    flattened = [qid for row in schedule for qid in row["rl_qids"]]
    expected = {qid for qid, task in dataset.tasks.items() if task["pool"] == "ENGINE"}
    assert len(flattened) == len(set(flattened)) == 64
    assert set(flattened) == expected
    for row in schedule:
        tasks = [dataset.tasks[qid] for qid in row["rl_qids"]]
        assert Counter(task["family"] for task in tasks) == dict.fromkeys(FAMILIES, 4)
        assert len(row["support_qids"]) == 4
        assert all(dataset.tasks[qid]["teacher"] is None for qid in row["support_qids"])
        assert all(
            dataset.tasks[qid]["chart"] == "grouped_bar" and dataset.tasks[qid]["mode"] == "repair"
            for qid in row["support_qids"]
        )


@pytest.mark.parametrize("arm", ["GOLD", "PREFIX", "EXEC"])
@pytest.mark.parametrize("engineering", [False, True])
@pytest.mark.parametrize("revision", ["v1", "v2", "c2"])
def test_support_passes_full_target_dictionary_from_correct_pool(
    dataset, tmp_path, arm, engineering, revision
):
    from sr_f3_ce.protocol_revision import serialize_teacher_target

    runtime = SupportRuntimeSpy(revision)
    serialization = "canonical" if revision == "v1" else "indent_2"
    store = InputStoreSpy(tmp_path, dataset)
    item = runner.engine_schedule(dataset.tasks)[0] if engineering else dataset.schedules[0][0]
    records = runner._support(
        runtime,
        store,
        dataset.tasks,
        item["support_qids"],
        arm,
        serialization,
        tmp_path,
        1,
    )
    assert len(records) == len(runtime.calls) == 4
    for qid, call in zip(item["support_qids"], runtime.calls, strict=True):
        task = dataset.tasks[qid]
        expected = (
            support_target(task["world"], task["query"], arm != "GOLD")
            if engineering
            else task["teacher"][arm]
        )
        expected = serialize_teacher_target(expected, revision)
        assert call == (expected, arm, serialization)
        saved = json.loads((tmp_path / "support" / f"01-{qid}.json").read_text())
        assert saved["target"] == expected
        assert saved["qid"] == qid and saved["step"] == 1
        assert saved["protocol_revision"] == revision
    if revision != "v1":
        calls = len(runtime.calls)
        with pytest.raises(PermissionError, match="indent=2"):
            runner._support(
                runtime, store, dataset.tasks, item["support_qids"], arm, "canonical", tmp_path, 2
            )
        assert len(runtime.calls) == calls


@pytest.mark.parametrize(
    "engineering, arm, restored_step, end",
    [
        (True, "EXEC", 0, 2),
        (True, "EXEC", 2, 4),
        (False, "GOLD", 8, 64),
        (False, "RL", 8, 64),
    ],
)
def test_runner_uses_updates_key_and_resume_cursor_without_off_by_one(
    dataset, tmp_path, monkeypatch, engineering, arm, restored_step, end
):
    """Actual runner loop; runtime/optimizer/gradients are replaced by strict spies."""
    runtime = SupportRuntimeSpy()
    store = InputStoreSpy(tmp_path, dataset)
    run = next(row for row in dataset.runs if row["block"] == 0 and row["arm"] == arm)
    schedule = runner.engine_schedule(dataset.tasks) if engineering else dataset.schedules[0]
    directory = tmp_path / ("engine/resume/continuous" if engineering else f"runs/{run['run_id']}")
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True)
    for name, value in {
        "RUN_IDENTITY.json": {"test": True},
        "native/SUPPORT_MASK_RECEIPT.json": {"serialization": "canonical"},
        "plan/config/SR_F3_CE.json": {"generation": {}},
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    actual_run = (
        dict(
            run_id="SRF3CE_ENGINE_EXEC_b0",
            arm="EXEC",
            block=0,
            future_seed=2026101601,
            parent_id=runner.PARENTS[0],
            updates=4,
        )
        if engineering
        else run
    )
    identity = dict(
        actual_run,
        stage="ENGINE" if engineering else "SCIENCE",
        experiment_id=runner.EXPERIMENT_ID,
        run_identity_sha256=runner.file_hash(tmp_path / "RUN_IDENTITY.json"),
        reference_hash="b" * 64,
    )
    restored = committed_fixture(directory, schedule, dataset.tasks, store, identity, restored_step)
    if restored_step:
        (checkpoints / f"commit-{restored_step:08d}.json").write_text("{}")
    monkeypatch.setattr(runner, "dataset_and_store", lambda root: (dataset, store))
    monkeypatch.setattr(runner, "verify_run", lambda *a, **kw: ({}, {"generation": {}}))
    monkeypatch.setattr(
        runner,
        "load_runtime",
        lambda *a: (runtime, SimpleNamespace(state={}), object()),
    )
    monkeypatch.setattr(runner, "shard_lease", lambda path: nullcontext())
    monkeypatch.setattr(runner, "seed_all", lambda seed: None)
    monkeypatch.setattr(runner, "capture_rng", lambda: {})
    monkeypatch.setattr(runner, "trainable_state", lambda model: {"adapter": [1]})
    monkeypatch.setattr(runner, "state_hash", lambda state: "b" * 64)
    monkeypatch.setattr(runner, "publish_adapter", lambda *a: None)
    monkeypatch.setattr(runner, "monitor", lambda *a: None)
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *a, **kw: copy.deepcopy(restored))
    monkeypatch.setattr(
        checkpoint, "read_checkpoint", lambda *a, **kw: (copy.deepcopy(restored), {})
    )
    probes, physical_steps, final_logps = [], [], []

    def engine_probe(*args):
        probes.append(True)
        return {"selected_microbatch_size": 4}

    monkeypatch.setattr(engine_recovery, "verify_engine_probes", engine_probe)
    monkeypatch.setattr(
        engine_recovery,
        "reserve_engine_step",
        lambda root, metadata: physical_steps.append(metadata),
    )
    monkeypatch.setattr(
        engine_recovery,
        "record_engine_final_logprobs",
        lambda root, runtime, segment: final_logps.append(segment),
    )
    committed = []

    def capture_checkpoint(runtime, optimizer, scheduler, *, step, **kwargs):
        return {
            "committed_logical_step": step,
            "microbatch_size": kwargs["microbatch_size"],
            "used_task_qids": list(kwargs["used_task_qids"]),
            "used_support_qids": list(kwargs["used_support_qids"]),
            "committed_records_hash": kwargs["committed_records_hash"],
            "run_identity": kwargs["run_identity"],
        }

    def commit(directory, state):
        committed.append(copy.deepcopy(state))
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "LATEST.json").write_text(
            json.dumps({"step": state["committed_logical_step"]})
        )

    monkeypatch.setattr(checkpoint, "checkpoint_state", capture_checkpoint)
    monkeypatch.setattr(checkpoint, "commit_checkpoint", commit)
    observed_groups, observed_updates = [], []

    def group(runtime, root, store, task, directory, *, run_identity, step, slot, seeds):
        assert len(seeds) == 8
        assert run_identity["updates"] == (4 if engineering else 64)
        assert task["qid"] == schedule[step - 1]["rl_qids"][slot]
        observed_groups.append((step, task["qid"]))
        return [{"qid": task["qid"]} for _ in range(8)], "d" * 64

    def update(runtime, optimizer, scheduler, records, *, step, support_records, **kwargs):
        assert len(records) == 128
        if arm == "RL":
            assert support_records is None
        else:
            assert [r["qid"] for r in support_records] == schedule[step - 1]["support_qids"]
        observed_updates.append(step)
        callback = kwargs.get("optimizer_step_callback")
        if callback:
            callback({"step": step})
        return {"microbatch_size": 4}

    monkeypatch.setattr(runner, "durable_training_group", group)
    monkeypatch.setattr(training, "update", update)
    result = runner.execute_training_segment(
        tmp_path, run=None if engineering else run, engine=engineering, stop_step=end
    )
    assert observed_updates == list(range(restored_step + 1, end + 1))
    assert len(observed_groups) == 16 * (end - restored_step)
    assert result["step"] == end
    assert committed[-1]["committed_logical_step"] == end
    assert len(committed[-1]["used_task_qids"]) == 16 * end
    assert len(committed[-1]["used_support_qids"]) == (0 if arm == "RL" else 4 * end)
    assert len(probes) == int(engineering and restored_step == 0)
    assert len(physical_steps) == (end - restored_step if engineering else 0)
    assert final_logps == (["continuous"] if engineering and end == 4 else [])


@pytest.mark.parametrize(
    "tamper", ["raw", "raw_and_row_hash", "request", "task", "input", "support", "cursor"]
)
def test_committed_records_reject_tampering_with_actual_disk_chain(dataset, tmp_path, tamper):
    schedule = dataset.schedules[0][:2]
    needed = {qid for row in schedule for key in ("rl_qids", "support_qids") for qid in row[key]}
    tasks = {qid: copy.deepcopy(dataset.tasks[qid]) for qid in needed}
    inputs = {qid: copy.deepcopy(dataset.inputs[qid]) for qid in needed}
    store = InputStoreSpy(tmp_path, SimpleNamespace(inputs=inputs))
    identity = {"arm": "EXEC", "run_id": "SRF3CE_EXEC_b0"}
    state = committed_fixture(tmp_path, schedule, tasks, store, identity, 2)
    assert (
        runner.verify_committed_records(tmp_path, state, schedule, tasks, store)
        == state["committed_records_hash"]
    )
    qid = schedule[0]["rl_qids"][0]
    raw_path = tmp_path / "rollouts/01-00.json"
    if tamper in ("raw", "raw_and_row_hash"):
        saved = json.loads(raw_path.read_text())
        saved["records"][0]["raw_text"] = "changed after commit"
        if tamper == "raw_and_row_hash":
            saved["records_sha256"] = runner.digest(saved["records"])
        write_json(raw_path, saved)
    elif tamper == "request":
        request_path = raw_path.with_suffix(".request.json")
        request = json.loads(request_path.read_text())
        request["input_hash"] = "e" * 64
        write_json(request_path, request)
    elif tamper == "task":
        tasks[qid]["answer"] = "changed gold sidecar"
    elif tamper == "input":
        inputs[qid]["prompt"] += " changed frozen input"
    elif tamper == "support":
        path = tmp_path / "support" / f"01-{schedule[0]['support_qids'][0]}.json"
        saved = json.loads(path.read_text())
        saved["target"]["text"] = "changed teacher target"
        write_json(path, saved)
    else:
        state["used_task_qids"] = state["used_task_qids"][1:]
    with pytest.raises(PermissionError, match="Committed"):
        runner.verify_committed_records(tmp_path, state, schedule, tasks, store)


def monitor_fixture(root, tasks, run_id, step, parameters):
    slots = evaluation.monitor_slots(tasks, run_id, step)
    groups = evaluation.group_slots(slots)
    directory = root / "monitor" / run_id / f"step-{step:02d}"
    hashes, rows = [], []
    for group in groups:
        records = [
            dict(
                slot,
                generation_status="COMPLETE",
                technical_validation_errors=[],
                raw_text="fixture-only monitor response",
            )
            for slot in group["slots"]
        ]
        path = directory / "groups" / (group["group_id"] + ".json")
        write_json(
            path,
            dict(
                group_id=group["group_id"], records=records, records_sha256=runner.digest(records)
            ),
        )
        hashes.append(dict(group_id=group["group_id"], sha256=runner.file_hash(path)))
        rows.extend(records)
    coverage = dict(
        evaluation.validate_coverage(rows, slots),
        groups=hashes,
        policy_hash=runner.state_hash(parameters),
        records_sha256=runner.digest(hashes),
        rng_isolated=True,
    )
    write_json(directory / "COVERAGE.json", coverage)
    write_json(directory / "COMPLETE.json", dict(coverage, model_id=run_id, step=step, families={}))
    return directory


def test_restore_current_monitor_boundary_runs_before_continuing(dataset, tmp_path, monkeypatch):
    parameters = {"adapter": [1, 2, 3]}
    run = next(row for row in dataset.runs if row["arm"] == "EXEC" and row["block"] == 0)
    prior = monitor_fixture(tmp_path, dataset.tasks, run["run_id"], 16, parameters)
    hashes = {str(p): runner.file_hash(p) for p in prior.rglob("*.json")}
    calls = []
    monkeypatch.setattr(
        checkpoint, "read_checkpoint", lambda directory, *, step: ({"parameters": parameters}, {})
    )
    monkeypatch.setattr(runner, "monitor", lambda *args: calls.append(args[-1]))
    runner.restore_monitor_boundary(
        object(), tmp_path, None, dataset.tasks, run, 32, tmp_path / "checkpoints"
    )
    assert calls == [32]
    assert hashes == {str(p): runner.file_hash(p) for p in prior.rglob("*.json")}


@pytest.mark.parametrize(
    "tamper",
    ["missing", "policy", "raw", "coverage_truncation", "coverage_rehash", "slot_metadata"],
)
def test_restore_monitor_rejects_missing_or_changed_prior_boundary(
    dataset, tmp_path, monkeypatch, tamper
):
    parameters = {"adapter": [1, 2, 3]}
    run = next(row for row in dataset.runs if row["arm"] == "EXEC" and row["block"] == 0)
    directory = monitor_fixture(tmp_path, dataset.tasks, run["run_id"], 16, parameters)
    if tamper == "missing":
        (directory / "COMPLETE.json").unlink()
    elif tamper == "policy":
        path = directory / "COMPLETE.json"
        record = json.loads(path.read_text())
        record["policy_hash"] = "d" * 64
        write_json(path, record)
    elif tamper == "raw":
        next((directory / "groups").glob("*.json")).write_text("changed raw record")
    elif tamper == "coverage_truncation":
        path = directory / "COVERAGE.json"
        coverage = json.loads(path.read_text())
        coverage["groups"] = []
        write_json(path, coverage)
    else:
        coverage = json.loads((directory / "COVERAGE.json").read_text())
        if tamper == "coverage_rehash":
            coverage["groups"] = []
        else:
            group = coverage["groups"][0]
            path = directory / "groups" / (group["group_id"] + ".json")
            saved = json.loads(path.read_text())
            saved["records"][0]["qid"] = "changed registered monitor slot"
            saved["records_sha256"] = runner.digest(saved["records"])
            write_json(path, saved)
            group["sha256"] = runner.file_hash(path)
        coverage["records_sha256"] = runner.digest(coverage["groups"])
        write_json(directory / "COVERAGE.json", coverage)
        complete = json.loads((directory / "COMPLETE.json").read_text())
        complete.update(coverage)
        write_json(directory / "COMPLETE.json", complete)
    monkeypatch.setattr(
        checkpoint, "read_checkpoint", lambda directory, *, step: ({"parameters": parameters}, {})
    )
    calls = []
    monkeypatch.setattr(runner, "monitor", lambda *args: calls.append(args[-1]))
    with pytest.raises(PermissionError):
        runner.restore_monitor_boundary(
            object(), tmp_path, None, dataset.tasks, run, 24, tmp_path / "checkpoints"
        )
    assert calls == []
