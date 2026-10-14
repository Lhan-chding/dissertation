import json
import os
import shutil
from types import SimpleNamespace

import pytest

from sr_f2_replay.replay import (
    archive_partial,
    build_plan_bindings,
    compose_canonical,
    exclusive_intent,
    file_hash,
    measured_generate,
    object_hash,
    publish_replay,
    replay_account,
    replay_once,
    verify_provenance,
    verify_replay_account,
    verify_replay_timing,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def setup_replay(tmp_path, monkeypatch, *, preserved=3):
    import sr_f2.evaluation

    monkeypatch.setattr(sr_f2.evaluation, "validate_record", lambda *args, **kwargs: None)
    monkeypatch.setenv("SLURM_JOB_ID", "333")
    task = dict(kind="endpoint", model_id="SRF2_ADAPT_s71001")
    key = "endpoint--" + task["model_id"]
    plan_dir = tmp_path / "replay/test"
    task_dir = plan_dir / "tasks" / key
    gid = "a" * 64
    seed = 2**61 + 17  # Synthetic exact integer, above JS integer precision.
    rows = [
        dict(
            tokens=[10, 20 + index],
            raw_tokens=[10, 20 + index],
            seed=seed + index,
            group_seed=seed,
            group_row_index=index,
            group_size=4,
            slot_id=f"slot-{index}",
            generation_seconds=10.0,
            source_sha256="b" * 64,
            sampler_logprobs=[-0.1, -0.2],
            prompt_tensor_hash="p" * 64,
        )
        for index in range(4)
    ]
    original = [dict(row, generation_seconds=7.0) for row in rows[:preserved]]
    ledger_path = tmp_path / "accounting" / ("endpoint-" + task["model_id"] + ".jsonl")
    write_rows(ledger_path, [dict(kind="completion_attempts", count=4)])
    pending_path = (
        tmp_path / "sealed/evaluation" / task["model_id"] / "groups" / (gid + ".pending.jsonl")
    )
    if preserved:
        write_rows(pending_path, original)
    group = dict(group_id=gid, group_index=7, slots=[dict(slot_id=f"slot-{i}") for i in range(4)])
    spec = dict(
        task=task,
        group_id=gid,
        group_seed=seed,
        count=4,
        preserved_count=preserved,
        pending_path=str(pending_path.relative_to(tmp_path)) if preserved else None,
        pending_sha256=file_hash(pending_path) if preserved else None,
        original_ledger_path=str(ledger_path.relative_to(tmp_path)),
        original_ledger_sha256=file_hash(ledger_path),
        original_ledger_bytes=ledger_path.stat().st_size,
    )
    plan = SimpleNamespace(
        root=tmp_path,
        directory=plan_dir,
        sha256="c" * 64,
        plan=dict(replay_id="test"),
        tasks={key: spec},
        task_directory=lambda task: task_dir,
    )
    save(tmp_path / "REGISTRATION.json", dict(source_manifest_sha256="b" * 64))
    save(
        task_dir / "INTENT.json",
        dict(
            plan_sha256=plan.sha256,
            task=task,
            group_id=gid,
            slots_sha256=object_hash(group["slots"]),
            input_hash="i" * 64,
            model_identity={},
            adapter_identity={},
            physical_attempt_budget=1,
            physical_rows=4,
        ),
    )
    save(
        task_dir / "staging/groups" / (gid + ".json"),
        dict(group_id=gid, group_index=7, records=rows, records_sha256=object_hash(rows)),
    )
    reserve = replay_account(plan, task)
    reserve("completion_attempts", 4, dict(group_seed=seed, qid=None))
    for index, row in enumerate(rows):
        reserve("generated_tokens", len(row["tokens"]), dict(group_seed=seed, row_index=index))
    return plan, task, group, original, rows


def test_preserved_prefix_reuses_original_objects_after_token_match():
    original = [dict(tokens=[1, 2], raw_tokens=[1, 2], generation_seconds=7) for _ in range(3)]
    replay = [dict(tokens=[1, 2], raw_tokens=[1, 2], generation_seconds=10) for _ in range(4)]
    canonical = compose_canonical(original, replay)
    assert all(canonical[index] is original[index] for index in range(3))
    assert canonical[3] is replay[3]


@pytest.mark.parametrize("change", ["tokens", "raw_tokens", "prompt_tensor_hash", "sampling_hash"])
def test_preserved_prefix_divergence_prevents_canonical_publication(tmp_path, monkeypatch, change):
    plan, task, group, _, replay = setup_replay(tmp_path, monkeypatch)
    staged = plan.task_directory(task) / "staging/groups" / (group["group_id"] + ".json")
    replay[1][change] = [999, 1] if change.endswith("tokens") else "changed"
    save(
        staged,
        dict(
            group_id=group["group_id"],
            group_index=7,
            records=replay,
            records_sha256=object_hash(replay),
        ),
    )
    with pytest.raises(PermissionError):
        publish_replay(plan, task, group)
    target = (
        tmp_path / "sealed/evaluation" / task["model_id"] / "groups" / (group["group_id"] + ".json")
    )
    assert not target.exists()
    assert target.with_suffix(".pending.jsonl").exists()


def test_publish_retains_original_bytes_and_moves_with_receipt(tmp_path, monkeypatch):
    plan, task, group, original, replay = setup_replay(tmp_path, monkeypatch)
    spec = plan.tasks["endpoint--" + task["model_id"]]
    old_ledger = (tmp_path / spec["original_ledger_path"]).read_bytes()
    original_pending = tmp_path / spec["pending_path"]
    original_bytes = original_pending.read_bytes()
    receipt = publish_replay(plan, task, group)
    assert receipt["accounting"]["physical_attempt_rows"] == 4
    assert receipt["accounting"]["canonical_new_rows"] == 1
    assert receipt["accounting"]["duplicate_verification_rows"] == 3
    assert receipt["accounting"]["duplicate_verification_tokens"] == 6
    assert not original_pending.exists()
    archive = plan.directory / receipt["archive"]["archived_path"]
    assert archive.read_bytes() == original_bytes
    assert (tmp_path / spec["original_ledger_path"]).read_bytes() == old_ledger
    canonical = json.loads((tmp_path / receipt["canonical_path"]).read_text())["records"]
    assert canonical == original + replay[3:]
    assert verify_provenance(plan, task, group) == receipt


def test_zero_preserved_group_publishes_all_four(tmp_path, monkeypatch):
    plan, task, group, _, replay = setup_replay(tmp_path, monkeypatch, preserved=0)
    receipt = publish_replay(plan, task, group)
    assert receipt["archive"] is None
    assert receipt["accounting"]["canonical_new_rows"] == 4
    assert receipt["accounting"]["duplicate_verification_rows"] == 0
    assert json.loads((tmp_path / receipt["canonical_path"]).read_text())["records"] == replay


def test_receipt_only_reentry_never_calls_runtime_factory(tmp_path, monkeypatch):
    plan, task, group, _, _ = setup_replay(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("A second model invocation is forbidden")

    first = replay_once(plan, task, group, {}, {}, runtime_factory=forbidden)
    second = replay_once(plan, task, group, {}, {}, runtime_factory=forbidden)
    assert first == second


def test_interrupted_attempt_without_staged_result_is_not_retried(tmp_path, monkeypatch):
    plan, task, group, _, _ = setup_replay(tmp_path, monkeypatch)
    (plan.task_directory(task) / "staging/groups" / (group["group_id"] + ".json")).unlink()
    with pytest.raises(PermissionError, match="no second attempt"):
        replay_once(plan, task, group, {}, {}, runtime_factory=lambda *a, **k: pytest.fail())


def test_attempt_reservation_is_exclusive_and_preserves_first(tmp_path, monkeypatch):
    plan, task, _, _, _ = setup_replay(tmp_path, monkeypatch)
    path = plan.task_directory(task) / "accounting.jsonl"
    before = path.read_bytes()
    with pytest.raises(PermissionError, match="already been charged"):
        replay_account(plan, task)("completion_attempts", 4, dict(group_seed=2**61 + 17))
    assert path.read_bytes() == before


def test_intent_o_excl_cannot_overwrite_first_attempt(tmp_path):
    path = tmp_path / "INTENT.json"
    exclusive_intent(path, dict(attempt=1))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        exclusive_intent(path, dict(attempt=2))
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "mode", ["missing", "extra", "wrong_row", "wrong_seed", "wrong_tokens", "boolean_row"]
)
def test_replay_token_ledger_must_exactly_cover_all_physical_rows(tmp_path, monkeypatch, mode):
    plan, task, _, _, rows = setup_replay(tmp_path, monkeypatch)
    path = plan.task_directory(task) / "accounting.jsonl"
    ledger = [json.loads(line) for line in path.read_text().splitlines()]
    if mode == "missing":
        ledger.pop()
    elif mode == "extra":
        ledger.append(dict(ledger[-1], time_ns=ledger[-1]["time_ns"] + 1))
    elif mode == "wrong_row":
        ledger[-1]["metadata"]["row_index"] = 0
    elif mode == "wrong_seed":
        ledger[-1]["metadata"]["group_seed"] += 1
    elif mode == "boolean_row":
        ledger[1]["metadata"]["row_index"] = False
    else:
        ledger[-1]["count"] += 1
    write_rows(path, ledger)
    with pytest.raises(PermissionError, match="accounting"):
        verify_replay_account(plan, task, rows)


def test_original_ledger_append_allowed_but_prefix_mutation_rejected(tmp_path, monkeypatch):
    plan, task, group, _, _ = setup_replay(tmp_path, monkeypatch)
    receipt = publish_replay(plan, task, group)
    path = tmp_path / receipt["original_ledger_prefix"]["path"]
    with path.open("a") as stream:
        stream.write('{"kind":"subsequent_frozen_generation"}\n')
    verify_provenance(plan, task, group)
    original = path.read_text()
    path.write_text(original.replace('"count": 4', '"count": 5', 1))
    with pytest.raises(PermissionError, match="ledger prefix"):
        verify_provenance(plan, task, group)


def test_crash_after_archive_link_is_receipt_only_recoverable(tmp_path, monkeypatch):
    plan, task, _, _, _ = setup_replay(tmp_path, monkeypatch)
    spec = plan.tasks["endpoint--" + task["model_id"]]
    original = tmp_path / spec["pending_path"]
    archived = plan.task_directory(task) / "archive" / original.name
    archived.parent.mkdir()
    os.link(original, archived)
    receipt = archive_partial(plan, task)
    assert not original.exists()
    assert file_hash(archived) == spec["pending_sha256"] == receipt["sha256"]


def test_canonical_existing_different_bytes_is_never_overwritten(tmp_path, monkeypatch):
    plan, task, group, _, _ = setup_replay(tmp_path, monkeypatch)
    target = (
        tmp_path / "sealed/evaluation" / task["model_id"] / "groups" / (group["group_id"] + ".json")
    )
    save(target, dict(unrelated="preserve"))
    before = target.read_bytes()
    with pytest.raises(PermissionError):
        publish_replay(plan, task, group)
    assert target.read_bytes() == before


def test_original_endpoint_shard_lock_blocks_even_receipt_only_reentry(tmp_path, monkeypatch):
    from sr_f12.evaluation import shard_lease

    plan, task, group, _, _ = setup_replay(tmp_path, monkeypatch)
    with (
        shard_lease(tmp_path / "sealed/evaluation" / task["model_id"]),
        pytest.raises(RuntimeError, match="Duplicate worker"),
    ):
        replay_once(plan, task, group, {}, {})


def test_crash_after_archive_before_canonical_is_receipt_only_recoverable(tmp_path, monkeypatch):
    import sr_f2_replay.replay as module

    plan, task, group, _, _ = setup_replay(tmp_path, monkeypatch)
    spec = plan.tasks["endpoint--" + task["model_id"]]
    target = (
        tmp_path / "sealed/evaluation" / task["model_id"] / "groups" / (group["group_id"] + ".json")
    )
    original_durable = module.durable

    def interrupted(path, value):
        if path == target:
            raise OSError("simulated publish crash")
        return original_durable(path, value)

    monkeypatch.setattr(module, "durable", interrupted)
    with pytest.raises(OSError, match="simulated"):
        publish_replay(plan, task, group)
    assert not (tmp_path / spec["pending_path"]).exists()
    archive = plan.task_directory(task) / "archive" / (group["group_id"] + ".pending.jsonl")
    assert file_hash(archive) == spec["pending_sha256"]
    assert not target.exists()
    monkeypatch.setattr(module, "durable", original_durable)
    receipt = replay_once(plan, task, group, {}, {}, runtime_factory=lambda *a, **k: pytest.fail())
    assert receipt["status"] == "REPLAY_GROUP_PUBLISHED"


def test_full_staging_pending_finishes_receipts_without_second_generation(tmp_path, monkeypatch):
    plan, task, group, _, rows = setup_replay(tmp_path, monkeypatch)
    staged = plan.task_directory(task) / "staging/groups" / (group["group_id"] + ".json")
    pending = staged.with_suffix(".pending.jsonl")
    write_rows(pending, rows)
    original_bytes = pending.read_bytes()
    staged.unlink()
    receipt = replay_once(plan, task, group, {}, {}, runtime_factory=lambda *a, **k: pytest.fail())
    assert receipt["status"] == "REPLAY_GROUP_PUBLISHED"
    assert pending.read_bytes() == original_bytes
    assert (plan.task_directory(task) / "STAGING_RECEIPT_RECOVERY.json").exists()
    assert receipt["timing"]["status"] == "UNKNOWN"
    assert receipt["timing"]["generation_call_wall_seconds"] is None


def test_first_replay_calls_frozen_path_once_and_retains_measured_cost(tmp_path, monkeypatch):
    import sr_f2.diagnostics
    import sr_f2.evaluation
    import sr_f2.workers
    import sr_f2_replay.replay as module

    plan, task, group, original, rows = setup_replay(tmp_path, monkeypatch)
    shutil.rmtree(plan.task_directory(task))
    calls = []
    plan.baseline = lambda received: calls.append(("baseline", received))
    save(tmp_path / "config/SR_F2.json", dict(frozen=True))
    (tmp_path / "image.bin").write_bytes(b"synthetic image")
    monkeypatch.setattr(
        sr_f2.diagnostics,
        "prepare_input",
        lambda *args: dict(text="synthetic prompt", image_file="image.bin"),
    )
    monkeypatch.setattr(
        sr_f2.workers, "load_policy", lambda *args: calls.append(("policy", args[2]))
    )
    ticks = iter([100, 150, 250, 280])
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(ticks))

    class Runtime:
        tokenizer = object()

        def stable_model_identity(self):
            return {}

        def current_adapter_identity(self):
            return {}

        def generate_group(self, marker):
            calls.append(("generate", marker))
            self.account("completion_attempts", 4, dict(group_seed=2**61 + 17, qid=None))
            for index, row in enumerate(rows):
                self.account(
                    "generated_tokens",
                    len(row["tokens"]),
                    dict(group_seed=2**61 + 17, row_index=index),
                )
            return rows

    def frozen_group(runtime, received, _inputs, _diagnostics, _tasks, root, directory, **options):
        assert root == plan.root and received == group and options["score"] is False
        generated = runtime.generate_group(marker="same arguments")
        save(
            directory / "groups" / (group["group_id"] + ".json"),
            dict(
                group_id=group["group_id"],
                group_index=7,
                records=generated,
                records_sha256=object_hash(generated),
            ),
        )

    monkeypatch.setattr(sr_f2.evaluation, "run_group", frozen_group)

    def factory(config, root, *, task_id):
        assert config == dict(frozen=True) and root == plan.root
        assert task_id == "endpoint-" + task["model_id"]
        return (Runtime(),)

    receipt = replay_once(plan, task, group, {}, {}, runtime_factory=factory)
    assert calls == [("baseline", task), ("policy", 96), ("generate", "same arguments")]
    assert receipt["retained_original_row_hashes"] == [object_hash(row) for row in original]
    timing = receipt["timing"]
    assert timing["status"] == "COMPLETE"
    assert timing["generation_call_wall_seconds"] == 100 / 1e9
    assert timing["preparation_wall_seconds"] == 50 / 1e9
    assert timing["post_generation_staging_validation_wall_seconds"] == 30 / 1e9
    assert timing["total_to_staged_validation_wall_seconds"] == 180 / 1e9
    assert (
        replay_once(plan, task, group, {}, {}, runtime_factory=lambda *a, **k: pytest.fail())
        == receipt
    )
    assert verify_provenance(plan, task, group) == receipt


@pytest.mark.parametrize("missing", ["generation", "end_receipt", "staging_receipt"])
def test_missing_timing_keeps_cost_honest_without_generating_again(tmp_path, monkeypatch, missing):
    import sr_f2_replay.replay as module

    plan, task, group, _, _ = setup_replay(tmp_path, monkeypatch)
    ticks = iter([200, 250])
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(ticks))
    original_durable = module.durable

    def durable_without_end(path, value):
        if path.name == "GENERATION_END.json":
            raise OSError("interrupted before durable end receipt")
        return original_durable(path, value)

    if missing == "end_receipt":
        monkeypatch.setattr(module, "durable", durable_without_end)

    def interrupted(**kwargs):
        assert kwargs == dict(same="arguments")
        if missing == "generation":
            raise RuntimeError("interrupted after raw persistence")
        return "generation returned successfully"

    measured, _ = measured_generate(plan, task, interrupted, 100)
    if missing == "staging_receipt":
        assert measured(same="arguments") == "generation returned successfully"
    else:
        with pytest.raises((RuntimeError, OSError)):
            measured(same="arguments")
    timing = verify_replay_timing(plan, task)
    assert timing["status"] == "PARTIAL"
    expected = 50 / 1e9 if missing == "staging_receipt" else None
    assert timing["generation_call_wall_seconds"] == expected
    receipt = replay_once(plan, task, group, {}, {}, runtime_factory=lambda *a, **k: pytest.fail())
    assert receipt["timing"] == timing
    assert (plan.task_directory(task) / "GENERATION_END.json").exists() == (
        missing == "staging_receipt"
    )
    assert publish_replay(plan, task, group) == receipt


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_partial_staging_pending_never_generates_again(tmp_path, monkeypatch, count):
    plan, task, group, _, rows = setup_replay(tmp_path, monkeypatch)
    staged = plan.task_directory(task) / "staging/groups" / (group["group_id"] + ".json")
    write_rows(staged.with_suffix(".pending.jsonl"), rows[:count])
    staged.unlink()
    with pytest.raises(PermissionError, match="no second attempt"):
        replay_once(plan, task, group, {}, {}, runtime_factory=lambda *a, **k: pytest.fail())
    assert not staged.exists()


def test_build_bindings_preserves_exact_seeds_and_frozen_group_ids(tmp_path, monkeypatch):
    import sr_f1.data
    import sr_f2.workers

    slots_by_model, allowed = {}, {}
    source_hash = "b" * 64
    save(tmp_path / "INPUT_IDENTITY.json", dict(test="input"))
    save(tmp_path / "MODEL_ENVIRONMENT_IDENTITY.json", dict(test="model"))
    for n in range(5):
        model, seed = f"SRF2_MODEL_{n}", 2**61 + 17 + n
        task = dict(kind="endpoint", model_id=model)
        key = "endpoint--" + model
        slots = [
            dict(
                model_id=model,
                step=96,
                pool="TEST_ID",
                protocol="answer_only",
                view="v0",
                qid=f"qid-{n}",
                samples_per_prompt=4,
                draw=i,
                generation={"max_new_tokens": 768},
                slot_id=f"{model}-{i}",
                group_seed=seed,
                sampling_seed=seed + i,
            )
            for i in range(4)
        ]
        slots_by_model[model] = slots
        from sr_f12.evaluation import group_slots

        group = group_slots(slots)[0]
        ledger = tmp_path / "accounting" / ("endpoint-" + model + ".jsonl")
        write_rows(ledger, [dict(kind="completion_attempts", count=4, seed=seed)])
        refs = [dict(path=str(ledger.relative_to(tmp_path)), sha256=file_hash(ledger))]
        if n == 0:
            pending = (
                tmp_path
                / "sealed/evaluation"
                / model
                / "groups"
                / (group["group_id"] + ".pending.jsonl")
            )
            write_rows(pending, [dict(index=i) for i in range(3)])
            refs.append(dict(path=str(pending.relative_to(tmp_path)), sha256=file_hash(pending)))
        save(tmp_path / "runs" / model / "checkpoints/commit-96.json", dict(sha256=source_hash))
        allowed[key] = dict(
            status="BLOCKED_INCOMPLETE_GENERATION",
            task=task,
            mutable_prelaunch_files=refs,
            generation_audit=dict(
                gaps=[
                    dict(code="ATTEMPT_WITHOUT_DURABLE_GROUP", group_seed=seed, count=4, groups=1)
                ]
            ),
        )
    parent = SimpleNamespace(
        root=tmp_path,
        allowed=allowed,
        verify=lambda: None,
        path=tmp_path / "parent.json",
        sha256="c" * 64,
        audit=dict(task_matrix_sha256="d" * 64),
        plan=dict(registration_sha256="e" * 64),
    )
    monkeypatch.setattr(sr_f1.data, "load_tasks", lambda root: {})
    monkeypatch.setattr(
        sr_f2.workers, "endpoint_slots", lambda questions, model: slots_by_model[model]
    )
    result = build_plan_bindings(parent.path, parent_factory=lambda path: parent)
    assert len(result["tasks"]) == 5
    assert result["tasks"]["endpoint--SRF2_MODEL_0"]["group_seed"] == 2**61 + 17
    assert sum(item["preserved_count"] for item in result["tasks"].values()) == 3
    assert all(item["count"] == 4 for item in result["tasks"].values())
