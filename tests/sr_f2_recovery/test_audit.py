import copy
import json

import pytest

from sr_f2_recovery.audit import (
    _checkpoint_evidence,
    audit_recovery,
    file_hash,
    generation_audit,
    object_hash,
)

TASK = dict(kind="endpoint", model_id="SRF2_J6_s71003")
SOURCE = "s" * 64


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))


def raw(seed, count, index, **extra):
    return dict(
        group_seed=seed,
        group_size=count,
        group_row_index=index,
        seed=seed + index,
        tokens=[11, 12],
        sampler_logprobs=[-0.1, -0.2],
        generation_status="COMPLETE",
        technical_validation_errors=[],
        qid="q",
        **extra,
    )


def account(root, attempts, task=TASK, *, include_tokens=True):
    name = task["model_id"] if task["kind"] == "train" else task["kind"] + "-" + task["model_id"]
    path = root / "accounting" / (name + ".jsonl")
    rows = []
    for seed, count in attempts:
        rows.append(
            dict(
                experiment_id="SR-F2-20261013",
                task_id=name,
                kind="completion_attempts",
                count=count,
                metadata=dict(group_seed=seed),
                pid=1,
                time_ns=len(rows) + 1,
                slurm_job_id="123",
            )
        )
        if include_tokens:
            for index in range(count):
                rows.append(
                    dict(
                        experiment_id="SR-F2-20261013",
                        task_id=name,
                        kind="generated_tokens",
                        count=2,
                        metadata=dict(group_seed=seed, row_index=index),
                        pid=1,
                        time_ns=len(rows) + 1,
                        slurm_job_id="123",
                    )
                )
    lines(path, rows)
    return path


def evaluation_group(root, *, seed=7, count=4, qid="q", pending=False, partial=None):
    rows = []
    for index in range(count):
        row = raw(
            seed,
            count,
            index,
            model_id=TASK["model_id"],
            source_sha256=SOURCE,
            step=96,
            pool="TEST_ID",
            protocol="answer_only",
            view="v0",
            draw=index,
            samples_per_prompt=count,
            sampling_seed=seed + index,
            slot_id=f"{qid}-{index}",
        )
        row["qid"] = qid
        rows.append(row)
    gid = object_hash(
        [rows[0][field] for field in ("model_id", "step", "pool", "protocol", "view", "qid")]
    )
    directory = root / "sealed/evaluation" / TASK["model_id"] / "groups"
    path = directory / (gid + (".pending.jsonl" if pending else ".json"))
    if pending:
        lines(path, rows if partial is None else rows[:partial])
    else:
        save(
            path, dict(group_id=gid, group_index=0, records=rows, records_sha256=object_hash(rows))
        )
    return path, rows


def test_matching_durable_groups_pass_without_exposing_answers(tmp_path):
    evaluation_group(tmp_path)
    account(tmp_path, [(7, 4)])
    result = generation_audit(tmp_path, TASK, source_sha256=SOURCE)
    assert result["status"] == "PASS"
    assert result["counts"]["attempt_records"] == result["counts"]["durable_records"] == 4
    assert "records" not in result and "scores" not in result
    assert len(result["reference_files"]) == 1
    assert result["mutable_prelaunch_files"][0]["path"].startswith("accounting/")


def test_attempt_without_even_a_pending_file_is_blocked(tmp_path):
    account(tmp_path, [(7, 4)], include_tokens=False)
    result = generation_audit(tmp_path, TASK)
    assert result["status"] == "BLOCKED_INCOMPLETE_GENERATION"
    assert result["gaps"] == [
        dict(code="ATTEMPT_WITHOUT_DURABLE_GROUP", group_seed=7, count=4, groups=1)
    ]


@pytest.mark.parametrize("partial", [0, 1, 2, 3])
def test_partial_pending_blocks_even_if_accounting_is_missing(tmp_path, partial):
    evaluation_group(tmp_path, pending=True, partial=partial)
    result = generation_audit(tmp_path, TASK)
    assert result["status"] == "BLOCKED_INCOMPLETE_GENERATION"
    assert any(gap["code"] == "INCOMPLETE_OR_INVALID_PENDING" for gap in result["gaps"])


def test_full_pending_is_reusable_without_new_generation(tmp_path):
    path, _ = evaluation_group(tmp_path, pending=True)
    account(tmp_path, [(7, 4)])
    result = generation_audit(tmp_path, TASK)
    assert result["status"] == "PASS"
    assert result["counts"]["full_pending_groups"] == 1
    assert str(path.relative_to(tmp_path)) in {
        item["path"] for item in result["mutable_prelaunch_files"]
    }
    assert not result["reference_files"]


def test_greedy_repeated_seed_preserves_multiset_multiplicity(tmp_path):
    evaluation_group(tmp_path, seed=0, count=1, qid="chart-1")
    evaluation_group(tmp_path, seed=0, count=1, qid="chart-2")
    account(tmp_path, [(0, 1), (0, 1), (0, 1)])
    result = generation_audit(tmp_path, TASK)
    assert result["counts"]["final_groups"] == 2
    assert [gap for gap in result["gaps"] if gap["code"] == "ATTEMPT_WITHOUT_DURABLE_GROUP"] == [
        dict(code="ATTEMPT_WITHOUT_DURABLE_GROUP", group_seed=0, count=1, groups=1)
    ]


def test_duplicate_ledger_row_cannot_manufacture_coverage(tmp_path):
    evaluation_group(tmp_path)
    path = account(tmp_path, [(7, 4)])
    path.write_text(path.read_text() * 2)
    result = generation_audit(tmp_path, TASK)
    assert any(gap["code"] == "MALFORMED_ACCOUNTING" for gap in result["gaps"])


def test_unaccounted_raw_group_blocks(tmp_path):
    evaluation_group(tmp_path)
    result = generation_audit(tmp_path, TASK)
    assert any(gap["code"] == "RAW_GROUP_WITHOUT_ATTEMPT" for gap in result["gaps"])


def test_checksum_mismatch_blocks(tmp_path):
    path, _ = evaluation_group(tmp_path)
    value = json.loads(path.read_text())
    value["records"][0]["tokens"] = [99, 99]
    save(path, value)
    account(tmp_path, [(7, 4)])
    assert generation_audit(tmp_path, TASK)["status"] == "BLOCKED_INCOMPLETE_GENERATION"


def test_duplicate_slot_even_with_valid_checksum_blocks(tmp_path):
    path, _ = evaluation_group(tmp_path)
    value = json.loads(path.read_text())
    value["records"][1]["slot_id"] = value["records"][0]["slot_id"]
    value["records_sha256"] = object_hash(value["records"])
    save(path, value)
    account(tmp_path, [(7, 4)])
    assert generation_audit(tmp_path, TASK)["status"] == "BLOCKED_INCOMPLETE_GENERATION"


def test_full_pending_twin_counts_once(tmp_path):
    evaluation_group(tmp_path)
    evaluation_group(tmp_path, pending=True)
    account(tmp_path, [(7, 4)])
    result = generation_audit(tmp_path, TASK)
    assert result["status"] == "PASS"
    assert result["counts"]["retained_pending_groups"] == 1
    assert result["counts"]["durable_records"] == 4


def training_group(root, *, final=True, pending=True):
    task = dict(kind="train", model_id="SRF2_J6_s71003")
    identity = dict(
        plan_id="SR-F2",
        run_id=task["model_id"],
        run_identity_sha256="m" * 64,
        source_sha256=SOURCE,
        scoring="SR-F2-v2",
        step=89,
        slot=0,
        qid="q",
        seeds=list(range(20, 28)),
    )
    directory = root / "runs" / task["model_id"] / "rollouts"
    rows = [raw(20, 8, index) for index in range(8)]
    save(directory / "89-00.request.json", identity)
    if pending:
        lines(directory / "89-00.pending.jsonl", rows)
    if final:
        final_rows = [dict(row, score=dict(A=0), root=str(root), row={}) for row in rows]
        save(
            directory / "89-00.json",
            dict(identity=identity, records=final_rows, sha256=object_hash(final_rows)),
        )
    account(root, [(20, 8)], task)
    return task, directory


def test_training_final_and_retained_raw_pending_count_once(tmp_path):
    task, _ = training_group(tmp_path)
    result = generation_audit(tmp_path, task, source_sha256=SOURCE, manifest_sha256="m" * 64)
    assert result["status"] == "PASS"
    assert result["counts"]["durable_records"] == 8
    assert result["counts"]["retained_pending_groups"] == 1


def test_training_full_pending_is_reusable(tmp_path):
    task, _ = training_group(tmp_path, final=False)
    assert generation_audit(tmp_path, task)["status"] == "PASS"


def test_preserved_failure_blocks_even_when_raw_is_complete(tmp_path):
    task, directory = training_group(tmp_path)
    save(directory / "89-00.failure.json", dict(completed_rows=8))
    result = generation_audit(tmp_path, task)
    assert any(gap["code"] == "PRESERVED_GENERATION_FAILURE" for gap in result["gaps"])


def test_training_request_mismatch_blocks(tmp_path):
    task, directory = training_group(tmp_path)
    request = json.loads((directory / "89-00.request.json").read_text())
    request["run_id"] = "wrong"
    save(directory / "89-00.request.json", request)
    assert generation_audit(tmp_path, task)["status"] == "BLOCKED_INCOMPLETE_GENERATION"


def setup_audit(tmp_path, monkeypatch, tasks=None):
    from sr_f2 import orchestration, protocol

    tasks = tasks or [TASK]
    registration = dict(main_matrix=[], config_sha256="c" * 64, source_manifest_sha256=SOURCE)
    monkeypatch.setattr(protocol, "verify_registration", lambda *a, **k: registration)
    monkeypatch.setattr(orchestration, "global_task_matrix", lambda: copy.deepcopy(tasks))
    (tmp_path / "orchestration/queue/tasks").mkdir(parents=True)
    save(
        tmp_path / "orchestration/queue/CONFIG.json",
        dict(
            tasks=tasks,
            registration_sha256=object_hash(registration),
            task_matrix_sha256=object_hash(tasks),
        ),
    )
    return registration


def test_never_attempted_requires_absence_of_all_original_artifacts(tmp_path, monkeypatch):
    setup_audit(tmp_path, monkeypatch)
    first = audit_recovery(tmp_path)
    key = "endpoint--" + TASK["model_id"]
    assert first["tasks"][key]["status"] == "NEVER_ATTEMPTED"
    account(tmp_path, [])
    second = audit_recovery(tmp_path)
    assert second["tasks"][key]["status"] == "BLOCKED_INCOMPLETE_GENERATION"


def completed_task(root, task, registration):
    key = task["kind"] + "--" + task["model_id"]
    queue = root / "orchestration/queue/tasks" / key
    claim = dict(
        task=task,
        pool_id="GPU_POOL_1",
        device="0",
        job_id="123",
        registration_sha256=object_hash(registration),
        claim_id="claim-id",
    )
    save(queue / "CLAIM.json", claim)
    worker = root / "orchestration/pools/GPU_POOL_1/workers" / key
    save(
        worker / "LAUNCH_INTENT.json",
        dict(
            task=task,
            device="0",
            claim_id="claim-id",
            command=[
                "python",
                "run_worker.py",
                "--root",
                str(root),
                "--kind",
                task["kind"],
                "--model-id",
                task["model_id"],
            ],
        ),
    )
    save(worker / "PROCESS_EXIT.json", dict(returncode=0))
    artifact = root / "sealed/evaluation" / task["model_id"] / "COMPLETE.json"
    save(artifact, dict(status="COMPLETE", model_id=task["model_id"], scores_sealed=True))
    receipt = dict(
        status="COMPLETE",
        **task,
        registration_sha256=object_hash(registration),
        artifacts=[dict(path=str(artifact.relative_to(root)), sha256=file_hash(artifact))],
    )
    save(root / "tasks" / task["kind"] / task["model_id"] / "COMPLETE.json", receipt)
    state = dict(
        claim,
        status="COMPLETE",
        claim_sha256=object_hash(claim),
        result=dict(**task, receipt_sha256=object_hash(receipt)),
    )
    save(queue / "STATE.json", state)
    return worker


def test_completed_inheritance_requires_queue_receipt_and_process_exit(tmp_path, monkeypatch):
    registration = setup_audit(tmp_path, monkeypatch)
    worker = completed_task(tmp_path, TASK, registration)
    result = audit_recovery(tmp_path)
    key = "endpoint--" + TASK["model_id"]
    assert result["tasks"][key]["status"] == "INHERITED_COMPLETE"
    save(worker / "PROCESS_EXIT.json", dict(returncode=-15, pool_aborted=True))
    assert audit_recovery(tmp_path)["tasks"][key]["status"] == "BLOCKED_INCOMPLETE_GENERATION"


def test_audit_does_not_modify_original_files(tmp_path, monkeypatch):
    registration = setup_audit(tmp_path, monkeypatch)
    completed_task(tmp_path, TASK, registration)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    audit_recovery(tmp_path)
    after = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


def test_symlink_raw_file_cannot_escape_original_root(tmp_path):
    path, _ = evaluation_group(tmp_path)
    target = tmp_path / "other.json"
    path.rename(target)
    path.symlink_to(target)
    account(tmp_path, [(7, 4)])
    with pytest.raises(PermissionError):
        generation_audit(tmp_path, TASK)


def test_interrupted_endpoint_resume_requires_complete_attempt_coverage(tmp_path, monkeypatch):
    registration = setup_audit(tmp_path, monkeypatch)
    worker = completed_task(tmp_path, TASK, registration)
    key = "endpoint--" + TASK["model_id"]
    state_path = tmp_path / "orchestration/queue/tasks" / key / "STATE.json"
    state = json.loads(state_path.read_text())
    state.update(status="FAILED")
    state.pop("result")
    save(state_path, state)
    save(worker / "PROCESS_EXIT.json", dict(returncode=-15, pool_aborted=True))
    receipt_dir = tmp_path / "tasks/endpoint" / TASK["model_id"]
    (receipt_dir / "COMPLETE.json").unlink()
    directory = tmp_path / "sealed/evaluation" / TASK["model_id"]
    (directory / "COMPLETE.json").unlink()
    save(directory / "PLAN.json", dict(source_sha256=SOURCE, scores_sealed=True))
    evaluation_group(tmp_path)
    ledger = account(tmp_path, [(7, 4)])
    result = audit_recovery(tmp_path)["tasks"][key]
    assert result["status"] == "RESUME_ALLOWED"
    assert not any("/groups/" in ref["path"] for ref in result["reference_files"])
    assert any("/groups/" in ref["path"] for ref in result["mutable_prelaunch_files"])
    assert str(ledger.relative_to(tmp_path)) in {
        ref["path"] for ref in result["mutable_prelaunch_files"]
    }
    account(tmp_path, [(7, 4), (8, 4)])
    assert audit_recovery(tmp_path)["tasks"][key]["status"] == "BLOCKED_INCOMPLETE_GENERATION"


def test_checkpoint_highest_commit_and_mutable_latest_guard(tmp_path):
    task = dict(kind="train", model_id=TASK["model_id"])
    registered = dict(model_id=task["model_id"], seed=71003, updates=96, arm="J6")
    registration = dict(
        main_matrix=[registered], config_sha256="c" * 64, source_manifest_sha256=SOURCE
    )
    directory = tmp_path / "runs" / task["model_id"]
    save(
        directory / "RUN_MANIFEST.json",
        dict(
            registered, config_sha256="c" * 64, source_manifest_sha256=SOURCE, technical_only=False
        ),
    )
    checkpoint_dir = directory / "checkpoints"
    checkpoint_dir.mkdir()
    payload = checkpoint_dir / "step88.pt"
    payload.write_bytes(b"test fixture: actual restore is verified by the frozen GPU worker")
    commit = dict(
        step=88,
        path=payload.name,
        sha256=file_hash(payload),
        state_hash="a" * 64,
        field_hashes={
            key: "b" * 64
            for key in (
                "parameters",
                "optimizer",
                "scheduler",
                "rng",
                "run_identity",
                "cursor",
                "adaptive_state",
            )
        },
    )
    save(checkpoint_dir / "commit-88.json", commit)
    save(checkpoint_dir / "LATEST.json", commit)
    result, _, refs, prelaunch, _ = _checkpoint_evidence(tmp_path, task, registration)
    assert result["step"] == 88 and result["gpu_restore_validation_required"]
    assert not any(ref["path"].endswith(".pt") for ref in refs)
    assert any(ref["path"].endswith(".pt") for ref in prelaunch)
    save(checkpoint_dir / "LATEST.json", dict(commit, step=96))
    with pytest.raises(PermissionError, match="LATEST"):
        _checkpoint_evidence(tmp_path, task, registration)


def test_malformed_or_nonfinite_raw_json_is_fail_closed(tmp_path):
    path, _ = evaluation_group(tmp_path, pending=True)
    path.write_text('{"group_seed":7,"group_seed":8}\n')
    account(tmp_path, [(7, 4)])
    assert generation_audit(tmp_path, TASK)["status"] == "BLOCKED_INCOMPLETE_GENERATION"
    path.write_text('{"sampler_logprobs":[NaN]}\n')
    assert generation_audit(tmp_path, TASK)["status"] == "BLOCKED_INCOMPLETE_GENERATION"


def test_complete_raw_without_generated_token_reservations_blocks(tmp_path):
    evaluation_group(tmp_path)
    account(tmp_path, [(7, 4)], include_tokens=False)
    result = generation_audit(tmp_path, TASK)
    assert result["status"] == "BLOCKED_INCOMPLETE_GENERATION"
    assert (
        len([gap for gap in result["gaps"] if gap["code"] == "RAW_ROW_WITHOUT_TOKEN_ACCOUNTING"])
        == 4
    )


def test_final_row_persisted_before_its_token_accounting_blocks(tmp_path):
    evaluation_group(tmp_path, pending=True)
    ledger = account(tmp_path, [(7, 4)])
    retained = [json.loads(line) for line in ledger.read_text().splitlines()]
    lines(ledger, retained[:-1])  # Frozen runtime persists row before reserving tokens.
    result = generation_audit(tmp_path, TASK)
    assert result["gaps"] == [
        dict(
            code="RAW_ROW_WITHOUT_TOKEN_ACCOUNTING",
            group_seed=7,
            row_index=3,
            token_count=2,
            records=1,
        )
    ]


def test_extra_generated_token_reservation_blocks(tmp_path):
    evaluation_group(tmp_path)
    ledger = account(tmp_path, [(7, 4)])
    retained = [json.loads(line) for line in ledger.read_text().splitlines()]
    retained.append(dict(retained[-1], time_ns=9999))
    lines(ledger, retained)
    result = generation_audit(tmp_path, TASK)
    assert result["gaps"] == [
        dict(
            code="TOKEN_ACCOUNTING_WITHOUT_RAW_ROW",
            group_seed=7,
            row_index=3,
            token_count=2,
            records=1,
        )
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", "other"),
        ("task_id", "other"),
        ("slurm_job_id", None),
        ("pid", 0),
        ("time_ns", 0),
    ],
)
def test_accounting_record_identity_is_exact(tmp_path, field, value):
    evaluation_group(tmp_path)
    ledger = account(tmp_path, [(7, 4)])
    retained = [json.loads(line) for line in ledger.read_text().splitlines()]
    retained[-1][field] = value
    lines(ledger, retained)
    assert generation_audit(tmp_path, TASK)["status"] == "BLOCKED_INCOMPLETE_GENERATION"


def test_token_accounting_uses_frozen_row_index_metadata(tmp_path):
    evaluation_group(tmp_path)
    ledger = account(tmp_path, [(7, 4)])
    retained = [json.loads(line) for line in ledger.read_text().splitlines()]
    metadata = retained[-1]["metadata"]
    metadata["group_row_index"] = metadata.pop("row_index")
    lines(ledger, retained)
    assert generation_audit(tmp_path, TASK)["status"] == "BLOCKED_INCOMPLETE_GENERATION"
