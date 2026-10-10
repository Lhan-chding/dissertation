import json
from unittest.mock import patch

import pytest
import torch

from mm_core.training import state_hash
from sr_f1.evaluation import sha_file
from sr_f12.endpoint import (
    bounded_file,
    chartqa_panel,
    read_completed_panel,
    verify_main_completion,
)
from sr_f12.evaluation import durable_json
from sr_f12.protocol import AMENDMENT_ID, BASELINE, build_config, object_hash, scientific_matrix
from sr_f12.training import CHECKPOINT_FIELDS, PLAN_ID


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed_matrix(root):
    plan = build_config()
    technical, predictions = (
        {"status": "TEST_TECHNICAL_FIXTURE"},
        {"status": "PREREGISTERED_BEFORE_SCIENCE"},
    )
    freeze = dict(
        status="FROZEN_BEFORE_SCIENCE",
        amendment_id=AMENDMENT_ID,
        main_matrix=scientific_matrix(),
        config_sha256=object_hash(plan),
        technical_check_sha256=object_hash(technical),
        predictions_sha256=object_hash(predictions),
    )
    for path, value in [
        ("SCIENCE_FREEZE.json", freeze),
        ("config/SR_F1_2_FROZEN.json", plan),
        ("TECHNICAL_CHECK_SR_F1_2.json", technical),
        ("SEMANTIC_PREDICTIONS.json", predictions),
    ]:
        write(root / path, value)
    for run in scientific_matrix():
        directory = root / "runs" / run["model_id"]
        manifest = dict(run, config_sha256=object_hash(plan), technical_only=False)
        write(directory / "RUN_MANIFEST.json", manifest)
        state = dict.fromkeys(CHECKPOINT_FIELDS)
        state.update(
            plan_id=PLAN_ID,
            run_identity=manifest,
            committed_logical_step=96,
            parameters={"test.lora_B.weight": torch.ones(2)},
            optimizer={"state": {0: {"step": torch.tensor(96.0), "exp_avg": torch.ones(2)}}},
            cursor=dict(next_logical_step=97, next_slot=0, next_sample_index=0),
            training_recipe={"learning_rate": plan["training"]["lr"]},
        )
        checkpoint = directory / "checkpoints/step-96.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save(state, checkpoint)
        commit = dict(
            step=96,
            path=checkpoint.name,
            sha256=sha_file(checkpoint),
            state_hash=state_hash(state),
            field_hashes={k: state_hash(v) for k, v in state.items()},
        )
        write(directory / "checkpoints/LATEST.json", commit)
        write(directory / "checkpoints/commit-96.json", commit)
        write(
            directory / "COMPLETE.json",
            dict(
                status="COMPLETE",
                model_id=run["model_id"],
                step=96,
                technical_only=False,
                checkpoint=commit,
            ),
        )
        adapter = directory / "adapters/step-96"
        adapter.mkdir(parents=True)
        (adapter / "adapter.safetensors").write_bytes(b"fixture weights")
        write(
            adapter / "IDENTITY.json",
            dict(
                step=96,
                run_identity_sha256=object_hash(manifest),
                policy_hash=state_hash(state["parameters"]),
                files={"adapter.safetensors": sha_file(adapter / "adapter.safetensors")},
            ),
        )
        for step in (32, 64, 96):
            panel = directory / "train_fit" / f"step-{step:02d}"
            records = []
            for group in range(32):
                rows = [{"fixture": group, "draw": draw} for draw in range(4)]
                write(panel / f"{group:02d}.json", dict(records=rows, sha256=object_hash(rows)))
                records.extend(rows)
            write(
                panel / "COMPLETE.json",
                dict(status="COMPLETE", count=128, records_sha256=object_hash(records)),
            )
    return root


def audit(root):
    with patch("sr_f12.runner.validate_technical_receipt"):
        return verify_main_completion(root)


def test_twelve_main_gate_checks_actual_complete_state_and_adapter(tmp_path):
    root = completed_matrix(tmp_path)
    receipt = audit(root)
    assert receipt["status"] == "ALL_TWELVE_MAIN_PATHS_VERIFIED"
    assert len(receipt["paths"]) == 12
    target = (
        root / "runs" / scientific_matrix()[-1]["model_id"] / "adapters/step-96/adapter.safetensors"
    )
    target.write_bytes(b"changed weights")
    with pytest.raises(PermissionError, match="adapter file identity"):
        audit(root)


def test_complete_marker_alone_does_not_release_test(tmp_path):
    root = completed_matrix(tmp_path)
    target = root / "runs" / scientific_matrix()[0]["model_id"] / "checkpoints/step-96.pt"
    target.unlink()
    with pytest.raises(FileNotFoundError):
        audit(root)


def test_missing_intermediate_fit_or_changed_predictions_blocks_release(tmp_path):
    root = completed_matrix(tmp_path)
    target = root / "runs" / scientific_matrix()[0]["model_id"] / "train_fit/step-32/COMPLETE.json"
    receipt = json.loads(target.read_text())
    receipt["count"] = 124
    write(target, receipt)
    with pytest.raises(PermissionError, match="training-fit"):
        audit(root)
    write(root / "SEMANTIC_PREDICTIONS.json", {"changed": True})
    with pytest.raises(PermissionError, match="predictions"):
        audit(root)


def test_final_checkpoint_hash_is_checked_before_deserialization(tmp_path):
    root = completed_matrix(tmp_path)
    target = root / "runs" / scientific_matrix()[0]["model_id"] / "checkpoints/step-96.pt"
    target.write_bytes(b"corrupt serialized state")
    with patch("torch.load") as loader, pytest.raises(PermissionError, match="byte hash"):
        audit(root)
    loader.assert_not_called()


def test_completed_panel_detects_duplicate_or_mutated_groups(tmp_path):
    records = [{"slot_id": "one"}, {"slot_id": "two"}]
    value = dict(records=records, records_sha256=object_hash(records))
    durable_json(tmp_path / "groups/group.json", value)
    plan = dict(assigned_groups=["group"], expected_records=2)
    done = dict(
        status="COMPLETE",
        plan_sha256=object_hash(plan),
        generated_records=2,
        groups=[
            dict(group_id="group", raw_sha256=sha_file(tmp_path / "groups/group.json"), records=2)
        ],
    )
    durable_json(tmp_path / "PLAN.json", plan)
    durable_json(tmp_path / "COMPLETE.json", done)
    assert read_completed_panel(tmp_path) == records
    value["records"][0]["slot_id"] = "changed"
    write(tmp_path / "groups/group.json", value)
    with pytest.raises(PermissionError, match="identity"):
        read_completed_panel(tmp_path)


def test_chartqa_is_complete_greedy128_and_image_verified(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture image")
    records = [
        dict(
            qid=f"q{i}",
            image_file="image.png",
            image_sha256=sha_file(image),
            question=f"question {i}",
            image_id="image",
            subset="human",
            answer="2",
        )
        for i in range(2500)
    ]
    with patch("sr_f12.endpoint.load_verified_manifest", return_value=records):
        slots, inputs, tasks = chartqa_panel(tmp_path, BASELINE)
    assert len(slots) == len(inputs) == len(tasks) == 2500
    assert all(
        r["generation"]["max_new_tokens"] == 128
        and not r["generation"]["do_sample"]
        and r["samples_per_prompt"] == 1
        for r in slots
    )
    image.write_bytes(b"changed image")
    with (
        patch("sr_f12.endpoint.load_verified_manifest", return_value=records),
        pytest.raises(PermissionError, match="image hash"),
    ):
        chartqa_panel(tmp_path, BASELINE)


def test_bounded_artifact_rejects_escape(tmp_path):
    with pytest.raises(PermissionError, match="contained"):
        bounded_file(tmp_path, "../outside")


def test_chartqa_release_preserves_official_metric_dictionary(tmp_path):
    from sr_f12.endpoint import release_scores

    model = BASELINE
    authorization = dict(status="fixture-authorized")
    root = tmp_path / "sealed/evaluation" / model
    row = dict(
        slot_id="chartqa-one",
        model_id=model,
        step=0,
        pool="CHARTQA",
        protocol="plain_answer",
        view=None,
        qid="one",
        root_id="image",
        family="CHARTQA",
        chart="external",
        variant="human",
        draw=0,
        raw_text="100",
    )
    write(root / "chartqa/COMPLETE.json", {"fixture": "completion"})
    write(
        root / "COMPLETE.json",
        dict(
            status="COMPLETE",
            main_completion_sha256=object_hash(authorization),
            panels=[
                dict(
                    panel="chartqa",
                    records=1,
                    completion_sha256=sha_file(root / "chartqa/COMPLETE.json"),
                )
            ],
        ),
    )
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl").write_text("")
    with (
        patch("sr_f12.endpoint.verify_endpoint_authorization", return_value=authorization),
        patch(
            "sr_f12.endpoint.load_verified_manifest", return_value=[dict(qid="one", answer="104")]
        ),
        patch("sr_f12.endpoint.read_completed_panel", return_value=[row]),
    ):
        receipt = release_scores(tmp_path, model)
    assert receipt["records"][0]["score"] == dict(relaxed_accuracy=1, exact_match=0)


class LiveMainSlurm:
    def __init__(self):
        self.jobs, self.accounting, self.queries = {}, {}, []

    def queue(self, user):
        self.queries.append(("queue", user))
        return dict(self.jobs)

    def terminal(self, job):
        self.queries.append(("terminal", job))
        return self.accounting.get(job)


def scheduler_gate_fixture(root):
    """Isolate scheduler authorization from separately tested deep checkpoint audit."""
    from sr_f12.protocol import TEACHER_QOS

    backend = LiveMainSlurm()
    primary = dict(root=str(root.resolve()), user="unitowner", version=AMENDMENT_ID)
    write(root / "orchestration/CONFIG.json", primary)
    write(root / "endpoint_orchestration/CONFIG.json", primary)
    write(root / "SCIENCE_FREEZE.json", dict(source_commit="b" * 40))
    result = dict(status="ALL_TWELVE_MAIN_PATHS_VERIFIED", paths=scientific_matrix())
    terminals, states = [], {}
    for index, run in enumerate(scientific_matrix()):
        model, job = run["model_id"], str(2000 + index)
        directory = root / "orchestration/tasks" / model
        directory.mkdir(parents=True)
        script = directory / "run.sbatch"
        script.write_text("#!/bin/sh\nexit 0\n")
        intent = dict(
            task_id=model,
            gpu_count=1,
            job_name="srf12-" + model,
            script=str(script),
            script_sha256=sha_file(script),
        )
        state = dict(task_id=model, status="COMPLETE", job_id=job)
        terminal = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user="unitowner",
            account="rose",
            qos=TEACHER_QOS,
            name=intent["job_name"],
        )
        done = dict(
            status="COMPLETE", model_id=model, step=96, checkpoint={"sha256": "checkpoint-" + model}
        )
        verified = dict(
            task_id=model,
            receipt_sha256=object_hash(done),
            checkpoint_sha256=done["checkpoint"]["sha256"],
        )
        write(root / "runs" / model / "COMPLETE.json", done)
        for name, value in [
            ("INTENT.json", intent),
            ("STATE.json", state),
            ("TERMINAL.json", terminal),
            ("VERIFIED_COMPLETE.json", verified),
        ]:
            write(directory / name, value)
        terminals.append(
            dict(
                model_id=model,
                terminal=terminal,
                completion_receipt_sha256=sha_file(directory / "VERIFIED_COMPLETE.json"),
            )
        )
        states[model] = state
        backend.accounting[job] = terminal
    write(root / "orchestration/STATE.json", dict(tasks=states))
    acceptance = dict(
        status="ALL_TWELVE_TERMINAL_AND_STATE_VERIFIED",
        main_completion=result,
        main_completion_sha256=object_hash(result),
        terminals=terminals,
        training_source_commit="b" * 40,
        registration_source_commit="a" * 40,
    )
    write(root / "endpoint_orchestration/MAIN_ACCEPTANCE.json", acceptance)
    return backend, result


def scheduler_audit(root, backend, result):
    from sr_f12.endpoint import verify_endpoint_authorization

    with (
        patch("getpass.getuser", return_value="unitowner"),
        patch(
            "sr_f12.source_revision.resolve_source_revision",
            return_value={"source_commit": "b" * 40, "registration_source_commit": "a" * 40},
        ),
        patch("sr_f12.endpoint.verify_main_completion", return_value=result),
    ):
        return verify_endpoint_authorization(root, slurm=backend)


def test_worker_gate_rechecks_all_live_terminals_and_accepted_binding(tmp_path):
    backend, result = scheduler_gate_fixture(tmp_path)
    assert scheduler_audit(tmp_path, backend, result) == result
    assert backend.queries == [("queue", "unitowner")] + [
        ("terminal", str(2000 + i)) for i in range(12)
    ]
    changed = dict(result, changed=True)
    with pytest.raises(PermissionError, match="complete training state"):
        scheduler_audit(tmp_path, backend, changed)


@pytest.mark.parametrize(
    "mutation",
    [
        "in_queue",
        "unknown",
        "failed",
        "nonzero_exit",
        "wrong_user",
        "wrong_qos",
        "wrong_name",
        "wrong_account",
        "recorded_terminal",
        "verified_receipt",
        "script",
        "registry_job",
    ],
)
def test_worker_rejects_stale_acceptance_or_nonterminal_main(tmp_path, mutation):
    backend, result = scheduler_gate_fixture(tmp_path)
    model = scientific_matrix()[0]["model_id"]
    directory = tmp_path / "orchestration/tasks" / model
    if mutation == "in_queue":
        backend.jobs["2000"] = {"JobState": "COMPLETING"}
    elif mutation == "unknown":
        backend.accounting["2000"] = None
    elif mutation in (
        "failed",
        "nonzero_exit",
        "wrong_user",
        "wrong_qos",
        "wrong_name",
        "wrong_account",
    ):
        key, value = {
            "failed": ("state", "FAILED"),
            "nonzero_exit": ("exit_code", "1:0"),
            "wrong_user": ("user", "other"),
            "wrong_qos": ("qos", "other"),
            "wrong_name": ("name", "unrelated"),
            "wrong_account": ("account", "other"),
        }[mutation]
        backend.accounting["2000"] = dict(backend.accounting["2000"], **{key: value})
    elif mutation == "recorded_terminal":
        write(directory / "TERMINAL.json", dict(backend.accounting["2000"], state="FAILED"))
    elif mutation == "verified_receipt":
        write(directory / "VERIFIED_COMPLETE.json", dict(changed=True))
    elif mutation == "script":
        (directory / "run.sbatch").write_text("different command")
    else:
        state = json.loads((directory / "STATE.json").read_text())
        state["job_id"] = "different"
        write(directory / "STATE.json", state)
    with pytest.raises(PermissionError):
        scheduler_audit(tmp_path, backend, result)


def test_direct_worker_and_release_cannot_bypass_missing_scheduler_acceptance(tmp_path):
    from sr_f12.endpoint import release_scores, run_endpoint

    with (
        patch("sr_f12.endpoint.verify_source_manifest", return_value="source"),
        patch(
            "sr_f12.endpoint.verify_main_completion", return_value={"status": "COMPLETE"}
        ) as deep,
    ):
        with pytest.raises(FileNotFoundError, match="MAIN_ACCEPTANCE"):
            run_endpoint(
                tmp_path,
                BASELINE,
                source_manifest=tmp_path / "source.json",
                code_root=tmp_path,
                model_path=tmp_path,
            )
        with pytest.raises(FileNotFoundError, match="MAIN_ACCEPTANCE"):
            release_scores(tmp_path, BASELINE)
    deep.assert_not_called()
    assert not (tmp_path / "sealed").exists() and not (tmp_path / "released").exists()


def test_main_acceptance_must_bind_unique_twelve_paths_and_training_revision(tmp_path):
    backend, result = scheduler_gate_fixture(tmp_path)
    path = tmp_path / "endpoint_orchestration/MAIN_ACCEPTANCE.json"
    value = json.loads(path.read_text())
    value["terminals"][1] = value["terminals"][0]
    write(path, value)
    with pytest.raises(PermissionError, match="unique twelve"):
        scheduler_audit(tmp_path, backend, result)
    value["training_source_commit"] = "c" * 40
    write(path, value)
    with pytest.raises(PermissionError, match="training-source"):
        scheduler_audit(tmp_path, backend, result)
