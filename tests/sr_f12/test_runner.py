import copy
from pathlib import Path

import pytest

from sr_f1.contract import digest
from sr_f12.protocol import AMENDMENT_ID, build_config
from sr_f12.runner import (
    _format_rate,
    durable_group,
    technical_schedule,
    validate_technical_receipt,
    write_once,
)
from sr_f12.training import select_pilot_learning_rate


def receipt():
    plan = build_config()
    metrics = [
        dict(
            step=i,
            loss=0.02,
            kl=0.001,
            total_gradient_norm=0.1,
            finite=True,
            sampling_seconds=1,
            training_seconds=2,
            step_seconds=3,
            peak_allocated_bytes=100,
            parameters_changed=True,
            adam_nonzero_moments=True,
            format_failure_rate=0.01,
            sampler_train_mean_absolute_difference=0.0001,
            sampler_train_p99_absolute_difference=0.0002,
            sampler_train_max_absolute_difference=0.001,
            tis_truncated_fraction=0,
            clip_fraction=0,
            optimizer_updates=1,
        )
        for i in range(1, 9)
    ]
    fields = {k: "ab" * 32 for k in ("parameters", "optimizer", "scheduler", "cursor", "rng")}
    pilot = dict(
        learning_rate=5e-5,
        metrics=metrics,
        movement=0.03,
        decision=select_pilot_learning_rate(metrics, 0.03),
        restoration=dict(
            initial_step=4,
            same_process=False,
            forced_termination_verified=True,
            restored_field_hashes=fields,
            checkpoint_field_hashes=fields.copy(),
        ),
    )
    return dict(
        status="PASS",
        version=AMENDMENT_ID,
        config_sha256=digest(plan),
        selected_learning_rate=5e-5,
        selected_microbatch=4,
        gradient_checkpointing=False,
        preflight=dict(
            zero_lora=dict(answer_count=32, bitwise_equal=True, maximum_logprob_difference=0),
            batch_single=dict(
                answer_count=16, mean_absolute_difference=0.001, maximum_absolute_difference=0.02
            ),
            sampler_training=dict(
                sampler_train_mean_absolute_difference=0.001, tis_truncated_fraction=0
            ),
            cuda_count=1,
            lora_modules=dict(target_modules=["language.layers.0.q"]),
            microbatch_selection=dict(
                microbatch_size=4,
                gradient_checkpointing=False,
                attempts=[dict(status="PASS", evidence={})],
            ),
            evaluation_engine=dict(
                count=8, assistant_prefills=["{"] * 8, cross_question_batching=False
            ),
        ),
        pilots=[pilot],
    )


def test_complete_measured_gate():
    assert validate_technical_receipt(receipt(), build_config())


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="RUNNING"),
        lambda r: r["preflight"]["zero_lora"].update(maximum_logprob_difference=1e-8),
        lambda r: r["preflight"]["batch_single"].update(mean_absolute_difference=0.003),
        lambda r: r["preflight"]["batch_single"].update(maximum_absolute_difference=0.051),
        lambda r: r["preflight"].update(cuda_count=4),
        lambda r: r["preflight"]["evaluation_engine"].update(assistant_prefills=[""] * 8),
        lambda r: r["pilots"][0]["restoration"].update(same_process=True),
        lambda r: r["pilots"][0]["restoration"]["restored_field_hashes"].update(rng="wrong"),
        lambda r: r["pilots"][0]["metrics"].pop(),
        lambda r: r["pilots"][0]["metrics"][0].update(training_seconds=None),
        lambda r: r["pilots"][0]["metrics"][0].update(tis_truncated_fraction=0.02),
        lambda r: r["pilots"][0]["metrics"][0].update(clip_fraction=0.01),
        lambda r: r.update(selected_learning_rate=1e-4),
        lambda r: r["pilots"][0].update(movement=0.001),
        lambda r: r.update(selected_microbatch=2),
        lambda r: r["preflight"]["microbatch_selection"]["attempts"][-1].update(status="CUDA_OOM"),
    ],
)
def test_fail_closed_on_incomplete_or_inconsistent_gate(change):
    value = receipt()
    change(value)
    with pytest.raises((PermissionError, ValueError, RuntimeError)):
        validate_technical_receipt(value, build_config())


def test_one_adjustment_only_and_actual_second_pilot():
    value = receipt()
    original = value["pilots"][0]
    original["movement"] = 0.001
    original["decision"] = select_pilot_learning_rate(original["metrics"], 0.001)
    second = copy.deepcopy(original)
    second["learning_rate"] = 1e-4
    second["decision"] = select_pilot_learning_rate(
        second["metrics"], 0.001, attempt=1, learning_rate=1e-4
    )
    value["pilots"].append(second)
    value["selected_learning_rate"] = 1e-4
    plan = build_config(learning_rate=1e-4)
    value["config_sha256"] = digest(plan)
    assert validate_technical_receipt(value, plan)
    value["pilots"].append(copy.deepcopy(second))
    with pytest.raises(PermissionError):
        validate_technical_receipt(value, plan)


def test_technical_schedule_engine_only_complete_balanced(monkeypatch, tmp_path):
    tasks = {
        f"e-{family}-{chart}-{i}": dict(pool="ENGINE", family=family, chart=chart)
        for family in range(4)
        for chart in range(2)
        for i in range(16)
    }
    tasks["test-secret"] = dict(pool="TEST_ID", family=0, chart=0)
    monkeypatch.setattr("sr_f12.runner.load_tasks", lambda _: tasks)
    rows = technical_schedule(tmp_path)
    assert len(rows) == 128 and len({r["qid"] for r in rows}) == 128
    assert not any(r["qid"] == "test-secret" for r in rows)
    assert all(len([r for r in rows if r["step"] == i]) == 16 for i in range(1, 9))
    assert technical_schedule(tmp_path) == rows


def test_group_is_atomically_reused_without_resampling(monkeypatch, tmp_path):
    class Runtime:
        calls = 0

        def generate_group(self, row, root, seeds):
            self.calls += 1
            return [
                dict(
                    tokens=[7],
                    sampler_logprobs=[-0.1],
                    raw_text="{}",
                    group_seed=seeds[0],
                    group_row_index=i,
                )
                for i in range(8)
            ]

    runtime = Runtime()
    monkeypatch.setattr(
        "sr_f12.runner.scored", lambda r, t: {"L_json": 1, "L_evidence": 1, "L_answer": 1}
    )
    slot = dict(step=1, slot=0, qid="engine-0", rollout_seeds=list(range(8)))
    first = durable_group(runtime, tmp_path, tmp_path, "pilot", slot, {"text": "x"}, {}, "abc")
    assert (
        durable_group(runtime, tmp_path, tmp_path, "pilot", slot, {"text": "x"}, {}, "abc") == first
    )
    assert runtime.calls == 1
    with pytest.raises(PermissionError):
        durable_group(runtime, tmp_path, tmp_path, "pilot", slot, {"text": "x"}, {}, "changed")


def test_strict_format_failure_is_not_answer_failure():
    assert _format_rate([[dict(score=dict(L_json=1, L_answer=1, L_evidence=1, A=0, J=0))]]) == 0
    assert _format_rate([[dict(score=dict(L_json=1, L_answer=1, L_evidence=0, A=1, J=0))]]) == 1


def test_write_once_does_not_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    write_once(path, {"x": 1})
    write_once(path, {"x": 1})
    with pytest.raises(PermissionError):
        write_once(path, {"x": 2})
    assert Path(path).read_text().strip() == '{"x": 1}'


def test_resume_committed_step96_completes_missing_adapter_and_fit(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import torch

    from mm_core.training import _cpu_tree, capture_rng, trainable_state
    from sr_f12.runner import execute_path

    plan = build_config()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    runtime = SimpleNamespace(model=model)
    run = dict(model_id="SRF1_2_A_s71001", seed=71001, arm="A", updates=96)
    directory = tmp_path / "runs" / run["model_id"]
    write_once(directory / "checkpoints/LATEST.json", dict(step=96))
    metrics = dict(step=96, loss=0)
    write_once(directory / "steps/96.json", metrics)

    def load(*args, **kwargs):
        return dict(
            committed_logical_step=96,
            parameters=trainable_state(model),
            optimizer=_cpu_tree(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),
            rng=capture_rng(),
            cursor=dict(next_logical_step=97),
            diagnostics_hash=digest(metrics),
        )

    monkeypatch.setattr(
        "sr_f12.runner.load_real_runtime", lambda *a, **k: (runtime, optimizer, scheduler, {})
    )
    monkeypatch.setattr("sr_f12.runner.enable_checkpointing", lambda *a: None)
    monkeypatch.setattr("sr_f12.runner.load_schedule", lambda *a: [])
    monkeypatch.setattr("sr_f12.runner.load_inputs", lambda *a: {})
    monkeypatch.setattr("sr_f12.runner.load_tasks", lambda *a: {})
    monkeypatch.setattr("sr_f12.training.load_checkpoint", load)
    calls = []
    monkeypatch.setattr("sr_f12.runner.publish_adapter", lambda *a: calls.append(("adapter", a[2])))
    monkeypatch.setattr("sr_f12.runner.run_train_fit", lambda *a: calls.append(("fit", a[4])))
    result = execute_path(plan, tmp_path, run)
    assert result["status"] == "COMPLETE"
    assert calls == [("adapter", 96), ("fit", 96)]


def test_forced_checkpoint_requires_actual_sigkill_and_all_identity_receipts(tmp_path):
    import subprocess
    import sys

    from sr_f1.contract import file_hash
    from sr_f12.runner import verify_forced_checkpoint

    result = subprocess.run(
        [sys.executable, "-c", "import os,signal; os.kill(os.getpid(),signal.SIGKILL)"], check=False
    )
    directory = tmp_path / "SRF1_2_TECHNICAL_R0"
    checkpoint = directory / "checkpoints/state.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"durable checkpoint")
    commit = dict(step=4, path="state.pt", sha256=file_hash(checkpoint))
    write_once(directory / "checkpoints/commit-04.json", commit)
    write_once(directory / "checkpoints/LATEST.json", commit)
    write_once(
        directory / "KILL_AFTER_STEP4.json",
        dict(step=4, pid=123, signal="SIGKILL", checkpoint=commit),
    )
    write_once(directory / "PROCESS-0.json", dict(pid=123, initial_step=0))
    write_once(
        directory / "SEGMENT-00-04.json",
        dict(
            status="CHECKPOINTED",
            technical_only=True,
            model_id=directory.name,
            step=4,
            checkpoint=commit,
        ),
    )
    assert verify_forced_checkpoint(directory, result.returncode)
    assert not verify_forced_checkpoint(directory, 0)
    assert not verify_forced_checkpoint(directory, 1)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(PermissionError):
        verify_forced_checkpoint(directory, result.returncode)


def test_actual_checkpoint_lifecycle_and_pilot_movement(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import torch

    from sr_f12.runner import execute_path, read_json

    schedule = [
        dict(step=step, slot=slot, qid=f"engine-{slot}", rollout_seeds=list(range(8)))
        for step in range(1, 9)
        for slot in range(16)
    ]
    questions = {f"engine-{i}": dict(qid=f"engine-{i}", text="safe") for i in range(16)}

    def loader(*args, **kwargs):
        torch.manual_seed(11)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        def generate(row, root, seeds):
            return [
                dict(
                    tokens=[7],
                    sampler_logprobs=[-0.1],
                    raw_text="{}",
                    group_seed=seeds[0],
                    group_row_index=i,
                )
                for i in range(8)
            ]

        def forward(records, **kwargs):
            return [model.weight.detach().reshape(-1).clone() for r in records]

        return (
            SimpleNamespace(model=model, generate_group=generate, batch_sequence_forward=forward),
            optimizer,
            scheduler,
            {},
        )

    def update(runtime, optimizer, scheduler, records, coefficients, **kwargs):
        old = [runtime.model.weight.detach().reshape(-1).tolist() for _ in records]
        optimizer.zero_grad()
        runtime.model.weight.sum().backward()
        optimizer.step()
        scheduler.step()
        result = dict(loss=0.1, kl=0.001, total_gradient_norm=1, finite=True)
        if kwargs["save_train_logprobs"]:
            result["train_logprobs"] = old
        return result

    monkeypatch.setattr("sr_f12.runner.load_real_runtime", loader)
    monkeypatch.setattr("sr_f12.runner.enable_checkpointing", lambda *a: None)
    monkeypatch.setattr("sr_f12.runner.technical_schedule", lambda *a: schedule)
    monkeypatch.setattr("sr_f12.runner.load_inputs", lambda *a: questions)
    monkeypatch.setattr("sr_f12.runner.load_tasks", lambda *a: {k: {} for k in questions})
    monkeypatch.setattr("sr_f12.runner.scored", lambda *a: dict(L_json=1, L_evidence=1, L_answer=1))
    monkeypatch.setattr("sr_f12.runner.reward_advantages", lambda *a: (np.ones((16, 8)), {}))
    monkeypatch.setattr("sr_f12.training.update", update)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    run = dict(model_id="SRF1_2_TECHNICAL_R0", seed=2026101202, arm="GATE", updates=8)
    first = execute_path(build_config(), tmp_path, run, technical=True, stop_step=4)
    second = execute_path(build_config(), tmp_path, run, technical=True, stop_step=8)
    assert first["status"] == "CHECKPOINTED" and first["step"] == 4
    assert second["status"] == "COMPLETE" and second["step"] == 8
    path = tmp_path / "technical" / run["model_id"]
    assert read_json(path / "MOVEMENT.json")["mean_absolute_logprob_movement"] > 0
    assert len(list((path / "rollouts").glob("*.json"))) == 128
    assert len(list((path / "checkpoints").glob("commit-*.json"))) == 9


def test_worker_checks_real_slurm_teacher_qos_and_single_gpu(monkeypatch):
    from types import SimpleNamespace

    from sr_f12.protocol import TEACHER_QOS
    from sr_f12.runner import verify_worker_allocation

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    state = {"qos": TEACHER_QOS, "gpus": 1}

    def query(*args, **kwargs):
        return SimpleNamespace(
            stdout=f"JobId=123 QOS={state['qos']} JobState=RUNNING "
            f"AllocTRES=cpu=4,gres/gpu={state['gpus']},gres/gpu:pro6000={state['gpus']} "
            "NodeList=gpu-1"
        )

    monkeypatch.setattr("sr_f12.runner.subprocess.run", query)
    assert verify_worker_allocation()["qos"] == TEACHER_QOS
    state["qos"] = "other"
    with pytest.raises(PermissionError):
        verify_worker_allocation()
    state["qos"] = TEACHER_QOS
    state["gpus"] = 4
    with pytest.raises(PermissionError):
        verify_worker_allocation()
