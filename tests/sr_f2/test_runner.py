import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mm_core.training import state_hash, trainable_state
from sr_f1.contract import digest
from sr_f2 import protocol, runner, training


class Runtime:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(0.2)
        self.generation_calls = 0

    def generate_group(self, row, root, seeds, on_completion=None):
        self.generation_calls += 1
        rows = [
            dict(
                tokens=[3],
                sampler_logprobs=[self.model.weight.item() * (i + 1) - 2],
                features=[float(i + 1)],
                raw_text=str(i),
                group_seed=seeds[0],
                group_row_index=i,
            )
            for i in range(8)
        ]
        if on_completion:
            for completion in rows:
                on_completion(completion)
        return rows

    def batch_sequence_forward(self, records, purpose, grad):
        return [self.model.weight.reshape(()) * torch.tensor(r["features"]) - 2 for r in records]

    def batch_reference_forward(self, records, reference):
        return [reference["weight"].reshape(()) * torch.tensor(r["features"]) - 2 for r in records]


def score(record, task):
    index = int(record["raw_text"])
    return dict(
        A=int(index < 5),
        E=int(index < 6),
        P=int(index < 5),
        p_read=float(index < 5),
        J=int(index < 4),
        J6=int(index < (5 if task.get("family") == "TOPK" else 4)),
        L_json=1,
        L_answer=1,
        L_evidence=1,
    )


def monitor(runtime, root, directory, model_id, step, **kwargs):
    random.random()
    np.random.random()
    torch.rand(3)
    runtime.model.train()
    families = {f: dict(A=0.6, E=0.8, E6=0.8, p_read=0.9, P=0.5, J=0.4) for f in training.FAMILIES}
    families["THRESHOLD"]["P"] = 0.2
    if step >= 8:
        families["TOPK"]["P"] = 0.2
    return dict(
        status="COMPLETE",
        model_id=model_id,
        step=step,
        count=512,
        records_sha256="abc",
        policy_hash=state_hash(trainable_state(runtime.model)),
        families=families,
    )


@pytest.fixture
def environment(monkeypatch, tmp_path):
    tasks = {
        f"e-{family}-{chart}-{i}": dict(pool="ENGINE", family=family, chart=chart)
        for family in training.FAMILIES
        for chart in ("bar", "line")
        for i in range(16)
    }
    plan = protocol.build_config()
    monkeypatch.setattr(runner, "load_tasks", lambda _: tasks)
    monkeypatch.setattr(runner, "load_inputs", lambda _: {qid: {"qid": qid} for qid in tasks})
    monkeypatch.setattr(runner, "scored", score)
    monkeypatch.setattr(
        protocol,
        "verify_registration",
        lambda *a, **kw: dict(source_manifest_sha256="frozen-code", config_sha256=digest(plan)),
    )
    instances = []

    def factory(plan, root, *, task_id):
        runtime = Runtime()
        instances.append(runtime)
        optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=5e-5, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return runtime, optimizer, scheduler, {"target_modules": ["weight"]}

    return plan, tmp_path, tasks, factory, instances


def test_engine_schedule_balanced_and_identical_for_all_arms(environment):
    _, root, tasks, _, _ = environment
    rows = runner.technical_schedule(root)
    assert len(rows) == len(set(r["qid"] for r in rows)) == 128
    for step in range(1, 9):
        slots = [s for s in rows if s["step"] == step]
        assert [s["slot"] for s in slots] == list(range(16))
        assert all(
            sum(tasks[s["qid"]]["family"] == f for s in slots) == 4 for f in training.FAMILIES
        )
    assert runner.technical_schedule(root) == rows
    assert set(r["paired_seed"] for r in rows) == {runner.PILOT_SEED}
    assert runner.PILOT_SEED not in protocol.SEEDS


@pytest.mark.parametrize("arm", runner.ARMS)
def test_each_arm_runs_fixed_loss_and_restores_monitor_latch(environment, arm):
    plan, root, _, factory, instances = environment
    run = next(r for r in runner.technical_matrix() if r["arm"] == arm)
    first = runner.execute_path(
        plan, root, run, technical=True, stop_step=4, runtime_factory=factory, monitor_fn=monitor
    )
    assert first["status"] == "CHECKPOINTED"
    second = runner.execute_path(
        plan, root, run, technical=True, stop_step=8, runtime_factory=factory, monitor_fn=monitor
    )
    assert second["status"] == "COMPLETE"
    assert [runtime.generation_calls for runtime in instances] == [64, 64]
    directory = root / "technical" / run["model_id"]
    metrics = [runner.read_json(directory / "steps" / f"{step:02d}.json") for step in range(1, 9)]
    assert all(m["finite"] and m["optimizer_updates"] == 1 for m in metrics)
    assert all(m["microbatch_size"] == 1 and m["backward_calls"] == 128 for m in metrics)
    assert all(m["advantage_energy"] >= 0 for m in metrics)
    assert all(m["semantic_groups_used"] == (4 if arm == "ADAPT" else 16) for m in metrics)
    saved = runner.read_json(directory / "monitor_state/step-08.json")
    assert saved["adaptive_state"]["active_families"] == ["THRESHOLD", "TOPK"]
    assert saved["identity"]["checkpoint_sha256"] == second["checkpoint"]["sha256"]
    processes = [runner.read_json(path) for path in directory.glob("PROCESS-*.json")]
    assert sorted(p["initial_step"] for p in processes) == [0, 4]
    assert all(p["restored_field_hashes"] == p["checkpoint_field_hashes"] for p in processes)


def test_incomplete_monitor_at_committed_boundary_finishes_before_any_new_sampling(environment):
    plan, root, _, factory, instances = environment
    run = runner.technical_matrix()[-1]

    def fail_at_eight(*args, **kwargs):
        if args[4] == 8:
            raise RuntimeError("monitor interrupted")
        return monitor(*args, **kwargs)

    with pytest.raises(RuntimeError, match="monitor interrupted"):
        runner.execute_path(
            plan, root, run, technical=True, runtime_factory=factory, monitor_fn=fail_at_eight
        )
    directory = root / "technical" / run["model_id"]
    assert runner.read_json(directory / "checkpoints/LATEST.json")["step"] == 8
    assert not (directory / "COMPLETE.json").exists()
    result = runner.execute_path(
        plan, root, run, technical=True, runtime_factory=factory, monitor_fn=monitor
    )
    assert result["status"] == "COMPLETE"
    assert instances[-1].generation_calls == 0
    assert result["last_monitor_step"] == 8


def test_pending_baseline_monitor_blocks_first_update(environment):
    plan, root, _, factory, instances = environment

    def failing(*a, **kw):
        raise RuntimeError("no monitor")

    with pytest.raises(RuntimeError, match="no monitor"):
        runner.execute_path(
            plan,
            root,
            runner.technical_matrix()[0],
            technical=True,
            runtime_factory=factory,
            monitor_fn=failing,
        )
    assert instances[-1].generation_calls == 0


def test_rollouts_cannot_reuse_scores_from_changed_source_or_task(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "scored", score)
    runtime = Runtime()
    slot = dict(step=1, slot=0, qid="e", rollout_seeds=list(range(8)))
    args = (runtime, tmp_path, tmp_path, "SRF2_PILOT_EARLY", slot, {"qid": "e"}, {}, "policy")
    kwargs = dict(source_sha256="frozen", run_identity_sha256="run")
    first = runner.durable_group(*args, **kwargs)
    assert runner.durable_group(*args, **kwargs) == first
    assert runtime.generation_calls == 1
    with pytest.raises(PermissionError, match="identity"):
        runner.durable_group(*args, **{**kwargs, "source_sha256": "changed"})
    path = tmp_path / "rollouts/01-00.json"
    saved = json.loads(path.read_text())
    saved["records"][0]["score"]["A"] = 0
    saved["sha256"] = digest(saved["records"])
    path.write_text(json.dumps(saved))
    with pytest.raises(PermissionError, match="scores"):
        runner.durable_group(*args, **kwargs)


def test_worker_allows_multiple_gpus_in_single_teacher_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout=f"JobId=123 QOS={protocol.TEACHER_QOS} JobState=RUNNING "
            "AllocTRES=cpu=16,gres/gpu=4,gres/gpu:pro6000=4 NodeList=n1"
        ),
    )
    assert "gres/gpu=4" in runner.verify_worker_allocation()["alloc_tres"]


def test_monitor_sidecar_detects_modified_latch(environment):
    plan, root, _, factory, _ = environment
    run = runner.technical_matrix()[-1]
    runner.execute_path(
        plan, root, run, technical=True, stop_step=0, runtime_factory=factory, monitor_fn=monitor
    )
    path = root / "technical" / run["model_id"] / "monitor_state/step-00.json"
    saved = runner.read_json(path)
    saved["adaptive_state"]["active_families"] = []
    path.write_text(json.dumps(saved))
    with pytest.raises(PermissionError, match="replayed"):
        runner.execute_path(
            plan,
            root,
            run,
            technical=True,
            stop_step=4,
            runtime_factory=factory,
            monitor_fn=monitor,
        )


def test_invalid_science_stop_not_a_durable_boundary(environment):
    plan, root, _, factory, _ = environment
    with pytest.raises(PermissionError, match="boundary"):
        runner.execute_path(
            plan,
            root,
            runner.technical_matrix()[0],
            technical=True,
            stop_step=3,
            runtime_factory=factory,
            monitor_fn=monitor,
        )


@pytest.mark.parametrize("count", [0, 3, 8])
def test_failed_raw_completions_are_retained_without_implicit_retry(monkeypatch, tmp_path, count):
    monkeypatch.setattr(runner, "scored", score)

    class FailedRuntime(Runtime):
        def generate_group(self, row, root, seeds, on_completion=None):
            records = super().generate_group(row, root, seeds)
            for record in records[:count]:
                record["technical_validation_errors"] = ["NATIVE_VISUAL_ENCODER_NOT_CALLED"]
                on_completion(record)
            raise RuntimeError("synthetic generation failure")

    runtime = FailedRuntime()
    slot = dict(step=1, slot=0, qid="e", rollout_seeds=list(range(8)))
    args = (runtime, tmp_path, tmp_path, "pilot", slot, {"qid": "e"}, {}, "policy")
    kwargs = dict(source_sha256="source", run_identity_sha256="run")
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        runner.durable_group(*args, **kwargs)
    raw = tmp_path / "rollouts/01-00.pending.jsonl"
    assert len(raw.read_text().splitlines()) == count
    assert runner.read_json(tmp_path / "rollouts/01-00.failure.json")["completed_rows"] == count
    with pytest.raises(RuntimeError, match="explicit technical recovery"):
        runner.durable_group(*args, **kwargs)
    assert runtime.generation_calls == 1


def test_complete_raw_group_recovers_receipt_without_resampling(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "scored", score)
    runtime = Runtime()
    slot = dict(step=1, slot=0, qid="e", rollout_seeds=list(range(8)))
    args = (runtime, tmp_path, tmp_path, "pilot", slot, {"qid": "e"}, {}, "policy")
    kwargs = dict(source_sha256="source", run_identity_sha256="run")
    first = runner.durable_group(*args, **kwargs)
    (tmp_path / "rollouts/01-00.json").unlink()
    assert runner.durable_group(*args, **kwargs) == first
    assert runtime.generation_calls == 1


def test_scientific_step_nine_uses_latched_monitor_before_training(environment, monkeypatch):
    plan, root, _, factory, _ = environment
    run = protocol.scientific_matrix()[-1]
    schedule = runner.technical_schedule(root)
    first_eight = [dict(slot, paired_seed=run["seed"]) for slot in schedule]
    # Repeating input qids in later updates is valid; every slot keeps its frozen seeds.
    science_schedule = first_eight + [dict(slot, step=slot["step"] + 8) for slot in first_eight]
    (root / "INPUT_IDENTITY.json").write_text(
        json.dumps(dict(schedule_hashes={str(run["seed"]): digest(science_schedule)}))
    )
    monkeypatch.setattr(runner, "load_schedule", lambda _, seed: science_schedule)
    monkeypatch.setattr(runner, "validate_technical_receipt", lambda *a: True)
    monkeypatch.setattr(runner, "publish_adapter", lambda *a: None)
    (root / "TECHNICAL_CHECK_SR_F2.json").write_text("{}")
    result = runner.execute_path(
        plan, root, run, stop_step=16, runtime_factory=factory, monitor_fn=monitor
    )
    assert result["status"] == "CHECKPOINTED" and result["last_monitor_step"] == 16
    directory = root / "runs" / run["model_id"]
    assert runner.read_json(directory / "steps/08.json")["semantic_groups_used"] == 4
    assert runner.read_json(directory / "steps/09.json")["semantic_groups_used"] == 8
    assert runner.read_json(directory / "steps/16.json")["semantic_groups_used"] == 8
    assert sorted(p.name for p in (directory / "checkpoints").glob("commit-*.json")) == [
        "commit-00.json",
        "commit-08.json",
        "commit-16.json",
    ]


def technical_receipt():
    hashes = {key: "ab" * 32 for key in runner.RECOVERY_FIELDS}
    metrics = [
        dict(
            step=step,
            finite=True,
            loss=0.1,
            kl=0.001,
            total_gradient_norm=0.2,
            step_seconds=3.0,
            sampling_seconds=1.0,
            training_seconds=2.0,
            advantage_energy=12.0,
            parameter_max_absolute_change=0.00005,
            optimizer_updates=1,
            microbatch_size=1,
            clip_fraction=0,
            sampler_train_mean_absolute_difference=0.001,
            sampler_train_p99_absolute_difference=0.002,
            sampler_train_max_absolute_difference=0.003,
            tis_truncated_fraction=0,
            parameters_changed=True,
            adam_nonzero_moments=True,
        )
        for step in range(1, 9)
    ]
    return dict(
        status="PASS",
        plan_id="SR-F2",
        config_sha256=digest(protocol.build_config()),
        pilots=[
            dict(
                arm=arm,
                seed=runner.PILOT_SEED,
                learning_rate=5e-5,
                microbatch_size=1,
                metrics=metrics,
                restoration=dict(
                    initial_step=4,
                    same_process=False,
                    restored_field_hashes=hashes,
                    checkpoint_field_hashes=hashes.copy(),
                ),
                monitor_steps=[0, 8],
                monitor_counts=[512, 512],
                monitor_seconds=3.0,
                completion=dict(
                    status="COMPLETE", step=8, technical_only=True, last_monitor_step=8
                ),
            )
            for arm in runner.ARMS
        ],
    )


def test_technical_qualification_accepts_all_four_measured_paths():
    assert runner.validate_technical_receipt(technical_receipt(), protocol.build_config())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["pilots"].pop(),
        lambda r: r["pilots"][0].update(learning_rate=1e-4),
        lambda r: r["pilots"][0]["restoration"].update(same_process=True),
        lambda r: r["pilots"][0].update(monitor_steps=[0]),
        lambda r: r["pilots"][0]["metrics"][0].update(loss=float("nan")),
        lambda r: r["pilots"][0]["metrics"][0].update(
            sampler_train_mean_absolute_difference=float("nan")
        ),
        lambda r: r["pilots"][0]["completion"].update(status="CHECKPOINTED"),
    ],
)
def test_technical_qualification_fails_on_partial_or_nonfinite_evidence(mutation):
    receipt = technical_receipt()
    mutation(receipt)
    with pytest.raises(PermissionError):
        runner.validate_technical_receipt(receipt, protocol.build_config())
