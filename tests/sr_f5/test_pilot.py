"""G3 gating, checkpoint transaction and recovery-evidence regressions."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sr_f3_ce.training import fresh_optimizer
from sr_f4.common import read, write, write_rows
from sr_f5 import pilot
from sr_f5.training import build_credit_group
from srf5_ref.generator import FAMILIES, make_root
from srf5_ref.oracle import canonical_evidence, encode, evaluate_a
from srf5_ref.schedule import build_schedule


@pytest.fixture
def root(tmp_path):
    config = read(Path(__file__).parents[2] / "configs/sr_f5/experiment.json")
    write(tmp_path / "CONFIG.json", config)
    write(tmp_path / "CPU_TESTS.json", {"status": "PASS"})
    metrics = dict(
        actual_response_count=4608,
        unique_reviewed_responses=360,
        unresolved_systematic_measurement_errors=0,
        missing_groups=[],
        failures=[],
        blocks={
            parent: dict(
                groups=192,
                known_fraction=1.0,
                minimum_family_known_fraction=1.0,
                mask_success_given_known=1.0,
                nonzero_groups=20,
                swappable_groups=10,
                changed_groups=5,
                families_with_signal=4,
                roots_with_signal=12,
                truncation_fraction=0.0,
            )
            for parent in config["parents"]
        },
    )
    write(tmp_path / "G2/METRICS.json", metrics)
    write(tmp_path / "GATES/G2.json", dict(status="PASS", actual_response_count=4608))
    write(
        tmp_path / "G2_SPEC_FROZEN.json",
        dict(config_digest=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()),
    )
    tasks = [
        task for family in FAMILIES for i in range(8) for task in make_root("pilot", family, i, 123)
    ]
    write_rows(tmp_path / "DATA/pilot/TASKS_PRIVATE.jsonl", tasks)
    write_rows(
        tmp_path / "DATA/pilot/MODEL_INPUTS.jsonl",
        [
            dict(
                qid=t["qid"],
                root_id=t["root_id"],
                protocol="structured_v3",
                model_input=dict(prompt="public question", image_file=None, image_sha256=None),
            )
            for t in tasks
        ],
    )
    for parent in config["parents"]:
        write(
            tmp_path / f"SCHEDULES/pilot-{parent}.json",
            build_schedule(tasks, config["master_seed"], parent, 0, 8),
        )
    return tmp_path


def test_freeze_is_gated_and_registers_bounded_recovery(root):
    plan = pilot.freeze(root, "abcdef12345")
    assert plan["primary_responses"] == 23040
    assert plan["recovery_responses"] == 5120
    assert plan["endpoint_probe_responses"] == 0
    assert plan["step12_review"]["total"] == 360
    assert pilot.freeze(root, "abcdef12345") == plan
    with pytest.raises(PermissionError, match="identity differs"):
        pilot.freeze(root, "newrevision")
    write(root / "GATES/G2.json", dict(status="NOT_COMPLETE", actual_response_count=4500))
    with pytest.raises(PermissionError, match="G2 PASS"):
        pilot.require_frozen(root)


def test_config_and_question_schedule_cannot_change_after_census(root):
    config = read(root / "CONFIG.json")
    config["training"]["aux_coefficient"] = 0.5
    write(root / "CONFIG.json", config)
    with pytest.raises(PermissionError, match="Configuration differs"):
        pilot.freeze(root, "abcdef12345")
    schedule = read(root / "SCHEDULES/pilot-b0.json")
    schedule[0]["groups"][0]["draw_seeds"][0] += 1
    write(root / "SCHEDULES/pilot-b0.json", schedule)
    with pytest.raises(PermissionError, match="schedule differs"):
        pilot.pilot_inputs(root, config)


def tiny_runtime():
    runtime = SimpleNamespace(model=torch.nn.Linear(2, 1), identity={})
    optimizer, scheduler = fresh_optimizer(runtime.model)
    return runtime, optimizer, scheduler


def test_checkpoint_directory_commit_keeps_step_four_and_latest(tmp_path):
    runtime, optimizer, scheduler = tiny_runtime()
    identity = dict(run_id="test")
    for step in (0, 1, 4, 5, 12):
        pilot.commit(tmp_path, runtime, optimizer, scheduler, identity, step, dict(step=step))
    assert set(pilot.committed(tmp_path, identity)) == {4, 12}
    assert read(tmp_path / "updates/001.json") == {"step": 1}
    pilot.checkpoint_header(tmp_path / "checkpoints/step-012/state.pt", identity, 12)
    with pytest.raises(PermissionError, match="Serialized checkpoint"):
        pilot.checkpoint_header(tmp_path / "checkpoints/step-012/state.pt", identity, 4)
    with pytest.raises(PermissionError, match="Never replace"):
        pilot.commit(tmp_path, runtime, optimizer, scheduler, identity, 12)


def test_failed_state_write_never_publishes_committed_step(tmp_path, monkeypatch):
    runtime, optimizer, scheduler = tiny_runtime()
    identity = dict(run_id="test")
    pilot.commit(tmp_path, runtime, optimizer, scheduler, identity, 0)

    def fail(*args, **kwargs):
        raise OSError("interrupted state write")

    monkeypatch.setattr(pilot, "save_checkpoint", fail)
    with pytest.raises(OSError):
        pilot.commit(tmp_path, runtime, optimizer, scheduler, identity, 1)
    assert set(pilot.committed(tmp_path, identity)) == {0}
    assert not (tmp_path / "updates/001.json").exists()


def test_rollout_rejects_other_arm_policy_and_changed_seed():
    assignment = dict(draw_seeds=list(range(8)))
    identity = dict(parent="b0", run_id="G3-b0-LOCAL")
    records = [
        dict(
            seed=i,
            model_input={},
            generation_status="COMPLETE",
            adapter_identity=dict(parent="b0", run_id="G3-b0-LOCAL", step=1),
        )
        for i in range(8)
    ]
    pilot.validate_rollout(records, assignment, identity, 2, {})
    records[0]["adapter_identity"]["run_id"] = "G3-b0-J"
    with pytest.raises(PermissionError, match="this arm"):
        pilot.validate_rollout(records, assignment, identity, 2, {})
    records[0]["seed"] = 99
    with pytest.raises(ValueError, match="eight registered"):
        pilot.validate_rollout(records, assignment, identity, 2, {})


def test_partial_group_is_not_silently_regenerated(tmp_path):
    identity = dict(parent="b0", run_id="test", arm="J")
    task = dict(qid="q")
    public = dict(model_input={}, protocol="structured_v3")
    assignment = dict(group_index=0, draw_seeds=list(range(8)))
    spec = dict(
        identity=identity,
        step=1,
        assignment=assignment,
        input_digest=pilot.digest(public),
        task_digest=pilot.digest(task),
        lane="primary",
    )
    directory = tmp_path / "path"
    write(directory / "rollouts/001/00.attempt.json", dict(spec=spec))
    write_rows(directory / "rollouts/001/00.raw.jsonl", [dict(seed=0)])
    with pytest.raises(RuntimeError, match="Partial pilot generation"):
        pilot.rollout(tmp_path, directory, None, task, public, assignment, identity, 1, "primary")


def _recovery_fixture(root, monkeypatch):
    monkeypatch.setattr(pilot, "require_frozen", lambda _: {})
    directory = root / "G3/RUNS/b0/J"
    runtime, optimizer, scheduler = tiny_runtime()
    identity = dict(run_id="G3-b0-J")
    for lane, pid, restored in (("primary", 1, 0), ("recovery", 2, 4)):
        target = directory / lane
        execution = dict(
            id=lane, process=dict(host="test", pid=pid, start_ticks="123"), restored_step=restored
        )
        pilot.commit(target, runtime, optimizer, scheduler, identity, 12)
        write(target / "COMPLETE.json", dict(execution=execution))
        for step in range(1 if lane == "primary" else 5, 13):
            write(
                target / "updates" / f"{step:03d}.json",
                dict(
                    step=step,
                    lane=lane,
                    training_seconds=pid,
                    wall_seconds=pid,
                    primary_answers=128 if lane == "primary" else 0,
                    execution=execution,
                    reference_logprobs=[[-0.5]],
                    actual_credit_groups=[dict(native=[1, -1])],
                ),
            )
            if step > 4:
                for group in range(16):
                    write(
                        target / "rollouts" / f"{step:03d}" / f"{group:02d}.json",
                        dict(
                            records=[
                                dict(
                                    tokens=[1, 2],
                                    full_text="answer",
                                    sampler_logprobs=[-0.5, -0.5],
                                    score={},
                                    selected_mask=[True, True],
                                )
                                for _ in range(8)
                            ]
                        ),
                    )
    return directory


def test_recovery_compares_actual_weights_rng_and_probability_updates(tmp_path, monkeypatch):
    directory = _recovery_fixture(tmp_path, monkeypatch)
    assert pilot.compare_recovery(tmp_path, "b0", "J")["status"] == "PASS"
    path = directory / "recovery/updates/005.json"
    changed = read(path)
    changed["reference_logprobs"][0][0] -= 0.01
    write(path, changed)
    result = pilot.compare_recovery(tmp_path, "b0", "J")
    assert result["status"] == "FAIL"
    assert not result["checks"]["numerical_credit_update_receipts"]
    state_path = directory / "recovery/checkpoints/step-012/state.pt"
    state = torch.load(state_path, weights_only=True)
    state["policy"]["weight"] += 0.01
    torch.save(state, state_path)
    assert not pilot.compare_recovery(tmp_path, "b0", "J")["checks"]["policy"]


def test_same_process_recovery_cannot_claim_fresh_process(tmp_path, monkeypatch):
    directory = _recovery_fixture(tmp_path, monkeypatch)
    path = directory / "recovery/COMPLETE.json"
    value = read(path)
    value["execution"]["process"] = read(directory / "primary/COMPLETE.json")["execution"][
        "process"
    ]
    write(path, value)
    result = pilot.compare_recovery(tmp_path, "b0", "J")
    assert result["status"] == "FAIL"
    assert result["checks"]["fresh_process"] is False


def test_incomplete_matrix_never_passes_g3(root):
    pilot.freeze(root, "abcdef12345")
    report = pilot.aggregate(root)
    assert report["actual_response_count"] == 0
    assert len(report["missing"]) == 15
    assert report["gate"]["status"] == "NOT_COMPLETE"


def test_checkpoint_state_must_match_marker(root):
    pilot.freeze(root, "abcdef12345")
    lane = root / "G3/RUNS/b1/LOCAL/primary"
    write(lane / "COMPLETE.json", dict(status="COMPLETE", committed_step=12, identity={}))
    with pytest.raises(PermissionError, match="bound to final"):
        pilot.aggregate(root)


def _one_path(root, monkeypatch):
    frozen = pilot.freeze(root, "abcdef12345")
    config = read(root / "CONFIG.json")
    tasks, inputs, schedules = pilot.pilot_inputs(root, config)
    parent, arm = "b1", "LOCAL"
    identity = dict(
        run_id="G3-b1-LOCAL",
        parent=parent,
        arm=arm,
        repeat=0,
        frozen_digest=pilot.digest(frozen),
        reference=config["model"]["parents"][parent],
        protocol_version=pilot.PROTOCOL_VERSION,
    )
    directory = root / "G3/RUNS" / parent / arm / "primary"
    runtime, optimizer, scheduler = tiny_runtime()
    pilot.commit(directory, runtime, optimizer, scheduler, identity, 4)
    for step, entry in enumerate(schedules[parent], start=1):
        groups, all_records = [], []
        for assignment in entry["groups"]:
            task, public = tasks[assignment["qid"]], inputs[assignment["qid"]]
            group = assignment["group_index"]
            group_id = f"{identity['run_id']}/{step:03d}/{group:02d}"
            records = []
            for draw, seed in enumerate(assignment["draw_seeds"]):
                s = "UNKNOWN" if draw < 3 else "PASS" if draw < 5 else "FAIL"
                score = dict(J=1, V=int(s != "UNKNOWN"), finish=1, value_correct=1, S=s, P="PASS")
                record = dict(
                    score=score,
                    **score,
                    qid=task["qid"],
                    root_id=task["root_id"],
                    cell=task["cell"],
                    group_id=group_id,
                    row_id=f"{group_id}/{draw}",
                    tokens=[1, 2],
                    sampler_logprobs=[-0.5, -0.5],
                    selected_mask=[s != "UNKNOWN"] * 2,
                    mask_ok=s != "UNKNOWN",
                    seed=seed,
                    generation_status="COMPLETE",
                    model_input=public["model_input"],
                    run_root=str(root),
                    protocol="structured_v3",
                    full_text=f"fixture-{draw}",
                    adapter_identity=dict(parent=parent, run_id=identity["run_id"], step=step - 1),
                )
                records.append(record)
            spec = dict(
                identity=identity,
                step=step,
                assignment=assignment,
                input_digest=pilot.digest(public),
                task_digest=pilot.digest(task),
                lane="primary",
            )
            write(
                directory / "rollouts" / f"{step:03d}" / f"{group:02d}.json",
                dict(spec=spec, records=records),
            )
            groups.append(build_credit_group(records, arm, assignment["shuffle_seed"]))
            all_records.extend(records)
        write(directory / "credit" / f"{step:03d}.json", groups)
        receipt = dict(
            run_id=identity["run_id"],
            step=step,
            parent=parent,
            arm=arm,
            lane="primary",
            actual_credit_groups=groups,
            sequence_count=128,
            optimizer_updates=1,
            status="FINITE",
            sampler_train_mean_absolute_difference=0.0,
            tis_truncated_fraction=0.0,
            sampler_train_max_absolute_difference=0.0,
            gradient_norm_before_clip=1.0,
            parameter_update_norm=0.01,
            channel_gradients=dict(
                status="MEASURED",
                row_indices=list(pilot.DIAGNOSTIC_ROWS),
                native_gradient_norm=1.0,
                auxiliary_gradient_norm=0.5,
            ),
        )
        write(directory / "updates" / f"{step:03d}.json", receipt)
        if step == 12:
            pilot.commit(directory, runtime, optimizer, scheduler, identity, 12, receipt)
    write(
        directory / "COMPLETE.json",
        dict(
            status="COMPLETE", identity=identity, lane="primary", committed_step=12, responses=1536
        ),
    )
    monkeypatch.setattr(
        pilot, "independent_review", lambda record, _: dict(status="PASS", row_id=record["row_id"])
    )
    return directory


def test_single_arm_observation_collapse_is_not_hidden_by_pooling(root, monkeypatch):
    _one_path(root, monkeypatch)
    report = pilot.aggregate(root)
    assert "b1/LOCAL:S_observation_below_90_percent" in report["errors"]
    assert report["blocks"]["b1"]["S_known_fraction_by_arm"]["LOCAL"] == 5 / 8
    assert report["unique_step12_reviews"] == 24


def test_aggregate_rejects_rollout_from_wrong_checkpoint(root, monkeypatch):
    directory = _one_path(root, monkeypatch)
    path = directory / "rollouts/001/00.json"
    value = copy.deepcopy(read(path))
    value["records"][0]["adapter_identity"]["step"] = 5
    write(path, value)
    with pytest.raises(PermissionError, match="committed pre-update"):
        pilot.aggregate(root)


def test_runner_resumes_committed_state_and_reuses_only_same_policy_batch(root, monkeypatch):
    """Tiny CPU model exercises actual checkpoint/Adam/RNG transactions, not GPU claims."""
    pilot.freeze(root, "abcdef12345")
    config = read(root / "CONFIG.json")
    tasks, _inputs, schedules = pilot.pilot_inputs(root, config)
    assignments = {tuple(a["draw_seeds"]): a for step in schedules["b0"] for a in step["groups"]}
    calls, updates, loads = [], [], []
    fail_once = [True]

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))

    class Runtime:
        def __init__(self, parent, task_id):
            torch.manual_seed(123)
            self.model = torch.nn.Linear(2, 1)
            self.processor = SimpleNamespace(tokenizer=Tokenizer())
            self.identity = dict(parent_id=parent, run_id=task_id, step=0)

        def generate_group(self, model_input, run_root, seeds, *, protocol, on_completion):
            assignment = assignments[tuple(seeds)]
            task = tasks[assignment["qid"]]
            calls.append((self.identity["step"], tuple(seeds)))
            gold = evaluate_a(task["world"], task["query"])
            text = json.dumps(
                dict(
                    evidence=canonical_evidence(task["world"], task["query"]),
                    selected=gold["selected"],
                    answer=encode(gold["answer"]),
                )
            )
            tokens = list(map(ord, text[1:]))
            for seed in seeds:
                on_completion(
                    dict(
                        seed=seed,
                        tokens=tokens,
                        sampler_logprobs=[-0.5] * len(tokens),
                        model_input=model_input,
                        full_text=text,
                        assistant_prefill="{",
                        finish_reason="balanced",
                        generation_status="COMPLETE",
                        technical_validation_errors=[],
                        truncated=False,
                        adapter_identity=dict(
                            parent=self.identity["parent_id"],
                            run_id=self.identity["run_id"],
                            step=self.identity["step"],
                        ),
                    )
                )

    def load_runtime(run_root, parent, task_id, training=False):
        assert training
        loads.append(task_id)
        runtime = Runtime(parent, task_id)
        write(run_root / "runtime_identity" / f"{task_id}.json", dict(kind="CPU_TEST_FIXTURE"))
        optimizer, scheduler = fresh_optimizer(runtime.model)
        return runtime, optimizer, scheduler

    def update_runtime(
        runtime, optimizer, scheduler, records, group_sizes, reference, credits, **kwargs
    ):
        assert len(records) == 128 and group_sizes == [8] * 16
        step = runtime.identity["step"] + 1
        assert all(row["adapter_identity"]["step"] == step - 1 for row in records)
        if step == 5 and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("CPU fixture interrupted before optimizer update")
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in runtime.model.parameters()).backward()
        optimizer.step()
        scheduler.step()
        updates.append(step)
        return dict(sequence_count=128, optimizer_updates=1)

    monkeypatch.setattr(pilot, "load", load_runtime)
    monkeypatch.setattr(pilot, "update", update_runtime)
    with pytest.raises(RuntimeError, match="CPU fixture interrupted"):
        pilot.run(root, "b0", "J")
    path = root / "G3/RUNS/b0/J/primary"
    assert read(path / "checkpoints/step-004/COMMITTED.json")["step"] == 4
    assert len(calls) == 5 * 16
    result = pilot.run(root, "b0", "J")
    assert result["committed_step"] == 12
    assert updates == list(range(1, 13))
    assert len(calls) == 12 * 16
    assert len({seeds for _, seeds in calls}) == 12 * 16
    assert read(path / "COMPLETE.json")["scientific_gate"] == "NOT_YET_AGGREGATED"
    completion = read(path / "COMPLETE.json")
    assert pilot.run(root, "b0", "J")["model_calls"] == 0
    assert len(loads) == 2
    assert read(path / "COMPLETE.json") == completion
    assert len(list((root / "accounting").glob("*"))) == 0  # mocked backend only
